"""Frozen inverse comparison on fresh sensor records; no inverse-based calibration."""
import argparse
import hashlib
import inspect
import json
import time
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import calibrate_m as cm
import check_cross_variance as cv
import invert_waves as iw
import variance_convergence as vc

GROUPS={"baseline":230000,"long_period":231000,"weak_waves":232000}
METHODS=["existing_2.5","calibrated_equal","coherence_corrected"]


def candidate_variances(spectral,pairs,m):
    if pairs!=iw.stack_pairs(spectral.shape[-1]):
        raise ValueError("unexpected spectral stacking")
    return cv.corrected_variances(*cv.variance_terms(spectral),m)


def operator_responses(cache,f,headings):
    """Reuse Will's batch RAO builder, loading the checked 40-frequency BEM cache."""
    import sensors
    import batch_invert as bi
    def cached_hydrodynamics(sub):
        with np.load(cache) as z:
            if str(z["fingerprint"])!=vc.model_fingerprint():
                raise ValueError("hydrodynamics source mismatch")
            for key in ["A","B","A_inf","F_exc","M_inv","A_sys"]:
                setattr(sub,key,z[key])
    with tempfile.TemporaryDirectory() as folder,patch.object(sensors.ControlledSubmarine,"solve_hydrodynamics",cached_hydrodynamics):
        rao=bi.rao_for_depth(15.,40,36,(.1,2.),SimpleNamespace(cache_dir=folder))
    keep=np.arange(0,36,36//headings)
    beta=rao["theta"][keep]
    responses=np.empty((len(f),headings,9),complex)
    positions=dict(IMU=sensors.R_IMU,**sensors.PRESSURE_SENSORS)
    for fi,frequency in enumerate(f):
        w=2*np.pi*frequency
        h=iw.interp_complex(w,rao["omega"],rao["rao"][:,keep])
        for ci,channel in enumerate(iw.CHANNELS):
            responses[fi,:,ci]=iw.channel_operator(channel,h,w,beta,positions,(1025.,9.81,15.))
    return beta,responses


def truth_spectrum(case,f,beta):
    """Scoring only: ensemble energy in Welch cells, not a realised surface trace."""
    from sea import Sea
    from variance_uniform import uniform_grid
    sea=Sea(n_freq=6400,n_dir=36,**cm.CASES[case],gamma=3.3,s=10,direction=np.pi/2,seed=0)
    uniform_grid(sea)
    df=float(f[1]-f[0])
    edges=np.r_[f-df/2,f[-1]+df/2]
    energy=sea.S_f*sea.delta_f
    frequency_energy=np.histogram(sea.f,bins=edges,weights=energy)[0]
    distance=abs(np.angle(np.exp(1j*(sea.theta[:,None]-beta[None,:]))))
    share=np.bincount(np.argmin(distance,axis=1),weights=sea.D*sea.delta_theta,minlength=len(beta))
    x=frequency_energy[:,None]/df*share[None,:]
    return x,float(4*np.sqrt(energy.sum()))


def recovery_metrics(x,cov,truth,f,beta,full_hs):
    df=float(f[1]-f[0])
    sf=x.sum(1); target_sf=truth.sum(1)
    m0=float(sf.sum()*df); true_m0=float(target_sf.sum()*df)
    var_sf=np.einsum("fij->f",cov)
    var_x=np.diagonal(cov,axis1=1,axis2=2)
    if (not np.isfinite(x).all() or not np.isfinite(cov).all()
            or np.any(var_sf<=0) or np.any(var_x<=0)):
        raise ValueError("nonfinite inverse or nonpositive posterior variance")
    sd_m0=float(df*np.sqrt(var_sf.sum()))
    hs=4*np.sqrt(max(m0,0)); true_hs=4*np.sqrt(true_m0)
    power=x.sum(0)*df; target_power=truth.sum(0)*df
    if m0<=0:
        raise ValueError("inverse has nonpositive total wave energy")
    direction=np.angle(np.sum(power*np.exp(1j*beta)))
    true_direction=np.angle(np.sum(target_power*np.exp(1j*beta)))
    direction_error=float(np.degrees(np.angle(np.exp(1j*(direction-true_direction)))))
    active=truth>=.01*truth.max(axis=1,keepdims=True)
    active_direction=target_power>=.01*target_power.max()
    z_x=(x-truth)/np.sqrt(var_x)
    z_sf=(sf-target_sf)/np.sqrt(var_sf)
    sd_direction=df*np.sqrt(var_x.sum(0))
    z_direction=(power-target_power)/sd_direction
    z_m0=(m0-true_m0)/sd_m0
    out=dict(Hs=hs,Hs_true_band=true_hs,Hs_true_full=full_hs,m0=m0,m0_true=true_m0,
             hs_err_pct=100*(hs/true_hs-1),hs_abs_pct=100*abs(hs/true_hs-1),
             hs_full_err_pct=100*(hs/full_hs-1),
             spectrum_nrmse=float(np.linalg.norm(sf-target_sf)/np.linalg.norm(target_sf)),
             directional_nrmse=float(np.linalg.norm(x-truth)/np.linalg.norm(truth)),
             direction_err_deg=direction_error,direction_abs_deg=abs(direction_error),
             direction_l1=float(.5*np.sum(abs(power/m0-target_power/true_m0))),
             negative_fraction=float(np.maximum(-x,0).sum()/abs(x).sum()),
             sd_Hs=float(2*sd_m0/np.sqrt(m0)),sd_m0=sd_m0,z_m0=float(z_m0),
             m0_cover68=float(abs(z_m0)<=1),m0_cover95=float(abs(z_m0)<=1.959963984540054),
             spectrum_cover95=float(np.mean(abs(z_sf)<=1.959963984540054)),
             active_x_cover95=float(np.mean(abs(z_x[active])<=1.959963984540054)),
             active_direction_cover95=float(np.mean(abs(z_direction[active_direction])<=1.959963984540054)))
    saved=dict(x=x,sd_x=np.sqrt(var_x),variance_spectrum=var_sf,sd_direction_energy=sd_direction)
    return out,saved


def solver_check(f,spectral,responses,tau,m,method,x,cov,scale):
    """Independent scaled Cholesky solve checks the production inverse algebra."""
    from scipy.linalg import cho_factor,cho_solve
    variance=candidate_variances if method=="coherence_corrected" else iw.noise_variances
    errors=[]
    pairs=iw.stack_pairs(9)
    for fi in range(len(f)):
        a=iw.operator_matrix(responses[fi],pairs)
        d=iw.data_vector(spectral[fi],pairs)
        w=1/np.sqrt(variance(spectral[fi],pairs,m))
        b=a*w[:,None]*scale[fi]
        precision=b.T@b+np.eye(len(x[fi]))/tau**2
        factor=cho_factor(precision)
        xx=scale[fi]*cho_solve(factor,b.T@(d*w))
        cc=scale[fi]**2*cho_solve(factor,np.eye(len(x[fi])))
        errors.append((np.linalg.norm(xx-x[fi])/max(np.linalg.norm(xx),1e-30),
                       np.linalg.norm(cc-cov[fi])/np.linalg.norm(cc),np.linalg.cond(precision)))
    return np.max(errors,axis=0)


def evaluate(training,data,output):
    protocol=json.loads((data/"protocol.json").read_text())
    if (protocol["inverse_sha256"]!=hashlib.sha256(Path(iw.__file__).read_bytes()).hexdigest()
            or protocol["batch_sha256"]!=hashlib.sha256((vc.ROOT/"batch_invert.py").read_bytes()).hexdigest()
            or protocol["native_generator_sha256"]!=hashlib.sha256(inspect.getsource(cm.native_samples).encode()).hexdigest()
            or protocol["parent"]["candidate_sha256"]!=hashlib.sha256((inspect.getsource(cv.variance_terms)+inspect.getsource(cv.corrected_variances)).encode()).hexdigest()
            or protocol["parent"]["parent_protocol"]["source_fingerprint"]!=vc.model_fingerprint()):
        raise ValueError("frozen sources changed")
    with np.load(data/"baseline"/"seed-230000.npz") as z:
        f=z["f"]
    results,numerical=[] ,[]
    for headings in protocol["headings"]:
        beta,responses=operator_responses(training/"hydrodynamics.npz",f,headings)
        for group,settings in protocol["groups"].items():
            scored=[]
            for seed in range(settings["seed_start"],settings["seed_start"]+settings["records"]):
                with np.load(data/group/f"seed-{seed}.npz") as z:
                    np.testing.assert_array_equal(f,z["f"])
                    spectral=z["spectral"]
                for method in protocol["methods"]:
                    m=2.5 if method=="existing_2.5" else protocol["M"]
                    if method=="coherence_corrected":
                        with patch.object(iw,"noise_variances",candidate_variances):
                            x,cov,scale=iw.invert(f,spectral,responses,protocol["tau"],m)
                    else:
                        x,cov,scale=iw.invert(f,spectral,responses,protocol["tau"],m)
                    scored.append((seed,method,x,cov,scale))
                    if seed==settings["seed_start"]:
                        xx,cc,condition=solver_check(f,spectral,responses,protocol["tau"],m,method,x,cov,scale)
                        numerical.append(dict(group=group,headings=headings,method=method,
                                              max_relative_mean_difference=float(xx),max_relative_cov_difference=float(cc),
                                              max_scaled_precision_condition=float(condition)))
            # Truth is constructed only after all methods have been inverted.
            truth,full_hs=truth_spectrum(group,f,beta)
            for seed,method,x,cov,scale in scored:
                metrics,saved=recovery_metrics(x,cov,truth,f,beta,full_hs)
                results.append(dict(group=group,headings=headings,seed=seed,method=method,**metrics))
                folder=data/"fits"/str(headings)/group
                folder.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(folder/f"seed-{seed}_{method}.npz",f=f,beta=beta,truth=truth,**saved)
            print(f"Inverted {group}: {settings['records']} records, {headings} headings, all methods; no retuning",flush=True)
    output.mkdir(parents=True,exist_ok=True)
    vc.write_csv(output/"inverse_recovery_records.csv",results)
    vc.write_csv(output/"inverse_recovery_numerical.csv",numerical)
    (output/"inverse_recovery_protocol.json").write_text(json.dumps(protocol,indent=2)+"\n")


def generate_record(task):
    group,seed,output=task
    path=Path(output)/group/f"seed-{seed}.npz"
    if path.exists():
        return "cached"
    _,samples=cm.native_samples("uniform6400",group,seed)
    values=cm.spectral_record(samples)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream,**values)
    temporary.replace(path)
    return "generated"


def generate(training,output,workers):
    parent=json.loads((vc.ROOT/"cross_variance_protocol.json").read_text())
    if parent["parent_protocol"]["source_fingerprint"]!=vc.model_fingerprint():
        raise ValueError("generator differs from the calibrated source")
    candidate_hash=hashlib.sha256((inspect.getsource(cv.variance_terms)+inspect.getsource(cv.corrected_variances)).encode()).hexdigest()
    if candidate_hash!=parent["candidate_sha256"]:
        raise ValueError("variance candidate changed")
    previous=[]
    for settings in [*parent["groups"].values(),*parent["parent_protocol"]["groups"].values()]:
        previous.extend(range(settings["seed_start"],settings["seed_start"]+settings["records"]))
    groups={name:dict(seed_start=start,records=200,sea=cm.CASES[name]) for name,start in GROUPS.items()}
    for settings in groups.values():
        if set(previous).intersection(range(settings["seed_start"],settings["seed_start"]+settings["records"])):
            raise ValueError("inverse validation seeds overlap previous calibration or validation")
    protocol=dict(parent=parent,M=parent["M"],tau=iw.TAU,groups=groups,methods=METHODS,
                  headings=[12,36],primary_headings=12,bootstrap=1000,
                  inverse_sha256=hashlib.sha256(Path(iw.__file__).read_bytes()).hexdigest(),
                  batch_sha256=hashlib.sha256((vc.ROOT/"batch_invert.py").read_bytes()).hexdigest(),
                  native_generator_sha256=hashlib.sha256(inspect.getsource(cm.native_samples).encode()).hexdigest(),
                  primary_metrics=["hs_abs_pct","spectrum_nrmse","direction_abs_deg","direction_l1","m0_cover95"],
                  additional="directional NRMSE, negative power, pointwise spectrum/direction/active-bin coverage, 95th percentile and worst errors, numerical checks",
                  truth="ensemble wave energy histogrammed into retained Welch cells; directions grouped to nearest inverse heading; full-sea Hs reported separately",
                  scoring="identical fresh spectra across methods; 1000 paired whole-record bootstrap draws for error differences; exact binomial intervals for scalar 95% coverage",
                  decision="no retuning; report improvement only with paired evidence; distinguish parameter recovery from conditional uncertainty; no production adoption by this script",
                  physics="same existing shared BEM/generator assumptions as main; linear inverse versus native nonlinear time simulation; no independent-physics validation")
    output.mkdir(parents=True,exist_ok=True)
    path=output/"protocol.json"
    if path.exists() and json.loads(path.read_text())!=protocol:
        raise ValueError("output folder contains a different frozen inverse protocol")
    path.write_text(json.dumps(protocol,indent=2)+"\n")
    tasks=[(name,s,str(output)) for name,start in GROUPS.items() for s in range(start,start+200)]
    began=time.monotonic()
    with ProcessPoolExecutor(max_workers=workers,initializer=vc.initialise_worker,
                             initargs=(training/"hydrodynamics.npz",)) as pool:
        futures={pool.submit(generate_record,task):task for task in tasks}
        for number,future in enumerate(as_completed(futures),1):
            print(f"{number}/{len(tasks)} {futures[future][:2]} {future.result()}; {time.monotonic()-began:.1f}s",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate",action="store_true")
    parser.add_argument("--training",type=Path,default=Path("variance_output/m_calibration"))
    parser.add_argument("--data",type=Path,default=Path("variance_output/inverse_recovery"))
    parser.add_argument("--workers",type=int,default=2)
    parser.add_argument("--output",type=Path,default=vc.ROOT)
    args=parser.parse_args()
    if args.workers<1:
        parser.error("use at least one worker")
    if args.generate:
        generate(args.training,args.data,args.workers)
    else:
        evaluate(args.training,args.data,args.output)
