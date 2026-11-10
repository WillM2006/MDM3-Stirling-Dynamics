"""Fixed-M covariance experiment: sensor-only structure, fresh inverse validation."""
import argparse
import hashlib
import inspect
import json
import time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
from unittest.mock import patch

import numpy as np
import invert_waves as iw
import variance_convergence as vc
import calibrate_m as cm
import correlated_spectral as cs
from check_inverse_recovery import (generate_record,operator_responses,candidate_variances,
                                    truth_spectrum)

GROUPS={"baseline":240000,"long_period":241000,"weak_waves":242000}
METHODS=["diagonal","same_frequency","joint_hann"]


def generate(training,data,workers):
    parent=json.loads((vc.ROOT/"inverse_recovery_protocol.json").read_text())
    if parent["parent"]["parent_protocol"]["source_fingerprint"]!=vc.model_fingerprint():
        raise ValueError("source generator changed")
    groups={name:dict(seed_start=start,records=200) for name,start in GROUPS.items()}
    protocol=dict(parent=parent,M=parent["M"],tau=parent["tau"],groups=groups,headings=[12,36],methods=METHODS,
                  covariance_sha256=hashlib.sha256(Path(cs.__file__).read_bytes()).hexdigest(),
                  native_generator_sha256=hashlib.sha256(inspect.getsource(cm.native_samples).encode()).hexdigest(),
                  inverse_sha256=hashlib.sha256(Path(iw.__file__).read_bytes()).hexdigest(),
                  correlation="full Wishart Hermitian-basis factor from each measured sensor correlation matrix; row-normalised to retain fixed-M variances",
                  stabilisation="shrink sensor correlation toward identity with clipped estimated complex-Wishart offdiagonal risk / measured squared offdiagonal norm; no inverse tuning",
                  frequency="exact white/local-flat Hann overlap power correlation, aligned through Hermitian spectral-square-root factors; approximate for varying coloured spectra",
                  calibration="only baseline sensor records 207000--207199 set units for fixed sensor-level probes; covariance estimator itself uses each record, M and Welch settings only",
                  comparison="diagonal versus correlated entries only versus jointly correlated entries/frequencies; prior scales identical across methods",
                  validation="20 fixed sensor-only projections first, then band Hs, spectrum/direction errors, negative power, exact binomial coverage; 1000 paired record bootstrap draws",
                  decision="no retuning after validation; no production, report or Drive writes; no proof of physical independence from fresh seeds")
    data.mkdir(parents=True,exist_ok=True)
    path=data/"protocol.json"
    if path.exists() and json.loads(path.read_text())!=protocol:
        raise ValueError("different frozen covariance protocol")
    path.write_text(json.dumps(protocol,indent=2)+"\n")
    tasks=[(group,seed,str(data)) for group,start in GROUPS.items() for seed in range(start,start+200)]
    began=time.monotonic()
    with ProcessPoolExecutor(max_workers=workers,initializer=vc.initialise_worker,
                             initargs=(training/"hydrodynamics.npz",)) as pool:
        futures={pool.submit(generate_record,task):task for task in tasks}
        for index,future in enumerate(as_completed(futures),1):
            print(f"{index}/{len(tasks)} {futures[future][:2]} {future.result()}; {time.monotonic()-began:.1f}s",flush=True)


def sensor_probes(training):
    from analyse_m_calibration import families
    d=[]
    for seed in range(207000,207200):
        with np.load(training/"uniform6400"/"train_baseline"/f"seed-{seed}.npz") as z:
            d.append(z["observed"])
    scales=np.std(d,axis=0,ddof=1)
    rng=np.random.default_rng(840137)
    weights=np.broadcast_to(rng.normal(size=(20,1,81)),(20,53,81)).copy()
    for index,family in enumerate(["auto","real","imag"]):
        weights[index]=0
        weights[index][:,families()[family]]=1
    weights/=scales[None]
    weights/=np.linalg.norm(weights,axis=(1,2))[:,None,None]
    return weights


def score(x,covariance,truth,f,beta):
    count=len(f);df=float(f[1]-f[0])
    sf=x.sum(1);target_sf=truth.sum(1)
    p=x.sum(0)*df;target_p=truth.sum(0)*df
    m0=float(p.sum());true_m0=float(target_p.sum())
    diagonal=np.diag(covariance)
    if m0<=0 or np.any(diagonal<=0) or not np.isfinite(x).all() or not np.isfinite(covariance).all():
        raise ValueError("invalid posterior result")
    hs=4*np.sqrt(m0);true_hs=4*np.sqrt(true_m0)
    direction=np.degrees(np.angle(np.sum(p*np.exp(1j*beta))/np.sum(target_p*np.exp(1j*beta))))
    z=(m0-true_m0)/np.sqrt(diagonal[-1])
    active=target_p>=.01*target_p.max()
    return dict(Hs=hs,Hs_true_band=true_hs,m0=m0,m0_true=true_m0,
                hs_abs_pct=100*abs(hs/true_hs-1),hs_err_pct=100*(hs/true_hs-1),
                spectrum_nrmse=float(np.linalg.norm(sf-target_sf)/np.linalg.norm(target_sf)),
                direction_abs_deg=float(abs(direction)),direction_l1=float(.5*np.sum(abs(p/m0-target_p/true_m0))),
                negative_fraction=float(np.maximum(-x,0).sum()/abs(x).sum()),
                sd_m0=float(np.sqrt(diagonal[-1])),sd_Hs=float(2*np.sqrt(diagonal[-1]/m0)),
                m0_cover68=float(abs(z)<=1),m0_cover95=float(abs(z)<=1.959963984540054),
                spectrum_cover95=float(np.mean(abs(sf-target_sf)<=1.959963984540054*np.sqrt(diagonal[:count]))),
                active_direction_cover95=float(np.mean(abs(p[active]-target_p[active])<=1.959963984540054*np.sqrt(diagonal[count:-1][active]))))


def initialise_evaluation(training,data):
    global CONTEXT
    protocol=json.loads((data/"protocol.json").read_text())
    with np.load(data/"baseline"/"seed-240000.npz") as z:f=z["f"]
    responses={h:operator_responses(training/"hydrodynamics.npz",f,h) for h in protocol["headings"]}
    rho=cs.welch_frequency_correlation(np.rint(f*iw.SEG).astype(int))
    CONTEXT=protocol,f,responses,rho,sensor_probes(training),data


def evaluate_record(task):
    group,seed=task
    protocol,f,responses,rho,probes,data=CONTEXT
    path=data/"results"/group/f"seed-{seed}.npz"
    if path.exists():
        return group,seed,"cached"
    with np.load(data/group/f"seed-{seed}.npz") as z:
        np.testing.assert_array_equal(f,z["f"])
        spectral=z["spectral"];observed=z["observed"]
    factors,shrinkage=cs.covariance_factors(spectral,protocol["M"])
    standard=np.sqrt(np.sum(factors**2,axis=2))
    diagonal_factors=np.array([np.diag(s) for s in standard])
    identity=np.eye(len(f))
    predicted=np.stack([cs.propagated_variance(probes,diagonal_factors,identity),
                        cs.propagated_variance(probes,factors,identity),cs.propagated_variance(probes,factors,rho)])
    saved=dict(f=f,probe_observed=np.einsum("bfc,fc->b",probes,observed),probe_predicted=predicted,shrinkage=shrinkage)
    posterior=[]
    for headings,(beta,h) in responses.items():
        with patch.object(iw,"noise_variances",candidate_variances):
            x,cov,scales=iw.invert(f,spectral,h,protocol["tau"],protocol["M"])
        t=cs.functional_vectors(np.ones(len(f)),headings,float(f[1]-f[0])).reshape(len(f),headings,-1)
        function_cov=np.einsum("fji,fjk,fkl->il",t,cov,t)
        posterior.append((headings,"diagonal",beta,x,function_cov,0.))
        for method,correlation in [("same_frequency",identity),("joint_hann",rho)]:
            x,covariance,residual=cs.correlated_inverse(f,spectral,h,protocol["tau"],protocol["M"],factors,correlation,scales)
            posterior.append((headings,method,beta,x,covariance,residual))
    # Scoring-only truth is not constructed until every inverse above has finished.
    rows=[]
    for headings,method,beta,x,covariance,residual in posterior:
        truth,_=truth_spectrum(group,f,beta)
        rows.append(dict(group=group,seed=seed,headings=headings,method=method,
                         solver_residual=residual,**score(x,covariance,truth,f,beta)))
        saved[f"x_{headings}_{method}"]=x
        saved[f"variance_{headings}_{method}"]=np.diag(covariance)
    saved["records"]=json.dumps(rows)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(".tmp")
    with temporary.open("wb") as stream:np.savez_compressed(stream,**saved)
    temporary.replace(path)
    return group,seed,"evaluated"


def evaluate(training,data,workers):
    protocol=json.loads((data/"protocol.json").read_text())
    if (protocol["covariance_sha256"]!=hashlib.sha256(Path(cs.__file__).read_bytes()).hexdigest()
            or protocol["native_generator_sha256"]!=hashlib.sha256(inspect.getsource(cm.native_samples).encode()).hexdigest()
            or protocol["inverse_sha256"]!=hashlib.sha256(Path(iw.__file__).read_bytes()).hexdigest()):
        raise ValueError("frozen sources changed")
    tasks=[(group,s) for group,settings in protocol["groups"].items() for s in range(settings["seed_start"],settings["seed_start"]+settings["records"])]
    began=time.monotonic()
    with ProcessPoolExecutor(max_workers=workers,initializer=initialise_evaluation,initargs=(training,data)) as pool:
        for index,result in enumerate(pool.map(evaluate_record,tasks),1):
            print(f"{index}/{len(tasks)} {result}; {time.monotonic()-began:.1f}s",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate",action="store_true")
    parser.add_argument("--training",type=Path,default=Path("variance_output/m_calibration"))
    parser.add_argument("--data",type=Path,default=Path("variance_output/correlated_covariance"))
    parser.add_argument("--workers",type=int,default=2)
    args=parser.parse_args()
    if args.generate:generate(args.training,args.data,args.workers)
    else:evaluate(args.training,args.data,args.workers)
