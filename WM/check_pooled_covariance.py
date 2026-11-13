"""Independent sensor pools: stationary controls and all-pairs transfer, with no oracle selection."""
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
import calibrate_m as cm
import variance_convergence as vc
import pooled_spectral as ps
import correlated_spectral as cs
from check_inverse_recovery import generate_record,operator_responses,candidate_variances,truth_spectrum
from check_correlated_covariance import score,sensor_probes

GROUPS={"baseline":250000,"long_period":251000,"weak_waves":252000}
CALIBRATION={"baseline":("train_baseline",207000),"long_period":("test_long_period",209000),
             "weak_waves":("test_weak_waves",210000)}


def generate(training,data,workers):
    parent=json.loads((vc.ROOT/"correlated_covariance_protocol.json").read_text())
    if parent["parent"]["parent"]["parent_protocol"]["source_fingerprint"]!=vc.model_fingerprint():
        raise ValueError("generator changed")
    groups={g:dict(seed_start=s,records=200) for g,s in GROUPS.items()}
    protocol=dict(M=parent["M"],tau=parent["tau"],groups=groups,headings=[12,36],
                  calibration={g:dict(folder=folder,seed_start=s,records=200) for g,(folder,s) in CALIBRATION.items()},
                  methods=["self_diagonal","pool_diagonal","pool_joint"],
                  pool_sha256=hashlib.sha256(Path(ps.__file__).read_bytes()).hexdigest(),
                  joint_sha256=hashlib.sha256(Path(cs.__file__).read_bytes()).hexdigest(),
                  generator_fingerprint=vc.model_fingerprint(),
                  native_sha256=hashlib.sha256(inspect.getsource(cm.native_samples).encode()).hexdigest(),
                  inverse_sha256=hashlib.sha256(Path(iw.__file__).read_bytes()).hexdigest(),
                  solver="prior-aware block-SVD preconditioned Gaussian solve; float-precision numerical-rank cutoff only",
                  failures="nonpositive total energy is a failed recovery: Hs clipped to zero as in production, 100 percent height error, undefined direction marked missing; failures never discarded",
                  covariance="mean fixed-M corrected record variances; empirical per-frequency correlation shrunk by estimated squared-error risk; joint Hann frequency kernel",
                  separation="every pool is frozen before test generation; every pool applied to every test sea; no true-condition selector, no current-record covariance adaptation",
                  interpretation="same-condition entries are stationary-calibration controls; off-condition entries are transfer stress tests, not an oracle selection rule",
                  calibration_noise="200 records learn the covariance of a NEW 900-second record; no division by pool size; empirical correlations and stabilisation use sensor records only",
                  prior="tau and recipient-record prior scales identical across all methods",
                  validation="20 fixed sensor-only projection variances; matched and all-pairs transfer recovery; exact binomial coverage and 1000 paired record bootstrap draws",
                  limitations="200 stationary calibration records are an idealised pool; intervals conditional on each fixed pool; existing shared simulator physics remains; no production adoption")
    data.mkdir(parents=True,exist_ok=True)
    path=data/"protocol.json"
    if path.exists() and json.loads(path.read_text())!=protocol:
        raise ValueError("different frozen pool protocol")
    path.write_text(json.dumps(protocol,indent=2)+"\n")
    for pool,settings in protocol["calibration"].items():
        target=data/f"pool-{pool}.npz"
        if target.exists() and (data/"pool_manifest.json").exists():
            frozen=json.loads((data/"pool_manifest.json").read_text())
            if hashlib.sha256(target.read_bytes()).hexdigest()!=frozen["pools"][target.name]:
                raise ValueError("cached calibration matrix changed")
            continue
        spectra=[]
        for seed in range(settings["seed_start"],settings["seed_start"]+settings["records"]):
            with np.load(training/"uniform6400"/settings["folder"]/f"seed-{seed}.npz") as z:
                f=z["f"];spectra.append(z["spectral"])
        factors,shrinkage=ps.fit_pool(np.stack(spectra),protocol["M"])
        np.savez_compressed(target,f=f,factors=factors,shrinkage=shrinkage)
    manifest=dict(pools={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(data.glob("pool-*.npz"))},
                  source={name:hashlib.sha256((vc.ROOT/name).read_bytes()).hexdigest() for name in
                          ["pooled_spectral.py","check_pooled_covariance.py","analyse_pooled_covariance.py","test_pooled_spectral.py"]},
                  stage="calibration matrices and statistical rules frozen before fresh outcomes",
                  calibration_records=200,target_noise="new native-duration recording; no division by calibration size")
    (data/"pool_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    tasks=[(g,s,str(data)) for g,start in GROUPS.items() for s in range(start,start+200)]
    began=time.monotonic()
    with ProcessPoolExecutor(max_workers=workers,initializer=vc.initialise_worker,initargs=(training/"hydrodynamics.npz",)) as pool:
        futures={pool.submit(generate_record,t):t for t in tasks}
        for index,future in enumerate(as_completed(futures),1):
            print(f"{index}/600 {futures[future][:2]} {future.result()}; {time.monotonic()-began:.1f}s",flush=True)


def initialise(training,data):
    global CONTEXT
    protocol=json.loads((data/"protocol.json").read_text())
    pools={}
    for name in protocol["calibration"]:
        with np.load(data/f"pool-{name}.npz") as z:
            f=z["f"];factors=z["factors"]
            pools[name]=(factors,np.array([np.diag(np.linalg.norm(a,axis=1)) for a in factors]))
    responses={h:operator_responses(training/"hydrodynamics.npz",f,h) for h in protocol["headings"]}
    rho=cs.welch_frequency_correlation(np.rint(f*iw.SEG).astype(int))
    CONTEXT=protocol,f,pools,responses,rho,sensor_probes(training),data


def evaluate_record(task):
    group,seed=task
    protocol,f,pools,responses,rho,probes,data=CONTEXT
    path=data/"results"/group/f"seed-{seed}.npz"
    if path.exists():return group,seed,"cached"
    with np.load(data/group/f"seed-{seed}.npz") as z:
        np.testing.assert_array_equal(f,z["f"])
        spectral=z["spectral"];observed=z["observed"]
    posterior=[];identity=np.eye(len(f))
    predicted=[]
    for factors,diagonal in pools.values():
        predicted.extend([cs.propagated_variance(probes,diagonal,identity),cs.propagated_variance(probes,factors,rho)])
    saved=dict(f=f,probe_observed=np.einsum("bfc,fc->b",probes,observed),probe_predicted=np.stack(predicted))
    for headings,(beta,h) in responses.items():
        with patch.object(iw,"noise_variances",candidate_variances):
            x,cov,scales=iw.invert(f,spectral,h,protocol["tau"],protocol["M"])
        t=cs.functional_vectors(np.ones(len(f)),headings,float(f[1]-f[0])).reshape(len(f),headings,-1)
        function_cov=np.einsum("fji,fjk,fkl->il",t,cov,t)
        posterior.append((headings,"none","self_diagonal",beta,x,function_cov,0.))
        for name,(factors,diagonal) in pools.items():
            for method,root,frequency in [("pool_diagonal",diagonal,identity),("pool_joint",factors,rho)]:
                x,functions,residual=ps.pooled_inverse(f,spectral,h,protocol["tau"],root,frequency,scales)
                posterior.append((headings,name,method,beta,x,functions,residual))
    rows=[]
    for headings,pool,method,beta,x,covariance,residual in posterior:
        truth,_=truth_spectrum(group,f,beta)
        status="ok"
        try:
            metrics=score(x,covariance,truth,f,beta)
        except ValueError:
            if (not np.isfinite(x).all() or not np.isfinite(covariance).all()
                    or np.any(np.diag(covariance)<=0)):
                raise
            df=float(f[1]-f[0]);m0=float(x.sum()*df)
            if m0>0:raise
            true_m0=float(truth.sum()*df);variance=np.diag(covariance)
            z=(m0-true_m0)/np.sqrt(variance[-1]);sf=x.sum(1);target_sf=truth.sum(1)
            status="nonpositive_energy"
            metrics=dict(Hs=0.,Hs_true_band=4*np.sqrt(true_m0),m0=m0,m0_true=true_m0,
                         hs_abs_pct=100.,hs_err_pct=-100.,
                         spectrum_nrmse=float(np.linalg.norm(sf-target_sf)/np.linalg.norm(target_sf)),
                         direction_abs_deg=float("nan"),direction_l1=float("nan"),
                         negative_fraction=float(np.maximum(-x,0).sum()/abs(x).sum()),
                         sd_m0=float(np.sqrt(variance[-1])),sd_Hs=float("nan"),
                         m0_cover68=float(abs(z)<=1),m0_cover95=float(abs(z)<=1.959963984540054),
                         spectrum_cover95=float(np.mean(abs(sf-target_sf)<=1.959963984540054*np.sqrt(variance[:len(f)]))),
                         active_direction_cover95=float("nan"))
        rows.append(dict(group=group,seed=seed,headings=headings,pool=pool,method=method,status=status,
                         solver_residual=residual,**metrics))
        saved[f"x_{headings}_{pool}_{method}"]=x
        saved[f"variance_{headings}_{pool}_{method}"]=np.diag(covariance)
    saved["records"]=json.dumps(rows)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(".tmp")
    with temporary.open("wb") as stream:np.savez_compressed(stream,**saved)
    temporary.replace(path)
    return group,seed,"evaluated"


def evaluate(training,data,workers):
    protocol=json.loads((data/"protocol.json").read_text())
    for key,module in [("pool_sha256",ps),("joint_sha256",cs),("inverse_sha256",iw)]:
        if protocol[key]!=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest():
            raise ValueError("frozen source changed")
    if protocol["generator_fingerprint"]!=vc.model_fingerprint():raise ValueError("generator changed")
    manifest=json.loads((data/"pool_manifest.json").read_text())
    for name,digest in manifest["pools"].items():
        if hashlib.sha256((data/name).read_bytes()).hexdigest()!=digest:
            raise ValueError("frozen calibration pool changed")
    tasks=[(g,s) for g,start in GROUPS.items() for s in range(start,start+200)]
    began=time.monotonic()
    with ProcessPoolExecutor(max_workers=workers,initializer=initialise,initargs=(training,data)) as pool:
        for index,result in enumerate(pool.map(evaluate_record,tasks),1):
            print(f"{index}/600 {result}; {time.monotonic()-began:.1f}s",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate",action="store_true")
    parser.add_argument("--training",type=Path,default=Path("variance_output/m_calibration"))
    parser.add_argument("--data",type=Path,default=Path("variance_output/pooled_covariance"))
    parser.add_argument("--workers",type=int,default=2)
    args=parser.parse_args()
    if args.generate:generate(args.training,args.data,args.workers)
    else:evaluate(args.training,args.data,args.workers)
