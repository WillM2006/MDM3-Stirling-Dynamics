"""Check coherence-aware marginal variances on fresh native-duration records."""
import argparse
import hashlib
import inspect
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import calibrate_m as cm
import invert_waves as iw
import variance_convergence as vc
from analyse_m_calibration import families, fitted_m, load_records, moments
from analyse_variance_convergence import interval
from analyse_variance_resolution import BANDS

GROUPS = {"test_baseline": ("baseline", 220000),
          "test_long_period": ("long_period", 221000),
          "test_weak_waves": ("weak_waves", 222000)}


def variance_terms(spectral):
    """Quadratic terms whose record averages give the candidate variance estimate."""
    if (spectral.shape[-1] != spectral.shape[-2] or not np.isfinite(spectral).all()
            or not np.allclose(spectral, spectral.swapaxes(-1,-2).conj(), rtol=1e-10, atol=0)):
        raise ValueError("expected finite Hermitian spectral matrices")
    residual, square = [], []
    for c,k in iw.stack_pairs(spectral.shape[-1]):
        a,b = spectral[...,c,c].real, spectral[...,k,k].real
        if np.any(a<=0) or np.any(b<=0):
            raise ValueError("nonpositive auto-spectrum")
        if c==k:
            residual.append(np.zeros_like(a)); square.append(a*a)
        else:
            z=spectral[...,c,k]
            difference=a*b-abs(z)**2
            if np.any(difference < -1e-10*a*b):
                raise ValueError("cross-spectrum violates positive semidefiniteness")
            # Roundoff can make an exactly coherent pair very slightly negative.
            base=np.maximum(difference,0)/2
            residual.extend([base,base]); square.extend([z.real**2,z.imag**2])
    return np.stack(residual,axis=-1),np.stack(square,axis=-1)


def corrected_variances(residual,square,m):
    """M is the existing plug-in denominator; effective Wishart averages are M-1."""
    if np.any(~np.isfinite(m)) or np.any(np.asarray(m)<=2):
        raise ValueError("the finite-average correction requires M > 2")
    return ((m-1)/(m-2)*residual+square)/m


def training(data):
    protocol=json.loads((data/"protocol.json").read_text())
    settings=protocol["groups"]["train_baseline"]
    f,d,q=load_records(data,"uniform6400","train_baseline",settings)
    n=len(d)
    weights=np.random.default_rng(8131).multinomial(n,np.full(n,1/n),size=1000)/n
    m,draws=fitted_m(moments(d,q,weights),families()["auto"])
    return protocol,f,m,draws


def generate_record(task):
    group,seed,output=task
    path=Path(output)/group/f"seed-{seed}.npz"
    if path.exists():
        return "cached"
    _,samples=cm.native_samples("uniform6400",GROUPS[group][0],seed)
    values=cm.spectral_record(samples)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream,**values)
    temporary.replace(path)
    return "generated"


def generate(data,output,workers):
    old,_,m,draws=training(data)
    if old["source_fingerprint"] != vc.model_fingerprint():
        raise ValueError("generator differs from calibration data")
    used=set()
    for settings in old["groups"].values():
        used.update(range(settings["seed_start"],settings["seed_start"]+settings["records"]))
    for _,start in GROUPS.values():
        if used.intersection(range(start,start+200)):
            raise ValueError("fresh validation seeds overlap previous study")
    protocol=dict(parent_protocol=old,training_group="train_baseline",M=m,
                  M_ci95=list(map(float,interval(draws))),effective_averages=m-1,
                  groups={name:dict(case=case,seed_start=seed,records=200) for name,(case,seed) in GROUPS.items()},
                  methods=["existing_2.5","calibrated_equal","coherence_corrected"],bootstrap=1000,
                  decision="fixed baseline-auto training M; no test fitting, inverse, or wave-truth inputs",
                  primary="median absolute log variance-ratio error across full-band cross entries; report signed bias and per-family results too",
                  tolerance="+/-10 percent for pooled median variance ratios; exploratory pointwise intervals are not simultaneous",
                  candidate="auto = a^2/M; cross Re/Im = [(M-1)/(2*(M-2))*(a*b-|z|^2) + Re(z)^2 or Im(z)^2]/M",
                  interpretation="M remains the fitted implemented auto denominator; complex-Wishart effective averages L=M-1; no extra fitted parameter",
                  candidate_sha256=hashlib.sha256((inspect.getsource(variance_terms)+inspect.getsource(corrected_variances)).encode()).hexdigest(),
                  native_generator_sha256=hashlib.sha256(inspect.getsource(cm.native_samples).encode()).hexdigest())
    output.mkdir(parents=True,exist_ok=True)
    path=output/"protocol.json"
    if path.exists() and json.loads(path.read_text())!=protocol:
        raise ValueError("output folder contains a different protocol")
    path.write_text(json.dumps(protocol,indent=2)+"\n")
    cache=data/"hydrodynamics.npz"
    tasks=[(group,s,str(output)) for group,(_,start) in GROUPS.items() for s in range(start,start+200)]
    began=time.monotonic()
    with ProcessPoolExecutor(max_workers=workers,initializer=vc.initialise_worker,initargs=(cache,)) as pool:
        futures={pool.submit(generate_record,task):task for task in tasks}
        for number,future in enumerate(as_completed(futures),1):
            print(f"{number}/{len(tasks)} {futures[future][:2]} {future.result()}; {time.monotonic()-began:.1f}s",flush=True)


def analyse(training_data,data,output):
    protocol=json.loads((data/"protocol.json").read_text())
    _,f,m,m_draws=training(training_data)
    candidate_hash=hashlib.sha256((inspect.getsource(variance_terms)+inspect.getsource(corrected_variances)).encode()).hexdigest()
    if m!=protocol["M"] or candidate_hash!=protocol["candidate_sha256"]:
        raise ValueError("candidate or training calibration changed after validation protocol was saved")
    if (protocol["parent_protocol"]["source_fingerprint"]!=vc.model_fingerprint()
            or protocol["native_generator_sha256"]!=hashlib.sha256(inspect.getsource(cm.native_samples).encode()).hexdigest()
            or protocol["parent_protocol"]["inverse_source_sha256"]!=hashlib.sha256(Path(iw.__file__).read_bytes()).hexdigest()):
        raise ValueError("generator or spectral preprocessing changed since calibration")
    columns=families()
    columns["cross"]=sorted(columns["real"]+columns["imag"])
    summary,scores,frequencies,entries=[],[],[],[]
    for index,(group,settings) in enumerate(protocol["groups"].items()):
        d,q,residual,square=[],[],[],[]
        for seed in range(settings["seed_start"],settings["seed_start"]+settings["records"]):
            with np.load(data/group/f"seed-{seed}.npz") as z:
                np.testing.assert_array_equal(f,z["f"])
                r,s=variance_terms(z["spectral"])
                d.append(z["observed"]);q.append(z["numerator"])
                residual.append(r);square.append(s)
        d,q,residual,square=map(np.stack,(d,q,residual,square))
        n=len(d)
        weights=np.random.default_rng(8632+index).multinomial(n,np.full(n,1/n),size=protocol["bootstrap"])/n
        stats=moments(d,q,weights)
        br=np.einsum("bn,nfc->bfc",weights,residual)
        bs=np.einsum("bn,nfc->bfc",weights,square)
        predicted={"existing_2.5":stats["numerator"]/2.5,"calibrated_equal":stats["numerator"]/m,
                   "coherence_corrected":corrected_variances(residual.mean(0),square.mean(0),m)}
        boot={"existing_2.5":stats["bq"]/(2.5*stats["bv"]),
              "calibrated_equal":stats["bq"]/(m_draws[:,None,None]*stats["bv"]),
              "coherence_corrected":corrected_variances(br,bs,m_draws[:,None,None])/stats["bv"]}
        ratio={method:value/stats["variance"] for method,value in predicted.items()}
        for family,selected in columns.items():
            for band,(start,stop) in BANDS.items():
                mask=(f>=start)&((f<=stop) if band in ("whole","tail") else (f<stop))
                for method in predicted:
                    r=ratio[method][mask][:,selected]
                    rb=boot[method][:,mask][:,:,selected]
                    low,high=interval(np.median(rb,axis=(1,2)))
                    factor=np.exp(np.median(abs(np.log(r))))
                    summary.append(dict(group=group,family=family,band=band,method=method,records=n,
                                        M=2.5 if method=="existing_2.5" else m,
                                        median_predicted_over_observed=float(np.median(r)),ci95_low=float(low),ci95_high=float(high),
                                        median_factor_error=float(factor),
                                        fraction_within_10pct=float(np.mean(abs(r-1)<=.1)),
                                        fraction_within_factor_two=float(np.mean((r>=.5)&(r<=2)))))
                for reference in ["existing_2.5","calibrated_equal"]:
                    old=boot[reference][:,mask][:,:,selected]
                    new=boot["coherence_corrected"][:,mask][:,:,selected]
                    change=np.median(abs(np.log(new)),axis=(1,2))-np.median(abs(np.log(old)),axis=(1,2))
                    lo,hi=interval(change)
                    point=np.median(abs(np.log(ratio["coherence_corrected"][mask][:,selected])))-np.median(abs(np.log(ratio[reference][mask][:,selected])))
                    scores.append(dict(group=group,family=family,band=band,reference=reference,
                                       change_in_median_absolute_log_error=float(point),ci95_low=float(lo),ci95_high=float(hi),
                                       basic_ci95_low=float(2*point-hi),basic_ci95_high=float(2*point-lo)))
            if family in ("auto","real","imag"):
                for method in predicted:
                    med=np.median(ratio[method][:,selected],axis=1)
                    lo,hi=interval(np.median(boot[method][:,:,selected],axis=2))
                    for i,frequency in enumerate(f):
                        frequencies.append(dict(group=group,family=family,method=method,frequency_Hz=frequency,
                                                median_predicted_over_observed=float(med[i]),ci95_low=float(lo[i]),ci95_high=float(hi[i])))
        for method,r in ratio.items():
            for fi,frequency in enumerate(f):
                for column in range(81):
                    entries.append(dict(group=group,method=method,frequency_Hz=frequency,column=column,
                                        family=next(name for name in ("auto","real","imag") if column in columns[name]),
                                        predicted_over_observed=float(r[fi,column])))
        print(f"Validated {group}: {n} fresh records; no refit",flush=True)
    output.mkdir(parents=True,exist_ok=True)
    for name,rows in [("validation",summary),("scores",scores),("frequencies",frequencies),("entries",entries)]:
        vc.write_csv(output/f"cross_variance_{name}.csv",rows)
    (output/"cross_variance_protocol.json").write_text(json.dumps(protocol,indent=2)+"\n")
    plot(summary,entries,output)


def plot(summary,entries,output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    methods=[("existing_2.5","#888888","Existing M=2.5"),
             ("calibrated_equal","#2864A5","Calibrated M; equal Re/Im"),
             ("coherence_corrected","#21835D","Same M; corrected Re/Im")]
    groups=list(GROUPS)
    fig,axes=plt.subplots(1,3,figsize=(14,5.2))
    for ax,family,title in zip(axes,["auto","real","imag"],["Auto-spectra","Real cross-spectra","Imaginary cross-spectra"]):
        for method,colour,label in methods:
            rows=[next(r for r in summary if r["group"]==group and r["family"]==family and r["band"]=="whole" and r["method"]==method) for group in groups]
            y=np.array([r["median_predicted_over_observed"] for r in rows])
            ax.plot(range(3),y,"o-",color=colour,label=label)
            ax.vlines(range(3),[r["ci95_low"] for r in rows],[r["ci95_high"] for r in rows],color=colour,alpha=.65)
        ax.axhspan(.9,1.1,color="#888888",alpha=.12)
        ax.axhline(1,color="#555555",ls="--",lw=1)
        ax.set_xticks(range(3),["Baseline","Long period","Weak waves"])
        ax.set(title=title,ylabel="Median predicted / observed variance",ylim=(.65,3.9))
        ax.grid(alpha=.15)
    axes[0].legend(frameon=False,fontsize=8)
    fig.suptitle("Fresh validation: does correcting the cross-spectral formula help?")
    fig.text(.04,.025,"200 new records per condition; M=8.785 fitted earlier to separate baseline auto-spectra. Blue and green auto-spectra coincide.\n"
             "Bars: 95% whole-record bootstrap intervals including training-M uncertainty. Target: one; grey band: ±10%. No inverse fitting.",fontsize=8,color="#555555")
    fig.tight_layout(rect=(.02,.14,.99,.94))
    fig.savefig(output/"cross_variance_validation.png",dpi=170)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,5.2))
    for ax,group,title in zip(axes,groups,["Baseline","Long period","Weak waves"]):
        for method,colour,label in methods:
            values=np.array([r["predicted_over_observed"] for r in entries if r["group"]==group and r["method"]==method and r["family"]!="auto"])
            factor=np.exp(abs(np.log(values)))
            ax.plot(np.sort(factor),np.arange(1,len(factor)+1)/len(factor),color=colour,label=label)
        ax.axvline(1.1,color="#555555",ls="--",lw=1)
        ax.set_xscale("log")
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticks([1,1.25,1.5,2,3,5],["1","1.25","1.5","2","3","5"])
        ax.set(title=title,xlabel="Variance error factor (closer to one is better)",ylabel="Fraction of cross-spectral entries",xlim=(1,6),ylim=(0,1))
        ax.grid(alpha=.15)
    axes[0].legend(frameon=False,fontsize=8,loc="lower right")
    fig.suptitle("Individual cross-spectral entries: improvement beyond a pooled average")
    fig.text(.04,.025,"Both real and imaginary entries across all 53 frequencies (3,816 entries per condition). Dashed line: factor 1.1. Factor = max(ratio, 1/ratio).\n"
             "Observed variances are estimated from 200 records, so entrywise scatter includes sampling error. Curves are descriptive, not simultaneous confidence bounds.",fontsize=8,color="#555555")
    fig.tight_layout(rect=(.02,.14,.99,.94))
    fig.savefig(output/"cross_variance_entries.png",dpi=170)
    plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training",type=Path,default=Path("variance_output/m_calibration"))
    parser.add_argument("--data",type=Path,default=Path("variance_output/cross_variance"))
    parser.add_argument("--generate",action="store_true")
    parser.add_argument("--output",type=Path,default=vc.ROOT)
    parser.add_argument("--workers",type=int,default=2)
    args=parser.parse_args()
    if args.workers<1:
        parser.error("use at least one worker")
    if args.generate:
        generate(args.training,args.data,args.workers)
    else:
        analyse(args.training,args.data,args.output)
