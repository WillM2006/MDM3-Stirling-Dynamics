"""Fit one M on training spectra; evaluate untouched native-duration records."""
import argparse
import json
from pathlib import Path

import numpy as np
import invert_waves as iw
from analyse_variance_convergence import bootstrap_moments, interval
from analyse_variance_resolution import BANDS
from variance_convergence import ROOT, write_csv


def families():
    result={"auto":[],"real":[],"imag":[],"all":list(range(81))}
    position=0
    for c,k in iw.stack_pairs(9):
        if c==k:
            result["auto"].append(position)
            position+=1
        else:
            result["real"].append(position)
            result["imag"].append(position+1)
            position+=2
    return result


def moments(observed,numerator,weights):
    if (observed.shape!=numerator.shape or len(observed)<8 or not np.isfinite(observed).all()
            or not np.isfinite(numerator).all() or np.any(numerator<=0)):
        raise ValueError("invalid calibration observations or variance numerators")
    variance=observed.var(0,ddof=1)
    if np.any(variance<=0):
        raise ValueError("zero repeat-record variance")
    bm,bv=bootstrap_moments(observed,weights)
    bq=np.einsum("bn,nfc->bfc",weights,numerator)
    return dict(mean=observed.mean(0),variance=variance,numerator=numerator.mean(0),bm=bm,bv=bv,bq=bq)


def fitted_m(statistics,columns):
    value=float(np.median((statistics["numerator"]/statistics["variance"])[:,columns]))
    draws=np.median((statistics["bq"]/statistics["bv"])[:,:,columns],axis=(1,2))
    return value,draws


def load_records(data,mode,group,settings):
    observations,numerators=[],[]
    f=None
    for seed in range(settings["seed_start"],settings["seed_start"]+settings["records"]):
        with np.load(data/mode/group/f"seed-{seed}.npz") as z:
            if f is None:
                f=z["f"].copy()
            np.testing.assert_array_equal(f,z["f"])
            if z["observed"].shape!=(len(f),81) or z["numerator"].shape!=(len(f),81):
                raise ValueError("invalid saved spectral shape")
            observations.append(z["observed"])
            numerators.append(z["numerator"])
    return f,np.stack(observations),np.stack(numerators)


def analyse(data,output):
    protocol=json.loads((data/"protocol.json").read_text())
    groups=protocol["groups"]
    train=groups["train_baseline"]
    train_seeds=set(range(train["seed_start"],train["seed_start"]+train["records"]))
    for group,settings in groups.items():
        if group.startswith("test_") and train_seeds.intersection(range(settings["seed_start"],settings["seed_start"]+settings["records"])):
            raise ValueError("training and held-out seeds overlap")
    weights={}
    for index,(group,settings) in enumerate(groups.items()):
        n=settings["records"]
        weights[group]=np.random.default_rng(8131+index).multinomial(n,np.full(n,1/n),size=protocol["bootstrap"])/n
    columns=families()
    fits,summary,frequency_rows,scores=[],[],[],[]
    for mode in protocol["modes"]:
        f,d,q=load_records(data,mode,"train_baseline",train)
        training=moments(d,q,weights["train_baseline"])
        fitted={}
        for family,selected in columns.items():
            value,draws=fitted_m(training,selected)
            low,high=interval(draws)
            equivalent,equivalent_low,equivalent_high="","",""
            if family=="auto":
                equivalent=float(np.median((training["mean"]**2/training["variance"])[:,selected]))
                eb=np.median((training["bm"]**2/training["bv"])[:,:,selected],axis=(1,2))
                equivalent_low,equivalent_high=map(float,interval(eb))
            fits.append(dict(mode=mode,family=family,primary=family=="auto",training_records=len(d),
                             M=value,ci95_low=float(low),ci95_high=float(high),
                             auto_variance_equivalent_M=equivalent,auto_equivalent_ci95_low=equivalent_low,
                             auto_equivalent_ci95_high=equivalent_high))
            fitted[family]=(value,draws)
        value,m_draws=fitted["auto"]
        print(f"{mode}: training auto M={value:.5f}, 95% interval {interval(m_draws)}",flush=True)
        del training,d,q,fitted
        for group,settings in groups.items():
            if not group.startswith("test_") or (mode=="current40" and group not in protocol["current40_groups"]):
                continue
            ff,d,q=load_records(data,mode,group,settings)
            np.testing.assert_array_equal(f,ff)
            test=moments(d,q,weights[group])
            ratios={"existing_2.5":test["numerator"]/(2.5*test["variance"]),
                    "calibrated":test["numerator"]/(value*test["variance"])}
            boot={"existing_2.5":test["bq"]/(2.5*test["bv"]),
                  "calibrated":test["bq"]/(m_draws[:,None,None]*test["bv"])}
            for family,selected in columns.items():
                for band,(start,stop) in BANDS.items():
                    mask=(f>=start)&((f<=stop) if band in ("whole","tail") else (f<stop))
                    for method,ratio in ratios.items():
                        r=ratio[mask][:,selected]
                        br=boot[method][:,mask][:,:,selected]
                        medians=np.median(br,axis=(1,2))
                        low,high=interval(medians)
                        summary.append(dict(mode=mode,group=group,method=method,family=family,band=band,
                                            records=len(d),M=2.5 if method=="existing_2.5" else value,
                                            median_predicted_over_observed=float(np.median(r)),ci95_low=float(low),ci95_high=float(high),
                                            median_factor_error=float(np.exp(np.median(abs(np.log(r))))),
                                            fraction_within_factor_two=float(np.mean((r>=.5)&(r<=2)))))
                    old=boot["existing_2.5"][:,mask][:,:,selected]
                    new=boot["calibrated"][:,mask][:,:,selected]
                    change=np.median(abs(np.log(new)),axis=(1,2))-np.median(abs(np.log(old)),axis=(1,2))
                    point=float(np.median(abs(np.log(ratios["calibrated"][mask][:,selected])))-
                                np.median(abs(np.log(ratios["existing_2.5"][mask][:,selected]))))
                    low,high=interval(change)
                    scores.append(dict(mode=mode,group=group,family=family,band=band,
                                       change_in_median_absolute_log_error=point,ci95_low=float(low),ci95_high=float(high)))
                if family!="all":
                    for method,ratio in ratios.items():
                        med=np.median(ratio[:,selected],axis=1)
                        low,high=interval(np.median(boot[method][:,:,selected],axis=2))
                        for i,frequency in enumerate(f):
                            frequency_rows.append(dict(mode=mode,group=group,method=method,family=family,
                                                       frequency_Hz=frequency,median_predicted_over_observed=med[i],
                                                       ci95_low=low[i],ci95_high=high[i]))
            print(f"Validated {mode} {group}: {len(d)} untouched records",flush=True)
            del test,d,q,ratios,boot
    output.mkdir(parents=True,exist_ok=True)
    for name,rows in [("fits",fits),("validation",summary),("frequencies",frequency_rows),("scores",scores)]:
        write_csv(output/f"m_calibration_{name}.csv",rows)
    (output/"m_calibration_protocol.json").write_text(json.dumps(protocol,indent=2)+"\n")
    plot(fits,summary,frequency_rows,output)


def plot(fits,summary,frequencies,output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter,LogLocator
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    mode="uniform6400"
    value=next(r["M"] for r in fits if r["mode"]==mode and r["family"]=="auto")
    methods=[("existing_2.5","#777777","Existing M=2.5"),("calibrated","#2864A5",f"Auto-calibrated M={float(value):.2f}")]
    families_plot=[("auto","Auto-spectra"),("real","Real cross-spectra"),("imag","Imaginary cross-spectra")]
    groups=["test_baseline","test_long_period","test_weak_waves"]
    fig,axes=plt.subplots(1,3,figsize=(14,5.2))
    for ax,(family,title) in zip(axes,families_plot):
        for method,colour,label in methods:
            rows=[next(r for r in summary if r["mode"]==mode and r["group"]==group and r["method"]==method and r["family"]==family and r["band"]=="whole") for group in groups]
            y=np.array([float(r["median_predicted_over_observed"]) for r in rows])
            low=np.array([float(r["ci95_low"]) for r in rows]);high=np.array([float(r["ci95_high"]) for r in rows])
            ax.plot(range(3),y,"o-",color=colour,label=label)
            ax.vlines(range(3),low,high,color=colour,alpha=.7)
        ax.axhspan(.9,1.1,color="#888888",alpha=.12)
        ax.axhline(1,color="#555555",ls="--",lw=1)
        ax.set_yscale("log")
        ax.yaxis.set_minor_locator(LogLocator(subs=(2,3,4,5,8)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x,pos:f"{x:g}"))
        ax.yaxis.set_minor_formatter(FuncFormatter(lambda x,pos:f"{x:g}"))
        ax.set_xticks(range(3),["Baseline","Long period","Weak waves"])
        ax.set(title=title,ylabel="Predicted / observed variance (log scale)")
        ax.grid(alpha=.15)
    axes[0].legend(frameon=False,fontsize=8)
    fig.suptitle("Held-out variance calibration: native-duration uniform 6,400 generator")
    fig.text(.04,.025,"One M fitted to baseline training auto-spectra only; never refitted to these held-out cases. Grey band: ±10%; target: one.\n"
             "Bars: 95% whole-record bootstrap intervals, including training-M uncertainty. 200 test records per case; existing variance formulas unchanged.",fontsize=8,color="#555555")
    fig.tight_layout(rect=(.02,.14,.99,.94))
    fig.savefig(output/"m_calibration_validation.png",dpi=170)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,5.2))
    for ax,(family,title) in zip(axes,families_plot):
        for method,colour,label in methods:
            rows=[r for r in frequencies if r["mode"]==mode and r["group"]=="test_baseline" and r["method"]==method and r["family"]==family]
            f=np.array([float(r["frequency_Hz"]) for r in rows])
            y=np.array([float(r["median_predicted_over_observed"]) for r in rows])
            low=np.array([float(r["ci95_low"]) for r in rows]);high=np.array([float(r["ci95_high"]) for r in rows])
            ax.plot(f,y,color=colour,label=label)
            ax.fill_between(f,low,high,color=colour,alpha=.15)
        ax.axhspan(.9,1.1,color="#888888",alpha=.12)
        ax.axhline(1,color="#555555",ls="--",lw=1)
        ax.set_yscale("log")
        ax.yaxis.set_minor_locator(LogLocator(subs=(2,3,4,5,8)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x,pos:f"{x:g}"))
        ax.yaxis.set_minor_formatter(FuncFormatter(lambda x,pos:f"{x:g}"))
        ax.set(title=title,xlabel="Frequency (Hz)",ylabel="Median predicted / observed variance (log scale)")
        ax.grid(alpha=.15)
    axes[0].legend(frameon=False,fontsize=8)
    fig.suptitle("Baseline held-out calibration by frequency")
    fig.text(.04,.025,"Shading: pointwise 95% paired whole-record bootstrap intervals, including calibration uncertainty; target ratio is one.\n"
             "Current diagonal variance formula is unchanged. 200 fresh training records and 200 separate baseline test records; no inverse fitting.",fontsize=8,color="#555555")
    fig.tight_layout(rect=(.02,.14,.99,.94))
    fig.savefig(output/"m_calibration_frequencies.png",dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=Path("variance_output/m_calibration"))
    parser.add_argument("--output",type=Path,default=ROOT)
    args=parser.parse_args()
    analyse(args.data,args.output)
