"""Evaluate fixed covariance rules without selecting them on inverse truth."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
from analyse_variance_convergence import bootstrap_moments,interval
from check_correlated_covariance import METHODS,GROUPS
from variance_convergence import ROOT,write_csv

METRICS=["hs_abs_pct","hs_err_pct","spectrum_nrmse","direction_abs_deg","direction_l1",
         "negative_fraction","sd_Hs","m0_cover68","m0_cover95","spectrum_cover95","active_direction_cover95"]
COLOURS=["#888888","#2864A5","#21835D"]
LABELS=["Diagonal; M=8.785","Correlations within each frequency","Joint correlations + Hann overlap"]


def analyse(data,output):
    protocol=json.loads((data/"protocol.json").read_text())
    records,summary,paired,probe_rows,shrinkage,uncertainty=[],[],[],[],[],[]
    for gi,(group,settings) in enumerate(protocol["groups"].items()):
        observed,predicted=[],[]
        for seed in range(settings["seed_start"],settings["seed_start"]+settings["records"]):
            with np.load(data/"results"/group/f"seed-{seed}.npz") as z:
                rows=json.loads(str(z["records"]))
                if len(rows)!=6 or any(r["seed"]!=seed or r["group"]!=group for r in rows):
                    raise ValueError("invalid paired result")
                records.extend(rows);observed.append(z["probe_observed"]);predicted.append(z["probe_predicted"])
                shrinkage.append(dict(group=group,seed=seed,mean=float(z["shrinkage"].mean()),
                                      minimum=float(z["shrinkage"].min()),maximum=float(z["shrinkage"].max())))
        observed=np.stack(observed);predicted=np.stack(predicted)
        n=len(observed)
        weights=np.random.default_rng(810230+gi).multinomial(n,np.full(n,1/n),size=1000)/n
        variance=observed.var(0,ddof=1)
        _,bv=bootstrap_moments(observed[:,None,:],weights)
        bpred=np.einsum("bn,nmp->bmp",weights,predicted)
        ratios=predicted.mean(0)/variance[None,:]
        lo,hi=interval(bpred/bv)
        for mi,method in enumerate(METHODS):
            for pi in range(observed.shape[1]):
                probe_rows.append(dict(group=group,method=method,probe=pi,
                                       description=["auto_sum","real_sum","imag_sum"][pi] if pi<3 else "fixed_random",
                                       predicted_over_observed=float(ratios[mi,pi]),ci95_low=float(lo[mi,pi]),ci95_high=float(hi[mi,pi])))
        for headings in protocol["headings"]:
            selected={m:sorted([r for r in records if r["group"]==group and r["headings"]==headings and r["method"]==m],key=lambda r:r["seed"]) for m in METHODS}
            if any(len(r)!=n or [x["seed"] for x in r]!=list(range(settings["seed_start"],settings["seed_start"]+n)) for r in selected.values()):
                raise ValueError("unpaired inverse results")
            values={m:np.array([[r[k] for k in METRICS] for r in selected[m]]) for m in METHODS}
            for method in METHODS:
                a=values[method]
                if not np.isfinite(a).all():raise ValueError("nonfinite score")
                low,high=interval(weights@a)
                for ki,metric in enumerate(METRICS):
                    l,h=float(low[ki]),float(high[ki])
                    if metric in ["m0_cover68","m0_cover95"]:
                        l,h=binomtest(int(a[:,ki].sum()),n).proportion_ci(.95,method="exact")
                    summary.append(dict(group=group,headings=headings,method=method,metric=metric,records=n,
                                        mean=float(a[:,ki].mean()),ci95_low=float(l),ci95_high=float(h),
                                        median=float(np.median(a[:,ki])),q95=float(np.quantile(a[:,ki],.95)),maximum=float(a[:,ki].max())))
                m0=np.array([r["m0"] for r in selected[method]])
                truth=np.array([r["m0_true"] for r in selected[method]])
                v=np.array([r["sd_m0"]**2 for r in selected[method]])
                uncertainty.append(dict(group=group,headings=headings,method=method,
                                        relative_energy_bias=float(np.mean(m0/truth-1)),
                                        posterior_variance_over_repeat_variance=float(v.mean()/m0.var(ddof=1)),
                                        posterior_variance_over_mse=float(v.mean()/np.mean((m0-truth)**2))))
            for method,reference in [("same_frequency","diagonal"),("joint_hann","diagonal"),("joint_hann","same_frequency")]:
                difference=values[method]-values[reference]
                low,high=interval(weights@difference)
                for ki,metric in enumerate(METRICS):
                    paired.append(dict(group=group,headings=headings,method=method,reference=reference,metric=metric,
                                       mean_change=float(difference[:,ki].mean()),ci95_low=float(low[ki]),ci95_high=float(high[ki])))
    output.mkdir(parents=True,exist_ok=True)
    for suffix,rows in [("records",records),("summary",summary),("paired",paired),("probes",probe_rows),
                        ("shrinkage",shrinkage),("uncertainty",uncertainty)]:
        write_csv(output/f"correlated_covariance_{suffix}.csv",rows)
    (output/"correlated_covariance_protocol.json").write_text(json.dumps(protocol,indent=2)+"\n")
    plot(summary,probe_rows,output)
    worst_plot(records,data,output)


def plot(summary,probes,output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    groups=list(GROUPS)
    fig,axes=plt.subplots(1,2,figsize=(12,5.2))
    for ax,headings in zip(axes,[12,36]):
        for mi,(method,colour,label) in enumerate(zip(METHODS,COLOURS,LABELS)):
            r=[next(r for r in summary if r["group"]==g and r["headings"]==headings and r["method"]==method and r["metric"]=="m0_cover95") for g in groups]
            xx=np.arange(3)+(mi-1)*.06
            ax.plot(xx,[a["mean"] for a in r],"o-",color=colour,label=label)
            ax.vlines(xx,[a["ci95_low"] for a in r],[a["ci95_high"] for a in r],color=colour)
        ax.axhline(.95,color="#555555",ls="--")
        ax.set_xticks(range(3),["Baseline","Long period","Weak waves"])
        ax.set(title=f"{headings} headings",ylabel="Fraction covering true band energy",ylim=(-.025,1.04))
        ax.grid(alpha=.15)
    axes[0].legend(frameon=False,fontsize=8)
    fig.suptitle("Fixed M and prior: do correlated-covariance intervals calibrate?")
    fig.text(.04,.025,"600 fresh records; M=8.785 and tau=5 held fixed. Covariance uses measured sensor spectra and Welch settings only. No tuning to recovery truth.\n"
             "Nominal 95% intervals for band energy, including posterior correlations across frequencies in the joint method. Bars: exact 95% binomial intervals.",fontsize=8,color="#555555")
    fig.tight_layout(rect=(.01,.15,.99,.94));fig.savefig(output/"correlated_covariance_coverage.png",dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(14,8.3))
    for ai,headings in enumerate([12,36]):
        for ax,metric,title,scale in zip(axes[ai],["hs_abs_pct","spectrum_nrmse","direction_l1"],
                                        ["Mean absolute band Hs error (%)","Mean spectrum NRMSE (%)","Mean signed direction-shape L1 distance"],[1,100,1]):
            for method,colour,label in zip(METHODS,COLOURS,LABELS):
                r=[next(r for r in summary if r["group"]==g and r["headings"]==headings and r["method"]==method and r["metric"]==metric) for g in groups]
                ax.plot(range(3),[a["mean"]*scale for a in r],"o-",color=colour,label=label)
                ax.vlines(range(3),[a["ci95_low"]*scale for a in r],[a["ci95_high"]*scale for a in r],color=colour)
            ax.set_xticks(range(3),["Baseline","Long period","Weak waves"])
            ax.set(title=f"{headings} headings\n{title}");ax.set_ylim(bottom=0);ax.grid(alpha=.15)
    axes[0,0].legend(frameon=False,fontsize=8)
    fig.suptitle("Recovery with the same M, prior, sensors and fresh records")
    fig.text(.04,.025,"Bars: 95% whole-record bootstrap intervals for means. M=8.785; tau=5. Signed direction-shape distance can exceed one because negative powers are allowed.\n"
             "A covariance change is not accepted solely because its intervals widen; point recovery and sensor-only variance predictions are checked too.",fontsize=8,color="#555555")
    fig.tight_layout(rect=(.01,.10,.99,.94));fig.savefig(output/"correlated_covariance_errors.png",dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,5.2))
    for ax,pi,title in zip(axes,range(3),["Sum of auto entries","Sum of real cross entries","Sum of imaginary cross entries"]):
        for method,colour,label in zip(METHODS,COLOURS,LABELS):
            r=[next(r for r in probes if r["group"]==g and r["method"]==method and r["probe"]==pi) for g in groups]
            ax.plot(range(3),[a["predicted_over_observed"] for a in r],"o-",color=colour,label=label)
            ax.vlines(range(3),[a["ci95_low"] for a in r],[a["ci95_high"] for a in r],color=colour)
        ax.axhline(1,color="#555555",ls="--");ax.set_yscale("log")
        ax.set_xticks(range(3),["Baseline","Long period","Weak waves"])
        ax.set(title=title,ylabel="Predicted / observed sensor-projection variance");ax.grid(alpha=.15)
    axes[0].legend(frameon=False,fontsize=8)
    fig.suptitle("Sensor-only covariance check: fixed projections across retained frequencies")
    fig.text(.04,.025,"Projection units fixed from earlier baseline sensor records; no wave truth, transfer functions or recovered spectra are used. Target: one.\n"
             "Bars: 95% whole-record bootstrap intervals. These three sums and 17 additional fixed random projections are all reported in the probes CSV.",fontsize=8,color="#555555")
    fig.tight_layout(rect=(.01,.15,.99,.94));fig.savefig(output/"correlated_covariance_probes.png",dpi=170);plt.close(fig)


def worst_plot(records,data,output):
    import matplotlib.pyplot as plt
    from check_inverse_recovery import truth_spectrum
    worst=max((r for r in records if r["headings"]==12 and r["method"]=="joint_hann"),key=lambda r:r["hs_abs_pct"])
    fig,ax=plt.subplots(figsize=(10,5.2))
    with np.load(data/"results"/worst["group"]/f"seed-{worst['seed']}.npz") as z:
        f=z["f"]
        for method,colour,label in zip(METHODS,COLOURS,LABELS):
            sf=z[f"x_12_{method}"].sum(1)
            ax.plot(f,sf,color=colour,label=label)
            if method=="joint_hann":
                sd=np.sqrt(z[f"variance_12_{method}"][:len(f)])
                ax.fill_between(f,sf-1.96*sd,sf+1.96*sd,color=colour,alpha=.15)
    beta=np.linspace(-np.pi,np.pi,12,endpoint=False)
    truth,_=truth_spectrum(worst["group"],f,beta)
    ax.plot(f,truth.sum(1),"k--",label="True retained-band spectrum")
    ax.set(xlabel="Frequency (Hz)",ylabel="Wave spectrum (m²/Hz)",
           title=f"Worst joint-method band Hs error: {worst['group']}, seed {worst['seed']}")
    ax.legend(frameon=False,fontsize=8);ax.grid(alpha=.15)
    fig.text(.05,.025,"Selected by maximum absolute Hs error among 600 primary 12-heading joint-method results. All methods shown on the same recording.\n"
             "Green shading: nominal pointwise 95% posterior intervals. Selection is diagnostic and is not used to adjust the estimator.",fontsize=8,color="#555555")
    fig.tight_layout(rect=(.01,.15,.99,.96));fig.savefig(output/"correlated_covariance_worst.png",dpi=170);plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=Path("variance_output/correlated_covariance"))
    parser.add_argument("--output",type=Path,default=ROOT)
    args=parser.parse_args();analyse(args.data,args.output)
