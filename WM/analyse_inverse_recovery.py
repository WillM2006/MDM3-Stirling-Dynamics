"""Paired recovery errors and empirical coverage; the inverse settings stay frozen."""
import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from analyse_variance_convergence import interval
from check_inverse_recovery import GROUPS,METHODS
from variance_convergence import ROOT,write_csv

METRICS=["hs_abs_pct","hs_err_pct","hs_full_err_pct","spectrum_nrmse","directional_nrmse",
         "direction_abs_deg","direction_l1","negative_fraction","sd_Hs","m0_cover68","m0_cover95",
         "spectrum_cover95","active_x_cover95","active_direction_cover95"]
COLOURS=["#888888","#2864A5","#21835D"]
LABELS=["Existing M=2.5","Calibrated M; equal Re/Im","Same M; corrected Re/Im"]


def analyse(source,data,output):
    with source.open() as stream:
        raw=list(csv.DictReader(stream))
    rows=[{k:(v if k in ("group","method") else float(v)) for k,v in row.items()} for row in raw]
    summary,changes,uncertainty,worst=[],[],[],[]
    for headings in [12,36]:
        for index,group in enumerate(GROUPS):
            selected={method:sorted([r for r in rows if r["headings"]==headings and r["group"]==group and r["method"]==method],key=lambda r:r["seed"]) for method in METHODS}
            n=len(selected[METHODS[0]])
            if n<8 or any([r["seed"] for r in value]!=[r["seed"] for r in selected[METHODS[0]]] for value in selected.values()):
                raise ValueError("unpaired or incomplete inverse records")
            weights=np.random.default_rng(31827+index).multinomial(n,np.full(n,1/n),size=1000)/n
            values={method:np.array([[r[metric] for metric in METRICS] for r in selected[method]]) for method in METHODS}
            for method in METHODS:
                if not np.isfinite(values[method]).all():
                    raise ValueError("nonfinite recovery metric")
                means=values[method].mean(0)
                low,high=interval(weights@values[method])
                for mi,metric in enumerate(METRICS):
                    lo,hi=float(low[mi]),float(high[mi])
                    if metric in ("m0_cover68","m0_cover95"):
                        lo,hi=binomtest(int(values[method][:,mi].sum()),n).proportion_ci(confidence_level=.95,method="exact")
                    summary.append(dict(headings=headings,group=group,method=method,metric=metric,records=n,
                                        mean=float(means[mi]),ci95_low=lo,ci95_high=hi,
                                        median=float(np.median(values[method][:,mi])),q05=float(np.quantile(values[method][:,mi],.05)),
                                        q95=float(np.quantile(values[method][:,mi],.95)),minimum=float(values[method][:,mi].min()),
                                        maximum=float(values[method][:,mi].max())))
                estimates=np.array([r["m0"] for r in selected[method]])
                truth=np.array([r["m0_true"] for r in selected[method]])
                posterior_var=np.array([r["sd_m0"]**2 for r in selected[method]])
                repeat_var=estimates.var(ddof=1)
                uncertainty.append(dict(headings=headings,group=group,method=method,records=n,
                                        relative_m0_bias=float(np.mean(estimates-truth)/truth.mean()),
                                        posterior_variance_over_repeat_variance=float(posterior_var.mean()/repeat_var),
                                        posterior_variance_over_mse=float(posterior_var.mean()/np.mean((estimates-truth)**2))))
                for metric in ["hs_abs_pct","spectrum_nrmse","direction_abs_deg","direction_l1"]:
                    worst.extend(dict(headings=headings,group=group,method=method,ranking_metric=metric,rank=rank,
                                      seed=int(r["seed"]),value=r[metric]) for rank,r in
                                 enumerate(sorted(selected[method],key=lambda r:r[metric],reverse=True)[:5],1))
            for reference in METHODS[:2]:
                difference=values["coherence_corrected"]-values[reference]
                low,high=interval(weights@difference)
                for mi,metric in enumerate(METRICS):
                    changes.append(dict(headings=headings,group=group,reference=reference,metric=metric,records=n,
                                        mean_change=float(difference[:,mi].mean()),ci95_low=float(low[mi]),ci95_high=float(high[mi])))
    output.mkdir(parents=True,exist_ok=True)
    for name,content in [("summary",summary),("paired",changes),("uncertainty",uncertainty),("worst",worst)]:
        write_csv(output/f"inverse_recovery_{name}.csv",content)
    plot(summary,output)
    plot_worst(rows,data,output)


def plot(summary,output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    conditions=list(GROUPS)
    panels=[("hs_abs_pct","Mean absolute band Hs error (%)",1),
            ("spectrum_nrmse","Mean spectrum NRMSE (%)",100),
            ("direction_abs_deg","Mean direction error (degrees)",1)]
    fig,axes=plt.subplots(2,3,figsize=(14,8.5))
    for ai,headings in enumerate([12,36]):
        for ax,(metric,title,scale) in zip(axes[ai],panels):
            for method,colour,label in zip(METHODS,COLOURS,LABELS):
                r=[next(r for r in summary if r["headings"]==headings and r["group"]==group and r["method"]==method and r["metric"]==metric) for group in conditions]
                ax.plot(range(3),[x["mean"]*scale for x in r],"o-",color=colour,label=label)
                ax.vlines(range(3),[x["ci95_low"]*scale for x in r],[x["ci95_high"]*scale for x in r],color=colour)
            ax.set_xticks(range(3),["Baseline","Long period","Weak waves"])
            ax.set(title=f"{headings} headings\n{title}")
            ax.set_ylim(bottom=0)
            ax.grid(alpha=.15)
    axes[0,0].legend(frameon=False,fontsize=8)
    fig.suptitle("Inverse recovery on 600 fresh native-duration simulations")
    fig.text(.035,.025,"Same spectra, response functions and prior (tau=5) across methods. Calibrated M=8.785 is frozen. Bars: 95% paired whole-record bootstrap intervals for means.\n"
             "12 headings: Will's batch default; 36: standalone configuration sensitivity. Truth uses retained-band energy; existing shared-physics assumptions remain.",fontsize=8,color="#555555")
    fig.tight_layout(rect=(.01,.10,.99,.95))
    fig.savefig(output/"inverse_recovery_errors.png",dpi=170)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5.2))
    for ax,headings in zip(axes,[12,36]):
        for mi,(method,colour,label) in enumerate(zip(METHODS,COLOURS,LABELS)):
            r=[next(r for r in summary if r["headings"]==headings and r["group"]==group and r["method"]==method and r["metric"]=="m0_cover95") for group in conditions]
            xx=np.arange(3)+(mi-1)*.07
            ax.plot(xx,[x["mean"] for x in r],"o-",color=colour,label=label)
            ax.vlines(xx,[x["ci95_low"] for x in r],[x["ci95_high"] for x in r],color=colour)
        ax.axhline(.95,color="#555555",ls="--",lw=1)
        ax.set_xticks(range(3),["Baseline","Long period","Weak waves"])
        ax.set(title=f"{headings} headings",ylabel="Fraction of records covering true band energy",ylim=(-.025,1.04))
        ax.grid(alpha=.15)
    axes[0].legend(frameon=False,fontsize=8)
    fig.suptitle("Do nominal 95% posterior intervals actually cover the true wave energy?")
    fig.text(.035,.025,"Coverage of linear band energy m0 ± 1.96 posterior SD; equivalent to transforming the interval to Hs. Bars: exact 95% binomial intervals over 200 records per case.\n"
             "The production covariance assumes independent spectral entries and frequencies. This test checks that assumption's practical consequence; no uncertainty inflation is fitted.",fontsize=8,color="#555555")
    fig.tight_layout(rect=(.01,.15,.99,.94))
    fig.savefig(output/"inverse_recovery_coverage.png",dpi=170)
    plt.close(fig)


def plot_worst(rows,data,output):
    import matplotlib.pyplot as plt
    worst=max([r for r in rows if r["headings"]==12 and r["method"]=="coherence_corrected"],key=lambda r:r["spectrum_nrmse"])
    fig,axes=plt.subplots(1,2,figsize=(12,5.2))
    for method,colour,label in zip(METHODS,COLOURS,LABELS):
        with np.load(data/"fits"/"12"/worst["group"]/f"seed-{int(worst['seed'])}_{method}.npz") as z:
            f,beta,x,truth=z["f"],z["beta"],z["x"],z["truth"]
            axes[0].plot(f,x.sum(1),color=colour,label=label)
            axes[1].plot(np.degrees(beta),x.sum(0)*(f[1]-f[0]),"o-",color=colour,label=label)
            if method=="coherence_corrected":
                sf=x.sum(1);sd=np.sqrt(z["variance_spectrum"])
                axes[0].fill_between(f,sf-1.96*sd,sf+1.96*sd,color=colour,alpha=.16)
                p=x.sum(0)*(f[1]-f[0]);pd=z["sd_direction_energy"]
                axes[1].vlines(np.degrees(beta),p-1.96*pd,p+1.96*pd,color=colour,alpha=.6)
    axes[0].plot(f,truth.sum(1),"k--",label="True band spectrum")
    axes[1].plot(np.degrees(beta),truth.sum(0)*(f[1]-f[0]),"k--",label="True direction-bin energy")
    axes[0].set(xlabel="Frequency (Hz)",ylabel="Wave spectrum (m²/Hz)")
    axes[1].set(xlabel="Heading (degrees)",ylabel="Band energy by direction (m²)")
    for ax in axes:
        ax.grid(alpha=.15)
    axes[0].legend(frameon=False,fontsize=8)
    fig.suptitle(f"Worst corrected frequency-spectrum recovery: {worst['group']}, seed {int(worst['seed'])}")
    fig.text(.035,.025,"Selected by maximum frequency-spectrum NRMSE across all 600 corrected, 12-heading results; the same record is shown for all methods.\n"
             "Green shading/bars: nominal pointwise 95% posterior intervals, without correction for omitted correlations or model bias.",fontsize=8,color="#555555")
    fig.tight_layout(rect=(.01,.15,.99,.94))
    fig.savefig(output/"inverse_recovery_worst.png",dpi=170)
    plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records",type=Path,default=ROOT/"inverse_recovery_records.csv")
    parser.add_argument("--data",type=Path,default=Path("variance_output/inverse_recovery"))
    parser.add_argument("--output",type=Path,default=ROOT)
    args=parser.parse_args()
    analyse(args.records,args.data,args.output)
