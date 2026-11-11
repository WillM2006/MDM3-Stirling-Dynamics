"""Independent-pool stationary controls and transfer; no true-condition pool selection."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
from analyse_variance_convergence import bootstrap_moments,interval
from check_pooled_covariance import GROUPS,CALIBRATION
from variance_convergence import ROOT,write_csv

METRICS=["hs_abs_pct","hs_err_pct","spectrum_nrmse","direction_abs_deg","direction_l1",
         "negative_fraction","sd_Hs","m0_cover68","m0_cover95","spectrum_cover95","active_direction_cover95"]


def finite_bootstrap(values,weights):
    valid=np.isfinite(values)
    denominator=weights@valid
    draws=np.divide(weights@np.where(valid,values,0),denominator,
                    out=np.full((len(weights),values.shape[1]),np.nan),where=denominator>0)
    low=np.full(values.shape[1],np.nan);high=low.copy();mean=low.copy()
    for column in range(values.shape[1]):
        if valid[:,column].any():
            mean[column]=values[valid[:,column],column].mean()
            low[column],high[column]=np.nanquantile(draws[:,column],[.025,.975])
    return mean,low,high,valid.sum(0)


def analyse(data,output):
    protocol=json.loads((data/"protocol.json").read_text())
    records,summary,paired,probes,uncertainty=[],[],[],[],[]
    keys=[("none","self_diagonal")]+[(p,m) for p in CALIBRATION for m in ["pool_diagonal","pool_joint"]]
    global_prediction=None
    for gi,(group,settings) in enumerate(protocol["groups"].items()):
        observations=[];predicted=None
        for seed in range(settings["seed_start"],settings["seed_start"]+settings["records"]):
            with np.load(data/"results"/group/f"seed-{seed}.npz") as z:
                rows=json.loads(str(z["records"]))
                if len(rows)!=14 or any(r["seed"]!=seed or r["group"]!=group for r in rows):
                    raise ValueError("invalid paired result")
                for r in rows:r.setdefault("status","ok")
                records.extend(rows);observations.append(z["probe_observed"])
                if predicted is None:predicted=z["probe_predicted"]
                np.testing.assert_allclose(predicted,z["probe_predicted"],rtol=0,atol=0)
                if global_prediction is None:global_prediction=predicted
                np.testing.assert_allclose(global_prediction,z["probe_predicted"],rtol=0,atol=0)
        observations=np.stack(observations);n=len(observations)
        weights=np.random.default_rng(612300+gi).multinomial(n,np.full(n,1/n),size=1000)/n
        variance=observations.var(0,ddof=1)
        _,bv=bootstrap_moments(observations[:,None,:],weights)
        low,high=interval(predicted[None,:,:]/bv)
        for ki,(pool,method) in enumerate(keys[1:]):
            for pi in range(20):
                probes.append(dict(group=group,pool=pool,method=method,probe=pi,
                                   predicted_over_observed=float(predicted[ki,pi]/variance[pi]),
                                   ci95_low=float(low[ki,pi]),ci95_high=float(high[ki,pi])))
        for headings in protocol["headings"]:
            selected={key:sorted([r for r in records if r["group"]==group and r["headings"]==headings and
                                  (r["pool"],r["method"])==key],key=lambda r:r["seed"]) for key in keys}
            expected=list(range(settings["seed_start"],settings["seed_start"]+n))
            if any([r["seed"] for r in value]!=expected for value in selected.values()):
                raise ValueError("missing or unpaired inverse score")
            values={key:np.array([[r[m] for m in METRICS] for r in selected[key]]) for key in keys}
            for pool,method in keys:
                a=values[(pool,method)]
                mean,low,high,valid=finite_bootstrap(a,weights)
                failed=sum(r["status"]!="ok" for r in selected[(pool,method)])
                for mi,metric in enumerate(METRICS):
                    lo,hi=float(low[mi]),float(high[mi])
                    if metric in ["m0_cover68","m0_cover95"]:
                        lo,hi=binomtest(int(a[:,mi].sum()),n).proportion_ci(.95,method="exact")
                    summary.append(dict(group=group,pool=pool,method=method,headings=headings,metric=metric,records=n,
                                        valid_metric_records=int(valid[mi]),failed_recoveries=failed,
                                        mean=float(mean[mi]),ci95_low=float(lo),ci95_high=float(hi),
                                        q95=float(np.nanquantile(a[:,mi],.95)) if valid[mi] else float("nan"),
                                        maximum=float(np.nanmax(a[:,mi])) if valid[mi] else float("nan")))
                r=selected[(pool,method)]
                estimated=np.array([x["m0"] for x in r]);truth=np.array([x["m0_true"] for x in r])
                v=np.array([x["sd_m0"]**2 for x in r])
                uncertainty.append(dict(group=group,pool=pool,method=method,headings=headings,
                                        relative_energy_bias=float(np.mean(estimated/truth-1)),
                                        posterior_variance_over_repeat_variance=float(v.mean()/estimated.var(ddof=1)),
                                        posterior_variance_over_mse=float(v.mean()/np.mean((estimated-truth)**2))))
                if pool=="none":continue
                references=[("none","self_diagonal")]
                if method=="pool_joint":references.append((pool,"pool_diagonal"))
                for ref_pool,ref_method in references:
                    difference=a-values[(ref_pool,ref_method)]
                    difference_mean,lo,hi,difference_valid=finite_bootstrap(difference,weights)
                    for mi,metric in enumerate(METRICS):
                        paired.append(dict(group=group,pool=pool,method=method,headings=headings,metric=metric,
                                           reference_pool=ref_pool,reference_method=ref_method,
                                           valid_pairs=int(difference_valid[mi]),
                                           mean_change=float(difference_mean[mi]),ci95_low=float(lo[mi]),ci95_high=float(hi[mi])))
    output.mkdir(parents=True,exist_ok=True)
    for suffix,rows in [("records",records),("summary",summary),("paired",paired),("probes",probes),("uncertainty",uncertainty)]:
        write_csv(output/f"pooled_covariance_{suffix}.csv",rows)
    shrink=[]
    for pool in CALIBRATION:
        with np.load(data/f"pool-{pool}.npz") as z:
            for f,s in zip(z["f"],z["shrinkage"]):shrink.append(dict(pool=pool,frequency_Hz=float(f),intensity=float(s)))
    write_csv(output/"pooled_covariance_shrinkage.csv",shrink)
    (output/"pooled_covariance_protocol.json").write_text(json.dumps(protocol,indent=2)+"\n")
    plot(summary,probes,output)


def plot(summary,probes,output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    groups=list(GROUPS);pools=list(CALIBRATION)
    labels=["Baseline","Long period","Weak waves"]
    for metric,title,name,scale in [("hs_abs_pct","Mean absolute band Hs error (%)","errors",1),
                                   ("m0_cover95","Coverage of nominal 95% band-energy intervals (%)","coverage",100)]:
        fig,axes=plt.subplots(1,2,figsize=(11.8,5.6))
        matrix_all=[]
        for headings in [12,36]:
            matrix=np.array([[next(r for r in summary if r["headings"]==headings and r["pool"]==p and r["group"]==g and r["method"]=="pool_joint" and r["metric"]==metric)["mean"]*scale for g in groups] for p in pools])
            matrix_all.append(matrix)
        vmax=100 if metric=="m0_cover95" else max(a.max() for a in matrix_all)
        for ax,headings,matrix in zip(axes,[12,36],matrix_all):
            im=ax.imshow(matrix,vmin=0,vmax=vmax,cmap="viridis" if metric=="m0_cover95" else "magma_r")
            for i in range(3):
                for j in range(3):
                    text_colour="white" if ((metric=="m0_cover95" and matrix[i,j]<.55*vmax) or (metric!="m0_cover95" and matrix[i,j]>.5*vmax)) else "black"
                    entry=next(r for r in summary if r["headings"]==headings and r["pool"]==pools[i] and r["group"]==groups[j] and r["method"]=="pool_joint" and r["metric"]==metric)
                    annotation=f"{matrix[i,j]:.1f}" if matrix[i,j]<1000 else f"{matrix[i,j]:.2g}"
                    if entry["failed_recoveries"]:annotation+=f"\n{entry['failed_recoveries']} failed"
                    ax.text(j,i,annotation,ha="center",va="center",color=text_colour,fontsize=11)
            ax.set_xticks(range(3),labels);ax.set_yticks(range(3),labels)
            ax.set(xlabel="Test sea",ylabel="Fixed independent sensor pool",title=f"{headings} headings")
        fig.suptitle(title)
        fig.subplots_adjust(left=.09,right=.88,bottom=.23,top=.85,wspace=.45)
        fig.colorbar(im,cax=fig.add_axes([.92,.25,.018,.56]))
        fig.text(.04,.045,"Each pool is applied to every test sea; there is no true-condition pool selector. Diagonal cells are stationary-calibration controls, not an automatic deployment rule.\n"
                 "M=8.785 and tau=5 fixed; 200 new records per test sea. Nonpositive energy is a failure with 100% Hs error, not an excluded run. Coverage target: 95%.",fontsize=8,color="#555555")
        fig.savefig(output/f"pooled_covariance_transfer_{name}.png",dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,5.2))
    methods=[("self_diagonal","#888888","Same-record diagonal"),("pool_diagonal","#2864A5","Independent pooled diagonal"),("pool_joint","#21835D","Independent pooled joint")]
    for ax,metric,title,scale in zip(axes,["hs_abs_pct","direction_l1","m0_cover95"],
                                    ["Mean absolute band Hs error (%)","Mean signed direction-shape L1 distance","Coverage of nominal 95% intervals"],[1,1,1]):
        for method,colour,label in methods:
            r=[next(r for r in summary if r["headings"]==12 and r["group"]==g and r["method"]==method and r["pool"]==("none" if method=="self_diagonal" else g) and r["metric"]==metric) for g in groups]
            ax.plot(range(3),[a["mean"]*scale for a in r],"o-",color=colour,label=label)
            ax.vlines(range(3),[a["ci95_low"]*scale for a in r],[a["ci95_high"]*scale for a in r],color=colour)
        if metric=="m0_cover95":ax.axhline(.95,color="#555555",ls="--");ax.set_ylim(-.025,1.04)
        else:ax.set_ylim(bottom=0)
        ax.set_xticks(range(3),labels);ax.set(title=title);ax.grid(alpha=.15)
    axes[0].legend(frameon=False,fontsize=8)
    fig.suptitle("Stationary-calibration controls only: pool and test share simulated conditions")
    fig.text(.04,.025,"12 headings; M=8.785 and tau=5 fixed. This assumes an independently observed representative calibration pool; it does not select a pool for changing seas.\n"
             "Bars: 95% record-bootstrap/exact coverage intervals, conditional on fixed pools. Hs errors retain failures; direction metrics require positive energy (counts in CSV).",fontsize=8,color="#555555")
    fig.tight_layout(rect=(.01,.15,.99,.94));fig.savefig(output/"pooled_covariance_stationary.png",dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,5.3))
    for ax,pi,title in zip(axes,range(3),["Auto-entry sum","Real-cross-entry sum","Imaginary-cross-entry sum"]):
        matrix=np.array([[next(r for r in probes if r["pool"]==p and r["group"]==g and r["method"]=="pool_joint" and r["probe"]==pi)["predicted_over_observed"] for g in groups] for p in pools])
        im=ax.imshow(matrix,norm=LogNorm(vmin=.01,vmax=100),cmap="coolwarm")
        for i in range(3):
            for j in range(3):ax.text(j,i,f"{matrix[i,j]:.2g}",ha="center",va="center",fontsize=11)
        ax.set_xticks(range(3),labels);ax.set_yticks(range(3),labels)
        ax.set(title=title,xlabel="Test sea",ylabel="Fixed sensor pool")
    fig.suptitle("Sensor-only covariance transfer, before inverse interpretation")
    fig.subplots_adjust(left=.07,right=.87,bottom=.23,top=.83,wspace=.42)
    fig.colorbar(im,cax=fig.add_axes([.92,.25,.015,.54]),label="Predicted / observed variance; target one")
    fig.text(.04,.04,"Three fixed sums across retained frequencies; no response matrix or wave truth. These and 17 fixed random sensor projections are all saved.\n"
             "Pool/test matching is a controlled stationarity assumption, not a true-parameter lookup supplied to an estimator.",fontsize=8,color="#555555")
    fig.savefig(output/"pooled_covariance_probes.png",dpi=170);plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=Path("variance_output/pooled_covariance"))
    parser.add_argument("--output",type=Path,default=ROOT)
    args=parser.parse_args();analyse(args.data,args.output)
