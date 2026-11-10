"""Post-check diagnostic: propagate sensor covariance through frozen inverse gains.

This does not replace the estimator or fit M. Previous sensor-only records
estimate variability; fresh inverse records check that variance prediction.
"""
import argparse
import csv
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import invert_waves as iw
from check_inverse_recovery import METHODS,candidate_variances,operator_responses
from analyse_variance_convergence import bootstrap_moments,interval
from variance_convergence import ROOT,write_csv


def load(folder,start):
    d,s=[],[]
    f=None
    for seed in range(start,start+200):
        with np.load(folder/f"seed-{seed}.npz") as z:
            if f is None:
                f=z["f"]
            np.testing.assert_array_equal(f,z["f"])
            d.append(z["observed"]);s.append(z["spectral"])
    return f,np.stack(d),np.mean(s,axis=0)


def diagnose(training,data,output):
    protocol=json.loads((data/"protocol.json").read_text())
    parent=json.loads((training/"protocol.json").read_text())
    old_groups={"baseline":"train_baseline","long_period":"test_long_period","weak_waves":"test_weak_waves"}
    records=list(csv.DictReader((output/"inverse_recovery_records.csv").open()))
    summary=[]
    for headings in [12,36]:
        responses=None
        for index,(group,settings) in enumerate(protocol["groups"].items()):
            old=old_groups[group]
            f,training_d,reference=load(training/"uniform6400"/old,parent["groups"][old]["seed_start"])
            ff,test_d,_=load(data/group,settings["seed_start"])
            np.testing.assert_array_equal(f,ff)
            if responses is None:
                beta,responses=operator_responses(training/"hydrodynamics.npz",f,headings)
            df=float(f[1]-f[0]);pairs=iw.stack_pairs(9)
            for method in METHODS:
                m=2.5 if method=="existing_2.5" else protocol["M"]
                variance=candidate_variances if method=="coherence_corrected" else iw.noise_variances
                if method=="coherence_corrected":
                    with patch.object(iw,"noise_variances",candidate_variances):
                        x,cov,_=iw.invert(f,reference,responses,protocol["tau"],m)
                else:
                    x,cov,_=iw.invert(f,reference,responses,protocol["tau"],m)
                gains=[]
                for fi in range(len(f)):
                    a=iw.operator_matrix(responses[fi],pairs)
                    v=variance(reference[fi],pairs,m)
                    gains.append(df*(np.ones(headings)@cov[fi]@a.T)/v)
                gains=np.stack(gains)
                mean_d=training_d.mean(0)
                np.testing.assert_allclose(np.sum(gains*mean_d),x.sum()*df,rtol=2e-7,atol=1e-12)
                per_frequency=np.einsum("nfc,fc->nf",training_d,gains)
                projected_train=per_frequency.sum(1)
                projected_test=np.einsum("nfc,fc->n",test_d,gains)
                diagonal=float(np.sum(gains**2*training_d.var(0,ddof=1)))
                within=float(per_frequency.var(0,ddof=1).sum())
                full=float(projected_train.var(ddof=1))
                test_variance=float(projected_test.var(ddof=1))
                weights=np.random.default_rng(99187+index).multinomial(200,np.full(200,1/200),size=1000)/200
                test_weights=np.random.default_rng(99287+index).multinomial(200,np.full(200,1/200),size=1000)/200
                _,train_b=bootstrap_moments(projected_train[:,None,None],weights)
                _,test_b=bootstrap_moments(projected_test[:,None,None],test_weights)
                lo,hi=interval((train_b/test_b)[:,0,0])
                actual=np.array([float(r["m0"]) for r in records if int(float(r["headings"]))==headings and r["group"]==group and r["method"]==method])
                summary.append(dict(headings=headings,group=group,method=method,
                                    reference_posterior_variance=float(df**2*cov.sum()),
                                    diagonal_empirical_variance=diagonal,within_frequency_variance=within,
                                    full_training_variance=full,frozen_gain_test_variance=test_variance,
                                    original_nonlinear_test_variance=float(actual.var(ddof=1)),
                                    channel_correlation_factor=within/diagonal,frequency_correlation_factor=full/within,
                                    total_correlation_factor=full/diagonal,
                                    training_over_test_variance=full/test_variance,
                                    ratio_ci95_low=float(lo),ratio_ci95_high=float(hi)))
    write_csv(output/"inverse_recovery_covariance.csv",summary)
    plot(summary,output)


def plot(summary,output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows=[r for r in summary if r["headings"]==12 and r["method"]=="coherence_corrected"]
    fig,ax=plt.subplots(figsize=(10,5.5))
    for key,colour,label in [("reference_posterior_variance","#888888","Current posterior covariance"),
                             ("diagonal_empirical_variance","#CB7B31","Measured marginal variances only"),
                             ("within_frequency_variance","#2864A5","Include correlations within each frequency"),
                             ("full_training_variance","#21835D","Include correlations across frequencies too")]:
        y=[r[key]/r["frozen_gain_test_variance"] for r in rows]
        ax.plot(range(3),y,"o-",color=colour,label=label)
        if key=="full_training_variance":
            ax.vlines(range(3),[r["ratio_ci95_low"] for r in rows],[r["ratio_ci95_high"] for r in rows],color=colour,alpha=.6)
    ax.axhline(1,color="#555555",ls="--",lw=1)
    ax.set_xticks(range(3),["Baseline","Long period","Weak waves"])
    ax.set(ylabel="Predicted variance / held-out fixed-gain variance",ylim=(0,1.65),
           title="Diagnostic only: correlations explain much of the missing band-energy variance")
    ax.grid(alpha=.15)
    ax.legend(frameon=False,fontsize=8,loc="upper left")
    fig.text(.07,.025,"12-heading corrected inverse gains are frozen at earlier sensor-only mean spectra. Covariance comes from earlier records; fresh records test the prediction.\n"
             "Green bars: 95% bootstrap intervals, conditional on these gains. This is a variance diagnostic, not a fitted correction or a validated replacement likelihood.",fontsize=8,color="#555555")
    fig.tight_layout(rect=(.01,.15,.99,.96))
    fig.savefig(output/"inverse_recovery_covariance.png",dpi=170)
    plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training",type=Path,default=Path("variance_output/m_calibration"))
    parser.add_argument("--data",type=Path,default=Path("variance_output/inverse_recovery"))
    parser.add_argument("--output",type=Path,default=ROOT)
    args=parser.parse_args()
    diagnose(args.training,args.data,args.output)
