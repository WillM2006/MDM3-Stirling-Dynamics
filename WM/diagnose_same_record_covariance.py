"""Post-validation diagnostic, not a deployable covariance rule or fresh validation.

Pair baseline recordings 100 seeds apart. Exchange covariance estimates while
retaining each recipient's data, operators and prior scales. No tuning occurs.
"""
import argparse
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import invert_waves as iw
import correlated_spectral as cs
from check_correlated_covariance import score
from check_inverse_recovery import operator_responses,candidate_variances,truth_spectrum
from analyse_variance_convergence import interval
from variance_convergence import ROOT,write_csv


def diagnose(training,data,output):
    protocol=json.loads((data/"protocol.json").read_text())
    settings=protocol["groups"]["baseline"]
    if settings["records"]!=200:raise ValueError("diagnostic expects 100 reciprocal pairs")
    donors=(np.arange(200)+100)%200
    np.testing.assert_array_equal(np.sort(donors),np.arange(200))
    if np.any(donors==np.arange(200)):raise ValueError("covariance donor must be a different recording")
    start=settings["seed_start"]
    spectra=[];original=[]
    for seed in range(start,start+200):
        with np.load(data/"baseline"/f"seed-{seed}.npz") as z:
            f=z["f"];spectra.append(z["spectral"])
        with np.load(data/"results"/"baseline"/f"seed-{seed}.npz") as z:
            original.append({r["method"]:r for r in json.loads(str(z["records"])) if r["headings"]==12})
    beta,h=operator_responses(training/"hydrodynamics.npz",f,12)
    rho=cs.welch_frequency_correlation(np.rint(f*iw.SEG).astype(int))
    rows=[];donor_scores={m:[] for m in ["same_frequency","joint_hann"]}
    for index,s in enumerate(spectra):
        donor=int(donors[index])
        factors,_=cs.covariance_factors(spectra[donor],protocol["M"])
        with patch.object(iw,"noise_variances",candidate_variances):
            _,_,scales=iw.invert(f,s,h,protocol["tau"],protocol["M"])
        posterior=[]
        for method,r in [("same_frequency",np.eye(len(f))),("joint_hann",rho)]:
            x,covariance,residual=cs.correlated_inverse(f,s,h,protocol["tau"],protocol["M"],factors,r,scales)
            posterior.append((method,x,covariance))
        truth,_=truth_spectrum("baseline",f,beta)
        for method,x,covariance in posterior:
            metrics=score(x,covariance,truth,f,beta)
            donor_scores[method].append(metrics)
            rows.append(dict(seed=start+index,donor_seed=start+donor,method=method,
                             original_hs_bias=original[index][method]["hs_err_pct"],donor_hs_bias=metrics["hs_err_pct"],
                             original_hs_abs=original[index][method]["hs_abs_pct"],donor_hs_abs=metrics["hs_abs_pct"],
                             original_cover95=original[index][method]["m0_cover95"],donor_cover95=metrics["m0_cover95"]))
    summary=[]
    weights=np.random.default_rng(39871).multinomial(100,np.full(100,.01),size=1000)/100
    for method in donor_scores:
        for metric in ["hs_err_pct","hs_abs_pct","m0_cover95"]:
            a=np.array([r[metric] for r in donor_scores[method]])
            b=np.array([r[method][metric] for r in original])
            pair_change=((a-b)[:100]+(a-b)[100:])/2
            lo,hi=interval(weights@pair_change)
            summary.append(dict(method=method,metric=metric,original_mean=float(b.mean()),donor_mean=float(a.mean()),
                                change=float((a-b).mean()),change_ci95_low=float(lo),change_ci95_high=float(hi),independent_pairs=100))
    write_csv(output/"correlated_covariance_donor_records.csv",rows)
    write_csv(output/"correlated_covariance_donor_summary.csv",summary)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training",type=Path,default=Path("variance_output/m_calibration"))
    parser.add_argument("--data",type=Path,default=Path("variance_output/correlated_covariance"))
    parser.add_argument("--output",type=Path,default=ROOT)
    args=parser.parse_args();diagnose(args.training,args.data,args.output)
