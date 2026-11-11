"""Post-validation reference-prior decomposition, not calibration or recovery evidence.

True x is used only to evaluate the recovery operator's expected mean and
separate prior shrinkage from observed mean-response mismatch. No fitting.
"""
import argparse
import csv
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import invert_waves as iw
import pooled_spectral as ps
import correlated_spectral as cs
from check_inverse_recovery import operator_responses,candidate_variances,truth_spectrum
from variance_convergence import ROOT,write_csv


def diagnose(training,data,output):
    protocol=json.loads((data/"protocol.json").read_text())
    records=list(csv.DictReader((output/"pooled_covariance_records.csv").open()))
    rows=[]
    for group,settings in protocol["groups"].items():
        calibration=protocol["calibration"][group]
        observed=[];reference=[]
        for seed in range(settings["seed_start"],settings["seed_start"]+200):
            with np.load(data/group/f"seed-{seed}.npz") as z:
                f=z["f"];observed.append(z["spectral"])
        for seed in range(calibration["seed_start"],calibration["seed_start"]+200):
            with np.load(training/"uniform6400"/calibration["folder"]/f"seed-{seed}.npz") as z:
                np.testing.assert_array_equal(f,z["f"]);reference.append(z["spectral"])
        observed=np.mean(observed,axis=0);reference=np.mean(reference,axis=0)
        with np.load(data/f"pool-{group}.npz") as z:factors=z["factors"]
        rho=cs.welch_frequency_correlation(np.rint(f*iw.SEG).astype(int));df=float(f[1]-f[0])
        for headings in [12,36]:
            beta,h=operator_responses(training/"hydrodynamics.npz",f,headings)
            truth,_=truth_spectrum(group,f,beta)
            model=np.einsum("fjc,fj,fjk->fck",h,truth,h.conj())
            with patch.object(iw,"noise_variances",candidate_variances):
                _,_,scales=iw.invert(f,reference,h,protocol["tau"],protocol["M"])
            x_model,_,_=ps.pooled_inverse(f,model,h,protocol["tau"],factors,rho,scales)
            x_observed,_,_=ps.pooled_inverse(f,observed,h,protocol["tau"],factors,rho,scales)
            x_residual,_,_=ps.pooled_inverse(f,observed-model,h,protocol["tau"],factors,rho,scales)
            np.testing.assert_allclose(x_model+x_residual,x_observed,rtol=1e-7,atol=1e-9)
            target=truth.sum()*df;pure=x_model.sum()*df;mean=x_observed.sum()*df
            actual=np.mean([float(r["m0"]) for r in records if r["group"]==group and r["pool"]==group and r["method"]=="pool_joint" and int(r["headings"])==headings])
            rows.append(dict(group=group,headings=headings,true_band_energy=float(target),
                             model_mean_recovery=float(pure),observed_mean_recovery=float(mean),
                             actual_mean_recovery=float(actual),prior_energy_bias_pct=float(100*(pure/target-1)),
                             mean_mismatch_effect_pct=float(100*(mean-pure)/target),
                             reference_energy_bias_pct=float(100*(mean/target-1)),
                             actual_energy_bias_pct=float(100*(actual/target-1))))
    write_csv(output/"pooled_covariance_mean_diagnostic.csv",rows)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training",type=Path,default=Path("variance_output/m_calibration"))
    parser.add_argument("--data",type=Path,default=Path("variance_output/pooled_covariance"))
    parser.add_argument("--output",type=Path,default=ROOT)
    args=parser.parse_args();diagnose(args.training,args.data,args.output)
