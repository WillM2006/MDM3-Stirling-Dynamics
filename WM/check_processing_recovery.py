"""Exploratory wave recovery after the processing-only forward prediction test.

Reuse its 600 records; this is not a new untouched final recovery test.
Change only the channel processing response, keeping M, variances, tau and
recipient-record prior scales identical between the paired methods.
"""
import argparse
import csv
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scipy.stats import binomtest
import invert_waves as iw
from audit_spectral_mean import processing_gain
from check_inverse_recovery import operator_responses,candidate_variances,truth_spectrum,recovery_metrics
from check_processed_prediction import GROUPS,digest
from analyse_pooled_covariance import finite_bootstrap
from variance_convergence import ROOT,write_csv

METRICS=['hs_abs_pct','hs_err_pct','spectrum_nrmse','direction_abs_deg','direction_l1','negative_fraction','m0_cover95','sd_m0']


def fixed_prior_inverse(spectral,responses,m,tau,scales):
    """Diagonal Gaussian model, solved by SVD with externally frozen scales."""
    pairs=iw.stack_pairs(9);xs=[];covariances=[]
    for s,h,scale in zip(spectral,responses,scales):
        variance=candidate_variances(s,pairs,m)
        b=iw.operator_matrix(h,pairs)*scale/np.sqrt(variance[:,None])
        d=iw.data_vector(s,pairs)/np.sqrt(variance)
        u,values,vt=np.linalg.svd(b,full_matrices=False)
        inverse=1/(values**2+1/tau**2)
        xs.append(scale*(vt.T@((values*inverse)*(u.T@d))))
        covariances.append(scale**2*(vt.T*inverse)@vt)
    return np.stack(xs),np.stack(covariances)


def run(training,data,output):
    protocol=json.loads((data/'protocol.json').read_text())
    rule=dict(M=protocol['M'],tau=protocol['tau'],headings=[12,36],groups=GROUPS,
              methods=['uncorrected_response','filter_alignment_response'],
              change='multiply responses by known FIR/common-grid interpolation gain; no Welch-cell inverse or sensor-noise subtraction',
              invariant='same corrected diagonal variance formula, data and original recipient-record prior scales; no M/tau/physical-parameter fitting',
              status='exploratory recovery follow-up using the 600 records already examined for forward prediction; not an untouched final recovery test',
              selection='filter/alignment candidate was predefined before forward validation, selected for this follow-up after those mean results; no recovery outcomes used to tune it',
              primary='paired mean absolute band-Hs error change, with direction/spectrum/negative-power and coverage checks',
              source_sha256=digest(__file__),data_protocol_sha256=digest(data/'protocol.json'))
    path=data/'recovery_protocol.json'
    if path.exists() and json.loads(path.read_text())!=rule:raise ValueError('recovery rule changed')
    path.write_text(json.dumps(rule,indent=2)+'\n')
    records=[]
    for group,start in GROUPS.items():
        for headings in rule['headings']:
            responses=None
            for seed in range(start,start+200):
                with np.load(data/group/f'seed-{seed}.npz') as z:f=z['f'];spectral=z['spectral']
                if responses is None:
                    beta,responses=operator_responses(training/'hydrodynamics.npz',f,headings)
                    corrected=responses*processing_gain(f)[:,None,:]
                with patch.object(iw,'noise_variances',candidate_variances):
                    baseline,cov,scales=iw.invert(f,spectral,responses,rule['tau'],rule['M'])
                changed,changed_cov=fixed_prior_inverse(spectral,corrected,rule['M'],rule['tau'],scales)
                if seed==start:
                    check,check_cov=fixed_prior_inverse(spectral,responses,rule['M'],rule['tau'],scales)
                    assert np.linalg.norm(check-baseline)/max(np.linalg.norm(check),1e-30)<2e-6
                    assert np.linalg.norm(check_cov-cov)/np.linalg.norm(check_cov)<2e-6
                # No generating spectrum enters either fit.
                truth,full_hs=truth_spectrum(group,f,beta)
                for method,x,covariance in [('uncorrected_response',baseline,cov),('filter_alignment_response',changed,changed_cov)]:
                    df=float(f[1]-f[0]);m0=float(x.sum()*df);status='ok'
                    if m0>0:
                        metrics,_=recovery_metrics(x,covariance,truth,f,beta,full_hs)
                        metrics={key:metrics[key] for key in METRICS}
                    else:
                        if not np.isfinite(x).all() or not np.isfinite(covariance).all():raise ValueError('nonfinite fit')
                        sd=df*np.sqrt(covariance.sum());z=(m0-truth.sum()*df)/sd
                        metrics=dict(hs_abs_pct=100.,hs_err_pct=-100.,spectrum_nrmse=float(np.linalg.norm(x.sum(1)-truth.sum(1))/np.linalg.norm(truth.sum(1))),
                                     direction_abs_deg=float('nan'),direction_l1=float('nan'),negative_fraction=float(np.maximum(-x,0).sum()/abs(x).sum()),
                                     m0_cover95=float(abs(z)<=1.959963984540054),sd_m0=float(sd))
                        status='nonpositive_energy'
                    records.append(dict(group=group,headings=headings,seed=seed,method=method,status=status,**metrics))
            print('Recovery checked',group,headings,flush=True)
    summary=[];paired=[]
    for gi,group in enumerate(GROUPS):
        weights=np.random.default_rng(274100+gi).multinomial(200,np.full(200,1/200),size=1000)/200
        for headings in rule['headings']:
            values={}
            for method in rule['methods']:
                selected=sorted([r for r in records if r['group']==group and r['headings']==headings and r['method']==method],key=lambda r:r['seed'])
                assert len(selected)==200
                values[method]=np.array([[r[m] for m in METRICS] for r in selected])
                means,low,high,valid=finite_bootstrap(values[method],weights)
                for mi,metric in enumerate(METRICS):
                    lo,hi=low[mi],high[mi]
                    if metric=='m0_cover95':lo,hi=binomtest(int(values[method][:,mi].sum()),200).proportion_ci(.95,method='exact')
                    summary.append(dict(group=group,headings=headings,method=method,metric=metric,mean=float(means[mi]),ci95_low=float(lo),ci95_high=float(hi),
                                        valid_records=int(valid[mi]),failures=sum(r['status']!='ok' for r in selected)))
            means,low,high,valid=finite_bootstrap(values['filter_alignment_response']-values['uncorrected_response'],weights)
            for mi,metric in enumerate(METRICS):paired.append(dict(group=group,headings=headings,metric=metric,mean_change=float(means[mi]),ci95_low=float(low[mi]),ci95_high=float(high[mi]),valid_pairs=int(valid[mi])))
    for suffix,values in [('records',records),('summary',summary),('paired',paired)]:write_csv(output/f'processing_recovery_{suffix}.csv',values)
    (output/'processing_recovery_protocol.json').write_text(json.dumps(rule,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training',type=Path,default=Path('variance_output/m_calibration'))
    parser.add_argument('--data',type=Path,default=Path('variance_output/processed_prediction'))
    parser.add_argument('--output',type=Path,default=ROOT)
    args=parser.parse_args();run(args.training,args.data,args.output)
