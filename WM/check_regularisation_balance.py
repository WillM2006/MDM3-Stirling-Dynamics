"""Algebraically fixed prior balance; no recovery tuning."""
import csv,json
from pathlib import Path
import numpy as np
import invert_waves as iw
from check_inverse_recovery import GROUPS,candidate_variances,operator_responses,truth_spectrum,recovery_metrics
from check_smoothed_weights import weighted_inverse
from check_fixed_estimator_scatter import digest
from variance_convergence import ROOT,write_csv

M=8.785033150237455
TAU=5*np.sqrt(2.5/M)
METRICS=['hs_abs_pct','hs_err_pct','direction_abs_deg','spectrum_nrmse','directional_nrmse','negative_fraction','m0_cover95']

def run():
    cache=Path('variance_output/regularisation_balance');cache.mkdir(parents=True,exist_ok=True)
    protocol=dict(M=M,tau=TAU,derivation='M*tau^2=2.5*5^2; fixed without recovery outcomes',
        data='existing 230xxx cohorts, 200 per condition; 12 headings primary, 36 sensitivity; retrospective mechanism diagnostic',
        invariant='original data-derived scales, response model, simulation, M and marginal variance formulas; only tau changes in corrected candidate',
        control='old equal-cross formula with new M and balance-preserving tau must reproduce old M=2.5 tau=5 point spectrum for every record; covariance scales by 2.5/M and need not reproduce old intervals',
        comparisons='original; corrected tau=5; corrected balance-preserving tau; all outcomes retained, no selection',
        inference='1000 paired whole-record bootstrap draws; pointwise intervals; no claim of fresh validation or optimal tau',
        source_sha256=digest(__file__),scores_sha256=digest(ROOT/'inverse_recovery_records.csv'))
    p=cache/'protocol.json'
    if p.exists():assert json.loads(p.read_text())==protocol
    p.write_text(json.dumps(protocol,indent=2)+'\n')
    saved=list(csv.DictReader((ROOT/'inverse_recovery_records.csv').open()));records=[];checks=[];inputs={}
    for gi,(g,start) in enumerate(GROUPS.items()):
        for headings in [12,36]:
            with np.load(Path('variance_output/inverse_recovery')/g/f'seed-{start}.npz') as z:f=z['f']
            beta,h=operator_responses(Path('variance_output/m_calibration/hydrodynamics.npz'),f,headings)
            truth,full_hs=truth_spectrum(g,f,beta)
            for seed in range(start,start+200):
                p=Path('variance_output/inverse_recovery')/g/f'seed-{seed}.npz';inputs[str(p)]=digest(p)
                with np.load(p) as z:s=z['spectral'];np.testing.assert_array_equal(f,z['f'])
                scales=np.array([np.mean([ss[c,c].real/np.mean(abs(hh[:,c])**2) for c in range(9)]) for ss,hh in zip(s,h)])
                pairs=iw.stack_pairs(9)
                v=np.stack([iw.noise_variances(ss,pairs,M) for ss in s])
                equivalent,eqcov=weighted_inverse(s,h,v,TAU,scales)
                p=Path('variance_output/inverse_recovery/fits')/str(headings)/g/f'seed-{seed}_existing_2.5.npz';inputs[str(p)]=digest(p)
                with np.load(p) as z:old=z['x']
                error=np.linalg.norm(equivalent-old)/np.linalg.norm(old)
                assert error<2e-6,error
                checks.append(dict(group=g,headings=headings,seed=seed,relative_spectrum_difference=error))
                corrected=np.stack([candidate_variances(ss,pairs,M) for ss in s])
                x,cov=weighted_inverse(s,h,corrected,TAU,scales)
                metrics,_=recovery_metrics(x,cov,truth,f,beta,full_hs)
                records.append(dict(group=g,headings=headings,seed=seed,method='corrected_balanced',**{k:metrics[k] for k in METRICS}))
                for method in ['existing_2.5','coherence_corrected']:
                    r=next(r for r in saved if r['group']==g and int(r['headings'])==headings and int(r['seed'])==seed and r['method']==method)
                    records.append(dict(group=g,headings=headings,seed=seed,method=method,**{k:float(r[k]) for k in METRICS}))
            print(g,headings,'complete',flush=True)
    summary=[];comparisons=[]
    for gi,g in enumerate(GROUPS):
        draws=np.random.default_rng(291100+gi).integers(0,200,(1000,200))
        for headings in [12,36]:
            vals={}
            for method in ['existing_2.5','coherence_corrected','corrected_balanced']:
                rr=sorted([r for r in records if r['group']==g and r['headings']==headings and r['method']==method],key=lambda r:r['seed']);assert len(rr)==200
                v=np.array([[r[k] for k in METRICS] for r in rr]);vals[method]=v
                lo,hi=np.quantile(v[draws].mean(1),[.025,.975],axis=0)
                for i,k in enumerate(METRICS):summary.append(dict(group=g,headings=headings,method=method,metric=k,mean=v[:,i].mean(),ci95_low=lo[i],ci95_high=hi[i]))
            for baseline in ['existing_2.5','coherence_corrected']:
                delta=vals['corrected_balanced']-vals[baseline];lo,hi=np.quantile(delta[draws].mean(1),[.025,.975],axis=0)
                for i,k in enumerate(METRICS):comparisons.append(dict(group=g,headings=headings,comparison='balanced_minus_'+baseline,metric=k,change=delta[:,i].mean(),ci95_low=lo[i],ci95_high=hi[i]))
    assert digest(ROOT/'inverse_recovery_records.csv')==protocol['scores_sha256']
    for name,rows in [('records',records),('summary',summary),('paired',comparisons),('equivalence',checks)]:write_csv(ROOT/f'regularisation_balance_{name}.csv',rows)
    (ROOT/'regularisation_balance_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    (ROOT/'regularisation_balance_manifest.json').write_text(json.dumps(dict(inputs=inputs,source_sha256=digest(__file__),max_equivalence_error=max(r['relative_spectrum_difference'] for r in checks)),indent=2)+'\n')

if __name__=='__main__':run()
