"""Diagnostic counterfactuals only: no estimator, M, covariance rule or report changes."""
import csv
import json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest, skew
import correlated_spectral as cs
from check_correlation_transfer import rescale_factor
from check_cross_variance import corrected_variances, variance_terms
from check_fixed_estimator_scatter import digest
from variance_convergence import ROOT, write_csv

Z = 1.959963984540054
SCENARIOS = [(bias, scale, mode) for bias in [0,1] for scale in [0,1]
             for mode in ['paired','independent','constant']]


def tails(error, variance, mode):
    """Lower/upper misses; independent averages all distinct donor assignments."""
    n=len(error)
    if n<3 or not np.isfinite(error).all() or not np.isfinite(variance).all() or np.any(variance<=0):
        raise ValueError('expected finite errors and positive variances')
    width=Z*np.sqrt(variance)
    if mode=='constant': width=np.full(n,Z*np.sqrt(variance.mean()))
    lower=error < -width; upper=error > width
    if mode=='independent':
        ordered=np.sort(width)
        low=(np.searchsorted(ordered,np.maximum(-error,0),side='left').sum()-lower.sum())/(n*(n-1))
        high=(np.searchsorted(ordered,np.maximum(error,0),side='left').sum()-upper.sum())/(n*(n-1))
    elif mode in ['paired','constant']: low,high=lower.mean(),upper.mean()
    else: raise ValueError('unknown diagnostic mode')
    return np.array([1-low-high,low,high])


def counterfactuals(error, variance, reference_error, reference_variance):
    bias=reference_error.mean()
    factor=reference_error.var(ddof=1)/reference_variance.mean()
    if factor<=0: raise ValueError('degenerate independent reference scatter')
    return np.stack([tails(error-b*bias,variance*(factor if s else 1),mode)
                     for b,s,mode in SCENARIOS])


def shape_metrics(error, variance, reference_error, reference_variance):
    factor=reference_error.var(ddof=1)/reference_variance.mean()
    centered=(error-error.mean())/error.std(ddof=1)
    return np.array([reference_error.mean(),error.mean(),reference_error.std(ddof=1),error.std(ddof=1),
        factor,variance.mean()/error.var(ddof=1),np.corrcoef(error,np.sqrt(variance))[0,1],
        skew(error,bias=False),*np.quantile(centered,[.025,.975]),np.mean(abs(centered)<=Z)])


def run():
    cache=Path('variance_output/undercoverage_diagnosis'); cache.mkdir(parents=True,exist_ok=True)
    parent=json.loads((ROOT/'correlation_transfer_protocol.json').read_text())
    frozen=json.loads((ROOT/'fixed_estimator_protocol.json').read_text())
    sources=['inverse_recovery_records.csv','correlated_covariance_records.csv','pooled_covariance_records.csv']
    hashes={name:digest(ROOT/name) for name in sources}
    for name,value in frozen['source_hashes'].items():
        if digest(ROOT/name)!=value: raise ValueError('fixed source changed')
    protocol=dict(M=parent['M'],tau=parent['tau'],primary='12 headings, baseline correlation pool and reference gain; 36-heading sensitivity',
        reference='400 earlier saved corrected-diagonal outputs per condition: 230xxx and 240xxx cohorts; disjoint from 250xxx target cohort and 207xxx/209xxx/210xxx correlation pools',
        evaluation='200 existing 250xxx outputs per condition; retrospective, not untouched final validation',
        scenarios=SCENARIOS,reference_parameters='diagnostic bias and variance-ratio estimates from independent reference outputs; truth is used only for diagnosis, never to create a deployable correction',
        invariant='original saved estimates, covariance rules, M, prior, generator and report remain unchanged; no inverse solve',
        dependence='expected coverage over all n*(n-1) distinct error/variance donor pairings, not 39800 independent observations',
        constant_width='preserve mean target predicted variance while removing width fluctuations; ensemble diagnostic, not an available per-record estimator',
        shape='additional within-test centring/scaling diagnostic isolates marginal tail shape; not a fitted interval method or coverage validation',
        inference='1000 independent reference/test whole-record bootstrap draws, recomputing diagnostic parameters, donor means and moments; conditional on existing correlation pool and gains; pointwise intervals',
        interpretation='factorial conditional contrasts, not additive causal allocations; no coverage or variance multiplier chosen to pass a test',
        source_sha256=digest(__file__),score_hashes=hashes,parent_sha256=digest(ROOT/'correlation_transfer_protocol.json'))
    path=cache/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=protocol: raise ValueError('diagnostic specification changed')
    path.write_text(json.dumps(protocol,indent=2)+'\n')
    datasets=[list(csv.DictReader((ROOT/name).open())) for name in sources]
    all_rows=[]; moments=[]; contrasts=[]; inputs={}
    with np.load('variance_output/pooled_covariance/pool-baseline.npz') as z: factor=z['factors']; f=z['f']
    assert digest('variance_output/pooled_covariance/pool-baseline.npz')==parent['pool_hashes']['pool-baseline.npz']
    rho=cs.welch_frequency_correlation(np.rint(f*200).astype(int))
    names=['reference_bias','test_bias','reference_sd','test_sd','diagnostic_variance_factor',
           'candidate_variance_ratio','error_sd_correlation','error_skewness','shape_q025','shape_q975','shape_only_coverage']
    pairs=[('bias_only',0,6),('scale_only',0,3),('remove_pairing_only',0,1),('constant_width_only',0,2),
           ('bias_and_scale',0,9),('remove_pairing_after_bias_scale',9,10),
           ('remove_width_variability_after_independence_bias_scale',10,11),('all_diagnostics',0,11)]
    for gi,group in enumerate(['baseline','long_period','weak_waves']):
        gain_path=Path('variance_output/fixed_estimator_scatter')/f'{group}-projected.npz'
        inputs[str(gain_path)]=digest(gain_path)
        with np.load(gain_path) as z:gains=z['gains'][[0,3]]
        variances=[]
        for folder,method,records in zip(['inverse_recovery','correlated_covariance'],['coherence_corrected','diagonal'],datasets[:2]):
            selected=sorted([r for r in records if r['group']==group and r['headings']=='12' and r['method']==method],key=lambda r:int(r['seed']))
            assert len(selected)==200
            for row in selected:
                p=Path('variance_output')/folder/group/f'seed-{row["seed"]}.npz'
                inputs[str(p)]=digest(p)
                with np.load(p) as z:
                    np.testing.assert_array_equal(z['f'],f)
                    v=corrected_variances(*variance_terms(z['spectral']),parent['M'])
                variances.append(cs.propagated_variance(gains,rescale_factor(factor,v),rho))
        variances=np.stack(variances)
        candidate=Path('variance_output/correlation_transfer')/f'{group}-baseline.npz'
        inputs[str(candidate)]=digest(candidate)
        with np.load(candidate) as z:target_variances=z['predictions'][:,1,[0,3]]
        rng=np.random.default_rng(285100+gi)
        reference_draws=rng.integers(0,400,(1000,400)); test_draws=rng.integers(0,200,(1000,200))
        for hi,headings in enumerate([12,36]):
            ref=[]
            for records,method in zip(datasets[:2],['coherence_corrected','diagonal']):
                rr=sorted([r for r in records if r['group']==group and int(r['headings'])==headings and r['method']==method],key=lambda r:int(r['seed']))
                assert len(rr)==200;ref.extend(rr)
            target=sorted([r for r in datasets[2] if r['group']==group and int(r['headings'])==headings and r['method']=='self_diagonal'],key=lambda r:int(r['seed']))
            assert len(target)==200 and not set(r['seed'] for r in ref)&set(r['seed'] for r in target)
            tr=np.array([float(r['m0_true']) for r in ref]); tt=np.array([float(r['m0_true']) for r in target])
            er=np.array([float(r['m0']) for r in ref])/tr-1; et=np.array([float(r['m0']) for r in target])/tt-1
            vr=variances[:,hi]/tr**2; vt=target_variances[:,hi]/tt**2
            result=counterfactuals(et,vt,er,vr)
            boot=np.stack([counterfactuals(et[t],vt[t],er[r],vr[r]) for r,t in zip(reference_draws,test_draws)])
            low,high=np.quantile(boot,[.025,.975],axis=0)
            for i,(b,s,mode) in enumerate(SCENARIOS):
                row=dict(group=group,headings=headings,remove_reference_bias=b,reference_variance_scale=s,width_mode=mode)
                for k,name in enumerate(['coverage','lower_miss','upper_miss']):
                    row.update({name:result[i,k],name+'_ci_low':low[i,k],name+'_ci_high':high[i,k]})
                if i==0:
                    ci=binomtest(round(200*result[i,0]),200).proportion_ci(.95,method='exact')
                    row.update(coverage_ci_low=ci.low,coverage_ci_high=ci.high)
                all_rows.append(row)
            for name,i,j in pairs:
                lo,high_change=np.quantile(boot[:,j,0]-boot[:,i,0],[.025,.975])
                contrasts.append(dict(group=group,headings=headings,contrast=name,change=result[j,0]-result[i,0],ci95_low=lo,ci95_high=high_change))
            values=shape_metrics(et,vt,er,vr)
            bs=np.stack([shape_metrics(et[t],vt[t],er[r],vr[r]) for r,t in zip(reference_draws,test_draws)])
            lo,high=np.quantile(bs,[.025,.975],axis=0)
            for i,name in enumerate(names):moments.append(dict(group=group,headings=headings,metric=name,value=values[i],ci95_low=lo[i],ci95_high=high[i]))
        print(group,'diagnostics complete; no point estimates modified',flush=True)
    assert len(all_rows)==72 and len(contrasts)==48 and len(moments)==66
    for name,value in hashes.items():assert digest(ROOT/name)==value
    for name,rows in [('undercoverage_scenarios.csv',all_rows),('undercoverage_contrasts.csv',contrasts),('undercoverage_moments.csv',moments)]:write_csv(ROOT/name,rows)
    (ROOT/'undercoverage_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    (ROOT/'undercoverage_manifest.json').write_text(json.dumps(dict(inputs=inputs,score_hashes=hashes,
        scenarios=len(all_rows),contrasts=len(contrasts),moments=len(moments),source_sha256=digest(__file__)),indent=2)+'\n')


if __name__=='__main__':run()
