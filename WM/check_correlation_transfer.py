"""Separate correlation transfer from variance-level transfer, without changing estimates."""
import csv
import json
from pathlib import Path
import numpy as np
import correlated_spectral as cs
from check_cross_variance import corrected_variances, variance_terms
from check_fixed_estimator_scatter import digest, summarise, scalar_bootstrap
from variance_convergence import ROOT, write_csv


def rescale_factor(factors, variances):
    """Keep donor correlations and replace only the diagonal variance levels."""
    norm = np.linalg.norm(factors, axis=-1)
    if (variances.shape != norm.shape or not np.isfinite(factors).all()
            or not np.isfinite(variances).all() or np.any(norm<=0) or np.any(variances<=0)):
        raise ValueError('expected matching finite factors and strictly positive variances')
    return factors/norm[..., None]*np.sqrt(variances[..., None])


def run():
    data = Path('variance_output/pooled_covariance')
    previous = Path('variance_output/fixed_estimator_scatter')
    cache = Path('variance_output/correlation_transfer')
    parent = json.loads((ROOT/'fixed_estimator_protocol.json').read_text())
    for name, value in parent['source_hashes'].items():
        if digest(ROOT/name)!=value: raise ValueError('previously fixed source changed')
    for name, value in parent['pool_hashes'].items():
        if digest(data/name)!=value: raise ValueError('previously fixed covariance pool changed')
    old_manifest = json.loads((ROOT/'fixed_estimator_manifest.json').read_text())
    for name, value in old_manifest['inputs'].items():
        if digest(name)!=value: raise ValueError('previous calibration/test record changed')
    score_path = ROOT/'pooled_covariance_records.csv'
    if digest(score_path)!=parent['saved_scores_sha256']: raise ValueError('saved estimates changed')
    protocol = dict(M=parent['M'], tau=parent['tau'], groups=parent['groups'], calibration=parent['calibration'],
        primary='12 headings, baseline-reference gain and baseline correlation pool fixed for all test conditions',
        sensitivity='all three gains, both heading grids, every correlation pool and test condition; no best-pool selection',
        candidate='row-rescale each frozen pooled factor to target-record corrected marginal variances; retain Hann frequency coupling',
        controls='previous absolute pooled covariance; within-frequency-only ablation; independent representative target-pool marginal levels as an explicitly idealised diagnostic',
        decision='variance agreement requires the 95% interval inside 0.9--1.1; inclusion of one is inconclusive; report uncentred 68/95 coverage and interval width, without fitting either',
        limitations='retrospective development test on previously examined data; conditional bootstrap excludes refitting M, gain and pool; no proof of physical or untouched final validation',
        invariants='all point estimates and biases unchanged; no simulation, inversion, M fit, correlation fit, prior change, selector or empirical inflation',
        bootstrap=1000, source_sha256=digest(__file__), parent_protocol_sha256=digest(ROOT/'fixed_estimator_protocol.json'),
        parent_manifest_sha256=digest(ROOT/'fixed_estimator_manifest.json'), score_sha256=digest(score_path),
        pool_hashes=parent['pool_hashes'])
    cache.mkdir(parents=True, exist_ok=True)
    frozen = cache/'protocol.json'
    if frozen.exists() and json.loads(frozen.read_text())!=protocol: raise ValueError('analysis rule changed')
    frozen.write_text(json.dumps(protocol, indent=2)+'\n')
    pools = {}
    for name in parent['calibration']:
        with np.load(data/f'pool-{name}.npz') as z: pools[name] = z['factors']
    scores = list(csv.DictReader(score_path.open()))
    old_rows = list(csv.DictReader((ROOT/'fixed_estimator_scatter.csv').open()))
    refs = [(h,p) for h in [12,36] for p in pools]
    rows = []; saved_hashes = {}; reference_gains = None
    for gi,(group,start) in enumerate(parent['groups'].items()):
        source = previous/f'{group}-projected.npz'
        saved_hashes[str(source)] = digest(source)
        with np.load(source) as z: gains=z['gains']; projected=z['projections']
        if reference_gains is None: reference_gains=gains.copy()
        np.testing.assert_array_equal(gains,reference_gains)
        marginals=[]; observations=[]
        for seed in range(start,start+200):
            with np.load(data/group/f'seed-{seed}.npz') as z:
                f=z['f']; marginals.append(corrected_variances(*variance_terms(z['spectral']),parent['M']))
                observations.append(z['observed'])
        np.testing.assert_allclose(projected,np.einsum('nfc,bfc->nb',np.stack(observations),gains),rtol=1e-12,atol=0)
        marginal=np.stack(marginals)
        rho=cs.welch_frequency_correlation(np.rint(f*200).astype(int)); identity=np.eye(len(f))
        target_variance=np.sum(pools[group]**2,axis=-1)
        draws=np.random.default_rng(283100+gi).integers(0,200,(1000,200))
        for pool,factor in pools.items():
            # This control assumes representative independent target calibration; it is not the candidate.
            ideal=rescale_factor(factor,target_variance)
            ideal_predictions=np.stack([cs.propagated_variance(gains,ideal,k) for k in [identity,rho]])
            predictions=np.stack([[cs.propagated_variance(gains,rescale_factor(factor,v),k)
                                   for k in [identity,rho]] for v in marginal])
            for ri,(headings,reference) in enumerate(refs):
                selected=sorted([r for r in scores if r['group']==group and int(r['headings'])==headings
                                 and r['method']=='self_diagonal' and r['pool']=='none'],key=lambda r:int(r['seed']))
                assert [int(r['seed']) for r in selected]==list(range(start,start+200))
                actual,truth=[np.array([float(r[k]) for r in selected]) for k in ['m0','m0_true']]
                for ki,coupling in enumerate(['within','joint']):
                    old=next(r for r in old_rows if r['group']==group and r['headings']==str(headings)
                        and r['gain_reference']==reference and r['covariance_pool']==pool and r['method']=='pooled_'+coupling)
                    for method,predicted in [('record_scaled',predictions[:,ki,ri]),('independent_target_scale',ideal_predictions[ki,ri])]:
                        stats=summarise(predicted,actual,projected[:,ri],truth,draws)
                        old_v=float(old['predicted_variance']); actual_b=scalar_bootstrap(actual,draws)
                        new_b=np.broadcast_to(predicted,actual.shape)[draws].mean(1)
                        change=abs(np.log(new_b/actual_b))-abs(np.log(old_v/actual_b))
                        lo,hi=np.quantile(change,[.025,.975])
                        rows.append(dict(group=group,headings=headings,gain_reference=reference,correlation_pool=pool,
                            method=method,coupling=coupling,records=200,**stats,
                            absolute_log_error_change=abs(np.log(stats['actual_ratio']))-abs(np.log(float(old['actual_ratio']))),
                            change_ci95_low=lo,change_ci95_high=hi))
            np.savez_compressed(cache/f'{group}-{pool}.npz',predictions=predictions,ideal=ideal_predictions)
            print(group,pool,'complete; original point estimates unchanged',flush=True)
    assert len(rows)==216
    assert len(set((r['group'],r['headings'],r['gain_reference'],r['correlation_pool'],r['method'],r['coupling']) for r in rows))==216
    assert digest(score_path)==parent['saved_scores_sha256']
    write_csv(ROOT/'correlation_transfer.csv',rows)
    (ROOT/'correlation_transfer_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    (ROOT/'correlation_transfer_manifest.json').write_text(json.dumps(dict(comparisons=len(rows),
        parent_inputs_verified=len(old_manifest['inputs']),reference_cache_hashes=saved_hashes,
        score_sha256=digest(score_path),source_sha256=digest(__file__),protocol_sha256=digest(frozen)),indent=2)+'\n')


if __name__=='__main__':
    run()
