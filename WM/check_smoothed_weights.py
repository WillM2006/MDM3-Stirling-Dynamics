"""Sensor-selected variance smoothing; no wave-truth inputs to selection or fitting."""
import csv
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.stats import binomtest
import invert_waves as iw
from check_inverse_recovery import candidate_variances, operator_responses, truth_spectrum, recovery_metrics
from check_fixed_estimator_scatter import digest
from check_pooled_covariance import GROUPS
from analyse_pooled_covariance import finite_bootstrap
from variance_convergence import ROOT, write_csv

WIDTHS=[1,3,5,9,17]
METRICS=['hs_abs_pct','hs_err_pct','energy_bias_pct','spectrum_nrmse','direction_abs_deg','direction_l1','negative_fraction','m0_cover95','sd_m0']


def smooth_variances(v,width):
    if v.ndim!=2 or not np.isfinite(v).all() or np.any(v<=0) or width not in WIDTHS:
        raise ValueError('expected positive finite frequency-by-entry variances and allowed width')
    return uniform_filter1d(v,size=width,axis=0,mode='reflect') if width>1 else v.copy()


def sensor_selection(spectra,m):
    """Two-fold independent-record variance targets; no inverse or sea truth."""
    pairs=iw.stack_pairs(spectra.shape[-1])
    data=np.stack([[iw.data_vector(s,pairs) for s in record] for record in spectra])
    variances=np.stack([[candidate_variances(s,pairs,m) for s in record] for record in spectra])
    if len(data)!=200:raise ValueError('expected frozen 200-record training cohort')
    losses=np.empty((200,len(WIDTHS)))
    for fold in [0,1]:
        ids=np.arange(fold,200,2);other=np.arange(1-fold,200,2)
        target=data[other].var(axis=0,ddof=1)
        if np.any(target<=0):raise ValueError('degenerate reference variance')
        for wi,width in enumerate(WIDTHS):
            predicted=np.stack([smooth_variances(v,width) for v in variances[ids]])
            losses[ids,wi]=np.mean((predicted/target-1)**2,axis=(1,2))
    return WIDTHS[int(np.argmin(losses.mean(0)))],losses


def weighted_inverse(spectral,responses,variances,tau,scales):
    """SVD solve with unchanged observed data, prior scales and forward operator."""
    pairs=iw.stack_pairs(spectral.shape[-1]);xs=[];covs=[]
    if variances.shape!=(len(spectral),len(iw.data_vector(spectral[0],pairs))) or np.any(variances<=0) or not np.isfinite(variances).all():
        raise ValueError('invalid diagonal variances')
    for s,h,v,scale in zip(spectral,responses,variances,scales):
        b=iw.operator_matrix(h,pairs)*scale/np.sqrt(v[:,None]);d=iw.data_vector(s,pairs)/np.sqrt(v)
        u,values,vt=np.linalg.svd(b,full_matrices=False);inverse=1/(values**2+1/tau**2)
        xs.append(scale*(vt.T@((values*inverse)*(u.T@d))))
        covs.append(scale**2*(vt.T*inverse)@vt)
    return np.stack(xs),np.stack(covs)


def run():
    cache=Path('variance_output/smoothed_weights');cache.mkdir(parents=True,exist_ok=True)
    parent=json.loads((ROOT/'fixed_estimator_protocol.json').read_text())
    for name,value in parent['source_hashes'].items():assert digest(ROOT/name)==value
    m,tau=parent['M'],parent['tau'];score_path=ROOT/'pooled_covariance_records.csv';scores_hash=digest(score_path)
    protocol=dict(M=m,tau=tau,widths=WIDTHS,frequency_step_hz=.005,
        selection='two-fold alternating-seed baseline calibration 207000:207199; minimum mean squared relative error of variance predictions against other-fold empirical sensor spectral scatter; tie selects smaller width',
        candidate='reflect-boundary centred box smoothing of corrected marginal variance estimates only; keep data, M, forward response and original record prior scales unchanged; no reduction by smoothing width',
        evaluation='retrospective existing 250xxx records, 200 per condition; 12 headings primary and 36 sensitivity; all outcomes retained',
        decision='primary mean absolute band-height error must improve with paired 95% interval below zero in all three conditions; check signed energy/height bias, spectrum, direction, negative power and coverage for harm',
        uncertainty='1000 paired whole-record bootstrap draws conditional on chosen width and calibration; exact binomial coverage; diagonal posterior coverage descriptive, not a repaired uncertainty model',
        exclusions='no wave truth, inverse outputs, generator spectrum shape or true sea label used to select width or form weights; no new physical matching, mean correction or prior change',
        limits='adjacent Welch frequencies remain correlated; smoothing reduces dependence but does not create independence; finite training target and edge reflection may distort peaks',
        source_sha256=digest(__file__),saved_scores_sha256=scores_hash)
    p=cache/'protocol.json'
    if p.exists():assert json.loads(p.read_text())==protocol
    p.write_text(json.dumps(protocol,indent=2)+'\n')
    inputs={};training=[]
    for seed in range(207000,207200):
        p=Path('variance_output/m_calibration/uniform6400/train_baseline')/f'seed-{seed}.npz';inputs[str(p)]=digest(p)
        with np.load(p) as z:training.append(z['spectral']);f=z['f']
    np.testing.assert_allclose(np.diff(f),.005,rtol=0,atol=1e-12)
    width,loss=sensor_selection(np.stack(training),m)
    selection=dict(width=width,losses={str(w):float(loss[:,i].mean()) for i,w in enumerate(WIDTHS)},source_sha256=digest(__file__),training_hashes=inputs.copy())
    p=cache/'selection.json'
    if p.exists():assert json.loads(p.read_text())==selection
    p.write_text(json.dumps(selection,indent=2)+'\n')
    (ROOT/'smoothed_weights_selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    print('Sensor-only selection frozen:',width,selection['losses'],flush=True)
    saved=list(csv.DictReader(score_path.open()));records=[]
    for group,start in GROUPS.items():
        for headings in [12,36]:
            beta,h=operator_responses(Path('variance_output/m_calibration/hydrodynamics.npz'),f,headings)
            # Truth enters scoring only, after the sensor-only selection is frozen.
            truth,full_hs=truth_spectrum(group,f,beta);target=truth.sum()*(f[1]-f[0])
            for seed in range(start,start+200):
                p=Path('variance_output/pooled_covariance')/group/f'seed-{seed}.npz';inputs[str(p)]=digest(p)
                with np.load(p) as z:np.testing.assert_array_equal(f,z['f']);s=z['spectral']
                with patch.object(iw,'noise_variances',candidate_variances):x,cov,scales=iw.invert(f,s,h,tau,m)
                original=next(r for r in saved if r['group']==group and int(r['seed'])==seed and int(r['headings'])==headings and r['method']=='self_diagonal')
                np.testing.assert_allclose(x.sum()*(f[1]-f[0]),float(original['m0']),rtol=2e-7,atol=1e-12)
                v=np.stack([candidate_variances(ss,iw.stack_pairs(9),m) for ss in s])
                if seed==start:
                    xx,cc=weighted_inverse(s,h,v,tau,scales)
                    np.testing.assert_allclose(xx,x,rtol=1e-5,atol=1e-8)
                    np.testing.assert_allclose(cc,cov,rtol=1e-5,atol=1e-8)
                changed,changed_cov=weighted_inverse(s,h,smooth_variances(v,width),tau,scales)
                for method,estimate,covariance in [('original',x,cov),('smoothed',changed,changed_cov)]:
                    energy=estimate.sum()*(f[1]-f[0]);status='ok'
                    if energy>0:
                        metrics,_=recovery_metrics(estimate,covariance,truth,f,beta,full_hs)
                        metrics['energy_bias_pct']=100*(energy/target-1)
                        metrics={k:metrics[k] for k in METRICS}
                    else:
                        if not np.isfinite(estimate).all() or not np.isfinite(covariance).all():raise ValueError('nonfinite recovery')
                        sd=(f[1]-f[0])*np.sqrt(covariance.sum());status='nonpositive_energy'
                        metrics=dict(hs_abs_pct=100.,hs_err_pct=-100.,energy_bias_pct=100*(energy/target-1),spectrum_nrmse=np.linalg.norm(estimate.sum(1)-truth.sum(1))/np.linalg.norm(truth.sum(1)),direction_abs_deg=float('nan'),direction_l1=float('nan'),negative_fraction=np.maximum(-estimate,0).sum()/abs(estimate).sum(),m0_cover95=float(abs(energy-target)<=1.959963984540054*sd),sd_m0=sd)
                    records.append(dict(group=group,headings=headings,seed=seed,method=method,status=status,**metrics))
            print(group,headings,'complete',flush=True)
    summary=[];paired=[]
    for gi,group in enumerate(GROUPS):
        weights=np.random.default_rng(287100+gi).multinomial(200,np.full(200,1/200),size=1000)/200
        for headings in [12,36]:
            values={}
            for method in ['original','smoothed']:
                rr=[r for r in records if r['group']==group and r['headings']==headings and r['method']==method];assert len(rr)==200
                values[method]=np.array([[r[k] for k in METRICS] for r in rr])
                means,lo,hi,n=finite_bootstrap(values[method],weights)
                for i,metric in enumerate(METRICS):
                    if metric=='m0_cover95':lo[i],hi[i]=binomtest(round(means[i]*200),200).proportion_ci(.95,method='exact')
                    summary.append(dict(group=group,headings=headings,method=method,metric=metric,mean=means[i],ci95_low=lo[i],ci95_high=hi[i],valid=int(n[i]),failures=sum(r['status']!='ok' for r in rr)))
            means,lo,hi,n=finite_bootstrap(values['smoothed']-values['original'],weights)
            for i,metric in enumerate(METRICS):paired.append(dict(group=group,headings=headings,metric=metric,change=means[i],ci95_low=lo[i],ci95_high=hi[i],valid=int(n[i])))
    assert digest(score_path)==scores_hash
    for name,rows in [('records',records),('summary',summary),('paired',paired)]:write_csv(ROOT/f'smoothed_weights_{name}.csv',rows)
    (ROOT/'smoothed_weights_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    (ROOT/'smoothed_weights_manifest.json').write_text(json.dumps(dict(inputs=inputs,score_sha256=scores_hash,source_sha256=digest(__file__),records=len(records)),indent=2)+'\n')


if __name__=='__main__':run()
