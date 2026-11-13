"""Exact energy-bias decomposition; truth is used for diagnosis only."""
import csv
import json
from pathlib import Path
import numpy as np
import invert_waves as iw
from check_fixed_estimator_scatter import digest, energy_gain
from check_inverse_recovery import operator_responses, truth_spectrum, candidate_variances
from check_pooled_covariance import GROUPS
from variance_convergence import ROOT, write_csv


def decompose(cross, reference, model, truth):
    """Rows identify gain donors, columns identify independent data donors."""
    n=len(model)
    if cross.shape!=(n,n) or reference.shape[0]!=n or n<3 or truth<=0:
        raise ValueError('invalid donor dimensions or truth')
    paired=np.trace(cross)/n
    independent=(cross.sum()-np.trace(cross))/(n*(n-1))
    mean_model=model.mean(); mean_reference=reference.mean()
    terms=np.array([mean_model-truth,mean_reference-mean_model,
                    independent-mean_reference,paired-independent])/truth*100
    np.testing.assert_allclose(terms.sum(),100*(paired/truth-1),atol=1e-10)
    return np.r_[terms,100*(paired/truth-1),100*(independent/truth-1)]


def fixed_prior_gain(f, spectral, responses, m, tau, scales):
    pairs=iw.stack_pairs(9); gains=[]
    for s,h,scale in zip(spectral,responses,scales):
        a=iw.operator_matrix(h,pairs); v=candidate_variances(s,pairs,m)
        cov=np.linalg.inv(a.T@(a/v[:,None])+np.eye(len(h))/(tau*scale)**2)
        gains.append((f[1]-f[0])*(np.ones(len(h))@cov@a.T)/v)
    return np.stack(gains)


def run():
    cache=Path('variance_output/diagonal_bias');cache.mkdir(parents=True,exist_ok=True)
    parent=json.loads((ROOT/'fixed_estimator_protocol.json').read_text())
    for name,value in parent['source_hashes'].items():assert digest(ROOT/name)==value
    scores_path=ROOT/'pooled_covariance_records.csv';score_hash=digest(scores_path)
    protocol=dict(M=parent['M'],tau=parent['tau'],primary_headings=12,sensitivity_headings=36,
        test='existing 200 records per condition, 250xxx; saved point estimates unchanged',
        reference='400 independent sensor spectra per condition, 230xxx and 240xxx; no truth used in gains',
        decomposition=['model response/prior shrinkage','independent reference mean mismatch','test-reference mean/donor difference','same-record gain-data coupling'],
        identity='sum of four terms equals original mean energy bias; all distinct gain/data donor pairs, not independent replicates',
        sensitivity='freeze prior scales at independent reference spectral mean, retain recording-dependent noise weights; diagnostic only',
        inference='1000 whole-record bootstrap draws of matched test gain/data pairs and separate reference spectra; conditional on M and response operator; prior-frozen sensitivity conditional on reference scales',
        limits='retrospective diagnostic, not deployable correction or physical validation; mean mismatch does not identify its physical source; adaptive gain includes both variance weights and prior scale',
        source_sha256=digest(__file__),score_sha256=score_hash)
    pp=cache/'protocol.json'
    if pp.exists():assert json.loads(pp.read_text())==protocol
    pp.write_text(json.dumps(protocol,indent=2)+'\n')
    scores=list(csv.DictReader(scores_path.open()));rows=[];inputs={}
    for gi,(group,start) in enumerate(GROUPS.items()):
        spectra=[];reference=[]
        for folder,first,dest in [('pooled_covariance',start,spectra),('inverse_recovery',230000+gi*1000,reference),('correlated_covariance',240000+gi*1000,reference)]:
            for seed in range(first,first+200):
                p=Path('variance_output')/folder/group/f'seed-{seed}.npz';inputs[str(p)]=digest(p)
                with np.load(p) as z:
                    if spectra:np.testing.assert_array_equal(f,z['f'])
                    f=z['f'];dest.append(z['spectral'])
        spectra=np.stack(spectra);reference=np.stack(reference);pairs=iw.stack_pairs(9)
        obs=np.stack([[iw.data_vector(s,pairs) for s in record] for record in spectra]).reshape(200,-1)
        ref=np.stack([[iw.data_vector(s,pairs) for s in record] for record in reference]).reshape(400,-1)
        rng=np.random.default_rng(286100+gi)
        draws=rng.integers(0,200,(1000,200));rdraws=rng.integers(0,400,(1000,400))
        for headings in [12,36]:
            beta,h=operator_responses(Path('variance_output/m_calibration/hydrodynamics.npz'),f,headings)
            truth,_=truth_spectrum(group,f,beta);target=truth.sum()*(f[1]-f[0])
            model=np.einsum('fjc,fj,fjk->fck',h,truth,h.conj())
            model_d=np.stack([iw.data_vector(s,pairs) for s in model]).ravel()
            mean_ref=reference.mean(0)
            scales=np.array([np.mean([s[c,c].real/np.mean(abs(hh[:,c])**2) for c in range(9)]) for s,hh in zip(mean_ref,h)])
            for mode in ['original','fixed_prior']:
                path=cache/f'{group}-{headings}-{mode}.npz'
                if path.exists():
                    with np.load(path) as z:gains=z['gains']
                else:
                    gains=np.stack([energy_gain(f,s,h,parent['M'],parent['tau']) if mode=='original'
                        else fixed_prior_gain(f,s,h,parent['M'],parent['tau'],scales) for s in spectra]).reshape(200,-1)
                    np.savez_compressed(path,gains=gains)
                cross=gains@obs.T;refcross=gains@ref.T;model_result=gains@model_d
                if mode=='original':
                    saved=sorted([r for r in scores if r['group']==group and r['headings']==str(headings) and r['method']=='self_diagonal'],key=lambda r:int(r['seed']))
                    np.testing.assert_allclose(np.diag(cross),[float(r['m0']) for r in saved],rtol=2e-7,atol=1e-12)
                result=decompose(cross,refcross,model_result,target)
                boot=np.stack([decompose(cross[np.ix_(t,t)],refcross[np.ix_(t,r)],model_result[t],target) for t,r in zip(draws,rdraws)])
                lo,hi=np.quantile(boot,[.025,.975],axis=0)
                for i,name in enumerate(['model_prior','mean_mismatch','reference_sampling_difference','same_record_coupling','total_bias','independent_donor_bias']):
                    rows.append(dict(group=group,headings=headings,mode=mode,component=name,energy_pct=result[i],ci95_low=lo[i],ci95_high=hi[i]))
                print(group,headings,mode,np.round(result,3),flush=True)
    assert digest(scores_path)==score_hash
    write_csv(ROOT/'diagonal_bias.csv',rows)
    (ROOT/'diagonal_bias_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    (ROOT/'diagonal_bias_manifest.json').write_text(json.dumps(dict(inputs=inputs,source_sha256=digest(__file__),score_sha256=score_hash,rows=len(rows)),indent=2)+'\n')


if __name__=='__main__':run()
