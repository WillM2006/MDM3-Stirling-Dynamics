"""Paired block dependence and variance-proxy stability, without inversion."""
import json
from pathlib import Path
import numpy as np
from check_block_information import CACHE,SCHEMES,theoretical_averages
from check_pooled_covariance import GROUPS
from check_fixed_estimator_scatter import digest
from variance_convergence import ROOT,write_csv
import invert_waves as iw


def moments(x):
    if x.ndim!=2 or not np.isfinite(x).all() or np.any(x<=0):raise ValueError('expected positive record-by-probe powers')
    mean=x.mean(0);var=x.var(0,ddof=1)
    return np.stack([mean**2/var,np.sqrt(var)/mean,(x*x).std(0,ddof=1)/(x*x).mean(0)])


def paired_stats(a,b):
    if a.shape!=b.shape:raise ValueError('matched block shapes required')
    ac=a-a.mean(0);bc=b-b.mean(0)
    corr=(ac*bc).sum(0)/np.sqrt((ac*ac).sum(0)*(bc*bc).sum(0))
    difference=100*(b.mean(0)-a.mean(0))/((a.mean(0)+b.mean(0))/2)
    return np.stack([corr,difference])


def analyse():
    protocol=json.loads((CACHE/'protocol.json').read_text())
    rule=dict(source_sha256=digest(__file__),generation_protocol_sha256=digest(CACHE/'protocol.json'),
        probes='each channel auto power at its peak frequency chosen from 400 independent reference spectra; each channel integrated auto power over all retained frequencies; 18 probes per scheme and condition',
        precision='effective averages and coefficient of variation of auto power and auto power squared, separately for full record and each block; squared auto is a variance-weight proxy, not calibrated block covariance',
        dependence='Pearson correlation between matched block probes; 95% interval inside +/-0.2 is practical weak-linear-dependence evidence for that probe only',
        resampling='1000 whole-record bootstrap draws, keeping both blocks paired; conditional on independently selected peak frequencies; pointwise, not simultaneous intervals',
        scope='no inverse, wave truth, M fitting or criterion based on recovery; few-segment instability considered alongside dependence; no universal numerical viability threshold asserted')
    p=CACHE/'analysis_protocol.json'
    if p.exists():assert json.loads(p.read_text())==rule
    p.write_text(json.dumps(rule,indent=2)+'\n')
    precision=[];dependence=[];inputs={};peaks=[]
    for gi,(group,start) in enumerate(GROUPS.items()):
        ref=[]
        for folder,first in [('inverse_recovery',230000+gi*1000),('correlated_covariance',240000+gi*1000)]:
            for seed in range(first,first+200):
                p=Path('variance_output')/folder/group/f'seed-{seed}.npz';inputs[str(p)]=digest(p)
                with np.load(p) as z:ref.append(z['spectral'].diagonal(axis1=-2,axis2=-1).real)
        mean=np.mean(ref,axis=0);peak=mean.argmax(0);channels=np.arange(9)
        values={'full':[],'halves_A':[],'halves_B':[],'gap100_A':[],'gap100_B':[]}
        for seed in range(start,start+200):
            p=CACHE/group/f'seed-{seed}.npz';inputs[str(p)]=digest(p)
            with np.load(p) as z:
                f=z['f'];matrices={'full':z['full']}
                for scheme in SCHEMES:
                    matrices[scheme+'_A']=z[scheme][0];matrices[scheme+'_B']=z[scheme][1]
                for name,s in matrices.items():
                    auto=s.diagonal(axis1=-2,axis2=-1).real
                    values[name].append(np.r_[auto[peak,channels],auto.sum(0)*(f[1]-f[0])])
        for c in channels:peaks.append(dict(group=group,channel=iw.CHANNELS[c],peak_frequency_hz=f[peak[c]]))
        values={k:np.stack(v) for k,v in values.items()}
        draws=np.random.default_rng(288100+gi).integers(0,200,(1000,200))
        labels=[('peak',c) for c in iw.CHANNELS]+[('band',c) for c in iw.CHANNELS]
        for name,x in values.items():
            result=moments(x);boot=np.stack([moments(x[d]) for d in draws]);lo,hi=np.quantile(boot,[.025,.975],axis=0)
            for mi,metric in enumerate(['effective_averages','power_cv','variance_proxy_cv']):
                for pi,(kind,c) in enumerate(labels):precision.append(dict(group=group,record=name,probe=kind,channel=c,metric=metric,value=result[mi,pi],ci95_low=lo[mi,pi],ci95_high=hi[mi,pi]))
        for scheme in SCHEMES:
            a,b=values[scheme+'_A'],values[scheme+'_B'];result=paired_stats(a,b)
            boot=np.stack([paired_stats(a[d],b[d]) for d in draws]);lo,hi=np.quantile(boot,[.025,.975],axis=0)
            for mi,metric in enumerate(['correlation','mean_difference_pct']):
                for pi,(kind,c) in enumerate(labels):dependence.append(dict(group=group,scheme=scheme,probe=kind,channel=c,metric=metric,value=result[mi,pi],ci95_low=lo[mi,pi],ci95_high=hi[mi,pi],inside_point2=bool(lo[mi,pi]>-.2 and hi[mi,pi]<.2) if mi==0 else ''))
        print('Analysed',group,flush=True)
    for name,rows in [('precision',precision),('dependence',dependence),('peaks',peaks)]:write_csv(ROOT/f'block_information_{name}.csv',rows)
    (ROOT/'block_information_analysis_protocol.json').write_text(json.dumps(rule,indent=2)+'\n')
    (ROOT/'block_information_manifest.json').write_text(json.dumps(dict(inputs=inputs,source_sha256=digest(__file__),generation_source_sha256=protocol['source_sha256'],no_inverse=True),indent=2)+'\n')


if __name__=='__main__':analyse()
