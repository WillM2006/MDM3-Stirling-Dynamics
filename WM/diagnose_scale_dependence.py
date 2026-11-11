"""Post-test diagnostic: exchange uncertainty scales, never the saved estimates."""
import csv
from pathlib import Path
import numpy as np
from variance_convergence import ROOT, write_csv


def paired_exchange(actual, truth, variance, draws):
    if len(actual)!=200 or np.any(variance<=0): raise ValueError('expected 200 paired records')
    original=(abs(actual-truth)<=1.959963984540054*np.sqrt(variance)).astype(float)
    donated=(abs(actual-truth)<=1.959963984540054*np.sqrt(np.roll(variance,100))).astype(float)
    pairs=(donated-original).reshape(2,100).mean(0)
    low,high=np.quantile(pairs[draws].mean(1),[.025,.975])
    return dict(original_coverage=original.mean(),donor_coverage=donated.mean(),
                coverage_change=donated.mean()-original.mean(),change_ci95_low=low,change_ci95_high=high,
                energy_sd_correlation=np.corrcoef(actual,np.sqrt(variance))[0,1])


def run():
    records=list(csv.DictReader((ROOT/'pooled_covariance_records.csv').open())); rows=[]
    for gi,group in enumerate(['baseline','long_period','weak_waves']):
        with np.load(Path('variance_output/correlation_transfer')/f'{group}-baseline.npz') as z: p=z['predictions']
        draws=np.random.default_rng(284200+gi).integers(0,100,(1000,100))
        for headings,index in [(12,0),(36,3)]:
            selected=sorted([r for r in records if r['group']==group and int(r['headings'])==headings
                and r['method']=='self_diagonal' and r['pool']=='none'],key=lambda r:int(r['seed']))
            actual,truth=[np.array([float(r[k]) for r in selected]) for k in ['m0','m0_true']]
            variance=p[:,1,index]
            result=paired_exchange(actual,truth,variance,draws)
            assert np.mean(np.roll(variance,100))==np.mean(variance) or np.isclose(np.mean(np.roll(variance,100)),np.mean(variance))
            rows.append(dict(group=group,headings=headings,**result))
    write_csv(ROOT/'correlation_scale_dependence.csv',rows)


if __name__=='__main__':
    run()
