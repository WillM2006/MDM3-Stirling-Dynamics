"""Per-frequency variance-weight stability across the nine channel peaks."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent
rows=list(csv.DictReader((root/'block_information_precision.csv').open()))
fig,axes=plt.subplots(1,3,figsize=(12,5),sharey=True)
names=['full','halves_A','halves_B','gap100_A','gap100_B']
for ax,group,title in zip(axes,['baseline','long_period','weak_waves'],['Baseline','Long period','Weak waves']):
    vals=[[100*float(r['value']) for r in rows if r['group']==group and r['record']==name and r['probe']=='peak' and r['metric']=='variance_proxy_cv'] for name in names]
    assert all(len(v)==9 for v in vals)
    ax.boxplot(vals,showfliers=False,widths=.5)
    for i,v in enumerate(vals):ax.scatter(i+1+np.linspace(-.12,.12,9),v,s=18,color='#235a86',alpha=.8)
    ax.axhline(100,color='#a44a25',ls='--',lw=1)
    ax.set(title=title,xticks=np.arange(1,6),xticklabels=['Full','Half A','Half B','Gap A','Gap B'])
    ax.tick_params(axis='x',rotation=30);ax.grid(axis='y',alpha=.2)
axes[0].set_ylabel('Variance-proxy standard deviation / mean (%)')
fig.suptitle('Splitting the recording: how noisy would individual variance weights be?',fontsize=14)
fig.tight_layout(rect=(0,.17,1,.93))
fig.text(.02,.04,'200 matched recordings per condition; peak frequencies selected from 400 independent reference recordings.\n'
         'Dots and boxes describe variation across nine channel-peak probes, not confidence intervals or independent replications.\n'
         'Dashed line: standard deviation equals mean. Full recording: 8 Welch segments; each split block: 3. No inverse recovery run.',fontsize=9)
fig.savefig(root/'block_information_precision.png',dpi=180)
