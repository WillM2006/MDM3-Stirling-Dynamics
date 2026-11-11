"""Signed contributions to bias, with conditional bootstrap intervals."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent
rows=list(csv.DictReader((root/'diagonal_bias.csv').open()))
components=['model_prior','mean_mismatch','reference_sampling_difference','same_record_coupling','total_bias']
labels=['Prior shrinkage','Mean sensor mismatch','Reference–test difference','Same-record weights / data','Total observed bias']
fig,axes=plt.subplots(1,3,figsize=(12,5),sharey=True)
for ax,group,title in zip(axes,['baseline','long_period','weak_waves'],['Baseline','Long period','Weak waves']):
    for i,component in enumerate(components):
        r=next(r for r in rows if r['group']==group and r['headings']=='12' and r['mode']=='original' and r['component']==component)
        v,lo,hi=[float(r[k]) for k in ['energy_pct','ci95_low','ci95_high']]
        ax.errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt='o',capsize=3,color='#235a86' if component!='total_bias' else '#a44a25')
    ax.axvline(0,color='black',lw=.8)
    ax.set(title=title,xlabel='Percentage of true band energy',yticks=range(5),yticklabels=labels,xlim=(-9,8))
    ax.grid(axis='x',alpha=.2)
axes[0].invert_yaxis()
fig.suptitle('Separating bias in the unchanged point estimator',fontsize=14)
fig.tight_layout(rect=(0,.16,1,.94))
fig.text(.02,.04,'200 existing test recordings and 400 independent reference recordings per condition; 12 headings; M = 8.785, tau = 5.\n'
         'Bars: pointwise 95% whole-record bootstrap intervals. The first four contributions sum to total bias.\n'
         'Diagnostic only: independent donor weights and known wave truth are not proposed production corrections.',fontsize=9)
fig.savefig(root/'diagonal_bias.png',dpi=180)
