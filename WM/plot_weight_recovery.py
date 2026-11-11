"""Paired recovery changes for the single sensor-selected weight candidate."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent
rows=list(csv.DictReader((root/'profile_weights_summary.csv').open()))
paired=list(csv.DictReader((root/'profile_weights_paired.csv').open()))
fig,axes=plt.subplots(1,2,figsize=(11,5))
groups=['baseline','long_period','weak_waves'];labels=['Baseline','Long period','Weak waves']
for i,g in enumerate(groups):
    r=next(r for r in paired if r['group']==g and r['headings']=='12' and r['metric']=='hs_abs_pct')
    v,lo,hi=[float(r[k]) for k in ['change','ci95_low','ci95_high']]
    axes[0].errorbar(i,v,yerr=[[v-lo],[hi-v]],fmt='o',color='#235a86',capsize=4)
    for j,(method,color) in enumerate([('original','#235a86'),('smoothed','#a44a25')]):
        r=next(r for r in rows if r['group']==g and r['headings']=='12' and r['metric']=='energy_bias_pct' and r['method']==method)
        v,lo,hi=[float(r[k]) for k in ['mean','ci95_low','ci95_high']]
        axes[1].errorbar(i+(j-.5)*.16,v,yerr=[[v-lo],[hi-v]],fmt='o',color=color,capsize=4,label=method if i==0 else None)
for ax in axes:
    ax.axhline(0,color='black',ls='--',lw=.8);ax.set_xticks(range(3),labels);ax.grid(axis='y',alpha=.2)
axes[0].set(title='Change in mean absolute wave-height error',ylabel='Percentage points (negative is better)')
axes[1].set(title='Signed mean wave-energy bias',ylabel='Percentage of true retained-band energy');axes[1].legend()
fig.suptitle('Sensor-selected weights: recovery must improve across conditions',fontsize=14)
fig.tight_layout(rect=(0,.16,1,.92))
fig.text(.025,.035,'12 headings; 200 existing recordings per condition; fixed M = 8.785 and original prior scales.\n'
         '17-bin smoothing relative to an independent baseline sensor-variance profile; no wave-truth tuning.\n'
         'Bars: pointwise paired/whole-record bootstrap 95% intervals, conditional on selected rule. Retrospective, not fresh validation.',fontsize=9)
fig.savefig(root/'profile_weights_recovery.png',dpi=180)
