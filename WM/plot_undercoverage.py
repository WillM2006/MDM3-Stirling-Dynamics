"""Diagnostic coverage comparisons; no fitted correction is proposed."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
rows=list(csv.DictReader((root/'undercoverage_scenarios.csv').open()))
choices=[('0','0','paired','Original intervals'),('1','0','paired','Remove reference bias only'),
         ('0','1','paired','Adjust variance scale only'),('0','0','independent','Break error–width pairing only'),
         ('0','0','constant','Constant widths only'),('1','1','constant','All diagnostic adjustments')]
fig,axes=plt.subplots(1,3,figsize=(12,5.4),sharey=True)
for ax,group,title in zip(axes,['baseline','long_period','weak_waves'],['Baseline','Long period','Weak waves']):
    for i,(b,s,mode,label) in enumerate(choices):
        r=next(r for r in rows if r['group']==group and r['headings']=='12' and r['remove_reference_bias']==b and r['reference_variance_scale']==s and r['width_mode']==mode)
        y=100*float(r['coverage']); lo=100*float(r['coverage_ci_low']); hi=100*float(r['coverage_ci_high'])
        ax.errorbar(y,i,xerr=[[y-lo],[hi-y]],fmt='o',color='#235a86' if i==0 else '#9b5d28',capsize=3)
    ax.axvline(95,color='black',ls='--',lw=1)
    ax.set(title=title,xlabel='Coverage (%)',xlim=(75,100),yticks=range(len(choices)),yticklabels=[x[3] for x in choices])
    ax.grid(axis='x',alpha=.2)
axes[0].invert_yaxis()
fig.suptitle('Why do nominal 95% intervals miss? Diagnostic comparisons',fontsize=14)
fig.tight_layout(rect=(0,.15,1,.94))
fig.text(.02,.035,'200 existing test records per condition; independent 400-record references; fixed M = 8.785, point estimator and prior.\n'
         'Bars: pointwise 95% intervals, conditional on fixed gains and covariance pool. Orange comparisons use diagnostic interventions,\n'
         'including known truth; they are not deployable corrections or fresh validation. Source: undercoverage_scenarios.csv.',fontsize=9)
fig.savefig(root/'undercoverage_diagnosis.png',dpi=180)
