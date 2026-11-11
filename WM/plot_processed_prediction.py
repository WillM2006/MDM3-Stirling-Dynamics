"""Plot held-out processing-only mean prediction checks from saved CSVs."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent


def plot():
    changes=list(csv.DictReader((ROOT/'processed_prediction_changes.csv').open()))
    scores=list(csv.DictReader((ROOT/'processed_prediction_scores.csv').open()))
    groups=['baseline','long_period','weak_waves'];labels=['Baseline','Long period','Weak waves']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,5.4),sharey=True)
    for ax,headings in zip(axes,[12,36]):
        for method,name,colour,offset in [('filter_alignment','Filter + time alignment','#777777',-.08),
                                         ('full_processing','Filter + alignment + Welch cell model','#2864A5',.08)]:
            rows=[next(r for r in changes if r['group']==g and r['headings']==str(headings) and r['method']==method and r['family']=='all') for g in groups]
            mean=np.array([float(r['improvement_pct']) for r in rows])
            lo=np.array([float(r['ci95_low']) for r in rows]);hi=np.array([float(r['ci95_high']) for r in rows])
            # Percentile bootstrap intervals can be biased around nonlinear scores.
            ax.plot(np.arange(3)+offset,mean,'o',color=colour,label=name)
            ax.vlines(np.arange(3)+offset,lo,hi,color=colour)
        ax.axhline(0,color='#444444',lw=1);ax.axhline(10,color='#555555',lw=1,ls='--')
        ax.set_xticks(range(3),labels);ax.set(title=f'{headings} headings',xlabel='Held-out sea condition')
        ax.grid(axis='y',alpha=.15)
    axes[0].set_ylabel('Reduction in standardised mean-prediction error (%)')
    axes[0].legend(frameon=False,fontsize=8,loc='best')
    fig.suptitle('Does known processing improve sensor-spectrum prediction?')
    fig.text(.04,.025,'Source: processed_prediction_changes.csv; 200 fresh native records per condition, nonlinear drag retained. Positive means improvement; dashed line: prespecified 10% practical target.\n'
             'Same fixed noise mean in control/candidates; 1000 paired record-bootstrap intervals, conditional on predictions and earlier normalisation. This is not wave-recovery or posterior-coverage evidence.',fontsize=8,color='#555555')
    fig.tight_layout(rect=(.01,.16,.99,.94));fig.savefig(ROOT/'processed_prediction_improvement.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,5.3),sharey=True)
    methods=[('current_plus_noise','Current + fixed noise','#777777'),('filter_alignment','Filter + alignment','#2864A5'),('full_processing','Full processing/cell model','#21835D')]
    ymax=1.08*max(float(r['ci95_high']) for r in scores if r['headings']=='12')
    for ax,family,title in zip(axes,['all','auto','cross'],['All 81 real spectral entries','Auto-spectra','Real + imaginary cross-spectra']):
        for method,label,colour in methods:
            rows=[next(r for r in scores if r['group']==g and r['headings']=='12' and r['method']==method and r['family']==family) for g in groups]
            mean=[float(r['mean_prediction_error']) for r in rows]
            ax.plot(range(3),mean,'o-',color=colour,label=label)
            ax.vlines(range(3),[float(r['ci95_low']) for r in rows],[float(r['ci95_high']) for r in rows],color=colour)
        ax.set_xticks(range(3),labels);ax.set(title=title,xlabel='Held-out sea condition');ax.set_ylim(0,ymax);ax.grid(axis='y',alpha=.15)
    axes[0].set_ylabel('RMS mean error / earlier repeat-record SD')
    axes[0].legend(frameon=False,fontsize=8,loc='best')
    fig.suptitle('Mean sensor-spectrum error on the primary 12-heading grid; lower is better')
    fig.text(.04,.025,'Source: processed_prediction_scores.csv. Each entry is scaled using earlier sensor-only repeat-record SD; scores compare predicted ensemble means with 200 new records.\n'
             'Bars: conditional 95% percentile bootstrap intervals, not simultaneous guarantees. The physical RAO and nonlinear generator/inverse mismatch are unchanged.',fontsize=8,color='#555555')
    fig.tight_layout(rect=(.01,.16,.99,.94));fig.savefig(ROOT/'processed_prediction_errors.png',dpi=170);plt.close(fig)
    if not (ROOT/'processing_recovery_summary.csv').exists():return
    recovery=list(csv.DictReader((ROOT/'processing_recovery_summary.csv').open()))
    fig,axes=plt.subplots(1,3,figsize=(13,5.3))
    for ax,metric,title,scale in zip(axes,['hs_abs_pct','direction_l1','m0_cover95'],
                                    ['Mean absolute band-Hs error (%)','Signed direction-shape L1 distance','Coverage of nominal 95% energy intervals (%)'],[1,1,100]):
        for method,label,colour in [('uncorrected_response','Uncorrected response','#777777'),('filter_alignment_response','Filter + alignment response','#2864A5')]:
            rows=[next(r for r in recovery if r['group']==g and r['headings']=='12' and r['method']==method and r['metric']==metric) for g in groups]
            ax.plot(range(3),[float(r['mean'])*scale for r in rows],'o-',color=colour,label=label)
            ax.vlines(range(3),[float(r['ci95_low'])*scale for r in rows],[float(r['ci95_high'])*scale for r in rows],color=colour)
        ax.set_xticks(range(3),labels);ax.set(title=title,xlabel='Sea condition');ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.15)
        if metric=='m0_cover95':ax.axhline(95,color='#444444',ls='--');ax.set_ylim(0,101)
    axes[0].legend(frameon=False,fontsize=8,loc='best')
    fig.suptitle('Exploratory wave recovery: processing response only, same M and prior')
    fig.text(.04,.025,'Source: processing_recovery_summary.csv. 12 headings shown; M=8.785, tau=5, corrected diagonal variances and each recording’s original prior scales unchanged.\n'
             '600 records already examined for forward prediction; this follow-up is not an untouched final recovery test. Nonlinear drag remains. Bars: conditional 95% bootstrap/exact coverage intervals.',fontsize=8,color='#555555')
    fig.tight_layout(rect=(.01,.16,.99,.94));fig.savefig(ROOT/'processing_recovery_comparison.png',dpi=170);plt.close(fig)


if __name__=='__main__':plot()
