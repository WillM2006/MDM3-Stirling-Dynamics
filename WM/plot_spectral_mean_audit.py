"""Export figures from the diagnostic CSVs, without rerunning simulation."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent


def plot():
    ratios=list(csv.DictReader((ROOT/'spectral_mean_band_ratios.csv').open()))
    effects=list(csv.DictReader((ROOT/'spectral_mean_effects.csv').open()))
    groups=['baseline','long_period','weak_waves'];titles=['Baseline','Long period','Weak waves']
    channels=['roll','pitch','vert_acc','P1','P2','P3','P4','P5','P6']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(13.7,5.1),sharey=True)
    for ax,group,title in zip(axes,groups,titles):
        for variant,label,colour,offset in [('centre36','Current bin-centre response','#777777',-.12),
                                           ('direct_impedance','Direct linear response + Welch/processing','#2864A5',.12)]:
            values=[next(r for r in ratios if r['group']==group and r['variant']==variant and r['channel']==c) for c in channels]
            y=np.array([float(r['predicted_over_observed']) for r in values])
            lo=np.array([float(r['ci95_low']) for r in values]);hi=np.array([float(r['ci95_high']) for r in values])
            ax.errorbar(np.arange(9)+offset,y,yerr=np.stack([y-lo,hi-y]),fmt='o',ms=4,color=colour,label=label)
        ax.axhline(1,color='#444444',ls='--',lw=1)
        ax.set_xticks(range(9),channels,rotation=45,ha='right');ax.set(title=title,xlabel='Sensor channel')
        ax.grid(axis='y',alpha=.16)
    axes[0].set_ylabel('Predicted / observed mean band power (ratio)')
    axes[0].legend(frameon=False,fontsize=8,loc='lower left')
    fig.suptitle('Mean sensor power: response interpolation matters, but is not the whole inverse problem')
    fig.text(.04,.025,'Source: spectral_mean_band_ratios.csv; 200 existing records per condition, 0.05–0.31 Hz, 36 headings. Bars: 95% record-bootstrap intervals for observed means, conditional on predictions.\n'
             'This compares diagnostic means, not corrected inverse recovery; intervals are not simultaneous and exclude physical-model uncertainty.',fontsize=8,color='#555555')
    fig.tight_layout(rect=(0,.16,1,.94));fig.savefig(ROOT/'spectral_mean_channel_ratios.png',dpi=170);plt.close(fig)
    stages=['centre_own_grid','centre36','cell_integrated','welch_averaged','processed','direct_impedance','memory_noise_clock','observed_no_drag','observed_paired_drag','observed']
    labels=['Own-grid model mean','36-direction mean','Frequency-cell integration','Welch window averaging','Sensor processing','Direct linear impedance','Memory + noise + clocks','Native mean: drag off (64)','Native mean: drag on (64)','Native mean: drag on (200)']
    fig,axes=plt.subplots(1,3,figsize=(13.5,6.8),sharey=True)
    for ax,group,title in zip(axes,groups,titles):
        values=[float(next(r for r in effects if r['group']==group and r['headings']=='12' and r['variant']==v)['recovered_energy_bias_pct']) for v in stages]
        ax.plot(values[:7],range(7),'o-',color='#2864A5',lw=1,ms=5)
        ax.scatter(values[7:],range(7,10),color=['#21835D','#A45B35','#444444'],marker='s',s=35)
        ax.axvline(0,color='#555555',ls='--');ax.set(title=title,xlabel='Projected band-energy bias (%)',xlim=(-50,2))
        ax.grid(axis='x',alpha=.16);ax.set_yticks(range(10),labels)
    axes[0].invert_yaxis()
    fig.suptitle('Mean-input diagnostic through the unchanged pooled-joint inverse (12 headings)')
    fig.text(.025,.025,'Source: spectral_mean_effects.csv. M=8.785, tau=5, covariance and reference prior scales fixed. Each point changes the INPUT MEAN to the original recovery operator.\n'
             'These are post-validation diagnostics, not predictions of how a corrected estimator would recover waves. Percentages concern energy, not wave height.\n'
             'Paired drag comparisons use the same 64 phases/noise seeds per sea; full native means use 200. The analytic sequence is order dependent and does not explain every residual.',fontsize=8,color='#555555')
    fig.tight_layout(rect=(0,.17,1,.94));fig.savefig(ROOT/'spectral_mean_energy_diagnostic.png',dpi=170);plt.close(fig)


if __name__=='__main__':plot()
