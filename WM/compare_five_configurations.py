"""Compare five NNLS sensor-ablation configurations using paired scenario-level CIs.
Place this file in WM/ and run: python WM/compare_five_configurations.py
Requires numpy, pandas, scipy, matplotlib. Does not rerun inversions.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import t

WM = Path(__file__).resolve().parent
FOLDERS = {
    'Baseline': 'ablation_new_states_nnls',
    'Pressure_only': 'ablation_new_states_nnls',
    'P1_P4_Pitch': 'ablation_new_states_nnls',
    'Motion_only': 'ablation_motion_only',
    'Four_pressure': 'ablation_four_pressure',
}
ORDER = list(FOLDERS)
LABELS = {'Baseline':'9 sensors', 'Pressure_only':'6 pressure',
          'P1_P4_Pitch':'P1+P4+Pitch', 'Motion_only':'Motion only',
          'Four_pressure':'4 pressure'}
METRICS = {'Hs_error_pct':r'$H_s$ error (%)', 'Tp_error_pct':r'$T_p$ error (%)',
           'directional_TV':'Directional TV', 'spectral_TV':'Spectral TV'}
KEY = ['scenario','sea_type','repetition','experiment']
OUT = WM / 'ablation_five_comparison'
OUT.mkdir(exist_ok=True)

frames=[]
for name, folder in FOLDERS.items():
    directory=WM/folder
    vpath=directory/'directional_truth_results.csv'
    rpath=directory/'results.csv'
    if not vpath.exists() or not rpath.exists():
        raise SystemExit(f'Missing {vpath if not vpath.exists() else rpath}')
    val=pd.read_csv(vpath)
    inv=pd.read_csv(rpath)
    val=val[val.experiment.eq(name)].copy()
    inv=inv[(inv.experiment.eq(name)) & (inv.status.eq('success'))].copy()
    if val.empty or inv.empty:
        raise SystemExit(f'No successful records for {name} in {directory}')
    if val.duplicated(KEY).any() or inv.duplicated(KEY).any():
        raise SystemExit(f'Duplicate recording keys for {name}')
    merged=val.merge(inv[KEY+['Hs_error_pct']], on=KEY, how='inner', validate='one_to_one')
    if len(merged)!=len(val) or len(merged)!=len(inv):
        raise SystemExit(f'Unmatched rows for {name}: validation={len(val)}, inversions={len(inv)}, merged={len(merged)}')
    frames.append(merged)

df=pd.concat(frames,ignore_index=True)
# Restrict to recordings present in every configuration, preserving pairing.
recording=['scenario','sea_type','repetition']
counts=df.groupby(recording).experiment.nunique()
complete=counts[counts.eq(len(ORDER))].index
if len(complete)!=600:
    raise SystemExit(f'Expected 600 paired recordings; found {len(complete)}. Check folders/configurations.')
df=df.set_index(recording).loc[complete].reset_index()
df.to_csv(OUT/'combined_results.csv',index=False)

summary=df.groupby('experiment')[list(METRICS)].agg(['mean','median','std']).reindex(ORDER)
summary.columns=['_'.join(c) for c in summary.columns]
summary.insert(0,'n',df.groupby('experiment').size().reindex(ORDER))
summary.to_csv(OUT/'overall_summary.csv')
bytype=df.groupby(['sea_type','experiment'])[list(METRICS)].mean().reset_index()
bytype.to_csv(OUT/'by_sea_type.csv',index=False)

# First compute paired differences within each recording, then average 20 reps per scenario.
# The 30 scenario means are the independent analysis units; t interval is descriptive,
# assuming scenarios are a representative sample. Do not claim equivalence from CI including zero.
rows=[]
for metric in METRICS:
    pivot=df.pivot(index=recording,columns='experiment',values=metric)
    for name in ORDER[1:]:
        dif=pivot[name]-pivot['Baseline']
        scen=dif.groupby(level='scenario').mean()
        n=len(scen); mean=scen.mean(); sd=scen.std(ddof=1)
        half=t.ppf(.975,n-1)*sd/np.sqrt(n)
        rows.append(dict(metric=metric,configuration=name,n_scenarios=n,
                         mean_difference=mean,ci95_lower=mean-half,ci95_upper=mean+half,
                         scenarios_worse=int((scen>0).sum()),scenarios_better=int((scen<0).sum())))
paired=pd.DataFrame(rows)
paired.to_csv(OUT/'paired_scenario_ci.csv',index=False)

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':220})
fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
for ax,(metric,label) in zip(axs.flat,METRICS.items()):
    means=df.groupby('experiment')[metric].mean().reindex(ORDER)
    ax.bar(range(len(ORDER)),means,color=['#32648b','#569a93','#d18b52','#8a6eaa','#719c51'])
    ax.set_xticks(range(len(ORDER)),[LABELS[x] for x in ORDER],rotation=23,ha='right')
    ax.set_ylabel(label);ax.set_ylim(bottom=0)
fig.suptitle('Sensor-ablation performance across 600 paired recordings')
fig.savefig(OUT/'figure_overall_five.png',bbox_inches='tight');plt.close(fig)

fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
sea_order=['wind','swell','mixed','bimodal']
x=np.arange(len(sea_order)); width=.16
for i,name in enumerate(ORDER):
    v=bytype[bytype.experiment.eq(name)].set_index('sea_type').reindex(sea_order)['directional_TV']
    ax.bar(x+(i-2)*width,v,width,label=LABELS[name])
ax.set_xticks(x,[v.title() for v in sea_order]);ax.set_ylabel('Mean directional TV (lower is better)')
ax.set_title('Directional accuracy by sea-state type');ax.legend(ncol=3,frameon=False)
fig.savefig(OUT/'figure_direction_by_type_five.png',bbox_inches='tight');plt.close(fig)

fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
subset=paired[paired.metric.eq('directional_TV')].set_index('configuration').reindex(ORDER[1:])
y=np.arange(len(subset)); mean=subset.mean_difference.to_numpy()
ax.errorbar(mean,y,xerr=[mean-subset.ci95_lower.to_numpy(),subset.ci95_upper.to_numpy()-mean],fmt='o',capsize=4)
ax.axvline(0,color='gray',linestyle='--')
ax.set_yticks(y,[LABELS[x] for x in ORDER[1:]])
ax.set_xlabel('Directional TV change vs 9 sensors (positive = worse)')
ax.set_title('Paired scenario-level mean differences with 95% t intervals')
fig.savefig(OUT/'figure_direction_paired_ci.png',bbox_inches='tight');plt.close(fig)

print(f'Analysed {len(df)} inversions over {len(complete)} paired recordings')
print('\nMean errors:')
print(summary[[m+'_mean' for m in METRICS]].round(4).to_string())
print('\nDirectional TV differences vs baseline (30 scenario-level means):')
print(subset[['mean_difference','ci95_lower','ci95_upper','scenarios_worse']].round(4).to_string())
print(f'\nSaved CSVs and PNG figures to {OUT}')
