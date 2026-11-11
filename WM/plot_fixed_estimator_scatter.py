"""Show variance prediction separately from point recovery and pool transfer."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parent
rows = list(csv.DictReader((ROOT/'fixed_estimator_scatter.csv').open()))
groups = ['baseline', 'long_period', 'weak_waves']
labels = ['Baseline', 'Long period', 'Weak waves']
methods = ['original_posterior', 'record_diagonal', 'record_joint', 'pooled_diagonal', 'pooled_joint', 'empirical_full']
names = ['Saved diagonal posterior', 'Record marginal variances', 'Record Wishart + Hann',
         'Pooled marginal variances', 'Pooled correlations + Hann', 'Full empirical reference']


def get(group, pool, method):
    selected = [r for r in rows if r['group']==group and r['covariance_pool']==pool and r['method']==method
                and r['headings']=='12' and r['gain_reference']=='baseline']
    assert len(selected)==1
    return selected[0]


plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
fig, axes = plt.subplots(1, 3, figsize=(13, 6), sharey=True)
for ax, group, label in zip(axes, groups, labels):
    for i, method in enumerate(methods):
        r = get(group, 'none' if method in methods[:3] else group, method)
        value, low, high = [float(r[k]) for k in ['actual_ratio', 'actual_ratio_ci_low', 'actual_ratio_ci_high']]
        color = '#2864A5' if method=='empirical_full' else '#444444'
        ax.errorbar(value, i, xerr=[[value-low], [high-value]], fmt='o', color=color, capsize=3)
    ax.axvspan(.9, 1.1, color='#EEEEEE')
    ax.axvline(1, color='#888888', ls='--', lw=1)
    ax.set(xscale='log', xlim=(.008, 2.5), title=label, xlabel='Predicted / observed energy variance')
    ax.set_yticks(range(len(names)), names)
axes[0].invert_yaxis()
fig.suptitle('Fixed point estimates: can the covariance rules predict their actual scatter?', fontsize=14)
fig.text(.025, .035, 'Source: fixed_estimator_scatter.csv. 200 existing records per condition; 12 headings; M = 8.785, tau = 5; baseline calibration fixes all gains.\n'
         'Pooled rows here assume representative same-condition calibration; transfer is shown separately. Grey band: 0.9–1.1; bars: pointwise 95% bootstrap intervals.\n'
         'Intervals condition on fixed gains, M and structured pools; the empirical reference also resamples calibration projections. No point estimate or bias was changed.', fontsize=9)
fig.tight_layout(rect=(0, .18, 1, .94))
fig.savefig(ROOT/'fixed_estimator_variance.png', dpi=180)
plt.close(fig)

matrices = []
for method in ['pooled_joint', 'empirical_full']:
    matrices.append(np.array([[float(get(group, pool, method)['actual_ratio']) for group in groups] for pool in groups]))
extent = max(abs(np.log10(matrix)).max() for matrix in matrices)
fig, axes = plt.subplots(1, 2, figsize=(10.8, 6))
fig.subplots_adjust(left=.13, right=.88, bottom=.24, top=.82, wspace=.35)
for ax, matrix, title in zip(axes, matrices, ['Pooled correlations + Hann', 'Full empirical reference']):
    im = ax.imshow(np.log10(matrix), cmap='RdBu_r', norm=TwoSlopeNorm(vmin=-extent, vcenter=0, vmax=extent))
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f'{matrix[i,j]:.3g}', ha='center', va='center', color='black',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=.85, pad=2))
    ax.set_xticks(range(3), labels); ax.set_yticks(range(3), labels)
    ax.set(title=title, xlabel='Test condition', ylabel='Covariance calibration pool')
fig.colorbar(im, cax=fig.add_axes([.91,.29,.02,.46]), label='log10(predicted / observed variance)')
fig.suptitle('Every calibration pool applied to every condition — no pool selection', fontsize=14)
fig.text(.025,.04,'Source: fixed_estimator_scatter.csv. Same saved point estimates, 12 headings and baseline-reference gains in every cell.\n'
         'One is the target. Different conditions can have very different absolute variability; a representative pool is an assumption, not an estimator.\n'
         'All pointwise confidence intervals, other gain references and the 36-heading sensitivity are retained in the CSV.',fontsize=9)
fig.savefig(ROOT/'fixed_estimator_transfer.png',dpi=180)
plt.close(fig)
