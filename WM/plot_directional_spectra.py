"""Compare estimated directional energy distributions for one repetition.

Run from the directory containing ablation_* result folders:
    python plot_directional_spectra.py --rep 0 --direction 90
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--rep', type=int, default=0)
parser.add_argument('--direction', type=float, default=90.0, help='True wave direction (degrees)')
args = parser.parse_args()

sources = {
    'P3 only': Path(f'ablation_combinations_nnls/spectra/rep{args.rep}_P3_only.npz'),
    'P3 + P5': Path(f'ablation_combinations_nnls/spectra/rep{args.rep}_P3_P5.npz'),
    'All nine sensors': Path(f'ablation_direction_nnls/spectra/rep{args.rep}_Baseline.npz'),
}

fig, ax = plt.subplots(figsize=(10, 5.5))
for label, path in sources.items():
    if not path.is_file():
        print(f'Missing: {path} (skipping)')
        continue
    with np.load(path) as data:
        x = np.asarray(data['x'])
        beta = np.asarray(data['beta']).reshape(-1)
    if x.ndim != 2 or x.shape[1] != len(beta):
        raise ValueError(f'{path}: expected x to have shape (frequency, direction); got {x.shape}, beta {beta.shape}')
    if np.max(np.abs(beta)) > 2 * np.pi + 0.01:
        raise ValueError(f'{path}: beta does not appear to be in radians')
    energy = np.maximum(x, 0).sum(axis=0)
    if energy.sum() <= 0:
        print(f'{label}: no positive energy; skipping')
        continue
    energy /= energy.sum()
    angles = (np.degrees(beta) + 180) % 360 - 180
    order = np.argsort(angles)
    ax.plot(angles[order], 100 * energy[order], marker='o', markersize=3, label=label)
    angular_distance = (angles - args.direction + 180) % 360 - 180
    outside = 100 * energy[np.abs(angular_distance) > 30].sum()
    c, s = np.sum(energy * np.cos(beta)), np.sum(energy * np.sin(beta))
    concentration = np.hypot(c, s)
    mean = np.degrees(np.arctan2(s, c)) if concentration > 1e-12 else float('nan')
    peak = angles[np.argmax(energy)]
    print(f'{label}: mean={mean:.2f}°, peak={peak:.1f}°, outside ±30°={outside:.1f}%, circular concentration={concentration:.3f}')

ax.axvline((args.direction + 180) % 360 - 180, linestyle='--', color='black', label='True direction')
ax.set(xlabel='Wave direction (degrees)', ylabel='Share of estimated directional energy per bin (%)',
       title=f'Directional energy distribution — repetition {args.rep}')
ax.set_xlim(-180, 180)
ax.set_xticks(np.arange(-180, 181, 45))
ax.grid(alpha=0.25)
ax.legend()
fig.tight_layout()
out = Path('ablation_comparison')
out.mkdir(exist_ok=True)
filename = out / f'directional_spectra_rep{args.rep}.png'
fig.savefig(filename, dpi=200)
print(f'Saved: {filename}')
plt.show()