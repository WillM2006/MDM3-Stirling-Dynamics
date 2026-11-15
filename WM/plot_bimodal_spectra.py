"""Plot theoretical vs reconstructed spectra for BI_O, repetition 0.

Place in WM/ and run from repository root:
    python WM/plot_bimodal_spectra.py
    python WM/plot_bimodal_spectra.py --all-five

Uses saved NPZ spectra and systems.csv; does not rerun any inversion.
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from validate_directional_truth import systems_from_csv, true_spectrum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', default='seastate_BI_O_bimodal')
    parser.add_argument('--rep', type=int, default=0)
    parser.add_argument('--all-five', action='store_true', help='Include six-pressure and P1+P4+Pitch panels')
    parser.add_argument('--out-dir', type=Path, default=Path('WM/ablation_five_comparison'))
    args = parser.parse_args()
    wm = Path(__file__).resolve().parent
    root = wm.parent
    data_path = root / 'sensor_data_states' / args.scenario / f'sensordata_{args.rep}' / 'systems.csv'
    systems = systems_from_csv(data_path)
    configurations = [
        ('9 sensors', 'ablation_new_states_nnls', 'Baseline'),
        ('4 pressure', 'ablation_four_pressure', 'Four_pressure'),
        ('Motion-only', 'ablation_motion_only', 'Motion_only'),
    ]
    if args.all_five:
        configurations = [
            configurations[0],
            ('6 pressure', 'ablation_new_states_nnls', 'Pressure_only'),
            configurations[1],
            ('P1 + P4 + Pitch', 'ablation_new_states_nnls', 'P1_P4_Pitch'),
            configurations[2],
        ]

    arrays = []
    f_ref = beta_ref = None
    for title, folder, config in configurations:
        path = wm / folder / 'spectra' / f'{args.scenario}_rep{args.rep}_{config}.npz'
        if not path.is_file():
            parser.error(f'Missing spectrum: {path}')
        with np.load(path) as z:
            f = np.asarray(z['f'], dtype=float)
            beta = np.asarray(z['beta'], dtype=float)
            x = np.asarray(z['x'], dtype=float)
        if x.shape != (len(f), len(beta)):
            raise ValueError(f'Invalid spectrum shape in {path}')
        if f_ref is None:
            f_ref, beta_ref = f, beta
        elif not (np.allclose(f, f_ref) and np.allclose(beta, beta_ref)):
            raise ValueError(f'Inversion grids differ: {path}')
        # Inversion x is wave power per Hz *per directional bin*.
        # Convert to S(f,theta) in m^2/(Hz rad) for comparison with Sea.
        dtheta = 2 * np.pi / len(beta)
        arrays.append((title, np.maximum(x, 0) / dtheta))

    truth = true_spectrum(f_ref, beta_ref, systems)
    panels = [('Theoretical generating spectrum', truth)] + arrays
    # Convert headings to [0,360), reorder to avoid seam in the heatmap.
    order = np.argsort(np.mod(np.rad2deg(beta_ref), 360))
    headings = np.mod(np.rad2deg(beta_ref[order]), 360)
    # Shared scale ensures differences in reconstructed amplitude are visible.
    vmax = max(float(np.max(z)) for _, z in panels)
    ncols = 3 if args.all_five else 2
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(5.5*ncols, 4.1*nrows), sharex=True, sharey=True, constrained_layout=True)
    axs = np.asarray(axs).ravel()
    for ax, (name, z) in zip(axs, panels):
        img = ax.pcolormesh(
            headings,
            f_ref,
            z[:, order],
            shading='nearest',
            cmap='viridis',
            norm=Normalize(vmin=0, vmax=vmax)
        )

        # Mark the true wave directions from systems.csv
        for system in systems:
            direction_deg = np.rad2deg(system['direction']) % 360

            ax.axvline(
                direction_deg,
                color='white',
                linestyle='--',
                linewidth=1.5,
                alpha=0.85
            )

        ax.set_title(name)
        ax.set_xlabel('Wave travel direction (degrees)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_xticks([0, 90, 180, 270, 360])
        ax.set_xlim(-5, 355)
        ax.set_ylim(f_ref[0], f_ref[-1])

    for ax in axs[len(panels):]:
        ax.set_visible(False)
    fig.colorbar(img, ax=list(axs[:len(panels)]), label=r'Directional spectral density $S(f,\theta)$ [$m^2/(Hz\,rad)$]', shrink=.83)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    filename = f'{args.scenario}_rep{args.rep}_spectra_{"five" if args.all_five else "four"}_panels.png'
    path = args.out_dir / filename
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f'Saved {path.resolve()}')
    print('True wave systems:')
    for s in systems:
        print(f"  Hs={s['Hs']:.2f} m, Tp={s['Tp']:.2f} s, direction={np.rad2deg(s['direction'])%360:.1f} degrees")

if __name__ == '__main__':
    main()
