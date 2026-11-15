"""Validate saved NNLS spectra against the theoretical generating directional spectrum.

Place in WM/ and run from repository root:
    python WM/validate_directional_truth.py
No inversions or hydrodynamics are rerun.
"""
import argparse
import csv
from pathlib import Path
import numpy as np
import pandas as pd
from sea import Sea


def systems_from_csv(path):
    with path.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f'Empty systems.csv: {path}')
    return [dict(Hs=float(r['Hs']), Tp=float(r['Tp']), gamma=float(r['gamma']),
                 s=float(r['s']), direction=float(r['direction_rad'])) for r in rows]


def true_spectrum(f, beta, systems):
    # Use the *same* spectral formulas and discrete angular normalization as Sea,
    # but evaluate on the inversion grid without constructing random wave phases.
    sea = Sea.__new__(Sea)
    sea.g = 9.81
    sea.delta_theta = np.full(beta.size, 2 * np.pi / beta.size)
    spectrum = np.zeros((len(f), len(beta)), dtype=float)
    for q in systems:
        spectrum += sea._jonswap_one(f, q['Hs'], q['Tp'], q['gamma'])[:, None] * \
                    sea._cos2s(beta, q['direction'], q['s'])[None, :]
    return spectrum


def compare(x, truth, f, beta):
    if x.shape != truth.shape or len(f) < 2:
        raise ValueError('Spectrum shapes or frequency grid invalid')
    if not (np.isfinite(x).all() and np.isfinite(truth).all()):
        raise ValueError('Nonfinite spectral values')
    if not np.all(np.diff(f) > 0):
        raise ValueError('Frequency grid must be increasing')
    # Discrete frequency-bin widths; exact for the uniform Welch grid.
    edges = np.r_[f[0] - (f[1] - f[0]) / 2, (f[1:] + f[:-1]) / 2,
                  f[-1] + (f[-1] - f[-2]) / 2]
    df = np.diff(edges)
    dtheta = 2 * np.pi / len(beta)
    weights = df[:, None] * dtheta
    x = np.maximum(x, 0)
    estimated_energy = x * df[:, None]
    true_energy = truth * weights
    total_est = estimated_energy.sum()
    total_true = true_energy.sum()
    if min(total_est, total_true) <= 0:
        raise ValueError('Nonpositive spectral energy')
    # Comparing normalized spectral shapes avoids conflating direction/frequency
    # errors with overall Hs bias. Total band power error is reported separately.
    p = (estimated_energy / total_est).ravel()
    q = (true_energy / total_true).ravel()
    tv = 0.5 * np.abs(p - q).sum()
    m = (p + q) / 2
    js = 0.5 * (np.sum(p[p > 0] * np.log2(p[p > 0] / m[p > 0])) +
                np.sum(q[q > 0] * np.log2(q[q > 0] / m[q > 0])))
    est_dir = estimated_energy.sum(axis=0) / total_est
    true_dir = true_energy.sum(axis=0) / total_true
    dir_tv = 0.5 * np.abs(est_dir - true_dir).sum()
    est_f = estimated_energy.sum(axis=1)
    true_f = true_energy.sum(axis=1)
    tp_est_from_x = 1 / f[np.argmax(est_f / df)]
    tp_true_band = 1 / f[np.argmax(true_f / df)]
    return dict(spectral_TV=tv, spectral_JS_bits=js, directional_TV=dir_tv,
                band_energy_error_pct=100 * abs(total_est - total_true) / total_true,
                Tp_true_band=tp_true_band, Tp_est_spectrum=tp_est_from_x,
                Tp_error_pct=100 * abs(tp_est_from_x - tp_true_band) / tp_true_band,
                Hs_true_band=4 * np.sqrt(total_true), Hs_est_band=4 * np.sqrt(total_est))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    wm = Path(__file__).resolve().parent
    root = wm.parent
    data_root = (args.data_root or root / 'sensor_data_states').resolve()
    output = (args.output or wm / 'ablation_new_states_nnls').resolve()
    results_file = output / 'results.csv'
    if not results_file.is_file():
        parser.error(f'Missing {results_file}')
    rows = pd.read_csv(results_file)
    output_rows = []
    errors = []
    for row in rows.itertuples(index=False):
        if row.status != 'success':
            continue
        spectrum_path = output / 'spectra' / f'{row.scenario}_rep{row.repetition}_{row.experiment}.npz'
        systems_path = data_root / row.scenario / f'sensordata_{row.repetition}' / 'systems.csv'
        try:
            systems = systems_from_csv(systems_path)
            with np.load(spectrum_path) as z:
                f = np.asarray(z['f'])
                beta = np.asarray(z['beta'])
                x = np.asarray(z['x'])
            truth = true_spectrum(f, beta, systems)
            metrics = compare(x, truth, f, beta)
            output_rows.append(dict(scenario=row.scenario, sea_type=row.sea_type,
                                    repetition=row.repetition, experiment=row.experiment, **metrics))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append((row.scenario, row.repetition, row.experiment, str(exc)))
    if not output_rows:
        raise SystemExit(f'No valid comparisons. First errors: {errors[:3]}')
    df = pd.DataFrame(output_rows)
    df.to_csv(output / 'directional_truth_results.csv', index=False)
    summary = df.groupby(['sea_type', 'experiment'], as_index=False).agg(
        mean_spectral_TV=('spectral_TV', 'mean'),
        mean_directional_TV=('directional_TV', 'mean'),
        mean_spectral_JS_bits=('spectral_JS_bits', 'mean'),
        mean_Tp_error_pct=('Tp_error_pct', 'mean'),
        n=('spectral_TV', 'count'))
    summary.to_csv(output / 'directional_truth_by_type.csv', index=False)
    overall = df.groupby('experiment', as_index=False).agg(
        mean_spectral_TV=('spectral_TV', 'mean'),
        mean_directional_TV=('directional_TV', 'mean'),
        mean_spectral_JS_bits=('spectral_JS_bits', 'mean'),
        mean_Tp_error_pct=('Tp_error_pct', 'mean'),
        n=('spectral_TV', 'count'))
    overall.to_csv(output / 'directional_truth_overall.csv', index=False)
    print(f'Compared {len(df)} saved spectra; {len(errors)} failed')
    for item in errors[:10]:
        print('ERROR:', item)
    print('\nOverall (lower TV/JS/error is better):')
    print(overall.round(3).to_string(index=False))
    print('\nBy sea type:')
    print(summary.round(3).to_string(index=False))
    print(f'\nSaved results in {output}')


if __name__ == '__main__':
    main()
