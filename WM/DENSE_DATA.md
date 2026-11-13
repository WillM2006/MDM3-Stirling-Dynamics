# Regenerating the sensor dataset

From `WM`, use a new output directory:

```sh
python create_data.py --out sensor_data_dense
python create_data.py --out sensor_data_dense --status
```

The same command resumes interrupted generation. The sweep is unchanged:
25 sea/depth settings, 20 seeds each, 1,200 seconds per recording. It now uses
6,400 uniformly spaced wave frequencies and 36 directions, with the original
40-frequency hydrodynamic grid. Wave-bin energies are integrated from the
same JONSWAP spectrum; the hull dynamics, controller and sensor errors are
unchanged. Uniform harmonic sums are evaluated using SciPy's zoom FFT.

`generation.json` records the settings and generator source hashes. Resuming
with different settings/code, or into an old dataset without this manifest,
is rejected. `--force` reruns matching data only. Keep the original dataset;
do not generate into its directory. Each `params.csv` records `n_freq`,
`n_bem` and `uniform_frequencies` so inversion uses the correct grids.

Run the existing inversions on this new root, using new results filenames:

```sh
python batch_invert.py --root sensor_data_dense --out dense_results.csv --save-spectra dense_spectra
python batch_invert.py --root sensor_data_dense --positive --out dense_results_positive.csv --save-spectra dense_spectra_positive
PYTHONPATH="../WB Simplest Model" python batch_invert.py --root sensor_data_dense --model six_dof --out dense_results_six_dof.csv
```

The existing M, prior and variance formulas are unchanged. Old datasets remain
readable: without `n_bem` or `uniform_frequencies`, inversion assumes the old
equal-energy generator and shared frequency count.

Before replacing the Drive dataset, confirm `--status` reports 500/500 and
check the inversion logs for failed runs. Archive/upload the whole new data
root, including `generation.json`, per-run parameters and completion flags.
This change does not upload to Drive or overwrite the existing archive.

## Checks

```sh
python -m unittest discover -s WM -p test_dense_generation.py -v
```

The tests compare accelerated sums and trajectories against direct evaluation,
check energy and pressure across all 25 sweep settings, separate the BEM grid,
and exercise resume protection and old/new inversion metadata. They require
the existing simulation dependencies, including Capytaine, but do not solve BEM.

The 6,400 choice comes from the separate resolution study: pooled spectral
variances were stable against 12,800 components in the tested baseline sea.
It is not a claim that all frequencies or all sea conditions have converged.
Dense waves reduce sparse-frequency artefacts; they do not guarantee smoother
inverse estimates or more accurate posterior uncertainty.
