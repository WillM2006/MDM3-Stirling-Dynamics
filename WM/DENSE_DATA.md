# Regenerating the sensor dataset

From `WM`, generate into a **new directory**, keeping the original Drive data:

```sh
python create_data.py --out sensor_data_dense
python create_data.py --out sensor_data_dense --status
```

This runs the existing 500 recordings with 6,400 wave frequencies and 40 BEM
frequencies. Repeat the first command to resume. `generation.json` prevents
resuming with different settings or generator code.

Run the inversions with separate output names:

```sh
python batch_invert.py --root sensor_data_dense --out dense_results.csv --save-spectra dense_spectra
python batch_invert.py --root sensor_data_dense --positive --out dense_results_positive.csv --save-spectra dense_spectra_positive
PYTHONPATH="../WB Simplest Model" python batch_invert.py --root sensor_data_dense --model six_dof --out dense_results_six_dof.csv
```

Before uploading to Drive, confirm `--status` reports 500/500 and check the
inversion logs for failed runs. Archive the whole data directory, including
`generation.json`, parameters and completion flags. Uploading is a separate step.
