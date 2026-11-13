# MDM3-Stirling-Dynamics

## Regenerating the sensor data

Run from the repository root using the existing simulation environment:

```sh
cd WM
python create_data.py --out sensor_data_dense
python create_data.py --out sensor_data_dense --status
python batch_invert.py --root sensor_data_dense --positive
```

This generates the existing 500-record sweep using 6,400 wave frequencies
and 40 hydrodynamic frequencies. Repeat the generation command to resume.

Use a new output directory; keep the original Drive dataset.
Inversion results are saved inside `sensor_data_dense`.
Omit `--positive` to run the unconstrained inverse.

Before uploading, confirm all 500 recordings are complete and check the
inversion output for failed runs. Upload the whole data directory,
including its parameters, completion flags and `generation.json`.
