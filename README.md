# MDM3-Stirling-Dynamics

## Regenerating the sensor data

Run from the repository root using the existing simulation environment.
Both datasets use 6,400 evenly spaced wave frequencies (fast FFT synthesis)
and 40 hydrodynamic frequencies.

**Sea states (default):** 30 sea states, 20 seeds each, 600 records. There are
wind seas, swells, mixed wind sea + swell, and seas split over two directions.
They are defined in `SEA_STATES` in `WM/create_data.py`.

```sh
cd WM
python create_data.py --out sensor_data_states
python create_data.py --out sensor_data_states --status
python batch_invert.py --root sensor_data_states --positive
```

**One-at-a-time sweep:** the original 500-record sweep around the baseline sea.

```sh
cd WM
python create_data.py --mode sweep --out sensor_data_dense
python create_data.py --mode sweep --out sensor_data_dense --status
python batch_invert.py --root sensor_data_dense --positive
```

Repeat the generation command to resume. A progress bar shows the time left.
Each record saves its sea in `systems.csv`, one row per wave system
(`Hs,Tp,gamma,s,direction_rad`). `params.csv` holds the run settings and
`Hs_actual`.

Use a new output directory; keep the original Drive dataset.
Inversion results are saved inside the data directory.
Omit `--positive` to run the unconstrained inverse.

Before uploading, confirm all records are complete and check the inversion
output for failed runs. Upload the whole data directory, including its
parameters, completion flags and `generation.json`.
