# Independent wave-package test, without measurement noise

15 October 2026. **The frozen pressure-only hybrid passes this independent-implementation test.** CFDwavemaker generates both the reference surface and incident underwater pressures, replacing our `Sea` generator and pressure-wave synthesis. The estimator was not changed or tuned after seeing these cases.

Open `index.html` for the package irregular sea and `gaussian.html` for the Gaussian sea. Each page recomputes estimated frames from six pressure packets using the filter learned from earlier pressures. Reference surface data are used only for comparison drawing.

## Results

Full 60 × 12 m rectangle, all 403 grid points, submarine centre depth 15 m, prescribed component-variance Hs 2 m. No added measurement noise and no crossing seas. Two fresh random phase draws are paired across the two spectral shapes.

| Independent wave input | Surface R², both phase runs | RMS elevation error | Lowest pointwise R² |
|---|---:|---:|---:|
| CFDwavemaker supplied irregular spectral shape | 0.994945–0.995318 | 3.44–3.55 cm | 0.972386 |
| Gaussian single-peak spectrum | 0.997393–0.997505 | 2.49–2.56 cm | 0.990208 |

The package irregular runs have maximum pointwise RMSE 8.16–8.27 cm; Gaussian runs 4.71–4.93 cm. Pooled R² is not a guarantee of identical accuracy everywhere.

The simpler flat-prior pressure inverse scores R² 0.8649–0.8670 on the package irregular cases and 0.9068–0.9089 on the Gaussian cases, using exactly the same readings and scored timestamps. The earlier ~0.993 matched-generator score is a separate reference: the new package has a different component count, directional discretization and narrower frequency band, so the slightly higher scores do not establish a general accuracy improvement.

The Gaussian case is a useful model-family mismatch: its true frequency shape is not JONSWAP, while the unchanged inverse still assumes JONSWAP with gamma 3.3. A good surface score does not imply recovery of correct statistical parameters. For example, the Gaussian prior Hs fits are 1.40 and 1.69 m despite prescribed Hs 2 m. Raw pressures can still yield accurate surface amplitudes even when the prior's fitted parameters are biased.

## Independent package and setup

[CFDwavemaker](https://github.com/oystelan/CFDwavemaker), version 3.7.13, commit `23c66197488d680c8c1b96d8aa9e023fefa361ba`. It is a wave-kinematics library, not itself a CFD solver. These tests use its **linear irregular-wave** mode and native `wave_SurfElev` / `wave_DynPres` APIs at every time and surface/port location. No reference surface is synthesized by our old `Sea` class or `truth_at` function.

The component frequencies, relative spectral amplitudes and directions come from the package's `tests/t008_3D_irregular_sea/waveinput_linear.dat` asset. Adapter transformations are explicit:

- Rotate all directions by +30° and scale amplitudes to prescribed component-variance Hs 2 m, preserving the upstream spectral/directional arrangement.
- Draw new independent uniform component phases with seeds 731927 and 731928. These are generator-only inputs.
- In the Gaussian comparison, replace amplitudes with a single Gaussian frequency-density shape centred at 0.125 Hz, standard deviation 0.018 Hz, then normalize to Hs 2 m. The adapter prescribes these amplitudes; CFDwavemaker evaluates the resulting wave field and incident pressures.
- Use water depth 1000 m, effectively deep for the included waves. The external library solves its dispersion relation itself. Water depth and submarine centre depth are different quantities.

There are 312 components, with frequencies **0.0843744–0.2787503 Hz** and continuous off-grid directions. These frequencies are not rounded to the estimator's Fourier bins; the forward drive accepts arbitrary sinusoidal frequencies. The full generated surface, including all components, is scored. No evaluation band cutoff, time shift, truth phase alignment, or best-fit rescaling is used.

The serial macOS build makes one non-physics portability edit: guard an otherwise unused `omp.h` include with `_OPENMP`. OpenMP, SWD and VTK extensions are not enabled. No wave equations were modified. Source commit, input/source/binary hashes and the patch description are saved in `protocol.json`.

## What remains shared

This is independent wave/incident-pressure code under shared linear theory. It is not a fully independent submarine/instrument simulator:

- The same approximate capsule strip-force law, mass, added mass, damping and restoring coefficients drive the vehicle.
- The vehicle is integrated from rest with RK4, a 60 s forcing ramp and 400 s settling. Motion generation does not use an inverse RAO.
- External incident pressures at nominal port locations are combined with the existing first-order hydrostatic correction for heave/roll/pitch.
- The inverse is given the same known nominal hull calibration and assumed six-port positions. Hull-scattered pressure, forward speed, instrument filters and hardware errors are not introduced.

All four records remain below the illustrative 5° small-angle screen (peak rotation 1.95–3.10°). That is a consistency screen, not real-world validation.

## Frozen estimator and timing

All original hybrid source hashes match the previous depth experiment. The prior learns only from the first 1024 pressure packets (511.5 s at 2 Hz), then remains frozen for the subsequent stationary record. Each filter output uses only current and past readings. It receives no IMU, component list, seed, true surface, generating parameters or future readings.

Each record contains 8192 samples (~68 min). Scores start at packet 1024 and span the remaining ~60 minutes. Initial statistical fitting/filter setup takes ~2.6–2.8 s; ongoing updates have ~0.064 ms median compute and ~0.12 ms p99 on this machine. Hardware acquisition and seamless background prior updates are not tested.

## Evidence and checks

- `protocol.json`: frozen estimator hashes, package commit/binaries, adapter hash, cases and shared assumptions; written before the test records were generated.
- `results.json`: every run, learned priors, metrics, timings, rotation screen and record hashes. Frozen-method integrity passes.
- `adapter-checks.json`: independent two-tone tests of the native surface phase/direction convention, pressure units/density/depth attenuation, and arbitrary-frequency strip-load equivalence. Maximum discrepancies are ~4e-16 m and ~1e-11 Pa.
- `cases/*/waveinput.dat`: exact external inputs, alongside pressure-only CSVs and separate evaluator-only component metadata.
- `cases/*/evaluation.npz`: estimates, independent reference surface, pointwise R² and timestamps.
- `browser-checks.json`: comparison of viewer streaming outputs with Python, controls, reference visibility, and desktop/mobile layout.

This answers the independent-package/noise-free question for the tested linear seas. A genuinely stronger physical test would add independent nonlinear wave kinematics and/or a different hydrodynamic/sensor model. Neither is claimed here.

If CFDwavemaker is used in later research reporting, cite its authors' paper: Lande and Helmers (2022), *CFDwavemaker: An Open-Source Library for Efficient Generation of Higher Order Wave Kinematics*, OMAE2022, V007T08A002. [Package and citation](https://github.com/oystelan/CFDwavemaker).
