# Six-pressure sea-surface estimator

This research snapshot reconstructs the **individual current-time waves over a surface patch** from six pressure histories. It combines a pressure-learned statistical prior with a causal, regularised linear filter. It is a controlled simulation result, not a validated real-submarine system.

**Start with [AI_HANDOVER.md](AI_HANDOVER.md)** for the full explanation, current learning checkpoint and exact core source listings in one uploadable file. [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) is the explanation without the source appendix. [PHONE_PROMPT.md](PHONE_PROMPT.md) is the prompt to give the next AI.

## What is here

| Location | Contents |
|---|---|
| `src/hybrid_pressure_live.py` | Initial pressure-only learning, filter calibration, current-time streaming |
| `src/pressure_prior.py` | Frequency-shape and directional-prior estimation from six pressures |
| `src/pressure_only_live.py` | Calibration storage and past/current-only pressure filter |
| `src/pressure_array.py` | Assumed six-port geometry, complex sensor response, and older comparison inverses |
| `src/submarine.py`, `src/sensors.py`, `src/time_domain.py`, `src/sea.py` | Explicit nominal physics and earlier synthetic generator |
| `src/results-joint/response_table.npz` | Nominal hull calibration supplied to the estimator |
| `tests/` | Existing 14 focused checks, with fixture paths made portable |
| `work/replay_saved_case.py` | Relearn from saved pressures and independently rescore the resulting surface |
| `work/` | Native external-wave adapter, build instructions, experiment/envelope runners |
| `work/CFDwavemaker/` | Vendored external library source, original upstream asset, commit and MIT license |
| `outputs/independent-wave-package/` | Four test records, calibrations, compact references, full metric JSON, offline viewers |
| `outputs/independent-depth/`, `outputs/wider-surface/` | Depth/area protocols, metric tables, maps and report |
| `PACKAGING_MANIFEST.json` | Source integrity and explicit packaging changes |

The main estimator sources are byte-for-byte copies of the tested implementation. No estimator tuning was performed for this handover. The checks and experiment launchers have portable paths; the original known error in a function docstring is documented in the guide rather than silently edited.

## Quick replay on macOS, Linux or Windows

Run from this folder with Python 3.11 or later and NumPy/SciPy installed:

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p '*_checks.py' -v
python work/replay_saved_case.py
python work/replay_saved_case.py --all
```

The replay defaults to the package-irregular case with phase seed 731927. It learns its prior from the first 1,024 pressure packets and streams the subsequent surface without using the reference. Only after estimation does it load the reference to score. Results go to ignored `outputs/phone-replay-results/`. Run `python work/verify_package.py` to check all packaged evidence against its manifest.

The stored reference fields use float32 to avoid committing duplicate, large prediction arrays. Their rounding is below 0.00000012 m; original full-precision metric JSON is retained. Replay allows 0.00000005 absolute difference in R² and RMSE to accommodate storage and platform arithmetic. This is a packaging precision tolerance, not a physical error allowance.

## Inspect the visual results

```sh
python -m http.server 8767 --bind 127.0.0.1
```

Open `http://127.0.0.1:8767/outputs/independent-wave-package/index.html` for the irregular sea, `gaussian.html` for the Gaussian sea, or `outputs/independent-depth/index.html` for depth/area comparisons. The two independent-sea viewers are standalone HTML and also open directly from disk. GitHub displays their source; it does not automatically host the interactive pages.

## Rebuild the independent generator

The archived build script targets **macOS with clang++**:

```sh
python work/build_cfdwavemaker_serial.py
python work/check_independent_adapter.py
python work/run_independent_wave_test.py
python work/map_independent_limits.py depth
python work/map_independent_limits.py space
python work/report_independent_limits.py
```

This regenerates and overwrites the corresponding output records. Keep a copy or use a separate checkout when comparing against the archived evidence. Native binaries are not shipped: replays from saved pressure records are portable, while rebuilding the native C++ adapter on another OS requires build/binding changes. This has not been tested on Linux or Windows.

## Main result and practical limit

At **15 m hull-centre depth**, no measurement noise, over **60 × 12 m / 403 surface locations**, the independent implementation checks give:

| Source | Pooled surface R² | Surface-height RMSE |
|---|---:|---:|
| Package irregular spectrum, two phase seeds | 0.994945–0.995318 | 3.44–3.55 cm |
| Gaussian single-peak spectrum, two phase seeds | 0.997393–0.997505 | 2.49–2.56 cm |

The Gaussian source does not have a JONSWAP frequency shape, but the inverse still uses a JONSWAP prior. Its biased fitted Hs values show that accurate surface reconstruction does not prove accurate recovery of every statistical parameter.

Shared linear wave/hull/instrument assumptions, zero sensor noise, known nominal calibration and assumed port positions make this a favourable test. Accuracy falls with depth and distance from the hull; pooled R² can hide weak surface locations. See the guide before making scientific claims.

## Phone and voice handover

Upload `AI_HANDOVER.md` to a new ChatGPT conversation in text mode and paste `PHONE_PROMPT.md`. Ask it to identify the files, parameters and current learning checkpoint it has read. Then open that same conversation on your phone and use voice where supported. The Markdown contains all eight core implementation files verbatim, so the teaching context does not depend on the voice model opening GitHub or running tools live. This does **not** verify voice/file behaviour for your particular account or app version.

An attached ZIP is available separately for tools that can unpack and run code. A GitHub URL by itself is only a reference: ask the next AI to confirm what it actually fetched rather than assuming it has the repository.
