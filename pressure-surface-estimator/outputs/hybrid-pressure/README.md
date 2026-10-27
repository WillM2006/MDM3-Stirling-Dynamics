# Hybrid pressure-only wave reconstruction research

15 October 2026. Research objective: combine the sea-state statistics and phase-preserving inverse to produce the best supported estimate of the actual surface above the submarine as depth changes. The 60 × 12 m rectangle is a comparison benchmark, not a limit on the research. Crossing seas have not been added.

**Working result:** the combined estimator improves fresh single-sea reconstruction at 15 m centre depth from pooled surface R² ~0.85 to ~0.993, and RMSE ~19 cm to ~4 cm over 60 × 12 m. This is a causal pressure-derived result, not an oracle result. It is not yet the best estimator at all depths; the noisy 60 m stress test exposes a failure.

Open `index.html` for the new comparison viewer. It replays one fresh 15 m noise-free record through the phase filter calibrated from that record's earlier pressures. The earlier flat-prior prototype remains preserved in `../pressure-only-live/`.

## What is combined

```
Earlier six-pressure samples → learned spectrum + direction + spread
                                           │
Known hull/port physics at current depth ───┤
Current/past six-pressure samples ──────────┤
                                           ↓
                          actual-wave surface estimate now
```

The statistical model supplies plausible allocation of energy among partly unobserved modes. The pressure samples supply the measured timing/phases. The output is not a new random sea realization drawn from a fitted spectrum.

The inverse accepts exactly one timestamp and six pressure readings per update, together with known nominal hull/sensor calibration. It receives no IMU, seed, generating Hs/Tp/direction/spreading, component phases, true sea surface or hidden motion states. The first 1024 pressure packets form the learning window; later samples are excluded from that fit. The learned prior is frozen for the subsequent stationary-sea test.

The learner assumes one separable JONSWAP frequency family and cos-2s directional family, with gamma fixed at 3.3. These are modeling assumptions, not independently measured facts. Direction, spreading, Hs and Tp are estimated from pressure data. This is not a spectrum-independent estimator, and a flexible/nonparametric alternative remains a research direction.

## Technical method

1. The top/bottom pair at identical horizontal coordinates has a pressure difference that cancels common first-order vehicle displacement. This supplies a scalar frequency-shape observation without requiring IMU or true surface data.
2. A finite-window expected-periodogram fit estimates Hs/Tp. It includes positive- and negative-frequency Hann-window leakage. This avoids interpreting short-window leakage as real low-frequency wave energy.
3. A complex matrix-Whittle working objective fits mean direction and spreading from the six-port relative phases/covariances, with response-only whitening and frequency-specific rank handling. The scalar spectrum is profiled as a nuisance in the angular fit. No confidence intervals are claimed.
4. The learned frequency/direction distribution calibrates a causal, 128-tap minimum-mean-square pressure filter. A 1% isotropic mixture and fixed `1e-8 m²` pressure-head regularization limit overconfidence/gain. These were fixed before the fresh validation. The regularizer is not a calibrated sensor-noise model.
5. The filter retains raw pressure timing and uses only past/current readings to estimate elevation on the target grid.

The provided hull response is recomputed separately for each depth, using the same known capsule physics. Depth is a calibration input, not an unknown estimated in this test. Sensor positions remain the explicitly assumed bow/stern/port/starboard/top/bottom arrangement, not verified sponsor hardware.

## Fresh validation

The method and parameters were frozen in `protocol.json` before fresh seeds 424243 and 424244 were generated. The same individual sea waves are used across depth conditions for controlled comparison. At each of 15, 30 and 60 m centre depth, both noise-free and hypothetical independent 1 Pa white pressure-noise cases are tested. Noise levels are not hardware specifications.

Each forward record comes from the separate time-domain RK4 vehicle simulator. Only pressures cross the estimator boundary. Hidden reference elevations are evaluated after streaming predictions. All generated frequencies, 0.0200195–0.3181152 Hz, enter the score. There are no 2-second waves or frequencies outside this band in these simulations.

Both the hybrid and earlier flat-prior baseline are scored at identical timestamps, after the first 1024 samples, on the same 403 surface-grid locations. No reference rescaling, time shifting or phase alignment is used. The sea family is Hs 2 m, Tp 8 s, gamma 3.3, mean direction 30°, spread 10; these values belong to the generator/evaluator only.

| Centre depth | Noise-free hybrid R² | Noise-free RMSE | Hybrid R² with hypothetical 1 Pa noise |
|---|---:|---:|---:|
| 15 m | 0.99307–0.99327 | 4.01–4.12 cm | 0.98898–0.98947 |
| 30 m | 0.97103–0.97109 | 8.19–8.56 cm | 0.95383–0.95386 |
| 60 m | 0.87941–0.87975 | 16.71–17.46 cm | 0.63691–0.77900 |

At 15 m, the worst individual grid point has R² ~0.9703–0.9710 and RMSE ~8.5–8.7 cm, even though pooled R² is ~0.993. Almost-perfect recovery at every point is not established.

At 60 m with noise, both prior fits reach the upper Hs bound and infer biased directions. One hybrid run is worse than its flat-prior baseline (R² 0.637 versus 0.753). This is a measured failure of the present learner/noise treatment, not proof that no better method exists. Likewise, the noiseless depth trend includes the effect of the fixed regularization: these scores are not optimal physical depth limits.

Raw results, learned parameters, timings and source hashes are in `results.json` and `protocol.json`. Spatial R² maps are saved per run. Two fresh phase realizations per depth/fixed sea family are limited evidence; other peak periods, directions, spreading, gamma, nonstationary conditions, hull errors and real data remain unvalidated.

## Timing and viewer boundary

The prior-learning history spans 511.5 s, about 8.5 minutes. Initial fitting and filter construction take ~2.6 s on this machine. The synchronous prototype pauses for that setup; the first fitted output includes this startup computation. It is not a seamless background learner.

After setup, typical current-time surface updates take ~0.064 ms in Python, with p99 generally ~0.11–0.14 ms, versus a 500 ms measurement interval. Hardware acquisition and background prior updates have not been connected. The prior is frozen after startup, so adaptation to a changing sea is not yet implemented.

The browser begins with the already learned calibration and primes its phase history from past samples. Its first valid frame is at pressure time 511.5 s; it does not refit the statistical prior in JavaScript. An isolated worker receives only calibration and pressure packets, while hidden surface data go solely to comparison drawing. Browser outputs are checked against Python, including reference-visibility independence, startup, playback and responsive layout.

## Why depth matters

In the current linear deep-water model, incident dynamic pressure at sensor depth d scales as `exp(-k d)`, with `k = (2πf)²/g`. Short waves decay much faster with depth than long waves. A statistically plausible high-frequency tail is different from recovery of the actual high-frequency phases. At large depths, finite sensor noise and calibration uncertainty can overwhelm those phases. [MIT water-wave notes](https://ocw.mit.edu/courses/2-017j-design-of-electromechanical-robotic-systems-fall-2009/51e6a1ab5a7fa9937bdb7aca645897fd_MIT2_017JF09_ch06.pdf).

The research objective is therefore an estimate and quantified reliability as functions of depth, position and frequency. It should not promise identical accuracy at every depth.

## Next research corrections

- Fit a noise covariance/floor jointly with the sea prior, and use it consistently in the phase inverse. The existing noisy deep-water failure makes this the immediate priority.
- Report spatial/frequency uncertainty and flag prior-boundary fits; current outputs do not provide calibrated uncertainty bands.
- Compare parametric and flexible directional/frequency priors, choosing/model-averaging from pressure evidence rather than generating truth.
- Update the prior asynchronously as the sea changes, preserving the same causal input boundary.
- Test depth/pose uncertainty, actual sensor positions, hull-scattered pressure, speed and replacement hydrodynamic response models.

These are research directions, not completed capabilities. Bayesian directional-spectrum estimation from measured responses has precedent, but does not itself recover individual wave phases: [Iseki and Ohtsu](https://www.sciencedirect.com/science/article/pii/S0967066199001562).
