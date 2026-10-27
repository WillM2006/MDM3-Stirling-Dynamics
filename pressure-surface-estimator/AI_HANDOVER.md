# AI handover: the high-R² six-pressure surface estimator

This is a self-contained text attachment for a new AI tutor. Read the implementation guide first, then inspect the exact source appendix as needed. Karan wants to rebuild and write about this implementation, with one concept taught at a time. Do not launch into the entire algorithm at once.

The next teaching checkpoint is the windowed Fourier spectrum of the demeaned bottom-minus-top pressure difference. The prior learner actually fits frequency shape before direction, despite an outdated source docstring. Keep that distinction clear.

All eight core implementation files are included verbatim below. Supporting replay/evaluation/forward-adapter source is also included. Raw pressure/reference/calibration data, native external library source and all result JSON are in the repository snapshot, not embedded as unreadable binary data here. You can explain the code from this text, but do not claim to have run a replay or read binary data unless you actually have.

Repository: https://github.com/WillM2006/MDM3-Stirling-Dynamics/tree/pressure-surface-estimator/pressure-surface-estimator

## Tutoring prompt and current user preference

# Prompt for a new text/voice tutoring chat

I am Karan, working on the MDM3 Stirling Dynamics sea-surface inverse model. Read the attached AI_HANDOVER.md before teaching. It contains the detailed implementation guide and exact source listings of the eight core files. If you can access the repository, inspect its pressure-surface-estimator branch and supporting data/results too. Tell me what you actually read and what you cannot access; a GitHub link is not proof you have loaded the files.

My goal is to understand our actual high-R² estimator deeply enough to rebuild it and write a paper. Teach one concept at a time, with physical intuition, a small example/diagram, then the equation and relevant source. Ask a short comprehension question and wait. Do not dump the whole pipeline or read code aloud line by line.

The current method learns a JONSWAP/cos-2s prior from the first 1,024 six-pressure packets, then applies a 128-tap causal regularised Wiener filter. It receives no IMU, generator phases, hidden surface/motion states or future readings. Known nominal hull/depth/assumed port geometry are supplied. Its strong scores are controlled noise-free linear-wave simulation results, not real-world accuracy guarantees. We have not implemented a Kalman filter.

We last covered subtracting demeaned bottom and top pressures to cancel common first-order hull motion while retaining their unequal wave attenuation. Continue with how the windowed Fourier transform reveals frequencies in that difference signal; then explain why pressure-difference power is not yet surface-wave power. Do not skip straight to direction matrices or filter calibration.

Before we switch to voice, confirm the sample rate, learning-window length, filter taps, sensor geometry, the actual order of fitting in pressure_prior.py, and the distinction between pooled and worst-location R². Summarise these briefly so I can verify you have the right implementation. Then wait for my questions.

---

# Implementation guide: pressure-learned, phase-preserving sea-surface reconstruction

Prepared 16 October 2026. This describes the **existing tested implementation**, not a proposed ideal algorithm. Read the actual source appendix in `AI_HANDOVER.md` alongside the equations. The user is Karan, researching the inverse model for the University of Bristol MDM3 Stirling Dynamics project.

## 1. Objective and distinctions that must remain clear

The objective is to estimate the **actual instantaneous surface elevation** `eta(x,y,t)` above a submarine from pressure measurements. The individual crests and troughs should match the generating surface. A plausible-looking random sea with the same spectrum does not satisfy this goal.

The current estimator receives **six pressure readings and a timestamp per packet**, plus known nominal hull calibration, port geometry, gravity, water density and depth. No IMU readings, generator phases, true surface, hidden vehicle state or future sensor samples are accepted by its public streaming interface. A 60 × 12 m patch is a benchmark, not a limit on the user's research ambition. Crossing seas are excluded for now. Current independent-source comparisons have no measurement noise.

Distinguish three objects:

1. **A statistical spectrum:** expected variance by frequency and direction. JONSWAP describes a frequency-energy shape, not actual crest locations.
2. **A realised wave field:** particular amplitudes/phases evolving across space and time. This is what we want to reconstruct.
3. **Measurements:** wave pressure mixed with motion-induced pressure changes at six known ports.

The model combines statistics and measurement timing through a causal linear estimator. It is not a neural network, not ordinary bilinear interpolation, not a full-record Fourier smoother, and not an explicit live fit of every component's phase. Spatial wave information is used implicitly in the model-derived filter weights.

## 2. How the work evolved

Earlier experiments fitted Hs/Tp and then mean direction/spreading from pressure and IMU spectra. A good spectral-fit R² established agreement of statistics, not a reconstruction of original waves.

Later six-pressure plus IMU, full-record methods reconstructed surface patches. Those results use different input/timing assumptions and must not be presented as proof of the current pressure-only causal workflow. Some older functions remain in `pressure_array.py` because they were comparison baselines and are tested separately.

The flat-prior causal pressure-only method in `pressure_only_live.calibrate` used equal directional/frequency variance. Over the full 60 × 12 m patch it was much weaker than at the centre. That motivated a pressure-learned prior: use observed sea statistics to help resolve the limited spatial information.

The resulting **hybrid** learns a JONSWAP/cos-2s prior from past pressures, then uses a causal phase-preserving pressure filter. It originally achieved about 0.993 pooled surface R² in a matched linear synthetic sea at 15 m depth. Its exact source was frozen and tested against a separate native wave-kinematics library. Those are the 0.995–0.998 independent-implementation results in this package. Higher scores on a different sea are not evidence of a universal improvement: component count, frequency support and angular sampling differ.

## 3. Geometry, frames and sampling

The assumed capsule is 70 m tip-to-tip and has radius 5 m. Coordinates are relative to the geometric hull centre: +x along the hull, +z upward. The six modelled ports are:

| Port | x (m) | y (m) | z (m) | Depth when centre is 15 m |
|---|---:|---:|---:|---:|
| bow | 35 | 0 | 0 | 15 m |
| stern | -35 | 0 | 0 | 15 m |
| port | 0 | 5 | 0 | 15 m |
| starboard | 0 | -5 | 0 | 15 m |
| top | 0 | 0 | 5 | 10 m |
| bottom | 0 | 0 | -5 | 20 m |

Six sensors are user-reported; these positions are a user-authorised **modelling assumption**, not sponsor-confirmed hardware coordinates. Top/bottom share xy, so there are only five unique horizontal locations. This vertical pair is important to the current statistical learner.

Sampling is 2 Hz (`dt=0.5 s`). There are 8,192 packets per record. The initial 1,024 packets span timestamps 0–511.5 s. The finite filter has 128 taps: current packet plus 127 previous packets, a 63.5 s span, often called approximately 64 seconds of history. Scoring starts at zero-based index 1023, when the prior first becomes available, and includes 7,169 current-time estimates. That first scored timestamp is also the last fitting-window sample; most scored samples are later, but this is not a strictly disjoint training/evaluation split at the boundary.

The original target grid has x=-30,-28,...,30 and y=-6,-5,...,6, giving 31 × 13 = 403 locations. It is a complete rectangle, not merely the convex hull of port positions.

## 4. Known physics: how one component reaches a sensor

Use a complex component convention with time factor `exp(-i omega t)`. For a unit surface component, its spatial factor at xy is

`E(x,y;omega,theta) = exp(i k (x cos(theta) + y sin(theta)))`, where `k = omega²/g`.

Within the deep-water linear-wave approximation, the incident pressure response at port j is

`H_inc,j = rho g exp(-k d_j) exp(i k (x_j cos(theta)+y_j sin(theta)))`.

Here `d_j = centre_depth - z_j`. Different depths attenuate frequencies differently; short waves attenuate much more strongly. At fixed xy, this linear incident-pressure relation has the same oscillatory phase as surface elevation. This is the incompressible linear model, not a claim of unlimited physical signal speed.

The submarine moves. At first order its port's vertical displacement is

`delta_z_j = q_heave + q_roll*y_j - q_pitch*x_j`.

With complex nominal motion responses q per unit wave amplitude, the measured wave-frequency pressure response is

`H_p,j = H_inc,j - rho g delta_z_j`.

This entire pressure response, including phase shifts due to hull dynamics, comes from `pressure_array.array_response`. Although that utility also computes six IMU responses for historical comparisons, the current learner/filter take only its first six pressure channels: `[..., :6]`. No IMU measurement enters.

The nominal hull has illustrative mass, added mass, damping and restoring matrices. Its frequency response solves `[-omega²(M+A) - i omega B + C] q = F_exc`. `submarine.py` explicitly labels the capsule strip-force/added-mass/inertia approximations as temporary research surrogates. It is not a calibrated real-vessel hydrodynamic model. The response table is supplied calibration, not learned solely from sea pressures.

## 5. Statistical learning, stage A: top/bottom pressure difference

The current teaching checkpoint is here. Do not jump straight to matrices when tutoring.

Each port has a large mean hydrostatic pressure. The learner subtracts each channel's own sample mean. It applies a periodic Hann window and takes an rFFT. For each retained bin it stores a complex coefficient with the chosen physical convention and one-sided PSD scaling:

`z = conj(rfft((p - mean(p))*window)) * sqrt(2/(fs*sum(window²)))`.

The actual code transforms all six channels first and then subtracts the top/bottom complex coefficients. This is algebraically equivalent to transforming their demeaned difference with the same window, because the transform is linear.

For the top/bottom pair, x=y=0 for both ports. First-order heave produces the same `-rho g q_heave` term at each; roll/pitch vertical-displacement terms vanish there. Subtracting bottom minus top therefore cancels common first-order motion while retaining different depth attenuation of incident wave pressure. It does not remove all possible real-vessel motion/scattering/nonlinear effects.

For each frequency, the difference response is

`G(omega) = rho g [exp(-k d_bottom) - exp(-k d_top)]`.

The learner uses the **power** `I_diff = abs(z_bottom-z_top)²`, so swapping the subtraction sign does not change the fit. It is still pressure-difference power, not directly a surface-elevation spectrum. Converting it requires the known response G and finite-record spectral effects.

## 6. Statistical learning, stage B: expected periodogram and frequency shape

We do not simply divide a noisy periodogram by `G²` at every frequency and call the result truth. A finite, windowed record smears neighbouring frequencies together. The implementation analytically models the periodic Hann transform using Dirichlet kernels, including positive- and negative-frequency contributions.

Let `K_bf` be the resulting nonnegative window-leakage kernel from model frequency f into observed Fourier bin b, including `G(f)²`. For a trial Tp, predicted difference power for unit Hs is

`v_b(Tp) = sum_f K_bf * JONSWAP(f; Hs=1, Tp, gamma=3.3)`.

Observed fitting bins are 0.06–0.26 Hz. The response grid itself is wider (roughly 0.0200–0.3181 Hz). JONSWAP in `sea.py` uses standard two-sided peak widths 0.07/0.09, with finite-band normalisation so `sum S(f) df = (Hs/4)²`. Gamma is **fixed at 3.3**, not estimated.

Because spectral power scales as Hs², define `r = mean_b(I_diff,b/v_b)`. The profiled height is `Hs² = clip(r, 0.2², 5²)`. The fitted scalar objective is

`mean(log(v_b)) + log(Hs²) + r/Hs²`.

The code searches Tp on a 141-point grid from 5 to 12 s, then uses a bounded scalar refinement around the best candidate. This is a working Whittle-style spectral fit to a finite-window expected periodogram; it is not an exact likelihood for a short record with independent Fourier bins. Confidence intervals are not computed.

## 7. Statistical learning, stage C: mean direction and spreading

One vertical pair cannot identify wave direction. Direction information comes from the **relative complex responses across all six ports**, including horizontally separated ports and the known hull response.

The assumed single-system directional distribution is proportional to

`D(theta; mu,s) = max(cos(wrap(theta-mu)/2),0)^(2s)`.

`mu` is mean wave travel direction; larger `s` is narrower spreading. Angles are travel directions in our model frame, not automatically meteorological "coming-from" headings. The implementation normalises probabilities across 36 evenly spaced full-circle directions. It does not independently fit 36 arbitrary angular weights, and it does not estimate the actual phase of each angular component at this stage. A frequency-independent directional shape is an assumption.

For each candidate direction distribution the model predicts the complex six-pressure covariance from outer products `H H*`, weighted by the fitted frequency shape and the window kernel. Off-diagonal entries encode relative phase as well as correlated power. This is what power-only single-port fitting loses.

For numerical stability, the learner rescales channels and whitens in an isotropic model covariance basis. It keeps eigenmodes above a relative threshold of 1e-7 and groups frequency bins by retained rank. Bins with vanishing observed power are screened out. A scalar spectral amplitude is profiled independently per observed bin for the angular objective, rather than forcing direction to explain every frequency-amplitude error.

In a retained rank-r basis, the angular objective contains

`log det(K_b) + r log[(z_b* K_b^(-1) z_b)/r]`,

summed over bins and divided by the total retained dimension. Cholesky failure or nonpositive quadratic power rejects a candidate. Coarse search uses mu at 15° spacing over the full circle and s in {1,3,8,20,50}. Powell refinement starts from the four best coarse fits, with local mu bounds and `1 <= s <= 50`. The best result supplies mean direction and spreading. Boundary flags are diagnostics, not uncertainty estimates.

**Frozen-source documentation mismatch:** `estimate_prior`'s docstring says direction/spread are fitted before frequency shape. The executed code fits Tp/Hs first, then direction/spread. This guide follows the executed code. The source was preserved to retain experiment hashes.

## 8. How the learned prior becomes a surface estimator

`PressurePrior.weights` builds a separable frequency/direction weight array. For the filter, the directional distribution includes 1% uniform probability and 99% fitted directional probability. This keeps weak nonpreferred directions represented; it does not prove their phases are observable.

The frequency spectrum is normalised by its sum. **Consequently fitted Hs cancels out of the filter's relative covariance weights.** Tp, direction and spreading change those weights; raw measured pressure amplitude supplies reconstructed amplitude. This explains why a biased fitted Hs can coexist with a good surface reconstruction. The Gaussian test fits Hs around 1.40/1.69 m despite prescribed component-variance Hs=2 m.

The code computes the sensor response in pressure-head units `h = H_p/(rho g)`. Weighted products `h h*`, integrated over model frequencies/directions and shifted through lags, supply a 768 × 768 covariance matrix R: 128 time lags × six ports.

For each requested surface point, the spatial factor E above gives a surface-to-pressure cross-covariance vector c with those same 768 pressure-history entries. The model then solves a regularised finite-history linear estimation problem. Ignoring the constraints for one line:

`w = (R + lambda I)^(-1) c`, with `lambda = 1e-8`.

The code uses a positive-definite linear solver rather than forming the inverse. Six exact constraints require weights for each individual sensor to sum to zero over time. A covariance-weighted projection enforces these constraints; a final small sum correction reduces numerical residual. Thus constant per-channel pressure offsets do not create a surface signal once the filter history is full.

The prior covariance has normalised surface variance. Lambda is numerical regularisation in pressure-head covariance units, not a fitted or measured sensor-noise model. It limits unstable gain even in the noise-free experiments. Calling this "best possible" without qualifying the assumed statistics, finite history, constraints and regularisation would be incorrect.

The final weights are divided by `rho g` so the public stream accepts pressure in **pascals** and outputs elevation in **metres**. Shapes are `[number_of_surface_points, 128, 6]`. Saved calibration also contains target xy, each port's nominal hydrostatic baseline, sample interval and metadata.

## 9. Actual online execution and causality

`HybridPressureLive.push(time_s, six_pressures_Pa)` validates channel count, finite values and uniform ordered timestamps. During startup it stores the first 1,024 packets. On packet 1,024 it synchronously fits the prior, calibrates the filter, and primes `PressureOnlyLive` with the final 128 packets from that already received history.

The first surface estimate corresponds to that current packet. The additional startup calculation took about 2.6 s for 403 points in the earlier local benchmark, so claiming zero hardware/display latency would be wrong. Subsequent calls use the current packet plus past pressure history; there is no intentional prediction-time offset and no future input.

For surface point p at packet n, actual execution is

`eta_hat_p[n] = sum_lag=0..127 sum_j=0..5 W[p,lag,j] * (pressure_j[n-lag] - baseline_j)`.

The prior stays fixed after startup for these stationary-sea records. There is no periodic adaptation, changing-sea detector, explicit live component-state Kalman filter or calibrated posterior uncertainty in the current implementation. Earlier measured update times (~0.064 ms median for 403 points) are local prototype timings, not a hardware real-time guarantee. The browser replay uses saved pressure-learned weights; it does not refit statistics in the browser.

## 10. What spatial extrapolation means here

For a distant target, `calibrate_hybrid` changes E(x,y) and recomputes its cross-covariance with the same pressure histories. Each point gets different weights. We do not linearly stretch the rendered image or interpolate six scalar surface heights.

The estimator uses known dispersion, relative phase constraints and assumed statistics to choose a low-average-error surface consistent with limited measurements. It need not explicitly identify one amplitude/phase for every modelled frequency-direction component. Different underlying wave mixtures can be weakly distinguishable at the ports but diverge elsewhere.

Errors in direction lead to greater spatial phase error at greater distance, particularly for short waves. Past measurements can constrain downstream regions differently from upstream regions. Adding more plotted pixels adds no measurements. Enlarging the calibration target set leaves existing point filters unchanged; the spatial experiment verified that original predictions and native surface values reproduce to numerical precision before packaging.

The next research comparison could use longer causal memory, a more flexible learned directional spectrum or a sequential wave-state estimator with uncertainty. Kalman/ensemble methods are legitimate candidates, but are not automatically superior under equivalent linear-Gaussian assumptions and cannot create unmeasured information. No such improvement was implemented in this snapshot.

## 11. Independent forward source and what remains shared

The external source is **CFDwavemaker 3.7.13**, commit `23c66197488d680c8c1b96d8aa9e023fefa361ba`. This is a wave-kinematics library, not a full CFD solver. Our tests use its linear irregular Airy mode, 1,000 m basin depth (distinct from submarine depth), density 1,025 kg/m³ and g=9.81 m/s².

The adapter takes 312 components from an upstream asset, rotates directions +30°, scales component variance to prescribed Hs=2 m and draws uniform random phases for seeds 731927/731928. A second case replaces component amplitudes with a Gaussian frequency shape centred at 0.125 Hz with sigma 0.018 Hz. There is one directional sea system, no crossing sea. Frequencies span 0.0843744–0.2787503 Hz and are not rounded onto the estimator grid.

Native `wave_SurfElev` and `wave_DynPres` evaluate the reference surface and incident pressures. The adapter's C++ batch wrapper merely reduces Python call overhead; its output is still evaluated by the external library. Adapter checks verify phase/direction conventions, density/pressure units and exponential attenuation against two-tone formulas.

Forward hull forcing uses the existing capsule strip-force law at the native components' frequencies/directions. Coupled RK4 motion is integrated from rest, with dt=0.1 s, a 60 s force ramp and 400 s settling before recording. The forward motion does not use the inverse response table. Absolute port pressures then include native incident pressure plus the first-order hydrostatic motion term. The inverse receives only the resulting six-pressure CSV.

The surface evaluator uses physical native time `pressure_record_relative_time + 400 s`. Forgetting this offset would give wrong reference phases. The hidden component file and hidden vehicle states are generator/evaluator inputs only; they are not accepted by the inverse.

Different wave and pressure implementations provide a stronger test than using our original `Sea` class for both sides. They still share linear potential-wave physics, nominal hull loading/dynamics and instrument approximations. There is no validated nonlinear wave-hull scattering, forward speed/current, sensor noise/quantisation, installation error, sea evolution or real-data validation.

## 12. Metrics and actual measured results

For each point p, define `MSE_p = mean_t((eta_hat-eta_true)²)` and `V_p = mean_t((eta_true-mean_t eta_true)²)`.

`R2_p = 1 - MSE_p/V_p`.

Pooled surface R² is `1 - sum_t,p(error²)/sum_t,p((truth-local_time_mean)²)`. It is not an average of per-frame R², not a frequency-spectrum score and not simply correlation squared. Pooled RMSE is the square root of mean squared height error over points and times. We report minimum pointwise R² because high pooled accuracy can hide weak locations. R² is a fit statistic, not a probability or a literal percentage of waves exactly recovered.

Four independent-implementation tests at 15 m centre depth:

| Spectrum / seed | Pooled R² | RMSE (m) | Minimum point R² |
|---|---:|---:|---:|
| Package irregular / 731927 | 0.994945447 | 0.035539776 | 0.972386076 |
| Package irregular / 731928 | 0.995318263 | 0.034356892 | 0.973871837 |
| Gaussian / 731927 | 0.997392969 | 0.025607355 | 0.990207666 |
| Gaussian / 731928 | 0.997505192 | 0.024916501 | 0.991194448 |

The flat-prior baseline on identical scored samples gives roughly 0.865–0.867 for package-irregular and 0.907–0.909 for Gaussian. This supports the usefulness of the pressure-learned prior for these cases, not a universal ranking of inverse methods.

Earlier matched-generator tests also included hypothetical independent 1 Pa white pressure noise at 15/30/60 m. A noisy 60 m case performed worse than the flat-prior baseline and its fitted height reached a bound. The learner does not fit noise covariance. Those stress results remain in `outputs/hybrid-pressure/results.json`; they must not be confused with the later noise-free independent-source sweeps or taken as calibrated real-sensor noise specifications.

Depth sweep: same four sea realisations and phases, same 60 × 12 m patch, prior relearned from each depth's initial pressures, known nominal calibration provided at that depth, no measurement noise. Ranges across four cases:

| Hull-centre depth | Pooled R² | Height RMSE (cm) | Minimum point R² range |
|---|---:|---:|---:|
| 15 m | 0.995–0.998 | 2.5–3.6 | 0.972–0.991 |
| 30 m | 0.982–0.997 | 2.5–6.7 | 0.954–0.992 |
| 45 m | 0.945–0.996 | 3.1–11.8 | 0.906–0.988 |
| 60 m | 0.895–0.992 | 4.6–16.3 | 0.847–0.977 |
| 80 m | 0.811–0.973 | 8.1–21.8 | 0.761–0.945 |
| 100 m | 0.726–0.921 | 14.0–26.3 | 0.675–0.871 |
| 150 m | 0.422–0.571 | 32.9–38.0 | 0.350–0.499 |

This is an envelope of the current regularised estimator, not a universal physical depth cutoff. Weak short-wave pressure signatures, finite-history inversion and statistical-fit errors all matter. Noise-free mathematics does not mean numerical stabilisation is absent.

Spatial sweep: same pressures and learned priors at 15 m centre depth, requesting additional surface locations without retuning the estimator. Centred, hull-aligned rectangles:

| Surface area | Pooled R² | Height RMSE (cm) | Minimum point R² range |
|---|---:|---:|---:|
| 60 × 12 m | 0.995–0.998 | 2.5–3.6 | 0.972–0.991 |
| 80 × 20 m | 0.986–0.995 | 3.7–5.9 | 0.910–0.977 |
| 100 × 40 m | 0.962–0.987 | 5.6–9.7 | 0.772–0.922 |
| 140 × 60 m | 0.910–0.964 | 9.4–15.0 | 0.585–0.759 |
| 200 × 100 m | 0.800–0.893 | 16.4–22.4 | 0.272–0.387 |
| 300 × 150 m | 0.634–0.738 | 25.6–30.4 | 0.012–0.065 |

The original patch retains its 2 m × 1 m grid. Larger patches use a 5 m grid. Maps evaluate the union of that large grid and the original dense grid. These tests do not establish accuracy between sample points. Depth and spatial sweeps were separate, not a joint claim for a large patch at large depth. The envelope sweeps reuse the four existing realisations and are exploratory paired comparisons, not fresh validation of a subsequently selected operating envelope.

## 13. Reproducibility and evidence limits

Original source hashes and all original pressure CSVs were checked unchanged. `PACKAGING_MANIFEST.json` records these hashes and the packaging adaptations. The core estimator, nominal physics and prior are preserved byte for byte. Test fixture paths and runner paths were made relative, and external Git metadata lookup uses a pinned vendored commit file. These launcher changes do not retune the estimator.

Every test case contains all 8,192 six-pressure samples, its pressure-learned calibration, exact native wave input and evaluator-only component metadata. Compact reference files retain all original scored truth samples as float32; their maximum storage rounding is recorded. Original metric JSON comes from double-precision reference calculations. Large duplicate predictions and repeatable depth pressure/calibration arrays are omitted from Git to keep the snapshot manageable. Replays recompute predictions, and source/inputs reproduce the omitted experiments on the documented native build platform.

The saved-case replay is a meaningful regression: it relearns the prior from pressure history, streams all packets, loads truth only for scoring afterward, and checks agreement with all original summary metrics within declared storage/platform tolerance. Existing focused checks cover causal input boundaries, timestamp/channel rejection, DC offsets, expected phase response and synthetic forward/inverse consistency. Some array checks exercise older IMU-based utilities; they are not evidence that the current estimator uses IMU.

Do not claim real-world identifiability from high synthetic R². Do not claim all directional components/phases are individually measured. Do not call the prior spectrum an entire realistic sea. Do not claim a Kalman implementation exists in this snapshot. Do not conflate earlier spectral/IMU/whole-record results with this current-time, six-pressure-only goal.

## 14. How to continue tutoring Karan

Karan wants enough depth to rebuild the estimator and eventually write a paper, but **one concept at a time**. Start with physical intuition, a small diagram or numeric example, then introduce the equation and the actual source lines. Ask a short comprehension question and wait. Distinguish assumed, fitted and measured quantities every time.

The last completed explanation was: demean each port, subtract bottom minus top to cancel common first-order motion while retaining unequal wave attenuation. The next lesson is **how the Hann-windowed Fourier transform reveals frequency content in that difference signal**, then why its spectrum is not yet a surface spectrum. Only after this should you cover the response-corrected JONSWAP fit, angular cross-spectra, covariance-based filter design and spatial extrapolation.

Explain the incorrect docstring ordering if relevant. Tell Karan that Hs cancels from filter-weight normalisation rather than suggesting the prior determines exact reconstructed amplitude. Keep existing results conditional on their experimental assumptions. When interpreting a plot or result, cite the corresponding packaged JSON/source, and say when a claim is a research hypothesis rather than a measured result.

## 15. Research references and their role

- [CFDwavemaker source](https://github.com/oystelan/CFDwavemaker): external implementation used for the independent wave/pressure test; pinned version in this package.
- [MIT wave-theory notes](https://ocw.mit.edu/courses/2-017j-design-of-electromechanical-robotic-systems-fall-2009/41e4cc0e5a0ebd845ea4f5e7c055fdf7_MIT2_017JF09_coursetext.pdf): linear pressure attenuation and approximate wave/body loading.
- [Qi et al., 2018: predictable zones](https://www.sciencedirect.com/science/article/pii/S0165212517301518): phase-resolved spatial/temporal predictability depends on measurements and spectral support. Its theoretical zones have not been derived quantitatively for our moving pressure ports.
- [Fisher et al., 2021: sparse buoy arrays](https://faculty.washington.edu/jmt3rd/Publications/Fisher_etal_2021_wave_reconstruction.pdf): related linear inverse wave reconstruction and measurement-footprint limits; different sensor configuration from ours.
- [Wang and Pan: ensemble-based wave assimilation](https://arxiv.org/abs/2012.00677): a candidate future sequential-assimilation approach, not our implemented algorithm.

These references contextualise the research. They do not independently validate our nominal hull or numerical results. An eventual paper needs an explicit observation model, identifiability/conditioning analysis, honest train/development/test boundaries, matched-baseline comparisons, model mismatch and noise experiments, uncertainty evaluation and ideally independent tank/field measurements.

---

# Exact implementation source appendix

The original seven-file frozen protocol hashes match the packaged core. Additional dependency files are included for inspection. Filenames are relative to the estimator folder.

## src/hybrid_pressure_live.py

```python
"""Learned sea-statistics prior + causal phase-preserving pressure inverse.

Statistical parameters come only from an initial past-pressure window. They
guide a causal FIR; raw current/past pressures supply individual wave timing.
No hidden surface, generator parameters, IMU or future samples are accepted.
"""
import time
import numpy as np
from scipy.linalg import solve
from pressure_array import array_response
from pressure_only_live import Calibration,PressureOnlyLive
from pressure_prior import estimate_prior


def calibrate_hybrid(model,config,targets,prior,*,sample_interval_s=.5,taps=128,
                     ridge_m2=1e-8,uniform_fraction=.01):
    xy=np.asarray(targets,float);omega=np.asarray(model.omega);theta=np.asarray(model.theta)
    if xy.ndim!=2 or xy.shape[1]!=2 or not np.isfinite(xy).all():raise ValueError('Finite target xy required')
    if not 0<=uniform_fraction<=1 or taps<8 or ridge_m2<=0 or sample_interval_s<=0:raise ValueError('Invalid filter settings')
    weights=prior.weights(omega/(2*np.pi),theta,uniform_fraction=uniform_fraction)
    H=array_response(model,omega,theta,config)[:,:,:6]/(model.rho*model.g)
    Q=np.einsum('fdc,fde,fd->fce',H,H.conj(),weights)
    lag=np.arange(taps)*sample_interval_s
    Rlag=np.einsum('fl,fce->lce',np.exp(1j*omega[:,None]*lag),Q).real
    delta=np.arange(taps)[:,None]-np.arange(taps)[None,:]
    R=np.where((delta>=0)[:,:,None,None],Rlag[abs(delta)],Rlag[abs(delta)].transpose(0,1,3,2)).transpose(0,2,1,3).reshape(taps*6,taps*6)
    R+=ridge_m2*np.eye(taps*6)
    P=np.empty((len(xy),taps*6));k=omega**2/model.g
    for start in range(0,len(xy),24):
        points=xy[start:start+24]
        E=np.exp(1j*k[:,None,None]*(np.cos(theta)[None,:,None]*points[None,None,:,0]+np.sin(theta)[None,:,None]*points[None,None,:,1]))
        cross=np.einsum('fdp,fdc,fd->fpc',E,H.conj(),weights)
        P[start:start+24]=np.einsum('fl,fpc->plc',np.exp(-1j*omega[:,None]*lag),cross).real.reshape(len(points),-1)
    W=solve(R,P.T,assume_a='pos').T
    D=np.tile(np.eye(6),(taps,1));RiD=solve(R,D,assume_a='pos')
    W-=(W@D)@solve(D.T@RiD,RiD.T,assume_a='pos')
    W=W.reshape(len(xy),taps,6);W-=W.sum(axis=1)[:,None,:]/taps
    return Calibration(W/(model.rho*model.g),xy,
                       (model.depth-np.asarray(config.positions)[:,2])*model.rho*model.g,
                       sample_interval_s,{'method':'Pressure-learned directional/frequency prior + causal current-time FIR',
                         'learned_prior':prior.to_dict(),'uniform_direction_fraction':uniform_fraction,
                         'taps':taps,'ridge_m2':ridge_m2,'output_time_delay_s':0.,
                         'depth_m':model.depth,'band_Hz':[float(omega[0]/(2*np.pi)),float(omega[-1]/(2*np.pi))],
                         'input':'Six past/current pressures only; nominal hull calibration known',
                         'limits':'Single-system separable prior, fixed gamma; numerical stabilization is not calibrated sensor noise; no model uncertainty intervals'})


class HybridPressureLive:
    """Synchronous initial-learning prototype; first output includes fit latency.

    During startup stores pressures. At learn_samples it fits ONLY this past
    window, calibrates, and primes the FIR with its last taps samples. Subsequent
    calls use only the current packet and causal history. The prior is then
    frozen for this stationary-sea benchmark; no periodic adaptation claimed.
    """
    def __init__(self,model,config,targets,*,sample_interval_s=.5,learn_samples=1024,
                 taps=128,ridge_m2=1e-8,uniform_fraction=.01):
        if learn_samples<max(512,taps):raise ValueError('Learning history too short')
        self.model=model;self.config=config;self.targets=np.asarray(targets,float)
        self.dt=sample_interval_s;self.learn_samples=learn_samples;self.taps=taps
        self.ridge_m2=ridge_m2;self.uniform_fraction=uniform_fraction
        self.samples=[];self.times=[];self.live=None;self.fit=None;self.learning_compute_s=None

    def push(self,time_s,pressures_Pa):
        p=np.asarray(pressures_Pa,float)
        if p.shape!=(6,) or not np.isfinite(p).all() or not np.isfinite(time_s):raise ValueError('One timestamp and six pressures required')
        if self.live is not None:return self.live.push(time_s,p)
        if self.times and not np.isclose(time_s-self.times[-1],self.dt,rtol=0,atol=1e-6):raise ValueError('Nonuniform/out-of-order samples')
        self.times.append(float(time_s));self.samples.append(p.copy())
        if len(self.samples)<self.learn_samples:return None
        start=time.perf_counter()
        self.fit=estimate_prior(self.samples,self.model,self.config,sample_interval_s=self.dt)
        self.calibration=calibrate_hybrid(self.model,self.config,self.targets,self.fit,
                        sample_interval_s=self.dt,taps=self.taps,ridge_m2=self.ridge_m2,uniform_fraction=self.uniform_fraction)
        self.live=PressureOnlyLive(self.calibration)
        for t,row in zip(self.times[-self.taps:],self.samples[-self.taps:]):out=self.live.push(t,row)
        self.learning_compute_s=time.perf_counter()-start
        self.samples=[];self.times=[]
        return out

```

## src/pressure_prior.py

```python
"""Learn a single-system directional/spectral prior from six pressures.

Only past samples and nominal hull calibration are accepted. This is a
working tapered matrix-Whittle fit, not a confidence interval or an exact
finite-record likelihood. A parametric prior is an assumption, not sea truth.
"""
from dataclasses import dataclass,asdict
import numpy as np
from scipy.optimize import minimize,minimize_scalar
from sea import FixedSea,jonswap
from pressure_array import array_response


@dataclass
class PressurePrior:
    mean_direction_deg: float
    spread_s: float
    Hs_m: float
    Tp_s: float
    gamma: float
    objective: float
    samples: int
    history_duration_s: float
    boundary: bool
    diagnostic: dict

    def weights(self,f,theta,*,uniform_fraction=.01):
        p=FixedSea(self.gamma,np.deg2rad(self.mean_direction_deg),self.spread_s).fractions(theta)
        p=(1-uniform_fraction)*p+uniform_fraction/len(theta)
        spectrum=jonswap(f,self.Hs_m,self.Tp_s,self.gamma)
        return spectrum[:,None]*p[None,:]/spectrum.sum()

    def to_dict(self):return asdict(self)


def estimate_prior(pressures_Pa,model,config,*,sample_interval_s=.5,
                   fit_band=(.06,.26),rank_tolerance=1e-7,gamma=3.3):
    """Frequency-profiled angular fit; scalar spectrum is a nuisance per bin.

    Fit direction/spread first using complex relative phases across six ports.
    Then fit Hs/Tp from the profiled spectral powers, with gamma fixed. No
    generator parameters, hidden motion, component phases or IMU enter.
    """
    y=np.asarray(pressures_Pa,float)
    if y.ndim!=2 or y.shape[1]!=6 or len(y)<512 or not np.isfinite(y).all():
        raise ValueError('At least 512 finite samples and exactly six pressures required')
    n=len(y);fs=1/sample_interval_s;freq=np.fft.rfftfreq(n,sample_interval_s)
    inside=(freq>=fit_band[0])&(freq<=fit_band[1])
    f=freq[inside];table=model
    H=array_response(table,table.omega,table.theta,config)[:,:,:6]
    window=np.hanning(n+1)[:-1]
    z=np.conj(np.fft.rfft((y-y.mean(axis=0))*window[:,None],axis=0)[inside])*np.sqrt(2/(fs*np.sum(window**2)))
    # Expected finite-window periodogram, including positive/negative-frequency
    # leakage. This avoids mistaking short-window leakage for long waves.
    source_f=table.omega/(2*np.pi);df=source_f[1]-source_f[0]
    def dirichlet(q):
        return n*np.exp(-1j*np.pi*(n-1)*q)*np.sinc(n*q)/np.sinc(q)
    def taper_transform(q):
        return .5*dirichlet(q)-.25*dirichlet(q-1/n)-.25*dirichlet(q+1/n)
    positive=abs(taper_transform((f[:,None]-source_f[None,:])*sample_interval_s))**2
    negative=abs(taper_transform((f[:,None]+source_f[None,:])*sample_interval_s))**2
    positive*=df/(fs*np.sum(window**2));negative*=df/(fs*np.sum(window**2))
    # Vertically separated ports at identical xy cancel first-order vehicle
    # motion. Learn the frequency shape from their difference, not true eta.
    positions=np.asarray(config.positions);pairs=[]
    for a in range(6):
        for b in range(a):
            if np.allclose(positions[a,:2],positions[b,:2]) and not np.isclose(positions[a,2],positions[b,2]):pairs.append((a,b))
    if not pairs:raise ValueError('This prior estimator requires a vertically separated port pair at identical xy')
    a,b=max(pairs,key=lambda ab:abs(positions[ab[0],2]-positions[ab[1],2]))
    k=table.omega**2/table.g
    gain=table.rho*table.g*(np.exp(-k*(table.depth-positions[a,2]))-np.exp(-k*(table.depth-positions[b,2])))
    scalar_I=abs(z[:,a]-z[:,b])**2
    kernel=(positive+negative)*gain[None,:]**2
    def period_value(Tp,return_height=False):
        shape=kernel@jonswap(source_f,1.,Tp,gamma)
        ratio=np.mean(scalar_I/shape)
        h2=np.clip(ratio,.2**2,5.**2)
        value=np.mean(np.log(shape))+np.log(h2)+ratio/h2
        return (float(value),float(np.sqrt(h2))) if return_height else value
    periods=np.linspace(5,12,141);values=[period_value(t) for t in periods];best=int(np.argmin(values))
    fit=minimize_scalar(period_value,bounds=(max(5,periods[best]-.15),min(12,periods[best]+.15)),method='bounded')
    Tp=float(fit.x);_,Hs=period_value(Tp,True)
    spectrum=jonswap(source_f,1.,Tp,gamma)
    outer_source=np.einsum('fdc,fde->fdce',H,H.conj())*spectrum[:,None,None,None]
    flat=outer_source.reshape(len(source_f),-1)
    outer_all=(positive@flat+negative@flat.conj()).reshape(len(f),len(table.theta),6,6)
    isotropic=outer_all.mean(axis=1)
    scale=np.sqrt(isotropic.diagonal(axis1=1,axis2=2).real)
    outer_all/=scale[:,None,:,None]*scale[:,None,None,:]
    z=z/scale
    isotropic=outer_all.mean(axis=1)
    ev,U=np.linalg.eigh(isotropic);ranks=(ev>rank_tolerance*ev[:,-1:]).sum(axis=1)
    power=np.sum(abs(z)**2,axis=1)
    use=(power>1e-7*power.max())
    groups=[]
    for rank in np.unique(ranks[use]):
        ids=np.flatnonzero(use&(ranks==rank))
        V=U[ids,:,-rank:]/np.sqrt(ev[ids,None,-rank:])
        zz=np.einsum('fcr,fc->fr',V.conj(),z[ids])
        outer=np.einsum('fcr,fdce,fes->fdrs',V.conj(),outer_all[ids],V)
        groups.append((int(rank),ids,outer,zz))
    count=int(sum(rank*len(ids) for rank,ids,_,_ in groups))
    if count<100:raise ValueError('Insufficient retained spectral information')

    def sufficient(mu,s):
        p=FixedSea(gamma,np.deg2rad(mu),s).fractions(table.theta)
        qfull=np.zeros(len(f));total=0.
        for rank,ids,outer,zz in groups:
            K=np.einsum('fdrs,d->frs',outer,p)
            try:
                L=np.linalg.cholesky(K)
                ld=2*np.log(np.diagonal(L,axis1=1,axis2=2).real).sum(axis=1)
                solved=np.linalg.solve(K,zz[...,None])[...,0]
                q=np.einsum('fr,fr->f',zz.conj(),solved).real
                if np.any(q<=0):return np.inf,qfull
                total+=float(np.sum(ld+rank*np.log(q/rank)))
                qfull[ids]=q
            except np.linalg.LinAlgError:return np.inf,qfull
        return total/count,qfull

    coarse=[]
    for s in [1.,3.,8.,20.,50.]:
        for mu in np.arange(0,360,15.):coarse.append((sufficient(mu,s)[0],mu,s))
    candidates=[]
    for value,mu,s in sorted(coarse)[:4]:
        initial=[mu,np.log(s)]
        result=minimize(lambda x:sufficient(x[0],np.exp(x[1]))[0],initial,method='Powell',
                        bounds=[(mu-30,mu+30),(0,np.log(50))],options={'xtol':1e-5,'ftol':1e-9,'maxiter':100})
        candidates.append((float(result.fun),float(result.x[0]%360),float(np.exp(result.x[1]))))
    objective,mu,s=min(candidates);_,q=sufficient(mu,s)
    return PressurePrior(mu,s,Hs,Tp,gamma,float(objective),n,(n-1)*sample_interval_s,
                         bool(s<1.01 or s>49.5 or Tp<5.01 or Tp>11.99 or Hs<.201 or Hs>4.99),
                         {'rank_min':int(ranks[use].min()),'rank_max':int(ranks[use].max()),
                          'frequency_bins':int(use.sum()),'fit_band_Hz':fit_band,
                          'method':'Finite-window expected-periodogram fit: motion-cancelling pressure pair for frequency shape, matrix-Whittle angular profile',
                          'motion_cancelling_pair':[int(a),int(b)],
                          'limits':'Single separable cos-2s direction prior; fixed gamma. Taper leakage modeled; no uncertainty intervals.'})

```

## src/pressure_only_live.py

```python
"""Causal six-pressure-only surface estimator; no sea or IMU input.

Offline calibration uses a known nominal linear hull/sensor response and an
explicit flat frequency, isotropic direction prior. This supplies missing
spatial information: it does not make the six-port inverse unique. Each push
uses exactly one timestamp and six pressures, retaining only past samples.
"""
from dataclasses import dataclass
import json
import numpy as np
from scipy.linalg import solve
from pressure_array import array_response


@dataclass
class Calibration:
    weights: np.ndarray  # [target, lag, port], lag 0 is current sample
    targets: np.ndarray
    baseline_Pa: np.ndarray
    sample_interval_s: float
    metadata: dict

    def save(self, path):
        np.savez_compressed(path, weights=self.weights, targets=self.targets,
                            baseline_Pa=self.baseline_Pa,
                            sample_interval_s=self.sample_interval_s,
                            metadata=json.dumps(self.metadata))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            return cls(z['weights'],z['targets'],z['baseline_Pa'],
                       float(z['sample_interval_s']),json.loads(str(z['metadata'])))


def calibrate(model, config, targets, *, sample_interval_s=.5, taps=128,
              ridge_m2=1e-8):
    """Solve a finite, causal, linear minimum-mean-square inverse from physics.

    No simulated/measured sea, fitted JONSWAP, phases, states or IMU supplied.
    Prior surface variance = 1 m^2, equally divided over response frequencies
    and directions. ridge_m2 penalizes gain as hypothetical independent noise
    in pressure-head metres squared; it is not a measured sensor noise spec.
    Six exact zero-DC constraints reject arbitrary constant pressure offsets.
    """
    xy=np.asarray(targets,float)
    if xy.ndim!=2 or xy.shape[1]!=2 or not np.isfinite(xy).all():
        raise ValueError('Finite target xy required')
    if taps<8 or sample_interval_s<=0 or ridge_m2<=0:
        raise ValueError('At least eight taps and positive interval/ridge required')
    omega=np.asarray(model.omega);theta=np.asarray(model.theta)
    if omega[-1]*sample_interval_s>=np.pi:
        raise ValueError('Response frequency exceeds sample Nyquist')
    if np.any(np.diff(omega)<=0) or not np.allclose(np.diff(omega),np.diff(omega)[0]):
        raise ValueError('Uniform increasing response frequencies required')
    # Pa -> metres pressure head; keep ONLY the six pressure responses.
    H=array_response(model,omega,theta,config)[:,:,:6]/(model.rho*model.g)
    Q=np.einsum('fdc,fde->fce',H,H.conj())/len(theta)
    lag=np.arange(taps)*sample_interval_s
    prior=np.full(len(omega),1/len(omega))
    Rlag=np.einsum('fl,fce,f->lce',np.exp(1j*omega[:,None]*lag),Q,prior).real
    delta=np.arange(taps)[:,None]-np.arange(taps)[None,:]
    R=np.where((delta>=0)[:,:,None,None],Rlag[abs(delta)],
               Rlag[abs(delta)].transpose(0,1,3,2)).transpose(0,2,1,3).reshape(taps*6,taps*6)
    R+=ridge_m2*np.eye(taps*6)
    k=omega**2/model.g
    P=np.empty((len(xy),taps*6))
    for start in range(0,len(xy),24):
        points=xy[start:start+24]
        E=np.exp(1j*k[:,None,None]*(np.cos(theta)[None,:,None]*points[None,None,:,0]
                                   +np.sin(theta)[None,:,None]*points[None,None,:,1]))
        cross=np.einsum('fdp,fdc->fpc',E,H.conj())/len(theta)
        P[start:start+24]=np.einsum('fl,fpc,f->plc',np.exp(-1j*omega[:,None]*lag),cross,prior).real.reshape(len(points),-1)
    W=solve(R,P.T,assume_a='pos').T
    D=np.tile(np.eye(6),(taps,1))
    RiD=solve(R,D,assume_a='pos')
    W-=(W@D)@solve(D.T@RiD,RiD.T,assume_a='pos')
    # Roundoff-only cleanup of exact zero-DC constraints.
    W=W.reshape(len(xy),taps,6)
    W-=W.sum(axis=1)[:,None,:]/taps
    prior_mse=1-2*np.einsum('pl,pl->p',W.reshape(len(xy),-1),P)
    prior_mse+=np.einsum('pl,lk,pk->p',W.reshape(len(xy),-1),R,W.reshape(len(xy),-1))
    baseline=(model.depth-np.asarray(config.positions)[:,2])*model.rho*model.g
    metadata={'inputs':'timestamp and six absolute pressures in Pa only',
              'positions_m':config.positions,'placement_status':config.placement_status,
              'known_calibration':'Nominal hull response, port geometry, depth, rho, g, sample interval',
              'prior':'flat over calibration frequencies, isotropic over directions; no JONSWAP fit',
              'band_Hz':[float(omega[0]/(2*np.pi)),float(omega[-1]/(2*np.pi))],
              'taps':taps,'startup_s':float(lag[-1]),'output_time_delay_s':0.,
              'ridge_m2':ridge_m2,'equivalent_regularizer_Pa':float(np.sqrt(ridge_m2)*model.rho*model.g),
              'zero_DC_constraints':True,'design_prior_MSE_m2':prior_mse.tolist(),
              'limits':'Matched linear physics, finite calibrated frequency band; unobservable spatial modes remain prior-dependent'}
    # Store direct weights from Pa to metres for a minimal deployment interface.
    return Calibration(W/(model.rho*model.g),xy,baseline,sample_interval_s,metadata)


class PressureOnlyLive:
    """One-way streaming boundary. Never accepts truth, sea objects, or IMU."""
    def __init__(self,calibration):
        self.calibration=calibration
        self.history=np.zeros(calibration.weights.shape[1:])
        self.count=0
        self.last_time=None

    def push(self,time_s,pressure_Pa):
        p=np.asarray(pressure_Pa,float)
        if p.shape!=(6,) or not np.isfinite(p).all() or not np.isfinite(time_s):
            raise ValueError('One finite timestamp and exactly six pressures required')
        if self.last_time is not None and not np.isclose(time_s-self.last_time,self.calibration.sample_interval_s,rtol=0,atol=1e-6):
            raise ValueError('Samples must arrive once, in order, at the calibrated interval')
        self.history[1:]=self.history[:-1]
        self.history[0]=p-self.calibration.baseline_Pa
        self.count+=1
        self.last_time=float(time_s)
        if self.count<len(self.history):
            return None
        return np.einsum('plc,lc->p',self.calibration.weights,self.history,optimize=False)

```

## src/pressure_array.py

```python
"""Six configurable pressure ports plus the existing raw IMU.

Sponsor count: six (user confirmation, 14 October 2026).
Actual mounting coordinates: unconfirmed. Layouts must be selected explicitly.
First-order incident-wave/hydrostatic sensor physics; hull-scattered pressure
and measurement noise remain outside this prototype.
"""
from dataclasses import dataclass
import json
import numpy as np
from sensors import Sensors,CHANNELS,sensor_response
from submarine import skew
from time_domain import simulate,_periodic_input


@dataclass(frozen=True)
class PressureArray:
    positions: tuple
    placement_status: str
    imu_position: tuple = (10.,0.,0.)
    names: tuple = ('p1','p2','p3','p4','p5','p6')

    def __post_init__(self):
        p=np.asarray(self.positions,float)
        if p.shape!=(6,3) or not np.all(np.isfinite(p)) or len(np.unique(p,axis=0))!=6:
            raise ValueError('Six distinct finite body-frame xyz coordinates are required')
        if len(self.names)!=6 or len(set(self.names))!=6 or not all(isinstance(n,str) and n for n in self.names):
            raise ValueError('Six unique sensor names are required')
        if np.shape(self.imu_position)!=(3,) or not np.all(np.isfinite(self.imu_position)):
            raise ValueError('Finite IMU xyz position required')
        if not isinstance(self.placement_status,str) or not self.placement_status.strip():
            raise ValueError('Placement provenance/status must be explicit')

    @property
    def channels(self):
        return tuple('pressure_'+name+'_Pa' for name in self.names)+CHANNELS[1:]

    def save(self,path):
        with open(path,'w') as out:
            json.dump({'positions_m':self.positions,'names':self.names,'imu_position_m':self.imu_position,
                       'placement_status':self.placement_status,'coordinate_convention':'+x along hull, +z up, origin geometric hull centre'},out,indent=2)

    @classmethod
    def load(cls,path):
        with open(path) as f:d=json.load(f)
        return cls(tuple(tuple(p) for p in d['positions_m']),d['placement_status'],tuple(d['imu_position_m']),tuple(d['names']))


def array_response(model,omega,theta,config):
    """[frequency,direction,12]: six pressures, 3 accelerometers, 3 gyros."""
    r=np.asarray(config.positions,float);depths=model.depth-r[:,2]
    if np.any(depths<=0):raise ValueError('Every mean pressure port must be submerged')
    q=model.motion_response(omega,theta);w=np.asarray(omega);k=w*w/model.g
    spatial=k[:,None,None]*(np.cos(theta)[None,:,None]*r[None,None,:,0]+np.sin(theta)[None,:,None]*r[None,None,:,1])
    incident=model.rho*model.g*np.exp(-k[:,None,None]*depths[None,None,:]+1j*spatial)
    port_z=q[:,:,2,None]+q[:,:,3,None]*r[None,None,:,1]-q[:,:,4,None]*r[None,None,:,0]
    pressure=incident-model.rho*model.g*port_z
    imu=sensor_response(model,omega,theta,Sensors(config.imu_position,config.positions[0]))[:,:,1:]
    return np.concatenate([pressure,imu],axis=2)


def simulate_array(sea,hull,config,**kwargs):
    """Coupled RK4 motion, then independent six-port TIME instrument equations.

    Does not use motion RAOs or array_response to generate pressure signals.
    True states returned separately for evaluation; not estimator inputs.
    """
    r=np.asarray(config.positions,float);depths=hull.depth-r[:,2]
    if np.any(depths<=0):raise ValueError('Every mean pressure port must be submerged')
    t,old,X,acceleration=simulate(sea,hull,config=Sensors(config.imu_position,config.positions[0]),**kwargs)
    fs=1/(t[1]-t[0]);n=len(t)
    spatial=sea.k[:,None,None]*(np.cos(sea.theta)[None,:,None]*r[None,None,:,0]+np.sin(sea.theta)[None,:,None]*r[None,None,:,1])
    pressure_amplitudes=hull.rho*hull.g*np.einsum('fd,fdp->fp',sea.complex_components,np.exp(-sea.k[:,None,None]*depths[None,None,:]+1j*spatial))
    pressure_amplitudes*=np.exp(-1j*sea.omega*t[0])[:,None]
    incident=_periodic_input(pressure_amplitudes,sea.f,fs,n)
    port_z=X[:,2,None]+X[:,3,None]*r[None,:,1]-X[:,4,None]*r[None,:,0]
    pressure=hull.rho*hull.g*(depths[None,:]-port_z)+incident
    return t,np.concatenate([pressure,old[:,1:]],axis=1),X,acceleration


def reconstruct_above_ports(time,signals,config,*,depth,rho=1025.,g=9.81,band=(.05,.30)):
    """Local zero-mean surface elevations at each port's horizontal location.

    Uses only the recorded 12 channels and nominal geometry/pressure physics.
    Angular-rate and acceleration phases correct motion at every port; no
    fitted sea shape, directional weights, generating phases or states enter.
    """
    time=np.asarray(time,float);y=np.asarray(signals,float);r=np.asarray(config.positions,float)
    if y.shape!=(len(time),12) or len(time)<16 or not np.all(np.isfinite(y)):
        raise ValueError('Finite samples x 12 array required')
    intervals=np.diff(time)
    if np.any(intervals<=0) or not np.allclose(intervals,intervals[0]):raise ValueError('Uniform increasing timestamps required')
    depths=depth-r[:,2]
    if np.any(depths<=0) or not 0<band[0]<band[1] or rho<=0 or g<=0:raise ValueError('Submerged ports and positive physics/band required')
    fs=1/intervals[0];n=len(time);f=np.fft.rfftfreq(n,1/fs)
    use=(f>=band[0])&(f<=band[1])&(f>0)&(f<fs/2);w=2*np.pi*f[use]
    if not use.any():raise ValueError('No supported Fourier frequencies')
    Y=np.fft.rfft(y-y.mean(axis=0),axis=0);roll=Y[use,9]/(1j*w);pitch=Y[use,10]/(1j*w)
    displacement=-Y[use,8,None]/w[:,None]**2
    displacement=displacement+roll[:,None]*(r[None,:,1]-config.imu_position[1])-pitch[:,None]*(r[None,:,0]-config.imu_position[0])
    coef=np.zeros((len(f),6),complex)
    coef[use]=np.exp(w[:,None]**2/g*depths[None,:])*(Y[use,:6]/(rho*g)+displacement)
    return np.fft.irfft(coef,n=n,axis=0)


def symmetric_six_port_layout(length=70.,radius=5.,imu_position=(10.,0.,0.)):
    """User-authorized modelling assumption, not verified sponsor hardware.

    Four horizontal extremities plus top/bottom at mid-length. Surface ports
    occupy +/- length/2 along x and +/- radius on y and z for the capsule.
    """
    if not np.isfinite(length) or not np.isfinite(radius) or not length>2*radius>0:
        raise ValueError('Finite positive capsule dimensions required')
    return PressureArray(((length/2,0.,0.),(-length/2,0.,0.),(0.,radius,0.),
                          (0.,-radius,0.),(0.,0.,radius),(0.,0.,-radius)),
                         'User-authorized symmetric modelling assumption; six-sensor count confirmed, actual hardware coordinates unverified',
                         tuple(imu_position),('bow','stern','port','starboard','top','bottom'))


def wave_interpolate(time,local_elevations,positions,targets,*,g=9.81,band=(.05,.30),relative_eigen_floor=1e-8):
    """Phase-preserving spatial interpolation with a dispersion-based kernel.

    Uses deep-water k=omega^2/g and isotropic angular covariance J0(k*r).
    Equal directional prior and minimum-norm regularization are assumptions,
    not proof of a unique wave field. No JONSWAP frequency shape or sea truth.
    Duplicate horizontal points (top/bottom) are averaged explicitly.
    """
    from scipy.special import j0
    time=np.asarray(time,float);eta=np.asarray(local_elevations,float)
    xy=np.asarray(positions,float)[:,:2];targets=np.asarray(targets,float)
    if eta.shape!=(len(time),len(xy)) or targets.ndim!=2 or targets.shape[1]!=2 or not np.all(np.isfinite(eta)) or not np.all(np.isfinite(targets)):
        raise ValueError('Finite sample x port elevations and target xy coordinates required')
    if not 0<relative_eigen_floor<1 or g<=0:raise ValueError('Positive gravity and relative eigen floor required')
    dt=np.diff(time)
    if np.any(dt<=0) or not np.allclose(dt,dt[0]):raise ValueError('Uniform increasing timestamps required')
    unique,labels=np.unique(xy,axis=0,return_inverse=True)
    merged=np.column_stack([eta[:,labels==j].mean(axis=1) for j in range(len(unique))])
    n=len(time);f=np.fft.rfftfreq(n,dt[0]);use=(f>=band[0])&(f<=band[1])&(f>0)&(f<1/(2*dt[0]));k=(2*np.pi*f[use])**2/g
    distance=np.linalg.norm(unique[:,None,:]-unique[None,:,:],axis=2)
    target_distance=np.linalg.norm(targets[:,None,:]-unique[None,:,:],axis=2)
    K=j0(k[:,None,None]*distance[None,:,:]);ev,U=np.linalg.eigh(K)
    inv=np.zeros_like(ev);np.divide(1.,ev,out=inv,where=ev>relative_eigen_floor*ev[:,-1:])
    inverse=np.einsum('fij,fj,fkj->fik',U,inv,U)
    field=j0(k[:,None,None]*target_distance[None,:,:])
    weights=np.einsum('ftp,fpq->ftq',field,inverse)
    coef=np.zeros((len(f),len(targets)),complex);z=np.fft.rfft(merged-merged.mean(axis=0),axis=0)
    coef[use]=np.einsum('ftp,fp->ft',weights,z[use])
    return np.fft.irfft(coef,n=n,axis=0)


def reconstruct_surface_joint(time,signals,config,model,targets,*,band=(.05,.30),rank_tolerance=1e-9):
    """Direct complex-array inverse over a target patch, retaining wave phase.

    Nominal six-DOF/sensor response is known. Directional Fourier amplitudes
    have an isotropic equal-energy prior at each frequency. A pseudoinverse
    handles response redundancy separately at every frequency. No JONSWAP
    frequency shape, generating phases, states or true sea parameters enter.
    Unobserved modes are set by a minimum-norm prior, not uniquely recovered.
    """
    time=np.asarray(time,float);y=np.asarray(signals,float);xy=np.asarray(targets,float)
    if y.shape!=(len(time),12) or xy.ndim!=2 or xy.shape[1]!=2 or not np.all(np.isfinite(y)) or not np.all(np.isfinite(xy)):
        raise ValueError('Finite samples x 12 recordings and target xy required')
    dt=np.diff(time)
    if np.any(dt<=0) or not np.allclose(dt,dt[0]) or not 0<rank_tolerance<1:raise ValueError('Uniform increasing time and relative rank tolerance required')
    n=len(time);grid=np.fft.rfftfreq(n,dt[0]);use=(grid>=max(band[0],model.omega[0]/(2*np.pi)))&(grid<=min(band[1],model.omega[-1]/(2*np.pi)))&(grid>0)&(grid<1/(2*dt[0]))
    f=grid[use];sample_model=model.on_grid(2*np.pi*f);H=array_response(sample_model,2*np.pi*f,sample_model.theta,config)
    scale=np.sqrt(np.mean(abs(H)**2,axis=1));scale=np.where(scale>0,scale,1.)
    A=H/scale[:,None,:];K=np.einsum('fdc,fde->fce',A,A.conj())/len(sample_model.theta)
    ev,U=np.linalg.eigh(K);keep=ev>rank_tolerance*ev[:,-1:];inverse_ev=np.zeros_like(ev);np.divide(1.,ev,out=inverse_ev,where=keep)
    # Use only recorded Fourier coefficients, preserving their common phase.
    physical=np.fft.rfft(y-y.mean(axis=0),axis=0)[use].conj()/scale
    coefficients=np.einsum('fci,fc->fi',U.conj(),physical)*inverse_ev
    dual=np.einsum('fci,fi->fc',U,coefficients)
    latent=np.einsum('fdc,fc->fd',A.conj(),dual)/len(sample_model.theta)
    field=np.empty((len(f),len(xy)),complex);k=(2*np.pi*f)**2/model.g
    for start in range(0,len(f),48):
        sl=slice(start,start+48)
        spatial=k[sl,None,None]*(np.cos(sample_model.theta)[None,:,None]*xy[None,None,:,0]+np.sin(sample_model.theta)[None,:,None]*xy[None,None,:,1])
        field[sl]=np.einsum('fd,fdp->fp',latent[sl],np.exp(1j*spatial))
    out=np.zeros((len(grid),len(xy)),complex);out[use]=field.conj()
    return np.fft.irfft(out,n=n,axis=0),{'rank_min':int(keep.sum(axis=1).min()),'rank_max':int(keep.sum(axis=1).max()),'rank_tolerance':rank_tolerance,'prior':'equal directional energy at each frequency; no JONSWAP frequency model'}

```

## src/submarine.py

```python
"""Replaceable 6-DOF linear hydrodynamics; no Capytaine dependency.

The temporary capsule strip approximation is an explicit sprint surrogate,
not WB's missing hull implementation or a validated real-submarine model.
"""
from dataclasses import dataclass
import numpy as np

RHO, G = 1025., 9.81
DOFS = ("Surge", "Sway", "Heave", "Roll", "Pitch", "Yaw")


def skew(r):
    x, y, z = np.asarray(r)
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def compute_rao(sub, omega, A, B, F):
    """WM's layout and equation: F[frequency,direction,dof]."""
    omega = np.asarray(omega)
    Z = -omega[:, None, None]**2 * (sub.M[None] + A) - 1j*omega[:, None, None]*B + sub.C_hydro[None]
    return np.linalg.solve(Z, F.transpose(0, 2, 1)).transpose(0, 2, 1)


@dataclass(frozen=True)
class Hull:
    length: float = 70.0  # TOTAL tip-to-tip length, not barrel length
    radius: float = 5.0
    depth: float = 15.0  # reference origin = geometric centre, z positive up
    bg: float = 0.5
    damping_rate: float = 0.08  # B = rate * M_eff; illustrative, in 1/s
    strips: int = 400


class Submarine:
    def __init__(self, *, omega, theta, hull=None, build_response=True):
        self.omega, self.theta = np.asarray(omega, float), np.asarray(theta, float)
        self.hull = hull or Hull()
        h = self.hull
        if h.length <= 2*h.radius or h.depth <= h.radius or h.bg < 0 or h.damping_rate < 0 or h.strips < 20:
            raise ValueError("Need submerged capsule, nonnegative BG/damping, >=20 strips")
        self.depth, self.rho, self.g = h.depth, RHO, G
        self.dofs = list(DOFS)
        self.reference = "geometric hull centre; axes +x along hull, +z upward"
        dx = h.length / h.strips
        x = -h.length/2 + (np.arange(h.strips)+.5)*dx
        barrel_half = h.length/2 - h.radius
        cap_x = np.maximum(np.abs(x)-barrel_half, 0.)
        r2 = h.radius**2 - cap_x**2
        area = np.pi*r2
        dV = area*dx
        self.volume = dV.sum()
        self.mass = RHO*self.volume
        # Uniform capsule inertias used as an illustrative I_CG, plus a shifted
        # CG/ballast surrogate. The ballast distribution is not a measured fit.
        Ixx = RHO*np.sum(dV*r2/2)
        Iyy = RHO*np.sum(dV*(x*x + r2/4))
        rg = np.array([0., 0., -h.bg])
        Sg = skew(rg)
        self.M = np.block([[self.mass*np.eye(3), -self.mass*Sg],
                           [self.mass*Sg, np.diag([Ixx,Iyy,Iyy]) - self.mass*Sg@Sg]])
        # Circular-strip transverse added mass Ca=1 follows Felix's derivation.
        # Axial approximation: 1/2 rho * combined cap volume (sphere surrogate).
        # This is NOT the exact axial added mass of a finite capsule.
        caps_volume = 4*np.pi*h.radius**3/3
        axial_ca = .5*caps_volume/self.volume
        added = np.diag([axial_ca*self.mass, self.mass, self.mass,
                         0., RHO*np.sum(dV*x*x), RHO*np.sum(dV*x*x)])
        self.A = np.broadcast_to(added, (len(self.omega),6,6)).copy()
        self.B = np.broadcast_to(h.damping_rate*(self.M+added), self.A.shape).copy()
        self.C_hydro = np.diag([0.,0.,0., self.mass*G*h.bg, self.mass*G*h.bg,0.])
        self.F_exc = np.zeros((len(self.omega),len(self.theta),6), complex)
        ct, st = np.cos(self.theta), np.sin(self.theta)
        # Evaluate wave fluid acceleration at strip centres at mean depth.
        # Small-section approximation: ignore phase/attenuation across radius.
        for start in range(0, len(self.omega), 64):
            sl = slice(start, start+64)
            w = self.omega[sl]
            k = w*w/G
            basis = np.exp(1j*k[:,None,None]*ct[None,:,None]*x[None,None,:])
            v0 = np.einsum('fdx,x->fd',basis,dV)
            v1 = np.einsum('fdx,x->fd',basis,dV*x)
            scale = RHO*w[:,None]**2*np.exp(-k[:,None]*h.depth)
            self.F_exc[sl,:,0] = -1j*scale*(1+axial_ca)*ct*v0
            self.F_exc[sl,:,1] = -1j*scale*2*st*v0
            self.F_exc[sl,:,2] = -scale*2*v0
            # r x F: pitch=-x Fz; yaw=x Fy. No direct roll excitation.
            self.F_exc[sl,:,4] = scale*2*v1
            self.F_exc[sl,:,5] = -1j*scale*2*st*v1
        self.rao = compute_rao(self,self.omega,self.A,self.B,self.F_exc) if build_response else None

    def motion_response(self, omega, theta):
        _same_grid(self.omega,self.theta,omega,theta)
        if self.rao is None:
            raise ValueError("This time-only hull has no motion-response table")
        return self.rao


def _same_grid(w, th, omega, theta):
    if np.shape(w) != np.shape(omega) or np.shape(th) != np.shape(theta) or not np.allclose(w,omega) or not np.allclose(th,theta):
        raise ValueError("Response table/grid mismatch; resample explicitly, never silently extrapolate")


class ResponseTable:
    """Estimator boundary: replace only the response provider, keeping sensors/fit.

    Existing solved WM objects can be adapted without importing their scripts.
    Caller must confirm the reference depth, axes and exp(-i omega t) convention.
    """
    def __init__(self, *, omega, theta, rao, depth, reference, rho=RHO, g=G):
        self.omega, self.theta = np.asarray(omega,float), np.asarray(theta,float)
        self.rao = np.asarray(rao,complex)
        if self.rao.shape != (len(self.omega),len(self.theta),6) or not np.all(np.isfinite(self.rao)):
            raise ValueError("Finite RAO[frequency,direction,6] required")
        if depth <= 0 or not reference:
            raise ValueError("Positive reference depth and explicit reference description required")
        self.depth, self.reference, self.rho, self.g = depth, reference, rho, g
        self.dofs = list(DOFS)

    @classmethod
    def from_team(cls, sub, *, omega, depth, reference):
        # Compatible with solved WM M/A/B/C_hydro/F_exc layouts. Reorder all
        # force/matrix axes together if the team changes its DOF ordering.
        order = [list(sub.dofs).index(d) for d in DOFS]
        a = np.asarray(sub.A)[:,order][:,:,order]
        b = np.asarray(sub.B)[:,order][:,:,order]
        m = np.asarray(sub.M)[order][:,order]
        c = np.asarray(sub.C_hydro)[order][:,order]
        force = np.asarray(sub.F_exc)[:,:,order]
        holder = type('Matrices',(),{'M':m,'C_hydro':c})()
        rao = compute_rao(holder,omega,a,b,force)
        return cls(omega=omega,theta=sub.theta,rao=rao,depth=depth,reference=reference)

    def motion_response(self, omega, theta):
        _same_grid(self.omega,self.theta,omega,theta)
        return self.rao

    def on_grid(self,omega):
        """Explicit complex-linear frequency interpolation, no extrapolation.

        Keeps the team's original direction grid. Computing responses directly
        on the target grid is preferable near sharp response peaks/notches.
        """
        omega = np.asarray(omega,float)
        if np.any(np.diff(self.omega)<=0) or np.any(np.diff(omega)<=0) or omega[0]<self.omega[0]-1e-12 or omega[-1]>self.omega[-1]+1e-12:
            raise ValueError("Ordered frequencies within source response band required")
        if omega.shape==self.omega.shape and np.allclose(omega,self.omega):
            return self
        flat = self.rao.reshape(len(self.omega),-1)
        out = np.stack([np.interp(omega,self.omega,c.real)+1j*np.interp(omega,self.omega,c.imag) for c in flat.T],axis=1)
        return ResponseTable(omega=omega,theta=self.theta,rao=out.reshape(len(omega),len(self.theta),6),
                             depth=self.depth,reference=self.reference,rho=self.rho,g=self.g)

    def save(self,path):
        np.savez(path,omega=self.omega,theta=self.theta,rao=self.rao,depth=self.depth,
                 reference=self.reference,rho=self.rho,g=self.g,convention='exp(-i omega t)',dofs=np.array(DOFS))

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as d:
            if str(d['convention']) != 'exp(-i omega t)' or tuple(d['dofs']) != DOFS:
                raise ValueError("Unsupported phase/DOF convention")
            return cls(omega=d['omega'],theta=d['theta'],rao=d['rao'],depth=float(d['depth']),
                       reference=str(d['reference']),rho=float(d['rho']),g=float(d['g']))

```

## src/time_domain.py

```python
"""Separate time-domain six-DOF ODE and ideal-instrument generator.

NEVER uses RAOs, sensor_response(), spectral_gain(), or the inverse estimator.
Wave forces are known inputs; the ODE is integrated from rest with a ramp.
Shared primitive hull coefficients are still assumed physics, not independent
real-world validation. Sensors use the same first-order physical assumptions.
"""
import numpy as np
from sensors import Sensors
from submarine import skew


def _periodic_input(amplitudes, frequencies, fs, samples):
    """Sample the prescribed wave INPUT sum efficiently, not vehicle response.

    Equivalent to sum Re[A_f exp(-i 2pi f t)]. FFT is only an evaluation of
    prescribed sinusoidal forcing; no forward response division takes place.
    """
    bins = np.rint(frequencies/(fs/samples)).astype(int)
    grid = np.fft.rfftfreq(samples,1/fs)
    if np.any(bins<=0) or np.any(bins>=len(grid)-1) or not np.allclose(grid[bins],frequencies):
        raise ValueError('Wave inputs must lie on interior Fourier bins')
    coef = np.zeros((len(grid),amplitudes.shape[1]),complex)
    coef[bins] = samples/2*np.conj(amplitudes)
    return np.fft.irfft(coef,n=samples,axis=0)


def simulate(sea, hull, *, fs=2., n_samples=8192, dt=.1,
             settle_seconds=400., ramp_seconds=60., config=None):
    """Return instrument time series and separate state diagnostics.

    X=[q,velocity], as in WM's time-domain model; coupled RK4, zero initial
    state. No steady-state amplitudes/initial conditions are imported.
    Inputs are sampled at RK4 half steps. Decimate to the ideal sensor rate;
    no noise or filters. Default band is below sensor Nyquist.
    """
    config = config or Sensors()
    if dt<=0 or fs<=0 or n_samples<4 or settle_seconds<ramp_seconds or ramp_seconds<=0:
        raise ValueError('Positive rates/ramp and sufficient settling required')
    sample_step = int(round(1/(fs*dt)))
    settle_steps = int(round(settle_seconds/dt))
    if sample_step<1 or not np.isclose(sample_step*dt,1/fs) or not np.isclose(settle_steps*dt,settle_seconds):
        raise ValueError('Integration and output times must align')
    if sea.f.max()>=fs/2 or not np.isclose(sea.delta_f,fs/n_samples):
        raise ValueError('Wave grid must match record resolution and lie below sensor Nyquist')
    if not np.allclose(hull.A,hull.A[0]) or not np.allclose(hull.B,hull.B[0]):
        raise ValueError('This baseline ODE uses constant added mass and damping; memory needs a team solver')
    total_record_steps=n_samples*sample_step
    half_samples=2*total_record_steps
    forcing_amplitudes=np.einsum('fdk,fd->fk',hull.F_exc,sea.complex_components)
    forcing=_periodic_input(forcing_amplitudes,sea.f,2/dt,half_samples)
    r_p=np.asarray(config.pressure_position)
    if hull.depth-r_p[2]<=0:
        raise ValueError('Mean pressure port must be submerged')
    phase=sea.k[:,None]*(r_p[0]*np.cos(sea.theta)[None,:]+r_p[1]*np.sin(sea.theta)[None,:])
    pressure_per_amplitude=hull.rho*hull.g*np.exp(-sea.k[:,None]*(hull.depth-r_p[2]))*np.exp(1j*phase)
    pressure_amplitudes=np.sum(pressure_per_amplitude*sea.complex_components,axis=1)[:,None]
    pressure_input=_periodic_input(pressure_amplitudes,sea.f,2/dt,half_samples)[:,0]
    mass=hull.M+hull.A[0]
    damping,restoring=hull.B[0],hull.C_hydro
    inv_mass=np.linalg.inv(mass)
    system=np.block([[np.zeros((6,6)),np.eye(6)],[-inv_mass@restoring,-inv_mass@damping]])
    t_half=np.arange(2*(settle_steps+total_record_steps)+1)*dt/2
    ramp=np.where(t_half<ramp_seconds,.5*(1-np.cos(np.pi*np.minimum(t_half,ramp_seconds)/ramp_seconds)),1.)
    X=np.zeros(12)
    states=np.zeros((n_samples,12))
    accelerations=np.zeros((n_samples,6))
    pressures=np.zeros(n_samples)
    out=0
    for step in range(settle_steps+total_record_steps):
        h=2*step
        force0=ramp[h]*forcing[h%half_samples]
        derivative0=system@X+np.r_[np.zeros(6),inv_mass@force0]
        if step>=settle_steps and (step-settle_steps)%sample_step==0:
            states[out]=X
            accelerations[out]=derivative0[6:]
            pressures[out]=ramp[h]*pressure_input[h%half_samples]
            out+=1
        half_force=ramp[h+1]*forcing[(h+1)%half_samples]
        end_force=ramp[h+2]*forcing[(h+2)%half_samples]
        add_half=np.r_[np.zeros(6),inv_mass@half_force]
        add_end=np.r_[np.zeros(6),inv_mass@end_force]
        k1=derivative0
        k2=system@(X+dt/2*k1)+add_half
        k3=system@(X+dt/2*k2)+add_half
        k4=system@(X+dt*k3)+add_end
        X+=dt/6*(k1+2*k2+2*k3+k4)
    if out!=n_samples:
        raise RuntimeError('Sensor record length mismatch')
    q,v=states[:,:6],states[:,6:]
    r_imu=np.asarray(config.imu_position)
    # First-order instrument equations in TIME, independent implementation of
    # the spectral sensor_response map. Acceleration comes from Newton/ODE.
    acc=accelerations[:,:3]-accelerations[:,3:]@skew(r_imu).T
    acc[:,0]-=hull.g*q[:,4]
    acc[:,1]+=hull.g*q[:,3]
    acc[:,2]+=hull.g
    port_displacement=q[:,:3]-q[:,3:]@skew(r_p).T
    pressure=hull.rho*hull.g*(hull.depth-r_p[2]-port_displacement[:,2])+pressures
    y=np.column_stack([pressure,acc,v[:,3:]])
    t=settle_seconds+np.arange(n_samples)/fs
    return t,y,states,accelerations

```

## src/sea.py

```python
"""KD/LR finite-band normalization + WM's Hz grid / cos-2s spreading.

One stationary, directionally spread JONSWAP system. No measurement noise.
"""
from dataclasses import dataclass
import numpy as np

G = 9.81


def jonswap(f, Hs, Tp, gamma=3.3):
    f = np.asarray(f, dtype=float)
    if f.ndim != 1 or len(f) < 2 or np.any(f <= 0):
        raise ValueError("Positive 1D frequency grid required")
    df = np.diff(f)
    if not np.allclose(df, df[0]) or df[0] <= 0:
        raise ValueError("Uniform increasing frequency grid required")
    if Hs <= 0 or Tp <= 0 or gamma < 1:
        raise ValueError("Hs and Tp positive, gamma >= 1 required")
    fp = 1.0 / Tp
    sigma = np.where(f <= fp, .07, .09)
    peak = np.exp(-.5 * ((f / fp - 1) / sigma)**2)
    # Work in log space to avoid underflow for trial peaks outside the band.
    log_shape = -5 * np.log(f) - 1.25 * (fp / f)**4 + peak * np.log(gamma)
    shape = np.exp(log_shape - np.max(log_shape))
    return shape * (Hs / 4)**2 / (np.sum(shape) * df[0])


@dataclass(frozen=True)
class FixedSea:
    gamma: float = 3.3
    mean_direction: float = np.deg2rad(30.0)
    spread_s: float = 10.0

    def fractions(self, theta):
        theta = np.asarray(theta, float)
        if self.spread_s < 0:
            raise ValueError("Spreading exponent must be nonnegative")
        # Full-circle, equal-width bins, as used by WM. Wrap the angle.
        ordered = np.sort(np.mod(theta, 2 * np.pi))
        gaps = np.diff(np.r_[ordered, ordered[0] + 2 * np.pi])
        if len(theta) < 2 or not np.allclose(gaps, 2 * np.pi / len(theta)):
            raise ValueError("Directions must cover the full circle evenly")
        delta = np.angle(np.exp(1j * (theta - self.mean_direction)))
        p = np.maximum(np.cos(delta / 2), 0)**(2 * self.spread_s)
        return p / p.sum()  # discrete normalization preserves Hs exactly


class Sea:
    """The team-style f/omega/theta/k/phi interface, with explicit Hs and Tp."""
    def __init__(self, *, freq=None, omega=None, theta=None, Hs=2., Tp=8.,
                 fixed=None, seed=42):
        if (freq is None) == (omega is None):
            raise ValueError("Give either freq (Hz) or omega (rad/s)")
        self.f = np.asarray(freq if freq is not None else np.asarray(omega) / (2*np.pi), float)
        self.omega = 2 * np.pi * self.f
        self.theta = np.asarray(theta if theta is not None else np.linspace(0, 2*np.pi, 36, endpoint=False), float)
        self.fixed = fixed or FixedSea()
        self.Hs, self.Tp = Hs, Tp
        self.S_f = jonswap(self.f, Hs, Tp, self.fixed.gamma)
        self.delta_f = self.f[1] - self.f[0]
        self.delta_theta = 2 * np.pi / len(self.theta)
        self.p = self.fixed.fractions(self.theta)
        self.k = self.omega**2 / G
        self.phi = np.random.default_rng(seed).uniform(0, 2*np.pi, (len(self.f), len(self.theta)))
        self.a = np.sqrt(2 * self.S_f[:, None] * self.delta_f * self.p[None, :])

    def sea_spectrum(self):
        D = self.p / self.delta_theta
        S = self.S_f[:, None] * D[None, :]
        return self.S_f, D, S, self.a, 4*np.sqrt(np.sum(self.a**2)/2)

    def superposition(self, x, y, t):
        phase = self.k[:, None] * (x*np.cos(self.theta)[None, :] + y*np.sin(self.theta)[None, :])
        return np.sum(self.a * np.cos(phase - self.omega[:, None]*t + self.phi)), self.a

    @property
    def complex_components(self):
        return self.a * np.exp(1j * self.phi)

```

## src/sensors.py

```python
"""Ideal first-order sensor layer: separate from vehicle and inverse model.

No measurement noise, filters, bias, quantization or differentiation of samples.
"""
from dataclasses import dataclass
import numpy as np
from submarine import skew

CHANNELS = ('pressure_Pa','accel_x_m_s2','accel_y_m_s2','accel_z_m_s2',
            'gyro_x_rad_s','gyro_y_rad_s','gyro_z_rad_s')


@dataclass(frozen=True)
class Sensors:
    # Off-centre IMU demonstrates rotational lever-arm acceleration.
    imu_position: tuple = (10.,0.,0.)
    pressure_position: tuple = (10.,0.,5.)


def sensor_response(model, omega, theta, config=None):
    """Response per metre of surface amplitude at the reference origin.

    Small-angle, level mean attitude; body axes coincide with earth mean axes.
    Pressure = incident wave + first-order hydrostatic change at moving port.
    Hull-scattered pressure is not supplied by this baseline.
    """
    config = config or Sensors()
    q = model.motion_response(omega,theta)
    w = np.asarray(omega)[:,None,None]
    r_imu, r_p = np.asarray(config.imu_position), np.asarray(config.pressure_position)
    if model.depth-r_p[2] <= 0:
        raise ValueError("Mean pressure port must be submerged")
    sensor_displacement = q[:,:,:3] - np.einsum('ij,fdj->fdi',skew(r_imu),q[:,:,3:])
    acc = -w*w*sensor_displacement
    # R^T(a-g): first-order tilt contribution; DC accel_z is +g.
    acc[:,:,0] -= model.g*q[:,:,4]
    acc[:,:,1] += model.g*q[:,:,3]
    gyro = -1j*w*q[:,:,3:]
    port_displacement = q[:,:,:3] - np.einsum('ij,fdj->fdi',skew(r_p),q[:,:,3:])
    k = np.asarray(omega)**2/model.g
    spatial = k[:,None]*(r_p[0]*np.cos(theta)[None,:]+r_p[1]*np.sin(theta)[None,:])
    incident = model.rho*model.g*np.exp(-k[:,None]*(model.depth-r_p[2]))*np.exp(1j*spatial)
    pressure = incident-model.rho*model.g*port_displacement[:,:,2]
    return np.concatenate([pressure[:,:,None],acc,gyro],axis=2)


def spectral_gain(transfer, fractions):
    return np.sum(np.abs(transfer)**2*np.asarray(fractions)[None,:,None],axis=1)


def synthesise(sea, model, *, fs, n_samples, config=None):
    """Exact Fourier-grid periodic steady state; no startup transient.

    Uses the team's fixed-amplitude/random-phase components. Distinct directions
    share each frequency; cross terms in an individual record are retained.
    """
    config = config or Sensors()
    full_f = np.fft.rfftfreq(n_samples,1/fs)
    bins = np.rint(sea.f/(fs/n_samples)).astype(int)
    if np.any(bins<=0) or np.any(bins>=len(full_f)-1) or not np.allclose(full_f[bins],sea.f):
        raise ValueError("Sea frequencies must be interior FFT bins")
    if not np.isclose(sea.delta_f,fs/n_samples):
        raise ValueError("Use consecutive Fourier bins at the record resolution")
    transfer = sensor_response(model,sea.omega,sea.theta,config)
    Y = np.einsum('fdc,fd->fc',transfer,sea.complex_components)
    Q = np.einsum('fdk,fd->fk',model.motion_response(sea.omega,sea.theta),sea.complex_components)
    def trace(z):
        coef = np.zeros((len(full_f),z.shape[1]),complex)
        # rfft uses exp(-i wt) analysis: its positive-frequency coefficients
        # are conjugates of our exp(-i wt) PHYSICAL complex amplitudes.
        coef[bins] = n_samples/2*np.conj(z)
        return np.fft.irfft(coef,n=n_samples,axis=0)
    y = trace(Y)
    y[:,0] += model.rho*model.g*(model.depth-config.pressure_position[2])
    y[:,3] += model.g
    return np.arange(n_samples)/fs,y,trace(Q)


def periodogram(y, fs, f):
    y = np.asarray(y,float)
    if y.ndim != 2 or len(y)<4 or fs<=0 or not np.all(np.isfinite(y)):
        raise ValueError("Finite sample-by-channel array and positive sampling rate required")
    n = len(y)
    grid = np.fft.rfftfreq(n,1/fs)
    bins = np.rint(np.asarray(f)/(fs/n)).astype(int)
    if np.any(bins<=0) or np.any(bins>=len(grid)-1) or not np.allclose(grid[bins],f):
        raise ValueError("Fit grid must consist of interior record Fourier bins")
    coef = np.fft.rfft(y-y.mean(axis=0),axis=0)
    return 2*np.abs(coef[bins])**2/(fs*n)

```

## Supporting source: src/run_pressure_array_experiment.py

```python
"""Six pressure ports + raw IMU: local wave recovery and spatial interpolation.

Explicitly assumed symmetric geometry. Existing hull and sensor physics stay
unchanged. No JONSWAP parameters or truth states enter the inverse. Linear
interpolation is scored against hidden surface truth at unsensed locations.
"""
import argparse,json,hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.spatial import Delaunay
from sea import Sea,FixedSea
from submarine import Submarine,ResponseTable
from pressure_array import symmetric_six_port_layout,simulate_array,reconstruct_above_ports,wave_interpolate,reconstruct_surface_joint

HERE=Path(__file__).resolve().parent


def make_sea(table,kind,seed):
    f=table.omega/(2*np.pi)
    if kind=='single_system':return Sea(freq=f,theta=table.theta,Hs=2.,Tp=8.,fixed=FixedSea(mean_direction=np.deg2rad(30),spread_s=10),seed=seed)
    a=Sea(freq=f,theta=table.theta,Hs=np.sqrt(2),Tp=6.5,fixed=FixedSea(mean_direction=np.deg2rad(20),spread_s=8),seed=seed)
    b=Sea(freq=f,theta=table.theta,Hs=np.sqrt(2),Tp=10.5,fixed=FixedSea(mean_direction=np.deg2rad(-70),spread_s=18),seed=seed+901)
    return SimpleNamespace(f=a.f,omega=a.omega,k=a.k,theta=a.theta,delta_f=a.delta_f,complex_components=a.complex_components+b.complex_components)


def truth_at(sea,xy,time,band):
    n=len(time);fs=1/(time[1]-time[0]);nf=n//2+1;xy=np.asarray(xy)
    z=np.empty((len(sea.f),len(xy)),complex)
    for start in range(0,len(sea.f),48):
        sl=slice(start,start+48)
        phase=sea.k[sl,None,None]*(np.cos(sea.theta)[None,:,None]*xy[None,None,:,0]+np.sin(sea.theta)[None,:,None]*xy[None,None,:,1])
        z[sl]=np.einsum('fd,fdp->fp',sea.complex_components[sl],np.exp(1j*phase))
    z*=np.exp(-1j*sea.omega*time[0])[:,None]
    z[(sea.f<band[0])|(sea.f>band[1])]=0
    coeff=np.zeros((nf,len(xy)),complex);coeff[np.rint(sea.f*n/fs).astype(int)]=n/2*z.conj()
    return np.fft.irfft(coeff,n=n,axis=0)


def errors(estimate,truth):
    delta=estimate-truth;mse=np.mean(delta**2,axis=0);variance=np.var(truth,axis=0)
    r2=1-mse/variance
    return {'pooled_R2':float(1-np.sum(delta**2)/np.sum((truth-truth.mean(axis=0))**2)),
            'minimum_point_R2':float(np.min(r2)),'median_point_R2':float(np.median(r2)),
            'pooled_RMSE_m':float(np.sqrt(np.mean(delta**2))),'maximum_point_RMSE_m':float(np.sqrt(mse.max()))},r2


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    cfg=symmetric_six_port_layout();cfg.save(out/'assumed_geometry.json')
    table=ResponseTable.load(HERE/'results-joint/response_table.npz');hull=Submarine(omega=table.omega,theta=table.theta,build_response=False)
    band=(.05,.30);ports=np.asarray(cfg.positions);xy=ports[:,:2]
    unique_xy,inverse=np.unique(xy,axis=0,return_inverse=True)
    # Two vertically aligned ports give two records at ONE horizontal location.
    combine=np.zeros((6,len(unique_xy)))
    for j in range(len(unique_xy)):
        members=np.flatnonzero(inverse==j);combine[members,j]=1/len(members)
    xx,yy=np.meshgrid(np.linspace(-35,35,43),np.linspace(-5,5,23));grid=np.column_stack([xx.ravel(),yy.ravel()])
    tri=Delaunay(unique_xy);simplex=tri.find_simplex(grid);inside=simplex>=0;targets=grid[inside]
    transforms=tri.transform[simplex[inside]];bary=np.einsum('pij,pj->pi',transforms[:,:2,:],targets-transforms[:,2,:]);bary=np.column_stack([bary,1-bary.sum(axis=1)])
    W=np.zeros((len(targets),len(unique_xy)))
    W[np.arange(len(targets))[:,None],tri.simplices[simplex[inside]]]=bary
    assert np.allclose(W.sum(axis=1),1) and np.allclose(W@unique_xy,targets)
    protocol={'sensor_count':6,'layout_status':cfg.placement_status,'imu_channels_retained':6,'phase_seeds':[42,81,2026],
              'seas':['single_system','crossing_two_systems'],'pressure_noise_std_Pa':[0.,1.,10.],
              'noise_status':'Hypothetical independent white pressure noise for sensitivity only; not sensor specification. IMU remains ideal.',
              'band_Hz':band,'surface_domain':'Within the convex hull of five unique horizontal sensor locations; no extrapolation',
              'interpolation':'Piecewise linear over Delaunay triangles; no spectral shape or directional prior',
              'joint_surface_inverse':'Six pressures plus raw IMU, known nominal complex response, isotropic equal-direction prior, per-frequency rank tolerance 1e-9; no JONSWAP spectrum',
              'wave_interpolation':'Deep-water dispersion covariance J0(k*r); isotropic angular prior, relative eigen cutoff 1e-8; no JONSWAP frequency shape or fitted sea parameters',
              'source_sha256':{name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in ['pressure_array.py','time_domain.py','submarine.py']}}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    records=out/'records';records.mkdir(exist_ok=True);runs=[]
    for kind in protocol['seas']:
        for seed in protocol['phase_seeds']:
            sea=make_sea(table,kind,seed);label=f'{kind}_seed{seed}'
            t,y,X,_=simulate_array(sea,hull,cfg,fs=2.,n_samples=8192,dt=.1,settle_seconds=400.)
            np.savetxt(records/(label+'.csv'),np.column_stack([t,y]),delimiter=',',header='time_s,'+','.join(cfg.channels),comments='')
            # Truth calculated separately, never supplied to the inverse.
            port_truth=truth_at(sea,xy,t,band);surface_truth=truth_at(sea,targets,t,band)
            peak=float(np.rad2deg(np.max(abs(X[:,3:6]))))
            for noise in protocol['pressure_noise_std_Pa']:
                measured=y.copy()
                if noise:
                    measured[:,:6]+=np.random.default_rng(seed+round(noise)*103).normal(0,noise,(len(t),6))
                local=reconstruct_above_ports(t,measured,cfg,depth=hull.depth,rho=hull.rho,g=hull.g,band=band)
                surface=(local@combine)@W.T
                wave_surface=wave_interpolate(t,local,cfg.positions,targets,g=hull.g,band=band)
                joint_surface,joint_diagnostics=reconstruct_surface_joint(t,measured,cfg,table,targets,band=band)
                local_metrics,_=errors(local,port_truth);spatial_metrics,r2map=errors(surface,surface_truth)
                wave_metrics,wave_r2map=errors(wave_surface,surface_truth)
                joint_metrics,joint_r2map=errors(joint_surface,surface_truth)
                runs.append({'sea':kind,'seed':seed,'noise_std_Pa':noise,'local_ports':local_metrics,'interpolated_surface':spatial_metrics,'wave_interpolated_surface':wave_metrics,'joint_surface':joint_metrics,'joint_diagnostics':joint_diagnostics,'peak_rotation_deg':peak,'small_angle_screen_pass':peak<5})
                print(f'{label}, noise={noise:g} Pa: local R2={local_metrics["pooled_R2"]:.6f}, interpolated R2={spatial_metrics["pooled_R2"]:.4f}, wave-based R2={wave_metrics["pooled_R2"]:.4f}, joint R2={joint_metrics["pooled_R2"]:.4f}, spatial RMSE={spatial_metrics["pooled_RMSE_m"]:.3f} m',flush=True)
                np.savez_compressed(out/(label+f'_noise{noise:g}_map.npz'),x=targets[:,0],y=targets[:,1],R2=r2map,wave_R2=wave_r2map,joint_R2=joint_r2map)
                if seed==42 and noise==0:
                    nearest=np.argmin(np.sum((targets-np.array([17.,2.]))**2,axis=1));window=(t>=500)&(t<=560)
                    np.savez(out/(label+'_example.npz'),time_s=t[window]-500,truth=surface_truth[window,nearest],interpolated=surface[window,nearest],wave_interpolated=wave_surface[window,nearest],joint=joint_surface[window,nearest],location_m=targets[nearest])
            (out/'runs.json').write_text(json.dumps(runs,indent=2)+'\n')
    summary=[]
    for kind in protocol['seas']:
        for noise in protocol['pressure_noise_std_Pa']:
            rs=[r for r in runs if r['sea']==kind and r['noise_std_Pa']==noise]
            summary.append({'sea':kind,'noise_std_Pa':noise,'records':len(rs),
                            'minimum_local_pooled_R2':min(r['local_ports']['pooled_R2'] for r in rs),
                            'minimum_interpolated_pooled_R2':min(r['interpolated_surface']['pooled_R2'] for r in rs),
                            'maximum_interpolated_pooled_RMSE_m':max(r['interpolated_surface']['pooled_RMSE_m'] for r in rs),
                            'minimum_wave_based_pooled_R2':min(r['wave_interpolated_surface']['pooled_R2'] for r in rs),
                            'maximum_wave_based_pooled_RMSE_m':max(r['wave_interpolated_surface']['pooled_RMSE_m'] for r in rs),
                            'minimum_joint_pooled_R2':min(r['joint_surface']['pooled_R2'] for r in rs),
                            'maximum_joint_pooled_RMSE_m':max(r['joint_surface']['pooled_RMSE_m'] for r in rs),
                            'minimum_unsensed_point_R2':min(r['interpolated_surface']['minimum_point_R2'] for r in rs)})
    report={'protocol':protocol,'geometry':{'positions_m':cfg.positions,'names':cfg.names,'unique_horizontal_locations':len(unique_xy),'interior_grid_points':len(targets)},
            'summary':summary,'runs':runs,
            'limits':['Assumed geometry; not verified sponsor sensor positions','Same primitive linear capsule/ideal incident-pressure physics; no hull scattering, speed or calibration error','No JONSWAP parameters, phases, sea truth or integrated states enter the reconstruction','Time histories are reconstructed within a finite band, not future-wave prediction','Six ports provide five unique horizontal surface locations in this assumed layout','Pressure noise levels hypothetical; IMU noiseless; limited to six phase records','Linear interpolation uses no wave dispersion constraint and is not exact between ports','Wave-based interpolation uses an isotropic angular prior; it supplies unmeasured spatial structure and does not establish physical uniqueness']}
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(summary,indent=2))

if __name__=='__main__':main()

```

## Supporting source: work/independent_wave_adapter.py

```python
"""CFDwavemaker wave/pressure adapter. Contains no inverse implementation.

CFDwavemaker supplies elevations and incident pressures. Existing capsule
strip loads/matrices, RK4 dynamics and first-order moving-port instrument
model remain shared assumptions. No RAOs are used for forward motion.
"""
from pathlib import Path
import ctypes,os
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
LIB=ROOT/'work/CFDwavemaker/builds/macos/libCFDwavemaker.dylib'
BATCH=ROOT/'work/CFDwavemaker/builds/macos/libCFDbatch.dylib'

def upstream_components():
    path=ROOT/'work/CFDwavemaker/tests/t008_3D_irregular_sea/waveinput_linear.dat'
    rows=[]
    for line in path.read_text().splitlines():
        try:row=list(map(float,line.split()))
        except ValueError:continue
        if len(row)==5:rows.append(row)
    table=np.array(rows)
    assert table.shape==(312,5)
    return table,path

def write_case(path,kind,seed):
    table,upstream=upstream_components();w=table[:,0].copy();a=table[:,1].copy();phi=table[:,3].copy();theta=table[:,4].copy()+np.deg2rad(30)
    if kind=='gaussian_single_peak':
        f=w/(2*np.pi);s=np.exp(-.5*((f-.125)/.018)**2)
        a=np.sqrt(s)
    elif kind!='upstream_single_system':raise ValueError(kind)
    a*=2/(4*np.sqrt(np.sum(a*a)/2))
    if seed is not None:phi=np.random.default_rng(seed).uniform(0,2*np.pi,len(a))
    path.mkdir(parents=True,exist_ok=True)
    lines=['@v301','[wave type]','irregular','[general input data]','depth 1000','mtheta 0','rho 1025','gravity 9.81','normalize 0','amplify 1',
           '[wave input irregular]','kinematics_surf_extrap 0','nfreq 312','ndir 0','# OMEGA A PHASE THETA']
    lines.extend(' '.join(f'{v:.17g}' for v in row) for row in np.c_[w,a,phi,theta])
    (path/'waveinput.dat').write_text('\n'.join(lines)+'\n')
    np.savez(path/'components_evaluator_only.npz',omega=w,amplitude=a,phase=phi,theta=theta)
    return w,a,phi,theta

class IndependentWave:
    def __init__(self,folder):
        self.native=ctypes.CDLL(str(LIB))
        self.batch=ctypes.CDLL(str(BATCH))
        self.folder=Path(folder).resolve()
        self.native.wave_Initialize.restype=ctypes.c_int
        self.native.wave_Cleanup.restype=ctypes.c_int
        nd=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
        for name in ['cfd_batch_surface','cfd_batch_pressure']:
            fun=getattr(self.batch,name);fun.argtypes=[ctypes.c_int,nd,ctypes.c_int,nd,nd];fun.restype=None
        old=Path.cwd();os.chdir(self.folder)
        try:self.native.wave_Initialize()
        finally:os.chdir(old)

    def surface(self,t,xy):
        t=np.ascontiguousarray(t,dtype=float);xy=np.ascontiguousarray(xy,dtype=float)
        output=np.empty((len(t),len(xy)));self.batch.cfd_batch_surface(len(t),t,len(xy),xy,output)
        return output

    def pressure(self,t,xyz):
        t=np.ascontiguousarray(t,dtype=float);xyz=np.ascontiguousarray(xyz,dtype=float)
        output=np.empty((len(t),len(xyz)));self.batch.cfd_batch_pressure(len(t),t,len(xyz),xyz,output)
        return output

    def close(self):self.native.wave_Cleanup()

def component_excitation(hull,omega,theta):
    """Same primitive capsule strip force law, evaluated at arbitrary directions."""
    h=hull.hull;dx=h.length/h.strips
    x=-h.length/2+(np.arange(h.strips)+.5)*dx
    cap=np.maximum(abs(x)-(h.length/2-h.radius),0)
    dV=np.pi*(h.radius**2-cap**2)*dx
    axial_ca=.5*(4*np.pi*h.radius**3/3)/hull.volume
    ct=np.cos(theta);st=np.sin(theta);k=omega**2/hull.g
    basis=np.exp(1j*k[:,None]*ct[:,None]*x)
    v0=basis@dV;v1=basis@(dV*x)
    factor=hull.rho*omega**2*np.exp(-k*h.depth)
    force=np.zeros((len(omega),6),complex)
    force[:,0]=-1j*factor*(1+axial_ca)*ct*v0
    force[:,1]=-1j*factor*2*st*v0
    force[:,2]=-factor*2*v0
    force[:,4]=factor*2*v1
    force[:,5]=-1j*factor*2*st*v1
    return force

def motion_from_components(hull,omega,amplitude,phase,theta,*,fs=2,n_samples=8192,dt=.1,settle=400):
    force_amp=component_excitation(hull,omega,theta)*(amplitude*np.exp(1j*phase))[:,None]
    total_steps=round((settle+n_samples/fs)/dt)
    half_times=np.arange(2*total_steps+1)*dt/2
    forcing=np.empty((len(half_times),6))
    for start in range(0,len(half_times),2048):
        times=half_times[start:start+2048]
        forcing[start:start+2048]=(np.exp(-1j*times[:,None]*omega)@force_amp).real
    ramp=np.where(half_times<60,.5*(1-np.cos(np.pi*np.minimum(half_times,60)/60)),1)
    forcing*=ramp[:,None]
    mass=hull.M+hull.A[0];im=np.linalg.inv(mass)
    system=np.block([[np.zeros((6,6)),np.eye(6)],[-im@hull.C_hydro,-im@hull.B[0]]])
    drive=np.c_[np.zeros((len(forcing),6)),forcing@im.T]
    state=np.zeros(12);X=np.empty((n_samples,12));out=0
    for step in range(total_steps):
        i=2*step
        if step>=round(settle/dt) and (step-round(settle/dt))%round(1/(fs*dt))==0:
            X[out]=state;out+=1
        k1=system@state+drive[i]
        k2=system@(state+dt/2*k1)+drive[i+1]
        k3=system@(state+dt/2*k2)+drive[i+1]
        k4=system@(state+dt*k3)+drive[i+2]
        state+=dt/6*(k1+2*k2+2*k3+k4)
    assert out==n_samples
    return settle+np.arange(n_samples)/fs,X

def pressure_record(package,t,X,config,*,depth=15,rho=1025,g=9.81):
    r=np.asarray(config.positions);xyz=r.copy();xyz[:,2]-=depth
    incident=package.pressure(t,xyz)
    port_z=X[:,2,None]+X[:,3,None]*r[None,:,1]-X[:,4,None]*r[None,:,0]
    return rho*g*(depth-r[None,:,2]-port_z)+incident

```

## Supporting source: work/replay_saved_case.py

```python
"""Pressure-only relearning/causal replay; load evaluation truth afterward."""
from pathlib import Path
import argparse,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from hybrid_pressure_live import HybridPressureLive
from pressure_array import PressureArray
from pressure_only_live import Calibration
from submarine import ResponseTable
from run_pressure_array_experiment import errors

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--case',default='upstream_single_system_seed731927')
    parser.add_argument('--all',action='store_true')
    args=parser.parse_args()
    base=ROOT/'outputs/independent-wave-package'
    original=json.loads((base/'results.json').read_text())
    available={f"{r['case']}_seed{r['phase_seed']}":r for r in original['runs']}
    labels=sorted(available) if args.all else [args.case]
    if any(label not in available for label in labels):parser.error('Unknown case; choose '+', '.join(available))
    model=ResponseTable.load(ROOT/'src/results-joint/response_table.npz')
    cfg=PressureArray.load(base/'assumed_geometry.json')
    # Requested surface locations are calibration metadata, not measured truth.
    targets=Calibration.load(base/'cases'/labels[0]/'calibration.npz').targets
    result=[]
    for label in labels:
        folder=base/'cases'/label
        record=np.loadtxt(folder/'pressures.csv',delimiter=',',skiprows=1)
        live=HybridPressureLive(model,cfg,targets)
        predictions=[];times=[];start=time.perf_counter()
        for row in record:
            estimate=live.push(row[0],row[1:])
            if estimate is not None:predictions.append(estimate);times.append(row[0])
        # No reference file or true surface entered the learner/stream above.
        with np.load(folder/'reference.npz') as reference:
            np.testing.assert_array_equal(np.asarray(times),reference['time_s'])
            np.testing.assert_array_equal(targets,reference['targets'])
            metric,_=errors(np.asarray(predictions),reference['truth'].astype('float64'))
        expected=available[label]['hybrid']
        for key,value in metric.items():
            np.testing.assert_allclose(value,expected[key],rtol=0,atol=5e-8,err_msg=label+' '+key)
        for key in ['mean_direction_deg','spread_s','Tp_s','Hs_m']:
            np.testing.assert_allclose(getattr(live.fit,key),available[label]['learned_prior'][key],rtol=1e-4,atol=1e-3,err_msg=label+' prior '+key)
        row={'case':label,'metrics':metric,'learned_prior':live.fit.to_dict(),'score_samples':len(times),
             'scored_surface_points':len(targets),'elapsed_s':time.perf_counter()-start,
             'matches_original_metrics_within_5e_minus8':True,
             'reference_used_only_after_prediction':True}
        result.append(row);print(json.dumps(row),flush=True)
    out=ROOT/'outputs/phone-replay-results';out.mkdir(exist_ok=True)
    (out/'replay-results.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()

```
