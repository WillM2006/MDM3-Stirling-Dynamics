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
