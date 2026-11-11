# Bias diagnosis of the unchanged corrected-diagonal estimator

The diagnostic identifies a systematic downward contribution from using the same recording to set the inverse weights and supply the fitted data. Freezing the prior scale barely changes it. The mean sensor-response mismatch also matters, and has the opposite sign for long-period waves. Their cancellation explains why a small net bias need not imply an unbiased estimator.

No implementation change is adopted. M = 8.785033150237455, tau = 5, production code, saved point estimates, main, the report and Drive data remain unchanged.

## Design and interpretation

The analysis reuses 200 test recordings per condition (250xxx) and 400 independent reference sensor spectra per condition (230xxx and 240xxx). It uses the original response builder and cached hydrodynamics, not a new physical simulator. It reconstructs the energy gain for each existing corrected-diagonal inverse and verifies that applying that gain to its own recording reproduces the saved wave-energy estimate. Twelve headings are primary; 36 are a sensitivity analysis.

For each condition, G_i is the energy gain computed from recording i, d_i its measured sensor spectral vector, mu an independently estimated mean sensor vector, and m the predicted vector A x_true. Truth is used only to evaluate bias; neither weights nor independent sensor means use wave truth. The predicted m is a scoring counterfactual, not data supplied to a proposed production method.

The original mean estimate is the average of G_i d_i. The independent-donor mean is the average of G_i d_j over every i != j. These 39,800 pairings reuse 200 records; they are not 39,800 independent experiments. Their difference estimates the contribution of same-record gain/data dependence while preserving the marginal distribution of gains and data.

The reported four terms sum exactly to the observed mean energy bias:

1. **Model/prior term:** average G_i m minus true energy. This measures recovery of the model's own expected spectrum with the actual gains; regularisation and numerical effects are included.
2. **Mean-response mismatch:** average G_i (mu - m), using independent sensor reference records. This does not identify the physical source of the mismatch.
3. **Reference/test difference:** independent-donor mean minus average G_i mu. This retains finite-cohort differences instead of incorrectly assigning all residual variation to physical mismatch or data-dependent weighting.
4. **Same-record coupling:** original mean estimate minus independent-donor mean.

The gain includes both inverse-variance weights and the data-derived prior scale. A second, explicitly diagnostic calculation freezes prior scales at the independent reference spectral mean but retains each record's variance weights. This isolates whether the prior adaptation materially explains the coupling. It does not change the saved point estimator.

Confidence intervals use 1,000 whole-record bootstrap draws, preserving the gain/data pairing within each resampled test record and separately resampling the reference records. They are pointwise, conditional on M and the response model. Fixed-prior sensitivity intervals additionally condition on its frozen reference scales. This is retrospective diagnosis on previously examined simulations, not fresh validation.

## Numerical results

All values below are percentage points of true retained-band wave energy, with pointwise 95% bootstrap intervals. Negative means downward bias; these are not wave-height percentage errors.

| Contribution, 12 headings | Baseline | Long period | Weak waves |
|---|---:|---:|---:|
| Model/prior | -0.004 [-0.004, -0.004] | +0.001 [+0.001, +0.001] | -0.004 [-0.004, -0.004] |
| Independent mean mismatch | -1.236 [-2.267, -0.127] | +3.386 [+2.031, +4.846] | -2.090 [-3.255, -0.912] |
| Reference/test difference | -1.228 [-3.376, +0.738] | -0.945 [-3.738, +1.897] | -0.601 [-2.678, +1.469] |
| Same-record coupling | -3.676 [-3.774, -3.554] | -2.167 [-2.262, -2.057] | -3.674 [-3.754, -3.555] |
| Observed total bias | -6.144 [-7.947, -4.430] | +0.275 [-2.113, +2.620] | -6.369 [-7.924, -4.649] |
| Independent-donor total bias | -2.468 [-4.267, -0.763] | +2.442 [+0.028, +4.805] | -2.695 [-4.303, -0.958] |

Same-record coupling is negative with intervals excluding zero in every condition and at both heading grids. At 36 headings its contributions are −3.246, −2.039 and −3.271 points; the respective total biases are −4.954, +0.585 and −5.074 points. Mean mismatch is less precisely known; the baseline 36-heading interval includes zero. Reference/test differences have intervals crossing zero in all cases and should not be treated as established distribution shifts.

Freezing prior scales changes the primary coupling contributions from −3.676 to −3.684, −2.167 to −2.170 and −3.674 to −3.680 points. This makes the variance weights, rather than prior-scale adaptation, the main identified source of this coupling in the tested cases. Prior/model energy effects themselves are under 0.005 percentage points.

The long-period diagnostic changes mean bias from +0.275% to +2.442% when gains and data are independently paired. This is direct evidence against treating removal of the downward contribution as a guaranteed overall improvement. It does not establish the recovery or coverage of a practical independent-weight implementation.

## What should change next

**Do not retune M or subtract the measured biases.** The sensor-level variance calibration and this inverse weighting issue have different meanings. These results do not establish a new M, a bias offset, or an interval inflation factor.

**Prioritise how the inverse obtains its weights, while retaining the mean mismatch as a separate check.** A reasonable next candidate is a sensor-only stabilised estimate of spectral variance, rather than letting each noisy spectral entry directly determine its own influence. Any smoothing or other estimation rule must be specified and selected using independent sensor-level evidence, not tuned to known wave recovery. A representative independent donor pool is an idealised diagnostic and is not automatically available in use; adopting it as a sea-condition selector would not be justified.

Before selecting a particular weight-estimation implementation, check what independent information the intended data pipeline can actually provide. If using split time blocks, account for temporal correlation and overlapping Welch windows; simply labelling overlapping segments independent would be incorrect. If using frequency smoothing, test preservation of peaks and the induced correlations. Do not introduce a new bandwidth solely to make these wave-energy errors smaller.

**Do not correct only the downward contribution and assume success.** In the long-period case a positive mean mismatch partly cancels the negative same-record weighting contribution. A weighting improvement can therefore expose a positive residual bias. Judge any candidate on signed bias, absolute recovery error and uncertainty coverage across all conditions, with the existing estimator as the comparator. Improvement in bias alone does not establish calibrated 95% intervals.

The diagnosis supports a focused weight-estimation pilot, not immediate merging of M plus an untested uncertainty fix. Keep the original physics, including nonlinear drag. A measurement-aware mean correction may still be justified separately, but this test does not identify drag, filtering or another physical mechanism as the sole cause.

## Reproduction and invariants

Run `WM/diagnose_diagonal_bias.py` with NumPy, SciPy, Matplotlib and the project's existing Capytaine dependencies available; then run `WM/plot_diagonal_bias.py`. It uses the existing ignored data caches. The frozen protocol, input hashes and all 72 result rows are saved alongside this note. The reconstruction of all 1,200 original gains (600 records, two heading grids) is checked against saved point estimates; the extra fixed-prior calculations are diagnostic only.

Tests cover the exact additive identity, full distinct-donor enumeration, unit invariance, zero coupling for constant gains, invalid dimensions and equivalence of the fixed-prior gain to the original gain when supplied the original record scales. All 58 repository tests pass. All 18 report-artifact hashes and five production source files were verified unchanged.

No fresh sea conditions or field data were tested. No claim is made that removing coupling will improve every recovery metric or fix interval undercoverage.
