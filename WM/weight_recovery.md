# Sensor-only weight estimation: recovery test

This study tests variance smoothing as a way to reduce recording-dependent inverse weighting, with M = 8.785033150237455, tau = 5 and the original recording-specific prior scales fixed. It changes weights only: the measured spectral data being fitted, forward responses, nonlinear simulation physics and wave truth are unchanged. Main, the report and Drive data are not edited.

## Selection before recovery

Two stages are retained, including the unsuccessful first stage.

1. **Raw variance smoothing.** Candidate widths were 1, 3, 5, 9 and 17 frequency bins, with reflected boundaries. Width 1 means no smoothing. Using only the 200 baseline calibration records (207000–207199), two alternating-seed folds compared each record's estimated variances with empirical spectral scatter from the other fold. The loss was mean squared relative variance error over frequencies and real spectral entries. Width 1 won: losses were 0.500, 70.597, 218070.373, 3.194e11 and 1.298e18. Raw averaging mixes entries with very different variance levels across frequency. It was rejected by the sensor criterion, not by wave-recovery tuning. Its selected width-1 recovery results are retained as a numerical no-change control.

2. **Calibration-profile smoothing.** This separate refinement divides each recording's variance estimates by a fixed baseline sensor-variance profile, smooths these dimensionless ratios, and restores the profile. The first 80 baseline calibration records estimate that profile. The remaining 120 records, disjoint from the profile records, select the width using two alternating-seed folds of 60 records each and the same independent empirical-scatter loss. The profile is not refitted afterwards. Losses were 0.526, 0.306, 0.219, 0.152 and 0.113; width 17 was selected and frozen before recovery evaluation.

The selected width is the widest of the five tested; it is not a demonstrated optimum. The frequency spacing is 0.005 Hz; a centred 17-bin window has offsets of up to 0.040 Hz. All candidate choices and failed stages are retained. There is no further width search based on recovery outcomes.

The profile and bandwidth use only sensor-calibration spectra, never the true sea spectrum, inverse outputs or the test-condition label. The same fixed baseline profile and selected width are used in all three test conditions. The method does not assume that a matching sea-condition pool can be selected in use. It does, however, rely on the representativeness of the baseline calibration profile, which is an explicit transfer risk.

M is unchanged, and the smoothed variance is not divided by the number of neighbouring bins: this estimates the variance of the original observation, not the variance of an averaged replacement observation. Adjacent Welch frequencies remain correlated. Smoothing does not make weights independent of fitted data.

## Evaluation and decision rule

The comparison uses the existing 600 recordings (250xxx): 200 each for baseline, long-period and weak-wave conditions. Twelve headings are primary; 36 provide sensitivity. Each original inverse is checked against its saved energy estimate. The candidate uses an SVD solve with exactly the original record's prior scales; the unsmoothed SVD calculation is checked against the original inverse.

The primary criterion, frozen before outcomes, is a decrease in mean absolute retained-band significant-wave-height error, with the paired 95% interval below zero in **all three** conditions. Signed height and energy bias, spectrum error, direction error, negative power and uncertainty coverage are guardrails. A reduction in bias alone is not sufficient.

Intervals use 1,000 paired whole-record bootstrap draws, conditional on the chosen profile and width. Coverage uses exact binomial intervals. These intervals do not account for reselecting the bandwidth or refitting the profile. The sensor-selection loss itself is a finite-reference development score, not an uncertainty-certified out-of-sample variance guarantee.

This is a retrospective pilot using previously examined simulations, not untouched final validation. No new shared forward/inverse physical assumptions or truth-based correction are introduced. Existing simulator/model similarities remain. A successful pilot would still need fresh validation, and does not automatically justify applying M = 8.8 to the original 40-component Drive data.

## Results: reject this candidate

The selected candidate fails the predeclared all-condition recovery criterion. It is not recommended for merging into main.

| Mean absolute band-height error, 12 headings | Original | Candidate | Paired change, 95% interval |
|---|---:|---:|---:|
| baseline | 5.553% | 4.915% | -0.638 [-0.851, -0.430] pp |
| long_period | 6.716% | 15.414% | +8.697 [+7.415, +9.823] pp |
| weak_waves | 5.341% | 4.819% | -0.522 [-0.706, -0.322] pp |

Baseline and weak-wave height-error improvements are supported by their paired intervals, but long-period error increases by 8.70 percentage points [7.42, 9.82]. Its mean energy bias changes from +0.27% to −26.80%. Spectrum error increases from 0.355 to 0.491 and direction error from 1.147 to 1.467 degrees. Weak-wave direction error also increases: 1.953 to 2.108 degrees, change +0.154 [0.070, 0.237]. Therefore it fails both the primary criterion and recovery guardrails. No nonpositive-energy failures were discarded; none occurred.

At 36 headings the pattern persists: absolute height errors change from 5.299% to 4.821%, 6.696% to 11.523%, and 5.037% to 4.656%. Long-period energy bias becomes −18.88%. Weak-wave mean direction error increases from 13.414 to 14.195 degrees. The choice of heading grid is not a remedy for this weighting rule.

The width-1 raw-smoothing control leaves recovery unchanged apart from numerical solver precision. This is a failed candidate-selection stage, not a successful improvement.

### Interpretation and next decision

The result supports the earlier warning that identifying a downward same-record weighting bias does not guarantee that a practical replacement improves recovery. Baseline-profile smoothing reduces baseline/weak-wave energy underestimation but transfers poorly to the long-period spectral shape. This is evidence against this profile and smoothing rule, not against all possible weight estimation methods or the sensor-level calibration of M.

A post-test weight-scale diagnostic confirms severe long-period distortion: 28.3% of variance entries exceed ten times their original value, versus 0.28% for baseline and 3.53% for weak waves. The long-period 95th percentile ratio is about 215,335. Its mean posterior energy standard deviation increases from 0.00314 to 0.28095 (about 89-fold). These ratios compare the candidate with the original variance estimates, not with known true variances; they show how dramatically the weighting changes. They explain why nominal 100% coverage is not a useful success criterion here. `diagnose_weight_scales.py` reproduces this post-test check; it does not tune a rescue rule.

Do not widen the bandwidth search, select a different width for the long-period test, clip large weights based on its recovery outcomes, or tune M to rescue this candidate. Those would convert the test cases into recovery-tuning data. Retain the failed result and leave the existing production estimator untouched.

If continuing, the most distinct next route is to test weights estimated from separate, non-overlapping time blocks of the same recording, with an explicit check of remaining temporal dependence. That avoids assuming a transferable baseline spectral profile, but costs information and is not proven to work here. Overlapping Welch segments must not simply be treated as independent. First establish whether the available duration supports enough independent information and a stable sensor-variance estimate; only then specify a recovery test. It may be more practical for the report to retain the bounded sensor-level M result and explain the inverse-uncertainty limitation than adopt an unvalidated replacement.

The saved coverage numbers are for each fit's diagonal Gaussian posterior, not the separate pooled-correlation uncertainty candidate from the preceding investigations. Long-period coverage reaches 100% despite much worse recovery; that is not success or evidence of calibration. These are different interval constructions, so they must not be compared directly with the earlier 84–90% propagated-covariance coverage.

## Evidence and reproduction

`check_smoothed_weights.py` and `check_profile_weights.py` record their protocols before evaluation and freeze sensor-only selection before wave-truth scoring. They reuse existing sensor caches and cached hydrodynamics. `plot_weight_recovery.py` produces the recovery figure. Each stage saves selection, protocol, manifest, record-level metrics, summaries and paired comparisons. The profile is retained in the ignored analysis cache.

The tests check positivity and shape validation, constant-scale preservation, channel separation, preservation of a sharp calibration profile, width-1 identity and solver equivalence to the original inverse. All 61 repository tests pass. Five production source files and all 18 report artifacts are unchanged.
