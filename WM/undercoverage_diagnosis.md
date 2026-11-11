# Undercoverage diagnosis with the estimator held fixed

The remaining coverage failure is not explained by a single incorrect M. Baseline and weak-wave cases show a combination of negative energy bias and error bars that become smaller when the estimated energy is smaller. The long-period case has little mean bias, but underestimated variance on the test cohort. No proposed correction is adopted here.

M = 8.785033150237455, tau = 5, all saved corrected-diagonal point estimates, simulation physics, production code and report remain unchanged. The quantity assessed is retained-band wave energy m0, not full directional-spectrum coverage. Primary results use 12 headings, the fixed baseline gain and baseline correlation pool from the previous study; 36 headings are retained as sensitivity results.

## What was compared

Each condition has 400 independent earlier reference recordings (230xxx and 240xxx cohorts) and 200 existing test recordings (250xxx). The reference outputs use the same corrected-diagonal estimator. The covariance candidate remains the baseline pooled correlation factor rescaled to each recording's corrected marginal variances, with Hann frequency coupling.

The independent reference records supply a diagnostic mean error and variance multiplier. Neither is selected to optimise test coverage. Known wave truth is used to diagnose bias, not to create a deployable calibration. Twelve factorial comparisons combine removing this reference bias, applying this reference variance multiplier, and using paired, independently reassigned or constant widths. The independently reassigned result averages all 39,800 distinct donor assignments; these are not 39,800 independent records. Constant widths preserve the test cohort's mean predicted variance and are an ensemble diagnostic, not an available single-record procedure.

All diagnostic confidence intervals use 1,000 whole-record bootstrap draws, independently resampling reference and test cohorts and recomputing reference quantities. Original coverage uses exact binomial intervals. Intervals are pointwise and conditional on frozen gains and covariance pool; they do not include refitting M or the pool. The protocol was saved before calculating the diagnostic results. These data have been examined previously, so this is retrospective diagnosis rather than untouched validation.

## Primary results

| Diagnostic coverage of nominal 95% intervals | Baseline | Long period | Weak waves |
|---|---:|---:|---:|
| Original candidate | 84.5% | 90.0% | 83.5% |
| Remove independent-reference bias only | 88.5% | 89.5% | 91.0% |
| Adjust variance scale only | 84.5% | 91.5% | 85.0% |
| Break estimate–width pairing only | 87.6% | 88.1% | 88.8% |
| Constant widths only | 89.0% | 90.5% | 91.5% |
| Reference bias and scale adjustments, constant widths | 93.5% | 92.0% | 96.0% |

The rows are controlled diagnostic comparisons, not candidate production methods. Their improvements interact and must not be added as unique percentages of the cause.

### Bias is material in two conditions

Test energy biases are −6.14% [−7.82, −4.47], +0.27% [−2.11, +2.68] and −6.37% [−7.98, −4.74]. Earlier reference biases are −4.97%, +1.24% and −5.84%. Removing reference bias alone increases baseline coverage by 4.0 percentage points [1.49, 7.50] and weak-wave coverage by 7.5 points [3.99, 12.0]. It does not improve long-period coverage.

This identifies a bias contribution, not its physical cause. It does not distinguish forward-mean discrepancy, data-dependent inverse weights, or other estimator effects. In particular, the earlier drag audit used another gain and cannot be treated as a numerical explanation of this estimator's bias.

### Estimate-dependent error bars matter, but not identically everywhere

Correlations between signed estimation error and predicted standard deviation are 0.946, 0.959 and 0.953. A low estimate tends to carry a smaller error bar, making an underestimate more likely to exclude the truth. Original lower-side misses are 15.0%, 8.5% and 16.5%, versus upper-side misses of 0.5%, 1.5% and 0%.

Breaking only this pairing increases baseline coverage by 3.11 points [0.46, 5.51] and weak-wave coverage by 5.28 points [2.31, 8.16]. It changes long-period coverage by −1.91 points [−5.22, 1.31]. A high correlation alone therefore does not establish that removing it will improve coverage in every condition. After removing bias and adjusting scale, the additional pairing effects have intervals crossing zero in all three conditions.

### One variance multiplier is not an established remedy

Predicted variance divided by actual test scatter is 0.864 [0.703, 1.088], 0.714 [0.596, 0.884] and 0.986 [0.807, 1.218]. The long-period candidate underestimates test scatter. Baseline and weak-wave results remain inconclusive rather than proven correct.

The independently estimated variance multipliers are only 1.012 [0.854, 1.180], 1.062 [0.886, 1.233] and 1.122 [0.983, 1.263]. They do not establish a portable corrective factor. In particular, long-period reference energy SD is 14.73% versus 16.82% in the test cohort; this finite-cohort difference means the earlier reference does not supply the large multiplier one would choose retrospectively from the test result. This is a reason not to tune to the test cohort, not proof that the simulations changed distribution.

### Tail shape alone does not explain the whole deficit

After centring and scaling each test cohort by its own empirical moments, the fraction within 1.96 SD is 93.5%, 94.0% and 95.5%. This uses test moments only as a shape diagnostic. It is not validation of an interval method. Long-period errors show positive skew (0.554 [0.191, 0.918]); 200 records per condition do not settle tail calibration. The combined diagnostic coverages of 93.5%, 92.0%, 96.0% have intervals [89.5, 97.0], [87.5, 95.5], [93.49, 99.0]; inclusion of 95% is not evidence of equivalence.

## Sensitivity and what the evidence supports changing

The 36-heading analysis preserves the broad pattern: weak-wave bias and pairing effects persist, variance-scale-only adjustment is insufficient, and long-period pairing removal does not improve coverage. Baseline bias-only and pairing-only contrasts individually become inconclusive at 36 headings. The combined changes are +6.5, +1.5 and +11.0 percentage points, compared with +9.0, +2.0 and +12.5 at 12 headings. Changing headings also changes the aggregate independent prior, so this is not a pure angular-resolution comparison.

1. **Do not change M to repair these interval failures.** Its sensor-marginal interpretation is separate from inverse bias and interval construction. These results do not invalidate its earlier calibration, and do not justify applying it to the old 40-component Drive data.
2. **Do not adopt another universal variance inflation or claim calibrated 95% inverse intervals.** Correct average scatter is insufficient when bias and record-dependent widths remain. The current research candidate remains inadequate.
3. **The interval construction is the clearest target for future development:** it must account for estimating the scale from the same record and for asymmetry, rather than assuming that a symmetric estimate ±1.96 predicted SD is calibrated. A separately specified resampling or sampling-distribution approach could address this, but has not been shown to work here. Simply borrowing another record's scale is not a deployable solution.
4. **Bias needs a separate explanation before a mean correction.** With the existing point estimator still fixed, the most focused next diagnostic is to separate its response to the ensemble mean sensor spectrum from the effect of record-dependent inverse weights. Use independent sensor cohorts for mean/weight references, and known truth only for scoring. This tests whether a measurement-aware forward-mean change is warranted for this actual estimator. Do not subtract the condition-specific biases above in production or remove nonlinear drag to force agreement.

There is no single verified fix from this test. For the report, the defensible boundary is sensor-level M calibration plus explicit limitations of inverse uncertainty. No report text has been changed. An eventual interval rule requires genuinely fresh validation after its construction is fixed, with larger or more varied cohorts if precise 95% coverage is claimed.

## Reproduction and evidence

Run `python3 WM/diagnose_undercoverage.py`, then `python3 WM/plot_undercoverage.py` from the repository root with the existing ignored data caches. No new simulation or inverse solve is performed. The diagnostic checks frozen source/pool hashes and verifies that saved score files remain unchanged.

- `undercoverage_protocol.json`: frozen design and interpretation boundaries.
- `undercoverage_manifest.json`: input and source hashes.
- `undercoverage_scenarios.csv`: all 72 comparisons, coverage and both tails with intervals.
- `undercoverage_contrasts.csv`: all 48 conditional contrasts with paired bootstrap intervals.
- `undercoverage_moments.csv`: all 66 bias, scatter, dependence and shape summaries.
- `undercoverage_diagnosis.png`: primary comparisons with confidence intervals.
- `test_undercoverage.py`: exact donor enumeration, reference-only adjustments, unit invariance, unchanged inputs and invalid variance checks.

No field-data validation, new physical model, refitted M, or new point estimator is included. The known simulation/model assumptions remain; using truth for diagnosis adds no deployable truth-based correction.
