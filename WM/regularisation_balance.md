# Algebraic regularisation-balance diagnostic

This test fixes tau = 5 sqrt(2.5 / 8.785033150237455), approximately 2.667, before examining any recovery results. There is no parameter search, wave-truth selection, change to M, or change to the report or production implementation.

For Sigma = C/M, dividing the quadratic posterior objective by M gives a data-fit term with C inverse and a prior penalty proportional to 1/(M tau^2 s^2). Holding M tau^2 fixed therefore preserves the point estimate when C and the original data-derived scale s remain unchanged. Posterior covariance does not remain unchanged: it scales by 2.5/M. Recovering the old point estimate is consequently not evidence that its uncertainty intervals are recovered or calibrated.

The algebraic control applies the old equal-real/imaginary variance formula at the new M and balance-preserving tau. Its complete directional point spectrum is compared with the saved original M=2.5, tau=5 spectrum for every record and heading grid. A relative discrepancy above 2e-6 fails the run.

The actual diagnostic then uses the corrected marginal formulas at the new M and balance-preserving tau. These formulas change relative entry weights as well as their overall scale, so there is no exact equivalence theorem for this comparison. The original recipient-record prior scales are retained in every solve.

The data are the previously used 600 recordings (230xxx), 200 each for baseline, long-period and weak waves. Twelve headings are primary, with 36 as sensitivity. Comparisons are paired at the record level and use 1,000 bootstrap draws with pointwise 95% intervals. This is a retrospective mechanism diagnostic, not new validation or evidence that this tau is optimal. The same existing nonlinear simulator and forward-model assumptions remain.

All finite recovery outcomes are retained; any nonfinite or nonpositive-energy recovery stops the diagnostic instead of being dropped. Wave truth enters scoring only. Existing point estimates are loaded unchanged from the earlier experiment; newly computed spectra belong solely to this diagnostic.

The protocol, per-record metrics, paired comparisons, numerical equivalence checks and input hashes are stored in regularisation_balance_* files. check_regularisation_balance.py reproduces the analysis using existing caches. No new simulation is required.

## Results

All 1,200 complete-spectrum equivalence controls passed (600 records, two heading grids). Maximum relative spectrum discrepancy was 3.84e-8. This verifies that changing M alone can be exactly compensated in the point estimate by the algebraic prior adjustment. It does not validate posterior uncertainty.

| Primary 12-heading mean absolute height error | Original | Corrected, tau=5 | Corrected, balanced tau |
|---|---:|---:|---:|
| Baseline | 5.106% | 5.259% | 5.269% |
| Long period | 5.373% | 5.306% | 5.302% |
| Weak waves | 4.863% | 5.016% | 5.026% |

Height changes from balancing are +0.0103 pp [0.0062, 0.0145], −0.0033 pp [−0.0123, 0.0056], and +0.0093 pp [0.0055, 0.0127]. They are negligible in practical scale and do not rescue height recovery. Against the original estimator, balanced-candidate height errors are higher by 0.163 pp [0.073, 0.250] and 0.162 pp [0.057, 0.248] in baseline and weak waves. The long-period difference of −0.070 pp [−0.175, 0.037] is inconclusive.

| Primary full directional-spectrum NRMSE (lower is better) | Original | Corrected, tau=5 | Corrected, balanced tau |
|---|---:|---:|---:|
| Baseline | 0.743 | 1.230 | 0.963 |
| Long period | 2.453 | 5.927 | 3.393 |
| Weak waves | 0.738 | 1.149 | 0.919 |

The directional-spectrum reductions against the candidate at tau=5 have paired 95% intervals entirely below zero in every condition and at both heading grids. At 12 headings, the balanced candidate remains worse than the original in all three conditions, also with paired intervals excluding zero. At 36 headings it remains worse in baseline and long-period cases; weak-wave difference from the original is inconclusive. The directional-distribution metric is distinct from mean-direction error, which changes very little at 12 headings.

Primary nominal 95% energy coverage remains 16.5%, 25% and 26%, unchanged from the candidate at tau=5. This is the diagonal Gaussian posterior, not the separate propagated-correlation interval experiment. These coverage numbers do not support calibrated uncertainty.

Conclusion: weaker effective regularisation explains a substantial part of the candidate's directional-spectrum deterioration, but does not explain or cure its lack of a useful height-recovery improvement. The corrected relative weighting still matters after global balance is restored. This diagnostic neither establishes an optimal prior nor justifies adopting the candidate. No further tau search is performed. Report and production sources remain unchanged.
