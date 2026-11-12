# Frozen candidate: adoption remains unapproved

The exact tested candidate is pinned in candidate_freeze.json with source and evidence hashes. It uses M=8.785033150237455, corrected marginal variance formulas, 6400 uniform wave components, the original 1200-second simulation and preprocessing, tau=5, and the original diagonal inverse. The primary inverse uses 12 headings; 36 is a sensitivity check. No covariance pooling, smoothing, split-block weights or mean correction is included.

The candidate has stronger sensor-marginal calibration evidence. This does not establish better recovery or calibrated posterior intervals. The existing inverse_recovery comparison already tests the old weights, calibrated M alone and calibrated M with corrected variances on the same 600 previously fresh recordings. Repeating those identical conditions is not needed to discover whether the candidate is uniformly better: the existing answer is no.

The completed primary comparison (old to candidate):

| Condition | Mean absolute band-height error | Mean absolute direction error | Mean absolute discrete peak-period error |
|---|---:|---:|---:|
| Baseline | 5.106% to 5.259% | 1.526 to 1.544 degrees | 2.875% to 2.889% |
| Long period | 5.373% to 5.306% | 1.106 to 1.045 degrees | 2.562% to 2.542% |
| Weak waves | 4.863% to 5.016% | 2.112 to 2.100 degrees | 3.276% to 3.244% |

Peak-period scoring is added retrospectively from all 3600 saved fits, using the reciprocal of the largest retained-bin summed directional spectrum. The target is the peak of the discrete true retained spectrum, not the nominal continuous sea parameter. All three primary paired 95% bootstrap intervals for the change in peak-period error include zero. No peak-period improvement is established. This is not a fresh independent validation.

Existing evidence also shows worse directional-distribution recovery and severe posterior undercoverage. See inverse_recovery.md and its paired results for the original metrics and confidence intervals. Neither M=2.5 nor this candidate supplies reliable nominal posterior coverage in these tests.

Decision: freeze for reproducibility, but do not merge or regenerate final data on the claim that the complete estimator is better. An adoption decision could accept modest point-estimation trade-offs for a better-founded sensor-variance model, but that is an explicit project judgement rather than an experimentally established overall improvement. If further testing is desired, it should address broader representative sea conditions or an explicitly changed estimator, with criteria chosen before examining new outcomes. Do not tune M using wave-recovery truth.

The report, production sources and Drive data are unchanged. freeze_candidate.py reproduces the source/evidence freeze and the added period comparison without running a simulation or inverse fit. Its manifest hashes every saved fit used. Known simulation assumptions remain; no real-vessel validation is claimed.
