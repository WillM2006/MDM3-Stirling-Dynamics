# Does the recording contain enough independent information for split-block weights?

This is a sensor-only feasibility check. No inverse recovery is run, no M or prior is changed, and main, the report and Drive data remain untouched.

## Recording geometry

The current native simulation lasts 1,200 seconds, of which 300 are discarded. The inverse receives 900 seconds at 2 Hz. Welch uses 200-second Hann segments with 50% overlap.

| Layout | Complete Welch segments per spectrum | Approximate independent averages |
|---|---:|---:|
| Full 900 seconds | 8 | 7.629 |
| Each 450-second half | 3 | 2.893 |
| Each 400-second block, separated by 100 seconds | 3 | 2.893 |

The effective averages in this table follow the Hann overlap calculation under the usual locally flat Gaussian spectral approximation. They are not fitted M values and are not replacements for M = 8.785. The existing plug-in variance parameter and the number of independent spectral averages have different meanings.

With the current Welch implementation, the first half uses segments starting at 0, 100 and 200 seconds. The second uses starts at 450, 550 and 650 seconds. Only 400 seconds of each half enter complete segments, leaving an effective 50-second separation between their contributing sample supports. The guarded layout instead uses starts at 0, 100, 200 and 500, 600, 700 seconds, giving a 100-second gap and retaining the same three segments per block. Frequency resolution is held at 0.005 Hz.

Thus the split retains six complete segments across the two spectra rather than eight. A simple one-way procedure would fit only one three-segment block using weights from the other. A reciprocal procedure could use both, but is a different estimator whose bias and variance have not been tested here.

## What was measured

The 600 existing seeded recordings (200 each for baseline, long-period and weak waves, seeds 250xxx) are regenerated with the unchanged nonlinear generator. Their whole-record spectral matrices must match the saved matrices before any block result is accepted. This restores information not retained in the original spectral-only caches. Every matched result is retained; the analysis uses only sensor outputs, with no inverse fit or wave-truth scoring.

For each channel, 400 independent earlier sensor recordings (230xxx and 240xxx) select the frequency of its mean auto-spectral peak. The test probes are those nine peak-frequency auto powers and nine integrated auto-spectral band powers. Peak frequencies are fixed before examining the test block results.

We assess:

- Cross-block correlation and relative mean difference for all 18 probes, separately for ordinary halves and guarded blocks.
- Empirical effective averages, mean(auto)^2 / var(auto), separately for the full record and each block.
- The coefficient of variation of auto power squared. Auto power squared is the numerator of the auto-spectral variance weight: its relative variability directly shows how noisy that part of a weight estimate is. Multiplication by a fixed 1/M does not reduce this coefficient of variation.

All intervals use 1,000 whole-record bootstrap draws, preserving paired blocks within a recording. They are pointwise, conditional on independently chosen peak frequencies. A correlation interval entirely within −0.2 to +0.2 is evidence of practically weak linear dependence for that probe; it is not proof of independence, and 18 probes are not 18 independent experiments. Band-integrated quantities cannot substitute for per-frequency information in the existing inverse.

The correlation margin is an explicit screening choice, not a universal threshold for successful recovery. No universal variance-proxy precision threshold is asserted. The decision weighs loss of per-frequency information against the intended reduction in same-record dependence.

## Results and recommendation

**Do not move directly to recovery with simple half-record weights at the current 900-second usable duration.** The data support a real precision problem. Approximate independence is plausible for many probes but is not established for all of them; adding a gap does not solve the precision loss. This is a recommendation against the straightforward proposal, not a proof that every cross-fitted or regularised estimator must fail.

| Condition | Median full-record variance-proxy CV | Median half-record variance-proxy CV | Half probes with CV lower 95% bound above 100% |
|---|---:|---:|---:|
| baseline | 79.6% | 134.0% | 15/18 |
| long_period | 86.4% | 131.0% | 14/18 |
| weak_waves | 76.5% | 126.4% | 12/18 |

Full-record medians describe nine channel-peak probes; half-record medians describe both blocks at those nine peaks. These are descriptive summaries across correlated probes, not independent replications. A CV of 130% means that the estimated variance numerator has a standard deviation 1.3 times its mean across recordings. It does **not** mean wave-height error is 130%. The full-record variance estimates are already noisy; splitting makes their per-frequency noise substantially larger.

For a concrete uncertainty comparison, baseline pressure channel P1 has variance-proxy CV 75.9% [67.7, 83.2] for the full record, 129.2% [104.4, 148.1] in half A and 128.9% [108.6, 146.3] in half B. Across all conditions and channel peaks, half-block point estimates range from 101.9% to 219.5%; 41 of the 54 individual pointwise intervals have lower bounds above 100%. This count is descriptive, not a multiple-testing-corrected claim about 54 independent probes.

The empirical per-frequency effective averages range from 5.66–8.48 for full records and 1.88–3.55 for ordinary halves. These peak-specific diagnostics are not a recalibration or contradiction of the separately defined whole-band median M calibration. Band-integrated powers contain considerably more information, but this does not supply the per-frequency weights required by the current inverse.

| Condition | Ordinary halves: correlation interval entirely within ±0.2 | Guarded blocks: interval entirely within ±0.2 |
|---|---:|---:|
| baseline | 6/18 | 8/18 |
| long_period | 13/18 | 11/18 |
| weak_waves | 15/18 | 16/18 |

All ordinary-half correlation point estimates lie between −0.156 and +0.129. The intervals are not narrow enough to establish the chosen weak-correlation criterion for every probe. The gap does not consistently improve the counts across conditions, and peak variance-proxy variability remains high (up to 228% in the guarded long-period block). The first guarded block uses the same complete segments as half A, so those results are exactly identical by design and must not be counted as extra evidence.

All block mean-difference intervals include zero. That is no detected mean shift in these stationary simulations; it is not proof of stationarity or a guarantee for a real recording. The observed mean differences can still be several percent and their uncertainty matters.

**Practical decision:** retain the current estimator and the bounded sensor-level M result. Do not treat splitting as a ready-made fix, and do not spend the next step on a recovery run that ignores this information loss. If independent weight estimation remains a priority, the options are longer recordings or an explicitly regularised estimator of block variances. Both are additional design work; neither has been validated here. Longer recordings offer a cleaner information-based direction than another arbitrary weight tweak, but only if the data-generation and eventual stationarity requirements are acceptable. No longer-data experiment is launched in this study.

## Implementation boundary

A shorter block has a different sampling variance from the full recording. Reusing M = 8.785 unchanged as the shorter block's variance denominator would not be justified. In this diagnostic M remains untouched: we measure information and dependence rather than fitting a replacement value or attaching new uncertainty intervals.

Independent blocks do not by themselves solve noisy inverse weights. Even a correctly scaled variance estimate can be highly variable, and inverse-variance weighting can amplify unusually small estimates. Conversely, weak observed correlation is only one requirement; sensor spectra may still change between blocks, and nonlinear dependence may remain.

If retaining approximately eight Welch segments in **each** block is necessary, each block needs at least 900 usable seconds. Two such blocks need 1,800 usable seconds, or 1,900 with a 100-second gap, plus the existing 300-second startup discard. That is about 35–37 minutes total instead of the current 20 minutes. This is a geometry calculation, not a recommendation to regenerate data or a validated solution; longer real recordings also raise stationarity questions.

## Evidence and reproduction

`check_block_information.py` regenerates and verifies the records, using cached hydrodynamics and the existing accelerated native simulation. `analyse_block_information.py` computes the independent-reference peak selection, paired dependence and precision diagnostics. Protocols are written before their calculations; the manifest records input hashes. Raw block spectra remain in the ignored analysis cache.

The analysis does not check every cross-spectral feature, nonlinear independence, recovery, or real-vessel stationarity. A favourable sensor result would support a bounded recovery experiment, not establish that split weighting works.

All 63 repository tests pass, including segment counts, the Hann overlap formula, scale-invariant precision summaries and paired-correlation checks. All 600 regenerated whole-record spectra matched their saved counterparts. Five production sources and all 18 report artifacts were verified unchanged.
