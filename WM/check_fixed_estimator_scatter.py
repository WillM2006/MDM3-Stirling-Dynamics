"""Retrospective covariance propagation; never refit the saved point estimates."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.stats import binomtest
import invert_waves as iw
import correlated_spectral as cs
from check_inverse_recovery import candidate_variances, operator_responses
from check_pooled_covariance import CALIBRATION, GROUPS
from analyse_variance_convergence import bootstrap_moments
from variance_convergence import ROOT, write_csv


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def energy_gain(f, reference, responses, m, tau):
    """Freeze the existing diagonal inverse weights and prior at sensor-only means."""
    with patch.object(iw, 'noise_variances', candidate_variances):
        x, cov, _ = iw.invert(f, reference, responses, tau, m)
    pairs = iw.stack_pairs(reference.shape[-1])
    gain = np.stack([(f[1]-f[0])*(np.ones(len(h))@c@iw.operator_matrix(h, pairs).T)
                     /candidate_variances(s, pairs, m) for s, h, c in zip(reference, responses, cov)])
    data = np.stack([iw.data_vector(s, pairs) for s in reference])
    np.testing.assert_allclose(np.sum(gain*data), x.sum()*(f[1]-f[0]), rtol=2e-7, atol=1e-12)
    return gain


def scalar_bootstrap(values, draws):
    return values[draws].var(axis=1, ddof=1)


def summarise(predicted, actual, frozen, truth, draws, prediction_draws=None):
    """Ratios and uncentred coverage; truth only scores already-fixed predictions."""
    predicted = np.broadcast_to(predicted, actual.shape)
    if (len(actual) < 3 or not all(np.isfinite(x).all() for x in [predicted, actual, frozen, truth])
            or np.any(predicted <= 0) or np.any(truth <= 0)
            or actual.var() <= 0 or frozen.var() <= 0):
        raise ValueError('nonfinite, nonpositive or degenerate variance comparison')
    pb = predicted[draws].mean(1) if prediction_draws is None else prediction_draws
    out = dict(predicted_variance=float(predicted.mean()))
    for name, values in [('actual', actual), ('frozen', frozen)]:
        vb = scalar_bootstrap(values, draws)
        lo, hi = np.quantile(pb/vb, [.025, .975])
        out.update({name+'_variance': float(values.var(ddof=1)),
                    name+'_ratio': float(predicted.mean()/values.var(ddof=1)),
                    name+'_ratio_ci_low': float(lo), name+'_ratio_ci_high': float(hi)})
    for nominal, z in [('68', 1.), ('95', 1.959963984540054)]:
        hits = int(np.sum(abs(actual-truth) <= z*np.sqrt(predicted)))
        lo, hi = binomtest(hits, len(actual)).proportion_ci(.95, method='exact')
        out.update({'coverage'+nominal: hits/len(actual), 'coverage'+nominal+'_ci_low': lo,
                    'coverage'+nominal+'_ci_high': hi})
    out['mean_energy_interval_width95_pct'] = float(np.mean(2*1.959963984540054*np.sqrt(predicted)/truth)*100)
    lower = 4*np.sqrt(np.maximum(actual-1.959963984540054*np.sqrt(predicted), 0))
    upper = 4*np.sqrt(np.maximum(actual+1.959963984540054*np.sqrt(predicted), 0))
    out['mean_Hs_interval_width95_pct'] = float(np.mean((upper-lower)/(4*np.sqrt(truth)))*100)
    return out


def run(training, data, output, cache):
    parent = json.loads((data/'protocol.json').read_text())
    for key, name in [('inverse_sha256', 'invert_waves.py'), ('joint_sha256', 'correlated_spectral.py'),
                      ('pool_sha256', 'pooled_spectral.py')]:
        if digest(ROOT/name) != parent[key]:
            raise ValueError('previously frozen covariance or inverse source changed')
    m, tau = parent['M'], parent['tau']
    assert m == 8.785033150237455 and tau == 5
    source_scores = ROOT/'pooled_covariance_records.csv'
    scores = list(csv.DictReader(source_scores.open()))
    scores_hash = digest(source_scores)
    manifest = json.loads((data/'pool_manifest.json').read_text())
    for name, value in manifest['pools'].items():
        if digest(data/name) != value:
            raise ValueError('frozen calibration covariance changed')
    protocol = dict(M=m, tau=tau, primary_headings=12, sensitivity_headings=36, records_per_group=200,
        groups=GROUPS, calibration=CALIBRATION, primary_gain_reference='baseline', bootstrap=1000,
        study='retrospective reuse of 600 previously examined test records; no fresh validation',
        reference='three independent calibration means define three frozen gains; every gain and covariance pool tested on every condition; no best-pool selection',
        invariant='saved self_diagonal estimates are read, never recomputed, refitted or bias-corrected; M, tau and response convention unchanged',
        primary='predicted band-energy variance / observed variance of the actual saved point estimator; fixed-gain surrogate ratio reported separately',
        inference='whole-record paired bootstrap conditional on frozen gains, M and fitted structured pools; empirical reference additionally resamples calibration projections with gains fixed',
        interpretation='95% interval within 0.9--1.1 establishes the existing practical variance criterion; inclusion of one alone is inconclusive; pointwise intervals are not simultaneous',
        coverage='uncentred Gaussian energy intervals attached to unchanged estimates; exact binomial intervals conditional on pools; Hs bounds are monotone transforms, not Gaussian Hs claims',
        source_hashes={name:digest(ROOT/name) for name in ['check_fixed_estimator_scatter.py', 'correlated_spectral.py',
            'pooled_spectral.py', 'invert_waves.py', 'batch_invert.py', 'check_cross_variance.py', 'check_inverse_recovery.py']},
        saved_scores_sha256=scores_hash, pool_hashes=manifest['pools'])
    cache.mkdir(parents=True, exist_ok=True)
    protocol_path = cache/'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError('analysis specification changed; use a new output directory')
    protocol_path.write_text(json.dumps(protocol, indent=2)+'\n')
    inputs = {}

    def load(folder, start):
        observed, spectra = [], []
        first = None
        for seed in range(start, start+200):
            path = folder/f'seed-{seed}.npz'
            inputs[str(path)] = digest(path)
            with np.load(path) as z:
                f = z['f']
                if first is None: first = f.copy()
                np.testing.assert_array_equal(first, f)
                observed.append(z['observed']); spectra.append(z['spectral'])
        return first, np.stack(observed), np.stack(spectra)

    calibration = {}
    for pool, (folder, start) in CALIBRATION.items():
        f, observed, spectra = load(training/'uniform6400'/folder, start)
        with np.load(data/f'pool-{pool}.npz') as z:
            np.testing.assert_array_equal(f, z['f'])
            factors = z['factors']
        calibration[pool] = (observed, spectra.mean(0), factors)
    references = [(h, pool) for h in [12, 36] for pool in CALIBRATION]
    gains = []
    for headings in [12, 36]:
        _, responses = operator_responses(training/'hydrodynamics.npz', f, headings)
        for pool in CALIBRATION:
            gains.append(energy_gain(f, calibration[pool][1], responses, m, tau))
    gains = np.stack(gains)
    rho = cs.welch_frequency_correlation(np.rint(f*iw.SEG).astype(int))
    identity = np.eye(len(f))
    pool_values, pool_boots = {}, {}
    for pi, (pool, (observed, _, factors)) in enumerate(calibration.items()):
        per_frequency = np.einsum('nfc,bfc->nbf', observed, gains)
        projected = per_frequency.sum(2)
        weights = np.random.default_rng(281100+pi).multinomial(200, np.full(200, 1/200), size=1000)/200
        _, within_boot = bootstrap_moments(per_frequency, weights)
        _, full_boot = bootstrap_moments(projected[:, :, None], weights)
        pool_values[pool] = dict(
            pooled_diagonal=np.einsum('bfc,fc->b', gains**2, np.sum(factors**2, axis=2)),
            pooled_within=cs.propagated_variance(gains, factors, identity),
            pooled_joint=cs.propagated_variance(gains, factors, rho),
            empirical_within=per_frequency.var(0, ddof=1).sum(1),
            empirical_full=projected.var(0, ddof=1))
        pool_boots[pool] = dict(empirical_within=within_boot.sum(2), empirical_full=full_boot[:, :, 0])
    rows, approximation = [], []
    for gi, (group, start) in enumerate(GROUPS.items()):
        ff, observed, spectra = load(data/group, start)
        np.testing.assert_array_equal(f, ff)
        projections = np.einsum('nfc,bfc->nb', observed, gains)
        record_hash = hashlib.sha256(spectra.tobytes()).hexdigest()
        adaptive_path = cache/f'{group}-projected.npz'
        if adaptive_path.exists():
            with np.load(adaptive_path) as z:
                if str(z['record_hash']) != record_hash:
                    raise ValueError('cached projections refer to different sensor records')
                np.testing.assert_allclose(z['gains'], gains, rtol=1e-12, atol=0)
                np.testing.assert_allclose(z['projections'], projections, rtol=1e-12, atol=0)
                adaptive = z['adaptive']
        else:
            adaptive = []
            for ni, s in enumerate(spectra):
                factors, _ = cs.covariance_factors(s, m)
                adaptive.append(np.stack([np.einsum('bfc,fc->b', gains**2, np.sum(factors**2, axis=2)),
                    cs.propagated_variance(gains, factors, identity), cs.propagated_variance(gains, factors, rho)]))
                if ni % 50 == 0: print(group, ni, 'sensor covariance projections; no inverse refit', flush=True)
            adaptive = np.stack(adaptive)
            np.savez_compressed(adaptive_path, adaptive=adaptive, gains=gains, projections=projections, record_hash=record_hash)
        assert adaptive.shape == (200, 3, 6)
        draws = np.random.default_rng(282100+gi).integers(0, 200, (1000, 200))
        for ri, (headings, reference) in enumerate(references):
            selected = sorted([r for r in scores if r['group']==group and int(r['headings'])==headings
                               and r['method']=='self_diagonal' and r['pool']=='none'], key=lambda r:int(r['seed']))
            assert [int(r['seed']) for r in selected] == list(range(start, start+200))
            assert all(r['status']=='ok' for r in selected)
            actual, truth, posterior = [np.array([float(r[key]) for r in selected]) for key in ['m0', 'm0_true', 'sd_m0']]
            frozen = projections[:, ri]
            ratio = frozen.var(ddof=1)/actual.var(ddof=1)
            lo, hi = np.quantile(scalar_bootstrap(frozen, draws)/scalar_bootstrap(actual, draws), [.025,.975])
            residual = (frozen-frozen.mean())-(actual-actual.mean())
            approximation.append(dict(group=group, headings=headings, gain_reference=reference,
                frozen_over_actual_variance=ratio, ci95_low=lo, ci95_high=hi,
                correlation=np.corrcoef(frozen, actual)[0,1],
                centred_rms_difference_over_actual_sd=np.sqrt(np.mean(residual**2))/actual.std(ddof=0),
                mean_energy_difference_pct=100*np.mean(frozen-actual)/truth.mean(),
                actual_energy_bias_pct=100*np.mean(actual-truth)/truth.mean()))
            choices = [('none', 'original_posterior', posterior**2, None)]
            choices += [('none', name, adaptive[:, mi, ri], None) for mi, name in enumerate(
                ['record_diagonal', 'record_within', 'record_joint'])]
            for pool in CALIBRATION:
                for method, values in pool_values[pool].items():
                    boot = pool_boots[pool].get(method)
                    choices.append((pool, method, values[ri], None if boot is None else boot[:, ri]))
            for pool, method, predicted, pb in choices:
                rows.append(dict(group=group, headings=headings, gain_reference=reference, covariance_pool=pool,
                    method=method, records=200, **summarise(predicted, actual, frozen, truth, draws, pb)))
        print('Completed fixed-estimator analysis:', group, flush=True)
    assert digest(source_scores) == scores_hash
    assert len(rows) == 342 and len(approximation) == 18
    assert len(set((r['group'],r['headings'],r['gain_reference'],r['covariance_pool'],r['method']) for r in rows)) == 342
    write_csv(output/'fixed_estimator_scatter.csv', rows)
    write_csv(output/'fixed_estimator_approximation.csv', approximation)
    (output/'fixed_estimator_protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    (output/'fixed_estimator_manifest.json').write_text(json.dumps(dict(inputs=inputs,
        saved_estimates_unchanged=digest(source_scores)==scores_hash, comparisons=len(rows), approximations=len(approximation),
        source_sha256=digest(__file__), protocol_sha256=digest(protocol_path)), indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', type=Path, default=Path('variance_output/m_calibration'))
    parser.add_argument('--data', type=Path, default=Path('variance_output/pooled_covariance'))
    parser.add_argument('--output', type=Path, default=ROOT)
    parser.add_argument('--cache', type=Path, default=Path('variance_output/fixed_estimator_scatter'))
    args = parser.parse_args()
    run(args.training, args.data, args.output, args.cache)
