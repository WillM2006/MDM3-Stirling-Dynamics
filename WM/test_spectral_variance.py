import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scipy.signal import get_window
from spectral_variance import effective_averages, recording_variance, segment_estimates
import invert_waves as iw


class SpectralVarianceTests(unittest.TestCase):
    def test_segment_mean_matches_existing_welch(self):
        rng = np.random.default_rng(291)
        for length, expected_count in [(1800, 7.629139072847682), (1801, 7.629139072847682),
                                       (3600, 16.15527950310559), (7200, 33.20783132530121)]:
            samples = rng.normal(size=(length, 3))
            f, pieces, count = segment_estimates(samples)
            original_f, original = iw.welch_matrix(samples)
            expected = np.stack([iw.data_vector(s, iw.stack_pairs(3)) for s in original])
            np.testing.assert_array_equal(f, original_f)
            np.testing.assert_allclose(pieces.mean(0), expected, atol=2e-14)
            self.assertAlmostEqual(count, expected_count)

    def test_overlap_and_independent_segments(self):
        window = get_window("hann", 400)
        self.assertEqual(effective_averages(window, 400, 4), 4)
        values = np.arange(60.).reshape(4, 5, 3)
        np.testing.assert_allclose(recording_variance(values, 4), values.var(0, ddof=1) / 4)

    def test_white_noise_variance_calibration(self):
        rng = np.random.default_rng(301)
        means, predicted = [], []
        for _ in range(1000):
            _, pieces, count = segment_estimates(rng.normal(size=(1800, 2)))
            means.append(pieces.mean(0)[25])
            predicted.append(recording_variance(pieces, count)[25])
        ratio = np.mean(predicted, axis=0) / np.var(means, axis=0, ddof=1)
        np.testing.assert_allclose(ratio, 1, atol=0.20)

    def test_invalid_inputs(self):
        for samples in (np.ones(20), np.ones((100, 2)), np.full((1800, 2), np.nan)):
            with self.assertRaises(ValueError):
                segment_estimates(samples)
        with self.assertRaises(ValueError):
            recording_variance(np.ones((1, 4, 2)), 1)

    def test_study_rejects_mixed_sea_conditions(self):
        from variance_study import read_groups
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for seed, period in [(1, 8), (2, 6)]:
                record = root / "case" / str(seed)
                record.mkdir(parents=True)
                (record / "pressure.csv").touch()
                (record / "params.csv").write_text(f"seed,Tp\n{seed},{period}\n")
            with patch.object(iw, "read_channels", return_value=(None, np.ones((1800, 2)))):
                with self.assertRaisesRegex(ValueError, "different simulation conditions"):
                    read_groups(root)

    def test_paired_convergence_comparison_preserves_units(self):
        from analyse_variance_convergence import bootstrap_moments, interval
        rng = np.random.default_rng(618)
        values = rng.normal(size=(96, 5, 3)) + 4
        weights = rng.multinomial(96, np.full(96, 1/96), size=200) / 96
        mu, var = bootstrap_moments(values, weights)
        scaled_mu, scaled_var = bootstrap_moments(values * 3, weights)
        np.testing.assert_allclose(interval(scaled_mu / mu), 3, atol=1e-13)
        np.testing.assert_allclose(interval(scaled_var / var), 9, atol=1e-12)
        np.testing.assert_allclose(interval(var / var), 1)

    def test_saved_carrier_matches_direct_harmonic_sums(self):
        from variance_convergence import save_carrier, harmonic_projection
        times = np.arange(1051) * .05
        frequencies = np.array([.1, .34, .82, 1.7])
        coefficients = np.random.default_rng(19).normal(size=(4, 6)) * (1 + 2j)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "waves.npy"
            save_carrier(path, frequencies, times, chunk=7)
            cached = np.load(path, mmap_mode="r")
            direct = np.exp(-1j * times[:, None] * frequencies)
            np.testing.assert_array_equal(cached, direct)
            np.testing.assert_allclose(cached @ coefficients, direct @ coefficients)
            np.testing.assert_allclose(cached[::2] @ coefficients, direct[::2] @ coefficients)
            np.testing.assert_allclose(harmonic_projection(cached, coefficients), np.real(direct @ coefficients))

    def test_resolution_comparison_separates_agreement_and_difference(self):
        from analyse_variance_resolution import comparison
        reference = np.ones((1, 3))
        values = np.array([[1., 1.3, 1.05]])
        draws = np.stack([values - .02, values + .02])
        draws[:, 0, 2] = [.8, 1.3]
        result, ratio, low, high = comparison(values, reference, draws,
                                              np.ones_like(draws), .1)
        self.assertAlmostEqual(result["fraction_intervals_within_tolerance"], 1/3)
        self.assertAlmostEqual(result["fraction_intervals_outside_tolerance"], 1/3)
        self.assertAlmostEqual(result["fraction_point_estimates_within_tolerance"], 2/3)

    def test_resolution_analysis_recovers_known_mean_and_variance_scaling(self):
        import csv
        import json
        from analyse_variance_resolution import analyse
        rng = np.random.default_rng(75)
        values = rng.uniform(1, 4, size=(8, 5, 81))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            protocol = dict(seeds=8, seed_start=0, bootstrap=20, tolerance=.1,
                            grids=[800, 1600, 3200], usable_lengths=[900, 1800, 3600])
            (root / "protocol.json").write_text(json.dumps(protocol))
            for grid, scale in [(800, 1.2), (1600, 1.1), (3200, 1)]:
                for length in protocol["usable_lengths"]:
                    directory = root / f"n{grid}_t{length}"
                    directory.mkdir()
                    for seed in range(8):
                        np.savez(directory / f"seed-{seed}.npz", f=[.05, .1, .16, .26, .31],
                                 observed=values[seed]*scale)
            with patch("analyse_variance_resolution.plot"), patch("builtins.print"):
                analyse(root, root)
            with (root / "resolution_summary.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 135)
            for row in rows:
                if row["coarse_grid"] == "800" and row["fine_grid"] == "3200":
                    expected = 1.2 if row["metric"] == "mean" else 1.44
                    self.assertAlmostEqual(float(row["median_ratio"]), expected)


if __name__ == "__main__":
    unittest.main()
