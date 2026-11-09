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
        for length in (1800, 1801):
            samples = rng.normal(size=(length, 3))
            f, pieces, count = segment_estimates(samples)
            original_f, original = iw.welch_matrix(samples)
            expected = np.stack([iw.data_vector(s, iw.stack_pairs(3)) for s in original])
            np.testing.assert_array_equal(f, original_f)
            np.testing.assert_allclose(pieces.mean(0), expected, atol=2e-14)
            self.assertAlmostEqual(count, 7.629139072847682)

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


if __name__ == "__main__":
    unittest.main()
