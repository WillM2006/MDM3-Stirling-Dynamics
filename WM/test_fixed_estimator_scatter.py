import unittest
from unittest.mock import patch
import numpy as np
import invert_waves as iw
from check_fixed_estimator_scatter import energy_gain, summarise
from check_inverse_recovery import candidate_variances


class FixedEstimatorTests(unittest.TestCase):
    def test_gain_matches_frozen_gaussian_solve_for_different_data(self):
        rng = np.random.default_rng(17)
        f = np.array([.1, .105])
        h = rng.normal(size=(2, 3, 3))+1j*rng.normal(size=(2, 3, 3))
        s = np.einsum('fjc,fjk->fck', h, h.conj())+np.eye(3)
        gain = energy_gain(f, s, h, 8.785, 5)
        with patch.object(iw, 'noise_variances', candidate_variances):
            _, cov, _ = iw.invert(f, s, h, 5, 8.785)
        pairs = iw.stack_pairs(3)
        data = rng.normal(size=(2, 9))
        explicit = sum((cov[i]@iw.operator_matrix(h[i], pairs).T@
                        (data[i]/candidate_variances(s[i], pairs, 8.785))).sum()*.005 for i in range(2))
        self.assertAlmostEqual(np.sum(gain*data), explicit)

    def test_bad_variance_stays_bad_and_bias_is_not_removed(self):
        actual = np.linspace(2., 3., 200)
        original = actual.copy()
        truth = np.ones(200)
        draws = np.random.default_rng(18).integers(0, 200, (1000, 200))
        result = summarise(.0001, actual, actual, truth, draws)
        self.assertLess(result['actual_ratio'], .002)
        self.assertEqual(result['coverage95'], 0)
        self.assertGreater(result['coverage95_ci_high'], 0)
        np.testing.assert_array_equal(actual, original)
        scaled = summarise(.0001*49, actual*7, actual*7, truth*7, draws)
        self.assertAlmostEqual(result['actual_ratio'], scaled['actual_ratio'])
        self.assertEqual(result['coverage95'], scaled['coverage95'])
        self.assertAlmostEqual(result['mean_Hs_interval_width95_pct'], scaled['mean_Hs_interval_width95_pct'])

    def test_degenerate_data_rejected(self):
        with self.assertRaises(ValueError):
            summarise(1., np.ones(5), np.ones(5), np.ones(5), np.zeros((3,5), int))


if __name__ == '__main__':
    unittest.main()
