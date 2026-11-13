import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sea import Sea, uniform_wave_sum
from submarine import Submarine
import sensors
import create_data
import batch_invert


class DenseGenerationTests(unittest.TestCase):
    def test_wave_sum_matches_direct_at_nonzero_start(self):
        rng = np.random.default_rng(7)
        for count in (40, 6400):
            omega = np.linspace(.1, 2, count)
            times = 13 + np.arange(31)*.05
            c = rng.normal(size=(count, 3)) + 1j*rng.normal(size=(count, 3))
            np.testing.assert_allclose(uniform_wave_sum(omega, times, c),
                                       np.real(np.exp(-1j*np.outer(times, omega)) @ c), atol=2e-9)
        with self.assertRaises(ValueError):
            uniform_wave_sum(np.array([.1, .2, .4]), times, c[:3])

    def test_sweep_energy_and_pressure(self):
        for params in create_data.make_combos():
            depth = params['depth']
            sea_params = {k: v for k, v in params.items() if k != 'depth'}
            old = Sea(**sea_params)
            dense = Sea(n_freq=6400, uniform_frequencies=True, **sea_params)
            self.assertAlmostEqual(dense.Hs_actual, old.Hs_actual, places=12)
            dense.w = dense.omega
            dense.cos_theta, dense.sin_theta = np.cos(dense.theta), np.sin(dense.theta)
            times = np.arange(6)*.1 + 301
            x, y = np.full(6, 40.), np.full(6, 2.)
            fast = sensors.incident_pressure(dense, dense.wave_amplitudes, x, y, -depth, times)
            dense.uniform_frequencies = False
            direct = sensors.incident_pressure(dense, dense.wave_amplitudes, x, y, -depth, times)
            np.testing.assert_allclose(fast, direct, rtol=2e-10, atol=1e-7)

    def test_bem_grid_is_independent(self):
        sub = Submarine(n_freq=6400, n_bem=40, uniform_frequencies=True)
        self.assertEqual(len(sub.f), 6400)
        self.assertEqual(len(sub.bem_omegas), 40)
        self.assertEqual(sub.params['n_bem'], 40)
        self.assertEqual(sub.params['uniform_frequencies'], 1)

    def test_accelerated_trajectory(self):
        sub = Submarine.__new__(Submarine)
        sub.uniform_frequencies, sub.forward_speed = True, 0
        sub.w = np.linspace(.1, 2, 64)
        sub.k = sub.w**2 / 9.81
        sub.cos_theta, sub.sin_theta = np.array([1.]), np.array([0.])
        sub.F_hat = np.random.default_rng(3).normal(size=(64, 1, 6)).astype(complex)
        sub.K = np.zeros((3, 6, 6))
        sub.M_inv, sub.B_visc = np.eye(6), np.eye(6)*.1
        sub.A_sys = np.block([[np.zeros((6, 6)), np.eye(6)], [-np.eye(6), np.zeros((6, 6))]])
        sub.drag = lambda v: -.03 * v * abs(v)
        _, fast = sub.simulate(4)
        sub.uniform_frequencies = False
        _, direct = sub.simulate(4)
        np.testing.assert_allclose(fast, direct, rtol=1e-10, atol=1e-12)

    def test_resume_and_sweep(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'data'
            runs = create_data.run_list(str(root))
            self.assertEqual(len(runs), 500)
            self.assertTrue(all(p['n_freq']==6400 and p['n_bem']==40 for _, p in runs))
            create_data.check_generation(root, runs)
            create_data.check_generation(root, runs)
            with patch.object(create_data, 'T_SIM', 600):
                with self.assertRaises(SystemExit):
                    create_data.check_generation(root, runs)
            legacy = Path(tmp)/'old'
            legacy.mkdir()
            (legacy/'done.flag').touch()
            with self.assertRaises(SystemExit):
                create_data.check_generation(legacy, runs)

    def test_bem_cache_reuses_wave_grids_but_not_depths(self):
        from types import SimpleNamespace
        def solve(sub):
            for name in create_data.BEM_ATTRS:
                setattr(sub, name, np.array([sub.params['depth']]))
        with patch.object(create_data, '_bem_cache', {}), \
             patch.object(create_data, '_solve_hydrodynamics', side_effect=solve) as build:
            for count, depth in ((40, 15), (6400, 15), (6400, 25)):
                sub = SimpleNamespace(params=dict(n_freq=count, n_bem=40, n_dir=36,
                                                  depth=depth, forward_speed=0, omega_range=(.1, 2)))
                create_data.cached_solve_hydrodynamics(sub)
                self.assertEqual(sub.A[0], depth)
            self.assertEqual(build.call_count, 2)

    def test_inverse_uses_bem_count_and_reads_legacy(self):
        from types import SimpleNamespace
        for extra, expected in (({}, 40), ({'n_freq':6400, 'n_bem':40}, 40)):
            params = dict(depth=15, n_freq=40, n_dir=36, omega_min=.1, omega_max=2)
            params.update(extra)
            with patch.object(batch_invert, 'read_params', return_value=params), \
                 patch.object(batch_invert, 'rao_for_depth', side_effect=RuntimeError('stop')) as build:
                with self.assertRaises(batch_invert.RaoError):
                    batch_invert.process_run('unused', SimpleNamespace(model='capytaine'), {})
                self.assertEqual(build.call_args.args[1], expected)


if __name__ == '__main__':
    unittest.main()
