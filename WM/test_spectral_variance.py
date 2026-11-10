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

    def test_uniform_grid_preserves_energy_directions_and_phases(self):
        from sea import Sea
        from variance_uniform import uniform_grid
        for count in [3200, 6400]:
            sub = Sea(n_freq=count, Hs=1.7, Tp=8, s=10, gamma=3.3, seed=104000)
            energy = np.sum(sub.wave_amplitudes**2) / 2
            phase, directions = sub.phi.copy(), sub.D.copy()
            uniform_grid(sub)
            np.testing.assert_allclose(np.diff(sub.f), np.diff(sub.f)[0], rtol=0, atol=1e-15)
            np.testing.assert_allclose(np.sum(sub.wave_amplitudes**2)/2, energy, rtol=1e-14)
            np.testing.assert_allclose(np.sum(sub.S_f*sub.delta_f), energy, rtol=1e-14)
            np.testing.assert_array_equal(sub.phi, phase)
            np.testing.assert_array_equal(sub.D, directions)
            self.assertTrue(np.isfinite(sub.wave_amplitudes).all())

    def test_uniform_zoom_sums_match_direct_sums_over_full_record(self):
        from variance_uniform import uniform_projection
        frequencies = np.linspace(.1, 2, 6400, endpoint=False) + .95/6400
        times = np.arange(78001) * .05
        rng = np.random.default_rng(891)
        coefficients = rng.normal(size=(6400, 3)) + 1j*rng.normal(size=(6400, 3))
        actual = uniform_projection(frequencies, times, coefficients)
        indices = np.r_[0, 1, 1201, 39000, 78000, rng.integers(0, len(times), 80)]
        expected = np.real(np.exp(-1j*times[indices, None]*frequencies) @ coefficients)
        np.testing.assert_allclose(actual[indices], expected, rtol=2e-9, atol=1e-8)
        frequencies[10] += .001
        with self.assertRaisesRegex(ValueError, "uniform frequencies"):
            uniform_projection(frequencies, times, coefficients)

    def test_force_check_scales_roundoff_but_rejects_material_errors(self):
        from variance_convergence import check_projected_forcing
        coefficients = np.full((100, 2), 1e5, dtype=complex)
        expected = np.array([0., 2e5])
        check_projected_forcing(expected + 1e-5, expected, coefficients)
        with self.assertRaises(AssertionError):
            check_projected_forcing(expected + 1., expected, coefficients)

    def test_hybrid_grid_endpoints_preserve_spectrum_energy_and_phases(self):
        from sea import Sea
        from variance_uniform import uniform_grid
        from variance_choice import hybrid_grid
        original = Sea(n_freq=3200, Hs=1.7, Tp=8, gamma=3.3, s=10, seed=104000)
        expected = Sea(n_freq=3200, Hs=1.7, Tp=8, gamma=3.3, s=10, seed=104000)
        uniform_grid(expected)
        for fraction in [0, .5, 1]:
            sub = Sea(n_freq=3200, Hs=1.7, Tp=8, gamma=3.3, s=10, seed=104000)
            hybrid_grid(sub, fraction)
            np.testing.assert_allclose(np.sum(sub.wave_amplitudes**2), np.sum(original.wave_amplitudes**2), rtol=1e-14)
            np.testing.assert_array_equal(sub.phi, original.phi)
            self.assertTrue(np.all(np.diff(sub.f) > 0))
            if fraction in (0, 1):
                target = original if fraction == 0 else expected
                np.testing.assert_allclose(sub.f, target.f, atol=1e-15)
                np.testing.assert_allclose(sub.wave_amplitudes, target.wave_amplitudes, rtol=2e-11, atol=1e-14)
        with self.assertRaises(ValueError):
            hybrid_grid(sub, 1.1)

    def test_m_calibration_records_use_actual_variance_numerators(self):
        from calibrate_m import spectral_record
        samples = np.random.default_rng(83).normal(size=(1800,9))
        record = spectral_record(samples)
        self.assertEqual(record["observed"].shape,(53,81))
        pairs = iw.stack_pairs(9)
        expected = np.stack([iw.noise_variances(s,pairs,1.) for s in record["spectral"]])
        np.testing.assert_array_equal(record["numerator"],expected)
        np.testing.assert_allclose(record["numerator"]/2.5,
                                   np.stack([iw.noise_variances(s,pairs,2.5) for s in record["spectral"]]))

    def test_corrected_cross_variance_matches_complex_wishart(self):
        from check_cross_variance import variance_terms,corrected_variances
        rng=np.random.default_rng(736)
        for count,coherence,phase in [(2,0.,0.),(8,.95,0.),(8,.95,np.pi/2),(8,.8,.7)]:
            z=(rng.normal(size=(100000,count,2))+1j*rng.normal(size=(100000,count,2)))/np.sqrt(2)
            z[:,:,1]=coherence*np.exp(1j*phase)*z[:,:,0]+np.sqrt(1-coherence**2)*z[:,:,1]
            spectral=np.einsum("rki,rkj->rij",z.conj(),z)/count
            residual,square=variance_terms(spectral)
            predicted=corrected_variances(residual,square,count+1)
            q=coherence*np.exp(1j*phase)
            expected=np.array([1/count,(1+(q*q).real)/(2*count),
                               (1-(q*q).real)/(2*count),1/count])
            observed=np.stack([spectral[:,0,0].real,spectral[:,0,1].real,
                               spectral[:,0,1].imag,spectral[:,1,1].real],axis=1)
            np.testing.assert_allclose(predicted.mean(0)/expected,1,atol=.025)
            np.testing.assert_allclose(observed.var(0,ddof=1)/expected,1,atol=.025)
            self.assertTrue(np.all(predicted>=0))
            scale=np.array([1e-3,1e3])
            rr,ss=variance_terms(spectral*scale[:,None]*scale[None,:])
            np.testing.assert_allclose(corrected_variances(rr,ss,count+1),
                                       predicted*np.array([1e-12,1,1,1e12]),rtol=1e-10)

    def test_corrected_cross_variance_rejects_invalid_input(self):
        from check_cross_variance import variance_terms,corrected_variances
        for m in [1,2,np.nan]:
            with self.assertRaises(ValueError):
                corrected_variances(np.ones((3,4)),np.ones((3,4)),m)
        for matrix in [np.array([[1,2],[2,1]]),np.array([[1,1j],[1j,1]]),np.diag([0.,1.])]:
            with self.assertRaises(ValueError):
                variance_terms(matrix)
        spectral=np.ones((2,2),dtype=complex)
        r,s=variance_terms(spectral)
        np.testing.assert_allclose(corrected_variances(r,s,9),[1/9,1/9,0,1/9])

    def test_cross_validation_does_not_refit_to_test_errors(self):
        import csv,hashlib,inspect,json
        import check_cross_variance as cv
        import calibrate_m as cm
        import variance_convergence as vc
        rng=np.random.default_rng(786)
        f=np.array([.05,.07,.125,.2,.28])
        settings={name:dict(seed_start=start,records=8) for name,start in
                  [("test_baseline",0),("test_long_period",100),("test_weak_waves",200)]}
        protocol=dict(M=9,bootstrap=20,groups=settings,
                      candidate_sha256=hashlib.sha256((inspect.getsource(cv.variance_terms)+inspect.getsource(cv.corrected_variances)).encode()).hexdigest(),
                      native_generator_sha256=hashlib.sha256(inspect.getsource(cm.native_samples).encode()).hexdigest(),
                      parent_protocol=dict(source_fingerprint=vc.model_fingerprint(),
                                           inverse_source_sha256=hashlib.sha256(Path(iw.__file__).read_bytes()).hexdigest()))
        pairs=iw.stack_pairs(9)
        records=[]
        for _ in range(8):
            z=(rng.normal(size=(5,8,9))+1j*rng.normal(size=(5,8,9)))/np.sqrt(2)
            spectral=np.einsum("fki,fkj->fij",z.conj(),z)/8
            records.append(dict(f=f,spectral=spectral,
                                observed=np.stack([iw.data_vector(s,pairs) for s in spectral]),
                                numerator=np.stack([iw.noise_variances(s,pairs,1) for s in spectral])))
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/"protocol.json").write_text(json.dumps(protocol))
            for group,setting in settings.items():
                (root/group).mkdir()
                for i,record in enumerate(records):
                    values=dict(record)
                    if group=="test_baseline":
                        values["observed"]=record["observed"]*3
                    np.savez(root/group/f"seed-{setting['seed_start']+i}.npz",**values)
            with patch.object(cv,"training",return_value=({},f,9,np.full(20,9))),patch.object(cv,"plot"),patch("builtins.print"):
                cv.analyse(root,root,root)
            with (root/"cross_variance_validation.csv").open() as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual(len(rows),225)
            for row in rows:
                if row["group"]!="test_baseline":
                    continue
                reference=next(r for r in rows if r["group"]=="test_long_period" and
                               all(r[k]==row[k] for k in ["method","family","band"]))
                self.assertAlmostEqual(float(row["median_predicted_over_observed"])*9,
                                       float(reference["median_predicted_over_observed"]))
                if row["method"]!="existing_2.5":
                    self.assertEqual(float(row["M"]),9)

    def test_m_fit_recovers_known_value_and_preserves_units(self):
        from analyse_m_calibration import moments,fitted_m,families
        rng=np.random.default_rng(678)
        values=rng.normal(size=(40,5,81))+3
        q=np.broadcast_to(7.25*values.var(0,ddof=1),values.shape)
        weights=rng.multinomial(40,np.full(40,1/40),size=100)/40
        original=moments(values,q,weights)
        m,draws=fitted_m(original,families()["auto"])
        self.assertAlmostEqual(m,7.25)
        scale=np.exp(rng.uniform(-6,6,size=(5,81)))
        changed=moments(values*scale,q*scale**2,weights)
        mm,dd=fitted_m(changed,families()["auto"])
        self.assertAlmostEqual(mm,m)
        np.testing.assert_allclose(dd,draws,rtol=1e-12)
        actual_auto=moments(values,values**2,weights)
        code_m,_=fitted_m(actual_auto,families()["auto"])
        equivalent=np.median((actual_auto["mean"]**2/actual_auto["variance"])[:,families()["auto"]])
        self.assertAlmostEqual(code_m-equivalent,39/40)

    def test_m_analysis_keeps_training_fit_fixed_on_miscalibrated_holdout(self):
        import csv,json
        from analyse_m_calibration import analyse
        rng=np.random.default_rng(679)
        values=rng.uniform(1,4,size=(12,5,81))
        q=np.broadcast_to(7.25*values.var(0,ddof=1),values.shape)
        groups={name:dict(case=name,seed_start=start,records=12) for name,start in
                [("train_baseline",0),("test_baseline",100),("test_long_period",200),("test_weak_waves",300)]}
        protocol=dict(modes={"uniform6400":6400,"current40":40},groups=groups,bootstrap=20,
                      current40_groups=["train_baseline","test_baseline"])
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/"protocol.json").write_text(json.dumps(protocol))
            for mode in protocol["modes"]:
                for group,settings in groups.items():
                    if mode=="current40" and group not in protocol["current40_groups"]:
                        continue
                    directory=root/mode/group
                    directory.mkdir(parents=True)
                    for i in range(12):
                        # Deliberate held-out variance error must remain visible after fitting.
                        observed=values[i]*(3 if group=="test_baseline" else 1)
                        np.savez(directory/f"seed-{settings['seed_start']+i}.npz",f=[.05,.07,.125,.2,.28],
                                 observed=observed,numerator=q[i])
            with patch("analyse_m_calibration.plot"),patch("builtins.print"):
                analyse(root,root)
            with (root/"m_calibration_fits.csv").open() as stream:
                fits=list(csv.DictReader(stream))
            self.assertEqual(len(fits),8)
            for row in fits:
                self.assertAlmostEqual(float(row["M"]),7.25)
            with (root/"m_calibration_validation.csv").open() as stream:
                rows=list(csv.DictReader(stream))
            for row in rows:
                if row["method"]=="calibrated" and row["group"]=="test_baseline":
                    self.assertAlmostEqual(float(row["median_predicted_over_observed"]),1/9)
            protocol["groups"]["test_baseline"]["seed_start"]=0
            (root/"protocol.json").write_text(json.dumps(protocol))
            with self.assertRaisesRegex(ValueError,"seeds overlap"):
                analyse(root,root)

    def test_choice_analysis_recovers_known_scaling_for_both_experiments(self):
        import csv
        import json
        from analyse_variance_choice import analyse
        from variance_choice import CASES
        scales = dict(zip(CASES, [1.2, 1.1, 1.05, 1.01, 1., 1.5, 1.2]))
        values = np.random.default_rng(89).uniform(1, 4, size=(8, 5, 81))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            protocol = dict(cases=CASES, seeds=8, seed_start=0, bootstrap=20, tolerance=.1,
                            usable_lengths=[900, 1800, 3600])
            (root/"protocol.json").write_text(json.dumps(protocol))
            for name, count in CASES.items():
                for length in protocol["usable_lengths"]:
                    directory = root/name/f"n{count}_t{length}"
                    directory.mkdir(parents=True)
                    for seed in range(8):
                        np.savez(directory/f"seed-{seed}.npz", f=[.05, .07, .125, .2, .28],
                                 observed=values[seed]*scales[name])
            with patch("analyse_variance_choice.plot"), patch("builtins.print"):
                analyse(root, root)
            with (root/"choice_summary.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 270)
            for row in rows:
                expected_ratio = scales[row["first_case"]]/scales[row["reference_case"]]
                if row["metric"] == "variance":
                    expected_ratio **= 2
                self.assertAlmostEqual(float(row["median_ratio"]), expected_ratio)

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
            analyse(root, root, pairs=[(800, 3200)], prefix="subset", plotter=lambda *args: None)
            with (root / "subset_summary.csv").open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 45)


if __name__ == "__main__":
    unittest.main()
