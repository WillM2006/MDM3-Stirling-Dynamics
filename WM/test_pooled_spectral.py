import unittest
import numpy as np
from pooled_spectral import fit_pool,shrink_correlation,pooled_inverse
from check_cross_variance import variance_terms,corrected_variances
from correlated_spectral import covariance_factors,welch_frequency_correlation,correlated_inverse


class PooledSpectralTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(72815)
        y=rng.normal(size=(40,3,8,3))+1j*rng.normal(size=(40,3,8,3))
        self.spectra=np.einsum("rfki,rfkj->rfij",y,y.conj())/8

    def test_pool_estimates_new_record_noise_not_mean_noise(self):
        factors,_=fit_pool(self.spectra,8.785)
        expected=corrected_variances(*variance_terms(self.spectra),8.785).mean(0)
        np.testing.assert_allclose(np.sum(factors**2,axis=2),expected,rtol=1e-13)
        doubled,_=fit_pool(np.concatenate([self.spectra,self.spectra]),8.785)
        np.testing.assert_allclose(np.sum(doubled**2,axis=2),expected,rtol=1e-13)

    def test_pool_is_invariant_to_physical_units(self):
        factors,shrink=fit_pool(self.spectra,8.785)
        units=np.array([.001,2,1000])
        changed,changed_shrink=fit_pool(self.spectra*units[:,None]*units[None,:],8.785)
        row_units=np.array([units[c]*units[k] for c in range(3) for k in range(c,3) for _ in range(1 if c==k else 2)])
        np.testing.assert_allclose(changed,factors*row_units[None,:,None],rtol=1e-11,atol=1e-9)
        np.testing.assert_allclose(changed_shrink,shrink,atol=1e-13)

    def test_zero_empirical_correlations_shrink_to_identity(self):
        t=np.arange(64)
        values=np.column_stack([np.sin(2*np.pi*t*k/64) for k in [1,2,3]])
        factor,intensity=shrink_correlation(values)
        np.testing.assert_allclose(factor@factor.T,np.eye(3),atol=1e-13)
        self.assertEqual(intensity,1)
        with self.assertRaises(ValueError):shrink_correlation(np.ones((20,3)))
        with self.assertRaises(ValueError):shrink_correlation(values[:3])

    def test_preconditioned_solver_preserves_the_gaussian_model(self):
        rng=np.random.default_rng(72816)
        f=np.array([.1,.105,.11]);s=self.spectra[0]
        h=rng.normal(size=(3,4,3))+1j*rng.normal(size=(3,4,3))
        factors,_=covariance_factors(s,8.785)
        rho=welch_frequency_correlation(np.arange(20,23))
        scales=np.array([1.,2.,3.])
        old_x,old_c,_=correlated_inverse(f,s,h,5,8.785,factors,rho,scales)
        x,c,residual=pooled_inverse(f,s,h,5,factors,rho,scales)
        np.testing.assert_allclose(x,old_x,rtol=1e-10,atol=1e-12)
        np.testing.assert_allclose(c,old_c,rtol=1e-10,atol=1e-12)
        self.assertLess(residual,1e-13)

    def test_preconditioned_solver_retains_prior_in_extreme_rank_deficiency(self):
        f=np.array([.1,.105,.11]);s=self.spectra[0]
        h=np.ones((3,4,3),complex)
        factors=np.tile(np.eye(9)*1e-9,(3,1,1))
        scales=np.ones(3)
        x,c,residual=pooled_inverse(f,s,h,5,factors,np.eye(3),scales)
        self.assertTrue(np.isfinite(x).all() and np.isfinite(c).all())
        self.assertTrue(np.all(np.diag(c)>0))
        np.testing.assert_allclose(x,x.mean(1,keepdims=True)*np.ones_like(x),rtol=1e-5,atol=1e-5)
        self.assertLess(residual,1e-13)

    def test_failed_recoveries_remain_in_height_statistics(self):
        from analyse_pooled_covariance import finite_bootstrap
        values=np.array([[5.,1.,np.nan],[7.,2.,np.nan],[100.,np.nan,np.nan],[9.,3.,np.nan]])
        weights=np.full((10,4),.25)
        mean,low,high,counts=finite_bootstrap(values,weights)
        self.assertAlmostEqual(mean[0],30.25)
        self.assertAlmostEqual(mean[1],2.)
        self.assertTrue(np.isnan(mean[2]))
        np.testing.assert_array_equal(counts,[4,3,0])
        np.testing.assert_allclose(low[:2],mean[:2])
        np.testing.assert_allclose(high[:2],mean[:2])


if __name__=="__main__":unittest.main()
