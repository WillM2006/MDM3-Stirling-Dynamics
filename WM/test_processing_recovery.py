"""Check that processing comparisons retain the stated Gaussian prior."""
import unittest
import numpy as np
import invert_waves as iw
from check_inverse_recovery import candidate_variances
from check_processing_recovery import fixed_prior_inverse


class ProcessingRecoveryTests(unittest.TestCase):
    def test_fixed_prior_svd_agrees_with_direct_gaussian_algebra(self):
        rng=np.random.default_rng(274)
        z=rng.normal(size=(2,9,25))+1j*rng.normal(size=(2,9,25))
        spectra=z@z.swapaxes(1,2).conj()/25
        h=rng.normal(size=(2,4,9))+1j*rng.normal(size=(2,4,9))
        scales=np.array([.2,.3]);m=8.785033150237455;tau=5.
        x,cov=fixed_prior_inverse(spectra,h,m,tau,scales)
        for fi in range(2):
            a=iw.operator_matrix(h[fi],iw.stack_pairs(9));d=iw.data_vector(spectra[fi],iw.stack_pairs(9))
            v=candidate_variances(spectra[fi],iw.stack_pairs(9),m)
            precision=a.T@(a/v[:,None])+np.eye(4)/(tau*scales[fi])**2
            expected=np.linalg.inv(precision)
            np.testing.assert_allclose(cov[fi],expected,rtol=1e-11,atol=1e-13)
            np.testing.assert_allclose(x[fi],expected@(a.T@(d/v)),rtol=1e-11,atol=1e-13)

    def test_unresolved_component_keeps_frozen_prior_variance(self):
        h=np.ones((1,2,9),complex);scales=np.array([.5]);tau=5.
        x,cov=fixed_prior_inverse(np.eye(9)[None,:,:].astype(complex),h,8.785033150237455,tau,scales)
        null=np.array([1.,-1.])/np.sqrt(2)
        self.assertAlmostEqual(float(null@cov[0]@null),(tau*scales[0])**2,places=11)
        self.assertAlmostEqual(float(null@x[0]),0.,places=11)


if __name__=='__main__':unittest.main()
