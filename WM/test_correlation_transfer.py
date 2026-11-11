import unittest
import numpy as np
from check_correlation_transfer import rescale_factor
from correlated_spectral import propagated_variance


class CorrelationTransferTests(unittest.TestCase):
    def test_rescaling_preserves_correlation_psd_and_exact_target_diagonal(self):
        rng=np.random.default_rng(284100)
        factor=rng.normal(size=(3,4,4))+3*np.eye(4)
        v=np.exp(rng.normal(size=(3,4)))
        changed=rescale_factor(factor,v)
        np.testing.assert_allclose(np.sum(changed**2,axis=-1),v,rtol=1e-13)
        original=rescale_factor(factor,np.ones_like(v))
        normalised=changed/np.sqrt(v[...,None])
        np.testing.assert_allclose(normalised,original,rtol=1e-13)
        np.testing.assert_allclose(rescale_factor(factor,np.sum(factor**2,axis=-1)),factor,rtol=1e-13)
        rho=.3*np.ones((3,3))+.7*np.eye(3)
        sigma=np.block([[rho[i,j]*changed[i]@changed[j].T for j in range(3)] for i in range(3)])
        self.assertGreater(np.linalg.eigvalsh(sigma).min(),0)
        g=rng.normal(size=(2,3,4))
        np.testing.assert_allclose(propagated_variance(g,changed,rho),
            np.einsum('bi,ij,bj->b',g.reshape(2,-1),sigma,g.reshape(2,-1)),rtol=1e-13)
        units=np.exp(rng.normal(size=(3,4)))
        converted=rescale_factor(factor*units[...,None],v*units**2)
        np.testing.assert_allclose(propagated_variance(g/units,converted,rho),propagated_variance(g,changed,rho),rtol=1e-13)

    def test_invalid_marginal_is_not_silently_clipped(self):
        for v in [np.zeros((2,3)),np.full((2,3),np.nan),np.ones((2,2))]:
            with self.assertRaises(ValueError):rescale_factor(np.broadcast_to(np.eye(3),(2,3,3)),v)

    def test_donor_diagnostic_retains_point_estimates_and_variance_distribution(self):
        from diagnose_scale_dependence import paired_exchange
        y=np.r_[np.ones(100),np.ones(100)*2]; truth=np.ones(200)
        variance=np.r_[np.ones(100)*.01,np.ones(100)]
        saved=y.copy(); draws=np.tile(np.arange(100),(10,1))
        result=paired_exchange(y,truth,variance,draws)
        self.assertEqual(result['original_coverage'],1)
        self.assertEqual(result['donor_coverage'],.5)
        self.assertEqual(result['change_ci95_high'],-.5)
        np.testing.assert_array_equal(y,saved)


if __name__=='__main__':
    unittest.main()
