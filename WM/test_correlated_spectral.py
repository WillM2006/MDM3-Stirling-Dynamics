import unittest
import numpy as np
from scipy.signal import get_window
import invert_waves as iw
from check_inverse_recovery import candidate_variances
from correlated_spectral import (wishart_factor,covariance_factors,welch_frequency_correlation,
                                 correlated_inverse,functional_vectors,propagated_variance)
from unittest.mock import patch


class CorrelatedSpectralTests(unittest.TestCase):
    def test_wishart_factor_matches_known_complex_gaussian_joint_covariance(self):
        rng=np.random.default_rng(581)
        root=np.array([[1,0,0],[.6+.2j,.7,0],[.1-.4j,.2j,.8]],complex)
        s=root@root.conj().T
        z=(rng.normal(size=(60000,5,3))+1j*rng.normal(size=(60000,5,3)))/np.sqrt(2)
        y=z@root.T
        measured=np.einsum("rki,rkj->rij",y,y.conj())/5
        data=np.stack([iw.data_vector(a,iw.stack_pairs(3)) for a in measured])
        expected=wishart_factor(s)@wishart_factor(s).T/5
        errors=(np.cov(data,rowvar=False)-expected)/np.sqrt(np.diag(expected)[:,None]*np.diag(expected)[None,:])
        self.assertLess(abs(errors).max(),.025)

    def test_factors_keep_calibrated_diagonal_and_preserve_units(self):
        rng=np.random.default_rng(582)
        y=rng.normal(size=(3,8,9))+1j*rng.normal(size=(3,8,9))
        s=np.einsum("fki,fkj->fij",y,y.conj())/8
        factors,shrink=covariance_factors(s,8.785)
        expected=np.stack([candidate_variances(a,iw.stack_pairs(9),8.785) for a in s])
        np.testing.assert_allclose(np.sum(factors**2,axis=2),expected,rtol=1e-13)
        self.assertTrue(np.all((shrink>0)&(shrink<=1)))
        units=np.exp(rng.uniform(-5,5,size=9))
        changed,changed_shrink=covariance_factors(s*units[:,None]*units[None,:],8.785)
        row_units=np.array([units[c]*units[k] for c,k in iw.stack_pairs(9) for _ in range(1 if c==k else 2)])
        np.testing.assert_allclose(changed, factors*row_units[None,:,None],rtol=1e-10,atol=1e-10)
        np.testing.assert_allclose(changed_shrink,shrink,atol=1e-13)

    def test_hann_overlap_correlations_match_white_noise_records(self):
        rng=np.random.default_rng(583)
        samples=rng.normal(size=(4000,1800))
        window=get_window("hann",400,fftbins=True)
        power=[]
        for start in range(0,1401,200):
            segment=samples[:,start:start+400]
            transformed=np.fft.rfft((segment-segment.mean(1,keepdims=True))*window,axis=1)
            power.append(abs(transformed[:,20:24])**2)
        observed=np.corrcoef(np.mean(power,axis=0),rowvar=False)
        expected=welch_frequency_correlation(np.arange(20,24))
        np.testing.assert_allclose(observed,expected,atol=.055)
        np.testing.assert_allclose(np.diag(expected),1)
        np.linalg.cholesky(expected)

    def test_structured_posterior_matches_explicit_dense_model(self):
        rng=np.random.default_rng(584)
        count,channels,headings=3,3,2
        f=np.array([.1,.105,.11])
        responses=rng.normal(size=(count,headings,channels))+1j*rng.normal(size=(count,headings,channels))
        y=rng.normal(size=(count,8,channels))+1j*rng.normal(size=(count,8,channels))
        s=np.einsum("fki,fkj->fij",y,y.conj())/8
        factors,_=covariance_factors(s,8.785)
        rho=welch_frequency_correlation(np.arange(20,23))
        scales=np.array([1.,2.,3.]);tau=5
        x,cov,residual=correlated_inverse(f,s,responses,tau,8.785,factors,rho,scales)
        d=np.concatenate([iw.data_vector(a,iw.stack_pairs(channels)) for a in s])
        a=np.zeros((count*channels**2,count*headings));factor=np.zeros((count*channels**2,count*channels**2))
        for fi in range(count):
            a[fi*channels**2:(fi+1)*channels**2,fi*headings:(fi+1)*headings]=iw.operator_matrix(responses[fi],iw.stack_pairs(channels))*scales[fi]
            factor[fi*channels**2:(fi+1)*channels**2,fi*channels**2:(fi+1)*channels**2]=factors[fi]
        sigma=factor@np.kron(rho,np.eye(channels**2))@factor.T
        precision=a.T@np.linalg.solve(sigma,a)+np.eye(count*headings)/tau**2
        dense_cov=np.linalg.inv(precision)
        dense_y=dense_cov@a.T@np.linalg.solve(sigma,d)
        np.testing.assert_allclose(x,dense_y.reshape(count,headings)*scales[:,None],rtol=1e-11)
        t=functional_vectors(scales,headings,.005)
        np.testing.assert_allclose(cov,t.T@dense_cov@t,rtol=1e-11,atol=1e-13)
        self.assertLess(residual,1e-13)
        probes=rng.normal(size=(5,count,channels**2))
        np.testing.assert_allclose(propagated_variance(probes,factors,rho),
                                   np.einsum("bi,ij,bj->b",probes.reshape(5,-1),sigma,probes.reshape(5,-1)),rtol=1e-12)

    def test_identity_correlations_reproduce_existing_diagonal_inverse(self):
        rng=np.random.default_rng(585)
        f=np.array([.1,.105,.11])
        h=rng.normal(size=(3,4,3))+1j*rng.normal(size=(3,4,3))
        s=np.einsum("fjc,fjk->fck",h,h.conj())+.1*np.eye(3)
        with patch.object(iw,"noise_variances",candidate_variances):
            x,cov,scales=iw.invert(f,s,h,5,8.785)
        factors=np.stack([np.diag(np.sqrt(candidate_variances(a,iw.stack_pairs(3),8.785))) for a in s])
        xx,functions,_=correlated_inverse(f,s,h,5,8.785,factors,np.eye(3),scales)
        np.testing.assert_allclose(xx,x,rtol=1e-10,atol=1e-11)
        np.testing.assert_allclose(functions[-1,-1],.005**2*cov.sum(),rtol=1e-10)

    def test_scoring_retains_joint_frequency_covariance_in_band_uncertainty(self):
        from check_correlated_covariance import score
        f=np.array([.1,.105,.11]);beta=np.array([0.,np.pi/2])
        truth=np.array([[1.,2.],[2.,4.],[1.,2.]])
        sigma=np.kron(np.full((3,3),.8)+.2*np.eye(3),np.eye(2)*.04)
        t=functional_vectors(np.ones(3),2,.005)
        covariance=t.T@sigma@t
        exact=score(truth,covariance,truth,f,beta)
        self.assertAlmostEqual(exact["sd_m0"]**2,.005**2*.04*2*(3+6*.8))
        self.assertAlmostEqual(exact["hs_abs_pct"],0)
        self.assertEqual(exact["m0_cover95"],1)
        self.assertEqual(exact["spectrum_cover95"],1)
        wrong=score(3*truth,covariance,truth,f,beta)
        self.assertAlmostEqual(wrong["hs_abs_pct"],100*(np.sqrt(3)-1))
        self.assertEqual(wrong["m0_cover95"],0)


if __name__=="__main__":
    unittest.main()
