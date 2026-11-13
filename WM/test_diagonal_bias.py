import unittest
import numpy as np
from diagnose_diagonal_bias import decompose, fixed_prior_gain
from check_fixed_estimator_scatter import energy_gain

class BiasTests(unittest.TestCase):
    def test_exact_identity_and_all_distinct_donors(self):
        g=np.array([1.,2.,4.,5.]);d=np.array([3.,2.,7.,8.]);ref=np.array([4.,5.,6.])
        cross=g[:,None]*d; r=g[:,None]*ref
        result=decompose(cross,r,g*5,15.)
        donor=np.mean([g[i]*d[j] for i in range(4) for j in range(4) if i!=j])
        self.assertAlmostEqual(result[-1],100*(donor/15-1))
        self.assertAlmostEqual(result[:4].sum(),result[4])
        np.testing.assert_allclose(result,decompose(cross*7,r*7,g*35,105))
    def test_constant_gain_has_no_coupling(self):
        d=np.array([1.,2.,3.,4.]);cross=np.tile(d,(4,1))
        r=decompose(cross,np.ones((4,6))*2.5,np.ones(4)*2.5,2.5)
        np.testing.assert_allclose(r,0,atol=1e-12)
    def test_fixed_prior_gain_matches_original_at_record_scales(self):
        rng=np.random.default_rng(81)
        h=rng.normal(size=(2,3,9))+1j*rng.normal(size=(2,3,9))
        a=rng.normal(size=(2,9,9))+1j*rng.normal(size=(2,9,9))
        s=a@a.conj().transpose(0,2,1)+np.eye(9)
        scales=np.array([np.mean([ss[c,c].real/np.mean(abs(hh[:,c])**2) for c in range(9)]) for ss,hh in zip(s,h)])
        f=np.array([.1,.105])
        np.testing.assert_allclose(fixed_prior_gain(f,s,h,8.785,5.,scales),energy_gain(f,s,h,8.785,5.),rtol=1e-10,atol=1e-12)
    def test_invalid_dimensions(self):
        with self.assertRaises(ValueError):decompose(np.eye(2),np.eye(2),np.ones(2),1.)
