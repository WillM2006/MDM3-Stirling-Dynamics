import unittest
from unittest.mock import patch
import numpy as np
import invert_waves as iw
from check_smoothed_weights import smooth_variances, weighted_inverse, candidate_variances

class WeightTests(unittest.TestCase):
    def test_smoothing_keeps_scale_and_channel_separation(self):
        v=np.ones((53,81))*np.arange(1,82)
        for width in [1,3,5,9,17]:np.testing.assert_allclose(smooth_variances(v,width),v)
        v[25,0]=10
        out=smooth_variances(v,5)
        self.assertLess(out[25,0],10)
        np.testing.assert_array_equal(out[:,1:],v[:,1:])
        np.testing.assert_allclose(out.sum(0),v.sum(0))
        with self.assertRaises(ValueError):smooth_variances(v*0,3)
    def test_unsmoothed_solver_matches_original(self):
        rng=np.random.default_rng(82)
        h=rng.normal(size=(3,4,9))+1j*rng.normal(size=(3,4,9))
        a=rng.normal(size=(3,9,9))+1j*rng.normal(size=(3,9,9))
        s=a@a.conj().transpose(0,2,1)+np.eye(9)
        with patch.object(iw,'noise_variances',candidate_variances):x,cov,scales=iw.invert(np.arange(3),s,h,5.,8.785)
        v=np.stack([candidate_variances(ss,iw.stack_pairs(9),8.785) for ss in s])
        xx,cc=weighted_inverse(s,h,v,5.,scales)
        np.testing.assert_allclose(xx,x,rtol=1e-10,atol=1e-12)
        np.testing.assert_allclose(cc,cov,rtol=1e-10,atol=1e-12)
