import unittest
import numpy as np
from check_profile_weights import profile_smooth

class ProfileTests(unittest.TestCase):
    def test_preserves_sharp_profile_and_global_amplitude(self):
        p=np.exp(np.linspace(-20,20,53))[:,None]*np.arange(1,82)[None,:]
        for width in [1,3,5,9,17]:np.testing.assert_allclose(profile_smooth(7*p,p,width),7*p,rtol=1e-14)
        noisy=p.copy();noisy[20,0]*=4
        np.testing.assert_allclose(profile_smooth(noisy,p,1),noisy)
        self.assertLess(profile_smooth(noisy,p,5)[20,0],noisy[20,0])
        with self.assertRaises(ValueError):profile_smooth(noisy,p*0,3)
