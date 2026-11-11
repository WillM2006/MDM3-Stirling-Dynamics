import unittest
import numpy as np
from diagnose_undercoverage import tails,counterfactuals,SCENARIOS


class UndercoverageTests(unittest.TestCase):
    def test_all_donor_pairings_equal_average_over_every_nonzero_rotation(self):
        e=np.array([-5.,-1.,0.,2.,7.]);v=np.array([.1,2.,3.,4.,9.])
        explicit=np.mean([tails(e,np.roll(v,k),'paired') for k in range(1,len(e))],axis=0)
        np.testing.assert_allclose(tails(e,v,'independent'),explicit,atol=1e-14)
        for mode in ['paired','independent','constant']:
            self.assertAlmostEqual(tails(e,v,mode).sum(),1)

    def test_diagnostic_parameters_use_reference_and_do_not_mutate_errors(self):
        e=np.linspace(2,3,20);v=np.full(20,.01);ref=np.linspace(2,3,40);rv=np.full(40,.01)
        original=e.copy();out=counterfactuals(e,v,ref,rv)
        self.assertEqual(out[SCENARIOS.index((0,0,'paired')),0],0)
        self.assertGreater(out[SCENARIOS.index((1,1,'paired')),0],.9)
        np.testing.assert_array_equal(e,original)
        np.testing.assert_allclose(counterfactuals(e*7,v*49,ref*7,rv*49),out,atol=1e-14)
        np.testing.assert_allclose(tails(e,v,'constant'),tails(e,v,'paired'))

    def test_bad_inputs_are_rejected(self):
        with self.assertRaises(ValueError):tails(np.ones(4),np.zeros(4),'paired')


if __name__=='__main__':unittest.main()
