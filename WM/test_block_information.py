import unittest
from check_block_information import segment_count,theoretical_averages,SCHEMES

class BlockTests(unittest.TestCase):
    def test_exact_windows_and_effective_counts(self):
        self.assertEqual(segment_count(1800),8)
        self.assertEqual(segment_count(900),3)
        self.assertEqual(segment_count(800),3)
        self.assertEqual(segment_count(399),0)
        self.assertAlmostEqual(theoretical_averages(3),2.892857142857143)
        self.assertAlmostEqual(theoretical_averages(8),7.629139072847682)
        for blocks in SCHEMES.values():self.assertLessEqual(blocks[0][1],blocks[1][0])

from analyse_block_information import moments,paired_stats
import numpy as np

class BlockAnalysisTests(unittest.TestCase):
    def test_scale_invariance_and_pairing(self):
        x=np.arange(1.,21.).reshape(10,2)
        np.testing.assert_allclose(moments(x),moments(x*5))
        result=paired_stats(x,x*2)
        np.testing.assert_allclose(result[0],1)
        np.testing.assert_allclose(result[1],200/3)
        with self.assertRaises(ValueError):moments(x*0)
