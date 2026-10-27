import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
"""Hybrid learning boundary and pressure-derived statistical-fit checks."""
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from hybrid_pressure_live import HybridPressureLive
from pressure_only_live import Calibration
from pressure_prior import estimate_prior,PressurePrior
from pressure_array import symmetric_six_port_layout
from submarine import ResponseTable

class HybridChecks(unittest.TestCase):
    def test_learning_and_phase_reconstruction_use_only_received_packets(self):
        rng=np.random.default_rng(19);p=rng.normal(size=(540,6));future=p.copy();future[530:]+=1e5
        captured=[]
        def fit(rows,*args,**kwargs):captured.append(np.array(rows));return SimpleNamespace()
        cal=Calibration(rng.normal(size=(2,8,6)),np.zeros((2,2)),np.zeros(6),.5,{})
        def run(data):
            live=HybridPressureLive(None,None,[[0,0],[10,0]],learn_samples=512,taps=8)
            output=[live.push(i*.5,row) for i,row in enumerate(data)]
            return output
        with patch('hybrid_pressure_live.estimate_prior',fit),patch('hybrid_pressure_live.calibrate_hybrid',return_value=cal):
            a=run(p);b=run(future)
        for seen in captured:np.testing.assert_array_equal(seen,p[:512])
        self.assertTrue(all(v is None for v in a[:511]))
        for i in range(511,530):np.testing.assert_array_equal(a[i],b[i])
        self.assertGreater(np.linalg.norm(a[-1]-b[-1]),1e4)

    def test_extra_channels_and_time_order_rejected_during_learning(self):
        live=HybridPressureLive(None,None,[[0,0]])
        with self.assertRaises(ValueError):live.push(0,np.zeros(12))
        live.push(0,np.zeros(6))
        with self.assertRaises(ValueError):live.push(0,np.zeros(6))
        with self.assertRaises(ValueError):live.push(1,np.zeros(6))

    def test_directional_prior_normalization_and_uniform_mixture(self):
        prior=PressurePrior(34,8,2,8,3.3,0,1024,511.5,False,{})
        f=np.linspace(.02,.32,151);theta=np.linspace(0,2*np.pi,36,endpoint=False)
        w=prior.weights(f,theta,uniform_fraction=.01)
        self.assertAlmostEqual(w.sum(),1,places=12)
        self.assertTrue((w>=0).all())
        active=w.sum(axis=1)>1e-12
        direction=w[active]/w[active].sum(axis=1,keepdims=True)
        self.assertTrue((direction>0).all())
        np.testing.assert_allclose(direction,np.tile(direction[0],(active.sum(),1)),atol=1e-12)

    def test_pressure_fit_handles_finite_window_and_constant_port_offsets(self):
        root=Path(__file__).resolve().parents[1]/'outputs/pressure-only-live'
        p=np.loadtxt(root/'pressures_seed42.csv',delimiter=',',skiprows=1)[:1024,1:]
        model=ResponseTable.load(Path(__file__).resolve().parents[1]/'src/results-joint/response_table.npz')
        cfg=symmetric_six_port_layout()
        a=estimate_prior(p,model,cfg)
        b=estimate_prior(p+np.array([2e4,-3e4,1e4,2e3,-3e3,3e4]),model,cfg)
        self.assertLess(abs(a.mean_direction_deg-30),2)
        self.assertLess(abs(a.Tp_s-8),.2)
        self.assertLess(abs(a.spread_s-10),2)
        self.assertAlmostEqual(a.mean_direction_deg,b.mean_direction_deg,places=3)
        self.assertAlmostEqual(a.Tp_s,b.Tp_s,places=5)
        self.assertAlmostEqual(a.Hs_m,b.Hs_m,places=5)

if __name__=='__main__':unittest.main()
