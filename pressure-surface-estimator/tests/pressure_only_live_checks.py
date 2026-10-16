import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
"""Boundary, causality and implementation checks for the pressure-only stream."""
import unittest
import numpy as np
from pressure_only_live import Calibration,PressureOnlyLive,calibrate
from pressure_array import symmetric_six_port_layout
from submarine import ResponseTable
from pathlib import Path

class LiveChecks(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(718)
        self.cal=Calibration(rng.normal(size=(3,8,6)),np.zeros((3,2)),np.arange(6.),.5,{})
        self.p=rng.normal(size=(30,6))

    def test_exact_past_only_convolution_and_startup(self):
        live=PressureOnlyLive(self.cal)
        for j,p in enumerate(self.p):
            out=live.push(j*.5,p)
            if j<7:self.assertIsNone(out)
            else:
                expected=np.einsum('plc,lc->p',self.cal.weights,self.p[j-7:j+1][::-1]-self.cal.baseline_Pa)
                np.testing.assert_allclose(out,expected,atol=1e-12)

    def test_future_packets_cannot_change_previous_outputs(self):
        def run(p):
            live=PressureOnlyLive(self.cal)
            return [live.push(i*.5,row) for i,row in enumerate(p)]
        changed=self.p.copy();changed[20:]+=1e8
        a=run(self.p);b=run(changed)
        for i in range(7,20):np.testing.assert_array_equal(a[i],b[i])
        self.assertGreater(np.linalg.norm(a[-1]-b[-1]),1e7)

    def test_reject_IMU_extra_columns_and_bad_time(self):
        live=PressureOnlyLive(self.cal)
        for bad in [np.zeros(12),np.zeros(7),np.zeros(5),np.full(6,np.nan)]:
            with self.assertRaises(ValueError):live.push(0,bad)
        live.push(0,np.zeros(6))
        for time in [0,-.5,1.,float('nan')]:
            with self.assertRaises(ValueError):live.push(time,np.zeros(6))

    def test_physics_calibration_is_phase_blind_and_rejects_DC(self):
        table=ResponseTable.load(Path(__file__).resolve().parents[1]/'src/results-joint/response_table.npz')
        cal=calibrate(table,symmetric_six_port_layout(),[[0,0],[2,4]],taps=32)
        np.testing.assert_allclose(cal.weights.sum(axis=1),0,atol=1e-13)
        self.assertEqual(cal.metadata['output_time_delay_s'],0)
        self.assertNotIn('seed',cal.metadata)
        a=PressureOnlyLive(cal);b=PressureOnlyLive(cal)
        rng=np.random.default_rng(19);offset=np.array([1e3,-1e4,3e3,4e3,-5e3,6e3])
        for i,p in enumerate(rng.normal(0,100,(40,6))+cal.baseline_Pa):
            x=a.push(i*.5,p);y=b.push(i*.5,p+offset)
            if x is not None:np.testing.assert_allclose(x,y,atol=1e-10)

if __name__=='__main__':unittest.main()
