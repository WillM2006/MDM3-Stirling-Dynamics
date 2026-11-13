"""Independent algebra/processing checks for the mean audit, not recovery evidence."""
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
from scipy.signal import csd
import invert_waves as iw
from audit_spectral_mean import welch_weights, processing_gain, expected_spectra


class SpectralMeanTests(unittest.TestCase):
    def test_hann_expectation_and_complex_sign(self):
        f=np.arange(10,63)*.005;nu=np.array([.1263]);t=np.arange(1800)/2
        h=np.arange(1,10)+1j*np.linspace(-2,3,9)
        phases=[]
        for phase in [0,np.pi/2]:
            x=np.real(np.sqrt(2)*np.exp(-2j*np.pi*nu[0]*t+1j*phase)[:,None]*h)
            phases.append(np.array([[csd(x[:,c],x[:,k],fs=2,nperseg=400,noverlap=200)[1][10:63]
                                      for k in range(9)] for c in range(9)]).transpose(2,0,1))
        positive,negative=welch_weights(f,nu)
        predicted=expected_spectra(h[None,None,:],np.ones((1,1)),positive,negative)
        np.testing.assert_allclose(np.mean(phases,axis=0),predicted,atol=2e-9,rtol=1e-9)
        # With Re{h exp(-iwt)}, scipy conj(FFT_c)*FFT_k has h_c*conj(h_k).
        self.assertEqual(np.sign(predicted[np.argmax(positive[:,0]),0,1].imag),
                         np.sign((h[0]*h[1].conj()).imag))

    def test_hann_preserves_on_bin_variance(self):
        f=np.arange(1,200)*.005
        positive,negative=welch_weights(f,np.array([.125]))
        self.assertAlmostEqual(float((positive+negative).sum()*.005),1.,places=12)

    def test_processing_gain_matches_record_reader(self):
        frequency=.273
        measured={}
        for name,fs in [('attitude',50.),('accel',100.),('pressure',10.)]:
            t=np.arange(1,int(1200*fs))/fs;y=np.cos(2*np.pi*frequency*t)
            if name=='attitude':measured[name]=np.rec.fromarrays([t,y,y],names='t_s,pitch_rad,roll_rad')
            if name=='accel':measured[name]=np.rec.fromarrays([t,y,y,y,y],names='t_s,f_x_m_s2,f_y_m_s2,f_z_m_s2,vert_acc_m_s2')
            if name=='pressure':measured[name]=np.rec.fromarrays([t,*([y]*6)],names='t_s,'+','.join(f'P{i}_Pa' for i in range(1,7)))
        with patch.object(np,'genfromtxt',side_effect=lambda path,**kw:measured[Path(path).stem]):
            t,x=iw.read_channels('.',iw.CHANNELS)
        keep=(t>310)&(t<1185)
        carrier=np.column_stack([np.cos(2*np.pi*frequency*t[keep]),np.sin(2*np.pi*frequency*t[keep]),np.ones(keep.sum())])
        fit=np.linalg.lstsq(carrier,x[keep],rcond=None)[0]
        np.testing.assert_allclose(fit[0]+1j*fit[1],processing_gain(np.array([frequency]))[0],rtol=1e-9,atol=1e-9)


if __name__=='__main__':unittest.main()
