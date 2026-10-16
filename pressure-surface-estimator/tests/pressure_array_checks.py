import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
"""Array contract, rigid-body compensation and time/frequency checks.

Test coordinates are a mathematical fixture, NOT confirmed sponsor geometry.
"""
import unittest
import numpy as np
from pressure_array import PressureArray,array_response,simulate_array,reconstruct_above_ports,wave_interpolate,reconstruct_surface_joint
from sea import Sea,FixedSea
from submarine import Submarine
from sensors import sensor_response,Sensors

POSITIONS=((35.,0.,0.),(-35.,0.,0.),(0.,5.,0.),(0.,-5.,0.),(0.,0.,5.),(0.,0.,-5.))

class ArrayChecks(unittest.TestCase):
    def setUp(self):
        self.cfg=PressureArray(POSITIONS,'test fixture only; actual six-port geometry unconfirmed')

    def test_requires_six_explicit_distinct_positions_and_provenance(self):
        with self.assertRaises(ValueError):PressureArray(POSITIONS[:5],'test')
        with self.assertRaises(ValueError):PressureArray((POSITIONS[0],)*6,'test')
        with self.assertRaises(ValueError):PressureArray(POSITIONS,'')
        self.assertEqual(len(self.cfg.channels),12)

    def test_matches_existing_single_port_response(self):
        f=np.arange(.02,.32001,.005);theta=np.linspace(0,2*np.pi,36,endpoint=False)
        m=Submarine(omega=2*np.pi*f,theta=theta);H=array_response(m,m.omega,theta,self.cfg)
        for j,r in enumerate(POSITIONS):
            old=sensor_response(m,m.omega,theta,Sensors(self.cfg.imu_position,r))
            np.testing.assert_allclose(H[:,:,j],old[:,:,0],atol=1e-9)
            np.testing.assert_allclose(H[:,:,6:],old[:,:,1:],atol=1e-12)

    def test_non_JONSWAP_multitone_with_heave_roll_pitch(self):
        fs=2.;n=2048;t=np.arange(n)/fs;rho=1025.;g=9.81;depth=15.
        frequencies=np.array([.09375,.1875]);w=2*np.pi*frequencies;k=w*w/g
        theta=np.deg2rad([25.,-70.]);amplitudes=np.array([.8,.2]);phases=np.array([.73,-1.12])
        # Independent prescribed rigid-body motion, including angular terms.
        wm=2*np.pi*.0625;heave=.13*np.cos(wm*t+.4);roll=.02*np.cos(wm*t-.7);pitch=.015*np.sin(wm*t+.2)
        ri=np.asarray(self.cfg.imu_position);r=np.asarray(POSITIONS)
        imu_z=heave+roll*ri[1]-pitch*ri[0];az=-wm*wm*imu_z+g
        gx=-.02*wm*np.sin(wm*t-.7);gy=.015*wm*np.cos(wm*t+.2)
        signal=np.zeros((n,12));signal[:,8]=az;signal[:,9]=gx;signal[:,10]=gy
        truth=[]
        for j,pos in enumerate(r):
            spatial=k*(pos[0]*np.cos(theta)+pos[1]*np.sin(theta))
            components=amplitudes[None,:]*np.cos(w[None,:]*t[:,None]-spatial[None,:]-phases[None,:])
            eta=components.sum(axis=1);truth.append(eta)
            incident=rho*g*(components*np.exp(-k*(depth-pos[2]))[None,:]).sum(axis=1)
            z=heave+roll*pos[1]-pitch*pos[0]
            signal[:,j]=rho*g*(depth-pos[2]-z)+incident
        actual=reconstruct_above_ports(t,signal,self.cfg,depth=depth,band=(.02,.32))
        np.testing.assert_allclose(actual,np.array(truth).T,atol=1e-11,rtol=1e-11)

    def test_time_forward_and_spectral_array_agree(self):
        fs=2.;n=2048;grid=np.fft.rfftfreq(n,1/fs);f=grid[(grid>=.02)&(grid<=.32)]
        theta=np.linspace(0,2*np.pi,36,endpoint=False);m=Submarine(omega=2*np.pi*f,theta=theta)
        sea=Sea(freq=f,theta=theta,Hs=2.,Tp=8.,fixed=FixedSea(spread_s=3),seed=42)
        t,y,_,_=simulate_array(sea,m,self.cfg,fs=fs,n_samples=n,dt=.1,settle_seconds=400.)
        actual=reconstruct_above_ports(t,y,self.cfg,depth=m.depth,band=(f[0],f[-1]))
        coeff=np.zeros((n//2+1,6),complex);r=np.asarray(POSITIONS)
        spatial=sea.k[:,None,None]*(np.cos(theta)[None,:,None]*r[None,None,:,0]+np.sin(theta)[None,:,None]*r[None,None,:,1])
        z=np.einsum('fd,fdp->fp',sea.complex_components,np.exp(1j*spatial))*np.exp(-1j*sea.omega*t[0])[:,None]
        bins=np.rint(f*n/fs).astype(int);coeff[bins]=n/2*z.conj();truth=np.fft.irfft(coeff,n=n,axis=0)
        self.assertLess(np.sqrt(np.mean((actual-truth)**2)),2e-6)
        H=array_response(m,m.omega,theta,self.cfg)
        Z=np.einsum('fdc,fd->fc',H,sea.complex_components)*np.exp(-1j*sea.omega*t[0])[:,None]
        coeff=np.zeros((n//2+1,12),complex);coeff[bins]=n/2*Z.conj();pred=np.fft.irfft(coeff,n=n,axis=0)
        demeaned=y-y.mean(axis=0)
        relative=np.sqrt(np.sum((demeaned-pred)**2,axis=0)/np.sum(pred**2,axis=0))
        self.assertLess(relative.max(),5e-5)

    def test_wave_interpolator_reproduces_port_values_and_phase(self):
        n=2048;t=np.arange(n)/2;w=2*np.pi*.125;k=w*w/9.81
        xy=np.asarray(POSITIONS)[:,:2]
        eta=np.cos(w*t[:,None]-k*(xy[None,:,0]*np.cos(.7)+xy[None,:,1]*np.sin(.7))+.6)
        result=wave_interpolate(t,eta,POSITIONS,xy,band=(.12,.13))
        np.testing.assert_allclose(result,eta,atol=1e-9,rtol=1e-9)
        shifted=wave_interpolate(t,np.roll(eta,13,axis=0),POSITIONS,[[17.,2.]],band=(.12,.13))
        original=wave_interpolate(t,eta,POSITIONS,[[17.,2.]],band=(.12,.13))
        np.testing.assert_allclose(shifted,np.roll(original,13,axis=0),atol=1e-10,rtol=1e-10)

    def test_joint_inverse_recovers_observable_mode_with_phase(self):
        fs=2.;n=2048;t=np.arange(n)/fs;grid=np.fft.rfftfreq(n,1/fs);f=grid[(grid>=.02)&(grid<=.32)]
        theta=np.linspace(0,2*np.pi,36,endpoint=False);m=Submarine(omega=2*np.pi*f,theta=theta)
        from submarine import ResponseTable
        table=ResponseTable(omega=m.omega,theta=theta,rao=m.rao,depth=m.depth,reference=m.reference)
        H=array_response(m,m.omega,theta,self.cfg);i=np.argmin(abs(f-.125));scale=np.sqrt(np.mean(abs(H[i])**2,axis=0));A=H[i]/scale[None,:]
        K=A.T@A.conj()/len(theta);e,U=np.linalg.eigh(K)
        mode=A.conj()@U[:,-1];mode*=.3/np.linalg.norm(mode)*np.exp(.73j)
        Y=H[i].T@mode;coef=np.zeros((n//2+1,12),complex);coef[round(f[i]*n/fs)]=n/2*Y.conj();signals=np.fft.irfft(coef,n=n,axis=0)
        targets=np.array([[0.,0.],[17.,2.],[-12.,-3.]])
        k=(2*np.pi*f[i])**2/m.g;G=np.exp(1j*k*(np.cos(theta)[:,None]*targets[None,:,0]+np.sin(theta)[:,None]*targets[None,:,1]))
        Z=G.T@mode;expected=np.real(Z[None,:]*np.exp(-2j*np.pi*f[i]*t)[:,None])
        result,diagnostics=reconstruct_surface_joint(t,signals,self.cfg,table,targets,band=(.12,.13))
        np.testing.assert_allclose(result,expected,atol=2e-8,rtol=2e-8)
        self.assertGreater(diagnostics['rank_min'],0)

if __name__=='__main__':unittest.main(verbosity=2)
