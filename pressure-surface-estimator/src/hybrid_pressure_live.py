"""Learned sea-statistics prior + causal phase-preserving pressure inverse.

Statistical parameters come only from an initial past-pressure window. They
guide a causal FIR; raw current/past pressures supply individual wave timing.
No hidden surface, generator parameters, IMU or future samples are accepted.
"""
import time
import numpy as np
from scipy.linalg import solve
from pressure_array import array_response
from pressure_only_live import Calibration,PressureOnlyLive
from pressure_prior import estimate_prior


def calibrate_hybrid(model,config,targets,prior,*,sample_interval_s=.5,taps=128,
                     ridge_m2=1e-8,uniform_fraction=.01):
    xy=np.asarray(targets,float);omega=np.asarray(model.omega);theta=np.asarray(model.theta)
    if xy.ndim!=2 or xy.shape[1]!=2 or not np.isfinite(xy).all():raise ValueError('Finite target xy required')
    if not 0<=uniform_fraction<=1 or taps<8 or ridge_m2<=0 or sample_interval_s<=0:raise ValueError('Invalid filter settings')
    weights=prior.weights(omega/(2*np.pi),theta,uniform_fraction=uniform_fraction)
    H=array_response(model,omega,theta,config)[:,:,:6]/(model.rho*model.g)
    Q=np.einsum('fdc,fde,fd->fce',H,H.conj(),weights)
    lag=np.arange(taps)*sample_interval_s
    Rlag=np.einsum('fl,fce->lce',np.exp(1j*omega[:,None]*lag),Q).real
    delta=np.arange(taps)[:,None]-np.arange(taps)[None,:]
    R=np.where((delta>=0)[:,:,None,None],Rlag[abs(delta)],Rlag[abs(delta)].transpose(0,1,3,2)).transpose(0,2,1,3).reshape(taps*6,taps*6)
    R+=ridge_m2*np.eye(taps*6)
    P=np.empty((len(xy),taps*6));k=omega**2/model.g
    for start in range(0,len(xy),24):
        points=xy[start:start+24]
        E=np.exp(1j*k[:,None,None]*(np.cos(theta)[None,:,None]*points[None,None,:,0]+np.sin(theta)[None,:,None]*points[None,None,:,1]))
        cross=np.einsum('fdp,fdc,fd->fpc',E,H.conj(),weights)
        P[start:start+24]=np.einsum('fl,fpc->plc',np.exp(-1j*omega[:,None]*lag),cross).real.reshape(len(points),-1)
    W=solve(R,P.T,assume_a='pos').T
    D=np.tile(np.eye(6),(taps,1));RiD=solve(R,D,assume_a='pos')
    W-=(W@D)@solve(D.T@RiD,RiD.T,assume_a='pos')
    W=W.reshape(len(xy),taps,6);W-=W.sum(axis=1)[:,None,:]/taps
    return Calibration(W/(model.rho*model.g),xy,
                       (model.depth-np.asarray(config.positions)[:,2])*model.rho*model.g,
                       sample_interval_s,{'method':'Pressure-learned directional/frequency prior + causal current-time FIR',
                         'learned_prior':prior.to_dict(),'uniform_direction_fraction':uniform_fraction,
                         'taps':taps,'ridge_m2':ridge_m2,'output_time_delay_s':0.,
                         'depth_m':model.depth,'band_Hz':[float(omega[0]/(2*np.pi)),float(omega[-1]/(2*np.pi))],
                         'input':'Six past/current pressures only; nominal hull calibration known',
                         'limits':'Single-system separable prior, fixed gamma; numerical stabilization is not calibrated sensor noise; no model uncertainty intervals'})


class HybridPressureLive:
    """Synchronous initial-learning prototype; first output includes fit latency.

    During startup stores pressures. At learn_samples it fits ONLY this past
    window, calibrates, and primes the FIR with its last taps samples. Subsequent
    calls use only the current packet and causal history. The prior is then
    frozen for this stationary-sea benchmark; no periodic adaptation claimed.
    """
    def __init__(self,model,config,targets,*,sample_interval_s=.5,learn_samples=1024,
                 taps=128,ridge_m2=1e-8,uniform_fraction=.01):
        if learn_samples<max(512,taps):raise ValueError('Learning history too short')
        self.model=model;self.config=config;self.targets=np.asarray(targets,float)
        self.dt=sample_interval_s;self.learn_samples=learn_samples;self.taps=taps
        self.ridge_m2=ridge_m2;self.uniform_fraction=uniform_fraction
        self.samples=[];self.times=[];self.live=None;self.fit=None;self.learning_compute_s=None

    def push(self,time_s,pressures_Pa):
        p=np.asarray(pressures_Pa,float)
        if p.shape!=(6,) or not np.isfinite(p).all() or not np.isfinite(time_s):raise ValueError('One timestamp and six pressures required')
        if self.live is not None:return self.live.push(time_s,p)
        if self.times and not np.isclose(time_s-self.times[-1],self.dt,rtol=0,atol=1e-6):raise ValueError('Nonuniform/out-of-order samples')
        self.times.append(float(time_s));self.samples.append(p.copy())
        if len(self.samples)<self.learn_samples:return None
        start=time.perf_counter()
        self.fit=estimate_prior(self.samples,self.model,self.config,sample_interval_s=self.dt)
        self.calibration=calibrate_hybrid(self.model,self.config,self.targets,self.fit,
                        sample_interval_s=self.dt,taps=self.taps,ridge_m2=self.ridge_m2,uniform_fraction=self.uniform_fraction)
        self.live=PressureOnlyLive(self.calibration)
        for t,row in zip(self.times[-self.taps:],self.samples[-self.taps:]):out=self.live.push(t,row)
        self.learning_compute_s=time.perf_counter()-start
        self.samples=[];self.times=[]
        return out
