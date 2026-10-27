"""Causal six-pressure-only surface estimator; no sea or IMU input.

Offline calibration uses a known nominal linear hull/sensor response and an
explicit flat frequency, isotropic direction prior. This supplies missing
spatial information: it does not make the six-port inverse unique. Each push
uses exactly one timestamp and six pressures, retaining only past samples.
"""
from dataclasses import dataclass
import json
import numpy as np
from scipy.linalg import solve
from pressure_array import array_response


@dataclass
class Calibration:
    weights: np.ndarray  # [target, lag, port], lag 0 is current sample
    targets: np.ndarray
    baseline_Pa: np.ndarray
    sample_interval_s: float
    metadata: dict

    def save(self, path):
        np.savez_compressed(path, weights=self.weights, targets=self.targets,
                            baseline_Pa=self.baseline_Pa,
                            sample_interval_s=self.sample_interval_s,
                            metadata=json.dumps(self.metadata))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            return cls(z['weights'],z['targets'],z['baseline_Pa'],
                       float(z['sample_interval_s']),json.loads(str(z['metadata'])))


def calibrate(model, config, targets, *, sample_interval_s=.5, taps=128,
              ridge_m2=1e-8):
    """Solve a finite, causal, linear minimum-mean-square inverse from physics.

    No simulated/measured sea, fitted JONSWAP, phases, states or IMU supplied.
    Prior surface variance = 1 m^2, equally divided over response frequencies
    and directions. ridge_m2 penalizes gain as hypothetical independent noise
    in pressure-head metres squared; it is not a measured sensor noise spec.
    Six exact zero-DC constraints reject arbitrary constant pressure offsets.
    """
    xy=np.asarray(targets,float)
    if xy.ndim!=2 or xy.shape[1]!=2 or not np.isfinite(xy).all():
        raise ValueError('Finite target xy required')
    if taps<8 or sample_interval_s<=0 or ridge_m2<=0:
        raise ValueError('At least eight taps and positive interval/ridge required')
    omega=np.asarray(model.omega);theta=np.asarray(model.theta)
    if omega[-1]*sample_interval_s>=np.pi:
        raise ValueError('Response frequency exceeds sample Nyquist')
    if np.any(np.diff(omega)<=0) or not np.allclose(np.diff(omega),np.diff(omega)[0]):
        raise ValueError('Uniform increasing response frequencies required')
    # Pa -> metres pressure head; keep ONLY the six pressure responses.
    H=array_response(model,omega,theta,config)[:,:,:6]/(model.rho*model.g)
    Q=np.einsum('fdc,fde->fce',H,H.conj())/len(theta)
    lag=np.arange(taps)*sample_interval_s
    prior=np.full(len(omega),1/len(omega))
    Rlag=np.einsum('fl,fce,f->lce',np.exp(1j*omega[:,None]*lag),Q,prior).real
    delta=np.arange(taps)[:,None]-np.arange(taps)[None,:]
    R=np.where((delta>=0)[:,:,None,None],Rlag[abs(delta)],
               Rlag[abs(delta)].transpose(0,1,3,2)).transpose(0,2,1,3).reshape(taps*6,taps*6)
    R+=ridge_m2*np.eye(taps*6)
    k=omega**2/model.g
    P=np.empty((len(xy),taps*6))
    for start in range(0,len(xy),24):
        points=xy[start:start+24]
        E=np.exp(1j*k[:,None,None]*(np.cos(theta)[None,:,None]*points[None,None,:,0]
                                   +np.sin(theta)[None,:,None]*points[None,None,:,1]))
        cross=np.einsum('fdp,fdc->fpc',E,H.conj())/len(theta)
        P[start:start+24]=np.einsum('fl,fpc,f->plc',np.exp(-1j*omega[:,None]*lag),cross,prior).real.reshape(len(points),-1)
    W=solve(R,P.T,assume_a='pos').T
    D=np.tile(np.eye(6),(taps,1))
    RiD=solve(R,D,assume_a='pos')
    W-=(W@D)@solve(D.T@RiD,RiD.T,assume_a='pos')
    # Roundoff-only cleanup of exact zero-DC constraints.
    W=W.reshape(len(xy),taps,6)
    W-=W.sum(axis=1)[:,None,:]/taps
    prior_mse=1-2*np.einsum('pl,pl->p',W.reshape(len(xy),-1),P)
    prior_mse+=np.einsum('pl,lk,pk->p',W.reshape(len(xy),-1),R,W.reshape(len(xy),-1))
    baseline=(model.depth-np.asarray(config.positions)[:,2])*model.rho*model.g
    metadata={'inputs':'timestamp and six absolute pressures in Pa only',
              'positions_m':config.positions,'placement_status':config.placement_status,
              'known_calibration':'Nominal hull response, port geometry, depth, rho, g, sample interval',
              'prior':'flat over calibration frequencies, isotropic over directions; no JONSWAP fit',
              'band_Hz':[float(omega[0]/(2*np.pi)),float(omega[-1]/(2*np.pi))],
              'taps':taps,'startup_s':float(lag[-1]),'output_time_delay_s':0.,
              'ridge_m2':ridge_m2,'equivalent_regularizer_Pa':float(np.sqrt(ridge_m2)*model.rho*model.g),
              'zero_DC_constraints':True,'design_prior_MSE_m2':prior_mse.tolist(),
              'limits':'Matched linear physics, finite calibrated frequency band; unobservable spatial modes remain prior-dependent'}
    # Store direct weights from Pa to metres for a minimal deployment interface.
    return Calibration(W/(model.rho*model.g),xy,baseline,sample_interval_s,metadata)


class PressureOnlyLive:
    """One-way streaming boundary. Never accepts truth, sea objects, or IMU."""
    def __init__(self,calibration):
        self.calibration=calibration
        self.history=np.zeros(calibration.weights.shape[1:])
        self.count=0
        self.last_time=None

    def push(self,time_s,pressure_Pa):
        p=np.asarray(pressure_Pa,float)
        if p.shape!=(6,) or not np.isfinite(p).all() or not np.isfinite(time_s):
            raise ValueError('One finite timestamp and exactly six pressures required')
        if self.last_time is not None and not np.isclose(time_s-self.last_time,self.calibration.sample_interval_s,rtol=0,atol=1e-6):
            raise ValueError('Samples must arrive once, in order, at the calibrated interval')
        self.history[1:]=self.history[:-1]
        self.history[0]=p-self.calibration.baseline_Pa
        self.count+=1
        self.last_time=float(time_s)
        if self.count<len(self.history):
            return None
        return np.einsum('plc,lc->p',self.calibration.weights,self.history,optimize=False)
