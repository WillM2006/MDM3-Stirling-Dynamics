"""Ideal first-order sensor layer: separate from vehicle and inverse model.

No measurement noise, filters, bias, quantization or differentiation of samples.
"""
from dataclasses import dataclass
import numpy as np
from submarine import skew

CHANNELS = ('pressure_Pa','accel_x_m_s2','accel_y_m_s2','accel_z_m_s2',
            'gyro_x_rad_s','gyro_y_rad_s','gyro_z_rad_s')


@dataclass(frozen=True)
class Sensors:
    # Off-centre IMU demonstrates rotational lever-arm acceleration.
    imu_position: tuple = (10.,0.,0.)
    pressure_position: tuple = (10.,0.,5.)


def sensor_response(model, omega, theta, config=None):
    """Response per metre of surface amplitude at the reference origin.

    Small-angle, level mean attitude; body axes coincide with earth mean axes.
    Pressure = incident wave + first-order hydrostatic change at moving port.
    Hull-scattered pressure is not supplied by this baseline.
    """
    config = config or Sensors()
    q = model.motion_response(omega,theta)
    w = np.asarray(omega)[:,None,None]
    r_imu, r_p = np.asarray(config.imu_position), np.asarray(config.pressure_position)
    if model.depth-r_p[2] <= 0:
        raise ValueError("Mean pressure port must be submerged")
    sensor_displacement = q[:,:,:3] - np.einsum('ij,fdj->fdi',skew(r_imu),q[:,:,3:])
    acc = -w*w*sensor_displacement
    # R^T(a-g): first-order tilt contribution; DC accel_z is +g.
    acc[:,:,0] -= model.g*q[:,:,4]
    acc[:,:,1] += model.g*q[:,:,3]
    gyro = -1j*w*q[:,:,3:]
    port_displacement = q[:,:,:3] - np.einsum('ij,fdj->fdi',skew(r_p),q[:,:,3:])
    k = np.asarray(omega)**2/model.g
    spatial = k[:,None]*(r_p[0]*np.cos(theta)[None,:]+r_p[1]*np.sin(theta)[None,:])
    incident = model.rho*model.g*np.exp(-k[:,None]*(model.depth-r_p[2]))*np.exp(1j*spatial)
    pressure = incident-model.rho*model.g*port_displacement[:,:,2]
    return np.concatenate([pressure[:,:,None],acc,gyro],axis=2)


def spectral_gain(transfer, fractions):
    return np.sum(np.abs(transfer)**2*np.asarray(fractions)[None,:,None],axis=1)


def synthesise(sea, model, *, fs, n_samples, config=None):
    """Exact Fourier-grid periodic steady state; no startup transient.

    Uses the team's fixed-amplitude/random-phase components. Distinct directions
    share each frequency; cross terms in an individual record are retained.
    """
    config = config or Sensors()
    full_f = np.fft.rfftfreq(n_samples,1/fs)
    bins = np.rint(sea.f/(fs/n_samples)).astype(int)
    if np.any(bins<=0) or np.any(bins>=len(full_f)-1) or not np.allclose(full_f[bins],sea.f):
        raise ValueError("Sea frequencies must be interior FFT bins")
    if not np.isclose(sea.delta_f,fs/n_samples):
        raise ValueError("Use consecutive Fourier bins at the record resolution")
    transfer = sensor_response(model,sea.omega,sea.theta,config)
    Y = np.einsum('fdc,fd->fc',transfer,sea.complex_components)
    Q = np.einsum('fdk,fd->fk',model.motion_response(sea.omega,sea.theta),sea.complex_components)
    def trace(z):
        coef = np.zeros((len(full_f),z.shape[1]),complex)
        # rfft uses exp(-i wt) analysis: its positive-frequency coefficients
        # are conjugates of our exp(-i wt) PHYSICAL complex amplitudes.
        coef[bins] = n_samples/2*np.conj(z)
        return np.fft.irfft(coef,n=n_samples,axis=0)
    y = trace(Y)
    y[:,0] += model.rho*model.g*(model.depth-config.pressure_position[2])
    y[:,3] += model.g
    return np.arange(n_samples)/fs,y,trace(Q)


def periodogram(y, fs, f):
    y = np.asarray(y,float)
    if y.ndim != 2 or len(y)<4 or fs<=0 or not np.all(np.isfinite(y)):
        raise ValueError("Finite sample-by-channel array and positive sampling rate required")
    n = len(y)
    grid = np.fft.rfftfreq(n,1/fs)
    bins = np.rint(np.asarray(f)/(fs/n)).astype(int)
    if np.any(bins<=0) or np.any(bins>=len(grid)-1) or not np.allclose(grid[bins],f):
        raise ValueError("Fit grid must consist of interior record Fourier bins")
    coef = np.fft.rfft(y-y.mean(axis=0),axis=0)
    return 2*np.abs(coef[bins])**2/(fs*n)
