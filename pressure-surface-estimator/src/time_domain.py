"""Separate time-domain six-DOF ODE and ideal-instrument generator.

NEVER uses RAOs, sensor_response(), spectral_gain(), or the inverse estimator.
Wave forces are known inputs; the ODE is integrated from rest with a ramp.
Shared primitive hull coefficients are still assumed physics, not independent
real-world validation. Sensors use the same first-order physical assumptions.
"""
import numpy as np
from sensors import Sensors
from submarine import skew


def _periodic_input(amplitudes, frequencies, fs, samples):
    """Sample the prescribed wave INPUT sum efficiently, not vehicle response.

    Equivalent to sum Re[A_f exp(-i 2pi f t)]. FFT is only an evaluation of
    prescribed sinusoidal forcing; no forward response division takes place.
    """
    bins = np.rint(frequencies/(fs/samples)).astype(int)
    grid = np.fft.rfftfreq(samples,1/fs)
    if np.any(bins<=0) or np.any(bins>=len(grid)-1) or not np.allclose(grid[bins],frequencies):
        raise ValueError('Wave inputs must lie on interior Fourier bins')
    coef = np.zeros((len(grid),amplitudes.shape[1]),complex)
    coef[bins] = samples/2*np.conj(amplitudes)
    return np.fft.irfft(coef,n=samples,axis=0)


def simulate(sea, hull, *, fs=2., n_samples=8192, dt=.1,
             settle_seconds=400., ramp_seconds=60., config=None):
    """Return instrument time series and separate state diagnostics.

    X=[q,velocity], as in WM's time-domain model; coupled RK4, zero initial
    state. No steady-state amplitudes/initial conditions are imported.
    Inputs are sampled at RK4 half steps. Decimate to the ideal sensor rate;
    no noise or filters. Default band is below sensor Nyquist.
    """
    config = config or Sensors()
    if dt<=0 or fs<=0 or n_samples<4 or settle_seconds<ramp_seconds or ramp_seconds<=0:
        raise ValueError('Positive rates/ramp and sufficient settling required')
    sample_step = int(round(1/(fs*dt)))
    settle_steps = int(round(settle_seconds/dt))
    if sample_step<1 or not np.isclose(sample_step*dt,1/fs) or not np.isclose(settle_steps*dt,settle_seconds):
        raise ValueError('Integration and output times must align')
    if sea.f.max()>=fs/2 or not np.isclose(sea.delta_f,fs/n_samples):
        raise ValueError('Wave grid must match record resolution and lie below sensor Nyquist')
    if not np.allclose(hull.A,hull.A[0]) or not np.allclose(hull.B,hull.B[0]):
        raise ValueError('This baseline ODE uses constant added mass and damping; memory needs a team solver')
    total_record_steps=n_samples*sample_step
    half_samples=2*total_record_steps
    forcing_amplitudes=np.einsum('fdk,fd->fk',hull.F_exc,sea.complex_components)
    forcing=_periodic_input(forcing_amplitudes,sea.f,2/dt,half_samples)
    r_p=np.asarray(config.pressure_position)
    if hull.depth-r_p[2]<=0:
        raise ValueError('Mean pressure port must be submerged')
    phase=sea.k[:,None]*(r_p[0]*np.cos(sea.theta)[None,:]+r_p[1]*np.sin(sea.theta)[None,:])
    pressure_per_amplitude=hull.rho*hull.g*np.exp(-sea.k[:,None]*(hull.depth-r_p[2]))*np.exp(1j*phase)
    pressure_amplitudes=np.sum(pressure_per_amplitude*sea.complex_components,axis=1)[:,None]
    pressure_input=_periodic_input(pressure_amplitudes,sea.f,2/dt,half_samples)[:,0]
    mass=hull.M+hull.A[0]
    damping,restoring=hull.B[0],hull.C_hydro
    inv_mass=np.linalg.inv(mass)
    system=np.block([[np.zeros((6,6)),np.eye(6)],[-inv_mass@restoring,-inv_mass@damping]])
    t_half=np.arange(2*(settle_steps+total_record_steps)+1)*dt/2
    ramp=np.where(t_half<ramp_seconds,.5*(1-np.cos(np.pi*np.minimum(t_half,ramp_seconds)/ramp_seconds)),1.)
    X=np.zeros(12)
    states=np.zeros((n_samples,12))
    accelerations=np.zeros((n_samples,6))
    pressures=np.zeros(n_samples)
    out=0
    for step in range(settle_steps+total_record_steps):
        h=2*step
        force0=ramp[h]*forcing[h%half_samples]
        derivative0=system@X+np.r_[np.zeros(6),inv_mass@force0]
        if step>=settle_steps and (step-settle_steps)%sample_step==0:
            states[out]=X
            accelerations[out]=derivative0[6:]
            pressures[out]=ramp[h]*pressure_input[h%half_samples]
            out+=1
        half_force=ramp[h+1]*forcing[(h+1)%half_samples]
        end_force=ramp[h+2]*forcing[(h+2)%half_samples]
        add_half=np.r_[np.zeros(6),inv_mass@half_force]
        add_end=np.r_[np.zeros(6),inv_mass@end_force]
        k1=derivative0
        k2=system@(X+dt/2*k1)+add_half
        k3=system@(X+dt/2*k2)+add_half
        k4=system@(X+dt*k3)+add_end
        X+=dt/6*(k1+2*k2+2*k3+k4)
    if out!=n_samples:
        raise RuntimeError('Sensor record length mismatch')
    q,v=states[:,:6],states[:,6:]
    r_imu=np.asarray(config.imu_position)
    # First-order instrument equations in TIME, independent implementation of
    # the spectral sensor_response map. Acceleration comes from Newton/ODE.
    acc=accelerations[:,:3]-accelerations[:,3:]@skew(r_imu).T
    acc[:,0]-=hull.g*q[:,4]
    acc[:,1]+=hull.g*q[:,3]
    acc[:,2]+=hull.g
    port_displacement=q[:,:3]-q[:,3:]@skew(r_p).T
    pressure=hull.rho*hull.g*(hull.depth-r_p[2]-port_displacement[:,2])+pressures
    y=np.column_stack([pressure,acc,v[:,3:]])
    t=settle_seconds+np.arange(n_samples)/fs
    return t,y,states,accelerations
