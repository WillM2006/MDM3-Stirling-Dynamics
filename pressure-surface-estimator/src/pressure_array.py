"""Six configurable pressure ports plus the existing raw IMU.

Sponsor count: six (user confirmation, 14 October 2026).
Actual mounting coordinates: unconfirmed. Layouts must be selected explicitly.
First-order incident-wave/hydrostatic sensor physics; hull-scattered pressure
and measurement noise remain outside this prototype.
"""
from dataclasses import dataclass
import json
import numpy as np
from sensors import Sensors,CHANNELS,sensor_response
from submarine import skew
from time_domain import simulate,_periodic_input


@dataclass(frozen=True)
class PressureArray:
    positions: tuple
    placement_status: str
    imu_position: tuple = (10.,0.,0.)
    names: tuple = ('p1','p2','p3','p4','p5','p6')

    def __post_init__(self):
        p=np.asarray(self.positions,float)
        if p.shape!=(6,3) or not np.all(np.isfinite(p)) or len(np.unique(p,axis=0))!=6:
            raise ValueError('Six distinct finite body-frame xyz coordinates are required')
        if len(self.names)!=6 or len(set(self.names))!=6 or not all(isinstance(n,str) and n for n in self.names):
            raise ValueError('Six unique sensor names are required')
        if np.shape(self.imu_position)!=(3,) or not np.all(np.isfinite(self.imu_position)):
            raise ValueError('Finite IMU xyz position required')
        if not isinstance(self.placement_status,str) or not self.placement_status.strip():
            raise ValueError('Placement provenance/status must be explicit')

    @property
    def channels(self):
        return tuple('pressure_'+name+'_Pa' for name in self.names)+CHANNELS[1:]

    def save(self,path):
        with open(path,'w') as out:
            json.dump({'positions_m':self.positions,'names':self.names,'imu_position_m':self.imu_position,
                       'placement_status':self.placement_status,'coordinate_convention':'+x along hull, +z up, origin geometric hull centre'},out,indent=2)

    @classmethod
    def load(cls,path):
        with open(path) as f:d=json.load(f)
        return cls(tuple(tuple(p) for p in d['positions_m']),d['placement_status'],tuple(d['imu_position_m']),tuple(d['names']))


def array_response(model,omega,theta,config):
    """[frequency,direction,12]: six pressures, 3 accelerometers, 3 gyros."""
    r=np.asarray(config.positions,float);depths=model.depth-r[:,2]
    if np.any(depths<=0):raise ValueError('Every mean pressure port must be submerged')
    q=model.motion_response(omega,theta);w=np.asarray(omega);k=w*w/model.g
    spatial=k[:,None,None]*(np.cos(theta)[None,:,None]*r[None,None,:,0]+np.sin(theta)[None,:,None]*r[None,None,:,1])
    incident=model.rho*model.g*np.exp(-k[:,None,None]*depths[None,None,:]+1j*spatial)
    port_z=q[:,:,2,None]+q[:,:,3,None]*r[None,None,:,1]-q[:,:,4,None]*r[None,None,:,0]
    pressure=incident-model.rho*model.g*port_z
    imu=sensor_response(model,omega,theta,Sensors(config.imu_position,config.positions[0]))[:,:,1:]
    return np.concatenate([pressure,imu],axis=2)


def simulate_array(sea,hull,config,**kwargs):
    """Coupled RK4 motion, then independent six-port TIME instrument equations.

    Does not use motion RAOs or array_response to generate pressure signals.
    True states returned separately for evaluation; not estimator inputs.
    """
    r=np.asarray(config.positions,float);depths=hull.depth-r[:,2]
    if np.any(depths<=0):raise ValueError('Every mean pressure port must be submerged')
    t,old,X,acceleration=simulate(sea,hull,config=Sensors(config.imu_position,config.positions[0]),**kwargs)
    fs=1/(t[1]-t[0]);n=len(t)
    spatial=sea.k[:,None,None]*(np.cos(sea.theta)[None,:,None]*r[None,None,:,0]+np.sin(sea.theta)[None,:,None]*r[None,None,:,1])
    pressure_amplitudes=hull.rho*hull.g*np.einsum('fd,fdp->fp',sea.complex_components,np.exp(-sea.k[:,None,None]*depths[None,None,:]+1j*spatial))
    pressure_amplitudes*=np.exp(-1j*sea.omega*t[0])[:,None]
    incident=_periodic_input(pressure_amplitudes,sea.f,fs,n)
    port_z=X[:,2,None]+X[:,3,None]*r[None,:,1]-X[:,4,None]*r[None,:,0]
    pressure=hull.rho*hull.g*(depths[None,:]-port_z)+incident
    return t,np.concatenate([pressure,old[:,1:]],axis=1),X,acceleration


def reconstruct_above_ports(time,signals,config,*,depth,rho=1025.,g=9.81,band=(.05,.30)):
    """Local zero-mean surface elevations at each port's horizontal location.

    Uses only the recorded 12 channels and nominal geometry/pressure physics.
    Angular-rate and acceleration phases correct motion at every port; no
    fitted sea shape, directional weights, generating phases or states enter.
    """
    time=np.asarray(time,float);y=np.asarray(signals,float);r=np.asarray(config.positions,float)
    if y.shape!=(len(time),12) or len(time)<16 or not np.all(np.isfinite(y)):
        raise ValueError('Finite samples x 12 array required')
    intervals=np.diff(time)
    if np.any(intervals<=0) or not np.allclose(intervals,intervals[0]):raise ValueError('Uniform increasing timestamps required')
    depths=depth-r[:,2]
    if np.any(depths<=0) or not 0<band[0]<band[1] or rho<=0 or g<=0:raise ValueError('Submerged ports and positive physics/band required')
    fs=1/intervals[0];n=len(time);f=np.fft.rfftfreq(n,1/fs)
    use=(f>=band[0])&(f<=band[1])&(f>0)&(f<fs/2);w=2*np.pi*f[use]
    if not use.any():raise ValueError('No supported Fourier frequencies')
    Y=np.fft.rfft(y-y.mean(axis=0),axis=0);roll=Y[use,9]/(1j*w);pitch=Y[use,10]/(1j*w)
    displacement=-Y[use,8,None]/w[:,None]**2
    displacement=displacement+roll[:,None]*(r[None,:,1]-config.imu_position[1])-pitch[:,None]*(r[None,:,0]-config.imu_position[0])
    coef=np.zeros((len(f),6),complex)
    coef[use]=np.exp(w[:,None]**2/g*depths[None,:])*(Y[use,:6]/(rho*g)+displacement)
    return np.fft.irfft(coef,n=n,axis=0)


def symmetric_six_port_layout(length=70.,radius=5.,imu_position=(10.,0.,0.)):
    """User-authorized modelling assumption, not verified sponsor hardware.

    Four horizontal extremities plus top/bottom at mid-length. Surface ports
    occupy +/- length/2 along x and +/- radius on y and z for the capsule.
    """
    if not np.isfinite(length) or not np.isfinite(radius) or not length>2*radius>0:
        raise ValueError('Finite positive capsule dimensions required')
    return PressureArray(((length/2,0.,0.),(-length/2,0.,0.),(0.,radius,0.),
                          (0.,-radius,0.),(0.,0.,radius),(0.,0.,-radius)),
                         'User-authorized symmetric modelling assumption; six-sensor count confirmed, actual hardware coordinates unverified',
                         tuple(imu_position),('bow','stern','port','starboard','top','bottom'))


def wave_interpolate(time,local_elevations,positions,targets,*,g=9.81,band=(.05,.30),relative_eigen_floor=1e-8):
    """Phase-preserving spatial interpolation with a dispersion-based kernel.

    Uses deep-water k=omega^2/g and isotropic angular covariance J0(k*r).
    Equal directional prior and minimum-norm regularization are assumptions,
    not proof of a unique wave field. No JONSWAP frequency shape or sea truth.
    Duplicate horizontal points (top/bottom) are averaged explicitly.
    """
    from scipy.special import j0
    time=np.asarray(time,float);eta=np.asarray(local_elevations,float)
    xy=np.asarray(positions,float)[:,:2];targets=np.asarray(targets,float)
    if eta.shape!=(len(time),len(xy)) or targets.ndim!=2 or targets.shape[1]!=2 or not np.all(np.isfinite(eta)) or not np.all(np.isfinite(targets)):
        raise ValueError('Finite sample x port elevations and target xy coordinates required')
    if not 0<relative_eigen_floor<1 or g<=0:raise ValueError('Positive gravity and relative eigen floor required')
    dt=np.diff(time)
    if np.any(dt<=0) or not np.allclose(dt,dt[0]):raise ValueError('Uniform increasing timestamps required')
    unique,labels=np.unique(xy,axis=0,return_inverse=True)
    merged=np.column_stack([eta[:,labels==j].mean(axis=1) for j in range(len(unique))])
    n=len(time);f=np.fft.rfftfreq(n,dt[0]);use=(f>=band[0])&(f<=band[1])&(f>0)&(f<1/(2*dt[0]));k=(2*np.pi*f[use])**2/g
    distance=np.linalg.norm(unique[:,None,:]-unique[None,:,:],axis=2)
    target_distance=np.linalg.norm(targets[:,None,:]-unique[None,:,:],axis=2)
    K=j0(k[:,None,None]*distance[None,:,:]);ev,U=np.linalg.eigh(K)
    inv=np.zeros_like(ev);np.divide(1.,ev,out=inv,where=ev>relative_eigen_floor*ev[:,-1:])
    inverse=np.einsum('fij,fj,fkj->fik',U,inv,U)
    field=j0(k[:,None,None]*target_distance[None,:,:])
    weights=np.einsum('ftp,fpq->ftq',field,inverse)
    coef=np.zeros((len(f),len(targets)),complex);z=np.fft.rfft(merged-merged.mean(axis=0),axis=0)
    coef[use]=np.einsum('ftp,fp->ft',weights,z[use])
    return np.fft.irfft(coef,n=n,axis=0)


def reconstruct_surface_joint(time,signals,config,model,targets,*,band=(.05,.30),rank_tolerance=1e-9):
    """Direct complex-array inverse over a target patch, retaining wave phase.

    Nominal six-DOF/sensor response is known. Directional Fourier amplitudes
    have an isotropic equal-energy prior at each frequency. A pseudoinverse
    handles response redundancy separately at every frequency. No JONSWAP
    frequency shape, generating phases, states or true sea parameters enter.
    Unobserved modes are set by a minimum-norm prior, not uniquely recovered.
    """
    time=np.asarray(time,float);y=np.asarray(signals,float);xy=np.asarray(targets,float)
    if y.shape!=(len(time),12) or xy.ndim!=2 or xy.shape[1]!=2 or not np.all(np.isfinite(y)) or not np.all(np.isfinite(xy)):
        raise ValueError('Finite samples x 12 recordings and target xy required')
    dt=np.diff(time)
    if np.any(dt<=0) or not np.allclose(dt,dt[0]) or not 0<rank_tolerance<1:raise ValueError('Uniform increasing time and relative rank tolerance required')
    n=len(time);grid=np.fft.rfftfreq(n,dt[0]);use=(grid>=max(band[0],model.omega[0]/(2*np.pi)))&(grid<=min(band[1],model.omega[-1]/(2*np.pi)))&(grid>0)&(grid<1/(2*dt[0]))
    f=grid[use];sample_model=model.on_grid(2*np.pi*f);H=array_response(sample_model,2*np.pi*f,sample_model.theta,config)
    scale=np.sqrt(np.mean(abs(H)**2,axis=1));scale=np.where(scale>0,scale,1.)
    A=H/scale[:,None,:];K=np.einsum('fdc,fde->fce',A,A.conj())/len(sample_model.theta)
    ev,U=np.linalg.eigh(K);keep=ev>rank_tolerance*ev[:,-1:];inverse_ev=np.zeros_like(ev);np.divide(1.,ev,out=inverse_ev,where=keep)
    # Use only recorded Fourier coefficients, preserving their common phase.
    physical=np.fft.rfft(y-y.mean(axis=0),axis=0)[use].conj()/scale
    coefficients=np.einsum('fci,fc->fi',U.conj(),physical)*inverse_ev
    dual=np.einsum('fci,fi->fc',U,coefficients)
    latent=np.einsum('fdc,fc->fd',A.conj(),dual)/len(sample_model.theta)
    field=np.empty((len(f),len(xy)),complex);k=(2*np.pi*f)**2/model.g
    for start in range(0,len(f),48):
        sl=slice(start,start+48)
        spatial=k[sl,None,None]*(np.cos(sample_model.theta)[None,:,None]*xy[None,None,:,0]+np.sin(sample_model.theta)[None,:,None]*xy[None,None,:,1])
        field[sl]=np.einsum('fd,fdp->fp',latent[sl],np.exp(1j*spatial))
    out=np.zeros((len(grid),len(xy)),complex);out[use]=field.conj()
    return np.fft.irfft(out,n=n,axis=0),{'rank_min':int(keep.sum(axis=1).min()),'rank_max':int(keep.sum(axis=1).max()),'rank_tolerance':rank_tolerance,'prior':'equal directional energy at each frequency; no JONSWAP frequency model'}
