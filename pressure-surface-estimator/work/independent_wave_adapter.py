"""CFDwavemaker wave/pressure adapter. Contains no inverse implementation.

CFDwavemaker supplies elevations and incident pressures. Existing capsule
strip loads/matrices, RK4 dynamics and first-order moving-port instrument
model remain shared assumptions. No RAOs are used for forward motion.
"""
from pathlib import Path
import ctypes,os
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
LIB=ROOT/'work/CFDwavemaker/builds/macos/libCFDwavemaker.dylib'
BATCH=ROOT/'work/CFDwavemaker/builds/macos/libCFDbatch.dylib'

def upstream_components():
    path=ROOT/'work/CFDwavemaker/tests/t008_3D_irregular_sea/waveinput_linear.dat'
    rows=[]
    for line in path.read_text().splitlines():
        try:row=list(map(float,line.split()))
        except ValueError:continue
        if len(row)==5:rows.append(row)
    table=np.array(rows)
    assert table.shape==(312,5)
    return table,path

def write_case(path,kind,seed):
    table,upstream=upstream_components();w=table[:,0].copy();a=table[:,1].copy();phi=table[:,3].copy();theta=table[:,4].copy()+np.deg2rad(30)
    if kind=='gaussian_single_peak':
        f=w/(2*np.pi);s=np.exp(-.5*((f-.125)/.018)**2)
        a=np.sqrt(s)
    elif kind!='upstream_single_system':raise ValueError(kind)
    a*=2/(4*np.sqrt(np.sum(a*a)/2))
    if seed is not None:phi=np.random.default_rng(seed).uniform(0,2*np.pi,len(a))
    path.mkdir(parents=True,exist_ok=True)
    lines=['@v301','[wave type]','irregular','[general input data]','depth 1000','mtheta 0','rho 1025','gravity 9.81','normalize 0','amplify 1',
           '[wave input irregular]','kinematics_surf_extrap 0','nfreq 312','ndir 0','# OMEGA A PHASE THETA']
    lines.extend(' '.join(f'{v:.17g}' for v in row) for row in np.c_[w,a,phi,theta])
    (path/'waveinput.dat').write_text('\n'.join(lines)+'\n')
    np.savez(path/'components_evaluator_only.npz',omega=w,amplitude=a,phase=phi,theta=theta)
    return w,a,phi,theta

class IndependentWave:
    def __init__(self,folder):
        self.native=ctypes.CDLL(str(LIB))
        self.batch=ctypes.CDLL(str(BATCH))
        self.folder=Path(folder).resolve()
        self.native.wave_Initialize.restype=ctypes.c_int
        self.native.wave_Cleanup.restype=ctypes.c_int
        nd=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
        for name in ['cfd_batch_surface','cfd_batch_pressure']:
            fun=getattr(self.batch,name);fun.argtypes=[ctypes.c_int,nd,ctypes.c_int,nd,nd];fun.restype=None
        old=Path.cwd();os.chdir(self.folder)
        try:self.native.wave_Initialize()
        finally:os.chdir(old)

    def surface(self,t,xy):
        t=np.ascontiguousarray(t,dtype=float);xy=np.ascontiguousarray(xy,dtype=float)
        output=np.empty((len(t),len(xy)));self.batch.cfd_batch_surface(len(t),t,len(xy),xy,output)
        return output

    def pressure(self,t,xyz):
        t=np.ascontiguousarray(t,dtype=float);xyz=np.ascontiguousarray(xyz,dtype=float)
        output=np.empty((len(t),len(xyz)));self.batch.cfd_batch_pressure(len(t),t,len(xyz),xyz,output)
        return output

    def close(self):self.native.wave_Cleanup()

def component_excitation(hull,omega,theta):
    """Same primitive capsule strip force law, evaluated at arbitrary directions."""
    h=hull.hull;dx=h.length/h.strips
    x=-h.length/2+(np.arange(h.strips)+.5)*dx
    cap=np.maximum(abs(x)-(h.length/2-h.radius),0)
    dV=np.pi*(h.radius**2-cap**2)*dx
    axial_ca=.5*(4*np.pi*h.radius**3/3)/hull.volume
    ct=np.cos(theta);st=np.sin(theta);k=omega**2/hull.g
    basis=np.exp(1j*k[:,None]*ct[:,None]*x)
    v0=basis@dV;v1=basis@(dV*x)
    factor=hull.rho*omega**2*np.exp(-k*h.depth)
    force=np.zeros((len(omega),6),complex)
    force[:,0]=-1j*factor*(1+axial_ca)*ct*v0
    force[:,1]=-1j*factor*2*st*v0
    force[:,2]=-factor*2*v0
    force[:,4]=factor*2*v1
    force[:,5]=-1j*factor*2*st*v1
    return force

def motion_from_components(hull,omega,amplitude,phase,theta,*,fs=2,n_samples=8192,dt=.1,settle=400):
    force_amp=component_excitation(hull,omega,theta)*(amplitude*np.exp(1j*phase))[:,None]
    total_steps=round((settle+n_samples/fs)/dt)
    half_times=np.arange(2*total_steps+1)*dt/2
    forcing=np.empty((len(half_times),6))
    for start in range(0,len(half_times),2048):
        times=half_times[start:start+2048]
        forcing[start:start+2048]=(np.exp(-1j*times[:,None]*omega)@force_amp).real
    ramp=np.where(half_times<60,.5*(1-np.cos(np.pi*np.minimum(half_times,60)/60)),1)
    forcing*=ramp[:,None]
    mass=hull.M+hull.A[0];im=np.linalg.inv(mass)
    system=np.block([[np.zeros((6,6)),np.eye(6)],[-im@hull.C_hydro,-im@hull.B[0]]])
    drive=np.c_[np.zeros((len(forcing),6)),forcing@im.T]
    state=np.zeros(12);X=np.empty((n_samples,12));out=0
    for step in range(total_steps):
        i=2*step
        if step>=round(settle/dt) and (step-round(settle/dt))%round(1/(fs*dt))==0:
            X[out]=state;out+=1
        k1=system@state+drive[i]
        k2=system@(state+dt/2*k1)+drive[i+1]
        k3=system@(state+dt/2*k2)+drive[i+1]
        k4=system@(state+dt*k3)+drive[i+2]
        state+=dt/6*(k1+2*k2+2*k3+k4)
    assert out==n_samples
    return settle+np.arange(n_samples)/fs,X

def pressure_record(package,t,X,config,*,depth=15,rho=1025,g=9.81):
    r=np.asarray(config.positions);xyz=r.copy();xyz[:,2]-=depth
    incident=package.pressure(t,xyz)
    port_z=X[:,2,None]+X[:,3,None]*r[None,:,1]-X[:,4,None]*r[None,:,0]
    return rho*g*(depth-r[None,:,2]-port_z)+incident
