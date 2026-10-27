from pathlib import Path
import sys,json,time
import numpy as np
SRC=Path(__file__).resolve().parents[1]/'src';sys.path.insert(0,str(SRC))
from submarine import Submarine
from independent_wave_adapter import IndependentWave,component_excitation,write_case
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/independent-wave-package';OUT.mkdir(exist_ok=True)
folder=ROOT/'work/cfd-adapter-check';folder.mkdir(exist_ok=True)
w=np.array([2*np.pi*.125,2*np.pi*.18]);a=np.array([.5,.2]);phase=np.array([.3,-.7]);theta=np.array([.52,-.42])
lines=['@v301','[wave type]','irregular','[general input data]','depth 1000','rho 1025','gravity 9.81','normalize 0','amplify 1','[wave input irregular]','kinematics_surf_extrap 0','nfreq 2','ndir 0']
lines+=[' '.join(f'{x:.17g}' for x in row) for row in np.c_[w,a,phase,theta]]
(folder/'waveinput.dat').write_text('\n'.join(lines)+'\n')
wave=IndependentWave(folder);t=np.array([10.,10.5,11.,12.25]);xy=np.array([[0,0],[12,-3],[-20,4.]])
k=w*w/9.81;arg=k[None,:,None]*(np.cos(theta)[None,:,None]*xy[None,None,:,0]+np.sin(theta)[None,:,None]*xy[None,None,:,1])-w[None,:,None]*t[:,None,None]+phase[None,:,None]
expected=np.einsum('f,tfp->tp',a,np.cos(arg));actual=wave.surface(t,xy)
np.testing.assert_allclose(actual,expected,atol=1e-11)
xyz=np.c_[xy,np.array([-10.,-15.,-20.])]
pressure=wave.pressure(t,xyz)
expectedp=1025*9.81*np.einsum('f,fp,tfp->tp',a,np.exp(k[:,None]*xyz[None,:,2]),np.cos(arg))
np.testing.assert_allclose(pressure,expectedp,atol=1e-7)
wave.close()
hull=Submarine(omega=w,theta=theta,build_response=False)
expected_force=np.array([hull.F_exc[i,i] for i in range(len(w))])
got=component_excitation(hull,w,theta)
np.testing.assert_allclose(got,expected_force,rtol=1e-12,atol=1e-7)
result={'package_surface_two_tone_phase_direction_convention':'passed',
        'package_pressure_density_units_depth_attenuation':'passed',
        'arbitrary_component_force_matches_existing_strip_law':'passed',
        'maximum_surface_discrepancy_m':float(abs(actual-expected).max()),
        'maximum_pressure_discrepancy_Pa':float(abs(pressure-expectedp).max())}
(OUT/'adapter-checks.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
