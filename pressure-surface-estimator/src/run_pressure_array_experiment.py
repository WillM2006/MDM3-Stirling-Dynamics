"""Six pressure ports + raw IMU: local wave recovery and spatial interpolation.

Explicitly assumed symmetric geometry. Existing hull and sensor physics stay
unchanged. No JONSWAP parameters or truth states enter the inverse. Linear
interpolation is scored against hidden surface truth at unsensed locations.
"""
import argparse,json,hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.spatial import Delaunay
from sea import Sea,FixedSea
from submarine import Submarine,ResponseTable
from pressure_array import symmetric_six_port_layout,simulate_array,reconstruct_above_ports,wave_interpolate,reconstruct_surface_joint

HERE=Path(__file__).resolve().parent


def make_sea(table,kind,seed):
    f=table.omega/(2*np.pi)
    if kind=='single_system':return Sea(freq=f,theta=table.theta,Hs=2.,Tp=8.,fixed=FixedSea(mean_direction=np.deg2rad(30),spread_s=10),seed=seed)
    a=Sea(freq=f,theta=table.theta,Hs=np.sqrt(2),Tp=6.5,fixed=FixedSea(mean_direction=np.deg2rad(20),spread_s=8),seed=seed)
    b=Sea(freq=f,theta=table.theta,Hs=np.sqrt(2),Tp=10.5,fixed=FixedSea(mean_direction=np.deg2rad(-70),spread_s=18),seed=seed+901)
    return SimpleNamespace(f=a.f,omega=a.omega,k=a.k,theta=a.theta,delta_f=a.delta_f,complex_components=a.complex_components+b.complex_components)


def truth_at(sea,xy,time,band):
    n=len(time);fs=1/(time[1]-time[0]);nf=n//2+1;xy=np.asarray(xy)
    z=np.empty((len(sea.f),len(xy)),complex)
    for start in range(0,len(sea.f),48):
        sl=slice(start,start+48)
        phase=sea.k[sl,None,None]*(np.cos(sea.theta)[None,:,None]*xy[None,None,:,0]+np.sin(sea.theta)[None,:,None]*xy[None,None,:,1])
        z[sl]=np.einsum('fd,fdp->fp',sea.complex_components[sl],np.exp(1j*phase))
    z*=np.exp(-1j*sea.omega*time[0])[:,None]
    z[(sea.f<band[0])|(sea.f>band[1])]=0
    coeff=np.zeros((nf,len(xy)),complex);coeff[np.rint(sea.f*n/fs).astype(int)]=n/2*z.conj()
    return np.fft.irfft(coeff,n=n,axis=0)


def errors(estimate,truth):
    delta=estimate-truth;mse=np.mean(delta**2,axis=0);variance=np.var(truth,axis=0)
    r2=1-mse/variance
    return {'pooled_R2':float(1-np.sum(delta**2)/np.sum((truth-truth.mean(axis=0))**2)),
            'minimum_point_R2':float(np.min(r2)),'median_point_R2':float(np.median(r2)),
            'pooled_RMSE_m':float(np.sqrt(np.mean(delta**2))),'maximum_point_RMSE_m':float(np.sqrt(mse.max()))},r2


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    cfg=symmetric_six_port_layout();cfg.save(out/'assumed_geometry.json')
    table=ResponseTable.load(HERE/'results-joint/response_table.npz');hull=Submarine(omega=table.omega,theta=table.theta,build_response=False)
    band=(.05,.30);ports=np.asarray(cfg.positions);xy=ports[:,:2]
    unique_xy,inverse=np.unique(xy,axis=0,return_inverse=True)
    # Two vertically aligned ports give two records at ONE horizontal location.
    combine=np.zeros((6,len(unique_xy)))
    for j in range(len(unique_xy)):
        members=np.flatnonzero(inverse==j);combine[members,j]=1/len(members)
    xx,yy=np.meshgrid(np.linspace(-35,35,43),np.linspace(-5,5,23));grid=np.column_stack([xx.ravel(),yy.ravel()])
    tri=Delaunay(unique_xy);simplex=tri.find_simplex(grid);inside=simplex>=0;targets=grid[inside]
    transforms=tri.transform[simplex[inside]];bary=np.einsum('pij,pj->pi',transforms[:,:2,:],targets-transforms[:,2,:]);bary=np.column_stack([bary,1-bary.sum(axis=1)])
    W=np.zeros((len(targets),len(unique_xy)))
    W[np.arange(len(targets))[:,None],tri.simplices[simplex[inside]]]=bary
    assert np.allclose(W.sum(axis=1),1) and np.allclose(W@unique_xy,targets)
    protocol={'sensor_count':6,'layout_status':cfg.placement_status,'imu_channels_retained':6,'phase_seeds':[42,81,2026],
              'seas':['single_system','crossing_two_systems'],'pressure_noise_std_Pa':[0.,1.,10.],
              'noise_status':'Hypothetical independent white pressure noise for sensitivity only; not sensor specification. IMU remains ideal.',
              'band_Hz':band,'surface_domain':'Within the convex hull of five unique horizontal sensor locations; no extrapolation',
              'interpolation':'Piecewise linear over Delaunay triangles; no spectral shape or directional prior',
              'joint_surface_inverse':'Six pressures plus raw IMU, known nominal complex response, isotropic equal-direction prior, per-frequency rank tolerance 1e-9; no JONSWAP spectrum',
              'wave_interpolation':'Deep-water dispersion covariance J0(k*r); isotropic angular prior, relative eigen cutoff 1e-8; no JONSWAP frequency shape or fitted sea parameters',
              'source_sha256':{name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in ['pressure_array.py','time_domain.py','submarine.py']}}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    records=out/'records';records.mkdir(exist_ok=True);runs=[]
    for kind in protocol['seas']:
        for seed in protocol['phase_seeds']:
            sea=make_sea(table,kind,seed);label=f'{kind}_seed{seed}'
            t,y,X,_=simulate_array(sea,hull,cfg,fs=2.,n_samples=8192,dt=.1,settle_seconds=400.)
            np.savetxt(records/(label+'.csv'),np.column_stack([t,y]),delimiter=',',header='time_s,'+','.join(cfg.channels),comments='')
            # Truth calculated separately, never supplied to the inverse.
            port_truth=truth_at(sea,xy,t,band);surface_truth=truth_at(sea,targets,t,band)
            peak=float(np.rad2deg(np.max(abs(X[:,3:6]))))
            for noise in protocol['pressure_noise_std_Pa']:
                measured=y.copy()
                if noise:
                    measured[:,:6]+=np.random.default_rng(seed+round(noise)*103).normal(0,noise,(len(t),6))
                local=reconstruct_above_ports(t,measured,cfg,depth=hull.depth,rho=hull.rho,g=hull.g,band=band)
                surface=(local@combine)@W.T
                wave_surface=wave_interpolate(t,local,cfg.positions,targets,g=hull.g,band=band)
                joint_surface,joint_diagnostics=reconstruct_surface_joint(t,measured,cfg,table,targets,band=band)
                local_metrics,_=errors(local,port_truth);spatial_metrics,r2map=errors(surface,surface_truth)
                wave_metrics,wave_r2map=errors(wave_surface,surface_truth)
                joint_metrics,joint_r2map=errors(joint_surface,surface_truth)
                runs.append({'sea':kind,'seed':seed,'noise_std_Pa':noise,'local_ports':local_metrics,'interpolated_surface':spatial_metrics,'wave_interpolated_surface':wave_metrics,'joint_surface':joint_metrics,'joint_diagnostics':joint_diagnostics,'peak_rotation_deg':peak,'small_angle_screen_pass':peak<5})
                print(f'{label}, noise={noise:g} Pa: local R2={local_metrics["pooled_R2"]:.6f}, interpolated R2={spatial_metrics["pooled_R2"]:.4f}, wave-based R2={wave_metrics["pooled_R2"]:.4f}, joint R2={joint_metrics["pooled_R2"]:.4f}, spatial RMSE={spatial_metrics["pooled_RMSE_m"]:.3f} m',flush=True)
                np.savez_compressed(out/(label+f'_noise{noise:g}_map.npz'),x=targets[:,0],y=targets[:,1],R2=r2map,wave_R2=wave_r2map,joint_R2=joint_r2map)
                if seed==42 and noise==0:
                    nearest=np.argmin(np.sum((targets-np.array([17.,2.]))**2,axis=1));window=(t>=500)&(t<=560)
                    np.savez(out/(label+'_example.npz'),time_s=t[window]-500,truth=surface_truth[window,nearest],interpolated=surface[window,nearest],wave_interpolated=wave_surface[window,nearest],joint=joint_surface[window,nearest],location_m=targets[nearest])
            (out/'runs.json').write_text(json.dumps(runs,indent=2)+'\n')
    summary=[]
    for kind in protocol['seas']:
        for noise in protocol['pressure_noise_std_Pa']:
            rs=[r for r in runs if r['sea']==kind and r['noise_std_Pa']==noise]
            summary.append({'sea':kind,'noise_std_Pa':noise,'records':len(rs),
                            'minimum_local_pooled_R2':min(r['local_ports']['pooled_R2'] for r in rs),
                            'minimum_interpolated_pooled_R2':min(r['interpolated_surface']['pooled_R2'] for r in rs),
                            'maximum_interpolated_pooled_RMSE_m':max(r['interpolated_surface']['pooled_RMSE_m'] for r in rs),
                            'minimum_wave_based_pooled_R2':min(r['wave_interpolated_surface']['pooled_R2'] for r in rs),
                            'maximum_wave_based_pooled_RMSE_m':max(r['wave_interpolated_surface']['pooled_RMSE_m'] for r in rs),
                            'minimum_joint_pooled_R2':min(r['joint_surface']['pooled_R2'] for r in rs),
                            'maximum_joint_pooled_RMSE_m':max(r['joint_surface']['pooled_RMSE_m'] for r in rs),
                            'minimum_unsensed_point_R2':min(r['interpolated_surface']['minimum_point_R2'] for r in rs)})
    report={'protocol':protocol,'geometry':{'positions_m':cfg.positions,'names':cfg.names,'unique_horizontal_locations':len(unique_xy),'interior_grid_points':len(targets)},
            'summary':summary,'runs':runs,
            'limits':['Assumed geometry; not verified sponsor sensor positions','Same primitive linear capsule/ideal incident-pressure physics; no hull scattering, speed or calibration error','No JONSWAP parameters, phases, sea truth or integrated states enter the reconstruction','Time histories are reconstructed within a finite band, not future-wave prediction','Six ports provide five unique horizontal surface locations in this assumed layout','Pressure noise levels hypothetical; IMU noiseless; limited to six phase records','Linear interpolation uses no wave dispersion constraint and is not exact between ports','Wave-based interpolation uses an isotropic angular prior; it supplies unmeasured spatial structure and does not establish physical uniqueness']}
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
