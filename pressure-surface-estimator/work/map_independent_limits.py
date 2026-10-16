"""Exploratory spatial/depth envelope of the unchanged pressure-only estimator.

Both tests reuse the four existing independent wave realizations. Calibration
uses pressure-only priors; native surface references are evaluator-only.
"""
from pathlib import Path
import sys, json, hashlib, time
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
ROOT=Path(__file__).resolve().parents[1]
SRC=Path(__file__).resolve().parents[1]/'src'
sys.path.insert(0,str(SRC))
from hybrid_pressure_live import calibrate_hybrid
from pressure_prior import PressurePrior, estimate_prior
from pressure_only_live import Calibration, PressureOnlyLive
from pressure_array import PressureArray
from submarine import ResponseTable, Submarine, Hull
from run_pressure_array_experiment import errors
from independent_wave_adapter import IndependentWave, motion_from_components, pressure_record

OLD=ROOT/'outputs/independent-wave-package'
frozen=json.loads((OLD/'protocol.json').read_text())['source_sha256']
source=ResponseTable.load(SRC/'results-joint/response_table.npz')
cfg=PressureArray.load(OLD/'assumed_geometry.json')
folders=sorted((OLD/'cases').iterdir())
xy0=Calibration.load(folders[0]/'calibration.npz').targets
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def check_frozen():
    assert all(sha(SRC/n)==h for n,h in frozen.items())

def histories(p,baseline):
    # Row t contains only p[t], p[t-1], ... p[t-127]. First scored t=1023.
    v=sliding_window_view(p-baseline,128,axis=0).transpose(0,2,1)[:,::-1,:]
    return np.ascontiguousarray(v[1023-127:].reshape(-1,768))

def apply(cal,p):
    h=histories(p,cal.baseline_Pa)
    result=h@cal.weights.reshape(len(cal.targets),-1).T
    # Meaningful regression against the actual per-packet causal implementation.
    live=PressureOnlyLive(cal)
    for j in range(1026):
        value=live.push(j*.5,p[j])
        if j>=1023:np.testing.assert_allclose(result[j-1023],value,atol=1e-10,rtol=1e-9)
    return result

def envelope(mode):
    out=ROOT/'outputs'/('wider-surface' if mode=='space' else 'independent-depth')
    out.mkdir(exist_ok=True)
    protocol={'test':mode,'source_sha256':frozen,'noise_std_Pa':0,
      'case_records':[{'case':f.name,'pressure_sha256':sha(f/'pressures.csv'),'waveinput_sha256':sha(f/'waveinput.dat')} for f in folders],
      'independence':'Native CFDwavemaker linear Airy wave source, existing shared known hull/instrument approximations; not nonlinear CFD or real measurements',
      'comparison':'Paired existing four realizations; exploratory envelope mapping, not fresh held-out sea validation',
      'inverse':'Unchanged 128-tap causal pressure-only FIR, ridge=1e-8, uniform directional mixture=.01, gamma fixed3.3, initial 1024 pressure packets only for prior learning',
      'score':'All7169 samples from packet1024 onward; current timestamp; no phase/time alignment or truth rescaling',
      'fs_Hz':2,'centre_depth_m':15,'reference_offset_s':400,
      'metrics':'R2=1-SSE/sum((truth-local_time_mean)^2); spatial pooling sums pointwise denominators, no per-frame normalization',
      'prespecified_depths_m':[15,30,45,60,80,100,150],
      'prespecified_centred_rectangles_m':[[60,12],[80,20],[100,40],[140,60],[200,100],[300,150]],
      'spatial_grid':'5m spacing over x[-150,150], y[-75,75]; original 403-point 60x12 grid scored separately',
      'depth_patch_m':[60,12],'depth_grid_points':403,
      'depth_calibration':'Nominal known hull/port depth supplied separately at each depth; prior relearned from that depth\u0027s pressures. Same sea phases/amplitudes across depths.',
      'interpretation':'No universal depth cutoff; frozen estimator and chosen seas only. Numerical regularization remains active even without sensor noise.'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    check_frozen();rows=[]
    if mode=='space':
        xx,yy=np.meshgrid(np.arange(-150,151,5),np.arange(-75,76,5))
        grid=np.c_[xx.ravel(),yy.ravel()]
        targets=np.unique(np.vstack([grid,xy0]),axis=0)
        ids={tuple(p):i for i,p in enumerate(targets)}
        original=np.array([ids[tuple(p)] for p in xy0]);gridids=np.array([ids[tuple(p)] for p in grid])
    for folder in folders:
        data=np.loadtxt(folder/'pressures.csv',delimiter=',',skiprows=1)
        t=data[:,0];p=data[:,1:]
        oldcal=Calibration.load(folder/'calibration.npz')
        old=dict(np.load(folder/'reference.npz')); old['estimate']=apply(oldcal,p)
        if mode=='space':
            print('START SPATIAL',folder.name,len(targets),'targets',flush=True);start=time.perf_counter()
            prior=PressurePrior(**oldcal.metadata['learned_prior'])
            cal=calibrate_hybrid(source,cfg,targets,prior)
            estimate=apply(cal,p)
            np.testing.assert_allclose(estimate[:,original],old['estimate'],atol=1e-9,rtol=1e-8)
            wave=IndependentWave(folder)
            truth=wave.surface(t[1023:]+400,targets);wave.close()
            np.testing.assert_allclose(truth[:,original],old['truth'],atol=2e-7) # compact float32 reference storage
            metrics,r2=errors(estimate,truth)
            rmse=np.sqrt(np.mean((estimate-truth)**2,axis=0))
            areas=[]
            for length,width in protocol['prespecified_centred_rectangles_m']:
                ix=original if (length,width)==(60,12) else gridids[(abs(grid[:,0])<=length/2)&(abs(grid[:,1])<=width/2)]
                metric,_=errors(estimate[:,ix],truth[:,ix])
                areas.append({'length_m':length,'width_m':width,'grid_points':len(ix),**metric})
            np.savez_compressed(out/(folder.name+'_map.npz'),xy=targets,R2=r2,RMSE_m=rmse,grid=grid,grid_indices=gridids,original_indices=original)
            row={'case':folder.name,'areas':areas,'regression_existing_predictions_and_truth_pass':True,'elapsed_s':time.perf_counter()-start}
            rows.append(row);print('RESULT',json.dumps(row),flush=True)
        else:
            for depth in protocol['prespecified_depths_m']:
                print('START DEPTH',folder.name,depth,flush=True);start=time.perf_counter()
                if depth==15:
                    metric,r2=errors(old['estimate'],old['truth']);learned=oldcal.metadata['learned_prior']
                    measured=p
                else:
                    # In this nominal linear hull model only excitation depends on depth.
                    scale=np.exp(-(source.omega**2/source.g)*(depth-source.depth))
                    model=ResponseTable(omega=source.omega,theta=source.theta,rao=source.rao*scale[:,None,None],depth=depth,reference=source.reference)
                    if depth==30:
                        direct=Submarine(omega=source.omega,theta=source.theta,hull=Hull(depth=depth))
                        np.testing.assert_allclose(model.rao,direct.rao,atol=1e-12,rtol=1e-10)
                    forward=Submarine(omega=source.omega,theta=source.theta,hull=Hull(depth=depth),build_response=False)
                    comp=np.load(folder/'components_evaluator_only.npz')
                    tt,X=motion_from_components(forward,comp['omega'],comp['amplitude'],comp['phase'],comp['theta'])
                    wave=IndependentWave(folder)
                    measured=pressure_record(wave,tt,X,cfg,depth=depth);wave.close()
                    prior=estimate_prior(measured[:1024],model,cfg)
                    cal=calibrate_hybrid(model,cfg,xy0,prior)
                    estimate=apply(cal,measured)
                    metric,r2=errors(estimate,old['truth']);learned=prior.to_dict()
                    np.savez_compressed(out/(folder.name+f'_depth{depth}_pressure.npz'),time_s=t,pressures_Pa=measured)
                    cal.save(out/(folder.name+f'_depth{depth}_calibration.npz'))
                row={'case':folder.name,'centre_depth_m':depth,**metric,'learned_prior':learned,
                     'pressure_dynamic_std_Pa':np.std(measured,axis=0).tolist(),'elapsed_s':time.perf_counter()-start}
                np.savez_compressed(out/(folder.name+f'_depth{depth}_map.npz'),xy=xy0,R2=r2)
                rows.append(row);print('RESULT',json.dumps(row),flush=True)
                (out/'results.json').write_text(json.dumps({'protocol':protocol,'runs':rows},indent=2)+'\n')
        (out/'results.json').write_text(json.dumps({'protocol':protocol,'runs':rows},indent=2)+'\n')
    check_frozen()
    assert all(sha(f/'pressures.csv')==spec['pressure_sha256'] for f,spec in zip(folders,protocol['case_records']))
    (out/'results.json').write_text(json.dumps({'protocol':protocol,'runs':rows,'frozen_method_and_original_pressure_integrity_pass':True},indent=2)+'\n')
    print('ALL INTEGRITY AND CAUSAL REGRESSIONS PASSED',mode,flush=True)

if __name__=='__main__':envelope(sys.argv[1])
