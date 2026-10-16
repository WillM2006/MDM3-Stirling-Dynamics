"""Frozen-estimator validation against native CFDwavemaker sea/pressures."""
from pathlib import Path
import sys,json,time,hashlib,subprocess,shutil
import numpy as np
SRC=Path(__file__).resolve().parents[1]/'src';sys.path.insert(0,str(SRC))
from hybrid_pressure_live import HybridPressureLive
from pressure_only_live import calibrate,PressureOnlyLive
from pressure_array import PressureArray
from submarine import ResponseTable,Submarine
from run_pressure_array_experiment import errors
from independent_wave_adapter import IndependentWave,write_case,upstream_components,motion_from_components,pressure_record,LIB,BATCH
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/independent-wave-package';OUT.mkdir(exist_ok=True)
PRIOR=ROOT/'outputs/hybrid-pressure';old_protocol=json.loads((PRIOR/'protocol.json').read_text())
model=ResponseTable.load(SRC/'results-joint/response_table.npz')
cfg=PressureArray.load(PRIOR/'assumed_geometry.json');xy=np.load(PRIOR/'calibration.npz')['targets']
frozen={n:hashlib.sha256((SRC/n).read_bytes()).hexdigest() for n in old_protocol['source_sha256']}
assert frozen==old_protocol['source_sha256'],'Estimator or shared source changed; do not silently retune'
_,asset=upstream_components()
repo=ROOT/'work/CFDwavemaker'
protocol={'external_package':'CFDwavemaker','version':(repo/'src/VERSION').read_text().strip(),
 'git_commit':(repo/'VENDORED_COMMIT').read_text().strip(),
 'package_url':'https://github.com/oystelan/CFDwavemaker',
 'upstream_component_asset':str(asset.relative_to(repo)),
 'upstream_asset_sha256':hashlib.sha256(asset.read_bytes()).hexdigest(),
 'package_local_patch':'Guard unused omp.h include with _OPENMP for serial macOS build; no wave equations changed',
 'native_library_sha256':hashlib.sha256(LIB.read_bytes()).hexdigest(),
 'native_bulk_wrapper_sha256':hashlib.sha256(BATCH.read_bytes()).hexdigest(),
 'source_sha256':frozen,'frozen_against_previous_hybrid':True,
 'adapter_sha256':hashlib.sha256((ROOT/'work/independent_wave_adapter.py').read_bytes()).hexdigest(),
 'surface_and_incident_pressure':'Native CFDwavemaker API at every evaluation sample; no sea.Sea or truth_at synthesis',
 'wave_theory':'Linear finite-depth Airy at 1000 m water depth, effectively deep for generated periods',
 'shared_assumptions':['Known capsule strip-force law, mass/added mass/damping/restoring matrices',
                       'RK4 time-domain vehicle dynamics from rest, first-order moving-port hydrostatic correction',
                       'Linear potential wave theory, no wave-hull scattering or forward speed'],
 'tests':['upstream_single_system','gaussian_single_peak'],'phase_seeds':[731927,731928],
 'phase_design':'Two fresh independent uniform-phase draws; same draws paired across spectral shapes',
 'input_transformations':'312 upstream frequencies/directions, rotate directions +30 deg, normalize component variance to prescribed Hs=2 m; Gaussian case replaces amplitudes only',
 'gaussian_generation_only':{'peak_Hz':.125,'sigma_Hz':.018},
 'surface_patch_m':[60,12],'grid_points':len(xy),'submarine_centre_depth_m':15,
 'pressure_noise_std_Pa':0,'crossing_seas':False,'fs_Hz':2,'record_samples':8192,
 'forward_dt_s':.1,'forward_settle_s':400,'learning_samples':1024,
 'score':'Current timestamp; all generated frequencies; all samples from packet 1024 onward; no rescaling, phase alignment or time shift',
 'limits':'Independent wave/incident-pressure implementation only; same linear theory and existing known hull/sensor approximations. Not nonlinear CFD or real-sea validation.'}
(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
cfg.save(OUT/'assumed_geometry.json')
forward=Submarine(omega=model.omega,theta=model.theta,build_response=False)
baseline=calibrate(model,cfg,xy);rows=[]
for kind in protocol['tests']:
 for seed in protocol['phase_seeds']:
  label=f'{kind}_seed{seed}';folder=OUT/'cases'/label
  omega,a,phi,theta=write_case(folder,kind,seed)
  print('START',label,'component band Hz',omega.min()/(2*np.pi),omega.max()/(2*np.pi),flush=True)
  begin=time.perf_counter();wave=IndependentWave(folder)
  t,X=motion_from_components(forward,omega,a,phi,theta)
  pressure=pressure_record(wave,t,X,cfg)
  ts=t-t[0]
  np.savetxt(folder/'pressures.csv',np.c_[ts,pressure],delimiter=',',header='time_s,'+','.join(cfg.channels[:6]),comments='')
  print('PRESSURES READY',label,round(time.perf_counter()-begin,2),'s',flush=True)
  stream=HybridPressureLive(model,cfg,xy);base=PressureOnlyLive(baseline);est=[];base_est=[];durations=[]
  for index,(ti,pi) in enumerate(zip(ts,pressure)):
   start=time.perf_counter_ns();v=stream.push(ti,pi);elapsed=(time.perf_counter_ns()-start)/1e6
   b=base.push(ti,pi)
   if v is not None:
    est.append(v);base_est.append(b)
    if index>=1024:durations.append(elapsed)
  est=np.asarray(est);base_est=np.asarray(base_est)
  # Native reference evaluated only after pressure-only streaming predictions.
  reference=np.empty((len(t),len(xy)))
  for start in range(0,len(t),512):
   reference[start:start+512]=wave.surface(t[start:start+512],xy)
   if start%2048==0:print('REFERENCE',label,start,'/',len(t),flush=True)
  wave.close()
  metric,r2=errors(est,reference[1023:]);bm,_=errors(base_est,reference[1023:])
  row={'case':kind,'phase_seed':seed,'hybrid':metric,'flat_prior_same_samples':bm,
       'learned_prior':stream.fit.to_dict(),'startup_compute_s':stream.learning_compute_s,
       'ongoing_compute_ms_median':float(np.median(durations)),'ongoing_compute_ms_p99':float(np.percentile(durations,99)),
       'generator_band_Hz':[float(omega.min()/(2*np.pi)),float(omega.max()/(2*np.pi))],
       'peak_rotation_deg':float(np.rad2deg(abs(X[:,3:6]).max())),
       'small_angle_screen_pass':bool(np.rad2deg(abs(X[:,3:6]).max())<5),
       'total_run_seconds':time.perf_counter()-begin,'pressure_record_sha256':hashlib.sha256((folder/'pressures.csv').read_bytes()).hexdigest()}
  rows.append(row)
  np.savez_compressed(folder/'evaluation.npz',time_s=ts[1023:],x=xy[:,0],y=xy[:,1],estimate=est,truth=reference[1023:],R2=r2)
  stream.calibration.save(folder/'calibration.npz')
  if seed==731927:
   first=1024-128;last=first+848
   np.savez_compressed(folder/'viewer-data.npz',time_s=ts[first:last],pressures_Pa=pressure[first:last],truth=reference[first:last],python_check_output=est[:721])
  (OUT/'results.json').write_text(json.dumps({'protocol':protocol,'runs':rows},indent=2)+'\n')
  print('RESULT',json.dumps(row),flush=True)
assert all(hashlib.sha256((SRC/n).read_bytes()).hexdigest()==h for n,h in frozen.items())
report={'protocol':protocol,'runs':rows,'frozen_method_integrity_pass':True}
(OUT/'results.json').write_text(json.dumps(report,indent=2)+'\n')
print('FROZEN ESTIMATOR INTEGRITY PASSED',flush=True)
