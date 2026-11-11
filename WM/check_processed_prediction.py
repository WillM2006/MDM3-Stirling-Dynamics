"""Freeze a processing-only mean prediction, then test fresh native sensors.

This is forward prediction validation with known sea inputs, not an inverse
recovery test. The operator itself receives no wave truth or generating phases.
"""
import argparse
import hashlib
import inspect
import json
import time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path

import numpy as np
import invert_waves as iw
import calibrate_m as cm
import variance_convergence as vc
from variance_uniform import uniform_grid
from check_inverse_recovery import operator_responses,generate_record
from diagnose_inverse_covariance import load
from analyse_m_calibration import families
from audit_spectral_mean import processing_gain
import processed_spectral as model

GROUPS={'baseline':270000,'long_period':271000,'weak_waves':272000}
OLD={'baseline':('train_baseline',207000),'long_period':('test_long_period',209000),
     'weak_waves':('test_weak_waves',210000)}
METHODS=['current_wave_only','current_plus_noise','filter_alignment','full_processing']


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(training,data):
    data.mkdir(parents=True,exist_ok=True)
    parent=json.loads((vc.ROOT/'spectral_mean_manifest.json').read_text())
    if parent['base_main']!='caaaf415dbe7bd372187758fe542c69999913d3e':raise ValueError('unexpected base model')
    for name in ['submarine.py','sensors.py','sea.py','invert_waves.py','batch_invert.py']:
        if parent['source_sha256'][name]!=digest(vc.ROOT/name):raise ValueError('production source changed')
    # The modelling domain is the available BEM range; guard cells contain unknown power.
    edges=np.r_[.1/(2*np.pi),np.arange(3.5,64)*.005,2/(2*np.pi)]
    centres=(edges[:-1]+edges[1:])/2
    f=np.arange(10,63)*.005
    protocol=dict(base_main=parent['base_main'],M=parent['M'],tau=parent['tau'],records_per_case=200,
                  groups=GROUPS,headings=[12,36],primary_headings=12,methods=METHODS,
                  mean_model='piecewise constant unknown directional power per Hz in each frequency cell, with independent guard powers; responses use unchanged batch RAO interpolation',
                  processing='known interior FIR/common-grid interpolation and exact periodic-Hann harmonic expectation with segment detrending; frequency-cell quadrature fixed at order 16',
                  retained_effects='native nonlinear drag, radiation memory, controller, sensor errors, native-duration boundaries and existing shared physics remain',
                  fixed_noise='same previously recorded 64 zero-wave sensor spectra added to all noise-inclusive methods; no test calibration or fitted noise level',
                  primary_metric='RMS error of predicted mean in 81 real spectral entries divided by fixed earlier repeat-record SD; full_processing vs current_plus_noise',
                  uncertainty='1000 paired whole-record bootstrap means of fresh test records; conditional on fixed predictions, earlier units and noise calibration; pointwise not simultaneous',
                  decision='support practical mean improvement only if at least 10 percent lower primary error and paired 95 percent interval excludes no improvement; keep all cases and both direction grids visible',
                  limitations='known sea inputs are validation fixtures only; no inverse recovery or posterior-coverage claim; constant within-cell powers, interior processing and unchanged linear RAO remain approximations',
                  source_fingerprint=vc.model_fingerprint(),source_sha256={name:digest(vc.ROOT/name) for name in
                      ['processed_spectral.py','check_processed_prediction.py','audit_spectral_mean.py','check_inverse_recovery.py']},
                  native_source_sha256=hashlib.sha256(inspect.getsource(cm.native_samples).encode()).hexdigest(),
                  edges_Hz=edges.tolist(),output_f_Hz=f.tolist())
    path=data/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=protocol:raise ValueError('frozen prediction protocol changed')
    path.write_text(json.dumps(protocol,indent=2)+'\n')
    target=data/'prediction.npz'
    if target.exists():
        if digest(target)!=json.loads((data/'prediction_manifest.json').read_text())['prediction_sha256']:raise ValueError('frozen predictions changed')
        return protocol
    noise=[]
    for seed in range(260000,260064):
        with np.load(data.parent/'spectral_mean_noise'/f'seed-{seed}.npz') as z:noise.append(z['spectral'])
    noise=np.mean(noise,axis=0)
    noise_d=np.stack([iw.data_vector(s,iw.stack_pairs(9)) for s in noise])
    saved=dict(f=f,edges=edges,noise=noise_d);numerical=[]
    for headings in protocol['headings']:
        nodes,weights=model.quadrature(edges)
        beta,h=operator_responses(training/'hydrodynamics.npz',nodes.ravel(),headings)
        h=h.reshape(len(nodes),nodes.shape[1],headings,9)
        operator=model.observation_operator(f,nodes,weights,h)
        # Independent numerical refinement, before any fresh test outcomes.
        fine_nodes,fine_weights=model.quadrature(edges,32)
        _,fine_h=operator_responses(training/'hydrodynamics.npz',fine_nodes.ravel(),headings)
        refined=model.observation_operator(f,fine_nodes,fine_weights,fine_h.reshape(len(nodes),32,headings,9))
        _,training_d,_=load(training/'uniform6400/train_baseline',207000)
        standard=training_d.std(0,ddof=1)
        relative=float(np.linalg.norm((operator-refined)/standard[:,:,None])/np.linalg.norm(refined/standard[:,:,None]))
        if relative>.001:raise ValueError(f'quadrature not accurate enough: {relative}')
        numerical.append(dict(headings=headings,relative_standardised_operator_refinement=relative))
        del refined
        _,centre_h=operator_responses(training/'hydrodynamics.npz',centres,headings)
        selected=np.array([np.argmin(abs(centres-v)) for v in f])
        np.testing.assert_allclose(centres[selected],f,atol=1e-14,rtol=0)
        filtered=centre_h*processing_gain(centres)[:,None,:]
        for group in GROUPS:
            from sea import Sea
            sea=Sea(n_freq=6400,n_dir=36,**cm.CASES[group],gamma=3.3,s=10,direction=np.pi/2,seed=0)
            uniform_grid(sea)
            frequency_energy=np.histogram(sea.f,bins=edges,weights=sea.S_f*sea.delta_f)[0]
            distance=abs(np.angle(np.exp(1j*(sea.theta[:,None]-beta[None,:]))))
            directional_share=np.bincount(np.argmin(distance,axis=1),weights=sea.D*sea.delta_theta,minlength=headings)
            x=frequency_energy[:,None]/np.diff(edges)[:,None]*directional_share[None,:]
            current=np.stack([iw.operator_matrix(centre_h[ci],iw.stack_pairs(9))@x[ci] for ci in selected])
            point_processed=np.stack([iw.operator_matrix(filtered[ci],iw.stack_pairs(9))@x[ci] for ci in selected])
            saved[f'{group}_{headings}']=np.stack([current,current+noise_d,point_processed+noise_d,operator@x.ravel()+noise_d])
            folder,start=OLD[group];_,old,_=load(training/'uniform6400'/folder,start)
            saved[f'{group}_standard']=old.std(0,ddof=1)
        print('Frozen predictions',headings,'headings; quadrature error',relative,flush=True)
    np.savez_compressed(target,**saved)
    manifest=dict(prediction_sha256=digest(target),protocol_sha256=digest(path),numerical=numerical,
                  stage='prediction rules, validation means, noise offset and normalisation frozen before fresh outcomes')
    (data/'prediction_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return protocol


def generate(training,data,workers):
    protocol=prepare(training,data)
    tasks=[(g,seed,str(data)) for g,start in GROUPS.items() for seed in range(start,start+protocol['records_per_case'])]
    began=time.monotonic()
    with ProcessPoolExecutor(max_workers=workers,initializer=vc.initialise_worker,initargs=(training/'hydrodynamics.npz',)) as pool:
        futures={pool.submit(generate_record,t):t for t in tasks}
        for index,future in enumerate(as_completed(futures),1):
            future.result()
            if index%50==0:print(index,'/600 fresh native records;',round(time.monotonic()-began,1),'seconds',flush=True)


def analyse(data,output):
    protocol=json.loads((data/'protocol.json').read_text());frozen=json.loads((data/'prediction_manifest.json').read_text())
    if digest(data/'prediction.npz')!=frozen['prediction_sha256'] or digest(data/'protocol.json')!=frozen['protocol_sha256']:raise ValueError('frozen inputs changed')
    rows=[];changes=[];curves=[]
    with np.load(data/'prediction.npz') as prediction:
        for gi,(group,start) in enumerate(GROUPS.items()):
            f,observed,_=load(data/group,start)
            np.testing.assert_allclose(f,prediction['f'],atol=1e-14,rtol=0)
            weights=np.random.default_rng(273100+gi).multinomial(200,np.full(200,1/200),size=1000)/200
            means=weights@observed.reshape(200,-1);standard=prediction[f'{group}_standard']
            for headings in protocol['headings']:
                predicted=prediction[f'{group}_{headings}']
                for family,indices in [('all',families()['all']),('auto',families()['auto']),('cross',families()['real']+families()['imag'])]:
                    bias=(predicted-observed.mean(0))/standard
                    score=np.sqrt(np.mean(bias[:,:,indices]**2,axis=(1,2)))
                    draws=(predicted[None,:,:,:]-means.reshape(1000,1,53,81))/standard
                    scores=np.sqrt(np.mean(draws[:,:,:,indices]**2,axis=(2,3)))
                    for mi,method in enumerate(METHODS):
                        lo,hi=np.quantile(scores[:,mi],[.025,.975])
                        rows.append(dict(group=group,headings=headings,family=family,method=method,
                                         mean_prediction_error=float(score[mi]),ci95_low=float(lo),ci95_high=float(hi)))
                    reference=METHODS.index('current_plus_noise')
                    for mi in [2,3]:
                        improvement=100*(1-scores[:,mi]/scores[:,reference]);lo,hi=np.quantile(improvement,[.025,.975])
                        changes.append(dict(group=group,headings=headings,family=family,method=METHODS[mi],
                                            improvement_pct=float(100*(1-score[mi]/score[reference])),ci95_low=float(lo),ci95_high=float(hi)))
                for fi,frequency in enumerate(f):
                    for ci in families()['auto']:
                        curves.append(dict(group=group,headings=headings,frequency_Hz=float(frequency),entry=ci,
                                           observed=float(observed[:,fi,ci].mean()),**{m:float(predicted[mi,fi,ci]) for mi,m in enumerate(METHODS)}))
    for suffix,values in [('scores',rows),('changes',changes),('curves',curves)]:vc.write_csv(output/f'processed_prediction_{suffix}.csv',values)
    (output/'processed_prediction_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    (output/'processed_prediction_manifest.json').write_text(json.dumps(frozen,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generate',action='store_true')
    parser.add_argument('--training',type=Path,default=Path('variance_output/m_calibration'))
    parser.add_argument('--data',type=Path,default=Path('variance_output/processed_prediction'))
    parser.add_argument('--output',type=Path,default=vc.ROOT)
    parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    if args.generate:generate(args.training,args.data,args.workers)
    else:analyse(args.data,args.output)
