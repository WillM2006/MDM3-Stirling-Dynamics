"""Post-validation mean-spectrum audit. No parameter fitting or inverse changes.

Known wave spectra are diagnostic inputs only. They never calibrate M, covariance
or a deployable response. Alternative means below are NOT candidate estimators.
"""
import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import scipy
from scipy.signal import firwin, freqz
import invert_waves as iw
import variance_convergence as vc
from check_inverse_recovery import operator_responses, truth_spectrum, candidate_variances
from diagnose_inverse_covariance import load
from analyse_m_calibration import families
from variance_uniform import uniform_grid
import correlated_spectral as cs
import pooled_spectral as ps


def exponential_sum(cycles, length):
    return length*np.exp(1j*np.pi*(length-1)*cycles)*np.sinc(length*cycles)/np.sinc(cycles)


def welch_weights(f, nu, fs=2., length=400):
    """Expected one-sided CSD of unit-variance random-phase real harmonics."""
    def window(x):
        return (.5*exponential_sum(x,length)-.25*exponential_sum(x+1/length,length)
                -.25*exponential_sum(x-1/length,length))
    mean=exponential_sum(nu/fs,length)/length
    positive=window((nu[None,:]-f[:,None])/fs)-window(-f[:,None]/fs)*mean
    negative=window((-nu[None,:]-f[:,None])/fs)-window(-f[:,None]/fs)*mean.conj()
    # Periodic Hann: sum(w**2)=3N/8; retained bins exclude DC/Nyquist.
    return abs(positive)**2/(fs*3*length/8),abs(negative)**2/(fs*3*length/8)


def processing_gain(f):
    """Interior FIR and common-grid interpolation, derived from read_channels."""
    gain=[]
    for raw_fs in [50.,50.,100.,10.,10.,10.,10.,10.,10.]:
        q=int(raw_fs/iw.FS);b=firwin(20*q+1,1/q,window='hamming')
        response=freqz(b,worN=2*np.pi*f/raw_fs)[1]*np.exp(1j*2*np.pi*f*(10*q)/raw_fs)
        # tc=300+0.5k; stream starts at 1/raw_fs. Fraction after preceding sample.
        alpha=(iw.SETTLE*iw.FS-iw.FS/raw_fs)%1
        interpolation=(1-alpha)*np.exp(1j*2*np.pi*f*alpha/iw.FS)+alpha*np.exp(-1j*2*np.pi*f*(1-alpha)/iw.FS)
        gain.append(response*interpolation)
    return np.stack(gain,axis=-1)


def response_from_impedance(sub, frequencies, memory=False):
    """Diagnosis only: contrast interpolated RAOs with two linear dynamics."""
    w=2*np.pi*frequencies
    excitation=np.stack([iw.interp_complex(v,sub.bem_omegas,sub.F_exc) for v in w])
    if memory:
        weights=np.full(len(sub.t_K),vc.sensors.dt);weights[[0,-1]]*=.5
        kernel=np.einsum('tw,tab,t->wab',np.exp(1j*sub.t_K[:,None]*w),sub.K,weights)
        z=-w[:,None,None]**2*(sub.M+sub.A_inf)-1j*w[:,None,None]*(kernel+sub.B_visc+np.diag(sub.Kd))+sub.C_hydro+np.diag(sub.Kp)
    else:
        added=np.stack([iw.interp_complex(v,sub.bem_omegas,sub.A) for v in w])
        damping=np.stack([iw.interp_complex(v,sub.bem_omegas,sub.B) for v in w])
        z=-w[:,None,None]**2*(sub.M+added)-1j*w[:,None,None]*(damping+sub.B_visc+np.diag(sub.Kd))+sub.C_hydro+np.diag(sub.Kp)
    motion=np.linalg.solve(z,excitation.transpose(0,2,1)).transpose(0,2,1)
    positions=dict(IMU=vc.sensors.R_IMU,**vc.sensors.PRESSURE_SENSORS)
    h=np.empty((len(w),36,9),complex)
    for fi,v in enumerate(w):
        for ci,name in enumerate(iw.CHANNELS):
            h[fi,:,ci]=iw.channel_operator(name,motion[fi],v,sub.theta,positions,(1025.,9.81,15.))
    return h


def expected_spectra(h, energy, positive, negative):
    lines=np.einsum('fjc,fj,fjk->fck',h,energy,h.conj())
    return (positive@lines.reshape(len(lines),-1)+negative@lines.conj().reshape(len(lines),-1)).reshape(len(positive),9,9)


def noise_records(folder, count=64):
    """Separate zero-wave sensor runs; no wave truth or fitted response."""
    folder.mkdir(parents=True,exist_ok=True);t=np.arange(12001)*.1
    with patch.object(vc.sensors,'incident_pressure',side_effect=lambda sub,a,x,y,z,t:np.zeros(len(t))):
        true=vc.sensors.true_signals(vc.base,np.zeros_like(vc.base.wave_amplitudes),t,np.zeros((len(t),12)))
    spectra=[]
    for seed in range(260000,260000+count):
        path=folder/f'seed-{seed}.npz'
        if path.exists():
            with np.load(path) as z:spectral=z['spectral']
        else:
            vc.sensors.rng=np.random.default_rng(seed)
            with patch.object(vc.sensors,'gauss_markov',vc.fast_gauss_markov):measured=vc.sensors.measure_all(t,true)
            _,samples=vc.preprocess(measured);f,spectral=iw.welch_matrix(samples)
            keep=(f>=iw.BAND[0])&(f<=iw.BAND[1]);spectral=spectral[keep]
            np.savez_compressed(path,f=f[keep],spectral=spectral)
        spectra.append(spectral)
    return np.stack(spectra)


def no_drag_record(task):
    from calibrate_m import native_samples, spectral_record
    group,seed,folder=task;path=Path(folder)/group/f'seed-{seed}.npz'
    if not path.exists():
        with patch.object(vc.sensors.Submarine,'drag',lambda self,v:np.zeros(6)):
            _,samples=native_samples('uniform6400',group,seed)
        path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,**spectral_record(samples))
    return group,seed


def audit(training,data,output):
    vc.initialise_worker(training/'hydrodynamics.npz')
    protocol=json.loads((data/'protocol.json').read_text())
    from sea import Sea
    from calibrate_m import CASES
    sea=Sea(n_freq=6400,n_dir=36,**CASES['baseline'],direction=np.pi/2,s=10,seed=0)
    uniform_grid(sea);nu=sea.f
    with np.load(data/'baseline/seed-250000.npz') as z:f=z['f']
    df=float(f[1]-f[0]);positive,negative=welch_weights(f,nu)
    rect=(abs(f[:,None]-nu[None,:])<df/2)/df
    _,fine=operator_responses(training/'hydrodynamics.npz',nu,36)
    direct=response_from_impedance(vc.base,nu)
    memory=response_from_impedance(vc.base,nu,memory=True)
    gain=processing_gain(nu)
    noise=noise_records(data.parent/'spectral_mean_noise')
    no_drag_folder=data.parent/'spectral_mean_no_drag'
    tasks=[(group,seed,str(no_drag_folder)) for group,settings in protocol['groups'].items()
           for seed in range(settings['seed_start'],settings['seed_start']+64)]
    with ProcessPoolExecutor(max_workers=2,initializer=vc.initialise_worker,initargs=(training/'hydrodynamics.npz',)) as pool:
        for index,result in enumerate(pool.map(no_drag_record,tasks),1):
            if index%32==0:print('Paired no-drag diagnostic',index,'/192',flush=True)
    rows=[];effects=[];curves=[];drag_rows=[];drag_effects=[]
    for gi,(group,settings) in enumerate(protocol['groups'].items()):
        f,observed,mean=load(data/group,settings['seed_start'])
        sea=Sea(n_freq=6400,n_dir=36,**CASES[group],direction=np.pi/2,s=10,seed=0);uniform_grid(sea)
        energy=sea.wave_amplitudes**2/2
        beta36,h36=operator_responses(training/'hydrodynamics.npz',f,36)
        truth36,_=truth_spectrum(group,f,beta36)
        centre36=np.einsum('fjc,fj,fjk->fck',h36,truth36,h36.conj())
        variants=dict(centre36=centre36,
                      cell_integrated=expected_spectra(fine,energy,rect,np.zeros_like(rect)),
                      welch_averaged=expected_spectra(fine,energy,positive,negative),
                      processed=expected_spectra(fine*gain[:,None,:],energy,positive,negative),
                      direct_impedance=expected_spectra(direct*gain[:,None,:],energy,positive,negative),
                      truncated_memory=expected_spectra(memory*gain[:,None,:],energy,positive,negative))
        # Independent per-channel Gaussian clocks attenuate expected cross-spectra.
        atten=np.exp(-(2*np.pi*nu*vc.sensors.TIMING_JITTER)**2)
        clock=(atten[:,None,None]+(1-atten)[:,None,None]*np.eye(9))
        lines=np.einsum('fjc,fj,fjk->fck',memory*gain[:,None,:],energy,(memory*gain[:,None,:]).conj())*clock
        variants['memory_noise_clock']=(positive@lines.reshape(len(nu),-1)+negative@lines.conj().reshape(len(nu),-1)).reshape(len(f),9,9)+noise.mean(0)
        off=[];on=[]
        for seed in range(settings['seed_start'],settings['seed_start']+64):
            with np.load(no_drag_folder/group/f'seed-{seed}.npz') as z:off.append(z['spectral'])
            with np.load(data/group/f'seed-{seed}.npz') as z:on.append(z['spectral'])
        off=np.stack(off);on=np.stack(on)
        variants['observed_no_drag']=np.mean(off,axis=0)
        variants['observed_paired_drag']=np.mean(on,axis=0)
        auto=observed[:,:,families()['auto']]
        weights=np.random.default_rng(719000+gi).multinomial(200,np.full(200,1/200),size=1000)/200
        band=auto.sum(1)*df;band_means=weights@band
        for name,predicted in variants.items():
            if name.startswith('observed_'):continue
            predicted_band=np.diagonal(predicted,axis1=1,axis2=2).real.sum(0)*df
            ratios=predicted_band/band.mean(0);low,high=np.quantile(predicted_band/band_means,[.025,.975],axis=0)
            for ci,channel in enumerate(iw.CHANNELS):
                rows.append(dict(group=group,variant=name,channel=channel,predicted_over_observed=float(ratios[ci]),ci95_low=float(low[ci]),ci95_high=float(high[ci])))
        paired_weights=np.random.default_rng(819000+gi).multinomial(64,np.full(64,1/64),size=1000)/64
        on_power=np.diagonal(on,axis1=2,axis2=3).real.sum(1)*df
        off_power=np.diagonal(off,axis1=2,axis2=3).real.sum(1)*df
        delta_draws=(paired_weights@(off_power-on_power))/(paired_weights@on_power)*100
        low,high=np.quantile(delta_draws,[.025,.975],axis=0)
        for ci,channel in enumerate(iw.CHANNELS):
            drag_rows.append(dict(group=group,channel=channel,records=64,off_minus_on_power_pct=float(100*(off_power[:,ci].mean()/on_power[:,ci].mean()-1)),ci95_low=float(low[ci]),ci95_high=float(high[ci])))
        for fi,v in enumerate(f):
            for ci,channel in enumerate(iw.CHANNELS):
                curves.append(dict(group=group,frequency_Hz=float(v),channel=channel,observed=float(mean[fi,ci,ci].real),
                                   **{name:float(value[fi,ci,ci].real) for name,value in variants.items()}))
        calibration=protocol['calibration'][group]
        _,_,reference=load(training/'uniform6400'/calibration['folder'],calibration['seed_start'])
        with np.load(data/f'pool-{group}.npz') as z:factors=z['factors']
        rho=cs.welch_frequency_correlation(np.rint(f*iw.SEG).astype(int))
        for headings in [12,36]:
            beta,h=operator_responses(training/'hydrodynamics.npz',f,headings)
            truth,_=truth_spectrum(group,f,beta);target=truth.sum()*df
            centre=np.einsum('fjc,fj,fjk->fck',h,truth,h.conj())
            with patch.object(iw,'noise_variances',candidate_variances):_,_,scales=iw.invert(f,reference,h,protocol['tau'],protocol['M'])
            for name,predicted in dict(centre_own_grid=centre,**variants,observed=mean).items():
                x,_,_=ps.pooled_inverse(f,predicted,h,protocol['tau'],factors,rho,scales)
                effects.append(dict(group=group,headings=headings,variant=name,recovered_energy_bias_pct=float(100*(x.sum()*df/target-1))))
            deltas=[]
            for difference in off-on:
                x,_,_=ps.pooled_inverse(f,difference,h,protocol['tau'],factors,rho,scales)
                deltas.append(float(100*x.sum()*df/target))
            low,high=np.quantile(paired_weights@np.array(deltas),[.025,.975])
            drag_effects.append(dict(group=group,headings=headings,records=64,off_minus_on_energy_pct=float(np.mean(deltas)),ci95_low=float(low),ci95_high=float(high)))
        print('Audited',group,flush=True)
    for suffix,values in [('band_ratios',rows),('curves',curves),('effects',effects),('drag_band',drag_rows),('drag_effects',drag_effects)]:vc.write_csv(output/f'spectral_mean_{suffix}.csv',values)
    source=['audit_spectral_mean.py','test_spectral_mean.py','plot_spectral_mean_audit.py','invert_waves.py','batch_invert.py','submarine.py','sensors.py','sea.py','pooled_spectral.py']
    input_hashes={}
    for label,paths in [('original',sorted(p for group in protocol['groups'] for p in (data/group).glob('seed-*.npz'))),
                        ('calibration',sorted(p for settings in protocol['calibration'].values() for p in (training/'uniform6400'/settings['folder']).glob('seed-*.npz'))),
                        ('no_drag',sorted(no_drag_folder.glob('*/seed-*.npz'))),
                        ('noise_only',sorted((data.parent/'spectral_mean_noise').glob('seed-*.npz')))]:
        digest=hashlib.sha256()
        for path in paths:digest.update(path.name.encode());digest.update(path.read_bytes())
        input_hashes[label]=dict(records=len(paths),ordered_filename_and_bytes_sha256=digest.hexdigest())
    manifest=dict(base_main='caaaf415dbe7bd372187758fe542c69999913d3e',M=protocol['M'],tau=protocol['tau'],
                  diagnostic_only=True,refitting=False,existing_records=600,noise_only_seeds=[260000,260063],paired_no_drag_records=192,
                  drag_pairing='First 64 existing seeds per condition, same wave phases and sensor RNG, only quadratic translational drag disabled. Fixed reference-prior recovery operator used for diagnostic energy projection.',
                  uncertainty='1000 whole-record bootstrap draws for observed mean band-power ratios, conditional on predicted means; not simultaneous intervals',
                  limitations='Post-validation model audit, not independent recovery validation. Exact Hann/detrend harmonic expectation; interior FIR/interpolation; continuous truncated-memory comparison excludes RK4 stage holding, nonlinear drag, startup and boundaries. No inverse correction is implemented.',
                  versions=dict(numpy=np.__version__,scipy=scipy.__version__),input_hashes=input_hashes,
                  hydrodynamics_sha256=hashlib.sha256((training/'hydrodynamics.npz').read_bytes()).hexdigest(),
                  calibration_pools_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(data.glob('pool-*.npz'))},
                  locally_flat_hann_check=dict(segments=8,adjacent_window_overlap=1/6,effective_L=64/(8+14/36),plugin_M=1+64/(8+14/36)),
                  source_sha256={name:hashlib.sha256((vc.ROOT/name).read_bytes()).hexdigest() for name in source})
    (output/'spectral_mean_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training',type=Path,default=Path('variance_output/m_calibration'))
    parser.add_argument('--data',type=Path,default=Path('variance_output/pooled_covariance'))
    parser.add_argument('--output',type=Path,default=vc.ROOT)
    args=parser.parse_args();audit(args.training,args.data,args.output)
