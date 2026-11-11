"""Sensor-only feasibility of time-block weight estimation; no inverse recovery."""
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from scipy.signal import get_window
import calibrate_m as cm
import variance_convergence as vc
import invert_waves as iw
from check_fixed_estimator_scatter import digest
from check_pooled_covariance import GROUPS

CACHE=Path('variance_output/block_information')
SCHEMES={'halves':[[0,900],[900,1800]],'gap100':[[0,800],[1000,1800]]}


def segment_count(n):
    window=int(iw.FS*iw.SEG);hop=window//2
    return 1+(n-window)//hop if n>=window else 0


def theoretical_averages(k):
    w=get_window('hann',int(iw.FS*iw.SEG));hop=len(w)//2
    overlap=np.dot(w[:-hop],w[hop:])/np.dot(w,w)
    return k/(1+2*(1-1/k)*overlap**2)


def generate_record(task):
    group,seed=task;p=CACHE/group/f'seed-{seed}.npz'
    if p.exists():return 'cached'
    _,samples=cm.native_samples('uniform6400',group,seed)
    full=cm.spectral_record(samples)
    original=Path('variance_output/pooled_covariance')/group/f'seed-{seed}.npz'
    with np.load(original) as z:
        np.testing.assert_array_equal(z['f'],full['f'])
        np.testing.assert_allclose(z['spectral'],full['spectral'],rtol=1e-10,atol=1e-16)
    saved=dict(f=full['f'],full=full['spectral'])
    for scheme,blocks in SCHEMES.items():
        saved[scheme]=np.stack([cm.spectral_record(samples[a:b])['spectral'] for a,b in blocks])
    p.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(p,**saved)
    return 'matched'


def run():
    CACHE.mkdir(parents=True,exist_ok=True)
    parent=json.loads((vc.ROOT/'m_calibration_protocol.json').read_text()) if (vc.ROOT/'m_calibration_protocol.json').exists() else json.loads(Path('variance_output/m_calibration/protocol.json').read_text())
    protocol=dict(groups=GROUPS,records_per_group=200,schemes=SCHEMES,full_samples=1800,sample_rate=iw.FS,segment_seconds=iw.SEG,
        counts=dict(full=segment_count(1800),halves=segment_count(900),gap100=segment_count(800)),
        setup='regenerate the same 250xxx seeded native 1200-second simulations; discard 300 seconds using unchanged preprocessing; verify whole-record spectral parity before using blocks',
        reference='230xxx and 240xxx independent spectra select each channel peak frequency; no wave truth or inverse fit',
        independence='whole-record paired bootstrap of cross-block correlations for nine auto-spectral peaks and nine integrated auto-spectral band powers; report pointwise 95% intervals; practical correlation band +/-0.2, not proof of independence',
        precision='empirical effective averages mean(auto)^2 / var(auto), and coefficient of variation of auto^2 as a variance-weight proxy; both blocks shown separately, not pooled as independent records',
        decision='do not run recovery in this study; assess whether split blocks provide approximately independent AND stable per-frequency variance estimates; no tuning M or selecting a split from recovery',
        limitations='stationary simulator, previously used seeds, conditional bootstrap; no field-data stationarity validation; dependence across nonlinear spectral features not exhausted by correlation probes',
        M='production/research M=8.785 remains unchanged; diagnostic effective-averages values are not replacement M values',
        source_sha256=digest(__file__),generator_fingerprint=vc.model_fingerprint(),hydrodynamics_sha256=digest('variance_output/m_calibration/hydrodynamics.npz'),
        parent_fingerprint=parent.get('source_fingerprint'))
    p=CACHE/'protocol.json'
    if p.exists():assert json.loads(p.read_text())==protocol
    p.write_text(json.dumps(protocol,indent=2)+'\n')
    print('Welch counts',protocol['counts'],'idealised averages',{k:theoretical_averages(v) for k,v in protocol['counts'].items()},flush=True)
    tasks=[(g,seed) for g,start in GROUPS.items() for seed in range(start,start+200)]
    with ProcessPoolExecutor(max_workers=2,initializer=vc.initialise_worker,initargs=(Path('variance_output/m_calibration/hydrodynamics.npz'),)) as pool:
        futures=[pool.submit(generate_record,t) for t in tasks]
        for i,future in enumerate(as_completed(futures),1):
            future.result()
            if i%50==0:print(i,'/600 regenerated and matched',flush=True)
    (vc.ROOT/'block_information_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')


if __name__=='__main__':run()
