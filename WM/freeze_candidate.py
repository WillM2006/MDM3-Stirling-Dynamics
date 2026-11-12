"""Pin the tested candidate and score peak period from existing frozen fits."""
import csv,json,hashlib
from pathlib import Path
import numpy as np
from variance_convergence import ROOT,write_csv
from check_inverse_recovery import GROUPS,METHODS

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def run():
    files=['submarine.py','sensors.py','sea.py','invert_waves.py','batch_invert.py','calibrate_m.py','variance_uniform.py','check_cross_variance.py','check_inverse_recovery.py']
    frozen=dict(status='research candidate frozen; not approved for production',M=8.785033150237455,tau=5,
        simulation='6400 uniform wave components, 36 generation directions, 40 BEM frequencies, 90m hull, 15m depth, zero speed; existing nonlinear physics and sensor errors',
        processing='1200s total, 300s discard, 2Hz samples, 200s Hann segments, 50% overlap, 0.05--0.31Hz retained band',
        inverse='12 headings primary, 36 sensitivity; original diagonal inverse with corrected marginal variances; original data-derived prior scale',
        excluded='no pooled covariance, smoothing, split weights, mean correction or fitted bias offsets',
        evidence='existing 600-seed inverse_recovery comparison; no repeat simulation; peak-period scoring added retrospectively, not an untouched test',
        adoption='not approved: marginal variance calibration improves but recovery tradeoffs and posterior undercoverage remain; do not apply to old 40-component data',
        sources={p:digest(ROOT/p) for p in files},
        evidence_hashes={p:digest(ROOT/p) for p in ['m_calibration_protocol.json','cross_variance_protocol.json','inverse_recovery_protocol.json','inverse_recovery_records.csv']})
    out=ROOT/'candidate_freeze.json'
    if out.exists():assert json.loads(out.read_text())==frozen
    out.write_text(json.dumps(frozen,indent=2)+'\n')
    records=[];hashes={}
    for h in [12,36]:
        for g,start in GROUPS.items():
            for seed in range(start,start+200):
                for method in METHODS:
                    p=Path('variance_output/inverse_recovery/fits')/str(h)/g/f'seed-{seed}_{method}.npz'
                    hashes[str(p)]=digest(p)
                    with np.load(p) as z:
                        f=z['f']; sf=z['x'].sum(1); truth=z['truth'].sum(1)
                        assert np.isfinite(sf).all() and sf.max()>0
                        estimate=1/f[np.argmax(sf)];target=1/f[np.argmax(truth)]
                    records.append(dict(group=g,headings=h,seed=seed,method=method,Tp=estimate,Tp_true_band=target,abs_error_pct=100*abs(estimate/target-1)))
    rows=[]
    for h in [12,36]:
        for gi,g in enumerate(GROUPS):
            draws=np.random.default_rng(290100+gi).integers(0,200,(1000,200))
            vals={m:np.array([r['abs_error_pct'] for r in records if r['group']==g and r['headings']==h and r['method']==m]) for m in METHODS}
            for m,v in vals.items():
                assert len(v)==200
                delta=v-vals['existing_2.5'];lo,hi=np.quantile(delta[draws].mean(1),[.025,.975])
                rows.append(dict(group=g,headings=h,method=m,mean_abs_Tp_error_pct=v.mean(),change_from_existing=delta.mean(),change_ci95_low=lo,change_ci95_high=hi))
    write_csv(ROOT/'candidate_peak_period_records.csv',records);write_csv(ROOT/'candidate_peak_period_summary.csv',rows)
    (ROOT/'candidate_peak_period_manifest.json').write_text(json.dumps(dict(inputs=hashes,definition='reciprocal of largest retained-bin summed directional power; ties use lowest frequency; true comparator is discrete retained spectrum peak, not nominal continuous Tp',source_sha256=digest(__file__)),indent=2)+'\n')
    print(json.dumps(rows,indent=2))

if __name__=='__main__':run()
