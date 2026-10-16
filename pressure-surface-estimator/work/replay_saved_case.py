"""Pressure-only relearning/causal replay; load evaluation truth afterward."""
from pathlib import Path
import argparse,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from hybrid_pressure_live import HybridPressureLive
from pressure_array import PressureArray
from pressure_only_live import Calibration
from submarine import ResponseTable
from run_pressure_array_experiment import errors

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--case',default='upstream_single_system_seed731927')
    parser.add_argument('--all',action='store_true')
    args=parser.parse_args()
    base=ROOT/'outputs/independent-wave-package'
    original=json.loads((base/'results.json').read_text())
    available={f"{r['case']}_seed{r['phase_seed']}":r for r in original['runs']}
    labels=sorted(available) if args.all else [args.case]
    if any(label not in available for label in labels):parser.error('Unknown case; choose '+', '.join(available))
    model=ResponseTable.load(ROOT/'src/results-joint/response_table.npz')
    cfg=PressureArray.load(base/'assumed_geometry.json')
    # Requested surface locations are calibration metadata, not measured truth.
    targets=Calibration.load(base/'cases'/labels[0]/'calibration.npz').targets
    result=[]
    for label in labels:
        folder=base/'cases'/label
        record=np.loadtxt(folder/'pressures.csv',delimiter=',',skiprows=1)
        live=HybridPressureLive(model,cfg,targets)
        predictions=[];times=[];start=time.perf_counter()
        for row in record:
            estimate=live.push(row[0],row[1:])
            if estimate is not None:predictions.append(estimate);times.append(row[0])
        # No reference file or true surface entered the learner/stream above.
        with np.load(folder/'reference.npz') as reference:
            np.testing.assert_array_equal(np.asarray(times),reference['time_s'])
            np.testing.assert_array_equal(targets,reference['targets'])
            metric,_=errors(np.asarray(predictions),reference['truth'].astype('float64'))
        expected=available[label]['hybrid']
        for key,value in metric.items():
            np.testing.assert_allclose(value,expected[key],rtol=0,atol=5e-8,err_msg=label+' '+key)
        for key in ['mean_direction_deg','spread_s','Tp_s','Hs_m']:
            np.testing.assert_allclose(getattr(live.fit,key),available[label]['learned_prior'][key],rtol=1e-4,atol=1e-3,err_msg=label+' prior '+key)
        row={'case':label,'metrics':metric,'learned_prior':live.fit.to_dict(),'score_samples':len(times),
             'scored_surface_points':len(targets),'elapsed_s':time.perf_counter()-start,
             'matches_original_metrics_within_5e_minus8':True,
             'reference_used_only_after_prediction':True}
        result.append(row);print(json.dumps(row),flush=True)
    out=ROOT/'outputs/phone-replay-results';out.mkdir(exist_ok=True)
    (out/'replay-results.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
