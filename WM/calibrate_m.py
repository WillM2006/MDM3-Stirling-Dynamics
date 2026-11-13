"""Native-duration spectral-variance calibration; no inverse or wave-truth fitting."""
import argparse
import copy
import hashlib
import inspect
import json
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from unittest.mock import patch

import numpy as np
import invert_waves as iw
import variance_convergence as vc
from variance_uniform import uniform_grid, uniform_projection

RECORDS = 200
CASES = {"baseline": dict(Hs=1.7, Tp=8), "long_period": dict(Hs=1.7, Tp=12),
         "weak_waves": dict(Hs=.5, Tp=8)}
GROUPS = {"train_baseline": ("baseline", 207000), "test_baseline": ("baseline", 208000),
          "test_long_period": ("long_period", 209000), "test_weak_waves": ("weak_waves", 210000)}
MODES = {"current40": 40, "uniform6400": 6400}
CHECKED_CASES = set()


def native_samples(mode, case, seed, accelerated=True):
    sub = copy.copy(vc.base)
    sub.params = vc.base.params.copy()
    settings = dict(CASES[case], gamma=3.3, s=10, direction=np.pi/2, seed=seed)
    vc.sea.Sea.__init__(sub, n_freq=MODES[mode], n_dir=36, **settings)
    if mode == "uniform6400":
        uniform_grid(sub)
    sub.params.update(settings)
    sub.build_excitation()
    if accelerated:
        half_time = np.arange(24001)*.05
        positions = [sub.centre+[0,0,-.2]+r for r in
                     [*vc.sensors.PRESSURE_SENSORS.values(), vc.sensors.R_DEPTH]]
        coefficients = np.column_stack([sub.F_hat.sum(1), *[
            vc.pressure_coefficient(sub, sub.wave_amplitudes, *position) for position in positions]])
        projected = (uniform_projection(sub.w, half_time, coefficients) if mode == "uniform6400"
                     else vc.harmonic_projection(np.exp(-1j*half_time[:,None]*sub.w), coefficients))
        forcing = vc.sensors.ramp(half_time)[:,None]*projected[:,:6]
        pressures = vc.sensors.RHO*vc.sensors.G*vc.sensors.ramp(half_time[::2])[:,None]*projected[::2,6:]
        for index in [1,1201,12000,24000]:
            vc.check_projected_forcing(forcing[index], sub.excitation(half_time[index],0,0),sub.F_hat.sum(1))
        if (mode,case) not in CHECKED_CASES:
            tt = half_time[6000:6020:2]
            for i, position in enumerate(positions):
                direct = vc.sensors.incident_pressure(sub, sub.wave_amplitudes,
                    np.full(len(tt),position[0]),np.full(len(tt),position[1]),position[2],tt)
                np.testing.assert_allclose(pressures[3000:3010,i],direct,rtol=1e-10,atol=1e-7)
            CHECKED_CASES.add((mode,case))
        sub.pressure_cache = {tuple(position): pressures[:,i] for i, position in enumerate(positions)}
        sub.time_carrier = []
        sub.excitation = lambda t,x,y: forcing[int(round(t/.05))]
    t, history = sub.simulate(1200)
    if not np.isfinite(history).all():
        raise ValueError("nonfinite hull trajectory")
    if accelerated:
        with patch.object(vc.sensors, "incident_pressure", vc.fast_pressure):
            true = vc.sensors.true_signals(sub, sub.wave_amplitudes,t,history)
    else:
        true = vc.sensors.true_signals(sub, sub.wave_amplitudes,t,history)
    vc.sensors.rng = np.random.default_rng(seed)
    if accelerated:
        with patch.object(vc.sensors, "gauss_markov", vc.fast_gauss_markov):
            measured = vc.sensors.measure_all(t,true)
    else:
        measured = vc.sensors.measure_all(t,true)
    tc, samples = vc.preprocess(measured)
    if samples.shape != (1800,9) or not np.isfinite(samples).all():
        raise ValueError("unexpected native-duration processed recording")
    return tc, samples


def spectral_record(samples):
    f, spectral = iw.welch_matrix(samples)
    selected = (f >= iw.BAND[0]) & (f <= iw.BAND[1])
    spectral = spectral[selected]
    pairs = iw.stack_pairs(samples.shape[1])
    observed = np.stack([iw.data_vector(s,pairs) for s in spectral])
    numerator = np.stack([iw.noise_variances(s,pairs,1.) for s in spectral])
    return dict(f=f[selected], observed=observed, numerator=numerator, spectral=spectral)


def simulate(task):
    mode, group, seed, output = task
    path = Path(output)/mode/group/f"seed-{seed}.npz"
    if path.exists():
        return mode, group, seed, "cached"
    _, samples = native_samples(mode,GROUPS[group][0],seed)
    values = spectral_record(samples)
    if not all(np.isfinite(value).all() for value in values.values()):
        raise ValueError("nonfinite spectral record")
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream,**values)
    temporary.replace(path)
    return mode, group, seed, "generated"


def generate(output, cache, workers):
    output.mkdir(parents=True,exist_ok=True)
    protocol = dict(base_main="caaaf415dbe7bd372187758fe542c69999913d3e", modes=MODES,cases=CASES,
                    groups={name:dict(case=case,seed_start=seed,records=RECORDS) for name,(case,seed) in GROUPS.items()},
                    current40_groups=["train_baseline","test_baseline"], native_duration_s=1200,
                    discarded_s=300, usable_s=900, hydrodynamic_frequencies=40,
                    length_m=90,radius_m=5.36,depth_m=15,speed_m_s=0,directions=36,
                    gamma=3.3,spreading_s=10,direction_rad=float(np.pi/2),
                    sample_rate_Hz=2,welch_seconds=200,overlap=.5,bootstrap=1000,
                    fit="baseline training only: median over auto-spectral cells of mean(numerator at M=1) / repeat-record variance",
                    primary_family="auto: exact auto-spectral variance form; real/imaginary cross approximations are validated separately",
                    interpretation="calibrate the implemented estimated-auto-spectrum numerator; fitted M is not a literal Welch-segment count",
                    diagnostics="separate auto, real-cross, imaginary-cross and all-entry required M; no frequency-dependent M fitted",
                    validation="fresh held-out baseline, long-period and weak-wave records; fixed training M; no inverse or wave-truth scoring",
                    adoption="do not apply an auto-calibrated value globally if the cross-spectral variance approximation is incompatible",
                    scope="preserve existing forward/inverse assumptions; no independent-model overhaul in this study",
                    source_fingerprint=vc.model_fingerprint(),
                    inverse_source_sha256=hashlib.sha256(Path(iw.__file__).read_bytes()).hexdigest(),
                    native_generator_sha256=hashlib.sha256(inspect.getsource(native_samples).encode()).hexdigest())
    path=output/"protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("output folder contains a different calibration protocol")
    path.write_text(json.dumps(protocol,indent=2)+"\n")
    target_cache=output/"hydrodynamics.npz"
    if cache.resolve()!=target_cache.resolve():
        shutil.copyfile(cache,target_cache)
    vc.initialise_worker(target_cache)
    checks=[]
    for index,case in enumerate(CASES):
        t,original=native_samples("current40",case,219000+index,accelerated=False)
        tt,fast=native_samples("current40",case,219000+index,accelerated=True)
        np.testing.assert_array_equal(t,tt)
        scale=np.maximum(np.max(abs(original),axis=0),1e-12)
        np.testing.assert_allclose(original/scale,fast/scale,rtol=2e-8,atol=2e-9)
        checks.append(dict(case=case,max_scaled_difference=float(np.max(abs(original-fast)/scale))))
        print(f"Native acceleration check passed: {case}",flush=True)
    (output/"acceleration_checks.json").write_text(json.dumps(checks,indent=2)+"\n")
    tasks=[]
    for mode in ["uniform6400","current40"]:
        for group,(_,seed) in GROUPS.items():
            if mode=="current40" and group not in protocol["current40_groups"]:
                continue
            tasks.extend((mode,group,s,str(output)) for s in range(seed,seed+RECORDS))
    start=time.monotonic()
    with ProcessPoolExecutor(max_workers=workers,initializer=vc.initialise_worker,initargs=(target_cache,)) as pool:
        futures={pool.submit(simulate,task):task for task in tasks}
        for number,future in enumerate(as_completed(futures),1):
            try:
                mode,group,seed,status=future.result()
            except Exception as error:
                print(f"FAILED {futures[future]}: {error}",flush=True)
                raise
            print(f"{number}/{len(tasks)} {mode} {group} seed={seed} {status}; {time.monotonic()-start:.1f}s",flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("variance_output/m_calibration"))
    parser.add_argument("--cache",type=Path,default=Path("variance_output/uniform/hydrodynamics.npz"))
    parser.add_argument("--workers",type=int,default=2)
    args=parser.parse_args()
    if args.workers<1:
        parser.error("use at least one worker")
    generate(args.output,args.cache,args.workers)
