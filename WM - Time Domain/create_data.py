import argparse
import os
import time
import numpy as np
from sea import Sea
from submarine import Submarine
from sensors import create_sensor_data

'''
multiple seeds per parameter combination - store in same folder with each folder inside is seed

WARNING  Mesh resolution for 28 problems:                                                                                                                                                          
The resolution of the mesh might be insufficient for omega ranging from 1.854 to 2.000.                                                                                                   
This warning appears because the largest panel of the mesh has radius (2.354 m) > wavelength/8 (1.926 to 2.242 m).    
'''


# CONSTS
N_FREQ = 40
N_DIR = 36
OMEGA_RANGE = (0.1, 2.0)
T_SIM = 1200.0


# params to test
HS_VALUES = [0.5, 1, 1.7, 2.5, 4]
TP_VALUES = [4.5, 6, 8, 10, 12]
DIRECTION_VALUES = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
GAMMA_VALUES = [1, 2, 3.3, 5, 7]
SPREAD_VALUES = [2, 5, 10, 25, 75]
DEPTH_VALUES = [10, 15, 25, 40, 60]

SEED_VALUES = range(20)

# held fixed while one param at a time is varied through its list above
BASELINE = dict(Hs=1.7, Tp=8, s=10, gamma=3.3, depth=15, direction=np.pi/2)


# BEM cache: A, B, F_exc only depend on the mesh, depth, speed and the omega/theta grids (theta is always the full circle),
# so Hs, Tp, gamma, s, direction and seed all reuse the same solve - 5 solves (one per depth) instead of 500
BEM_ATTRS = ("A", "B", "A_inf", "F_exc", "M_inv", "A_sys")
_bem_cache = {}
_solve_hydrodynamics = Submarine.solve_hydrodynamics

def cached_solve_hydrodynamics(self):
    p = self.params
    key = (p["depth"], p["forward_speed"], p["n_freq"], p["n_dir"], tuple(p["omega_range"]))
    if key not in _bem_cache:
        _solve_hydrodynamics(self)
        _bem_cache[key] = {a: getattr(self, a) for a in BEM_ATTRS}
    for a, v in _bem_cache[key].items():
        setattr(self, a, v)

Submarine.solve_hydrodynamics = cached_solve_hydrodynamics # ControlledSubmarine in sensors.py inherits this


def make_combos():
    """one-at-a-time sweep: every param sits at BASELINE while one is varied through its list"""
    sweeps = dict(Hs=HS_VALUES, Tp=TP_VALUES, s=SPREAD_VALUES, gamma=GAMMA_VALUES, depth=DEPTH_VALUES, direction=DIRECTION_VALUES)

    combos = [BASELINE]
    for k, values in sweeps.items():
        combos += [{**BASELINE, k: v} for v in values if v != BASELINE[k]] # baseline value already included once
    return combos


# resume / progress: a run gets DONE_FLAG written only after all its csvs are saved, so a run without it
# (never started, or interrupted part way) is (re)done next time and finished runs are skipped
DONE_FLAG = "done.flag"


def run_list(out_root):
    """every (run folder, create_sensor_data kwargs) of the sweep, in run order"""
    runs = []
    for swept in make_combos():
        # direction is simulated in radians but named in degrees (pi/2 -> direction90)
        filedir = "sensordata_" + "_".join(f"{k}{round(float(np.rad2deg(v)), 3):g}" if k == "direction" else f"{k}{v}"
                                           for k, v in swept.items())
        for seed in SEED_VALUES:
            params = dict(n_freq=N_FREQ, n_dir=N_DIR, omega_range=OMEGA_RANGE, seed=seed, **swept)
            runs.append((os.path.join(out_root, filedir, f"sensordata_{seed}"), params))
    return runs


def is_done(run_dir):
    return os.path.exists(os.path.join(run_dir, DONE_FLAG))


def hms(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def print_status(runs, out_root):
    """per parameter combination: finished seeds, and any run folder left unfinished (interrupted)"""
    by_combo = {}
    for run_dir, _ in runs:
        by_combo.setdefault(os.path.dirname(run_dir), []).append(run_dir)
    for combo, dirs in by_combo.items():
        done = sum(is_done(d) for d in dirs)
        broken = [os.path.basename(d) for d in dirs if os.path.isdir(d) and not is_done(d)]
        mark = "done" if done == len(dirs) else f"{done}/{len(dirs)}"
        print(f"  {mark:>6}  {os.path.relpath(combo, out_root)}" + (f"   (unfinished: {', '.join(broken)})" if broken else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="One-at-a-time sensor data sweep. Re-running resumes: finished runs are skipped, "
                                             "interrupted ones are redone.")
    ap.add_argument("--out", default=".", help="folder the sensordata_* folders go in (default: the current folder)")
    ap.add_argument("--status", action="store_true", help="only show progress, don't run anything")
    ap.add_argument("--force", action="store_true", help="redo every run, including finished ones")
    args = ap.parse_args()

    runs = run_list(args.out)
    todo = [(d, p) for d, p in runs if args.force or not is_done(d)]
    n_done = len(runs) - len(todo)
    print(f"{n_done}/{len(runs)} runs finished, {len(todo)} to do   (output folder: {os.path.abspath(args.out)})")

    if args.status:
        print_status(runs, args.out)
        raise SystemExit

    t_start = time.time()
    for i, (run_dir, params) in enumerate(todo, 1):
        t_run = time.time()
        create_sensor_data(out_dir=run_dir, t_sim=T_SIM, **params)
        with open(os.path.join(run_dir, DONE_FLAG), "w") as f:
            f.write(f"finished {time.strftime('%Y-%m-%d %H:%M:%S')} in {time.time() - t_run:.1f} s\n")

        elapsed = time.time() - t_start
        eta = elapsed / i * (len(todo) - i) # average so far, the first run at each depth also does the BEM solve
        print(f"[{n_done + i}/{len(runs)}] {os.path.relpath(run_dir, args.out)}  {time.time() - t_run:.1f} s"
              f"  | elapsed {hms(elapsed)}, ETA {hms(eta)}", flush=True)

    print(f"sweep complete: {len(runs)}/{len(runs)} runs in {os.path.abspath(args.out)}")
