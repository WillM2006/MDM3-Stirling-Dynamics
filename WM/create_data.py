import argparse
import contextlib
import io
import os
import time
import json
import hashlib
from pathlib import Path
import numpy as np
from tqdm import tqdm
from sea import Sea
from submarine import Submarine
from sensors import create_sensor_data

'''
two modes, multiple seeds per run folder (sensordata_{seed} inside each folder):
  --mode states (default): 30 realistic sea states (wind / swell / mixed wind+swell / two-direction bimodal),
                           headings over 0-360 deg -> 600 runs
  --mode sweep:            one-at-a-time sweep of each param around BASELINE

WARNING  Mesh resolution for 28 problems:                                                                                                                                                          
The resolution of the mesh might be insufficient for omega ranging from 1.854 to 2.000.                                                                                                   
This warning appears because the largest panel of the mesh has radius (2.354 m) > wavelength/8 (1.926 to 2.242 m).    
'''


# CONSTS
N_FREQ = 6400 # sea wave components, evenly spaced (UNIFORM_FREQUENCIES) so the fast FFT wave synthesis can be used
N_BEM = 100    # BEM frequency grid, independent of N_FREQ (the sea's wave components); see Submarine(n_bem=...)
UNIFORM_FREQUENCIES = True
N_DIR = 72
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


# ---------------------------------------------------------------- sea states (--mode states)
# directions in degrees (0-360), the direction the waves travel TOWARDS, from the bow (+x) towards port (+y)
# wind seas: s = 1.5 (DNV cos^n, n = 3), Tp inside DNV 3.6*sqrt(Hs)-5*sqrt(Hs) and below Tpf = 6.6*Hs^(1/3)
# swells:    s = 6 (Goda s_max 25), Tp above Tpf
# gamma from Torsethaugen's wind-sea / swell formulas (via WAFO), rounded. See WB.md for the sources.

# wind seas
A = dict(Hs=1.0, Tp=5.0,  gamma=3.5, s=1.5, direction=15)    # WMO SS3 slight (~18% of the tail is above omega=2.0, Hs_actual ~0.9)
B = dict(Hs=1.5, Tp=6.0,  gamma=2.8, s=1.5, direction=70)    # SS4 moderate
C = dict(Hs=2.0, Tp=6.5,  gamma=2.6, s=1.5, direction=120)
D = dict(Hs=2.5, Tp=7.0,  gamma=2.4, s=1.5, direction=165)
E = dict(Hs=3.0, Tp=8.0,  gamma=2.0, s=1.5, direction=225)   # Nielsen (2007) wind sea, SS5 rough
F = dict(Hs=4.0, Tp=9.0,  gamma=1.9, s=1.5, direction=280)
G = dict(Hs=5.0, Tp=10.0, gamma=1.9, s=1.5, direction=330)   # SS6 very rough

# swell seas
H = dict(Hs=0.5, Tp=10.0, gamma=5.9, s=6,  direction=45)
I = dict(Hs=1.0, Tp=12.0, gamma=5.9, s=6,  direction=100)
J = dict(Hs=1.5, Tp=13.0, gamma=5.4, s=6,  direction=150)
K = dict(Hs=2.0, Tp=14.0, gamma=5.0, s=6,  direction=195)    # Nielsen (2007) swell
L = dict(Hs=3.0, Tp=15.0, gamma=4.7, s=6,  direction=250)
M = dict(Hs=2.5, Tp=16.0, gamma=5.7, s=6,  direction=300)
N = dict(Hs=2.0, Tp=14.0, gamma=5.0, s=19, direction=345)    # long-travelled narrow swell (Goda s_max 75)

# mixed seas: a list of wave systems, each keeps its own Hs/Tp/gamma/s (bimodal in frequency AND direction),
# only the direction is set here. Total Hs = sqrt(sum Hs_i^2)
O = [dict(E, direction=15),  dict(K, direction=225)]   # Nielsen I
P = [dict(E, direction=180), dict(K, direction=225)]   # Nielsen J
Q = [dict(E, direction=105), dict(K, direction=105)]   # Nielsen K, aligned
R = [dict(B, direction=60),  dict(I, direction=150)]   # 90 deg crossing, moderate
S = [dict(C, direction=300), dict(J, direction=240)]   # 60 deg
T = [dict(D, direction=200), dict(L, direction=20)]    # opposing, swell-dominated
U = [dict(F, direction=130), dict(H, direction=250)]   # 120 deg, wind-dominated, weak swell
V = [dict(A, direction=340), dict(M, direction=85)]    # 105 deg, light wind on big swell
W = [dict(G, direction=270), dict(K, direction=310)]   # 40 deg, storm wind sea + swell
X = [dict(B, direction=165), dict(N, direction=300)]   # 135 deg, narrow swell
Y = [dict(I, direction=35),  dict(M, direction=125)]   # two swells from different storms, 90 deg

# bimodal in direction only: one sea (same Tp/gamma) split over two directions. Hs_i = Hs * sqrt(power share), so the
# total Hs and the split are unchanged (e.g. E 50/50: Hs 3.0 * sqrt(0.5) = 2.12 each; giving each the full Hs would
# double the energy). Five are used, the rest are kept commented to swap in
BI_O = [dict(E, Hs=2.12, direction=15),  dict(E, Hs=2.12, direction=105)]            # wind sea crossing at 90 deg, 50/50
BI_P = [dict(E, Hs=2.51, direction=180), dict(E, Hs=1.64, direction=225)]            # wind veered 45 deg, 70/30
BI_Q = [dict(K, Hs=1.41, direction=15),  dict(K, Hs=1.41, direction=225)]            # two swells 150 deg apart, 50/50
BI_T = [dict(D, Hs=1.94, direction=200), dict(D, Hs=1.58, direction=20)]             # opposing (reflected sea), 60/40
BI_X = [dict(K, Hs=1.41, direction=165), dict(K, Hs=1.41, s=19, direction=300)]      # normal + narrow swell, 135 deg
# BI_R = [dict(B, Hs=1.16, direction=60),  dict(B, Hs=0.95, direction=150)]            # moderate, 90 deg, 60/40
# BI_S = [dict(C, Hs=1.41, direction=300), dict(C, Hs=1.41, direction=240)]            # 60 deg, 50/50
# BI_U = [dict(F, Hs=3.58, direction=130), dict(F, Hs=1.79, direction=250)]            # weak secondary, 120 deg, 80/20
# BI_V = [dict(M, Hs=1.77, direction=340), dict(M, Hs=1.77, direction=85)]             # long swell, 105 deg, 50/50
# BI_W = [dict(G, Hs=3.74, direction=270), dict(G, Hs=2.74, direction=310)]            # storm, 40 deg, 70/30
# BI_Y = [dict(I, Hs=0.71, direction=35),  dict(I, Hs=0.71, direction=125)]            # low swell, 90 deg, 50/50

SEA_STATES = dict(A=A, B=B, C=C, D=D, E=E, F=F, G=G,
                  H=H, I=I, J=J, K=K, L=L, M=M, N=N,
                  O=O, P=P, Q=Q, R=R, S=S, T=T, U=U, V=V, W=W, X=X, Y=Y,
                  BI_O=BI_O, BI_P=BI_P, BI_Q=BI_Q, BI_T=BI_T, BI_X=BI_X)
SEA_TYPE = {**dict.fromkeys("ABCDEFG", "wind"), **dict.fromkeys("HIJKLMN", "swell"), **dict.fromkeys("OPQRSTUVWXY", "mixed"),
            **dict.fromkeys(["BI_O", "BI_P", "BI_Q", "BI_T", "BI_X"], "bimodal")}
SEA_DEPTH = 15


def sea_state_params(state):
    """table entry (one system dict, or a list of them; directions in degrees) -> Submarine kwargs
    systems=[...] with directions in radians. The sea is recorded in each run's systems.csv"""
    systems = state if isinstance(state, (list, tuple)) else [state]
    return dict(systems=[dict(q, direction=float(np.deg2rad(q["direction"]))) for q in systems])


# BEM cache: A, B, F_exc only depend on the mesh, depth, speed and the omega/theta grids (theta is always the full circle),
# so the sea (systems) and seed all reuse the same solve - one solve per depth
# (sweep mode: 5 solves instead of 500; states mode: depth is fixed, so 1)
BEM_ATTRS = ("A", "B", "A_inf", "F_exc", "M_inv", "A_sys")
_bem_cache = {}
_solve_hydrodynamics = Submarine.solve_hydrodynamics

def cached_solve_hydrodynamics(self):
    p = self.params
    key = (p["depth"], p["forward_speed"], p["n_bem"], p["n_dir"], tuple(p["omega_range"]))   # the BEM depends on n_bem, not n_freq
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


def run_list(out_root, mode="states"):
    """every (run folder, create_sensor_data kwargs) of the chosen mode, in run order"""
    runs = []
    if mode == "states":
        for name, state in SEA_STATES.items():
            filedir = f"seastate_{name}_{SEA_TYPE[name]}"
            for seed in SEED_VALUES:
                params = dict(n_freq=N_FREQ, n_bem=N_BEM, uniform_frequencies=UNIFORM_FREQUENCIES, n_dir=N_DIR,
                              omega_range=OMEGA_RANGE, seed=seed, depth=SEA_DEPTH, **sea_state_params(state))
                runs.append((os.path.join(out_root, filedir, f"sensordata_{seed}"), params))
        return runs
    if mode != "sweep":
        raise ValueError(f"mode must be 'states' or 'sweep', got {mode!r}")

    for swept in make_combos():
        # direction is simulated in radians but named in degrees (pi/2 -> direction90)
        filedir = "sensordata_" + "_".join(f"{k}{round(float(np.rad2deg(v)), 3):g}" if k == "direction" else f"{k}{v}"
                                           for k, v in swept.items())
        for seed in SEED_VALUES:
            params = dict(n_freq=N_FREQ, n_bem=N_BEM, uniform_frequencies=UNIFORM_FREQUENCIES, n_dir=N_DIR,
                          omega_range=OMEGA_RANGE, seed=seed, **swept)
            runs.append((os.path.join(out_root, filedir, f"sensordata_{seed}"), params))
    return runs


def check_generation(out_root, runs):
    """Do not resume a dataset made with different settings or generator code."""
    root = Path(out_root)
    path = root / "generation.json"
    source = Path(__file__).resolve().parent
    expected = dict(t_sim=T_SIM, runs=[p for _, p in runs],
                    sources={name: hashlib.sha256((source/name).read_bytes()).hexdigest()
                             for name in ("sea.py", "submarine.py", "sensors.py", "create_data.py")})
    expected = json.loads(json.dumps(expected))
    if path.exists():
        if json.loads(path.read_text()) != expected:
            raise SystemExit("Dataset settings or generator code differ; use a new --out folder.")
    elif root.exists() and any(root.iterdir()):
        raise SystemExit("Output folder has no generation manifest; use a new empty --out folder.")
    else:
        root.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(expected, indent=2) + "\n")


def is_done(run_dir):
    return os.path.exists(os.path.join(run_dir, DONE_FLAG))


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
    ap = argparse.ArgumentParser(description="Sensor data generation: 30 sea states (default) or the one-at-a-time sweep. "
                                             "Re-running resumes: finished runs are skipped, interrupted ones are redone.")
    ap.add_argument("--mode", choices=("states", "sweep"), default="states",
                    help="states: 30 wind / swell / mixed / bimodal sea states (600 runs); sweep: one-at-a-time sweep (500 runs)")
    ap.add_argument("--out", default=None, help="new folder for the dataset "
                                                "(default: sensor_data_states, or sensor_data_dense for --mode sweep)")
    ap.add_argument("--status", action="store_true", help="only show progress, don't run anything")
    ap.add_argument("--force", action="store_true", help="redo every run, including finished ones")
    args = ap.parse_args()
    if args.out is None:
        args.out = "sensor_data_dense" if args.mode == "sweep" else "sensor_data_states"

    runs = run_list(args.out, args.mode)
    todo = [(d, p) for d, p in runs if args.force or not is_done(d)]
    check_generation(args.out, runs)
    n_done = len(runs) - len(todo)
    print(f"{n_done}/{len(runs)} runs finished, {len(todo)} to do   (output folder: {os.path.abspath(args.out)})")

    if args.status:
        print_status(runs, args.out)
        raise SystemExit

    # progress bar over all runs (finished ones count from the start), time left from the average run so far
    # (smoothing=0: the first run at each depth also does the BEM solve, so a recent-runs average would jump about)
    bar = tqdm(total=len(runs), initial=n_done, unit="run", smoothing=0, dynamic_ncols=True)
    for run_dir, params in todo:
        name = os.path.relpath(run_dir, args.out)
        bar.set_postfix_str(name)
        t_run = time.time()
        flag = Path(run_dir) / DONE_FLAG
        flag.unlink(missing_ok=True)
        with contextlib.redirect_stdout(io.StringIO()):  # create_sensor_data's own prints would break up the bar
            create_sensor_data(out_dir=run_dir, t_sim=T_SIM, **params)
        with open(os.path.join(run_dir, DONE_FLAG), "w") as f:
            f.write(f"finished {time.strftime('%Y-%m-%d %H:%M:%S')} in {time.time() - t_run:.1f} s\n")
        bar.write(f"done {name}  {time.time() - t_run:.1f} s")
        bar.update(1)
    bar.close()

    print(f"{args.mode} complete: {len(runs)}/{len(runs)} runs in {os.path.abspath(args.out)}")
