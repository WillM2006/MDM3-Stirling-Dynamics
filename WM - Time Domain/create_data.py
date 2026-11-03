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
DIRECTION_VALUES = [0, 45, 90, 135, 180]
GAMMA_VALUES = [1, 2, 3.3, 5, 7]
SPREAD_VALUES = [2, 5, 10, 25, 75]
DEPTH_VALUES = [10, 15, 25, 40, 60]

SEED_VALUES = range(20)

# held fixed while one param at a time is varied through its list above
BASELINE = dict(Hs=1.7, Tp=8, s=10, gamma=3.3, depth=15, direction=90)


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


if __name__ == "__main__":

    for swept in make_combos():
        filedir = "sensordata_" + "_".join(f"{k}{v}" for k, v in swept.items())

        for seed in SEED_VALUES:
            filename = f"sensordata_{seed}"

            params = dict(n_freq=N_FREQ, n_dir=N_DIR, omega_range=OMEGA_RANGE, seed=seed, **swept)
            create_sensor_data(out_dir=f"{filedir}/{filename}", t_sim=T_SIM, **params)
