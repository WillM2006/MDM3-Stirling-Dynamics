"""Wave-grid and recording-length checks with the inverse held fixed.

Run generation with --generate, then analyse the saved spectral estimates.
Raw recordings and the hydrodynamic cache stay in the ignored output folder.
"""
import argparse
import copy
import csv
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.signal import lfilter
import invert_waves as iw
from variance_study import estimates, summarise

GRIDS = [40, 100, 200, 400, 800]
LENGTHS = [900, 1800, 3600]
SEED_START = 104000
SEEDS = 96
BOOTSTRAP = 1000
TOLERANCE = 0.10
ROOT = Path(__file__).resolve().parent


def model_fingerprint():
    return hashlib.sha256(b"".join((ROOT / name).read_bytes() for name in
                                  ["submarine.py", "sensors.py", "sea.py"])).hexdigest()


def prepare_cache(path):
    from sensors import ControlledSubmarine
    sub = ControlledSubmarine(Hs=1.7, Tp=8, s=10, gamma=3.3, direction=np.pi / 2)
    if not path.exists():
        sub.solve_hydrodynamics()
        names = ["A", "B", "A_inf", "F_exc", "M_inv", "A_sys"]
        np.savez(path, fingerprint=model_fingerprint(), **{k: getattr(sub, k) for k in names})
    return sub


def initialise_worker(cache):
    global sensors, sea, base, carrier, carrier_grid, checks_done
    import sensors
    import sea
    base = sensors.ControlledSubmarine(Hs=1.7, Tp=8, s=10, gamma=3.3, direction=np.pi / 2)
    with np.load(cache) as z:
        if str(z["fingerprint"]) != model_fingerprint():
            raise ValueError("hydrodynamic cache does not match the model sources")
        for key in ["A", "B", "A_inf", "F_exc", "M_inv", "A_sys"]:
            setattr(base, key, z[key])
    base.build_kernel()
    base.build_viscous_damping()
    base.build_controller()
    carrier, carrier_grid, checks_done = None, None, set()


def fast_gauss_markov(sigma, tau, fs, n):
    phi = np.exp(-1 / (fs * tau))
    first = sigma * sensors.rng.standard_normal()
    noise = sigma * np.sqrt(1 - phi**2) * sensors.rng.standard_normal(n)
    return np.r_[first, lfilter([1.], [1., -phi], noise[1:], zi=[phi * first])[0]]


def fast_pressure(sub, amplitudes, x, y, z, t, chunk=1000):
    if np.ptp(x) > 1e-12 or np.ptp(y) > 1e-12:
        raise ValueError("cached pressure requires the fixed mean sensor positions")
    key = (x[0], y[0], z)
    if key in getattr(sub, "pressure_cache", {}) and len(t) == len(sub.pressure_cache[key]):
        return sub.pressure_cache[key]
    coefficient = pressure_coefficient(sub, amplitudes, *key)
    waves = sub.time_carrier if len(t) == len(sub.time_carrier) else np.exp(-1j * t[:, None] * sub.w)
    return sensors.RHO * sensors.G * sensors.ramp(t) * np.real(waves @ coefficient)


def pressure_coefficient(sub, amplitudes, x, y, z):
    return np.sum(amplitudes * np.exp(sub.k[:, None] * z + 1j *
                  (sub.phi + sub.k[:, None] * (x * sub.cos_theta + y * sub.sin_theta))), axis=1)


def harmonic_projection(waves, coefficients):
    # Bound working memory: strided full-record pressure products can copy gigabytes.
    result = np.empty((len(waves), coefficients.shape[1]))
    for start in range(0, len(waves), 1024):
        window = slice(start, start + 1024)
        result[window] = np.real(waves[window] @ coefficients)
    return result


def check_projected_forcing(actual, expected, coefficients):
    # Scale roundoff by force/torque RMS; instantaneous cancellation is ill-conditioned.
    scale = np.maximum(np.sqrt(np.sum(abs(coefficients)**2, axis=0) / 2), 1.)
    np.testing.assert_allclose(actual / scale, expected / scale, rtol=2e-10, atol=1e-10)


def save_carrier(path, frequencies, times, chunk=1024):
    """Keep large, deterministic time carriers on disk, shared by workers."""
    temporary = path.with_suffix(".tmp")
    waves = np.lib.format.open_memmap(temporary, mode="w+", dtype=complex,
                                     shape=(len(times), len(frequencies)))
    for start in range(0, len(times), chunk):
        window = slice(start, start + chunk)
        waves[window] = np.exp(-1j * times[window, None] * frequencies)
    waves.flush()
    del waves
    temporary.replace(path)


def preprocess(measured):
    streams = {
        "attitude": np.rec.fromarrays([measured["t_att"], measured["pitch"], measured["roll"]],
                                      names="t_s,pitch_rad,roll_rad"),
        "accel": np.rec.fromarrays([measured["t_accel"], *measured["accel"].T, measured["vert_acc"]],
                                   names="t_s,f_x_m_s2,f_y_m_s2,f_z_m_s2,vert_acc_m_s2"),
        "pressure": np.rec.fromarrays([measured["t_pres"], *measured["pressure"].values()],
                                      names="t_s," + ",".join(f"P{i}_Pa" for i in range(1, 7))),
    }
    with patch.object(np, "genfromtxt", side_effect=lambda path, **kw: streams[Path(path).stem]):
        return iw.read_channels(".", iw.CHANNELS)


def simulate_record(task, lengths=None, grid_transform=None, projection=None):
    global carrier, carrier_grid, checks_done
    grid, seed, output = task
    if lengths is None:
        lengths = LENGTHS if grid in (40, 800) else [900]
    targets = [Path(output) / f"n{grid}_t{length}" / f"seed-{seed}.npz" for length in lengths]
    if all(path.exists() for path in targets):
        return grid, seed, "cached"
    sub = copy.copy(base)
    sub.params = base.params.copy()
    sea.Sea.__init__(sub, n_freq=grid, n_dir=36, Hs=1.7, Tp=8, gamma=3.3,
                     s=10, direction=np.pi / 2, seed=seed)
    if grid_transform is not None:
        grid_transform(sub)
    sub.build_excitation()
    duration = max(lengths) + 300
    half_time = np.arange(2 * int(duration / .1) + 1) * .05
    if projection is not None:
        carrier, carrier_grid = None, None
    elif carrier_grid != (grid, duration):
        saved = Path(output) / f"carrier-{grid}-{duration}.npy"
        carrier = (np.load(saved, mmap_mode="r") if saved.exists() else
                   np.exp(-1j * half_time[:, None] * sub.w))
        if carrier.shape != (len(half_time), grid) or carrier.dtype != np.complex128:
            raise ValueError("time carrier does not match the simulation grid")
        carrier_grid = grid, duration
    if projection is not None or isinstance(carrier, np.memmap):
        positions = [sub.centre + [0, 0, -.2] + r for r in
                     [*sensors.PRESSURE_SENSORS.values(), sensors.R_DEPTH]]
        coefficients = np.column_stack([sub.F_hat.sum(1), *[
            pressure_coefficient(sub, sub.wave_amplitudes, *position) for position in positions]])
        projected = (projection(sub.w, half_time, coefficients) if projection is not None
                     else harmonic_projection(carrier, coefficients))
        forcing = sensors.ramp(half_time)[:, None] * projected[:, :6]
        pressures = sensors.RHO * sensors.G * sensors.ramp(half_time[::2])[:, None] * projected[::2, 6:]
        sub.pressure_cache = {tuple(position): pressures[:, i] for i, position in enumerate(positions)}
    else:
        forcing = sensors.ramp(half_time)[:, None] * np.real(carrier @ sub.F_hat.sum(1))
    original = sub.excitation
    for index in [1, 1201, len(half_time) // 2, len(half_time) - 1]:
        expected = original(half_time[index], 0, 0)
        if projection is None:
            np.testing.assert_allclose(forcing[index], expected, rtol=2e-10, atol=1e-5)
        else:
            check_projected_forcing(forcing[index], expected, sub.F_hat.sum(1))
    sub.excitation = lambda t, x, y: forcing[int(round(t / .05))]
    sub.time_carrier = carrier[::2] if carrier is not None else []
    t, history = sub.simulate(duration)
    if not np.isfinite(history).all():
        raise ValueError("nonfinite hull trajectory")
    if grid not in checks_done:
        tt = t[3000:3010]
        x, y = np.full(len(tt), 11.), np.full(len(tt), 2.)
        np.testing.assert_allclose(fast_pressure(sub, sub.wave_amplitudes, x, y, -15, tt),
                                   sensors.incident_pressure(sub, sub.wave_amplitudes, x, y, -15, tt), rtol=1e-10, atol=1e-7)
        for position, pressure in getattr(sub, "pressure_cache", {}).items():
            x, y = np.full(len(tt), position[0]), np.full(len(tt), position[1])
            np.testing.assert_allclose(pressure[3000:3010], sensors.incident_pressure(
                sub, sub.wave_amplitudes, x, y, position[2], tt), rtol=1e-10, atol=1e-7)
        sensors.rng = np.random.default_rng(seed)
        original_noise = sensors.gauss_markov(20, 900, 10, 1000)
        sensors.rng = np.random.default_rng(seed)
        np.testing.assert_allclose(fast_gauss_markov(20, 900, 10, 1000), original_noise, atol=1e-10)
        checks_done.add(grid)
    with patch.object(sensors, "incident_pressure", fast_pressure):
        true = sensors.true_signals(sub, sub.wave_amplitudes, t, history)
    sensors.rng = np.random.default_rng(seed)
    with patch.object(sensors, "gauss_markov", fast_gauss_markov):
        measured = sensors.measure_all(t, true)
    tc, samples = preprocess(measured)
    for length, path in zip(lengths, targets):
        x = samples[:int(length * iw.FS)]
        if len(x) != int(length * iw.FS):
            raise ValueError("recording too short")
        f, observed, predicted = estimates(x)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, f=f, observed=observed, **predicted)
        temporary.replace(path)
    return grid, seed, "generated"


def generate(output, workers, seeds):
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "hydrodynamics.npz"
    prepare_cache(cache)
    # This protocol is saved before any new simulation is evaluated.
    protocol = dict(base_main="caaaf415dbe7bd372187758fe542c69999913d3e", grids=GRIDS,
                    usable_lengths=LENGTHS, length_test_grids=[40, 800], seed_start=SEED_START,
                    seeds=seeds, bootstrap=BOOTSTRAP, tolerance=TOLERANCE,
                    fixed_settings=dict(Hs=1.7, Tp=8, gamma=3.3, s=10, direction_rad=float(np.pi/2),
                                        depth_m=15, length_m=90, radius_m=5.36, speed_m_s=0,
                                        generation_directions=36, hydrodynamic_frequencies=40,
                                        settle_s=300, sample_rate_Hz=2, welch_seconds=200),
                    inverse_weights="unchanged; no inverse fitting or truth-dependent variance rules",
                    length_comparison="nested prefixes of the same preprocessed long record; noise and phases paired",
                    processing="in-memory native sensor streams; exact cached forcing, pressure sums and Gaussian recurrence",
                    source_fingerprint=model_fingerprint())
    protocol_path = output / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("output folder contains a different experiment protocol")
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    tasks = [(grid, SEED_START + i, str(output)) for grid in GRIDS for i in range(seeds)]
    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers, initializer=initialise_worker, initargs=(cache,)) as pool:
        futures = [pool.submit(simulate_record, task) for task in tasks]
        for number, future in enumerate(as_completed(futures), 1):
            grid, seed, status = future.result()
            print(f"{number}/{len(tasks)} n={grid} seed={seed} {status}; {time.monotonic()-start:.1f}s", flush=True)


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("variance_output/convergence"))
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seeds", type=int, default=SEEDS)
    args = parser.parse_args()
    if args.seeds < 8 or args.workers < 1:
        parser.error("use at least eight seeds and one worker")
    if args.generate:
        generate(args.output, args.workers, args.seeds)


if __name__ == "__main__":
    main()
