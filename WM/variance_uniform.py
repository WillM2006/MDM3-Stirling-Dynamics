"""Compare uniform wave-frequency grids at 3200 and 6400 components."""
import argparse
import hashlib
import inspect
import json
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.signal import ZoomFFT
from variance_convergence import (LENGTHS, SEEDS, SEED_START, model_fingerprint,
                                  initialise_worker, simulate_record)

GRIDS = [3200, 6400]


def uniform_grid(sub):
    """Move bin centres, retaining the original JONSWAP CDF and total energy."""
    fine = np.linspace(*sub.omega_range, 5000) / (2 * np.pi)
    cumulative = cumulative_trapezoid(sub.jonswap(fine), fine, initial=0)
    edges = np.linspace(fine[0], fine[-1], len(sub.f) + 1)
    energy = np.diff(np.interp(edges, fine, cumulative))
    sub.f = (edges[:-1] + edges[1:]) / 2
    sub.delta_f = np.diff(edges)
    sub.omega = 2 * np.pi * sub.f
    sub.k = sub.omega**2 / sub.g
    sub.S_f = energy / sub.delta_f
    sub.directional_frequency_spectrum = sub.S_f[:, None] * sub.D[None, :]
    sub.wave_amplitudes = np.sqrt(2 * energy[:, None] * sub.D[None, :] * sub.delta_theta)
    sub.Hs_actual = 4 * np.sqrt(np.sum(sub.wave_amplitudes**2) / 2)


@lru_cache(maxsize=2)
def time_transform(n, count, step):
    return ZoomFFT(n, [0, count * step], m=count, fs=2 * np.pi)


def uniform_projection(frequencies, times, coefficients):
    # Evaluate the same harmonic sum using a zoom FFT, without a huge time carrier.
    spacing = (frequencies[-1] - frequencies[0]) / (len(frequencies) - 1)
    dt = times[1] - times[0]
    if (times[0] != 0 or not np.allclose(np.diff(frequencies), spacing, rtol=0, atol=2e-15)
            or not np.allclose(np.diff(times), dt, rtol=0, atol=1e-11)):
        raise ValueError("zoom synthesis requires uniform frequencies and times starting at zero")
    values = time_transform(len(frequencies), len(times), spacing * dt)(coefficients, axis=0)
    return np.real(np.exp(-1j * frequencies[0] * times)[:, None] * values)


def simulate(task):
    return simulate_record(task, lengths=LENGTHS, grid_transform=uniform_grid,
                           projection=uniform_projection)


def generate(previous, output, workers):
    old = json.loads((previous / "protocol.json").read_text())
    if (old["source_fingerprint"] != model_fingerprint() or old["seeds"] != SEEDS
            or old["seed_start"] != SEED_START or old["usable_lengths"] != LENGTHS):
        raise ValueError("previous records do not match this comparison protocol")
    output.mkdir(parents=True, exist_ok=True)
    protocol = dict(old, grids=["equal3200", *GRIDS], length_test_grids=GRIDS,
                    question="Does uniform frequency coverage stabilise sensor spectral means and variances?",
                    frequency_grid="uniform bin centres across the original omega range 0.1--2 rad/s",
                    bin_weights="integrate the same 5000-point JONSWAP CDF as main; retain total energy and direction weights",
                    grid_transform_sha256=hashlib.sha256(inspect.getsource(uniform_grid).encode()).hexdigest(),
                    projection_sha256=hashlib.sha256((inspect.getsource(uniform_projection) + inspect.getsource(time_transform)).encode()).hexdigest(),
                    comparisons=[["equal3200", 3200], [3200, 6400]],
                    reference="6400 is a refinement comparison, not proven truth",
                    reused="equal-energy 3200-frequency records, identical seeds and preprocessing",
                    processing="same physical equations; uniform harmonic sums evaluated by checked zoom FFT",
                    decision="10% comparison band; distinguish agreement, disagreement and insufficient precision; no inverse fitting")
    path = output / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("output folder contains a different experiment protocol")
    path.write_text(json.dumps(protocol, indent=2) + "\n")
    shutil.copyfile(previous / "hydrodynamics.npz", output / "hydrodynamics.npz")
    hashes = {}
    for length in LENGTHS:
        target = output / f"nequal3200_t{length}"
        target.mkdir(exist_ok=True)
        for seed in range(SEED_START, SEED_START + SEEDS):
            source = previous / f"n3200_t{length}" / f"seed-{seed}.npz"
            hashes[f"{target.name}/{source.name}"] = hashlib.sha256(source.read_bytes()).hexdigest()
            shutil.copyfile(source, target / source.name)
    (output / "reused_sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    tasks = [(grid, seed, str(output)) for grid in GRIDS
             for seed in range(SEED_START, SEED_START + SEEDS)]
    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers, initializer=initialise_worker,
                             initargs=(output / "hydrodynamics.npz",)) as pool:
        futures = [pool.submit(simulate, task) for task in tasks]
        for number, future in enumerate(as_completed(futures), 1):
            grid, seed, status = future.result()
            print(f"{number}/{len(tasks)} n={grid} seed={seed} {status}; {time.monotonic()-start:.1f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, default=Path("variance_output/resolution"))
    parser.add_argument("--output", type=Path, default=Path("variance_output/uniform"))
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("use at least one worker")
    generate(args.previous, args.output, args.workers)
