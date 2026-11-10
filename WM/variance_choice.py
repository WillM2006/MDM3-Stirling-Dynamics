"""Separate uniform frequency-count convergence from frequency-placement effects."""
import argparse
import hashlib
import inspect
import json
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid
from variance_convergence import (LENGTHS, SEEDS, SEED_START, model_fingerprint,
                                  initialise_worker, simulate_record, save_carrier)
from variance_uniform import uniform_grid, uniform_projection, time_transform

COUNTS = [800, 1600, 3200, 6400, 12800]
CASES = {f"uniform{n}": n for n in COUNTS} | {"equal3200": 3200, "half3200": 3200}
NEW_CASES = ["uniform800", "uniform1600", "uniform12800", "half3200"]


def hybrid_grid(sub, fraction=.5):
    """Interpolate bin edges and centres; integrate the same spectrum in each bin."""
    if not 0 <= fraction <= 1:
        raise ValueError("uniform fraction must be between zero and one")
    fine = np.linspace(*sub.omega_range, 5000) / (2 * np.pi)
    cumulative = cumulative_trapezoid(sub.jonswap(fine), fine, initial=0)
    equal_edges = np.interp(np.linspace(0, cumulative[-1], len(sub.f)+1), cumulative, fine)
    equal_edges[[0, -1]] = fine[[0, -1]]
    uniform_edges = np.linspace(fine[0], fine[-1], len(sub.f)+1)
    uniform_centres = (uniform_edges[:-1] + uniform_edges[1:]) / 2
    edges = (1-fraction)*equal_edges + fraction*uniform_edges
    centres = (1-fraction)*sub.f + fraction*uniform_centres
    if not np.all((centres >= edges[:-1]) & (centres <= edges[1:])):
        raise ValueError("interpolated centres lie outside their bins")
    energy = np.diff(np.interp(edges, fine, cumulative))
    sub.f = centres
    sub.delta_f = np.diff(edges)
    sub.omega = 2*np.pi*centres
    sub.k = sub.omega**2/sub.g
    sub.S_f = energy/sub.delta_f
    sub.directional_frequency_spectrum = sub.S_f[:, None]*sub.D[None, :]
    sub.wave_amplitudes = np.sqrt(2*energy[:, None]*sub.D[None, :]*sub.delta_theta)
    sub.Hs_actual = 4*np.sqrt(np.sum(sub.wave_amplitudes**2)/2)


def simulate(task):
    name, seed, output = task
    half = name == "half3200"
    result = simulate_record((CASES[name], seed, str(Path(output)/name)), lengths=LENGTHS,
                             grid_transform=hybrid_grid if half else uniform_grid,
                             projection=None if half else uniform_projection)
    return name, seed, result[2]


def generate(previous, output, workers):
    from sea import Sea
    old = json.loads((previous/"protocol.json").read_text())
    grid_hash = hashlib.sha256(inspect.getsource(uniform_grid).encode()).hexdigest()
    projection_hash = hashlib.sha256((inspect.getsource(uniform_projection)+inspect.getsource(time_transform)).encode()).hexdigest()
    if (old["source_fingerprint"] != model_fingerprint() or old["seeds"] != SEEDS
            or old["seed_start"] != SEED_START or old["usable_lengths"] != LENGTHS
            or old["grid_transform_sha256"] != grid_hash or old["projection_sha256"] != projection_hash):
        raise ValueError("existing uniform records do not match the comparison protocol")
    output.mkdir(parents=True, exist_ok=True)
    protocol = dict(old, grids=list(CASES), cases=CASES, uniform_counts=COUNTS,
                    length_test_grids=list(CASES),
                    frequency_grid="uniform count sweep; equal-energy, half and uniform placement at fixed count 3200",
                    processing="same physical equations; checked zoom sums for uniform grids, cached direct sums for the midpoint grid",
                    spacing_cases={"equal3200": 0., "half3200": .5, "uniform3200": 1.},
                    question="Which uniform frequency count is stable, and what does placement change at fixed count?",
                    candidate_count=6400, reference="Count comparisons use N versus 2N; placement uses uniform3200 as a comparison, not truth",
                    reused="equal3200, uniform3200 and uniform6400; same seeds and processing",
                    hybrid_transform_sha256=hashlib.sha256(inspect.getsource(hybrid_grid).encode()).hexdigest(),
                    hybrid_definition="convex combinations of equal-energy and uniform bin edges and centres; same integrated JONSWAP energy",
                    comparisons=[[f"uniform{n}", f"uniform{2*n}"] for n in COUNTS[:-1]],
                    decision="Preserve 10% band and report confidence; no forced plateau, fitted smoothing or inverse-based selection")
    path = output/"protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("output folder contains a different experiment protocol")
    path.write_text(json.dumps(protocol, indent=2)+"\n")
    shutil.copyfile(previous/"hydrodynamics.npz", output/"hydrodynamics.npz")
    hashes = {}
    for name in ["equal3200", "uniform3200", "uniform6400"]:
        source_name = "equal3200" if name == "equal3200" else str(CASES[name])
        for length in LENGTHS:
            target = output/name/f"n{CASES[name]}_t{length}"
            target.mkdir(parents=True, exist_ok=True)
            for seed in range(SEED_START, SEED_START+SEEDS):
                source = previous/f"n{source_name}_t{length}"/f"seed-{seed}.npz"
                hashes[f"{name}/{target.name}/{source.name}"] = hashlib.sha256(source.read_bytes()).hexdigest()
                shutil.copyfile(source, target/source.name)
    (output/"reused_sha256.json").write_text(json.dumps(hashes, indent=2)+"\n")
    carrier_path = output/"half3200"/f"carrier-3200-{max(LENGTHS)+300}.npy"
    carrier_path.parent.mkdir(exist_ok=True)
    if not carrier_path.exists():
        sub = Sea(n_freq=3200, n_dir=36, Hs=1.7, Tp=8, gamma=3.3, s=10,
                  direction=np.pi/2, seed=SEED_START)
        hybrid_grid(sub)
        times = np.arange(2*int((max(LENGTHS)+300)/.1)+1)*.05
        save_carrier(carrier_path, sub.omega, times)
    tasks = [(name, seed, str(output)) for name in NEW_CASES
             for seed in range(SEED_START, SEED_START+SEEDS)]
    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers, initializer=initialise_worker,
                             initargs=(output/"hydrodynamics.npz",)) as pool:
        futures = {pool.submit(simulate, task): task for task in tasks}
        for number, future in enumerate(as_completed(futures), 1):
            try:
                name, seed, status = future.result()
            except Exception as error:
                print(f"FAILED {futures[future]}: {error}", flush=True)
                raise
            print(f"{number}/{len(tasks)} {name} seed={seed} {status}; {time.monotonic()-start:.1f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, default=Path("variance_output/uniform"))
    parser.add_argument("--output", type=Path, default=Path("variance_output/choice"))
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("use at least one worker")
    generate(args.previous, args.output, args.workers)
