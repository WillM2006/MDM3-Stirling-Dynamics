"""Refine wave synthesis alone: 800, 1600 and 3200 frequencies, unchanged hull."""
import argparse
import hashlib
import json
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from variance_convergence import (LENGTHS, SEEDS, SEED_START,
                                  model_fingerprint, initialise_worker, save_carrier,
                                  simulate_record)

GRIDS = [800, 1600, 3200]


def simulate(task):
    return simulate_record(task, lengths=LENGTHS)


def generate(previous, output, workers):
    from sea import Sea
    old = json.loads((previous / "protocol.json").read_text())
    if (old["source_fingerprint"] != model_fingerprint() or old["seeds"] != SEEDS
            or old["seed_start"] != SEED_START or old["usable_lengths"] != LENGTHS):
        raise ValueError("previous records do not match this refinement protocol")
    output.mkdir(parents=True, exist_ok=True)
    protocol = dict(old, grids=GRIDS, length_test_grids=GRIDS,
                    question="Are sensor spectral means and repeat-record variances stable under wave-grid refinement?",
                    reference="3200 is a comparison reference, not proven truth",
                    reused="800-frequency records: identical seeds, lengths, sources and processing",
                    decision="10% comparison band; report both agreement and Monte Carlo inconclusiveness; no inverse-based grid selection",
                    processing="same equations; memory-mapped time carriers share deterministic exponentials")
    path = output / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("output folder contains a different experiment protocol")
    path.write_text(json.dumps(protocol, indent=2) + "\n")
    shutil.copyfile(previous / "hydrodynamics.npz", output / "hydrodynamics.npz")
    hashes = {}
    for length in LENGTHS:
        name = f"n800_t{length}"
        (output / name).mkdir(exist_ok=True)
        for seed in range(SEED_START, SEED_START + SEEDS):
            source = previous / name / f"seed-{seed}.npz"
            hashes[f"{name}/{source.name}"] = hashlib.sha256(source.read_bytes()).hexdigest()
            shutil.copyfile(source, output / name / source.name)
    (output / "reused_sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    times = np.arange(2 * int((max(LENGTHS) + 300) / .1) + 1) * .05
    for grid in GRIDS[1:]:
        path = output / f"carrier-{grid}-{max(LENGTHS)+300}.npy"
        if not path.exists():
            sea = Sea(n_freq=grid, n_dir=36, Hs=1.7, Tp=8, gamma=3.3,
                      s=10, direction=np.pi / 2, seed=SEED_START)
            save_carrier(path, sea.omega, times)
            print(f"Prepared {grid}-frequency time carrier", flush=True)
    tasks = [(grid, seed, str(output)) for grid in GRIDS[1:]
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
    parser.add_argument("--previous", type=Path, default=Path("variance_output/convergence"))
    parser.add_argument("--output", type=Path, default=Path("variance_output/resolution"))
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("use at least one worker")
    generate(args.previous, args.output, args.workers)
