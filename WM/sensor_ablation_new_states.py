"""Run resumable NNLS sensor ablations on sensor_data_states.

Examples (from repository root):
  python WM/sensor_ablation_new_states.py --pilot
  python WM/sensor_ablation_new_states.py --reps 20
"""
import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import pandas as pd
from pathlib import Path

import numpy as np

EXPERIMENTS = {
    "Four_pressure": ["P1", "P2", "P3", "P4"],
}

PILOT = ("seastate_E_wind", "seastate_L_swell", "seastate_P_mixed", "seastate_BI_O_bimodal")
FIELDS = ["scenario", "sea_type", "repetition", "experiment", "channels", "Hs_true", "Hs_est", "Hs_error_pct", "Tp_est", "negative_power_pct", "direction_est_deg", "circular_concentration", "status"]


def read_hs(params_path):
    with params_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1 or "Hs_actual" not in rows[0]:
        raise ValueError(f"Expected one params.csv row with Hs_actual: {params_path}")
    hs = float(rows[0]["Hs_actual"])
    if not np.isfinite(hs) or hs <= 0:
        raise ValueError(f"Invalid Hs_actual: {hs}")
    return hs


def direction_summary(path):
    with np.load(path) as z:
        x = np.asarray(z["x"])
        beta = np.asarray(z["beta"]).reshape(-1)
    if x.ndim != 2 or x.shape[1] != len(beta) or not np.isfinite(x).all() or not np.isfinite(beta).all():
        raise ValueError("Invalid directional spectrum")
    if np.max(np.abs(beta)) > 2 * np.pi + 0.01:
        raise ValueError("Expected heading angles in radians")
    power = np.maximum(x, 0).sum(axis=0)
    if power.sum() <= 0:
        return "", ""
    c = np.sum(power * np.cos(beta))
    s = np.sum(power * np.sin(beta))
    concentration = float(np.hypot(c, s) / power.sum())
    heading = float(np.degrees(np.arctan2(s, c))) if concentration > 1e-12 else ""
    return heading, concentration


def save_rows(path, rows):
    temp = path.with_suffix(".csv.tmp")
    with temp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows[k] for k in sorted(rows))
    temp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--cache", type=Path, default=None)
    p.add_argument("--pilot", action="store_true", help="One recording from each of four representative scenarios")
    p.add_argument("--reps", type=int, default=20, help="Maximum recordings per scenario (full mode)")
    p.add_argument("--tau", type=float, default=5.0)
    p.add_argument("--meff", type=float, default=2.5)
    p.add_argument("--model", choices=["capytaine", "six_dof"], default="capytaine")
    args = p.parse_args()
    if args.reps < 1 or args.tau < 0 or args.meff <= 0:
        p.error("--reps and --meff must be positive; --tau must be nonnegative")

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    data_root = (args.data_root or repo_root / "sensor_data_states").expanduser().resolve()
    output = (args.output or script_dir / "ablation_new_states_nnls").expanduser().resolve()
    cache = (args.cache or script_dir / "rao_cache_new.npz").expanduser().resolve()
    inverse = script_dir / "invert_waves_positive.py"
    if not data_root.is_dir() or not inverse.is_file():
        p.error(f"Missing data root ({data_root}) or inversion script ({inverse})")
    scenarios = sorted(q for q in data_root.glob("seastate_*") if q.is_dir())
    if args.pilot:
        scenarios = [data_root / name for name in PILOT]
        missing = [str(q) for q in scenarios if not q.is_dir()]
        if missing:
            p.error("Missing pilot scenarios: " + ", ".join(missing))
    if not scenarios:
        p.error("No seastate_* folders found")

    output.mkdir(parents=True, exist_ok=True)
    (output / "spectra").mkdir(exist_ok=True)
    (output / "logs").mkdir(exist_ok=True)
    workdir = output / "working"
    workdir.mkdir(exist_ok=True)
    result_npz = workdir / "inversion_result.npz"
    results_csv = output / "results.csv"
    config_file = output / "config.json"
    config = {"data_root": str(data_root), "cache": str(cache), "tau": args.tau, "meff": args.meff, "model": args.model, "experiments": EXPERIMENTS}
    if config_file.exists():
        if json.loads(config_file.read_text()) != config:
            p.error("Output folder has different inversion settings; use --output with a fresh directory")
    else:
        if results_csv.exists():
            p.error("Existing results.csv lacks config.json")
        config_file.write_text(json.dumps(config, indent=2) + "\n")

    results = {}
    if results_csv.exists():
        with results_csv.open(newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != FIELDS:
                p.error("Existing results.csv has incompatible columns")
            for row in reader:
                results[(row["scenario"], int(row["repetition"]), row["experiment"])] = row
        print(f"Resuming: {len(results)} successful inversions already saved", flush=True)

    attempted = failed = 0
    for scenario in scenarios:
        sea_type = scenario.name.rsplit("_", 1)[-1]
        repetitions = sorted((q for q in scenario.glob("sensordata_*") if q.is_dir()), key=lambda q: int(q.name.split("_")[-1]))
        repetitions = [q for q in repetitions if int(q.name.split("_")[-1]) < (1 if args.pilot else args.reps)]
        for recording in repetitions:
            rep = int(recording.name.split("_")[-1])
            if not (recording / "sensor_info.csv").is_file() or not (recording / "params.csv").is_file():
                print(f"Skipping incomplete recording: {recording}", flush=True)
                continue
            try:
                hs_true = read_hs(recording / "params.csv")
            except (ValueError, OSError) as exc:
                print(f"Skipping invalid truth: {exc}", flush=True)
                continue
            for name, channels in EXPERIMENTS.items():
                key = (scenario.name, rep, name)
                if key in results:
                    continue
                attempted += 1
                tag = f"{scenario.name}_rep{rep}_{name}"
                print(f"[{attempted}] {tag}", flush=True)
                result_npz.unlink(missing_ok=True)
                cmd = [sys.executable, str(inverse), "--data", str(recording.resolve()), "--repo", str(script_dir), "--channels", ",".join(channels), "--cache", str(cache), "--tau", str(args.tau), "--meff", str(args.meff), "--model", args.model]
                run = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True)
                log = output / "logs" / f"{tag}.txt"
                log.write_text("COMMAND: " + " ".join(cmd) + "\n\n" + run.stdout + "\n" + run.stderr)
                if run.returncode:
                    failed += 1
                    print(f"  FAILED (exit {run.returncode}); see {log}", flush=True)
                    continue
                match = re.search(r"Hs estimate\s*=\s*([\d.eE+-]+).*?Tp\s*=\s*([\d.eE+-]+)", run.stdout)
                negative = re.search(r"negative power:\s*([-\d.eE+]+)%", run.stdout)
                if not match or not result_npz.is_file():
                    failed += 1
                    print(f"  FAILED: no parsed estimates or inversion_result.npz; see {log}", flush=True)
                    continue
                try:
                    direction, concentration = direction_summary(result_npz)
                except (ValueError, KeyError, OSError) as exc:
                    failed += 1
                    print(f"  FAILED: invalid spectrum ({exc}); see {log}", flush=True)
                    continue
                hs_est, tp_est = map(float, match.groups())
                shutil.copy2(result_npz, output / "spectra" / f"{tag}.npz")
                results[key] = {"scenario": scenario.name, "sea_type": sea_type, "repetition": rep, "experiment": name, "channels": ",".join(channels), "Hs_true": hs_true, "Hs_est": hs_est, "Hs_error_pct": abs(hs_est - hs_true) / hs_true * 100, "Tp_est": tp_est, "negative_power_pct": float(negative.group(1)) if negative else "", "direction_est_deg": direction, "circular_concentration": concentration, "status": "success"}
                save_rows(results_csv, results)
                print(f"  Hs={hs_est:.2f}m, true={hs_true:.2f}m, error={results[key]['Hs_error_pct']:.2f}%, Tp={tp_est:.2f}s", flush=True)

    if results:
        df = pd.read_csv(results_csv)
        summary = df.groupby(["sea_type", "scenario", "experiment"], as_index=False).agg(mean_Hs_error_pct=("Hs_error_pct", "mean"), mean_concentration=("circular_concentration", "mean"), n=("Hs_est", "count"))
        summary.to_csv(output / "summary_by_scenario.csv", index=False)
        overall = df.groupby(["sea_type", "experiment"], as_index=False).agg(mean_Hs_error_pct=("Hs_error_pct", "mean"), mean_concentration=("circular_concentration", "mean"), n=("Hs_est", "count"))
        overall.to_csv(output / "summary_by_type.csv", index=False)
        print("\nSummary by sea type (completed recordings only):")
        print(overall.round(2).to_string(index=False))
    print(f"\nDone. {len(results)} saved successful inversions, {failed} failed this run. Results: {results_csv}")


if __name__ == "__main__":
    main()
