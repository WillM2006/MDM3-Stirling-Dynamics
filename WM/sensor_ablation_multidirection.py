"""Run NNLS sensor-ablation inversions across several true wave directions.

Uses the same inversion CLI and NPZ format as sensor_ablation_direction.py.
Run from any directory; paths to the inversion script and outputs are resolved
relative to this file. Existing completed rows are resumed safely.
"""
import argparse
import csv
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ALL = ["roll", "pitch", "vert_acc", "P1", "P2", "P3", "P4", "P5", "P6"]
EXPERIMENTS = {
    "Baseline": ALL,
    "P3_only": ["P3"],
    "P3_P5": ["P3", "P5"],
    "P1_P4_Pitch": ["P1", "P4", "pitch"],
}
FIELDS = [
    "direction_true_deg", "repetition", "experiment", "channels",
    "Hs_est", "Tp_est", "Hs_error_pct", "Tp_error_pct",
    "negative_power_pct", "direction_est_deg", "direction_error_deg",
    "wrong_energy_pct", "circular_concentration",
]


def direction_metrics(npz_path, true_direction_deg):
    with np.load(npz_path) as data:
        x = np.asarray(data["x"])
        beta = np.asarray(data["beta"]).reshape(-1)
    if x.ndim != 2 or x.shape[1] != len(beta):
        raise ValueError(f"Spectrum shape {x.shape} incompatible with beta {beta.shape}")
    if not np.isfinite(x).all() or not np.isfinite(beta).all():
        raise ValueError("Spectrum contains nonfinite values")
    if np.max(np.abs(beta)) > 2 * np.pi + 0.01:
        raise ValueError("Heading grid appears to be in degrees rather than radians")
    energy = np.maximum(x, 0).sum(axis=0)
    total = energy.sum()
    if total <= 0:
        return np.nan, np.nan, np.nan, np.nan
    c = np.sum(energy * np.cos(beta))
    s = np.sum(energy * np.sin(beta))
    concentration = np.hypot(c, s) / total
    if concentration <= 1e-12:
        mean_deg, error = np.nan, np.nan
    else:
        mean_deg = np.degrees(np.arctan2(s, c))
        error = abs((mean_deg - true_direction_deg + 180) % 360 - 180)
    distance = (np.degrees(beta) - true_direction_deg + 180) % 360 - 180
    outside = 100 * energy[np.abs(distance) > 30].sum() / total
    return mean_deg, error, outside, concentration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path,
                        default=Path("/Users/leorameto/Downloads/sensor_data_new 2"),
                        help="Parent folder containing sensordata_Hs..._direction... folders")
    parser.add_argument("--hs", type=float, default=1.7)
    parser.add_argument("--tp", type=float, default=8.0)
    parser.add_argument("--spreading", type=int, default=10)
    parser.add_argument("--gamma", type=str, default="3.3")
    parser.add_argument("--depth", type=int, default=15)
    parser.add_argument("--directions", type=int, nargs="+", default=[0, 45, 90, 135, 180])
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--tau", type=float, default=5.0)
    args = parser.parse_args()
    if args.hs <= 0 or args.tp <= 0 or args.reps < 1:
        parser.error("--hs, --tp and --reps must be positive")
    if len(set(args.directions)) != len(args.directions):
        parser.error("--directions must not contain duplicates")

    script_dir = Path(__file__).resolve().parent
    inverse = script_dir / "invert_waves_positive.py"
    if not inverse.is_file():
        parser.error(f"Missing inversion script: {inverse}")
    root = args.data_root.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"Missing data root: {root}")
    output = script_dir / "ablation_multidirection_nnls"
    output.mkdir(parents=True, exist_ok=True)
    spectra_dir = output / "spectra"
    spectra_dir.mkdir(exist_ok=True)
    csv_file = output / "results.csv"
    npz_file = script_dir / "inversion_result_nnls.npz"

    results = {}
    if csv_file.exists():
        with csv_file.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != FIELDS:
                parser.error("Existing results.csv has different columns; use a fresh output folder")
            for row in reader:
                key = (int(row["direction_true_deg"]), int(row["repetition"]), row["experiment"])
                results[key] = row
        print(f"Resuming {len(results)} saved inversions")

    def save():
        temp = csv_file.with_suffix(".csv.tmp")
        with temp.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(results[key] for key in sorted(results))
        temp.replace(csv_file)

    for direction in args.directions:
        # :g formatting matches names such as Hs1.7 and Tp8.
        folder_name = (f"sensordata_Hs{args.hs:g}_Tp{args.tp:g}_s{args.spreading}"
                       f"_gamma{args.gamma}_depth{args.depth}_direction{direction}")
        folder = root / folder_name
        if not folder.is_dir():
            print(f"MISSING DATASET (skipping): {folder}", flush=True)
            continue
        print(f"\n=== True direction {direction} degrees: {folder_name} ===", flush=True)
        for rep in range(args.reps):
            data_path = folder / f"sensordata_{rep}"
            if not (data_path / "sensor_info.csv").is_file():
                print(f"Missing repetition (skipping): {data_path}")
                continue
            for name, channels in EXPERIMENTS.items():
                key = (direction, rep, name)
                if key in results:
                    continue
                print(f"[NNLS] Direction {direction} | rep {rep} | {name}", flush=True)
                if npz_file.exists():
                    npz_file.unlink()  # Avoid mistaking a previous run's spectrum for this one.
                command = [sys.executable, str(inverse), "--data", str(data_path),
                           "--channels", ",".join(channels), "--tau", str(args.tau)]
                run = subprocess.run(command, capture_output=True, text=True, cwd=script_dir)
                log = output / f"direction{direction}_rep{rep}_{name}.txt"
                log.write_text(run.stdout + "\n" + run.stderr)
                if run.returncode != 0:
                    print(f"  FAILED: see {log}", flush=True)
                    continue
                match = re.search(r"Hs estimate\s*=\s*([\d.]+).*?Tp\s*=\s*([\d.]+)", run.stdout)
                neg_match = re.search(r"negative power:\s*([\d.]+)%", run.stdout)
                if not match or not npz_file.is_file():
                    print(f"  Missing parsed estimates or spectrum: see {log}", flush=True)
                    continue
                hs_est, tp_est = map(float, match.groups())
                try:
                    mean_dir, dir_err, outside, concentration = direction_metrics(npz_file, direction)
                except (OSError, ValueError, KeyError) as exc:
                    print(f"  Invalid directional spectrum: {exc}", flush=True)
                    continue
                saved_npz = spectra_dir / f"direction{direction}_rep{rep}_{name}.npz"
                shutil.copy2(npz_file, saved_npz)
                results[key] = {
                    "direction_true_deg": direction,
                    "repetition": rep,
                    "experiment": name,
                    "channels": ",".join(channels),
                    "Hs_est": hs_est,
                    "Tp_est": tp_est,
                    "Hs_error_pct": abs(hs_est - args.hs) / args.hs * 100,
                    "Tp_error_pct": abs(tp_est - args.tp) / args.tp * 100,
                    "negative_power_pct": float(neg_match.group(1)) if neg_match else "",
                    "direction_est_deg": mean_dir,
                    "direction_error_deg": dir_err,
                    "wrong_energy_pct": outside,
                    "circular_concentration": concentration,
                }
                save()
                print(f"  Hs={hs_est:.2f}, Tp={tp_est:.2f}, "
                      f"direction={mean_dir:.1f}, error={dir_err:.1f}, "
                      f"outside30={outside:.1f}%, concentration={concentration:.3f}", flush=True)

    print(f"\nFinished: {len(results)} results in {csv_file}")
    if results:
        import pandas as pd
        frame = pd.read_csv(csv_file)
        summary = frame.groupby(["direction_true_deg", "experiment"]).agg(
            mean_Hs_error=("Hs_error_pct", "mean"),
            mean_Tp_error=("Tp_error_pct", "mean"),
            mean_direction_error=("direction_error_deg", "mean"),
            mean_outside_30=("wrong_energy_pct", "mean"),
            mean_concentration=("circular_concentration", "mean"),
            n=("repetition", "count"),
        ).reset_index()
        summary.to_csv(output / "summary.csv", index=False)
        print("\nSummary (partial if runs failed or datasets missing):")
        print(summary.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
