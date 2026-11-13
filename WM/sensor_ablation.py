

import csv
import re
import subprocess
import sys
from pathlib import Path
import argparse
import numpy as np
import shutil

# --------------------------------------------------
# 1. Command-line arguments
# --------------------------------------------------

parser = argparse.ArgumentParser(
    description="Run sensor ablation with directional analysis"
)

parser.add_argument(
    "--data",
    type=Path,
    required=True,
    help="Sea-state folder containing repetitions"
)

parser.add_argument("--hs", type=float, required=True)
parser.add_argument("--tp", type=float, required=True)

parser.add_argument(
    "--direction",
    type=float,
    required=True,
    help="True mean wave direction in degrees"
)

parser.add_argument(
    "--method",
    choices=["bayesian", "nnls"],
    default="nnls"
)

parser.add_argument(
    "--reps",
    type=int,
    default=20
)

parser.add_argument(
    "--tau",
    type=float,
    default=5.0
)

args = parser.parse_args()

if args.hs <= 0 or args.tp <= 0:
    parser.error("--hs and --tp must be positive")

if not 1 <= args.reps <= 20:
    parser.error("--reps must be between 1 and 20")

DATA_FOLDER = args.data.resolve()
TRUE_HS = args.hs
TRUE_TP = args.tp
TRUE_DIRECTION = args.direction

# --------------------------------------------------
# 2. Sensor configurations
# --------------------------------------------------

ALL = [
    "roll", "pitch", "vert_acc",
    "P1", "P2", "P3", "P4", "P5", "P6"
]

experiments = {"Baseline": ALL}

for sensor in ALL:
    experiments[f"No_{sensor}"] = [
        ch for ch in ALL if ch != sensor
    ]

experiments["Pressure_only"] = ALL[3:]
experiments["Motion_only"] = ALL[:3]

# --------------------------------------------------
# 3. Paths
# --------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent

INVERSION_SCRIPT = SCRIPT_DIR / {
    "bayesian": "invert_waves.py",
    "nnls": "invert_waves_positive.py"
}[args.method]

if not INVERSION_SCRIPT.exists():
    parser.error(
        f"Missing inversion script: {INVERSION_SCRIPT}"
    )

# Separate output folder to protect previous results
OUTPUT = SCRIPT_DIR / f"ablation_direction_{args.method}"
OUTPUT.mkdir(parents=True, exist_ok=True)

SPECTRA_FOLDER = OUTPUT / "spectra"
SPECTRA_FOLDER.mkdir(exist_ok=True)

CSV_FILE = OUTPUT / "results.csv"

NPZ_FILE = SCRIPT_DIR / (
    "inversion_result_nnls.npz"
    if args.method == "nnls"
    else "inversion_result.npz"
)

# --------------------------------------------------
# 4. Directional metrics
# --------------------------------------------------

def calculate_direction_metrics(npz_path, true_direction_deg):

    with np.load(npz_path) as data:
        x = data["x"]
        beta = data["beta"]

    # Verify heading angles are in radians
    if np.max(np.abs(beta)) > 2 * np.pi + 0.01:
        raise ValueError(
            "Heading grid appears to be in degrees"
        )

    # Total nonnegative energy per heading
    energy = np.maximum(x, 0).sum(axis=0)

    total_energy = energy.sum()

    if total_energy <= 0:
        return np.nan, np.nan, np.nan

    # Energy-weighted circular mean
    c = np.sum(energy * np.cos(beta))
    s = np.sum(energy * np.sin(beta))

    # Mean direction is undefined if vectors cancel
    if np.hypot(c, s) <= 1e-12 * total_energy:
        mean_direction = np.nan
        direction_error = np.nan
    else:
        mean_direction = np.degrees(
            np.arctan2(s, c)
        )

        direction_error = abs(
            (mean_direction - true_direction_deg + 180)
            % 360 - 180
        )

    # Energy more than 30 degrees from truth
    headings_deg = np.degrees(beta)

    angular_distance = (
        headings_deg - true_direction_deg + 180
    ) % 360 - 180

    wrong_energy = (
        energy[np.abs(angular_distance) > 30].sum()
        / total_energy * 100
    )

    return (
        mean_direction,
        direction_error,
        wrong_energy
    )

# --------------------------------------------------
# 5. CSV fields and resuming
# --------------------------------------------------

FIELDS = [
    "repetition",
    "experiment",
    "channels",
    "Hs_est",
    "Tp_est",
    "Hs_error_pct",
    "Tp_error_pct",
    "negative_power_pct",
    "direction_est_deg",
    "direction_error_deg",
    "wrong_energy_pct"
]

results = {}

if CSV_FILE.exists():

    with CSV_FILE.open(newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames != FIELDS:
            parser.error(
                "Existing results.csv has different columns. "
                "Use a fresh output folder."
            )

        for row in reader:
            key = (
                int(row["repetition"]),
                row["experiment"]
            )
            results[key] = row

    print(f"Loaded {len(results)} existing results")


def save_results():

    with CSV_FILE.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=FIELDS
        )

        writer.writeheader()

        writer.writerows(
            results[key]
            for key in sorted(results)
        )

# --------------------------------------------------
# 6. Run experiments
# --------------------------------------------------

for repetition in range(args.reps):

    data_path = (
        DATA_FOLDER / f"sensordata_{repetition}"
    )

    if not (data_path / "sensor_info.csv").exists():
        print(f"Skipping missing folder: {data_path}")
        continue

    for name, channels in experiments.items():

        key = (repetition, name)

        if key in results:
            print(
                f"Already completed: rep {repetition}, {name}"
            )
            continue

        print(
            f"[{args.method}] Repetition {repetition}: {name}",
            flush=True
        )

        command = [
            sys.executable,
            str(INVERSION_SCRIPT),
            "--data", str(data_path),
            "--channels", ",".join(channels),
            "--tau", str(args.tau)
        ]

        # Remove previous output to prevent stale results
        if NPZ_FILE.exists():
            NPZ_FILE.unlink()

        run = subprocess.run(
            command,
            capture_output=True,
            text=True,
            cwd=SCRIPT_DIR
        )

        # Save full terminal output
        log_file = (
            OUTPUT / f"rep{repetition}_{name}.txt"
        )

        log_file.write_text(
            run.stdout + "\n" + run.stderr
        )

        if run.returncode != 0:
            print(
                f"FAILED: rep {repetition}, {name}"
            )
            print(f"Check: {log_file}")
            continue

        # Extract wave height and peak period
        match = re.search(
            r"Hs estimate\s*=\s*([\d.]+).*?Tp\s*=\s*([\d.]+)",
            run.stdout
        )

        neg_match = re.search(
            r"negative power:\s*([\d.]+)%",
            run.stdout
        )

        if not match:
            print(
                f"Could not parse: rep {repetition}, {name}"
            )
            continue

        if not NPZ_FILE.exists():
            print(
                f"Missing spectrum: rep {repetition}, {name}"
            )
            continue

        hs_est = float(match.group(1))
        tp_est = float(match.group(2))

        # Calculate directional accuracy
        try:
            (
                direction_est,
                direction_error,
                wrong_energy
            ) = calculate_direction_metrics(
                NPZ_FILE,
                TRUE_DIRECTION
            )
        except (OSError, ValueError, KeyError) as exc:
            print(
                f"Direction analysis failed: "
                f"rep {repetition}, {name}: {exc}"
            )
            continue

        # Save individual spectrum
        saved_spectrum = (
            SPECTRA_FOLDER /
            f"rep{repetition}_{name}.npz"
        )

        shutil.copy2(
            NPZ_FILE,
            saved_spectrum
        )

        # Record results
        results[key] = {
            "repetition": repetition,
            "experiment": name,
            "channels": ",".join(channels),

            "Hs_est": hs_est,
            "Tp_est": tp_est,

            "Hs_error_pct": (
                abs(hs_est - TRUE_HS)
                / TRUE_HS * 100
            ),

            "Tp_error_pct": (
                abs(tp_est - TRUE_TP)
                / TRUE_TP * 100
            ),

            "negative_power_pct": (
                float(neg_match.group(1))
                if neg_match else None
            ),

            "direction_est_deg": direction_est,
            "direction_error_deg": direction_error,
            "wrong_energy_pct": wrong_energy
        }

        # Save progress after every inversion
        save_results()

        print(
            f"  Hs={hs_est:.2f} m, "
            f"Tp={tp_est:.1f} s, "
            f"Direction={direction_est:.1f} deg, "
            f"Error={direction_error:.1f} deg",
            flush=True
        )

print(
    f"\nFinished! {len(results)} results "
    f"saved in {CSV_FILE}"
)
