"""Plot the segment-variance result and its bootstrap intervals."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
with (root / "variance_study_results.csv").open() as stream:
    rows = {row["case"]: row for row in csv.DictReader(stream)
            if row["method"] == "segment_variance" and row["band_Hz"] == "whole"}

cases = ["gaussian_control", "baseline", "dense_waves", "different_period"]
labels = ["Independent Gaussian control\n1,000 recordings",
          "Current sea: 40 frequencies, peak period 8 s\n48 recordings",
          "Denser sea: 400 frequencies, peak period 8 s\n48 recordings",
          "Current sea: 40 frequencies, peak period 6 s\n48 recordings"]
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(12, 5.6))
for y, case in enumerate(cases):
    row = rows[case]
    value = float(row["median_predicted_over_actual"])
    low, high = float(row["ci95_low"]), float(row["ci95_high"])
    assert 0 < low < value < high
    ax.errorbar(value, y, xerr=[[value - low], [high - value]],
                fmt="o", color="#2864A5", capsize=5, markersize=7, linewidth=2)
    ax.annotate(f"{value:.0%}  (95% interval {low:.0%}–{high:.0%})",
                (high, y), xytext=(9, 0), textcoords="offset points", va="center", fontsize=10)
ax.axvline(1, color="#666666", linestyle="--", linewidth=1)
ax.set_yticks(range(len(cases)), labels)
ax.set(xlim=(0, 1.7), ylim=(3.5, -.65),
       xlabel="Predicted variance / observed variance between recordings",
       title="Segment-based estimates miss much of the variance between 40-frequency recordings")
ax.set_xticks([0, .25, .5, .75, 1, 1.25, 1.5])
ax.grid(axis="x", alpha=.16)
ax.text(1.02, -.48, "1 = correct on this summary", fontsize=9, color="#555555")
fig.text(.025, .015,
         "Source: variance_study_results.csv. Median ratio across 0.05–0.31 Hz; nine sensor auto-spectra (three in the control).\n"
         "Intervals: 600 whole-record bootstrap resamples. These quantify sampling uncertainty in the tested cases;\n"
         "they do not establish the cause, physical-model validity, or Bayesian posterior coverage.",
         fontsize=9, color="#555555")
fig.tight_layout(rect=(0, .14, 1, 1))
fig.savefig(root / "variance_study_calibration.png", dpi=160)
