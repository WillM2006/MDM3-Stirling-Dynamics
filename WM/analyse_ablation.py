
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


BAYESIAN = Path("ablation_results_bayesian/results.csv")
NNLS = Path("ablation_results_nnls/results.csv")
OUTPUT = Path("ablation_comparison")
OUTPUT.mkdir(exist_ok=True)

bayesian = pd.read_csv(BAYESIAN)
nnls = pd.read_csv(NNLS)

bayesian["method"] = "Bayesian"
nnls["method"] = "NNLS"

df = pd.concat([bayesian, nnls], ignore_index=True)


summary = df.groupby(["experiment", "method"]).agg(
    mean_Hs_error=("Hs_error_pct", "mean"),
    std_Hs_error=("Hs_error_pct", "std"),
    mean_Tp_error=("Tp_error_pct", "mean"),
    std_Tp_error=("Tp_error_pct", "std"),
    mean_negative_power=("negative_power_pct", "mean"),
    n=("repetition", "count")
).reset_index()

summary.to_csv(OUTPUT / "summary.csv", index=False)

print("\nBayesian vs NNLS summary:")
print(summary.round(2).to_string(index=False))

order = [
    "Baseline",
    "No_roll",
    "No_pitch",
    "No_vert_acc",
    "No_P1",
    "No_P2",
    "No_P3",
    "No_P4",
    "No_P5",
    "No_P6",
    "Pressure_only",
    "Motion_only"
]

def comparison_plot(metric, ylabel, filename):

    means = summary.pivot(
        index="experiment",
        columns="method",
        values=f"mean_{metric}"
    ).reindex(order)

    stds = summary.pivot(
        index="experiment",
        columns="method",
        values=f"std_{metric}"
    ).reindex(order)

    x = np.arange(len(order))
    width = 0.38

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, method in enumerate(["Bayesian", "NNLS"]):
        ax.bar(
            x + (i - 0.5) * width,
            means[method],
            width,
            yerr=stds[method],
            capsize=3,
            label=method
        )

    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Sensor configuration")
    ax.set_title(f"Sensor ablation: Bayesian vs NNLS ({metric})")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(OUTPUT / filename, dpi=200)
    plt.show()
    plt.close(fig)


comparison_plot(
    "Hs_error",
    "Mean absolute percentage error in Hs (%)",
    "Hs_comparison.png"
)

comparison_plot(
    "Tp_error",
    "Mean absolute percentage error in Tp (%)",
    "Tp_comparison.png"
)


paired = bayesian.merge(
    nnls,
    on=["repetition", "experiment"],
    suffixes=("_bayesian", "_nnls"),
    validate="one_to_one"
)

paired["Hs_improvement_pp"] = (
    paired["Hs_error_pct_bayesian"]
    - paired["Hs_error_pct_nnls"]
)

paired["Tp_improvement_pp"] = (
    paired["Tp_error_pct_bayesian"]
    - paired["Tp_error_pct_nnls"]
)

paired_summary = paired.groupby("experiment").agg(
    mean_Hs_improvement_pp=("Hs_improvement_pp", "mean"),
    std_Hs_improvement_pp=("Hs_improvement_pp", "std"),
    mean_Tp_improvement_pp=("Tp_improvement_pp", "mean"),
    n=("repetition", "count")
).reindex(order)

paired_summary.to_csv(OUTPUT / "paired_comparison.csv")

print("\nPaired improvement (positive means NNLS is better):")
print(paired_summary.round(2))


nnls_baseline = nnls.loc[
    nnls["experiment"] == "Baseline",
    ["repetition", "Hs_error_pct", "Tp_error_pct"]
].rename(columns={
    "Hs_error_pct": "baseline_Hs_error",
    "Tp_error_pct": "baseline_Tp_error"
})

removals = nnls[
    nnls["experiment"].str.startswith("No_")
].merge(
    nnls_baseline,
    on="repetition",
    validate="many_to_one"
)

removals["Hs_change_pp"] = (
    removals["Hs_error_pct"]
    - removals["baseline_Hs_error"]
)

removals["Tp_change_pp"] = (
    removals["Tp_error_pct"]
    - removals["baseline_Tp_error"]
)

importance = removals.groupby("experiment").agg(
    mean_Hs_change_pp=("Hs_change_pp", "mean"),
    std_Hs_change_pp=("Hs_change_pp", "std"),
    mean_Tp_change_pp=("Tp_change_pp", "mean"),
    n=("repetition", "count")
).sort_values("mean_Hs_change_pp", ascending=False)

importance.to_csv(OUTPUT / "sensor_importance.csv")

print("\nSensor removal effect under NNLS:")
print(importance.round(2))

fig, ax = plt.subplots(figsize=(10, 5))

importance["mean_Hs_change_pp"].plot(
    kind="bar",
    yerr=importance["std_Hs_change_pp"],
    capsize=3,
    ax=ax
)

ax.axhline(0, color="black", linewidth=1)
ax.set_ylabel("Change in Hs error after removal (percentage points)")
ax.set_xlabel("Removed sensor")
ax.set_title("NNLS: Effect of removing individual sensors")

plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT / "sensor_importance.png", dpi=200)
plt.show()
plt.close(fig)

print(f"\nAll results saved to {OUTPUT}/")



df_dir = pd.read_csv("ablation_direction_nnls/results.csv")

direction_summary = df_dir.groupby("experiment").agg(
    mean_direction_error=("direction_error_deg", "mean"),
    std_direction_error=("direction_error_deg", "std"),
    mean_outside_30=("wrong_energy_pct", "mean"),
    std_outside_30=("wrong_energy_pct", "std"),
    n=("repetition", "count")
)

print("\nDirectional ablation summary:")
print(
    direction_summary
    .sort_values("mean_direction_error")
    .round(2)
)

# Save summary
direction_summary.to_csv(
    OUTPUT / "direction_summary.csv"
)


direction_plot = direction_summary.reindex(order)

fig, ax = plt.subplots(figsize=(12, 5))

direction_plot["mean_direction_error"].plot(
    kind="bar",
    yerr=direction_plot["std_direction_error"],
    capsize=3,
    ax=ax
)

ax.set_ylabel("Mean absolute directional error (degrees)")
ax.set_xlabel("Sensor configuration")
ax.set_title("Sensor ablation: Wave direction estimation")

plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig(
    OUTPUT / "direction_ablation.png",
    dpi=200
)

plt.show()
plt.close(fig)

fig, ax = plt.subplots(figsize=(12, 5))

direction_plot["mean_outside_30"].plot(
    kind="bar",
    yerr=direction_plot["std_outside_30"],
    capsize=3,
    ax=ax
)

ax.set_ylabel("Energy outside ±30° of true direction (%)")
ax.set_xlabel("Sensor configuration")
ax.set_title("Sensor ablation: Directional energy concentration")

plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig(
    OUTPUT / "direction_energy.png",
    dpi=200
)

plt.show()
plt.close(fig)

print("\nDirectional analysis saved to:", OUTPUT)


df = pd.read_csv("ablation_combinations_nnls/results.csv")

summary = df.groupby("experiment").agg(
    mean_Hs_error=("Hs_error_pct", "mean"),
    mean_Tp_error=("Tp_error_pct", "mean"),
    mean_direction_error=("direction_error_deg", "mean"),
    std_direction_error=("direction_error_deg", "std"),
    n=("repetition", "count")
)

print(summary.sort_values("mean_direction_error").round(2))
summary.sort_values("mean_direction_error").to_csv(
    OUTPUT / "combination_summary.csv"
)

