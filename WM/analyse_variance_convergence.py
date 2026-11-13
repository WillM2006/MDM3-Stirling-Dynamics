"""Paired whole-record bootstrap analysis of the convergence experiment."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.signal import get_window
from spectral_variance import effective_averages
from variance_study import summarise
from variance_convergence import GRIDS, LENGTHS, ROOT, write_csv
import invert_waves as iw


def bootstrap_moments(values, weights):
    n = len(values)
    mean = np.einsum("bn,nfc->bfc", weights, values)
    variance = (np.einsum("bn,nfc->bfc", weights, values**2) - mean**2) * n / (n - 1)
    return mean, np.maximum(variance, np.finfo(float).tiny)


def interval(values):
    return np.quantile(values, [.025, .975], axis=0)


def auto_columns():
    selected, position = [], 0
    for c, k in iw.stack_pairs(9):
        if c == k:
            selected.append(position)
        position += 1 if c == k else 2
    return selected


def analyse(data, output):
    protocol = json.loads((data / "protocol.json").read_text())
    n = protocol["seeds"]
    bootstrap, tolerance = protocol["bootstrap"], protocol["tolerance"]
    rng = np.random.default_rng(781024)
    weights = rng.multinomial(n, np.full(n, 1 / n), size=bootstrap) / n
    columns = auto_columns()
    cases = {}
    methods = []
    for grid in GRIDS:
        for length in LENGTHS if grid in (40, 800) else [900]:
            name = f"n{grid}_t{length}"
            records = []
            predictions = {}
            for seed in range(protocol["seed_start"], protocol["seed_start"] + n):
                with np.load(data / name / f"seed-{seed}.npz") as z:
                    f = z["f"]
                    records.append(z["observed"][:, columns])
                    for method in ["fixed_2.5", "overlap_count", "segment_variance", "nonoverlap_segments"]:
                        predictions.setdefault(method, []).append(z[method][:, columns])
            x = np.stack(records)
            mean, var = x.mean(0), x.var(0, ddof=1)
            bm, bv = bootstrap_moments(x, weights)
            cases[name] = dict(grid=grid, length=length, mean=mean, var=var, bm=bm, bv=bv,
                               empirical_m=mean**2 / var, bootstrap_m=bm**2 / bv,
                               predicted={key: np.stack(value).mean(0) for key, value in predictions.items()})
            methods.extend(summarise(name, x, predictions, f, list(range(9)), bootstrap=bootstrap))
            print(f"Analysed {name}: {n} paired seeds", flush=True)
    summary, details, lengths = [], [], []
    for name, case in cases.items():
        reference = cases[f'n800_t{case["length"]}']
        mr, vr = case["mean"] / reference["mean"], case["var"] / reference["var"]
        bmr, bvr = case["bm"] / reference["bm"], case["bv"] / reference["bv"]
        ml, mh = interval(bmr)
        vl, vh = interval(bvr)
        eml, emh = interval(case["bootstrap_m"])
        mean_ci = interval(np.median(bmr, axis=(1, 2)))
        var_ci = interval(np.median(bvr, axis=(1, 2)))
        m_ci = interval(np.median(case["bootstrap_m"], axis=(1, 2)))
        count = 1 + (int(case["length"] * iw.FS) - 400) // 200
        nominal = effective_averages(get_window("hann", 400), 200, count)
        summary.append(dict(case=name, wave_frequencies=case["grid"], usable_seconds=case["length"],
                            reference_case=f'n800_t{case["length"]}', reference_identity=case["grid"] == 800,
                            recordings=n, welch_segments=count, overlap_averages=nominal,
                            median_empirical_M=float(np.median(case["empirical_m"])),
                            M_ci95_low=float(m_ci[0]), M_ci95_high=float(m_ci[1]),
                            median_mean_ratio_to_800=float(np.median(mr)), mean_ratio_ci95_low=float(mean_ci[0]),
                            mean_ratio_ci95_high=float(mean_ci[1]),
                            median_variance_ratio_to_800=float(np.median(vr)), variance_ratio_ci95_low=float(var_ci[0]),
                            variance_ratio_ci95_high=float(var_ci[1]),
                            fraction_mean_points_within_10pct=float(np.mean(abs(mr - 1) <= tolerance)),
                            fraction_variance_points_within_10pct=float(np.mean(abs(vr - 1) <= tolerance)),
                            fraction_mean_intervals_within_10pct=float(np.mean((ml >= 1 - tolerance) & (mh <= 1 + tolerance))),
                            fraction_variance_intervals_within_10pct=float(np.mean((vl >= 1 - tolerance) & (vh <= 1 + tolerance)))))
        for i, frequency in enumerate(f):
            for c, channel in enumerate(iw.CHANNELS):
                details.append(dict(case=name, frequency_Hz=frequency, channel=channel,
                                    mean_auto_spectrum=case["mean"][i, c], variance_auto_spectrum=case["var"][i, c],
                                    empirical_M=case["empirical_m"][i, c], M_ci95_low=eml[i, c], M_ci95_high=emh[i, c],
                                    mean_ratio_to_800=mr[i, c], mean_ratio_ci95_low=ml[i, c], mean_ratio_ci95_high=mh[i, c],
                                    variance_ratio_to_800=vr[i, c], variance_ratio_ci95_low=vl[i, c], variance_ratio_ci95_high=vh[i, c]))
        if case["grid"] in (40, 800):
            short = cases[f'n{case["grid"]}_t900']
            actual = case["var"] / short["var"]
            boot = case["bv"] / short["bv"]
            ci = interval(np.median(boot, axis=(1, 2)))
            mean_ratio = case["mean"] / short["mean"]
            mean_ci = interval(np.median(case["bm"] / short["bm"], axis=(1, 2)))
            lengths.append(dict(wave_frequencies=case["grid"], usable_seconds=case["length"],
                                median_variance_ratio_to_900=float(np.median(actual)),
                                ci95_low=float(ci[0]), ci95_high=float(ci[1]),
                                median_mean_ratio_to_900=float(np.median(mean_ratio)),
                                mean_ratio_ci95_low=float(mean_ci[0]), mean_ratio_ci95_high=float(mean_ci[1]),
                                independent_averaging_prediction=7.629139072847682 / nominal))
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "convergence_summary.csv", summary)
    write_csv(output / "convergence_frequencies.csv", details)
    write_csv(output / "convergence_lengths.csv", lengths)
    write_csv(output / "convergence_methods.csv", methods)
    (output / "convergence_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    plot(cases, summary, methods, f, output)


def plot(cases, summary, methods, f, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.9))
    rows = [next(row for row in summary if row["case"] == f"n{grid}_t900") for grid in GRIDS]
    axes[0].errorbar(GRIDS, [row["median_variance_ratio_to_800"] for row in rows],
                     yerr=[[row["median_variance_ratio_to_800"] - row["variance_ratio_ci95_low"] for row in rows],
                           [row["variance_ratio_ci95_high"] - row["median_variance_ratio_to_800"] for row in rows]],
                     fmt="o-", capsize=3, color="#2864A5")
    axes[0].axhline(1, color="#666666", ls="--", lw=1)
    axes[0].set(title="Grid effect on repeat-record variance", xlabel="Generated wave frequencies (count)",
                ylabel="Median variance / 800-frequency reference")
    axes[0].set_xscale("log")
    axes[0].set_xticks(GRIDS, [str(n) for n in GRIDS])
    for grid, color in [(40, "#2864A5"), (800, "#546B54")]:
        rows = [next(row for row in summary if row["case"] == f"n{grid}_t{length}") for length in LENGTHS]
        axes[1].errorbar(LENGTHS, [row["median_empirical_M"] for row in rows],
                         yerr=[[row["median_empirical_M"] - row["M_ci95_low"] for row in rows],
                               [row["M_ci95_high"] - row["median_empirical_M"] for row in rows]],
                         fmt="o-", capsize=3, color=color, label=f"{grid} wave frequencies")
    axes[1].plot(LENGTHS, [next(row for row in summary if row["case"] == f"n800_t{length}")["overlap_averages"]
                          for length in LENGTHS], ls="--", color="#777777", label="Hann overlap benchmark")
    axes[1].set(title="Effective averaging as recordings lengthen", xlabel="Usable recording length (s)",
                ylabel="Median empirical M across auto-spectra")
    axes[1].set_xticks(LENGTHS)
    axes[1].legend(frameon=False, fontsize=8)
    for method, label, color in [("fixed_2.5", "Existing M = 2.5", "#777777"),
                                 ("segment_variance", "Segment-based variance", "#2864A5")]:
        rows = [next(row for row in methods if row["case"] == f"n{grid}_t900" and row["method"] == method and row["band_Hz"] == "whole") for grid in GRIDS]
        axes[2].errorbar(GRIDS, [row["median_predicted_over_actual"] for row in rows],
                         yerr=[[row["median_predicted_over_actual"] - row["ci95_low"] for row in rows],
                               [row["ci95_high"] - row["median_predicted_over_actual"] for row in rows]],
                         fmt="o-", capsize=3, color=color, label=label)
    axes[2].axhline(1, color="#666666", ls="--", lw=1)
    axes[2].set(title="Do the variance weights match?", xlabel="Generated wave frequencies (count)",
                ylabel="Median predicted / repeat-record variance")
    axes[2].set_xscale("log")
    axes[2].set_xticks(GRIDS, [str(n) for n in GRIDS])
    axes[2].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.grid(alpha=.16)
    fig.text(.035, .015, f"{summary[0]['recordings']} seeds per case; 95% paired whole-record bootstrap intervals. Nine auto-spectra, 0.05–0.31 Hz.\n"
             "800 frequencies is a reference, not proven truth. The inverse is unchanged. Synthetic variance checks do not establish posterior coverage.",
             fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, .12, 1, 1))
    fig.savefig(output / "variance_convergence.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), sharey=True)
    for ax, grid in zip(axes, [40, 800]):
        for length, color in zip(LENGTHS, ["#2864A5", "#777777", "#546B54"]):
            case = cases[f"n{grid}_t{length}"]
            line = np.median(case["empirical_m"], axis=1)
            boot = np.median(case["bootstrap_m"], axis=2)
            low, high = interval(boot)
            ax.plot(f, line, color=color, label=f"{length} s usable")
            ax.fill_between(f, low, high, color=color, alpha=.12)
        ax.set(title=f"{grid} generated wave frequencies", xlabel="Frequency (Hz)")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=.16)
    axes[0].set_ylabel("Empirical M, median across nine channels")
    fig.text(.04, .01, f"Source: convergence_frequencies.csv. Pointwise 95% bootstrap bands; same physical sea conditions, {summary[0]['recordings']} seeds.\n"
             "Frequency and record-length dependence are diagnostics, not a fitted rule supplied to the inverse.", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, .12, 1, 1))
    fig.savefig(output / "variance_convergence_frequencies.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("variance_output/convergence"))
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args()
    analyse(args.data, args.output)
