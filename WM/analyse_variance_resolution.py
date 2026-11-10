"""Check spectral means and variances without fitting the inverse or M."""
import argparse
import json
from pathlib import Path

import numpy as np
import invert_waves as iw
from analyse_variance_convergence import auto_columns, bootstrap_moments, interval
from variance_convergence import ROOT, write_csv

PAIRS = [(800, 1600), (1600, 3200), (800, 3200)]
BANDS = {"whole": (.05, .31), "low": (.05, .10), "peak": (.10, .16),
         "upper": (.16, .26), "tail": (.26, .31)}


def comparison(values, reference, draws, reference_draws, tolerance):
    ratio, bootstrap = values / reference, draws / reference_draws
    low, high = interval(bootstrap)
    median_ci = interval(np.median(bootstrap, axis=(1, 2)))
    row = dict(median_ratio=float(np.median(ratio)),
               median_ci95_low=float(median_ci[0]), median_ci95_high=float(median_ci[1]),
               p90_absolute_relative_difference=float(np.quantile(abs(ratio - 1), .9)),
               fraction_point_estimates_within_tolerance=float(np.mean(abs(ratio - 1) <= tolerance)),
               fraction_intervals_within_tolerance=float(np.mean((low >= 1-tolerance) & (high <= 1+tolerance))),
               fraction_intervals_outside_tolerance=float(np.mean((high < 1-tolerance) | (low > 1+tolerance))))
    return row, ratio, low, high


def analyse(data, output):
    protocol = json.loads((data / "protocol.json").read_text())
    n = protocol["seeds"]
    weights = np.random.default_rng(781024).multinomial(n, np.full(n, 1/n),
                                                      size=protocol["bootstrap"]) / n
    autos = auto_columns()
    crosses = [i for i in range(81) if i not in autos]
    labels = []
    for c, k in iw.stack_pairs(9):
        pair = f"{iw.CHANNELS[c]}:{iw.CHANNELS[k]}"
        labels.extend([pair] if c == k else [f"Re({pair})", f"Im({pair})"])
    cases = {}
    frequencies = None
    for grid in protocol["grids"]:
        for length in protocol["usable_lengths"]:
            records = []
            for seed in range(protocol["seed_start"], protocol["seed_start"] + n):
                with np.load(data / f"n{grid}_t{length}" / f"seed-{seed}.npz") as z:
                    if frequencies is None:
                        frequencies = z["f"].copy()
                    np.testing.assert_array_equal(frequencies, z["f"])
                    f = frequencies
                    if z["observed"].shape != (len(f), 81) or not np.isfinite(z["observed"]).all():
                        raise ValueError("invalid spectral record")
                    records.append(z["observed"])
            x = np.stack(records)
            bm, bv = bootstrap_moments(x, weights)
            cases[grid, length] = dict(mean=x.mean(0), variance=x.var(0, ddof=1),
                                       bmean=bm, bvariance=bv)
            print(f"Analysed {grid} frequencies, {length} s, {n} seeds", flush=True)
    summary, details = [], []
    for length in protocol["usable_lengths"]:
        for coarse, fine in PAIRS:
            a, b = cases[coarse, length], cases[fine, length]
            for metric, columns, family in [("mean", autos, "auto"),
                                             ("variance", autos, "auto"),
                                             ("variance", crosses, "cross")]:
                for band, (start, stop) in BANDS.items():
                    selected = (f >= start) & ((f <= stop) if band in ("whole", "tail") else (f < stop))
                    take = np.ix_(selected, columns)
                    btake = np.ix_(np.arange(len(weights)), np.flatnonzero(selected), columns)
                    result, _, _, _ = comparison(a[metric][take], b[metric][take],
                                                  a["b"+metric][btake], b["b"+metric][btake], protocol["tolerance"])
                    summary.append(dict(usable_seconds=length, coarse_grid=coarse, fine_grid=fine,
                                        metric=metric, family=family, band=band, recordings=n,
                                        spectral_entries=int(selected.sum()*len(columns)), **result))
                _, ratio, low, high = comparison(a[metric][:, columns], b[metric][:, columns],
                                                a["b"+metric][:, :, columns], b["b"+metric][:, :, columns], protocol["tolerance"])
                for i, frequency in enumerate(f):
                    for j, column in enumerate(columns):
                        details.append(dict(usable_seconds=length, coarse_grid=coarse, fine_grid=fine,
                                            metric=metric, family=family, frequency_Hz=frequency,
                                            component=labels[column], ratio=ratio[i, j],
                                            ci95_low=low[i, j], ci95_high=high[i, j]))
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "resolution_summary.csv", summary)
    write_csv(output / "resolution_frequencies.csv", details)
    (output / "resolution_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    plot(cases, f, autos, crosses, protocol, output)


def plot(cases, f, autos, crosses, protocol, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.0))
    length = max(protocol["usable_lengths"])
    reference = cases[3200, length]
    for ax, metric, columns, title in zip(axes, ["mean", "variance", "variance"],
                                         [autos, autos, crosses],
                                         ["Mean auto-spectra", "Auto-spectral variances", "Cross-component variances"]):
        for grid, color in [(800, "#2864A5"), (1600, "#52744F")]:
            case = cases[grid, length]
            ratio = np.median(case[metric][:, columns] / reference[metric][:, columns], axis=1)
            boot = np.median(case["b"+metric][:, :, columns] / reference["b"+metric][:, :, columns], axis=2)
            low, high = interval(boot)
            ax.plot(f, ratio, color=color, label=f"{grid} / 3200 frequencies")
            ax.fill_between(f, low, high, color=color, alpha=.15)
        ax.axhspan(.9, 1.1, color="#888888", alpha=.12, label="10% comparison band")
        ax.axhline(1, color="#666666", ls="--", lw=1)
        ax.set(title=title, xlabel="Frequency (Hz)", ylabel="Median ratio across components")
        ax.grid(alpha=.16)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Wave-grid refinement: {length} s usable recordings, {protocol['seeds']} seeds per grid")
    fig.text(.035, .02, "Shading: pointwise 95% paired whole-record bootstrap intervals for component medians. Individual entries can differ more.\n"
             "3200 is a comparison reference, not proven truth. Hull responses, sensor processing, inverse and M are unchanged.",
             fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, .12, 1, .95))
    fig.savefig(output / "variance_resolution.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("variance_output/resolution"))
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args()
    analyse(args.data, args.output)
