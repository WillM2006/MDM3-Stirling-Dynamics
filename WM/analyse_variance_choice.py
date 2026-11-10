"""Plot count convergence separately from placement at a fixed component count."""
import argparse
import json
from pathlib import Path

import numpy as np
from analyse_variance_convergence import auto_columns, bootstrap_moments, interval
from analyse_variance_resolution import BANDS, comparison
from variance_convergence import ROOT, write_csv
from variance_choice import COUNTS

PAIRS = [(f"uniform{n}", f"uniform{2*n}") for n in COUNTS[:-1]] + [
    ("equal3200", "uniform3200"), ("half3200", "uniform3200")]
COLOURS = [(900, "#2864A5"), (1800, "#777777"), (3600, "#52744F")]


def analyse(data, output):
    protocol = json.loads((data/"protocol.json").read_text())
    n = protocol["seeds"]
    weights = np.random.default_rng(781024).multinomial(n, np.full(n, 1/n),
                                                      size=protocol["bootstrap"])/n
    auto = auto_columns()
    cross = [i for i in range(81) if i not in auto]
    summary, frequencies = [], []
    f = None
    # Process lengths separately to bound bootstrap memory.
    for length in protocol["usable_lengths"]:
        cases = {}
        for name, count in protocol["cases"].items():
            records = []
            for seed in range(protocol["seed_start"], protocol["seed_start"]+n):
                with np.load(data/name/f"n{count}_t{length}"/f"seed-{seed}.npz") as z:
                    if f is None:
                        f = z["f"].copy()
                    np.testing.assert_array_equal(f, z["f"])
                    if z["observed"].shape != (len(f), 81) or not np.isfinite(z["observed"]).all():
                        raise ValueError("invalid spectral record")
                    records.append(z["observed"])
            x = np.stack(records)
            bm, bv = bootstrap_moments(x, weights)
            cases[name] = dict(mean=x.mean(0), variance=x.var(0, ddof=1), bmean=bm, bvariance=bv)
            print(f"Analysed {name}, {length}s, {n} seeds", flush=True)
        for first, second in PAIRS:
            a, b = cases[first], cases[second]
            for metric, columns, family in [("mean", auto, "auto"),
                                             ("variance", auto, "auto"),
                                             ("variance", cross, "cross")]:
                values, reference = a[metric][:, columns], b[metric][:, columns]
                draws, reference_draws = a["b"+metric][:, :, columns], b["b"+metric][:, :, columns]
                for band, (start, stop) in BANDS.items():
                    selected = (f >= start) & ((f <= stop) if band in ("whole", "tail") else (f < stop))
                    result, _, _, _ = comparison(values[selected], reference[selected], draws[:, selected],
                                                  reference_draws[:, selected], protocol["tolerance"])
                    summary.append(dict(first_case=first, reference_case=second, usable_seconds=length,
                                        metric=metric, family=family, band=band, recordings=n,
                                        spectral_entries=int(selected.sum()*len(columns)), **result))
                ratio = np.median(values/reference, axis=1)
                low, high = interval(np.median(draws/reference_draws, axis=2))
                for i, frequency in enumerate(f):
                    frequencies.append(dict(first_case=first, reference_case=second, usable_seconds=length,
                                            metric=metric, family=family, frequency_Hz=frequency,
                                            median_ratio=ratio[i], median_ci95_low=low[i], median_ci95_high=high[i]))
        del cases
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output/"choice_summary.csv", summary)
    write_csv(output/"choice_frequencies.csv", frequencies)
    (output/"choice_protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    plot(summary, frequencies, output)


def plot(summary, frequencies, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    def find(rows, first, second, length, metric, family, band=None, frequency=None):
        return next(r for r in rows if r["first_case"] == first and r["reference_case"] == second
                    and int(r["usable_seconds"]) == length and r["metric"] == metric and r["family"] == family
                    and (band is None or r["band"] == band)
                    and (frequency is None or abs(float(r["frequency_Hz"])-frequency) < 1e-9))

    def draw(ax, x, rows, colour, label):
        y, low, high = [np.array([float(r[key]) for r in rows]) for key in
                        ["median_ratio", "median_ci95_low", "median_ci95_high"]]
        ax.plot(x, y, "o-", color=colour, label=label, ms=4)
        ax.vlines(x, low, high, color=colour, alpha=.65, lw=1.5)

    panels = [("auto", "whole", "Auto-spectral variance: full band"),
              ("cross", "whole", "Cross-component variance: full band"),
              ("auto", "tail", "Auto-spectral variance: 0.26–0.31 Hz")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.2))
    x = np.array(COUNTS[:-1])
    for ax, (family, band, title) in zip(axes, panels):
        for length, colour in COLOURS:
            rows = [find(summary, f"uniform{count}", f"uniform{2*count}", length,
                         "variance", family, band=band) for count in x]
            draw(ax, x, rows, colour, f"{length}s usable")
        ax.axhspan(.9, 1.1, color="#888888", alpha=.12)
        ax.axhline(1, color="#666666", ls="--", lw=1)
        ax.axvline(6400, color="#888888", ls=":", lw=1)
        ax.set_xscale("log", base=2)
        ax.set_xticks(x, [f"{v:,}" for v in x])
        ax.set(title=title, xlabel="Uniform frequency count N (compared with 2N)",
               ylabel="Median variance ratio: N / 2N")
        ax.grid(alpha=.15)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Does adding more wave frequencies make little difference?")
    fig.text(.04, .025, "Vertical bars: 95% paired whole-record bootstrap intervals; grey band: ±10%. Each point is a real N-versus-2N test.\n"
             "Last point: 6,400 versus 12,800. Dotted line: candidate, not a justified choice by definition. 96 seeds; nested prefixes; one sea condition.",
             fontsize=8, color="#555555")
    fig.tight_layout(rect=(.02, .14, .99, .94))
    fig.savefig(output/"frequency_count_convergence.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.2))
    settings = [("equal3200", 0.), ("half3200", .5)]
    for ax, (metric, family, band, title) in zip(axes, [
            ("mean", "auto", None, "Mean auto-spectrum at 0.07 Hz"),
            ("variance", "auto", "tail", "Auto-spectral variance: 0.26–0.31 Hz"),
            ("variance", "cross", "tail", "Cross-component variance: 0.26–0.31 Hz")]):
        for length, colour in COLOURS:
            source = frequencies if metric == "mean" else summary
            rows = [find(source, name, "uniform3200", length, metric, family,
                         band=band, frequency=.07 if metric == "mean" else None) for name, _ in settings]
            draw(ax, [0., .5], rows, colour, f"{length}s usable")
            ax.plot([.5, 1.], [float(rows[-1]["median_ratio"]), 1.], color=colour)
        ax.scatter([1], [1], facecolors="white", edgecolors="#333333", zorder=5, s=35)
        ax.axhspan(.9, 1.1, color="#888888", alpha=.12)
        ax.axhline(1, color="#666666", ls="--", lw=1)
        ax.set_xticks([0, .5, 1], ["Equal-energy", "50/50", "Uniform"])
        ax.set(title=title, xlabel="Frequency placement (3,200 components throughout)",
               ylabel="Median ratio to uniform 3,200")
        ax.grid(alpha=.15)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("What changes when we move the frequencies, holding the count fixed?")
    fig.text(.04, .025, "Three actual grids; straight lines guide the eye, not a fitted convergence law. Vertical bars: 95% paired bootstrap intervals.\n"
             "The open uniform marker is a reference identity, not convergence evidence. Same spectrum and energy; 96 seeds; nested prefixes; one sea condition.",
             fontsize=8, color="#555555")
    fig.tight_layout(rect=(.02, .14, .99, .94))
    fig.savefig(output/"frequency_placement_transition.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("variance_output/choice"))
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args()
    analyse(args.data, args.output)
