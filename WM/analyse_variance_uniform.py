"""Compare grid placement and uniform-grid refinement without fitting M."""
import argparse
from pathlib import Path

import numpy as np
from analyse_variance_convergence import interval
from analyse_variance_resolution import analyse
from variance_convergence import ROOT

PAIRS = [("equal3200", 3200), (3200, 6400)]


def plot(cases, f, autos, crosses, protocol, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    length = max(protocol["usable_lengths"])
    comparisons = [(*PAIRS[0], "#2864A5", "Equal-energy 3200 / uniform 3200", "placement"),
                   (*PAIRS[1], "#52744F", "Uniform 3200 / uniform 6400", "refinement")]
    for coarse, fine, color, label, kind in comparisons:
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        for ax, metric, columns, title in zip(axes, ["mean", "variance", "variance"],
                                             [autos, autos, crosses],
                                             ["Mean auto-spectra", "Auto-spectral variances", "Cross-component variances"]):
            a, b = cases[coarse, length], cases[fine, length]
            ratio = np.median(a[metric][:, columns] / b[metric][:, columns], axis=1)
            boot = np.median(a["b"+metric][:, :, columns] / b["b"+metric][:, :, columns], axis=2)
            low, high = interval(boot)
            ax.plot(f, ratio, color=color, label=label)
            ax.fill_between(f, low, high, color=color, alpha=.15)
            ax.axhspan(.9, 1.1, color="#888888", alpha=.12, label="10% comparison band")
            ax.axhline(1, color="#666666", ls="--", lw=1)
            ax.set(title=title, xlabel="Frequency (Hz)", ylabel="Median ratio across components")
            ax.grid(alpha=.16)
        axes[0].legend(frameon=False, fontsize=8)
        fig.suptitle(f"Grid {kind}: {label}; {length} s, {protocol['seeds']} seeds per case")
        fig.text(.035, .02, "Shading: pointwise 95% paired whole-record bootstrap intervals for component medians. Individual entries can differ more.\n"
                 "Same underlying sea spectrum, integrated energy and physical model. Inverse and M unchanged; 6400 is not proven truth.",
                 fontsize=8, color="#555555")
        fig.tight_layout(rect=(.02, .12, .99, .95))
        filename = "variance_uniform.png" if kind == "refinement" else "variance_uniform_placement.png"
        fig.savefig(output / filename, dpi=170)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("variance_output/uniform"))
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args()
    analyse(args.data, args.output, pairs=PAIRS, prefix="uniform", plotter=plot)
