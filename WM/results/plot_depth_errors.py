"""
Error (estimate - truth) in Hs, Tp, directional spread and mean direction against submarine depth, for every method.

Runs (or resumes) the inversions with the runner in plot_param_errors.py, by default on the depth study
(sensor_data_depth, cache WM/output/inversion_runs_depth.csv), so each method sees exactly the inputs it does in
compare_methods.py. Tp, spread and direction are per wave system, as described in plot_param_errors.py.

    python plot_depth_errors.py                     # the depth study, all methods
    python plot_depth_errors.py --jobs 6
    python plot_depth_errors.py --plot-only
    python plot_depth_errors.py --root ../sensor_data_states --tag ""   # any other data set with more than one depth

Figure (WM/output/plot_depth_errors[_tag].png): rows = error in Hs, Tp, spread, mean direction; columns = methods;
x = depth from params.csv. One line per sea state (its depth suffix removed) through the mean error at each depth.
Nothing is drawn when every run is at one depth.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_methods as cm  # noqa: E402
from plot_param_errors import (KINDS, NSYS, PARAMS, PER_SYSTEM, base_state, cli, read_rows, sea_kind,  # noqa: E402
                               system_values, wrap)

ROOT = os.path.join(cm.REPO, "..", "..", "sensor_data_depth")


def plot_depth_errors(out_csv, names, out_png):
    """error against submarine depth; skipped (returns False) when every run is at one depth"""
    import matplotlib.pyplot as plt
    rows = read_rows(out_csv)
    col = lambda k: np.array([float(r[k]) for r in rows])               # noqa: E731
    depth = col("depth")
    if len(np.unique(depth[np.isfinite(depth)])) < 2:
        print(f"{out_csv}: every run is at one depth, no depth figure")
        return False
    state = np.array([base_state(r["state"]) for r in rows])
    order = {k: i for i, k in enumerate(KINDS)}
    states = sorted(set(state), key=lambda s: (order.get(sea_kind(s), len(KINDS)), s))
    style = {s: (cm.COLOURS[i % len(cm.COLOURS)], cm.MARKERS[i % len(cm.MARKERS)]) for i, s in enumerate(states)}
    units = {"Hs": "m", "Tp": "s", "spread": "deg", "mdir": "deg"}

    fig, axes = plt.subplots(len(PARAMS), len(names), figsize=(3.9 * len(names), 3.0 * len(PARAMS)),
                             layout="constrained", squeeze=False, sharex=True, sharey="row")
    for i, (p, label) in enumerate(PARAMS):
        if p in PER_SYSTEM:                                             # one point per wave system
            t = system_values(col, "true", p)
            st, dp = np.tile(state, NSYS), np.tile(depth, NSYS)
        else:
            t, st, dp = col(f"true_{p}"), state, depth
        for j, n in enumerate(names):
            ax = axes[i, j]
            err = (system_values(col, n, p) if p in PER_SYSTEM else col(f"{n}_{p}")) - t
            if p == "mdir":
                err = wrap(err)
            ax.axhline(0, color="k", ls="--", lw=1, zorder=1)
            for s in states:
                c, mk = style[s]
                m = (st == s) & np.isfinite(err) & np.isfinite(dp)
                if not m.any():
                    continue
                ax.scatter(dp[m], err[m], s=22, marker=mk, alpha=0.75, zorder=3, linewidths=1, color=c)
                ds = np.unique(dp[m])
                ax.plot(ds, [err[m & (dp == d)].mean() for d in ds], color=c, lw=1.2, alpha=0.8, zorder=2)
            ax.grid(alpha=0.3, lw=0.5)
            ax.spines[["top", "right"]].set_visible(False)
            if i == 0:
                ax.set_title(n, fontsize=12, fontweight="bold")
            if i == len(PARAMS) - 1:
                ax.set_xlabel("depth (m)")
            if j == 0:
                ax.set_ylabel(f"{label.split(' (')[0]} error\n(estimate - truth, {units[p]})")
    handles = [plt.Line2D([], [], marker=style[s][1], color=style[s][0], label=s) for s in states]
    fig.legend(handles=handles, loc="outside upper center", ncol=len(handles), frameon=False)
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    print(f"saved {out_png}  ({len(rows)} runs, depths {sorted(set(depth[np.isfinite(depth)].tolist()))} m)")
    return True


def main():
    args, names, out_csv = cli(__doc__, ROOT, tag="depth")
    if not args.no_plot:
        png = os.path.join(args.out, "plot_depth_errors" + (f"_{args.tag}" if args.tag else "") + ".png")
        plot_depth_errors(out_csv, names, png)


if __name__ == "__main__":
    main()
