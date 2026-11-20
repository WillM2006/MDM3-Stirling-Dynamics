"""
Error in Hs, Tp and directional spread (signed %, 100 (estimate - truth) / truth), mean direction (|wrapped error|,
deg) and the full directional spectrum E(f, heading) (relative L2 error, %) against submarine depth, for every method.

Runs (or resumes) the inversions with the runner in plot_param_errors.py, by default on the depth study
(sensor_data_depth, cache WM/output/data/inversion_runs_depth.csv), so each method sees exactly the inputs it does in
compare_methods.py. Tp, spread and direction are per wave system, as described in plot_param_errors.py.

    python plot_depth_errors.py                     # the depth study, all methods
    python plot_depth_errors.py --jobs 6
    python plot_depth_errors.py --plot-only
    python plot_depth_errors.py --root ../sensor_data_states --tag ""   # any other data set with more than one depth

Figure (WM/output/images/plot_depth_errors[_tag].png): rows = Hs, Tp, spread, mean direction, E(f, heading); columns =
methods; x = depth from params.csv. One coloured line per sea state (its depth suffix removed) through the MEDIAN over
its seeds at each depth (the E(f, heading) row: the MEAN); for seas with two wave systems, system 1 is solid and
system 2 dashed (Tp, spread and direction are per system).
How to read each row (the grey line is a perfect estimate):
    Hs          100 (estimate - truth) / truth (%) on a symlog axis (linear within +-10 %, log beyond): 0 = right,
                +900 % = ten times too big, -100 % = no waves found
    Tp          100 (estimate - truth) / truth (%): below the line = period too short
    spread      100 (estimate - truth) / truth (%): above the line = energy smeared over too many headings
    direction   |estimate - truth| wrapped to 0-180 deg: always >= 0, higher = worse. Absolute because signed errors
                of +100 and -100 deg from different seeds would average to a misleading 0
    E(f,th)     100 ||x - x_true|| / ||x_true|| (%), compare_methods' E_err, log axis: 0 = perfect, higher = worse; the
                grey line is 100 %, the error of estimating no waves at all
E_err is stored in the run cache from this version on; an older cache is re-inverted (not usable with --plot-only).
A system a method missed (< 5% of its power in that system's cells, see plot_param_errors.py) has no estimate, so the
median is over the seeds that found it. Nothing is drawn when every run is at one depth.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_methods as cm  # noqa: E402
from plot_param_errors import KINDS, PARAMS, base_state, cli, read_rows, sea_kind, wrap  # noqa: E402

ROOT = os.path.join(cm.REPO, "..", "..", "sensor_data_depth")
YLABEL = {"Hs": "Hs error (%)\n100 (estimate - truth) / truth, symlog",
          "Tp": "Tp error (%)\n100 (estimate - truth) / truth",
          "spread": "directional spread error (%)\n100 (estimate - truth) / truth",
          "mdir": "mean direction error (deg)\n|estimate - truth|, 0 = perfect",
          "E_err": "E(f, θ) error (%)\nmean over seeds, log scale"}
ROWS = [p for p, _ in PARAMS] + ["E_err"]
WHOLE_SEA = ("Hs", "E_err")                                         # one line per sea; the rest one per wave system
LOG = ("E_err",)
SYMLOG = ("Hs",)                                                    # signed % that can blow up to 1e4 %
REF = {"Hs": 0.0, "Tp": 0.0, "spread": 0.0, "mdir": 0.0, "E_err": 100.0}   # grey line: perfect (E_err: no waves)
LS = {None: "-", 1: "-", 2: "--", 3: ":", 4: "-."}                    # line style per wave system (None: whole sea)


def agg_by(v, groups, keys, fn=np.median):
    """fn (median or mean) of v over each group, NaNs ignored; NaN for a group with no finite value"""
    out = []
    for k in keys:
        x = v[(groups == k) & np.isfinite(v)]
        out.append(fn(x) if x.size else np.nan)
    return np.array(out)


def plot_depth_errors(out_csv, names, out_png):
    """error against submarine depth; skipped (returns False) when every run is at one depth"""
    import matplotlib.pyplot as plt
    rows = read_rows(out_csv)
    missing = [n for n in names if f"{n}_Hs" not in rows[0]]           # cache made with other --methods: say so,
    if missing:                                                         # never advise a re-run that would wipe it
        have = [k[:-3] for k in rows[0] if k.endswith("_Hs") and k != "true_Hs"]
        raise SystemExit(f"{out_csv} has no columns for {missing}; it holds {have}. Use --methods {','.join(have)} "
                         f"(or a new --tag to invert the others into a separate cache)")
    if any(f"{n}_E_err" not in rows[0] for n in names):
        raise SystemExit(f"{out_csv} has no E_err columns (written by an older version); re-run without --plot-only "
                         f"(every run is inverted again)")
    col = lambda k: np.array([float(r[k]) for r in rows])               # noqa: E731
    depth = col("depth")
    depths = np.unique(depth[np.isfinite(depth)])
    if len(depths) < 2:
        print(f"{out_csv}: every run is at one depth, no depth figure")
        return False
    state = np.array([base_state(r["state"]) for r in rows])
    order = {k: i for i, k in enumerate(KINDS)}
    states = sorted(set(state), key=lambda s: (order.get(sea_kind(s), len(KINDS)), s))
    colour = {s: cm.COLOURS[i % len(cm.COLOURS)] for i, s in enumerate(states)}
    n_sys = col("n_systems")
    sys_of = {s: range(1, int(np.nanmax(n_sys[state == s])) + 1) for s in states}   # wave systems each sea has
    max_sys = max(len(r) for r in sys_of.values())
    true_tp = {(s, k): np.nanmedian(col(f"true_Tp_{k}")[state == s]) for s in states for k in sys_of[s]}

    def values(n, p, k):
        """per-run value plotted for method n, parameter p, wave system k (k=None: whole sea)"""
        if p == "E_err":
            return col(f"{n}_E_err")
        if p == "Hs":
            return cm.rel_err(col(f"{n}_Hs"), col("true_Hs"))           # signed %
        if p == "mdir":
            return np.abs(wrap(col(f"{n}_{p}_{k}") - col(f"true_{p}_{k}")))   # deg
        return cm.rel_err(col(f"{n}_{p}_{k}"), col(f"true_{p}_{k}"))    # Tp, spread: signed %

    fig, axes = plt.subplots(len(ROWS), len(names), figsize=(4.2 * len(names), 3.1 * len(ROWS)),
                             layout="constrained", squeeze=False, sharex=True, sharey="row")
    for i, p in enumerate(ROWS):
        fn = np.mean if p == "E_err" else np.median
        for j, n in enumerate(names):
            ax = axes[i, j]
            ax.axhline(REF[p], color="0.45", lw=1, zorder=1)
            for k in ([None] if p in WHOLE_SEA else range(1, max_sys + 1)):
                v = values(n, p, k)                                     # every run; grouped per state below
                for s in states:
                    if k is not None and k not in sys_of[s]:
                        continue
                    y = agg_by(v, np.where(state == s, depth, np.nan), depths, fn)
                    ok = np.isfinite(y)                                 # bridge depths where every seed missed
                    ax.plot(depths[ok], y[ok], color=colour[s], lw=2, ls=LS[k], zorder=2)
            if p in LOG:
                ax.set_yscale("log")
            elif p in SYMLOG:
                ax.set_yscale("symlog", linthresh=10)
            if p == "E_err" and j == 0:
                ax.text(depths[0], REF[p], "100 % = no better than estimating no waves", fontsize=8, color="0.3",
                        va="bottom")
            if p == "mdir":
                ax.set_ylim(0, 180)
                ax.set_yticks(range(0, 181, 45))
            ax.grid(alpha=0.3, lw=0.5)
            ax.spines[["top", "right"]].set_visible(False)
            if i == 0:
                ax.set_title(n, fontsize=12, fontweight="bold")
            if i == len(ROWS) - 1:
                ax.set_xlabel("submarine depth (m)")
            if j == 0:
                ax.set_ylabel(YLABEL[p])
    handles = []
    for s in states:
        for k in sys_of[s]:
            sysname = f", system {k}" if len(sys_of[s]) > 1 else ""
            handles.append(plt.Line2D([], [], color=colour[s], lw=2, ls=LS[k],
                                      label=f"{s}{sysname} (true Tp {true_tp[s, k]:.1f} s)"))
    handles.append(plt.Line2D([], [], color="0.45", lw=1, label="perfect estimate (E(f, θ) row: 100 %)"))
    fig.legend(handles=handles, loc="outside upper center", ncol=4, frameon=False, fontsize=9)
    counts = [c for s in states for d in depths if (c := int(np.sum((state == s) & (depth == d))))]
    n_seeds = f"{min(counts)}" if min(counts) == max(counts) else f"{min(counts)}-{max(counts)}"
    fig.text(0.5, -0.005, f"lines: median over the {n_seeds} seeds at each depth (E(f, θ): mean). Hs and E(f, θ) are "
             f"whole-sea; Tp, spread and direction are per wave system\n(solid = system 1 or the only system, "
             f"dashed = system 2). Hs, Tp and spread errors are signed % of the truth; direction is in degrees",
             ha="center", va="top", fontsize=9, color="0.3")
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    print(f"saved {out_png}  ({len(rows)} runs, depths {depths.tolist()} m)")
    return True


def main():
    args, names, out_csv = cli(__doc__, ROOT, tag="depth")
    if not args.no_plot:
        png = cm.out_file(args.out, "images", "plot_depth_errors" + (f"_{args.tag}" if args.tag else "") + ".png")
        plot_depth_errors(out_csv, names, png)


if __name__ == "__main__":
    main()
