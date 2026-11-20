"""
True vs estimated Hs, Tp, directional spread and mean direction for every method, over every run in the sea-state data.

For each run folder (sensor_data_states/seastate_*/sensordata_*, with done.flag) this reuses prepare(), build_truth()
and run_methods() from compare_methods.py, so each method sees exactly the inputs it does there. Per-run numbers are
cached in WM/output/data/inversion_runs[_tag].csv; re-running resumes (use --force to redo everything, --plot-only to
just redraw). plot_depth_errors.py uses the same runner and cache for the depth study.

    python plot_param_errors.py                          # all runs, all methods
    python plot_param_errors.py --jobs 6                 # parallel over runs
    python plot_param_errors.py --states O_mixed,A_wind  # a subset of sea states
    python plot_param_errors.py --limit 3                # the first 3 seeds per sea state (quick look)
    python plot_param_errors.py --plot-only

Figure (WM/output/images/plot_param_errors[_tag].png): rows = Hs, Tp, spread, mean direction; columns = methods. One
point per run, coloured and shaped by sea type; the black dashed line is estimate = truth. The axes are the values
themselves (m, s, deg); the bias and RMSE printed in each panel are of the signed % error 100 (estimate - truth) / truth
for Hs, Tp and spread, and in deg for mean direction (the one error not expressed in %).

Tp, spread and mean direction are PER WAVE SYSTEM. For seas with more than one system the whole-sea values mislead:
the mean direction is a vector average that can sit between the peaks, and Tp = argmax of the total S(f) flips between
a wind sea and a swell whose peaks are of similar height. So compare_methods.system_masks gives each (f, heading) cell
to the true system with the most power there and the stats are taken inside each system's cells (the same cells for
truth and estimate). A single-system run gives one point (= the whole-sea value), a two-system run (mixed, bimodal) two
points. A system the method missed (< 5% of its power in that system's cells) has no estimate; the panel counts them
as "missed". Direction errors are wrapped to +-180 deg, so the estimate is plotted as truth + wrapped error.

True Hs and Tp are those of the wave power realised in the record (Welch-smeared, band-limited), see build_truth().
"""
import argparse
import csv
import glob
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import compare_methods as cm  # noqa: E402

ROOT = os.path.join(cm.REPO, "..", "..", "sensor_data_states")
CACHE = "inversion_runs"                                           # WM/output/data/<CACHE>[_tag].csv
KINDS = ["wind", "swell", "mixed", "bimodal"]                       # sea type = last word of the state name
PARAMS = [("Hs", "Hs (m)"), ("Tp", "Tp (s)"), ("spread", "directional spread (deg)"), ("mdir", "mean direction (deg)")]
NSYS = 2                                                           # wave systems per sea stored (create_data makes <= 2)
SYS = ["Tp", "mdir", "spread", "power"]                            # per-system stats from compare_methods.per_system_stats
PER_SYSTEM = ("Tp", "spread", "mdir")                              # PARAMS plotted per system rather than whole-sea
FIELDS = (["state", "kind", "run", "depth", "n_systems"] + [f"true_{p}" for p, _ in PARAMS]
          + [f"true_{q}_{i}" for i in range(1, NSYS + 1) for q in SYS])


# ------------------------------------------------------------------ helpers shared with plot_depth_errors.py
def base_state(state):
    """the sea state without a depth-study suffix: A_wind_depth25 -> A_wind"""
    return re.sub(r"_depth[\d.]+$", "", state)


def sea_kind(state):
    return base_state(state).rsplit("_", 1)[1]


def wrap(a):
    return (np.asarray(a, float) + 180) % 360 - 180


def method_fields(names):
    """per-method columns: the PARAMS, the per-system SYS stats and E_err (relative L2 error of E(f, heading), %)"""
    return ([f"{n}_{p}" for n in names for p, _ in PARAMS]
            + [f"{n}_{q}_{i}" for n in names for i in range(1, NSYS + 1) for q in SYS]
            + [f"{n}_E_err" for n in names])


def system_values(col, prefix, p):
    """per-system values of p stacked over systems: (NSYS * runs,), system 1 of every run, then system 2 ..."""
    return np.concatenate([col(f"{prefix}_{p}_{i}") for i in range(1, NSYS + 1)])


def read_rows(out_csv):
    if not os.path.exists(out_csv):
        raise SystemExit(f"no run cache {out_csv}; run without --plot-only to invert the runs first")
    with open(out_csv) as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"no rows in {out_csv}")
    if "true_Tp_1" not in rows[0]:
        raise SystemExit(f"{out_csv} predates the per-system Tp/direction/spread columns; re-run without --plot-only "
                         f"(every run is inverted again)")
    return rows


# ------------------------------------------------------------------ running the inversions (cached)
def find_runs(root, states=None, limit=None):
    """[(state, run_dir)] for every finished run, sorted by state then seed"""
    runs = []
    for sdir in sorted(glob.glob(os.path.join(root, "seastate_*"))):
        state = os.path.basename(sdir)[len("seastate_"):]
        if states and state not in states:
            continue
        rdirs = [d for d in glob.glob(os.path.join(sdir, "sensordata_*")) if os.path.exists(os.path.join(d, "done.flag"))]
        rdirs.sort(key=lambda d: int(d.rsplit("_", 1)[1]))
        runs += [(state, d) for d in rdirs[:limit]]
    return runs


def one_run(state, run_dir, names, args_dict):
    """invert one run with every method; returns a row dict (None if the truth cannot be rebuilt)"""
    args = argparse.Namespace(**{**args_dict, "data": run_dir})
    base = cm.import_base(args.repo)
    inp = cm.prepare(args, base, verbose=False)
    truth = cm.build_truth(args, base, inp, verbose=False)
    if truth is None:
        return None
    n_sys = len(truth["se"].systems)
    if n_sys > NSYS:
        raise ValueError(f"{n_sys} wave systems but only NSYS={NSYS} are stored; raise NSYS in plot_param_errors.py")
    res = cm.run_methods(names, base, inp, truth, repo=args.repo, verbose=False)
    row = dict(state=state, kind=sea_kind(state), run=os.path.basename(run_dir),
               depth=float(inp["params"].get("depth", float("nan"))), n_systems=n_sys)
    row.update({f"true_{p}": truth[p] for p, _ in PARAMS})
    for i in range(1, NSYS + 1):                                      # systems the sea does not have: NaN
        t = truth["sys"][i - 1] if i <= n_sys else {}
        row.update({f"true_{q}_{i}": (100 * t["power"] if q == "power" else t[q]) if t else float("nan") for q in SYS})
    for n, r in res.items():
        row[f"{n}_E_err"] = r["E_err"]
        for p, _ in PARAMS:
            row[f"{n}_{p}"] = r[p]
        for i in range(1, NSYS + 1):
            row.update({f"{n}_{q}_{i}": r.get(f"{q}_{i}", float("nan")) for q in SYS})
    return row


def load_rao_cache(repo, cache, use_controller, depth, grid):
    """worker for build_rao_caches: load or build+save one transfer-function cache. Returns nothing, so the arrays are
    not pickled back to the parent"""
    cm.import_base(repo).load_rao(repo, cache, use_controller=use_controller, depth=depth, grid=grid)


def build_rao_caches(args, todo):
    """load or build every transfer-function cache the runs in todo need, once per cache and before any run starts.
    Without this, parallel runs at a depth with no cache each solve the same BEM (minutes) and write the same .npz at
    the same time. Caches are built in parallel across depths (up to --jobs at once). Returns the run folders whose
    cache could not be built, so the caller can leave them out instead of each one retrying the build."""
    if args.model != "capytaine":                                     # six_dof transfer functions: analytic, no cache
        return set()
    base = cm.import_base(args.repo)
    need = {}                                                         # cache path -> (depth, grid, [run folders])
    for _, d in todo:
        depth, grid, cache = cm.rao_source(args, base, base.read_params(d))
        if cache in need and need[cache][1] != grid:
            raise SystemExit(f"runs at depth {depth:g} m use different grids ({need[cache][1]} and {grid}), but share "
                             f"the cache {cache}; split them into separate --out folders")
        need.setdefault(cache, (depth, grid, []))[2].append(d)
    missing = sorted(dep for c, (dep, _, _) in need.items() if not os.path.exists(c))
    print(f"transfer functions for {len(need)} depth(s); "
          + (f"building {len(missing)} with no cache (minutes each): {missing} m" if missing else "all cached"))
    os.makedirs(args.out, exist_ok=True)
    jobs = [(c, dep, grid) for c, (dep, grid, _) in need.items()]
    failed = set()

    def fail(c, dep, e):
        print(f"transfer functions for depth {dep:g} m: FAILED {e!r}; its {len(need[c][2])} run(s) are skipped")
        failed.update(need[c][2])

    if args.jobs > 1 and len(missing) > 1:                            # else just load (cheap) in this process
        with ProcessPoolExecutor(min(args.jobs, len(jobs))) as ex:
            futs = {ex.submit(load_rao_cache, args.repo, c, not args.no_controller, dep, grid): (c, dep)
                    for c, dep, grid in jobs}
            for fu in as_completed(futs):
                try:
                    fu.result()
                except Exception as e:
                    fail(*futs[fu], e)
    else:
        for c, dep, grid in jobs:
            try:
                load_rao_cache(args.repo, c, not args.no_controller, dep, grid)
            except Exception as e:
                fail(c, dep, e)
    return failed


def run_all(args, names, out_csv):
    fields = FIELDS + method_fields(names)
    done = set()
    if os.path.exists(out_csv) and not args.force:
        with open(out_csv) as fh:
            rd = csv.DictReader(fh)
            if rd.fieldnames == fields:
                done = {(r["state"], r["run"]) for r in rd}
            else:
                print(f"{out_csv} has other columns (different --methods, or written by an older version); "
                      f"starting it again")
                args.force = True
    todo = [(s, d) for s, d in find_runs(args.root, args.states, args.limit) if (s, os.path.basename(d)) not in done]
    print(f"{len(done)} runs cached, {len(todo)} to invert with {names}")
    if not todo:
        return
    failed = build_rao_caches(args, todo)                            # once per depth, before the runs share them
    todo = [(s, d) for s, d in todo if d not in failed]
    if not todo:
        return
    mode = "w" if args.force or not done else "a"
    args_dict = {k: v for k, v in vars(args).items()}
    with open(out_csv, mode, newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        if mode == "w":
            wr.writeheader()

        def write(row, i, label):
            if row is None:
                print(f"[{i}/{len(todo)}] {label}: no truth, skipped")
                return
            wr.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()})
            fh.flush()
            print(f"[{i}/{len(todo)}] {label}")

        if args.jobs > 1:
            with ProcessPoolExecutor(args.jobs) as ex:
                futs = {ex.submit(one_run, s, d, names, args_dict): f"{s}/{os.path.basename(d)}" for s, d in todo}
                for i, fu in enumerate(as_completed(futs), 1):
                    try:
                        write(fu.result(), i, futs[fu])
                    except Exception as e:
                        print(f"[{i}/{len(todo)}] {futs[fu]}: FAILED {e!r}")
        else:
            for i, (s, d) in enumerate(todo, 1):
                label = f"{s}/{os.path.basename(d)}"
                try:
                    write(one_run(s, d, names, args_dict), i, label)
                except Exception as e:
                    print(f"[{i}/{len(todo)}] {label}: FAILED {e!r}")


def cli(doc, root, tag=""):
    """parse the shared options, run (or resume) the inversions; returns (args, method names, cache csv path)"""
    ap = cm.build_parser()
    ap.description = doc
    ap.formatter_class = argparse.RawDescriptionHelpFormatter
    ap.set_defaults(tag=tag)
    ap.add_argument("--root", default=root, help="folder holding the seastate_* folders")
    ap.add_argument("--states", default=None, help="comma-separated sea states, e.g. O_mixed,A_wind (default: all)")
    ap.add_argument("--limit", type=int, default=None, help="only the first LIMIT runs of each sea state")
    ap.add_argument("--jobs", type=int, default=1, help="runs inverted in parallel")
    ap.add_argument("--force", action="store_true", help="ignore the cache and redo every run")
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    args.states = [s.strip() for s in args.states.split(",")] if args.states else None
    names = [m.strip() for m in args.methods.split(",") if m.strip()]
    bad = [m for m in names if m not in cm.METHODS]
    if bad:
        raise SystemExit(f"unknown method(s) {bad}; choose from {list(cm.METHODS)}")
    os.makedirs(args.out, exist_ok=True)
    out_csv = cm.out_file(args.out, "data", CACHE + (f"_{args.tag}" if args.tag else "") + ".csv")
    if not args.plot_only:
        cm.import_base(args.repo)
        run_all(args, names, out_csv)
    return args, names, out_csv


# ------------------------------------------------------------------ the figure
def plot_param_errors(out_csv, names, out_png):
    import matplotlib.pyplot as plt
    rows = read_rows(out_csv)
    col = lambda k: np.array([float(r[k]) for r in rows])               # noqa: E731
    kind = np.array([r["kind"] for r in rows])
    multi = col("n_systems") > 1
    kinds = [k for k in KINDS if k in kind] + sorted(set(kind) - set(KINDS))
    style = {k: (cm.COLOURS[i % len(cm.COLOURS)], cm.MARKERS[i % len(cm.MARKERS)]) for i, k in enumerate(kinds)}

    fig, axes = plt.subplots(len(PARAMS), len(names), figsize=(3.9 * len(names), 3.6 * len(PARAMS)),
                             layout="constrained", squeeze=False)
    for i, (p, label) in enumerate(PARAMS):
        if p in PER_SYSTEM:                                             # one point per wave system
            t = system_values(col, "true", p)
            ests = {n: system_values(col, n, p) for n in names}
            kd = np.tile(kind, NSYS)
            have = np.isfinite(t)                                       # drop systems the sea does not have
            t, kd, ests = t[have], kd[have], {n: e[have] for n, e in ests.items()}
        else:
            t, kd = col(f"true_{p}"), kind
            ests = {n: col(f"{n}_{p}") for n in names}
        if p == "mdir":                                                 # plot truth + wrapped error
            t = wrap(t)
            ests = {n: t + wrap(e - t) for n, e in ests.items()}
        allv = np.concatenate([t] + list(ests.values()))
        allv = allv[np.isfinite(allv)]
        lo, hi = allv.min(), allv.max()
        pad = 0.05 * (hi - lo or 1)
        lim = (lo - pad, hi + pad)
        for j, n in enumerate(names):
            ax = axes[i, j]
            e = ests[n]
            ax.plot(lim, lim, "k--", lw=1, zorder=1)
            for k in kinds:
                c, mk = style[k]
                m = kd == k
                if m.any():
                    ax.scatter(t[m], e[m], s=22, marker=mk, alpha=0.75, zorder=2, linewidths=1, color=c)
            ok = np.isfinite(e)
            if p == "mdir":                                             # degrees: the one error not in %
                err, unit = wrap(e[ok] - t[ok]), " deg"
            else:                                                       # signed % of the truth
                err, unit = cm.rel_err(e[ok], t[ok]), " %"
                err = err[np.isfinite(err)]
            stats = (f"bias {np.mean(err):+.1f}{unit}\nRMSE {np.sqrt(np.mean(err ** 2)):.1f}{unit}" if err.size
                     else "no estimates")
            if p in PER_SYSTEM and (~ok).any():
                stats += f"\n{(~ok).sum()} system(s) missed"
            ax.text(0.03, 0.97, stats, transform=ax.transAxes, va="top", fontsize=8, color="0.25")
            ax.set(xlim=lim, ylim=lim, aspect="equal")
            ax.grid(alpha=0.3, lw=0.5)
            ax.spines[["top", "right"]].set_visible(False)
            if i == 0:
                ax.set_title(n, fontsize=12, fontweight="bold")
            if i == len(PARAMS) - 1:
                ax.set_xlabel("true")
            if j == 0:
                ax.set_ylabel(f"{label}\nestimated")
    handles = [plt.Line2D([], [], ls="", marker=style[k][1], color=style[k][0], label=k) for k in kinds]
    handles += [plt.Line2D([], [], ls="--", color="k", label="estimate = truth")]
    fig.legend(handles=handles, loc="outside upper center", ncol=len(handles), frameon=False)
    if multi.any():
        fig.text(0.5, -0.005, "Tp, spread and direction: one point per wave system (multi-system seas split into each "
                 "system's own (f, heading) cells)", ha="center", va="top", fontsize=9, color="0.3")
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    print(f"saved {out_png}  ({len(rows)} runs)")


def main():
    args, names, out_csv = cli(__doc__, ROOT)
    if not args.no_plot:
        png = cm.out_file(args.out, "images", "plot_param_errors" + (f"_{args.tag}" if args.tag else "") + ".png")
        plot_param_errors(out_csv, names, png)


if __name__ == "__main__":
    main()
