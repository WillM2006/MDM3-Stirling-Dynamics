"""
TEMPORARY experiment: compare_methods.py with a depth-dependent frequency cutoff, to stop the inversions amplifying noise.

Wave pressure and motion decay with depth z as exp(-k z), k = (2 pi f)^2 / g (deep water), so short waves vanish below
the sensor noise first and every method then divides noise by a tiny transfer function (E(f,theta) error ~ e^(2 k z)).
Here the frequency bins that cannot hold signal are dropped BEFORE the inversion and set to zero in the estimate (the
estimator says "no waves" there rather than amplified noise). Scores are on the full Welch band against the full
truth, exactly as compare_methods.py, so the cut is not hidden.
    --cutoff decay (default)  keep k z <= KZ_MAX:  f_c(z) = sqrt(g KZ_MAX / z) / (2 pi)
                              KZ_MAX = 7 (attenuation e^-7 ~ 1e-3): f_c = 0.30 Hz at 20 m, 0.21 Hz at 40 m, 0.15 Hz
                              at 80 m. The noise-limited SNR crosses 1 at k z ~ 8, but the last bins below that still
                              amplify noise into a spike at the cut edge; 7 removes it (E error on H_swell 40 m: kz 6
                              33 %, 7 36 %, 8 95 %; A_wind 40 m: 93 / 82 / 119 %; O_mixed 60 m: ~62 % for all three)
    --cutoff snr              data-driven, needs no depth: keep bins where the record's own noise-whitened auto-spectra
                              sum_c (S_cc / N_c - 1) >= SNR_MIN, N_c = channel c's noise floor measured above the wave band

Part 1, one record (--data, default H_swell at 40 m): the compare_methods figure (S(f) with the cut band shaded,
direction distribution, error bars, E(f,theta) polar maps) and its score table, with and without the cutoff.
Part 2, the depth study (--depth-root, default sensor_data_depth): Hs, Tp and spread errors (signed %, 100 (estimate -
truth) / truth; Tp and spread per wave system), mean direction error (|wrapped|, deg) and mean E(f,theta) error (%)
against depth; medians over seeds except E(f,theta) (mean). Rows = those errors, columns = methods, one line per sea
state; solid = with the cutoff, dashed = without (read from WM/output/data/inversion_runs_depth.csv).
Linear axes fitted to the solid lines, so the no-cutoff blow-up runs off the top. Drawn as its own figure, like
plot_depth_errors.py. The cutoff runs are inverted here, cached and resumable in two files in WM/output/data:
cutoff_runs_depth_<setting>[_tag].csv (E_err, Hs) and cutoff_params_depth_<setting>[_tag].csv (per-system Tp, direction,
spread); a run is only inverted if a cache lacks it, and a cache with other columns is never overwritten. The caches
hold raw estimates and truths, so changing how errors are expressed never needs a re-inversion.

    python compare_methods_cutoff.py --jobs 10                       # both parts, all methods
    python compare_methods_cutoff.py --methods NNLS,MEM --jobs 10
    python compare_methods_cutoff.py --kz-max 8                      # a gentler cut
    python compare_methods_cutoff.py --cutoff snr --snr-min 2
    python compare_methods_cutoff.py --no-sweep                      # Part 1 only (seconds)
    python compare_methods_cutoff.py --states A_wind_depth20,A_wind_depth40 --limit 1 --tag quick
Saves WM/output/images/compare_methods_cutoff_<setting>[_tag].png (Part 1) and
WM/output/images/depth_errors_cutoff_<setting>[_tag].png (Part 2). With the sweep already cached, --no-sweep just
re-reads the cache and redraws both.
"""
import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import compare_methods as cm  # noqa: E402
import plot_param_errors as ppe  # noqa: E402

KZ_MAX = 7.0                    # decay cutoff: keep bins with k z <= KZ_MAX
SNR_MIN = 1.0                   # snr cutoff: keep bins with noise-whitened signal power >= SNR_MIN
NOISE_BAND = (0.4, 0.7)         # Hz, above the wave band and below the decimation filter's roll-off: noise only
DATA = os.path.join(cm.REPO, "..", "..", "sensor_data_depth", "seastate_H_swell_depth40", "sensordata_0")
DEPTH_ROOT = os.path.join(cm.REPO, "..", "..", "sensor_data_depth")
NO_CUT_CACHE = "inversion_runs_depth.csv"                         # plot_depth_errors.py's cache, in <out>/data


# ---------------------------------------------------------------- the cutoff
def cutoff_mask(args, base, inp):
    """(L,) bool over the Welch bins inp['fb']: True = keep (invert), False = cut (estimate 0)"""
    fb = inp["fb"]
    if args.cutoff == "decay":
        g, depth = float(inp["rao"]["g"]), float(inp["params"].get("depth", inp["rao"]["depth"]))
        return (2 * np.pi * fb) ** 2 / g * depth <= args.kz_max
    channels = [c.strip() for c in args.channels.split(",")] if args.channels else list(base.CHANNELS)
    tc, X = base.read_channels(args.data, channels)                    # the full spectrum, above the band too
    if args.t_max:
        X = X[tc < tc[0] + args.t_max]
    seg_was, base.SEG = base.SEG, inp["seg"]                           # same segments as prepare()
    try:
        f, S = base.welch_matrix(X)
    finally:
        base.SEG = seg_was
    auto = np.real(np.einsum("lcc->lc", S))                            # (F, C) auto-spectra
    nb = (f >= NOISE_BAND[0]) & (f <= NOISE_BAND[1])
    noise = np.median(auto[nb], axis=0)                                # (C,) white-noise floor per channel
    band = np.isin(f, fb)
    return (auto[band] / noise - 1).sum(1) >= args.snr_min


def setting(args):
    return f"decay{args.kz_max:g}" if args.cutoff == "decay" else f"snr{args.snr_min:g}"


def run_cut(names, base, inp, truth, keep, repo):
    """compare_methods.run_methods, but each invert() sees only the kept bins and returns zeros in the cut ones, so
    every score is computed on the full band exactly as in compare_methods"""
    load_orig = cm.load_script

    def load(repo_, fname):
        mod = load_orig(repo_, fname)

        def invert(fb, Sb, h_all, tau, meff, **kw):
            x, cov = np.zeros((len(fb), h_all.shape[1])), None
            if keep.any():
                xk, cov, _ = mod.invert(fb[keep], Sb[keep], h_all[keep], tau, meff, **kw)
                x[keep] = np.real(xk)
            return x, cov, None
        return SimpleNamespace(invert=invert)

    cm.load_script = load
    try:
        return cm.run_methods(names, base, inp, truth, repo=repo, verbose=False)
    finally:
        cm.load_script = load_orig


# ---------------------------------------------------------------- part 2: the depth sweep (cached, resumable)
def sweep_one(state, run_dir, names, args_dict):
    args = argparse.Namespace(**{**args_dict, "data": run_dir})
    base = cm.import_base(args.repo)
    inp = cm.prepare(args, base, verbose=False)
    truth = cm.build_truth(args, base, inp, verbose=False)
    if truth is None:
        return None
    keep = cutoff_mask(args, base, inp)
    res = run_cut(names, base, inp, truth, keep, args.repo)
    n_sys = len(truth["se"].systems)
    if n_sys > ppe.NSYS:
        raise ValueError(f"{n_sys} wave systems but only NSYS={ppe.NSYS} are stored")
    row = dict(state=state, run=os.path.basename(run_dir), depth=float(inp["params"].get("depth", np.nan)),
               f_c=float(inp["fb"][keep].max()) if keep.any() else np.nan, n_kept=int(keep.sum()), n_systems=n_sys)
    for n in names:
        row[f"{n}_E_err"] = res[n]["E_err"]
        row[f"{n}_Hs"] = res[n]["Hs"]
        for i in range(1, ppe.NSYS + 1):                             # per-system Tp, mdir, spread, power
            row.update({f"{n}_{q}_{i}": res[n].get(f"{q}_{i}", np.nan) for q in ppe.SYS})
    row["true_Hs"] = truth["Hs"]
    for i in range(1, ppe.NSYS + 1):                                  # systems the sea does not have: NaN
        t = truth["sys"][i - 1] if i <= n_sys else {}
        row.update({f"true_{q}_{i}": (100 * t["power"] if q == "power" else t[q]) if t else np.nan for q in ppe.SYS})
    return row


def sweep_fields(names):
    """columns of the two sweep caches: the original one (E_err, Hs; unchanged so old caches stay valid) and the
    per-wave-system one (Tp, mdir, spread, power), kept in a separate file"""
    main = ["state", "run", "depth", "f_c", "n_kept", "true_Hs"] + [f"{n}_{q}" for n in names for q in ("E_err", "Hs")]
    sys_cols = lambda who: [f"{who}_{q}_{i}" for i in range(1, ppe.NSYS + 1) for q in ppe.SYS]   # noqa: E731
    params = (["state", "run", "depth", "n_systems"] + sys_cols("true")
              + [c for n in names for c in sys_cols(n)])
    return main, params


def sweep(args, names, caches):
    """invert every depth-study run some cache in caches = [(csv path, columns)] is missing, and append it to each cache
    that lacks it. A cache with other columns (other --methods) is never overwritten: the sweep stops instead"""
    done = []
    for path, fields in caches:
        d = set()
        if os.path.exists(path):
            with open(path) as fh:
                rd = csv.DictReader(fh)
                if rd.fieldnames != fields:
                    raise SystemExit(f"{path} has other columns (different --methods?); not touching it. Use --tag "
                                     f"for a separate cache, or --no-sweep to just plot")
                d = {(r["state"], r["run"]) for r in rd}
        done.append(d)
    runs = ppe.find_runs(args.depth_root, args.states, args.limit)
    todo = [(s, d) for s, d in runs if any((s, os.path.basename(d)) not in dn for dn in done)]
    print(f"depth sweep ({setting(args)}): {len(runs) - len(todo)} runs cached in every cache, {len(todo)} to invert "
          f"with {names}")
    if not todo:
        return
    failed = ppe.build_rao_caches(args, todo)                         # once per depth, before the runs share them
    todo = [(s, d) for s, d in todo if d not in failed]
    args_dict = vars(args).copy()
    handles, writers = [], []
    try:
        for (path, fields), dn in zip(caches, done):
            fh = open(path, "a" if dn else "w", newline="")
            handles.append(fh)
            wr = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            if not dn:
                wr.writeheader()
            writers.append(wr)
        _sweep_runs(args, names, todo, args_dict, done, handles, writers)
    finally:
        for fh in handles:
            fh.close()


def _sweep_runs(args, names, todo, args_dict, done, handles, writers):
    """invert the todo runs (in parallel with --jobs) and write each row to every cache that lacks it"""
    def write(row, i, label):
        if row is None:
            print(f"[{i}/{len(todo)}] {label}: no truth, skipped")
            return
        out = {k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()}
        for dn, fh, wr in zip(done, handles, writers):
            if (row["state"], row["run"]) not in dn:
                wr.writerow(out)
                fh.flush()
        print(f"[{i}/{len(todo)}] {label}  f_c = {row['f_c']:.3f} Hz")

    if args.jobs > 1:
        with ProcessPoolExecutor(args.jobs) as ex:
            futs = {ex.submit(sweep_one, s, d, names, args_dict): f"{s}/{os.path.basename(d)}" for s, d in todo}
            for i, fu in enumerate(as_completed(futs), 1):
                try:
                    write(fu.result(), i, futs[fu])
                except Exception as e:
                    print(f"[{i}/{len(todo)}] {futs[fu]}: FAILED {e!r}")
    else:
        for i, (s, d) in enumerate(todo, 1):
            label = f"{s}/{os.path.basename(d)}"
            try:
                write(sweep_one(s, d, names, args_dict), i, label)
            except Exception as e:
                print(f"[{i}/{len(todo)}] {label}: FAILED {e!r}")


def by_depth(path, names, q="E_err", k=None):
    """{(base state, method): (depths, value)} from a run cache; {} if missing or without the columns. Over the seeds
    at each depth: q = "E_err" mean E(f,theta) error (%); "Hs" median signed error 100 (estimate - truth) / truth (%);
    "Tp", "spread" the same signed % error for wave system k; "mdir" median |wrapped estimate - truth| of system k
    (deg, the one error not in %)"""
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    out = {}
    if not rows:
        return out
    col = lambda key: np.array([float(r[key]) for r in rows])          # noqa: E731
    st = np.array([ppe.base_state(r["state"]) for r in rows])
    dp = col("depth")
    fn = np.mean if q == "E_err" else np.median
    for n in names:
        key = f"{n}_{q}" if k is None else f"{n}_{q}_{k}"
        if key not in rows[0]:
            continue
        if q == "Hs":
            v = cm.rel_err(col(key), col("true_Hs"))                     # an estimate of 0 (all cut) is -100 %
        elif q == "E_err":
            v = col(key)
        elif q == "mdir":
            v = np.abs(ppe.wrap(col(key) - col(f"true_{q}_{k}")))
        else:                                                            # Tp, spread
            v = cm.rel_err(col(key), col(f"true_{q}_{k}"))
        for s in np.unique(st):
            ds = np.unique(dp[st == s])
            y = []
            for d in ds:
                x = v[(st == s) & (dp == d) & np.isfinite(v)]
                y.append(fn(x) if x.size else np.nan)
            y = np.array(y)
            if np.isfinite(y).any():                                     # a system this sea does not have: skip
                out[s, n] = (ds, y)
    return out


DEPTH_ROWS = [  # (quantity, per wave system?, y label, grey reference line = perfect estimate)
    ("Hs", False, "Hs error (%)\n100 (estimate - truth) / truth", 0.0),
    ("Tp", True, "Tp error (%)\n100 (estimate - truth) / truth", 0.0),
    ("spread", True, "directional spread error (%)\n100 (estimate - truth) / truth", 0.0),
    ("mdir", True, "mean direction error (deg)\n|estimate - truth|, 0 = perfect", 0.0),
    ("E_err", False, "E(f, θ) error (%)\nmean over seeds, 100 % = no waves", 100.0),
]
SYS_MARK = {None: None, 1: None, 2: "^"}                                # wave system 2 gets markers


def plot_depth(names, cut_csv, cut_params_csv, nocut_csv, label, out_png):
    """errors vs depth, rows = DEPTH_ROWS, columns = methods; solid = with the cutoff, dashed = without. Linear y
    axes fitted to the solid lines (the no-cutoff blow-up runs off the top). Returns False when nothing is cached"""
    import matplotlib.pyplot as plt
    data = {}                                                           # (row, system) -> (cut, nocut)
    for q, per_sys, *_ in DEPTH_ROWS:
        for k in (range(1, ppe.NSYS + 1) if per_sys else [None]):
            cut_src = cut_params_csv if per_sys else cut_csv
            data[q, k] = (by_depth(cut_src, names, q, k), by_depth(nocut_csv, names, q, k))
    keys = [key for cut, nocut in data.values() for key in list(cut) + list(nocut)]
    if not keys:
        print("no depth-sweep data cached, no depth figure")
        return False
    no_nocut = [n for n in names if not any(n == m for _, nocut in data.values() for _, m in nocut)]
    if no_nocut:                                                        # e.g. MLM: not in inversion_runs_depth.csv
        print(f"no no-cutoff data for {no_nocut} in {nocut_csv}: their columns have no dashed lines")
    kinds = {kd: i for i, kd in enumerate(ppe.KINDS)}
    states = sorted({s for s, _ in keys}, key=lambda s: (kinds.get(ppe.sea_kind(s), len(kinds)), s))
    colour = {s: cm.COLOURS[i % len(cm.COLOURS)] for i, s in enumerate(states)}
    fig, axes = plt.subplots(len(DEPTH_ROWS), len(names), figsize=(4.2 * len(names), 3.0 * len(DEPTH_ROWS)),
                             layout="constrained", squeeze=False, sharex=True, sharey="row")
    missing_cut = []
    for i, (q, per_sys, ylabel, ref) in enumerate(DEPTH_ROWS):
        solid = [ref]                                                   # y values the axis is fitted to
        for j, n in enumerate(names):
            ax = axes[i, j]
            ax.axhline(ref, color="0.45", lw=1, zorder=1)
            for k in (range(1, ppe.NSYS + 1) if per_sys else [None]):
                cut, nocut = data[q, k]
                for s in states:
                    for src, kw in ((nocut, dict(lw=1.2, ls="--", alpha=0.7)), (cut, dict(lw=2))):
                        if (s, n) in src:
                            d, y = src[s, n]
                            ok = np.isfinite(y)
                            ax.plot(d[ok], y[ok], color=colour[s], zorder=2, marker=SYS_MARK[k], ms=4, **kw)
                            if src is cut:
                                solid += list(y[ok])
            ax.grid(alpha=0.3, lw=0.5)
            ax.spines[["top", "right"]].set_visible(False)
            if i == 0:
                ax.set_title(n, fontsize=12, fontweight="bold")
            if i == len(DEPTH_ROWS) - 1:
                ax.set_xlabel("submarine depth (m)")
            if j == 0:
                ax.set_ylabel(ylabel)
        if len(solid) > 1:                                              # fit to the cutoff lines (shared per row)
            lo, hi = min(solid), max(solid)
            pad = 0.08 * (hi - lo or 1)
            axes[i, 0].set_ylim(max(lo - pad, 0) if q in ("mdir", "E_err") else lo - pad, hi + pad)   # signed % rows
        else:
            missing_cut.append(q)
    h = [plt.Line2D([], [], color=colour[s], lw=2, label=s) for s in states]
    h += [plt.Line2D([], [], color="0.3", lw=2, label=f"with cutoff ({label})"),
          plt.Line2D([], [], color="0.3", lw=1.2, ls="--", label="without cutoff (may run off the axis)"),
          plt.Line2D([], [], color="0.3", lw=1.2, marker="^", ms=4, label="wave system 2 (two-system seas)"),
          plt.Line2D([], [], color="0.45", lw=1, label="perfect estimate (E(f, θ) row: 100 % = no waves)")]
    fig.legend(handles=h, loc="outside upper center", ncol=4, frameon=False, fontsize=9)
    fig.text(0.5, -0.005, "lines: median over seeds at each depth (E(f, θ): mean). Hs and E(f, θ) are whole-sea; "
             "Tp, spread and direction are per wave system (no marker = system 1 or the only one).\n"
             "Hs, Tp and spread errors are signed % of the truth; direction is in degrees"
             + (f". No no-cutoff (dashed) data for {', '.join(no_nocut)}" if no_nocut else ""),
             ha="center", va="top", fontsize=9, color="0.3")
    if missing_cut:
        print(f"no cutoff data for {missing_cut} (run the sweep without --no-sweep to fill {cut_params_csv})")
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    print(f"saved {out_png}")
    return True


# ---------------------------------------------------------------- main
def main():
    ap = cm.build_parser()
    ap.description, ap.formatter_class = __doc__, argparse.RawDescriptionHelpFormatter
    ap.set_defaults(data=DATA)
    ap.add_argument("--cutoff", choices=["decay", "snr"], default="decay")
    ap.add_argument("--kz-max", type=float, default=KZ_MAX, help="decay cutoff: keep k z <= KZ_MAX (default 7)")
    ap.add_argument("--snr-min", type=float, default=SNR_MIN, help="snr cutoff: keep bins with SNR >= SNR_MIN")
    ap.add_argument("--depth-root", default=DEPTH_ROOT, help="folder holding the depth study's seastate_* folders")
    ap.add_argument("--states", default=None, help="comma-separated sea-state folders for the sweep (default: all)")
    ap.add_argument("--limit", type=int, default=None, help="sweep: only the first LIMIT seeds of each sea state")
    ap.add_argument("--jobs", type=int, default=1, help="sweep: runs inverted in parallel")
    ap.add_argument("--no-sweep", action="store_true", help="part 1 only")
    args = ap.parse_args()
    args.states = [s.strip() for s in args.states.split(",")] if args.states else None
    args.force = False
    names = [m.strip() for m in args.methods.split(",") if m.strip()]
    bad = [m for m in names if m not in cm.METHODS]
    if not names or bad:
        raise SystemExit(f"bad --methods {args.methods!r}; choose from {list(cm.METHODS)}")
    os.makedirs(args.out, exist_ok=True)
    suffix = setting(args) + (f"_{args.tag}" if args.tag else "")
    sweep_csv = cm.out_file(args.out, "data", f"cutoff_runs_depth_{suffix}.csv")      # E_err, Hs (original cache)
    params_csv = cm.out_file(args.out, "data", f"cutoff_params_depth_{suffix}.csv")   # per-system Tp, mdir, spread

    # ---- part 1: one record, with and without the cutoff
    base = cm.import_base(args.repo)
    inp = cm.prepare(args, base)
    truth = cm.build_truth(args, base, inp)
    if truth is None:
        raise SystemExit("no truth for this record, nothing to score")
    keep = cutoff_mask(args, base, inp)
    fb, beta = inp["fb"], inp["beta"]
    f_c = fb[keep].max() if keep.any() else np.nan
    print(f"cutoff {setting(args)}: keeping {keep.sum()}/{len(fb)} Welch bins, f <= {f_c:.3f} Hz "
          f"(T >= {1 / f_c:.1f} s) at depth {inp['params'].get('depth', float('nan')):g} m")
    results = run_cut(names, base, inp, truth, keep, args.repo)
    plain = cm.run_methods(names, base, inp, truth, repo=args.repo, verbose=False)
    cols = [("Hs (m)", "Hs", "{:.2f}"), ("Tp (s)", "Tp", "{:.1f}"), ("S(f) err %", "S_err", "{:.3g}"),
            ("D TV %", "D_tv", "{:.1f}"), ("E(f,th) err %", "E_err", "{:.3g}")]
    head = f"{'method':16s}" + "".join(f"{c[0]:>14s}" for c in cols)
    print("\n" + head + "\n" + "-" * len(head))
    print(f"{'true':16s}{truth['Hs']:>14.2f}{truth['Tp']:>14.1f}" + f"{'-':>14s}" * 3)
    for n in names:
        for lab, r in (("no cutoff", plain[n]), ("cutoff", results[n])):
            print(f"{n + ' ' + lab:16s}" + "".join(f"{fmt.format(r[k]):>14s}" for _, k, fmt in cols))

    # ---- part 2: the depth sweep
    if not args.no_sweep:
        main_fields, param_fields = sweep_fields(names)
        sweep(args, names, [(sweep_csv, main_fields), (params_csv, param_fields)])
    if args.no_plot:
        return

    # ---- figure 2: errors vs depth (plot_depth_errors style), from the caches
    plot_depth(names, sweep_csv, params_csv, cm.out_file(args.out, "data", NO_CUT_CACHE), setting(args),
               cm.out_file(args.out, "images", f"depth_errors_cutoff_{suffix}.png"))

    # ---- figure 1: compare_methods' panels for the one record
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from polar_plot import polar_spectrum
    style = cm.method_style(cm.METHODS)
    order = np.argsort(np.angle(np.exp(1j * beta)))
    deg = np.degrees(np.angle(np.exp(1j * beta[order])))
    n_maps = len(results) + 1
    ncol = max(3, int(np.ceil(n_maps / 2)))
    map_rows = int(np.ceil(n_maps / ncol))
    rows = 1 + map_rows
    fig = plt.figure(figsize=(4.6 * ncol, 4.0 * rows), layout="constrained")
    gs = GridSpec(rows, ncol, figure=fig, height_ratios=[1.1] + [1] * map_rows)

    ax = fig.add_subplot(gs[0, 0])                                   # frequency spectrum, cut band shaded
    for name, r in results.items():
        c, mk = style[name]
        ax.plot(fb, r["Sf"], color=c, lw=2, marker=mk, ms=4, markevery=4, label=name)
    ax.plot(fb, truth["S_jonswap"], ":", color="0.55", lw=1.5, label="JONSWAP target (smooth)")
    ax.plot(fb, truth["S"], "k--", lw=1.5, label="true (realised lines)")
    if (~keep).any():
        ax.axvspan(fb[~keep].min(), fb[-1], color="0.5", alpha=0.15, lw=0, label=f"cut ({setting(args)})")
    ax.legend(handles=ax.get_lines()[-2:] + ax.patches[-1:], fontsize=8, frameon=False)
    ax.set(xlabel="f (Hz)", ylabel="S(f)  (m$^2$/Hz)", title="Frequency spectrum (with cutoff)")

    ax2 = fig.add_subplot(gs[0, 1])                                  # heading distribution
    for name, r in results.items():
        c, mk = style[name]
        ax2.plot(deg, 100 * r["share"][order], color=c, lw=2, marker=mk, ms=4, label=name)
    ax2.plot(deg, 100 * truth["D"][order], "k--", lw=1.5, label="true")
    ax2.set(xlabel="heading (deg)", ylabel="share of power (%)", title="Direction distribution")

    ax3 = fig.add_subplot(gs[0, 2])                                  # error summary
    metrics = [("|Hs err|", lambda r: abs(r["Hs_err"])),
               ("S(f) err", lambda r: r["S_err"]), ("D TV", lambda r: r["D_tv"]), ("E(f,θ) err", lambda r: r["E_err"])]
    bw = 0.8 / len(results)
    for i, (name, r) in enumerate(results.items()):
        ax3.bar(np.arange(len(metrics)) + (i - (len(results) - 1) / 2) * bw, [fn(r) for _, fn in metrics],
                width=bw * 0.9, color=style[name][0], label=name)
    ax3.set_xticks(range(len(metrics)), [m[0] for m in metrics])
    ax3.set(ylabel="error (%)", title="Errors vs truth, with cutoff (lower is better)")
    for a in (ax, ax2, ax3):
        a.grid(alpha=0.3, lw=0.5)
        a.spines[["top", "right"]].set_visible(False)
    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=len(labels), frameon=False)

    maps = [("true", truth["x"])] + [(n, r["x"]) for n, r in results.items()]
    vmax = 1.2 * truth["x"].max()
    map_axes = []
    for i, (name, xm) in enumerate(maps):
        a = fig.add_subplot(gs[1 + i // ncol, i % ncol], projection="polar")
        im = polar_spectrum(a, fb, beta, xm, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        a.set_title(f"{name}: E(f, θ)", fontsize=10)
        map_axes.append(a)
    fig.colorbar(im, ax=map_axes, label="power per Hz per heading bin  (m$^2$/Hz)\n(angle = heading, radius = f in Hz)",
                 extend="both", shrink=0.8)
    out_png = cm.out_file(args.out, "images", f"compare_methods_cutoff_{suffix}.png")
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
