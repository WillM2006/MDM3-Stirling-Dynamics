"""
Run the NNLS, MLM and MEM inversions on the SAME sensor data and score each one against the true sea.

This script loads the data, the transfer functions and the Welch cross-spectra once, using the helpers in
invert_waves.py, and passes the identical inputs to the invert() of each script:
    NNLS    invert_waves_positive.py    Gaussian prior, x >= 0
    MLM     invert_waves_mlm.py         EMLM (Capon)
    MEM     invert_waves_mem.py         maximum entropy
So any difference between them comes from the estimator alone.

Scores (truth rebuilt from sensor_data/params.csv with sea.py, as in the inverse scripts):
    Hs, Tp                       estimate vs the true sea (Hs and the peak Welch bin of the wave power realised in BAND)
    mean dir, spread, dir err    circular mean and spread of the heading distribution, and mean-direction error (deg)
    sys<i> power/Tp/dir/spread   seas with several wave systems only: each (f, heading) cell is given to the true system
                                 with the most power there, and power share, Tp, mean direction and spread are taken inside
                                 each system's cells (same cells for truth and estimate). NaN = the method put < 5% of
                                 its power there, i.e. it missed that system
    S(f) err                     relative L2 error of the frequency spectrum on the Welch bins:  ||S - S_true|| / ||S_true||
                                 (S_true = the realised line powers smeared by the Welch window, see section 2)
    D TV                         total-variation distance of the heading shares: 1/2 sum |p_j - p_true_j|  (0 = identical)
    E(f,theta) err               relative L2 error of the full directional spectrum x[f, heading]
    neg                          share of |x| that is negative power (a sanity check)
    chi2/N                       noise-weighted misfit to the cross-spectra (Wishart variances as in invert_waves.py)
    time                         seconds spent in invert()

Lives in WM/results; the method scripts, invert_waves.py, polar_plot.py, sea.py etc. stay one folder up in WM. Reads
DATA_DIR below (or --data) and writes to WM/output, the folder invert_waves.py uses, so it shares the RAO cache
(WM/output/rao_cache_depth<D>.npz). Works from any folder:
    python compare_methods.py
    python compare_methods.py --data path/to/sensordata_0
    python compare_methods.py --methods NNLS,MEM              # a subset
    python compare_methods.py --model six_dof                 # transfer functions from six_dof.py
    python compare_methods.py --t-max 300                     # only the first 300 s of the record
Saves WM/output/compare_methods.csv (the score table) and .png  (compare_methods_<tag>.* with --tag <tag>).
prepare(), build_truth() and run_methods() are reused by plot_results.py.
"""
import argparse
import csv
import importlib.util
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))   # .../WM/results
REPO = os.path.dirname(HERE)                        # .../WM: invert_waves*.py, polar_plot.py, submarine.py, sea.py, sensors.py
OUT = os.path.join(REPO, "output")                  # score table, figure and RAO cache (same folder as invert_waves.py)
# ---------------------------------------------------------------- input data (edit this to run on another record)
DATA_DIR = os.path.join(REPO, "..", "..", "sensor_data_states", "seastate_A_wind", "sensordata_0")
# name -> (script, extra keyword arguments for its invert())
METHODS = {
    "NNLS": ("invert_waves_positive.py", {}),
    "MLM": ("invert_waves_mlm.py", {}),
    "MEM": ("invert_waves_mem.py", {}),
}
# categorical slots 1-6 (validated palette: blue, orange, aqua, yellow, magenta, green) + a marker each, so identity
# is never colour alone; the truth is always black dashed
COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MARKERS = ["o", "s", "^", "D", "v", "P"]


def load_script(repo, fname):
    """import a script by file name"""
    path = os.path.join(repo, fname)
    spec = importlib.util.spec_from_file_location("inv_" + fname[:-3].replace("-", "_"), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def welch_kernel(f_out, f_lines, seg, fs):
    """K[l, n]: Welch (Hann, density-scaled, one-sided) response at f_out[l] to a sinusoid of unit power at f_lines[n].
    K = |W(f - f_n)|^2 / (fs sum w^2), W the DFT of the window; it integrates to 1 over f, so sum_n P_n K[:, n] is the
    expected Welch auto-spectrum of lines with powers P_n (the negative-frequency image is negligible in the wave band)."""
    from scipy.signal import get_window
    w = get_window("hann", int(seg * fs))                            # scipy.signal.csd's default window
    k = np.arange(len(w))
    W = (np.exp(-2j * np.pi * np.outer(f_out, k) / fs) * w) @ np.exp(2j * np.pi * np.outer(k, f_lines) / fs)   # (L, n)
    return np.abs(W) ** 2 / (fs * np.sum(w ** 2))


def circ_stats(share, beta):
    """circular mean and spread (deg) of a heading distribution that sums to 1"""
    c1 = np.sum(share * np.exp(1j * beta))
    return np.degrees(np.angle(c1)), np.degrees(np.sqrt(max(2 * (1 - abs(c1)), 0.0)))


def system_masks(se, fb, beta):
    """(n_sys, L, M) bool: which true wave system owns each (Welch bin, heading) cell = the one with the most power
    there. Cells no system reaches (beyond every system's ±90 deg cos^2s cutoff) go to the nearest system direction.
    With one system every cell is its own, so the per-system stats equal the whole-sea ones."""
    E = np.stack([se._jonswap_one(fb, q["Hs"], q["Tp"], q["gamma"])[:, None]
                  * se._cos2s(beta, q["direction"], q["s"])[None, :] for q in se.systems])
    dirs = np.array([q["direction"] for q in se.systems])
    nearest = np.abs(np.angle(np.exp(1j * (beta[None, :] - dirs[:, None])))).argmin(0)     # (M,)
    owner = np.where(E.max(0) > 0, E.argmax(0), nearest[None, :])                            # (L, M)
    return np.stack([owner == i for i in range(len(se.systems))])


def per_system_stats(x, masks, beta, fb, min_share=0.05):
    """power share, peak period (s), mean direction and spread (deg) of x inside each system's cells. A system holding
    <= min_share of the power was missed, so its Tp, direction and spread are NaN (they would be noise).
    Tp per system matters for mixed seas: their total S(f) has two peaks of similar height, so the whole-sea argmax
    flips between the wind sea and the swell on small level errors, while each system's own peak is stable."""
    tot = x.sum()
    out = []
    for m in masks:
        xm = np.where(m, x, 0.0)
        P = xm.sum()
        if tot <= 0 or P <= min_share * tot:
            out.append(dict(power=P / tot if tot > 0 else np.nan, Tp=np.nan, mdir=np.nan, spread=np.nan))
            continue
        mdir, spread = circ_stats(xm.sum(0) / P, beta)
        out.append(dict(power=P / tot, Tp=1 / fb[np.argmax(xm.sum(1))], mdir=mdir, spread=spread))
    return out


def chi2_per_n(base, Sb, h_all, x, meff):
    """noise-weighted misfit of A x to the stacked cross-spectra, averaged over data points and bins"""
    pairs = base.stack_pairs(h_all.shape[2])
    tot, n = 0.0, 0
    for l in range(len(x)):
        A = base.operator_matrix(h_all[l], pairs)
        d = base.data_vector(Sb[l], pairs)
        r = (A @ x[l] - d) / np.sqrt(base.noise_variances(Sb[l], pairs, meff))
        tot, n = tot + r @ r, n + len(r)
    return tot / n


def method_style(names):
    """colour + marker per method, fixed by its place in METHODS, so a subset keeps its colours"""
    order = list(METHODS)
    return {n: (COLOURS[order.index(n) % len(COLOURS)], MARKERS[order.index(n) % len(MARKERS)]) for n in names}


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DATA_DIR)
    ap.add_argument("--repo", default=REPO, help="folder holding submarine.py, sea.py, sensors.py (default: WM)")
    ap.add_argument("--out", default=OUT, help="folder for the score table, figure and RAO cache (default: WM/output)")
    ap.add_argument("--methods", default=",".join(METHODS), help="comma-separated subset of " + ",".join(METHODS))
    ap.add_argument("--model", choices=["capytaine", "six_dof"], default="capytaine")
    ap.add_argument("--no-controller", action="store_true", help="capytaine model only: leave the controller out")
    ap.add_argument("--channels", default=None, help="default: CHANNELS in invert_waves.py")
    ap.add_argument("--tau", type=float, default=None, help="default: TAU in invert_waves.py (NNLS only)")
    ap.add_argument("--meff", type=float, default=None, help="default: M_EFF in invert_waves.py")
    ap.add_argument("--t-max", type=float, default=None,
                    help="use only the first T_MAX s of the record (Welch segments shrink to fit if T_MAX < SEG)")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--tag", default="", help="suffix for the output files: compare_methods_<tag>.csv / .png")
    return ap


def import_base(repo):
    """invert_waves.py from the WM folder: the same module the method scripts import"""
    if repo not in sys.path:
        sys.path.insert(0, repo)
    import invert_waves as base
    return base


# ---------------------------------------------------------------- 1. inputs, built once with invert_waves.py's helpers
def prepare(args, base, rao=None, verbose=True):
    """everything invert() needs, for the record in args.data. rao: transfer functions already loaded (else read
    from the cache in args.out, or built). Returns a dict (fb, Sb, df, h_all, beta, params, grid, seg, tau, meff, ...)"""
    channels = [c.strip() for c in args.channels.split(",")] if args.channels else list(base.CHANNELS)
    tau = base.TAU if args.tau is None else args.tau
    meff = base.M_EFF if args.meff is None else args.meff
    params = base.read_params(args.data)
    depth = params.get("depth", 15.0)
    grid = base.grid_kwargs(params) if params else {}                 # as invert_waves.main()
    if rao is None:
        os.makedirs(args.out, exist_ok=True)
        cache = os.path.join(args.out, f"rao_cache_depth{depth:g}.npz")  # the same cache invert_waves.py uses
        rao = (base.six_dof_rao(args.repo, depth, grid) if args.model == "six_dof"
               else base.load_rao(args.repo, cache, use_controller=not args.no_controller, depth=depth, grid=grid))
    wg, beta = rao["omega"], rao["theta"]
    consts = (float(rao["rho"]), float(rao["g"]), float(rao["depth"]))
    pos, _ = base.sensor_positions(os.path.join(args.data, "sensor_info.csv"))
    tc, X = base.read_channels(args.data, channels)
    if args.t_max is not None and args.t_max <= 0:
        raise ValueError(f"--t-max must be > 0 s, got {args.t_max}")
    if args.t_max:                                                    # record-length study: keep the first t_max s
        keep = tc < tc[0] + args.t_max
        tc, X = tc[keep], X[keep]
    seg = min(base.SEG, len(tc) / base.FS)                            # a record shorter than one segment is one segment
    seg_was, base.SEG = base.SEG, seg                                 # welch_matrix reads SEG from invert_waves
    try:
        f, S = base.welch_matrix(X)
    finally:
        base.SEG = seg_was
    sel = (f >= base.BAND[0]) & (f <= base.BAND[1])
    if sel.sum() == 0:
        raise ValueError(f"record too short: {len(tc) / base.FS:.0f} s gives no Welch bins in {base.BAND} Hz")
    fb, Sb, df = f[sel], S[sel], f[1] - f[0]
    h_all = np.zeros((len(fb), len(beta), len(channels)), complex)
    for l, fl in enumerate(fb):
        w = 2 * np.pi * fl
        H = base.interp_complex(w, wg, rao["rao"])
        for c, ch in enumerate(channels):
            h_all[l, :, c] = base.channel_operator(ch, H, w, beta, pos, consts)
    if verbose:
        print(f"transfer functions: {args.model}   channels: {channels}")
        print(f"record {tc[0]:.0f}-{tc[-1]:.0f} s, {len(fb)} Welch bins in {base.BAND[0]}-{base.BAND[1]} Hz, "
              f"tau={tau}, M_eff={meff}")
    return dict(fb=fb, Sb=Sb, df=df, h_all=h_all, beta=beta, params=params, grid=grid, seg=seg, tau=tau, meff=meff,
                t_rec=tc[-1] - tc[0] + 1 / base.FS, rao=rao)


# ---------------------------------------------------------------- 2. truth on the same (f, heading) grid
def build_truth(args, base, inp, verbose=True):
    """the simulated sea, smeared onto the Welch bins exactly as the record is; None if it cannot be rebuilt"""
    fb, beta, df, params, grid = inp["fb"], inp["beta"], inp["df"], inp["params"], inp["grid"]
    try:
        from sea import Sea
        systems = base.read_systems(args.data, params)
        if systems is None:
            raise ValueError("no systems.csv and no sea state in params.csv")
        se = Sea(**base.sea_grid(grid), seed=int(params.get("seed", 42)), systems=systems,
                 uniform_frequencies=bool(params.get("uniform_frequencies", 0)))
        if "Hs_actual" in params and not np.isclose(se.Hs_actual, params["Hs_actual"], rtol=1e-6):
            print(f"WARNING: rebuilt sea has Hs {se.Hs_actual:.4f} m but the record says {params['Hs_actual']:.4f} m, "
                  f"so the truth is not the simulated sea (sea.py differs from the one that made the data?)")
        if len(se.theta) != len(beta) or not np.allclose(np.angle(np.exp(1j * (se.theta - beta))), 0, atol=1e-9):
            raise ValueError("Sea heading grid differs from the transfer-function grid")
        # The simulated sea is a LINE spectrum (n_freq discrete frequencies), so the record does not contain the smooth
        # JONSWAP curve. The references are what an ideal estimator would get from this record, smeared over the Welch
        # bins by the Hann-window kernel K (unit area), the same smearing csd applies:
        #   E(f, theta): each component's power a^2/2                              (the expected directional spectrum)
        #   S(f):        the power each line actually has at the origin, |sum_j a_j e^(i phi_j)|^2 / 2. All headings of a
        #                line share ONE frequency with fixed phases, so they interfere rather than add in power; the
        #                realised line powers are what the record holds (this is the "Hs realised at the origin" that
        #                invert_waves.py prints). Against the expected sum a^2/2 every method shows ~40% S(f) error.
        # All three truths (Hs, S(f), E(f,theta)) use the realised, band-limited level, since every estimator is levelled
        # to the recorded auto-spectra; E keeps the expected heading shape a^2/2 rescaled per frequency to S_real, so a
        # method that matches the record exactly in S(f) is not penalised in E(f,theta) for the level.
        K = welch_kernel(fb, se.f, inp["seg"], base.FS)              # (L, n_freq)
        x_exp = K @ (se.wave_amplitudes ** 2 / 2)                    # expected: power per Hz per heading bin, as x
        z = (se.wave_amplitudes * np.exp(1j * se.phi)).sum(1)
        S_real = K @ (np.abs(z) ** 2 / 2)                            # realised line powers
        x_true = x_exp * (S_real / np.maximum(x_exp.sum(1), 1e-300))[:, None]
        D = x_true.sum(0) / x_true.sum()                             # share of power per heading bin, sums to 1
        mdir_t, spread_t = circ_stats(D, beta)
        masks = system_masks(se, fb, beta)
        # Tp: the peak Welch bin of the realised S(f), the same definition as every estimate (1/fb[argmax S]) and the
        # same realised level as Hs. The nominal se.Tp (the dominant system's input Tp) is the wrong reference for mixed
        # seas: e.g. O_mixed's 8 s wind sea and 14 s swell have near-equal peaks, and in the record the 8 s one is taller
        return dict(se=se, D=D, x=x_true, S=S_real, S_jonswap=se.jonswap(fb), Hs=4 * np.sqrt(S_real.sum() * df),
                    Hs_actual=se.Hs_actual, Tp=1 / fb[np.argmax(S_real)], Tp_nominal=se.Tp, mdir=mdir_t,
                    spread=spread_t, masks=masks, sys=per_system_stats(x_true, masks, beta, fb, min_share=0))
    except Exception as e:
        if verbose:
            print(f"(truth comparison skipped: {e!r})")
        return None


# ---------------------------------------------------------------- 3. run every method on the identical inputs
def run_methods(names, base, inp, truth, repo=REPO, verbose=True):
    """invert() of each method script (in repo) on the inputs from prepare(); scores against truth when there is one"""
    fb, Sb, h_all, beta, df = inp["fb"], inp["Sb"], inp["h_all"], inp["beta"], inp["df"]
    tau, meff = inp["tau"], inp["meff"]
    results = {}
    for name in names:
        fname, kw = METHODS[name][0], dict(METHODS[name][1])
        if verbose:
            print(f"\n--- {name} ({fname}{', ' + str(kw) if kw else ''})")
        mod = load_script(repo, fname)
        t0 = time.perf_counter()
        x, cov, _ = mod.invert(fb, Sb, h_all, tau, meff, **kw)
        dt = time.perf_counter() - t0
        Sf = x.sum(1)
        m0 = Sf.sum() * df
        tot = x.sum()
        share = x.sum(0) / tot if tot > 0 else np.full(len(beta), np.nan)
        mdir, spread = circ_stats(share, beta) if tot > 0 else (np.nan, np.nan)   # no power: direction undefined
        r = dict(x=x, Sf=Sf, share=share, time=dt, Hs=4 * np.sqrt(max(m0, 0.0)), Tp=1 / fb[np.argmax(Sf)],
                 mdir=mdir, spread=spread, neg=-100 * np.minimum(x, 0).sum() / max(np.abs(x).sum(), 1e-300),
                 chi2=chi2_per_n(base, Sb, h_all, x, meff))
        if truth is not None:
            r.update(S_err=100 * np.linalg.norm(Sf - truth["S"]) / np.linalg.norm(truth["S"]),
                     D_tv=100 * 0.5 * np.abs(share - truth["D"]).sum(),
                     E_err=100 * np.linalg.norm(x - truth["x"]) / np.linalg.norm(truth["x"]),
                     dir_err=np.degrees(np.angle(np.exp(1j * np.radians(mdir - truth["mdir"])))))
            for i, (s, t) in enumerate(zip(per_system_stats(x, truth["masks"], beta, fb), truth["sys"]), 1):
                r.update({f"Tp_{i}": s["Tp"], f"mdir_{i}": s["mdir"], f"spread_{i}": s["spread"],
                          f"power_{i}": 100 * s["power"],
                          f"dir_err_{i}": np.degrees(np.angle(np.exp(1j * np.radians(s["mdir"] - t["mdir"])))),
                          f"spread_err_{i}": s["spread"] - t["spread"],
                          f"power_err_{i}": 100 * (s["power"] - t["power"])})
        results[name] = r
        if verbose:
            print(f"    done in {dt:.2f} s")
    return results


def main():
    args = build_parser().parse_args()
    names = [m.strip() for m in args.methods.split(",") if m.strip()]
    bad = [m for m in names if m not in METHODS]
    if not names:
        raise SystemExit("--methods is empty")
    if bad:
        raise SystemExit(f"unknown method(s) {bad}; choose from {list(METHODS)}")
    base = import_base(args.repo)
    inp = prepare(args, base)
    truth = build_truth(args, base, inp)
    results = run_methods(names, base, inp, truth, repo=args.repo)
    fb, beta = inp["fb"], inp["beta"]

    # ---------------------------------------------------------------- 4. score table
    cols = [("Hs (m)", "Hs", "{:.2f}"), ("Tp (s)", "Tp", "{:.1f}"), ("mean dir", "mdir", "{:.1f}"),
            ("spread", "spread", "{:.1f}")]
    if truth is not None:
        cols += [("dir err", "dir_err", "{:.1f}"), ("S(f) err %", "S_err", "{:.1f}"), ("D TV %", "D_tv", "{:.1f}"),
                 ("E(f,th) err %", "E_err", "{:.1f}")]
        if len(truth["sys"]) > 1:
            for i in range(1, len(truth["sys"]) + 1):
                cols += [(f"sys{i} power %", f"power_{i}", "{:.0f}"), (f"sys{i} Tp", f"Tp_{i}", "{:.1f}"),
                         (f"sys{i} dir", f"mdir_{i}", "{:.1f}"),
                         (f"sys{i} spread", f"spread_{i}", "{:.1f}"), (f"sys{i} dir err", f"dir_err_{i}", "{:.1f}")]
    cols += [("neg %", "neg", "{:.1f}"), ("chi2/N", "chi2", "{:.2f}"), ("time (s)", "time", "{:.2f}")]
    head = f"{'method':8s}" + "".join(f"{c[0]:>14s}" for c in cols)
    print("\n" + head + "\n" + "-" * len(head))
    if truth is not None:
        tvals = {"Hs": truth["Hs"], "Tp": truth["Tp"], "mdir": truth["mdir"], "spread": truth["spread"]}
        for i, t in enumerate(truth["sys"], 1):
            tvals.update({f"power_{i}": 100 * t["power"], f"Tp_{i}": t["Tp"], f"mdir_{i}": t["mdir"],
                          f"spread_{i}": t["spread"]})
        print(f"{'true':8s}" + "".join(f"{(fmt.format(tvals[k]) if k in tvals else '-'):>14s}" for _, k, fmt in cols))
    for name, r in results.items():
        print(f"{name:8s}" + "".join(f"{fmt.format(r[k]):>14s}" for _, k, fmt in cols))
    print("\nerrors: lower is better (0 = perfect). D TV = % of the heading distribution in the wrong place. "
          "chi2/N near 1 = fits the cross-spectra to within the Wishart noise.")
    if truth is not None:
        print(f"true Hs/S(f)/E(f,th) use the wave power realised in the {base.BAND[0]}-{base.BAND[1]} Hz band "
              f"(Hs {truth['Hs']:.2f} m); the expected Hs of the discretised spectrum is {truth['Hs_actual']:.2f} m "
              f"(see the comment in section 2)")
        print(f"true Tp = peak Welch bin of the realised S(f) ({truth['Tp']:.1f} s); the nominal input Tp of the "
              f"dominant system is {truth['Tp_nominal']:.1f} s")
        if len(truth["se"].systems) > 1:
            print(f"NOTE: the true sea has {len(truth['se'].systems)} wave systems, so 'mean dir', 'spread' and 'dir err' "
                  f"(a vector average that can sit between the peaks) are not meaningful; judge direction by the "
                  f"per-system sys<i> columns (each system's own cells), D TV and E(f,th). sys<i> dir NaN = system missed")
    if truth is not None:
        ranked = sorted(results, key=lambda n: results[n]["E_err"])
        print("ranked by full directional-spectrum error: " + "  <  ".join(f"{n} ({results[n]['E_err']:.0f}%)" for n in ranked))

    os.makedirs(args.out, exist_ok=True)
    out = os.path.join(args.out, "compare_methods" + (f"_{args.tag}" if args.tag else ""))
    with open(out + ".csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["method"] + [c[0] for c in cols])
        if truth is not None:
            wr.writerow(["true"] + [f"{tvals[k]:.4g}" if k in tvals else "" for _, k, _ in cols])
        for name, r in results.items():
            wr.writerow([name] + [f"{r[k]:.4g}" for _, k, _ in cols])
    print(f"saved {out}.csv")

    if args.no_plot:
        return

    # ---------------------------------------------------------------- 5. figure
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from polar_plot import polar_spectrum                            # WM/polar_plot.py (WM is on sys.path)
    style = method_style(METHODS)
    order = np.argsort(np.angle(np.exp(1j * beta)))
    deg = np.degrees(np.angle(np.exp(1j * beta[order])))
    n_maps = len(results) + (truth is not None)
    ncol = max(3, int(np.ceil(n_maps / 2)))
    map_rows = int(np.ceil(n_maps / ncol))
    fig = plt.figure(figsize=(4.6 * ncol, 4.0 * (1 + map_rows)), layout="constrained")
    gs = GridSpec(1 + map_rows, ncol, figure=fig, height_ratios=[1.1] + [1] * map_rows)

    ax = fig.add_subplot(gs[0, 0])                                   # frequency spectrum
    for name, r in results.items():
        c, mk = style[name]
        ax.plot(fb, r["Sf"], color=c, lw=2, marker=mk, ms=4, markevery=4, label=name)
    if truth is not None:
        ax.plot(fb, truth["S_jonswap"], ":", color="0.55", lw=1.5, label="JONSWAP target (smooth)")
        ax.plot(fb, truth["S"], "k--", lw=1.5, label="true (realised lines)")
        ax.legend(handles=ax.get_lines()[-2:], fontsize=8, frameon=False)
    ax.set(xlabel="f (Hz)", ylabel="S(f)  (m$^2$/Hz)", title="Frequency spectrum")

    ax2 = fig.add_subplot(gs[0, 1])                                  # heading distribution
    for name, r in results.items():
        c, mk = style[name]
        ax2.plot(deg, 100 * r["share"][order], color=c, lw=2, marker=mk, ms=4, label=name)
    if truth is not None:
        ax2.plot(deg, 100 * truth["D"][order], "k--", lw=1.5, label="true")
    ax2.set(xlabel="heading (deg)", ylabel="share of power (%)", title="Direction distribution")

    ax3 = fig.add_subplot(gs[0, 2])                                  # error summary
    if truth is not None:
        metrics = [("|Hs err|", lambda r: 100 * abs(r["Hs"] - truth["Hs"]) / truth["Hs"]),
                   ("S(f) err", lambda r: r["S_err"]), ("D TV", lambda r: r["D_tv"]), ("E(f,θ) err", lambda r: r["E_err"])]
        bw = 0.8 / len(results)
        for i, (name, r) in enumerate(results.items()):
            vals = [fn(r) for _, fn in metrics]
            ax3.bar(np.arange(len(metrics)) + (i - (len(results) - 1) / 2) * bw, vals, width=bw * 0.9,
                    color=style[name][0], label=name)
        ax3.set_xticks(range(len(metrics)), [m[0] for m in metrics])
        ax3.set(ylabel="error (%)", title="Errors vs truth (lower is better)")
    else:
        ax3.axis("off")
    for a in (ax, ax2, ax3):
        a.grid(alpha=0.3, lw=0.5)
        a.spines[["top", "right"]].set_visible(False)
    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=len(labels), frameon=False)

    maps = ([("true", truth["x"])] if truth is not None else []) + [(n, r["x"]) for n, r in results.items()]
    ref = truth["x"] if truth is not None else np.abs(maps[0][1])
    vmax = 1.2 * ref.max()                                           # shared scale; sharper estimates saturate (clipped)
    map_axes = []
    for i, (name, xm) in enumerate(maps):
        a = fig.add_subplot(gs[1 + i // ncol, i % ncol], projection="polar")
        im = polar_spectrum(a, fb, beta, xm, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        a.set_title(f"{name}: E(f, θ)", fontsize=10)
        map_axes.append(a)
    fig.colorbar(im, ax=map_axes, label="power per Hz per heading bin  (m$^2$/Hz)\n(angle = heading, radius = f in Hz)",
                 extend="both", shrink=0.8)
    fig.savefig(out + ".png", dpi=130, bbox_inches="tight")
    print(f"saved {out}.png")


if __name__ == "__main__":
    main()
