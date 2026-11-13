"""
Batch Bayesian inversion over the create_data.py sweep, scored against each run's true sea.

Put this file next to invert_waves.py, sea.py, submarine.py, sensors.py and submarine_extra.py, then run
    python batch_invert.py --root "sensor_data - Copy"              # inverts every finished run, writes batch_results.csv
    python batch_invert.py --root "sensor_data - Copy" --summary    # only rebuild batch_summary.csv / .png from the results
    python batch_invert.py --root "sensor_data - Copy" --model six_dof   # independent strip-theory model instead of
                                                                          # Capytaine -> batch_results_six_dof.csv
    (six_dof needs the updated six_dof.py and waves.py next to this script; --zeta-roll / --zeta-pitch set its damping)
    python batch_invert.py --root "sensor_data - Copy" --positive        # wave power >= 0 (invert_waves_positive.py)
                                                                          # -> batch_results_positive.csv

Each run folder needs params.csv, sensor_info.csv and the sensor csvs; runs without done.flag are skipped.
Re-running resumes: runs already in batch_results.csv are skipped (use --force to redo everything).

What it does per run
    1. transfer functions: ControlledSubmarine from sensors.py at the run's depth, with the viscous roll/pitch
       damping (B_visc) and the depth/heading controller, i.e. the same frequency-domain model as the generator.
       One BEM solve per hull and depth, cached in rao_cache_L<length>_R<radius>_depth<d>.npz.
       Hull length, radius, CG and sensor origin are read from submarine.py, for both models.
    2. inversion: invert_waves.py unchanged (same channels, band, Welch settings, prior, M_eff).
       The 36 generator headings are thinned to every --head-step'th one (default 3 -> 12 headings, 30 deg apart).
    3. truth: Sea() rebuilt from params.csv (same seed), so the comparison is with the sea that was simulated.

Metrics (one row per run)
    hs_err_pct      100 (Hs_est - Hs_actual) / Hs_actual
    z_hs            (Hs_est - Hs_actual) / posterior sd;  |z| < 1 should happen ~68% of the time if the error bar is honest
    tp_err_pct      100 (Tp_est - Tp) / Tp, Tp_est from the peak Welch bin (0.005 Hz steps, so coarse)
    dir_err_deg     estimated mean direction - true mean direction, wrapped to +-180
    wrong_dir       fraction of power in the wrong heading: 0.5 * sum |share_est - share_true| (0 perfect, 1 all wrong)
    spread_*_deg    circular spread sqrt(2(1-R)); true one computed on the inversion's heading grid, estimate from the
                    positive part of the estimate (negative power makes R meaningless)
    neg_frac        negative part of x as a fraction of sum |x|
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import invert_waves as iw  # noqa: E402

BASELINE = dict(Hs=1.7, Tp=8.0, s=10.0, gamma=3.3, depth=15.0, direction=90.0)   # from create_data.py (direction in deg)
SWEPT = list(BASELINE)
FIELDS = ["run", "swept", "Hs", "Tp", "s", "gamma", "depth", "direction", "seed",
          "Hs_actual", "Hs_est", "sd_Hs", "hs_err_pct", "z_hs", "Tp_est", "tp_err_pct",
          "dir_true_deg", "dir_est_deg", "dir_err_deg", "wrong_dir", "spread_true_deg", "spread_est_deg", "neg_frac"]


# ------------------------------------------------------------------ reading
def read_params(run_dir):
    with open(os.path.join(run_dir, "params.csv")) as fh:
        row = next(csv.DictReader(fh))
    return {k: float(v) for k, v in row.items()}


def read_channels_fast(data_dir, channels):
    """invert_waves.read_channels, but with pandas when available (genfromtxt is slow on 120k-row files)"""
    try:
        import pandas as pd
        cache = {}
        orig = np.genfromtxt

        def fast(fname, delimiter=",", names=True, **kw):
            if fname not in cache:
                df = pd.read_csv(fname)
                cache[fname] = df.to_records(index=False)
            return cache[fname]

        np.genfromtxt = fast
        try:
            return iw.read_channels(data_dir, channels)
        finally:
            np.genfromtxt = orig
    except ImportError:
        return iw.read_channels(data_dir, channels)


# ------------------------------------------------------------------ transfer functions with the new generator
_VEHICLE = {}


def vehicle(depth):
    """hull facts read from the generator files (submarine.py), so nothing is typed in twice:
    length, radius, CG depth below the hull centre (bg), sensor origin O below the hull centre (ref)"""
    if depth not in _VEHICLE:
        import submarine
        from sensors import ControlledSubmarine
        sub = ControlledSubmarine(depth=depth)                  # builds the mesh only, no BEM solve
        centre_z = float(sub.centre[2])
        _VEHICLE[depth] = dict(length=float(submarine.LENGTH), radius=float(submarine.RADIUS),
                               bg=centre_z - float(sub.body.center_of_mass[2]),
                               ref=centre_z - float(sub.body.rotation_center[2]))
    return _VEHICLE[depth]


def rao_six_dof(depth, n_freq, n_dir, omega_range, args):
    """independent model: strip theory (six_dof.generator_hull) on the generator's frequency and heading grids"""
    import six_dof
    v = vehicle(depth)
    om = np.linspace(*omega_range, n_freq)                       # = Submarine.bem_omegas
    theta = np.linspace(-np.pi, np.pi, n_dir, endpoint=False)    # = Sea.theta
    hull = six_dof.generator_hull(depth=depth, length=v["length"], radius=v["radius"], bg=v["bg"],
                                  ref_offset=v["ref"], zeta_roll=args.zeta_roll, zeta_pitch=args.zeta_pitch)
    rao = np.stack([hull.transfer(om, b) for b in theta], axis=1)   # pitch already bow down
    return dict(omega=om, theta=theta, rao=rao, rho=np.array(1025.0), g=np.array(9.81), depth=np.array(depth))


def rao_for_depth(depth, n_freq, n_dir, omega_range, args):
    v = vehicle(depth)
    # the cache name and contents carry the hull, so a cache from another hull can never be picked up
    tag = f"L{v['length']:g}_R{v['radius']:g}_depth{depth:g}"
    cache = os.path.join(args.cache_dir, f"rao_cache_{tag}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        if z["rao"].shape[:2] == (n_freq, n_dir) and str(z["hull"]) == tag:
            return {k: z[k] for k in z.files}
    from sensors import ControlledSubmarine
    from submarine import RHO, G
    print(f"  BEM solve for depth {depth:g} m (one-off, a few minutes) ...", flush=True)
    sub = ControlledSubmarine(n_freq=n_freq, n_dir=n_dir, omega_range=omega_range, depth=depth)
    sub.solve_hydrodynamics()
    sub.build_viscous_damping()
    sub.build_controller()
    om = sub.bem_omegas
    rao = np.zeros((len(om), len(sub.theta), 6), complex)
    for i, w in enumerate(om):
        Z = (-w**2 * (sub.M + sub.A[i]) - 1j * w * (sub.B[i] + sub.B_visc) + sub.C_hydro
             + np.diag(sub.Kp) - 1j * w * np.diag(sub.Kd))
        rao[i] = np.linalg.solve(Z, sub.F_exc[i].T).T
    out = dict(omega=om, theta=np.asarray(sub.theta), rao=rao, rho=np.array(RHO), g=np.array(G), depth=np.array(depth),
               hull=np.array(tag))
    np.savez(cache, **out)
    return out


# ------------------------------------------------------------------ one run
class RaoError(RuntimeError):
    pass


def circ_mean_spread(weights, beta):
    w = np.asarray(weights, float)
    c = np.sum(w * np.exp(1j * beta)) / np.sum(w)
    R = min(abs(c), 1.0)
    return np.degrees(np.angle(c)), np.degrees(np.sqrt(2 * (1 - R)))


def wrap_deg(a):
    return (a + 180) % 360 - 180


def process_run(run_dir, args, rao_store):
    from sea import Sea
    p = read_params(run_dir)
    n_bem = int(p.get("n_bem", p["n_freq"]))
    key = (p["depth"], n_bem, int(p["n_dir"]), p["omega_min"], p["omega_max"])
    if key not in rao_store:
        try:
            build = rao_six_dof if args.model == "six_dof" else rao_for_depth
            rao_store[key] = build(p["depth"], n_bem, int(p["n_dir"]),
                                   (p["omega_min"], p["omega_max"]), args)
        except Exception as e:
            raise RaoError(str(e)) from e
    rao = rao_store[key]
    keep = np.arange(0, rao["theta"].size, args.head_step)
    wg, beta = rao["omega"], rao["theta"][keep]
    R = rao["rao"][:, keep, :]
    consts = (float(rao["rho"]), float(rao["g"]), float(rao["depth"]))
    channels = [c.strip() for c in args.channels.split(",")]

    pos, _ = iw.sensor_positions(os.path.join(run_dir, "sensor_info.csv"))
    _, X = read_channels_fast(run_dir, channels)
    f, S = iw.welch_matrix(X)
    sel = (f >= iw.BAND[0]) & (f <= iw.BAND[1])
    fb, Sb, df = f[sel], S[sel], f[1] - f[0]
    h_all = np.zeros((len(fb), len(beta), len(channels)), complex)
    for l, fl in enumerate(fb):
        w = 2 * np.pi * fl
        H = iw.interp_complex(w, wg, R)
        for c, ch in enumerate(channels):
            h_all[l, :, c] = iw.channel_operator(ch, H, w, beta, pos, consts)
    x, cov, _ = iw.invert(fb, Sb, h_all, args.tau, args.meff)

    # estimate
    S_f = x.sum(1)
    m0 = S_f.sum() * df
    hs = 4 * np.sqrt(max(m0, 0.0))
    var_m0 = df**2 * sum(np.ones(len(beta)) @ cov[l] @ np.ones(len(beta)) for l in range(len(fb)))
    sd_hs = 2 * np.sqrt(var_m0) / np.sqrt(max(m0, 1e-12))
    P = x.sum(0)
    share_est = P / P.sum()
    tp_est = 1 / fb[np.argmax(S_f)]
    dir_est, _ = circ_mean_spread(P, beta)
    _, spread_est = circ_mean_spread(np.clip(P, 0, None), beta)

    # truth: the simulated sea, its 36 headings grouped onto the inversion headings (nearest, circular)
    sea = Sea(n_freq=int(p["n_freq"]), n_dir=int(p["n_dir"]), omega_range=(p["omega_min"], p["omega_max"]),
              Hs=p["Hs"], Tp=p["Tp"], direction=p["direction"], gamma=p["gamma"], s=p["s"], seed=int(p["seed"]),
              uniform_frequencies=bool(p.get("uniform_frequencies", 0)))
    frac = sea.D * sea.delta_theta                                      # share of power per generator heading
    nearest = np.argmin(abs(wrap_deg(np.degrees(sea.theta[:, None] - beta[None, :]))), axis=1)
    share_true = np.bincount(nearest, weights=frac, minlength=len(beta))
    share_true /= share_true.sum()
    dir_true = np.degrees(p["direction"])
    _, spread_true = circ_mean_spread(share_true, beta)

    if args.save_spectra:
        # the estimate and the truth on the same (frequency, heading) grid, for spectrum plots (make_figures.py)
        E = sea.wave_amplitudes**2 / 2                                  # variance of each wave component
        fbin = np.round((sea.f - fb[0]) / df).astype(int)
        T = np.zeros_like(x)
        for n in range(len(sea.f)):
            if 0 <= fbin[n] < len(fb):
                np.add.at(T[fbin[n]], nearest, E[n] / df)
        name = os.path.relpath(run_dir, args.root).replace(os.sep, "__").replace("/", "__") + ".npz"
        os.makedirs(args.save_spectra, exist_ok=True)
        np.savez(os.path.join(args.save_spectra, name), f=fb, beta=beta, x=x, true_binned=T,
                 true_Sf=sea.jonswap(fb) * (p["Hs_actual"] / p["Hs"])**2,     # smooth target, scaled to Hs_actual
                 theta_gen=sea.theta, D_gen=sea.D, Hs_est=hs, Hs_actual=p["Hs_actual"],
                 **{k: p[k] for k in ("Hs", "Tp", "s", "gamma", "depth", "direction", "seed")})

    hs_true = p["Hs_actual"]
    dirdeg = round(float(np.degrees(p["direction"])), 3)
    vals = dict(Hs=p["Hs"], Tp=p["Tp"], s=p["s"], gamma=p["gamma"], depth=p["depth"], direction=dirdeg)
    off = [k for k in SWEPT if not np.isclose(vals[k], BASELINE[k])]
    return dict(run=os.path.relpath(run_dir, args.root), swept=off[0] if off else "baseline", **vals, seed=int(p["seed"]),
                Hs_actual=hs_true, Hs_est=hs, sd_Hs=sd_hs, hs_err_pct=100 * (hs - hs_true) / hs_true,
                z_hs=(hs - hs_true) / sd_hs, Tp_est=tp_est, tp_err_pct=100 * (tp_est - p["Tp"]) / p["Tp"],
                dir_true_deg=dir_true, dir_est_deg=dir_est, dir_err_deg=wrap_deg(dir_est - dir_true),
                wrong_dir=0.5 * np.abs(share_est - share_true).sum(), spread_true_deg=spread_true,
                spread_est_deg=spread_est, neg_frac=np.maximum(-x, 0).sum() / np.abs(x).sum())


# ------------------------------------------------------------------ summary over seeds
METRICS = [("hs_err_pct", "Hs error (%)"), ("dir_err_deg", "direction error (deg)"),
           ("wrong_dir", "power in wrong heading"), ("neg_frac", "negative power fraction"),
           ("cover", "1-sigma coverage of Hs")]


def summarise(results_csv, out_prefix):
    import pandas as pd
    df = pd.read_csv(results_csv)
    df["cover"] = (df["z_hs"].abs() < 1).astype(float)
    rows = []
    for name in SWEPT:
        sub = df[(df["swept"] == name) | (df["swept"] == "baseline")]
        for v, g in sub.groupby(name):
            r = dict(sweep=name, value=v, n_seeds=len(g))
            for m, _ in METRICS:
                r[m + "_mean"], r[m + "_sd"] = g[m].mean(), g[m].std(ddof=1) if len(g) > 1 else np.nan
            rows.append(r)
    summ = pd.DataFrame(rows)
    summ.to_csv(out_prefix + ".csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, grid, mark = "#1f1f1f", "#6b6b6b", "#e4e4e4", "#2f6db3"
    fig, axs = plt.subplots(len(METRICS), len(SWEPT), figsize=(3.1 * len(SWEPT), 2.3 * len(METRICS)),
                            sharey="row", squeeze=False)
    for j, name in enumerate(SWEPT):
        s = summ[summ["sweep"] == name].sort_values("value")
        for i, (m, label) in enumerate(METRICS):
            ax = axs[i, j]
            ax.errorbar(s["value"], s[m + "_mean"], yerr=s[m + "_sd"], color=mark, lw=2, marker="o", ms=5,
                        capsize=3, elinewidth=1)
            ref = 0.68 if m == "cover" else 0.0
            ax.axhline(ref, color=muted, lw=1, ls="--")
            ax.axvline(BASELINE[name], color=grid, lw=6, zorder=0)
            ax.grid(color=grid, lw=0.6)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.tick_params(colors=muted, labelsize=8)
            if j == 0:
                ax.set_ylabel(label, color=ink, fontsize=9)
            if i == len(METRICS) - 1:
                ax.set_xlabel(name + (" (deg)" if name == "direction" else ""), color=ink, fontsize=9)
    fig.suptitle("Inversion quality over the one-at-a-time sweep: mean and sd over seeds "
                 "(grey band = baseline value; dashed = ideal)", color=ink, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_prefix + ".png", dpi=130)
    print(f"saved {out_prefix}.csv and {out_prefix}.png")


# ------------------------------------------------------------------ main
def preflight():
    """stop straight away (not 500 times) if the generator files found are not the ones that made the data"""
    import inspect
    import sea
    import submarine
    import sensors
    print("using generator files:")
    for m in (sea, submarine, sensors):
        print(f"   {m.__name__:10s} {os.path.abspath(m.__file__)}")
    print(f"hull from submarine.py: length {submarine.LENGTH:g} m, radius {submarine.RADIUS:g} m")
    need = {"n_freq", "n_dir", "omega_range", "depth", "seed"}
    missing = need - set(inspect.signature(submarine.Submarine.__init__).parameters)
    if missing or not hasattr(submarine, "ZETA_VISC"):
        raise SystemExit(f"\nThis submarine.py is an older version (Submarine() has no {sorted(missing) or 'viscous damping'}).\n"
                         "Put batch_invert.py and invert_waves.py in the folder holding the sea.py, submarine.py, sensors.py\n"
                         "and submarine_extra.py that create_data.py used, and run it from there.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="folder holding the sensordata_* folders")
    ap.add_argument("--out", default=None, help="default batch_results.csv (capytaine) or batch_results_six_dof.csv")
    ap.add_argument("--model", choices=["capytaine", "six_dof"], default="capytaine",
                    help="capytaine = the generator's own transfer functions (best case); six_dof = independent strip theory")
    ap.add_argument("--positive", action="store_true", help="use invert_waves_positive.py (wave power >= 0)")
    ap.add_argument("--zeta-roll", type=float, default=0.05, help="six_dof only: assumed roll damping ratio")
    ap.add_argument("--zeta-pitch", type=float, default=0.0, help="six_dof only: assumed pitch damping ratio")
    ap.add_argument("--cache-dir", default=".")
    ap.add_argument("--head-step", type=int, default=3, help="use every n-th of the 36 headings (3 -> 12 headings)")
    ap.add_argument("--tau", type=float, default=iw.TAU)
    ap.add_argument("--meff", type=float, default=iw.M_EFF)
    ap.add_argument("--channels", default=",".join(iw.CHANNELS))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--save-spectra", default=None, metavar="FOLDER",
                    help="also save each run's estimated and true spectrum (one small .npz per run) in FOLDER, "
                         "for the spectrum plots in make_figures.py")
    ap.add_argument("--only", default=None,
                    help="only runs whose folder path contains one of these comma-separated pieces, "
                         "e.g. 'Tp8_s10_gamma3.3_depth15_direction90' (the baseline) or 'direction90' ")
    ap.add_argument("--summary", action="store_true", help="only rebuild the summary from the results csv")
    args = ap.parse_args()
    if args.out is None:
        args.out = ("batch_results" + ("_six_dof" if args.model == "six_dof" else "")
                    + ("_positive" if args.positive else "") + ".csv")
    if args.positive:
        import invert_waves_positive
        iw.invert = invert_waves_positive.invert            # the only change: x >= 0 (NNLS)
        print("inversion: positive (invert_waves_positive.py)")

    prefix = os.path.splitext(args.out)[0].replace("results", "summary")
    prefix = prefix if prefix != os.path.splitext(args.out)[0] else prefix + "_summary"
    if args.summary:
        summarise(args.out, prefix)
        return

    preflight()
    if args.model == "six_dof":
        import six_dof
        if not hasattr(six_dof, "generator_hull"):
            raise SystemExit(f"{os.path.abspath(six_dof.__file__)} is the old six_dof.py: put the updated one next to this script.")
        print(f"model: six_dof strip theory ({os.path.abspath(six_dof.__file__)}), "
              f"zeta_roll {args.zeta_roll}, zeta_pitch {args.zeta_pitch}")
    runs = sorted(os.path.dirname(pth) for pth in glob.glob(os.path.join(args.root, "**", "params.csv"), recursive=True)
                  if os.path.exists(os.path.join(os.path.dirname(pth), "done.flag")))
    if args.only:
        keys = [k.strip() for k in args.only.split(",") if k.strip()]
        runs = [r for r in runs if any(k in os.path.relpath(r, args.root) for k in keys)]
    done = set()
    if os.path.exists(args.out) and not args.force:
        with open(args.out) as fh:
            done = {r["run"] for r in csv.DictReader(fh)}
    todo = [r for r in runs if os.path.relpath(r, args.root) not in done]
    print(f"{len(runs)} finished runs found, {len(runs) - len(todo)} already scored, {len(todo)} to do")

    new_file = args.force or not os.path.exists(args.out)
    rao_store = {}
    with open(args.out, "w" if new_file else "a", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            wr.writeheader()
        for i, run_dir in enumerate(todo, 1):
            try:
                row = process_run(run_dir, args, rao_store)
            except RaoError as e:                       # the model itself failed: every run would fail, so stop
                raise SystemExit(f"\nStopped: building the transfer functions failed ({e}).")
            except Exception as e:                      # one bad run should not stop the batch
                print(f"[{i}/{len(todo)}] FAILED {os.path.relpath(run_dir, args.root)}: {e}", flush=True)
                continue
            wr.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()})
            fh.flush()
            print(f"[{i}/{len(todo)}] {row['run']}: Hs {row['Hs_est']:.2f} vs {row['Hs_actual']:.2f} m, "
                  f"dir err {row['dir_err_deg']:.0f} deg, wrong-heading {row['wrong_dir']:.2f}, "
                  f"neg {row['neg_frac']:.2f}", flush=True)
    summarise(args.out, prefix)


if __name__ == "__main__":
    main()