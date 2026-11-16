"""
Minimal Bayesian inversion: sensor CSVs from sensors.py -> directional wave spectrum.

Method (derivation in minimal_bayes_inversion.md), applied independently at each Welch frequency:
    data     d = Welch cross-spectra of the chosen sensor channels (real/imag parts stacked)
    model    d = A x          x_j = wave power per Hz from BEM heading j, A from the transfer functions
    noise    Gaussian, diagonal, variances from the Wishart moments with M_EFF independent pieces
    prior    x_j ~ N(0, (TAU*s)^2), same for every heading
    result   x_hat = (A' S^-1 A + I/(TAU s)^2)^-1 A' S^-1 d,   covariance = (A' S^-1 A + I/(TAU s)^2)^-1

Run from the folder that holds submarine.py, sea.py, sensors.py and submarine_extra.py:
    python invert_waves.py                       # reads ./sensor_data, builds/loads output/rao_cache_depth<D>.npz
    python invert_waves.py --tau 5 --plot
    python invert_waves.py --model six_dof       # transfer functions from six_dof.py (needs waves.py) instead of Capytaine
First run solves the BEM problem (a few minutes) and caches the RAOs; later runs load the cache.
Depth, grid and seed come from sensor_data/params.csv, the true sea from sensor_data/systems.csv. Results, plots and
the RAO cache go to --out (default ./output).

Conventions (taken from sensors.py / submarine.py): x forward, y to port, z up; beta is the direction the waves
TRAVEL towards (beta = 0: towards +x); phasors are Re{X exp(-i w t)}; roll/pitch/yaw angles in rad.
"""
import argparse
import csv
import os
import sys

import numpy as np
from scipy.signal import csd, decimate

# ------------------------------------------------------------------ settings (all overridable on the command line)
CHANNELS = ["roll", "pitch", "vert_acc", "P1", "P2", "P3", "P4", "P5", "P6"]
# available: roll, pitch, vert_acc, acc_x, acc_y, gyro_x, gyro_y, gyro_z, P1..P6
BAND = (0.05, 0.31)       # Hz, wave band; the BEM grid stops at 2.0 rad/s = 0.318 Hz
SETTLE = 300.0            # s discarded at the start (same as T_SETTLE in submarine_extra.py)
FS = 2.0                  # Hz, common sample rate after low-pass filtering (waves are below 0.32 Hz)
SEG = 200.0               # s per Welch segment
M_EFF = 2.5               # independent pieces averaged per Welch bin (about 2.5 lines for 200 s segments)
TAU = 5.0                 # prior width in units of s (the scale of power per Hz). Set by hand. With the full channel set the
                          # answer barely moves for TAU between 1 and 30; with few channels it matters much more.
RESULT = "inversion_result"   # output file name (.npz/.png); scripts that swap in another invert() overwrite it too


# ------------------------------------------------------------------ 1. transfer functions (frequency-domain model)
def read_params(data_dir):
    """the one-row params.csv written by sensors.create_sensor_data (depth, grid, seed, ...); {} if absent"""
    path = os.path.join(data_dir, "params.csv")
    if not os.path.exists(path):
        return {}
    row = np.genfromtxt(path, delimiter=",", names=True)
    return {k: float(row[k]) for k in row.dtype.names}


def read_systems(data_dir, params):
    """the simulated sea: one dict(Hs, Tp, gamma, s, direction) per wave system from systems.csv, as Sea(systems=...)
    takes them; older runs without systems.csv had one system stored in params.csv; None if neither"""
    keys = ("Hs", "Tp", "gamma", "s", "direction")
    path = os.path.join(data_dir, "systems.csv")
    if os.path.exists(path):
        rows = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
        return [dict(zip(keys, map(float, r))) for r in rows if len(r) == len(keys)] or None   # header only: no truth
    return [{k: params[k] for k in keys}] if all(k in params for k in keys) else None


def grid_kwargs(p):
    """BEM / Sea grid arguments from params.csv, else the Submarine defaults. n_freq = number of sea wave components,
    n_bem = BEM frequency grid (the RAO grid); data made before n_bem existed used n_bem = n_freq."""
    return dict(n_freq=int(p.get("n_freq", 40)), n_dir=int(p.get("n_dir", 36)),
                omega_range=(p.get("omega_min", 0.1), p.get("omega_max", 2.0)),
                n_bem=int(p.get("n_bem", p.get("n_freq", 40))))


def sea_grid(grid):
    """grid_kwargs without the BEM-only n_bem, for Sea()"""
    return {k: v for k, v in grid.items() if k != "n_bem"}


def load_rao(repo, cache, use_controller=True, depth=None, grid=None):
    """RAO[omega, beta, dof] per metre of wave amplitude, from the Capytaine BEM data, with the depth/heading
    controller of sensors.py included (it adds Kp - i w Kd to surge, sway, heave and yaw).
    depth/grid: as the data was made (params.csv); default submarine.DEPTH and the Submarine default grid."""
    sys.path.insert(0, repo)
    import submarine
    depth = submarine.DEPTH if depth is None else depth
    grid = grid or {}
    hull = f"L{submarine.LENGTH:g}_R{submarine.RADIUS:g}_depth{depth:g}"      # cache must match the hull
    if os.path.exists(cache):
        z = np.load(cache)
        if ("hull" in z.files and "controller" in z.files and str(z["hull"]) == hull
                and bool(z["controller"]) == use_controller
                and "visc" in z.files and bool(z["visc"])                   # caches built without B_visc are rebuilt
                and len(z["omega"]) == grid.get("n_bem", len(z["omega"])) and len(z["theta"]) == grid.get("n_dir", len(z["theta"]))):
            return {k: z[k] for k in z.files}
    from sensors import ControlledSubmarine
    sub = ControlledSubmarine(depth=depth, **grid)
    sub.solve_hydrodynamics()
    sub.build_controller()
    B_visc = np.zeros((6, 6))
    if hasattr(sub, "build_viscous_damping"):         # new generator: viscous roll/pitch damping
        sub.build_viscous_damping()
        B_visc = sub.B_visc
    om = submarine.OMEGAS if hasattr(submarine, "OMEGAS") else sub.bem_omegas
    rao = np.zeros((len(om), len(sub.theta), 6), complex)
    for i, w in enumerate(om):
        Z = -w**2 * (sub.M + sub.A[i]) - 1j * w * (sub.B[i] + B_visc) + sub.C_hydro
        if use_controller:
            Z = Z + np.diag(sub.Kp) - 1j * w * np.diag(sub.Kd)
        rao[i] = np.linalg.solve(Z, sub.F_exc[i].T).T
    out = dict(omega=om, theta=np.asarray(sub.theta), rao=rao, controller=np.array(use_controller),
               rho=np.array(submarine.RHO), g=np.array(submarine.G), depth=np.array(depth),
               hull=np.array(hull), visc=np.array(hasattr(sub, "build_viscous_damping")))
    np.savez(cache, **out)
    return out


def six_dof_rao(repo, depth=None, grid=None):
    """The analytic model from six_dof.py (Hull6) in place of the Capytaine RAOs, on the same frequency and heading grid.
    Geometry from submarine.py. Hull6 takes pitch positive bow UP; sensors.py/Capytaine use bow DOWN, so pitch is negated
    (found by comparing predicted and recorded waveforms, see forward_check.py).
    depth/grid: as the data was made (params.csv); default submarine.DEPTH and the submarine.py grid."""
    sys.path.insert(0, repo)
    import submarine
    import six_dof
    depth = submarine.DEPTH if depth is None else depth
    grid = grid or {}
    if hasattr(submarine, "OMEGAS"):                   # old generator: fixed grids in submarine.py
        om, th = submarine.OMEGAS, np.asarray(submarine.BETAS)
    else:                                              # new generator: the grids Submarine/Sea build from these
        om = np.linspace(*grid.get("omega_range", submarine.OMEGA_RANGE), grid.get("n_bem", submarine.N_FREQ))
        th = np.linspace(-np.pi, np.pi, grid.get("n_dir", submarine.N_DIR), endpoint=False)
    if hasattr(six_dof, "generator_hull"):             # updated six_dof.py: capsule, controller, pitch bow down
        from sensors import ControlledSubmarine
        b = ControlledSubmarine(depth=depth)               # mesh only: CG and sensor origin of the generator's hull
        hull = six_dof.generator_hull(depth=depth, length=submarine.LENGTH, radius=submarine.RADIUS, bg=float(b.centre[2] - b.body.center_of_mass[2]), ref_offset=float(b.centre[2] - b.body.rotation_center[2]))
        flip = 1
    else:
        hull, flip = six_dof.Hull6(length=submarine.LENGTH, radius=submarine.RADIUS, depth=depth), -1
    rao = np.zeros((len(om), len(th), 6), complex)
    for j, b in enumerate(th):
        rao[:, j, :] = hull.transfer(om, b)            # mu = beta: both are the direction the waves travel from +x towards +y
    rao[:, :, 4] *= flip
    return dict(omega=om, theta=th, rao=rao, rho=np.array(submarine.RHO), g=np.array(submarine.G), depth=np.array(depth))


def interp_complex(w, wgrid, f):
    """linear interpolation along axis 0 of a complex array"""
    flat = f.reshape(len(wgrid), -1)
    cols = [np.interp(w, wgrid, flat[:, c].real) + 1j * np.interp(w, wgrid, flat[:, c].imag) for c in range(flat.shape[1])]
    return np.stack(cols, axis=-1).reshape(f.shape[1:])


# ------------------------------------------------------------------ 2. channel operators: what each sensor reads per unit wave
def sensor_positions(info_file):
    pos = {}
    with open(info_file) as fh:
        for row in csv.DictReader(fh):
            if row["r_y_m"] != "":
                pos[row["name"]] = np.array([float(row["r_x_m"]), float(row["r_y_m"]), float(row["r_z_m"])])
    hs = None
    with open(info_file) as fh:
        for row in csv.reader(fh):
            if row and row[0] == "Hs_m":
                hs = float(row[1])
    return pos, hs


def channel_operator(name, H, w, beta, pos, rao_consts):
    """H: (J, 6) RAO at this frequency for every heading beta (J,). Returns (J,) complex: channel response per metre of wave.
    Operators are derived from true_signals() / pressure_sensor() in sensors.py (small-angle, linear)."""
    rho, g, depth = rao_consts
    k = w**2 / g
    if name == "roll":
        return H[:, 3]
    if name == "pitch":
        return H[:, 4]
    if name.startswith("gyro_"):
        return -1j * w * H[:, 3 + "xyz".index(name[-1])]
    if name in ("vert_acc", "acc_x", "acc_y"):
        rx, ry, rz = pos["IMU"]
        if name == "vert_acc":      # f_z - g
            return -w**2 * (H[:, 2] + ry * H[:, 3] - rx * H[:, 4])
        if name == "acc_x":         # f_x = x'' + lever - g*pitch
            return -w**2 * (H[:, 0] + rz * H[:, 4] - ry * H[:, 5]) - g * H[:, 4]
        return -w**2 * (H[:, 1] + rx * H[:, 5] - rz * H[:, 3]) + g * H[:, 3]
    if name in pos:                 # pressure port: incident wave + sensor moving in the static gradient
        rx, ry, rz = pos[name]
        z_mean = -depth - 0.2 + rz  # O_EARTH z + r_z
        incident = np.exp(k * z_mean) * np.exp(1j * k * (rx * np.cos(beta) + ry * np.sin(beta)))
        return rho * g * (incident - (H[:, 2] + ry * H[:, 3] - rx * H[:, 4]))
    raise ValueError(f"unknown channel {name}")


# ------------------------------------------------------------------ 3. read the CSVs and build one common time grid
SOURCES = {  # channel -> (file, column)
    "roll": ("attitude", "roll_rad"), "pitch": ("attitude", "pitch_rad"), "vert_acc": ("accel", "vert_acc_m_s2"),
    "acc_x": ("accel", "f_x_m_s2"), "acc_y": ("accel", "f_y_m_s2"),
    "gyro_x": ("gyro", "gyro_x_rad_s"), "gyro_y": ("gyro", "gyro_y_rad_s"), "gyro_z": ("gyro", "gyro_z_rad_s"),
    **{f"P{i}": ("pressure", f"P{i}_Pa") for i in range(1, 7)},
}


def read_channels(data_dir, channels):
    cache, series = {}, {}
    for ch in channels:
        fname, col = SOURCES[ch]
        if fname not in cache:
            cache[fname] = np.genfromtxt(os.path.join(data_dir, fname + ".csv"), delimiter=",", names=True)
        d = cache[fname]
        t, y = d["t_s"], d[col]
        fs = 1.0 / np.median(np.diff(t))
        y = y - np.mean(y)
        q = int(round(fs / FS))
        if q > 1:                                   # low-pass and decimate (zero-phase FIR)
            y = decimate(y, q, ftype="fir", zero_phase=True)
            t = t[::q][: len(y)]
        series[ch] = (t, y)
    t0 = max(SETTLE, max(s[0][0] for s in series.values()))
    t1 = min(s[0][-1] for s in series.values())
    tc = np.arange(t0, t1, 1 / FS)
    X = np.column_stack([np.interp(tc, *series[ch]) for ch in channels])
    return tc, X


def welch_matrix(X):
    n = int(SEG * FS)
    C = X.shape[1]
    f, _ = csd(X[:, 0], X[:, 0], fs=FS, nperseg=n, noverlap=n // 2)
    S = np.empty((len(f), C, C), complex)
    for i in range(C):
        for k in range(C):
            S[:, i, k] = csd(X[:, i], X[:, k], fs=FS, nperseg=n, noverlap=n // 2)[1]   # = Z_i conj(Z_k), the S_ik convention
    return f, S


# ------------------------------------------------------------------ 4. the inversion
def stack_pairs(C):
    return [(i, k) for i in range(C) for k in range(i, C)]


def data_vector(S, pairs):
    out = []
    for i, k in pairs:
        out += [S[i, i].real] if i == k else [S[i, k].real, S[i, k].imag]
    return np.array(out)


def operator_matrix(h, pairs):
    """h: (J, C) channel responses per heading -> A (N, J) real"""
    rows = []
    for i, k in pairs:
        g = h[:, i] * np.conj(h[:, k])
        rows += [g.real] if i == k else [g.real, g.imag]
    return np.array(rows)


def noise_variances(S, pairs, M):
    v = []
    for i, k in pairs:
        a, b = S[i, i].real, S[k, k].real
        v += [a * a / M] if i == k else [a * b / (2 * M)] * 2
    return np.array(v)


def invert(f, S, h_all, tau, M):
    """f: (L,), S: (L,C,C), h_all: (L,J,C). Returns x_hat (L,J), covariance (L,J,J), prior scale s (L,)"""
    L, J, C = h_all.shape
    pairs = stack_pairs(C)
    X, COV, SC = np.zeros((L, J)), np.zeros((L, J, J)), np.zeros(L)
    for l in range(L):
        A = operator_matrix(h_all[l], pairs)
        d = data_vector(S[l], pairs)
        w = 1 / np.sqrt(noise_variances(S[l], pairs, M))
        At, dt = A * w[:, None], d * w
        s = np.mean([S[l, c, c].real / np.mean(np.abs(h_all[l, :, c]) ** 2) for c in range(C)])   # scale: wave power per Hz
        Lam = At.T @ At + np.eye(J) / (tau * s) ** 2
        cov = np.linalg.inv(Lam)
        X[l], COV[l], SC[l] = cov @ At.T @ dt, cov, s
    return X, COV, SC


# ------------------------------------------------------------------ 5. main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="sensor_data")
    ap.add_argument("--repo", default=os.path.dirname(os.path.abspath(__file__)) if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "submarine.py")) else os.getcwd())
    ap.add_argument("--cache", default=None, help="default <out>/rao_cache_depth<D>.npz")
    ap.add_argument("--out", default="output", help="folder for the results, plots and RAO cache")
    ap.add_argument("--tau", type=float, default=TAU)
    ap.add_argument("--meff", type=float, default=M_EFF)
    ap.add_argument("--channels", default=",".join(CHANNELS))
    ap.add_argument("--model", choices=["capytaine", "six_dof"], default="capytaine",
                    help="where the transfer functions come from: the Capytaine BEM data (matches the generator) or six_dof.py (independent)")
    ap.add_argument("--no-controller", action="store_true", help="capytaine model only: leave the controller out")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    channels = [c.strip() for c in args.channels.split(",")]

    params = read_params(args.data)
    depth = params.get("depth", 15.0)
    grid = grid_kwargs(params) if params else {}          # no params.csv: Submarine defaults, as before
    os.makedirs(args.out, exist_ok=True)
    cache = args.cache or os.path.join(args.out, f"rao_cache_depth{depth:g}.npz")
    rao = six_dof_rao(args.repo, depth, grid) if args.model == "six_dof" else load_rao(args.repo, cache, use_controller=not args.no_controller, depth=depth, grid=grid)
    wg, beta = rao["omega"], rao["theta"]
    consts = (float(rao["rho"]), float(rao["g"]), float(rao["depth"]))
    pos, hs_printed = sensor_positions(os.path.join(args.data, "sensor_info.csv"))

    tc, X = read_channels(args.data, channels)
    f, S = welch_matrix(X)
    sel = (f >= BAND[0]) & (f <= BAND[1])
    fb, Sb, df = f[sel], S[sel], f[1] - f[0]

    h_all = np.zeros((len(fb), len(beta), len(channels)), complex)
    for l, fl in enumerate(fb):
        w = 2 * np.pi * fl
        H = interp_complex(w, wg, rao["rao"])            # (J, 6)
        for c, ch in enumerate(channels):
            h_all[l, :, c] = channel_operator(ch, H, w, beta, pos, consts)

    x, cov, s = invert(fb, Sb, h_all, args.tau, args.meff)

    S_f = x.sum(1)                                        # wave power per Hz
    m0 = S_f.sum() * df
    hs = 4 * np.sqrt(max(m0, 0.0))
    var_m0 = df**2 * sum(np.ones(len(beta)) @ cov[l] @ np.ones(len(beta)) for l in range(len(fb)))
    sd_hs = 2 * np.sqrt(var_m0) / np.sqrt(max(m0, 1e-12))
    share = 100 * x.sum(0) / x.sum()
    neg = -100 * np.minimum(x, 0).sum() / np.abs(x).sum()
    Tp = 1 / fb[np.argmax(S_f)]

    print(f"transfer functions: {args.model}")
    print(f"channels: {channels}")
    print(f"record used: {tc[0]:.0f}-{tc[-1]:.0f} s, {len(fb)} Welch bins in {BAND[0]}-{BAND[1]} Hz, tau={args.tau}, M_eff={args.meff}")
    sd_txt = f"(posterior sd {sd_hs:.2f} m; this is optimistic, do not quote it)" if np.any(cov) else "(no posterior sd for this method)"
    print(f"\nHs estimate = {hs:.2f} m   {sd_txt}   Tp = {Tp:.1f} s")
    if hs_printed is not None:
        print(f"Hs printed by sensors.py = {hs_printed:.2f} m")
    try:
        sys.path.insert(0, args.repo)
        from sea import Sea
        seed = int(params.get("seed", 42))
        systems = read_systems(args.data, params)
        if systems is None:
            raise ValueError("no systems.csv and no sea state in params.csv")
        sea = Sea(**sea_grid(grid), seed=seed, systems=systems,                # same sea as the run that made the data
                 uniform_frequencies=bool(params.get("uniform_frequencies", 0)))
        _, D, _, a, _ = sea.sea_spectrum()
        z = (a * np.exp(1j * sea.phi)).sum(1)
        print(f"Hs realised at the origin by the seed-{seed} waves = {4 * np.sqrt(0.5 * np.sum(abs(z) ** 2)):.2f} m")
        true_share = 100 * D / D.sum()
    except Exception as e:                                # sea.py missing or no sea recorded: skip the comparison
        print(f"(truth comparison skipped: {e!r})")
        sea, true_share = None, None
    print(f"negative power: {neg:.1f}% of total magnitude")
    print("\nheading (deg)   estimated share (%)" + ("   true share (%)" if true_share is not None else ""))
    for j in np.argsort(np.angle(np.exp(1j * beta))):
        deg = np.degrees(np.angle(np.exp(1j * beta[j])))
        print(f"{deg:9.0f}       {share[j]:8.1f}" + (f"          {true_share[j]:8.1f}" if true_share is not None else ""))

    print("\nmodel check: band-integrated power, data vs fitted A x_hat (ratio near 1 means the operator and transfer functions fit)")
    pairs = stack_pairs(len(channels))
    fit_diag = np.zeros((len(fb), len(channels)))
    for l in range(len(fb)):
        A = operator_matrix(h_all[l], pairs)
        pred = A @ x[l]
        r = 0
        for i, k in pairs:
            if i == k:
                fit_diag[l, i] = pred[r]
            r += 1 if i == k else 2
    for c, ch in enumerate(channels):
        obs = Sb[:, c, c].real.sum() * df
        print(f"  {ch:9s} data {obs:10.3e}   fit {fit_diag[:, c].sum() * df:10.3e}   ratio {fit_diag[:, c].sum() * df / obs:5.2f}")

    np.savez(os.path.join(args.out, RESULT + ".npz"), f=fb, x=x, cov=cov, beta=beta, Hs=hs, sd_Hs=sd_hs, channels=channels)
    print(f"\nsaved {os.path.join(args.out, RESULT + '.npz')}  (x[f, heading] = wave power per Hz)")

    if args.plot:
        import matplotlib.pyplot as plt
        from polar_plot import polar_spectrum
        fig = plt.figure(figsize=(15, 4.6))
        ax = [fig.add_subplot(1, 3, 1), fig.add_subplot(1, 3, 2), fig.add_subplot(1, 3, 3, projection="polar")]
        ax[0].plot(fb, S_f, label="estimate")
        if sea is not None:
            S_true, *_ = sea.sea_spectrum()
            ax[0].plot(sea.f, S_true, "k--", label="true S(f)")   # S_f lives on the Sea's own frequencies, not the BEM grid
        ax[0].set(xlabel="f (Hz)", ylabel="S(f)  (m^2/Hz)", title=f"Frequency spectrum, Hs = {hs:.2f} m")
        ax[0].legend()
        order = np.argsort(np.angle(np.exp(1j * beta)))
        deg = np.degrees(np.angle(np.exp(1j * beta[order])))
        ax[1].bar(deg, share[order], width=30, alpha=0.6, label="estimate")
        if true_share is not None:
            ax[1].plot(deg, true_share[order], "ko", label="true")
        ax[1].set(xlabel="heading (deg)", ylabel="share of power (%)", title="Direction distribution")
        ax[1].legend()
        im = polar_spectrum(ax[2], fb, beta, x, cmap="RdBu_r", vmin=-np.abs(x).max(), vmax=np.abs(x).max())
        fig.colorbar(im, ax=ax[2], label="power per Hz", pad=0.1)
        ax[2].set_title("Estimated directional spectrum\n(angle = heading, radius = f in Hz)", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, RESULT + ".png"), dpi=130)
        print(f"saved {os.path.join(args.out, RESULT + '.png')}")


if __name__ == "__main__":
    main()
