import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from scipy.signal import welch

import submarine
from submarine import Submarine, RHO, G, LENGTH, RADIUS, DEPTH, OMEGAS


T_SIM = 1200.0      # simulation length (s)
T_SETTLE = 300.0    # start-up discarded from the plots / spectra
F_PEAK = 0.17       # peak frequency, same as f_m in sea.py's jonswap

# Sensors
FS_SENSOR = 2.0     # samples per second
NOISE_STD = np.array([0.02, 0.02, 0.02, np.deg2rad(0.05), np.deg2rad(0.05), np.deg2rad(0.05)])

DOFS = ["Surge", "Sway", "Heave", "Roll", "Pitch", "Yaw"]
UNITS = ["m", "m", "m", "rad", "rad", "rad"]
HEAVE, ROLL, PITCH = 2, 3, 4
rng = np.random.default_rng(0)



# ================================================================
#  EXTRA CALCULATIONS (things submarine_new.py had that submarine.py doesn't)
# ================================================================
def kinematics(sub, a, x, y, z, t):
    """Linear deep water waves at points x, y, z (same shape, z <= 0): dynamic pressure p, velocities u, v, w
       eta = a cos(psi), u = a w e^(kz) cos(psi) (cos, sin)(theta), w = a w e^(kz) sin(psi), p = rho g a e^(kz) cos(psi)"""
    x, y, z = (np.asarray(q, dtype=float)[..., None] for q in (x, y, z))
    p, u, v, w = (np.zeros(x.shape[:-1]) for _ in range(4))
    for n in range(len(sub.f)): # one frequency at a time to save memory
        wn, kn = 2 * np.pi * sub.f[n], sub.k[n]
        psi = kn * (x * sub.cos_theta + y * sub.sin_theta) - wn * t + sub.phi[n]
        amp = a[n] * np.exp(kn * z)
        p += RHO * G * np.sum(amp * np.cos(psi), axis=-1)
        u += wn * np.sum(amp * np.cos(psi) * sub.cos_theta, axis=-1)
        v += wn * np.sum(amp * np.cos(psi) * sub.sin_theta, axis=-1)
        w += wn * np.sum(amp * np.sin(psi), axis=-1)
    return p, u, v, w


def interp_omega(w, wp, fp, outside=None):
    """np.interp along axis 0 for complex arrays of any shape, outside=None holds the end values"""
    flat = fp.reshape(len(wp), -1)
    cols = [np.interp(w, wp, flat[:, c].real, left=outside, right=outside)
            + 1j * np.interp(w, wp, flat[:, c].imag, left=outside, right=outside) for c in range(flat.shape[1])]
    return np.stack(cols, axis=-1).reshape(len(w), *fp.shape[1:])


def compute_rao(sub, omega, A, B, F):
    """[-w^2 (M + A) - i w B + C] X = F_exc  ->  RAO[omega, theta, dof] (motion per m of wave)"""
    RAO = np.zeros_like(F, dtype=complex)
    for i, w in enumerate(omega):
        Z = -w**2 * (sub.M + A[i]) - 1j * w * B[i] + sub.C_hydro
        RAO[i] = np.linalg.solve(Z, F[i].T).T
    return RAO


def rotation_matrix(roll, pitch, yaw):
    """Body -> earth rotation (yaw, then pitch, then roll)."""
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def hull_surface(state, exag):
    """Hull cylinder surface moved by the 6-DOF state: rotate (roll, pitch, yaw) about the centre, then shift."""
    TH_, XH = np.meshgrid(np.linspace(0, 2 * np.pi, 30), np.linspace(-LENGTH / 2, LENGTH / 2, 8))
    pts = np.stack([XH, RADIUS * np.cos(TH_), RADIUS * np.sin(TH_)]).reshape(3, -1)
    pts = rotation_matrix(*(exag * state[3:6])) @ pts + (exag * state[:3] + [0, 0, -DEPTH])[:, None]
    return pts.reshape(3, *XH.shape)


def colour_line3d(ax, x, y, z, c, cmap, lw=2.0):
    """3D line coloured point-by-point by c. Returns the collection (for a colourbar)."""
    pts = np.stack([x, y, z], axis=-1)
    lc = Line3DCollection(np.stack([pts[:-1], pts[1:]], axis=1), cmap=cmap, linewidths=lw)
    lc.set_array(np.asarray(c)[:-1])
    ax.add_collection(lc)
    ax.auto_scale_xyz(x, y, z)
    return lc


def clean_3d(ax):
    """Transparent panes and light grid: less clutter on 3D axes."""
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor("lightgrey")
        axis._axinfo["grid"]["color"] = (0.85, 0.85, 0.85, 1)




if __name__ == "__main__":
    dt = submarine.dt

    # ---- simulation (submarine.py)
    sub = Submarine()
    sub.solve_hydrodynamics()
    sub.build_excitation()
    sub.build_kernel()
    T, X_hist = sub.simulate(duration=T_SIM)
    t, X, V = T, X_hist[:, :6], X_hist[:, 6:]

    # ---- sea spectrum
    S_f, D, S_sea, a, Hs = sub.sea_spectrum()
    print(f"Sea: Hs = {Hs:.2f} m, Tp = {1 / F_PEAK:.1f} s")

    # ---- RAOs: on the BEM grid (plots) and at the Sea frequencies (response spectra)
    RAO = compute_rao(sub, OMEGAS, sub.A, sub.B, sub.F_exc)
    A_s, B_s = interp_omega(sub.w, OMEGAS, sub.A).real, interp_omega(sub.w, OMEGAS, sub.B).real
    F_s = interp_omega(sub.w, OMEGAS, sub.F_exc, outside=0) # no force above OMEGAS[-1], same as build_excitation
    H_sea = compute_rao(sub, sub.w, A_s, B_s, F_s)
    S_resp = np.sum(np.abs(H_sea)**2 * S_sea[..., None] * sub.delta_theta, axis=1) # (N_f, 6)

    # ---- sensors
    step = max(1, int(round(1 / (FS_SENSOR * dt))))
    fs_actual = 1 / (step * dt)
    t_sens = t[::step]
    X_sens = X[::step] + NOISE_STD * rng.standard_normal((len(t_sens), 6))
    settled = t_sens > T_SETTLE
    f_w, S_meas = welch(X_sens[settled], fs=fs_actual, nperseg=256, axis=0)



    # ================================================================
    #  PLOTS
    # ================================================================
    i_beta = np.argmin(abs(np.angle(np.exp(1j * sub.theta)))) # BEM direction nearest the mean (0 in sea.py)
    beta_lbl = f"beta = {np.rad2deg(sub.theta[i_beta]):.0f} deg"
    t_snap = t[len(t) // 2]
    deg = 180 / np.pi
    win = (t > T_SETTLE) & (t < T_SETTLE + 300)
    win_s = (t_sens > T_SETTLE) & (t_sens < T_SETTLE + 300)

    # 1. Sea spectrum
    th_c = np.append(sub.theta, sub.theta[0] + 2 * np.pi)                          # close the circle
    S_ft = S_f[:, None] * D[None, :]
    S_ft = np.hstack([S_ft, S_ft[:, :1]])
    TH, FF = np.meshgrid(th_c, sub.f)

    fig = plt.figure(figsize=(18, 5))
    ax0 = fig.add_subplot(1, 3, 1)
    ax0.plot(sub.f, S_f)
    ax0.set(xlabel="f (Hz)", ylabel="S(f)  (m^2/Hz)", title=f"JONSWAP spectrum, Hs = {Hs:.2f} m")

    ax1 = fig.add_subplot(1, 3, 2, projection="polar")
    cs = ax1.contourf(TH, FF, S_ft, levels=30, cmap="viridis")
    fig.colorbar(cs, ax=ax1, pad=0.1, label="S(f, theta)  (m^2/Hz/rad)")
    ax1.set_title("Directional spectrum (radius = f in Hz)", pad=20)

    ax2 = fig.add_subplot(1, 3, 3, projection="3d")
    ax2.plot_surface(FF * np.cos(TH), FF * np.sin(TH), S_ft, cmap="viridis", linewidth=0, antialiased=True)
    ax2.set(xlabel="f cos(theta) (Hz)", ylabel="f sin(theta) (Hz)", zlabel="S(f, theta)",
            title="Directional spectrum (3D)")
    fig.tight_layout()

    # 6. Motion time series + sensor samples
    fig, ax = plt.subplots(6, 1, figsize=(12, 11), sharex=True)
    for d in range(6):
        scale = 1 if UNITS[d] == "m" else deg
        ax[d].plot(t[win], X[win, d] * scale, lw=1, label="true motion")
        ax[d].plot(t_sens[win_s], X_sens[win_s, d] * scale, ".", ms=3, label=f"sensor ({fs_actual:.1f} Hz + noise)")
        ax[d].set_ylabel(f"{DOFS[d]} ({'m' if scale == 1 else 'deg'})")
    ax[0].legend(loc="upper right")
    ax[-1].set_xlabel("t (s)")
    fig.suptitle("6-DOF motion from the Cummins time-domain simulation (5-minute window)")
    fig.tight_layout()

    # 7. Validation: sensor spectrum (Welch) vs frequency-domain prediction |H|^2 S
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for a_, d in zip(ax, (HEAVE, PITCH, ROLL)):
        noise_floor = 2 * NOISE_STD[d]**2 / fs_actual                 # white sensor noise, one-sided PSD
        a_.plot(sub.f, S_resp[:, d] + noise_floor, label="frequency domain |H|^2 S + noise floor")
        a_.plot(f_w, S_meas[:, d], label="time domain sensors (Welch)")
        a_.set(xlabel="f (Hz)", ylabel=f"{UNITS[d]}^2/Hz", title=f"{DOFS[d]} response spectrum", xlim=(0, 0.3))
        a_.legend()
    fig.tight_layout()

    # 8. 3D motion of the submarine (path of the hull centre, true scale)
    fig = plt.figure(figsize=(10, 7))
    ax3 = fig.add_subplot(projection="3d")
    p = ax3.scatter(X[win, 0], X[win, 1], X[win, 2] - DEPTH, c=t[win], cmap="plasma", s=2)
    fig.colorbar(p, ax=ax3, label="t (s)", shrink=0.6)
    ax3.set(xlabel="surge x (m)", ylabel="sway y (m)", zlabel="z (m)",
            title="Path of the hull centre over 5 minutes (true scale)")

    tw, Xw, Vw = t[win], X[win], V[win]

    # 10. Long-exposure view: centre path coloured by speed, bow/stern tracks, ghost hulls (motion exaggerated)
    EXAG_G = 6
    fig = plt.figure(figsize=(11, 7))
    ax3 = fig.add_subplot(projection="3d")
    speed = np.linalg.norm(Vw[:, :3], axis=1)
    P = EXAG_G * Xw[:, :3] + [0, 0, -DEPTH]
    lc = colour_line3d(ax3, *P.T, speed, "turbo", lw=2.5)
    for end, col in ((LENGTH / 2, "crimson"), (-LENGTH / 2, "teal")):
        track = np.array([rotation_matrix(*(EXAG_G * s[3:])) @ [end, 0, 0] for s in Xw]) + P
        ax3.plot(*track.T, color=col, lw=0.8, label="bow track" if end > 0 else "stern track")
    ghosts = np.arange(0, len(tw), int(60 / dt))
    cm = plt.get_cmap("plasma")
    for j, k in enumerate(ghosts):
        ax3.plot_wireframe(*hull_surface(Xw[k], EXAG_G), color=cm(j / len(ghosts)), alpha=0.15, lw=0.5)
    fig.colorbar(lc, ax=ax3, label="hull-centre speed (m/s)", shrink=0.6)
    ax3.set(xlabel="x (m)", ylabel="y (m)", zlabel="z (m)",
            title=f"Long-exposure view: 5 min of motion, ghost hull every 60 s (all motion x{EXAG_G})")
    ax3.set_box_aspect((1, 0.5, 0.35))
    ax3.legend(loc="upper left")
    clean_3d(ax3)

    # 12. Phase-space trajectories
    fig = plt.figure(figsize=(14, 6))
    for i, (d_a, d_b, title) in enumerate([(ROLL, 1, "Roll phase space (with sway)"),
                                           (HEAVE, PITCH, "Heave phase space (with pitch)")]):
        ax3 = fig.add_subplot(1, 2, i + 1, projection="3d")
        s_a = deg if UNITS[d_a] == "rad" else 1
        s_b = deg if UNITS[d_b] == "rad" else 1
        lc = colour_line3d(ax3, Xw[:, d_a] * s_a, Vw[:, d_a] * s_a, Xw[:, d_b] * s_b, tw, "viridis", lw=1.0)
        u_a = "deg" if s_a != 1 else "m"
        u_b = "deg" if s_b != 1 else "m"
        ax3.set(xlabel=f"{DOFS[d_a]} ({u_a})", ylabel=f"{DOFS[d_a]} rate ({u_a}/s)",
                zlabel=f"{DOFS[d_b]} ({u_b})", title=title)
        clean_3d(ax3)
    fig.colorbar(lc, ax=fig.axes, label="t (s)", shrink=0.6)

    # 13. Underwater velocity field: stacked horizontal planes (3D) and a vertical x-z slice (2D)
    fig = plt.figure(figsize=(15, 6.5))
    ax3 = fig.add_subplot(1, 2, 1, projection="3d")
    xq = np.linspace(-120, 120, 9)
    depths = np.array([-2.0, -15.0, -40.0])
    XQ, YQ, ZQ = np.meshgrid(xq, xq, depths, indexing="ij")
    _, uq, vq, wq = kinematics(sub, a, XQ, YQ, ZQ, t_snap)
    spd = np.sqrt(uq**2 + vq**2 + wq**2).ravel()
    cols = plt.get_cmap("magma_r")(spd / spd.max())
    cols = np.concatenate([cols, np.repeat(cols, 2, axis=0)])              # 3D quiver: shaft + 2 head lines
    ax3.quiver(XQ.ravel(), YQ.ravel(), ZQ.ravel(), uq.ravel(), vq.ravel(), wq.ravel(),
               length=10, colors=cols, lw=1.2, arrow_length_ratio=0.3)
    ax3.plot_surface(*hull_surface(np.zeros(6), 1), color="dimgray", alpha=0.6)
    xs_ = np.linspace(-120, 120, 40)
    XS, YS = np.meshgrid(xs_, xs_)
    ax3.plot_surface(XS, YS, sub.superposition(XS, YS, t_snap)[0], cmap="Blues_r", alpha=0.25, linewidth=0)
    ax3.set(xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", zlim=(-45, 6),
            title=f"Wave orbital velocity at 3 depths, t = {t_snap:.0f} s\n(arrow length = 10 s of travel)")
    ax3.set_box_aspect((1, 1, 0.5))
    clean_3d(ax3)
    sm = plt.cm.ScalarMappable(cmap="magma_r", norm=plt.Normalize(0, spd.max()))
    fig.colorbar(sm, ax=ax3, label="|u| (m/s)", shrink=0.6, pad=0.12)

    ax2 = fig.add_subplot(1, 2, 2)
    xf, zf = np.linspace(-150, 150, 150), np.linspace(-50, 0, 50)
    XF, ZF = np.meshgrid(xf, zf)
    pf, uf, _, wf = kinematics(sub, a, XF, 0 * XF, ZF, t_snap)
    cs = ax2.contourf(XF, ZF, pf / 1e3, levels=30, cmap="RdBu_r")
    fig.colorbar(cs, ax=ax2, label="dynamic pressure (kPa)")
    ax2.streamplot(xf, zf, uf, wf, color="k", density=1.3, linewidth=0.6, arrowsize=0.8)
    ax2.plot(xf, sub.superposition(xf, 0 * xf, t_snap)[0], color="navy", lw=2)
    ax2.add_patch(plt.Rectangle((-LENGTH / 2, -DEPTH - RADIUS), LENGTH, 2 * RADIUS, color="dimgray", zorder=3))
    ax2.set(xlabel="x (m)", ylabel="z (m)", title="Vertical slice y = 0: pressure + flow streamlines (incident wave)",
            ylim=(-50, 8))

    # 14. Particle orbits under ONE regular wave (Tp, Hs): circles shrinking as e^{kz}
    EXAG_O = 3
    a_reg, w_reg = Hs / 2, 2 * np.pi * F_PEAK
    k_reg = w_reg**2 / G
    lam = 2 * np.pi / k_reg
    fig = plt.figure(figsize=(12, 6.5))
    ax3 = fig.add_subplot(projection="3d")
    xr_, yr_ = np.linspace(0, 2 * lam, 120), np.linspace(-40, 40, 10)
    XR, YR = np.meshgrid(xr_, yr_)
    ax3.plot_surface(XR, YR, a_reg * np.cos(k_reg * XR), cmap="Blues_r", alpha=0.4, linewidth=0)
    ph = np.linspace(0, 2 * np.pi, 60)
    z_levels = np.array([-3.0, -12.0, -25.0, -40.0, -60.0])
    zc = plt.get_cmap("viridis")
    for iz, z0 in enumerate(z_levels):
        r = EXAG_O * a_reg * np.exp(k_reg * z0)
        for x0 in np.linspace(0, 2 * lam, 17):
            ax3.plot(x0 - r * np.sin(k_reg * x0 - ph), 0 * ph, z0 + r * np.cos(k_reg * x0 - ph),
                     color=zc(iz / len(z_levels)), lw=1.2)
            ax3.scatter(x0 - r * np.sin(k_reg * x0), 0, z0 + r * np.cos(k_reg * x0), color="k", s=6)
    ax3.plot([0, 2 * lam], [0, 0], [-DEPTH, -DEPTH], "k--", lw=1, label=f"sub depth ({DEPTH:.0f} m)")
    ax3.set(xlabel="x (m)", ylabel="y (m)", zlabel="z (m)",
            title=f"Particle orbits under a regular wave: T = {1 / F_PEAK:.0f} s, lambda = {lam:.0f} m "
                  f"(orbits x{EXAG_O}, dots = t = 0)")
    ax3.set_box_aspect((2.2, 0.4, 0.9))
    ax3.view_init(elev=12, azim=-75)
    ax3.set_yticks([])
    ax3.legend()
    clean_3d(ax3)

    # 15. Dynamic pressure isosurfaces around the hull
    
    from skimage.measure import marching_cubes
    xi, zi = np.linspace(-150, 150, 40), np.linspace(-50, 0, 18)
    XI, YI, ZI = np.meshgrid(xi, xi, zi, indexing="ij")
    p_vol = kinematics(sub, a, XI, YI, ZI, t_snap)[0] / 1e3
    spacing = (xi[1] - xi[0], xi[1] - xi[0], zi[1] - zi[0])
    fig = plt.figure(figsize=(11, 7))
    ax3 = fig.add_subplot(projection="3d")
    p_lvl = 0.3 * np.abs(p_vol).max()
    for lvl, col in ((p_lvl, "tab:red"), (-p_lvl, "tab:blue"), (0.6 * p_lvl, "salmon"), (-0.6 * p_lvl, "skyblue")):
        verts, faces, _, _ = marching_cubes(p_vol, lvl, spacing=spacing)
        verts += [xi[0], xi[0], zi[0]]
        ax3.add_collection3d(Poly3DCollection(verts[faces], facecolor=col, alpha=0.35, linewidth=0))
    ax3.plot_surface(*hull_surface(np.zeros(6), 1), color="dimgray")
    ax3.set(xlim=(-150, 150), ylim=(-150, 150), zlim=(-50, 0), xlabel="x (m)", ylabel="y (m)", zlabel="z (m)",
            title=f"Wave pressure isosurfaces, t = {t_snap:.0f} s: red +{p_lvl:.0f} / +{0.6 * p_lvl:.0f} kPa, "
                    f"blue -{p_lvl:.0f} / -{0.6 * p_lvl:.0f} kPa")
    ax3.set_box_aspect((1, 1, 0.4))
    clean_3d(ax3)
    


    # 16. RAO "flowers": |H(w, beta)| in polar form (radius = w, angle = wave direction)
    W_, B_ = np.meshgrid(OMEGAS, np.append(sub.theta, sub.theta[0] + 2 * np.pi), indexing="ij")
    RAO_w = np.concatenate([RAO, RAO[:, :1]], axis=1)
    fig = plt.figure(figsize=(15, 9))
    for d in range(6):
        ax3 = fig.add_subplot(2, 3, d + 1, projection="3d")
        sc = deg if UNITS[d] == "rad" else 1
        Z = np.abs(RAO_w[:, :, d]) * sc
        ax3.plot_surface(W_ * np.cos(B_), W_ * np.sin(B_), Z, cmap="viridis", linewidth=0, antialiased=True)
        ax3.set(title=f"{DOFS[d]} |RAO| ({'deg' if sc != 1 else 'm'} per m)", xlabel="w cos(beta)", ylabel="w sin(beta)")
        ax3.view_init(elev=35, azim=-60)
        clean_3d(ax3)
    fig.suptitle("RAO surfaces: radius = wave frequency w (rad/s), angle = wave direction beta")
    fig.tight_layout()

    # 9. Animation: Sea.plot with the sub in it, 2 minutes after settling (shows all the figures above too)
    sub.animate(t[win][:int(120 / dt)], X_hist[win][:int(120 / dt)], frame_every=1.0, scale=3.0)
