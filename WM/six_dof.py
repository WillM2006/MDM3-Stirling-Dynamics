"""Six degrees of freedom of the hull in the frequency domain. Linear, deep water, no forward speed.

Same hull as submarine.py (a uniform cylinder, strip theory), now with waves from any heading.
Everything is a phasor, Re{ X e^{-i w t} }, so the whole model is one 6x6 solve per frequency:

    [ C - w^2 M - i w B ] X = F        H = X / a

Axes: x along the hull (bow +x), y to port, z up, origin at the centre of buoyancy (the middle of
the hull axis). Order of the six: surge, sway, heave, roll, pitch, yaw.
Pitch is positive bow up, as in submarine.py. Roll and yaw are right-handed about x and z.
Wave heading mu: angle from +x towards +y of the direction the waves travel (0 = towards the bow).
H is metres per metre for the first three and radians per metre for the last three.

Options added for the new generator (Nov 2026). With every option at its default the model is exactly the
original one. generator_hull() switches them on to describe the generator's vehicle, using only what you would
know about your own submarine (shape, size, mass, CG, reference point, controller); the hydrodynamics stay strip
theory, independent of the Capytaine solve that makes the data:
    shape="capsule"      hemispherical ends: section radius r(x), all strip integrals done numerically
    controller=True      the depth/heading controller of sensors.py on surge, sway, heave, yaw
                         (replaces the heave-only hold_period spring)
    zeta_pitch           pitch damping as a fraction of critical (0 = none, as before)
    pitch_down=True      return pitch positive bow DOWN, as sensors.py / Capytaine (no sign flip needed outside)
    ref_offset           report the motion of a point this far below the centre of buoyancy (the generator's
                         sensor origin O is 0.2 m below the hull centre)
"""
import numpy as np

from waves import G, RHO

DOF = ["surge", "sway", "heave", "roll", "pitch", "yaw"]


class Hull6:
    def __init__(self, length=40.0, radius=2.0, depth=20.0, mass_ratio=1.0,
                 ca=1.0, ca_axial=0.02, bg=0.5, hold_period=60.0,
                 roll_gyration=None, zeta_roll=0.05, couple=True,
                 shape="cylinder", controller=False, control_period=60.0, control_zeta=0.7,
                 zeta_pitch=0.0, pitch_down=False, ref_offset=0.0, n_strips=701):
        self.L, self.R, self.depth, self.bg = length, radius, depth, bg
        self.ca, self.ca_axial = ca, ca_axial      # cross-flow and axial added-mass coefficients
        self.couple = couple                       # False drops the BG coupling terms
        self.shape, self.pitch_down, self.ref_offset = shape, pitch_down, ref_offset
        self.A = np.pi * radius**2
        if shape == "capsule":
            # strips along the hull: straight middle of half-length L/2 - R, hemispherical caps of radius R
            self.xs = np.linspace(-length / 2, length / 2, n_strips)
            half = length / 2 - radius
            ex = np.clip(np.abs(self.xs) - half, 0, None)
            self.rs = np.sqrt(np.clip(radius**2 - ex**2, 0, None))      # section radius r(x)
            self.a = np.pi * self.rs**2                                  # section area a(x)
            self.V = np.trapezoid(self.a, self.xs)
        elif shape == "cylinder":
            self.V = self.A * length
        else:
            raise ValueError("shape must be 'cylinder' or 'capsule'")
        self.m = mass_ratio * RHO * self.V

        m, b, L, R = self.m, bg, length, radius
        k_gyr = R / np.sqrt(2) if roll_gyration is None else roll_gyration    # solid cylinder: I = m R^2 / 2

        # added mass about the centre of buoyancy. A circle gives no roll added inertia.
        A11 = ca_axial * RHO * self.V
        A22 = A33 = ca * RHO * self.V
        A44 = 0.0
        if shape == "capsule":
            A55 = A66 = ca * RHO * np.trapezoid(self.a * self.xs**2, self.xs)   # strip theory: sum of a(x) x^2
        else:
            A55 = A66 = ca * RHO * self.V * L**2 / 12

        # rigid body about the centre of buoyancy, with the centre of gravity a distance b below it.
        # The parallel-axis terms m b^2 are in the roll and pitch inertias.
        if shape == "capsule":
            # solid of revolution of uniform density: discs of radius r(x)
            dens = m / self.V
            I_roll = (dens * np.trapezoid(np.pi * self.rs**4 / 2, self.xs) if roll_gyration is None
                      else m * roll_gyration**2) + m * b**2
            I_yaw = dens * np.trapezoid(self.a * (self.xs**2 + self.rs**2 / 4), self.xs)
            I_pitch = I_yaw + m * b**2
        else:
            I_roll = m * k_gyr**2 + m * b**2
            I_pitch = m * L**2 / 12 + m * b**2
            I_yaw = m * L**2 / 12
        M = np.diag([m + A11, m + A22, m + A33, I_roll + A44, I_pitch + A55, I_yaw + A66])
        if couple:
            M[0, 4] = M[4, 0] = m * b      # surge with pitch
            M[1, 3] = M[3, 1] = m * b      # sway with roll
        self.M = M

        # restoring: only heave (a stand-in depth controller), roll and pitch (CG below CB).
        # Surge, sway and yaw are free, as for a hull with no station keeping.
        c_heave = (2 * np.pi / hold_period) ** 2 * (m + A33)
        k_tilt = RHO * G * self.V * b               # same as submarine.py
        self.C = np.diag([0.0, 0.0, c_heave, k_tilt, k_tilt, 0.0])

        # damping: roll only, as a fraction of critical. An assumption, see the md file.
        B = np.zeros((6, 6))
        B[3, 3] = 2 * zeta_roll * np.sqrt(k_tilt * M[3, 3])
        B[4, 4] = 2 * zeta_pitch * np.sqrt(k_tilt * M[4, 4])      # zero unless zeta_pitch is given

        # controller of sensors.py: K_p = m_eff w_c^2, K_d = 2 zeta m_eff w_c on surge, sway, heave and yaw,
        # with m_eff from this model's own mass + added mass. Replaces the heave-only hold_period spring.
        if controller:
            w_c = 2 * np.pi / control_period
            for i in (0, 1, 2, 5):
                self.C[i, i] = M[i, i] * w_c**2
                B[i, i] = 2 * control_zeta * M[i, i] * w_c
        self.B = B

    # ---- the wave force and moment on the hull, per metre of wave amplitude ----
    def excitation(self, omega, mu):
        """F_hat for a = 1 m. Returns shape (n, 6), complex."""
        om = np.atleast_1d(np.asarray(omega, float))
        K = om**2 / G
        E = np.exp(-K * self.depth)                 # depth decay uses the true wavenumber
        kap = K * np.cos(mu)                        # wavenumber seen along the hull axis
        if self.shape == "capsule":
            # the same strip sums as below, for a varying section area a(x):
            # I0 = sum a(x) e^{i kap x} (plays V sinc), I1 = sum a(x) x e^{i kap x} (plays i A SM)
            ph = np.exp(1j * kap[:, None] * self.xs[None, :])
            I0 = np.trapezoid(self.a * ph, self.xs, axis=1)
            I1 = np.trapezoid(self.a * self.xs * ph, self.xs, axis=1)
            fz = (1 + self.ca) * RHO * om**2 * E
            F = np.zeros((om.size, 6), complex)
            F[:, 0] = -1j * (1 + self.ca_axial) * RHO * om**2 * E * np.cos(mu) * I0
            F[:, 1] = -1j * fz * np.sin(mu) * I0
            F[:, 2] = -fz * I0
            F[:, 3] = 0.0                           # pressure acts through the axis of a circle
            F[:, 4] = -fz * I1
            F[:, 5] = -1j * fz * np.sin(mu) * I1
            return F
        u = kap * self.L / 2
        sinc = np.sinc(u / np.pi)
        small = np.abs(u) < 1e-3
        with np.errstate(divide="ignore", invalid="ignore"):
            SM = np.where(small, kap * self.L**3 / 12,
                          2 * (np.sin(u) - u * np.cos(u)) / kap**2)    # integral of x sin(kap x) dx
        fz = (1 + self.ca) * RHO * om**2 * E        # common factor, per unit volume or area
        F = np.zeros((om.size, 6), complex)
        F[:, 0] = -1j * (1 + self.ca_axial) * RHO * self.V * om**2 * E * np.cos(mu) * sinc
        F[:, 1] = -1j * fz * self.V * np.sin(mu) * sinc
        F[:, 2] = -fz * self.V * sinc
        F[:, 3] = 0.0                               # pressure acts through the axis of a circle
        F[:, 4] = -1j * fz * self.A * SM
        F[:, 5] = fz * self.A * np.sin(mu) * SM
        return F

    def transfer(self, omega, mu):
        """H = X_hat / a for the six motions. Shape (n, 6), or (6,) for a scalar omega."""
        om = np.atleast_1d(np.asarray(omega, float))
        D = self.C[None] - om[:, None, None]**2 * self.M[None] - 1j * om[:, None, None] * self.B[None]
        X = np.linalg.solve(D, self.excitation(om, mu)[..., None])[..., 0]
        if self.pitch_down or self.ref_offset:
            X = X.copy()
            X[:, 4] *= -1                           # bow-down pitch from here on
            h = self.ref_offset                     # point h below the centre of buoyancy: xi + Theta x (0, 0, -h)
            X[:, 0] += -h * X[:, 4]
            X[:, 1] += h * X[:, 3]
            if not self.pitch_down:
                X[:, 4] *= -1                       # back to bow-up if that was asked for
        return X[0] if np.isscalar(omega) else X


def generator_hull(depth=15.0, length=70.0, radius=5.0, zeta_roll=0.05, zeta_pitch=0.0, **kw):
    """Hull6 set up for the vehicle in submarine.py / sensors.py (capsule 70 m x 5 m radius, CG 0.5 m below the
    centre, sensor origin 0.2 m below it, controller with a 60 s period and damping ratio 0.7).
    The damping ratios are ASSUMPTIONS: the defaults are this model's own (roll 0.05 as before, no pitch damping),
    which are not taken from the generator; vary them to see how much they matter."""
    args = dict(length=length, radius=radius, depth=depth, bg=0.5, shape="capsule", controller=True,
                control_period=60.0, control_zeta=0.7, zeta_roll=zeta_roll, zeta_pitch=zeta_pitch,
                pitch_down=True, ref_offset=0.2)
    args.update(kw)
    return Hull6(**args)


def period_to_omega(T):
    return 2 * np.pi / np.asarray(T, float)


def per_metre(H):
    """Translations in m/m, rotations in degrees per metre of wave."""
    out = np.abs(H).copy()
    out[..., 3:] = np.degrees(out[..., 3:])
    return out


# ---- plot -------------------------------------------------------------------
def demo():
    import matplotlib.pyplot as plt

    surface, ink, grey = "#fcfcfb", "#0b0b0b", "#52514e"
    colours = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]       # validated, adjacent pairs
    styles = ["-", "--", "-.", ":"]
    headings = [0, 30, 60, 90]

    hull = Hull6()
    T = np.linspace(4.0, 16.0, 400)
    om = period_to_omega(T)
    units = ["m per m of wave"] * 3 + ["degrees per m of wave"] * 3

    fig, axs = plt.subplots(2, 3, figsize=(15, 8), facecolor=surface)
    for j, (h, c, s) in enumerate(zip(headings, colours, styles)):
        H = per_metre(hull.transfer(om, np.radians(h)))
        for i, ax in enumerate(axs.flat):
            ax.plot(T, H[:, i], color=c, ls=s, lw=1.8, label=f"heading {h}°")
    for i, ax in enumerate(axs.flat):
        ax.set_facecolor(surface)
        ax.set_title(DOF[i], loc="left", color=ink, fontsize=11)
        ax.set_ylabel(units[i], color=grey, fontsize=9)
        ax.set_xlabel("wave period (s)", color=grey, fontsize=9)
        ax.grid(color="#e4e3df", lw=0.6)
        ax.tick_params(colors=grey, labelsize=8)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        for sp in ["left", "bottom"]:
            ax.spines[sp].set_color("#c9c8c3")
    axs[1, 0].annotate("← roll–sway resonance near 4 s\n    (roll damping is assumed)", (6.2, 0.08),
                       fontsize=8, color=grey)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=10, labelcolor=ink)
    fig.suptitle("Six-degree-of-freedom response, 40 m hull at 20 m depth, a = 1 m", x=0.01, ha="left",
                 color=ink, fontsize=13)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    return fig


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    fig = demo()
    fig.savefig("six_dof_response.png", dpi=100, facecolor=fig.get_facecolor())
    plt.show()
