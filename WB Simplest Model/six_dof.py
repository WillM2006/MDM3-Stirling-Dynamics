
# FREQUENCY DOMAIN: hull moves with six degrees of freedom. Linear waves, deep water, hull not moving forward.
# Waves can approach via any heading.
# We are in frequency domain meaning whole model is one 6x6 solve per frequency:
#     [C - w^2 M - i w B]X = F        H = X/a (response function)
# Axes: x along hull (bow +x), y to port, z up, origin at the centre of buoyancy (middle of
# hull axis) 
# Order of six degrees of freedom is: surge, sway, heave, roll, pitch, yaw.
# Pitch is positive bow up. Roll and yaw are right-handed about x and z.
# Wave heading is mu (angle from +x towards +y of the direction the waves travel and 0 = towards the bow)
# H (transfer function) is metres per metre for the first three and radians per metre for the last three

import numpy as np
from waves import G, RHO

DOF = ["surge", "sway", "heave", "roll", "pitch", "yaw"]

class Hull6:
    def __init__(self, length=40.0, radius=2.0, depth=20.0, mass_ratio=1.0, ca=1.0, ca_axial=0.02, bg=0.5, hold_period=60.0, roll_gyration=None, zeta_roll=0.05, couple=True):
        self.L, self.R, self.depth, self.bg = length, radius, depth, bg
        self.ca, self.ca_axial = ca, ca_axial      # cross-flow and axial added-mass coefficients
        self.couple = couple                       # setting False drops the BG coupling terms
        self.A = np.pi * radius**2
        self.V = self.A * length
        self.m = mass_ratio * RHO * self.V

        m, b, L, R = self.m, bg, length, radius
        if roll_gyration is None:
            k_gyr = R / np.sqrt(2) # solid cylinder: I = m R^2 / 2
        else:
            k_gyr = roll_gyration    

        # added mass about the centre of buoyancy
        A11 = ca_axial * RHO * self.V
        A22 = A33 = ca * RHO * self.V
        A44 = 0.0
        A55 = A66 = ca * RHO * self.V * L**2 / 12
        # moment of intertia of hull about centre of buoyancy: solid cyclidder about axis plus parallel-axis terms
        I_roll = m * k_gyr**2 + m * b**2
        I_pitch = m * L**2 / 12 + m * b**2
        I_yaw = m * L**2 / 12
        M = np.diag([m + A11, m + A22, m + A33, I_roll + A44, I_pitch + A55, I_yaw + A66])
        # coupled terms due to CoG being distance b below CoB
        if couple:
            M[0, 4] = M[4, 0] = m * b      # surge and pitch
            M[1, 3] = M[3, 1] = m * b      # sway and roll
        self.M = M

        # restoring/ stiffness matrix
        # only heave (makeshift depth controller), roll and pitch (CoG below CoB).
        c_heave = (2 * np.pi / hold_period) ** 2 * (m + A33)
        k_tilt = RHO * G * self.V * b               # same as submarine.py
        self.C = np.diag([0.0, 0.0, c_heave, k_tilt, k_tilt, 0.0])

        # damping matrix: roll only, zeta_roll percent of critical rest is zeros
        B = np.zeros((6, 6))
        B[3, 3] = 2 * zeta_roll * np.sqrt(k_tilt * M[3, 3])
        self.B = B

    # Wave force and moment on the hull (per metre of wave amplitude)
    def excitation(self, omega, mu):
        # F_hat for amplitude a = 1 m - returns shape (n, 6) and is complex
        om = np.atleast_1d(np.asarray(omega, float))
        K = om**2 / G
        E = np.exp(-K * self.depth)                 # depth decay uses the true wavenumber
        kap = K * np.cos(mu)                        # kappa: wavenumber seen along the hull axis i.e. waves align with hull
        u = kap * self.L / 2
        sinc = np.sinc(u / np.pi)
        small = np.abs(u) < 1e-3
        with np.errstate(divide="ignore", invalid="ignore"):
            SM = np.where(small, kap * self.L**3 / 12, 2 * (np.sin(u) - u * np.cos(u)) / kap**2)    # integral of x sin(kap x) dx
        fz = (1 + self.ca) * RHO * om**2 * E        # common factor (per unit volume or area)
        F = np.zeros((om.size, 6), complex)
        F[:, 0] = -1j * (1 + self.ca_axial) * RHO * self.V * om**2 * E * np.cos(mu) * sinc
        F[:, 1] = -1j * fz * self.V * np.sin(mu) * sinc
        F[:, 2] = -fz * self.V * sinc
        F[:, 3] = 0.0                               # pressure acts through axis of a circle
        F[:, 4] = -1j * fz * self.A * SM
        F[:, 5] = fz * self.A * np.sin(mu) * SM
        return F

    def transfer(self, omega, mu):
        # H = X_hat / a for the six motions - shape (n, 6) or (6,) for a scalar omega
        om = np.atleast_1d(np.asarray(omega, float))
        D = self.C[None] - om[:, None, None]**2 * self.M[None] - 1j * om[:, None, None] * self.B[None]
        X = np.linalg.solve(D, self.excitation(om, mu)[..., None])[..., 0]
        return X[0] if np.isscalar(omega) else X


def period_to_omega(T):
    return 2 * np.pi / np.asarray(T, float)


def per_metre(H):
    # Translations in m per metre of wave, rotations in degrees per metre of wave
    out = np.abs(H).copy()
    out[..., 3:] = np.degrees(out[..., 3:])
    return out


# PLOTTING motion in all 6 degrees of freedom for range of wave periods + headings
def demo():
    import matplotlib.pyplot as plt

    surface, ink, grey = "#fcfcfb", "#0b0b0b", "#52514e"
    colours = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]       
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
