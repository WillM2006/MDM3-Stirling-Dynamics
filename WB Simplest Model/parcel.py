"""A bit of water that goes wherever the flow takes it.

Mean depth is z0 (the centre of its circle). A crest is overhead at t = 0,
so it starts at the top of the circle.
"""
import numpy as np
from scipy.integrate import solve_ivp

from waves import RegularWave


def analytic_parcel(wave, z0, t):
    """Integrate u and w over time at x = 0. Gives a circle of radius a*exp(k z0)."""
    r = wave.a * np.exp(wave.k * z0)
    return r * np.sin(wave.omega * t), z0 + r * np.cos(wave.omega * t)


def simulate_parcel(wave, z0, t):
    """Same thing numerically, with the field evaluated where the parcel actually is."""
    r = wave.a * np.exp(wave.k * z0)

    def rhs(tt, p):
        return [wave.u(p[0], p[1], tt), wave.w(p[0], p[1], tt)]

    sol = solve_ivp(rhs, (t[0], t[-1]), [0.0, z0 + r], t_eval=t, rtol=1e-9, atol=1e-12)
    return sol.y[0], sol.y[1]


def acceleration_along_path(wave, x, z, t):
    """Differentiate the water velocity sampled along the path."""
    return np.gradient(wave.w(x, z, t), t)


def demo():
    import matplotlib.pyplot as plt

    wave = RegularWave(1.0, 8.0)
    t = np.linspace(0, 16, 1601)
    depths = [0.0, -10.0, -20.0, -50.0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for i, z0 in enumerate(depths):
        c = f"C{i}"
        x, z = simulate_parcel(wave, z0, t)
        ax1.plot(x, z - z0, color=c, label=f"z0 = {z0:.0f} m")  # centred, so the circles nest

        p_parcel = wave.pressure(x, z, t)
        p_fixed = wave.pressure(0.0, z0, t)
        ax2.plot(t, (p_parcel - p_parcel.mean()) / 1e3, color=c)
        ax2.plot(t, (p_fixed - p_fixed.mean()) / 1e3, color=c, ls="--")

    ax1.set_aspect("equal")
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("height above mean depth (m)")
    ax1.set_title("parcel paths")
    ax1.legend()

    ax2.plot([], [], "k-", label="on the parcel")
    ax2.plot([], [], "k--", label="fixed point, same depth")
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("pressure minus mean (kPa)")
    ax2.set_title("pressure")
    ax2.legend()
    fig.tight_layout()
    return fig


def stokes_demo():
    """Linear theory says the circle closes. Evaluated at the true position it creeps forward."""
    import matplotlib.pyplot as plt

    z0, T = -5.0, 8.0
    t = np.linspace(0, 5 * T, 5 * 400 + 1)
    fig, ax = plt.subplots(figsize=(6, 4))
    for a in [0.1, 0.5, 1.0, 2.0]:
        wave = RegularWave(a, T)
        r = a * np.exp(wave.k * z0)
        x, _ = simulate_parcel(wave, z0, t)
        ax.plot(range(6), x[::400] / r, "o-", label=f"a = {a} m")
    ax.set_xlabel("periods")
    ax.set_ylabel("x at the end of each period / radius")
    ax.legend()
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    demo()
    stokes_demo()
    plt.show()
