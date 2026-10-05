"""Animate the wave and the water parcels under it.

python animate.py         show it
python animate.py save    write waves.gif
"""
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle

from waves import RegularWave

A, T = 3.0, 8.0        # big enough to see, still gentle (ka is about 0.19)
N_FRAMES = 120         # one period

wave = RegularWave(A, T)
L = wave.wavelength

x_line = np.linspace(0, 2 * L, 600)
x0, z0 = np.meshgrid(np.arange(0, 2 * L + 1, L / 8), [0, -10, -20, -30, -40, -50])
r = A * np.exp(wave.k * z0)          # orbit radius shrinks with depth


def parcels(t):
    """First-order positions: each parcel circles about its mean position."""
    th = wave.theta(x0, t)
    return x0 - r * np.sin(th), z0 + r * np.cos(th)


fig, ax = plt.subplots(figsize=(12, 4.8))
ax.set_xlim(0, 2 * L)
ax.set_ylim(-60, 10)
ax.set_aspect("equal")
ax.set_xlabel("x (m)")
ax.set_ylabel("z (m)")

water = ax.fill([], [], color="#d6e6f5")[0]
surface, = ax.plot(x_line, wave.eta(x_line, 0), color="C0", lw=2)
dots = ax.scatter(x0.ravel(), z0.ravel(), c=z0.ravel(), cmap="viridis", s=16, zorder=3)

# orbits for one column, to show they shrink with depth
for zz in np.unique(z0):
    ax.add_patch(Circle((L, zz), A * np.exp(wave.k * zz), fill=False, ec="grey", lw=0.7))

title = ax.set_title("")


def update(i):
    t = i * T / N_FRAMES
    eta = wave.eta(x_line, t)
    water.set_xy(np.column_stack([np.r_[x_line, x_line[::-1]],
                                  np.r_[eta, np.full_like(x_line, -60)]]))
    surface.set_ydata(eta)
    x, z = parcels(t)
    dots.set_offsets(np.column_stack([x.ravel(), z.ravel()]))
    title.set_text(f"t = {t:.1f} s   (a = {A} m, T = {T} s, wavelength = {L:.0f} m)")
    return water, surface, dots, title


anim = FuncAnimation(fig, update, frames=N_FRAMES, interval=1000 * T / N_FRAMES, blit=False)

if __name__ == "__main__":
    if "save" in sys.argv:
        anim.save("waves.gif", writer=PillowWriter(fps=N_FRAMES / T), dpi=80)
        print("wrote waves.gif")
    else:
        plt.show()
