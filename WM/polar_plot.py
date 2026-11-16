"""
Shared helper: draw a directional spectrum E(f, theta) on a polar axis (angle = heading, radius = frequency).

    ax = fig.add_subplot(..., projection="polar")
    im = polar_spectrum(ax, f, beta, x, cmap="RdBu_r", vmin=-v, vmax=v)

f is the frequency grid (Hz), beta the headings (rad, any range), x[f, heading] the spectrum. Cell edges are taken
half-way between neighbouring headings, wrapping round 360 deg so the map closes with no gap.
"""
import numpy as np


def _edges(c):
    """edges half-way between sorted centres, extrapolated half a step at each end"""
    c = np.asarray(c, float)
    if c.size == 1:
        return np.array([c[0] - 0.5, c[0] + 0.5])
    mid = 0.5 * (c[1:] + c[:-1])
    return np.concatenate([[c[0] - (mid[0] - c[0])], mid, [c[-1] + (c[-1] - mid[-1])]])


def polar_spectrum(ax, f, beta, x, **kw):
    th = np.mod(np.asarray(beta, float), 2 * np.pi)
    o = np.argsort(th)
    th = th[o]
    ext = np.concatenate([[th[-1] - 2 * np.pi], th, [th[0] + 2 * np.pi]])   # periodic neighbours
    th_e = 0.5 * (ext[1:] + ext[:-1])                                       # n_dir + 1 edges
    f_e = np.clip(_edges(f), 0.0, None)
    im = ax.pcolormesh(th_e, f_e, np.asarray(x)[:, o], shading="flat", **kw)
    ax.set_ylim(0, f_e[-1])
    ax.set_thetagrids(range(0, 360, 45), [f"{d if d <= 180 else d - 360}°" for d in range(0, 360, 45)])
    ax.set_rlabel_position(112.5)
    ax.tick_params(axis="y", labelsize=7)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(alpha=0.35, lw=0.5)
    return im
