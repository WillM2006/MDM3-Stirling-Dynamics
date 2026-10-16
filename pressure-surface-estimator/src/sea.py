"""KD/LR finite-band normalization + WM's Hz grid / cos-2s spreading.

One stationary, directionally spread JONSWAP system. No measurement noise.
"""
from dataclasses import dataclass
import numpy as np

G = 9.81


def jonswap(f, Hs, Tp, gamma=3.3):
    f = np.asarray(f, dtype=float)
    if f.ndim != 1 or len(f) < 2 or np.any(f <= 0):
        raise ValueError("Positive 1D frequency grid required")
    df = np.diff(f)
    if not np.allclose(df, df[0]) or df[0] <= 0:
        raise ValueError("Uniform increasing frequency grid required")
    if Hs <= 0 or Tp <= 0 or gamma < 1:
        raise ValueError("Hs and Tp positive, gamma >= 1 required")
    fp = 1.0 / Tp
    sigma = np.where(f <= fp, .07, .09)
    peak = np.exp(-.5 * ((f / fp - 1) / sigma)**2)
    # Work in log space to avoid underflow for trial peaks outside the band.
    log_shape = -5 * np.log(f) - 1.25 * (fp / f)**4 + peak * np.log(gamma)
    shape = np.exp(log_shape - np.max(log_shape))
    return shape * (Hs / 4)**2 / (np.sum(shape) * df[0])


@dataclass(frozen=True)
class FixedSea:
    gamma: float = 3.3
    mean_direction: float = np.deg2rad(30.0)
    spread_s: float = 10.0

    def fractions(self, theta):
        theta = np.asarray(theta, float)
        if self.spread_s < 0:
            raise ValueError("Spreading exponent must be nonnegative")
        # Full-circle, equal-width bins, as used by WM. Wrap the angle.
        ordered = np.sort(np.mod(theta, 2 * np.pi))
        gaps = np.diff(np.r_[ordered, ordered[0] + 2 * np.pi])
        if len(theta) < 2 or not np.allclose(gaps, 2 * np.pi / len(theta)):
            raise ValueError("Directions must cover the full circle evenly")
        delta = np.angle(np.exp(1j * (theta - self.mean_direction)))
        p = np.maximum(np.cos(delta / 2), 0)**(2 * self.spread_s)
        return p / p.sum()  # discrete normalization preserves Hs exactly


class Sea:
    """The team-style f/omega/theta/k/phi interface, with explicit Hs and Tp."""
    def __init__(self, *, freq=None, omega=None, theta=None, Hs=2., Tp=8.,
                 fixed=None, seed=42):
        if (freq is None) == (omega is None):
            raise ValueError("Give either freq (Hz) or omega (rad/s)")
        self.f = np.asarray(freq if freq is not None else np.asarray(omega) / (2*np.pi), float)
        self.omega = 2 * np.pi * self.f
        self.theta = np.asarray(theta if theta is not None else np.linspace(0, 2*np.pi, 36, endpoint=False), float)
        self.fixed = fixed or FixedSea()
        self.Hs, self.Tp = Hs, Tp
        self.S_f = jonswap(self.f, Hs, Tp, self.fixed.gamma)
        self.delta_f = self.f[1] - self.f[0]
        self.delta_theta = 2 * np.pi / len(self.theta)
        self.p = self.fixed.fractions(self.theta)
        self.k = self.omega**2 / G
        self.phi = np.random.default_rng(seed).uniform(0, 2*np.pi, (len(self.f), len(self.theta)))
        self.a = np.sqrt(2 * self.S_f[:, None] * self.delta_f * self.p[None, :])

    def sea_spectrum(self):
        D = self.p / self.delta_theta
        S = self.S_f[:, None] * D[None, :]
        return self.S_f, D, S, self.a, 4*np.sqrt(np.sum(self.a**2)/2)

    def superposition(self, x, y, t):
        phase = self.k[:, None] * (x*np.cos(self.theta)[None, :] + y*np.sin(self.theta)[None, :])
        return np.sum(self.a * np.cos(phase - self.omega[:, None]*t + self.phi)), self.a

    @property
    def complex_components(self):
        return self.a * np.exp(1j * self.phi)
