import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma as G
from scipy.integrate import cumulative_trapezoid
from scipy.signal import ZoomFFT


def uniform_wave_sum(omega, times, coefficients):
    """Evaluate Re(sum c exp(-i omega t)) on uniform grids, without a large phase array."""
    if len(omega) < 2 or len(times) < 2:
        return np.real(np.exp(-1j * np.outer(times, omega)) @ coefficients)
    dw, dt = omega[1] - omega[0], times[1] - times[0]
    if not (np.allclose(np.diff(omega), dw, rtol=0, atol=2e-14)
            and np.allclose(np.diff(times), dt, rtol=0, atol=1e-10)):
        raise ValueError("wave synthesis requires uniform frequency and time grids")
    shifted = coefficients * np.exp(-1j * omega * times[0])[:, None]
    values = ZoomFFT(len(omega), [0, len(times) * dw * dt], m=len(times), fs=2*np.pi)(shifted, axis=0)
    return np.real(np.exp(-1j * omega[0] * (times-times[0]))[:, None] * values)


class Sea():

    def __init__(self, 
                 n_freq=40, # 
                 n_dir=36, # 
                 omega_range=(0.1, 2.0),
                 # one wave system (shorthand, used when systems is None)
                 Hs=1.7, # Significant Wave Height
                 Tp=5.9, # Peak Period
                 direction=0.0, # mean direction (radians)
                 gamma=3.3, # peak enhancement factor: 1 for pm / 3.3 for jonswap
                 s=5, # directional spreading factor
                 seed=42,
                 systems=None, # [dict(Hs, Tp, gamma, s, direction (rad)), ...] wave systems, each with its own spectrum
                               # and spreading; overrides the single-system args above. Wind sea + swell = two systems
                               # with different Hs/Tp; two directions of one sea = two systems with the same Tp/gamma
                               # (the power splits as Hs_i², total Hs = sqrt(Σ Hs_i²))
                 uniform_frequencies=False, # evenly spaced frequencies (each bin's energy integrated) for the fast
                                            # FFT wave synthesis; False = equal-energy bins
                 ):

        self.g = 9.81

        # every sea is a list of wave systems: S(f, θ) = Σ_i S_i(f; Hs_i, Tp_i, gamma_i) · D_i(θ; direction_i, s_i)
        if systems is None or len(systems) == 0:
            systems = [dict(Hs=Hs, Tp=Tp, gamma=gamma, s=s, direction=direction)]
        keys = ("Hs", "Tp", "gamma", "s", "direction")
        self.systems = []
        for q in systems:
            missing = [k for k in keys if k not in q]
            if missing:
                raise ValueError(f"wave system {q} is missing {missing}")
            q = {k: float(q[k]) for k in keys}
            if not all(np.isfinite(v) for v in q.values()) or min(q["Hs"], q["Tp"], q["gamma"]) <= 0 or q["s"] < 0:
                raise ValueError(f"wave system needs finite values with Hs, Tp, gamma > 0 and s >= 0, got {q}")
            self.systems.append(q)

        rng = np.random.default_rng(seed)

        self.seed, self.omega_range = seed, omega_range
        self.uniform_frequencies = uniform_frequencies

        # summary values (printing / plots only, the truth is self.systems): total Hs = sqrt(Σ Hs_i²), the rest from
        # the dominant system = the one with the tallest S(f) peak, so Tp is (to within the overlap of the spectra) the
        # peak period of the total spectrum. Not the largest Hs: a swell with lower Hs often has the taller peak.
        # With one system these are just its values
        dom = max(self.systems, key=lambda q: self._jonswap_one(1 / q["Tp"], q["Hs"], q["Tp"], q["gamma"]))
        self.Hs = float(np.sqrt(sum(q["Hs"]**2 for q in self.systems))) if len(self.systems) > 1 else dom["Hs"]
        self.Tp, self.gamma, self.s, self.theta_m = dom["Tp"], dom["gamma"], dom["s"], dom["direction"]  # theta_m: mean direction
        self.f_p = 1 / self.Tp # peak frequency: 0.17Hz calm / 0.12Hz moderate / 0.08Hz rough / 0.065Hz storm

        # directions evenly spaced round a full circle
        self.theta = np.linspace(-np.pi, np.pi, n_dir, endpoint=False)
        self.delta_theta = np.full(n_dir, 2 * np.pi / n_dir) 

        # frequencies equal energy bins / each frequency is at its bins energy midpoint
        f_fine = np.linspace(*omega_range, 5000) / (2 * np.pi)
        E = cumulative_trapezoid(self.jonswap(f_fine), f_fine, initial=0)

        self.f = np.interp(np.linspace(0, E[-1], 2 * n_freq + 1), E, f_fine)[1::2]
        self.delta_f = (E[-1] / n_freq) / self.jonswap(self.f)
        if uniform_frequencies:
            edges = np.linspace(f_fine[0], f_fine[-1], n_freq + 1)
            self.f = (edges[:-1] + edges[1:]) / 2
            self.delta_f = np.diff(edges)
        self.omega = 2 * np.pi * self.f

        # wavenumber from deep-water dispersion relation: w² = g k
        self.k = self.omega**2 / self.g
        self.phi = rng.uniform(0, 2 * np.pi, (len(self.f), len(self.theta)))

        # compute wave frequencies, directions, directional_frequency_spectrum, amplitudes
        if uniform_frequencies:   # each system's energy in each uniform bin / bin width (their sum is the total above)
            self.S_sys = [np.diff(np.interp(edges, f_fine, cumulative_trapezoid(
                              self._jonswap_one(f_fine, q["Hs"], q["Tp"], q["gamma"]), f_fine, initial=0))) / self.delta_f
                          for q in self.systems] # each (N,)
        else:
            self.S_sys = [self._jonswap_one(self.f, q["Hs"], q["Tp"], q["gamma"]) for q in self.systems] # each (N,)
        self.D_sys = [self._cos2s(self.theta, q["direction"], q["s"]) for q in self.systems] # each (M,), integrates to 1
        self.directional_frequency_spectrum = sum(S[:, None] * D[None, :] for S, D in zip(self.S_sys, self.D_sys)) # (N, M)
        self.S_f = sum(self.S_sys) # (N,) total
        self.D = self.directional_spreading(self.theta) # (M,) energy-weighted over frequency
        self.wave_amplitudes = np.sqrt(2 * self.directional_frequency_spectrum * self.delta_f[:, None] * self.delta_theta[None, :]) # (N, M)

        self.Hs_actual = 4 * np.sqrt(np.sum(self.wave_amplitudes**2) / 2)

    def _jonswap_one(self, f, Hs, Tp, gamma):
        """S_f spectrum of one wave system"""
        sigma_1, sigma_2 = 0.07, 0.09 # peak width
        f_p = 1 / Tp

        sigma = np.where(f <= f_p, sigma_1, sigma_2)
        b = np.exp(-(f / f_p - 1)**2 / (2 * sigma**2))
        alpha = (5 / 16) * Hs**2 * (2 * np.pi * f_p)**4 / self.g**2 * (1 - 0.287 * np.log(gamma))
        S_f = alpha * self.g**2 / (16 * np.pi**4) * f**-5 * np.exp(-1.25 * (f / f_p)**-4) * gamma**b

        return S_f

    def jonswap(self, f):
        """total S_f spectrum: the sum of every wave system's JONSWAP (one system = the plain JONSWAP)"""
        return sum(self._jonswap_one(f, q["Hs"], q["Tp"], q["gamma"]) for q in self.systems)

    def _cos2s(self, theta, theta_m, s):
        """one cos^2s spreading peak at theta_m, cut off at ±90°, normalised to integrate to 1 over the theta grid"""
        d = np.arctan2(np.sin(theta - theta_m), np.cos(theta - theta_m))  # wrap to [-π, π]
        P = np.where(np.abs(d) <= np.pi / 2, np.cos(d)**(2 * s), 0.0)
        return P / np.sum(P * self.delta_theta)

    def directional_spreading(self, theta):
        """Directional Spectrum / spreading function / D(θ): the energy-weighted mix of every system's cos^2s peak,
        D = Σ_i m0_i D_i / Σ_i m0_i (m0_i = power of system i), so a system's share is its power whatever its spread;
        with one system it is the plain cos^2s spreading at (direction, s)."""
        m0 = [np.sum(S * self.delta_f) for S in self.S_sys] # power of each system in the sea's bins
        D = sum(m * self._cos2s(theta, q["direction"], q["s"]) for m, q in zip(m0, self.systems))
        return D / np.sum(D * self.delta_theta)


    '''    
    def longuet_higgins(self, theta):
        # longuet_higgins cos^(2s)(Δθ/2) is much broader than directional_spreadings cos^(2s)(Δθ) 
        """Directional Spectrum / spreading function / D(θ)"""

        Q = G(self.s + 1) / (2 * np.sqrt(np.pi) * G(self.s + 0.5)) # normalizing factor (integration of D_theta goes to 1)
        d = np.arctan2(np.sin(theta - self.theta_m), np.cos(theta - self.theta_m))  # wrap to [-π, π]
        D = Q * np.cos(d / 2)**(2 * self.s)

        return D    
    '''


    def superposition(self, x, y, t):
        '''Surface elavation'''
        H_t = 0
        for n in range(len(self.f)):
            for m in range(len(self.theta)):
                k_dot_X = self.k[n] * (x * np.cos(self.theta[m]) + y * np.sin(self.theta[m]))
                random_phase = np.cos(k_dot_X - 2 * np.pi * self.f[n] * t + self.phi[n, m])
                H_t += self.wave_amplitudes[n, m] * random_phase

        return H_t

    def sea_spectrum(self):
        """S(f), D(theta), S(f, theta), wave amplitudes and Hs"""
        Hs = 4 * np.sqrt(np.sum(self.wave_amplitudes**2) / 2)
        return self.S_f, self.D, self.directional_frequency_spectrum, self.wave_amplitudes, Hs



    def plot_directional_spectrum(self, show=True):
        """S(f, θ) as a 3D surface and a polar (circular) map, D(θ) on a polar axis, and the frequency spectrum S(f)"""
        S = self.directional_frequency_spectrum  # (N, M)

        # close the circle so the polar plots wrap round from θ = π back to -π
        theta_c = np.append(self.theta, self.theta[0] + 2 * np.pi)
        S_c = np.column_stack([S, S[:, 0]])
        D_c = np.append(self.D, self.D[0])

        fig = plt.figure(figsize=(13, 10))

        # 3D surface S(f, θ)
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        TH, F = np.meshgrid(np.degrees(theta_c), self.f)
        ax1.plot_surface(TH, F, S_c, cmap='viridis', linewidth=0)
        ax1.set_xlabel('θ (deg)')
        ax1.set_ylabel('f (Hz)')
        ax1.set_zlabel('S(f, θ) (m²/Hz/rad)')
        ax1.set_title('Directional spectrum S(f, θ)')

        # polar map: angle = direction, radius = frequency, colour = energy density
        ax2 = fig.add_subplot(2, 2, 2, projection='polar')
        TH, F = np.meshgrid(theta_c, self.f)
        pcm = ax2.pcolormesh(TH, F, S_c, cmap='viridis', shading='gouraud')
        fig.colorbar(pcm, ax=ax2, pad=0.1, label='S(f, θ) (m²/Hz/rad)')
        ax2.set_rlabel_position(-60)   # keep the frequency labels off the energy peak
        ax2.set_title('S(f, θ) polar (radius = f in Hz)')

        # circular spreading function D(θ)
        ax3 = fig.add_subplot(2, 2, 3, projection='polar')
        ax3.plot(theta_c, D_c)
        ax3.fill(theta_c, D_c, alpha=0.3)
        ax3.set_title('Spreading function D(θ)')

        # 2D frequency spectrum S(f): smooth JONSWAP curve + the discrete frequencies the sea actually uses
        ax4 = fig.add_subplot(2, 2, 4)
        f_fine = np.linspace(*self.omega_range, 1000) / (2 * np.pi)
        ax4.plot(f_fine, self.jonswap(f_fine), label='S(f) (JONSWAP)')
        ax4.plot(self.f, self.S_f, 'o', ms=4, label=f'{len(self.f)} discrete components')
        if len(self.systems) > 1 and len({(q["Hs"], q["Tp"], q["gamma"]) for q in self.systems}) > 1:
            for i, q in enumerate(self.systems):   # each wave system's own spectrum under the total
                ax4.plot(f_fine, self._jonswap_one(f_fine, q["Hs"], q["Tp"], q["gamma"]), '--', lw=1,
                         label=f'system {i + 1}: Hs {q["Hs"]:g}, Tp {q["Tp"]:g}, {np.degrees(q["direction"]):.0f}°')
        ax4.axvline(self.f_p, color='grey', ls='--', lw=1, label=f'f_p = {self.f_p:.3f} Hz')
        ax4.set_xlabel('f (Hz)')
        ax4.set_ylabel('S(f) (m²/Hz)')
        ax4.set_title(f'Frequency spectrum S(f)   Hs = {self.Hs_actual:.2f} m, Tp = {self.Tp} s')
        ax4.grid(alpha=0.3)
        ax4.legend()

        fig.tight_layout()
        if show:
            plt.show()
        return fig

    def plot(self, X=None, Y=None, T=None, extra=None):
        """extra(ax3, i) - optional function to draw something else (e.g. the sub) on frame i"""

        X = np.linspace(0, 10, 50) if X is None else X     # metres
        Y = np.linspace(0, 10, 50) if Y is None else Y
        T = np.linspace(0, 10, 100) if T is None else T      # seconds

        Xg, Yg = np.meshgrid(X, Y)      # 2D grids of every (x, y) point, shape (50, 50)

        surface = np.empty((len(T), len(Y), len(X)))     # note: np.empty takes a tuple

        for i, t in enumerate(T):
            surface[i] = self.superposition(Xg, Yg, t)    # whole surface at time t

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        for i in range(len(T)):
            ax.clear()
            ax.set_zlim(-5, 5)
            ax.plot_surface(Xg, Yg, surface[i], cmap='Blues_r')
            ax.set_zlabel('η (m)')
            if extra is not None:
                extra(ax, i)
            plt.pause(0.05)

        plt.show()


if __name__ == "__main__":
    # sea = Sea(n_freq=40, n_dir=36, direction=np.pi/4, s=2.5)          # one peak
    # two directions of one sea (same Tp/gamma): 70% of the power at -45 deg, 30% at +90 deg -> Hs_i = 1.7*sqrt(0.7), 1.7*sqrt(0.3)
    # sea = Sea(systems=[dict(Hs=1.42, Tp=5.9, gamma=3.3, s=5, direction=-np.pi/4), dict(Hs=0.93, Tp=5.9, gamma=3.3, s=2, direction=np.pi/2)])
    sea = Sea(systems=[dict(Hs=3.0, Tp=8.0,  gamma=2.0, s=1.5, direction=np.deg2rad(15)),     # wind sea
                       dict(Hs=2.0, Tp=14.0, gamma=5.0, s=6,   direction=np.deg2rad(225))])   # swell (bimodal in f and θ)
    sea.plot_directional_spectrum()   # close this window to start the surface animation
    sea.plot(X=np.linspace(0, 100, 50), Y=np.linspace(0, 100, 50))
