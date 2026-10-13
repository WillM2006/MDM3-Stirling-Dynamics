import numpy as np
from scipy.special import gamma as G
import matplotlib.pyplot as plt
import time 

# change var idkname



class Sea():

    def __init__(self, freq=None, omega=None, theta=None, seed=42):
        """Give EITHER freq (Hz) OR omega (rad/s), plus theta (wave directions, rad)
           e.g. Sea(omega=OMEGAS, theta=BETAS)  or  Sea(freq=np.linspace(0.03, 1.0, 100), theta=BETAS)"""

        g = 9.81

        rng = np.random.default_rng(seed)

        if (freq is None) == (omega is None):
            raise ValueError("give either freq= (Hz) or omega= (rad/s), not both")

        # frequency grid - store both, the rest of the class uses self.f (Hz)
        if freq is not None:
            self.f = np.asarray(freq, dtype=float)
            self.omega = 2 * np.pi * self.f
        else:
            self.omega = np.asarray(omega, dtype=float)
            self.f = self.omega / (2 * np.pi)
        self.delta_f = self.f[1] - self.f[0]

        # direction grid
        self.theta = np.linspace(-np.pi, np.pi, 36, endpoint=False) if theta is None else np.asarray(theta, dtype=float)
        self.delta_theta = 2 * np.pi / len(self.theta) # assumes the directions cover the full circle evenly

        # wavenumber from deep-water dispersion relation: w² = g k
        self.k = self.omega**2 / g
        self.phi = rng.uniform(0, 2 * np.pi, (len(self.f), len(self.theta)))



    def jonswap(self, f):
        """S_f spectrum """

        g = 9.81 # gravity
        gamma = 3.3 # peak enhancement factor: 1 for pm / 3.3 for jonswap
        sigma_1, sigma_2 = 0.07, 0.09 # peak width
        alpha = 0.0081 # energy scale: 0.0081 pm value (can use DNV normailsation instead)
        f_m = 0.17 # peak frequency: 0.17Hz calm / 0.12Hz moderate / 0.08Hz rough / 0.065Hz storm

        sigma = sigma_1 if f <= f_m else sigma_2
        b = np.exp(-1/(2*sigma**2) * (f/f_m - 1)**2)

        S_f = alpha * g**2 / (16 * np.pi**4) * f**-5 * np.exp(-5/4 * (f/f_m)**-4) * gamma**b

        return S_f

    def longuet_higgins(self, theta):
        """Directional Spectrum / spreading function / D(θ)"""

        theta_m = 0 # mean wave direction
        s = 10 # directional spreading factor: 10 wind waves / 25 short swell / 75 long swell
        Q = G(s + 1) / (2 * np.sqrt(np.pi) * G(s + 0.5)) # normalizing factor (integration of D_theta goes to 1)

        D_theta = Q * np.cos(np.abs(theta - theta_m)/2)**(2 * s)

        return D_theta

    def directional_frequency_spectrum(self, f, theta):
        return self.jonswap(f) * self.longuet_higgins(theta)

    def superposition(self, x, y, t):
        '''Surface elavation'''

        H_t = 0
        wave_amplitudes = np.empty((len(self.f), len(self.theta)))

        for n in range(len(self.f)):
            for m in range(len(self.theta)):

                k_dot_X = self.k[n] * (x * np.cos(self.theta[m]) + y * np.sin(self.theta[m]))

                wave_amplitude = np.sqrt(2 * self.directional_frequency_spectrum(self.f[n], self.theta[m]) * self.delta_f * self.delta_theta)
                idkname = np.cos(k_dot_X - 2 * np.pi * self.f[n] * t + self.phi[n, m])

                wave_amplitudes[n,m] = wave_amplitude
                H_t += wave_amplitude * idkname

        return H_t, wave_amplitudes


    def sea_spectrum(self):
        """S(f), D(theta), S(f, theta), wave amplitudes and Hs"""
        S_f = np.array([self.jonswap(f) for f in self.f]) # jonswap only takes scalars
        D = self.longuet_higgins(self.theta)
        S = S_f[:, None] * D[None, :]
        a = np.sqrt(2 * S * self.delta_f * self.delta_theta)
        Hs = 4 * np.sqrt(np.sum(a**2) / 2)
        return S_f, D, S, a, Hs


    def plot(self, X=None, Y=None, T=None, extra=None):
        """extra(ax3, i) - optional function to draw something else (e.g. the sub) on frame i"""

        X = np.linspace(0, 10, 50) if X is None else X     # metres
        Y = np.linspace(0, 10, 50) if Y is None else Y
        T = np.linspace(0, 10, 100) if T is None else T      # seconds

        Xg, Yg = np.meshgrid(X, Y)      # 2D grids of every (x, y) point, shape (50, 50)

        surface = np.empty((len(T), len(Y), len(X)))     # note: np.empty takes a tuple

        for i, t in enumerate(T):
            surface[i], _ = self.superposition(Xg, Yg, t)    # whole surface at time t


        '''fig, ax = plt.subplots()
        c = ax.contourf(X, Y, surface[0], levels=30, cmap='viridis')
        fig.colorbar(c, label='η (m)')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_title(f't = {T[0]:.1f} s')
        ax.set_aspect('equal')'''


        fig = plt.figure()
        ax3 = fig.add_subplot(projection='3d')
        for i in range(len(T)):
            ax3.clear()
            ax3.set_zlim(-2, 2)
            ax3.plot_surface(Xg, Yg, surface[i], cmap='viridis')
            ax3.set_zlabel('η (m)')
            if extra is not None:
                extra(ax3, i)
            plt.pause(0.05)

        plt.show()


if __name__ == "__main__":
    OMEGAS = np.linspace(0.1, 2.0, 40)
    BETAS = np.deg2rad(np.arange(0, 360, 10))


    FREQS = np.linspace(0.03, 1.0, 100)
    THETAS = np.linspace(-np.pi, np.pi, 36, endpoint=False)


    sea = Sea(freq=FREQS, theta=THETAS)
    sea.plot()
