import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma as G
from scipy.integrate import cumulative_trapezoid

class Sea():

    def __init__(self, 
                 n_freq=40, # 
                 n_dir=36, # 
                 omega_range=(0.1, 2.0),
                 Hs=1.7, # Significant Wave Height 
                 Tp=5.9, # Peak Period
                 direction=0.0, # mean direction (radians)
                 gamma=3.3, # peak enhancement factor: 1 for pm / 3.3 for jonswap
                 s=5, # directional spreading factor
                 seed=42
                 ):
        

        rng = np.random.default_rng(seed)

        self.g = 9.81
        self.Hs, self.Tp, self.gamma, self.s = Hs, Tp, gamma, s
        self.seed, self.omega_range = seed, omega_range


        self.theta_m = direction # mean wave direction
        self.f_p = 1 / Tp # peak frequency: 0.17Hz calm / 0.12Hz moderate / 0.08Hz rough / 0.065Hz storm

        # directions evenly spaced round a full circle
        self.theta = np.linspace(-np.pi, np.pi, n_dir, endpoint=False)
        self.delta_theta = np.full(n_dir, 2 * np.pi / n_dir) 

        # frequencies equal energy bins / each frequency is at its bins energy midpoint
        f_fine = np.linspace(*omega_range, 5000) / (2 * np.pi)
        E = cumulative_trapezoid(self.jonswap(f_fine), f_fine, initial=0)

        self.f = np.interp(np.linspace(0, E[-1], 2 * n_freq + 1), E, f_fine)[1::2]
        self.delta_f = (E[-1] / n_freq) / self.jonswap(self.f)
        self.omega = 2 * np.pi * self.f

        # wavenumber from deep-water dispersion relation: w² = g k
        self.k = self.omega**2 / self.g
        self.phi = rng.uniform(0, 2 * np.pi, (len(self.f), len(self.theta)))

        # compute wave frequencies, directions, directional_frequency_spectrum, amplitudes
        self.S_f = self.jonswap(self.f) # (N,)
        self.D = self.directional_spreading(self.theta) # (M,)
        self.directional_frequency_spectrum = self.S_f[:, None] * self.D[None, :] # (N, M)
        self.wave_amplitudes = np.sqrt(2 * self.directional_frequency_spectrum * self.delta_f[:, None] * self.delta_theta[None, :]) # (N, M)

    def jonswap(self, f):
        """S_f spectrum """
        sigma_1, sigma_2 = 0.07, 0.09 # peak width

        sigma = np.where(f <= self.f_p, sigma_1, sigma_2)
        b = np.exp(-(f / self.f_p - 1)**2 / (2 * sigma**2))
        alpha = (5 / 16) * self.Hs**2 * (2 * np.pi * self.f_p)**4 / self.g**2 * (1 - 0.287 * np.log(self.gamma))
        S_f = alpha * self.g**2 / (16 * np.pi**4) * f**-5 * np.exp(-1.25 * (f / self.f_p)**-4) * self.gamma**b

        return S_f

    def directional_spreading(self, theta):
        """Directional Spectrum / spreading function / D(θ)"""
        d = np.arctan2(np.sin(theta - self.theta_m), np.cos(theta - self.theta_m))  # wrap to [-π, π]
        D = np.where(np.abs(d) <= np.pi / 2, np.cos(d)**(2 * self.s), 0.0)
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
    sea = Sea(n_freq=40, n_dir=36, direction=np.pi/4, s=2.5)
    sea.plot(X=np.linspace(0, 100, 50), Y=np.linspace(0, 100, 50))
    
