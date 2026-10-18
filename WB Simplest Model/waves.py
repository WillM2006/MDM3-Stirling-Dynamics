
#class deifining one regular wave assuming it is linear (not too steep) and in deep water
# z is up and z=0 is the mean surface
import numpy as np

G = 9.81
RHO = 1025.0

class RegularWave:
    def __init__(self, amplitude, period):
        self.a = amplitude
        self.T = period
        self.omega = 2 * np.pi / period
        self.k = self.omega**2 / G          # deep water: omega^2 = g k THIS is dispersion relation
        self.wavelength = 2 * np.pi / self.k
        self.c = self.omega / self.k

    def theta(self, x, t):
        return self.k * x - self.omega * t

    def eta(self, x, t):
        return self.a * np.cos(self.theta(x, t))

    # u, w, pressure all contain exp(kz) meaning they decay w/ depth
    def u(self, x, z, t):
        return self.a * self.omega * np.exp(self.k * z) * np.cos(self.theta(x, t))

    def w(self, x, z, t):
        return self.a * self.omega * np.exp(self.k * z) * np.sin(self.theta(x, t))

    def accel_w(self, x, z, t):
        #dw/dt at a fixed point
        return -self.a * self.omega**2 * np.exp(self.k * z) * np.cos(self.theta(x, t))

    def pressure(self, x, z, t):
        #Gauge pressure (difference vs atmospheric): hydrostatic plus the wave part.
        return -RHO * G * z + RHO * G * self.a * np.exp(self.k * z) * np.cos(self.theta(x, t))
