import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation



# JONSWAP WAVE SPECTRUM


def jonswap_spectrum(omega, Hs, Tp, gamma=3.3):
    """
    Calculate the JONSWAP wave spectrum.

    Hs    : Significant wave height (m)
    Tp    : Peak wave period (s)
    gamma : Peak enhancement factor
    omega : Angular frequency (rad/s)
    """

    g = 9.81
    peak_frequency = 2 * np.pi / Tp

    # Calculate the Phillips-type scaling parameter
    alpha = (
        (5 / 16)
        * (Hs**2 * peak_frequency**4 / g**2)
        / (0.065 * gamma**0.803 + 0.135)
    )

    # Pierson-Moskowitz spectrum
    spectrum = (
        alpha * g**2 / omega**5
        * np.exp(-1.25 * (peak_frequency / omega)**4)
    )

    # Width of the spectral peak
    sigma = np.where(
        omega <= peak_frequency,
        0.07,
        0.09
    )

    # Enhance the spectrum around the peak frequency
    peak_enhancement = np.exp(
        -0.5
        * ((omega - peak_frequency) /
           (sigma * peak_frequency))**2
    )

    return spectrum * gamma**peak_enhancement



#  DIRECTIONAL SPREADING


def directional_spreading(theta, mean_direction, s):
    """
    Distribute wave energy around the mean direction.

    theta           : Wave directions (radians)
    mean_direction  : Dominant wave direction (radians)
    s               : Directional spreading parameter
    """

    # Find the angular difference from the mean direction
    difference = theta - mean_direction

    # Keep angular differences within [-pi, pi]
    difference = np.arctan2(
        np.sin(difference),
        np.cos(difference)
    )

    # Waves outside +/- 90 degrees receive no energy
    spreading = np.zeros_like(theta)
    valid_directions = np.abs(difference) <= np.pi / 2

    spreading[valid_directions] = (
        np.cos(difference[valid_directions]) ** (2 * s)
    )

    # Normalise the distribution so its integral is one
    spreading /= np.trapezoid(spreading, theta)

    return spreading



# SEA-STATE PARAMETERS


Hs = 4.0                   # Significant wave height (m)
Tp = 10.0                  # Peak wave period (s)
gamma = 3.3                # JONSWAP peak enhancement factor

mean_direction = np.deg2rad(90)  # Dominant wave direction
s = 10                     # Directional spreading parameter



# FREQUENCY DISCRETISATION

N_frequency = 150

omega = np.linspace(0.1, 4.0, N_frequency)
d_omega = omega[1] - omega[0]

# Calculate the energy distribution across frequencies
S = jonswap_spectrum(omega, Hs, Tp, gamma)



# DIRECTION DISCRETISATION

N_directions = 30

theta = np.linspace(
    mean_direction - np.pi / 2,
    mean_direction + np.pi / 2,
    N_directions
)

d_theta = theta[1] - theta[0]

# Calculate how wave energy is distributed by direction
D = directional_spreading(theta, mean_direction, s)


# 2D SPATIAL GRID


Lx = 200                    # Length of the domain (m)
Ly = 200                    # Width of the domain (m)

Nx = 150                    # Grid points along X
Ny = 150                    # Grid points along Y

x = np.linspace(-Lx / 2, Lx / 2, Nx)
y = np.linspace(-Ly / 2, Ly / 2, Ny)

X, Y = np.meshgrid(x, y)

g = 9.81

# Deep-water dispersion approximation
k = omega**2 / g


# CREATES INDIVIDUAL WAVE COMPONENTS


np.random.seed(42)

wave_components = []

for i in range(N_frequency):
    for j in range(N_directions):

        # Energy contributed by this frequency-direction bin
        energy = (
            S[i]
            * D[j]
            * d_omega
            * d_theta
        )

        # Convert spectral energy into wave amplitude
        amplitude = np.sqrt(2 * energy)

        # Resolve the wave number into X and Y components
        kx = k[i] * np.cos(theta[j])
        ky = k[i] * np.sin(theta[j])

        # Assign a random starting phase to each wave
        phase = np.random.uniform(0, 2 * np.pi)

        wave_components.append(
            (amplitude, kx, ky, omega[i], phase)
        )



# ANIMATE THE SEA SURFACE

simulation_time = 100       # Total simulation duration (s)
fps = 30                    # Animation frame rate

times = np.arange(
    0,
    simulation_time,
    1 / fps
)


fig = plt.figure(figsize=(10, 8))

ax = fig.add_subplot(111, projection="3d")


def update(frame):
    """Calculate and display the sea surface at each time step."""

    t = times[frame]

    # Start with a flat surface
    elevation = np.zeros_like(X)

    # Add the contribution from every wave component
    for amplitude, kx, ky, omega_i, phase in wave_components:

        wave_phase = (
            kx * X
            + ky * Y
            - omega_i * t
            + phase
        )

        elevation += amplitude * np.cos(wave_phase)

    # Redraw the surface for the current time
    ax.clear()

    ax.plot_surface(
        X,
        Y,
        elevation,
        cmap="viridis",
        linewidth=0
    )

    # Set the display limits and labels
    ax.set_xlim(-Lx / 2, Lx / 2)
    ax.set_ylim(-Ly / 2, Ly / 2)
    ax.set_zlim(-Hs, Hs)

    ax.set_xlabel("X position (m)")
    ax.set_ylabel("Y position (m)")
    ax.set_zlabel("Wave elevation (m)")

    ax.set_title(
        f"Directional JONSWAP Sea State\n"
        f"Hs = {Hs} m | Tp = {Tp} s | "
        f"Mean direction = {np.rad2deg(mean_direction):.0f}° | "
        f"Time = {t:.1f} s"
    )


# Create and display the animation
animation = FuncAnimation(
    fig,
    update,
    frames=len(times),
    interval=1000 / fps
)

plt.show()

