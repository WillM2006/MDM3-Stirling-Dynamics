"""Compare one spectral sample with a full sampled JONSWAP spectrum.

Both surfaces travel along +y and are uniform across x. The first 10 seconds
assemble the spectrum at frozen wave time; then both surfaces animate together.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# Same JONSWAP function as wave_formation_1.py.
def jonswap_spectrum(omega, Hs, Tp, gamma=3.3):
    g = 9.81
    omega_p = 2 * np.pi / Tp
    alpha = (
        (5 / 16) * (Hs**2 * omega_p**4 / g**2) * (1 / (0.065 * gamma**0.803 + 0.135))
    )
    S_pm = (alpha * g**2 / omega**5) * np.exp(-1.25 * (omega_p / omega) ** 4)
    sigma = np.where(omega <= omega_p, 0.07, 0.09)
    b = np.exp(-0.5 * ((omega - omega_p) / (sigma * omega_p)) ** 2)
    return S_pm * gamma**b


Hs = 4.0
Tp = 10.0
gamma = 3.3
mean_direction = np.deg2rad(90)
N_frequency = 150
N_directions = 1

# A finite sampling of the full spectrum, using the original frequency range.
omega = np.linspace(0.1, 4.0, N_frequency)
d_omega = omega[1] - omega[0]
S = jonswap_spectrum(omega, Hs, Tp, gamma)

# Select the strongest sampled frequency for the single-wave comparison.
selected = np.argmax(S)
# Start at the dominant frequency, then add nearby frequencies outwards.
assembly_order = np.argsort(np.abs(omega - omega[selected]), kind="stable")

# All energy travels in one direction; there is no directional spreading here.
theta = np.array([mean_direction])
D = np.array([1.0])
d_theta = 1.0

x = np.linspace(-100, 100, 40)
y = np.linspace(-100, 100, 500)
X, Y = np.meshgrid(x, y)
g = 9.81
k = omega**2 / g

# Same spectrum -> energy -> amplitude -> wave component logic.
np.random.seed(42)
elevation_components = []
for i in range(N_frequency):
    for j in range(N_directions):
        energy = S[i] * D[j] * d_omega * d_theta
        amplitude = np.sqrt(2 * energy)
        kx = k[i] * np.cos(theta[j])
        ky = k[i] * np.sin(theta[j])
        phase = np.random.uniform(0, 2 * np.pi)
        elevation_components.append((amplitude, kx, ky, omega[i], phase))

fps = 10
simulation_time = 70
times = np.arange(0, simulation_time, 1 / fps)
# Fixed viewing range shared by both surfaces throughout the animation.
z_limit = 2.0 * Hs

fig = plt.figure(figsize=(15, 10))
layout = fig.add_gridspec(2, 2, height_ratios=[1, 2.2])
spectrum_ax = fig.add_subplot(layout[0, :])
single_ax = fig.add_subplot(layout[1, 0], projection="3d")
full_ax = fig.add_subplot(layout[1, 1], projection="3d")
fig.subplots_adjust(left=0.07, right=0.96, top=0.87, bottom=0.13, hspace=0.38)
fig.suptitle(
    "One JONSWAP spectrum: one sample versus all 150 samples\nBoth surfaces travel along +y; both are uniform across x"
)

spectrum_ax.plot(omega, S, color="0.25", label="Full spectral energy density S(ω)")
(included_points,) = spectrum_ax.plot(
    [], [], ".", color="tab:orange", label="Samples included in the right-hand sum"
)
spectrum_ax.plot(
    omega[selected],
    S[selected],
    "o",
    color="tab:blue",
    markersize=8,
    label="Single sample used on the left",
)
spectrum_ax.set_xlim(0.1, 4.0)
spectrum_ax.set_ylim(0, 1.15 * np.max(S))
spectrum_ax.set_xlabel("Angular frequency ω (rad/s)")
spectrum_ax.set_ylabel("Energy density S(ω)\n[m² / (rad/s)]")
spectrum_ax.legend(loc="upper right")
spectrum_ax.set_title(
    "The spectrum specifies energy across frequencies; it is not a surface-height plot."
)

readout = fig.text(0.5, 0.075, "", ha="center", fontsize=11)
fig.text(0.5, 0.03, "Space: pause/resume  |  R: replay frequency assembly", ha="center")


def update(frame):
    elapsed = times[frame]
    t = max(0.0, elapsed - 10.0)
    # First hold one component, then progressively assemble the full spectrum.
    fraction = np.clip((elapsed - 3.0) / 7.0, 0.0, 1.0)
    count = 1 + int(fraction * (N_frequency - 1))
    included = assembly_order[:count]
    included_points.set_data(omega[included], S[included])

    single_elevation = None
    elevation = np.zeros_like(X)
    for i, (amplitude, kx, ky, w, phase) in enumerate(elevation_components):
        component = amplitude * np.cos(kx * X + ky * Y - w * t + phase)
        if i == selected:
            single_elevation = component
        if i in included:
            elevation += component

    for ax, surface, title, color in [
        (
            single_ax,
            single_elevation,
            "ONE SAMPLE → ONE sinusoidal component",
            "tab:blue",
        ),
        (
            full_ax,
            elevation,
            f"{count} SAMPLES → SUM of {count} components",
            "tab:orange",
        ),
    ]:
        ax.clear()
        ax.plot_surface(X, Y, surface, color=color, linewidth=0, rcount=250, ccount=10)
        ax.set_xlim(-100, 100)
        ax.set_ylim(-100, 100)
        ax.set_zlim(-z_limit, z_limit)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Elevation (m)")
        ax.set_title(title)
        ax.view_init(elev=25, azim=-55)

    a = elevation_components[selected][0]
    readout.set_text(
        f"Selected sample: ω = {omega[selected]:.3f} rad/s, S = {S[selected]:.2f}\n"
        f"Bin energy = S × Δω = {S[selected] * d_omega:.3f} m²  →  amplitude = √(2 × energy) = {a:.2f} m"
        f"   |   wave time = {t:.1f} s"
    )


animation = FuncAnimation(fig, update, frames=len(times), interval=1000 / fps)
paused = False


def on_key(event):
    global paused
    if event.key == " ":
        paused = not paused
        animation.pause() if paused else animation.resume()
    elif event.key == "r":
        animation.frame_seq = animation.new_frame_seq()
        update(0)
        fig.canvas.draw_idle()
        animation.resume()
        paused = False


fig.canvas.mpl_connect("key_press_event", on_key)
plt.show()
