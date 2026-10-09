import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# JONSWAP spectrum
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


# Two full wave systems. Change these parameters independently as we build up.
# Each tuple is (significant wave height [m], peak period [s], gamma, direction [deg]).
sea_states = [(4.0, 10.0, 3.3, 90.0), (4.0, 10.0, 3.3, 0.0)]
N_frequency = 150  # Per spectrum: 300 components in total.
omega_min = 0.1  # rad/s
omega_max = 4.0  # rad/s

# Divide the frequency range into bins and sample at each bin centre.
omega_edges = np.linspace(omega_min, omega_max, N_frequency + 1)
d_omega = omega_edges[1] - omega_edges[0]
omega = (omega_edges[:-1] + omega_edges[1:]) / 2
spectra = []
for Hs, Tp, gamma, direction in sea_states:
    S = jonswap_spectrum(omega, Hs, Tp, gamma)
    # Preserve the shape, but ensure m0 = sum(S * d_omega) = (Hs / 4)**2.
    S *= (Hs / 4) ** 2 / np.sum(S * d_omega)
    spectra.append(S)

# Surface grid. Both plots use exactly the same coordinates and axes.
x = np.linspace(-100, 100, 240)
y = np.linspace(-100, 100, 240)
X, Y = np.meshgrid(x, y)
g = 9.81
k = omega**2 / g

# Spectrum -> bin energy -> amplitude -> sinusoidal wave components.
np.random.seed(42)
component_groups = []
for S, (Hs, Tp, gamma, direction) in zip(spectra, sea_states):
    theta = np.deg2rad(direction)
    components = []
    for i in range(N_frequency):
        energy = S[i] * d_omega
        amplitude = np.sqrt(2 * energy)
        kx = k[i] * np.cos(theta)
        ky = k[i] * np.sin(theta)
        phase = np.random.uniform(0, 2 * np.pi)
        components.append((amplitude, kx, ky, omega[i], phase))
    component_groups.append(components)

fps = 10
simulation_time = 60
times = np.arange(0, simulation_time, 1 / fps)
z_limit = 8.0  # Fixed in all three panels; never changes during animation.
fig = plt.figure(figsize=(17, 9))
layout = fig.add_gridspec(2, 3, height_ratios=[1, 2.5])
spectrum_axes = [fig.add_subplot(layout[0, j]) for j in range(2)]
summary_ax = fig.add_subplot(layout[0, 2])
summary_ax.axis("off")
axes = [fig.add_subplot(layout[1, j], projection="3d") for j in range(3)]
fig.subplots_adjust(
    left=0.055, right=0.95, top=0.87, bottom=0.09, wspace=0.3, hspace=0.45
)
colors = ["tab:blue", "tab:orange"]
spectrum_limit = 1.15 * max(np.max(S) for S in spectra)
for j, (ax, S, color, state) in enumerate(
    zip(spectrum_axes, spectra, colors, sea_states)
):
    Hs, Tp, gamma, direction = state
    ax.plot(omega, S, color=color)
    ax.plot(omega, S, ".", color=color, markersize=3)
    ax.set_xlim(omega_min, omega_max)
    ax.set_ylim(0, spectrum_limit)
    ax.set_xlabel("Angular frequency ω (rad/s)")
    ax.set_ylabel("S(ω) [m² / (rad/s)]")
    ax.set_title(
        f"Spectrum {j + 1}: Hs={Hs:g} m, Tp={Tp:g} s, direction={direction:g}°"
    )
summary_ax.text(
    0.05,
    0.85,
    f"{N_frequency} bins per spectrum\n"
    f"{2 * N_frequency} sinusoidal components in the sum\n\n"
    "Each bin: E = S(ω) × Δω\n"
    "Component amplitude: A = √(2E)\n\n"
    "All bins are used; amplitudes vary.",
    transform=summary_ax.transAxes,
    va="top",
    fontsize=12,
)
fig.text(0.5, 0.025, "Space: pause/resume  |  R: restart time", ha="center")


def update(frame):
    t = times[frame]
    surfaces = []
    for components in component_groups:
        elevation = np.zeros_like(X)
        for amplitude, kx, ky, w, phase in components:
            elevation += amplitude * np.cos(kx * X + ky * Y - w * t + phase)
        surfaces.append(elevation)
    total = surfaces[0] + surfaces[1]
    titles = [
        "Spectrum 1 → full surface travelling along +y",
        "Spectrum 2 → full surface travelling along +x",
        "Superposition: surface 1 + surface 2",
    ]
    for ax, surface, title, color in zip(
        axes, surfaces + [total], titles, colors + [None]
    ):
        ax.clear()
        if color is None:
            ax.plot_surface(
                X,
                Y,
                surface,
                cmap="viridis",
                vmin=-z_limit,
                vmax=z_limit,
                linewidth=0,
                rcount=120,
                ccount=120,
            )
        else:
            ax.plot_surface(
                X, Y, surface, color=color, linewidth=0, rcount=120, ccount=120
            )
        ax.set_xlim(-100, 100)
        ax.set_ylim(-100, 100)
        ax.set_zlim(-z_limit, z_limit)
        # Equal physical scale: one metre has the same size on x, y and z.
        ax.set_box_aspect((200, 200, 2 * z_limit))
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Elevation (m)")
        ax.set_title(title, fontsize=10)
    fig.suptitle(
        f"Two complete sampled JONSWAP spectra, perpendicular directions | t = {t:.1f} s"
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
