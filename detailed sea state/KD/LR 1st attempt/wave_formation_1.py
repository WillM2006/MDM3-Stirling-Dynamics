"""Directional-spreading lesson: isolate direction, then use full JONSWAP.

Slider = angular half-width around +y. Energy is distributed, never duplicated.
The older animations are archived in deprecated/ and are not imported here.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, RadioButtons
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


def jonswap_spectrum(omega, Hs, Tp, gamma=3.3):
    g = 9.81
    omega_p = 2 * np.pi / Tp
    sigma = np.where(omega <= omega_p, 0.07, 0.09)
    b = np.exp(-0.5 * ((omega - omega_p) / (sigma * omega_p)) ** 2)
    shape = omega**-5 * np.exp(-1.25 * (omega_p / omega) ** 4) * gamma**b
    # Normalise this finite sampled spectrum to its specified significant height.
    d_omega = omega[1] - omega[0]
    return shape * (Hs / 4) ** 2 / np.sum(shape * d_omega)


Hs, Tp, gamma = 3.0, 8.0, 3.3
N_frequency = 150
N_directions = 25
omega = np.linspace(0.2, 2.0, N_frequency)
d_omega = omega[1] - omega[0]
S = jonswap_spectrum(omega, Hs, Tp, gamma)
frequency_amplitudes = np.sqrt(2 * S * d_omega)
peak = np.argmax(S)
# Directions are centred on +y; these are angles relative to +y.
angle_offsets = np.linspace(-80, 80, N_directions)
theta = np.deg2rad(90 + angle_offsets)
centre = N_directions // 2

x = np.linspace(-100, 100, 80)
y = np.linspace(-100, 100, 80)
X, Y = np.meshgrid(x, y)
k = omega**2 / 9.81
rng = np.random.default_rng(42)
phases = rng.uniform(0, 2 * np.pi, (N_frequency, N_directions))

# Cache the spatial cosine and sine factors for responsive animation.
# cos(kx*x + ky*y - omega*t + phase)
# = cos(spatial_phase)*cos(omega*t) + sin(spatial_phase)*sin(omega*t).
spatial_phase = (
    k[:, None, None, None]
    * (np.cos(theta)[None, :, None, None] * X + np.sin(theta)[None, :, None, None] * Y)
    + phases[:, :, None, None]
).astype(np.float32)
cos_basis = np.cos(spatial_phase).reshape(N_frequency * N_directions, -1)
sin_basis = np.sin(spatial_phase).reshape(N_frequency * N_directions, -1)
del spatial_phase

fig = plt.figure(figsize=(15, 10))
layout = fig.add_gridspec(2, 6, height_ratios=[1, 2])
fan_ax = fig.add_subplot(layout[0, :2], projection="polar")
density_ax = fig.add_subplot(layout[0, 2:4])
info_ax = fig.add_subplot(layout[0, 4:])
info_ax.axis("off")
info = info_ax.text(0, 1, "", va="top", transform=info_ax.transAxes, fontsize=12)
axes = [
    fig.add_subplot(layout[1, :3], projection="3d"),
    fig.add_subplot(layout[1, 3:], projection="3d"),
]
fig.subplots_adjust(left=0.06, right=0.96, top=0.86, bottom=0.24, hspace=0.4)
slider_ax = fig.add_axes([0.17, 0.12, 0.68, 0.035])
spread_slider = Slider(
    slider_ax, "Spread ±", 0, 80, valinit=40, valstep=1, valfmt="%d°"
)
mode_ax = fig.add_axes([0.37, 0.025, 0.3, 0.075])
mode_control = RadioButtons(
    mode_ax, ["Isolate direction: one wavelength", "Full JONSWAP: 150 frequencies"]
)
fig.text(0.05, 0.045, "Space: pause/resume\nR: restart time", fontsize=10)
color_ax = fig.add_axes([0.35, 0.20, 0.3, 0.015])
fig.colorbar(
    ScalarMappable(norm=Normalize(-3, 3), cmap="viridis"),
    cax=color_ax,
    orientation="horizontal",
    label="Elevation (m) — same colour scale in both plots",
)

spread_degrees = 40.0
isolated = True
current_t = 0.0
paused = False
weights = None
amplitudes = None


def set_distribution():
    global weights, amplitudes
    # These are BIN ENERGY FRACTIONS, not a new full spectrum per direction.
    weights = np.zeros(N_directions)
    if spread_degrees == 0:
        weights[centre] = 1.0
    else:
        inside = np.abs(angle_offsets) < spread_degrees
        weights[inside] = (
            np.cos((np.pi / 2) * angle_offsets[inside] / spread_degrees) ** 2
        )
        weights /= weights.sum()
    # Directional fractions always sum to 1: spread preserves total energy.
    selected_amplitudes = frequency_amplitudes.copy()
    if isolated:
        selected_amplitudes[:] = 0
        selected_amplitudes[peak] = np.sqrt(2 * (Hs / 4) ** 2)
    amplitudes = selected_amplitudes[:, None] * np.sqrt(weights)[None, :]

    fan_ax.clear()
    fan_ax.set_theta_zero_location("E")
    fan_ax.set_thetamin(0)
    fan_ax.set_thetamax(180)
    fan_ax.set_ylim(0, 1.15)
    fan_ax.set_yticks([])
    fan_ax.set_xticks(np.deg2rad([0, 45, 90, 135, 180]))
    fan_ax.set_xticklabels(["+x", "45°", "+y", "135°", "−x"])
    for angle, weight in zip(theta, weights):
        if weight > 0:
            fan_ax.annotate(
                "",
                xy=(angle, weight / weights.max()),
                xytext=(angle, 0),
                arrowprops={"arrowstyle": "->", "color": "tab:orange", "lw": 1.6},
            )
    fan_ax.set_title("Travel directions\nRay length ∝ energy fraction", pad=20)

    density_ax.clear()
    density_ax.bar(angle_offsets, 100 * weights, width=5, color="tab:orange")
    density_ax.set_xlim(-85, 85)
    density_ax.set_ylim(0, 105)
    density_ax.set_xlabel("Angle from +y (degrees)")
    density_ax.set_ylabel("Energy in direction bin (%)")
    density_ax.set_title("Same total energy: bars always sum to 100%")
    active = np.count_nonzero(weights)
    mode = "ONE wavelength" if isolated else "FULL JONSWAP spectrum"
    info.set_text(
        f"{mode}\n\n"
        f"Dominant travel direction: +y\n"
        f"Angular support: ±{spread_degrees:g}°\n"
        f"Active direction bins: {active}\n\n"
        "0° spread: parallel, long crests\n"
        "Wider spread: crossing components\n"
        "and shorter, irregular crests\n\n"
        "A(direction) = A × √(energy fraction)"
    )


def surfaces_at(t):
    # Same physical wave equation in both views; only direction weights differ.
    selected_amplitudes = frequency_amplitudes.copy()
    if isolated:
        selected_amplitudes[:] = 0
        selected_amplitudes[peak] = np.sqrt(2 * (Hs / 4) ** 2)
    cos_t = np.cos(omega * t)
    sin_t = np.sin(omega * t)
    centre_cos = cos_basis.reshape(N_frequency, N_directions, -1)[:, centre]
    centre_sin = sin_basis.reshape(N_frequency, N_directions, -1)[:, centre]
    single = (
        (selected_amplitudes * cos_t) @ centre_cos
        + (selected_amplitudes * sin_t) @ centre_sin
    ).reshape(X.shape)
    c = (amplitudes * cos_t[:, None]).ravel().astype(np.float32)
    s = (amplitudes * sin_t[:, None]).ravel().astype(np.float32)
    spread = (c @ cos_basis + s @ sin_basis).reshape(X.shape)
    return single, spread


def update(frame):
    global current_t
    current_t = frame / 10
    single, spread = surfaces_at(current_t)
    for ax, surface, title, color in zip(
        axes,
        [single, spread],
        [
            "All energy travels along +y",
            "Same energy spread across the orange directions",
        ],
        ["tab:blue", "tab:orange"],
    ):
        ax.clear()
        ax.plot_surface(
            X,
            Y,
            surface,
            cmap="viridis",
            vmin=-3,
            vmax=3,
            linewidth=0,
            rcount=80,
            ccount=80,
        )
        ax.set_xlim(-100, 100)
        ax.set_ylim(-100, 100)
        ax.set_zlim(-6, 6)
        ax.set_zticks([-6, -3, 0, 3, 6])
        # Enlarge vertical display scale by 10x; calculated heights are unchanged.
        ax.set_box_aspect((200, 200, 120))
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Elevation (m)")
        ax.set_title(title)
    fig.suptitle(
        f"Directional spreading: share energy among travel directions | t = {current_t:.1f} s\n"
        "Vertical scale exaggerated 10× for clarity. Both plots keep identical, fixed axes."
    )


def change_spread(value):
    global spread_degrees
    spread_degrees = value
    set_distribution()
    update(round(current_t * 10))
    fig.canvas.draw_idle()


def change_mode(label):
    global isolated
    isolated = label.startswith("Isolate")
    set_distribution()
    update(round(current_t * 10))
    fig.canvas.draw_idle()


spread_slider.on_changed(change_spread)
mode_control.on_clicked(change_mode)
set_distribution()
animation = FuncAnimation(fig, update, frames=600, interval=100)


def on_key(event):
    global paused
    if event.key == " ":
        paused = not paused
        animation.pause() if paused else animation.resume()
    elif event.key == "r":
        animation.frame_seq = animation.new_frame_seq()
        update(0)
        fig.canvas.draw_idle()


fig.canvas.mpl_connect("key_press_event", on_key)
plt.show()
