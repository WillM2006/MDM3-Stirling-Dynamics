
import numpy as np
import matplotlib.pyplot as plt
import capytaine as cpt
from capytaine.bem.airy_waves import airy_waves_pressure

from submarine import Submarine, RHO, G, LENGTH, RADIUS

# =====================================================
# 1. Define test wave and submarine
# =====================================================

OMEGA = 1.0
WAVE_DIRECTION = 0.0

sub = Submarine(depth=15.0)


# All six pressure sensor positions in global coordinates
sensor_positions = {
    "P1": np.array([+LENGTH / 2 - 5.0, 0.0, -sub.depth - RADIUS]),
    "P2": np.array([-LENGTH / 2 + 5.0, 0.0, -sub.depth - RADIUS]),
    "P3": np.array([0.0, +RADIUS, -sub.depth]),
    "P4": np.array([0.0, -RADIUS, -sub.depth]),
    "P5": np.array([0.0, 0.0, -sub.depth + RADIUS]),
    "P6": np.array([0.0, 0.0, -sub.depth - RADIUS]),
}


# =====================================================
# 2. Solve diffraction problem using Capytaine
# =====================================================

solver = cpt.BEMSolver()

problem = cpt.DiffractionProblem(
    body=sub.body,
    omega=OMEGA,
    wave_direction=WAVE_DIRECTION,
    water_depth=np.inf,
    rho=RHO,
    g=G,
    forward_speed=0.0
)

result = solver.solve(problem, keep_details=True)



# Evaluate diffraction pressure at all six sensors
sensor_names = list(sensor_positions.keys())
points = np.array(list(sensor_positions.values()))

p_diff_all = np.asarray(
    solver.compute_pressure(points, result)
).reshape(-1)

# Extract P1 for the existing plot
p1 = sensor_positions["P1"]
p_diff = p_diff_all[0]

print("\nDiffraction pressure at all sensors")
print("------------------------------------")

for name, pressure in zip(sensor_names, p_diff_all):
    print(
        f"{name}: "
        f"Amplitude = {abs(pressure):.2f} Pa, "
        f"Phase = {np.angle(pressure, deg=True):.2f} degrees"
    )


# Diffraction-pressure complex amplitude
p_diff = np.asarray(
    solver.compute_pressure(p1, result)
).item()

# =====================================================
# 3. Calculate incident-wave pressure
# =====================================================

# Deep-water dispersion relation
k = OMEGA**2 / G

# Unit-amplitude wave travelling in +x direction
p_inc = (
    RHO * G
    * np.exp(k * p1[2])
    * np.exp(1j * k * p1[0])
)

# Incident pressure using Capytaine's convention
p_inc_capytaine = np.asarray(
    airy_waves_pressure(
        p1.reshape(1, 3),
        problem
    )
).item()

print("\nIncident pressure comparison")
print("Our formula:", p_inc)
print("Capytaine:", p_inc_capytaine)
print("Difference:", abs(p_inc - p_inc_capytaine))

# Combined pressure (incident + diffraction)
p_total = p_inc + p_diff

# =====================================================
# 4. Print numerical results
# =====================================================

print("\nPressure at sensor P1")
print("---------------------")
print("Position:", p1)

print("\nIncident pressure:")
print("Amplitude:", abs(p_inc), "Pa")

print("\nDiffraction pressure:")
print("Amplitude:", abs(p_diff), "Pa")
print("Phase:", np.angle(p_diff, deg=True), "degrees")

print("\nCombined pressure:")
print("Amplitude:", abs(p_total), "Pa")

# =====================================================
# 5. Reconstruct time-domain pressure
# =====================================================

t = np.linspace(0, 20, 1000)

p_inc_t = np.real(p_inc * np.exp(-1j * OMEGA * t))
p_diff_t = np.real(p_diff * np.exp(-1j * OMEGA * t))
p_total_t = np.real(p_total * np.exp(-1j * OMEGA * t))

# =====================================================
# 6. Plot pressure contributions
# =====================================================

plt.figure(figsize=(10, 5))

plt.plot(t, p_inc_t, label="Incident pressure")
plt.plot(t, p_diff_t, label="Diffraction pressure")
plt.plot(
    t,
    p_total_t,
    label="Incident + diffraction",
    linewidth=2
)

plt.xlabel("Time (s)")
plt.ylabel("Pressure fluctuation (Pa)")
plt.title("Incident and diffraction pressure at P1")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

