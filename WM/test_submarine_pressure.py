from submarine import Submarine
import numpy as np

sub = Submarine(depth=15.0)

pressures = sub.calculate_diffraction_pressure(
    omega=1.0,
    wave_direction=0.0
)

print("\nDiffraction pressure from Submarine class")
print("-----------------------------------------")

for name, pressure in pressures.items():
    print(
        f"{name}: "
        f"{abs(pressure):.2f} Pa, "
        f"phase = {np.angle(pressure, deg=True):.2f}°"
    )
