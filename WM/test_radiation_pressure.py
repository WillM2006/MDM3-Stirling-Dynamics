import numpy as np
from submarine import Submarine

sub = Submarine(depth=15.0)

OMEGA = 1.0

for dof in sub.dofs:
    pressures = sub.calculate_radiation_pressure(
        omega=OMEGA,
        radiating_dof=dof
    )

    print(f"\nRadiation pressure: {dof}")
    print("-" * 40)

    for name, pressure in pressures.items():
        print(
            f"{name}: "
            f"Amplitude = {abs(pressure):.2f}, "
            f"Phase = {np.angle(pressure, deg=True):.2f}°"
        )
