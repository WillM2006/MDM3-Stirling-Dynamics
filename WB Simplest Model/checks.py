"""Run with: python checks.py. Each line says PASS or FAIL, with what disagreed."""
import traceback
import numpy as np

from waves import RegularWave, G, RHO
import parcel

A, T = 0.05, 8.0          # small, so linear theory holds
Z0 = -20.0                # parcel mean depth


def W():
    return RegularWave(A, T)


def close(a, b, tol, what):
    err = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
    assert err <= tol, f"{what}: max difference {err:.3g} exceeds {tol:.3g}"


# ---- 1 ---------------------------------------------------------------
def step1():
    w = W()
    close(w.omega, 2 * np.pi / T, 1e-9, "omega = 2*pi/T")
    close(w.k, w.omega ** 2 / G, 1e-9, "k from omega^2 = g k")
    close(w.wavelength, 2 * np.pi / w.k, 1e-6, "wavelength = 2*pi/k")
    close(w.c, w.omega / w.k, 1e-9, "phase speed c = omega/k")
    assert abs(w.wavelength - 99.9) < 0.2, f"wavelength {w.wavelength:.1f} m, expected about 99.9 m"


# ---- 2 ---------------------------------------------------------------
def step2():
    w = W()
    close(w.eta(0.0, 0.0), A, 1e-12, "crest overhead at x=0, t=0")
    close(w.eta(0.0, T / 2), -A, 1e-12, "trough overhead at t=T/2")
    t = 1.3
    close(w.eta(10.0 + w.c * t, t), w.eta(10.0, 0.0), 1e-9,
          "pattern should move in +x at speed c")


# ---- 3 ---------------------------------------------------------------
def step3():
    w = W()
    # w at z=0 should equal d(eta)/dt
    h = 1e-5
    for x, t in [(0.0, 0.3), (7.0, 2.1), (-12.0, 5.5)]:
        deta = (w.eta(x, t + h) - w.eta(x, t - h)) / (2 * h)
        close(w.w(x, 0.0, t), deta, 1e-7, f"w(z=0) vs d(eta)/dt at x={x}, t={t}")
    # incompressible: du/dx + dw/dz = 0
    h = 1e-4
    x, z, t = 3.0, -5.0, 0.7
    div = ((w.u(x + h, z, t) - w.u(x - h, z, t)) / (2 * h)
           + (w.w(x, z + h, t) - w.w(x, z - h, t)) / (2 * h))
    close(div, 0.0, 1e-8, "continuity du/dx + dw/dz")
    # decay with depth
    ratio = w.w(0.0, -5.0, 1.0) / w.w(0.0, -25.0, 1.0)
    close(ratio, np.exp(w.k * 20.0), 1e-6 * np.exp(w.k * 20.0), "w(-5)/w(-25) should be exp(k*20)")
    # at a crest the water goes forward and isn't moving vertically
    close(w.u(0.0, 0.0, 0.0), A * w.omega, 1e-12, "u at the crest should be +a*omega")
    close(w.w(0.0, 0.0, 0.0), 0.0, 1e-12, "w at the crest should be 0")


# ---- 4 ---------------------------------------------------------------
def step4():
    w = W()
    h = 1e-5
    for x, z, t in [(0.0, -20.0, 0.4), (5.0, -3.0, 2.2)]:
        dw = (w.w(x, z, t + h) - w.w(x, z, t - h)) / (2 * h)
        close(w.accel_w(x, z, t), dw, 1e-7, f"accel_w vs d(w)/dt at ({x},{z},{t})")
    close(w.accel_w(0.0, 0.0, 0.0), -A * w.omega ** 2, 1e-12, "at a crest the surface accelerates DOWN: -a*omega^2")


# ---- 5 ---------------------------------------------------------------
def step5():
    w0 = RegularWave(0.0, T)
    close(w0.pressure(0.0, -10.0, 0.0), RHO * G * 10.0, 1e-6, "still water: hydrostatic rho*g*depth")
    w = W()
    # fixed point at 20 m
    t = np.linspace(0, T, 401)
    p = w.pressure(0.0, Z0, t) - RHO * G * 20.0
    amp = RHO * G * A * np.exp(w.k * Z0)
    close(np.max(p), amp, 1e-3 * amp, "fixed-point pressure amplitude rho*g*a*exp(k z)")
    close(p[0], amp, 1e-3 * amp, "pressure is a maximum when a crest is overhead")
    # p = 0 at the real surface, z = eta
    for x, tt in [(0.0, 0.0), (4.0, 1.7), (-9.0, 3.3)]:
        pe = w.pressure(x, w.eta(x, tt), tt)
        assert abs(pe) < 0.01 * RHO * G * A, f"p at the surface z=eta is {pe:.3g} Pa, should be about 0"


# ---- 6 ---------------------------------------------------------------
def step6():
    w = W()
    t = np.linspace(0, 16, 801)
    x, z = parcel.analytic_parcel(w, Z0, t)
    r = A * np.exp(w.k * Z0)
    close(np.hypot(x, z - Z0), r, 1e-9, "path should be a circle of radius a*exp(k z0) about (0, z0)")
    close(z[0], Z0 + r, 1e-9, "parcel starts at the TOP of its circle")
    assert x[1] > 0, "just after the crest passes, the parcel should be moving forward (+x)"
    assert x[0] == 0 or abs(x[0]) < 1e-12, "parcel starts at x = 0"


# ---- 7 ---------------------------------------------------------------
def step7():
    w = W()
    t = np.linspace(0, 16, 801)
    x, z = parcel.simulate_parcel(w, Z0, t)
    xa, za = parcel.analytic_parcel(w, Z0, t)
    r = A * np.exp(w.k * Z0)
    close(x, xa, 0.05 * r, "simulated x vs analytic x")
    close(z, za, 0.05 * r, "simulated z vs analytic z")


# ---- 8 ---------------------------------------------------------------
def step8():
    w = W()
    t = np.linspace(0, 16, 801)
    x, z = parcel.simulate_parcel(w, Z0, t)
    p = w.pressure(x, z, t)
    wave_p = RHO * G * A * np.exp(w.k * Z0)
    on_parcel = p.max() - p.min()
    fixed = w.pressure(0.0, Z0, t)
    fixed_pp = fixed.max() - fixed.min()
    close(fixed_pp, 2 * wave_p, 0.01 * wave_p, "fixed-point peak-to-peak should be 2*rho*g*a*exp(k z)")
    assert on_parcel < 0.05 * fixed_pp, (
        f"pressure on the parcel varies by {on_parcel:.3g} Pa vs {fixed_pp:.3g} Pa at a fixed point, should be nearly flat")


# ---- 9 ---------------------------------------------------------------
def step9():
    w = W()
    t = np.linspace(0, 16, 1601)
    x, z = parcel.simulate_parcel(w, Z0, t)
    acc = parcel.acceleration_along_path(w, x, z, t)
    expected = w.accel_w(0.0, Z0, t)
    amp = A * w.omega ** 2 * np.exp(w.k * Z0)
    close(acc[2:-2], expected[2:-2], 0.05 * amp, "parcel acceleration vs -a*omega^2*exp(k z0)*cos(omega t)")


CHECKS = [
    ("1", "dispersion relation", step1),
    ("2", "surface elevation", step2),
    ("3", "water velocity", step3),
    ("4", "water acceleration", step4),
    ("5", "pressure", step5),
    ("6", "parcel path (paper)", step6),
    ("7", "parcel path (numerical)", step7),
    ("8", "pressure on the parcel", step8),
    ("9", "acceleration of the parcel", step9),
]


def main():
    counts = {"PASS": 0, "FAIL": 0, "TODO": 0, "ERROR": 0}
    for tag, name, fn in CHECKS:
        try:
            fn()
            status, msg = "PASS", ""
        except (NotImplementedError, TypeError) as e:
            status, msg = "TODO", str(e)[:90]
        except AssertionError as e:
            status, msg = "FAIL", str(e)
        except Exception:
            status, msg = "ERROR", traceback.format_exc(limit=2).strip().splitlines()[-1]
        counts[status] += 1
        print(f"{tag}  {name:<28} {status}  {msg}")
    print("\n" + "  ".join(f"{k}: {v}" for k, v in counts.items()))


if __name__ == "__main__":
    main()
