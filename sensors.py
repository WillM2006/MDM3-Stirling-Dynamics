"""
forward model: simulated hull motion -> what the sensors would read.

Sign conventions (same as capytaine / submarine.py):
    x forward, y to port, z up, origin on the mean free surface
    roll phi about x, pitch theta about y (positive = bow down), yaw psi about z
    beta is the wave direction here (called theta in sea.py), so theta is free for pitch

"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.signal import welch

import submarine
from submarine import Submarine, RHO, G, LENGTH, RADIUS, DEPTH, T_RAMP
from submarine_extra import hull_surface, clean_3d, T_SIM, T_SETTLE

dt = submarine.dt                 # simulation step 
DEG = 180 / np.pi
rng = np.random.default_rng(1)    # set random seed to ensure repeatable results
P_ATM = 101325.0                  # atmospheric pressure in Pascals

# sensor positions are given by r = (r_x, r_y, r_z) measured from O (centre of rotation)
# centre = (0, 0, -DEPTH), rotation_center = centre - (0, 0, 0.2) 
O_EARTH = np.array([0.0, 0.0, -DEPTH - 0.2])   # defining O on earth scale
Z_AXIS = 0.2                                    # hull axis/ z axis is 0.2 m above O ( in earth frame )

#defining imu position
R_IMU = np.array([5.0, 0.0, Z_AXIS]) # z component means lies on hull axis

#defining pressyre sensor positions
X_END = LENGTH / 2 - 5.0                        # 5 m in from each end, easier to change when moving to a different sub shape
PRESSURE_SENSORS = {
    "P1":       np.array([+X_END, 0.0, Z_AXIS - RADIUS]),   # 5 metres in from each end on the bottom of the hull
    "P2":     np.array([-X_END, 0.0, Z_AXIS - RADIUS]),
    "P3":       np.array([0.0, +RADIUS, Z_AXIS]),      # half way along the hull on each side half way up
    "P4":  np.array([0.0, -RADIUS, Z_AXIS]),
    "P5":        np.array([0.0, 0.0, Z_AXIS + RADIUS]),   # top and bottom of hull half way along    
    "P6":       np.array([0.0, 0.0, Z_AXIS - RADIUS]),
}

#defining depth gauge position
R_DEPTH = np.array([-10.0, 0.0, Z_AXIS + RADIUS])

# adding noise to the sensors to take them from perfect physics based data to more accurate sensor data, all have a random timing offset with sd of 5 ms.

# different noise used for different sensors based on their typical real world errors:

# gyro and accelerometer: bias wander, constant bias, scale factor and misalingment, quantisation, white noise

# attitude heading reference system ( pitch and roll ): white noise, drift

# depth gauge: bias, drift, quantisation, white noise ( modelled as a seventh pressure sensor )

# pressure sensors: bias, drift, quantisation, white noise

FS = dict(gyro=100.0,       # IMU (Hz)
          accel=100.0,      # vertical acceleration 50 - 100 Hz, chose top of the range for all
          attitude=50.0,    # pitch / roll 10 - 50 Hz
          depth=10.0,       # depth rate 1 - 10 Hz
          pressure=10.0)    # hull pressure sensors -> modelled as same as depth gauge so 10 Hz chosen

GYRO_ERR = dict(density=np.deg2rad(0.002 / 60),   # angle random walk 0.002 deg/sqrt(h) -> rad/s/sqrt(Hz) , amount of noise in each hertz i.e. higher frqeuency more noise 
                bias=np.deg2rad(0.01 / 3600),      # constant turn-on bias, 0.01 deg/h (rad/s)
                bias_rw=np.deg2rad(1e-4 / 3600),   # slow bias wander (rad/s per sqrt(s)), gets worse over time
                scale=50e-6,                       # scale factor error (fraction)
                misalign=50e-6,                    # axis misalignment (rad)
                quant=1e-8)                        # quantisation step (rad/s)

ACC_ERR = dict(density=50e-6 * G,                  # 50 micro-g / sqrt(Hz)  (m/s^2/sqrt(Hz))
               bias=100e-6 * G,                    # 100 micro-g turn-on bias (m/s^2)
               bias_rw=1e-6 * G,                   # bias wander (m/s^2 per sqrt(s))
               scale=100e-6,
               misalign=50e-6,
               quant=1e-5)                         # (m/s^2)

ATT_ERR = dict(density=np.deg2rad(0.001),          # white part of the AHRS output, rad/sqrt(Hz)
               drift=np.deg2rad(0.02),             # slow wandering error (first-order Markov) std (rad)
               tau=300.0)                          # its correlation time (s)

PRES_ERR = dict(full_scale=2e6,                    # 20 bar maximum (Pa)
                density=5.0,                       # Pa/sqrt(Hz)
                bias=1e-4 * 2e6,                   # 0.01 % of full scale constant offset (Pa)
                drift=20.0, tau=900.0,             # slow thermal drift (Pa), correlation time (s)
                quant=1e-6 * 2e6)                  # 2 Pa

TIMING_JITTER = 0.005   # each sensor stream gets a fixed random clock offset of this order (s), matters when fusing streams

# setting orientation of submarine (facing north , latitude of 50 degrees)
# cannot discern between rotation of sub and earth rotation so is added to readings
OMEGA_EARTH, LAT = 7.292e-5, np.deg2rad(50.0)
EARTH_RATE = np.array([OMEGA_EARTH * np.cos(LAT), 0.0, OMEGA_EARTH * np.sin(LAT)])   # (rad/s) body x, y, z


#  depth / heading controller
#  surge, sway, heave and yaw have no restoring force on a submerged body, so they drift without a controller.
#  roll and pitch already have hydrostatic stiffness (centre of mass below centre of buoyancy).
#  control_force = - K_p * x - K_d * velocity on the 4 DOFs without a restoring force, a mass-spring-damper with period T_CONTROL.
#  control period >> wave period so it stops slow drift but has minimal effect on wave motion, shortening causes stiffer resistance
   

CONTROL_DOFS = [0, 1, 2, 5]     # surge, sway, heave, yaw ( indexing from previous list of 6 DOFs )
T_CONTROL = 60.0                # period in seconds 
ZETA_CONTROL = 0.7              # damping ratio
COMPARE_CONTROLLER = True       # run without for comparison


class ControlledSubmarine(Submarine):
    """
    submarine plus the controller force ( controller force allow for the sub to maintain depth and heading over long periods,giving easy readings )
    """
    control_on = True

    def build_controller(self):
        """
        K_p = m_eff w_c^2,  (srping stiffness)
        
        K_d = 2 zeta m_eff w_c (damping coefficient)
        
        m_eff = mass + infinite-frequency added mass of each DOF
        """
        m_eff = np.diag(self.M + self.A_inf) # self.A_inf is the added mass at infinite frequency ( from submarine.py )
        
        w_c = 2 * np.pi / T_CONTROL # w_c is natural frequency of the controller
        
        self.Kp, self.Kd = np.zeros(6), np.zeros(6)
        
        self.Kp[CONTROL_DOFS] = m_eff[CONTROL_DOFS] * w_c**2
        
        self.Kd[CONTROL_DOFS] = 2 * ZETA_CONTROL * m_eff[CONTROL_DOFS] * w_c

    def dXdt(self, t, X):
        dX = super().dXdt(t, X)                       
        if self.control_on:
            F_ctrl = -self.Kp * X[:6] - self.Kd * X[6:]
            dX[6:] += self.M_inv @ F_ctrl             # same M_inv as the other forces
        return dX  # velocities and accelerations for one time step, integrated at the end to find total 



#  true physical quanitites at each sensor (on the simulation grid)

def sensor_displacement(X6, r):
    """
    Small-angle displacement of a hull point at offset r from O in earth dimensions
    
    delta_P = xi + Theta x r  
    
    Theta x r = (theta r_z - psi r_y,  psi r_x - phi r_z,  phi r_y - theta r_x)
    
    X6 = (N, 6) history of (x, y, z, phi, theta, psi). Returns dx, dy, dz, for each row (N) -> i.e. each time step 
    """
    x, y, z, roll, pitch, yaw = X6.T
    r_x, r_y, r_z = r

    # rotation part of the displacement, Theta x r
    rot_x = pitch * r_z - yaw * r_y
    rot_y = yaw * r_x - roll * r_z
    rot_z = roll * r_y - pitch * r_x

    return x + rot_x, y + rot_y, z + rot_z


def ramp(t):
    """
    Same start-up ramp as Submarine.excitation, gradual increase in waves rather than all at once
    - would cause a shock to the hull and the controller if it were all at once, creating bad data; first 300 
    seconds of the simulation are collected but not used in plots, purely to allow the sub to settle into a steady state.
    """
    return np.where(t < T_RAMP, 0.5 * (1 - np.cos(np.pi * t / T_RAMP)), 1.0)


def incident_pressure(sub, wave_amps, x_sensor, y_sensor, z_mean, t, chunk=1000):
    """
    Incident wave pressure at a point, according tolinear deep-water theory .
    p_I = rho g sum_nm a_nm e^(k_n z) cos(k_n (x cos beta_m + y sin beta_m) - w_n t + eps_nm)

    x_sensor, y_sensor (N,)
    z_mean: sensor's mean depth over time i.e. resting depth (using the moving depth only matters at 2nd order, so we use the mean depth here)
    written as Re[ sum c_nm e^(i(k.x - w t)) ] with c_nm = a_nm e^(k z_mean) e^(i eps).
    """
    
    # complex amplitude of each wave component at the sensor depth including random phase 
    depth_decay = np.exp(sub.k[:, None] * z_mean)
    phase_offset = np.exp(1j * sub.phi)
    coeff = wave_amps * depth_decay * phase_offset            # (n, m)

    pressure = np.empty(len(t))
    for start in range(0, len(t), chunk):
        window = slice(start, start + chunk)
        x_w = x_sensor[window, None, None]
        y_w = y_sensor[window, None, None]
        t_w = t[window, None, None]

        k_dot_x = sub.k[None, :, None] * (x_w * sub.cos_theta + y_w * sub.sin_theta)   # (t, n, m)
        phase = k_dot_x - sub.w[None, :, None] * t_w
        pressure[window] = np.real(np.sum(coeff * np.exp(1j * phase), axis=(1, 2)))

    return RHO * G * ramp(t) * pressure


def pressure_sensor(sub, wave_amps, t, X6, r):
    """
    Pressure at a sensor, only froude-kylov (linear pressure) no radiation or diffraction.
    Delta p = -rho g dz_P  +  p_I(mean_pos, t)
    first term: sensor moving up/down in the static pressure gradient
    second term: incident wave pressure at the sensor
    If the sensor moved exactly with the water these two cancel
    """
    dx, dy, dz = sensor_displacement(X6, r)
    mean_pos = O_EARTH + r                                    # mean position of the sensor in earth dimensions

    hydro = -RHO * G * dz
    incident = incident_pressure(sub, wave_amps, mean_pos[0] + dx, mean_pos[1] + dy, mean_pos[2], t)
    total = hydro + incident

    return dict(total=total, hydro=hydro, incident=incident, dz=dz, mean_depth=-mean_pos[2])


def true_signals(sub, wave_amps, t, X_hist):
    """
    Everything each sensor would read if it were perfect (i.e. no noise or bias), on the model's time grid.
    """
    pose, velocity = X_hist[:, :6], X_hist[:, 6:]
    # only x and x_dot are stored, so accelerations are calculated using central difference from the velocity history.
    # central difference means using the time step neighbours on both sides
    # motion is slow (< 0.32 Hz) compared with the time step dt = 0.1 s so the error is tiny
    accel = np.gradient(velocity, dt, axis=0)

    roll, pitch = pose[:, 3], pose[:, 4]  # pose is displacement and orientation
    roll_acc, pitch_acc, yaw_acc = accel[:, 3], accel[:, 4], accel[:, 5]
    r_x, r_y, r_z = R_IMU
    signals = {}

    # gyros = body angular rate + earth rotation
    signals["gyro"] = velocity[:, 3:6] + EARTH_RATE                         # columns 3 - 5 in rad/s

    # accelerometer specific force at the IMU 
    #   f_x = x_dd + theta_dd r_z - psi_dd r_y - g theta      (tilt of sub means gravity effetcs horizontal forces)
    #   f_y = y_dd + psi_dd r_x - phi_dd r_z   + g phi
    #   f_z = z_dd + phi_dd r_y - theta_dd r_x + g            (vertical: gravity coupling only 2nd order)
    # centripetal omega x (omega x r) dropped, rates around 0.01 rad/s ; second order so can be ignored
    
    lever_x = pitch_acc * r_z - yaw_acc * r_y                 # lever arm terms, alpha x r i.e. offset from the centre (rigid body rotation) 
    lever_y = yaw_acc * r_x - roll_acc * r_z
    lever_z = roll_acc * r_y - pitch_acc * r_x

    f_x = accel[:, 0] + lever_x - G * pitch
    f_y = accel[:, 1] + lever_y + G * roll
    f_z = accel[:, 2] + lever_z + G
    signals["accel"] = np.column_stack([f_x, f_y, f_z])                     # (N, 3) m/s^2
    signals["fx_parts"] = dict(translation=accel[:, 0], lever=lever_x, tilt=-G * pitch)

    # true attitude (AHRS output = this + its own errors, added later)
    signals["roll"], signals["pitch"] = roll, pitch

    #  the six hull pressure sensors 
    signals["pressure"] = {name: pressure_sensor(sub, wave_amps, t, pose, r) for name, r in PRESSURE_SENSORS.items()}

    # depth gauge = one more pressure port, depth = (p - p_atm) / (rho g)
    depth_gauge = pressure_sensor(sub, wave_amps, t, pose, R_DEPTH)
    signals["depth_gauge"] = depth_gauge
    signals["depth_geom"] = depth_gauge["mean_depth"] - depth_gauge["dz"]                        # where the sensor really is (m below surface)
    signals["depth_from_p"] = depth_gauge["mean_depth"] + depth_gauge["total"] / (RHO * G)      # what its pressure says (includes waves)
    signals["depth_rate_geom"] = np.gradient(signals["depth_geom"], dt)
    return signals


# adding noise to create realistic data

def resample(t_sim, y_sim, fs):
    """
    Cubic spline used to interpolate the 10 Hz simulation grid to the higher frequency sensors 
    Signal has nothing above about 0.3 Hz so this is essentially exact.
    Each dataset gets a fixed clock offset: sensor reports time t but actually sampled at t + offset.
    """
    offset = TIMING_JITTER * rng.standard_normal()
    t_s = np.arange(abs(offset), t_sim[-1] - abs(offset), 1 / fs)
    return t_s, CubicSpline(t_sim, y_sim, axis=0)(t_s + offset)


def white(density, fs, n):
    """white noise with one-sided density n"""
    return density * np.sqrt(fs / 2) * rng.standard_normal(n)


def random_walk(density, fs, n):
    """slowly wandering bias: integrated white noise, density in units per sqrt(s)"""
    return np.cumsum(density * np.sqrt(1 / fs) * rng.standard_normal(n))


def gauss_markov(sigma, tau, fs, n):
    """first-order Markov process: wanders with sd sigma, forgets itself over tau seconds therefore bringing it back to the mean"""
    phi = np.exp(-1 / (fs * tau))
    x = np.empty(n)
    x[0] = sigma * rng.standard_normal()
    w = sigma * np.sqrt(1 - phi**2) * rng.standard_normal(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + w[i]
    return x


def quantise(x, step): 
    return step * np.round(x / step) if step > 0 else x


def triad_measure(true, fs, e):
    """
    Sensor error model for a 3-axis gyro or accelerometer:
    meas = (I + scale + misalignment) true + bias + bias wander + white noise, then quantised
    """
    n = len(true)
    S = np.diag(e["scale"] * rng.standard_normal(3))                          # each axis reads a bit large/small
    Mis = e["misalign"] * rng.standard_normal((3, 3))                         # axes not quite orthogonal / aligned
    np.fill_diagonal(Mis, 0)
    meas = true @ (np.eye(3) + S + Mis).T
    meas = meas + e["bias"] * rng.standard_normal(3)
    meas = meas + np.column_stack([random_walk(e["bias_rw"], fs, n) for _ in range(3)])
    meas = meas + np.column_stack([white(e["density"], fs, n) for _ in range(3)])
    return quantise(meas, e["quant"])


def pressure_measure(t_sim, port, fs):
    """
    Error model for a pressure sensor. Quantisation acts on the absolute pressure the gauge sees
    (p_atm + hydrostatic + fluctuation), so we build that, corrupt it, then remove the known mean again.
    """
    t_s, dp = resample(t_sim, port["total"], fs)
    n = len(t_s)
    p_mean = P_ATM + RHO * G * port["mean_depth"]
    p_abs = p_mean + dp + PRES_ERR["bias"] * rng.standard_normal() \
        + gauss_markov(PRES_ERR["drift"], PRES_ERR["tau"], fs, n) + white(PRES_ERR["density"], fs, n)
    p_abs = quantise(p_abs, PRES_ERR["quant"])
    return t_s, p_abs - p_mean, p_abs                      # fluctuation (Pa), absolute reading (Pa)


def measure_all(t, true):
    """
    Turn every true signal into a sensor reading at that sensor's own rate.
    """
    m = {}
    # raw IMU triads
    m["t_gyro"], gyro_true = resample(t, true["gyro"], FS["gyro"])
    m["gyro"] = triad_measure(gyro_true, FS["gyro"], GYRO_ERR)
    m["t_accel"], accel_true = resample(t, true["accel"], FS["accel"])
    m["accel"] = triad_measure(accel_true, FS["accel"], ACC_ERR)
    m["vert_acc"] = m["accel"][:, 2] - G                    # vertical acceleration = f_z - g

    # AHRS pitch / roll = truth + slow wandering error + white noise
    for key in ("pitch", "roll"):
        m["t_att"], y = resample(t, true[key], FS["attitude"])
        n = len(y)
        m[key] = y + gauss_markov(ATT_ERR["drift"], ATT_ERR["tau"], FS["attitude"], n) + white(ATT_ERR["density"], FS["attitude"], n)

    # hull pressure sensors 
    m["pressure"] = {}
    for name, port in true["pressure"].items():
        m["t_pres"], m["pressure"][name], _ = pressure_measure(t, port, FS["pressure"])

    # depth gauge -> depth -> depth rate (central difference of the measured depth, as a real system would)
    m["t_depth"], _, p_abs = pressure_measure(t, true["depth_gauge"], FS["depth"])
    m["depth"] = (p_abs - P_ATM) / (RHO * G)
    m["depth_rate"] = np.gradient(m["depth"], 1 / FS["depth"])
    return m


#  Plotting helpers

def psd(y, fs, seg_seconds=200.0):
    """
    One-sided power spectral density (Welch's Method)
    
    y is signal, fs is sample rate (Hz), seg_seconds is length of each section used.

    Averages over many overlapping sections to returns frequencies and their power density.
    - 200s chunk length gives 0.005 Hz resolution which is enough to see wave peak (big chunk means less averaging though meaning more noise).
    """
    return welch(y, fs=fs, nperseg=min(len(y), int(seg_seconds * fs)))


if __name__ == "__main__":

    #  1. Simulation (submarine.py + the controller)
    sub = ControlledSubmarine()
    sub.solve_hydrodynamics()
    sub.build_excitation()
    sub.build_kernel()
    sub.build_controller()
    S_f, D_sp, S_sea, wave_amps, Hs = sub.sea_spectrum()      # wave_amps = wave amplitudes a_nm, same as used in the excitation
    print(f"Sea: Hs = {Hs:.2f} m")

    sub.control_on = True
    t, X_hist = sub.simulate(duration=T_SIM)
    print("simulation (controller ON) done")
    if COMPARE_CONTROLLER:
        sub.control_on = False
        _, X_free = sub.simulate(duration=T_SIM)
        print("simulation (controller OFF) done")

    #  2 - 4. True signals -> sensor readings
    true = true_signals(sub, wave_amps, t, X_hist)
    meas = measure_all(t, true)

    # save the synthetic data set for the estimator (fluctuations about the mean, SI units)
    # one csv per sensor stream because each stream has its own sample frequency and time column
    os.makedirs("sensor_data", exist_ok=True)

    def save_csv(name, columns, header):
        np.savetxt(f"sensor_data/{name}.csv", np.column_stack(columns), fmt="%.10g", delimiter=",", header=header, comments="")

    save_csv("gyro", [meas["t_gyro"], meas["gyro"]], "t_s,gyro_x_rad_s,gyro_y_rad_s,gyro_z_rad_s")
    save_csv("accel", [meas["t_accel"], meas["accel"], meas["vert_acc"]],
             "t_s,f_x_m_s2,f_y_m_s2,f_z_m_s2,vert_acc_m_s2")
    save_csv("attitude", [meas["t_att"], meas["pitch"], meas["roll"]], "t_s,pitch_rad,roll_rad")
    save_csv("depth", [meas["t_depth"], meas["depth"], meas["depth_rate"]], "t_s,depth_m,depth_rate_m_s")
    pressure_names = [name.replace(" ", "_") for name in PRESSURE_SENSORS]
    save_csv("pressure", [meas["t_pres"], np.column_stack(list(meas["pressure"].values()))],
             "t_s," + ",".join(f"{name}_Pa" for name in pressure_names))
    save_csv("truth", [t, X_hist],
             "t_s,x_m,y_m,z_m,phi_rad,theta_rad,psi_rad,x_dot,y_dot,z_dot,phi_dot,theta_dot,psi_dot")

    # sensor positions (body axes, m from O) and the sea state used
    with open("sensor_data/sensor_info.csv", "w") as f:
        f.write("name,r_x_m,r_y_m,r_z_m\n")
        for name, r in {**PRESSURE_SENSORS, "IMU": R_IMU, "depth gauge": R_DEPTH}.items():
            f.write(f"{name.replace(' ', '_')},{r[0]},{r[1]},{r[2]}\n")
        f.write(f"Hs_m,{Hs},,\n")
    print("saved sensor_data/*.csv")

    # plotting windows: 2 minutes of time series after the start-up, spectra over everything after the start-up
    T0, T1 = T_SETTLE, T_SETTLE + 120.0
    def w_(tt):
        return (tt >= T0) & (tt <= T1)
    def s_(tt):
        return tt >= T_SETTLE
    TRUE_KW = dict(color="k", lw=1.2, label="true")
    MEAS_KW = dict(color="tab:orange", lw=0.6, alpha=0.8)

    # Figure 1: where the sensors are
    fig = plt.figure(figsize=(11, 6))
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(*hull_surface(np.zeros(6), 1), color="lightgrey", alpha=0.35, linewidth=0)
    for name, r in PRESSURE_SENSORS.items():
        P = O_EARTH + r
        ax.scatter(*P, color="tab:blue", s=50)
        ax.text(P[0], P[1], P[2] + 1.5, name, fontsize=8)
    for P, lbl, col in ((O_EARTH + R_IMU, "IMU", "tab:red"), (O_EARTH + R_DEPTH, "depth gauge", "tab:green"),
                        (O_EARTH, "O (ref. point)", "k")):
        ax.scatter(*P, color=col, s=60, marker="^")
        ax.text(P[0], P[1], P[2] - 2.5, lbl, fontsize=8, color=col)
    ax.set(xlabel="x (m, forward)", ylabel="y (m, port)", zlabel="z (m)", title="Sensor layout on the hull")
    ax.set_box_aspect((1, 0.3, 0.3))
    clean_3d(ax)

    # Figure 2: the four outputs from your list (pitch, roll, vertical acceleration, depth rate)
    fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    for ax_, key, tm, lab in ((axs[0], "pitch", meas["t_att"], f"Pitch (deg), {FS['attitude']:.0f} Hz"),
                              (axs[1], "roll", meas["t_att"], f"Roll (deg), {FS['attitude']:.0f} Hz")):
        ax_.plot(tm[w_(tm)], meas[key][w_(tm)] * DEG, label="sensor", **MEAS_KW)
        ax_.plot(t[w_(t)], true[key][w_(t)] * DEG, **TRUE_KW)
        ax_.set_ylabel(lab)
    tm = meas["t_accel"]
    axs[2].plot(tm[w_(tm)], meas["vert_acc"][w_(tm)], label="sensor", **MEAS_KW)
    axs[2].plot(t[w_(t)], true["accel"][w_(t), 2] - G, **TRUE_KW)
    axs[2].set_ylabel(f"Vertical acc. (m/s²), {FS['accel']:.0f} Hz")
    tm = meas["t_depth"]
    axs[3].plot(tm[w_(tm)], meas["depth_rate"][w_(tm)], label="sensor (from pressure)", **MEAS_KW)
    axs[3].plot(t[w_(t)], true["depth_rate_geom"][w_(t)], **TRUE_KW)
    axs[3].set_ylabel(f"Depth rate (m/s), {FS['depth']:.0f} Hz")
    axs[3].set_xlabel("t (s)")
    for ax_ in axs:
        ax_.legend(loc="upper right", fontsize=8)
    fig.suptitle("Listed sensor outputs: true quantity vs sensor reading (2-minute window)\n"
                 "depth rate 'true' = how fast the gauge really moves; the sensor also sees wave pressure")
    fig.tight_layout()

    # Figure 3: raw IMU triads
    fig, axs = plt.subplots(3, 2, figsize=(14, 8), sharex=True)
    tg, ta = meas["t_gyro"], meas["t_accel"]
    for i, ax_name in enumerate("xyz"):
        axs[i, 0].plot(tg[w_(tg)], meas["gyro"][w_(tg), i] * DEG, label="sensor", **MEAS_KW)
        axs[i, 0].plot(t[w_(t)], true["gyro"][w_(t), i] * DEG, **TRUE_KW)
        axs[i, 0].set_ylabel(f"gyro {ax_name} (deg/s)")
        off = G if i == 2 else 0.0
        axs[i, 1].plot(ta[w_(ta)], meas["accel"][w_(ta), i] - off, label="sensor", **MEAS_KW)
        axs[i, 1].plot(t[w_(t)], true["accel"][w_(t), i] - off, **TRUE_KW)
        axs[i, 1].set_ylabel(f"f_{ax_name}{' - g' if i == 2 else ''} (m/s²)")
    axs[0, 0].legend(fontsize=8)
    axs[-1, 0].set_xlabel("t (s)")
    axs[-1, 1].set_xlabel("t (s)")
    axs[0, 0].set_title("Gyroscopes (body angular rate)")
    axs[0, 1].set_title(f"Accelerometers (specific force at IMU, r = {R_IMU.tolist()} m)")
    fig.tight_layout()

    # Figure 4: why the horizontal accelerometers are contaminated by tilt
    fig, ax_ = plt.subplots(figsize=(12, 4))
    for key, col in (("translation", "tab:blue"), ("lever", "tab:green"), ("tilt", "tab:red")):
        ax_.plot(t[w_(t)], true["fx_parts"][key][w_(t)], color=col, lw=1,
                 label={"translation": "surge acceleration x_dd", "lever": "lever arm theta_dd r_z - psi_dd r_y",
                        "tilt": "gravity / tilt term -g theta"}[key])
    ax_.plot(t[w_(t)], true["accel"][w_(t), 0], "k", lw=1.5, label="f_x = sum (what the sensor sees)")
    ax_.set(xlabel="t (s)", ylabel="m/s²", title="Forward accelerometer f_x broken into its parts")
    ax_.legend(fontsize=8)
    fig.tight_layout()

    # Figure 5: pressure ports, with the cancellation made visible
    fig, axs = plt.subplots(len(PRESSURE_SENSORS), 1, figsize=(12, 13), sharex=True)
    tp = meas["t_pres"]
    for ax_, (name, port) in zip(axs, true["pressure"].items()):
        ax_.plot(t[w_(t)], port["incident"][w_(t)], color="tab:blue", lw=0.8, label="incident wave p_I")
        ax_.plot(t[w_(t)], port["hydro"][w_(t)], color="tab:red", lw=0.8, label="-rho g dz (port moving)")
        ax_.plot(tp[w_(tp)], meas["pressure"][name][w_(tp)], ".", color="tab:orange", ms=2, label="sensor reading")
        ax_.plot(t[w_(t)], port["total"][w_(t)], "k", lw=1.2, label="true total Delta p")
        ax_.set_ylabel(f"{name}\n(Pa)", fontsize=8)
    axs[0].legend(loc="upper right", fontsize=7, ncol=4)
    axs[-1].set_xlabel("t (s)")
    fig.suptitle(f"Hull pressure, Level 1 (Froude-Krylov), {FS['pressure']:.0f} Hz\n"
                 "blue and red largely cancel when the hull moves with the water")
    fig.tight_layout()

    # Figure 6: what the depth gauge really reports
    fig, ax_ = plt.subplots(figsize=(12, 4))
    tm = meas["t_depth"]
    ax_.plot(tm[w_(tm)], meas["depth"][w_(tm)], color="tab:orange", lw=0.8, label="depth gauge reading")
    ax_.plot(t[w_(t)], true["depth_from_p"][w_(t)], "k--", lw=1, label="noise-free (p - p_atm)/(rho g)")
    ax_.plot(t[w_(t)], true["depth_geom"][w_(t)], "tab:blue", lw=1.2, label="real depth of the gauge")
    ax_.invert_yaxis()
    ax_.set(xlabel="t (s)", ylabel="depth (m)", title="A depth gauge measures pressure, so it sees the waves too")
    ax_.legend(fontsize=8)
    fig.tight_layout()

    # Figure 7: spectra, signal vs noise floor
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    f_band = np.logspace(-3, 2, 400)
    panels = [
        ("Pitch", meas["t_att"], meas["pitch"] * DEG, true["pitch"] * DEG, FS["attitude"], ATT_ERR["density"] * DEG, "deg²/Hz"),
        ("Roll", meas["t_att"], meas["roll"] * DEG, true["roll"] * DEG, FS["attitude"], ATT_ERR["density"] * DEG, "deg²/Hz"),
        ("Vertical acceleration", meas["t_accel"], meas["vert_acc"], true["accel"][:, 2] - G, FS["accel"], ACC_ERR["density"], "(m/s²)²/Hz"),
        ("Depth rate", meas["t_depth"], meas["depth_rate"], true["depth_rate_geom"], FS["depth"], None, "(m/s)²/Hz"),
        ("Gyro y (pitch rate)", meas["t_gyro"], meas["gyro"][:, 1] * DEG, true["gyro"][:, 1] * DEG, FS["gyro"], GYRO_ERR["density"] * DEG, "(deg/s)²/Hz"),
        ("Pressure P5 (top)", meas["t_pres"], meas["pressure"]["P5"], true["pressure"]["P5"]["total"],
         FS["pressure"], PRES_ERR["density"], "Pa²/Hz"),
    ]
    for ax_, (title, tm, ym, yt, fs, dens, unit) in zip(axs.ravel(), panels):
        fm, Pm = psd(ym[s_(tm)], fs)
        ft, Pt = psd(yt[s_(t)], 1 / dt)
        ax_.loglog(fm[1:], Pm[1:], color="tab:orange", lw=0.8, label="sensor")
        ax_.loglog(ft[1:], Pt[1:], "k", lw=1.2, label="true")
        if dens is not None:                                      # white noise density n -> one-sided PSD n²
            ax_.axhline(dens**2, color="grey", ls="--", label="white-noise floor n²")
        else:                                                     # depth rate: noise of a differentiated pressure
            n_depth = PRES_ERR["density"] / (RHO * G)
            fb = f_band[f_band < fs / 2]
            ax_.loglog(fb, (n_depth * fs * np.sin(2 * np.pi * fb / fs))**2, color="grey", ls="--",
                       label="differentiated pressure noise")
        ax_.axvspan(0.05, 0.32, color="tab:blue", alpha=0.08, label="wave band")
        ax_.set(title=title, xlabel="f (Hz)", ylabel=unit, xlim=(1e-3, fs / 2))
    axs[0, 0].legend(fontsize=7)
    fig.suptitle("Sensor spectra. Everything above ~0.3 Hz is noise: high rates buy averaging, not information")
    fig.tight_layout()

    # Figure 8: controller on vs off
    if COMPARE_CONTROLLER:
        true_free = true_signals(sub, wave_amps, t, X_free)
        fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))
        for X_, lbl, col in ((X_free, "controller OFF", "tab:red"), (X_hist, "controller ON", "tab:blue")):
            axs[0].plot(t, X_[:, 0], color=col, lw=1, label=f"surge, {lbl}")
            axs[0].plot(t, X_[:, 1], color=col, lw=1, ls="--", label=f"sway, {lbl}")
        axs[0].set(xlabel="t (s)", ylabel="m", title="Slow drift of the unrestrained DOFs")
        axs[0].legend(fontsize=7)
        for tr, X_, lbl, col in ((true_free, X_free, "OFF", "tab:red"), (true, X_hist, "ON", "tab:blue")):
            f_, P_ = psd(X_[s_(t), 2], 1 / dt)
            axs[1].loglog(f_[1:], P_[1:], color=col, label=f"heave, controller {lbl}")
            port = tr["pressure"]["P5"]
            for key, ls in (("total", "-"), ("incident", ":")):
                f_, P_ = psd(port[key][s_(t)], 1 / dt)
                axs[2].loglog(f_[1:], P_[1:], color=col, ls=ls,
                              label=f"{'Delta p (sensor sees)' if key == 'total' else 'p_I (incident only)'}, {lbl}")
        axs[1].set(xlabel="f (Hz)", ylabel="m²/Hz", title="Heave spectrum", xlim=(1e-3, 0.5))
        axs[2].set(xlabel="f (Hz)", ylabel="Pa²/Hz", title="P5 pressure: sensor signal vs incident wave",
                   xlim=(1e-3, 0.5))
        axs[1].legend(fontsize=7)
        axs[2].legend(fontsize=7)
        fig.suptitle(f"Controller with period {T_CONTROL:.0f} s. "
                     "Gap between solid and dotted = how much of the wave pressure the hull's motion cancels")
        fig.tight_layout()

    plt.show()


#  Noise-free versions of the code:

# First;  Line 280 : inside resample()
# Replace  `offset = TIMING_JITTER * rng.standard_normal()`  with: offset = 0.0

# Second;  Lines 285 - 307 : the noise helper functions white(), random_walk(), gauss_markov() and quantise()
# Comment out all four (or leave them in, nothing calls them once the functions below are swapped)

# Third;  Lines 310 - 323 : triad_measure()
# Replace the whole function with:
#
# def triad_measure(true, fs, e):
#     """returns the true triad unchanged"""
#     return true.copy()

# Fourth;  Lines 326 - 337 : pressure_measure()
# Replace the whole function with:
#
# def pressure_measure(t_sim, port, fs):
#     """no bias / drift / white noise / quantisation, still resamples and returns (t, fluctuation, absolute reading)"""
#     t_s, dp = resample(t_sim, port["total"], fs)
#     p_mean = P_ATM + RHO * G * port["mean_depth"]
#     p_abs = p_mean + dp
#     return t_s, p_abs - p_mean, p_abs                      # fluctuation (Pa), absolute reading (Pa)

# Fifth;  Lines 355 - 356 : inside measure_all(), in the pitch / roll loop
# Replace  `n = len(y)`  and  `m[key] = y + gauss_markov(...) + white(...)`  with: m[key] = y

# Sixth;  Lines 552 - 558 : Figure 7 noise-floor lines
# Comment out the whole  `if dens is not None: ... else: ...`  block, it sits just before  `ax_.axvspan(...)`

# Line numbers are for the file as it is now, so do these in reverse order (Sixth first) to stop them moving

#  Noise-free data saved as csv files too:

# Insert after the line  `print("saved sensor_data/*.csv")`  (Line 431), inside the __main__ block

#    # noise-free copy of the data: every error is set to zero, the sensors are measured again and saved
#    # into sensor_data_noise_free/, then the error settings are put back so the plots still show the noisy data
#    noisy_settings = (GYRO_ERR, ACC_ERR, ATT_ERR, PRES_ERR, TIMING_JITTER)
#
#    def zero_errors(e):
#        return {k: (v if k in ("tau", "full_scale") else 0.0) for k, v in e.items()}    # tau and full_scale are not errors
#
#    GYRO_ERR, ACC_ERR, ATT_ERR, PRES_ERR, TIMING_JITTER = zero_errors(GYRO_ERR), zero_errors(ACC_ERR), zero_errors(ATT_ERR), zero_errors(PRES_ERR), 0.0
#    clean = measure_all(t, true)
#    GYRO_ERR, ACC_ERR, ATT_ERR, PRES_ERR, TIMING_JITTER = noisy_settings
#
#    os.makedirs("sensor_data_noise_free", exist_ok=True)
#
#    def save_clean_csv(name, columns, header):
#        np.savetxt(f"sensor_data_noise_free/{name}.csv", np.column_stack(columns), fmt="%.10g", delimiter=",", header=header, comments="")
#
#    save_clean_csv("gyro", [clean["t_gyro"], clean["gyro"]], "t_s,gyro_x_rad_s,gyro_y_rad_s,gyro_z_rad_s")
#    save_clean_csv("accel", [clean["t_accel"], clean["accel"], clean["vert_acc"]], "t_s,f_x_m_s2,f_y_m_s2,f_z_m_s2,vert_acc_m_s2")
#    save_clean_csv("attitude", [clean["t_att"], clean["pitch"], clean["roll"]], "t_s,pitch_rad,roll_rad")
#    save_clean_csv("depth", [clean["t_depth"], clean["depth"], clean["depth_rate"]], "t_s,depth_m,depth_rate_m_s")
#    save_clean_csv("pressure", [clean["t_pres"], np.column_stack(list(clean["pressure"].values()))],
#                   "t_s," + ",".join(f"{name}_Pa" for name in pressure_names))
#    print("saved sensor_data_noise_free/*.csv")
