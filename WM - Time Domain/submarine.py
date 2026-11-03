import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import capytaine as cpt

from sea import Sea

# mesh.volume comes out at 5408 m³ but πr²L = 5498 m³ ----------------------------------------------------------------------------


# Sub conditions
LENGTH = 70.0
RADIUS = 5.0
DEPTH = 15.0
FORWARD_SPEED = 1.0 # problem.encounter_freq instead if speed is non 0

# Sea conditions
N_FREQ = 40
N_DIR = 36
OMEGA_RANGE = (0.1, 2.0)
DIRECTION = np.pi/4 # heading / mean direction of waves (also angle between wave direction and sub direction)

# Consts
RHO, G = 1025.0, 9.81
CD_CROSS, CD_AXIAL = 1.0, 0.1 # drag contst

# Solving
dt = 0.1 # runge kutta 4 step (s)
T_MEMORY = 30.0 # how far back the radiation memory convolution looks (s)
T_RAMP = 60.0 # waves ramped in smoothly over this time, a sudden start kicks the unrestrained dofs (surge/sway/yaw) into drift


class Submarine(Sea):
    def __init__(
                self, 
                
                # sea params
                n_freq=40, 
                n_dir=36, 
                omega_range=(0.1, 2.0),
                Hs=1.7,
                Tp=5.9, 
                direction=0.0, 
                gamma=3.3, 
                s=10, 
                seed=42,

                # sub params
                depth=15.0,
                forward_speed=0.0,
                ):

        super().__init__(n_freq=n_freq, n_dir=n_dir, omega_range=omega_range, Hs=Hs, Tp=Tp, direction=direction, gamma=gamma, s=s, seed=seed,)
        
        self.depth = depth
        self.forward_speed = forward_speed
        self.centre = np.array([0.0, 0.0, -depth])

        self.mesh = self.create_mesh(length=LENGTH, radius=RADIUS)
        self.mass = RHO * self.mesh.volume 
        self.body = cpt.FloatingBody(
            mesh=self.mesh,
            dofs=cpt.rigid_body_dofs(rotation_center=(self.centre - np.array([0, 0, 0.2]))), # centre of rotation slightly below bodys centre
            center_of_mass=self.centre - np.array([0, 0, 0.5]), # centre of mass slightly below bodys centre
            mass=self.mass
        )
        self.dofs = list(self.body.dofs) # Surge, Sway, Heave, Roll, Pitch, Yaw

        # 6x6 numpy matrices, rows = influenced_dof, cols = radiating_dof
        self.C_hydro = self.as_matrix(self.body.compute_hydrostatic_stiffness(rho=RHO, g=G))
        self.M = self.as_matrix(self.body.compute_rigid_body_inertia(rho=RHO))

        self.F_conv_history = np.zeros(6) # radiation memory from past steps, set by convolution()

        self.bem_omegas = np.linspace(*omega_range, n_freq) # BEM frequency grid (rad/s), evenly spaced for solving hydrodynamics

        self.params = dict(n_freq=n_freq, n_dir=n_dir, omega_range=omega_range, Hs=Hs, Tp=Tp,
                   direction=direction, gamma=gamma, s=s, seed=seed,
                   depth=depth, forward_speed=forward_speed)



    def create_mesh(self, length, radius, n_cap=8, n_mid=20, n_around=20):
        # sausage shape mesh / 680 panels
        half = length / 2 - radius # half-length of the straight part

        # outline as (r, z): back cap, straight middle, front cap
        phi = np.linspace(-np.pi / 2, 0, n_cap) # -90° (tip) to 0° (where cap meets cylinder)
        back = np.stack([radius * np.cos(phi), -half + radius * np.sin(phi)], axis=1)
        mid = np.stack([np.full(n_mid - 1, radius), np.linspace(-half, half, n_mid + 1)[1:-1]], axis=1)
        front = np.stack([radius * np.cos(phi[::-1]), half - radius * np.sin(phi[::-1])], axis=1)
        rz = np.vstack([back, mid, front])

        profile = np.stack([rz[:, 0], np.zeros(len(rz)), rz[:, 1]], axis=1) # (x, y, z) points, spun about z
        full = (cpt.RotationSymmetricMesh.from_profile_points(profile, n_around)
        .rotated_y(np.pi / 2) # lay it along x
        .translated_z(-self.depth))
        full = cpt.Mesh(full.vertices, full.faces)
        half = full.extract_faces(np.where(full.faces_centers[:, 1] > 0)[0]) # port side only (y > 0)
        return cpt.ReflectionSymmetricMesh(half, plane='xOz') # mirrored across y = 0



    def as_matrix(self, da):
        """xarray (influenced_dof, radiating_dof) -> 6x6 numpy array in dof order"""
        return da.sel(influenced_dof=self.dofs, radiating_dof=self.dofs).values


    def solve_hydrodynamics(self):
        """BEM solve - done ONCE, linear theory assumes small motions about the mean position"""
        solver = cpt.BEMSolver()
        base = dict(forward_speed=self.forward_speed, water_depth=[np.inf], rho=RHO, g=G)

        # radiation (A, B): beam direction, so encounter freq = omega (k U cos 90° = 0), one solve per omega
        rad = solver.fill_dataset(xr.Dataset(coords=dict(omega=self.bem_omegas, wave_direction=[np.pi / 2],
                                                         radiating_dof=self.dofs, **base)), self.body)

        # diffraction + Froude-Krylov (excitation): every wave direction, no radiation problems
        dif = solver.fill_dataset(xr.Dataset(coords=dict(omega=self.bem_omegas, wave_direction=self.theta, **base)), self.body)

        # A_inf: capytaine only supports omega=inf at zero speed
        inf = solver.fill_dataset(xr.Dataset(coords=dict(omega=[np.inf], radiating_dof=self.dofs, forward_speed=0.0,
                                                         water_depth=[np.inf], rho=RHO, g=G)), self.body)

        self.A = self.as_matrix(rad.added_mass) # (omega, 6, 6)
        self.B = self.as_matrix(rad.radiation_damping) # (omega, 6, 6)
        self.A_inf = self.as_matrix(inf.added_mass.squeeze("omega"))

        # Froude-Krylov + diffraction per 1m wave amplitude, (omega, theta, 6)
        self.F_exc = dif.excitation_force.transpose("omega", "wave_direction", "influenced_dof").sel(influenced_dof=self.dofs).values


        # constant part of the state space system X' = A_sys X + [0, M_inv F]
        self.M_inv = np.linalg.inv(self.M + self.A_inf)
        Z, I = np.zeros((6, 6)), np.eye(6)
        self.A_sys = np.block([[Z, I], [-self.M_inv @ self.C_hydro, Z]])


    def build_excitation(self):
        """Combine the BEM excitation with the Sea's wave amplitudes and phases
           F_exc(t) = Re[ sum_nm F(w_n, theta_m) * a_nm * e^(i(k_n . x - w_n t + phi_nm)) ]
           (capytaine uses e^(-iwt) time dependence, matching cos(k.x - wt + phi) in sea.py)"""

        self.w = 2 * np.pi * self.f # Sea frequencies in rad/s

        # interpolate the BEM excitation onto the Sea frequencies, 0 outside OMEGAS (decayed with depth)
        F = np.zeros((len(self.w), len(self.theta), 6), dtype=complex)
        for m in range(len(self.theta)):
            for d in range(6):
                F[:, m, d] = (np.interp(self.w, self.bem_omegas, self.F_exc[:, m, d].real, left=0, right=0)
                              + 1j * np.interp(self.w, self.bem_omegas, self.F_exc[:, m, d].imag, left=0, right=0))

        self.F_hat = F * (self.wave_amplitudes * np.exp(1j * self.phi))[:, :, None] # (n, m, dof)
        self.cos_theta, self.sin_theta = np.cos(self.theta), np.sin(self.theta)


    def excitation(self, t, x, y):
        """Wave excitation force at time t, phase shifted to the sub's current horizontal position"""
        phase = np.exp(1j * (self.k[:, None] * (x * self.cos_theta + y * self.sin_theta) - self.w[:, None] * t)) # (n, m)
        ramp = 0.5 * (1 - np.cos(np.pi * t / T_RAMP)) if t < T_RAMP else 1.0
        return ramp * np.einsum("nm,nmd->d", phase, self.F_hat).real


    def build_kernel(self):
        """Radiation impulse response K(t) = (2/π) ∫ B(ω) cos(ωt) dω"""
        self.t_K = np.arange(0, T_MEMORY + dt/2, dt)
        cos_wt = np.cos(np.outer(self.t_K, self.bem_omegas)) # (t, omega)
        self.K = 2 / np.pi * np.trapezoid(cos_wt[:, :, None, None] * self.B[None], self.bem_omegas, axis=1) # (t, 6, 6)


    def convolution(self, v_history):
        """F_conv(t) = -∫ K(s) v(t - s) ds, trapezoid rule, v_history[-1] is the current velocity
           only the PAST part (s >= dt) is done here, once per step - the s = 0 term
           -dt/2 K(0) v(t) uses each RK4 stage's own velocity in dXdt"""
        n = min(len(v_history), len(self.K))
        v_recent = v_history[::-1][:n] # v(t), v(t - dt), v(t - 2dt) ...
        w = np.ones(n)
        w[0] = 0 # s = 0 handled in dXdt
        w[-1] = 0.5 if n > 1 else 0 # trapezoid end weight
        self.F_conv_history = -dt * np.einsum("jab,j,jb->a", self.K[:n], w, v_recent)
        return self.F_conv_history


    def drag(self, v):
        """DRAG ONLY AROUND POSITION NOT ROTATIONS
           USE HULL VELOCITY NOT VELOCITY RELATIVE TO WAVES"""
        F_drag = np.zeros(6)
        F_drag[0] = -0.5 * RHO * CD_AXIAL * np.pi * RADIUS**2 * abs(v[0]) * v[0]
        F_drag[1] = -0.5 * RHO * CD_CROSS * 2 * RADIUS * LENGTH * abs(v[1]) * v[1]
        F_drag[2] = -0.5 * RHO * CD_CROSS * 2 * RADIUS * LENGTH * abs(v[2]) * v[2]

        return F_drag


    def dXdt(self, t, X):

        v = X[6:]
        F_conv = self.F_conv_history - 0.5 * dt * self.K[0] @ v # past memory + instantaneous s = 0 term
        F = self.excitation(t, X[0] + self.forward_speed * t, X[1]) + F_conv + self.drag(v)
        B = np.concatenate([np.zeros(6), self.M_inv @ F])
        return self.A_sys @ X + B


    def RK4(self, t, X):
        
        k1 = self.dXdt(t, X)
        k2 = self.dXdt(t + dt/2, X + dt/2 * k1)
        k3 = self.dXdt(t + dt/2, X + dt/2 * k2)
        k4 = self.dXdt(t + dt, X + dt * k3)

        return X + dt/6 * (k1 + 2*k2 + 2*k3 + k4)


    def simulate(self, duration, X0=None):
        """Step the sub forward n times, returns times (N,) and states (N, 12)"""
        n_steps = int(round(duration / dt))
        T = np.arange(n_steps + 1) * dt
        X_hist = np.zeros((n_steps + 1, 12))
        if X0 is not None:
            X_hist[0] = X0

        for n in range(n_steps):
            self.convolution(X_hist[:n + 1, 6:])
            X_hist[n + 1] = self.RK4(T[n], X_hist[n])

        return T, X_hist


    def plot_motion(self, T, X_hist):
        
        labels = ["Surge (m)", "Sway (m)", "Heave (m)", "Roll (deg)", "Pitch (deg)", "Yaw (deg)"]
        fig, axes = plt.subplots(6, 1, sharex=True, figsize=(9, 10))
        for d, ax in enumerate(axes):
            y = X_hist[:, d] if d < 3 else np.rad2deg(X_hist[:, d])
            ax.plot(T, y)
            ax.set_ylabel(labels[d])
        axes[-1].set_xlabel("t (s)")
        fig.tight_layout()
        plt.show()

    


    def animate(self, T, X_hist, frame_every=1.0, scale=1.0):
        from matplotlib import animation

        step = max(1, int(round(frame_every / dt)))
        T_frames, X_frames = T[::step], X_hist[::step]

        # sub hull surface in its own frame (centred at 0)
        xs = np.linspace(-LENGTH/2, LENGTH/2, 15)
        th = np.linspace(0, 2*np.pi, 20)
        Xs, Th = np.meshgrid(xs, th)
        hull = np.stack([Xs, RADIUS * np.cos(Th), RADIUS * np.sin(Th)]) # (3, 20, 15)

        # sea surface grid around the sub
        xa = np.linspace(-120, 120, 100)
        XA, YA = np.meshgrid(xa, xa)

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(projection='3d')

        def rotation_matrix(roll, pitch, yaw):
            """Body -> earth rotation (yaw, then pitch, then roll)."""
            cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
            return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                            [-sp, cp * sr, cp * cr]])


        def draw(i):
            X = X_frames[i]
            x_travel = self.forward_speed * T_frames[i] # real distance travelled, not exaggerated
            R = rotation_matrix(*(scale * X[3:6]))
            pos = self.centre + scale * X[:3] + [x_travel, 0, 0]

            # rotate about the sub's centre: move to origin, rotate, move to new position
            hull = self.body.mesh.translated(-self.centre).rotated_with_matrix(R).translated(pos)

            # sea grid follows the sub
            XS = XA + x_travel
            eta = self.superposition(XS, YA, T_frames[i])

            ax.clear()
            hull.show_matplotlib(ax=ax, facecolor='dimgray', edgecolor='k', linewidth=0.2,
                                 bounding_box=((x_travel - 120, x_travel + 120), (-120, 120), (-30, 10)))
            ax.plot_surface(XS, YA, eta, cmap='Blues_r', alpha=0.45, linewidth=0)
            ax.set(xlabel='x (m)', ylabel='y (m)', zlabel='z (m)',
                   title=f't = {T_frames[i]:.0f} s   x = {x_travel:.0f} m   (sub motion x{scale:g})')
            ax.set_box_aspect((1, 1, 0.35))



        anim = animation.FuncAnimation(fig, draw, frames=len(T_frames), interval=100)
        plt.show()
        return anim # keep a reference or the animation stops

    


if __name__ == "__main__":
    sub = Submarine()
    sub.solve_hydrodynamics()
    sub.build_excitation()
    sub.build_kernel()

    T, X_hist = sub.simulate(duration=600.0)
    sub.plot_motion(T, X_hist)
    sub.animate(T, X_hist, frame_every=2.0, scale=10.0) # motion x10, it's only ~0.3 deg otherwise
