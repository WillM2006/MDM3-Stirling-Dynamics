
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import capytaine as cpt

from sea import Sea


RHO, G = 1025.0, 9.81

# Sub dimensions
LENGTH = 70.0
RADIUS = 5.0
DEPTH = 15.0

# Sea conditions
OMEGAS = np.linspace(0.1, 2.0, 100)
BETAS = np.deg2rad(np.arange(0, 360, 36))

# Consts
CD_CROSS, CD_AXIAL = 1.0, 0.1 # drag contst

# Solving
dt = 0.1 # runge kutta 4 step (s)
T_MEMORY = 30.0 # how far back the radiation memory convolution looks (s)
T_RAMP = 60.0 # waves ramped in smoothly over this time, a sudden start kicks the unrestrained dofs (surge/sway/yaw) into drift




class Submarine(Sea):
    def __init__(self):

        super().__init__(omega=OMEGAS, theta=BETAS)

        self.forward_speed = 0.0 # problem.encounter_freq instead if speed is non 0

        self.centre = np.array([0.0, 0.0, -DEPTH])
        mesh = cpt.mesh_horizontal_cylinder(
            length=LENGTH, radius=RADIUS, center=(0, 0, -DEPTH),
            resolution=(4, 20, 40)
        )
        self.mass = RHO * mesh.volume # neutrally buoyant, uses mesh volume (not πr²L) so the hydrostatics balance exactly

        self.body = cpt.FloatingBody(
            mesh=mesh,
            dofs=cpt.rigid_body_dofs(rotation_center=(self.centre - np.array([0, 0, 0.2]))), # centre of rotation slightly below bodys centre
            center_of_mass=self.centre - np.array([0, 0, 0.5]), # centre of mass slightly below bodys centre
            mass=self.mass
        )
        self.dofs = list(self.body.dofs) # Surge, Sway, Heave, Roll, Pitch, Yaw

        # 6x6 numpy matrices, rows = influenced_dof, cols = radiating_dof
        self.C_hydro = self._as_matrix(self.body.compute_hydrostatic_stiffness(rho=RHO, g=G))
        self.M = self._as_matrix(self.body.compute_rigid_body_inertia(rho=RHO))

        self.F_conv_history = np.zeros(6) # radiation memory from past steps, set by convolution()

        # solve directly on the Sea's directions so only omega needs interpolating
        self.test_matrix = xr.Dataset(coords={
            "omega": OMEGAS,
            "wave_direction": self.theta,
            "radiating_dof": self.dofs,
            "forward_speed": self.forward_speed,
            "water_depth": [np.inf],
            "rho": RHO,
            "g": G
        })


    def _as_matrix(self, da):
        """xarray (influenced_dof, radiating_dof) -> 6x6 numpy array in dof order"""
        return da.sel(influenced_dof=self.dofs, radiating_dof=self.dofs).values


    def solve_hydrodynamics(self):
        """BEM solve - done ONCE, linear theory assumes small motions about the mean position"""

        # solve normally
        solver = cpt.BEMSolver()
        ds = solver.fill_dataset(self.test_matrix, self.body)

        # solve for A_inf
        lim = xr.Dataset(coords={"omega": [np.inf],
                                 "radiating_dof": self.dofs,
                                 "forward_speed": self.forward_speed,
                                 "water_depth": [np.inf],
                                 "rho": [RHO],
                                 "g": [G]})
        ds_inf = solver.fill_dataset(lim, self.body)

        self.A = ds.added_mass.sel(influenced_dof=self.dofs, radiating_dof=self.dofs).values # (omega, 6, 6)
        self.B = ds.radiation_damping.sel(influenced_dof=self.dofs, radiating_dof=self.dofs).values # (omega, 6, 6)
        self.A_inf = self._as_matrix(ds_inf.added_mass.squeeze("omega"))

        # Froude-Krylov + diffraction per 1m wave amplitude, (omega, theta, 6)
        self.F_exc = ds.excitation_force.transpose("omega", "wave_direction", "influenced_dof").sel(influenced_dof=self.dofs).values

        self.M_inv = np.linalg.inv(self.M + self.A_inf)

        # constant part of the state space system X' = A_sys X + [0, M_inv F]
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
                F[:, m, d] = (np.interp(self.w, OMEGAS, self.F_exc[:, m, d].real, left=0, right=0)
                              + 1j * np.interp(self.w, OMEGAS, self.F_exc[:, m, d].imag, left=0, right=0))

        # wave amplitudes a_nm = sqrt(2 S(f, theta) df dtheta), same as Sea.superposition
        a = np.empty((len(self.f), len(self.theta)))
        for n in range(len(self.f)):
            for m in range(len(self.theta)):
                a[n, m] = np.sqrt(2 * self.directional_frequency_spectrum(self.f[n], self.theta[m]) * self.delta_f * self.delta_theta)

        self.F_hat = F * (a * np.exp(1j * self.phi))[:, :, None] # (n, m, dof)
        self.cos_theta, self.sin_theta = np.cos(self.theta), np.sin(self.theta)


    def excitation(self, t, x, y):
        """Wave excitation force at time t, phase shifted to the sub's current horizontal position"""
        phase = np.exp(1j * (self.k[:, None] * (x * self.cos_theta + y * self.sin_theta) - self.w[:, None] * t)) # (n, m)
        ramp = 0.5 * (1 - np.cos(np.pi * t / T_RAMP)) if t < T_RAMP else 1.0
        return ramp * np.einsum("nm,nmd->d", phase, self.F_hat).real


    def build_kernel(self):
        """Radiation impulse response K(t) = (2/π) ∫ B(ω) cos(ωt) dω"""
        self.t_K = np.arange(0, T_MEMORY + dt/2, dt)
        cos_wt = np.cos(np.outer(self.t_K, OMEGAS)) # (t, omega)
        self.K = 2 / np.pi * np.trapezoid(cos_wt[:, :, None, None] * self.B[None], OMEGAS, axis=1) # (t, 6, 6)


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
        F = self.excitation(t, X[0], X[1]) + F_conv + self.drag(v)
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

        def draw(i):
            X = X_frames[i]

            # rotate (roll, pitch, yaw) then move to the sub's position
            r, p, yw = scale * X[3:6]
            Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
            Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
            Rz = np.array([[np.cos(yw), -np.sin(yw), 0], [np.sin(yw), np.cos(yw), 0], [0, 0, 1]])
            pos = self.centre + scale * X[:3]
            H = np.einsum("ij,jab->iab", Rz @ Ry @ Rx, hull) + pos[:, None, None]

            eta, _ = self.superposition(XA, YA, T_frames[i])

            ax.clear()
            ax.plot_surface(H[0], H[1], H[2], color='dimgray')
            ax.plot_surface(XA, YA, eta, cmap='Blues_r', alpha=0.45, linewidth=0)
            ax.set(xlim=(-120, 120), ylim=(-120, 120), zlim=(-30, 10), xlabel='x (m)', ylabel='y (m)', zlabel='z (m)',
                   title=f't = {T_frames[i]:.0f} s   (sub motion x{scale:g})')
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
