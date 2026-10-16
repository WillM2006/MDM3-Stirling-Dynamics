"""Replaceable 6-DOF linear hydrodynamics; no Capytaine dependency.

The temporary capsule strip approximation is an explicit sprint surrogate,
not WB's missing hull implementation or a validated real-submarine model.
"""
from dataclasses import dataclass
import numpy as np

RHO, G = 1025., 9.81
DOFS = ("Surge", "Sway", "Heave", "Roll", "Pitch", "Yaw")


def skew(r):
    x, y, z = np.asarray(r)
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def compute_rao(sub, omega, A, B, F):
    """WM's layout and equation: F[frequency,direction,dof]."""
    omega = np.asarray(omega)
    Z = -omega[:, None, None]**2 * (sub.M[None] + A) - 1j*omega[:, None, None]*B + sub.C_hydro[None]
    return np.linalg.solve(Z, F.transpose(0, 2, 1)).transpose(0, 2, 1)


@dataclass(frozen=True)
class Hull:
    length: float = 70.0  # TOTAL tip-to-tip length, not barrel length
    radius: float = 5.0
    depth: float = 15.0  # reference origin = geometric centre, z positive up
    bg: float = 0.5
    damping_rate: float = 0.08  # B = rate * M_eff; illustrative, in 1/s
    strips: int = 400


class Submarine:
    def __init__(self, *, omega, theta, hull=None, build_response=True):
        self.omega, self.theta = np.asarray(omega, float), np.asarray(theta, float)
        self.hull = hull or Hull()
        h = self.hull
        if h.length <= 2*h.radius or h.depth <= h.radius or h.bg < 0 or h.damping_rate < 0 or h.strips < 20:
            raise ValueError("Need submerged capsule, nonnegative BG/damping, >=20 strips")
        self.depth, self.rho, self.g = h.depth, RHO, G
        self.dofs = list(DOFS)
        self.reference = "geometric hull centre; axes +x along hull, +z upward"
        dx = h.length / h.strips
        x = -h.length/2 + (np.arange(h.strips)+.5)*dx
        barrel_half = h.length/2 - h.radius
        cap_x = np.maximum(np.abs(x)-barrel_half, 0.)
        r2 = h.radius**2 - cap_x**2
        area = np.pi*r2
        dV = area*dx
        self.volume = dV.sum()
        self.mass = RHO*self.volume
        # Uniform capsule inertias used as an illustrative I_CG, plus a shifted
        # CG/ballast surrogate. The ballast distribution is not a measured fit.
        Ixx = RHO*np.sum(dV*r2/2)
        Iyy = RHO*np.sum(dV*(x*x + r2/4))
        rg = np.array([0., 0., -h.bg])
        Sg = skew(rg)
        self.M = np.block([[self.mass*np.eye(3), -self.mass*Sg],
                           [self.mass*Sg, np.diag([Ixx,Iyy,Iyy]) - self.mass*Sg@Sg]])
        # Circular-strip transverse added mass Ca=1 follows Felix's derivation.
        # Axial approximation: 1/2 rho * combined cap volume (sphere surrogate).
        # This is NOT the exact axial added mass of a finite capsule.
        caps_volume = 4*np.pi*h.radius**3/3
        axial_ca = .5*caps_volume/self.volume
        added = np.diag([axial_ca*self.mass, self.mass, self.mass,
                         0., RHO*np.sum(dV*x*x), RHO*np.sum(dV*x*x)])
        self.A = np.broadcast_to(added, (len(self.omega),6,6)).copy()
        self.B = np.broadcast_to(h.damping_rate*(self.M+added), self.A.shape).copy()
        self.C_hydro = np.diag([0.,0.,0., self.mass*G*h.bg, self.mass*G*h.bg,0.])
        self.F_exc = np.zeros((len(self.omega),len(self.theta),6), complex)
        ct, st = np.cos(self.theta), np.sin(self.theta)
        # Evaluate wave fluid acceleration at strip centres at mean depth.
        # Small-section approximation: ignore phase/attenuation across radius.
        for start in range(0, len(self.omega), 64):
            sl = slice(start, start+64)
            w = self.omega[sl]
            k = w*w/G
            basis = np.exp(1j*k[:,None,None]*ct[None,:,None]*x[None,None,:])
            v0 = np.einsum('fdx,x->fd',basis,dV)
            v1 = np.einsum('fdx,x->fd',basis,dV*x)
            scale = RHO*w[:,None]**2*np.exp(-k[:,None]*h.depth)
            self.F_exc[sl,:,0] = -1j*scale*(1+axial_ca)*ct*v0
            self.F_exc[sl,:,1] = -1j*scale*2*st*v0
            self.F_exc[sl,:,2] = -scale*2*v0
            # r x F: pitch=-x Fz; yaw=x Fy. No direct roll excitation.
            self.F_exc[sl,:,4] = scale*2*v1
            self.F_exc[sl,:,5] = -1j*scale*2*st*v1
        self.rao = compute_rao(self,self.omega,self.A,self.B,self.F_exc) if build_response else None

    def motion_response(self, omega, theta):
        _same_grid(self.omega,self.theta,omega,theta)
        if self.rao is None:
            raise ValueError("This time-only hull has no motion-response table")
        return self.rao


def _same_grid(w, th, omega, theta):
    if np.shape(w) != np.shape(omega) or np.shape(th) != np.shape(theta) or not np.allclose(w,omega) or not np.allclose(th,theta):
        raise ValueError("Response table/grid mismatch; resample explicitly, never silently extrapolate")


class ResponseTable:
    """Estimator boundary: replace only the response provider, keeping sensors/fit.

    Existing solved WM objects can be adapted without importing their scripts.
    Caller must confirm the reference depth, axes and exp(-i omega t) convention.
    """
    def __init__(self, *, omega, theta, rao, depth, reference, rho=RHO, g=G):
        self.omega, self.theta = np.asarray(omega,float), np.asarray(theta,float)
        self.rao = np.asarray(rao,complex)
        if self.rao.shape != (len(self.omega),len(self.theta),6) or not np.all(np.isfinite(self.rao)):
            raise ValueError("Finite RAO[frequency,direction,6] required")
        if depth <= 0 or not reference:
            raise ValueError("Positive reference depth and explicit reference description required")
        self.depth, self.reference, self.rho, self.g = depth, reference, rho, g
        self.dofs = list(DOFS)

    @classmethod
    def from_team(cls, sub, *, omega, depth, reference):
        # Compatible with solved WM M/A/B/C_hydro/F_exc layouts. Reorder all
        # force/matrix axes together if the team changes its DOF ordering.
        order = [list(sub.dofs).index(d) for d in DOFS]
        a = np.asarray(sub.A)[:,order][:,:,order]
        b = np.asarray(sub.B)[:,order][:,:,order]
        m = np.asarray(sub.M)[order][:,order]
        c = np.asarray(sub.C_hydro)[order][:,order]
        force = np.asarray(sub.F_exc)[:,:,order]
        holder = type('Matrices',(),{'M':m,'C_hydro':c})()
        rao = compute_rao(holder,omega,a,b,force)
        return cls(omega=omega,theta=sub.theta,rao=rao,depth=depth,reference=reference)

    def motion_response(self, omega, theta):
        _same_grid(self.omega,self.theta,omega,theta)
        return self.rao

    def on_grid(self,omega):
        """Explicit complex-linear frequency interpolation, no extrapolation.

        Keeps the team's original direction grid. Computing responses directly
        on the target grid is preferable near sharp response peaks/notches.
        """
        omega = np.asarray(omega,float)
        if np.any(np.diff(self.omega)<=0) or np.any(np.diff(omega)<=0) or omega[0]<self.omega[0]-1e-12 or omega[-1]>self.omega[-1]+1e-12:
            raise ValueError("Ordered frequencies within source response band required")
        if omega.shape==self.omega.shape and np.allclose(omega,self.omega):
            return self
        flat = self.rao.reshape(len(self.omega),-1)
        out = np.stack([np.interp(omega,self.omega,c.real)+1j*np.interp(omega,self.omega,c.imag) for c in flat.T],axis=1)
        return ResponseTable(omega=omega,theta=self.theta,rao=out.reshape(len(omega),len(self.theta),6),
                             depth=self.depth,reference=self.reference,rho=self.rho,g=self.g)

    def save(self,path):
        np.savez(path,omega=self.omega,theta=self.theta,rao=self.rao,depth=self.depth,
                 reference=self.reference,rho=self.rho,g=self.g,convention='exp(-i omega t)',dofs=np.array(DOFS))

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as d:
            if str(d['convention']) != 'exp(-i omega t)' or tuple(d['dofs']) != DOFS:
                raise ValueError("Unsupported phase/DOF convention")
            return cls(omega=d['omega'],theta=d['theta'],rao=d['rao'],depth=float(d['depth']),
                       reference=str(d['reference']),rho=float(d['rho']),g=float(d['g']))
