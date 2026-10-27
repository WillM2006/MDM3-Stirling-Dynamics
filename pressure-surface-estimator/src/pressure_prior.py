"""Learn a single-system directional/spectral prior from six pressures.

Only past samples and nominal hull calibration are accepted. This is a
working tapered matrix-Whittle fit, not a confidence interval or an exact
finite-record likelihood. A parametric prior is an assumption, not sea truth.
"""
from dataclasses import dataclass,asdict
import numpy as np
from scipy.optimize import minimize,minimize_scalar
from sea import FixedSea,jonswap
from pressure_array import array_response


@dataclass
class PressurePrior:
    mean_direction_deg: float
    spread_s: float
    Hs_m: float
    Tp_s: float
    gamma: float
    objective: float
    samples: int
    history_duration_s: float
    boundary: bool
    diagnostic: dict

    def weights(self,f,theta,*,uniform_fraction=.01):
        p=FixedSea(self.gamma,np.deg2rad(self.mean_direction_deg),self.spread_s).fractions(theta)
        p=(1-uniform_fraction)*p+uniform_fraction/len(theta)
        spectrum=jonswap(f,self.Hs_m,self.Tp_s,self.gamma)
        return spectrum[:,None]*p[None,:]/spectrum.sum()

    def to_dict(self):return asdict(self)


def estimate_prior(pressures_Pa,model,config,*,sample_interval_s=.5,
                   fit_band=(.06,.26),rank_tolerance=1e-7,gamma=3.3):
    """Frequency-profiled angular fit; scalar spectrum is a nuisance per bin.

    Fit direction/spread first using complex relative phases across six ports.
    Then fit Hs/Tp from the profiled spectral powers, with gamma fixed. No
    generator parameters, hidden motion, component phases or IMU enter.
    """
    y=np.asarray(pressures_Pa,float)
    if y.ndim!=2 or y.shape[1]!=6 or len(y)<512 or not np.isfinite(y).all():
        raise ValueError('At least 512 finite samples and exactly six pressures required')
    n=len(y);fs=1/sample_interval_s;freq=np.fft.rfftfreq(n,sample_interval_s)
    inside=(freq>=fit_band[0])&(freq<=fit_band[1])
    f=freq[inside];table=model
    H=array_response(table,table.omega,table.theta,config)[:,:,:6]
    window=np.hanning(n+1)[:-1]
    z=np.conj(np.fft.rfft((y-y.mean(axis=0))*window[:,None],axis=0)[inside])*np.sqrt(2/(fs*np.sum(window**2)))
    # Expected finite-window periodogram, including positive/negative-frequency
    # leakage. This avoids mistaking short-window leakage for long waves.
    source_f=table.omega/(2*np.pi);df=source_f[1]-source_f[0]
    def dirichlet(q):
        return n*np.exp(-1j*np.pi*(n-1)*q)*np.sinc(n*q)/np.sinc(q)
    def taper_transform(q):
        return .5*dirichlet(q)-.25*dirichlet(q-1/n)-.25*dirichlet(q+1/n)
    positive=abs(taper_transform((f[:,None]-source_f[None,:])*sample_interval_s))**2
    negative=abs(taper_transform((f[:,None]+source_f[None,:])*sample_interval_s))**2
    positive*=df/(fs*np.sum(window**2));negative*=df/(fs*np.sum(window**2))
    # Vertically separated ports at identical xy cancel first-order vehicle
    # motion. Learn the frequency shape from their difference, not true eta.
    positions=np.asarray(config.positions);pairs=[]
    for a in range(6):
        for b in range(a):
            if np.allclose(positions[a,:2],positions[b,:2]) and not np.isclose(positions[a,2],positions[b,2]):pairs.append((a,b))
    if not pairs:raise ValueError('This prior estimator requires a vertically separated port pair at identical xy')
    a,b=max(pairs,key=lambda ab:abs(positions[ab[0],2]-positions[ab[1],2]))
    k=table.omega**2/table.g
    gain=table.rho*table.g*(np.exp(-k*(table.depth-positions[a,2]))-np.exp(-k*(table.depth-positions[b,2])))
    scalar_I=abs(z[:,a]-z[:,b])**2
    kernel=(positive+negative)*gain[None,:]**2
    def period_value(Tp,return_height=False):
        shape=kernel@jonswap(source_f,1.,Tp,gamma)
        ratio=np.mean(scalar_I/shape)
        h2=np.clip(ratio,.2**2,5.**2)
        value=np.mean(np.log(shape))+np.log(h2)+ratio/h2
        return (float(value),float(np.sqrt(h2))) if return_height else value
    periods=np.linspace(5,12,141);values=[period_value(t) for t in periods];best=int(np.argmin(values))
    fit=minimize_scalar(period_value,bounds=(max(5,periods[best]-.15),min(12,periods[best]+.15)),method='bounded')
    Tp=float(fit.x);_,Hs=period_value(Tp,True)
    spectrum=jonswap(source_f,1.,Tp,gamma)
    outer_source=np.einsum('fdc,fde->fdce',H,H.conj())*spectrum[:,None,None,None]
    flat=outer_source.reshape(len(source_f),-1)
    outer_all=(positive@flat+negative@flat.conj()).reshape(len(f),len(table.theta),6,6)
    isotropic=outer_all.mean(axis=1)
    scale=np.sqrt(isotropic.diagonal(axis1=1,axis2=2).real)
    outer_all/=scale[:,None,:,None]*scale[:,None,None,:]
    z=z/scale
    isotropic=outer_all.mean(axis=1)
    ev,U=np.linalg.eigh(isotropic);ranks=(ev>rank_tolerance*ev[:,-1:]).sum(axis=1)
    power=np.sum(abs(z)**2,axis=1)
    use=(power>1e-7*power.max())
    groups=[]
    for rank in np.unique(ranks[use]):
        ids=np.flatnonzero(use&(ranks==rank))
        V=U[ids,:,-rank:]/np.sqrt(ev[ids,None,-rank:])
        zz=np.einsum('fcr,fc->fr',V.conj(),z[ids])
        outer=np.einsum('fcr,fdce,fes->fdrs',V.conj(),outer_all[ids],V)
        groups.append((int(rank),ids,outer,zz))
    count=int(sum(rank*len(ids) for rank,ids,_,_ in groups))
    if count<100:raise ValueError('Insufficient retained spectral information')

    def sufficient(mu,s):
        p=FixedSea(gamma,np.deg2rad(mu),s).fractions(table.theta)
        qfull=np.zeros(len(f));total=0.
        for rank,ids,outer,zz in groups:
            K=np.einsum('fdrs,d->frs',outer,p)
            try:
                L=np.linalg.cholesky(K)
                ld=2*np.log(np.diagonal(L,axis1=1,axis2=2).real).sum(axis=1)
                solved=np.linalg.solve(K,zz[...,None])[...,0]
                q=np.einsum('fr,fr->f',zz.conj(),solved).real
                if np.any(q<=0):return np.inf,qfull
                total+=float(np.sum(ld+rank*np.log(q/rank)))
                qfull[ids]=q
            except np.linalg.LinAlgError:return np.inf,qfull
        return total/count,qfull

    coarse=[]
    for s in [1.,3.,8.,20.,50.]:
        for mu in np.arange(0,360,15.):coarse.append((sufficient(mu,s)[0],mu,s))
    candidates=[]
    for value,mu,s in sorted(coarse)[:4]:
        initial=[mu,np.log(s)]
        result=minimize(lambda x:sufficient(x[0],np.exp(x[1]))[0],initial,method='Powell',
                        bounds=[(mu-30,mu+30),(0,np.log(50))],options={'xtol':1e-5,'ftol':1e-9,'maxiter':100})
        candidates.append((float(result.fun),float(result.x[0]%360),float(np.exp(result.x[1]))))
    objective,mu,s=min(candidates);_,q=sufficient(mu,s)
    return PressurePrior(mu,s,Hs,Tp,gamma,float(objective),n,(n-1)*sample_interval_s,
                         bool(s<1.01 or s>49.5 or Tp<5.01 or Tp>11.99 or Hs<.201 or Hs>4.99),
                         {'rank_min':int(ranks[use].min()),'rank_max':int(ranks[use].max()),
                          'frequency_bins':int(use.sum()),'fit_band_Hz':fit_band,
                          'method':'Finite-window expected-periodogram fit: motion-cancelling pressure pair for frequency shape, matrix-Whittle angular profile',
                          'motion_cancelling_pair':[int(a),int(b)],
                          'limits':'Single separable cos-2s direction prior; fixed gamma. Taper leakage modeled; no uncertainty intervals.'})
