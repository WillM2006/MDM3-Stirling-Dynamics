"""Sensor-only structured spectral covariance; production code is unchanged."""
import numpy as np
from scipy.linalg import cho_factor,cho_solve,toeplitz
from scipy.signal import get_window
import invert_waves as iw
from check_cross_variance import variance_terms,corrected_variances


def hermitian_basis(channels):
    basis=[]
    for c,k in iw.stack_pairs(channels):
        b=np.zeros((channels,channels),complex)
        if c==k:
            b[c,c]=1
            basis.append(b)
        else:
            b[c,k]=b[k,c]=1/np.sqrt(2)
            basis.append(b)
            b=np.zeros_like(b)
            b[c,k]=1j/np.sqrt(2);b[k,c]=-1j/np.sqrt(2)
            basis.append(b)
    return np.stack(basis)


def wishart_factor(spectral):
    """Factor of real-stacked complex-Wishart covariance for one effective average."""
    channels=spectral.shape[-1]
    eigen,vectors=np.linalg.eigh(spectral)
    if np.any(eigen<=0) or not np.isfinite(spectral).all():
        raise ValueError("a finite positive-definite spectral matrix is required")
    root=(vectors*np.sqrt(eigen))@vectors.conj().T
    transformed=root@hermitian_basis(channels)@root
    return np.stack([iw.data_vector(b,iw.stack_pairs(channels)) for b in transformed],axis=1)


def covariance_factors(spectra,m):
    """D times a Wishart correlation factor; diagonals retain calibrated variances."""
    variances=corrected_variances(*variance_terms(spectra),m)
    if np.any(variances<=0):
        raise ValueError("nonpositive measured spectral variance")
    factors=[];shrinkage=[]
    channels=spectra.shape[-1]
    mask=~np.eye(channels,dtype=bool)
    averages=m-1
    for s,v in zip(spectra,variances):
        scale=np.sqrt(s.diagonal().real)
        correlation=s/scale[:,None]/scale[None,:]
        power=abs(correlation[mask])**2
        # Estimated complex-Wishart squared-error risk, shrinking sensor correlations to zero.
        risk=(averages-power)/(averages**2-1)
        intensity=float(np.clip(risk.sum()/power.sum(),0,1)) if power.sum()>0 else 1.
        regularised=(1-intensity)*correlation+intensity*np.eye(channels)
        factor=wishart_factor(regularised)
        factor=factor/np.linalg.norm(factor,axis=1)[:,None]*np.sqrt(v)[:,None]
        factors.append(factor);shrinkage.append(intensity)
    return np.stack(factors),np.array(shrinkage)


def welch_frequency_correlation(bins,samples=1800,length=400,step=200):
    """Locally flat proper-Gaussian window/overlap covariance, normalised to unit diagonal."""
    bins=np.asarray(bins)
    if (bins.ndim!=1 or len(bins)<1 or not np.array_equal(np.diff(bins),np.ones(len(bins)-1))
            or samples<length or not 0<step<=length):
        raise ValueError("expected contiguous Fourier bins and valid Welch settings")
    count=1+(samples-length)//step
    window=get_window("hann",length,fftbins=True)
    lag=np.arange(len(bins))
    phase=np.exp(-2j*np.pi*lag[:,None]*np.arange(length)[None,:]/length)
    covariance=count*abs(phase@(window**2))**2
    for distance in range(1,min(count,(length-1)//step+1)):
        shift=distance*step
        product=window[shift:]*window[:-shift]
        covariance+=2*(count-distance)*abs(phase[:,:len(product)]@product)**2
    correlation=toeplitz(covariance/covariance[0])
    np.linalg.cholesky(correlation)
    return correlation


def functional_vectors(scales,headings,df):
    count=len(scales)
    vectors=np.zeros((count,headings,count+headings+1))
    for fi,s in enumerate(scales):
        vectors[fi,:,fi]=s
        vectors[fi,:,count:count+headings]=df*s*np.eye(headings)
        vectors[fi,:,-1]=df*s
    return vectors.reshape(count*headings,-1)


def correlated_inverse(f,spectra,responses,tau,m,factors,frequency_correlation,scales):
    """Exact Gaussian posterior for the specified structured covariance and frozen prior."""
    count,headings,channels=responses.shape
    pairs=iw.stack_pairs(channels)
    design=[];data=[]
    for fi in range(count):
        a=iw.operator_matrix(responses[fi],pairs)
        d=iw.data_vector(spectra[fi],pairs)
        standard=np.linalg.norm(factors[fi],axis=1)
        white=np.linalg.solve(factors[fi]/standard[:,None],np.column_stack((a*scales[fi],d))/standard[:,None])
        design.append(white[:,:headings]);data.append(white[:,-1])
    design=np.stack(design);data=np.stack(data)
    inverse_frequency=cho_solve(cho_factor(frequency_correlation),np.eye(count))
    flattened=design.transpose(0,2,1).reshape(count*headings,channels**2)
    precision=(flattened@flattened.T)*np.repeat(np.repeat(inverse_frequency,headings,axis=0),headings,axis=1)
    precision.flat[::len(precision)+1]+=1/tau**2
    rhs=np.einsum("fcj,fc->fj",design,inverse_frequency@data).ravel()
    factor=cho_factor(precision)
    y=cho_solve(factor,rhs)
    x=y.reshape(count,headings)*scales[:,None]
    functionals=functional_vectors(scales,headings,float(f[1]-f[0]))
    covariance=functionals.T@cho_solve(factor,functionals)
    residual=np.linalg.norm(precision@y-rhs)/(np.linalg.norm(precision)*np.linalg.norm(y)+np.linalg.norm(rhs))
    return x,covariance,float(residual)


def propagated_variance(weights,factors,frequency_correlation):
    projected=np.einsum("bfc,fck->bfk",weights,factors)
    return np.einsum("bfk,fg,bgk->b",projected,frequency_correlation,projected)
