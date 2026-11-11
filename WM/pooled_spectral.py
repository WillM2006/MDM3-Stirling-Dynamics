"""Independent-pool covariance: fixed-M marginal variances and empirical correlations."""
import numpy as np
from scipy.linalg import cho_factor,cho_solve
import invert_waves as iw
from check_cross_variance import variance_terms,corrected_variances
from correlated_spectral import functional_vectors


def shrink_correlation(values):
    """Diagonal-target estimated-risk shrinkage, using independent sensor records only."""
    n,p=values.shape
    if n<8 or not np.isfinite(values).all():
        raise ValueError("at least eight finite calibration records are required")
    centred=values-values.mean(0)
    standard=centred.std(0,ddof=1)
    if np.any(standard<=0):raise ValueError("constant calibration feature")
    z=centred/standard
    correlation=z.T@z/(n-1)
    products=z[:,:,None]*z[:,None,:]
    variance=n/(n-1)**3*np.sum((products-products.mean(0))**2,axis=0)
    off=~np.eye(p,dtype=bool)
    denominator=np.sum(correlation[off]**2)
    intensity=float(np.clip(variance[off].sum()/denominator,0,1)) if denominator>0 else 1.
    regularised=(1-intensity)*correlation+intensity*np.eye(p)
    eigen,vectors=np.linalg.eigh(regularised)
    if eigen.min()<=0:raise ValueError("estimated correlation is not positive definite")
    return (vectors*np.sqrt(eigen))@vectors.T,intensity


def fit_pool(spectra,m):
    """Noise of a new native-duration record, not uncertainty of the calibration mean."""
    if spectra.ndim!=4 or not np.isfinite(spectra).all():
        raise ValueError("expected record-by-frequency sensor spectral matrices")
    n,count,channels,_=spectra.shape
    pairs=iw.stack_pairs(channels)
    observations=np.stack([[iw.data_vector(s,pairs) for s in record] for record in spectra])
    marginal=corrected_variances(*variance_terms(spectra),m).mean(0)
    factors=[];shrinkage=[]
    for fi in range(count):
        root,intensity=shrink_correlation(observations[:,fi])
        factors.append(np.sqrt(marginal[fi])[:,None]*root)
        shrinkage.append(intensity)
    return np.stack(factors),np.array(shrinkage)


def pooled_inverse(f,spectra,responses,tau,factors,frequency_correlation,scales):
    """Equivalent Gaussian solve, block-preconditioned before forming normal equations."""
    count,headings,channels=responses.shape
    design=[];data=[];transforms=[];prior=[]
    pairs=iw.stack_pairs(channels)
    for fi in range(count):
        standard=np.linalg.norm(factors[fi],axis=1)
        target=np.column_stack((iw.operator_matrix(responses[fi],pairs)*scales[fi],iw.data_vector(spectra[fi],pairs)))
        white=np.linalg.solve(factors[fi]/standard[:,None],target/standard[:,None])
        u,s,vt=np.linalg.svd(white[:,:headings],full_matrices=False)
        # Remove only numerical rank noise before the prior-aware transformation.
        s=np.where(s>np.finfo(float).eps*max(white.shape)*s[0],s,0.)
        inverse_root=1/np.sqrt(s*s+1/tau**2)
        transform=(vt.T*inverse_root)@vt
        transforms.append(transform)
        design.append((u*(s*inverse_root))@vt)
        prior.append((vt.T*(inverse_root**2/tau**2))@vt)
        data.append(white[:,-1])
    design=np.stack(design);data=np.stack(data)
    ri=cho_solve(cho_factor(frequency_correlation),np.eye(count))
    flat=design.transpose(0,2,1).reshape(count*headings,channels**2)
    precision=(flat@flat.T)*np.repeat(np.repeat(ri,headings,axis=0),headings,axis=1)
    for fi in range(count):
        index=slice(fi*headings,(fi+1)*headings)
        precision[index,index]+=prior[fi]
    rhs=np.einsum("fcj,fc->fj",design,ri@data).ravel()
    factor=cho_factor(precision)
    z=cho_solve(factor,rhs).reshape(count,headings)
    x=np.einsum("fij,fj->fi",np.stack(transforms),z)*scales[:,None]
    vectors=functional_vectors(scales,headings,float(f[1]-f[0])).reshape(count,headings,-1)
    transformed=np.einsum("fij,fjk->fik",np.stack(transforms),vectors).reshape(count*headings,-1)
    covariance=transformed.T@cho_solve(factor,transformed)
    residual=np.linalg.norm(precision@z.ravel()-rhs)/(np.linalg.norm(precision)*np.linalg.norm(z)+np.linalg.norm(rhs))
    return x,covariance,float(residual)
