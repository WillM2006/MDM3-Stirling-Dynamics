"""Linear spectral observation operator for unknown frequency-cell powers.

No wave family, phases, simulated truth, covariance or inverse scores enter
this operator. Responses and processing describe the observation process.
"""
import numpy as np
import invert_waves as iw
from audit_spectral_mean import processing_gain, welch_weights


def quadrature(edges, order=16):
    edges=np.asarray(edges,float)
    if edges.ndim!=1 or len(edges)<2 or not np.isfinite(edges).all() or np.any(np.diff(edges)<=0):
        raise ValueError('frequency edges must be finite and increasing')
    nodes,weights=np.polynomial.legendre.leggauss(order)
    width=np.diff(edges)/2
    return (edges[:-1,None]+edges[1:,None])/2+width[:,None]*nodes, width[:,None]*weights


def observation_operator(f,nodes,weights,responses):
    """Return B[f,entry,cell*heading] for cell-constant power per Hz.

    Responses have shape (cell,quadrature,heading,channel), with the nine
    channels in iw.CHANNELS order. Negative-frequency leakage is retained.
    """
    if (nodes.ndim!=2 or weights.ndim!=2 or responses.ndim!=4 or np.asarray(f).ndim!=1
            or nodes.shape!=weights.shape or responses.shape[:2]!=nodes.shape or responses.shape[-1]!=9
            or not all(np.isfinite(a).all() for a in [f,nodes,weights,responses])
            or np.any(weights<=0) or np.any(np.asarray(f)<=0) or np.any(np.asarray(f)>=iw.FS/2)):
        raise ValueError('invalid spectral quadrature or response shape')
    cells,order,headings,channels=responses.shape
    pairs=iw.stack_pairs(channels)
    signs=np.array([s for c,k in pairs for s in ([1] if c==k else [1,-1])])
    result=np.empty((len(f),channels**2,cells*headings))
    for cell in range(cells):
        positive,negative=welch_weights(f,nodes[cell])
        filtered=responses[cell]*processing_gain(nodes[cell])[:,None,:]
        local=np.stack([iw.operator_matrix(h,pairs) for h in filtered])
        block=((positive*weights[cell])@local.reshape(order,-1)
               +(negative*weights[cell])@(local*signs[None,:,None]).reshape(order,-1))
        result[:,:,cell*headings:(cell+1)*headings]=block.reshape(len(f),channels**2,headings)
    return result
