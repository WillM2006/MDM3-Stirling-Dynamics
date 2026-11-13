"""Time-domain checks of the cell-power observation operator."""
import unittest
import numpy as np
import invert_waves as iw
from audit_spectral_mean import processing_gain
from processed_spectral import observation_operator,quadrature


class ProcessedSpectralTests(unittest.TestCase):
    def test_unknown_cell_powers_match_independent_harmonic_csds(self):
        f=np.arange(10,63)*.005
        nodes,weights=quadrature(np.array([.107,.118,.131]),2)
        rng=np.random.default_rng(739)
        responses=rng.normal(size=(2,2,2,9))+1j*rng.normal(size=(2,2,2,9))
        powers=np.array([[.3,.8],[1.1,.4]])
        operator=observation_operator(f,nodes,weights,responses)
        predicted=operator@powers.ravel()
        expected=np.zeros((53,81));t=np.arange(1800)/iw.FS
        for cell in range(2):
            for node in range(2):
                nu=nodes[cell,node]
                for heading in range(2):
                    h=responses[cell,node,heading]*processing_gain(np.array([nu]))[0]
                    for phase in [0,np.pi/2]:
                        x=np.real(np.sqrt(2*powers[cell,heading]*weights[cell,node])
                                  *np.exp(-2j*np.pi*nu*t+1j*phase)[:,None]*h)
                        ff,s=iw.welch_matrix(x)
                        s=s[(ff>=iw.BAND[0])&(ff<=iw.BAND[1])]
                        expected+=np.stack([iw.data_vector(v,iw.stack_pairs(9)) for v in s])/2
        np.testing.assert_allclose(predicted,expected,rtol=1e-9,atol=1e-10)
        np.testing.assert_allclose(operator@(2*powers.ravel()),2*predicted,rtol=1e-14,atol=1e-14)

    def test_quadrature_keeps_cell_width_and_moments(self):
        edges=np.array([.01,.03,.1])
        nodes,weights=quadrature(edges,4)
        np.testing.assert_allclose(weights.sum(1),np.diff(edges),rtol=1e-14)
        np.testing.assert_allclose(np.sum(weights*nodes**2,1),np.diff(edges**3)/3,rtol=1e-14)

    def test_rejects_invalid_operator_inputs(self):
        with self.assertRaises(ValueError):quadrature(np.array([.1,.05]))
        n,w=quadrature(np.array([.1,.2]),2);h=np.ones((1,2,2,9),complex)
        with self.assertRaises(ValueError):observation_operator(np.array([0.]),n,w,h)
        with self.assertRaises(ValueError):observation_operator(np.array([.1]),n,-w,h)
        with self.assertRaises(ValueError):observation_operator(np.array([.1]),n,w,h[:,:,:,:8])


if __name__=='__main__':unittest.main()
