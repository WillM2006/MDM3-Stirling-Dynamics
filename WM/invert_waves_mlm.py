"""
invert_waves.py with a different estimator: the Extended Maximum Likelihood Method (EMLM / IMLM).

Everything else is imported unchanged from invert_waves.py (reading, Welch matrix, transfer functions, channel
operators and every summary), so run it exactly the same way, plus --load / --iter / --gain:

    python invert_waves_mlm.py --data sensor_data [--model six_dof] [--plot]
    python invert_waves_mlm.py --iter 20 --plot        # IMLM
    python invert_waves_mlm.py --load 1e-2             # more diagonal loading = smoother, more robust
Saves output/inversion_result_mlm.npz (and .png with --plot).

What changes, per frequency
    invert_waves.py:  fit d = A x (real/imag parts of the cross-spectra) with a Gaussian prior        (closed form)
    this file:        work on the cross-spectral matrix itself, with the same model Phi = sum_j x_j g_j g_j^H
        shape    P_j = 1 / (g_j^H (Phi + load I)^-1 g_j)                  (Capon 1969; Isobe et al. 1984)
        IMLM     optional: iterate P until MLM applied to the model of P gives back the MLM of the data (--iter)
        level    x = kappa P, kappa = median over channels of measured / predicted auto-spectrum
MLM is positive by construction but smears peaks; it gives no posterior covariance (zeros, so no sd is reported).
"""
import argparse
import functools
import sys

import numpy as np

import invert_waves as iw
from invert_waves import stack_pairs, operator_matrix, data_vector

LOAD = 1e-3                # diagonal loading on the (unit-diagonal) cross-spectral matrix before inverting
N_ITER = 0                 # IMLM iterations (0 = plain EMLM)
GAIN = 1.0                 # IMLM step exponent


def mlm_shape(G, Phi, load):
    """Capon / MLM estimate at one frequency. G: (J, C) channel responses per heading, Phi: (C, C) cross-spectra
    (both already scaled so diag(Phi) = 1). Returns P (J,) >= 0, proportional to the power from each heading:
        P_j = 1 / (g_j^H Phi^-1 g_j),   g_j = row j of G      (Phi = sum_j x_j g_j g_j^H is the model, see operator_matrix)
    Diagonal loading (load * I) keeps Phi^-1 sane: the pressure ports are almost perfectly correlated."""
    C = Phi.shape[0]
    Pinv = np.linalg.inv(Phi + load * np.real(np.trace(Phi)) / C * np.eye(C))
    q = np.einsum("ji,ik,jk->j", np.conj(G), Pinv, G).real
    return 1.0 / np.maximum(q, 1e-300)


def invert(f, S, h_all, tau, M, load=LOAD, n_iter=N_ITER, gain=GAIN):
    """same inputs and outputs as invert_waves.invert: x_hat (L,J), covariance (L,J,J) (zeros: MLM gives none), scale s (L,).
    EMLM (Isobe et al. 1984) at each Welch bin, optionally iterated (IMLM, Pawka 1983 / Oltman-Shay & Guza 1984).
    tau and M are unused (kept so the call matches the Bayesian version).
    MLM only gives the SHAPE over heading; the level kappa is the median over channels of
    (measured auto-spectrum) / (auto-spectrum predicted by P)."""
    L, J, C = h_all.shape
    pairs = stack_pairs(C)
    auto = np.array(sum(([True] if i == k else [False, False] for i, k in pairs), []))   # auto-spectrum rows of d
    X, COV, SC = np.zeros((L, J)), np.zeros((L, J, J)), np.zeros(L)
    for l in range(L):
        sc = np.sqrt(np.maximum(S[l].diagonal().real, 1e-300))      # scale every channel to unit power: MLM is
        Phi = S[l] / np.outer(sc, sc)                               # invariant to this, but inv() is happier
        G = h_all[l] / sc[None, :]
        P0 = mlm_shape(G, Phi, load)
        P = P0 / P0.sum()
        for _ in range(n_iter):                                     # IMLM: push P until MLM(model of P) reproduces P0
            Phi_n = (G.T * P) @ np.conj(G)                          # sum_j P_j g_j g_j^H
            T = mlm_shape(G, Phi_n / np.mean(Phi_n.diagonal().real), load)
            P = P * (P0 / P0.sum() / (T / T.sum())) ** gain         # multiplicative update keeps P >= 0
            P /= P.sum()
        A = operator_matrix(h_all[l], pairs)
        d = data_vector(S[l], pairs)
        kappa = np.median(d[auto] / (A @ P)[auto])                      # level from the auto-spectra only (the cross terms
                                                                         # carry direction, which a smeared MLM shape fits
                                                                         # badly); median over channels so one badly-fitted
                                                                         # channel (often roll) can't drag the level
        s = np.mean([S[l, c, c].real / np.mean(np.abs(h_all[l, :, c]) ** 2) for c in range(C)])
        X[l], SC[l] = kappa * P, s
    return X, COV, SC


if __name__ == "__main__":
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)    # this script's options; everything else goes on to invert_waves.main()
    ap.add_argument("--load", type=float, default=LOAD, help="diagonal loading (fraction of mean channel power)")
    ap.add_argument("--iter", type=int, default=N_ITER, help="IMLM iterations, 0 = plain EMLM")
    ap.add_argument("--gain", type=float, default=GAIN, help="IMLM step exponent")
    opt, sys.argv[1:] = ap.parse_known_args()
    print(f"estimator: {'IMLM, ' + str(opt.iter) + ' iterations' if opt.iter else 'EMLM'}, load={opt.load}")
    iw.invert = functools.partial(invert, load=opt.load, n_iter=opt.iter, gain=opt.gain)   # main() calls invert(); point it at this one
    iw.RESULT = "inversion_result_mlm"
    iw.main()
