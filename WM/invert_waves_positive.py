"""
invert_waves.py with one change: wave power cannot be negative.

Everything else is imported unchanged from invert_waves.py (reading, Welch matrix, transfer functions, channel
operators, Wishart noise variances, the prior N(0, (tau s)^2) and every summary), so run it exactly the same way:

    python invert_waves_positive.py --data sensor_data [--model six_dof] [--plot]

What changes, per frequency
    invert_waves.py:     x_hat = argmin |A~ x - d~|^2 + |x|^2/(tau s)^2                (closed form, x can be < 0)
    this file:           the same objective, minimised subject to x >= 0              (non-negative least squares)

Why it is still the same Bayesian method: the prior becomes the same Gaussian cut off at zero (a half-normal),
so the posterior is the same Gaussian cut off at zero, and its peak (the MAP estimate) is the constrained minimum.
Writing the prior term as extra rows, the problem is ordinary NNLS:

    minimise | [ A~         ] x  -  [ d~ ] |^2    subject to x >= 0
             | [ I/(tau s)  ]       [ 0  ] |

Uncertainty (approximate): headings that come out exactly 0 are pinned at the boundary and given zero variance;
for the others the covariance is the inverse of Lambda restricted to them (a Laplace approximation at the MAP).
Like the original error bar it ignores model error and row correlations, so it is still optimistic.
"""
import numpy as np
from scipy.optimize import nnls

import invert_waves as iw
from invert_waves import stack_pairs, operator_matrix, data_vector, noise_variances


def invert(f, S, h_all, tau, M):
    """same inputs and outputs as invert_waves.invert: x_hat (L,J) >= 0, covariance (L,J,J), prior scale s (L,)"""
    L, J, C = h_all.shape
    pairs = stack_pairs(C)
    X, COV, SC = np.zeros((L, J)), np.zeros((L, J, J)), np.zeros(L)
    for l in range(L):
        A = operator_matrix(h_all[l], pairs)
        d = data_vector(S[l], pairs)
        w = 1 / np.sqrt(noise_variances(S[l], pairs, M))
        At, dt = A * w[:, None], d * w
        s = np.mean([S[l, c, c].real / np.mean(np.abs(h_all[l, :, c]) ** 2) for c in range(C)])   # as before
        # --- the only new part: prior as extra rows, then non-negative least squares
        sc = 1.0 / (tau * s)                                          # prior weight, as in Lambda below
        x, _ = nnls(np.vstack([At, np.eye(J) * sc]), np.concatenate([dt, np.zeros(J)]), maxiter=50 * J)
        # --- approximate covariance: Lambda on the free (positive) headings, zero for those pinned at 0
        Lam = At.T @ At + np.eye(J) * sc**2
        free = x > 0
        cov = np.zeros((J, J))
        if free.any():
            cov[np.ix_(free, free)] = np.linalg.inv(Lam[np.ix_(free, free)])
        X[l], COV[l], SC[l] = x, cov, s
    return X, COV, SC


if __name__ == "__main__":
    iw.invert = invert          # main() in invert_waves.py calls invert(); point it at this one
    iw.main()
