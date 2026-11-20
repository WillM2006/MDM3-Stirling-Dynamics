"""
invert_waves.py with a different estimator: the Maximum Entropy Method (MEM).

Everything else is imported unchanged from invert_waves.py (reading, Welch matrix, transfer functions, channel
operators, Wishart noise variances and every summary), so run it exactly the same way, plus --dchi2:

    python invert_waves_mem.py --data sensor_data [--model six_dof] [--plot]
    python invert_waves_mem.py --dchi2 10              # looser fit = smoother spectrum
Saves output/data/inversion_result_mem.npz (and output/images/inversion_result_mem.png with --plot).

What changes, per frequency
    invert_waves.py:  x_hat = argmin |A~ x - d~|^2 + |x|^2/(tau s)^2                       (closed form, x can be < 0)
    this file:        the x >= 0 that maximises the entropy  -sum x ln(x/m)  among those that fit the data within dchi2
                      of the best possible fit (Lygre & Krogstad 1986; Kobune & Hashimoto 1986; Skilling & Bryan 1984)
    result            x_j = m_j exp(-sum_n lambda_n A_nj / beta)  -  ln x is a combination of the data kernels
Default model m flat in heading at the Bayesian prior scale s (tau is unused). The fit tolerance is set by --dchi2.
"""
import argparse
import functools
import sys

import numpy as np

import invert_waves as iw
from invert_waves import stack_pairs, operator_matrix, data_vector, noise_variances

DCHI2 = 1.0                # accept a fit up to DCHI2 worse (in chi^2) than the best x >= 0 gives; bigger = smoother


def mem_solve(At, dt, m, beta, y0, iters=500):
    """Maximum entropy fit at one frequency:
        min_x  beta * sum_j [x_j ln(x_j/m_j) - x_j + m_j]  +  1/2 ||At x - dt||^2,    x > 0
    At the optimum ln(x/m) = -At' r / beta (r = residual): ln x is a combination of the data kernels, the
    Lygre & Krogstad / Kobune & Hashimoto MEM form, for any mix of sensors. The problem is strictly convex. It is
    solved in u = ln(x/m) (so x > 0 automatically and headings with no power can go to ~0) by damped Newton steps
    with a positive-definite Hessian approximation and a backtracking line search. Returns x (J,), chi^2, y = x/m."""
    B = At * m[None, :]
    BtB, Btd = B.T @ B, B.T @ dt

    def obj(u):
        y = np.exp(u)
        r = B @ y - dt
        return beta * np.sum(m * (y * u - y + 1)) + 0.5 * r @ r

    u = np.log(np.maximum(np.asarray(y0, float), 1e-300))
    u = np.clip(u, -700, 50)
    f = obj(u)
    for _ in range(iters):
        y = np.exp(u)
        gy = beta * m * u + BtB @ y - Btd                           # gradient w.r.t. y
        g = y * gy                                                  # gradient w.r.t. u
        H = (BtB * y[:, None]) * y[None, :] + np.diag(beta * m * y + np.maximum(g, 0))
        H += 1e-12 * np.trace(H) / len(u) * np.eye(len(u))
        step = -np.linalg.solve(H, g)
        step = np.clip(step, -20, 20)                               # trust region on ln x
        a, slope = 1.0, g @ step
        while a > 1e-10:
            un = np.clip(u + a * step, -700, 50)
            fn = obj(un)
            if fn <= f + 1e-4 * a * slope:
                break
            a *= 0.5
        if a <= 1e-10 or f - fn <= 1e-14 * max(abs(f), 1e-300):
            if a > 1e-10:
                u, f = un, fn
            break
        u, f = un, fn
    y = np.exp(u)
    r = B @ y - dt
    return m * y, r @ r, y


def invert(f, S, h_all, tau, M, dchi2=DCHI2):
    """same inputs and outputs as invert_waves.invert: x_hat (L,J) (>= 0 by construction), covariance (L,J,J), prior scale s (L,).
    Default model m_j = s/J: flat in heading, at the Bayesian prior scale s (tau is unused).
    The entropy weight beta is chosen per bin: the largest beta (smoothest, least committal x) whose chi^2 is within
    dchi2 of the best fit any x >= 0 can give. (The usual "chi^2 = N" rule is not used: with these Wishart noise
    variances every auto-spectrum is allowed a ~60% error, so chi^2 = N lets MEM shrink the total power a lot.)
    Covariance: Laplace approximation (At'At + beta diag(1/x))^-1, optimistic like the Bayesian one."""
    L, J, C = h_all.shape
    pairs = stack_pairs(C)
    X, COV, SC = np.zeros((L, J)), np.zeros((L, J, J)), np.zeros(L)
    chi2_fit = []
    for l in range(L):
        A = operator_matrix(h_all[l], pairs)
        d = data_vector(S[l], pairs)
        w = 1 / np.sqrt(noise_variances(S[l], pairs, M))
        At, dt = A * w[:, None], d * w
        N = len(dt)
        s = np.mean([S[l, c, c].real / np.mean(np.abs(h_all[l, :, c]) ** 2) for c in range(C)])
        m = np.full(J, s / J)
        b0 = N / s                                                  # entropy ~ s and chi^2 ~ N are then comparable
        # 1. the best fit any x >= 0 can give (entropy weight -> 0, warm-started down from a heavy weight)
        y = np.ones(J)
        for lb in np.log(b0) + np.linspace(6.0, -16.0, 23):
            x_min, chi2_min, y = mem_solve(At, dt, m, np.exp(lb), y)
        lo, best = lb, (x_min, np.exp(lb))
        # 2. the largest entropy weight (smoothest, least committal x) whose fit is within dchi2 of that best fit
        target = chi2_min + dchi2
        hi = np.log(b0) + 6.0
        x_hi, chi2_hi, _ = mem_solve(At, dt, m, np.exp(hi), np.ones(J))
        if chi2_hi <= target:                                       # even the heaviest weight fits: take it
            lo, best = hi, (x_hi, np.exp(hi))
        while hi - lo > 0.05:                        # bisection on ln(beta); chi^2 rises with beta
            mid = 0.5 * (lo + hi)
            xm, chi2m, ym = mem_solve(At, dt, m, np.exp(mid), y)
            if chi2m <= target:
                lo, best, y = mid, (xm, np.exp(mid)), ym
            else:
                hi = mid
        chi2_fit.append(chi2_min / N)
        x, beta = best
        COV[l] = np.linalg.pinv(At.T @ At + beta * np.diag(1 / np.maximum(x, 1e-300)), hermitian=True)
        X[l], SC[l] = x, s
    print(f"MEM: best achievable chi^2/N per bin: median {np.median(chi2_fit):.2f}, max {np.max(chi2_fit):.2f} "
          f"(about 1 if the noise model is right; well below 1 means it is pessimistic)")
    return X, COV, SC


if __name__ == "__main__":
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)    # this script's options; everything else goes on to invert_waves.main()
    ap.add_argument("--dchi2", type=float, default=DCHI2, help="allowed chi^2 above the best fit; bigger = smoother")
    opt, sys.argv[1:] = ap.parse_known_args()
    print(f"estimator: MEM, dchi2 = {opt.dchi2}")
    iw.invert = functools.partial(invert, dchi2=opt.dchi2)   # main() calls invert(); point it at this one
    iw.RESULT = "inversion_result_mem"
    iw.main()
