
# Integrating JONSWAP into frequency domain work to see how motion in 6 degrees of freedom varies with multiple waves forming a real(ish) sea.
# Two outputs: response spectra (frequency domain danswer) and a time series for checking 
import numpy as np

from six_dof import Hull6, DOF


def jonswap(omega, Hs, Tp, gamma=3.3):
    # copied from Karan + Leo's JONSWAP code. Normalised so 4 sqrt(sum S d_omega) = Hs
    omega_p = 2 * np.pi / Tp
    sigma = np.where(omega <= omega_p, 0.07, 0.09)
    b = np.exp(-0.5 * ((omega - omega_p) / (sigma * omega_p)) ** 2)
    shape = omega**-5 * np.exp(-1.25 * (omega_p / omega) ** 4) * gamma**b
    return shape * (Hs / 4) ** 2 / np.sum(shape * (omega[1] - omega[0]))


def direction_weights(mean_deg, spread_deg, n=25, max_offset=80.0):
    #Energy fraction per direction bin, same cos^2 law and grid as Karan + Leo's work
    # mean_deg is the mean direction of travel vs hull axis (mu: from +x towards +y).
    # returns headings in radians with weights summing to 1

    offsets = np.linspace(-max_offset, max_offset, n)
    w = np.zeros(n)
    if spread_deg == 0:
        w[n // 2] = 1.0
    else:
        inside = np.abs(offsets) < spread_deg
        w[inside] = np.cos(0.5 * np.pi * offsets[inside] / spread_deg) ** 2
        w /= w.sum()
    return np.radians(mean_deg + offsets), w


def response_spectra(hull, omega, S, headings, weights):
    #Cross-spectral matrix of the six motions, shape (n_omega, 6, 6), complex. Radians for rotations
    Sxx = np.zeros((omega.size, 6, 6), complex)
    for mu, wt in zip(headings, weights):
        if wt == 0:
            continue
        H = hull.transfer(omega, mu)                       # (n_omega, 6)
        Sxx += wt * S[:, None, None] * H[:, :, None] * np.conj(H[:, None, :])
    return Sxx


def rms(Sxx, d_omega):
    #RMS of each motionL: Metres for the first three, degrees for the last three
    var = np.real(np.einsum("wii->i", Sxx)) * d_omega
    r = np.sqrt(var)
    r[3:] = np.degrees(r[3:])
    return r


def components(hull, omega, S, headings, weights, seed=0, jitter=True):
    # The wave components of one random sea, as flat arrays (N = frequencies x occupied directions)

    # Each component has amplitude a = sqrt(2 S d_omega w) and a random phase phi. 
    # 'A' is its phasor a e^{i phi} at the hull centre, and 'H' is hull's six transfer functions at (omega, theta).
    # jitter moves each component to random frequency in its bin, ensures no two directions share a frequnecy
    # A fixed grid of 150 sinusoids has a random total variance (Hs scatters by about 10 % from
    # seed to seed) and repeats every 2 pi / d_omega = 520 s - jitter halves the scatter and removes the repeat.

    rng = np.random.default_rng(seed)
    d_omega = omega[1] - omega[0]
    W, TH, A, HH = [], [], [], []
    for mu, wt in zip(headings, weights):
        if wt == 0:
            continue
        w_k = omega + (rng.uniform(-0.5, 0.5, omega.size) * d_omega if jitter else 0.0)
        phi = rng.uniform(0, 2 * np.pi, omega.size)
        W.append(w_k)
        TH.append(np.full(omega.size, mu))
        A.append(np.sqrt(2 * np.interp(w_k, omega, S) * d_omega * wt) * np.exp(1j * phi))
        HH.append(hull.transfer(w_k, mu))                       # (n_omega, 6)
    return dict(w=np.concatenate(W), theta=np.concatenate(TH), A=np.concatenate(A), H=np.concatenate(HH))


def time_series(hull, omega, S, headings, weights, t, seed=0, jitter=True):
    # Motion time series by superposition not for solving just checkign

    #    motion_m(t) = Re sum_n  H_m(omega_n, theta_n) A_n e^{-i omega_n t}      A_n = a_n e^{i phi_n}

    #Returns eta (n_t,), the wave at the hull centre, and X (n_t, 6). Rotations in radians.

    c = components(hull, omega, S, headings, weights, seed, jitter)
    AH = c["A"][:, None] * c["H"]
    eta = np.zeros(t.size)
    X = np.zeros((t.size, 6))
    for i in range(0, t.size, 1000):                            # chunks keep the exponential matrix small
        E = np.exp(-1j * np.outer(t[i:i + 1000], c["w"]))
        eta[i:i + 1000] = np.real(E @ c["A"])
        X[i:i + 1000] = np.real(E @ AH)
    return eta, X


def demo():
    hull = Hull6()
    Hs, Tp, gamma = 3.0, 8.0, 3.3                          # default
    omega = np.linspace(0.2, 2.0, 150)
    d_omega = omega[1] - omega[0]
    S = jonswap(omega, Hs, Tp, gamma)

    print(f"JONSWAP: Hs = {4*np.sqrt(np.sum(S*d_omega)):.3f} m (asked for {Hs}), Tp = {Tp} s")
    print(f"{'mean heading':>12} {'spread':>7} | " + " ".join(f"{n:>7}" for n in DOF))
    for mean in [0, 30, 90]:
        for spread in [0, 40]:
            head, w = direction_weights(mean, spread)
            r = rms(response_spectra(hull, omega, S, head, w), d_omega)
            print(f"{mean:>11}° {('±%d°' % spread):>7} | " + " ".join(f"{v:7.3f}" for v in r))
    print("(surge, sway, heave in m; roll, pitch, yaw in degrees)")

    # sanity checks
    head, w = direction_weights(30, 40)
    assert abs(w.sum() - 1) < 1e-12, "weights must sum to 1"
    a = rms(response_spectra(hull, omega, S, *direction_weights(30, 40)), d_omega)
    b = rms(response_spectra(hull, omega, S, *direction_weights(-30, 40)), d_omega)
    assert np.allclose(a[[0, 2, 4]], b[[0, 2, 4]]) and np.allclose(a[[1, 3, 5]], b[[1, 3, 5]]), "mirror in y"
    print("checks: weights sum to 1, mirror heading gives the same RMS: PASS")

    # time series must have  variance the spectra predict, on average over seeds
    head, w = direction_weights(30, 40)
    t = np.arange(0, 3000, 0.25)
    spec = rms(response_spectra(hull, omega, S, head, w), d_omega)
    sims, hs_sim = [], []
    for seed in range(6):
        eta, X = time_series(hull, omega, S, head, w, t, seed=seed)
        sd = X.std(axis=0)
        sd[3:] = np.degrees(sd[3:])
        sims.append(sd)
        hs_sim.append(4 * eta.std())
    sims = np.array(sims)
    print(f"\nsimulated wave Hs over 6 seeds: mean {np.mean(hs_sim):.2f} m, range {min(hs_sim):.2f} to {max(hs_sim):.2f} (asked for {Hs})")
    print(f"{'':>14} | " + " ".join(f"{n:>7}" for n in DOF))
    print(f"{'spectral':>14} | " + " ".join(f"{v:7.3f}" for v in spec))
    print(f"{'simulated mean':>14} | " + " ".join(f"{v:7.3f}" for v in sims.mean(axis=0)))
    assert np.allclose(sims.mean(axis=0), spec, rtol=0.04), "mean time-series RMS should match the spectral RMS"
    print("check: time-series RMS (mean of 6 seeds) matches spectral RMS within 4 %: PASS")


if __name__ == "__main__":
    demo()