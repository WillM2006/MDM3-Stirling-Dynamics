# Detailed sea state: KD/LR

Python/Matplotlib learning examples for constructing a linear sea surface from a JONSWAP spectrum and understanding directional spreading.

## Run the active example

Requires Python, NumPy and Matplotlib, with a desktop plotting backend.

```sh
python -m pip install -r requirements.txt
python wave_formation_1.py
```

The active example compares a unidirectional wave system with the same spectral energy distributed across multiple directions around +y.

- The direction fan shows travel directions; ray length is proportional to relative energy per direction bin.
- The bar chart shows energy fractions across direction bins. They sum to 100%.
- The two surface plots compare unidirectional and directionally spread motion.
- The spread slider sets the angular half-width around +y.
- **Isolate direction: one wavelength** uses a single frequency to show only the effect of directional spreading. Its total variance is held equal to the full-spectrum case for comparison.
- **Full JONSWAP: 150 frequencies** synthesizes all selected frequency bins.
- Space pauses/resumes; R restarts time.

The surface plots keep identical fixed limits. Vertical display scale is exaggerated by 10x for clarity; physical elevation values are unchanged. The common colour scale shows elevation.

## Spectrum to surface

A spectrum is an energy distribution, not an individual wave. A frequency-direction pair creates one sinusoidal component:

```text
S(omega) -> bin variance E_i = S(omega_i) * delta_omega
         -> directional fractions p_j, with sum(p_j) = 1
         -> amplitude A_ij = sqrt(2 * E_i * p_j)
         -> eta_ij = A_ij * cos(kx*x + ky*y - omega_i*t + phase_ij)
         -> eta = sum(eta_ij)
```

Here spectral "energy" is surface-elevation variance in square metres; physical wave energy per unit horizontal area is proportional to it. Random phases produce one repeatable realization of the chosen spectrum.

The full spectrum is normalized so that `sum(S * delta_omega) = (Hs / 4)**2`. This gives the specified spectral significant wave height `Hs = 4*sqrt(m0)`. The normalization is the spectrum's overall scale (equivalent to choosing its alpha); it preserves the shape and peak. `Tp` sets peak period and `gamma` controls peak enhancement.

Directional spreading redistributes energy rather than duplicating a full sea state for every direction. Component amplitudes scale with the square root of the directional energy fraction.

The active settings are `Hs=3 m`, `Tp=8 s`, `gamma=3.3`, 150 frequencies between 0.2 and 2.0 rad/s and 25 candidate direction bins. A narrower angular support activates fewer bins. These are teaching settings, not a measured or sponsor-confirmed sea state. There is no universal discretization count: frequency range, peak resolution, spatial/time resolution and convergence must be checked before relying on a simulation.

## Modelling scope

This is synthetic linear deep-water wave superposition, using `k = omega**2 / g`. It does not model breaking, nonlinear crest asymmetry, shallow-water dispersion, a submerged vehicle's response or validated reconstruction performance. The isolated-frequency mode is specifically an intuition aid, not a full irregular sea.

## Archived examples

`deprecated/` contains earlier demonstrations retained for reference:

- `two_full_spectra.py`: two full sampled spectra in perpendicular directions, each normalized independently, and their sum.
- `jonswap_sample_vs_full.py`: one frequency sample versus an accumulating full-spectrum sum.
- `single-jonswap-component.html`: the earlier chat visual fragment; it is not a standalone browser document.

Only the active example runs when launching `wave_formation_1.py`. Archived examples are not imported. The sample-versus-full archive retains the earlier spectrum scaling; its nominal `Hs` does not equal the resulting spectral height. Use the active example or `two_full_spectra.py` for the corrected normalization.

## References

- [Orcina: wave theory, spectrum discretization and directional spreading](https://www.orcina.com/webhelp/OrcaFlex/Content/html/Wavetheory.htm)
- [Orcina: random-wave data](https://www.orcina.com/webhelp/OrcaFlex/Content/html/Environment%2CDataforrandomwaves.htm)
