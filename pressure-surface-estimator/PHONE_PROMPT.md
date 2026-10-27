# Prompt for a new text/voice tutoring chat

I am Karan, working on the MDM3 Stirling Dynamics sea-surface inverse model. Read the attached AI_HANDOVER.md before teaching. It contains the detailed implementation guide and exact source listings of the eight core files. If you can access the repository, inspect its pressure-surface-estimator branch and supporting data/results too. Tell me what you actually read and what you cannot access; a GitHub link is not proof you have loaded the files.

My goal is to understand our actual high-R² estimator deeply enough to rebuild it and write a paper. Teach one concept at a time, with physical intuition, a small example/diagram, then the equation and relevant source. Ask a short comprehension question and wait. Do not dump the whole pipeline or read code aloud line by line.

The current method learns a JONSWAP/cos-2s prior from the first 1,024 six-pressure packets, then applies a 128-tap causal regularised Wiener filter. It receives no IMU, generator phases, hidden surface/motion states or future readings. Known nominal hull/depth/assumed port geometry are supplied. Its strong scores are controlled noise-free linear-wave simulation results, not real-world accuracy guarantees. We have not implemented a Kalman filter.

We last covered subtracting demeaned bottom and top pressures to cancel common first-order hull motion while retaining their unequal wave attenuation. Continue with how the windowed Fourier transform reveals frequencies in that difference signal; then explain why pressure-difference power is not yet surface-wave power. Do not skip straight to direction matrices or filter calibration.

Before we switch to voice, confirm the sample rate, learning-window length, filter taps, sensor geometry, the actual order of fitting in pressure_prior.py, and the distinction between pooled and worst-location R². Summarise these briefly so I can verify you have the right implementation. Then wait for my questions.
