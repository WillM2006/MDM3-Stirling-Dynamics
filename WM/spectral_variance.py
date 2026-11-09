"""Experimental diagonal uncertainty estimates for Welch spectra."""
import numpy as np
from scipy.signal import get_window


def effective_averages(window, step, count):
    """Overlap correction for a locally flat Gaussian spectrum."""
    window = np.asarray(window, dtype=float)
    if window.ndim != 1 or not np.isfinite(window).all() or window @ window <= 0:
        raise ValueError("window must have finite, nonzero energy")
    if step < 1 or count < 1:
        raise ValueError("step and count must be positive")
    correction = 1.0
    for lag in range(1, min(count, (len(window) - 1) // step + 1)):
        shift = lag * step
        correlation = window[shift:] @ window[:-shift] / (window @ window)
        correction += 2 * (1 - lag / count) * correlation**2
    return count / correction


def segment_estimates(samples, fs=2.0, seconds=200.0, overlap=0.5):
    """Return frequency, stacked segment spectra, and effective average count.

    Rows follow invert_waves.data_vector: autos, then real/imaginary cross entries.
    Detrending, periodic Hann taper and density scaling match scipy.signal.csd.
    """
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2 or samples.shape[1] < 1 or not np.isfinite(samples).all():
        raise ValueError("samples must be a finite time-by-channel array")
    if not np.isfinite(fs) or not np.isfinite(seconds) or fs <= 0 or seconds <= 0:
        raise ValueError("sample rate and segment duration must be positive")
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be between zero and one")
    length = int(round(seconds * fs))
    step = length - int(length * overlap)
    if length < 2 or len(samples) < length:
        raise ValueError("recording is shorter than one segment")
    window = get_window("hann", length, fftbins=True)
    segments = np.stack([samples[start:start + length] for start in range(0, len(samples) - length + 1, step)])
    segments = segments - segments.mean(axis=1, keepdims=True)
    fourier = np.fft.rfft(segments * window[None, :, None], axis=1)
    scale = np.full(fourier.shape[1], 2 / (fs * (window @ window)))
    scale[0] /= 2
    if length % 2 == 0:
        scale[-1] /= 2
    entries = []
    for c in range(samples.shape[1]):
        for k in range(c, samples.shape[1]):
            spectrum = fourier[:, :, c].conj() * fourier[:, :, k] * scale
            entries.append(spectrum.real)
            if c != k:
                entries.append(spectrum.imag)
    values = np.stack(entries, axis=-1)
    return np.fft.rfftfreq(length, 1 / fs), values, effective_averages(window, step, len(segments))


def recording_variance(segment_values, averages):
    """Estimate variance of the averaged spectrum from its segment variation.

    The correction also accounts for the bias in segment sample variance caused
    by overlap. It assumes the same correlation correction for each entry.
    """
    values = np.asarray(segment_values)
    count = len(values)
    if values.ndim != 3 or count < 2 or not np.isfinite(values).all():
        raise ValueError("at least two finite segment estimates are required")
    if not np.isfinite(averages) or not 1 < averages <= count:
        raise ValueError("effective averages must be greater than one and at most the segment count")
    return values.var(axis=0, ddof=1) * (count - 1) / (count * (averages - 1))
