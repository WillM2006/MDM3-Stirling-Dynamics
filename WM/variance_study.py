"""Compare fixed and recording-derived diagonal Welch variances.

Input: folders of native sensor CSVs, or preprocessed NPZ recordings with X.
No wave truth, response functions or fitted calibration constants are used.
"""
import argparse
import csv
from pathlib import Path
import numpy as np
from scipy.signal import lfilter
import invert_waves as iw
from spectral_variance import segment_estimates, recording_variance


def estimates(samples):
    f, pieces, count = segment_estimates(samples)
    selection = (f >= iw.BAND[0]) & (f <= iw.BAND[1])
    _, spectral = iw.welch_matrix(samples)
    pairs = iw.stack_pairs(samples.shape[1])
    observed = np.stack([iw.data_vector(s, pairs) for s in spectral])
    np.testing.assert_allclose(pieces.mean(0), observed, rtol=1e-10, atol=1e-9)
    fixed = np.stack([iw.noise_variances(s, pairs, 2.5) for s in spectral])
    overlap = np.stack([iw.noise_variances(s, pairs, count) for s in spectral])
    _, independent, _ = segment_estimates(samples, overlap=0)
    return f[selection], observed[selection], {
        "fixed_2.5": fixed[selection],
        "overlap_count": overlap[selection],
        "segment_variance": recording_variance(pieces, count)[selection],
        "nonoverlap_segments": independent.var(0, ddof=1)[selection] / count,
    }


def summarise(group, observations, predictions, frequencies, auto_columns, bootstrap=600):
    observations = np.asarray(observations)
    n = len(observations)
    if n < 8:
        raise ValueError(f"{group}: need at least eight repeated recordings, got {n}")
    variance = observations.var(0, ddof=1)
    rng = np.random.default_rng(39281)
    weights = rng.multinomial(n, np.full(n, 1 / n), size=bootstrap) / n
    mu = np.einsum("bn,nfe->bfe", weights, observations)
    bootvar = (np.einsum("bn,nfe->bfe", weights, observations**2) - mu**2) * n / (n - 1)
    rows = []
    for method, values in predictions.items():
        values = np.asarray(values)
        average = values.mean(0)
        bootpred = np.einsum("bn,nfe->bfe", weights, values)
        for band, selected in [("whole", np.ones(len(frequencies), dtype=bool)),
                               ("0.10-0.16", (frequencies >= .10) & (frequencies < .16)),
                               ("0.16-0.26", (frequencies >= .16) & (frequencies < .26))]:
            actual = variance[selected][:, auto_columns]
            predicted = average[selected][:, auto_columns]
            valid = actual > 0
            ratio = predicted[valid] / actual[valid]
            bratio = (bootpred[:, selected][:, :, auto_columns] / np.maximum(bootvar[:, selected][:, :, auto_columns], np.finfo(float).tiny))[:, valid]
            median_boot = np.median(bratio, axis=1)
            low, high = np.quantile(median_boot, [.025, .975])
            rows.append(dict(case=group, method=method, band_Hz=band, recordings=n,
                             median_predicted_over_actual=float(np.median(ratio)),
                             ci95_low=float(low), ci95_high=float(high),
                             median_factor_error=float(np.exp(np.median(abs(np.log(np.maximum(ratio, 1e-30)))))),
                             fraction_within_factor_two=float(np.mean((ratio >= .5) & (ratio <= 2)))))
    return rows


def read_groups(root):
    groups = {}
    conditions = {}
    for path in sorted(root.rglob("*.npz")):
        with np.load(path) as z:
            if "X" not in z:
                continue
            if "t" in z and not np.allclose(np.diff(z["t"]), 1 / iw.FS):
                raise ValueError(f"{path}: preprocessed recordings must use {iw.FS} Hz")
            groups.setdefault(path.parent.name, []).append(z["X"])
    for path in sorted(root.rglob("pressure.csv")):
        group = path.parent.parent.name
        with (path.parent / "params.csv").open() as fh:
            params = next(csv.DictReader(fh))
        condition = {k: float(v) for k, v in params.items() if k not in ("seed", "Hs_actual")}
        if group in conditions and condition != conditions[group]:
            raise ValueError(f"{group}: recordings have different simulation conditions")
        conditions[group] = condition
        _, samples = iw.read_channels(str(path.parent), iw.CHANNELS)
        groups.setdefault(group, []).append(samples)
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--control", action="store_true", help="independent correlated Gaussian AR(1) control")
    args = parser.parse_args()
    if not args.control and args.root is None:
        parser.error("supply --root or --control")
    if args.control:
        rng = np.random.default_rng(57281)
        records = []
        for _ in range(1000):
            noise = rng.normal(size=(2000, 3))
            samples = lfilter([1], [1, -.8], noise, axis=0)[200:]
            samples[:, 1] += .6 * samples[:, 0]
            records.append(samples)
        groups = {"gaussian_control": records}
    else:
        groups = read_groups(args.root)
    if not groups:
        raise ValueError("no sensor recordings found")
    rows = []
    for name, records in groups.items():
        observations, predictions = [], {}
        for samples in records:
            f, observed, predicted = estimates(samples)
            # Convert triangular pair indices to positions in the real data vector.
            auto_columns, position = [], 0
            for c, k in iw.stack_pairs(samples.shape[1]):
                if c == k:
                    auto_columns.append(position)
                position += 1 if c == k else 2
            observations.append(observed)
            for method, values in predicted.items():
                predictions.setdefault(method, []).append(values)
        rows.extend(summarise(name, observations, predictions, f, auto_columns))
        print(f"{name}: {len(records)} recordings", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
