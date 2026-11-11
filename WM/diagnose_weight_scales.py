"""Post-test scale distortion diagnostic; no parameter selection or refitting."""
from pathlib import Path
import numpy as np
from check_profile_weights import profile_smooth
from check_inverse_recovery import candidate_variances
from check_pooled_covariance import GROUPS
from variance_convergence import ROOT,write_csv
import invert_waves as iw

with np.load('variance_output/profile_weights/profile.npz') as z:profile=z['profile']
rows=[]
for group,start in GROUPS.items():
    ratios=[]
    for seed in range(start,start+200):
        with np.load(Path('variance_output/pooled_covariance')/group/f'seed-{seed}.npz') as z:s=z['spectral']
        v=np.stack([candidate_variances(ss,iw.stack_pairs(9),8.785033150237455) for ss in s])
        ratios.append(profile_smooth(v,profile,17)/v)
    ratio=np.stack(ratios)
    assert ratio.shape==(200,53,81) and np.isfinite(ratio).all() and np.all(ratio>0)
    rows.append(dict(group=group,median=float(np.median(ratio)),p95=float(np.quantile(ratio,.95)),
        p99=float(np.quantile(ratio,.99)),maximum=float(ratio.max()),fraction_above_10=float(np.mean(ratio>10)),
        fraction_below_point1=float(np.mean(ratio<.1))))
write_csv(ROOT/'profile_weights_scale_diagnostic.csv',rows)
