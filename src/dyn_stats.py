"""Seed-paired t intervals and Holm correction for DynIRS."""
import numpy as np
from scipy import stats


def paired_t(diffs):
    d = np.asarray(diffs, float)
    n, m, sd = len(d), d.mean(), d.std(ddof=1)
    half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    p = float(stats.ttest_1samp(d, 0).pvalue)
    return {"mean": float(m), "sd": float(sd), "n": n, "t_ci": [float(m - half), float(m + half)], "p": p}


def holm(pvals):
    """Holm step-down adjusted p-values. A NaN p-value stays NaN and is ignored for the others:
    it does not count in the multiplier (m is the number of finite p-values) and does not raise
    the running maximum."""
    p = np.asarray(pvals, float)
    ok = ~np.isnan(p)
    m = int(ok.sum())
    order = [i for i in np.argsort(p) if ok[i]]
    adj = np.full_like(p, np.nan)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(run, 1.0)
    return adj.tolist()
