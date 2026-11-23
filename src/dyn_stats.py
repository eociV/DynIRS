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
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(run, 1.0)
    return adj.tolist()
