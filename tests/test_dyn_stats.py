import numpy as np
from scipy import stats

import dyn_stats as ds


def test_paired_t_matches_scipy():
    d = np.array([0.4, 0.9, 0.2])
    r = ds.paired_t(d)
    t = stats.ttest_1samp(d, 0)
    assert np.isclose(r["p"], t.pvalue) and np.isclose(r["mean"], d.mean())
    half = stats.t.ppf(0.975, 2) * d.std(ddof=1) / np.sqrt(3)
    assert np.allclose(r["t_ci"], [d.mean() - half, d.mean() + half])


def test_holm():
    assert np.allclose(ds.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    assert np.allclose(ds.holm([0.5, 0.9]), [1.0, 1.0])


def test_holm_propagates_nan_without_affecting_others():
    out = ds.holm([0.01, float("nan"), 0.03])
    assert np.isnan(out[1])
    assert np.allclose([out[0], out[2]], [0.02, 0.03])
    assert all(np.isnan(ds.holm([float("nan")])))
