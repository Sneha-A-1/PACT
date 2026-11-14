"""Conformal building blocks: nonconformity scores and (weighted) conformal quantiles."""
import numpy as np


def lac_scores(P):
    """Least-ambiguous-classifier score s(x, k) = 1 - p_k(x) for every class k."""
    return 1.0 - P


def aps_scores(P):
    """Adaptive prediction set score (deterministic variant): cumulative probability mass of all
    classes ranked at least as high as k."""
    order = np.argsort(-P, axis=1)
    cum = np.cumsum(np.take_along_axis(P, order, 1), 1)
    S = np.empty_like(P)
    np.put_along_axis(S, order, cum, 1)
    return S


def weighted_quantile(s, w, alpha, test_weight=1.0):
    """Smallest q such that the weighted mass of calibration scores <= q is at least 1 - alpha, with one
    extra point of mass `test_weight` at +infinity (finite-sample correction of weighted conformal
    prediction).  Returns np.inf when the calibration mass is insufficient."""
    s = np.asarray(s, float); w = np.asarray(w, float)
    if len(s) == 0:
        return np.inf
    o = np.argsort(s)
    c = np.cumsum(w[o]) / (w.sum() + test_weight)
    i = np.searchsorted(c, 1.0 - alpha, side="left")
    return s[o][i] if i < len(s) else np.inf


def kish_ess(w):
    w = np.asarray(w, float)
    return (w.sum() ** 2) / (np.square(w).sum() + 1e-12)
