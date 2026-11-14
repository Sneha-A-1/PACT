"""Explanation layer: exact TreeSHAP attributions of the deployed model and their quality metrics."""
import time
import numpy as np


def tree_shap(model, X, K):
    """Per-class additive attributions phi[n, k, d] of the raw model output (TreeSHAP, exact for tree
    ensembles).  Binary models return one output; it is mirrored for the negative class."""
    C = model.predict(X, pred_contrib=True)
    d = X.shape[1]
    if C.shape[1] == d + 1:
        phi = C[:, :d]
        return np.stack([-phi, phi], 1)
    k = C.shape[1] // (d + 1)
    return C.reshape(len(X), k, d + 1)[:, :, :d]


def _p(model, X, K):
    P = np.full((len(X), K), 1e-6); P[:, model.classes_] = model.predict_proba(X); return P / P.sum(1, keepdims=True)


def faithfulness(model, X, mu, K, fracs=(0.05, 0.1, 0.2), seed=0):
    """Deletion test: drop in the predicted-class probability when the top-attributed features are
    replaced by the training mean, compared with deleting the same number of random features."""
    rng = np.random.RandomState(seed)
    P = _p(model, X, K); c = P.argmax(1); n, d = X.shape
    phi = tree_shap(model, X, K)[np.arange(n), c]
    rank = np.argsort(-np.abs(phi), 1)
    out = {}
    for f in fracs:
        k = max(1, int(round(f * d)))
        Xt = X.copy(); Xr = X.copy()
        for i in range(n):
            Xt[i, rank[i, :k]] = mu[rank[i, :k]]
            r = rng.choice(d, k, replace=False); Xr[i, r] = mu[r]
        out[f] = (float((P[np.arange(n), c] - _p(model, Xt, K)[np.arange(n), c]).mean()),
                  float((P[np.arange(n), c] - _p(model, Xr, K)[np.arange(n), c]).mean()))
    return out


def stability(model, X, sd, K, sigma=0.02, top=5, seed=0):
    """Attribution stability under a small input perturbation: cosine similarity and top-k overlap."""
    rng = np.random.RandomState(seed); n = len(X)
    c = _p(model, X, K).argmax(1)
    a = tree_shap(model, X, K)[np.arange(n), c]
    b = tree_shap(model, X + rng.randn(*X.shape) * sigma * sd, K)[np.arange(n), c]
    cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
    ta = np.argsort(-np.abs(a), 1)[:, :top]; tb = np.argsort(-np.abs(b), 1)[:, :top]
    jac = np.mean([len(set(x) & set(z)) / len(set(x) | set(z)) for x, z in zip(ta, tb)])
    return float(cos.mean()), float(jac)


def explain_time(model, X, K):
    t = time.perf_counter(); tree_shap(model, X, K); return 1e3 * (time.perf_counter() - t) / len(X)
