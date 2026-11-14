"""Controlled, defensive stress tests applied to the deployment stream (never to the initial pool).
All perturbations are generic signal-level corruptions of sensor features."""
import numpy as np


def gaussian_noise(sigma):
    """Additive sensor noise with standard deviation sigma x (training standard deviation)."""
    def f(X, y, c):
        return X + c["rng"].randn(*X.shape) * sigma * c["sd"]
    return f


def sensor_dropout(p=0.3, group=8, frac=0.1):
    """With probability p a sample loses one sensor: its features are replaced by the training mean
    (mean imputation of a missing channel).  `group` consecutive features form one sensor when the
    dimensionality is a multiple of `group`; otherwise a random 10 % of the features is dropped."""
    def f(X, y, c):
        X = X.copy(); n, d = X.shape; rng = c["rng"]
        for i in np.where(rng.rand(n) < p)[0]:
            if d % group == 0 and d // group >= 4:
                s = rng.randint(d // group); idx = np.arange(s * group, (s + 1) * group)
            else:
                idx = rng.choice(d, max(1, int(frac * d)), replace=False)
            X[i, idx] = c["mu"][idx]
        return X
    return f


def centroid_evasion(eps, frac=0.3):
    """Bounded evasion perturbation: a fraction of samples is moved by eps training standard deviations
    per feature (sign step) towards the centroid of the nearest other class."""
    def f(X, y, c):
        X = X.copy(); rng = c["rng"]; sd = c["sd"]
        cls = np.unique(c["ytr"]); cen = np.stack([c["Xtr"][c["ytr"] == k].mean(0) for k in cls])
        for i in np.where(rng.rand(len(X)) < frac)[0]:
            d = (((cen - X[i]) / sd) ** 2).sum(1); d[cls == y[i]] = np.inf
            X[i] = X[i] + eps * sd * np.sign(cen[d.argmin()] - X[i])
        return X
    return f
