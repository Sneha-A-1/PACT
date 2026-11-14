"""Public interface of the uncertainty layer (thin wrappers around the engine internals)."""
import numpy as np
from .conformal import lac_scores, aps_scores, weighted_quantile, kish_ess  # noqa: F401


def pact_threshold(scores, propensities, alpha=0.1, window=300):
    """Propensity-weighted conformal threshold from verified scores and their verification propensities."""
    s = np.asarray(scores, float)[-window:]; w = 1.0 / np.asarray(propensities, float)[-window:]
    return weighted_quantile(s, w, alpha)


class CoverageDebt:
    """Exponentially forgetting, propensity-weighted miscoverage estimator and its two actuators."""

    def __init__(self, alpha=0.1, eps=0.05, eps_max=0.30, lam=0.97):
        self.alpha, self.eps, self.eps_max, self.lam = alpha, eps, eps_max, lam
        self.num, self.den = alpha, 1.0

    def update(self, error, propensity):
        w = 1.0 / propensity
        self.num = self.lam * self.num + (1 - self.lam) * w * float(error)
        self.den = self.lam * self.den + (1 - self.lam) * w
        return self.estimate

    @property
    def estimate(self):
        return self.num / self.den

    @property
    def working_alpha(self):
        return float(np.clip(2 * self.alpha - self.estimate, 0.25 * self.alpha, 1.5 * self.alpha))

    @property
    def audit_rate(self):
        return float(min(self.eps_max, self.eps * (1 + max(0.0, (self.estimate - self.alpha) / self.alpha))))
