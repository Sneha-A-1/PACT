"""Biosafety triage: mapping a conformal prediction set to an action and a decision record."""
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class Decision:
    action: str            # "release" or "escalate"
    prediction_set: list   # class indices retained in the conformal set
    point_prediction: int
    target_coverage: float
    risk_flag: bool        # True when a hazard class is in the set
    propensity: float      # probability with which the sample is verified
    top_features: list     # names of the most attributed features (optional)


def triage(scores, q, hazard=-1, eps=0.05, cap=0.6, esc_rate=0.1, alpha=0.1, attribution=None, feature_names=None, top=5):
    """Triage rule of PACT for one sample.

    scores : array (K,) nonconformity scores s(x, k) = 1 - p(k | x)
    q      : array (K,) class-wise conformal thresholds
    Returns a Decision; the caller draws the audit A ~ Bernoulli(propensity).
    """
    scores = np.asarray(scores, float)
    inset = scores <= np.asarray(q, float)
    risk = bool(hazard >= 0 and inset[hazard])
    escalate = (inset.sum() != 1) or risk
    pi = min(1.0, cap / max(esc_rate, 1e-6)) if escalate else eps
    topf = []
    if attribution is not None:
        idx = np.argsort(-np.abs(attribution))[:top]
        topf = [feature_names[i] if feature_names is not None else int(i) for i in idx]
    return Decision("escalate" if escalate else "release", np.where(inset)[0].tolist(), int(scores.argmin()),
                    1.0 - alpha, risk, float(pi), topf)


def as_record(decision, audited=None, verified_label=None):
    """Flat dictionary suitable for an append-only audit log."""
    r = asdict(decision); r.update(audited=audited, verified_label=verified_label)
    return r
