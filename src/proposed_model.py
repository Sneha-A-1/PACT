"""PACT: Propensity-weighted Audit-fed Conformal Triage (proposed method) and its ablations."""
from dataclasses import replace
from .engine import Cfg, run

PACT = Cfg(name="PACT")

ABLATIONS = {
    "PACT (full)": PACT,
    "w/o propensity weighting": replace(PACT, name="w/o propensity weighting", ipw=False),
    "w/o random audits": replace(PACT, name="w/o random audits", eps=0.0),
    "w/o model refresh": replace(PACT, name="w/o model refresh", refresh=False),
    "w/o recalibration": replace(PACT, name="w/o recalibration", recal=False),
    "w/o recency window": replace(PACT, name="w/o recency window", W=10 ** 9),
    "w/o coverage-debt controller": replace(PACT, name="w/o coverage-debt controller", adapt=False),
    "w/o targeted escalation": None,   # random audits at the PACT label rate; filled at run time
}


def run_pact(ds, seed=0, **kw):
    return run(ds, PACT, seed, **kw)
