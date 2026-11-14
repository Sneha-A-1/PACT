"""Baseline configurations.  Budget-matched baselines receive the label rate realised by PACT on the
same dataset and seed, and use the same base learner and refresh rule."""
from dataclasses import replace
from .engine import Cfg

_B = Cfg(audit="random", ipw=False, adapt=False)


def baselines(budget):
    return [
        Cfg(name="Split CP (static)", calib="static", audit="none", refresh=False, adapt=False),
        Cfg(name="APS (static)", score="aps", calib="static", audit="none", refresh=False, adapt=False),
        replace(_B, name="ACI", calib="aci", budget=budget),
        replace(_B, name="NexCP", calib="nexcp", budget=budget),
        replace(_B, name="Conformal P-control", calib="pid", budget=budget),
        replace(_B, name="Uncertainty sampling", calib="window", audit="uncert", budget=budget),
        replace(_B, name="Random audit", calib="window", budget=budget),
    ]


ORACLE = Cfg(name="Full feedback (oracle)", calib="window", audit="all", ipw=False, adapt=False)
