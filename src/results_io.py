"""Robust reader for the append-only metric files (rows of hazard tasks carry seven extra columns,
poisoning runs two more; the header is written by whichever run starts first)."""
import csv
import pandas as pd

BASE = ["dataset", "method", "seed", "n", "coverage", "set_size", "singleton", "empty", "label_rate", "escalation", "accuracy",
        "macro_f1", "mcc", "ece", "brier", "nll", "auto_acc", "sys_acc", "latency_ms", "update_ms", "e2e_ms", "throughput",
        "train_time", "rss_mb", "n_refresh", "worst_block_cov", "cov_gap"]
HAZ = ["hazard_cov", "safe_cov", "missed_risk", "false_alarm", "sensitivity", "specificity", "risk_sens"]
POI = ["poison_detect", "poison_fpr"]
EXTRA = {"main": [], "ablation": [], "sensitivity": ["param", "value"], "robust": ["condition"], "poison": ["poison"]}
HAZARD_SETS = {"ferment", "water"}


def read(family, root="results/metrics"):
    rows = []
    ex = EXTRA[family]
    with open(f"{root}/{family}.csv") as f:
        for r in csv.reader(f):
            if not r or r[0] == "dataset":
                continue
            cols = BASE + (HAZ if r[0] in HAZARD_SETS else [])
            k = len(r) - len(cols) - len(ex)
            cols = cols + (POI if k == 2 else []) + ex
            assert len(cols) == len(r), (family, len(cols), len(r), r[:3])
            rows.append(dict(zip(cols, r)))
    d = pd.DataFrame(rows)
    for c in d.columns:
        if c not in ("dataset", "method", "param", "condition"):
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def read_static(path="results/metrics/static.csv"):
    base = ["dataset", "method", "seed", "latency_ms", "accuracy", "macro_f1", "mcc", "ece", "brier", "nll", "roc_auc"]
    rows = []
    with open(path) as f:
        for r in csv.reader(f):
            if not r or r[0] == "dataset":
                continue
            cols = base + (["pr_auc", "sensitivity", "specificity"] if len(r) == 14 else [])
            assert len(cols) == len(r), r[:3]
            rows.append(dict(zip(cols, r)))
    d = pd.DataFrame(rows)
    for c in d.columns:
        if c not in ("dataset", "method"):
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d
