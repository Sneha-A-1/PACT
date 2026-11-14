"""Metric computation from an engine log."""
import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score, average_precision_score


def ece(pmax, correct, bins=15):
    e = 0.0; edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (pmax > lo) & (pmax <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - pmax[m].mean())
    return float(e)


def summarise(L):
    y, pred, aud = L["y"], L["pred"], L["aud"]
    correct = pred == y; n = L["n"]
    out = dict(dataset=L["dataset"], method=L["cfg"]["name"], seed=L["seed"], n=n,
               coverage=L["cov"].mean(), set_size=L["size"].mean(), singleton=(L["size"] == 1).mean(),
               empty=(L["size"] == 0).mean(), label_rate=aud.mean(), escalation=L["esc"].mean(),
               accuracy=correct.mean(), macro_f1=f1_score(y, pred, average="macro"),
               mcc=matthews_corrcoef(y, pred), ece=ece(L["pmax"], correct), brier=L["brier"].mean(),
               nll=float(-np.log(np.clip(L["ptrue"], 1e-12, 1)).mean()),
               auto_acc=correct[~aud].mean() if (~aud).any() else np.nan,
               sys_acc=(correct | aud).mean(),
               latency_ms=1e3 * L["t_dec"] / n, update_ms=1e3 * L["t_upd"] / n,
               e2e_ms=1e3 * (L["t_dec"] + L["t_upd"]) / n, throughput=n / (L["t_dec"] + L["t_upd"]),
               train_time=L["train_time"], rss_mb=L["rss_mb"], n_refresh=L["n_refresh"])
    seg = L["block"] if L["dataset"] == "gas" else (np.arange(n) * 10 // n)   # gas: batches; others: stream deciles
    bc = [L["cov"][seg == b].mean() for b in np.unique(seg)]
    out["worst_block_cov"] = float(np.min(bc)); out["cov_gap"] = float(np.mean(np.abs(np.array(bc) - (1 - L["cfg"]["alpha"]))))
    hz = L["hazard"]
    if hz >= 0:
        h = y == hz
        out["hazard_cov"] = L["cov"][h].mean()
        out["safe_cov"] = L["cov"][~h].mean()
        missed = h & ~aud & (pred != hz)                 # hazard sample released automatically as safe
        out["missed_risk"] = missed.sum() / max(h.sum(), 1)
        out["false_alarm"] = (L["esc"] & ~h).sum() / max((~h).sum(), 1)
        out["sensitivity"] = ((pred == hz) & h).sum() / max(h.sum(), 1)
        out["specificity"] = ((pred != hz) & ~h).sum() / max((~h).sum(), 1)
        out["risk_sens"] = 1 - out["missed_risk"]
    if L["poisoned"].any() or L["flagged"].any():
        pz, fl = L["poisoned"], L["flagged"]
        out["poison_detect"] = (pz & fl).sum() / max(pz.sum(), 1)
        out["poison_fpr"] = (fl & ~pz).sum() / max((aud & ~pz).sum(), 1)
    return {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in out.items()}
