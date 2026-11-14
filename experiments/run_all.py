"""Runs one experiment family and appends rows to results/metrics/<family>.csv.
Usage: python -m experiments.run_all <family> <datasets,comma> [seeds,comma]"""
import sys, os, json, time
from dataclasses import replace
import numpy as np, pandas as pd
from src.engine import load, run, Cfg
from src.evaluation import summarise
from src.proposed_model import PACT, ABLATIONS
from src.baseline_models import baselines, ORACLE
from src import robustness as rb

fam, dsets = sys.argv[1], sys.argv[2].split(",")
seeds = [int(s) for s in sys.argv[3].split(",")] if len(sys.argv) > 3 else [1, 2, 3, 4, 5]
out = f"results/metrics/{fam}.csv"; os.makedirs("results/logs", exist_ok=True)


from src.results_io import read as _read
_done = {}
if os.path.exists(out):
    _d = _read(fam)
    _ex = [c for c in ("param", "value", "condition", "poison") if c in _d.columns]
    for _, _r in _d.iterrows():
        _done[(_r["dataset"], _r["method"], int(_r["seed"])) + tuple(str(_r[c]) for c in _ex)] = _r.to_dict()


def cached(ds, cfg, seed, extra=None, **kw):
    """Run a configuration unless its row already exists (resumable grid)."""
    extra = extra or {}
    key = (ds["name"], cfg.name, seed) + tuple(str(float(v)) if isinstance(v, (int, float)) else str(v) for v in extra.values())
    if key in _done:
        return _done[key], None
    L = run(ds, cfg, seed, **kw); r = summarise(L); emit(r, **extra)
    return r, L


def emit(row, **extra):
    row.update(extra)
    pd.DataFrame([row]).to_csv(out, mode="a", header=not os.path.exists(out), index=False)
    print(time.strftime("%H:%M:%S"), fam, row["dataset"], row["method"], row["seed"], extra,
          round(row["coverage"], 3), round(row["label_rate"], 3), round(row["sys_acc"], 3), flush=True)


def save_log(L, tag):
    np.savez_compressed(f"results/logs/{L['dataset']}_{tag}_s{L['seed']}.npz", cov=L["cov"], size=L["size"], aud=L["aud"],
                        esc=L["esc"], pred=L["pred"], y=L["y"], block=L["block"], pmax=L["pmax"], hz_in=L["hz_in"])


for name in dsets:
    ds = load(name)
    for seed in seeds:
        if fam == "main":
            r, L = cached(ds, PACT, seed)
            if L is not None:
                save_log(L, "PACT")
            for c in baselines(r["label_rate"]) + [ORACLE]:
                _, L = cached(ds, c, seed)
                if L is not None:
                    save_log(L, c.name.split(" (")[0].replace(" ", "_"))
        elif fam == "ablation":
            r0 = None
            for k, c in ABLATIONS.items():
                if c is None:
                    c = Cfg(name=k, audit="random", budget=r0["label_rate"], ipw=False, adapt=False)
                r, _ = cached(ds, c, seed)
                if k == "PACT (full)":
                    r0 = r
        elif fam == "sensitivity":
            grid = [("eps", v) for v in (0.01, 0.02, 0.05, 0.10, 0.20)] + [("W", v) for v in (100, 300, 1000)] + \
                   [("R", v) for v in (25, 50, 150)] + [("alpha", v) for v in (0.05, 0.10, 0.20)]
            for p, v in grid:
                cached(ds, replace(PACT, **{p: v}), seed, extra=dict(param=p, value=v))
        elif fam == "robust":
            conds = {"clean": None, "noise 0.1": rb.gaussian_noise(0.1), "noise 0.3": rb.gaussian_noise(0.3),
                     "sensor dropout": rb.sensor_dropout(0.3), "evasion 0.5": rb.centroid_evasion(0.5),
                     "evasion 1.0": rb.centroid_evasion(1.0)}
            for cn, pf in conds.items():
                r, _ = cached(ds, PACT, seed, extra=dict(condition=cn), perturb=pf)
                bl = baselines(r["label_rate"])
                for c in (bl[0], bl[3], bl[6]):                    # static split CP, NexCP, random audit
                    cached(ds, c, seed, extra=dict(condition=cn), perturb=pf)
        elif fam == "poison":
            for rho in (0.0, 0.1, 0.2, 0.3):
                r, _ = cached(ds, PACT, seed, extra=dict(poison=rho), poison=rho)
                cached(ds, replace(PACT, name="PACT + screening", screen=True), seed, extra=dict(poison=rho), poison=rho)
                cached(ds, replace(baselines(r["label_rate"])[6], name="Random audit"), seed, extra=dict(poison=rho), poison=rho)
