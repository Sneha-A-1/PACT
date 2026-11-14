"""Explainability evaluation of the deployed PACT model (final refreshed model) versus the static model.
Usage: python -m experiments.explain <datasets,comma> [seeds]"""
import sys, os, json
import numpy as np, pandas as pd
from src.engine import load, run, fit_model, make_order
from src.proposed_model import PACT
from src import explainability as ex

rows = []; os.makedirs("results/metrics", exist_ok=True)
for name in sys.argv[1].split(","):
    ds = load(name); K = len(ds["classes"]); feat = list(ds["feat"])
    for seed in [int(s) for s in (sys.argv[2] if len(sys.argv) > 2 else "1,2,3").split(",")]:
        L = run(ds, PACT, seed, return_log=True, keep_model=True)
        X = ds["X"].astype(float); tr = L["tr"]; mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        static = fit_model(X[tr], ds["y"][tr], None, K, seed)
        rng = np.random.RandomState(seed); Xs = L["Xs"]; n = len(Xs)
        idx = np.sort(rng.choice(np.arange(n // 2, n), min(1000, n - n // 2), replace=False))   # second half of the stream
        for tag, m in (("PACT (refreshed)", L["model"]), ("Static model", static)):
            f = ex.faithfulness(m, Xs[idx], mu, K, seed=seed); cos, jac = ex.stability(m, Xs[idx], sd, K, seed=seed)
            rows.append(dict(dataset=name, model=tag, seed=seed, del5_top=f[0.05][0], del5_rand=f[0.05][1], del10_top=f[0.1][0],
                             del10_rand=f[0.1][1], del20_top=f[0.2][0], del20_rand=f[0.2][1], cosine=cos, top5_jaccard=jac,
                             explain_ms=ex.explain_time(m, Xs[idx], K)))
            print(rows[-1], flush=True)
        if seed == 1:
            phi = ex.tree_shap(L["model"], Xs, K); pred = L["pred"]
            A = np.abs(phi[np.arange(n), pred])
            g = pd.Series(A.mean(0), index=feat).sort_values(ascending=False)
            g.to_csv(f"results/metrics/shap_global_{name}.csv", header=["mean_abs_shap"])
            if name == "gas":                       # sensor-level attribution share per batch (16 sensors x 8 features)
                S = A.reshape(n, 16, 8).sum(2); S = S / S.sum(1, keepdims=True)
                st = ex.tree_shap(static, Xs, K); As = np.abs(st[np.arange(n), static.predict(Xs)]).reshape(n, 16, 8).sum(2); As = As / As.sum(1, keepdims=True)
                b = L["block"]
                pd.DataFrame([S[b == k].mean(0) for k in np.unique(b)], index=np.unique(b)).to_csv("results/metrics/shap_sensor_batch_pact.csv")
                pd.DataFrame([As[b == k].mean(0) for k in np.unique(b)], index=np.unique(b)).to_csv("results/metrics/shap_sensor_batch_static.csv")
pd.DataFrame(rows).to_csv("results/metrics/explain.csv", index=False)
