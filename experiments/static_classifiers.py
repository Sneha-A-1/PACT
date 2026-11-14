"""Conventional and uncertainty-aware static classifiers (no deployment-time feedback).
Trained on the initial pool, model-selected on its validation split, evaluated on the deployment stream.
Usage: python -m experiments.static_classifiers <datasets,comma> [seeds,comma]"""
import sys, os, time, warnings
import numpy as np, pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score, average_precision_score, log_loss
from scipy.optimize import minimize_scalar
import lightgbm as lgb
from src.engine import load, make_order
from src.evaluation import ece
warnings.filterwarnings("ignore")


def grids(seed):
    S = StandardScaler
    return {
        "Logistic regression": [make_pipeline(S(), LogisticRegression(C=c, max_iter=2000)) for c in (0.1, 1, 10)],
        "SVM (RBF)": [make_pipeline(S(), SVC(C=c, gamma=g, probability=True, random_state=seed)) for c in (1, 10) for g in ("scale", 0.01)],
        "k-NN": [make_pipeline(S(), KNeighborsClassifier(k)) for k in (3, 7, 15)],
        "Random forest": [RandomForestClassifier(300, max_features=m, n_jobs=1, random_state=seed) for m in ("sqrt", 0.3)],
        "MLP": [make_pipeline(S(), MLPClassifier((h, h), alpha=a, max_iter=500, early_stopping=True, random_state=seed)) for h in (64, 128) for a in (1e-4, 1e-2)],
        "LightGBM": [lgb.LGBMClassifier(n_estimators=n, learning_rate=0.1, num_leaves=l, colsample_bytree=0.5, min_child_samples=5, verbose=-1, random_state=seed, n_jobs=1) for n in (80, 300) for l in (15, 31)],
    }


def full(P, classes, K):
    Q = np.full((len(P), K), 1e-6); Q[:, classes] = P; return Q / Q.sum(1, keepdims=True)


def metrics(P, y, K, hz):
    pred = P.argmax(1); c = pred == y
    r = dict(accuracy=c.mean(), macro_f1=f1_score(y, pred, average="macro"), mcc=matthews_corrcoef(y, pred),
             ece=ece(P.max(1), c), brier=((P - np.eye(K)[y]) ** 2).sum(1).mean(), nll=log_loss(y, P, labels=list(range(K))))
    try:
        r["roc_auc"] = roc_auc_score(y, P[:, 1]) if K == 2 else roc_auc_score(y, P, multi_class="ovr", labels=list(range(K)))
    except ValueError:
        r["roc_auc"] = np.nan
    if hz >= 0:
        h = y == hz
        r["pr_auc"] = average_precision_score(h, P[:, hz]); r["sensitivity"] = (pred[h] == hz).mean(); r["specificity"] = (pred[~h] != hz).mean()
    return r


out = "results/metrics/static.csv"
for name in sys.argv[1].split(","):
    ds = load(name); X, y = ds["X"].astype(float), ds["y"]; K = len(ds["classes"]); hz = int(ds["hazard"])
    for seed in [int(s) for s in (sys.argv[2] if len(sys.argv) > 2 else "1,2,3,4,5").split(",")]:
        init, stream = make_order(ds, seed)
        rng = np.random.RandomState(1000 + seed); p = rng.permutation(len(init)); ntr = int(0.7 * len(init))
        tr, va = init[p[:ntr]], init[p[ntr:]]
        rows = []
        for mname, cands in grids(seed).items():
            best, bl = None, np.inf
            for m in cands:
                m.fit(X[tr], y[tr]); l = log_loss(y[va], full(m.predict_proba(X[va]), m.classes_, K), labels=list(range(K)))
                if l < bl:
                    best, bl = m, l
            t = time.perf_counter(); P = full(best.predict_proba(X[stream]), best.classes_, K); lat = 1e3 * (time.perf_counter() - t) / len(stream)
            rows.append(dict(dataset=name, method=mname, seed=seed, latency_ms=lat, **metrics(P, y[stream], K, hz)))
        # deep ensemble of five MLPs with temperature scaling on the validation split
        ms = [make_pipeline(StandardScaler(), MLPClassifier((128, 128), max_iter=500, early_stopping=True, random_state=seed * 10 + j)).fit(X[tr], y[tr]) for j in range(5)]
        ens = lambda Z: np.mean([full(m.predict_proba(Z), m.classes_, K) for m in ms], 0)
        Pv = ens(X[va]); temp = lambda P, T: (lambda L: L / L.sum(1, keepdims=True))(np.exp(np.log(P) / T))
        T = minimize_scalar(lambda T: log_loss(y[va], temp(Pv, T), labels=list(range(K))), bounds=(0.05, 20), method="bounded").x
        t = time.perf_counter(); P = temp(ens(X[stream]), T); lat = 1e3 * (time.perf_counter() - t) / len(stream)
        rows.append(dict(dataset=name, method="Deep ensemble + TS", seed=seed, latency_ms=lat, **metrics(P, y[stream], K, hz)))
        pd.DataFrame(rows).to_csv(out, mode="a", header=not os.path.exists(out), index=False)
        print(name, seed, {r["method"]: round(r["accuracy"], 3) for r in rows}, flush=True)
