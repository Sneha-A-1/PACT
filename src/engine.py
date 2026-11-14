"""Unified prequential (test-then-verify) streaming engine.

Every method evaluated in the study -- PACT, its ablations and all baselines -- is a configuration of
this engine, which guarantees identical data partitions, preprocessing, base learner, label budget and
metric computation across methods.
"""
from dataclasses import dataclass, asdict, replace
import time, resource
import numpy as np
import lightgbm as lgb
from .conformal import lac_scores, aps_scores, weighted_quantile


@dataclass
class Cfg:
    name: str = "PACT"
    alpha: float = 0.10          # target miscoverage
    score: str = "lac"           # lac | aps
    calib: str = "window"        # static | window | aci | nexcp | pid
    audit: str = "targeted"      # none | random | targeted | uncert | all
    budget: float = 0.0          # label budget of budget-matched baselines
    eps: float = 0.05            # random-audit rate of auto-accepted samples (PACT)
    ipw: bool = True             # inverse-propensity weighting of verified samples
    refresh: bool = True         # model refresh from verified samples
    R: int = 50                  # refresh period (number of new verified samples)
    M: int = 1500                # refresh memory (most recent verified samples)
    W: int = 300                 # calibration window (verified samples; per class if label-conditional)
    w_max: float = 10.0          # weight clipping in model refresh
    init_w: float = 0.2          # weight of the initial training pool in a refresh
    cap: float = 0.6             # verification capacity (max. escalation-audit rate)
    hazard_escalate: bool = True # sets containing the hazard class are escalated
    screen: bool = False         # consensus screening of verified labels (feedback-poisoning defence)
    gamma: float = 0.01          # ACI step
    rho: float = 0.99            # NexCP decay
    eta: float = 0.05            # quantile-tracking (P-control) step
    chunk: int = 20              # samples processed per decision step
    n_estimators: int = 80
    recal: bool = True           # recalibrate thresholds from verified samples (ablation switch)
    adapt: bool = True           # coverage-debt controller (adaptive audit rate and level)
    eps_max: float = 0.30        # upper bound of the adaptive random-audit rate
    decay: float = 1.0           # recency decay over the verified sequence inside the window
    refresh_ipw: bool = True     # use propensity weights in the model refresh (else unweighted)
    lam: float = 0.97            # forgetting factor of the coverage-debt estimator


def load(name, root="datasets/processed"):
    z = np.load(f"{root}/{name}.npz", allow_pickle=False)
    return {k: z[k] for k in z.files} | {"name": name}


def fit_model(X, y, w, K, seed, n_estimators=80):
    m = lgb.LGBMClassifier(n_estimators=n_estimators, learning_rate=0.1, num_leaves=15, colsample_bytree=0.5,
                           min_child_samples=5, verbose=-1, random_state=seed, n_jobs=1)
    m.fit(X, y, sample_weight=w)
    return m


def proba(m, X, K):
    P = np.full((len(X), K), 1e-6)
    P[:, m.classes_] = m.predict_proba(X)
    return P / P.sum(1, keepdims=True)


def make_order(ds, seed):
    """Initial pool and ordered stream indices for a given seed."""
    rng = np.random.RandomState(seed)
    y, blk, init, mode = ds["y"], ds["block"], ds["init"].copy(), int(ds["shuffle"])
    if mode == 2:                                    # wine: two bottles per class form the initial pool
        for k in np.unique(y):
            b = rng.permutation(np.unique(blk[y == k]))[:2]
            init |= np.isin(blk, b)
        stream = rng.permutation(np.where(~init)[0])
        return np.where(init)[0], stream
    stream = np.where(~init)[0]
    if mode == 1:                                    # unknown within-block order: permute inside each block
        stream = np.concatenate([rng.permutation(stream[blk[stream] == b]) for b in np.unique(blk[stream])])
    return np.where(init)[0], stream


def run(ds, cfg, seed=0, perturb=None, poison=0.0, return_log=False, keep_model=False):
    rng = np.random.RandomState(1000 + seed)
    X, y = ds["X"].astype(np.float64), ds["y"]
    K = len(ds["classes"]); hz = int(ds["hazard"]); mond = hz >= 0
    init, stream = make_order(ds, seed)
    p = rng.permutation(len(init)); ntr = int(0.7 * len(init))
    tr, ca = init[p[:ntr]], init[p[ntr:]]
    t0 = time.perf_counter()
    model = fit_model(X[tr], y[tr], None, K, seed, cfg.n_estimators)
    train_time = time.perf_counter() - t0
    score_fn = lac_scores if cfg.score == "lac" else aps_scores
    Sc = score_fn(proba(model, X[ca], K))
    # verified-sample store (prequential scores)
    cs = list(Sc[np.arange(len(ca)), y[ca]]); cw = [1.0] * len(ca); cy = list(y[ca])
    bx, by, bw = [], [], []                          # refresh buffer
    sx = [X[tr]]; sy = [y[tr]]                       # screening reference
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9

    def thresholds(alpha_vec=None):
        s = np.array(cs); w = np.array(cw); yy = np.array(cy); n = len(s)
        q = np.empty(K)
        groups = [np.where(yy == k)[0] for k in range(K)] if mond else [np.arange(n)] * K
        for k in range(K):
            g = groups[k]
            a = cfg.alpha if alpha_vec is None else alpha_vec[k]
            if cfg.calib == "static":
                g = g[g < len(ca)]; ww = np.ones(len(g))
            elif cfg.calib == "nexcp":
                age = (n - 1 - g) if not mond else np.arange(len(g))[::-1]
                ww = cfg.rho ** age
            else:
                g = g[-cfg.W:]; ww = w[g] if cfg.ipw else np.ones(len(g))
                if cfg.decay < 1.0:
                    ww = ww * cfg.decay ** np.arange(len(g))[::-1]
            q[k] = weighted_quantile(s[g], ww, a) if len(g) >= 10 else np.inf
            if not mond:
                q[:] = q[k]; break
        return q

    q_static = thresholds()
    q = q_static.copy()
    a_t = np.full(K, cfg.alpha)                      # ACI state
    q_pid = np.where(np.isfinite(q_static), q_static, 1.0)
    tau = 0.2; r_esc = 0.1; since = 0; n_refresh = 0
    G = K if mond else 1
    num = np.full(G, cfg.alpha); den = np.ones(G); eps_t = cfg.eps; eps_sum = 0.0
    eps_ref = max(cfg.eps, 1e-3)
    Xs, ys = X[stream].copy(), y[stream]
    if perturb is not None:
        Xs = perturb(Xs, ys, dict(mu=mu, sd=sd, Xtr=X[tr], ytr=y[tr], rng=np.random.RandomState(7 + seed)))
    n = len(ys)
    L = dict(cov=np.zeros(n, bool), size=np.zeros(n, int), aud=np.zeros(n, bool), esc=np.zeros(n, bool),
             pred=np.zeros(n, int), ptrue=np.zeros(n), pmax=np.zeros(n), brier=np.zeros(n),
             hz_in=np.zeros(n, bool), flagged=np.zeros(n, bool), poisoned=np.zeros(n, bool))
    Pall = np.zeros((n, K)) if return_log else None
    t_dec = 0.0; t_upd = 0.0
    for i in range(0, n, cfg.chunk):
        sl = slice(i, min(n, i + cfg.chunk)); xb, yb = Xs[sl], ys[sl]; m_ = len(yb)
        t1 = time.perf_counter()
        P = proba(model, xb, K); S = score_fn(P)
        if cfg.calib in ("static",) or not cfg.recal:
            q = q_static
        elif cfg.calib == "aci":
            q = thresholds(np.clip(a_t, 1e-3, 0.999))
        elif cfg.calib == "pid":
            q = q_pid
        elif cfg.adapt:
            mhat = num / den                              # propensity-weighted miscoverage estimate
            a_eff = np.clip(2 * cfg.alpha - mhat, 0.25 * cfg.alpha, 1.5 * cfg.alpha)
            q = thresholds(np.repeat(a_eff, K) if not mond else a_eff)
            eps_t = float(min(cfg.eps_max, cfg.eps * (1 + max(0.0, (mhat.max() - cfg.alpha) / cfg.alpha))))
        else:
            q = thresholds()
        eps_sum += eps_t * m_
        sets = S <= q[None, :]
        size = sets.sum(1)
        esc = size != 1
        if mond and cfg.hazard_escalate:
            esc |= sets[:, hz]
        # ---- verification (audit) policy
        if cfg.audit == "none":
            pi = np.zeros(m_)
        elif cfg.audit == "all":
            pi = np.ones(m_)
        elif cfg.audit == "random":
            pi = np.full(m_, cfg.budget)
        elif cfg.audit == "uncert":
            pi = (1 - P.max(1) >= tau).astype(float)
        else:
            r_esc = 0.98 * r_esc + 0.02 * esc.mean()
            pe = min(1.0, cfg.cap / max(r_esc, 1e-6))
            pi = np.where(esc, pe, eps_t)
        aud = rng.rand(m_) < pi
        if cfg.audit == "uncert":
            tau = float(np.clip(tau + 0.02 * (aud.mean() - cfg.budget), 0.0, 1.0)); pi = np.ones(m_)
        t_dec += time.perf_counter() - t1
        cov = sets[np.arange(m_), yb]
        L["cov"][sl] = cov; L["size"][sl] = size; L["aud"][sl] = aud; L["esc"][sl] = esc
        L["pred"][sl] = P.argmax(1); L["ptrue"][sl] = P[np.arange(m_), yb]; L["pmax"][sl] = P.max(1)
        oh = np.eye(K)[yb]; L["brier"][sl] = ((P - oh) ** 2).sum(1)
        if mond:
            L["hz_in"][sl] = sets[:, hz]
        if return_log:
            Pall[sl] = P
        # ---- feedback from verified samples
        t2 = time.perf_counter()
        for j in np.where(aud)[0]:
            yo = yb[j]
            if poison > 0 and rng.rand() < poison:
                yo = rng.choice([k for k in range(K) if k != yb[j]]); L["poisoned"][i + j] = True
            if cfg.screen:
                R_x = np.vstack(sx[-8:]); R_y = np.concatenate(sy[-8:])
                dd = (((R_x - xb[j]) / sd) ** 2).sum(1)
                nb = R_y[np.argpartition(dd, min(7, len(dd) - 1))[:7]]
                if (nb == yo).sum() <= 1 and P[j, yo] < 0.05 and len(nb) >= 7 and (nb == nb[0]).mean() >= 0.7:
                    L["flagged"][i + j] = True
                    continue                          # quarantined: not used for calibration or refresh
            err = not sets[j, yo]
            if cfg.calib == "aci":
                a_t[yo if mond else slice(None)] += cfg.gamma * (cfg.alpha - err)
            if cfg.adapt:
                g_ = yo if mond else 0
                num[g_] = cfg.lam * num[g_] + (1 - cfg.lam) * err / pi[j] * eps_ref
                den[g_] = cfg.lam * den[g_] + (1 - cfg.lam) / pi[j] * eps_ref
            if cfg.calib == "pid":
                q_pid[yo if mond else slice(None)] += cfg.eta * (err - cfg.alpha)
            cs.append(S[j, yo]); cw.append(1.0 / pi[j]); cy.append(yo)
            bx.append(xb[j]); by.append(yo); bw.append(1.0 / pi[j]); since += 1
        if cfg.refresh and since >= cfg.R:
            since = 0; n_refresh += 1
            nx = np.array(bx[-cfg.M:]); ny = np.array(by[-cfg.M:])
            nw = np.minimum(np.array(bw[-cfg.M:]), cfg.w_max) if (cfg.ipw and cfg.refresh_ipw) else np.ones(len(ny))
            model = fit_model(np.vstack([X[tr], nx]), np.concatenate([y[tr], ny]),
                              np.concatenate([np.full(len(tr), cfg.init_w), nw]), K, seed, cfg.n_estimators)
            if cfg.screen:
                sx.append(nx[-cfg.R:]); sy.append(ny[-cfg.R:])
        t_upd += time.perf_counter() - t2
    L.update(y=ys, block=ds["block"][stream], K=K, hazard=hz, n=n, t_dec=t_dec, t_upd=t_upd, train_time=train_time,
             n_refresh=n_refresh, eps_mean=eps_sum / max(n, 1), rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
             cfg=asdict(cfg), seed=seed, dataset=ds["name"])
    if return_log:
        L["P"] = Pall; L["Xs"] = Xs
    if keep_model:
        L["model"] = model; L["tr"] = tr
    return L
