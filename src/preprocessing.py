"""Dataset preprocessing for the PACT study.

Each raw dataset is converted into a stream description saved as
datasets/processed/<name>.npz with fields
  X        float32 (n, d)   feature matrix
  y        int     (n,)     class label
  block    int     (n,)     ordering unit (batch / recording / bottle / campaign batch)
  init     bool    (n,)     True for the initial (pre-deployment) pool
  feat     str     (d,)     feature names
  classes  str     (K,)     class names
  hazard   int              index of the hazard class, -1 if none
  shuffle  int              1 if within-block order is unknown and is permuted per seed
Usage: python -m src.preprocessing <raw_root> <out_dir>
Raw-data layout expected under <raw_root> is documented in datasets/README.md.
"""
import sys, os, glob, re
import numpy as np, pandas as pd


def _save(out, name, X, y, block, init, feat, classes, hazard, shuffle):
    X = np.asarray(X, np.float32)
    assert np.isfinite(X).all(), name
    np.savez_compressed(os.path.join(out, name + ".npz"), X=X, y=np.asarray(y, int), block=np.asarray(block, int),
                        init=np.asarray(init, bool), feat=np.array(feat), classes=np.array(classes),
                        hazard=hazard, shuffle=shuffle)
    yy = np.asarray(y); ini = np.asarray(init, bool)
    print(f"{name}: n={len(yy)} d={X.shape[1]} init={ini.sum()} stream={(~ini).sum()} "
          f"class counts init={np.bincount(yy[ini], minlength=len(classes)).tolist()} "
          f"stream={np.bincount(yy[~ini], minlength=len(classes)).tolist()}")


# ---------------------------------------------------------------- gas drift
def gas(root, out):
    X, y, b = [], [], []
    for k in range(1, 11):
        for l in open(os.path.join(root, "drift", f"batch{k}.dat")):
            p = l.split()
            y.append(int(p[0].split(";")[0]) - 1)
            X.append([float(v.split(":")[1]) for v in p[1:]])
            b.append(k)
    X = np.array(X); b = np.array(b)
    names = ["dR", "ndR", "EMAi001", "EMAi01", "EMAi1", "EMAd001", "EMAd01", "EMAd1"]
    feat = [f"S{s+1}_{n}" for s in range(16) for n in names]
    _save(out, "gas", X, y, b, b <= 2, feat,
          ["Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene"], -1, 1)


# ---------------------------------------------------------------- home activity
def home(root, out, win_min=5.0):
    meta = pd.read_csv(os.path.join(root, "home", "HT_Sensor_metadata.dat"), sep=r"\s+")
    d = pd.read_csv(glob.glob(os.path.join(root, "home_x", "**", "HT_Sensor_dataset.dat"), recursive=True)[0], sep=r"\s+")
    meta["when"] = pd.to_datetime(meta["date"], format="%m-%d-%y") + pd.to_timedelta(meta["t0"], unit="h")
    meta = meta.sort_values("when").reset_index(drop=True)
    R = [f"R{i}" for i in range(1, 9)]
    cls = {"background": 0, "banana": 1, "wine": 2}
    rows, y, blk = [], [], []
    w = win_min / 60.0
    for order, m in meta.iterrows():
        g = d[d.id == m["id"]]
        if len(g) == 0:
            continue
        base = g[(g.time < -0.5)][R].mean().values            # early pre-stimulus baseline
        if not np.isfinite(base).all():
            base = g[R].iloc[:200].mean().values
        # windows: late pre-stimulus period (background) and the stimulus period (class of the recording)
        spans = [(-0.5, 0.0, 0), (0.0, float(m["dt"]), cls[m["class"]])]
        for lo, hi, lab in spans:
            t = lo
            while t + w <= hi + 1e-9:
                s = g[(g.time >= t) & (g.time < t + w)]
                t += w
                if len(s) < 30:
                    continue
                r = np.log(s[R].values / base)
                tt = s.time.values - s.time.values.mean()
                slope = (tt[:, None] * (r - r.mean(0))).sum(0) / ((tt ** 2).sum() + 1e-12)
                rows.append(np.concatenate([r.mean(0), r.std(0), slope, [s["Temp."].mean(), s["Humidity"].mean()]]))
                y.append(lab); blk.append(order)
    X = np.array(rows); blk = np.array(blk)
    feat = [f"{r}_logratio" for r in R] + [f"{r}_std" for r in R] + [f"{r}_slope" for r in R] + ["Temp", "Humidity"]
    n_init = int(round(0.25 * (blk.max() + 1)))            # first 25 % of recordings (chronological)
    _save(out, "home", X, y, blk, blk < n_init, feat, ["Background", "Banana", "Wine"], -1, 0)


# ---------------------------------------------------------------- wine spoilage
def wine(root, out):
    rows, y, blk = [], [], []
    cls = {"HQ": 0, "AQ": 1, "LQ": 2}
    files = sorted(p for p in glob.glob(os.path.join(root, "wine_x", "**", "*.txt"), recursive=True)
                   if os.path.basename(p)[:2] in cls)
    bottles = sorted({os.path.basename(p)[:13] for p in files})
    for p in files:
        a = np.nan_to_num(np.loadtxt(p), nan=0.0, posinf=0.0, neginf=0.0)
        hum, tmp, r = a[:, 0], a[:, 1], a[:, 2:]
        r0 = np.maximum(np.abs(r[:20]).mean(0), 1e-6)
        rel = r / r0
        f = [rel.min(0), rel.max(0), rel.mean(0), rel[-200:].mean(0), rel.argmin(0) / len(rel),
             np.abs(np.diff(rel, axis=0)).max(0)]
        rows.append(np.concatenate(f + [[hum.mean(), tmp.mean()]]))
        n = os.path.basename(p)
        y.append(cls[n[:2]]); blk.append(bottles.index(n[:13]))
    s = ["MQ3a", "MQ4a", "MQ6a", "MQ3b", "MQ4b", "MQ6b"]
    feat = [f"{k}_{q}" for q in ["min", "max", "mean", "tail", "tmin", "maxstep"] for k in s] + ["Humidity", "Temp"]
    # initial pool is chosen per seed at run time (two bottles per class); mark none here
    _save(out, "wine", np.array(rows), y, blk, np.zeros(len(y), bool), feat,
          ["High quality", "Average quality", "Low quality (spoiled)"], -1, 2)


# ---------------------------------------------------------------- penicillin fermentation
def ferment(root, out, step=4):
    d = pd.read_csv(os.path.join(root, "ips", "indpensim_process.csv"))
    c = list(d.columns)
    # the published header is shifted by one comma inside a column name; true roles by position:
    # 31 = step-level fault flag, 34 = batch number, 36 = batch-level fault flag
    fault, batch = d.iloc[:, 31].values.astype(int), d.iloc[:, 34].values.astype(int)
    online = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16, 17, 18, 19, 20, 22, 24, 25, 28]
    short = ["Time", "Fg", "RPM", "Fs", "Fa", "Fb", "Fc", "Fh", "Fw", "Pressure", "Fremoved", "DO2", "V", "Wt",
             "pH", "T", "Q", "CO2out", "Fpaa", "Foil", "OUR", "O2out", "CER"]
    V = d.iloc[:, online].copy(); V.columns = short
    dyn = ["DO2", "pH", "T", "CO2out", "OUR", "O2out", "CER", "Wt", "Fs", "Fg", "Pressure"]
    g = V.groupby(batch)
    for k in dyn:
        V["d_" + k] = g[k].diff(5).fillna(0.0)                 # change over the previous hour (5 x 0.2 h)
        V["sd_" + k] = g[k].transform(lambda s: s.rolling(10, min_periods=2).std()).fillna(0.0)
    keep = np.zeros(len(V), bool)
    for b in np.unique(batch):
        idx = np.where(batch == b)[0]
        keep[idx[::step]] = True
    normal = [b for b in range(1, 91)]; faulty = list(range(91, 101))
    order = []
    for i, b in enumerate(normal):                              # documented campaign re-ordering:
        order.append(b)                                         # one fault batch after every nine normal batches
        if (i + 1) % 9 == 0:
            order.append(faulty[(i + 1) // 9 - 1])
    pos = {b: i for i, b in enumerate(order)}
    blk = np.array([pos[b] for b in batch])
    o = np.lexsort((np.arange(len(blk)), blk))
    o = o[keep[o]]
    _save(out, "ferment", V.values[o], fault[o], blk[o], blk[o] < 20, list(V.columns), ["Normal", "Fault"], 1, 0)


# ---------------------------------------------------------------- on-site water reuse
def water(root, out, thr_log10=5.0):
    f = pd.read_csv(os.path.join(root, "eawag", "fcm_data.csv"), sep=";", encoding="latin-1")
    f["ICC"] = pd.to_numeric(f["ICC"].astype(str).str.replace(r"[^0-9.]", "", regex=True), errors="coerce")
    f["ts"] = pd.to_datetime(f.DateTime, format="%d.%m.%Y %H:%M")
    s = pd.read_csv(os.path.join(root, "eawag", "sensor_data.csv"), sep=None, engine="python", encoding="latin-1")
    s = s.dropna(subset=["DateTime"])
    s["ts"] = pd.to_datetime(s.DateTime, format="%d.%m.%Y %H:%M:%S")
    sv = ["ORP", "Cond", "pH", "Turb", "T", "FC_cal"]
    rows, y, blk, icc = [], [], [], []
    for site_i, site in enumerate(["ONWS1", "ONWS2"]):
        ss = s[s.Site == site].sort_values("ts")
        ss[sv] = ss[sv].ffill(limit=12)                         # carry a sensor value forward for at most 1 h
        for _, r in f[f.Site == site].sort_values("ts").iterrows():
            wdw = ss[(ss.ts <= r.ts) & (ss.ts > r.ts - pd.Timedelta(hours=1))]
            if len(wdw) < 3 or wdw[sv].iloc[-1].isna().any() or not np.isfinite(r.ICC):
                continue
            last = wdw[sv].iloc[-1].values
            mean = wdw[sv].mean().values
            delta = (wdw[sv].iloc[-1] - wdw[sv].iloc[0]).values
            rows.append(np.concatenate([last, mean, delta, [r.ts.hour + r.ts.minute / 60.0]]))
            y.append(int(np.log10(r.ICC) >= thr_log10)); blk.append(site_i); icc.append(r.ICC)
    X = np.array(rows); y = np.array(y); blk = np.array(blk)
    feat = [f"{v}_last" for v in sv] + [f"{v}_mean1h" for v in sv] + [f"{v}_delta1h" for v in sv] + ["HourOfDay"]
    n = len(y); t = np.arange(n)
    init = t < int(0.25 * n)                                   # chronological: first 25 % of paired samples
    print("water: ICC>=1e%g rate site1=%.3f site2=%.3f  paired=%d" % (thr_log10, y[blk == 0].mean(), y[blk == 1].mean(), n))
    np.save(os.path.join(out, "water_icc.npy"), np.array(icc))
    # blocks for reporting: split the stream into consecutive segments of ~100 samples
    _save(out, "water", X, y, (t // 100), init, feat, ["Low microbial load", "High microbial load"], 1, 0)


if __name__ == "__main__":
    root, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    which = sys.argv[3:] or ["gas", "home", "wine", "ferment", "water"]
    for w in which:
        globals()[w](root, out)
