"""LaTeX tables and number macros generated directly from results/metrics (single source of truth)."""
import os, sys, platform
import numpy as np, pandas as pd
from scipy import stats
from src.results_io import read, read_static

T = "manuscript/tables"; os.makedirs(T, exist_ok=True)
DS = {"gas": "Gas drift", "home": "Home activity", "wine": "Wine spoilage", "ferment": "Fermentation", "water": "Water reuse"}
DSC = {"gas": "Gas", "home": "Home", "wine": "Wine", "ferment": "Ferm", "water": "Water"}
M = ["Split CP (static)", "APS (static)", "ACI", "NexCP", "Conformal P-control", "Uncertainty sampling", "Random audit", "PACT", "Full feedback (oracle)"]
MC = {"Split CP (static)": "Split", "APS (static)": "APS", "ACI": "ACI", "NexCP": "Nex", "Conformal P-control": "Pid", "Uncertainty sampling": "Unc",
      "Random audit": "Rand", "PACT": "PACT", "Full feedback (oracle)": "Orc"}
LAB = {"PACT": "\\textbf{PACT (proposed)}", "Full feedback (oracle)": "Full feedback (reference)"}
pc = lambda v, k=1: "--" if pd.isna(v) else f"{100 * v:.{k}f}"
fx = lambda v, k=2: "--" if pd.isna(v) else f"{v:.{k}f}"


def write(name, body):
    open(f"{T}/{name}.tex", "w").write(body + "\n")


def block(g, datasets, cols, name):
    """cols: list of (key, header, formatter)."""
    out = ["\\begin{tabular}{ll" + "r" * len(cols) + "}", "\\toprule", "Dataset & Method & " + " & ".join(h for _, h, _ in cols) + " \\\\", "\\midrule"]
    for i, d in enumerate(datasets):
        for j, m in enumerate(M):
            if (d, m) not in g.index:
                continue
            r = g.loc[(d, m)]
            out.append((DS[d] if j == 0 else "") + " & " + LAB.get(m, m) + " & " + " & ".join(f(r[k]) for k, _, f in cols) + " \\\\")
        out.append("\\midrule" if i < len(datasets) - 1 else "\\bottomrule")
    out.append("\\end{tabular}"); write(name, "\n".join(out))


def main_tables(d):
    g = d.groupby(["dataset", "method"]).mean(numeric_only=True)
    block(g, ["gas", "home", "wine"], [("coverage", "Coverage (\\%)", pc), ("set_size", "Set size", fx), ("label_rate", "Label rate (\\%)", pc),
          ("auto_acc", "Auto-decision acc. (\\%)", pc), ("sys_acc", "System acc. (\\%)", pc), ("worst_block_cov", "Worst-segment cov. (\\%)", pc)], "main")
    block(g, ["ferment", "water"], [("coverage", "Coverage (\\%)", pc), ("hazard_cov", "Hazard cov. (\\%)", pc), ("missed_risk", "Missed risk (\\%)", pc),
          ("false_alarm", "False alarm (\\%)", pc), ("label_rate", "Label rate (\\%)", pc), ("sys_acc", "System acc. (\\%)", pc)], "hazard")
    # latency
    cols = [("latency_ms", "Decision (ms)", lambda v: fx(v, 3)), ("update_ms", "Feedback (ms)", lambda v: fx(v, 3)), ("e2e_ms", "End-to-end (ms)", lambda v: fx(v, 3)),
            ("throughput", "Throughput (samples/s)", lambda v: fx(v, 0)), ("n_refresh", "Refreshes", lambda v: fx(v, 1)), ("rss_mb", "Peak memory (MB)", lambda v: fx(v, 0))]
    out = ["\\begin{tabular}{ll" + "r" * len(cols) + "}", "\\toprule", "Dataset & Method & " + " & ".join(h for _, h, _ in cols) + " \\\\", "\\midrule"]
    for i, dd in enumerate(["gas", "ferment"]):
        for j, m in enumerate(["Split CP (static)", "NexCP", "Random audit", "PACT", "Full feedback (oracle)"]):
            if (dd, m) in g.index:
                out.append((DS[dd] if j == 0 else "") + " & " + LAB.get(m, m) + " & " + " & ".join(f(g.loc[(dd, m)][k]) for k, _, f in cols) + " \\\\")
        out.append("\\midrule" if i == 0 else "\\bottomrule")
    out.append("\\end{tabular}"); write("latency", "\n".join(out))
    return g


def static_table(g):
    s = read_static().groupby(["dataset", "method"]).mean(numeric_only=True)
    ms = ["Logistic regression", "SVM (RBF)", "k-NN", "Random forest", "MLP", "LightGBM", "Deep ensemble + TS"]
    ds = [d for d in DS if d in s.index.get_level_values(0)]
    out = ["\\begin{tabular}{l" + "rrr" * len(ds) + "}", "\\toprule", " & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{DS[d]}}}" for d in ds) + " \\\\",
           "Method & " + " & ".join("Acc. & MCC & ECE" for _ in ds) + " \\\\", "\\midrule"]
    for m in ms:
        out.append(m + " & " + " & ".join(f"{pc(s.loc[(d, m), 'accuracy'])} & {fx(s.loc[(d, m), 'mcc'])} & {fx(s.loc[(d, m), 'ece'], 3)}" if (d, m) in s.index else "-- & -- & --" for d in ds) + " \\\\")
    out.append("\\midrule")
    for m in ["Random audit", "PACT"]:
        out.append(LAB.get(m, m + " (budget-matched refresh)") + " & " + " & ".join(f"{pc(g.loc[(d, m), 'accuracy'])} & {fx(g.loc[(d, m), 'mcc'])} & {fx(g.loc[(d, m), 'ece'], 3)}" for d in ds) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]; write("static", "\n".join(out))
    full = ["\\begin{tabular}{llrrrrrrr}", "\\toprule", "Dataset & Method & Acc. (\\%) & Macro-F1 (\\%) & MCC & ROC-AUC & ECE & Brier & NLL \\\\", "\\midrule"]
    for i, d in enumerate(ds):
        for j, m in enumerate(ms):
            r = s.loc[(d, m)]
            full.append((DS[d] if j == 0 else "") + f" & {m} & {pc(r.accuracy)} & {pc(r.macro_f1)} & {fx(r.mcc)} & {fx(r.roc_auc, 3)} & {fx(r.ece, 3)} & {fx(r.brier, 3)} & {fx(r.nll, 3)} \\\\")
        for m in ["Random audit", "PACT"]:
            r = g.loc[(d, m)]
            full.append(f" & {LAB.get(m, m)} & {pc(r.accuracy)} & {pc(r.macro_f1)} & {fx(r.mcc)} & -- & {fx(r.ece, 3)} & {fx(r.brier, 3)} & {fx(r.nll, 3)} \\\\")
        full.append("\\midrule" if i < len(ds) - 1 else "\\bottomrule")
    full.append("\\end{tabular}"); write("static_full", "\n".join(full))
    return s


def ablation_table():
    a = read("ablation").groupby(["dataset", "method"]).mean(numeric_only=True)
    names = ["PACT", "w/o propensity weighting", "w/o random audits", "w/o model refresh", "w/o recalibration", "w/o recency window", "w/o coverage-debt controller", "w/o targeted escalation"]
    ds = [d for d in ["gas", "home", "ferment", "water"] if d in a.index.get_level_values(0)]
    out = ["\\begin{tabular}{l" + "rrr" * len(ds) + "}", "\\toprule", " & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{DS[d]}}}" for d in ds) + " \\\\",
           "Configuration & " + " & ".join("Cov. & Label & " + ("Missed" if d in ("ferment", "water") else "Sys.") for d in ds) + " \\\\", "\\midrule"]
    for n in names:
        cells = []
        for d in ds:
            if (d, n) in a.index:
                r = a.loc[(d, n)]; cells.append(f"{pc(r.coverage)} & {pc(r.label_rate)} & {pc(r.missed_risk if d in ('ferment', 'water') else r.sys_acc)}")
            else:
                cells.append("-- & -- & --")
        out.append(("\\textbf{PACT (full)}" if n == "PACT" else n) + " & " + " & ".join(cells) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]; write("ablation", "\n".join(out)); return a


def supp_tables(d):
    if os.path.exists("results/metrics/sensitivity.csv"):
        s = read("sensitivity").groupby(["param", "value"]).mean(numeric_only=True)
        out = ["\\begin{tabular}{llrrrrr}", "\\toprule", "Parameter & Value & Coverage (\\%) & Set size & Label rate (\\%) & System acc. (\\%) & Worst-segment cov. (\\%) \\\\", "\\midrule"]
        nm = {"eps": "$\\varepsilon$", "W": "$W$", "R": "$R$", "alpha": "$\\alpha$"}
        for p in ["eps", "W", "R", "alpha"]:
            if p in s.index.get_level_values(0):
                for v, r in s.xs(p).iterrows():
                    out.append(f"{nm[p]} & {v:g} & {pc(r.coverage)} & {fx(r.set_size)} & {pc(r.label_rate)} & {pc(r.sys_acc)} & {pc(r.worst_block_cov)} \\\\")
        out += ["\\bottomrule", "\\end{tabular}"]; write("sensitivity", "\n".join(out))
    if os.path.exists("results/metrics/robust.csv"):
        r_ = read("robust").groupby(["dataset", "condition", "method"]).mean(numeric_only=True)
        out = ["\\begin{tabular}{lllrrrr}", "\\toprule", "Dataset & Condition & Method & Coverage (\\%) & Set size & Label rate (\\%) & System acc. (\\%) \\\\", "\\midrule"]
        for dd in [x for x in ("gas", "home") if x in r_.index.get_level_values(0)]:
            for c in ["clean", "noise 0.1", "noise 0.3", "sensor dropout", "evasion 0.5", "evasion 1.0"]:
                for j, m in enumerate(["Split CP (static)", "NexCP", "Random audit", "PACT"]):
                    if (dd, c, m) in r_.index:
                        r = r_.loc[(dd, c, m)]
                        out.append(f"{DS[dd] if (j == 0 and c == 'clean') else ''} & {c if j == 0 else ''} & {LAB.get(m, m)} & {pc(r.coverage)} & {fx(r.set_size)} & {pc(r.label_rate)} & {pc(r.sys_acc)} \\\\")
            out.append("\\midrule")
        out[-1] = "\\bottomrule"; out.append("\\end{tabular}"); write("robust", "\n".join(out))
    if os.path.exists("results/metrics/poison.csv"):
        p_ = read("poison").groupby(["dataset", "poison", "method"]).mean(numeric_only=True)
        out = ["\\begin{tabular}{lrlrrrrr}", "\\toprule", "Dataset & Corrupted (\\%) & Method & Coverage (\\%) & Label rate (\\%) & Accuracy (\\%) & Detection (\\%) & False flags (\\%) \\\\", "\\midrule"]
        for dd in [x for x in ("gas", "home") if x in p_.index.get_level_values(0)]:
            for rho in (0.0, 0.1, 0.2, 0.3):
                for j, m in enumerate(["Random audit", "PACT", "PACT + screening"]):
                    if (dd, rho, m) in p_.index:
                        r = p_.loc[(dd, rho, m)]
                        det = pc(r.get("poison_detect", np.nan)) if m.endswith("screening") and rho > 0 else "--"
                        fpr = pc(r.get("poison_fpr", np.nan), 2) if m.endswith("screening") else "--"
                        out.append(f"{DS[dd] if (j == 0 and rho == 0) else ''} & {100 * rho:.0f} & {m} & {pc(r.coverage)} & {pc(r.label_rate)} & {pc(r.accuracy)} & {det} & {fpr} \\\\")
            out.append("\\midrule")
        out[-1] = "\\bottomrule"; out.append("\\end{tabular}"); write("poison", "\n".join(out))
    if os.path.exists("results/metrics/explain.csv"):
        e = pd.read_csv("results/metrics/explain.csv").groupby(["dataset", "model"]).mean(numeric_only=True)
        out = ["\\begin{tabular}{llrrrrrrr}", "\\toprule", "Dataset & Model & Del. 5\\% top & Del. 5\\% rand. & Del. 10\\% top & Del. 10\\% rand. & Cosine stab. & Top-5 Jaccard & Time (ms) \\\\", "\\midrule"]
        for dd in [x for x in DS if x in e.index.get_level_values(0)]:
            for j, m in enumerate(["Static model", "PACT (refreshed)"]):
                r = e.loc[(dd, m)]
                out.append(f"{DS[dd] if j == 0 else ''} & {m} & {fx(r.del5_top, 3)} & {fx(r.del5_rand, 3)} & {fx(r.del10_top, 3)} & {fx(r.del10_rand, 3)} & {fx(r.cosine, 3)} & {fx(r.top5_jaccard, 3)} & {fx(r.explain_ms, 3)} \\\\")
        out += ["\\bottomrule", "\\end{tabular}"]; write("explain", "\n".join(out))
    # paired significance: PACT vs each budget-matched baseline, system accuracy and coverage error
    out = ["\\begin{tabular}{llrrrr}", "\\toprule", "Dataset & Baseline & $\\Delta$ System acc. (points) & $p$ (paired $t$) & $\\Delta$ $|$Cov.$-$90$|$ (points) & $p$ (paired $t$) \\\\", "\\midrule"]
    for dd in DS:
        x = d[d.dataset == dd].pivot_table(index="seed", columns="method", values=["sys_acc", "coverage"])
        if "PACT" not in x["sys_acc"]:
            continue
        for j, b in enumerate(["ACI", "NexCP", "Conformal P-control", "Uncertainty sampling", "Random audit"]):
            pr = x.dropna(subset=[("sys_acc", "PACT"), ("sys_acc", b)])
            a1, b1 = pr[("sys_acc", "PACT")], pr[("sys_acc", b)]
            a2, b2 = (pr[("coverage", "PACT")] - 0.9).abs(), (pr[("coverage", b)] - 0.9).abs()
            p1 = stats.ttest_rel(a1, b1).pvalue if len(pr) > 1 else np.nan; p2 = stats.ttest_rel(a2, b2).pvalue if len(pr) > 1 else np.nan
            out.append(f"{DS[dd] if j == 0 else ''} & {b} & {100 * (a1 - b1).mean():+.2f} & {p1:.3f} & {100 * (a2 - b2).mean():+.2f} & {p2:.3f} \\\\")
        out.append("\\midrule")
    out[-1] = "\\bottomrule"; out.append("\\end{tabular}"); write("stats", "\n".join(out))


def dataset_table():
    rows = []
    info = {"gas": ("Six volatile compounds, 16 metal-oxide sensors, 36 months", "Long-term sensor drift (batches 3--10)"),
            "home": ("Banana, wine and background odours, 8 sensors plus temperature and humidity", "Chronological, 59 days, ambient interference"),
            "wine": ("Wine spoilage thresholds, 6 sensors plus temperature and humidity", "Unseen bottles (bottle-held-out)"),
            "ferment": ("Industrial-scale penicillin fed-batch simulation, 23 process variables", "Batch campaign with recurring and new fault types"),
            "water": ("On-site water reuse, 6 online probes paired with flow cytometry", "Chronological, change of treatment site")}
    out = ["\\begin{tabular}{p{2.0cm}p{5.2cm}rrrrp{2.6cm}p{4.2cm}}", "\\toprule", "Dataset & Measurement context & Initial & Stream & $d$ & $K$ & Hazard class & Deployment shift \\\\", "\\midrule"]
    for k in DS:
        z = np.load(f"datasets/processed/{k}.npz"); init = z["init"]; hz = int(z["hazard"])
        ni = "6 bottles" if k == "wine" else str(int(init.sum())); ns = "remaining" if k == "wine" else str(int((~init).sum()))
        out.append(f"{DS[k]} & {info[k][0]} & {ni} & {ns} & {z['X'].shape[1]} & {len(z['classes'])} & {str(z['classes'][hz]) if hz >= 0 else 'none'} & {info[k][1]} \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]; write("datasets", "\n".join(out))


def macros(g, s=None, a=None):
    L = []
    met = {"Cov": ("coverage", 100, 1), "Size": ("set_size", 1, 2), "Lab": ("label_rate", 100, 1), "Auto": ("auto_acc", 100, 1), "Sys": ("sys_acc", 100, 1),
           "Worst": ("worst_block_cov", 100, 1), "HazCov": ("hazard_cov", 100, 1), "Miss": ("missed_risk", 100, 1), "FA": ("false_alarm", 100, 1),
           "Acc": ("accuracy", 100, 1), "ECE": ("ece", 1, 3), "Lat": ("e2e_ms", 1, 3), "Dec": ("latency_ms", 1, 3)}
    for (d, m), r in g.iterrows():
        for c, (k, sc, nd) in met.items():
            if k in r and not pd.isna(r[k]):
                L.append(f"\\newcommand{{\\res{DSC[d]}{MC[m]}{c}}}{{{sc * r[k]:.{nd}f}}}")
    open("manuscript/numbers.tex", "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    d = read("main"); g = main_tables(d); dataset_table(); macros(g)
    if os.path.exists("results/metrics/static.csv"):
        static_table(g)
    if os.path.exists("results/metrics/ablation.csv"):
        ablation_table()
    supp_tables(d)
    print("tables written:", sorted(os.listdir(T)))
