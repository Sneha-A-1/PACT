"""Result figures generated from results/metrics and results/logs."""
import os, glob
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.results_io import read

INK, MUTED, SURF, GRID = "#0b0b0b", "#52514e", "#ffffff", "#e4e3df"
COL = {"PACT": "#2a78d6", "Random audit": "#eb6834", "NexCP": "#1baf7a", "ACI": "#eda100", "Conformal P-control": "#e87ba4",
       "Uncertainty sampling": "#008300", "Split CP (static)": "#6250d6", "APS (static)": "#e34948", "Full feedback (oracle)": "#8a8985",
       "PACT + screening": "#1baf7a"}
ORDER = ["PACT", "Random audit", "NexCP", "ACI", "Conformal P-control", "Uncertainty sampling", "Split CP (static)", "APS (static)"]
DS = {"gas": "Gas drift", "home": "Home activity", "wine": "Wine spoilage", "ferment": "Fermentation", "water": "Water reuse"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK, "legend.frameon": False, "figure.dpi": 300,
                     "savefig.facecolor": SURF, "axes.titlesize": 8.5, "axes.titleweight": "bold", "axes.titlelocation": "left"})
os.makedirs("figures", exist_ok=True)


def save(fig, name):
    fig.savefig(f"figures/{name}.png", bbox_inches="tight", pad_inches=0.05); plt.close(fig)


def roll(x, w):
    c = np.cumsum(np.insert(x.astype(float), 0, 0)); return (c[w:] - c[:-w]) / w


def fig_stream(seed=1):
    """Rolling coverage and cumulative verification rate along the gas-drift stream."""
    fig, ax = plt.subplots(3, 1, figsize=(7.0, 6.0), sharex=True, gridspec_kw=dict(hspace=0.22))
    tags = {"PACT": "PACT", "Random audit": "Random_audit", "NexCP": "NexCP", "Split CP (static)": "Split_CP"}
    w = 400
    for m, t in tags.items():
        L = np.load(f"results/logs/gas_{t}_s{seed}.npz"); n = len(L["cov"]); x = np.arange(w, n + 1)
        ax[0].plot(x, 100 * roll(L["cov"], w), color=COL[m], lw=1.4, label=m)
        ax[1].plot(x, roll(L["size"], w), color=COL[m], lw=1.4)
        ax[2].plot(x, 100 * roll(L["pred"] == L["y"], w), color=COL[m], lw=1.4)
        blk = L["block"]
    ax[0].axhline(90, color=INK, lw=0.8, ls=(0, (4, 3))); ax[0].text(len(blk) * 0.995, 86, "90% target", fontsize=6.5, va="top", ha="right", color=INK)
    edges = np.where(np.diff(blk) != 0)[0] + 1
    for a in ax:
        for e in edges:
            a.axvline(e, color=MUTED, lw=0.5, ls=":")
    for b, s, e in zip(np.unique(blk), np.r_[0, edges], np.r_[edges, len(blk)]):
        if e - s > 350:
            ax[0].text((s + e) / 2, 104, f"B{b}", fontsize=6.5, ha="center", color=MUTED)
    ax[0].set_ylim(0, 100); ax[0].set_ylabel("Rolling coverage (%)"); ax[1].set_ylabel("Rolling set size"); ax[2].set_ylabel("Rolling accuracy (%)")
    ax[2].set_xlabel("Deployment sample index (batches B3 to B10)")
    ax[0].legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.12), fontsize=7.5, handlelength=1.6)
    save(fig, "stream_trajectories")


def fig_main(d):
    """Coverage, set size and system accuracy of budget-matched methods on the five streams."""
    g = d.groupby(["dataset", "method"]).mean(numeric_only=True)
    ms = [m for m in ORDER if m not in ("Split CP (static)", "APS (static)")] + ["Split CP (static)"]
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 3.3), gridspec_kw=dict(wspace=0.42))
    ds = list(DS); y = np.arange(len(ds)); h = 0.11
    for k, (met, lab, sc) in enumerate([("coverage", "Coverage (%)", 100), ("set_size", "Mean set size", 1), ("sys_acc", "System accuracy (%)", 100)]):
        for i, m in enumerate(ms):
            v = [sc * g.loc[(dd, m), met] if (dd, m) in g.index else np.nan for dd in ds]
            ax[k].barh(y + (i - (len(ms) - 1) / 2) * (h + 0.012), v, height=h, color=COL[m], label=m if k == 0 else None, edgecolor=SURF, linewidth=0.4)
        ax[k].set_yticks(y); ax[k].set_yticklabels([DS[x] for x in ds] if k == 0 else []); ax[k].invert_yaxis(); ax[k].set_xlabel(lab); ax[k].grid(axis="y", visible=False)
    ax[0].axvline(90, color=INK, lw=0.8, ls=(0, (4, 3))); ax[0].set_xlim(0, 100); ax[2].set_xlim(20, 100)
    ax[1].axvline(1, color=INK, lw=0.8, ls=(0, (4, 3)))
    fig.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 0.9), fontsize=7.2, handlelength=1.2)
    save(fig, "performance_comparison")


def fig_hazard(d):
    """Missed-risk rate against false-alarm rate on the two hazard-monitoring streams."""
    g = d.groupby(["dataset", "method"]).mean(numeric_only=True)
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.1), gridspec_kw=dict(wspace=0.28))
    for a, dd in zip(ax, ("ferment", "water")):
        for m in ORDER:
            if (dd, m) not in g.index:
                continue
            r = g.loc[(dd, m)]
            a.scatter(100 * r.false_alarm, 100 * r.missed_risk, s=22 + 170 * r.label_rate, color=COL[m], edgecolor=SURF, linewidth=1.2, zorder=3,
                      marker="D" if m == "PACT" else "o", label=m if dd == "ferment" else None)
        a.set_title(DS[dd]); a.set_xlabel("False-alarm rate (%)"); a.set_xlim(left=0); a.set_ylim(bottom=0)
    ax[0].set_ylabel("Missed-risk rate (%)")
    lg = fig.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.2), fontsize=7, handletextpad=0.3)
    for h in lg.legend_handles:
        h.set_sizes([40])
    fig.text(0.5, -0.24, "Marker area grows with the verification (label) rate; PACT is drawn as a diamond.", ha="center", fontsize=6.8, color=MUTED)
    save(fig, "hazard_tradeoff")


def fig_ablation_sens(ab, se):
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 3.0), gridspec_kw=dict(wspace=0.5, width_ratios=[1.25, 1, 1]))
    g = ab[ab.dataset == "gas"].groupby("method").mean(numeric_only=True)
    names = [n for n in ["PACT", "w/o propensity weighting", "w/o random audits", "w/o model refresh", "w/o recalibration", "w/o recency window",
                         "w/o coverage-debt controller", "w/o targeted escalation"] if n in g.index]
    y = np.arange(len(names))
    ax[0].barh(y, [100 * abs(g.loc[n, "coverage"] - 0.9) for n in names], color=[COL["PACT"] if n == "PACT" else "#9fc1ea" for n in names], height=0.6)
    for i, n in enumerate(names):
        ax[0].text(100 * abs(g.loc[n, "coverage"] - 0.9) + 0.4, i, f"{100 * g.loc[n, 'sys_acc']:.1f}", va="center", fontsize=6.5, color=INK)
    ax[0].set_yticks(y); ax[0].set_yticklabels([n.replace("PACT", "PACT (full)") for n in names], fontsize=6.8); ax[0].invert_yaxis(); ax[0].grid(axis="y", visible=False)
    ax[0].set_xlabel("|Coverage - 90| (points)"); ax[0].set_title("Ablation (gas drift)")
    ax[0].text(0.98, 0.02, "labels: system accuracy (%)", transform=ax[0].transAxes, ha="right", fontsize=6.3, color=MUTED)
    for a, p, lab in ((ax[1], "eps", "Base audit rate $\\varepsilon$"), (ax[2], "W", "Calibration window $W$")):
        s = se[se.param == p].groupby("value").mean(numeric_only=True)
        a.plot(s.index, 100 * s.coverage, "-o", color=COL["PACT"], ms=4, lw=1.4, label="Coverage")
        a.plot(s.index, 100 * s.sys_acc, "-s", color=COL["NexCP"], ms=4, lw=1.4, label="System accuracy")
        a.plot(s.index, 100 * s.label_rate, "-^", color=COL["Random audit"], ms=4, lw=1.4, label="Label rate")
        a.set_xscale("log"); a.set_xlabel(lab); a.set_title("Sensitivity (gas drift)"); a.axhline(90, color=INK, lw=0.7, ls=(0, (4, 3)))
    ax[1].set_ylabel("Percent"); ax[2].legend(fontsize=6.5, loc="center right")
    save(fig, "ablation_sensitivity")


def fig_robust(ro, po):
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 3.0), gridspec_kw=dict(wspace=0.42))
    conds = ["clean", "noise 0.1", "noise 0.3", "sensor dropout", "evasion 0.5", "evasion 1.0"]
    g = ro[ro.dataset == "gas"].groupby(["condition", "method"]).mean(numeric_only=True)
    ms = ["PACT", "Random audit", "NexCP", "Split CP (static)"]; x = np.arange(len(conds)); w = 0.19
    for k, (met, lab) in enumerate((("coverage", "Coverage (%)"), ("sys_acc", "System accuracy (%)"))):
        for i, m in enumerate(ms):
            ax[k].bar(x + (i - 1.5) * (w + 0.01), [100 * g.loc[(c, m), met] if (c, m) in g.index else np.nan for c in conds], width=w, color=COL[m], label=m, edgecolor=SURF, linewidth=0.4)
        ax[k].set_xticks(x); ax[k].set_xticklabels(["Clean", "Noise\n0.1", "Noise\n0.3", "Drop-\nout", "Evas.\n0.5", "Evas.\n1.0"], fontsize=6); ax[k].set_ylabel(lab); ax[k].grid(axis="x", visible=False)
    ax[0].axhline(90, color=INK, lw=0.7, ls=(0, (4, 3))); ax[0].set_title("Input stress (gas drift)")
    ax[0].legend(fontsize=6.3, ncol=2, loc="lower center", bbox_to_anchor=(1.1, 1.1))
    p = po[po.dataset == "gas"].groupby(["poison", "method"]).mean(numeric_only=True)
    for m in ("PACT", "PACT + screening", "Random audit"):
        s = p.xs(m, level="method"); ax[2].plot(100 * s.index, 100 * s.label_rate, "-o", color=COL[m], ms=4, lw=1.4, label=m)
    ax[2].set_xlabel("Corrupted verification labels (%)"); ax[2].set_ylabel("Label rate (%)"); ax[2].set_title("Feedback poisoning"); ax[2].legend(fontsize=6.3)
    save(fig, "robustness")


def fig_explain(ex):
    fig, ax = plt.subplots(1, 3, figsize=(7.6, 3.0), gridspec_kw=dict(wspace=0.62, width_ratios=[1.3, 1.3, 1]))
    for a, f, t in ((ax[0], "shap_sensor_batch_static.csv", "Static model"), (ax[1], "shap_sensor_batch_pact.csv", "PACT (refreshed)")):
        S = pd.read_csv(f"results/metrics/{f}", index_col=0)
        im = a.imshow(100 * S.values.T, aspect="auto", cmap="Blues", vmin=0, vmax=max(20, 100 * S.values.max()))
        a.set_xticks(range(len(S))); a.set_xticklabels([f"B{b}" for b in S.index], fontsize=6.5); a.set_yticks(range(0, 16, 3)); a.set_yticklabels([f"S{i+1}" for i in range(0, 16, 3)], fontsize=6.5)
        a.set_title(t); a.grid(False); a.set_xlabel("Batch")
    ax[0].set_ylabel("Sensor"); cb = fig.colorbar(im, ax=ax[1], fraction=0.05, pad=0.03); cb.set_label("Attribution share (%)", fontsize=6.5, labelpad=2); cb.ax.tick_params(labelsize=6)
    g = ex[ex.model == "PACT (refreshed)"].groupby("dataset").mean(numeric_only=True)
    ds = [d for d in ("gas", "ferment", "water") if d in g.index]; x = np.arange(len(ds))
    ax[2].bar(x - 0.18, [100 * g.loc[d, "del10_top"] for d in ds], width=0.34, color=COL["PACT"], label="Top 10% attributed")
    ax[2].bar(x + 0.18, [100 * g.loc[d, "del10_rand"] for d in ds], width=0.34, color="#9fc1ea", label="Random 10%")
    ax[2].set_xticks(x); ax[2].set_xticklabels([{"gas": "Gas\ndrift", "ferment": "Fermen-\ntation", "water": "Water\nreuse"}[d] for d in ds], fontsize=6.3); ax[2].set_ylabel("Probability drop (points)"); ax[2].set_title("Deletion test"); ax[2].legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.2)); ax[2].set_ylim(0, 75); ax[2].grid(axis="x", visible=False)
    save(fig, "explainability")


if __name__ == "__main__":
    d = read("main"); fig_main(d); fig_hazard(d); fig_stream()
    if os.path.exists("results/metrics/ablation.csv") and os.path.exists("results/metrics/sensitivity.csv"):
        fig_ablation_sens(read("ablation"), read("sensitivity"))
    if os.path.exists("results/metrics/robust.csv") and os.path.exists("results/metrics/poison.csv"):
        fig_robust(read("robust"), read("poison"))
    if os.path.exists("results/metrics/explain.csv"):
        fig_explain(pd.read_csv("results/metrics/explain.csv"))
