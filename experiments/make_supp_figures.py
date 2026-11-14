"""Supplementary figures: class distributions, feature correlation, reliability diagrams, confusion matrices."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from experiments.make_figures import COL, DS, INK, MUTED, SURF, save


def class_dist():
    fig, ax = plt.subplots(1, 5, figsize=(7.4, 2.4), gridspec_kw=dict(wspace=0.55))
    for a, k in zip(ax, DS):
        z = np.load(f"datasets/processed/{k}.npz"); y, init, K = z["y"], z["init"], len(z["classes"])
        if k == "wine":
            a.bar(np.arange(K), np.bincount(y, minlength=K), color=COL["PACT"], width=0.6, label="All samples")
        else:
            a.bar(np.arange(K) - 0.2, np.bincount(y[init], minlength=K), width=0.38, color="#9fc1ea", label="Initial pool")
            a.bar(np.arange(K) + 0.2, np.bincount(y[~init], minlength=K), width=0.38, color=COL["PACT"], label="Stream")
        a.set_yscale("log"); a.set_xticks(range(K)); a.set_xticklabels([str(i + 1) for i in range(K)], fontsize=6.5); a.set_title(DS[k], fontsize=7.5); a.grid(axis="x", visible=False)
        a.set_xlabel("Class index")
    ax[0].set_ylabel("Samples (log scale)"); ax[0].legend(fontsize=6, loc="lower center", bbox_to_anchor=(2.9, 1.18), ncol=2)
    save(fig, "supp_class_distribution")


def correlation():
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.6), gridspec_kw=dict(wspace=0.35))
    for a, k in zip(ax, ("gas", "ferment", "water")):
        z = np.load(f"datasets/processed/{k}.npz"); X = z["X"][z["init"]].astype(float)
        C = np.corrcoef(X[:, X.std(0) > 0], rowvar=False)
        im = a.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1); a.set_title(DS[k] + " (initial pool)", fontsize=7.5); a.grid(False); a.set_xlabel("Feature index"); a.tick_params(labelsize=6)
    ax[0].set_ylabel("Feature index"); cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02); cb.set_label("Pearson correlation", fontsize=6.5); cb.ax.tick_params(labelsize=6)
    save(fig, "supp_feature_correlation")


def reliability(seed=1):
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.6), gridspec_kw=dict(wspace=0.32))
    for a, k in zip(ax, ("gas", "home", "wine")):
        for m, t in (("PACT", "PACT"), ("Random audit", "Random_audit"), ("Split CP (static)", "Split_CP")):
            L = np.load(f"results/logs/{k}_{t}_s{seed}.npz"); p = L["pmax"]; c = L["pred"] == L["y"]
            e = np.linspace(0, 1, 11); xs, ys = [], []
            for lo, hi in zip(e[:-1], e[1:]):
                msk = (p > lo) & (p <= hi)
                if msk.sum() >= 10:
                    xs.append(p[msk].mean()); ys.append(c[msk].mean())
            a.plot(xs, ys, "-o", ms=3, lw=1.3, color=COL[m], label=m if m != "Split CP (static)" else "Static model")
        a.plot([0, 1], [0, 1], color=INK, lw=0.7, ls=(0, (4, 3))); a.set_title(DS[k], fontsize=7.5); a.set_xlabel("Predicted confidence"); a.set_xlim(0.2, 1.02); a.set_ylim(0, 1.02)
    ax[0].set_ylabel("Observed accuracy"); ax[0].legend(fontsize=6.3)
    save(fig, "supp_reliability")


def confusion(seed=1):
    fig, ax = plt.subplots(1, 2, figsize=(6.0, 2.9), gridspec_kw=dict(wspace=0.3))
    for a, (m, t) in zip(ax, (("Static model", "Split_CP"), ("PACT", "PACT"))):
        L = np.load(f"results/logs/gas_{t}_s{seed}.npz"); K = 6
        C = np.zeros((K, K)); np.add.at(C, (L["y"], L["pred"]), 1); C = C / C.sum(1, keepdims=True)
        a.imshow(100 * C, cmap="Blues", vmin=0, vmax=100); a.grid(False); a.set_title(f"{m} (gas drift)", fontsize=7.5)
        for i in range(K):
            for j in range(K):
                a.text(j, i, f"{100 * C[i, j]:.0f}", ha="center", va="center", fontsize=6, color="white" if C[i, j] > 0.55 else INK)
        a.set_xticks(range(K)); a.set_yticks(range(K)); a.set_xticklabels(range(1, K + 1)); a.set_yticklabels(range(1, K + 1)); a.set_xlabel("Predicted class")
    ax[0].set_ylabel("True class")
    save(fig, "supp_confusion")


if __name__ == "__main__":
    class_dist(); correlation(); reliability(); confusion()
