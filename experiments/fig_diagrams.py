"""Original schematic figures: architecture, dataflow and algorithm workflow."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

INK, MUTED, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
C = dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", yellow="#eda100", magenta="#e87ba4", green="#008300", violet="#6250d6", red="#e34948")
TINT = dict(blue="#e6f0fb", orange="#fdece5", aqua="#e2f6ee", yellow="#fdf3dc", magenta="#fcecf2", green="#e3f1e3", violet="#eceafa", red="#fce8e8", grey="#f1f0ee")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})


def box(ax, x, y, w, h, text, col="grey", edge=None, fs=7.5, bold=False, icon=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.12", fc=TINT[col], ec=edge or C.get(col, '#8a8985'), lw=1.0, zorder=2))
    if icon:
        ax.text(x + 0.16, y + h / 2, icon, ha="left", va="center", fontsize=fs + 4, color=C.get(col, MUTED), zorder=3)
        ax.text(x + 0.62 + (w - 0.62) / 2 - 0.05, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, zorder=3, weight="bold" if bold else "normal", linespacing=1.25)
    else:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, zorder=3, weight="bold" if bold else "normal", linespacing=1.25)


def arrow(ax, p, q, col=MUTED, style="-|>", ls="-", rad=0.0, lw=1.2, label=None, lp=None, fs=6.5):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=9, lw=lw, color=col, ls=ls, connectionstyle=f"arc3,rad={rad}", zorder=4, shrinkA=1, shrinkB=1))
    if label:
        lp = lp or ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
        ax.text(lp[0], lp[1], label, fontsize=fs, color=col, ha="center", va="center", zorder=5, bbox=dict(fc=SURF, ec="none", pad=0.6))


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h), dpi=300); fig.patch.set_facecolor(SURF)
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off"); return fig, ax


# ------------------------------------------------------------------ Figure: architecture
def architecture(path):
    W, H = 10.0, 13.1
    fig, ax = canvas(W, H)
    layers = [
        ("L1  Bioengineering data acquisition", "blue", "⚗", [("Electronic-nose\nsensor arrays", None), ("Bioreactor process\nsensors", None), ("Online water-quality\nprobes", None), ("Off-line laboratory\nassays (delayed)", None)]),
        ("L2  Preprocessing and data-quality assessment", "aqua", "⚙", [("Timestamp alignment,\nbounded forward fill", None), ("Baseline-ratio and\ntrend features", None), ("Integrity and range\nchecks", None), ("Leakage-safe grouping\n(batch / bottle / site)", None)]),
        ("L3  Intelligent learning", "violet", "◈", [("Gradient-boosted\nclassifier $f_t$", None), ("Class probabilities\n$p_t(k\\mid x)$", None), ("Prequential scores\n$s=1-p_t(y\\mid x)$", None), ("Periodic model refresh\n(weighted)", None)]),
        ("L4  Uncertainty quantification  (PACT core)", "orange", "⚖", [("Propensity-weighted\nconformal quantile $\\hat q_t$", None), ("Label-conditional\nthresholds (hazard)", None), ("Prediction set\n$C_t(x)$", None), ("Coverage-debt\ncontroller ($\\varepsilon_t$, $\\alpha_t$)", None)]),
        ("L5  Explainability", "yellow", "☀", [("TreeSHAP sensor\nattributions", None), ("Faithfulness and\nstability checks", None), ("Sensor-level attribution\ndrift monitor", None)]),
        ("L6  Security and integrity", "red", "⚠", [("Input stress screening\n(noise, dropout, evasion)", None), ("Consensus screening of\nverified labels", None), ("Propensity and decision\naudit log", None)]),
        ("L7  Biosafety triage and decision", "green", "✔", [("Automatic release\n(singleton, non-hazard)", None), ("Escalation\n(ambiguous / hazard in set)", None), ("Random audit\n(probability $\\varepsilon_t$)", None), ("Human / laboratory\nverification", None)]),
        ("L8  Deployment and monitoring", "grey", "\u2601", [("Edge node:\ninference and triage", None), ("Site server:\nrecalibration and refresh", None), ("Access control and\nintegrity hashes", None), ("Fixed seeds, versioned\nconfiguration", None)]),
    ]
    lh, gap, top = 1.28, 0.26, H - 0.55
    ypos = {}
    ax.text(0.25, H - 0.22, "Sensor stream  $x_t$", fontsize=8, color=MUTED, style="italic")
    for i, (title, col, icon, comps) in enumerate(layers):
        y = top - (i + 1) * lh - i * gap
        ypos[i] = y
        ax.add_patch(Rectangle((0.2, y), W - 0.4, lh, fc=SURF, ec=C.get(col, "#8a8985"), lw=1.3, zorder=1))
        ax.add_patch(Rectangle((0.2, y), 0.09, lh, fc=C.get(col, "#8a8985"), ec="none", zorder=2))
        ax.text(0.42, y + lh - 0.2, icon + "  " + title, fontsize=8.6, weight="bold", color=INK, va="center")
        n = len(comps); bw = (W - 1.0 - (n - 1) * 0.22) / n
        for j, (t, _) in enumerate(comps):
            box(ax, 0.5 + j * (bw + 0.22), y + 0.14, bw, 0.72, t, col)
            if j < n - 1 and i not in (0, 4, 5, 7):
                arrow(ax, (0.5 + j * (bw + 0.22) + bw, y + 0.5), (0.5 + (j + 1) * (bw + 0.22), y + 0.5), col=C.get(col, "#8a8985"), lw=1.0)
        if i > 0:
            arrow(ax, (W / 2, ypos[i - 1]), (W / 2, y + lh), col=MUTED, lw=1.4)
    # feedback paths
    xr = W - 0.02
    y7, y4, y3 = ypos[6] + 0.5, ypos[3] + 0.5, ypos[2] + 0.5
    ax.plot([W - 0.5, xr + 0.25, xr + 0.25], [y7, y7, y3], color=C["green"], lw=1.4, ls=(0, (4, 2)), zorder=4, clip_on=False)
    arrow(ax, (xr + 0.25, y4), (W - 0.5, y4), col=C["green"], lw=1.4)
    arrow(ax, (xr + 0.25, y3), (W - 0.5, y3), col=C["green"], lw=1.4)
    ax.text(xr + 0.42, (y7 + y3) / 2, "verified labels  $(x_i, y_i, \\pi_i)$  feed recalibration and refresh", rotation=90, fontsize=7.4, color=C["green"], ha="center", va="center", clip_on=False)
    ax.text(W / 2 + 0.12, ypos[6] - 0.13, "decision, set, explanation, risk flag, audit log", fontsize=6.8, color=MUTED, va="center")
    fig.savefig(path, bbox_inches="tight", pad_inches=0.08, facecolor=SURF); plt.close(fig)


# ------------------------------------------------------------------ Figure: dataflow
def dataflow(path):
    bw, bh, g = 1.5, 0.74, 0.44
    W, H = 6 * bw + 5 * g + 0.9, 7.0
    fig, ax = canvas(W, H)
    xs = [0.45 + i * (bw + g) for i in range(6)]
    y0, y1, y2, y3, y4 = 6.0, 4.75, 3.05, 1.55, 0.25
    fs = 7.0
    top = [("Raw sensor\nrecord $r_t$", "blue"), ("Feature vector\n$x_t\\in\\mathbb{R}^d$", "aqua"), ("Probabilities\n$p_t(\\cdot\\mid x_t)$", "violet"), ("Scores\n$s_t(x_t,k)$", "violet"), ("Prediction set\n$C_t(x_t)$", "orange"), ("Triage: release\nor escalate", "green")]
    for (t, c), x in zip(top, xs):
        box(ax, x, y1, bw, bh, t, c, fs=fs)
    for i, l in enumerate(["pre-\nprocess", "model\n$f_t$", "$1-p$", "$s\\leq\\hat q_t$", "rule"]):
        arrow(ax, (xs[i] + bw, y1 + bh / 2), (xs[i + 1], y1 + bh / 2))
        ax.text(xs[i] + bw + g / 2, y1 + bh / 2 + 0.23, l, fontsize=5.8, color=MUTED, ha="center", va="center", linespacing=1.0)
    box(ax, xs[2], y0, bw, bh, "TreeSHAP\nattribution $\\phi_t$", "yellow", fs=fs)
    arrow(ax, (xs[2] + bw / 2, y1 + bh), (xs[2] + bw / 2, y0))
    row2 = [(5, "Audit draw $A_t\\sim$\nBernoulli$(\\pi_t)$", "green"), (4, "Verification\n(laboratory / expert)", "green"), (3, "Label screening\n(consensus check)", "red"),
            (2, "Verified store $\\mathcal{V}_t$\n$\\{(s_i,y_i,\\pi_i)\\}$", "orange"), (1, "Weighted quantile\n$\\hat q_t$ (per class)", "orange"), (0, "Model refresh\nevery $R$ labels", "violet")]
    for i, t, c in row2:
        box(ax, xs[i], y2, bw, bh, t, c, fs=fs)
    arrow(ax, (xs[5] + bw / 2, y1), (xs[5] + bw / 2, y2 + bh), label="$\\pi_t$", lp=(xs[5] + bw / 2 + 0.2, (y1 + y2 + bh) / 2))
    for i in (5, 4, 3, 2):
        arrow(ax, (xs[i], y2 + bh / 2), (xs[i - 1] + bw, y2 + bh / 2), col=C["green"])
    yb = y2 - 0.28                                                # store -> refresh, routed below the quantile box
    ax.plot([xs[2] + 0.35, xs[2] + 0.35, xs[0] + bw / 2], [y2, yb, yb], color=C["green"], lw=1.2, zorder=4)
    arrow(ax, (xs[0] + bw / 2, yb), (xs[0] + bw / 2, y2), col=C["green"])
    arrow(ax, (xs[1] + bw / 2, y2 + bh), (xs[4] + 0.3, y1), col=C["orange"], ls=(0, (4, 2)), label="$\\hat q_t$", lp=(xs[2] + bw + g / 2, 4.02))
    arrow(ax, (xs[0] + bw / 2, y2 + bh), (xs[2] + 0.3, y1), col=C["violet"], ls=(0, (4, 2)), label="$f_{t+1}$", lp=(xs[1] + 0.3, 4.2))
    box(ax, xs[2], y3, bw, bh, "Coverage-debt\nestimate $\\hat m_t$", "orange", fs=fs)
    box(ax, xs[3], y3, bw, bh, "Controller\n$\\varepsilon_t,\\ \\alpha_t$", "orange", fs=fs)
    arrow(ax, (xs[2] + bw / 2 + 0.3, y2), (xs[2] + bw / 2 + 0.3, y3 + bh), col=C["orange"])
    arrow(ax, (xs[2] + bw, y3 + bh / 2), (xs[3], y3 + bh / 2), col=C["orange"])
    ax.plot([xs[3] + bw, xs[5] + bw / 2], [y3 + bh / 2, y3 + bh / 2], color=C["orange"], lw=1.2, ls=(0, (4, 2)), zorder=4)
    arrow(ax, (xs[5] + bw / 2, y3 + bh / 2), (xs[5] + bw / 2, y2), col=C["orange"], ls=(0, (4, 2)), label="$\\varepsilon_t$", lp=(xs[4] + bw / 2, y3 + bh / 2 + 0.16))
    ax.plot([xs[3] + bw / 2, xs[3] + bw / 2], [y3 + bh, y3 + bh + 0.0], lw=0)
    box(ax, xs[0], y4, xs[5] + bw - xs[0], 0.62, "Decision record:  class or prediction set  |  target coverage level  |  top-attributed sensors  |  risk flag  |  propensity $\\pi_t$  |  audit outcome", "grey", fs=fs)
    xl, xr = 0.12, W - 0.12
    ax.plot([xs[2], xl, xl], [y0 + bh / 2, y0 + bh / 2, y4 + 0.31], color=MUTED, lw=1.2, zorder=4); arrow(ax, (xl, y4 + 0.31), (xs[0], y4 + 0.31))
    ax.plot([xs[5] + bw, xr, xr], [y1 + bh / 2, y1 + bh / 2, y4 + 0.31], color=MUTED, lw=1.2, zorder=4); arrow(ax, (xr, y4 + 0.31), (xs[5] + bw, y4 + 0.31))
    ax.text(xs[0], y1 + bh + 0.22, "Forward path (every sample)", fontsize=7.5, weight="bold", color=INK)
    ax.text(xs[4] - 0.2, y2 + bh + 0.14, "Feedback path (verified samples only)", fontsize=7.5, weight="bold", color=C["green"])
    ax.text(xs[0], y3 + bh / 2, "Monitoring and control", fontsize=7.5, weight="bold", color=C["orange"], va="center")
    fig.savefig(path, bbox_inches="tight", pad_inches=0.08, facecolor=SURF); plt.close(fig)


# ------------------------------------------------------------------ Figure: algorithm workflow
def workflow(path):
    W, H = 8.2, 10.3
    fig, ax = canvas(W, H)
    cx, bw, bh = 3.4, 3.0, 0.62
    steps = [("Receive sample $x_t$", "blue"), ("Predict $p_t(\\cdot\\mid x_t)$ and scores $s_t(x_t,k)$", "violet"), ("Compute $\\hat q_t$ from $\\mathcal{V}_t$ with weights $1/\\pi_i$", "orange"),
             ("Form set $C_t(x_t)=\\{k: s_t(x_t,k)\\leq\\hat q_{t,k}\\}$", "orange")]
    y = H - 0.9; ys = []
    for t, c in steps:
        box(ax, cx - bw / 2, y, bw, bh, t, c); ys.append(y); y -= 0.95
    for a, b in zip(ys[:-1], ys[1:]):
        arrow(ax, (cx, a), (cx, b + bh))
    yd = y - 0.25
    ax.add_patch(plt.Polygon([(cx, yd + 0.95), (cx + 1.75, yd + 0.4), (cx, yd - 0.15), (cx - 1.75, yd + 0.4)], fc=TINT["green"], ec=C["green"], lw=1.0, zorder=2))
    ax.text(cx, yd + 0.4, "$|C_t|=1$ and\nhazard $\\notin C_t$ ?", ha="center", va="center", fontsize=7.5, zorder=3)
    arrow(ax, (cx, ys[-1]), (cx, yd + 0.95))
    box(ax, cx - 2.35, yd - 1.35, 2.1, bh, "Release automatically\n$\\pi_t=\\varepsilon_t$", "green")
    box(ax, cx + 0.25, yd - 1.35, 2.1, bh, "Escalate\n$\\pi_t=\\min(1,\\kappa/\\hat r_t)$", "red")
    arrow(ax, (cx - 1.3, yd + 0.12), (cx - 1.3, yd - 1.35 + bh), label="yes", lp=(cx - 1.55, yd - 0.3))
    arrow(ax, (cx + 1.3, yd + 0.12), (cx + 1.3, yd - 1.35 + bh), label="no", lp=(cx + 1.55, yd - 0.3))
    y5 = yd - 2.45
    box(ax, cx - bw / 2, y5, bw, bh, "Draw $A_t\\sim$ Bernoulli$(\\pi_t)$", "green")
    arrow(ax, (cx - 1.3, yd - 1.35), (cx - 0.6, y5 + bh)); arrow(ax, (cx + 1.3, yd - 1.35), (cx + 0.6, y5 + bh))
    y6 = y5 - 0.95
    box(ax, cx - bw / 2, y6, bw, bh, "If $A_t=1$: verify $y_t$, screen label,\nstore $(s_t(x_t,y_t), y_t, \\pi_t)$ in $\\mathcal{V}_t$", "green")
    arrow(ax, (cx, y5), (cx, y6 + bh))
    y7 = y6 - 0.95
    box(ax, cx - bw / 2, y7, bw, bh, "Update debt $\\hat m_t$; set $\\varepsilon_{t+1}$, $\\alpha_{t+1}$;\nrefresh $f$ after $R$ new labels", "orange")
    arrow(ax, (cx, y6), (cx, y7 + bh))
    box(ax, 5.75, ys[1], 2.15, bh, "TreeSHAP $\\phi_t$ for the\npredicted class", "yellow")
    arrow(ax, (cx + bw / 2, ys[1] + bh / 2), (5.75, ys[1] + bh / 2))
    box(ax, 5.75, y7, 2.15, bh, "Decision record and\naudit log", "grey")
    arrow(ax, (cx + bw / 2, y7 + bh / 2), (5.75, y7 + bh / 2))
    ax.plot([cx - bw / 2, 0.35, 0.35], [y7 + bh / 2, y7 + bh / 2, ys[0] + bh / 2], color=MUTED, lw=1.2, ls=(0, (4, 2)))
    arrow(ax, (0.35, ys[0] + bh / 2), (cx - bw / 2, ys[0] + bh / 2), ls=(0, (4, 2)))
    ax.text(0.2, (y7 + ys[0]) / 2, "$t\\leftarrow t+1$", rotation=90, fontsize=7.5, color=MUTED, va="center", ha="center")
    fig.savefig(path, bbox_inches="tight", pad_inches=0.08, facecolor=SURF); plt.close(fig)


if __name__ == "__main__":
    architecture("figures/architecture.png"); dataflow("figures/dataflow.png"); workflow("figures/algorithm_workflow.png")
