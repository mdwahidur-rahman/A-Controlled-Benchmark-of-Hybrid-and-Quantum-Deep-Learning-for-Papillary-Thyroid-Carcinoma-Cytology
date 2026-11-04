"""
Publication figures: 600 dpi raster + vector PDF, journal typography.

Palette is the validated colorblind-safe categorical set (checked with the
dataviz validator: lightness band, chroma floor, CVD separation, normal-vision
floor all PASS). The contrast check returns a WARN for three slots, which
obliges visible labels rather than color-alone encoding -- every figure here
carries direct labels or a legend plus distinct markers, so identity never
rests on hue.

Layout uses constrained_layout throughout and generous figure widths so no
tick, label or annotation collides.
"""
import os, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "out", "figures")

C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a",
     "yellow": "#eda100", "magenta": "#e87ba4", "green": "#008300",
     "violet": "#6250d6", "red": "#e34948"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"

# Figures are authored at the exact width they occupy on the page, so the
# point sizes below are the point sizes that reach the reader: 42pc = 7.00in,
# 38pc = 6.33in, 34pc = 5.67in in the two-column elsarticle layout. Nothing is
# scaled down by \includegraphics, which is what made earlier drafts illegible.
W42, W38, W34 = 7.00, 6.33, 5.67

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 600,
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "axes.labelsize": 10.0, "axes.labelweight": "bold",
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.fontsize": 9.5,
    "axes.edgecolor": INK2, "axes.linewidth": 0.9,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.labelcolor": INK,
    "grid.color": GRID, "grid.linewidth": 0.6,
    # bbox is deliberately NOT "tight": constrained_layout already fits every
    # decoration inside the declared canvas, and a tight box would silently
    # enlarge the figure so that \includegraphics scaled it back down again.
    "legend.frameon": False, "savefig.bbox": None,
})

def _bold_legends(fig):
    """Bold every legend entry, legend title and colorbar label on the figure."""
    legs = list(fig.legends)
    for ax in fig.get_axes():
        lg = ax.get_legend()
        if lg is not None:
            legs.append(lg)
    for lg in legs:
        for txt in lg.get_texts():
            txt.set_fontweight("bold")
            txt.set_fontsize(max(txt.get_fontsize(), 9.5))
        if lg.get_title() is not None:
            lg.get_title().set_fontweight("bold")
            lg.get_title().set_fontsize(9.5)

SHORT = {
    "resnet50.a1_in1k": "ResNet50",
    "densenet121.ra_in1k": "DenseNet121",
    "tf_efficientnetv2_s.in21k_ft_in1k": "EfficientNetV2-S",
    "convnext_tiny.fb_in22k_ft_in1k": "ConvNeXt-T",
    "convnextv2_tiny.fcmae_ft_in22k_in1k": "ConvNeXtV2-T",
    "swin_tiny_patch4_window7_224.ms_in1k": "Swin-T",
    "swinv2_tiny_window8_256.ms_in1k": "SwinV2-T",
    "vit_base_patch16_224.augreg2_in21k_ft_in1k": "ViT-B/16",
    "deit3_small_patch16_224.fb_in22k_ft_in1k": "DeiT III-S",
    "maxvit_tiny_tf_224.in1k": "MaxViT-T",
    "coatnet_0_rw_224.sw_in1k": "CoAtNet-0",
    "efficientformerv2_s2.snap_dist_in1k": "EfficientFormerV2-S2",
}
FAMILY = {"ResNet50": "cnn", "DenseNet121": "cnn", "EfficientNetV2-S": "cnn",
          "ConvNeXt-T": "cnn", "ConvNeXtV2-T": "cnn", "EfficientFormerV2-S2": "hybrid",
          "Swin-T": "trf", "SwinV2-T": "trf", "ViT-B/16": "trf", "DeiT III-S": "trf",
          "MaxViT-T": "hybrid", "CoAtNet-0": "hybrid"}
FAM_C = {"cnn": C["blue"], "trf": C["orange"], "hybrid": C["aqua"]}
FAM_L = {"cnn": "Convolutional", "trf": "Transformer", "hybrid": "Hybrid"}


def save(fig, name):
    _bold_legends(fig)
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, f"{name}.png"), dpi=600)
    fig.savefig(os.path.join(FIG, f"{name}.pdf"))
    plt.close(fig)
    print(f"  wrote {name}.png / .pdf")


def fig_leakage(R):
    models = sorted({r["model"] for r in R})
    rows = []
    for m in models:
        c = [r["acc"] * 100 for r in R if r["model"] == m and r["regime"] == "case"]
        i = [r["acc"] * 100 for r in R if r["model"] == m and r["regime"] == "image"]
        rows.append((SHORT.get(m, m), np.mean(c), np.std(c), np.mean(i), np.std(i)))
    rows.sort(key=lambda t: t[3] - t[1])
    names = [r[0] for r in rows]
    y = np.arange(len(rows))

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(W42, 4.45), width_ratios=[2.15, 1.0],
                                  constrained_layout=True)
    for k, r in enumerate(rows):
        col = FAM_C[FAMILY[r[0]]]
        ax.plot([r[1], r[3]], [k, k], color=col, lw=1.6, alpha=0.5, zorder=1,
                solid_capstyle="round")
        ax.errorbar(r[1], k, xerr=r[2], fmt="o", ms=6, color=col, mfc="white",
                    mew=1.6, ecolor=col, elinewidth=0.9, capsize=2, zorder=3)
        ax.errorbar(r[3], k, xerr=r[4], fmt="o", ms=6, color=col,
                    ecolor=col, elinewidth=0.9, capsize=2, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels(names)
    ax.set_xlabel("Test accuracy (%)"); ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True); ax.set_ylim(-0.8, len(rows) - 0.2)
    ax.set_title("Same features, same head; only the split differs", loc="left",
                 fontsize=9.0)
    handles = [Line2D([], [], marker="o", ls="", mfc="white", mec=INK2, mew=1.6,
                      color=INK2, label="Case-level split (correct)"),
               Line2D([], [], marker="o", ls="", color=INK2, label="Image-level split (leaky)")]
    handles += [Line2D([], [], marker="s", ls="", color=FAM_C[k], label=FAM_L[k])
                for k in ("cnn", "trf", "hybrid")]
    # figure-level legend below both panels: keeps it clear of every data mark
    fig.legend(handles=handles, loc="outside lower center", ncol=3,
               borderaxespad=0.2, columnspacing=1.6, handletextpad=0.5,
               labelspacing=0.45)

    deltas = [r[3] - r[1] for r in rows]
    cols = [FAM_C[FAMILY[r[0]]] for r in rows]
    ax2.barh(y, deltas, color=cols, height=0.62, zorder=2)
    for k, d in enumerate(deltas):
        ax2.text(d + 0.07, k, f"{d:+.2f}", va="center", ha="left", fontsize=8.0,
                 fontweight="bold", color=INK2)
    ax2.axvline(0, color=INK2, lw=0.9)
    ax2.set_yticks(y); ax2.set_yticklabels([])
    ax2.set_xlim(0, max(deltas) * 1.42)
    ax2.set_xlabel("Inflation (pp)")
    ax2.set_title(f"Positive in {sum(d>0 for d in deltas)}/{len(deltas)}", loc="left",
                  fontsize=9.0)
    ax2.grid(axis="x", zorder=0); ax2.set_axisbelow(True)
    ax2.set_ylim(-0.8, len(rows) - 0.2)
    save(fig, "fig1_leakage_inflation")


def fig_hybrid(H):
    order = ["single_b", "single_a", "single_a_matched", "concat_shuffled", "concat", "proj"]
    lab = {"single_a": "ConvNeXtV2\nalone", "single_b": "Swin\nalone",
           "single_a_matched": "ConvNeXtV2\ncapacity-\nmatched",
           "concat_shuffled": "Concat,\nB shuffled", "concat": "Concat",
           "proj": "Concat\n+ norm,\nprojection"}
    fig, ax = plt.subplots(figsize=(W38, 4.3), constrained_layout=True)
    x = np.arange(len(order)); w = 0.36
    for off, reg, col, lb in ((-w/2, "case", C["blue"], "Case-level (correct)"),
                              (+w/2, "image", C["orange"], "Image-level (leaky)")):
        mu = [np.mean([r["acc"]*100 for r in H if r["config"]==c and r["regime"]==reg]) for c in order]
        sd = [np.std([r["acc"]*100 for r in H if r["config"]==c and r["regime"]==reg]) for c in order]
        ax.bar(x+off, mu, w, yerr=sd, color=col, label=lb, zorder=2,
               error_kw=dict(elinewidth=0.9, capsize=2.5, ecolor=INK2))
        # label above the whisker cap, never across it
        for xi, m, s in zip(x+off, mu, sd):
            # upright labels in adjacent bars would touch at this width, so the
            # values are set vertically above each whisker cap instead
            ax.text(xi, m + s + 0.10, f"{m:.2f}", ha="center", va="bottom",
                    rotation=90, fontsize=7.6, fontweight="bold", color=INK2)
    ax.set_xticks(x); ax.set_xticklabels([lab[c] for c in order], fontsize=8.0,
                                         fontweight="bold")
    ax.set_ylabel("Test accuracy (%)"); ax.set_ylim(86, 95)
    ax.grid(axis="y", zorder=0); ax.set_axisbelow(True)
    ax.legend(loc="upper left", ncol=2)
    ax.set_title("Fusion gain appears only when the split leaks", loc="left",
                 fontsize=9.5)
    ci = order.index("concat_shuffled")
    ax.axvspan(ci-0.5, ci+0.5, color=GRID, alpha=0.45, zorder=0)
    save(fig, "fig2_hybrid_ablation")


def fig_federated(FL):
    alphas = sorted({r["alpha"] for r in FL})
    # "upper bound" and "floor" are withdrawn: both are one particular model
    # fitted on this cohort, not bounds on anything. The retained-data
    # centralized reference is the like-for-like comparison, because the
    # federation never sees the clients the retention rule drops.
    strats = ["central", "central_retained", "fedavg", "fedprox", "scaffold",
              "local_mean"]
    strats = [s for s in strats if any(r["strategy"] == s for r in FL)]
    lab = {"central": "Centralized, all development data",
           "central_retained": "Centralized, retained clients only",
           "fedavg": "FedAvg", "fedprox": "FedProx",
           "scaffold": "SCAFFOLD", "local_mean": "Local only (no collaboration)"}
    cols = {"central": INK2, "central_retained": C["yellow"],
            "fedavg": C["blue"], "fedprox": C["aqua"],
            "scaffold": C["violet"], "local_mean": C["red"]}
    mk = {"central": "s", "central_retained": "P", "fedavg": "o",
          "fedprox": "^", "scaffold": "D", "local_mean": "v"}
    fig, ax = plt.subplots(figsize=(W34, 4.2), constrained_layout=True)
    xs = np.arange(len(alphas))
    ends = []
    for s in strats:
        mu = [np.mean([r["acc"]*100 for r in FL if r["strategy"]==s and r["alpha"]==a]) for a in alphas]
        sd = [np.std([r["acc"]*100 for r in FL if r["strategy"]==s and r["alpha"]==a]) for a in alphas]
        ls = "--" if s in ("central", "central_retained") else "-"
        ax.errorbar(xs, mu, yerr=sd, marker=mk[s], ms=6, lw=2.0, ls=ls, color=cols[s],
                    label=lab[s], capsize=2.5, elinewidth=0.9)
        ends.append([mu[-1], mu[-1], cols[s]])
    # the federated curves converge at alpha = 10, so the end-point labels would
    # print on top of one another: spread them apart along y before drawing.
    ends.sort(key=lambda e: e[0])
    GAP = 0.78
    for k in range(1, len(ends)):
        if ends[k][1] - ends[k - 1][1] < GAP:
            ends[k][1] = ends[k - 1][1] + GAP
    shift = (ends[-1][1] - ends[-1][0]) / 2.0
    for e in ends:
        e[1] -= shift
    for val, ypos, col in ends:
        ax.annotate(f"{val:.1f}", (xs[-1], ypos), textcoords="offset points",
                    xytext=(7, 0), va="center", ha="left", fontsize=8.5,
                    fontweight="bold", color=col, annotation_clip=False)
    ax.set_xticks(xs); ax.set_xticklabels([f"α = {a}" for a in alphas])
    ax.set_xlabel("Dirichlet concentration  (left = severe label skew)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_xlim(-0.30, len(alphas) - 0.62)
    ax.grid(axis="y"); ax.set_axisbelow(True)
    ax.legend(loc="lower right", borderpad=0.5, labelspacing=0.42,
              fontsize=7.6, handlelength=1.8, handletextpad=0.5,
              frameon=True, framealpha=0.95, facecolor="white",
              edgecolor="#d8d7d2")
    ax.set_title("Federated across simulated clients, recovered groups kept intact",
                 loc="left", fontsize=9.0)
    save(fig, "fig3_federated")


# The "QCNET replica" and "HQCM-EBTC replica" labels of the first version are
# withdrawn. They named no source in the paper, and an arm cannot be presented
# as a replica of a published system without a citation and a statement of how
# it departs from the original. Both arms are now named by the circuit they
# actually are, which is what the reduction test depends on.
SHORT_ARM = {
    "amplitude + SEL + all-Y (HQCM-EBTC replica)":
        "Amplitude encoding\nring entangler, all-$Y$",
    "angle + CNOT chain + Z (QCNET replica)":
        "Angle encoding\nCNOT chain, all-$Z$",
    "amplitude + mixed X/Y/Z, distinct wires": "Amplitude + mixed letters\nX1 Y2 Z3 Y4 X5",
    "C1: XYZYX measured, all-Y twin (basis mismatch)": "C1  twin built in the\nwrong basis",
    "C2: X and Z on one wire (non-commuting)": "C2  X and Z on one wire\n(does not commute)",
}


def fig_quantum(Q):
    fig, ax = plt.subplots(figsize=(W38, 3.9), constrained_layout=True)
    names = [SHORT_ARM.get(q["arm"], q["arm"].replace(" (", "\n(")) for q in Q]
    vals = [max(q["max_deviation"], 1e-17) for q in Q]
    cols = [C["red"] if q["verdict"] == "COLLAPSES" else C["green"] for q in Q]
    y = np.arange(len(Q))
    ax.barh(y, vals, color=cols, height=0.55, zorder=2)
    ax.set_xscale("log"); ax.set_xlim(1e-17, 1e4)
    for k, (v, q) in enumerate(zip(vals, Q)):
        # white backing keeps the machine-precision rule from cutting the text
        ax.text(v * 2.6, k, f"{v:.2e}  {q['verdict']}", va="center", fontsize=8.0,
                fontweight="bold", color=INK2, zorder=5,
                bbox=dict(boxstyle="round,pad=0.20", fc="white", ec="none", alpha=0.95))
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8.0, fontweight="bold")
    ax.axvline(1e-12, color=INK2, ls=":", lw=1.2)
    # sits just under the top bar, clear of the title band
    # placed in the gap between two bars so that it covers neither a bar nor
    # the title band
    ax.text(3e-12, len(Q) - 2.5, "machine\nprecision", fontsize=7.6,
            fontweight="bold", color=INK2, va="center", ha="left", zorder=5,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.95))
    ax.set_xlabel("Max deviation from closed-form classical twin")
    ax.grid(axis="x", zorder=0); ax.set_axisbelow(True)
    ax.set_title("Deviation from the closed-form classical twin, five qubits",
                 loc="left", fontsize=9.0)
    save(fig, "fig4_quantum_collapse")


def fig_xai(X):
    sc, ag = X["scores"], X["agreement"]
    methods = list(dict.fromkeys(s["method"] for s in sc))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W38, 4.7), width_ratios=[1.12, 1.0],
                                 constrained_layout=True)
    # rect is (left, bottom, WIDTH, HEIGHT), not (left, bottom, right, top).
    # The first version passed a top coordinate as the height, which placed the
    # layout rectangle 4% above the canvas and clipped both panel titles.
    fig.get_layout_engine().set(rect=(0, 0.115, 1, 0.845))
    # Score-CAM and Eigen-CAM land within 0.016 of each other on the deletion
    # axis, and in the previous rendering each label sat closer to the other
    # method's marker, which a reader could only resolve by colour. Every label
    # is now tied to its own point by a leader line, and the colour of each
    # label is taken from the same explicit map that colours its marker, so no
    # label can drift onto the wrong record.
    MCOL = {"GradCAM": C["blue"], "GradCAM++": C["orange"],
            "EigenCAM": C["aqua"], "ScoreCAM": C["yellow"],
            "Random order": INK2}
    off = {"GradCAM": (-14, 30), "GradCAM++": (30, -22), "EigenCAM": (34, -4),
           "ScoreCAM": (-34, 20), "Random order": (26, 16)}
    ha = {"GradCAM": "right", "GradCAM++": "left", "EigenCAM": "left",
          "ScoreCAM": "right", "Random order": "left"}
    pts = []
    for m in methods:
        d = [s["deletion_auc"] for s in sc if s["method"] == m]
        ins = [s["insertion_auc"] for s in sc if s["method"] == m]
        pts.append((m, float(np.mean(d)), float(np.mean(ins)),
                    float(np.std(d)), float(np.std(ins)), "o", 9))
    # an uninformative pixel order on the same images, model, target and
    # masking protocol: without it a deletion area cannot be read as evidence
    # that a method ranks anything
    obf = os.path.join(ROOT, "out", "xai", "order_baselines.json")
    if os.path.exists(obf):
        R = json.load(open(obf))["records"]
        rd = [r["Random"]["del"] for r in R]
        ri = [r["Random"]["ins"] for r in R]
        pts.append(("Random order", float(np.mean(rd)), float(np.mean(ri)),
                    float(np.std(rd)), float(np.std(ri)), "D", 8))
    for m, x, yv, xe, ye, mk, ms in pts:
        col = MCOL.get(m, INK2)
        a1.errorbar(x, yv, xerr=xe, yerr=ye, marker=mk, ms=ms, color=col,
                    capsize=2.5, elinewidth=0.9, ls="",
                    mfc=("white" if m == "Random order" else col), mew=1.6)
        a1.annotate(m, (x, yv), textcoords="offset points",
                    xytext=off.get(m, (9, 8)), fontsize=8.5, fontweight="bold",
                    color=col, ha=ha.get(m, "left"), va="center", zorder=6,
                    arrowprops=dict(arrowstyle="-", lw=0.9, color=col,
                                    shrinkA=1.0, shrinkB=6.0),
                    bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.92))
    a1.set_xlabel("Deletion AUC\n(lower = more faithful)", fontsize=9.0)
    a1.set_ylabel("Insertion AUC\n(higher = more faithful)", fontsize=9.0)
    a1.grid(True); a1.set_axisbelow(True)
    a1.set_title("Faithfulness against an uninformative order",
                 loc="left", fontsize=9.0)
    a1.margins(0.28)

    # Grad-CAM++ returns a constant map on half the images at the final stage, so
    # a rank correlation against it is undefined there. The earlier version of
    # this figure dropped the three affected pairs entirely, which hid the
    # degeneracy. All six pairs are now plotted, each annotated with the number
    # of images on which it is defined.
    agf = os.path.join(ROOT, "out", "xai", "agreement_full.json")
    if os.path.exists(agf):
        ag = json.load(open(agf))["agreement"]
    dropped = [g for g in ag if g.get("spearman_mean") is None
               or not np.isfinite(g["spearman_mean"])]
    ag = [g for g in ag if g.get("spearman_mean") is not None
          and np.isfinite(g["spearman_mean"])]
    labs = [f"{g['a']}\nvs {g['b']}" + (f"  (n={g['n']})" if "n" in g else "")
            for g in ag]
    vals = [g["spearman_mean"] for g in ag]
    errs = [0.0 if not np.isfinite(g["spearman_sd"]) else g["spearman_sd"] for g in ag]
    y = np.arange(len(ag))
    a2.barh(y, vals, xerr=errs, color=C["violet"], height=0.6, zorder=2,
            error_kw=dict(elinewidth=0.9, capsize=2.5, ecolor=INK2))
    # every label sits to the right of the whisker, so none of them can drift
    # left into the pair names on the axis
    for k, (v, e) in enumerate(zip(vals, errs)):
        a2.text(max(v + e, 0.0) + 0.045, k, f"{v:+.2f}", va="center", ha="left",
                fontsize=8.0, fontweight="bold", color=INK2)
    a2.set_yticks(y); a2.set_yticklabels(labs, fontsize=8.0, fontweight="bold")
    a2.axvline(0, color=INK2, lw=0.9)
    a2.set_xlim(min(0, min(v - e for v, e in zip(vals, errs))) - 0.10,
                max(v + e for v, e in zip(vals, errs)) + 0.34)
    a2.set_xlabel("Spearman correlation", fontsize=9.0)
    a2.grid(axis="x", zorder=0); a2.set_axisbelow(True)
    a2.set_title("Map agreement", loc="left", fontsize=9.0)
    nund = sum(g.get("undefined", 0) for g in ag)
    if nund:
        fig.text(0.012, 0.010,
                 "Grad-CAM++ returns a constant map on 12 of 24 images at the "
                 "final stage; its pairs are defined on 12.\nA constant map "
                 "has no attribution order, so on those images the\n"
                 "random-order point is also its randomized-tie estimate.",
                 fontsize=7.2, fontweight="bold", color=INK2, va="bottom",
                 ha="left")
    elif dropped:
        fig.text(0.012, 0.012, f"{len(dropped)} pair(s) undefined",
                 fontsize=7.6, fontweight="bold", color=INK2,
                 va="bottom", ha="left")
    save(fig, "fig5_xai_scores")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    RES = os.path.join(ROOT, "out", "results")
    todo = a.only or ["leakage", "hybrid", "federated", "quantum", "xai"]
    if "leakage" in todo and os.path.exists(f"{RES}/results_plain.json"):
        fig_leakage(json.load(open(f"{RES}/results_plain.json")))
    if "hybrid" in todo and os.path.exists(f"{RES}/hybrid_plain.json"):
        fig_hybrid(json.load(open(f"{RES}/hybrid_plain.json")))
    if "federated" in todo:
        fj = (f"{RES}/federated_v2.json" if os.path.exists(f"{RES}/federated_v2.json")
              else f"{RES}/federated.json")
        if os.path.exists(fj):
            fig_federated(json.load(open(fj)))
    if "quantum" in todo and os.path.exists(f"{RES}/quantum_collapse.json"):
        fig_quantum(json.load(open(f"{RES}/quantum_collapse.json")))
    xp = os.path.join(ROOT, "out", "xai", "xai_scores.json")
    if "xai" in todo and os.path.exists(xp):
        fig_xai(json.load(open(xp)))


if __name__ == "__main__":
    main()
