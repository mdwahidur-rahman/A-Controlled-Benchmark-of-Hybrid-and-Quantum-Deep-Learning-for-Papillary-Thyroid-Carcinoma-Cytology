"""
Third figure set: quantum control bank, and per-model comparison views.

fig14  quantum control bank -- the collapsing circuit and its classical twin
       plotted as overlapping marks, because they are identical to every digit
fig15  grouped bars, every architecture x every headline metric
fig16  line chart, metric trajectories across architectures ranked by MCC
fig17  paired per-model case-vs-image bars for all metrics at once
"""
import os, sys, json, argparse, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "out", "results")
FIG = os.path.join(ROOT, "out", "figures")
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
     "magenta": "#e87ba4", "green": "#008300", "violet": "#6250d6", "red": "#e34948"}
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
    "resnet50.a1_in1k": "ResNet50", "densenet121.ra_in1k": "DenseNet121",
    "tf_efficientnetv2_s.in21k_ft_in1k": "EfficientNetV2-S",
    "convnext_tiny.fb_in22k_ft_in1k": "ConvNeXt-T",
    "convnextv2_tiny.fcmae_ft_in22k_in1k": "ConvNeXtV2-T",
    "swin_tiny_patch4_window7_224.ms_in1k": "Swin-T",
    "swinv2_tiny_window8_256.ms_in1k": "SwinV2-T",
    "vit_base_patch16_224.augreg2_in21k_ft_in1k": "ViT-B/16",
    "deit3_small_patch16_224.fb_in22k_ft_in1k": "DeiT III-S",
    "maxvit_tiny_tf_224.in1k": "MaxViT-T", "coatnet_0_rw_224.sw_in1k": "CoAtNet-0",
    "efficientformerv2_s2.snap_dist_in1k": "EfficientFormerV2-S2"}
FAMILY = {"ResNet50": "CNN", "DenseNet121": "CNN", "EfficientNetV2-S": "CNN",
          "ConvNeXt-T": "CNN", "ConvNeXtV2-T": "CNN", "Swin-T": "Transformer",
          "SwinV2-T": "Transformer", "ViT-B/16": "Transformer",
          "DeiT III-S": "Transformer", "MaxViT-T": "Hybrid",
          "CoAtNet-0": "Hybrid", "EfficientFormerV2-S2": "Hybrid"}
FAM_C = {"CNN": C["blue"], "Transformer": C["orange"], "Hybrid": C["aqua"]}


def save(fig, name):
    _bold_legends(fig)
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, f"{name}.png"), dpi=600)
    fig.savefig(os.path.join(FIG, f"{name}.pdf"))
    plt.close(fig)
    print(f"  wrote {name}")


def fig_qml():
    Q = json.load(open(os.path.join(RES, "qml_arm.json")))
    order = ["identity", "norm_only", "dense_k", "dense_k_norm",
             "vqc_amp_mixed", "vqc_amp_Y", "twin_amp_Y"]
    # kept short so the seven labels do not touch at 42pc; the blocks are
    # described in full in the caption
    lab = {"identity": "Identity", "norm_only": "Unit\nnorm",
           "dense_k": "Dense",
           "dense_k_norm": "Dense\n+norm",
           "vqc_amp_mixed": "VQC\nmixed",
           "vqc_amp_Y": "VQC\nall-Y", "twin_amp_Y": "Twin"}
    col = {"identity": INK2, "norm_only": C["aqua"], "dense_k": C["blue"],
           "dense_k_norm": C["orange"], "vqc_amp_mixed": C["green"],
           "vqc_amp_Y": C["red"], "twin_amp_Y": C["violet"]}
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(W42, 3.9), width_ratios=[1.85, 1.0],
                                  constrained_layout=True)
    x = np.arange(len(order)); w = 0.40
    for off, rg, hatch, lb in ((-w/2, "case", None, "Case-level (correct)"),
                               (+w/2, "image", "///", "Image-level (leaky)")):
        mu = [np.mean([r["acc"] for r in Q if r["arm"] == a and r["regime"] == rg]) * 100
              for a in order]
        sd = [np.std([r["acc"] for r in Q if r["arm"] == a and r["regime"] == rg]) * 100
              for a in order]
        ax.bar(x + off, mu, w, yerr=sd, color=[col[a] for a in order], hatch=hatch,
               edgecolor="white", linewidth=0.6, zorder=2,
               error_kw=dict(elinewidth=0.9, capsize=2.5, ecolor=INK2))
        for xi, m, s in zip(x + off, mu, sd):
            # upright values would touch between paired bars at this width
            ax.text(xi, m + s + 0.5, f"{m:.2f}", ha="center", va="bottom",
                    rotation=90, fontsize=7.4, fontweight="bold", color=INK2)
    ax.set_xticks(x); ax.set_xticklabels([lab[a] for a in order], fontsize=7.6,
                                         fontweight="bold")
    ax.set_ylabel("Test accuracy (%)"); ax.set_ylim(60, 102)
    ax.grid(axis="y", zorder=0); ax.set_axisbelow(True)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, fc=INK2, label="Case-level (correct)"),
                       plt.Rectangle((0, 0), 1, 1, fc=INK2, hatch="///", ec="white",
                                     label="Image-level (leaky)")],
              loc="upper right", ncol=1, frameon=True, framealpha=0.95,
              facecolor="white", edgecolor="#d8d7d2", borderpad=0.4, fontsize=8.0,
              labelspacing=0.32, handlelength=1.4,
              handletextpad=0.5).set_zorder(6)
    ax.set_title("Quantum block against its control bank", loc="left", fontsize=9.0)
    ax.axvspan(4.5, 6.5, color=GRID, alpha=0.35, zorder=0)

    # twin identity: plot per-seed accuracy of VQC vs twin -- they lie on y=x
    cq = [r["acc"] * 100 for r in Q if r["arm"] == "vqc_amp_Y"]
    ct = [r["acc"] * 100 for r in Q if r["arm"] == "twin_amp_Y"]
    rg = [r["regime"] for r in Q if r["arm"] == "vqc_amp_Y"]
    lo, hi = min(cq + ct) - 2.5, max(cq + ct) + 2.5
    ax2.plot([lo, hi], [lo, hi], ls=":", lw=1.1, color=INK2, zorder=1)
    for q, t, g in zip(cq, ct, rg):
        ax2.plot(q, t, "o" if g == "case" else "^", ms=9, color=C["violet"],
                 mfc=C["violet"] if g == "image" else "white", mew=1.7, zorder=3)
    ax2.set_xlim(lo, hi); ax2.set_ylim(lo, hi)
    ax2.set_xlabel("Quantum circuit accuracy (%)", fontsize=8.4)
    ax2.set_ylabel("Classical twin accuracy (%)", fontsize=9.0)
    ax2.grid(True); ax2.set_axisbelow(True)
    ax2.set_title(f"On y = x  (max |Δ| = {max(abs(q-t) for q,t in zip(cq,ct)):.3f} pp)",
                  loc="left", fontsize=9.0)
    ax2.legend(handles=[Line2D([], [], marker="o", ls="", mfc="white", mec=C["violet"],
                               mew=1.7, color=C["violet"], label="Case-level"),
                        Line2D([], [], marker="^", ls="", color=C["violet"],
                               label="Image-level")], loc="upper left",
               fontsize=8.0, labelspacing=0.32, handletextpad=0.5)
    save(fig, "fig14_quantum_control_bank")


def _panel(regime="case"):
    P = json.load(open(os.path.join(RES, "panel.json")))
    rows = [r for r in P if r["regime"] == regime]
    rows.sort(key=lambda r: -r["mcc"])
    return rows


def fig_model_bars():
    rows = _panel("case")
    names = [SHORT[r["model"]] for r in rows]
    metrics = [("acc", "Accuracy", C["blue"]), ("sens", "Sensitivity", C["orange"]),
               ("spec", "Specificity", C["aqua"]), ("f1", "F1", C["violet"]),
               ("mcc", "MCC ×100", C["magenta"])]
    fig, ax = plt.subplots(figsize=(W42, 3.9), constrained_layout=True)
    n = len(metrics); x = np.arange(len(rows)); w = 0.84 / n
    for i, (k, lb, col) in enumerate(metrics):
        off = (i - (n - 1) / 2) * w
        v = [r[k] * 100 for r in rows]
        sd = [r.get(k + "_sd", 0) * 100 for r in rows]
        ax.bar(x + off, v, w, yerr=sd, color=col, label=lb, zorder=2,
               error_kw=dict(elinewidth=0.7, capsize=1.8, ecolor=INK2))
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=7.4, fontweight="bold")
    # family is carried by the color of the model name rather than by a badge
    # printed inside the bars, where it was unreadable
    for lbl, r in zip(ax.get_xticklabels(), rows):
        lbl.set_color(FAM_C[FAMILY[SHORT[r["model"]]]])
    ax.set_ylabel("Score (%)"); ax.set_ylim(70, 100)
    ax.grid(axis="y", zorder=0); ax.set_axisbelow(True)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5,
               columnspacing=1.6, handletextpad=0.45, borderaxespad=0.15,
               fontsize=8.5, handlelength=1.3)
    # the family key goes inside the panel, in the clear band above the bars,
    # so it cannot be confused with the metric key below the axis
    fam = ax.legend(handles=[Line2D([], [], marker="s", ls="", color=FAM_C[f], label=f)
                             for f in ("CNN", "Transformer", "Hybrid")],
                    title="Encoder name colored by family",
                    loc="upper right", ncol=3, frameon=True, framealpha=0.95,
                    facecolor="white", edgecolor="#d8d7d2", borderpad=0.4,
                    fontsize=8.0, title_fontsize=8.0, columnspacing=1.1,
                    handletextpad=0.4)
    fam.set_zorder(6)
    ax.add_artist(fam)
    ax.set_title("Case-level performance by architecture, ranked by MCC",
                 loc="left", pad=6, fontsize=9.0)
    save(fig, "fig15_model_comparison_bars")


def fig_model_lines():
    case = _panel("case")
    order = [r["model"] for r in case]
    img = {r["model"]: r for r in json.load(open(os.path.join(RES, "panel.json")))
           if r["regime"] == "image"}
    names = [SHORT[m] for m in order]
    x = np.arange(len(order))
    series = [("acc", "Accuracy", C["blue"], "o"), ("sens", "Sensitivity", C["orange"], "s"),
              ("spec", "Specificity", C["aqua"], "^"), ("f1", "F1", C["violet"], "D"),
              ("auc", "AUC ×100", C["green"], "v"), ("mcc", "MCC ×100", C["magenta"], "P")]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(W42, 6.1), height_ratios=[1.22, 1.0],
                                  constrained_layout=True, sharex=True)
    ends = []
    for k, lb, col, mk in series:
        v = [r[k] * 100 for r in case]
        ax.plot(x, v, marker=mk, ms=5.5, lw=1.8, color=col, label=lb)
        ends.append((v[-1], col))
    # annotate only end values that are far enough apart to not collide; the
    # legend already carries identity, so a clustered label adds nothing
    ends.sort(key=lambda t: t[0])
    keep = []
    for val, col in ends:
        if not keep or abs(val - keep[-1][0]) > 1.6:
            keep.append((val, col))
    for val, col in keep:
        ax.annotate(f"{val:.1f}", (x[-1], val), textcoords="offset points",
                    xytext=(6, 0), va="center", fontsize=7.6, fontweight="bold",
                    color=col, annotation_clip=False)
    ax.set_ylabel("Score (%)  — case-level", fontsize=9.0)
    ax.grid(True); ax.set_axisbelow(True)
    ax.set_xlim(-0.4, len(order) - 0.52)
    # open a clear band below the lowest curve so the legend never sits on data
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - 0.34 * (hi - lo), hi)
    ax.legend(loc="lower left", ncol=6, frameon=True, framealpha=0.95,
              facecolor="white", edgecolor="#d8d7d2", borderpad=0.4, fontsize=8.0,
              columnspacing=0.9, handlelength=1.3, handletextpad=0.4).set_zorder(6)
    ax.set_title("Metric trajectories across architectures (ranked by MCC)",
                 loc="left", fontsize=9.0)

    for k, lb, col, mk in [("acc", "Accuracy", C["blue"], "o"),
                           ("mcc", "MCC ×100", C["magenta"], "P"),
                           ("auc", "AUC ×100", C["green"], "v")]:
        d = [(img[m][k] - next(r[k] for r in case if r["model"] == m)) * 100 for m in order]
        ax2.plot(x, d, marker=mk, ms=5.5, lw=1.8, color=col, label=f"Δ {lb}")
    ax2.axhline(0, color=INK2, lw=1.0)
    ax2.set_ylabel("Image − case (pp)", fontsize=9.0)
    ax2.set_xticks(x); ax2.set_xticklabels(names, rotation=30, ha="right", fontsize=7.4,
                                           fontweight="bold")
    ax2.grid(True); ax2.set_axisbelow(True)
    lo2, hi2 = ax2.get_ylim()
    ax2.set_ylim(lo2, hi2 + 0.40 * (hi2 - lo2))
    ax2.legend(loc="upper right", ncol=3, frameon=True, framealpha=0.95,
               facecolor="white", edgecolor="#d8d7d2", borderpad=0.4, fontsize=8.0,
               columnspacing=0.9, handlelength=1.3, handletextpad=0.4).set_zorder(6)
    ax2.set_title("Leakage inflation per architecture — above zero everywhere",
                  loc="left", fontsize=9.0)
    save(fig, "fig16_model_metric_lines")


def fig_paired_metrics():
    P = json.load(open(os.path.join(RES, "panel.json")))
    case = sorted([r for r in P if r["regime"] == "case"], key=lambda r: -r["mcc"])
    img = {r["model"]: r for r in P if r["regime"] == "image"}
    names = [SHORT[r["model"]] for r in case]
    keys = [("acc", "Accuracy"), ("sens", "Sensitivity"), ("spec", "Specificity"),
            ("f1", "F1"), ("mcc", "MCC ×100"), ("err", "Error rate")]
    # x axis is shared down each column so only the bottom row carries model
    # names: that buys the room to set them at a legible size
    fig, axes = plt.subplots(2, 3, figsize=(W42, 6.5), constrained_layout=True,
                             sharex=True)
    x = np.arange(len(case)); w = 0.38
    for ax, (k, lb) in zip(axes.ravel(), keys):
        cv = [r[k] * 100 for r in case]
        iv = [img[r["model"]][k] * 100 for r in case]
        ax.bar(x - w/2, cv, w, color=C["blue"], label="Case-level", zorder=2)
        ax.bar(x + w/2, iv, w, color=C["orange"], label="Image-level", zorder=2)
        # twelve names in a 2.1in panel only clear one another upright, so the
        # labels are set vertically rather than on a slant
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=90, ha="center",
                                             va="top", fontsize=7.0,
                                             fontweight="bold")
        ax.set_title(lb, loc="left", fontsize=9.0, fontweight="bold")
        ax.grid(axis="y", zorder=0); ax.set_axisbelow(True)
        lo = min(min(cv), min(iv)); hi = max(max(cv), max(iv))
        pad = (hi - lo) * 0.35 + 0.5
        ax.set_ylim(max(0, lo - pad), hi + pad)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=2, columnspacing=2.0,
               borderaxespad=0.15, fontsize=8.5, handlelength=1.4)
    fig.suptitle("Every metric, every architecture, both split regimes", x=0.006,
                 ha="left", fontsize=9.5, fontweight="bold")
    save(fig, "fig17_paired_metric_panel")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    todo = a.only or ["qml", "bars", "lines", "paired"]
    if "qml" in todo: fig_qml()
    if "bars" in todo: fig_model_bars()
    if "lines" in todo: fig_model_lines()
    if "paired" in todo: fig_paired_metrics()


if __name__ == "__main__":
    main()
