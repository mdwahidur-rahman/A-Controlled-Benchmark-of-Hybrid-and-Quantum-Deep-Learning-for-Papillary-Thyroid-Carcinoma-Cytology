"""
Second figure set: curves, calibration, error decomposition, sensitivity
surfaces, dataset flow and preprocessing panels.

Same conventions as figures.py -- validated colorblind-safe palette, 600 dpi
raster plus vector PDF, constrained layout, and every label placed clear of
marks and whiskers.
"""
import os, sys, json, csv, argparse, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.metrics import roc_curve, precision_recall_curve
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import all_metrics

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
TOP6 = ["convnextv2_tiny.fcmae_ft_in22k_in1k", "convnext_tiny.fb_in22k_ft_in1k",
        "resnet50.a1_in1k", "vit_base_patch16_224.augreg2_in21k_ft_in1k",
        "swinv2_tiny_window8_256.ms_in1k", "swin_tiny_patch4_window7_224.ms_in1k"]
SER = [C["blue"], C["orange"], C["aqua"], C["violet"], C["magenta"], C["yellow"]]


def save(fig, name):
    _bold_legends(fig)
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, f"{name}.png"), dpi=600)
    fig.savefig(os.path.join(FIG, f"{name}.pdf"))
    plt.close(fig)
    print(f"  wrote {name}")


def pooled(model, regime, head="linear", variant="plain", seeds=(0, 1, 2, 3, 4)):
    Y, P = [], []
    for s in seeds:
        p = os.path.join(RES, f"preds_{variant}_{model}_{regime}_{head}_s{s}.npz")
        if not os.path.exists(p): continue
        d = np.load(p, allow_pickle=True)
        Y.append(d["y"].astype(int)); P.append(d["p"].astype(float))
    if not Y: return None, None
    return np.concatenate(Y), np.concatenate(P)


def fig_roc_pr():
    fig, axes = plt.subplots(2, 2, figsize=(W38, 6.1), constrained_layout=True)
    for col, regime in enumerate(("case", "image")):
        ax, ax2 = axes[0, col], axes[1, col]
        for i, m in enumerate(TOP6):
            y, p = pooled(m, regime)
            if y is None: continue
            fpr, tpr, _ = roc_curve(y, p)
            a = all_metrics(y, p)["auc"]
            ax.plot(fpr, tpr, lw=1.8, color=SER[i],
                    label=f"{SHORT[m]}  {a:.3f}")
            pr, rc, _ = precision_recall_curve(y, p)
            ap = all_metrics(y, p)["auprc"]
            ax2.plot(rc, pr, lw=1.8, color=SER[i], label=f"{SHORT[m]}  {ap:.3f}")
        ax.plot([0, 1], [0, 1], ls=":", lw=1.0, color=INK2)
        ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
        ax.set_title(f"ROC — {'case-level (correct)' if regime=='case' else 'image-level (leaky)'}",
                     loc="left", fontsize=9.0)
        ax.grid(True); ax.set_axisbelow(True); ax.set_xlim(0, 1); ax.set_ylim(0, 1.005)
        ax.legend(loc="lower right", title="AUC", title_fontsize=8.5,
                  fontsize=8.0, labelspacing=0.32, handlelength=1.5,
                  handletextpad=0.5, borderpad=0.4)
        ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
        ax2.set_title(f"Precision–recall — {'case-level' if regime=='case' else 'image-level'}",
                      loc="left")
        ax2.grid(True); ax2.set_axisbelow(True); ax2.set_xlim(0, 1); ax2.set_ylim(0.5, 1.005)
        ax2.legend(loc="lower left", title="AUPRC", title_fontsize=8.5,
                   fontsize=8.0, labelspacing=0.32, handlelength=1.5,
                   handletextpad=0.5, borderpad=0.4)
    save(fig, "fig7_roc_pr_curves")


def fig_calibration():
    cal = json.load(open(os.path.join(RES, "calibration.json")))
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(W42, 3.7), constrained_layout=True)
    for i, m in enumerate(TOP6[:4]):
        for regime, ls in (("case", "-"), ("image", "--")):
            k = f"{m}|{regime}"
            if k not in cal: continue
            bins = [b for b in cal[k]["bins"] if b[2] > 0 and np.isfinite(b[3])]
            conf = [b[3] for b in bins]; acc = [b[4] for b in bins]
            ax.plot(conf, acc, ls=ls, lw=1.6, marker="o", ms=4, color=SER[i],
                    label=f"{SHORT[m]} ({regime})" if regime == "case" else None)
    ax.plot([0, 1], [0, 1], ls=":", lw=1.0, color=INK2)
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed frequency")
    ax.set_title("Reliability: solid = case, dashed = image", loc="left", fontsize=9.0)
    ax.grid(True); ax.set_axisbelow(True); ax.legend(loc="upper left", fontsize=8.0, labelspacing=0.34,
              handlelength=1.5, handletextpad=0.5)

    models = sorted(SHORT, key=lambda m: SHORT[m])
    y = np.arange(len(models)); w = 0.38
    for off, regime, col, lb in ((-w/2, "case", C["blue"], "Case-level"),
                                 (+w/2, "image", C["orange"], "Image-level")):
        v = [cal.get(f"{m}|{regime}", {}).get("ece", np.nan) for m in models]
        ax2.barh(y + off, v, w, color=col, label=lb, zorder=2)
    ax2.set_yticks(y); ax2.set_yticklabels([SHORT[m] for m in models], fontsize=7.6,
                                           fontweight="bold")
    ax2.set_xlabel("Expected calibration error")
    ax2.set_xlim(0, max(np.nanmax([cal.get(f"{m}|{r}", {}).get("ece", np.nan)
                                   for m in models]) for r in ("case", "image")) * 1.42)
    ax2.grid(axis="x", zorder=0); ax2.set_axisbelow(True)
    # bars reach the right edge of the panel, so the legend gets an opaque
    # backing rather than sitting transparently on top of them
    lg = ax2.legend(loc="lower right", frameon=True, framealpha=0.95,
                    edgecolor="#d8d7d2", facecolor="white", borderpad=0.4,
                    fontsize=8.5, labelspacing=0.34, handlelength=1.4,
                    handletextpad=0.5)
    lg.set_zorder(6)
    ax2.set_title("Calibration error", loc="left", fontsize=9.0)
    save(fig, "fig8_calibration")


def fig_confusion():
    """Six encoders x two partitioning rules, as light, airy 2 x 2 matrices.

    Each cell carries one number, the row-normalized rate, which is what the
    off-diagonal comparison is about; the raw counts live in Table 4 and in
    Eq. (errorrates). Printing count and percentage together, in bold, on a
    near-black diagonal, buried the figure in ink -- the matrices are small and
    repeated twelve times, so the type has to stay light to stay readable.

    The rows run in blocks: within each block of three encoders the upper row
    is the case-level rule and the lower row the image-level rule.
    """
    from matplotlib.colors import LinearSegmentedColormap
    # Blues truncated below its darkest end: 1.0 lands on a medium blue rather
    # than near-black, so the diagonal reads as emphasis and not as a blot
    CM = LinearSegmentedColormap.from_list(
        "cm_light", plt.cm.Blues(np.linspace(0.0, 0.86, 256)))

    models = TOP6[:6]
    blocks = [models[:3], models[3:]]
    cls = ["Benign", "PTC"]
    fig, axes = plt.subplots(4, 3, figsize=(W42, 6.1), constrained_layout=True)
    im = None
    for bi, block in enumerate(blocks):
        for ci, m in enumerate(block):
            for ri, regime in enumerate(("case", "image")):
                ax = axes[2 * bi + ri, ci]
                y, p = pooled(m, regime)
                if y is None:
                    ax.axis("off"); continue
                mt = all_metrics(y, p)
                M = np.array([[mt["tn"], mt["fp"]], [mt["fn"], mt["tp"]]], dtype=float)
                Mn = M / M.sum(axis=1, keepdims=True)
                im = ax.imshow(Mn, cmap=CM, vmin=0, vmax=1)
                for i in range(2):
                    for j in range(2):
                        ax.text(j, i, f"{Mn[i, j]:.2f}", ha="center", va="center",
                                fontsize=8.0,
                                color="white" if Mn[i, j] > 0.55 else INK)
                ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
                ax.set_xticklabels(cls, fontsize=7.2)
                ax.set_yticklabels(cls, fontsize=7.2)
                ax.tick_params(length=0, pad=2.5)
                # hairline separators instead of cell borders
                ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
                ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
                ax.grid(which="minor", color="white", linewidth=1.1)
                ax.tick_params(which="minor", length=0)
                for sp in ax.spines.values():
                    sp.set_visible(False)
                if ri == 0:
                    ax.set_title(SHORT[m], fontsize=8.6, fontweight="bold", pad=4)
                if ci == 0:
                    ax.set_ylabel("Case-level" if regime == "case" else "Image-level",
                                  fontsize=8.2, fontweight="bold", labelpad=3)
                if ri == 1:
                    ax.set_xlabel("Predicted class", fontsize=7.6, labelpad=2)
    cb = fig.colorbar(im, ax=axes, location="right", fraction=0.018, pad=0.006,
                      aspect=42)
    cb.set_label("Row-normalized rate", fontsize=8.0, fontweight="bold", color=INK2)
    cb.ax.tick_params(labelsize=7.2, length=2, color=INK2, labelcolor=INK2)
    cb.outline.set_visible(False)
    save(fig, "fig9_confusion_matrices")


def fig_sweep():
    S = json.load(open(os.path.join(RES, "sweep.json")))
    A = [r for r in S if r["sweep"] == "A"]
    Cs = sorted({r["C"] for r in A})
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(W42, 3.9), constrained_layout=True)
    models = sorted({r["model"] for r in A}, key=lambda m: SHORT.get(m, m))
    for m in models:
        fam = FAMILY[SHORT[m]]
        d = [(np.mean([r["acc"] for r in A if r["model"] == m and r["C"] == c and r["regime"] == "image"])
              - np.mean([r["acc"] for r in A if r["model"] == m and r["C"] == c and r["regime"] == "case"])) * 100
             for c in Cs]
        ax.plot(Cs, d, marker="o", ms=4.5, lw=1.5, color=FAM_C[fam], alpha=0.8)
    ax.axhline(0, color=INK2, lw=1.0)
    ax.set_xscale("log"); ax.set_xlabel("Regularization strength C")
    ax.set_ylabel("Leakage inflation (pp)")
    ax.set_title("Inflation across six values of $C$", loc="left", fontsize=9.0)
    ax.grid(True); ax.set_axisbelow(True)
    # twelve curves fill the panel, so the legend is given an opaque backing
    # and headroom is added above the highest curve to keep the two apart
    ax.set_ylim(top=ax.get_ylim()[1] + 0.95)
    lg = ax.legend(handles=[Line2D([], [], color=FAM_C[f], marker="o", label=f)
                            for f in ("CNN", "Transformer", "Hybrid")],
                   loc="upper right", ncol=3, frameon=True, framealpha=0.95,
                   facecolor="white", edgecolor="#d8d7d2", borderpad=0.4,
                   fontsize=8.5, columnspacing=1.0, handletextpad=0.4,
                   handlelength=1.4)
    lg.set_zorder(6)

    names = {"A": "Regularization", "B": "Head architecture",
             "C": "Preprocessing", "D": "Feature scaling"}
    key = {"A": "C", "B": "head", "C": "variant", "D": "scale"}
    labs, vals, cols = [], [], []
    for sw in ("A", "B", "C", "D"):
        R = [r for r in S if r["sweep"] == sw]
        if not R: continue
        for setting in sorted({r[key[sw]] for r in R}, key=str):
            c = np.mean([r["acc"] for r in R if r[key[sw]] == setting and r["regime"] == "case"])
            i = np.mean([r["acc"] for r in R if r[key[sw]] == setting and r["regime"] == "image"])
            PRETTY = {"l2": "$\\ell_2$", "none": "no scaling",
                      "standard": "standardize", "plain": "plain", "log": "LoG",
                      "linear": "linear", "mlp-128": "MLP-128",
                      "mlp-256": "MLP-256"}
            pretty = PRETTY.get(str(setting))
            if pretty is None:
                pretty = f"$10^{{{int(round(np.log10(float(setting))))}}}$"
            labs.append(f"{names[sw][:4]}: {pretty}"); vals.append((i - c) * 100)
            cols.append([C["blue"], C["orange"], C["aqua"], C["violet"]]["ABCD".index(sw)])
    yy = np.arange(len(labs))
    ax2.barh(yy, vals, color=cols, height=0.65, zorder=2)
    ax2.axvline(0, color=INK2, lw=1.0)
    ax2.set_yticks(yy); ax2.set_yticklabels(labs, fontsize=7.4, fontweight="bold")
    ax2.set_xlabel("Leakage inflation (pp)")
    ax2.set_title(f"Positive at {sum(v>0 for v in vals)} of {len(vals)} settings",
                  loc="left", fontsize=9.0)
    ax2.grid(axis="x", zorder=0); ax2.set_axisbelow(True)
    ax2.set_xlim(min(0, min(vals)) - 0.3, max(vals) + 0.4)
    save(fig, "fig10_hyperparameter_sensitivity")


def fig_mcc_errors():
    P = json.load(open(os.path.join(RES, "panel.json")))
    # the primary interval is the group-cluster bootstrap, not the field-level
    # one: resampling fields independently treats several fields of one group
    # as independent observations and is narrower by a factor of 1.76 here
    CL = json.load(open(os.path.join(RES, "cluster_intervals.json")))
    for r in P:
        c = CL.get(f'{r["model"]}|{r["regime"]}')
        if c:
            r["mcc_ci_lo"], r["mcc_ci_hi"] = c["mcc_cluster_ci"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(W42, 4.0), constrained_layout=True)
    rows = [r for r in P if r["regime"] == "case"]
    rows.sort(key=lambda r: r["mcc"])
    y = np.arange(len(rows))
    for k, r in enumerate(rows):
        n = SHORT[r["model"]]; col = FAM_C[FAMILY[n]]
        ax.plot([r["mcc_ci_lo"], r["mcc_ci_hi"]], [k, k], color=col, lw=2.0,
                solid_capstyle="round", alpha=0.65)
        ax.plot(r["mcc"], k, "o", ms=6.5, color=col)
        ax.text(r["mcc_ci_hi"] + 0.005, k, f"{r['mcc']:.3f}", va="center",
                fontsize=7.6, fontweight="bold", color=INK2)
    ax.set_yticks(y); ax.set_yticklabels([SHORT[r["model"]] for r in rows], fontsize=7.6,
                                         fontweight="bold")
    ax.set_xlabel("Matthews correlation (case-level)", fontsize=9.0)
    ax.set_xlim(min(r["mcc_ci_lo"] for r in rows) - 0.01,
                max(r["mcc_ci_hi"] for r in rows) + 0.055)
    ax.grid(axis="x"); ax.set_axisbelow(True)
    ax.set_title("MCC with 95% group-cluster intervals", loc="left", fontsize=9.0)

    for r in P:
        n = SHORT[r["model"]]
        mk = "o" if r["regime"] == "case" else "^"
        fc = FAM_C[FAMILY[n]] if r["regime"] == "image" else "white"
        ax2.plot(r["fpr"] * 100, r["fnr"] * 100, mk, ms=8, color=FAM_C[FAMILY[n]],
                 mfc=fc, mew=1.6)
    lim = [min(min(r["fpr"] for r in P), min(r["fnr"] for r in P)) * 100 - 0.6,
           max(max(r["fpr"] for r in P), max(r["fnr"] for r in P)) * 100 + 0.6]
    ax2.plot(lim, lim, ls=":", lw=1.0, color=INK2)
    ax2.set_xlim(lim); ax2.set_ylim(lim)
    ax2.set_xlabel("False positive rate %\n(benign called PTC)", fontsize=9.0)
    ax2.set_ylabel("False negative rate %\n(PTC missed)", fontsize=9.0)
    ax2.grid(True); ax2.set_axisbelow(True)
    ax2.set_title("Error decomposition", loc="left", fontsize=9.0)
    ax2.legend(handles=[Line2D([], [], marker="o", ls="", mfc="white", mec=INK2,
                               mew=1.6, color=INK2, label="Case-level"),
                        Line2D([], [], marker="^", ls="", color=INK2, label="Image-level")] +
                       [Line2D([], [], marker="s", ls="", color=FAM_C[f], label=f)
                        for f in ("CNN", "Transformer", "Hybrid")],
               loc="upper left", fontsize=8.0, labelspacing=0.34,
               handletextpad=0.5, borderpad=0.4)
    save(fig, "fig11_mcc_and_error_rates")


def fig_flow():
    fig, ax = plt.subplots(figsize=(W42, 4.6), constrained_layout=True)
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    # left column sits at x = 3.0 so the widest box (w = 5.2) clears the
    # left edge of the axes instead of being clipped by it
    LX = 3.0
    boxes = [
        (5.0, 9.2, 6.9, 1.00, "Pang et al. thyroid cohort  (Mendeley 10.17632/dp7zp2pb2v.1)\n"
                              "1,719 files · 2.47 GB · CC BY 4.0", C["blue"]),
        (LX, 7.5, 4.2, 0.92, "Cytology  1,332 images\n384 cases", C["aqua"]),
        (7.9, 7.5, 4.0, 0.92, "Ultrasound  387 images\n387 cases  (not used)", GRID),
        (LX, 5.9, 4.6, 0.92, "PTC  697 images / 219 cases\nBenign  635 images / 165 cases", C["aqua"]),
        (LX, 4.3, 4.8, 0.92, "Preprocessing  →  384×384\nplain  +  Laplacian-of-Gaussian", C["violet"]),
        (LX, 2.6, 5.4, 1.10, "Two split regimes × 5 seeds\n"
                             "case-level: 0 cases span subsets\n"
                             "image-level: ~288 / 384 cases span subsets", C["orange"]),
        (LX, 0.85, 5.4, 0.92, "12 frozen backbones → feature cache\n"
                              "benchmark · hybrid · quantum · federated · XAI", C["green"]),
    ]
    for x, yy, w, h, txt, col in boxes:
        ax.add_patch(FancyBboxPatch((x - w / 2, yy - h / 2), w, h,
                                    boxstyle="round,pad=0.08,rounding_size=0.12",
                                    fc="white", ec=col, lw=1.9))
        ax.text(x, yy, txt, ha="center", va="center", fontsize=7.4,
                fontweight="bold", color=INK, linespacing=1.45)
    arrows = [((5.0, 8.70), (LX, 7.96)), ((5.0, 8.70), (7.9, 7.96)),
              ((LX, 7.04), (LX, 6.36)), ((LX, 5.44), (LX, 4.76)),
              ((LX, 3.84), (LX, 3.15)), ((LX, 2.05), (LX, 1.31))]
    for p0, p1 in arrows:
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=13,
                                     color=INK2, lw=1.2, shrinkA=0, shrinkB=0))
    ax.text(9.9, 6.78, "excluded", fontsize=7.2, fontweight="bold", color=INK2,
            style="italic", ha="right")
    save(fig, "fig12_dataset_flow")


def fig_preprocessing():
    man = list(csv.DictReader(open(os.path.join(ROOT, "manifest.csv"))))

    def distinct_cases(cls, k):
        seen, out = set(), []
        for r in man:
            if r["cls"] != cls or r["case"] in seen:
                continue
            seen.add(r["case"]); out.append(r["file"])
            if len(out) == k:
                break
        return out

    picks = distinct_cases("PTC", 2) + distinct_cases("BEN", 2)
    from PIL import Image
    fig, axes = plt.subplots(len(picks), 4, figsize=(W42, 1.78 * len(picks)),
                             constrained_layout=True)
    for r, f in enumerate(picks):
        p = np.asarray(Image.open(os.path.join(ROOT, "data", "plain", f)).convert("RGB"))
        l = np.asarray(Image.open(os.path.join(ROOT, "data", "log", f)).convert("RGB"))
        diff = np.abs(l.astype(float) - p.astype(float)).mean(2)
        axes[r, 0].imshow(p)
        axes[r, 1].imshow(l)
        axes[r, 2].imshow(diff, cmap="magma")
        # the LoG stage turns out to be close to a no-op at sigma=1.0; say so numerically
        axes[r, 2].text(0.03, 0.97, f"mean |Δ| = {diff.mean():.2f}/255",
                        transform=axes[r, 2].transAxes, va="top", ha="left",
                        fontsize=7.4, fontweight="bold", color="white",
                        bbox=dict(boxstyle="round,pad=0.2", fc="#0b0b0b", ec="none", alpha=0.6))
        axes[r, 3].hist(p.mean(2).ravel(), bins=64, color=C["blue"], alpha=0.65,
                        label="plain", density=True)
        axes[r, 3].hist(l.mean(2).ravel(), bins=64, color=C["orange"], alpha=0.6,
                        label="LoG", density=True)
        axes[r, 3].set_yticks([]); axes[r, 3].tick_params(labelsize=7.6)
        axes[r, 3].spines["left"].set_visible(False)
        # clamp to the data so the last tick label cannot run off the canvas
        axes[r, 3].set_xlim(0, 255)
        axes[r, 3].set_xticks([0, 64, 128, 192, 255])
        if r == 0:
            for c, t in enumerate(["Resized 384×384", "Laplacian-of-Gaussian",
                                   "|difference|", "Intensity histogram"]):
                axes[r, c].set_title(t, fontsize=9.0, fontweight="bold")
            axes[r, 3].set_ylim(top=axes[r, 3].get_ylim()[1] * 1.30)
            lg = axes[r, 3].legend(fontsize=8.5, loc="upper left", frameon=False,
                                   labelspacing=0.3, handlelength=1.2,
                                   handletextpad=0.45)
            for t_ in lg.get_texts():
                t_.set_fontweight("bold")
        axes[r, 0].set_ylabel(("PTC" if f.startswith("PTC") else "Benign") +
                              f"\ncase {f.split('_')[1]}", fontsize=8.4,
                              fontweight="bold")
        for c in range(3):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            for s in axes[r, c].spines.values():
                s.set_color(GRID); s.set_linewidth(0.6)
    save(fig, "fig13_preprocessing_panel")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    todo = a.only or ["roc", "cal", "cm", "sweep", "mcc", "flow", "prep"]
    if "roc" in todo: fig_roc_pr()
    if "cal" in todo: fig_calibration()
    if "cm" in todo: fig_confusion()
    if "sweep" in todo: fig_sweep()
    if "mcc" in todo: fig_mcc_errors()
    if "flow" in todo: fig_flow()
    if "prep" in todo: fig_preprocessing()


if __name__ == "__main__":
    main()
