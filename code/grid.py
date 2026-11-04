"""
Publication explanation grid: 4 cases x (original + 4 CAM methods).

Generated at stages[-2]. The final stage is the conventional choice, but
GradCAM++ silently returns an all-zero map on ~25% of images there while
GradCAM never does -- see the layer-sensitivity result. stages[-2] is stable
for all four methods, so the grid is built there and every panel shown is a
real map rather than a blank one.
"""
import os, sys, json, csv, argparse, warnings
import numpy as np
import torch, timm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM, EigenCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xai import FrozenHead, load_img

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
torch.set_num_threads(2)
INK, INK2 = "#0b0b0b", "#52514e"
# authored at the width it occupies on the page (42pc = 7.00in), so these
# point sizes are the point sizes the reader sees
W42 = 7.00
plt.rcParams.update({"savefig.dpi": 600, "font.family": "DejaVu Sans",
                     "font.size": 9, "axes.titleweight": "bold",
                     "savefig.bbox": None})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="convnextv2_tiny.fcmae_ft_in22k_in1k")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rows = list(csv.DictReader(open(os.path.join(ROOT, "manifest.csv"))))
    label = {r["file"]: int(r["label"]) for r in rows}
    d = np.load(os.path.join(ROOT, "out", "features", f"plain__{a.model}.npz"), allow_pickle=True)
    F, files = d["features"], [str(x) for x in d["files"]]
    idx = {f: i for i, f in enumerate(files)}
    y = np.array([label[f] for f in files])
    sp = json.load(open(os.path.join(ROOT, "out", "splits", f"split_case_seed{a.seed}.json")))["assign"]
    dev = [f for f in files if sp[f] in ("train", "val")]
    sc = StandardScaler().fit(F[[idx[f] for f in dev]])
    clf = LogisticRegression(C=1.0, max_iter=3000, random_state=a.seed)
    clf.fit(sc.transform(F[[idx[f] for f in dev]]), y[[idx[f] for f in dev]])

    bb = timm.create_model(a.model, pretrained=False, num_classes=0)
    sd = torch.load(os.path.join(ROOT, "weights", f"{a.model}.pth"),
                    map_location="cpu", weights_only=True)
    bb.load_state_dict({k: v for k, v in sd.items() if not k.startswith("head.fc")}, strict=False)
    bb.eval()
    model = FrozenHead(bb, sc.mean_, sc.scale_, clf.coef_[0], float(clf.intercept_[0])).eval()
    cfg = bb.default_cfg; size = cfg["input_size"][-1]
    layer = [bb.stages[-2]]

    grid = json.load(open(os.path.join(ROOT, "out", "xai", "xai_scores.json")))["grid_files"]
    methods = [("Grad-CAM", GradCAM), ("Grad-CAM++", GradCAMPlusPlus),
               ("Eigen-CAM", EigenCAM), ("Score-CAM", ScoreCAM)]

    fig, axes = plt.subplots(len(grid), 5, figsize=(W42, 1.42 * len(grid) + 0.42),
                             constrained_layout=True)
    for r, f in enumerate(grid):
        x, raw = load_img(os.path.join(ROOT, "data", "plain", f), size, cfg["mean"], cfg["std"])
        cls = label[f]
        with torch.no_grad():
            p = torch.softmax(model(x), 1)[0, 1].item()
        axes[r, 0].imshow(raw)
        axes[r, 0].set_ylabel(f"{'PTC' if cls else 'Benign'}\ncase {f.split('_')[1]}",
                              fontsize=8.0, fontweight="bold", color=INK)
        if r == 0:
            axes[r, 0].set_title("Cytological image", fontsize=8.4, fontweight="bold", color=INK)
        axes[r, 0].text(0.02, 0.97, f"p(PTC) = {p:.3f}", transform=axes[r, 0].transAxes,
                        va="top", ha="left", fontsize=7.0, fontweight="bold", color="white",
                        bbox=dict(boxstyle="round,pad=0.20", fc="#0b0b0b", ec="none", alpha=0.68))
        for c, (mname, M) in enumerate(methods, start=1):
            cam = M(model=model, target_layers=layer)
            g = cam(input_tensor=x, targets=[ClassifierOutputTarget(cls)])[0]
            g = (g - g.min()) / (g.max() - g.min() + 1e-8)
            axes[r, c].imshow(raw)
            axes[r, c].imshow(g, cmap="inferno", alpha=0.52, vmin=0, vmax=1)
            if r == 0:
                axes[r, c].set_title(mname, fontsize=8.4, fontweight="bold", color=INK)
        print(f"  row {r+1}/{len(grid)}  {f}")
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#d8d7d2"); s.set_linewidth(0.6)

    sm = plt.cm.ScalarMappable(cmap="inferno", norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=axes, location="bottom", fraction=0.022, pad=0.012, aspect=55)
    cb.set_label("Normalized class-discriminative attribution", fontsize=8.2,
                 fontweight="bold", color=INK2)
    cb.ax.tick_params(labelsize=7.4, color=INK2, labelcolor=INK2)
    cb.outline.set_visible(False)

    out = os.path.join(ROOT, "out", "figures")
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, "fig6_xai_grid.png"), dpi=600)
    fig.savefig(os.path.join(out, "fig6_xai_grid.pdf"))
    print("  wrote fig6_xai_grid.png / .pdf")


if __name__ == "__main__":
    main()
