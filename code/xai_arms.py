"""
Attribution grids for the federated and quantum arms.

The centralized grid in grid.py explains a linear head fitted on pooled data.
This module produces the matching grids for the other two arms, so that each
results subsection carries its own explanation evidence rather than reusing the
centralized panel:

  federated  backbone -> standardize -> FedAvg global linear head
  quantum    backbone -> standardize -> PCA(32) -> VQC -> linear head

Both models are rebuilt end to end in torch so that gradients reach the
convolutional feature maps; the PCA projection is expressed as a fixed affine
layer and the variational circuit as a differentiable state-vector simulation,
so neither breaks the attribution path.

Maps are produced at the penultimate encoder stage, where all four CAM methods
are non-degenerate, and are scored by deletion and insertion rather than
presented qualitatively.
"""
import os, sys, json, csv, argparse, warnings
import numpy as np
import torch
import torch.nn as nn
import timm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM, EigenCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xai import load_img, del_ins
from qml_arm import VQC, DIM, N_QUBITS
from federated import dirichlet_cases, make_model, set_flat, get_flat, local_train

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
torch.set_num_threads(2)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"
plt.rcParams.update({"savefig.dpi": 600, "font.family": "DejaVu Sans",
                     "font.size": 9, "axes.titleweight": "bold",
                     "savefig.bbox": None})


class ArmHead(nn.Module):
    """backbone -> standardize -> [optional PCA -> VQC] -> 2-logit head."""

    def __init__(self, backbone, mean, scale, w, b, pca=None, vqc=None):
        super().__init__()
        self.backbone = backbone
        self.register_buffer("m", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("s", torch.tensor(scale, dtype=torch.float32))
        self.pca, self.vqc = pca, vqc
        self.fc = nn.Linear(len(w), 2)
        with torch.no_grad():
            self.fc.weight.copy_(torch.tensor(np.vstack([-np.asarray(w),
                                                         np.asarray(w)]),
                                              dtype=torch.float32))
            self.fc.bias.copy_(torch.tensor([-b, b], dtype=torch.float32))

    def forward(self, x):
        z = (self.backbone(x) - self.m) / self.s
        if self.pca is not None:
            z = self.pca(z)
        if self.vqc is not None:
            z = self.vqc(z)
        return self.fc(z)


def make_pca_layer(pca):
    """PCA as a fixed affine map, so gradients pass through unchanged."""
    lin = nn.Linear(pca.components_.shape[1], pca.components_.shape[0], bias=True)
    with torch.no_grad():
        lin.weight.copy_(torch.tensor(pca.components_, dtype=torch.float32))
        lin.bias.copy_(torch.tensor(-pca.components_ @ pca.mean_,
                                    dtype=torch.float32))
    for p in lin.parameters():
        p.requires_grad_(False)
    return lin


def build_backbone(model_name):
    bb = timm.create_model(model_name, pretrained=False, num_classes=0)
    sd = torch.load(os.path.join(ROOT, "weights", f"{model_name}.pth"),
                    map_location="cpu", weights_only=True)
    bb.load_state_dict({k: v for k, v in sd.items()
                        if not k.startswith("head.fc")}, strict=False)
    bb.eval()
    return bb


def fit_federated_head(F, idx, y, files, assign, case_of, seed, alpha, K,
                       rounds, epochs, lr):
    """Re-run FedAvg and return the global linear head as (w, b)."""
    tr = [f for f in files if assign[f] in ("train", "val")]
    sc = StandardScaler().fit(F[[idx[f] for f in tr]])
    cases = sorted({case_of[f] for f in tr})
    clab = {c: y[idx[next(f for f in tr if case_of[f] == c)]] for c in cases}
    rng = np.random.default_rng(seed)
    cmap = dirichlet_cases(cases, clab, K, alpha, rng)
    clients = {}
    for k in range(K):
        fk = [f for f in tr if cmap[case_of[f]] == k]
        if len(fk) < 8 or len(set(y[[idx[f] for f in fk]])) < 2:
            continue
        clients[k] = (torch.tensor(sc.transform(F[[idx[f] for f in fk]]),
                                   dtype=torch.float32),
                      torch.tensor(y[[idx[f] for f in fk]], dtype=torch.float32))
    dim = F.shape[1]
    g = make_model(dim, seed)
    gw = get_flat(g).clone()
    ntot = sum(len(v[1]) for v in clients.values())
    for _ in range(rounds):
        acc = torch.zeros_like(gw)
        for k, (Xc, yc) in clients.items():
            mdl = make_model(dim, seed)
            set_flat(mdl, gw.clone())
            local_train(mdl, Xc, yc, epochs, lr)
            acc += get_flat(mdl) * (len(yc) / ntot)
        gw = acc
    w = gw[:dim].numpy()
    b = float(gw[dim].item())
    return sc, w, b, len(clients)


def cams_for(model, layer, files, variant, size, cfg, label, methods, out_npz):
    store, scores = {}, []
    for mname, M in methods:
        cam = M(model=model, target_layers=layer)
        for f in files:
            x, raw = load_img(os.path.join(ROOT, "data", variant, f),
                              size, cfg["mean"], cfg["std"])
            cls = label[f]
            g = cam(input_tensor=x, targets=[ClassifierOutputTarget(cls)])[0]
            g = (g - g.min()) / (g.max() - g.min() + 1e-8)
            store[(mname, f)] = g
            dA, _ = del_ins(model, x, g, cls, mode="del")
            iA, _ = del_ins(model, x, g, cls, mode="ins")
            scores.append({"method": mname, "file": f, "label": int(cls),
                           "deletion_auc": dA, "insertion_auc": iA})
        print(f"    {mname:12} done", flush=True)
    np.savez_compressed(out_npz, **{f"{m}|{f}": c for (m, f), c in store.items()})
    return store, scores


def draw_grid(store, files, model, size, cfg, variant, label, methods,
              title, out_png):
    fig, axes = plt.subplots(len(files), 5, figsize=(7.00, 1.42 * len(files) + 0.42),
                             constrained_layout=True)
    for r, f in enumerate(files):
        x, raw = load_img(os.path.join(ROOT, "data", variant, f),
                          size, cfg["mean"], cfg["std"])
        with torch.no_grad():
            p = torch.softmax(model(x), 1)[0, 1].item()
        axes[r, 0].imshow(raw)
        axes[r, 0].set_ylabel(("PTC" if label[f] else "Benign") +
                              f"\ncase {f.split('_')[1]}", fontsize=8.0,
                              fontweight="bold", color=INK)
        if r == 0:
            axes[r, 0].set_title("Cytological image", fontsize=8.4, fontweight="bold", color=INK)
        axes[r, 0].text(0.02, 0.97, f"p(PTC) = {p:.3f}",
                        transform=axes[r, 0].transAxes, va="top", ha="left",
                        fontsize=7.0, fontweight="bold", color="white",
                        bbox=dict(boxstyle="round,pad=0.20", fc="#0b0b0b",
                                  ec="none", alpha=0.68))
        for c, (mname, _) in enumerate(methods, start=1):
            axes[r, c].imshow(raw)
            axes[r, c].imshow(store[(mname, f)], cmap="inferno", alpha=0.52,
                              vmin=0, vmax=1)
            if r == 0:
                axes[r, c].set_title(mname, fontsize=8.4, fontweight="bold", color=INK)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(GRID); s.set_linewidth(0.6)
    sm = plt.cm.ScalarMappable(cmap="inferno", norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=axes, location="bottom", fraction=0.022,
                      pad=0.012, aspect=55)
    cb.set_label(f"Normalized class-discriminative attribution — {title}",
                 fontsize=8.2, fontweight="bold", color=INK2)
    cb.ax.tick_params(labelsize=7.4, color=INK2, labelcolor=INK2)
    cb.outline.set_visible(False)
    fig.savefig(out_png, dpi=600)
    fig.savefig(out_png.replace(".png", ".pdf"))
    plt.close(fig)
    print(f"  wrote {os.path.basename(out_png)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="convnextv2_tiny.fcmae_ft_in22k_in1k")
    ap.add_argument("--variant", default="plain")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--clients", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--arms", nargs="+", default=["federated", "quantum"])
    a = ap.parse_args()
    out = os.path.join(ROOT, "out", "xai")
    figdir = os.path.join(ROOT, "out", "figures")
    os.makedirs(out, exist_ok=True); os.makedirs(figdir, exist_ok=True)

    rows = list(csv.DictReader(open(os.path.join(ROOT, "manifest.csv"))))
    label = {r["file"]: int(r["label"]) for r in rows}
    case_of = {r["file"]: r["case"] for r in rows}
    d = np.load(os.path.join(ROOT, "out", "features",
                             f"{a.variant}__{a.model}.npz"), allow_pickle=True)
    F, files = d["features"], [str(x) for x in d["files"]]
    idx = {f: i for i, f in enumerate(files)}
    y = np.array([label[f] for f in files])
    assign = json.load(open(os.path.join(ROOT, "out", "splits",
                                         f"split_case_seed{a.seed}.json")))["assign"]
    grid = json.load(open(os.path.join(out, "xai_scores.json")))["grid_files"]

    bb = build_backbone(a.model)
    cfg = bb.default_cfg
    size = cfg["input_size"][-1]
    layer = [bb.stages[-2]]
    methods = [("Grad-CAM", GradCAM), ("Grad-CAM++", GradCAMPlusPlus),
               ("Eigen-CAM", EigenCAM), ("Score-CAM", ScoreCAM)]
    summary = {}

    if "federated" in a.arms:
        print("federated arm:")
        sc, w, b, nk = fit_federated_head(F, idx, y, files, assign, case_of,
                                          a.seed, a.alpha, a.clients,
                                          a.rounds, a.epochs, a.lr)
        print(f"    FedAvg global head from {nk} effective clients, "
              f"alpha={a.alpha}")
        model = ArmHead(bb, sc.mean_, sc.scale_, w, b).eval()
        store, scores = cams_for(model, layer, grid, a.variant, size, cfg,
                                 label, methods,
                                 os.path.join(out, "cams_federated.npz"))
        draw_grid(store, grid, model, size, cfg, a.variant, label, methods,
                  f"FedAvg global model, $\\alpha={a.alpha}$",
                  os.path.join(figdir, "fig18_xai_grid_federated.png"))
        summary["federated"] = {"scores": scores, "n_clients": nk,
                                "alpha": a.alpha}

    if "quantum" in a.arms:
        print("quantum arm:")
        tr = [f for f in files if assign[f] in ("train", "val")]
        sc = StandardScaler().fit(F[[idx[f] for f in tr]])
        pca = PCA(n_components=DIM, random_state=a.seed).fit(
            sc.transform(F[[idx[f] for f in tr]]))
        Xtr = torch.tensor(pca.transform(sc.transform(F[[idx[f] for f in tr]])),
                           dtype=torch.float32)
        ytr = torch.tensor(y[[idx[f] for f in tr]], dtype=torch.float32)
        torch.manual_seed(a.seed)
        vqc = VQC("YYYYY", 2, a.seed)
        head = nn.Linear(N_QUBITS, 1)
        net = nn.Sequential(vqc, head)
        opt = torch.optim.Adam(net.parameters(), lr=0.05)
        lossf = nn.BCEWithLogitsLoss()
        for _ in range(260):
            opt.zero_grad()
            lossf(net(Xtr).squeeze(-1), ytr).backward()
            opt.step()
        net.eval()
        w = head.weight.detach().numpy()[0]
        b = float(head.bias.detach().numpy()[0])
        model = ArmHead(bb, sc.mean_, sc.scale_, w, b,
                        pca=make_pca_layer(pca), vqc=vqc).eval()
        store, scores = cams_for(model, layer, grid, a.variant, size, cfg,
                                 label, methods,
                                 os.path.join(out, "cams_quantum.npz"))
        draw_grid(store, grid, model, size, cfg, a.variant, label, methods,
                  "variational quantum block, $n=5$ qubits",
                  os.path.join(figdir, "fig19_xai_grid_quantum.png"))
        summary["quantum"] = {"scores": scores}

    json.dump(summary, open(os.path.join(out, "xai_arms.json"), "w"), indent=2)
    print(f"\n{'arm':10} {'method':12} {'deletion':>10} {'insertion':>11}")
    print("-" * 46)
    for arm, blk in summary.items():
        for m, _ in methods:
            dd = [s["deletion_auc"] for s in blk["scores"] if s["method"] == m]
            ii = [s["insertion_auc"] for s in blk["scores"] if s["method"] == m]
            if dd:
                print(f"{arm:10} {m:12} {np.mean(dd):10.4f} {np.mean(ii):11.4f}")


if __name__ == "__main__":
    main()
