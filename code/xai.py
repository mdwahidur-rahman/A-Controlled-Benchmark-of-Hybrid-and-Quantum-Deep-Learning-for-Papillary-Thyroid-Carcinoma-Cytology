"""
Explanation analysis: four CAM methods, scored rather than displayed.

The source paper's explanation evidence is two images and a table of narrated
numbers. Qualitative panels cannot distinguish a faithful explanation from a
pretty one, so every map here is scored by deletion/insertion -- progressively
removing (or revealing) the pixels a method ranks highest and measuring what
the model's confidence does. A faithful map drops confidence fast under
deletion and recovers it fast under insertion.

Pairwise Spearman between methods is also reported. If four methods that claim
to explain the same decision disagree with each other, no single qualitative
panel from any one of them is evidence of anything -- which is the point.

The classifier is the frozen backbone plus the linear head fitted under the
CASE-LEVEL split, so the explanations describe the honest model, not the
leaky one.
"""
import os, json, csv, argparse, warnings
import numpy as np
import torch, torch.nn as nn, timm
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM, EigenCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
torch.set_num_threads(2)


class FrozenHead(nn.Module):
    """backbone -> standardize -> linear. Mirrors the sklearn head exactly."""
    def __init__(self, backbone, mean, scale, w, b):
        super().__init__()
        self.backbone = backbone
        self.register_buffer("m", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("s", torch.tensor(scale, dtype=torch.float32))
        self.fc = nn.Linear(len(w), 2)
        with torch.no_grad():
            self.fc.weight.copy_(torch.tensor(np.vstack([-w, w]), dtype=torch.float32))
            self.fc.bias.copy_(torch.tensor([-b, b], dtype=torch.float32))

    def forward(self, x):
        return self.fc((self.backbone(x) - self.m) / self.s)


def load_img(path, size, mean, std):
    im = Image.open(path).convert("RGB")
    if im.size != (size, size):
        im = im.resize((size, size), Image.BICUBIC)
    a = np.asarray(im).astype(np.float32) / 255.0
    x = (a.transpose(2, 0, 1) - np.array(mean).reshape(3, 1, 1)) / np.array(std).reshape(3, 1, 1)
    return torch.tensor(x[None], dtype=torch.float32), a


def del_ins(model, x, cam, cls, steps=25, mode="del"):
    """Progressively remove/reveal top-ranked pixels; return normalized AUC."""
    flat = cam.flatten()
    order = np.argsort(-flat)
    n = len(order)
    base = torch.zeros_like(x) if mode == "del" else x.clone()
    cur = x.clone() if mode == "del" else torch.zeros_like(x)
    ys = []
    with torch.no_grad():
        for i in range(steps + 1):
            k = int(n * i / steps)
            m = np.zeros(n, dtype=bool); m[order[:k]] = True
            mt = torch.tensor(m.reshape(cam.shape), dtype=torch.bool)
            z = cur.clone()
            z[0, :, mt] = base[0, :, mt]
            ys.append(torch.softmax(model(z), 1)[0, cls].item())
    return float(np.trapezoid(ys, dx=1.0 / steps)), ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="convnextv2_tiny.fcmae_ft_in22k_in1k")
    ap.add_argument("--variant", default="plain")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-score", type=int, default=24, help="images for del/ins scoring")
    ap.add_argument("--n-grid", type=int, default=4, help="images in the figure grid")
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "xai"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rows = list(csv.DictReader(open(os.path.join(ROOT, "manifest.csv"))))
    label = {r["file"]: int(r["label"]) for r in rows}
    d = np.load(os.path.join(ROOT, "out", "features", f"{a.variant}__{a.model}.npz"),
                allow_pickle=True)
    F, files = d["features"], [str(x) for x in d["files"]]
    idx = {f: i for i, f in enumerate(files)}
    y_all = np.array([label[f] for f in files])
    sp = json.load(open(os.path.join(ROOT, "out", "splits", f"split_case_seed{a.seed}.json")))
    assign = sp["assign"]

    dev = [f for f in files if assign[f] in ("train", "val")]
    te = [f for f in files if assign[f] == "test"]
    sc = StandardScaler().fit(F[[idx[f] for f in dev]])
    clf = LogisticRegression(C=1.0, max_iter=3000, random_state=a.seed)
    clf.fit(sc.transform(F[[idx[f] for f in dev]]), y_all[[idx[f] for f in dev]])

    bb = timm.create_model(a.model, pretrained=False, num_classes=0)
    sd = torch.load(os.path.join(ROOT, "weights", f"{a.model}.pth"),
                    map_location="cpu", weights_only=True)
    bb.load_state_dict({k: v for k, v in sd.items() if not k.startswith("head.fc")}, strict=False)
    bb.eval()
    model = FrozenHead(bb, sc.mean_, sc.scale_, clf.coef_[0], float(clf.intercept_[0])).eval()
    cfg = bb.default_cfg; size = cfg["input_size"][-1]

    target_layer = [bb.stages[-1]]
    methods = {"GradCAM": GradCAM, "GradCAM++": GradCAMPlusPlus,
               "EigenCAM": EigenCAM, "ScoreCAM": ScoreCAM}

    rng = np.random.default_rng(a.seed)
    probs = clf.predict_proba(sc.transform(F[[idx[f] for f in te]]))[:, 1]
    correct = [f for f, p in zip(te, probs) if (p >= 0.5) == (label[f] == 1)]
    pick_pos = [f for f in correct if label[f] == 1]
    pick_neg = [f for f in correct if label[f] == 0]
    rng.shuffle(pick_pos); rng.shuffle(pick_neg)
    score_set = (pick_pos[:a.n_score // 2] + pick_neg[:a.n_score // 2])
    grid_set = (pick_pos[:a.n_grid // 2] + pick_neg[:a.n_grid // 2])

    scores, cams_store = [], {}
    for mi, (mname, M) in enumerate(methods.items()):
        subset = score_set if mname != "ScoreCAM" else score_set[:8]
        cam_alg = M(model=model, target_layers=target_layer)
        for f in subset:
            x, raw = load_img(os.path.join(ROOT, "data", a.variant, f), size, cfg["mean"], cfg["std"])
            cls = label[f]
            g = cam_alg(input_tensor=x, targets=[ClassifierOutputTarget(cls)])[0]
            g = (g - g.min()) / (g.max() - g.min() + 1e-8)
            dA, _ = del_ins(model, x, g, cls, mode="del")
            iA, _ = del_ins(model, x, g, cls, mode="ins")
            scores.append({"method": mname, "file": f, "label": cls,
                           "deletion_auc": dA, "insertion_auc": iA})
            if f in grid_set:
                cams_store[(mname, f)] = g
                if mi == 0:
                    np.save(os.path.join(a.out, f"raw_{f}.npy"), raw)
        print(f"  {mname:12} scored {len(subset)} images")

    # pairwise agreement between methods on the grid images
    agree = []
    names = list(methods)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            rs = []
            for f in grid_set:
                A, B = cams_store.get((names[i], f)), cams_store.get((names[j], f))
                if A is None or B is None: continue
                rs.append(spearmanr(A.flatten(), B.flatten()).statistic)
            if rs:
                agree.append({"a": names[i], "b": names[j],
                              "spearman_mean": float(np.mean(rs)),
                              "spearman_sd": float(np.std(rs))})

    np.savez_compressed(os.path.join(a.out, "cams.npz"),
                        **{f"{m}|{f}": c for (m, f), c in cams_store.items()})
    json.dump({"scores": scores, "agreement": agree, "grid_files": grid_set},
              open(os.path.join(a.out, "xai_scores.json"), "w"), indent=2)

    print(f"\n{'method':12} {'deletion (lower=better)':>24} {'insertion (higher=better)':>26}")
    print("-" * 64)
    for m in names:
        dd = [s["deletion_auc"] for s in scores if s["method"] == m]
        ii = [s["insertion_auc"] for s in scores if s["method"] == m]
        print(f"{m:12} {np.mean(dd):14.4f} +/-{np.std(dd):5.4f} {np.mean(ii):16.4f} +/-{np.std(ii):5.4f}")
    print("\npairwise Spearman between methods:")
    for g in agree:
        print(f"  {g['a']:11} vs {g['b']:11} r = {g['spearman_mean']:+.3f} +/- {g['spearman_sd']:.3f}")


if __name__ == "__main__":
    main()
