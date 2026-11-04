"""Pairwise attribution agreement on the full 24-image set.

The published agreement statistics were computed on the four images of the
figure grid, and three of the six method pairs came back undefined because
Grad-CAM++ returned a constant map on those images -- Spearman is undefined
against a constant vector. Four images cannot support a statement about how
much two methods agree, and reporting three of six pairs as blank hides the
degeneracy rather than measuring it.

This recomputes every map on the same 24 patient-disjoint images used for the
deletion and insertion scores, stores all six pairwise Spearman correlations
with the number of images on which each is undefined, and reports the
constant-map count per method. Same model, same split, same seed, same target
layer as xai.py.
"""
import os, sys, json, csv, warnings, time, itertools
import numpy as np
import torch, timm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM, EigenCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xai import FrozenHead, load_img

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
torch.set_num_threads(2)
MODEL = "convnextv2_tiny.fcmae_ft_in22k_in1k"
VARIANT, SEED = "plain", 0

rows = list(csv.DictReader(open(os.path.join(ROOT, "manifest.csv"))))
label = {r["file"]: int(r["label"]) for r in rows}
d = np.load(os.path.join(ROOT, "out", "features", f"{VARIANT}__{MODEL}.npz"), allow_pickle=True)
F, files = d["features"], [str(x) for x in d["files"]]
idx = {f: i for i, f in enumerate(files)}
y_all = np.array([label[f] for f in files])
assign = json.load(open(os.path.join(ROOT, "out", "splits", f"split_case_seed{SEED}.json")))["assign"]
dev = [f for f in files if assign[f] in ("train", "val")]
te = [f for f in files if assign[f] == "test"]
sc = StandardScaler().fit(F[[idx[f] for f in dev]])
clf = LogisticRegression(C=1.0, max_iter=3000, random_state=SEED)
clf.fit(sc.transform(F[[idx[f] for f in dev]]), y_all[[idx[f] for f in dev]])

bb = timm.create_model(MODEL, pretrained=False, num_classes=0)
sd = torch.load(os.path.join(ROOT, "weights", f"{MODEL}.pth"), map_location="cpu", weights_only=True)
bb.load_state_dict({k: v for k, v in sd.items() if not k.startswith("head.fc")}, strict=False)
bb.eval()
model = FrozenHead(bb, sc.mean_, sc.scale_, clf.coef_[0], float(clf.intercept_[0])).eval()
cfg = bb.default_cfg; size = cfg["input_size"][-1]

rng = np.random.default_rng(SEED)
probs = clf.predict_proba(sc.transform(F[[idx[f] for f in te]]))[:, 1]
correct = [f for f, p in zip(te, probs) if (p >= 0.5) == (label[f] == 1)]
pos = [f for f in correct if label[f] == 1]; neg = [f for f in correct if label[f] == 0]
rng.shuffle(pos); rng.shuffle(neg)
score_set = pos[:12] + neg[:12]

methods = {"GradCAM": GradCAM, "GradCAM++": GradCAMPlusPlus,
           "EigenCAM": EigenCAM, "ScoreCAM": ScoreCAM}
maps, const = {}, {m: 0 for m in methods}
t0 = time.time()
for mname, M in methods.items():
    alg = M(model=model, target_layers=[bb.stages[-1]])
    for f in score_set:
        x, _ = load_img(os.path.join(ROOT, "data", VARIANT, f), size, cfg["mean"], cfg["std"])
        g = alg(input_tensor=x, targets=[ClassifierOutputTarget(label[f])])[0]
        if np.ptp(g) < 1e-8:
            const[mname] += 1
        g = (g - g.min()) / (g.max() - g.min() + 1e-8)
        maps[(mname, f)] = g.astype(np.float32)
    print(f"  {mname:11} {len(score_set)} maps, {const[mname]} constant  {time.time()-t0:6.0f}s", flush=True)

agree = []
for a, b in itertools.combinations(methods, 2):
    rs, undef = [], 0
    for f in score_set:
        A, B = maps[(a, f)].ravel(), maps[(b, f)].ravel()
        if np.ptp(A) < 1e-8 or np.ptp(B) < 1e-8:
            undef += 1; continue
        r = spearmanr(A, B).statistic
        if np.isfinite(r): rs.append(float(r))
        else: undef += 1
    agree.append({"a": a, "b": b, "n": len(rs), "undefined": undef,
                  "spearman_mean": float(np.mean(rs)) if rs else None,
                  "spearman_sd": float(np.std(rs)) if rs else None,
                  "spearman_lo": float(np.percentile(rs, 2.5)) if rs else None,
                  "spearman_hi": float(np.percentile(rs, 97.5)) if rs else None})

json.dump({"images": score_set, "constant_maps": const, "agreement": agree},
          open(os.path.join(ROOT, "out", "xai", "agreement_full.json"), "w"), indent=2)
np.savez_compressed(os.path.join(ROOT, "out", "xai", "cams_all24.npz"),
                    **{f"{m}|{f}": v for (m, f), v in maps.items()})
print(f"\n{'pair':26} {'n':>3} {'undef':>6} {'mean r':>8} {'sd':>7}")
for g in agree:
    mr = f"{g['spearman_mean']:+.3f}" if g["spearman_mean"] is not None else "   -  "
    sd = f"{g['spearman_sd']:.3f}" if g["spearman_sd"] is not None else "  -  "
    print(f"{g['a']+' vs '+g['b']:26} {g['n']:>3} {g['undefined']:>6} {mr:>8} {sd:>7}")
