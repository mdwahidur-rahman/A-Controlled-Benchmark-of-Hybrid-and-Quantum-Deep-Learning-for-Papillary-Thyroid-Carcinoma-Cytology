"""
Complete the Score-CAM arm.

The original xai.py scored Score-CAM on only the first 8 of the 24 images
because Score-CAM needs one forward pass per channel of the target stage and
is roughly fifty times slower than the gradient methods. A reviewer correctly
pointed out that the four methods are then compared on different image sets,
so the Score-CAM row of the table is not commensurable with the other three.

This script reproduces xai.py's model, split and image selection exactly --
same seed, same shuffle, same slice -- and scores Score-CAM on the full
24-image set. Image 0 is recomputed deliberately as a consistency check: if
this script's pipeline matched the original, its deletion and insertion areas
must reproduce the stored values to within floating-point noise.

Output: out/xai/scorecam_full.json, merged into xai_scores.json by
merge_scorecam.py.
"""
import os, sys, json, csv, warnings, time
import numpy as np
import torch, timm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pytorch_grad_cam import ScoreCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xai import FrozenHead, load_img, del_ins

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
torch.set_num_threads(2)
MODEL = "convnextv2_tiny.fcmae_ft_in22k_in1k"
VARIANT, SEED, N_SCORE = "plain", 0, 24

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
pick_pos = [f for f in correct if label[f] == 1]
pick_neg = [f for f in correct if label[f] == 0]
rng.shuffle(pick_pos); rng.shuffle(pick_neg)
score_set = pick_pos[:N_SCORE // 2] + pick_neg[:N_SCORE // 2]
assert len(score_set) == 24, len(score_set)

# image 0 is the reproducibility check, images 8..23 are the new work
todo = [score_set[0]] + score_set[8:]
cam_alg = ScoreCAM(model=model, target_layers=[bb.stages[-1]])
out, t0 = [], time.time()
for n, f in enumerate(todo):
    x, _ = load_img(os.path.join(ROOT, "data", VARIANT, f), size, cfg["mean"], cfg["std"])
    cls = label[f]
    g = cam_alg(input_tensor=x, targets=[ClassifierOutputTarget(cls)])[0]
    g = (g - g.min()) / (g.max() - g.min() + 1e-8)
    dA, _ = del_ins(model, x, g, cls, mode="del")
    iA, _ = del_ins(model, x, g, cls, mode="ins")
    out.append({"method": "ScoreCAM", "file": f, "label": cls,
                "deletion_auc": dA, "insertion_auc": iA,
                "role": "recheck" if n == 0 else "new"})
    print(f"  [{n+1:2}/{len(todo)}] {f:28} del {dA:.4f} ins {iA:.4f}  {time.time()-t0:6.0f}s", flush=True)

json.dump({"score_set": score_set, "rows": out},
          open(os.path.join(ROOT, "out", "xai", "scorecam_full.json"), "w"), indent=2)
print(f"\nwrote {len(out)} rows in {time.time()-t0:.0f}s")
