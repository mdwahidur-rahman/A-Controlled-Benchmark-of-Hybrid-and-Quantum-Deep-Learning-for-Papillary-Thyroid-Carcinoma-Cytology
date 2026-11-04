"""Target-layer degeneracy sweep on the full 24-image scoring set.

The published sweep used the eight images Score-CAM had been scored on, which
were all malignant. This repeats it on the same 24 patient-disjoint images the
faithfulness scores now use, for the three candidate target layers.
"""
import os, sys, json, csv, warnings
import numpy as np
import torch, timm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xai import FrozenHead, load_img

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
torch.set_num_threads(2)
MODEL = "convnextv2_tiny.fcmae_ft_in22k_in1k"

rows = list(csv.DictReader(open(os.path.join(ROOT, "manifest.csv"))))
label = {r["file"]: int(r["label"]) for r in rows}
d = np.load(os.path.join(ROOT, "out", "features", f"plain__{MODEL}.npz"), allow_pickle=True)
F, files = d["features"], [str(x) for x in d["files"]]
idx = {f: i for i, f in enumerate(files)}
y_all = np.array([label[f] for f in files])
assign = json.load(open(os.path.join(ROOT, "out", "splits", "split_case_seed0.json")))["assign"]
dev = [f for f in files if assign[f] in ("train", "val")]
te = [f for f in files if assign[f] == "test"]
sc = StandardScaler().fit(F[[idx[f] for f in dev]])
clf = LogisticRegression(C=1.0, max_iter=3000, random_state=0)
clf.fit(sc.transform(F[[idx[f] for f in dev]]), y_all[[idx[f] for f in dev]])
bb = timm.create_model(MODEL, pretrained=False, num_classes=0)
sd = torch.load(os.path.join(ROOT, "weights", f"{MODEL}.pth"), map_location="cpu", weights_only=True)
bb.load_state_dict({k: v for k, v in sd.items() if not k.startswith("head.fc")}, strict=False)
bb.eval()
model = FrozenHead(bb, sc.mean_, sc.scale_, clf.coef_[0], float(clf.intercept_[0])).eval()
cfg = bb.default_cfg; size = cfg["input_size"][-1]

rng = np.random.default_rng(0)
probs = clf.predict_proba(sc.transform(F[[idx[f] for f in te]]))[:, 1]
correct = [f for f, p in zip(te, probs) if (p >= 0.5) == (label[f] == 1)]
pos = [f for f in correct if label[f] == 1]; neg = [f for f in correct if label[f] == 0]
rng.shuffle(pos); rng.shuffle(neg)
score_set = pos[:12] + neg[:12]

LAYERS = {"stages[-1] (conventional)": [bb.stages[-1]],
          "stages[-1].blocks[-1]": [bb.stages[-1].blocks[-1]],
          "stages[-2]": [bb.stages[-2]]}
out = {}
for lname, layer in LAYERS.items():
    out[lname] = {}
    for mname, M in (("Grad-CAM", GradCAM), ("Grad-CAM++", GradCAMPlusPlus)):
        alg = M(model=model, target_layers=layer)
        n = 0
        for f in score_set:
            x, _ = load_img(os.path.join(ROOT, "data", "plain", f), size, cfg["mean"], cfg["std"])
            g = alg(input_tensor=x, targets=[ClassifierOutputTarget(label[f])])[0]
            if np.ptp(g) < 1e-8:
                n += 1
        out[lname][mname] = n
        print(f"  {lname:28} {mname:11} {n} / {len(score_set)} degenerate", flush=True)
json.dump({"n_images": len(score_set), "counts": out},
          open(os.path.join(ROOT, "out", "xai", "cam_degeneracy.json"), "w"), indent=2)
