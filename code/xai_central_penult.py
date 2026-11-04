"""Score the centralized arm at the PENULTIMATE stage on the four common images.

Table 14 of the manuscript compares attribution faithfulness across the
centralized, federated and quantum arms. The federated and quantum arms were
scored by xai_arms.py at bb.stages[-2]; the centralized numbers in Table 12
come from xai.py at bb.stages[-1]. Those are different target layers and
produce different maps, so the three arms were not on a common footing even
though the text described them as being so.

This scores the centralized arm at stages[-2] on exactly the four images the
other two arms use, so that Table 14 compares like with like.
"""
import os, sys, json, csv, warnings, time
import numpy as np
import torch, timm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM, EigenCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xai import FrozenHead, load_img, del_ins

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
torch.set_num_threads(2)
MODEL = "convnextv2_tiny.fcmae_ft_in22k_in1k"
VARIANT, SEED = "plain", 0

arms = json.load(open(os.path.join(ROOT, "out", "xai", "xai_arms.json")))
common = sorted({r["file"] for r in arms["federated"]["scores"]})
print("common images:", common)

rows = list(csv.DictReader(open(os.path.join(ROOT, "manifest.csv"))))
label = {r["file"]: int(r["label"]) for r in rows}
d = np.load(os.path.join(ROOT, "out", "features", f"{VARIANT}__{MODEL}.npz"), allow_pickle=True)
F, files = d["features"], [str(x) for x in d["files"]]
idx = {f: i for i, f in enumerate(files)}
y_all = np.array([label[f] for f in files])
assign = json.load(open(os.path.join(ROOT, "out", "splits", f"split_case_seed{SEED}.json")))["assign"]
dev = [f for f in files if assign[f] in ("train", "val")]
sc = StandardScaler().fit(F[[idx[f] for f in dev]])
clf = LogisticRegression(C=1.0, max_iter=3000, random_state=SEED)
clf.fit(sc.transform(F[[idx[f] for f in dev]]), y_all[[idx[f] for f in dev]])

bb = timm.create_model(MODEL, pretrained=False, num_classes=0)
sd = torch.load(os.path.join(ROOT, "weights", f"{MODEL}.pth"), map_location="cpu", weights_only=True)
bb.load_state_dict({k: v for k, v in sd.items() if not k.startswith("head.fc")}, strict=False)
bb.eval()
model = FrozenHead(bb, sc.mean_, sc.scale_, clf.coef_[0], float(clf.intercept_[0])).eval()
cfg = bb.default_cfg; size = cfg["input_size"][-1]

methods = {"Grad-CAM": GradCAM, "Grad-CAM++": GradCAMPlusPlus,
           "Eigen-CAM": EigenCAM, "Score-CAM": ScoreCAM}
out, maps, t0 = [], {}, time.time()
for mname, M in methods.items():
    alg = M(model=model, target_layers=[bb.stages[-2]])
    for f in common:
        x, _ = load_img(os.path.join(ROOT, "data", VARIANT, f), size, cfg["mean"], cfg["std"])
        cls = label[f]
        g = alg(input_tensor=x, targets=[ClassifierOutputTarget(cls)])[0]
        g = (g - g.min()) / (g.max() - g.min() + 1e-8)
        dA, _ = del_ins(model, x, g, cls, mode="del")
        iA, _ = del_ins(model, x, g, cls, mode="ins")
        out.append({"method": mname, "file": f, "label": cls,
                    "deletion_auc": dA, "insertion_auc": iA})
        maps[f"{mname}|{f}"] = g.astype(np.float32)
    print(f"  {mname:11} {len(common)} images  {time.time()-t0:6.0f}s", flush=True)

json.dump({"scores": out, "layer": "stages[-2]", "images": common},
          open(os.path.join(ROOT, "out", "xai", "central_penult.json"), "w"), indent=2)
np.savez_compressed(os.path.join(ROOT, "out", "xai", "cams_central_penult.npz"), **maps)

# the Eigen-CAM invariance check the three arms make possible
cf = np.load(os.path.join(ROOT, "out", "xai", "cams_federated.npz"))
cq = np.load(os.path.join(ROOT, "out", "xai", "cams_quantum.npz"))
print("\nEigen-CAM map agreement across arms at stages[-2] (should be exact):")
for f in common:
    a = maps[f"Eigen-CAM|{f}"]; b = cf[f"Eigen-CAM|{f}"]; c = cq[f"Eigen-CAM|{f}"]
    print(f"  {f:24} |cen-fed| {np.abs(a-b).max():.2e}  |cen-qnt| {np.abs(a-c).max():.2e}  "
          f"|fed-qnt| {np.abs(b-c).max():.2e}")
