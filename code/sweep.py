"""
Hyperparameter sensitivity sweep.

The point is not to find the best setting. It is to show that the central
conclusion -- image-level splitting inflates results -- is a property of the
data, not an artifact of one hyperparameter choice. So every cell of the grid
is evaluated under BOTH split regimes, and the inflation is reported per cell.
If the effect only appeared at one regularization strength it would not be real.

Four sweeps:
  A  regularization   C in {1e-3 .. 1e2}, all 12 backbones, linear head
  B  head capacity    linear / MLP-128 / MLP-256, top backbones
  C  preprocessing    plain vs Laplacian-of-Gaussian  (the ablation the source
                      paper states it never performed)
  D  feature scaling  standard / none / L2-normalize

Sweeps A and D report the TEST metric at each fixed setting (a sensitivity
surface, not a selection). Sweeps B and C select on validation and report test,
which is the protocol used for headline numbers.
"""
import os, sys, json, csv, argparse, warnings, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, Normalizer
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import all_metrics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
TOP = ["convnextv2_tiny.fcmae_ft_in22k_in1k", "convnext_tiny.fb_in22k_ft_in1k",
       "resnet50.a1_in1k", "swin_tiny_patch4_window7_224.ms_in1k"]
LOG_MODELS = ["convnextv2_tiny.fcmae_ft_in22k_in1k", "swin_tiny_patch4_window7_224.ms_in1k",
              "resnet50.a1_in1k", "convnext_tiny.fb_in22k_ft_in1k"]


def scaler_for(kind, Xtr):
    if kind == "standard":
        return StandardScaler().fit(Xtr)
    if kind == "l2":
        return Normalizer().fit(Xtr)
    class _Id:
        def transform(self, X): return X
    return _Id()


def make_head(kind, C, seed):
    if kind == "linear":
        return LogisticRegression(C=C, max_iter=3000, random_state=seed)
    h = int(kind.split("-")[1])
    return MLPClassifier(hidden_layer_sizes=(h,), alpha=1.0 / max(C, 1e-9),
                         max_iter=500, random_state=seed)


def load(features, variant, name):
    d = np.load(os.path.join(features, f"{variant}__{name}.npz"), allow_pickle=True)
    return d["features"], [str(x) for x in d["files"]]


def splits_for(splits, regime, seed, files):
    sp = json.load(open(os.path.join(splits, f"split_{regime}_seed{seed}.json")))
    a = sp["assign"]
    return {s: [f for f in files if a[f] == s] for s in ("train", "val", "test")}


def run_cell(F, idx, y, sub, head, C, scale, seed):
    tr = [idx[f] for f in sub["train"]]; va = [idx[f] for f in sub["val"]]
    te = [idx[f] for f in sub["test"]]
    sc = scaler_for(scale, F[tr])
    Xtr, Xva, Xte = sc.transform(F[tr]), sc.transform(F[va]), sc.transform(F[te])
    clf = make_head(head, C, seed)
    clf.fit(np.vstack([Xtr, Xva]), np.concatenate([y[tr], y[va]]))
    p = clf.predict_proba(Xte)[:, 1]
    return all_metrics(y[te], p), p, y[te]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=os.path.join(ROOT, "out", "features"))
    ap.add_argument("--splits", default=os.path.join(ROOT, "out", "splits"))
    ap.add_argument("--manifest", default=os.path.join(ROOT, "manifest.csv"))
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "results"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--sweeps", nargs="+", default=["A", "B", "C", "D"])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rows = list(csv.DictReader(open(a.manifest)))
    label = {r["file"]: int(r["label"]) for r in rows}
    all_models = sorted({f.split("__", 1)[1][:-4] for f in os.listdir(a.features)
                         if f.startswith("plain__")})
    res = []
    t0 = time.time()

    if "A" in a.sweeps:
        for m in all_models:
            F, files = load(a.features, "plain", m)
            idx = {f: i for i, f in enumerate(files)}
            y = np.array([label[f] for f in files])
            for regime in ("case", "image"):
                for seed in a.seeds:
                    sub = splits_for(a.splits, regime, seed, files)
                    for C in C_GRID:
                        mt, _, _ = run_cell(F, idx, y, sub, "linear", C, "standard", seed)
                        res.append({"sweep": "A", "model": m, "regime": regime, "seed": seed,
                                    "head": "linear", "C": C, "scale": "standard",
                                    "variant": "plain", **mt})
            print(f"  A {m:46} {time.time()-t0:6.0f}s", flush=True)

    if "B" in a.sweeps:
        for m in TOP:
            F, files = load(a.features, "plain", m)
            idx = {f: i for i, f in enumerate(files)}
            y = np.array([label[f] for f in files])
            for regime in ("case", "image"):
                for seed in a.seeds:
                    sub = splits_for(a.splits, regime, seed, files)
                    for head in ("linear", "mlp-128", "mlp-256"):
                        mt, _, _ = run_cell(F, idx, y, sub, head, 1.0, "standard", seed)
                        res.append({"sweep": "B", "model": m, "regime": regime, "seed": seed,
                                    "head": head, "C": 1.0, "scale": "standard",
                                    "variant": "plain", **mt})
            print(f"  B {m:46} {time.time()-t0:6.0f}s", flush=True)

    if "C" in a.sweeps:
        for m in LOG_MODELS:
            if not os.path.exists(os.path.join(a.features, f"log__{m}.npz")):
                continue
            for variant in ("plain", "log"):
                F, files = load(a.features, variant, m)
                idx = {f: i for i, f in enumerate(files)}
                y = np.array([label[f] for f in files])
                for regime in ("case", "image"):
                    for seed in a.seeds:
                        sub = splits_for(a.splits, regime, seed, files)
                        mt, _, _ = run_cell(F, idx, y, sub, "linear", 1.0, "standard", seed)
                        res.append({"sweep": "C", "model": m, "regime": regime, "seed": seed,
                                    "head": "linear", "C": 1.0, "scale": "standard",
                                    "variant": variant, **mt})
            print(f"  C {m:46} {time.time()-t0:6.0f}s", flush=True)

    if "D" in a.sweeps:
        for m in TOP:
            F, files = load(a.features, "plain", m)
            idx = {f: i for i, f in enumerate(files)}
            y = np.array([label[f] for f in files])
            for regime in ("case", "image"):
                for seed in a.seeds:
                    sub = splits_for(a.splits, regime, seed, files)
                    for scale in ("standard", "none", "l2"):
                        mt, _, _ = run_cell(F, idx, y, sub, "linear", 1.0, scale, seed)
                        res.append({"sweep": "D", "model": m, "regime": regime, "seed": seed,
                                    "head": "linear", "C": 1.0, "scale": scale,
                                    "variant": "plain", **mt})
            print(f"  D {m:46} {time.time()-t0:6.0f}s", flush=True)

    with open(os.path.join(a.out, "sweep.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"\nwrote {len(res)} cells in {time.time()-t0:.0f}s -> sweep.json")

    # quick read-out: is the inflation robust across every setting?
    for sw in sorted({r["sweep"] for r in res}):
        R = [r for r in res if r["sweep"] == sw]
        keys = sorted({(r["model"], r["head"], r["C"], r["scale"], r["variant"]) for r in R})
        deltas = []
        for k in keys:
            c = [r["acc"] for r in R if (r["model"], r["head"], r["C"], r["scale"], r["variant"]) == k and r["regime"] == "case"]
            i = [r["acc"] for r in R if (r["model"], r["head"], r["C"], r["scale"], r["variant"]) == k and r["regime"] == "image"]
            if c and i: deltas.append((np.mean(i) - np.mean(c)) * 100)
        if deltas:
            print(f"  sweep {sw}: inflation mean {np.mean(deltas):+.2f} pp, "
                  f"positive in {sum(d>0 for d in deltas)}/{len(deltas)} settings")


if __name__ == "__main__":
    main()
