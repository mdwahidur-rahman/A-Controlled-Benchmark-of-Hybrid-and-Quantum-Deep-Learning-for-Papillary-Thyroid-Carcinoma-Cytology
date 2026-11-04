"""Group-cluster percentile intervals for every encoder under both rules, for
MCC, accuracy and AUC. These are the primary intervals of the revision: the
resampling unit is the recovered group, drawn with all of its fields and all of
its appearances across seeds. The field-level interval is computed alongside
only so that the two can be shown together and the wider one used."""
import os, json
import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "out", "results")
B = 10000
SHORT = {
    "resnet50.a1_in1k": "ResNet50", "densenet121.ra_in1k": "DenseNet121",
    "tf_efficientnetv2_s.in21k_ft_in1k": "EfficientNetV2-S",
    "convnext_tiny.fb_in22k_ft_in1k": "ConvNeXt-T",
    "convnextv2_tiny.fcmae_ft_in22k_in1k": "ConvNeXtV2-T",
    "swin_tiny_patch4_window7_224.ms_in1k": "Swin-T",
    "swinv2_tiny_window8_256.ms_in1k": "SwinV2-T",
    "vit_base_patch16_224.augreg2_in21k_ft_in1k": "ViT-B/16",
    "deit3_small_patch16_224.fb_in22k_ft_in1k": "DeiT III-S",
    "maxvit_tiny_tf_224.in1k": "MaxViT-T",
    "coatnet_0_rw_224.sw_in1k": "CoAtNet-0",
    "efficientformerv2_s2.snap_dist_in1k": "EfficientFormerV2-S2"}


def load(m, reg):
    Y, P, C = [], [], []
    for s in range(5):
        d = np.load(os.path.join(RES, f"preds_plain_{m}_{reg}_linear_s{s}.npz"),
                    allow_pickle=True)
        Y.append(d["y"].astype(int)); P.append(d["p"].astype(float))
        C.append(np.array([str(x) for x in d["cases"]]))
    return np.concatenate(Y), np.concatenate(P), np.concatenate(C)


def mcc(y, p):
    yh = (p >= 0.5).astype(int)
    tp = int(((yh == 1) & (y == 1)).sum()); tn = int(((yh == 0) & (y == 0)).sum())
    fp = int(((yh == 1) & (y == 0)).sum()); fn = int(((yh == 0) & (y == 1)).sum())
    den = np.sqrt(float(tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return (tp*tn - fp*fn)/den if den > 0 else 0.0


def auc(y, p):
    if len(set(y)) < 2: return float("nan")
    r = stats.rankdata(p); n1 = int((y == 1).sum()); n0 = len(y)-n1
    return float((r[y == 1].sum() - n1*(n1+1)/2)/(n1*n0))


def acc(y, p):
    return float(((p >= 0.5).astype(int) == y).mean())


def ci(v, lo=2.5, hi=97.5):
    return float(np.percentile(v, lo)), float(np.percentile(v, hi))


out = {}
models = sorted({f.split("preds_plain_")[1].rsplit("_case_linear_s", 1)[0]
                 for f in os.listdir(RES)
                 if f.startswith("preds_plain_") and "_case_linear_s" in f})
for m in models:
    for reg in ("case", "image"):
        y, p, c = load(m, reg)
        uniq = np.array(sorted(set(c))); idx = {k: np.flatnonzero(c == k) for k in uniq}
        rg = np.random.default_rng(7)
        cl = {"mcc": [], "auc": [], "acc": []}
        for _ in range(B):
            dr = rg.choice(uniq, size=len(uniq), replace=True)
            sel = np.concatenate([idx[k] for k in dr])
            if len(set(y[sel])) < 2: continue
            cl["mcc"].append(mcc(y[sel], p[sel]))
            cl["auc"].append(auc(y[sel], p[sel]))
            cl["acc"].append(acc(y[sel], p[sel])*100)
        rf = np.random.default_rng(8); fl = {"mcc": [], "auc": [], "acc": []}
        n = len(y)
        for _ in range(B):
            sel = rf.integers(0, n, n)
            if len(set(y[sel])) < 2: continue
            fl["mcc"].append(mcc(y[sel], p[sel]))
            fl["auc"].append(auc(y[sel], p[sel]))
            fl["acc"].append(acc(y[sel], p[sel])*100)
        rec = {"model": m, "short": SHORT[m], "regime": reg,
               "n_fields": int(len(y)), "n_groups": int(len(uniq)),
               "mcc": mcc(y, p), "auc": auc(y, p), "acc": acc(y, p)*100}
        for k in ("mcc", "auc", "acc"):
            rec[f"{k}_cluster_ci"] = ci(np.array(cl[k]))
            rec[f"{k}_field_ci"] = ci(np.array(fl[k]))
            a = rec[f"{k}_cluster_ci"]; b = rec[f"{k}_field_ci"]
            rec[f"{k}_width_ratio"] = (a[1]-a[0])/(b[1]-b[0])
        out[f"{m}|{reg}"] = rec
        print(f"  {SHORT[m]:22}{reg:6} MCC {rec['mcc']:.4f} "
              f"cluster [{rec['mcc_cluster_ci'][0]:.3f},{rec['mcc_cluster_ci'][1]:.3f}] "
              f"field [{rec['mcc_field_ci'][0]:.3f},{rec['mcc_field_ci'][1]:.3f}] "
              f"x{rec['mcc_width_ratio']:.2f}", flush=True)
json.dump(out, open(os.path.join(RES, "cluster_intervals.json"), "w"), indent=2)
r = [v["mcc_width_ratio"] for v in out.values()]
print(f"\nMCC cluster/field width ratio: mean {np.mean(r):.3f}, "
      f"min {np.min(r):.3f}, max {np.max(r):.3f}")
r2 = [v["auc_width_ratio"] for v in out.values()]
print(f"AUC cluster/field width ratio: mean {np.mean(r2):.3f}, "
      f"min {np.min(r2):.3f}, max {np.max(r2):.3f}")
print("wrote out/results/cluster_intervals.json")
