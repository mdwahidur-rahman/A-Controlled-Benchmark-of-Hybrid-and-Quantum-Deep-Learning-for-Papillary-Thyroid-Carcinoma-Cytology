"""LaTeX fragments for the round-two statistical tables, generated from
out/results/stats_round2.json and stats_round2b.json so that no number in the
manuscript is typed by hand."""
import os, json, itertools
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "out", "results")
OUT = os.path.join(ROOT, "out", "tables")
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
    "efficientformerv2_s2.snap_dist_in1k": "EfficientFormerV2-S2",
}
ORDER = ["ResNet50", "DenseNet121", "EfficientNetV2-S", "ConvNeXt-T",
         "ConvNeXtV2-T", "Swin-T", "SwinV2-T", "ViT-B/16", "DeiT III-S",
         "MaxViT-T", "CoAtNet-0", "EfficientFormerV2-S2"]

a = json.load(open(os.path.join(RES, "stats_round2.json")))
b = json.load(open(os.path.join(RES, "stats_round2b.json")))

# ---- Table: per-encoder partitioning contrast, cluster intervals ----------
rows = []
by = {r["model"]: r for r in b["case_vs_image"]}
for m in sorted(by, key=lambda k: ORDER.index(SHORT[k])):
    r = by[m]
    rows.append(f"{SHORT[m]:22} & ${r['delta_acc_pp']:+.2f}$ & "
                f"$[{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}]$ & "
                f"${r['boot_p']:.3f}$ & "
                f"${'1.000' if r['boot_p_holm'] >= 0.9995 else f'{r[chr(34)+chr(34)] if False else r[chr(98)+chr(111)+chr(111)+chr(116)+chr(95)+chr(112)+chr(95)+chr(104)+chr(111)+chr(108)+chr(109)]:.3f}'}$ \\\\")
t1 = "\n".join(rows)

# ---- Table: full 12 x 12 architecture contrast matrix (Delta AUC x 100) ---
mm = {(r["a"], r["b"]): r for r in b["model_vs_model"]}
inv = {v: k for k, v in SHORT.items()}
grid = []
for i, A in enumerate(ORDER):
    cells = []
    for j, B in enumerate(ORDER):
        if i == j:
            cells.append("---"); continue
        ka, kb = inv[A], inv[B]
        if (ka, kb) in mm:
            v = mm[(ka, kb)]["delta_auc"] * 100
        else:
            v = -mm[(kb, ka)]["delta_auc"] * 100
        cells.append(f"${v:+.2f}$")
    grid.append(f"{A:22} & " + " & ".join(cells) + r" \\")
t2 = "\n".join(grid)

os.makedirs(OUT, exist_ok=True)
open(os.path.join(OUT, "round2_per_encoder_cluster.tex"), "w").write(t1 + "\n")
open(os.path.join(OUT, "round2_auc_matrix.tex"), "w").write(t2 + "\n")
print(t1); print(); print(t2)

summary = {
    "encoder_level": a["encoder_level"],
    "mvm_unadjusted_excl_zero": sum(r["excludes_zero"] for r in b["model_vs_model"]),
    "mvm_holm_significant": sum(r["holm_significant"] for r in b["model_vs_model"]),
    "mvm_n": len(b["model_vs_model"]),
    "delong": b["delong_reference"],
    "cvi_unadjusted_excl_zero": sum(r["excludes_zero"] for r in b["case_vs_image"]),
    "cvi_holm_significant": sum(r["holm_significant"] for r in b["case_vs_image"]),
    "min_boot_p_mvm": min(r["boot_p_holm"] for r in b["model_vs_model"]),
    "largest_auc_gap": max(abs(r["delta_auc"]) for r in b["model_vs_model"]) * 100,
}
json.dump(summary, open(os.path.join(RES, "round2_summary.json"), "w"), indent=2)
print("\n" + json.dumps(summary, indent=1)[:1200])
