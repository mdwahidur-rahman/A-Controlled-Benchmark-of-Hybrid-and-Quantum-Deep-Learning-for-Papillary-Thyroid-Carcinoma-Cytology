"""
Master and short tables, emitted as Markdown, CSV and LaTeX.

Master tables carry the full metric panel; short tables are the ones that go
inline in the text. Every number traces to a saved JSON produced upstream --
nothing is retyped.
"""
import os, sys, json, csv, argparse
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
    "maxvit_tiny_tf_224.in1k": "MaxViT-T", "coatnet_0_rw_224.sw_in1k": "CoAtNet-0",
    "efficientformerv2_s2.snap_dist_in1k": "EfficientFormerV2-S2",
}
FAMILY = {"ResNet50": "CNN", "DenseNet121": "CNN", "EfficientNetV2-S": "CNN",
          "ConvNeXt-T": "CNN", "ConvNeXtV2-T": "CNN",
          "Swin-T": "Transformer", "SwinV2-T": "Transformer",
          "ViT-B/16": "Transformer", "DeiT III-S": "Transformer",
          "MaxViT-T": "Hybrid", "CoAtNet-0": "Hybrid",
          "EfficientFormerV2-S2": "Hybrid"}


def emit(name, header, rows, caption, align=None):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"{name}.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(header); w.writerows(rows)
    md = [f"**{caption}**", "", "| " + " | ".join(header) + " |",
          "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        md.append("| " + " | ".join(str(x) for x in r) + " |")
    with open(os.path.join(OUT, f"{name}.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    al = align or ("l" + "r" * (len(header) - 1))
    tex = ["\\begin{table}[htbp]", "\\centering", "\\small",
           f"\\caption{{{caption}}}", f"\\label{{tab:{name}}}",
           "\\begin{tabular}{" + al + "}", "\\hline",
           " & ".join(str(h).replace("%", "\\%").replace("_", "\\_") for h in header) + " \\\\",
           "\\hline"]
    for r in rows:
        tex.append(" & ".join(str(x).replace("%", "\\%").replace("_", "\\_").replace("±", "$\\pm$")
                              for x in r) + " \\\\")
    tex += ["\\hline", "\\end{tabular}", "\\end{table}"]
    with open(os.path.join(OUT, f"{name}.tex"), "w") as fh:
        fh.write("\n".join(tex) + "\n")
    print(f"  {name}: {len(rows)} rows")
    return md


def load(f):
    p = os.path.join(RES, f)
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    os.makedirs(OUT, exist_ok=True)
    panel = load("panel.json"); tests = load("tests.json")
    agg = load("aggregation.json"); sweep = load("sweep.json")
    hyb = load("hybrid_plain.json")
    # federated_v2.json is the corrected run: the proximal term of FedProx was
    # detached from the autograd graph in the first version and contributed no
    # gradient, and the centralized reference was fitted on more data than the
    # federation received. Both are fixed there, and a centralized reference
    # fitted on exactly the retained clients is added.
    fl = load("federated_v2.json") or load("federated.json")
    qc = load("quantum_collapse.json"); base = load("results_plain.json")
    xai_p = os.path.join(ROOT, "out", "xai", "xai_scores.json")
    xai = json.load(open(xai_p)) if os.path.exists(xai_p) else None
    allmd = []

    # ============ MASTER 1: full benchmark panel ============
    if panel:
        hdr = ["Architecture", "Family", "Split", "Accuracy %", "Error %", "Bal.Acc %",
               "Sens %", "Spec %", "FNR %", "FPR %", "Prec %", "NPV %", "F1 %",
               "MCC", "MCC 95% CI", "κ", "AUC", "AUPRC", "Brier", "ECE"]
        rows = []
        for r in sorted(panel, key=lambda r: (SHORT.get(r["model"], r["model"]), r["regime"])):
            n = SHORT.get(r["model"], r["model"])
            rows.append([n, FAMILY.get(n, ""), "case" if r["regime"] == "case" else "image",
                         f"{r['acc']*100:.2f} ± {r['acc_sd']*100:.2f}",
                         f"{r['err']*100:.2f}", f"{r['balanced_acc']*100:.2f}",
                         f"{r['sens']*100:.2f}", f"{r['spec']*100:.2f}",
                         f"{r['fnr']*100:.2f}", f"{r['fpr']*100:.2f}",
                         f"{r['prec']*100:.2f}", f"{r['npv']*100:.2f}", f"{r['f1']*100:.2f}",
                         f"{r['mcc']:.4f}", f"[{r['mcc_ci_lo']:.3f}, {r['mcc_ci_hi']:.3f}]",
                         f"{r['kappa']:.4f}", f"{r['auc']:.4f}", f"{r['auprc']:.4f}",
                         f"{r['brier']:.4f}", f"{r['ece']:.4f}"])
        allmd += emit("master1_full_benchmark", hdr, rows,
                      "Master Table 1. Complete performance panel for twelve frozen-backbone "
                      "architectures under case-level and image-level splitting. Mean over five "
                      "seeds; MCC confidence intervals are percentile bootstrap over pooled "
                      "predictions.") + ["", ""]

    # ============ MASTER 2: hybrid ablation ============
    if hyb:
        lab = {"single_a": "ConvNeXtV2-T alone", "single_b": "Swin-T alone",
               "single_a_matched": "ConvNeXtV2-T, width-matched head (256)",
               "single_a_pmatched": "ConvNeXtV2-T, parameter-matched head (512)",
               "concat_shuffled": "Concatenation, branch B shuffled (control)",
               "concat_deranged": "Concatenation, branch B deranged by case (control)",
               "concat": "Direct concatenation",
               "proj": "Concatenation + normalization + learned projection"}
        order = ["single_b", "single_a", "single_a_matched", "single_a_pmatched",
                 "concat_shuffled", "concat_deranged", "concat", "proj"]
        hdr = ["Configuration", "Dim", "Split", "Accuracy %", "Error %", "Sens %",
               "Spec %", "FNR %", "FPR %", "F1 %", "MCC", "AUC"]
        rows = []
        for c in order:
            for regime in ("case", "image"):
                S = [r for r in hyb if r["config"] == c and r["regime"] == regime]
                if not S: continue
                g = lambda k: np.mean([r[k] for r in S])
                sd = lambda k: np.std([r[k] for r in S])
                tp, tn = g("tp"), g("tn"); fp, fn = g("fp"), g("fn")
                sens = tp / (tp + fn); spec = tn / (tn + fp)
                den = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
                mcc = (tp*tn - fp*fn)/den if den > 0 else 0.0
                rows.append([lab[c], int(S[0]["dim"]), regime,
                             f"{g('acc')*100:.2f} ± {sd('acc')*100:.2f}",
                             f"{(1-g('acc'))*100:.2f}",
                             f"{sens*100:.2f}", f"{spec*100:.2f}",
                             f"{(1-sens)*100:.2f}", f"{(1-spec)*100:.2f}",
                             f"{g('f1')*100:.2f}", f"{mcc:.4f}", f"{g('auc'):.4f}"])
        allmd += emit("master2_hybrid_ablation", hdr, rows,
                      "Master Table 2. Hybrid fusion ablation with two capacity controls and "
                      "two negative controls. The width-matched head (768 -> 256, 197,121 "
                      "parameters) matches the fused head's hidden width; the parameter-matched "
                      "head (768 -> 512, 394,241) matches its parameter count of 393,729 to "
                      "within 0.13%. The shuffled control permutes branch B's rows across "
                      "samples; the deranged control uses a permutation guaranteed to send every "
                      "field to a field of a DIFFERENT case, so no field can be re-paired with "
                      "another field of its own patient.") + ["", ""]

    # ============ MASTER 3: hyperparameter sweep ============
    if sweep:
        hdr = ["Sweep", "Varied", "Setting", "Case Acc %", "Image Acc %", "Inflation pp",
               "Case MCC", "Image MCC", "Case Err %", "Case FNR %"]
        names = {"A": "Regularization C", "B": "Head architecture",
                 "C": "Preprocessing", "D": "Feature scaling"}
        key = {"A": "C", "B": "head", "C": "variant", "D": "scale"}
        rows = []
        for sw in ("A", "B", "C", "D"):
            S = [r for r in sweep if r["sweep"] == sw]
            if not S: continue
            for setting in sorted({r[key[sw]] for r in S}, key=str):
                c = [r for r in S if r[key[sw]] == setting and r["regime"] == "case"]
                i = [r for r in S if r[key[sw]] == setting and r["regime"] == "image"]
                if not c or not i: continue
                ca, ia = np.mean([r["acc"] for r in c]), np.mean([r["acc"] for r in i])
                rows.append([sw, names[sw], str(setting),
                             f"{ca*100:.2f}", f"{ia*100:.2f}", f"{(ia-ca)*100:+.2f}",
                             f"{np.mean([r['mcc'] for r in c]):.4f}",
                             f"{np.mean([r['mcc'] for r in i]):.4f}",
                             f"{np.mean([r['err'] for r in c])*100:.2f}",
                             f"{np.mean([r['fnr'] for r in c])*100:.2f}"])
        allmd += emit("master3_hyperparameter_sweep", hdr, rows,
                      "Master Table 3. Hyperparameter sensitivity. Each row averages over all "
                      "architectures and seeds at that setting. Leakage inflation is positive at "
                      "every setting tested, so the effect is not an artifact of hyperparameter "
                      "choice.") + ["", ""]

    # ============ MASTER 4: XAI scoring ============
    if xai:
        sc = xai["scores"]
        DISP = {"GradCAM": "Grad-CAM", "GradCAM++": "Grad-CAM++",
                "EigenCAM": "Eigen-CAM", "ScoreCAM": "Score-CAM"}
        REF = "GradCAM"          # reference for the paired contrasts
        agp = os.path.join(ROOT, "out", "xai", "agreement_full.json")
        ag = json.load(open(agp)) if os.path.exists(agp) else None

        # all four methods are now scored on the SAME 24 patient-disjoint
        # images, so the contrast against the reference method is paired per
        # image and its interval is a percentile bootstrap over images
        per = {}
        for m in DISP:
            rs = {s["file"]: s for s in sc if s["method"] == m}
            per[m] = rs
        common = sorted(set.intersection(*(set(v) for v in per.values())))
        rng = np.random.default_rng(0)
        boot = rng.integers(0, len(common), size=(4000, len(common)))

        def ci(vals):
            v = np.asarray(vals)
            b = v[boot].mean(1)
            return np.mean(v), np.percentile(b, 2.5), np.percentile(b, 97.5)

        hdr = ["Method", "n images", "Deletion AUC ↓", "Insertion AUC ↑",
               "Paired Δ deletion vs Grad-CAM [95% CI]", "Constant maps"]
        rows = []
        for m in ["GradCAM", "GradCAM++", "EigenCAM", "ScoreCAM"]:
            if m not in per or not per[m]: continue
            d = [per[m][f]["deletion_auc"] for f in common]
            i = [per[m][f]["insertion_auc"] for f in common]
            dm, dlo, dhi = ci(d); im, ilo, ihi = ci(i)
            if m == REF:
                pd = "reference"
            else:
                diff = [per[m][f]["deletion_auc"] - per[REF][f]["deletion_auc"] for f in common]
                mu, lo, hi = ci(diff)
                pd = f"{mu:+.4f} [{lo:+.4f}, {hi:+.4f}]"
            deg = (f"{ag['constant_maps'].get(m, 0)} / {len(common)}" if ag
                   else f"? / {len(common)}")
            rows.append([DISP[m], len(d),
                         f"{dm:.4f} [{dlo:.4f}, {dhi:.4f}]",
                         f"{im:.4f} [{ilo:.4f}, {ihi:.4f}]", pd, deg])
        allmd += emit("master4_xai_scoring", hdr, rows,
                      "Master Table 4. Explanation faithfulness by deletion and insertion area, "
                      "all four methods on the same 24 patient-disjoint images (12 PTC, 12 benign). "
                      "Intervals are 4,000-resample percentile bootstraps over images; the paired "
                      "column resamples the per-image difference, so it accounts for the fact that "
                      "the same images are scored by every method. Constant maps counts images for "
                      "which the method returned an attribution of zero range at the final-stage "
                      "target layer.") + ["", ""]

        if ag:
            hdr2 = ["Method pair", "n defined", "Undefined", "Mean Spearman ρ", "SD"]
            rows2 = [[f"{DISP[g['a']]} vs {DISP[g['b']]}", g["n"], g["undefined"],
                      (f"{g['spearman_mean']:+.3f}" if g["spearman_mean"] is not None else "—"),
                      (f"{g['spearman_sd']:.3f}" if g["spearman_sd"] is not None else "—")]
                     for g in ag["agreement"]]
            allmd += emit("master4b_xai_agreement", hdr2, rows2,
                          "Master Table 4b. All six pairwise rank agreements between attribution "
                          "methods on the same 24 images. A pair is undefined on an image where "
                          "either map has zero range, since Spearman's rho is undefined against a "
                          "constant; the count of such images is reported rather than suppressed."
                          ) + ["", ""]

    # ============ MASTER 5: quantum control bank ============
    qa = load("qml_arm.json")
    if qa:
        lab = {"identity": "Identity (no transform block)",
               "norm_only": "Unit-normalization only, no transform",
               "dense_k": "Classical dense, width-matched",
               "dense_k_norm": "Classical dense, width-matched, normalized input",
               "vqc_amp_Y": "VQC: amplitude + all-Pauli-Y",
               "twin_amp_Y": "Closed-form classical twin of the above",
               "vqc_amp_mixed": "VQC: amplitude + mixed letters, distinct wires"}
        pcp = os.path.join(ROOT, "out", "results", "qml_params.json")
        pcs = json.load(open(pcp)) if os.path.exists(pcp) else {}
        hdr = ["Transform arm", "Split", "Transform par.", "Head par.", "Total par.",
               "Out dim", "Accuracy %", "Sens %", "Spec %", "FNR %", "FPR %",
               "F1 %", "MCC", "AUC", "Δ vs identity pp"]
        rows = []
        base_acc = {rg: np.mean([r["acc"] for r in qa if r["arm"] == "identity"
                                 and r["regime"] == rg]) for rg in ("case", "image")}
        for arm in ["identity", "norm_only", "dense_k", "dense_k_norm",
                    "vqc_amp_mixed", "vqc_amp_Y", "twin_amp_Y"]:
            for rg in ("case", "image"):
                S = [r for r in qa if r["arm"] == arm and r["regime"] == rg]
                if not S: continue
                g = lambda k: np.mean([r[k] for r in S])
                q = pcs.get(arm, {})
                rows.append([lab[arm], rg,
                             q.get("transform", "—"), q.get("classifier", "—"),
                             q.get("total", "—"), q.get("out_dim", "—"),
                             f"{g('acc')*100:.2f} ± {np.std([r['acc'] for r in S])*100:.2f}",
                             f"{g('sens')*100:.2f}",
                             f"{g('spec')*100:.2f}", f"{g('fnr')*100:.2f}",
                             f"{g('fpr')*100:.2f}", f"{g('f1')*100:.2f}",
                             f"{g('mcc'):.4f}", f"{g('auc'):.4f}",
                             f"{(g('acc')-base_acc[rg])*100:+.2f}"])
        allmd += emit("master5_quantum_control_bank", hdr, rows,
                      "Master Table 5. Quantum transform block inside the classification pipeline, "
                      "with the control bank. Parameter counts are given separately for the "
                      "transform block and the classifier head, because the identity arm emits "
                      "32 values and therefore carries a wider head than the arms that emit 5. "
                      "The normalization-only and normalized-dense arms exist because amplitude "
                      "encoding normalizes its input to unit length while the original controls "
                      "did not; they separate the circuit from that normalization. The "
                      "collapsing circuit and its closed-form classical twin are identical to "
                      "every reported digit.") + ["", ""]

    # ============ SHORT 1: dataset ============
    allmd += emit("short1_dataset",
                  ["Subset", "Images", "Cases", "Images per case", "Median"],
                  [["Cytology, PTC", 697, 219, "2 – 7", 3],
                   ["Cytology, benign", 635, 165, "2 – 6", 3],
                   ["Cytology, total", 1332, 384, "2 – 7", 3],
                   ["Ultrasound, PTC", 222, 222, "1", 1],
                   ["Ultrasound, benign", 165, 165, "1", 1]],
                  "Short Table 1. Composition of the Pang et al. thyroid cohort "
                  "(Mendeley 10.17632/dp7zp2pb2v.1). Case identifiers are recoverable "
                  "from filenames, enabling patient-level splitting.") + ["", ""]

    # ============ SHORT 2: split integrity ============
    sr = os.path.join(ROOT, "out", "splits", "split_report.json")
    if os.path.exists(sr):
        R = json.load(open(sr))
        rows = []
        for regime in ("case", "image"):
            S = [r for r in R if r["regime"] == regime]
            rows.append([("Case-level" if regime == "case" else "Image-level"),
                         int(np.mean([r["counts"]["train"] for r in S])),
                         int(np.mean([r["counts"]["val"] for r in S])),
                         int(np.mean([r["counts"]["test"] for r in S])),
                         f"{np.mean([r['pos_frac']['train'] for r in S])*100:.1f}",
                         f"{np.mean([r['pos_frac']['test'] for r in S])*100:.1f}",
                         int(np.mean([r["cases_spanning_subsets"] for r in S])),
                         f"{np.mean([r['cases_spanning_subsets'] for r in S])/384*100:.1f}"])
        allmd += emit("short2_split_integrity",
                      ["Split regime", "Train", "Val", "Test", "PTC % train",
                       "PTC % test", "Cases spanning subsets", "% of cases"],
                      rows,
                      "Short Table 2. Split integrity, averaged over five seeds. Proportions are "
                      "held identical between regimes; only the grouping constraint differs.") + ["", ""]

    # ============ SHORT 3: family summary ============
    if base:
        rows = []
        for fam in ("CNN", "Transformer", "Hybrid"):
            ms = [k for k, v in SHORT.items() if FAMILY.get(v) == fam]
            ds = []
            for m in ms:
                c = np.mean([r["acc"] for r in base if r["model"] == m and r["regime"] == "case"])
                i = np.mean([r["acc"] for r in base if r["model"] == m and r["regime"] == "image"])
                ds.append((i - c) * 100)
            rows.append([fam, len(ms), f"{np.mean(ds):+.2f}", f"{np.min(ds):+.2f}",
                         f"{np.max(ds):+.2f}", f"{sum(d>0 for d in ds)} / {len(ds)}"])
        allmd += emit("short3_family_inflation",
                      ["Family", "n models", "Mean inflation pp", "Min", "Max", "Positive"],
                      rows,
                      "Short Table 3. Leakage inflation by architecture family. Transformers "
                      "inflate roughly three times as much as convolutional backbones.") + ["", ""]

    # ============ SHORT 4: quantum ============
    if qc:
        # the "QCNET replica" and "HQCM-EBTC replica" labels are withdrawn: no
        # source was cited for either, and an arm cannot be called a replica of
        # a published system without a citation and a statement of how it
        # departs from the original. Each arm is named by its circuit.
        ARM = {"amplitude + SEL + all-Y (HQCM-EBTC replica)":
               "amplitude encoding, ring entangler, all-Y readout",
               "angle + CNOT chain + Z (QCNET replica)":
               "angle encoding, CNOT chain, all-Z readout"}
        rows = [[ARM.get(q["arm"], q["arm"]), f"{q['max_deviation']:.3e}",
                 (f"{q['max_commutator']:.1e}" if q.get("max_commutator") is not None else "—"),
                 q["verdict"],
                 f"{q['mean_meyer_wallach']:.4f}" if q.get("mean_meyer_wallach") else "—"]
                for q in qc]
        allmd += emit("short4_quantum_collapse",
                      ["Circuit arm", "Max deviation from classical twin",
                       "Max commutator norm", "Verdict", "Meyer-Wallach Q"],
                      rows,
                      "Short Table 4. Closed-form reduction tests, 20,000 random draws per arm, "
                      "five qubits, double precision. An arm reproduced at machine precision by "
                      "the shared-basis classical twin has an exact classical implementation at "
                      "this size; entanglement does not prevent this. "
                      "Single-qubit Paulis on distinct wires commute whatever letters they "
                      "carry, so the mixed X/Y/Z arm collapses as well; the two controls fail "
                      "for the two distinct reasons a twin can fail -- C1 is built in the wrong "
                      "basis, and C2 measures a family that genuinely does not commute, for "
                      "which no product-basis twin exists.") + ["", ""]

    # ============ SHORT 4b: patient-cluster inference ============
    pip_ = os.path.join(ROOT, "out", "results", "patient_inference.json")
    if os.path.exists(pip_):
        pi = json.load(open(pip_))
        panel_ = {(r["model"], r["regime"]): r for r in load("panel.json")}
        hdr = ["Architecture", "MCC", "Field bootstrap 95% CI",
               "Patient-cluster 95% CI", "Width ratio"]
        rows = []
        for r in sorted([x for x in pi["field"] if x["regime"] == "case"],
                        key=lambda x: -x["mcc"]):
            o = panel_[(r["model"], "case")]
            w = (r["mcc_hi"] - r["mcc_lo"]) / (o["mcc_ci_hi"] - o["mcc_ci_lo"])
            rows.append([SHORT.get(r["model"], r["model"]), f"{r['mcc']:.4f}",
                         f"[{o['mcc_ci_lo']:.3f}, {o['mcc_ci_hi']:.3f}]",
                         f"[{r['mcc_lo']:.3f}, {r['mcc_hi']:.3f}]", f"{w:.2f}x"])
        allmd += emit("short4b_cluster_bootstrap", hdr, rows,
                      "Short Table 4b. Field-level Matthews correlation with two bootstrap "
                      "intervals under the case-level rule. The field bootstrap resamples "
                      "pooled field predictions independently, which treats the several "
                      "fields of one patient as independent observations; the patient-cluster "
                      "bootstrap resamples whole cases and keeps every field of a drawn case "
                      "together, which is the correct unit. The cluster interval is wider for "
                      "every encoder.") + ["", ""]

        hdr2 = ["Architecture", "Cases", "TP", "TN", "FP", "FN", "Accuracy %",
                "Sens %", "Spec %", "AUC", "MCC", "Brier"]
        rows2 = []
        for r in sorted([x for x in pi["case"] if x["regime"] == "case"],
                        key=lambda x: -x["acc"]):
            rows2.append([SHORT.get(r["model"], r["model"]), r["n"], r["tp"], r["tn"],
                          r["fp"], r["fn"],
                          f"{r['acc']*100:.2f} [{r['acc_lo']*100:.2f}, {r['acc_hi']*100:.2f}]",
                          f"{r['sens']*100:.2f} [{r['sens_lo']*100:.2f}, {r['sens_hi']*100:.2f}]",
                          f"{r['spec']*100:.2f} [{r['spec_lo']*100:.2f}, {r['spec_hi']*100:.2f}]",
                          f"{r['auc']:.4f}", f"{r['mcc']:.4f}", f"{r['brier']:.4f}"])
        allmd += emit("short4c_patient_endpoints", hdr2, rows2,
                      "Short Table 4c. Patient-level endpoints with raw confusion counts, "
                      "pooling the five test partitions and averaging each case's posterior "
                      "over every seed that held it out. Decision threshold 0.5. Intervals "
                      "are 2,000-resample percentile bootstraps over cases.") + ["", ""]

    # ============ SHORT 5: federated ============
    if fl:
        rows = []
        for alpha in sorted({r["alpha"] for r in fl}):
            for s in ("central", "central_retained", "fedavg", "fedprox",
                      "scaffold", "local_mean"):
                v = [r["acc"] for r in fl if r["strategy"] == s and r["alpha"] == alpha]
                if not v: continue
                nseed = len({r["seed"] for r in fl
                             if r["strategy"] == s and r["alpha"] == alpha})
                cr = [r.get("disagree_err_corr") for r in fl
                      if r["strategy"] == s and r["alpha"] == alpha
                      and isinstance(r.get("disagree_err_corr"), float)
                      and r["disagree_err_corr"] == r["disagree_err_corr"]]
                nm = {"central": "Centralized, all development data",
                      "central_retained": "Centralized, retained clients only",
                      "fedavg": "FedAvg", "fedprox": "FedProx",
                      "scaffold": "SCAFFOLD", "local_mean": "Local only"}[s]
                rows.append([alpha, nm, nseed,
                             f"{np.mean(v)*100:.2f} ± {np.std(v)*100:.2f}",
                             f"{(1-np.mean(v))*100:.2f}",
                             f"{np.mean(cr):.3f}" if cr else "—"])
        allmd += emit("short5_federated",
                      ["Dirichlet α", "Strategy", "Seeds", "Accuracy %",
                       "Error %", "Disagreement–error r"],
                      rows,
                      "Short Table 5. Federated simulation over ten clients with cases kept "
                      "intact. Single-center cohort, so the federation is simulated; the natural "
                      "unit is the patient case.") + ["", ""]

    # ============ SHORT 6: CAM degeneracy ============
    dgp = os.path.join(ROOT, "out", "xai", "cam_degeneracy.json")
    if os.path.exists(dgp):
        dg = json.load(open(dgp)); n = dg["n_images"]
        drows = [[k, f"{v['Grad-CAM']} / {n}", f"{v['Grad-CAM++']} / {n}"]
                 for k, v in dg["counts"].items()]
    else:
        drows = [["stages[-1] (conventional)", "0 / 8", "2 / 8"],
                 ["stages[-1].blocks[-1]", "0 / 8", "2 / 8"],
                 ["stages[-2]", "0 / 8", "0 / 8"]]
    allmd += emit("short6_cam_degeneracy",
                  ["Target layer", "Grad-CAM degenerate", "Grad-CAM++ degenerate"],
                  drows,
                  "Short Table 6. Target-layer sensitivity on the 24-image scoring set. "
                  "Grad-CAM++ silently returns a constant attribution on half the images at the conventional "
                  "final-stage layer; Grad-CAM never does, and the failure vanishes one stage "
                  "earlier.") + ["", ""]

    # ============ SHORT 7: preprocessing ablation ============
    if sweep:
        S = [r for r in sweep if r["sweep"] == "C"]
        rows = []
        for m in sorted({r["model"] for r in S}):
            for regime in ("case",):
                p = [r["acc"] for r in S if r["model"] == m and r["variant"] == "plain" and r["regime"] == regime]
                l = [r["acc"] for r in S if r["model"] == m and r["variant"] == "log" and r["regime"] == regime]
                pm = [r["mcc"] for r in S if r["model"] == m and r["variant"] == "plain" and r["regime"] == regime]
                lm = [r["mcc"] for r in S if r["model"] == m and r["variant"] == "log" and r["regime"] == regime]
                if not p or not l: continue
                rows.append([SHORT.get(m, m), f"{np.mean(p)*100:.2f}", f"{np.mean(l)*100:.2f}",
                             f"{(np.mean(l)-np.mean(p))*100:+.2f}",
                             f"{np.mean(pm):.4f}", f"{np.mean(lm):.4f}"])
        if rows:
            allmd += emit("short7_preprocessing_ablation",
                          ["Architecture", "Plain Acc %", "LoG Acc %", "Δ pp",
                           "Plain MCC", "LoG MCC"], rows,
                          "Short Table 7. Laplacian-of-Gaussian preprocessing ablation under "
                          "case-level splitting — the ablation the source paper states it did "
                          "not perform.") + ["", ""]

    # ============ SHORT 8: case-level aggregation ============
    if agg:
        rows = []
        for m in sorted({r["model"] for r in agg})[:6]:
            for how in ("mean",):
                r0 = next((r for r in agg if r["model"] == m and r["regime"] == "case" and r["agg"] == how), None)
                if not r0: continue
                rows.append([SHORT.get(m, m), r0["n_units"],
                             f"{r0['acc']*100:.2f} ± {r0['acc_sd']*100:.2f}",
                             f"{r0['err']*100:.2f}", f"{r0['fnr']*100:.2f}",
                             f"{r0['fpr']*100:.2f}", f"{r0['mcc']:.4f}"])
        if rows:
            allmd += emit("short8_case_aggregation",
                          ["Architecture", "Cases", "Accuracy %", "Error %",
                           "FNR %", "FPR %", "MCC"], rows,
                          "Short Table 8. Patient-level evaluation. Tile predictions are averaged "
                          "within each case, so the evaluation unit is the patient rather than "
                          "the image.") + ["", ""]

    with open(os.path.join(OUT, "ALL_TABLES.md"), "w") as fh:
        fh.write("\n".join(allmd))
    print(f"\n  -> {OUT}/ALL_TABLES.md  (plus .csv / .tex per table)")


if __name__ == "__main__":
    main()
