"""
Post-hoc statistics from saved test predictions.

Everything here reads the per-run prediction files written by benchmark.py, so
nothing is refitted and every number traces to a specific saved prediction
vector.

Produces:
  1. Full metric panel per (model, regime) with bootstrap CIs -- including both
     error rates separately and MCC.
  2. Pairwise DeLong (AUC) and McNemar (decisions) across all 12 backbones,
     within the case-level regime, and case-vs-image for each backbone.
  3. Case-level aggregation. Images are tiles from one slide; the clinically
     meaningful unit is the patient. Majority vote and mean-probability
     aggregation are both reported.
  4. Calibration: ECE and reliability bins.
"""
import os, sys, json, csv, argparse, warnings, itertools
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import all_metrics, ece, bootstrap_ci, delong_test, mcnemar_test

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_preds(resdir, variant, model, regime, head, seed):
    p = os.path.join(resdir, f"preds_{variant}_{model}_{regime}_{head}_s{seed}.npz")
    if not os.path.exists(p):
        return None
    d = np.load(p, allow_pickle=True)
    return (d["y"].astype(int), d["p"].astype(float),
            [str(x) for x in d["files"]], [str(x) for x in d["cases"]])


def aggregate_cases(y, p, cases, how="mean"):
    """Collapse tile predictions to one prediction per patient case."""
    by = {}
    for yi, pi, c in zip(y, p, cases):
        by.setdefault(c, {"y": yi, "p": []})["p"].append(pi)
    ys = np.array([v["y"] for v in by.values()])
    if how == "mean":
        ps = np.array([np.mean(v["p"]) for v in by.values()])
    else:  # majority vote on thresholded tiles, ties -> mean
        ps = np.array([float(np.mean([1.0 if x >= 0.5 else 0.0 for x in v["p"]]))
                       for v in by.values()])
    return ys, ps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(ROOT, "out", "results"))
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "results"))
    ap.add_argument("--variant", default="plain")
    ap.add_argument("--head", default="linear")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--boot", type=int, default=1500)
    a = ap.parse_args()

    base = json.load(open(os.path.join(a.results, f"results_{a.variant}.json")))
    models = sorted({r["model"] for r in base})

    # ---------- 1. metric panel with bootstrap CIs ----------
    panel = []
    pooled = {}
    for m in models:
        for regime in ("case", "image"):
            per_seed, Y, P = [], [], []
            for s in a.seeds:
                d = load_preds(a.results, a.variant, m, regime, a.head, s)
                if d is None: continue
                y, p, _, cases = d
                per_seed.append(all_metrics(y, p))
                Y.append(y); P.append(p)
            if not per_seed: continue
            yc, pc = np.concatenate(Y), np.concatenate(P)
            pooled[(m, regime)] = (yc, pc)
            row = {"model": m, "regime": regime, "n_seeds": len(per_seed)}
            for k in ("acc", "err", "balanced_acc", "sens", "spec", "fnr", "fpr",
                      "prec", "npv", "f1", "mcc", "kappa", "youden_j",
                      "auc", "auprc", "brier"):
                v = [r[k] for r in per_seed]
                row[k] = float(np.mean(v)); row[k + "_sd"] = float(np.std(v))
            lo, hi = bootstrap_ci(yc, pc, lambda yy, pp: all_metrics(yy, pp)["mcc"],
                                  n_boot=a.boot)
            row["mcc_ci_lo"], row["mcc_ci_hi"] = lo, hi
            lo, hi = bootstrap_ci(yc, pc, lambda yy, pp: all_metrics(yy, pp)["auc"],
                                  n_boot=a.boot)
            row["auc_ci_lo"], row["auc_ci_hi"] = lo, hi
            e, bins = ece(yc, pc)
            row["ece"] = e
            panel.append(row)
    json.dump(panel, open(os.path.join(a.out, "panel.json"), "w"), indent=2)
    print(f"  metric panel: {len(panel)} rows")

    # ---------- 2. statistical tests ----------
    tests = []
    for m in models:                       # case vs image, same backbone
        if (m, "case") in pooled and (m, "image") in pooled:
            y1, p1 = pooled[(m, "case")]; y2, p2 = pooled[(m, "image")]
            n = min(len(y1), len(y2))
            z, pv = delong_test(y1[:n], p1[:n], p2[:n])
            b, c, pm = mcnemar_test(y1[:n], p1[:n], p2[:n])
            tests.append({"kind": "case_vs_image", "a": m, "b": m,
                          "delong_z": z, "delong_p": pv,
                          "mcnemar_b": b, "mcnemar_c": c, "mcnemar_p": pm})
    for m1, m2 in itertools.combinations(models, 2):   # model vs model, case-level
        if (m1, "case") not in pooled or (m2, "case") not in pooled: continue
        y1, p1 = pooled[(m1, "case")]; y2, p2 = pooled[(m2, "case")]
        n = min(len(y1), len(y2))
        z, pv = delong_test(y1[:n], p1[:n], p2[:n])
        b, c, pm = mcnemar_test(y1[:n], p1[:n], p2[:n])
        tests.append({"kind": "model_vs_model_case", "a": m1, "b": m2,
                      "delong_z": z, "delong_p": pv,
                      "mcnemar_b": b, "mcnemar_c": c, "mcnemar_p": pm})
    # Holm-Bonferroni within the model-vs-model family
    fam = [t for t in tests if t["kind"] == "model_vs_model_case"]
    order = sorted(range(len(fam)), key=lambda i: fam[i]["delong_p"])
    k = len(fam)
    for rank, i in enumerate(order):
        fam[i]["delong_p_holm"] = min(1.0, fam[i]["delong_p"] * (k - rank))
    json.dump(tests, open(os.path.join(a.out, "tests.json"), "w"), indent=2)
    sig = sum(1 for t in fam if t.get("delong_p_holm", 1) < 0.05)
    print(f"  tests: {len(tests)} comparisons, {sig}/{k} model pairs significant after Holm")

    # ---------- 3. case-level aggregation ----------
    agg = []
    for m in models:
        for regime in ("case", "image"):
            for how in ("mean", "vote"):
                per_seed = []
                for s in a.seeds:
                    d = load_preds(a.results, a.variant, m, regime, a.head, s)
                    if d is None: continue
                    y, p, _, cases = d
                    ys, ps = aggregate_cases(y, p, cases, how)
                    per_seed.append(all_metrics(ys, ps))
                if not per_seed: continue
                row = {"model": m, "regime": regime, "agg": how,
                       "n_units": int(np.mean([r["n"] for r in per_seed]))}
                for k2 in ("acc", "err", "balanced_acc", "sens", "spec",
                           "fnr", "fpr", "mcc", "f1", "auc"):
                    row[k2] = float(np.mean([r[k2] for r in per_seed]))
                    row[k2 + "_sd"] = float(np.std([r[k2] for r in per_seed]))
                agg.append(row)
    json.dump(agg, open(os.path.join(a.out, "aggregation.json"), "w"), indent=2)
    print(f"  aggregation: {len(agg)} rows")

    # ---------- 4. calibration bins for the headline model ----------
    cal = {}
    for m in models:
        for regime in ("case", "image"):
            if (m, regime) not in pooled: continue
            y, p = pooled[(m, regime)]
            e, bins = ece(y, p)
            cal[f"{m}|{regime}"] = {"ece": e, "bins": bins}
    json.dump(cal, open(os.path.join(a.out, "calibration.json"), "w"), indent=2)

    # read-out
    print(f"\n{'model':44} {'regime':6} {'MCC':>16} {'err%':>7} {'FNR%':>7} {'FPR%':>7}")
    print("-" * 92)
    for r in sorted(panel, key=lambda r: (-r["mcc"], r["regime"])):
        print(f"{r['model']:44} {r['regime']:6} "
              f"{r['mcc']:.4f}[{r['mcc_ci_lo']:.3f},{r['mcc_ci_hi']:.3f}] "
              f"{r['err']*100:6.2f} {r['fnr']*100:6.2f} {r['fpr']*100:6.2f}")


if __name__ == "__main__":
    main()
