"""Patient-level endpoints and patient-cluster bootstrap.

Two defects in the published inference are repaired here.

First, the bootstrap intervals in the metric panel resampled pooled FIELD
predictions. Fields from one case are correlated, so resampling them
independently understates the variance of any field-level statistic. This
script resamples whole CASES, keeping every field of a drawn case together,
which is the correct cluster for this cohort.

Second, the patient-level results were reported as a mean accuracy and a mean
false-negative rate with no denominators, no confusion counts and no interval.
This script reports the patient-level confusion matrix per seed, the pooled
patient-level counts, and patient-level AUC, sensitivity, specificity and
calibration, with cluster bootstrap intervals.

Both are computed from the archived per-seed prediction files, so nothing is
refitted and the numbers describe exactly the models the paper reports.
"""
import os, sys, json, csv, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import all_metrics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "out", "results")
N_BOOT = 2000


def load_seed(model, regime, seed, variant="plain", head="linear"):
    p = os.path.join(RES, f"preds_{variant}_{model}_{regime}_{head}_s{seed}.npz")
    if not os.path.exists(p):
        return None
    d = np.load(p, allow_pickle=True)
    return (d["y"].astype(int), d["p"].astype(float),
            np.array([str(x) for x in d["cases"]]))


def by_case(y, p, cases):
    """Average field posteriors within a case; the case label is its fields'."""
    order = sorted(set(cases))
    ys = np.array([int(y[cases == c][0]) for c in order])
    ps = np.array([float(p[cases == c].mean()) for c in order])
    return ys, ps, np.array(order)


def cluster_boot(y, p, cases, fn, n_boot=N_BOOT, seed=0):
    """Percentile interval resampling CASES with replacement, keeping all the
    fields of a drawn case together. The case is the independent unit."""
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(set(cases)))
    index = {c: np.flatnonzero(cases == c) for c in uniq}
    vals = []
    for _ in range(n_boot):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([index[c] for c in draw])
        if len(set(y[sel])) < 2:
            continue
        v = fn(y[sel], p[sel])
        if np.isfinite(v):
            vals.append(v)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), len(vals)


def counts(ys, ps, thr=0.5):
    yh = (ps >= thr).astype(int)
    tp = int(((yh == 1) & (ys == 1)).sum()); tn = int(((yh == 0) & (ys == 0)).sum())
    fp = int(((yh == 1) & (ys == 0)).sum()); fn = int(((yh == 0) & (ys == 1)).sum())
    return tp, tn, fp, fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default=RES)
    a = ap.parse_args()

    models = sorted({f.split("preds_plain_")[1].rsplit("_case_linear_s", 1)[0]
                     for f in os.listdir(RES)
                     if f.startswith("preds_plain_") and "_case_linear_s" in f})

    field_rows, case_rows, per_seed = [], [], []
    for m in models:
        for regime in ("case", "image"):
            Y, P, C = [], [], []
            for s in a.seeds:
                d = load_seed(m, regime, s)
                if d is None: continue
                y, p, c = d
                Y.append(y); P.append(p)
                # a case that recurs across seeds is the same patient, so the
                # cluster label must carry the seed only for the IMAGE-level
                # rule, where the same case appears in several roles. For the
                # case-level rule a patient is in exactly one partition per
                # seed and recurs across seeds; it is still one cluster.
                C.append(np.array([f"{x}" for x in c]))
                if regime == "case":
                    ys, ps, _ = by_case(y, p, c)
                    tp, tn, fp, fn = counts(ys, ps)
                    per_seed.append({"model": m, "seed": s, "n_cases": int(len(ys)),
                                     "tp": tp, "tn": tn, "fp": fp, "fn": fn,
                                     "acc": (tp + tn) / len(ys),
                                     "sens": tp / (tp + fn) if tp + fn else float("nan"),
                                     "spec": tn / (tn + fp) if tn + fp else float("nan"),
                                     "auc": all_metrics(ys, ps)["auc"]})
            if not Y: continue
            y, p, c = np.concatenate(Y), np.concatenate(P), np.concatenate(C)

            # ---- field level, resampled by patient cluster ----
            row = {"model": m, "regime": regime, "level": "field", "n": int(len(y)),
                   "n_clusters": int(len(set(c)))}
            for name, fn in (("mcc", lambda yy, pp: all_metrics(yy, pp)["mcc"]),
                             ("auc", lambda yy, pp: all_metrics(yy, pp)["auc"]),
                             ("acc", lambda yy, pp: all_metrics(yy, pp)["acc"])):
                row[name] = fn(y, p)
                lo, hi, nb = cluster_boot(y, p, c, fn)
                row[name + "_lo"], row[name + "_hi"], row[name + "_nboot"] = lo, hi, nb
            field_rows.append(row)

            # ---- patient level ----
            ys, ps, cs = by_case(y, p, c)
            tp, tn, fp, fn = counts(ys, ps)
            crow = {"model": m, "regime": regime, "level": "case",
                    "n": int(len(ys)), "tp": tp, "tn": tn, "fp": fp, "fn": fn}
            for name, f2 in (("mcc", lambda yy, pp: all_metrics(yy, pp)["mcc"]),
                             ("auc", lambda yy, pp: all_metrics(yy, pp)["auc"]),
                             ("acc", lambda yy, pp: all_metrics(yy, pp)["acc"]),
                             ("sens", lambda yy, pp: all_metrics(yy, pp)["sens"]),
                             ("spec", lambda yy, pp: all_metrics(yy, pp)["spec"]),
                             ("brier", lambda yy, pp: all_metrics(yy, pp)["brier"])):
                crow[name] = f2(ys, ps)
                lo, hi, nb = cluster_boot(ys, ps, cs, f2)
                crow[name + "_lo"], crow[name + "_hi"] = lo, hi
            case_rows.append(crow)
        print(f"  {m:46} done", flush=True)

    json.dump({"field": field_rows, "case": case_rows, "per_seed_case": per_seed},
              open(os.path.join(a.out, "patient_inference.json"), "w"), indent=2)

    print(f"\n{'model':40} {'lvl':5} {'MCC':>7} {'cluster 95% CI':>20} {'field 95% CI (old)':>20}")
    panel = {(r["model"], r["regime"]): r for r in json.load(open(os.path.join(RES, "panel.json")))}
    for r in field_rows:
        if r["regime"] != "case": continue
        o = panel[(r["model"], "case")]
        print(f"{r['model'][:38]:40} field {r['mcc']:7.4f} "
              f"[{r['mcc_lo']:.3f}, {r['mcc_hi']:.3f}]".rjust(20) +
              f"   [{o['mcc_ci_lo']:.3f}, {o['mcc_ci_hi']:.3f}]")
    print()
    for r in case_rows:
        if r["regime"] != "case": continue
        print(f"{r['model'][:38]:40} case  acc {r['acc']*100:6.2f} "
              f"[{r['acc_lo']*100:.2f}, {r['acc_hi']*100:.2f}]  "
              f"TP {r['tp']:3} TN {r['tn']:3} FP {r['fp']:3} FN {r['fn']:3}  n={r['n']}")


if __name__ == "__main__":
    main()
