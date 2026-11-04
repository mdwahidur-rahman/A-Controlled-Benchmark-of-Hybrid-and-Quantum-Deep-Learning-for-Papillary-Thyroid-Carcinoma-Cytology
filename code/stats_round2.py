"""
Corrected statistical analysis of the splitting-rule contrast.

WHY THIS FILE EXISTS
--------------------
analysis.py tested the case-level against the image-level rule like this:

    n = min(len(y1), len(y2))
    delong_test(y1[:n], p1[:n], p2[:n])
    mcnemar_test(y1[:n], p1[:n], p2[:n])

That truncates two pooled prediction vectors to the shorter length and pairs
them by row position. The two vectors do not contain the same images: under the
case-level rule the pooled test partition holds 1,354 field predictions over 936
distinct images, under the image-level rule 1,330 over 916, and the two orders
are unrelated. Measured directly, row i of the truncated pair refers to the same
image in 1 of 1,330 rows. The label vector passed to both arms was y1, taken
from the case-level split, so for the image-level arm the labels did not even
correspond to the predictions.

Correlated DeLong and McNemar both require paired observations on the same
evaluation units. Neither requirement is met, so the reported
"Holm-adjusted p < 1e-4 for every encoder" is withdrawn. Multiplicity correction
cannot repair an invalid underlying test.

WHAT REPLACES IT
----------------
Two analyses, each valid for the design that was actually run.

(1) ENCODER AS THE UNIT. The twelve encoders were each evaluated under both
    rules. The per-encoder difference is a paired observation at the level of
    the encoder, which is the level at which the two rules really are matched:
    same cohort, same proportions, same seeds, same head. An exact two-sided
    sign test and a Wilcoxon signed-rank test on those twelve differences
    answer "does the rule move the number in a consistent direction", which is
    the question the paper actually asks. This makes no claim about any single
    encoder.

(2) PER ENCODER, GROUP-CLUSTER BOOTSTRAP, REPORTED AS UNPAIRED. For each
    encoder the two regimes are resampled independently, each by drawing whole
    recovered groups with all their fields and all their appearances across
    seeds. The difference of the two resampled means gives a percentile
    interval for that encoder. Because the two resamples are independent this
    is a two-sample interval, not a paired one, and it is labelled as such.

The model-vs-model family is different and is NOT withdrawn: both arms there
are the case-level rule with the same seeds, so the file order and labels are
identical and the pairing is genuine. That is verified in code below. DeLong on
pooled fields is still anticonservative under clustering, so a group-cluster
bootstrap is reported alongside it.
"""
import os, sys, json, itertools
import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "out", "results")
N_BOOT = 4000
RNG = np.random.default_rng(0)


def load(model, regime, seeds=range(5)):
    Y, P, F, Cs = [], [], [], []
    for s in seeds:
        d = np.load(os.path.join(RES, f"preds_plain_{model}_{regime}_linear_s{s}.npz"),
                    allow_pickle=True)
        Y.append(d["y"].astype(int)); P.append(d["p"].astype(float))
        F.append(np.array([str(x) for x in d["files"]]))
        Cs.append(np.array([str(x) for x in d["cases"]]))
    return (np.concatenate(Y), np.concatenate(P),
            np.concatenate(F), np.concatenate(Cs))


def acc(y, p):
    return float(((p >= 0.5).astype(int) == y).mean())


def auc(y, p):
    y = np.asarray(y); p = np.asarray(p)
    if len(set(y)) < 2:
        return float("nan")
    r = stats.rankdata(p)
    n1 = int((y == 1).sum()); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def cluster_boot(y, p, cases, fn, n_boot=N_BOOT, seed=0):
    """Resample whole groups, keeping every field and every repeat of a drawn
    group together. The group is the independent unit."""
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(set(cases)))
    idx = {c: np.flatnonzero(cases == c) for c in uniq}
    out = []
    for _ in range(n_boot):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([idx[c] for c in draw])
        if len(set(y[sel])) < 2:
            continue
        v = fn(y[sel], p[sel])
        if np.isfinite(v):
            out.append(v)
    return np.array(out)


def main():
    models = sorted({f.split("preds_plain_")[1].rsplit("_case_linear_s", 1)[0]
                     for f in os.listdir(RES)
                     if f.startswith("preds_plain_") and "_case_linear_s" in f})
    report = {"withdrawn": {}, "encoder_level": {}, "per_encoder": [],
              "model_vs_model": []}

    # ---- the evidence that the old pairing was invalid --------------------
    m0 = models[0]
    yc, pc, fc, cc = load(m0, "case"); yi, pi, fi, ci = load(m0, "image")
    n = min(len(fc), len(fi))
    report["withdrawn"] = {
        "n_case_fields": int(len(fc)), "n_image_fields": int(len(fi)),
        "distinct_images_case": int(len(set(fc))),
        "distinct_images_image": int(len(set(fi))),
        "truncation_length": int(n),
        "rows_referring_to_same_image": int((fc[:n] == fi[:n]).sum()),
        "note": ("row-position pairing after truncation; the two vectors hold "
                 "different images, so the paired tests were not valid"),
    }
    print("Old pairing, measured directly:")
    print(f"  case fields {len(fc)} over {len(set(fc))} images; "
          f"image fields {len(fi)} over {len(set(fi))} images")
    print(f"  truncating to {n} rows pairs the SAME image in "
          f"{(fc[:n]==fi[:n]).sum()} of {n} rows\n")

    # ---- (1) encoder as the unit ------------------------------------------
    dacc, dauc, names = [], [], []
    for m in models:
        yc, pc, fc, cc = load(m, "case"); yi, pi, fi, ci = load(m, "image")
        dacc.append((acc(yi, pi) - acc(yc, pc)) * 100)
        dauc.append((auc(yi, pi) - auc(yc, pc)) * 100)
        names.append(m)
    dacc = np.array(dacc); dauc = np.array(dauc)

    npos = int((dacc > 0).sum()); k = len(dacc)
    sign_p = float(stats.binomtest(npos, k, 0.5, alternative="two-sided").pvalue)
    w = stats.wilcoxon(dacc, alternative="two-sided", zero_method="wilcox")
    bs = np.array([RNG.choice(dacc, size=k, replace=True).mean() for _ in range(N_BOOT)])
    report["encoder_level"] = {
        "n_encoders": k, "n_positive": npos,
        "mean_delta_acc_pp": float(dacc.mean()),
        "ci_lo": float(np.percentile(bs, 2.5)), "ci_hi": float(np.percentile(bs, 97.5)),
        "sign_test_p": sign_p,
        "wilcoxon_stat": float(w.statistic), "wilcoxon_p": float(w.pvalue),
        "mean_delta_auc_pp": float(dauc.mean()),
        "per_encoder_delta_acc": {n_: float(v) for n_, v in zip(names, dacc)},
    }
    print("(1) Encoder as the unit, 12 paired per-encoder differences:")
    print(f"  mean delta accuracy  {dacc.mean():+.4f} pp  "
          f"[{np.percentile(bs,2.5):+.4f}, {np.percentile(bs,97.5):+.4f}]")
    print(f"  positive in {npos}/{k};  exact two-sided sign test p = {sign_p:.6f}")
    print(f"  Wilcoxon signed-rank  W = {w.statistic:.1f},  p = {w.pvalue:.6f}")
    print(f"  mean delta AUC x100  {dauc.mean():+.4f}\n")

    # ---- (2) per encoder, independent group-cluster bootstrap -------------
    print("(2) Per encoder, independent group-cluster bootstrap "
          "(UNPAIRED, two-sample):")
    print(f"  {'encoder':42}{'d acc pp':>10}{'95% CI':>22}{'excl. 0':>9}")
    for m in models:
        yc, pc, fc, cc = load(m, "case"); yi, pi, fi, ci = load(m, "image")
        bc = cluster_boot(yc, pc, cc, acc, seed=1)
        bi = cluster_boot(yi, pi, ci, acc, seed=2)
        n_ = min(len(bc), len(bi))
        d = (bi[:n_] - bc[:n_]) * 100
        lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
        pt = (acc(yi, pi) - acc(yc, pc)) * 100
        ex = "yes" if lo > 0 or hi < 0 else "no"
        report["per_encoder"].append(
            {"model": m, "delta_acc_pp": pt, "ci_lo": lo, "ci_hi": hi,
             "excludes_zero": ex == "yes", "paired": False})
        print(f"  {m[:40]:42}{pt:>10.2f}  [{lo:>7.2f}, {hi:>7.2f}]{ex:>9}")
    nex = sum(r["excludes_zero"] for r in report["per_encoder"])
    print(f"  interval excludes zero for {nex} of {len(models)} encoders\n")

    # ---- (3) model vs model: verify the pairing IS valid, then cluster ----
    print("(3) Model-vs-model under the case rule: pairing check and "
          "cluster-aware contrast")
    ya, pa, fa, ca = load(models[0], "case")
    ok = True
    for m in models[1:]:
        yb, pb, fb, cb = load(m, "case")
        if not (np.array_equal(fa, fb) and np.array_equal(ya, yb)):
            ok = False
    print(f"  identical file order and labels across all encoders: {ok}")
    for m1, m2 in itertools.combinations(models, 2):
        y1, p1, f1, c1 = load(m1, "case"); y2, p2, f2, c2 = load(m2, "case")
        assert np.array_equal(f1, f2)
        rng = np.random.default_rng(3)
        uniq = np.array(sorted(set(c1)))
        idx = {c: np.flatnonzero(c1 == c) for c in uniq}
        diffs = []
        for _ in range(1500):
            draw = rng.choice(uniq, size=len(uniq), replace=True)
            sel = np.concatenate([idx[c] for c in draw])
            if len(set(y1[sel])) < 2:
                continue
            diffs.append(auc(y1[sel], p1[sel]) - auc(y1[sel], p2[sel]))
        diffs = np.array(diffs)
        lo, hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
        report["model_vs_model"].append(
            {"a": m1, "b": m2, "delta_auc": float(auc(y1, p1) - auc(y1, p2)),
             "ci_lo": lo, "ci_hi": hi, "excludes_zero": bool(lo > 0 or hi < 0),
             "paired": True})
    nex = sum(r["excludes_zero"] for r in report["model_vs_model"])
    print(f"  group-cluster paired AUC intervals exclude zero for "
          f"{nex} of {len(report['model_vs_model'])} contrasts")

    json.dump(report, open(os.path.join(RES, "stats_round2.json"), "w"), indent=2)
    print("\nwrote out/results/stats_round2.json")


if __name__ == "__main__":
    main()
