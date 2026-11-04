"""
Multiplicity-adjusted, cluster-aware contrasts (round-two addendum).

WHY THIS FILE EXISTS
--------------------
stats_round2.py reported, for the 66 encoder-pair AUC contrasts under the
case-level rule, that a group-cluster percentile interval excluded zero for 11
of them, whereas the first submission reported that a correlated DeLong test
survived Holm correction for only 2 of them. Read side by side those two
numbers look as though respecting the clustering made the analysis *less*
conservative, which would be backwards. It did not. The two numbers answer
different questions:

    correlated DeLong, unadjusted               20 of 66 at p < 0.05
    group-cluster percentile interval           11 of 66 exclude zero
    correlated DeLong, Holm-adjusted             2 of 66

Like for like - both unadjusted - the cluster-aware procedure is strictly more
conservative, and the 11 it flags are a subset of DeLong's 20; it never flags a
contrast DeLong did not. The 11-versus-2 contrast was unadjusted versus
adjusted, not clustered versus unclustered.

This file closes that gap by producing the cluster-aware analogue of the
adjusted number: a bootstrap p-value per contrast, obtained by inverting the
percentile interval, then Holm-corrected within each family. All 66 contrasts
share one set of bootstrap resamples, so the family is internally coherent and
the 12 encoder AUCs on a given resample are directly comparable.
"""
import os, json, itertools
import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "out", "results")
B = 10000


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


def auc(y, p):
    if len(set(y)) < 2:
        return float("nan")
    r = stats.rankdata(p)
    n1 = int((y == 1).sum()); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def acc(y, p):
    return float(((p >= 0.5).astype(int) == y).mean())


def boot_p(d):
    """Two-sided p by inverting the percentile interval: the smallest alpha at
    which the (1-alpha) interval still excludes zero."""
    d = np.asarray(d); n = len(d)
    lo = (np.sum(d <= 0) + 1) / (n + 1)
    hi = (np.sum(d >= 0) + 1) / (n + 1)
    return float(min(1.0, 2 * min(lo, hi)))


def holm(p):
    p = np.asarray(p, float); m = len(p)
    o = np.argsort(p); adj = np.empty(m); run = 0.0
    for i, j in enumerate(o):
        run = max(run, (m - i) * p[j])
        adj[j] = min(run, 1.0)
    return adj


def main():
    models = sorted({f.split("preds_plain_")[1].rsplit("_case_linear_s", 1)[0]
                     for f in os.listdir(RES)
                     if f.startswith("preds_plain_") and "_case_linear_s" in f})
    out = {"n_boot": B, "model_vs_model": [], "case_vs_image": [],
           "delong_reference": {}}

    # ---------- family A: 66 encoder-pair AUC contrasts, case rule ----------
    print(f"(A) {len(models)} encoders -> "
          f"{len(models)*(len(models)-1)//2} paired AUC contrasts, "
          f"shared group-cluster resamples, B = {B}")
    ref = {m: load(m, "case") for m in models}
    y0, _, f0, c0 = ref[models[0]]
    for m in models[1:]:
        ym, _, fm, _ = ref[m]
        assert np.array_equal(f0, fm) and np.array_equal(y0, ym), \
            "file order / labels differ; the pairing assumption fails"
    print("  identical file order and labels across all encoders: True")

    uniq = np.array(sorted(set(c0)))
    idx = {c: np.flatnonzero(c0 == c) for c in uniq}
    rng = np.random.default_rng(20261103)
    P = np.vstack([ref[m][1] for m in models])            # 12 x n
    curves = np.full((B, len(models)), np.nan)
    for b in range(B):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([idx[c] for c in draw])
        ys = y0[sel]
        if len(set(ys)) < 2:
            continue
        for k in range(len(models)):
            curves[b, k] = auc(ys, P[k, sel])
    keep = np.all(np.isfinite(curves), axis=1)
    curves = curves[keep]
    print(f"  usable resamples: {len(curves)} of {B}")

    obs = {m: auc(y0, ref[m][1]) for m in models}
    pairs, pv = [], []
    for i, j in itertools.combinations(range(len(models)), 2):
        d = curves[:, i] - curves[:, j]
        pairs.append((models[i], models[j],
                      obs[models[i]] - obs[models[j]],
                      float(np.percentile(d, 2.5)),
                      float(np.percentile(d, 97.5))))
        pv.append(boot_p(d))
    adj = holm(pv)
    for (a, b_, d, lo, hi), p, q in zip(pairs, pv, adj):
        out["model_vs_model"].append(
            {"a": a, "b": b_, "delta_auc": d, "ci_lo": lo, "ci_hi": hi,
             "excludes_zero": bool(lo > 0 or hi < 0),
             "boot_p": float(p), "boot_p_holm": float(q),
             "holm_significant": bool(q < 0.05)})
    nex = sum(r["excludes_zero"] for r in out["model_vs_model"])
    nho = sum(r["holm_significant"] for r in out["model_vs_model"])
    print(f"  unadjusted intervals exclude zero : {nex} of 66")
    print(f"  Holm-adjusted bootstrap p < 0.05  : {nho} of 66")

    # the DeLong family, same data, for the side-by-side the text needs
    try:
        t = json.load(open(os.path.join(RES, "tests.json")))
        dl = [r for r in t if r["kind"] == "model_vs_model_case"]
        dp = np.array([r["delong_p"] for r in dl])
        dq = holm(dp)
        out["delong_reference"] = {
            "n": len(dp), "raw_lt_05": int((dp < 0.05).sum()),
            "holm_lt_05": int((dq < 0.05).sum())}
        print(f"  reference, correlated DeLong      : "
              f"{int((dp<0.05).sum())} of 66 unadjusted, "
              f"{int((dq<0.05).sum())} of 66 Holm-adjusted")
        key = {(r["a"], r["b"]): r for r in out["model_vs_model"]}
        sub = all((not key[(r['a'], r['b'])]["excludes_zero"]) or r["delong_p"] < 0.05
                  for r in dl)
        print(f"  cluster-flagged set is a subset of DeLong's unadjusted set: {sub}")
        out["delong_reference"]["cluster_subset_of_delong"] = bool(sub)
    except FileNotFoundError:
        pass
    for r in sorted(out["model_vs_model"], key=lambda r: r["boot_p_holm"])[:6]:
        print(f"    {r['a'][:34]:36}{r['b'][:34]:36}"
              f"{r['delta_auc']:+.4f} [{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}]"
              f"  p_holm={r['boot_p_holm']:.3g}")

    # ---------- family B: 12 case-vs-image contrasts, per encoder ----------
    print(f"\n(B) {len(models)} case-vs-image accuracy contrasts, per encoder, "
          "independent group-cluster resamples (unpaired), Holm within family")
    pv2, rows = [], []
    for m in models:
        ya, pa, _, ca = load(m, "case")
        yb, pb, _, cb = load(m, "image")
        da = []
        # sign convention matches stats_round2.py: image-level minus case-level,
        # so a positive number means the looser rule reports the higher score
        for arm, (y, p, c, seed) in enumerate(((yb, pb, cb, 22), (ya, pa, ca, 11))):
            u = np.array(sorted(set(c)))
            ix = {k: np.flatnonzero(c == k) for k in u}
            r2 = np.random.default_rng(seed)
            vals = []
            for _ in range(B):
                dr = r2.choice(u, size=len(u), replace=True)
                s = np.concatenate([ix[k] for k in dr])
                vals.append(acc(y[s], p[s]) * 100)
            da.append(np.array(vals))
        d = da[0] - da[1]
        pv2.append(boot_p(d))
        rows.append((m, (acc(yb, pb) - acc(ya, pa)) * 100,
                     float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))))
    adj2 = holm(pv2)
    for (m, d, lo, hi), p, q in zip(rows, pv2, adj2):
        out["case_vs_image"].append(
            {"model": m, "delta_acc_pp": d, "ci_lo": lo, "ci_hi": hi,
             "excludes_zero": bool(lo > 0 or hi < 0), "boot_p": float(p),
             "boot_p_holm": float(q), "holm_significant": bool(q < 0.05),
             "paired": False})
        print(f"  {m[:40]:42}{d:>+7.2f} pp  [{lo:>+6.2f},{hi:>+6.2f}]"
              f"  p={p:.3g}  p_holm={q:.3g}")
    print(f"  unadjusted intervals exclude zero : "
          f"{sum(r['excludes_zero'] for r in out['case_vs_image'])} of {len(models)}")
    print(f"  Holm-adjusted bootstrap p < 0.05  : "
          f"{sum(r['holm_significant'] for r in out['case_vs_image'])} of {len(models)}")

    json.dump(out, open(os.path.join(RES, "stats_round2b.json"), "w"), indent=2)
    print("\nwrote out/results/stats_round2b.json")


if __name__ == "__main__":
    main()
