"""Decision-level (accuracy) analogue of the 66 architecture contrasts, with
the same shared group-cluster resamples and the same Holm correction, so that
the McNemar family of the first submission has a cluster-aware replacement."""
import os, json, itertools
import numpy as np
from scipy import stats
sys_path = os.path.dirname(os.path.abspath(__file__))
import sys; sys.path.insert(0, sys_path)
from stats_round2b import load, boot_p, holm, B

ROOT = os.path.dirname(sys_path)
RES = os.path.join(ROOT, "out", "results")

models = sorted({f.split("preds_plain_")[1].rsplit("_case_linear_s", 1)[0]
                 for f in os.listdir(RES)
                 if f.startswith("preds_plain_") and "_case_linear_s" in f})
ref = {m: load(m, "case") for m in models}
y0, _, f0, c0 = ref[models[0]]
P = np.vstack([ref[m][1] for m in models])
D = (P >= 0.5).astype(int)                      # 12 x n decisions
uniq = np.array(sorted(set(c0)))
idx = {c: np.flatnonzero(c0 == c) for c in uniq}
rng = np.random.default_rng(20261103)
acc = np.empty((B, len(models)))
for b in range(B):
    draw = rng.choice(uniq, size=len(uniq), replace=True)
    sel = np.concatenate([idx[c] for c in draw])
    acc[b] = (D[:, sel] == y0[sel]).mean(axis=1) * 100
obs = (D == y0).mean(axis=1) * 100

# McNemar discordant counts on the pooled fields, reported as the reviewer asked
rows, pv = [], []
for i, j in itertools.combinations(range(len(models)), 2):
    d = acc[:, i] - acc[:, j]
    ci = (D[i] == y0) & (D[j] != y0)
    cj = (D[j] == y0) & (D[i] != y0)
    bn = int(ci.sum()); cn = int(cj.sum())
    mc = stats.binomtest(bn, bn + cn, 0.5).pvalue if bn + cn else 1.0
    rows.append({"a": models[i], "b": models[j],
                 "delta_acc_pp": float(obs[i] - obs[j]),
                 "ci_lo": float(np.percentile(d, 2.5)),
                 "ci_hi": float(np.percentile(d, 97.5)),
                 "mcnemar_b": bn, "mcnemar_c": cn, "mcnemar_p_exact": float(mc)})
    pv.append(boot_p(d))
adj = holm(pv); mcadj = holm([r["mcnemar_p_exact"] for r in rows])
for r, p, q, mq in zip(rows, pv, adj, mcadj):
    r.update(boot_p=float(p), boot_p_holm=float(q),
             holm_significant=bool(q < 0.05),
             mcnemar_p_holm=float(mq), mcnemar_holm_significant=bool(mq < 0.05))
out = {"n_boot": B, "n_contrasts": len(rows),
       "cluster_unadjusted_excl_zero": sum(r["ci_lo"] > 0 or r["ci_hi"] < 0 for r in rows),
       "cluster_holm_significant": sum(r["holm_significant"] for r in rows),
       "mcnemar_unadjusted": sum(r["mcnemar_p_exact"] < 0.05 for r in rows),
       "mcnemar_holm": sum(r["mcnemar_holm_significant"] for r in rows),
       "max_abs_delta_acc_pp": float(max(abs(r["delta_acc_pp"]) for r in rows)),
       "contrasts": rows}
json.dump(out, open(os.path.join(RES, "stats_round2c.json"), "w"), indent=2)
for k, v in out.items():
    if k != "contrasts":
        print(f"  {k}: {v}")
