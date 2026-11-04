"""
Hybrid fusion over cached features, with the controls the source paper omitted.

The paper reports ConvNeXtV2 82.84 -> Swin 89.62 -> direct concat 93.56 ->
"proposed fusion" 96.19, and attributes the last step to normalization plus a
learned projection. That ladder has no capacity control: the hybrid holds
roughly the sum of both branches' parameters, so part of any gain is simply
more parameters.

This module reproduces their ladder AND adds the missing controls:

  single_a / single_b    each branch alone
  concat                 direct concatenation  (their row 3)
  proj                   concat + normalize + learned projection (their row 4)
  single_a_matched       branch A alone, head widened to the concat parameter
                         count. If `concat` does not beat this, the gain was
                         capacity, not complementarity.
  concat_shuffled        concat where branch B's rows are permuted across
                         samples, destroying correspondence while preserving
                         B's marginal distribution and the parameter count.
                         This is the sharpest control: whatever beats it is
                         genuine cross-branch information.

Everything runs under both split regimes so the leakage question stays live.
"""
import os, json, csv, argparse, warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C_GRID = [0.01, 0.1, 1.0, 10.0]


def metrics(y, p):
    yh = (p >= 0.5).astype(int)
    tp = int(((yh == 1) & (y == 1)).sum()); tn = int(((yh == 0) & (y == 0)).sum())
    fp = int(((yh == 1) & (y == 0)).sum()); fn = int(((yh == 0) & (y == 1)).sum())
    sens = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    return {"acc": (tp + tn) / len(y), "sens": sens, "spec": spec, "prec": prec,
            "f1": 2 * prec * sens / (prec + sens) if prec + sens else 0.0,
            "auc": roc_auc_score(y, p), "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def load(features, variant, name):
    d = np.load(os.path.join(features, f"{variant}__{name}.npz"), allow_pickle=True)
    return d["features"], [str(x) for x in d["files"]]


def head_fit(X, y, Xv, yv, seed, hidden=None):
    best, best_auc = None, -1
    for C in C_GRID:
        clf = (LogisticRegression(C=C, max_iter=3000, random_state=seed) if hidden is None
               else MLPClassifier(hidden_layer_sizes=(hidden,), alpha=1.0 / C,
                                  max_iter=500, random_state=seed))
        clf.fit(X, y)
        a = roc_auc_score(yv, clf.predict_proba(Xv)[:, 1])
        if a > best_auc: best_auc, best = a, C
    Xf, yf = np.vstack([X, Xv]), np.concatenate([y, yv])
    clf = (LogisticRegression(C=best, max_iter=3000, random_state=seed) if hidden is None
           else MLPClassifier(hidden_layer_sizes=(hidden,), alpha=1.0 / best,
                              max_iter=500, random_state=seed))
    clf.fit(Xf, yf)
    return clf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=os.path.join(ROOT, "out", "features"))
    ap.add_argument("--splits", default=os.path.join(ROOT, "out", "splits"))
    ap.add_argument("--manifest", default=os.path.join(ROOT, "manifest.csv"))
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "results"))
    ap.add_argument("--variant", default="plain")
    ap.add_argument("--branch-a", default="convnextv2_tiny.fcmae_ft_in22k_in1k")
    ap.add_argument("--branch-b", default="swin_tiny_patch4_window7_224.ms_in1k")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rows = list(csv.DictReader(open(a.manifest)))
    label = {r["file"]: int(r["label"]) for r in rows}
    case_of = {r["file"]: r["case"] for r in rows}
    Fa, files = load(a.features, a.variant, a.branch_a)
    Fb, files_b = load(a.features, a.variant, a.branch_b)
    assert files == files_b, "feature file order mismatch between branches"
    idx = {f: i for i, f in enumerate(files)}
    y_all = np.array([label[f] for f in files])

    results = []
    for regime in ("case", "image"):
        for seed in a.seeds:
            sp = json.load(open(os.path.join(a.splits, f"split_{regime}_seed{seed}.json")))
            assign = sp["assign"]
            sel = {s: np.array([idx[f] for f in files if assign[f] == s])
                   for s in ("train", "val", "test")}
            rng = np.random.default_rng(seed + 777)

            def derange_by_case(rows_cases, gen):
                """Permutation that sends every field to a field of a DIFFERENT
                case. A plain permutation can leave a field paired with another
                field of its own patient, which preserves exactly the
                correspondence the control is supposed to destroy; on this
                cohort roughly one field in four hundred would survive that way
                per draw, and the point of a control is that it does not."""
                cs = np.asarray(rows_cases)
                n = len(cs)
                for _ in range(200):
                    perm = gen.permutation(n)
                    if not np.any(cs[perm] == cs):
                        return perm
                # fall back to a repair pass if rejection sampling is unlucky
                perm = gen.permutation(n)
                bad = np.flatnonzero(cs[perm] == cs)
                for i in bad:
                    for j in range(n):
                        if cs[perm[j]] != cs[i] and cs[perm[i]] != cs[j]:
                            perm[i], perm[j] = perm[j], perm[i]
                            break
                return perm

            def prep(A, B=None, shuffle_b=False, derange_b=False):
                out = {}
                for s in ("train", "val", "test"):
                    xa = A[sel[s]]
                    if B is None:
                        out[s] = xa
                    else:
                        xb = B[sel[s]]
                        if derange_b:
                            cs = [case_of[files[i]] for i in sel[s]]
                            xb = xb[derange_by_case(cs, rng)]
                        elif shuffle_b:
                            xb = xb[rng.permutation(len(xb))]
                        out[s] = np.hstack([xa, xb])
                sc = StandardScaler().fit(out["train"])
                return {s: sc.transform(out[s]) for s in out}

            configs = {
                "single_a":         (prep(Fa), None),
                "single_b":         (prep(Fb), None),
                "concat":           (prep(Fa, Fb), None),
                "concat_shuffled":  (prep(Fa, Fb, shuffle_b=True), None),
                "proj":             (prep(Fa, Fb), 256),
                "single_a_matched": (prep(Fa), 256),
                # 768 -> 256 is 197,121 parameters against the fused head's
                # 1536 -> 256 = 393,729: that control matches the WIDTH, not
                # the parameter count. 768 -> 512 = 394,241 matches the count
                # to within 0.13%, so both controls are reported and named for
                # what they match.
                "single_a_pmatched": (prep(Fa), 512),
                # a plain row shuffle can re-pair a field with another field of
                # its own case; this one cannot
                "concat_deranged":  (prep(Fa, Fb, derange_b=True), None),
            }
            for cfgname, (X, hidden) in configs.items():
                clf = head_fit(X["train"], y_all[sel["train"]],
                               X["val"], y_all[sel["val"]], seed, hidden)
                p = clf.predict_proba(X["test"])[:, 1]
                m = metrics(y_all[sel["test"]], p)
                m.update(config=cfgname, regime=regime, seed=seed, variant=a.variant,
                         dim=int(X["train"].shape[1]))
                results.append(m)
                np.savez_compressed(
                    os.path.join(a.out, f"hyb_{a.variant}_{cfgname}_{regime}_s{seed}.npz"),
                    y=y_all[sel["test"]], p=p,
                    files=np.array([files[i] for i in sel["test"]]))

    with open(os.path.join(a.out, f"hybrid_{a.variant}.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    print(f"{'config':20} {'case acc':>14} {'image acc':>14} {'case auc':>10}")
    print("-" * 62)
    for c in ["single_a", "single_b", "single_a_matched", "single_a_pmatched",
              "concat_shuffled", "concat_deranged", "concat", "proj"]:
        cc = [r["acc"] for r in results if r["config"] == c and r["regime"] == "case"]
        ii = [r["acc"] for r in results if r["config"] == c and r["regime"] == "image"]
        au = [r["auc"] for r in results if r["config"] == c and r["regime"] == "case"]
        print(f"{c:20} {np.mean(cc)*100:7.2f}+/-{np.std(cc)*100:4.2f} "
              f"{np.mean(ii)*100:7.2f}+/-{np.std(ii)*100:4.2f} {np.mean(au):10.4f}")


if __name__ == "__main__":
    main()
