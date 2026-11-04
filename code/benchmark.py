"""
Classifier benchmark over cached frozen-backbone features.

The central comparison is not "which backbone wins". It is the SAME backbone,
the SAME features and the SAME head evaluated under two split regimes that
differ only in whether patient grouping is respected:

  case  : correct. no patient appears in more than one subset.
  image : the common shortcut. tiles from one patient's slide land in both
          train and test.

The difference between the two is the leakage inflation, measured rather than
asserted. Everything else is held fixed so that difference is attributable.

Heads: L2 logistic regression (linear probe, the standard frozen-feature
protocol) and a one-hidden-layer MLP. Regularization is selected on the
validation split only; the test split is touched once, at the end.
"""
import os, json, csv, argparse, warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]


def metrics(y, p, thr=0.5):
    yh = (p >= thr).astype(int)
    tp = int(((yh == 1) & (y == 1)).sum()); tn = int(((yh == 0) & (y == 0)).sum())
    fp = int(((yh == 1) & (y == 0)).sum()); fn = int(((yh == 0) & (y == 1)).sum())
    sens = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * prec * sens / (prec + sens) if prec + sens else 0.0
    return {"acc": (tp + tn) / len(y), "sens": sens, "spec": spec, "prec": prec,
            "f1": f1, "auc": roc_auc_score(y, p) if len(set(y)) > 1 else float("nan"),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def fit_head(kind, Xtr, ytr, Xva, yva, seed):
    """Select hyperparameter on validation only, then refit on train+val."""
    best, best_auc = None, -1.0
    for C in C_GRID:
        if kind == "linear":
            clf = LogisticRegression(C=C, max_iter=2000, random_state=seed)
        else:
            clf = MLPClassifier(hidden_layer_sizes=(256,), alpha=1.0 / max(C, 1e-6),
                                max_iter=400, random_state=seed, early_stopping=False)
        clf.fit(Xtr, ytr)
        a = roc_auc_score(yva, clf.predict_proba(Xva)[:, 1]) if len(set(yva)) > 1 else 0.0
        if a > best_auc:
            best_auc, best = a, C
    X = np.vstack([Xtr, Xva]); y = np.concatenate([ytr, yva])
    if kind == "linear":
        clf = LogisticRegression(C=best, max_iter=2000, random_state=seed)
    else:
        clf = MLPClassifier(hidden_layer_sizes=(256,), alpha=1.0 / max(best, 1e-6),
                            max_iter=400, random_state=seed)
    clf.fit(X, y)
    return clf, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=os.path.join(ROOT, "out", "features"))
    ap.add_argument("--splits", default=os.path.join(ROOT, "out", "splits"))
    ap.add_argument("--manifest", default=os.path.join(ROOT, "manifest.csv"))
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "results"))
    ap.add_argument("--variant", default="plain")
    ap.add_argument("--heads", nargs="+", default=["linear"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--models", nargs="+", default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rows = list(csv.DictReader(open(a.manifest)))
    label = {r["file"]: int(r["label"]) for r in rows}
    case = {r["file"]: r["case"] for r in rows}

    avail = sorted(f for f in os.listdir(a.features) if f.startswith(a.variant + "__"))
    models = a.models or [f.split("__", 1)[1][:-4] for f in avail]

    results = []
    for m in models:
        fp = os.path.join(a.features, f"{a.variant}__{m}.npz")
        if not os.path.exists(fp):
            print(f"  missing {m}"); continue
        d = np.load(fp, allow_pickle=True)
        F, files = d["features"], [str(x) for x in d["files"]]
        idx = {f: i for i, f in enumerate(files)}
        y_all = np.array([label[f] for f in files])

        for regime in ("case", "image"):
            for seed in a.seeds:
                sp = json.load(open(os.path.join(a.splits, f"split_{regime}_seed{seed}.json")))
                assign = sp["assign"]
                sel = {s: np.array([idx[f] for f in files if assign[f] == s]) for s in ("train", "val", "test")}
                sc = StandardScaler().fit(F[sel["train"]])
                Xtr, Xva, Xte = (sc.transform(F[sel[s]]) for s in ("train", "val", "test"))
                ytr, yva, yte = (y_all[sel[s]] for s in ("train", "val", "test"))
                for head in a.heads:
                    clf, C = fit_head(head, Xtr, ytr, Xva, yva, seed)
                    p = clf.predict_proba(Xte)[:, 1]
                    mt = metrics(yte, p)
                    mt.update(model=m, variant=a.variant, regime=regime, seed=seed,
                              head=head, C=C, n_test=len(yte), dim=int(F.shape[1]))
                    results.append(mt)
                    np.savez_compressed(
                        os.path.join(a.out, f"preds_{a.variant}_{m}_{regime}_{head}_s{seed}.npz"),
                        y=yte, p=p, files=np.array([files[i] for i in sel["test"]]),
                        cases=np.array([case[files[i]] for i in sel["test"]]))
        done = [r for r in results if r["model"] == m and r["head"] == a.heads[0]]
        for regime in ("case", "image"):
            v = [r["acc"] for r in done if r["regime"] == regime]
            if v: print(f"  {m:46} {regime:5} acc={np.mean(v)*100:5.2f} +/- {np.std(v)*100:4.2f}")

    with open(os.path.join(a.out, f"results_{a.variant}.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {len(results)} rows -> results_{a.variant}.json")


if __name__ == "__main__":
    main()
