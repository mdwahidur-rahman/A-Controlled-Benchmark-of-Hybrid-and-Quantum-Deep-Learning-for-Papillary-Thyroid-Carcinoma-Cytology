"""
Additional fusion controls, and repeated permutations.

WHY THIS FILE EXISTS
--------------------
The first submission reported one negative control for the second branch: an
unconditional shuffle, later strengthened to a different-group derangement.
Both destroy branch B's correspondence with branch A, but they also destroy
branch B's association with the target label, so a configuration that survives
them has not been shown to carry no cross-branch information -- it has been
shown to lose nothing when branch B is replaced by noise. Those are different
statements.

Two things are added here.

1. A LABEL-PRESERVING, DIFFERENT-GROUP PERMUTATION. Every field's branch-B
   vector is replaced by the branch-B vector of a field with the SAME label
   from a DIFFERENT recovered group, within the same partition. Branch B then
   still carries the class, and only its field-level correspondence with
   branch A is gone. The difference between this control and the intact
   concatenation isolates what the correspondence contributes, which the
   unconditional shuffle cannot. This control uses the labels of the partition
   it permutes within; for the training partition those labels are already
   available to the fitting procedure, and for the evaluation partition they
   are not available to the model but are used to build the permutation, so
   this arm is a diagnostic and not a performance estimate. It is reported as
   such.

2. REPEATED PERMUTATIONS PER SPLIT. Each permutation arm is redrawn R times
   within each (regime, seed), so that the variability contributed by the
   permutation can be separated from the variability contributed by the split.

The controls are also applied to the nonlinear projected head, which the first
submission left with no control at all.
"""
import os, sys, json, csv, argparse, warnings
import numpy as np
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hybrid import metrics, load, head_fit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def derange_by_case(cs, gen):
    """Permutation sending every row to a row of a different group."""
    cs = np.asarray(cs)
    n = len(cs)
    for _ in range(200):
        perm = gen.permutation(n)
        if not np.any(cs[perm] == cs):
            return perm
    perm = gen.permutation(n)
    bad = np.flatnonzero(cs[perm] == cs)
    for i in bad:
        for j in range(n):
            if cs[perm[j]] != cs[i] and cs[perm[i]] != cs[j]:
                perm[i], perm[j] = perm[j], perm[i]
                break
    return perm


def label_preserving(cs, ys, gen):
    """Permutation sending every row to a row of a DIFFERENT group that carries
    the SAME label. Branch B keeps the class and loses the correspondence."""
    cs = np.asarray(cs); ys = np.asarray(ys)
    perm = np.arange(len(cs))
    for lab in np.unique(ys):
        pos = np.flatnonzero(ys == lab)
        sub = derange_by_case(cs[pos], gen)
        perm[pos] = pos[sub]
    stuck = int(np.sum(cs[perm] == cs))
    return perm, stuck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=os.path.join(ROOT, "out", "features"))
    ap.add_argument("--splits", default=os.path.join(ROOT, "out", "splits"))
    ap.add_argument("--variant", default="plain")
    ap.add_argument("--branch-a", default="convnextv2_tiny.fcmae_ft_in22k_in1k")
    ap.add_argument("--branch-b", default="swin_tiny_patch4_window7_224.ms_in1k")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--reps-linear", type=int, default=5)
    ap.add_argument("--reps-proj", type=int, default=3)
    ap.add_argument("--regimes", nargs="+", default=["case", "image"])
    ap.add_argument("--proj-regimes", nargs="+", default=["case"])
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "results"))
    a = ap.parse_args()

    rows = list(csv.DictReader(open(os.path.join(ROOT, "manifest.csv"))))
    label = {r["file"]: int(r["label"]) for r in rows}
    case_of = {r["file"]: r["case"] for r in rows}
    Fa, files = load(a.features, a.variant, a.branch_a)
    Fb, files_b = load(a.features, a.variant, a.branch_b)
    assert files == files_b, "the two branches are not in the same row order"
    idx = {f: i for i, f in enumerate(files)}
    y_all = np.array([label[f] for f in files])

    results, diag = [], []
    for regime in a.regimes:
        for seed in a.seeds:
            sp = json.load(open(os.path.join(
                a.splits, f"split_{regime}_seed{seed}.json")))
            assign = sp["assign"]
            sel = {s: np.array([idx[f] for f in files if assign[f] == s])
                   for s in ("train", "val", "test")}
            cs = {s: [case_of[files[i]] for i in sel[s]] for s in sel}
            ys = {s: y_all[sel[s]] for s in sel}

            def build(kind, gen):
                out, stuck = {}, 0
                for s in ("train", "val", "test"):
                    xa, xb = Fa[sel[s]], Fb[sel[s]]
                    if kind == "intact":
                        pass
                    elif kind == "shuffled":
                        xb = xb[gen.permutation(len(xb))]
                    elif kind == "deranged":
                        xb = xb[derange_by_case(cs[s], gen)]
                    elif kind == "labelpreserved":
                        pm, st = label_preserving(cs[s], ys[s], gen)
                        xb = xb[pm]; stuck += st
                    out[s] = np.hstack([xa, xb])
                sc = StandardScaler().fit(out["train"])
                return {s: sc.transform(out[s]) for s in out}, stuck

            plan = [("concat", None, k, a.reps_linear)
                    for k in ("shuffled", "deranged", "labelpreserved")]
            if regime in a.proj_regimes:
                plan += [("proj", 256, k, a.reps_proj)
                         for k in ("intact", "shuffled", "deranged",
                                   "labelpreserved")]
            for family, hidden, kind, reps in plan:
                n_rep = 1 if kind == "intact" else reps
                for rep in range(n_rep):
                    gen = np.random.default_rng(
                        hash((regime, seed, family, kind, rep)) % 2**31)
                    X, stuck = build(kind, gen)
                    clf = head_fit(X["train"], ys["train"], X["val"], ys["val"],
                                   seed, hidden)
                    p = clf.predict_proba(X["test"])[:, 1]
                    m = metrics(ys["test"], p)
                    m.update(config=f"{family}_{kind}", family=family, kind=kind,
                             hidden=hidden, regime=regime, seed=seed, rep=rep,
                             same_group_pairs_remaining=stuck)
                    results.append(m)
                    print(f"  {regime:5} seed {seed} {family}_{kind:15} "
                          f"rep {rep}  acc {m['acc']*100:6.2f}  auc {m['auc']:.4f}"
                          f"{'  [same-group pairs left: %d]' % stuck if stuck else ''}",
                          flush=True)
            json.dump(results, open(os.path.join(a.out, "hybrid_v2.json"), "w"),
                      indent=2)

    print("\n%-28s %-6s %7s %7s %7s" % ("arm", "regime", "mean", "sd_split",
                                        "sd_perm"))
    print("-" * 62)
    for cfg in sorted({r["config"] for r in results}):
        for regime in a.regimes:
            R = [r for r in results if r["config"] == cfg and r["regime"] == regime]
            if not R:
                continue
            per_seed = {}
            for r in R:
                per_seed.setdefault(r["seed"], []).append(r["acc"] * 100)
            means = [np.mean(v) for v in per_seed.values()]
            within = [np.std(v) for v in per_seed.values() if len(v) > 1]
            print("%-28s %-6s %7.2f %7.2f %7.2f"
                  % (cfg, regime, np.mean(means), np.std(means),
                     np.mean(within) if within else 0.0))


if __name__ == "__main__":
    main()
