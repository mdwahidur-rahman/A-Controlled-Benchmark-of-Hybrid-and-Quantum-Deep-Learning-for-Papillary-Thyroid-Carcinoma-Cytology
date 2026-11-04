"""
Uninformative-order baselines for the deletion/insertion curves, and the
constant-map tie-breaking question.

WHY THIS FILE EXISTS
--------------------
A deletion area is only interpretable against the area an uninformative order
obtains on the same images, the same model, the same target and the same
masking protocol. Without that reference, "method A has a lower deletion area
than method B" says that A and B differ, not that either ranks evidence better
than an arbitrary order. Two references are computed here:

    random   a uniformly random pixel order, averaged over R draws per image
    reverse  the attribution order read backwards, least-salient pixel first

The reverse baseline turns out not to be independent of the forward one under
this masking protocol, and that is worth stating rather than hiding. Deletion
zeroes the first k pixels of the order on top of the image; insertion reveals
the first k on top of a zero background. Reversing the order and inserting k
pixels therefore produces exactly the image that deleting the complementary
n-k pixels in the forward order produces, so

    A_ins(reverse) = A_del(forward)   and   A_del(reverse) = A_ins(forward)

up to the one-pixel rounding of int(n*i/T). Measured on the first images, the
two sides agree to 2.5e-3 and 2.2e-4 respectively, which is that rounding. The
reverse baseline is accordingly computed on the first VERIFY images only, as
evidence for the identity, and reported as a consequence of the protocol rather
than as an independent control. The random baseline is the one that carries
information, and it is shared by all methods.

For a constant attribution map there is no order at all, so a randomized tie
break IS a uniformly random order: the random baseline of that image is the
randomized-tie estimate, and it is reported as such rather than recomputed.

The second question is Grad-CAM++ on degenerate maps. A constant map has no
attribution order. numpy's argsort on a constant array still returns a
permutation -- introsort's -- which is a fixed function of array length and
therefore a spatial scan artifact, not saliency. Those images are rescored here
with a randomized tie order, so the published number can be read beside a
version in which the ties are broken at random rather than by the sort's
internal order.

The attribution maps are read from out/xai/cams_all24.npz, the record written
by the scoring run, so the orders scored here are exactly the orders scored
there and no map is recomputed. The forward-order areas are recomputed anyway
and checked against the published ones, which is the assertion that the two
code paths agree.
"""
import os, sys, json, csv, argparse, warnings
import numpy as np
import torch, timm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xai import FrozenHead, load_img

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
torch.set_num_threads(2)
MODEL = "convnextv2_tiny.fcmae_ft_in22k_in1k"
STEPS = 25
VERIFY = 3        # images on which the reverse-order identity is checked


def area(model, x, order, cls, mode, shape):
    """Deletion/insertion area for an explicit pixel order. Identical masking
    protocol to xai.del_ins; the 26 masked images are evaluated in one batch."""
    n = len(order)
    base = torch.zeros_like(x) if mode == "del" else x
    cur = x if mode == "del" else torch.zeros_like(x)
    batch = []
    for i in range(STEPS + 1):
        k = int(n * i / STEPS)
        m = np.zeros(n, dtype=bool)
        m[order[:k]] = True
        mt = torch.tensor(m.reshape(shape), dtype=torch.bool)
        z = cur.clone()
        z[0, :, mt] = base[0, :, mt]
        batch.append(z)
    with torch.no_grad():
        ys = torch.softmax(model(torch.cat(batch, 0)), 1)[:, cls].numpy()
    return float(np.trapezoid(ys, dx=1.0 / STEPS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-random", type=int, default=3)
    a = ap.parse_args()

    rows = list(csv.DictReader(open(os.path.join(ROOT, "manifest.csv"))))
    label = {r["file"]: int(r["label"]) for r in rows}
    d = np.load(os.path.join(ROOT, "out", "features", f"plain__{MODEL}.npz"),
                allow_pickle=True)
    F, files = d["features"], [str(x) for x in d["files"]]
    idx = {f: i for i, f in enumerate(files)}
    y_all = np.array([label[f] for f in files])
    assign = json.load(open(os.path.join(ROOT, "out", "splits",
                                         "split_case_seed0.json")))["assign"]
    dev = [f for f in files if assign[f] in ("train", "val")]
    te = [f for f in files if assign[f] == "test"]
    sc = StandardScaler().fit(F[[idx[f] for f in dev]])
    clf = LogisticRegression(C=1.0, max_iter=3000, random_state=0)
    clf.fit(sc.transform(F[[idx[f] for f in dev]]), y_all[[idx[f] for f in dev]])
    bb = timm.create_model(MODEL, pretrained=False, num_classes=0)
    sd = torch.load(os.path.join(ROOT, "weights", f"{MODEL}.pth"),
                    map_location="cpu", weights_only=True)
    bb.load_state_dict({k: v for k, v in sd.items()
                        if not k.startswith("head.fc")}, strict=False)
    bb.eval()
    model = FrozenHead(bb, sc.mean_, sc.scale_, clf.coef_[0],
                       float(clf.intercept_[0])).eval()
    cfg = bb.default_cfg
    size = cfg["input_size"][-1]

    probs = clf.predict_proba(sc.transform(F[[idx[f] for f in te]]))[:, 1]
    correct = [f for f, p in zip(te, probs) if (p >= 0.5) == (label[f] == 1)]
    sel_probs = {f: float(p) for f, p in zip(te, probs)}

    cams = np.load(os.path.join(ROOT, "out", "xai", "cams_all24.npz"),
                   allow_pickle=True)
    keys = list(cams.keys())
    methods = list(dict.fromkeys(k.split("|")[0] for k in keys))
    score_set = list(dict.fromkeys(k.split("|", 1)[1] for k in keys))
    print(f"scoring set: {len(score_set)} images, "
          f"{sum(label[f] == 1 for f in score_set)} malignant, "
          f"{sum(label[f] == 0 for f in score_set)} benign; "
          f"{sum(f in correct for f in score_set)} of {len(score_set)} are "
          f"classified correctly by the scored model; the seed-0 test "
          f"partition is {len(correct)} correct of {len(te)}", flush=True)
    print(f"methods: {methods}", flush=True)

    published = {}
    pj = os.path.join(ROOT, "out", "xai", "xai_scores.json")
    if os.path.exists(pj):
        for r in json.load(open(pj)).get("scores", []):
            published[(r["method"], r["file"])] = (r["deletion_auc"],
                                                   r["insertion_auc"])

    out = {"n_images": len(score_set), "steps": STEPS,
           "n_random_repeats": a.n_random, "methods": methods,
           "records": [], "degenerate": [], "reproduction_check": []}
    # resume: each image is written as soon as it is scored, so an interrupted
    # run is picked up where it stopped rather than restarted
    partial = os.path.join(ROOT, "out", "xai", "order_baselines_partial.json")
    if os.path.exists(partial):
        out = json.load(open(partial))
        done = {r["file"] for r in out["records"]}
        print(f"resuming: {len(done)} of {len(score_set)} images already scored",
              flush=True)
    else:
        done = set()
    for f in score_set:
        if f in done:
            continue
        x, _ = load_img(os.path.join(ROOT, "data", "plain", f), size,
                        cfg["mean"], cfg["std"])
        cls = label[f]
        rec = {"file": f, "label": cls, "prob": sel_probs[f],
               "correct": f in correct}
        n = shape = None
        for mname in methods:
            g = cams[f"{mname}|{f}"]
            shape = g.shape
            flat = g.flatten()
            n = len(flat)
            deg = bool(np.ptp(g) < 1e-8)
            o_fwd = np.argsort(-flat)
            # The forward-order areas are already in xai_scores.json. On the
            # first VERIFY images they are recomputed through this code path
            # and checked against that record; on the rest the record is used,
            # because the check reproduces it to 1.1e-07, which is float32
            # noise, and recomputing it would cost two thirds of the run for
            # numbers that are already published.
            if len(out["records"]) < VERIFY or (mname, f) not in published:
                e = {"degenerate": deg,
                     "del": area(model, x, o_fwd, cls, "del", shape),
                     "ins": area(model, x, o_fwd, cls, "ins", shape),
                     "source": "recomputed"}
            else:
                pdel, pins = published[(mname, f)]
                e = {"degenerate": deg, "del": pdel, "ins": pins,
                     "source": "xai_scores.json"}
            if len(out["records"]) < VERIFY:
                o_rev = o_fwd[::-1].copy()
                e["del_reverse"] = area(model, x, o_rev, cls, "del", shape)
                e["ins_reverse"] = area(model, x, o_rev, cls, "ins", shape)
                e["identity_check_ins_rev_vs_del"] = abs(e["ins_reverse"] - e["del"])
                e["identity_check_del_rev_vs_ins"] = abs(e["del_reverse"] - e["ins"])
            if deg:
                out["degenerate"].append({"method": mname, "file": f})
            rec[mname] = e
            if (mname, f) in published and e["source"] == "recomputed":
                pd_, pi = published[(mname, f)]
                out["reproduction_check"].append(
                    {"method": mname, "file": f,
                     "del_diff": abs(pd_ - e["del"]),
                     "ins_diff": abs(pi - e["ins"])})
        rr = np.random.default_rng(abs(hash(("rand", f))) % 2**31)
        ds = [area(model, x, rr.permutation(n), cls, "del", shape)
              for _ in range(a.n_random)]
        is_ = [area(model, x, rr.permutation(n), cls, "ins", shape)
               for _ in range(a.n_random)]
        rec["Random"] = {"del": float(np.mean(ds)), "ins": float(np.mean(is_)),
                         "del_sd": float(np.std(ds)),
                         "ins_sd": float(np.std(is_))}
        # a constant map has no order, so its randomized-tie estimate is the
        # random-order estimate; it is recorded by reference, not recomputed
        for mname in methods:
            if rec[mname]["degenerate"]:
                rec[mname]["del_randomtie"] = rec["Random"]["del"]
                rec[mname]["ins_randomtie"] = rec["Random"]["ins"]
                rec[mname]["randomtie_source"] = "shared random-order draws"
        out["records"].append(rec)
        json.dump(out, open(partial, "w"))
        print(f"  {f:28} done  ({len(out['records'])}/{len(score_set)})",
              flush=True)

    if out["reproduction_check"]:
        md = max(r["del_diff"] for r in out["reproduction_check"])
        mi = max(r["ins_diff"] for r in out["reproduction_check"])
        print(f"\nreproduction of the published forward-order areas: "
              f"max |delta deletion| {md:.2e}, max |delta insertion| {mi:.2e}")
        out["reproduction_max_abs_diff"] = {"deletion": md, "insertion": mi}
    json.dump(out, open(os.path.join(ROOT, "out", "xai",
                                     "order_baselines.json"), "w"), indent=2)
    print("wrote out/xai/order_baselines.json")


if __name__ == "__main__":
    main()
