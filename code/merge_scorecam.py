"""Merge the completed Score-CAM run into xai_scores.json.

Score-CAM had been scored on score_set[:8] only. Those eight happen to be
pick_pos[:8] -- every one malignant -- because score_set is built as
pick_pos[:12] + pick_neg[:12]. The published Score-CAM row was therefore not
merely smaller than the other three methods' rows but drawn from a different,
all-positive population, which is why it could not be ranked against them.

This script takes the 16 newly scored images from scorecam_full.json, checks
that the recomputed image 0 reproduces the stored value, and writes the merged
24-image Score-CAM row back into xai_scores.json. It also recomputes the
pairwise agreement so that Score-CAM's correlations use every grid image rather
than the subset that happened to be covered.
"""
import json, os, itertools
import numpy as np
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XD = os.path.join(ROOT, "out", "xai")

full = json.load(open(os.path.join(XD, "scorecam_full.json")))
sc = json.load(open(os.path.join(XD, "xai_scores.json")))
old = {r["file"]: r for r in sc["scores"] if r["method"] == "ScoreCAM"}

recheck = [r for r in full["rows"] if r["role"] == "recheck"]
assert len(recheck) == 1
f = recheck[0]["file"]
for k in ("deletion_auc", "insertion_auc"):
    d = abs(recheck[0][k] - old[f][k])
    assert d < 1e-9, (k, d)
    print(f"  reproducibility check on {f}: {k} matches to {d:.2e}")

new = [{k: r[k] for k in ("method", "file", "label", "deletion_auc", "insertion_auc")}
       for r in full["rows"] if r["role"] == "new"]
assert len(new) == 16, len(new)
kept = [r for r in sc["scores"] if r["method"] != "ScoreCAM"]
merged = kept + list(old.values()) + new
by = {}
for r in merged:
    by.setdefault(r["method"], []).append(r)
for m, rs in by.items():
    assert len({r["file"] for r in rs}) == len(rs), m
    print(f"  {m:11} {len(rs)} images, "
          f"{sum(r['label'] == 1 for r in rs)} PTC / {sum(r['label'] == 0 for r in rs)} benign")

sc["scores"] = merged
sc["scorecam_note"] = ("Score-CAM scored on the full 24-image set; the first 8 "
                       "are from the original run and reproduce exactly.")
json.dump(sc, open(os.path.join(XD, "xai_scores.json"), "w"), indent=2)
print("\nmerged -> xai_scores.json")

for m, rs in sorted(by.items()):
    d = [r["deletion_auc"] for r in rs]; i = [r["insertion_auc"] for r in rs]
    print(f"  {m:11} n={len(rs):2}  del {np.mean(d):.4f} +/- {np.std(d):.4f}   "
          f"ins {np.mean(i):.4f} +/- {np.std(i):.4f}")
