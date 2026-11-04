"""
Export every per-field prediction to one tidy CSV.

The pipeline writes predictions as one ``.npz`` per (arm, model, regime, seed),
which is convenient for the statistics scripts and inconvenient for everyone
else. This collapses all of them into a single long-format table so that the
per-field record underlying every number in the paper can be inspected, joined
against ``manifest.csv`` and recomputed without this codebase.

Columns
-------
arm        ``benchmark`` for the twelve-encoder panel, ``hybrid`` for the
           fusion ladder and its controls
config     the encoder name for the benchmark arm, the fusion configuration
           for the hybrid arm
regime     ``case`` (recovered-group-disjoint) or ``image``
head       the classifier head, where the filename records one
seed       0-4
file       image filename, joins to ``manifest.csv``
case       the recovered group identifier
y          ground-truth label, 1 = papillary thyroid carcinoma
p          predicted posterior for the positive class

Probabilities are written with the shortest decimal string that converts back
to the identical ``float32``, so the CSV is exact rather than rounded.
"""
import argparse
import gzip
import os
import re

import csv

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "out", "results")
CASE_OF = {r["file"]: r["case"]
           for r in csv.DictReader(open(os.path.join(ROOT, "manifest.csv")))}

BENCH = re.compile(r"^preds_(?P<variant>\w+?)_(?P<config>.+)_(?P<regime>case|image)"
                   r"_(?P<head>\w+)_s(?P<seed>\d)\.npz$")
HYB = re.compile(r"^hyb_(?P<variant>\w+?)_(?P<config>.+)_(?P<regime>case|image)"
                 r"_s(?P<seed>\d)\.npz$")


def rows():
    for fn in sorted(os.listdir(RES)):
        if not fn.endswith(".npz"):
            continue
        m = BENCH.match(fn)
        arm, head = "benchmark", None
        if m is None:
            m = HYB.match(fn)
            arm, head = "hybrid", ""
        if m is None:
            print(f"  unmatched, skipped: {fn}")
            continue
        g = m.groupdict()
        d = np.load(os.path.join(RES, fn), allow_pickle=True)
        head = g.get("head", head)
        # the hybrid arm's records predate the group column; recover it from
        # the manifest rather than leaving the field blank
        cases = (d["cases"] if "cases" in d.files
                 else np.array([CASE_OF.get(str(f), "") for f in d["files"]]))
        for f, c, y, p in zip(d["files"], cases, d["y"], d["p"]):
            yield (arm, g["variant"], g["config"], g["regime"], head,
                   g["seed"], str(f), str(c), int(y), float(p))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join(
        ROOT, "data", "predictions", "per_field_predictions.csv.gz"))
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    n = 0
    with gzip.open(a.out, "wt", compresslevel=6, newline="\n") as fh:
        fh.write("arm,variant,config,regime,head,seed,file,case,y,p\n")
        for r in rows():
            fh.write(",".join(str(x) for x in r[:-1]) + "," +
                     np.format_float_positional(np.float32(r[-1]),
                                                unique=True, trim="-") + "\n")
            n += 1
    print(f"  wrote {os.path.relpath(a.out, ROOT)}  {n:,} rows  "
          f"{os.path.getsize(a.out) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
