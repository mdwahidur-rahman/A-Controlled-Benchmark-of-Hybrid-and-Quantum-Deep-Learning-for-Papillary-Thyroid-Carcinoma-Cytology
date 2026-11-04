"""
Convert the cached feature matrices between the ``.npz`` form the pipeline
reads and the ``.csv.gz`` form the repository ships.

The repository distributes extracted features as gzip-compressed CSV because a
CSV is readable without this codebase and without NumPy's pickle machinery.
The pipeline itself reads ``.npz``, so a fresh clone runs

    python code/features_csv.py to-npz

once, after which every downstream script works against ``out/features``
without re-running the encoders.

Each CSV has one row per field. The first column is the image filename, which
joins to ``manifest.csv``; the remaining columns ``f0 ... f{d-1}`` are the
pooled encoder features. Each value is written with the shortest decimal string
that converts back to the identical ``float32``, so the CSV reproduces the
cached matrix bit for bit rather than approximately. The encoder name, the
preprocessing variant and the input resolution are carried in a sidecar
``<name>.meta.json`` so that nothing is inferred from the filename.
"""
import argparse
import gzip
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT = os.path.join(ROOT, "out", "features")


def to_csv(outdir):
    os.makedirs(outdir, exist_ok=True)
    total = 0
    for fn in sorted(os.listdir(FEAT)):
        if not fn.endswith(".npz"):
            continue
        d = np.load(os.path.join(FEAT, fn), allow_pickle=True)
        F, files = d["features"], d["files"]
        stem = fn[:-4]
        path = os.path.join(outdir, stem + ".csv.gz")
        with gzip.open(path, "wt", compresslevel=6, newline="\n") as fh:
            fh.write("file," + ",".join(f"f{i}" for i in range(F.shape[1])) + "\n")
            for name, row in zip(files, F):
                fh.write(str(name) + "," + ",".join(
                    np.format_float_positional(v, unique=True, trim="-")
                    for v in row) + "\n")
        meta = {"model": str(d["model"]), "variant": str(d["variant"]),
                "input_size": int(d["input_size"]),
                "n_fields": int(F.shape[0]), "n_features": int(F.shape[1]),
                "dtype": "float32",
                "written_as": "shortest decimal that round-trips to float32"}
        json.dump(meta, open(os.path.join(outdir, stem + ".meta.json"), "w"),
                  indent=2)
        size = os.path.getsize(path)
        total += size
        print(f"  {stem:52} {F.shape[0]:5} x {F.shape[1]:5}  "
              f"{size / 1e6:6.1f} MB")
    print(f"  {'total':52} {total / 1e6:21.1f} MB")


def to_npz(indir):
    os.makedirs(FEAT, exist_ok=True)
    for fn in sorted(os.listdir(indir)):
        if not fn.endswith(".csv.gz"):
            continue
        stem = fn[:-7]
        meta = json.load(open(os.path.join(indir, stem + ".meta.json")))
        names, rows = [], []
        with gzip.open(os.path.join(indir, fn), "rt") as fh:
            next(fh)
            for line in fh:
                parts = line.rstrip("\n").split(",")
                names.append(parts[0])
                rows.append(np.asarray(parts[1:], dtype=np.float32))
        np.savez_compressed(
            os.path.join(FEAT, stem + ".npz"),
            features=np.vstack(rows), files=np.array(names),
            model=np.array(meta["model"]), variant=np.array(meta["variant"]),
            input_size=np.array(meta["input_size"]))
        print(f"  wrote out/features/{stem}.npz  "
              f"{len(rows)} x {len(rows[0])}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("direction", choices=["to-csv", "to-npz"])
    ap.add_argument("--dir", default=os.path.join(ROOT, "data", "features"),
                    help="directory of .csv.gz files (default: data/features)")
    a = ap.parse_args()
    if a.direction == "to-csv":
        to_csv(a.dir)
    else:
        to_npz(a.dir)


if __name__ == "__main__":
    main()
