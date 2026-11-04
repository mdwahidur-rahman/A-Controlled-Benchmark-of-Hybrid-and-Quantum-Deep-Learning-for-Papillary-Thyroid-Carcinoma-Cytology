# A Controlled Benchmark of Hybrid and Quantum Deep Learning for Papillary Thyroid Carcinoma Cytology

Code, extracted features and result records for a controlled benchmark on
thyroid fine-needle aspiration cytology. The study isolates the **partitioning
rule** as an experimental factor across twelve frozen encoders and five seeds,
and tests every architectural component that is credited with a gain — feature
fusion, a variational quantum block, federated training, four attribution
methods — against a control matched to it.

The headline results are negative, and deliberately so. The repository exists
so that each one can be re-derived rather than taken on trust.

---

## What the study found

| | Result |
|---|---|
| **Partitioning rule** | Image-level splitting gave the higher mean accuracy for **all twelve** encoders, by 1.26 points on average. With the recovered group as the resampling unit the direction is consistent (exact sign test, *p* = 0.00049), but **no individual encoder's difference** and **none of the 66 architecture contrasts** is separable from zero after Holm correction. |
| **Fusion** | Neither the direct concatenation nor the projected head exceeded the stronger single branch. Against a branch carrying no information, the margin is not separable from zero in either regime. |
| **Quantum block** | The five-qubit amplitude-encoded map has an **exact closed-form classical twin** — agreeing end to end in the expectations, the logit, the loss, all 30 parameter gradients and the training trajectory — and sits 19.38 points below omitting the block and 20.05 below a budget-matched classical control. |
| **Federation** | Matched a centralized model fitted on the **same retained data** to within 0.01 points at severe skew. The apparent deficit against an all-data reference was the cost of the clients the retention rule removed. |
| **Attribution** | A **uniformly random pixel order** attains a *lower* deletion area than every one of the four methods. Deletion area therefore cannot rank them; insertion area can, and all four beat random there. |

---

## Repository layout

```
.
├── code/                      # 36 Python modules, flat (see the map below)
├── data/
│   ├── features/              # extracted features, one .csv.gz + .meta.json per encoder
│   └── predictions/           # every per-field prediction, one tidy .csv.gz
├── out/
│   ├── splits/                # the exact partition assignment for every seed and rule
│   ├── results/               # per-run records: predictions, metrics, statistics
│   ├── tables/                # every table in the paper, as .csv and .md
│   └── xai/                   # attribution scores, agreement, order baselines
├── manifest.csv               # image -> label, recovered group, field index
├── requirements.txt
└── README.md
```

`code/` is intentionally flat: the modules import each other directly
(`from metrics import all_metrics`, `from xai import FrozenHead`) and resolve
paths as `ROOT = dirname(dirname(__file__))`. Moving a file into a subpackage
breaks both. The logical grouping is:

| Group | Modules |
|---|---|
| **Data and splits** | `make_splits.py`, `extract_features.py`, `features_csv.py`, `predictions_csv.py` |
| **Benchmark** | `benchmark.py`, `sweep.py`, `metrics.py`, `patient_inference.py` |
| **Fusion** | `hybrid.py`, `hybrid_v2.py` |
| **Quantum** | `quantum.py`, `qml_arm.py`, `quantum_endtoend.py` |
| **Federated** | `federated.py`, `federated_v2.py`, `fedprox_probe.py` |
| **Attribution** | `xai.py`, `xai_arms.py`, `xai_agreement_full.py`, `xai_central_penult.py`, `xai_scorecam_full.py`, `xai_order_baselines.py`, `cam_degeneracy.py`, `merge_scorecam.py`, `grid.py` |
| **Statistics** | `analysis.py`, `stats_round2.py`, `stats_round2b.py`, `stats_round2c.py`, `cluster_intervals.py` |
| **Reporting** | `tables.py`, `round2_tables.py`, `sync_tables.py`, `figures.py`, `figures2.py`, `figures3.py`, `fig_fl_qml.py` |

---

## Quick start

```bash
git clone <this-repository>
cd ptc-cytology-benchmark
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# rebuild the .npz feature cache the pipeline reads, from the shipped CSVs
python code/features_csv.py to-npz
```

That is enough to reproduce **every number in the paper**. The twelve encoders
do not need to be run and no image data is required, because the features they
produce are shipped here.

```bash
python code/benchmark.py          # the twelve-encoder panel under both rules
python code/hybrid.py             # fusion ladder and its controls
python code/hybrid_v2.py          # label-preserving control, repeated permutations
python code/qml_arm.py            # quantum block against its control bank
python code/quantum_endtoend.py   # circuit vs twin: expectations, gradients, trajectory
python code/federated_v2.py       # federation, corrected FedProx, retained-data reference
python code/stats_round2.py       # the corrected partitioning analysis
python code/stats_round2b.py      # multiplicity-adjusted cluster contrasts
python code/cluster_intervals.py  # group-cluster intervals for every encoder
python code/tables.py             # regenerate out/tables/
```

Attribution and figure generation additionally need the image corpus; see
**Working from raw images** below.

---

## The feature data

`data/features/` holds one gzip-compressed CSV per encoder and preprocessing
variant, sixteen in all. The format is deliberately boring: it can be read by
anything, with or without this codebase.

| | |
|---|---|
| **Rows** | 1,332 — one per cytological field, in a fixed order |
| **Column 1** | `file`, joining to `manifest.csv` |
| **Columns 2…d+1** | `f0 … f{d-1}`, the pooled encoder features |
| **Precision** | Each value is the **shortest decimal string that converts back to the identical `float32`**, so the CSV reproduces the cached matrix bit for bit, not approximately |
| **Sidecar** | `<name>.meta.json` carries the encoder name, the preprocessing variant, the input resolution and the matrix shape, so nothing is inferred from a filename |

```python
import gzip, numpy as np
with gzip.open("data/features/plain__convnextv2_tiny.fcmae_ft_in22k_in1k.csv.gz", "rt") as fh:
    header = next(fh).rstrip().split(",")
    names, rows = [], []
    for line in fh:
        p = line.rstrip().split(",")
        names.append(p[0]); rows.append(np.asarray(p[1:], dtype=np.float32))
X = np.vstack(rows)          # (1332, 768)
```

The twelve `plain__*` files are the benchmark encoders; the four `log__*` files
are the Laplacian-of-Gaussian preprocessing variant used in the enhancement
ablation.

`code/features_csv.py` converts in both directions and is the script that
produced these files.

### Per-field predictions

`data/predictions/per_field_predictions.csv.gz` is one tidy table of **53,680
rows** — every posterior the study produced, for every arm, configuration,
partitioning rule and seed.

| column | meaning |
|---|---|
| `arm` | `benchmark` (twelve-encoder panel) or `hybrid` (fusion ladder and controls) |
| `config` | encoder name, or fusion configuration |
| `regime` | `case` (recovered-group-disjoint) or `image` |
| `head`, `seed` | classifier head; seed 0–4 |
| `file`, `case` | image filename and recovered group, both joining to `manifest.csv` |
| `y`, `p` | ground-truth label (1 = carcinoma) and predicted posterior |

Posteriors use the same exact `float32` round-trip formatting as the features,
so any metric in the paper can be recomputed from this file alone.
`code/predictions_csv.py` regenerates it.

---

## Working from raw images

The image corpus is **not redistributed here**. It is the public release of
Pang et al., and it should be obtained from its depositors so that it reaches
you under their terms, with their documentation.

```
data/
├── plain/          # 1,332 fields, resampled to 384x384
└── log/            # the Laplacian-of-Gaussian variant
```

```bash
python code/make_splits.py        # regenerate out/splits/ (deterministic, seeded)
python code/extract_features.py   # regenerate out/features/ from the images
python code/xai.py                # attribution maps and faithfulness scores
python code/figures.py            # all figures
```

`extract_features.py` downloads the twelve pretrained checkpoints through
`timm`; the exact identifiers are recorded in each `.meta.json`.

---

## Design notes worth knowing before you read the code

**The group, not the image, is the unit.** Several fields come from one
acquisition group, and a group that survives into the evaluation partition
contributes to every seed in which it is held out. Every interval in this work
is a *group-cluster* bootstrap that draws whole groups with all their fields
and all their seed-appearances (`cluster_intervals.py`). A field-level
bootstrap is narrower by a factor of 1.76 for MCC and 1.94 for AUC under the
case-level rule, and understates the variance.

**The identifiers are recovered, not verified.** They are parsed from filename
stems. The release documentation does not state whether a stem is a patient, a
nodule, a slide or a series, so the guarantee this work makes is
*recovered-group separation*, never patient separation.

**Two comparison families, two valid tests.** Encoder-vs-encoder contrasts
under one rule are genuinely paired — the file order and labels are asserted
identical in code before any paired statistic runs. The two partitioning rules
are *not* paired at the observation level, because they induce different
evaluation partitions; that contrast is analysed at the level of the encoder.
`stats_round2.py` documents why, with the measurements.

**Superseded code is kept, not deleted.** `analysis.py` contains the invalid
paired test and `federated.py` the FedProx implementation whose proximal term
was detached from the gradient. Both are retained so the corrections in
`stats_round2.py` and `federated_v2.py` can be checked against what they
replace. `fedprox_probe.py` demonstrates the defect directly and takes about
ten seconds to run.

---

## Compute

Everything was produced on **CPU only** — two cores, 7 GB of RAM, no GPU at any
stage. Feature extraction is the long pole at roughly 38 minutes for all twelve
encoders; from the shipped features, the full statistical pipeline runs in a
few minutes. The quantum arm is an exact state-vector simulation at five
qubits, so it needs no quantum hardware and carries no shot noise.

---

## Citing

If you use this code or these features, please cite the paper. The citation
block will be added here on acceptance.

## License

To be added by the authors before publication. Until a license file is present,
no license is granted.
