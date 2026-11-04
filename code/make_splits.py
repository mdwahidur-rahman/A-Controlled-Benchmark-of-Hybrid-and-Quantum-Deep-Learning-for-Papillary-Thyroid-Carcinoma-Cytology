"""
Leakage-aware split generation for the Pang thyroid cytology cohort.

Produces two split regimes over the SAME images so the inflation attributable
purely to split granularity can be measured:

  case  : stratified over CASES. Every image of a case lands in one subset.
          This is the correct protocol.
  image : stratified over IMAGES, ignoring case. Tiles from one patient's
          slide land in train AND test. This is what much of the literature
          does, and it is the quantity under audit.

Proportions are held identical (64/16/20) so the only difference between the
regimes is the grouping constraint.

Also emits 5-fold CV folds within the development portion of each regime,
honouring the same grouping constraint.
"""
import os, json, csv, argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_FRAC, VAL_FRAC = 0.64, 0.16   # test = 0.20
N_FOLDS = 5


def load_manifest(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def _assign(units, fracs, rng):
    """Shuffle `units` and cut into len(fracs)+1 contiguous blocks."""
    u = list(units)
    rng.shuffle(u)
    n = len(u)
    cuts, acc = [], 0.0
    for f in fracs:
        acc += f
        cuts.append(int(round(acc * n)))
    out, prev = [], 0
    for c in cuts:
        out.append(u[prev:c]); prev = c
    out.append(u[prev:])
    return out


def split_case_level(rows, seed):
    rng = np.random.default_rng(seed)
    by_case = {}
    for r in rows:
        by_case.setdefault(r['case'], []).append(r)
    assign = {}
    # stratify by class: split each class's cases independently
    for cls in sorted({r['cls'] for r in rows}):
        cases = sorted([c for c, v in by_case.items() if v[0]['cls'] == cls])
        tr, va, te = _assign(cases, [TRAIN_FRAC, VAL_FRAC], rng)
        for c in tr: assign[c] = 'train'
        for c in va: assign[c] = 'val'
        for c in te: assign[c] = 'test'
    return {r['file']: assign[r['case']] for r in rows}


def split_image_level(rows, seed):
    rng = np.random.default_rng(seed)
    assign = {}
    for cls in sorted({r['cls'] for r in rows}):
        files = sorted([r['file'] for r in rows if r['cls'] == cls])
        tr, va, te = _assign(files, [TRAIN_FRAC, VAL_FRAC], rng)
        for f in tr: assign[f] = 'train'
        for f in va: assign[f] = 'val'
        for f in te: assign[f] = 'test'
    return assign


def cv_folds(rows, assign, regime, seed):
    """5-fold CV inside the development (train+val) portion, same grouping rule."""
    rng = np.random.default_rng(seed + 10_000)
    dev = [r for r in rows if assign[r['file']] in ('train', 'val')]
    folds = {}
    if regime == 'case':
        by_case = {}
        for r in dev:
            by_case.setdefault(r['case'], []).append(r)
        for cls in sorted({r['cls'] for r in dev}):
            cases = sorted([c for c, v in by_case.items() if v[0]['cls'] == cls])
            rng.shuffle(cases)
            for i, c in enumerate(cases):
                for r in by_case[c]:
                    folds[r['file']] = i % N_FOLDS
    else:
        for cls in sorted({r['cls'] for r in dev}):
            files = sorted([r['file'] for r in dev if r['cls'] == cls])
            rng.shuffle(files)
            for i, f in enumerate(files):
                folds[f] = i % N_FOLDS
    return folds


def verify(rows, assign, regime):
    """Hard integrity checks. Case-level must have ZERO case overlap across subsets."""
    by_case = {}
    for r in rows:
        by_case.setdefault(r['case'], set()).add(assign[r['file']])
    spanning = {c: s for c, s in by_case.items() if len(s) > 1}
    counts = {s: sum(1 for r in rows if assign[r['file']] == s) for s in ('train', 'val', 'test')}
    pos = {s: sum(1 for r in rows if assign[r['file']] == s and r['label'] == '1') for s in counts}
    if regime == 'case':
        assert not spanning, f"case-level split leaked {len(spanning)} cases across subsets"
    return {
        'regime': regime,
        'counts': counts,
        'pos_frac': {s: round(pos[s] / counts[s], 4) for s in counts},
        'cases_spanning_subsets': len(spanning),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default=os.path.join(ROOT, 'manifest.csv'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'out', 'splits'))
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    a = ap.parse_args()

    rows = load_manifest(a.manifest)
    os.makedirs(a.out, exist_ok=True)
    report = []
    for seed in a.seeds:
        for regime, fn in (('case', split_case_level), ('image', split_image_level)):
            assign = fn(rows, seed)
            folds = cv_folds(rows, assign, regime, seed)
            info = verify(rows, assign, regime)
            info['seed'] = seed
            report.append(info)
            with open(os.path.join(a.out, f'split_{regime}_seed{seed}.json'), 'w') as fh:
                json.dump({'assign': assign, 'cv_fold': folds, 'info': info}, fh)
    with open(os.path.join(a.out, 'split_report.json'), 'w') as fh:
        json.dump(report, fh, indent=2)

    print(f"{'regime':6} {'seed':4} {'train':>6} {'val':>5} {'test':>5}  {'pos%tr':>7} {'pos%te':>7}  spanning")
    for r in report:
        c, p = r['counts'], r['pos_frac']
        print(f"{r['regime']:6} {r['seed']:<4} {c['train']:>6} {c['val']:>5} {c['test']:>5}"
              f"  {p['train']*100:>6.1f}% {p['test']*100:>6.1f}%  {r['cases_spanning_subsets']}")


if __name__ == '__main__':
    main()
