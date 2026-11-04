"""
Shared metric definitions.

Accuracy is a poor summary for clinical binary tasks, so the full set is
reported everywhere: both error rates separately (a false negative and a false
positive are not interchangeable in a cancer screen), MCC as the balanced
single-number summary that stays honest under class imbalance, and calibration
alongside discrimination.
"""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def confusion(y, p, thr=0.5):
    yh = (np.asarray(p) >= thr).astype(int)
    y = np.asarray(y).astype(int)
    tp = int(((yh == 1) & (y == 1)).sum())
    tn = int(((yh == 0) & (y == 0)).sum())
    fp = int(((yh == 1) & (y == 0)).sum())
    fn = int(((yh == 0) & (y == 1)).sum())
    return tp, tn, fp, fn


def all_metrics(y, p, thr=0.5):
    y = np.asarray(y).astype(int); p = np.asarray(p, dtype=float)
    tp, tn, fp, fn = confusion(y, p, thr)
    n = tp + tn + fp + fn
    sens = tp / (tp + fn) if (tp + fn) else 0.0          # TPR, recall
    spec = tn / (tn + fp) if (tn + fp) else 0.0          # TNR
    prec = tp / (tp + fp) if (tp + fp) else 0.0          # PPV
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    acc = (tp + tn) / n if n else 0.0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) else 0.0

    # Matthews correlation coefficient -- balanced even under heavy imbalance
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn - fp * fn) / den) if den > 0 else 0.0

    # Cohen's kappa
    po = acc
    pe = (((tp + fp) * (tp + fn)) + ((tn + fn) * (tn + fp))) / (n * n) if n else 0.0
    kappa = (po - pe) / (1 - pe) if (1 - pe) else 0.0

    two_class = len(set(y.tolist())) > 1
    return {
        "n": n, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "acc": acc,
        "err": 1.0 - acc,                                 # overall error rate
        "balanced_acc": (sens + spec) / 2,
        "sens": sens, "spec": spec,
        "fnr": 1.0 - sens,                                # missed cancers
        "fpr": 1.0 - spec,                                # false alarms
        "prec": prec, "npv": npv,
        "f1": f1,
        "mcc": mcc,
        "kappa": kappa,
        "youden_j": sens + spec - 1.0,
        "auc": roc_auc_score(y, p) if two_class else float("nan"),
        "auprc": average_precision_score(y, p) if two_class else float("nan"),
        "brier": float(np.mean((p - y) ** 2)),
    }


def ece(y, p, bins=10):
    """Expected calibration error, equal-width bins."""
    y = np.asarray(y).astype(int); p = np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    e, n = 0.0, len(y)
    rows = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p > lo) & (p <= hi) if i else (p >= lo) & (p <= hi)
        if not m.any():
            rows.append((lo, hi, 0, float("nan"), float("nan"))); continue
        conf, acc_b = float(p[m].mean()), float(y[m].mean())
        e += m.sum() / n * abs(acc_b - conf)
        rows.append((lo, hi, int(m.sum()), conf, acc_b))
    return float(e), rows


def bootstrap_ci(y, p, fn, n_boot=2000, alpha=0.05, seed=0):
    """Percentile bootstrap CI for any metric function of (y, p)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y); p = np.asarray(p); n = len(y)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if len(set(y[i].tolist())) < 2:
            continue
        try:
            vals.append(fn(y[i], p[i]))
        except Exception:
            continue
    if not vals:
        return float("nan"), float("nan")
    return (float(np.percentile(vals, 100 * alpha / 2)),
            float(np.percentile(vals, 100 * (1 - alpha / 2))))


def delong_test(y, p1, p2):
    """DeLong test for two correlated ROC curves. Returns (z, two-sided p)."""
    from scipy import stats
    y = np.asarray(y).astype(int)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    m, n = len(pos), len(neg)
    if m == 0 or n == 0:
        return float("nan"), float("nan")

    def structural(p):
        X, Y = np.asarray(p)[pos], np.asarray(p)[neg]
        # midrank-free direct kernel: psi(x, y) = 1 if x>y, .5 if ==, 0 else
        K = (X[:, None] > Y[None, :]).astype(float) + 0.5 * (X[:, None] == Y[None, :])
        theta = K.mean()
        v10 = K.mean(axis=1)      # per-positive
        v01 = K.mean(axis=0)      # per-negative
        return theta, v10, v01

    t1, a10, a01 = structural(p1)
    t2, b10, b01 = structural(p2)
    s10 = np.cov(np.vstack([a10, b10]))
    s01 = np.cov(np.vstack([a01, b01]))
    S = s10 / m + s01 / n
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    if var <= 0:
        return float("nan"), float("nan")
    z = (t1 - t2) / np.sqrt(var)
    return float(z), float(2 * (1 - stats.norm.cdf(abs(z))))


def mcnemar_test(y, p1, p2, thr=0.5):
    """Exact-ish McNemar on paired correctness. Returns (b, c, p)."""
    from scipy import stats
    y = np.asarray(y).astype(int)
    c1 = ((np.asarray(p1) >= thr).astype(int) == y)
    c2 = ((np.asarray(p2) >= thr).astype(int) == y)
    b = int((c1 & ~c2).sum())
    c = int((~c1 & c2).sum())
    if b + c == 0:
        return b, c, 1.0
    p = float(stats.binomtest(b, b + c, 0.5).pvalue)
    return b, c, p
