"""Shared evaluation utilities (corrected protocol; see R4_CONTROL.md Part 1).

A "method" is a list of one or more *repetitions*; each repetition is a dict:
    {"score": float array (higher = more anomalous), "pred": {op_name: bool array}}
aligned with a single label vector y (1 = attacked, 0 = benign/negative).

Stratified bootstrap over final test predictions:
  each replicate draws n_neg negatives and n_pos positives WITH replacement (optionally
  weighted positives), picks one repetition per method uniformly at random, and computes
  AUROC, AUPRC and, per operating point, precision/recall/FPR/FNR/alerts-per-1000.
All methods share the resampled indices of a replicate, so method differences are PAIRED.
"""
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score

NAN = float("nan")


def auroc(y, s):
    npos = int(y.sum()); nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return NAN
    r = rankdata(s)
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def auprc(y, s):
    return float(average_precision_score(y, s)) if y.sum() > 0 else NAN


def opm(y, pred):
    tp = int(np.sum(pred & (y == 1))); fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1))); tn = int(np.sum(~pred & (y == 0)))
    return {"precision": tp / (tp + fp) if tp + fp else NAN,
            "recall": tp / (tp + fn) if tp + fn else NAN,
            "FPR": fp / (fp + tn) if fp + tn else NAN,
            "FNR": fn / (fn + tp) if fn + tp else NAN,
            "alerts_per_1000": 1000.0 * (tp + fp) / len(y)}


def metrics(y, rep):
    m = {"AUROC": auroc(y, rep["score"]), "AUPRC": auprc(y, rep["score"])}
    for op, pred in rep.get("pred", {}).items():
        for k, v in opm(y, pred).items():
            m[f"{k}@{op}"] = v
    return m


def point(y, method, idx=None):
    """Metric on the full (or given) population, averaged over repetitions."""
    idx = np.arange(len(y)) if idx is None else idx
    ms = [metrics(y[idx], {"score": r["score"][idx],
                           "pred": {k: v[idx] for k, v in r.get("pred", {}).items()}}) for r in method]
    return {k: float(np.nanmean([m[k] for m in ms])) for k in ms[0]}


def bootstrap(y, methods, n_pos=None, n_neg=None, B=2000, seed=12345, pos_weights=None, pool=None):
    """Return {method: {metric: np.array(B)}}. pool = optional index subset defining the population."""
    rng = np.random.default_rng(seed)
    pool = np.arange(len(y)) if pool is None else np.asarray(pool)
    pos = pool[y[pool] == 1]; neg = pool[y[pool] == 0]
    npos = len(pos) if n_pos is None else n_pos
    nneg = len(neg) if n_neg is None else n_neg
    pw = None
    if pos_weights is not None:
        pw = np.asarray([pos_weights[i] for i in pos], float); pw = pw / pw.sum()
    out = {m: {} for m in methods}
    for b in range(B):
        ib = np.concatenate([rng.choice(pos, npos, replace=True, p=pw), rng.choice(neg, nneg, replace=True)])
        yb = y[ib]
        for m, reps in methods.items():
            rep = reps[rng.integers(len(reps))]
            mb = metrics(yb, {"score": rep["score"][ib],
                              "pred": {k: v[ib] for k, v in rep.get("pred", {}).items()}})
            for k, v in mb.items():
                out[m].setdefault(k, np.empty(B))[b] = v
    return out


def ci(a):
    a = np.asarray(a, float); a = a[~np.isnan(a)]
    if len(a) == 0:
        return (NAN, NAN, NAN)
    return (float(a.mean()), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))


def paired_delta(boot, a, b, metric):
    d = boot[a][metric] - boot[b][metric]
    mu, lo, hi = ci(d)
    d = d[~np.isnan(d)]
    return {"delta": mu, "lo": lo, "hi": hi, "frac_le0": float(np.mean(d <= 0)) if len(d) else NAN}


def indep_delta(boot_a, boot_b, metric):
    """Difference between two INDEPENDENT bootstraps (different test sets): not paired."""
    d = boot_a[metric] - boot_b[metric]
    mu, lo, hi = ci(d)
    return {"delta": mu, "lo": lo, "hi": hi}


def thr_at_recall(train_scores_pos, target=0.80):
    """Threshold t such that fraction of TRAINING positives with score >= t is ~target."""
    return float(np.quantile(np.asarray(train_scores_pos, float), 1 - target))


def fmt(t, d=3):
    return f"{t[0]:.{d}f} [{t[1]:.{d}f}, {t[2]:.{d}f}]"
