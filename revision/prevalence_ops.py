"""Operating points and prevalence (revision stream `prevalence_ops`).

Addresses review comments M8, M9, m3 (prevalence family), m4, m9, m10, m11 and m17 item 5, following
REVISION_ANALYSIS_PLAN.md Sections 3.1, 3.5, 3.6(e), 3.7, 3.8 and the "negative-class and hijack-pattern breakdown"
item of Section 4. All test scores and thresholded predictions are read from the shared caches
(revision/cache/linear/<setting>.pkl, revision/cache/gru/<setting>.pkl). No threshold is set on test data, except
the descriptive matched-recall quantity "FPR at a fixed test TPR of 0.8" (plan 3.8, review m4), which is labelled as
such and is not an operating point.

The only new model fits: the R1 linear detectors (LR-A, LR-B, LR-C, Markov, IForest, TFIDF) are refitted with the
unmodified cache code (build_linear_cache.py, imported) to recover their in-sample TRAINING scores, which the cache does
not store and which the FPR-budget thresholds of m10 need. The refit is verified against the cached test scores and
t_r80 thresholds. GRU training scores are reconstructed from the cached logit decomposition (train_tool_contrib +
train_beh_contrib + bias, fit partition).

Stages (each is idempotent; `all` runs them in order and skips the refit if its part file exists):
  python prevalence_ops.py train_scores   # R1 refit -> prevalence_ops_parts/R1_train_scores.pkl  (~10 s)
  python prevalence_ops.py analyze        # resumable (parts in prevalence_ops_parts/, --budget s per call, prints
                                          # INCOMPLETE or ALL_DONE); ~8 min of work in total; then runs derive
  python prevalence_ops.py derive         # derived keys (chance levels, criteria, counts) added to the JSON
  python prevalence_ops.py summary        # prevalence_ops_summary.md from the JSON
  python prevalence_ops.py all
Environment: AGENTDRIFT (and AGENTDOJO) as for the cache builders.
"""
# Pickles read here are the revision's own cache files, written by the revision scripts on this machine.
import os, sys, json, time, math, pickle, argparse, hashlib, platform
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))            # .../agent-ids/revision
ROOT = os.path.dirname(HERE)                                  # .../agent-ids
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import evalkit as ek                                          # noqa: E402

CACHE = os.path.join(HERE, "cache")
PARTS = os.path.join(HERE, "prevalence_ops_parts")
TRAIN_PKL = os.path.join(PARTS, "R1_train_scores.pkl")
OUT_JSON = os.path.join(HERE, "prevalence_ops_results.json")
OUT_MD = os.path.join(HERE, "prevalence_ops_summary.md")

B = 2000                      # bootstrap replicates everywhere (plan 3.1)
ROUND = 12                    # scores rounded to 12 decimals before ranking (review m9)
BASE = 20260926               # seed base; every block below adds a fixed offset
PIS = {"pi01": 0.01, "pi02": 0.02, "pi05": 0.05, "pi10": 0.10}
LOWPIS = {"pi0.1pct": 0.001, "pi0.01pct": 0.0001}
R1_DETS = ["LR-A", "LR-B", "LR-C", "Markov", "IForest", "GRU", "TFIDF"]
R1_PAIRS = [("LR-C", "Markov"), ("LR-C", "LR-B"), ("LR-C", "LR-A"), ("GRU", "LR-C"), ("TFIDF", "LR-C")]
DOJO_DETS = ["LR-A", "LR-B", "LR-C", "GRU", "TFIDF"]
DOJO_PAIRS = [("LR-C", "LR-B"), ("LR-C", "LR-A"), ("GRU", "LR-C"), ("TFIDF", "LR-C")]
DOMAINS = ["medical", "banking", "web", "email", "coding"]
M9_SETTINGS = ["R1", "R2"] + [f"R3_{d}" for d in DOMAINS] + ["R4", "WUT"]
THETA05_DETS = ["LR-A", "LR-B", "LR-C", "GRU", "TFIDF"]
THETA05_SETTINGS = ["R1"] + [f"R3_{d}" for d in DOMAINS] + ["R4"]
BUDGETS = {"fpr01": 0.01, "fpr05": 0.05}
LABELFREE_DETS = ["LR-C", "LR-B", "GRU"]
N_UNLAB = 200
N_DRAWS = 5
CLUSTER_OF = {"R2": "task", "R4": "task", "WUT": "task"}   # plan 3.5: R2 tasks; AgentDojo suite|user_task
PATTERNS = ["attacked_full", "attacked_partial", "attacked_delayed"]


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rnd(x, d=6):
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return [rnd(v, d) for v in x]
    x = float(x)
    return None if math.isnan(x) else round(x, d)


# =============================================================================== cache loading
def load_setting(s):
    lin = pickle.load(open(os.path.join(CACHE, "linear", f"{s}.pkl"), "rb"))
    gp = os.path.join(CACHE, "gru", f"{s}.pkl")
    gru = pickle.load(open(gp, "rb")) if os.path.exists(gp) else None
    meta = lin["test_meta"]
    y = np.asarray(lin["y"]).astype(int)
    poolE = np.asarray(lin["pool_E"]).astype(bool)
    if gru is not None:
        assert [m["id"] for m in gru["test_meta"]] == [m["id"] for m in meta], f"{s}: row order differs"
        assert np.array_equal(np.asarray(gru["y"]).astype(int), y) and np.array_equal(np.asarray(gru["pool_E"]).astype(bool), poolE)
    dets = {}
    for fam, obj in (("linear", lin), ("gru", gru)):
        if obj is None:
            continue
        for d, sc in obj["scores"].items():
            S = np.atleast_2d(np.asarray(sc, float))
            P = {op: np.atleast_2d(np.asarray(v, bool)) for op, v in obj["pred"][d].items()}
            assert all(p.shape == S.shape for p in P.values())
            dets[d] = {"family": fam, "raw": S, "s": np.round(S, ROUND), "pred": P, "R": S.shape[0]}
    cat = np.array([m["category"] for m in meta])
    pat = np.array([m["pattern"] if m["pattern"] else "" for m in meta])
    return {"setting": s, "meta": meta, "y": y, "poolE": poolE, "cat": cat, "pat": pat, "dets": dets,
            "lin": lin, "gru": gru, "bench": meta[0]["benchmark"]}


# =============================================================================== bootstrap weights
def _counts(rows, B_, n):
    """rows: (B_, k) row indices -> (B_, n) float32 multiplicities."""
    flat = (np.arange(B_)[:, None] * n + rows).ravel()
    return np.bincount(flat, minlength=B_ * n).reshape(B_, n).astype(np.float32)


def w_strata(labels, B_, rng):
    """Stratified bootstrap: every stratum (label != '') resampled with replacement to its own size."""
    labels = np.asarray(labels)
    n = len(labels)
    W = np.zeros((B_, n), np.float32)
    for lab in sorted(set(labels.tolist()) - {""}):
        idx = np.where(labels == lab)[0]
        W += _counts(idx[rng.integers(0, len(idx), size=(B_, len(idx)))], B_, n)
    return W


def w_subsample(y, npos, nneg, B_, rng):
    """v0.3 estimator (evalkit.bootstrap with n_pos): npos positives and nneg negatives, with replacement."""
    n = len(y)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    return (_counts(pos[rng.integers(0, len(pos), size=(B_, npos))], B_, n)
            + _counts(neg[rng.integers(0, len(neg), size=(B_, nneg))], B_, n))


def w_cluster(keys, B_, rng):
    """Cluster bootstrap: clusters (sorted distinct keys) resampled with replacement; row weight = multiplicity."""
    keys = list(keys)
    ks = sorted(set(keys)); kid = {k: j for j, k in enumerate(ks)}
    ci = np.array([kid[k] for k in keys])
    M = _counts(rng.integers(0, len(ks), size=(B_, len(ks))), B_, len(ks))
    return M[:, ci], len(ks)


# =============================================================================== rank-based metrics on weights
class Ranked:
    """Descending sort of one score vector with tie groups; metrics for any row-weight matrix W (B, n)."""

    def __init__(self, s, y):
        o = np.argsort(-s, kind="mergesort")
        ss = s[o]
        self.o = o
        self.ys = y[o].astype(np.float64)
        self.ends = np.r_[np.nonzero(ss[1:] != ss[:-1])[0], len(ss) - 1]
        self.n_groups = len(self.ends)

    def tpfp(self, W, chunk=250):
        tps, fps = [], []
        for i in range(0, W.shape[0], chunk):
            Ws = np.asarray(W[i:i + chunk], np.float64)[:, self.o]
            tps.append(np.cumsum(Ws * self.ys, axis=1)[:, self.ends])
            fps.append(np.cumsum(Ws * (1.0 - self.ys), axis=1)[:, self.ends])
        return np.vstack(tps), np.vstack(fps)


def ap_from(tp, fp, c):
    """(Prevalence-weighted) average precision: each positive weight multiplied by c (scalar or (B,))."""
    c = np.asarray(c, float).reshape(-1, 1) if np.ndim(c) else float(c)
    Wp = tp[:, -1]
    dtp = np.diff(tp, axis=1, prepend=0.0)
    num = c * tp
    den = num + fp
    prec = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    return (dtp * prec).sum(1) / Wp


def auroc_from(tp, fp):
    Wp, Wn = tp[:, -1], fp[:, -1]
    dtp = np.diff(tp, axis=1, prepend=0.0)
    dfp = np.diff(fp, axis=1, prepend=0.0)
    return (dtp * (Wn[:, None] - fp + 0.5 * dfp)).sum(1) / (Wp * Wn)


def fpr_at_tpr_from(tp, fp, target=0.8):
    """Smallest-alert threshold whose (weighted) TPR >= target: returns (FPR, achieved TPR)."""
    Wp, Wn = tp[:, -1], fp[:, -1]
    g = (tp >= target * Wp[:, None] - 1e-9).argmax(1)
    r = np.arange(len(g))
    return fp[r, g] / Wn, tp[r, g] / Wp


def prev_mult(pi, tp, fp):
    """Per-replicate positive-weight multiplier pi/(1-pi) * Wneg/Wpos (plan 3.1)."""
    return pi / (1.0 - pi) * fp[:, -1] / tp[:, -1]


# =============================================================================== summaries
def est(point, boot):
    """point + bootstrap mean and 2.5/97.5 percentiles (evalkit.ci), NaN-aware."""
    b = np.asarray(boot, float)
    mu, lo, hi = ek.ci(b)
    out = {"point": rnd(point), "boot_mean": rnd(mu), "lo": rnd(lo), "hi": rnd(hi)}
    nn = int(np.isnan(b).sum())
    if nn:
        out["n_nan_replicates"] = nn
    return out


def delta_est(point, d):
    d = np.asarray(d, float)
    dd = d[~np.isnan(d)]
    mu, lo, hi = ek.ci(dd)
    le = int((dd <= 0).sum()); ge = int((dd >= 0).sum())
    p = min(1.0, 2.0 * (min(le, ge) + 1) / (len(dd) + 1))
    return {"point": rnd(point), "boot_mean": rnd(mu), "lo": rnd(lo), "hi": rnd(hi),
            "frac_le0": rnd(le / len(dd)), "p_boot": rnd(p), "detected": bool(lo > 0 or hi < 0)}


def holm(fam):
    """Holm step-down on {name: delta_est}; adds p_holm and detected_holm (alpha 0.05)."""
    names = sorted(fam, key=lambda k: fam[k]["p_boot"])
    m = len(names); run = 0.0
    for i, k in enumerate(names):
        run = max(run, min(1.0, (m - i) * fam[k]["p_boot"]))
        fam[k]["p_holm"] = rnd(run)
        fam[k]["detected_holm"] = bool(run < 0.05)
    return fam


def pick(arr, choice):
    """arr (R, B) -> (B,) using the repetition drawn for each replicate."""
    return arr[choice, np.arange(arr.shape[1])]


def eq6(pi, tpr, fpr):
    tpr = np.asarray(tpr, float); fpr = np.asarray(fpr, float)
    den = pi * tpr + (1 - pi) * fpr
    return np.divide(pi * tpr, den, out=np.full(np.broadcast(tpr, fpr).shape, np.nan), where=den > 0)


def alerts(pi, tpr, fpr):
    return 1000.0 * (pi * np.asarray(tpr, float) + (1 - pi) * np.asarray(fpr, float))


# =============================================================================== chance AP (review m9)
def chance_ap_exact(npos, nneg):
    """Expected AP of a uniformly random ranking without ties: (n+-1)/(N-1) + H_N (N-n+)/(N(N-1))."""
    N = npos + nneg
    H = float(np.sum(1.0 / np.arange(1, N + 1)))
    return (npos - 1) / (N - 1) + H * (N - npos) / (N * (N - 1))


def chance_ap_mc(npos, nneg, sims, rng):
    N = npos + nneg
    k = np.arange(1, npos + 1)
    v = np.empty(sims)
    for i in range(sims):
        r = np.sort(rng.choice(N, npos, replace=False)) + 1
        v[i] = np.mean(k / r)
    return float(v.mean()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


# =============================================================================== stage 1: R1 training scores
def stage_train_scores():
    import build_linear_cache as blc
    from sklearn.ensemble import IsolationForest
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_extraction.text import TfidfVectorizer
    from featurise import build_vocab, tool_seq
    from baseline_drift import Markov
    t0 = time.time()
    tr, te, _ = blc.single_split("R1")
    ytr = blc.ylab(tr)
    vocab = build_vocab([r for r in tr if r["category"] == "benign"])
    V = len(vocab)
    Xtr = np.hstack([blc.beh(tr), blc.bag(tr, vocab)]); Xte = np.hstack([blc.beh(te), blc.bag(te, vocab)])
    S_tr, S_te, T80 = {}, {}, {}
    for det in ["LR-A", "LR-B", "LR-C"]:
        c = blc.det_cols(det, V)
        sc, lr, t80, s_tr, _ = blc.lr_fit(Xtr[:, c], ytr, None)
        S_tr[det] = s_tr
        S_te[det] = blc.by_unique_rows(lambda U: lr.predict_proba(sc.transform(U))[:, 1], Xte[:, c])
        T80[det] = t80
    ben = np.array([r["category"] == "benign" for r in tr])
    mk = Markov().fit([tool_seq(r) for r in tr if r["category"] == "benign"])
    S_tr["Markov"] = np.round(np.array([mk.score(tool_seq(r)) for r in tr]), blc.MARKOV_DECIMALS)
    S_te["Markov"] = np.round(np.array([mk.score(tool_seq(r)) for r in te]), blc.MARKOV_DECIMALS)
    T80["Markov"] = ek.thr_at_recall(S_tr["Markov"][ytr == 1])
    sc = StandardScaler().fit(Xtr)
    ifo = IsolationForest(n_estimators=200, random_state=0).fit(sc.transform(Xtr)[ben])
    S_tr["IForest"] = blc.by_unique_rows(lambda U: -ifo.score_samples(sc.transform(U)), Xtr)
    S_te["IForest"] = blc.by_unique_rows(lambda U: -ifo.score_samples(sc.transform(U)), Xte)
    T80["IForest"] = ek.thr_at_recall(S_tr["IForest"][ytr == 1])
    vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True)
    ttr = blc.texts(tr, "TFIDF"); Mtr = vec.fit_transform(ttr)
    lr = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0).fit(Mtr, ytr)
    prob = lambda U: lr.predict_proba(vec.transform(U))[:, 1]
    S_tr["TFIDF"] = blc.by_unique_texts(prob, ttr)
    S_te["TFIDF"] = blc.by_unique_texts(prob, blc.texts(te, "TFIDF"))
    T80["TFIDF"] = ek.thr_at_recall(S_tr["TFIDF"][ytr == 1])
    cache = pickle.load(open(os.path.join(CACHE, "linear", "R1.pkl"), "rb"))
    assert [r["id"] for r in te] == [m["id"] for m in cache["test_meta"]]
    ver = {}
    for det in S_te:
        ref = np.asarray(cache["scores"][det], float)
        ver[det] = {"max_abs_diff_test_scores": float(np.max(np.abs(ref - S_te[det]))),
                    "test_scores_bitwise_equal": bool(np.array_equal(ref, S_te[det])),
                    "thr_t_r80_abs_diff": float(abs(float(cache["thr"][det]["t_r80"]) - T80[det]))}
        ver[det]["pass"] = ver[det]["max_abs_diff_test_scores"] <= 1e-9 and ver[det]["thr_t_r80_abs_diff"] <= 1e-9
    obj = {"train_ids": [r["id"] for r in tr], "train_category": np.array([r["category"] for r in tr]),
           "train_y": ytr, "train_scores": S_tr, "thr_t_r80": T80, "verification": ver,
           "build_linear_cache_sha256": sha256(os.path.join(HERE, "build_linear_cache.py")),
           "script_sha256": sha256(os.path.abspath(__file__)), "runtime_s": round(time.time() - t0, 1)}
    os.makedirs(PARTS, exist_ok=True)
    pickle.dump(obj, open(TRAIN_PKL, "wb"))
    log(f"train_scores written ({obj['runtime_s']} s); verification: " +
        ", ".join(f"{d} {'ok' if v['pass'] else 'FAIL'} ({v['max_abs_diff_test_scores']:.1e})" for d, v in ver.items()))
    assert all(v["pass"] for v in ver.values()), "refit does not reproduce the cache"


# =============================================================================== analysis helpers
def reps_choice(R, rng):
    return rng.integers(0, R, size=B) if R > 1 else np.zeros(B, int)


def ranked_block(S, y, idx, W, pis, extra=True):
    """For score matrix S (R, n_all) restricted to rows idx: per-rep bootstrap arrays of wAP (per pi), AUROC,
    FPR@TPR0.8 and the plug-in point values. Returns dict of (R, B) arrays and per-rep points."""
    R = S.shape[0]
    yy = y[idx]
    boots = {k: np.empty((R, W.shape[0])) for k in list(pis) + ["native", "AUROC", "FPR_at_TPR80", "TPR_at_TPR80"]}
    pts = {k: np.empty(R) for k in boots}
    one = np.ones((1, len(idx)))
    for r in range(R):
        rk = Ranked(S[r, idx], yy)
        for Wm, tgt, j in ((W, boots, None), (one, pts, 0)):
            tp, fp = rk.tpfp(Wm)
            for k, pi in pis.items():
                v = ap_from(tp, fp, prev_mult(pi, tp, fp))
                if j is None:
                    tgt[k][r] = v
                else:
                    tgt[k][r] = v[0]
            vals = {"native": ap_from(tp, fp, 1.0), "AUROC": auroc_from(tp, fp)}
            if extra:
                f, t = fpr_at_tpr_from(tp, fp, 0.8)
                vals["FPR_at_TPR80"] = f; vals["TPR_at_TPR80"] = t
            for k, v in vals.items():
                if j is None:
                    tgt[k][r] = v
                else:
                    tgt[k][r] = v[0]
    return boots, pts


def rate_block(pred, masks, W, idx=None):
    """pred (R, n) bool; masks {name: (n,) bool}; W (B, n) weights (or None for point only).
    Returns boot {name: (R, B)} and pts {name: (R,)}; rates with an empty denominator are NaN."""
    R = pred.shape[0]
    boot, pts = {}, {}
    for name, m in masks.items():
        mf = m.astype(np.float32)
        pts[name] = np.array([pred[r][m].mean() if m.any() else np.nan for r in range(R)])
        if W is not None:
            den = W @ mf
            num = np.stack([W @ (pred[r] & m).astype(np.float32) for r in range(R)])
            with np.errstate(invalid="ignore", divide="ignore"):
                boot[name] = np.where(den[None, :] > 0, num / np.where(den > 0, den, 1)[None, :], np.nan)
    return boot, pts


def summarize_det(boots, pts, choice, keys=None, ns=None):
    """ns = {rate name: denominator count}; when every repetition's point rate is exactly 0 or exactly 1 the
    bootstrap interval is degenerate, so an exact Clopper-Pearson 95% interval (per repetition) is added."""
    keys = keys or list(boots)
    out = {}
    for k in keys:
        e = est(np.nanmean(pts[k]), pick(boots[k], choice))
        if len(pts[k]) > 1:
            e["point_by_rep"] = rnd(list(pts[k]))
        if ns is not None and k in ns and ns[k] > 0:
            n = ns[k]; p = np.asarray(pts[k], float)
            if np.all(p == 0):
                e["clopper_pearson_95"] = [0.0, rnd(1 - 0.025 ** (1 / n))]
            elif np.all(p == 1):
                e["clopper_pearson_95"] = [rnd(0.025 ** (1 / n)), 1.0]
        out[k] = e
    return out


def zero_fp_bound(pi, tpr_lo, n_neg):
    """Lower bound on Eq. (6) precision when no negative is flagged: CP upper FPR bound with the TPR lower limit."""
    fu = 1 - 0.025 ** (1 / n_neg)
    return rnd(float(eq6(pi, tpr_lo, fu)))


# =============================================================================== M8 (R1): weighted AUPRC, paired, subsampling
def m8_r1(D, TS):
    y = D["y"]; E = np.where(D["poolE"])[0]; yE = y[E]
    npos, nneg = int(yE.sum()), int((yE == 0).sum())
    dets = [d for d in D["dets"]]            # every cached detector (R1_DETS are the requested ones)
    res = {"population": "pool E (benign + hard_negative + attacked); failed attacks excluded",
           "n_pos": npos, "n_neg": nneg, "native_prevalence": rnd(npos / (npos + nneg)),
           "positive_weight": {k: rnd(pi / (1 - pi) * nneg / npos) for k, pi in PIS.items()},
           "bootstrap": f"stratified by class on pool E, B={B}, seed {BASE + 1}; repetition (GRU seed) drawn per replicate",
           "requested_detectors": R1_DETS, "detectors": {}}
    rng = np.random.default_rng(BASE + 1)
    W = w_strata(np.where(yE == 1, "pos", "neg"), B, rng)
    rng_rep = np.random.default_rng(BASE + 1001)
    choice = {d: reps_choice(D["dets"][d]["R"], rng_rep) for d in dets}
    BO, PT = {}, {}
    for d in dets:
        BO[d], PT[d] = ranked_block(D["dets"][d]["s"], y, E, W, PIS)
        res["detectors"][d] = {"wAUPRC": summarize_det(BO[d], PT[d], choice[d], list(PIS) + ["native"]),
                               "AUROC": summarize_det(BO[d], PT[d], choice[d], ["AUROC"])["AUROC"],
                               "FPR_at_test_TPR80": summarize_det(BO[d], PT[d], choice[d], ["FPR_at_TPR80", "TPR_at_TPR80"])}
    paired = {}
    for k in list(PIS) + ["native", "AUROC"]:
        fam = {}
        for a, b in R1_PAIRS:
            fam[f"{a}-minus-{b}"] = delta_est(np.mean(PT[a][k]) - np.mean(PT[b][k]),
                                              pick(BO[a][k], choice[a]) - pick(BO[b][k], choice[b]))
        paired[k] = holm(fam)
    res["paired"] = paired
    res["paired_note"] = ("wAUPRC differences at each pi (and native AP, and AUROC) on the same resampled indices; "
                          "Holm within each pi over the five comparisons (plan 3.6(e)); p_boot = two-sided bootstrap "
                          "p-value 2*(min(#d<=0, #d>=0)+1)/(B+1)")
    # ---- v0.3 subsampling estimator (secondary): expected AP of a test set with n_pos positives
    sub = {}
    for j, (k, pi) in enumerate(PIS.items()):
        np_k = int(round(pi / (1 - pi) * nneg))
        rng = np.random.default_rng(BASE + 10 + j)
        Ws = w_subsample(yE, np_k, nneg, B, rng)
        ch = {d: reps_choice(D["dets"][d]["R"], np.random.default_rng(BASE + 1010 + j * 50 + i)) for i, d in enumerate(dets)}
        bo = {}
        for d in dets:
            S = D["dets"][d]["s"]; R = S.shape[0]
            ap = np.empty((R, B)); au = np.empty((R, B))
            for r in range(R):
                tp, fp = Ranked(S[r, E], yE).tpfp(Ws)
                ap[r] = ap_from(tp, fp, 1.0); au[r] = auroc_from(tp, fp)
            bo[d] = {"AP": pick(ap, ch[d]), "AUROC": pick(au, ch[d])}
        entry = {"n_pos": np_k, "n_neg": nneg, "sample_prevalence": rnd(np_k / (np_k + nneg)),
                 "expected_AP_random_scorer": rnd(chance_ap_exact(np_k, nneg)),
                 "AP": {d: est(np.nan, bo[d]["AP"]) for d in dets},
                 "AUROC": {d: est(np.nan, bo[d]["AUROC"]) for d in dets}}
        for d in dets:        # the v0.3 point estimate is the replicate mean
            entry["AP"][d]["point"] = entry["AP"][d]["boot_mean"]; entry["AUROC"][d]["point"] = entry["AUROC"][d]["boot_mean"]
        pf = {}; pa = {}
        for a, b in R1_PAIRS:
            da = bo[a]["AP"] - bo[b]["AP"]; pf[f"{a}-minus-{b}"] = delta_est(np.nanmean(da), da)
            du = bo[a]["AUROC"] - bo[b]["AUROC"]; pa[f"{a}-minus-{b}"] = delta_est(np.nanmean(du), du)
        entry["paired_AP"] = holm(pf); entry["paired_AUROC"] = holm(pa)
        entry["bootstrap"] = f"v0.3 evalkit-style subsampling: {np_k} positives and {nneg} negatives with replacement, B={B}, seed {BASE + 10 + j}"
        sub[k] = entry
    res["subsampling_v03_secondary"] = sub
    res["subsampling_note"] = ("Secondary quantity (plan 3.1): the replicate mean is the expected AP of a test set with that "
                               "many positives, and the percentile spread is the sampling distribution of such a test "
                               "set, not a confidence interval for AUPRC at prevalence pi. Point = replicate mean (v0.3).")
    # ---- world-cluster sensitivity (plan 3.5)
    worlds = [D["meta"][i]["world"] for i in E]
    Wc, nclu = w_cluster(worlds, B, np.random.default_rng(BASE + 2))
    wc = {"n_clusters": nclu, "bootstrap": f"world clusters of pool E resampled with replacement, B={B}, seed {BASE + 2}; "
                                            "positive weight recomputed per replicate", "detectors": {}}
    bo = {}
    for d in R1_DETS:
        bo[d], _ = ranked_block(D["dets"][d]["s"], y, E, Wc, PIS, extra=False)
        wc["detectors"][d] = {k: est(np.mean(PT[d][k]), pick(bo[d][k], choice[d])) for k in list(PIS) + ["native", "AUROC"]}
    wp = {}
    for k in list(PIS) + ["native", "AUROC"]:
        fam = {f"{a}-minus-{b}": delta_est(np.mean(PT[a][k]) - np.mean(PT[b][k]),
                                           pick(bo[a][k], choice[a]) - pick(bo[b][k], choice[b])) for a, b in R1_PAIRS}
        wp[k] = holm(fam)
    wc["paired"] = wp
    res["world_cluster_sensitivity"] = wc
    # ---- chance level of the weighted estimator (random scorer on the full pool)
    rng = np.random.default_rng(BASE + 3)
    ch = {k: [] for k in list(PIS) + ["native"]}
    one = np.ones((1, len(E)))
    for _ in range(2000):
        tp, fp = Ranked(rng.random(len(E)), yE).tpfp(one)
        for k, pi in PIS.items():
            ch[k].append(ap_from(tp, fp, prev_mult(pi, tp, fp))[0])
        ch["native"].append(ap_from(tp, fp, 1.0)[0])
    res["chance_wAUPRC_random_scorer"] = {k: {"mean": rnd(np.mean(v)), "p2.5": rnd(np.percentile(v, 2.5)),
                                              "p97.5": rnd(np.percentile(v, 97.5))} for k, v in ch.items()}
    res["chance_note"] = "2,000 uniformly random scorers on the R1 pool E (seed %d); weighted estimator as above" % (BASE + 3)
    return res, (W, choice)


# =============================================================================== M8 (2) + m4 + m10 on R1 (same replicates)
def r1_operating_points(D, TS, Wch):
    W, choice = Wch
    y = D["y"]; E = np.where(D["poolE"])[0]; yE = y[E]
    masks = {"TPR": yE == 1, "FPR": yE == 0}
    allpis = dict(PIS); allpis["native"] = float(yE.mean()); allpis.update(LOWPIS)
    out = {"population": "pool E", "bootstrap": "same replicates as M8_R1 (seed %d)" % (BASE + 1),
           "precision_formula": "Eq. (6): pi*TPR/(pi*TPR + (1-pi)*FPR) per replicate; alerts per 1,000 = 1000*(pi*TPR + (1-pi)*FPR)",
           "point_note": "point = plug-in on the full pool E, mean over repetitions (GRU seeds) of the per-repetition value",
           "detectors": {}}
    for d in D["dets"]:
        dd = D["dets"][d]
        ops = [op for op in ("t_r80", "t_val80", "t05") if op in dd["pred"]]
        od = {}
        for op in ops:
            P = dd["pred"][op][:, E]
            bo, pt = rate_block(P, masks, W)
            tb = pick(bo["TPR"], choice[d]); fb = pick(bo["FPR"], choice[d])
            e = summarize_det(bo, pt, choice[d], ["TPR", "FPR"], ns={"TPR": int(masks["TPR"].sum()), "FPR": int(masks["FPR"].sum())})
            zero_fp = bool(np.all(pt["FPR"] == 0))
            for k, pi in allpis.items():
                pr_pt = eq6(pi, pt["TPR"], pt["FPR"]); al_pt = alerts(pi, pt["TPR"], pt["FPR"])
                e[f"precision_{k}"] = est(np.nanmean(pr_pt) if np.any(~np.isnan(pr_pt)) else np.nan, eq6(pi, tb, fb))
                e[f"alerts_per_1000_{k}"] = est(np.mean(al_pt), alerts(pi, tb, fb))
                if dd["R"] > 1:
                    e[f"precision_{k}"]["point_by_rep"] = rnd(list(pr_pt))
                if zero_fp and np.any(pt["TPR"] > 0):
                    e[f"precision_{k}"]["lower_bound_if_FPR_at_CP_upper"] = zero_fp_bound(pi, e["TPR"]["lo"], int(masks["FPR"].sum()))
                if np.all(pt["TPR"] == 0) and zero_fp:
                    e[f"precision_{k}"]["note"] = "no trajectory flagged: precision undefined"
            od[op] = e
        out["detectors"][d] = od
    out["native_prevalence"] = rnd(allpis["native"])
    # ---- m10: FPR budgets from training-negative quantiles
    bud = {"rule": "threshold t = (n-k)-th smallest training-negative score, k = floor(budget*n); flag s > t "
                   "(so at most budget*n training negatives are flagged); scores rounded to 12 decimals",
           "negatives_primary": "benign + hard_negative training rows (the classes of the test pool-E negatives)",
           "negatives_sensitivity": "all label-0 training rows (benign, hard_negative, failed_attack)",
           "test_population": "pool E of the R1 test set", "bootstrap": "same replicates as M8_R1", "detectors": {}}
    for d in R1_DETS:
        sc_te = D["dets"][d]["s"]; R = sc_te.shape[0]
        per_def = {}
        for ndef in ("benign_hard", "all_label0"):
            per_b = {}
            for bk, bv in BUDGETS.items():
                thr, trfpr, preds = [], [], []
                for r in range(R):
                    tn, cat_tr = TS[d][r]
                    tn_sel = tn[np.isin(cat_tr, ["benign", "hard_negative"])] if ndef == "benign_hard" else tn
                    v = np.sort(tn_sel); n = len(v); k = int(math.floor(bv * n))
                    t = v[n - k - 1]
                    thr.append(float(t)); trfpr.append(float(np.mean(tn_sel > t)))
                    preds.append(sc_te[r] > t)
                P = np.array(preds)[:, E]
                bo, pt = rate_block(P, masks, W)
                tb = pick(bo["TPR"], choice[d]); fb = pick(bo["FPR"], choice[d])
                e = {"threshold": rnd(thr if R > 1 else thr[0], 9), "train_FPR": rnd(trfpr if R > 1 else trfpr[0]),
                     "n_train_negatives": int(n),
                     "test_TPR": est(np.mean(pt["TPR"]), tb), "test_FPR": est(np.mean(pt["FPR"]), fb)}
                if R > 1:
                    e["test_TPR"]["point_by_rep"] = rnd(list(pt["TPR"])); e["test_FPR"]["point_by_rep"] = rnd(list(pt["FPR"]))
                for k2, pi in list(PIS.items()) + list(LOWPIS.items()):
                    e[f"precision_{k2}"] = est(np.nanmean(eq6(pi, pt["TPR"], pt["FPR"])), eq6(pi, tb, fb))
                per_b[bk] = e
            per_def[ndef] = per_b
        bud["detectors"][d] = per_def
    out["m10_fpr_budgets"] = bud
    return out


def raw_vs_rounded_budget(D, TS):
    """m9 check for the new thresholds: do raw (unrounded) scores give the same budget predictions?"""
    diff = {}
    for d in R1_DETS:
        raw = D["dets"][d]["raw"]; sr = D["dets"][d]["s"]; R = raw.shape[0]
        n_diff = 0
        for r in range(R):
            tn_r, cat = TS[d][r]
            tn_raw = TS[d + "__raw"][r][0]
            for bv in BUDGETS.values():
                for sel in (np.isin(cat, ["benign", "hard_negative"]), np.ones(len(cat), bool)):
                    v1 = np.sort(tn_r[sel]); v2 = np.sort(tn_raw[sel]); n = len(v1); k = int(math.floor(bv * n))
                    n_diff += int(np.sum((sr[r] > v1[n - k - 1]) != (raw[r] > v2[n - k - 1])))
        diff[d] = n_diff
    return diff


# =============================================================================== AgentDojo (R4 transfer, WUT) weighted AUPRC
def m8_dojo(DR4, DWUT, skey="s"):
    """skey = "s" (scores rounded to 12 decimals, primary per m9) or "raw" (cache tie-exact scores, sensitivity)."""
    y = DR4["y"]; E = np.where(DR4["poolE"])[0]; yE = y[E]
    assert np.array_equal(DWUT["y"], y) and np.array_equal(DWUT["poolE"], DR4["poolE"])
    npos, nneg = int(yE.sum()), int((yE == 0).sum())
    pis = dict(PIS)
    res = {"population": "AgentDojo P0 (392 benign, 2,294 compromised); unsuccessful attacks excluded",
           "n_pos": npos, "n_neg": nneg, "native_prevalence": rnd(npos / (npos + nneg)),
           "positive_weight": {k: rnd(pi / (1 - pi) * nneg / npos) for k, pi in pis.items()},
           "requested": "pi = 5% and 10% (review m17 item 3 / M8); 1% and 2% also given because reweighting uses all positives",
           "score_version": ("rounded to 12 decimals" if skey == "s" else "raw cache scores (tie-exact, unrounded)"),
           "schemes": {}}
    keys = [DR4["meta"][i]["task"] for i in E]
    for scheme in ("traj_strat", "cluster_suite_user_task"):
        if scheme == "traj_strat":
            W = w_strata(np.where(yE == 1, "pos", "neg"), B, np.random.default_rng(BASE + 20)); nclu = None
            desc = f"stratified by class on P0, B={B}, seed {BASE + 20} (plan 3.1)"
        else:
            W, nclu = w_cluster(keys, B, np.random.default_rng(BASE + 21))
            desc = f"{nclu} suite|user_task clusters resampled, B={B}, seed {BASE + 21}; positive weight recomputed per replicate (plan 3.5)"
        rng_rep = np.random.default_rng(BASE + 1020 + (scheme != "traj_strat"))
        BO, PT, CH = {}, {}, {}
        sc = {"bootstrap": desc, "settings": {}}
        for sname, D in (("R4", DR4), ("WUT", DWUT)):
            ent = {}
            for d in DOJO_DETS:
                dd = D["dets"][d]
                CH[(sname, d)] = reps_choice(dd["R"], rng_rep)
                BO[(sname, d)], PT[(sname, d)] = ranked_block(dd[skey], y, E, W, pis, extra=False)
                ent[d] = summarize_det(BO[(sname, d)], PT[(sname, d)], CH[(sname, d)], list(pis) + ["native", "AUROC"])
            pr = {}
            for k in list(pis) + ["native", "AUROC"]:
                fam = {f"{a}-minus-{b}": delta_est(np.mean(PT[(sname, a)][k]) - np.mean(PT[(sname, b)][k]),
                                                   pick(BO[(sname, a)][k], CH[(sname, a)]) - pick(BO[(sname, b)][k], CH[(sname, b)]))
                       for a, b in DOJO_PAIRS}
                pr[k] = holm(fam)
            sc["settings"][sname] = {"detectors": ent, "paired": pr}
        wt = {}
        for k in list(pis) + ["native", "AUROC"]:
            fam = {d: delta_est(np.mean(PT[("WUT", d)][k]) - np.mean(PT[("R4", d)][k]),
                                pick(BO[("WUT", d)][k], CH[("WUT", d)]) - pick(BO[("R4", d)][k], CH[("R4", d)]))
                   for d in DOJO_DETS}
            wt[k] = holm(fam)
        sc["within_minus_transfer"] = wt
        sc["within_minus_transfer_note"] = "WUT minus R4 on the same P0 rows and replicates; Holm over the five detectors at each pi (plan 3.6(d))"
        res["schemes"][scheme] = sc
    sub = {}
    for j, (k, pi) in enumerate([("pi05", 0.05), ("pi10", 0.10)]):
        np_k = int(round(pi / (1 - pi) * nneg))
        Ws = w_subsample(yE, np_k, nneg, B, np.random.default_rng(BASE + 30 + j))
        rng_rep = np.random.default_rng(BASE + 1030 + j)
        e = {"n_pos": np_k, "n_neg": nneg, "expected_AP_random_scorer": rnd(chance_ap_exact(np_k, nneg)), "AP": {}}
        for sname, D in (("R4", DR4), ("WUT", DWUT)):
            e["AP"][sname] = {}
            for d in DOJO_DETS:
                S = D["dets"][d][skey]; R = S.shape[0]; ap = np.empty((R, B))
                for r in range(R):
                    tp, fp = Ranked(S[r, E], yE).tpfp(Ws); ap[r] = ap_from(tp, fp, 1.0)
                v = pick(ap, reps_choice(R, rng_rep)); x = est(np.nan, v); x["point"] = x["boot_mean"]
                e["AP"][sname][d] = x
        sub[k] = e
    res["subsampling_v03_secondary"] = sub
    return res


# =============================================================================== M9 breakdown (+ theta=0.5 and theta_val80 tables)
def m9_masks(D):
    cat, pat, y, E = D["cat"], D["pat"], D["y"], D["poolE"]
    m = {"recall_poolE": E & (y == 1), "FPR_poolE": E & (y == 0), "FPR_benign": cat == "benign"}
    if D["bench"] == "agentdrift":
        m["FPR_hard_negative"] = cat == "hard_negative"
        m["flag_failed_attack"] = cat == "failed_attack"
        for p in PATTERNS:
            m["recall_" + p.split("_")[1]] = pat == p
    else:
        m["flag_failed_attack"] = cat == "attack_failed"
    return m


def subgroup_auroc(S, D, masks, W):
    """AUROC of subgroup positives against a negative class, per repetition, on bootstrap weights W (all rows)."""
    y = D["y"]
    pos_groups = {"poolE": masks["recall_poolE"]}
    for p in ("full", "partial", "delayed"):
        if f"recall_{p}" in masks:
            pos_groups[p] = masks[f"recall_{p}"]
    neg_groups = {"poolE_neg": masks["FPR_poolE"], "failed": masks["flag_failed_attack"]}
    if "FPR_hard_negative" in masks:          # AgentDrift; for AgentDojo benign = pool-E negatives
        neg_groups["benign"] = masks["FPR_benign"]
        neg_groups["hard_negative"] = masks["FPR_hard_negative"]
    combos = [(pg, ng) for pg in pos_groups for ng in neg_groups if pg == "poolE" or ng == "poolE_neg"]
    R = S.shape[0]
    boot, pts = {}, {}
    for pg, ng in combos:
        idx = np.where(pos_groups[pg] | neg_groups[ng])[0]
        yy = (pos_groups[pg][idx]).astype(int)
        k = f"{pg}_vs_{ng}"
        boot[k] = np.empty((R, W.shape[0])); pts[k] = np.empty(R)
        Wi = W[:, idx]
        for r in range(R):
            rk = Ranked(S[r, idx], yy)
            tp, fp = rk.tpfp(Wi); boot[k][r] = auroc_from(tp, fp)
            tp, fp = rk.tpfp(np.ones((1, len(idx)))); pts[k][r] = auroc_from(tp, fp)[0]
    return boot, pts


def m9_setting(D, si):
    masks = m9_masks(D)
    strata = np.array([c + ("|" + p if p else "") for c, p in zip(D["cat"], D["pat"])])
    comp = {k: int(v.sum()) for k, v in masks.items()}
    out = {"n_rows": {k: v for k, v in comp.items()},
           "hard_negative_share_of_poolE_negatives": (rnd(comp.get("FPR_hard_negative", 0) / comp["FPR_poolE"])),
           "schemes": {}}
    schemes = [("traj_strat", None)]
    if D["setting"] in CLUSTER_OF:
        schemes.append(("cluster_" + ("suite_user_task" if D["bench"] == "agentdojo" else "task"), CLUSTER_OF[D["setting"]]))
    for sname, ckey in schemes:
        if ckey is None:
            W = w_strata(strata, B, np.random.default_rng(BASE + 100 + si)); nclu = None
            desc = f"stratified by category x hijack pattern over all test rows (incl. failed attacks), B={B}, seed {BASE + 100 + si}"
        else:
            W, nclu = w_cluster([m[ckey] for m in D["meta"]], B, np.random.default_rng(BASE + 200 + si))
            desc = f"{nclu} {'suite|user_task' if D['bench'] == 'agentdojo' else 'task'} clusters over all test rows, B={B}, seed {BASE + 200 + si} (plan 3.5)"
        rng_rep = np.random.default_rng(BASE + 300 + si)
        dets = {}
        CH, SUBB, SUBP = {}, {}, {}
        for d, dd in D["dets"].items():
            ch = reps_choice(dd["R"], rng_rep); CH[d] = ch
            od = {}
            for op in [op for op in ("t_r80", "t05", "t_val80") if op in dd["pred"]]:
                bo, pt = rate_block(dd["pred"][op], masks, W)
                e = summarize_det(bo, pt, ch, ns=comp)
                nf = [int(dd["pred"][op][r][D["poolE"]].sum()) for r in range(dd["R"])]
                e["n_flagged_poolE"] = float(np.mean(nf))
                if dd["R"] > 1:
                    e["n_flagged_poolE_by_rep"] = nf
                tb = pick(bo["recall_poolE"], ch)
                prec = {}
                mixes = {"benign_only": "FPR_benign", "poolE_mix": "FPR_poolE", "failed_only": "flag_failed_attack"}
                if "FPR_hard_negative" in masks:
                    mixes["hard_only"] = "FPR_hard_negative"
                for pk in ("pi02", "pi10"):
                    pi = PIS[pk]
                    prec[pk] = {}
                    for mix, fk in mixes.items():
                        pp = eq6(pi, pt["recall_poolE"], pt[fk])
                        x = est(np.nanmean(pp) if np.any(~np.isnan(pp)) else np.nan, eq6(pi, tb, pick(bo[fk], ch)))
                        if dd["R"] > 1:
                            x["point_by_rep"] = rnd(list(pp))
                        if np.all(pt[fk] == 0) and np.any(pt["recall_poolE"] > 0):
                            x["lower_bound_if_FPR_at_CP_upper"] = zero_fp_bound(pi, e["recall_poolE"]["lo"], comp[fk])
                        if np.all(pt[fk] == 0) and np.all(pt["recall_poolE"] == 0):
                            x["note"] = "no trajectory flagged: precision undefined"
                        prec[pk][mix] = x
                e["precision_eq6"] = prec
                od[op] = e
            # ---- rank-based subgroup AUROC (positives of a subgroup against negatives of a class)
            versions = [("s", "AUROC_subgroups")] + ([("raw", "AUROC_subgroups_raw_scores")] if D["bench"] == "agentdojo" else [])
            for skey, name in (versions if ckey is None else []):     # rank-based subgroups: trajectory scheme only
                bo, pt = subgroup_auroc(dd[skey], D, masks, W)
                od[name] = summarize_det(bo, pt, ch)
                SUBB[(d, skey)], SUBP[(d, skey)] = bo, pt
            dets[d] = od
        pairs = {}
        for a, b in ([("LR-C", "LR-B"), ("GRU", "LR-C")] if ckey is None else []):
            if a in D["dets"] and b in D["dets"]:
                pairs[f"{a}-minus-{b}"] = {k: delta_est(np.mean(SUBP[(a, "s")][k]) - np.mean(SUBP[(b, "s")][k]),
                                                        pick(SUBB[(a, "s")][k], CH[a]) - pick(SUBB[(b, "s")][k], CH[b]))
                                           for k in SUBB[(a, "s")]}
        out["schemes"][sname] = {"bootstrap": desc, "detectors": dets, "paired_AUROC_subgroups": pairs}
    out["mix_note"] = ("precision_eq6 uses the pooled pool-E recall and the FPR of the stated negative mix: benign_only, "
                       "poolE_mix (the test pool's own benign/hard-negative composition), hard_only (AgentDrift), and "
                       "failed_only (unsuccessful attacks treated as negatives; extra). For AgentDojo pool E has benign "
                       "negatives only, so benign_only = poolE_mix.")
    return out


# =============================================================================== m11: label-free threshold
def labelfree(D1, DR3s):
    E1 = D1["poolE"]
    out = {"rule": (f"threshold = quantile (numpy linear) at 1 - a of the scores of {N_UNLAB} unlabeled held-out pool-E "
                    f"traces drawn without replacement, where a = the detector's R1 alert rate at theta_r80 on the R1 "
                    f"test pool E; flag s >= threshold; FPR and recall on the remaining held-out pool-E traces; "
                    f"{N_DRAWS} draws with default_rng({BASE + 500} + draw), shared by all detectors of a fold; "
                    "GRU: each seed uses its own R1 alert rate"),
           "sensitivity_all_rows": ("a = R1 alert rate over all R1 test rows (incl. failed attacks); draws from all held-out "
                                    "rows; FPR/recall on the remaining pool-E rows; failed-attack flag rate on the remaining failed rows"),
           "deployment_prev02": (f"unlabeled traffic at 2% prevalence: each draw has {int(round(0.02 * N_UNLAB))} attacked and "
                                 f"{N_UNLAB - int(round(0.02 * N_UNLAB))} negative held-out pool-E traces (default_rng({BASE + 600} + draw)); "
                                 "a = the R1 alert rate at 2% prevalence, 0.02*TPR_R1 + 0.98*FPR_R1 at theta_r80 on the R1 pool E "
                                 "(alerts per 1,000 at 2% / 1000); FPR/recall on the remaining held-out pool-E traces. Added after "
                                 "seeing that the native-prevalence reading does not reproduce the review's m11 numbers (deviation P9)"),
           "folds": {}}
    targets = {}
    for d in LABELFREE_DETS:
        P = D1["dets"][d]["pred"]["t_r80"]
        y1 = D1["y"]
        tpr1 = [float(P[r][E1 & (y1 == 1)].mean()) for r in range(P.shape[0])]
        fpr1 = [float(P[r][E1 & (y1 == 0)].mean()) for r in range(P.shape[0])]
        targets[d] = {"poolE": [float(P[r][E1].mean()) for r in range(P.shape[0])],
                      "all_rows": [float(P[r].mean()) for r in range(P.shape[0])],
                      "prev02": [0.02 * a + 0.98 * b for a, b in zip(tpr1, fpr1)]}
    out["R1_alert_rate_targets"] = {d: {k: rnd(v) for k, v in t.items()} for d, t in targets.items()}
    for fold, D in DR3s.items():
        y = D["y"]; E = np.where(D["poolE"])[0]; allr = np.arange(len(y))
        fail = D["cat"] == "failed_attack"; ben = D["cat"] == "benign"; hard = D["cat"] == "hard_negative"
        fo = {"n_poolE": int(len(E)), "detectors": {}}
        Epos = E[y[E] == 1]; Eneg = E[y[E] == 0]
        npos02 = int(round(0.02 * N_UNLAB))
        for variant, pool, tkey in (("primary_poolE", E, "poolE"), ("sensitivity_all_rows", allr, "all_rows"),
                                    ("deployment_prev02", None, "prev02")):
            if pool is not None:
                draws = [np.random.default_rng(BASE + 500 + k).choice(pool, N_UNLAB, replace=False) for k in range(N_DRAWS)]
            else:
                draws = []
                for k in range(N_DRAWS):
                    g = np.random.default_rng(BASE + 600 + k)
                    draws.append(np.concatenate([g.choice(Epos, npos02, replace=False), g.choice(Eneg, N_UNLAB - npos02, replace=False)]))
            for d in LABELFREE_DETS:
                S = D["dets"][d]["s"]; Sraw = D["dets"][d]["raw"]; R = S.shape[0]
                rows = []
                n_pred_diff_raw = 0
                for r in range(R):
                    a = targets[d][tkey][r]
                    for k, dr in enumerate(draws):
                        t = float(np.quantile(S[r, dr], 1 - a))
                        t_raw = float(np.quantile(Sraw[r, dr], 1 - a))
                        rest = np.ones(len(y), bool); rest[dr] = False
                        flag = S[r] >= t
                        n_pred_diff_raw += int(np.sum(flag != (Sraw[r] >= t_raw)))
                        pe = rest & D["poolE"]
                        rec = float(flag[pe & (y == 1)].mean()); fpr = float(flag[pe & (y == 0)].mean())
                        rows.append({"rep": r, "draw": k, "threshold": rnd(t, 9), "target_alert_rate": rnd(a),
                                     "alert_rate_draw": rnd(float((S[r, dr] >= t).mean())),
                                     "alert_rate_rest_poolE": rnd(float(flag[pe].mean())),
                                     "recall": rnd(rec), "FPR": rnd(fpr),
                                     "FPR_benign": rnd(float(flag[pe & ben].mean())),
                                     "FPR_hard_negative": rnd(float(flag[pe & hard].mean())),
                                     "flag_failed_attack": rnd(float(flag[fail & rest].mean())),
                                     "precision_pi02": rnd(float(eq6(0.02, rec, fpr))), "precision_pi10": rnd(float(eq6(0.10, rec, fpr)))})
                summ = {}
                for q in ("recall", "FPR", "FPR_benign", "FPR_hard_negative", "flag_failed_attack", "alert_rate_rest_poolE",
                          "precision_pi02", "precision_pi10"):
                    v = np.array([x[q] for x in rows], float)
                    summ[q] = {"mean": rnd(np.nanmean(v)), "min": rnd(np.nanmin(v)), "max": rnd(np.nanmax(v))}
                    if R > 1:
                        summ[q]["mean_by_rep"] = rnd([float(np.nanmean([x[q] for x in rows if x["rep"] == r])) for r in range(R)])
                P = D["dets"][d]["pred"]["t_r80"]
                ref = {"recall": rnd(float(np.mean([P[r][D["poolE"] & (y == 1)].mean() for r in range(R)]))),
                       "FPR": rnd(float(np.mean([P[r][D["poolE"] & (y == 0)].mean() for r in range(R)]))),
                       "flag_failed_attack": rnd(float(np.mean([P[r][fail].mean() for r in range(R)])))}
                fo["detectors"].setdefault(d, {})[variant] = {"summary_over_draws": summ, "draws": rows,
                                                              "theta_r80_same_fold_full_poolE": ref,
                                                              "n_pred_changed_by_rounding": n_pred_diff_raw}
        out["folds"][fold] = fo
    return out


# =============================================================================== m9: ties / rounding check
def ties_check(Ds):
    from scipy.stats import rankdata
    out = {}
    for s, D in Ds.items():
        E = np.where(D["poolE"])[0]; yE = D["y"][E]
        o = {}
        for d, dd in D["dets"].items():
            same = True; mx = 0.0
            au_raw, au_rnd, nd_raw, nd_rnd, n_small = [], [], [], [], []
            for r in range(dd["R"]):
                a = dd["raw"][r, E]; b = dd["s"][r, E]
                same &= bool(np.array_equal(rankdata(a, "dense"), rankdata(b, "dense")))
                one = np.ones((1, len(E)))
                t1, f1 = Ranked(a, yE).tpfp(one); t2, f2 = Ranked(b, yE).tpfp(one)
                for pi in PIS.values():
                    mx = max(mx, abs(ap_from(t1, f1, prev_mult(pi, t1, f1))[0] - ap_from(t2, f2, prev_mult(pi, t2, f2))[0]))
                mx = max(mx, abs(auroc_from(t1, f1)[0] - auroc_from(t2, f2)[0]), abs(ap_from(t1, f1, 1.0)[0] - ap_from(t2, f2, 1.0)[0]))
                au_raw.append(float(auroc_from(t1, f1)[0])); au_rnd.append(float(auroc_from(t2, f2)[0]))
                nd_raw.append(int(len(np.unique(a)))); nd_rnd.append(int(len(np.unique(b))))
                n_small.append(int(np.sum((np.abs(a) < 5e-13) & (a != 0))))
            o[d] = {"order_and_ties_identical": same, "max_abs_change_point_AUROC_wAUPRC": float(mx)}
            if not same:
                o[d].update({"AUROC_raw_by_rep": rnd(au_raw), "AUROC_rounded_by_rep": rnd(au_rnd),
                             "n_distinct_raw_by_rep": nd_raw, "n_distinct_rounded_by_rep": nd_rnd,
                             "n_nonzero_scores_below_5e-13_by_rep": n_small,
                             "cause": "distinct scores below 5e-13 (saturated probabilities) become tied at 0 after rounding"
                                      if sum(n_small) else "rounding merges distinct scores"})
        out[s] = o
    return out


# =============================================================================== self tests
def self_tests():
    from sklearn.metrics import average_precision_score
    rng = np.random.default_rng(1)
    res = {}
    mx_ap = mx_w = mx_au = mx_sub = 0.0
    for _ in range(50):
        n = 300; y = (rng.random(n) < 0.4).astype(int)
        s = np.round(rng.normal(size=n) + y, 1)                  # many ties
        w = rng.integers(0, 4, size=n).astype(float)
        if w[y == 1].sum() == 0 or w[y == 0].sum() == 0:
            continue
        tp, fp = Ranked(s, y).tpfp(np.ones((1, n)))
        mx_ap = max(mx_ap, abs(ap_from(tp, fp, 1.0)[0] - average_precision_score(y, s)))
        mx_au = max(mx_au, abs(auroc_from(tp, fp)[0] - ek.auroc(y, s)))
        c = 0.3
        sw = np.where(y == 1, c, 1.0)
        mx_w = max(mx_w, abs(ap_from(tp, fp, c)[0] - average_precision_score(y, s, sample_weight=sw)))
        tp, fp = Ranked(s, y).tpfp(w[None, :])
        ib = np.repeat(np.arange(n), w.astype(int))
        mx_sub = max(mx_sub, abs(ap_from(tp, fp, 1.0)[0] - average_precision_score(y[ib], s[ib])),
                     abs(auroc_from(tp, fp)[0] - ek.auroc(y[ib], s[ib])))
    res = {"AP_vs_sklearn": mx_ap, "weighted_AP_vs_sklearn_sample_weight": mx_w, "AUROC_vs_evalkit": mx_au,
           "multiplicity_weights_vs_explicit_resample": mx_sub}
    assert max(res.values()) < 1e-10, res
    return res


# =============================================================================== stage 2: analyze
class BudgetExceeded(Exception):
    pass


def analysis_code_sha():
    """Identifies the analysis code and inputs a part file was computed with: source of the analysis functions,
    the module constants, the cache pickles and the R1 training-score part."""
    import inspect
    funcs = [load_setting, _counts, w_strata, w_subsample, w_cluster, Ranked, ap_from, auroc_from, fpr_at_tpr_from,
             prev_mult, est, delta_est, holm, pick, eq6, alerts, chance_ap_exact, chance_ap_mc, reps_choice, ranked_block,
             rate_block, summarize_det, zero_fp_bound, m8_r1, r1_operating_points, raw_vs_rounded_budget, m8_dojo,
             m9_masks, subgroup_auroc, m9_setting, labelfree, ties_check]
    h = hashlib.sha256()
    for f in funcs:
        h.update(inspect.getsource(f).encode())
    h.update(repr((B, ROUND, BASE, PIS, LOWPIS, R1_DETS, R1_PAIRS, DOJO_DETS, DOJO_PAIRS, M9_SETTINGS, BUDGETS,
                   LABELFREE_DETS, N_UNLAB, N_DRAWS, CLUSTER_OF, PATTERNS)).encode())
    for s in M9_SETTINGS:
        for fam in ("linear", "gru"):
            f = os.path.join(CACHE, fam, f"{s}.pkl")
            if os.path.exists(f):
                h.update(sha256(f).encode())
    h.update(sha256(TRAIN_PKL).encode())
    return h.hexdigest()


def part(name, fn, st):
    """Resumable unit: reuse prevalence_ops_parts/analyze_<name>.pkl if it was computed with the same analysis code and
    inputs; otherwise compute it, unless the time budget of this invocation is used up."""
    path = os.path.join(PARTS, f"analyze_{name}.pkl")
    if os.path.exists(path):
        obj = pickle.load(open(path, "rb"))
        if obj.get("code_sha") == st["sha"]:
            st["runtime"][name] = obj["runtime_s"]
            return obj["value"]
    if time.time() - st["t0"] > st["budget"]:
        raise BudgetExceeded(name)
    t1 = time.time()
    v = fn()
    rt = round(time.time() - t1, 1)
    pickle.dump({"code_sha": st["sha"], "value": v, "runtime_s": rt}, open(path, "wb"))
    st["runtime"][name] = rt
    log(f"part {name} computed ({rt} s)")
    return v


def stage_analyze(budget=480.0):
    t0 = time.time()
    os.makedirs(PARTS, exist_ok=True)
    st = {"t0": t0, "budget": budget, "sha": analysis_code_sha(), "runtime": {}}
    tests = self_tests(); log(f"self tests pass {tests}")
    TSobj = pickle.load(open(TRAIN_PKL, "rb"))
    assert all(v["pass"] for v in TSobj["verification"].values())
    Ds = {s: load_setting(s) for s in M9_SETTINGS}
    log("caches loaded")
    D1 = Ds["R1"]
    # training-negative scores per detector and repetition: list of (scores, category)
    TS = {}
    cat_tr = np.asarray(TSobj["train_category"])
    for d in ["LR-A", "LR-B", "LR-C", "Markov", "IForest", "TFIDF"]:
        raw = np.asarray(TSobj["train_scores"][d], float)
        neg = TSobj["train_y"] == 0
        TS[d] = [(np.round(raw[neg], ROUND), cat_tr[neg])]
        TS[d + "__raw"] = [(raw[neg], cat_tr[neg])]
    g = D1["gru"]
    TS["GRU"], TS["GRU__raw"] = [], []
    gru_thr_check = []
    for r in range(3):
        lg = (np.asarray(g["train_tool_contrib"]["GRU"][r], float) + np.asarray(g["train_beh_contrib"]["GRU"][r], float)
              + float(g["bias"]["GRU"][r]))
        p = 1.0 / (1.0 + np.exp(-lg)); ty = np.asarray(g["train_y"]["GRU"][r]).astype(int)
        tc = np.asarray(g["train_category"]["GRU"][r]).astype(str)
        gru_thr_check.append(abs(float(np.quantile(p[ty == 1], 0.20)) - float(g["thr"]["GRU"]["t_r80"][r])))
        TS["GRU"].append((np.round(p[ty == 0], ROUND), tc[ty == 0])); TS["GRU__raw"].append((p[ty == 0], tc[ty == 0]))
    R = {"meta": {"script": os.path.abspath(__file__), "script_sha256": sha256(os.path.abspath(__file__)),
                  "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "python": platform.python_version(),
                  "numpy": np.__version__, "B": B, "round_decimals": ROUND, "seed_base": BASE,
                  "caches": {s: {"linear_sha256": sha256(os.path.join(CACHE, "linear", f"{s}.pkl")),
                                 "gru_sha256": (sha256(os.path.join(CACHE, "gru", f"{s}.pkl"))
                                                if os.path.exists(os.path.join(CACHE, "gru", f"{s}.pkl")) else None)}
                             for s in M9_SETTINGS},
                  "self_tests_max_abs_error": tests,
                  "R1_train_score_refit": {"verification": TSobj["verification"],
                                           "build_linear_cache_sha256": TSobj["build_linear_cache_sha256"],
                                           "GRU_train_scores": "sigmoid(train_tool_contrib + train_beh_contrib + bias), fit partition (90%)",
                                           "GRU_reconstructed_t_r80_abs_diff_by_seed": [float(x) for x in gru_thr_check]}}}
    # ---- M8 R1 (+ Eq. (6), m4 matched recall, m10 budgets on the same replicates)
    def r1_block():
        m8, Wch = m8_r1(D1, TS)
        ops = r1_operating_points(D1, TS, Wch)
        ops["m10_fpr_budgets"]["rounding_changes_budget_predictions"] = raw_vs_rounded_budget(D1, TS)
        return m8, ops
    try:
        R["M8_R1"], R["R1_operating_points"] = part("R1", r1_block, st)

        # ---- chance AP (m9)
        def chance_block():
            rng = np.random.default_rng(BASE + 40)
            ca = {}
            for npos, nneg in [(8, 825), (17, 825), (43, 825), (92, 825), (44, 392), (21, 392)]:
                mu, lo, hi = chance_ap_mc(npos, nneg, 20000, rng)
                ca[f"{npos}/{nneg}"] = {"exact": rnd(chance_ap_exact(npos, nneg)), "monte_carlo_mean": rnd(mu),
                                        "monte_carlo_p2.5": rnd(lo), "monte_carlo_p97.5": rnd(hi),
                                        "prevalence": rnd(npos / (npos + nneg))}
            return {"formula": "E[AP] = (n+ - 1)/(N - 1) + H_N (N - n+)/(N (N - 1)), N = n+ + n-, random ranking without ties",
                    "monte_carlo": f"20,000 random rankings each, seed {BASE + 40}", "values": ca}
        R["m9_chance_AP"] = part("chance_AP", chance_block, st)
        # ---- AgentDojo
        R["M8_AgentDojo"] = part("AgentDojo_rounded", lambda: m8_dojo(Ds["R4"], Ds["WUT"], "s"), st)
        R["M8_AgentDojo_raw_scores"] = part("AgentDojo_raw", lambda: m8_dojo(Ds["R4"], Ds["WUT"], "raw"), st)
        # ---- M9
        R["M9"] = {}
        for si, s in enumerate(M9_SETTINGS):
            R["M9"][s] = part(f"M9_{s}", lambda s=s, si=si: m9_setting(Ds[s], si), st)
        R["m11_label_free"] = part("m11", lambda: labelfree(D1, {d: Ds[f"R3_{d}"] for d in DOMAINS}), st)
        R["m9_rounding_check"] = part("ties", lambda: ties_check(Ds), st)
    except BudgetExceeded as ex:
        log(f"INCOMPLETE: time budget used before part {ex}; run 'analyze' again to continue")
        return False
    # ---- m4 theta = 0.5 table and GRU theta_val80 (views into M9, traj_strat)
    t05 = {}
    for s in THETA05_SETTINGS:
        t05[s] = {}
        for d in THETA05_DETS:
            x = R["M9"][s]["schemes"]["traj_strat"]["detectors"][d]["t05"]
            t05[s][d] = {k: x[k] for k in ("recall_poolE", "FPR_poolE", "flag_failed_attack")}
    R["m4_theta05"] = {"source": "M9.<setting>.schemes.traj_strat.detectors.<det>.t05", "table": t05}
    gv = {}
    for s in M9_SETTINGS:
        for d in [d for d in Ds[s]["dets"] if Ds[s]["dets"][d]["family"] == "gru"]:
            x = R["M9"][s]["schemes"]["traj_strat"]["detectors"][d]
            gv.setdefault(s, {})[d] = {op: {"recall_poolE": x[op]["recall_poolE"], "FPR_poolE": x[op]["FPR_poolE"],
                                            "precision_pi02_poolE_mix": x[op]["precision_eq6"]["pi02"]["poolE_mix"],
                                            "precision_pi10_poolE_mix": x[op]["precision_eq6"]["pi10"]["poolE_mix"]}
                                       for op in ("t_r80", "t_val80")}
    R["m4_gru_t_val80"] = {"source": "M9.<setting>.schemes.traj_strat.detectors.<GRU det>", "note":
                           "t_val80 = 20th percentile of validation-positive scores (validation = stratified 10% of the "
                           "training set, gru_ladder.stratified_val; deviation G1)", "table": gv}
    R["deviations"] = DEVIATIONS
    R["meta"]["analysis_code_sha256"] = st["sha"]
    R["meta"]["part_runtime_s"] = st["runtime"]
    R["meta"]["runtime_s"] = round(sum(st["runtime"].values()), 1)
    json.dump(R, open(OUT_JSON, "w"), indent=1)
    log(f"wrote {OUT_JSON} (analysis parts {R['meta']['runtime_s']} s; this call {round(time.time() - t0, 1)} s)")
    return True


DEVIATIONS = [
    "P1 (interpretation, AgentDojo and R2 intervals). Plan 3.1 asks for a stratified full-pool bootstrap for AUPRC at "
    "controlled prevalence; plan 3.5 asks for a (suite, user task) cluster bootstrap for the AgentDojo UT analyses and a "
    "task cluster bootstrap for R2. Both are reported: 'traj_strat' (plan 3.1, the scheme named in the task) and "
    "'cluster_*' (plan 3.5). R1 and R3 use the trajectory-level bootstrap; R1 also has a world-cluster sensitivity.",
    "P2 (interpretation, m10 training negatives). The plan does not say which training rows are 'training negatives'. "
    "Primary: benign + hard_negative training rows (the classes of the test pool-E negatives); sensitivity: all label-0 "
    "training rows (incl. failed attacks). Threshold rule: flag s > t, t = (n-k)-th smallest training-negative score, "
    "k = floor(budget*n). The scores are in-sample for the supervised detectors (as for theta_r80).",
    "P3 (new fits). The cache stores no training scores for the linear detectors. The R1 linear detectors were refitted "
    "with the unmodified cache code (build_linear_cache.py functions, imported) only to recover in-sample training "
    "scores; the refit reproduces the cached test scores (max |diff| reported in meta.R1_train_score_refit) and t_r80. "
    "GRU training scores are reconstructed in float64 from the cached logit decomposition of the 90% fit partition; the "
    "reconstructed t_r80 differs from the cached float32 value by < 1e-7.",
    "P4 (interpretation, m11). The 'R1 alert rate' is the detector's alert rate at theta_r80 on the R1 test pool E "
    "(GRU: per seed). The unlabeled draws come from the held-out pool E; a sensitivity uses all rows (incl. failed "
    "attacks) both for the R1 alert rate and for the draws. LR-B and the GRU are added to LR-C as the task requests.",
    "P5 (M9 bootstrap strata). The M9 breakdown resamples each stratum category x hijack pattern (and the failed attacks) "
    "separately, so the class composition is fixed as in the test set; the M8 blocks stratify by class (pool E) as evalkit.",
    "P6 (point estimates). Point estimates are plug-in values on the full pool (mean over GRU seeds or CV repetitions); "
    "bootstrap means are reported next to them. For the secondary subsampling estimator the point is the replicate mean (v0.3).",
    "P7 (FPR at a fixed test TPR of 0.8, plan 3.8). This uses test labels to place the threshold (matched recall); it is a "
    "descriptive comparison, not a deployable operating point. The threshold is re-chosen inside each bootstrap replicate.",
    "P8 (precision under negative mixes). Precision uses the pooled pool-E recall of the setting with the FPR of each "
    "negative mix. For AgentDojo pool E has benign negatives only; 'failed_only' (unsuccessful attacks as negatives) is "
    "added as an extra mix in every setting.",
    "P9 (post hoc, m11). The pre-specified reading (P4: native-prevalence unlabeled sample, native R1 alert rate) does not "
    "reproduce the review's m11 figures (FPR 0.191-0.209). A second reading was added after seeing this: unlabeled traffic "
    "at 2% prevalence (4 attacked + 196 negative held-out traces per draw) with the R1 alert rate at 2% prevalence "
    "(0.02*TPR + 0.98*FPR at theta_r80, i.e. the alerts per 1,000 at 2% of Table IV / 1000). Both are reported "
    "('primary_poolE', 'deployment_prev02'); the second is labelled post hoc.",
    "P10 (m9 rounding). As instructed, scores are rounded to 12 decimals before every ranking. This changes no value in "
    "AgentDrift settings, but in R4 and WUT it merges distinct saturated scores below 5e-13 into ties at 0 (e.g. R4 LR-C "
    "AUROC 0.4733 raw vs 0.4697 rounded; see m9_rounding_check). For AgentDojo every rank-based M8 quantity is therefore "
    "also given with the raw (cache tie-exact) scores in M8_AgentDojo_raw_scores (same replicates), and the M9 subgroup "
    "AUROCs in AUROC_subgroups_raw_scores; the raw values match the cache validation files. Thresholded rates are unaffected "
    "(they use the cached predictions).",
    "P11 (degenerate intervals). When a rate is exactly 0 or 1 in every repetition, the bootstrap interval collapses; an "
    "exact Clopper-Pearson 95% interval is added ('clopper_pearson_95'), and for precision with zero false positives a "
    "lower bound from the CP upper FPR limit and the bootstrap TPR lower limit ('lower_bound_if_FPR_at_CP_upper').",
]


# =============================================================================== stage 3: derived keys
def stage_derive():
    """Adds derived, traceable keys to the results JSON (no bootstrap of scores; cheap)."""
    R = json.load(open(OUT_JSON))
    # chance level of the weighted estimator on the AgentDojo P0 composition
    DJ = R["M8_AgentDojo"]
    npos, nneg = DJ["n_pos"], DJ["n_neg"]
    yE = np.r_[np.ones(npos, int), np.zeros(nneg, int)]
    rng = np.random.default_rng(BASE + 4)
    ch = {k: [] for k in PIS}
    one = np.ones((1, len(yE)))
    for _ in range(2000):
        tp, fp = Ranked(rng.random(len(yE)), yE).tpfp(one)
        for k, pi in PIS.items():
            ch[k].append(ap_from(tp, fp, prev_mult(pi, tp, fp))[0])
    DJ["chance_wAUPRC_random_scorer"] = {k: {"mean": rnd(np.mean(v)), "p2.5": rnd(np.percentile(v, 2.5)),
                                             "p97.5": rnd(np.percentile(v, 97.5))} for k, v in ch.items()}
    DJ["chance_note"] = f"2,000 uniformly random scorers on {npos} positives / {nneg} negatives (seed {BASE + 4})"
    # detector order under the two R1 estimators
    M = R["M8_R1"]; order = {}
    for k in PIS:
        w = sorted(R1_DETS, key=lambda d: -M["detectors"][d]["wAUPRC"][k]["point"])
        s = sorted(R1_DETS, key=lambda d: -M["subsampling_v03_secondary"][k]["AP"][d]["point"])
        order[k] = {"weighted": w, "subsampling": s, "same_order": w == s}
    M["detector_order_by_estimator"] = order
    M["n_detected_holm"] = {"weighted": {k: sum(v["detected_holm"] for v in M["paired"][k].values()) for k in list(PIS) + ["native", "AUROC"]},
                            "subsampling_AP": {k: sum(v["detected_holm"] for v in M["subsampling_v03_secondary"][k]["paired_AP"].values()) for k in PIS},
                            "world_cluster": {k: sum(v["detected_holm"] for v in M["world_cluster_sensitivity"]["paired"][k].values()) for k in list(PIS) + ["native", "AUROC"]}}
    # R4_CONTROL.md Sec. 7 "reasonable within-AgentDojo detection" criteria, P0 / UT, per estimator (no Case decision here)
    crit = {"note": ("R4_CONTROL.md Sec. 7: AUROC >= 0.80, AUROC lower bound >= 0.70, AUPRC@10% >= 0.30, AUPRC@10% lower "
                     "bound > 0.10, evaluated on WUT (P0, UT scheme) per detector and per AUPRC estimator. C1, IT and the "
                     "Case decision are outside this stream."), "detectors": {}}
    sources = {"weighted_strat_rounded": R["M8_AgentDojo"]["schemes"]["traj_strat"]["settings"]["WUT"]["detectors"],
               "weighted_strat_raw": R["M8_AgentDojo_raw_scores"]["schemes"]["traj_strat"]["settings"]["WUT"]["detectors"],
               "weighted_cluster_rounded": R["M8_AgentDojo"]["schemes"]["cluster_suite_user_task"]["settings"]["WUT"]["detectors"]}
    for d in DOJO_DETS:
        e = {}
        for name, src in sources.items():
            x = src[d]
            e[name] = {"AUROC": x["AUROC"]["point"], "AUROC_lo": x["AUROC"]["lo"], "AUPRC10": x["pi10"]["point"],
                       "AUPRC10_lo": x["pi10"]["lo"], "AUROC_ge_0.80": x["AUROC"]["point"] >= 0.80,
                       "AUROC_lo_ge_0.70": x["AUROC"]["lo"] >= 0.70, "AUPRC10_ge_0.30": x["pi10"]["point"] >= 0.30,
                       "AUPRC10_lo_gt_0.10": x["pi10"]["lo"] > 0.10}
            e[name]["all_four_met"] = all(v for k, v in e[name].items() if k.endswith(("0.80", "0.70", "0.30", "0.10")))
        sx = R["M8_AgentDojo"]["subsampling_v03_secondary"]["pi10"]["AP"]["WUT"][d]
        e["subsampling_v03_AUPRC10"] = {"AUPRC10": sx["point"], "AUPRC10_lo": sx["lo"], "AUPRC10_ge_0.30": sx["point"] >= 0.30,
                                        "AUPRC10_lo_gt_0.10": sx["lo"] > 0.10}
        crit["detectors"][d] = e
    R["R4_CONTROL_within_criteria_WUT"] = crit
    # ---- derived scalars quoted in the summary
    w = M["detectors"]["LR-C"]["wAUPRC"]["pi02"]; s = M["subsampling_v03_secondary"]["pi02"]["AP"]["LR-C"]
    p2 = R["R1_operating_points"]["detectors"]["LR-C"]["t_r80"]["precision_pi02"]["point"]
    m9p = R["M9"]["R1"]["schemes"]["traj_strat"]["detectors"]["LR-C"]["t_r80"]["precision_eq6"]["pi02"]
    best = (-1, None, None)
    for d in [x for x in R1_DETS if x != "TFIDF"]:
        for op, x in R["R1_operating_points"]["detectors"][d].items():
            v = x["precision_pi0.1pct"]["point"]
            if v is not None and v > best[0]:
                best = (v, d, op)
        for nd_, per in R["R1_operating_points"]["m10_fpr_budgets"]["detectors"][d].items():
            for bk, x in per.items():
                v = x["precision_pi0.1pct"]["point"]
                if v is not None and v > best[0]:
                    best = (v, d, f"{nd_}.{bk}")
    RC = R["m9_rounding_check"]
    mxr = max(((RC[s_][d]["max_abs_change_point_AUROC_wAUPRC"], s_, d) for s_ in RC for d in RC[s_]))
    R["summary_derived"] = {
        "M8_LR-C_pi02_interval_width_weighted": rnd(w["hi"] - w["lo"]),
        "M8_LR-C_pi02_interval_width_subsampling": rnd(s["hi"] - s["lo"]),
        "M8_LR-C_pi02_width_ratio_subsampling_over_weighted": rnd((s["hi"] - s["lo"]) / (w["hi"] - w["lo"])),
        "EQ6_LR-C_t_r80_false_alerts_per_true_alert_pi02": rnd((1 - p2) / p2),
        "M9_R1_LR-C_pi02_precision_ratio_benign_only_over_hard_only": rnd(m9p["benign_only"]["point"] / m9p["hard_only"]["point"]),
        "m10_max_precision_pi0.1pct_content_blind_R1": {"value": rnd(best[0]), "detector": best[1], "threshold": best[2],
                                                         "searched": "R1_operating_points t_r80/t05/t_val80 and m10 budgets, all detectors except TFIDF"},
        "m9_max_abs_change_from_rounding": {"value": rnd(mxr[0]), "setting": mxr[1], "detector": mxr[2]}}
    # ---- the review's figures next to ours (review values quoted from REVIEW_REPORT.md; ours by key)
    def g(path):
        x = R
        for k in path.split("|"):
            x = x[int(k)] if isinstance(x, list) else x[k]
        return x
    refs = [
        ("M8", "LR-C wAUPRC 1%", 0.156, "M8_R1|detectors|LR-C|wAUPRC|pi01|point"),
        ("M8", "LR-C wAUPRC 2%", 0.232, "M8_R1|detectors|LR-C|wAUPRC|pi02|point"),
        ("M8", "LR-C wAUPRC 5%", 0.380, "M8_R1|detectors|LR-C|wAUPRC|pi05|point"),
        ("M8", "LR-C wAUPRC 10%", 0.523, "M8_R1|detectors|LR-C|wAUPRC|pi10|point"),
        ("M8", "LR-C wAUPRC 2% lower", 0.185, "M8_R1|detectors|LR-C|wAUPRC|pi02|lo"),
        ("M8", "LR-C wAUPRC 2% upper", 0.325, "M8_R1|detectors|LR-C|wAUPRC|pi02|hi"),
        ("M8", "LR-C wAUPRC 2% bootstrap mean", 0.245, "M8_R1|detectors|LR-C|wAUPRC|pi02|boot_mean"),
        ("M8", "LR-C minus Markov wAUPRC 1%", 0.132, "M8_R1|paired|pi01|LR-C-minus-Markov|point"),
        ("M8", "LR-C minus Markov wAUPRC 2%", 0.185, "M8_R1|paired|pi02|LR-C-minus-Markov|point"),
        ("M8", "LR-C minus LR-B wAUPRC 2%", 0.082, "M8_R1|paired|pi02|LR-C-minus-LR-B|point"),
        ("M8", "LR-C minus LR-B wAUPRC 2% lower", 0.003, "M8_R1|paired|pi02|LR-C-minus-LR-B|lo"),
        ("M8", "GRU minus LR-C AUROC native", 0.056, "M8_R1|paired|AUROC|GRU-minus-LR-C|point"),
        ("M8", "GRU minus LR-C AUROC on 2% replicates, lower", -0.021, "M8_R1|subsampling_v03_secondary|pi02|paired_AUROC|GRU-minus-LR-C|lo"),
        ("M8", "LR-C minus Markov AUROC on 1% replicates, lower", -0.035, "M8_R1|subsampling_v03_secondary|pi01|paired_AUROC|LR-C-minus-Markov|lo"),
        ("M9", "LR-C FPR benign", 0.118, "M9|R1|schemes|traj_strat|detectors|LR-C|t_r80|FPR_benign|point"),
        ("M9", "LR-C FPR hard negative", 0.387, "M9|R1|schemes|traj_strat|detectors|LR-C|t_r80|FPR_hard_negative|point"),
        ("M9", "LR-C failed flagged", 0.218, "M9|R1|schemes|traj_strat|detectors|LR-C|t_r80|flag_failed_attack|point"),
        ("M9", "LR-C precision 2% benign-only", 0.117, "M9|R1|schemes|traj_strat|detectors|LR-C|t_r80|precision_eq6|pi02|benign_only|point"),
        ("M9", "LR-C precision 2% pool-E mix", 0.077, "M9|R1|schemes|traj_strat|detectors|LR-C|t_r80|precision_eq6|pi02|poolE_mix|point"),
        ("M9", "LR-C precision 2% hard-only", 0.039, "M9|R1|schemes|traj_strat|detectors|LR-C|t_r80|precision_eq6|pi02|hard_only|point"),
        ("M9", "LR-C recall full", 0.830, "M9|R1|schemes|traj_strat|detectors|LR-C|t_r80|recall_full|point"),
        ("M9", "LR-C recall partial", 0.727, "M9|R1|schemes|traj_strat|detectors|LR-C|t_r80|recall_partial|point"),
        ("M9", "LR-C recall delayed", 0.642, "M9|R1|schemes|traj_strat|detectors|LR-C|t_r80|recall_delayed|point"),
        ("M9", "LR-A recall partial", 0.502, "M9|R1|schemes|traj_strat|detectors|LR-A|t_r80|recall_partial|point"),
        ("M9", "LR-B partial-hijack AUROC", 0.872, "M9|R1|schemes|traj_strat|detectors|LR-B|AUROC_subgroups|partial_vs_poolE_neg|point"),
        ("M9", "LR-C partial-hijack AUROC", 0.858, "M9|R1|schemes|traj_strat|detectors|LR-C|AUROC_subgroups|partial_vs_poolE_neg|point"),
        ("M9", "R3 medical LR-B full recall", 0.47, "M9|R3_medical|schemes|traj_strat|detectors|LR-B|t_r80|recall_full|point"),
        ("M9", "R3 banking LR-B full recall", 0.66, "M9|R3_banking|schemes|traj_strat|detectors|LR-B|t_r80|recall_full|point"),
        ("m4", "GRU t_val80 precision 2%, seed 0", 0.188, "m4_gru_t_val80|table|R1|GRU|t_val80|precision_pi02_poolE_mix|point_by_rep|0"),
        ("m4", "GRU FPR at test TPR 0.8, seed 1", 0.087, "M8_R1|detectors|GRU|FPR_at_test_TPR80|FPR_at_TPR80|point_by_rep|1"),
        ("m4", "LR-C FPR at test TPR 0.8", 0.210, "M8_R1|detectors|LR-C|FPR_at_test_TPR80|FPR_at_TPR80|point"),
        ("m4", "GRU R1 theta=0.5 FPR", 0.119, "m4_theta05|table|R1|GRU|FPR_poolE|point"),
        ("m4", "GRU R1 theta=0.5 recall", 0.856, "m4_theta05|table|R1|GRU|recall_poolE|point"),
        ("m9", "chance AP 8/825", 0.0170, "m9_chance_AP|values|8/825|exact"),
        ("m9", "chance AP 17/825", 0.0275, "m9_chance_AP|values|17/825|exact"),
        ("m9", "chance AP 44/392", 0.1127, "m9_chance_AP|values|44/392|exact"),
        ("m10", "LR-C precision 0.1%", 0.004, "R1_operating_points|detectors|LR-C|t_r80|precision_pi0.1pct|point"),
        ("m10", "LR-C precision 0.01%", 0.0004, "R1_operating_points|detectors|LR-C|t_r80|precision_pi0.01pct|point"),
        ("m10", "LR-C 1% budget TPR (all label-0 negatives)", 0.27, "R1_operating_points|m10_fpr_budgets|detectors|LR-C|all_label0|fpr01|test_TPR|point"),
        ("m10", "LR-C 1% budget FPR (all label-0 negatives)", 0.017, "R1_operating_points|m10_fpr_budgets|detectors|LR-C|all_label0|fpr01|test_FPR|point"),
        ("m10", "LR-C 5% budget TPR (all label-0 negatives)", 0.52, "R1_operating_points|m10_fpr_budgets|detectors|LR-C|all_label0|fpr05|test_TPR|point"),
        ("m10", "LR-C 5% budget FPR (all label-0 negatives)", 0.059, "R1_operating_points|m10_fpr_budgets|detectors|LR-C|all_label0|fpr05|test_FPR|point"),
        ("m17.5", "LR-C minus Markov subsampled AP 2%", 0.198, "M8_R1|subsampling_v03_secondary|pi02|paired_AP|LR-C-minus-Markov|point"),
    ]
    rows = [{"comment": c, "quantity": q, "review": rv, "ours": g(k), "ours_key": k.replace("|", "."),
             "abs_diff": rnd(abs(g(k) - rv))} for c, q, rv, k in refs]
    lf = R["m11_label_free"]["folds"]
    fl = [lf[f]["detectors"]["LR-C"]["deployment_prev02"]["summary_over_draws"]["FPR"]["mean"] for f in DOMAINS]
    rc = [lf[f]["detectors"]["LR-C"]["deployment_prev02"]["summary_over_draws"]["recall"]["mean"] for f in DOMAINS]
    rows.append({"comment": "m11", "quantity": "LR-C label-free FPR range over folds (reading b, fold means)", "review": [0.191, 0.209],
                 "ours": [min(fl), max(fl)], "ours_key": "m11_label_free.folds.<domain>.detectors.LR-C.deployment_prev02.summary_over_draws.FPR.mean"})
    rows.append({"comment": "m11", "quantity": "LR-C label-free recall range over folds (reading b, fold means)", "review": [0.483, 0.728],
                 "ours": [min(rc), max(rc)], "ours_key": "m11_label_free.folds.<domain>.detectors.LR-C.deployment_prev02.summary_over_draws.recall.mean"})
    R["review_reference_values"] = {"note": "Review values are quoted from REVIEW_REPORT.md (rounded as printed there); 'ours' is read from this file.",
                                    "rows": rows}
    json.dump(R, open(OUT_JSON, "w"), indent=1)
    log("derived keys added")


# =============================================================================== stage 4: summary markdown
def f3(e, d=3, key="point"):
    if e is None:
        return "n/a"
    v = e.get(key)
    if v is None:
        return "undefined"
    lo, hi = e.get("lo"), e.get("hi")
    s = f"{v:.{d}f}" if lo is None or hi is None else f"{v:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"
    if "clopper_pearson_95" in e:
        c = e["clopper_pearson_95"]; s += f" (CP {c[0]:.{d}f}-{c[1]:.{d}f})"
    if "lower_bound_if_FPR_at_CP_upper" in e:
        s += f" (>= {e['lower_bound_if_FPR_at_CP_upper']:.{d}f})"
    return s


def fd(e, d=3):
    s = lambda x: f"{x:+.{d}f}"
    return f"{s(e['point'])} [{s(e['lo'])}, {s(e['hi'])}]"


def tag(e):
    return "H" if e["detected_holm"] else ("d" if e["detected"] else "nd")


def mm(e, d=3):
    return f"{e['mean']:.{d}f} [{e['min']:.{d}f}, {e['max']:.{d}f}]"


def fprec(x, e, d=3):
    """Precision cell; flags estimates that rest on fewer than 10 flagged pool-E traces (mean over repetitions)."""
    n = x.get("n_flagged_poolE")
    if n is not None and n < 10:
        if n == 0:
            return "undefined (no alert)"
        return f"{e['point']:.{d}f}, unreliable ({n:g} alert{'' if n == 1 else 's'})"
    return f3(e, d)


def stage_summary():
    R = json.load(open(OUT_JSON))
    L = []; A = L.append
    M = R["M8_R1"]; OP = R["R1_operating_points"]; DJ = R["M8_AgentDojo"]; DR = R["M8_AgentDojo_raw_scores"]
    PK = ["pi01", "pi02", "pi05", "pi10"]
    A("# Operating points and prevalence (revision stream `prevalence_ops`)")
    A("")
    A(f"Script `revision/prevalence_ops.py` (SHA-256 `{R['meta']['script_sha256'][:16]}...`); results "
      f"`revision/prevalence_ops_results.json` (generated {R['meta']['generated']}). Every number below is a value in that "
      "JSON. Key paths are given in brackets; `<pi>` is one of pi01, pi02, pi05, pi10 or native. Estimates are shown as point "
      f"[95% bootstrap percentile interval], with B = {R['meta']['B']}. The point is the plug-in value on the full pool; for "
      "the GRU it is the mean over its three seeds, and for within-AgentDojo detectors the mean over the five CV repetitions. "
      "Each bootstrap replicate draws one GRU seed or CV repetition. Scores are rounded to 12 decimals before ranking (m9); "
      "for AgentDojo, rank-based values from unrounded scores are also given (deviation P10). Thresholds come from training "
      "or validation data only. The one exception is the matched-recall quantity (FPR at a fixed test TPR of 0.8), which is "
      "labelled wherever it appears. 'Detected' means the 95% interval excludes zero. H means detected after Holm correction "
      "within the stated family, d means detected before correction but not after Holm, and nd means not detected. "
      "'CP' is an exact Clopper-Pearson 95% interval, given when the bootstrap interval collapses. '>=' is a lower bound on "
      "precision when no negative was flagged.")
    A("")
    A("Pools. R1 pool E has 831 attacked and 825 negative traces (600 benign, 225 hard negatives, so 27% hard negatives). "
      "Its 225 failed attacks are scored separately. AgentDojo P0 has 2,294 compromised and 392 benign traces. Its 4,987 "
      "unsuccessful attacks are scored separately.")
    A("")
    # ================================================================== M8
    A("## M8. AUPRC at controlled prevalence")
    A("")
    A("**Question.** What is AUPRC at prevalence pi when it is estimated from all positives? The estimator is "
      "prevalence-weighted average precision (each positive weighted by pi/(1-pi) * n-/n+), with a stratified bootstrap of "
      "the full pool (plan 3.1). How does it compare with the v0.3 subsampling estimate?")
    A("")
    A("**R1, prevalence-weighted AUPRC** [`M8_R1.detectors.<det>.wAUPRC.<pi>`; AUROC `M8_R1.detectors.<det>.AUROC`]")
    A("")
    A("| Detector | AUROC | wAUPRC 1% | 2% | 5% | 10% | native (0.502) |")
    A("|---|---|---|---|---|---|---|")
    for d in R1_DETS:
        x = M["detectors"][d]
        A(f"| {d} | {f3(x['AUROC'])} | " + " | ".join(f3(x["wAUPRC"][k]) for k in PK + ["native"]) + " |")
    A("")
    ch = M["chance_wAUPRC_random_scorer"]
    A("Chance level of this estimator, as the mean [2.5th, 97.5th percentile] of 2,000 random scorers "
      "[`M8_R1.chance_wAUPRC_random_scorer.<pi>`]: " +
      ", ".join(f"{k[2:].lstrip('0')}%: {ch[k]['mean']:.4f} [{ch[k]['p2.5']:.4f}, {ch[k]['p97.5']:.4f}]" for k in PK) + ".")
    A("")
    A("Plug-in point versus bootstrap mean for LR-C [`M8_R1.detectors.LR-C.wAUPRC.<pi>.point` / `.boot_mean`]: " +
      ", ".join(f"{k[2:].lstrip('0')}%: {M['detectors']['LR-C']['wAUPRC'][k]['point']:.3f} vs {M['detectors']['LR-C']['wAUPRC'][k]['boot_mean']:.3f}" for k in PK) + ".")
    A("")
    A("**R1, paired differences in wAUPRC.** Same resampled indices for both detectors; Holm over the five comparisons "
      "within each pi. The AUROC column is prevalence-free [`M8_R1.paired.<pi>.<comparison>`, with `p_boot` and `p_holm`].")
    A("")
    A("| Comparison | 1% | 2% | 5% | 10% | native AP | AUROC |")
    A("|---|---|---|---|---|---|---|")
    for a, b in R1_PAIRS:
        k2 = f"{a}-minus-{b}"
        A(f"| {k2} | " + " | ".join(f"{fd(M['paired'][k][k2])} {tag(M['paired'][k][k2])}" for k in PK + ["native", "AUROC"]) + " |")
    A("")
    nd = M["n_detected_holm"]
    A("Comparisons detected after Holm, out of five [`M8_R1.n_detected_holm`]:")
    A("")
    A("- weighted: " + ", ".join(f"{k[2:].lstrip('0')}% {nd['weighted'][k]}" for k in PK) + ";")
    A("- world-cluster bootstrap: " + ", ".join(f"{k[2:].lstrip('0')}% {nd['world_cluster'][k]}" for k in PK) + ";")
    A("- v0.3 subsampling: " + ", ".join(f"{k[2:].lstrip('0')}% {nd['subsampling_AP'][k]}" for k in PK) + ".")
    A("")
    W = M["world_cluster_sensitivity"]
    A(f"**World-cluster sensitivity** ({W['n_clusters']} worlds, plan 3.5) [`M8_R1.world_cluster_sensitivity`]:")
    A("")
    A("- LR-C wAUPRC: " + ", ".join(f"{k[2:].lstrip('0')}% {f3(W['detectors']['LR-C'][k])}" for k in PK) + ".")
    A("- At 1%: " + ", ".join(f"{k} {fd(v)} {tag(v)}" for k, v in W["paired"]["pi01"].items()) + ".")
    A("- At 2%: " + ", ".join(f"{k} {fd(v)} {tag(v)}" for k, v in W["paired"]["pi02"].items()) + ".")
    A("")
    S = M["subsampling_v03_secondary"]
    A("**Secondary: the v0.3 subsampling estimator.** Each replicate draws n+ positives against 825 negatives. The "
      "replicate mean is the expected AP of a test set with that many positives, and the spread is the sampling "
      "distribution of such a small test set, not a confidence interval for AUPRC [`M8_R1.subsampling_v03_secondary.<pi>.AP.<det>`].")
    A("")
    A("| Detector | 1% (n+=8) | 2% (n+=17) | 5% (n+=43) | 10% (n+=92) |")
    A("|---|---|---|---|---|")
    for d in R1_DETS:
        A(f"| {d} | " + " | ".join(f3(S[k]["AP"][d]) for k in PK) + " |")
    A(f"| random scorer, expected AP | " + " | ".join(f"{S[k]['expected_AP_random_scorer']:.4f}" for k in PK) + " |")
    A("")
    A("Detectors rank in the same order under both estimators at every pi "
      f"[`M8_R1.detector_order_by_estimator.<pi>.same_order`: {', '.join(str(M['detector_order_by_estimator'][k]['same_order']) for k in PK)}].")
    A("")
    A("Paired differences under subsampling, with Holm within each pi [`M8_R1.subsampling_v03_secondary.<pi>.paired_AP` and `.paired_AUROC`]:")
    A("")
    A("| Comparison | AP 1% | AP 2% | AP 5% | AP 10% | AUROC, 1% replicates | AUROC, 2% replicates |")
    A("|---|---|---|---|---|---|---|")
    for a, b in R1_PAIRS:
        k2 = f"{a}-minus-{b}"
        A(f"| {k2} | " + " | ".join(f"{fd(S[k]['paired_AP'][k2])} {tag(S[k]['paired_AP'][k2])}" for k in PK) +
          f" | {fd(S['pi01']['paired_AUROC'][k2])} {tag(S['pi01']['paired_AUROC'][k2])} | {fd(S['pi02']['paired_AUROC'][k2])} {tag(S['pi02']['paired_AUROC'][k2])} |")
    A("")
    A("**AgentDojo P0, prevalence-weighted AUPRC.** R4 is transfer from AgentDrift and WUT is within AgentDojo (user-task "
      "CV). The main interval is the stratified bootstrap of plan 3.1 [`M8_AgentDojo.schemes.traj_strat.settings.<S>.detectors.<det>.<pi>`]. "
      "The cluster column is the 117-cluster suite|user_task bootstrap of plan 3.5 [`...schemes.cluster_suite_user_task...`]. "
      "The unrounded column uses raw scores [`M8_AgentDojo_raw_scores.schemes.traj_strat...`].")
    A("")
    A("| Setting | Detector | AUROC | AUROC unrounded | wAUPRC 5% | wAUPRC 10% | 10% cluster CI | 10% unrounded |")
    A("|---|---|---|---|---|---|---|---|")
    for sname in ("R4", "WUT"):
        for d in DOJO_DETS:
            x = DJ["schemes"]["traj_strat"]["settings"][sname]["detectors"][d]
            c = DJ["schemes"]["cluster_suite_user_task"]["settings"][sname]["detectors"][d]
            r = DR["schemes"]["traj_strat"]["settings"][sname]["detectors"][d]
            A(f"| {sname} | {d} | {f3(x['AUROC'])} | {r['AUROC']['point']:.3f} | {f3(x['pi05'])} | {f3(x['pi10'])} | "
              f"[{c['pi10']['lo']:.3f}, {c['pi10']['hi']:.3f}] | {f3(r['pi10'])} |")
    A("")
    cd = DJ["chance_wAUPRC_random_scorer"]
    A("Chance level of the estimator on P0 [`M8_AgentDojo.chance_wAUPRC_random_scorer.<pi>`]: 5% "
      f"{cd['pi05']['mean']:.4f} [{cd['pi05']['p2.5']:.4f}, {cd['pi05']['p97.5']:.4f}], 10% {cd['pi10']['mean']:.4f} "
      f"[{cd['pi10']['p2.5']:.4f}, {cd['pi10']['p97.5']:.4f}].")
    A("")
    wt = DJ["schemes"]["traj_strat"]["within_minus_transfer"]; wc = DJ["schemes"]["cluster_suite_user_task"]["within_minus_transfer"]
    A("Within minus transfer (WUT minus R4), wAUPRC at 10%, with Holm over five detectors "
      "[`M8_AgentDojo.schemes.<scheme>.within_minus_transfer.pi10`]:")
    A("")
    A("- stratified bootstrap: " + ", ".join(f"{d} {fd(wt['pi10'][d])} {tag(wt['pi10'][d])}" for d in DOJO_DETS) + ";")
    A("- cluster bootstrap: " + ", ".join(f"{d} {fd(wc['pi10'][d])} {tag(wc['pi10'][d])}" for d in DOJO_DETS) + ".")
    A("")
    A("Paired detector differences at 10%, with Holm over four comparisons [`M8_AgentDojo.schemes.<scheme>.settings.<S>.paired.pi10`; "
      "unrounded in `M8_AgentDojo_raw_scores`]:")
    A("")
    A("| Setting | Comparison | stratified | cluster | stratified, unrounded |")
    A("|---|---|---|---|---|")
    for sname in ("R4", "WUT"):
        for a, b in DOJO_PAIRS:
            k2 = f"{a}-minus-{b}"
            e1 = DJ["schemes"]["traj_strat"]["settings"][sname]["paired"]["pi10"][k2]
            e2 = DJ["schemes"]["cluster_suite_user_task"]["settings"][sname]["paired"]["pi10"][k2]
            e3 = DR["schemes"]["traj_strat"]["settings"][sname]["paired"]["pi10"][k2]
            A(f"| {sname} | {k2} | {fd(e1)} {tag(e1)} | {fd(e2)} {tag(e2)} | {fd(e3)} {tag(e3)} |")
    A("")
    CR = R["R4_CONTROL_within_criteria_WUT"]["detectors"]
    A("**'Reasonable within-AgentDojo detection' criteria of R4_CONTROL.md Sec. 7 on WUT (P0)** [`R4_CONTROL_within_criteria_WUT.detectors.<det>`]. "
      "This stream does not decide the Case, which needs C1 and IT.")
    A("")
    A("| Detector | AUROC >= 0.80 | AUROC lower >= 0.70 | wAUPRC@10% >= 0.30 (stratified, rounded) | same, unrounded | v0.3 subsampling AP@10% >= 0.30 |")
    A("|---|---|---|---|---|---|")
    for d in DOJO_DETS:
        e = CR[d]; w = e["weighted_strat_rounded"]; wr = e["weighted_strat_raw"]; s = e["subsampling_v03_AUPRC10"]
        A(f"| {d} | {w['AUROC']:.3f} {'yes' if w['AUROC_ge_0.80'] else 'no'} | {w['AUROC_lo']:.3f} {'yes' if w['AUROC_lo_ge_0.70'] else 'no'} | "
          f"{w['AUPRC10']:.3f} {'yes' if w['AUPRC10_ge_0.30'] else 'no'} | {wr['AUPRC10']:.4f} {'yes' if wr['AUPRC10_ge_0.30'] else 'no'} | "
          f"{s['AUPRC10']:.3f} {'yes' if s['AUPRC10_ge_0.30'] else 'no'} |")
    A("")
    SD = DJ["subsampling_v03_secondary"]
    A("v0.3 subsampling on AgentDojo (secondary) [`M8_AgentDojo.subsampling_v03_secondary.<pi>.AP.<S>.<det>`], LR-C: "
      f"at 10% (n+ = 44) R4 {f3(SD['pi10']['AP']['R4']['LR-C'])} and WUT {f3(SD['pi10']['AP']['WUT']['LR-C'])}; "
      f"at 5% (n+ = 21) R4 {f3(SD['pi05']['AP']['R4']['LR-C'])} and WUT {f3(SD['pi05']['AP']['WUT']['LR-C'])}. "
      f"A random scorer's expected AP is {SD['pi10']['expected_AP_random_scorer']:.4f} at 10% and "
      f"{SD['pi05']['expected_AP_random_scorer']:.4f} at 5%.")
    A("")
    A("**Plain reading.** " + M8_READING(R))
    A("")
    # ================================================================== m3
    A("## m3 (prevalence family). Multiplicity at each prevalence")
    A("")
    A("**Question.** Which of the paired prevalence comparisons remain detected after Holm correction within each "
      "prevalence (plan 3.6(e))?")
    A("")
    A("The counts and decisions are in the M8 tables above [`M8_R1.paired.<pi>.<comparison>.p_holm` / `.detected_holm`; "
      "`M8_R1.n_detected_holm`]. The borderline cases at 2% are:")
    A("")
    for k2 in ("LR-C-minus-LR-B", "GRU-minus-LR-C"):
        e = M["paired"]["pi02"][k2]; ew = W["paired"]["pi02"][k2]
        A(f"- {k2}: {fd(e)}, p_boot {e['p_boot']:.4f}, p_holm {e['p_holm']:.4f} ({tag(e)}); under the world-cluster "
          f"bootstrap {fd(ew)}, p_holm {ew['p_holm']:.4f} ({tag(ew)}).")
    e1 = DJ["schemes"]["traj_strat"]["settings"]["WUT"]["paired"]["pi10"]["LR-C-minus-LR-B"]
    e2 = DR["schemes"]["traj_strat"]["settings"]["WUT"]["paired"]["pi10"]["LR-C-minus-LR-B"]
    e3 = DJ["schemes"]["cluster_suite_user_task"]["settings"]["WUT"]["paired"]["pi10"]["LR-C-minus-LR-B"]
    A(f"- AgentDojo WUT, LR-C-minus-LR-B at 10%: {fd(e1)}, p_holm {e1['p_holm']:.4f} ({tag(e1)}) with rounded scores; "
      f"{fd(e2)}, p_holm {e2['p_holm']:.4f} ({tag(e2)}) with unrounded scores; {fd(e3)} ({tag(e3)}) under the cluster bootstrap.")
    A("")
    A("**Plain reading.** " + M3_READING(R))
    A("")
    # ================================================================== m17.5
    A("## m17 item 5. Each comparison at every prevalence")
    A("")
    A("**Question.** Section VI-A shows each comparison only at the prevalence where it fails. What does each comparison "
      "show at every prevalence?")
    A("")
    e = S["pi02"]["paired_AP"]["LR-C-minus-Markov"]
    A(f"The M8 tables give all five comparisons at 1%, 2%, 5%, 10% and native prevalence, under both estimators. The "
      f"review's example, LR-C minus Markov under v0.3 subsampling at 2%, is {fd(e)} ({tag(e)}) "
      f"[`M8_R1.subsampling_v03_secondary.pi02.paired_AP.LR-C-minus-Markov`]. The same comparison with the weighted "
      f"estimator is {fd(M['paired']['pi02']['LR-C-minus-Markov'])} ({tag(M['paired']['pi02']['LR-C-minus-Markov'])}), and "
      f"at 1% {fd(M['paired']['pi01']['LR-C-minus-Markov'])} ({tag(M['paired']['pi01']['LR-C-minus-Markov'])}).")
    A("")
    A("**Plain reading.** Reported at a common prevalence, the comparisons do not support the Section VI-A contrast. "
      "Under subsampling, AUPRC and AUROC lose power together on the same small replicates (see the AUROC columns). With "
      "the weighted estimator, most AUPRC differences are detected even at 1% to 2%. Only LR-C minus LR-B and GRU minus "
      "LR-C remain undetected at 1%.")
    A("")
    # ================================================================== Eq. (6)
    A("## M8 (2). Precision at theta_r80 by prevalence (Eq. 6) and alerts per 1,000 traces")
    A("")
    A("**Question.** What precision and alert load does the training-derived threshold give at each prevalence? Both are "
      "computed from bootstrap TPR and FPR on pool E, using the pool-E negative mix.")
    A("")
    A("R1 [`R1_operating_points.detectors.<det>.t_r80.<quantity>`]:")
    A("")
    A("| Detector | TPR | FPR | Prec. 1% | Prec. 2% | Prec. 5% | Prec. 10% | Alerts/1k at 2% | Alerts/1k at 10% |")
    A("|---|---|---|---|---|---|---|---|---|")
    for d in R1_DETS:
        x = OP["detectors"][d]["t_r80"]
        A(f"| {d} | {f3(x['TPR'])} | {f3(x['FPR'])} | " + " | ".join(f3(x[f"precision_{k}"]) for k in PK) +
          f" | {f3(x['alerts_per_1000_pi02'], 0)} | {f3(x['alerts_per_1000_pi10'], 0)} |")
    A("")
    A("**Plain reading.** " + EQ6_READING(R))
    A("")
    # ================================================================== M9
    A("## M9. Breakdown by negative class and hijack pattern")
    A("")
    A("**Question.** At theta_r80, how do the operating-point rates split by negative class (benign, hard negative, failed "
      "attack) and by hijack pattern (full, partial, delayed)? How does precision at 2% and 10% depend on the negative mix?")
    A("")
    A("The bootstrap is trajectory-level and stratified by category x pattern [`M9.<S>.schemes.traj_strat.detectors.<det>.t_r80.<quantity>`]. "
      "The table shows the main detectors. Every cached detector, theta = 0.5, and the GRU at theta_val80 are in the JSON. "
      "Precision uses the pooled pool-E recall together with the FPR of each negative mix. A precision resting on fewer "
      "than 10 flagged pool-E traces is marked with its alert count [`...t_r80.n_flagged_poolE`].")
    A("")
    for s in M9_SETTINGS:
        X = R["M9"][s]["schemes"]["traj_strat"]["detectors"]
        dojo = s in ("R4", "WUT")
        A(f"**{s}** (rows: " + ", ".join(f"{k} {v}" for k, v in R["M9"][s]["n_rows"].items()) + ")")
        A("")
        if not dojo:
            A("| Detector | FPR benign | FPR hard neg. | Failed flagged | Recall full | Recall partial | Recall delayed | Prec. 2%: benign-only / pool-E mix / hard-only | Prec. 10%: benign-only / pool-E mix / hard-only |")
            A("|---|---|---|---|---|---|---|---|---|")
        else:
            A("| Detector | FPR benign (= pool E) | Unsuccessful attacks flagged | Recall | Prec. 2% | Prec. 10% | Prec. 2%, failed-only negatives | Prec. 10%, failed-only |")
            A("|---|---|---|---|---|---|---|---|")
        for d in ["LR-A", "LR-B", "LR-C", "LR-C-drop", "Markov", "IForest", "TFIDF", "GRU", "GRU-drop0.2"]:
            if d not in X:
                continue
            x = X[d]["t_r80"]; p = x["precision_eq6"]
            if not dojo:
                A(f"| {d} | {f3(x['FPR_benign'])} | {f3(x['FPR_hard_negative'])} | {f3(x['flag_failed_attack'])} | "
                  f"{f3(x['recall_full'])} | {f3(x['recall_partial'])} | {f3(x['recall_delayed'])} | "
                  f"{fprec(x, p['pi02']['benign_only'])} / {fprec(x, p['pi02']['poolE_mix'])} / {fprec(x, p['pi02']['hard_only'])} | "
                  f"{fprec(x, p['pi10']['benign_only'])} / {fprec(x, p['pi10']['poolE_mix'])} / {fprec(x, p['pi10']['hard_only'])} |")
            else:
                A(f"| {d} | {f3(x['FPR_benign'])} | {f3(x['flag_failed_attack'])} | {f3(x['recall_poolE'])} | "
                  f"{fprec(x, p['pi02']['poolE_mix'])} | {fprec(x, p['pi10']['poolE_mix'])} | {fprec(x, p['pi02']['failed_only'])} | {fprec(x, p['pi10']['failed_only'])} |")
        A("")
        if s in CLUSTER_OF:
            k = "cluster_suite_user_task" if dojo else "cluster_task"
            XC = R["M9"][s]["schemes"][k]["detectors"]["LR-C"]["t_r80"]
            A(f"The cluster bootstrap for LR-C ({R['M9'][s]['schemes'][k]['bootstrap'].split(' over')[0]}; plan 3.5) "
              f"[`M9.{s}.schemes.{k}.detectors.LR-C.t_r80`] gives: FPR benign {f3(XC['FPR_benign'])}; failed flagged "
              f"{f3(XC['flag_failed_attack'])}; recall {f3(XC['recall_poolE'])}; precision at 2% (pool-E mix) "
              f"{f3(XC['precision_eq6']['pi02']['poolE_mix'])}.")
            A("")
    A("**AUROC by subgroup.** Each AUROC ranks the positives of one subgroup against the pool-E negatives, or all positives "
      "against one negative class [`M9.<S>.schemes.traj_strat.detectors.<det>.AUROC_subgroups`; paired LR-C minus LR-B in "
      "`M9.<S>.schemes.traj_strat.paired_AUROC_subgroups`].")
    A("")
    A("| Setting | Detector | full | partial | delayed | vs benign | vs hard neg. | vs failed |")
    A("|---|---|---|---|---|---|---|---|")
    for s in ["R1"] + [f"R3_{d}" for d in DOMAINS]:
        X = R["M9"][s]["schemes"]["traj_strat"]
        for d in ["LR-B", "LR-C", "GRU"]:
            g = X["detectors"][d]["AUROC_subgroups"]
            A(f"| {s} | {d} | {f3(g['full_vs_poolE_neg'])} | {f3(g['partial_vs_poolE_neg'])} | {f3(g['delayed_vs_poolE_neg'])} | "
              f"{f3(g['poolE_vs_benign'])} | {f3(g['poolE_vs_hard_negative'])} | {f3(g['poolE_vs_failed'])} |")
        p = X["paired_AUROC_subgroups"]["LR-C-minus-LR-B"]
        A(f"| {s} | LR-C minus LR-B | {fd(p['full_vs_poolE_neg'])} | {fd(p['partial_vs_poolE_neg'])} | {fd(p['delayed_vs_poolE_neg'])} | "
          f"{fd(p['poolE_vs_benign'])} | {fd(p['poolE_vs_hard_negative'])} | {fd(p['poolE_vs_failed'])} |")
    A("")
    A("AgentDojo, AUROC of compromised against unsuccessful attacks [`M9.<R4|WUT>.schemes.traj_strat.detectors.<det>.AUROC_subgroups.poolE_vs_failed`]: " +
      "; ".join(f"{s}: " + ", ".join(f"{d} {f3(R['M9'][s]['schemes']['traj_strat']['detectors'][d]['AUROC_subgroups']['poolE_vs_failed'])}" for d in DOJO_DETS) for s in ("R4", "WUT")) + ".")
    A("")
    A("**Plain reading.** " + M9_READING(R))
    A("")
    # ================================================================== m4
    A("## m4. Held-out GRU threshold, matched recall, and theta = 0.5")
    A("")
    A("**Question.** How does the GRU perform at its validation threshold theta_val80? What is each detector's FPR at a "
      "fixed test TPR of 0.8? What does theta = 0.5 give?")
    A("")
    g = R["m4_gru_t_val80"]["table"]["R1"]["GRU"]
    A("R1 GRU on pool E [`m4_gru_t_val80.table.R1.GRU.<op>`; per-seed values are in `point_by_rep`]:")
    A("")
    A("| Threshold | Recall (per seed) | FPR (per seed) | Precision 2%, pool-E mix (per seed) | Precision 10% |")
    A("|---|---|---|---|---|")
    for op in ("t_r80", "t_val80"):
        ps = lambda e: ", ".join(f"{v:.3f}" for v in e["point_by_rep"])
        A(f"| {op} | {f3(g[op]['recall_poolE'])} ({ps(g[op]['recall_poolE'])}) | {f3(g[op]['FPR_poolE'])} ({ps(g[op]['FPR_poolE'])}) | "
          f"{f3(g[op]['precision_pi02_poolE_mix'])} ({ps(g[op]['precision_pi02_poolE_mix'])}) | {f3(g[op]['precision_pi10_poolE_mix'])} |")
    A("")
    A("**FPR at a fixed test TPR of 0.8.** The threshold is placed using test labels, so this is a matched-recall "
      "comparison, not an operating point [`M8_R1.detectors.<det>.FPR_at_test_TPR80.FPR_at_TPR80`]: " +
      ", ".join(f"{d} {f3(M['detectors'][d]['FPR_at_test_TPR80']['FPR_at_TPR80'])}" for d in R1_DETS) +
      f". For the GRU the per-seed values are {', '.join(f'{v:.3f}' for v in M['detectors']['GRU']['FPR_at_test_TPR80']['FPR_at_TPR80']['point_by_rep'])}.")
    A("")
    A("**theta = 0.5**, recall / FPR on pool E [`m4_theta05.table.<S>.<det>.recall_poolE` / `.FPR_poolE`; failed-attack flag rates in `.flag_failed_attack`]:")
    A("")
    A("| Setting | " + " | ".join(THETA05_DETS) + " |")
    A("|---|" + "---|" * len(THETA05_DETS))
    for s in THETA05_SETTINGS:
        A(f"| {s} | " + " | ".join(f"{f3(R['m4_theta05']['table'][s][d]['recall_poolE'])} / {f3(R['m4_theta05']['table'][s][d]['FPR_poolE'])}" for d in THETA05_DETS) + " |")
    A("")
    A("**Plain reading.** " + M4_READING(R))
    A("")
    # ================================================================== m10
    A("## m10. FPR budgets set on training negatives; Eq. (6) at 0.1% and 0.01%")
    A("")
    A("**Question.** Suppose the threshold flags at most 1% or 5% of training negatives. What TPR and FPR does it then give "
      "on the R1 test pool E? What precision does LR-C have at 0.1% and 0.01% prevalence?")
    A("")
    BU = OP["m10_fpr_budgets"]["detectors"]
    tf = lambda v: (f"{min(v):.4f}-{max(v):.4f}" if isinstance(v, list) else f"{v:.4f}")
    A("The primary definition of training negatives is benign plus hard-negative training rows "
      "[`R1_operating_points.m10_fpr_budgets.detectors.<det>.benign_hard.<fpr01|fpr05>`]. The sensitivity uses all label-0 "
      "training rows, failed attacks included [`...all_label0...`].")
    A("")
    A("| Detector | 1%: train FPR | 1%: test TPR | 1%: test FPR | 5%: train FPR | 5%: test TPR | 5%: test FPR | all label-0, 1%: TPR / FPR | all label-0, 5%: TPR / FPR |")
    A("|---|---|---|---|---|---|---|---|---|")
    for d in R1_DETS:
        a1 = BU[d]["benign_hard"]["fpr01"]; a5 = BU[d]["benign_hard"]["fpr05"]
        b1 = BU[d]["all_label0"]["fpr01"]; b5 = BU[d]["all_label0"]["fpr05"]
        A(f"| {d} | {tf(a1['train_FPR'])} | {f3(a1['test_TPR'])} | {f3(a1['test_FPR'])} | {tf(a5['train_FPR'])} | {f3(a5['test_TPR'])} | {f3(a5['test_FPR'])} | "
          f"{b1['test_TPR']['point']:.3f} / {b1['test_FPR']['point']:.3f} | {b5['test_TPR']['point']:.3f} / {b5['test_FPR']['point']:.3f} |")
    A("")
    x = OP["detectors"]["LR-C"]["t_r80"]; xt = OP["detectors"]["TFIDF"]["t_r80"]
    A("**Eq. (6) at very low prevalence.**")
    A("")
    A(f"- LR-C at theta_r80: precision {f3(x['precision_pi0.1pct'], 4)} at 0.1% and {f3(x['precision_pi0.01pct'], 5)} at 0.01%, "
      f"with {f3(x['alerts_per_1000_pi0.1pct'], 0)} and {f3(x['alerts_per_1000_pi0.01pct'], 0)} alerts per 1,000 "
      f"[`R1_operating_points.detectors.LR-C.t_r80.precision_pi0.1pct` / `precision_pi0.01pct` / `alerts_per_1000_*`].")
    A(f"- LR-C at the 1% budget: {f3(BU['LR-C']['benign_hard']['fpr01']['precision_pi0.1pct'], 4)} at 0.1% and "
      f"{f3(BU['LR-C']['benign_hard']['fpr01']['precision_pi0.01pct'], 5)} at 0.01% "
      f"[`...m10_fpr_budgets.detectors.LR-C.benign_hard.fpr01.precision_pi0.1pct`].")
    A(f"- GRU at theta_r80, at 0.1%: {f3(OP['detectors']['GRU']['t_r80']['precision_pi0.1pct'], 4)}.")
    A(f"- TFIDF at theta_r80 flags no pool-E negative. With the CP upper FPR bound, its precision at 0.1% is at least "
      f"{xt['precision_pi0.1pct']['lower_bound_if_FPR_at_CP_upper']:.3f} "
      f"[`R1_operating_points.detectors.TFIDF.t_r80.precision_pi0.1pct.lower_bound_if_FPR_at_CP_upper`].")
    A("")
    A("**Plain reading.** " + M10_READING(R))
    A("")
    # ================================================================== m11
    A("## m11. Label-free threshold on the held-out domain (plan 3.7)")
    A("")
    A("**Question.** On each held-out R3 domain, set the threshold at the quantile of 200 unlabeled held-out traces that "
      "reproduces the R1 alert rate. What FPR and recall does this give on the remaining held-out traces?")
    A("")
    LF = R["m11_label_free"]; T = LF["R1_alert_rate_targets"]
    A("Two readings of 'the R1 alert rate' are reported [`m11_label_free.R1_alert_rate_targets`]:")
    A("")
    A("- (a) Pre-specified: the R1 alert rate at native prevalence on pool E, with a native-prevalence unlabeled sample. "
      f"Targets: LR-C {T['LR-C']['poolE'][0]:.3f}, LR-B {T['LR-B']['poolE'][0]:.3f}, GRU "
      f"{', '.join(f'{v:.3f}' for v in T['GRU']['poolE'])}.")
    A("- (b) Post hoc (P9): the R1 alert rate at 2% prevalence, with unlabeled traffic at 2% (4 attacked and 196 negative "
      f"traces per draw). Targets: LR-C {T['LR-C']['prev02'][0]:.3f}, LR-B {T['LR-B']['prev02'][0]:.3f}, GRU "
      f"{', '.join(f'{v:.3f}' for v in T['GRU']['prev02'])}.")
    A("")
    A("Values are mean [min, max] over 5 draws; for the GRU, over 3 seeds x 5 draws "
      "[`m11_label_free.folds.<domain>.detectors.<det>.<primary_poolE|deployment_prev02>.summary_over_draws`]. "
      "theta_r80 in the same fold is shown for comparison [`...theta_r80_same_fold_full_poolE`].")
    A("")
    A("| Domain | Detector | (a) FPR | (a) Recall | (b) FPR | (b) Recall | (b) Prec. 2% | theta_r80 FPR / recall |")
    A("|---|---|---|---|---|---|---|---|")
    for fold in DOMAINS:
        for d in LABELFREE_DETS:
            xa = LF["folds"][fold]["detectors"][d]["primary_poolE"]; xb = LF["folds"][fold]["detectors"][d]["deployment_prev02"]
            sa, sb, rf = xa["summary_over_draws"], xb["summary_over_draws"], xa["theta_r80_same_fold_full_poolE"]
            A(f"| {fold} | {d} | {mm(sa['FPR'])} | {mm(sa['recall'])} | {mm(sb['FPR'])} | {mm(sb['recall'])} | {mm(sb['precision_pi02'])} | {rf['FPR']:.3f} / {rf['recall']:.3f} |")
    A("")
    A("Sensitivity for reading (a), with failed attacks included in both the target and the draws "
      "[`...sensitivity_all_rows.summary_over_draws.FPR.mean`]. Mean FPR:")
    A("")
    for fold in DOMAINS:
        A(f"- {fold}: " + ", ".join(f"{d} {LF['folds'][fold]['detectors'][d]['sensitivity_all_rows']['summary_over_draws']['FPR']['mean']:.3f}" for d in LABELFREE_DETS) + ".")
    A("")
    A("**Plain reading.** " + M11_READING(R))
    A("")
    # ================================================================== m9
    A("## m9. Finite-sample chance AP and floating-point ties")
    A("")
    A("**Question.** What is the expected AP of a random scorer at the v0.3 subsample sizes? Does rounding scores to 12 "
      "decimals before ranking change any value?")
    A("")
    CA = R["m9_chance_AP"]["values"]
    A("Expected AP of a random scorer [`m9_chance_AP.values.<n+>/<n->`], as exact value (Monte Carlo mean of 20,000 random "
      "rankings; prevalence):")
    A("")
    A("| n+ / n- | exact | Monte Carlo mean | prevalence |")
    A("|---|---|---|---|")
    for k, v in CA.items():
        A(f"| {k} | {v['exact']:.4f} | {v['monte_carlo_mean']:.4f} | {v['prevalence']:.4f} |")
    A("")
    RC = R["m9_rounding_check"]
    nchg = [(s, d) for s in RC for d in RC[s] if not RC[s][d]["order_and_ties_identical"]]
    lfc = sum(LF["folds"][f]["detectors"][d][v]["n_pred_changed_by_rounding"] for f in DOMAINS for d in LABELFREE_DETS
              for v in ("primary_poolE", "sensitivity_all_rows", "deployment_prev02"))
    bc = OP["m10_fpr_budgets"]["rounding_changes_budget_predictions"]
    A(f"**Rounding check** [`m9_rounding_check.<S>.<det>`], over {sum(len(RC[s]) for s in RC)} (setting, detector) pairs "
      "in R1, R2, R3, R4 and WUT:")
    A("")
    A(f"- Rounding changes the ordering or tie structure of pool-E scores in {len(nchg)} pairs, all in AgentDojo settings: "
      + ", ".join(f"{s} {d}" for s, d in nchg) + ".")
    A("- In R4 the cause is saturated scores. Rows with nonzero scores below 5e-13 become tied at 0, e.g. "
      f"{RC['R4']['LR-C']['n_nonzero_scores_below_5e-13_by_rep'][0]} rows for LR-C. "
      f"R4 LR-C AUROC moves from {RC['R4']['LR-C']['AUROC_raw_by_rep'][0]:.4f} (raw, equal to the cache validation) to "
      f"{RC['R4']['LR-C']['AUROC_rounded_by_rep'][0]:.4f} (rounded). R4 GRU moves from "
      f"{', '.join(f'{v:.4f}' for v in RC['R4']['GRU']['AUROC_raw_by_rep'])} to {', '.join(f'{v:.4f}' for v in RC['R4']['GRU']['AUROC_rounded_by_rep'])}.")
    A("- No AgentDrift value changes: R1, R2 and every R3 fold have identical order and ties.")
    A(f"- Predictions of the new thresholds that differ between raw and rounded scores: {sum(bc.values())} for the FPR "
      f"budgets and {lfc} for the label-free thresholds.")
    A("- R3 LR-A and Markov AUROC are exactly 0.5000 in the cache, e.g. "
      f"R3_email LR-A {R['M9']['R3_email']['schemes']['traj_strat']['detectors']['LR-A']['AUROC_subgroups']['poolE_vs_poolE_neg']['point']:.4f} "
      "[`M9.R3_email.schemes.traj_strat.detectors.LR-A.AUROC_subgroups.poolE_vs_poolE_neg`].")
    A("")
    A("**Plain reading.** " + M9TIES_READING(R))
    A("")
    A("## Reproduction of the review's figures")
    A("")
    A("Each row sets a value quoted in REVIEW_REPORT.md next to ours [`review_reference_values.rows`; our key in `ours_key`].")
    A("")
    A("| Comment | Quantity | Review | Ours |")
    A("|---|---|---|---|")
    for r_ in R["review_reference_values"]["rows"]:
        rv, ov = r_["review"], r_["ours"]
        fmt = lambda v: (f"{v[0]:.3f}-{v[1]:.3f}" if isinstance(v, list) else f"{v:.4f}")
        A(f"| {r_['comment']} | {r_['quantity']} | {fmt(rv)} | {fmt(ov)} |")
    A("")
    A("## Deviations and interpretation notes")
    A("")
    for dv in R["deviations"]:
        A(f"- {dv}")
    A("")
    v = R["meta"]["R1_train_score_refit"]
    A("The R1 refit that recovers training scores reproduces the cached test scores exactly "
      "[`meta.R1_train_score_refit.verification`]: " +
      ", ".join(f"{d} max |diff| {x['max_abs_diff_test_scores']:.1e}" for d, x in v["verification"].items()) +
      ". The reconstructed GRU t_r80 differs from the cached value by "
      f"{', '.join(f'{x:.1e}' for x in v['GRU_reconstructed_t_r80_abs_diff_by_seed'])}. The self-tests of the estimator "
      "code against sklearn and evalkit pass, with maximum error "
      f"{max(R['meta']['self_tests_max_abs_error'].values()):.1e} [`meta.self_tests_max_abs_error`].")
    open(OUT_MD, "w", encoding="utf-8").write("\n".join(L) + "\n")
    log(f"wrote {OUT_MD}")


# ----- plain readings: templates filled from the JSON (qualitative statements checked against these numbers)
def _pk(M, d, k):
    return M["detectors"][d]["wAUPRC"][k]


def M8_READING(R):
    M = R["M8_R1"]; S = M["subsampling_v03_secondary"]; DJ = R["M8_AgentDojo"]; CR = R["R4_CONTROL_within_criteria_WUT"]["detectors"]
    w1, w2, w10 = _pk(M, "LR-C", "pi01"), _pk(M, "LR-C", "pi02"), _pk(M, "LR-C", "pi10")
    s1, s2, s10 = S["pi01"]["AP"]["LR-C"], S["pi02"]["AP"]["LR-C"], S["pi10"]["AP"]["LR-C"]
    cb = [d for d in R1_DETS if d != "TFIDF"]
    mx2 = max(cb, key=lambda d: _pk(M, d, "pi02")["point"])
    T = DJ["schemes"]["traj_strat"]["settings"]
    tr = {d: T["R4"]["detectors"][d]["pi10"]["point"] for d in DOJO_DETS}
    wi = {d: T["WUT"]["detectors"][d]["pi10"]["point"] for d in DOJO_DETS}
    ch10 = DJ["chance_wAUPRC_random_scorer"]["pi10"]["mean"]
    return (f"Estimated from all 831 positives, LR-C's AUPRC at 2% is {w2['point']:.3f} [{w2['lo']:.3f}, {w2['hi']:.3f}]. "
            f"This interval is {R['summary_derived']['M8_LR-C_pi02_width_ratio_subsampling_over_weighted']:.1f} times narrower than the v0.3 "
            f"subsampling spread [{s2['lo']:.3f}, {s2['hi']:.3f}]. The subsampling mean is higher than the weighted value "
            f"at 1% ({s1['point']:.3f} vs {w1['point']:.3f}) and at 2% ({s2['point']:.3f} vs {w2['point']:.3f}), and the "
            f"two agree at 10% ({s10['point']:.3f} vs {w10['point']:.3f}). Detectors rank in the same order under both "
            "estimators; what changes is power. With the weighted estimator, LR-C minus Markov is detected at every "
            "prevalence, 1% included. GRU minus LR-C and LR-C minus LR-B are detected from 2% upward, both close to the "
            "Holm threshold at 2%. Under subsampling, only TFIDF minus LR-C survives Holm at 1% and at 2%. The RQ1 message "
            f"stands and is sharper. At 2% the best content-blind detector ({mx2}) has AUPRC {_pk(M, mx2, 'pi02')['point']:.3f}, "
            f"although its AUROC is {M['detectors'][mx2]['AUROC']['point']:.3f}. The Isolation Forest sits at the "
            f"estimator's chance level, and only the content detector TFIDF is near 1 ({_pk(M, 'TFIDF', 'pi02')['point']:.3f}). "
            "On AgentDojo, reweighting removes the need for 21- or 44-positive subsamples. Transfer AUPRC at 10% is "
            f"{min(tr[d] for d in ('LR-B', 'LR-C', 'GRU', 'TFIDF')):.3f}-{max(tr[d] for d in ('LR-B', 'LR-C', 'GRU', 'TFIDF')):.3f} "
            f"for LR-B, LR-C, GRU and TFIDF, against a chance level of {ch10:.3f}, and {tr['LR-A']:.3f} for LR-A. Within "
            f"AgentDojo it is {min(wi[d] for d in ('LR-A', 'LR-B', 'LR-C', 'GRU')):.3f}-{max(wi[d] for d in ('LR-A', 'LR-B', 'LR-C', 'GRU')):.3f} "
            f"for the content-blind detectors and {wi['TFIDF']:.3f} for TFIDF. Within minus transfer is detected for every "
            f"detector under both bootstraps. LR-C's within-AgentDojo wAUPRC at 10% "
            f"({CR['LR-C']['weighted_strat_rounded']['AUPRC10']:.3f}; {CR['LR-C']['weighted_strat_raw']['AUPRC10']:.4f} "
            "unrounded) falls just below the 0.30 'reasonable' threshold of R4_CONTROL.md. Its v0.3 subsampling estimate "
            f"({CR['LR-C']['subsampling_v03_AUPRC10']['AUPRC10']:.3f}) passes that threshold, so the choice of estimator "
            "flips this criterion for LR-C. The numbers do not show what AUPRC a deployment would see. Reweighting assumes "
            "that the class-conditional score distributions of the test pool carry over, including its 27% hard-negative "
            "share. The intervals cover test-set sampling and seed or repetition choice, not retraining.")


def M3_READING(R):
    M = R["M8_R1"]; nd = M["n_detected_holm"]
    return (f"With the weighted estimator, {nd['weighted']['pi01']} of five comparisons survive Holm at 1% and "
            f"{nd['weighted']['pi02']}, {nd['weighted']['pi05']} and {nd['weighted']['pi10']} at 2%, 5% and 10%. Under "
            f"v0.3 subsampling the counts are {nd['subsampling_AP']['pi01']}, {nd['subsampling_AP']['pi02']}, "
            f"{nd['subsampling_AP']['pi05']} and {nd['subsampling_AP']['pi10']}. Two results at 2% are fragile: LR-C minus "
            "LR-B and GRU minus LR-C pass Holm with adjusted p just below 0.05. Under world clustering, LR-C minus LR-B at 2% "
            "is no longer detected. Within AgentDojo, LR-C minus LR-B at 10% passes Holm only with unrounded scores, and it is "
            "not detected under the cluster bootstrap. The paper should report LR-C > LR-B at 2% (R1) and at 10% (WUT) as "
            "borderline, and should not base a claim on any single comparison at 1%.")


def EQ6_READING(R):
    OP = R["R1_operating_points"]["detectors"]
    x = OP["LR-C"]["t_r80"]; g = OP["GRU"]["t_r80"]; t = OP["TFIDF"]["t_r80"]
    return (f"At theta_r80, LR-C flags {x['TPR']['point']:.3f} of R1 attacks and {x['FPR']['point']:.3f} of negatives. "
            f"Eq. (6) then gives a precision of {x['precision_pi02']['point']:.3f} [{x['precision_pi02']['lo']:.3f}, "
            f"{x['precision_pi02']['hi']:.3f}] at 2%, with {x['alerts_per_1000_pi02']['point']:.0f} alerts per 1,000 traces, "
            f"or {R['summary_derived']['EQ6_LR-C_t_r80_false_alerts_per_true_alert_pi02']:.1f} false alerts per true alert. The GRU's lower FPR "
            f"({g['FPR']['point']:.3f}) gives {g['precision_pi02']['point']:.3f} at 2% and "
            f"{g['alerts_per_1000_pi02']['point']:.0f} alerts per 1,000, at a recall of {g['TPR']['point']:.3f}, below the 0.80 "
            "target. TFIDF flags no negative in pool E. Its bootstrap precision is therefore degenerate at 1, and the "
            "Clopper-Pearson bound only guarantees at least "
            f"{t['precision_pi02']['lower_bound_if_FPR_at_CP_upper']:.3f} at 2%. All values refer to the R1 pool-E negative "
            "mix (M9 shows how the mix changes them). Eq. (6) assumes that TPR and FPR do not change with prevalence.")


def M9_READING(R):
    X = R["M9"]["R1"]["schemes"]["traj_strat"]["detectors"]
    c = X["LR-C"]["t_r80"]; p = c["precision_eq6"]["pi02"]
    sub = R["M9"]["R1"]["schemes"]["traj_strat"]["paired_AUROC_subgroups"]["LR-C-minus-LR-B"]
    gB = X["LR-B"]["AUROC_subgroups"]; gC = X["LR-C"]["AUROC_subgroups"]
    trained = ["LR-A", "LR-B", "LR-C", "Markov", "GRU"]
    hard_higher = all(X[d]["t_r80"]["FPR_hard_negative"]["point"] > X[d]["t_r80"]["FPR_benign"]["point"] for d in trained)
    r3 = {d: R["M9"][f"R3_{d}"]["schemes"]["traj_strat"]["detectors"]["LR-B"]["t_r80"]["recall_full"]["point"] for d in DOMAINS}
    r4 = R["M9"]["R4"]["schemes"]["traj_strat"]["detectors"]["LR-C"]["t_r80"]
    wu = R["M9"]["WUT"]["schemes"]["traj_strat"]["detectors"]
    return (f"In R1, LR-C at theta_r80 flags {c['FPR_benign']['point']:.3f} of benign traces, "
            f"{c['FPR_hard_negative']['point']:.3f} of hard negatives and {c['flag_failed_attack']['point']:.3f} of failed "
            f"attacks. Its recall is {c['recall_full']['point']:.3f} on full, {c['recall_partial']['point']:.3f} on partial "
            f"and {c['recall_delayed']['point']:.3f} on delayed hijacks. Its precision at 2% therefore ranges from "
            f"{p['benign_only']['point']:.3f} with benign-only negatives, through {p['poolE_mix']['point']:.3f} with the pool-E "
            f"mix, to {p['hard_only']['point']:.3f} with hard negatives only. At fixed prevalence the negative mix changes "
            f"precision by a factor of {R['summary_derived']['M9_R1_LR-C_pi02_precision_ratio_benign_only_over_hard_only']:.1f}, so every precision figure must state its mix. "
            + ("In R1 every trained detector (LR-A, LR-B, LR-C, Markov, GRU) flags hard negatives more often than benign "
               "traces. " if hard_higher else "") +
            f"LR-B ranks partial hijacks slightly better than LR-C (AUROC {gB['partial_vs_poolE_neg']['point']:.3f} vs "
            f"{gC['partial_vs_poolE_neg']['point']:.3f}; LR-C minus LR-B {fd(sub['partial_vs_poolE_neg'])}, not detected). "
            f"LR-C's overall advantage comes from full hijacks ({fd(sub['full_vs_poolE_neg'])}), from hard negatives "
            f"({fd(sub['poolE_vs_hard_negative'])}) and from failed attacks ({fd(sub['poolE_vs_failed'])}). Under domain "
            "shift, LR-B's full-hijack recall at theta_r80 is " + ", ".join(f"{v:.3f} ({d})" for d, v in r3.items()) +
            f". In R4 transfer, LR-C flags {r4['FPR_benign']['point']:.3f} of AgentDojo benign traces, about as often as it "
            f"flags compromised ones ({r4['recall_poolE']['point']:.3f}), and {r4['flag_failed_attack']['point']:.3f} of "
            "unsuccessful attacks. Within AgentDojo, TFIDF flags no benign trace but "
            f"{wu['TFIDF']['t_r80']['flag_failed_attack']['point']:.3f} of unsuccessful attacks. Its AUROC is "
            f"{wu['TFIDF']['AUROC_subgroups']['poolE_vs_failed']['point']:.3f} against unsuccessful attacks and "
            f"{wu['TFIDF']['AUROC_subgroups']['poolE_vs_poolE_neg']['point']:.3f} against benign traces. So it largely "
            "detects injected content rather than compromise: if unsuccessful attacks count as negatives, its precision at "
            f"2% is {wu['TFIDF']['t_r80']['precision_eq6']['pi02']['failed_only']['point']:.3f}. The breakdown shows where "
            "alerts come from. It does not say which negative mix a deployment would have, and the AgentDrift hard "
            "negatives are synthetic.")


def M4_READING(R):
    g = R["m4_gru_t_val80"]["table"]["R1"]["GRU"]; M = R["M8_R1"]["detectors"]; T = R["m4_theta05"]["table"]
    r3c = [T[f"R3_{d}"]["LR-C"]["FPR_poolE"]["point"] for d in DOMAINS]
    r3g = [T[f"R3_{d}"]["GRU"]["FPR_poolE"]["point"] for d in DOMAINS]
    return (f"With its validation threshold, the GRU's recall rises from {g['t_r80']['recall_poolE']['point']:.3f} to "
            f"{g['t_val80']['recall_poolE']['point']:.3f} and its FPR from {g['t_r80']['FPR_poolE']['point']:.3f} to "
            f"{g['t_val80']['FPR_poolE']['point']:.3f}. Precision at 2% falls from "
            f"{g['t_r80']['precision_pi02_poolE_mix']['point']:.3f} to {g['t_val80']['precision_pi02_poolE_mix']['point']:.3f}. "
            "The in-sample threshold was optimistic about recall and flattered the GRU's precision. At a matched test "
            f"recall of 0.8, the GRU's FPR is {M['GRU']['FPR_at_test_TPR80']['FPR_at_TPR80']['point']:.3f} against "
            f"{M['LR-C']['FPR_at_test_TPR80']['FPR_at_TPR80']['point']:.3f} for LR-C, so its advantage in R1 does not depend "
            f"on the threshold. At theta = 0.5 the GRU in R1 has FPR {T['R1']['GRU']['FPR_poolE']['point']:.3f} and recall "
            f"{T['R1']['GRU']['recall_poolE']['point']:.3f}. Under domain shift the default cut fails like theta_r80: FPR "
            f"{min(r3c):.3f}-{max(r3c):.3f} for LR-C and {min(r3g):.3f}-{max(r3g):.3f} for the GRU. LR-A flags nothing in any R3 "
            "fold. In R4, recall at 0.5 is at most "
            f"{max(T['R4'][d]['recall_poolE']['point'] for d in THETA05_DETS):.3f} for every detector. The matched-recall "
            "numbers use test labels and are not achievable operating points.")


def M10_READING(R):
    BU = R["R1_operating_points"]["m10_fpr_budgets"]["detectors"]
    c1, c5 = BU["LR-C"]["benign_hard"]["fpr01"], BU["LR-C"]["benign_hard"]["fpr05"]
    a1, a5 = BU["LR-C"]["all_label0"]["fpr01"], BU["LR-C"]["all_label0"]["fpr05"]
    x = R["R1_operating_points"]["detectors"]["LR-C"]["t_r80"]
    return (f"With thresholds that flag at most 1% (or 5%) of training negatives, LR-C catches {c1['test_TPR']['point']:.3f} "
            f"({c5['test_TPR']['point']:.3f}) of R1 attacks, at a test FPR of {c1['test_FPR']['point']:.3f} "
            f"({c5['test_FPR']['point']:.3f}). The test FPR exceeds the budget because training scores are in-sample. The "
            f"overshoot is largest for the flexible models: at the 1% budget the GRU's test FPR is "
            f"{BU['GRU']['benign_hard']['fpr01']['test_FPR']['point']:.3f} and TFIDF's "
            f"{BU['TFIDF']['benign_hard']['fpr01']['test_FPR']['point']:.3f}. Counting all label-0 training rows as "
            f"negatives gives LR-C {a1['test_TPR']['point']:.3f} / {a1['test_FPR']['point']:.3f} and "
            f"{a5['test_TPR']['point']:.3f} / {a5['test_FPR']['point']:.3f}, which are the review's figures. At 0.1% and "
            f"0.01% prevalence, LR-C at theta_r80 has precision {x['precision_pi0.1pct']['point']:.4f} and "
            f"{x['precision_pi0.01pct']['point']:.5f}. The 1% budget only lifts this to "
            f"{c1['precision_pi0.1pct']['point']:.3f} at 0.1%. The best content-blind value at 0.1% over all thresholds "
            f"examined is {R['summary_derived']['m10_max_precision_pi0.1pct_content_blind_R1']['value']:.4f} "
            f"({R['summary_derived']['m10_max_precision_pi0.1pct_content_blind_R1']['detector']}, "
            f"{R['summary_derived']['m10_max_precision_pi0.1pct_content_blind_R1']['threshold']}). These are extrapolations "
            "by Eq. (6) and assume that TPR and FPR stay fixed.")


def M11_READING(R):
    LF = R["m11_label_free"]["folds"]
    def rng_(d, v, q):
        vals = [LF[f]["detectors"][d][v]["summary_over_draws"][q]["mean"] for f in DOMAINS]
        return f"{min(vals):.3f}-{max(vals):.3f}"
    r80 = [LF[f]["detectors"]["LR-C"]["primary_poolE"]["theta_r80_same_fold_full_poolE"]["FPR"] for f in DOMAINS]
    return (f"Setting the threshold without labels lowers LR-C's held-out FPR from {min(r80):.3f}-{max(r80):.3f} at "
            f"theta_r80. Under reading (a) the mean FPR across folds is {rng_('LR-C', 'primary_poolE', 'FPR')}, with recall "
            f"{rng_('LR-C', 'primary_poolE', 'recall')}. Under reading (b) it is {rng_('LR-C', 'deployment_prev02', 'FPR')}, "
            f"with recall {rng_('LR-C', 'deployment_prev02', 'recall')}, close to the review's FPR 0.191-0.209 and recall "
            "0.483-0.728 (see the reproduction table). For the GRU, "
            f"reading (b) gives FPR {rng_('GRU', 'deployment_prev02', 'FPR')} but recall only "
            f"{rng_('GRU', 'deployment_prev02', 'recall')}. LR-B already had moderate FPR at theta_r80, and the remedy makes "
            f"it worse in some folds (reading (b): {rng_('LR-B', 'deployment_prev02', 'FPR')}). The remedy restores an alert "
            "budget; it does not restore ranking quality. Precision at 2% stays at "
            f"{rng_('LR-C', 'deployment_prev02', 'precision_pi02')} for LR-C. It also works only if the unlabeled sample "
            "has the prevalence that the target alert rate assumes. Reading (a) matches LR-C's native R1 alert rate "
            f"({R['m11_label_free']['R1_alert_rate_targets']['LR-C']['poolE'][0]:.3f}) on held-out pools that are about half "
            f"attacks. Reading (b) matches its 2%-prevalence alert rate ({R['m11_label_free']['R1_alert_rate_targets']['LR-C']['prev02'][0]:.3f}) "
            "on traffic at 2% prevalence.")


def M9TIES_READING(R):
    CA = R["m9_chance_AP"]["values"]
    return (f"A random scorer's expected AP is {CA['8/825']['exact']:.4f} at 8 positives and {CA['17/825']['exact']:.4f} "
            f"at 17, not the prevalence ({CA['8/825']['prevalence']:.4f} and {CA['17/825']['prevalence']:.4f}). The chance "
            "lines of the v0.3 subsample figures should use these values. The Isolation Forest's v0.3 subsample APs (0.017 "
            "and 0.030) are at this chance level. For the weighted estimator the chance level is close to pi (M8). The "
            "caches already store tie-exact scores, so rounding is only needed for code that recomputes scores with float "
            "noise (v0.3 R3 LR-A 0.499). Applied to AgentDojo scores, 12-decimal rounding creates ties between distinct "
            "saturated probabilities. It changes point AUROC or wAUPRC by up to "
            f"{R['summary_derived']['m9_max_abs_change_from_rounding']['value']:.4f} "
            f"({R['summary_derived']['m9_max_abs_change_from_rounding']['setting']} {R['summary_derived']['m9_max_abs_change_from_rounding']['detector']}). "
            "For AgentDojo we therefore recommend the unrounded values, which are reported alongside.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["train_scores", "analyze", "derive", "summary", "all"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--budget", type=float, default=480.0, help="seconds per 'analyze' call before it stops (resumable)")
    a = ap.parse_args()
    if a.stage in ("train_scores", "all"):
        if os.path.exists(TRAIN_PKL) and not a.force and a.stage == "all":
            log("train_scores exists, skipped")
        else:
            stage_train_scores()
    if a.stage in ("analyze", "all"):
        if not stage_analyze(a.budget):
            print("INCOMPLETE", flush=True)
            return
    if a.stage in ("analyze", "derive", "all"):
        stage_derive()
        print("ALL_DONE", flush=True)
    if a.stage in ("summary", "all"):
        stage_summary()


if __name__ == "__main__":
    main()
