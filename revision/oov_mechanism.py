"""Out-of-vocabulary mechanism of the R3 reversals (review M1 actions 1-5, Q1, m6 <UNK>-norm part).

Reads the shared prediction caches (revision/cache/linear, revision/cache/gru). The only new fits are the six R1
GRU models (GRU and GRU-drop0.2, training seeds 0-2) that the cache did not save; they are refitted with the
cache's own training function (gru_variants.train_gru_drop, called exactly as build_gru_cache.run_unit does) and
checked to reproduce the cached scores. Nothing is tuned on test data: every threshold is the cached training
threshold (theta_r80).

Sections (keys of oov_mechanism_results.json):
  s1_lrc_offset   LR-C / LR-C-drop tool-block offset w_A^T c per R3 fold (from lrc_parts) + share of training rows
  s2_graded       graded tool coverage inside R1 (tool columns zeroed / tokens mapped to <UNK> for a random fraction
                  of tool names; 5 draws per level)
  s3_dropout      dropout arms in R1, domain-matched R1 and R3; paired contrasts with Holm; plan 3.4 reversal rule
  s4_gru_offset   GRU tool-pathway logit offset over all folds and seeds; <UNK> embedding norm (init reproduced by
                  re-seeding; trained norm reproduced by simulating Adam's coupled weight decay alone)
  s5_loso         AgentDojo leave-one-suite-out: coverage, C-B / Cdrop-B / Cdrop-C with a suite|user_task cluster
                  bootstrap, and the in-distribution (WUT) references
  s6_rule4        pre-registered Rule 4 (PRE_ABLATION_DESIGN.md) on the corrected R4 data
  s7_q1           per-seed GRU R3 AUROC, GRU - LR-C per seed, two-level (seeds, then trajectories) bootstrap

Usage (Git Bash; always set AGENTDRIFT and AGENTDOJO):
  python oov_mechanism.py train      # resumable: fits the 6 R1 GRU models not saved by the cache (~3 min)
  python oov_mechanism.py graded     # resumable: scores R1 pool E under every coverage mask
  python oov_mechanism.py analyze    # all bootstraps -> oov_mechanism_results.json
  python oov_mechanism.py summary    # oov_mechanism_summary.md from the results JSON
  python oov_mechanism.py all        # the four steps in order (each skips finished work)
"""
# Pickle is the shared cache format fixed by the revision plan; only files written by the revision scripts on this
# machine (cache builders, and this script's own work files) are loaded.
import os, sys, json, time, pickle, hashlib, datetime, collections, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)

import scipy.sparse as sp  # noqa: E402
from scipy.special import expit  # noqa: E402
import torch  # noqa: E402
import evalkit as ek  # noqa: E402
from featurise import load, vectorise, tool_seq  # noqa: E402
import gru_detector as gd  # noqa: E402
from gru_detector import GRUDet, _tensors, score_gru, UNK, PAD, LR, WD, BATCH  # noqa: E402
from gru_ladder import stratified_val  # noqa: E402
from gru_variants import train_gru_drop, decompose, embedding_norms  # noqa: E402
import ablation  # noqa: E402  (coverage() reused for LOSO)

MP = os.environ["AGENTDRIFT"]
CACHE = os.path.join(HERE, "cache")
WORK = os.path.join(HERE, "oov_mechanism_work")
RESULTS = os.path.join(HERE, "oov_mechanism_results.json")
SUMMARY = os.path.join(HERE, "oov_mechanism_summary.md")
LOGF = os.path.join(HERE, "logs", "oov_mechanism.log")
DOMAINS = ["medical", "banking", "web", "email", "coding"]
SUITES = ["banking", "slack", "travel", "workspace"]
SEEDS = [0, 1, 2]
B = 2000                       # bootstrap replicates everywhere (fixed seeds below)
LEVELS = [1.0, 0.75, 0.5, 0.25, 0.0]
N_DRAWS = 5
GRADED_SEED = 7100             # mask draw d of level index li uses default_rng(GRADED_SEED + 100*li + d)
R1_GRU_DETS = {"GRU": 0.0, "GRU-drop0.2": 0.2}
LIN = ["LR-B", "LR-C", "LR-C-drop"]
GRUS = ["GRU", "GRU-drop0.2", "GRU-drop0.1"]
DETS = LIN + GRUS
PAIRS = [("LR-C", "LR-B"), ("GRU", "LR-C"), ("GRU", "LR-B"),
         ("LR-C-drop", "LR-B"), ("LR-C-drop", "LR-C"),
         ("GRU-drop0.2", "LR-C"), ("GRU-drop0.2", "LR-B"),
         ("GRU-drop0.1", "LR-C"), ("GRU-drop0.1", "LR-B"),
         ("GRU-drop0.2", "LR-C-drop"), ("GRU-drop0.2", "GRU"), ("GRU-drop0.1", "GRU")]
BUDGET = 540.0                 # seconds; a call stops starting new model fits after this
T0 = time.time()


# ============================================================================ utilities
def log(msg):
    os.makedirs(os.path.dirname(LOGF), exist_ok=True)
    line = f"{datetime.datetime.now().isoformat(timespec='seconds')} {msg}"
    print(line, flush=True)
    with open(LOGF, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def pload(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def pdump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=4)
    os.replace(tmp, path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_C = {}


def cache(fam, s):
    k = (fam, s)
    if k not in _C:
        _C[k] = pload(os.path.join(CACHE, fam, f"{s}.pkl"))
    return _C[k]


def r4(x, d=4):
    if x is None:
        return None
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    x = float(x)
    return None if not np.isfinite(x) else round(x, d)


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return [jsonable(v) for v in o.tolist()]
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    return o


# ============================================================================ bootstrap engine
# All bootstraps are expressed as integer weight matrices W (B x n): W[b, i] = how often row i is drawn in replicate b.
# Stratified: n_pos positives and n_neg negatives drawn with replacement within class (evalkit.bootstrap design).
# Cluster: n_clusters clusters drawn with replacement; every row of a drawn cluster gets that cluster's count.
def _counts(B_, k, rng):
    draws = rng.integers(0, k, size=(B_, k))
    flat = (draws + (np.arange(B_) * k)[:, None]).ravel()
    return np.bincount(flat, minlength=B_ * k).reshape(B_, k)


def strat_W(y, rng, B_=B):
    W = np.zeros((B_, len(y)), np.float64)
    for c in (0, 1):
        ids = np.where(y == c)[0]
        W[:, ids] = _counts(B_, len(ids), rng)
    return W


def cluster_W(clusters, rng, B_=B):
    u, inv = np.unique(np.asarray(clusters, dtype=object).astype(str), return_inverse=True)
    return _counts(B_, len(u), rng)[:, inv].astype(np.float64), len(u)


# One resample per population, shared by every section, so that the same estimand gets the same interval everywhere.
POOL_SEEDS = {"R1": 20201}
POOL_SEEDS.update({f"R1dm_{d}": 20210 + j for j, d in enumerate(["medical", "banking", "web", "email", "coding"])})
POOL_SEEDS.update({f"R3_{d}": 20220 + j for j, d in enumerate(["medical", "banking", "web", "email", "coding"])})
POOL_SEEDS.update({"P0_cluster": 50900, "P0_strat": 50902, "C1_cluster": 50910, "C1_strat": 50912})
POOL_SEEDS.update({f"LOSO_{x}_cluster": 50500 + j for j, x in enumerate(["banking", "slack", "travel", "workspace"])})
POOL_SEEDS.update({f"LOSO_{x}_strat": 50600 + j for j, x in enumerate(["banking", "slack", "travel", "workspace"])})
POOL_SEEDS.update({f"WUTsuite_{x}_cluster": 50700 + j for j, x in enumerate(["banking", "slack", "travel", "workspace"])})
REP_SEEDS = {"P0": 50901, "C1": 50911}
REP_SEEDS.update({f"WUTsuite_{x}": 50800 + j for j, x in enumerate(["banking", "slack", "travel", "workspace"])})
_POOLW = {}


def pool_W(name, y=None, clusters=None):
    """Weight matrix of population `name` (stratified if y is given, cluster if clusters is given); cached, so every
    section that bootstraps this population uses the identical resample. Returns (W, n_clusters or None)."""
    if name not in _POOLW:
        rng = np.random.default_rng(POOL_SEEDS[name])
        if clusters is not None:
            _POOLW[name] = cluster_W(clusters, rng)
        else:
            _POOLW[name] = (strat_W(np.asarray(y), rng), None)
    return _POOLW[name]


def rep_choice(name):
    """Shared choice of the WUT CV repetition per replicate (paired across detectors and sections)."""
    return np.random.default_rng(REP_SEEDS[name]).integers(5, size=B)


def wauroc(s, y, W):
    """Weighted Mann-Whitney AUROC (ties count 1/2) for every row of W. Equals evalkit.auroc when W = 1."""
    s = np.asarray(s, np.float64); y = np.asarray(y)
    u, inv = np.unique(s, return_inverse=True); n = len(s); U = len(u)
    inv = np.asarray(inv).reshape(-1)
    P = sp.csr_matrix(((y == 1).astype(float), (np.arange(n), inv)), shape=(n, U))
    N = sp.csr_matrix(((y == 0).astype(float), (np.arange(n), inv)), shape=(n, U))
    Wp = np.asarray((P.T @ W.T).T); Wn = np.asarray((N.T @ W.T).T)
    below = np.cumsum(Wn, axis=1) - Wn
    num = (Wp * (below + 0.5 * Wn)).sum(1); den = Wp.sum(1) * Wn.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def wrate(pred, y, cls, W):
    m = (np.asarray(y) == cls)
    num = W[:, m] @ np.asarray(pred, float)[m]; den = W[:, m].sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def point_auc(s, y):
    return ek.auroc(np.asarray(y), np.asarray(s, np.float64))


def point_rate(pred, y, cls):
    m = np.asarray(y) == cls
    return float(np.mean(np.asarray(pred)[m])) if m.any() else float("nan")


def pctl(a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    if len(a) == 0:
        return [None, None]
    return [r4(np.percentile(a, 2.5)), r4(np.percentile(a, 97.5))]


def p_boot(d):
    """Two-sided bootstrap p-value for H0: difference = 0 (add-one): 2 * min(#<=0, #>=0) share, capped at 1."""
    d = np.asarray(d, float); d = d[np.isfinite(d)]
    if len(d) == 0:
        return None
    lo = int(np.sum(d <= 0)) + 1; hi = int(np.sum(d >= 0)) + 1
    return float(min(1.0, 2.0 * min(lo, hi) / (len(d) + 1)))


def holm(ps):
    ps = np.asarray([1.0 if p is None else p for p in ps], float); m = len(ps)
    order = np.argsort(ps, kind="stable"); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * ps[i])); adj[i] = run
    return adj


def summ(point, boot):
    ci = pctl(boot)
    return {"point": r4(point), "ci": ci}


def contrast(pa, pb, ba, bb):
    d = np.asarray(ba) - np.asarray(bb)
    ci = pctl(d)
    det = (ci[0] is not None) and (ci[0] > 0 or ci[1] < 0)
    return {"point": r4(pa - pb), "ci": ci, "p_boot": r4(p_boot(d), 5), "detected": bool(det), "_boot": d}


def add_holm(items):
    """items: list of contrast dicts (same family). Adds p_holm and detected_holm in place."""
    adj = holm([it["p_boot"] for it in items])
    for it, a in zip(items, adj):
        it["p_holm"] = r4(a, 5); it["detected_holm"] = bool(a < 0.05)


def strip(o):
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, list):
        return [strip(v) for v in o]
    return o


def select_reps(M, mode, rng=None, k=None):
    """M: (R, B) metric per repetition and replicate. mixture: one repetition per replicate (pre-registered GRU
    convention, evalkit); mean: average over repetitions; twolevel: R repetitions drawn with replacement, averaged;
    shared: use the given index vector k."""
    M = np.atleast_2d(M); R, B_ = M.shape
    if R == 1:
        return M[0]
    if mode == "mean":
        return M.mean(0)
    if mode == "mixture":
        kk = rng.integers(R, size=B_); return M[kk, np.arange(B_)]
    if mode == "twolevel":
        K = rng.integers(R, size=(B_, R)); return M[K, np.arange(B_)[:, None]].mean(1)
    if mode == "shared":
        return M[k, np.arange(B_)]
    raise ValueError(mode)


# ============================================================================ data
_D = {}


def drift():
    if "st" not in _D:
        _D["st"] = (load(os.path.join(MP, "data", "train.jsonl")), load(os.path.join(MP, "data", "test.jsonl")))
    return _D["st"]


def dojo_p0():
    if "p0" not in _D:
        _D["p0"] = [json.loads(l) for l in open(os.path.join(ROOT, "agentdojo_traces.jsonl"), encoding="utf-8")]
    return _D["p0"]


# ============================================================================ step 1: R1 GRU models
def model_path(det, seed):
    return os.path.join(WORK, "models", f"R1_{det}_seed{seed}.pkl")


def cmd_train():
    st_tr, st_te = drift()
    g = cache("gru", "R1")
    pE = g["pool_E"]
    teE = [r for r, m in zip(st_te, g["test_meta"]) if m["in_pool_E"]]
    assert [r["id"] for r in st_te] == [m["id"] for m in g["test_meta"]]
    for det, p in R1_GRU_DETS.items():
        for seed in SEEDS:
            path = model_path(det, seed)
            if os.path.exists(path):
                continue
            if time.time() - T0 > BUDGET:
                log("train: budget used, call again"); return False
            t0 = time.time()
            tr, va = stratified_val(st_tr, seed)                      # exactly build_gru_cache.run_unit
            model, vocab, scaler, thr, aux = train_gru_drop(tr, va, seed, p)
            aux.pop("Xtr", None); aux.pop("s_train", None); aux.pop("y_train", None)
            d = score_gru(model, vocab, scaler, thr, teE)            # pool-E rows in one pass, as the cache
            ref = g["scores"][det][seed][pE]
            chk = {"max_abs_diff_scores_vs_cache": float(np.max(np.abs(d["score"].astype(np.float64) - ref))),
                   "thr_r80_diff_vs_cache": float(abs(thr - float(g["thr"][det]["t_r80"][seed]))),
                   "thr_val80_diff_vs_cache": float(abs(aux["thr_val80"] - float(g["thr"][det]["t_val80"][seed])))}
            X = _tensors(teE, vocab, scaler)
            tc, _, _, _ = decompose(model, X)
            chk["max_abs_diff_tool_contrib_vs_cache"] = float(np.max(np.abs(tc - g["tool_contrib"][det][seed][pE])))
            obj = {"det": det, "seed": seed, "p": p, "vocab": vocab, "scaler": scaler, "thr": thr, "aux": aux,
                   "state_dict": {k: v.detach().clone() for k, v in model.state_dict().items()},
                   "check": chk, "fit_seconds": time.time() - t0}
            pdump(obj, path)
            log(f"train R1 {det} seed{seed}: {time.time() - t0:.1f}s check {chk}")
    return True


def load_model(det, seed):
    obj = pload(model_path(det, seed))
    m = GRUDet(len(obj["vocab"]))
    m.load_state_dict(obj["state_dict"]); m.eval()
    return m, obj


# ============================================================================ step 2: graded-coverage scores
def graded_masks(names):
    """Masks (sets of tool names treated as unseen) per level. Level L masks round((1-L)*|V|) names drawn at random
    from the R1 benign-training vocabulary. Levels 1 and 0 have a single (deterministic) mask."""
    out = collections.OrderedDict()
    V = len(names)
    for li, L in enumerate(LEVELS):
        k = int(round((1 - L) * V))
        if k == 0:
            out[(L, 0)] = []
        elif k == V:
            out[(L, 0)] = list(names)
        else:
            for dr in range(N_DRAWS):
                rng = np.random.default_rng(GRADED_SEED + 100 * li + dr)
                out[(L, dr)] = sorted(rng.choice(np.array(names, dtype=object), k, replace=False).tolist())
    return out


def r1_linear_parts():
    st_tr, st_te = drift()
    L1 = cache("linear", "R1")
    assert [r["id"] for r in st_te] == [m["id"] for m in L1["test_meta"]]
    from featurise import build_vocab
    vocab = build_vocab([r for r in st_tr if r["category"] == "benign"])
    names = [t for t, _ in sorted(vocab.items(), key=lambda kv: kv[1])]
    assert L1["coefs"]["LR-C"]["names"][10:] == ["tool:" + t for t in names]
    pE = L1["pool_E"]
    teE = [r for r, m in zip(st_te, L1["test_meta"]) if m["in_pool_E"]]
    X = np.array([np.concatenate(vectorise(r, vocab)) for r in teE], float)
    return vocab, names, teE, X


def lr_score(cf, X):
    """sigmoid of the cached LR fit (scaler + coefficients), scored once per distinct row (tie-exact, as the cache)."""
    U, inv = np.unique(X, axis=0, return_inverse=True)
    Z = (U - cf["scaler_mean"]) / cf["scaler_scale"]          # StandardScaler.transform
    z = (Z @ np.asarray(cf["coef"]).reshape(-1, 1)).ravel() + cf["intercept"]   # decision_function
    s = expit(z)                                                                # predict_proba[:, 1]
    Zfull = (X - cf["scaler_mean"]) / cf["scaler_scale"]
    tool = Zfull[:, 10:] @ cf["coef"][10:]
    return s[np.asarray(inv).reshape(-1)], tool


def cmd_graded():
    path = os.path.join(WORK, "graded_scores.pkl")
    if os.path.exists(path):
        return True
    vocab, names, teE, X = r1_linear_parts()
    L1 = cache("linear", "R1"); pE = L1["pool_E"]
    masks = graded_masks(names)
    col = {t: 10 + i for t, i in vocab.items()}
    out = {"names": names, "masks": {f"{L}|{d}": m for (L, d), m in masks.items()}, "lin": {}, "gru": {}, "cov": {},
           "n_tools_masked": {}, "checks": {}}
    for (L, d), m in masks.items():
        key = f"{L}|{d}"
        mset = set(m)
        occ = [s["tool"] in vocab and s["tool"] not in mset for r in teE for s in r["steps"]]
        out["cov"][key] = float(np.mean(occ)); out["n_tools_masked"][key] = len(m)
        Xm = X.copy()
        for t in m:
            Xm[:, col[t]] = 0.0
        for det in ["LR-C", "LR-C-drop"]:
            s, tool = lr_score(L1["coefs"][det], Xm)
            out["lin"][(det, key)] = {"score": s, "tool": tool}
        if key == "1.0|0":
            for det in ["LR-C", "LR-C-drop"]:
                out["checks"][f"{det}_level1_max_abs_diff_vs_cache"] = float(np.max(np.abs(out["lin"][(det, key)]["score"] - L1["scores"][det][pE])))
                out["checks"][f"{det}_level1_tool_part_max_abs_diff_vs_cache"] = float(np.max(np.abs(out["lin"][(det, key)]["tool"] - L1["lrc_parts"][det]["tool"][pE])))
    g = cache("gru", "R1")
    for det in R1_GRU_DETS:
        for seed in SEEDS:
            model, obj = load_model(det, seed)
            voc = obj["vocab"]; scaler = obj["scaler"]; thr = obj["thr"]
            out["checks"][f"{det}_seed{seed}_vocab_equals_linear_vocab"] = sorted(voc) == sorted(vocab)
            for (L, d), m in masks.items():
                key = f"{L}|{d}"
                mset = set(m)
                vm = {t: i for t, i in voc.items() if t not in mset}   # masked names -> <UNK>; other ids unchanged
                sc = score_gru(model, vm, scaler, thr, teE)["score"]
                tc, _, _, _ = decompose(model, _tensors(teE, vm, scaler))
                out["gru"][(det, seed, key)] = {"score": sc, "tool": tc}
            out["checks"][f"{det}_seed{seed}_level1_max_abs_diff_vs_cache"] = float(np.max(np.abs(
                out["gru"][(det, seed, "1.0|0")]["score"].astype(np.float64) - g["scores"][det][seed][pE])))
    pdump(out, path)
    log(f"graded: scored {len(masks)} masks; checks {out['checks']}")
    return True


# ============================================================================ step 3: analysis
def meta_arrays(obj):
    tm = obj["test_meta"]
    return (np.array([m["y"] for m in tm]), np.array([m["in_pool_E"] for m in tm]),
            np.array([m["domain"] for m in tm], dtype=object), tm)


def det_scores(setting, det):
    """(R, n) scores and (R, n) t_r80 predictions for a detector in a setting (linear: R = 1)."""
    if det in GRUS:
        g = cache("gru", setting)
        return np.atleast_2d(g["scores"][det]).astype(np.float64), np.atleast_2d(g["pred"][det]["t_r80"])
    L = cache("linear", setting)
    return np.atleast_2d(L["scores"][det]).astype(np.float64), np.atleast_2d(L["pred"][det]["t_r80"])


class Pool:
    """A test population with a bootstrap weight matrix; computes per-detector (R, B) metric matrices."""

    def __init__(self, y, W):
        self.y = np.asarray(y); self.W = W

    def auc(self, S):
        S = np.atleast_2d(S)
        return np.stack([wauroc(s, self.y, self.W) for s in S]), np.array([point_auc(s, self.y) for s in S])

    def rate(self, P, cls):
        P = np.atleast_2d(P)
        return np.stack([wrate(p, self.y, cls, self.W) for p in P]), np.array([point_rate(p, self.y, cls) for p in P])


def metrics_for(pool, dets, S, P, mode, rng, shared_k=None):
    """Per detector: metric bootstrap vectors (after rep selection) and points. The same repetition choice is used for
    every metric of a detector within a replicate."""
    out = {}
    for det in dets:
        Ma, pa = pool.auc(S[det]); Mr, pr = pool.rate(P[det], 1); Mf, pf = pool.rate(P[det], 0)
        R = Ma.shape[0]
        if R == 1 or mode == "mean":
            sel = lambda M: select_reps(M, "mean")
        elif mode == "mixture":
            k = rng.integers(R, size=pool.W.shape[0]); sel = lambda M, k=k: M[k, np.arange(M.shape[1])]
        elif mode == "twolevel":
            K = rng.integers(R, size=(pool.W.shape[0], R)); sel = lambda M, K=K: M[K, np.arange(M.shape[1])[:, None]].mean(1)
        elif mode == "shared":
            sel = lambda M: M[shared_k, np.arange(M.shape[1])]
        else:
            raise ValueError(mode)
        out[det] = {"AUROC": (float(pa.mean()), sel(Ma)), "recall@t_r80": (float(pr.mean()), sel(Mr)),
                    "FPR@t_r80": (float(pf.mean()), sel(Mf)), "per_rep_AUROC": pa, "_Ma": Ma}
    return out


def fmt_metrics(m):
    return {k: summ(*m[k]) for k in ["AUROC", "recall@t_r80", "FPR@t_r80"]}


# ---------------------------------------------------------------- s1
def s1_lrc_offset():
    res = {"definition": ("offset = w_A^T c, the LR-C logit contribution of an all-zero raw tool block, c = (0 - mu)/sigma "
                          "with the training scaler (lrc_parts.zero_tool_part); share_train_ge_offset = share of training "
                          "trajectories whose tool-block contribution w_A^T z_i is at least the offset (by class too). "
                          "For LR-C the training tool contributions have mean 0 by construction (scaler fitted on the "
                          "training rows); for LR-C-drop the scaler was fitted on [X; X with the tool block zeroed], so "
                          "the zeroed copies (half of the fit rows) carry exactly the offset and shares refer to the "
                          "original training rows."), "settings": {}}
    for s in [f"R3_{d}" for d in DOMAINS] + ["R1", "R4"] + [f"LOSO_{x}" for x in SUITES]:
        L = cache("linear", s); y, pE, dom, tm = meta_arrays(L)
        res["settings"][s] = {"kappa": r4(L["info"]["kappa"])}
        for det in ["LR-C", "LR-C-drop"]:
            lp = L["lrc_parts"][det]; z = float(lp["zero_tool_part"]); tt = np.asarray(lp["train_tool_part"])
            ty = np.asarray(lp["train_y"]); tcat = np.array(lp["train_category"])
            tool_te = np.asarray(lp["tool"])
            e = {"offset": r4(z), "intercept": r4(lp["intercept"]),
                 "share_train_ge_offset": r4(np.mean(tt >= z)),
                 "share_train_neg_ge_offset": r4(np.mean(tt[ty == 0] >= z)),
                 "share_train_pos_ge_offset": r4(np.mean(tt[ty == 1] >= z)),
                 "train_tool_mean": r4(tt.mean()), "train_tool_mean_neg": r4(tt[ty == 0].mean()),
                 "train_tool_mean_pos": r4(tt[ty == 1].mean()),
                 "offset_minus_train_neg_mean": r4(z - tt[ty == 0].mean()),
                 "train_tool_q": {q: r4(np.quantile(tt, q)) for q in [0.5, 0.85, 0.9, 0.95]},
                 "n_train": int(len(tt)), "n_train_neg": int((ty == 0).sum()),
                 "test_tool_mean_poolE_neg": r4(tool_te[pE & (y == 0)].mean()),
                 "test_tool_mean_poolE_pos": r4(tool_te[pE & (y == 1)].mean()),
                 "test_tool_all_equal_offset": bool(np.allclose(tool_te, z, atol=1e-9)),
                 "share_train_by_category_ge_offset": {c: r4(np.mean(tt[tcat == c] >= z)) for c in sorted(set(tcat))}}
            res["settings"][s][det] = e
    r3 = [res["settings"][f"R3_{d}"] for d in DOMAINS]
    res["R3_range"] = {det: {"offset_min": min(x[det]["offset"] for x in r3), "offset_max": max(x[det]["offset"] for x in r3),
                             "share_min": min(x[det]["share_train_ge_offset"] for x in r3),
                             "share_max": max(x[det]["share_train_ge_offset"] for x in r3),
                             "share_neg_min": min(x[det]["share_train_neg_ge_offset"] for x in r3),
                             "share_neg_max": max(x[det]["share_train_neg_ge_offset"] for x in r3),
                             "share_pos_min": min(x[det]["share_train_pos_ge_offset"] for x in r3),
                             "share_pos_max": max(x[det]["share_train_pos_ge_offset"] for x in r3)}
                       for det in ["LR-C", "LR-C-drop"]}
    return res


# ---------------------------------------------------------------- s2
def s2_graded():
    G = pload(os.path.join(WORK, "graded_scores.pkl"))
    L1 = cache("linear", "R1"); g1 = cache("gru", "R1")
    y_all, pE_all, _, _ = meta_arrays(L1)
    y = y_all[pE_all]
    W = pool_W("R1", y)[0]
    pool = Pool(y, W)
    keys_by_level = collections.OrderedDict((L, [k for k in G["masks"] if float(k.split("|")[0]) == L]) for L in LEVELS)
    thr = {det: float(L1["thr"][det]["t_r80"]) for det in ["LR-B", "LR-C", "LR-C-drop"]}
    thr_g = {det: np.asarray(g1["thr"][det]["t_r80"], float) for det in R1_GRU_DETS}
    sB = L1["scores"]["LR-B"][pE_all].astype(np.float64)
    res = {"design": {"pool": "R1 test pool E (831 attacked / 825 negatives)",
                      "vocab_size": len(G["names"]), "levels": LEVELS, "draws_per_level": N_DRAWS,
                      "mask_rule": ("level L masks round((1 - L) * 103) tool names drawn at random (without replacement) "
                                    "from the R1 benign-training vocabulary; masked names have their bag-of-tools column "
                                    "set to 0 for LR-C / LR-C-drop and are mapped to <UNK> for the GRUs; behavioural "
                                    "features and all fits and thresholds are unchanged. Levels 1 and 0 admit a single mask."),
                      "mask_seed": f"default_rng({GRADED_SEED} + 100*level_index + draw)",
                      "estimand": ("mean over draws (and over the 3 training seeds for the GRUs) of AUROC / FPR / recall at "
                                   "the cached theta_r80; 95% interval = stratified trajectory bootstrap (B = 2000, rng 20201) "
                                   "of that draw- and seed-averaged metric, conditional on the drawn masks; draw_min / draw_max "
                                   "give the spread of the per-draw point values"),
                      "LR-B": "unchanged by construction (no tool columns); the same scores at every level"},
           "checks": G["checks"], "levels": collections.OrderedDict()}
    MB_a, pB_a = pool.auc(sB)
    MB_f, pB_f = pool.rate(sB >= thr["LR-B"], 0); MB_r, pB_r = pool.rate(sB >= thr["LR-B"], 1)
    lrb = {"AUROC": (pB_a[0], MB_a[0]), "FPR@t_r80": (pB_f[0], MB_f[0]), "recall@t_r80": (pB_r[0], MB_r[0])}
    for L, keys in keys_by_level.items():
        covs = [G["cov"][k] for k in keys]
        lev = {"n_draws": len(keys), "n_tools_masked": G["n_tools_masked"][keys[0]],
               "coverage_mean": r4(np.mean(covs)), "coverage_per_draw": [r4(c) for c in covs],
               "coverage_min": r4(min(covs)), "coverage_max": r4(max(covs)), "detectors": {}, "contrasts": {}}
        agg = {"LR-B": lrb}
        for det in ["LR-C", "LR-C-drop"]:
            S = np.stack([G["lin"][(det, k)]["score"] for k in keys])
            Ma, pa = pool.auc(S); Mf, pf = pool.rate(S >= thr[det], 0); Mr, pr = pool.rate(S >= thr[det], 1)
            tool_neg = [float(G["lin"][(det, k)]["tool"][y == 0].mean()) for k in keys]
            agg[det] = {"AUROC": (pa.mean(), Ma.mean(0)), "FPR@t_r80": (pf.mean(), Mf.mean(0)),
                        "recall@t_r80": (pr.mean(), Mr.mean(0)), "per_draw": pa, "per_draw_fpr": pf,
                        "tool_neg_mean": float(np.mean(tool_neg))}
        for det in R1_GRU_DETS:
            per_draw = []; per_draw_f = []; Ma_l, Mf_l, Mr_l, pa_l, pf_l, pr_l = [], [], [], [], [], []; tn = []
            for k in keys:
                S = np.stack([G["gru"][(det, sd, k)]["score"].astype(np.float64) for sd in SEEDS])
                P = S >= thr_g[det][:, None]
                Ma, pa = pool.auc(S); Mf, pf = pool.rate(P, 0); Mr, pr = pool.rate(P, 1)
                Ma_l.append(Ma); Mf_l.append(Mf); Mr_l.append(Mr); pa_l.append(pa); pf_l.append(pf); pr_l.append(pr)
                per_draw.append(pa.mean()); per_draw_f.append(pf.mean())
                tn.append(np.mean([G["gru"][(det, sd, k)]["tool"][y == 0].mean() for sd in SEEDS]))
            Ma = np.concatenate(Ma_l); Mf = np.concatenate(Mf_l); Mr = np.concatenate(Mr_l)
            agg[det] = {"AUROC": (np.mean(pa_l), Ma.mean(0)), "FPR@t_r80": (np.mean(pf_l), Mf.mean(0)),
                        "recall@t_r80": (np.mean(pr_l), Mr.mean(0)), "per_draw": np.array(per_draw),
                        "per_draw_fpr": np.array(per_draw_f), "tool_neg_mean": float(np.mean(tn))}
        for det, a in agg.items():
            e = {k: summ(*a[k]) for k in ["AUROC", "FPR@t_r80", "recall@t_r80"]}
            if "per_draw" in a:
                e["AUROC"].update({"draw_min": r4(np.min(a["per_draw"])), "draw_max": r4(np.max(a["per_draw"])),
                                   "per_draw": [r4(v) for v in a["per_draw"]]})
                e["FPR@t_r80"].update({"draw_min": r4(np.min(a["per_draw_fpr"])), "draw_max": r4(np.max(a["per_draw_fpr"]))})
                e["tool_pathway_mean_on_negatives"] = r4(a["tool_neg_mean"])
            lev["detectors"][det] = e
        for a_, b_ in [("LR-C", "LR-B"), ("LR-C-drop", "LR-B"), ("GRU", "LR-B"), ("GRU-drop0.2", "LR-B"),
                       ("GRU", "LR-C"), ("GRU-drop0.2", "LR-C"), ("GRU-drop0.2", "LR-C-drop"), ("LR-C-drop", "LR-C")]:
            c = contrast(agg[a_]["AUROC"][0], agg[b_]["AUROC"][0], agg[a_]["AUROC"][1], agg[b_]["AUROC"][1])
            cf = contrast(agg[a_]["FPR@t_r80"][0], agg[b_]["FPR@t_r80"][0], agg[a_]["FPR@t_r80"][1], agg[b_]["FPR@t_r80"][1])
            lev["contrasts"][f"{a_}-minus-{b_}"] = {"AUROC": strip(c), "FPR@t_r80": strip(cf)}
        lev["_agg"] = agg
        res["levels"][str(L)] = lev
    # level-0 minus level-1 change per detector (paired on the same trajectories)
    res["change_level0_minus_level1"] = {}
    a0 = res["levels"]["0.0"]["_agg"]; a1 = res["levels"]["1.0"]["_agg"]
    for det in ["LR-C", "LR-C-drop", "GRU", "GRU-drop0.2"]:
        res["change_level0_minus_level1"][det] = {
            "AUROC": strip(contrast(a0[det]["AUROC"][0], a1[det]["AUROC"][0], a0[det]["AUROC"][1], a1[det]["AUROC"][1])),
            "FPR@t_r80": strip(contrast(a0[det]["FPR@t_r80"][0], a1[det]["FPR@t_r80"][0], a0[det]["FPR@t_r80"][1], a1[det]["FPR@t_r80"][1]))}
    for L in res["levels"]:
        res["levels"][L].pop("_agg")
    return res


# ---------------------------------------------------------------- s3 / s7 shared machinery
def r1r3_pools(mode, seed_base):
    """Bootstrap metric vectors for DETS on R1 pooled, R1 per domain (domain-matched) and R3 per fold."""
    out = {}
    L1 = cache("linear", "R1"); y1, pE1, dom1, _ = meta_arrays(L1)
    S1 = {}; P1 = {}
    for det in DETS:
        S, P = det_scores("R1", det); S1[det] = S; P1[det] = P
    assert [m["id"] for m in L1["test_meta"]] == [m["id"] for m in cache("gru", "R1")["test_meta"]]
    specs = [("R1", "R1", pE1)] + [(f"R1dm_{d}", "R1", pE1 & (dom1 == d)) for d in DOMAINS]
    for i, (name, setting, idx) in enumerate(specs):
        rng = np.random.default_rng(seed_base + i)            # repetition (seed) selection only
        ys = y1[idx]; pool = Pool(ys, pool_W(name, ys)[0])
        out[name] = metrics_for(pool, DETS, {d: S1[d][:, idx] for d in DETS}, {d: P1[d][:, idx] for d in DETS}, mode, rng)
        out[name]["_n"] = (int(ys.sum()), int((ys == 0).sum()))
    for j, d in enumerate(DOMAINS):
        s = f"R3_{d}"; L = cache("linear", s); y, pE, _, _ = meta_arrays(L)
        assert [m["id"] for m in L["test_meta"]] == [m["id"] for m in cache("gru", s)["test_meta"]]
        rng = np.random.default_rng(seed_base + 50 + j)       # repetition (seed) selection only
        S = {}; P = {}
        for det in DETS:
            Sd, Pd = det_scores(s, det); S[det] = Sd[:, pE]; P[det] = Pd[:, pE]
        ys = y[pE]; pool = Pool(ys, pool_W(s, ys)[0])
        out[s] = metrics_for(pool, DETS, S, P, mode, rng)
        out[s]["_n"] = (int(ys.sum()), int((ys == 0).sum()))
    return out


def reversal_block(M, pairs, metric="AUROC"):
    """Plan 3.4 rule per pair: reversal in fold d iff the domain-matched R1 difference is detected after Holm (family
    of five domains) AND the R3 difference is detected after Holm (family of five folds) with the opposite sign."""
    res = {}
    for a, b in pairs:
        r1d = [contrast(M[f"R1dm_{d}"][a][metric][0], M[f"R1dm_{d}"][b][metric][0],
                        M[f"R1dm_{d}"][a][metric][1], M[f"R1dm_{d}"][b][metric][1]) for d in DOMAINS]
        r3 = [contrast(M[f"R3_{d}"][a][metric][0], M[f"R3_{d}"][b][metric][0],
                       M[f"R3_{d}"][a][metric][1], M[f"R3_{d}"][b][metric][1]) for d in DOMAINS]
        add_holm(r1d); add_holm(r3)
        pooled = contrast(M["R1"][a][metric][0], M["R1"][b][metric][0], M["R1"][a][metric][1], M["R1"][b][metric][1])
        per = {}
        for d, x1, x3 in zip(DOMAINS, r1d, r3):
            rev_h = x1["detected_holm"] and x3["detected_holm"] and np.sign(x1["point"]) != np.sign(x3["point"])
            rev_u = x1["detected"] and x3["detected"] and np.sign(x1["point"]) != np.sign(x3["point"])
            per[d] = {"R1_domain_matched": strip(x1), "R3": strip(x3), "reversal_holm": bool(rev_h),
                      "reversal_unadjusted": bool(rev_u),
                      "R3_detected_below_holm": bool(x3["detected_holm"] and x3["point"] < 0),
                      "R3_detected_above_holm": bool(x3["detected_holm"] and x3["point"] > 0)}
        res[f"{a}-minus-{b}"] = {
            "R1_pooled": strip(pooled), "per_fold": per,
            "folds_reversal_holm": [d for d in DOMAINS if per[d]["reversal_holm"]],
            "folds_reversal_unadjusted": [d for d in DOMAINS if per[d]["reversal_unadjusted"]],
            "folds_R3_detected_below_holm": [d for d in DOMAINS if per[d]["R3_detected_below_holm"]],
            "folds_R3_detected_above_holm": [d for d in DOMAINS if per[d]["R3_detected_above_holm"]],
            "folds_R1dm_detected_above_holm": [d for d in DOMAINS if r1d[DOMAINS.index(d)]["detected_holm"] and r1d[DOMAINS.index(d)]["point"] > 0],
            "folds_R1dm_detected_below_holm": [d for d in DOMAINS if r1d[DOMAINS.index(d)]["detected_holm"] and r1d[DOMAINS.index(d)]["point"] < 0]}
    return res


_S3_M = {}


def s3_dropout():
    M = r1r3_pools("mixture", 30300)
    _S3_M["M"] = M
    res = {"design": {"bootstrap": ("stratified trajectory bootstrap, B = 2000, one pool per population (R1 pool E; R1 pool E "
                                    "restricted to one domain = domain-matched comparator, plan 3.2; R3 fold pool E); all "
                                    "detectors share the resampled rows (paired). GRU arms: one training seed drawn per "
                                    "replicate (pre-registered evalkit convention), point = mean over the 3 seeds; see s7 for "
                                    "the two-level and seed-averaged alternatives"),
                      "p_values": "two-sided bootstrap p (add-one); Holm within the family of five domains (R1dm) or five folds (R3)",
                      "rule_3_4": ("reversal in fold d iff the domain-matched R1 difference is detected after Holm AND the R3 "
                                   "difference is detected after Holm with the opposite sign"),
                      "thresholds": "cached theta_r80 (training positives, 20th percentile); never from test data"},
           "populations": {}, "reversal_rule_3_4": {}, "threshold_change_R3_minus_R1dm": {}, "auroc_change_R3_minus_R1dm": {}}
    for name, m in M.items():
        res["populations"][name] = {"n_pos": m["_n"][0], "n_neg": m["_n"][1],
                                    "detectors": {det: fmt_metrics(m[det]) for det in DETS}}
        for det in GRUS:
            res["populations"][name]["detectors"][det]["AUROC_per_seed"] = [r4(v) for v in m[det]["per_rep_AUROC"]]
    res["reversal_rule_3_4"] = reversal_block(M, PAIRS)
    # R3 minus domain-matched R1 (independent bootstraps): AUROC, recall, FPR at theta_r80; Holm over five folds
    for det in DETS:
        for metric, tgt in [("AUROC", "auroc_change_R3_minus_R1dm"), ("recall@t_r80", "threshold_change_R3_minus_R1dm"),
                            ("FPR@t_r80", "threshold_change_R3_minus_R1dm")]:
            items = [contrast(M[f"R3_{d}"][det][metric][0], M[f"R1dm_{d}"][det][metric][0],
                              M[f"R3_{d}"][det][metric][1], M[f"R1dm_{d}"][det][metric][1]) for d in DOMAINS]
            add_holm(items)
            res[tgt].setdefault(det, {})[metric] = {d: strip(it) for d, it in zip(DOMAINS, items)}
    # decisions
    dec = {}
    for rem, pair_list in [("LR-C-drop", [("LR-C-drop", "LR-B")]),
                           ("GRU-drop0.2", [("GRU-drop0.2", "LR-C"), ("GRU-drop0.2", "LR-B")]),
                           ("GRU-drop0.1", [("GRU-drop0.1", "LR-C"), ("GRU-drop0.1", "LR-B")])]:
        dec[rem] = {f"{a}-minus-{b}": {"folds_reversal_holm": res["reversal_rule_3_4"][f"{a}-minus-{b}"]["folds_reversal_holm"],
                                       "folds_R3_detected_below_holm": res["reversal_rule_3_4"][f"{a}-minus-{b}"]["folds_R3_detected_below_holm"],
                                       "folds_R3_detected_above_holm": res["reversal_rule_3_4"][f"{a}-minus-{b}"]["folds_R3_detected_above_holm"]}
                    for a, b in pair_list}
    dec["original_LR-C-minus-LR-B"] = res["reversal_rule_3_4"]["LR-C-minus-LR-B"]["folds_reversal_holm"]
    dec["original_GRU-minus-LR-C"] = res["reversal_rule_3_4"]["GRU-minus-LR-C"]["folds_reversal_holm"]
    res["decision"] = dec
    cnt = {}
    for det in DETS:
        cnt[det] = {}
        for met in ["recall@t_r80", "FPR@t_r80"]:
            it = res["threshold_change_R3_minus_R1dm"][det][met]
            cnt[det][met] = {"n_up_holm": int(sum(v["detected_holm"] and v["point"] > 0 for v in it.values())),
                             "n_down_holm": int(sum(v["detected_holm"] and v["point"] < 0 for v in it.values()))}
    res["threshold_change_counts"] = cnt
    mag = {}
    for key, rv in res["reversal_rule_3_4"].items():
        r3p = [rv["per_fold"][d]["R3"]["point"] for d in DOMAINS]
        r1p = [rv["per_fold"][d]["R1_domain_matched"]["point"] for d in DOMAINS]
        mag[key] = {"R3_min": min(r3p), "R3_max": max(r3p), "R1dm_min": min(r1p), "R1dm_max": max(r1p),
                    "R3_by_fold": dict(zip(DOMAINS, r3p))}
    from scipy.stats import spearmanr
    rho = {}
    for d in DOMAINS:
        L = cache("linear", f"R3_{d}"); pE = L["pool_E"]
        rho[d] = {"LR-C-drop_vs_LR-B": r4(spearmanr(L["scores"]["LR-C-drop"][pE], L["scores"]["LR-B"][pE])[0]),
                  "LR-C_vs_LR-B": r4(spearmanr(L["scores"]["LR-C"][pE], L["scores"]["LR-B"][pE])[0])}
    mag["spearman_R3_poolE"] = rho
    mag["spearman_R3_LR-C-drop_vs_LR-B_min"] = min(v["LR-C-drop_vs_LR-B"] for v in rho.values())
    mag["spearman_R3_LR-C_vs_LR-B_min"] = min(v["LR-C_vs_LR-B"] for v in rho.values())
    mag["spearman_R3_LR-C_vs_LR-B_max"] = max(v["LR-C_vs_LR-B"] for v in rho.values())
    ratio = {d: r4(abs(mag["LR-C-drop-minus-LR-B"]["R3_by_fold"][d]) / abs(mag["LR-C-minus-LR-B"]["R3_by_fold"][d]), 3) for d in DOMAINS}
    mag["ratio_abs_R3_LR-C-drop-minus-LR-B_to_LR-C-minus-LR-B"] = {"by_fold": ratio, "min": min(ratio.values()), "max": max(ratio.values())}
    res["magnitudes"] = mag
    return res


# ---------------------------------------------------------------- s4
def init_embedding(seed, V):
    torch.manual_seed(seed); np.random.seed(seed)       # exactly as train_gru before GRUDet(...)
    m = GRUDet(V)
    W = m.emb.weight.detach().clone()
    return W


def adam_decay_only(w0, T):
    """Row with zero data gradient: Adam (lr, coupled weight decay WD, default betas/eps) for T steps."""
    p = torch.nn.Parameter(w0.clone())
    opt = torch.optim.Adam([p], lr=LR, weight_decay=WD)
    for _ in range(T):
        p.grad = torch.zeros_like(p); opt.step()
    return p.detach()


def s4_gru_offset():
    res = {"definition": ("tool pathway = GRU head weights over the 64 hidden units dotted with the final hidden state "
                          "(cache field tool_contrib; train_tool_contrib on the 90% fit partition). Training negatives = all "
                          "label-0 rows of the fit partition (benign, hard_negative, failed_attack); held-out negatives = "
                          "pool-E negatives of the test fold (benign, hard_negative); held-out positives = attacked test "
                          "rows. offset = held-out-negative mean minus training-negative mean."),
           "settings": {}, "summary_R3": {}, "unk": {}, "wut_unk": {}}
    rng = np.random.default_rng(40400)
    for s in ["R1"] + [f"R3_{d}" for d in DOMAINS] + ["R4"]:
        g = cache("gru", s); y, pE, _, _ = meta_arrays(g)
        res["settings"][s] = {}
        for det in GRUS:
            rows = []
            for k, sd in enumerate(SEEDS):
                tc = g["tool_contrib"][det][k]; bc = g["beh_contrib"][det][k]
                ttc = np.asarray(g["train_tool_contrib"][det][k]); tbc = np.asarray(g["train_beh_contrib"][det][k])
                ty = np.asarray(g["train_y"][det][k]); tcat = np.asarray(g["train_category"][det][k])
                hn = tc[pE & (y == 0)]; hp = tc[pE & (y == 1)]; tn = ttc[ty == 0]; tp = ttc[ty == 1]
                # bootstrap of the offset (held-out and training negatives resampled independently)
                bi_h = rng.integers(0, len(hn), size=(B, len(hn))); bi_t = rng.integers(0, len(tn), size=(B, len(tn)))
                off_b = hn[bi_h].mean(1) - tn[bi_t].mean(1)
                e = {"seed": sd, "train_neg_mean": float(tn.mean()), "train_pos_mean": float(tp.mean()),
                     "train_negpoolEcats_mean": float(ttc[(ty == 0) & (tcat != "failed_attack")].mean()),
                     "heldout_neg_mean": float(hn.mean()), "heldout_pos_mean": float(hp.mean()),
                     "heldout_failed_mean": float(tc[~pE].mean()) if (~pE).any() else None,
                     "heldout_all_mean": float(tc[pE].mean()),
                     "offset_neg": float(hn.mean() - tn.mean()), "_off_b": off_b,
                     "share_train_neg_ge_heldout_neg_mean": float(np.mean(tn >= hn.mean())),
                     "share_train_all_ge_heldout_neg_mean": float(np.mean(ttc >= hn.mean())),
                     "beh_train_neg_mean": float(np.asarray(tbc)[ty == 0].mean()), "beh_heldout_neg_mean": float(bc[pE & (y == 0)].mean()),
                     "beh_heldout_pos_mean": float(bc[pE & (y == 1)].mean()), "bias": float(g["bias"][det][k]),
                     "heldout_FPR_t_r80": float(np.asarray(g["pred"][det]["t_r80"][k])[pE & (y == 0)].mean()),
                     "heldout_recall_t_r80": float(np.asarray(g["pred"][det]["t_r80"][k])[pE & (y == 1)].mean())}
                rows.append(e)
            off_b = np.mean([r.pop("_off_b") for r in rows], axis=0)
            mean_rows = {kk: float(np.mean([r[kk] for r in rows])) for kk in rows[0] if kk not in ("seed", "heldout_failed_mean")}
            neg_cells = [r["seed"] for r in rows if r["heldout_neg_mean"] <= r["train_neg_mean"]]
            res["settings"][s][det] = {"per_seed": [{kk: (r4(v) if isinstance(v, float) else v) for kk, v in r.items()} for r in rows],
                                       "mean_over_seeds": {kk: r4(v) for kk, v in mean_rows.items()},
                                       "offset_neg_seedmean_ci": pctl(off_b),
                                       "seeds_with_heldout_neg_not_above_train_neg": neg_cells}
    for det in GRUS:
        cells = [(d, r) for d in DOMAINS for r in res["settings"][f"R3_{d}"][det]["per_seed"]]
        agg = {}
        for kk in ["train_neg_mean", "heldout_neg_mean", "heldout_pos_mean", "offset_neg", "train_pos_mean",
                   "share_train_neg_ge_heldout_neg_mean"]:
            v = np.array([r[kk] for _, r in cells])
            agg[kk] = {"mean": r4(v.mean()), "min": r4(v.min()), "max": r4(v.max())}
        agg["n_cells"] = len(cells)
        agg["n_cells_heldout_neg_above_train_neg"] = int(sum(r["heldout_neg_mean"] > r["train_neg_mean"] for _, r in cells))
        agg["cells_not_above"] = [{"fold": d, "seed": r["seed"], "offset_neg": r["offset_neg"], "heldout_FPR_t_r80": r["heldout_FPR_t_r80"],
                                   "heldout_recall_t_r80": r["heldout_recall_t_r80"]} for d, r in cells if r["heldout_neg_mean"] <= r["train_neg_mean"]]
        res["summary_R3"][det] = agg
    # <UNK> norms: cache values, re-seeded initialisation, and Adam-with-weight-decay-only simulation (p = 0 only)
    for s in ["R1"] + [f"R3_{d}" for d in DOMAINS] + ["R4"]:
        g = cache("gru", s); res["unk"][s] = {}
        for det in GRUS:
            per = []
            for k, sd in enumerate(SEEDS):
                td = g["train_diag"][det][k]; V = int(td["vocab_size_model"])
                W0 = init_embedding(sd, V)
                init_unk = float(W0[UNK].norm()); init_tool = W0[2:].norm(dim=1).numpy()
                tr_norms = np.asarray(g["tool_emb_norms"][det][k], float)
                e = {"seed": sd, "unk_init_norm_cache": r4(g["unk_init_norm"][det][k]),
                     "unk_init_norm_reproduced": r4(init_unk),
                     "unk_init_reproduced_equal": bool(abs(init_unk - float(g["unk_init_norm"][det][k])) < 1e-5),
                     "unk_trained_norm": r4(g["unk_norm"][det][k]),
                     "tool_emb_trained_norm_median": r4(np.median(tr_norms)), "tool_emb_trained_norm_mean": r4(tr_norms.mean()),
                     "tool_emb_trained_norm_min": r4(tr_norms.min()), "tool_emb_trained_norm_max": r4(tr_norms.max()),
                     "tool_emb_init_norm_median": r4(np.median(init_tool)),
                     "unk_trained_rank_among_tool_norms": int(np.sum(tr_norms < float(g["unk_norm"][det][k]))),
                     "n_tool_rows": int(len(tr_norms)), "unk_train_steps": int(g["unk_train_steps"][det][k]),
                     "best_epoch": int(td["best_epoch"]), "n_fit": int(td["n_fit"]),
                     "optimizer_steps_to_best": int(td["best_epoch"]) * math.ceil(int(td["n_fit"]) / BATCH)}
                e["multiplicative_decay_prediction"] = r4(init_unk * (1 - LR * WD) ** e["optimizer_steps_to_best"])
                if det == "GRU" and e["unk_train_steps"] == 0:
                    wT = adam_decay_only(W0[UNK], e["optimizer_steps_to_best"])
                    e["adam_decay_only_predicted_norm"] = r4(float(wT.norm()))
                    e["note_unk_norm_cache_precision"] = "cache unk_norm is float32; errors are |simulated - cached|"
                    e["adam_prediction_abs_err"] = float(abs(float(wT.norm()) - float(g["unk_norm"][det][k])))
                    if s == "R1" and os.path.exists(model_path("GRU", sd)):
                        mdl = pload(model_path("GRU", sd))
                        wtrained = mdl["state_dict"]["emb.weight"][UNK]
                        e["adam_prediction_max_abs_coord_err_vs_trained_vector"] = float((wT - wtrained).abs().max())
                per.append(e)
            res["unk"][s][det] = per
    for det in GRUS:
        cells = [e for s in [f"R3_{d}" for d in DOMAINS] for e in res["unk"][s][det]]
        res["unk"].setdefault("summary_R3", {})[det] = {
            "unk_trained_norm_min": min(e["unk_trained_norm"] for e in cells), "unk_trained_norm_max": max(e["unk_trained_norm"] for e in cells),
            "unk_trained_norm_mean": r4(np.mean([e["unk_trained_norm"] for e in cells])),
            "unk_init_norm_min": min(e["unk_init_norm_reproduced"] for e in cells), "unk_init_norm_max": max(e["unk_init_norm_reproduced"] for e in cells),
            "tool_median_trained_min": min(e["tool_emb_trained_norm_median"] for e in cells),
            "tool_median_trained_max": max(e["tool_emb_trained_norm_median"] for e in cells),
            "tool_median_init_min": min(e["tool_emb_init_norm_median"] for e in cells),
            "tool_median_init_max": max(e["tool_emb_init_norm_median"] for e in cells),
            "all_init_reproduced": all(e["unk_init_reproduced_equal"] for e in cells),
            "optimizer_steps_to_best_min": min(e["optimizer_steps_to_best"] for e in cells),
            "optimizer_steps_to_best_max": max(e["optimizer_steps_to_best"] for e in cells),
            "multiplicative_decay_prediction_min": min(e["multiplicative_decay_prediction"] for e in cells),
            "unk_train_steps_max": max(e["unk_train_steps"] for e in cells)}
        if det == "GRU":
            res["unk"]["summary_R3"][det]["adam_prediction_max_abs_err"] = max(e["adam_prediction_abs_err"] for e in cells)
    gw = cache("gru", "WUT")
    res["wut_unk"] = {"unk_norm_by_rep_fold": np.round(np.asarray(gw["unk_norm"]["GRU"], float), 4).tolist(),
                      "unk_train_steps_by_rep_fold": np.asarray(gw["unk_train_steps"]["GRU"]).tolist(),
                      "unk_train_traces_by_rep_fold": np.asarray(gw["unk_train_traces"]["GRU"]).tolist(),
                      "unk_train_traces_pos_by_rep_fold": np.asarray(gw["unk_train_traces_pos"]["GRU"]).tolist(),
                      "unk_norm_min": r4(np.min(gw["unk_norm"]["GRU"])), "unk_norm_max": r4(np.max(gw["unk_norm"]["GRU"])),
                      "unk_train_steps_max": int(np.max(gw["unk_train_steps"]["GRU"])),
                      "unk_train_traces_max": int(np.max(gw["unk_train_traces"]["GRU"])),
                      "note": "within-AgentDojo UNK tokens occur in training (fold vocabularies from benign P0 only); the "
                              "decay-only prediction does not apply there. The m6 request for a WUT GRU without UNK-bearing "
                              "traces is not addressed here (no such fit in the cache)."}
    return res


# ---------------------------------------------------------------- s5
def s5_loso():
    res = {"design": {"pool": "pool E (P0 benign + attacked) of the held-out suite; unsuccessful attacks excluded",
                      "bootstrap_primary": "cluster bootstrap over suite|user_task (plan 3.5), B = 2000",
                      "bootstrap_sensitivity": "stratified trajectory bootstrap, B = 2000",
                      "in_distribution": ("WUT out-of-fold scores (5-fold CV grouped by suite|user_task, 5 repetitions); one "
                                          "repetition drawn per replicate and shared by all detectors (paired by fold "
                                          "assignment); point = mean over repetitions; pooled over all suites and restricted "
                                          "to each suite (suite-matched)"),
                      "rule": "plan 3.4 analogue: Holm within the family of four suites for the suite-matched WUT and for LOSO"},
           "suites": {}, "WUT_pooled": {}, "reversal": {}}
    p0 = dojo_p0()
    pairs = [("LR-C", "LR-B"), ("LR-C-drop", "LR-B"), ("LR-C-drop", "LR-C")]
    loso_c = {pr: [] for pr in pairs}; wut_c = {pr: [] for pr in pairs}
    loso_s = {pr: [] for pr in pairs}
    Lw = cache("linear", "WUT"); yw, pEw, domw, tmw = meta_arrays(Lw)
    tasks_w = np.array([m["task"] for m in tmw], dtype=object)
    for i, st in enumerate(SUITES):
        s = f"LOSO_{st}"; L = cache("linear", s); y, pE, dom, tm = meta_arrays(L)
        tasks = np.array([m["task"] for m in tm], dtype=object)
        tr = [r for r in p0 if r["agent"] != st]; te = [r for r in p0 if r["agent"] == st]
        assert [r["id"] for r in te] == [m["id"] for m, e in zip(tm, pE) if e]
        cov = ablation.coverage(tr, te)
        ys = y[pE]
        Wc, ncl = pool_W(f"LOSO_{st}_cluster", clusters=tasks[pE])
        Ws = pool_W(f"LOSO_{st}_strat", ys)[0]
        S = {d: np.atleast_2d(L["scores"][d][pE]).astype(np.float64) for d in LIN}
        P = {d: np.atleast_2d(L["pred"][d]["t_r80"][pE]) for d in LIN}
        mc = metrics_for(Pool(ys, Wc), LIN, S, P, "mean", None)
        ms = metrics_for(Pool(ys, Ws), LIN, S, P, "mean", None)
        # suite-matched in-distribution (WUT)
        idx = pEw & (domw == st); yws = yw[idx]
        Wwc, nclw = pool_W(f"WUTsuite_{st}_cluster", clusters=tasks_w[idx])
        kk = rep_choice(f"WUTsuite_{st}")
        Sw = {d: np.asarray(Lw["scores"][d][:, idx], np.float64) for d in LIN}
        Pw = {d: np.asarray(Lw["pred"][d]["t_r80"][:, idx]) for d in LIN}
        mw = metrics_for(Pool(yws, Wwc), LIN, Sw, Pw, "shared", None, shared_k=kk)
        e = {"n_pos": int(ys.sum()), "n_neg": int((ys == 0).sum()), "n_clusters": ncl,
             "n_failed_rows_in_test": int((~pE).sum()), "coverage": cov, "kappa_cache": r4(L["info"]["kappa"]),
             "vocab_size": int(L["info"]["vocab_size"]), "n_train": int(L["info"]["n_train"]),
             "LOSO_cluster": {d: fmt_metrics(mc[d]) for d in LIN},
             "LOSO_stratified": {d: {"AUROC": summ(*ms[d]["AUROC"])} for d in LIN},
             "WUT_suite_matched_cluster": {d: fmt_metrics(mw[d]) for d in LIN}, "n_clusters_WUT_suite": nclw,
             "contrasts": {}}
        for a, b in pairs:
            cc = contrast(mc[a]["AUROC"][0], mc[b]["AUROC"][0], mc[a]["AUROC"][1], mc[b]["AUROC"][1])
            cs = contrast(ms[a]["AUROC"][0], ms[b]["AUROC"][0], ms[a]["AUROC"][1], ms[b]["AUROC"][1])
            cw = contrast(mw[a]["AUROC"][0], mw[b]["AUROC"][0], mw[a]["AUROC"][1], mw[b]["AUROC"][1])
            loso_c[(a, b)].append(cc); loso_s[(a, b)].append(cs); wut_c[(a, b)].append(cw)
            e["contrasts"][f"{a}-minus-{b}"] = {"LOSO_cluster": cc, "LOSO_stratified": cs, "WUT_suite_matched_cluster": cw}
        res["suites"][st] = e
    for pr in pairs:
        add_holm(loso_c[pr]); add_holm(wut_c[pr]); add_holm(loso_s[pr])
        key = f"{pr[0]}-minus-{pr[1]}"
        per = {}
        for st, cl, cw in zip(SUITES, loso_c[pr], wut_c[pr]):
            rev = cw["detected_holm"] and cl["detected_holm"] and np.sign(cw["point"]) != np.sign(cl["point"])
            per[st] = {"reversal_holm": bool(rev), "LOSO_detected_holm": cl["detected_holm"], "LOSO_sign": r4(np.sign(cl["point"]), 0),
                       "WUT_suite_detected_holm": cw["detected_holm"], "WUT_suite_sign": r4(np.sign(cw["point"]), 0)}
        res["reversal"][key] = {"per_suite": per, "suites_reversal_holm": [s for s in SUITES if per[s]["reversal_holm"]],
                                "suites_LOSO_detected_below_holm": [s for s, c in zip(SUITES, loso_c[pr]) if c["detected_holm"] and c["point"] < 0],
                                "suites_LOSO_detected_above_holm": [s for s, c in zip(SUITES, loso_c[pr]) if c["detected_holm"] and c["point"] > 0]}
    for st in SUITES:
        for key in res["suites"][st]["contrasts"]:
            res["suites"][st]["contrasts"][key] = strip(res["suites"][st]["contrasts"][key])
    # pooled WUT (all suites)
    yp = yw[pEw]; Wc, ncl = pool_W("P0_cluster", clusters=tasks_w[pEw])
    kk = rep_choice("P0")
    Sw = {d: np.asarray(Lw["scores"][d][:, pEw], np.float64) for d in LIN}
    Pw = {d: np.asarray(Lw["pred"][d]["t_r80"][:, pEw]) for d in LIN}
    mw = metrics_for(Pool(yp, Wc), LIN, Sw, Pw, "shared", None, shared_k=kk)
    Ws = pool_W("P0_strat", yp)[0]
    mws = metrics_for(Pool(yp, Ws), LIN, Sw, Pw, "shared", None, shared_k=kk)
    res["WUT_pooled"] = {"n_pos": int(yp.sum()), "n_neg": int((yp == 0).sum()), "n_clusters": ncl,
                         "cluster": {d: fmt_metrics(mw[d]) for d in LIN},
                         "stratified": {d: {"AUROC": summ(*mws[d]["AUROC"])} for d in LIN}, "contrasts": {}}
    for a, b in pairs:
        res["WUT_pooled"]["contrasts"][f"{a}-minus-{b}"] = {
            "cluster": strip(contrast(mw[a]["AUROC"][0], mw[b]["AUROC"][0], mw[a]["AUROC"][1], mw[b]["AUROC"][1])),
            "stratified": strip(contrast(mws[a]["AUROC"][0], mws[b]["AUROC"][0], mws[a]["AUROC"][1], mws[b]["AUROC"][1]))}
    return res


# ---------------------------------------------------------------- s6
def s6_rule4():
    res = {"design": {"rule": ("PRE_ABLATION_DESIGN.md Rule 4: if C > B in R1 (paired, detected) but C degrades more than B "
                               "(degradation difference detected favouring B in >= 3/5 R3 folds or in R4), report 'benchmark-"
                               "specific tool information improves in-distribution scores while reducing robustness'; the R4 "
                               "test is primary"),
                      "estimands": {"transfer_C_minus_B": "AUROC(LR-C, R4) - AUROC(LR-B, R4), paired",
                                    "within_C_minus_B": "AUROC(LR-C, WUT) - AUROC(LR-B, WUT), paired",
                                    "DD_within_ref": "(within_C - transfer_C) - (within_B - transfer_B), all four on the same resample",
                                    "DD_R1_ref": "(R1_C - R4_C) - (R1_B - R4_B), R1 and R4 resampled independently"},
                      "bootstrap": ("primary: cluster bootstrap over suite|user_task on AgentDojo (plan 3.5); pre-registered "
                                    "variant: stratified trajectory bootstrap (evalkit design). B = 2000. WUT: one repetition "
                                    "per replicate shared by LR-B and LR-C; point = mean over the 5 repetitions"),
                      "populations": {"P0": "AgentDojo P0 (2,294 attacked, 392 benign)",
                                      "C1": "P0 without benign runs whose user task is an injection task (r4_control.py C1)"}},
           "R1_C_minus_B": None, "R4": {}, "R3_supplementary": {}}
    L1 = cache("linear", "R1"); y1, pE1, dom1, _ = meta_arrays(L1)
    ys1 = y1[pE1]; W1 = pool_W("R1", ys1)[0]
    S1 = {d: np.atleast_2d(L1["scores"][d][pE1]).astype(np.float64) for d in ["LR-B", "LR-C"]}
    P1 = {d: np.atleast_2d(L1["pred"][d]["t_r80"][pE1]) for d in ["LR-B", "LR-C"]}
    m1 = metrics_for(Pool(ys1, W1), ["LR-B", "LR-C"], S1, P1, "mean", None)
    res["R1_C_minus_B"] = strip(contrast(m1["LR-C"]["AUROC"][0], m1["LR-B"]["AUROC"][0], m1["LR-C"]["AUROC"][1], m1["LR-B"]["AUROC"][1]))
    res["R1_AUROC"] = {d: summ(*m1[d]["AUROC"]) for d in ["LR-B", "LR-C"]}
    L4 = cache("linear", "R4"); Lw = cache("linear", "WUT")
    y4, pE4, _, tm4 = meta_arrays(L4)
    assert [m["id"] for m in tm4] == [m["id"] for m in Lw["test_meta"]]
    tasks = np.array([m["task"] for m in tm4], dtype=object)
    c1 = pE4 & ~np.array([(m["category"] == "benign") and str(m["user_task"]).startswith("injection_task") for m in tm4])
    for pi, (pop, idx) in enumerate([("P0", pE4), ("C1", c1)]):
        ys = y4[idx]; e = {"n_pos": int(ys.sum()), "n_neg": int((ys == 0).sum())}
        for bi, (bname, W) in enumerate([("cluster", pool_W(f"{pop}_cluster", clusters=tasks[idx])[0]),
                                         ("stratified", pool_W(f"{pop}_strat", ys)[0])]):
            pool = Pool(ys, W)
            kk = rep_choice(pop)
            St = {d: np.atleast_2d(L4["scores"][d][idx]).astype(np.float64) for d in ["LR-B", "LR-C"]}
            Pt = {d: np.atleast_2d(L4["pred"][d]["t_r80"][idx]) for d in ["LR-B", "LR-C"]}
            Sw = {d: np.asarray(Lw["scores"][d][:, idx], np.float64) for d in ["LR-B", "LR-C"]}
            Pw = {d: np.asarray(Lw["pred"][d]["t_r80"][:, idx]) for d in ["LR-B", "LR-C"]}
            mt = metrics_for(pool, ["LR-B", "LR-C"], St, Pt, "mean", None)
            mw = metrics_for(pool, ["LR-B", "LR-C"], Sw, Pw, "shared", None, shared_k=kk)
            tC, tB = mt["LR-C"]["AUROC"], mt["LR-B"]["AUROC"]; wC, wB = mw["LR-C"]["AUROC"], mw["LR-B"]["AUROC"]
            ddp = (wC[0] - tC[0]) - (wB[0] - tB[0]); ddb = (wC[1] - tC[1]) - (wB[1] - tB[1])
            dd = {"point": r4(ddp), "ci": pctl(ddb), "p_boot": r4(p_boot(ddb), 5)}
            dd["detected"] = bool(dd["ci"][0] > 0 or dd["ci"][1] < 0)
            # R1-referenced degradation difference (independent R1 resample)
            ddr_p = (m1["LR-C"]["AUROC"][0] - tC[0]) - (m1["LR-B"]["AUROC"][0] - tB[0])
            ddr_b = (m1["LR-C"]["AUROC"][1] - tC[1]) - (m1["LR-B"]["AUROC"][1] - tB[1])
            ddr = {"point": r4(ddr_p), "ci": pctl(ddr_b), "p_boot": r4(p_boot(ddr_b), 5)}
            ddr["detected"] = bool(ddr["ci"][0] > 0 or ddr["ci"][1] < 0)
            e[bname] = {"AUROC": {"transfer_LR-C": summ(*tC), "transfer_LR-B": summ(*tB), "within_LR-C": summ(*wC), "within_LR-B": summ(*wB)},
                        "transfer_C_minus_B": strip(contrast(tC[0], tB[0], tC[1], tB[1])),
                        "within_C_minus_B": strip(contrast(wC[0], wB[0], wC[1], wB[1])),
                        "degradation_within_minus_transfer": {"LR-C": strip(contrast(wC[0], tC[0], wC[1], tC[1])),
                                                              "LR-B": strip(contrast(wB[0], tB[0], wB[1], tB[1]))},
                        "DD_within_ref": dd, "DD_R1_ref": ddr}
            if bname == "cluster":
                e["n_clusters"] = int(len(set(tasks[idx])))
        res["R4"][pop] = e
    # supplementary R3 clause: (R1_C - R3_C) - (R1_B - R3_B), pooled R1 (as pre-registered) and domain-matched R1
    items_p, items_d = [], []
    for j, d in enumerate(DOMAINS):
        L = cache("linear", f"R3_{d}"); y, pE, _, _ = meta_arrays(L); ys = y[pE]
        W3 = pool_W(f"R3_{d}", ys)[0]
        m3 = metrics_for(Pool(ys, W3), ["LR-B", "LR-C"], {k: np.atleast_2d(L["scores"][k][pE]).astype(np.float64) for k in ["LR-B", "LR-C"]},
                         {k: np.atleast_2d(L["pred"][k]["t_r80"][pE]) for k in ["LR-B", "LR-C"]}, "mean", None)
        idx = pE1 & (dom1 == d); ysd = y1[idx]
        Wd = pool_W(f"R1dm_{d}", ysd)[0]
        md = metrics_for(Pool(ysd, Wd), ["LR-B", "LR-C"], {k: np.atleast_2d(L1["scores"][k][idx]).astype(np.float64) for k in ["LR-B", "LR-C"]},
                         {k: np.atleast_2d(L1["pred"][k]["t_r80"][idx]) for k in ["LR-B", "LR-C"]}, "mean", None)
        for ref, mm, items in [("pooled_R1", m1, items_p), ("domain_matched_R1", md, items_d)]:
            pt = (mm["LR-C"]["AUROC"][0] - m3["LR-C"]["AUROC"][0]) - (mm["LR-B"]["AUROC"][0] - m3["LR-B"]["AUROC"][0])
            bt = (mm["LR-C"]["AUROC"][1] - m3["LR-C"]["AUROC"][1]) - (mm["LR-B"]["AUROC"][1] - m3["LR-B"]["AUROC"][1])
            items.append({"point": r4(pt), "ci": pctl(bt), "p_boot": r4(p_boot(bt), 5),
                          "detected": bool(pctl(bt)[0] > 0 or pctl(bt)[1] < 0)})
    add_holm(items_p); add_holm(items_d)
    res["R3_supplementary"] = {"definition": "DD = (R1_C - R3_C) - (R1_B - R3_B); positive = C degrades more than B",
                               "pooled_R1": dict(zip(DOMAINS, items_p)), "domain_matched_R1": dict(zip(DOMAINS, items_d)),
                               "n_folds_detected_favouring_B_holm_pooled": int(sum(i["detected_holm"] and i["point"] > 0 for i in items_p)),
                               "n_folds_detected_favouring_B_holm_domain_matched": int(sum(i["detected_holm"] and i["point"] > 0 for i in items_d))}
    return res


# ---------------------------------------------------------------- s7
def s7_q1(Mm=None):
    res = {"design": {"per_seed": "AUROC of each training seed on the full fold; GRU - LR-C per seed with a stratified trajectory bootstrap (B = 2000)",
                      "two_level": "each replicate draws 3 seeds with replacement, then resamples trajectories (stratified); statistic = mean over drawn seeds of AUROC_GRU(seed) - AUROC_LR-C",
                      "mixture": "pre-registered evalkit convention: one seed per replicate (the interval is a mixture of single-seed sampling distributions)",
                      "seed_averaged": "statistic = mean over the 3 seeds within each trajectory replicate (conditional on the 3 trained seeds)",
                      "note": "with only 3 seeds the seed level of the two-level bootstrap is coarse (10 distinct seed multisets)"},
           "folds": {}, "reversal_two_level": {}, "reversal_seed_averaged": {}}
    Mt = r1r3_pools("twolevel", 70700)       # trajectory resamples are the shared population resamples (pool_W)
    Ms = r1r3_pools("mean", 70700)
    Mm = Mm if Mm is not None else r1r3_pools("mixture", 30300)   # the s3 result: same resamples and seed choices
    for name in ["R1"] + [f"R1dm_{d}" for d in DOMAINS] + [f"R3_{d}" for d in DOMAINS]:
        e = {}
        for det in GRUS:
            per_seed_auc = [r4(v) for v in Mt[name][det]["per_rep_AUROC"]]
            Ma = Mt[name][det]["_Ma"]; Mc = Mt[name]["LR-C"]["_Ma"][0]; pc = Mt[name]["LR-C"]["per_rep_AUROC"][0]
            Mb = Mt[name]["LR-B"]["_Ma"][0]; pb = Mt[name]["LR-B"]["per_rep_AUROC"][0]
            ps = [strip(contrast(Mt[name][det]["per_rep_AUROC"][k], pc, Ma[k], Mc)) for k in range(Ma.shape[0])]
            psb = [strip(contrast(Mt[name][det]["per_rep_AUROC"][k], pb, Ma[k], Mb)) for k in range(Ma.shape[0])]
            e[det] = {"AUROC_per_seed": per_seed_auc,
                      "AUROC_seed_ci": [pctl(Ma[k]) for k in range(Ma.shape[0])],
                      "minus_LR-C_per_seed": ps, "minus_LR-B_per_seed": psb,
                      "minus_LR-C": {"two_level": strip(contrast(Mt[name][det]["AUROC"][0], Mt[name]["LR-C"]["AUROC"][0], Mt[name][det]["AUROC"][1], Mt[name]["LR-C"]["AUROC"][1])),
                                     "mixture": strip(contrast(Mm[name][det]["AUROC"][0], Mm[name]["LR-C"]["AUROC"][0], Mm[name][det]["AUROC"][1], Mm[name]["LR-C"]["AUROC"][1])),
                                     "seed_averaged": strip(contrast(Ms[name][det]["AUROC"][0], Ms[name]["LR-C"]["AUROC"][0], Ms[name][det]["AUROC"][1], Ms[name]["LR-C"]["AUROC"][1]))},
                      "minus_LR-B": {"two_level": strip(contrast(Mt[name][det]["AUROC"][0], Mt[name]["LR-B"]["AUROC"][0], Mt[name][det]["AUROC"][1], Mt[name]["LR-B"]["AUROC"][1])),
                                     "mixture": strip(contrast(Mm[name][det]["AUROC"][0], Mm[name]["LR-B"]["AUROC"][0], Mm[name][det]["AUROC"][1], Mm[name]["LR-B"]["AUROC"][1])),
                                     "seed_averaged": strip(contrast(Ms[name][det]["AUROC"][0], Ms[name]["LR-B"]["AUROC"][0], Ms[name][det]["AUROC"][1], Ms[name]["LR-B"]["AUROC"][1]))},
                      "AUROC": {"two_level": summ(*Mt[name][det]["AUROC"]), "mixture": summ(*Mm[name][det]["AUROC"]),
                                "seed_averaged": summ(*Ms[name][det]["AUROC"])},
                      "seed_range_minus_LR-C": [min(x["point"] for x in ps), max(x["point"] for x in ps)]}
        e["LR-C_AUROC"] = summ(*Mt[name]["LR-C"]["AUROC"]); e["LR-B_AUROC"] = summ(*Mt[name]["LR-B"]["AUROC"])
        res["folds"][name] = e
    gpairs = [p for p in PAIRS if p[0] in GRUS and p[1] in ("LR-C", "LR-B")]
    res["reversal_two_level"] = reversal_block(Mt, gpairs)
    res["reversal_seed_averaged"] = reversal_block(Ms, gpairs)
    return res


# ---------------------------------------------------------------- analyze
def cmd_analyze():
    t0 = time.time()
    out = {"meta": {"script": os.path.abspath(__file__), "script_sha256": sha256(os.path.abspath(__file__)),
                    "generated": datetime.datetime.now().isoformat(timespec="seconds"), "B": B,
                    "caches": {f"{fam}/{s}.pkl": sha256(os.path.join(CACHE, fam, f"{s}.pkl"))
                               for fam, s in [("linear", "R1"), ("gru", "R1"), ("linear", "R4"), ("linear", "WUT"), ("gru", "R4"), ("gru", "WUT")]
                               + [("linear", f"R3_{d}") for d in DOMAINS] + [("gru", f"R3_{d}") for d in DOMAINS]
                               + [("linear", f"LOSO_{s}") for s in SUITES]},
                    "linear_cache_script_sha256": cache("linear", "R1")["provenance"]["script_sha256"],
                    "rerun_v03_log_has_gru_ladder_exit0": "gru_ladder exit 0" in open(os.path.join(HERE, "rerun_v03_scripts.log"), encoding="utf-8").read(),
                    "conventions": {"detected": "95% percentile bootstrap interval excludes 0 (differences) or 0.5 (AUROC levels)",
                                    "p_boot": "two-sided, add-one: min(1, 2*min(#d<=0 + 1, #d>=0 + 1)/(B + 1))",
                                    "holm": "Holm step-down on p_boot within the stated family; detected_holm = p_holm < 0.05",
                                    "rounding": "4 decimals (p-values 5)"}}}
    # engine self-check: weighted AUROC with unit weights equals evalkit.auroc
    L1 = cache("linear", "R1"); y1, pE1, _, _ = meta_arrays(L1)
    chk = []
    for det in ["LR-B", "LR-C", "LR-C-drop"]:
        s = L1["scores"][det][pE1].astype(np.float64)
        chk.append(abs(float(wauroc(s, y1[pE1], np.ones((1, pE1.sum())))[0]) - ek.auroc(y1[pE1], s)))
    out["meta"]["engine_check_max_abs_diff_vs_evalkit_auroc"] = float(max(chk))
    train_checks = {}
    for det in R1_GRU_DETS:
        for sd in SEEDS:
            if os.path.exists(model_path(det, sd)):
                ob = pload(model_path(det, sd)); train_checks[f"{det}_seed{sd}"] = ob["check"]
    out["checks_r1_gru_refit"] = train_checks
    for name, fn in [("s1_lrc_offset", s1_lrc_offset), ("s2_graded", s2_graded), ("s3_dropout", s3_dropout),
                     ("s4_gru_offset", s4_gru_offset), ("s5_loso", s5_loso), ("s6_rule4", s6_rule4),
                     ("s7_q1", lambda: s7_q1(_S3_M.get("M")))]:
        t1 = time.time(); out[name] = strip(fn()); log(f"analyze {name}: {time.time() - t1:.1f}s")
    out["deviations"] = DEVIATIONS
    out["meta"]["runtime_s"] = round(time.time() - t0, 1)
    with open(RESULTS, "w", encoding="utf-8") as f:
        json.dump(jsonable(out), f, indent=1)
    log(f"wrote {RESULTS}")


DEVIATIONS = [
    "O1. Bootstrap size: B = 2000 in every analysis here (the v0.3 AgentDrift tables used B = 1000); fixed rng seeds are listed per analysis in the code.",
    "O2. Multi-seed GRU intervals in s3 follow the pre-registered evalkit convention (one training seed drawn per replicate; point = seed mean). s7 adds the two-level (seeds then trajectories) and seed-averaged intervals and re-applies the plan 3.4 rule under both.",
    "O3. Holm correction needs p-values; they are two-sided bootstrap p-values (add-one), not analytic. 'Detected' without Holm = 95% percentile interval excludes 0.",
    "O4. Graded coverage (plan Sec. 4 'Mechanism'): level L masks round((1-L)*103) randomly drawn tool NAMES of the R1 benign-training vocabulary (26, 52, 77 names; levels 1 and 0 have one deterministic mask), as in the plan's 'random 25/50/75/100% of tools'. The achieved tool-occurrence coverage varies by draw and is reported, not forced to the target. The interval is for the draw- (and seed-) averaged metric conditional on the drawn masks; the per-draw range is given separately.",
    "O5. The R1 GRU and GRU-drop0.2 models were not saved by the cache; they were refitted with gru_variants.train_gru_drop exactly as build_gru_cache.run_unit calls it (same seeds, validation split and data) and checked against the cached scores (checks_r1_gru_refit). GRU-drop0.1 is not part of the graded-coverage request and was not refitted.",
    "O6. Within-AgentDojo (WUT) references: one CV repetition drawn per replicate and shared by all detectors (paired by fold assignment); evalkit draws the repetition independently per method.",
    "O7. AgentDojo intervals (LOSO, WUT, R4 Rule 4) use the plan 3.5 cluster bootstrap over suite|user_task as primary. For Rule 4 the pre-registered design (PRE_ABLATION_DESIGN.md) used the stratified evalkit bootstrap; it is reported alongside. The LOSO stratified intervals are a sensitivity analysis.",
    "O8. GRU tool-pathway 'training negatives' are all label-0 rows of the 90% fit partition (benign, hard_negative, failed_attack), the negatives the network was trained on; the mean over benign + hard_negative rows only is also given (train_negpoolEcats_mean).",
    "O9. Rule 4's R3 clause (>= 3/5 folds) is reported as a supplementary analysis with pooled R1 (as pre-registered) and domain-matched R1 (plan 3.2) references; its R4 clause is primary.",
    "O10. LOSO has no GRU (none in the cache), and the m6 request for a within-AgentDojo GRU without <UNK>-bearing traces is not run here.",
    "O11. R3 fold pools contain every pooled trajectory of the held-out domain, including the R1 test trajectories of that domain, so R3 and domain-matched R1 estimates share rows. The R3-minus-R1 differences (s3 threshold_change and auroc_change, and the s6 R3 clause; the R1-versus-R4 comparison is unaffected) resample the two populations independently, which ignores the positive covariance and makes those intervals conservative.",
    "O12. Every population is resampled once (pool_W, fixed seed per population) and the same resample is used in every section, so an estimand that appears in two sections (for example R1 LR-C - LR-B, or WUT LR-C - LR-B in s5 and s6) has the identical interval. s7 reuses the s3 mixture results.",
]


# ============================================================================ step 4: summary markdown
def f3(e, d=3):
    """'point [lo, hi]' from a {point, ci} dict."""
    if e is None or e.get("point") is None:
        return "n/a"
    lo, hi = e["ci"]
    return f"{e['point']:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def fd(e, d=3):
    """signed difference 'point [lo, hi]'."""
    if e is None or e.get("point") is None:
        return "n/a"
    lo, hi = e["ci"]
    return f"{e['point']:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}]"


def cmd_summary():
    R = json.load(open(RESULTS, encoding="utf-8"))
    L = []
    w = L.append
    w("# OOV mechanism of the R3 reversals (review M1, Q1, m6 <UNK> norm)\n")
    w(f"Generated from `revision/oov_mechanism_results.json` (script `revision/oov_mechanism.py`, SHA-256 `{R['meta']['script_sha256'][:16]}...`). "
      "Every number below is a key of that JSON; the key path is given in brackets. Intervals are 95% percentile bootstrap intervals "
      f"(B = {R['meta']['B']}). 'Detected' means the interval excludes 0; 'Holm' means detected after Holm correction within the stated family. "
      "Thresholds are the cached training thresholds theta_r80; nothing was tuned on test data.\n")
    ck = R["checks_r1_gru_refit"]
    w("**Reproduction checks.** "
      f"Weighted-bootstrap AUROC engine vs evalkit.auroc: max abs diff {R['meta']['engine_check_max_abs_diff_vs_evalkit_auroc']:.1e} [meta.engine_check_max_abs_diff_vs_evalkit_auroc]. "
      "Refitted R1 GRU models vs cached scores (max abs diff per model): "
      + ", ".join(f"{k} {v['max_abs_diff_scores_vs_cache']:.1e}" for k, v in ck.items()) + " [checks_r1_gru_refit]. "
      "Graded-coverage level 1 vs cache: " + ", ".join(f"{k} {v:.1e}" for k, v in R["s2_graded"]["checks"].items() if "max_abs_diff" in k) + " [s2_graded.checks].\n")

    # ---------------- M1 action 1
    s1 = R["s1_lrc_offset"]
    w("## M1, action 1: LR-C tool-block offset w_A^T c per R3 fold\n")
    w("**Question.** At zero coverage every held-out trajectory has an all-zero raw tool block, which the training scaler maps to a constant "
      "vector c. How large is the resulting logit offset w_A^T c, and how unusual is it relative to the training trajectories?\n")
    w("| Fold | LR-C offset | share of training rows >= offset (all / negatives / positives) | LR-C-drop offset | LR-C-drop share >= offset (all / negatives) |")
    w("|---|---|---|---|---|")
    for d in DOMAINS:
        a = s1["settings"][f"R3_{d}"]["LR-C"]; b = s1["settings"][f"R3_{d}"]["LR-C-drop"]
        w(f"| {d} | {a['offset']:+.3f} | {a['share_train_ge_offset']:.3f} / {a['share_train_neg_ge_offset']:.3f} / {a['share_train_pos_ge_offset']:.3f} | "
          f"{b['offset']:+.3f} | {b['share_train_ge_offset']:.3f} / {b['share_train_neg_ge_offset']:.3f} |")
    w(f"\n[s1_lrc_offset.settings.R3_<fold>.<LR-C|LR-C-drop>.offset / share_train_ge_offset / share_train_neg_ge_offset / share_train_pos_ge_offset]. "
      f"The LR-C training tool contributions average 0 by construction; on training negatives they average "
      + ", ".join(f"{s1['settings'][f'R3_{d}']['LR-C']['train_tool_mean_neg']:+.3f}" for d in DOMAINS)
      + " (medical ... coding), so the offset sits "
      + ", ".join(f"{s1['settings'][f'R3_{d}']['LR-C']['offset_minus_train_neg_mean']:+.3f}" for d in DOMAINS)
      + " logits above the mean training negative [offset_minus_train_neg_mean]. In every R3 fold every test row carries exactly the offset "
      f"[test_tool_all_equal_offset = {all(s1['settings'][f'R3_{d}']['LR-C']['test_tool_all_equal_offset'] for d in DOMAINS)}]. "
      "For comparison, the same quantity for the models trained on the AgentDojo suites (LOSO) is "
      + ", ".join(f"{x} {s1['settings'][f'LOSO_{x}']['LR-C']['offset']:+.3f}" for x in SUITES)
      + " [s1_lrc_offset.settings.LOSO_<suite>.LR-C.offset].\n")
    rr = s1["R3_range"]
    lb = s1["settings"]["LOSO_banking"]["LR-C"]["offset"]; ls = s1["settings"]["LOSO_slack"]["LR-C"]["offset"]
    w(f"**Plain reading.** The review's offsets are reproduced: LR-C adds {rr['LR-C']['offset_min']:+.2f} to {rr['LR-C']['offset_max']:+.2f} logits "
      f"to every held-out trajectory. Only {rr['LR-C']['share_min']:.1%} to {rr['LR-C']['share_max']:.1%} of training trajectories reach that "
      f"tool contribution: {rr['LR-C']['share_neg_min']:.1%} to {rr['LR-C']['share_neg_max']:.1%} of training negatives and "
      f"{rr['LR-C']['share_pos_min']:.1%} to {rr['LR-C']['share_pos_max']:.1%} of training positives [s1_lrc_offset.R3_range]. So an all-zero tool block "
      "looks more attack-like to LR-C than the tool profile of most attacked training trajectories, and at the training threshold this raises "
      "recall and FPR together (action 3). The offset is the same for every held-out row, so it cannot change the ranking within a fold. It explains "
      "the threshold behaviour, not the AUROC loss. At zero coverage LR-C ranks held-out trajectories with its behavioural weights only, and those "
      "weights were fitted jointly with an informative tool block (action 2 shows this directly). Tool-block dropout removes the offset almost "
      f"entirely ({rr['LR-C-drop']['offset_min']:+.3f} to {rr['LR-C-drop']['offset_max']:+.3f}): half of its fit rows have an all-zero tool block. "
      f"For the AgentDojo suites held out at zero coverage, the fitted offset is negative ({lb:+.3f} for the model that holds out banking, "
      f"{ls:+.3f} for slack). There an unseen-tool trajectory is pushed towards benign, and alert rates fall rather than rise (action 5a). "
      "This concerns thresholds only.\n")

    # ---------------- M1 action 2
    s2 = R["s2_graded"]
    w("## M1, action 2: graded tool coverage inside R1\n")
    w("**Question.** Does the C-versus-B reversal (and the GRU's collapse) appear inside R1, with no change of domain, when test-time tool coverage "
      "alone is reduced? " + s2["design"]["mask_rule"] + " " + s2["design"]["estimand"] + ".\n")
    w("| Level | achieved coverage mean (range) | LR-C AUROC | LR-C-drop AUROC | LR-B AUROC | GRU AUROC | GRU-drop0.2 AUROC |")
    w("|---|---|---|---|---|---|---|")
    for Lk, lev in s2["levels"].items():
        D = lev["detectors"]
        w(f"| {Lk} | {lev['coverage_mean']:.3f} ({lev['coverage_min']:.3f}-{lev['coverage_max']:.3f}) | {f3(D['LR-C']['AUROC'])} | {f3(D['LR-C-drop']['AUROC'])} | "
          f"{f3(D['LR-B']['AUROC'])} | {f3(D['GRU']['AUROC'])} | {f3(D['GRU-drop0.2']['AUROC'])} |")
    w("\n| Level | LR-C FPR@theta_r80 | LR-C-drop FPR | LR-B FPR | GRU FPR | GRU-drop0.2 FPR |")
    w("|---|---|---|---|---|---|")
    for Lk, lev in s2["levels"].items():
        D = lev["detectors"]
        w(f"| {Lk} | {f3(D['LR-C']['FPR@t_r80'])} | {f3(D['LR-C-drop']['FPR@t_r80'])} | {f3(D['LR-B']['FPR@t_r80'])} | {f3(D['GRU']['FPR@t_r80'])} | {f3(D['GRU-drop0.2']['FPR@t_r80'])} |")
    w("\n| Level | LR-C - LR-B | LR-C-drop - LR-B | GRU - LR-B | GRU-drop0.2 - LR-B | GRU - LR-C |")
    w("|---|---|---|---|---|---|")
    for Lk, lev in s2["levels"].items():
        C = lev["contrasts"]
        w(f"| {Lk} | {fd(C['LR-C-minus-LR-B']['AUROC'])} | {fd(C['LR-C-drop-minus-LR-B']['AUROC'])} | {fd(C['GRU-minus-LR-B']['AUROC'])} | "
          f"{fd(C['GRU-drop0.2-minus-LR-B']['AUROC'])} | {fd(C['GRU-minus-LR-C']['AUROC'])} |")
    w("\n[s2_graded.levels.<level>.detectors.<det>.AUROC / FPR@t_r80; .contrasts.<A>-minus-<B>.AUROC; per-draw values in .per_draw, recall in .recall@t_r80]. "
      "Mean tool-pathway contribution on R1 negatives, level 1 -> level 0: "
      + "; ".join(f"{det} {s2['levels']['1.0']['detectors'][det]['tool_pathway_mean_on_negatives']:+.3f} -> {s2['levels']['0.0']['detectors'][det]['tool_pathway_mean_on_negatives']:+.3f}"
                  for det in ["LR-C", "LR-C-drop", "GRU", "GRU-drop0.2"]) + " [s2_graded.levels.<level>.detectors.<det>.tool_pathway_mean_on_negatives].\n")
    ch = s2["change_level0_minus_level1"]
    a0 = s2["levels"]["0.0"]["contrasts"]; l5 = s2["levels"]["0.5"]
    w("**Plain reading.** Lowering tool coverage inside R1 reproduces the R3 pattern with no change of domain. "
      f"LR-C falls from {s2['levels']['1.0']['detectors']['LR-C']['AUROC']['point']:.3f} to {s2['levels']['0.0']['detectors']['LR-C']['AUROC']['point']:.3f} "
      f"(change {fd(ch['LR-C']['AUROC'])}) [s2_graded.change_level0_minus_level1.LR-C.AUROC]. Its FPR at the training threshold rises by "
      f"{fd(ch['LR-C']['FPR@t_r80'])}. The C > B advantage is gone at coverage about 0.5 (LR-C - LR-B {fd(l5['contrasts']['LR-C-minus-LR-B']['AUROC'])}), "
      f"and at zero coverage the sign has reversed ({fd(a0['LR-C-minus-LR-B']['AUROC'])}). "
      f"The GRU falls by {fd(ch['GRU']['AUROC'])} and ends at {fd(a0['GRU-minus-LR-B']['AUROC'])} relative to LR-B and {fd(a0['GRU-minus-LR-C']['AUROC'])} "
      f"relative to LR-C, with FPR rising by {fd(ch['GRU']['FPR@t_r80'])}. "
      f"Dropout reduces the AUROC loss (LR-C-drop {fd(ch['LR-C-drop']['AUROC'])}, GRU-drop0.2 {fd(ch['GRU-drop0.2']['AUROC'])}). It largely removes the FPR rise "
      f"(LR-C-drop {fd(ch['LR-C-drop']['FPR@t_r80'])}, GRU-drop0.2 {fd(ch['GRU-drop0.2']['FPR@t_r80'])}). It does not remove the loss: at zero coverage LR-C-drop is still "
      f"{fd(a0['LR-C-drop-minus-LR-B']['AUROC'])} below LR-B, GRU-drop0.2 is {fd(a0['GRU-drop0.2-minus-LR-B']['AUROC'])} relative to LR-B and "
      f"{fd(a0['GRU-drop0.2-minus-LR-C']['AUROC'])} relative to LR-C. "
      "So removing tool identity at test time is enough to produce both reversals within one distribution. This does not show that nothing else "
      "changes under R3, where the behavioural distribution also shifts. The masks also remove tool names at random across domains, whereas R3 removes "
      "whole domains.\n")

    # ---------------- M1 action 3
    s3 = R["s3_dropout"]
    w("## M1, action 3: dropout arms in R1 and R3, paired contrasts, plan 3.4 rule\n")
    w("**Question.** With tool-block dropout (LR-C-drop) and token dropout (GRU-drop0.2 primary, GRU-drop0.1 sensitivity), do the in-distribution "
      "orderings still reverse under R3? " + s3["design"]["bootstrap"] + ". " + s3["design"]["rule_3_4"] + ".\n")
    w("AUROC (pool E) by population [s3_dropout.populations.<pop>.detectors.<det>.AUROC]:\n")
    w("| Population | LR-B | LR-C | LR-C-drop | GRU | GRU-drop0.2 | GRU-drop0.1 |")
    w("|---|---|---|---|---|---|---|")
    for pop in ["R1"] + [f"R1dm_{d}" for d in DOMAINS] + [f"R3_{d}" for d in DOMAINS]:
        D = s3["populations"][pop]["detectors"]
        w(f"| {pop} | " + " | ".join(f3(D[det]["AUROC"]) for det in DETS) + " |")
    w("\nRecall and FPR at theta_r80 [s3_dropout.populations.<pop>.detectors.<det>.recall@t_r80 / FPR@t_r80]:\n")
    w("| Population | LR-B rec / FPR | LR-C rec / FPR | LR-C-drop rec / FPR | GRU rec / FPR | GRU-drop0.2 rec / FPR | GRU-drop0.1 rec / FPR |")
    w("|---|---|---|---|---|---|---|")
    for pop in ["R1"] + [f"R1dm_{d}" for d in DOMAINS] + [f"R3_{d}" for d in DOMAINS]:
        D = s3["populations"][pop]["detectors"]
        w(f"| {pop} | " + " | ".join(f"{f3(D[det]['recall@t_r80'])} / {f3(D[det]['FPR@t_r80'])}" for det in DETS) + " |")
    w("\nPaired AUROC differences, domain-matched R1 and R3, with Holm over the five folds [s3_dropout.reversal_rule_3_4.<A>-minus-<B>.per_fold.<fold>]:\n")
    w("| Contrast | Fold | R1 domain-matched | Holm p | R3 | Holm p | reversal (3.4) |")
    w("|---|---|---|---|---|---|---|")
    for key in ["LR-C-minus-LR-B", "GRU-minus-LR-C", "LR-C-drop-minus-LR-B", "GRU-drop0.2-minus-LR-C", "GRU-drop0.2-minus-LR-B",
                "GRU-drop0.1-minus-LR-C", "GRU-drop0.1-minus-LR-B", "GRU-drop0.2-minus-LR-C-drop", "GRU-minus-LR-B", "LR-C-drop-minus-LR-C"]:
        rv = s3["reversal_rule_3_4"][key]
        for d in DOMAINS:
            x = rv["per_fold"][d]
            w(f"| {key} | {d} | {fd(x['R1_domain_matched'])} | {x['R1_domain_matched']['p_holm']:.4f} | {fd(x['R3'])} | {x['R3']['p_holm']:.4f} | {'yes' if x['reversal_holm'] else 'no'} |")
    w("")
    w("Folds meeting the plan 3.4 reversal rule [s3_dropout.reversal_rule_3_4.<contrast>.folds_reversal_holm], and folds where the R3 difference is "
      "detected below / above zero after Holm:\n")
    for key in ["LR-C-minus-LR-B", "GRU-minus-LR-C", "GRU-minus-LR-B", "LR-C-drop-minus-LR-B", "GRU-drop0.2-minus-LR-C", "GRU-drop0.2-minus-LR-B",
                "GRU-drop0.1-minus-LR-C", "GRU-drop0.1-minus-LR-B", "GRU-drop0.2-minus-LR-C-drop"]:
        rv = s3["reversal_rule_3_4"][key]
        w(f"- {key}: reversal {rv['folds_reversal_holm'] or 'none'}; R3 below {rv['folds_R3_detected_below_holm'] or 'none'}; "
          f"R3 above {rv['folds_R3_detected_above_holm'] or 'none'}; R1 domain-matched above {rv['folds_R1dm_detected_above_holm'] or 'none'}; "
          f"R1 pooled {fd(rv['R1_pooled'])}.")
    w("")
    th = s3["threshold_change_R3_minus_R1dm"]
    w("Threshold change, R3 minus domain-matched R1 at theta_r80 (independent bootstraps, Holm over five folds) "
      "[s3_dropout.threshold_change_R3_minus_R1dm.<det>.<recall@t_r80|FPR@t_r80>.<fold>]:\n")
    w("| Detector | metric | medical | banking | web | email | coding |")
    w("|---|---|---|---|---|---|---|")
    for det in DETS:
        for met in ["recall@t_r80", "FPR@t_r80"]:
            w(f"| {det} | {met} | " + " | ".join(fd(th[det][met][d]) + (" H" if th[det][met][d]["detected_holm"] else "") for d in DOMAINS) + " |")
    w("\n(H = detected after Holm.)\n")
    dec = s3["decision"]; mg = s3["magnitudes"]; tc = s3["threshold_change_counts"]

    def fl(x):
        return ", ".join(x) if x else "none"
    cd = dec["LR-C-drop"]["LR-C-drop-minus-LR-B"]; g2c = dec["GRU-drop0.2"]["GRU-drop0.2-minus-LR-C"]; g2b = dec["GRU-drop0.2"]["GRU-drop0.2-minus-LR-B"]
    g1c = dec["GRU-drop0.1"]["GRU-drop0.1-minus-LR-C"]; g1b = dec["GRU-drop0.1"]["GRU-drop0.1-minus-LR-B"]
    w("**Plain reading (plan 3.4 rule).** "
      f"Without remedies, the C-versus-B reversal meets the rule in {fl(dec['original_LR-C-minus-LR-B'])}. It cannot be claimed in medical or banking, "
      "where no in-distribution C > B advantage is detected. The GRU-versus-C reversal meets the rule in "
      f"{fl(dec['original_GRU-minus-LR-C'])}; in web the in-distribution GRU advantage is not detected. "
      f"(1) Tool-block dropout: LR-C-drop - LR-B still meets the rule in {fl(cd['folds_reversal_holm'])}. Its R3 disadvantage is small, "
      f"{mg['LR-C-drop-minus-LR-B']['R3_min']:+.3f} to {mg['LR-C-drop-minus-LR-B']['R3_max']:+.3f} across folds, against "
      f"{mg['LR-C-minus-LR-B']['R3_min']:+.3f} to {mg['LR-C-minus-LR-B']['R3_max']:+.3f} for LR-C [s3_dropout.magnitudes]. It is detected because the two "
      f"scores rank the held-out trajectories almost identically (Spearman rho >= {mg['spearman_R3_LR-C-drop_vs_LR-B_min']:.3f} in every fold, against "
      f"{mg['spearman_R3_LR-C_vs_LR-B_min']:.3f}-{mg['spearman_R3_LR-C_vs_LR-B_max']:.3f} for LR-C vs LR-B) [s3_dropout.magnitudes.spearman_R3_poolE], "
      "so the paired interval is very narrow. So under the letter of the rule the reversal survives tool-block dropout in these folds. "
      f"In absolute size it is {mg['ratio_abs_R3_LR-C-drop-minus-LR-B_to_LR-C-minus-LR-B']['min']:.0%} to "
      f"{mg['ratio_abs_R3_LR-C-drop-minus-LR-B_to_LR-C-minus-LR-B']['max']:.0%} of the LR-C gap in the same fold "
      "[s3_dropout.magnitudes.ratio_abs_R3_LR-C-drop-minus-LR-B_to_LR-C-minus-LR-B], a near tie in practice. "
      f"(2) Token dropout, p = 0.2 (primary): GRU-drop0.2 - LR-C meets the rule in {fl(g2c['folds_reversal_holm'])} and is detected above LR-C after Holm in {fl(g2c['folds_R3_detected_above_holm'])}; its R3 difference ranges "
      f"{mg['GRU-drop0.2-minus-LR-C']['R3_min']:+.3f} to {mg['GRU-drop0.2-minus-LR-C']['R3_max']:+.3f}. So the GRU-versus-C reversal does not survive. "
      f"GRU-drop0.2 is detected below LR-B under R3 in {fl(g2b['folds_R3_detected_below_holm'])} (R3 differences over the five folds {mg['GRU-drop0.2-minus-LR-B']['R3_min']:+.3f} to "
      f"{mg['GRU-drop0.2-minus-LR-B']['R3_max']:+.3f}), although it is {mg['GRU-drop0.2-minus-LR-B']['R1dm_min']:+.3f} to {mg['GRU-drop0.2-minus-LR-B']['R1dm_max']:+.3f} "
      f"above LR-B in-distribution. That GRU-versus-B reversal meets the rule in {fl(g2b['folds_reversal_holm'])}. "
      f"(3) p = 0.1 (sensitivity): versus LR-C the rule is met in {fl(g1c['folds_reversal_holm'])}; versus LR-B in {fl(g1b['folds_reversal_holm'])}. "
      "Dropout changes the direction of the threshold failure rather than removing it [s3_dropout.threshold_change_counts]. Relative to the "
      f"domain-matched R1 value (Holm), LR-C's recall rises in {tc['LR-C']['recall@t_r80']['n_up_holm']}/5 folds and its FPR in {tc['LR-C']['FPR@t_r80']['n_up_holm']}/5. "
      f"For GRU-drop0.2 recall falls in {tc['GRU-drop0.2']['recall@t_r80']['n_down_holm']}/5 and FPR changes in "
      f"{tc['GRU-drop0.2']['FPR@t_r80']['n_up_holm'] + tc['GRU-drop0.2']['FPR@t_r80']['n_down_holm']}/5. "
      f"For GRU-drop0.1 recall falls in {tc['GRU-drop0.1']['recall@t_r80']['n_down_holm']}/5 and FPR changes in "
      f"{tc['GRU-drop0.1']['FPR@t_r80']['n_up_holm'] + tc['GRU-drop0.1']['FPR@t_r80']['n_down_holm']}/5. "
      f"For LR-C-drop recall rises in {tc['LR-C-drop']['recall@t_r80']['n_up_holm']} and falls in {tc['LR-C-drop']['recall@t_r80']['n_down_holm']} folds, "
      f"and FPR rises in {tc['LR-C-drop']['FPR@t_r80']['n_up_holm']} and falls in {tc['LR-C-drop']['FPR@t_r80']['n_down_holm']}. "
      f"LR-B changes in {tc['LR-B']['recall@t_r80']['n_up_holm'] + tc['LR-B']['recall@t_r80']['n_down_holm']} (recall) and "
      f"{tc['LR-B']['FPR@t_r80']['n_up_holm'] + tc['LR-B']['FPR@t_r80']['n_down_holm']} (FPR) folds.\n")

    # ---------------- M1 action 4 + m6
    s4 = R["s4_gru_offset"]
    w("## M1, action 4: GRU tool-pathway offset over all folds and seeds; m6: <UNK> norm\n")
    w("**Question.** " + s4["definition"] + "\n")
    w("| Fold | detector | train negatives | held-out negatives | held-out positives | offset (held-out - train negatives), seed mean [95% CI] |")
    w("|---|---|---|---|---|---|")
    for s in ["R1"] + [f"R3_{d}" for d in DOMAINS] + ["R4"]:
        for det in GRUS:
            m = s4["settings"][s][det]["mean_over_seeds"]; ci = s4["settings"][s][det]["offset_neg_seedmean_ci"]
            w(f"| {s} | {det} | {m['train_neg_mean']:+.3f} | {m['heldout_neg_mean']:+.3f} | {m['heldout_pos_mean']:+.3f} | {m['offset_neg']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] |")
    w("\n[s4_gru_offset.settings.<setting>.<det>.mean_over_seeds; per seed in .per_seed]. Over the 15 R3 fold x seed cells:\n")
    for det in GRUS:
        a = s4["summary_R3"][det]
        w(f"- {det}: training negatives mean {a['train_neg_mean']['mean']:+.3f} (range {a['train_neg_mean']['min']:+.3f} to {a['train_neg_mean']['max']:+.3f}); "
          f"held-out negatives {a['heldout_neg_mean']['mean']:+.3f} ({a['heldout_neg_mean']['min']:+.3f} to {a['heldout_neg_mean']['max']:+.3f}); "
          f"held-out positives {a['heldout_pos_mean']['mean']:+.3f} ({a['heldout_pos_mean']['min']:+.3f} to {a['heldout_pos_mean']['max']:+.3f}); "
          f"offset {a['offset_neg']['mean']:+.3f} ({a['offset_neg']['min']:+.3f} to {a['offset_neg']['max']:+.3f}); held-out negative mean above training negative mean in "
          f"{a['n_cells_heldout_neg_above_train_neg']}/{a['n_cells']} cells [s4_gru_offset.summary_R3.{det}].")
    u = s4["unk"]["summary_R3"]
    w("\n<UNK> embedding norm, R3 folds x seeds [s4_gru_offset.unk.summary_R3.<det>; per cell in s4_gru_offset.unk.<setting>.<det>]:\n")
    for det in GRUS:
        a = u[det]
        extra = (f"; Adam-with-weight-decay-only simulation reproduces the trained norm (largest difference {a['adam_prediction_max_abs_err']:.1e})" if det == "GRU" else "")
        w(f"- {det}: initial norm {a['unk_init_norm_min']:.3f}-{a['unk_init_norm_max']:.3f} (reproduced by re-seeding: {a['all_init_reproduced']}); trained norm "
          f"{a['unk_trained_norm_min']:.3f}-{a['unk_trained_norm_max']:.3f} (mean {a['unk_trained_norm_mean']:.3f}); median trained tool-embedding norm "
          f"{a['tool_median_trained_min']:.3f}-{a['tool_median_trained_max']:.3f}; median initial tool-embedding norm {a['tool_median_init_min']:.3f}-{a['tool_median_init_max']:.3f}; "
          f"<UNK> training steps (before dropout) max {a['unk_train_steps_max']}{extra}.")
    e0 = s4["unk"]["R3_medical"]["GRU"][0]
    w(f"\nExample (R3 medical, seed 0): initial {e0['unk_init_norm_reproduced']:.3f}, trained {e0['unk_trained_norm']:.3f}, median trained tool norm "
      f"{e0['tool_emb_trained_norm_median']:.3f}; {e0['optimizer_steps_to_best']} optimizer steps to the restored epoch; plain multiplicative decay (1 - lr*wd)^T "
      f"would leave {e0['multiplicative_decay_prediction']:.3f}; Adam with the coupled L2 term alone gives {e0['adam_decay_only_predicted_norm']:.3f} "
      f"[s4_gru_offset.unk.R3_medical.GRU.0]. For the refitted R1 models the simulated <UNK> vector matches the trained one coordinate-wise to "
      + ", ".join(f"{e.get('adam_prediction_max_abs_coord_err_vs_trained_vector', float('nan')):.1e}" for e in s4["unk"]["R1"]["GRU"])
      + " [s4_gru_offset.unk.R1.GRU.<seed>.adam_prediction_max_abs_coord_err_vs_trained_vector]. "
      f"Within AgentDojo (WUT), <UNK> does occur in training: up to {s4['wut_unk']['unk_train_steps_max']} steps in {s4['wut_unk']['unk_train_traces_max']} traces per fold, "
      f"trained norm {s4['wut_unk']['unk_norm_min']:.3f}-{s4['wut_unk']['unk_norm_max']:.3f} [s4_gru_offset.wut_unk].\n")
    a = s4["summary_R3"]["GRU"]; b = s4["summary_R3"]["GRU-drop0.2"]; c = s4["summary_R3"]["GRU-drop0.1"]
    exc = "; ".join(f"{x['fold']} seed {x['seed']}: offset {x['offset_neg']:+.2f}, FPR at theta_r80 {x['heldout_FPR_t_r80']:.3f}, recall {x['heldout_recall_t_r80']:.3f}"
                    for x in a["cells_not_above"]) or "none"
    ug = u["GRU"]; u2 = u["GRU-drop0.2"]
    r1off = s4["settings"]["R1"]["GRU"]["mean_over_seeds"]["offset_neg"]
    w("**Plain reading.** The review's one-model observation holds across folds and seeds. In "
      f"{a['n_cells_heldout_neg_above_train_neg']} of {a['n_cells']} R3 fold x seed cells, the pre-registered GRU's tool pathway gives held-out "
      f"negatives a higher contribution than training negatives. The mean offset is {a['offset_neg']['mean']:+.2f} logits (range "
      f"{a['offset_neg']['min']:+.2f} to {a['offset_neg']['max']:+.2f}), against {r1off:+.2f} in R1. So under the tool pathway, held-out benign "
      f"traces look like attacked training traces (training positives average {a['train_pos_mean']['mean']:+.2f}). The exception ({exc}) "
      "is the seed that raises almost no alerts in that fold. "
      "The <UNK> row gets no data gradient in AgentDrift training (0 <UNK> training steps). Its trained value is what Adam's coupled weight decay "
      "alone does to the random initialisation: the simulation matches the trained norm in every cell "
      f"(largest |simulated - cached| norm difference {ug['adam_prediction_max_abs_err']:.1e}"
      + (", i.e. bit-identical in float32" if ug['adam_prediction_max_abs_err'] == 0 else "")
      + "); for the refitted R1 models the vectors match coordinate for coordinate "
      f"(largest difference {max(e['adam_prediction_max_abs_coord_err_vs_trained_vector'] for e in s4['unk']['R1']['GRU']):.1e}). "
      f"Adam normalises each coordinate's "
      "gradient, so the small L2 gradient moves every coordinate towards 0 by about lr per step. After "
      f"{ug['optimizer_steps_to_best_min']:,}-{ug['optimizer_steps_to_best_max']:,} steps the row falls from "
      f"{ug['unk_init_norm_min']:.2f}-{ug['unk_init_norm_max']:.2f} to {ug['unk_trained_norm_min']:.2f}-{ug['unk_trained_norm_max']:.2f}. The trained "
      f"tool embeddings have median norms {ug['tool_median_trained_min']:.2f}-{ug['tool_median_trained_max']:.2f} "
      f"(initial medians {ug['tool_median_init_min']:.2f}-{ug['tool_median_init_max']:.2f}). Plain multiplicative decay, (1 - lr*wd)^T, would have left "
      f"the row almost unchanged (at least {ug['multiplicative_decay_prediction_min']:.2f}). "
      "The manuscript's 'changes only through weight decay' is literally true but misleading: the row is an untrained random vector, shrunk by the "
      "optimiser, and its norm depends on how many steps early stopping allowed. "
      f"With token dropout the <UNK> row is trained (p = 0.2: norm {u2['unk_trained_norm_min']:.2f}-{u2['unk_trained_norm_max']:.2f}). The offset falls to "
      f"{b['offset_neg']['min']:+.2f} to {b['offset_neg']['max']:+.2f} (p = 0.1: {c['offset_neg']['min']:+.2f} to {c['offset_neg']['max']:+.2f}). "
      "It remains positive in every cell, which fits the lower but still present R3 loss of the dropout arms. These means describe the mechanism. "
      "They do not show by themselves how much of the AUROC loss the offset causes, because the tool contribution also varies across held-out "
      "trajectories.\n")

    # ---------------- M1 action 5 (LOSO)
    s5 = R["s5_loso"]
    w("## M1, action 5a: AgentDojo leave-one-suite-out\n")
    w("**Question.** Does the C-versus-B reversal replicate on real traces when a whole suite (and most of its tools) is held out? "
      + s5["design"]["bootstrap_primary"] + "; " + s5["design"]["in_distribution"] + ".\n")
    w("| Held-out suite | pos / neg (clusters) | tool-occurrence coverage | traces with zero known tools | LR-B | LR-C | LR-C-drop |")
    w("|---|---|---|---|---|---|---|")
    for st in SUITES:
        e = s5["suites"][st]; C = e["LOSO_cluster"]
        w(f"| {st} | {e['n_pos']} / {e['n_neg']} ({e['n_clusters']}) | {e['coverage']['tool_occurrence_cov']:.3f} | {e['coverage']['frac_traces_zero_known']:.3f} | "
          f"{f3(C['LR-B']['AUROC'])} | {f3(C['LR-C']['AUROC'])} | {f3(C['LR-C-drop']['AUROC'])} |")
    w("\n[s5_loso.suites.<suite>.coverage; .LOSO_cluster.<det>.AUROC]\n")
    w("| Held-out suite | LR-C - LR-B, LOSO (cluster) | Holm p | same, WUT suite-matched | Holm p | LR-C-drop - LR-B, LOSO | LR-C-drop - LR-C, LOSO |")
    w("|---|---|---|---|---|---|---|")
    for st in SUITES:
        c = s5["suites"][st]["contrasts"]
        w(f"| {st} | {fd(c['LR-C-minus-LR-B']['LOSO_cluster'])} | {c['LR-C-minus-LR-B']['LOSO_cluster']['p_holm']:.4f} | {fd(c['LR-C-minus-LR-B']['WUT_suite_matched_cluster'])} | "
          f"{c['LR-C-minus-LR-B']['WUT_suite_matched_cluster']['p_holm']:.4f} | {fd(c['LR-C-drop-minus-LR-B']['LOSO_cluster'])} | {fd(c['LR-C-drop-minus-LR-C']['LOSO_cluster'])} |")
    wp = s5["WUT_pooled"]["contrasts"]
    w(f"\nIn distribution, pooled over all suites (WUT; the same resample as the P0 rows of s6_rule4): LR-C - LR-B {fd(wp['LR-C-minus-LR-B']['cluster'])} (cluster), {fd(wp['LR-C-minus-LR-B']['stratified'])} (stratified); "
      f"LR-C-drop - LR-B {fd(wp['LR-C-drop-minus-LR-B']['cluster'])}; LR-C-drop - LR-C {fd(wp['LR-C-drop-minus-LR-C']['cluster'])} [s5_loso.WUT_pooled.contrasts]. "
      "LOSO stratified-bootstrap sensitivity for LR-C - LR-B: "
      + ", ".join(f"{st} {fd(s5['suites'][st]['contrasts']['LR-C-minus-LR-B']['LOSO_stratified'])}" for st in SUITES)
      + " [s5_loso.suites.<suite>.contrasts.LR-C-minus-LR-B.LOSO_stratified].\n")
    rv = s5["reversal"]["LR-C-minus-LR-B"]
    cb = {st: s5["suites"][st]["contrasts"]["LR-C-minus-LR-B"] for st in SUITES}
    thr_txt = "; ".join(f"{st}: LR-C FPR {s5['suites'][st]['LOSO_cluster']['LR-C']['FPR@t_r80']['point']:.3f} / recall {s5['suites'][st]['LOSO_cluster']['LR-C']['recall@t_r80']['point']:.3f}, "
                        f"LR-B FPR {s5['suites'][st]['LOSO_cluster']['LR-B']['FPR@t_r80']['point']:.3f} / recall {s5['suites'][st]['LOSO_cluster']['LR-B']['recall@t_r80']['point']:.3f}"
                        for st in ["banking", "slack"])
    w("**Plain reading.** "
      f"No held-out suite meets the reversal rule for LR-C versus LR-B (suites: {', '.join(rv['suites_reversal_holm']) or 'none'}). With the cluster "
      "bootstrap, the in-distribution C > B advantage is not detected in any single suite, so the rule's first condition fails everywhere. "
      f"Pooled over suites it is also not detected ({fd(wp['LR-C-minus-LR-B']['cluster'])}); it is detected only with the trajectory-level bootstrap "
      f"({fd(wp['LR-C-minus-LR-B']['stratified'])}). "
      f"On the held-out suites, C - B is detected above zero after Holm in {', '.join(rv['suites_LOSO_detected_above_holm']) or 'no suite'} and below zero in "
      f"{', '.join(rv['suites_LOSO_detected_below_holm']) or 'no suite'}. Banking has zero coverage and C - B = {fd(cb['banking']['LOSO_cluster'])}: not detected with "
      f"clusters, but detected with the trajectory-level bootstrap ({fd(cb['banking']['LOSO_stratified'])}). Slack also has zero coverage and shows the "
      f"opposite sign ({fd(cb['slack']['LOSO_cluster'])}). "
      "At zero coverage the ranking depends only on LR-C's behavioural weights, and whether they rank a new suite better or worse than LR-B's is "
      "not consistent. The zero-block offset is negative in AgentDojo (action 1), so at the training threshold LR-C raises fewer alerts than "
      f"LR-B on those suites ({thr_txt}) [s5_loso.suites.<suite>.LOSO_cluster.<det>]. That is the opposite of the AgentDrift FPR rise. "
      "Tool-block dropout gives LR-C-drop - LR-B detected above zero in "
      f"{', '.join(s5['reversal']['LR-C-drop-minus-LR-B']['suites_LOSO_detected_above_holm']) or 'no suite'}. "
      "Overall the AgentDrift reversal does not replicate as a consistent pattern on real multi-suite traces. With 22-47 clusters per suite the "
      "intervals are wide, and a small reversal in single suites (for example banking) cannot be excluded.\n")

    # ---------------- M1 action 5 (Rule 4)
    s6 = R["s6_rule4"]
    w("## M1, action 5b: pre-registered Rule 4 in R4 (corrected data)\n")
    w("**Question.** " + s6["design"]["rule"] + ".\n")
    w(f"Precondition, R1 LR-C - LR-B: {fd(s6['R1_C_minus_B'])} [s6_rule4.R1_C_minus_B].\n")
    w("| Population | bootstrap | transfer C - B | within C - B | degradation C (within - transfer) | degradation B | DD (within ref.) | DD (R1 ref.) |")
    w("|---|---|---|---|---|---|---|---|")
    for pop in ["P0", "C1"]:
        for bn in ["cluster", "stratified"]:
            e = s6["R4"][pop][bn]
            w(f"| {pop} | {bn} | {fd(e['transfer_C_minus_B'])} | {fd(e['within_C_minus_B'])} | {fd(e['degradation_within_minus_transfer']['LR-C'])} | "
              f"{fd(e['degradation_within_minus_transfer']['LR-B'])} | {fd(e['DD_within_ref'])} | {fd(e['DD_R1_ref'])} |")
    w("\n[s6_rule4.R4.<P0|C1>.<cluster|stratified>]. AUROC levels (P0, cluster): "
      + ", ".join(f"{k} {f3(v)}" for k, v in s6["R4"]["P0"]["cluster"]["AUROC"].items()) + ".\n")
    rs = s6["R3_supplementary"]
    w("Supplementary R3 clause, DD = (R1_C - R3_C) - (R1_B - R3_B): pooled R1 "
      + ", ".join(f"{d} {fd(rs['pooled_R1'][d])}" for d in DOMAINS)
      + f" ({rs['n_folds_detected_favouring_B_holm_pooled']}/5 detected favouring B after Holm); domain-matched R1 "
      + ", ".join(f"{d} {fd(rs['domain_matched_R1'][d])}" for d in DOMAINS)
      + f" ({rs['n_folds_detected_favouring_B_holm_domain_matched']}/5) [s6_rule4.R3_supplementary].\n")
    p0 = s6["R4"]["P0"]["cluster"]; p0s = s6["R4"]["P0"]["stratified"]
    w("**Plain reading.** The precondition holds: C > B in R1 is detected "
      f"({fd(s6['R1_C_minus_B'])}). On the corrected data both detectors are near chance in transfer "
      f"(LR-C {f3(p0['AUROC']['transfer_LR-C'])}, LR-B {f3(p0['AUROC']['transfer_LR-B'])}). C - B in transfer is small but detected "
      f"({fd(p0['transfer_C_minus_B'])}, cluster). The degradation difference with the within-AgentDojo reference is {fd(p0['DD_within_ref'])} with the "
      f"cluster bootstrap and {fd(p0s['DD_within_ref'])} with the pre-registered stratified bootstrap. With the R1 reference it is {fd(p0['DD_R1_ref'])}. "
      "All are positive and detected, so LR-C loses more than LR-B, and Rule 4's R4 clause is met (Outcome 4). The test is weak, as the review "
      "notes. Both transfer AUROCs are close to 0.5, so the within-AgentDojo degradation difference mostly restates the within-AgentDojo C - B gap, "
      f"which the cluster bootstrap does not detect on its own ({fd(p0['within_C_minus_B'])}); the cluster interval of the degradation difference "
      "only just excludes 0. The R3 clause is also met in all five folds with either R1 reference, but the pre-registration calls it partly structural. "
      "Supported reading: the detector that uses more benchmark-specific information loses more when moved across benchmarks. It is not evidence that "
      "the behavioural representation transfers, since LR-B is itself at chance in R4.\n")

    # ---------------- Q1
    s7 = R["s7_q1"]
    w("## Q1: GRU interval estimand, per-seed R3 values and a two-level bootstrap\n")
    w("**Question.** Is the GRU interval that of a seed-averaged detector or of a single-seed deployment? " + s7["design"]["two_level"] + "; "
      + s7["design"]["mixture"] + "; " + s7["design"]["seed_averaged"] + ". " + s7["design"]["note"] + ".\n")
    w("| Fold | GRU AUROC seeds 0/1/2 | GRU - LR-C per seed | two-level | mixture (pre-registered) | seed-averaged |")
    w("|---|---|---|---|---|---|")
    for name in ["R1"] + [f"R3_{d}" for d in DOMAINS]:
        e = s7["folds"][name]["GRU"]
        seeds_auc = " / ".join("%.3f" % v for v in e["AUROC_per_seed"])
        seeds_d = " / ".join("%+.3f" % x["point"] for x in e["minus_LR-C_per_seed"])
        w(f"| {name} | {seeds_auc} | {seeds_d} | "
          f"{fd(e['minus_LR-C']['two_level'])} | {fd(e['minus_LR-C']['mixture'])} | {fd(e['minus_LR-C']['seed_averaged'])} |")
    w("\n[s7_q1.folds.<fold>.GRU]. Per-seed GRU - LR-C intervals (trajectory bootstrap) are in .minus_LR-C_per_seed. Dropout arms (sensitivity), two-level GRU-drop - LR-C: "
      + "; ".join(f"{det}: " + ", ".join(f"{d} {fd(s7['folds'][f'R3_{d}'][det]['minus_LR-C']['two_level'])}" for d in DOMAINS) for det in ["GRU-drop0.2", "GRU-drop0.1"]) + ".\n")
    rt = s7["reversal_two_level"]; rsa = s7["reversal_seed_averaged"]
    w("Plan 3.4 rule re-applied with the two-level and the seed-averaged bootstrap [s7_q1.reversal_two_level / reversal_seed_averaged.<contrast>.folds_reversal_holm]: "
      + "; ".join(f"{k}: two-level {rt[k]['folds_reversal_holm'] or 'none'}, seed-averaged {rsa[k]['folds_reversal_holm'] or 'none'}" for k in rt) + ".\n")
    em = s7["folds"]["R3_email"]["GRU"]
    em_d = ", ".join("%+.3f" % x["point"] for x in em["minus_LR-C_per_seed"])
    allneg = all(s7["folds"][f"R3_{d}"]["GRU"]["minus_LR-C"][v]["detected"] and s7["folds"][f"R3_{d}"]["GRU"]["minus_LR-C"][v]["point"] < 0
                 for d in DOMAINS for v in ["two_level", "mixture", "seed_averaged"])
    seedneg = all(x["ci"][1] < 0 for d in DOMAINS for x in s7["folds"][f"R3_{d}"]["GRU"]["minus_LR-C_per_seed"])
    w("**Plain reading.** "
      f"The per-seed email differences are {em_d}, as the review reported. The sign is the same for every seed, but the size varies about fourfold. "
      + ("In every R3 fold each single seed's GRU - LR-C interval lies below zero, " if seedneg else "Not every single-seed GRU - LR-C interval lies below zero, ")
      + ("and the fold-level difference is detected below zero under all three estimands. So the GRU-versus-C conclusion in R3 does not depend on the estimand. "
         if allneg else "and the fold-level conclusion differs between estimands in at least one fold. ")
      + "What does depend on the estimand is the "
      "width of the intervals. The pre-registered interval samples one seed per replicate, so it describes a randomly drawn single-seed deployment, "
      "while its point estimate is the seed mean. The seed-averaged interval is the narrowest and describes the mean of these three trained models "
      "only. The two-level interval is the closest to 'a seed-averaged detector with training randomness included'. With three seeds its seed "
      "level is coarse, so it should be read as approximate. For the dropout arms, where differences to LR-C are small, the reversal decisions under "
      "the three estimands are listed above. The manuscript should state which estimand it reports and give the per-seed values.\n")

    w("## Deviations and limitations\n")
    for d in R["deviations"]:
        w(f"- {d}")
    w("")
    with open(SUMMARY, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    log(f"wrote {SUMMARY}")


# ============================================================================ main
if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "train":
        cmd_train()
    elif cmd == "graded":
        cmd_graded()
    elif cmd == "analyze":
        cmd_analyze()
    elif cmd == "summary":
        cmd_summary()
    elif cmd == "all":
        if cmd_train() and cmd_graded():
            cmd_analyze(); cmd_summary()
        else:
            print("INCOMPLETE: call again", flush=True)
    else:
        raise SystemExit(__doc__)
