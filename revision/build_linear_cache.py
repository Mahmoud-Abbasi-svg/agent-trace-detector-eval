"""Build the LINEAR prediction cache for the revision (REVISION_ANALYSIS_PLAN.md, Section 2).

One pickle per setting: revision/cache/linear/<setting>.pkl  (format: see CACHE_README.md).
Resumable: `--setting R1` builds one setting (skips it if its pickle exists AND was written by the current
version of this script, identified by its SHA-256; --force rebuilds regardless);
`--all` loops over every setting, skipping finished ones, and stops starting new work once the
time budget (--budget seconds, default 420) is used, so that it can be called repeatedly.
Cross-validated settings (WUT, WUT_F, WIT) save one part file per repetition in
cache/linear/_parts/ and are assembled when all five repetitions exist.
`--status` prints which settings are done / stale / missing without fitting anything.

Detectors (fitted exactly like ablation.py / rerun_r1r3.py; no tuning; thresholds from training data):
  LR-A, LR-B, LR-C                 as ablation.py (StandardScaler on training, LogisticRegression(max_iter=1000,
                                   class_weight="balanced"), vocabulary from benign training trajectories)
  LR-B-noobs, LR-C-noobs           behavioural features 8, 9 (max / mean observation length) dropped
  LR-B-minus                       behavioural features 4, 5 (keyword state-changing count / fraction) dropped
  LR-C-drop                        LR-C fitted on [X; X with the tool block zeroed], labels duplicated, scaler on
                                   the stacked matrix; theta_r80 from the ORIGINAL training positives only
  Markov, IForest                  exactly as rerun_r1r3.py
  TFIDF, TFIDF-nodigit             TfidfVectorizer(lowercase, (1,2), min_df=2, max_features=50000, sublinear_tf)
                                   + LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
  TFIDF-worldmask (R1 only)        world-dict string values replaced by WORLDTOKEN before vectorising
Thresholds: t_r80 = evalkit.thr_at_recall(in-sample training-positive scores, 0.80); t05 = 0.5.
Exact ties: fitting is exactly the v0.3 recipe, but mathematically tied scores are made bitwise tied (see
by_unique_rows / MARKOV_DECIMALS), because float noise in tied scores otherwise moves AUROC (e.g. R4 Markov).

Run (Git Bash):
  AGENTDRIFT=<clone> AGENTDOJO=<repo> C:/Python310/python.exe revision/build_linear_cache.py --all
"""
# The cache format (one pickle per setting) is fixed by the revision's shared-cache specification. Pickles are
# written and read only by the revision scripts on this machine (trusted, self-produced files).
import os, sys, re, json, time, pickle, argparse, hashlib, collections, platform, warnings
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))            # .../agent-ids/revision
ROOT = os.path.dirname(HERE)                                  # .../agent-ids
sys.path.insert(0, ROOT)

import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.exceptions import ConvergenceWarning

from featurise import load, build_vocab, vectorise, tool_seq, AGNOSTIC_NAMES   # noqa: E402
import evalkit as ek                                                           # noqa: E402
from baseline_drift import Markov                                              # noqa: E402
from world_disjoint_r1 import world_disjoint_split                             # noqa: E402
import ablation                                                                # noqa: E402  (reproduction checks)
import r4_control                                                              # noqa: E402  (grouping + checks)
import rerun_r1r3                                                              # noqa: E402  (Markov/IForest check)

MP = os.environ["AGENTDRIFT"]
CACHE = os.path.join(HERE, "cache", "linear")
PARTS = os.path.join(CACHE, "_parts")
LOGF = os.path.join(CACHE, "build_log.txt")
K = 5
CV_SEEDS = [0, 1, 2, 3, 4]
DOMAINS = ["medical", "banking", "web", "email", "coding"]
SUITES = ["banking", "slack", "travel", "workspace"]
SETTINGS = (["R1"] + [f"R1wd_s{i}" for i in range(5)] + ["R2"] + [f"R3_{d}" for d in DOMAINS]
            + ["R4", "WUT", "WUT_F", "WIT"] + [f"LOSO_{s}" for s in SUITES])
CV_SETTINGS = {"WUT": ("UT", False), "WUT_F": ("UT", True), "WIT": ("IT", False)}
LR_DETS = ["LR-A", "LR-B", "LR-C", "LR-B-noobs", "LR-C-noobs", "LR-B-minus", "LR-C-drop"]
ONECLASS_DETS = ["Markov", "IForest"]
TEXT_DETS = ["TFIDF", "TFIDF-nodigit"]
NBEH = 10
POOL_E_CATS = ("benign", "hard_negative", "attacked")
DOJO_P0 = os.path.join(ROOT, "agentdojo_traces.jsonl")
DOJO_FAILED = os.path.join(ROOT, "agentdojo_failed.jsonl")


def log(msg):
    os.makedirs(CACHE, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOGF, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


SCRIPT_SHA = sha256(os.path.abspath(__file__))
TOL = 1e-6          # reproduction tolerance on scores / thresholds. Differences are floating-point only (matrix
                    # assembly order); observed maxima: <1e-11 single fits, <1e-7 within-AgentDojo LR-C folds.


# ----------------------------------------------------------------------------- data
_DATA = {}


def data(name):
    if name in _DATA:
        return _DATA[name]
    if name == "st_tr":
        v = load(os.path.join(MP, "data", "train.jsonl"))
    elif name == "st_te":
        v = load(os.path.join(MP, "data", "test.jsonl"))
    elif name == "td_tr":
        v = load(os.path.join(MP, "data_taskdisjoint", "train.jsonl"))
    elif name == "td_te":
        v = load(os.path.join(MP, "data_taskdisjoint", "test.jsonl"))
    elif name == "pool":
        v = data("st_tr") + data("st_te")
    elif name == "p0":
        v = [json.loads(l) for l in open(DOJO_P0, encoding="utf-8")]
    elif name == "failed":
        v = [json.loads(l) for l in open(DOJO_FAILED, encoding="utf-8")]
    else:
        raise KeyError(name)
    bench = "agentdojo" if name in ("p0", "failed") else "agentdrift"
    for r in v:
        r.setdefault("_bench", bench)
    _DATA[name] = v
    return v


def ylab(recs):
    return np.array([int(r["category"] == "attacked") for r in recs], int)


def parse_id(r):
    m, suite, ut, inj, at = r["id"].split("|")
    return dict(model=m, suite=suite, ut=ut, inj=inj, at=at)


def world_key(r):
    return json.dumps(r["world"], sort_keys=True)


def meta_row(r, cov):
    y = int(r["category"] == "attacked")
    if r["_bench"] == "agentdrift":
        return {"id": r["id"], "benchmark": "agentdrift", "category": r["category"], "y": y,
                "in_pool_E": r["category"] in POOL_E_CATS, "domain": r["agent"],
                "pattern": r.get("source_category") if r["category"] == "attacked" else None,
                "world": world_key(r), "task": r["task"], "model": None, "user_task": None,
                "injection_task": None, "attack_type": None, "is_dos": None,
                "n_steps": len(r["steps"]), "cov": cov}
    mt = r["meta"]
    return {"id": r["id"], "benchmark": "agentdojo", "category": r["category"], "y": y,
            "in_pool_E": r["category"] in ("benign", "attacked"), "domain": mt["suite"], "pattern": None,
            "world": None, "task": f"{mt['suite']}|{mt['user_task']}", "model": mt["model"],
            "user_task": mt["user_task"], "injection_task": mt["injection_task"],
            "attack_type": mt["attack_type"], "is_dos": bool(mt["is_dos"]),
            "n_steps": len(r["steps"]), "cov": cov}


# ----------------------------------------------------------------------------- features
_BEH = {}


def beh(recs):
    """Behavioural (vocabulary-independent) block = vectorise(...)[0]; cached per record id."""
    out = []
    for r in recs:
        k = id(r)            # object identity: ids repeat across the stratified and task-disjoint files
        if k not in _BEH:
            _BEH[k] = vectorise(r, {})[0]
        out.append(_BEH[k])
    return np.array(out, float)


def bag(recs, vocab):
    """Bag-of-tools block, identical to vectorise(...)[1] (asserted in check_features)."""
    X = np.zeros((len(recs), len(vocab)), float)
    for i, r in enumerate(recs):
        for t in tool_seq(r):
            j = vocab.get(t)
            if j is not None:
                X[i, j] += 1.0
    return X


def check_features(recs, vocab):
    Xc = np.hstack([beh(recs), bag(recs, vocab)])
    Xref = ablation.X_of(recs, vocab, "C")
    assert Xc.shape == Xref.shape and np.array_equal(Xc, Xref), "feature matrix differs from ablation.X_of"


def row_cov(recs, vocab):
    return np.array([float(np.mean([s["tool"] in vocab for s in r["steps"]])) if r["steps"] else float("nan")
                     for r in recs])


def occ_cov(recs, vocab):
    occ = [s["tool"] in vocab for r in recs for s in r["steps"]]
    return float(np.mean(occ)) if occ else float("nan")


def det_cols(det, V):
    bagc = list(range(NBEH, NBEH + V))
    return {"LR-A": bagc, "LR-B": list(range(NBEH)), "LR-C": list(range(NBEH)) + bagc,
            "LR-B-noobs": list(range(8)), "LR-C-noobs": list(range(8)) + bagc,
            "LR-B-minus": [0, 1, 2, 3, 6, 7, 8, 9], "LR-C-drop": list(range(NBEH)) + bagc}[det]


# ----------------------------------------------------------------------------- text
def traj_text(r):
    return " || ".join(f"{s['tool']} {str(s['args'])} {s['obs']}" for s in r["steps"])


_DIG = re.compile(r"\d")


def world_strings(o):
    if isinstance(o, dict):
        for v in o.values():
            yield from world_strings(v)
    elif isinstance(o, list):
        for v in o:
            yield from world_strings(v)
    elif isinstance(o, str):
        yield o


_MASK_STATS = []
_MASK_RESID = []


def person_name_tokens(w):
    """Alphabetic tokens (>= 3 letters) of the user's and the contacts' names (for the residual-leak statistic)."""
    names = [w.get("user", "")] + [c.get("name", "") for c in w.get("contacts", []) if isinstance(c, dict)]
    return sorted({t for n in names if isinstance(n, str) for t in re.findall(r"[A-Za-z]{3,}", n)})


def worldmask_text(r):
    """Every string value of the world dict (user, email, company, contact name/email/relation, date) is
    replaced by WORLDTOKEN: case-insensitive, whole occurrences (no adjacent word character), longest first.
    Also records (i) the number of replacements and (ii) whether a single first/last-name token of the user or a
    contact still occurs after masking (a partial name such as "Hi Mai" is not a full world string value)."""
    vals = sorted({v for v in world_strings(r["world"]) if v.strip()}, key=len, reverse=True)
    t = traj_text(r)
    if not vals:
        _MASK_STATS.append(0); _MASK_RESID.append(False)
        return t
    pat = re.compile(r"(?<!\w)(?:" + "|".join(re.escape(v) for v in vals) + r")(?!\w)", re.IGNORECASE)
    t2, n = pat.subn("WORLDTOKEN", t)
    _MASK_STATS.append(n)
    toks = person_name_tokens(r["world"])
    _MASK_RESID.append(bool(toks) and re.search(r"(?<![A-Za-z])(?:" + "|".join(map(re.escape, toks)) + r")(?![A-Za-z])",
                                                t2, re.IGNORECASE) is not None)
    return t2


TEXT_FNS = {"TFIDF": traj_text,
            "TFIDF-nodigit": lambda r: _DIG.sub("", traj_text(r)),
            "TFIDF-worldmask": worldmask_text}
_TXT = {}


def texts(recs, det):
    out = []
    for r in recs:
        k = (det, id(r))    # records are kept alive in _DATA, so object identity is stable
        if k not in _TXT:
            _TXT[k] = TEXT_FNS[det](r)
        out.append(_TXT[k])
    return out


# ----------------------------------------------------------------------------- fitting
# ----------------------------------------------------------------------------- exact ties
# Mathematically tied scores must be bitwise tied, otherwise AUROC/AUPRC depend on floating-point noise. Two sources:
#  (1) dense BLAS products give identical feature rows slightly different results depending on their position in the
#      matrix (~1e-16). Fix: score each DISTINCT feature row (or distinct text) once and broadcast. Fitting is untouched.
#  (2) Markov.score is a mean of per-transition log-probabilities; trajectories whose transitions are all unseen have
#      the same mathematical score log(|V|+1) but the float mean depends on the length. Fix: round Markov scores to
#      MARKOV_DECIMALS decimals (scores are O(1); float noise O(1e-15)). Thresholds use the canonicalized train scores.
MARKOV_DECIMALS = 12
SCORE_NOTES = ("scores are P(attacked) for LR-*/TFIDF*, -score_samples for IForest, mean negative transition "
               "log-prob for Markov; exact ties made bitwise exact (LR-*, IForest, TFIDF* scored once per distinct "
               "feature row / text; Markov rounded to 12 decimals); t05 = 0.5 is a probability threshold for LR-*/TFIDF*, "
               "equals sklearn's default outlier cut for IForest (offset -0.5), and has no calibrated meaning for Markov")


def by_unique_rows(fn, A):
    """fn applied to the distinct rows of dense A, broadcast back (identical rows -> bitwise-identical output)."""
    U, inv = np.unique(A, axis=0, return_inverse=True)
    return np.asarray(fn(U))[np.asarray(inv).reshape(-1)]


def by_unique_texts(fn, txts):
    U, inv = np.unique(np.asarray(txts, dtype=object), return_inverse=True)
    return np.asarray(fn(list(U)))[np.asarray(inv).reshape(-1)]


def lr_fit(Xtr, ytr, drop_tool_from=None):
    """ablation.fit recipe on a given matrix. drop_tool_from = first tool column -> tool-block dropout."""
    if drop_tool_from is not None:
        Xz = Xtr.copy(); Xz[:, drop_tool_from:] = 0.0
        Xfit = np.vstack([Xtr, Xz]); yfit = np.concatenate([ytr, ytr])
    else:
        Xfit, yfit = Xtr, ytr
    sc = StandardScaler().fit(Xfit)
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always", ConvergenceWarning)
        lr = LogisticRegression(max_iter=1000, class_weight="balanced").fit(sc.transform(Xfit), yfit)
    conv = not any(issubclass(w.category, ConvergenceWarning) for w in wl)
    s_tr = by_unique_rows(lambda U: lr.predict_proba(sc.transform(U))[:, 1], Xtr)   # ORIGINAL training rows
    t80 = ek.thr_at_recall(s_tr[ytr == 1], 0.80)
    return sc, lr, t80, s_tr, conv


def fit_score(tr, te, parts=False, text_dets=TEXT_DETS):
    """Fit every detector on `tr`, score `te`. Returns dict with scores/pred/thr/info (+ parts/coefs)."""
    ytr = ylab(tr)
    assert 0 < ytr.sum() < len(ytr), "single-class training set"
    vocab = build_vocab([r for r in tr if r["category"] == "benign"])
    V = len(vocab)
    names = list(AGNOSTIC_NAMES) + ["tool:" + t for t, _ in sorted(vocab.items(), key=lambda kv: kv[1])]
    Xtr = np.hstack([beh(tr), bag(tr, vocab)]); Xte = np.hstack([beh(te), bag(te, vocab)])
    S, P, T, conv, niter = {}, {}, {}, {}, {}
    lrc_parts, coefs = {}, {}
    for det in LR_DETS:
        c = det_cols(det, V)
        Atr, Ate = Xtr[:, c], Xte[:, c]
        first_tool = NBEH if det == "LR-C-drop" else None
        sc, lr, t80, s_tr, ok = lr_fit(Atr, ytr, first_tool)
        Zte = sc.transform(Ate)
        s = by_unique_rows(lambda U: lr.predict_proba(sc.transform(U))[:, 1], Ate)
        S[det] = s; P[det] = {"t_r80": s >= t80, "t05": s >= 0.5}; T[det] = {"t_r80": t80, "t05": 0.5}
        conv[det] = ok; niter[det] = int(lr.n_iter_[0])
        if parts:
            w = lr.coef_[0]; b0 = float(lr.intercept_[0])
            coefs[det] = {"names": [names[i] for i in c], "coef": w.copy(), "intercept": b0,
                          "scaler_mean": sc.mean_.copy(), "scaler_scale": sc.scale_.copy()}
            if det in ("LR-C", "LR-C-drop"):
                Ztr = sc.transform(Atr)
                tool_te = Zte[:, NBEH:] @ w[NBEH:]; beh_te = Zte[:, :NBEH] @ w[:NBEH]
                assert np.allclose(tool_te + beh_te + b0, lr.decision_function(Zte), atol=1e-8)
                zero_tool = float(((0.0 - sc.mean_[NBEH:]) / sc.scale_[NBEH:]) @ w[NBEH:])
                lrc_parts[det] = {"tool": tool_te, "beh": beh_te, "intercept": b0,
                                  "zero_tool_part": zero_tool,
                                  "train_tool_part": Ztr[:, NBEH:] @ w[NBEH:],
                                  "train_beh_part": Ztr[:, :NBEH] @ w[:NBEH],
                                  "train_y": ytr.copy(),
                                  "train_category": [r["category"] for r in tr],
                                  "train_ids": [r["id"] for r in tr]}
    # ---- Markov + IForest exactly as rerun_r1r3.run
    ben = np.array([r["category"] == "benign" for r in tr])
    mk = Markov().fit([tool_seq(r) for r in tr if r["category"] == "benign"])
    s_tr = np.round(np.array([mk.score(tool_seq(r)) for r in tr]), MARKOV_DECIMALS)
    s = np.round(np.array([mk.score(tool_seq(r)) for r in te]), MARKOV_DECIMALS)
    t80 = ek.thr_at_recall(s_tr[ytr == 1])
    S["Markov"] = s; P["Markov"] = {"t_r80": s >= t80, "t05": s >= 0.5}; T["Markov"] = {"t_r80": t80, "t05": 0.5}
    sc = StandardScaler().fit(Xtr)
    ifo = IsolationForest(n_estimators=200, random_state=0).fit(sc.transform(Xtr)[ben])
    s_tr = by_unique_rows(lambda U: -ifo.score_samples(sc.transform(U)), Xtr)
    s = by_unique_rows(lambda U: -ifo.score_samples(sc.transform(U)), Xte)
    t80 = ek.thr_at_recall(s_tr[ytr == 1])
    S["IForest"] = s; P["IForest"] = {"t_r80": s >= t80, "t05": s >= 0.5}; T["IForest"] = {"t_r80": t80, "t05": 0.5}
    # ---- TF-IDF content detectors
    nfeat = {}
    for det in text_dets:
        vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True)
        ttr = texts(tr, det); Mtr = vec.fit_transform(ttr)
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter("always", ConvergenceWarning)
            lr = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0).fit(Mtr, ytr)
        conv[det] = not any(issubclass(w.category, ConvergenceWarning) for w in wl)
        niter[det] = int(lr.n_iter_[0]); nfeat[det] = len(vec.vocabulary_)
        prob = lambda U, vec=vec, lr=lr: lr.predict_proba(vec.transform(U))[:, 1]
        s_tr = by_unique_texts(prob, ttr); t80 = ek.thr_at_recall(s_tr[ytr == 1])
        s = by_unique_texts(prob, texts(te, det))
        S[det] = s; P[det] = {"t_r80": s >= t80, "t05": s >= 0.5}; T[det] = {"t_r80": t80, "t05": 0.5}
    te_E = [r for r in te if r["category"] in POOL_E_CATS]
    info = {"n_train": len(tr), "n_train_pos": int(ytr.sum()),
            "n_train_by_category": dict(collections.Counter(r["category"] for r in tr)),
            "vocab_size": V, "kappa": occ_cov(te_E, vocab), "kappa_all_rows": occ_cov(te, vocab),
            "lr_converged": conv, "lr_n_iter": niter, "tfidf_n_features": nfeat}
    out = {"scores": S, "pred": P, "thr": T, "info": info, "cov": row_cov(te, vocab), "vocab": vocab}
    if parts:
        out["lrc_parts"] = lrc_parts; out["coefs"] = coefs
    return out


# ----------------------------------------------------------------------------- settings
def single_split(setting):
    """Return (train records, test records, notes) for a single-fit setting."""
    if setting == "R1":
        return data("st_tr"), data("st_te"), ["train data/train.jsonl, test data/test.jsonl; val unused"]
    if setting.startswith("R1wd_s"):
        sd = int(setting[-1]); pool = data("pool")
        tr_i, te_i = world_disjoint_split(pool, sd)
        tr = [pool[i] for i in tr_i]; te = [pool[i] for i in te_i]
        ov = len({world_key(r) for r in tr} & {world_key(r) for r in te})
        return tr, te, [f"pooled train+test split by world_disjoint_r1.world_disjoint_split(pool, {sd}); world overlap {ov}"]
    if setting == "R2":
        return data("td_tr"), data("td_te"), ["data_taskdisjoint train/test"]
    if setting.startswith("R3_"):
        d = setting[3:]; pool = data("pool")
        return ([r for r in pool if r["agent"] != d], [r for r in pool if r["agent"] == d],
                [f"pooled train+test; train on the other four domains, test on {d}"])
    if setting == "R4":
        return data("pool"), data("p0") + data("failed"), \
            ["train all 10,656 pooled AgentDrift trajectories; test AgentDojo P0 then agentdojo_failed.jsonl (file order)"]
    if setting.startswith("LOSO_"):
        s = setting[5:]
        tr = [r for r in data("p0") if r["agent"] != s]
        te = [r for r in data("p0") if r["agent"] == s] + [r for r in data("failed") if r["agent"] == s]
        return tr, te, [f"train P0 rows of the other three suites; test all rows (P0 then failed) of {s}"]
    raise KeyError(setting)


def build_single(setting):
    t0 = time.time()
    tr, te, notes = single_split(setting)
    text_dets = TEXT_DETS + (["TFIDF-worldmask"] if setting == "R1" else [])
    _MASK_STATS.clear(); _MASK_RESID.clear(); _TXT.clear()
    res = fit_score(tr, te, parts=True, text_dets=text_dets)
    y = ylab(te); poolE = np.array([r["category"] in POOL_E_CATS for r in te])
    meta = [meta_row(r, float(c)) for r, c in zip(te, res["cov"])]
    info = res["info"]
    info.update({"n_test": len(te), "n_test_pos": int(y.sum()),
                 "n_test_by_category": dict(collections.Counter(r["category"] for r in te)),
                 "n_pool_E": int(poolE.sum()), "notes": notes, "scheme": "single", "score_notes": SCORE_NOTES})
    if setting == "R1":
        info["worldmask_replacements_per_traj"] = {"mean": float(np.mean(_MASK_STATS)),
                                                   "min": int(np.min(_MASK_STATS)), "max": int(np.max(_MASK_STATS)),
                                                   "frac_traj_zero": float(np.mean(np.array(_MASK_STATS) == 0)),
                                                   "n": len(_MASK_STATS),
                                                   "note": "over train + test trajectories (each text built once)"}
        info["worldmask_frac_traj_with_residual_name_token"] = float(np.mean(_MASK_RESID))
    info["checks"] = run_checks(setting, tr, te, res)
    info["runtime_s"] = round(time.time() - t0, 1)
    obj = {"setting": setting, "family": "linear", "test_meta": meta, "y": y, "pool_E": poolE,
           "scores": res["scores"], "pred": res["pred"], "thr": res["thr"], "info": info,
           "lrc_parts": res["lrc_parts"], "coefs": res["coefs"], "detectors": list(res["scores"])}
    return obj


def cmp_scores(ref_s, s, ref_pred=None, pred=None, ref_thr=None, thr=None, y=None):
    """Quantitative comparison with a v0.3 code path. Scores may differ slightly because the matrices are assembled
    in a different order (floating-point summation order) and because the cache makes exact ties bitwise exact;
    pass = max |diff| <= TOL and identical thresholded predictions. AUROC of both is reported (if y is given)."""
    d = {"max_abs_diff": float(np.max(np.abs(np.asarray(ref_s, float) - np.asarray(s, float))))}
    ok = d["max_abs_diff"] <= TOL
    if y is not None:
        d["auroc_ref"] = ek.auroc(np.asarray(y), np.asarray(ref_s, float)); d["auroc_cache"] = ek.auroc(np.asarray(y), np.asarray(s, float))
        d["auroc_equal"] = bool(d["auroc_ref"] == d["auroc_cache"])   # can differ when the v0.3 path broke exact ties by float noise
    if ref_thr is not None:
        d["thr_abs_diff"] = float(abs(ref_thr - thr)); ok = ok and d["thr_abs_diff"] <= TOL
    if ref_pred is not None:
        d["n_pred_t_r80_differ"] = int(np.sum(np.asarray(ref_pred) != np.asarray(pred))); ok = ok and d["n_pred_t_r80_differ"] == 0
    d["pass"] = bool(ok)
    return d


def run_checks(setting, tr, te, res):
    """Reproduction checks against the v0.3 code paths (same data, same code), run in-process."""
    ch = {}
    if setting in ("R1", "R2") or setting.startswith("R3_") or setting == "R4":
        vocab = build_vocab([r for r in tr if r["category"] == "benign"])
        check_features(te[:300], vocab)
        ch["feature_matrix_equals_ablation.X_of_first300"] = {"pass": True}
        for rep, det in [("A", "LR-A"), ("B", "LR-B"), ("C", "LR-C")]:
            md = ablation.fit(tr, rep); sref = ablation.sco(md, te, rep)
            ch[f"{det}_vs_ablation.fit"] = cmp_scores(sref["score"], res["scores"][det], sref["pred"]["t_r80"],
                                                     res["pred"][det]["t_r80"], md[3], res["thr"][det]["t_r80"], ylab(te))
    if setting in ("R1", "R2") or setting.startswith("R3_"):
        y, meth, comp = rerun_r1r3.run(tr, te)
        keep = np.array([r["category"] in POOL_E_CATS for r in te])
        for ref, det in [("MARKOV", "Markov"), ("IFOREST", "IForest"), ("LOGREG", "LR-C")]:
            ch[f"{det}_vs_rerun_r1r3.run"] = cmp_scores(meth[ref][0]["score"], res["scores"][det][keep],
                                                        meth[ref][0]["pred"]["t_r80"], res["pred"][det]["t_r80"][keep], y=y)
    if setting.startswith("R1wd_s"):
        import world_disjoint_r1 as wd
        for rep, det in [("A", "LR-A"), ("B", "LR-B"), ("C", "LR-C")]:
            md = wd.fit(tr, rep); sref = wd.sco(md, te, rep)
            ch[f"{det}_vs_world_disjoint_r1.fit"] = cmp_scores(sref["score"], res["scores"][det], sref["pred"]["t_r80"],
                                                              res["pred"][det]["t_r80"], md[3], res["thr"][det]["t_r80"], ylab(te))
    if setting == "R4":
        for agn, det in [(False, "LR-C"), (True, "LR-B")]:
            md = r4_control.fit_lr(tr, agn); s, pr = r4_control.score(md, te, agn)
            ch[f"{det}_vs_r4_control.fit_lr"] = cmp_scores(s, res["scores"][det], pr["t_r80"], res["pred"][det]["t_r80"],
                                                          md[3], res["thr"][det]["t_r80"], ylab(te))
    return ch


# ----------------------------------------------------------------------------- cross-validated settings
def assign_folds(P0, FAILED, scheme, seed):
    """P0 folds exactly as r4_control.within_oof (and ablation.py's within-UT): groups = sorted keys of the P0
    rows, shuffled with default_rng(seed), fold = position % K. Unsuccessful attacks use their suite|user_task
    group; a group that no P0 row has is assigned by the same rule with default_rng(10000 + seed) (logged)."""
    keys = [r4_control.group_key(r, scheme) for r in P0]
    groups = sorted(set(keys)); rng = np.random.default_rng(seed); rng.shuffle(groups)
    fold_of = {g: i % K for i, g in enumerate(groups)}
    fkeys = [r4_control.group_key(r, "UT") for r in FAILED]          # suite|user_task
    extra = sorted(set(fkeys) - set(fold_of))
    rng2 = np.random.default_rng(10000 + seed); rng2.shuffle(extra)
    fold_extra = {g: i % K for i, g in enumerate(extra)}
    folds = np.array([fold_of[k] for k in keys] + [fold_of.get(k, fold_extra.get(k)) for k in fkeys])
    n_extra_rows = int(sum(k not in fold_of for k in fkeys))
    return folds, {"n_groups_P0": len(groups), "extra_groups": extra, "n_failed_rows_in_extra_groups": n_extra_rows}


def build_cv_rep(setting, rep):
    scheme, add_failed = CV_SETTINGS[setting]
    P0, FAILED = data("p0"), data("failed")
    rows = P0 + FAILED; n = len(rows); nP = len(P0)
    seed = CV_SEEDS[rep]
    folds, finfo = assign_folds(P0, FAILED, scheme, seed)
    S = {}; P = {}; T = {}; cov = np.full(n, np.nan)
    finfo_rows = []
    for f in range(K):
        tr_idx = [i for i in range(nP) if folds[i] != f]
        if add_failed:
            tr_idx += [i for i in range(nP, n) if folds[i] != f]
        te_idx = np.where(folds == f)[0]
        tr = [rows[i] for i in tr_idx]; te = [rows[i] for i in te_idx]
        yte_p0 = ylab([rows[i] for i in te_idx if i < nP])
        assert 0 < yte_p0.sum() < len(yte_p0), f"{setting} rep {rep} fold {f}: single-class P0 test fold"
        res = fit_score(tr, te, parts=False)
        for det, s in res["scores"].items():
            if det not in S:
                S[det] = np.full(n, np.nan); P[det] = {"t_r80": np.zeros(n, bool), "t05": np.zeros(n, bool)}
                T[det] = {"t_r80": [np.nan] * K, "t05": 0.5}
            S[det][te_idx] = s
            for k in ("t_r80", "t05"):
                P[det][k][te_idx] = res["pred"][det][k]
            T[det]["t_r80"][f] = res["thr"][det]["t_r80"]
        cov[te_idx] = res["cov"]
        fi = dict(res["info"]); fi.update({"fold": f, "n_test": len(te_idx)})
        finfo_rows.append(fi)
    assert all(not np.isnan(S[d]).any() for d in S), "missing out-of-fold scores"
    # exact check of the P0 part against r4_control.within_oof (v0.3 code path on the corrected data)
    ch = {}
    rfolds = None
    for agn, det in [(False, "LR-C"), (True, "LR-B")]:
        ref, rfolds = r4_control.within_oof(P0, scheme, seed, agn)
        if not add_failed:     # WUT_F trains on extra negatives, so only its fold assignment can be compared
            ch[f"{det}_P0_vs_r4_control.within_oof"] = cmp_scores(ref["score"], S[det][:nP], ref["pred"]["t_r80"],
                                                                 P[det]["t_r80"][:nP], y=ylab(P0))
    ch[f"P0_folds_equal_r4_control_{scheme}"] = {"pass": bool(np.array_equal(rfolds, folds[:nP]))}
    return {"rep": rep, "seed": seed, "scores": S, "pred": P, "thr": T, "cov": cov, "folds": folds,
            "fold_info": finfo_rows, "fold_assignment": finfo, "checks": ch}


def assemble_cv(setting):
    scheme, add_failed = CV_SETTINGS[setting]
    P0, FAILED = data("p0"), data("failed"); rows = P0 + FAILED
    reps = [pickle.load(open(os.path.join(PARTS, f"{setting}_rep{r}.pkl"), "rb")) for r in range(len(CV_SEEDS))]
    dets = list(reps[0]["scores"])
    scores = {d: np.vstack([rp["scores"][d] for rp in reps]) for d in dets}
    pred = {d: {k: np.vstack([rp["pred"][d][k] for rp in reps]) for k in ("t_r80", "t05")} for d in dets}
    thr = {d: {"t_r80": np.array([rp["thr"][d]["t_r80"] for rp in reps]), "t05": 0.5} for d in dets}
    cov_reps = np.vstack([rp["cov"] for rp in reps])
    meta = [meta_row(r, float(c)) for r, c in zip(rows, cov_reps.mean(0))]
    y = ylab(rows); poolE = np.array([r["category"] in ("benign", "attacked") for r in rows])
    fi = [rp["fold_info"] for rp in reps]
    notes = [f"within-AgentDojo {K}-fold CV, scheme {scheme} (grouping of r4_control.group_key), fold seeds {CV_SEEDS}; "
             "training on P0 rows" + (" plus the unsuccessful attacks of the training folds (as negatives)" if add_failed else " only")
             + "; every row (P0 then failed, file order) gets an out-of-fold score from the model whose test fold holds its group",
             "unsuccessful attacks are grouped by suite|user_task; groups absent from P0 assigned with default_rng(10000+seed)",
             "test_meta cov = mean over repetitions of the per-row coverage under the scoring fold's vocabulary; per-rep values in cov_reps"]
    info = {"scheme": "cv", "cv_scheme": scheme, "failed_as_train_negatives": add_failed, "n_reps": len(reps), "K": K,
            "n_train": [[x["n_train"] for x in r] for r in fi], "n_train_pos": [[x["n_train_pos"] for x in r] for r in fi],
            "n_train_mean": float(np.mean([[x["n_train"] for x in r] for r in fi])),
            "n_train_pos_mean": float(np.mean([[x["n_train_pos"] for x in r] for r in fi])),
            "vocab_size": [[x["vocab_size"] for x in r] for r in fi],
            "kappa_folds": [[x["kappa"] for x in r] for r in fi],
            "kappa": float(np.mean([np.mean([x["kappa"] for x in r]) for r in fi])),
            "n_test": len(rows), "n_test_pos": int(y.sum()),
            "n_test_by_category": dict(collections.Counter(r["category"] for r in rows)), "n_pool_E": int(poolE.sum()),
            "fold_assignment": [rp["fold_assignment"] for rp in reps],
            "lr_converged_all": {d: bool(all(x["lr_converged"].get(d, True) for r in fi for x in r)) for d in dets if d not in ONECLASS_DETS},
            "fold_info": fi, "checks": [rp["checks"] for rp in reps], "notes": notes, "score_notes": SCORE_NOTES,
            "runtime_s": float(sum(rp.get("runtime_s", 0) for rp in reps))}
    # tool-occurrence coverage over all rows (occurrence-weighted), averaged over repetitions
    steps = np.array([len(r["steps"]) for r in rows], float)
    info["kappa_all_rows"] = float(np.mean([np.sum(c * steps) / steps.sum() for c in cov_reps]))
    return {"setting": setting, "family": "linear", "test_meta": meta, "y": y, "pool_E": poolE,
            "scores": scores, "pred": pred, "thr": thr, "info": info, "folds": np.vstack([rp["folds"] for rp in reps]),
            "cov_reps": cov_reps, "detectors": dets}


# ----------------------------------------------------------------------------- driver
def out_path(setting):
    return os.path.join(CACHE, f"{setting}.pkl")


def save(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(obj, fh, protocol=4)
    os.replace(tmp, path)


def provenance():
    return {"script": os.path.abspath(__file__), "script_sha256": SCRIPT_SHA, "python": platform.python_version(),
            "sklearn": sklearn.__version__, "numpy": np.__version__, "built": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "agentdojo_traces_sha256": sha256(DOJO_P0), "agentdojo_failed_sha256": sha256(DOJO_FAILED)}


def is_current(path, part=False):
    """True if `path` exists and was written by the current version of this script."""
    if not os.path.exists(path):
        return False
    try:
        with open(path, "rb") as fh:
            o = pickle.load(fh)
        sha = o.get("script_sha256") if part else o.get("provenance", {}).get("script_sha256")
        return sha == SCRIPT_SHA
    except Exception:
        return False


def state(setting):
    p = out_path(setting)
    return "done" if is_current(p) else ("stale" if os.path.exists(p) else "missing")


def do_setting(setting, force, t_start, budget, forced_this_call):
    """Returns True when the setting's current pickle exists at the end of the call."""
    path = out_path(setting)
    if not force and is_current(path):
        return True
    if setting in CV_SETTINGS:
        os.makedirs(PARTS, exist_ok=True)
        for rep in range(len(CV_SEEDS)):
            pp = os.path.join(PARTS, f"{setting}_rep{rep}.pkl")
            if is_current(pp, part=True) and not (force and (setting, rep) not in forced_this_call):
                continue
            if time.time() - t_start > budget:
                log(f"{setting}: budget used, stopping before rep {rep}")
                return False
            t0 = time.time(); part = build_cv_rep(setting, rep); part["runtime_s"] = round(time.time() - t0, 1)
            part["script_sha256"] = SCRIPT_SHA
            save(part, pp); forced_this_call.add((setting, rep))
            log(f"{setting}: rep {rep} done in {part['runtime_s']} s checks={part['checks']}")
        obj = assemble_cv(setting)
    else:
        if time.time() - t_start > budget:
            log(f"{setting}: budget used, not started")
            return False
        obj = build_single(setting)
    obj["provenance"] = provenance()
    save(obj, path)
    log(f"{setting}: wrote {path} n_test={obj['info']['n_test']} runtime={obj['info'].get('runtime_s')} s "
        f"checks={obj['info'].get('checks')}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", action="append", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--budget", type=float, default=420.0)
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()
    if a.status:
        for s in SETTINGS:
            print(f"{s:16s} {state(s)}")
        print(f"script_sha256 {SCRIPT_SHA}")
        return
    todo = SETTINGS if a.all else a.setting
    for s in todo:
        assert s in SETTINGS, f"unknown setting {s}"
    t_start = time.time(); done = []; forced = set()
    for s in todo:
        ok = do_setting(s, a.force, t_start, a.budget, forced)
        if ok:
            done.append(s)
        elif time.time() - t_start > a.budget:
            break
    missing = [s for s in SETTINGS if state(s) != "done"]
    print(f"STATUS done_or_current={done} not_current={missing}", flush=True)
    print("ALL_DONE" if not missing else "INCOMPLETE", flush=True)


if __name__ == "__main__":
    main()
