"""r4_deep.py - cross-benchmark transfer (R4) on the corrected AgentDojo extraction.

Addresses review comments M6, M7, m1, m2 (AgentDojo part), m6 (within-AgentDojo <UNK>), m7, m15 (model probe), m16,
m17 item 3, Q2 and Q4 (REVIEW_REPORT.md), following REVISION_ANALYSIS_PLAN.md. Every number written to
r4_deep_results.json is computed here from the shared prediction caches (revision/cache/linear, revision/cache/gru) or
from the new fits listed below; nothing is tuned on test data (thresholds come from the training folds).

New fits (the caches have no predictions for them; configurations copied by import, not re-implemented):
  lin_C1_UT, lin_C1_IT     every linear detector of the cache (build_linear_cache.fit_score) re-fitted within AgentDojo
                           on population C1 (P0 minus the 65 benign runs whose user task is an injection task), folds
                           exactly as r4_control.within_oof / ablation.py on C1 (the v0.3 C1 design).
  lin_noDoS_UT             same, on P0 without the 101 denial-of-service positives (Q4 refit sensitivity).
  gru_WIT                  GRU (gru_detector.train_gru, unmodified) under the injection-goal-disjoint scheme, folds of the
                           linear WIT cache (build_linear_cache.assign_folds), failed attacks scored out of fold.
  gru_WUT_F                GRU within-UT with the unsuccessful attacks of the training folds added as negatives.
  gru_C1_UT, gru_C1_IT     GRU on C1 (UT exactly as gru_ladder.py's C1 arm; IT new).
  gru_noDoS_UT             GRU on P0 without the DoS positives (Q4 refit sensitivity).
  gru_WUT_noUNK            GRU within-UT with the <UNK>-bearing traces removed from each fit partition (m6).
  gru_v03_transfer         the three R4 transfer GRUs refitted (deterministic) to score the ARCHIVED v0.3 extraction
                           (per-model v0.3 values were never computed for the GRU); refit checked against the cache.
  lin_v03                  LR-A/B/C transfer (ablation.fit) and LR-A within-UT/IT on the ARCHIVED v0.3 extraction.
Within-AgentDojo GRU units follow gru_ladder.py: repetition r uses fold seed r, training seed r and
gru_ladder.stratified_val(train, r).

Usage (Git Bash; AGENTDRIFT and AGENTDOJO must be set):
  C:/Python310/python.exe r4_deep.py status
  C:/Python310/python.exe r4_deep.py fit <job|all> [--budget S]
  C:/Python310/python.exe r4_deep.py analyze <section|all> [--budget S]
  C:/Python310/python.exe r4_deep.py assemble
Resumable: every fit unit and every analysis section (or sub-configuration) is saved to its own file under
revision/r4_deep_parts/ and skipped when it exists.
"""
# Pickle is the shared cache format fixed by the revision plan; only files written by the revision scripts on this
# machine (trusted, self-produced) are loaded.
import os, sys, json, time, pickle, argparse, collections, hashlib, datetime, warnings, re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))           # .../agent-ids/revision
ROOT = os.path.dirname(HERE)                                 # .../agent-ids
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)

import evalkit as ek                                         # noqa: E402
from featurise import load, build_vocab, vectorise            # noqa: E402
import r4_control                                             # noqa: E402

MP = os.environ["AGENTDRIFT"]
PARTS = os.path.join(HERE, "r4_deep_parts")
FITS = os.path.join(PARTS, "fits")
SECS = os.path.join(PARTS, "sections")
LOGF = os.path.join(HERE, "logs", "r4_deep.log")
RESULTS = os.path.join(HERE, "r4_deep_results.json")
SUMMARY = os.path.join(HERE, "r4_deep_summary.md")
CACHE_LIN = os.path.join(HERE, "cache", "linear")
CACHE_GRU = os.path.join(HERE, "cache", "gru")
ARCH = os.path.join(ROOT, "archive_v0.3")
DOJO_P0 = os.path.join(ROOT, "agentdojo_traces.jsonl")
DOJO_FA = os.path.join(ROOT, "agentdojo_failed.jsonl")
K = 5
REPS = [0, 1, 2, 3, 4]
B = 2000
SEED_STRAT = 12345          # r4_control.py / evalkit default (v0.3 convention)
SEED_CLUSTER = 20260927     # cluster bootstrap (new)
SEED_FAILED = 4242          # failed-attack flag-rate bootstrap (new)
GPT4O = "gpt-4o-2024-05-13"
LIN_DETS = ["LR-A", "LR-B", "LR-C", "LR-B-noobs", "LR-C-noobs", "LR-B-minus", "LR-C-drop", "Markov", "IForest",
            "TFIDF", "TFIDF-nodigit"]
MAIN5 = ["LR-A", "LR-B", "LR-C", "GRU", "TFIDF"]
PIS = (0.10, 0.05)


def log(msg):
    os.makedirs(os.path.dirname(LOGF), exist_ok=True)
    line = f"{datetime.datetime.now().isoformat(timespec='seconds')} {msg}"
    print(line, flush=True)
    with open(LOGF, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def pdump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(obj, fh, protocol=4)
    os.replace(tmp, path)


def pload(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def jdump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, default=_jdefault)
    os.replace(tmp, path)


def _jdefault(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def r4(x, d=4):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), d)


# =============================================================================================== data
_D = {}


def rows():
    """AgentDojo rows in cache order: P0 (agentdojo_traces.jsonl) then failed (agentdojo_failed.jsonl)."""
    if "rows" not in _D:
        P0 = [json.loads(l) for l in open(DOJO_P0, encoding="utf-8")]
        FA = [json.loads(l) for l in open(DOJO_FA, encoding="utf-8")]
        _D["rows"] = (P0, FA, P0 + FA)
    return _D["rows"]


def drift_pool():
    if "pool" not in _D:
        _D["pool"] = load(os.path.join(MP, "data", "train.jsonl")) + load(os.path.join(MP, "data", "test.jsonl"))
    return _D["pool"]


def v03_rows():
    if "v03" not in _D:
        _D["v03"] = [json.loads(l) for l in open(os.path.join(ARCH, "agentdojo_traces.jsonl"), encoding="utf-8")]
    return _D["v03"]


def meta():
    """Per-row metadata arrays over the 7,673 rows."""
    if "meta" in _D:
        return _D["meta"]
    P0, FA, R = rows()
    nP = len(P0)
    M = {"n": len(R), "nP0": nP}
    M["id"] = np.array([r["id"] for r in R], object)
    M["cat"] = np.array([r["category"] for r in R], object)
    M["y"] = np.array([int(r["category"] == "attacked") for r in R], int)
    M["is_p0"] = np.arange(len(R)) < nP
    M["is_failed"] = ~M["is_p0"]
    for k in ("model", "suite", "user_task", "injection_task", "attack_type"):
        M[k] = np.array([str(r["meta"][k]) for r in R], object)
    M["is_dos"] = np.array([bool(r["meta"]["is_dos"]) for r in R])
    M["n_steps"] = np.array([len(r["steps"]) for r in R], int)
    M["inj_benign"] = np.array([r["category"] == "benign" and r["meta"]["user_task"].startswith("injection_task")
                                for r in R])
    M["ut_key"] = np.array([f"{r['meta']['suite']}|{r['meta']['user_task']}" for r in R], object)
    M["it_key"] = np.array([f"{r['meta']['suite']}|{r['meta']['injection_task']}" for r in R], object)
    M["uti_key"] = np.array([f"{r['meta']['suite']}|{r['meta']['user_task']}|{r['meta']['injection_task']}" for r in R],
                            object)
    # check against the cache row order
    lin = cache("linear", "R4")
    assert [m["id"] for m in lin["test_meta"]] == list(M["id"]), "row order differs from the linear cache"
    _D["meta"] = M
    return M


def cache(fam, setting):
    k = ("cache", fam, setting)
    if k not in _D:
        _D[k] = pload(os.path.join(CACHE_LIN if fam == "linear" else CACHE_GRU, f"{setting}.pkl"))
    return _D[k]


# =============================================================================================== fold assignment
def subset_folds(recs, scheme, seed):
    """Exactly r4_control.within_oof's fold assignment on a population `recs` (also ablation.py / gru_ladder.py)."""
    keys = [r4_control.group_key(r, scheme) for r in recs]
    groups = sorted(set(keys)); rng = np.random.default_rng(seed); rng.shuffle(groups)
    fold_of = {g: i % K for i, g in enumerate(groups)}
    return np.array([fold_of[k] for k in keys])


def pop_indices(name):
    """Global row indices of a fitting population."""
    M = meta()
    P0idx = np.arange(M["nP0"])
    if name == "C1":
        return P0idx[~M["inj_benign"][:M["nP0"]]]
    if name == "P0_noDoS":
        return P0idx[~(M["is_dos"][:M["nP0"]] & (M["y"][:M["nP0"]] == 1))]
    if name == "P0F":
        return np.arange(M["n"])
    raise KeyError(name)


def job_folds(job, rep):
    """Global fold array (-1 = row not in the job) for one repetition."""
    M = meta(); P0, FA, R = rows()
    cfg = JOBS[job]
    f = np.full(M["n"], -1, int)
    if cfg["pop"] == "P0F":
        import build_linear_cache as blc
        folds, _ = blc.assign_folds(P0, FA, cfg["scheme"], rep)
        f[:] = folds
    else:
        idx = pop_indices(cfg["pop"])
        f[idx] = subset_folds([R[i] for i in idx], cfg["scheme"], rep)
    return f


JOBS = {
    "lin_C1_UT": dict(kind="lin", pop="C1", scheme="UT"),
    "lin_C1_IT": dict(kind="lin", pop="C1", scheme="IT"),
    "lin_noDoS_UT": dict(kind="lin", pop="P0_noDoS", scheme="UT"),
    "gru_WIT": dict(kind="gru", pop="P0F", scheme="IT", add_failed=False, drop_unk=False),
    "gru_WUT_F": dict(kind="gru", pop="P0F", scheme="UT", add_failed=True, drop_unk=False),
    "gru_C1_UT": dict(kind="gru", pop="C1", scheme="UT", add_failed=False, drop_unk=False),
    "gru_C1_IT": dict(kind="gru", pop="C1", scheme="IT", add_failed=False, drop_unk=False),
    "gru_noDoS_UT": dict(kind="gru", pop="P0_noDoS", scheme="UT", add_failed=False, drop_unk=False),
    "gru_WUT_noUNK": dict(kind="gru", pop="P0F", scheme="UT", add_failed=False, drop_unk=True),
    "gru_v03_transfer": dict(kind="gru_transfer"),
    "lin_v03": dict(kind="lin_v03"),
}


def units(job):
    k = JOBS[job]["kind"]
    if k == "lin":
        return [f"rep{r}" for r in REPS]
    if k == "gru":
        return [f"rep{r}_fold{f}" for r in REPS for f in range(K)]
    if k == "gru_transfer":
        return ["seed0", "seed1", "seed2"]
    if k == "lin_v03":
        return ["transfer", "withinA_UT", "withinA_IT"]
    raise KeyError(k)


def unit_path(job, u):
    return os.path.join(FITS, job, u + ".pkl")


# =============================================================================================== fitting
def fit_lin_unit(job, u):
    import build_linear_cache as blc
    P0, FA, R = rows(); M = meta()
    rep = int(u[3:])
    folds = job_folds(job, rep)
    cfg = JOBS[job]
    idx = np.where(folds >= 0)[0]
    S, P, T = {}, {}, {}
    cov = np.full(M["n"], np.nan)
    finfo = []
    for f in range(K):
        tr_i = idx[folds[idx] != f]; te_i = idx[folds[idx] == f]
        tr = [R[i] for i in tr_i]; te = [R[i] for i in te_i]
        res = blc.fit_score(tr, te, parts=False)
        for d, s in res["scores"].items():
            if d not in S:
                S[d] = np.full(M["n"], np.nan); P[d] = {"t_r80": np.zeros(M["n"], bool), "t05": np.zeros(M["n"], bool)}
                T[d] = [np.nan] * K
            S[d][te_i] = s
            for k in ("t_r80", "t05"):
                P[d][k][te_i] = res["pred"][d][k]
            T[d][f] = res["thr"][d]["t_r80"]
        cov[te_i] = res["cov"]
        finfo.append({"fold": f, "n_train": len(tr), "n_train_pos": int(sum(r["category"] == "attacked" for r in tr)),
                      "n_test": len(te_i), "vocab_size": res["info"]["vocab_size"], "kappa": res["info"]["kappa"]})
    # checks against the v0.3 code paths on the same population
    recs = [R[i] for i in idx]
    ch = {}
    for agn, d in [(False, "LR-C"), (True, "LR-B")]:
        ref, rf = r4_control.within_oof(recs, cfg["scheme"], rep, agn)
        ch[f"{d}_vs_r4_control.within_oof_max_abs_diff"] = float(np.max(np.abs(ref["score"] - S[d][idx])))
        ch[f"{d}_t_r80_pred_equal"] = bool(np.array_equal(ref["pred"]["t_r80"], P[d]["t_r80"][idx]))
        ch["folds_equal_r4_control"] = bool(np.array_equal(rf, folds[idx]))
    import ablation
    sA = np.full(len(idx), np.nan)
    for f in range(K):
        trf = [recs[i] for i in np.where(folds[idx] != f)[0]]; te_l = np.where(folds[idx] == f)[0]
        md = ablation.fit(trf, "A"); sA[te_l] = ablation.sco(md, [recs[i] for i in te_l], "A")["score"]
    ch["LR-A_vs_ablation.fit_max_abs_diff"] = float(np.max(np.abs(sA - S["LR-A"][idx])))
    return {"job": job, "unit": u, "rep": rep, "folds": folds, "scores": S, "pred": P, "thr": T, "cov": cov,
            "fold_info": finfo, "checks": ch}


def _unk_counts(recs, vocab, maxlen=16):
    steps_all = sum(sum(s["tool"] not in vocab for s in r["steps"]) for r in recs)
    steps_enc = sum(sum(s["tool"] not in vocab for s in r["steps"][-maxlen:]) for r in recs)
    traces = sum(any(s["tool"] not in vocab for s in r["steps"]) for r in recs)
    traces_pos = sum(any(s["tool"] not in vocab for s in r["steps"]) for r in recs if r["category"] == "attacked")
    traces_failed = sum(any(s["tool"] not in vocab for s in r["steps"]) for r in recs if r["category"] == "attack_failed")
    return {"unk_steps": int(steps_all), "unk_steps_encoded": int(steps_enc), "unk_traces": int(traces),
            "unk_traces_pos": int(traces_pos), "unk_traces_failed": int(traces_failed)}


def fit_gru_unit(job, u):
    import torch
    from gru_detector import train_gru, score_gru, build_tool_vocab
    from gru_ladder import stratified_val
    P0, FA, R = rows(); M = meta()
    cfg = JOBS[job]
    rep, fold = [int(x) for x in u.replace("rep", "").split("_fold")]
    folds = job_folds(job, rep)
    in_job = folds >= 0
    tr_mask = in_job & (folds != fold)
    if cfg["pop"] == "P0F" and not cfg["add_failed"]:
        tr_mask &= M["is_p0"]
    tr_i = np.where(tr_mask)[0]
    te_i = np.where(in_job & (folds == fold))[0]
    tr = [R[i] for i in tr_i]
    t0 = time.time()
    tr2, va2 = stratified_val(tr, rep)
    V0 = build_tool_vocab(tr2)
    unk_before = _unk_counts(tr2, V0)
    n_removed = 0
    if cfg["drop_unk"]:
        keep = [r for r in tr2 if all(s["tool"] in V0 for s in r["steps"])]
        n_removed = len(tr2) - len(keep); tr2 = keep
        assert build_tool_vocab(tr2) == V0
    model, voc, scl, thr = train_gru(tr2, va2, rep)
    fit_s = time.time() - t0
    s = np.full(M["n"], np.nan, np.float64)
    for part in (te_i[M["is_p0"][te_i]], te_i[~M["is_p0"][te_i]]):     # P0 rows in one pass, failed rows in another
        if len(part):
            s[part] = score_gru(model, voc, scl, thr, [R[i] for i in part])["score"]
    sv = score_gru(model, voc, scl, thr, va2)["score"]
    yv = np.array([r["category"] == "attacked" for r in va2])
    thr_val = float(np.quantile(sv[yv], 0.20)) if yv.any() else 0.5
    V = set(voc)
    cov_model = np.array([np.mean([st["tool"] in V for st in R[i]["steps"]]) for i in te_i])
    with torch.no_grad():
        unk_norm = float(model.emb.weight[1].norm())
    return {"job": job, "unit": u, "rep": rep, "fold": fold, "rows": te_i, "score": s[te_i], "thr_r80": float(thr),
            "thr_val80": thr_val, "cov_model": cov_model, "vocab": sorted(V), "n_fit": len(tr2), "n_val": len(va2),
            "n_fit_pos": int(sum(r["category"] == "attacked" for r in tr2)),
            "n_fit_failed": int(sum(r["category"] == "attack_failed" for r in tr2)),
            "unk_fit_before_removal": unk_before, "n_removed_unk": n_removed, "unk_fit": _unk_counts(tr2, V0),
            "unk_norm": unk_norm, "fit_seconds": fit_s}


def fit_gru_transfer_unit(job, u):
    from gru_detector import train_gru, score_gru
    from gru_ladder import stratified_val
    P0, FA, R = rows()
    sd = int(u[4:])
    pool = drift_pool()
    t0 = time.time()
    tr, va = stratified_val(pool, sd)
    m, voc, scl, thr = train_gru(tr, va, sd)
    A = v03_rows()
    sA = score_gru(m, voc, scl, thr, A)
    sP = score_gru(m, voc, scl, thr, P0)
    ref = cache("gru", "R4")["scores"]["GRU"][sd][:len(P0)]
    return {"job": job, "unit": u, "seed": sd, "v03_ids": [r["id"] for r in A], "v03_score": sA["score"],
            "v03_t_r80": sA["pred"]["t_r80"], "thr_r80": float(thr),
            "check_corrected_P0_max_abs_diff_vs_cache": float(np.max(np.abs(sP["score"] - ref))),
            "fit_seconds": time.time() - t0}


def fit_lin_v03_unit(job, u):
    import ablation
    A = v03_rows(); P0, FA, R = rows()
    pool = drift_pool()
    out = {"job": job, "unit": u, "v03_ids": [r["id"] for r in A]}
    if u == "transfer":
        lin = cache("linear", "R4")
        for rep, d in [("A", "LR-A"), ("B", "LR-B"), ("C", "LR-C")]:
            md = ablation.fit(pool, rep)
            sa = ablation.sco(md, A, rep); sp = ablation.sco(md, P0, rep)
            out[d] = {"score": sa["score"], "t_r80": sa["pred"]["t_r80"], "thr": float(md[3]),
                      "check_corrected_P0_max_abs_diff_vs_cache": float(np.max(np.abs(sp["score"] - lin["scores"][d][:len(P0)])))}
        return out
    scheme = "UT" if u.endswith("UT") else "IT"
    reps = []
    for sd in REPS:
        folds = subset_folds(A, scheme, sd)
        s = np.full(len(A), np.nan); p8 = np.zeros(len(A), bool)
        for f in range(K):
            trf = [A[i] for i in np.where(folds != f)[0]]; te = np.where(folds == f)[0]
            md = ablation.fit(trf, "A"); so = ablation.sco(md, [A[i] for i in te], "A")
            s[te] = so["score"]; p8[te] = so["pred"]["t_r80"]
        reps.append({"score": s, "t_r80": p8, "folds": folds})
    out["LR-A"] = reps
    return out


def run_fit(job, budget):
    t_start = time.time()
    kind = JOBS[job]["kind"]
    fn = {"lin": fit_lin_unit, "gru": fit_gru_unit, "gru_transfer": fit_gru_transfer_unit, "lin_v03": fit_lin_v03_unit}[kind]
    done = 0
    for u in units(job):
        p = unit_path(job, u)
        if os.path.exists(p):
            done += 1; continue
        if time.time() - t_start > budget:
            log(f"fit {job}: budget used, stopping before {u}")
            return False
        t0 = time.time()
        out = fn(job, u)
        out["seconds"] = round(time.time() - t0, 1)
        pdump(out, p); done += 1
        chk = out.get("checks", {k: v for k, v in out.items() if k.startswith("check")})
        log(f"fit {job} {u} done in {out['seconds']} s {chk if chk else ''}")
    return done == len(units(job))


def job_done(job):
    return all(os.path.exists(unit_path(job, u)) for u in units(job))


# =============================================================================================== score bank
def _as2d(a):
    a = np.asarray(a)
    return a[None, :] if a.ndim == 1 else a


def bank(setting, det):
    """{'S': (R, n) float scores over the 7,673 rows (NaN = not scored), 'p80': (R, n) bool, 'p05': (R, n) bool}."""
    k = ("bank", setting, det)
    if k in _D:
        return _D[k]
    M = meta(); n = M["n"]
    if setting in ("R4", "WUT", "WIT", "WUT_F") and not (det.startswith("GRU") and setting in ("WIT", "WUT_F")):
        fam = "gru" if det.startswith("GRU") else "linear"
        c = cache(fam, setting)
        out = {"S": _as2d(c["scores"][det]).astype(float), "p80": _as2d(c["pred"][det]["t_r80"]),
               "p05": _as2d(c["pred"][det]["t05"])}
        if "t_val80" in c["pred"][det]:
            out["pval80"] = _as2d(c["pred"][det]["t_val80"])
    else:
        job = {("WIT", "GRU"): "gru_WIT", ("WUT_F", "GRU"): "gru_WUT_F", ("C1_UT", "GRU"): "gru_C1_UT",
               ("C1_IT", "GRU"): "gru_C1_IT", ("WUT_noDoS", "GRU"): "gru_noDoS_UT",
               ("WUT_noUNK", "GRU"): "gru_WUT_noUNK"}.get((setting, det))
        if job is None:
            job = {"C1_UT": "lin_C1_UT", "C1_IT": "lin_C1_IT", "WUT_noDoS": "lin_noDoS_UT"}[setting]
        assert job_done(job), f"fit job {job} not finished"
        if JOBS[job]["kind"] == "lin":
            parts = [pload(unit_path(job, f"rep{r}")) for r in REPS]
            out = {"S": np.vstack([p["scores"][det] for p in parts]),
                   "p80": np.vstack([p["pred"][det]["t_r80"] for p in parts]),
                   "p05": np.vstack([p["pred"][det]["t05"] for p in parts])}
        else:
            S = np.full((len(REPS), n), np.nan); p80 = np.zeros((len(REPS), n), bool)
            pv = np.zeros((len(REPS), n), bool)
            for r in REPS:
                for f in range(K):
                    p = pload(unit_path(job, f"rep{r}_fold{f}"))
                    S[r, p["rows"]] = p["score"]; p80[r, p["rows"]] = p["score"] >= p["thr_r80"]
                    pv[r, p["rows"]] = p["score"] >= p["thr_val80"]
            out = {"S": S, "p80": p80, "p05": S >= 0.5, "pval80": pv}
    _D[k] = out
    return out


# =============================================================================================== metrics
def auc_ap(y, s, wpos=None, cs=(1.0,)):
    """AUROC (positives weighted by wpos; ties count 1/2) and average precision (sklearn definition, ties grouped)
    with positive weights c * wpos for each c in cs. y: 0/1 int array; s: float array."""
    yb = y.astype(bool)
    wp = np.where(yb, 1.0 if wpos is None else wpos, 0.0)
    wn = (~yb).astype(float)
    o = np.argsort(s, kind="mergesort"); ss = s[o]
    brk = np.empty(len(ss), bool); brk[0] = True; brk[1:] = ss[1:] != ss[:-1]
    bid = np.cumsum(brk) - 1; nb = int(bid[-1]) + 1
    Pb = np.bincount(bid, weights=wp[o], minlength=nb); Nb = np.bincount(bid, weights=wn[o], minlength=nb)
    totP = Pb.sum(); totN = Nb.sum()
    if totP == 0 or totN == 0:
        return float("nan"), [float("nan")] * len(cs)
    Nbelow = np.cumsum(Nb) - Nb
    auc = float(np.sum(Pb * (Nbelow + 0.5 * Nb)) / (totP * totN))
    tp = np.cumsum(Pb[::-1]); fp = np.cumsum(Nb[::-1])
    drec = np.diff(np.r_[0.0, tp / totP])
    aps = []
    for c in cs:
        with np.errstate(invalid="ignore", divide="ignore"):
            prec = np.where(c * tp + fp > 0, c * tp / (c * tp + fp), 0.0)
        aps.append(float(np.sum(drec * prec)))
    return auc, aps


def prev_scale(y, wpos, pi):
    """Scale c so that positives weighted c*wpos make up prevalence pi (plan 3.1)."""
    totP = float(np.sum(wpos[y == 1])) if wpos is not None else float(np.sum(y == 1))
    return pi / (1 - pi) * float(np.sum(y == 0)) / totP


def all_metrics(y, s, p80=None, wpos=None):
    """AUROC, AP (native), prevalence-weighted AP at 10% and 5% (and FPR / recall at theta_r80)."""
    if not (y == 1).any() or not (y == 0).any():      # a cluster draw without one of the classes
        nanm = {"AUROC": float("nan"), "AP_native": float("nan"), "APw10": float("nan"), "APw05": float("nan")}
        if p80 is not None:
            nanm["FPR@t_r80"] = float(p80[y == 0].mean()) if (y == 0).any() else float("nan")
            nanm["recall@t_r80"] = float("nan") if not (y == 1).any() else float(p80[y == 1].mean())
        return nanm
    nat =float(np.sum(y == 1)) / (np.sum(wpos[y == 1]) if wpos is not None else float(np.sum(y == 1)))
    cs = (nat, prev_scale(y, wpos, 0.10), prev_scale(y, wpos, 0.05))
    auc, aps = auc_ap(y, s, wpos, cs)
    m = {"AUROC": auc, "AP_native": aps[0], "APw10": aps[1], "APw05": aps[2]}
    if p80 is not None:
        neg = y == 0; pos = y == 1
        m["FPR@t_r80"] = float(p80[neg].mean()) if neg.any() else float("nan")
        if wpos is None:
            m["recall@t_r80"] = float(p80[pos].mean()) if pos.any() else float("nan")
        else:
            m["recall@t_r80"] = float(np.sum(wpos[pos] * p80[pos]) / np.sum(wpos[pos]))
    return m


def point_metrics(pop, y, meth, wpos=None):
    """Point estimate on the full population, mean over repetitions (ek.point convention)."""
    ms = [all_metrics(y[pop], meth["S"][r][pop], meth["p80"][r][pop], None if wpos is None else wpos[pop])
          for r in range(meth["S"].shape[0])]
    return {k: float(np.nanmean([m[k] for m in ms])) for k in ms[0]}


def ci3(a):
    a = np.asarray(a, float); a = a[~np.isnan(a)]
    if len(a) == 0:
        return [None, None, None]
    return [float(a.mean()), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def boot_p(d):
    d = np.asarray(d, float); d = d[~np.isnan(d)]
    return float(min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0))))


# ----------------------------------------------------------------------------------------- bootstraps
def strat_boot(pop, y, methods, B=B, seed=SEED_STRAT, n_pos=None, pos_weights=None, metrics=("AUROC",)):
    """Mirror of evalkit.bootstrap (identical random draws and repetition choice), with fast metrics.
    methods: {name: bank dict}. Returns {name: {metric: array(B)}}. metrics from all_metrics' keys."""
    rng = np.random.default_rng(seed)
    pool = np.asarray(pop)
    pos = pool[y[pool] == 1]; neg = pool[y[pool] == 0]
    npos = len(pos) if n_pos is None else n_pos; nneg = len(neg)
    pw = None
    if pos_weights is not None:
        pw = np.asarray([pos_weights[i] for i in pos], float); pw = pw / pw.sum()
    out = {m: {k: np.empty(B) for k in metrics} for m in methods}
    for b in range(B):
        ib = np.concatenate([rng.choice(pos, npos, replace=True, p=pw), rng.choice(neg, nneg, replace=True)])
        yb = y[ib]
        for m, meth in methods.items():
            r = rng.integers(meth["S"].shape[0])
            mm = all_metrics(yb, meth["S"][r][ib], meth["p80"][r][ib])
            for k in metrics:
                out[m][k][b] = mm[k]
    return out


class Clusters:
    """Cluster bootstrap design. joint: one set of clusters over all rows (UT: suite|user_task).
    two-level: positives resampled over positive clusters, negatives over negative clusters (IT)."""

    def __init__(self, pop, y, pos_keys, neg_keys=None):
        pop = np.asarray(pop)
        self.two = neg_keys is not None
        if not self.two:
            g = collections.defaultdict(list)
            for i in pop:
                g[pos_keys[i]].append(i)
            self.groups = [np.array(v) for _, v in sorted(g.items())]
        else:
            gp = collections.defaultdict(list); gn = collections.defaultdict(list)
            for i in pop:
                (gp[pos_keys[i]] if y[i] == 1 else gn[neg_keys[i]]).append(i)
            self.gpos = [np.array(v) for _, v in sorted(gp.items())]
            self.gneg = [np.array(v) for _, v in sorted(gn.items())]

    def n_clusters(self):
        return {"joint": len(self.groups)} if not self.two else {"pos": len(self.gpos), "neg": len(self.gneg)}

    def draw(self, rng):
        if not self.two:
            c = rng.integers(len(self.groups), size=len(self.groups))
            return np.concatenate([self.groups[j] for j in c])
        cp = rng.integers(len(self.gpos), size=len(self.gpos)); cn = rng.integers(len(self.gneg), size=len(self.gneg))
        return np.concatenate([self.gpos[j] for j in cp] + [self.gneg[j] for j in cn])


def clus_boot(y, methods, design, B=B, seed=SEED_CLUSTER, wpos=None, metrics=("AUROC", "APw10", "APw05")):
    """Cluster bootstrap; each method draws its own repetition per replicate (evalkit convention).
    wpos: fixed positive weights over all rows (model-balanced), or None."""
    rng = np.random.default_rng(seed)
    out = {m: {k: np.empty(B) for k in metrics} for m in methods}
    for b in range(B):
        ib = design.draw(rng); yb = y[ib]
        wb = None if wpos is None else wpos[ib]
        for m, meth in methods.items():
            r = rng.integers(meth["S"].shape[0])
            mm = all_metrics(yb, meth["S"][r][ib], meth["p80"][r][ib], wb)
            for k in metrics:
                out[m][k][b] = mm[k]
    return out


def summarize_boot(bt, names, metrics, pairs=()):
    res = {m: {k: ci3(bt[m][k]) for k in metrics} for m in names}
    pr = {}
    for a, b_ in pairs:
        pr[f"{a}-minus-{b_}"] = {}
        for k in metrics:
            d = bt[a][k] - bt[b_][k]
            mu, lo, hi = ci3(d)
            pr[f"{a}-minus-{b_}"][k] = {"mean": mu, "lo": lo, "hi": hi, "p_boot": boot_p(d),
                                        "frac_le0": float(np.mean(d[~np.isnan(d)] <= 0))}
    return res, pr


def holm(pvals):
    """Holm step-down; returns adjusted p-values in input order."""
    p = np.asarray(pvals, float); order = np.argsort(p); m = len(p)
    adj = np.empty(m); run = 0.0
    for k, i in enumerate(order):
        run = max(run, (m - k) * p[i]); adj[i] = min(1.0, run)
    return adj


# =============================================================================================== populations
def mb_weights(pop):
    """r4_control.py model-balanced weights: positives reweighted so their model mix equals the benign model mix,
    over the models with >= 50 benign and >= 50 attacked traces in the population. Returns (pool, dict, array)."""
    M = meta(); y = M["y"]; models = M["model"]
    cnt = collections.Counter((models[i], int(y[i])) for i in pop)
    elig = sorted({m for m in set(models[pop]) if cnt[(m, 0)] >= 50 and cnt[(m, 1)] >= 50})
    pool = np.array([i for i in pop if models[i] in elig])
    bshare = collections.Counter(models[i] for i in pool if y[i] == 0)
    ashare = collections.Counter(models[i] for i in pool if y[i] == 1)
    nb, na = sum(bshare.values()), sum(ashare.values())
    w = {int(i): (bshare[models[i]] / nb) / (ashare[models[i]] / na) for i in pool if y[i] == 1}
    warr = np.ones(M["n"]);
    for i, v in w.items():
        warr[i] = v
    return pool, w, warr, elig


def population(name):
    """(index array, label array (global), description)."""
    M = meta(); y = M["y"].copy(); nP = M["nP0"]
    P0 = np.arange(nP)
    if name == "P0":
        return P0, y, "AgentDojo P0: 392 benign negatives, 2,294 compromised positives"
    if name == "C1":
        return P0[~M["inj_benign"][:nP]], y, "C1: P0 minus the 65 benign runs whose user task is an injection task"
    if name == "P0F":
        return np.arange(M["n"]), y, "P0 plus the 4,987 unsuccessful attacks as additional negatives"
    if name == "P0_noDoS":
        return P0[~(M["is_dos"][:nP] & (y[:nP] == 1))], y, "P0 without the 101 DoS positives"
    if name == "P0F_noDoS":
        allr = np.arange(M["n"])
        return allr[~M["is_dos"]], y, "P0F without every DoS row (101 positives, 252 unsuccessful DoS runs)"
    if name == "ATTEMPT":
        ya = y.copy(); ya[M["is_failed"]] = 1
        return np.arange(M["n"]), ya, ("attempt target (exploratory, not in the plan): unsuccessful attacks relabelled "
                                        "as positives; negatives = 392 benign")
    raise KeyError(name)


def cluster_design(pop, y, scheme):
    M = meta()
    if scheme == "UT":
        return Clusters(pop, y, M["ut_key"])
    # IT: positives by suite|injection_task, negatives by suite|user_task
    return Clusters(pop, y, M["it_key"], M["ut_key"])


# =============================================================================================== sections
def sec_path(name):
    return os.path.join(SECS, name + ".json")


def eval_config(pop_name, within_setting, dets, scheme, transfer_setting="R4", mb=False, subsample=True,
                B_=B, extra_methods=None, pos_override=None):
    """Transfer vs within for `dets` on one population. Returns point estimates, v0.3-style stratified bootstrap
    (evalkit mirror, native AUROC; subsample AP@10/5), plan-style cluster bootstrap (AUROC, prevalence-weighted AP),
    and paired within-minus-transfer deltas under both."""
    M = meta()
    pop, y, desc = population(pop_name)
    if pos_override is not None:
        y = pos_override
    wdict = None; warr = None; elig = None
    if mb:
        pop, wdict, warr, elig = mb_weights(pop)
    methods = {}
    for d in dets:
        methods[f"transfer:{d}"] = bank(transfer_setting, d)
        if within_setting is not None:
            methods[f"within:{d}"] = bank(within_setting, d)
    if extra_methods:
        methods.update(extra_methods)
    for m, meth in methods.items():
        assert not np.isnan(meth["S"][:, pop]).any(), f"{m}: unscored rows in population {pop_name}"
    names = list(methods)
    pairs = [(f"within:{d}", f"transfer:{d}") for d in dets] if within_setting is not None else []
    out = {"population": pop_name, "description": desc, "within_setting": within_setting, "scheme": scheme,
           "n_pos": int(y[pop].sum()), "n_neg": int((y[pop] == 0).sum()), "model_balanced": mb,
           "eligible_models": elig}
    # ---- point estimates
    out["point"] = {m: {k: r4(v) for k, v in point_metrics(pop, y, methods[m], warr if mb else None).items()}
                    for m in names}
    # ---- v0.3-style stratified bootstrap at native counts (evalkit mirror)
    mets = ("AUROC", "AP_native", "APw10", "APw05", "FPR@t_r80", "recall@t_r80")
    bt = strat_boot(pop, y, methods, B=B_, seed=SEED_STRAT, pos_weights=wdict, metrics=mets)
    s, pr = summarize_boot(bt, names, mets, pairs)
    out["strat_native"] = {"seed": SEED_STRAT, "B": B_, "ci": s, "paired": pr}
    if subsample:
        nneg = int((y[pop] == 0).sum())
        out["strat_subsample"] = {}
        for pi, lab in [(0.10, "p10"), (0.05, "p05")]:
            npos = int(round(pi / (1 - pi) * nneg))
            bt2 = strat_boot(pop, y, methods, B=B_, seed=SEED_STRAT, n_pos=npos, pos_weights=wdict,
                             metrics=("AUROC", "AP_native"))
            s2, pr2 = summarize_boot(bt2, names, ("AUROC", "AP_native"), pairs)
            out["strat_subsample"][lab] = {"n_pos": npos, "n_neg": nneg, "ci": s2, "paired": pr2,
                                           "note": "v0.3 secondary estimator: AP of a test set with n_pos positives "
                                                   "(estimate = replicate mean)"}
    # ---- plan-style cluster bootstrap
    des = cluster_design(pop, y, scheme)
    bt3 = clus_boot(y, methods, des, B=B_, seed=SEED_CLUSTER, wpos=warr if mb else None,
                    metrics=("AUROC", "APw10", "APw05", "FPR@t_r80", "recall@t_r80"))
    s3, pr3 = summarize_boot(bt3, names, ("AUROC", "APw10", "APw05", "FPR@t_r80", "recall@t_r80"), pairs)
    out["cluster"] = {"seed": SEED_CLUSTER, "B": B_, "scheme": scheme, "n_clusters": des.n_clusters(),
                      "ci": s3, "paired": pr3}
    return out


def reasonable(auc, auc_lo, ap10, ap10_lo):
    c = {"AUROC>=0.80": auc >= 0.80, "AUROC_lo>=0.70": auc_lo >= 0.70, "AP10>=0.30": ap10 >= 0.30,
         "AP10_lo>0.10": ap10_lo > 0.10}
    return all(c.values()), c


def near_chance(auc, lo, hi):
    return bool((lo <= 0.5 <= hi) or auc < 0.60)


def decide(cfg, d, procedure):
    """Pre-registered criteria (R4_CONTROL.md Sec. 7) for detector d in one evaluated configuration."""
    P = cfg["point"]; w = f"within:{d}"; t = f"transfer:{d}"
    if procedure == "v03":
        ci = cfg["strat_native"]["ci"]; sub = cfg["strat_subsample"]["p10"]["ci"]
        auc_w = P[w]["AUROC"] if not cfg["model_balanced"] else ci[w]["AUROC"][0]
        auc_t = P[t]["AUROC"] if not cfg["model_balanced"] else ci[t]["AUROC"][0]
        ap10, ap10_lo = sub[w]["AP_native"][0], sub[w]["AP_native"][1]
        lo_w = ci[w]["AUROC"][1]; lo_t, hi_t = ci[t]["AUROC"][1], ci[t]["AUROC"][2]
        dl = cfg["strat_native"]["paired"][f"{w}-minus-{t}"]["AUROC"]
    else:
        ci = cfg["cluster"]["ci"]
        auc_w, auc_t = P[w]["AUROC"], P[t]["AUROC"]
        ap10, ap10_lo = P[w]["APw10"], ci[w]["APw10"][1]
        lo_w = ci[w]["AUROC"][1]; lo_t, hi_t = ci[t]["AUROC"][1], ci[t]["AUROC"][2]
        dl = cfg["cluster"]["paired"][f"{w}-minus-{t}"]["AUROC"]
    ok, crit = reasonable(auc_w, lo_w, ap10, ap10_lo)
    return {"within_AUROC": r4(auc_w), "within_AUROC_lo": r4(lo_w), "within_AP10": r4(ap10), "within_AP10_lo": r4(ap10_lo),
            "criteria": crit, "within_reasonable": bool(ok),
            "transfer_AUROC": r4(auc_t), "transfer_ci": [r4(lo_t), r4(hi_t)],
            "transfer_near_chance": near_chance(auc_t, lo_t, hi_t),
            "delta_AUROC": [r4(dl["mean"]), r4(dl["lo"]), r4(dl["hi"])], "delta_p_boot": dl["p_boot"],
            "delta_excludes_0": bool(dl["lo"] > 0 or dl["hi"] < 0)}


CASE_CONFIGS = {
    # name: (population, within setting, scheme, model_balanced)
    "P0_UT": ("P0", "WUT", "UT", False),
    "P0_IT": ("P0", "WIT", "IT", False),
    "C1_UT": ("C1", "C1_UT", "UT", False),
    "C1_IT": ("C1", "C1_IT", "IT", False),
    "MB_UT": ("P0", "WUT", "UT", True),
    "MB_IT": ("P0", "WIT", "IT", True),
    "P0F_UT": ("P0F", "WUT", "UT", False),
    "P0F_IT": ("P0F", "WIT", "IT", False),
    "WUTF_P0": ("P0", "WUT_F", "UT", False),
    "WUTF_P0F": ("P0F", "WUT_F", "UT", False),
    "ATTEMPT_UT": ("ATTEMPT", "WUT", "UT", False),
    "ATTEMPT_IT": ("ATTEMPT", "WIT", "IT", False),
}


def section_case(budget):
    """M6: per-configuration evaluation of the five representations (saved one file per configuration)."""
    t0 = time.time(); done = True
    for name, (pop, ws, scheme, mb) in CASE_CONFIGS.items():
        p = sec_path(f"case__{name}")
        if os.path.exists(p):
            continue
        if time.time() - t0 > budget:
            done = False; break
        t1 = time.time()
        y_over = population(pop)[1] if pop == "ATTEMPT" else None
        cfg = eval_config(pop, ws, MAIN5, scheme, mb=mb, pos_override=y_over)
        cfg["seconds"] = round(time.time() - t1, 1)
        jdump(cfg, p); log(f"case {name} done in {cfg['seconds']} s")
    return done


def _per_model_reasonable(pm, d, procedure):
    """Within-model UT strata of R4_CONTROL.md Sec. 1 (models with >= 50 benign and >= 50 attacked traces).
    plan: point AUROC / prevalence-weighted AP@10% with suite|user_task cluster lower bounds;
    v03: the same point values with trajectory-level stratified lower bounds (no subsample AP exists per model)."""
    out = {}
    for m, e in pm.items():
        if e.get("within_UT_reasonable_cluster") is None:
            continue
        k = f"within_UT:{d}"
        ci = e["cluster_UT"]["ci"] if procedure == "plan" else e["strat"]["ci"]
        ok, crit = reasonable(e["point"][k]["AUROC"], ci[k]["AUROC"][1], e["point"][k]["APw10"], ci[k]["APw10"][1])
        out[m] = {"reasonable": bool(ok), "AUROC": r4(e["point"][k]["AUROC"]), "AUROC_lo": r4(ci[k]["AUROC"][1]),
                  "AP10": r4(e["point"][k]["APw10"]), "AP10_lo": r4(ci[k]["APw10"][1])}
    return out


def _pattern(x):
    if x["within_reasonable"] and x["transfer_near_chance"] and x["delta_excludes_0"]:
        return "A-pattern (within reasonable, transfer near chance, delta detected)"
    if (not x["within_reasonable"]) and x["transfer_near_chance"]:
        return "B-pattern (within not reasonable, transfer near chance)"
    if x["within_reasonable"] and not x["transfer_near_chance"]:
        return "within reasonable, transfer NOT near chance"
    return "other (within not reasonable, transfer not near chance)"


def derive_cases():
    """Apply R4_CONTROL.md Sec. 7 per representation, under both procedures.
    Case A: within UT reasonable for P0 and for C1, transfer near chance (P0 and C1), paired delta excludes 0 (P0, C1).
    Case B: within UT not reasonable for P0 or for C1, and transfer near chance.
    Case C: within UT reasonable only in controlled populations or strata (C1, within-model, model-balanced), not P0.
    Mixed clause: reported as observed; UT reasonable but IT not reasonable is flagged."""
    cfgs = {n: json.load(open(sec_path(f"case__{n}"))) for n in CASE_CONFIGS}
    pm = json.load(open(sec_path("per_model")))
    res = {}
    for proc in ("plan", "v03"):
        res[proc] = {}
        for d in MAIN5:
            dd = {n: decide(cfgs[n], d, proc) for n in CASE_CONFIGS}
            for x in dd.values():
                x["near_chance_only_via_below_0.60"] = bool(x["transfer_near_chance"] and
                                                             not (x["transfer_ci"][0] <= 0.5 <= x["transfer_ci"][1]))
            ut_p0, ut_c1 = dd["P0_UT"], dd["C1_UT"]
            strata = _per_model_reasonable(pm, d, proc)
            nc = ut_p0["transfer_near_chance"] and ut_c1["transfer_near_chance"]
            A = ut_p0["within_reasonable"] and ut_c1["within_reasonable"] and nc and                 ut_p0["delta_excludes_0"] and ut_c1["delta_excludes_0"]
            Bc = (not ut_p0["within_reasonable"] or not ut_c1["within_reasonable"]) and nc
            ctrl = [n for n in ("C1_UT", "MB_UT") if dd[n]["within_reasonable"]] +                    [f"model:{m}" for m, v in strata.items() if v["reasonable"]]
            Cc = (not ut_p0["within_reasonable"]) and len(ctrl) > 0
            it_ok = dd["P0_IT"]["within_reasonable"]
            it_txt = "IT reasonable" if it_ok else "IT not reasonable"
            if A:
                case = "Case A under UT" + ("" if it_ok else "; mixed: UT reasonable but IT not reasonable (seen attack "
                                                            "goals only)")
            elif Bc and Cc:
                case = f"mixed: Case B and Case C conditions both hold ({it_txt})"
            elif Cc:
                case = f"Case C ({it_txt})"
            elif Bc:
                case = f"Case B ({it_txt})"
            else:
                case = f"other, reported as observed ({it_txt})"
            alt = {n: _pattern(dd[n]) for n in CASE_CONFIGS}
            res[proc][d] = {"case_literal": case, "conditions": {"A": bool(A), "B": bool(Bc), "C": bool(Cc),
                                                                  "C_controls_reasonable": ctrl,
                                                                  "IT_P0_reasonable": bool(it_ok),
                                                                  "IT_C1_reasonable": bool(dd["C1_IT"]["within_reasonable"])},
                            "per_config": dd, "within_model_strata_UT": strata, "per_population_reading": alt}
    res["rule_text"] = {
        "reasonable": "AUROC >= 0.80, lower 95% bound of AUROC >= 0.70, AUPRC@10% >= 0.30, lower bound of AUPRC@10% > 0.10",
        "near_chance": "AUROC CI includes 0.5, or AUROC < 0.60 (one-sided clause written after the transfer AUROC was known, "
                       "review M6(d); flagged per configuration as near_chance_only_via_below_0.60)",
        "plan_procedure": "point estimates on the full population (mean over CV repetitions; model-balanced = weighted point), "
                          "95% cluster bootstrap (UT: suite|user_task; IT: positives suite|injection_task, negatives "
                          "suite|user_task), AUPRC@10% = prevalence-weighted AP (plan 3.1)",
        "v03_procedure": "v0.3 r4_control.py conventions: trajectory-level stratified bootstrap (seed 12345, B = 2000), "
                         "AUPRC@10% = subsample estimator (mean and 2.5th percentile of AP over replicates with "
                         "n_pos = round(0.1/0.9 n_neg)); model-balanced AUROC = bootstrap mean",
        "C1_transfer": "Case A/B also require near-chance transfer on C1 (R4_CONTROL.md says 'the transfer'; P0 and C1 agree "
                       "for every detector here)"}
    hol = {}
    for n in CASE_CONFIGS:
        ps = [cfgs[n]["cluster"]["paired"][f"within:{d}-minus-transfer:{d}"]["AUROC"]["p_boot"] for d in MAIN5]
        adj = holm(ps)
        hol[n] = {d: {"p_boot": ps[i], "p_holm": float(adj[i]), "detected_holm": bool(adj[i] < 0.05)}
                  for i, d in enumerate(MAIN5)}
    res["holm_within_minus_transfer_cluster"] = {"family": "five representations per configuration (plan 3.6 d)",
                                                 "note": "bootstrap p = 2 min(P(d<=0), P(d>=0)); 0 means no replicate "
                                                         "on the other side (p < 1/B = 0.0005)", "by_config": hol}
    return res


# ----------------------------------------------------------------------------------------- per-model strata
def section_per_model(budget):
    """Within-model evaluation (Case C strata; M7 per-model transfer), all detectors of MAIN5 + all linear."""
    M = meta(); y = M["y"]
    P0 = np.arange(M["nP0"])
    models = sorted(set(M["model"][P0]))
    dets = MAIN5
    res = {}
    for m in models:
        pool = P0[M["model"][P0] == m]
        npos, nneg = int(y[pool].sum()), int((y[pool] == 0).sum())
        entry = {"n_pos": npos, "n_neg": nneg}
        if npos < 2 or nneg < 2:
            res[m] = entry; continue
        meths = {}
        for d in dets:
            meths[f"transfer:{d}"] = bank("R4", d); meths[f"within_UT:{d}"] = bank("WUT", d)
            meths[f"within_IT:{d}"] = bank("WIT", d)
        names = list(meths)
        entry["point"] = {k: {kk: r4(vv) for kk, vv in point_metrics(pool, y, meths[k]).items()} for k in names}
        bt = strat_boot(pool, y, meths, seed=SEED_STRAT, metrics=("AUROC", "APw10"))
        pairs = [(f"within_UT:{d}", f"transfer:{d}") for d in dets] + [(f"within_IT:{d}", f"transfer:{d}") for d in dets]
        s, pr = summarize_boot(bt, names, ("AUROC", "APw10"), pairs)
        entry["strat"] = {"ci": s, "paired": pr}
        des = Clusters(pool, y, M["ut_key"])
        bt2 = clus_boot(y, meths, des, seed=SEED_CLUSTER, metrics=("AUROC", "APw10"))
        s2, pr2 = summarize_boot(bt2, names, ("AUROC", "APw10"), pairs)
        entry["cluster_UT"] = {"ci": s2, "paired": pr2, "n_clusters": des.n_clusters()}
        entry["within_UT_reasonable_cluster"] = {
            d: reasonable(entry["point"][f"within_UT:{d}"]["AUROC"], s2[f"within_UT:{d}"]["AUROC"][1],
                          entry["point"][f"within_UT:{d}"]["APw10"], s2[f"within_UT:{d}"]["APw10"][1])[0]
            for d in dets} if npos >= 50 and nneg >= 50 else None
        res[m] = entry
    jdump(res, sec_path("per_model"))
    return True


# ----------------------------------------------------------------------------------------- all-detector tables
ALL_TRANSFER = LIN_DETS + ["GRU", "GRU-drop0.2", "GRU-drop0.1"]
ALL_WITHIN = LIN_DETS + ["GRU"]


def all_det_methods(within_sets=("WUT", "WIT")):
    meths = {}
    for d in ALL_TRANSFER:
        meths[f"transfer:{d}"] = bank("R4", d)
    for ws in within_sets:
        for d in ALL_WITHIN:
            meths[f"{ws}:{d}"] = bank(ws, d)
    return meths


def section_alldet(budget):
    """m1, m2, m17.3: all detectors on P0. Stratified (v0.3) and cluster bootstraps (UT and IT designs), model-balanced
    at native counts (r4_control weighting). One file per sub-analysis."""
    M = meta(); y = M["y"]
    P0 = np.arange(M["nP0"])
    meths = all_det_methods()
    names = list(meths)
    pairs_ut = [(f"WUT:{d}", f"transfer:{d}") for d in ALL_WITHIN]
    pairs_it = [(f"WIT:{d}", f"transfer:{d}") for d in ALL_WITHIN]
    pairs_extra = [("transfer:LR-A", "transfer:LR-C"), ("transfer:GRU", "transfer:LR-C"), ("WUT:LR-A", "WUT:LR-C"),
                   ("WUT:GRU", "WUT:LR-C"), ("WIT:GRU", "WIT:LR-C")]
    mets = ("AUROC", "APw10", "APw05", "FPR@t_r80", "recall@t_r80")
    t0 = time.time(); done = True
    jobs = ["alldet_point", "alldet_strat", "alldet_cluster_UT", "alldet_cluster_IT", "alldet_mb_strat",
            "alldet_mb_cluster_UT", "alldet_mb_cluster_IT", "alldet_sub"]
    for j in jobs:
        p = sec_path(j)
        if os.path.exists(p):
            continue
        if time.time() - t0 > budget:
            done = False; break
        t1 = time.time()
        if j == "alldet_point":
            out = {"P0": {m: {k: r4(v) for k, v in point_metrics(P0, y, meths[m]).items()} for m in names}}
            pool, wd, warr, elig = mb_weights(P0)
            out["MB_weighted_point"] = {m: {k: r4(v) for k, v in point_metrics(pool, y, meths[m], warr).items()}
                                        for m in names}
            out["MB_eligible"] = elig; out["MB_n_pos"] = int(y[pool].sum()); out["MB_n_neg"] = int((y[pool] == 0).sum())
            out["MB_weights_by_model"] = {m: r4(warr[pool][(M["model"][pool] == m) & (y[pool] == 1)][0], 6) for m in elig}
        elif j == "alldet_strat":
            bt = strat_boot(P0, y, meths, seed=SEED_STRAT, metrics=mets)
            s, pr = summarize_boot(bt, names, mets, pairs_ut + pairs_it + pairs_extra)
            out = {"ci": s, "paired": pr, "seed": SEED_STRAT, "B": B}
        elif j in ("alldet_cluster_UT", "alldet_cluster_IT"):
            sch = j[-2:]
            des = cluster_design(P0, y, sch)
            bt = clus_boot(y, meths, des, seed=SEED_CLUSTER, metrics=mets)
            s, pr = summarize_boot(bt, names, mets, pairs_ut + pairs_it + pairs_extra)
            fam_ut = [pr[f"WUT:{d}-minus-transfer:{d}"]["AUROC"]["p_boot"] for d in ALL_WITHIN]
            fam_it = [pr[f"WIT:{d}-minus-transfer:{d}"]["AUROC"]["p_boot"] for d in ALL_WITHIN]
            out = {"ci": s, "paired": pr, "seed": SEED_CLUSTER, "B": B, "n_clusters": des.n_clusters(),
                   "holm_WUT_minus_transfer": dict(zip(ALL_WITHIN, map(float, holm(fam_ut)))),
                   "holm_WIT_minus_transfer": dict(zip(ALL_WITHIN, map(float, holm(fam_it))))}
        elif j == "alldet_mb_strat":
            pool, wd, warr, elig = mb_weights(P0)
            bt = strat_boot(pool, y, meths, seed=SEED_STRAT, pos_weights=wd, metrics=("AUROC", "APw10", "APw05"))
            s, pr = summarize_boot(bt, names, ("AUROC", "APw10", "APw05"), pairs_ut + pairs_it + pairs_extra)
            nneg = int((y[pool] == 0).sum()); npos10 = int(round(0.1 / 0.9 * nneg))
            bt2 = strat_boot(pool, y, meths, seed=SEED_STRAT, n_pos=npos10, pos_weights=wd, metrics=("AUROC", "AP_native"))
            s2, pr2 = summarize_boot(bt2, names, ("AUROC", "AP_native"), pairs_ut + pairs_it + pairs_extra)
            out = {"native_counts": {"ci": s, "paired": pr, "n_pos": int(y[pool].sum()), "n_neg": nneg,
                                     "note": "r4_control.py convention: positives drawn with model-balance weights, "
                                             "estimate = bootstrap mean"},
                   "p10_subsample": {"ci": s2, "paired": pr2, "n_pos": npos10, "n_neg": nneg}, "seed": SEED_STRAT}
        elif j in ("alldet_mb_cluster_UT", "alldet_mb_cluster_IT"):
            sch = j[-2:]
            pool, wd, warr, elig = mb_weights(P0)
            des = cluster_design(pool, y, sch)
            bt = clus_boot(y, meths, des, seed=SEED_CLUSTER, wpos=warr, metrics=("AUROC", "APw10", "APw05"))
            s, pr = summarize_boot(bt, names, ("AUROC", "APw10", "APw05"), pairs_ut + pairs_it + pairs_extra)
            out = {"ci": s, "paired": pr, "n_clusters": des.n_clusters(),
                   "note": "weighted AUROC / AP with fixed full-pool model-balance weights; clusters resampled"}
        elif j == "alldet_sub":
            nneg = int((y[P0] == 0).sum()); out = {}
            for pi, lab in [(0.10, "p10"), (0.05, "p05")]:
                npos = int(round(pi / (1 - pi) * nneg))
                bt = strat_boot(P0, y, meths, seed=SEED_STRAT, n_pos=npos, metrics=("AP_native",))
                s, pr = summarize_boot(bt, names, ("AP_native",), pairs_ut + pairs_it)
                out[lab] = {"n_pos": npos, "n_neg": nneg, "ci": s, "paired": pr}
        out["seconds"] = round(time.time() - t1, 1)
        jdump(out, p); log(f"{j} done in {out['seconds']} s")
    return done


# ----------------------------------------------------------------------------------------- failed-attack flag rates
def section_flags(budget):
    """Flag rates of the 4,987 unsuccessful attacks at theta_r80 (and 0.5), every detector, transfer and within;
    benign FPR and compromised recall alongside. Trajectory-level and suite|user_task cluster bootstraps."""
    M = meta()
    F = np.where(M["is_failed"])[0]
    P0 = np.arange(M["nP0"])
    ben = P0[M["y"][P0] == 0]
    sets = {"R4": ALL_TRANSFER, "WUT": ALL_WITHIN, "WIT": ALL_WITHIN, "WUT_F": ALL_WITHIN}
    rng_seed = SEED_FAILED
    groups = collections.defaultdict(list)
    for i in F:
        groups[M["ut_key"][i]].append(i)
    garr = [np.array(v) for _, v in sorted(groups.items())]
    out = {"n_failed": len(F), "n_failed_clusters": len(garr), "by_setting": {}}
    for st, dets in sets.items():
        out["by_setting"][st] = {}
        for d in dets:
            b_ = bank(st, d)
            R = b_["S"].shape[0]
            fr = [float(b_["p80"][r][F].mean()) for r in range(R)]
            fr05 = [float(b_["p05"][r][F].mean()) for r in range(R)]
            fpr = [float(b_["p80"][r][ben].mean()) for r in range(R)]
            rec = [float(b_["p80"][r][P0[M["y"][P0] == 1]].mean()) for r in range(R)]
            rng = np.random.default_rng(rng_seed)
            bt = np.empty(B); btc = np.empty(B)
            for b in range(B):
                ib = rng.choice(F, len(F), replace=True); r = rng.integers(R)
                bt[b] = b_["p80"][r][ib].mean()
                c = rng.integers(len(garr), size=len(garr)); ic = np.concatenate([garr[j] for j in c])
                btc[b] = b_["p80"][rng.integers(R)][ic].mean()
            e = {"failed_flag_rate@t_r80": r4(np.mean(fr)), "ci_traj": [r4(x) for x in ci3(bt)[1:]],
                 "ci_cluster_UT": [r4(x) for x in ci3(btc)[1:]], "per_rep": [r4(x) for x in fr],
                 "failed_flag_rate@t05": r4(np.mean(fr05)), "benign_FPR@t_r80": r4(np.mean(fpr)),
                 "compromised_recall@t_r80": r4(np.mean(rec))}
            if "pval80" in b_:
                e["failed_flag_rate@t_val80"] = r4(np.mean([b_["pval80"][r][F].mean() for r in range(R)]))
            # by DoS / non-DoS and by model
            e["by_is_dos"] = {str(k): r4(np.mean([b_["p80"][r][F[M["is_dos"][F] == k]].mean() for r in range(R)]))
                              for k in (False, True)}
            e["by_model"] = {m: r4(np.mean([b_["p80"][r][F[M["model"][F] == m]].mean() for r in range(R)]))
                             for m in sorted(set(M["model"][F]))}
            out["by_setting"][st][d] = e
    jdump(out, sec_path("flags"))
    return True


# ----------------------------------------------------------------------------------------- M7: v0.3 vs corrected
def _arch(name):
    return json.load(open(os.path.join(ARCH, name), encoding="utf-8"))


def _cur(name):
    return json.load(open(os.path.join(ROOT, name), encoding="utf-8"))


def _v03_bank():
    """Scores on the ARCHIVED v0.3 extraction (recomputed; row order of archive_v0.3/agentdojo_traces.jsonl)."""
    A = v03_rows()
    y = np.array([int(r["category"] == "attacked") for r in A])
    lt = pload(unit_path("lin_v03", "transfer"))
    out = {}
    for d in ("LR-A", "LR-B", "LR-C"):
        out[f"transfer:{d}"] = {"S": np.asarray(lt[d]["score"], float)[None, :], "p80": np.asarray(lt[d]["t_r80"])[None, :]}
    g = [pload(unit_path("gru_v03_transfer", f"seed{s}")) for s in (0, 1, 2)]
    assert all(p["v03_ids"] == [r["id"] for r in A] for p in g)
    out["transfer:GRU"] = {"S": np.vstack([p["v03_score"] for p in g]).astype(float),
                           "p80": np.vstack([p["v03_t_r80"] for p in g])}
    for u, key in (("withinA_UT", "WUT:LR-A"), ("withinA_IT", "WIT:LR-A")):
        w = pload(unit_path("lin_v03", u))
        out[key] = {"S": np.vstack([r["score"] for r in w["LR-A"]]), "p80": np.vstack([r["t_r80"] for r in w["LR-A"]])}
    models = np.array([r["id"].split("|")[0] for r in A], object)
    chk = {"lin_transfer_corrected_P0_max_abs_diff_vs_cache": {d: lt[d]["check_corrected_P0_max_abs_diff_vs_cache"]
                                                               for d in ("LR-A", "LR-B", "LR-C")},
           "gru_transfer_corrected_P0_max_abs_diff_vs_cache": [p["check_corrected_P0_max_abs_diff_vs_cache"] for p in g]}
    return A, y, models, out, chk


def _mirror(y, pop, spec, seed=SEED_STRAT, n_pos=None, pos_weights=None,
            metrics=("AUROC", "AP_native", "FPR@t_r80", "recall@t_r80")):
    """spec: ordered list of (name, bank dict) exactly in the v0.3 script's method order."""
    methods = collections.OrderedDict(spec)
    bt = strat_boot(pop, y, methods, seed=seed, n_pos=n_pos, pos_weights=pos_weights, metrics=metrics)
    return methods, bt


def _cmp(ours, ref):
    return float(max(abs(a - b) for a, b in zip(ours, ref)))


def _tri(pop, y, meth, bt_m, k, w=None):
    return [r4(point_metrics(pop, y, meth, w)[k])] + [r4(x) for x in ci3(bt_m[k])[1:]]


def section_m7(budget):
    M = meta(); y = M["y"]; P0 = np.arange(M["nP0"])
    out = {"notes": []}
    rc = _cur("r4_control_results.json"); ab = _cur("ablation_results.json"); gr = _cur("gru_results.json")
    rerun_final = "gru_ladder exit 0" in open(os.path.join(HERE, "rerun_v03_scripts.log"), encoding="utf-8").read()
    out["rerun_final"] = rerun_final
    chk = {}
    corr = {}
    # r4_control P0 native (method order of r4_control.py)
    spec = [("transfer_full", bank("R4", "LR-C")), ("transfer_agn", bank("R4", "LR-B")),
            ("within_UT_full", bank("WUT", "LR-C")), ("within_UT_agn", bank("WUT", "LR-B")),
            ("within_IT_full", bank("WIT", "LR-C")), ("within_IT_agn", bank("WIT", "LR-B"))]
    meths, bt = _mirror(y, P0, spec)
    r4c = {m: {k: _tri(P0, y, meths[m], bt[m], k) for k in ("AUROC", "AP_native", "FPR@t_r80", "recall@t_r80")} for m in meths}
    pr = {f"{a}-minus-{b}": [r4(v) for v in ci3(bt[a]["AUROC"] - bt[b]["AUROC"])]
          for a, b in [("within_UT_full", "transfer_full"), ("within_UT_agn", "transfer_agn"),
                       ("within_IT_full", "transfer_full"), ("within_IT_agn", "transfer_agn")]}
    ref = rc["P0"]["native"]["metrics"]
    chk["r4_control_P0_native_max_abs_diff"] = {
        m: {k: _cmp(r4c[m][kk], ref[m][k]) for k, kk in (("AUROC", "AUROC"), ("AUPRC", "AP_native"),
                                                           ("FPR@t_r80", "FPR@t_r80"), ("recall@t_r80", "recall@t_r80"))}
        for m in meths}
    chk["r4_control_P0_paired_AUROC_max_abs_diff"] = {
        k: _cmp(pr[k], [rc["P0"]["native"]["paired"][k]["AUROC"][x] for x in ("delta", "lo", "hi")]) for k in pr}
    corr["r4_control_mirror"] = {"metrics": r4c, "paired_AUROC": pr}
    # ablation R4 P0 native
    spec = [("transfer_A", bank("R4", "LR-A")), ("transfer_B", bank("R4", "LR-B")), ("transfer_C", bank("R4", "LR-C")),
            ("within_A", bank("WUT", "LR-A")), ("within_B", bank("WUT", "LR-B")), ("within_C", bank("WUT", "LR-C"))]
    meths, bt = _mirror(y, P0, spec)
    abm = {m: {k: _tri(P0, y, meths[m], bt[m], k) for k in ("AUROC", "FPR@t_r80")} for m in meths}
    abp = {f"within_{r}-minus-transfer_{r}": [r4(v) for v in ci3(bt[f"within_{r}"]["AUROC"] - bt[f"transfer_{r}"]["AUROC"])]
           for r in "ABC"}
    ref = ab["R4"]["P0"]["by_prev"]["native"]["metrics"]
    chk["ablation_P0_native_max_abs_diff"] = {m: {k: _cmp(abm[m][k], ref[m][k]) for k in ("AUROC", "FPR@t_r80")} for m in meths}
    chk["ablation_P0_paired_AUROC_max_abs_diff"] = {
        k: _cmp(abp[k], [ab["R4"]["P0"]["by_prev"]["native"]["paired"][k]["AUROC"][x] for x in ("delta", "lo", "hi")]) for k in abp}
    corr["ablation_mirror"] = {"metrics": abm, "paired_AUROC": abp}
    # gru_ladder R4 P0 native
    spec = [("GRU_transfer", bank("R4", "GRU")), ("C_transfer", bank("R4", "LR-C")),
            ("GRU_within", bank("WUT", "GRU")), ("C_within", bank("WUT", "LR-C"))]
    meths, bt = _mirror(y, P0, spec)
    grm = {m: {k: _tri(P0, y, meths[m], bt[m], k) for k in ("AUROC", "FPR@t_r80", "recall@t_r80")} for m in meths}
    grp = {"GRU_within-minus-GRU_transfer": [r4(v) for v in ci3(bt["GRU_within"]["AUROC"] - bt["GRU_transfer"]["AUROC"])]}
    ref = gr["R4"]["P0"]["by_prev"]["native"]["metrics"]
    chk["gru_ladder_P0_native_max_abs_diff"] = {m: {k: _cmp(grm[m][k], ref[m][k]) for k in ("AUROC", "FPR@t_r80", "recall@t_r80")}
                                                for m in meths}
    chk["gru_ladder_P0_paired_AUROC_max_abs_diff"] = _cmp(
        grp["GRU_within-minus-GRU_transfer"],
        [gr["R4"]["P0"]["by_prev"]["native"]["paired"]["GRU_within-minus-GRU_transfer"]["AUROC"][x] for x in ("delta", "lo", "hi")])
    corr["gru_ladder_mirror"] = {"metrics": grm, "paired_AUROC": grp}
    # per-model (r4_control mirror)
    pm = {}; chk["r4_control_per_model_max_abs_diff"] = {}
    for m in rc["model_eligible"]:
        pool = P0[M["model"][P0] == m]
        spec = [("within_UT_full", bank("WUT", "LR-C")), ("within_IT_full", bank("WIT", "LR-C")),
                ("transfer_full", bank("R4", "LR-C")), ("within_UT_agn", bank("WUT", "LR-B")),
                ("transfer_agn", bank("R4", "LR-B"))]
        meths, bt = _mirror(y, pool, spec, metrics=("AUROC", "AP_native"))
        pm[m] = {k: _tri(pool, y, meths[k], bt[k], "AUROC") for k in meths}
        chk["r4_control_per_model_max_abs_diff"][m] = {k: _cmp(pm[m][k], rc["per_model"][m]["metrics"][k]["AUROC"]) for k in meths}
    corr["r4_control_per_model_mirror"] = pm
    pm2 = {}
    for m in rc["model_eligible"]:
        pool = P0[M["model"][P0] == m]
        meths = collections.OrderedDict([(f"transfer:{d}", bank("R4", d)) for d in ("LR-A", "LR-B", "LR-C", "GRU")])
        bt = strat_boot(pool, y, meths, seed=SEED_STRAT, metrics=("AUROC",))
        pm2[m] = {k: _tri(pool, y, meths[k], bt[k], "AUROC") for k in meths}
    corr["per_model_transfer_all4"] = pm2
    # model-balanced (r4_control mirror)
    pool, wd, warr, elig = mb_weights(P0)
    spec = [("within_UT_full", bank("WUT", "LR-C")), ("within_IT_full", bank("WIT", "LR-C")),
            ("transfer_full", bank("R4", "LR-C")), ("within_UT_agn", bank("WUT", "LR-B")),
            ("transfer_agn", bank("R4", "LR-B"))]
    meths, bt = _mirror(y, pool, spec, pos_weights=wd, metrics=("AUROC", "AP_native"))
    mbm = {k: [r4(x) for x in ci3(bt[k]["AUROC"])] for k in meths}
    chk["r4_control_model_balanced_native_max_abs_diff"] = {
        k: _cmp(mbm[k], rc["model_balanced"]["native_counts"]["metrics"][k]["AUROC"]) for k in meths}
    corr["r4_control_model_balanced_mirror"] = mbm
    # C1 new fits vs re-run v0.3 scripts (point values)
    C1, _, _ = population("C1")
    c1 = {}
    for st, d, ref_v in [("C1_UT", "LR-C", rc["C1"]["native"]["metrics"]["within_UT_full"]["AUROC"][0]),
                         ("C1_UT", "LR-B", rc["C1"]["native"]["metrics"]["within_UT_agn"]["AUROC"][0]),
                         ("C1_IT", "LR-C", rc["C1"]["native"]["metrics"]["within_IT_full"]["AUROC"][0]),
                         ("C1_IT", "LR-B", rc["C1"]["native"]["metrics"]["within_IT_agn"]["AUROC"][0]),
                         ("C1_UT", "LR-A", ab["R4"]["C1"]["by_prev"]["native"]["metrics"]["within_A"]["AUROC"][0]),
                         ("C1_UT", "GRU", gr["R4"]["C1"]["by_prev"]["native"]["metrics"]["GRU_within"]["AUROC"][0]),
                         ("R4", "LR-C", rc["C1"]["native"]["metrics"]["transfer_full"]["AUROC"][0]),
                         ("R4", "GRU", gr["R4"]["C1"]["by_prev"]["native"]["metrics"]["GRU_transfer"]["AUROC"][0])]:
        v = point_metrics(C1, y, bank(st, d))["AUROC"]
        c1[f"{st}:{d}"] = {"ours": r4(v), "rerun_v03_script": ref_v, "abs_diff": r4(abs(v - ref_v), 6)}
    chk["C1_point_vs_rerun_scripts"] = c1
    chk["fit_checks_vs_v03_code_paths"] = {job: [pload(unit_path(job, f"rep{r}"))["checks"] for r in REPS]
                                           for job in ("lin_C1_UT", "lin_C1_IT", "lin_noDoS_UT")}
    out["checks_rerun_v03_vs_cache"] = chk
    out["corrected"] = corr
    # ---------------- v0.3 values (archive) and v0.3 recomputed where the archive has none
    arc = _arch("r4_control_results.json"); aab = _arch("ablation_results.json"); agr = _arch("gru_results.json")
    a_nat = arc["P0"]["native"]; b_nat = aab["R4"]["P0"]["by_prev"]["native"]; g_nat = agr["R4"]["P0"]["by_prev"]["native"]
    trip = lambda d: [d[x] for x in ("delta", "lo", "hi")]
    archive = {
        "LR-C": {"transfer_AUROC": a_nat["metrics"]["transfer_full"]["AUROC"], "WUT_AUROC": a_nat["metrics"]["within_UT_full"]["AUROC"],
                 "WIT_AUROC": a_nat["metrics"]["within_IT_full"]["AUROC"],
                 "WUT_minus_transfer": trip(a_nat["paired"]["within_UT_full-minus-transfer_full"]["AUROC"]),
                 "WIT_minus_transfer": trip(a_nat["paired"]["within_IT_full-minus-transfer_full"]["AUROC"]),
                 "transfer_FPR@t_r80": a_nat["metrics"]["transfer_full"]["FPR@t_r80"],
                 "per_model_transfer": {m: arc["per_model"][m]["metrics"]["transfer_full"]["AUROC"] for m in arc["per_model"]},
                 "source": "archive_v0.3/r4_control_results.json (full = C)"},
        "LR-B": {"transfer_AUROC": a_nat["metrics"]["transfer_agn"]["AUROC"], "WUT_AUROC": a_nat["metrics"]["within_UT_agn"]["AUROC"],
                 "WIT_AUROC": a_nat["metrics"]["within_IT_agn"]["AUROC"],
                 "WUT_minus_transfer": trip(a_nat["paired"]["within_UT_agn-minus-transfer_agn"]["AUROC"]),
                 "WIT_minus_transfer": trip(a_nat["paired"]["within_IT_agn-minus-transfer_agn"]["AUROC"]),
                 "transfer_FPR@t_r80": a_nat["metrics"]["transfer_agn"]["FPR@t_r80"],
                 "per_model_transfer": {m: arc["per_model"][m]["metrics"]["transfer_agn"]["AUROC"] for m in arc["per_model"]},
                 "source": "archive_v0.3/r4_control_results.json (agn = B)"},
        "LR-A": {"transfer_AUROC": b_nat["metrics"]["transfer_A"]["AUROC"], "WUT_AUROC": b_nat["metrics"]["within_A"]["AUROC"],
                 "WUT_minus_transfer": trip(b_nat["paired"]["within_A-minus-transfer_A"]["AUROC"]),
                 "transfer_FPR@t_r80": b_nat["metrics"]["transfer_A"]["FPR@t_r80"],
                 "source": "archive_v0.3/ablation_results.json"},
        "GRU": {"transfer_AUROC": g_nat["metrics"]["GRU_transfer"]["AUROC"], "WUT_AUROC": g_nat["metrics"]["GRU_within"]["AUROC"],
                "WUT_minus_transfer": trip(g_nat["paired"]["GRU_within-minus-GRU_transfer"]["AUROC"]),
                "transfer_FPR@t_r80": g_nat["metrics"]["GRU_transfer"]["FPR@t_r80"],
                "source": "archive_v0.3/gru_results.json"}}
    A, yA, modA, vb, vchk = _v03_bank()
    popA = np.arange(len(A))
    rec = {"checks": vchk}
    meths = collections.OrderedDict((k, vb[k]) for k in ("transfer:LR-A", "transfer:LR-B", "transfer:LR-C", "transfer:GRU",
                                                        "WUT:LR-A", "WIT:LR-A"))
    bt = strat_boot(popA, yA, meths, seed=SEED_STRAT, metrics=("AUROC", "FPR@t_r80"))
    rec["P0"] = {k: {kk: _tri(popA, yA, meths[k], bt[k], kk) for kk in ("AUROC", "FPR@t_r80")} for k in meths}
    rec["P0_paired_AUROC"] = {f"{w}-minus-transfer:LR-A": [r4(v) for v in ci3(bt[w]["AUROC"] - bt["transfer:LR-A"]["AUROC"])]
                              for w in ("WUT:LR-A", "WIT:LR-A")}
    rec["per_model_transfer"] = {}
    for m in arc["model_eligible"]:
        pool = popA[modA == m]
        mm = collections.OrderedDict((k, vb[k]) for k in ("transfer:LR-A", "transfer:LR-B", "transfer:LR-C", "transfer:GRU"))
        btm = strat_boot(pool, yA, mm, seed=SEED_STRAT, metrics=("AUROC",))
        rec["per_model_transfer"][m] = {k: _tri(pool, yA, mm[k], btm[k], "AUROC") for k in mm}
    rec["abs_diff_vs_archive_point"] = {
        "transfer:LR-A": r4(abs(rec["P0"]["transfer:LR-A"]["AUROC"][0] - archive["LR-A"]["transfer_AUROC"][0]), 6),
        "transfer:LR-B": r4(abs(rec["P0"]["transfer:LR-B"]["AUROC"][0] - archive["LR-B"]["transfer_AUROC"][0]), 6),
        "transfer:LR-C": r4(abs(rec["P0"]["transfer:LR-C"]["AUROC"][0] - archive["LR-C"]["transfer_AUROC"][0]), 6),
        "transfer:GRU": r4(abs(rec["P0"]["transfer:GRU"]["AUROC"][0] - archive["GRU"]["transfer_AUROC"][0]), 6),
        "WUT:LR-A": r4(abs(rec["P0"]["WUT:LR-A"]["AUROC"][0] - archive["LR-A"]["WUT_AUROC"][0]), 6),
        "per_model_LR-C_max": r4(max(abs(rec["per_model_transfer"][m]["transfer:LR-C"][0] - archive["LR-C"]["per_model_transfer"][m][0])
                                     for m in arc["model_eligible"]), 6),
        "per_model_LR-B_max": r4(max(abs(rec["per_model_transfer"][m]["transfer:LR-B"][0] - archive["LR-B"]["per_model_transfer"][m][0])
                                     for m in arc["model_eligible"]), 6)}
    out["v03"] = {"archive": archive, "recomputed_on_archived_extraction": rec}
    # corrected LR-A / GRU WIT (not produced by any v0.3 script): new bootstrap
    meths = collections.OrderedDict([("transfer:LR-A", bank("R4", "LR-A")), ("WIT:LR-A", bank("WIT", "LR-A")),
                                     ("transfer:GRU", bank("R4", "GRU")), ("WIT:GRU", bank("WIT", "GRU"))])
    bt = strat_boot(P0, y, meths, seed=SEED_STRAT, metrics=("AUROC",))
    wit = {k: _tri(P0, y, meths[k], bt[k], "AUROC") for k in meths}
    witd = {d: [r4(v) for v in ci3(bt[f"WIT:{d}"]["AUROC"] - bt[f"transfer:{d}"]["AUROC"])] for d in ("LR-A", "GRU")}
    corr["WIT_new"] = {"metrics": wit, "paired_AUROC": witd}
    # ---------------- table
    tab = {}
    for d in ("LR-A", "LR-B", "LR-C", "GRU"):
        a = archive[d]
        v = {"transfer_AUROC": a["transfer_AUROC"], "WUT_AUROC": a["WUT_AUROC"],
             "WIT_AUROC": a.get("WIT_AUROC") or (rec["P0"]["WIT:LR-A"]["AUROC"] if d == "LR-A" else None),
             "WUT_minus_transfer": a["WUT_minus_transfer"],
             "WIT_minus_transfer": a.get("WIT_minus_transfer") or (rec["P0_paired_AUROC"]["WIT:LR-A-minus-transfer:LR-A"] if d == "LR-A" else None),
             "transfer_FPR@t_r80": a["transfer_FPR@t_r80"],
             "per_model_transfer": a.get("per_model_transfer") or {m: rec["per_model_transfer"][m][f"transfer:{d}"] for m in arc["model_eligible"]},
             "sources": {"main": a["source"],
                         "WIT": "archive" if a.get("WIT_AUROC") else ("recomputed on archived v0.3 extraction" if d == "LR-A" else "not run in v0.3"),
                         "per_model": "archive" if a.get("per_model_transfer") else "recomputed on archived v0.3 extraction"}}
        tab[d] = {"v03": v}
    rm = corr["r4_control_mirror"]; am = corr["ablation_mirror"]; gm = corr["gru_ladder_mirror"]
    tab["LR-C"]["corrected"] = {"transfer_AUROC": rm["metrics"]["transfer_full"]["AUROC"], "WUT_AUROC": rm["metrics"]["within_UT_full"]["AUROC"],
                                "WIT_AUROC": rm["metrics"]["within_IT_full"]["AUROC"],
                                "WUT_minus_transfer": rm["paired_AUROC"]["within_UT_full-minus-transfer_full"],
                                "WIT_minus_transfer": rm["paired_AUROC"]["within_IT_full-minus-transfer_full"],
                                "transfer_FPR@t_r80": rm["metrics"]["transfer_full"]["FPR@t_r80"],
                                "per_model_transfer": {m: pm[m]["transfer_full"] for m in pm}}
    tab["LR-B"]["corrected"] = {"transfer_AUROC": rm["metrics"]["transfer_agn"]["AUROC"], "WUT_AUROC": rm["metrics"]["within_UT_agn"]["AUROC"],
                                "WIT_AUROC": rm["metrics"]["within_IT_agn"]["AUROC"],
                                "WUT_minus_transfer": rm["paired_AUROC"]["within_UT_agn-minus-transfer_agn"],
                                "WIT_minus_transfer": rm["paired_AUROC"]["within_IT_agn-minus-transfer_agn"],
                                "transfer_FPR@t_r80": rm["metrics"]["transfer_agn"]["FPR@t_r80"],
                                "per_model_transfer": {m: pm[m]["transfer_agn"] for m in pm}}
    tab["LR-A"]["corrected"] = {"transfer_AUROC": am["metrics"]["transfer_A"]["AUROC"], "WUT_AUROC": am["metrics"]["within_A"]["AUROC"],
                                "WIT_AUROC": wit["WIT:LR-A"], "WUT_minus_transfer": am["paired_AUROC"]["within_A-minus-transfer_A"],
                                "WIT_minus_transfer": witd["LR-A"], "transfer_FPR@t_r80": am["metrics"]["transfer_A"]["FPR@t_r80"],
                                "per_model_transfer": {m: pm2[m]["transfer:LR-A"] for m in pm2}}
    tab["GRU"]["corrected"] = {"transfer_AUROC": gm["metrics"]["GRU_transfer"]["AUROC"], "WUT_AUROC": gm["metrics"]["GRU_within"]["AUROC"],
                               "WIT_AUROC": wit["WIT:GRU"], "WUT_minus_transfer": gm["paired_AUROC"]["GRU_within-minus-GRU_transfer"],
                               "WIT_minus_transfer": witd["GRU"], "transfer_FPR@t_r80": gm["metrics"]["GRU_transfer"]["FPR@t_r80"],
                               "per_model_transfer": {m: pm2[m]["transfer:GRU"] for m in pm2}}
    out["table"] = tab
    out["notes"] = [
        "Corrected LR-B/LR-C values use the method order, seed (12345) and B (2000) of r4_control.py, LR-A those of "
        "ablation.py and the GRU those of gru_ladder.py, so their intervals can be compared with the re-run v0.3 scripts "
        "(checks_rerun_v03_vs_cache). WIT for LR-A and the GRU, and per-model LR-A/GRU, are new bootstraps (seed 12345).",
        "v0.3 per-model LR-A/GRU and v0.3 WIT LR-A were never computed by the v0.3 scripts; they are recomputed here on "
        "archive_v0.3/agentdojo_traces.jsonl with the v0.3 code paths (ablation.fit, gru_detector.train_gru). The GRU "
        "was not run under WIT in v0.3."]
    jdump(out, sec_path("m7"))
    return True


# ----------------------------------------------------------------------------------------- m1 / m16: GRU outcomes
def section_gru_outcomes(budget):
    """GRU_PREREGISTRATION.md Outcomes 1, 2 and 4 (m1, m16)."""
    out = {}
    lin_R1 = cache("linear", "R1"); g_R1 = cache("gru", "R1")

    def mk(gc, lc, det="GRU"):
        E = np.where(gc["pool_E"])[0]
        yE = gc["y"][E].astype(int)
        m = collections.OrderedDict([("GRU", {"S": gc["scores"][det][:, E].astype(float), "p80": gc["pred"][det]["t_r80"][:, E]}),
                                     ("C", {"S": np.asarray(lc["scores"]["LR-C"], float)[None, E],
                                            "p80": np.asarray(lc["pred"]["LR-C"]["t_r80"])[None, E]})])
        return yE, m
    yE, m1 = mk(g_R1, lin_R1)
    popE = np.arange(len(yE))
    bt1 = strat_boot(popE, yE, m1, B=1000, seed=5000, metrics=("AUROC",))
    out["R1"] = {"AUROC": _tri(popE, yE, m1["GRU"], bt1["GRU"], "AUROC"),
                 "note": "gru_ladder.py R1 bootstrap: B = 1000, seed 5000, methods GRU and C (pool E)"}
    out["R1"]["lower_bound_gt_0.7"] = bool(out["R1"]["AUROC"][1] > 0.7)
    out["R3"] = {}
    for i, d in enumerate(["medical", "banking", "web", "email", "coding"]):
        y3, m3 = mk(cache("gru", f"R3_{d}"), cache("linear", f"R3_{d}"))
        p3 = np.arange(len(y3))
        bt3 = strat_boot(p3, y3, m3, B=1000, seed=5200 + i, metrics=("AUROC",))
        dl = ek.indep_delta(bt3["GRU"], bt1["GRU"], "AUROC")
        out["R3"][d] = {"AUROC": _tri(p3, y3, m3["GRU"], bt3["GRU"], "AUROC"),
                        "R3_minus_R1_indep": [r4(dl["delta"]), r4(dl["lo"]), r4(dl["hi"])],
                        "degradation_detected": bool(dl["hi"] < 0)}
    out["R3_folds_degradation_detected"] = int(sum(v["degradation_detected"] for v in out["R3"].values()))
    M = meta(); y = M["y"]; P0 = np.arange(M["nP0"])
    spec = [("GRU_transfer", bank("R4", "GRU")), ("C_transfer", bank("R4", "LR-C")),
            ("GRU_within", bank("WUT", "GRU")), ("C_within", bank("WUT", "LR-C"))]
    meths, bt = _mirror(y, P0, spec, metrics=("AUROC",))
    out["R4_P0_transfer"] = _tri(P0, y, meths["GRU_transfer"], bt["GRU_transfer"], "AUROC")
    out["R4_P0_within"] = _tri(P0, y, meths["GRU_within"], bt["GRU_within"], "AUROC")
    pool, wd, warr, elig = mb_weights(P0)
    meths, btn = _mirror(y, pool, spec, pos_weights=wd, metrics=("AUROC",))
    out["R4_MB_native_transfer"] = [r4(x) for x in ci3(btn["GRU_transfer"]["AUROC"])]
    out["R4_MB_native_within"] = [r4(x) for x in ci3(btn["GRU_within"]["AUROC"])]
    out["R4_MB_native_transfer_weighted_point"] = r4(point_metrics(pool, y, bank("R4", "GRU"), warr)["AUROC"])
    nneg = int((y[pool] == 0).sum())
    meths, bt10 = _mirror(y, pool, spec, seed=999, n_pos=int(round(0.10 / 0.90 * nneg)), pos_weights=wd, metrics=("AUROC",))
    out["R4_MB_p10_transfer_gru_ladder_mirror"] = [r4(x) for x in ci3(bt10["GRU_transfer"]["AUROC"])]
    des = cluster_design(pool, y, "UT")
    btc = clus_boot(y, {"GRU_transfer": bank("R4", "GRU")}, des, wpos=warr, metrics=("AUROC",))
    out["R4_MB_native_transfer_cluster_UT"] = [out["R4_MB_native_transfer_weighted_point"]] + [r4(x) for x in ci3(btc["GRU_transfer"]["AUROC"])[1:]]
    des0 = cluster_design(P0, y, "UT")
    btc0 = clus_boot(y, {"GRU_transfer": bank("R4", "GRU")}, des0, metrics=("AUROC",))
    out["R4_P0_transfer_cluster_UT"] = [out["R4_P0_transfer"][0]] + [r4(x) for x in ci3(btc0["GRU_transfer"]["AUROC"])[1:]]

    def nc(t):   # Outcome 1 wording: "AUROC CI includes 0.5 or below"
        return bool(t[1] <= 0.5)
    o1 = {"strong_R1": out["R1"]["lower_bound_gt_0.7"], "R3_degradation_ge3": out["R3_folds_degradation_detected"] >= 3,
          "R4_P0_ci_includes_0.5_or_below": nc(out["R4_P0_transfer"]),
          "R4_MB_native_ci_includes_0.5_or_below": nc(out["R4_MB_native_transfer"]),
          "R4_MB_p10_ci_includes_0.5_or_below": nc(out["R4_MB_p10_transfer_gru_ladder_mirror"]),
          "R4_MB_native_cluster_ci_includes_0.5_or_below": nc(out["R4_MB_native_transfer_cluster_UT"])}
    o1["outcome1_met_native_counts_literal"] = bool(o1["strong_R1"] and o1["R3_degradation_ge3"] and
                                                    o1["R4_P0_ci_includes_0.5_or_below"] and o1["R4_MB_native_ci_includes_0.5_or_below"])
    o1["outcome1_met_p10_as_in_v03"] = bool(o1["strong_R1"] and o1["R3_degradation_ge3"] and
                                            o1["R4_P0_ci_includes_0.5_or_below"] and o1["R4_MB_p10_ci_includes_0.5_or_below"])
    o1["wording"] = ("Outcome 1: strong R1 (GRU AUROC CI lower bound > 0.7), detected degradation R1->R3 in >= 3/5 folds, "
                     "and near-chance R4 transfer (AUROC CI includes 0.5 or below on P0 and model-balanced)")
    out["outcome1"] = o1
    o2 = {"R4_P0_lower_gt_0.6": bool(out["R4_P0_transfer"][1] > 0.6),
          "R4_MB_native_lower_gt_0.6": bool(out["R4_MB_native_transfer"][1] > 0.6),
          "R3_no_degradation_ge3": bool((5 - out["R3_folds_degradation_detected"]) >= 3)}
    o2["met"] = bool((o2["R4_P0_lower_gt_0.6"] and o2["R4_MB_native_lower_gt_0.6"]) or o2["R3_no_degradation_ge3"])
    out["outcome2"] = o2
    out["within_AgentDojo_check"] = {"WUT_lower_gt_0.7": bool(out["R4_P0_within"][1] > 0.7),
                                     "transfer_near_chance": bool(out["R4_P0_transfer"][1] <= 0.5 <= out["R4_P0_transfer"][2]
                                                                  or out["R4_P0_transfer"][0] < 0.6)}
    # ---------------- m16: per-seed spread
    ss = {}
    for st in ["R1"] + [f"R3_{d}" for d in ["medical", "banking", "web", "email", "coding"]] + ["R4"]:
        gc = cache("gru", st)
        E = np.where(gc["pool_E"])[0]; yE_ = gc["y"][E].astype(int)
        notE = np.where(~gc["pool_E"])[0]
        ss[st] = {}
        for det in [d for d in ("GRU", "GRU-drop0.2", "GRU-drop0.1") if d in gc["scores"]]:
            per = []
            for sd in range(gc["scores"][det].shape[0]):
                s = gc["scores"][det][sd, E].astype(float)
                p8 = gc["pred"][det]["t_r80"][sd, E]; pv = gc["pred"][det]["t_val80"][sd, E]
                auc, _ = auc_ap(yE_, s)
                one = {"meth": {"S": s[None, :], "p80": p8[None, :]}}
                bts = strat_boot(np.arange(len(yE_)), yE_, one, B=1000, seed=6000 + sd,
                                 metrics=("AUROC", "FPR@t_r80", "recall@t_r80"))
                cw = {k: r4(ci3(bts["meth"][k])[2] - ci3(bts["meth"][k])[1]) for k in ("AUROC", "FPR@t_r80", "recall@t_r80")}
                per.append({"seed": sd, "AUROC": r4(auc), "FPR@t_r80": r4(p8[yE_ == 0].mean()),
                            "recall@t_r80": r4(p8[yE_ == 1].mean()), "FPR@t_val80": r4(pv[yE_ == 0].mean()),
                            "recall@t_val80": r4(pv[yE_ == 1].mean()),
                            "failed_flag@t_r80": r4(gc["pred"][det]["t_r80"][sd, notE].mean()) if len(notE) else None,
                            "thr_r80": r4(float(gc["thr"][det]["t_r80"][sd]), 6), "single_seed_ci_width": cw})
            keys = ("AUROC", "FPR@t_r80", "recall@t_r80", "FPR@t_val80", "recall@t_val80")
            rng_ = {k: r4(max(p[k] for p in per) - min(p[k] for p in per)) for k in keys}
            sd_ = {k: r4(float(np.std([p[k] for p in per], ddof=1))) for k in keys}
            wid = {k: r4(float(np.mean([p["single_seed_ci_width"][k] for p in per]))) for k in ("AUROC", "FPR@t_r80", "recall@t_r80")}
            crit = {"C2_AUROC_range_gt_single_seed_CI_width": bool(rng_["AUROC"] > wid["AUROC"]),
                    "C3_FPR_range_gt_single_seed_CI_width": bool(rng_["FPR@t_r80"] > wid["FPR@t_r80"]),
                    "C3_recall_range_gt_single_seed_CI_width": bool(rng_["recall@t_r80"] > wid["recall@t_r80"]),
                    "C4_FPR_or_recall_range_ge_0.10": bool(rng_["FPR@t_r80"] >= 0.10 or rng_["recall@t_r80"] >= 0.10)}
            ss[st][det] = {"per_seed": per, "range": rng_, "sd": sd_, "mean_single_seed_ci_width": wid, "criteria": crit}
    r1_auc = [p["AUROC"] for p in ss["R1"]["GRU"]["per_seed"]]
    sign = {}
    for d in ["medical", "banking", "web", "email", "coding"]:
        dif = [ss[f"R3_{d}"]["GRU"]["per_seed"][k]["AUROC"] - r1_auc[k] for k in range(3)]
        sign[d] = {"per_seed_R3_minus_R1": [r4(x) for x in dif],
                   "same_sign_all_seeds": bool(all(x < 0 for x in dif) or all(x > 0 for x in dif))}
    out["seed_spread"] = ss
    out["seed_sign_consistency"] = sign
    r3g = [ss[f"R3_{d}"]["GRU"]["criteria"] for d in ["medical", "banking", "web", "email", "coding"]]
    o4 = {"criteria_definitions": {
              "C1": "sign of R3-R1 AUROC identical across the 3 seeds (the criterion GRU_RESULTS.md used)",
              "C2": "across-seed AUROC range exceeds the mean single-seed 95% bootstrap interval width (B = 1000)",
              "C3": "across-seed range of FPR or recall at theta_r80 exceeds the mean single-seed 95% interval width",
              "C4": "across-seed range of FPR or recall at theta_r80 >= 0.10 (absolute)"},
          "note": "GRU_PREREGISTRATION.md does not operationalise 'large seed variance'; C2-C4 are defined here, post hoc.",
          "folds_C1_sign_consistent": int(sum(v["same_sign_all_seeds"] for v in sign.values())),
          "folds_C2": int(sum(c["C2_AUROC_range_gt_single_seed_CI_width"] for c in r3g)),
          "folds_C3": int(sum(c["C3_FPR_range_gt_single_seed_CI_width"] or c["C3_recall_range_gt_single_seed_CI_width"] for c in r3g)),
          "folds_C4": int(sum(c["C4_FPR_or_recall_range_ge_0.10"] for c in r3g))}
    o4["triggered_ranking"] = bool(o4["folds_C2"] > 0)
    o4["triggered_operating_point"] = bool(o4["folds_C3"] > 0 or o4["folds_C4"] > 0)
    out["outcome4"] = o4
    jdump(out, sec_path("gru_outcomes"))
    return True


# ----------------------------------------------------------------------------------------- m6: <UNK> within AgentDojo
def section_unk(budget):
    import inspect
    import gru_detector as gd
    from gru_ladder import stratified_val
    M = meta(); P0, FA, R = rows(); y = M["y"]; nP = M["nP0"]
    out = {"build_tool_vocab_source": inspect.getsource(gd.build_tool_vocab).strip(),
           "vocab_from_benign_only": "r[\"category\"] == \"benign\"" in inspect.getsource(gd.build_tool_vocab)}
    g = cache("gru", "WUT")
    out["cache_unk_train_steps_encoded"] = np.asarray(g["unk_train_steps"]["GRU"]).tolist()
    out["cache_unk_train_traces"] = np.asarray(g["unk_train_traces"]["GRU"]).tolist()
    out["cache_unk_train_traces_pos"] = np.asarray(g["unk_train_traces_pos"]["GRU"]).tolist()
    out["cache_unk_norm"] = np.round(np.asarray(g["unk_norm"]["GRU"]), 4).tolist()
    # recount on the fit partitions (90% after gru_ladder.stratified_val), untruncated and encoded, by class
    tools_benign_all = {s["tool"] for r in P0 if r["category"] == "benign" for s in r["steps"]}
    tools_attacked_all = {s["tool"] for r in P0 if r["category"] == "attacked" for s in r["steps"]}
    out["tools_only_in_attacked_P0"] = sorted(tools_attacked_all - tools_benign_all)
    out["n_traces_with_attack_only_tools"] = int(sum(any(s["tool"] in (tools_attacked_all - tools_benign_all) for s in r["steps"])
                                                    for r in P0))
    per = []
    from build_gru_cache import wut_folds
    for rep in REPS:
        f_p0, f_fa, _ = wut_folds(P0, FA, rep)
        for f in range(K):
            tr = [P0[i] for i in np.where(f_p0 != f)[0]]
            tr2, va2 = stratified_val(tr, rep)
            V = gd.build_tool_vocab(tr2)
            Vall = {s["tool"] for r in tr2 for s in r["steps"]}
            c = _unk_counts(tr2, V)
            src = collections.Counter(s["tool"] for r in tr2 if r["category"] == "attacked" for s in r["steps"] if s["tool"] not in V)
            per.append({"rep": rep, "fold": f, "n_fit": len(tr2), "n_fit_pos": int(sum(r["category"] == "attacked" for r in tr2)),
                        "vocab_benign_only": len(V), "vocab_if_all_fit_traces": len(Vall), **c,
                        "unk_tools_in_pos": len(src), "unk_tools_also_benign_in_P0": int(sum(t in tools_benign_all for t in src)),
                        "unk_steps_from_tools_absent_from_all_benign_P0": int(sum(v for t, v in src.items() if t not in tools_benign_all))})
    out["fit_partitions"] = per
    out["fit_partitions_summary"] = {k: {"min": int(min(p[k] for p in per)), "max": int(max(p[k] for p in per)),
                                         "mean": r4(np.mean([p[k] for p in per]), 1)}
                                     for k in ("unk_steps", "unk_steps_encoded", "unk_traces", "unk_traces_pos",
                                               "unk_steps_from_tools_absent_from_all_benign_P0", "vocab_benign_only",
                                               "vocab_if_all_fit_traces")}
    out["cache_matches_recount_encoded"] = bool(np.array_equal(np.array([[p["unk_steps_encoded"] for p in per if p["rep"] == r] for r in REPS]),
                                                               np.asarray(g["unk_train_steps"]["GRU"])))
    # WUT GRU excluding test traces with <UNK> (per repetition, the model's own vocabulary = cov_model < 1)
    P0i = np.arange(nP)
    cm = np.asarray(g["cov_model"]["GRU"])            # (5, n)
    has_unk = cm[:, :nP] < 1.0
    gb = bank("WUT", "GRU"); lb = bank("WUT", "LR-C"); tb = bank("R4", "GRU"); nb = bank("WUT_noUNK", "GRU")
    res = {"n_excluded_per_rep": [{"pos": int((has_unk[r] & (y[P0i] == 1)).sum()), "neg": int((has_unk[r] & (y[P0i] == 0)).sum())}
                                  for r in REPS]}

    def per_rep_auc(S, keep):
        return [auc_ap(y[P0i][keep[r]], S[r if S.shape[0] > 1 else 0][P0i][keep[r]])[0] for r in REPS]
    keep = ~has_unk; allk = np.ones_like(keep)
    for name, bk in (("WUT_GRU", gb), ("WUT_LR-C", lb), ("WUT_noUNKtrain_GRU", nb)):
        res[name] = {"all_test": r4(np.mean(per_rep_auc(bk["S"], allk))), "no_unk_test": r4(np.mean(per_rep_auc(bk["S"], keep))),
                     "per_rep_no_unk_test": [r4(x) for x in per_rep_auc(bk["S"], keep)]}
    # bootstrap: replicate draws a repetition r, then resamples positives / negatives of that repetition's subset
    rng = np.random.default_rng(SEED_STRAT)
    names = ("WUT_GRU", "WUT_LR-C", "WUT_noUNKtrain_GRU")
    bts = {k: {"all": np.empty(B), "nounk": np.empty(B)} for k in names}
    dl = {"GRU_nounk_minus_all": np.empty(B), "noUNKtrain_minus_GRU_all": np.empty(B), "noUNKtrain_minus_GRU_nounk": np.empty(B),
          "GRU_minus_LRC_nounk": np.empty(B)}
    for b in range(B):
        r = rng.integers(len(REPS))
        pos = P0i[(y[P0i] == 1)]; neg = P0i[(y[P0i] == 0)]
        ib = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        kb = keep[r][ib]
        vals = {}
        for k, bk in zip(names, (gb, lb, nb)):
            s = bk["S"][r][ib]
            vals[(k, "all")] = auc_ap(y[ib], s)[0]; vals[(k, "nounk")] = auc_ap(y[ib][kb], s[kb])[0]
            bts[k]["all"][b] = vals[(k, "all")]; bts[k]["nounk"][b] = vals[(k, "nounk")]
        dl["GRU_nounk_minus_all"][b] = vals[("WUT_GRU", "nounk")] - vals[("WUT_GRU", "all")]
        dl["noUNKtrain_minus_GRU_all"][b] = vals[("WUT_noUNKtrain_GRU", "all")] - vals[("WUT_GRU", "all")]
        dl["noUNKtrain_minus_GRU_nounk"][b] = vals[("WUT_noUNKtrain_GRU", "nounk")] - vals[("WUT_GRU", "nounk")]
        dl["GRU_minus_LRC_nounk"][b] = vals[("WUT_GRU", "nounk")] - vals[("WUT_LR-C", "nounk")]
    for k in names:
        res[k]["ci_all_test"] = [r4(x) for x in ci3(bts[k]["all"])[1:]]
        res[k]["ci_no_unk_test"] = [r4(x) for x in ci3(bts[k]["nounk"])[1:]]
    res["paired"] = {k: [r4(x) for x in ci3(v)] for k, v in dl.items()}
    res["note"] = ("Exclusion uses each repetition's own fold model vocabulary (cache cov_model < 1): a P0 test trace is "
                   "excluded when any of its tools is <UNK> for the model that scored it. Point = mean over the 5 "
                   "repetitions; the bootstrap resamples P0 (stratified, B = 2000, seed 12345) and applies the drawn "
                   "repetition's exclusion mask; paired contrasts use the same replicate.")
    nu = [pload(unit_path("gru_WUT_noUNK", f"rep{r}_fold{f}")) for r in REPS for f in range(K)]
    res["noUNKtrain_fit"] = {"n_removed_unk_traces": [p["n_removed_unk"] for p in nu],
                             "unk_before_removal_traces_pos": [p["unk_fit_before_removal"]["unk_traces_pos"] for p in nu],
                             "unk_after_removal_steps": [p["unk_fit"]["unk_steps"] for p in nu]}
    out["WUT_GRU_without_unk"] = res
    jdump(out, sec_path("unk"))
    return True


# ----------------------------------------------------------------------------------------- m7 / Q4: short runs, DoS
def _git_show(path):
    import subprocess
    adj = os.environ["AGENTDOJO"]
    return subprocess.run(["git", "-C", adj, "show", f"HEAD:{path}"], capture_output=True, text=True, encoding="utf-8").stdout


def section_dos(budget):
    M = meta(); P0, FA, R = rows()
    man = json.load(open(os.path.join(ROOT, "agentdojo_extract_manifest.json"), encoding="utf-8"))
    out = {}
    tab = {}
    for cls in ("benign", "attacked", "attack_failed"):
        allc = sum(v.get(f"{cls}_all", 0) for v in man["by_model"].values())
        exc = sum(v.get(f"{cls}_excluded_short", 0) for v in man["by_model"].values())
        tab[cls] = {"all": allc, "excluded_fewer_than_two_calls": exc, "kept": allc - exc,
                    "excluded_fraction": r4(exc / allc if allc else float("nan"))}
    out["exclusion_by_class"] = tab
    out["exclusion_by_model"] = {m: {cls: {"all": v.get(f"{cls}_all", 0), "excluded": v.get(f"{cls}_excluded_short", 0),
                                           "kept": v.get(cls, 0),
                                           "excluded_fraction": r4(v.get(f"{cls}_excluded_short", 0) / v[f"{cls}_all"])
                                           if v.get(f"{cls}_all") else None}
                                     for cls in ("benign", "attacked", "attack_failed")}
                                 for m, v in man["by_model"].items()}
    out["manifest_counts_kept"] = man["counts_kept"]
    out["manifest_alignment"] = man["alignment"]
    # length is discriminative (univariate AUROC of n_steps on P0), which is why the asymmetric filter matters
    P0i = np.arange(M["nP0"])
    out["n_steps_univariate_AUROC_P0"] = r4(auc_ap(M["y"][P0i], M["n_steps"][P0i].astype(float))[0])
    # ---- DoS label semantics (AgentDojo source at the pinned commit)
    bench = _git_show("src/agentdojo/benchmark.py").splitlines()
    dos = _git_show("src/agentdojo/attacks/dos_attacks.py").splitlines()
    import subprocess
    commit = subprocess.run(["git", "-C", os.environ["AGENTDOJO"], "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()

    def lines(src, a, b):
        return [f"{i}: {src[i - 1]}" for i in range(a, b + 1)]
    out["dos_semantics"] = {
        "commit": commit,
        "benchmark.py_73_75": lines(bench, 73, 75),
        "benchmark.py_149_150": lines(bench, 149, 150),
        "benchmark.py_error_paths_129_130_138_139_146_147": lines(bench, 129, 130) + lines(bench, 138, 139) + lines(bench, 146, 147),
        "dos_attacks.py_class_header": [f"{i}: {dos[i - 1]}" for i in range(10, 15)],
        "reading": ("For attacks with is_dos_attack = True, AgentDojo runs only the first injection task of the suite "
                    "(benchmark.py 73-75), so the injection_task id of a DoS run is a placeholder, and it overwrites "
                    "security with 'not utility' (149-150): a DoS run counts as a successful attack exactly when the "
                    "user task was NOT completed, whatever the agent did. In all three exception paths (context length, "
                    "internal server error, server error) the run is scored utility = False, security = True.")}
    dosP = [i for i in range(M["nP0"]) if M["is_dos"][i] and M["y"][i] == 1]
    dosF = [i for i in range(M["nP0"], M["n"]) if M["is_dos"][i]]
    out["dos_counts"] = {"P0_positives": len(dosP), "failed": len(dosF),
                         "P0_positives_by_attack_type": dict(collections.Counter(M["attack_type"][dosP])),
                         "P0_positives_by_suite_injection_task": dict(collections.Counter(f"{M['suite'][i]}|{M['injection_task'][i]}" for i in dosP)),
                         "P0_positives_utility_false": int(sum(not R[i]["meta"]["utility"] for i in dosP)),
                         "failed_utility_true": int(sum(bool(R[i]["meta"]["utility"]) for i in dosF)),
                         "P0_positives_models": dict(collections.Counter(M["model"][dosP]))}
    # error field of the raw runs (exception paths are labelled compromised)
    adj = os.environ["AGENTDOJO"]
    err = collections.Counter(); err_ex = []
    for i, r in enumerate(R):
        m = r["meta"]
        at = m["attack_type"] or "none"; inj = m["injection_task"] or "none"
        f = os.path.join(adj, "runs", m["model"], m["suite"], m["user_task"], at, f"{inj}.json")
        try:
            e = json.load(open(f, encoding="utf-8")).get("error")
        except Exception:
            err[(r["category"], "file_missing")] += 1; continue
        if e:
            err[(r["category"], "error")] += 1
            if len(err_ex) < 5:
                err_ex.append({"id": r["id"], "error": str(e)[:160]})
        else:
            err[(r["category"], "no_error")] += 1
    out["raw_run_error_field"] = {f"{a}|{b}": v for (a, b), v in sorted(err.items())}
    out["raw_run_error_examples"] = err_ex
    # ---- main R4 results with and without the 101 DoS positives (evaluation level; refit sensitivity)
    res = {}
    for name, (pop, ws, sch) in {"P0_UT": ("P0", "WUT", "UT"), "P0noDoS_UT": ("P0_noDoS", "WUT", "UT"),
                                  "P0_IT": ("P0", "WIT", "IT"), "P0noDoS_IT": ("P0_noDoS", "WIT", "IT"),
                                  "P0noDoS_UT_refit": ("P0_noDoS", "WUT_noDoS", "UT"),
                                  "P0F_UT": ("P0F", "WUT", "UT"), "P0FnoDoS_UT": ("P0F_noDoS", "WUT", "UT")}.items():
        p = sec_path(f"dos__{name}")
        if not os.path.exists(p):
            cfg = eval_config(pop, ws, MAIN5, sch, subsample=False)
            jdump(cfg, p); log(f"dos {name} done")
        res[name] = json.load(open(p))
    out["with_without_dos"] = {k: {"n_pos": v["n_pos"], "n_neg": v["n_neg"], "point": v["point"],
                                   "strat_ci_AUROC": {m: v["strat_native"]["ci"][m]["AUROC"] for m in v["point"]},
                                   "cluster_ci": {m: {kk: v["cluster"]["ci"][m][kk] for kk in ("AUROC", "APw10")} for m in v["point"]},
                                   "cluster_paired_AUROC": {kk: vv["AUROC"] for kk, vv in v["cluster"]["paired"].items()}}
                               for k, v in res.items()}
    # recall of DoS vs non-DoS positives at theta_r80 (transfer and within)
    rr = {}
    pos = np.arange(M["nP0"])[M["y"][:M["nP0"]] == 1]
    for st in ("R4", "WUT", "WIT"):
        for d in MAIN5:
            bk = bank(st, d)
            rr[f"{st}:{d}"] = {"recall_dos": r4(np.mean([bk["p80"][r][pos[M['is_dos'][pos]]].mean() for r in range(bk["S"].shape[0])])),
                               "recall_non_dos": r4(np.mean([bk["p80"][r][pos[~M['is_dos'][pos]]].mean() for r in range(bk["S"].shape[0])]))}
    out["recall_dos_vs_non_dos"] = rr
    jdump(out, sec_path("dos"))
    return True


# ----------------------------------------------------------------------------------------- Q2: attack templates
II_FAMILY = ("important_instructions", "important_instructions_no_model_name", "important_instructions_no_names",
             "important_instructions_no_user_name", "important_instructions_wrong_model_name",
             "important_instructions_wrong_user_name")


def section_templates(budget):
    M = meta(); y = M["y"]; P0 = np.arange(M["nP0"])
    dets = MAIN5
    meths = collections.OrderedDict()
    for d in dets:
        meths[f"transfer:{d}"] = bank("R4", d)
    for d in dets:
        meths[f"WUT:{d}"] = bank("WUT", d)
    out = {"per_template_gpt4o": {}}
    g_ben = P0[(M["model"][P0] == GPT4O) & (y[P0] == 0)]
    tmpls = sorted(set(M["attack_type"][P0][(M["model"][P0] == GPT4O) & (y[P0] == 1)]))
    for t in tmpls:
        posi = P0[(M["model"][P0] == GPT4O) & (y[P0] == 1) & (M["attack_type"][P0] == t)]
        pool = np.concatenate([posi, g_ben])
        bt = strat_boot(pool, y, meths, seed=SEED_STRAT, metrics=("AUROC",))
        out["per_template_gpt4o"][t] = {"n_pos": len(posi), "n_neg": len(g_ben), "is_dos": bool(M["is_dos"][posi[0]]),
                                        **{k: _tri(pool, y, meths[k], bt[k], "AUROC") for k in meths}}
    out["per_template_range_transfer"] = {d: [min(v[f"transfer:{d}"][0] for v in out["per_template_gpt4o"].values()),
                                              max(v[f"transfer:{d}"][0] for v in out["per_template_gpt4o"].values())] for d in dets}
    # families within gpt-4o
    fam = {"important_instructions_family": lambda t: t in II_FAMILY, "dos_family": lambda t: "dos" in t,
           "other_templates": lambda t: t not in II_FAMILY and "dos" not in t}
    out["per_family_gpt4o"] = {}
    for fn, f in fam.items():
        posi = P0[(M["model"][P0] == GPT4O) & (y[P0] == 1) & np.array([f(t) for t in M["attack_type"][P0]])]
        pool = np.concatenate([posi, g_ben])
        bt = strat_boot(pool, y, meths, seed=SEED_STRAT, metrics=("AUROC",))
        out["per_family_gpt4o"][fn] = {"n_pos": len(posi), "n_neg": len(g_ben),
                                       **{k: _tri(pool, y, meths[k], bt[k], "AUROC") for k in meths}}
    # common-template design: positives restricted to attack_type == important_instructions (all five models) and to the
    # important_instructions family; pooled, model-balanced (native counts), and cluster bootstraps
    for lab, keepf in (("II_only", lambda t: t == "important_instructions"), ("II_family", lambda t: t in II_FAMILY)):
        sel = (y[P0] == 0) | np.array([keepf(t) for t in M["attack_type"][P0]])
        pool = P0[sel]
        e = {"n_pos": int(y[pool].sum()), "n_neg": int((y[pool] == 0).sum()),
             "pos_by_model": dict(collections.Counter(M["model"][pool][y[pool] == 1]))}
        bt = strat_boot(pool, y, meths, seed=SEED_STRAT, metrics=("AUROC", "APw10"))
        s, pr = summarize_boot(bt, list(meths), ("AUROC", "APw10"), [(f"WUT:{d}", f"transfer:{d}") for d in dets])
        e["pooled"] = {"point": {k: {kk: r4(vv) for kk, vv in point_metrics(pool, y, meths[k]).items()} for k in meths},
                       "strat_ci": s, "strat_paired": pr}
        mpool, wd, warr, elig = mb_weights(pool)
        bt2 = strat_boot(mpool, y, meths, seed=SEED_STRAT, pos_weights=wd, metrics=("AUROC",))
        s2, pr2 = summarize_boot(bt2, list(meths), ("AUROC",), [(f"WUT:{d}", f"transfer:{d}") for d in dets] +
                                 [("transfer:LR-A", "transfer:LR-C")])
        e["model_balanced_native"] = {"eligible": elig, "n_pos": int(y[mpool].sum()), "n_neg": int((y[mpool] == 0).sum()),
                                      "ci": s2, "paired": pr2,
                                      "weighted_point": {k: r4(point_metrics(mpool, y, meths[k], warr)["AUROC"]) for k in meths}}
        des = Clusters(pool, y, M["uti_key"], M["ut_key"])
        bt3 = clus_boot(y, meths, des, metrics=("AUROC",))
        s3, pr3 = summarize_boot(bt3, list(meths), ("AUROC",), [(f"WUT:{d}", f"transfer:{d}") for d in dets])
        e["cluster_UTI"] = {"n_clusters": des.n_clusters(), "ci": s3, "paired": pr3,
                            "note": "positives resampled by suite|user_task|injection_task, negatives by suite|user_task"}
        des4 = Clusters(pool, y, M["ut_key"])
        bt4 = clus_boot(y, meths, des4, metrics=("AUROC",))
        s4, pr4 = summarize_boot(bt4, list(meths), ("AUROC",), [(f"WUT:{d}", f"transfer:{d}") for d in dets])
        e["cluster_UT"] = {"n_clusters": des4.n_clusters(), "ci": s4, "paired": pr4}
        out[lab] = e
    # per model within the common template (transfer)
    out["II_only_per_model"] = {}
    for m in sorted(set(M["model"][P0])):
        pool = P0[(M["model"][P0] == m) & ((y[P0] == 0) | (M["attack_type"][P0] == "important_instructions"))]
        if y[pool].sum() < 2:
            continue
        bt = strat_boot(pool, y, meths, seed=SEED_STRAT, metrics=("AUROC",))
        out["II_only_per_model"][m] = {"n_pos": int(y[pool].sum()), "n_neg": int((y[pool] == 0).sum()),
                                       **{k: _tri(pool, y, meths[k], bt[k], "AUROC") for k in meths}}
    jdump(out, sec_path("templates"))
    return True


# ----------------------------------------------------------------------------------------- m15: model-identity probe
def section_probe(budget):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import StratifiedKFold, GridSearchCV
    P0, FA, R = rows()
    ben = [r for r in P0 if r["category"] == "benign"]
    yb = np.array([int(r["meta"]["model"] == GPT4O) for r in ben])
    suites = sorted({r["agent"] for r in ben})
    Cs = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]

    def X_of(recs, rep, vocab):
        if rep == "suite":
            return np.array([[float(r["agent"] == s) for s in suites] for r in recs])
        out = []
        for r in recs:
            a, b = vectorise(r, vocab)
            out.append(a if rep == "B" else a + b)
        return np.array(out, float)
    reps_scores = {k: [] for k in ("B", "C", "suite")}
    chosen = {k: [] for k in reps_scores}
    fold_auc = {k: [] for k in reps_scores}
    for rep_i in REPS:
        outer = StratifiedKFold(n_splits=K, shuffle=True, random_state=rep_i)
        for k in reps_scores:
            s = np.zeros(len(ben))
            for tr, te in outer.split(np.zeros(len(ben)), yb):
                vocab = build_vocab([ben[i] for i in tr])
                Xtr = X_of([ben[i] for i in tr], k, vocab); Xte = X_of([ben[i] for i in te], k, vocab)
                pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced"))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    gs = GridSearchCV(pipe, {"logisticregression__C": Cs}, scoring="roc_auc",
                                      cv=StratifiedKFold(n_splits=K, shuffle=True, random_state=100 + rep_i)).fit(Xtr, yb[tr])
                s[te] = gs.predict_proba(Xte)[:, 1]
                chosen[k].append(gs.best_params_["logisticregression__C"])
                fold_auc[k].append(auc_ap(yb[te], s[te])[0])
            reps_scores[k].append(s)
    meths = collections.OrderedDict((k, {"S": np.vstack(v), "p80": np.zeros((len(REPS), len(ben)), bool)})
                                    for k, v in reps_scores.items())
    meths["majority"] = {"S": np.zeros((1, len(ben))), "p80": np.zeros((1, len(ben)), bool)}
    pop = np.arange(len(ben))
    bt = strat_boot(pop, yb, meths, seed=SEED_STRAT, metrics=("AUROC",))
    res = {"n_gpt4o": int(yb.sum()), "n_other": int(len(yb) - yb.sum()),
           "design": ("nested CV: outer stratified 5-fold (5 repetitions, shuffle seeds 0-4), inner stratified 5-fold "
                      "GridSearchCV over C in {1e-3,...,100} (L2 logistic regression, class_weight balanced, "
                      "StandardScaler), AUROC scoring; bag-of-tools vocabulary built on each outer training fold; "
                      "point = mean over repetitions of pooled out-of-fold AUROC; 95% stratified bootstrap (B = 2000, "
                      "one repetition per replicate)"),
           "AUROC": {k: _tri(pop, yb, meths[k], bt[k], "AUROC") for k in meths},
           "chosen_C": {k: dict(collections.Counter(map(str, v))) for k, v in chosen.items()},
           "fold_mean_AUROC": {k: {"mean": r4(np.mean(v)), "min": r4(np.min(v)), "max": r4(np.max(v)), "n_folds": len(v),
                                   "note": "secondary: mean of the 25 per-outer-fold AUROCs (avoids the downward bias "
                                           "of pooling out-of-fold scores across folds when the signal is weak)"}
                               for k, v in fold_auc.items()},
           "majority_accuracy": r4(max(yb.mean(), 1 - yb.mean())),
           "paired": {f"{a}-minus-{b}": [r4(x) for x in ci3(bt[a]["AUROC"] - bt[b]["AUROC"])]
                      for a, b in (("B", "suite"), ("C", "suite"), ("C", "B"), ("B", "majority"), ("C", "majority"))}}
    rc = _cur("r4_control_results.json"); arc = _arch("r4_control_results.json")
    res["v03_probe_unregularized"] = {"archive_v0.3": arc["model_identity_probe_benign"],
                                      "rerun_on_corrected": rc["model_identity_probe_benign"],
                                      "note": "r4_control.py probe (5-fold, default C = 1, no nested selection); full = C, agn = B"}
    jdump(res, sec_path("probe"))
    return True


# =============================================================================================== assemble
def _j(name):
    return json.load(open(sec_path(name), encoding="utf-8"))


def _tri_ci(point, ci):
    """[point, lo, hi] from a point value and a ci3 triple (mean, lo, hi)."""
    return [r4(point), r4(ci[1]), r4(ci[2])]


def _maxleaf(o):
    if isinstance(o, dict):
        v = [_maxleaf(x) for x in o.values()]
        v = [x for x in v if x is not None]
        return max(v) if v else None
    if isinstance(o, (int, float)) and not isinstance(o, bool):
        return float(o)
    return None


def _file_info(path):
    return {"path": path.replace("\\", "/"), "sha256": sha256(path),
            "mtime": datetime.datetime.fromtimestamp(os.path.getmtime(path)).isoformat(timespec="seconds")}


def assemble():
    missing = [s for s in SECTIONS if not section_done(s)]
    assert not missing, f"sections not done: {missing}"
    M = meta()
    R = collections.OrderedDict()
    inputs = [os.path.join(CACHE_LIN, f"{s}.pkl") for s in ("R4", "WUT", "WUT_F", "WIT")] + \
             [os.path.join(CACHE_GRU, f"{s}.pkl") for s in ["R1", "R4", "WUT"] +
              [f"R3_{d}" for d in ("medical", "banking", "web", "email", "coding")]] + \
             [DOJO_P0, DOJO_FA, os.path.join(ROOT, "agentdojo_extract_manifest.json"),
              os.path.join(ARCH, "agentdojo_traces.jsonl"), os.path.join(ARCH, "r4_control_results.json"),
              os.path.join(ARCH, "ablation_results.json"), os.path.join(ARCH, "gru_results.json"),
              os.path.join(ROOT, "r4_control_results.json"), os.path.join(ROOT, "ablation_results.json"),
              os.path.join(ROOT, "gru_results.json"), os.path.join(CACHE_LIN, "r4_truncation_check.json")]
    R["meta"] = {
        "script": _file_info(os.path.abspath(__file__)),
        "assembled": datetime.datetime.now().isoformat(timespec="seconds"),
        "review_comments": ["M6", "M7", "m1", "m2 (AgentDojo)", "m6 (within-AgentDojo <UNK>)", "m7", "m15 (model probe)",
                            "m16", "m17 item 3", "Q2", "Q4"],
        "inputs": [_file_info(p) for p in inputs],
        "rerun_v03_scripts_final": "gru_ladder exit 0" in open(os.path.join(HERE, "rerun_v03_scripts.log"), encoding="utf-8").read(),
        "fits": {j: {"units": len(units(j)), "done": sum(os.path.exists(unit_path(j, u)) for u in units(j)),
                     "config": {k: v for k, v in JOBS[j].items()}} for j in JOBS},
        "populations": {"P0": "392 benign + 2,294 compromised (agentdojo_traces.jsonl)",
                        "C1": "P0 minus the 65 benign runs whose user task is an injection task (327 benign)",
                        "MB": "model-balanced: 4 models with >= 50 benign and >= 50 compromised traces (claude excluded); "
                              "positives weighted to the benign model mix (r4_control.py)",
                        "P0F": "P0 plus the 4,987 unsuccessful attacks as additional evaluation negatives",
                        "WUT_F": "within-UT models trained with the unsuccessful attacks of the training folds as negatives",
                        "ATTEMPT": "exploratory: unsuccessful attacks relabelled as positives (attempt target)"},
        "conventions": {
            "point": "metric on the full population; within-AgentDojo = mean over the 5 CV repetitions; GRU transfer = mean "
                     "over 3 training seeds; model-balanced = weighted point (positives weighted, ties 1/2)",
            "primary_interval": "95% cluster bootstrap (plan 3.5), B = 2000, seed 20260927: UT designs resample "
                                "suite|user_task clusters; IT designs resample positives by suite|injection_task and "
                                "negatives by suite|user_task; each replicate draws one CV repetition / seed per method",
            "secondary_interval": "v0.3 trajectory-level stratified bootstrap (evalkit.bootstrap mirror), B = 2000, seed "
                                  "12345; used for the v0.3-vs-corrected comparison (M7) and as a sensitivity",
            "AP_weighted": "prevalence-weighted average precision (plan 3.1): positive weights pi/(1-pi) n_neg/n_pos, "
                           "APw10 = 10%, APw05 = 5%",
            "AP_subsample": "v0.3 secondary estimator: mean AP over replicates with n_pos = round(pi/(1-pi) n_neg) positives",
            "thresholds": "theta_r80 = 20th percentile of in-sample training-positive scores of each fitted model; no test "
                          "data used",
            "detected": "95% interval excludes 0 (differences) or 0.5 (AUROC level)",
            "rounding": "values stored to 4 decimals (some raw floats); r4_deep_summary.md rounds to 3 decimals"}}
    # ------------------------------------------------------------------ M7
    m7 = _j("m7")
    chk = m7["checks_rerun_v03_vs_cache"]
    recon = {"rerun_final": m7["rerun_final"],
             "max_abs_diff_point_and_ci_metrics": max(_maxleaf(chk[k]) for k in (
                 "r4_control_P0_native_max_abs_diff", "ablation_P0_native_max_abs_diff", "gru_ladder_P0_native_max_abs_diff",
                 "r4_control_per_model_max_abs_diff", "r4_control_model_balanced_native_max_abs_diff")),
             "max_abs_diff_paired_AUROC_deltas": max(_maxleaf(chk[k]) if isinstance(chk[k], dict) else chk[k] for k in (
                 "r4_control_P0_paired_AUROC_max_abs_diff", "ablation_P0_paired_AUROC_max_abs_diff",
                 "gru_ladder_P0_paired_AUROC_max_abs_diff")),
             "max_abs_diff_C1_points": max(v["abs_diff"] for v in chk["C1_point_vs_rerun_scripts"].values()),
             "v03_recomputed_vs_archive_max_abs_diff": _maxleaf(m7["v03"]["recomputed_on_archived_extraction"]["abs_diff_vs_archive_point"]),
             "note": "metrics compared at the 4 decimals stored by the v0.3 scripts; paired deltas differ by < 1e-4 "
                     "because the v0.3 scripts store the bootstrap-mean delta of float-noise-tied scores"}
    trunc = json.load(open(os.path.join(CACHE_LIN, "r4_truncation_check.json"), encoding="utf-8"))
    R["M7"] = {"table": m7["table"], "reconciliation_with_rerun_v03_scripts": recon,
               "truncation_check_linear_cache": {"source": "revision/cache/linear/r4_truncation_check.json",
                                                 "untruncated_corrected": trunc["untruncated_corrected"],
                                                 "truncated_2000": trunc["truncated_2000"],
                                                 "review_corrected_estimates": trunc["review_corrected_estimates"]},
               "checks_rerun_v03_vs_cache": chk, "corrected_details": m7["corrected"], "v03_details": m7["v03"],
               "notes": m7["notes"]}
    # ------------------------------------------------------------------ M6
    cfgs = {n: _j(f"case__{n}") for n in CASE_CONFIGS}
    R["M6"] = {"decisions": derive_cases(), "configs": cfgs, "failed_flag_rates": _j("flags"), "per_model": _j("per_model")}
    # ------------------------------------------------------------------ m1 / m2 / m17.3
    pt = _j("alldet_point"); st = _j("alldet_strat"); cu = _j("alldet_cluster_UT"); ciT = _j("alldet_cluster_IT")
    mbs = _j("alldet_mb_strat"); mbu = _j("alldet_mb_cluster_UT"); mbi = _j("alldet_mb_cluster_IT"); sub = _j("alldet_sub")
    go = _j("gru_outcomes")
    names = list(pt["P0"])
    mb_tab = {}
    for m in names:
        cl = mbi if m.startswith("WIT:") else mbu
        mb_tab[m] = {"weighted_point": r4(pt["MB_weighted_point"][m]["AUROC"]),
                     "strat_native_bootstrap_mean_ci": [r4(x) for x in mbs["native_counts"]["ci"][m]["AUROC"]],
                     "cluster_ci": [r4(x) for x in cl["ci"][m]["AUROC"][1:]],
                     "cluster_design": "IT" if m.startswith("WIT:") else "UT",
                     "P0_point": r4(pt["P0"][m]["AUROC"])}
    pair_keys = ["transfer:LR-A-minus-transfer:LR-C", "transfer:GRU-minus-transfer:LR-C", "WUT:LR-A-minus-WUT:LR-C",
                 "WUT:GRU-minus-WUT:LR-C", "WIT:GRU-minus-WIT:LR-C"]
    pairs = {}
    for k in pair_keys:
        pairs[k] = {"MB_strat_native": {x: r4(mbs["native_counts"]["paired"][k]["AUROC"][x]) for x in ("mean", "lo", "hi")},
                    "MB_cluster_UT": {x: r4(mbu["paired"][k]["AUROC"][x]) for x in ("mean", "lo", "hi")},
                    "P0_strat": {x: r4(st["paired"][k]["AUROC"][x]) for x in ("mean", "lo", "hi")},
                    "P0_cluster_UT": {x: r4(cu["paired"][k]["AUROC"][x]) for x in ("mean", "lo", "hi")}}
    o3 = {"strong_R1": go["outcome1"]["strong_R1"],
          "R4_P0_near_chance_strat": bool(go["R4_P0_transfer"][1] <= 0.5 <= go["R4_P0_transfer"][2] or go["R4_P0_transfer"][0] < 0.6),
          "R4_P0_near_chance_cluster": bool(go["R4_P0_transfer_cluster_UT"][1] <= 0.5 <= go["R4_P0_transfer_cluster_UT"][2])}
    o3["met"] = bool(o3["strong_R1"] and o3["R4_P0_near_chance_strat"])
    R["m1"] = {"model_balanced_native_counts": mb_tab, "eligible_models": pt["MB_eligible"], "n_pos": pt["MB_n_pos"],
               "n_neg": pt["MB_n_neg"], "weights_by_model": pt["MB_weights_by_model"],
               "paired": pairs,
               "p10_subsample_v03": {m: [r4(x) for x in mbs["p10_subsample"]["ci"][m]["AUROC"]] for m in names},
               "GRU_outcome1": {"assessment": go["outcome1"], "R1_AUROC": go["R1"]["AUROC"],
                                "R3": go["R3"], "R3_folds_degradation_detected": go["R3_folds_degradation_detected"],
                                "R4_P0_transfer_strat": go["R4_P0_transfer"],
                                "R4_P0_transfer_cluster_UT": go["R4_P0_transfer_cluster_UT"],
                                "R4_MB_native_transfer_strat_bootstrap_mean": go["R4_MB_native_transfer"],
                                "R4_MB_native_transfer_weighted_point": go["R4_MB_native_transfer_weighted_point"],
                                "R4_MB_native_transfer_cluster_UT": go["R4_MB_native_transfer_cluster_UT"],
                                "R4_MB_p10_transfer_gru_ladder_mirror": go["R4_MB_p10_transfer_gru_ladder_mirror"],
                                "R4_MB_native_within_strat": go["R4_MB_native_within"]},
               "GRU_outcome2": go["outcome2"], "GRU_outcome3": o3, "GRU_within_AgentDojo_check": go["within_AgentDojo_check"],
               "sources": {"strat": "alldet_mb_strat (r4_control.py convention: positives drawn with weights, estimate = "
                                    "bootstrap mean, seed 12345)", "cluster": "alldet_mb_cluster_UT/IT (fixed weights, "
                                                                             "clusters resampled)"}}
    R["m2"] = {"strat": st, "cluster_UT": cu, "cluster_IT": ciT, "point": pt["P0"]}
    R["m17_3"] = {"point": {m: {k: pt["P0"][m][k] for k in ("AP_native", "APw10", "APw05")} for m in names},
                  "cluster_UT_ci": {m: {k: [r4(x) for x in cu["ci"][m][k][1:]] for k in ("APw10", "APw05")} for m in names},
                  "cluster_IT_ci": {m: {k: [r4(x) for x in ciT["ci"][m][k][1:]] for k in ("APw10", "APw05")} for m in names},
                  "subsample_v03": {lab: {"n_pos": sub[lab]["n_pos"], "n_neg": sub[lab]["n_neg"],
                                          "ci": {m: [r4(x) for x in sub[lab]["ci"][m]["AP_native"]] for m in names}}
                                    for lab in ("p10", "p05")},
                  "paired_APw_cluster_UT": {k: {mm: {x: r4(v[mm][x]) for x in ("mean", "lo", "hi")} for mm in ("APw10", "APw05")}
                                            for k, v in cu["paired"].items()},
                  "chance": {"APw10": 0.10, "APw05": 0.05}}
    # ------------------------------------------------------------------ rest
    R["m7_Q4"] = _j("dos")
    R["Q2"] = _j("templates")
    R["m6"] = _j("unk")
    R["m16"] = {k: go[k] for k in ("seed_spread", "seed_sign_consistency", "outcome4")}
    R["m15"] = _j("probe")
    R["deviations"] = DEVIATIONS
    jdump(R, RESULTS)
    write_summary(R)
    log(f"assembled {RESULTS} and {SUMMARY}")


DEVIATIONS = [
    "R4D1 (seeds of new within-AgentDojo GRU fits). gru_WIT, gru_WUT_F, gru_C1_UT, gru_C1_IT, gru_noDoS_UT and "
    "gru_WUT_noUNK follow gru_ladder.py (one training seed per CV repetition: fold seed = training seed = repetition 0-4; "
    "gru_ladder.stratified_val(train, repetition)), not the preregistered 3 seeds (same as cache deviation G2).",
    "R4D2 (AUPRC@10% in the Case rule). R4_CONTROL.md Sec. 7 uses 'AUPRC@10%'. The plan procedure uses the plan's "
    "prevalence-weighted AP (point, cluster lower bound); the v0.3 procedure uses the v0.3 subsample estimator. Both are "
    "reported. For the within-model strata no subsample AP exists, so both procedures use the weighted AP there.",
    "R4D3 (Case A/B transfer condition). 'The transfer is near chance' is applied to both P0 and C1 (the text does not say "
    "which population); the conclusion is the same with P0 only for every detector.",
    "R4D4 (near-chance rule). The pre-registered rule (CI includes 0.5, or AUROC < 0.60) is applied literally, including its "
    "one-sided '< 0.60' clause (review M6(d)); every configuration flags whether near chance holds only through that clause. "
    "Under the ATTEMPT relabelling, transfer AUROCs below 0.5 with intervals excluding 0.5 count as 'near chance' by this "
    "rule.",
    "R4D5 (Outcome 4 criteria). GRU_PREREGISTRATION.md does not operationalise 'large seed variance'. Criteria C1-C4 are "
    "defined here, post hoc (results m16.outcome4.criteria_definitions).",
    "R4D6 (attempt target, not in the plan). M6 action 6 asks for the Case decision under an 'attempt' target; the ATTEMPT "
    "configurations relabel the 4,987 unsuccessful attacks as positives at evaluation only (models unchanged). Exploratory.",
    "R4D7 (model-identity probe design, m15). Nested CV: outer stratified 5-fold x 5 repetitions (shuffle seeds 0-4), inner "
    "stratified 5-fold grid over C in {1e-3, 1e-2, 0.1, 1, 10, 100}, L2 logistic regression with class_weight balanced "
    "and StandardScaler, inner scoring AUROC; floors = majority (AUROC 0.5) and suite one-hot (the AgentDojo 'domain'). "
    "Primary metric = pooled out-of-fold AUROC; fold-mean AUROC is stored as a secondary metric.",
    "R4D8 (model-balanced cluster bootstrap). New: fixed full-pool model-balance weights with suite|user_task (or IT) "
    "clusters resampled. The r4_control.py convention (weighted positive draws, bootstrap-mean estimate) is also reported.",
    "R4D9 (m6 exclusion rule). A P0 test trace is 'UNK-bearing' when any of its tools is outside the vocabulary of the fold "
    "model that scored it (cache cov_model < 1). The noUNK-train arm removes UNK-bearing traces from the 90% fit partition "
    "only; the 10% early-stopping partition is unchanged.",
    "R4D10 (failed-attack flag-rate intervals). Trajectory-level bootstrap and a suite|user_task cluster bootstrap over the "
    "4,987 unsuccessful attacks (94 clusters), B = 2000, seed 4242; one repetition / seed drawn per replicate.",
    "R4D11 (DoS removal). Primary: the 101 DoS positives removed at evaluation (models unchanged). Sensitivity: within-UT "
    "models refitted without them (lin_noDoS_UT, gru_noDoS_UT). P0F without DoS also drops the 252 unsuccessful DoS runs.",
    "R4D12 (Q2 design). Per-template transfer within gpt-4o-2024-05-13 uses that model's 92 benign traces as the negatives "
    "of every template; trajectory-level stratified intervals. The pooled template-restricted designs add a cluster "
    "bootstrap with positives by suite|user_task|injection_task (the review's clustering) and negatives by suite|user_task.",
    "R4D13 (degenerate cluster draws). A cluster replicate that contains no positive or no negative returns NaN and is "
    "dropped from the percentiles; this occurs only for the claude-3-5-sonnet stratum (7 positives), which is outside the "
    "model-controlled analyses.",
    "R4D14 (v0.3 values the v0.3 scripts never produced). v0.3 per-model LR-A and GRU transfer, and v0.3 WIT LR-A, are "
    "recomputed on archive_v0.3/agentdojo_traces.jsonl with the v0.3 code paths (ablation.fit; gru_detector.train_gru "
    "refitted deterministically, checked bit-identical against the cache on corrected P0). The GRU was not run under WIT "
    "in v0.3.",
    "R4D15 (Holm p-values). Bootstrap two-sided p = 2 min(P(d <= 0), P(d >= 0)); Holm within the five representations of a "
    "configuration (plan 3.6 d) and within the 12 within-capable detectors for the all-detector tables."]


# ----------------------------------------------------------------------------------------- summary
def F(t, d=3):
    """'x [lo, hi]' for a triple, 'x' for a scalar."""
    if t is None:
        return "n/a"
    if isinstance(t, dict) and "mean" in t:
        t = [t["mean"], t["lo"], t["hi"]]
    if isinstance(t, (list, tuple)):
        if len(t) == 3:
            return f"{t[0]:.{d}f} [{t[1]:.{d}f}, {t[2]:.{d}f}]" if t[0] is not None else "n/a"
        if len(t) == 2:
            return f"[{t[0]:.{d}f}, {t[1]:.{d}f}]"
    if isinstance(t, bool):
        return "yes" if t else "no"
    if isinstance(t, (int,)) and not isinstance(t, bool):
        return str(t)
    return f"{t:.{d}f}"


def S(x, d=3):
    """signed triple"""
    if isinstance(x, dict):
        x = [x["mean"], x["lo"], x["hi"]]
    return f"{x[0]:+.{d}f} [{x[1]:+.{d}f}, {x[2]:+.{d}f}]"


def YN(b):
    return "yes" if b else "no"


def write_summary(R):
    L = []
    A = L.append
    A("# R4 deep: cross-benchmark transfer on the corrected AgentDojo extraction")
    A("")
    A(f"Generated by `revision/r4_deep.py assemble` on {R['meta']['assembled']} from `revision/r4_deep_results.json` "
      "(every number below is a rounded value of a key of that file; the key is given under each table). "
      "Review comments: M6, M7, m1, m2 (AgentDojo), m6 (within-AgentDojo `<UNK>`), m7, m15 (model probe), m16, "
      "m17 item 3, Q2, Q4.")
    A("")
    A("**Conventions.** Point estimates are computed on the full population (within-AgentDojo: mean of 5 CV repetitions; "
      "GRU transfer: mean of 3 seeds). Unless a table says otherwise, the 95% interval is the plan's cluster bootstrap "
      "(B = 2000; UT designs resample suite|user_task clusters, 117 in P0; IT designs resample positives by "
      "suite|injection_task (30 clusters) and negatives by suite|user_task). 'strat' marks the v0.3 trajectory-level "
      "stratified bootstrap (seed 12345). APw10 / APw05 = prevalence-weighted AUPRC at 10% / 5% (plan 3.1; chance 0.10 / "
      "0.05). theta_r80 is set on training data only. 'Detected' = interval excludes 0 (differences) or 0.5 (AUROC). "
      f"The v0.3 scripts re-run on the corrected data are final (`meta.rerun_v03_scripts_final` = "
      f"{R['meta']['rerun_v03_scripts_final']}).")
    A("")
    # ---------------------------------------------------------------- M7
    A("## M7. v0.3 (buggy extraction) versus corrected values")
    A("")
    A("**Question.** How do the R4 numbers change once tool messages are paired with calls by position and observations "
      "are no longer cut at 2,000 characters, and do the re-run v0.3 scripts agree with the shared cache?")
    A("")
    T = R["M7"]["table"]
    A("| Quantity | Detector | v0.3 (archive) | corrected |")
    A("|---|---|---|---|")
    qn = [("transfer_AUROC", "R4 transfer AUROC"), ("WUT_AUROC", "within-UT AUROC"), ("WIT_AUROC", "within-IT AUROC"),
          ("WUT_minus_transfer", "WUT minus transfer (paired)"), ("WIT_minus_transfer", "WIT minus transfer (paired)"),
          ("transfer_FPR@t_r80", "transfer FPR at theta_r80")]
    for q, lab in qn:
        for d in ("LR-A", "LR-B", "LR-C", "GRU"):
            v0 = T[d]["v03"].get(q); v1 = T[d]["corrected"].get(q)
            f0 = (S(v0) if "minus" in q else F(v0)) if v0 is not None else "not run in v0.3"
            f1 = S(v1) if "minus" in q else F(v1)
            A(f"| {lab} | {d} | {f0} | {f1} |")
    for mdl in sorted(T["LR-C"]["corrected"]["per_model_transfer"]):
        for d in ("LR-A", "LR-B", "LR-C", "GRU"):
            A(f"| per-model transfer AUROC, {mdl} | {d} | {F(T[d]['v03']['per_model_transfer'][mdl])} | "
              f"{F(T[d]['corrected']['per_model_transfer'][mdl])} |")
    A("")
    A("Keys: `M7.table.<detector>.v03|corrected.<quantity>`. Intervals: v0.3 trajectory-level stratified bootstrap in "
      "the method order of the v0.3 script that produced the value (r4_control.py for LR-B/LR-C, ablation.py for LR-A, "
      "gru_ladder.py for the GRU), so both columns use the same estimator. Sources of v0.3 values that no v0.3 script "
      "produced (WIT LR-A, per-model LR-A and GRU) are in `M7.table.<detector>.v03.sources` (recomputed on the archived "
      "extraction with the v0.3 code paths).")
    A("")
    rc = R["M7"]["reconciliation_with_rerun_v03_scripts"]
    tr = R["M7"]["truncation_check_linear_cache"]
    A(f"**Reconciliation with the re-run v0.3 scripts** (`M7.reconciliation_with_rerun_v03_scripts`): the cache-based "
      f"values reproduce `r4_control_results.json`, `ablation_results.json` and `gru_results.json` (re-run on the corrected "
      f"file) with a maximum absolute difference of {rc['max_abs_diff_point_and_ci_metrics']:.4f} for points and interval "
      f"bounds (P0, per model, model-balanced), {rc['max_abs_diff_paired_AUROC_deltas']:.5f} for paired AUROC deltas and "
      f"{rc['max_abs_diff_C1_points']:.5f} for the C1 points of the new C1 fits. The v0.3 values recomputed on the archived "
      f"extraction match the archive to {rc['v03_recomputed_vs_archive_max_abs_diff']:.4f}.")
    A("")
    A(f"**Truncation diagnostic** (`M7.truncation_check_linear_cache`, from the linear cache): with the 2,000-character "
      f"cut re-applied to the corrected records, within-UT LR-C is {F(tr['truncated_2000']['withinUT_LR-C']['AUROC'])} "
      f"and LR-B {F(tr['truncated_2000']['withinUT_LR-B']['AUROC'])} (strat), the review's corrected estimates; without the "
      f"cut they are {F(tr['untruncated_corrected']['withinUT_LR-C']['AUROC'])} and "
      f"{F(tr['untruncated_corrected']['withinUT_LR-B']['AUROC'])}.")
    A("")
    lc = T["LR-C"]; lb = T["LR-B"]; gg = T["GRU"]
    A("**Reading.** LR-A uses only the bag of tools, which the fix does not touch, so its values are identical in both "
      "columns. The fix raises pooled transfer AUROC for the observation-using detectors (LR-C "
      f"{lc['v03']['transfer_AUROC'][0]:.3f} to {lc['corrected']['transfer_AUROC'][0]:.3f}, GRU "
      f"{gg['v03']['transfer_AUROC'][0]:.3f} to {gg['corrected']['transfer_AUROC'][0]:.3f}); the stratified intervals of "
      "LR-B, LR-C and the GRU now include 0.5. It lowers within-AgentDojo AUROC (LR-C WUT "
      f"{lc['v03']['WUT_AUROC'][0]:.3f} to {lc['corrected']['WUT_AUROC'][0]:.3f}, WIT {lc['v03']['WIT_AUROC'][0]:.3f} to "
      f"{lc['corrected']['WIT_AUROC'][0]:.3f}; LR-B WUT {lb['v03']['WUT_AUROC'][0]:.3f} to "
      f"{lb['corrected']['WUT_AUROC'][0]:.3f}; GRU WUT {gg['v03']['WUT_AUROC'][0]:.3f} to "
      f"{gg['corrected']['WUT_AUROC'][0]:.3f}). Most of the linear within-UT drop comes from removing the 2,000-character "
      "cut rather than from the pairing fix (truncation diagnostic). The within-minus-transfer gap shrinks for LR-B, "
      "LR-C and the GRU but stays large and detected for every detector. The per-model transfer values of the two Gemini models fall towards 0.5 "
      f"(LR-C gemini-2.0-flash {lc['v03']['per_model_transfer']['gemini-2.0-flash-001'][0]:.3f} to "
      f"{lc['corrected']['per_model_transfer']['gemini-2.0-flash-001'][0]:.3f}), so the per-model spread that the "
      "manuscript attributed to model composition was partly an artefact of the Gemini mis-pairing. The LR-C transfer "
      f"FPR at theta_r80 falls from {lc['v03']['transfer_FPR@t_r80'][0]:.3f} to {lc['corrected']['transfer_FPR@t_r80'][0]:.3f}. "
      "The numbers do not show that transfer works: every corrected pooled transfer AUROC is within 0.07 of 0.5.")
    A("")
    # ---------------------------------------------------------------- M6
    D = R["M6"]["decisions"]
    A("## M6. Pre-registered Case decision per representation and population")
    A("")
    A("**Question.** Applying R4_CONTROL.md Sec. 7 literally, including its clause 'Mixed or other outcomes are reported "
      "as observed', which Case holds for each representation, under UT and IT, on P0, C1, model-balanced, with "
      "unsuccessful attacks as training negatives (WUT_F) and with unsuccessful attacks as evaluation negatives only (P0F)?")
    A("")
    A(f"Rule (`M6.decisions.rule_text`): reasonable = {D['rule_text']['reasonable']}; near chance = "
      f"{D['rule_text']['near_chance']}.")
    A("")
    A("| Representation | Case (plan procedure: cluster bootstrap, weighted AP) | Case (v0.3 procedure: stratified bootstrap, subsample AP) | controls reasonable (plan) |")
    A("|---|---|---|---|")
    for d in MAIN5:
        A(f"| {d} | {D['plan'][d]['case_literal']} | {D['v03'][d]['case_literal']} | "
          f"{', '.join(D['plan'][d]['conditions']['C_controls_reasonable']) or 'none'} |")
    A("")
    A("Keys: `M6.decisions.plan|v03.<detector>.case_literal`, `.conditions`, `.within_model_strata_UT`.")
    A("")
    cfg_order = [("P0_UT", "P0, UT"), ("C1_UT", "C1, UT"), ("MB_UT", "model-balanced, UT"),
                 ("WUTF_P0", "WUT_F (failed as training negatives), P0 evaluation"),
                 ("P0F_UT", "failed as evaluation negatives only (P0F), UT"),
                 ("WUTF_P0F", "WUT_F, P0F evaluation"),
                 ("P0_IT", "P0, IT"), ("C1_IT", "C1, IT"), ("MB_IT", "model-balanced, IT"),
                 ("P0F_IT", "P0F, IT"), ("ATTEMPT_UT", "ATTEMPT (exploratory), UT"), ("ATTEMPT_IT", "ATTEMPT (exploratory), IT")]
    A("| Configuration | Detector | n pos / neg | within AUROC | within APw10 | transfer AUROC | within minus transfer | reasonable | near chance | reading |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for n, lab in cfg_order:
        c = R["M6"]["configs"][n]
        for d in MAIN5:
            x = D["plan"][d]["per_config"][n]
            w = f"within:{d}"; t = f"transfer:{d}"
            wa = _tri_ci(c["point"][w]["AUROC"], c["cluster"]["ci"][w]["AUROC"])
            wp = _tri_ci(c["point"][w]["APw10"], c["cluster"]["ci"][w]["APw10"])
            ta = _tri_ci(c["point"][t]["AUROC"], c["cluster"]["ci"][t]["AUROC"])
            dl = c["cluster"]["paired"][f"{w}-minus-{t}"]["AUROC"]
            nc = YN(x["transfer_near_chance"]) + (" (only via < 0.60)" if x["near_chance_only_via_below_0.60"] else "")
            A(f"| {lab} | {d} | {c['n_pos']} / {c['n_neg']} | {F(wa)} | {F(wp)} | {F(ta)} | {S(dl)} | "
              f"{YN(x['within_reasonable'])} | {nc} | {D['plan'][d]['per_population_reading'][n].split(' (')[0]} |")
    A("")
    A("Keys: `M6.configs.<configuration>.point[\"within:<d>\"|\"transfer:<d>\"]`, `.cluster.ci`, "
      "`.cluster.paired[\"within:<d>-minus-transfer:<d>\"].AUROC`; decisions in `M6.decisions.plan.<d>.per_config.<configuration>`. "
      "Stratified intervals and subsample AP@10% (v0.3 procedure) are in `.strat_native` and `.strat_subsample`. "
      "Holm-adjusted p-values for within minus transfer (family = five representations per configuration) are in "
      "`M6.decisions.holm_within_minus_transfer_cluster`.")
    A("")
    hol = D["holm_within_minus_transfer_cluster"]["by_config"]
    A("Holm (plan 3.6 d), within minus transfer detected after correction: " + "; ".join(
        f"{n}: " + ", ".join(d for d in MAIN5 if hol[n][d]["detected_holm"]) for n, _ in cfg_order[:10]) + ".")
    A("")
    # flags
    fl = R["M6"]["failed_flag_rates"]
    A(f"**Failed-attack flag rates at theta_r80** ({fl['n_failed']} unsuccessful attacks; interval = suite|user_task "
      f"cluster bootstrap over {fl['n_failed_clusters']} clusters). Benign FPR and compromised recall at the same threshold "
      "are shown for reference.")
    A("")
    A("| Detector | transfer (R4): failed / benign FPR / recall | WUT | WIT | WUT_F |")
    A("|---|---|---|---|---|")
    for d in ALL_TRANSFER:
        cells = []
        for stn in ("R4", "WUT", "WIT", "WUT_F"):
            e = fl["by_setting"][stn].get(d)
            cells.append("n/a" if e is None else
                         f"{e['failed_flag_rate@t_r80']:.3f} {F(e['ci_cluster_UT'])} / {e['benign_FPR@t_r80']:.3f} / "
                         f"{e['compromised_recall@t_r80']:.3f}")
        A(f"| {d} | " + " | ".join(cells) + " |")
    A("")
    A("Keys: `M6.failed_flag_rates.by_setting.<R4|WUT|WIT|WUT_F>.<detector>` (also trajectory-level intervals `ci_traj`, "
      "rates at theta = 0.5, per model and DoS / non-DoS).")
    A("")
    p0 = R["M6"]["configs"]["P0_UT"]; wf = R["M6"]["configs"]["WUTF_P0F"]; pf = R["M6"]["configs"]["P0F_UT"]
    fa = R["M6"]["failed_flag_rates"]["by_setting"]
    nonc = [fa[s_][d_]["failed_flag_rate@t_r80"] for s_ in ("WUT", "WIT") for d_ in ALL_WITHIN if not d_.startswith("TFIDF")]
    A("**Reading.** On the corrected data the literal rule gives Case A under UT only for the GRU "
      f"(within-UT {p0['point']['within:GRU']['AUROC']:.3f}) and the TF-IDF content detector "
      f"({p0['point']['within:TFIDF']['AUROC']:.3f}). LR-C no longer meets the P0 bar: its within-UT AUROC is "
      f"{p0['point']['within:LR-C']['AUROC']:.3f}, below 0.80 by a small margin, while C1 is reasonable, so the Case B and "
      "Case C conditions hold together and the outcome is mixed. The same holds for LR-A. LR-B is Case B (no population "
      "or stratum reaches the bar). Both procedures (plan and v0.3) give the same decisions. Under IT (unseen attack "
      "goals) only TF-IDF is reasonable; for the GRU the pre-registered mixed clause applies ('UT reasonable but IT not "
      "reasonable'). Pooled transfer from AgentDrift is near chance for every representation on P0, C1 and "
      "model-balanced. For LR-A (all three populations) and TF-IDF (C1) this rests only on the one-sided '< 0.60' clause, "
      "because their intervals exclude 0.5. With unsuccessful attacks as additional evaluation negatives (P0F), transfer "
      f"rises to {pf['point']['transfer:LR-C']['AUROC']:.3f} (LR-C), {pf['point']['transfer:LR-B']['AUROC']:.3f} (LR-B) and "
      f"{pf['point']['transfer:GRU']['AUROC']:.3f} (GRU), with intervals above 0.5 and points above 0.60, so for this "
      f"population the near-chance arm fails (LR-A, {pf['point']['transfer:LR-A']['AUROC']:.3f}, stays below 0.60). The "
      "reason is that the AgentDrift-trained models score unsuccessful attacks lower than benign runs: at theta_r80, "
      f"transfer LR-C flags {fa['R4']['LR-C']['failed_flag_rate@t_r80']:.3f} of unsuccessful attacks against "
      f"{fa['R4']['LR-C']['benign_FPR@t_r80']:.3f} of benign runs. Training within AgentDojo with unsuccessful attacks as "
      f"negatives (WUT_F) does not lift LR-C to the bar on P0 ({R['M6']['configs']['WUTF_P0']['point']['within:LR-C']['AUROC']:.3f}), "
      f"but it does on P0F ({wf['point']['within:LR-C']['AUROC']:.3f}; GRU {wf['point']['within:GRU']['AUROC']:.3f}). "
      "Within minus transfer is detected under UT for every representation and population, also after Holm correction. "
      "Under IT on P0 it is detected before correction for LR-B, LR-C and TF-IDF but not for LR-A or the GRU; after Holm "
      "correction only LR-B and TF-IDF remain. Within AgentDojo (WUT, WIT), every non-content detector flags between "
      f"{min(nonc):.3f} and {max(nonc):.3f} of the unsuccessful attacks at theta_r80; training on them (WUT_F) lowers this "
      f"to {fa['WUT_F']['GRU']['failed_flag_rate@t_r80']:.3f} for the GRU and {fa['WUT_F']['LR-C']['failed_flag_rate@t_r80']:.3f} "
      "for LR-C. The AgentDrift-trained transfer models flag few of them, except the unsupervised Markov and IForest "
      "detectors and the variants without observation length. The numbers do not show that any behavioural "
      "representation detects unseen attack goals within AgentDojo. They show that the Case outcome for the linear "
      f"representations depends on the population, and that for LR-C it turns on margins of 0.003 AUROC and "
      f"{0.30 - p0['point']['within:LR-C']['APw10']:.3f} weighted AP@10%.")
    A("")
    # ---------------------------------------------------------------- m1
    m1 = R["m1"]
    A("## m1. Model-balanced AUROC at native counts, and GRU Outcome 1")
    A("")
    A(f"**Question.** What is AUROC when positives are reweighted to the benign model mix at native counts "
      f"({m1['n_pos']} positives, {m1['n_neg']} negatives; models {', '.join(m1['eligible_models'])}), for every detector "
      "including the GRU; is LR-A better than LR-C; and is the GRU's pre-registered Outcome 1 met as written?")
    A("")
    A("| Method | P0 AUROC | model-balanced weighted point [cluster CI] | model-balanced, r4_control convention: bootstrap mean [strat CI] |")
    A("|---|---|---|---|")
    for m, v in m1["model_balanced_native_counts"].items():
        A(f"| {m} | {v['P0_point']:.3f} | {v['weighted_point']:.3f} {F(v['cluster_ci'])} | {F(v['strat_native_bootstrap_mean_ci'])} |")
    A("")
    A("Keys: `m1.model_balanced_native_counts.<method>` (weights per model in `m1.weights_by_model`; the v0.3 10% "
      "subsample values in `m1.p10_subsample_v03`).")
    A("")
    A("| Paired contrast (AUROC) | P0, strat | P0, cluster UT | model-balanced, strat | model-balanced, cluster UT |")
    A("|---|---|---|---|---|")
    for k, v in m1["paired"].items():
        A(f"| {k} | {S(v['P0_strat'])} | {S(v['P0_cluster_UT'])} | {S(v['MB_strat_native'])} | {S(v['MB_cluster_UT'])} |")
    A("")
    A("Keys: `m1.paired.<contrast>`.")
    A("")
    o1 = m1["GRU_outcome1"]
    A(f"**GRU Outcome 1, literal** (`m1.GRU_outcome1.assessment`; wording: \"{o1['assessment']['wording']}\"): "
      f"R1 GRU AUROC {F(o1['R1_AUROC'])} (lower bound > 0.7: {YN(o1['assessment']['strong_R1'])}); degradation detected in "
      f"{o1['R3_folds_degradation_detected']}/5 R3 folds; R4 P0 transfer {F(o1['R4_P0_transfer_strat'])} strat and "
      f"{F(o1['R4_P0_transfer_cluster_UT'])} cluster; model-balanced at native counts "
      f"{F(o1['R4_MB_native_transfer_strat_bootstrap_mean'])} (r4_control convention, strat) and "
      f"{F(o1['R4_MB_native_transfer_cluster_UT'])} (weighted point, cluster); the v0.3 gru_ladder.py 10% subsample gives "
      f"{F(o1['R4_MB_p10_transfer_gru_ladder_mirror'])}. Outcome 1 met at native counts with the pre-registered "
      f"(stratified) interval: {YN(o1['assessment']['outcome1_met_native_counts_literal'])}; met with the v0.3 10% "
      f"subsample: {YN(o1['assessment']['outcome1_met_p10_as_in_v03'])}; the native-count cluster interval includes 0.5: "
      f"{YN(o1['assessment']['R4_MB_native_cluster_ci_includes_0.5_or_below'])}. Outcome 2 met: "
      f"{YN(m1['GRU_outcome2']['met'])}. Outcome 3 (GRU near chance in R4 while strong in R1) met: "
      f"{YN(m1['GRU_outcome3']['met'])}. Within-AgentDojo check (WUT lower bound > 0.7 with near-chance transfer): "
      f"{YN(m1['GRU_within_AgentDojo_check']['WUT_lower_gt_0.7'] and m1['GRU_within_AgentDojo_check']['transfer_near_chance'])}.")
    A("")
    mbt = m1["model_balanced_native_counts"]
    pa = m1["paired"]["transfer:LR-A-minus-transfer:LR-C"]
    A("**Reading.** Reweighting the positives to the benign model mix moves pooled transfer AUROC by at most a few "
      f"hundredths (LR-C {mbt['transfer:LR-C']['P0_point']:.3f} to {mbt['transfer:LR-C']['weighted_point']:.3f}, GRU "
      f"{mbt['transfer:GRU']['P0_point']:.3f} to {mbt['transfer:GRU']['weighted_point']:.3f}), so model composition does "
      "not explain the near-chance transfer on the corrected data. The GRU's model-balanced interval under the "
      "pre-registered stratified bootstrap lies just above 0.5 "
      f"({F(o1['R4_MB_native_transfer_strat_bootstrap_mean'])}), so Outcome 1 is not met as literally written, by a "
      "margin of under 0.001. The cluster interval, which the plan now makes primary for AgentDojo, includes 0.5. "
      "Reporting Outcome 1 as met therefore depends on the interval chosen. It should be described as 'met under the "
      "cluster bootstrap; missed by 0.001 under the pre-registered stratified bootstrap'. LR-A minus LR-C in transfer is "
      f"{S(pa['P0_cluster_UT'])} on P0 (cluster), {S(pa['MB_strat_native'])} model-balanced with the r4_control.py "
      f"stratified convention and {S(pa['MB_cluster_UT'])} model-balanced with the cluster bootstrap. On corrected data, "
      "therefore, the LR-A advantage survives model balancing under the stratified interval and is not detected only "
      "under the cluster interval, so the manuscript should not cite model balancing as removing it. LR-A is unchanged "
      "by the extraction fix, whereas LR-C moved. LR-A's own transfer is only "
      f"{mbt['transfer:LR-A']['weighted_point']:.3f} (model-balanced), so the difference does not show that tool "
      "identity transfers in a useful way.")
    A("")
    # ---------------------------------------------------------------- m2
    m2 = R["m2"]
    A("## m2. Cluster bootstrap for AgentDojo (transfer, within, within minus transfer)")
    A("")
    A(f"**Question.** How wide are the AgentDojo intervals when the bootstrap respects the suite|user_task clusters (UT: "
      f"{m2['cluster_UT']['n_clusters']['joint']} clusters) or the injection-goal clusters (IT: "
      f"{m2['cluster_IT']['n_clusters']['pos']} positive and {m2['cluster_IT']['n_clusters']['neg']} negative clusters), "
      "and do the within-minus-transfer gaps survive?")
    A("")
    A("| Detector | transfer: strat / cluster UT / cluster IT | WUT: strat / cluster UT | WIT: strat / cluster IT | WUT minus transfer, cluster UT (Holm p) | WIT minus transfer, cluster IT (Holm p) |")
    A("|---|---|---|---|---|---|")
    for d in ALL_TRANSFER:
        t = f"transfer:{d}"
        pt_ = m2["point"][t]["AUROC"]
        cell_t = f"{pt_:.3f} {F(m2['strat']['ci'][t]['AUROC'][1:])} / {F(m2['cluster_UT']['ci'][t]['AUROC'][1:])} / " \
                 f"{F(m2['cluster_IT']['ci'][t]['AUROC'][1:])}"
        if d in ALL_WITHIN:
            w = f"WUT:{d}"; wi = f"WIT:{d}"
            cw = f"{m2['point'][w]['AUROC']:.3f} {F(m2['strat']['ci'][w]['AUROC'][1:])} / {F(m2['cluster_UT']['ci'][w]['AUROC'][1:])}"
            ci_ = f"{m2['point'][wi]['AUROC']:.3f} {F(m2['strat']['ci'][wi]['AUROC'][1:])} / {F(m2['cluster_IT']['ci'][wi]['AUROC'][1:])}"
            du = m2["cluster_UT"]["paired"][f"{w}-minus-{t}"]["AUROC"]; di = m2["cluster_IT"]["paired"][f"{wi}-minus-{t}"]["AUROC"]
            hu = m2["cluster_UT"]["holm_WUT_minus_transfer"][d]; hi = m2["cluster_IT"]["holm_WIT_minus_transfer"][d]
            A(f"| {d} | {cell_t} | {cw} | {ci_} | {S(du)} ({hu:.3f}) | {S(di)} ({hi:.3f}) |")
        else:
            A(f"| {d} | {cell_t} | n/a | n/a | n/a | n/a |")
    A("")
    A("Keys: points `m2.point.<method>.AUROC`; intervals `m2.strat|cluster_UT|cluster_IT.ci.<method>.AUROC` "
      "(mean, lo, hi); deltas `m2.cluster_UT.paired[\"WUT:<d>-minus-transfer:<d>\"]`, "
      "`m2.cluster_IT.paired[\"WIT:<d>-minus-transfer:<d>\"]`; Holm (family of the 12 within-capable detectors) "
      "`m2.cluster_UT.holm_WUT_minus_transfer`, `m2.cluster_IT.holm_WIT_minus_transfer`. Holm p = 0.000 means no "
      "replicate on the other side of 0 (p < 0.0005).")
    A("")
    lcT = m2["cluster_IT"]["ci"]["transfer:LR-C"]["AUROC"]; lcU = m2["cluster_UT"]["ci"]["transfer:LR-C"]["AUROC"]
    lcS = m2["strat"]["ci"]["transfer:LR-C"]["AUROC"]
    A("**Reading.** Clustering widens the intervals, modestly with UT clusters and substantially with IT clusters: LR-C transfer goes from "
      f"{F(lcS[1:])} (strat) to {F(lcU[1:])} (UT clusters) and {F(lcT[1:])} (IT clusters). Under the IT design every "
      "transfer interval includes 0.5 except those of LR-B-noobs, LR-C-noobs, Markov and IForest, which lie above 0.5. "
      "Within-UT minus transfer stays detected for every supervised detector after Holm correction; for Markov and "
      "IForest it is not detected. Within-IT minus transfer, with only 30 positive clusters, is detected after Holm "
      "correction only for LR-B, TF-IDF and TF-IDF-nodigit; for LR-C and the GRU it is not. "
      "The resampling units are the fixed out-of-fold predictions, so these intervals omit training variability "
      "(review m2).")
    A("")
    # ---------------------------------------------------------------- m17.3
    m17 = R["m17_3"]
    A("## m17 item 3. AgentDojo at 5% and 10% prevalence")
    A("")
    A("**Question.** What are the AgentDojo AUPRC values at 5% next to 10% prevalence (plan 3.1 prevalence-weighted "
      "AP, with the v0.3 subsample estimator as secondary)?")
    A("")
    A("| Method | APw10 [cluster CI] | APw05 [cluster CI] | subsample AP, 10% (44 pos) | subsample AP, 5% (21 pos) |")
    A("|---|---|---|---|---|")
    for m in m17["point"]:
        cl = m17["cluster_IT_ci"] if m.startswith("WIT:") else m17["cluster_UT_ci"]
        A(f"| {m} | {m17['point'][m]['APw10']:.3f} {F(cl[m]['APw10'])} | {m17['point'][m]['APw05']:.3f} {F(cl[m]['APw05'])} | "
          f"{F(m17['subsample_v03']['p10']['ci'][m])} | {F(m17['subsample_v03']['p05']['ci'][m])} |")
    A("")
    A("Keys: `m17_3.point.<method>.APw10|APw05`; `m17_3.cluster_UT_ci` (transfer and WUT) / `m17_3.cluster_IT_ci` (WIT); "
      "`m17_3.subsample_v03.p10|p05.ci.<method>` (mean [2.5th, 97.5th percentile] over 2,000 stratified replicates). "
      "Chance = 0.10 and 0.05.")
    A("")
    sup = ("LR-A", "LR-B", "LR-C", "GRU")
    r10 = [m17["point"][f"WUT:{d}"]["APw10"] / 0.10 for d in sup]
    r05 = [m17["point"][f"WUT:{d}"]["APw05"] / 0.05 for d in sup]
    tr05 = [m17["point"][f"transfer:{d}"]["APw05"] for d in ALL_TRANSFER]
    A("**Reading.** At 5% the ranking and the gaps are the same as at 10%. Transfer APw05 ranges from "
      f"{min(tr05):.3f} to {max(tr05):.3f}: at chance for LR-B, LR-C, the GRU and TF-IDF, and slightly above chance for "
      "LR-A, the variants without observation length, Markov and IForest. Within-UT AP of the supervised behavioural "
      f"detectors (LR-A, LR-B, LR-C, GRU) is {min(r10):.1f} to {max(r10):.1f} times chance at 10% and {min(r05):.1f} to "
      f"{max(r05):.1f} times chance at 5%; TF-IDF is near perfect. LR-C's within-UT APw10 "
      f"({m17['point']['WUT:LR-C']['APw10']:.3f}) sits just below the 0.30 criterion of R4_CONTROL.md. The v0.3 subsample "
      "estimator (the expected AP of a test set with 44 or 21 positives) gives slightly higher centres.")
    A("")
    # ---------------------------------------------------------------- m7 / Q4
    dq = R["m7_Q4"]
    A("## m7 and Q4. Short-run filter, DoS label semantics, and R4 with and without DoS positives")
    A("")
    A("**Question.** Is the 'at least two tool calls' filter class-asymmetric, what does a DoS 'success' mean in "
      "AgentDojo, and do the main R4 results change without the 101 DoS positives?")
    A("")
    A("| Class | runs | excluded (< 2 calls) | kept | excluded fraction |")
    A("|---|---|---|---|---|")
    for c, v in dq["exclusion_by_class"].items():
        A(f"| {c} | {v['all']} | {v['excluded_fewer_than_two_calls']} | {v['kept']} | {v['excluded_fraction']:.3f} |")
    A("")
    A("| Model | benign excluded / all | compromised excluded / all | unsuccessful excluded / all |")
    A("|---|---|---|---|")
    for mdl, v in dq["exclusion_by_model"].items():
        A(f"| {mdl} | {v['benign']['excluded']} / {v['benign']['all']} ({v['benign']['excluded_fraction']:.3f}) | "
          f"{v['attacked']['excluded']} / {v['attacked']['all']} ({v['attacked']['excluded_fraction']:.3f}) | "
          f"{v['attack_failed']['excluded']} / {v['attack_failed']['all']} ({v['attack_failed']['excluded_fraction']:.3f}) |")
    A("")
    A(f"Keys: `m7_Q4.exclusion_by_class`, `m7_Q4.exclusion_by_model` (from agentdojo_extract_manifest.json). Trace length "
      f"alone separates compromised from benign P0 traces with AUROC {dq['n_steps_univariate_AUROC_P0']:.3f} "
      "(`m7_Q4.n_steps_univariate_AUROC_P0`).")
    A("")
    ds = dq["dos_semantics"]
    A(f"**DoS label semantics** (AgentDojo commit `{ds['commit'][:8]}`, `src/agentdojo/benchmark.py`; `m7_Q4.dos_semantics`):")
    A("")
    A("```")
    for ln in ds["benchmark.py_73_75"] + ["..."] + ds["benchmark.py_149_150"]:
        A(ln)
    A("```")
    A("")
    dc = dq["dos_counts"]
    A(f"The DoS attack classes set `is_dos_attack = True` (`src/agentdojo/attacks/dos_attacks.py`, class DoSAttack). "
      f"For these attacks only the first injection task of the suite is run, so the injection task of a DoS run is a "
      f"placeholder (P0 DoS positives by suite|injection_task: {dc['P0_positives_by_suite_injection_task']}). Security is "
      f"then overwritten with `not utility`: a DoS run counts as a successful attack exactly when the user task was not "
      f"completed, whatever the agent did. Consistently, all {dc['P0_positives']} P0 DoS positives have utility False "
      f"({dc['P0_positives_utility_false']}), and all {dc['failed']} unsuccessful DoS runs have utility True "
      f"({dc['failed_utility_true']}). The three exception paths (context length, internal server error, server error) "
      f"score utility False and security True (lines 129-130, 138-139, 146-147), but none of the kept runs has an error "
      f"field ({dq['raw_run_error_field']}). All DoS positives come from {list(dc['P0_positives_models'])[0]}.")
    A("")
    A("| Configuration | n pos / neg | Detector | within AUROC [cluster CI] | transfer AUROC [cluster CI] | within minus transfer |")
    A("|---|---|---|---|---|---|")
    labs = {"P0_UT": "P0, UT (with DoS)", "P0noDoS_UT": "P0 without DoS positives, UT (models unchanged)",
            "P0noDoS_UT_refit": "P0 without DoS, UT, within models refitted without DoS",
            "P0_IT": "P0, IT (with DoS)", "P0noDoS_IT": "P0 without DoS positives, IT",
            "P0F_UT": "P0F, UT (with DoS)", "P0FnoDoS_UT": "P0F without any DoS row, UT"}
    for k, lab in labs.items():
        v = dq["with_without_dos"][k]
        for d in MAIN5:
            w = f"within:{d}"; t = f"transfer:{d}"
            A(f"| {lab} | {v['n_pos']} / {v['n_neg']} | {d} | {v['point'][w]['AUROC']:.3f} {F(v['cluster_ci'][w]['AUROC'][1:])} | "
              f"{v['point'][t]['AUROC']:.3f} {F(v['cluster_ci'][t]['AUROC'][1:])} | {S(v['cluster_paired_AUROC'][f'{w}-minus-{t}'])} |")
    A("")
    A("Keys: `m7_Q4.with_without_dos.<configuration>.point|cluster_ci|cluster_paired_AUROC` (APw10 in `cluster_ci`); "
      "recall of DoS versus non-DoS positives at theta_r80 in `m7_Q4.recall_dos_vs_non_dos`.")
    A("")
    rr = dq["recall_dos_vs_non_dos"]
    w0 = dq["with_without_dos"]["P0_UT"]["point"]; w1 = dq["with_without_dos"]["P0noDoS_UT"]["point"]
    A("**Reading.** The short-run filter removes 37.6% of benign runs but 3.0% of compromised runs (all of them "
      "gpt-4o), and 58.3% of gemini-2.0-flash benign runs. Because trace length alone has AUROC "
      f"{dq['n_steps_univariate_AUROC_P0']:.3f}, the filter shapes the P0 negatives in a class- and model-dependent way. "
      "The DoS positives are labelled by task failure, not by any action of the attacker's choosing, so they are a "
      "different target from action hijacks. Behavioural detectors catch them less often (within-UT LR-C recall at "
      f"theta_r80 {rr['WUT:LR-C']['recall_dos']:.3f} for DoS versus {rr['WUT:LR-C']['recall_non_dos']:.3f} otherwise). "
      f"Removing them raises within-UT AUROC slightly (LR-C {w0['within:LR-C']['AUROC']:.3f} to "
      f"{w1['within:LR-C']['AUROC']:.3f}, GRU {w0['within:GRU']['AUROC']:.3f} to {w1['within:GRU']['AUROC']:.3f}). Without "
      "DoS, LR-C within-UT reaches the 0.80 point criterion, which it misses with DoS. Transfer is essentially unchanged. "
      "The conclusions about transfer do not depend on the DoS positives. The LR-C Case outcome is sensitive to them, "
      "because it sits at the 0.80 boundary.")
    A("")
    # ---------------------------------------------------------------- Q2
    q2 = R["Q2"]
    A("## Q2. Attack templates within gpt-4o-2024-05-13 and the important_instructions design")
    A("")
    A("**Question.** Does transfer differ by attack template within the one model that was run with many templates, and "
      "what does R4 look like when positives are restricted to the important_instructions template (the only template "
      "run for every model), pooled and model-balanced?")
    A("")
    A("| Template (gpt-4o) | n pos (neg 92) | DoS | LR-A | LR-B | LR-C | GRU | TFIDF |")
    A("|---|---|---|---|---|---|---|---|")
    for t, e in q2["per_template_gpt4o"].items():
        A(f"| {t} | {e['n_pos']} | {YN(e['is_dos'])} | " + " | ".join(F(e[f"transfer:{d}"]) for d in MAIN5) + " |")
    A("")
    A("Transfer AUROC with 95% stratified intervals. Keys: `Q2.per_template_gpt4o.<template>.transfer:<d>` (within-UT "
      "values under `WUT:<d>`), range `Q2.per_template_range_transfer`, families `Q2.per_family_gpt4o`.")
    A("")
    A("| Design | Detector | pooled transfer [strat] | pooled transfer [cluster UTI] | model-balanced transfer: weighted point, bootstrap mean [strat] | pooled WUT [strat] | model-balanced WUT minus transfer [strat] |")
    A("|---|---|---|---|---|---|---|")
    for lab in ("II_only", "II_family"):
        e = q2[lab]
        for d in MAIN5:
            t = f"transfer:{d}"; w = f"WUT:{d}"
            A(f"| {lab} ({e['n_pos']} pos / {e['n_neg']} neg) | {d} | {e['pooled']['point'][t]['AUROC']:.3f} "
              f"{F(e['pooled']['strat_ci'][t]['AUROC'][1:])} | {F(e['cluster_UTI']['ci'][t]['AUROC'][1:])} | "
              f"{e['model_balanced_native']['weighted_point'][t]:.3f}, {F(e['model_balanced_native']['ci'][t]['AUROC'])} | "
              f"{e['pooled']['point'][w]['AUROC']:.3f} {F(e['pooled']['strat_ci'][w]['AUROC'][1:])} | "
              f"{S(e['model_balanced_native']['paired'][f'{w}-minus-{t}']['AUROC'])} |")
    A("")
    A("Keys: `Q2.II_only|II_family.pooled`, `.cluster_UTI` (positives clustered by suite|user_task|injection_task, negatives "
      "by suite|user_task), `.cluster_UT`, `.model_balanced_native`; per model within important_instructions in "
      "`Q2.II_only_per_model`. II_family = the six important_instructions variants (1,455 gpt-4o positives).")
    A("")
    rg = q2["per_template_range_transfer"]
    A(f"**Reading.** Within gpt-4o, transfer AUROC varies by template from {rg['LR-C'][0]:.3f} to {rg['LR-C'][1]:.3f} "
      f"(LR-C) and {rg['GRU'][0]:.3f} to {rg['GRU'][1]:.3f} (GRU). For LR-C, the five DoS templates, tool_knowledge and "
      "four of the six important_instructions variants sit below 0.5, while direct, ignore_previous, injecagent and the "
      "two 'wrong name' variants sit above it. LR-A is above 0.5 for most non-DoS templates. The per-template intervals are wide, and the "
      "benign negatives (92) are shared, so the templates are not independent comparisons. When positives are "
      "restricted to important_instructions for every model, pooled transfer is "
      f"{q2['II_only']['pooled']['point']['transfer:LR-C']['AUROC']:.3f} (LR-C) and "
      f"{q2['II_only']['pooled']['point']['transfer:GRU']['AUROC']:.3f} (GRU), still near chance. Within-UT AUROC on "
      f"the same design is {q2['II_only']['pooled']['point']['WUT:LR-C']['AUROC']:.3f} (LR-C) and "
      f"{q2['II_only']['pooled']['point']['WUT:GRU']['AUROC']:.3f} (GRU). The model-template confound therefore does not "
      "create the near-chance pooled transfer. The template mix does drive part of the spread across subgroups.")
    A("")
    # ---------------------------------------------------------------- m6
    u = R["m6"]; wu = u["WUT_GRU_without_unk"]; fs = u["fit_partitions_summary"]
    A("## m6. `<UNK>` in the within-AgentDojo GRU")
    A("")
    A("**Question.** How is the GRU vocabulary built, how often does `<UNK>` occur in the training partitions actually "
      "passed to `train_gru` within AgentDojo, and how does the WUT GRU do on test traces without `<UNK>`?")
    A("")
    A("`gru_detector.build_tool_vocab` (`m6.build_tool_vocab_source`):")
    A("")
    A("```python")
    A(u["build_tool_vocab_source"])
    A("```")
    A("")
    A(f"The vocabulary comes from benign training trajectories only (`m6.vocab_from_benign_only` = "
      f"{u['vocab_from_benign_only']}). Over the 25 WUT fit partitions (90% of each training fold, after "
      f"gru_ladder.stratified_val): `<UNK>` steps min {fs['unk_steps']['min']}, max {fs['unk_steps']['max']}, mean "
      f"{fs['unk_steps']['mean']}; traces with `<UNK>` min {fs['unk_traces']['min']}, max {fs['unk_traces']['max']}, mean "
      f"{fs['unk_traces']['mean']}. All of them are compromised traces (`unk_traces_pos` equals `unk_traces`), since benign "
      f"fit traces define the vocabulary. Benign-only vocabulary size is {fs['vocab_benign_only']['min']}-{fs['vocab_benign_only']['max']} "
      f"tools, against {fs['vocab_if_all_fit_traces']['min']}-{fs['vocab_if_all_fit_traces']['max']} if every fit trace "
      f"were used. The tools that occur only in compromised P0 traces ({', '.join(u['tools_only_in_attacked_P0'])}) appear "
      f"in {u['n_traces_with_attack_only_tools']} traces. Encoded (last 16 steps) and untruncated counts agree, and they "
      f"match the cache (`m6.cache_matches_recount_encoded` = {u['cache_matches_recount_encoded']}). `<UNK>` embedding "
      f"norms after training: `m6.cache_unk_norm`.")
    A("")
    A("| Model | all P0 test traces | test traces without `<UNK>` |")
    A("|---|---|---|")
    for k, lab in (("WUT_GRU", "WUT GRU (cache)"), ("WUT_noUNKtrain_GRU", "WUT GRU refitted without `<UNK>`-bearing fit traces"),
                   ("WUT_LR-C", "WUT LR-C (reference)")):
        A(f"| {lab} | {wu[k]['all_test']:.3f} {F(wu[k]['ci_all_test'])} | {wu[k]['no_unk_test']:.3f} {F(wu[k]['ci_no_unk_test'])} |")
    A("")
    ex = wu["n_excluded_per_rep"]
    A(f"Excluded test traces per repetition (positives / negatives): {', '.join(str(e['pos']) + '/' + str(e['neg']) for e in ex)}. "
      f"Paired contrasts (same replicate): no-UNK test minus all test {S(wu['paired']['GRU_nounk_minus_all'])}; "
      f"refitted minus cache GRU, all test {S(wu['paired']['noUNKtrain_minus_GRU_all'])}; GRU minus LR-C on no-UNK test "
      f"{S(wu['paired']['GRU_minus_LRC_nounk'])}. Intervals: stratified bootstrap of P0 (B = 2000) with the drawn "
      "repetition's exclusion mask. Keys: `m6.WUT_GRU_without_unk`, per-partition counts in `m6.fit_partitions`.")
    A("")
    A("**Reading.** The manuscript's statement that `<UNK>` never occurs in a training step holds for AgentDrift but not "
      f"within AgentDojo, where up to {fs['unk_steps']['max']} steps in {fs['unk_traces']['max']} compromised fit traces "
      "map to `<UNK>`. They come from tools that happen to be absent from the fold's benign partition. Only "
      f"{u['n_traces_with_attack_only_tools']} traces use tools that never occur in any benign P0 trace. Refitting without "
      f"the `<UNK>`-bearing fit traces ({S(wu['paired']['noUNKtrain_minus_GRU_all'])}) or excluding `<UNK>`-bearing test "
      f"traces ({S(wu['paired']['GRU_nounk_minus_all'])}) changes WUT GRU AUROC by less than 0.01, and neither change is "
      "detected. So `<UNK>`-as-attack-marker does not drive the within-AgentDojo GRU result. "
      "The within-AgentDojo seeds are 0-4 (one per repetition).")
    A("")
    # ---------------------------------------------------------------- m16
    s16 = R["m16"]; o4 = s16["outcome4"]
    A("## m16. Per-seed spread of the GRU in R3 and pre-registered Outcome 4")
    A("")
    A("**Question.** How much do GRU FPR and recall at theta_r80 vary across the three training seeds in R3, and is "
      "Outcome 4 ('inconsistent ... or large seed variance') triggered?")
    A("")
    A("| Setting | Detector | seed 0: AUROC / FPR / recall | seed 1 | seed 2 | range AUROC / FPR / recall | mean single-seed CI width AUROC / FPR / recall |")
    A("|---|---|---|---|---|---|---|")
    for stn, v in s16["seed_spread"].items():
        for det, e in v.items():
            ps = e["per_seed"]
            cells = [f"{p['AUROC']:.3f} / {p['FPR@t_r80']:.3f} / {p['recall@t_r80']:.3f}" for p in ps]
            A(f"| {stn} | {det} | " + " | ".join(cells) + f" | {e['range']['AUROC']:.3f} / {e['range']['FPR@t_r80']:.3f} / "
              f"{e['range']['recall@t_r80']:.3f} | {e['mean_single_seed_ci_width']['AUROC']:.3f} / "
              f"{e['mean_single_seed_ci_width']['FPR@t_r80']:.3f} / {e['mean_single_seed_ci_width']['recall@t_r80']:.3f} |")
    A("")
    A("Keys: `m16.seed_spread.<setting>.<detector>.per_seed|range|sd|mean_single_seed_ci_width|criteria` (pool E; also "
      "theta_val80 rates and failed-attack flag rates per seed).")
    A("")
    A("Criteria (`m16.outcome4.criteria_definitions`; defined here, post hoc, because the preregistration does not "
      "operationalise 'large'): " + "; ".join(f"{k}: {v}" for k, v in o4["criteria_definitions"].items()) + ".")
    A("")
    A(f"R3 folds (GRU, p = 0): C1 sign-consistent in {o4['folds_C1_sign_consistent']}/5; C2 in {o4['folds_C2']}/5; "
      f"C3 in {o4['folds_C3']}/5; C4 in {o4['folds_C4']}/5. Outcome 4 triggered on ranking (C2): "
      f"{YN(o4['triggered_ranking'])}; on the operating point (C3 or C4): {YN(o4['triggered_operating_point'])}. "
      f"Per-seed R3 minus R1 AUROC: `m16.seed_sign_consistency`.")
    A("")
    sb = s16["seed_spread"]["R3_banking"]["GRU"]["per_seed"]
    _cr = [s16["seed_spread"][f"R3_{d}"]["GRU-drop0.2"]["criteria"] for d in ("medical", "banking", "web", "email", "coding")]
    c2d = sum(c["C2_AUROC_range_gt_single_seed_CI_width"] for c in _cr)
    c3d = sum(c["C3_FPR_range_gt_single_seed_CI_width"] or c["C3_recall_range_gt_single_seed_CI_width"] for c in _cr)
    A("**Reading.** The direction of the R3 degradation is the same for all three seeds in every fold, the criterion "
      "GRU_RESULTS.md used. The operating point is not stable: in R3 banking one seed flags almost nothing at "
      f"theta_r80 (FPR {sb[1]['FPR@t_r80']:.3f}, recall {sb[1]['recall@t_r80']:.3f}) while the other two flag most "
      f"traces (FPR {sb[0]['FPR@t_r80']:.3f} and {sb[2]['FPR@t_r80']:.3f}). Across-seed ranges of FPR or recall exceed "
      "the single-seed sampling interval in every fold. The AUROC range also exceeds it in three folds. Under each of "
      "C2, C3 and C4 as a reading of 'large seed variance', Outcome 4 is triggered, and the preregistration then requires the "
      "inconsistency and the seed variance to be reported. With token dropout the AUROC spread shrinks (C2 in "
      f"{c2d}/5 folds at p = 0.2) but the threshold spread remains (C3 in {c3d}/5 folds at p = 0.2). This is an operational finding about thresholds, not a reversal of the ranking conclusions.")
    A("")
    # ---------------------------------------------------------------- m15
    pr = R["m15"]
    A("## m15. Model-identity probe on benign P0 traces (corrected data)")
    A("")
    A(f"**Question.** With a regularized, nested cross-validated probe and majority and suite-only floors, do "
      f"representations B and C encode whether a benign trace came from gpt-4o-2024-05-13 ({pr['n_gpt4o']}) or another "
      f"model ({pr['n_other']})?")
    A("")
    A("| Probe | AUROC (pooled out-of-fold) [strat CI] | fold-mean AUROC (min, max over 25 folds) |")
    A("|---|---|---|")
    for k in ("B", "C", "suite", "majority"):
        fm = pr["fold_mean_AUROC"].get(k)
        A(f"| {k} | {F(pr['AUROC'][k])} | " + (f"{fm['mean']:.3f} ({fm['min']:.3f}, {fm['max']:.3f})" if fm else "0.500") + " |")
    A("")
    A("| Contrast | AUROC difference |")
    A("|---|---|")
    for k, v in pr["paired"].items():
        A(f"| {k} | {S(v)} |")
    A("")
    vv = pr["v03_probe_unregularized"]
    A(f"v0.3 unregularized probe (r4_control.py, C = 1, no nested selection): archive (buggy extraction) C "
      f"{F(vv['archive_v0.3']['full']['AUROC'])}, B {F(vv['archive_v0.3']['agn']['AUROC'])}; re-run on corrected data C "
      f"{F(vv['rerun_on_corrected']['full']['AUROC'])}, B {F(vv['rerun_on_corrected']['agn']['AUROC'])}. Design: "
      f"`m15.design`; selected C values: `m15.chosen_C`. Keys: `m15.AUROC`, `m15.paired`, `m15.fold_mean_AUROC`, "
      f"`m15.v03_probe_unregularized`.")
    A("")
    A("**Reading.** On the corrected data, neither B nor C separates gpt-4o benign traces from other models' benign "
      "traces better than the floors: every probe interval includes 0.5, and B does not beat the suite-only floor. The "
      f"v0.3 finding that B encodes model identity ({vv['archive_v0.3']['agn']['AUROC'][0]:.3f}) disappears on corrected "
      f"data ({vv['rerun_on_corrected']['agn']['AUROC'][0]:.3f} with the same unregularized probe), so it was produced by "
      "the Gemini mis-pairing of observations (M7). This is a failure to detect encoding with 92 against 300 traces, not "
      "a demonstration that model identity is absent. Values below 0.5 for C and for the suite floor are typical of "
      "cross-validated AUROC when there is little signal (the fold-mean values agree), and the probe cannot rule out "
      "weaker encodings.")
    A("")
    # ---------------------------------------------------------------- deviations
    A("## Deviations and implementation notes")
    A("")
    for dv in R["deviations"]:
        A(f"- {dv}")
    A("")
    A("These are also appended to `revision/DEVIATIONS.md` (section 'R4 deep').")
    A("")
    with open(SUMMARY, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))


# =============================================================================================== driver
SECTIONS = collections.OrderedDict([
    ("m7", (section_m7, ["lin_C1_UT", "lin_C1_IT", "lin_noDoS_UT", "gru_WIT", "gru_v03_transfer", "lin_v03"])),
    ("case", (section_case, ["lin_C1_UT", "lin_C1_IT", "gru_WIT", "gru_WUT_F", "gru_C1_UT", "gru_C1_IT"])),
    ("alldet", (section_alldet, ["gru_WIT"])),
    ("flags", (section_flags, ["gru_WIT", "gru_WUT_F"])),
    ("per_model", (section_per_model, ["gru_WIT"])),
    ("gru_outcomes", (section_gru_outcomes, [])),
    ("unk", (section_unk, ["gru_WUT_noUNK"])),
    ("dos", (section_dos, ["lin_noDoS_UT", "gru_noDoS_UT", "gru_WIT"])),
    ("templates", (section_templates, [])),
    ("probe", (section_probe, [])),
])
SINGLE_FILE = {"m7", "flags", "per_model", "gru_outcomes", "unk", "dos", "templates", "probe"}


def section_done(name):
    if name == "case":
        return all(os.path.exists(sec_path(f"case__{n}")) for n in CASE_CONFIGS)
    if name == "alldet":
        return all(os.path.exists(sec_path(j)) for j in ("alldet_point", "alldet_strat", "alldet_cluster_UT", "alldet_cluster_IT",
                                                          "alldet_mb_strat", "alldet_mb_cluster_UT", "alldet_mb_cluster_IT",
                                                          "alldet_sub"))
    return os.path.exists(sec_path(name))


def status():
    for j in JOBS:
        n = sum(os.path.exists(unit_path(j, u)) for u in units(j))
        print(f"fit {j:18s} {n}/{len(units(j))}")
    for s in SECTIONS:
        print(f"section {s:14s} {'done' if section_done(s) else 'missing'}")
    print("results", os.path.exists(RESULTS), "summary", os.path.exists(SUMMARY))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["status", "fit", "analyze", "assemble"])
    ap.add_argument("what", nargs="?", default="all")
    ap.add_argument("--budget", type=float, default=480.0)
    a = ap.parse_args()
    if a.cmd == "status":
        status(); return
    t0 = time.time()
    if a.cmd == "fit":
        todo = list(JOBS) if a.what == "all" else [a.what]
        for j in todo:
            left = a.budget - (time.time() - t0)
            if left <= 0:
                break
            run_fit(j, left)
        status(); return
    if a.cmd == "analyze":
        todo = list(SECTIONS) if a.what == "all" else [a.what]
        for s in todo:
            if section_done(s):
                continue
            fn, needs = SECTIONS[s]
            miss = [j for j in needs if not job_done(j)]
            if miss:
                print(f"section {s}: waiting for fits {miss}"); continue
            left = a.budget - (time.time() - t0)
            if left <= 0:
                break
            t1 = time.time()
            ok = fn(left)
            log(f"section {s} {'done' if ok else 'partial'} in {round(time.time() - t1, 1)} s")
        status(); return
    if a.cmd == "assemble":
        assemble(); return


if __name__ == "__main__":
    main()
