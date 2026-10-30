"""Prediction caches for the SE detector (SENCODER_PREREGISTRATION.md, amendment 3) and the 10-seed GRU re-run.

Families written (new folders; no existing cache file is touched):
  cache/sencoder_fits/<setting>.pkl   detectors SE and SE-W0 (Sec. 12.3), row-aligned with cache/linear/<setting>.pkl
  cache/gru10/<setting>.pkl           detector GRU (p = 0), seeds 0-9, produced by build_gru_cache.py's own
                                      run_unit / assemble / merge_setting (imported; only SEEDS, WUT_REPS and the
                                      output folders are re-bound), plus C1 within-AgentDojo (r4_deep gru_C1_UT logic)

Settings and units
  R1, R1wd_s<k>, R2, R3_<d>, R4: build_gru_cache.setting_data(name); validation = gru_ladder.stratified_val(train, seed)
            at every rung (amendment 2, D7). R1 units also score R1 pool E under every graded-coverage mask and every
            padding level. SE R4 units also score AgentDojo with the four world features zeroed and no refit: the
            feature-block offset probe of amendment 4, reported separately and never as SE-W0.
  WUT       within-AgentDojo UT, rep r = fold seed = training seed, r = 0..9, 5 folds (build_gru_cache.wut_folds);
            fit on P0 rows of the other folds; P0 and failed rows of the fold are scored (P0F comes from these).
  C1        within-AgentDojo UT on C1 (P0 minus the benign runs whose user task is an injection task), folds as
            r4_control.within_oof on C1 (r4_deep.subset_folds), rep r = fold seed = training seed, r = 0..9.
  SE: all settings. SE-W0 (amendment 4): R4, WUT and C1 refits with the world features zeroed in training and test.
  GRU: all settings.

Usage:  python build_sencoder_cache.py run [--workers N] [--threads T] [--only SE|SE-W0|GRU] [--settings ...]
        python build_sencoder_cache.py smoke         # one unit per detector family, timing + determinism check
        python build_sencoder_cache.py assemble
        python build_sencoder_cache.py drift          # embedding-drift diagnostic (plan Sec. 7)
        python build_sencoder_cache.py status
Environment: AGENTDRIFT, SENCODER_EMB (embedding cache dir, default revision/cache/sencoder_gpu).
"""
import os, sys, json, time, pickle, argparse, collections, datetime, hashlib, platform
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)

import torch  # noqa: E402
import evalkit as ek  # noqa: E402
import build_gru_cache as bgc  # noqa: E402  (imported; never modified)
import r4_control  # noqa: E402
import sencoder_detector as sd  # noqa: E402
import sencoder_text as stx  # noqa: E402
from gru_ladder import stratified_val  # noqa: E402
from featurise import load  # noqa: E402

EMB_DIR = os.environ.get("SENCODER_EMB", os.path.join(HERE, "cache", "sencoder_gpu"))
SE_CACHE = os.path.join(HERE, "cache", "sencoder_fits"); SE_PARTS = os.path.join(SE_CACHE, "_parts")
G10_CACHE = os.path.join(HERE, "cache", "gru10"); G10_PARTS = os.path.join(G10_CACHE, "_parts")
LOGF = os.path.join(HERE, "logs", "build_sencoder_cache.log")
STOPF = os.path.join(HERE, "sencoder_run", "STOP_RULE_TRIGGERED.json")
SEEDS = list(range(10)); REPS = list(range(10)); K = 5
DOMAINS = bgc.DOMAINS
SINGLE = ["R1"] + [f"R1wd_s{k}" for k in range(5)] + ["R2"] + [f"R3_{d}" for d in DOMAINS] + ["R4"]
PLAN = {"SE": SINGLE + ["WUT", "C1"],
        "SE-W0": ["R4", "WUT", "C1"],
        "GRU": SINGLE + ["WUT", "C1"]}
FIT_LIMIT_S = 600.0   # plan Sec. 10: a single fit over 10 minutes stops the run

# re-bind the executed GRU pipeline to 10 seeds and new output folders (the module file is unchanged)
bgc.SEEDS = SEEDS; bgc.WUT_REPS = REPS
bgc.CACHE = G10_CACHE; bgc.PARTS = G10_PARTS
bgc.LOGF = os.path.join(HERE, "logs", "build_gru10_cache.log")
bgc.PLAN = {"GRU": SINGLE + ["WUT"]}


def log(msg):
    os.makedirs(os.path.dirname(LOGF), exist_ok=True)
    line = f"{datetime.datetime.now().isoformat(timespec='seconds')} {msg}"
    print(line, flush=True)
    with open(LOGF, "a", encoding="utf-8") as f:
        f.write(line + "\n")


pdump, pload = bgc.pdump, bgc.pload


# ------------------------------------------------------------------ global step table
_T = {}


def records_all():
    st_tr, st_te, td_tr, td_te = bgc.drift()
    st_va = load(os.path.join(bgc.MP, "data/val.jsonl"))
    P0, FA = bgc.dojo()
    return st_tr, st_va, st_te, td_tr, td_te, P0, FA


def table():
    """EMB (N,768) float32, SCAL (N,4) raw world features (Sec. 12.2), rows_of {record id: global step rows}."""
    if "T" in _T:
        return _T["T"]
    st_tr, st_va, st_te, td_tr, td_te, P0, FA = records_all()
    man = json.load(open(os.path.join(EMB_DIR, "manifest.json")))
    assert man["taskdisjoint"]["identical_to_standard"], "task-disjoint texts differ: embed them separately"
    srcs = [("agentdrift_train", st_tr), ("agentdrift_val", st_va), ("agentdrift_test", st_te),
            ("agentdojo", P0), ("agentdojo_failed", FA)]
    embs, scal, rows_of, off = [], [], {}, 0
    for name, recs in srcs:
        z = np.load(os.path.join(EMB_DIR, f"{name}.npz"), allow_pickle=True)   # self-produced cache
        tids = list(z["trace_ids"]); ids = z["ids"]; E = z["emb"]
        assert len(tids) == len(recs) and tids == [r["id"] for r in recs], name
        cnt = np.bincount(ids[:, 0], minlength=len(recs))
        for ti, r in enumerate(recs):
            assert cnt[ti] == len(r["steps"]), (name, r["id"])
        order = np.lexsort((ids[:, 1], ids[:, 0])); assert np.all(order == np.arange(len(ids)))
        start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
        for ti, r in enumerate(recs):
            assert r["id"] not in rows_of, f"duplicate id {r['id']}"
            rows_of[r["id"]] = off + start[ti] + np.arange(cnt[ti])
        embs.append(E); scal.append(np.asarray([stx.world_features(s, stx.world_of(r)) for r in recs for s in r["steps"]], np.float32).reshape(-1, 4))
        off += len(E)
    EMB = np.concatenate(embs).astype(np.float32); SCAL = np.concatenate(scal)
    assert EMB.shape[0] == SCAL.shape[0] == off
    # task-disjoint records reuse the standard rows (identity checked by the embedding job)
    for r in td_tr + td_te:
        assert r["id"] in rows_of and len(rows_of[r["id"]]) == len(r["steps"])
    _T["T"] = (EMB, SCAL, rows_of, man)
    return _T["T"]


def enc(recs):
    EMB, SCAL, rows_of, _ = table()
    return sd.Encoded(recs, lambda r: rows_of[r["id"]])


def variant_tables(fname, keyname):
    """R1 pool-E step table per variant (masks or pads): keys, index (n_var, n_steps), emb, WF_local, teE, loc."""
    k = ("V", fname)
    if k in _T:
        return _T[k]
    z = np.load(os.path.join(EMB_DIR, fname), allow_pickle=True)
    st_tr, st_va, st_te, *_ = records_all()
    teE = [r for r in st_te if r["category"] in ("benign", "hard_negative", "attacked")]
    assert list(z["trace_ids"]) == [r["id"] for r in teE]
    WL = np.asarray([stx.world_features(s, stx.world_of(r)) for r in teE for s in r["steps"]], np.float32)
    loc, o = {}, 0
    for r in teE:
        loc[r["id"]] = o + np.arange(len(r["steps"])); o += len(r["steps"])
    assert o == z["index"].shape[1]
    _T[k] = (list(z[keyname]), z["index"], z["emb"], WL, teE, loc)
    return _T[k]


# ------------------------------------------------------------------ settings
def se_setting(name):
    if name == "C1":
        P0, FA = bgc.dojo()
        inj = np.array([r["category"] == "benign" and r["meta"]["user_task"].startswith("injection_task") for r in P0])
        idx = np.where(~inj)[0]
        return {"kind": "c1", "bench": "agentdojo", "P0": P0, "FA": FA, "test": P0 + FA, "c1_idx": idx,
                "notes": "within-AgentDojo UT on C1 (P0 minus the benign runs whose user task is an injection task); "
                         "folds = r4_control.within_oof assignment on C1, rep = fold seed = training seed"}
    S = bgc.setting_data(name)
    return S


def c1_folds(S, rep):
    recs = [S["P0"][i] for i in S["c1_idx"]]
    keys = [r4_control.group_key(r, "UT") for r in recs]
    groups = sorted(set(keys)); rng = np.random.default_rng(rep); rng.shuffle(groups)
    fo = {g: i % K for i, g in enumerate(groups)}
    return np.array([fo[k] for k in keys])


def units_of(name):
    return [f"seed{s}" for s in SEEDS] if name not in ("WUT", "C1") else [f"rep{r}_fold{f}" for r in REPS for f in range(K)]


def split_unit(name, unit, S):
    """-> (train recs, val recs, test recs, global test rows, seed, extra)."""
    if S["kind"] == "single":
        seed = int(unit[4:])
        tr, va = stratified_val(S["train"], seed)
        return tr, va, S["test"], np.arange(len(S["test"])), seed, {}
    rep, fold = [int(x) for x in unit.replace("rep", "").split("_fold")]
    if S["kind"] == "cv":
        f_p0, f_fa, _ = bgc.wut_folds(S["P0"], S["FA"], rep)
        tri = np.where(f_p0 != fold)[0]
        tr, va = stratified_val([S["P0"][i] for i in tri], rep)
        rows = np.concatenate([np.where(f_p0 == fold)[0], len(S["P0"]) + np.where(f_fa == fold)[0]])
    else:  # C1
        f = c1_folds(S, rep); idx = S["c1_idx"]
        tr, va = stratified_val([S["P0"][idx[i]] for i in np.where(f != fold)[0]], rep)
        rows = idx[f == fold]
    test = S["test"]
    return tr, va, [test[i] for i in rows], rows, rep, {"rep": rep, "fold": fold}


# ------------------------------------------------------------------ one SE unit
def run_se_unit(name, det, unit):
    S = se_setting(name)
    EMB, WF, rows_of, man = table()
    zero_w = det == "SE-W0"
    tr, va, te, rows, seed, extra = split_unit(name, unit, S)
    e_tr, e_va, e_te = enc(tr), enc(va), enc(te)
    t0 = time.time()
    model, thr, thr_val, diag = sd.train_se(e_tr, e_va, EMB, WF, seed, zero_w=zero_w)
    fit_s = time.time() - t0
    z = sd.logits(model, e_te, EMB, WF, zero_w)
    out = {"setting": name, "detector": det, "unit": unit, "seed": seed, "rows": rows, "logit": z.astype(np.float32),
           "score": sd.sigmoid(z).astype(np.float32), "thr_r80": thr, "thr_val80": thr_val, "fit_seconds": fit_s,
           "test_truncated": e_te.n_truncated,
           "train_category": np.array([r["category"] for r in tr]),
           "train_ids_sha256": hashlib.sha256("|".join(r["id"] for r in tr).encode()).hexdigest(),
           "val_ids_sha256": hashlib.sha256("|".join(r["id"] for r in va).encode()).hexdigest(), **extra}
    out["diag"] = dict(diag)
    if name == "R4" and det == "SE":
        zw = sd.logits(model, e_te, EMB, WF, zero_w=True)          # feature-block offset probe (amendment 4), not SE-W0
        out["probe_logit"] = zw.astype(np.float32); out["probe_score"] = sd.sigmoid(zw).astype(np.float32)
        z_trw = sd.logits(model, e_tr, EMB, WF, zero_w=True)
        out["probe_train_logit"] = z_trw.astype(np.float32)
    if name == "R1" and det == "SE":
        pE = np.array([r["category"] in ("benign", "hard_negative", "attacked") for r in te])
        for fname, keyname, tag in (("masked_r1.npz", "mask_keys", "masked"), ("padded_r1.npz", "pad_keys", "padded")):
            keys, index, vemb, WL, teE, loc = variant_tables(fname, keyname)
            e_v = sd.Encoded(teE, lambda r: loc[r["id"]])
            vs = {k: sd.sigmoid(sd.logits(model, e_v, vemb[index[ki]], WL)).astype(np.float32) for ki, k in enumerate(keys)}
            out[f"{tag}_scores"] = vs
            base = vs["1.0|0"] if tag == "masked" else vs["pad0"]
            out[f"{tag}_baseline_max_abs_diff"] = float(np.max(np.abs(base - out["score"][pE])))
    out["total_seconds"] = time.time() - t0
    return out


# ------------------------------------------------------------------ one GRU unit
def run_gru_unit(name, unit):
    if name != "C1":
        S = bgc.setting_data(name)
        return bgc.run_unit(name, "GRU", unit, S)
    from gru_detector import train_gru, score_gru
    S = se_setting("C1")
    tr, va, te, rows, rep, extra = split_unit("C1", unit, S)
    t0 = time.time()
    model, voc, scl, thr = train_gru(tr, va, rep)
    fit_s = time.time() - t0
    s = score_gru(model, voc, scl, thr, te)["score"]
    sv = score_gru(model, voc, scl, thr, va)["score"]
    yv = np.array([r["category"] == "attacked" for r in va])
    thr_val = float(np.quantile(sv[yv], 0.20)) if yv.any() else 0.5
    return {"setting": "C1", "detector": "GRU", "unit": unit, "rows": rows, "score": s, "thr_r80": float(thr),
            "thr_val80": thr_val, "fit_seconds": fit_s, "total_seconds": time.time() - t0, **extra}


def part_path(fam, name, det, unit):
    base = SE_PARTS if fam == "se" else G10_PARTS
    return os.path.join(base, name, det, unit + ".pkl")


def do_unit(job):
    fam, name, det, unit, threads = job
    torch.set_num_threads(threads)
    p = part_path(fam, name, det, unit)
    if os.path.exists(p):
        return (job, "skip", 0.0)
    if os.path.exists(STOPF):
        return (job, "stopped", 0.0)
    t0 = time.time()
    out = run_se_unit(name, det, unit) if fam == "se" else run_gru_unit(name, unit)
    out["threads"] = threads; out["torch"] = torch.__version__; out["host"] = platform.node()
    out["slurm_job_id"] = os.environ.get("SLURM_JOB_ID")
    pdump(out, p)
    if out["fit_seconds"] > FIT_LIMIT_S:
        json.dump({"job": list(job[:4]), "fit_seconds": out["fit_seconds"], "limit": FIT_LIMIT_S,
                   "time": datetime.datetime.now().isoformat()}, open(STOPF, "w"))
    return (job, "done", time.time() - t0)


def job_list(only=None, settings=None, threads=2):
    jobs = []
    order = ["SE", "GRU", "SE-W0"]             # most important first
    for det in order:
        if only and det not in only:
            continue
        fam = "gru" if det == "GRU" else "se"
        for name in PLAN[det]:
            if settings and name not in settings:
                continue
            for u in units_of(name):
                jobs.append((fam, name, det, u, threads))
    return jobs


def cmd_run(workers, threads, only, settings):
    if not only or any(d != "GRU" for d in only):
        table()                                # load once before forking (shared copy-on-write)
        variant_tables("masked_r1.npz", "mask_keys"); variant_tables("padded_r1.npz", "pad_keys")
    bgc.drift(); bgc.dojo()
    jobs = job_list(only, settings, threads)
    todo = [j for j in jobs if not os.path.exists(part_path(j[0], j[1], j[2], j[3]))]
    log(f"run: {len(jobs)} units, {len(todo)} to do, workers {workers} x threads {threads}")
    import multiprocessing as mp
    ctx = mp.get_context("fork")
    t0 = time.time(); n = 0
    with ctx.Pool(workers, maxtasksperchild=20) as pool:
        for job, st, sec in pool.imap_unordered(do_unit, todo, chunksize=1):
            n += 1
            if st == "done":
                p = pload(part_path(job[0], job[1], job[2], job[3]))
                d = p.get("diag", {})
                log(f"  [{n}/{len(todo)}] {job[2]} {job[1]} {job[3]}: fit {p['fit_seconds']:.1f}s "
                    f"epochs {d.get('epochs_run')} best {d.get('best_epoch')} valAP {d.get('best_val_ap', float('nan')):.4f}")
            elif st == "stopped":
                log(f"  [{n}/{len(todo)}] {job[2]} {job[1]} {job[3]}: not started (stop rule)")
    log(f"run finished in {time.time() - t0:.0f}s; stop rule {'TRIGGERED' if os.path.exists(STOPF) else 'not triggered'}")


# ------------------------------------------------------------------ assembly (SE family)
def se_assemble(name):
    S = se_setting(name)
    if name == "R1":
        # test_meta / y / pool_E exactly as the GRU cache builds them for R1
        S_meta = bgc.setting_data("R1"); tm, y, pE, info, extra = bgc.setting_meta("R1", S_meta)
        info["notes"] = S["notes"]
    elif name == "C1":
        S_meta = bgc.setting_data("WUT"); tm, y, pE, info, extra = bgc.setting_meta("WUT", S_meta)
        info = {"notes": S["notes"], "n_test": len(tm)}; extra = {}
    else:
        tm, y, pE, info, extra = bgc.setting_meta(name, S)
    n = len(tm)
    data = {"setting": name, "family": "sencoder", "test_meta": tm, "y": y, "pool_E": pE, "scores": {}, "pred": {},
            "thr": {}, "logit": {}, "info": info, "train_diag": {}, "fit_seconds": {}, "units": {}}
    if name == "C1":
        P0 = S["P0"]; inC1 = np.zeros(n, bool); inC1[S["c1_idx"]] = True
        data["in_C1"] = inC1
    for det in ("SE", "SE-W0"):
        if name not in PLAN[det]:
            continue
        us = units_of(name)
        ps = [part_path("se", name, det, u) for u in us]
        if not all(os.path.exists(p) for p in ps):
            log(f"assemble {name} {det}: missing {sum(not os.path.exists(p) for p in ps)} units")
            continue
        parts = [pload(p) for p in ps]
        R = len(SEEDS) if name not in ("WUT", "C1") else len(REPS)
        sc = np.full((R, n), np.nan, np.float32); lg = np.full((R, n), np.nan, np.float32)
        t80 = np.zeros((R, n), bool); t05 = np.zeros((R, n), bool); tv = np.zeros((R, n), bool)
        for p in parts:
            ri = SEEDS.index(p["seed"]) if name not in ("WUT", "C1") else REPS.index(p["rep"])
            r_ = p["rows"]
            assert np.all(np.isnan(sc[ri, r_])), "row scored twice"
            sc[ri, r_] = p["score"]; lg[ri, r_] = p["logit"]
            t80[ri, r_] = p["score"] >= p["thr_r80"]; t05[ri, r_] = p["score"] >= 0.5; tv[ri, r_] = p["score"] >= p["thr_val80"]
        scored = ~np.isnan(sc[0])
        if name == "C1":
            assert np.array_equal(scored, data["in_C1"])
        else:
            assert scored.all(), f"{name} {det}: unscored rows"
        data["scores"][det] = sc; data["logit"][det] = lg
        data["pred"][det] = {"t_r80": t80, "t05": t05, "t_val80": tv}
        data["thr"][det] = {"t_r80": np.array([p["thr_r80"] for p in parts]), "t05": 0.5,
                            "t_val80": np.array([p["thr_val80"] for p in parts]),
                            "units": [p["unit"] for p in parts]}
        data["fit_seconds"][det] = [p["fit_seconds"] for p in parts]
        data["train_diag"][det] = [{"unit": p["unit"], "seed": p["seed"],
                                    **{k: v for k, v in p["diag"].items() if k not in ("train_logit", "train_y")}}
                                   for p in parts]
        data["units"][det] = [{"unit": p["unit"], "seed": p["seed"], "rows": p["rows"],
                               "train_logit": p["diag"]["train_logit"], "train_y": p["diag"]["train_y"],
                               "train_category": p["train_category"], "test_truncated": p["test_truncated"],
                               "masked_scores": p.get("masked_scores"), "padded_scores": p.get("padded_scores"),
                               "masked_baseline_max_abs_diff": p.get("masked_baseline_max_abs_diff"),
                               "padded_baseline_max_abs_diff": p.get("padded_baseline_max_abs_diff"),
                               "probe_logit": p.get("probe_logit"), "probe_score": p.get("probe_score"),
                               "probe_train_logit": p.get("probe_train_logit"),
                               "best_epoch": p["diag"]["best_epoch"], "epochs_run": p["diag"]["epochs_run"],
                               "threads": p.get("threads"), "slurm_job_id": p.get("slurm_job_id")} for p in parts]
    os.makedirs(SE_CACHE, exist_ok=True)
    pdump(data, os.path.join(SE_CACHE, f"{name}.pkl"))
    return data


def gru10_assemble(name):
    if name == "C1":
        S = se_setting("C1"); n = len(S["test"])
        us = units_of("C1"); parts = [pload(part_path("gru", "C1", "GRU", u)) for u in us]
        sc = np.full((len(REPS), n), np.nan, np.float32); t80 = np.zeros((len(REPS), n), bool); t05 = t80.copy(); tv = t80.copy()
        for p in parts:
            ri = REPS.index(p["rep"]); r_ = p["rows"]
            sc[ri, r_] = p["score"]; t80[ri, r_] = p["score"] >= p["thr_r80"]; t05[ri, r_] = p["score"] >= 0.5
            tv[ri, r_] = p["score"] >= p["thr_val80"]
        inC1 = np.zeros(n, bool); inC1[S["c1_idx"]] = True
        assert np.array_equal(~np.isnan(sc[0]), inC1)
        data = {"setting": "C1", "family": "gru10", "scores": {"GRU": sc}, "pred": {"GRU": {"t_r80": t80, "t05": t05, "t_val80": tv}},
                "in_C1": inC1, "test_ids": [r["id"] for r in S["test"]], "y": np.array([int(r["category"] == "attacked") for r in S["test"]], np.int8),
                "fit_seconds": {"GRU": [p["fit_seconds"] for p in parts]}}
        pdump(data, os.path.join(G10_CACHE, "C1.pkl"))
        return data
    S = bgc.setting_data(name)
    d = bgc.part_dir(name, "GRU")
    # rename SE-style part files into build_gru_cache's layout (same folder) is not needed: same path scheme
    blk = bgc.assemble(name, "GRU", S, len(S["test"]))
    pdump(blk, os.path.join(d, "ASSEMBLED.pkl"))
    return bgc.merge_setting(name)


def cmd_assemble():
    for name in sorted(set(PLAN["SE"]) | set(PLAN["SE-W0"])):
        se_assemble(name); log(f"assembled sencoder {name}")
    for name in PLAN["GRU"]:
        gru10_assemble(name); log(f"assembled gru10 {name}")


# ------------------------------------------------------------------ embedding drift (plan Sec. 7)
def nn_cos_dist(Q, Rf, chunk=4096):
    """Mean over rows of Q of (1 - max cosine to rows of Rf); embeddings are L2-normalised."""
    Rt = torch.from_numpy(Rf)
    out = np.empty(len(Q))
    for i in range(0, len(Q), chunk):
        q = torch.from_numpy(Q[i:i + chunk])
        out[i:i + chunk] = 1.0 - (q @ Rt.T).max(1).values.numpy()
    return out


def cmd_drift():
    EMB, SCAL, rows_of, man = table()
    res = {"definition": "per held-out step: 1 - max cosine similarity to any step of the rung's full training set "
                         "(all categories, before the validation split); summarised over steps of pool-E test "
                         "trajectories (and over all test trajectories)", "rungs": {}}
    def steps(recs):
        return np.concatenate([rows_of[r["id"]] for r in recs if r["steps"]])
    def summ(d):
        return {"n_steps": int(len(d)), "mean": float(d.mean()), "median": float(np.median(d)),
                "p90": float(np.quantile(d, 0.9)), "frac_exact_match_le_1e-6": float(np.mean(d <= 1e-6))}
    for name in SINGLE:
        S = se_setting(name)
        trR = EMB[steps(S["train"])]
        teE = [r for r in S["test"] if r["category"] in ("benign", "hard_negative", "attacked")]
        dE = nn_cos_dist(EMB[steps(teE)], trR)
        ent = {"pool_E": summ(dE)}
        if name == "R4":
            P0, FA = bgc.dojo()
            ent["failed_attacks"] = summ(nn_cos_dist(EMB[steps(FA)], trR))
        res["rungs"][name] = ent
        log(f"drift {name}: {ent['pool_E']}")
    # within-AgentDojo UT, rep 0, per fold
    S = bgc.setting_data("WUT"); f_p0, f_fa, _ = bgc.wut_folds(S["P0"], S["FA"], 0)
    res["rungs"]["WUT_rep0"] = {}
    for f in range(K):
        trR = EMB[steps([S["P0"][i] for i in np.where(f_p0 != f)[0]])]
        res["rungs"]["WUT_rep0"][f"fold{f}"] = summ(nn_cos_dist(EMB[steps([S["P0"][i] for i in np.where(f_p0 == f)[0]])], trR))
    # masked R1 levels (graded sweep), against the R1 training steps
    keys, index, memb, SL, teE, loc = variant_tables("masked_r1.npz", "mask_keys")
    st_tr = records_all()[0]
    trR = EMB[steps(st_tr)]
    res["R1_masked"] = {k: summ(nn_cos_dist(memb[index[i]], trR)) for i, k in enumerate(keys)}
    os.makedirs(os.path.join(HERE, "sencoder_run"), exist_ok=True)
    json.dump(res, open(os.path.join(HERE, "sencoder_run", "embedding_drift.json"), "w"), indent=1)
    log("drift written")


def cmd_smoke(threads):
    torch.set_num_threads(threads)
    table(); variant_tables("masked_r1.npz", "mask_keys"); variant_tables("padded_r1.npz", "pad_keys")
    res = {"n_params_SE": sd.n_params(sd.SEDet()), "n_params_reported_L3": 1845637}
    _L = sd.N_LAYERS; sd.N_LAYERS = 2; res["n_params_L2"] = sd.n_params(sd.SEDet()); res["n_params_reported_L2"] = 1318533; sd.N_LAYERS = _L
    EMB, WF, rows_of, man = table()
    res["world_feature_means"] = {"all_steps": WF.mean(0).round(4).tolist()}
    for fam, name, det, unit in [("se", "R1", "SE", "seed0"), ("se", "R4", "SE", "seed0"), ("se", "WUT", "SE", "rep0_fold0"),
                                 ("gru", "R1", "GRU", "seed0"), ("gru", "C1", "GRU", "rep0_fold0")]:
        t0 = time.time()
        out = run_se_unit(name, det, unit) if fam == "se" else run_gru_unit(name, unit)
        d = out.get("diag", {})
        res[f"{det}|{name}|{unit}"] = {"fit_seconds": out["fit_seconds"], "total_seconds": time.time() - t0,
                                       "epochs": d.get("epochs_run"), "best_epoch": d.get("best_epoch"),
                                       "best_val_f1": d.get("best_val_f1"), "class_weights": d.get("class_weights"),
                                       "pos_weight": d.get("pos_weight"),
                                       "masked_baseline_max_abs_diff": out.get("masked_baseline_max_abs_diff"),
                                       "padded_baseline_max_abs_diff": out.get("padded_baseline_max_abs_diff")}
        if fam == "se" and name == "R1":
            yE = np.array([r["category"] == "attacked" for r in se_setting("R1")["test"]])
            res[f"{det}|{name}|{unit}"]["auroc_all_test_sanity"] = float(ek.auroc(yE.astype(int), out["score"]))
            out2 = run_se_unit(name, det, unit)
            res["determinism_R1_seed0_max_abs_diff"] = float(np.max(np.abs(out["score"] - out2["score"])))
        if fam == "gru" and name == "R1":
            ref = os.path.join(HERE, "cache", "gru", "R1.pkl")
            if os.path.exists(ref):
                g = pload(ref); res["gru_R1_seed0_max_abs_diff_vs_3seed_cache"] = float(np.max(np.abs(out["score"] - g["scores"]["GRU"][0])))
        log(f"smoke {det} {name} {unit}: {res[f'{det}|{name}|{unit}']}")
    res["threads"] = threads; res["torch"] = torch.__version__
    os.makedirs(os.path.join(HERE, "sencoder_run"), exist_ok=True)
    json.dump(res, open(os.path.join(HERE, "sencoder_run", "smoke.json"), "w"), indent=1, default=float)


def cmd_status():
    for det, sets in PLAN.items():
        fam = "gru" if det == "GRU" else "se"
        for name in sets:
            us = units_of(name)
            done = sum(os.path.exists(part_path(fam, name, det, u)) for u in us)
            print(f"{det:9} {name:12} {done:3}/{len(us)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--settings", nargs="*")
    a = ap.parse_args()
    if a.cmd == "run":
        cmd_run(a.workers, a.threads, a.only, a.settings)
    elif a.cmd == "assemble":
        cmd_assemble()
    elif a.cmd == "drift":
        cmd_drift()
    elif a.cmd == "smoke":
        cmd_smoke(a.threads)
    elif a.cmd == "status":
        cmd_status()
    else:
        raise SystemExit(a.cmd)
