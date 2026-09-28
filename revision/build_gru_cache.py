"""Build the GRU prediction cache (family "gru") for the revision (REVISION_ANALYSIS_PLAN.md Sec. 2).

Resumable: one setting and one detector per call. Work is split into units (one training seed for
single-fit settings; one (repetition, fold) for WUT); every finished unit is saved to its own file
under cache/gru/_parts/<setting>/<detector>/ and skipped on the next call. A call stops starting new
units when its time budget is used up; call it again until it prints COMPLETE. When all units of a
(setting, detector) exist they are assembled and merged into cache/gru/<setting>.pkl.

Usage (always with AGENTDRIFT and AGENTDOJO set):
  python build_gru_cache.py run <setting> <detector> [--budget SECONDS]
  python build_gru_cache.py verify            # p = 0 copy reproduces gru_detector.train_gru (R1 seed 0)
  python build_gru_cache.py status            # what is done / missing
  python build_gru_cache.py summarize         # metrics + anchors -> revision/gru_cache_results.json

Detectors: GRU (p = 0, pre-registered), GRU-drop0.2 (primary), GRU-drop0.1 (sensitivity).
Settings : R1, R1wd_s0..R1wd_s4, R2, R3_<domain>, R4, WUT (GRU); R1, R3_<domain>, R4 (dropout arms).
Everything else (architecture, constants, validation split, seeds, thresholds, WUT folds) is copied
from gru_detector.py / gru_ladder.py / r4_control.py / world_disjoint_r1.py, which are imported, not
modified. Thresholds come from training (t_r80) or validation (t_val80) data only.
"""
import os, sys, json, time, pickle, argparse, collections, datetime
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)

import torch  # noqa: E402
from featurise import load  # noqa: E402
import evalkit as ek  # noqa: E402
import gru_detector as gd  # noqa: E402
from gru_detector import score_gru, build_tool_vocab, _tensors, ylab, UNK, PAD  # noqa: E402
from gru_ladder import stratified_val, coverage as ladder_coverage  # noqa: E402
from world_disjoint_r1 import world_disjoint_split  # noqa: E402
from gru_variants import train_gru_drop, decompose, embedding_norms  # noqa: E402

MP = os.environ["AGENTDRIFT"]
CACHE = os.path.join(HERE, "cache", "gru"); PARTS = os.path.join(CACHE, "_parts")
LOGF = os.path.join(HERE, "logs", "build_gru_cache.log")
DETECTORS = {"GRU": 0.0, "GRU-drop0.2": 0.2, "GRU-drop0.1": 0.1}
DOMAINS = ["medical", "banking", "web", "email", "coding"]
SEEDS = [0, 1, 2]              # training seeds, single-fit settings (gru_ladder.TRAIN_SEEDS)
WUT_REPS = [0, 1, 2, 3, 4]     # fold-assignment seed = training seed (gru_ladder within-AgentDojo)
K = 5
SETTINGS_ALL = ["R1"] + [f"R1wd_s{k}" for k in range(5)] + ["R2"] + [f"R3_{d}" for d in DOMAINS] + ["R4", "WUT"]
PLAN = {"GRU": SETTINGS_ALL,
        "GRU-drop0.2": ["R1"] + [f"R3_{d}" for d in DOMAINS] + ["R4"],
        "GRU-drop0.1": ["R1"] + [f"R3_{d}" for d in DOMAINS] + ["R4"]}
DOJO_TRACES = os.path.join(ROOT, "agentdojo_traces.jsonl")
DOJO_FAILED = os.path.join(ROOT, "agentdojo_failed.jsonl")


def log(msg):
    os.makedirs(os.path.dirname(LOGF), exist_ok=True)
    line = f"{datetime.datetime.now().isoformat(timespec='seconds')} {msg}"
    print(line, flush=True)
    with open(LOGF, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def pdump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=4)
    os.replace(tmp, path)


def pload(path):
    # pickle is the shared cache format fixed by the revision plan; only files written by this
    # script (or the sibling linear-cache builder) under revision/cache are ever loaded.
    with open(path, "rb") as f:
        return pickle.load(f)


# ------------------------------------------------------------------ data
_D = {}


def drift():
    if "drift" not in _D:
        st_tr = load(os.path.join(MP, "data/train.jsonl")); st_te = load(os.path.join(MP, "data/test.jsonl"))
        td_tr = load(os.path.join(MP, "data_taskdisjoint/train.jsonl")); td_te = load(os.path.join(MP, "data_taskdisjoint/test.jsonl"))
        _D["drift"] = (st_tr, st_te, td_tr, td_te)
    return _D["drift"]


def dojo():
    if "dojo" not in _D:
        P0 = [json.loads(l) for l in open(DOJO_TRACES, encoding="utf-8")]
        FA = [json.loads(l) for l in open(DOJO_FAILED, encoding="utf-8")]
        _D["dojo"] = (P0, FA)
    return _D["dojo"]


def ut_key(r):
    """suite|user_task from the id, exactly as gru_ladder.py / r4_control.py (UT scheme)."""
    p = r["id"].split("|")
    return f"{p[1]}|{p[2]}"


def meta_row(r, bench, V):
    tools = [s["tool"] for s in r["steps"]]
    cov = float(np.mean([t in V for t in tools])) if tools else float("nan")
    cat = r["category"]
    if bench == "agentdrift":
        return {"id": r["id"], "benchmark": "agentdrift", "category": cat, "y": int(cat == "attacked"),
                "in_pool_E": cat in ("benign", "hard_negative", "attacked"), "domain": r["agent"],
                "pattern": r.get("source_category") if cat == "attacked" else None,
                "world": json.dumps(r["world"], sort_keys=True), "task": r["task"],
                "model": None, "user_task": None, "injection_task": None, "attack_type": None, "is_dos": None,
                "n_steps": len(r["steps"]), "cov": cov}
    m = r["meta"]
    return {"id": r["id"], "benchmark": "agentdojo", "category": cat, "y": int(cat == "attacked"),
            "in_pool_E": cat in ("benign", "attacked"), "domain": m["suite"], "pattern": None, "world": None,
            "task": f"{m['suite']}|{m['user_task']}", "model": m["model"], "user_task": m["user_task"],
            "injection_task": m["injection_task"], "attack_type": m["attack_type"], "is_dos": bool(m["is_dos"]),
            "n_steps": len(r["steps"]), "cov": cov}


def benign_tools(recs):
    return {s["tool"] for r in recs if r["category"] == "benign" for s in r["steps"]}


def row_cov(recs, V):
    return np.array([np.mean([s["tool"] in V for s in r["steps"]]) if r["steps"] else np.nan for r in recs], float)


def wut_folds(P0, FA, rep):
    """Fold of every WUT row (P0 then failed). P0 groups and their folds are exactly gru_ladder.py's
    (= r4_control.py UT scheme). A failed attack gets the fold of its suite|user_task group; groups
    that occur only among failed attacks (not in P0) are assigned by a separate rng (seed 10000+rep)."""
    keys = [ut_key(r) for r in P0]
    groups = sorted(set(keys)); rng = np.random.default_rng(rep); rng.shuffle(groups)
    fo = {g: j % K for j, g in enumerate(groups)}
    extra = sorted({ut_key(r) for r in FA} - set(fo))
    rng2 = np.random.default_rng(10000 + rep); rng2.shuffle(extra)
    fo_extra = {g: j % K for j, g in enumerate(extra)}
    f_p0 = np.array([fo[k] for k in keys])
    f_fa = np.array([fo.get(ut_key(r), fo_extra.get(ut_key(r))) for r in FA])
    return f_p0, f_fa, extra


def setting_data(name):
    """Returns dict(kind, bench, train, test, ...). Test always includes unsuccessful attacks."""
    st_tr, st_te, td_tr, td_te = drift()
    pool = st_tr + st_te
    if name == "R1":
        return {"kind": "single", "bench": "agentdrift", "train": st_tr, "test": st_te,
                "notes": "train data/train.jsonl, test data/test.jsonl; GRU validation = stratified 10% of train "
                         "(gru_ladder.stratified_val, as gru_ladder.py; data/val.jsonl unused)"}
    if name.startswith("R1wd_s"):
        k = int(name[len("R1wd_s"):])
        tr_i, te_i = world_disjoint_split(pool, k)
        return {"kind": "single", "bench": "agentdrift", "train": [pool[i] for i in tr_i], "test": [pool[i] for i in te_i],
                "notes": f"world_disjoint_r1.world_disjoint_split(pool=train+test, seed={k})"}
    if name == "R2":
        return {"kind": "single", "bench": "agentdrift", "train": td_tr, "test": td_te, "notes": "data_taskdisjoint train/test"}
    if name.startswith("R3_"):
        d = name[3:]; assert d in DOMAINS
        return {"kind": "single", "bench": "agentdrift", "train": [r for r in pool if r["agent"] != d],
                "test": [r for r in pool if r["agent"] == d], "notes": f"leave-one-domain-out, held-out domain {d}"}
    if name == "R4":
        P0, FA = dojo()
        return {"kind": "single", "bench": "agentdojo", "train": pool, "test": P0 + FA,
                "notes": "train all 10,656 AgentDrift (train+test); test AgentDojo P0 + agentdojo_failed.jsonl"}
    if name == "WUT":
        P0, FA = dojo()
        return {"kind": "cv", "bench": "agentdojo", "P0": P0, "FA": FA, "test": P0 + FA,
                "notes": "within-AgentDojo, 5-fold CV grouped by suite|user_task (gru_ladder/r4_control UT folds), "
                         "fold seed = training seed = rep (0..4); training on P0 rows of the other folds only; "
                         "failed attacks scored by the model whose test fold holds their group"}
    raise ValueError(name)


def units_of(name):
    return [f"seed{s}" for s in SEEDS] if name != "WUT" else [f"rep{r}_fold{f}" for r in WUT_REPS for f in range(K)]


def part_dir(name, det):
    return os.path.join(PARTS, name, det)


# ------------------------------------------------------------------ one unit
def score_block(model, vocab, scaler, thr, recs):
    d = score_gru(model, vocab, scaler, thr, recs)
    X = _tensors(recs, vocab, scaler)
    tc, bc, bias, lg = decompose(model, X)
    return d["score"], tc, bc, bias, lg


def run_unit(name, det, unit, S):
    p = DETECTORS[det]
    t0 = time.time()
    if S["kind"] == "single":
        seed = int(unit[4:])
        tr, va = stratified_val(S["train"], seed)
        test = S["test"]; rows = np.arange(len(test))
        tsets = None
    else:
        rep, fold = [int(x) for x in unit.replace("rep", "").split("_fold")]
        seed = rep
        f_p0, f_fa, _ = wut_folds(S["P0"], S["FA"], rep)
        tri = np.where(f_p0 != fold)[0]
        tr, va = stratified_val([S["P0"][i] for i in tri], seed)
        te_p0 = np.where(f_p0 == fold)[0]; te_fa = np.where(f_fa == fold)[0]
        test = S["test"]; nP0 = len(S["P0"])
        rows = np.concatenate([te_p0, nP0 + te_fa])
        tsets = (te_p0, nP0 + te_fa)
    model, vocab, scaler, thr, aux = train_gru_drop(tr, va, seed, p)
    fit_s = time.time() - t0
    Xtr = aux.pop("Xtr")
    # score: pool-E rows in one pass (the exact batch v0.3 scored), other rows in a separate pass
    if S["kind"] == "single":
        inE = np.array([meta_row(r, S["bench"], set())["in_pool_E"] for r in test])
        passes = [np.where(inE)[0], np.where(~inE)[0]]
    else:
        passes = list(tsets)
    n = len(test)
    s = np.full(n, np.nan, np.float32); tc = np.full(n, np.nan, np.float32); bc = np.full(n, np.nan, np.float32)
    lg = np.full(n, np.nan, np.float32); bias = None
    for idx in passes:
        if len(idx) == 0:
            continue
        sc, t_, b_, bias, l_ = score_block(model, vocab, scaler, thr, [test[i] for i in idx])
        s[idx] = sc; tc[idx] = t_; bc[idx] = b_; lg[idx] = l_
    out = {"setting": name, "detector": det, "unit": unit, "seed": seed, "p": p, "rows": rows,
           "score": s[rows], "tool_contrib": tc[rows], "beh_contrib": bc[rows], "logit": lg[rows], "bias": bias,
           "thr_r80": thr, "thr_val80": aux["thr_val80"], "fit_seconds": fit_s}
    recon = np.max(np.abs(out["tool_contrib"] + out["beh_contrib"] + bias - out["logit"]))
    out["decomp_max_abs_err"] = float(recon)
    unk_norm, emb_norms, emb_names = embedding_norms(model, vocab)
    V_model = set(vocab)
    out["cov_model"] = row_cov([test[i] for i in rows], V_model).astype(np.float32)
    ttc, tbc, _, tlg = decompose(model, Xtr)
    if S["kind"] == "single":
        out.update({"train_tool_contrib": ttc, "train_beh_contrib": tbc, "train_y": aux["y_train"],
                    "train_category": np.array([r["category"] for r in tr]),
                    "train_domain": np.array([r["agent"] for r in tr])})
    else:
        out.update({"fold": fold, "rep": rep,
                    "train_tool_contrib_summary": {c: float(ttc[aux["y_train"] == c].mean()) for c in (0, 1)}})
    out.update({"unk_norm": unk_norm, "tool_emb_norms": emb_norms, "tool_emb_names": emb_names})
    out["diag"] = {k: v for k, v in aux.items() if k not in ("s_train", "y_train")}
    out["diag"].update({"fit_seconds": fit_s, "thr_r80": thr, "decomp_max_abs_err": float(recon),
                        "n_train_setting": (len(S["train"]) if S["kind"] == "single" else len(tri))})
    out["total_seconds"] = time.time() - t0
    return out


# ------------------------------------------------------------------ assembly
def setting_meta(name, S):
    test = S["test"]; bench = S["bench"]
    if S["kind"] == "single":
        V = benign_tools(S["train"])
        tm = [meta_row(r, bench, V) for r in test]
        E = [r for r, m in zip(test, tm) if m["in_pool_E"]]
        info = {"n_train": len(S["train"]), "n_train_pos": int(ylab(S["train"]).sum()),
                "kappa": ladder_coverage(S["train"], E)["tool_occ_cov"],
                "coverage_poolE": ladder_coverage(S["train"], E), "kappa_all_test": ladder_coverage(S["train"], test)["tool_occ_cov"],
                "vocab_size": len(build_tool_vocab(S["train"])), "notes": S["notes"],
                "train_composition": dict(collections.Counter(r["category"] for r in S["train"]))}
        extra = {}
    else:
        P0, FA = S["P0"], S["FA"]
        tm = [meta_row(r, bench, set()) for r in test]
        cov_by_rep = np.zeros((len(WUT_REPS), len(test)))
        fold_of_row = np.zeros((len(WUT_REPS), len(test)), int)
        ntr = np.zeros((len(WUT_REPS), K), int); ntrp = np.zeros((len(WUT_REPS), K), int)
        vs = np.zeros((len(WUT_REPS), K), int); kap = np.zeros((len(WUT_REPS), K))
        extra_groups = None
        for ri, rep in enumerate(WUT_REPS):
            f_p0, f_fa, extra_groups = wut_folds(P0, FA, rep)
            fr = np.concatenate([f_p0, f_fa]); fold_of_row[ri] = fr
            for f in range(K):
                trn = [P0[i] for i in np.where(f_p0 != f)[0]]
                V = benign_tools(trn); te = np.where(fr == f)[0]
                cov_by_rep[ri, te] = row_cov([test[i] for i in te], V)
                ntr[ri, f] = len(trn); ntrp[ri, f] = int(ylab(trn).sum()); vs[ri, f] = len(V)
                teE = [test[i] for i in te if tm[i]["in_pool_E"]]
                kap[ri, f] = ladder_coverage(trn, teE)["tool_occ_cov"]
        for i, m in enumerate(tm):
            m["cov"] = float(np.nanmean(cov_by_rep[:, i]))
        info = {"n_train": int(round(ntr.mean())), "n_train_pos": int(round(ntrp.mean())),
                "n_train_by_rep_fold": ntr.tolist(), "n_train_pos_by_rep_fold": ntrp.tolist(),
                "kappa": float(round(kap.mean(), 4)), "kappa_by_rep_fold": kap.round(4).tolist(),
                "vocab_size": int(round(vs.mean())), "vocab_size_by_rep_fold": vs.tolist(),
                "n_P0": len(P0), "n_failed": len(FA), "failed_only_groups": extra_groups,
                "notes": S["notes"] + ". test_meta cov = mean over the 5 repetitions of the out-of-fold coverage "
                         "(per-repetition values in cov_by_rep). n_train/n_train_pos/vocab_size/kappa are means over "
                         "repetitions x folds (full arrays in *_by_rep_fold)."}
        extra = {"cov_by_rep": cov_by_rep, "fold_of_row": fold_of_row}
    y = np.array([m["y"] for m in tm], np.int8); pE = np.array([m["in_pool_E"] for m in tm], bool)
    info.update({"n_test": len(tm), "n_test_poolE": int(pE.sum()), "n_pos_poolE": int(y[pE].sum()),
                 "n_neg_poolE": int((pE & (y == 0)).sum()), "n_not_poolE": int((~pE).sum()),
                 "test_composition": dict(collections.Counter(m["category"] for m in tm))})
    return tm, y, pE, info, extra


def assemble(name, det, S, n_test):
    d = part_dir(name, det); us = units_of(name)
    parts = [pload(os.path.join(d, u + ".pkl")) for u in us]
    blk = {"diag": [dict(unit=p["unit"], seed=p["seed"], **p["diag"]) for p in parts]}
    if S["kind"] == "single":
        blk["scores"] = np.stack([p["score"] for p in parts])
        thr80 = np.array([p["thr_r80"] for p in parts]); thrv = np.array([p["thr_val80"] for p in parts])
        blk["pred"] = {"t_r80": np.stack([p["score"] >= p["thr_r80"] for p in parts]),
                       "t05": np.stack([p["score"] >= 0.5 for p in parts]),
                       "t_val80": np.stack([p["score"] >= p["thr_val80"] for p in parts])}
        blk["thr"] = {"t_r80": thr80, "t05": 0.5, "t_val80": thrv, "seeds": np.array(SEEDS)}
        for k in ("tool_contrib", "beh_contrib", "logit", "cov_model"):
            blk[k] = np.stack([p[k] for p in parts])
        blk["bias"] = np.array([p["bias"] for p in parts])
        for k in ("train_tool_contrib", "train_beh_contrib", "train_y", "train_category", "train_domain",
                  "tool_emb_norms", "tool_emb_names"):
            blk[k] = [p[k] for p in parts]
        blk["unk_norm"] = np.array([p["unk_norm"] for p in parts])
        blk["unk_init_norm"] = np.array([p["diag"]["unk_init_norm"] for p in parts])
        blk["unk_train_steps"] = np.array([p["diag"]["unk_train_steps"] for p in parts])
        for p in parts:
            assert np.all(p["rows"] == np.arange(n_test))
    else:
        R = len(WUT_REPS)
        sc = np.full((R, n_test), np.nan, np.float32); tr80 = np.zeros((R, n_test), bool)
        t05 = np.zeros((R, n_test), bool); tv = np.zeros((R, n_test), bool)
        cm = np.full((R, n_test), np.nan, np.float32); tcon = np.full((R, n_test), np.nan, np.float32)
        thr80 = np.zeros((R, K)); thrv = np.zeros((R, K)); unk = np.zeros((R, K), int); unkn = np.zeros((R, K))
        unktr = np.zeros((R, K), int); unkpos = np.zeros((R, K), int)
        for p in parts:
            ri = WUT_REPS.index(p["rep"]); f = p["fold"]; rows = p["rows"]
            assert np.all(np.isnan(sc[ri, rows])), "row scored twice"
            sc[ri, rows] = p["score"]; tr80[ri, rows] = p["score"] >= p["thr_r80"]
            t05[ri, rows] = p["score"] >= 0.5; tv[ri, rows] = p["score"] >= p["thr_val80"]
            cm[ri, rows] = p["cov_model"]; tcon[ri, rows] = p["tool_contrib"]
            thr80[ri, f] = p["thr_r80"]; thrv[ri, f] = p["thr_val80"]
            unk[ri, f] = p["diag"]["unk_train_steps"]; unkn[ri, f] = p["unk_norm"]
            unktr[ri, f] = p["diag"]["unk_train_traces"]; unkpos[ri, f] = p["diag"]["unk_train_traces_pos"]
        assert not np.isnan(sc).any(), "some rows lack an out-of-fold score"
        blk["scores"] = sc
        blk["pred"] = {"t_r80": tr80, "t05": t05, "t_val80": tv}
        blk["thr"] = {"t_r80": thr80, "t05": 0.5, "t_val80": thrv, "layout": "(rep, fold); rep = fold seed = training seed"}
        blk["cov_model"] = cm; blk["tool_contrib_oof"] = tcon
        blk["unk_train_steps"] = unk; blk["unk_train_traces"] = unktr; blk["unk_train_traces_pos"] = unkpos
        blk["unk_norm"] = unkn
    blk["fit_seconds"] = [p["fit_seconds"] for p in parts]
    return blk


def merge_setting(name):
    S = setting_data(name)
    tm, y, pE, info, extra = setting_meta(name, S)
    data = {"setting": name, "family": "gru", "test_meta": tm, "y": y, "pool_E": pE,
            "scores": {}, "pred": {}, "thr": {}, "info": info, "train_diag": {}, "fit_seconds": {}}
    data.update(extra)
    mech_keys_single = ["tool_contrib", "beh_contrib", "bias", "logit", "train_tool_contrib", "train_beh_contrib",
                        "train_y", "train_category", "train_domain", "unk_norm", "unk_init_norm", "tool_emb_norms",
                        "tool_emb_names", "unk_train_steps", "cov_model"]
    mech_keys_cv = ["cov_model", "tool_contrib_oof", "unk_train_steps", "unk_train_traces", "unk_train_traces_pos", "unk_norm"]
    present = []
    for det in DETECTORS:
        f = os.path.join(part_dir(name, det), "ASSEMBLED.pkl")
        if not os.path.exists(f):
            continue
        blk = pload(f); present.append(det)
        data["scores"][det] = blk["scores"]; data["pred"][det] = blk["pred"]; data["thr"][det] = blk["thr"]
        data["train_diag"][det] = blk["diag"]; data["fit_seconds"][det] = blk["fit_seconds"]
        for k in (mech_keys_single if S["kind"] == "single" else mech_keys_cv):
            data.setdefault(k, {})[det] = blk[k]
    info["detectors_present"] = present
    info["detector_p"] = {d: DETECTORS[d] for d in present}
    info["score_shape"] = "(n_seeds=3, n_test) single-fit; (n_reps=5, n_test) out-of-fold for WUT"
    pdump(data, os.path.join(CACHE, f"{name}.pkl"))
    return data


def cmd_run(name, det, budget):
    assert det in DETECTORS and name in PLAN[det], f"{det} not planned for {name}"
    t_start = time.time()
    S = setting_data(name)
    d = part_dir(name, det); os.makedirs(d, exist_ok=True)
    us = units_of(name)
    todo = [u for u in us if not os.path.exists(os.path.join(d, u + ".pkl"))]
    prev = [pload(os.path.join(d, u + ".pkl"))["total_seconds"] for u in us if u not in todo]
    est = max(prev) if prev else (120.0 if name == "R4" else 60.0)
    log(f"run {name} {det}: {len(us) - len(todo)}/{len(us)} units done; budget {budget}s")
    for u in todo:
        el = time.time() - t_start
        if el + 1.25 * est > budget:
            log(f"  stop: elapsed {el:.0f}s + est {est:.0f}s > budget; INCOMPLETE ({name} {det}), call again")
            return False
        out = run_unit(name, det, u, S)
        pdump(out, os.path.join(d, u + ".pkl"))
        prev.append(out["total_seconds"]); est = max(prev)
        log(f"  {name} {det} {u}: fit {out['fit_seconds']:.1f}s total {out['total_seconds']:.1f}s "
            f"epochs {out['diag']['epochs_run']} best {out['diag']['best_epoch']} valAP {out['diag']['best_val_ap']:.4f} "
            f"thr_r80 {out['thr_r80']:.4f} thr_val80 {out['thr_val80']:.4f} unk_train_steps {out['diag']['unk_train_steps']} "
            f"drop {out['diag']['drop_rate_realised']:.4f} decomp_err {out['decomp_max_abs_err']:.2e}")
    n_test = len(S["test"])
    blk = assemble(name, det, S, n_test)
    pdump(blk, os.path.join(d, "ASSEMBLED.pkl"))
    data = merge_setting(name)
    yE = data["y"][data["pool_E"]]
    au = [ek.auroc(yE, s[data["pool_E"]]) for s in (blk["scores"] if blk["scores"].ndim == 2 else [blk["scores"]])]
    log(f"  COMPLETE {name} {det}: pool-E AUROC per seed/rep {np.round(au, 4).tolist()} mean {np.mean(au):.4f}; "
        f"merged -> cache/gru/{name}.pkl detectors {data['info']['detectors_present']}")
    return True


# ------------------------------------------------------------------ verification of the p = 0 copy
def cmd_verify():
    st_tr, st_te, _, _ = drift()
    keep = [r for r in st_te if r["category"] in ("benign", "hard_negative", "attacked")]
    tr, va = stratified_val(st_tr, 0)
    t0 = time.time(); m1, v1, s1, th1 = gd.train_gru(tr, va, 0); t1 = time.time() - t0
    t0 = time.time(); m2, v2, s2, th2, aux = train_gru_drop(tr, va, 0, 0.0); t2 = time.time() - t0
    a = score_gru(m1, v1, s1, th1, keep)["score"]; b = score_gru(m2, v2, s2, th2, keep)["score"]
    sd1, sd2 = m1.state_dict(), m2.state_dict()
    same_params = all(torch.equal(sd1[k], sd2[k]) for k in sd1)
    res = {"setting": "R1", "seed": 0, "n_test_poolE": len(keep), "scores_identical": bool(np.array_equal(a, b)),
           "max_abs_score_diff": float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))),
           "params_identical": bool(same_params), "thr_identical": th1 == th2, "thr_r80": th1,
           "vocab_identical": v1 == v2, "auroc_train_gru": ek.auroc(ylab(keep), a), "auroc_copy_p0": ek.auroc(ylab(keep), b),
           "seconds_train_gru": t1, "seconds_copy": t2, "epochs_run": aux["epochs_run"], "best_epoch": aux["best_epoch"],
           "torch": torch.__version__, "threads": torch.get_num_threads(),
           "timestamp": datetime.datetime.now().isoformat(timespec="seconds")}
    # dropout leaves the main random stream unchanged: record every torch.randperm output (batch order)
    # during p = 0 and p = 0.2 training on a 1,500-row subset; the common epochs must match exactly.
    import gru_variants as gv
    rec = {"cur": None}; orig = torch.randperm
    def spy(*a, **k):
        out = orig(*a, **k); rec["cur"].append(out.clone()); return out
    sub_tr, sub_va = tr[:1500], va[:300]
    perms = {}
    gv.torch.randperm = spy
    try:
        for pp in (0.0, 0.2):
            rec["cur"] = []; _, _, _, _, ax = train_gru_drop(sub_tr, sub_va, 0, pp); perms[pp] = (rec["cur"], ax)
    finally:
        gv.torch.randperm = orig
    ncommon = min(len(perms[0.0][0]), len(perms[0.2][0]))
    res["stream_check"] = {"n_rows": len(sub_tr), "epochs_p0": perms[0.0][1]["epochs_run"], "epochs_p02": perms[0.2][1]["epochs_run"],
                           "common_epochs": ncommon,
                           "batch_orders_identical": bool(all(torch.equal(perms[0.0][0][i], perms[0.2][0][i]) for i in range(ncommon))),
                           "realised_drop_rate_p02": perms[0.2][1]["drop_rate_realised"],
                           "tokens_dropped_p0": perms[0.0][1]["drop_tokens_dropped"]}
    json.dump(res, open(os.path.join(HERE, "gru_p0_verification.json"), "w"), indent=1, default=float)
    log(f"verify: {res}")
    return res


def cmd_status():
    for det, sets in PLAN.items():
        for name in sets:
            d = part_dir(name, det); us = units_of(name)
            done = sum(os.path.exists(os.path.join(d, u + ".pkl")) for u in us)
            asm = os.path.exists(os.path.join(d, "ASSEMBLED.pkl"))
            print(f"{det:12} {name:12} {done:2}/{len(us)} {'ASSEMBLED' if asm else ''}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd"); ap.add_argument("setting", nargs="?"); ap.add_argument("detector", nargs="?")
    ap.add_argument("--budget", type=float, default=520.0)
    a = ap.parse_args()
    if a.cmd == "run":
        ok = cmd_run(a.setting, a.detector, a.budget)
        print("COMPLETE" if ok else "INCOMPLETE")
    elif a.cmd == "verify":
        cmd_verify()
    elif a.cmd == "status":
        cmd_status()
    elif a.cmd == "summarize":
        import summarize_gru_cache
        summarize_gru_cache.main()
    else:
        raise SystemExit(f"unknown command {a.cmd}")
