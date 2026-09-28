"""Summary, reproduction anchors and integrity checks for the GRU prediction cache (family "gru").

Reads only cache/gru/<setting>.pkl (written by build_gru_cache.py) plus the archived v0.3 result file
(archive_v0.3/gru_results.json) and the v0.3 scripts' re-run on the corrected AgentDojo data
(gru_results.json in agent-ids/, final because revision/rerun_v03_scripts.log contains "gru_ladder exit 0").
Writes revision/gru_cache_results.json and revision/cache/gru/CACHE_README.md. No model is fitted here except
in `recheck`, which refits a few units with the unmodified gru_detector.train_gru to prove the cached
scores are reproducible bit for bit.

Resumable: each setting's summary is saved to cache/gru/_summary/<setting>.json and skipped next time.
Usage (AGENTDRIFT and AGENTDOJO set):
  python summarize_gru_cache.py summarize [SETTING ...]   # per-setting summaries (skips finished ones)
  python summarize_gru_cache.py recheck                   # exact refit checks -> cache/gru/_summary/_recheck.json
  python summarize_gru_cache.py assemble                  # gru_cache_results.json + CACHE_README.md
  python summarize_gru_cache.py all                       # the three steps above

Statistics. Point estimates are evalkit.point (metric averaged over the training seeds, or over the five
repetitions for WUT); 95% intervals are percentile intervals of evalkit.bootstrap (stratified over test
trajectories, B = 2000, one seed/repetition drawn per replicate), i.e. the v0.3 protocol. For R2, R4 and WUT a
cluster bootstrap (tasks for R2; suite|user_task for AgentDojo) is added as a labelled sensitivity interval.
All metrics are computed on pool E (AgentDrift benign + hard_negative + attacked; AgentDojo P0); unsuccessful
attacks (not in pool E) are summarised separately as flag rates.
"""
import os, sys, json, time, pickle, hashlib, collections, datetime
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import evalkit as ek  # noqa: E402

CACHE = os.path.join(HERE, "cache", "gru")
SUMM = os.path.join(CACHE, "_summary")
OUT = os.path.join(HERE, "gru_cache_results.json")
README = os.path.join(CACHE, "CACHE_README.md")
B = 2000
DOMAINS = ["medical", "banking", "web", "email", "coding"]
SETTINGS = ["R1"] + [f"R1wd_s{k}" for k in range(5)] + ["R2"] + [f"R3_{d}" for d in DOMAINS] + ["R4", "WUT"]
DETS = ["GRU", "GRU-drop0.2", "GRU-drop0.1"]
OPS = ["t_r80", "t_val80", "t05"]
OPM = ["precision", "recall", "FPR", "FNR", "alerts_per_1000"]
# anchors as given in the revision task (3 d.p.) -> exact v0.3 values are read from the archive
TASK_ANCHORS = {"R1": 0.936, "R1wd_s0": 0.956, "R2": 0.915, "R3_medical": 0.639, "R3_banking": 0.736,
                "R3_web": 0.726, "R3_email": 0.671, "R3_coding": 0.695}


def pload(path):
    # the shared cache format is pickle (fixed by the revision plan); only files written by build_gru_cache.py
    with open(path, "rb") as f:
        return pickle.load(f)


def jdump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=_default)
    os.replace(tmp, path)


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


def r4(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)


def trip(pt, arr):
    mu, lo, hi = ek.ci(arr)
    return [r4(pt), r4(lo), r4(hi)]


def reps_of(d, det, idx):
    S = d["scores"][det]; P = d["pred"][det]
    return [{"score": S[r][idx], "pred": {"t_r80": P["t_r80"][r][idx], "t_val80": P["t_val80"][r][idx],
                                          "t05": P["t05"][r][idx]}} for r in range(S.shape[0])]


def metric_block(y, methods, bt):
    out = {}
    for m, reps in methods.items():
        pt = ek.point(y, reps)
        blk = {"AUROC": trip(pt["AUROC"], bt[m]["AUROC"]), "AUPRC": trip(pt["AUPRC"], bt[m]["AUPRC"]), "ops": {}}
        for op in OPS:
            blk["ops"][op] = {k: trip(pt[f"{k}@{op}"], bt[m][f"{k}@{op}"]) for k in OPM}
        out[m] = blk
    return out


def paired_block(bt, pts, dets):
    out = {}
    for det in dets:
        if det == "GRU":
            continue
        out[f"{det}-minus-GRU"] = {}
        for k in ["AUROC", "AUPRC", "recall@t_r80", "FPR@t_r80"]:
            pd = ek.paired_delta(bt, det, "GRU", k)
            out[f"{det}-minus-GRU"][k] = {"point_delta": r4(pts[det][k] - pts["GRU"][k]), "boot_mean_delta": r4(pd["delta"]),
                                          "lo": r4(pd["lo"]), "hi": r4(pd["hi"]), "frac_le0": r4(pd["frac_le0"]),
                                          "detected": bool(pd["lo"] > 0 or pd["hi"] < 0)}
    return out


def flag_rates(d, det, idx, seed):
    """Flag rate (fraction with pred True) on rows idx, point = mean over seeds/reps; 95% interval from a
    row bootstrap (B = 2000) with one seed/repetition drawn per replicate (as evalkit)."""
    P = d["pred"][det]; nrep = P["t_r80"].shape[0]; rng = np.random.default_rng(seed)
    out = {}
    for op in OPS:
        pt = float(np.mean([P[op][r][idx].mean() for r in range(nrep)]))
        bs = np.empty(B)
        for b in range(B):
            ib = rng.choice(idx, len(idx), replace=True); r = rng.integers(nrep)
            bs[b] = P[op][r][ib].mean()
        out[op] = [r4(pt), r4(np.percentile(bs, 2.5)), r4(np.percentile(bs, 97.5))]
    return out


def cluster_boot(d, dets, idxE, keyname, seed):
    tm = d["test_meta"]; y = d["y"][idxE].astype(int)
    keys = np.array([tm[i][keyname] for i in idxE]); uniq, inv = np.unique(keys, return_inverse=True)
    members = [np.where(inv == c)[0] for c in range(len(uniq))]
    rng = np.random.default_rng(seed); res = {det: {"AUROC": np.empty(B), "AUPRC": np.empty(B)} for det in dets}
    SE = {det: d["scores"][det][:, idxE] for det in dets}
    for b in range(B):
        cs = rng.integers(len(uniq), size=len(uniq)); ib = np.concatenate([members[c] for c in cs]); yb = y[ib]
        for det in dets:
            S = SE[det]; r = rng.integers(S.shape[0])
            res[det]["AUROC"][b] = ek.auroc(yb, S[r][ib]); res[det]["AUPRC"][b] = ek.auprc(yb, S[r][ib])
    out = {"cluster_key": keyname, "n_clusters": int(len(uniq)), "B": B, "seed": seed,
           "method": "resample whole clusters of pool-E rows with replacement (n_clusters draws), one seed/rep per replicate"}
    for det in dets:
        S = d["scores"][det][:, idxE]
        pa = float(np.mean([ek.auroc(y, s) for s in S])); pp = float(np.mean([ek.auprc(y, s) for s in S]))
        out[det] = {"AUROC": trip(pa, res[det]["AUROC"]), "AUPRC": trip(pp, res[det]["AUPRC"])}
    return out


def mech_single(d, det, idxE, idxF):
    y = d["y"]; tc = d["tool_contrib"][det]; bc = d["beh_contrib"][det]; nrep = tc.shape[0]
    negE = idxE[y[idxE] == 0]; posE = idxE[y[idxE] == 1]
    def per(arr, idx): return [r4(arr[r][idx].mean()) if len(idx) else None for r in range(nrep)]
    ttc = d["train_tool_contrib"][det]; ty = d["train_y"][det]
    out = {"tool_contrib_mean": {"train_neg": [r4(ttc[r][ty[r] == 0].mean()) for r in range(nrep)],
                                 "train_pos": [r4(ttc[r][ty[r] == 1].mean()) for r in range(nrep)],
                                 "test_neg_poolE": per(tc, negE), "test_pos": per(tc, posE), "test_failed": per(tc, idxF)},
           "beh_contrib_mean": {"train_neg": [r4(d["train_beh_contrib"][det][r][ty[r] == 0].mean()) for r in range(nrep)],
                                "train_pos": [r4(d["train_beh_contrib"][det][r][ty[r] == 1].mean()) for r in range(nrep)],
                                "test_neg_poolE": per(bc, negE), "test_pos": per(bc, posE), "test_failed": per(bc, idxF)},
           "bias": [r4(b) for b in d["bias"][det]],
           "unk_norm": [r4(v) for v in d["unk_norm"][det]], "unk_init_norm": [r4(v) for v in d["unk_init_norm"][det]],
           "tool_emb_norm_median": [r4(np.median(v)) for v in d["tool_emb_norms"][det]],
           "tool_emb_norm_mean": [r4(np.mean(v)) for v in d["tool_emb_norms"][det]],
           "n_tool_emb_rows": [int(len(v)) for v in d["tool_emb_norms"][det]],
           "unk_train_steps": [int(v) for v in d["unk_train_steps"][det]],
           "logit_absmax_test": [r4(np.abs(d["logit"][det][r]).max()) for r in range(nrep)],
           "score_frac_exactly_1_poolE": [r4(np.mean(d["scores"][det][r][idxE] == 1.0)) for r in range(nrep)],
           "score_frac_exactly_0_poolE": [r4(np.mean(d["scores"][det][r][idxE] == 0.0)) for r in range(nrep)],
           "train_rows_note": "train_* = the 90% fit partition passed to the training function (validation rows excluded)"}
    for k in ("tool_contrib_mean", "beh_contrib_mean"):
        out[k]["mean_over_seeds"] = {g: r4(np.mean([v for v in vals if v is not None])) for g, vals in out[k].items()
                                     if isinstance(vals, list) and any(v is not None for v in vals)}
    for k in ("unk_norm", "tool_emb_norm_median"):
        out[k + "_mean_over_seeds"] = r4(np.mean(out[k]))
    return out


def mech_cv(d, det, name):
    y = d["y"]; pE = d["pool_E"]; tc = d["tool_contrib_oof"][det]; R = tc.shape[0]
    negE = np.where(pE & (y == 0))[0]; posE = np.where(pE & (y == 1))[0]; fa = np.where(~pE)[0]
    tr = collections.defaultdict(dict)
    pdir = os.path.join(CACHE, "_parts", name, det)
    for f in sorted(os.listdir(pdir)):
        if f.startswith("rep") and f.endswith(".pkl"):
            p = pload(os.path.join(pdir, f))
            tr[f"rep{p['rep']}"][f"fold{p['fold']}"] = {"train_neg": r4(p["train_tool_contrib_summary"][0]),
                                                         "train_pos": r4(p["train_tool_contrib_summary"][1])}
    return {"tool_contrib_mean_oof": {"test_neg_poolE": [r4(tc[r][negE].mean()) for r in range(R)],
                                      "test_pos": [r4(tc[r][posE].mean()) for r in range(R)],
                                      "test_failed": [r4(tc[r][fa].mean()) for r in range(R)]},
            "train_tool_contrib_mean_by_rep_fold": tr,
            "unk_norm_by_rep_fold": np.round(d["unk_norm"][det], 4).tolist(),
            "unk_train_steps_by_rep_fold": d["unk_train_steps"][det].tolist(),
            "unk_train_traces_by_rep_fold": d["unk_train_traces"][det].tolist(),
            "unk_train_traces_pos_by_rep_fold": d["unk_train_traces_pos"][det].tolist(),
            "unk_train_steps_max": int(d["unk_train_steps"][det].max()),
            "unk_train_traces_max": int(d["unk_train_traces"][det].max())}


def train_summary(d, det):
    rows = d["train_diag"][det]
    keys = ["epochs_run", "best_epoch", "best_val_ap", "thr_r80", "thr_val80", "n_fit", "n_fit_pos", "n_val", "n_val_pos",
            "vocab_size_model", "unk_train_steps", "unk_train_traces", "drop_rate_realised", "fit_seconds", "decomp_max_abs_err"]
    return {"units": [r["unit"] for r in rows], **{k: [r4(r[k]) if isinstance(r[k], float) else r[k] for r in rows] for k in keys}}


def r4pops(d, dets, idxE, seed):
    """AgentDojo populations as in r4_control.py / gru_ladder.py: C1 (P0 without benign runs of injection-task
    user tasks) and the model-balanced pool at native counts (positives reweighted to the benign model mix,
    models with >= 50 per class)."""
    tm = d["test_meta"]; y = d["y"].astype(int)
    out = {}
    c1 = np.array([i for i in idxE if not (tm[i]["category"] == "benign" and str(tm[i]["user_task"]).startswith("injection_task"))])
    meth = {det: reps_of(d, det, c1) for det in dets}; yc = y[c1]
    bt = ek.bootstrap(yc, meth, B=B, seed=seed)
    out["C1"] = {"n_pos": int(yc.sum()), "n_neg": int((yc == 0).sum()), "metrics": metric_block(yc, meth, bt),
                 "paired": paired_block(bt, {k: ek.point(yc, v) for k, v in meth.items()}, dets)}
    # model-balanced (r4_control.py lines "model-balanced pooled"), on pool-E positions
    yE = y[idxE]; models = [tm[i]["model"] for i in idxE]
    cnt = collections.Counter((m, int(t)) for m, t in zip(models, yE))
    elig = sorted({m for m in set(models) if cnt[(m, 0)] >= 50 and cnt[(m, 1)] >= 50})
    pool = np.array([i for i, m in enumerate(models) if m in elig])
    bsh = collections.Counter(models[i] for i in pool if yE[i] == 0); ash = collections.Counter(models[i] for i in pool if yE[i] == 1)
    nb, na = sum(bsh.values()), sum(ash.values())
    w = {i: (bsh[models[i]] / nb) / (ash[models[i]] / na) for i in pool if yE[i] == 1}
    methE = {det: reps_of(d, det, idxE) for det in dets}
    btm = ek.bootstrap(yE, methE, B=B, pool=pool, pos_weights=w)          # default seed 12345 as r4_control.py
    out["model_balanced_native_counts"] = {
        "eligible_models": elig, "n_neg": int(sum(1 for i in pool if yE[i] == 0)), "n_pos": int(sum(1 for i in pool if yE[i] == 1)),
        "estimate_note": "[bootstrap mean, 2.5%, 97.5%] exactly as r4_control.py reports model_balanced",
        "metrics": {det: {k: [r4(x) for x in ek.ci(btm[det][k])] for k in ["AUROC", "AUPRC"]} for det in dets},
        "paired": {f"{det}-minus-GRU": {k: {kk: r4(vv) for kk, vv in ek.paired_delta(btm, det, "GRU", k).items()} for k in ["AUROC", "AUPRC"]}
                   for det in dets if det != "GRU"}}
    return out


def summarize_setting(name):
    t0 = time.time()
    d = pload(os.path.join(CACHE, f"{name}.pkl"))
    y = d["y"].astype(int); pE = d["pool_E"]; idxE = np.where(pE)[0]; idxF = np.where(~pE)[0]
    dets = [k for k in DETS if k in d["scores"]]
    seed = 7000 + SETTINGS.index(name)
    yE = y[idxE]
    methods = {det: reps_of(d, det, idxE) for det in dets}
    bt = ek.bootstrap(yE, methods, B=B, seed=seed)
    mb = metric_block(yE, methods, bt)
    pts = {det: ek.point(yE, methods[det]) for det in dets}
    info = d["info"]
    res = {"setting": name, "n_test": int(len(y)), "n_poolE": int(len(idxE)), "n_pos": int(yE.sum()), "n_neg": int((yE == 0).sum()),
           "n_not_poolE": int(len(idxF)), "test_composition": info["test_composition"],
           "n_train": info["n_train"], "n_train_pos": info["n_train_pos"], "kappa": info["kappa"], "vocab_size": info["vocab_size"],
           "bootstrap": {"B": B, "seed": seed, "type": "evalkit.bootstrap, stratified over pool-E test trajectories"},
           "detectors": {}, "paired": paired_block(bt, pts, dets)}
    cats = sorted(collections.Counter(m["category"] for m in d["test_meta"]))
    catarr = np.array([m["category"] for m in d["test_meta"]])
    for det in dets:
        S = d["scores"][det]
        blk = dict(mb[det])
        blk["per_seed_or_rep"] = {"AUROC": [r4(ek.auroc(yE, s[idxE])) for s in S], "AUPRC": [r4(ek.auprc(yE, s[idxE])) for s in S]}
        blk["thresholds"] = {k: (np.round(np.asarray(v, float), 4).tolist() if k in ("t_r80", "t_val80") else v)
                             for k, v in d["thr"][det].items() if k in ("t_r80", "t_val80", "t05")}
        blk["flag_rate_by_category"] = {c: {op: r4(np.mean([d["pred"][det][op][r][catarr == c].mean() for r in range(S.shape[0])]))
                                            for op in OPS} for c in cats}
        blk["mean_score_by_category"] = {c: r4(np.mean(S[:, catarr == c])) for c in cats}
        blk["failed_flag_rate"] = flag_rates(d, det, idxF, seed + 100) if len(idxF) else None
        blk["failed_category"] = sorted(set(catarr[idxF])) if len(idxF) else None
        blk["mechanism"] = mech_single(d, det, idxE, idxF) if name != "WUT" else mech_cv(d, det, name)
        blk["train"] = train_summary(d, det)
        blk["fit_seconds_total"] = r4(sum(d["fit_seconds"][det])); blk["n_fits"] = len(d["fit_seconds"][det])
        res["detectors"][det] = blk
    if name in ("R2", "R4", "WUT"):
        res["cluster_bootstrap_sensitivity"] = cluster_boot(d, dets, idxE, "task", seed + 200)
    if name in ("R4", "WUT"):
        res["agentdojo_populations"] = r4pops(d, dets, idxE, 12345)
        # failed attacks by denial-of-service flag
        dos = np.array([bool(d["test_meta"][i]["is_dos"]) for i in idxF])
        res["failed_flag_rate_by_dos"] = {det: {lab: {op: r4(np.mean([d["pred"][det][op][r][idxF[msk]].mean() for r in range(d["scores"][det].shape[0])]))
                                                      for op in OPS} | {"n": int(msk.sum())}
                                                for lab, msk in [("dos", dos), ("non_dos", ~dos)] if msk.sum()} for det in dets}
    res["seconds"] = r4(time.time() - t0)
    jdump(res, os.path.join(SUMM, f"{name}.json"))
    print(f"summarized {name} in {res['seconds']}s: " + ", ".join(f"{det} AUROC {res['detectors'][det]['AUROC']}" for det in dets), flush=True)
    return res


def cmd_summarize(names, budget=540.0):
    t0 = time.time()
    for name in (names or SETTINGS):
        if os.path.exists(os.path.join(SUMM, f"{name}.json")):
            continue
        if time.time() - t0 > budget:
            print("budget used; call again"); return False
        summarize_setting(name)
    return True


# ---------------------------------------------------------------- exact refit checks (unmodified train_gru)
def cmd_recheck():
    import torch
    import gru_detector as gd
    from featurise import load
    from gru_ladder import stratified_val
    from gru_variants import train_gru_drop
    MP = os.environ["AGENTDRIFT"]
    out = {"torch": torch.__version__, "threads": torch.get_num_threads(), "timestamp": datetime.datetime.now().isoformat(timespec="seconds")}
    st_tr = load(os.path.join(MP, "data/train.jsonl")); st_te = load(os.path.join(MP, "data/test.jsonl"))
    # 1) R1 GRU seed 0 with gru_detector.train_gru (the v0.3 code path of gru_ladder.gru_reps)
    d = pload(os.path.join(CACHE, "R1.pkl")); pE = d["pool_E"]
    keep = [r for r in st_te if r["category"] in ("benign", "hard_negative", "attacked")]
    assert [r["id"] for r in keep] == [m["id"] for m, e in zip(d["test_meta"], pE) if e]
    tr, va = stratified_val(st_tr, 0); t = time.time()
    m, v, s, th = gd.train_gru(tr, va, 0); sc = gd.score_gru(m, v, s, th, keep)["score"]
    out["R1_GRU_seed0_train_gru_vs_cache"] = {"identical": bool(np.array_equal(sc, d["scores"]["GRU"][0][pE])),
                                              "max_abs_diff": float(np.max(np.abs(sc - d["scores"]["GRU"][0][pE]))),
                                              "thr_equal": th == float(d["thr"]["GRU"]["t_r80"][0]), "seconds": round(time.time() - t, 1)}
    # 2) R1 GRU-drop0.2 seed 0 refit with the dropout function (run-to-run determinism of the dropout arm)
    t = time.time(); m2, v2, s2, th2, _ = train_gru_drop(tr, va, 0, 0.2); sc2 = gd.score_gru(m2, v2, s2, th2, keep)["score"]
    out["R1_GRUdrop0.2_seed0_refit_vs_cache"] = {"identical": bool(np.array_equal(sc2, d["scores"]["GRU-drop0.2"][0][pE])),
                                                 "max_abs_diff": float(np.max(np.abs(sc2 - d["scores"]["GRU-drop0.2"][0][pE]))),
                                                 "seconds": round(time.time() - t, 1)}
    # 3) R4 GRU seed 0: train on all AgentDrift, score AgentDojo P0 (one pass, as gru_ladder) and failed (one pass)
    P0 = [json.loads(l) for l in open(os.path.join(ROOT, "agentdojo_traces.jsonl"), encoding="utf-8")]
    FA = [json.loads(l) for l in open(os.path.join(ROOT, "agentdojo_failed.jsonl"), encoding="utf-8")]
    d4 = pload(os.path.join(CACHE, "R4.pkl")); pE4 = d4["pool_E"]
    assert [r["id"] for r in P0 + FA] == [m["id"] for m in d4["test_meta"]]
    tr4, va4 = stratified_val(st_tr + st_te, 0); t = time.time()
    m4, v4, s4, th4 = gd.train_gru(tr4, va4, 0)
    a = gd.score_gru(m4, v4, s4, th4, P0)["score"]; b = gd.score_gru(m4, v4, s4, th4, FA)["score"]
    out["R4_GRU_seed0_train_gru_vs_cache"] = {"P0_identical": bool(np.array_equal(a, d4["scores"]["GRU"][0][pE4])),
                                              "failed_identical": bool(np.array_equal(b, d4["scores"]["GRU"][0][~pE4])),
                                              "max_abs_diff": float(max(np.max(np.abs(a - d4["scores"]["GRU"][0][pE4])), np.max(np.abs(b - d4["scores"]["GRU"][0][~pE4])))),
                                              "seconds": round(time.time() - t, 1)}
    # 4) WUT rep 0 fold 0 exactly as gru_ladder.full(): P0 folds, stratified_val(seed 0), train_gru(seed 0)
    dW = pload(os.path.join(CACHE, "WUT.pkl"))
    keys = [f"{r['id'].split('|')[1]}|{r['id'].split('|')[2]}" for r in P0]
    groups = sorted(set(keys)); rng = np.random.default_rng(0); rng.shuffle(groups)
    fo = {g: j % 5 for j, g in enumerate(groups)}; folds = np.array([fo[k] for k in keys])
    tri = np.where(folds != 0)[0]; tei = np.where(folds == 0)[0]
    trw, vaw = stratified_val([P0[i] for i in tri], 0); t = time.time()
    mw, vw, sw, thw = gd.train_gru(trw, vaw, 0); sw_ = gd.score_gru(mw, vw, sw, thw, [P0[i] for i in tei])["score"]
    out["WUT_rep0_fold0_train_gru_vs_cache"] = {"folds_equal_cache": bool(np.array_equal(folds, dW["fold_of_row"][0][:len(P0)])),
                                                "identical": bool(np.array_equal(sw_, dW["scores"]["GRU"][0][tei])),
                                                "max_abs_diff": float(np.max(np.abs(sw_ - dW["scores"]["GRU"][0][tei]))),
                                                "thr_equal": thw == float(dW["thr"]["GRU"]["t_r80"][0][0]), "seconds": round(time.time() - t, 1)}
    # 5) UNK steps counted on untruncated trajectories of the fit partition (the cache counts encoded, <=16 steps)
    untr = {}
    for name, trn in [("R1", st_tr), ("R4", st_tr + st_te)]:
        for sd in (0, 1, 2):
            f, _ = stratified_val(trn, sd); V = gd.build_tool_vocab(f)
            untr[f"{name}_seed{sd}"] = int(sum(s["tool"] not in V for r in f for s in r["steps"]))
    pool = st_tr + st_te
    for dm in DOMAINS:
        trn = [r for r in pool if r["agent"] != dm]
        for sd in (0, 1, 2):
            f, _ = stratified_val(trn, sd); V = gd.build_tool_vocab(f)
            untr[f"R3_{dm}_seed{sd}"] = int(sum(s["tool"] not in V for r in f for s in r["steps"]))
    out["unk_train_steps_untruncated"] = untr
    jdump(out, os.path.join(SUMM, "_recheck.json"))
    print(json.dumps({k: v for k, v in out.items() if k != "unk_train_steps_untruncated"}, indent=1))
    return out


# ---------------------------------------------------------------- assembly
def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def cmd_assemble():
    per = {n: json.load(open(os.path.join(SUMM, f"{n}.json"), encoding="utf-8")) for n in SETTINGS}
    rc_path = os.path.join(SUMM, "_recheck.json")
    recheck = json.load(open(rc_path, encoding="utf-8")) if os.path.exists(rc_path) else None
    ver = json.load(open(os.path.join(HERE, "gru_p0_verification.json"), encoding="utf-8"))
    arch = json.load(open(os.path.join(ROOT, "archive_v0.3", "gru_results.json"), encoding="utf-8"))
    rerun_log = open(os.path.join(HERE, "rerun_v03_scripts.log"), encoding="utf-8").read()
    rerun_final = "gru_ladder exit 0" in rerun_log
    rerun = json.load(open(os.path.join(ROOT, "gru_results.json"), encoding="utf-8")) if rerun_final else None

    def arch_auc(path, R=arch, m="GRU"):
        x = R
        for p in path:
            x = x[p]
        return x["by_prev"]["native"]["metrics"][m]["AUROC"]
    apath = {"R1": ["R1"], "R1wd_s0": ["R1_wd"], "R2": ["R2"], **{f"R3_{d}": ["R3", d] for d in DOMAINS}}
    anchors = {}
    for s, task_val in TASK_ANCHORS.items():
        a = arch_auc(apath[s]); got = per[s]["detectors"]["GRU"]["AUROC"][0]
        anchors[s] = {"expected_task_3dp": task_val, "v03_archive_point_ci": a, "reproduced_point": got,
                      "reproduced_ci": per[s]["detectors"]["GRU"]["AUROC"][1:], "abs_diff_vs_archive_point": r4(abs(got - a[0])),
                      "match_3dp": f"{got:.3f}" == f"{task_val:.3f}", "match_archive_4dp": abs(got - a[0]) < 5e-5,
                      "per_seed": per[s]["detectors"]["GRU"]["per_seed_or_rep"]["AUROC"]}
    anchors["_note"] = ("v0.3 point = evalkit.point (mean over training seeds 0,1,2) on pool E; R1wd_s0 is the only world-disjoint "
                        "split v0.3 ran. Intervals differ from v0.3 only through the bootstrap seed/B (v0.3: B = 1000).")
    r4v = {}
    for pop in ["P0", "C1"]:
        v03 = arch["R4"][pop]["by_prev"]["native"]["metrics"]
        r4v[pop] = {"v03_buggy_extraction": {"GRU_transfer_AUROC": v03["GRU_transfer"]["AUROC"], "GRU_transfer_AUPRC": v03["GRU_transfer"]["AUPRC"],
                                             "GRU_within_AUROC": v03["GRU_within"]["AUROC"], "GRU_within_AUPRC": v03["GRU_within"]["AUPRC"]}}
        if rerun is not None:
            rr = rerun["R4"][pop]["by_prev"]["native"]["metrics"]
            r4v[pop]["v03_script_rerun_corrected"] = {"GRU_transfer_AUROC": rr["GRU_transfer"]["AUROC"], "GRU_transfer_AUPRC": rr["GRU_transfer"]["AUPRC"],
                                                      "GRU_within_AUROC": rr["GRU_within"]["AUROC"], "GRU_within_AUPRC": rr["GRU_within"]["AUPRC"]}
    r4v["P0"]["cache_R4"] = {det: {"AUROC": per["R4"]["detectors"][det]["AUROC"], "AUPRC": per["R4"]["detectors"][det]["AUPRC"],
                                  "recall@t_r80": per["R4"]["detectors"][det]["ops"]["t_r80"]["recall"],
                                  "FPR@t_r80": per["R4"]["detectors"][det]["ops"]["t_r80"]["FPR"]} for det in per["R4"]["detectors"]}
    r4v["P0"]["cache_WUT"] = {det: {"AUROC": per["WUT"]["detectors"][det]["AUROC"], "AUPRC": per["WUT"]["detectors"][det]["AUPRC"]}
                              for det in per["WUT"]["detectors"]}
    r4v["C1"]["cache_R4"] = {det: per["R4"]["agentdojo_populations"]["C1"]["metrics"][det]["AUROC"] for det in per["R4"]["detectors"]}
    r4v["model_balanced_native_counts"] = {"R4": per["R4"]["agentdojo_populations"]["model_balanced_native_counts"]["metrics"],
                                           "WUT": per["WUT"]["agentdojo_populations"]["model_balanced_native_counts"]["metrics"]}
    checks = {"p0_copy_verification": ver, "exact_refit_checks": recheck, "rerun_v03_final": rerun_final}
    if rerun is not None:
        rr = rerun["R4"]
        checks["R4_P0_point_equals_rerun_gru_ladder"] = {
            "GRU_transfer_AUROC": [per["R4"]["detectors"]["GRU"]["AUROC"][0], rr["P0"]["by_prev"]["native"]["metrics"]["GRU_transfer"]["AUROC"][0]],
            "GRU_transfer_AUPRC": [per["R4"]["detectors"]["GRU"]["AUPRC"][0], rr["P0"]["by_prev"]["native"]["metrics"]["GRU_transfer"]["AUPRC"][0]],
            "GRU_within_AUROC(WUT)": [per["WUT"]["detectors"]["GRU"]["AUROC"][0], rr["P0"]["by_prev"]["native"]["metrics"]["GRU_within"]["AUROC"][0]],
            "GRU_within_AUPRC(WUT)": [per["WUT"]["detectors"]["GRU"]["AUPRC"][0], rr["P0"]["by_prev"]["native"]["metrics"]["GRU_within"]["AUPRC"][0]],
            "C1_GRU_transfer_AUROC": [per["R4"]["agentdojo_populations"]["C1"]["metrics"]["GRU"]["AUROC"][0], rr["C1"]["by_prev"]["native"]["metrics"]["GRU_transfer"]["AUROC"][0]]}
        checks["R4_P0_point_equals_rerun_gru_ladder"]["all_equal_4dp"] = all(abs(a - b) < 5e-5 for a, b in checks["R4_P0_point_equals_rerun_gru_ladder"].values())
        # R1/R2/R3/R1-wd of the rerun equal the archive (AgentDrift unchanged)
        checks["rerun_equals_archive_on_agentdrift"] = all(arch_auc(apath[s], rerun) == arch_auc(apath[s]) for s in apath)
    # consistency with the linear cache (same test rows, order, metadata, coverage; same WUT folds)
    lin = {}
    for n in SETTINGS:
        lp = os.path.join(HERE, "cache", "linear", f"{n}.pkl")
        if not os.path.exists(lp):
            lin[n] = None; continue
        g = pload(os.path.join(CACHE, f"{n}.pkl")); l = pload(lp)
        same_ids = [m["id"] for m in g["test_meta"]] == [m["id"] for m in l["test_meta"]]
        keys = sorted(set(g["test_meta"][0]) & set(l["test_meta"][0]))
        cov_g = np.array([m["cov"] for m in g["test_meta"]], float); cov_l = np.array([m["cov"] for m in l["test_meta"]], float)
        meta_eq = same_ids and all(all((a[k] == b[k]) or (k == "cov") for k in keys) for a, b in zip(g["test_meta"], l["test_meta"]))
        lin[n] = {"same_ids_and_order": same_ids, "same_meta_fields": sorted(set(g["test_meta"][0])) == sorted(set(l["test_meta"][0])),
                  "meta_values_equal_except_cov": bool(meta_eq),
                  "cov_max_abs_diff": float(np.nanmax(np.abs(cov_g - cov_l))) if same_ids else None,
                  "y_equal": bool(np.array_equal(g["y"], l["y"])), "pool_E_equal": bool(np.array_equal(g["pool_E"], l["pool_E"]))}
        if n == "WUT" and "folds" in l:
            lin[n]["folds_equal"] = bool(np.array_equal(g["fold_of_row"], l["folds"]))
            lin[n]["n_failed_only_groups"] = len(g["info"]["failed_only_groups"])
            fo = set(g["info"]["failed_only_groups"])
            lin[n]["n_failed_rows_in_failed_only_groups"] = int(sum(m["task"] in fo for m in g["test_meta"] if not m["in_pool_E"]))
    checks["consistency_with_linear_cache"] = lin
    checks["consistency_with_linear_cache_all_ok"] = all(v is not None and v["same_ids_and_order"] and v["meta_values_equal_except_cov"]
                                                         and v["cov_max_abs_diff"] == 0.0 and v["y_equal"] and v["pool_E_equal"]
                                                         and v.get("folds_equal", True) for v in lin.values())
    files = {}
    for n in SETTINGS:
        p = os.path.join(CACHE, f"{n}.pkl")
        files[n] = {"bytes": os.path.getsize(p), "MB": round(os.path.getsize(p) / 2 ** 20, 2), "sha256": sha256(p),
                    "mtime": datetime.datetime.fromtimestamp(os.path.getmtime(p)).isoformat(timespec="seconds")}
    timing = {n: {det: {"n_fits": per[n]["detectors"][det]["n_fits"], "fit_seconds_total": per[n]["detectors"][det]["fit_seconds_total"],
                        "fit_seconds_mean": r4(per[n]["detectors"][det]["fit_seconds_total"] / per[n]["detectors"][det]["n_fits"])}
                  for det in per[n]["detectors"]} for n in SETTINGS}
    tot_fits = sum(v["n_fits"] for n in timing for v in timing[n].values())
    tot_sec = sum(v["fit_seconds_total"] for n in timing for v in timing[n].values())
    rwd = [per[f"R1wd_s{k}"]["detectors"]["GRU"]["AUROC"][0] for k in range(5)]
    res = {"meta": {"created": datetime.datetime.now().isoformat(timespec="seconds"), "script": "revision/summarize_gru_cache.py",
                    "cache_builder": "revision/build_gru_cache.py", "variant_module": "revision/gru_variants.py",
                    "statistics": __doc__.split("Statistics.")[1].strip(), "B": B,
                    "pool_E": "AgentDrift benign + hard_negative + attacked; AgentDojo P0 (benign + attacked)",
                    "detectors": {"GRU": "p = 0 (pre-registered)", "GRU-drop0.2": "token dropout p = 0.2 (primary)", "GRU-drop0.1": "p = 0.1 (sensitivity)"}},
           "anchors": anchors,
           "R1wd_across_splits_GRU_AUROC": {"values_s0_to_s4": rwd, "mean": r4(np.mean(rwd)), "min": min(rwd), "max": max(rwd)},
           "R4_values": r4v, "checks": checks, "files": files,
           "timing": {"per_setting": timing, "total_fits": tot_fits, "total_fit_seconds": r4(tot_sec), "mean_fit_seconds": r4(tot_sec / tot_fits)},
           "settings": per}
    jdump(res, OUT)
    write_readme(res)
    print(f"wrote {OUT} and {README}")
    return res


def fmt(t):
    return "n/a" if t is None else f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"


def write_readme(res):
    S = res["settings"]; L = []
    L += ["# GRU prediction cache (family `gru`)", "",
          f"Generated {res['meta']['created']} by `revision/summarize_gru_cache.py` from the cache written by "
          "`revision/build_gru_cache.py` (training function in `revision/gru_variants.py`). Every number below is a key of "
          "`revision/gru_cache_results.json` (paths given in brackets).", "",
          "## What is in the cache", "",
          "One pickle per setting, `cache/gru/<setting>.pkl`, in the shared cache format (`setting`, `test_meta`, `y`, `pool_E`, "
          "`scores`, `pred`, `thr`, `info`). Detectors: `GRU` (token dropout p = 0, the pre-registered configuration), "
          "`GRU-drop0.2` (primary), `GRU-drop0.1` (sensitivity). Architecture, constants, optimiser, early stopping, "
          "vocabulary (benign training trajectories only) and seeds are those of `gru_detector.py`/`gru_ladder.py`, imported unmodified.", "",
          "- Scores: single-fit settings `scores[det]` has shape (3, n_test), training seeds 0, 1, 2 (row r = seed r). "
          "WUT: (5, n_test) out-of-fold scores, repetition r uses fold-assignment seed r and training seed r (as `gru_ladder.py`).",
          "- Validation (early stopping, `t_val80`): `gru_ladder.stratified_val` (10% of the setting's training set, stratified, "
          "rng seed 1000 + training seed), for every setting including R1 (as v0.3 `gru_ladder.py`; `data/val.jsonl` is not used).",
          "- `pred[det]`: `t_r80` (20th percentile of in-sample training-positive scores, from `train_gru`), `t_val80` "
          "(20th percentile of validation-positive scores), `t05`; same shape as scores. `thr[det]` holds the values "
          "(single fit: arrays over seeds; WUT: arrays (rep, fold)).",
          "- Test rows always include the unsuccessful attacks (AgentDrift `failed_attack`; AgentDojo `agentdojo_failed.jsonl`, "
          "category `attack_failed`), flagged `in_pool_E = False`; they are scored, never fitted.",
          "- `test_meta[i]['cov']`: fraction of the trajectory's tool calls whose tool is in the benign tool vocabulary of the "
          "setting's full training set (identical to the linear cache). `cov_model[det]` uses the vocabulary the network "
          "actually had (90% fit partition). WUT: `cov` is the mean over repetitions of the out-of-fold value; `cov_by_rep` per repetition.",
          "- Mechanism fields (single fit, per detector, per seed): `tool_contrib` (head weights over the 64 GRU hidden units "
          "dotted with the final hidden state), `beh_contrib` (head weights over the 10 standardised behavioural features dotted "
          "with them), `bias`, `logit` (= tool + beh + bias; max reconstruction error per unit in `train_diag`); "
          "`train_tool_contrib`/`train_beh_contrib` with `train_y`, `train_category`, `train_domain` on the fit partition "
          "(validation rows excluded); `unk_norm`, `unk_init_norm`, `tool_emb_norms` (+ `tool_emb_names`), `unk_train_steps` "
          "(UNK tokens among the encoded training steps, before dropout). WUT: `tool_contrib_oof`, `unk_norm`, "
          "`unk_train_steps`, `unk_train_traces(_pos)` as (rep, fold) arrays; per-unit training summaries in `_parts/WUT/GRU/`.",
          "- `train_diag[det]`: one dict per unit (epochs run, best epoch, validation AP curve, thresholds, realised dropout rate, "
          "sizes, UNK counts, timings). `_parts/<setting>/<det>/` keeps every unit's raw output (resumable build).", "",
          "## Settings, sizes and timings [settings.<S>, files, timing]", "",
          "| setting | detectors | n_test | pool E (pos/neg) | not pool E | n_train (pos) | kappa | vocab | file MB | fits | fit s (mean) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for n in SETTINGS:
        s = S[n]; tm = res["timing"]["per_setting"][n]
        nf = sum(v["n_fits"] for v in tm.values()); ts = sum(v["fit_seconds_total"] for v in tm.values())
        L.append(f"| {n} | {', '.join(s['detectors'])} | {s['n_test']} | {s['n_poolE']} ({s['n_pos']}/{s['n_neg']}) | {s['n_not_poolE']} | "
                 f"{s['n_train']} ({s['n_train_pos']}) | {s['kappa']} | {s['vocab_size']} | {res['files'][n]['MB']} | {nf} | {ts / nf:.1f} |")
    L += ["", f"Total: {res['timing']['total_fits']} fits, {res['timing']['total_fit_seconds'] / 60:.1f} min of fitting "
          f"(mean {res['timing']['mean_fit_seconds']:.1f} s per fit; CPU, torch {res['checks']['p0_copy_verification']['torch']}, "
          f"{res['checks']['p0_copy_verification']['threads']} threads). WUT n_train/kappa/vocab are means over the 25 (rep, fold) units.", "",
          "## Checks [checks]", ""]
    v = res["checks"]["p0_copy_verification"]
    L.append(f"- p = 0 copy vs `gru_detector.train_gru` on R1 seed 0: scores identical = {v['scores_identical']} "
             f"(max abs diff {v['max_abs_score_diff']}), parameters identical = {v['params_identical']}, threshold identical = "
             f"{v['thr_identical']}; batch order unchanged by dropout = {v['stream_check']['batch_orders_identical']} "
             f"(realised p = 0.2 drop rate {v['stream_check']['realised_drop_rate_p02']:.4f}).")
    rc = res["checks"]["exact_refit_checks"]
    if rc:
        L.append(f"- Cached scores equal a fresh unmodified `train_gru` refit: R1 seed 0 {rc['R1_GRU_seed0_train_gru_vs_cache']['identical']}, "
                 f"R4 seed 0 P0 {rc['R4_GRU_seed0_train_gru_vs_cache']['P0_identical']} / failed {rc['R4_GRU_seed0_train_gru_vs_cache']['failed_identical']}, "
                 f"WUT rep 0 fold 0 {rc['WUT_rep0_fold0_train_gru_vs_cache']['identical']} (folds equal {rc['WUT_rep0_fold0_train_gru_vs_cache']['folds_equal_cache']}); "
                 f"GRU-drop0.2 R1 seed 0 refit identical {rc['R1_GRUdrop0.2_seed0_refit_vs_cache']['identical']}.")
    if "R4_P0_point_equals_rerun_gru_ladder" in res["checks"]:
        L.append(f"- R4/WUT GRU point estimates equal those of `gru_ladder.py` re-run on the corrected extraction "
                 f"(`agent-ids/gru_results.json`): {res['checks']['R4_P0_point_equals_rerun_gru_ladder']['all_equal_4dp']}.")
    lc = res["checks"]["consistency_with_linear_cache"]
    L += [f"- Same test rows, order, `test_meta`, `cov`, `y` and `pool_E` as the linear cache in every setting, and WUT folds equal "
          f"the linear cache's: {res['checks']['consistency_with_linear_cache_all_ok']} "
          f"(WUT: {lc['WUT']['n_failed_only_groups']} suite|user_task groups occur only among failed attacks, "
          f"{lc['WUT']['n_failed_rows_in_failed_only_groups']} failed rows) [checks.consistency_with_linear_cache].", "",
          "## Reproduction anchors (GRU, pool E AUROC, mean of 3 seeds) [anchors]", "",
          "| setting | expected (v0.3) | reproduced [95% CI] | match |", "|---|---|---|---|"]
    for s_, a in res["anchors"].items():
        if s_.startswith("_"):
            continue
        L.append(f"| {s_} | {a['expected_task_3dp']:.3f} ({a['v03_archive_point_ci'][0]:.4f}) | {a['reproduced_point']:.4f} "
                 f"[{a['reproduced_ci'][0]:.3f}, {a['reproduced_ci'][1]:.3f}] | {a['match_archive_4dp']} |")
    rw = res["R1wd_across_splits_GRU_AUROC"]
    L += ["", f"R1-wd across the five split seeds: {', '.join(f'{x:.4f}' for x in rw['values_s0_to_s4'])} (mean {rw['mean']:.4f}).", "",
          "## Pool-E results (point [95% stratified bootstrap CI], B = 2000) [settings.<S>.detectors.<det>]", "",
          "| setting | detector | AUROC | AUPRC | recall@t_r80 | FPR@t_r80 | FPR@t_val80 | failed flagged@t_r80 |", "|---|---|---|---|---|---|---|---|"]
    for n in SETTINGS:
        for det, b in S[n]["detectors"].items():
            L.append(f"| {n} | {det} | {fmt(b['AUROC'])} | {fmt(b['AUPRC'])} | {fmt(b['ops']['t_r80']['recall'])} | "
                     f"{fmt(b['ops']['t_r80']['FPR'])} | {fmt(b['ops']['t_val80']['FPR'])} | {fmt(b['failed_flag_rate']['t_r80'] if b['failed_flag_rate'] else None)} |")
    L += ["", "Dropout minus GRU, paired bootstrap on the same test rows [settings.<S>.paired]:", "",
          "| setting | contrast | dAUROC (point; 95% CI) | dAUPRC | detected (AUROC) |", "|---|---|---|---|---|"]
    for n in SETTINGS:
        for c, v in S[n]["paired"].items():
            L.append(f"| {n} | {c} | {v['AUROC']['point_delta']:+.3f} [{v['AUROC']['lo']:+.3f}, {v['AUROC']['hi']:+.3f}] | "
                     f"{v['AUPRC']['point_delta']:+.3f} [{v['AUPRC']['lo']:+.3f}, {v['AUPRC']['hi']:+.3f}] | {v['AUROC']['detected']} |")
    r = res["R4_values"]
    L += ["", "## R4 after the corrected AgentDojo extraction [R4_values]", "",
          f"- v0.3 (buggy extraction) GRU transfer AUROC on P0: {fmt(r['P0']['v03_buggy_extraction']['GRU_transfer_AUROC'])}; "
          f"within-AgentDojo UT: {fmt(r['P0']['v03_buggy_extraction']['GRU_within_AUROC'])}."]
    for det, v in r["P0"]["cache_R4"].items():
        L.append(f"- Cache R4 {det} (transfer, P0): AUROC {fmt(v['AUROC'])}, AUPRC {fmt(v['AUPRC'])}, recall@t_r80 {fmt(v['recall@t_r80'])}, FPR@t_r80 {fmt(v['FPR@t_r80'])}.")
    for det, v in r["P0"]["cache_WUT"].items():
        L.append(f"- Cache WUT {det} (within-AgentDojo UT, P0): AUROC {fmt(v['AUROC'])}, AUPRC {fmt(v['AUPRC'])}.")
    cb = S["R4"].get("cluster_bootstrap_sensitivity", {})
    if cb:
        L.append(f"- Cluster bootstrap over {cb['n_clusters']} suite|user_task clusters (sensitivity), R4 GRU AUROC {fmt(cb['GRU']['AUROC'])}; "
                 f"WUT GRU AUROC {fmt(S['WUT']['cluster_bootstrap_sensitivity']['GRU']['AUROC'])}.")
    mbv = r["model_balanced_native_counts"]
    L.append(f"- Model-balanced (native counts, r4_control.py weighting), GRU AUROC: R4 {fmt(mbv['R4']['GRU']['AUROC'])}, WUT {fmt(mbv['WUT']['GRU']['AUROC'])} "
             "(estimate = bootstrap mean, as r4_control.py).")
    m4 = S["R4"]["detectors"]["GRU"]["mechanism"]
    L += [f"- R4 saturation: the behavioural pathway reaches |logit| up to {max(m4['logit_absmax_test']):.0f} on AgentDojo "
          f"(observation lengths far outside the AgentDrift standardisation); fraction of P0 scores exactly 1.0 per seed "
          f"{m4['score_frac_exactly_1_poolE']}, exactly 0.0 {m4['score_frac_exactly_0_poolE']} [settings.R4.detectors.GRU.mechanism]. "
          "Ties are handled by evalkit's rank-based AUROC.", "",
          "## Known limitations", "",
          "- Intervals here are the v0.3 trajectory-level stratified bootstrap. The plan's primary estimands (prevalence-weighted AUPRC, "
          "cluster bootstraps, Holm families, domain-matched R1 comparators) are computed by the analysis scripts from this cache; the "
          "cluster intervals given here for R2/R4/WUT are a labelled sensitivity only.",
          "- WUT: failed attacks whose suite|user_task group has no P0 row are assigned to folds by default_rng(10000 + rep) "
          "(same rule as the linear cache); any of the five models is out-of-fold for them because training uses P0 rows only.",
          "- `unk_train_steps` counts UNK among the encoded steps (last 16 steps per trajectory); the untruncated count is in "
          "`checks.exact_refit_checks.unk_train_steps_untruncated` for R1, R3 and R4.", ""]
    with open(README, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


def main():
    ok = cmd_summarize([])
    if ok:
        if not os.path.exists(os.path.join(SUMM, "_recheck.json")):
            cmd_recheck()
        cmd_assemble()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "summarize":
        print("COMPLETE" if cmd_summarize(sys.argv[2:]) else "INCOMPLETE")
    elif cmd == "recheck":
        cmd_recheck()
    elif cmd == "assemble":
        cmd_assemble()
    elif cmd == "all":
        main()
    else:
        raise SystemExit(f"unknown command {cmd}")
