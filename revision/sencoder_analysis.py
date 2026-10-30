"""Analysis of the SE detector and the 10-seed GRU (SENCODER_PREREGISTRATION.md Sec. 6, 7, 9, 12, 13).

Reads cache/sencoder_fits, cache/gru10, cache/linear, cache/gru (3-seed reference), sencoder_run/embedding_drift.json,
the embedding manifest, and the published reference values in domain_shift_results.json, oov_mechanism_results.json and
features_obslen_results.json (read only). Writes revision/sencoder_results.json. Nothing is fitted here.

Statistics
  * Ladder (plan Sec. 6): evalkit.bootstrap, B = 1000 for R1 to R3 and 2000 for R4, native prevalence and the
    existing prevalence grids, seeds as gru_ladder.py (R1 5000, R2 5100, R3 5200 + fold index, R1-wd 5300 + split
    seed) and r4_control.py (12345). Point at native prevalence = evalkit.point (mean over seeds).
  * Reversal rule (revision plan 3.4) and R3 minus R1dm: domain_shift.py's estimator (vectorised stratified
    bootstrap, B = 2000, population seed crc32(name), Holm over the five folds), with domain_shift.metrics_rows,
    dstat and holm_family imported unchanged. LR-B and LR-C are recomputed and checked against domain_shift_results.json.
  * Graded coverage and padding: stratified trajectory bootstrap of R1 pool E (B = 2000), metric averaged over seeds
    (and mask draws) inside each replicate, as oov_mechanism.s2_graded does. Thresholds are the training t_r80.
Usage: python sencoder_analysis.py
"""
import os, sys, json, pickle, time, collections, zlib, hashlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
os.environ.setdefault("AGENTDRIFT", os.path.join(os.path.expanduser("~"), "sencoder", "AgentDrift"))
import evalkit as ek  # noqa: E402
import domain_shift as dsh  # noqa: E402  (imported for its estimator functions only; nothing is written by it)
import sencoder_text as stx  # noqa: E402

CACHE = os.path.join(HERE, "cache")
OUT = os.path.join(HERE, "sencoder_results.json")
DOMAINS = ["medical", "banking", "web", "email", "coding"]
B13, B4, B2 = 1000, 2000, 2000
PREV13 = [None, 0.01, 0.02, 0.05, 0.10]
PREV4 = [None, 0.10, 0.05]
POOL_E = ("benign", "hard_negative", "attacked")


def pl(path):
    with open(path, "rb") as f:          # self-produced project caches only
        return pickle.load(f)


_C = {}


def cache(fam, s):
    k = (fam, s)
    if k not in _C:
        p = os.path.join(CACHE, fam, f"{s}.pkl")
        _C[k] = pl(p) if os.path.exists(p) else None
    return _C[k]


def r4(x):
    if x is None:
        return None
    x = float(x)
    return None if np.isnan(x) else round(x, 4)


def reps_of(S, P, idx=None):
    S = np.atleast_2d(np.asarray(S, float))
    out = []
    for r in range(S.shape[0]):
        pr = {op: np.atleast_2d(np.asarray(P[op], bool))[r] for op in ("t_r80", "t05")}
        if idx is None:
            out.append({"score": S[r], "pred": pr})
        else:
            out.append({"score": S[r][idx], "pred": {k: v[idx] for k, v in pr.items()}})
    return out


def methods_for(s, idx, which):
    """{method: reps} restricted to rows idx of setting s."""
    M = {}
    lin = cache("linear", s); se = cache("sencoder_fits", s); g10 = cache("gru10", s); g3 = cache("gru", s)
    for name in which:
        if name in ("LR-B", "LR-C") and lin is not None:
            M[name] = reps_of(lin["scores"][name], lin["pred"][name], idx)
        elif name in ("SE", "SE-W0") and se is not None and name in se["scores"]:
            M[name] = reps_of(se["scores"][name], se["pred"][name], idx)
        elif name == "GRU10" and g10 is not None:
            M[name] = reps_of(g10["scores"]["GRU"], g10["pred"]["GRU"], idx)
        elif name == "GRU3" and g3 is not None:
            M[name] = reps_of(g3["scores"]["GRU"], g3["pred"]["GRU"], idx)
    return M


def f1(y, p):
    tp = np.sum(p & (y == 1)); fp = np.sum(p & (y == 0)); fn = np.sum(~p & (y == 1))
    return float(2 * tp / (2 * tp + fp + fn)) if tp else 0.0


def table(y, methods, prevs, B, seed, pairs, pool=None, pos_weights=None, label=""):
    idx = np.arange(len(y)) if pool is None else np.asarray(pool)
    out = {"n_pos": int(y[idx].sum()), "n_neg": int((y[idx] == 0).sum()), "B": B, "seed": seed, "by_prev": {}}
    for p in prevs:
        nneg = int((y[idx] == 0).sum())
        npos = None if p is None else int(round(p / (1 - p) * nneg))
        bt = ek.bootstrap(y, methods, n_pos=npos, B=B, seed=seed, pool=pool, pos_weights=pos_weights)
        key = "native" if p is None else f"p{int(round(p * 100)):02d}"
        rows = {}
        for m in methods:
            pt = ek.point(y, methods[m], idx) if (p is None and pos_weights is None) else None
            rows[m] = {k: [r4(pt[k] if pt is not None else ek.ci(v)[0]), r4(ek.ci(v)[1]), r4(ek.ci(v)[2])] for k, v in bt[m].items()}
            if p is None:
                rows[m]["F1@t05"] = r4(np.mean([f1(y[idx], rp["pred"]["t05"][idx]) for rp in methods[m]]))
        pr = {f"{a}-minus-{b}": {k: {kk: r4(vv) for kk, vv in ek.paired_delta(bt, a, b, k).items()} for k in ("AUROC", "AUPRC")}
              for a, b in pairs if a in methods and b in methods}
        out["by_prev"][key] = {"n_pos": out["n_pos"] if npos is None else npos, "metrics": rows, "paired": pr}
    return out


# ============================================================================ ladder R1 to R3
def poolE(s):
    lin = cache("linear", s)
    return np.where(np.asarray(lin["pool_E"], bool))[0], np.asarray(lin["y"]).astype(int), lin


def check_alignment(s):
    lin = cache("linear", s); ids = [m["id"] for m in lin["test_meta"]]
    res = {}
    for fam in ("sencoder_fits", "gru10"):
        c = cache(fam, s)
        if c is not None:
            res[fam] = bool([m["id"] for m in c["test_meta"]] == ids and np.array_equal(np.asarray(c["y"]).astype(int), np.asarray(lin["y"]).astype(int)))
    return res


def per_seed_auroc(S, y, idx):
    S = np.atleast_2d(S)
    return [r4(ek.auroc(y[idx], S[r][idx])) for r in range(S.shape[0])]


def sec_ladder():
    R = {}
    settings = [("R1", 5000)] + [(f"R1wd_s{k}", 5300 + k) for k in range(5)] + [("R2", 5100)] + \
               [(f"R3_{d}", 5200 + i) for i, d in enumerate(DOMAINS)]
    for s, seed in settings:
        idx, y, lin = poolE(s)
        M = methods_for(s, idx, ["SE", "GRU10", "LR-B", "LR-C", "GRU3"])
        yE = y[idx]
        t0 = time.time()
        ent = table(yE, M, PREV13, B13, seed, [("SE", "LR-B"), ("SE", "LR-C"), ("SE", "GRU10"), ("GRU10", "LR-C")])
        ent["alignment"] = check_alignment(s)
        ent["coverage"] = lin["info"].get("coverage_poolE", lin["info"].get("kappa"))
        ent["kappa"] = lin["info"].get("kappa")
        se = cache("sencoder_fits", s); g10 = cache("gru10", s)
        ent["per_seed_AUROC"] = {"SE": per_seed_auroc(se["scores"]["SE"], y, idx), "GRU10": per_seed_auroc(g10["scores"]["GRU"], y, idx)}
        for k in ("SE", "GRU10"):
            v = [x for x in ent["per_seed_AUROC"][k] if x is not None]
            ent.setdefault("seed_range_AUROC", {})[k] = r4(max(v) - min(v))
        ent["fit_seconds_SE"] = {"max": r4(max(se["fit_seconds"]["SE"])), "mean": r4(np.mean(se["fit_seconds"]["SE"]))}
        ent["runtime_s"] = round(time.time() - t0, 1)
        R[s] = ent
        print("ladder", s, ent["by_prev"]["native"]["metrics"]["SE"]["AUROC"], flush=True)
    return R


# ============================================================================ D5 comparison point (R2, all test rows, t = 0.5)
def sec_r2_paper_point():
    se = cache("sencoder_fits", "R2"); tm = se["test_meta"]; y = np.asarray(se["y"]).astype(int)
    cat = np.array([m["category"] for m in tm])
    P = np.asarray(se["pred"]["SE"]["t05"], bool)
    per = []
    for r in range(P.shape[0]):
        p = P[r]; tp = int(np.sum(p & (y == 1))); fp = int(np.sum(p & (y == 0))); fn = int(np.sum(~p & (y == 1)))
        d = {"precision": tp / (tp + fp) if tp + fp else None, "recall": tp / (tp + fn), "F1": f1(y, p),
             "flag_rate": {c: float(p[cat == c].mean()) for c in ("attacked", "benign", "failed_attack", "hard_negative")}}
        per.append(d)
    agg = lambda f: {"mean": r4(np.mean([f(d) for d in per])), "min": r4(min(f(d) for d in per)), "max": r4(max(f(d) for d in per))}
    return {"population": "all R2 (task-disjoint) test trajectories, attacked = positive, threshold 0.5",
            "n": int(len(y)), "n_by_category": {c: int((cat == c).sum()) for c in sorted(set(cat))},
            "precision": agg(lambda d: d["precision"]), "recall": agg(lambda d: d["recall"]), "F1": agg(lambda d: d["F1"]),
            "flag_rate": {c: agg(lambda d, c=c: d["flag_rate"][c]) for c in ("attacked", "benign", "failed_attack", "hard_negative")},
            "published_reference_not_ours": {"source": "DriftNet arXiv:2609.10892v1 Table V and VI, task-disjoint test, single run",
                                             "precision": 0.982, "recall": 0.985, "F1": 0.983,
                                             "flag_rate": {"attacked": 0.985, "benign": 0.015, "failed_attack": 0.000, "hard_negative": 0.029}},
            "comparability": "not directly comparable (D6 THOUGHT dropped, D1 fourth feature, D7 validation carve-out, 10 seeds vs seed 42, CPU)"}


# ============================================================================ reversal rule, domain_shift estimator
_POP = {}


def dets_of(s):
    lin = cache("linear", s); se = cache("sencoder_fits", s); g10 = cache("gru10", s); g3 = cache("gru", s)
    D = {}
    for d in ("LR-B", "LR-C"):
        D[d] = (np.atleast_2d(np.asarray(lin["scores"][d], float)), {op: np.atleast_2d(np.asarray(lin["pred"][d][op], bool)) for op in ("t_r80",)})
    D["SE"] = (np.asarray(se["scores"]["SE"], float), {"t_r80": np.asarray(se["pred"]["SE"]["t_r80"], bool)})
    D["GRU10"] = (np.asarray(g10["scores"]["GRU"], float), {"t_r80": np.asarray(g10["pred"]["GRU"]["t_r80"], bool)})
    if g3 is not None:
        D["GRU"] = (np.asarray(g3["scores"]["GRU"], float), {"t_r80": np.asarray(g3["pred"]["GRU"]["t_r80"], bool)})
    return D


def boot_pop(name, s, domain):
    """domain_shift.boot_pop logic (same seeds, same metrics_rows), without its file cache."""
    if name in _POP:
        return _POP[name]
    lin = cache("linear", s)
    y = np.asarray(lin["y"]).astype(int); E = np.asarray(lin["pool_E"], bool)
    dom = np.array([m["domain"] for m in lin["test_meta"]])
    m = E.copy()
    if domain is not None:
        m &= dom == domain
    idx = np.where(m)[0]
    pos, neg = idx[y[idx] == 1], idx[y[idx] == 0]
    npos, nneg = len(pos), len(neg)
    rng = np.random.default_rng(dsh.seed_of(name))
    IB = np.concatenate([pos[rng.integers(0, npos, (B2, npos))], neg[rng.integers(0, nneg, (B2, nneg))]], axis=1)
    base = np.concatenate([pos, neg])[None, :]
    point, boot, by_rep = {}, {}, {}
    for d, (S2, P2) in dets_of(s).items():
        Rn = S2.shape[0]
        rr = np.random.default_rng([dsh.seed_of(name), dsh.seed_of(d)]).integers(0, Rn, B2)
        boot[d] = dsh.metrics_rows(S2[rr[:, None], IB], {op: P2[op][rr[:, None], IB] for op in P2}, npos)
        reps = [dsh.metrics_rows(S2[r][base], {op: P2[op][r][base] for op in P2}, npos) for r in range(Rn)]
        point[d] = {k: float(np.nanmean([x[k][0] for x in reps])) for k in reps[0]}
        by_rep[d] = {k: [float(x[k][0]) for x in reps] for k in ("AUROC", "AUPRC_p10")}
    info = {"name": name, "n_pos": npos, "n_neg": nneg, "point": point, "boot": boot, "point_by_rep": by_rep}
    _POP[name] = info
    return info


def est(pop, d, m):
    a = pop["boot"][d][m]; a = a[~np.isnan(a)]
    return {"point": r4(pop["point"][d][m]), "lo": r4(np.percentile(a, 2.5)), "hi": r4(np.percentile(a, 97.5))}


def sec_reversal():
    R1D = {d: boot_pop(f"R1_dom_{d}", "R1", d) for d in DOMAINS}
    R3 = {d: boot_pop(f"R3_{d}", f"R3_{d}", None) for d in DOMAINS}
    R1P = boot_pop("R1_pooled", "R1", None)
    mets = ["AUROC", "AUPRC_p10"]
    out = {"estimator": "domain_shift.py M3 (stratified, B = 2000, crc32 population seeds, Holm over 5 folds)",
           "values": {"R1_pooled": {d: {m: est(R1P, d, m) for m in mets} for d in R1P["point"]},
                      "R1_domain_matched": {f: {d: {m: est(R1D[f], d, m) for m in mets} for d in R1D[f]["point"]} for f in DOMAINS},
                      "R3": {f: {d: {m: est(R3[f], d, m) for m in mets} for d in R3[f]["point"]} for f in DOMAINS}}}
    # reproduction check of the published LR-B / LR-C values
    ref = json.load(open(os.path.join(HERE, "domain_shift_results.json")))["M3"]
    chk = []
    for f in DOMAINS:
        for d in ("LR-B", "LR-C", "GRU"):
            for m in mets:
                for blk, ours in (("R3", out["values"]["R3"][f]), ("R1_domain_matched", out["values"]["R1_domain_matched"][f])):
                    if d in ours and d in ref[blk][f]:
                        a, b = ours[d][m], ref[blk][f][d][m]
                        chk.append(max(abs(a[k] - b[k]) for k in ("point", "lo", "hi")))
    out["reproduction_max_abs_diff_vs_domain_shift_results"] = r4(max(chk)) if chk else None
    pairs = [("SE", "LR-B"), ("SE", "LR-C"), ("SE", "GRU10"), ("LR-C", "LR-B"), ("GRU10", "LR-C")]
    rev = {}
    for a, b in pairs:
        k = f"{a}-minus-{b}"; rev[k] = {}
        for m in mets:
            r1f = dsh.clean(dsh.holm_family({f: dsh.paired(R1D[f], a, b, m) for f in DOMAINS}))
            r3f = dsh.clean(dsh.holm_family({f: dsh.paired(R3[f], a, b, m) for f in DOMAINS}))
            per = {f: {"R1dm": r1f[f], "R3": r3f[f],
                       "established": bool(r1f[f]["detected_holm"] and dsh.sign_of(r1f[f]) > 0 and r3f[f]["detected_holm"] and dsh.sign_of(r3f[f]) < 0),
                       "established_unadjusted": bool(r1f[f]["detected"] and dsh.sign_of(r1f[f]) > 0 and r3f[f]["detected"] and dsh.sign_of(r3f[f]) < 0)}
                   for f in DOMAINS}
            rev[k][m] = {"per_fold": per, "folds_established": [f for f in DOMAINS if per[f]["established"]],
                         "folds_established_unadjusted": [f for f in DOMAINS if per[f]["established_unadjusted"]],
                         "folds_R3_below_holm": [f for f in DOMAINS if r3f[f]["detected_holm"] and dsh.sign_of(r3f[f]) < 0],
                         "folds_R1dm_advantage_holm": [f for f in DOMAINS if r1f[f]["detected_holm"] and dsh.sign_of(r1f[f]) > 0]}
    out["reversal"] = rev
    out["R3_minus_R1dm"] = {d: {m: dsh.clean(dsh.holm_family({f: dsh.indep(R3[f], R1D[f], d, m) for f in DOMAINS})) for m in mets}
                            for d in ("SE", "GRU10", "LR-B", "LR-C")}
    out["per_seed_points"] = {f: {d: R3[f]["point_by_rep"][d] for d in ("SE", "GRU10")} for f in DOMAINS}
    return out


# ============================================================================ offset analogue (Sec. 7)
def sec_offset():
    out = {"definition": "mean detection logit on held-out pool-E negatives minus mean logit on the fit set's training "
                         "negatives (all negative categories), per fold and seed; also with training negatives restricted "
                         "to benign + hard_negative", "R3": {}, "R1_reference": None}
    for s in [f"R3_{d}" for d in DOMAINS] + ["R1"]:
        se = cache("sencoder_fits", s); y = np.asarray(se["y"]).astype(int); E = np.asarray(se["pool_E"], bool)
        per, per_bh = [], []
        for u in se["units"]["SE"]:
            r = u["seed"]
            zt = np.asarray(se["logit"]["SE"][r], float)
            held = zt[E & (y == 0)].mean()
            tl = np.asarray(u["train_logit"], float); ty = np.asarray(u["train_y"]).astype(int); tc = np.asarray(u["train_category"])
            per.append(float(held - tl[ty == 0].mean()))
            bh = (ty == 0) & np.isin(tc, ["benign", "hard_negative"])
            per_bh.append(float(held - tl[bh].mean()))
        ent = {"per_seed": [r4(v) for v in per], "mean": r4(np.mean(per)), "min": r4(min(per)), "max": r4(max(per)),
               "mean_benign_hardneg_train": r4(np.mean(per_bh))}
        if s == "R1":
            out["R1_reference"] = ent
        else:
            out["R3"][s[3:]] = ent
    vals = [out["R3"][d]["mean"] for d in DOMAINS]
    out["folds_mean_offset_below_0.5"] = [d for d in DOMAINS if out["R3"][d]["mean"] < 0.5]
    out["comparators_published"] = {"LR-C": "+1.653 to +1.899", "GRU": "+6.291 (range -0.782 to +8.617)"}
    out["range_of_fold_means"] = [r4(min(vals)), r4(max(vals))]
    return out


# ============================================================================ R4
def dojo_meta():
    lin = cache("linear", "R4")
    tm = lin["test_meta"]
    y = np.asarray(lin["y"]).astype(int)
    isP0 = np.array([m["category"] in ("benign", "attacked") for m in tm])
    model = np.array([m["model"] for m in tm]); suite = np.array([m["domain"] for m in tm])
    ut = np.array([m["user_task"] for m in tm])
    injb = np.array([m["category"] == "benign" and str(m["user_task"]).startswith("injection_task") for m in tm])
    return tm, y, isP0, model, suite, ut, injb


def r4_methods(idx, with_lin=True, with_c1=False):
    M = {}
    seR4 = cache("sencoder_fits", "R4"); seW = cache("sencoder_fits", "WUT"); g10R4 = cache("gru10", "R4"); g10W = cache("gru10", "WUT")
    linR4 = cache("linear", "R4"); linW = cache("linear", "WUT")
    M["SE_transfer"] = reps_of(seR4["scores"]["SE"], seR4["pred"]["SE"], idx)
    M["SE-W0_transfer"] = reps_of(seR4["scores"]["SE-W0"], seR4["pred"]["SE-W0"], idx)
    M["GRU10_transfer"] = reps_of(g10R4["scores"]["GRU"], g10R4["pred"]["GRU"], idx)
    if with_c1:
        seC = cache("sencoder_fits", "C1"); g10C = cache("gru10", "C1")
        M["SE_within"] = reps_of(seC["scores"]["SE"], seC["pred"]["SE"], idx)
        M["SE-W0_within"] = reps_of(seC["scores"]["SE-W0"], seC["pred"]["SE-W0"], idx)
        M["GRU10_within"] = reps_of(g10C["scores"]["GRU"], g10C["pred"]["GRU"], idx)
    else:
        M["SE_within"] = reps_of(seW["scores"]["SE"], seW["pred"]["SE"], idx)
        M["SE-W0_within"] = reps_of(seW["scores"]["SE-W0"], seW["pred"]["SE-W0"], idx)
        M["GRU10_within"] = reps_of(g10W["scores"]["GRU"], g10W["pred"]["GRU"], idx)
    if with_lin:
        for d in ("LR-B", "LR-C"):
            M[f"{d}_transfer"] = reps_of(linR4["scores"][d], linR4["pred"][d], idx)
            if not with_c1:
                M[f"{d}_within"] = reps_of(linW["scores"][d], linW["pred"][d], idx)
    return M


R4_PAIRS = [("SE_within", "SE_transfer"), ("SE-W0_within", "SE-W0_transfer"), ("GRU10_within", "GRU10_transfer"),
            ("SE_transfer", "SE-W0_transfer"), ("SE_within", "SE-W0_within"), ("SE_transfer", "LR-B_transfer"),
            ("SE_transfer", "LR-C_transfer"), ("SE_transfer", "GRU10_transfer"), ("LR-C_within", "LR-C_transfer")]


def cluster_auroc(y, methods, keys, B=2000, seed=20260927):
    groups = collections.defaultdict(list)
    for i, k in enumerate(keys):
        groups[k].append(i)
    gl = [np.array(v) for v in groups.values()]
    rng = np.random.default_rng(seed)
    out = {m: np.full(B, np.nan) for m in methods}
    for b in range(B):
        ib = np.concatenate([gl[j] for j in rng.integers(0, len(gl), len(gl))])
        for m, reps in methods.items():
            rep = reps[rng.integers(len(reps))]
            out[m][b] = ek.auroc(y[ib], rep["score"][ib])
    return {m: [r4(ek.point(y, methods[m])["AUROC"])] + [r4(x) for x in np.nanpercentile(out[m], [2.5, 97.5])] for m in methods}, len(gl)


def sec_r4():
    tm, y, isP0, model, suite, ut, injb = dojo_meta()
    for fam, s in (("sencoder_fits", "R4"), ("sencoder_fits", "WUT"), ("gru10", "R4"), ("gru10", "WUT")):
        c = cache(fam, s); assert [m["id"] for m in c["test_meta"]] == [m["id"] for m in tm], (fam, s)
    out = {}
    P0 = np.where(isP0)[0]; ALL = np.arange(len(y))
    # P0
    M = r4_methods(P0); yP = y[P0]
    out["P0"] = table(yP, M, PREV4, B4, 12345, R4_PAIRS)
    keys = [f"{suite[i]}|{ut[i]}" for i in P0]
    out["P0"]["cluster_UT_AUROC"], out["P0"]["n_clusters"] = cluster_auroc(yP, {k: M[k] for k in ("SE_transfer", "SE_within", "SE-W0_transfer", "SE-W0_within", "GRU10_transfer", "GRU10_within", "LR-C_transfer", "LR-C_within")}, keys)
    print("R4 P0 done", flush=True)
    # P0F: unsuccessful attacks as additional negatives
    M = r4_methods(ALL)
    out["P0F"] = table(y, M, [None], B4, 12345, R4_PAIRS)
    print("R4 P0F done", flush=True)
    # C1
    C1 = np.where(isP0 & ~injb)[0]
    seC = cache("sencoder_fits", "C1"); assert np.array_equal(np.where(np.asarray(seC["in_C1"]))[0], C1)
    M = r4_methods(C1, with_c1=True); yC = y[C1]
    out["C1"] = table(yC, M, PREV4, B4, 12345, R4_PAIRS)
    print("R4 C1 done", flush=True)
    # model-balanced (r4_control.py definition) on P0
    M = r4_methods(P0); mP = model[P0]
    cnt = collections.Counter((m, int(t)) for m, t in zip(mP, yP))
    elig = sorted({m for m in set(mP) if cnt[(m, 0)] >= 50 and cnt[(m, 1)] >= 50})
    pool = np.array([i for i, m in enumerate(mP) if m in elig])
    bs = collections.Counter(mP[i] for i in pool if yP[i] == 0); as_ = collections.Counter(mP[i] for i in pool if yP[i] == 1)
    nb, na = sum(bs.values()), sum(as_.values())
    w = {i: (bs[mP[i]] / nb) / (as_[mP[i]] / na) for i in pool if yP[i] == 1}
    nneg = int(sum(1 for i in pool if yP[i] == 0))
    mb = {"eligible_models": elig, "n_neg": nneg, "n_pos": int(sum(1 for i in pool if yP[i] == 1))}
    for lab, p in (("native_counts", None), ("p10", 0.10)):
        npos = None if p is None else int(round(p / (1 - p) * nneg))
        bt = ek.bootstrap(yP, M, n_pos=npos, B=B4, pool=pool, pos_weights=w)
        mb[lab] = {"metrics": {m: {k: [r4(x) for x in ek.ci(bt[m][k])] for k in ("AUROC", "AUPRC")} for m in M},
                   "paired": {f"{a}-minus-{b}": {k: {kk: r4(vv) for kk, vv in ek.paired_delta(bt, a, b, k).items()} for k in ("AUROC", "AUPRC")}
                              for a, b in R4_PAIRS if a in M and b in M}}
    out["model_balanced"] = mb
    print("R4 MB done", flush=True)
    # feature-block offset probe (amendment 4): SE R4 models, features zeroed at test, no refit. Not SE-W0.
    seR4 = cache("sencoder_fits", "R4")
    pr = []
    for u in seR4["units"]["SE"]:
        r = u["seed"]
        z = np.asarray(seR4["logit"]["SE"][r], float); zp = np.asarray(u["probe_logit"], float)
        tl = np.asarray(u["train_logit"], float); tpl = np.asarray(u["probe_train_logit"], float); ty = np.asarray(u["train_y"]).astype(int)
        pr.append({"seed": r, "shift_P0_negatives": float((zp - z)[P0][yP == 0].mean()), "shift_P0_positives": float((zp - z)[P0][yP == 1].mean()),
                   "shift_AgentDrift_train_negatives": float((tpl - tl)[ty == 0].mean()),
                   "AUROC_P0_with": float(ek.auroc(yP, z[P0])), "AUROC_P0_zeroed": float(ek.auroc(yP, zp[P0]))})
    out["feature_block_offset_probe"] = {"label": "offset probe, NOT SE-W0 (amendment 4)", "per_seed": pr,
                                         **{k: r4(np.mean([d[k] for d in pr])) for k in pr[0] if k != "seed"}}
    # per-suite share of AgentDojo traces with any email / URL in args (D8)
    P0r = [json.loads(l) for l in open(os.path.join(ROOT, "agentdojo_traces.jsonl"), encoding="utf-8")]
    FAr = [json.loads(l) for l in open(os.path.join(ROOT, "agentdojo_failed.jsonl"), encoding="utf-8")]
    sh = {}
    for lab, recs in (("P0", P0r), ("failed", FAr), ("all", P0r + FAr)):
        for su in ("banking", "slack", "travel", "workspace"):
            rr = [r for r in recs if r["meta"]["suite"] == su]
            anye = [any(stx.EMAIL_RE.findall(json.dumps(s.get("args")).lower()) for s in r["steps"]) for r in rr]
            anyu = [any(stx.URL_RE.findall(json.dumps(s.get("args")).lower()) for s in r["steps"]) for r in rr]
            nz = [any(any(stx.world_features(s, stx.world_of(r))) for s in r["steps"]) for r in rr]
            sh.setdefault(lab, {})[su] = {"n_traces": len(rr), "share_any_email_in_args": r4(np.mean(anye)) if rr else None,
                                          "share_any_url_in_args": r4(np.mean(anyu)) if rr else None,
                                          "share_any_nonzero_world_feature": r4(np.mean(nz)) if rr else None}
    out["agentdojo_feature_support_by_suite"] = sh
    return out


# ============================================================================ graded coverage and padding (R1, pool E)
def strat_IB(y, B, seed):
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    return np.concatenate([pos[rng.integers(0, len(pos), (B, len(pos)))], neg[rng.integers(0, len(neg), (B, len(neg)))]], axis=1), len(pos)


def auc_rows(S, npos):
    from scipy.stats import rankdata
    nneg = S.shape[1] - npos
    Rk = rankdata(S, axis=1)
    return (Rk[:, :npos].sum(1) - npos * (npos + 1) / 2) / (npos * nneg)


def boot_metric_avg(score_list, thr_list, y, IB, npos):
    """score_list: list of score vectors (seeds x draws); returns point and bootstrap arrays of AUROC / recall / FPR averaged."""
    A, Rc, Fp, pA, pR, pF = [], [], [], [], [], []
    for s, t in zip(score_list, thr_list):
        Sb = s[IB]; A.append(auc_rows(Sb, npos)); Pb = Sb >= t
        Rc.append(Pb[:, :npos].mean(1)); Fp.append(Pb[:, npos:].mean(1))
        pA.append(ek.auroc(y, s)); pR.append(np.mean(s[y == 1] >= t)); pF.append(np.mean(s[y == 0] >= t))
    summ = lambda pt, arr: {"point": r4(np.mean(pt)), "lo": r4(np.percentile(np.mean(arr, 0), 2.5)), "hi": r4(np.percentile(np.mean(arr, 0), 97.5))}
    return {"AUROC": summ(pA, A), "recall@t_r80": summ(pR, Rc), "FPR@t_r80": summ(pF, Fp)}, np.mean(A, 0), np.mean(Rc, 0)


def sec_graded_padding():
    se = cache("sencoder_fits", "R1"); lin = cache("linear", "R1")
    E = np.asarray(se["pool_E"], bool); y = np.asarray(se["y"]).astype(int)[E]
    order = np.concatenate([np.where(y == 1)[0], np.where(y == 0)[0]])
    IB, npos = strat_IB(y, B2, 20201)
    units = sorted(se["units"]["SE"], key=lambda u: u["seed"])
    thr = {u["seed"]: float(se["thr"]["SE"]["t_r80"][i]) for i, u in enumerate(se["units"]["SE"])}
    sB = np.asarray(lin["scores"]["LR-B"], float)[E]; tB = float(lin["thr"]["LR-B"]["t_r80"])
    resB, AB, _ = boot_metric_avg([sB], [tB], y, IB, npos)
    om = json.load(open(os.path.join(HERE, "oov_mechanism_results.json")))["s2_graded"]["levels"]
    gm = json.load(open(os.path.join(HERE, "sencoder_run", "graded_masks.json")))
    out = {"graded": {"design": "R1 pool E, masks of oov_mechanism.py reused (paired), tool name -> 'unknown_tool' in the step text; "
                                "mean over 10 seeds x draws; stratified bootstrap B = 2000, rng 20201, conditional on the masks",
                      "LR-B": resB, "levels": {}}}
    for L in ("1.0", "0.75", "0.5", "0.25", "0.0"):
        keys = [k for k in gm["masks"] if k.split("|")[0] == L]
        sl, tl, per_draw = [], [], []
        for k in keys:
            dr = []
            for u in units:
                s = np.asarray(u["masked_scores"][k], float); sl.append(s); tl.append(thr[u["seed"]]); dr.append(ek.auroc(y, s))
            per_draw.append(float(np.mean(dr)))
        res, A, _ = boot_metric_avg(sl, tl, y, IB, npos)
        d = A - AB
        res["AUROC_minus_LR-B"] = {"point": r4(res["AUROC"]["point"] - resB["AUROC"]["point"]), "lo": r4(np.percentile(d, 2.5)), "hi": r4(np.percentile(d, 97.5))}
        res["per_draw_AUROC"] = [r4(v) for v in per_draw]
        res["coverage_mean"] = r4(np.mean([gm["cov_existing"][k] for k in keys]))
        res["published_same_masks"] = {det: om[L]["detectors"][det]["AUROC"] for det in om[L]["detectors"]}
        out["graded"]["levels"][L] = res
    # padding test
    fo = json.load(open(os.path.join(HERE, "features_obslen_results.json")))["M5_3_padding"]
    pad = {"design": "R1 pool E; filler appended to every injection_point observation of attacked trajectories, re-embedded; "
                     "10 R1 models scored at test time, no refit; recall at training t_r80; stratified bootstrap B = 2000, rng 20201",
           "levels": {}}
    base = None
    for k in ("pad0", "pad100", "pad1200"):
        sl = [np.asarray(u["padded_scores"][k], float) for u in units]; tl = [thr[u["seed"]] for u in units]
        res, A, Rc = boot_metric_avg(sl, tl, y, IB, npos)
        if base is None:
            base = (A, Rc, res)
        else:
            res["recall_change_vs_pad0"] = {"point": r4(res["recall@t_r80"]["point"] - base[2]["recall@t_r80"]["point"]),
                                            "lo": r4(np.percentile(Rc - base[1], 2.5)), "hi": r4(np.percentile(Rc - base[1], 97.5))}
            res["AUROC_change_vs_pad0"] = {"point": r4(res["AUROC"]["point"] - base[2]["AUROC"]["point"]),
                                           "lo": r4(np.percentile(A - base[0], 2.5)), "hi": r4(np.percentile(A - base[0], 97.5))}
        pad["levels"][k] = res
    pad["published_same_protocol"] = {
        "LR-C": {k: fo["R1_linear"]["eval"]["metrics"].get(f"LR-C|{k}") for k in ("pad0", "pad100", "pad1200")},
        "LR-B": {k: fo["R1_linear"]["eval"]["metrics"].get(f"LR-B|{k}") for k in ("pad0", "pad100", "pad1200")},
        "GRU": {k: fo["R1_GRU"]["eval"]["metrics"].get(f"GRU|{k}") for k in ("pad0", "pad100", "pad1200")}}
    out["padding"] = pad
    out["checks"] = {"masked_baseline_max_abs_diff": r4(max(u["masked_baseline_max_abs_diff"] for u in units)),
                     "padded_baseline_max_abs_diff": r4(max(u["padded_baseline_max_abs_diff"] for u in units))}
    return out


# ============================================================================ GRU 10 seeds vs 3 seeds
def sec_gru10():
    out = {}
    for s in ["R1"] + [f"R1wd_s{k}" for k in range(5)] + ["R2"] + [f"R3_{d}" for d in DOMAINS] + ["R4", "WUT"]:
        g10, g3 = cache("gru10", s), cache("gru", s)
        if g10 is None or g3 is None:
            continue
        S10 = np.asarray(g10["scores"]["GRU"], float); S3 = np.asarray(g3["scores"]["GRU"], float)
        n = min(S3.shape[0], 3)
        E = np.asarray(g10["pool_E"], bool); y = np.asarray(g10["y"]).astype(int)
        out[s] = {"seeds0to2_max_abs_score_diff_vs_3seed_cache": r4(np.nanmax(np.abs(S10[:n] - S3[:n]))),
                  "seeds0to2_t_r80_pred_mismatches": int(np.sum(np.asarray(g10["pred"]["GRU"]["t_r80"])[:n] != np.asarray(g3["pred"]["GRU"]["t_r80"])[:n])),
                  "AUROC_poolE_mean_3seed": r4(np.mean([ek.auroc(y[E], S3[r][E]) for r in range(S3.shape[0])])),
                  "AUROC_poolE_mean_10seed": r4(np.mean([ek.auroc(y[E], S10[r][E]) for r in range(S10.shape[0])])),
                  "AUROC_poolE_per_seed_10": [r4(ek.auroc(y[E], S10[r][E])) for r in range(S10.shape[0])]}
    return out


def sec_outcomes(L, REV, OFF, R4):
    o = {}
    small = OFF["folds_mean_offset_below_0.5"]
    revB = {m: REV["reversal"]["SE-minus-LR-B"][m]["folds_established"] for m in ("AUROC", "AUPRC_p10")}
    o["Outcome1"] = {"offset_folds_below_0.5": small, "n": len(small), "reversal_vs_LR-B_folds": revB,
                     "met": bool(len(small) >= 4 and not revB["AUROC"] and not revB["AUPRC_p10"])}
    o["Outcome2"] = {"reversal_vs_LR-B_folds": revB, "offset_small_folds": small,
                     "met_AUROC": bool(len(revB["AUROC"]) >= 3), "met_AUPRC_p10": bool(len(revB["AUPRC_p10"]) >= 3),
                     "met": bool((len(revB["AUROC"]) >= 3 or len(revB["AUPRC_p10"]) >= 3) and len(small) >= 4)}
    tP0 = R4["P0"]["by_prev"]["native"]["metrics"]["SE_transfer"]["AUROC"]
    tMB = R4["model_balanced"]["native_counts"]["metrics"]["SE_transfer"]["AUROC"]
    wP0 = R4["P0"]["by_prev"]["native"]["metrics"]["SE_within"]["AUROC"]
    inc = lambda c: c[1] <= 0.5 <= c[2]
    o["Outcome3"] = {"transfer_P0": tP0, "transfer_model_balanced_native": tMB, "within_P0": wP0,
                     "met_point_rule": bool(inc(tP0) and inc(tMB) and wP0[0] > 0.70),
                     "met_lower_bound_rule": bool(inc(tP0) and inc(tMB) and wP0[1] > 0.70)}
    o["Outcome4"] = {"transfer_P0_lo": tP0[1], "transfer_model_balanced_lo": tMB[1], "met": bool(tP0[1] > 0.60 and tMB[1] > 0.60)}
    rng5 = {f: L[f"R3_{f}"]["seed_range_AUROC"]["SE"] for f in DOMAINS}
    o["Outcome5"] = {"R3_across_seed_AUROC_range_SE": rng5, "met": bool(any(v > 0.10 for v in rng5.values()))}
    return o


def main():
    t0 = time.time()
    res = {"meta": {"script": "revision/sencoder_analysis.py",
                    "script_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
                    "plan_sha256": hashlib.sha256(open(os.path.join(ROOT, "SENCODER_PREREGISTRATION.md"), "rb").read()).hexdigest(),
                    "naming": "SE = our reimplementation of the DriftNet architecture (no code or weights were released); "
                              "no number here is DriftNet's reported performance"}}
    for p, k in ((os.path.join(os.environ.get("SENCODER_EMB", os.path.join(CACHE, "sencoder_gpu")), "manifest.json"), "embedding_manifest"),
                 (os.path.join(HERE, "sencoder_run", "embedding_drift.json"), "embedding_drift"),
                 (os.path.join(HERE, "sencoder_run", "smoke.json"), "smoke")):
        if os.path.exists(p):
            d = json.load(open(p)); d.pop("cpu_check", None) if False else None
            if k == "embedding_manifest" and "cpu_check" in d:
                d["cpu_check"] = {kk: v for kk, v in d["cpu_check"].items() if kk != "sample"}
            res[k] = d
    res["ladder"] = sec_ladder()
    res["R2_paper_comparison_point"] = sec_r2_paper_point()
    res["reversal"] = sec_reversal()
    res["offset_analogue"] = sec_offset()
    res["R4"] = sec_r4()
    res["graded_and_padding"] = sec_graded_padding()
    res["gru10"] = sec_gru10()
    res["outcomes"] = sec_outcomes(res["ladder"], res["reversal"], res["offset_analogue"], res["R4"])
    res["meta"]["runtime_s"] = round(time.time() - t0, 1)

    def _d(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return None if np.isnan(o) else float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, (np.bool_,)): return bool(o)
        raise TypeError(type(o))
    tmp = OUT + ".tmp"
    json.dump(res, open(tmp, "w", encoding="utf-8"), indent=1, default=_d)
    os.replace(tmp, OUT)
    print("wrote", OUT, res["meta"]["runtime_s"], "s")


if __name__ == "__main__":
    main()
