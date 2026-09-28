"""Domain shift, thresholds and the world-disjoint control (revision stream `domain_shift`).

Addresses review comments M3, M4, m2 (R2 part), m3 (R3 families and test-set overlap), m5, m12 and m13
(REVIEW_REPORT.md). Binding plan: REVISION_ANALYSIS_PLAN.md, Sec. 3.1-3.6 and Sec. 4 ("Shift size").

Inputs: the shared prediction caches revision/cache/linear/<setting>.pkl and revision/cache/gru/<setting>.pkl.
No detector is fitted here; every score, prediction and threshold is read from the caches (thresholds were
set on training data by the cache builders). The only models fitted in this script are the classifier
two-sample tests of m12 (logistic regression on representation B predicting train-versus-test membership),
which read the raw AgentDrift / AgentDojo records.

Usage (resumable; every section writes revision/domain_shift_work/parts/<section>.json and skips work done):
    python domain_shift.py run [M3 M3world M4 m5 m2 m12 m13 m3overlap]     (no names = all sections)
    python domain_shift.py assemble                          (writes domain_shift_results.json + _summary.md)
    python domain_shift.py all                               (run + assemble)
    python domain_shift.py status
Environment: AGENTDRIFT (AgentDrift clone; needed by m12 only).

Conventions (also in results["meta"]):
  * Populations are pool E (AgentDrift benign, hard_negative, attacked; AgentDojo P0). Unsuccessful attacks are
    never in a population here.
  * Bootstrap: trajectory-level, stratified by class (n_pos and n_neg fixed), B = 2000, one RNG seed per
    population (crc32 of its name), so different populations are independent and every detector within a
    population shares the resampled rows (paired). Multi-seed detectors (GRU*: 3 training seeds) draw one seed
    per replicate, as evalkit.bootstrap does; their point estimate is the mean over seeds (evalkit.point).
  * AUPRC_p10 = prevalence-weighted average precision at pi = 0.10 on the full population (plan 3.1): each
    positive gets weight pi/(1-pi) * n_neg/n_pos, negatives weight 1 (sklearn average_precision_score semantics).
  * Intervals are 2.5/97.5 percentiles of the bootstrap distribution; "point" is the full-sample estimate.
    detected = 95% interval excludes 0 (plan 3.3). p = two-sided bootstrap p value
    2*min((1+#{d<=0})/(B+1), (1+#{d>=0})/(B+1)); p_holm = Holm step-down within the stated family of five;
    detected_holm = p_holm < 0.05.
"""
import os, sys, json, pickle, zlib, time, collections, hashlib
import numpy as np
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import evalkit as ek                                    # noqa: E402  (reused by import, unmodified)
from featurise import load, vectorise                   # noqa: E402
from world_disjoint_r1 import world_disjoint_split      # noqa: E402

CACHE = os.path.join(HERE, "cache")
WORK = os.path.join(HERE, "domain_shift_work")
BOOTDIR = os.path.join(WORK, "boot")
PARTS = os.path.join(WORK, "parts")
OUT_JSON = os.path.join(HERE, "domain_shift_results.json")
OUT_MD = os.path.join(HERE, "domain_shift_summary.md")
DOJO_P0 = os.path.join(ROOT, "agentdojo_traces.jsonl")
DOJO_FAILED = os.path.join(ROOT, "agentdojo_failed.jsonl")

B = 2000
PI = 0.10
ALPHA = 0.05
DOMAINS = ["medical", "banking", "web", "email", "coding"]
PATTERNS = ["attacked_full", "attacked_partial", "attacked_delayed"]
DET_M3 = ["LR-A", "LR-B", "LR-C", "LR-C-drop", "LR-B-noobs", "GRU", "GRU-drop0.2", "TFIDF"]
DET_ALL = DET_M3 + ["LR-C-noobs", "LR-B-minus", "Markov", "IForest", "TFIDF-nodigit", "GRU-drop0.1"]
DET_SEEDS = [f"{g}#seed{r}" for g in ("GRU", "GRU-drop0.2", "GRU-drop0.1") for r in range(3)]
DET_M5 = ["LR-A", "LR-B", "LR-C", "GRU", "TFIDF"]
DET_M2 = ["LR-B", "LR-C", "GRU"]
PAIRS = [("LR-C", "LR-B"), ("GRU", "LR-C"), ("TFIDF", "LR-C"), ("LR-C-drop", "LR-B"), ("GRU-drop0.2", "LR-C")]
REVERSAL_PAIRS = [("LR-C", "LR-B"), ("GRU", "LR-C"), ("LR-C-drop", "LR-B"), ("GRU-drop0.2", "LR-C")]
REMEDY_OF = {("LR-C-drop", "LR-B"): ("LR-C", "LR-B"), ("GRU-drop0.2", "LR-C"): ("GRU", "LR-C")}
SCALAR_METRICS = ["AUROC", "AUPRC_p10", "AUPRC_native"]


def pkey(a, b):
    return f"{a}-minus-{b}"


def seed_of(name):
    return zlib.crc32(name.encode()) & 0x7FFFFFFF


def r4(x):
    if x is None:
        return None
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    x = float(x)
    return None if np.isnan(x) else round(x, 4)


def jdump(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    os.replace(tmp, path)


# ============================================================================ cache access
# The pickles read below are the revision's own prediction caches (written by build_linear_cache.py and
# build_gru_cache.py in this project directory), i.e. trusted local files, not external input.
_SET = {}


def setting(s):
    """Load one setting from the linear cache and (if present) the GRU cache; check row alignment."""
    if s in _SET:
        return _SET[s]
    lin = pickle.load(open(os.path.join(CACHE, "linear", f"{s}.pkl"), "rb"))
    meta = lin["test_meta"]
    y = np.asarray(lin["y"]).astype(int)
    E = np.asarray(lin["pool_E"]).astype(bool)
    ids = [m["id"] for m in meta]
    dets = {}
    for d, sc in lin["scores"].items():
        dets[d] = (np.atleast_2d(np.asarray(sc, float)),
                   {op: np.atleast_2d(np.asarray(v, bool)) for op, v in lin["pred"][d].items()})
    gp = os.path.join(CACHE, "gru", f"{s}.pkl")
    if os.path.exists(gp):
        g = pickle.load(open(gp, "rb"))
        assert [m["id"] for m in g["test_meta"]] == ids, f"row order differs between caches in {s}"
        assert np.array_equal(np.asarray(g["y"]).astype(int), y) and np.array_equal(np.asarray(g["pool_E"]).astype(bool), E)
        for d, sc in g["scores"].items():
            dets[d] = (np.atleast_2d(np.asarray(sc, float)),
                       {op: np.atleast_2d(np.asarray(v, bool)) for op, v in g["pred"][d].items()})
        del g
    for d, (S, P) in list(dets.items()):
        for op, v in P.items():
            assert v.shape == S.shape, (s, d, op, v.shape, S.shape)
        if S.shape[0] > 1:                      # single-seed views of multi-seed detectors (GRU*)
            for r in range(S.shape[0]):
                dets[f"{d}#seed{r}"] = (S[r:r + 1], {op: v[r:r + 1] for op, v in P.items()})
    pc = np.array([PATTERNS.index(m["pattern"]) if m.get("pattern") in PATTERNS else -1 for m in meta])
    _SET[s] = dict(meta=meta, y=y, E=E, dets=dets, ids=np.array(ids), pattern=pc,
                   domain=np.array([m["domain"] for m in meta]),
                   provenance=lin.get("provenance", {}))
    return _SET[s]


def pop_idx(s, domain=None):
    st = setting(s)
    m = st["E"].copy()
    if domain is not None:
        m &= st["domain"] == domain
    return np.where(m)[0]


# ============================================================================ vectorised metrics
def wap_rows(S, npos, wpos):
    """Weighted average precision per row of S (rows = replicates); the first npos columns are positives.
    Positives have weight wpos, negatives 1; ties handled as sklearn (one threshold per distinct score)."""
    n = S.shape[1]
    order = np.argsort(-S, axis=1, kind="stable")
    Ss = np.take_along_axis(S, order, 1)
    ispos = order < npos
    wp = np.where(ispos, wpos, 0.0)
    tp = np.cumsum(wp, 1)
    fp = np.cumsum(~ispos, 1).astype(float)
    end = np.empty(Ss.shape, bool)
    end[:, -1] = True
    end[:, :-1] = Ss[:, :-1] != Ss[:, 1:]
    ar = np.where(end, np.arange(n)[None, :], n)
    nxt = np.minimum.accumulate(ar[:, ::-1], axis=1)[:, ::-1]
    prec = tp / (tp + fp)
    return (wp * np.take_along_axis(prec, nxt, 1)).sum(1) / tp[:, -1]


def metrics_rows(S, preds, npos, Pm=None, ref_mix=None):
    """S: (R, n) scores with the first npos columns positive. preds: {op: bool (R, n)}.
    Pm: (R, npos) pattern codes of the positive columns (AgentDrift only)."""
    n = S.shape[1]
    nneg = n - npos
    Rk = rankdata(S, axis=1)
    out = {"AUROC": (Rk[:, :npos].sum(1) - npos * (npos + 1) / 2) / (npos * nneg),
           "AUPRC_p10": wap_rows(S, npos, PI / (1 - PI) * nneg / npos),
           "AUPRC_native": wap_rows(S, npos, 1.0)}
    for op, Pr in preds.items():
        rec = Pr[:, :npos].mean(1)
        fpr = Pr[:, npos:].mean(1)
        out[f"recall@{op}"] = rec
        out[f"FPR@{op}"] = fpr
        for pi in (0.02, 0.10):
            num = pi * rec
            den = pi * rec + (1 - pi) * fpr
            out[f"precision_p{int(round(pi * 100)):02d}@{op}"] = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
    if Pm is not None:
        Rp = rankdata(S[:, :npos], axis=1)
        c = (Rk[:, :npos] - Rp) / nneg          # per positive: P(negative below) + 0.5 P(tie)
        std = np.zeros(S.shape[0])
        Pr80 = preds.get("t_r80")
        for k, p in enumerate(PATTERNS):
            mk = Pm == k
            cnt = mk.sum(1)
            a = np.where(cnt > 0, (c * mk).sum(1) / np.maximum(cnt, 1), np.nan)
            out[f"AUROC|{p}"] = a
            std = std + ref_mix[k] * a
            if Pr80 is not None:
                out[f"recall@t_r80|{p}"] = np.where(cnt > 0, (Pr80[:, :npos] & mk).sum(1) / np.maximum(cnt, 1), np.nan)
        out["AUROC_patstd"] = std
    return out


_REF_MIX = None


def ref_mix():
    """Reference hijack-pattern mix for the pattern-standardised AUROC: all AgentDrift attacked trajectories
    (union of the five R3 held-out pools = the 10,656 pooled trajectories)."""
    global _REF_MIX
    if _REF_MIX is None:
        cnt = collections.Counter()
        for d in DOMAINS:
            st = setting(f"R3_{d}")
            for i in pop_idx(f"R3_{d}"):
                if st["y"][i] == 1:
                    cnt[st["meta"][i]["pattern"]] += 1
        tot = sum(cnt[p] for p in PATTERNS)
        _REF_MIX = np.array([cnt[p] / tot for p in PATTERNS])
        _REF_MIX_COUNTS.update({p: cnt[p] for p in PATTERNS})
    return _REF_MIX


_REF_MIX_COUNTS = {}


# ============================================================================ bootstrap of one population
_POP = {}


def boot_pop(name, s, domain, dets):
    """Stratified bootstrap of population (setting s, pool E, optional domain). Returns dict with
    point[det][metric], point_by_rep[det][metric] (multi-seed detectors), boot[det][metric] (B,), sizes."""
    if name in _POP and set(dets) <= set(_POP[name]["dets"]):
        return _POP[name]
    npz, js = os.path.join(BOOTDIR, name + ".npz"), os.path.join(BOOTDIR, name + ".json")
    if os.path.exists(npz) and os.path.exists(js):
        info = json.load(open(js))
        if set(dets) <= set(info["dets"]) and info["B"] == B:
            z = np.load(npz)
            boot = collections.defaultdict(dict)
            for k in z.files:
                d, m = k.split("||")
                boot[d][m] = z[k]
            info["boot"] = dict(boot)
            _POP[name] = info
            return info
    t0 = time.time()
    st = setting(s)
    y = st["y"]
    idx = pop_idx(s, domain)
    pos, neg = idx[y[idx] == 1], idx[y[idx] == 0]
    npos, nneg = len(pos), len(neg)
    rng = np.random.default_rng(seed_of(name))
    IB = np.concatenate([pos[rng.integers(0, npos, (B, npos))], neg[rng.integers(0, nneg, (B, nneg))]], axis=1)
    base = np.concatenate([pos, neg])[None, :]
    use_pat = bool(np.all(st["pattern"][pos] >= 0))
    mix = ref_mix() if use_pat else None
    Pm_b = st["pattern"][IB[:, :npos]] if use_pat else None
    Pm_0 = st["pattern"][pos][None, :] if use_pat else None
    point, point_by_rep, boot = {}, {}, {}
    for d in dets:
        S2, P2 = st["dets"][d]
        R = S2.shape[0]
        rr = np.random.default_rng([seed_of(name), seed_of(d)]).integers(0, R, B)
        Sb = S2[rr[:, None], IB]
        Pb = {op: P2[op][rr[:, None], IB] for op in P2}
        boot[d] = metrics_rows(Sb, Pb, npos, Pm_b, mix)
        reps = [metrics_rows(S2[r][base], {op: P2[op][r][base] for op in P2}, npos, Pm_0, mix) for r in range(R)]
        point[d] = {k: float(np.nanmean([m[k][0] for m in reps])) for k in reps[0]}
        if R > 1:
            point_by_rep[d] = {k: [float(m[k][0]) for m in reps] for k in reps[0]}
    pat_counts = ({p: int(np.sum(st["pattern"][pos] == k)) for k, p in enumerate(PATTERNS)} if use_pat else None)
    info = {"name": name, "setting": s, "domain": domain, "dets": list(dets), "B": B, "seed": seed_of(name),
            "n_pos": npos, "n_neg": nneg, "n": npos + nneg, "prevalence": npos / (npos + nneg),
            "pattern_counts": pat_counts, "point": point, "point_by_rep": point_by_rep,
            "ids_sha256": hashlib.sha256("|".join(sorted(st["ids"][idx])).encode()).hexdigest(),
            "seconds": round(time.time() - t0, 1)}
    np.savez_compressed(npz, **{f"{d}||{m}": v for d in boot for m, v in boot[d].items()})
    jdump(info, js)
    info["boot"] = boot
    _POP[name] = info
    print(f"  boot {name}: n={npos}+{nneg} dets={len(dets)} {info['seconds']}s", flush=True)
    return info


def P(name):
    """Named populations used by the sections."""
    if name == "R1_pooled":
        return boot_pop(name, "R1", None, DET_ALL + DET_SEEDS)
    if name.startswith("R1_dom_"):
        return boot_pop(name, "R1", name[7:], DET_ALL + DET_SEEDS)
    if name.startswith("R3_"):
        return boot_pop(name, name, None, DET_ALL + DET_SEEDS)
    if name.startswith("R1wd_s") and name.endswith("_pooled"):
        return boot_pop(name, name[:7], None, DET_M5)
    if name.startswith("R1wd_s") and "_dom_" in name:
        return boot_pop(name, name[:7], name.split("_dom_")[1], DET_M5)
    if name == "R2_pooled":
        return boot_pop(name, "R2", None, DET_M2)
    raise KeyError(name)


# ============================================================================ statistics helpers
def est(pop, det, metric):
    a = pop["boot"][det][metric]
    a = a[~np.isnan(a)]
    pt = pop["point"][det][metric]
    if len(a) == 0:
        return {"point": r4(pt), "lo": None, "hi": None}
    lo, hi = np.percentile(a, [2.5, 97.5])
    return {"point": r4(pt), "lo": r4(lo), "hi": r4(hi)}


def dstat(d, point):
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    n = len(d)
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = min(1.0, 2 * min((1 + np.sum(d <= 0)) / (n + 1), (1 + np.sum(d >= 0)) / (n + 1)))
    return {"point": r4(point), "lo": r4(lo), "hi": r4(hi), "boot_mean": r4(d.mean()), "p": r4(p),
            "p_raw": float(p), "detected": bool(lo > 0 or hi < 0), "B_valid": n}


def paired(pop, a, b, metric):
    return dstat(pop["boot"][a][metric] - pop["boot"][b][metric],
                 pop["point"][a][metric] - pop["point"][b][metric])


def indep(popA, popB, det, metric, detB=None):
    detB = det if detB is None else detB
    return dstat(popA["boot"][det][metric] - popB["boot"][detB][metric],
                 popA["point"][det][metric] - popB["point"][detB][metric])


def holm_family(fam):
    """fam: {key: dstat dict}. Adds p_holm and detected_holm (Holm step-down over the family)."""
    keys = list(fam)
    ps = np.array([fam[k]["p_raw"] for k in keys])
    m = len(ps)
    order = np.argsort(ps, kind="stable")
    adj = np.empty(m)
    run = 0.0
    for j, i in enumerate(order):
        run = max(run, min(1.0, (m - j) * ps[i]))
        adj[i] = run
    for k, a in zip(keys, adj):
        fam[k]["p_holm"] = r4(a)
        fam[k]["detected_holm"] = bool(a < ALPHA)
        fam[k]["family_size"] = m
    return fam


def clean(fam):
    for v in fam.values():
        v.pop("p_raw", None)
    return fam


def sign_of(st):
    return 1 if st["point"] > 0 else -1 if st["point"] < 0 else 0


def dom_list(fam, flag, sign):
    return [k for k, v in fam.items() if v[flag] and sign_of(v) == sign]


# ============================================================================ section M3
def sec_M3():
    R1P = P("R1_pooled")
    R1D = {d: P(f"R1_dom_{d}") for d in DOMAINS}
    R3 = {d: P(f"R3_{d}") for d in DOMAINS}
    mets = ["AUROC", "AUPRC_p10", "recall@t_r80", "FPR@t_r80"]
    out = {"population_sizes": {
        "R1_pooled": {k: R1P[k] for k in ("n_pos", "n_neg", "prevalence")},
        "R1_domain_matched": {d: {k: R1D[d][k] for k in ("n_pos", "n_neg", "prevalence")} for d in DOMAINS},
        "R3": {d: {k: R3[d][k] for k in ("n_pos", "n_neg", "prevalence")} for d in DOMAINS}}}
    tab = lambda pop, dets: {det: {m: est(pop, det, m) for m in mets + ["AUPRC_native"]} for det in dets}
    out["R1_pooled"] = tab(R1P, DET_M3)
    out["R1_domain_matched"] = {d: tab(R1D[d], DET_M3) for d in DOMAINS}
    out["R3"] = {d: tab(R3[d], DET_M3) for d in DOMAINS}
    out["GRU_per_seed_points"] = {
        pop_name: {det: {m: [r4(v) for v in pop["point_by_rep"][det][m]] for m in mets}
                   for det in ("GRU", "GRU-drop0.2")}
        for pop_name, pop in [("R1_pooled", R1P)] + [(f"R1_dom_{d}", R1D[d]) for d in DOMAINS]
        + [(f"R3_{d}", R3[d]) for d in DOMAINS]}
    # paired differences
    pm = ["AUROC", "AUPRC_p10"]
    out["R1_pooled_paired"] = {pkey(a, b): {m: clean({"x": paired(R1P, a, b, m)})["x"] for m in pm} for a, b in PAIRS}
    out["R1_domain_paired"] = {pkey(a, b): {m: clean(holm_family({d: paired(R1D[d], a, b, m) for d in DOMAINS}))
                                            for m in pm} for a, b in PAIRS}
    out["R3_paired"] = {pkey(a, b): {m: clean(holm_family({d: paired(R3[d], a, b, m) for d in DOMAINS}))
                                     for m in pm} for a, b in PAIRS}
    # R3 minus domain-matched R1 and minus pooled R1 (independent bootstraps), Holm over the five folds
    out["R3_minus_R1dm"] = {det: {m: clean(holm_family({d: indep(R3[d], R1D[d], det, m) for d in DOMAINS}))
                                  for m in mets} for det in DET_M3}
    out["R3_minus_R1pooled"] = {det: {m: clean(holm_family({d: indep(R3[d], R1P, det, m) for d in DOMAINS}))
                                      for m in mets} for det in DET_M3}
    # degradation counts (AUROC and AUPRC_p10 decreases)
    deg = {}
    for det in DET_M3:
        deg[det] = {}
        for m in ["AUROC", "AUPRC_p10"]:
            deg[det][m] = {comp: {"unadjusted": dom_list(out[key][det][m], "detected", -1),
                                  "holm": dom_list(out[key][det][m], "detected_holm", -1),
                                  "increase_unadjusted": dom_list(out[key][det][m], "detected", +1),
                                  "increase_holm": dom_list(out[key][det][m], "detected_holm", +1)}
                           for comp, key in [("vs_domain_matched_R1", "R3_minus_R1dm"), ("vs_pooled_R1", "R3_minus_R1pooled")]}
    out["degradation_folds"] = deg
    # reversal rule (plan 3.4)
    rev = {}
    for a, b in REVERSAL_PAIRS:
        k = pkey(a, b)
        rev[k] = {}
        for m in pm:
            r1f, r3f = out["R1_domain_paired"][k][m], out["R3_paired"][k][m]
            per = {}
            for d in DOMAINS:
                adv_h = r1f[d]["detected_holm"] and sign_of(r1f[d]) > 0
                opp_h = r3f[d]["detected_holm"] and sign_of(r3f[d]) < 0
                adv_u = r1f[d]["detected"] and sign_of(r1f[d]) > 0
                opp_u = r3f[d]["detected"] and sign_of(r3f[d]) < 0
                per[d] = {"R1dm_advantage_detected_holm": bool(adv_h), "R3_opposite_detected_holm": bool(opp_h),
                          "established": bool(adv_h and opp_h),
                          "R1dm_advantage_detected_unadjusted": bool(adv_u),
                          "R3_opposite_detected_unadjusted": bool(opp_u),
                          "established_unadjusted": bool(adv_u and opp_u)}
            rev[k][m] = {"per_fold": per,
                         "folds_established": [d for d in DOMAINS if per[d]["established"]],
                         "folds_established_unadjusted": [d for d in DOMAINS if per[d]["established_unadjusted"]],
                         "folds_R1dm_advantage_holm": [d for d in DOMAINS if per[d]["R1dm_advantage_detected_holm"]],
                         "folds_R3_opposite_holm": [d for d in DOMAINS if per[d]["R3_opposite_detected_holm"]],
                         "folds_R3_opposite_unadjusted": [d for d in DOMAINS if per[d]["R3_opposite_detected_unadjusted"]]}
            if (a, b) in REMEDY_OF:
                oa, ob = REMEDY_OF[(a, b)]
                orig = rev[pkey(oa, ob)][m]["per_fold"]
                rev[k][m]["survives_remedy"] = {
                    "definition": f"reversal established for {pkey(oa, ob)} in the fold AND R3 {k} detected negative after Holm",
                    "folds": [d for d in DOMAINS if orig[d]["established"] and per[d]["R3_opposite_detected_holm"]],
                    "folds_original_established": [d for d in DOMAINS if orig[d]["established"]]}
    out["reversal"] = rev
    return out


# ============================================================================ section M3world (sensitivity)
def cluster_boot_pop(name, s, domain, key, dets):
    """Cluster bootstrap of a pool-E population: resample clusters (distinct test_meta[key]) with replacement;
    all detectors share the resampled clusters (paired). Metrics: AUROC, AUPRC_p10 (weights per replicate)."""
    st = setting(s)
    idx = pop_idx(s, domain)
    y = st["y"]
    groups = collections.defaultdict(list)
    for i in idx:
        groups[st["meta"][i][key]].append(i)
    gl = [np.array(v) for v in groups.values()]
    G = len(gl)
    rng = np.random.default_rng(seed_of(name))
    rrs = {d: np.random.default_rng([seed_of(name), seed_of(d)]).integers(0, st["dets"][d][0].shape[0], B) for d in dets}
    boot = {d: {"AUROC": np.full(B, np.nan), "AUPRC_p10": np.full(B, np.nan)} for d in dets}
    for b in range(B):
        ib = np.concatenate([gl[j] for j in rng.integers(0, G, G)])
        ib = ib[np.argsort(-y[ib], kind="stable")]
        npos = int(y[ib].sum())
        if npos == 0 or npos == len(ib):
            continue
        for d in dets:
            m = metrics_rows(st["dets"][d][0][rrs[d][b]][ib][None, :], {}, npos)
            boot[d]["AUROC"][b] = m["AUROC"][0]
            boot[d]["AUPRC_p10"][b] = m["AUPRC_p10"][0]
    base = P(name.replace("W_", ""))
    return {"boot": boot, "point": base["point"], "n_clusters": G, "name": name}


def sec_M3world():
    dets = ["LR-B", "LR-C", "LR-C-drop", "GRU", "GRU-drop0.2"]
    R1D = {d: cluster_boot_pop(f"W_R1_dom_{d}", "R1", d, "world", dets) for d in DOMAINS}
    R3 = {d: cluster_boot_pop(f"W_R3_{d}", f"R3_{d}", None, "world", dets) for d in DOMAINS}
    out = {"definition": ("world-cluster bootstrap (clusters = world keys within each population, resampled with "
                          "replacement, B = 2000), sensitivity analysis for plan 3.5; the primary M3 analysis uses "
                          "the trajectory-level stratified bootstrap"),
           "n_clusters": {"R1_domain_matched": {d: R1D[d]["n_clusters"] for d in DOMAINS},
                          "R3": {d: R3[d]["n_clusters"] for d in DOMAINS}}}
    out["R3_minus_R1dm"] = {det: {m: clean(holm_family({d: indep(R3[d], R1D[d], det, m) for d in DOMAINS}))
                                  for m in ["AUROC", "AUPRC_p10"]} for det in dets}
    pairs = [("LR-C", "LR-B"), ("GRU", "LR-C"), ("LR-C-drop", "LR-B"), ("GRU-drop0.2", "LR-C")]
    out["R1_domain_paired"] = {pkey(a, b): {m: clean(holm_family({d: paired(R1D[d], a, b, m) for d in DOMAINS}))
                                            for m in ["AUROC", "AUPRC_p10"]} for a, b in pairs}
    out["R3_paired"] = {pkey(a, b): {m: clean(holm_family({d: paired(R3[d], a, b, m) for d in DOMAINS}))
                                     for m in ["AUROC", "AUPRC_p10"]} for a, b in pairs}
    rev = {}
    for a, b in pairs:
        k = pkey(a, b)
        rev[k] = {}
        for m in ["AUROC", "AUPRC_p10"]:
            r1f, r3f = out["R1_domain_paired"][k][m], out["R3_paired"][k][m]
            rev[k][m] = {"folds_established": [d for d in DOMAINS if r1f[d]["detected_holm"] and sign_of(r1f[d]) > 0
                                               and r3f[d]["detected_holm"] and sign_of(r3f[d]) < 0]}
    out["reversal"] = rev
    out["degradation_folds_AUROC_holm"] = {det: dom_list(out["R3_minus_R1dm"][det]["AUROC"], "detected_holm", -1) for det in dets}
    return out


# ============================================================================ section M4
def sec_M4():
    R1P = P("R1_pooled")
    R1D = {d: P(f"R1_dom_{d}") for d in DOMAINS}
    R3 = {d: P(f"R3_{d}") for d in DOMAINS}
    rows = ([(det, "t_r80") for det in DET_ALL] + [(g, "t_val80") for g in ("GRU", "GRU-drop0.2", "GRU-drop0.1")]
            + [(g, "t_r80") for g in DET_SEEDS])
    out = {"rows": [f"{d}@{op}" for d, op in rows], "values": {}, "deltas": {}, "status": {}, "tables": {}}
    for det, op in rows:
        rk = f"{det}@{op}"
        out["values"][rk] = {
            "R1_pooled": {mm: est(R1P, det, f"{mm}@{op}") for mm in ("recall", "FPR", "precision_p02", "precision_p10")},
            "R1_domain_matched": {d: {mm: est(R1D[d], det, f"{mm}@{op}") for mm in ("recall", "FPR", "precision_p02", "precision_p10")} for d in DOMAINS},
            "R3": {d: {mm: est(R3[d], det, f"{mm}@{op}") for mm in ("recall", "FPR", "precision_p02", "precision_p10")} for d in DOMAINS}}
        out["deltas"][rk] = {}
        for comp, ref in [("vs_domain_matched_R1", lambda d: R1D[d]), ("vs_pooled_R1", lambda d: R1P)]:
            out["deltas"][rk][comp] = {mm: clean(holm_family({d: indep(R3[d], ref(d), det, f"{mm}@{op}") for d in DOMAINS}))
                                       for mm in ("recall", "FPR")}
        out["status"][rk] = {}
        for comp in ("vs_domain_matched_R1", "vs_pooled_R1"):
            out["status"][rk][comp] = {}
            for flag, lab in (("detected", "unadjusted"), ("detected_holm", "holm")):
                cell = {}
                for d in DOMAINS:
                    parts = []
                    for mm, short in (("FPR", "FPR"), ("recall", "rec")):
                        st_ = out["deltas"][rk][comp][mm][d]
                        if st_[flag]:
                            parts.append(short + ("+" if sign_of(st_) > 0 else "-"))
                    cell[d] = ",".join(parts) if parts else "holds"
                out["status"][rk][comp][lab] = cell
    # compact tables and counts
    for comp in ("vs_domain_matched_R1", "vs_pooled_R1"):
        for lab in ("unadjusted", "holm"):
            t = {rk: out["status"][rk][comp][lab] for rk in out["rows"]}
            out["tables"][f"{comp}|{lab}"] = {
                "cells": t,
                "n_folds_failing": {rk: sum(v != "holds" for v in t[rk].values()) for rk in t},
                "fails_in_all_five": [rk for rk in t if all(v != "holds" for v in t[rk].values())],
                "holds_in_all_five": [rk for rk in t if all(v == "holds" for v in t[rk].values())]}
    out["legend"] = ("cell lists each detected change of R3 relative to the comparator at the stated threshold: "
                     "FPR+ / FPR- = false-positive rate up / down, rec+ / rec- = recall up / down; 'holds' = no "
                     "detected change (plan 3.3 threshold failure = any detected change)")
    return out


# ============================================================================ section m5
def sec_m5():
    R1P = P("R1_pooled")
    R1D = {d: P(f"R1_dom_{d}") for d in DOMAINS}
    mets = ["AUROC", "AUPRC_p10"]
    out = {"seeds": {}, "R1": {det: {m: est(R1P, det, m) for m in mets} for det in DET_M5}}
    out["R1_domain_mix"] = {d: {"n": R1D[d]["n"], "n_pos": R1D[d]["n_pos"], "n_neg": R1D[d]["n_neg"],
                                "frac_of_poolE": r4(R1D[d]["n"] / R1P["n"])} for d in DOMAINS}
    # R1 macro (mean of per-domain AUROC)
    r1_macro_boot = {det: np.mean([R1D[d]["boot"][det]["AUROC"] for d in DOMAINS], axis=0) for det in DET_M5}
    r1_macro_pt = {det: float(np.mean([R1D[d]["point"][det]["AUROC"] for d in DOMAINS])) for det in DET_M5}
    loss_pooled, loss_pooled_h, loss_dom, loss_dom_h = [], [], [], []
    fam_pooled = {det: {} for det in DET_M5}
    for k in range(5):
        s = f"R1wd_s{k}"
        WP = P(f"{s}_pooled")
        WD = {}
        for d in DOMAINS:
            WD[d] = P(f"{s}_dom_{d}")
        st = setting(s)
        worlds_test = len({st["meta"][i]["world"] for i in pop_idx(s)})
        so = {"n_pos": WP["n_pos"], "n_neg": WP["n_neg"], "n_worlds_poolE": worlds_test,
              "domain_mix": {d: {"n": WD[d]["n"], "n_pos": WD[d]["n_pos"], "n_neg": WD[d]["n_neg"],
                                 "frac_of_poolE": r4(WD[d]["n"] / WP["n"])} for d in DOMAINS},
              "metrics": {det: {m: est(WP, det, m) for m in mets} for det in DET_M5},
              "minus_R1": {det: {m: clean({"x": indep(WP, R1P, det, m)})["x"] for m in mets} for det in DET_M5},
              "domain_matched": {}, "macro_AUROC": {}}
        for det in DET_M5:
            fam_pooled[det][k] = so["minus_R1"][det]["AUROC"]
            dm = {d: {"R1wd": est(WD[d], det, "AUROC"), "R1": est(R1D[d], det, "AUROC")} for d in DOMAINS}
            fam = holm_family({d: indep(WD[d], R1D[d], det, "AUROC") for d in DOMAINS})
            for d in DOMAINS:
                dm[d]["delta"] = fam[d]
                if fam[d]["detected"] and fam[d]["point"] < 0:
                    loss_dom.append(f"{det}|s{k}|{d}")
                if fam[d]["detected_holm"] and fam[d]["point"] < 0:
                    loss_dom_h.append(f"{det}|s{k}|{d}")
            clean(fam)
            so["domain_matched"][det] = dm
            mb = np.mean([WD[d]["boot"][det]["AUROC"] for d in DOMAINS], axis=0)
            mp = float(np.mean([WD[d]["point"][det]["AUROC"] for d in DOMAINS]))
            dd = dstat(mb - r1_macro_boot[det], mp - r1_macro_pt[det])
            dd.pop("p_raw")
            so["macro_AUROC"][det] = {"R1wd": {"point": r4(mp), "lo": r4(np.percentile(mb, 2.5)), "hi": r4(np.percentile(mb, 97.5))},
                                      "delta_vs_R1_macro": dd}
            if so["minus_R1"][det]["AUROC"]["detected"] and so["minus_R1"][det]["AUROC"]["point"] < 0:
                loss_pooled.append(f"{det}|s{k}")
        out["seeds"][k] = so
    # Holm over the five split seeds per detector for the pooled-pool-E delta (secondary)
    for det in DET_M5:
        fam = {k: dict(v, p_raw=v["p"] if v["p"] is not None else 1.0) for k, v in fam_pooled[det].items()}
        holm_family(fam)
        for k in range(5):
            out["seeds"][k]["minus_R1"][det]["AUROC"]["p_holm_over_seeds"] = fam[k]["p_holm"]
            out["seeds"][k]["minus_R1"][det]["AUROC"]["detected_holm_over_seeds"] = fam[k]["detected_holm"]
            if fam[k]["detected_holm"] and fam[k]["point"] < 0:
                loss_pooled_h.append(f"{det}|s{k}")
    out["R1_macro_AUROC"] = {det: {"point": r4(r1_macro_pt[det]), "lo": r4(np.percentile(r1_macro_boot[det], 2.5)),
                                   "hi": r4(np.percentile(r1_macro_boot[det], 97.5))} for det in DET_M5}
    out["across_seeds_AUROC"] = {det: {"values": [out["seeds"][k]["metrics"][det]["AUROC"]["point"] for k in range(5)]}
                                 for det in DET_M5}
    for det in DET_M5:
        v = out["across_seeds_AUROC"][det]["values"]
        out["across_seeds_AUROC"][det].update({"mean": r4(np.mean(v)), "min": r4(min(v)), "max": r4(max(v))})
    increases = [f"{det}|s{k}" for det in DET_M5 for k in range(5)
                 if out["seeds"][k]["minus_R1"][det]["AUROC"]["detected"] and out["seeds"][k]["minus_R1"][det]["AUROC"]["point"] > 0]
    out["detected_losses"] = {"pooled_AUROC_unadjusted": loss_pooled, "pooled_AUROC_holm_over_seeds": loss_pooled_h,
                              "domain_matched_AUROC_unadjusted": loss_dom, "domain_matched_AUROC_holm_over_domains": loss_dom_h,
                              "pooled_AUROC_increases_unadjusted": increases,
                              "pooled_AUPRC_p10_losses_unadjusted": [f"{det}|s{k}" for det in DET_M5 for k in range(5)
                                                                     if out["seeds"][k]["minus_R1"][det]["AUPRC_p10"]["detected"]
                                                                     and out["seeds"][k]["minus_R1"][det]["AUPRC_p10"]["point"] < 0],
                              "macro_AUROC_losses_unadjusted": [f"{det}|s{k}" for det in DET_M5 for k in range(5)
                                                                if out["seeds"][k]["macro_AUROC"][det]["delta_vs_R1_macro"]["detected"]
                                                                and out["seeds"][k]["macro_AUROC"][det]["delta_vs_R1_macro"]["point"] < 0]}
    return out


# ============================================================================ section m2
def cluster_boot(s, key, dets, name, B_=B):
    """Cluster bootstrap over pool E of setting s: clusters = distinct values of test_meta[key]."""
    st = setting(s)
    idx = pop_idx(s)
    y = st["y"]
    labs = [st["meta"][i][key] for i in idx]
    groups = collections.defaultdict(list)
    for i, g in zip(idx, labs):
        groups[g].append(i)
    gl = [np.array(v) for v in groups.values()]
    G = len(gl)
    rng = np.random.default_rng(seed_of(name))
    out = {d: {"AUROC": np.empty(B_), "AUPRC_p10": np.empty(B_)} for d in dets}
    rrs = {d: np.random.default_rng([seed_of(name), seed_of(d)]).integers(0, st["dets"][d][0].shape[0], B_) for d in dets}
    npos_b = np.empty(B_)
    for b in range(B_):
        pick = rng.integers(0, G, G)
        ib = np.concatenate([gl[j] for j in pick])
        yb = y[ib]
        order = np.argsort(-yb, kind="stable")          # positives first
        ib = ib[order]
        npos = int(yb.sum())
        nneg = len(ib) - npos
        npos_b[b] = npos
        for d in dets:
            sc = st["dets"][d][0][rrs[d][b]][ib][None, :]
            if npos == 0 or nneg == 0:
                out[d]["AUROC"][b] = np.nan
                out[d]["AUPRC_p10"][b] = np.nan
                continue
            m = metrics_rows(sc, {}, npos)
            out[d]["AUROC"][b] = m["AUROC"][0]
            out[d]["AUPRC_p10"][b] = m["AUPRC_p10"][0]
    sizes = [len(v) for v in gl]
    pos_per = [int(y[v].sum()) for v in gl]
    info = {"n_clusters": G, "cluster_size_min": int(min(sizes)), "cluster_size_median": float(np.median(sizes)),
            "cluster_size_max": int(max(sizes)), "clusters_without_positive": int(sum(p == 0 for p in pos_per)),
            "clusters_without_negative": int(sum(p == len(v) for p, v in zip(pos_per, gl))),
            "n_pos_resampled_min": int(npos_b.min()), "n_pos_resampled_max": int(npos_b.max()), "seed": seed_of(name)}
    return out, info


def sec_m2():
    R1P = P("R1_pooled")
    R2P = P("R2_pooled")
    mets = ["AUROC", "AUPRC_p10"]
    st2 = setting("R2")
    idx2 = pop_idx("R2")
    test_tasks = sorted({st2["meta"][i]["task"] for i in idx2})
    out = {"R2_n_distinct_test_tasks_poolE": len(test_tasks),
           "R2_n_distinct_test_tasks_all_rows": len({m["task"] for m in st2["meta"]}),
           "R2_test_tasks_by_domain": dict(collections.Counter(st2["meta"][i]["domain"] for i in idx2
                                                               if st2["meta"][i]["task"] in set(test_tasks)).items()),
           "R2_n_distinct_tasks_by_domain": {d: len({st2["meta"][i]["task"] for i in idx2 if st2["meta"][i]["domain"] == d}) for d in DOMAINS}}
    mp = os.environ.get("AGENTDRIFT")
    if mp:
        tr_tasks = {r["task"] for r in load(os.path.join(mp, "data_taskdisjoint", "train.jsonl"))}
        out["R2_test_tasks_also_in_R2_train"] = len(set(test_tasks) & tr_tasks)
        st1 = setting("R1")
        r1_train_tasks = {r["task"] for r in load(os.path.join(mp, "data", "train.jsonl"))}
        out["R1_test_tasks_also_in_R1_train"] = len({m["task"] for m in st1["meta"]} & r1_train_tasks)
    st1 = setting("R1")
    idx1 = pop_idx("R1")
    out["R1_n_distinct_worlds_poolE"] = len({st1["meta"][i]["world"] for i in idx1})
    out["R1_n_distinct_tasks_poolE"] = len({st1["meta"][i]["task"] for i in idx1})
    r2c, r2c_info = cluster_boot("R2", "task", DET_M2, "m2_R2_taskcluster")
    r1w, r1w_info = cluster_boot("R1", "world", DET_M2, "m2_R1_worldcluster")
    out["R2_taskcluster_info"] = r2c_info
    out["R1_worldcluster_info"] = r1w_info
    pts = {"R1": R1P["point"], "R2": R2P["point"]}

    def ci_arr(a, pt):
        a = a[~np.isnan(a)]
        return {"point": r4(pt), "lo": r4(np.percentile(a, 2.5)), "hi": r4(np.percentile(a, 97.5)), "B_valid": int(len(a))}

    res = {}
    for det in DET_M2:
        res[det] = {}
        for m in mets:
            r1t, r2t = R1P["boot"][det][m], R2P["boot"][det][m]
            r1wc, r2tc = r1w[det][m], r2c[det][m]
            p1, p2 = pts["R1"][det][m], pts["R2"][det][m]
            e = {"R1_trajectory": ci_arr(r1t, p1), "R1_worldcluster": ci_arr(r1wc, p1),
                 "R2_trajectory": ci_arr(r2t, p2), "R2_taskcluster": ci_arr(r2tc, p2)}
            dd = {"primary_R2taskcluster_minus_R1trajectory": dstat(r2tc - r1t, p2 - p1),
                  "sensitivity_R2taskcluster_minus_R1worldcluster": dstat(r2tc - r1wc, p2 - p1),
                  "reference_v03_R2trajectory_minus_R1trajectory": dstat(r2t - r1t, p2 - p1)}
            for v in dd.values():
                v.pop("p_raw")
            e["R2_minus_R1"] = dd
            res[det][m] = e
    out["estimates"] = res
    return out


# ============================================================================ section m12
def _dojo(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def c2st(Xtr, Xte, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import StratifiedKFold
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    X = np.vstack([Xtr, Xte])
    z = np.r_[np.zeros(len(Xtr), int), np.ones(len(Xte), int)]
    oof = np.empty(len(z))
    fold_auc, conv = [], []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, z):
        mdl = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced"))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", ConvergenceWarning)
            mdl.fit(X[tr], z[tr])
            conv.append(not any(issubclass(x.category, ConvergenceWarning) for x in w))
        oof[te] = mdl.predict_proba(X[te])[:, 1]
        fold_auc.append(ek.auroc(z[te], oof[te]))
    auc = ek.auroc(z, oof)
    # stratified bootstrap of the out-of-fold scores (test resampling only; refit variability omitted)
    rng = np.random.default_rng(seed_of("m12") + len(z))
    i1, i0 = np.where(z == 1)[0], np.where(z == 0)[0]
    bs = np.empty(B)
    s1, s0 = oof[i1], oof[i0]
    for b in range(B):
        a1 = s1[rng.integers(0, len(i1), len(i1))]
        a0 = s0[rng.integers(0, len(i0), len(i0))]
        sc = np.r_[a1, a0][None, :]
        bs[b] = metrics_rows(sc, {}, len(a1))["AUROC"][0]
    lr_full = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced")).fit(X, z)
    coef = lr_full[-1].coef_[0]
    return {"AUROC_oof": r4(auc), "lo": r4(np.percentile(bs, 2.5)), "hi": r4(np.percentile(bs, 97.5)),
            "AUROC_fold_mean": r4(np.mean(fold_auc)), "AUROC_fold_sd": r4(np.std(fold_auc, ddof=1)),
            "AUROC_folds": [r4(a) for a in fold_auc], "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
            "all_folds_converged": bool(all(conv)),
            "std_coef_full_fit": {n: r4(c) for n, c in zip(["len", "uniq", "diversity", "repeats", "write_n", "write_frac",
                                                            "max_args", "mean_args", "max_obs", "mean_obs"], coef)}}


def sec_m12():
    mp = os.environ["AGENTDRIFT"]
    st_tr = load(os.path.join(mp, "data", "train.jsonl"))
    st_te = load(os.path.join(mp, "data", "test.jsonl"))
    td_tr = load(os.path.join(mp, "data_taskdisjoint", "train.jsonl"))
    td_te = load(os.path.join(mp, "data_taskdisjoint", "test.jsonl"))
    pool = st_tr + st_te
    p0, failed = _dojo(DOJO_P0), _dojo(DOJO_FAILED)
    splits = {"R1": (st_tr, st_te), "R2": (td_tr, td_te)}
    tr_i, te_i = world_disjoint_split(pool, 0)
    splits["R1wd_s0"] = ([pool[i] for i in tr_i], [pool[i] for i in te_i])
    for d in DOMAINS:
        splits[f"R3_{d}"] = ([r for r in pool if r["agent"] != d], [r for r in pool if r["agent"] == d])
    splits["R4"] = (pool, p0 + failed)
    fail_cats = {"failed_attack", "attack_failed"}
    feat = lambda recs: np.array([vectorise(r, {})[0] for r in recs], float)
    out = {"definition": ("classifier two-sample test: StandardScaler + LogisticRegression(max_iter=5000, "
                          "class_weight='balanced') on representation B (10 behavioural features) predicting test "
                          "membership; StratifiedKFold(5, shuffle=True, random_state=0); AUROC of the pooled "
                          "out-of-fold scores with a stratified bootstrap interval (B=2000) over those scores. "
                          "0.5 = train and test indistinguishable in representation B"),
           "settings": {}}
    for s, (tr, te) in splits.items():
        cache_ids = list(setting(s)["ids"])
        assert [r["id"] for r in te] == cache_ids, f"test rows of {s} differ from the cache"
        Xtr, Xte = feat(tr), feat(te)
        mtr = np.array([r["category"] not in fail_cats for r in tr])
        mte = np.array([r["category"] not in fail_cats for r in te])
        t0 = time.time()
        out["settings"][s] = {"all_rows": c2st(Xtr, Xte), "poolE_rows": c2st(Xtr[mtr], Xte[mte]),
                              "test_rows_equal_cache": True}
        print(f"  m12 {s}: all {out['settings'][s]['all_rows']['AUROC_oof']} poolE "
              f"{out['settings'][s]['poolE_rows']['AUROC_oof']} ({time.time() - t0:.1f}s)", flush=True)
    return out


# ============================================================================ section m13
def sec_m13():
    R3 = {d: P(f"R3_{d}") for d in DOMAINS}
    R1D = {d: P(f"R1_dom_{d}") for d in DOMAINS}
    mix = ref_mix()
    out = {"reference_mix": {"counts": dict(_REF_MIX_COUNTS), "fractions": {p: r4(v) for p, v in zip(PATTERNS, mix)},
                             "definition": "all 4,706 AgentDrift attacked trajectories (union of the five R3 held-out pools)"},
           "folds": {}}
    for d in DOMAINS:
        pc = R3[d]["pattern_counts"]
        npos = R3[d]["n_pos"]
        # training positives of the fold = attacked rows of the other four domains
        tr_cnt = collections.Counter()
        for o in DOMAINS:
            if o != d:
                for p, c in R3[o]["pattern_counts"].items():
                    tr_cnt[p] += c
        trn = sum(tr_cnt.values())
        f = {"n_pos": npos, "n_neg": R3[d]["n_neg"], "prevalence": r4(R3[d]["prevalence"]),
             "pattern_counts": pc, "pattern_fractions": {p: r4(pc[p] / npos) for p in PATTERNS},
             "train_positive_pattern_counts": dict(tr_cnt),
             "train_positive_pattern_fractions": {p: r4(tr_cnt[p] / trn) for p in PATTERNS},
             "R1_domain_matched_pattern_counts": R1D[d]["pattern_counts"],
             "R1_domain_matched_prevalence": r4(R1D[d]["prevalence"]),
             "by_detector": {}}
        for det in ("LR-B", "LR-C", "GRU", "LR-C-drop", "GRU-drop0.2"):
            f["by_detector"][det] = {
                "AUROC": est(R3[d], det, "AUROC"),
                "AUROC_pattern_standardised": est(R3[d], det, "AUROC_patstd"),
                "AUROC_by_pattern": {p: est(R3[d], det, f"AUROC|{p}") for p in PATTERNS},
                "recall@t_r80_by_pattern": {p: est(R3[d], det, f"recall@t_r80|{p}") for p in PATTERNS}}
        f["paired_patstd"] = {}
        for a, b in [("LR-C", "LR-B"), ("GRU", "LR-C")]:
            dd = paired(R3[d], a, b, "AUROC_patstd")
            dd.pop("p_raw")
            f["paired_patstd"][pkey(a, b)] = dd
        out["folds"][d] = f
    fam = {d: dict(out["folds"][d]["paired_patstd"]["LR-C-minus-LR-B"]) for d in DOMAINS}
    for d in DOMAINS:
        fam[d]["p_raw"] = fam[d]["p"]
    holm_family(fam)
    for d in DOMAINS:
        out["folds"][d]["paired_patstd"]["LR-C-minus-LR-B"]["p_holm"] = fam[d]["p_holm"]
        out["folds"][d]["paired_patstd"]["LR-C-minus-LR-B"]["detected_holm"] = fam[d]["detected_holm"]
    full = {d: out["folds"][d]["pattern_fractions"]["attacked_full"] for d in DOMAINS}
    prev = {d: out["folds"][d]["prevalence"] for d in DOMAINS}
    others = [x for x in DOMAINS if x != "banking"]
    out["banking_caveat"] = {
        "banking_full_fraction": full["banking"], "other_folds_full_fraction_min": min(full[x] for x in others),
        "other_folds_full_fraction_max": max(full[x] for x in others),
        "banking_prevalence": prev["banking"], "other_folds_prevalence": sorted({prev[x] for x in others}),
        "banking_n_pos": out["folds"]["banking"]["n_pos"]}
    return out


# ============================================================================ section m3
def sec_m3():
    st1 = setting("R1")
    ids1_all = {d: set(st1["ids"][st1["domain"] == d]) for d in DOMAINS}
    ids1_E = {d: set(st1["ids"][st1["E"] & (st1["domain"] == d)]) for d in DOMAINS}
    out = {"per_domain": {}}
    for d in DOMAINS:
        s3 = setting(f"R3_{d}")
        ids3_all, ids3_E = set(s3["ids"]), set(s3["ids"][s3["E"]])
        out["per_domain"][d] = {
            "R1_test_rows_domain_all": len(ids1_all[d]), "R1_test_rows_domain_poolE": len(ids1_E[d]),
            "R3_heldout_rows_all": len(ids3_all), "R3_heldout_rows_poolE": len(ids3_E),
            "shared_all_rows": len(ids1_all[d] & ids3_all), "shared_poolE": len(ids1_E[d] & ids3_E),
            "shared_frac_of_R3_poolE": r4(len(ids1_E[d] & ids3_E) / len(ids3_E)),
            "R1_domain_rows_all_contained_in_R3": ids1_all[d] <= ids3_all}
        out["per_domain"][d]["shared_poolE_pos"] = int(sum(st1["y"][i] for i in np.where(st1["E"] & (st1["domain"] == d))[0]))
    tot_sh = sum(out["per_domain"][d]["shared_poolE"] for d in DOMAINS)
    tot_r3 = sum(out["per_domain"][d]["R3_heldout_rows_poolE"] for d in DOMAINS)
    out["total_shared_poolE"] = tot_sh
    out["total_R3_poolE"] = tot_r3
    out["total_shared_frac"] = r4(tot_sh / tot_r3)
    # sensitivity: joint bootstrap that resamples the R3 held-out pool and evaluates the domain-matched R1 metric
    # on the resampled rows that are R1 test rows (keeps the overlap), versus the independent bootstraps of M3
    R1D = {d: P(f"R1_dom_{d}") for d in DOMAINS}
    R3 = {d: P(f"R3_{d}") for d in DOMAINS}
    sens = {}
    for d in DOMAINS:
        s3 = setting(f"R3_{d}")
        pos1 = {i: k for k, i in enumerate(st1["ids"])}
        idx3 = pop_idx(f"R3_{d}")
        pos3, neg3 = idx3[s3["y"][idx3] == 1], idx3[s3["y"][idx3] == 0]
        map1 = np.array([pos1.get(x, -1) for x in s3["ids"]])       # R3 row -> R1 row (or -1)
        rng = np.random.default_rng(seed_of(f"m3_joint_{d}"))
        dd = {det: np.empty(B) for det in ("LR-B", "LR-C")}
        for b in range(B):
            ib = np.r_[pos3[rng.integers(0, len(pos3), len(pos3))], neg3[rng.integers(0, len(neg3), len(neg3))]]
            j1 = map1[ib]
            j1 = j1[j1 >= 0]
            y3, y1 = s3["y"][ib], st1["y"][j1]
            for det in dd:
                a3 = ek.auroc(y3, s3["dets"][det][0][0][ib])
                a1 = ek.auroc(y1, st1["dets"][det][0][0][j1])
                dd[det][b] = a3 - a1
        sens[d] = {}
        for det in dd:
            pt = R3[d]["point"][det]["AUROC"] - R1D[d]["point"][det]["AUROC"]
            j = dstat(dd[det], pt)
            j.pop("p_raw")
            ind = indep(R3[d], R1D[d], det, "AUROC")
            ind.pop("p_raw")
            sens[d][det] = {"joint_overlap_bootstrap": j, "independent_bootstrap": ind,
                            "width_ratio_joint_over_independent": r4((j["hi"] - j["lo"]) / (ind["hi"] - ind["lo"]))}
    out["overlap_sensitivity_R3_minus_R1dm_AUROC"] = sens
    return out


# ============================================================================ revision 2 (verifier fixes)
# Added after an independent verification of the first version of this stream: an overlap-aware bootstrap for m5
# (the R1 and R1-wd test pools share ~300 trajectories), a Monte Carlo stability check at B = 10,000 for every
# Holm family that contains a borderline p value, and the C2ST for the remaining settings plus the spread over
# fold-assignment seeds. Nothing computed by the sections above is changed.
B_HI = 10000
MC_WINDOW = (0.03, 0.07)
BOOTHI = os.path.join(WORK, "boot_hi")
M5OV = os.path.join(WORK, "m5overlap")
TIME_BUDGET = 480.0            # seconds per invocation for the resumable sections below


def pop_def(name):
    """(setting, domain) of a named population (same definitions as P())."""
    if name == "R1_pooled":
        return "R1", None
    if name.startswith("R1_dom_"):
        return "R1", name[7:]
    if name.startswith("R3_"):
        return name, None
    if name.startswith("R1wd_s") and name.endswith("_pooled"):
        return name[:7], None
    if name.startswith("R1wd_s") and "_dom_" in name:
        return name[:7], name.split("_dom_")[1]
    if name == "R2_pooled":
        return "R2", None
    raise KeyError(name)


_IB_HI = {}


def boot_hi(name, det):
    """Stratified bootstrap of a named population with B = 10,000 and a fresh RNG stream
    (seed = crc32(name + '|B10000')); all detectors of the population share the resampled rows (paired).
    Returns {metric: (B_HI,) array}; cached per (population, detector)."""
    os.makedirs(BOOTHI, exist_ok=True)
    path = os.path.join(BOOTHI, f"{name}__{det.replace('#', '_')}.npz")
    if os.path.exists(path):
        z = np.load(path)
        return {k: z[k] for k in z.files}
    s, domain = pop_def(name)
    st = setting(s)
    if name not in _IB_HI:
        y = st["y"]
        idx = pop_idx(s, domain)
        pos, neg = idx[y[idx] == 1], idx[y[idx] == 0]
        rng = np.random.default_rng(seed_of(name + "|B10000"))
        _IB_HI[name] = (np.concatenate([pos[rng.integers(0, len(pos), (B_HI, len(pos)))],
                                        neg[rng.integers(0, len(neg), (B_HI, len(neg)))]], axis=1), len(pos))
    IB, npos = _IB_HI[name]
    S2, P2 = st["dets"][det]
    rr = np.random.default_rng([seed_of(name + "|B10000"), seed_of(det)]).integers(0, S2.shape[0], B_HI)
    acc = collections.defaultdict(list)
    for c0 in range(0, B_HI, 2000):
        I, r = IB[c0:c0 + 2000], rr[c0:c0 + 2000]
        m = metrics_rows(S2[r[:, None], I], {op: P2[op][r[:, None], I] for op in P2}, npos)
        for k, v in m.items():
            acc[k].append(v)
    out = {k: np.concatenate(v) for k, v in acc.items()}
    np.savez_compressed(path, **out)
    return out


def _in_window(p):
    return p is not None and MC_WINDOW[0] <= p <= MC_WINDOW[1]


def mc_families():
    """Every Holm family (or unadjusted comparison set) of M3, M4 and m5 with its members.
    member = (kind, popA, popB, detA, detB, metric); stored = the B = 2,000 dstat dict from the part file."""
    M3 = json.load(open(os.path.join(PARTS, "M3.json")))
    M4 = json.load(open(os.path.join(PARTS, "M4.json")))
    m5 = json.load(open(os.path.join(PARTS, "m5.json")))
    fams = []
    for a, b in PAIRS:
        k = pkey(a, b)
        for m in ("AUROC", "AUPRC_p10"):
            fams.append((f"M3.R1_domain_paired.{k}.{m}", True,
                         {d: ("paired", f"R1_dom_{d}", None, a, b, m) for d in DOMAINS}, M3["R1_domain_paired"][k][m]))
            fams.append((f"M3.R3_paired.{k}.{m}", True,
                         {d: ("paired", f"R3_{d}", None, a, b, m) for d in DOMAINS}, M3["R3_paired"][k][m]))
    for det in DET_M3:
        for m in ("AUROC", "AUPRC_p10", "recall@t_r80", "FPR@t_r80"):
            fams.append((f"M3.R3_minus_R1dm.{det}.{m}", True,
                         {d: ("indep", f"R3_{d}", f"R1_dom_{d}", det, det, m) for d in DOMAINS}, M3["R3_minus_R1dm"][det][m]))
            fams.append((f"M3.R3_minus_R1pooled.{det}.{m}", True,
                         {d: ("indep", f"R3_{d}", "R1_pooled", det, det, m) for d in DOMAINS}, M3["R3_minus_R1pooled"][det][m]))
    for rk in M4["rows"]:
        det, op = rk.split("@")
        for comp, ref in (("vs_domain_matched_R1", lambda d: f"R1_dom_{d}"), ("vs_pooled_R1", lambda d: "R1_pooled")):
            for mm in ("recall", "FPR"):
                fams.append((f"M4.deltas.{rk}.{comp}.{mm}", True,
                             {d: ("indep", f"R3_{d}", ref(d), det, det, f"{mm}@{op}") for d in DOMAINS},
                             M4["deltas"][rk][comp][mm]))
    for det in DET_M5:
        st_ = {str(k): dict(m5["seeds"][str(k)]["minus_R1"][det]["AUROC"]) for k in range(5)}
        for v in st_.values():
            v["p_holm"] = v.get("p_holm_over_seeds")
            v["detected_holm"] = v.get("detected_holm_over_seeds")
        fams.append((f"m5.minus_R1.{det}.AUROC", True,
                     {str(k): ("indep", f"R1wd_s{k}_pooled", "R1_pooled", det, det, "AUROC") for k in range(5)}, st_))
        fams.append((f"m5.minus_R1.{det}.AUPRC_p10", False,
                     {str(k): ("indep", f"R1wd_s{k}_pooled", "R1_pooled", det, det, "AUPRC_p10") for k in range(5)},
                     {str(k): m5["seeds"][str(k)]["minus_R1"][det]["AUPRC_p10"] for k in range(5)}))
        for k in range(5):
            fams.append((f"m5.domain_matched.{det}.s{k}", True,
                         {d: ("indep", f"R1wd_s{k}_dom_{d}", f"R1_dom_{d}", det, det, "AUROC") for d in DOMAINS},
                         {d: m5["seeds"][str(k)]["domain_matched"][det][d]["delta"] for d in DOMAINS}))
    return fams


def _hi_dstat(member, cache):
    kind, pa, pb, da, db, m = member
    A = P(pa)
    if kind == "paired":
        ha, hb = cache(pa, da), cache(pa, db)
        return dstat(ha[m] - hb[m], A["point"][da][m] - A["point"][db][m])
    Bp = P(pb)
    ha, hb = cache(pa, da), cache(pb, db)
    return dstat(ha[m] - hb[m], A["point"][da][m] - Bp["point"][db][m])


def sec_mccheck():
    """Monte Carlo stability of borderline decisions: every family with a member whose p or p_holm lies in
    [0.03, 0.07] at B = 2,000 is recomputed at B = 10,000 with fresh RNG streams (all members, Holm redone)."""
    t0 = time.time()
    fams = mc_families()
    flagged = [f for f in fams if any(_in_window(v.get("p")) or (f[1] and _in_window(v.get("p_holm"))) for v in f[3].values())]
    mem = {}

    def cache(pop, det):
        if (pop, det) not in mem:
            mem[(pop, det)] = boot_hi(pop, det)
        return mem[(pop, det)]

    # compute (resumably) every high-B array needed
    need = sorted({(x[1], x[3]) for f in flagged for x in f[2].values()} | {(x[2], x[4]) for f in flagged for x in f[2].values() if x[0] == "indep"}
                  | {(x[1], x[4]) for f in flagged for x in f[2].values() if x[0] == "paired"})
    for i, (pop, det) in enumerate(need):
        if not os.path.exists(os.path.join(BOOTHI, f"{pop}__{det.replace('#', '_')}.npz")):
            if time.time() - t0 > TIME_BUDGET:
                print(f"  mccheck: {i}/{len(need)} high-B arrays ready; time budget reached, run again", flush=True)
                return None
            boot_hi(pop, det)
            print(f"  boot_hi {pop} {det} ({time.time() - t0:.0f}s)", flush=True)
    out = {"definition": (f"families with any member p or p_holm in [{MC_WINDOW[0]}, {MC_WINDOW[1]}] at B = 2,000 are "
                          f"recomputed at B = {B_HI} with fresh RNG streams (crc32(population + '|B10000')); Holm is "
                          "redone over the whole family; point estimates are unchanged"),
           "n_families_total": len(fams), "n_families_flagged": len(flagged), "families": {}, "cells": [],
           "decision_changes": []}
    for fid, has_holm, members, stored in flagged:
        hi = {k: _hi_dstat(mb, cache) for k, mb in members.items()}
        if has_holm:
            holm_family(hi)
        clean(hi)
        out["families"][fid] = {"holm": has_holm, "B10000": hi}
        for k in members:
            s0, s1 = stored[k], hi[k]
            if not (_in_window(s0.get("p")) or (has_holm and _in_window(s0.get("p_holm")))):
                continue
            cell = {"family": fid, "member": k,
                    "B2000": {x: s0.get(x) for x in ("point", "lo", "hi", "p", "p_holm", "detected", "detected_holm")},
                    "B10000": {x: s1.get(x) for x in ("point", "lo", "hi", "p", "p_holm", "detected", "detected_holm")}}
            cell["unadjusted_decision_stable"] = bool(s0["detected"] == s1["detected"])
            cell["holm_decision_stable"] = (bool(s0.get("detected_holm") == s1.get("detected_holm")) if has_holm else None)
            cell["still_in_window_at_B10000"] = bool(_in_window(s1.get("p")) or (has_holm and _in_window(s1.get("p_holm"))))
            out["cells"].append(cell)
            if not cell["unadjusted_decision_stable"] or cell["holm_decision_stable"] is False:
                out["decision_changes"].append(f"{fid}|{k}")
    # m2: cluster bootstrap cells in the window
    m2 = json.load(open(os.path.join(PARTS, "m2.json")))
    out["m2"] = {}
    for det in DET_M2:
        for m in ("AUROC", "AUPRC_p10"):
            dd = m2["estimates"][det][m]["R2_minus_R1"]
            if not any(_in_window(dd[x]["p"]) for x in dd):
                continue
            r2c, _ = cluster_boot("R2", "task", [det], "m2_R2_taskcluster|B10000", B_=B_HI)
            r1w, _ = cluster_boot("R1", "world", [det], "m2_R1_worldcluster|B10000", B_=B_HI)
            r1t = cache("R1_pooled", det)[m]
            r2t = cache("R2_pooled", det)[m]
            p1, p2 = P("R1_pooled")["point"][det][m], P("R2_pooled")["point"][det][m]
            new = {"primary_R2taskcluster_minus_R1trajectory": dstat(r2c[det][m] - r1t, p2 - p1),
                   "sensitivity_R2taskcluster_minus_R1worldcluster": dstat(r2c[det][m] - r1w[det][m], p2 - p1),
                   "reference_v03_R2trajectory_minus_R1trajectory": dstat(r2t - r1t, p2 - p1)}
            for x in new.values():
                x.pop("p_raw")
            out["m2"][f"{det}.{m}"] = {"B2000": dd, "B10000": new,
                                       "decision_stable": {x: bool(dd[x]["detected"] == new[x]["detected"]) for x in dd}}
            for x in dd:
                if dd[x]["detected"] != new[x]["detected"]:
                    out["decision_changes"].append(f"m2.{det}.{m}.{x}")
    return out


def _fast_auc(sc, yv):
    """AUROC with midranks (ties 0.5), sc and yv 1-D."""
    npos = int(yv.sum())
    nneg = len(yv) - npos
    rk = rankdata(sc)
    return (rk[yv == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def _ap10(sc, yv):
    o = np.argsort(-yv, kind="stable")
    npos = int(yv.sum())
    return wap_rows(sc[o][None, :], npos, PI / (1 - PI) * (len(yv) - npos) / npos)[0]


def overlap_boot(sA, sB, domain, dets, B_, seed, with_ap=False):
    """Overlap-preserving bootstrap of a difference A - B between two test pools that share trajectories:
    resample the union of both pools (stratified by class, union sizes fixed); each pool is evaluated on the
    resampled rows that belong to it (with multiplicity), scored by its own model. Multi-seed detectors draw
    one training seed per replicate for each pool independently."""
    stA, stB = setting(sA), setting(sB)
    iA, iB = pop_idx(sA, domain), pop_idx(sB, domain)
    mA = {stA["ids"][i]: i for i in iA}
    mB = {stB["ids"][i]: i for i in iB}
    U = sorted(set(mA) | set(mB))
    posA = np.array([mA.get(u, -1) for u in U])
    posB = np.array([mB.get(u, -1) for u in U])
    yU = np.array([stA["y"][mA[u]] if u in mA else stB["y"][mB[u]] for u in U])
    both = (posA >= 0) & (posB >= 0)
    assert np.array_equal(stA["y"][posA[both]], stB["y"][posB[both]]), "labels differ on shared rows"
    up, un = np.where(yU == 1)[0], np.where(yU == 0)[0]
    rng = np.random.default_rng(seed)
    rA = {d: np.random.default_rng([seed, seed_of(d), 1]).integers(0, stA["dets"][d][0].shape[0], B_) for d in dets}
    rB = {d: np.random.default_rng([seed, seed_of(d), 2]).integers(0, stB["dets"][d][0].shape[0], B_) for d in dets}
    mets = ["AUROC"] + (["AUPRC_p10"] if with_ap else [])
    out = {d: {m: np.full(B_, np.nan) for m in mets} for d in dets}
    yA_all, yB_all = stA["y"], stB["y"]
    for b in range(B_):
        ib = np.r_[up[rng.integers(0, len(up), len(up))], un[rng.integers(0, len(un), len(un))]]
        a = posA[ib]
        a = a[a >= 0]
        c = posB[ib]
        c = c[c >= 0]
        ya, yc = yA_all[a], yB_all[c]
        if ya.min() == ya.max() or yc.min() == yc.max():
            continue
        for d in dets:
            sa = stA["dets"][d][0][rA[d][b]][a]
            sc = stB["dets"][d][0][rB[d][b]][c]
            out[d]["AUROC"][b] = _fast_auc(sa, ya) - _fast_auc(sc, yc)
            if with_ap:
                out[d]["AUPRC_p10"][b] = _ap10(sa, ya) - _ap10(sc, yc)
    info = {"n_union": len(U), "n_shared": int(both.sum()), "n_A": len(iA), "n_B": len(iB),
            "n_shared_pos": int(yU[both].sum()), "seed": int(seed)}
    return out, info


def _indep_auc_boot(s, domain, det, B_, seed):
    """Plain stratified trajectory bootstrap of AUROC for one population (vectorised), fresh RNG stream."""
    st = setting(s)
    y = st["y"]
    idx = pop_idx(s, domain)
    pos, neg = idx[y[idx] == 1], idx[y[idx] == 0]
    rng = np.random.default_rng(seed)
    S2 = st["dets"][det][0]
    rr = rng.integers(0, S2.shape[0], B_)
    res = []
    for c0 in range(0, B_, 2000):
        n = min(2000, B_ - c0)
        I = np.concatenate([pos[rng.integers(0, len(pos), (n, len(pos)))], neg[rng.integers(0, len(neg), (n, len(neg)))]], 1)
        Sb = S2[rr[c0:c0 + n][:, None], I]
        Rk = rankdata(Sb, axis=1)
        res.append((Rk[:, :len(pos)].sum(1) - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
    return np.concatenate(res)


def sec_m5overlap():
    """m5 sensitivity: overlap-aware bootstrap of R1-wd minus R1 (pooled pool E and per domain), plus RNG reruns
    for the borderline LR-B seed-1 comparison."""
    os.makedirs(M5OV, exist_ok=True)
    t0 = time.time()
    R1P = P("R1_pooled")
    for k in range(5):
        path = os.path.join(M5OV, f"s{k}.json")
        if os.path.exists(path):
            continue
        if time.time() - t0 > TIME_BUDGET:
            print("  m5overlap: time budget reached, run again", flush=True)
            return None
        s = f"R1wd_s{k}"
        WP = P(f"{s}_pooled")
        res = {"pooled": {}, "domain_matched": {}}
        ob, info = overlap_boot(s, "R1", None, DET_M5, B, seed_of(f"m5ov_{s}"), with_ap=True)
        res["pooled_info"] = info
        for det in DET_M5:
            res["pooled"][det] = {}
            for m in ("AUROC", "AUPRC_p10"):
                dd = dstat(ob[det][m], WP["point"][det][m] - R1P["point"][det][m])
                res["pooled"][det][m] = dd
        res["domain_info"] = {}
        for d in DOMAINS:
            WD, RD = P(f"{s}_dom_{d}"), P(f"R1_dom_{d}")
            obd, infod = overlap_boot(s, "R1", d, DET_M5, B, seed_of(f"m5ov_{s}_{d}"))
            res["domain_info"][d] = infod
            for det in DET_M5:
                res["domain_matched"].setdefault(det, {})[d] = dstat(obd[det]["AUROC"], WD["point"][det]["AUROC"] - RD["point"][det]["AUROC"])
        for det in DET_M5:
            clean(holm_family(res["domain_matched"][det]))
        jdump(res, path)
        print(f"  m5overlap s{k} done ({time.time() - t0:.0f}s)", flush=True)
    path = os.path.join(M5OV, "reruns.json")
    if not os.path.exists(path):
        if time.time() - t0 > TIME_BUDGET * 0.6:
            print("  m5overlap: reruns pending, run again", flush=True)
            return None
        rr = {"LR-B": {}}
        for k in range(5):
            s = f"R1wd_s{k}"
            pt = P(f"{s}_pooled")["point"]["LR-B"]["AUROC"] - R1P["point"]["LR-B"]["AUROC"]
            ind, ov = [], []
            for r in range(10):
                a = _indep_auc_boot(s, None, "LR-B", B, seed_of(f"m5rerun_ind_{s}_{r}"))
                b_ = _indep_auc_boot("R1", None, "LR-B", B, seed_of(f"m5rerun_ind_R1_{s}_{r}"))
                x = dstat(a - b_, pt)
                ind.append({kk: x[kk] for kk in ("lo", "hi", "p", "detected")})
                o, _ = overlap_boot(s, "R1", None, ["LR-B"], B, seed_of(f"m5rerun_ov_{s}_{r}"))
                x = dstat(o["LR-B"]["AUROC"], pt)
                ov.append({kk: x[kk] for kk in ("lo", "hi", "p", "detected")})
            o, _ = overlap_boot(s, "R1", None, ["LR-B"], B_HI, seed_of(f"m5ov_B10000_{s}"))
            hi = dstat(o["LR-B"]["AUROC"], pt)
            rr["LR-B"][str(k)] = {
                "independent_reruns_B2000": ind, "overlap_reruns_B2000": ov, "overlap_B10000": hi,
                "independent_detected_count": sum(v["detected"] for v in ind),
                "overlap_detected_count": sum(v["detected"] for v in ov),
                "independent_p_range": _rng([v["p"] for v in ind]), "overlap_p_range": _rng([v["p"] for v in ov]),
                "independent_hi_range": _rng([v["hi"] for v in ind]), "overlap_hi_range": _rng([v["hi"] for v in ov])}
            print(f"  m5overlap reruns s{k} ({time.time() - t0:.0f}s)", flush=True)
        fam = {str(k): dict(rr["LR-B"][str(k)]["overlap_B10000"]) for k in range(5)}
        holm_family(fam)
        for k in range(5):
            rr["LR-B"][str(k)]["overlap_B10000"]["p_holm_over_seeds"] = fam[str(k)]["p_holm"]
            rr["LR-B"][str(k)]["overlap_B10000"]["detected_holm_over_seeds"] = fam[str(k)]["detected_holm"]
            rr["LR-B"][str(k)]["overlap_B10000"].pop("p_raw", None)
        jdump(rr, path)
    # assemble
    out = {"definition": ("overlap-preserving bootstrap: the union of the R1 and R1-wd pool-E test sets (of the domain, "
                          "for the domain-matched version) is resampled with replacement, stratified by class; each pool is "
                          "evaluated on the resampled rows that belong to it, scored by its own model; B = 2,000; one "
                          "training seed per replicate and pool for the GRU; point estimates as in m5"),
           "seeds": {}, "reruns": json.load(open(path))}
    losses = {"pooled_AUROC_unadjusted": [], "pooled_AUROC_holm_over_seeds": [], "pooled_AUPRC_p10_unadjusted": [],
              "domain_matched_AUROC_unadjusted": [], "domain_matched_AUROC_holm_over_domains": []}
    per = {k: json.load(open(os.path.join(M5OV, f"s{k}.json"))) for k in range(5)}
    for det in DET_M5:
        fam = {str(k): dict(per[k]["pooled"][det]["AUROC"]) for k in range(5)}
        holm_family(fam)
        for k in range(5):
            e = per[k]["pooled"][det]["AUROC"]
            e["p_holm_over_seeds"] = fam[str(k)]["p_holm"]
            e["detected_holm_over_seeds"] = fam[str(k)]["detected_holm"]
            e.pop("p_raw", None)
            per[k]["pooled"][det]["AUPRC_p10"].pop("p_raw", None)
            if e["detected"] and e["point"] < 0:
                losses["pooled_AUROC_unadjusted"].append(f"{det}|s{k}")
            if e["detected_holm_over_seeds"] and e["point"] < 0:
                losses["pooled_AUROC_holm_over_seeds"].append(f"{det}|s{k}")
            a = per[k]["pooled"][det]["AUPRC_p10"]
            if a["detected"] and a["point"] < 0:
                losses["pooled_AUPRC_p10_unadjusted"].append(f"{det}|s{k}")
            for d in DOMAINS:
                x = per[k]["domain_matched"][det][d]
                if x["detected"] and x["point"] < 0:
                    losses["domain_matched_AUROC_unadjusted"].append(f"{det}|s{k}|{d}")
                if x["detected_holm"] and x["point"] < 0:
                    losses["domain_matched_AUROC_holm_over_domains"].append(f"{det}|s{k}|{d}")
    m5 = json.load(open(os.path.join(PARTS, "m5.json")))
    for k in range(5):
        per[k]["width_ratio_overlap_over_independent_AUROC"] = {
            det: r4((per[k]["pooled"][det]["AUROC"]["hi"] - per[k]["pooled"][det]["AUROC"]["lo"])
                    / (m5["seeds"][str(k)]["minus_R1"][det]["AUROC"]["hi"] - m5["seeds"][str(k)]["minus_R1"][det]["AUROC"]["lo"]))
            for det in DET_M5}
        out["seeds"][str(k)] = per[k]
    out["detected_losses"] = losses
    out["increases_pooled_AUROC_unadjusted"] = [f"{det}|s{k}" for det in DET_M5 for k in range(5)
                                                if per[k]["pooled"][det]["AUROC"]["detected"] and per[k]["pooled"][det]["AUROC"]["point"] > 0]
    return out


def _feat_cache(recs, cache):
    for r in recs:
        if r["id"] not in cache:
            cache[r["id"]] = vectorise(r, {})[0]
    return np.array([cache[r["id"]] for r in recs], float)


def c2st_quick(X, z, seed):
    """AUROC of pooled out-of-fold scores of the m12 C2ST (no bootstrap)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import StratifiedKFold
    oof = np.empty(len(z))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, z):
        mdl = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced"))
        mdl.fit(X[tr], z[tr])
        oof[te] = mdl.predict_proba(X[te])[:, 1]
    return float(ek.auroc(z, oof))


M12X = os.path.join(WORK, "m12extra")


def _unit(key, fn, t0):
    """Resumable unit of sec_m12extra: cached in domain_shift_work/m12extra/<key>.json."""
    os.makedirs(M12X, exist_ok=True)
    path = os.path.join(M12X, key + ".json")
    if os.path.exists(path):
        return json.load(open(path)), False
    if time.time() - t0 > TIME_BUDGET:
        return None, True
    t1 = time.time()
    v = fn()
    jdump(v, path)
    print(f"  m12extra {key} ({time.time() - t1:.0f}s)", flush=True)
    return v, False


def sec_m12extra():
    """C2ST for the settings m12 left out (R1wd_s1-s4; within-AgentDojo WUT, WUT_F, WIT, LOSO_*) and the spread
    of the C2ST AUROC over fold-assignment seeds 0-4 for every AgentDrift setting and R4. Resumable: each unit is
    cached in domain_shift_work/m12extra/ and the section returns None (run again) when the time budget is used."""
    t0 = time.time()
    mp = os.environ["AGENTDRIFT"]
    st_tr = load(os.path.join(mp, "data", "train.jsonl"))
    st_te = load(os.path.join(mp, "data", "test.jsonl"))
    td_tr = load(os.path.join(mp, "data_taskdisjoint", "train.jsonl"))
    td_te = load(os.path.join(mp, "data_taskdisjoint", "test.jsonl"))
    pool = st_tr + st_te
    p0, failed = _dojo(DOJO_P0), _dojo(DOJO_FAILED)
    dojo = p0 + failed
    fail_cats = {"failed_attack", "attack_failed"}
    fc = {}
    splits = {"R1": (st_tr, st_te), "R2": (td_tr, td_te)}
    for k in range(5):
        tr_i, te_i = world_disjoint_split(pool, k)
        splits[f"R1wd_s{k}"] = ([pool[i] for i in tr_i], [pool[i] for i in te_i])
    for d in DOMAINS:
        splits[f"R3_{d}"] = ([r for r in pool if r["agent"] != d], [r for r in pool if r["agent"] == d])
    splits["R4"] = (pool, dojo)
    out = {"definition": ("same C2ST as m12 (representation B, StandardScaler + LR(class_weight='balanced'), "
                          "StratifiedKFold(5, shuffle)). 'full' entries add the m12 bootstrap interval; 'fold_seed_spread' "
                          "repeats the pooled out-of-fold AUROC for fold-assignment seeds 0-4 (seed 0 = the m12 value); "
                          "cross-validated AgentDojo settings: one C2ST per (repetition, fold) train/test pair of the "
                          "shared cache (training rows exactly as the cache's fits: P0 rows of the other folds, plus the "
                          "failed attacks of the other folds for WUT_F), fold-assignment seed 0 for the C2ST itself"),
           "full": {}, "fold_seed_spread": {}, "within_agentdojo_cv": {}}
    pending = False

    def full_unit(s):
        tr, te = splits[s]
        assert [r["id"] for r in te] == list(setting(s)["ids"]), s
        Xtr, Xte = _feat_cache(tr, fc), _feat_cache(te, fc)
        mtr = np.array([r["category"] not in fail_cats for r in tr])
        mte = np.array([r["category"] not in fail_cats for r in te])
        return {"all_rows": c2st(Xtr, Xte), "poolE_rows": c2st(Xtr[mtr], Xte[mte]), "test_rows_equal_cache": True}

    for s in [f"R1wd_s{k}" for k in range(1, 5)]:
        v, stop = _unit(f"full_{s}", lambda s=s: full_unit(s), t0)
        pending |= stop
        if v is not None:
            out["full"][s] = v
    suites = sorted({r["agent"] for r in p0})

    def loso_unit(su):
        s = f"LOSO_{su}"
        tr = [r for r in p0 if r["agent"] != su]
        te = [r for r in dojo if r["agent"] == su]
        lin = pickle.load(open(os.path.join(CACHE, "linear", f"{s}.pkl"), "rb"))
        assert [r["id"] for r in te] == [m["id"] for m in lin["test_meta"]], s
        assert int(np.ravel(lin["info"]["n_train"])[0]) == len(tr), (s, lin["info"]["n_train"], len(tr))
        Xtr, Xte = _feat_cache(tr, fc), _feat_cache(te, fc)
        mte = np.array([r["category"] not in fail_cats for r in te])
        return {"all_rows": c2st(Xtr, Xte), "poolE_rows": c2st(Xtr, Xte[mte]), "test_rows_equal_cache": True,
                "n_train_equals_cache": True}

    for su in suites:
        v, stop = _unit(f"full_LOSO_{su}", lambda su=su: loso_unit(su), t0)
        pending |= stop
        if v is not None:
            out["full"][f"LOSO_{su}"] = v
    isP0 = np.r_[np.ones(len(p0), bool), np.zeros(len(failed), bool)]

    def cv_unit(s):
        Xd = _feat_cache(dojo, fc)
        lin = pickle.load(open(os.path.join(CACHE, "linear", f"{s}.pkl"), "rb"))
        assert [m["id"] for m in lin["test_meta"]] == [r["id"] for r in dojo], s
        folds = np.asarray(lin["folds"])
        use_failed = bool(lin["info"].get("failed_as_train_negatives"))
        if s == "WUT_F":
            assert use_failed, "WUT_F cache does not flag failed attacks as training negatives"
        vals, ntr, nte = [], [], []
        for r in range(folds.shape[0]):
            row = []
            for k in range(int(folds.max()) + 1):
                te = folds[r] == k
                tr = (folds[r] != k) & (isP0 | use_failed)
                assert int(tr.sum()) == int(np.asarray(lin["info"]["n_train"])[r][k]), (s, r, k, int(tr.sum()))
                z = np.r_[np.zeros(int(tr.sum()), int), np.ones(int(te.sum()), int)]
                row.append(r4(c2st_quick(np.vstack([Xd[tr], Xd[te]]), z, 0)))
                ntr.append(int(tr.sum()))
                nte.append(int(te.sum()))
            vals.append(row)
        flat = [v for rw in vals for v in rw]
        return {"AUROC_by_rep_fold": vals, "mean": r4(np.mean(flat)), "min": r4(min(flat)),
                "max": r4(max(flat)), "n_pairs": len(flat), "n_train_mean": r4(np.mean(ntr)),
                "n_test_mean": r4(np.mean(nte)), "n_train_equals_cache": True,
                "failed_attacks_in_training": use_failed}

    for s in ("WUT", "WUT_F", "WIT"):
        v, stop = _unit(f"cv_{s}", lambda s=s: cv_unit(s), t0)
        pending |= stop
        if v is not None:
            out["within_agentdojo_cv"][s] = v
    m12 = json.load(open(os.path.join(PARTS, "m12.json")))

    def spread_unit(s):
        tr, te = splits[s]
        Xtr, Xte = _feat_cache(tr, fc), _feat_cache(te, fc)
        X = np.vstack([Xtr, Xte])
        z = np.r_[np.zeros(len(Xtr), int), np.ones(len(Xte), int)]
        v = [r4(c2st_quick(X, z, sd)) for sd in range(5)]
        ref = m12["settings"][s]["all_rows"]["AUROC_oof"] if s in m12["settings"] else out["full"][s]["all_rows"]["AUROC_oof"]
        return {"AUROC_by_fold_seed": v, "mean": r4(np.mean(v)), "min": r4(min(v)), "max": r4(max(v)),
                "seed0_equals_reported": bool(abs(v[0] - ref) < 1e-4)}

    for s in splits:
        if s not in m12["settings"] and s not in out["full"]:
            pending = True
            continue
        v, stop = _unit(f"spread_{s}", lambda s=s: spread_unit(s), t0)
        pending |= stop
        if v is not None:
            out["fold_seed_spread"][s] = v
    if pending:
        print("  m12extra: time budget reached, run again", flush=True)
        return None
    return out


# ============================================================================ driver
SECTIONS = {"M3": sec_M3, "M3world": sec_M3world, "M4": sec_M4, "m5": sec_m5, "m2": sec_m2, "m12": sec_m12, "m13": sec_m13,
            "m3overlap": sec_m3, "m5overlap": sec_m5overlap, "mccheck": sec_mccheck, "m12extra": sec_m12extra}


def run(names):
    os.makedirs(BOOTDIR, exist_ok=True)
    os.makedirs(PARTS, exist_ok=True)
    for n in names:
        path = os.path.join(PARTS, f"{n}.json")
        if os.path.exists(path):
            print(f"section {n}: done (skip)")
            continue
        t0 = time.time()
        print(f"section {n}: running", flush=True)
        res = SECTIONS[n]()
        if res is None:                       # resumable section that ran out of its time budget
            print(f"section {n}: incomplete, run again", flush=True)
            continue
        res["_seconds"] = round(time.time() - t0, 1)
        jdump(res, path)
        print(f"section {n}: written ({res['_seconds']}s)", flush=True)


def status():
    for n in SECTIONS:
        print(n, "done" if os.path.exists(os.path.join(PARTS, f"{n}.json")) else "missing")


# ============================================================================ assembly and summary
def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rng(vals):
    vals = [v for v in vals if v is not None]
    return {"min": r4(min(vals)), "max": r4(max(vals))}


def _and(lst):
    lst = list(lst)
    if not lst:
        return "none"
    if len(lst) == 1:
        return lst[0]
    return ", ".join(lst[:-1]) + " and " + lst[-1]


def F(e, d=3):
    return f"{e['point']:.{d}f} [{e['lo']:.{d}f}, {e['hi']:.{d}f}]"


def Dd(e, d=3):
    return f"{e['point']:+.{d}f} [{e['lo']:+.{d}f}, {e['hi']:+.{d}f}]"


def Dh(e, d=3):
    s = Dd(e, d)
    if "p_holm" in e:
        ph = e["p_holm"]
        txt = f"{ph:.4f}" if 0.045 <= ph < 0.055 else f"{ph:.3f}"
        s += f" (pH {txt}{'*' if e['detected_holm'] else ''})"
    return s


def _req(cond, msg):
    """Guard for fixed wording in the summary: fail loudly instead of printing a statement the data no longer support."""
    if not cond:
        raise AssertionError("summary wording no longer supported by the results: " + msg)


def _nm(rows):
    return [r.replace("@t_r80", "") for r in rows]


def build_derived(R):
    M3, M4, m5, m2, m12, m13, m3, W = (R["M3"], R["M4"], R["m5"], R["m2"], R["m12"], R["m13"], R["m3"],
                                      R["M3_world_cluster_sensitivity"])
    dv = {"M3": {}, "M4": {}, "m5": {}, "m2": {}, "m12": {}, "m13": {}, "m3": {}}
    x = dv["M3"]
    for det in DET_M3:
        x[f"{det}|R3_minus_R1dm_AUROC|range"] = _rng([M3["R3_minus_R1dm"][det]["AUROC"][d]["point"] for d in DOMAINS])
        x[f"{det}|R3_minus_R1dm_AUROC|folds_decrease_holm"] = M3["degradation_folds"][det]["AUROC"]["vs_domain_matched_R1"]["holm"]
        x[f"{det}|R3_minus_R1dm_AUROC|folds_increase_holm"] = M3["degradation_folds"][det]["AUROC"]["vs_domain_matched_R1"]["increase_holm"]
        x[f"{det}|R3_minus_R1pooled_AUROC|folds_decrease_holm"] = M3["degradation_folds"][det]["AUROC"]["vs_pooled_R1"]["holm"]
        x[f"{det}|R3_minus_R1dm_AUPRC_p10|folds_decrease_holm"] = M3["degradation_folds"][det]["AUPRC_p10"]["vs_domain_matched_R1"]["holm"]
        x[f"{det}|R1dm_AUROC|range"] = _rng([M3["R1_domain_matched"][d][det]["AUROC"]["point"] for d in DOMAINS])
        x[f"{det}|R3_AUROC|range"] = _rng([M3["R3"][d][det]["AUROC"]["point"] for d in DOMAINS])
    for a, b in PAIRS:
        k = pkey(a, b)
        for m in ("AUROC", "AUPRC_p10"):
            x[f"{k}|R1dm_{m}|range"] = _rng([M3["R1_domain_paired"][k][m][d]["point"] for d in DOMAINS])
            x[f"{k}|R3_{m}|range"] = _rng([M3["R3_paired"][k][m][d]["point"] for d in DOMAINS])
            x[f"{k}|R1dm_{m}|folds_positive_holm"] = dom_list(M3["R1_domain_paired"][k][m], "detected_holm", +1)
            x[f"{k}|R1dm_{m}|folds_negative_holm"] = dom_list(M3["R1_domain_paired"][k][m], "detected_holm", -1)
            x[f"{k}|R3_{m}|folds_negative_holm"] = dom_list(M3["R3_paired"][k][m], "detected_holm", -1)
            x[f"{k}|R3_{m}|folds_positive_holm"] = dom_list(M3["R3_paired"][k][m], "detected_holm", +1)
            x[f"{k}|R3_{m}|folds_negative_unadjusted"] = dom_list(M3["R3_paired"][k][m], "detected", -1)
            x[f"{k}|R3_{m}|folds_positive_unadjusted"] = dom_list(M3["R3_paired"][k][m], "detected", +1)
    ns = M3["population_sizes"]["R1_domain_matched"]
    x["R1dm_n_range"] = _rng([ns[d]["n_pos"] + ns[d]["n_neg"] for d in DOMAINS])
    x["R1dm_npos_range"] = _rng([ns[d]["n_pos"] for d in DOMAINS])
    x["R1dm_nneg_range"] = _rng([ns[d]["n_neg"] for d in DOMAINS])
    x["LR-C-minus-LR-B|R1dm_AUROC|folds_without_advantage_holm"] = [
        d for d in DOMAINS if d not in x["LR-C-minus-LR-B|R1dm_AUROC|folds_positive_holm"]]
    x["world_cluster_reversal_folds_equal_primary"] = {
        f"{k}|{m}": W["reversal"][k][m]["folds_established"] == M3["reversal"][k][m]["folds_established"]
        for k in W["reversal"] for m in W["reversal"][k]}
    x["world_cluster_reversal_mismatches"] = [
        {"pair_metric": f"{k}|{m}", "primary": M3["reversal"][k][m]["folds_established"],
         "world_cluster": W["reversal"][k][m]["folds_established"]}
        for k in W["reversal"] for m in W["reversal"][k]
        if W["reversal"][k][m]["folds_established"] != M3["reversal"][k][m]["folds_established"]]
    x["world_cluster_degradation_folds_equal_primary"] = {
        det: W["degradation_folds_AUROC_holm"][det] == M3["degradation_folds"][det]["AUROC"]["vs_domain_matched_R1"]["holm"]
        for det in W["degradation_folds_AUROC_holm"]}
    # M4
    y = dv["M4"]
    th = M4["tables"]["vs_domain_matched_R1|holm"]
    tu = M4["tables"]["vs_domain_matched_R1|unadjusted"]
    tp = M4["tables"]["vs_pooled_R1|holm"]
    y["fails_all_five_vs_dm_holm"] = [r for r in th["fails_in_all_five"] if "#seed" not in r]
    y["holds_all_five_vs_dm_holm"] = [r for r in th["holds_in_all_five"] if "#seed" not in r]
    y["n_folds_failing_vs_dm_holm"] = {r: th["n_folds_failing"][r] for r in M4["rows"]}
    y["n_folds_failing_vs_dm_unadjusted"] = {r: tu["n_folds_failing"][r] for r in M4["rows"]}
    y["n_folds_failing_vs_pooled_holm"] = {r: tp["n_folds_failing"][r] for r in M4["rows"]}
    y["LR-B@t_r80|failing_folds_vs_dm_holm"] = [d for d in DOMAINS if th["cells"]["LR-B@t_r80"][d] != "holds"]
    y["LR-B@t_r80|failing_folds_vs_pooled_holm"] = [d for d in DOMAINS if tp["cells"]["LR-B@t_r80"][d] != "holds"]
    y["GRU_seeds|every_seed_fails_every_fold_vs_dm_holm"] = all(
        th["cells"][f"GRU#seed{r}@t_r80"][d] != "holds" for r in range(3) for d in DOMAINS)
    y["GRU@t_r80|holds_folds_vs_dm_holm"] = [d for d in DOMAINS if th["cells"]["GRU@t_r80"][d] == "holds"]
    y["LR-C@t_r80|R3_FPR|range"] = _rng([M4["values"]["LR-C@t_r80"]["R3"][d]["FPR"]["point"] for d in DOMAINS])
    y["LR-C@t_r80|R1dm_FPR|range"] = _rng([M4["values"]["LR-C@t_r80"]["R1_domain_matched"][d]["FPR"]["point"] for d in DOMAINS])
    y["LR-C@t_r80|R3_recall|range"] = _rng([M4["values"]["LR-C@t_r80"]["R3"][d]["recall"]["point"] for d in DOMAINS])
    y["GRU-drop0.2@t_r80|R3_recall|range"] = _rng([M4["values"]["GRU-drop0.2@t_r80"]["R3"][d]["recall"]["point"] for d in DOMAINS])
    y["TFIDF@t_r80|R3_recall|range"] = _rng([M4["values"]["TFIDF@t_r80"]["R3"][d]["recall"]["point"] for d in DOMAINS])
    dl_b = M4["deltas"]["LR-B@t_r80"]["vs_domain_matched_R1"]
    for d in ("medical", "email"):
        for mm in ("FPR", "recall"):
            y[f"LR-B@t_r80|{d}|{mm}_detected_unadjusted"] = dl_b[mm][d]["detected"]
            y[f"LR-B@t_r80|{d}|{mm}_detected_holm"] = dl_b[mm][d]["detected_holm"]
    y["over_alert_all_five_holm"] = [r for r in y["fails_all_five_vs_dm_holm"]
                                     if all("FPR+" in th["cells"][r][d] for d in DOMAINS)]
    y["under_alert_all_five_holm"] = [r for r in y["fails_all_five_vs_dm_holm"]
                                      if all("rec-" in th["cells"][r][d] and "FPR+" not in th["cells"][r][d] for d in DOMAINS)]
    y["GRU_banking_seed_recall"] = M3["GRU_per_seed_points"]["R3_banking"]["GRU"]["recall@t_r80"]
    y["TFIDF@t_r80|R1dm_recall|range"] = _rng([M4["values"]["TFIDF@t_r80"]["R1_domain_matched"][d]["recall"]["point"] for d in DOMAINS])
    # m5
    z = dv["m5"]
    z["n_pooled_comparisons"] = len(DET_M5) * 5
    z["n_domain_comparisons"] = len(DET_M5) * 5 * len(DOMAINS)
    z["email_frac_range"] = _rng([m5["seeds"][str(k)]["domain_mix"]["email"]["frac_of_poolE"] for k in range(5)])
    for det in DET_M5:
        z[f"{det}|minus_R1_AUROC|range"] = _rng([m5["seeds"][str(k)]["minus_R1"][det]["AUROC"]["point"] for k in range(5)])
        z[f"{det}|minus_R1_AUROC|lo_min"] = r4(min(m5["seeds"][str(k)]["minus_R1"][det]["AUROC"]["lo"] for k in range(5)))
        z[f"{det}|minus_R1_AUROC|n_negative_point"] = sum(m5["seeds"][str(k)]["minus_R1"][det]["AUROC"]["point"] < 0 for k in range(5))
    lo_all = [(m5["seeds"][str(k)]["minus_R1"][det]["AUROC"]["lo"], f"{det}|s{k}") for det in DET_M5 for k in range(5)]
    z["pooled_minus_R1_AUROC_lo_min"] = {"value": r4(min(lo_all)[0]), "where": min(lo_all)[1]}
    dls = []
    for item in m5["detected_losses"]["domain_matched_AUROC_unadjusted"]:
        det, sk, d = item.split("|")
        dls.append((m5["seeds"][sk[1:]]["domain_matched"][det][d]["delta"]["point"], item))
    z["domain_loss_largest"] = ({"value": r4(min(dls)[0]), "where": min(dls)[1]} if dls else None)
    tf = [abs(v) for v, it in dls if it.startswith("TFIDF|")]
    z["TFIDF_domain_loss_absmax"] = r4(max(tf)) if tf else None
    bl = [(m5["seeds"][str(k)]["domain_matched"][det]["banking"]["delta"]["lo"], f"{det}|s{k}")
          for det in DET_M5 if det != "TFIDF" for k in range(5)]
    z["banking_delta_lo_min_nonTFIDF"] = {"value": r4(min(bl)[0]), "where": min(bl)[1]}
    inc = collections.defaultdict(list)
    for item in m5["detected_losses"]["pooled_AUROC_increases_unadjusted"]:
        det, sk = item.split("|")
        inc[det].append(sk)
    z["pooled_AUROC_increases_by_detector"] = dict(inc)
    z["banking_n_range"] = _rng([m5["seeds"][str(k)]["domain_mix"]["banking"]["n"] for k in range(5)])
    # m12
    c = dv["m12"]
    c["R1_ci_includes_0.5"] = m12["settings"]["R1"]["all_rows"]["lo"] <= 0.5 <= m12["settings"]["R1"]["all_rows"]["hi"]
    c["R2_lo_above_0.5"] = m12["settings"]["R2"]["all_rows"]["lo"] > 0.5
    c["R1wd_s0_lo_above_0.5"] = m12["settings"]["R1wd_s0"]["all_rows"]["lo"] > 0.5
    c["R3_all_rows_range"] = _rng([m12["settings"][f"R3_{d}"]["all_rows"]["AUROC_oof"] for d in DOMAINS])
    c["R3_smallest_shift_fold"] = min(DOMAINS, key=lambda d: m12["settings"][f"R3_{d}"]["all_rows"]["AUROC_oof"])
    c["R3_largest_shift_fold"] = max(DOMAINS, key=lambda d: m12["settings"][f"R3_{d}"]["all_rows"]["AUROC_oof"])
    c["R3_largest_LRC_loss_vs_dm_fold"] = min(DOMAINS, key=lambda d: M3["R3_minus_R1dm"]["LR-C"]["AUROC"][d]["point"])
    c["R3_lowest_LRB_AUROC_fold"] = min(DOMAINS, key=lambda d: M3["R3"][d]["LR-B"]["AUROC"]["point"])
    c["R3_highest_LRB_AUROC_fold"] = max(DOMAINS, key=lambda d: M3["R3"][d]["LR-B"]["AUROC"]["point"])
    c["order_R1_lt_R2_R1wd_lt_R3_lt_R4"] = bool(
        m12["settings"]["R1"]["all_rows"]["AUROC_oof"] < min(m12["settings"]["R2"]["all_rows"]["AUROC_oof"], m12["settings"]["R1wd_s0"]["all_rows"]["AUROC_oof"])
        and max(m12["settings"]["R2"]["all_rows"]["AUROC_oof"], m12["settings"]["R1wd_s0"]["all_rows"]["AUROC_oof"])
        < min(m12["settings"][f"R3_{d}"]["all_rows"]["AUROC_oof"] for d in DOMAINS)
        and max(m12["settings"][f"R3_{d}"]["all_rows"]["AUROC_oof"] for d in DOMAINS) < m12["settings"]["R4"]["all_rows"]["AUROC_oof"])
    # rank agreement of shift size with LR-C loss over the five folds (descriptive only)
    sh = [m12["settings"][f"R3_{d}"]["all_rows"]["AUROC_oof"] for d in DOMAINS]
    ls = [-M3["R3_minus_R1dm"]["LR-C"]["AUROC"][d]["point"] for d in DOMAINS]
    from scipy.stats import spearmanr
    c["spearman_shift_vs_LRC_loss_5folds"] = r4(spearmanr(sh, ls).correlation)
    # m13
    q = dv["m13"]
    q["patstd_minus_raw"] = {det: {d: r4(m13["folds"][d]["by_detector"][det]["AUROC_pattern_standardised"]["point"]
                                        - m13["folds"][d]["by_detector"][det]["AUROC"]["point"]) for d in DOMAINS}
                             for det in ("LR-B", "LR-C", "GRU")}
    q["patstd_minus_raw_nonbanking_absmax"] = r4(max(abs(q["patstd_minus_raw"][det][d]) for det in ("LR-B", "LR-C", "GRU")
                                                   for d in DOMAINS if d != "banking"))
    q["patstd_CminusB_folds_negative_holm"] = [d for d in DOMAINS if m13["folds"][d]["paired_patstd"]["LR-C-minus-LR-B"]["detected_holm"]
                                               and m13["folds"][d]["paired_patstd"]["LR-C-minus-LR-B"]["point"] < 0]
    q["folds_full_hardest_for_LRB_and_LRC"] = [
        d for d in DOMAINS if all(
            min(PATTERNS, key=lambda p: m13["folds"][d]["by_detector"][det]["AUROC_by_pattern"][p]["point"]) == "attacked_full"
            for det in ("LR-B", "LR-C"))]
    q["R1dm_banking_full_fraction"] = r4(m13["folds"]["banking"]["R1_domain_matched_pattern_counts"]["attacked_full"]
                                         / sum(m13["folds"]["banking"]["R1_domain_matched_pattern_counts"].values()))
    # m3
    w = dv["m3"]
    ratios = [m3["overlap_sensitivity_R3_minus_R1dm_AUROC"][d][det]["width_ratio_joint_over_independent"]
              for d in DOMAINS for det in ("LR-B", "LR-C")]
    w["width_ratio_range"] = _rng(ratios)
    w["detection_changes"] = [f"{det}|{d}" for d in DOMAINS for det in ("LR-B", "LR-C")
                              if m3["overlap_sensitivity_R3_minus_R1dm_AUROC"][d][det]["joint_overlap_bootstrap"]["detected"]
                              != m3["overlap_sensitivity_R3_minus_R1dm_AUROC"][d][det]["independent_bootstrap"]["detected"]]
    w["shared_frac_range"] = _rng([m3["per_domain"][d]["shared_frac_of_R3_poolE"] for d in DOMAINS])
    # m5 overlap-aware sensitivity
    ov = R["m5_overlap_sensitivity"]
    z["overlap_detected_losses"] = ov["detected_losses"]
    z["overlap_increases_pooled_AUROC_unadjusted"] = ov["increases_pooled_AUROC_unadjusted"]
    z["overlap_width_ratio_range"] = _rng([ov["seeds"][str(k)]["width_ratio_overlap_over_independent_AUROC"][det]
                                           for k in range(5) for det in DET_M5])
    z["overlap_n_shared_range"] = _rng([ov["seeds"][str(k)]["pooled_info"]["n_shared"] for k in range(5)])
    rr = ov["reruns"]["LR-B"]
    z["LR-B_s1_independent_reruns_detected"] = rr["1"]["independent_detected_count"]
    z["LR-B_s1_overlap_reruns_detected"] = rr["1"]["overlap_detected_count"]
    z["LR-B_s1_independent_p_range"] = rr["1"]["independent_p_range"]
    z["LR-B_s1_overlap_p_range"] = rr["1"]["overlap_p_range"]
    z["LR-B|minus_R1_AUROC|n_negative_point_overlap"] = sum(
        ov["seeds"][str(k)]["pooled"]["LR-B"]["AUROC"]["point"] < 0 for k in range(5))
    z["nonLRB_min_point_minus_R1_AUROC"] = r4(min(m5["seeds"][str(k)]["minus_R1"][det]["AUROC"]["point"]
                                               for k in range(5) for det in DET_M5 if det != "LR-B"))
    # Monte Carlo stability
    mc = R["mc_stability_B10000"]
    dv["mc"] = {"n_families_total": mc["n_families_total"], "n_families_flagged": mc["n_families_flagged"],
                "n_cells_checked": len(mc["cells"]), "decision_changes": mc["decision_changes"],
                "n_decision_changes": len(mc["decision_changes"]),
                "changes_in_named_detectors": [c for c in mc["decision_changes"]
                                               if not any(t in c for t in ("GRU-drop0.1", "LR-B-minus", "LR-B-noobs"))]}
    # m12 extra
    mx = R["m12_extra"]
    dv["m12"]["R1wd_all_seeds_all_rows_range"] = _rng([m12["settings"]["R1wd_s0"]["all_rows"]["AUROC_oof"]]
                                                      + [mx["full"][f"R1wd_s{k}"]["all_rows"]["AUROC_oof"] for k in range(1, 5)])
    dv["m12"]["fold_seed_spread_max_range"] = r4(max(v["max"] - v["min"] for v in mx["fold_seed_spread"].values()))
    dv["m12"]["fold_seed_seed0_all_equal_reported"] = all(v["seed0_equals_reported"] for v in mx["fold_seed_spread"].values())
    dv["m12"]["LOSO_all_rows_range"] = _rng([mx["full"][f"LOSO_{su}"]["all_rows"]["AUROC_oof"]
                                             for su in ("banking", "slack", "travel", "workspace")])
    dv["m12"]["within_agentdojo_cv_mean_range"] = _rng([v["mean"] for v in mx["within_agentdojo_cv"].values()])
    return dv


def assemble():
    parts = {}
    for n in SECTIONS:
        p = os.path.join(PARTS, f"{n}.json")
        if not os.path.exists(p):
            raise SystemExit(f"section {n} missing; run `python domain_shift.py run {n}` first")
        parts[n] = json.load(open(p))
    used = sorted({f"{fam}/{s}.pkl" for s in ["R1", "R2", "R4", "WUT", "WUT_F", "WIT"] + [f"R1wd_s{k}" for k in range(5)]
                   + [f"R3_{d}" for d in DOMAINS] + [f"LOSO_{su}" for su in ("banking", "slack", "travel", "workspace")]
                   for fam in ("linear", "gru") if os.path.exists(os.path.join(CACHE, fam, f"{s}.pkl"))})
    R = {"meta": {
        "script": os.path.abspath(__file__), "script_sha256": _sha(os.path.abspath(__file__)),
        "assembled": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "plan": "REVISION_ANALYSIS_PLAN.md (sha256 recorded in revision/PLAN_MANIFEST.txt)",
        "review_comments": ["M3", "M4", "m2 (R2 part)", "m3 (R3 families and overlap)", "m5", "m12", "m13"],
        "caches_used_sha256": {u: _sha(os.path.join(CACHE, u)) for u in used},
        "B": B, "prevalence_for_AUPRC": PI, "alpha": ALPHA,
        "conventions": {
            "population": "pool E only (AgentDrift benign, hard_negative, attacked); unsuccessful attacks excluded",
            "bootstrap": ("trajectory-level stratified bootstrap (n_pos, n_neg fixed), B=2000, one RNG seed per population "
                          "(crc32 of its name, stored in domain_shift_work/boot/<name>.json); all detectors of a population "
                          "share the resampled rows (paired); different populations are independent"),
            "multi_seed": ("GRU, GRU-drop0.2, GRU-drop0.1 have 3 training seeds: each bootstrap replicate uses one seed drawn "
                           "at random (evalkit.bootstrap convention), so intervals include seed variation; the point "
                           "estimate is the mean over seeds (evalkit.point). 'GRU#seedk' rows use seed k only"),
            "AUPRC_p10": "prevalence-weighted average precision at pi=0.10, positive weight pi/(1-pi)*n_neg/n_pos (plan 3.1)",
            "AUPRC_native": "unweighted average precision at the population's own prevalence",
            "precision_pXX@op": "precision implied at prevalence XX% by recall and FPR at the operating point",
            "interval": "2.5 and 97.5 percentiles of the bootstrap distribution; point = full-sample estimate",
            "detected": "95% percentile interval excludes 0 (plan 3.3)",
            "p": "two-sided bootstrap p = 2*min((1+#{d<=0})/(B+1), (1+#{d>=0})/(B+1))",
            "p_holm": "Holm step-down over the stated family of five (folds, domains or seeds); detected_holm = p_holm < 0.05",
            "families": {
                "M3.R1_domain_paired.<pair>.<metric>": "five domains (R1 domain-matched), per pair and metric",
                "M3.R3_paired.<pair>.<metric>": "five R3 folds, per pair and metric (plan 3.6 a, b)",
                "M3.R3_minus_R1dm.<det>.<metric>": "five folds, per detector and metric (plan 3.6 c)",
                "M4.deltas.<det@op>.<comparator>.<recall|FPR>": "five folds, per detector, threshold, comparator and metric",
                "m5 domain-matched": "five domains per detector and split seed; pooled: five split seeds per detector"},
            "reversal_rule": ("plan 3.4: established in a fold iff the R1 domain-matched paired advantage is detected "
                              "(Holm over five domains, positive) AND the R3 paired difference is detected with the opposite "
                              "sign (Holm over five folds). Primary metric AUROC (the manuscript's reversal claim); "
                              "AUPRC_p10 reported as secondary")}},
        "M3": parts["M3"], "M3_world_cluster_sensitivity": parts["M3world"], "M4": parts["M4"], "m5": parts["m5"],
        "m2": parts["m2"], "m12": parts["m12"], "m13": parts["m13"], "m3": parts["m3overlap"],
        "m5_overlap_sensitivity": parts["m5overlap"], "m12_extra": parts["m12extra"],
        "mc_stability_B10000": parts["mccheck"]}
    # context for m12: pool-E AUROC of LR-B and LR-C per setting (R4 computed here from the linear cache, point only)
    ctx = {"R1": {det: R["M3"]["R1_pooled"][det]["AUROC"] for det in ("LR-B", "LR-C")},
           "R1wd_s0": {det: R["m5"]["seeds"]["0"]["metrics"][det]["AUROC"] for det in ("LR-B", "LR-C")},
           "R2": {det: R["m2"]["estimates"][det]["AUROC"]["R2_trajectory"] for det in ("LR-B", "LR-C")}}
    for d in DOMAINS:
        ctx[f"R3_{d}"] = {det: R["M3"]["R3"][d][det]["AUROC"] for det in ("LR-B", "LR-C")}
        ctx[f"R3_{d}"]["LR-C_R3_minus_R1dm"] = R["M3"]["R3_minus_R1dm"]["LR-C"]["AUROC"][d]
    st4 = setting("R4")
    i4 = pop_idx("R4")
    ctx["R4"] = {det: {"point": r4(ek.auroc(st4["y"][i4], st4["dets"][det][0][0][i4])), "lo": None, "hi": None,
                       "note": "transfer AUROC on AgentDojo P0 from the linear cache (point only)"} for det in ("LR-B", "LR-C")}
    R["m12_context"] = ctx
    arch = json.load(open(os.path.join(ROOT, "archive_v0.3", "r1r3_corrected.json")))
    R["references_v03"] = {
        "R2_minus_R1_LR-C_AUROC_trajectory_bootstrap": {
            k: r4(v) for k, v in arch["cross_rung_LOGREG_independent"]["R2_newtask-minus-R1"]["native"]["AUROC"].items()},
        "source": ("archive_v0.3/r1r3_corrected.json cross_rung_LOGREG_independent.R2_newtask-minus-R1.native.AUROC "
                   "(v0.3 manuscript Sec. VI-C: +0.023 [+0.001, +0.046]); delta = bootstrap mean")}
    R["derived"] = build_derived(R)
    jdump(R, OUT_JSON)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(write_summary(R))
    print("wrote", OUT_JSON, "and", OUT_MD)


def write_summary(R):
    M3, M4, m5, m2, m12, m13, m3, W, DV = (R["M3"], R["M4"], R["m5"], R["m2"], R["m12"], R["m13"], R["m3"],
                                          R["M3_world_cluster_sensitivity"], R["derived"])
    L = []
    a = L.append
    a("# Domain shift, thresholds and the world-disjoint control (review M3, M4, m2, m3, m5, m12, m13)\n")
    a(f"Generated by `revision/domain_shift.py` (SHA-256 `{R['meta']['script_sha256'][:16]}...`) on {R['meta']['assembled']}. "
      "Every number below is read from `revision/domain_shift_results.json`; the key path is given with each table "
      "(`<...>` = placeholder). Ranges and fold lists quoted in the readings are stored under `derived`. "
      "No detector was refitted: scores, predictions and thresholds come from the shared caches "
      "(`revision/cache/linear`, `revision/cache/gru`), whose thresholds were set on training data only. "
      "The only models fitted here are the classifier two-sample tests of m12.\n")
    a("**Conventions.** Populations are pool E (benign, hard negatives, attacked; unsuccessful attacks excluded). "
      "Intervals are 95% percentile intervals of a trajectory-level stratified bootstrap (B = 2,000), independent across "
      "populations and paired across detectors within a population. The GRU rows pool three training seeds (one seed drawn "
      "per replicate, point = mean over seeds), so their intervals include seed variation. AUPRC@10% is the prevalence-weighted "
      "average precision of plan 3.1 (pi = 0.10). \"Detected\" means the 95% interval excludes zero. pH is the Holm-adjusted "
      "p value within the family of five folds (or domains); an asterisk marks pH < 0.05. The intervals and pH values are "
      "marginal over the bootstrap; they do not include retraining variability beyond the three GRU seeds.\n")

    # ------------------------------------------------------------------ M3
    a("## M3. Domain-matched R1 baselines, R3 degradation and the reversal rule\n")
    a("**Question.** Each R3 AUROC ranks trajectories of one domain, but v0.3 compared it with the pooled five-domain R1 "
      "AUROC. What are the R1 values on the same domain, which R3 changes are detected against them, and in which folds "
      "does a reversal meet plan 3.4 (in-distribution advantage detected in the domain-matched R1 comparison and the "
      "opposite sign detected in R3, both after Holm over five folds)?\n")
    sz = M3["population_sizes"]
    a("Sizes (pool E, positives/negatives) [`M3.population_sizes`]: R1 pooled "
      f"{sz['R1_pooled']['n_pos']}/{sz['R1_pooled']['n_neg']}; R1 domain-matched "
      + ", ".join(f"{d} {sz['R1_domain_matched'][d]['n_pos']}/{sz['R1_domain_matched'][d]['n_neg']}" for d in DOMAINS)
      + "; R3 held-out pools " + ", ".join(f"{d} {sz['R3'][d]['n_pos']}/{sz['R3'][d]['n_neg']}" for d in DOMAINS) + ".\n")
    a("### M3-1. AUROC and AUPRC@10%: R1 domain-matched (R1dm), R3, and R3 minus R1dm (independent bootstraps)\n")
    a("Keys: `M3.R1_pooled.<det>.<metric>`, `M3.R1_domain_matched.<domain>.<det>.<metric>`, `M3.R3.<domain>.<det>.<metric>`, "
      "`M3.R3_minus_R1dm.<det>.<metric>.<domain>` (Holm family: five folds per detector and metric).\n")
    a("| Domain | Detector | R1dm AUROC | R3 AUROC | R3 − R1dm AUROC | R1dm AUPRC@10% | R3 AUPRC@10% | R3 − R1dm AUPRC@10% |")
    a("|---|---|---|---|---|---|---|---|")
    for det in DET_M3:
        e = M3["R1_pooled"][det]
        a(f"| R1 pooled | {det} | {F(e['AUROC'])} | | | {F(e['AUPRC_p10'])} | | |")
    for d in DOMAINS:
        for det in DET_M3:
            r1, r3 = M3["R1_domain_matched"][d][det], M3["R3"][d][det]
            dl = M3["R3_minus_R1dm"][det]
            a(f"| {d} | {det} | {F(r1['AUROC'])} | {F(r3['AUROC'])} | {Dh(dl['AUROC'][d])} | {F(r1['AUPRC_p10'])} | "
              f"{F(r3['AUPRC_p10'])} | {Dh(dl['AUPRC_p10'][d])} |")
    a("")
    a("### M3-2. Recall and FPR at theta_r80: R1dm, R3, and R3 minus R1dm\n")
    a("Keys: as M3-1 with metrics `recall@t_r80`, `FPR@t_r80`. Thresholds are the training-derived theta_r80 of each fit "
      "(R1 fit for R1dm, fold fit for R3).\n")
    a("| Domain | Detector | R1dm recall | R3 recall | R3 − R1dm recall | R1dm FPR | R3 FPR | R3 − R1dm FPR |")
    a("|---|---|---|---|---|---|---|---|")
    for det in DET_M3:
        e = M3["R1_pooled"][det]
        a(f"| R1 pooled | {det} | {F(e['recall@t_r80'])} | | | {F(e['FPR@t_r80'])} | | |")
    for d in DOMAINS:
        for det in DET_M3:
            r1, r3 = M3["R1_domain_matched"][d][det], M3["R3"][d][det]
            dl = M3["R3_minus_R1dm"][det]
            a(f"| {d} | {det} | {F(r1['recall@t_r80'])} | {F(r3['recall@t_r80'])} | {Dh(dl['recall@t_r80'][d])} | "
              f"{F(r1['FPR@t_r80'])} | {F(r3['FPR@t_r80'])} | {Dh(dl['FPR@t_r80'][d])} |")
    a("")
    a("### M3-3. Paired differences in R1 (pooled and per domain) and in R3\n")
    a("Keys: `M3.R1_pooled_paired.<pair>.<metric>`, `M3.R1_domain_paired.<pair>.<metric>.<domain>`, "
      "`M3.R3_paired.<pair>.<metric>.<domain>` (Holm: five domains in R1dm, five folds in R3).\n")
    a("| Pair | Metric | R1 pooled | Domain | R1dm (paired) | R3 (paired) |")
    a("|---|---|---|---|---|---|")
    for aa, bb in PAIRS:
        k = pkey(aa, bb)
        for m in ("AUROC", "AUPRC_p10"):
            for i, d in enumerate(DOMAINS):
                pooled = Dd(M3["R1_pooled_paired"][k][m]) if i == 0 else ""
                a(f"| {aa} − {bb} | {m} | {pooled} | {d} | {Dh(M3['R1_domain_paired'][k][m][d])} | {Dh(M3['R3_paired'][k][m][d])} |")
    a("")
    a("### M3-4. Degradation counts and the reversal rule\n")
    a("Folds with a detected AUROC decrease after Holm, against the domain-matched and the pooled R1 comparator "
      "[`M3.degradation_folds.<det>.<metric>`]:\n")
    a("| Detector | AUROC: vs R1dm | AUROC: vs pooled R1 (v0.3 comparator) | AUPRC@10%: vs R1dm | AUPRC@10%: vs pooled R1 |")
    a("|---|---|---|---|---|")
    for det in DET_M3:
        g = M3["degradation_folds"][det]
        a(f"| {det} | {_and(g['AUROC']['vs_domain_matched_R1']['holm'])} | {_and(g['AUROC']['vs_pooled_R1']['holm'])} | "
          f"{_and(g['AUPRC_p10']['vs_domain_matched_R1']['holm'])} | {_and(g['AUPRC_p10']['vs_pooled_R1']['holm'])} |")
    a("")
    a("Reversal rule (plan 3.4) [`M3.reversal.<pair>.<metric>`; world-cluster sensitivity "
      "`M3_world_cluster_sensitivity.reversal`]:\n")
    a("| Pair | Metric | R1dm advantage detected (Holm) | R3 opposite sign detected (Holm) | Reversal established (Holm) | Established, unadjusted | Established, world-cluster bootstrap |")
    a("|---|---|---|---|---|---|---|")
    for aa, bb in REVERSAL_PAIRS:
        k = pkey(aa, bb)
        for m in ("AUROC", "AUPRC_p10"):
            rv = M3["reversal"][k][m]
            a(f"| {aa} vs {bb} | {m} | {_and(rv['folds_R1dm_advantage_holm'])} | {_and(rv['folds_R3_opposite_holm'])} | "
              f"**{_and(rv['folds_established'])}** | {_and(rv['folds_established_unadjusted'])} | "
              f"{_and(W['reversal'][k][m]['folds_established'])} |")
    a("")
    for aa, bb in [("LR-C-drop", "LR-B"), ("GRU-drop0.2", "LR-C")]:
        k = pkey(aa, bb)
        sr = M3["reversal"][k]["AUROC"]["survives_remedy"]
        a(f"- Does the {pkey(*REMEDY_OF[(aa, bb)])} reversal survive the remedy ({aa})? Folds where the original reversal "
          f"is established and R3 {aa} − {bb} is still detected negative after Holm: **{_and(sr['folds'])}** "
          f"(original reversal folds: {_and(sr['folds_original_established'])}) [`M3.reversal.{k}.AUROC.survives_remedy`].")
    a("")
    x = DV["M3"]
    lc = M3["R3_minus_R1dm"]["LR-C"]["AUROC"]
    web_r1 = M3["R1_domain_matched"]["web"]["LR-C"]["AUROC"]
    _req(not x['LR-B|R3_minus_R1dm_AUROC|folds_decrease_holm'] and not x['LR-B|R3_minus_R1dm_AUROC|folds_increase_holm'], "LR-B no change")
    _req(len(x['GRU|R3_minus_R1dm_AUROC|folds_decrease_holm']) == 5 and len(x['GRU-drop0.2|R3_minus_R1dm_AUROC|folds_decrease_holm']) == 5, "GRU losses")
    _req(x['GRU-drop0.2|R3_minus_R1dm_AUROC|range']['min'] > x['GRU|R3_minus_R1dm_AUROC|range']['min']
         and x['GRU-drop0.2|R3_minus_R1dm_AUROC|range']['max'] > x['GRU|R3_minus_R1dm_AUROC|range']['max'], "dropout smaller loss")
    _req(len(x['TFIDF-minus-LR-C|R1dm_AUROC|folds_positive_holm']) == 5 and len(x['TFIDF-minus-LR-C|R3_AUROC|folds_positive_holm']) == 5, "TFIDF beats LR-C")
    _req(max(abs(x['LR-C-drop-minus-LR-B|R3_AUROC|range']['min']), abs(x['LR-C-drop-minus-LR-B|R3_AUROC|range']['max']))
         < min(abs(x['LR-C-minus-LR-B|R3_AUROC|range']['min']), abs(x['LR-C-minus-LR-B|R3_AUROC|range']['max'])), "dropout shrinks C-B deficit")
    _req(not x['GRU-drop0.2-minus-LR-C|R3_AUROC|folds_negative_holm'], "no GRU-drop0.2 deficit")
    wc_mis = x["world_cluster_reversal_mismatches"]
    wc_txt = ("A world-cluster bootstrap gives the same reversal folds for every pair and metric. " if not wc_mis else
              "A world-cluster bootstrap gives the same reversal folds for every pair and metric except "
              + _and([f"{mm['pair_metric'].replace('-minus-', ' vs ').replace('|', ' on ')} "
                      f"({_and(mm['primary'])} with trajectory resampling, {_and(mm['world_cluster'])} with world clusters)"
                      for mm in wc_mis]) + " [`derived.M3.world_cluster_reversal_mismatches`]. ")
    a("**Reading.** With the comparator matched by domain, the LR-C AUROC decrease is detected (after Holm) in "
      f"{_and(x['LR-C|R3_minus_R1dm_AUROC|folds_decrease_holm'])}. Against the pooled R1 value, which is the v0.3 comparator, "
      f"the detected folds are {_and(x['LR-C|R3_minus_R1pooled_AUROC|folds_decrease_holm'])}, which reproduces the "
      "manuscript's \"four of five folds, not web\". The web example is "
      f"reversed: web's own R1 AUROC is {F(web_r1)}, and R3 web is {Dd(lc['web'])} below it. LR-B shows no detected AUROC "
      f"change in any fold (R3 − R1dm between {x['LR-B|R3_minus_R1dm_AUROC|range']['min']:+.3f} and "
      f"{x['LR-B|R3_minus_R1dm_AUROC|range']['max']:+.3f}). At 10% prevalence, LR-C's AUPRC falls in "
      f"{_and(x['LR-C|R3_minus_R1dm_AUPRC_p10|folds_decrease_holm'])}. The GRU loses AUROC in every fold "
      f"({x['GRU|R3_minus_R1dm_AUROC|range']['min']:+.3f} to {x['GRU|R3_minus_R1dm_AUROC|range']['max']:+.3f}), and so does GRU-drop0.2, by less "
      f"({x['GRU-drop0.2|R3_minus_R1dm_AUROC|range']['min']:+.3f} to {x['GRU-drop0.2|R3_minus_R1dm_AUROC|range']['max']:+.3f}). "
      f"In-distribution, the LR-C over LR-B advantage is detected only in {_and(x['LR-C-minus-LR-B|R1dm_AUROC|folds_positive_holm'])}. "
      f"In R3, LR-C ranks below LR-B in {_and(x['LR-C-minus-LR-B|R3_AUROC|folds_negative_holm'])} "
      f"({x['LR-C-minus-LR-B|R3_AUROC|range']['min']:+.3f} to {x['LR-C-minus-LR-B|R3_AUROC|range']['max']:+.3f}). The AUROC reversal "
      f"is therefore established in {_and(M3['reversal']['LR-C-minus-LR-B']['AUROC']['folds_established'])} only; "
      f"{_and(x['LR-C-minus-LR-B|R1dm_AUROC|folds_without_advantage_holm'])} have no detected in-distribution advantage to reverse. For the GRU against LR-C, the in-distribution advantage is detected in "
      f"{_and(x['GRU-minus-LR-C|R1dm_AUROC|folds_positive_holm'])} (web {Dd(M3['R1_domain_paired']['GRU-minus-LR-C']['AUROC']['web'])}), "
      f"and the R3 deficit in {_and(x['GRU-minus-LR-C|R3_AUROC|folds_negative_holm'])}. The reversal is established in "
      f"{_and(M3['reversal']['GRU-minus-LR-C']['AUROC']['folds_established'])}. "
      f"Tool-block dropout shrinks the LR-C deficit against LR-B to between {x['LR-C-drop-minus-LR-B|R3_AUROC|range']['min']:+.3f} and "
      f"{x['LR-C-drop-minus-LR-B|R3_AUROC|range']['max']:+.3f}. Small as they are, those deficits are still detected in "
      f"{_and(x['LR-C-drop-minus-LR-B|R3_AUROC|folds_negative_holm'])}, so under the rule the reversal survives the remedy in "
      f"{_and(M3['reversal']['LR-C-drop-minus-LR-B']['AUROC']['survives_remedy']['folds'])}, at a much smaller size. "
      f"GRU-drop0.2 shows no detected R3 deficit against LR-C in any fold after Holm "
      f"({x['GRU-drop0.2-minus-LR-C|R3_AUROC|range']['min']:+.3f} to {x['GRU-drop0.2-minus-LR-C|R3_AUROC|range']['max']:+.3f}), "
      + ("so the GRU reversal does not survive dropout. " if not M3['reversal']['GRU-drop0.2-minus-LR-C']['AUROC']['survives_remedy']['folds']
         else f"and the GRU reversal survives dropout only in {_and(M3['reversal']['GRU-drop0.2-minus-LR-C']['AUROC']['survives_remedy']['folds'])}. ")
      + f"On AUPRC@10%, the C vs B reversal is established in "
      f"{_and(M3['reversal']['LR-C-minus-LR-B']['AUPRC_p10']['folds_established'])}, and the GRU vs C reversal in "
      f"{_and(M3['reversal']['GRU-minus-LR-C']['AUPRC_p10']['folds_established'])}. "
      f"TFIDF beats LR-C in every domain in R1 ({x['TFIDF-minus-LR-C|R1dm_AUROC|range']['min']:+.3f} to "
      f"{x['TFIDF-minus-LR-C|R1dm_AUROC|range']['max']:+.3f}) and in R3 ({x['TFIDF-minus-LR-C|R3_AUROC|range']['min']:+.3f} to "
      f"{x['TFIDF-minus-LR-C|R3_AUROC|range']['max']:+.3f}). Its own R3 loss is small ({x['TFIDF|R3_minus_R1dm_AUROC|range']['min']:+.3f} to "
      f"{x['TFIDF|R3_minus_R1dm_AUROC|range']['max']:+.3f}) but detected in "
      f"{_and(x['TFIDF|R3_minus_R1dm_AUROC|folds_decrease_holm'])}. " + wc_txt +
      "These numbers do not explain the reversal (M1 covers the mechanism). They also do not show that "
      f"{_and(x['LR-C-minus-LR-B|R1dm_AUROC|folds_without_advantage_holm'])} lack an in-distribution advantage: the "
      f"domain-matched R1 pools hold only {x['R1dm_n_range']['min']} to {x['R1dm_n_range']['max']} trajectories, and "
      "\"not detected\" there means the intervals are wide, not that the advantage is zero. The R1dm and R3 pools also share "
      "trajectories (see m3), which makes the independent-bootstrap intervals conservative.\n")

    # ------------------------------------------------------------------ M4
    a("## M4. Threshold failure per detector and fold\n")
    a("**Question.** Plan 3.3 defines threshold failure as a detected change in recall or FPR at theta_r80 relative to the "
      "domain-matched R1 value. Which detectors fail in which R3 folds, in which direction, and how does the picture change "
      "if the pooled R1 value is used as the comparator?\n")
    a(f"Legend: {M4['legend']}. Holm family: five folds per detector, threshold, comparator and metric. Keys: "
      "`M4.status.<det@op>.<comparator>.<holm|unadjusted>.<domain>`, deltas `M4.deltas.<det@op>.<comparator>.<recall|FPR>.<domain>`, "
      "values `M4.values.<det@op>`.\n")
    for comp, title in (("vs_domain_matched_R1", "relative to domain-matched R1 (plan 3.3)"),
                        ("vs_pooled_R1", "relative to pooled R1")):
        th, tu = M4["tables"][f"{comp}|holm"], M4["tables"][f"{comp}|unadjusted"]
        a(f"### M4-{1 if comp.startswith('vs_d') else 2}. Compact table, {title}\n")
        a("Cells: Holm-adjusted status. Where the unadjusted status differs, it is shown after a slash.\n")
        a("| Detector @ threshold | " + " | ".join(DOMAINS) + " | folds failing (Holm / unadjusted) |")
        a("|---|" + "---|" * (len(DOMAINS) + 1))
        for rk in M4["rows"]:
            if "#seed" in rk:
                continue
            cells = []
            for d in DOMAINS:
                h, u = th["cells"][rk][d], tu["cells"][rk][d]
                cells.append(h if h == u else f"{h} / {u}")
            a(f"| {rk} | " + " | ".join(cells) + f" | {th['n_folds_failing'][rk]} / {tu['n_folds_failing'][rk]} |")
        a("")
    th = M4["tables"]["vs_domain_matched_R1|holm"]
    a("### M4-3. GRU variants by training seed (relative to domain-matched R1, Holm)\n")
    a("| Detector | " + " | ".join(DOMAINS) + " |")
    a("|---|" + "---|" * len(DOMAINS))
    for rk in M4["rows"]:
        if "#seed" in rk:
            a(f"| {rk} | " + " | ".join(th["cells"][rk][d] for d in DOMAINS) + " |")
    a("")
    a("Per-seed point values for GRU and GRU-drop0.2 are in `M3.GRU_per_seed_points.<population>.<det>.<metric>`; for "
      "example, R3 banking GRU recall at theta_r80 by seed is "
      f"{', '.join(f'{v:.3f}' for v in M3['GRU_per_seed_points']['R3_banking']['GRU']['recall@t_r80'])} with FPR "
      f"{', '.join(f'{v:.3f}' for v in M3['GRU_per_seed_points']['R3_banking']['GRU']['FPR@t_r80'])}.\n")
    a("### M4-4. Recall and FPR at the threshold, all detectors (R1 pooled, R1dm, R3; deltas relative to R1dm and to pooled R1)\n")
    a("| Detector @ threshold | Domain | R1 pooled recall / FPR | R1dm recall | R3 recall | Δ recall vs R1dm | Δ recall vs pooled R1 | R1dm FPR | R3 FPR | Δ FPR vs R1dm | Δ FPR vs pooled R1 |")
    a("|---|---|---|---|---|---|---|---|---|---|---|")
    for rk in M4["rows"]:
        if "#seed" in rk:
            continue
        v, dl, dp = M4["values"][rk], M4["deltas"][rk]["vs_domain_matched_R1"], M4["deltas"][rk]["vs_pooled_R1"]
        for i, d in enumerate(DOMAINS):
            pooled = f"{F(v['R1_pooled']['recall'])} / {F(v['R1_pooled']['FPR'])}" if i == 0 else ""
            a(f"| {rk} | {d} | {pooled} | {F(v['R1_domain_matched'][d]['recall'])} | {F(v['R3'][d]['recall'])} | "
              f"{Dh(dl['recall'][d])} | {Dh(dp['recall'][d])} | {F(v['R1_domain_matched'][d]['FPR'])} | {F(v['R3'][d]['FPR'])} | "
              f"{Dh(dl['FPR'][d])} | {Dh(dp['FPR'][d])} |")
    a("")
    y = DV["M4"]
    vb = M4["values"]["LR-B@t_r80"]
    dbp = M4["deltas"]["LR-B@t_r80"]
    th_ = M4["tables"]["vs_domain_matched_R1|holm"]["cells"]
    _req(all(th_[r][d] == "FPR+,rec+" for r in y['over_alert_all_five_holm'] for d in DOMAINS), "over-alert = FPR+ and rec+")
    _req(max(M4["values"][r]["R3"][d]["recall"]["point"] for r in ("LR-A@t_r80", "IForest@t_r80") for d in DOMAINS) < 0.02,
         "LR-A and IForest flag almost nothing")
    _req(y['LR-B@t_r80|failing_folds_vs_dm_holm'] == ["medical"], "LR-B fails only in medical")
    _req(y['GRU_seeds|every_seed_fails_every_fold_vs_dm_holm'], "every GRU seed fails in every fold")
    _req(y['GRU_banking_seed_recall'][1] < 0.05 and min(y['GRU_banking_seed_recall'][0], y['GRU_banking_seed_recall'][2]) > 0.5,
         "banking GRU seeds disagree")
    _req(all(r in y['fails_all_five_vs_dm_holm'] for r in ("LR-C@t_r80", "Markov@t_r80", "IForest@t_r80")), "review list fails in all five")
    a("**Reading.** Threshold failure depends on the detector. After Holm, relative to the domain-matched R1 value, the "
      f"training threshold fails in all five folds for {_and(_nm(y['fails_all_five_vs_dm_holm']))}. The direction differs. "
      f"{_and(_nm(y['over_alert_all_five_holm']))} over-alert: FPR and recall both rise (LR-C FPR {y['LR-C@t_r80|R1dm_FPR|range']['min']:.3f}–"
      f"{y['LR-C@t_r80|R1dm_FPR|range']['max']:.3f} in R1dm, against {y['LR-C@t_r80|R3_FPR|range']['min']:.3f}–"
      f"{y['LR-C@t_r80|R3_FPR|range']['max']:.3f} in R3). "
      f"{_and(_nm(y['under_alert_all_five_holm']))} under-alert: recall falls without a rise in FPR (LR-A and the Isolation "
      f"Forest flag almost nothing; GRU-drop0.2 R3 recall {y['GRU-drop0.2@t_r80|R3_recall|range']['min']:.3f}–"
      f"{y['GRU-drop0.2@t_r80|R3_recall|range']['max']:.3f}; TFIDF {y['TFIDF@t_r80|R3_recall|range']['min']:.3f}–"
      f"{y['TFIDF@t_r80|R3_recall|range']['max']:.3f}, against {y['TFIDF@t_r80|R1dm_recall|range']['min']:.3f}–"
      f"{y['TFIDF@t_r80|R1dm_recall|range']['max']:.3f} in R1dm). "
      f"LR-B's threshold fails only in {_and(y['LR-B@t_r80|failing_folds_vs_dm_holm'])}. There its recall is "
      f"{F(vb['R3']['medical']['recall'])} against {F(vb['R1_domain_matched']['medical']['recall'])}, and its FPR "
      f"{F(vb['R3']['medical']['FPR'])} against {F(vb['R1_domain_matched']['medical']['FPR'])}"
      + ("; the FPR change is detected only before Holm. " if (y['LR-B@t_r80|medical|FPR_detected_unadjusted'] and not y['LR-B@t_r80|medical|FPR_detected_holm'])
         else f" (FPR change detected unadjusted: {y['LR-B@t_r80|medical|FPR_detected_unadjusted']}, after Holm: {y['LR-B@t_r80|medical|FPR_detected_holm']}). ")
      + ("In email, the review's \"FPR rises\" does not hold against email's own R1 value: "
         if not y['LR-B@t_r80|email|FPR_detected_holm'] else "In email, the FPR rise is detected against email's own R1 value: ")
      + f"{F(vb['R3']['email']['FPR'])} against {F(vb['R1_domain_matched']['email']['FPR'])}, Δ {Dh(dbp['vs_domain_matched_R1']['FPR']['email'])}. "
      f"Against the pooled R1 value, LR-B would appear to fail in {_and(y['LR-B@t_r80|failing_folds_vs_pooled_holm'])}. "
      f"The pooled-seed GRU fails in {5 - len(y['GRU@t_r80|holds_folds_vs_dm_holm'])} folds and 'holds' in "
      f"{_and(y['GRU@t_r80|holds_folds_vs_dm_holm'])}, but only because its three seeds disagree: every single seed fails in "
      "every fold (M4-3). In banking, seed 1 flags almost "
      f"nothing while seeds 0 and 2 flag most trajectories (recall by seed {', '.join(f'{v:.3f}' for v in y['GRU_banking_seed_recall'])}). "
      "The review's statement that the threshold fails in every fold for LR-C, the GRU, Markov and the Isolation Forest is "
      "therefore supported, for the GRU per seed, and LR-B's threshold carries over in "
      f"{5 - len(y['LR-B@t_r80|failing_folds_vs_dm_holm'])} of 5 folds. "
      "What the numbers do not show: the cause of LR-C's over-alerting, which M1 attributes to the tool-block offset "
      "w_A^T c and which is not measured here. Nor do they show that a 'holds' cell guarantees stability, since the "
      f"domain-matched R1 intervals come from {DV['M3']['R1dm_npos_range']['min']} to {DV['M3']['R1dm_npos_range']['max']} positives "
      f"and {DV['M3']['R1dm_nneg_range']['max']} negatives per domain.\n")

    # ------------------------------------------------------------------ m5
    a("## m5. World-disjoint control on pool E\n")
    a("**Question.** Does holding out whole worlds (R1-wd, split seeds 0–4) cost anything relative to R1, on pool E, "
      "overall and within domain, given that the split changes the domain mix?\n")
    a("### m5-1. Pool-E AUROC and differences from R1 (independent bootstraps)\n")
    a("Keys: `m5.seeds.<k>.metrics.<det>.<metric>`, `m5.seeds.<k>.minus_R1.<det>.<metric>` (Holm over seeds: "
      "`p_holm_over_seeds`), `m5.seeds.<k>.macro_AUROC.<det>` (mean of per-domain AUROCs, removes the domain-mix effect), "
      "R1 in `m5.R1` and `m5.R1_macro_AUROC`.\n")
    a("| Seed | Detector | R1-wd AUROC | R1-wd − R1 AUROC | R1-wd − R1 AUPRC@10% | macro AUROC R1-wd − R1 |")
    a("|---|---|---|---|---|---|")
    for det in DET_M5:
        a(f"| R1 | {det} | {F(m5['R1'][det]['AUROC'])} | | AUPRC@10% {F(m5['R1'][det]['AUPRC_p10'])} | macro {F(m5['R1_macro_AUROC'][det])} |")
    for k in range(5):
        s = m5["seeds"][str(k)]
        for det in DET_M5:
            e = s["minus_R1"][det]["AUROC"]
            a(f"| s{k} | {det} | {F(s['metrics'][det]['AUROC'])} | {Dd(e)} (pH {e['p_holm_over_seeds']:.3f}) | "
              f"{Dd(s['minus_R1'][det]['AUPRC_p10'])} | {Dd(s['macro_AUROC'][det]['delta_vs_R1_macro'])} |")
    a("")
    a("Across split seeds (point AUROC, mean [min, max]) [`m5.across_seeds_AUROC`]: "
      + "; ".join(f"{det} {m5['across_seeds_AUROC'][det]['mean']:.3f} [{m5['across_seeds_AUROC'][det]['min']:.3f}, "
                  f"{m5['across_seeds_AUROC'][det]['max']:.3f}]" for det in DET_M5) + ".\n")
    a("### m5-2. Domain mix of pool E (fraction of trajectories) [`m5.seeds.<k>.domain_mix`, `m5.R1_domain_mix`]\n")
    a("| Split | n pos/neg | worlds | " + " | ".join(DOMAINS) + " |")
    a("|---|---|---|" + "---|" * len(DOMAINS))
    a(f"| R1 | {M3['population_sizes']['R1_pooled']['n_pos']}/{M3['population_sizes']['R1_pooled']['n_neg']} | "
      f"{m2['R1_n_distinct_worlds_poolE']} | " + " | ".join(f"{m5['R1_domain_mix'][d]['frac_of_poolE']:.3f} (n={m5['R1_domain_mix'][d]['n']})" for d in DOMAINS) + " |")
    for k in range(5):
        s = m5["seeds"][str(k)]
        a(f"| s{k} | {s['n_pos']}/{s['n_neg']} | {s['n_worlds_poolE']} | "
          + " | ".join(f"{s['domain_mix'][d]['frac_of_poolE']:.3f} (n={s['domain_mix'][d]['n']})" for d in DOMAINS) + " |")
    a("")
    a("### m5-3. Domain-matched: per-domain AUROC, R1-wd minus R1 (independent bootstraps)\n")
    a("Keys: `m5.seeds.<k>.domain_matched.<det>.<domain>.{R1wd,R1,delta}` (Holm over five domains per detector and seed).\n")
    for det in DET_M5:
        a(f"**{det}** (R1 per domain: " + ", ".join(f"{d} {F(m5['seeds']['0']['domain_matched'][det][d]['R1'])}" for d in DOMAINS) + ")\n")
        a("| Seed | " + " | ".join(DOMAINS) + " |")
        a("|---|" + "---|" * len(DOMAINS))
        for k in range(5):
            dm = m5["seeds"][str(k)]["domain_matched"][det]
            a(f"| s{k} | " + " | ".join(f"{Dd(dm[d]['delta'])}{'*' if dm[d]['delta']['detected_holm'] else ''}" for d in DOMAINS) + " |")
        a("")
    dl = m5["detected_losses"]
    z = DV["m5"]
    _req(len(dl['pooled_AUROC_unadjusted']) == 0 and not dl['domain_matched_AUROC_holm_over_domains'], "no detected loss")
    _req(all(m5["seeds"][str(k)]["minus_R1"]["LR-B"]["AUROC"]["hi"] > 0 for k in range(5)), "LR-B intervals include zero")
    _req(z['email_frac_range']['min'] > 0.5, "email dominates")
    inc_txt = "; ".join(f"{det} in {_and(v)}" for det, v in z['pooled_AUROC_increases_by_detector'].items())
    a("**Reading.** No detector shows a detected loss in pool-E AUROC in any split seed: "
      f"{len(dl['pooled_AUROC_unadjusted'])} of {z['n_pooled_comparisons']} detector-by-seed comparisons, with no multiplicity "
      "correction, which is the stricter test for a claim of no loss. Detected losses in AUPRC@10%: "
      f"{len(dl['pooled_AUPRC_p10_losses_unadjusted'])} of {z['n_pooled_comparisons']}; in the domain-mix-free macro AUROC: "
      f"{len(dl['macro_AUROC_losses_unadjusted'])} of {z['n_pooled_comparisons']}. "
      f"LR-B is below R1 in {z['LR-B|minus_R1_AUROC|n_negative_point']} of 5 seeds "
      f"({z['LR-B|minus_R1_AUROC|range']['min']:+.3f} to {z['LR-B|minus_R1_AUROC|range']['max']:+.3f}), but every interval includes zero "
      f"(smallest lower bound {z['LR-B|minus_R1_AUROC|lo_min']:+.3f}). Detected increases (unadjusted): {inc_txt}. "
      f"The world-disjoint test pools are dominated by email ({z['email_frac_range']['min']:.3f}–{z['email_frac_range']['max']:.3f} "
      f"of pool E, against {m5['R1_domain_mix']['email']['frac_of_poolE']:.3f} in R1). Within domains, "
      f"{len(dl['domain_matched_AUROC_unadjusted'])} of {z['n_domain_comparisons']} per-domain comparisons show a loss detected "
      f"without correction ({_and(dl['domain_matched_AUROC_unadjusted'])}), and none after Holm within the five domains. "
      + (f"The largest is {z['domain_loss_largest']['where']} ({z['domain_loss_largest']['value']:+.3f}); the TFIDF cases have "
         f"|delta| <= {z['TFIDF_domain_loss_absmax']:.4f}. " if z['domain_loss_largest'] else "")
      + "So, under the plan's independent bootstraps, \"no detectable loss\" holds on pool E, in pooled and in domain-matched "
      "form (but see m5-4: an overlap-aware bootstrap detects the LR-B seed-1 loss before correction). It is a statement about "
      f"non-detection. The pooled lower bounds reach {z['pooled_minus_R1_AUROC_lo_min']['value']:+.3f} "
      f"({z['pooled_minus_R1_AUROC_lo_min']['where']}), and in the small R1-wd banking pools "
      f"({z['banking_n_range']['min']}–{z['banking_n_range']['max']} trajectories) they reach "
      f"{z['banking_delta_lo_min_nonTFIDF']['value']:+.3f} ({z['banking_delta_lo_min_nonTFIDF']['where']}), so losses of that size "
      "are not excluded.\n")

    ov = R["m5_overlap_sensitivity"]
    a("### m5-4. Sensitivity: overlap-aware bootstrap of R1-wd minus R1\n")
    a("The R1 and R1-wd pool-E test sets share trajectories "
      f"({z['overlap_n_shared_range']['min']}–{z['overlap_n_shared_range']['max']} per split seed), so the independent bootstraps "
      f"of m5-1 are conservative. Definition: {ov['definition']} [`m5_overlap_sensitivity.seeds.<k>.pooled.<det>.<metric>`, "
      "`...domain_matched.<det>.<domain>`, `...pooled_info`].\n")
    a("| Seed | shared rows (pos) | " + " | ".join(f"{det} R1-wd − R1 AUROC" for det in DET_M5) + " |")
    a("|---|---|" + "---|" * len(DET_M5))
    for k in range(5):
        so = ov["seeds"][str(k)]
        a(f"| s{k} | {so['pooled_info']['n_shared']} ({so['pooled_info']['n_shared_pos']}) | "
          + " | ".join(f"{Dd(so['pooled'][det]['AUROC'])}{'*' if so['pooled'][det]['AUROC']['detected_holm_over_seeds'] else ''}"
                       for det in DET_M5) + " |")
    a("")
    a("Asterisk: detected after Holm over the five split seeds. Width of the overlap-aware interval relative to the "
      f"independent one: {z['overlap_width_ratio_range']['min']:.2f}–{z['overlap_width_ratio_range']['max']:.2f} "
      "[`m5_overlap_sensitivity.seeds.<k>.width_ratio_overlap_over_independent_AUROC`].\n")
    r1b = ov["reruns"]["LR-B"]["1"]
    e_ind = m5["seeds"]["1"]["minus_R1"]["LR-B"]["AUROC"]
    e_ov = ov["seeds"]["1"]["pooled"]["LR-B"]["AUROC"]
    a("Monte Carlo reruns of the borderline LR-B seed-1 comparison (10 fresh RNG streams each, B = 2,000; plus one "
      "overlap-aware run at B = 10,000) [`m5_overlap_sensitivity.reruns.LR-B.1`]: independent bootstrap detected in "
      f"{z['LR-B_s1_independent_reruns_detected']} of 10 (p {z['LR-B_s1_independent_p_range']['min']:.3f}–"
      f"{z['LR-B_s1_independent_p_range']['max']:.3f}); overlap-aware detected in {z['LR-B_s1_overlap_reruns_detected']} of 10 "
      f"(p {z['LR-B_s1_overlap_p_range']['min']:.3f}–{z['LR-B_s1_overlap_p_range']['max']:.3f}); overlap-aware at B = 10,000 "
      f"{Dd(r1b['overlap_B10000'])}, p {r1b['overlap_B10000']['p']:.4f}, Holm over seeds {r1b['overlap_B10000']['p_holm_over_seeds']:.3f}.\n")
    ol = z["overlap_detected_losses"]
    _req(ol["pooled_AUROC_unadjusted"] == ["LR-B|s1"] and not ol["pooled_AUROC_holm_over_seeds"], "overlap LR-B s1 only")
    _req(not ol["domain_matched_AUROC_holm_over_domains"], "no overlap domain loss after Holm")
    _req(sorted(ol["domain_matched_AUROC_unadjusted"]) == sorted(dl["domain_matched_AUROC_unadjusted"]), "same domain cells")
    _req(z["LR-B|minus_R1_AUROC|n_negative_point"] == 5 and z["nonLRB_min_point_minus_R1_AUROC"] >= 0, "sign pattern")
    a("**Reading.** When the shared trajectories are kept shared, the intervals narrow and one pooled loss becomes "
      f"detected before correction: LR-B in split seed 1, {Dd(e_ov)}, against {Dd(e_ind)} with independent "
      "bootstraps. The independent version sits on the boundary (it is detected in "
      f"{z['LR-B_s1_independent_reruns_detected']} of 10 RNG reruns); the overlap-aware version is detected in "
      f"{z['LR-B_s1_overlap_reruns_detected']} of 10. After Holm over the five split seeds it is not detected "
      f"(pH {e_ov['p_holm_over_seeds']:.3f}). Its size matches the review panel's -0.027 [-0.054, -0.000] for this seed. "
      f"No other detector loses in any seed: detected pooled losses {len(ol['pooled_AUROC_unadjusted'])} of {z['n_pooled_comparisons']}, "
      f"AUPRC@10% losses {len(ol['pooled_AUPRC_p10_unadjusted'])}. Per domain, the overlap-aware version finds "
      f"{len(ol['domain_matched_AUROC_unadjusted'])} losses before correction ({_and(ol['domain_matched_AUROC_unadjusted'])}) and "
      "none after Holm, the same cells as the independent version. Detected increases (unadjusted): "
      f"{_and(z['overlap_increases_pooled_AUROC_unadjusted'])}. So the manuscript's \"no detectable loss\" should be worded "
      "per detector. It holds for LR-A, LR-C, the GRU and TFIDF: their pooled R1-wd − R1 point estimate is at least "
      f"{z['nonLRB_min_point_minus_R1_AUROC']:+.4f} in every seed. For LR-B it depends on the bootstrap. LR-B is below R1 "
      f"in all five seeds ({z['LR-B|minus_R1_AUROC|range']['min']:+.3f} to {z['LR-B|minus_R1_AUROC|range']['max']:+.3f}), and the "
      "seed-1 loss is detected when the overlap is kept, though not after correction over seeds.\n")

    # ------------------------------------------------------------------ m2
    a("## m2 (R2 part). Task-cluster bootstrap for R2 minus R1\n")
    a("**Question.** R2's test set has few distinct tasks. Does the R2 − R1 difference survive a bootstrap that resamples "
      "tasks, and does world clustering matter for R1?\n")
    a(f"R2 pool E has {m2['R2_n_distinct_test_tasks_poolE']} distinct test tasks "
      f"({', '.join(f'{d} {n}' for d, n in m2['R2_n_distinct_tasks_by_domain'].items())}), none of which occurs in R2 training "
      f"({m2.get('R2_test_tasks_also_in_R2_train')}); cluster sizes {m2['R2_taskcluster_info']['cluster_size_min']}–"
      f"{m2['R2_taskcluster_info']['cluster_size_max']} (median {m2['R2_taskcluster_info']['cluster_size_median']}), and every task has "
      f"both classes. R1 pool E has {m2['R1_n_distinct_worlds_poolE']} worlds (sizes {m2['R1_worldcluster_info']['cluster_size_min']}–"
      f"{m2['R1_worldcluster_info']['cluster_size_max']}) and {m2['R1_n_distinct_tasks_poolE']} tasks, all of which also occur in R1 "
      f"training ({m2.get('R1_test_tasks_also_in_R1_train')} of the 250 test tasks) [`m2.*`].\n")
    a("| Detector | Metric | R1 (trajectory) | R1 (world cluster) | R2 (trajectory) | R2 (task cluster) | R2 − R1, primary: task cluster − trajectory | sensitivity: task − world cluster | v0.3-style: trajectory − trajectory |")
    a("|---|---|---|---|---|---|---|---|---|")
    for det in DET_M2:
        for m in ("AUROC", "AUPRC_p10"):
            e = m2["estimates"][det][m]
            dd = e["R2_minus_R1"]
            a(f"| {det} | {m} | {F(e['R1_trajectory'])} | {F(e['R1_worldcluster'])} | {F(e['R2_trajectory'])} | {F(e['R2_taskcluster'])} | "
              f"{Dd(dd['primary_R2taskcluster_minus_R1trajectory'])} | {Dd(dd['sensitivity_R2taskcluster_minus_R1worldcluster'])} | "
              f"{Dd(dd['reference_v03_R2trajectory_minus_R1trajectory'])} |")
    a("")
    ec, eg = m2["estimates"]["LR-C"], m2["estimates"]["GRU"]
    _req(not ec['AUROC']['R2_minus_R1']['primary_R2taskcluster_minus_R1trajectory']['detected'], "LR-C R2-R1 not detected")
    _req(ec['AUROC']['R2_minus_R1']['reference_v03_R2trajectory_minus_R1trajectory']['detected'], "trajectory-style detected")
    _req(not eg['AUROC']['R2_minus_R1']['primary_R2taskcluster_minus_R1trajectory']['detected'], "GRU not detected")
    _req(ec['AUPRC_p10']['R2_minus_R1']['primary_R2taskcluster_minus_R1trajectory']['detected']
         and ec['AUPRC_p10']['R2_minus_R1']['primary_R2taskcluster_minus_R1trajectory']['lo'] < 0.01, "borderline AUPRC")
    wr = ((ec['AUROC']['R1_worldcluster']['hi'] - ec['AUROC']['R1_worldcluster']['lo'])
          / (ec['AUROC']['R1_trajectory']['hi'] - ec['AUROC']['R1_trajectory']['lo']))
    _req(0.9 < wr < 1.2, "world clustering barely changes R1")
    a("Keys: `m2.estimates.<det>.<metric>`. The interval of each difference combines two independent bootstraps.\n")
    a("**Reading.** When whole tasks are resampled, the R2 − R1 AUROC difference for LR-C is "
      f"{Dd(ec['AUROC']['R2_minus_R1']['primary_R2taskcluster_minus_R1trajectory'])}, which is not detected. The v0.3-style "
      f"trajectory bootstrap gives {Dd(ec['AUROC']['R2_minus_R1']['reference_v03_R2trajectory_minus_R1trajectory'])}, "
      f"which reproduces the v0.3 value {R['references_v03']['R2_minus_R1_LR-C_AUROC_trajectory_bootstrap']['delta']:+.3f} "
      f"[{R['references_v03']['R2_minus_R1_LR-C_AUROC_trajectory_bootstrap']['lo']:+.3f}, "
      f"{R['references_v03']['R2_minus_R1_LR-C_AUROC_trajectory_bootstrap']['hi']:+.3f}] (`references_v03`). The manuscript's "
      "detected improvement therefore comes from ignoring task clustering. "
      f"For the GRU the difference is {Dd(eg['AUROC']['R2_minus_R1']['primary_R2taskcluster_minus_R1trajectory'])}, "
      "also not detected. At 10% prevalence, LR-C's AUPRC difference is "
      f"{Dd(ec['AUPRC_p10']['R2_minus_R1']['primary_R2taskcluster_minus_R1trajectory'])} under the primary scheme, a borderline "
      f"positive, and {Dd(ec['AUPRC_p10']['R2_minus_R1']['sensitivity_R2taskcluster_minus_R1worldcluster'])} with R1 "
      "world-clustered. World clustering barely changes the R1 intervals (LR-C AUROC "
      f"{F(ec['AUROC']['R1_worldcluster'])} against {F(ec['AUROC']['R1_trajectory'])}). The numbers support \"no detected loss "
      "under task shift\" for both detectors. They do not support a detected gain. They also say nothing about new tasks whose "
      "tools are absent from training, because R2 keeps full tool coverage.\n")

    # ------------------------------------------------------------------ m12
    a("## m12. Classifier two-sample test (shift size) per setting\n")
    a(f"**Question.** Is the ordering of the ladder by shift size measured rather than asserted? Definition: {m12['definition']}.\n")
    a("Keys: `m12.settings.<setting>.<all_rows|poolE_rows>`; context columns from `m12_context.<setting>`.\n")
    a("| Setting | n train / n test | C2ST AUROC, all rows | fold SD | C2ST AUROC, pool-E rows | LR-B AUROC (pool E) | LR-C AUROC (pool E) | LR-C R3 − R1dm AUROC |")
    a("|---|---|---|---|---|---|---|---|")
    ctx = R["m12_context"]
    for s in ["R1", "R1wd_s0", "R2"] + [f"R3_{d}" for d in DOMAINS] + ["R4"]:
        e = m12["settings"][s]
        c = ctx[s]
        cb = F(c["LR-B"]) if c["LR-B"]["lo"] is not None else f"{c['LR-B']['point']:.3f} (point)"
        cc = F(c["LR-C"]) if c["LR-C"]["lo"] is not None else f"{c['LR-C']['point']:.3f} (point)"
        cd = Dd(c["LR-C_R3_minus_R1dm"]) if "LR-C_R3_minus_R1dm" in c else ""
        a(f"| {s} | {e['all_rows']['n_train']} / {e['all_rows']['n_test']} | {e['all_rows']['AUROC_oof']:.3f} [{e['all_rows']['lo']:.3f}, "
          f"{e['all_rows']['hi']:.3f}] | {e['all_rows']['AUROC_fold_sd']:.3f} | {e['poolE_rows']['AUROC_oof']:.3f} [{e['poolE_rows']['lo']:.3f}, "
          f"{e['poolE_rows']['hi']:.3f}] | {cb} | {cc} | {cd} |")
    a("")
    c = DV["m12"]
    a("Context columns: `m12_context.<setting>` (R4: transfer AUROC on AgentDojo P0, point only, from the linear cache).\n")
    _req(c["order_R1_lt_R2_R1wd_lt_R3_lt_R4"], "C2ST order")
    same = c['R3_smallest_shift_fold'] == c['R3_largest_LRC_loss_vs_dm_fold'] == c['R3_lowest_LRB_AUROC_fold']
    shift_txt = (f"{c['R3_smallest_shift_fold'].capitalize()} has the smallest behavioural shift, yet it has both the largest LR-C loss "
                 "against domain-matched R1 and the lowest LR-B R3 AUROC. " if same else
                 f"The smallest behavioural shift is in {c['R3_smallest_shift_fold']}, the largest LR-C loss against "
                 f"domain-matched R1 in {c['R3_largest_LRC_loss_vs_dm_fold']}, and the lowest LR-B R3 AUROC in "
                 f"{c['R3_lowest_LRB_AUROC_fold']}. ")
    shift_txt += (f"{c['R3_largest_shift_fold'].capitalize()} has the largest shift and also LR-B's highest R3 AUROC. "
                  if c['R3_largest_shift_fold'] == c['R3_highest_LRB_AUROC_fold'] else
                  f"The largest shift is in {c['R3_largest_shift_fold']}; LR-B's highest R3 AUROC is in {c['R3_highest_LRB_AUROC_fold']}. ")
    a("**Reading.** The measured shift follows the intended order: R1 "
      f"{m12['settings']['R1']['all_rows']['AUROC_oof']:.3f}"
      + (" (interval includes 0.5: train and test indistinguishable)" if c["R1_ci_includes_0.5"] else " (interval excludes 0.5)")
      + f", R2 {m12['settings']['R2']['all_rows']['AUROC_oof']:.3f} and R1-wd {m12['settings']['R1wd_s0']['all_rows']['AUROC_oof']:.3f}"
      + (" (small, but both intervals exclude 0.5)" if (c["R2_lo_above_0.5"] and c["R1wd_s0_lo_above_0.5"]) else "")
      + f", the R3 folds {c['R3_all_rows_range']['min']:.3f}–{c['R3_all_rows_range']['max']:.3f}, and R4 "
      f"{m12['settings']['R4']['all_rows']['AUROC_oof']:.3f}. The R3 folds are not homogeneous. " + shift_txt +
      f"Over the five folds, the Spearman correlation between shift and LR-C loss is "
      f"{c['spearman_shift_vs_LRC_loss_5folds']:+.2f}; with five points this is descriptive only. So shift size in "
      "representation B does not predict degradation. The test measures only the behavioural features. It does not capture "
      "the tool-vocabulary shift, which is total in every R3 fold by construction (kappa = 0), and it hardly separates R2 "
      "from R1-wd, although the two splits are built differently. The m12 values depend on the classifier; the review's "
      "values came from another method.\n")

    mx = R["m12_extra"]
    a("### m12-2. Further settings and stability over fold-assignment seeds\n")
    a("Keys: `m12_extra.full.<setting>` (with bootstrap interval), `m12_extra.within_agentdojo_cv.<setting>` (one C2ST per "
      "cross-validation training/test pair of the shared cache, 25 pairs), `m12_extra.fold_seed_spread.<setting>` "
      "(C2ST fold-assignment seeds 0–4).\n")
    a("| Setting | n train / n test | C2ST AUROC, all rows | C2ST AUROC, pool-E rows |")
    a("|---|---|---|---|")
    for s_, e in mx["full"].items():
        a(f"| {s_} | {e['all_rows']['n_train']} / {e['all_rows']['n_test']} | {e['all_rows']['AUROC_oof']:.3f} "
          f"[{e['all_rows']['lo']:.3f}, {e['all_rows']['hi']:.3f}] | {e['poolE_rows']['AUROC_oof']:.3f} "
          f"[{e['poolE_rows']['lo']:.3f}, {e['poolE_rows']['hi']:.3f}] |")
    for s_, e in mx["within_agentdojo_cv"].items():
        a(f"| {s_} (25 CV pairs) | {e['n_train_mean']:.0f} / {e['n_test_mean']:.0f} (means) | mean {e['mean']:.3f} "
          f"(range {e['min']:.3f}–{e['max']:.3f}) | |")
    a("")
    a("Spread over C2ST fold-assignment seeds 0–4 (all rows): "
      + "; ".join(f"{s_} {v['min']:.3f}–{v['max']:.3f}" for s_, v in mx["fold_seed_spread"].items())
      + f". The largest range is {DV['m12']['fold_seed_spread_max_range']:.4f}; seed 0 reproduces the reported value in "
      f"every setting: {DV['m12']['fold_seed_seed0_all_equal_reported']}.\n")
    _req(DV['m12']['fold_seed_spread_max_range'] < 0.015, "fold-seed spread small")
    a("**Reading.** The five world-disjoint splits give almost the same shift "
      f"({DV['m12']['R1wd_all_seeds_all_rows_range']['min']:.3f}–{DV['m12']['R1wd_all_seeds_all_rows_range']['max']:.3f}). "
      "The AgentDojo leave-one-suite-out splits shift about as much as the AgentDrift R3 folds, or more "
      f"({DV['m12']['LOSO_all_rows_range']['min']:.3f}–{DV['m12']['LOSO_all_rows_range']['max']:.3f}). The within-AgentDojo "
      f"cross-validation pairs show a moderate shift (means {DV['m12']['within_agentdojo_cv_mean_range']['min']:.3f}–"
      f"{DV['m12']['within_agentdojo_cv_mean_range']['max']:.3f}). Part of that shift is built in: their training folds hold "
      "P0 rows only (WUT_F also holds unsuccessful attacks), while their test folds hold both P0 rows and unsuccessful attacks. "
      f"The fold-assignment seed of the test changes the statistic by at most {DV['m12']['fold_seed_spread_max_range']:.3f}.\n")

    # ------------------------------------------------------------------ m13
    a("## m13. Hijack-pattern composition and prevalence of each R3 pool E\n")
    a("**Question.** Do the R3 held-out pools differ in hijack pattern and prevalence, and does that affect cross-fold "
      "comparisons, especially banking?\n")
    rm = m13["reference_mix"]
    a(f"Reference mix for standardisation: {rm['definition']}: full {rm['counts']['attacked_full']}, partial "
      f"{rm['counts']['attacked_partial']}, delayed {rm['counts']['attacked_delayed']} "
      f"({rm['fractions']['attacked_full']:.3f} / {rm['fractions']['attacked_partial']:.3f} / {rm['fractions']['attacked_delayed']:.3f}) "
      "[`m13.reference_mix`].\n")
    a("| Fold | pos / neg | prevalence | full / partial / delayed (fraction) | training positives full / partial / delayed | R1dm positives full / partial / delayed |")
    a("|---|---|---|---|---|---|")
    for d in DOMAINS:
        fo = m13["folds"][d]
        pc, pf, tf, rc = fo["pattern_counts"], fo["pattern_fractions"], fo["train_positive_pattern_fractions"], fo["R1_domain_matched_pattern_counts"]
        a(f"| {d} | {fo['n_pos']} / {fo['n_neg']} | {fo['prevalence']:.3f} | {pc['attacked_full']} / {pc['attacked_partial']} / "
          f"{pc['attacked_delayed']} ({pf['attacked_full']:.3f} / {pf['attacked_partial']:.3f} / {pf['attacked_delayed']:.3f}) | "
          f"{tf['attacked_full']:.3f} / {tf['attacked_partial']:.3f} / {tf['attacked_delayed']:.3f} | "
          f"{rc['attacked_full']} / {rc['attacked_partial']} / {rc['attacked_delayed']} |")
    a("")
    a("Pattern-specific and pattern-standardised AUROC in R3 (positives of one pattern against all pool-E negatives; the "
      "standardised value re-weights the three pattern AUROCs to the reference mix; the raw AUROC is exactly the "
      "positive-count-weighted mean of the pattern AUROCs) [`m13.folds.<fold>.by_detector.<det>`]:\n")
    a("| Fold | Detector | AUROC | pattern-standardised | full | partial | delayed | recall@theta_r80 full / partial / delayed |")
    a("|---|---|---|---|---|---|---|---|")
    for d in DOMAINS:
        for det in ("LR-B", "LR-C", "GRU"):
            b = m13["folds"][d]["by_detector"][det]
            rr = b["recall@t_r80_by_pattern"]
            a(f"| {d} | {det} | {F(b['AUROC'])} | {F(b['AUROC_pattern_standardised'])} | {F(b['AUROC_by_pattern']['attacked_full'])} | "
              f"{F(b['AUROC_by_pattern']['attacked_partial'])} | {F(b['AUROC_by_pattern']['attacked_delayed'])} | "
              f"{rr['attacked_full']['point']:.3f} / {rr['attacked_partial']['point']:.3f} / {rr['attacked_delayed']['point']:.3f} |")
    a("")
    a("Pattern-standardised paired LR-C − LR-B in R3 [`m13.folds.<fold>.paired_patstd`]: "
      + "; ".join(f"{d} {Dd(m13['folds'][d]['paired_patstd']['LR-C-minus-LR-B'])} (pH {m13['folds'][d]['paired_patstd']['LR-C-minus-LR-B']['p_holm']:.3f})"
                  for d in DOMAINS) + ".\n")
    bc = m13["banking_caveat"]
    q = DV["m13"]
    _req(len(bc['other_folds_prevalence']) == 1, "single prevalence elsewhere")
    _req(all(q['patstd_minus_raw'][det]['banking'] < 0 for det in ("LR-B", "LR-C", "GRU")), "standardisation lowers banking")
    a("**Reading and banking caveat.** Banking differs from the other held-out pools in two ways. Its prevalence is "
      f"{bc['banking_prevalence']:.3f} ({bc['banking_n_pos']} positives), against {bc['other_folds_prevalence'][0]:.3f} in every other fold. "
      f"Only {bc['banking_full_fraction']:.3f} of its positives are full hijacks, against {bc['other_folds_full_fraction_min']:.3f}–"
      f"{bc['other_folds_full_fraction_max']:.3f} elsewhere. The README attributes this to the validator cutting off banking "
      f"full-hijack generation. Full hijacks are the hardest of the three patterns for both LR-B and LR-C in "
      f"{_and(q['folds_full_hardest_for_LRB_and_LRC'])}. Re-weighting "
      "banking to the common pattern mix changes its AUROC by "
      f"{q['patstd_minus_raw']['LR-B']['banking']:+.3f} (LR-B), {q['patstd_minus_raw']['LR-C']['banking']:+.3f} (LR-C) and "
      f"{q['patstd_minus_raw']['GRU']['banking']:+.3f} (GRU). No other fold moves by more than {q['patstd_minus_raw_nonbanking_absmax']:.3f}. "
      "Banking's relatively high R3 AUROC is thus partly a composition effect. Cross-fold statements that rank banking "
      "against other folds, such as \"banking is the easiest held-out domain\", should carry this caveat. AUPRC@10% removes "
      "the prevalence difference but not the pattern mix. Within-fold conclusions do not depend on it. The LR-C < LR-B "
      f"ordering survives pattern standardisation in {_and(q['patstd_CminusB_folds_negative_holm'])} (Holm). The banking "
      f"domain-matched R1 subset has a similar mix ({q['R1dm_banking_full_fraction']:.3f} full), so the R3 − R1dm comparison "
      "for banking is roughly pattern-matched. The standardisation is descriptive: it assumes pattern-specific AUROCs would "
      "not change if the mix changed.\n")

    # ------------------------------------------------------------------ m3
    a("## m3. Test trajectories shared between the R1 test pool and each R3 held-out pool\n")
    a("**Question.** How many trajectories do the 'independent' R1 and R3 bootstraps share, and does the overlap matter?\n")
    a("| Domain | R1 test rows of the domain (all / pool E) | R3 held-out pool (all / pool E) | shared (all / pool E) | shared fraction of R3 pool E | R1 rows all contained in R3 |")
    a("|---|---|---|---|---|---|")
    for d in DOMAINS:
        p = m3["per_domain"][d]
        a(f"| {d} | {p['R1_test_rows_domain_all']} / {p['R1_test_rows_domain_poolE']} | {p['R3_heldout_rows_all']} / {p['R3_heldout_rows_poolE']} | "
          f"{p['shared_all_rows']} / {p['shared_poolE']} | {p['shared_frac_of_R3_poolE']:.4f} | {p['R1_domain_rows_all_contained_in_R3']} |")
    a(f"| total | | | {m3['total_shared_poolE']} (pool E) | {m3['total_shared_frac']:.4f} | |")
    a("")
    a("Sensitivity: a joint bootstrap resamples the R3 pool and evaluates the R1dm metric on the resampled rows that are R1 "
      "test rows, which keeps the overlap. It is compared with the independent bootstraps of M3 "
      "[`m3.overlap_sensitivity_R3_minus_R1dm_AUROC.<domain>.<det>`]:\n")
    a("| Domain | Detector | R3 − R1dm AUROC, joint | independent | width ratio joint / independent |")
    a("|---|---|---|---|---|")
    for d in DOMAINS:
        for det in ("LR-B", "LR-C"):
            s = m3["overlap_sensitivity_R3_minus_R1dm_AUROC"][d][det]
            a(f"| {d} | {det} | {Dd(s['joint_overlap_bootstrap'])} | {Dd(s['independent_bootstrap'])} | {s['width_ratio_joint_over_independent']:.3f} |")
    a("")
    w = DV["m3"]
    _req(w['width_ratio_range']['max'] < 1, "joint intervals narrower")
    _req(all(m3["per_domain"][d]["R1_domain_rows_all_contained_in_R3"] for d in DOMAINS), "R1 rows contained in R3")
    a("**Reading.** Every R3 held-out pool contains all R1 test trajectories of its domain: "
      f"{m3['per_domain']['medical']['shared_poolE']} pool-E trajectories per domain ({m3['per_domain']['banking']['shared_poolE']} for banking), "
      f"{w['shared_frac_range']['min']:.4f}–{w['shared_frac_range']['max']:.4f} of each R3 pool, and {m3['total_shared_poolE']} in total. "
      "Different models score the shared rows, but row difficulty is common to both, so the R1dm and R3 estimates are "
      "positively correlated and the independent-bootstrap intervals are conservative. Keeping the overlap narrows the "
      f"LR-B and LR-C intervals to {w['width_ratio_range']['min']:.2f}–{w['width_ratio_range']['max']:.2f} of their independent "
      f"width, and no detection conclusion changes ({len(w['detection_changes'])} changes). On multiplicity, the Holm families "
      "of plan 3.6 are applied throughout M3 and M4 (families of five folds or domains). Conjunctive claims are reported "
      f"with and without correction. For example, LR-C < LR-B in R3 holds in "
      f"{len(DV['M3']['LR-C-minus-LR-B|R3_AUROC|folds_negative_unadjusted'])} of 5 folds unadjusted and "
      f"{len(DV['M3']['LR-C-minus-LR-B|R3_AUROC|folds_negative_holm'])} of 5 after Holm, and GRU < LR-C in "
      f"{len(DV['M3']['GRU-minus-LR-C|R3_AUROC|folds_negative_unadjusted'])} of 5 unadjusted and "
      f"{len(DV['M3']['GRU-minus-LR-C|R3_AUROC|folds_negative_holm'])} of 5 after Holm.\n")

    # ------------------------------------------------------------------ Monte Carlo stability
    mc, q_ = R["mc_stability_B10000"], DV["mc"]
    a("## Monte Carlo stability of borderline decisions (B = 10,000)\n")
    a(f"Definition: {mc['definition']}. {q_['n_families_flagged']} of {q_['n_families_total']} families had a member in the "
      f"window, and {q_['n_cells_checked']} cells were checked [`mc_stability_B10000.cells`, `.families`]. The tables above "
      "report the pre-specified B = 2,000 values. Decisions that change at B = 10,000 "
      f"({q_['n_decision_changes']}) [`mc_stability_B10000.decision_changes`]:\n")
    a("| Family | Member | B = 2,000: delta [interval], p, pH | B = 10,000: delta [interval], p, pH |")
    a("|---|---|---|---|")
    fmt_ = lambda e: (f"{e['point']:+.4f} [{e['lo']:+.4f}, {e['hi']:+.4f}], p {e['p']:.4f}"
                      + (f", pH {e['p_holm']:.4f}" if e.get('p_holm') is not None else ""))
    for c in mc["cells"]:
        if c["unadjusted_decision_stable"] and c["holm_decision_stable"] is not False:
            continue
        a(f"| {c['family']} | {c['member']} | {fmt_(c['B2000'])} | {fmt_(c['B10000'])} |")
    for k, v in mc["m2"].items():
        for x, ok in v["decision_stable"].items():
            if not ok:
                a(f"| m2.{k}.{x} | | {fmt_(v['B2000'][x])} | {fmt_(v['B10000'][x])} |")
    a("")
    expected = {"M3.R3_paired.GRU-drop0.2-minus-LR-C.AUROC|banking", "M3.R3_minus_R1dm.LR-C-drop.AUROC|coding",
                "m5.domain_matched.TFIDF.s2|banking", "m5.domain_matched.TFIDF.s3|banking",
                "m5.domain_matched.TFIDF.s4|banking", "m2.LR-C.AUPRC_p10.sensitivity_R2taskcluster_minus_R1worldcluster"}
    _req(set(q_["changes_in_named_detectors"]) == expected, "MC change list as described")
    _req(not any(c.startswith("M3.R1_domain_paired") or ".LR-C@" in c or ".LR-B@" in c or ".GRU@" in c or "m5.minus_R1" in c
                 or "LR-C-minus-LR-B" in c or "GRU-minus-LR-C" in c or "LR-C-drop-minus-LR-B" in c
                 for c in q_["decision_changes"]), "no reversal or key threshold cell changes")
    a("**Reading.** Every change is a p value near 0.05 that crosses it at the larger B. None changes a reversal fold, a "
      "threshold-failure cell of LR-B, LR-C or the pooled GRU, or a pooled no-loss statement of m5. The remaining changes "
      "concern LR-B-minus, LR-B-noobs and GRU-drop0.1 cells. Among the named detectors, four things change. (1) LR-C-drop's "
      "R3 − R1dm AUROC loss in coding becomes detected after Holm, so LR-C-drop's Holm-detected degradation folds gain coding. "
      "(2) GRU-drop0.2's R3 advantage over LR-C in banking becomes detected after Holm. That is an advantage, not a deficit, "
      "so the reversal conclusions do not change. (3) Three TFIDF banking increases in m5, each about +0.001, lose "
      "unadjusted detection. (4) The m2 LR-C AUPRC@10% R2 − R1 difference with R1 world-clustered becomes detected. Statements "
      "that depend on these cells should be worded as borderline.\n")

    # ------------------------------------------------------------------ deviations
    a("## Deviations and analysis choices (also appended to revision/DEVIATIONS.md, section 'Domain shift analysis')\n")
    for item in DEVIATION_ITEMS:
        a(f"- {item}")
    a("")
    return "\n".join(L)


DEVIATION_ITEMS = [
    "DS1. GRU seeds. The plan does not say how the three GRU training seeds enter intervals. Following evalkit.bootstrap "
    "(the v0.3 convention used by the GRU cache summary), each bootstrap replicate uses one randomly drawn seed and the "
    "point estimate is the mean over seeds. Where the seeds disagree strongly (R3 banking and coding thresholds), this "
    "gives very wide intervals, so per-seed rows (GRU#seedk) are also reported for the threshold analysis.",
    "DS2. Holm p values. The plan asks for Holm correction but does not say how to get p values from a bootstrap. We use the "
    "two-sided percentile-bootstrap p = min(1, 2*min(1+#{d<=0}, 1+#{d>=0})/(B+1)). 'Detected' (unadjusted) keeps the plan 3.3 "
    "definition (95% percentile interval excludes 0). The two criteria can disagree only at the boundary.",
    "DS3. Reversal metric. Plan 3.4 does not name the metric. AUROC is primary because the manuscript's reversal claim is about "
    "AUROC; AUPRC@10% is reported as secondary. The rule is also applied to the dropout pairs (LR-C-drop vs LR-B, "
    "GRU-drop0.2 vs LR-C), together with the plan 5 question of whether the original reversal survives the remedy.",
    "DS4. m5 multiplicity. For the claim of no loss, detection without correction is the primary criterion because it is the "
    "stricter test for that claim; Holm over split seeds (pooled) and over domains (domain-matched) is reported as secondary. "
    "A macro (mean of per-domain) AUROC is added to remove the domain-mix change.",
    "DS5. m2. The task-cluster bootstrap resamples the 35 R2 test tasks with replacement (not stratified by class; every task "
    "contains both classes). The R1 world-cluster sensitivity resamples the 676 pool-E worlds. AUPRC@10% weights are "
    "recomputed per replicate from its own class counts.",
    "DS6. m12. Classifier two-sample test: StandardScaler + LogisticRegression(class_weight='balanced', max_iter=5000) on "
    "representation B, StratifiedKFold(5, shuffle, seed 0) on train-versus-test membership. The interval is a stratified "
    "bootstrap of the out-of-fold scores (it omits refitting variability). The primary variant uses all rows of the setting's "
    "train and test sets; a pool-E-rows variant (failed attacks excluded on both sides) is a sensitivity analysis. The test "
    "rows equal the cache rows in every setting (asserted).",
    "DS7. Additions not in the plan: world-cluster sensitivity for the M3 comparisons (plan 3.5 names world clustering as a "
    "sensitivity for R1/R3), pattern-standardised AUROC and per-pattern AUROC/recall (m13), a joint-overlap bootstrap (m3), "
    "and derived precision at 2%/10% prevalence from recall and FPR (M4 values).",
    "DS8. The plan names B = 2,000 for the AUPRC bootstrap; B = 2,000 is used for every metric here. Intervals therefore differ "
    "slightly from the cache summaries (B = 1,000 for AgentDrift in validation.json).",
    "DS9. m5 overlap (addition). The R1 and R1-wd pool-E test sets share 287-327 trajectories per split seed. The plan asks "
    "for independent bootstraps, which are reported as primary. An overlap-preserving bootstrap (resample the union of both "
    "pools, stratified by class, evaluate each pool on its own resampled rows) is added as a sensitivity analysis, together "
    "with 10 RNG reruns of the borderline LR-B seed-1 comparison. The overlap-aware version detects that loss before "
    "correction.",
    "DS10. Monte Carlo stability (addition). Every Holm family of M3, M4 and m5 (and the m2 comparisons) with a member whose "
    "p or pH lies in [0.03, 0.07] at B = 2,000 is recomputed at B = 10,000 with fresh RNG streams. The pre-specified "
    "B = 2,000 values stay primary; decisions that change are listed.",
    "DS11. m12 extra (addition). The C2ST is also reported for R1wd_s1-s4, LOSO_* and, per cross-validation pair, WUT, "
    "WUT_F and WIT, and over C2ST fold-assignment seeds 0-4. The task asks only for R1, R1wd_s0, R2, R3 and R4.",
]


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "run":
        run(sys.argv[2:] or list(SECTIONS))
    elif cmd == "assemble":
        assemble()
    elif cmd == "all":
        run(list(SECTIONS))
        assemble()
    else:
        status()
