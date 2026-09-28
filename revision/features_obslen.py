"""features_obslen.py -- review comments M5 (observation-length cue, padding, truncation) and m8 (keyword
state-changing flag). Follows REVISION_ANALYSIS_PLAN.md Sec. 4 ("Observation length", "Keyword feature").

Usage (environment variables AGENTDRIFT and AGENTDOJO must be set):
    python features_obslen.py status
    python features_obslen.py <part>         one part (resumable; parts already written are skipped by `all`)
    python features_obslen.py all            every missing part, in order, within a time budget
    python features_obslen.py assemble       writes features_obslen_results.json
    python features_obslen.py summary        writes features_obslen_summary.md from the results JSON
Parts: annotation coefs univariate noobs pad_r1_linear pad_r1_gru pad_wut truncation annot_lrb

Inputs are read only. Scores of the cached detectors come from revision/cache/{linear,gru}; new fits (coefficient
bootstraps, the WUT fold models used for padding, LR-obsonly, LR-B-annot and the three R1 GRU seeds for padding) reuse
the cache recipes exactly and are checked against the cache before use. Thresholds always come from training data.
"""
import os, sys, json, time, pickle, re, collections, hashlib, warnings, datetime
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
MP = os.environ["AGENTDRIFT"]
ADJ = os.environ["AGENTDOJO"]

from featurise import load, build_vocab, vectorise, tool_seq, AGNOSTIC_NAMES  # noqa: E402
import evalkit as ek  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

PARTS = os.path.join(HERE, "features_obslen_parts")
os.makedirs(PARTS, exist_ok=True)
CACHE = os.path.join(HERE, "cache")
OUT_JSON = os.path.join(HERE, "features_obslen_results.json")
OUT_MD = os.path.join(HERE, "features_obslen_summary.md")
ANNOT_JSON = os.path.join(HERE, "tool_state_annotation.json")

FEAT = list(AGNOSTIC_NAMES)                 # 0 len .. 9 mean_obs
DOMAINS = ["medical", "banking", "web", "email", "coding"]
PADS = [0, 100, 300, 600, 1200]
SENT = " The quick brown fox jumps over the lazy dog."   # neutral filler sentence, repeated and cut to n characters
B = 2000
KW = ("send", "delete", "update", "create", "transfer", "add", "download",
      "post", "write", "cancel", "apply", "convert", "share", "grant")   # featurise.vectorise keyword list
POOL_E = ("benign", "hard_negative", "attacked")
LIN4 = ["LR-B", "LR-C", "LR-B-noobs", "LR-C-noobs"]
NAN = float("nan")


def log(msg):
    line = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(os.path.join(PARTS, "log.txt"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def kwflag(t):
    return any(k in t.lower() for k in KW)


def r4(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)


# ============================================================================ data
_D = {}


def D(name):
    if name in _D:
        return _D[name]
    if name == "st_tr":
        v = load(os.path.join(MP, "data", "train.jsonl"))
    elif name == "st_te":
        v = load(os.path.join(MP, "data", "test.jsonl"))
    elif name == "val":
        v = load(os.path.join(MP, "data", "val.jsonl"))
    elif name == "pool":
        v = D("st_tr") + D("st_te")
    elif name == "p0":
        v = load(os.path.join(ROOT, "agentdojo_traces.jsonl"))
    elif name == "failed":
        v = load(os.path.join(ROOT, "agentdojo_failed.jsonl"))
    elif name == "p0_v03":
        v = load(os.path.join(ROOT, "archive_v0.3", "agentdojo_traces.jsonl"))
    else:
        raise KeyError(name)
    _D[name] = v
    return v


_LC = {}
# The pickles read below are the project's own prediction caches (revision/cache/*, written by build_linear_cache.py
# and build_gru_cache.py in this revision, format fixed by the analysis plan); they are trusted local files.


def lin(setting):
    if setting not in _LC:
        _LC[setting] = pickle.load(open(os.path.join(CACHE, "linear", f"{setting}.pkl"), "rb"))
    return _LC[setting]


def gru(setting):
    return pickle.load(open(os.path.join(CACHE, "gru", f"{setting}.pkl"), "rb"))


def ylab(recs):
    return np.array([int(r["category"] == "attacked") for r in recs], int)


def beh(recs):
    return np.array([vectorise(r, {})[0] for r in recs], float)


def bag(recs, vocab):
    X = np.zeros((len(recs), len(vocab)), float)
    for i, r in enumerate(recs):
        for t in tool_seq(r):
            j = vocab.get(t)
            if j is not None:
                X[i, j] += 1.0
    return X


def dojo_cluster(r):
    return f"{r['meta']['suite']}|{r['meta']['user_task']}"


def world_key(r):
    return json.dumps(r["world"], sort_keys=True)


def single_split(setting):
    """Train / test records in the cache's row order (build_linear_cache.single_split)."""
    if setting == "R1":
        return D("st_tr"), D("st_te")
    if setting.startswith("R3_"):
        d = setting[3:]
        return [r for r in D("pool") if r["agent"] != d], [r for r in D("pool") if r["agent"] == d]
    if setting == "R4":
        return D("pool"), D("p0") + D("failed")
    raise KeyError(setting)


def check_ids(recs, cache):
    return [r["id"] for r in recs] == [m["id"] for m in cache["test_meta"]]


# ============================================================================ fitting (cache recipe)
def by_unique_rows(fn, A):
    U, inv = np.unique(A, axis=0, return_inverse=True)
    return np.asarray(fn(U))[np.asarray(inv).reshape(-1)]


def lr_fit(X, y):
    """ablation.fit / build_linear_cache.lr_fit recipe: StandardScaler + LogisticRegression(max_iter=1000,
    class_weight='balanced'); t_r80 = 20th percentile of in-sample training-positive scores (tie-exact scoring)."""
    sc = StandardScaler().fit(X)
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always", ConvergenceWarning)
        lr = LogisticRegression(max_iter=1000, class_weight="balanced").fit(sc.transform(X), y)
    conv = not any(issubclass(w.category, ConvergenceWarning) for w in wl)
    s_tr = by_unique_rows(lambda U: lr.predict_proba(sc.transform(U))[:, 1], X)
    return {"sc": sc, "lr": lr, "t80": ek.thr_at_recall(s_tr[y == 1], 0.80), "conv": conv}


def lr_score(m, X):
    return by_unique_rows(lambda U: m["lr"].predict_proba(m["sc"].transform(U))[:, 1], X)


def cols_of(det, V):
    bagc = list(range(10, 10 + V))
    return {"LR-B": list(range(10)), "LR-C": list(range(10)) + bagc, "LR-B-noobs": list(range(8)),
            "LR-C-noobs": list(range(8)) + bagc, "LR-obsonly": [8, 9]}[det]


def coef_model(c):
    """Rebuild a cached single-fit LR from its stored coefficients (scaler mean/scale, coef, intercept)."""
    mean, scale, w, b0 = c["scaler_mean"], c["scaler_scale"], c["coef"], c["intercept"]

    def f(X):
        return by_unique_rows(lambda U: 1.0 / (1.0 + np.exp(-(((U - mean) / scale) @ w + b0))), X)
    return f


# ============================================================================ statistics
def fast_ap(y, s, pi=None):
    """Average precision (sklearn definition, ties grouped). pi: prevalence weighting of positives
    (weight pi/(1-pi) * n_neg/n_pos; plan Sec. 3.1)."""
    y = np.asarray(y); npos = int(y.sum()); nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return NAN
    w = np.ones(len(y))
    if pi is not None:
        w = np.where(y == 1, pi / (1 - pi) * nneg / npos, 1.0)
    o = np.argsort(-np.asarray(s), kind="mergesort")
    ss = np.asarray(s)[o]; ys = y[o]; ws = w[o]
    tp = np.cumsum(ws * ys); fp = np.cumsum(ws * (1 - ys))
    last = np.r_[np.where(np.diff(ss) != 0)[0], len(ss) - 1]
    tps, fps = tp[last], fp[last]
    prec = tps / (tps + fps); rec = tps / tps[-1]
    return float(np.sum((rec - np.r_[0.0, rec[:-1]]) * prec))


def metric(name, y, s, p):
    if name == "AUROC":
        return ek.auroc(y, s)
    if name == "AUPRC":
        return fast_ap(y, s)
    if name.startswith("wAUPRC@"):
        return fast_ap(y, s, float(name[7:]))
    if name.startswith("recall@"):
        m = y == 1
        return float(p[name[7:]][m].mean()) if m.any() else NAN
    if name.startswith("FPR@"):
        m = y == 0
        return float(p[name[4:]][m].mean()) if m.any() else NAN
    if name == "mean_score_pos":
        return float(np.mean(s[y == 1]))
    raise KeyError(name)


def gen_idx(y, pool, nb, seed, clusters=None):
    rng = np.random.default_rng(seed)
    pool = np.asarray(pool)
    out = []
    if clusters is None:
        pos = pool[y[pool] == 1]; neg = pool[y[pool] == 0]
        for _ in range(nb):
            out.append(np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]))
        ncl = None
    else:
        cl = np.asarray(clusters, dtype=object)[pool]
        keys = sorted(set(cl)); pos_of = {k: i for i, k in enumerate(keys)}
        mem = [[] for _ in keys]
        for i, c in zip(pool, cl):
            mem[pos_of[c]].append(i)
        mem = [np.asarray(m) for m in mem]
        for _ in range(nb):
            pick = rng.integers(0, len(keys), len(keys))
            out.append(np.concatenate([mem[k] for k in pick]))
        ncl = len(keys)
    u = rng.random(nb)          # repetition / seed selector, shared by every method in a replicate (paired)
    return out, u, ncl


def pval(d):
    d = d[~np.isnan(d)]
    if len(d) == 0:
        return NAN
    nb = len(d)
    return float(min(1.0, 2 * min((1 + np.sum(d <= 0)) / (nb + 1), (1 + np.sum(d >= 0)) / (nb + 1))))


def evaluate(methods, y, pool, metrics, seed, clusters=None, contrasts=(), nb=B, label=""):
    """methods: {name: {"score": (R, n) or (n,), "pred": {op: same shape bool}}}. Point = mean over repetitions of
    the metric on the whole pool (evalkit.point); interval = percentile bootstrap (stratified by class, or by
    cluster), one repetition drawn per replicate and shared by all methods, so contrasts are paired."""
    t0 = time.time()
    pool = np.asarray(pool)
    idx, u, ncl = gen_idx(y, pool, nb, seed, clusters)
    point, boot = {}, {}
    for m, spec in methods.items():
        S = np.atleast_2d(np.asarray(spec["score"], float)); R = S.shape[0]
        P = {k: np.atleast_2d(np.asarray(v, bool)) for k, v in spec.get("pred", {}).items()}
        mets = [k for k in metrics if not (k.startswith("recall@") and k[7:] not in P)
                and not (k.startswith("FPR@") and k[4:] not in P)]
        yp = y[pool]
        point[m] = {k: float(np.nanmean([metric(k, yp, S[r][pool], {o: P[o][r][pool] for o in P})
                                         for r in range(R)])) for k in mets}
        bm = {k: np.empty(nb) for k in mets}
        for b in range(nb):
            ib = idx[b]; r = min(int(u[b] * R), R - 1); yb = y[ib]
            pb = {o: P[o][r][ib] for o in P}
            sb = S[r][ib]
            for k in mets:
                bm[k][b] = metric(k, yb, sb, pb)
        boot[m] = bm
    res = {"population": label, "n_pos": int(y[pool].sum()), "n_neg": int((y[pool] == 0).sum()),
           "bootstrap": "stratified trajectory-level" if clusters is None else f"cluster ({ncl} clusters)",
           "B": nb, "seed": seed, "metrics": {}, "contrasts": {}}
    for m in methods:
        res["metrics"][m] = {}
        for k, v in boot[m].items():
            vv = v[~np.isnan(v)]
            res["metrics"][m][k] = [r4(point[m][k]), r4(np.percentile(vv, 2.5)) if len(vv) else None,
                                    r4(np.percentile(vv, 97.5)) if len(vv) else None]
    for a, bb in contrasts:
        cname = f"{a} minus {bb}"; res["contrasts"][cname] = {}
        for k in boot[a]:
            if k not in boot[bb]:
                continue
            d = boot[a][k] - boot[bb][k]; dd = d[~np.isnan(d)]
            lo, hi = (np.percentile(dd, 2.5), np.percentile(dd, 97.5)) if len(dd) else (NAN, NAN)
            res["contrasts"][cname][k] = {"delta": r4(point[a][k] - point[bb][k]), "lo": r4(lo), "hi": r4(hi),
                                          "p_boot": r4(pval(d)), "detected": bool(lo > 0 or hi < 0)}
    res["runtime_s"] = round(time.time() - t0, 1)
    return res


def holm(ps):
    ps = np.asarray(ps, float); m = len(ps); order = np.argsort(ps); adj = np.empty(m); run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * ps[i])); adj[i] = run
    return adj


def ratio_boot(num, den, nb, seed, clusters=None):
    """Ratio sum(num)/sum(den) over units (trajectories), bootstrap over units or clusters."""
    num = np.asarray(num, float); den = np.asarray(den, float); n = len(num)
    rng = np.random.default_rng(seed)
    if clusters is None:
        vals = []
        for _ in range(nb):
            i = rng.integers(0, n, n); vals.append(num[i].sum() / den[i].sum())
    else:
        cl = np.asarray(clusters, dtype=object); keys = sorted(set(cl))
        cn = np.array([num[cl == k].sum() for k in keys]); cd = np.array([den[cl == k].sum() for k in keys])
        vals = []
        for _ in range(nb):
            i = rng.integers(0, len(keys), len(keys)); vals.append(cn[i].sum() / cd[i].sum())
    vals = np.asarray(vals)
    return [r4(num.sum() / den.sum()), r4(np.percentile(vals, 2.5)), r4(np.percentile(vals, 97.5))]


# ============================================================================ padding helpers
def filler(n):
    return (SENT * (n // len(SENT) + 1))[:n]


def pad_records(recs, n, steps_of):
    """Copy of recs where the observation of every step in steps_of(r) gets n filler characters appended."""
    if n == 0:
        return recs
    out = []
    for r in recs:
        idx = steps_of(r)
        if not idx:
            out.append(r); continue
        r2 = dict(r); r2["steps"] = [dict(s) for s in r["steps"]]
        for i in idx:
            r2["steps"][i]["obs"] = str(r2["steps"][i]["obs"]) + filler(n)
        out.append(r2)
    return out


def drift_injection_steps(r):
    if r["category"] != "attacked":
        return []
    return [i for i, s in enumerate(r["steps"]) if s.get("label") == "injection_point"]


_AN = re.compile(r"[^a-z0-9]")
_ESC = re.compile(r"\\[ntr]")


def _norm(s):
    return _AN.sub("", _ESC.sub(" ", str(s)).lower())


def raw_run_path(meta):
    at = meta["attack_type"]
    if at is None:
        return os.path.join(ADJ, "runs", meta["model"], meta["suite"], meta["user_task"], "none", "none.json")
    return os.path.join(ADJ, "runs", meta["model"], meta["suite"], meta["user_task"], at,
                        f"{meta['injection_task']}.json")


_INJ = {}


def dojo_injection_steps(r):
    """Steps of a compromised AgentDojo trace whose observation contains an injection string of the raw run's
    `injections` field (match on lower-case alphanumerics, first or last 60 characters of the injection text)."""
    if r["category"] not in ("attacked", "attack_failed"):
        return []
    if r["id"] in _INJ:
        return _INJ[r["id"]][0]
    p = raw_run_path(r["meta"])
    if not os.path.exists(p):
        _INJ[r["id"]] = ([], "no_raw_file"); return []
    d = json.load(open(p, encoding="utf-8"))
    vals = [_norm(v) for v in (d.get("injections") or {}).values()]
    vals = [v for v in vals if len(v) >= 20]
    if not vals:
        _INJ[r["id"]] = ([], "no_injection_field"); return []
    probes = sorted({q for v in vals for q in ((v[:60], v[-60:]) if len(v) > 60 else (v,))})
    hits = [i for i, s in enumerate(r["steps"]) if any(q in _norm(s["obs"]) for q in probes)]
    ok = (d.get("security") == r["meta"]["security"]) and (d.get("user_task_id") == r["meta"]["user_task"])
    _INJ[r["id"]] = (hits, ("located" if hits else "not_found") + ("" if ok else "_meta_mismatch"))
    return hits


# ============================================================================ parts
def part_annotation():
    """m8: write tool_state_annotation.json and the keyword-flag confusion / miss analysis."""
    from tool_state_labels import AGENTDRIFT as LD, AGENTDOJO as LJ
    out = {}
    ann = {"description": "State-changing annotation of every tool name in AgentDrift (train, val, test) and AgentDojo "
                          "(P0 and unsuccessful attacks), for review comment m8. Labels were assigned by reading the "
                          "tool name, an example of its arguments and an example observation, following the criterion "
                          "below; no detector output was used, but the listing the annotator read also showed each "
                          "tool's call and step-label counts (not blind). The labels live in revision/tool_state_labels.py.",
           "criterion": (sys.modules["tool_state_labels"].__doc__ or "").strip(),
           "keyword_list_featurise": list(KW),
           "fields": {"state_changing": "annotation", "ambiguous": "a reasonable annotator could flip the label",
                      "reason": "one-line justification", "keyword_flag": "featurise.vectorise keyword match",
                      "n_calls": "occurrences in the files listed under source",
                      "example_args": "first non-empty arguments in file order (cut at 200 characters)",
                      "example_obs": "observation of the same step (cut at 160 characters)"},
           "source": {"agentdrift": "data/train.jsonl + data/val.jsonl + data/test.jsonl",
                      "agentdojo": "agentdojo_traces.jsonl + agentdojo_failed.jsonl"}}
    for bench, recs, lab in [("agentdrift", D("pool") + D("val"), LD), ("agentdojo", D("p0") + D("failed"), LJ)]:
        cnt = collections.Counter(); ex = {}; exo = {}; labc = collections.defaultdict(collections.Counter)
        doms = collections.defaultdict(set)
        for r in recs:
            for s in r["steps"]:
                t = s["tool"]; cnt[t] += 1; doms[t].add(r["agent"]); labc[t][s.get("label", r["category"])] += 1
                if t not in ex or (ex[t] in ("{}", "") and str(s["args"]) not in ("{}", "")):
                    ex[t] = str(s["args"])[:200]; exo[t] = str(s["obs"])[:160]
        missing = sorted(set(cnt) - set(lab)); unused = sorted(set(lab) - set(cnt))
        assert not missing, f"{bench}: tools without annotation {missing}"
        ann[bench] = {t: {"state_changing": lab[t][0], "ambiguous": lab[t][1], "reason": lab[t][2],
                          "keyword_flag": kwflag(t), "n_calls": cnt[t], "domains": sorted(doms[t]),
                          "step_label_counts": dict(labc[t]), "example_args": ex[t], "example_obs": exo[t]}
                      for t in sorted(cnt)}
        ann[f"{bench}_summary"] = {"n_tools": len(cnt), "n_state_changing": sum(lab[t][0] for t in cnt),
                                   "n_ambiguous": sum(lab[t][1] for t in cnt), "unused_labels": unused}
    json.dump(ann, open(ANNOT_JSON, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    log(f"wrote {ANNOT_JSON}")

    def conf(items):
        tp = sum(w for k, a, w in items if k and a); fp = sum(w for k, a, w in items if k and not a)
        fn = sum(w for k, a, w in items if (not k) and a); tn = sum(w for k, a, w in items if (not k) and not a)
        return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": r4(tp / (tp + fp)) if tp + fp else None,
                "recall": r4(tp / (tp + fn)) if tp + fn else None}

    populations = {"agentdrift": ("pooled train+test (10,656 trajectories)", D("pool"), LD, None),
                   "agentdojo": ("P0 + unsuccessful attacks (7,673 traces)", D("p0") + D("failed"), LJ, dojo_cluster),
                   "agentdojo_P0": ("P0 (2,686 traces)", D("p0"), LJ, dojo_cluster)}
    for key, (desc, recs, lab, clf) in populations.items():
        tools = collections.Counter(s["tool"] for r in recs for s in r["steps"])
        res = {"population": desc,
               "tool_level": conf([(kwflag(t), lab[t][0], 1) for t in tools]),
               "tool_level_excluding_ambiguous": conf([(kwflag(t), lab[t][0], 1) for t in tools if not lab[t][1]]),
               "occurrence_level": conf([(kwflag(t), lab[t][0], c) for t, c in tools.items()]),
               "occurrence_level_excluding_ambiguous": conf([(kwflag(t), lab[t][0], c) for t, c in tools.items()
                                                             if not lab[t][1]]),
               "false_positive_tools": {t: tools[t] for t in sorted(tools) if kwflag(t) and not lab[t][0]},
               "false_negative_tools": {t: tools[t] for t in sorted(tools, key=lambda x: -tools[x])
                                        if lab[t][0] and not kwflag(t)}}
        # occurrence-level precision / recall with a bootstrap over trajectories (clusters for AgentDojo)
        k_ = np.array([sum(kwflag(s["tool"]) for s in r["steps"]) for r in recs])
        a_ = np.array([sum(lab[s["tool"]][0] for s in r["steps"]) for r in recs])
        ka = np.array([sum(kwflag(s["tool"]) and lab[s["tool"]][0] for s in r["steps"]) for r in recs])
        cl = [clf(r) for r in recs] if clf else None
        res["occurrence_precision_ci"] = ratio_boot(ka, k_, B, 8001, cl)
        res["occurrence_recall_ci"] = ratio_boot(ka, a_, B, 8002, cl)
        res["ci_note"] = ("tool-level counts are a census of the vocabulary (no sampling interval); occurrence-level "
                          "intervals resample trajectories" + (" by suite|user_task cluster" if clf else ""))
        out[f"keyword_vs_annotation_{key}"] = res

    # hijacked AgentDrift steps
    for key, recs in [("pooled", D("pool")), ("train_only", D("st_tr"))]:
        att = [r for r in recs if r["category"] == "attacked"]
        hs = [[s for s in r["steps"] if s["label"] == "hijacked"] for r in att]
        n = np.array([len(h) for h in hs]); nk = np.array([sum(kwflag(s["tool"]) for s in h) for h in hs])
        na = np.array([sum(LD[s["tool"]][0] for s in h) for h in hs])
        nam = np.array([sum(LD[s["tool"]][0] and not kwflag(s["tool"]) for s in h) for h in hs])
        missed_tools = collections.Counter(s["tool"] for h in hs for s in h if not kwflag(s["tool"]))
        missed_sc_tools = collections.Counter(s["tool"] for h in hs for s in h
                                              if LD[s["tool"]][0] and not kwflag(s["tool"]))
        out[f"agentdrift_hijacked_steps_{key}"] = {
            "n_attacked_trajectories": len(att), "n_hijacked_steps": int(n.sum()),
            "share_keyword_flagged": ratio_boot(nk, n, B, 8011),
            "share_not_keyword_flagged": ratio_boot(n - nk, n, B, 8012),
            "share_annotated_state_changing": ratio_boot(na, n, B, 8013),
            "n_annotated_state_changing": int(na.sum()),
            "n_annotated_state_changing_missed_by_keyword": int(nam.sum()),
            "share_of_annotated_state_changing_missed_by_keyword": ratio_boot(nam, na, B, 8014),
            "share_trajectories_with_no_keyword_flagged_hijacked_step": r4(np.mean(nk == 0)),
            "top_not_flagged_tools": dict(missed_tools.most_common(12)),
            "top_state_changing_missed_tools": dict(missed_sc_tools.most_common(12)),
            "ci_note": "bootstrap over attacked trajectories (B = 2000)"}
    # compromised AgentDojo calls (P0 attacked)
    att = [r for r in D("p0") if r["category"] == "attacked"]
    cl = [dojo_cluster(r) for r in att]
    n = np.array([len(r["steps"]) for r in att]); nk = np.array([sum(kwflag(s["tool"]) for s in r["steps"]) for r in att])
    na = np.array([sum(LJ[s["tool"]][0] for s in r["steps"]) for r in att])
    nam = np.array([sum(LJ[s["tool"]][0] and not kwflag(s["tool"]) for s in r["steps"]) for r in att])
    nkf = np.array([sum(kwflag(s["tool"]) and not LJ[s["tool"]][0] for s in r["steps"]) for r in att])
    post, post_a, post_am, loc = [], [], [], 0
    for r in att:
        h = dojo_injection_steps(r)
        if h:
            loc += 1; ss = r["steps"][h[0] + 1:]
        else:
            ss = []
        post.append(len(ss)); post_a.append(sum(LJ[s["tool"]][0] for s in ss))
        post_am.append(sum(LJ[s["tool"]][0] and not kwflag(s["tool"]) for s in ss))
    missed = collections.Counter(s["tool"] for r in att for s in r["steps"] if LJ[s["tool"]][0] and not kwflag(s["tool"]))
    out["agentdojo_compromised_calls"] = {
        "n_compromised_traces": len(att), "n_calls": int(n.sum()),
        "share_keyword_flagged": ratio_boot(nk, n, B, 8021, cl),
        "n_annotated_state_changing": int(na.sum()),
        "share_annotated_state_changing": ratio_boot(na, n, B, 8022, cl),
        "n_annotated_state_changing_missed_by_keyword": int(nam.sum()),
        "share_of_annotated_state_changing_missed_by_keyword": ratio_boot(nam, na, B, 8023, cl),
        "n_keyword_flagged_but_read_only": int(nkf.sum()),
        "top_state_changing_missed_tools": dict(missed.most_common(12)),
        "post_injection_proxy": {
            "definition": "calls after the first step whose observation carries the injection text (located from the raw "
                          "run's `injections` field); AgentDojo has no per-call hijack label, so this is a proxy",
            "n_traces_with_located_injection": loc, "n_calls_after_injection": int(np.sum(post)),
            "n_annotated_state_changing_after_injection": int(np.sum(post_a)),
            "n_missed_by_keyword": int(np.sum(post_am)),
            "share_missed_by_keyword": ratio_boot(post_am, post_a, B, 8024, cl) if np.sum(post_a) else None},
        "ci_note": "cluster bootstrap over suite|user_task (B = 2000)"}
    return out


def part_coefs():
    """M5 item 1: standardized coefficients (coefficients on StandardScaler-transformed features)."""
    out = {"note": "all detectors standardize features with StandardScaler before the logistic regression, so the "
                   "fitted coefficients are standardized coefficients (log-odds change per training-set SD)."}
    for s in ["R1"] + [f"R3_{d}" for d in DOMAINS] + ["R4"]:
        c = lin(s)["coefs"]; out[f"cache_{s}"] = {}
        for det, k in [("LR-B", 10), ("LR-C", 10), ("LR-B-noobs", 8), ("LR-C-noobs", 8)]:
            out[f"cache_{s}"][det] = {"names": c[det]["names"][:k], "coef": [r4(x) for x in c[det]["coef"][:k]],
                                      "intercept": r4(c[det]["intercept"]),
                                      "block": "all" if det in ("LR-B", "LR-B-noobs") else "behavioural block only"}
    # ---- R1 bootstrap refits (stratified resampling of the training set, vocabulary fixed)
    tr = D("st_tr"); y = ylab(tr); Xb = beh(tr)
    vocab = build_vocab([r for r in tr if r["category"] == "benign"])
    names = [n[5:] for n in lin("R1")["coefs"]["LR-C"]["names"][10:]]
    assert names == sorted(vocab, key=lambda t: vocab[t]), "vocabulary order differs from the cache"
    X = np.hstack([Xb, bag(tr, vocab)])
    rng = np.random.default_rng(8101); pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    NB = 500
    boots = {"LR-B": [], "LR-C": []}; t0 = time.time()
    ref = {det: lr_fit(X[:, cols_of(det, len(vocab))], y) for det in ["LR-B", "LR-C"]}
    chk = {det: float(np.max(np.abs(ref[det]["lr"].coef_[0] - lin("R1")["coefs"][det]["coef"]))) for det in ref}
    for b in range(NB):
        i = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        for det in boots:
            m = lr_fit(X[i][:, cols_of(det, len(vocab))], y[i]); boots[det].append(m["lr"].coef_[0][:10])
    for det in boots:
        A = np.array(boots[det]); pt = lin("R1")["coefs"][det]["coef"][:10]
        out[f"R1_{det}_bootstrap"] = {FEAT[j]: [r4(pt[j]), r4(np.percentile(A[:, j], 2.5)),
                                                r4(np.percentile(A[:, j], 97.5))] for j in range(10)}
    out["R1_bootstrap_note"] = (f"{NB} refits on stratified resamples of data/train.jsonl, benign tool vocabulary of the "
                                f"original training set; point = cached fit; {round(time.time() - t0)} s")
    out["R1_refit_equals_cache_max_abs_coef_diff"] = {k: float(v) for k, v in chk.items()}
    # ---- WUT: the 25 fold models (P0 rows of the training folds), checked against the cached OOF scores
    p0 = D("p0"); W = lin("WUT"); nP = len(p0)
    assert [m["id"] for m in W["test_meta"][:nP]] == [r["id"] for r in p0]
    folds = W["folds"][:, :nP]; yP = ylab(p0); XbP = beh(p0)
    co = {"LR-B": [], "LR-C": []}; maxdiff = {"LR-B": 0.0, "LR-C": 0.0}; thrdiff = {"LR-B": 0.0, "LR-C": 0.0}
    for rep in range(5):
        for f in range(5):
            tri = np.where(folds[rep] != f)[0]; tei = np.where(folds[rep] == f)[0]
            trr = [p0[i] for i in tri]
            voc = build_vocab([r for r in trr if r["category"] == "benign"])
            Xtr = np.hstack([XbP[tri], bag(trr, voc)]); Xte = np.hstack([XbP[tei], bag([p0[i] for i in tei], voc)])
            for det in co:
                c = cols_of(det, len(voc)); m = lr_fit(Xtr[:, c], yP[tri]); co[det].append(m["lr"].coef_[0][:10])
                s = lr_score(m, Xte[:, c])
                maxdiff[det] = max(maxdiff[det], float(np.max(np.abs(s - W["scores"][det][rep][tei]))))
                thrdiff[det] = max(thrdiff[det], abs(m["t80"] - float(W["thr"][det]["t_r80"][rep][f])))
    for det in co:
        A = np.array(co[det])
        out[f"WUT_{det}_25fold"] = {FEAT[j]: {"mean": r4(A[:, j].mean()), "min": r4(A[:, j].min()),
                                              "max": r4(A[:, j].max()),
                                              "share_same_sign_as_mean": r4(np.mean(np.sign(A[:, j]) == np.sign(A[:, j].mean())))}
                                    for j in range(10)}
    out["WUT_refit_check"] = {"max_abs_score_diff_vs_cache": maxdiff, "max_abs_t_r80_diff_vs_cache": thrdiff}
    # ---- within-AgentDojo model on all P0 rows, cluster bootstrap of the coefficients
    voc = build_vocab([r for r in p0 if r["category"] == "benign"])
    XP = np.hstack([XbP, bag(p0, voc)]); cl = np.array([dojo_cluster(r) for r in p0], dtype=object)
    keys = sorted(set(cl)); mem = [np.where(cl == k)[0] for k in keys]; rng = np.random.default_rng(8102)
    full = {det: lr_fit(XP[:, cols_of(det, len(voc))], yP)["lr"].coef_[0][:10] for det in co}
    bo = {det: [] for det in co}
    for b in range(NB):
        i = np.concatenate([mem[k] for k in rng.integers(0, len(keys), len(keys))])
        if len(set(yP[i])) < 2:
            continue
        for det in co:
            bo[det].append(lr_fit(XP[i][:, cols_of(det, len(voc))], yP[i])["lr"].coef_[0][:10])
    for det in co:
        A = np.array(bo[det])
        out[f"P0all_{det}_cluster_bootstrap"] = {FEAT[j]: [r4(full[det][j]), r4(np.percentile(A[:, j], 2.5)),
                                                           r4(np.percentile(A[:, j], 97.5))] for j in range(10)}
    out["P0all_note"] = (f"within-AgentDojo model fitted on all 2,686 P0 rows (same recipe; vocabulary from the 392 benign "
                         f"traces); {NB} cluster-bootstrap refits over {len(keys)} suite|user_task clusters. This is a "
                         f"descriptive fit, not one of the WUT fold models.")
    # sign reversals of the behavioural coefficients between R1 and WUT
    rev = {}
    for det in co:
        r1 = out[f"R1_{det}_bootstrap"]; wa = out[f"P0all_{det}_cluster_bootstrap"]
        rev[det] = [FEAT[j] for j in range(10) if (r1[FEAT[j]][1] > 0 and wa[FEAT[j]][2] < 0)
                    or (r1[FEAT[j]][2] < 0 and wa[FEAT[j]][1] > 0)]
    out["coef_sign_reversals_R1_vs_P0all_detected"] = rev
    return out


def part_univariate():
    """M5 item 1: univariate AUROC of each behavioural feature per population; direction reversals; range overlap."""
    out = {"populations": {}}
    pops = []
    te = D("st_te"); keep = [r for r in te if r["category"] in POOL_E]
    pops.append(("R1_poolE", keep, None, 9101))
    for k, d in enumerate(DOMAINS):
        pops.append((f"R3_{d}_poolE", [r for r in D("pool") if r["agent"] == d and r["category"] in POOL_E], None, 9111 + k))
    pops.append(("AgentDrift_pooled_poolE", [r for r in D("pool") if r["category"] in POOL_E], None, 9121))
    pops.append(("AgentDojo_P0", D("p0"), "cluster", 9131))
    pops.append(("AgentDojo_P0_stratified", D("p0"), None, 9131))
    for name, recs, clu, seed in pops:
        X = beh(recs); y = ylab(recs)
        methods = {FEAT[j]: {"score": X[:, j]} for j in range(10)}
        cl = [dojo_cluster(r) for r in recs] if clu else None
        res = evaluate(methods, y, np.arange(len(y)), ["AUROC"], seed, clusters=cl, label=name)
        res["median_by_class"] = {FEAT[j]: {"attacked": r4(np.median(X[y == 1, j])), "negative": r4(np.median(X[y == 0, j]))}
                                  for j in range(10)}
        out["populations"][name] = res
        log(f"univariate {name} {res['runtime_s']} s")
    P = out["populations"]
    au = lambda p, f: P[p]["metrics"][f]["AUROC"]
    rev = {}
    for p in [x[0] for x in pops if x[0].startswith(("R1", "R3", "AgentDrift"))]:
        rev[p] = {"reversal_detected": [f for f in FEAT if (au(p, f)[1] > 0.5 and au("AgentDojo_P0", f)[2] < 0.5)
                                        or (au(p, f)[2] < 0.5 and au("AgentDojo_P0", f)[1] > 0.5)],
                  "opposite_side_of_0.5_point": [f for f in FEAT if (au(p, f)[0] - 0.5) * (au("AgentDojo_P0", f)[0] - 0.5) < 0]}
    out["reversals_vs_AgentDojo_P0"] = rev
    out["reversals_R1_vs_R3"] = {d: [f for f in FEAT if (au("R1_poolE", f)[1] > 0.5 and au(f"R3_{d}_poolE", f)[2] < 0.5)
                                     or (au("R1_poolE", f)[2] < 0.5 and au(f"R3_{d}_poolE", f)[1] > 0.5)] for d in DOMAINS}
    out["reversal_definition"] = ("detected reversal: the 95% interval lies above 0.5 in one population and below 0.5 in "
                                  "the other (AgentDojo intervals: cluster bootstrap)")
    # range overlap: AgentDrift pooled (R4 training set, all 10,656) vs AgentDojo
    Xd = beh(D("pool")); Xp = beh(D("p0")); Xf = beh(D("failed"))
    c4 = lin("R4")["coefs"]["LR-B"]; mu, sd = c4["scaler_mean"], c4["scaler_scale"]
    rng_ = {}
    for j in range(10):
        lo, hi = Xd[:, j].min(), Xd[:, j].max()
        rng_[FEAT[j]] = {"agentdrift_min_median_max": [r4(lo), r4(np.median(Xd[:, j])), r4(hi)],
                         "agentdojo_P0_min_median_max": [r4(Xp[:, j].min()), r4(np.median(Xp[:, j])), r4(Xp[:, j].max())],
                         "agentdojo_P0_share_above_agentdrift_max": r4(np.mean(Xp[:, j] > hi)),
                         "agentdojo_P0_share_below_agentdrift_min": r4(np.mean(Xp[:, j] < lo)),
                         "agentdojo_failed_share_above_agentdrift_max": r4(np.mean(Xf[:, j] > hi)),
                         "agentdojo_P0_share_abs_z_gt_3_under_R4_scaler": r4(np.mean(np.abs((Xp[:, j] - mu[j]) / sd[j]) > 3)),
                         "agentdojo_P0_max_abs_z_under_R4_scaler": r4(np.max(np.abs((Xp[:, j] - mu[j]) / sd[j])))}
    out["range_overlap"] = rng_
    # reconciliation with the review (which used the v0.3 extraction): point AUROC on archive_v0.3 P0
    a3 = D("p0_v03"); X3 = beh(a3); y3 = ylab(a3)
    out["v03_archive_P0_point_AUROC"] = {FEAT[j]: r4(ek.auroc(y3, X3[:, j])) for j in range(10)}
    return out


def _lin_methods(cache, dets, extra=None):
    m = {}
    for d in dets:
        m[d] = {"score": cache["scores"][d], "pred": {"t_r80": cache["pred"][d]["t_r80"]}}
    if extra:
        m.update(extra)
    return m


def _obsonly_single(setting):
    tr, te = single_split(setting)
    assert check_ids(te, lin(setting))
    ytr = ylab(tr); Xtr = beh(tr)[:, [8, 9]]; Xte = beh(te)[:, [8, 9]]
    m = lr_fit(Xtr, ytr); s = lr_score(m, Xte)
    return {"score": s, "pred": {"t_r80": s >= m["t80"]}, "t80": m["t80"], "coef": m["lr"].coef_[0].tolist()}


def _obsonly_wut():
    W = lin("WUT"); recs = D("p0") + D("failed"); assert check_ids(recs, W)
    nP = len(D("p0")); X = beh(recs)[:, [8, 9]]; y = ylab(recs)
    S = np.zeros((5, len(recs))); Pd = np.zeros((5, len(recs)), bool)
    for rep in range(5):
        for f in range(5):
            tri = np.where((W["folds"][rep] != f) & (np.arange(len(recs)) < nP))[0]; tei = np.where(W["folds"][rep] == f)[0]
            m = lr_fit(X[tri], y[tri]); s = lr_score(m, X[tei]); S[rep, tei] = s; Pd[rep, tei] = s >= m["t80"]
    return {"score": S, "pred": {"t_r80": Pd}}


NOOBS_SETTINGS = ["R1"] + [f"R3_{d}" for d in DOMAINS] + ["R4", "WUT"]
NOOBS_METRICS = ["AUROC", "AUPRC", "wAUPRC@0.01", "wAUPRC@0.05", "wAUPRC@0.1", "recall@t_r80", "FPR@t_r80"]
NOOBS_CONTRASTS = [("LR-C-noobs", "LR-B-noobs"), ("LR-B-noobs", "LR-B"), ("LR-C-noobs", "LR-C"), ("LR-B", "LR-C"),
                   ("LR-obsonly", "LR-B")]


def part_noobs(budget=560):
    """M5 item 2: LR-B-noobs / LR-C-noobs in R1, R3, R4, WUT (cache) + LR-obsonly (features 8, 9 only; new fit)."""
    t_start = time.time(); out = {}
    for k, s in enumerate(NOOBS_SETTINGS):
        f = os.path.join(PARTS, f"noobs_{s}.json")
        if os.path.exists(f):
            out[s] = json.load(open(f)); continue
        if time.time() - t_start > budget:
            log("noobs: budget reached; re-run to continue"); return None
        c = lin(s); y = c["y"]; pool = np.where(c["pool_E"])[0]
        ob = _obsonly_wut() if s == "WUT" else _obsonly_single(s)
        meth = _lin_methods(c, LIN4, {"LR-obsonly": {"score": ob["score"], "pred": ob["pred"]}})
        res = {}
        if c["test_meta"][0]["benchmark"] == "agentdrift":
            res["primary"] = evaluate(meth, y, pool, NOOBS_METRICS, 9200 + k, contrasts=NOOBS_CONTRASTS, label=f"{s} pool E")
            res["world_cluster_sensitivity"] = evaluate(meth, y, pool, ["AUROC", "AUPRC"], 9300 + k,
                                                        clusters=[m["world"] for m in c["test_meta"]],
                                                        contrasts=NOOBS_CONTRASTS, label=f"{s} pool E, world clusters")
        else:
            cl = [m["task"] for m in c["test_meta"]]
            res["primary"] = evaluate(meth, y, pool, NOOBS_METRICS, 9200 + k, clusters=cl, contrasts=NOOBS_CONTRASTS,
                                      label=f"{s} P0, suite|user_task clusters")
            res["stratified_sensitivity"] = evaluate(meth, y, pool, ["AUROC", "AUPRC"], 9300 + k,
                                                     contrasts=NOOBS_CONTRASTS, label=f"{s} P0, stratified")
        if "coef" in ob:
            res["LR-obsonly_coef_max_obs_mean_obs"] = [r4(x) for x in ob["coef"]]
        json.dump(res, open(f, "w"), indent=1); out[s] = res
        log(f"noobs {s} done ({res['primary']['runtime_s']} s)")
    # within (WUT) minus transfer (R4), paired on the P0 rows
    f = os.path.join(PARTS, "noobs_within_minus_transfer.json")
    if not os.path.exists(f):
        c4, cw = lin("R4"), lin("WUT"); nP = len(D("p0"))
        assert [m["id"] for m in c4["test_meta"][:nP]] == [m["id"] for m in cw["test_meta"][:nP]]
        ob4 = _obsonly_single("R4"); obw = _obsonly_wut()
        meth = {}
        for d in LIN4:
            meth[f"R4:{d}"] = {"score": np.asarray(c4["scores"][d])[:nP], "pred": {"t_r80": np.asarray(c4["pred"][d]["t_r80"])[:nP]}}
            meth[f"WUT:{d}"] = {"score": np.asarray(cw["scores"][d])[:, :nP], "pred": {"t_r80": np.asarray(cw["pred"][d]["t_r80"])[:, :nP]}}
        meth["R4:LR-obsonly"] = {"score": ob4["score"][:nP], "pred": {"t_r80": ob4["pred"]["t_r80"][:nP]}}
        meth["WUT:LR-obsonly"] = {"score": obw["score"][:, :nP], "pred": {"t_r80": obw["pred"]["t_r80"][:, :nP]}}
        y = c4["y"][:nP]; cl = [m["task"] for m in c4["test_meta"][:nP]]
        dets = LIN4 + ["LR-obsonly"]
        res = evaluate(meth, y, np.arange(nP), ["AUROC", "AUPRC", "wAUPRC@0.1"], 9400, clusters=cl,
                       contrasts=[(f"WUT:{d}", f"R4:{d}") for d in dets], label="P0, suite|user_task clusters")
        for mt in ["AUROC", "AUPRC", "wAUPRC@0.1"]:
            ps = [res["contrasts"][f"WUT:{d} minus R4:{d}"][mt]["p_boot"] for d in dets]
            for d, pa in zip(dets, holm(ps)):
                res["contrasts"][f"WUT:{d} minus R4:{d}"][mt]["p_holm"] = r4(pa)
                res["contrasts"][f"WUT:{d} minus R4:{d}"][mt]["detected_holm"] = bool(pa < 0.05)
        res["holm_family"] = "within minus transfer, over the 5 detectors of this table (plan family d restricted)"
        json.dump(res, open(f, "w"), indent=1)
    out["within_minus_transfer"] = json.load(open(f))
    # Holm across the five R3 folds for every contrast and metric
    hol = {}
    for cn in [f"{a} minus {b}" for a, b in NOOBS_CONTRASTS]:
        hol[cn] = {}
        for mt in NOOBS_METRICS:
            try:
                ps = [out[f"R3_{d}"]["primary"]["contrasts"][cn][mt]["p_boot"] for d in DOMAINS]
            except KeyError:
                continue
            adj = holm(ps)
            hol[cn][mt] = {d: {"delta": out[f"R3_{d}"]["primary"]["contrasts"][cn][mt]["delta"],
                               "lo": out[f"R3_{d}"]["primary"]["contrasts"][cn][mt]["lo"],
                               "hi": out[f"R3_{d}"]["primary"]["contrasts"][cn][mt]["hi"],
                               "p_boot": ps[i], "p_holm": r4(adj[i]), "detected_holm": bool(adj[i] < 0.05)}
                           for i, d in enumerate(DOMAINS)}
            hol[cn][mt]["n_folds_detected_holm"] = int(sum(adj < 0.05))
            hol[cn][mt]["n_folds_detected_uncorrected"] = int(sum(out[f"R3_{d}"]["primary"]["contrasts"][cn][mt]["detected"] for d in DOMAINS))
    out["R3_holm"] = hol
    return out


def part_pad_r1_linear():
    """M5 item 3: padding of the injection-point observation of attacked R1 test trajectories, linear detectors."""
    tr, te = D("st_tr"), D("st_te"); c = lin("R1"); assert check_ids(te, c)
    vocab_names = [n[5:] for n in c["coefs"]["LR-C"]["names"][10:]]
    vocab = {t: i for i, t in enumerate(vocab_names)}
    assert vocab == build_vocab([r for r in tr if r["category"] == "benign"])
    dets = ["LR-B", "LR-C", "LR-B-noobs", "LR-C-noobs"]
    fns = {d: coef_model(c["coefs"][d]) for d in dets}
    chk = {}
    S = {d: [] for d in dets}; featshift = {}
    for n in PADS:
        recs = pad_records(te, n, drift_injection_steps)
        Xb = beh(recs); X = np.hstack([Xb, bag(recs, vocab)])
        for d in dets:
            s = fns[d](X[:, cols_of(d, len(vocab))]); S[d].append(s)
        att = c["y"] == 1
        featshift[f"pad{n}"] = {"attacked_median_max_obs": r4(np.median(Xb[att, 8])),
                                "attacked_median_mean_obs": r4(np.median(Xb[att, 9]))}
    for d in dets:
        s0 = S[d][0]
        chk[d] = {"max_abs_diff_vs_cache": float(np.max(np.abs(s0 - c["scores"][d]))),
                  "t_r80_preds_equal_cache": bool(np.array_equal(s0 >= c["thr"][d]["t_r80"], c["pred"][d]["t_r80"]))}
    # independent refit with the v0.3 code path (ablation.fit) for LR-B and LR-C
    import ablation
    for d, rep in [("LR-B", "B"), ("LR-C", "C")]:
        mdl = ablation.fit(tr, rep); s = ablation.sco(mdl, te, rep)["score"]
        chk[d]["ablation_fit_max_abs_diff_vs_cache"] = float(np.max(np.abs(s - c["scores"][d])))
        chk[d]["ablation_t_r80_minus_cache"] = float(mdl[3] - c["thr"][d]["t_r80"])
    np.savez(os.path.join(PARTS, "pad_r1_linear_scores.npz"), **{d: np.array(S[d]) for d in dets})
    meth = {}
    for d in dets:
        for i, n in enumerate(PADS):
            meth[f"{d}|pad{n}"] = {"score": S[d][i], "pred": {"t_r80": S[d][i] >= c["thr"][d]["t_r80"]}}
    y = c["y"]; pool = np.where(c["pool_E"])[0]
    con = [(f"{d}|pad{n}", f"{d}|pad0") for d in dets for n in PADS[1:]]
    res = evaluate(meth, y, pool, ["AUROC", "recall@t_r80", "AUPRC", "mean_score_pos"], 9200, contrasts=con,
                   label="R1 pool E (padded attacked, unpadded negatives)")
    return {"eval": res, "checks": chk, "feature_shift": featshift,
            "n_attacked_padded": int(sum(1 for r in te if r["category"] == "attacked")),
            "n_injection_point_steps_per_attacked": sorted(set(len(drift_injection_steps(r)) for r in te if r["category"] == "attacked"))}


def part_pad_r1_gru():
    """M5 item 3, GRU: retrain the three pre-registered R1 GRU seeds (gru_detector.train_gru on
    gru_ladder.stratified_val of data/train.jsonl), confirm scores equal the cache, score the padded test sets."""
    import torch
    from gru_detector import train_gru, score_gru
    from gru_ladder import stratified_val
    tr, te = D("st_tr"), D("st_te"); g = gru("R1"); assert [m["id"] for m in g["test_meta"]] == [r["id"] for r in te]
    padded = {n: pad_records(te, n, drift_injection_steps) for n in PADS}
    for sd in [0, 1, 2]:
        f = os.path.join(PARTS, f"pad_r1_gru_seed{sd}.npz")
        if os.path.exists(f):
            continue
        t0 = time.time()
        trs, va = stratified_val(tr, sd)
        model, voc, scl, thr = train_gru(trs, va, sd)
        S = np.stack([score_gru(model, voc, scl, thr, padded[n])["score"] for n in PADS])
        np.savez(f, S=S, thr=thr, fit_s=time.time() - t0, torch_threads=torch.get_num_threads())
        log(f"GRU R1 seed {sd} trained and scored in {time.time() - t0:.1f} s")
    S = []; thr = []; chk = {}
    for sd in [0, 1, 2]:
        z = np.load(os.path.join(PARTS, f"pad_r1_gru_seed{sd}.npz")); S.append(z["S"]); thr.append(float(z["thr"]))
        cs = np.asarray(g["scores"]["GRU"][sd]); pE = np.where(g["pool_E"])[0]
        chk[f"seed{sd}"] = {"max_abs_diff_vs_cache": float(np.max(np.abs(z["S"][0] - cs))),
                            "bitwise_equal": bool(np.array_equal(z["S"][0].astype(cs.dtype), cs)),
                            "n_rows_not_bitwise_equal": int(np.sum(z["S"][0].astype(cs.dtype) != cs)),
                            "t_r80_preds_equal_cache": bool(np.array_equal(z["S"][0] >= float(z["thr"]),
                                                                           np.asarray(g["pred"]["GRU"]["t_r80"][sd]))),
                            "poolE_AUROC_refit_minus_cache": float(ek.auroc(g["y"][pE], z["S"][0][pE]) - ek.auroc(g["y"][pE], cs[pE])),
                            "t_r80_minus_cache": float(thr[-1] - float(g["thr"]["GRU"]["t_r80"][sd])),
                            "fit_s": round(float(z["fit_s"]), 1), "torch_threads": int(z["torch_threads"])}
    S = np.array(S)       # (seed, pad, n)
    thr = np.array(thr)
    meth = {f"GRU|pad{n}": {"score": S[:, i, :], "pred": {"t_r80": S[:, i, :] >= thr[:, None]}} for i, n in enumerate(PADS)}
    y = g["y"]; pool = np.where(g["pool_E"])[0]
    res = evaluate(meth, y, pool, ["AUROC", "recall@t_r80", "AUPRC", "mean_score_pos"], 9200,
                   contrasts=[(f"GRU|pad{n}", "GRU|pad0") for n in PADS[1:]],
                   label="R1 pool E (padded attacked, unpadded negatives); 3 seeds")
    return {"eval": res, "checks": chk, "thr": thr.tolist()}


def part_pad_wut():
    """M5 item 3, within-AgentDojo (WUT): pad the injected step(s) of compromised P0 traces, rescore with the WUT
    fold models (refitted with the cache recipe and checked against the cached out-of-fold scores)."""
    p0 = D("p0"); W = lin("WUT"); nP = len(p0)
    assert [m["id"] for m in W["test_meta"][:nP]] == [r["id"] for r in p0]
    y = ylab(p0); folds = W["folds"][:, :nP]
    loc = {}; status = collections.Counter(); by_at = collections.defaultdict(collections.Counter)
    for i, r in enumerate(p0):
        if r["category"] == "attacked":
            h = dojo_injection_steps(r); loc[i] = h; st = _INJ[r["id"]][1]; status[st] += 1
            by_at[r["meta"]["attack_type"]][bool(h)] += 1
    steps_of = lambda r: _INJ[r["id"]][0] if r["category"] == "attacked" else []
    located = np.array([bool(loc.get(i)) for i in range(nP)])
    dets = ["LR-C", "LR-B", "LR-B-noobs"]
    padded = {n: pad_records(p0, n, steps_of) for n in PADS}
    XbP = {n: beh(padded[n]) for n in PADS}
    S = {d: np.zeros((len(PADS), 5, nP)) for d in dets}; Pr = {d: np.zeros((len(PADS), 5, nP), bool) for d in dets}
    chk = {d: {"max_abs_score_diff_vs_cache": 0.0, "max_abs_t_r80_diff_vs_cache": 0.0, "t_r80_pred_mismatches": 0}
           for d in dets}
    for rep in range(5):
        for f in range(5):
            tri = np.where(folds[rep] != f)[0]; tei = np.where(folds[rep] == f)[0]
            trr = [p0[i] for i in tri]; voc = build_vocab([r for r in trr if r["category"] == "benign"])
            Xtr = np.hstack([XbP[0][tri], bag(trr, voc)])
            bag_te = bag([p0[i] for i in tei], voc)
            for d in dets:
                c = cols_of(d, len(voc)); m = lr_fit(Xtr[:, c], y[tri])
                for k, n in enumerate(PADS):
                    s = lr_score(m, np.hstack([XbP[n][tei], bag_te])[:, c])
                    S[d][k, rep, tei] = s; Pr[d][k, rep, tei] = s >= m["t80"]
                ch = chk[d]
                ch["max_abs_score_diff_vs_cache"] = max(ch["max_abs_score_diff_vs_cache"],
                                                        float(np.max(np.abs(S[d][0, rep, tei] - W["scores"][d][rep][tei]))))
                ch["max_abs_t_r80_diff_vs_cache"] = max(ch["max_abs_t_r80_diff_vs_cache"],
                                                        abs(m["t80"] - float(W["thr"][d]["t_r80"][rep][f])))
                ch["t_r80_pred_mismatches"] += int(np.sum(Pr[d][0, rep, tei] != W["pred"][d]["t_r80"][rep][tei]))
    meth = {f"{d}|pad{n}": {"score": S[d][k], "pred": {"t_r80": Pr[d][k]}} for d in dets for k, n in enumerate(PADS)}
    con = [(f"{d}|pad{n}", f"{d}|pad0") for d in dets for n in PADS[1:]]
    cl = [dojo_cluster(r) for r in p0]
    mets = ["AUROC", "recall@t_r80", "AUPRC"]
    res = {"primary": evaluate(meth, y, np.arange(nP), mets, 9207, clusters=cl, contrasts=con,
                               label="WUT P0 (padded compromised traces, unpadded benign), suite|user_task clusters"),
           "stratified_sensitivity": evaluate(meth, y, np.arange(nP), mets, 9307, contrasts=con,
                                              label="WUT P0, stratified"),
           "located_subset": evaluate(meth, y, np.where((y == 0) | located)[0], mets, 9602, clusters=cl, contrasts=con,
                                      label="WUT P0 restricted to benign + compromised traces with a located injected step")}
    return {"eval": res, "checks": chk,
            "location": {"method": "raw run file runs/<model>/<suite>/<user_task>/<attack_type>/<injection_task>.json; a "
                                   "step is an injected step if its observation contains the first or last 60 "
                                   "alphanumeric characters (lower case, escapes removed) of any value of `injections`",
                         "n_compromised": int(y.sum()), "status_counts": dict(status),
                         "n_located": int(located.sum()), "share_located": r4(located.sum() / y.sum()),
                         "n_injected_steps_per_located_trace": dict(collections.Counter(len(v) for v in loc.values() if v)),
                         "located_by_attack_type": {k: {"located": v[True], "not_located": v[False]} for k, v in sorted(by_at.items(), key=lambda kv: str(kv[0]))},
                         "unlocated_traces_are_left_unpadded": True}}


def _q(a):
    a = np.asarray(a, float)
    return {k: r4(v) for k, v in zip(["min", "p05", "p25", "p50", "p75", "p95", "p99", "max"],
                                     np.percentile(a, [0, 5, 25, 50, 75, 95, 99, 100]))} | {"mean": r4(a.mean()), "n": int(len(a))}


def part_truncation():
    """M5 item 4: observation-length distributions, share of AgentDojo traces beyond the AgentDrift maximum, and the
    steps affected by the v0.3 2,000-character cut."""
    out = {}
    pool, val, tr = D("pool"), D("val"), D("st_tr")
    # AgentDrift per-step lengths by step label (training set, as the review; and pooled)
    for key, recs in [("agentdrift_train", tr), ("agentdrift_pooled", pool)]:
        grp = collections.defaultdict(list)
        for r in recs:
            for s in r["steps"]:
                L = len(str(s["obs"]))
                grp["all"].append(L)
                grp[f"{s['label']}_steps_of_{r['category']}"].append(L)
                if s["label"] == "benign":
                    grp[f"benign_steps_of_{r['category']}"].append(L)
        out[f"{key}_step_obs_length"] = {k: _q(v) for k, v in sorted(grp.items())}
    # per-trajectory max_obs and mean_obs
    def traj(recs, by):
        X = beh(recs); g = collections.defaultdict(list)
        for r, x in zip(recs, X):
            g[by(r)].append(x); g["all"].append(x)
        return {k: {"max_obs": _q(np.array(v)[:, 8]), "mean_obs": _q(np.array(v)[:, 9])} for k, v in sorted(g.items())}
    out["agentdrift_pooled_trajectory"] = traj(pool, lambda r: r["category"])
    out["agentdojo_P0_trajectory"] = traj(D("p0"), lambda r: r["category"])
    out["agentdojo_failed_trajectory"] = traj(D("failed"), lambda r: r["category"])
    for key, recs in [("agentdojo_P0_step_obs_length", D("p0")), ("agentdojo_failed_step_obs_length", D("failed"))]:
        g = collections.defaultdict(list)
        for r in recs:
            for s in r["steps"]:
                g["all"].append(len(s["obs"])); g[f"steps_of_{r['category']}"].append(len(s["obs"]))
        out[key] = {k: _q(v) for k, v in g.items()}
    Xd = beh(pool); Xv = beh(val); Xp = beh(D("p0")); Xf = beh(D("failed"))
    dmax_max, dmax_mean = float(Xd[:, 8].max()), float(Xd[:, 9].max())
    out["agentdrift_max_obs_range_pooled"] = [r4(Xd[:, 8].min()), r4(dmax_max)]
    out["agentdrift_max_obs_range_including_val"] = [r4(min(Xd[:, 8].min(), Xv[:, 8].min())), r4(max(dmax_max, Xv[:, 8].max()))]
    out["agentdrift_mean_obs_max_pooled"] = r4(dmax_mean)
    cl = [dojo_cluster(r) for r in D("p0")]; yP = ylab(D("p0"))
    share = {}
    for nm, v in [("max_obs_above_agentdrift_max_obs", Xp[:, 8] > dmax_max),
                  ("mean_obs_above_agentdrift_max_obs", Xp[:, 9] > dmax_max),
                  ("mean_obs_above_agentdrift_max_mean_obs", Xp[:, 9] > dmax_mean)]:
        share[nm] = {"all_P0": ratio_boot(v.astype(float), np.ones(len(v)), B, 9701, cl),
                     "benign": ratio_boot(v[yP == 0].astype(float), np.ones(int((yP == 0).sum())), B, 9702,
                                          [c for c, t in zip(cl, yP) if t == 0]),
                     "compromised": ratio_boot(v[yP == 1].astype(float), np.ones(int(yP.sum())), B, 9703,
                                               [c for c, t in zip(cl, yP) if t == 1]),
                     "failed_attacks": r4(np.mean({"max_obs_above_agentdrift_max_obs": Xf[:, 8] > dmax_max,
                                                   "mean_obs_above_agentdrift_max_obs": Xf[:, 9] > dmax_max,
                                                   "mean_obs_above_agentdrift_max_mean_obs": Xf[:, 9] > dmax_mean}[nm]))}
    Xa3 = beh(D("p0_v03"))
    share["v03_archive_extraction_for_comparison_with_review"] = {
        "max_obs_above_agentdrift_max_obs": r4(np.mean(Xa3[:, 8] > dmax_max)),
        "mean_obs_above_agentdrift_max_obs": r4(np.mean(Xa3[:, 9] > dmax_max)),
        "mean_obs_above_agentdrift_max_mean_obs": r4(np.mean(Xa3[:, 9] > dmax_mean)),
        "note": "archive_v0.3/agentdojo_traces.jsonl (id-paired, cut at 2,000 characters), descriptive"}
    out["agentdojo_share_beyond_agentdrift"] = share
    out["agentdojo_share_beyond_agentdrift_note"] = ("AgentDrift maximum over the pooled 10,656 train+test trajectories; "
                                                     "intervals: cluster bootstrap over suite|user_task, B = 2000; "
                                                     "failed-attack shares are descriptive (no interval)")
    # the v0.3 2,000-character cut
    a = {r["id"]: r for r in D("p0_v03")}; bcur = {r["id"]: r for r in D("p0")}
    La = [len(s["obs"]) for r in D("p0_v03") for s in r["steps"]]
    Lb = [len(s["obs"]) for r in D("p0") for s in r["steps"]]
    Lf = [len(s["obs"]) for r in D("failed") for s in r["steps"]]
    same_steps = pure_trunc = other_diff = trunc_pos_cap = cap_not_long = 0
    tr_only_traces = 0
    for i, rb in bcur.items():
        ra = a.get(i)
        if ra is None or len(ra["steps"]) != len(rb["steps"]):
            continue
        same_steps += 1
        if [s["obs"] for s in ra["steps"]] == [s["obs"][:2000] for s in rb["steps"]] and any(len(s["obs"]) > 2000 for s in rb["steps"]):
            tr_only_traces += 1
        for sa, sb in zip(ra["steps"], rb["steps"]):
            if sa["obs"] == sb["obs"]:
                continue
            if len(sb["obs"]) > 2000 and sa["obs"] == sb["obs"][:2000]:
                pure_trunc += 1
            else:
                other_diff += 1
            if len(sa["obs"]) == 2000 and len(sb["obs"]) > 2000:
                trunc_pos_cap += 1
        cap_not_long += sum(1 for sa, sb in zip(ra["steps"], rb["steps"]) if len(sa["obs"]) == 2000 and len(sb["obs"]) <= 2000)
    Xc = beh([{"steps": [dict(s, obs=s["obs"][:2000]) for s in r["steps"]]} for r in D("p0")])
    out["v03_cut"] = {
        "archive_v03_P0": {"n_traces": len(a), "n_steps": len(La), "steps_at_2000_cap": int(sum(x == 2000 for x in La)),
                           "share_steps_at_cap": r4(np.mean(np.array(La) == 2000)), "max_step_obs": int(max(La)),
                           "traces_with_a_capped_step": int(sum(any(len(s["obs"]) == 2000 for s in r["steps"]) for r in D("p0_v03")))},
        "corrected_P0": {"n_steps": len(Lb), "steps_longer_than_2000": int(sum(x > 2000 for x in Lb)),
                         "share_steps_longer_than_2000": r4(np.mean(np.array(Lb) > 2000)), "max_step_obs": int(max(Lb)),
                         "traces_with_a_step_longer_than_2000": int(sum(any(len(s["obs"]) > 2000 for s in r["steps"]) for r in D("p0"))),
                         "traces_with_a_step_longer_than_2000_by_class": dict(collections.Counter(r["category"] for r in D("p0") if any(len(s["obs"]) > 2000 for s in r["steps"])))},
        "corrected_failed": {"n_steps": len(Lf), "steps_longer_than_2000": int(sum(x > 2000 for x in Lf)),
                             "traces_with_a_step_longer_than_2000": int(sum(any(len(s["obs"]) > 2000 for s in r["steps"]) for r in D("failed"))),
                             "note": "v0.3 did not extract unsuccessful attacks"},
        "step_by_step_comparison": {"traces_with_same_step_count": same_steps,
                                    "steps_differing_only_by_the_cut": pure_trunc,
                                    "steps_differing_otherwise_(pairing)": other_diff,
                                    "capped_v03_steps_whose_corrected_obs_is_longer_than_2000": trunc_pos_cap,
                                    "capped_v03_steps_whose_corrected_obs_is_at_most_2000_(mispaired)": cap_not_long,
                                    "traces_whose_only_difference_is_the_cut": tr_only_traces},
        "feature_effect_of_the_cut_on_corrected_P0": {
            "traces_whose_max_obs_changes": int(np.sum(Xc[:, 8] != Xp[:, 8])),
            "traces_whose_mean_obs_changes": int(np.sum(Xc[:, 9] != Xp[:, 9])),
            "share_P0_traces_affected": r4(np.mean(Xc[:, 9] != Xp[:, 9]))}}
    return out


def _beh_annot(recs, lab):
    X = beh(recs)
    for i, r in enumerate(recs):
        n = len(r["steps"]); c = sum(lab[s["tool"]][0] for s in r["steps"])
        X[i, 4] = c; X[i, 5] = c / n if n else 0
    return X


def part_annot_lrb():
    """m8 sensitivity: LR-B with the annotation-based state-changing count and fraction (features 4, 5) in place of
    the keyword ones; R1, R3 (5 folds), R4. Same recipe as the cache; LR-B refitted and checked against the cache."""
    from tool_state_labels import AGENTDRIFT as LD, AGENTDOJO as LJ
    lab = dict(LD); lab.update({t: v for t, v in LJ.items()})
    # AgentDrift and AgentDojo tool names do not collide except for names that mean the same thing; check
    clash = {t for t in LD if t in LJ and LD[t][0] != LJ[t][0]}
    assert not clash, clash
    out = {"shared_names_between_benchmarks": sorted(set(LD) & set(LJ))}
    settings = ["R1"] + [f"R3_{d}" for d in DOMAINS] + ["R4"]
    for k, s in enumerate(settings):
        f = os.path.join(PARTS, f"annot_lrb_{s}.json")
        if os.path.exists(f):
            out[s] = json.load(open(f)); continue
        tr, te = single_split(s); c = lin(s); assert check_ids(te, c)
        ytr = ylab(tr)
        Xk_tr, Xk_te = beh(tr), beh(te)
        Xa_tr, Xa_te = _beh_annot(tr, lab), _beh_annot(te, lab)
        mk = lr_fit(Xk_tr, ytr); sk = lr_score(mk, Xk_te)
        ma = lr_fit(Xa_tr, ytr); sa = lr_score(ma, Xa_te)
        chk = {"LR-B_refit_max_abs_diff_vs_cache": float(np.max(np.abs(sk - c["scores"]["LR-B"]))),
               "LR-B_refit_t_r80_minus_cache": float(mk["t80"] - c["thr"]["LR-B"]["t_r80"])}
        meth = {"LR-B": {"score": c["scores"]["LR-B"], "pred": {"t_r80": c["pred"]["LR-B"]["t_r80"]}},
                "LR-B-annot": {"score": sa, "pred": {"t_r80": sa >= ma["t80"]}},
                "LR-B-minus": {"score": c["scores"]["LR-B-minus"], "pred": {"t_r80": c["pred"]["LR-B-minus"]["t_r80"]}}}
        y = c["y"]; pool = np.where(c["pool_E"])[0]
        con = [("LR-B-annot", "LR-B"), ("LR-B-minus", "LR-B")]
        mets = ["AUROC", "AUPRC", "wAUPRC@0.01", "wAUPRC@0.1", "recall@t_r80", "FPR@t_r80"]
        res = {"checks": chk,
               "coef_LR-B": dict(zip(FEAT, [r4(x) for x in mk["lr"].coef_[0]])),
               "coef_LR-B-annot": dict(zip(FEAT, [r4(x) for x in ma["lr"].coef_[0]])),
               "train_mean_state_changing_fraction": {"keyword": r4(Xk_tr[:, 5].mean()), "annotation": r4(Xa_tr[:, 5].mean())},
               "test_mean_state_changing_fraction_poolE": {"keyword": r4(Xk_te[pool, 5].mean()), "annotation": r4(Xa_te[pool, 5].mean())}}
        if s == "R4":
            cl = [m["task"] for m in c["test_meta"]]
            res["primary"] = evaluate(meth, y, pool, mets, 9200 + k, clusters=cl, contrasts=con,
                                      label="R4 P0, suite|user_task clusters")
            res["stratified_sensitivity"] = evaluate(meth, y, pool, ["AUROC", "AUPRC"], 9300 + k, contrasts=con,
                                                     label="R4 P0, stratified")
        else:
            res["primary"] = evaluate(meth, y, pool, mets, 9200 + k, contrasts=con, label=f"{s} pool E")
        json.dump(res, open(f, "w"), indent=1); out[s] = res
        log(f"annot_lrb {s} done")
    hol = {}
    for cn in ["LR-B-annot minus LR-B", "LR-B-minus minus LR-B"]:
        hol[cn] = {}
        for mt in ["AUROC", "AUPRC", "wAUPRC@0.01", "wAUPRC@0.1", "recall@t_r80", "FPR@t_r80"]:
            ps = [out[f"R3_{d}"]["primary"]["contrasts"][cn][mt]["p_boot"] for d in DOMAINS]; adj = holm(ps)
            hol[cn][mt] = {d: {"p_holm": r4(adj[i]), "detected_holm": bool(adj[i] < 0.05)} for i, d in enumerate(DOMAINS)}
            hol[cn][mt]["n_folds_detected_holm"] = int(sum(adj < 0.05))
    out["R3_holm"] = hol
    return out


PART_FUNCS = {"annotation": part_annotation, "coefs": part_coefs, "univariate": part_univariate, "noobs": part_noobs,
              "pad_r1_linear": part_pad_r1_linear, "pad_r1_gru": part_pad_r1_gru, "pad_wut": part_pad_wut,
              "truncation": part_truncation, "annot_lrb": part_annot_lrb}


def pj(p):
    return os.path.join(PARTS, f"{p}.json")


def run_part(p):
    t0 = time.time(); log(f"part {p} start")
    res = PART_FUNCS[p]()
    if res is None:
        return False
    res["_runtime_s"] = round(time.time() - t0, 1)
    res["_script_sha256"] = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()
    json.dump(res, open(pj(p), "w"), indent=1, default=float)
    log(f"part {p} done in {res['_runtime_s']} s")
    return True


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


DEVIATIONS = [
    "F1. Bootstrap for AgentDojo quantities. Plan 3.5 prescribes suite|user_task cluster resampling for the within-"
    "AgentDojo UT analyses; it is used here as the primary interval for every AgentDojo P0 quantity (WUT, and also R4 "
    "transfer and the univariate AUROCs on P0, which use the same rows), with the stratified trajectory bootstrap as a "
    "sensitivity. AgentDrift R1/R3 use the stratified trajectory bootstrap (plan 3.5), with a world-cluster sensitivity "
    "for the observation-length ablations.",
    "F2. Tests and multiplicity. Two-sided percentile-bootstrap p-values, p = min(1, 2 min((1 + #{d <= 0})/(B + 1), "
    "(1 + #{d >= 0})/(B + 1))), so the smallest attainable value with B = 2000 is 0.001. Holm is applied across the five "
    "R3 folds separately for each contrast and metric, and, for within-minus-transfer, across the five detectors of that "
    "table only (plan family d restricted to the detectors computed here).",
    "F3. Added detector LR-obsonly (logistic regression on max_obs and mean_obs only, cache recipe). It is not in the "
    "plan's detector list; it reproduces the review's 'observation features alone' numbers and is reported as post hoc.",
    "F4. Coefficient intervals are not specified by the plan. R1: 500 refits on stratified resamples of data/train.jsonl "
    "(vocabulary fixed). WUT: mean, minimum and maximum over the 25 fold models, plus a descriptive model fitted on all "
    "P0 rows with 500 suite|user_task cluster-bootstrap refits.",
    "F5. Padding specifics the plan leaves open: the filler is the sentence ' The quick brown fox jumps over the lazy "
    "dog.' repeated and cut to exactly n characters, appended to the observation string; only positives are padded, "
    "negatives and training data are unchanged; recall uses the unchanged training threshold t_r80. In AgentDojo "
    "(WUT) the injected step(s) are located from the raw run's `injections` field; 9 of 2,294 compromised P0 traces "
    "have no located step and stay unpadded (results are also given on the located subset). Only lengthening is tested; "
    "in AgentDojo longer observations raise the attack score, so the evasive direction there would be shortening, "
    "which was not tested. The GRU is padded in R1 only (the task asked for WUT LR-C).",
    "F6. The three R1 GRU seeds were retrained for the padding test with gru_detector.train_gru and "
    "gru_ladder.stratified_val (as the cache). Thresholds and t_r80 predictions equal the cache exactly and pool-E AUROC "
    "is identical, but 1-2 of 1,881 scores per seed differ from the cache by at most 6e-8 (float32 rounding; not "
    "bitwise identical as the GRU cache README reports for its own refit). All padding levels use the refitted models.",
    "F7. Prevalence-weighted AUPRC (plan 3.1) is reported at pi = 0.01, 0.05 and 0.10 for the ablation tables; the plan "
    "does not fix the grid for this analysis. The t05 operating point is not reported here.",
    "F8. The state-change annotation (tool_state_annotation.json) was made by a single annotator (an AI assistant working "
    "for the authors) from tool names, example arguments and example observations. It was not blind: the keyword list was "
    "known and the listing also showed each tool's call count and step-label counts (how often it is a hijacked step or "
    "occurs in attacked runs). 13 AgentDrift labels and 1 AgentDojo label are marked ambiguous. No second rater, so no "
    "agreement statistic.",
    "F9. Bootstrap seeds aligned across parts (second run of this script, 2026-09-27). In the first run each part drew its "
    "own bootstrap resamples, so the same quantity (for example R4 LR-B AUROC) had slightly different intervals in the "
    "ablation, padding and annotation tables. The padding (R1 linear, R1 GRU, WUT) and annotation-sensitivity parts now use "
    "the resampling seed of the matching setting in the ablation part (9200 + setting index; 9300 + index for the "
    "sensitivity bootstrap), so identical quantities have identical intervals. Point estimates and fitted models are "
    "unchanged; only interval endpoints moved, by at most a few thousandths.",
]


def assemble():
    need = list(PART_FUNCS)
    miss = [p for p in need if not os.path.exists(pj(p))]
    assert not miss, f"missing parts {miss}"
    P = {p: json.load(open(pj(p))) for p in need}
    # enrich the LR-B-annot R3 Holm table with the per-fold deltas
    A = P["annot_lrb"]
    for cn, per in A["R3_holm"].items():
        for mt, t in per.items():
            for d in DOMAINS:
                c = A[f"R3_{d}"]["primary"]["contrasts"][cn][mt]
                t[d].update({"delta": c["delta"], "lo": c["lo"], "hi": c["hi"], "p_boot": c["p_boot"]})
    tc = json.load(open(os.path.join(CACHE, "linear", "r4_truncation_check.json")))
    an = json.load(open(ANNOT_JSON, encoding="utf-8"))
    P["annotation"]["annotation_summary"] = {"file": ANNOT_JSON, "sha256": sha(ANNOT_JSON),
                                             "agentdrift_summary": an["agentdrift_summary"],
                                             "agentdojo_summary": an["agentdojo_summary"]}
    res = {
        "meta": {"script": os.path.abspath(__file__), "script_sha256": sha(os.path.abspath(__file__)),
                 "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                 "review_comments": ["M5", "m8"], "plan": "REVISION_ANALYSIS_PLAN.md Sec. 4 (Observation length; Keyword feature)",
                 "plan_sha256": sha(os.path.join(ROOT, "REVISION_ANALYSIS_PLAN.md")),
                 "tool_state_labels_sha256": sha(os.path.join(HERE, "tool_state_labels.py")),
                 "parts_script_sha256": sorted({P[p].get("_script_sha256") for p in need}),
                 "inputs": {"agentdojo_traces.jsonl": sha(os.path.join(ROOT, "agentdojo_traces.jsonl")),
                            "agentdojo_failed.jsonl": sha(os.path.join(ROOT, "agentdojo_failed.jsonl")),
                            "linear_cache_script_sha256": lin("R1")["provenance"]["script_sha256"]},
                 "conventions": {"interval": "point estimate [2.5th, 97.5th bootstrap percentile], B = 2000 unless stated",
                                 "point": "metric on the full population; mean over repetitions/seeds for CV settings and the GRU",
                                 "AgentDrift_bootstrap": "stratified trajectory-level (plan 3.5); world clusters as sensitivity",
                                 "AgentDojo_bootstrap": "suite|user_task clusters (primary); stratified as sensitivity",
                                 "paired": "all methods of one table share the resampled rows and the drawn repetition/seed",
                                 "pool_E": "AgentDrift benign + hard_negative + attacked; AgentDojo P0 (benign + compromised)",
                                 "wAUPRC@pi": "prevalence-weighted average precision (plan 3.1), positives weighted pi/(1-pi) n_neg/n_pos",
                                 "thresholds": "t_r80 from training data only (20th percentile of training-positive scores)",
                                 "features": dict(enumerate(FEAT)), "filler_sentence": SENT, "pad_levels": PADS},
                 "parts_runtime_s": {p: P[p].get("_runtime_s") for p in need}},
        "M5_1_coefficients": P["coefs"],
        "M5_1_univariate": P["univariate"],
        "M5_2_without_observation_length": P["noobs"],
        "M5_3_padding": {"R1_linear": P["pad_r1_linear"], "R1_GRU": P["pad_r1_gru"], "WUT": P["pad_wut"]},
        "M5_4_truncation": P["truncation"],
        "m8_keyword_flag": P["annotation"],
        "m8_LRB_annotation_features": A,
        "cross_references": {"linear_cache_r4_truncation_check": {
            "source": "revision/cache/linear/r4_truncation_check.json",
            "withinUT_LR-C_untruncated_AUROC": tc["untruncated_corrected"]["withinUT_LR-C"]["AUROC"],
            "withinUT_LR-C_cut_at_2000_AUROC": tc["truncated_2000"]["withinUT_LR-C"]["AUROC"],
            "withinUT_LR-B_untruncated_AUROC": tc["untruncated_corrected"]["withinUT_LR-B"]["AUROC"],
            "withinUT_LR-B_cut_at_2000_AUROC": tc["truncated_2000"]["withinUT_LR-B"]["AUROC"],
            "transfer_LR-C_untruncated_AUROC": tc["untruncated_corrected"]["transfer_LR-C"]["AUROC"],
            "transfer_LR-C_cut_at_2000_AUROC": tc["truncated_2000"]["transfer_LR-C"]["AUROC"]},
            "review_reported_values": {
                "source": "REVIEW_REPORT.md, comment M5 (Evidence) and m8; quoted for comparison only",
                "share_P0_max_obs_above_agentdrift_max_pct": 48.8, "share_P0_mean_obs_above_agentdrift_max_pct": 48.2,
                "share_steps_at_2000_cap_pct": 3.2, "R1_LRB_mean_obs_coef": -1.86, "R1_LRB_len_coef": -1.38,
                "R1_LRB_max_obs_coef": 0.51, "R1_LRC_mean_obs_coef": -2.01, "univariate_mean_obs_AUROC_R1": 0.257,
                "univariate_mean_obs_AUROC_AgentDojo_P0_v03": 0.595,
                "padding_LRC_recall": [0.771, 0.631, 0.357, 0.170, 0.032], "padding_LRC_AUROC": [0.879, 0.809, 0.631, 0.381, 0.132],
                "padding_LRB_recall": [0.795, 0.663, 0.446, 0.274, 0.161], "padding_within_AgentDojo_LRC_recall": [0.779, 0.848, 0.924],
                "hijacked_steps_keyword_flagged_pct": 36.5, "agentdojo_compromised_state_changes_missed": 815,
                "R3_web_B_minus_C_without_obs": [0.003, -0.020, 0.024]}},
    }
    ck = {"R1_LR_coefficients_refit_equal_cache": P["coefs"]["R1_refit_equals_cache_max_abs_coef_diff"],
          "WUT_fold_models_equal_cache": P["coefs"]["WUT_refit_check"],
          "padding_R1_linear_scores_equal_cache": P["pad_r1_linear"]["checks"],
          "padding_R1_GRU_refits_vs_cache": P["pad_r1_gru"]["checks"],
          "padding_WUT_fold_models_equal_cache": P["pad_wut"]["checks"],
          "LR-B_refit_equal_cache_in_annotation_sensitivity": {s: A[s]["checks"] for s in ["R1"] + [f"R3_{d}" for d in DOMAINS] + ["R4"]},
          "fast_ap_equals_sklearn": self_test()}
    res["checks"] = ck
    res["deviations"] = DEVIATIONS
    json.dump(res, open(OUT_JSON, "w", encoding="utf-8"), indent=1, default=float)
    print("wrote", OUT_JSON)


# ---------------------------------------------------------------------------- summary markdown
def summary():
    R = json.load(open(OUT_JSON, encoding="utf-8"))

    def G(path):
        v = R
        for k in path.split("/"):
            v = v[k]
        return v

    def f3(x):
        return "n/a" if x is None else f"{x:.3f}"

    def F(path):
        v = G(path)
        return f"{f3(v[0])} [{f3(v[1])}, {f3(v[2])}]"

    def S(x):
        return "n/a" if x is None else f"{x:+.3f}"

    def Dl(path, holm=False):
        v = G(path)
        s = f"{S(v['delta'])} [{S(v['lo'])}, {S(v['hi'])}]"
        if holm and "p_holm" in v:
            s += f", p_Holm {v['p_holm']:.3f}"
        return s

    def pc(path):
        v = G(path)
        return f"{100 * v[0]:.1f}% [{100 * v[1]:.1f}, {100 * v[2]:.1f}]"

    L = []
    w = L.append
    w("# Features, observation length and the keyword flag (review M5 and m8)")
    w("")
    w(f"Script `revision/features_obslen.py` (SHA-256 `{R['meta']['script_sha256'][:16]}...`); results "
      f"`revision/features_obslen_results.json` (generated {R['meta']['created']}); annotation `revision/tool_state_annotation.json`. "
      "Every number below is read from the results JSON; the key path is given in brackets (`/` separates keys). Values are "
      "rounded to 3 decimals from the 4-decimal JSON values.")
    w("")
    w("Conventions. Point estimate [95% bootstrap interval], B = 2000. AgentDrift (R1, R3): stratified trajectory bootstrap. "
      "AgentDojo (R4 transfer, WUT, P0 univariate): cluster bootstrap over the 117 suite|user_task groups (primary); "
      "stratified results are in the JSON as a sensitivity. 'Detected' means the 95% interval excludes zero (or 0.5 for a "
      "single AUROC). Holm = Holm-adjusted bootstrap p-value across the five R3 folds. Thresholds (t_r80) always come from "
      "training data. Pool E = benign + hard negatives + attacked (AgentDrift), P0 (AgentDojo). Deviations F1-F9 are at the end.")
    w("")
    # ---------------- M5.1
    w("## M5.1 Per-feature attributions: coefficients, univariate AUROC, range overlap")
    w("")
    w("**Question.** Which behavioural features carry LR-B and the behavioural block of LR-C, is their association with the "
      "label the same in AgentDrift and AgentDojo, and do the AgentDojo values fall inside the AgentDrift range?")
    w("")
    w("Standardized coefficients (features z-scored on the training set). R1: cached fit with 500 stratified bootstrap refits "
      "[`M5_1_coefficients/R1_LR-B_bootstrap/<feature>`, `M5_1_coefficients/R1_LR-C_bootstrap/<feature>`]. WUT: mean "
      "(minimum, maximum) over the 25 fold models [`M5_1_coefficients/WUT_<det>_25fold/<feature>`].")
    w("")
    w("| Feature | R1 LR-B | R1 LR-C (behavioural block) | WUT LR-B, mean (min, max) | WUT LR-C, mean (min, max) |")
    w("|---|---|---|---|---|")
    for f in FEAT:
        wb = G(f"M5_1_coefficients/WUT_LR-B_25fold/{f}"); wc = G(f"M5_1_coefficients/WUT_LR-C_25fold/{f}")
        w(f"| {f} | {F(f'M5_1_coefficients/R1_LR-B_bootstrap/{f}')} | {F(f'M5_1_coefficients/R1_LR-C_bootstrap/{f}')} | "
          f"{wb['mean']:+.3f} ({wb['min']:+.3f}, {wb['max']:+.3f}) | {wc['mean']:+.3f} ({wc['min']:+.3f}, {wc['max']:+.3f}) |")
    w("")
    w("Univariate AUROC of each feature on its own (score = feature value; above 0.5 means larger in attacked or compromised "
      "runs) [`M5_1_univariate/populations/<population>/metrics/<feature>/AUROC`].")
    w("")
    pops = ["R1_poolE"] + [f"R3_{d}_poolE" for d in DOMAINS] + ["AgentDojo_P0"]
    w("| Feature | " + " | ".join(["R1"] + [f"R3 {d}" for d in DOMAINS] + ["AgentDojo P0 (cluster)"]) + " |")
    w("|---|" + "---|" * len(pops))
    for f in FEAT:
        w(f"| {f} | " + " | ".join(F(f"M5_1_univariate/populations/{p}/metrics/{f}/AUROC") for p in pops) + " |")
    w("")
    rv = G("M5_1_univariate/reversals_vs_AgentDojo_P0")
    w("Detected direction reversals (interval above 0.5 in one benchmark, below 0.5 in the other) "
      "[`M5_1_univariate/reversals_vs_AgentDojo_P0/<population>/reversal_detected`]: " +
      "; ".join(f"{p.replace('_poolE', '')} vs AgentDojo P0: {', '.join(v['reversal_detected']) or 'none'}" for p, v in rv.items()) + ". "
      "Within AgentDrift, R1 vs R3 [`M5_1_univariate/reversals_R1_vs_R3`]: " +
      "; ".join(f"{d}: {', '.join(v) or 'none'}" for d, v in G("M5_1_univariate/reversals_R1_vs_R3").items()) + ". "
      f"For comparison with the review, mean_obs AUROC on the v0.3 AgentDojo extraction is "
      f"{f3(G('M5_1_univariate/v03_archive_P0_point_AUROC/mean_obs'))} [`M5_1_univariate/v03_archive_P0_point_AUROC/mean_obs`].")
    w("")
    ro = "M5_1_univariate/range_overlap"
    w("Range overlap with the R4 training data (all 10,656 AgentDrift trajectories) [`" + ro + "/<feature>`]: "
      f"max_obs AgentDrift min/median/max {G(ro + '/max_obs/agentdrift_min_median_max')}, AgentDojo P0 "
      f"{G(ro + '/max_obs/agentdojo_P0_min_median_max')}, share of P0 traces above the AgentDrift maximum "
      f"{f3(G(ro + '/max_obs/agentdojo_P0_share_above_agentdrift_max'))}; mean_obs AgentDrift "
      f"{G(ro + '/mean_obs/agentdrift_min_median_max')}, P0 {G(ro + '/mean_obs/agentdojo_P0_min_median_max')}, share above "
      f"{f3(G(ro + '/mean_obs/agentdojo_P0_share_above_agentdrift_max'))}. Under the R4 scaler, the share of P0 traces with "
      f"|z| > 3 is {f3(G(ro + '/max_obs/agentdojo_P0_share_abs_z_gt_3_under_R4_scaler'))} for max_obs and "
      f"{f3(G(ro + '/mean_obs/agentdojo_P0_share_abs_z_gt_3_under_R4_scaler'))} for mean_obs (largest |z| "
      f"{G(ro + '/max_obs/agentdojo_P0_max_abs_z_under_R4_scaler'):.0f} and {G(ro + '/mean_obs/agentdojo_P0_max_abs_z_under_R4_scaler'):.0f}); "
      f"for the other eight features it is at most {max(G(ro + '/' + f + '/agentdojo_P0_share_abs_z_gt_3_under_R4_scaler') for f in FEAT[:8]):.3f}.")
    w("")
    med = "M5_4_truncation/agentdrift_train_step_obs_length"
    w("Median observation length per step in AgentDrift training data [`" + med + "/<group>/p50`]: benign steps of benign "
      f"trajectories {G(med + '/benign_steps_of_benign/p50'):.0f}, benign steps of attacked trajectories "
      f"{G(med + '/benign_steps_of_attacked/p50'):.0f}, hijacked steps {G(med + '/hijacked_steps_of_attacked/p50'):.0f}, "
      f"injection points {G(med + '/injection_point_steps_of_attacked/p50'):.0f}, failed injections "
      f"{G(med + '/failed_injection_steps_of_failed_attack/p50'):.0f}, benign steps of hard negatives "
      f"{G(med + '/benign_steps_of_hard_negative/p50'):.0f} characters.")
    w("")
    c1 = "M5_1_coefficients"
    r1b = G(c1 + "/R1_LR-B_bootstrap"); r1c = G(c1 + "/R1_LR-C_bootstrap")
    assert max(FEAT, key=lambda f: abs(r1b[f][0])) == "mean_obs" and max(FEAT, key=lambda f: abs(r1c[f][0])) == "mean_obs"
    assert all(G(f"M5_1_univariate/populations/{p}/metrics/mean_obs/AUROC")[2] < 0.5 for p in pops[:-1])
    assert G("M5_1_univariate/populations/AgentDojo_P0/metrics/mean_obs/AUROC")[1] > 0.5
    other_rev = [f for f in G("M5_1_univariate/reversals_vs_AgentDojo_P0/R1_poolE/reversal_detected") if f != "mean_obs"]
    wmo = G(c1 + "/WUT_LR-B_25fold/max_obs")
    all25 = wmo["share_same_sign_as_mean"] == 1.0 and wmo["mean"] > 0
    w("**Reading.** mean_obs has the largest standardized weight in R1 LR-B "
      f"({F(c1 + '/R1_LR-B_bootstrap/mean_obs')}) and in LR-C's behavioural block ({F(c1 + '/R1_LR-C_bootstrap/mean_obs')}). "
      "On its own its AUROC is below 0.5 in every AgentDrift population, that is, attacked trajectories "
      f"have shorter mean observations (AUROC {F('M5_1_univariate/populations/R1_poolE/metrics/mean_obs/AUROC')} in R1 and "
      f"{min(G(f'M5_1_univariate/populations/R3_{d}_poolE/metrics/mean_obs/AUROC')[0] for d in DOMAINS):.3f} to "
      f"{max(G(f'M5_1_univariate/populations/R3_{d}_poolE/metrics/mean_obs/AUROC')[0] for d in DOMAINS):.3f} in the five "
      "held-out domains). The shortening is not confined to the attack steps: the benign-labelled steps of attacked "
      "trajectories are also shorter than the steps of benign trajectories (medians above), which points to a property of "
      "the generator's attacked category rather than to the injection itself. In AgentDojo P0 the direction flips "
      f"(mean_obs {F('M5_1_univariate/populations/AgentDojo_P0/metrics/mean_obs/AUROC')}); "
      f"{' and '.join(other_rev) or 'no other feature'} also reverse{'s' if len(other_rev) == 1 else ''} against R1. AgentDojo "
      "observation lengths also lie far outside the AgentDrift range, so a transferred linear model extrapolates on exactly "
      "these two features. Within AgentDojo the fitted models still give mean_obs a negative "
      f"weight (WUT LR-B mean {G(c1 + '/WUT_LR-B_25fold/mean_obs/mean'):+.3f}) but a larger positive weight to max_obs "
      f"({G(c1 + '/WUT_LR-B_25fold/max_obs/mean'):+.3f}{', positive in all 25 fold models' if all25 else ''}), so the sign of mean_obs in a "
      "multivariate model is not by itself the reversal. The numbers do not show which benchmark difference causes the "
      "reversal, and the univariate AUROCs say nothing about how the features combine.")
    w("")
    # ---------------- M5.2
    w("## M5.2 R3, R4 and within-AgentDojo without the observation-length features")
    w("")
    w("**Question.** How much of LR-B and LR-C's performance under zero tool coverage (R3), under transfer (R4) and within "
      "AgentDojo (WUT) depends on max_obs and mean_obs, and does the R3 advantage of LR-B over LR-C survive without them?")
    w("")
    w("AUROC on pool E [`M5_2_without_observation_length/<setting>/primary/metrics/<detector>/AUROC`]. LR-obsonly = "
      "logistic regression on max_obs and mean_obs only (post hoc, F3).")
    w("")
    dets = ["LR-B", "LR-B-noobs", "LR-C", "LR-C-noobs", "LR-obsonly"]
    w("| Setting | " + " | ".join(dets) + " |")
    w("|---|" + "---|" * len(dets))
    for s in NOOBS_SETTINGS:
        w(f"| {s} | " + " | ".join(F(f"M5_2_without_observation_length/{s}/primary/metrics/{d}/AUROC") for d in dets) + " |")
    w("")
    w("Prevalence-weighted AUPRC at pi = 0.10 [`.../metrics/<detector>/wAUPRC@0.1`]:")
    w("")
    w("| Setting | " + " | ".join(dets) + " |")
    w("|---|" + "---|" * len(dets))
    for s in NOOBS_SETTINGS:
        w(f"| {s} | " + " | ".join(F(f"M5_2_without_observation_length/{s}/primary/metrics/{d}/wAUPRC@0.1") for d in dets) + " |")
    w("")
    w("Recall and FPR at t_r80 [`.../metrics/<detector>/recall@t_r80`, `FPR@t_r80`]:")
    w("")
    w("| Setting | LR-B-noobs recall | LR-B-noobs FPR | LR-C-noobs recall | LR-C-noobs FPR |")
    w("|---|---|---|---|---|")
    for s in NOOBS_SETTINGS:
        b = f"M5_2_without_observation_length/{s}/primary/metrics"
        w(f"| {s} | {F(b + '/LR-B-noobs/recall@t_r80')} | {F(b + '/LR-B-noobs/FPR@t_r80')} | {F(b + '/LR-C-noobs/recall@t_r80')} | {F(b + '/LR-C-noobs/FPR@t_r80')} |")
    w("")
    w("Paired AUROC differences per R3 fold with Holm across the five folds "
      "[`M5_2_without_observation_length/R3_holm/<contrast>/AUROC/<domain>`]:")
    w("")
    cons = ["LR-C-noobs minus LR-B-noobs", "LR-B minus LR-C", "LR-B-noobs minus LR-B", "LR-C-noobs minus LR-C", "LR-obsonly minus LR-B"]
    w("| Contrast | " + " | ".join(DOMAINS) + " | Folds detected after Holm |")
    w("|---|" + "---|" * (len(DOMAINS) + 1))
    for cn in cons:
        h = f"M5_2_without_observation_length/R3_holm/{cn}/AUROC"
        w(f"| {cn} | " + " | ".join(Dl(f"{h}/{d}", True) for d in DOMAINS) +
          f" | {G(h + '/n_folds_detected_holm')} of 5 |")
    w("")
    hw = "M5_2_without_observation_length/R3_holm/LR-C-noobs minus LR-B-noobs/wAUPRC@0.1"
    w(f"The same C-noobs minus B-noobs contrast on wAUPRC at pi = 0.10 is detected in {G(hw + '/n_folds_detected_holm')} of 5 "
      f"folds after Holm [`{hw}`]; with world-cluster resampling (sensitivity) the web fold AUROC contrast is "
      f"{Dl('M5_2_without_observation_length/R3_web/world_cluster_sensitivity/contrasts/LR-C-noobs minus LR-B-noobs/AUROC')} "
      "[`M5_2_without_observation_length/R3_web/world_cluster_sensitivity/contrasts/...`].")
    w("")
    p4 = "M5_2_without_observation_length/R4/primary/contrasts"
    pw = "M5_2_without_observation_length/WUT/primary/contrasts"
    w(f"R4 (transfer) paired AUROC: LR-B-noobs minus LR-B {Dl(p4 + '/LR-B-noobs minus LR-B/AUROC')}; LR-C-noobs minus LR-C "
      f"{Dl(p4 + '/LR-C-noobs minus LR-C/AUROC')}; LR-C-noobs minus LR-B-noobs {Dl(p4 + '/LR-C-noobs minus LR-B-noobs/AUROC')}. "
      f"WUT: LR-B-noobs minus LR-B {Dl(pw + '/LR-B-noobs minus LR-B/AUROC')}; LR-C-noobs minus LR-C "
      f"{Dl(pw + '/LR-C-noobs minus LR-C/AUROC')}. Within (WUT) minus transfer (R4) on the same P0 rows, Holm over five "
      "detectors [`M5_2_without_observation_length/within_minus_transfer/contrasts/WUT:<d> minus R4:<d>/AUROC`]: " +
      "; ".join(f"{d} {Dl(f'M5_2_without_observation_length/within_minus_transfer/contrasts/WUT:{d} minus R4:{d}/AUROC', True)}"
                for d in LIN4 + ["LR-obsonly"]) + ".")
    w("")
    h1 = "M5_2_without_observation_length/R3_holm/LR-B-noobs minus LR-B/AUROC"
    ho = "M5_2_without_observation_length/R3_holm/LR-obsonly minus LR-B/AUROC"
    hcb = "M5_2_without_observation_length/R3_holm/LR-C-noobs minus LR-B-noobs/AUROC"
    assert G(h1 + "/n_folds_detected_holm") == 5 and all(G(f"{h1}/{d}")["delta"] < 0 for d in DOMAINS)
    assert G("M5_2_without_observation_length/R3_holm/LR-B minus LR-C/AUROC/n_folds_detected_holm") == 5
    not_det_obs = [d for d in DOMAINS if not G(f"{ho}/{d}/detected_holm")]
    not_det_cb = [d for d in DOMAINS if not G(f"{hcb}/{d}/detected_holm")]
    assert all(G(f"{hcb}/{d}")["delta"] < 0 for d in DOMAINS if G(f"{hcb}/{d}/detected_holm"))
    wt = "M5_2_without_observation_length/within_minus_transfer/contrasts"
    assert all(G(f"{wt}/WUT:{d} minus R4:{d}/AUROC/detected_holm") for d in LIN4 + ["LR-obsonly"])
    assert not G(pw + "/LR-B-noobs minus LR-B/AUROC/detected") and not G(pw + "/LR-C-noobs minus LR-C/AUROC/detected")
    w("**Reading.** In every R3 fold, removing max_obs and mean_obs costs LR-B between "
      f"{min(-G(h1 + '/' + d)['delta'] for d in DOMAINS):.3f} and {max(-G(h1 + '/' + d)['delta'] for d in DOMAINS):.3f} AUROC "
      f"(all detected after Holm), and the two observation features alone come within {max(-G(ho + '/' + d)['delta'] for d in DOMAINS):.3f} "
      f"AUROC of full LR-B (no detected difference after Holm in {', '.join(not_det_obs) or 'no fold'}). So most of LR-B's "
      "zero-coverage signal in AgentDrift is the observation-length cue. With the features included, LR-B is above LR-C in all "
      "five folds after Holm; without them LR-C-noobs is below LR-B-noobs in "
      f"{G(hcb + '/n_folds_detected_holm')} of 5 folds after Holm, and the difference is not detected in "
      f"{', '.join(not_det_cb) or 'no fold'} ({'; '.join(Dl(hcb + '/' + d) for d in not_det_cb)}). "
      "On AUROC, an 'all five folds' statement therefore depends on the observation-length features; on the plan's primary "
      "metric (prevalence-weighted AUPRC at pi = 0.10) LR-C-noobs is below LR-B-noobs in "
      f"{G(hw + '/n_folds_detected_holm')} of 5 folds after Holm (web {Dl(hw + '/web', True)}), so the count depends on the "
      "metric. Without the observation features LR-C-noobs's training threshold does not hold under R3: at t_r80 it flags "
      f"{min(G(f'M5_2_without_observation_length/R3_{d}/primary/metrics/LR-C-noobs/FPR@t_r80')[0] for d in DOMAINS):.3f} to "
      f"{max(G(f'M5_2_without_observation_length/R3_{d}/primary/metrics/LR-C-noobs/FPR@t_r80')[0] for d in DOMAINS):.3f} of the "
      "held-out negatives. In R4, dropping the two features "
      f"raises transfer AUROC (LR-B {F('M5_2_without_observation_length/R4/primary/metrics/LR-B/AUROC')} to "
      f"{F('M5_2_without_observation_length/R4/primary/metrics/LR-B-noobs/AUROC')}; LR-C to "
      f"{F('M5_2_without_observation_length/R4/primary/metrics/LR-C-noobs/AUROC')}), consistent with removing a reversed, "
      "out-of-range cue (M5.1). It does not close the gap to within-AgentDojo performance: the within-minus-transfer difference is still detected "
      "after Holm for every detector. Within "
      "AgentDojo removing the two features costs nothing detectable (WUT LR-B-noobs minus LR-B "
      f"{Dl('M5_2_without_observation_length/WUT/primary/contrasts/LR-B-noobs minus LR-B/AUROC')}), although on their own they "
      f"carry signal there (LR-obsonly {F('M5_2_without_observation_length/WUT/primary/metrics/LR-obsonly/AUROC')}); the "
      "within-AgentDojo information in observation length is therefore redundant with the other features, not absent. The noobs subsets were chosen after the review had seen these results, so the R4 gain is a "
      "post hoc observation, not evidence that the remaining features transfer.")
    w("")
    # ---------------- M5.3
    w("## M5.3 Padding test (attacker-controlled observation length)")
    w("")
    w("**Question.** If the attacker lengthens the injected observation, how do recall at the training threshold t_r80 and "
      "AUROC change? We add n filler characters (a neutral repeated sentence) to the injection-point observation of every "
      "attacked R1 test trajectory (one step each; 831 trajectories) and rescore with the unchanged R1 models; negatives are "
      "not padded. We do not claim that padded injections would still succeed against an agent; the test only asks how the "
      "detectors' scores respond to a quantity the attacker writes.")
    w("")
    w("R1, recall at t_r80 and AUROC on pool E [`M5_3_padding/R1_linear/eval/metrics/<det>|pad<n>/<metric>`, "
      "`M5_3_padding/R1_GRU/eval/metrics/GRU|pad<n>/<metric>`; GRU = mean over seeds 0-2]:")
    w("")
    w("| Detector | Metric | " + " | ".join(f"+{n}" for n in PADS) + " |")
    w("|---|---|" + "---|" * len(PADS))
    for d in ["LR-B", "LR-C", "LR-B-noobs", "LR-C-noobs", "GRU"]:
        base = "M5_3_padding/R1_GRU/eval/metrics" if d == "GRU" else "M5_3_padding/R1_linear/eval/metrics"
        for mt in ["recall@t_r80", "AUROC"]:
            w(f"| {d} | {mt} | " + " | ".join(F(f"{base}/{d}|pad{n}/{mt}") for n in PADS) + " |")
    w("")
    w("Paired change from +0 to +100 and to +1,200 characters [`M5_3_padding/R1_*/eval/contrasts/<det>|pad<n> minus <det>|pad0/<metric>`]: " +
      "; ".join(f"{d}: recall {Dl(('M5_3_padding/R1_GRU' if d == 'GRU' else 'M5_3_padding/R1_linear') + f'/eval/contrasts/{d}|pad100 minus {d}|pad0/recall@t_r80')} (+100), "
                f"{Dl(('M5_3_padding/R1_GRU' if d == 'GRU' else 'M5_3_padding/R1_linear') + f'/eval/contrasts/{d}|pad1200 minus {d}|pad0/recall@t_r80')} (+1,200)"
                for d in ["LR-B", "LR-C", "GRU"]) +
      ". LR-B-noobs and LR-C-noobs do not read observation lengths, so their scores are identical at every level (by construction).")
    w("")
    gk = "M5_3_padding/R1_GRU/checks"
    w(f"GRU refit check [`{gk}/seed<k>`]: the three seeds were retrained with gru_detector.train_gru on gru_ladder.stratified_val; "
      "t_r80 thresholds and t_r80 predictions equal the cache and pool-E AUROC is identical; "
      f"{sum(G(gk + f'/seed{k}/n_rows_not_bitwise_equal') for k in range(3))} of the 5,643 scores (3 seeds x 1,881 rows) differ from the cache by at most "
      f"{max(G(gk + f'/seed{k}/max_abs_diff_vs_cache') for k in range(3)):.1e} (float32 rounding). Linear scores rebuilt from the "
      f"cached coefficients equal the cache to {max(G(f'M5_3_padding/R1_linear/checks/{d}/max_abs_diff_vs_cache') for d in LIN4):.1e}.")
    w("")
    lw = "M5_3_padding/WUT/location"
    w(f"Within AgentDojo (WUT, LR-C where feasible, plus LR-B and LR-B-noobs). The injected step was located in {G(lw + '/n_located')} of "
      f"{G(lw + '/n_compromised')} compromised P0 traces ({100 * G(lw + '/share_located'):.1f}%) by matching the raw run's `injections` "
      f"text in the observations [`{lw}`]; the other {G(lw + '/n_compromised') - G(lw + '/n_located')} stay unpadded. All located "
      "injected steps of a trace are padded. The 25 WUT fold models were refitted with the cache recipe and reproduce the cached "
      f"out-of-fold scores to {G('M5_3_padding/WUT/checks/LR-C/max_abs_score_diff_vs_cache'):.1e} with identical thresholds. "
      "Recall at t_r80 and AUROC on P0, cluster bootstrap [`M5_3_padding/WUT/eval/primary/metrics/<det>|pad<n>/<metric>`]:")
    w("")
    w("| Detector | Metric | " + " | ".join(f"+{n}" for n in PADS) + " |")
    w("|---|---|" + "---|" * len(PADS))
    for d in ["LR-C", "LR-B", "LR-B-noobs"]:
        for mt in ["recall@t_r80", "AUROC"]:
            w(f"| {d} | {mt} | " + " | ".join(F(f"M5_3_padding/WUT/eval/primary/metrics/{d}|pad{n}/{mt}") for n in PADS) + " |")
    w("")
    wc = "M5_3_padding/WUT/eval/primary/contrasts"
    w(f"Paired change +0 to +1,200: LR-C recall {Dl(wc + '/LR-C|pad1200 minus LR-C|pad0/recall@t_r80')}, AUROC "
      f"{Dl(wc + '/LR-C|pad1200 minus LR-C|pad0/AUROC')}; LR-B recall {Dl(wc + '/LR-B|pad1200 minus LR-B|pad0/recall@t_r80')}. "
      "Restricting to traces with a located injection gives the same picture [`M5_3_padding/WUT/eval/located_subset`].")
    w("")
    for d in ["LR-B", "LR-C"]:
        assert all(G(f"M5_3_padding/R1_linear/eval/contrasts/{d}|pad{n} minus {d}|pad0/recall@t_r80/detected") for n in PADS[1:])
    assert all(G(f"M5_3_padding/R1_GRU/eval/contrasts/GRU|pad{n} minus GRU|pad0/recall@t_r80/detected") for n in PADS[1:])
    assert G(wc + "/LR-C|pad1200 minus LR-C|pad0/recall@t_r80/delta") > 0
    w("**Reading.** In R1, every detector that reads observation lengths loses recall steeply as the injected observation "
      "grows, and the GRU degrades in the same way as the linear models: 100 extra characters (the median injection-point "
      f"observation is {G(med + '/injection_point_steps_of_attacked/p50'):.0f} characters) already cost LR-C "
      f"{-G('M5_3_padding/R1_linear/eval/contrasts/LR-C|pad100 minus LR-C|pad0/recall@t_r80/delta'):.3f} recall and the GRU "
      f"{-G('M5_3_padding/R1_GRU/eval/contrasts/GRU|pad100 minus GRU|pad0/recall@t_r80/delta'):.3f}, and at +1,200 the "
      f"GRU flags {f3(G('M5_3_padding/R1_GRU/eval/metrics/GRU|pad1200/recall@t_r80')[0])} of attacked trajectories with AUROC "
      f"{f3(G('M5_3_padding/R1_GRU/eval/metrics/GRU|pad1200/AUROC')[0])}, well below 0.5. The in-distribution scores of these "
      "detectors therefore depend on a quantity that an indirect-injection attacker writes (the attacker needs to know only the "
      "direction of the cue, not the detector's parameters). "
      "Within AgentDojo the direction is reversed: lengthening the injected observation makes LR-C slightly more likely to flag "
      "the trace, so padding is not an evasion there; the evasive direction would be shortening, which we did not test. The "
      "review reported a larger within-AgentDojo rise in LR-C recall ("
      f"{' to '.join(f'{x:.3f}' for x in G('cross_references/review_reported_values/padding_within_AgentDojo_LRC_recall'))}); "
      "that run used an earlier AgentDojo extraction and the review gives three values without mapping them to padding levels, "
      "so we do not reproduce its magnitude and have not diagnosed the difference. "
      "The test does not show that padded injections still succeed, that real attackers pad, or how a detector retrained on "
      "padded data would behave.")
    w("")
    # ---------------- M5.4
    w("## M5.4 Truncation of AgentDojo observations")
    w("")
    w("**Question.** How do observation lengths compare between the benchmarks, how many AgentDojo traces lie beyond the "
      "AgentDrift maximum, and which steps did the v0.3 cut at 2,000 characters affect?")
    w("")
    t = "M5_4_truncation"
    w(f"- AgentDrift trajectory max_obs ranges from {G(t + '/agentdrift_max_obs_range_pooled')[0]:.0f} to "
      f"{G(t + '/agentdrift_max_obs_range_pooled')[1]:.0f} characters in the pooled train+test data "
      f"[`{t}/agentdrift_max_obs_range_pooled`] ({G(t + '/agentdrift_max_obs_range_including_val')[1]:.0f} including data/val.jsonl); "
      f"the largest trajectory mean_obs is {G(t + '/agentdrift_mean_obs_max_pooled'):.0f}. Median per-trajectory max_obs: AgentDrift "
      f"{G(t + '/agentdrift_pooled_trajectory/all/max_obs/p50'):.0f}, AgentDojo P0 {G(t + '/agentdojo_P0_trajectory/all/max_obs/p50'):.0f} "
      f"(benign {G(t + '/agentdojo_P0_trajectory/benign/max_obs/p50'):.0f}, compromised {G(t + '/agentdojo_P0_trajectory/attacked/max_obs/p50'):.0f}), "
      f"unsuccessful attacks {G(t + '/agentdojo_failed_trajectory/all/max_obs/p50'):.0f}; per-step 99th percentile AgentDrift "
      f"{G(t + '/agentdrift_pooled_step_obs_length/all/p99'):.0f}, AgentDojo P0 {G(t + '/agentdojo_P0_step_obs_length/all/p99'):.0f}, maximum "
      f"{G(t + '/agentdojo_P0_step_obs_length/all/max'):.0f} [`{t}/*_trajectory`, `{t}/*_step_obs_length`].")
    sh = f"{t}/agentdojo_share_beyond_agentdrift"
    w(f"- Share of AgentDojo P0 traces whose max_obs exceeds the AgentDrift maximum: {pc(sh + '/max_obs_above_agentdrift_max_obs/all_P0')} "
      f"(benign {pc(sh + '/max_obs_above_agentdrift_max_obs/benign')}, compromised {pc(sh + '/max_obs_above_agentdrift_max_obs/compromised')}, "
      f"unsuccessful attacks {100 * G(sh + '/max_obs_above_agentdrift_max_obs/failed_attacks'):.1f}%). Share whose mean_obs exceeds the largest "
      f"AgentDrift mean_obs: {pc(sh + '/mean_obs_above_agentdrift_max_mean_obs/all_P0')}; whose mean_obs exceeds the AgentDrift maximum "
      f"max_obs: {pc(sh + '/mean_obs_above_agentdrift_max_obs/all_P0')} [`{sh}`]. On the v0.3 extraction the max_obs share is "
      f"{100 * G(sh + '/v03_archive_extraction_for_comparison_with_review/max_obs_above_agentdrift_max_obs'):.1f}% and the "
      f"mean_obs-above-largest-mean share {100 * G(sh + '/v03_archive_extraction_for_comparison_with_review/mean_obs_above_agentdrift_max_mean_obs'):.1f}%, "
      f"which reproduces the review's {G('cross_references/review_reported_values/share_P0_max_obs_above_agentdrift_max_pct')}% and "
      f"{G('cross_references/review_reported_values/share_P0_mean_obs_above_agentdrift_max_pct')}% [`cross_references/review_reported_values`].")
    v = f"{t}/v03_cut"
    w(f"- v0.3 cut: {G(v + '/archive_v03_P0/steps_at_2000_cap')} of {G(v + '/archive_v03_P0/n_steps')} P0 steps "
      f"({100 * G(v + '/archive_v03_P0/share_steps_at_cap'):.2f}%) sit exactly at 2,000 characters, in "
      f"{G(v + '/archive_v03_P0/traces_with_a_capped_step')} traces. In the corrected extraction "
      f"{G(v + '/corrected_P0/steps_longer_than_2000')} P0 steps ({100 * G(v + '/corrected_P0/share_steps_longer_than_2000'):.2f}%) in "
      f"{G(v + '/corrected_P0/traces_with_a_step_longer_than_2000')} traces "
      f"({G(v + '/corrected_P0/traces_with_a_step_longer_than_2000_by_class/attacked')} compromised, "
      f"{G(v + '/corrected_P0/traces_with_a_step_longer_than_2000_by_class/benign')} benign) are longer than 2,000 characters and would be "
      f"cut, and {G(v + '/corrected_failed/steps_longer_than_2000')} steps in {G(v + '/corrected_failed/traces_with_a_step_longer_than_2000')} "
      "unsuccessful-attack traces (not extracted in v0.3). Step by step against the v0.3 file (same step counts in all 2,686 traces): "
      f"{G(v + '/step_by_step_comparison/steps_differing_only_by_the_cut')} steps differ only by the cut and "
      f"{G(v + '/step_by_step_comparison/steps_differing_otherwise_(pairing)')} differ otherwise (the pairing fix); "
      f"{G(v + '/step_by_step_comparison/capped_v03_steps_whose_corrected_obs_is_longer_than_2000')} of the capped v0.3 steps have a "
      f"corrected observation longer than 2,000 and {G(v + '/step_by_step_comparison/capped_v03_steps_whose_corrected_obs_is_at_most_2000_(mispaired)')} "
      f"were capped only because a mispaired observation was long; in {G(v + '/step_by_step_comparison/traces_whose_only_difference_is_the_cut')} "
      f"traces the cut is the only difference. Re-applying the cut to the corrected P0 changes max_obs and mean_obs of "
      f"{G(v + '/feature_effect_of_the_cut_on_corrected_P0/traces_whose_mean_obs_changes')} traces [`{v}`].")
    x = "cross_references/linear_cache_r4_truncation_check"
    w(f"- Effect on results (copied from the linear cache check): within-AgentDojo LR-C AUROC {F(x + '/withinUT_LR-C_cut_at_2000_AUROC')} "
      f"with the cut vs {F(x + '/withinUT_LR-C_untruncated_AUROC')} without; LR-B {F(x + '/withinUT_LR-B_cut_at_2000_AUROC')} vs "
      f"{F(x + '/withinUT_LR-B_untruncated_AUROC')}; transfer LR-C {F(x + '/transfer_LR-C_cut_at_2000_AUROC')} vs "
      f"{F(x + '/transfer_LR-C_untruncated_AUROC')} [`{x}`; stratified bootstrap].")
    w("")
    xc = G(x + "/withinUT_LR-C_cut_at_2000_AUROC")[0] - G(x + "/withinUT_LR-C_untruncated_AUROC")[0]
    w(f"**Reading.** The v0.3 extractor did cut AgentDojo observations at 2,000 characters: "
      f"{100 * G(v + '/archive_v03_P0/share_steps_at_cap'):.1f}% of P0 steps, in {G(v + '/archive_v03_P0/traces_with_a_capped_step')} "
      f"of {G(v + '/archive_v03_P0/n_traces')} traces, and {G(v + '/corrected_P0/traces_with_a_step_longer_than_2000')} corrected P0 "
      f"traces ({100 * G(v + '/feature_effect_of_the_cut_on_corrected_P0/share_P0_traces_affected'):.1f}%) contain a longer "
      "observation. The manuscript should disclose this and say that the revision stores observations uncut in both benchmarks "
      "(harmonized: neither is cut). The cut is not what makes the benchmarks incompatible: even with the cut, "
      f"{100 * G(sh + '/v03_archive_extraction_for_comparison_with_review/max_obs_above_agentdrift_max_obs'):.1f}% of AgentDojo traces "
      "lay beyond the largest AgentDrift observation, and removing the cut widens the range gap. Removing the cut leaves transfer "
      f"near 0.5 and lowers within-AgentDojo LR-C AUROC by {xc:.3f}. The numbers do not say which cut, if any, a deployed "
      "detector should use.")
    w("")
    # ---------------- m8
    w("## m8 Keyword-based state-changing features")
    w("")
    w("**Question.** How well does the keyword flag in featurise.py (a tool name containing send, delete, update, create, "
      "transfer, add, download, post, write, cancel, apply, convert, share or grant) identify state-changing tools, how many "
      "hijacked or compromised calls does it miss, and does LR-B change if the two write features use a state-change annotation "
      "instead?")
    w("")
    a = "m8_keyword_flag"
    asd = G(a + "/annotation_summary/agentdrift_summary"); asj = G(a + "/annotation_summary/agentdojo_summary")
    w(f"The annotation labels all {asd['n_tools']} AgentDrift tool names ({asd['n_state_changing']} state-changing, "
      f"{asd['n_ambiguous']} marked ambiguous) and all {asj['n_tools']} AgentDojo tool names ({asj['n_state_changing']} state-changing, "
      f"{asj['n_ambiguous']} ambiguous) [`{a}/annotation_summary`] (`revision/tool_state_annotation.json`, criterion in its header: "
      "persistent write, transmission to another party, or command execution). Keyword flag against the annotation "
      "[`m8_keyword_flag/keyword_vs_annotation_<benchmark>`]:")
    w("")
    w("| Benchmark | Tool level: precision (TP/(TP+FP)) | Tool level: recall (TP/(TP+FN)) | Call level: precision | Call level: recall |")
    w("|---|---|---|---|---|")
    for bk, nm in [("agentdrift", "AgentDrift (pooled train+test)"), ("agentdojo", "AgentDojo (P0 + unsuccessful)"), ("agentdojo_P0", "AgentDojo P0")]:
        tl = G(f"{a}/keyword_vs_annotation_{bk}/tool_level")
        w(f"| {nm} | {tl['precision']:.3f} ({tl['tp']}/{tl['tp'] + tl['fp']}) | {tl['recall']:.3f} ({tl['tp']}/{tl['tp'] + tl['fn']}) | "
          f"{F(f'{a}/keyword_vs_annotation_{bk}/occurrence_precision_ci')} | {F(f'{a}/keyword_vs_annotation_{bk}/occurrence_recall_ci')} |")
    w("")
    fp1 = G(f"{a}/keyword_vs_annotation_agentdrift/false_positive_tools"); fp2 = G(f"{a}/keyword_vs_annotation_agentdojo/false_positive_tools")
    fn1 = G(f"{a}/keyword_vs_annotation_agentdrift/false_negative_tools"); fn2 = G(f"{a}/keyword_vs_annotation_agentdojo/false_negative_tools")
    w("Read-only tools the keyword flags: " + ", ".join(f"{k} ({v} call{'s' if v != 1 else ''})" for k, v in {**fp1, **fp2}.items()) +
      ". State-changing tools it misses, most frequent: AgentDrift " + ", ".join(f"{k} {v}" for k, v in list(fn1.items())[:8]) +
      f" ({len(fn1)} tools in all); AgentDojo " + ", ".join(f"{k} {v}" for k, v in fn2.items()) + f". Excluding the {asd['n_ambiguous']} AgentDrift "
      f"tools marked ambiguous, tool-level precision is {G(a + '/keyword_vs_annotation_agentdrift/tool_level_excluding_ambiguous/precision'):.3f} "
      f"and recall {G(a + '/keyword_vs_annotation_agentdrift/tool_level_excluding_ambiguous/recall'):.3f}.")
    w("")
    hj = f"{a}/agentdrift_hijacked_steps_pooled"
    w(f"Hijacked AgentDrift steps (pooled train+test, {G(hj + '/n_hijacked_steps')} steps in {G(hj + '/n_attacked_trajectories')} attacked "
      f"trajectories) [`{hj}`]: keyword-flagged {pc(hj + '/share_keyword_flagged')}, so {pc(hj + '/share_not_keyword_flagged')} are missed; "
      f"state-changing by the annotation {pc(hj + '/share_annotated_state_changing')}; of those, the keyword misses "
      f"{G(hj + '/n_annotated_state_changing_missed_by_keyword')} of {G(hj + '/n_annotated_state_changing')} "
      f"({pc(hj + '/share_of_annotated_state_changing_missed_by_keyword')}). Most-missed state-changing tools: " +
      ", ".join(f"{k} {v}" for k, v in list(G(hj + '/top_state_changing_missed_tools').items())[:6]) +
      f". {100 * G(hj + '/share_trajectories_with_no_keyword_flagged_hijacked_step'):.1f}% of attacked trajectories have no keyword-flagged hijacked step.")
    w("")
    dj = f"{a}/agentdojo_compromised_calls"
    w(f"Compromised AgentDojo P0 traces ({G(dj + '/n_compromised_traces')} traces, {G(dj + '/n_calls')} calls) [`{dj}`]: "
      f"{G(dj + '/n_annotated_state_changing')} calls are state-changing by the annotation and the keyword misses "
      f"{G(dj + '/n_annotated_state_changing_missed_by_keyword')} of them ({pc(dj + '/share_of_annotated_state_changing_missed_by_keyword')}): " +
      ", ".join(f"{k} {v}" for k, v in G(dj + '/top_state_changing_missed_tools').items()) +
      f". AgentDojo has no per-call hijack label; counting only calls after the first located injected observation gives "
      f"{G(dj + '/post_injection_proxy/n_missed_by_keyword')} missed of {G(dj + '/post_injection_proxy/n_annotated_state_changing_after_injection')} "
      f"[`{dj}/post_injection_proxy`].")
    w("")
    w("Sensitivity: LR-B with the annotation-based count and fraction of state-changing calls (features 4 and 5) in place of "
      "the keyword ones (LR-B-annot), same recipe; LR-B-minus (no state-change features, cache) for reference. AUROC "
      "[`m8_LRB_annotation_features/<setting>/primary/metrics/<det>/AUROC`] and paired difference LR-B-annot minus LR-B "
      "[`.../contrasts/LR-B-annot minus LR-B/AUROC`; R3 Holm in `m8_LRB_annotation_features/R3_holm`]:")
    w("")
    w("| Setting | LR-B | LR-B-annot | LR-B-minus | LR-B-annot minus LR-B | Holm (R3) |")
    w("|---|---|---|---|---|---|")
    for s in ["R1"] + [f"R3_{d}" for d in DOMAINS] + ["R4"]:
        b = f"m8_LRB_annotation_features/{s}/primary"
        hl = (f"p {G(f'm8_LRB_annotation_features/R3_holm/LR-B-annot minus LR-B/AUROC/{s[3:]}/p_holm'):.3f}"
              if s.startswith("R3_") else "")
        w(f"| {s} | {F(b + '/metrics/LR-B/AUROC')} | {F(b + '/metrics/LR-B-annot/AUROC')} | {F(b + '/metrics/LR-B-minus/AUROC')} | "
          f"{Dl(b + '/contrasts/LR-B-annot minus LR-B/AUROC')} | {hl} |")
    w("")
    w(f"R1 standardized coefficients of the two write features: keyword write_n {G('m8_LRB_annotation_features/R1/coef_LR-B/write_n'):+.3f}, "
      f"write_frac {G('m8_LRB_annotation_features/R1/coef_LR-B/write_frac'):+.3f}; annotation write_n "
      f"{G('m8_LRB_annotation_features/R1/coef_LR-B-annot/write_n'):+.3f}, write_frac {G('m8_LRB_annotation_features/R1/coef_LR-B-annot/write_frac'):+.3f} "
      f"[`m8_LRB_annotation_features/R1/coef_*`]. Recall and FPR at t_r80 for both variants are in the JSON (`.../metrics/<det>/recall@t_r80`).")
    w("")
    nh = G("m8_LRB_annotation_features/R3_holm/LR-B-annot minus LR-B/AUROC/n_folds_detected_holm")
    hh = "m8_LRB_annotation_features/R3_holm/LR-B-annot minus LR-B/AUROC"
    det_signs = sorted({np.sign(G(f"{hh}/{d}/delta")) for d in DOMAINS if G(f"{hh}/{d}/detected_holm")})
    dirn = ("in opposite directions" if len(det_signs) == 2 else
            ("all positive" if det_signs == [1.0] else "all negative")) if det_signs else "none"
    assert G("m8_LRB_annotation_features/R4/primary/metrics/LR-B-annot/AUROC")[2] < 0.6
    w("**Reading.** The keyword flag is precise but incomplete: almost everything it flags changes state (the exceptions are "
      "address and currency lookups matched by 'add' and 'convert'), but it misses about half of the state-changing calls in "
      "AgentDrift, including the most frequent hijack actions (execute_command, forward_email, fill_form, submit_form), and "
      "about one in six in compromised AgentDojo traces (Slack invitations and removals). Describing the two features as the "
      "count and fraction of 'state-changing calls' (manuscript Sec. V-B) therefore overstates what they measure; 'calls whose "
      "tool name matches a write-like keyword' is accurate. Replacing them with the "
      f"annotation changes LR-B's AUROC by at most {max(abs(G(f'm8_LRB_annotation_features/{s}/primary/contrasts/LR-B-annot minus LR-B/AUROC/delta')) for s in ['R1'] + [f'R3_{d}' for d in DOMAINS] + ['R4']):.3f} "
      f"in any setting (detected after Holm in {nh} of 5 R3 folds, {dirn}) and does not rescue transfer, so the "
      "keyword errors do not drive LR-B's R1, R3 or R4 results. The annotation is one annotator's reading of tool names, not "
      "ground truth from the environments.")
    w("")
    w("## Deviations from the plan and caveats")
    w("")
    for dv in R["deviations"]:
        w(f"- {dv}")
    w("")
    w("Checks [`checks`]: cached coefficients reproduce the R1 fits exactly; the 25 WUT fold models reproduce the cached "
      "out-of-fold scores (max difference below 1e-15) and thresholds; the refitted LR-B used in the annotation sensitivity "
      "equals the cached LR-B (below 1e-13) in all seven settings; the fast average-precision routine equals sklearn's.")
    open(OUT_MD, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote", OUT_MD)


def self_test():
    rng = np.random.default_rng(0)
    for _ in range(20):
        y = (rng.random(300) < 0.3).astype(int); s = np.round(rng.random(300), 1)
        assert abs(fast_ap(y, s) - average_precision_score(y, s)) < 1e-12
        w = np.where(y == 1, 0.1 / 0.9 * (y == 0).sum() / y.sum(), 1.0)
        assert abs(fast_ap(y, s, 0.1) - average_precision_score(y, s, sample_weight=w)) < 1e-12
    return True


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "status":
        for p in PART_FUNCS:
            print(p, "done" if os.path.exists(pj(p)) else "missing")
    elif cmd == "selftest":
        print("fast_ap matches sklearn:", self_test())
    elif cmd == "all":
        t0 = time.time()
        for p in PART_FUNCS:
            if os.path.exists(pj(p)):
                continue
            if time.time() - t0 > 300:
                print("time budget used; run again"); break
            run_part(p)
    elif cmd in PART_FUNCS:
        run_part(cmd)
    elif cmd == "assemble":
        assemble()
    elif cmd == "summary":
        summary()
    else:
        raise SystemExit(f"unknown command {cmd}")
