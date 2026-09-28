"""Run the GRU through the full ladder per GRU_PREREGISTRATION.md, plus a pre-run split audit.
Primary comparator = combined LogReg (C). Uses evalkit for bootstrap/paired deltas.
Run: AGENTDRIFT=<clone> python gru_ladder.py [audit|full]
"""
import os, sys, json, collections, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from featurise import load, build_vocab, vectorise
import evalkit as ek
from gru_detector import train_gru, score_gru, ylab, beh_feats, build_tool_vocab, MAXLEN

MP = os.environ["AGENTDRIFT"]; HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_SEEDS = [0, 1, 2]; VAL_FRAC = 0.10; B13 = 1000; B4 = 2000; K = 5
MODE = sys.argv[1] if len(sys.argv) > 1 else "full"


def stratified_val(train, seed=0, frac=VAL_FRAC):
    rng = np.random.default_rng(1000 + seed); y = ylab(train)
    tr, va = [], []
    for c in (0, 1):
        idx = np.where(y == c)[0]; rng.shuffle(idx); k = int(round(frac * len(idx)))
        va += list(idx[:k]); tr += list(idx[k:])
    return [train[i] for i in tr], [train[i] for i in va]


def coverage(train, test):
    V = {s["tool"] for r in train if r["category"] == "benign" for s in r["steps"]}
    occ = [s["tool"] in V for r in test for s in r["steps"]]
    per = [np.mean([s["tool"] in V for s in r["steps"]]) for r in test]
    dt = {s["tool"] for r in test for s in r["steps"]}
    return {"tool_occ_cov": round(float(np.mean(occ)), 4), "distinct_cov": round(len(dt & V) / max(len(dt), 1), 4),
            "frac_zero_known": round(float(np.mean([p == 0 for p in per])), 4),
            "frac_any_unseen": round(float(np.mean([p < 1 for p in per])), 4)}


# ---- combined LogReg C (unchanged recipe) for the paired comparison ----
def fitC(train):
    vocab = build_vocab([r for r in train if r["category"] == "benign"])
    def X(recs): return np.array([np.concatenate(vectorise(r, vocab)) for r in recs], float)
    y = ylab(train); sc = StandardScaler().fit(X(train))
    lr = LogisticRegression(max_iter=1000, class_weight="balanced").fit(sc.transform(X(train)), y)
    s = lr.predict_proba(sc.transform(X(train)))[:, 1]
    thr = float(np.quantile(s[y == 1], 0.20))
    return lambda recs: (lambda ss: {"score": ss, "pred": {"t_r80": ss >= thr, "t0.5": ss >= 0.5}})(lr.predict_proba(sc.transform(X(recs)))[:, 1])


def gru_reps(train, test, tool_only=False):
    reps = []
    for sd in TRAIN_SEEDS:
        tr, va = stratified_val(train, sd)
        m, voc, scl, thr = train_gru(tr, va, sd, tool_only)
        reps.append(score_gru(m, voc, scl, thr, test))
    return reps


def audit():
    A = {}
    st_tr = load(os.path.join(MP, "data/train.jsonl")); st_te = load(os.path.join(MP, "data/test.jsonl"))
    st_val = load(os.path.join(MP, "data/val.jsonl")); pool = st_tr + st_te
    def rec(name, tr, te, val=None):
        tv = build_tool_vocab(tr)
        A[name] = {"n_train": len(tr), "n_val": (len(val) if val is not None else None), "n_test": len(te),
                   "train_prev": round(float(ylab(tr).mean()), 4), "test_prev": round(float(ylab(te).mean()), 4),
                   "tool_vocab_size": len(tv), "coverage": coverage(tr, te)}
    rec("R1", st_tr, st_te, st_val)
    td_tr = load(os.path.join(MP, "data_taskdisjoint/train.jsonl")); td_te = load(os.path.join(MP, "data_taskdisjoint/test.jsonl"))
    rec("R2", td_tr, td_te)
    for d in ["medical", "banking", "web", "email", "coding"]:
        rec(f"R3_{d}", [r for r in pool if r["agent"] != d], [r for r in pool if r["agent"] == d])
    dojo = [json.loads(l) for l in open(os.path.join(HERE, "agentdojo_traces.jsonl"), encoding="utf-8")]
    rec("R4_transfer", pool, dojo)
    A["seeds"] = {"train_seeds": TRAIN_SEEDS, "val_frac": VAL_FRAC, "val_seed_base": 1000, "K": K}
    A["R1_val_source"] = "data/val.jsonl (1880)"; A["other_val_source"] = "stratified 10% of that rung's train"
    json.dump(A, open(os.path.join(HERE, "gru_audit.json"), "w"), indent=1, default=float)
    for k, v in A.items():
        if isinstance(v, dict) and "coverage" in v:
            print(f"{k:14} tr={v['n_train']} val={v['n_val']} te={v['n_test']} prev tr/te={v['train_prev']}/{v['test_prev']} "
                  f"vocab={v['tool_vocab_size']} cov={v['coverage']}")
    print("seeds:", A["seeds"]); print("wrote gru_audit.json")
    return A


def eval_pool(train, test, B, prevs, seed, extra_C=True):
    keep = [r for r in test if r["category"] in ("benign", "hard_negative", "attacked")]
    y = ylab(keep)
    methods = {"GRU": gru_reps(train, keep)}
    if extra_C:
        Cf = fitC(train); methods["C"] = [Cf(keep)]
    out = {"cov": coverage(train, keep), "n_neg": int((y == 0).sum()), "n_pos": int(y.sum()),
           "native_prev": round(float(y.mean()), 4), "by_prev": {}}
    for p in prevs:
        npos = None if p is None else int(round(p / (1 - p) * (y == 0).sum()))
        bt = ek.bootstrap(y, methods, n_pos=npos, B=B, seed=seed)
        key = "native" if p is None else f"p{int(p*100):02d}"
        rows = {}
        for m in methods:
            pt = ek.point(y, methods[m]) if p is None else None
            rows[m] = {k: [round(pt[k] if p is None else ek.ci(v)[0], 4)] + [round(x, 4) for x in ek.ci(v)[1:]] for k, v in bt[m].items()}
        paired = {"GRU-minus-C": {k: ek.paired_delta(bt, "GRU", "C", k) for k in ["AUROC", "AUPRC"]}} if extra_C else {}
        out["by_prev"][key] = {"n_pos": int(y.sum()) if npos is None else npos, "n_neg": int((y == 0).sum()), "metrics": rows, "paired": paired}
    return out


def full():
    st_tr = load(os.path.join(MP, "data/train.jsonl")); st_te = load(os.path.join(MP, "data/test.jsonl")); pool = st_tr + st_te
    R = {"meta": {"train_seeds": TRAIN_SEEDS}}
    prevs = [None, 0.01, 0.02, 0.05, 0.10]
    R["R1"] = eval_pool(st_tr, st_te, B13, prevs, 5000); print("done R1", flush=True)
    td_tr = load(os.path.join(MP, "data_taskdisjoint/train.jsonl")); td_te = load(os.path.join(MP, "data_taskdisjoint/test.jsonl"))
    R["R2"] = eval_pool(td_tr, td_te, B13, prevs, 5100); print("done R2", flush=True)
    R["R3"] = {}
    for ri, d in enumerate(["medical", "banking", "web", "email", "coding"]):
        R["R3"][d] = eval_pool([r for r in pool if r["agent"] != d], [r for r in pool if r["agent"] == d], B13, prevs, 5200 + ri)
        print("done R3", d, flush=True)
    # R1 world-disjoint (reuse split from world_disjoint_r1); seed 0 only, 3 train seeds
    from world_disjoint_r1 import world_disjoint_split
    tr_i, te_i = world_disjoint_split(pool, 0)
    R["R1_wd"] = eval_pool([pool[i] for i in tr_i], [pool[i] for i in te_i], B13, prevs, 5300)
    R["R1_wd"]["world_overlap"] = len(({json.dumps(pool[i]["world"], sort_keys=True) for i in tr_i}) & ({json.dumps(pool[i]["world"], sort_keys=True) for i in te_i}))
    print("done R1_wd", flush=True)
    # R4 transfer + within-AgentDojo (P0, C1, model-balanced) with GRU and C
    dojo = [json.loads(l) for l in open(os.path.join(HERE, "agentdojo_traces.jsonl"), encoding="utf-8")]
    def parse(r): m, s, u, i, a = r["id"].split("|"); return dict(model=m, ut=u)
    R["R4"] = {"cov_transfer": coverage(pool, dojo)}
    inj = {i for i, r in enumerate(dojo) if r["category"] == "benign" and parse(r)["ut"].startswith("injection_task")}
    # transfer models trained once per seed on full AgentDrift
    tr_reps = gru_reps(pool, dojo)  # scores over full dojo, per seed
    Cf = fitC(pool); P0_methods = {}
    for pname, idxs in [("P0", list(range(len(dojo)))), ("C1", [i for i in range(len(dojo)) if i not in inj])]:
        recs = [dojo[i] for i in idxs]; y = ylab(recs)
        methods = {"GRU_transfer": [{"score": rp["score"][idxs], "pred": {k: v[idxs] for k, v in rp["pred"].items()}} for rp in tr_reps],
                   "C_transfer": [ (lambda d: {"score": d["score"][idxs], "pred": {k: v[idxs] for k, v in d["pred"].items()}})(Cf(dojo)) ]}
        # within-AgentDojo UT, GRU, 5 folds x 3 seeds -> 15 reps; C within 5 folds
        keys = [f"{r['id'].split('|')[1]}|{parse(r)['ut']}" for r in recs]
        gru_within, c_within = [], []
        for sd in range(K):
            groups = sorted(set(keys)); rng = np.random.default_rng(sd); rng.shuffle(groups)
            fo = {g: j % K for j, g in enumerate(groups)}; folds = np.array([fo[k] for k in keys])
            # C within (deterministic): one rep
            sC = np.full(len(recs), np.nan); p5 = np.zeros(len(recs), bool); p8 = np.zeros(len(recs), bool)
            for f in range(K):
                trf = [recs[i] for i in np.where(folds != f)[0]]; te_i2 = np.where(folds == f)[0]
                cf = fitC(trf); d = cf([recs[i] for i in te_i2]); sC[te_i2] = d["score"]; p5[te_i2] = d["pred"]["t0.5"]; p8[te_i2] = d["pred"]["t_r80"]
            c_within.append({"score": sC, "pred": {"t0.5": p5, "t_r80": p8}})
            # GRU within: for this fold-assignment, train 1 seed (sd) -> keeps ~5 reps
            sG = np.full(len(recs), np.nan); q5 = np.zeros(len(recs), bool); q8 = np.zeros(len(recs), bool)
            for f in range(K):
                tri = np.where(folds != f)[0]; te_i2 = np.where(folds == f)[0]
                tr2, va2 = stratified_val([recs[i] for i in tri], sd)
                from gru_detector import train_gru as _tg
                m, voc, scl, thr = _tg(tr2, va2, sd)
                d = score_gru(m, voc, scl, thr, [recs[i] for i in te_i2]); sG[te_i2] = d["score"]; q5[te_i2] = d["pred"]["t0.5"]; q8[te_i2] = d["pred"]["t_r80"]
            gru_within.append({"score": sG, "pred": {"t0.5": q5, "t_r80": q8}})
        methods["GRU_within"] = gru_within; methods["C_within"] = c_within
        if pname == "P0": P0_methods = (methods, y, [parse(r)["model"] for r in recs])
        R["R4"][pname] = {"n_neg": int((y == 0).sum()), "n_pos": int(y.sum()), "native_prev": round(float(y.mean()), 4), "by_prev": {}}
        for p in [None, 0.10, 0.05]:
            npos = None if p is None else int(round(p / (1 - p) * (y == 0).sum()))
            bt = ek.bootstrap(y, methods, n_pos=npos, B=B4, seed=12345)
            key = "native" if p is None else f"p{int(p*100):02d}"
            rows = {m: {k: [round((ek.point(y, methods[m])[k] if p is None else ek.ci(v)[0]), 4)] + [round(x, 4) for x in ek.ci(v)[1:]] for k, v in bt[m].items()} for m in methods}
            R["R4"][pname]["by_prev"][key] = {"n_pos": int(y.sum()) if npos is None else npos, "n_neg": int((y == 0).sum()), "metrics": rows,
                "paired": {"GRU_within-minus-GRU_transfer": {k: ek.paired_delta(bt, "GRU_within", "GRU_transfer", k) for k in ["AUROC", "AUPRC"]},
                           "GRU_transfer-minus-C_transfer": {k: ek.paired_delta(bt, "GRU_transfer", "C_transfer", k) for k in ["AUROC", "AUPRC"]}}}
        print("done R4", pname, flush=True)
    # model-balanced @10% (reuse P0 scores; reweight positives to benign model mix; no retraining)
    methods, y, models_of = P0_methods
    cnt = collections.Counter((m, int(t)) for m, t in zip(models_of, y))
    elig = sorted({m for m in set(models_of) if cnt[(m, 0)] >= 50 and cnt[(m, 1)] >= 50})
    poolm = np.array([i for i, m in enumerate(models_of) if m in elig])
    bshare = collections.Counter(models_of[i] for i in poolm if y[i] == 0); ashare = collections.Counter(models_of[i] for i in poolm if y[i] == 1)
    nb, na = sum(bshare.values()), sum(ashare.values())
    w = {i: (bshare[models_of[i]] / nb) / (ashare[models_of[i]] / na) for i in poolm if y[i] == 1}
    nneg = int(sum(1 for i in poolm if y[i] == 0))
    bt = ek.bootstrap(y, methods, n_pos=int(round(0.10 / 0.90 * nneg)), B=B4, seed=999, pool=poolm, pos_weights=w)
    R["R4"]["model_balanced_p10"] = {"n_neg": nneg, "eligible_models": elig,
        "metrics": {m: {k: [round(x, 4) for x in ek.ci(bt[m][k])] for k in ["AUROC", "AUPRC"]} for m in methods},
        "paired": {"GRU_within-minus-GRU_transfer": {k: ek.paired_delta(bt, "GRU_within", "GRU_transfer", k) for k in ["AUROC", "AUPRC"]}}}
    print("done R4 model_balanced", flush=True)
    json.dump(R, open(os.path.join(HERE, "gru_results.json"), "w"), indent=1, default=float)
    print("wrote gru_results.json")


if __name__ == "__main__":
    audit()
    if MODE == "full":
        full()
