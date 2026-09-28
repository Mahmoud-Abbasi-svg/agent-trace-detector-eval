"""Pre-registered core feature ablation (PRE_ABLATION_DESIGN.md, Decision B).
Representations A (tool-only), B (behavioural), C (combined). Same LogReg, same corrected
framework as rerun_r1r3.py (R1,R3) and r4_control.py (R4). Coverage reported next to every A.
Run: AGENTDRIFT=<clone> python ablation.py
"""
import os, json, collections, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from featurise import load, build_vocab, vectorise
import evalkit as ek

HERE = os.path.dirname(os.path.abspath(__file__)); MP = os.environ["AGENTDRIFT"]
REPS = ["A", "B", "C"]; B13 = 1000; B4 = 2000; K = 5; SEEDS = [0, 1, 2, 3, 4]
PREVS13 = [None, 0.01, 0.02, 0.05, 0.10]


def X_of(recs, vocab, rep):
    out = []
    for r in recs:
        a, b = vectorise(r, vocab)
        out.append(b if rep == "A" else a if rep == "B" else a + b)
    return np.array(out, float)


def ylab(recs): return np.array([int(r["category"] == "attacked") for r in recs])


def fit(train, rep):
    vocab = build_vocab([r for r in train if r["category"] == "benign"])
    X = X_of(train, vocab, rep); y = ylab(train); sc = StandardScaler().fit(X)
    lr = LogisticRegression(max_iter=1000, class_weight="balanced").fit(sc.transform(X), y)
    s = lr.predict_proba(sc.transform(X))[:, 1]
    return (vocab, sc, lr, ek.thr_at_recall(s[y == 1], 0.80))


def sco(model, recs, rep):
    vocab, sc, lr, t80 = model
    s = lr.predict_proba(sc.transform(X_of(recs, vocab, rep)))[:, 1]
    return {"score": s, "pred": {"t_r80": s >= t80, "t0.5": s >= 0.5}}


def coverage(train, test):
    V = {s["tool"] for r in train if r["category"] == "benign" for s in r["steps"]}
    occ = [s["tool"] in V for r in test for s in r["steps"]]
    per = [np.mean([s["tool"] in V for s in r["steps"]]) for r in test]
    dt = {s["tool"] for r in test for s in r["steps"]}
    return {"tool_occurrence_cov": round(float(np.mean(occ)), 4),
            "distinct_tool_cov": round(len(dt & V) / len(dt), 4),
            "frac_traces_full_cov": round(float(np.mean([p == 1 for p in per])), 4),
            "frac_traces_any_unseen": round(float(np.mean([p < 1 for p in per])), 4),
            "frac_traces_zero_known": round(float(np.mean([p == 0 for p in per])), 4)}


def rows(y, methods, bt, native, pool=None):
    idx = np.arange(len(y)) if pool is None else np.asarray(pool)
    pt = {m: ek.point(y, methods[m], idx) for m in methods} if native else None
    return {m: {k: [round(pt[m][k] if native else ek.ci(v)[0], 4)] + [round(x, 4) for x in ek.ci(v)[1:]]
                for k, v in bt[m].items()} for m in methods}


def eval_rung(train, test, B, prevs, seed):
    models = {rep: fit(train, rep) for rep in REPS}
    keep = np.isin([r["category"] for r in test], ["benign", "hard_negative", "attacked"]) \
        if any(r["category"] in ("hard_negative", "failed_attack") for r in test) else np.ones(len(test), bool)
    test_k = [test[i] for i in np.where(keep)[0]]
    y = ylab(test_k)
    methods = {rep: [sco(models[rep], test_k, rep)] for rep in REPS}
    out = {"cov": coverage(train, test_k), "n_neg": int((y == 0).sum()), "n_pos_native": int(y.sum()),
           "native_prevalence": round(float(y.mean()), 4), "by_prev": {}}
    for p in prevs:
        npos = None if p is None else int(round(p / (1 - p) * (y == 0).sum()))
        bt = ek.bootstrap(y, methods, n_pos=npos, B=B, seed=seed)
        key = "native" if p is None else f"p{int(p*100):02d}"
        out["by_prev"][key] = {"n_pos": int(y.sum()) if npos is None else npos, "n_neg": int((y == 0).sum()),
                               "metrics": rows(y, methods, bt, native=(p is None)),
                               "paired": {f"{a}-minus-{b}": {k: ek.paired_delta(bt, a, b, k) for k in ["AUROC", "AUPRC"]}
                                          for a, b in [("C", "A"), ("C", "B"), ("B", "A")]}}
    return out, models


def main():
    st_tr = load(os.path.join(MP, "data/train.jsonl")); st_te = load(os.path.join(MP, "data/test.jsonl"))
    pool = st_tr + st_te; res = {"meta": {"reps": REPS, "B13": B13, "B4": B4, "K": K}}

    res["R1"], _ = eval_rung(st_tr, st_te, B13, PREVS13, 5000)
    print("done R1", flush=True)

    res["R3"] = {}
    for ri, d in enumerate(["medical", "banking", "web", "email", "coding"]):
        tr = [r for r in pool if r["agent"] != d]; te = [r for r in pool if r["agent"] == d]
        res["R3"][d], _ = eval_rung(tr, te, B13, PREVS13, 5034 + 17 * ri)
        print("done R3", d, flush=True)

    # ---------- R4: transfer + within-AgentDojo references ----------
    dojo = [json.loads(l) for l in open(os.path.join(HERE, "agentdojo_traces.jsonl"), encoding="utf-8")]
    tr_models = {rep: fit(pool, rep) for rep in REPS}
    def parse(r): m, s, u, i, a = r["id"].split("|"); return dict(model=m, ut=u)
    inj = {i for i, r in enumerate(dojo) if r["category"] == "benign" and parse(r)["ut"].startswith("injection_task")}
    R4 = {"cov_full": coverage(pool, dojo)}
    for pname, recs_idx in [("P0", list(range(len(dojo)))), ("C1", [i for i in range(len(dojo)) if i not in inj])]:
        recs = [dojo[i] for i in recs_idx]; y = ylab(recs)
        methods = {f"transfer_{rep}": [sco(tr_models[rep], recs, rep)] for rep in REPS}
        # within-AgentDojo UT reference per representation (5 CV reps)
        keys = [f"{r['id'].split('|')[1]}|{parse(r)['ut']}" for r in recs]
        for rep in REPS:
            reps5 = []
            for sd in SEEDS:
                groups = sorted(set(keys)); rng = np.random.default_rng(sd); rng.shuffle(groups)
                fo = {g: j % K for j, g in enumerate(groups)}; folds = np.array([fo[k] for k in keys])
                s = np.full(len(recs), np.nan); p5 = np.zeros(len(recs), bool); p8 = np.zeros(len(recs), bool)
                for f in range(K):
                    trf = [recs[i] for i in np.where(folds != f)[0]]; te_i = np.where(folds == f)[0]
                    md = fit(trf, rep); sf = sco(md, [recs[i] for i in te_i], rep)
                    s[te_i] = sf["score"]; p5[te_i] = sf["pred"]["t0.5"]; p8[te_i] = sf["pred"]["t_r80"]
                reps5.append({"score": s, "pred": {"t0.5": p5, "t_r80": p8}})
            methods[f"within_{rep}"] = reps5
        R4[pname] = {"n_neg": int((y == 0).sum()), "n_pos": int(y.sum()), "native_prevalence": round(float(y.mean()), 4), "by_prev": {}}
        for p in [None, 0.10, 0.05]:
            npos = None if p is None else int(round(p / (1 - p) * (y == 0).sum()))
            bt = ek.bootstrap(y, methods, n_pos=npos, B=B4, seed=12345)
            key = "native" if p is None else f"p{int(p*100):02d}"
            R4[pname]["by_prev"][key] = {"n_pos": int(y.sum()) if npos is None else npos, "n_neg": int((y == 0).sum()),
                "metrics": rows(y, methods, bt, native=(p is None)),
                "paired": {f"within_{r}-minus-transfer_{r}": {k: ek.paired_delta(bt, f"within_{r}", f"transfer_{r}", k) for k in ["AUROC", "AUPRC"]} for r in REPS}}
        print("done R4", pname, flush=True)
    # model-balanced within/transfer (P0), reuse composition logic
    recs = dojo; y = ylab(recs); models_of = [parse(r)["model"] for r in recs]
    cnt = collections.Counter((m, int(t)) for m, t in zip(models_of, y))
    elig = sorted({m for m in set(models_of) if cnt[(m, 0)] >= 50 and cnt[(m, 1)] >= 50})
    poolm = np.array([i for i, m in enumerate(models_of) if m in elig])
    bshare = collections.Counter(models_of[i] for i in poolm if y[i] == 0); ashare = collections.Counter(models_of[i] for i in poolm if y[i] == 1)
    nb, na = sum(bshare.values()), sum(ashare.values())
    w = {i: (bshare[models_of[i]] / nb) / (ashare[models_of[i]] / na) for i in poolm if y[i] == 1}
    methods = {f"transfer_{rep}": [sco(tr_models[rep], recs, rep)] for rep in REPS}
    keys = [f"{r['id'].split('|')[1]}|{parse(r)['ut']}" for r in recs]
    for rep in REPS:
        reps5 = []
        for sd in SEEDS:
            groups = sorted(set(keys)); rng = np.random.default_rng(sd); rng.shuffle(groups)
            fo = {g: j % K for j, g in enumerate(groups)}; folds = np.array([fo[k] for k in keys])
            s = np.full(len(recs), np.nan)
            for f in range(K):
                trf = [recs[i] for i in np.where(folds != f)[0]]; te_i = np.where(folds == f)[0]
                md = fit(trf, rep); s[te_i] = sco(md, [recs[i] for i in te_i], rep)["score"]
            reps5.append({"score": s})
        methods[f"within_{rep}"] = reps5
    nneg = int(sum(1 for i in poolm if y[i] == 0))
    bt = ek.bootstrap(y, methods, n_pos=int(round(0.10 / 0.90 * nneg)), B=B4, seed=999, pool=poolm, pos_weights=w)
    R4["model_balanced_p10"] = {"n_neg": nneg, "eligible_models": elig,
        "metrics": {m: {k: [round(x, 4) for x in ek.ci(bt[m][k])] for k in ["AUROC", "AUPRC"]} for m in methods},
        "paired": {f"within_{r}-minus-transfer_{r}": {k: ek.paired_delta(bt, f"within_{r}", f"transfer_{r}", k) for k in ["AUROC", "AUPRC"]} for r in REPS}}
    res["R4"] = R4
    json.dump(res, open(os.path.join(HERE, "ablation_results.json"), "w"), indent=1, default=float)
    print("wrote ablation_results.json")


if __name__ == "__main__":
    main()
