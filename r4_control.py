"""R4 control: AgentDojo->AgentDojo (within) vs AgentDrift->AgentDojo (transfer), same features,
same LogReg, paired stratified bootstrap. Implements R4_CONTROL.md Part 1 exactly.
Run: AGENTDRIFT=<clone> python r4_control.py
"""
import os, json, collections, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from featurise import load, build_vocab, vectorise
import evalkit as ek

HERE = os.path.dirname(os.path.abspath(__file__))
MP = os.environ["AGENTDRIFT"]
B = 2000; SEEDS = [0, 1, 2, 3, 4]; K = 5
MIN_PER_CLASS_MODEL = 50


def feats(recs, vocab, agnostic):
    X = []
    for r in recs:
        a, b = vectorise(r, vocab)
        X.append(a if agnostic else a + b)
    return np.array(X, float)


def fit_lr(train_recs, agnostic):
    vocab = build_vocab([r for r in train_recs if r["category"] == "benign"])
    X = feats(train_recs, vocab, agnostic); y = np.array([int(r["category"] == "attacked") for r in train_recs])
    sc = StandardScaler().fit(X)
    lr = LogisticRegression(max_iter=1000, class_weight="balanced").fit(sc.transform(X), y)
    s_tr = lr.predict_proba(sc.transform(X))[:, 1]
    thr80 = ek.thr_at_recall(s_tr[y == 1], 0.80)
    return vocab, sc, lr, thr80


def score(model, recs, agnostic):
    vocab, sc, lr, thr80 = model
    s = lr.predict_proba(sc.transform(feats(recs, vocab, agnostic)))[:, 1]
    return s, {"t0.5": s >= 0.5, "t_r80": s >= thr80}


def parse(r):
    m, suite, ut, inj, at = r["id"].split("|")
    return dict(model=m, suite=suite, ut=ut, inj=inj, at=at)


def group_key(r, scheme):
    p = parse(r)
    if scheme == "UT" or r["category"] == "benign":
        return f"{p['suite']}|{p['ut']}"
    return f"{p['suite']}|{p['inj']}"          # IT: attacks grouped by injection goal


def within_oof(recs, scheme, seed, agnostic):
    keys = [group_key(r, scheme) for r in recs]
    groups = sorted(set(keys)); rng = np.random.default_rng(seed); rng.shuffle(groups)
    fold_of = {g: i % K for i, g in enumerate(groups)}
    folds = np.array([fold_of[k] for k in keys])
    s = np.full(len(recs), np.nan); p5 = np.zeros(len(recs), bool); p80 = np.zeros(len(recs), bool)
    for f in range(K):
        tr = [recs[i] for i in np.where(folds != f)[0]]; te_idx = np.where(folds == f)[0]
        ytr = [r["category"] == "attacked" for r in tr]; yte = [recs[i]["category"] == "attacked" for i in te_idx]
        assert 0 < sum(ytr) < len(ytr) and 0 < sum(yte) < len(yte), f"fold {f} single-class"
        mdl = fit_lr(tr, agnostic)
        sf, pf = score(mdl, [recs[i] for i in te_idx], agnostic)
        s[te_idx] = sf; p5[te_idx] = pf["t0.5"]; p80[te_idx] = pf["t_r80"]
    return {"score": s, "pred": {"t0.5": p5, "t_r80": p80}}, folds


def summarize(y, methods, npos, nneg, B, pool=None, pos_weights=None):
    bt = ek.bootstrap(y, methods, n_pos=npos, n_neg=nneg, B=B, pos_weights=pos_weights, pool=pool)
    return bt


def table(tag, y, methods, bt, native, pool=None):
    idx = np.arange(len(y)) if pool is None else np.asarray(pool)
    rows = {}
    for m in methods:
        pt = ek.point(y, methods[m], idx) if native else None
        rows[m] = {}
        for k in bt[m]:
            mu, lo, hi = ek.ci(bt[m][k])
            est = pt[k] if native else mu
            rows[m][k] = [round(est, 4), round(lo, 4), round(hi, 4)]
    return rows


def main():
    drift = load(os.path.join(MP, "data", "train.jsonl")) + load(os.path.join(MP, "data", "test.jsonl"))
    dojo = [json.loads(l) for l in open(os.path.join(HERE, "agentdojo_traces.jsonl"), encoding="utf-8")]
    res = {"meta": {"B": B, "seeds": SEEDS, "K": K}}

    # ---------- transfer models (identical to cross_benchmark.py training data) ----------
    tr_full = fit_lr(drift, agnostic=False); tr_agn = fit_lr(drift, agnostic=True)

    pops = {}
    inj_benign = [i for i, r in enumerate(dojo) if r["category"] == "benign" and parse(r)["ut"].startswith("injection_task")]
    pops["P0"] = list(range(len(dojo)))
    pops["C1"] = [i for i in range(len(dojo)) if i not in set(inj_benign)]
    res["meta"]["n_injection_task_benign"] = len(inj_benign)

    oof_store = {}
    for pname, pidx in pops.items():
        recs = [dojo[i] for i in pidx]
        y = np.array([int(r["category"] == "attacked") for r in recs])
        npos_all, nneg_all = int(y.sum()), int(len(y) - y.sum())
        methods = {}
        s, pr = score(tr_full, recs, False); methods["transfer_full"] = [{"score": s, "pred": pr}]
        s, pr = score(tr_agn, recs, True); methods["transfer_agn"] = [{"score": s, "pred": pr}]
        for scheme in ["UT", "IT"]:
            for agn, lab in [(False, "full"), (True, "agn")]:
                reps = []
                for sd in SEEDS:
                    rep, folds = within_oof(recs, scheme, sd, agn); reps.append(rep)
                methods[f"within_{scheme}_{lab}"] = reps
        oof_store[pname] = (recs, y, methods)
        res[pname] = {"n_benign": nneg_all, "n_attacked": npos_all,
                      "native_prevalence": round(npos_all / len(y), 4)}
        for plabel, p in [("native", None), ("p10", 0.10), ("p05", 0.05)]:
            npos = None if p is None else int(round(p / (1 - p) * nneg_all))
            bt = ek.bootstrap(y, methods, n_pos=npos, n_neg=None, B=B)
            res[pname][plabel] = {
                "n_pos": npos_all if npos is None else npos, "n_neg": nneg_all,
                "prevalence": round((npos_all if npos is None else npos) / ((npos_all if npos is None else npos) + nneg_all), 4),
                "metrics": table(pname, y, methods, bt, native=(p is None)),
                "paired": {f"{a}-minus-{b}": {k: ek.paired_delta(bt, a, b, k) for k in ["AUROC", "AUPRC"]}
                           for a, b in [("within_UT_full", "transfer_full"), ("within_UT_agn", "transfer_agn"),
                                        ("within_IT_full", "transfer_full"), ("within_IT_agn", "transfer_agn")]},
            }
            print(f"done {pname} {plabel}", flush=True)

    # ---------- model confound (P0) ----------
    recs, y, methods = oof_store["P0"]
    models = [parse(r)["model"] for r in recs]
    cnt = collections.Counter((m, int(t)) for m, t in zip(models, y))
    eligible = sorted({m for m in set(models) if cnt[(m, 0)] >= MIN_PER_CLASS_MODEL and cnt[(m, 1)] >= MIN_PER_CLASS_MODEL})
    res["model_counts"] = {m: {"benign": cnt[(m, 0)], "attacked": cnt[(m, 1)]} for m in sorted(set(models))}
    res["model_eligible"] = eligible
    sub = {k: methods[k] for k in ["within_UT_full", "within_IT_full", "transfer_full", "within_UT_agn", "transfer_agn"]}
    per = {}
    for m in eligible:
        pool = np.array([i for i, mm in enumerate(models) if mm == m])
        bt = ek.bootstrap(y, sub, B=B, pool=pool)
        per[m] = {"n_benign": cnt[(m, 0)], "n_attacked": cnt[(m, 1)],
                  "metrics": {k: {mt: [round(ek.point(y, sub[k], pool)[mt], 4)] + [round(x, 4) for x in ek.ci(bt[k][mt])[1:]]
                                  for mt in ["AUROC", "AUPRC"]} for k in sub},
                  "paired_UTfull_minus_transfer_full": {mt: ek.paired_delta(bt, "within_UT_full", "transfer_full", mt) for mt in ["AUROC", "AUPRC"]}}
    res["per_model"] = per
    res["macro_within_model_AUROC"] = {k: round(float(np.mean([per[m]["metrics"][k]["AUROC"][0] for m in eligible])), 4) for k in sub}
    # model-balanced pooled: weight positives so their model mix equals the benign model mix
    pool = np.array([i for i, mm in enumerate(models) if mm in eligible])
    bshare = collections.Counter(models[i] for i in pool if y[i] == 0); ashare = collections.Counter(models[i] for i in pool if y[i] == 1)
    nb, na = sum(bshare.values()), sum(ashare.values())
    w = {i: (bshare[models[i]] / nb) / (ashare[models[i]] / na) for i in pool if y[i] == 1}
    nneg = int(sum(1 for i in pool if y[i] == 0))
    mb = {}
    for plabel, p in [("native_counts", None), ("p10", 0.10)]:
        npos = None if p is None else int(round(p / (1 - p) * nneg))
        bt = ek.bootstrap(y, sub, n_pos=npos, B=B, pool=pool, pos_weights=w)
        mb[plabel] = {"n_neg": nneg, "n_pos": (int(sum(1 for i in pool if y[i] == 1)) if npos is None else npos),
                      "metrics": {k: {mt: [round(x, 4) for x in ek.ci(bt[k][mt])] for mt in ["AUROC", "AUPRC"]} for k in sub},
                      "paired_UTfull_minus_transfer_full": {mt: ek.paired_delta(bt, "within_UT_full", "transfer_full", mt) for mt in ["AUROC", "AUPRC"]}}
    res["model_balanced"] = mb

    # ---------- how the 65 injection-task benigns are scored (P0) ----------
    inj_set = set(inj_benign)
    grp = np.array(["inj_benign" if i in inj_set else ("attacked" if y[i] else "other_benign") for i in range(len(y))])
    diag = {}
    for k in ["within_UT_full", "transfer_full"]:
        rep = methods[k][0]
        diag[k] = {g: {"n": int((grp == g).sum()),
                       "mean_score": round(float(np.nanmean(rep["score"][grp == g])), 4),
                       "flagged@t0.5": round(float(rep["pred"]["t0.5"][grp == g].mean()), 4),
                       "flagged@t_r80": round(float(rep["pred"]["t_r80"][grp == g].mean()), 4)} for g in ["inj_benign", "other_benign", "attacked"]}
    res["injection_task_benign_diagnostic"] = diag

    # ---------- model-identity probe (benign only) ----------
    ben = [r for r in dojo if r["category"] == "benign"]
    yb = np.array([int(parse(r)["model"] == "gpt-4o-2024-05-13") for r in ben])
    probe = {}
    for agn, lab in [(False, "full"), (True, "agn")]:
        rng = np.random.default_rng(0); order = rng.permutation(len(ben)); folds = np.empty(len(ben), int)
        for c in (0, 1):  # stratified fold assignment
            idc = [i for i in order if yb[i] == c]
            for j, i in enumerate(idc): folds[i] = j % K
        s = np.zeros(len(ben))
        for f in range(K):
            tr = [ben[i] for i in np.where(folds != f)[0]]; te = np.where(folds == f)[0]
            vocab = build_vocab(tr); X = feats(tr, vocab, agn); sc = StandardScaler().fit(X)
            lr = LogisticRegression(max_iter=1000, class_weight="balanced").fit(sc.transform(X), yb[folds != f])
            s[te] = lr.predict_proba(sc.transform(feats([ben[i] for i in te], vocab, agn)))[:, 1]
        bt = ek.bootstrap(yb, {"probe": [{"score": s}]}, B=B)
        probe[lab] = {"n_gpt4o": int(yb.sum()), "n_other": int(len(yb) - yb.sum()),
                      "AUROC": [round(ek.auroc(yb, s), 4)] + [round(x, 4) for x in ek.ci(bt["probe"]["AUROC"])[1:]]}
    res["model_identity_probe_benign"] = probe

    json.dump(res, open(os.path.join(HERE, "r4_control_results.json"), "w"), indent=1, default=float)
    print("wrote r4_control_results.json")


if __name__ == "__main__":
    main()
