"""Experiment 2: corrected evaluation of R1-R3 (protocol in R4_CONTROL.md Part 1, section 8).

Changes vs sweep.py (trained detectors are identical):
  - AUROC and AUPRC computed on the SAME evaluation pool E = benign + hard_negative + attacked
    (failed_attack excluded); previously AUROC used the full test set incl. failed_attack.
  - Stratified bootstrap (B=1000) over test predictions at native prevalence and at p in {1,2,5,10}%,
    resampling negatives AND positives; previously only positives were subsampled (no replacement).
  - Operating thresholds from TRAINING data (t_r80 = training recall 0.80; LOGREG also t=0.5);
    previously the 80%-recall threshold was chosen on the test set.
  - Paired bootstrap deltas between detectors within a rung; independent-bootstrap deltas across rungs.
Run: AGENTDRIFT=<clone> python rerun_r1r3.py
"""
import os, json, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from featurise import load, tool_seq, build_vocab
from baseline_drift import Markov, feats
import evalkit as ek

HERE = os.path.dirname(os.path.abspath(__file__)); MP = os.environ["AGENTDRIFT"]
B = 1000; PREVS = [None, 0.01, 0.02, 0.05, 0.10]


def run(train, test):
    vocab = build_vocab([r for r in train if r["category"] == "benign"])
    Xtr, ytr, gtr, _ = feats(train, vocab); Xte, yte, gte, _ = feats(test, vocab)
    sc = StandardScaler().fit(Xtr); Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    keep = np.isin(gte, ["benign", "hard_negative", "attacked"])
    y = yte[keep]; meth = {}
    mk = Markov().fit([tool_seq(r) for r in train if r["category"] == "benign"])
    s_tr = np.array([mk.score(tool_seq(r)) for r in train]); s_te = np.array([mk.score(tool_seq(r)) for r in test])[keep]
    t = ek.thr_at_recall(s_tr[ytr == 1]); meth["MARKOV"] = [{"score": s_te, "pred": {"t_r80": s_te >= t}}]
    lr = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xtr_s, ytr)
    s_tr = lr.predict_proba(Xtr_s)[:, 1]; s_te = lr.predict_proba(Xte_s)[:, 1][keep]
    t = ek.thr_at_recall(s_tr[ytr == 1])
    meth["LOGREG"] = [{"score": s_te, "pred": {"t_r80": s_te >= t, "t0.5": s_te >= 0.5}}]
    ifo = IsolationForest(n_estimators=200, random_state=0).fit(Xtr_s[gtr == "benign"])
    s_tr = -ifo.score_samples(Xtr_s); s_te = -ifo.score_samples(Xte_s)[keep]
    t = ek.thr_at_recall(s_tr[ytr == 1]); meth["IFOREST"] = [{"score": s_te, "pred": {"t_r80": s_te >= t}}]
    comp = {g: int((gte[keep] == g).sum()) for g in ["benign", "hard_negative", "attacked"]}
    return y, meth, comp


def main():
    st_tr = load(os.path.join(MP, "data", "train.jsonl")); st_te = load(os.path.join(MP, "data", "test.jsonl"))
    td_tr = load(os.path.join(MP, "data_taskdisjoint", "train.jsonl")); td_te = load(os.path.join(MP, "data_taskdisjoint", "test.jsonl"))
    rungs = {"R1_same": (st_tr, st_te), "R2_newtask": (td_tr, td_te)}
    pool = st_tr + st_te
    for d in ["medical", "banking", "web", "email", "coding"]:
        rungs[f"R3_LODO_{d}"] = ([r for r in pool if r["agent"] != d], [r for r in pool if r["agent"] == d])
    res, boots = {}, {}
    for ri, (name, (tr, te)) in enumerate(rungs.items()):
        y, meth, comp = run(tr, te)
        nneg, npos_all = int((y == 0).sum()), int(y.sum())
        res[name] = {"pool_composition": comp, "n_neg": nneg, "n_pos_native": npos_all,
                     "native_prevalence": round(npos_all / len(y), 4), "by_prev": {}}
        for p in PREVS:
            npos = None if p is None else int(round(p / (1 - p) * nneg))
            bt = ek.bootstrap(y, meth, n_pos=npos, B=B, seed=5000 + 17 * ri)
            key = "native" if p is None else f"p{int(p*100):02d}"
            boots[(name, key)] = bt
            rows = {}
            for m in meth:
                pt = ek.point(y, meth[m]) if p is None else None
                rows[m] = {k: [round((pt[k] if p is None else ek.ci(v)[0]), 4)] + [round(x, 4) for x in ek.ci(v)[1:]]
                           for k, v in bt[m].items()}
            res[name]["by_prev"][key] = {
                "n_pos": npos_all if npos is None else npos, "n_neg": nneg,
                "prevalence": round((npos_all if npos is None else npos) / ((npos_all if npos is None else npos) + nneg), 4),
                "metrics": rows,
                "paired": {f"LOGREG-minus-{o}": {k: ek.paired_delta(bt, "LOGREG", o, k) for k in ["AUROC", "AUPRC"]}
                           for o in ["MARKOV", "IFOREST"]}}
        print("done", name, flush=True)
    # independent deltas across rungs (LOGREG)
    xr = {}
    for name in rungs:
        if name == "R1_same": continue
        xr[name + "-minus-R1"] = {key: {k: ek.indep_delta(boots[(name, key)]["LOGREG"], boots[("R1_same", key)]["LOGREG"], k)
                                        for k in ["AUROC", "AUPRC"]} for key in ["native", "p01", "p02", "p05", "p10"]}
    res["cross_rung_LOGREG_independent"] = xr
    json.dump(res, open(os.path.join(HERE, "r1r3_corrected.json"), "w"), indent=1, default=float)
    print("wrote r1r3_corrected.json")


if __name__ == "__main__":
    main()
