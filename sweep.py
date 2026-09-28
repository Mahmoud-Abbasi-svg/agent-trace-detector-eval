"""Multi-seed x prevalence sweep with bootstrap CIs for the agent-trace IDS baselines.

For each drift rung and detector, sweep test-attack prevalence in {1,2,5,10}% and repeat the
subsampling over N_SEED seeds; report mean AUPRC / prec@rec80 / alerts-per-1000 with 95% CIs
across seeds. Detectors are trained ONCE per rung (deterministic); only the test subsampling
varies across seeds, so the CIs quantify sampling variability of the operating metrics under
realistic prevalence. AUROC is prevalence-invariant so reported once.
Runs on CPU. Data: AgentDrift (CC BY 4.0).
"""
import os, json, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve
from featurise import load, tool_seq, label_pos, group, domain, build_vocab, vectorise
from baseline_drift import Markov, feats

MP = os.environ["AGENTDRIFT"]
PREVS = [0.01, 0.02, 0.05, 0.10]
N_SEED = 20


def op_metrics(ys, ss):
    m = {"AUPRC": average_precision_score(ys, ss) if ys.sum() else np.nan}
    p, r, th = precision_recall_curve(ys, ss)
    ok = np.where(r[:-1] >= 0.8)[0]
    if len(ok):
        j = ok[-1]
        m["prec@rec80"] = float(p[j])
        m["alerts_per_1000"] = float(1000 * (ss >= th[j]).mean())
    else:
        m["prec@rec80"] = np.nan; m["alerts_per_1000"] = np.nan
    return m


def ci(vals):
    v = np.array([x for x in vals if not np.isnan(x)], float)
    if len(v) == 0: return (np.nan, np.nan, np.nan)
    lo, hi = np.percentile(v, [2.5, 97.5])
    return float(v.mean()), float(lo), float(hi)


def train_scores(train, test):
    """Return dict detector-> full-test anomaly scores, plus y and grp arrays (order == test)."""
    vocab = build_vocab([r for r in train if r["category"] == "benign"])
    Xtr, ytr, gtr, _ = feats(train, vocab)
    Xte, yte, gte, _ = feats(test, vocab)
    sc = StandardScaler().fit(Xtr); Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    out = {}
    mk = Markov().fit([tool_seq(r) for r in train if r["category"] == "benign"])
    out["MARKOV"] = np.array([mk.score(tool_seq(r)) for r in test])
    lr = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xtr_s, ytr)
    out["LOGREG"] = lr.predict_proba(Xte_s)[:, 1]
    ifo = IsolationForest(n_estimators=200, random_state=0).fit(Xtr_s[gtr == "benign"])
    out["IFOREST"] = -ifo.score_samples(Xte_s)
    return out, yte, gte


def sweep_rung(name, train, test, res):
    scores, y, grp = train_scores(train, test)
    neg = np.where((grp == "benign") | (grp == "hard_negative"))[0]
    pos = np.where(grp == "attacked")[0]
    for det, sc in scores.items():
        auroc = roc_auc_score(y, sc) if 0 < y.sum() < len(y) else np.nan
        for prev in PREVS:
            per = {"AUPRC": [], "prec@rec80": [], "alerts_per_1000": []}
            for seed in range(N_SEED):
                rng = np.random.default_rng(1000 + seed)
                n_pos = max(5, int(round(prev / (1 - prev) * len(neg))))
                keep = np.concatenate([neg, rng.choice(pos, min(n_pos, len(pos)), replace=False)])
                ys, ss = y[keep], sc[keep]
                m = op_metrics(ys, ss)
                for k in per: per[k].append(m[k])
            row = {"rung": name, "detector": det, "prevalence": prev, "AUROC": round(float(auroc), 3)}
            for k in per:
                mu, lo, hi = ci(per[k]); row[k] = round(mu, 3); row[k + "_lo"] = round(lo, 3); row[k + "_hi"] = round(hi, 3)
            res.append(row)


def main():
    strat_tr = load(os.path.join(MP, "data", "train.jsonl"))
    strat_te = load(os.path.join(MP, "data", "test.jsonl"))
    td_tr = load(os.path.join(MP, "data_taskdisjoint", "train.jsonl"))
    td_te = load(os.path.join(MP, "data_taskdisjoint", "test.jsonl"))
    res = []
    sweep_rung("R1_same", strat_tr, strat_te, res)
    sweep_rung("R2_newtask", td_tr, td_te, res)
    pool = strat_tr + strat_te
    for held in ["medical", "banking", "web", "email", "coding"]:
        tr = [r for r in pool if r["agent"] != held]; te = [r for r in pool if r["agent"] == held]
        sweep_rung(f"R3_LODO_{held}", tr, te, res)
    json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweep_results.json"), "w"), indent=1)
    # print a focused table: LOGREG AUPRC across rung x prevalence with CI
    print("LOGREG  AUPRC (mean [95% CI])  by rung x prevalence")
    print(f"{'rung':16} {'AUROC':>6} " + " ".join(f"{int(p*100):>16}%" for p in PREVS))
    rungs = ["R1_same", "R2_newtask"] + [f"R3_LODO_{d}" for d in ["medical","banking","web","email","coding"]]
    for rg in rungs:
        rows = [r for r in res if r["rung"] == rg and r["detector"] == "LOGREG"]
        auroc = rows[0]["AUROC"] if rows else float("nan")
        cells = []
        for p in PREVS:
            r = next((x for x in rows if x["prevalence"] == p), None)
            cells.append(f"{r['AUPRC']:.2f}[{r['AUPRC_lo']:.2f},{r['AUPRC_hi']:.2f}]" if r else "-")
        print(f"{rg:16} {auroc:>6.3f} " + " ".join(f"{c:>17}" for c in cells))
    # LODO summary: mean AUPRC across 5 held-out domains at 2%
    lodo2 = [r for r in res if r["rung"].startswith("R3_LODO") and r["detector"]=="LOGREG" and r["prevalence"]==0.02]
    import statistics as st
    print(f"\nLOGREG mean AUPRC @2% : R1={next(x for x in res if x['rung']=='R1_same' and x['detector']=='LOGREG' and x['prevalence']==0.02)['AUPRC']:.3f}"
          f"  R3-LODO(avg 5 domains)={st.mean(x['AUPRC'] for x in lodo2):.3f} (range {min(x['AUPRC'] for x in lodo2):.3f}-{max(x['AUPRC'] for x in lodo2):.3f})")
    print("wrote sweep_results.json")


if __name__ == "__main__":
    main()
