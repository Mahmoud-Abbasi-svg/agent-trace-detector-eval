"""Rung 4: cross-benchmark transfer. Train on AgentDrift, test on AgentDojo (10% tool overlap).

Tests the hypothesis from earlier rungs: vocabulary-AGNOSTIC behavioural features transfer
across benchmarks; tool-IDENTITY (bag-of-tools) features do not. Detectors:
  LOGREG-agnostic : logistic regression on the 10 vocabulary-agnostic features only.
  LOGREG-full     : logistic regression on agnostic + bag-of-tools (AgentDrift vocab).
  MARKOV          : benign tool-transition model from AgentDrift (mostly unseen tools on AgentDojo).
Report AUPRC (mean [95% CI] over 20 seeds) at realistic attack prevalence 1/2/5/10%, with
AgentDojo benign fixed (392) and attacked subsampled. AUROC reported once (prevalence-invariant).
"""
import os, json, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve
from featurise import load, tool_seq, label_pos, group, build_vocab, vectorise, AGNOSTIC_NAMES
from baseline_drift import Markov

MP = os.environ["AGENTDRIFT"]
PREVS = [0.01, 0.02, 0.05, 0.10]; N_SEED = 20


def split_feats(recs, vocab):
    ag, full, y = [], [], []
    for r in recs:
        a, b = vectorise(r, vocab)
        ag.append(a); full.append(a + b); y.append(label_pos(r))
    return np.array(ag, float), np.array(full, float), np.array(y)


def op(ys, ss):
    m = {"AUPRC": average_precision_score(ys, ss) if ys.sum() else np.nan}
    return m


def ci(vals):
    v = np.array([x for x in vals if not np.isnan(x)], float)
    return (float(v.mean()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if len(v) else (np.nan,)*3


def main():
    drift = load(os.path.join(MP, "data", "train.jsonl")) + load(os.path.join(MP, "data", "test.jsonl"))
    dojo = [json.loads(l) for l in open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                     "agentdojo_traces.jsonl"), encoding="utf-8")]
    vocab = build_vocab([r for r in drift if r["category"] == "benign"])

    ag_tr, full_tr, y_tr = split_feats(drift, vocab)
    ag_te, full_te, y_te = split_feats(dojo, vocab)
    grp_te = np.array([group(r) for r in dojo])

    sc_a = StandardScaler().fit(ag_tr); sc_f = StandardScaler().fit(full_tr)
    lr_a = LogisticRegression(max_iter=1000, class_weight="balanced").fit(sc_a.transform(ag_tr), y_tr)
    lr_f = LogisticRegression(max_iter=1000, class_weight="balanced").fit(sc_f.transform(full_tr), y_tr)
    mk = Markov().fit([tool_seq(r) for r in drift if r["category"] == "benign"])

    scores = {
        "LOGREG-agnostic": lr_a.predict_proba(sc_a.transform(ag_te))[:, 1],
        "LOGREG-full": lr_f.predict_proba(sc_f.transform(full_te))[:, 1],
        "MARKOV": np.array([mk.score(tool_seq(r)) for r in dojo]),
    }
    neg = np.where(grp_te == "benign")[0]; pos = np.where(grp_te == "attacked")[0]
    print(f"AgentDojo test: {len(neg)} benign, {len(pos)} attacked; 10% tool overlap with train.\n")
    print(f"{'detector':16} {'AUROC':>6} " + " ".join(f"AUPRC@{int(p*100)}%" for p in PREVS))
    out = {}
    for det, sc in scores.items():
        auroc = roc_auc_score(y_te, sc)
        cells = []; out[det] = {"AUROC": round(float(auroc), 3)}
        for prev in PREVS:
            vals = []
            for seed in range(N_SEED):
                rng = np.random.default_rng(seed)
                npos = max(5, int(round(prev / (1 - prev) * len(neg))))
                keep = np.concatenate([neg, rng.choice(pos, min(npos, len(pos)), replace=False)])
                vals.append(op(y_te[keep], sc[keep])["AUPRC"])
            mu, lo, hi = ci(vals); out[det][f"AUPRC@{prev}"] = [round(mu,3), round(lo,3), round(hi,3)]
            cells.append(f"{mu:.2f}[{lo:.2f},{hi:.2f}]")
        print(f"{det:16} {auroc:>6.3f} " + " ".join(f"{c:>15}" for c in cells))
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cross_benchmark_results.json"), "w"), indent=1)
    print("\nwrote cross_benchmark_results.json")


if __name__ == "__main__":
    main()
