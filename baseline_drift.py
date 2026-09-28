"""First detection results on the drift ladder, rungs 1-2 (+ rung 3 leave-one-domain-out).

Detectors (all CPU-friendly honest baselines):
  - MARKOV : first-order tool-transition model trained on BENIGN traces only; anomaly score =
             mean negative log-prob of a trajectory's transitions (unsupervised, one-class).
  - LOGREG : supervised logistic regression on [agnostic + bag-of-tools] features.
  - IFOREST: Isolation Forest on the same features, trained on benign only (one-class).

Evaluation protocol (IDS-grade):
  Positive class = 'attacked'. We report AUPRC (primary under imbalance) and AUROC.
  We SUBSAMPLE attacked traces in the TEST set to a realistic prevalence (default 2%) and report
  precision@recall and alerts-per-1000-traces. Hard-negative false-positive rate reported separately.

Drift rungs:
  R1 same-distribution : train on stratified train, test on stratified test.
  R2 new-task same-tools: train on task-disjoint train, test on task-disjoint test.
  R3 new-domain (tool drift): leave-one-domain-out (train on 4 domains, test on held-out domain).
"""
import os, json, math, collections, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve
from featurise import load, tool_seq, label_pos, group, domain, build_vocab, vectorise

MP = os.environ["AGENTDRIFT"]
RNG = np.random.default_rng(42)


# ---------- Markov one-class ----------
class Markov:
    def fit(self, benign_seqs, k_smooth=0.5):
        trans = collections.defaultdict(lambda: collections.Counter())
        starts = collections.Counter()
        vocab = set()
        for s in benign_seqs:
            if not s: continue
            starts[s[0]] += 1
            for a, b in zip(s, s[1:]):
                trans[a][b] += 1; vocab.add(a); vocab.add(b)
        self.vocab = vocab | {"<unk>"}; self.k = k_smooth
        self.trans = trans; self.starts = starts
        self.tot_start = sum(starts.values())
        return self

    def _lp(self, a, b):
        row = self.trans.get(a, {})
        denom = sum(row.values()) + self.k * len(self.vocab)
        return math.log((row.get(b, 0) + self.k) / denom)

    def score(self, seq):
        if len(seq) < 2: return 0.0
        lp = sum(self._lp(a, b) for a, b in zip(seq, seq[1:]))
        return -lp / (len(seq) - 1)  # higher = more anomalous


def feats(recs, vocab):
    X, y, grp, dom = [], [], [], []
    for r in recs:
        a, b = vectorise(r, vocab)
        X.append(a + b); y.append(label_pos(r)); grp.append(group(r)); dom.append(domain(r))
    return np.array(X, float), np.array(y), np.array(grp), np.array(dom)


def subsample_prevalence(y, grp, prev=0.02, seed=0):
    """Return index mask giving 'attacked' prevalence ~prev among {attacked + benign+hardneg}.
    failed_attack excluded from the test pool (ambiguous)."""
    rng = np.random.default_rng(seed)
    neg = np.where((grp == "benign") | (grp == "hard_negative"))[0]
    pos = np.where(grp == "attacked")[0]
    n_pos_target = max(5, int(round(prev / (1 - prev) * len(neg))))
    pos_keep = rng.choice(pos, min(n_pos_target, len(pos)), replace=False)
    idx = np.concatenate([neg, pos_keep]); rng.shuffle(idx)
    return idx


def metrics(y, score, grp, prev=0.02):
    out = {}
    idx = subsample_prevalence(y, grp, prev)
    ys, ss, gs = y[idx], score[idx], grp[idx]
    out["prevalence"] = float(ys.mean())
    out["AUPRC"] = float(average_precision_score(ys, ss)) if ys.sum() else float("nan")
    out["AUROC"] = float(roc_auc_score(ys, ss)) if 0 < ys.sum() < len(ys) else float("nan")
    # operating point: threshold at 80% recall, report precision + alerts/1000
    p, r, th = precision_recall_curve(ys, ss)
    ok = np.where(r[:-1] >= 0.8)[0]
    if len(ok):
        j = ok[-1]; thr = th[j]
        alerts = (ss >= thr)
        out["prec@rec80"] = float(p[j])
        out["alerts_per_1000"] = float(1000 * alerts.mean())
        # false positives specifically on hard negatives
        hn = gs == "hard_negative"
        out["FP_rate_hardneg"] = float(alerts[hn].mean()) if hn.any() else float("nan")
    return out


def run_rung(name, train, test, prev=0.02):
    vocab = build_vocab([r for r in train if r["category"] == "benign"])
    Xtr, ytr, gtr, dtr = feats(train, vocab)
    Xte, yte, gte, dte = feats(test, vocab)
    sc = StandardScaler().fit(Xtr); Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    res = {}
    # Markov (benign-only)
    mk = Markov().fit([tool_seq(r) for r in train if r["category"] == "benign"])
    s_mk = np.array([mk.score(tool_seq(r)) for r in test])
    res["MARKOV"] = metrics(yte, s_mk, gte, prev)
    # LogReg (supervised: benign+hardneg+failed as 0, attacked as 1)
    lr = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xtr_s, ytr)
    res["LOGREG"] = metrics(yte, lr.predict_proba(Xte_s)[:, 1], gte, prev)
    # IsolationForest (benign-only)
    ben = ytr == 0
    ifo = IsolationForest(n_estimators=200, random_state=0).fit(Xtr_s[gtr == "benign"])
    res["IFOREST"] = metrics(yte, -ifo.score_samples(Xte_s), gte, prev)
    print(f"\n===== {name}  (test n={len(test)}) =====")
    for det, m in res.items():
        print(f"  {det:8} AUPRC={m['AUPRC']:.3f} AUROC={m['AUROC']:.3f} "
              f"prec@rec80={m.get('prec@rec80', float('nan')):.3f} "
              f"alerts/1k={m.get('alerts_per_1000', float('nan')):.0f} "
              f"FP_hardneg={m.get('FP_rate_hardneg', float('nan')):.3f}")
    return res


def main():
    strat_tr = load(os.path.join(MP, "data", "train.jsonl"))
    strat_te = load(os.path.join(MP, "data", "test.jsonl"))
    td_tr = load(os.path.join(MP, "data_taskdisjoint", "train.jsonl"))
    td_te = load(os.path.join(MP, "data_taskdisjoint", "test.jsonl"))
    all_res = {}
    all_res["R1_same"] = run_rung("R1: same distribution (stratified)", strat_tr, strat_te)
    all_res["R2_newtask"] = run_rung("R2: new task, same tools (task-disjoint)", td_tr, td_te)
    # R3: leave-one-domain-out (genuine tool drift) -- one held-out domain as demo (medical)
    for held in ["medical", "banking"]:
        tr = [r for r in strat_tr + strat_te if r["agent"] != held]
        te = [r for r in strat_tr + strat_te if r["agent"] == held]
        all_res[f"R3_LODO_{held}"] = run_rung(f"R3: leave-one-domain-out (hold out {held}) -- tool drift", tr, te)
    json.dump(all_res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "baseline_results.json"), "w"), indent=2)
    print("\nwrote baseline_results.json")


if __name__ == "__main__":
    main()
