"""Experiment 1b: world-disjoint R1. Does reusing `world`s between train and test inflate R1?

Same detector, same A/B/C features, same evalkit framework. We pool the R1 corpus
(data/train + data/test), then split by WHOLE worlds so no world appears in both sides,
while (a) keeping all 5 domains in both sides -> tool coverage stays ~100% (isolates world
reuse from tool-vocabulary loss, unlike R3), and (b) matching the standard R1 test fraction
and class balance as closely as world-disjointness allows.

We run 5 world-assignment seeds (split variability) and, on seed 0, a full stratified bootstrap
(test-resampling CI). Thresholds are training-derived. Compare to standard R1 (A .777/B .830/C .879).
Run: AGENTDRIFT=<clone> python world_disjoint_r1.py
"""
import os, json, collections, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from featurise import load, build_vocab, vectorise
import evalkit as ek

MP = os.environ["AGENTDRIFT"]; HERE = os.path.dirname(os.path.abspath(__file__))
REPS = ["A", "B", "C"]; TEST_FRAC = 1881 / (8775 + 1881)  # match standard R1 (~0.1765)
SPLIT_SEEDS = [0, 1, 2, 3, 4]; B = 1000


def wid(r): return json.dumps(r["world"], sort_keys=True)
def ylab(recs): return np.array([int(r["category"] == "attacked") for r in recs])


def X_of(recs, vocab, rep):
    out = []
    for r in recs:
        a, b = vectorise(r, vocab)
        out.append(b if rep == "A" else a if rep == "B" else a + b)
    return np.array(out, float)


def fit(train, rep):
    vocab = build_vocab([r for r in train if r["category"] == "benign"])
    X = X_of(train, vocab, rep); y = ylab(train); sc = StandardScaler().fit(X)
    lr = LogisticRegression(max_iter=1000, class_weight="balanced").fit(sc.transform(X), y)
    s = lr.predict_proba(sc.transform(X))[:, 1]
    return vocab, sc, lr, ek.thr_at_recall(s[y == 1], 0.80)


def sco(model, recs, rep):
    vocab, sc, lr, t80 = model
    s = lr.predict_proba(sc.transform(X_of(recs, vocab, rep)))[:, 1]
    return {"score": s, "pred": {"t_r80": s >= t80, "t0.5": s >= 0.5}}


def world_disjoint_split(pool, seed):
    """Assign whole worlds GLOBALLY to test/train (a world -> exactly one side, even if it spans
    domains) until ~TEST_FRAC of all traces are in test. Worlds are visited in an order that
    interleaves domains, so every domain is represented on both sides."""
    rng = np.random.default_rng(seed)
    worlds = collections.defaultdict(list)                 # world -> trajectory indices (global)
    world_dom = {}                                         # world -> a representative domain
    for i, r in enumerate(pool):
        worlds[wid(pool[i])].append(i); world_dom.setdefault(wid(pool[i]), r["agent"])
    # order worlds round-robin by their representative domain, shuffled within each domain
    by_dom = collections.defaultdict(list)
    for w, d in world_dom.items():
        by_dom[d].append(w)
    for d in by_dom:
        rng.shuffle(by_dom[d])
    order, ptrs = [], {d: 0 for d in by_dom}
    doms = list(by_dom)
    while len(order) < len(worlds):
        for d in doms:
            if ptrs[d] < len(by_dom[d]):
                order.append(by_dom[d][ptrs[d]]); ptrs[d] += 1
    target = TEST_FRAC * len(pool); acc = 0; test_idx = set()
    for w in order:
        if acc >= target:
            break
        test_idx.update(worlds[w]); acc += len(worlds[w])
    train = [i for i in range(len(pool)) if i not in test_idx]
    test = [i for i in range(len(pool)) if i in test_idx]
    return train, test


def coverage(train, test):
    V = {s["tool"] for r in train if r["category"] == "benign" for s in r["steps"]}
    occ = [s["tool"] in V for r in test for s in r["steps"]]
    return round(float(np.mean(occ)), 4)


def main():
    pool = load(os.path.join(MP, "data/train.jsonl")) + load(os.path.join(MP, "data/test.jsonl"))
    res = {"meta": {"reps": REPS, "test_frac_target": round(TEST_FRAC, 4), "n_pool": len(pool),
                    "standard_R1_AUROC": {"A": 0.777, "B": 0.830, "C": 0.879}}, "splits": []}
    per_seed = {r: [] for r in REPS}
    boot0 = None
    for sd in SPLIT_SEEDS:
        tr_i, te_i = world_disjoint_split(pool, sd)
        tr = [pool[i] for i in tr_i]; te = [pool[i] for i in te_i]
        # disjointness + stats
        wtr = {wid(r) for r in tr}; wte = {wid(r) for r in te}
        overlap = len(wtr & wte)
        ytr, yte = ylab(tr), ylab(te)
        models = {r: fit(tr, r) for r in REPS}
        methods = {r: [sco(models[r], te, r)] for r in REPS}
        pt = {r: ek.point(yte, methods[r]) for r in REPS}
        for r in REPS: per_seed[r].append(round(pt[r]["AUROC"], 4))
        info = {"seed": sd, "n_train": len(tr), "n_test": len(te),
                "world_overlap": overlap, "n_worlds_train": len(wtr), "n_worlds_test": len(wte),
                "train_attacked_frac": round(float(ytr.mean()), 4), "test_attacked_frac": round(float(yte.mean()), 4),
                "test_domains": sorted(collections.Counter(r["agent"] for r in te)),
                "tool_occ_coverage": coverage(tr, te),
                "AUROC": {r: round(pt[r]["AUROC"], 4) for r in REPS}}
        res["splits"].append(info)
        if sd == 0:
            bt = ek.bootstrap(yte, methods, B=B, seed=777)
            boot0 = {r: {k: [round(pt[r][k], 4)] + [round(x, 4) for x in ek.ci(bt[r][k])[1:]]
                         for k in bt[r]} for r in REPS}
            res["seed0_bootstrap"] = {"n_test": len(te), "n_pos": int(yte.sum()), "n_neg": int((yte == 0).sum()),
                                      "test_attacked_frac": round(float(yte.mean()), 4), "metrics": boot0}
        print(f"seed {sd}: overlap={overlap} cov={info['tool_occ_coverage']} "
              f"A={pt['A']['AUROC']:.3f} B={pt['B']['AUROC']:.3f} C={pt['C']['AUROC']:.3f}", flush=True)
    res["across_seed_AUROC"] = {r: {"mean": round(float(np.mean(per_seed[r])), 4),
                                    "min": min(per_seed[r]), "max": max(per_seed[r]),
                                    "values": per_seed[r]} for r in REPS}
    json.dump(res, open(os.path.join(HERE, "world_disjoint_r1_results.json"), "w"), indent=1, default=float)
    print("\nacross-seed AUROC:", {r: res["across_seed_AUROC"][r]["mean"] for r in REPS})
    print("wrote world_disjoint_r1_results.json")


if __name__ == "__main__":
    main()
