"""Read-only data audit: world, domain and label diagnostics for AgentDrift (no detector experiments).

Restored for the revision (review comments m5, m13, m14, m15; REVISION_ANALYSIS_PLAN.md Sec. 4, "Diagnostics").
The v0.3 file ended mid-line at line 44 and did not parse; a byte copy of it is kept in
revision/pre_ablation_audit_v03_truncated.py.txt. This rewrite keeps its purpose:

Part A  tool-vocabulary coverage (R1, R2, R3 per held-out domain, R4; overall and by class), the share of attacked
        training traces that contain tools absent from the benign vocabulary, and the structure of the AgentDrift
        `world` field versus the binary label (attacked vs not): world count, majority-label purity, largest worlds,
        worlds with >= 30 trajectories, world reuse between R1 train and test and how well the training-world attack
        rate predicts the test label, attack prevalence by domain.
Part B  probes: can representation A (bag of tools), B (10 behavioural features) or B-minus (B without the two
        keyword state-changing features) identify the DOMAIN or the WORLD? Regularised multinomial logistic
        regression with nested cross-validation (outer 5-fold stratified, 3 repetitions; inner 3-fold stratified
        grid search over C), with floors: majority class, and for the world probe a domain-only classifier.
Part C  benchmark provenance (m13): lines of the AgentDrift README / CHANGELOG / CITATION / datasheet and the AgentDojo
        licence, with line numbers, plus re-computations of the README's documented regularities.

Each section is saved to revision/diagnostics_parts/<section>.json together with the SHA-256 of this file; a section
is skipped when its part exists and was written by the same version of the file (--force recomputes). A call stops
starting new sections once --budget seconds are used; call it again until it prints "ASSEMBLED". The final step
writes revision/diagnostics_results.json (bootstrap intervals are computed there, from the stored out-of-fold
predictions), and copies revision/repro_checks.json (written by revision/repro_checks.py) under "m14" if present.

Statistics: 95% percentile bootstrap intervals, B = 2000, fixed seeds (SEED_BASE + offset, listed in the results
"meta"). Proportions and accuracies: trajectory-level bootstrap; world purity: world-cluster bootstrap; AUROC of the
training-world attack rate: evalkit.bootstrap (stratified, trajectory level) with a test-world cluster bootstrap as a
sensitivity analysis. Descriptive counts of the fixed corpus (numbers of worlds, sizes) are exact and have no interval.

Run (Git Bash, from agent-ids/):
  AGENTDRIFT=<AgentDrift clone> AGENTDOJO=<agentdojo repo> C:/Python310/python.exe pre_ablation_audit.py [--section S ...]
      [--force] [--budget SECONDS] [--status]
"""
import os, sys, re, json, time, hashlib, argparse, collections, itertools, subprocess, shutil, warnings, platform
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.exceptions import ConvergenceWarning
from featurise import load, vectorise, tool_seq
import evalkit as ek

HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.join(HERE, "revision")
PARTS = os.path.join(REV, "diagnostics_parts")
OUT = os.path.join(REV, "diagnostics_results.json")
REPRO = os.path.join(REV, "repro_checks.json")
MP = os.environ["AGENTDRIFT"]
ADJ = os.environ.get("AGENTDOJO", "")
KW = (4, 5)                        # behavioural features 4, 5 = keyword-based state-changing count / fraction
DOMAINS = ["medical", "banking", "web", "email", "coding"]
PATTERNS = ["attacked_full", "attacked_partial", "attacked_delayed"]
B = 2000                           # bootstrap replicates
SEED_BASE = 20260926               # bootstrap seeds are SEED_BASE + a fixed offset per quantity (see SEEDS)
C_GRID = [float(c) for c in np.logspace(-5, 3, 9)]   # inner grid for the probes' inverse regularisation strength
OUTER_K, INNER_K, OUTER_REPS = 5, 3, [0, 1, 2]
MIN_WORLD = 30                     # world probe: worlds with at least this many trajectories (as v0.3)
N_JOBS = int(os.environ.get("AUDIT_NJOBS", "4"))   # GridSearchCV workers (results do not depend on it)
MIN_WORLD_NONEMAIL = 10            # supplementary world probe on the four label-pure domains (not in v0.3)
SEEDS = {"purity_cluster": 1, "loo_cluster": 2, "prevalence": 3, "overlap_share": 4, "overlap_auroc": 5,
         "overlap_cluster": 6, "lookup_acc": 7, "probe_domain": 10, "probe_domain_benign": 11, "probe_world": 12,
         "templates": 13, "probe_world_nonemail": 14}
MANUSCRIPT = {  # values stated in MANUSCRIPT_DRAFT.md v0.3 (Sec. III-A line 80, Sec. VI-B line 261) and PRE_ABLATION_DESIGN.md
    "n_worlds": 1029, "purity": 0.858, "majority_baseline": 0.558, "largest3": [194, 169, 144], "n_worlds_ge30": 54,
    "n_traj_ge30": 2724, "r1_share_world_in_train": 0.993, "r1_world_rate_auroc": 0.963, "r1_base_rate": 0.441,
    "domain_probe": {"A": 1.000, "B": 0.455, "B-minus": 0.381, "majority": 0.207},
    "world_probe": {"A": 0.066, "B": 0.046, "B-minus": 0.043, "majority": 0.044, "domain_only": 0.097}}


# ----------------------------------------------------------------------------------------------------- utilities
def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _compute_sha():
    """SHA-256 of the part of this file that computes the sections (everything above the ASSEMBLY marker)."""
    txt = open(os.path.abspath(__file__), "rb").read()
    cut = txt.find(b"\n# ==== ASSEMBLY: code below")
    return hashlib.sha256(txt if cut < 0 else txt[:cut]).hexdigest()


SCRIPT_SHA = _compute_sha()                  # identifies the version of the section code (stored in each part)
FILE_SHA = sha256_file(os.path.abspath(__file__))


def r4(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)


def trip(point, boot):
    b = np.asarray(boot, float); b = b[~np.isnan(b)]
    return [r4(point), r4(np.percentile(b, 2.5)), r4(np.percentile(b, 97.5))]


def boot_mean(x, seed, B=B):
    """[mean, 2.5%, 97.5%] of a per-trajectory quantity (trajectory-level bootstrap)."""
    x = np.asarray(x, float); rng = np.random.default_rng(seed); n = len(x)
    bs = np.array([x[rng.integers(0, n, n)].mean() for _ in range(B)])
    return trip(x.mean(), bs)


def wid(r): return json.dumps(r["world"], sort_keys=True)          # the world key used by world_disjoint_r1.py
def lab(r): return int(r["category"] == "attacked")


_D = {}


def data():
    if "pool" not in _D:
        j = lambda *p: os.path.join(MP, *p)
        _D["train"] = load(j("data", "train.jsonl")); _D["val"] = load(j("data", "val.jsonl"))
        _D["test"] = load(j("data", "test.jsonl"))
        _D["td_train"] = load(j("data_taskdisjoint", "train.jsonl")); _D["td_test"] = load(j("data_taskdisjoint", "test.jsonl"))
        _D["pool"] = _D["train"] + _D["test"]                       # 10,656: the pool of R1-wd, R3 and R4 training
        _D["full"] = _D["train"] + _D["val"] + _D["test"]          # 12,536: the whole corpus
    return _D


def dojo():
    if "p0" not in _D:
        _D["p0"] = [json.loads(l) for l in open(os.path.join(HERE, "agentdojo_traces.jsonl"), encoding="utf-8")]
        _D["failed"] = [json.loads(l) for l in open(os.path.join(HERE, "agentdojo_failed.jsonl"), encoding="utf-8")]
    return _D["p0"], _D["failed"]


# ----------------------------------------------------------------------------------------------- Part A: coverage
def sec_coverage():
    D = data(); pool = D["pool"]; p0, failed = dojo()

    def V(train): return {s["tool"] for r in train if r["category"] == "benign" for s in r["steps"]}

    def cov(Vs, recs):
        if not recs: return None
        occ = [s["tool"] in Vs for r in recs for s in r["steps"]]
        per = [np.mean([s["tool"] in Vs for s in r["steps"]]) for r in recs]
        return {"n_traces": len(recs), "occ_coverage": r4(np.mean(occ)),
                "frac_traces_zero_known": r4(np.mean([p == 0 for p in per])),
                "frac_traces_all_known": r4(np.mean([p == 1 for p in per]))}

    def by_class(Vs, recs):
        return {"all": cov(Vs, recs), "attacked": cov(Vs, [r for r in recs if lab(r) == 1]),
                "not_attacked": cov(Vs, [r for r in recs if lab(r) == 0])}

    def unseen_in_attacked_train(train, Vs):
        att = [r for r in train if lab(r) == 1]
        return r4(np.mean([any(s["tool"] not in Vs for s in r["steps"]) for r in att])) if att else None

    settings = {"R1": (D["train"], D["test"]), "R2": (D["td_train"], D["td_test"])}
    for d in DOMAINS:
        settings[f"R3_{d}"] = ([r for r in pool if r["agent"] != d], [r for r in pool if r["agent"] == d])
    out = {}
    for s, (tr, te) in settings.items():
        Vs = V(tr)
        out[s] = {"benign_vocab_size": len(Vs), "test": by_class(Vs, te),
                  "frac_attacked_train_with_tool_outside_benign_vocab": unseen_in_attacked_train(tr, Vs)}
    Vs = V(pool)
    out["R4"] = {"benign_vocab_size": len(Vs), "test_P0": by_class(Vs, p0), "test_failed": cov(Vs, failed),
                 "frac_attacked_train_with_tool_outside_benign_vocab": unseen_in_attacked_train(pool, Vs),
                 "n_agentdojo_tools_P0_and_failed": len({s["tool"] for r in p0 + failed for s in r["steps"]}),
                 "n_shared_tools_P0_and_failed": len({s["tool"] for r in p0 + failed for s in r["steps"]} & Vs),
                 "n_agentdojo_tools_P0": len({s["tool"] for r in p0 for s in r["steps"]}),
                 "n_shared_tools_P0": len({s["tool"] for r in p0 for s in r["steps"]} & Vs)}
    out["tools_disjoint_across_domains"] = all(
        not ({s["tool"] for r in pool if r["agent"] == a for s in r["steps"]} & {s["tool"] for r in pool if r["agent"] == b for s in r["steps"]})
        for a, b in itertools.combinations(DOMAINS, 2))
    out["n_tools_pool"] = len({s["tool"] for r in pool for s in r["steps"]})
    return out


# --------------------------------------------------------------------------------------------- Part A: worlds
def group_stats(recs, key, labf=lab):
    g = collections.defaultdict(list)
    for r in recs:
        g[key(r)].append(labf(r))
    sizes = sorted((len(v) for v in g.values()), reverse=True)
    correct = sum(collections.Counter(v).most_common(1)[0][1] for v in g.values())
    return g, sizes, correct / len(recs)


def purity_cluster_ci(g, seed):
    """World-cluster bootstrap of in-sample majority-label purity (resample worlds with replacement)."""
    n_i = np.array([len(v) for v in g.values()]); m_i = np.array([collections.Counter(v).most_common(1)[0][1] for v in g.values()])
    rng = np.random.default_rng(seed); k = len(n_i)
    bs = []
    for _ in range(B):
        c = rng.integers(0, k, k); bs.append(m_i[c].sum() / n_i[c].sum())
    return trip(m_i.sum() / n_i.sum(), bs)


def loo_world_majority(recs, seed):
    """Leave-one-out world lookup: predict each trajectory's label by the majority label of the OTHER trajectories of
    its world (ties and singleton worlds -> 0, the overall majority class). Interval: world-cluster bootstrap."""
    g = collections.defaultdict(list)
    for i, r in enumerate(recs):
        g[wid(r)].append(i)
    y = np.array([lab(r) for r in recs]); corr = np.zeros(len(recs))
    for idx in g.values():
        ys = y[idx]; s = ys.sum(); n = len(idx)
        for j, i in enumerate(idx):
            s1 = s - ys[j]; n1 = n - 1
            pred = 1 if (n1 > 0 and s1 > n1 - s1) else 0
            corr[i] = float(pred == y[i])
    groups = list(g.values()); rng = np.random.default_rng(seed); k = len(groups)
    tot = np.array([corr[idx].sum() for idx in groups]); cnt = np.array([len(idx) for idx in groups])
    bs = []
    for _ in range(B):
        c = rng.integers(0, k, k); bs.append(tot[c].sum() / cnt[c].sum())
    return trip(corr.mean(), bs)


def world_block(recs, seed_off=0):
    g, sizes, pur = group_stats(recs, wid)
    gc, _, pur_cat = group_stats(recs, wid, labf=lambda r: r["category"])
    y = np.array([lab(r) for r in recs])
    doms_of = collections.defaultdict(set)
    for r in recs:
        doms_of[wid(r)].add(r["agent"])
    ge = [s for s in sizes if s >= MIN_WORLD]
    blk = {"n_trajectories": len(recs), "n_worlds": len(g),
           "purity_binary": purity_cluster_ci(g, SEED_BASE + SEEDS["purity_cluster"] + seed_off),
           "purity_category4": r4(pur_cat),
           "loo_world_majority_accuracy": loo_world_majority(recs, SEED_BASE + SEEDS["loo_cluster"] + seed_off),
           "attacked_rate": r4(y.mean()), "majority_baseline": r4(max(y.mean(), 1 - y.mean())),
           "largest_world_sizes": sizes[:3], "largest_10": sizes[:10],
           "n_worlds_ge30": len(ge), "n_traj_in_worlds_ge30": int(sum(ge)),
           "n_singleton_worlds": int(sum(1 for s in sizes if s == 1)),
           "world_size_quantiles_25_50_75": [float(q) for q in np.percentile(sizes, [25, 50, 75])],
           "n_worlds_spanning_more_than_one_domain": int(sum(1 for v in doms_of.values() if len(v) > 1)),
           "per_domain": {}}
    for d in DOMAINS:
        sub = [r for r in recs if r["agent"] == d]
        gd, sd, pd = group_stats(sub, wid)
        blk["per_domain"][d] = {"n": len(sub), "n_worlds": len(gd), "purity_binary": r4(pd), "largest_3": sd[:3],
                                "n_worlds_ge30": int(sum(1 for s in sd if s >= MIN_WORLD))}
    return blk


def sec_worlds():
    D = data()
    out = {"key": "json.dumps(world, sort_keys=True) (world_disjoint_r1.wid; the key of the world-disjoint split)",
           "label": "attacked (category == 'attacked') vs not; failed attacks, hard negatives and benign count as 0",
           "pool_10656": world_block(D["pool"]), "full_12536": world_block(D["full"], 100),
           "train_8775": world_block(D["train"], 200), "test_1881": world_block(D["test"], 300)}
    # the world-disjoint split as run in v0.3 (world counts per side; overlap must be 0)
    from world_disjoint_r1 import world_disjoint_split
    wd = {}
    for sd in range(5):
        tr_i, te_i = world_disjoint_split(D["pool"], sd)
        wtr = {wid(D["pool"][i]) for i in tr_i}; wte = {wid(D["pool"][i]) for i in te_i}
        wd[f"seed{sd}"] = {"n_train": len(tr_i), "n_test": len(te_i), "n_worlds_train": len(wtr), "n_worlds_test": len(wte),
                           "overlap": len(wtr & wte), "n_worlds_total": len(wtr | wte)}
    out["world_disjoint_splits"] = wd
    return out


# ------------------------------------------------------------------------------------------- Part A: key search
CON = {"full": lambda cs: json.dumps(cs, sort_keys=True),
       "names": lambda cs: json.dumps([c.get("name") for c in cs]),
       "emails": lambda cs: json.dumps([c.get("email") for c in cs]),
       "names_sorted": lambda cs: json.dumps(sorted(str(c.get("name")) for c in cs)),
       "set_full": lambda cs: json.dumps(sorted(json.dumps(c, sort_keys=True) for c in cs)),
       "first": lambda cs: json.dumps(cs[0] if cs else None, sort_keys=True),
       "count": lambda cs: str(len(cs)),
       "relations": lambda cs: json.dumps([c.get("relation") for c in cs])}
FIELDS = ["user", "email", "company", "date"]


def candidate_keys():
    keys = {}
    for k in range(0, len(FIELDS) + 1):
        for sub in itertools.combinations(FIELDS, k):
            for cm in [None] + list(CON):
                if not sub and cm is None:
                    continue
                name = "+".join(sub) + ("" if cm is None else ("+" if sub else "") + f"contacts[{cm}]")
                keys[name] = (sub, cm)
    return keys


def sec_key_search():
    D = data()
    pops = {"pool_10656": D["pool"], "full_12536": D["full"], "train_8775": D["train"],
            "train+val_10655": D["train"] + D["val"], "taskdisjoint_train+test": D["td_train"] + D["td_test"],
            "pool_without_failed": [r for r in D["pool"] if r["category"] != "failed_attack"]}
    keys = candidate_keys()
    ms = MANUSCRIPT
    rows = []
    for pn, recs in pops.items():
        for kn, (sub, cm) in keys.items():
            g = collections.defaultdict(list)
            for r in recs:
                w = r["world"]; parts = [str(w.get(x)) for x in sub]
                if cm: parts.append(CON[cm](w.get("contacts", []) or []))
                g["\x1f".join(parts)].append(r)
            sizes = sorted((len(v) for v in g.values()), reverse=True)
            ge = [s for s in sizes if s >= MIN_WORLD]
            for ln, lf in [("attacked", lab), ("category4", lambda r: r["category"])]:
                pur = sum(collections.Counter(lf(r) for r in v).most_common(1)[0][1] for v in g.values()) / len(recs)
                row = {"population": pn, "n": len(recs), "key": kn, "label": ln, "n_worlds": len(g), "purity": round(pur, 4),
                       "largest3": sizes[:3], "n_worlds_ge30": len(ge), "n_traj_ge30": int(sum(ge))}
                row["match"] = {"n_worlds_1029": len(g) == ms["n_worlds"], "purity_0.858": round(pur, 3) == ms["purity"],
                                "largest_194_169_144": sizes[:3] == ms["largest3"],
                                "ge30_54_2724": len(ge) == ms["n_worlds_ge30"] and int(sum(ge)) == ms["n_traj_ge30"]}
                row["n_matches"] = int(sum(row["match"].values()))
                rows.append(row)
    best = max(r["n_matches"] for r in rows)
    out = {"n_candidate_keys": len(keys), "populations": {p: len(v) for p, v in pops.items()},
           "labels": ["attacked", "category4"], "n_rows": len(rows),
           "key_grammar": "any subset of world fields {user, email, company, date} (string values) plus optionally the contacts "
                          "list represented as: " + ", ".join(CON),
           "targets": {k: ms[k] for k in ["n_worlds", "purity", "largest3", "n_worlds_ge30", "n_traj_ge30"]},
           "max_criteria_matched_by_any_row": best,
           "rows_matching_all_four": [r for r in rows if r["n_matches"] == 4],
           "rows_with_most_matches": [r for r in rows if r["n_matches"] == best][:40],
           "rows_n_worlds_1029": [r for r in rows if r["match"]["n_worlds_1029"]],
           "rows_largest_194_169_144": [r for r in rows if r["match"]["largest_194_169_144"]],
           "rows_ge30_54_2724": [r for r in rows if r["match"]["ge30_54_2724"]],
           "n_rows_purity_0.858": int(sum(r["match"]["purity_0.858"] for r in rows)),
           "table_pool_10656_attacked": [r for r in rows if r["population"] == "pool_10656" and r["label"] == "attacked"]}
    # a numerical coincidence worth recording (not evidence of provenance): 1,029 is also the number of negatives in the
    # seed-0 world-disjoint test partition (world_disjoint_r1_results.json, seed0_bootstrap.n_neg)
    try:
        wdres = json.load(open(os.path.join(HERE, "world_disjoint_r1_results.json"), encoding="utf-8"))
        out["coincidence_world_disjoint_seed0_n_neg"] = wdres["seed0_bootstrap"]["n_neg"]
    except Exception as e:  # noqa: BLE001
        out["coincidence_world_disjoint_seed0_n_neg"] = f"unavailable: {e}"
    return out


# --------------------------------------------------------------------------------- Part A: world reuse R1 / R2
def overlap_block(train, test, seed_off):
    trw = collections.defaultdict(list)
    for r in train:
        trw[wid(r)].append(lab(r))
    out = {}
    for pname, recs in [("all_test", test), ("pool_E", [r for r in test if r["category"] != "failed_attack"])]:
        y = np.array([lab(r) for r in recs]); w = [wid(r) for r in recs]
        inn = np.array([x in trw for x in w])
        blk = {"n": len(recs), "n_world_in_train": int(inn.sum()),
               "share_world_in_train": boot_mean(inn.astype(float), SEED_BASE + SEEDS["overlap_share"] + seed_off)}
        idx = np.where(inn)[0]
        if len(idx) and 0 < y[idx].sum() < len(idx):
            rate = np.array([np.mean(trw[w[i]]) for i in idx]); yy = y[idx]
            bt = ek.bootstrap(yy, {"rate": [{"score": rate}]}, B=B, seed=SEED_BASE + SEEDS["overlap_auroc"] + seed_off)
            # test-world cluster bootstrap (sensitivity)
            wk = np.array([w[i] for i in idx]); uq, inv = np.unique(wk, return_inverse=True)
            mem = [np.where(inv == c)[0] for c in range(len(uq))]
            rng = np.random.default_rng(SEED_BASE + SEEDS["overlap_cluster"] + seed_off); cb = []
            for _ in range(B):
                ib = np.concatenate([mem[c] for c in rng.integers(0, len(uq), len(uq))])
                cb.append(ek.auroc(yy[ib], rate[ib]))
            pred = (rate > 0.5).astype(int)                     # training-world majority label (ties -> 0)
            corr = (pred == yy).astype(float)
            blk.update({"base_rate_on_subset": r4(yy.mean()),
                        "auroc_train_world_attack_rate": trip(ek.auroc(yy, rate), bt["rate"]["AUROC"]),
                        "auroc_train_world_attack_rate_world_cluster_ci": trip(ek.auroc(yy, rate), cb),
                        "n_test_worlds_on_subset": int(len(uq)),
                        "world_lookup_accuracy": boot_mean(corr, SEED_BASE + SEEDS["lookup_acc"] + seed_off),
                        "per_domain": {}})
            doms = np.array([recs[i]["agent"] for i in idx])
            for di, d in enumerate(DOMAINS):
                m = doms == d
                if m.sum() == 0: continue
                blk["per_domain"][d] = {"n": int(m.sum()), "base_rate": r4(yy[m].mean()),
                                        "auroc": r4(ek.auroc(yy[m], rate[m])) if 0 < yy[m].sum() < m.sum() else None,
                                        "world_lookup_accuracy": boot_mean(corr[m], SEED_BASE + SEEDS["lookup_acc"] + seed_off + 10 + di)}
        out[pname] = blk
    return out


def sec_r1_overlap():
    D = data()
    return {"note": "score = attack rate (share of category 'attacked') of the trajectory's world among TRAINING trajectories; "
                    "computed on the test trajectories whose world occurs in training. all_test includes failed attacks "
                    "(label 0), as the v0.3 diagnostic (base rate 0.441); pool_E excludes them.",
            "R1": overlap_block(D["train"], D["test"], 0), "R2": overlap_block(D["td_train"], D["td_test"], 50)}


# --------------------------------------------------------------------------------- Part A: prevalence by domain
def sec_prevalence():
    D = data(); out = {}
    pops = {"pool_10656_all_categories": D["pool"],
            "pool_10656_pool_E": [r for r in D["pool"] if r["category"] != "failed_attack"],
            "full_12536_all_categories": D["full"],
            "R1_test_pool_E": [r for r in D["test"] if r["category"] != "failed_attack"],
            "R1_train_all_categories": D["train"]}
    for pi, (pn, recs) in enumerate(pops.items()):
        blk = {}
        for di, d in enumerate(["all"] + DOMAINS):
            sub = recs if d == "all" else [r for r in recs if r["agent"] == d]
            y = np.array([lab(r) for r in sub], float)
            cat = collections.Counter(r["category"] for r in sub)
            blk[d] = {"n": len(sub), "n_attacked": int(y.sum()),
                      "prevalence": boot_mean(y, SEED_BASE + SEEDS["prevalence"] + 100 * pi + di),
                      "categories": dict(sorted(cat.items()))}
        out[pn] = blk
    # hijack-pattern composition of the positives by domain (m13: the banking fold differs in hijack mix)
    comp = {}
    for pi, (pn, recs) in enumerate([("pool_10656", D["pool"]), ("full_12536", D["full"]), ("R1_test", D["test"])]):
        blk = {}
        for di, d in enumerate(["all"] + DOMAINS):
            att = [r for r in recs if lab(r) == 1 and (d == "all" or r["agent"] == d)]
            c = collections.Counter(r["source_category"] for r in att)
            full = np.array([r["source_category"] == "attacked_full" for r in att], float)
            blk[d] = {"n_attacked": len(att), **{p: int(c.get(p, 0)) for p in PATTERNS},
                      "share_full": boot_mean(full, SEED_BASE + SEEDS["prevalence"] + 1000 + 100 * pi + di)}
        comp[pn] = blk
    out["hijack_pattern_composition"] = comp
    return out


# ------------------------------------------------------------------------------ Part C: documented regularities
EXT_RE = re.compile(r"[\w.+-]+@ext-audit\.com", re.I)
AMT_RE = re.compile(r"\$\s?5,?000(?![\d])")
AUD_RE = re.compile(r"\baudit@([\w-]+(?:\.[\w-]+)+)", re.I)
TRF_RE = re.compile(r"transfer", re.I)


def rec_text(r):
    parts = [r.get("task", "")]
    for s in r["steps"]:
        parts += [str(s.get("thought", "")), str(s.get("tool", "")), json.dumps(s.get("args", {})), str(s.get("obs", ""))]
    return "\n".join(parts)


def sec_templates():
    D = data(); out = {}
    for pn, recs in [("full_12536", D["full"]), ("pool_10656", D["pool"])]:
        inj = [s for r in recs for s in r["steps"] if s.get("label") == "injection_point"]
        inj_f = [s for r in recs for s in r["steps"] if s.get("label") in ("injection_point", "failed_injection")]
        a = np.array([bool(EXT_RE.search(str(s.get("obs", "")))) for s in inj], float)
        a2 = np.array(["ext-audit.com" in str(s.get("obs", "")).lower() for s in inj], float)
        b = np.array([bool(EXT_RE.search(str(s.get("obs", "")))) for s in inj_f], float)
        hn = [r for r in recs if r["category"] == "hard_negative"]
        amt, aud, trf, amt_o, aud_o, trf_o = [], [], [], [], [], []
        for r in hn:
            t = rec_text(r); dom = str(r["world"].get("email", "")).split("@")[-1].lower().strip()
            o = "\n".join(str(s.get("obs", "")) for s in r["steps"])
            for txt, A_, U_, T_ in [(t, amt, aud, trf), (o, amt_o, aud_o, trf_o)]:
                A_.append(bool(AMT_RE.search(txt)))
                U_.append(any(m.group(1).lower().strip(". ") == dom for m in AUD_RE.finditer(txt)))
                T_.append(bool(TRF_RE.search(txt)))
        amt, aud, trf, amt_o, aud_o, trf_o = (np.array(x, float) for x in (amt, aud, trf, amt_o, aud_o, trf_o))
        uni = np.maximum(np.maximum(amt, aud), trf); uni_o = np.maximum(np.maximum(amt_o, aud_o), trf_o)
        sd = SEED_BASE + SEEDS["templates"] + (0 if pn.startswith("full") else 50)
        pos = collections.Counter()
        for r in recs:
            for k, s in enumerate(r["steps"]):
                if s.get("label") in ("injection_point", "failed_injection"):
                    pos[k + 1] += 1
        out[pn] = {"injection_point_observations": {"n": len(inj), "share_naming_ext_audit_com_address": boot_mean(a, sd),
                                                    "share_containing_substring_ext_audit_com": boot_mean(a2, sd + 6)},
                   "injection_point_or_failed_injection_observations": {"n": len(inj_f), "share_naming_ext_audit_com_address": boot_mean(b, sd + 1)},
                   "hard_negatives": {"n": len(hn), "share_5000_amount": boot_mean(amt, sd + 2),
                                      "share_in_domain_audit_address": boot_mean(aud, sd + 3),
                                      "share_mention_transfer": boot_mean(trf, sd + 4),
                                      "share_any_of_three": boot_mean(uni, sd + 5),
                                      "observations_only": {"share_5000_amount": boot_mean(amt_o, sd + 7),
                                                            "share_in_domain_audit_address": boot_mean(aud_o, sd + 8),
                                                            "share_mention_transfer": boot_mean(trf_o, sd + 9),
                                                            "share_any_of_three": boot_mean(uni_o, sd + 10)}},
                   "poison_step_position_1based": dict(sorted(pos.items()))}
    full = D["full"]
    out["banking_source_categories_full_12536"] = dict(sorted(collections.Counter(r["source_category"] for r in full if r["agent"] == "banking").items()))
    out["domain_sizes_full_12536"] = dict(sorted(collections.Counter(r["agent"] for r in full).items()))
    out["attacked_full_by_domain_full_12536"] = {d: int(sum(1 for r in full if r["agent"] == d and r["source_category"] == "attacked_full")) for d in DOMAINS}
    em = [e for r in full for e in [r["world"].get("email", "")] + [c.get("email", "") for c in r["world"].get("contacts", [])]]
    out["world_email_addresses_full_12536"] = {"n": len(em), "n_with_space": int(sum(" " in str(e) for e in em))}
    out["definitions"] = ("ext-audit: an address matching [\\w.+-]+@ext-audit.com in the observation of steps labelled "
                          "injection_point (or also failed_injection). Hard negatives: text = task + every step's thought, tool, "
                          "json(args), obs; '$5,000' = \\$\\s?5,?000; in-domain audit@ = audit@<domain of the world's user email>; "
                          "transfer = case-insensitive substring 'transfer'; observations_only uses the observation texts only. The README "
                          "does not define its terms, so these "
                          "are approximations.")
    return out


# ------------------------------------------------------------------------------------------------ Part B: probes
def feature_mats(recs):
    tools = sorted({s["tool"] for r in recs for s in r["steps"]}); ti = {t: i for i, t in enumerate(tools)}
    XA = np.zeros((len(recs), len(tools)))
    for i, r in enumerate(recs):
        for t in tool_seq(r):
            XA[i, ti[t]] += 1.0
    XB = np.array([vectorise(r, {})[0] for r in recs], float)
    keep = [j for j in range(XB.shape[1]) if j not in KW]
    return {"A": XA, "B": XB, "B-minus": XB[:, keep], "C": np.hstack([XB, XA])}, len(tools)


def onehot(vals):
    u = sorted(set(vals)); ix = {v: i for i, v in enumerate(u)}
    X = np.zeros((len(vals), len(u)))
    for i, v in enumerate(vals):
        X[i, ix[v]] = 1.0
    return X


def run_nested(X, y, bag_cols=None, kind="lr", reps=None):
    """Out-of-fold predictions of a nested-CV probe. kind 'lr': StandardScaler + L2 multinomial LogisticRegression (lbfgs),
    C chosen by inner StratifiedKFold(3) grid search on accuracy; kind 'majority': training-fold majority class.
    bag_cols: boolean mask of bag-of-tools columns; those whose tool never occurs in the outer training fold are dropped
    (the vocabulary is built from the outer training fold only)."""
    reps = OUTER_REPS if reps is None else reps
    n = len(y); correct = np.zeros((len(reps), n), np.int8); chosen = []; nwarn = 0; n_edge = 0
    for ri, rs in enumerate(reps):
        outer = StratifiedKFold(OUTER_K, shuffle=True, random_state=rs)
        for f, (tr, te) in enumerate(outer.split(np.zeros(n), y)):
            if kind == "majority":
                vals, cnts = np.unique(y[tr], return_counts=True); p = np.full(len(te), vals[np.argmax(cnts)])
            else:
                cols = np.ones(X.shape[1], bool)
                if bag_cols is not None:
                    cols = ~bag_cols | (X[tr].sum(0) > 0)
                gs = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
                                  {"logisticregression__C": C_GRID},
                                  cv=StratifiedKFold(INNER_K, shuffle=True, random_state=100 + 10 * rs + f),
                                  scoring="accuracy", n_jobs=N_JOBS, refit=True)
                with warnings.catch_warnings(record=True) as wl:
                    warnings.simplefilter("always", ConvergenceWarning)
                    gs.fit(X[tr][:, cols], y[tr])
                nwarn += sum(1 for w in wl if issubclass(w.category, ConvergenceWarning))
                c = float(gs.best_params_["logisticregression__C"]); chosen.append(c)
                n_edge += int(c in (C_GRID[0], C_GRID[-1]))
                p = gs.predict(X[te][:, cols])
            correct[ri, te] = (p == y[te])
    return {"correct": correct.tolist(), "C_chosen": chosen, "n_convergence_warnings_refit": nwarn,
            "n_folds_C_at_grid_edge": n_edge, "acc_per_rep": [r4(c.mean()) for c in correct], "outer_reps": list(reps)}


def probe_inputs(which):
    D = data(); pool = D["pool"]
    if which == "domain":
        recs = pool
    elif which == "domain_benign":
        recs = [r for r in pool if r["category"] == "benign"]
    elif which == "world":
        cnt = collections.Counter(wid(r) for r in pool)
        big = {w for w, c in cnt.items() if c >= MIN_WORLD}
        recs = [r for r in pool if wid(r) in big]
    else:  # world_nonemail: trajectories of medical/banking/web/coding whose world has >= 10 trajectories there
        ne = [r for r in pool if r["agent"] != "email"]
        cnt = collections.Counter(wid(r) for r in ne)
        big = {w for w, c in cnt.items() if c >= MIN_WORLD_NONEMAIL}
        recs = [r for r in ne if wid(r) in big]
    mats, ntools = feature_mats(recs)
    y = np.array([r["agent"] for r in recs]) if which.startswith("domain") else np.array([wid(r) for r in recs])
    return recs, mats, y


def population_info(recs, which):
    """Composition of a probe population: domains, worlds, label purity (to read the probe against the world leakage)."""
    doms = collections.Counter(r["agent"] for r in recs)
    g, sizes, pur = group_stats(recs, wid)
    dom_sets = None
    if which.startswith("world"):
        wd = collections.defaultdict(set)
        for r in recs:
            wd[wid(r)].add(r["agent"])
        dom_sets = collections.Counter("+".join(sorted(v)) for v in wd.values())
    return {"n": len(recs), "domains": dict(sorted(doms.items())), "share_email": r4(doms.get("email", 0) / len(recs)),
            "n_worlds": len(g), "world_purity_binary": r4(pur), "attacked_rate": r4(np.mean([lab(r) for r in recs])),
            "world_domain_sets": dict(sorted(dom_sets.items())) if dom_sets else None}


PROBE_UNITS = {"domain": ["A", "B", "B-minus", "C", "majority"],
               "domain_benign": ["A", "B", "B-minus", "majority"],
               "world": ["A", "B", "B-minus", "C", "majority", "domain_only", "domain_x_label"],
               "world_nonemail": ["A", "B", "B-minus", "C", "majority", "domain_only", "domain_x_label"]}
PROBE_REPS = {"world_nonemail": [0]}   # one outer repetition for the 301-class supplementary probe (run time)


def make_probe_section(which, unit):
    def f():
        recs, mats, y = probe_inputs(which)
        t0 = time.time()
        reps = PROBE_REPS.get(which, OUTER_REPS)
        if unit == "majority":
            res = run_nested(None, y, kind="majority", reps=reps)
        elif unit == "domain_only":
            res = run_nested(onehot([r["agent"] for r in recs]), y, reps=reps)
        elif unit == "domain_x_label":
            res = run_nested(onehot([f"{r['agent']}|{lab(r)}" for r in recs]), y, reps=reps)
        else:
            X = mats[unit]; nb = mats["A"].shape[1]
            bag = None
            if unit == "A":
                bag = np.ones(nb, bool)
            elif unit == "C":
                bag = np.r_[np.zeros(mats["B"].shape[1], bool), np.ones(nb, bool)]
            res = run_nested(X, y, bag_cols=bag, reps=reps)
        res.update({"probe": which, "unit": unit, "n": len(y), "n_classes": int(len(set(y.tolist()))),
                    "seconds": r4(time.time() - t0)})
        return res
    return f


def sec_probe_v03():
    """Reconstruction attempt of the v0.3 probes (the v0.3 code is truncated before the probe): StandardScaler +
    LogisticRegression(max_iter=1000) at the default C = 1 (no tuning), stratified K-fold, vocabulary from the training
    fold. Four fold configurations are tried; the manuscript reports 3-fold, the v0.3 docstring 5-fold."""
    out = {}
    cfgs = {"3fold_shuffle_rs0": (3, True, 0), "3fold_noshuffle": (3, False, None),
            "5fold_shuffle_rs0": (5, True, 0), "5fold_noshuffle": (5, False, None)}
    for which in ["domain", "world"]:
        recs, mats, y = probe_inputs(which)
        units = {"A": mats["A"], "B": mats["B"], "B-minus": mats["B-minus"]}
        if which == "world":
            units["domain_only"] = onehot([r["agent"] for r in recs])
        blk = {"n": len(y), "n_classes": int(len(set(y.tolist()))),
               "majority_in_sample": r4(collections.Counter(y.tolist()).most_common(1)[0][1] / len(y))}
        for cn, (k, sh, rs) in cfgs.items():
            skf = StratifiedKFold(k, shuffle=sh, random_state=rs); row = {}
            for un, X in units.items():
                corr = np.zeros(len(y))
                for tr, te in skf.split(np.zeros(len(y)), y):
                    cols = (X[tr].sum(0) > 0) if un == "A" else np.ones(X.shape[1], bool)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", ConvergenceWarning)
                        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X[tr][:, cols], y[tr])
                    corr[te] = m.predict(X[te][:, cols]) == y[te]
                row[un] = r4(corr.mean())
            blk[cn] = row
        if which == "world":   # spread over fold assignments (3-fold, shuffled, random_state 0..9)
            spread = {un: [] for un in units}
            for rs in range(10):
                skf = StratifiedKFold(3, shuffle=True, random_state=rs)
                for un, X in units.items():
                    corr = np.zeros(len(y))
                    for tr, te in skf.split(np.zeros(len(y)), y):
                        cols = (X[tr].sum(0) > 0) if un == "A" else np.ones(X.shape[1], bool)
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", ConvergenceWarning)
                            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X[tr][:, cols], y[tr])
                        corr[te] = m.predict(X[te][:, cols]) == y[te]
                    spread[un].append(r4(corr.mean()))
            blk["fold_seed_spread_3fold_shuffle_rs0_to_9"] = {un: {"min": min(v), "max": max(v), "mean": r4(np.mean(v)), "values": v}
                                                              for un, v in spread.items()}
        tgt = MANUSCRIPT["domain_probe" if which == "domain" else "world_probe"]
        blk["manuscript_values"] = tgt
        blk["configs_reproducing_all_manuscript_values_to_3dp"] = [
            cn for cn in cfgs if all(round(blk[cn][u], 3) == round(tgt[u if u != "domain_only" else "domain_only"], 3)
                                     for u in blk[cn])]
        out[which] = blk
    return out


# ----------------------------------------------------------------------------------------- Part C: provenance
def grep_lines(path, patterns, maxlen=240):
    """[{pattern, line, text}] for every line of `path` matching a pattern (case-insensitive)."""
    out = []
    try:
        lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
    except OSError as e:
        return [{"error": str(e)}]
    for pat in patterns:
        rx = re.compile(pat, re.I)
        for i, t in enumerate(lines, 1):
            if rx.search(t):
                out.append({"pattern": pat, "line": i, "text": t.strip()[:maxlen]})
    return out


def git_head(repo):
    g = shutil.which("git")
    if g:
        try:
            h = subprocess.run([g, "-C", repo, "log", "-1", "--format=%H %cI"], capture_output=True, text=True, timeout=30)
            if h.returncode == 0 and h.stdout.strip():
                sha, date = h.stdout.strip().split(" ", 1)
                return {"commit": sha, "commit_date": date}
        except Exception:  # noqa: BLE001
            pass
    try:
        head = open(os.path.join(repo, ".git", "HEAD")).read().strip()
        if head.startswith("ref:"):
            return {"commit": open(os.path.join(repo, ".git", head[5:].strip())).read().strip(), "commit_date": None}
        return {"commit": head, "commit_date": None}
    except OSError as e:
        return {"error": str(e)}


def sec_provenance():
    D = data(); j = lambda *p: os.path.join(MP, *p)
    readme, changelog = j("README.md"), j("CHANGELOG.md")
    out = {"agentdrift": {
        "repo_path": MP, "git": git_head(MP),
        "readme_sha256": sha256_file(readme), "changelog_sha256": sha256_file(changelog),
        "generator_model": grep_lines(readme, [r"^\|\s*Generator\s*\|"]),
        "licence": grep_lines(readme, [r"^\|\s*License\s*\|", r"^\[CC BY 4\.0\]"]),
        "release_version_changelog": grep_lines(changelog, [r"^## "]),
        "changelog_full_text": open(changelog, encoding="utf-8").read().splitlines(),
        "record_version_field_in_schema": grep_lines(readme, [r'"version"\s*:']),
        "record_version_field_values_full_12536": dict(collections.Counter(str(r.get("version")) for r in D["full"])),
        "generation_code_availability": grep_lines(readme, [r"generation prompts and scripts", r"^of this release"]),
        "world_identity_leakage": grep_lines(readme, [r"World-identity leakage \(most important\)", r"93 to 98%"]),
        "template_concentration": grep_lines(readme, [r"Template concentration", r"75% of hard negatives"]),
        "attack_type_tags": grep_lines(readme, [r"Attack-type tags are metadata"]),
        "early_injection_position": grep_lines(readme, [r"Early injection position", r"falls at step 2 to 4"]),
        "banking_cutoff": grep_lines(readme, [r"Banking has 2,136", r"cut\s*$", r"off by the validator", r"256 accepted"]),
        "cosmetic_email_spaces": grep_lines(readme, [r"263 of 58,044"]),
        "worlds_in_corpus_count": grep_lines(readme, [r"distinct world objects"]),
        "real_organisation_note": grep_lines(readme, [r"Some generated company", r"coincide with real organi"]),
        "citation_cff_version_date_licence": grep_lines(j("CITATION.cff"), [r"^version:", r"^date-released:", r"^license:"]),
        "datasheet_generator": grep_lines(j("docs", "DATASHEET.md"), [r"Llama-3\.3-70B", r"Temperature"]),
        "datasheet_real_organisation_note": grep_lines(j("docs", "DATASHEET.md"), [r"coincide with real organi"]),
        "datasheet_worlds": grep_lines(j("docs", "DATASHEET.md"), [r"world objects \(1,141\)"]),
        "licence_file_first_line": open(j("LICENSE"), encoding="utf-8").readline().strip()}}
    try:
        gm = json.load(open(j("pools", "generation_metadata.json"), encoding="utf-8"))
        out["agentdrift"]["pools_generation_metadata"] = {k: gm.get(k) for k in ["generated_at", "model", "temperature"]}
    except OSError as e:
        out["agentdrift"]["pools_generation_metadata"] = {"error": str(e)}
    if ADJ and os.path.isdir(ADJ):
        lic = os.path.join(ADJ, "LICENSE")
        md_files = [os.path.join(ADJ, "README.md")] + [os.path.join(dp, f) for dp, _, fs in os.walk(os.path.join(ADJ, "docs")) for f in fs if f.endswith(".md")]
        note = []
        for p in md_files:
            for m in grep_lines(p, [r"coincide", r"real organi", r"real compan"]):
                m["file"] = os.path.relpath(p, ADJ).replace("\\", "/"); note.append(m)
        out["agentdojo"] = {"repo_path": ADJ, "git": git_head(ADJ),
                            "licence_file": grep_lines(lic, [r"^MIT License", r"^Copyright"]),
                            "pyproject_licence": grep_lines(os.path.join(ADJ, "pyproject.toml"), [r"License ::", r"^license"]),
                            "real_organisation_note_search": {"files_searched": len(md_files), "matches": note}}
    else:
        out["agentdojo"] = {"error": "AGENTDOJO not set or not a directory"}
    out["manuscript_statement"] = grep_lines(os.path.join(HERE, "MANUSCRIPT_DRAFT.md"), [r"AgentDrift \[5\] \(v2\.0"], maxlen=160)
    return out


# ------------------------------------------------------------------------------------------------ orchestration
SECTIONS = {"coverage": sec_coverage, "worlds": sec_worlds, "key_search": sec_key_search, "r1_overlap": sec_r1_overlap,
            "prevalence": sec_prevalence, "templates": sec_templates, "probe_v03": sec_probe_v03, "provenance": sec_provenance}
for _w, _units in PROBE_UNITS.items():
    for _u in _units:
        SECTIONS[f"probe_{_w}__{_u}"] = make_probe_section(_w, _u)


def part_path(s): return os.path.join(PARTS, f"{s}.json")


def part_ok(s):
    p = part_path(s)
    if not os.path.exists(p):
        return False
    try:
        return json.load(open(p, encoding="utf-8")).get("_script_sha256") == SCRIPT_SHA
    except Exception:  # noqa: BLE001
        return False


def run_section(s):
    t0 = time.time(); res = SECTIONS[s]()
    res = {"_script_sha256": SCRIPT_SHA, "_seconds": round(time.time() - t0, 1),
           "_written": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "result": res}
    os.makedirs(PARTS, exist_ok=True)
    tmp = part_path(s) + ".tmp"
    json.dump(res, open(tmp, "w", encoding="utf-8"), indent=1, default=float)
    os.replace(tmp, part_path(s))
    print(f"  section {s} done in {res['_seconds']} s", flush=True)


# ==== ASSEMBLY: code below this line only summarises saved sections; editing it does not invalidate them ====
def probe_summary(which):
    """Accuracy and balanced accuracy with trajectory-level bootstrap intervals (one shared set of resampled indices per
    probe, so differences between units are paired). Point = mean over the outer repetitions."""
    units = {u: json.load(open(part_path(f"probe_{which}__{u}"), encoding="utf-8"))["result"] for u in PROBE_UNITS[which]}
    recs, _, y = probe_inputs(which)
    cls, yi = np.unique(y, return_inverse=True); K = len(cls); n = len(y)
    corr = {u: np.array(units[u]["correct"], float) for u in units}          # (reps, n)
    rng = np.random.default_rng(SEED_BASE + SEEDS[f"probe_{which}"])
    idx = [rng.integers(0, n, n) for _ in range(B)]

    def bal(c, ib):
        tot = np.bincount(yi[ib], minlength=K).astype(float); hit = np.bincount(yi[ib], weights=c[ib], minlength=K)
        m = tot > 0
        return float(np.mean(hit[m] / tot[m]))

    out = {"n": n, "n_classes": int(K), "population": population_info(recs, which), "units": {}, "paired_vs_floors": {}}
    boots = {}
    for u, c in corr.items():
        cm = c.mean(0)
        acc_b = np.array([cm[ib].mean() for ib in idx]); boots[u] = acc_b
        bal_pt = float(np.mean([bal(c[r], np.arange(n)) for r in range(c.shape[0])]))
        bal_b = np.array([np.mean([bal(c[r], ib) for r in range(c.shape[0])]) for ib in idx[:500]])
        out["units"][u] = {"accuracy": trip(cm.mean(), acc_b), "accuracy_per_rep": units[u]["acc_per_rep"],
                           "balanced_accuracy": trip(bal_pt, bal_b), "balanced_accuracy_note": "interval from the first 500 replicates",
                           "C_chosen_per_outer_fold": units[u].get("C_chosen"),
                           "n_folds_C_at_grid_edge": units[u].get("n_folds_C_at_grid_edge"),
                           "n_convergence_warnings_refit": units[u].get("n_convergence_warnings_refit"),
                           "convergence_note": "counted for the refitted outer-fold models that make the predictions (inner grid fits run in worker processes)"}
    floors = ["majority"] + (["domain_only", "domain_x_label"] if which.startswith("world") else [])
    pairs = []
    for u in units:
        for fl in floors:
            if u == fl or u in floors and floors.index(u) < floors.index(fl):
                continue
            pairs.append((u, fl))
    pairs += [(a, b) for a, b in [("B", "B-minus"), ("A", "B"), ("C", "A")] if a in units and b in units]
    for u, fl in pairs:
        d = corr[u].mean(0) - corr[fl].mean(0)
        db = np.array([d[ib].mean() for ib in idx])
        out["paired_vs_floors"][f"{u} minus {fl}"] = trip(d.mean(), db) + [r4(np.mean(db <= 0))]
    out["paired_note"] = ("key 'X minus Y' = accuracy of X minus accuracy of Y: [point, 2.5%, 97.5%, share of replicates <= 0]; "
                          "paired trajectory-level bootstrap of the difference in out-of-fold correctness (same resampled rows)")
    return out


def assemble():
    P = {s: json.load(open(part_path(s), encoding="utf-8"))["result"] for s in SECTIONS if not s.startswith("probe_") or s == "probe_v03"}
    res = {"meta": {"script": os.path.abspath(__file__), "file_sha256": FILE_SHA, "section_code_sha256": SCRIPT_SHA,
                    "assembled": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "python": platform.python_version(),
                    "numpy": np.__version__, "sklearn": __import__("sklearn").__version__,
                    "AGENTDRIFT": MP, "AGENTDOJO": ADJ, "B": B, "seed_base": SEED_BASE, "seed_offsets": SEEDS,
                    "probe_design": {"model": "StandardScaler + LogisticRegression(penalty='l2', solver='lbfgs', multinomial, max_iter=5000)",
                                     "C_grid": C_GRID, "outer": f"StratifiedKFold({OUTER_K}, shuffle=True, random_state in {OUTER_REPS})",
                                     "inner": f"StratifiedKFold({INNER_K}, shuffle=True, random_state=100+10*rep+fold), GridSearchCV scoring='accuracy'",
                                     "A_vocabulary": "tools occurring in the outer training fold", "min_world_size": MIN_WORLD},
                    "inputs_sha256": {f: sha256_file(os.path.join(MP, *f.split("/"))) for f in
                                      ["data/train.jsonl", "data/val.jsonl", "data/test.jsonl", "data_taskdisjoint/train.jsonl", "data_taskdisjoint/test.jsonl"]},
                    "manuscript_values": MANUSCRIPT},
           "coverage": P["coverage"], "m5_worlds": P["worlds"], "m5_key_search": P["key_search"],
           "m5_world_reuse": P["r1_overlap"], "prevalence_by_domain": P["prevalence"],
           "m13_regularities_recomputed": P["templates"], "m13_provenance": P["provenance"],
           "m15_probes": {"v03_reconstruction": P["probe_v03"], "domain": probe_summary("domain"),
                          "domain_benign_only": probe_summary("domain_benign"), "world": probe_summary("world"),
                          "world_nonemail_supplementary": probe_summary("world_nonemail")}}
    if os.path.exists(REPRO):
        res["m14"] = json.load(open(REPRO, encoding="utf-8"))
    else:
        res["m14"] = {"note": "revision/repro_checks.json not found; run revision/repro_checks.py"}
    w = res["m5_worlds"]["pool_10656"]; ov = res["m5_world_reuse"]["R1"]["all_test"]
    pw = res["m15_probes"]["world"]["units"]; pdm = res["m15_probes"]["domain"]["units"]
    rows = [("n_worlds", MANUSCRIPT["n_worlds"], w["n_worlds"], "m5_worlds.pool_10656.n_worlds"),
            ("purity", MANUSCRIPT["purity"], w["purity_binary"][0], "m5_worlds.pool_10656.purity_binary"),
            ("majority_baseline", MANUSCRIPT["majority_baseline"], w["majority_baseline"], "m5_worlds.pool_10656.majority_baseline"),
            ("largest3", MANUSCRIPT["largest3"], w["largest_world_sizes"], "m5_worlds.pool_10656.largest_world_sizes"),
            ("n_worlds_ge30", MANUSCRIPT["n_worlds_ge30"], w["n_worlds_ge30"], "m5_worlds.pool_10656.n_worlds_ge30"),
            ("n_traj_ge30", MANUSCRIPT["n_traj_ge30"], w["n_traj_in_worlds_ge30"], "m5_worlds.pool_10656.n_traj_in_worlds_ge30"),
            ("r1_share_world_in_train", MANUSCRIPT["r1_share_world_in_train"], ov["share_world_in_train"][0], "m5_world_reuse.R1.all_test.share_world_in_train"),
            ("r1_world_rate_auroc", MANUSCRIPT["r1_world_rate_auroc"], ov["auroc_train_world_attack_rate"][0], "m5_world_reuse.R1.all_test.auroc_train_world_attack_rate"),
            ("r1_base_rate", MANUSCRIPT["r1_base_rate"], ov["base_rate_on_subset"], "m5_world_reuse.R1.all_test.base_rate_on_subset")]
    for u in ["A", "B", "B-minus", "majority"]:
        rows.append((f"domain_probe_{u}_nested", MANUSCRIPT["domain_probe"][u], pdm[u]["accuracy"][0], f"m15_probes.domain.units.{u}.accuracy"))
    for u, mk in [("A", "A"), ("B", "B"), ("B-minus", "B-minus"), ("majority", "majority"), ("domain_only", "domain_only")]:
        rows.append((f"world_probe_{u}_nested", MANUSCRIPT["world_probe"][mk], pw[u]["accuracy"][0], f"m15_probes.world.units.{u}.accuracy"))
    v3 = res["m15_probes"]["v03_reconstruction"]
    for which, tg in [("domain", "domain_probe"), ("world", "world_probe")]:
        for cfg in ["3fold_noshuffle", "3fold_shuffle_rs0"]:
            for u in v3[which][cfg]:
                rows.append((f"{which}_probe_{u}_v03_reconstruction_{cfg}", MANUSCRIPT[tg][u], v3[which][cfg][u],
                             f"m15_probes.v03_reconstruction.{which}.{cfg}.{u}"))
    cmp = []
    for q, mv, ov_, key in rows:
        same = (mv == ov_) if isinstance(mv, list) else (ov_ is not None and round(float(ov_), 3) == round(float(mv), 3))
        cmp.append({"quantity": q, "manuscript_or_v03": mv, "recomputed": ov_, "key": key, "equal_to_3dp": bool(same)})
    res["manuscript_comparison"] = cmp
    tmp = OUT + ".tmp"
    json.dump(res, open(tmp, "w", encoding="utf-8"), indent=1, default=float)
    os.replace(tmp, OUT)
    print(f"ASSEMBLED {OUT}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", action="append", default=[])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--budget", type=float, default=300.0)
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()
    todo = a.section or list(SECTIONS)
    if a.status:
        for s in SECTIONS:
            print(f"{s:32s} {'done' if part_ok(s) else ('stale' if os.path.exists(part_path(s)) else 'missing')}")
        return
    t0 = time.time()
    for s in todo:
        if s not in SECTIONS:
            raise SystemExit(f"unknown section {s}; choose from {list(SECTIONS)}")
        if part_ok(s) and not a.force:
            continue
        if time.time() - t0 > a.budget:
            print("budget used; call again", flush=True); return
        run_section(s)
    if all(part_ok(s) for s in SECTIONS):
        assemble()
    else:
        print("missing sections:", [s for s in SECTIONS if not part_ok(s)], flush=True)


if __name__ == "__main__":
    main()
