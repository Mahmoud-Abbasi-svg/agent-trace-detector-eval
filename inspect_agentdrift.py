"""Descriptive inspection of AgentDrift for the agent-trace IDS evaluation study.
Read-only. Computes tool vocabularies, train/task-disjoint overlap (covariate-drift risk),
trajectory-length and step-label distributions. Writes a summary to summary.json.
"""
import json, os, collections

ROOT = os.environ.get("AGENTDRIFT", "")
if not ROOT:
    # default clone location in scratchpad
    ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "AgentDrift")
def load(p): return [json.loads(l) for l in open(p, encoding="utf-8")]

strat_tr = load(os.path.join(ROOT, "data", "train.jsonl"))
strat_te = load(os.path.join(ROOT, "data", "test.jsonl"))
td_tr = load(os.path.join(ROOT, "data_taskdisjoint", "train.jsonl"))
td_te = load(os.path.join(ROOT, "data_taskdisjoint", "test.jsonl"))


def tools_of(rows):
    by_dom = collections.defaultdict(set); allt = set()
    for r in rows:
        for s in r["steps"]:
            t = s.get("tool")
            if t: by_dom[r["agent"]].add(t); allt.add(t)
    return by_dom, allt


def steplabels(rows):
    c = collections.Counter()
    for r in rows:
        for s in r["steps"]:
            c[s.get("label", "?")] += 1
    return c


def cat_counts(rows):
    return collections.Counter(r["category"] for r in rows)


by_dom, all_strat = tools_of(strat_tr)
_, all_td_te = tools_of(td_te)

print("=== tool vocabulary per domain (stratified train) ===")
for d, s in sorted(by_dom.items()):
    print(f"  {d:8} {len(s)} tools  e.g. {sorted(s)[:6]}")
print(f"  TOTAL unique tools (strat train): {len(all_strat)}")

print("\n=== covariate-drift check: tool overlap train vs task-disjoint test ===")
inter = all_strat & all_td_te
print(f"  strat-train tools={len(all_strat)}  td-test tools={len(all_td_te)}  "
      f"shared={len(inter)}  td-test-only(unseen)={len(all_td_te - all_strat)}")
print(f"  => {100*len(all_td_te & all_strat)/max(1,len(all_td_te)):.0f}% of task-disjoint-test tools were seen in training")

print("\n=== step-label distribution (stratified train) ===")
for k, v in steplabels(strat_tr).most_common():
    print(f"  {k:16} {v}")

print("\n=== trajectory category counts ===")
for name, rows in [("strat_train", strat_tr), ("strat_test", strat_te),
                   ("td_train", td_tr), ("td_test", td_te)]:
    print(f"  {name:12} {dict(cat_counts(rows))}")

# realistic-prevalence note: attacked fraction in test as-is
att = sum(cat_counts(strat_te)[c] for c in ("attacked",))
print(f"\n  stratified-test attacked fraction (as shipped) = {att}/{len(strat_te)} "
      f"= {100*att/len(strat_te):.0f}%  (IDS study will SUBSAMPLE attacks to 1-5% to mimic real prevalence)")

os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
json.dump({
    "n_domains": len(by_dom),
    "tools_per_domain": {d: len(s) for d, s in by_dom.items()},
    "total_tools_strat_train": len(all_strat),
    "td_test_tools": len(all_td_te),
    "td_test_tools_unseen_in_train": len(all_td_te - all_strat),
    "step_labels_strat_train": dict(steplabels(strat_tr)),
    "category_counts": {n: dict(cat_counts(r)) for n, r in
                        [("strat_train", strat_tr), ("strat_test", strat_te), ("td_test", td_te)]},
}, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "summary.json"), "w"), indent=2)
print("\nwrote summary.json")
