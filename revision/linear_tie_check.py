"""Diagnostic for the linear cache README (not part of the cache): how much do exact ties matter?

The cache makes mathematically tied scores bitwise tied (build_linear_cache.by_unique_rows, MARKOV_DECIMALS). The v0.3
code paths did not, so float noise broke such ties. This script recomputes, for every single-fit setting, the v0.3-style
raw scores of Markov (rerun_r1r3.py recipe, no rounding) and LR-A (ablation.fit / ablation.sco) and reports their AUROC
on pool E next to the AUROC of the cached (tie-exact) scores. Output: cache/linear/tie_check.json

Run: AGENTDRIFT=<clone> C:/Python310/python.exe revision/linear_tie_check.py
"""
import os, sys, json, pickle
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
from featurise import tool_seq                         # noqa: E402
from baseline_drift import Markov                      # noqa: E402
import evalkit as ek                                   # noqa: E402
import ablation                                        # noqa: E402
import build_linear_cache as blc                       # noqa: E402  (split definitions only; nothing is written)

OUT = os.path.join(HERE, "cache", "linear", "tie_check.json")
SINGLE = [s for s in blc.SETTINGS if s not in blc.CV_SETTINGS]


def main():
    res = {"note": "AUROC on pool E. raw = v0.3 code path (float noise can break mathematically tied scores); "
                   "cache = tie-exact scores stored in cache/linear/<setting>.pkl.", "settings": {}}
    for s in SINGLE:
        tr, te, _ = blc.single_split(s)
        # cache pickle is self-produced by build_linear_cache.py (trusted)
        with open(blc.out_path(s), "rb") as fh:
            o = pickle.load(fh)
        assert [m["id"] for m in o["test_meta"]] == [r["id"] for r in te]
        y = o["y"]; E = o["pool_E"]
        mk = Markov().fit([tool_seq(r) for r in tr if r["category"] == "benign"])
        raw_mk = np.array([mk.score(tool_seq(r)) for r in te])
        raw_a = ablation.sco(ablation.fit(tr, "A"), te, "A")["score"]
        row = {}
        for det, raw in [("Markov", raw_mk), ("LR-A", raw_a)]:
            row[det] = {"AUROC_raw_v03_path": round(ek.auroc(y[E], raw[E]), 4),
                        "AUROC_cache": round(ek.auroc(y[E], o["scores"][det][E]), 4),
                        "n_distinct_raw": int(len(np.unique(raw[E]))),
                        "n_distinct_cache": int(len(np.unique(o["scores"][det][E])))}
            row[det]["diff"] = round(row[det]["AUROC_raw_v03_path"] - row[det]["AUROC_cache"], 4)
        res["settings"][s] = row
        print(s, row, flush=True)
    json.dump(res, open(OUT, "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
