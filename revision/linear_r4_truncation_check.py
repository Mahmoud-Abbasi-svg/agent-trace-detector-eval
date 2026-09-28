"""Diagnostic for the linear cache README (not part of the cache): why do the corrected R4 values differ from the
review's corrected estimates (transfer LR-C 0.476, LR-B 0.483; within-UT LR-C 0.846, LR-B 0.773)?

The corrected extraction (agentdojo_traces.jsonl) changes two things relative to v0.3: positional pairing of tool
messages AND no truncation of observations (v0.3 cut them at 2,000 characters). The review's numbers were computed
with positional pairing. This script re-applies the v0.3 truncation (str(obs)[:2000]) to the corrected P0 records and
re-runs the unmodified v0.3 code paths (r4_control.fit_lr / score / within_oof, 5 fold seeds), so that the two effects
can be told apart. AUROC point estimates as in r4_control.py (mean over the 5 CV repetitions); 95% intervals from
evalkit.bootstrap (stratified, B = 2000, seed 12345). Output: cache/linear/r4_truncation_check.json

Run: AGENTDRIFT=<clone> C:/Python310/python.exe revision/linear_r4_truncation_check.py
"""
import os, sys, json, copy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from featurise import load                      # noqa: E402
import evalkit as ek                            # noqa: E402
import r4_control                               # noqa: E402

MP = os.environ["AGENTDRIFT"]
OUT = os.path.join(HERE, "cache", "linear", "r4_truncation_check.json")
TRUNC = 2000


def truncated(recs):
    out = copy.deepcopy(recs)
    for r in out:
        for s in r["steps"]:
            s["obs"] = str(s.get("obs", ""))[:TRUNC]
    return out


def main():
    drift = load(os.path.join(MP, "data", "train.jsonl")) + load(os.path.join(MP, "data", "test.jsonl"))
    p0 = [json.loads(l) for l in open(os.path.join(ROOT, "agentdojo_traces.jsonl"), encoding="utf-8")]
    variants = {"untruncated_corrected": p0, "truncated_2000": truncated(p0)}
    n_changed = sum(any(len(str(s["obs"])) > TRUNC for s in r["steps"]) for r in p0)
    y = np.array([int(r["category"] == "attacked") for r in p0])
    models = {agn: r4_control.fit_lr(drift, agn) for agn in (False, True)}
    res = {"meta": {"n_P0": len(p0), "n_pos": int(y.sum()), "n_neg": int((y == 0).sum()),
                    "n_P0_traces_with_an_observation_over_2000_chars": int(n_changed),
                    "B": 2000, "seed": 12345, "cv_seeds": r4_control.SEEDS,
                    "note": "LR-C = r4_control 'full' (agnostic=False); LR-B = 'agn' (agnostic=True). Transfer models "
                            "trained on all 10,656 pooled AgentDrift trajectories (unchanged across variants)."},
           "review_corrected_estimates": {"transfer_LR-C": 0.476, "transfer_LR-B": 0.483,
                                          "withinUT_LR-C": 0.846, "withinUT_LR-B": 0.773}}
    for vname, recs in variants.items():
        methods = {}
        for agn, lab in [(False, "LR-C"), (True, "LR-B")]:
            s, pr = r4_control.score(models[agn], recs, agn)
            methods[f"transfer_{lab}"] = [{"score": s, "pred": pr}]
            methods[f"withinUT_{lab}"] = [r4_control.within_oof(recs, "UT", sd, agn)[0] for sd in r4_control.SEEDS]
        bt = ek.bootstrap(y, methods, B=2000, seed=12345)
        res[vname] = {}
        for m in methods:
            pt = ek.point(y, methods[m])
            res[vname][m] = {"AUROC": [round(pt["AUROC"], 4)] + [round(x, 4) for x in ek.ci(bt[m]["AUROC"])[1:]],
                             "AUPRC_native": [round(pt["AUPRC"], 4)] + [round(x, 4) for x in ek.ci(bt[m]["AUPRC"])[1:]],
                             "FPR@t_r80": [round(pt["FPR@t_r80"], 4)] + [round(x, 4) for x in ek.ci(bt[m]["FPR@t_r80"])[1:]]}
        print(vname, {m: res[vname][m]["AUROC"] for m in methods}, flush=True)
    json.dump(res, open(OUT, "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
