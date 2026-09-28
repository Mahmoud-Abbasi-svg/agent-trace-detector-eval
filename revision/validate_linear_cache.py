"""Validate the LINEAR prediction cache and write its summary.

Reads revision/cache/linear/<setting>.pkl (built by build_linear_cache.py) and writes
  cache/linear/_validation/<setting>.json   per-setting metrics (resumable: skipped when newer than the pickle)
  cache/linear/validation.json               format checks, sizes, metrics, reproduction anchors, R4 values
  cache/linear/CACHE_README.md               human-readable summary; every number in it is read from validation.json
Nothing here fits a model; everything is computed from the cached test scores.

Metrics (population = pool E: AgentDrift benign + hard_negative + attacked, AgentDojo benign + attacked):
  AUROC, AUPRC (native prevalence), recall / FPR at t_r80 and t05. Point estimate = evalkit.point (mean over CV
  repetitions); 95% interval = evalkit.bootstrap (stratified, trajectory level; a random repetition per replicate),
  B = 1000 for AgentDrift settings (as v0.3 R1-R3) and B = 2000 for AgentDojo settings (as v0.3 R4), seed
  20260926 + setting index. Failed-attack flag rates (rows outside pool E) use a row bootstrap with the same B.
  These are the v0.3-style trajectory-level intervals used for the anchors; the plan's cluster bootstrap and
  prevalence-weighted AUPRC belong to the downstream analyses.
Run: C:/Python310/python.exe revision/validate_linear_cache.py [--budget 500]
"""
import os, sys, json, time, pickle, argparse, math
import numpy as np
from scipy.special import expit

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import evalkit as ek  # noqa: E402

CACHE = os.path.join(HERE, "cache", "linear"); GRU = os.path.join(HERE, "cache", "gru")
VDIR = os.path.join(CACHE, "_validation"); ARCH = os.path.join(ROOT, "archive_v0.3")
DOMAINS = ["medical", "banking", "web", "email", "coding"]; SUITES = ["banking", "slack", "travel", "workspace"]
SETTINGS = (["R1"] + [f"R1wd_s{i}" for i in range(5)] + ["R2"] + [f"R3_{d}" for d in DOMAINS]
            + ["R4", "WUT", "WUT_F", "WIT"] + [f"LOSO_{s}" for s in SUITES])
DOJO = {"R4", "WUT", "WUT_F", "WIT"} | {f"LOSO_{s}" for s in SUITES}
MKEYS = ["AUROC", "AUPRC", "recall@t_r80", "FPR@t_r80", "recall@t05", "FPR@t05"]
META_KEYS = {"id", "benchmark", "category", "y", "in_pool_E", "domain", "pattern", "world", "task", "model", "user_task",
             "injection_task", "attack_type", "is_dos", "n_steps", "cov"}


def r4(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), 4)


# The shared cache format (pickle per setting) is fixed by the revision specification; the pickles read here are
# produced on this machine by build_linear_cache.py / build_gru_cache.py (trusted, self-produced files).
def load(setting):
    with open(os.path.join(CACHE, f"{setting}.pkl"), "rb") as fh:
        return pickle.load(fh)


# ----------------------------------------------------------------------------- format checks
def format_checks(o):
    c = {}
    n = len(o["test_meta"]); y = np.asarray(o["y"]); E = np.asarray(o["pool_E"])
    cv = o["info"].get("scheme") == "cv"
    c["meta_keys_complete"] = all(META_KEYS <= set(m) for m in o["test_meta"])
    c["y_matches_meta"] = bool(np.array_equal(y, [m["y"] for m in o["test_meta"]]))
    c["poolE_matches_meta"] = bool(np.array_equal(E, [m["in_pool_E"] for m in o["test_meta"]]))
    c["failed_rows_not_in_poolE"] = bool(all((m["category"] in ("failed_attack", "attack_failed")) == (not m["in_pool_E"])
                                             for m in o["test_meta"]))
    c["y1_only_attacked"] = bool(all((m["y"] == 1) == (m["category"] == "attacked") for m in o["test_meta"]))
    c["ids_unique"] = len({m["id"] for m in o["test_meta"]}) == n
    shape_ok, finite_ok, pred_ok, t05_ok = True, True, True, True
    for d, s in o["scores"].items():
        s = np.asarray(s)
        shape_ok &= s.shape == ((5, n) if cv else (n,))
        finite_ok &= bool(np.isfinite(s).all())
        S = np.atleast_2d(s)
        for k in ("t_r80", "t05"):
            p = np.atleast_2d(np.asarray(o["pred"][d][k]))
            shape_ok &= p.shape == S.shape and p.dtype == bool
        t05_ok &= o["thr"][d]["t05"] == 0.5
        P05 = np.atleast_2d(o["pred"][d]["t05"]); pred_ok &= bool(np.array_equal(P05, S >= 0.5))
        P80 = np.atleast_2d(o["pred"][d]["t_r80"])
        if cv:
            thr = np.asarray(o["thr"][d]["t_r80"]); folds = np.asarray(o["folds"])
            T = np.take_along_axis(thr, folds, axis=1)
        else:
            T = np.full(S.shape, float(o["thr"][d]["t_r80"]))
        pred_ok &= bool(np.array_equal(P80, S >= T))
    c["score_and_pred_shapes"] = bool(shape_ok); c["scores_finite"] = bool(finite_ok)
    c["pred_equals_score_vs_training_threshold"] = bool(pred_ok); c["t05_is_0.5"] = bool(t05_ok)
    if not cv:
        ok = True
        for d in ("LR-C", "LR-C-drop"):
            lp = o["lrc_parts"][d]; z = lp["tool"] + lp["beh"] + lp["intercept"]
            ok &= bool(np.allclose(expit(z), o["scores"][d], atol=1e-9))
            ok &= len(lp["train_tool_part"]) == o["info"]["n_train"] == len(lp["train_y"])
        c["lrc_parts_reproduce_scores"] = ok
        c["coefs_present_LR-B_LR-C"] = all(k in o["coefs"] and len(o["coefs"][k]["names"]) == len(o["coefs"][k]["coef"])
                                          for k in ("LR-B", "LR-C"))
    gp = os.path.join(GRU, f"{o['setting']}.pkl")
    if os.path.exists(gp):
        with open(gp, "rb") as fh:
            g = pickle.load(fh)
        c["row_order_equals_gru_cache"] = [m["id"] for m in g["test_meta"]] == [m["id"] for m in o["test_meta"]]
    else:
        c["row_order_equals_gru_cache"] = "no GRU pickle for this setting"
    return c


# ----------------------------------------------------------------------------- metrics
def setting_metrics(setting, idx):
    o = load(setting)
    y = np.asarray(o["y"]); E = np.asarray(o["pool_E"]); F = np.where(~E)[0]; Ei = np.where(E)[0]
    B = 2000 if setting in DOJO else 1000; seed = 20260926 + idx
    methods = {}
    for d, s in o["scores"].items():
        S = np.atleast_2d(s)
        P = {k: np.atleast_2d(o["pred"][d][k]) for k in ("t_r80", "t05")}
        methods[d] = [{"score": S[i][Ei], "pred": {k: P[k][i][Ei] for k in P}} for i in range(S.shape[0])]
    yE = y[Ei]
    bt = ek.bootstrap(yE, methods, B=B, seed=seed)
    out = {"setting": setting, "B": B, "seed": seed, "n_pool_E": int(len(Ei)), "n_pos": int(yE.sum()),
           "n_neg": int((yE == 0).sum()), "n_failed": int(len(F)), "metrics": {}}
    rng = np.random.default_rng(seed + 1)
    for d in methods:
        pt = ek.point(yE, methods[d])
        row = {k: [r4(pt[k])] + [r4(v) for v in ek.ci(bt[d][k])[1:]] for k in MKEYS}
        if len(F):
            for k in ("t_r80", "t05"):
                P = np.atleast_2d(o["pred"][d][k])[:, F].astype(float)
                point = float(P.mean(1).mean())
                reps = rng.integers(P.shape[0], size=B); rates = np.empty(B)
                for b in range(B):
                    rates[b] = P[reps[b], rng.integers(0, len(F), len(F))].mean()
                row[f"failed_flag_rate@{k}"] = [r4(point), r4(np.percentile(rates, 2.5)), r4(np.percentile(rates, 97.5))]
        # AUROC over ALL test rows (failed attacks counted as negatives), as world_disjoint_r1.py computes it
        row["AUROC_all_rows"] = r4(np.mean([ek.auroc(y, r) for r in np.atleast_2d(o["scores"][d])]))
        out["metrics"][d] = row
    info = o["info"]
    out["sizes"] = {"n_train": info["n_train"] if not isinstance(info["n_train"], list) else info["n_train_mean"],
                    "n_train_pos": info["n_train_pos"] if not isinstance(info["n_train_pos"], list) else info["n_train_pos_mean"],
                    "n_test": info["n_test"], "n_test_by_category": info["n_test_by_category"],
                    "vocab_size": info["vocab_size"] if not isinstance(info["vocab_size"], list)
                    else [min(min(v) for v in info["vocab_size"]), max(max(v) for v in info["vocab_size"])],
                    "kappa": r4(info["kappa"]), "kappa_all_rows": r4(info["kappa_all_rows"]),
                    "scheme": info.get("cv_scheme", "single"), "detectors": list(o["scores"])}
    out["format_checks"] = format_checks(o)
    chs = info.get("checks", {}); chs = chs if isinstance(chs, list) else [chs]
    out["reproduction_checks"] = {"n": sum(len(c) for c in chs), "n_pass": sum(bool(v.get("pass")) for c in chs for v in c.values()),
                                  "max_abs_score_diff": max([v.get("max_abs_diff", 0.0) for c in chs for v in c.values()] or [0.0]),
                                  "auroc_differs_from_v03_code_path": [
                                      {"rep": i, "check": k, "auroc_v03_code": r4(v["auroc_ref"]), "auroc_cache": r4(v["auroc_cache"])}
                                      for i, c in enumerate(chs) for k, v in c.items() if "auroc_equal" in v and not v["auroc_equal"]]}
    for k in ("worldmask_replacements_per_traj", "worldmask_frac_traj_with_residual_name_token", "fold_assignment",
              "failed_as_train_negatives", "lr_converged", "lr_converged_all", "tfidf_n_features"):
        if k in info:
            v = info[k]
            if k == "fold_assignment":
                v = [{"n_groups_P0": f["n_groups_P0"], "n_extra_groups": len(f["extra_groups"]),
                      "n_failed_rows_in_extra_groups": f["n_failed_rows_in_extra_groups"]} for f in v]
            out[k] = v
    out["provenance"] = o.get("provenance", {})
    return out


# ----------------------------------------------------------------------------- anchors
def arch(name):
    return json.load(open(os.path.join(ARCH, name)))


def anchors(M):
    ab = arch("ablation_results.json"); rr = arch("r1r3_corrected.json"); wd = arch("world_disjoint_r1_results.json")
    A = []

    def add(setting, det, exp3, v03_4dp, source, metric="AUROC", required=True):
        got = M[setting]["metrics"][det]
        val = got["AUROC_all_rows"] if metric == "AUROC_all_rows" else got["AUROC"][0]
        ci = None if metric == "AUROC_all_rows" else got["AUROC"][1:]
        A.append({"setting": setting, "detector": det, "metric": metric, "required": required,
                  "expected_3dp": exp3, "v03_archived_4dp": v03_4dp, "source": source,
                  "reproduced": val, "reproduced_ci95": ci,
                  "reproduced_3dp": round(val, 3), "match_3dp": round(val, 3) == exp3 if exp3 is not None else None,
                  "abs_diff_vs_archived": r4(abs(val - v03_4dp)) if v03_4dp is not None else None})
    R1 = ab["R1"]["by_prev"]["native"]["metrics"]
    for det, rep, e in [("LR-A", "A", 0.777), ("LR-B", "B", 0.830), ("LR-C", "C", 0.879)]:
        add("R1", det, e, R1[rep]["AUROC"][0], "archive_v0.3/ablation_results.json R1.by_prev.native")
    r1s = rr["R1_same"]["by_prev"]["native"]["metrics"]
    add("R1", "Markov", 0.724, r1s["MARKOV"]["AUROC"][0], "archive_v0.3/r1r3_corrected.json R1_same")
    add("R1", "IForest", 0.588, r1s["IFOREST"]["AUROC"][0], "archive_v0.3/r1r3_corrected.json R1_same")
    exp = {"LR-C": [0.745, 0.828, 0.862, 0.739, 0.814], "LR-B": [0.808, 0.854, 0.896, 0.792, 0.852]}
    for det, rep in [("LR-C", "C"), ("LR-B", "B")]:
        for i, d in enumerate(DOMAINS):
            add(f"R3_{d}", det, exp[det][i], ab["R3"][d]["by_prev"]["native"]["metrics"][rep]["AUROC"][0],
                f"archive_v0.3/ablation_results.json R3.{d}")
    add("R2", "LR-C", 0.902, rr["R2_newtask"]["by_prev"]["native"]["metrics"]["LOGREG"]["AUROC"][0],
        "archive_v0.3/r1r3_corrected.json R2_newtask LOGREG")
    # additional (not requested) anchors
    for det, key in [("Markov", "MARKOV"), ("IForest", "IFOREST")]:
        add("R2", det, None, rr["R2_newtask"]["by_prev"]["native"]["metrics"][key]["AUROC"][0],
            "archive_v0.3/r1r3_corrected.json R2_newtask", required=False)
        for d in DOMAINS:
            add(f"R3_{d}", det, None, rr[f"R3_LODO_{d}"]["by_prev"]["native"]["metrics"][key]["AUROC"][0],
                f"archive_v0.3/r1r3_corrected.json R3_LODO_{d}", required=False)
    for d in DOMAINS:
        add(f"R3_{d}", "LR-A", None, ab["R3"][d]["by_prev"]["native"]["metrics"]["A"]["AUROC"][0],
            f"archive_v0.3/ablation_results.json R3.{d}", required=False)
    for sp in wd["splits"]:
        for det, rep in [("LR-A", "A"), ("LR-B", "B"), ("LR-C", "C")]:
            add(f"R1wd_s{sp['seed']}", det, None, sp["AUROC"][rep],
                "archive_v0.3/world_disjoint_r1_results.json splits (AUROC over all test rows incl. failed attacks)",
                metric="AUROC_all_rows", required=False)
    # LR-A in R4 uses only tool counts, which the extraction fix does not change -> must equal v0.3
    P0v = ab["R4"]["P0"]["by_prev"]["native"]["metrics"]
    add("R4", "LR-A", None, P0v["transfer_A"]["AUROC"][0], "archive_v0.3/ablation_results.json R4.P0 transfer_A "
        "(tool counts unchanged by the extraction fix)", required=False)
    add("WUT", "LR-A", None, P0v["within_A"]["AUROC"][0], "archive_v0.3/ablation_results.json R4.P0 within_A "
        "(tool counts unchanged by the extraction fix)", required=False)
    return A


def r4_values(M):
    cur_ab = json.load(open(os.path.join(ROOT, "ablation_results.json")))
    cur_rc = json.load(open(os.path.join(ROOT, "r4_control_results.json")))
    log_ok = "gru_ladder exit 0" in open(os.path.join(HERE, "rerun_v03_scripts.log"), encoding="utf-8").read()
    P0 = cur_ab["R4"]["P0"]["by_prev"]["native"]["metrics"]; rc = cur_rc["P0"]["native"]["metrics"]
    tr = os.path.join(CACHE, "r4_truncation_check.json")
    trunc = json.load(open(tr)) if os.path.exists(tr) else None
    out = {"population": "AgentDojo P0 (392 benign, 2,294 compromised); unsuccessful attacks excluded from AUROC",
           "transfer_R4": {d: M["R4"]["metrics"][d] for d in M["R4"]["metrics"]},
           "withinUT_WUT": {d: M["WUT"]["metrics"][d] for d in M["WUT"]["metrics"]},
           "withinUT_failed_as_negatives_WUT_F": {d: M["WUT_F"]["metrics"][d] for d in M["WUT_F"]["metrics"]},
           "withinIT_WIT": {d: M["WIT"]["metrics"][d] for d in M["WIT"]["metrics"]},
           "rerun_v03_scripts_final": log_ok,
           "v03_scripts_rerun_on_corrected_data": {
               "source": "ablation_results.json / r4_control_results.json (re-run on corrected data; log has 'gru_ladder exit 0')",
               "transfer_LR-A": P0["transfer_A"]["AUROC"], "transfer_LR-B": P0["transfer_B"]["AUROC"],
               "transfer_LR-C": P0["transfer_C"]["AUROC"], "withinUT_LR-A": P0["within_A"]["AUROC"],
               "withinUT_LR-B": P0["within_B"]["AUROC"], "withinUT_LR-C": P0["within_C"]["AUROC"],
               "withinIT_LR-C": rc["within_IT_full"]["AUROC"], "withinIT_LR-B": rc["within_IT_agn"]["AUROC"]},
           "review_corrected_estimates": {"transfer_LR-C": 0.476, "transfer_LR-B": 0.483, "withinUT_LR-C": 0.846,
                                          "withinUT_LR-B": 0.773},
           "truncation_check": trunc}
    v = out["v03_scripts_rerun_on_corrected_data"]
    pairs = [("transfer_LR-A", "R4", "LR-A"), ("transfer_LR-B", "R4", "LR-B"), ("transfer_LR-C", "R4", "LR-C"),
             ("withinUT_LR-A", "WUT", "LR-A"), ("withinUT_LR-B", "WUT", "LR-B"), ("withinUT_LR-C", "WUT", "LR-C"),
             ("withinIT_LR-B", "WIT", "LR-B"), ("withinIT_LR-C", "WIT", "LR-C")]
    out["cache_equals_rerun_v03_point_4dp"] = {k: M[s]["metrics"][d]["AUROC"][0] == v[k][0] for k, s, d in pairs}
    return out


# ----------------------------------------------------------------------------- README
def fmt(t):
    return "n/a" if t is None or t[0] is None else f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"


def readme(V):
    L = []; S = V["settings"]
    L += ["# Linear prediction cache (revision)", "",
          f"Built by `revision/build_linear_cache.py` (script SHA-256 `{V['build_script_sha256']}`), validated by "
          f"`revision/validate_linear_cache.py` on {V['validated']}. Every number below is read from "
          "`cache/linear/validation.json` (key given in brackets where useful). Scores are for the 20 settings of "
          "REVISION_ANALYSIS_PLAN.md Section 2; nothing was tuned on test data (all thresholds come from the training "
          "folds).", "",
          f"Overall: format checks {'all pass' if V['all_format_checks_pass'] else 'SOME FAIL (see validation.json)'}; "
          f"in-process reproduction checks against the v0.3 code paths: {V['reproduction_checks_total']['n_pass']} of "
          f"{V['reproduction_checks_total']['n']} pass (largest score difference "
          f"{V['reproduction_checks_total']['max_abs_score_diff']:.1e}); requested anchors reproduced: "
          f"{V['anchor_summary']['required_match']} of {V['anchor_summary']['required_n']}.", "",
          "## Files", "",
          "- `<setting>.pkl`: one dict per setting in the shared cache format (keys `setting`, `family`, `test_meta`, `y`, "
          "`pool_E`, `scores`, `pred`, `thr`, `info`, plus `lrc_parts` and `coefs` for single-fit settings and "
          "`folds`, `cov_reps` for cross-validated ones, `detectors`, `provenance`).",
          "- `_parts/`: one file per CV repetition of WUT, WUT_F, WIT (assembled into the pickles).",
          "- `_validation/<setting>.json`, `validation.json`: the numbers in this file.",
          "- `r4_truncation_check.json`: diagnostic for the R4 values (from `revision/linear_r4_truncation_check.py`).",
          "- `build_log.txt`: build log.", "",
          "Row order within each setting is identical to the GRU cache (`cache/gru/<setting>.pkl`) wherever that pickle "
          "exists (checked: `settings.<s>.format_checks.row_order_equals_gru_cache`).", "",
          "## Detectors", "",
          "| Detector | Definition |", "|---|---|",
          "| LR-A, LR-B, LR-C | as ablation.py: StandardScaler on training rows, LogisticRegression(max_iter=1000, class_weight='balanced'); A = bag of tools (vocabulary from benign training trajectories), B = 10 behavioural features, C = B + A |",
          "| LR-B-noobs, LR-C-noobs | B or C without behavioural features 8, 9 (max and mean observation length) |",
          "| LR-B-minus | B without behavioural features 4, 5 (keyword state-changing count and fraction) |",
          "| LR-C-drop | LR-C fitted on [X; X with the tool block zeroed], labels duplicated, scaler on the stacked matrix; t_r80 from the original training positives |",
          "| Markov | baseline_drift.Markov fitted on benign training tool sequences, as rerun_r1r3.py |",
          "| IForest | IsolationForest(200 trees, random_state=0) on standardized C features of benign training rows, as rerun_r1r3.py |",
          "| TFIDF | TfidfVectorizer(lowercase, (1,2)-grams, min_df=2, max_features=50000, sublinear_tf) on training texts + LogisticRegression(max_iter=2000, class_weight='balanced', C=1); text = ' \\|\\| '.join(f'{tool} {str(args)} {obs}') |",
          "| TFIDF-nodigit | TFIDF with all digits removed from the text |",
          "| TFIDF-worldmask (R1 only) | TFIDF with every string value of the trajectory's world dict replaced by WORLDTOKEN |", "",
          "Thresholds: `t_r80` = evalkit.thr_at_recall(in-sample training-positive scores, 0.80) (for CV settings one "
          "value per repetition and fold, `thr[d]['t_r80']` has shape (5, 5) and `folds` gives each row's fold); "
          "`t05` = 0.5 (a probability cut for LR-*/TFIDF*, sklearn's default outlier cut for IForest, and not a "
          "calibrated cut for Markov, whose score is a mean negative log-probability).", "",
          "## Settings and sizes", "",
          "| Setting | Scheme | n_train (pos) | Test pool E pos / neg | Failed attacks scored | Vocab | kappa |",
          "|---|---|---|---|---|---|---|"]
    for s in SETTINGS:
        z = S[s]["sizes"]; m = S[s]
        ntr = f"{z['n_train']:.0f} ({z['n_train_pos']:.0f})" if not isinstance(z["n_train"], float) or z["n_train"].is_integer() \
            else f"{z['n_train']:.1f} ({z['n_train_pos']:.1f}) mean"
        voc = z["vocab_size"] if not isinstance(z["vocab_size"], list) else f"{z['vocab_size'][0]}-{z['vocab_size'][1]}"
        L.append(f"| {s} | {z['scheme']} | {ntr} | {m['n_pos']} / {m['n_neg']} | {m['n_failed']} | {voc} | {z['kappa']:.3f} |")
    L += ["", "kappa = tool-occurrence coverage of the pool-E test rows by the benign training vocabulary (CV settings: "
          "mean over repetitions and folds). For CV settings n_train is the mean over the 25 fits.", "",
          "## Reproduction anchors (v0.3)", "",
          "AUROC on pool E, point estimate with 95% stratified bootstrap interval [`anchors`].", "",
          "| Setting | Detector | v0.3 | Reproduced | Match (3 dp) |", "|---|---|---|---|---|"]
    for a in V["anchors"]:
        if a["required"]:
            L.append(f"| {a['setting']} | {a['detector']} | {a['expected_3dp']:.3f} | "
                     f"{a['reproduced']:.4f} [{a['reproduced_ci95'][0]:.3f}, {a['reproduced_ci95'][1]:.3f}] | "
                     f"{'yes' if a['match_3dp'] else 'NO'} |")
    extra = [a for a in V["anchors"] if not a["required"]]
    nm = sum(1 for a in extra if a["abs_diff_vs_archived"] == 0)
    L += ["", f"Additional anchors (not requested): {nm} of {len(extra)} equal the archived v0.3 value to 4 decimals "
          "(R2 and R3 Markov/IForest, R3 LR-A, R1-wd LR-A/B/C for 5 split seeds, R4 LR-A transfer and within-UT). "
          "The exceptions are listed here:", ""]
    for a in extra:
        if a["abs_diff_vs_archived"] != 0:
            L.append(f"- {a['setting']} {a['detector']} ({a['metric']}): v0.3 {a['v03_archived_4dp']:.4f}, cache "
                     f"{a['reproduced']:.4f}.")
    L += ["", "These exceptions are tied scores: when every test trajectory has an unseen tool vocabulary, LR-A (and "
          "Markov) give all rows the same score mathematically, but the v0.3 code broke the tie with float noise "
          "(values such as 0.4989). The cache makes such ties bitwise exact, so the AUROC is exactly 0.5. The noisy "
          "value is not even stable across runs of the same v0.3 code: re-running it today gives the values in the "
          "next table, which differ from the archived ones.", ""]
    if V.get("tie_check"):
        L += ["The same float noise matters much more for Markov at low coverage, because its mean of identical "
              "log-probabilities differs in the last bits with trajectory length, so the ties are broken in a systematic, not "
              "random, order (in LOSO_slack every row is tied mathematically). "
              "AUROC on pool E, v0.3 code path versus tie-exact cache [`tie_check`] (settings with a difference of "
              "at least 0.0005):", "", "| Setting | Detector | v0.3 code path (raw) | Cache (tie-exact) |", "|---|---|---|---|"]
        for st, row in V["tie_check"]["settings"].items():
            for d, v in row.items():
                if abs(v["diff"]) >= 0.0005:
                    L.append(f"| {st} | {d} | {v['AUROC_raw_v03_path']:.4f} | {v['AUROC_cache']:.4f} |")
        L += [""]
    L += [
          "## R4 with the corrected AgentDojo extraction", "",
          "These values differ from v0.3 because the extraction was corrected. Population: P0 (unsuccessful attacks "
          "are scored but excluded from AUROC) [`r4.transfer_R4`, `r4.withinUT_WUT`, `r4.withinIT_WIT`, "
          "`r4.withinUT_failed_as_negatives_WUT_F`].", "",
          "| Detector | Transfer (R4) | Within-UT (WUT) | Within-UT, failed as negatives (WUT_F) | Within-IT (WIT) |",
          "|---|---|---|---|---|"]
    R = V["r4"]
    for d in R["transfer_R4"]:
        L.append(f"| {d} | {fmt(R['transfer_R4'][d]['AUROC'])} | {fmt(R['withinUT_WUT'].get(d, {}).get('AUROC'))} | "
                 f"{fmt(R['withinUT_failed_as_negatives_WUT_F'].get(d, {}).get('AUROC'))} | {fmt(R['withinIT_WIT'].get(d, {}).get('AUROC'))} |")
    v = R["v03_scripts_rerun_on_corrected_data"]; eq = R["cache_equals_rerun_v03_point_4dp"]
    L += ["", f"The LR-A/B/C point values equal the re-run v0.3 scripts (ablation.py, r4_control.py on the corrected "
          f"data) to 4 decimals: {sum(eq.values())} of {len(eq)} [`r4.cache_equals_rerun_v03_point_4dp`]. "
          "The within-UT intervals here come from this validator's bootstrap, so they differ slightly from "
          "ablation_results.json.", ""]
    if R["truncation_check"]:
        T = R["truncation_check"]; u = T["untruncated_corrected"]; t = T["truncated_2000"]; rv = R["review_corrected_estimates"]
        L += ["The review's corrected estimates used positional pairing but still truncated observations at 2,000 "
              "characters. The corrected file also removes that truncation (plan item 1.2). Re-applying the "
              "truncation to the corrected records and re-running the unmodified r4_control code paths "
              f"(`r4_truncation_check.json`; {T['meta']['n_P0_traces_with_an_observation_over_2000_chars']} P0 "
              "traces have an observation longer than 2,000 characters):", "",
              "| Quantity | Review | Corrected, truncated at 2,000 | Corrected, untruncated (this cache) |", "|---|---|---|---|"]
        for k in ["transfer_LR-C", "transfer_LR-B", "withinUT_LR-C", "withinUT_LR-B"]:
            L.append(f"| {k} | {rv[k]:.3f} | {fmt(t[k]['AUROC'])} | {fmt(u[k]['AUROC'])} |")
        L += ["", "So the review's numbers are reproduced once truncation is re-applied. Removing the truncation leaves "
              "transfer near 0.5 and lowers the within-AgentDojo AUROC of LR-C and LR-B.", ""]
    # headline table
    L += ["## AUROC by setting (pool E)", "",
          "Point estimate with 95% stratified bootstrap interval [`settings.<s>.metrics.<detector>.AUROC`]. The file also "
          "gives AUPRC, recall and FPR at t_r80 and t05, and failed-attack flag rates.", ""]
    dets = ["LR-A", "LR-B", "LR-C", "LR-C-drop", "LR-B-noobs", "LR-C-noobs", "LR-B-minus", "Markov", "IForest", "TFIDF", "TFIDF-nodigit"]
    L += ["| Setting | " + " | ".join(dets) + " |", "|---|" + "---|" * len(dets)]
    for s in SETTINGS:
        L.append(f"| {s} | " + " | ".join(fmt(S[s]["metrics"][d]["AUROC"]) for d in dets) + " |")
    r1 = S["R1"]["metrics"]
    L += ["", f"R1 TFIDF-worldmask: {fmt(r1['TFIDF-worldmask']['AUROC'])}.", "",
          "## Implementation notes (also logged in revision/DEVIATIONS.md)", ""]
    L += [f"- {x}" for x in V["implementation_notes"]]
    L += ["", "## Problems and caveats", ""] + [f"- {x}" for x in V["problems"]]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--budget", type=float, default=500.0)
    a = ap.parse_args(); t0 = time.time(); os.makedirs(VDIR, exist_ok=True)
    M = {}
    for i, s in enumerate(SETTINGS):
        p = os.path.join(VDIR, f"{s}.json"); pk = os.path.join(CACHE, f"{s}.pkl")
        if os.path.exists(p) and os.path.getmtime(p) > os.path.getmtime(pk):
            M[s] = json.load(open(p)); continue
        if time.time() - t0 > a.budget:
            print("budget used; rerun to continue", flush=True); print("INCOMPLETE"); return
        ts = time.time(); M[s] = setting_metrics(s, i)
        json.dump(M[s], open(p, "w"), indent=1); print(f"{s}: metrics in {time.time() - ts:.0f} s", flush=True)
    A = anchors(M); R = r4_values(M)
    req = [x for x in A if x["required"]]
    fc_all = all(v is True or isinstance(v, str) for s in SETTINGS for v in M[s]["format_checks"].values())
    shas = {M[s]["provenance"].get("script_sha256") for s in SETTINGS}
    wit = M["WIT"]["fold_assignment"]; wut = M["WUT"]["fold_assignment"]
    wm = M["R1"]
    notes = [
        "Fitting follows ablation.py / rerun_r1r3.py exactly; the cache only makes mathematically tied scores bitwise "
        "tied (LR-*, IForest and TFIDF* are scored once per distinct feature row or text and broadcast; Markov scores "
        "are rounded to 12 decimals, since a mean of identical log-probabilities depends on the trajectory length in "
        "floating point). Scores differ from the v0.3 code paths by at most "
        f"{max(M[s]['reproduction_checks']['max_abs_score_diff'] for s in SETTINGS):.1e}.",
        "Unsuccessful AgentDojo attacks are placed in the fold of their suite|user_task group. Groups that no P0 row "
        "defines are assigned with np.random.default_rng(10000 + seed), position mod 5: "
        f"{wut[0]['n_extra_groups']} group ({wut[0]['n_failed_rows_in_extra_groups']} row) in WUT and WUT_F; "
        f"{wit[0]['n_extra_groups']} groups ({wit[0]['n_failed_rows_in_extra_groups']} rows) in WIT, where P0 attacks "
        "are grouped by injection goal, so user tasks with no benign P0 run have no P0 group.",
        "WUT_F uses the WUT fold assignment; the unsuccessful attacks of the four training folds are added as label-0 "
        "training rows (they do not enter the benign tool vocabulary, the Markov model or the IForest).",
        "TFIDF-worldmask replaces whole world string values (user, email, company, contact name, email and relation, "
        "date), case-insensitively, as whole occurrences. Mean replacements per trajectory "
        f"{wm['worldmask_replacements_per_traj']['mean']:.2f}; {wm['worldmask_replacements_per_traj']['frac_traj_zero']:.3f} "
        "of trajectories contain no world string; a first- or last-name token of the user or a contact survives in "
        f"{wm['worldmask_frac_traj_with_residual_name_token']:.3f} of trajectories (partial names are not world values).",
        "In the AgentDrift files `args` is a dict, not a string; the text uses str(args) in both benchmarks, and the "
        "v0.3 argument-size features (len(args)) therefore count argument keys in both.",
        "Coverage `cov` in test_meta for CV settings is the mean over repetitions (per-repetition values in `cov_reps`).",
        "Metric intervals in this file are trajectory-level stratified bootstraps (v0.3 style). The plan's cluster "
        "bootstraps and prevalence-weighted AUPRC are left to the downstream analyses.",
    ]
    tpath = os.path.join(CACHE, "tie_check.json")
    tie = json.load(open(tpath)) if os.path.exists(tpath) else None
    problems = []
    if tie:
        big = [f"{st} {d}: v0.3 code path {v['AUROC_raw_v03_path']:.4f}, cache {v['AUROC_cache']:.4f}"
               for st, row in tie["settings"].items() for d, v in row.items() if abs(v["diff"]) >= 0.0005]
        problems.append("Float noise in tied scores changes AUROC in the v0.3 code paths when coverage is low "
                        "(tie_check.json; cache values are tie-exact): " + "; ".join(big) + ".")
    for x in req:
        if not x["match_3dp"]:
            problems.append(f"Anchor not reproduced: {x['setting']} {x['detector']} expected {x['expected_3dp']}, got {x['reproduced']}.")
    if not fc_all:
        problems.append("Some format checks fail; see settings.<s>.format_checks.")
    if len(shas) != 1:
        problems.append(f"Pickles come from different script versions: {sorted(map(str, shas))}.")
    problems += [
        "R4 values do not match the review's corrected estimates because the corrected extraction also removes the "
        "2,000-character truncation (see the truncation check); they are not a reproduction failure.",
        "TFIDF reaches AUROC near 1 in R1 (" + fmt(M["R1"]["metrics"]["TFIDF"]["AUROC"]) + ") and TFIDF-worldmask barely "
        "changes it (" + fmt(M["R1"]["metrics"]["TFIDF-worldmask"]["AUROC"]) + "); the content detector separates the "
        "AgentDrift classes almost perfectly in-distribution.",
        "When the held-out tool vocabulary is disjoint (R3 folds, LOSO_banking, LOSO_slack) LR-A and Markov give every "
        "test row the same score: AUROC is exactly 0.5, LR-A flags no trajectory at t_r80 or t05 (recall 0, FPR 0) "
        "and Markov flags every trajectory at t_r80 (recall 1, FPR 1) [settings.<s>.metrics.<d>.recall@t_r80 / FPR@t_r80].",
    ]
    V = {"validated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "build_script_sha256": sorted(map(str, shas))[0],
         "all_format_checks_pass": fc_all,
         "reproduction_checks_total": {"n": sum(M[s]["reproduction_checks"]["n"] for s in SETTINGS),
                                       "n_pass": sum(M[s]["reproduction_checks"]["n_pass"] for s in SETTINGS),
                                       "max_abs_score_diff": max(M[s]["reproduction_checks"]["max_abs_score_diff"] for s in SETTINGS)},
         "anchor_summary": {"required_n": len(req), "required_match": sum(bool(x["match_3dp"]) for x in req),
                            "additional_n": len(A) - len(req),
                            "additional_equal_4dp": sum(1 for x in A if not x["required"] and x["abs_diff_vs_archived"] == 0)},
         "anchors": A, "r4": R, "tie_check": tie, "settings": M, "implementation_notes": notes, "problems": problems}
    json.dump(V, open(os.path.join(CACHE, "validation.json"), "w"), indent=1)
    open(os.path.join(CACHE, "CACHE_README.md"), "w", encoding="utf-8").write(readme(V))
    print("anchors:", V["anchor_summary"], "format ok:", fc_all, "checks:", V["reproduction_checks_total"])
    print("wrote validation.json and CACHE_README.md"); print("ALL_DONE")


if __name__ == "__main__":
    main()
