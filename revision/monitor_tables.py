"""Tables for the AlignmentCheck monitor arms (MONITOR_PREREGISTRATION.md Section 11).

Formatting of revision/monitor_results.json, plus two estimates computed with the paper's own estimators so that the
monitor rows of the main tables match the columns they enter. Neither changes any rule or reading of the plan.
  (1) Table II rows (R1 pool E). wAUPRC at 1, 2, 5 and 10%, AUROC, recall, FPR, precision and alerts per 1,000 with
      prevalence_ops.ranked_block, rate_block, eq6 and alerts, on the bootstrap replicates of prevalence_ops.m8_r1
      (stratified by class, seed BASE + 1). A check reproduces the published TF-IDF AUROC interval of Table II.
  (2) Table VI rows (AgentDojo). P0 and P0F AUROC with the suite|user_task cluster bootstrap of content_detector
      (ut_cluster, seed_of("R4", "ut_cluster") for P0 as in monitor_results.json, seed_of("P0F", "ut_cluster") for P0F).
Monitor scores are rebuilt with the functions of the frozen monitor_detector.py (imported, not modified).
Writes revision/monitor_tables/: s_monitor_ladder.{tex,md}, s_monitor_arms.{tex,md}, colspecs.json,
monitor_table_values.json. Row builders for Table II and Table VI are importable (prevalence_rows, r4_rows).
Usage: python monitor_tables.py [--synthetic]
"""
import os, sys, json, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "paper"))
import monitor_detector as md          # noqa: E402
import content_detector as cd          # noqa: E402
import prevalence_ops as po            # noqa: E402

OUT = os.path.join(HERE, "monitor_tables")
SYN = False
LABS = {"MON-L": "MON-L (Llama 3.3 70B w4a16)", "MON-L-ex": "MON-L, error steps excluded", "MON-L-th": "MON-L-th (with thought)",
        "MON-A": "MON-A (Claude Sonnet 5.5)", "MON-A-ex": "MON-A, error steps excluded"}
SET_LAB = {"WUT": "Within UT", "WUT_F": "Within UT-F", "WIT": "Within IT"}


def res_path():
    return os.path.join(HERE, "monitor_results_synthetic.json" if SYN else "monitor_results.json")


def vals_path():
    return os.path.join(OUT, "monitor_table_values_synthetic.json" if SYN else "monitor_table_values.json")


# ============================================================================= derived estimates
def build_traces():
    md.SYN = SYN; md.prep_data()
    A = {"T_L": md.traces(md.L_GROUPS, md.load_l, graded=True)}
    have_a = SYN or all(os.path.exists(os.path.join(md.API, f"results_{g}.jsonl")) for g in md.A_GROUPS)
    A["T_A"] = md.traces(md.A_GROUPS, md.load_a, graded=False) if have_a else None
    return A


def r1_prevalence_values(A):
    lin = cd.load_pkl(os.path.join(cd.LIN, "R1.pkl")); tm = lin["test_meta"]; ids = [m["id"] for m in tm]
    y = np.asarray(lin["y"], int); E = np.where(np.asarray(lin["pool_E"], bool))[0]; yE = y[E]
    rng = np.random.default_rng(po.BASE + 1)
    W = po.w_strata(np.where(yE == 1, "pos", "neg"), po.B, rng)
    choice = np.zeros(po.B, int)
    out = {"bootstrap": f"prevalence_ops.m8_r1 replicates (stratified, B = {po.B}, seed {po.BASE + 1})", "rows": {}}
    # check: the TF-IDF row of Table II is reproduced on these replicates
    S_tf = np.atleast_2d(np.asarray(lin["scores"]["TFIDF"], float))
    bo, pt = po.ranked_block(S_tf, y, E, W, po.PIS)
    ref = po_ref()["M8_R1"]["detectors"]["TFIDF"]["AUROC"]
    mine = po.summarize_det(bo, pt, choice, ["AUROC"])["AUROC"]
    out["check_TFIDF_AUROC_equals_Table_II"] = bool(abs(mine["lo"] - ref["lo"]) < 1e-6 and abs(mine["hi"] - ref["hi"]) < 1e-6)
    P80, _ = md.thresholds("R1", ids, A["T_L"], "s")
    arms = {"MON-L": (np.array([A["T_L"][i]["s"] for i in ids])[None], {"t_r80": P80, "native": np.array([A["T_L"][i]["nat"] for i in ids])[None]})}
    if A["T_A"] is not None:
        arms["MON-A"] = (np.array([A["T_A"][i]["s"] for i in ids])[None], {"native": np.array([A["T_A"][i]["nat"] for i in ids])[None]})
    masks = {"TPR": yE == 1, "FPR": yE == 0}
    for arm, (S, preds) in arms.items():
        bo, pt = po.ranked_block(S, y, E, W, po.PIS)
        e = {"AUROC": po.summarize_det(bo, pt, choice, ["AUROC"])["AUROC"], "wAUPRC": po.summarize_det(bo, pt, choice, list(po.PIS))}
        for op, P in preds.items():
            rb, rp = po.rate_block(P[:, E], masks, W)
            tb = po.pick(rb["TPR"], choice); fb = po.pick(rb["FPR"], choice)
            o = po.summarize_det(rb, rp, choice, ["TPR", "FPR"], ns={"TPR": int(masks["TPR"].sum()), "FPR": int(masks["FPR"].sum())})
            for k, pi in po.PIS.items():
                o[f"precision_{k}"] = po.est(float(po.eq6(pi, rp["TPR"], rp["FPR"])[0]), po.eq6(pi, tb, fb))
                o[f"alerts_per_1000_{k}"] = po.est(float(po.alerts(pi, rp["TPR"], rp["FPR"])[0]), po.alerts(pi, tb, fb))
            e[op] = o
        out["rows"][arm] = e
    return out


_PO = {}


def po_ref():
    if not _PO:
        _PO.update(json.load(open(os.path.join(HERE, "prevalence_ops_results.json"), encoding="utf-8")))
    return _PO


def dojo_cluster_values(A):
    lin = cd.load_pkl(os.path.join(cd.LIN, "R4.pkl")); tm = lin["test_meta"]; ids = [m["id"] for m in tm]
    y = np.asarray(lin["y"], int); ALL = np.arange(len(y))
    out = {}
    dets = {"MON-L": np.array([A["T_L"][i]["s"] for i in ids])}
    if A["T_A"] is not None:
        dets["MON-A"] = np.array([A["T_A"][i]["s"] for i in ids])
    fn, binfo = cd.make_index_fn("ut_cluster", y, cd.keys_for("ut_cluster", tm, ALL))
    seed = cd.seed_of("P0F", "ut_cluster"); rng = np.random.default_rng(seed)
    bs = {d: np.empty(md.B) for d in dets}
    for b in range(md.B):
        ib = fn(rng)
        for d, s in dets.items():
            bs[d][b] = ek_auroc(y[ib], s[ib])
    for d, s in dets.items():
        out[d] = {"P0F_AUROC_cluster": cd.trip(float(ek_auroc(y, s)), bs[d])}
    out["bootstrap"] = {"scheme": "ut_cluster", "seed": seed, "B": md.B, **binfo}
    return out


def ek_auroc(y, s):
    return cd.rank_block(np.asarray(y, int), np.asarray(s, float))["AUROC"]


# ============================================================================= cell helpers (gen_tables_v04 Cells)
def T(gt, x, sign=False, mark=""):
    if x is None or (isinstance(x, (list, tuple)) and (x[0] is None)):
        return gt.NA
    if isinstance(x, dict):
        if x.get("point") is None and x.get("delta") is None:
            return gt.NA
        p = x.get("point", x.get("delta")); return gt.CP(p, x["lo"], x["hi"], sign=sign, mark=mark)
    return gt.CP(x[0], x[1], x[2], sign=sign, mark=mark)


def D(gt, e):
    if not isinstance(e, dict) or e.get("delta") is None:
        return gt.NA
    return gt.CP(e["delta"], e["lo"], e["hi"], sign=True, mark="*" if e.get("detected") else "")


def outcome_line(o, arm):
    m1 = o["M1"].get(arm, {}).get("met"); m2 = o["M2"].get(arm, {}).get("met")
    if m1:
        return "M1, transfers"
    if m2:
        return "M2, no transfer"
    return "Neither M1 nor M2"


# ============================================================================= rows for the main tables
def prevalence_rows(gt):
    """Rows appended to Table II (v04_prevalence), after a rule, in its 10 value columns."""
    V = json.load(open(vals_path(), encoding="utf-8"))["r1_prevalence"]["rows"]
    rows = [gt.MID]
    for arm, ops in (("MON-L", ("t_r80", "native")), ("MON-A", ("native",))):
        if arm not in V:
            continue
        e = V[arm]
        for op in ops:
            o = e[op]
            lab = {"t_r80": f"{arm} at " + "θr80", "native": f"{arm}, native decision"}[op]
            labt = {"t_r80": f"{arm} at $\\theta_{{r80}}$", "native": f"{arm}, native decision"}[op]
            score = [T(gt, e["AUROC"])] + [T(gt, e["wAUPRC"][k]) for k in gt.PK]
            if arm == "MON-A":
                score = [gt.R(c.tex() + r"$^{\ddagger}$", c.md() + " ‡") if c.raw is None else c for c in score]
            if op == "native" and arm == "MON-L":
                score = [gt.R("as above", "as above")] + [gt.R("", "")] * len(gt.PK)
            cells = score + [T(gt, o["precision_pi02"]), T(gt, o["precision_pi10"]), gt.pt(T(gt, o["TPR"])), gt.pt(T(gt, o["FPR"])),
                             gt.pt(gt.CP(o["alerts_per_1000_pi02"]["point"], o["alerts_per_1000_pi02"]["lo"], o["alerts_per_1000_pi02"]["hi"], dec=0))]
            rows.append(gt.row(labt, cells, lab))
    return rows


def r4_rows(gt):
    """Rows appended to Table VI (v04_r4), 7 value columns."""
    R = json.load(open(res_path(), encoding="utf-8")); V = json.load(open(vals_path(), encoding="utf-8"))["dojo_cluster"]
    r4s = R["settings"]["R4"]["detectors"]; o = R["outcomes"]; rows = []
    for arm in ("MON-L", "MON-A"):
        if arm not in r4s:
            continue
        d = r4s[arm]
        cells = [T(gt, d["AUROC"]), gt.R(r"\makecell{untrained,\\as transfer}", "untrained, as transfer"), gt.NA, gt.NA,
                 T(gt, V[arm]["P0F_AUROC_cluster"]), T(gt, d["AP_w10"]), gt.R(outcome_line(o, arm))]
        rows.append(gt.row(LABS[arm], cells))
    return rows


# ============================================================================= supplement tables
def s_ladder(gt, R):
    rows = [gt.sub("(a) MON-L at each rung. Native is the released decision, $\\theta_{r80}$ is read from the setting's training positives",
                   "(a) MON-L at each rung. Native is the released decision, θr80 is read from the setting's training positives")]
    for S, r in R["settings"].items():
        d = r["detectors"]["MON-L"]
        rows.append(gt.row(SET_LAB.get(S, S.replace("_", " ").replace("R1wd s", "R1-wd split ").replace("LOSO ", "LOSO ")),
                           [T(gt, d["AUROC"]), T(gt, d["AP_w02"]), T(gt, d["AP_w10"]), T(gt, d["recall_t05"]), T(gt, d["FPR_t05"]),
                            T(gt, d["prec02_t05"]), T(gt, d["recall_t_r80"]), T(gt, d["FPR_t_r80"])]))
    rows.append(gt.hdr(["(b) MON-L $-$ comparator, AUROC", "TF-IDF", "LR-C", "LR-B", "GRU", "SE", "", "", ""],
                       ["(b) MON-L − comparator, AUROC", "TF-IDF", "LR-C", "LR-B", "GRU", "SE", "", "", ""]))
    for S, r in R["settings"].items():
        cells = [D(gt, r["paired"].get(f"MON-L - {c}", {}).get("AUROC")) for c in md.COMPARATORS] + [gt.R("", "")] * 3
        rows.append(gt.row(SET_LAB.get(S, S.replace("_", " ")), cells))
    ht = (r"Setting & AUROC & \makecell{wAUPRC\\2\%} & \makecell{wAUPRC\\10\%} & \makecell{Recall,\\native} & \makecell{FPR,\\native} & "
          r"\makecell{Precision 2\%,\\native} & \makecell{Recall at\\$\theta_{r80}$} & \makecell{FPR at\\$\theta_{r80}$} \\")
    hm = ["Setting", "AUROC", "wAUPRC 2%", "wAUPRC 10%", "Recall, native", "FPR, native", "Precision 2%, native", "Recall at θr80", "FPR at θr80"]
    gt.write("s_monitor_ladder", ht, hm, rows, "table*", gt.CS(8, "3pt"))


def s_arms(gt, R):
    rows = []
    for S, lab in (("R1", "(a) AgentDrift R1 test, pool E"), ("R4", "(b) AgentDojo P0")):
        rows.append(gt.sub(lab))
        for arm in ("MON-L", "MON-L-ex", "MON-L-th", "MON-A", "MON-A-ex"):
            d = R["settings"][S]["detectors"].get(arm)
            if d is None:
                continue
            rows.append(gt.row(LABS[arm], [T(gt, d["AUROC"]), T(gt, d["AP_w02"]), T(gt, d["recall_t05"]), T(gt, d["FPR_t05"]),
                                           T(gt, d["prec02_t05"]), T(gt, d.get("recall_t_r80")), T(gt, d.get("FPR_t_r80")), T(gt, d.get("prec02_t_r80"))]))
    P = R["agentdojo_populations"]
    cols = [a for a in ("MON-L", "MON-L-th", "MON-A", "TFIDF_transfer", "LR-C_transfer", "GRU_transfer", "SE_transfer") if a in P["P0"]["metrics"]]
    names = {"MON-L": "MON-L", "MON-L-th": "MON-L-th", "MON-A": "MON-A", "TFIDF_transfer": "TF-IDF", "LR-C_transfer": "LR-C",
             "GRU_transfer": "GRU", "SE_transfer": "SE"}
    rows.append(gt.hdr(["(c) Population, AUROC"] + [names[c] for c in cols] + [""] * (8 - len(cols)),
                       ["(c) Population, AUROC"] + [names[c] for c in cols] + [""] * (8 - len(cols))))
    for pop, lab in (("P0", "P0"), ("C1", "C1"), ("P0F", "P0F"), ("model_balanced", "model-balanced")):
        rows.append(gt.row(lab, [T(gt, P[pop]["metrics"][c]["AUROC"]) for c in cols] + [gt.R("", "")] * (8 - len(cols))))
    o = R["outcomes"]
    rows.append(gt.hdr(["(d) Unsuccessful attacks", "Flag rate", "Recall on P0", "Ratio", "\\makecell{Monitor $-$\\\\TF-IDF}", "", "", "", ""],
                       ["(d) Unsuccessful attacks", "Flag rate", "Recall on P0", "Ratio", "Monitor − TF-IDF", "", "", "", ""]))
    for key, comp in (("M5", "R4"), ("M5b", "WUT")):
        for arm in ("MON-L", "MON-A"):
            e = o[key].get(arm)
            if not e:
                continue
            q = e["ratio_failed_flag_to_P0_recall"]
            rows.append(gt.row(f"{arm}, native ({key})", [T(gt, e["flag_rate_monitor_native"]), gt.P(q["recall_P0_monitor"]),
                                                       gt.P(q["monitor"]) if q["monitor"] is not None else gt.NA, D(gt, e["difference"])] + [gt.R("", "")] * 4))
        e = o[key]["MON-L"]; q = e["ratio_failed_flag_to_P0_recall"]
        rows.append(gt.row(f"TF-IDF at {'R4' if comp == 'R4' else 'Within UT'} " + r"$\theta_{r80}$",
                           [T(gt, e["flag_rate_TFIDF"]), gt.P(q["recall_P0_TFIDF"]), gt.P(q["TFIDF"]) if q["TFIDF"] is not None else gt.NA, gt.R("", "")] + [gt.R("", "")] * 4,
                           f"TF-IDF at {'R4' if comp == 'R4' else 'Within UT'} θr80"))
    M8 = R["M8_masking"]["levels"]
    rows.append(gt.hdr(["(e) Tool names kept", "\\makecell{Achieved\\\\coverage}", "\\makecell{Mean AUROC\\\\over draws}", "\\makecell{Lowest\\\\draw}", "\\makecell{Highest\\\\draw}", "", "", "", ""],
                       ["(e) Tool names kept", "Achieved coverage", "Mean AUROC over draws", "Lowest draw", "Highest draw", "", "", "", ""]))
    for lev in ("1.0", "0.75", "0.5", "0.25", "0.0"):
        e = M8[lev]; cov = [x.get("coverage") for x in e["draws"] if isinstance(x.get("coverage"), (int, float))]
        au = [x["AUROC"] for x in e["draws"]]
        rows.append(gt.row(lev, [gt.P(float(np.mean(cov))) if cov else gt.NA, T(gt, e["AUROC_mean_over_draws"]), gt.P(min(au)), gt.P(max(au))] + [gt.R("", "")] * 4))
    ht = (r" & AUROC & \makecell{wAUPRC\\2\%} & \makecell{Recall,\\native} & \makecell{FPR,\\native} & \makecell{Precision 2\%,\\native} & "
          r"\makecell{Recall at\\$\theta_{r80}$} & \makecell{FPR at\\$\theta_{r80}$} & \makecell{Precision 2\%\\at $\theta_{r80}$} \\")
    hm = ["Arm", "AUROC", "wAUPRC 2%", "Recall, native", "FPR, native", "Precision 2%, native", "Recall at θr80", "FPR at θr80", "Precision 2% at θr80"]
    gt.write("s_monitor_arms", ht, hm, rows, "table*", gt.CS(8, "3pt"))


def main():
    global SYN
    ap = argparse.ArgumentParser(); ap.add_argument("--synthetic", action="store_true"); a = ap.parse_args(); SYN = a.synthetic
    os.makedirs(OUT, exist_ok=True)
    A = build_traces()
    V = {"synthetic": SYN, "monitor_detector_sha256": md.sha256(md.__file__),
         "r1_prevalence": r1_prevalence_values(A), "dojo_cluster": dojo_cluster_values(A)}
    json.dump(cd.jsonable(V), open(vals_path(), "w", encoding="utf-8"), indent=1)
    import gen_tables_v04 as gt
    gt.OUT = OUT if not SYN else os.path.join(OUT, "synthetic"); os.makedirs(gt.OUT, exist_ok=True)
    R = json.load(open(res_path(), encoding="utf-8"))
    s_ladder(gt, R); s_arms(gt, R)
    json.dump({k: gt.COLSPECS[k] for k in ("s_monitor_ladder", "s_monitor_arms")}, open(os.path.join(gt.OUT, "colspecs.json"), "w"), indent=1)
    print(json.dumps({"check_TFIDF_AUROC_equals_Table_II": V["r1_prevalence"]["check_TFIDF_AUROC_equals_Table_II"], "out": gt.OUT}))


if __name__ == "__main__":
    main()
