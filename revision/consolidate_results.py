"""Consolidation helper for RESULTS_SUMMARY.md (revision, 2026-09-27).

Reads the seven verified stream result files (read-only), pulls the values quoted in
RESULTS_SUMMARY.md, and writes them with their source file and key path to
revision/RESULTS_SUMMARY_values.json. Every number in RESULTS_SUMMARY.md is one of these
values (rounded to 3 decimals unless stated) or a count stated in a stream summary.

It also compares the R4 outputs of the v0.3 scripts re-run on the corrected extraction
(agent-ids/r4_control_results.json, ablation_results.json, gru_results.json; final when
revision/rerun_v03_scripts.log contains 'gru_ladder exit 0') with the shared caches.

Run: C:/Python310/python.exe revision/consolidate_results.py
No model is fitted and no stream file is modified.
"""
import json, os, pickle, sys, hashlib, datetime
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

FILES = {
    "PO": "prevalence_ops_results.json",
    "DS": "domain_shift_results.json",
    "OOV": "oov_mechanism_results.json",
    "CD": "content_detector_results.json",
    "FO": "features_obslen_results.json",
    "R4D": "r4_deep_results.json",
    "DG": "diagnostics_results.json",
}
_cache = {}


def load(tag):
    if tag not in _cache:
        with open(os.path.join(HERE, FILES[tag]), encoding="utf-8") as fh:
            _cache[tag] = json.load(fh)
    return _cache[tag]


def get(tag, path):
    d = load(tag)
    for p in path.split(" > "):
        if isinstance(d, list):
            d = d[int(p)]
        else:
            d = d[p]
    return d


def triple(v):
    """Return (point, lo, hi) or None for the value formats used by the streams."""
    if isinstance(v, list) and len(v) == 3 and all(isinstance(x, (int, float)) or x is None for x in v):
        return tuple(v)
    if isinstance(v, dict):
        pt = None
        for k in ("point", "delta", "value", "mean"):
            if k in v and isinstance(v[k], (int, float)):
                pt = v[k]
                break
        if "lo" in v and "hi" in v:
            return (pt, v["lo"], v["hi"])
        if "ci" in v and isinstance(v["ci"], list) and len(v["ci"]) == 2:
            return (pt, v["ci"][0], v["ci"][1])
    return None


def fmt(v, nd=3):
    t = triple(v)
    if t is not None and t[0] is not None:
        sg = any(isinstance(x, (int, float)) and x < 0 for x in t)
        f = (lambda x: f"{x:+.{nd}f}") if sg else (lambda x: f"{x:.{nd}f}")
        return f"{f(t[0])} [{f(t[1])}, {f(t[2])}]"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    if isinstance(v, (int, str, bool)) or v is None:
        return str(v)
    if isinstance(v, list):
        return ", ".join(fmt(x, nd) for x in v)
    return json.dumps(v)[:300]


# (id, file tag, key path, note)
SPEC = [
    # ---------------- M8 / RQ1 (prevalence_ops) ----------------
    ("M8.R1.LR-C.AUROC", "PO", "M8_R1 > detectors > LR-C > AUROC", ""),
    ("M8.R1.LR-C.wAUPRC.pi01", "PO", "M8_R1 > detectors > LR-C > wAUPRC > pi01", ""),
    ("M8.R1.LR-C.wAUPRC.pi02", "PO", "M8_R1 > detectors > LR-C > wAUPRC > pi02", ""),
    ("M8.R1.LR-C.wAUPRC.pi05", "PO", "M8_R1 > detectors > LR-C > wAUPRC > pi05", ""),
    ("M8.R1.LR-C.wAUPRC.pi10", "PO", "M8_R1 > detectors > LR-C > wAUPRC > pi10", ""),
    ("M8.R1.LR-B.wAUPRC.pi02", "PO", "M8_R1 > detectors > LR-B > wAUPRC > pi02", ""),
    ("M8.R1.LR-A.wAUPRC.pi02", "PO", "M8_R1 > detectors > LR-A > wAUPRC > pi02", ""),
    ("M8.R1.Markov.wAUPRC.pi02", "PO", "M8_R1 > detectors > Markov > wAUPRC > pi02", ""),
    ("M8.R1.IForest.wAUPRC.pi02", "PO", "M8_R1 > detectors > IForest > wAUPRC > pi02", ""),
    ("M8.R1.GRU.AUROC", "PO", "M8_R1 > detectors > GRU > AUROC", ""),
    ("M8.R1.GRU.wAUPRC.pi02", "PO", "M8_R1 > detectors > GRU > wAUPRC > pi02", ""),
    ("M8.R1.GRU.wAUPRC.pi10", "PO", "M8_R1 > detectors > GRU > wAUPRC > pi10", ""),
    ("M8.R1.TFIDF.wAUPRC.pi02", "PO", "M8_R1 > detectors > TFIDF > wAUPRC > pi02", ""),
    ("M8.R1.chance.pi02", "PO", "M8_R1 > chance_wAUPRC_random_scorer > pi02", ""),
    ("M8.R1.paired.pi01.LR-C-minus-Markov", "PO", "M8_R1 > paired > pi01 > LR-C-minus-Markov", ""),
    ("M8.R1.paired.pi02.LR-C-minus-Markov", "PO", "M8_R1 > paired > pi02 > LR-C-minus-Markov", ""),
    ("M8.R1.paired.pi01.LR-C-minus-LR-B", "PO", "M8_R1 > paired > pi01 > LR-C-minus-LR-B", ""),
    ("M8.R1.paired.pi02.LR-C-minus-LR-B", "PO", "M8_R1 > paired > pi02 > LR-C-minus-LR-B", ""),
    ("M8.R1.paired.pi02.LR-C-minus-LR-B.p_holm", "PO", "M8_R1 > paired > pi02 > LR-C-minus-LR-B > p_holm", ""),
    ("M8.R1.paired.pi01.GRU-minus-LR-C", "PO", "M8_R1 > paired > pi01 > GRU-minus-LR-C", ""),
    ("M8.R1.paired.pi02.GRU-minus-LR-C", "PO", "M8_R1 > paired > pi02 > GRU-minus-LR-C", ""),
    ("M8.R1.paired.pi02.GRU-minus-LR-C.p_holm", "PO", "M8_R1 > paired > pi02 > GRU-minus-LR-C > p_holm", ""),
    ("M8.R1.n_detected_holm", "PO", "M8_R1 > n_detected_holm", ""),
    ("M8.R1.subsampling.pi02.LR-C", "PO", "M8_R1 > subsampling_v03_secondary > pi02 > AP > LR-C", "secondary estimator"),
    ("M8.R1.subsampling.pi02.paired.LR-C-minus-Markov", "PO", "M8_R1 > subsampling_v03_secondary > pi02 > paired_AP > LR-C-minus-Markov", ""),
    ("M8.width_ratio", "PO", "summary_derived > M8_LR-C_pi02_width_ratio_subsampling_over_weighted", ""),
    ("M8.AgentDojo.R4.LR-C.pi10", "PO", "M8_AgentDojo > schemes > traj_strat > settings > R4 > detectors > LR-C > pi10", "rounded scores"),
    ("M8.AgentDojo.WUT.LR-C.pi10", "PO", "M8_AgentDojo > schemes > traj_strat > settings > WUT > detectors > LR-C > pi10", "rounded scores"),
    ("M8.AgentDojo.WUT.GRU.pi10", "PO", "M8_AgentDojo > schemes > traj_strat > settings > WUT > detectors > GRU > pi10", ""),
    ("M8.AgentDojo.WUT.TFIDF.pi10", "PO", "M8_AgentDojo > schemes > traj_strat > settings > WUT > detectors > TFIDF > pi10", ""),
    ("M8.AgentDojo.chance.pi10", "PO", "M8_AgentDojo > chance_wAUPRC_random_scorer > pi10", ""),
    # ---------------- Eq. 6 operating points ----------------
    ("OP.R1.LR-C.TPR", "PO", "R1_operating_points > detectors > LR-C > t_r80 > TPR", ""),
    ("OP.R1.LR-C.FPR", "PO", "R1_operating_points > detectors > LR-C > t_r80 > FPR", ""),
    ("OP.R1.LR-C.prec.pi02", "PO", "R1_operating_points > detectors > LR-C > t_r80 > precision_pi02", ""),
    ("OP.R1.LR-C.prec.pi10", "PO", "R1_operating_points > detectors > LR-C > t_r80 > precision_pi10", ""),
    ("OP.R1.LR-C.alerts.pi02", "PO", "R1_operating_points > detectors > LR-C > t_r80 > alerts_per_1000_pi02", ""),
    ("OP.R1.LR-C.false_per_true.pi02", "PO", "summary_derived > EQ6_LR-C_t_r80_false_alerts_per_true_alert_pi02", ""),
    ("OP.R1.LR-C.prec.pi0.1pct", "PO", "R1_operating_points > detectors > LR-C > t_r80 > precision_pi0.1pct", ""),
    ("OP.R1.LR-C.prec.pi0.01pct", "PO", "R1_operating_points > detectors > LR-C > t_r80 > precision_pi0.01pct", ""),
    ("OP.R1.GRU.prec.pi02", "PO", "R1_operating_points > detectors > GRU > t_r80 > precision_pi02", ""),
    ("OP.R1.GRU.alerts.pi02", "PO", "R1_operating_points > detectors > GRU > t_r80 > alerts_per_1000_pi02", ""),
    ("OP.R1.TFIDF.prec.pi02.lower", "CD", "zero_fp_bounds > R1 > TFIDF > prec02_lower_at_FPR_upper", ""),
    # ---------------- M9 ----------------
    ("M9.R1.LR-C.FPR_benign", "PO", "M9 > R1 > schemes > traj_strat > detectors > LR-C > t_r80 > FPR_benign", ""),
    ("M9.R1.LR-C.FPR_hard", "PO", "M9 > R1 > schemes > traj_strat > detectors > LR-C > t_r80 > FPR_hard_negative", ""),
    ("M9.R1.LR-C.flag_failed", "PO", "M9 > R1 > schemes > traj_strat > detectors > LR-C > t_r80 > flag_failed_attack", ""),
    ("M9.R1.LR-C.recall_full", "PO", "M9 > R1 > schemes > traj_strat > detectors > LR-C > t_r80 > recall_full", ""),
    ("M9.R1.LR-C.recall_partial", "PO", "M9 > R1 > schemes > traj_strat > detectors > LR-C > t_r80 > recall_partial", ""),
    ("M9.R1.LR-C.recall_delayed", "PO", "M9 > R1 > schemes > traj_strat > detectors > LR-C > t_r80 > recall_delayed", ""),
    ("M9.R1.LR-C.prec_eq6", "PO", "M9 > R1 > schemes > traj_strat > detectors > LR-C > t_r80 > precision_eq6", ""),
    ("M9.R1.ratio_benign_over_hard", "PO", "summary_derived > M9_R1_LR-C_pi02_precision_ratio_benign_only_over_hard_only", ""),
    ("M9.R1.paired_sub.LR-C-minus-LR-B", "PO", "M9 > R1 > schemes > traj_strat > paired_AUROC_subgroups > LR-C-minus-LR-B", ""),
    ("M9.R4.LR-C.t_r80", "PO", "M9 > R4 > schemes > traj_strat > detectors > LR-C > t_r80", ""),
    ("M9.WUT.TFIDF.t_r80", "PO", "M9 > WUT > schemes > traj_strat > detectors > TFIDF > t_r80", ""),
    ("M9.WUT.TFIDF.AUROC_vs_failed", "PO", "M9 > WUT > schemes > traj_strat > detectors > TFIDF > AUROC_subgroups > poolE_vs_failed", ""),
    # ---------------- m4 / m10 / m11 / m9 ----------------
    ("m4.GRU.t_val80", "PO", "m4_gru_t_val80 > table > R1 > GRU > t_val80", ""),
    ("m4.GRU.t_r80", "PO", "m4_gru_t_val80 > table > R1 > GRU > t_r80", ""),
    ("m4.FPRatTPR80.LR-C", "PO", "M8_R1 > detectors > LR-C > FPR_at_test_TPR80 > FPR_at_TPR80", "uses test labels; descriptive"),
    ("m4.FPRatTPR80.GRU", "PO", "M8_R1 > detectors > GRU > FPR_at_test_TPR80 > FPR_at_TPR80", "uses test labels; descriptive"),
    ("m4.theta05.R1.GRU", "PO", "m4_theta05 > table > R1 > GRU", ""),
    ("m10.LR-C.bh.fpr01", "PO", "R1_operating_points > m10_fpr_budgets > detectors > LR-C > benign_hard > fpr01", ""),
    ("m10.LR-C.bh.fpr05", "PO", "R1_operating_points > m10_fpr_budgets > detectors > LR-C > benign_hard > fpr05", ""),
    ("m10.GRU.bh.fpr01", "PO", "R1_operating_points > m10_fpr_budgets > detectors > GRU > benign_hard > fpr01", ""),
    ("m10.TFIDF.bh.fpr01", "PO", "R1_operating_points > m10_fpr_budgets > detectors > TFIDF > benign_hard > fpr01", ""),
    ("m10.LR-C.all.fpr01", "PO", "R1_operating_points > m10_fpr_budgets > detectors > LR-C > all_label0 > fpr01", ""),
    ("m10.LR-C.all.fpr05", "PO", "R1_operating_points > m10_fpr_budgets > detectors > LR-C > all_label0 > fpr05", ""),
    ("m10.max_content_blind_0.1pct", "PO", "summary_derived > m10_max_precision_pi0.1pct_content_blind_R1", ""),
    ("m11.targets", "PO", "m11_label_free > R1_alert_rate_targets", ""),
    ("m11.medical.LR-C.b", "PO", "m11_label_free > folds > medical > detectors > LR-C > deployment_prev02 > summary_over_draws", "post hoc reading (b)"),
    ("m11.medical.LR-C.a", "PO", "m11_label_free > folds > medical > detectors > LR-C > primary_poolE > summary_over_draws", "pre-specified reading (a)"),
    ("m9.chance", "PO", "m9_chance_AP > values", ""),
    ("m9.rounding_max", "PO", "summary_derived > m9_max_abs_change_from_rounding", ""),
    # ---------------- M3 / M4 / m2 / m3 / m5 / m12 / m13 (domain_shift) ----------------
    ("M3.R1dm.LR-C.AUROC.web", "DS", "M3 > R1_domain_matched > web > LR-C > AUROC", ""),
    ("M3.R3.LR-C.AUROC.web", "DS", "M3 > R3 > web > LR-C > AUROC", ""),
    ("M3.R3mR1dm.LR-C.AUROC", "DS", "M3 > R3_minus_R1dm > LR-C > AUROC", ""),
    ("M3.R3mR1dm.LR-B.AUROC", "DS", "M3 > R3_minus_R1dm > LR-B > AUROC", ""),
    ("M3.degradation.LR-C.AUROC", "DS", "M3 > degradation_folds > LR-C > AUROC", ""),
    ("M3.R1dm_paired.LR-C-minus-LR-B.AUROC", "DS", "M3 > R1_domain_paired > LR-C-minus-LR-B > AUROC", ""),
    ("M3.R3_paired.LR-C-minus-LR-B.AUROC", "DS", "M3 > R3_paired > LR-C-minus-LR-B > AUROC", ""),
    ("M3.R1dm_paired.GRU-minus-LR-C.AUROC", "DS", "M3 > R1_domain_paired > GRU-minus-LR-C > AUROC", ""),
    ("M3.R3_paired.GRU-minus-LR-C.AUROC", "DS", "M3 > R3_paired > GRU-minus-LR-C > AUROC", ""),
    ("M3.R3_paired.LR-C-drop-minus-LR-B.AUROC", "DS", "M3 > R3_paired > LR-C-drop-minus-LR-B > AUROC", ""),
    ("M3.R3_paired.GRU-drop0.2-minus-LR-C.AUROC", "DS", "M3 > R3_paired > GRU-drop0.2-minus-LR-C > AUROC", ""),
    ("M3.reversal.LR-C-minus-LR-B.AUROC", "DS", "M3 > reversal > LR-C-minus-LR-B > AUROC > folds_established", ""),
    ("M3.reversal.LR-C-minus-LR-B.AUPRC_p10", "DS", "M3 > reversal > LR-C-minus-LR-B > AUPRC_p10 > folds_established", ""),
    ("M3.reversal.GRU-minus-LR-C.AUROC", "DS", "M3 > reversal > GRU-minus-LR-C > AUROC > folds_established", ""),
    ("M3.reversal.GRU-minus-LR-C.AUPRC_p10", "DS", "M3 > reversal > GRU-minus-LR-C > AUPRC_p10 > folds_established", ""),
    ("M3.reversal.LR-C-drop.survives", "DS", "M3 > reversal > LR-C-drop-minus-LR-B > AUROC > survives_remedy", ""),
    ("M3.reversal.GRU-drop0.2.survives", "DS", "M3 > reversal > GRU-drop0.2-minus-LR-C > AUROC > survives_remedy", ""),
    ("M4.n_fail_dm_holm", "DS", "derived > M4 > n_folds_failing_vs_dm_holm", ""),
    ("M4.n_fail_dm_unadj", "DS", "derived > M4 > n_folds_failing_vs_dm_unadjusted", ""),
    ("M4.LR-B.failing_dm_holm", "DS", "derived > M4 > LR-B@t_r80|failing_folds_vs_dm_holm", ""),
    ("M4.LR-B.failing_pooled_holm", "DS", "derived > M4 > LR-B@t_r80|failing_folds_vs_pooled_holm", ""),
    ("M4.GRU.every_seed_fails", "DS", "derived > M4 > GRU_seeds|every_seed_fails_every_fold_vs_dm_holm", ""),
    ("M4.LR-B.medical.recall", "DS", "M4 > deltas > LR-B@t_r80 > vs_domain_matched_R1 > recall > medical", ""),
    ("M4.LR-B.email.FPR", "DS", "M4 > deltas > LR-B@t_r80 > vs_domain_matched_R1 > FPR > email", ""),
    ("m5.LR-B.s1", "DS", "m5 > seeds > 1 > minus_R1 > LR-B > AUROC", ""),
    ("m5.overlap.LR-B.s1", "DS", "m5_overlap_sensitivity > seeds > 1 > pooled > LR-B > AUROC", ""),
    ("m5.detected_losses", "DS", "m5 > detected_losses", ""),
    ("m2.LR-C.AUROC", "DS", "m2 > estimates > LR-C > AUROC > R2_minus_R1", ""),
    ("m2.GRU.AUROC", "DS", "m2 > estimates > GRU > AUROC > R2_minus_R1", ""),
    ("m2.n_tasks", "DS", "m2 > R2_n_distinct_test_tasks_poolE", ""),
    ("m12.R1", "DS", "m12 > settings > R1 > all_rows", ""),
    ("m12.R2", "DS", "m12 > settings > R2 > all_rows", ""),
    ("m12.R1wd_s0", "DS", "m12 > settings > R1wd_s0 > all_rows", ""),
    ("m12.R3_email", "DS", "m12 > settings > R3_email > all_rows", ""),
    ("m12.R3_web", "DS", "m12 > settings > R3_web > all_rows", ""),
    ("m12.R4", "DS", "m12 > settings > R4 > all_rows", ""),
    ("m13.banking", "DS", "m13 > banking_caveat", ""),
    ("m3.total_shared", "DS", "m3 > total_shared_poolE", ""),
    # ---------------- M1 / Q1 (oov_mechanism) ----------------
    ("M1.offset.range", "OOV", "s1_lrc_offset > R3_range > LR-C", ""),
    ("M1.offset.drop.range", "OOV", "s1_lrc_offset > R3_range > LR-C-drop", ""),
    ("M1.graded.L1.LR-C", "OOV", "s2_graded > levels > 1.0 > detectors > LR-C > AUROC", ""),
    ("M1.graded.L0.LR-C", "OOV", "s2_graded > levels > 0.0 > detectors > LR-C > AUROC", ""),
    ("M1.graded.L0.LR-C.FPR", "OOV", "s2_graded > levels > 0.0 > detectors > LR-C > FPR@t_r80", ""),
    ("M1.graded.L0.GRU", "OOV", "s2_graded > levels > 0.0 > detectors > GRU > AUROC", ""),
    ("M1.graded.L0.GRU-drop0.2", "OOV", "s2_graded > levels > 0.0 > detectors > GRU-drop0.2 > AUROC", ""),
    ("M1.graded.L1.C-B", "OOV", "s2_graded > levels > 1.0 > contrasts > LR-C-minus-LR-B > AUROC", ""),
    ("M1.graded.L0.5.C-B", "OOV", "s2_graded > levels > 0.5 > contrasts > LR-C-minus-LR-B > AUROC", ""),
    ("M1.graded.L0.C-B", "OOV", "s2_graded > levels > 0.0 > contrasts > LR-C-minus-LR-B > AUROC", ""),
    ("M1.graded.L0.GRU-B", "OOV", "s2_graded > levels > 0.0 > contrasts > GRU-minus-LR-B > AUROC", ""),
    ("M1.graded.L0.GRUdrop-B", "OOV", "s2_graded > levels > 0.0 > contrasts > GRU-drop0.2-minus-LR-B > AUROC", ""),
    ("M1.graded.change.LR-C.AUROC", "OOV", "s2_graded > change_level0_minus_level1 > LR-C > AUROC", ""),
    ("M1.graded.change.LR-C.FPR", "OOV", "s2_graded > change_level0_minus_level1 > LR-C > FPR@t_r80", ""),
    ("M1.drop.magnitudes.C-B", "OOV", "s3_dropout > magnitudes > LR-C-minus-LR-B", ""),
    ("M1.drop.magnitudes.Cdrop-B", "OOV", "s3_dropout > magnitudes > LR-C-drop-minus-LR-B", ""),
    ("M1.GRUoffset.R3", "OOV", "s4_gru_offset > summary_R3 > GRU > offset_neg", ""),
    ("M1.GRUoffset.R3.n_above", "OOV", "s4_gru_offset > summary_R3 > GRU > n_cells_heldout_neg_above_train_neg", ""),
    ("M1.GRUoffset.R3.drop0.2", "OOV", "s4_gru_offset > summary_R3 > GRU-drop0.2 > offset_neg", ""),
    ("M1.GRUoffset.R1", "OOV", "s4_gru_offset > settings > R1 > GRU > mean_over_seeds", ""),
    ("M1.LOSO.coverage.banking", "OOV", "s5_loso > suites > banking > coverage", ""),
    ("M1.LOSO.CB.banking", "OOV", "s5_loso > suites > banking > contrasts > LR-C-minus-LR-B > LOSO_cluster", ""),
    ("M1.LOSO.CB.slack", "OOV", "s5_loso > suites > slack > contrasts > LR-C-minus-LR-B > LOSO_cluster", ""),
    ("M1.LOSO.CB.travel", "OOV", "s5_loso > suites > travel > contrasts > LR-C-minus-LR-B > LOSO_cluster", ""),
    ("M1.LOSO.CB.workspace", "OOV", "s5_loso > suites > workspace > contrasts > LR-C-minus-LR-B > LOSO_cluster", ""),
    ("M1.LOSO.WUT_pooled", "OOV", "s5_loso > WUT_pooled > contrasts", ""),
    ("M1.Rule4.R1_CB", "OOV", "s6_rule4 > R1_C_minus_B", ""),
    ("M1.Rule4.transfer_CB.strat", "OOV", "s6_rule4 > R4 > P0 > stratified > transfer_C_minus_B", ""),
    ("M1.Rule4.DD_R1.strat", "OOV", "s6_rule4 > R4 > P0 > stratified > DD_R1_ref", "pre-registered estimand"),
    ("M1.Rule4.DD_within.strat", "OOV", "s6_rule4 > R4 > P0 > stratified > DD_within_ref", "supplementary"),
    ("M1.Rule4.DD_within.cluster", "OOV", "s6_rule4 > R4 > P0 > cluster > DD_within_ref", "supplementary"),
    ("Q1.two_level", "OOV", "s7_q1 > reversal_two_level", ""),
    ("Q1.email.per_seed", "OOV", "s7_q1 > folds > R3_email > GRU > minus_LR-C_per_seed", ""),
    # ---------------- M2 (content_detector) ----------------
    ("M2.R1.TFIDF.AUROC", "CD", "settings > R1 > detectors > TFIDF > AUROC", ""),
    ("M2.R1.TFIDF.AP_w02", "CD", "settings > R1 > detectors > TFIDF > AP_w02", ""),
    ("M2.R3_banking.TFIDF.AUROC", "CD", "settings > R3_banking > detectors > TFIDF > AUROC", ""),
    ("M2.R3_web.TFIDF.AUROC", "CD", "settings > R3_web > detectors > TFIDF > AUROC", ""),
    ("M2.R3_banking.TFIDF.recall", "CD", "settings > R3_banking > detectors > TFIDF > recall_t_r80", ""),
    ("M2.R4.TFIDF.AUROC", "CD", "settings > R4 > detectors > TFIDF > AUROC", ""),
    ("M2.R4.TFIDF-LR-C", "CD", "settings > R4 > paired > TFIDF - LR-C > AUROC", "borderline, cluster p 0.057"),
    ("M2.WUT.TFIDF.AUROC", "CD", "settings > WUT > detectors > TFIDF > AUROC", ""),
    ("M2.WUT_F.TFIDF.AP_w02", "CD", "settings > WUT_F > detectors > TFIDF > AP_w02", ""),
    ("M2.markers", "CD", "separability > agentdojo_template_markers", ""),
    ("M2.toolname.R3_medical", "CD", "settings > R3_medical > detectors > TFIDF-toolname > AUROC", ""),
    ("M2.toolname.R3_coding", "CD", "settings > R3_coding > detectors > TFIDF-toolname > AUROC", ""),
    # ---------------- M5 / m8 (features_obslen) ----------------
    ("M5.coef.R1.LR-B.mean_obs", "FO", "M5_1_coefficients > R1_LR-B_bootstrap > mean_obs", ""),
    ("M5.coef.R1.LR-C.mean_obs", "FO", "M5_1_coefficients > R1_LR-C_bootstrap > mean_obs", ""),
    ("M5.uni.R1.mean_obs", "FO", "M5_1_univariate > populations > R1_poolE > metrics > mean_obs > AUROC", ""),
    ("M5.uni.P0.mean_obs", "FO", "M5_1_univariate > populations > AgentDojo_P0 > metrics > mean_obs > AUROC", ""),
    ("M5.noobs.CB.holm", "FO", "M5_2_without_observation_length > R3_holm > LR-C-noobs minus LR-B-noobs > AUROC", ""),
    ("M5.noobs.CB.holm.wAUPRC", "FO", "M5_2_without_observation_length > R3_holm > LR-C-noobs minus LR-B-noobs > wAUPRC@0.1", ""),
    ("M5.pad.LR-C.pad0.recall", "FO", "M5_3_padding > R1_linear > eval > metrics > LR-C|pad0 > recall@t_r80", ""),
    ("M5.pad.LR-C.pad100.recall", "FO", "M5_3_padding > R1_linear > eval > metrics > LR-C|pad100 > recall@t_r80", ""),
    ("M5.pad.LR-C.pad1200.recall", "FO", "M5_3_padding > R1_linear > eval > metrics > LR-C|pad1200 > recall@t_r80", ""),
    ("M5.pad.GRU.pad1200.recall", "FO", "M5_3_padding > R1_GRU > eval > metrics > GRU|pad1200 > recall@t_r80", ""),
    ("M5.pad.GRU.pad1200.AUROC", "FO", "M5_3_padding > R1_GRU > eval > metrics > GRU|pad1200 > AUROC", ""),
    ("M5.trunc.v03_cut", "FO", "M5_4_truncation > v03_cut", ""),
    ("m8.agentdrift", "FO", "m8_keyword_flag > agentdrift_hijacked_steps_pooled", ""),
    # ---------------- M6 / M7 / m1 / m2 / m7 / m15 / m16 / m17.3 / Q2 (r4_deep) ----------------
    ("M7.LR-C.v03", "R4D", "M7 > table > LR-C > v03", ""),
    ("M7.LR-C.corrected", "R4D", "M7 > table > LR-C > corrected", ""),
    ("M7.LR-B.corrected", "R4D", "M7 > table > LR-B > corrected", ""),
    ("M7.GRU.corrected", "R4D", "M7 > table > GRU > corrected", ""),
    ("M7.LR-A.corrected", "R4D", "M7 > table > LR-A > corrected", ""),
    ("M7.reconciliation", "R4D", "M7 > reconciliation_with_rerun_v03_scripts", ""),
    ("M6.case.LR-A", "R4D", "M6 > decisions > plan > LR-A > case_literal", ""),
    ("M6.case.LR-B", "R4D", "M6 > decisions > plan > LR-B > case_literal", ""),
    ("M6.case.LR-C", "R4D", "M6 > decisions > plan > LR-C > case_literal", ""),
    ("M6.case.GRU", "R4D", "M6 > decisions > plan > GRU > case_literal", ""),
    ("M6.case.TFIDF", "R4D", "M6 > decisions > plan > TFIDF > case_literal", ""),
    ("M6.case_v03.LR-C", "R4D", "M6 > decisions > v03 > LR-C > case_literal", ""),
    ("M6.P0_UT.within.LR-C", "R4D", "M6 > configs > P0_UT > cluster > ci > within:LR-C > AUROC", "ci list = bootstrap mean, lo, hi"),
    ("M6.P0_UT.within.LR-C.APw10", "R4D", "M6 > configs > P0_UT > point > within:LR-C > APw10", ""),
    ("M6.C1_UT.within.LR-C", "R4D", "M6 > configs > C1_UT > point > within:LR-C > AUROC", ""),
    ("M6.P0F_UT.transfer.LR-C", "R4D", "M6 > configs > P0F_UT > point > transfer:LR-C > AUROC", ""),
    ("M6.P0F_UT.transfer.GRU", "R4D", "M6 > configs > P0F_UT > point > transfer:GRU > AUROC", ""),
    ("M6.holm_IT", "R4D", "M6 > decisions > holm_within_minus_transfer_cluster > by_config > P0_IT", ""),
    ("m1.GRU_outcome1", "R4D", "m1 > GRU_outcome1 > R4_MB_native_transfer_strat_bootstrap_mean", ""),
    ("m1.GRU_outcome1.cluster", "R4D", "m1 > GRU_outcome1 > R4_MB_native_transfer_cluster_UT", ""),
    ("m1.LRA-LRC", "R4D", "m1 > paired > transfer:LR-A-minus-transfer:LR-C", ""),
    ("m2.LR-C.transfer.cluster_IT", "R4D", "m2 > cluster_IT > ci > transfer:LR-C > AUROC", ""),
    ("m7.dos_within", "R4D", "m7_Q4 > with_without_dos > P0noDoS_UT > point", ""),
    ("m7.exclusion", "R4D", "m7_Q4 > exclusion_by_class", ""),
    ("m16.outcome4", "R4D", "m16 > outcome4", ""),
    ("m15.AUROC", "R4D", "m15 > AUROC", ""),
    ("m6.WUT_GRU", "R4D", "m6 > WUT_GRU_without_unk", ""),
    ("Q2.range", "R4D", "Q2 > per_template_range_transfer", ""),
    ("M5.uni.R3.mean_obs.banking", "FO", "M5_1_univariate > populations > R3_banking_poolE > metrics > mean_obs > AUROC", ""),
    ("M5.R4.LR-B-noobs", "FO", "M5_2_without_observation_length > R4 > primary > metrics > LR-B-noobs > AUROC", ""),
    ("M5.R4.LR-C-noobs", "FO", "M5_2_without_observation_length > R4 > primary > metrics > LR-C-noobs > AUROC", ""),
    ("M5.WUT.pad.LR-C.pad1200.recall", "FO", "M5_3_padding > WUT > eval > primary > metrics > LR-C|pad1200 > recall@t_r80", ""),
    ("M5.cross.truncation", "FO", "cross_references > linear_cache_r4_truncation_check", ""),
    ("m8.agentdojo", "FO", "m8_keyword_flag > agentdojo_compromised_calls", ""),
    ("m8.annot.email", "FO", "m8_LRB_annotation_features > R3_email > primary > contrasts > LR-B-annot minus LR-B > AUROC", ""),
    ("m1.MB.LR-C", "R4D", "m1 > model_balanced_native_counts > transfer:LR-C", ""),
    ("m17_3.WUT.GRU", "R4D", "m17_3 > point > WUT:GRU", ""),
    ("m17_3.transfer.LR-C", "R4D", "m17_3 > point > transfer:LR-C", ""),
    ("m7.recall_dos", "R4D", "m7_Q4 > recall_dos_vs_non_dos", ""),
    ("Q2.II_only", "R4D", "Q2 > II_only > pooled", ""),
    ("M6.failed.R4.LR-C", "R4D", "M6 > failed_flag_rates > by_setting > R4 > LR-C", ""),
    ("M6.failed.WUT_F.GRU", "R4D", "M6 > failed_flag_rates > by_setting > WUT_F > GRU", ""),
    ("M6.WUTF_P0F.within.LR-C", "R4D", "M6 > configs > WUTF_P0F > point > within:LR-C > AUROC", ""),
    ("M6.P0_IT.within.LR-C", "R4D", "M6 > configs > P0_IT > point > within:LR-C > AUROC", ""),
    ("M6.P0_IT.within.GRU", "R4D", "M6 > configs > P0_IT > point > within:GRU > AUROC", ""),
    ("M6.MB_UT.within.LR-C", "R4D", "M6 > configs > MB_UT > point > within:LR-C > AUROC", ""),
    ("M6.MB_UT.within.GRU", "R4D", "M6 > configs > MB_UT > point > within:GRU > AUROC", ""),
    ("m2.cluster_IT.holm", "R4D", "m2 > cluster_IT > holm_WIT_minus_transfer", ""),
    ("DS.m13.folds.banking", "DS", "m13 > folds > banking", ""),
    ("DS.M3.R1dm.LR-C.AUROC", "DS", "M3 > R1_domain_matched", ""),
    ("DG.manuscript_comparison", "DG", "manuscript_comparison", ""),
    ("DG.m14.alignment", "DG", "m14 > alignment_test", ""),
    ("DG.m13.version", "DG", "m13_provenance > agentdrift > record_version_field_values_full_12536", ""),
    ("CD.R1wd.TFIDF", "CD", "R1wd_summary > TFIDF > AUROC", ""),
    ("CD.domain_matched.failure", "CD", "domain_matched > threshold_failure", ""),
    ("CD.WUT.failed", "CD", "settings > WUT > per_class_t_r80 > TFIDF > flag_failed_attack", ""),
    ("OOV.s3.threshold_counts", "OOV", "s3_dropout > threshold_change_counts", ""),
    ("OOV.s3.reversal", "OOV", "s3_dropout > reversal_rule_3_4", ""),
    # ---------------- m5 / m13 / m14 / m15 (diagnostics) ----------------
    ("DG.pool.n_worlds", "DG", "m5_worlds > pool_10656 > n_worlds", ""),
    ("DG.pool.purity", "DG", "m5_worlds > pool_10656 > purity_binary", ""),
    ("DG.full.n_worlds", "DG", "m5_worlds > full_12536 > n_worlds", ""),
    ("DG.key1029", "DG", "m5_key_search > rows_n_worlds_1029", ""),
    ("DG.reuse.R1", "DG", "m5_world_reuse > R1 > all_test > share_world_in_train", ""),
    ("DG.reuse.R1.auroc", "DG", "m5_world_reuse > R1 > all_test > auroc_train_world_attack_rate", ""),
    ("DG.probe.world", "DG", "m15_probes > world > paired_vs_floors", ""),
    ("DG.probe.domain", "DG", "m15_probes > domain > units", ""),
]


def compare_rerun_with_cache():
    """Compare the re-run v0.3 R4 outputs with values computed directly from the caches."""
    out = {}
    log = open(os.path.join(HERE, "rerun_v03_scripts.log"), encoding="utf-8").read()
    out["rerun_final"] = "gru_ladder exit 0" in log
    from evalkit import auroc
    # The pickles are the project's own revision caches (written by build_linear_cache.py / build_gru_cache.py), not external data.
    lin = pickle.load(open(os.path.join(HERE, "cache", "linear", "R4.pkl"), "rb"))
    wut = pickle.load(open(os.path.join(HERE, "cache", "linear", "WUT.pkl"), "rb"))
    gru = pickle.load(open(os.path.join(HERE, "cache", "gru", "R4.pkl"), "rb"))
    gwut = pickle.load(open(os.path.join(HERE, "cache", "gru", "WUT.pkl"), "rb"))
    m = np.asarray(lin["pool_E"], bool)
    y = np.asarray(lin["y"])[m]
    cache = {}
    for d in ("LR-A", "LR-B", "LR-C"):
        cache[f"transfer_{d}"] = float(auroc(y, np.asarray(lin["scores"][d])[m]))
        s = np.asarray(wut["scores"][d])
        cache[f"withinUT_{d}"] = float(np.mean([auroc(y, s[r][np.asarray(wut["pool_E"], bool)]) for r in range(s.shape[0])]))
    gs = np.asarray(gru["scores"]["GRU"])
    cache["transfer_GRU"] = float(np.mean([auroc(y, gs[r][m]) for r in range(gs.shape[0])]))
    gw = np.asarray(gwut["scores"]["GRU"])
    mw = np.asarray(gwut["pool_E"], bool)
    cache["withinUT_GRU"] = float(np.mean([auroc(np.asarray(gwut["y"])[mw], gw[r][mw]) for r in range(gw.shape[0])]))
    abl = json.load(open(os.path.join(ROOT, "ablation_results.json")))
    r4c = json.load(open(os.path.join(ROOT, "r4_control_results.json")))
    grr = json.load(open(os.path.join(ROOT, "gru_results.json")))
    rer = {}
    for rep, d in (("A", "LR-A"), ("B", "LR-B"), ("C", "LR-C")):
        mt = abl["R4"]["P0"]["by_prev"]["native"]["metrics"]
        rer[f"transfer_{d}"] = mt[f"transfer_{rep}"]["AUROC"][0]
        rer[f"withinUT_{d}"] = mt[f"within_{rep}"]["AUROC"][0]
    gm = grr["R4"]["P0"]["by_prev"]["native"]["metrics"]
    rer["transfer_GRU"] = gm["GRU_transfer"]["AUROC"][0]
    rer["withinUT_GRU"] = gm["GRU_within"]["AUROC"][0]
    # r4_control: find LR-C (full) and LR-B (agnostic) within-UT and transfer on P0 native
    r4 = {k: v["AUROC"][0] for k, v in r4c["P0"]["native"]["metrics"].items()}
    cmp = {"transfer_full": "transfer_LR-C", "transfer_agn": "transfer_LR-B",
           "within_UT_full": "withinUT_LR-C", "within_UT_agn": "withinUT_LR-B"}
    out["max_abs_diff_cache_vs_r4_control"] = round(max(abs(r4[k] - cache[v]) for k, v in cmp.items()), 5)
    arch = json.load(open(os.path.join(ROOT, "archive_v0.3", "r4_control_results.json")))
    out["archive_v03_r4_control_P0_native"] = {k: v["AUROC"][0] for k, v in arch["P0"]["native"]["metrics"].items()}
    out["cache_points"] = {k: round(v, 4) for k, v in cache.items()}
    out["rerun_points_ablation_gru"] = rer
    out["rerun_points_r4_control_P0_native"] = r4
    out["max_abs_diff_cache_vs_ablation_gru"] = round(max(abs(cache[k] - rer[k]) for k in rer), 5)
    return out


def main():
    vals, errors = {}, []
    for vid, tag, path, note in SPEC:
        try:
            v = get(tag, path)
            vals[vid] = {"file": "revision/" + FILES[tag], "key": path, "value": v, "formatted": fmt(v), "note": note}
        except Exception as e:
            errors.append((vid, tag, path, repr(e)))
    res = {
        "meta": {
            "generated": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "script": "revision/consolidate_results.py",
            "script_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
            "sources_sha256": {FILES[t]: hashlib.sha256(open(os.path.join(HERE, FILES[t]), "rb").read()).hexdigest() for t in FILES},
            "key_separator": " > ",
            "formatted": "point [lo, hi] rounded to 3 decimals; for r4_deep cluster 'ci' lists the first element is the bootstrap mean",
        },
        "values": vals,
        "errors": errors,
        "rerun_v03_vs_cache": compare_rerun_with_cache(),
    }
    with open(os.path.join(HERE, "RESULTS_SUMMARY_values.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=float)
    for vid in vals:
        print(f"{vid:45s} {vals[vid]['formatted'][:160]}")
    print("ERRORS:", len(errors))
    for e in errors:
        print("  ", e)
    print(json.dumps(res["rerun_v03_vs_cache"], indent=1))


if __name__ == "__main__":
    main()
