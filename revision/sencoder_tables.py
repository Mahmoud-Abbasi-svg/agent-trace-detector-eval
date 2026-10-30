"""Markdown tables from revision/sencoder_results.json (no computation beyond formatting). Returns a dict of tables."""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
DOM = ["medical", "banking", "web", "email", "coding"]


def F(c, d=3):
    if c is None:
        return "n/a"
    if isinstance(c, dict):
        c = [c.get("point", c.get("delta")), c.get("lo"), c.get("hi")]
    if c[1] is None:
        return f"{c[0]:.{d}f}"
    return f"{c[0]:.{d}f} [{c[1]:.{d}f}, {c[2]:.{d}f}]"


def S(c, d=3):
    if isinstance(c, dict):
        c = [c.get("point", c.get("delta")), c.get("lo"), c.get("hi")]
    f = lambda x: f"{x:+.{d}f}".replace("-", "\u2212")
    return f"{f(c[0])} [{f(c[1])}, {f(c[2])}]"


def tables(R):
    T = {}
    L = R["ladder"]
    rows = ["| Rung | κ | SE | GRU (10 seeds) | LR-C | LR-B | SE − LR-B, paired | SE seed range |", "|---|---|---|---|---|---|---|---|"]
    names = [("R1", "R1")] + [(f"R1wd_s{k}", f"R1-wd split {k}") for k in range(5)] + [("R2", "R2")] + [(f"R3_{d}", f"R3 {d}") for d in DOM]
    for s, lab in names:
        m = L[s]["by_prev"]["native"]["metrics"]; p = L[s]["by_prev"]["native"]["paired"]["SE-minus-LR-B"]["AUROC"]
        rows.append(f"| {lab} | {L[s]['kappa']:.3f} | {F(m['SE']['AUROC'])} | {F(m['GRU10']['AUROC'])} | {F(m['LR-C']['AUROC'])} | "
                    f"{F(m['LR-B']['AUROC'])} | {S(p)} | {L[s]['seed_range_AUROC']['SE']:.3f} |")
    T["ladder_auroc"] = "\n".join(rows)
    rows = ["| Rung | AUPRC native | AUPRC at 2% | AUPRC at 10% | recall at θr80 | FPR at θr80 | recall at 0.5 | FPR at 0.5 |", "|---|---|---|---|---|---|---|---|"]
    for s, lab in [("R1", "R1"), ("R2", "R2")] + [(f"R3_{d}", f"R3 {d}") for d in DOM]:
        bp = L[s]["by_prev"]; m = bp["native"]["metrics"]["SE"]
        rows.append(f"| {lab} | {F(m['AUPRC'])} | {F(bp['p02']['metrics']['SE']['AUPRC'])} | {F(bp['p10']['metrics']['SE']['AUPRC'])} | "
                    f"{F(m['recall@t_r80'])} | {F(m['FPR@t_r80'])} | {F(m['recall@t05'])} | {F(m['FPR@t05'])} |")
    T["se_operating"] = "\n".join(rows)
    V = R["reversal"]; rv = V["reversal"]["SE-minus-LR-B"]
    rows = ["| Held-out domain | R1dm SE − LR-B (AUROC) | R3 SE − LR-B (AUROC) | R3 SE − LR-B (AUPRC at 10%) | reversal established (AUROC / AUPRC10) | SE R3 − R1dm (AUROC) |", "|---|---|---|---|---|---|"]
    for d in DOM:
        a = rv["AUROC"]["per_fold"][d]; b = rv["AUPRC_p10"]["per_fold"][d]; g = V["R3_minus_R1dm"]["SE"]["AUROC"][d]
        rows.append(f"| {d} | {S(a['R1dm'])} | {S(a['R3'])}{'' if a['R3']['detected_holm'] else ' (not detected)'} | {S(b['R3'])} | "
                    f"{'yes' if a['established'] else 'no'} / {'yes' if b['established'] else 'no'} | {S(g)} |")
    T["reversal"] = "\n".join(rows)
    O = R["offset_analogue"]; G = R.get("post_hoc_not_preregistered", {}).get("GRU10_full_logit_offset", {}).get("values", {})
    rows = ["| Held-out domain | SE offset, mean over 10 seeds | SE range over seeds | SE, benign and hard-negative training negatives only | GRU, same estimator (post hoc) | embedding drift, mean cosine distance |", "|---|---|---|---|---|---|"]
    dr = R["embedding_drift"]["rungs"]
    for d in DOM:
        o = O["R3"][d]; g = G.get(f"R3_{d}", {})
        rows.append(f"| {d} | {o['mean']:+.3f} | {o['min']:+.3f} to {o['max']:+.3f} | {o['mean_benign_hardneg_train']:+.3f} | "
                    f"{g.get('mean', float('nan')):+.3f} | {dr[f'R3_{d}']['pool_E']['mean']:.3f} |")
    o = O["R1_reference"]; g = G.get("R1", {})
    rows.append(f"| R1 (reference) | {o['mean']:+.3f} | {o['min']:+.3f} to {o['max']:+.3f} | {o['mean_benign_hardneg_train']:+.3f} | {g.get('mean', float('nan')):+.3f} | {dr['R1']['pool_E']['mean']:.3f} |")
    T["offset"] = "\n".join(rows).replace("-", "-")
    R4 = R["R4"]
    rows = ["| Population | n pos / n neg | SE transfer | SE within (UT) | within − transfer | SE-W0 transfer | SE-W0 within | GRU transfer | GRU within | LR-C transfer |", "|---|---|---|---|---|---|---|---|---|---|"]
    for pop, lab in [("P0", "P0"), ("C1", "C1"), ("P0F", "P0F")]:
        m = R4[pop]["by_prev"]["native"]["metrics"]; p = R4[pop]["by_prev"]["native"]["paired"]
        rows.append(f"| {lab} | {R4[pop]['n_pos']:,} / {R4[pop]['n_neg']:,} | {F(m['SE_transfer']['AUROC'])} | {F(m['SE_within']['AUROC'])} | "
                    f"{S(p['SE_within-minus-SE_transfer']['AUROC'])} | {F(m['SE-W0_transfer']['AUROC'])} | {F(m['SE-W0_within']['AUROC'])} | "
                    f"{F(m['GRU10_transfer']['AUROC'])} | {F(m['GRU10_within']['AUROC'])} | {F(m['LR-C_transfer']['AUROC'])} |")
    mb = R4["model_balanced"]; m = mb["native_counts"]["metrics"]; p = mb["native_counts"]["paired"]
    rows.append(f"| model-balanced, native counts | {mb['n_pos']:,} / {mb['n_neg']:,} | {F(m['SE_transfer']['AUROC'])} | {F(m['SE_within']['AUROC'])} | "
                f"{S(p['SE_within-minus-SE_transfer']['AUROC'])} | {F(m['SE-W0_transfer']['AUROC'])} | {F(m['SE-W0_within']['AUROC'])} | "
                f"{F(m['GRU10_transfer']['AUROC'])} | {F(m['GRU10_within']['AUROC'])} | {F(m['LR-C_transfer']['AUROC'])} |")
    T["r4"] = "\n".join(rows)
    p = R4["P0"]["by_prev"]["native"]["paired"]
    T["r4_w0"] = "\n".join(["| Contrast (P0, AUROC, paired) | Difference |", "|---|---|",
                            f"| SE transfer − SE-W0 transfer | {S(p['SE_transfer-minus-SE-W0_transfer']['AUROC'])} |",
                            f"| SE within − SE-W0 within | {S(p['SE_within-minus-SE-W0_within']['AUROC'])} |",
                            f"| SE transfer − LR-B transfer | {S(p['SE_transfer-minus-LR-B_transfer']['AUROC'])} |",
                            f"| SE transfer − GRU transfer | {S(p['SE_transfer-minus-GRU10_transfer']['AUROC'])} |"])
    cl = R4["P0"]["cluster_UT_AUROC"]
    T["r4_cluster"] = "\n".join(["| Method (P0) | AUROC, suite and user-task cluster bootstrap |", "|---|---|"] +
                                [f"| {k.replace('GRU10', 'GRU')} | {F(v)} |" for k, v in cl.items()])
    G2 = R["graded_and_padding"]["graded"]
    rows = ["| Level | Achieved coverage | SE AUROC | SE FPR at θr80 | SE recall at θr80 | SE − LR-B | LR-C (published) | GRU (published) | LR-B |", "|---|---|---|---|---|---|---|---|---|"]
    for lv, v in G2["levels"].items():
        pub = v["published_same_masks"]
        rows.append(f"| {lv} | {v['coverage_mean']:.3f} | {F(v['AUROC'])} | {F(v['FPR@t_r80'])} | {F(v['recall@t_r80'])} | {S(v['AUROC_minus_LR-B'])} | "
                    f"{pub['LR-C']['point']:.3f} | {pub['GRU']['point']:.3f} | {pub['LR-B']['point']:.3f} |")
    T["graded"] = "\n".join(rows)
    P = R["graded_and_padding"]["padding"]; pub = P["published_same_protocol"]
    rows = ["| Filler characters | SE recall at θr80 | SE AUROC | LR-C recall (published) | LR-B recall (published) | GRU recall (published) |", "|---|---|---|---|---|---|"]
    for k, n in (("pad0", "0"), ("pad100", "100"), ("pad1200", "1,200")):
        v = P["levels"][k]
        rows.append(f"| {n} | {F(v['recall@t_r80'])} | {F(v['AUROC'])} | {F(pub['LR-C'][k]['recall@t_r80'])} | {F(pub['LR-B'][k]['recall@t_r80'])} | {F(pub['GRU'][k]['recall@t_r80'])} |")
    T["padding"] = "\n".join(rows)
    rows = ["| Rung | Mean cosine distance to nearest training step | Median | 90th percentile | Share of steps with an exact match |", "|---|---|---|---|---|"]
    for s, lab in [("R1", "R1"), ("R1wd_s0", "R1-wd split 0"), ("R2", "R2")] + [(f"R3_{d}", f"R3 {d}") for d in DOM] + [("R4", "R4, P0")]:
        e = dr[s]["pool_E"]
        rows.append(f"| {lab} | {e['mean']:.3f} | {e['median']:.3f} | {e['p90']:.3f} | {e['frac_exact_match_le_1e-6']:.3f} |")
    e = dr["R4"]["failed_attacks"]
    rows.append(f"| R4, unsuccessful attacks | {e['mean']:.3f} | {e['median']:.3f} | {e['p90']:.3f} | {e['frac_exact_match_le_1e-6']:.3f} |")
    T["drift"] = "\n".join(rows)
    P2 = R["R2_paper_comparison_point"]; pubr = P2["published_reference_not_ours"]
    rows = ["| Quantity at threshold 0.5 | Our reimplementation, mean over 10 seeds (min to max) | DriftNet as published, single run (not ours) |", "|---|---|---|"]
    for k in ("precision", "recall", "F1"):
        rows.append(f"| {k} | {P2[k]['mean']:.3f} ({P2[k]['min']:.3f} to {P2[k]['max']:.3f}) | {pubr[k]:.3f} |")
    for c in ("benign", "failed_attack", "hard_negative"):
        rows.append(f"| flag rate, {c.replace('_', ' ')} | {P2['flag_rate'][c]['mean']:.3f} ({P2['flag_rate'][c]['min']:.3f} to {P2['flag_rate'][c]['max']:.3f}) | {pubr['flag_rate'][c]:.3f} |")
    T["r2_paper"] = "\n".join(rows)
    sup = R4["agentdojo_feature_support_by_suite"]
    rows = ["| Suite | Traces (P0 + unsuccessful) | Any address in arguments | Any URL in arguments | Any non-zero world feature |", "|---|---|---|---|---|"]
    for su in ("banking", "slack", "travel", "workspace"):
        a = sup["all"][su]
        rows.append(f"| {su} | {a['n_traces']:,} | {a['share_any_email_in_args']:.3f} | {a['share_any_url_in_args']:.3f} | {a['share_any_nonzero_world_feature']:.3f} |")
    T["support"] = "\n".join(rows)
    g10 = R["gru10"]
    rows = ["| Setting | GRU AUROC, 3 seeds (paper) | GRU AUROC, 10 seeds | per-seed range, 10 seeds | seeds 0 to 2 reproduce cache |", "|---|---|---|---|---|"]
    for s, v in g10.items():
        rows.append(f"| {s.replace('_', ' ')} | {v['AUROC_poolE_mean_3seed']:.3f} | {v['AUROC_poolE_mean_10seed']:.3f} | {min(v['AUROC_poolE_per_seed_10']):.3f} to {max(v['AUROC_poolE_per_seed_10']):.3f} | "
                    f"{'yes' if v['seeds0to2_t_r80_pred_mismatches'] == 0 and v['seeds0to2_max_abs_score_diff_vs_3seed_cache'] == 0 else 'no'} |")
    T["gru10"] = "\n".join(rows)
    pr = R4["feature_block_offset_probe"]
    T["probe"] = "\n".join(["| Feature-block offset probe (P0, no refit, not SE-W0) | Mean over 10 seeds |", "|---|---|",
                            f"| logit shift, AgentDojo negatives | {pr['shift_P0_negatives']:+.3f} |",
                            f"| logit shift, AgentDojo positives | {pr['shift_P0_positives']:+.3f} |",
                            f"| logit shift, AgentDrift training negatives | {pr['shift_AgentDrift_train_negatives']:+.3f} |",
                            f"| AUROC with features / zeroed | {pr['AUROC_P0_with']:.3f} / {pr['AUROC_P0_zeroed']:.3f} |"]).replace("+-", "\u2212")
    return T


if __name__ == "__main__":
    R = json.load(open(os.path.join(HERE, "sencoder_results.json"), encoding="utf-8"))
    for k, v in tables(R).items():
        print(f"### {k}\n{v}\n")
