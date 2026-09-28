# Revision results summary (manuscript v0.3 → v0.4)

Consolidated on 27 September 2026 from the seven verified analysis streams. This file is for the authors revising `MANUSCRIPT_DRAFT.md`. It is organised by review comment (Part 1), gives a claims decision table for the v0.3 manuscript (Part 2), and lists open items (Part 3). Deviations from all plans are in `revision/DEVIATIONS.md`.

## How to read this file

**Sources.** Each number is followed by its source as `TAG:key`, where the key is the path inside that stream's results file, written as in the stream summary (dots or slashes separate levels; `<d>` is a fold or domain). Where a key is long, the stream summary section is named instead.

| Tag | Results file | Summary file | Review comments covered |
|---|---|---|---|
| PO | `revision/prevalence_ops_results.json` | `prevalence_ops_summary.md` | M8, M9, m3 (prevalence), m4, m9, m10, m11, m17.5 |
| DS | `revision/domain_shift_results.json` | `domain_shift_summary.md` | M3, M4, m2 (R2), m3 (R3), m5, m12, m13 |
| OOV | `revision/oov_mechanism_results.json` | `oov_mechanism_summary.md` | M1, Q1, m6 (`<UNK>` norm) |
| CD | `revision/content_detector_results.json` | `content_detector_summary.md` | M2, m13 (templates) |
| FO | `revision/features_obslen_results.json` | `features_obslen_summary.md` | M5, m8 |
| R4D | `revision/r4_deep_results.json` | `r4_deep_summary.md` | M6, M7, m1, m2 (AgentDojo), m6, m7, m15 (model), m16, m17.3, Q2, Q4 |
| DG | `revision/diagnostics_results.json` | `diagnostics_summary.md`, `REPRODUCIBILITY.md` | m5 (worlds), m13 (provenance), m14, m15 (world/domain) |
| RW | `revision/related_work_v04.md`, `references_v04.json` | same | M12, m19, m20, Q3 |
| RR | `agent-ids/r4_control_results.json`, `ablation_results.json`, `gru_results.json` (v0.3 scripts re-run on the corrected extraction; final, `rerun_v03_scripts.log` contains "gru_ladder exit 0") | n/a | M7 |

The values quoted most often are also collected, with their file and key, in `revision/RESULTS_SUMMARY_values.json` (written by `revision/consolidate_results.py`, which reads the stream files and fits nothing).

**Conventions.**
- Values are point [95% bootstrap percentile interval], with B = 2,000.
- AgentDrift intervals resample trajectories (stratified by class). AgentDojo intervals resample suite|user_task clusters (117 clusters) unless the text says "strat" (the v0.3 trajectory-level bootstrap). R2 intervals resample the 35 test tasks.
- "Detected" means the interval excludes 0 (0.5 for a single AUROC). "Holm" means still detected after Holm correction within the plan 3.6 family (five R3 folds, the comparisons at one prevalence, or the representations of one R4 configuration).
- wAUPRC@π is the prevalence-weighted average precision of plan 3.1. It is the primary AUPRC estimand everywhere in the revision.
- θr80 is set on training positives and θval80 on validation positives. No threshold uses test labels. The one exception is "FPR at a fixed test TPR of 0.8", which is descriptive and is labelled wherever it appears.
- "R1dm" is the R1 model scored on the R1 test pool of one domain (plan 3.2), the comparator for R3.
- "Pool E" is benign + hard negatives + attacked for AgentDrift, and P0 for AgentDojo. Unsuccessful attacks are scored separately.
- 4 decimals are kept where a value sits on a decision boundary.
- In `r4_deep` tables, paired-delta centres are bootstrap means. They can differ from the point difference in the third decimal (for example, LR-C WUT minus transfer: point 0.797 − 0.473 = 0.3237, cluster centre +0.327).

**Status of the evidence.** Every stream was verified independently (own code, own resamples), and every stream was approved. The verifiers raised four medium-severity issues, all of which are corrected in this file:
- the Rule-4 estimand (M1);
- the world-key lead (m5);
- the borderline R4 TF-IDF contrast (M2);
- the scope of the M9 breakdown (logged as deviation P12).

The low-severity wording fixes are also applied here. As plan Section 0 states, these analyses are confirmatory re-runs of analyses whose approximate outcome the review panel had already seen. The paper must say so.

---

# Part 1. Results by review comment

## Major comments

### M1. R3 reversals and out-of-vocabulary (OOV) encoding [analysis done + text]

**What was done.**
- Measured the LR-C tool-block offset w_A^T c per fold.
- Graded tool coverage inside R1: 5 levels, 5 random masks per intermediate level.
- Tool-block-dropout LR-C (LR-C-drop).
- Token-dropout GRU: p = 0.2 primary and p = 0.1 sensitivity, with 3 seeds, intervals, paired contrasts and the plan 3.4 rule.
- The GRU tool-pathway offset and `<UNK>` norm over all 15 R3 fold × seed cells.
- AgentDojo leave-one-suite-out (LOSO).
- The pre-registered Rule 4 test in R4.

**Key numbers.**
1. **Offset (action 1).** LR-C adds +1.653 to +1.899 logits to every held-out R3 trajectory. Only 10.7% to 15.0% of training trajectories reach a tool contribution that large. LR-C-drop reduces the offset to −0.057 to −0.021. [OOV:s1_lrc_offset.R3_range.LR-C, .LR-C-drop] The offset is constant within a fold. It explains the threshold behaviour (recall and FPR rise together), not the ranking loss.
2. **Graded coverage inside R1 (action 2).** Removing tool identity at test time, with no domain change, reproduces both reversals.
   - LR-C AUROC falls from 0.879 [0.863, 0.895] at full coverage to 0.787 [0.765, 0.808] at zero coverage (change −0.091 [−0.110, −0.074]).
   - LR-C FPR at θr80 rises by +0.466 [+0.433, +0.497], to 0.657 [0.627, 0.687].
   - LR-C − LR-B moves from +0.049 [+0.034, +0.064] (full coverage) to −0.005 [−0.016, +0.006] (coverage about 0.5) to −0.043 [−0.055, −0.031] (zero).
   - At zero coverage the GRU is at 0.674 [0.650, 0.699], and GRU − LR-B is −0.156 [−0.179, −0.131].

   [OOV:s2_graded.levels.<L>.detectors / .contrasts; s2_graded.change_level0_minus_level1.LR-C] The GRU intervals in this section are seed-averaged (conditional on the 3 seeds), so they are narrower than the mixture intervals of M1 action 3.
3. **Dropout arms and the plan 3.4 rule (action 3).** AUROC, Holm over five folds.
   - The C < B reversal is established in web, email and coding [DS:M3.reversal.LR-C-minus-LR-B.AUROC.folds_established].
   - The GRU < C reversal is established in medical, banking, email and coding [DS:M3.reversal.GRU-minus-LR-C.AUROC.folds_established].
   - With tool-block dropout, LR-C-drop − LR-B still meets the rule in web, email and coding, but only as a near tie: R3 −0.015 to −0.001, against −0.063 to −0.026 for LR-C [OOV:s3_dropout.magnitudes]. It is detected only because the two scores are almost rank-identical (Spearman ≥ 0.997).
   - With token dropout at p = 0.2 (pre-specified), GRU-drop0.2 − LR-C is −0.003 to +0.026 and meets the rule in **no** fold. It is detected *above* LR-C in banking, +0.019 [+0.004, +0.033] [OOV:s3_dropout.reversal_rule_3_4; DS:M3.reversal.GRU-drop0.2-minus-LR-C.AUROC.survives_remedy].
   - At p = 0.1 the GRU reversal survives in medical only.
   - Dropout changes the direction of the threshold failure rather than removing it. GRU-drop0.2 recall at θr80 falls in 5/5 folds (R3 recall 0.174 to 0.397), and its FPR changes in 0/5 [OOV:s3_dropout.threshold_change_counts].
4. **GRU mechanism (action 4, m6).**
   - The tool pathway gives held-out negatives a mean logit +6.291 higher than training negatives (range −0.782 to +8.617), positive in 14/15 cells. The exception is banking seed 1, which flags almost nothing. In R1 the same offset is +0.532 [+0.387, +0.678] [OOV:s4_gru_offset.summary_R3.GRU.offset_neg; .settings.R1.GRU].
   - The `<UNK>` row receives no data gradient in AgentDrift. Adam's coupled weight decay alone shrinks it from 5.597–6.145 to 0.113–2.332 (mean 1.228), against median trained tool-embedding norms of 4.284–4.824.
   - With GRU-drop0.2 the `<UNK>` norm is 4.887–5.630 and the offset falls to +2.179 (range +1.633 to +2.891) [OOV:s4_gru_offset.unk.summary_R3; .summary_R3.GRU-drop0.2].
   - Training negatives here include failed attacks (deviation O8). The benign + hard-negative-only mean is stored per setting as `train_negpoolEcats_mean`.
5. **AgentDojo LOSO (action 5a).**
   - Coverage: banking 0.000, slack 0.000, travel 0.084, workspace 0.362.
   - LR-C − LR-B on the held-out suite (cluster): banking −0.059 [−0.172, +0.034]; slack +0.044 [+0.016, +0.076] (Holm); travel +0.048 [+0.011, +0.084] (Holm); workspace −0.001 [−0.019, +0.019] [OOV:s5_loso.suites.<s>.contrasts.LR-C-minus-LR-B.LOSO_cluster].
   - In distribution (pooled WUT), C − B is +0.056 [−0.003, +0.111] (cluster), which is not detected [OOV:s5_loso.WUT_pooled.contrasts].
   - No suite meets the reversal rule, and at zero coverage the sign differs between banking and slack.
   - In the two zero-coverage suites the LR-C offset is negative (banking −0.358, slack −2.217), so LR-C raises fewer alerts than LR-B there. The claim is limited to banking and slack, where the offset applies to every row.
6. **Rule 4, R4 clause (action 5b).**
   - The pre-registered estimand (PRE_ABLATION_DESIGN.md lines 73 and 79: degradation from R1 to the shift, independent bootstrap) is the R1-referenced degradation difference: **+0.057 [+0.042, +0.073]** (stratified, as pre-registered) [OOV:s6_rule4.R4.P0.stratified.DD_R1_ref].
   - Precondition: R1 C − B is +0.049 [+0.034, +0.064].
   - Transfer C − B is −0.008 [−0.011, −0.006].
   - So the R4 clause is met (Outcome 4). The test is weak, because both detectors are near chance in transfer (LR-C 0.473, LR-B 0.482).
   - Supplementary, within-AgentDojo-referenced: +0.064 [+0.022, +0.101] (strat) and +0.064 [+0.005, +0.120] (cluster). This variant mostly restates the within-AgentDojo C − B gap. The review's +0.056 was also R1-referenced; the corrected value is +0.057. The OOV summary mislabels the within-referenced value as pre-registered (corrected in DEVIATIONS.md, X4).

**Change to the paper.**
- Remove "the in-distribution ranking reverses in two detector families" from the Abstract, contributions and Conclusion.
- Replace it with: "Under AgentDrift's disjoint tool vocabularies, LR-C ranks below LR-B in all five held-out domains and the GRU below LR-C in all five. By the pre-specified reversal rule (an in-distribution advantage detected against a domain-matched comparator, then reversed), these are reversals in three and four of five domains. Both are reproduced inside R1 by removing tool coverage, so they follow from untrained out-of-vocabulary inputs. Token dropout removes the GRU reversal, and tool-block dropout shrinks the LR-C reversal to a near tie."
- Report Eq. (10) with the measured offset and delete "we did not measure it" (line 308).
- Report LOSO as a failed real-trace replication.
- Report Rule 4 with the R1-referenced estimand.

### M2. Content-reading baseline [analysis done (TF-IDF only) + text]

**What was done.** TF-IDF logistic regression through every rung, as configured in plan Section 2, with these variants:
- nodigit;
- world mask and README-style anonymisation;
- tool names only;
- template and separability analysis.

A frozen sentence encoder and a released component (PromptGuard 2, AlignmentCheck) were **not** run, so M2 action 1 is only partly met.

**Key numbers.**
1. **In distribution.**
   - R1 AUROC 0.999 [0.999, 1.000] and wAUPRC@2% 0.984 [0.973, 0.994], with FPR 0 at θr80.
   - The exact 95% upper bound on the FPR is 0.0045, so precision at 2% is at least 0.777 [CD:zero_fp_bounds.R1.TFIDF].
   - TFIDF − LR-C is detected in all seven in-distribution settings (R1 +0.121 [+0.105, +0.136]).
   - R1-wd mean AUROC is 0.9997 [CD:R1wd_summary.TFIDF.AUROC].
2. **R3.**
   - AUROC 0.980 to 0.996, with FPR exactly 0 in every fold.
   - TFIDF − LR-C +0.118 to +0.251, detected after Holm in 5/5 folds.
   - No reversal by the plan 3.4 rule (0 folds) [CD:domain_matched.reversal_check_AUROC].
   - Its threshold still fails by plan 3.3 in 5/5 folds (Holm), but in the recall direction. R3 recall at θr80 is 0.272–0.696 against R1dm 0.685–0.817; banking falls −0.413 [−0.508, −0.323] [CD:domain_matched.threshold_failure; domain_matched.domains.<d>.R3_minus_R1dm.TFIDF.recall_t_r80].
   - TFIDF recalls 0.733–0.988 of full hijacks but only 0.038–0.345 of partial and 0.006–0.225 of delayed ones [CD:settings.R3_<d>.per_class_t_r80.TFIDF].
3. **R4.**
   - Transfer AUROC 0.520 [0.471, 0.564]. It flags nothing at θr80.
   - TFIDF − LR-C is +0.047 [−0.002, +0.096], p = 0.057 under cluster resampling. This is **borderline**: it is detected under trajectory resampling ([+0.019, +0.075], p = 0.002), and it is Monte Carlo sensitive [CD:settings.R4.paired."TFIDF - LR-C".AUROC; .sensitivity.trajectory_stratified].
   - Within AgentDojo it is near perfect: WUT 0.996 [0.991, 0.999], WIT 0.987 [0.972, 0.996], LOSO 0.989–0.998.
   - This is largely injection presence. Five fixed attack strings occur in 0.996 of attacked P0, 0.947 of unsuccessful attacks and 0.000 of benign runs, and used as a fixed rule they give AUROC 0.998 [CD:separability.agentdojo_template_markers].
   - With unsuccessful attacks as training negatives (WUT_F), AUROC is 0.941 [0.911, 0.966] and wAUPRC@2% 0.349 [0.196, 0.785].
   - TFIDF still ranks compromised traces above unsuccessful attacks, with AUROC 0.808 [0.796, 0.820] [PO:M9.WUT…TFIDF.AUROC_subgroups.poolE_vs_failed].
4. **Separability (external validity).**
   - In-sample R1 AUROC 0.99997. Every category carries its own generator template.
   - "2026" (coefficient −9.68) occurs in 0.742 of benign and 0.848 of failed-attack training trajectories, against 0.000 of hard negatives and 0.046 of attacked ones.
   - The hard-negative template ("invoice", "5000", "to audit") and the failed-attack template ("10 minutes", "security alert") are equally specific.
   - None of the ten training-selected single-token rules exceeds a test AUROC of 0.780. The rule ranked first on training data ("2026") gives 0.753 [CD:separability.R1_TFIDF.single_token_rules].
   - The masks change R1 AUROC by at most 0.001. Among the digit-free variants that difference is detected, but it is negligible.
5. **Tool-name words only (TFIDF-toolname).**
   - R1 0.839 [0.821, 0.857].
   - R3 0.520 to 0.663, below LR-B in 5/5 folds after Holm [CD:settings.<S>.detectors.TFIDF-toolname].

**Change to the paper.**
- Add TF-IDF as a content-aware reference detector in every table. Alternatively, scope the title, Abstract, VII-E and X to content-blind tool-call detectors. The minimum is to do both: report TF-IDF and scope the general lessons.
- State which conclusions do not hold for content: in-distribution difficulty, the FPR explosion and the ranking reversal.
- State which do hold: R4 transfer failure and threshold non-transfer under R3, in the recall direction.
- Report AgentDrift's near-trivial content separability as an external-validity limitation.
- Report that within-AgentDojo content detection largely detects injections, not compromise.
- Consider a scoped title.

### M3. Domain-matched R1 comparator [analysis done]

**Key numbers.**
- **LR-C R1dm AUROC:** medical 0.786 [0.737, 0.831], banking 0.853 [0.803, 0.896], web 0.927 [0.898, 0.953], email 0.894 [0.859, 0.926], coding 0.894 [0.861, 0.925].
- **R3 − R1dm, LR-C:** medical −0.041 [−0.091, +0.012], banking −0.024 [−0.072, +0.029], web −0.065 [−0.095, −0.033], email −0.155 [−0.194, −0.113], coding −0.080 [−0.117, −0.042]. The decrease is Holm-detected in **web, email and coding** [DS:M3.R3_minus_R1dm.LR-C.AUROC; M3.degradation_folds.LR-C.AUROC]. With the pooled v0.3 comparator, the detected folds are medical, banking, email and coding instead.
- **On wAUPRC@10%:** the LR-C decrease is Holm-detected in 5/5 folds.
- **LR-B:** no detected R3 − R1dm AUROC change in any fold (−0.012 to +0.022).
- **In-distribution C − B (R1dm):** detected (Holm) only in web (+0.041), email (+0.090) and coding (+0.049). Medical is −0.007 [−0.035, +0.023] and banking +0.021 [−0.014, +0.053].
- **GRU − C (R1dm):** detected in medical, banking, email and coding. Web is −0.002 [−0.030, +0.023] [DS:M3.R1_domain_paired].
- **Reversal folds (plan 3.4):** listed under M1. A world-cluster bootstrap gives the same folds [DS:derived.M3.world_cluster_reversal_mismatches]. The only exception is GRU-drop0.2 vs LR-C on AUPRC@10%.
- **Test-set overlap (m3):** the R1dm and R3 pools share 345 pool-E trajectories per domain (276 for banking; 1,656 in total). The independent-bootstrap intervals are therefore conservative. A joint bootstrap narrows them to 0.80–0.96 of their width and changes no decision [DS:m3].
- **Pool sizes:** the domain-matched pools are small (111–180 positives, 165 negatives). "No detected in-distribution advantage" in medical and banking means wide intervals, not a zero effect.

**Change to the paper.**
- Add R1dm columns (AUROC, wAUPRC@10%, recall, FPR, paired C − B and GRU − C) to Tables V and VI and to Fig. 3.
- Base the degradation counts on R1dm.
- Reverse the web example: web's AUROC is **not** preserved (−0.065 [−0.095, −0.033] against 0.927).
- Delete lines 290, 292 and 403 as written.

### M4. Threshold failure is detector-specific [analysis done]

**Key numbers.** These are threshold failures against R1dm (plan 3.3), counted after Holm with the unadjusted count in brackets [DS:derived.M4.n_folds_failing_vs_dm_holm / _unadjusted]:
- LR-A 5 (5); LR-C 5 (5); Markov 5 (5); IForest 5 (5); LR-C-noobs 5 (5); GRU-drop0.2 5 (5); TFIDF 5 (5).
- LR-B **1 (1)**, in medical only.
- The pooled-seed GRU fails in 3 (4) folds, but every single GRU seed fails in every fold [DS:derived.M4.GRU_seeds|every_seed_fails_every_fold_vs_dm_holm = true; table M4-3]. For example, R3 banking recall by seed is 0.903, 0.006 and 0.957.

Direction of failure:
- **Over-alerting (FPR and recall up):** LR-C, LR-C-noobs and Markov. LR-C FPR is 0.121–0.321 in R1dm against 0.503–0.722 in R3.
- **Under-alerting (recall down):** LR-A, IForest, GRU-drop0.2 and TFIDF.
- **LR-B medical:** recall 0.560 [0.528, 0.590] against 0.694 [0.622, 0.761], Δ −0.135 [−0.208, −0.059]. Its FPR drop (0.150 against 0.236) is detected only before Holm.
- **LR-B email:** the FPR rise is **not** detected against email's own R1 value: 0.418 against 0.358, Δ +0.061 [−0.018, +0.138] [DS:M4.deltas.LR-B@t_r80.vs_domain_matched_R1].

**Change to the paper.**
- Define threshold failure as in plan 3.3.
- State it per detector. It fails in every fold for LR-C, Markov, IForest and TF-IDF, and for every GRU seed. LR-B's threshold carries over in four of five domains (all but medical).
- Add recall with intervals next to FPR in Table VI and Fig. 2.
- Relate LR-C's over-alerting to the measured offset (M1).
- Base the GRU statement on the per-seed table (M4-3), not the pooled-seed row.
- Reword lines 290, 294, 447 and 449.

### M5. Observation length is the dominant behavioural cue; truncation [analysis done + text]

**Key numbers.**
1. **Weights and univariate AUROC.**
   - mean_obs has the largest standardized weight: R1 LR-B −1.861 [−1.965, −1.770], and −2.008 [−2.183, −1.919] in LR-C's behavioural block [FO:M5_1_coefficients].
   - Its univariate AUROC is below 0.5 in every AgentDrift population: R1 0.257 [0.234, 0.280], R3 folds 0.187–0.313.
   - The direction flips in AgentDojo P0: 0.558 [0.518, 0.606] (0.595 on the v0.3 extraction). Detected reversals between R1 and P0 are max_args, mean_args and mean_obs [FO:M5_1_univariate].
2. **Range overlap.** 54.1% of P0 traces have max_obs above the AgentDrift maximum of 706 characters (48.8% on v0.3).
3. **Shortening outside attacked steps.** The benign-labelled steps of attacked trajectories are shorter (median 59 characters) than steps of benign trajectories (98). Steps of hard negatives are also short (73). So non-benign generated trajectories have shorter observations, most markedly the attacked ones. This is consistent with a generator property rather than with the injection itself.
4. **Without the observation-length features.**
   - LR-B loses 0.112–0.221 AUROC in every R3 fold (Holm).
   - The two features alone (LR-obsonly, post hoc) come within 0.053 of full LR-B.
   - LR-C-noobs < LR-B-noobs is detected in **5/5 folds on wAUPRC@10%, the plan's primary metric**, and in 4/5 on AUROC. Web is −0.003 [−0.024, +0.019] [FO:M5_2_without_observation_length.R3_holm].
   - R4 transfer rises: LR-B-noobs 0.630 [0.582, 0.680], LR-C-noobs 0.657 [0.611, 0.706]. This is a post hoc observation, and within minus transfer stays detected for every detector.
5. **Padding (R1).** Adding characters to the injection-point observation:
   - LR-C recall at θr80 falls 0.771 → 0.634 (+100) → 0.033 (+1,200).
   - GRU recall falls 0.682 → 0.000 (+1,200), with AUROC 0.159 [0.119, 0.199] [FO:M5_3_padding].
   - Within AgentDojo padding does not evade: LR-C recall 0.720 → 0.759 at +1,200.
   - We do not claim that padded injections still succeed. The review's larger AgentDojo rise is not reproduced.
6. **Truncation.**
   - The v0.3 extractor cut 448 of 14,079 P0 steps (3.18%, in 262 traces) at 2,000 characters.
   - The corrected extraction stores observations uncut: 500 P0 steps in 300 traces exceed 2,000 [FO:M5_4_truncation.v03_cut].
   - With the cut re-applied, within-AgentDojo LR-C is 0.846, against 0.797 without it. This is a descriptive point difference with no paired interval [FO:cross_references.linear_cache_r4_truncation_check].

**Change to the paper.**
- Revise VI-E, VII-B and VII-D to say that the zero-coverage behavioural signal in AgentDrift is largely an observation-length cue produced by one generator and written by the attacker, and that its sign reverses in AgentDojo.
- Disclose the v0.3 truncation and state that neither benchmark is cut now.
- Report the padding result under the threat model (M11).
- Label the noobs subsets as post hoc.

### M6. R4 Case decision [analysis done + text]

**Key numbers.** R4_CONTROL.md Sec. 7 is applied literally, including the "mixed or other" clause. The plan procedure and the v0.3 procedure give the same decisions [R4D:M6.decisions.plan|v03.<d>.case_literal]:

| Representation | Case (P0, C1, model-balanced, within-model; UT and IT) | Deciding numbers |
|---|---|---|
| LR-A | mixed: Case B and Case C conditions both hold (IT not reasonable) | P0 WUT 0.788; C1 WUT 0.802 reasonable; transfer 0.5655, near chance only via the "< 0.60" clause |
| LR-B | Case B | WUT 0.741, not reasonable in any population |
| LR-C | mixed: Case B and Case C conditions both hold (IT not reasonable) | P0 WUT **0.797 [0.747, 0.842]**, wAUPRC@10% **0.2985**, which misses the 0.80 and 0.30 bars by 0.003 and 0.002; C1 WUT 0.832 reasonable; IT 0.688 |
| GRU | Case A under UT; mixed: UT reasonable but IT not (seen attack goals only) | WUT 0.840 [0.784, 0.884], APw10 0.411; IT 0.645 |
| TF-IDF | Case A under UT | WUT 0.996; template-driven (M2) |

Further results:
- **Within minus transfer, UT.** Detected after Holm for every representation in P0, C1, model-balanced, WUT_F and P0F.
- **Within minus transfer, IT.** Only LR-B and TF-IDF survive Holm. LR-C: +0.220 [+0.038, +0.394], Holm p 0.051, borderline. GRU: +0.138 [−0.080, +0.337] [R4D:M6.decisions.holm_within_minus_transfer_cluster].
- **Unsuccessful attacks as evaluation negatives (P0F).** Transfer is no longer near chance: LR-C 0.626 [0.574, 0.670], LR-B 0.626, GRU 0.645 [0.593, 0.689]. The AgentDrift-trained models score unsuccessful attacks lower than benign runs. At θr80, transfer LR-C flags 0.056 of unsuccessful attacks against 0.225 of benign runs [R4D:M6.failed_flag_rates.by_setting.R4.LR-C].
- **WUT_F (unsuccessful attacks as training negatives).** On P0F, LR-C within is 0.819 and the GRU 0.869. Both are reasonable, but transfer is not near chance there, so this is not Case A.
- **Exploratory attempt target.** Every non-content representation is Case-B-like (within 0.584–0.613).
- **The "< 0.60" clause.** It was written after transfer was known (Part A of DEVIATIONS.md). It decides "near chance" for LR-A (all populations) and TF-IDF (C1).

**Change to the paper.**
- Restore the omitted clause in IV-F.
- Replace "The pre-registered conditions for Case A are met" with the per-representation outcome: "Case A under UT only for the GRU and the content detector. LR-C and LR-A mixed, missing the P0 bar by 0.003 AUROC and passing it on C1. LR-B Case B. Under IT no content-blind representation reaches the bar."
- Label the "< 0.60" clause post hoc.
- State the operational target (compromise). Report the attempt target as exploratory.
- Report P0F and WUT_F.
- Add LR-A, LR-B and GRU rows (UT, IT, model-balanced) to Table VII(b), and add IT to Fig. 4.
- Word RQ4 per representation.

### M7. AgentDojo extraction bug [analysis done]

**What was done.**
- Positional pairing, with a unit test that passes [DG:m14.alignment_test].
- No truncation.
- Re-extraction; 413 traces changed by content and 408 by per-step length, the count depending on the criterion [`revision/extraction_diff.json`].
- Re-ran `ablation.py`, `r4_control.py` and `gru_ladder.py` (final).
- The cache-based values reproduce the re-run scripts with maximum difference 0.0000 for points and intervals, and < 0.00005 for paired deltas [R4D:M7.reconciliation_with_rerun_v03_scripts]. A separate check in this consolidation found the same: re-run v0.3 R4 points against the caches agree to ≤ 5e-5, which is 4-decimal rounding [values file: rerun_v03_vs_cache].

**Key numbers.** v0.3 → corrected, with strat intervals as in v0.3 [R4D:M7.table]:

| Quantity | LR-A | LR-B | LR-C | GRU |
|---|---|---|---|---|
| Transfer AUROC | 0.566 → 0.5655 [0.543, 0.586] (unchanged) | 0.447 → 0.482 [0.448, 0.516] | 0.440 → 0.473 [0.441, 0.508] | 0.488 → 0.513 [0.482, 0.544] |
| Within-UT AUROC | 0.788 (unchanged) | 0.798 → 0.741 [0.708, 0.771] | 0.855 → 0.797 [0.766, 0.829] | 0.860 → 0.840 [0.794, 0.871] |
| Within-IT AUROC | 0.690 (unchanged) | 0.771 → 0.719 | 0.766 → 0.688 [0.634, 0.736] | not run → 0.645 [0.581, 0.691] |
| WUT − transfer | +0.222 → +0.223 | +0.351 → +0.259 | +0.415 → +0.323 [+0.277, +0.370] | +0.371 → +0.326 |
| Transfer FPR at θr80 | 0.043 (unchanged) | 0.332 → 0.225 | 0.347 → 0.225 | 0.230 → 0.168 |

- Per-model LR-C transfer moves 0.593 → 0.477 (gemini-1.5-pro) and 0.640 → 0.562 (gemini-2.0-flash).
- Per-model within-UT LR-C on corrected data is 0.745–0.794 in the four eligible models, against 0.792–0.849 in v0.3. These are the re-run `r4_control.py` values [RR:r4_control_results.json per_model.<m>.metrics.within_UT_full.AUROC].
- Most of the within-UT drop comes from removing the 2,000-character cut (see M5, item 6).

**Change to the paper.**
- Regenerate Table VII, Fig. 4 and the Abstract ranges.
- State the fix and the number of traces changed, with the criterion.
- Delete the model-composition explanation of the sub-0.5 pooled AUROC (see m1).

### M8. AUPRC at controlled prevalence [analysis done]

**Key numbers.** R1, wAUPRC at 1/2/5/10% [PO:M8_R1.detectors.<d>.wAUPRC.<π>]:
- LR-C 0.156 [0.121, 0.261] / **0.232 [0.183, 0.330]** / 0.380 [0.321, 0.462] / 0.523 [0.466, 0.590], with AUROC 0.879.
- GRU 0.265 / **0.382 [0.282, 0.537]** / 0.564 / 0.699, with AUROC 0.936.
- At 2%: LR-A 0.140, LR-B 0.152, Markov 0.047.
- IForest is close to the estimator's chance level at 2% (0.023 against 0.022). Its interval lies just above the chance mean.
- TFIDF 0.984 [0.973, 0.994].

The v0.3 subsampling estimate is kept only as a secondary quantity (the expected AP of a test set with n+ positives): LR-C at 2% is 0.259 [0.105, 0.466]. The weighted interval is 2.46 times narrower [PO:summary_derived.M8_LR-C_pi02_width_ratio_subsampling_over_weighted].

Paired comparisons with Holm within each prevalence [PO:M8_R1.paired.<π>]:
- **Detected comparisons after Holm (of 5):**
  - weighted estimator: 3/5/5/5 at 1/2/5/10%;
  - world-cluster bootstrap: 4/4/5/5;
  - v0.3 subsampling: 1/1/4/5.
- **LR-C − Markov:** +0.132 [+0.093, +0.235] at 1%, detected.
- **Borderline at 2%:** LR-C − LR-B +0.081 [+0.003, +0.168], Holm p 0.044, not detected under world clustering; GRU − LR-C +0.150 [+0.023, +0.296], Holm p 0.040.
- **Not detected at 1%:** LR-C − LR-B and GRU − LR-C.

AgentDojo, wAUPRC@10%, chance 0.103 (unrounded scores are preferred for AgentDojo, deviation P10):
- **Transfer (R4).** Near chance: LR-B 0.090 and LR-C 0.089 lie slightly below the chance mean, the GRU is at 0.104, and LR-A is slightly above chance at 0.138 [0.121, 0.178].
- **Within (WUT).** LR-B 0.228, LR-C 0.299 [0.246, 0.388], LR-A 0.303, GRU 0.411 [0.325, 0.528], TFIDF 0.980 [PO:M8_AgentDojo_raw_scores; M8_AgentDojo].
- **5% values.** Reported for every detector [R4D:m17_3].

**Change to the paper.**
- Make wAUPRC with the full-pool stratified bootstrap the primary estimate in Table IV, Fig. 1, the Abstract, Sec. VI-A and Table VII.
- Present subsampling only as small-test-set variability.
- Withdraw the "not detected" comparisons of VI-A (LR-C − Markov at 1%).
- Drop "which is how such numbers are often reported".
- Revise VII-E to recommend reweighting.
- Cite Boyd et al. 2013 and Siblini et al. 2020 (entries exist in RW).

### M9. Negative-class and hijack-pattern breakdown [analysis done; partial scope]

**Key numbers.**
1. **R1 LR-C at θr80** [PO:M9.R1.schemes.traj_strat.detectors.LR-C.t_r80]:
   - FPR on benign 0.118 [0.092, 0.147], on hard negatives 0.387 [0.320, 0.449]; failed attacks flagged 0.218 [0.164, 0.271].
   - Recall on full hijacks 0.830 [0.796, 0.863], partial 0.727 [0.671, 0.784], delayed 0.642 [0.567, 0.724].
2. **Precision at 2% by negative mix:**
   - benign-only 0.117 [0.097, 0.145];
   - the pool-E mix (27% hard negatives) 0.076 [0.067, 0.087];
   - hard-only 0.039 [0.034, 0.047].
   - That is a ratio of 3.0 [PO:summary_derived.M9_R1_LR-C_pi02_precision_ratio_benign_only_over_hard_only]. At 10% the three values are 0.420, 0.309 and 0.181.
3. **Hard negatives vs benign traces.** In R1, each of LR-A, LR-B, LR-C, Markov and the GRU flags hard negatives more often than benign traces. The Isolation Forest does not (0.609 against 0.628).
4. **LR-C − LR-B by subgroup (AUROC).**
   - The advantage is detected against both negative classes: benign +0.036 [+0.021, +0.051], hard negatives +0.082 [+0.054, +0.111].
   - It is detected on full hijacks: +0.092 [+0.075, +0.109].
   - It is **not** detected on partial hijacks (−0.015 [−0.040, +0.009]) or delayed hijacks (+0.007 [−0.022, +0.035]).
   - Against failed attacks it is +0.144 [+0.114, +0.175], which is a separate quantity outside pool E [PO:M9.R1…paired_AUROC_subgroups].
5. **R3 LR-B full-hijack recall:** medical 0.468, banking 0.659, web 0.840, email 0.820, coding 0.834.
6. **AgentDojo.**
   - R4 LR-C flags 0.224 of benign traces, as often as compromised ones (recall 0.225).
   - WUT TFIDF flags 0.000 of benign runs and 0.308 [0.286, 0.331] of unsuccessful attacks.
7. **Scope.** The breakdown covers all cached detectors in R1, R2, the five R3 folds, R4 and WUT. It is not given for R1-wd, WIT, WUT_F or LOSO (deviation P12). Failed-attack flag rates for WIT and WUT_F are in R4D:M6.failed_flag_rates, and TF-IDF per-class rates for every setting are in CD:settings.<S>.per_class_t_r80.

**Change to the paper.**
- Add the breakdown table.
- Treat the hard-negative share as a second mixing parameter.
- State the mix in the Abstract: "precision 0.076 with the pool-E negative mix of 27% hard negatives; 0.039 to 0.117 depending on the mix".

### M10. "Pre-registered" and deviations [text only; the deviation table is in DEVIATIONS.md]

The plans were internal, and their file times show which results were already known when each was written (DEVIATIONS.md Part A):
- R4_CONTROL.md: the transfer AUROCs 0.440 and 0.447 were known.
- PRE_ABLATION_DESIGN.md: LR-C R1/R2/R3 and LR-B/LR-C R4 were known.
- GRU_PREREGISTRATION.md: all linear results were known, including the C < B R3 pattern.

The review's M10 and m16 deviation lists were verified against the code (DEVIATIONS.md Part B), and one undisclosed deviation was added: gradient clipping at norm 1.0 in `gru_detector.py:83`, which the GRU plan does not list.

Two numbers the review asked for are now reported:
- B− (LR-B-minus): R1 0.827; R3 0.803/0.839/0.899/0.807/0.849 [linear cache validation, CACHE_README AUROC table].
- The Rule-4 test (M1).

**Change to the paper.**
- Replace "pre-registered" with "pre-specified internal analysis plan, not externally registered" throughout: Sec. I line 33, IV-F line 168, V-E, VI-G, IX line 439.
- For each plan, state which results were already known.
- Add the deviation table as supplementary material.
- Label the GRU reversal and the GRU-B diagnostic variant as exploratory.
- Correct `gru_audit.json` ("R1_val_source": it was a stratified 10% of train, not `data/val.jsonl`) and `GRU_RESULTS.md` lines 3 and 11. Both are text corrections to v0.3 files, which this consolidation did not edit.
- Supply the three plans, and this revision plan with its SHA-256 (`revision/PLAN_MANIFEST.txt`), as supplementary material.

### M11. Threat and deployment model [text only; optional experiment not run]

No prefix-level (before-each-call) evaluation was run.

**Text to add.** A "Threat and Deployment Model" subsection in Sec. III covering:
- the attacker: an indirect injector who controls tool-output content, with no model access and no adaptivity, whose goals include denial of service;
- what the defender sees: tool name, argument keys and values, and observation text for the content detector; argument-key counts and observation lengths only for the content-blind detectors;
- when the defender decides: post hoc, over whole trajectories;
- what an alert triggers;
- the unit behind "alerts per 1,000 traces", which is trajectories;
- attacker control of each feature. Observation length is attacker-written, as shown by the padding result under M5. Tool and argument features depend on the hijack.

Also reword "runtime detectors" (VII-E line 419) to post hoc or audit detection, and add an ethics and data statement (Q4).

### M12. Novelty and related work [text only; drafted]

`revision/related_work_v04.md` (sections 1–2) contains a rewritten gap statement crediting:
- Agent-Sentry's within-target grouped K-fold;
- TrajAD's leave-one-domain-out against an upper bound;
- MCPShield's Finding 3 and fingerprint probe;
- Fomin 2026.

It also has a revised Table I and an expanded Sec. II with subsections. `references_v04.json` holds about 75 verified entries.

**Change to the paper.**
- Insert the drafted text, and make the consequential edits in RW §5: the Abstract "usually" scoped to the audited sample, 16%–86%, 4 of 11 threshold-free, and crediting Agent-Sentry in contributions and VII-D.
- Resolve the open items in RW §6 (StepGuard timing, Skynet DOI).

## Minor comments

### m1. Model-balanced AUROC at native counts [analysis done]

Transfer AUROC with positives reweighted to the benign model mix (2,287 positives, 308 negatives) moves by at most a few hundredths from P0: LR-C 0.486 [0.435, 0.530], GRU 0.536 [0.487, 0.575] (cluster) [R4D:m1.model_balanced_native_counts]. So model composition does **not** explain the near-chance transfer on the corrected data.

LR-A − LR-C transfer [R4D:m1.paired["transfer:LR-A-minus-transfer:LR-C"]]:
- P0: +0.094 [+0.020, +0.168] (cluster);
- model-balanced: +0.068 [+0.016, +0.119] (strat) and +0.072 [−0.009, +0.153] (cluster).

This contradicts the v0.3 statement that the LR-A advantage does not survive balancing. LR-A's own transfer is only 0.555 (model-balanced).

**Change to the paper.**
- Report native-count model-balanced values for all detectors.
- Drop "at 10%" from Table VII(a).
- Delete "the apparent advantage of LR-A in transfer does not survive this control" and "the pooled transfer AUROC below 0.5 comes largely from model composition".

### m2. Cluster bootstrap [analysis done]

- **R2 minus R1, LR-C.** With a task-cluster bootstrap (35 tasks), R2 − R1 AUROC is +0.023 [−0.005, +0.049], which is not detected. The v0.3 trajectory bootstrap gave a detected +0.023 [+0.002, +0.045] [DS:m2.estimates.LR-C.AUROC.R2_minus_R1].
- **R2 minus R1, GRU.** −0.020 [−0.054, +0.008].
- **LR-C wAUPRC@10%, R2 − R1.** +0.092 [+0.002, +0.174], borderline. It is not detected with R1 world-clustered at B = 2,000 and is detected at B = 10,000.
- **AgentDojo.**
  - LR-C transfer widens from [0.441, 0.508] (strat) to [0.426, 0.512] (UT clusters) and [0.329, 0.599] (IT clusters, 30 positive clusters).
  - Within-UT minus transfer stays detected after Holm for every supervised detector.
  - Within-IT minus transfer is detected only for LR-B, TF-IDF and TF-IDF-nodigit (LR-C Holm p 0.160 in the 12-detector family) [R4D:m2].
- **R1 worlds.** World clustering barely changes the R1 intervals: LR-C 0.879 [0.861, 0.896].
- **Caveat.** The intervals omit training variability (Bengio and Grandvalet 2004).

**Change to the paper.**
- Use cluster intervals for AgentDojo and R2.
- Replace "+0.023 [+0.001, +0.046]" in VI-C with the task-cluster value, and say that no change is detected in either direction.

### m3. Multiplicity and overlap [analysis done]

- Holm is applied within the plan 3.6 families in every stream.
- Conjunctive claims that survive Holm:
  - LR-C < LR-B in R3 in 5/5 folds;
  - GRU < LR-C in R3 in 5/5;
  - TFIDF > LR-C in R3 in 5/5.
- Isolated marginal results that are now borderline:
  - R1 LR-C − LR-B wAUPRC at 2% (Holm p 0.044);
  - GRU − LR-C at 2% (Holm p 0.040);
  - WUT LR-C − LR-B at 10% (Holm p 0.048 unrounded, 0.051 rounded);
  - IT LR-C within minus transfer (Holm p 0.051);
  - R4 TFIDF − LR-C (p 0.057).
- The R1/R3 overlap is quantified under M3.
- Monte Carlo check: 12 borderline decisions change at B = 10,000. None changes a reversal fold, a threshold-failure cell of LR-B, LR-C or the pooled GRU, or a pooled m5 no-loss statement [DS:mc_stability_B10000].

**Change to the paper.** Add a multiplicity paragraph to IV-H, report counts with and without Holm, and describe the borderline cells as borderline.

### m4. Operating-point reporting [analysis done]

- **GRU, R1.** At θval80 (validation threshold): recall 0.775 [0.745, 0.804], FPR 0.073 [0.055, 0.092], precision at 2% 0.179 [0.147, 0.225]. At θr80: 0.682, 0.045 and 0.238 [PO:m4_gru_t_val80.table.R1.GRU].
- **FPR at a fixed test TPR of 0.8.** This uses test labels, so it is descriptive only: GRU 0.082 [0.061, 0.108], LR-C 0.210 [0.175, 0.244]. The GRU's advantage does not depend on the threshold.
- **θ = 0.5.** GRU in R1: recall 0.856 [0.819, 0.893], FPR 0.119 [0.084, 0.159]. Under R3, the LR-C FPR at θ = 0.5 is 0.538–0.733. Under R4, recall at θ = 0.5 is at most 0.301 [PO:m4_theta05].

**Change to the paper.**
- Report the GRU at θval80 next to θr80.
- Add the matched-recall comparison, labelled descriptive.
- Add a θ = 0.5 table, or delete "results at θ = 0.5 are qualitatively similar" (IV-I).

### m5. World-disjoint control and world count [analysis done]

**Worlds.**
- The split's own key gives **1,133 worlds** in the pool of 10,656 trajectories (1,141 in the full corpus, as the AgentDrift README states), with purity **0.883 [0.860, 0.909]** and largest worlds 120/95/91 [DG:m5_worlds.pool_10656].
- The v0.3 numbers (1,029 worlds, purity 0.858, largest 194/169/144) do not reproduce on the pool.
- A lead, not proof: the key user+date+sorted contact names on train+val (10,655 trajectories) reproduces 1,029 worlds, purity 0.8575, 54 worlds with ≥ 30 trajectories and a majority baseline of 0.558, but not the largest sizes (231/209/155) [DG:m5_key_search.rows_n_worlds_1029[2]]. So the v0.3 count was probably computed on the wrong population with a coarser key.
- R1 world reuse is 99.3% [98.9, 99.7]. The training-world attack rate predicts the test label with AUROC 0.963 [0.956, 0.970] [DG:m5_world_reuse.R1.all_test].
- R2 reuses worlds just as much: 99.3%.
- World lookup accuracy outside email is 0.94–0.99 in R1 (0.92–0.99 over R1 and R2); in email it is 0.48.

**World-disjoint versus R1, pool E, five split seeds.**
- Independent bootstraps detect no pooled loss (0 of 25).
- LR-B is below R1 in 5/5 seeds (−0.027 to −0.008). In seed 1, −0.027 [−0.054, +0.001] is borderline. With an overlap-aware bootstrap it becomes −0.027 [−0.052, −0.002], detected before correction but not after Holm over seeds (p_Holm 0.192) [DS:m5; m5_overlap_sensitivity].
- LR-A gains in every seed (+0.039 to +0.045).
- Per domain, 3 of 125 losses are detected unadjusted and none after Holm.
- Email makes up 0.553–0.621 of R1-wd pool E, against 0.208 in R1.
- Pool-E means across seeds: LR-A 0.821, LR-B 0.817, LR-C 0.892, GRU 0.956.

**Change to the paper.**
- Correct Sec. III-A to 1,133 worlds and purity 0.883.
- Replace the full-partition R1-wd row with pool-E values.
- Word "no detectable loss" per detector: it holds for LR-A, LR-C, the GRU and TF-IDF; for LR-B the verdict depends on the bootstrap.
- Note the changed domain mix.
- `pre_ablation_audit.py` is restored and listed.

### m6. GRU vocabulary and `<UNK>` [analysis done + text]

- The vocabulary is built from **benign** training trajectories only [R4D:m6.vocab_from_benign_only].
- In AgentDrift, `<UNK>` never occurs in training. Its norm falls from 5.60–6.15 to 0.11–2.33 through Adam's coupled weight decay alone (see M1, item 4).
- Within AgentDojo, `<UNK>` occurs in up to 334 steps in 327 compromised fit traces per fold.
- Removing `<UNK>`-bearing traces changes WUT GRU AUROC by less than 0.01, and the change is not detected:
  - refit without those traces: +0.005 [−0.009, +0.032];
  - test set without them: +0.007 [−0.007, +0.030] [R4D:m6.WUT_GRU_without_unk].
- The within-AgentDojo seeds are 0–4, one per CV repetition.

**Change to the paper.**
- Correct Sec. III-C line 120 ("any training trajectory" should be "benign training trajectories").
- Correct V-E line 228: `<UNK>` does occur in within-AgentDojo training, and "changes only through weight decay" should say it is an untrained random vector shrunk by the optimizer.
- Correct V-E line 226 and IX line 439: within-AgentDojo seeds are 0–4.

### m7. Short-run filter and DoS labels [analysis done + text]

- **The filter.** It removes 236 of 628 benign runs (0.376) but 70 of 2,364 compromised runs (0.030, all gpt-4o). By model, it removes 77 of 132 gemini-2.0-flash benign runs (0.583) [R4D:m7_Q4.exclusion_by_class, .exclusion_by_model].
- **Length alone.** Trace length has univariate AUROC 0.675.
- **DoS labels.** AgentDojo `benchmark.py:149-150` sets `security = not utility`, and lines 73-75 run only the first injection task. All 101 DoS positives have utility False; all come from gpt-4o.
- **Without DoS positives.**
  - Within-UT LR-C rises from 0.797 to 0.813 [0.763, 0.859] and the GRU from 0.840 to 0.855. LR-C then reaches the 0.80 point criterion.
  - Transfer LR-C is unchanged (0.475).
  - Within-UT LR-C recall at θr80 is 0.176 on DoS positives against 0.745 on the others.

**Change to the paper.** Report the exclusions by class and model, state the DoS label semantics precisely, and report R4 with and without DoS. The LR-C Case verdict is sensitive to DoS (M6).

### m8. Keyword state-change features [analysis done]

- **Against a state-change annotation** (single, non-blind annotator; F8), the keyword flag has tool-level precision / recall of 0.957 / 0.440 in AgentDrift and 0.833 / 0.682 in AgentDojo.
- **AgentDrift hijacked steps.** The flag catches 36.5% [35.6, 37.5]. It misses 4,101 of the 7,903 hijacked steps the annotation marks state-changing.
- **AgentDojo.** It misses 815 of 4,797 state-changing calls in compromised traces [FO:m8_keyword_flag].
- **Replacing the keyword features with annotation-based ones.** LR-B AUROC changes by at most 0.024. The change is detected after Holm in 2 of 5 R3 folds, in opposite directions (email −0.024, coding +0.007), and does not rescue transfer [FO:m8_LRB_annotation_features].

**Change to the paper.** Rename the features "calls whose tool name matches a write-like keyword", report the confusion, and add the sensitivity analysis.

### m9. Chance AP and ties [analysis done]

- **Expected AP of a random scorer.** 0.0171 at 8/825 (prevalence 0.0096), 0.0275 at 17/825, and 0.1126 at 44/392 [PO:m9_chance_AP.values]. The v0.3 IForest subsample APs (0.017, 0.030) are at chance.
- **Ties.** The caches store tie-exact scores, so R3 LR-A and Markov AUROC are exactly 0.5000.
- **12-decimal rounding.** It changes no AgentDrift value. In AgentDojo it merges saturated scores and moves point values by up to 0.0056 [PO:summary_derived.m9_max_abs_change_from_rounding], so unrounded AgentDojo values are quoted in this summary.

**Change to the paper.** Correct the chance lines in the figures, report exactly 0.500 for R3 LR-A, and use the unrounded AgentDojo values.

### m10. Very low prevalence and FPR budgets [analysis done]

**Eq. (6) for LR-C at θr80.** Precision is 0.0040 [0.0035, 0.0047] at 0.1% and 0.00040 [0.00035, 0.00047] at 0.01%.

**Thresholds that flag at most 1% or 5% of training negatives** (benign + hard negatives; primary reading) [PO:R1_operating_points.m10_fpr_budgets]:
- LR-C catches 0.276 [0.244, 0.307] of R1 attacks at a test FPR of 0.017 [0.008, 0.027] (1% budget), and 0.529 [0.495, 0.563] at 0.065 [0.048, 0.082] (5% budget). Its test FPR is not detectably above the budget.
- The GRU and TF-IDF overshoot clearly at the 1% budget: GRU FPR 0.070 [0.050, 0.092] (recall 0.770), TF-IDF FPR 0.033 [0.021, 0.045]. This is consistent with in-sample optimism of flexible models.
- With all label-0 training rows as negatives, LR-C gives 0.270 / 0.017 and 0.520 / 0.059, which are the review's figures.
- The largest content-blind precision at 0.1% over all thresholds examined is 0.0225 [0.0110, 0.0994] (LR-B, 1% budget, all label-0). It is a maximum selected after the fact and rests on about 5 false alarms.

**Change to the paper.**
- Add the FPR-budget operating points and the Eq. (6) extrapolation.
- Cite or drop "attacks should be rare".
- State that the 1% floor is set by pool size. Weighting removes that limit for AUPRC, but not for precision intervals.

### m11. Label-free threshold [analysis done]

A threshold set on 200 unlabeled held-out traces to reproduce the R1 alert rate lowers LR-C's R3 FPR from 0.503–0.722 at θr80 as follows [PO:m11_label_free]:
- **Reading (a), pre-specified** (native-prevalence sample, R1 alert rate 0.482): fold-mean FPR 0.217–0.312, recall 0.609–0.770.
- **Reading (b), post hoc** (2%-prevalence traffic, R1 alert rate at 2% of 0.203): FPR 0.179–0.221, recall 0.489–0.745, precision at 2% 0.045–0.071.
- **Other detectors.** For the GRU under (b), recall collapses to 0.123–0.273. For LR-B, the FPR under (b) is higher than its θr80 FPR in four of five folds (all but email).

The remedy restores an alert budget, not ranking quality. It works only if the unlabeled sample has the prevalence that the target alert rate assumes.

**Change to the paper.** Recommend it in VII-E as a partial remedy for detectors whose scores shift uniformly (LR-C), with both readings reported and reading (b) labelled post hoc.

### m12. Measured shift size [analysis done]

Classifier two-sample test on representation B, AUROC [DS:m12.settings.<S>.all_rows]:
- R1 0.501 [0.485, 0.515];
- R2 0.551 [0.537, 0.565];
- R1-wd 0.553–0.564;
- R3 folds: email 0.622, banking 0.701, coding 0.751, medical 0.842, web 0.866;
- R4 0.992 [0.991, 0.993];
- LOSO 0.887–0.971 [DS:m12_extra].

Shift size does not track degradation. Email has the smallest R3 shift but the largest LR-C loss against R1dm and the lowest LR-B R3 AUROC (Spearman −0.30 over five folds, descriptive only). The statistic does not capture the tool-vocabulary shift, which is total in R3 (κ = 0).

**Change to the paper.** Add the column to Table III and state that it measures behavioural shift only.

### m13. Benchmark provenance and regularities [analysis done + text]

**Provenance** [DG:m13_provenance]:
- Generator: Llama-3.3-70B-Instruct (quantized), temperature 0.8 (README:24; DATASHEET:64, 66).
- Generation prompts and scripts not released (README:177-178).
- Release 1.0 of 2026-09-07 (CHANGELOG:3); every record carries schema version "2.0".
- Licence CC BY 4.0. AgentDojo is MIT.
- The note that company names coincide with real organizations is in AgentDrift (README:182-183), not AgentDojo.

**Regularities, recomputed** (the README does not define its terms):
- ext-audit.com appears in 0.281 of all 7,036 poisoned observations, close to the README's 28.9%. It appears in 0.357 of successful-injection observations and 0.000 of failed ones.
- 0.985 of hard negatives carry at least one of the three markers the README names; the README's 75% is not reproduced under any definition tried.
- Banking has 256 full hijacks against 720 in each other domain.
- Per-fold pattern composition: banking 217/260/149 (0.347 full) against 0.585–0.606 full elsewhere.
- Pattern standardisation lowers banking's R3 AUROC by 0.037 (LR-B), 0.050 (LR-C) and 0.051 (GRU), while no other fold moves by more than 0.008 [DS:m13.banking_caveat].
- LR-C < LR-B survives standardisation in 5/5 folds.

**Change to the paper.**
- Name the generator and state the version accurately ("release 1.0, record schema 2.0").
- State that the generation code is unavailable.
- Summarise the template regularities, which link to M2.
- Add per-fold composition next to Tables V and VI, and caveat cross-fold comparisons that involve banking.

### m14. Reproducibility [done]

- **Documentation.** `revision/REPRODUCIBILITY.md` documents the environment variables (with file:line), `baseline_drift.py`, the LR and IForest settings, and every bootstrap seed.
- **Lock file.** `revision/requirements-lock.txt` lists numpy 2.2.6, scipy 1.15.3, scikit-learn 1.7.2, torch 2.11.0+cpu and matplotlib 3.10.8 on Python 3.10.4.
- **Alignment test.** `tests/test_agentdojo_alignment.py` passes (3 passed).
- **Parsing.** All 38 scripts parse.
- **Diagnostics.** The Sec. III-A and VI-B diagnostic numbers now have keys [DG:manuscript_comparison]. Of 32 comparisons, 18 agree to 3 decimals; the rest are the corrections under m5 and m15.

**Change to the paper.**
- Replace the file inventory in Sec. IX with a code-and-data availability statement.
- List the revision result files.
- Keep the "every number comes from the result files" claim only once the Sec. IX list includes them.

### m15. Probes [analysis done]

**Domain and world probes** (nested, regularized, with floors) [DG:m15_probes]:
- Domain: A 1.000 against a majority floor of 0.207; B 0.458 [0.449, 0.467], and 0.505 on benign-only trajectories.
- World (54 worlds): A 0.096, equal to the domain-only floor of 0.097; B 0.059 is detectably *below* domain-only. The probe population is 81% email.

**Model probe on corrected P0** [R4D:m15.AUROC]:
- B 0.488 [0.402, 0.578], C 0.427 [0.342, 0.504], suite floor 0.459. No encoding is detected.
- The v0.3 B value of 0.615 came from the extraction bug: the same unregularized probe gives 0.503 on corrected data.

**Change to the paper.**
- Replace "our features do not encode world identity" with "linear probes do not identify worlds beyond the domain". Rest world-leakage conclusions on the world-disjoint control.
- Replace "model identity is only weakly encoded" with "a regularized probe does not detect model identity in B or C (92 against 300 traces)".

### m16. GRU Outcome 4 and model-balanced population [analysis done]

**Outcome 4.** Criteria were defined post hoc (R4D5) [R4D:m16]:
- The sign of the R3 degradation is consistent across seeds in 5/5 folds (C1).
- The across-seed AUROC range exceeds the single-seed interval width in 3/5 (C2).
- The range of FPR or recall exceeds it in 5/5 (C3).
- The range is ≥ 0.10 in 4/5 (C4).

So Outcome 4 ("large seed variance") is triggered on the operating point. For example, R3 banking FPR by seed is 0.619, 0.000 and 0.694.

**Model-balanced GRU at native counts.** 0.536 [0.501, 0.569] under the pre-specified stratified interval, so Outcome 1 is missed by 0.0007, within Monte Carlo error. Under the cluster interval it is 0.536 [0.487, 0.575], which includes 0.5 [R4D:m1.GRU_outcome1]. Report Outcome 1 as indeterminate at the boundary. Outcome 3 is met; Outcome 2 is not.

**Change to the paper.** Rewrite VI-G lines 376 and 378. The GRU outcomes do not simply "match the pattern we specified in advance": Outcome 4 is triggered and Outcome 1 is at the boundary. Add both to the deviation table.

### m17. Internal inconsistencies [text + analysis done]

1. **Eq. (2).** The Markov score is unbounded, so write f(τ) ∈ ℝ. This is a text change.
2. **"Always the same pool" (line 114).** It holds now: the R1-wd rows use pool E (m5).
3. **AgentDojo at 5%.** Reported [R4D:m17_3]. Transfer wAUPRC@5% is 0.044–0.086 (chance 0.05). WUT: GRU 0.271, LR-C 0.173.
4. **Coverage for the within-AgentDojo control.** Tool-occurrence κ is 0.976 for WUT and WUT_F and 0.980 for WIT (mean over folds). LOSO κ is 0 / 0 / 0.084 / 0.362 [`revision/cache/linear/CACHE_README.md`, Settings table; validation.json]. Add these to Table III.
5. **Each comparison at every prevalence.** Given under M8. The subsampled LR-C − Markov at 2% is +0.196 [+0.030, +0.403], detected before Holm but not after.

### m18. Overreaching statements [text only]

- Replace "rank did not carry over ... to another benchmark" with "no detector transferred".
- Scope "two detector families" and the GRU generality to one untuned configuration (see M1).
- Delete "for these detectors, in-distribution performance was a poor guide to robustness" unless it is qualified (Part 2, A10).

### m19. Related-work characterizations [text only; drafted in RW]

- Replace 19% with 16%.
- Replace "threshold-free metrics" with "4 of 11 report a threshold-free metric".
- Recharacterize AlignmentCheck as a few-shot LLM auditor over the whole trace.

### m20. References [text only; not this consolidation's task]

Corrected metadata is in `references_v04.json` (single-author entries, the Praetor/Dang title, authors for [9]–[12], volumes and pages). Insert it at submission.

### m21 to m25, m27 to m29. Presentation [text only]

- m21: remove the draft header, footnote and revision-history appendix, and add an availability statement.
- m22: bring the abstract to 250 words or fewer and expand AUROC, AUPRC and GRU.
- m23: bring the paper to 13 pages or fewer and remove figure/table duplication. For the one-shared-bootstrap request, cite one canonical source per table:
  - prevalence tables: PO;
  - R1/R1dm/R3 tables: DS;
  - mechanism: OOV;
  - R4: R4D;
  - content: CD.
- m24: use θ everywhere, and rename the Cases (Case A/B/C) or the representations to remove the clash. B for bootstrap replicates → B_boot. GRU-B → "LR head on B, GRU training".
- m25: `\mathbf{1}`.
- m27: figure fixes; Fig. 4 needs a zero baseline and IT bars.
- m28: define settings once and remove repeated caveats.
- m29: "the features are not inadequate" is withdrawn anyway (M6). Replace "identical values" with "identical point estimates".

### m26. Terminology [text only]

"Detected" → keep, but define it once as "the 95% interval excludes zero", or use "statistically detectable". "Near chance" should say explicitly when an interval lies below 0.5: on P0, LR-B and LR-C are within 0.02–0.03 of 0.5, with intervals that include 0.5 under the cluster bootstrap. Define "native prevalence" at first use.

## Questions

### Q1. GRU interval estimand [analysis done]

- **Per-seed GRU − LR-C, email:** −0.096, −0.025 and −0.085, each below zero [OOV:s7_q1.folds.R3_email.GRU.minus_LR-C_per_seed].
- **Estimands:** in every R3 fold, GRU − LR-C is detected below zero under all three:
  - pre-registered mixture (one seed per replicate);
  - two-level (seeds then trajectories);
  - seed-averaged.
- **Example, email, all three:** mixture [−0.106, −0.009], two-level −0.069 [−0.101, −0.027], seed-averaged [−0.082, −0.055].
- **Seed level:** with 3 seeds it is coarse, so the two-level interval is approximate.

**Change to the paper.** State the estimand (the mixture interval describes a randomly drawn single-seed deployment), report per-seed values, and optionally add the two-level interval.

### Q2. Model-template confound in R4 [analysis done]

- **Per template:** within gpt-4o, transfer varies by template from 0.309 to 0.666 for LR-C and from 0.295 to 0.716 for the GRU. The comparisons share 92 negatives, so they are not independent [R4D:Q2.per_template_range_transfer].
- **important_instructions positives only** (728 positives, 392 negatives): pooled transfer is LR-C 0.476, GRU 0.522; model-balanced 0.477 and 0.526; within-UT 0.794 and 0.842 [R4D:Q2.II_only].
- **Conclusion:** the template confound does not create the near-chance transfer, but it drives part of the subgroup spread.

**Change to the paper.** Add a per-template sensitivity analysis to VI-F.

### Q3. Audit selection [text only; drafted in RW §3]

The drafted text gives:
- targeted arXiv and Google Scholar searches up to September 2026;
- the inclusion criterion;
- exclusions with reasons (MELON, TraceSafe, ATBench, AgentDoG, StepGuard);
- the reason TrajAD and Trajectory Guard are included;
- a plain statement that the audit is not a systematic review.

Scope the Abstract's "usually" to the audited sample.

### Q4. DoS positives and ethics statement [analysis done (DoS) + text (ethics)]

The DoS analysis is under m7. The ethics and data statement still has to be written. It should cover:
- licences (CC BY 4.0, MIT);
- the AgentDrift real-organization note (README:182-183);
- redistribution of the extracted traces;
- responsible reporting of the observation-length evasion finding (M5, padding).

### Q5. Why R2 shows no MCPShield-style drop [text only]

Evidence for the answer:
- R2 keeps full tool coverage (κ = 1).
- Its 35 test tasks give wide intervals: LR-C R2 − R1 is +0.023 [−0.005, +0.049] (m2).
- R2 reuses worlds as much as R1 (99.3%; DG).
- In AgentDrift, all tasks of a domain share one tool set, so a task-disjoint split does not remove task-specific tool signatures, whereas MCPShield's shift does. RW §2 states the tool-overlap point for MCPShield.
- LR-A in R2 is 0.788 [0.765, 0.809] against 0.777 in R1 (linear cache). No task-cluster difference is computed for LR-A.

Draft the Q5 answer in VI-C along these lines.

---

# Part 2. Claims decision table (v0.3 manuscript)

The Section 5 decision rules of REVISION_ANALYSIS_PLAN.md are applied literally:
- **(R-a)** A claim that "X holds in all five folds" is kept only if it holds after Holm; otherwise the count is reported.
- **(R-b)** A reversal is stated only for folds that meet plan 3.4, together with whether it survives dropout at the pre-specified p = 0.2.
- **(R-c)** The R4 Case is re-derived per representation and population with R4_CONTROL.md Sec. 7, including its mixed clause.
- **(R-d)** Deviations are logged.

Status is **kept**, **revised** (the claim stays in changed form) or **withdrawn**. Line numbers refer to MANUSCRIPT_DRAFT.md v0.3.

| # | Location | v0.3 claim (short) | Status | Revised wording (plain) | Evidence | Rule |
|---|---|---|---|---|---|---|
| A1 | Abstract, l.13 | Detectors are "usually" tested with many attacks and threshold-free metrics | revised | "In the eleven evaluations we audited, test sets contain 16% to 86% attacks, only four report a threshold-free metric, and none varies attack prevalence." | RW §5 items 1–2; m19, Q3 | |
| A2 | Abstract; VI-A; X | LR-C AUPRC at 2% 0.261 (CI 0.109–0.456); precision 0.077 | revised | "At 2% prevalence LR-C's AUPRC is 0.232 (0.183 to 0.330) and its precision at the training threshold is 0.076 with the evaluation pool's negative mix (27% hard negatives); precision ranges from 0.039 to 0.117 as the mix changes." | PO:M8_R1.detectors.LR-C.wAUPRC.pi02; R1_operating_points…LR-C.t_r80.precision_pi02; M9 | |
| A3 | Abstract; VI-B; VI-C; X | World- and task-disjoint splits show no detectable loss | revised | "Holding out worlds costs no detectable AUROC for LR-A, LR-C, the GRU and the content detector. LR-B is lower in all five split seeds, and one seed's loss is detectable only when the shared test trajectories are taken into account. On unseen tasks with known tools no change is detected in either direction." | DS:m5; m5_overlap_sensitivity; m2 | |
| A4 | Abstract; I (contributions); VII-A; X | "The in-distribution ranking reverses in two detector families" | revised | "LR-C ranks below LR-B, and the GRU below LR-C, in all five held-out domains. A reversal of a detected in-distribution advantage is established in three domains (C vs B) and four (GRU vs C). Both follow from untrained out-of-vocabulary inputs and are reproduced inside R1 by removing tool coverage. The GRU reversal does not survive token dropout, and the C vs B reversal shrinks to a near tie under tool-block dropout." | DS:M3.reversal; OOV:s2_graded, s3_dropout | R-a, R-b |
| A5 | Abstract; VI-E | Combined features lose to behavioral in all five held-out domains | kept (as an R3 ordering) + revised (as a reversal) | Keep "LR-C ranks below LR-B in all five held-out domains (after Holm)". Drop "beat ... in-distribution" as a per-domain statement: the in-distribution advantage is detected in 3 of 5 domains. | DS:M3.R3_paired / R1_domain_paired LR-C−LR-B | R-a, R-b |
| A6 | Abstract; VI-G | GRU beats LR-C in-distribution and loses in all five | kept (R3 ordering) + revised | "The GRU ranks below LR-C in all five held-out domains (after Holm; every seed). The in-distribution advantage is detected in four of five domains (not web)." | DS:M3.R3_paired / R1_domain_paired GRU−LR-C; OOV:s7_q1 | R-a, R-b |
| A7 | Abstract; VI-D | LR-C FPR at θr80 rises from 0.19 to 0.50–0.72 | kept | Keep, and add the domain-matched baseline (0.12–0.32) and the measured cause (a constant +1.65 to +1.90 logit offset). | DS:M4.values LR-C@t_r80; OOV:s1_lrc_offset | |
| A8 | Abstract; VI-F | AgentDrift-trained detectors are near chance on AgentDojo (AUROC 0.44 to 0.57) | revised | "Transfer AUROC is 0.47 to 0.57 (LR-C 0.473, LR-B 0.482, GRU 0.513, content 0.520, LR-A 0.5655). LR-A counts as near chance only through the one-sided < 0.60 clause. If unsuccessful attacks are counted as negatives, transfer rises to 0.63–0.65 and is no longer near chance for LR-B, LR-C and the GRU." | R4D:M7.table; M6.configs P0_UT / P0F_UT | R-c |
| A9 | Abstract; VII-D | The same features trained within AgentDojo reach 0.79 to 0.86 | revised | "Trained within AgentDojo on unseen user tasks, the content-blind detectors reach 0.74 to 0.84 (LR-B 0.741, LR-A 0.788, LR-C 0.797, GRU 0.840), and 0.65 to 0.72 on unseen attack goals." | R4D:M7.table | |
| A10 | Abstract (last sentence); X | In-distribution performance was a poor guide to robustness | revised | "For the content-blind detectors we studied, the in-distribution ranking did not predict the ranking under a disjoint-vocabulary domain shift because of untrained out-of-vocabulary inputs, and no detector, content-aware or not, transferred to AgentDojo." | M1, M2, M6 | |
| C1 | I, l.29 and II-D l.74 | No audited evaluation combines graded shift with cross-benchmark transfer and a within-target control | revised | Credit Agent-Sentry's within-target control, TrajAD, MCPShield and Fomin. What is new is running controlled prevalence, coverage reporting, a zero-overlap domain shift and synthetic-to-recorded transfer jointly on one detector set. | RW §1 | |
| C2 | I, l.33 | Tool-vocabulary coverage is reported for every setting | revised | True after the revision: add within-AgentDojo κ (0.976 to 0.980) and LOSO κ to Table III. | cache/linear CACHE_README | |
| C3 | I, l.33 | Ablation: the in-distribution gain of tool identity disappears; C ranks worse than B in all five; its threshold fails | revised | As A4 and A5. Also: "most of the behavioral signal at zero coverage is an observation-length cue that the attacker writes and that reverses across benchmarks." | FO:M5_2; M5_3 | R-a, R-b |
| C4 | I, l.33; V-E; VI-G; IX | A "pre-registered" GRU; decision rules of three experiments written down before their results | revised | "pre-specified internal plans, not externally registered". For each plan, state the results that were already known (DEVIATIONS.md Part A). | file mtimes | R-d |
| C5 | I, l.33 | The GRU shows the reversal is not specific to the linear model; diagnostic ties it to the tool-sequence pathway | revised | "The GRU's R3 deficit comes from an untrained `<UNK>` embedding (tool-pathway offset +6.3 logits on held-out negatives). Token dropout trains it and removes the deficit relative to LR-C." Label the reversal exploratory. | OOV:s4_gru_offset; s3_dropout | R-b |
| C6 | I, l.33 | Every estimate has stratified bootstrap CIs; thresholds are fixed on training data | revised | Keep the thresholds statement. Add: weighted AUPRC as the primary estimand; cluster bootstraps for AgentDojo and R2; Holm within families. | plan §3 | |
| C7 | I, l.33 | Claims concern the detectors evaluated here, not published detectors | kept | Keep, and add that a content-aware TF-IDF reference is included. | CD | |
| S1 | III-A, l.80 | 1,029 worlds; purity 0.858; AgentDrift "v2.0" | revised | "1,133 worlds; purity 0.883 (0.860 to 0.909); release 1.0 (record schema 2.0), generated by Llama-3.3-70B-Instruct." | DG:m5_worlds; m13_provenance | |
| S2 | III-B, l.86; VI-F l.368; VIII | The 101 DoS positives were not analyzed separately | withdrawn | Replace with the m7/Q4 results (DoS labels mean task failure; removing them gives within-UT LR-C 0.813, transfer unchanged). | R4D:m7_Q4 | |
| S3 | III-C, l.114 | AUROC and AUPRC are always computed on pool E | kept (made true) | All revision tables use pool E, including R1-wd. | DS:m5 | |
| S4 | III-C, l.120; V-E l.228 | GRU vocabulary from any training trajectory; `<UNK>` never occurs in training; changes only through weight decay | withdrawn | "Vocabulary from benign training trajectories. `<UNK>` does not occur in AgentDrift training but does within AgentDojo (up to 334 steps); in AgentDrift it is an untrained random vector that the optimizer shrinks to a norm of 0.11 to 2.33." | R4D:m6; OOV:s4_gru_offset.unk | |
| S5 | V-B, l.214 | State-changing count and fraction | revised | "calls whose tool name contains a write-like keyword" (precision 0.96, recall 0.44 against an annotation) | FO:m8_keyword_flag | |
| S6 | IV-F, l.168 | We fixed the R4 decision rule before computing its results | withdrawn | "Part 1 of R4_CONTROL.md was written after the transfer AUROC (0.440) was known; the < 0.60 clause is therefore post hoc." Restore the mixed clause. | Part A of DEVIATIONS.md | R-c, R-d |
| S7 | IV-I, l.202 | Results at θ = 0.5 are qualitatively similar | revised | Show them. Under R3, LR-C FPR at 0.5 is 0.538 to 0.733; under R4, recall at 0.5 is at most 0.301. | PO:m4_theta05 | |
| S8 | V-E, l.226; IX l.439 | Seeds 0, 1, 2 | revised | "Within AgentDojo, one training seed per CV repetition (0 to 4)." Also state gradient clipping as a deviation from the GRU plan. | gru_ladder.py:136-153; gru_detector.py:83 | R-d |
| R1a | VI-A | LR-C AUPRC 0.528 at 10% → 0.261 at 2% → 0.192 at 1% | revised | 0.523 / 0.232 / 0.156 (weighted estimator) | PO:M8_R1 | |
| R1b | VI-A | Precision 0.310 at 10% → 0.077 at 2%; 204 alerts per 1,000; about 12 false alarms per true detection | kept | 0.309 / 0.076 [0.067, 0.088]; 203 alerts per 1,000; 12.2 false alerts per true alert (pool-E mix) | PO:R1_operating_points; summary_derived | |
| R1c | VI-A | GRU AUPRC at 2% 0.418; precision 0.243; recall 0.682 | revised | wAUPRC 0.382 [0.282, 0.537]; precision 0.238 at θr80 and 0.179 [0.147, 0.225] at the validation threshold (recall 0.775) | PO:M8_R1; m4_gru_t_val80 | |
| R1d | VI-A, l.251 | AUPRC at low prevalence is much less stable than AUROC; LR-C − Markov not detected at 1%; GRU − LR-C not detected at 2% | withdrawn | "With all positives used, LR-C − Markov is detected at every prevalence (+0.132 [+0.093, +0.235] at 1%). GRU − LR-C is detected from 2% upward, borderline (+0.150 [+0.023, +0.296], Holm p 0.040). A single small test set is highly variable (subsampled interval 2.5 times wider)." Drop "which is how such numbers are often reported". | PO:M8_R1.paired | |
| R1e | VI-A, l.253 | AUROC paints a far too favorable picture at low prevalence | kept | Strengthened: GRU AUROC 0.936 with wAUPRC 0.382 at 2%. | PO | |
| R2a | VI-B, l.261 | 99.3% world reuse; training-world attack rate predicts labels with AUROC 0.963 | kept | Same values. Add that R2 reuses worlds equally (99.3%). | DG:m5_world_reuse | |
| R2b | VI-B, l.261 | Features do not encode world identity (A 0.066, B 0.046 < domain-only 0.097) | revised | "Linear probes do not identify worlds beyond what the domain gives (A 0.096 = domain-only 0.097; B 0.059)." | DG:m15_probes.world | |
| R2c | VI-B, l.261 | Domain: A 1.000 vs 0.207; B 0.455 | kept | 1.000 and 0.458 (nested probe) | DG:m15_probes.domain | |
| R2d | VI-B, l.263 | World reuse does not inflate R1; LR-C 0.891, GRU 0.956 (seed 0); GRU advantage +0.065 | revised | Keep the seed-0 values. Replace the five-seed full-partition means with pool-E means (LR-A 0.821, LR-B 0.817, LR-C 0.892, GRU 0.956). Word the no-loss claim per detector (A3). | DS:m5.across_seeds_AUROC | |
| R2e | VI-B, l.265 | Domain shortcut possible; banking prevalence 0.401 vs 0.522 | kept | Add the hijack-mix difference (banking 34.7% full hijacks). | DG:prevalence_by_domain; DS:m13 | |
| R2f | VI-C, l.286 | R2 − R1 +0.023 [+0.001, +0.046]; GRU − LR-C in R2 not detected | revised | "+0.023 (−0.005 to +0.049) with task-cluster resampling; not detected." Keep the GRU statement. | DS:m2 | |
| R3a | VI-D, l.290 | LR-C AUROC decrease detected in four of five folds, not web | revised | "Against the same domain's R1 score, the decrease is detected in web, email and coding (−0.065 to −0.155), not medical or banking." | DS:M3.degradation_folds.LR-C | R-a |
| R3b | VI-D, l.290 | AUPRC@10% decrease in three folds; at 2% only medical | revised | "The wAUPRC@10% decrease against R1dm is detected in all five folds (after Holm)." | DS:M3-4 | R-a |
| R3c | VI-D, l.290 and 294; X l.447 and 449 | Training-derived operating points fail in every fold; thresholds do not carry over | revised | "The threshold fails in every fold for LR-C, Markov, the Isolation Forest, the content detector and every GRU seed. LR-B's threshold carries over in four of five domains (it fails in medical, in the recall direction). The direction of failure depends on the detector." | DS:derived.M4 | R-a |
| R3d | VI-D, l.292; VII-A l.403 | Web: AUROC close to R1, yet FPR 0.707 | withdrawn | Web's AUROC falls 0.065 against its own R1 value. Use banking or coding to illustrate threshold failure, or none. | DS:M3 | |
| R3e | VI-D, l.292 | Markov constant (FPR 1.000); IForest flags almost nothing | kept | AUROC exactly 0.500 (tie-exact) | cache CACHE_README | |
| R3f | VI-E, l.302 | R1 order C > B > A, all detected | kept | +0.049 / +0.052 | DS:M3.R1_pooled_paired | |
| R3g | VI-E, l.302 | LR-C below LR-B in all five held-out domains (−0.026 to −0.063, all detected); LR-B threshold holds better | kept | Same numbers, detected after Holm. LR-B's threshold holds in 4 of 5 domains. | DS:M3.R3_paired; M4 | R-a |
| R3h | VI-E, l.308 | A common offset would fit, but we did not measure it | withdrawn | "The offset is +1.65 to +1.90 logits in every fold; only 11% to 15% of training trajectories reach it." | OOV:s1_lrc_offset | |
| R3i | VI-E, l.310 | Behavioral features keep much of their within-benchmark signal at zero coverage | revised | "... and that signal is largely the mean observation length, which is shorter in the generator's attacked trajectories, written by the attacker, and reversed in AgentDojo; without it LR-B loses 0.11 to 0.22 AUROC." | FO:M5_1; M5_2 | |
| R4a | VI-F, l.334 | Every detector near chance in transfer (0.566/0.447/0.440/0.488) | revised | As A8, with corrected values | R4D:M7.table | R-c |
| R4b | VI-F, l.334 | Within 0.788–0.860; within − transfer +0.222 to +0.415, all detected | revised | "Within 0.741–0.840; within − transfer +0.223 to +0.329 (cluster), detected after Holm for every representation under UT." | R4D:M7; M6 | |
| R4c | VI-F, l.334 | Transferred LR-C recall 0.283, FPR 0.347, precision at 10% 0.084; within recall 0.749, FPR 0.173, precision 0.328 | revised | Transfer: recall 0.225, FPR 0.224, precision at 10% 0.100. Within (WUT): recall 0.720, FPR 0.216, precision 0.274. | PO:M9.R4 / WUT LR-C t_r80 | |
| R4d | VI-F, l.334 | IT: LR-C 0.766, above transfer by +0.326 | revised | "0.688; within − transfer +0.220 (+0.038 to +0.394), not detected after Holm (p 0.051)." | R4D:M6.configs.P0_IT | |
| R4e | VI-F, l.334 | The pre-registered conditions for Case A are met; the features are not inadequate | withdrawn | Per-representation outcome as in M6: Case A under UT only for the GRU and TF-IDF; LR-C and LR-A mixed; LR-B Case B; no content-blind representation reasonable under IT. | R4D:M6.decisions | R-c |
| R4f | VI-F, l.351 | C1: within 0.867, transfer 0.444; within flags 35.4% of removed traces vs 21.4% | revised | "C1: within 0.832, transfer 0.472; at θ = 0.5, within LR-C flags a share of 0.369 of the 65 injection-task benign traces, against 0.2355 of the other benign traces." | R4D:M6.configs.C1_UT; RR:r4_control_results.json injection_task_benign_diagnostic.within_UT_full | |
| R4g | VI-F, l.351 | Per-model within 0.792–0.849, transfer 0.447–0.640, all detected | revised | "Per model, within-UT LR-C is 0.745–0.794 and transfer 0.444–0.562; the within − transfer difference is detected in every model." | RR:r4_control_results.json per_model | |
| R4h | VI-F, l.351 | Sub-0.5 pooled transfer comes largely from model composition; LR-A advantage does not survive model balancing | withdrawn | "Model balancing moves transfer by at most about 0.04; LR-A − LR-C stays detected under the stratified interval." | R4D:m1 | |
| R4i | VI-F, l.351 | Model identity is only weakly encoded (B 0.615) | withdrawn | "A regularized probe does not detect model identity (B 0.488); the v0.3 value came from the extraction bug." | R4D:m15 | |
| R4j | VI-F, l.368 | The problem lies in the transfer and not in the features | revised | "Transfer fails for every representation, including a content detector. The within-AgentDojo control reaches the pre-specified bar for the GRU and the content detector on unseen user tasks, for LR-C only after removing injection-task benign traces, and for no content-blind representation on unseen attack goals." | R4D:M6 | R-c |
| R5a | VI-G, l.376 | GRU beats LR-C in R1 (+0.056); advantage persists in R1-wd; no difference in R2 | kept | +0.057 [+0.042, +0.072]; R1-wd +0.065 (seed 0); R2 not detected | DS; v0.3 | |
| R5b | VI-G, l.376 | GRU below LR-C in all five (−0.068 to −0.136, all detected); R3 AUROC 0.639–0.736 | kept | Detected after Holm in 5/5, for each seed. Reversal established in 4/5. | DS:M3; OOV:s7_q1 | R-a, R-b |
| R5c | VI-G, l.376 | Threshold differs strongly between seeds | kept | Add that Outcome 4 is triggered (m16). | R4D:m16 | |
| R5d | VI-G, l.376 | R4 GRU 0.488 (P0), 0.579 model-balanced; within 0.860 | revised | 0.513 (P0); model-balanced at native counts 0.536 (0.487 to 0.575, cluster); within 0.840 | R4D:M7; m1 | |
| R5e | VI-G, l.378 | Outcomes match the pattern specified in advance | withdrawn | "Outcome 3 is met. Outcome 1 is at the boundary (missed by 0.0007 under the pre-specified interval, met under the cluster interval). Outcome 4 (large seed variance) is triggered on the operating point." | R4D:m1.GRU_outcome1; m16 | R-d |
| R5f | VI-G, l.378 | GRU below LR-C is an unexpected finding | revised | Label it exploratory and explain it by the untrained `<UNK>` (C5). | OOV | |
| R5g | VI-H | Order GRU-B > LR-C > GRU-C > GRU-A in every fold; six differences detected | kept | This is a v0.3 result (gru_r3_diagnostic_results.json), not re-run. Label GRU-B as a post hoc variant. | v0.3 | |
| R5h | VI-H, l.397 | Mechanism untested; would need token dropout | withdrawn | Now tested (C5). | OOV:s3_dropout, s4_gru_offset | |
| D1 | VII-A, l.403 | Better in-distribution variant ranks lower in every held-out domain, in two detector families; web AUROC preserved | revised | As A4 and R3d. Also: "under world- and task-disjoint splits the GRU does not fall below LR-C" (kept). | | R-a, R-b |
| D2 | VII-B, l.407 | Coverage must be read with tool-identity results; jointly trained behavioral weights rank worse | revised | Keep the coverage argument. Replace the second part with the OOV mismatch (tool-block dropout almost closes the gap) and the observation-length cue. | OOV; FO | |
| D3 | VII-C, l.411 | A GRU with a trained unknown-token embedding might behave differently | revised | "It does: token dropout removes the R3 deficit relative to LR-C." | OOV:s3_dropout | |
| D4 | VII-D, l.415 | Same features reach 0.79–0.86 within AgentDojo; model imbalance pushes pooled transfer below 0.5 | revised / withdrawn | Revise the range (A9). Withdraw the model-imbalance explanation (R4h). | R4D | |
| D5 | VII-E, l.419 | Recommendations for "runtime detectors"; single subsamples unstable; check thresholds under shift | revised | Replace "runtime" with "post hoc". Recommend prevalence-weighted AP. Keep "re-check thresholds under shift". Add: report content-aware baselines, per-negative-class FPR, out-of-vocabulary training (dropout), and label-free recalibration as a partial remedy. | PO; OOV | |
| L1 | VIII, l.425 | No breakdown by hard negatives or hijack pattern; DoS not analyzed | withdrawn | Now reported (M9, m7). | | |
| L2 | VIII, l.427 | No text-embedding detector; GRU mechanism untested | revised | "A TF-IDF content detector is included; no sentence encoder or released guard model." Withdraw the untested mechanism. Add: AgentDrift is nearly separable by generator templates; the observation-length cue is attacker-controlled. | CD; OOV | |
| X1 | X, l.447 | RQ1: AUPRC 0.261 at 2%, precision 0.077, wide intervals | revised | 0.232 [0.183, 0.330]; 0.076 with the pool-E mix | PO | |
| X2 | X, l.447 | RQ2: no degradation under world- or task-disjoint splitting | revised | As A3 | DS | |
| X3 | X, l.447 | RQ3: domain shift broke thresholds in every fold and reversed the order of the representations | revised | As A4 and R3c | DS; OOV | R-a, R-b |
| X4 | X, l.447 | RQ4: failure of transfer, not of the features | revised | As R4j | R4D | R-c |
| X5 | X, l.447 | RQ5: pre-registered GRU repeated the patterns; deficit tied to the tool-sequence pathway | revised | "pre-specified" GRU; the deficit comes from the untrained `<UNK>` embedding and token dropout removes it | OOV | |
| X6 | X, l.449 | Neither the rank of a detector nor its operating point carried over to held-out domains or to another benchmark | revised | "No detector transferred to AgentDojo. Within AgentDrift the content detector kept its rank, LR-B's threshold carried over in four of five domains, and the rank changes of LR-C and the GRU follow from untrained out-of-vocabulary inputs." | DS; CD; R4D | |
| X7 | X, l.449 | Why rankings reverse remains open | withdrawn | The mechanism is now measured (M1). The cause of the cross-benchmark failure remains open. | OOV | |
| P1 | Appendix | Breakdown by hard negatives and hijack pattern still outstanding | withdrawn | Delete the appendix (m21). | | |

---

# Part 3. What remains open, and how to cite

**Not done (state as limitations or future work).**
1. No sentence-encoder or released guard model (PromptGuard 2, AlignmentCheck). M2 is addressed only with TF-IDF.
2. No prefix-level evaluation (M11, optional).
3. No second benchmark pair and no reverse transfer (Tier 3).
4. The M9 breakdown is missing for R1-wd, WIT, WUT_F and LOSO (deviation P12).
5. TPR at FPR budgets is computed in R1 only (plan 3.8; deviation X2 in DEVIATIONS.md).
6. There is no GRU in LOSO, and no GRU dropout arm in R1-wd, R2 or WUT (G4).
7. `gru_audit.json` and `GRU_RESULTS.md` still contain the v0.3 errors. Correct them as text (M10).
8. No external timestamp exists for the v0.3 plans. For the revision plan, provide its SHA-256 (`PLAN_MANIFEST.txt`), and push a signed commit or register on OSF before submission if possible (M10 action 7).

**Borderline results that must be worded as borderline.**
- R1 LR-C − LR-B wAUPRC at 2%.
- R1 GRU − LR-C wAUPRC at 2%.
- WUT LR-C − LR-B wAUPRC at 10%.
- IT LR-C within minus transfer.
- R4 TFIDF − LR-C.
- R2 − R1 LR-C wAUPRC@10%.
- LR-B R1-wd seed 1.
- GRU Outcome 1.
- LR-C's R4 Case, which sits 0.003 AUROC and 0.002 wAUPRC from the bar and flips if the DoS positives are removed.

**Canonical source per manuscript table.**
- Table III: DS m12 and cache κ.
- Table IV: PO M8_R1 and R1_operating_points.
- Tables V and VI: DS M3, M4 and m5, plus OOV s3 for the dropout arms.
- Table VII: R4D M7, M6, m1, m2 and m17_3.
- Table VIII: v0.3 `gru_r3_diagnostic_results.json`.
- New tables: PO M9 (breakdown), CD (content), FO (observation length), OOV s1–s2 (mechanism).

Intervals for the same quantity can differ in the third decimal between streams, because each stream draws its own resamples. Cite one stream per table.
