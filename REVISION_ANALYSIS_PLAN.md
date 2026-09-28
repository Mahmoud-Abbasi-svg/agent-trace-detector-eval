# Revision Analysis Plan (internal pre-specification, not externally registered)

Written 26 September 2026, before any revision analysis was run. Its SHA-256 hash and modification time are recorded in `revision/PLAN_MANIFEST.txt`. This plan fixes the analysis choices for the revision of the manuscript "Evaluating Detectors of Hijacked LLM Agents Under Rare Attacks and Distribution Shift" in response to the independent review (`REVIEW_REPORT.md`).

## 0. What was already known when this plan was written

All v0.3 results (archived in `archive_v0.3/`) were known. The review panel had also run exploratory versions of several analyses below and reported approximate numbers (tool-block dropout for LR-C, token dropout for the GRU at p = 0.1, 0.2 and 0.3, a TF-IDF content detector, prevalence-reweighted AUPRC, domain-matched R1 baselines, observation-length ablations and padding, failed attacks as AgentDojo negatives, corrected AgentDojo extraction, cluster bootstraps). The analyses below are therefore confirmatory re-runs of analyses whose approximate outcome is partly known, not prospective tests. The paper will say so.

## 1. Data corrections

1. AgentDojo extraction pairs each assistant tool call with the tool messages that follow it, in order (positional pairing). Tool-call ids are not used for pairing. A unit test checks alignment on a known Gemini run and on all runs with unique non-empty ids.
2. Observations are stored without truncation in both benchmarks (the v0.3 extractor cut AgentDojo observations at 2,000 characters).
3. Unsuccessful attack runs are extracted into a separate file (`agentdojo_failed.jsonl`) with the same schema and category `attack_failed`. Exclusions are reported by class and by model. Denial-of-service variants are flagged (`attack_type` containing "dos").
4. All v0.3 scripts that read AgentDojo (`ablation.py`, `r4_control.py`, `gru_ladder.py`) are re-run unchanged on the corrected file.

## 2. Shared prediction cache

All new analyses read test-set scores from one cache (`revision/cache/`), so that every table in the revision shares the same fitted models. Detectors, all with the v0.3 configurations unless stated:

- LR-A, LR-B, LR-C, Markov, Isolation Forest (as in v0.3).
- LR-B-noobs and LR-C-noobs: representation B (or C) without the two observation-length features (maximum and mean observation length).
- LR-B-minus: representation B without the two keyword-based state-changing features.
- LR-C-drop (tool-block dropout): LR-C trained on the training set plus a copy of it whose tool block is zeroed (equivalent to dropping the tool block with probability 0.5).
- TF-IDF content detector (TFIDF): logistic regression (class-weighted, C = 1) on TF-IDF features of the concatenated text of every step (tool name, argument keys and values, observation text), word unigrams and bigrams, lowercase, min_df = 2, max_features = 50,000, sublinear tf. Sensitivity variants: TFIDF-nodigit (all digits removed), and, for R1 only, TFIDF-worldmask (strings of the trajectory's world record replaced by a placeholder).
- GRU (pre-registered configuration, token dropout p = 0) and GRU-drop (token dropout: during training each tool token is replaced by `<UNK>` independently with probability p). The primary value is p = 0.2; p = 0.1 is a sensitivity analysis. Three training seeds (0, 1, 2); everything else identical to `GRU_PREREGISTRATION.md`.

Settings: R1, R1-wd (split seeds 0 to 4), R2, R3 (five folds), R4 transfer (AgentDojo P0, with unsuccessful attacks also scored), within-AgentDojo UT and IT (five repetitions), within-AgentDojo UT with unsuccessful attacks added as training negatives, and AgentDojo leave-one-suite-out.

Thresholds: theta_r80 (20th percentile of in-sample training-positive scores) and theta = 0.5 for every detector; for the GRU also theta_val80 (20th percentile of validation-positive scores).

## 3. Estimands and statistics

1. **AUPRC at controlled prevalence (primary).** Prevalence-weighted average precision on the full pool: each positive gets weight pi/(1 - pi) * n_neg/n_pos. The interval comes from a stratified bootstrap of the full pool (B = 2,000). The v0.3 subsampling estimate is kept only as a secondary quantity, described as the expected AP of a test set with that many positives.
2. **Domain-matched R1 comparator.** R1 models scored on the R1 test pool restricted to one domain. The R3 minus R1 differences for each domain use independent bootstraps.
3. **Threshold failure.** A detected change (95% interval excludes zero) in recall or in FPR at theta_r80, relative to the domain-matched R1 value.
4. **Reversal.** Claimed for a fold only if the in-distribution advantage is detected in the domain-matched R1 comparison AND the R3 paired difference is detected with the opposite sign, both after Holm correction within the family of five folds.
5. **Cluster bootstrap.** AgentDojo resamples (suite, user task) clusters for the UT analyses; the IT analyses resample (suite, injection goal) for positives and (suite, user task) for negatives. R2 resamples tasks. AgentDrift R1 and R3 keep the trajectory-level bootstrap, and world clustering is reported as a sensitivity analysis.
6. **Multiplicity.** Holm within these families. (a) R3 per-fold C-B. (b) R3 per-fold GRU-C. (c) R3 minus domain-matched R1, per detector. (d) R4 within minus transfer, per detector. (e) Paired comparisons at each controlled prevalence. Conjunctive "all five folds" statements are reported with and without correction.
7. **Label-free threshold remedy.** On each held-out domain, set LR-C's threshold at the quantile of 200 unlabeled held-out traces that reproduces the R1 alert rate, then report FPR and recall on the remaining held-out traces.
8. **Operating points.** Also report FPR at a fixed test TPR of 0.8, and TPR at FPR budgets of 1% and 5% set on training negatives.

## 4. Additional analyses

- **Observation length.** Per-feature standardized coefficients and univariate AUROC for each benchmark. A padding test adds 0, 100, 300, 600 and 1,200 filler characters to the injection-point observation of attacked R1 test trajectories, then reports recall at theta_r80 and AUROC for LR-B, LR-C, LR-B-noobs and the GRU. We do not claim that padded injections would still succeed.
- **Keyword feature.** Precision and recall of the keyword flag against a state-change annotation of the tool vocabulary, and results without the keyword features (LR-B-minus).
- **Negative-class and hijack-pattern breakdown.** FPR per negative class (benign, hard negative, failed attack) and recall per hijack pattern (full, partial, delayed) for every detector and setting. Precision at 2% and 10% with benign-only, pool-E and hard-negative-only negatives.
- **Composition and templates in R4.** Model-balanced AUROC at native counts for every detector, including the GRU. Paired LR-A minus LR-C. Per-template transfer. R4 with and without the denial-of-service positives. The short-run filter reported by class and by model.
- **Mechanism.** For LR-C, the per-fold offset w_A^T c. Graded coverage within R1 (tool columns zeroed for a random 25%, 50%, 75% and 100% of tools; five random draws). For the GRU, the logit contribution of the tool pathway on training negatives and on held-out negatives, over all folds and seeds, with the `<UNK>` embedding norm.
- **Real-trace replication of R3.** AgentDojo leave-one-suite-out for LR-B, LR-C and LR-C-drop, reporting coverage.
- **Shift size.** A classifier two-sample test (logistic regression on representation B, 5-fold AUROC of train versus test) for each setting.
- **Diagnostics.** Restore `pre_ablation_audit.py` and write its outputs (world count, purity, overlap, probes) to a result file. Probes are regularized and use nested cross-validation, with domain-only and majority floors.

## 5. Decision rules for revised claims

- A claim of the form "X holds in all five folds" is kept only if it holds after Holm correction; otherwise the paper reports the count.
- The R3 reversal is stated only for folds meeting the definition in 3.4. It is also stated whether the reversal survives the dropout remedies at the pre-specified p.
- The R4 Case decision is re-derived per representation and per population (P0; C1; model-balanced; unsuccessful attacks as negatives), using the v0.3 criteria in `R4_CONTROL.md` Section 7, including its "mixed or other outcomes" clause.
- Any deviation from this plan is logged in `revision/DEVIATIONS.md`.
