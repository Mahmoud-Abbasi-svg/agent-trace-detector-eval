# Deviations: v0.3 plans and the revision analysis plan

Consolidated on 27 September 2026. This file has five parts.

- **Part A.** For each v0.3 plan, what was already known when it was written (file modification times).
- **Part B.** A table of deviations from the three v0.3 plans (`R4_CONTROL.md`, `PRE_ABLATION_DESIGN.md`, `GRU_PREREGISTRATION.md`). It covers the review's M10 and m16 lists, each item checked against the code, plus items found during the check.
- **Part C.** A table of deviations from `REVISION_ANALYSIS_PLAN.md`, by plan section, pointing to the stream entries.
- **Part D.** Corrections that this consolidation makes to the stream logs. These come from the independent verification of each stream.
- **Part E.** The stream logs, as each analysis stream wrote them. Where Part D corrects an entry, the entry carries a bracketed note.

"Verified" means the item was checked in the named file and line during this consolidation.

---

## Part A. What was known when each plan was written

Times are `os.path.getmtime` in local time (CEST), read on 27 September 2026, for the files in `agent-ids/` and `agent-ids/archive_v0.3/`. The archive copies keep their original times (`cp -p`). The root copies of `r4_control_results.json`, `ablation_results.json`, `gru_results.json` and `gru_audit.json` were overwritten on 26 September by the re-run on the corrected extraction, so for v0.3 the archive times are used.

**Caveat.** File times show when a file was last written, not when each part of it was written. They cannot show that a plan was not edited after results existed. Only the revision plan has an independent record: its SHA-256 and time are in `revision/PLAN_MANIFEST.txt`. None of the plans is externally registered.

| Time (2026-09-25 unless stated) | File | What it contains |
|---|---|---|
| 13:26:50 | summary.json | AgentDrift descriptives (tools per domain, category counts) |
| 13:31:35 | baseline_results.json | First sweep: LOGREG (= LR-C), Markov and IForest on R1, R2 and R3 (medical, banking). v0.1 protocol: R1 LR-C AUPRC 0.400 at 2%, AUROC 0.912 |
| 13:34:29 | sweep_results.json | Same detectors, prevalence sweep (v0.1 protocol) |
| 13:42:46 / 13:42:56 | cross_benchmark.py / cross_benchmark_results.json | **R4 transfer: LOGREG-full (LR-C) 0.440, LOGREG-agnostic (LR-B) 0.447, Markov 0.508** |
| 13:55:27 | NEXT_EXPERIMENTS.md | Work plan listing items 1 to 6 |
| 14:22:37 / 14:23:43 | evalkit.py / r4_control.py | Evaluation code and R4 control code |
| 14:26:35 | archive_v0.3/r4_control_results.json | **Within-AgentDojo UT and IT for LR-C and LR-B (0.855 / 0.798; IT 0.766 / 0.771), C1, per-model rows, model-balanced, model probe** |
| 14:28:02 / 14:29:48 | rerun_r1r3.py / r1r3_corrected.json | **LR-C (and Markov, IForest) on R1, R2 and R3 with the corrected protocol: R1 0.879, R3 AUROC and thresholds (FPR 0.50 to 0.72), R2 − R1 and R3 − R1 deltas** |
| **14:32:54** | **R4_CONTROL.md** | The plan (Part 1) and results (Part 2) in one file |
| **14:42:08** | **PRE_ABLATION_DESIGN.md** | The plan |
| 14:44:38 / 14:47:34 | ablation.py / archive_v0.3/ablation_results.json | **LR-A, LR-B and LR-C on R1, R3 (C < B in all five folds) and R4** |
| 14:53:13 / 14:53:36 | world_disjoint_r1.py / archive_v0.3/world_disjoint_r1_results.json | R1-wd, linear detectors, five split seeds |
| 14:54:22 / 14:54:30 | ABLATION_RESULTS.md / FINDINGS.md | Ablation write-up |
| **15:02:47** | **GRU_PREREGISTRATION.md** | The plan |
| 15:05:05 / 15:11:42 | gru_detector.py / gru_ladder.py | GRU code |
| 15:12:08 / 15:34:18 | archive_v0.3/gru_audit.json / archive_v0.3/gru_results.json | GRU split audit and results |
| 15:41:46 | GRU_RESULTS.md | GRU write-up |
| 15:44:03 / 16:02:26 | gru_r3_diagnostic.py / archive_v0.3/gru_r3_diagnostic_results.json | GRU-A / GRU-B / GRU-C diagnostic (after the GRU results) |
| 2026-09-26 16:12:06 | MANUSCRIPT_DRAFT.md (v0.3) | Manuscript |
| 2026-09-26 18:24:33 | REVIEW_REPORT.md | Independent review, with the panel's exploratory numbers |
| **2026-09-26 18:30:32** | **REVISION_ANALYSIS_PLAN.md** | Revision plan. SHA-256 `14bed377…` recorded at 18:30:33, when 0 files existed in `revision/cache` |
| 2026-09-26 18:31:16 to 18:57:58 | agentdojo_extract.py, agentdojo_traces.jsonl, agentdojo_failed.jsonl; re-run ablation_results.json, r4_control_results.json, gru_results.json | Corrected extraction and v0.3 re-run (after the revision plan) |

### R4_CONTROL.md (last written 14:32:54)

**Already known when the file was last written.**
- R4 transfer for LR-C (0.440) and LR-B (0.447) (cross_benchmark_results.json, 13:42:56). Part 2 line 87 says the transfer run "reproduces the earlier R4 values exactly".
- The complete within-AgentDojo results (r4_control_results.json, 14:26:35).
- LR-C R1 to R3 (r1r3_corrected.json, 14:29:48).

**What is claimed, and what the file times allow.** Line 3 claims that Part 1 was written before any within-AgentDojo result. The file time cannot confirm or refute this, because Part 2 was appended afterwards.

**What can be established.** The near-chance rule "or AUROC < 0.60" (line 61) and the Case rules were fixed knowing that transfer was 0.440, with an interval that excludes 0.5.

**Prospective at most.** The within-AgentDojo results, if Part 1 predates 14:26:35.

### PRE_ABLATION_DESIGN.md (14:42:08)

**Already known.**
- LR-C on R1, R2 and R3 (AUROC, AUPRC and thresholds; r1r3_corrected.json).
- LR-C and LR-B in R4: transfer 0.440 / 0.447, within-UT 0.855 / 0.798, and IT (r4_control_results.json).
- The plan lists the R4 values and LR-C R1 0.879 as "reproduction anchors" (line 93).

**Rule 4 (line 79).** It names R4 as the primary evidence for Outcome 4, although both detectors' R4 values were already known.

**Not yet known.**
- LR-A in every setting.
- LR-B in R1 and R3.
- Hence the R3 C-versus-B comparison. The linear ablation file is dated 14:47:34.

### GRU_PREREGISTRATION.md (15:02:47)

**Already known.** All linear results:
- R1, R3 and R4 for LR-A, LR-B and LR-C, including C < B in all five R3 folds (ablation_results.json, 14:47:34);
- R1-wd (14:53:36);
- the R4 control (14:26:35).

**Not yet known.** Every GRU result. The GRU code (15:05:05) and results (15:34:18) came after the plan.

**Written after the GRU results.** The GRU R3 diagnostic (gru_r3_diagnostic.py at 15:44:03, results at 16:02:26). It is post hoc.

### REVISION_ANALYSIS_PLAN.md (2026-09-26 18:30:32)

**Already known.**
- All v0.3 results.
- The review panel's exploratory numbers for most planned analyses. Plan Section 0 lists them: tool-block dropout, token dropout at p = 0.1, 0.2 and 0.3, TF-IDF, reweighted AUPRC, domain-matched R1, observation-length ablations and padding, failed attacks as negatives, the corrected extraction, and cluster bootstraps.

**Not yet computed.** No revision output existed (PLAN_MANIFEST.txt). The corrected extraction and the re-run of the v0.3 scripts come after the plan.

**Status.** The revision analyses are therefore confirmatory re-runs, not prospective tests.

---

## Part B. Deviations from the three v0.3 plans

"Verified" gives the file and line checked in this consolidation. "Where now reported" points to RESULTS_SUMMARY.md.

| ID | Plan (file:line) | Plan text (short) | What was executed | Verified | Impact | Where now reported |
|---|---|---|---|---|---|---|
| **R4_CONTROL.md** |  |  |  |  |  |  |
| V1 | R4_CONTROL.md:3 | "Part 1 ... was written before any within-AgentDojo result was computed" | The file's last write (14:32:54) is after r4_control.py (14:23:43) and its results (14:26:35). Part 2 was appended to the same file. | file times (Part A) | Part 1's timing cannot be verified. No independent record (commit, hash) exists. | M10 |
| V2 | R4_CONTROL.md:61 | "Near chance means the AUROC CI includes 0.5, or AUROC < 0.60" | The one-sided "< 0.60" clause was written with transfer 0.440 [0.407, 0.471] known (cross_benchmark_results.json 13:42:56; Part 2 line 87). | R4_CONTROL.md:87; cross_benchmark_results.json | It is post hoc. On corrected data it alone decides "near chance" for LR-A (all populations) and TF-IDF (C1) (r4_deep R4D4). | M6 |
| V3 | R4_CONTROL.md:65 | "Mixed or other outcomes are reported as observed ... This includes UT reasonable but IT not reasonable" | R4_CONTROL.md Part 2 (line 172) reports IT as a "mixed element". The manuscript (IV-F, l.168) omits the clause, and VI-F (l.334) states "The pre-registered conditions for Case A are met". | MANUSCRIPT_DRAFT.md:168, 334 | The v0.3 Case A verdict misstates the rule. On corrected data, LR-C is mixed and GRU is "Case A under UT; mixed under IT". | M6 |
| V4 | R4_CONTROL.md:62-64 | Case rules per population. Both representations were run (lines 28-29). | Case A was declared for LR-C (full) only (Part 2 lines 166-170). LR-B (agnostic, 0.798) is noted as just below the bar (line 174). The manuscript generalizes the verdict to "the features". | R4_CONTROL.md:166-179; MANUSCRIPT_DRAFT.md:334, 368, 415 | Verdict overgeneralized. Revision: per-representation decisions. | M6 |
| V5 | R4_CONTROL.md:10 | "Failed attacks ... were already excluded at extraction" | As planned. It is inconsistent with the AgentDrift training construction, where failed attacks count as negatives (review M6(c)). | agentdojo_extract.py (v0.3) | A design choice, not a deviation. The revision adds P0F and WUT_F. | M6 |
| V6 | R4_CONTROL.md:13 | Within-model control: "AUROC and AUPRC computed within each model ... and their macro-average" | Per-model AUROC and AUPRC reported; the macro-average only for AUROC (r4_control.py:148). | r4_control.py:139-148 | Minor. | none |
| V7 | R4_CONTROL.md:16 | Model-identity probe: 5-fold CV LR on benign traces | Implemented as planned (unregularized, C = 1, no nesting). | r4_control.py:176-194 | No deviation. The probe is weak and its B signal (0.615) came from the extraction bug (review m15). | m15 |
| **PRE_ABLATION_DESIGN.md** |  |  |  |  |  |  |
| V8 | PRE_ABLATION_DESIGN.md:87, 90, 92 | "run the planned A/B/B−/C ablation"; report paired "C−B−" | The B− arm and the C−B− contrast were not run. ablation.py:13 has `REPS = ["A", "B", "C"]`; the paired contrasts are C−A, C−B and B−A (ablation.py:77). ABLATION_RESULTS.md:166 calls them "not triggered", but only A+UNK was optional (line 100). | ablation.py:13, 77; ABLATION_RESULTS.md:166 | None on conclusions. Revision LR-B-minus: R1 0.827; R3 0.803 / 0.839 / 0.899 / 0.807 / 0.849 (cache/linear/CACHE_README.md). | M10 |
| V9 | PRE_ABLATION_DESIGN.md:73 | Rule 2: "B degrades ... if the independent-bootstrap ΔAUROC ... R1→shift is detected as negative" | No R1→R3 independent-bootstrap delta was computed for B (ablation.py has none). The manuscript compares LR-B R3 values (0.792 to 0.896) with 0.830 descriptively. | ablation.py:61-78 | The rule was not applied. Revision: LR-B R3 − R1dm is detected in no fold (−0.012 to +0.022). | M3 |
| V10 | PRE_ABLATION_DESIGN.md:79 | Rule 4: degradation difference, "primary evidence for Outcome 4 is R4" | Not computed and not reported. | ablation.py (no such test); MANUSCRIPT_DRAFT.md | Revision: R1-referenced degradation difference +0.057 [+0.042, +0.073] (stratified, as planned). Outcome 4's R4 clause is met, but the test is weak. | M1 |
| V11 | PRE_ABLATION_DESIGN.md:91 | R4 conditions: "transfer on P0 / C1 / model-balanced, and the within-AgentDojo UT and IT references" | Within-UT only, per representation (ablation.py:103-116). IT exists only for LR-B and LR-C, from r4_control.py. LR-A IT was never computed in v0.3. Model-balanced at 10% only (ablation.py:126-151). | ablation.py:103-151 | LR-A IT missing; model-balanced not at native counts. Revision: LR-A WIT 0.690 (r4_deep M7; R4D14); native-count model-balanced for all detectors (m1). | M6, m1 |
| V12 | PRE_ABLATION_DESIGN.md:19-22 | 1,029 worlds; largest 194/169/144; purity 0.858 | The audit script that produced these (pre_ablation_audit.py) was truncated mid-line at line 44 and did not parse. | revision/pre_ablation_audit_v03_truncated.py.txt | The numbers do not reproduce on the pool. The rewritten audit gives 1,133 worlds and purity 0.883. One key on train+val reproduces 1,029 and 0.858 but not the largest sizes (Part D, X5). | m5 |
| V13 | PRE_ABLATION_DESIGN.md:29, 36 | D1 and D2 probes "3-fold CV" | ABLATION_RESULTS.md:83 (Part 1) and the audit docstring say 5-fold. The v0.3 probe code is lost. | ABLATION_RESULTS.md:83 | The specification is inconsistent. Revision: nested regularized probes. An unregularized 3-fold reconstruction reproduces the domain values (1.000 / 0.455 / 0.381). | m15 |
| V14 | ABLATION_RESULTS.md:86 (the Part 1 design that PRE_ABLATION_DESIGN.md:3 points to) | D4 dataset-identity probe (AgentDrift vs AgentDojo, benign only) | Not run and not reported. | ABLATION_RESULTS.md:86 | Not run in the revision either. The m12 two-sample test on representation B (AgentDrift train vs AgentDojo test, all rows) gives AUROC 0.992 [0.991, 0.993], which answers the same question in B. | m12 |
| **GRU_PREREGISTRATION.md** |  |  |  |  |  |  |
| V15 | GRU_PREREGISTRATION.md:28 | R1 early stopping on `data/val.jsonl` (1,880) | A stratified 10% of the R1 training data (gru_ladder.py:17-23, 50, 108). gru_audit.json:124 (written at gru_ladder.py:73) and GRU_RESULTS.md:11 still record "data/val.jsonl (1880)". | gru_ladder.py:50, 73, 108; gru_audit.json:124 | No effect on the reported numbers. The records are wrong. The revision cache follows the code (G1). | M10 |
| V16 | GRU_PREREGISTRATION.md:29, 70 | Every (rung, split) trained with 3 seeds; within-AgentDojo "5 folds × {P0, C1, model-balanced} × 3 seeds" | Within AgentDojo, one training seed per fold assignment, seeds 0 to 4 (gru_ladder.py:136-153). The manuscript (l.226, l.439) says seeds 0, 1, 2. | gru_ladder.py:136-153 | Within-AgentDojo seed variance is not separable from fold-assignment variance. The revision follows the code (G2, R4D1). | m6, M10 |
| V17 | GRU_PREREGISTRATION.md:36 | R1-wd: 5 split seeds × 3 training seeds | Split seed 0 only (gru_ladder.py:116-118). | gru_ladder.py:116-118 | Revision cache: all five split seeds × 3 training seeds (GRU pool-E AUROC 0.949 to 0.962). | m5 |
| V18 | GRU_PREREGISTRATION.md:55 | Independent-bootstrap ΔAUROC of each shift rung against its own R1 | Not computed. GRU_RESULTS.md:111 uses non-overlap of the R3 and R1 intervals. | gru_ladder.py (no delta); GRU_RESULTS.md:111 | Valid and conservative. Revision: GRU R3 − R1dm detected in 5/5 folds (Holm). | M3 |
| V19 | GRU_PREREGISTRATION.md:39 | R4 on P0, C1 and model-balanced "exactly as in r4_control.py" (which includes native counts) | Model-balanced GRU only at 10% prevalence (gru_ladder.py:166-178). | gru_ladder.py:166-178 | On corrected data, native-count model-balanced transfer is 0.536 [0.501, 0.569] (stratified), so Outcome 1 is missed by 0.0007, or met under the cluster interval [0.487, 0.575] (m1). | m1, m16 |
| V20 | GRU_PREREGISTRATION.md:61 | Outcome 1: near-chance R4 "on P0 and model-balanced" | Assessed on the 10% subsample only (GRU_RESULTS.md:112). | GRU_RESULTS.md:112 | At the boundary (V19). | m16 |
| V21 | GRU_PREREGISTRATION.md:64 | Outcome 4: "large seed variance → report the inconsistency and the seed variance" | GRU_RESULTS.md:120 dismisses Outcome 4 ("tight CIs, no sign flips"). The manuscript (l.376) reports seed-dependent thresholds, yet l.378 says the outcomes "match the pattern we specified in advance". | GRU_RESULTS.md:120; MANUSCRIPT_DRAFT.md:376-378 | Revision (post hoc criteria C1 to C4, R4D5): Outcome 4 is triggered on the operating point (C3 in 5/5 folds). | m16 |
| V22 | GRU_PREREGISTRATION.md:24-31, 73 | The training procedure is fixed; "the only fixed values are those in §2/§4" | **Not in the review.** The code clips the gradient norm at 1.0 (gru_detector.py:83). The plan does not list it; the manuscript does (V-E). | gru_detector.py:83 | An undisclosed addition to the planned procedure. It applies to every v0.3 and revision GRU fit, because the revision imports gru_detector.py unchanged. Its effect is not measured. | M10 |
| V23 | GRU_PREREGISTRATION.md:19 | Optional secondary: GRU-tool-only only, "if needed" | The diagnostic added GRU-B, a linear head on the ten behavioural features with no recurrence, which the plan does not name. It ran after the GRU results (15:44 against 15:34). | gru_r3_diagnostic.py; GRU_R3_DIAGNOSTIC.md:18 | Post hoc. Report the diagnostic as exploratory. | M10 |
| V24 | GRU_PREREGISTRATION.md:77 | "Any deviation ... will be recorded in the results file and here" | GRU_RESULTS.md:3 says the plan was implemented "exactly". No deviation is recorded. | GRU_RESULTS.md:3 | Text correction. | M10 |
| V25 | GRU_PREREGISTRATION.md:12, 22 | Vocabulary = benign-training tool names | The code matches the plan (gru_detector.py:18-20). | gru_detector.py:18-20 | No plan deviation. The manuscript (l.120) misstates it as "any training trajectory" (m6). | m6 |

---

## Part C. Deviations from REVISION_ANALYSIS_PLAN.md

Each row gives the plan item, how it was executed, and the log entries (Part D or Part E) that explain the difference. "Interpretation" means the plan was silent or ambiguous and a choice was made. "Restriction" means less was done than the plan asks. "Addition" means more was done. "Post hoc" means the choice was made after seeing results.

| Plan item | Status | Deviation or interpretation | Log IDs | Impact on conclusions |
|---|---|---|---|---|
| §1.1 positional pairing + unit test | done | None. The alignment test passes (3 passed). | D (m14) | none |
| §1.2 no truncation | done | Observations are uncut in both benchmarks. The review's corrected estimates still used the 2,000-character cut. | L5 | Explains why corrected within-AgentDojo values (0.797) are below the review's (0.846). |
| §1.3 failed attacks, exclusions, DoS flag | done | None. | R4D11 | none |
| §1.4 re-run the v0.3 scripts unchanged | done | Final ("gru_ladder exit 0"). The cache reproduces the re-run points to ≤ 5e-5, which is 4-decimal rounding (this consolidation, `RESULTS_SUMMARY_values.json` rerun_v03_vs_cache; r4_deep M7 reconciliation). | none | none |
| §2 shared cache, detectors | done, with restrictions | The cache makes tied scores exact (L1). TFIDF-worldmask semantics (L3). GRU-drop is not fitted in R1-wd, R2 or WUT; there is no GRU in WIT, WUT_F or LOSO in the cache (G4). r4_deep later fitted the WIT and WUT_F GRUs with one seed per repetition (R4D1). | L1, L3, G4, R4D1 | L1 moves R4 Markov AUROC 0.508 → 0.590 and R3 LR-A 0.499 → 0.500; no conclusion depends on it. There is no GRU LOSO replication. |
| §2 GRU seeds, validation | as v0.3 code | R1 validation is a stratified 10% of train, not val.jsonl (G1, = V15). Within AgentDojo, one seed per repetition (G2, = V16). Gradient clipping as in gru_detector.py (V22). | G1, G2, V22 | none beyond V15, V16 and V22 |
| §2 added detectors | addition | TFIDF-toolname, TFIDF-anon, TFIDF-anon-nodigit (C3); LR-obsonly, post hoc (F3); LR-B-annot (m8); GRU fits without `<UNK>` and without DoS (R4D1). | C3, F3, R4D1 | Sensitivity analyses only |
| §3.1 weighted AUPRC primary | done | Subsampling is kept as secondary (PO). It is not reported in the content stream (C4). Scores were rounded to 12 decimals, which merges saturated AgentDojo scores, so unrounded values are also given (P10). This consolidation quotes the unrounded AgentDojo values. | P6, P10, C4, X11 | Changes by up to 0.0056. It flips LR-C's R4_CONTROL AUPRC criterion (0.2985 weighted against 0.322 subsampled). |
| §3.2 domain-matched R1 | done | None. The R1dm and R3 pools share 345 trajectories per domain (276 for banking), so the independent-bootstrap intervals are conservative (m3; O11). | O11 | Conservative intervals; no decision changes under a joint bootstrap. |
| §3.3 threshold failure | done | Headline counts use Holm. The plan's definition is unadjusted, and both counts are reported. They differ only for the pooled GRU (3 against 4) and LR-B-noobs (1 against 2). | DS2 | none |
| §3.4 reversal rule | done | Interpretation: AUROC is primary and AUPRC@10% secondary (DS3). The rule is also applied to the dropout pairs. The GRU seed estimand has three versions (O2, s7). | DS3, O2 | Reversal folds are identical under all three GRU estimands. |
| §3.5 cluster bootstrap | done, extended | Interpretation: the cluster bootstrap is also used for R4 transfer and LOSO, which the plan does not name (C1, F1, O7). The WUT repetition is shared across detectors (O6, C2). The R1 world-cluster sensitivity is in PO and DS. R3 world-cluster is in DS only. | C1, C2, F1, O6, O7, DS5, P1 | Borderline results depend on the resampling unit (for example R4 TFIDF − LR-C, X7). |
| §3.6 Holm | done | Interpretation: bootstrap p-values, with different floors across streams (X8). Family (d) is read per configuration (R4D15), over content detectors only (C5), or over five detectors (F2); see X9. | DS2, O3, C5, F2, R4D15, X8, X9 | A verifier's recomputation with a per-detector family across configurations changed no decision. |
| §3.7 label-free threshold | done + post hoc | The pre-specified reading (a) and a post hoc reading (b) are both reported. The "R1 alert rate" had to be interpreted (P4). | P4, P9 | Reading (b), not (a), matches the review's figures. |
| §3.8 FPR at fixed TPR; TPR at FPR budgets | done, with a restriction | FPR at test TPR 0.8 is descriptive, because it uses test labels (P7, C6). TPR at 1% / 5% budgets is reported for R1 only, for all detectors including TF-IDF (P2, P3). The other settings are not covered (X2). | P2, P3, P7, C4, C6, X2 | m10 is answered for R1 only. |
| §4 observation length | done | Padding specifics (F5). The GRU is padded in R1 only. Only lengthening was tested. | F4, F5, F6, F7, F9 | The within-AgentDojo padding magnitude reported by the review is not reproduced. |
| §4 keyword feature | done | One non-blind annotator (F8). | F8 | Annotation-based features change LR-B by at most 0.024. |
| §4 negative-class and pattern breakdown "for every detector and setting" | restriction | Done for every cached detector in R1, R2, the five R3 folds, R4 and WUT. Not for R1-wd, WIT, WUT_F or LOSO (X1 = P12). Partly covered elsewhere: TF-IDF per-class rates in every setting (CD), and failed-attack flag rates in WIT and WUT_F (R4D). | P5, P8, P11, X1 | The missing settings are sensitivity settings. No headline depends on them. |
| §4 composition and templates in R4 | done | Case-rule interpretations (R4D2 to R4D4). Exploratory attempt target (R4D6). Q2 design (R4D12). v0.3 values that no v0.3 script produced (R4D14). | R4D2 to R4D14 | The LR-C Case is at the boundary (0.003 AUROC). |
| §4 mechanism | done | Graded coverage masks tool names, not domains (O4). The R1 GRU was refitted (O5). "Training negatives" include failed attacks (O8). | O4, O5, O8 | none |
| §4 LOSO | done | Linear detectors only; no GRU (O10). | O10 | none |
| §4 shift size (C2ST) | done, extended | Specification (DS6). Extra settings (DS11). | DS6, DS11 | none |
| §4 diagnostics | done | The audit script was rewritten (D1). Probe design (D2). Additions (D3). Exploratory key search (D4, corrected by X5). README definitions (D5). | D1 to D6, X5 | none |
| §5 decision rules | applied | Applied literally in RESULTS_SUMMARY.md Part 2. The Rule-4 estimand is corrected (X4). | X4 | Wording only |
| (review, outside the plan) M2 sentence encoder and released guard model | not done | Not in plan §2. The review asks for them. | X16 | M2 action 1 is only partly met. |

---

## Part D. Consolidation corrections and additions (logged 2026-09-27)

These entries come from the independent verification of each stream. They take precedence over the stream text in Part E.

- **X1 (= P12, prevalence stream; restriction, not previously logged).**
  - *Plan.* Plan §4 asks for the negative-class and hijack-pattern breakdown "for every detector and setting". `prevalence_ops.py` (M9_SETTINGS, line 51) computes it for R1, R2, the five R3 folds, R4 and WUT only.
  - *Missing.* R1wd_s0 to s4, WIT, WUT_F and LOSO_* have no per-class FPR or pattern recall for all detectors. The caches for them exist, so this is a scope restriction, not a data limitation.
  - *Partial coverage elsewhere.* content_detector_results.json `settings.<S>.per_class_t_r80` for TF-IDF, TF-IDF-nodigit and LR-C in every setting; r4_deep_results.json `M6.failed_flag_rates.by_setting` for WIT and WUT_F.
  - *Impact.* No headline number depends on the missing settings.
- **X2 (plan 3.8; restriction).** TPR at FPR budgets of 1% and 5%, set on training negatives, is reported for R1 only (prevalence_ops `R1_operating_points.m10_fpr_budgets`, all detectors including TF-IDF), not for R2, R3, R4 or within AgentDojo. Stream entry C4 says the text detectors were not covered; that applies to the content stream only, and TF-IDF is covered in R1 by the prevalence stream.
- **X3 (plan §2; restriction).** The same as G4, restated. There is no GRU in LOSO, and no GRU dropout arm in R1-wd, R2 or WUT.
- **X4 (Rule 4 estimand; correction to the OOV stream's reading of O7 and O9).**
  - *The pre-registered estimand.* PRE_ABLATION_DESIGN.md lines 73 and 79 define degradation "from R1 → shift" with an independent bootstrap. The estimand is therefore the R1-referenced degradation difference (R1_C − R4_C) − (R1_B − R4_B): +0.057 [+0.042, +0.073], stratified (oov_mechanism_results.json `s6_rule4.R4.P0.stratified.DD_R1_ref`).
  - *The value mislabelled as pre-registered.* The within-AgentDojo-referenced value (+0.064 [+0.022, +0.101] stratified; +0.064 [+0.005, +0.120] cluster) is supplementary. The OOV summary and its key findings call it "as pre-registered" and present it as the correction of the review's +0.056. The review's figure was R1-referenced, and its corrected counterpart is +0.057.
  - *Impact.* The outcome is the same (Rule 4's R4 clause is met).
- **X5 (correction to D4).** The key search found one joint match that the stream summary understates.
  - *The match.* Population train+val (10,655 trajectories), key user+date+contacts[names_sorted], binary label. It gives exactly 1,029 worlds, purity 0.8575 (0.858 after rounding), 54 worlds with ≥ 30 trajectories, and a majority baseline of 0.558 (diagnostics_results.json `m5_key_search.rows_n_worlds_1029[2]`).
  - *What it does not reproduce.* The largest sizes: 231/209/155 against 194/169/144.
  - *Reading.* The v0.3 diagnostics were probably computed on the wrong population (train+val instead of train+test) with a coarser key. This is a lead, not proof. The recommendation stays the same: report 1,133 worlds and purity 0.883 [0.860, 0.909].
  - *Related correction to D5.* The README's 28.9% ext-audit share is approximately reproduced by the second definition: 0.281 [0.270, 0.291] of 7,036 poisoned observations. Only the 75% hard-negative figure is not reproduced.
- **X6 (m10 wording; correction to the prevalence stream's reading).**
  - *Correction.* "Test FPR exceeds the budget because training scores are in-sample" is supported only for the GRU (test FPR 0.070 [0.050, 0.092] at a 1% budget) and TF-IDF (0.033 [0.021, 0.045]). For LR-C (0.017 [0.008, 0.027] at 1%; 0.065 [0.048, 0.082] at 5%) and the other linear detectors, no exceedance is detected. The causal attribution to in-sample scoring was not tested.
  - *Related wording fixes from the same verification.*
    - "Every trained detector flags hard negatives more often than benign traces" holds in R1 for LR-A, LR-B, LR-C, Markov and the GRU, not for IForest.
    - LR-C's advantage over LR-B is detected against both negative classes and on full hijacks. It is not detected on partial or delayed hijacks.
    - R4 wAUPRC@10% is near chance for every detector except LR-A, which is slightly above chance.
    - IForest is "close to chance", not "at chance".
    - LR-B's label-free FPR is higher than its θr80 FPR in four of five folds.
    - WUT TF-IDF detects injected content to a large extent, but still ranks compromised traces above unsuccessful attacks (AUROC 0.808).
    - The 0.0225 maximum precision at 0.1% is selected after the fact and rests on about 5 false alarms.
- **X7 (content stream).**
  - R4 TFIDF − LR-C AUROC, +0.047 [−0.002, +0.096], p = 0.057 under cluster resampling, is borderline: it is detected under trajectory resampling (p = 0.002), and the decision depends on the Monte Carlo seed.
  - The "best single-token rule" test AUROC of 0.780 is the maximum over ten training-selected rules. The rule ranked first on training data gives 0.753.
- **X8 (p-value conventions).**
  - *Floors.* domain_shift, oov_mechanism and features_obslen use add-one bootstrap p-values, with a floor of about 1/(B+1). content_detector and r4_deep report p = 0 when all replicates fall on one side.
  - *Holm decisions.* No Holm decision changes, because 5 × 0.0005 < 0.05.
  - *Detection criteria.* "Detected" (the percentile interval) and the Holm p-value can disagree by one bootstrap tick at the boundary. Such cells are listed in the domain-shift Monte Carlo stability section.
- **X9 (Holm family d).** Streams read family (d) differently: per configuration over five representations (R4D15), over the content detectors only (C5), or over the five detectors of one table (F2). A verifier recomputed Holm per detector across the 10 non-exploratory R4 configurations and found no decision change. LR-C under IT is borderline in every reading (Holm p 0.051 per configuration).
- **X10 (r4_deep rounding).** The r4_deep summary rounds 4-decimal JSON values again to 3 decimals, so a few printed values differ by 0.001 from correctly rounded raw values. For example, LR-A transfer is 0.56545, printed 0.566. RESULTS_SUMMARY.md quotes boundary values to 4 decimals (0.5655, 0.2985).
- **X11 (AgentDojo rounded against unrounded scores).** Following P10, RESULTS_SUMMARY.md quotes the unrounded AgentDojo rank values, for example WUT LR-C wAUPRC@10% 0.299 [0.246, 0.388] and R4 LR-C AUROC 0.473.
- **X12 (r4_deep delta centres).** Paired-delta centres in r4_deep are bootstrap means, not point differences. They can differ in the third decimal between tables that use different method sets.
- **X13 (OOV stream sensitivity).** The plan 3.5 world-cluster sensitivity is not run in the OOV stream. The domain-shift stream runs it for the same R1dm and R3 contrasts and finds the same reversal folds.
- **X14 (= V22).** Gradient clipping at norm 1.0 (gru_detector.py:83) is part of every GRU fit in v0.3 and in the revision. It is not in GRU_PREREGISTRATION.md. It should be disclosed as a deviation from that plan.
- **X15 (OOV stream wording).**
  - "Alert rates fall instead" (LOSO) holds for the two zero-coverage suites (banking, slack), where the negative offset applies to every row. It does not hold in general.
  - The s2 graded-coverage GRU intervals are seed-averaged, and so they differ from the s3 mixture intervals (for example, R1 GRU FPR [0.034, 0.057] against [0.028, 0.062]).
- **X16 (review request outside the plan).** No frozen sentence encoder and no released component (PromptGuard 2, AlignmentCheck) was run. M2 action 1 is only partly addressed, with TF-IDF.
- **X17 (features stream wording).**
  - The within-AgentDojo effect of the 2,000-character cut (LR-C 0.846 with the cut, 0.797 without) is a descriptive point difference with no paired interval.
  - Shortened benign-step observations also occur in hard negatives (median 73 against 98), not only in attacked trajectories.
  - The M5.2 headline should give both metrics: LR-C-noobs < LR-B-noobs in 5/5 folds on wAUPRC@10% (the plan's primary metric) and in 4/5 on AUROC.
- **X18 (r4_deep GRU Outcome 1).** The model-balanced stratified lower bound, 0.5007, lies within bootstrap Monte Carlo error of 0.5. The results file holds two stratified intervals for the same quantity, [0.501, 0.569] and [0.501, 0.571], which differ only in method order. Report Outcome 1 as indeterminate at the boundary, and cite `m1.GRU_outcome1`.

---

## Part E. Stream logs (as written by each analysis stream)


### GRU prediction cache (cache/gru; build_gru_cache.py, gru_variants.py, summarize_gru_cache.py) - logged 2026-09-26

G1. R1 validation source. GRU_PREREGISTRATION.md Sec. 4 says R1 early-stops on data/val.jsonl. The v0.3 implementation
    (gru_ladder.py, which produced the published R1 GRU AUROC 0.936) used gru_ladder.stratified_val, a stratified 10% of
    data/train.jsonl, for R1 as for every other rung. The revision cache follows gru_ladder.py (as instructed, and so that
    the v0.3 anchor is reproduced exactly: 0.9356). This deviation already existed in v0.3; the manuscript should state it.
G2. Within-AgentDojo GRU seeds. The preregistration trains every (rung, split) with 3 training seeds. gru_ladder.py trains
    the within-AgentDojo GRU once per fold-assignment repetition (repetitions 0-4; training seed = fold seed = repetition).
    The WUT cache copies gru_ladder.py (25 fits). Pre-existing in v0.3 (review m6 asks for the seeds to be stated).
G3. WUT fold of failed attacks without a P0 group. 1 suite|user_task group (1 failed row) has no P0 row. It is assigned
    to a fold with default_rng(10000 + repetition), the same rule as the linear cache. The plan does not specify this;
    since training uses P0 rows only, every one of the five fold models is out-of-fold for that row.
G4. Scope. The GRU is fitted in R1, R1wd_s0-s4, R2, R3 (5 folds), R4 and WUT; GRU-drop0.2 and GRU-drop0.1 in R1, R3 and
    R4 only. No GRU is fitted in WIT, WUT_F or LOSO, and no dropout arm in R1-wd, R2 or WUT. If Sec. 2 of the plan is read
    as "every detector in every setting", this is a restriction.
G5. Summary intervals. gru_cache_results.json reports the v0.3 trajectory-level stratified bootstrap (evalkit.bootstrap)
    with B = 2000 in every setting (the preregistration used B = 1000 for R1-R3), plus a labelled cluster-bootstrap
    sensitivity for R2 (tasks), R4 and WUT (suite|user_task). The plan's primary estimands (Sec. 3) are computed by the
    analysis scripts from the cache, not in this summary.

### Linear prediction cache (cache/linear; build_linear_cache.py, validate_linear_cache.py) - logged 2026-09-26

L1. Exact ties. Fitting is exactly the v0.3 recipe (ablation.py, rerun_r1r3.py), but the cache stores tie-exact scores:
    LR-*, IForest and TFIDF* are scored once per distinct feature row (or text) and broadcast, and Markov scores are
    rounded to 12 decimals. Without this, float noise breaks mathematically tied scores in a length-dependent order.
    Scores differ from the v0.3 code paths by at most 3.9e-08 and all thresholded predictions agree, but AUROC can
    move where many rows are tied (low coverage): R4 Markov 0.5082 (v0.3 code) vs 0.5902 (cache); LOSO_slack Markov
    0.6636 vs 0.5000; LOSO_travel Markov 0.4237 vs 0.5498; R3 LR-A about 0.499 vs 0.500 (cache/linear/tie_check.json).
    The plan does not address ties. None of the 16 requested v0.3 anchors changes; archived R3 LR-A / Markov values
    move by at most 0.0011 (for example 0.4989 -> 0.5000).
L2. Folds of unsuccessful AgentDojo attacks without a P0 group (WUT, WUT_F, WIT). Failed attacks use their suite|user_task
    group. Groups that no P0 row defines are assigned with default_rng(10000 + seed), position mod 5: 1 group
    (1 row) in WUT/WUT_F, 11 groups (64 rows) in WIT (P0 attacks there are grouped by injection goal, so a user task
    with no benign P0 run has no P0 group). Same rule as the GRU cache (G3).
L3. TFIDF-worldmask semantics. Whole world string values (user, email, company, contact name/email/relation, date) are
    replaced case-insensitively as whole occurrences; partial names are not. A first/last-name token of the user or a
    contact survives in 0.198 of R1 trajectories (validation.json settings.R1).
L4. Validation intervals (validation.json, CACHE_README.md) are v0.3-style trajectory-level stratified bootstraps
    (B = 1000 AgentDrift, 2000 AgentDojo); the plan's cluster bootstraps and prevalence-weighted AUPRC are left to the
    analysis scripts.
L5. Scope note. The R4 LR-C/LR-B values differ from the review's corrected estimates only because the corrected extraction
    also removes the 2,000-character truncation (plan item 1.2); re-applying the truncation reproduces the review's
    values to 3 decimals (0.476, 0.483, 0.846, 0.773; cache/linear/r4_truncation_check.json). Not a deviation;
    recorded so that the change is not misread.

### Domain shift analysis (domain_shift.py; review M3, M4, m2, m3, m5, m12, m13) - logged 2026-09-26

DS1. GRU seeds. The plan does not say how the three GRU training seeds enter intervals. Following evalkit.bootstrap (the
    v0.3 convention used by the GRU cache summary), each bootstrap replicate uses one randomly drawn seed and the point
    estimate is the mean over seeds. Where the seeds disagree strongly (R3 banking and coding thresholds), this gives
    very wide intervals, so per-seed rows (GRU#seedk) are also reported for the threshold analysis.
DS2. Holm p values. The plan asks for Holm correction but does not say how to get p values from a bootstrap. We use the
    two-sided percentile-bootstrap p = min(1, 2*min(1+#{d<=0}, 1+#{d>=0})/(B+1)). 'Detected' (unadjusted) keeps the plan 3.3
    definition (95% percentile interval excludes 0). The two criteria can disagree only at the boundary.
DS3. Reversal metric. Plan 3.4 does not name the metric. AUROC is primary because the manuscript's reversal claim is
    about AUROC; AUPRC@10% is reported as secondary. The rule is also applied to the dropout pairs (LR-C-drop vs LR-B,
    GRU-drop0.2 vs LR-C), together with the plan 5 question of whether the original reversal survives the remedy.
DS4. m5 multiplicity. For the claim of no loss, detection without correction is the primary criterion because it is the
    stricter test for that claim; Holm over split seeds (pooled) and over domains (domain-matched) is reported as
    secondary. A macro (mean of per-domain) AUROC is added to remove the domain-mix change.
DS5. m2. The task-cluster bootstrap resamples the 35 R2 test tasks with replacement (not stratified by class; every task
    contains both classes). The R1 world-cluster sensitivity resamples the 676 pool-E worlds. AUPRC@10% weights are
    recomputed per replicate from its own class counts.
DS6. m12. Classifier two-sample test: StandardScaler + LogisticRegression(class_weight='balanced', max_iter=5000) on
    representation B, StratifiedKFold(5, shuffle, seed 0) on train-versus-test membership. The interval is a stratified
    bootstrap of the out-of-fold scores (it omits refitting variability). The primary variant uses all rows of the
    setting's train and test sets; a pool-E-rows variant (failed attacks excluded on both sides) is a sensitivity
    analysis. The test rows equal the cache rows in every setting (asserted).
DS7. Additions not in the plan: world-cluster sensitivity for the M3 comparisons (plan 3.5 names world clustering as a
    sensitivity for R1/R3), pattern-standardised AUROC and per-pattern AUROC/recall (m13), a joint-overlap bootstrap
    (m3), and derived precision at 2%/10% prevalence from recall and FPR (M4 values).
DS8. The plan names B = 2,000 for the AUPRC bootstrap; B = 2,000 is used for every metric here. Intervals therefore
    differ slightly from the cache summaries (B = 1,000 for AgentDrift in validation.json).
DS9. m5 overlap (addition, logged 2026-09-27). The R1 and R1-wd pool-E test sets share 287-327 trajectories per split
    seed. The plan asks for independent bootstraps, which stay primary. An overlap-preserving bootstrap (resample the
    union of both pools, stratified by class, and evaluate each pool on its own resampled rows) is added as a
    sensitivity analysis, with 10 RNG reruns of the borderline LR-B seed-1 comparison. The overlap-aware version detects
    that loss before correction, -0.027 [-0.052, -0.002], but not after Holm over seeds.
DS10. Monte Carlo stability (addition, logged 2026-09-27). Every Holm family of M3, M4 and m5, and every m2
    comparison, with a member whose p or pH lies in [0.03, 0.07] at B = 2,000 is recomputed at B = 10,000 with fresh RNG
    streams. The pre-specified B = 2,000 values stay primary. The 12 decisions that change are listed in the summary.
DS11. m12 extra (addition, logged 2026-09-27). The C2ST is also reported for R1wd_s1-s4, for LOSO_*, for each
    cross-validation pair of WUT, WUT_F and WIT, and over C2ST fold-assignment seeds 0-4.

### Content detector (content_detector.py; review M2, m13) - logged 2026-09-26

C1. Resampling unit. Plan 3.5 names clusters for the UT and IT analyses, R2 (tasks) and AgentDrift R1/R3 (trajectories,
    world clustering as sensitivity). R4 transfer and LOSO are not named; they use suite|user_task clusters, like WUT and
    WUT_F. WIT resamples positives by suite|injection goal and negatives by suite|user_task. Sensitivity intervals:
    world clusters for R1, R1-wd, R3; trajectory-level stratified for R2 and the AgentDojo settings.
C2. Cross-validated settings (WUT, WUT_F, WIT): point = mean over the 5 repetitions; each bootstrap replicate draws one
    repetition, shared by all detectors (paired), as evalkit.bootstrap but shared.
C3. Detectors added beyond plan Sec. 2: TFIDF-toolname (task item 4; R1, R3; extras R4 and LOSO); TFIDF-anon (R1) and
    TFIDF-anon-nodigit (R1, R3), a README-style anonymisation (AgentDrift README lines 151-153, 157-158) run as a
    sensitivity analysis stronger than the plan's TFIDF-worldmask, because name tokens survive that mask in 0.198 of R1
    trajectories (L3). WUT_F included for TFIDF as an extra setting. All use the plan's TF-IDF / LR configuration.
C4. Not computed in this stream: plan 3.8 TPR at FPR budgets of 1% / 5% set on training negatives (the cache stores no
    training scores for the text detectors); [Consolidation, Part D X2: TF-IDF budgets are reported for R1 by the prevalence stream]; the plan 3.1 secondary subsampling AP estimate; cluster bootstraps for the
    domain-matched R3 minus R1 differences (trajectory level, as plan 3.5 prescribes for R1 and R3).
C5. p-values: two-sided bootstrap p = 2 min(P(d<=0), P(d>=0)). Holm families: R3 five folds (per contrast and metric),
    LOSO four suites, R1-wd five seeds (reported only), R3 minus domain-matched R1 (per detector and metric), and
    within-minus-transfer over {TFIDF, TFIDF-nodigit} x {WUT, WIT} per metric. Plan family (d) presumably spans all
    detectors; this stream's family covers the content detectors only.
C6. Extras: precision at 2% / 10% implied at theta_r80 by recall and FPR (Eq. 6); FPR at a fixed test TPR of 0.8
    (test-oracle operating point, plan 3.8); exact (Clopper-Pearson) 95% upper bounds of the FPR where no pool-E
    negative is flagged, with the implied lower bound on precision.
C7. README claims (m13) checked under our own operationalisations (the README defines none): poisoned observation =
    the injection_point or failed_injection step's observation; ext-audit as domain or substring; hard-negative
    markers = '$5,000' amount, audit@ address in a world domain, 'transfer' substring. Neither README figure is
    reproduced exactly (28.9% vs 0.281-0.292; 75% vs 0.985 any / 0.864 at least two / 0.509 all three).
C8. '2026' counted as TF-IDF analyzer token and as substring; the review's 75.6% equals the substring share on the
    10,656-trajectory analysis pool.
C9. TFIDF-toolname text: tool names with underscores replaced by spaces, steps joined by ' || ' (dropped by the
    tokenizer), so bigrams span consecutive tools, as in TFIDF.
C10. Point estimates are full-sample values (evalkit.ci reports the bootstrap mean); intervals are percentile intervals.

### Features and observation length (features_obslen.py; review M5, m8) - logged 2026-09-26

The full text of F1-F9 is also in features_obslen_results.json (key `deviations`) and features_obslen_summary.md.
F1. AgentDojo intervals: suite|user_task cluster bootstrap is the primary interval for every AgentDojo P0 quantity (WUT, and
    also R4 transfer and univariate AUROC on P0); stratified trajectory bootstrap reported as sensitivity. AgentDrift R1/R3:
    stratified trajectory bootstrap, world-cluster sensitivity for the observation-length ablations.
F2. Bootstrap p-values (two-sided, (1 + count)/(B + 1), minimum 0.001 at B = 2000); Holm across the five R3 folds per contrast
    and metric; within-minus-transfer Holm over the five detectors of that table only.
F3. Added post hoc detector LR-obsonly (max_obs and mean_obs only; cache recipe), not in the plan's detector list.
F4. Coefficient intervals (not specified by the plan): R1 500 stratified refits; WUT range over the 25 fold models plus a
    descriptive all-P0 fit with 500 cluster-bootstrap refits.
F5. Padding: filler ' The quick brown fox jumps over the lazy dog.' repeated and cut to n characters; only positives padded;
    AgentDojo injected steps located from the raw run's `injections` field (2,285 of 2,294 located; 9 left unpadded); only
    lengthening tested (in AgentDojo lengthening raises the score; shortening not tested); GRU padded in R1 only.
F6. R1 GRU seeds retrained for padding: thresholds, t_r80 predictions and AUROC equal the cache; 5 of 5,643 scores differ by
    at most 6e-8 (float32), so not bitwise identical.
F7. Prevalence-weighted AUPRC at pi = 0.01, 0.05, 0.10 (grid not fixed by the plan); t05 not reported.
F8. tool_state_annotation.json: single annotator (an AI assistant working for the authors), not blind (keyword list known;
    the listing also showed per-tool call and step-label counts); 13 AgentDrift and 1 AgentDojo labels marked ambiguous;
    no second rater.
F9. (added 2026-09-27, second run) Bootstrap seeds aligned across parts: the padding (R1 linear, R1 GRU, WUT) and
    annotation-sensitivity parts now reuse the resampling seed of the matching setting in the ablation part (9200 + setting
    index; 9300 + index for sensitivity), so identical quantities have identical intervals in all tables. Point estimates
    and fitted models unchanged; interval endpoints moved by at most a few thousandths. All parts were re-run with one
    script version.

### OOV mechanism (oov_mechanism.py; review M1 actions 1-5, Q1, m6 <UNK> norm) - logged 2026-09-26

The full text is also in oov_mechanism_results.json (key deviations) and oov_mechanism_summary.md.
O1. Bootstrap size: B = 2000 in every analysis here (the v0.3 AgentDrift tables used B = 1000); fixed rng seeds are
    listed per analysis in the code.
O2. Multi-seed GRU intervals in s3 follow the pre-registered evalkit convention (one training seed drawn per
    replicate; point = seed mean). s7 adds the two-level (seeds then trajectories) and seed-averaged intervals and re-
    applies the plan 3.4 rule under both.
O3. Holm correction needs p-values; they are two-sided bootstrap p-values (add-one), not analytic. 'Detected' without
    Holm = 95% percentile interval excludes 0.
O4. Graded coverage (plan Sec. 4 'Mechanism'): level L masks round((1-L)*103) randomly drawn tool NAMES of the R1
    benign-training vocabulary (26, 52, 77 names; levels 1 and 0 have one deterministic mask), as in the plan's
    'random 25/50/75/100% of tools'. The achieved tool-occurrence coverage varies by draw and is reported, not forced
    to the target. The interval is for the draw- (and seed-) averaged metric conditional on the drawn masks; the per-
    draw range is given separately.
O5. The R1 GRU and GRU-drop0.2 models were not saved by the cache; they were refitted with gru_variants.train_gru_drop
    exactly as build_gru_cache.run_unit calls it (same seeds, validation split and data) and checked against the
    cached scores (checks_r1_gru_refit). GRU-drop0.1 is not part of the graded-coverage request and was not refitted.
O6. Within-AgentDojo (WUT) references: one CV repetition drawn per replicate and shared by all detectors (paired by
    fold assignment); evalkit draws the repetition independently per method.
O7. AgentDojo intervals (LOSO, WUT, R4 Rule 4) use the plan 3.5 cluster bootstrap over suite|user_task as primary. For
    Rule 4 the pre-registered design (PRE_ABLATION_DESIGN.md) used the stratified evalkit bootstrap; it is reported
    alongside. The LOSO stratified intervals are a sensitivity analysis.
O8. GRU tool-pathway 'training negatives' are all label-0 rows of the 90% fit partition (benign, hard_negative,
    failed_attack), the negatives the network was trained on; the mean over benign + hard_negative rows only is also
    given (train_negpoolEcats_mean).
O9. Rule 4's R3 clause (>= 3/5 folds) is reported as a supplementary analysis with pooled R1 (as pre-registered) and
    domain-matched R1 (plan 3.2) references; its R4 clause is primary.
    [Consolidation, Part D X4: for the R4 clause the pre-registered estimand is the R1-referenced degradation difference
    DD_R1_ref, +0.057 [+0.042, +0.073] (stratified); the within-AgentDojo-referenced DD_within_ref (+0.064) is supplementary.]
O10. LOSO has no GRU (none in the cache), and the m6 request for a within-AgentDojo GRU without <UNK>-bearing traces
    is not run here.
O11. R3 fold pools contain every pooled trajectory of the held-out domain, including the R1 test trajectories of that
    domain, so R3 and domain-matched R1 estimates share rows. The R3-minus-R1 differences (s3 threshold_change and
    auroc_change, and the s6 R3 clause; the R1-versus-R4 comparison is unaffected) resample the two populations
    independently, which ignores the positive covariance and makes those intervals conservative.
O12. Every population is resampled once (pool_W, fixed seed per population) and the same resample is used in every
    section, so an estimand that appears in two sections (for example R1 LR-C - LR-B, or WUT LR-C - LR-B in s5 and s6)
    has the identical interval. s7 reuses the s3 mixture results.

### Operating points and prevalence (prevalence_ops.py; review M8, M9, m3, m4, m9, m10, m11, m17.5) - logged 2026-09-26

P1 (interpretation, AgentDojo and R2 intervals). Plan 3.1 asks for a stratified full-pool bootstrap for AUPRC at controlled prevalence; plan 3.5 asks for a (suite, user task) cluster bootstrap for the AgentDojo UT analyses and a task cluster bootstrap for R2. Both are reported: 'traj_strat' (plan 3.1, the scheme named in the task) and 'cluster_*' (plan 3.5). R1 and R3 use the trajectory-level bootstrap; R1 also has a world-cluster sensitivity.
P2 (interpretation, m10 training negatives). The plan does not say which training rows are 'training negatives'. Primary: benign + hard_negative training rows (the classes of the test pool-E negatives); sensitivity: all label-0 training rows (incl. failed attacks). Threshold rule: flag s > t, t = (n-k)-th smallest training-negative score, k = floor(budget*n). The scores are in-sample for the supervised detectors (as for theta_r80).
P3 (new fits). The cache stores no training scores for the linear detectors. The R1 linear detectors were refitted with the unmodified cache code (build_linear_cache.py functions, imported) only to recover in-sample training scores; the refit reproduces the cached test scores (max |diff| reported in meta.R1_train_score_refit) and t_r80. GRU training scores are reconstructed in float64 from the cached logit decomposition of the 90% fit partition; the reconstructed t_r80 differs from the cached float32 value by < 1e-7.
P4 (interpretation, m11). The 'R1 alert rate' is the detector's alert rate at theta_r80 on the R1 test pool E (GRU: per seed). The unlabeled draws come from the held-out pool E; a sensitivity uses all rows (incl. failed attacks) both for the R1 alert rate and for the draws. LR-B and the GRU are added to LR-C as the task requests.
P5 (M9 bootstrap strata). The M9 breakdown resamples each stratum category x hijack pattern (and the failed attacks) separately, so the class composition is fixed as in the test set; the M8 blocks stratify by class (pool E) as evalkit.
P6 (point estimates). Point estimates are plug-in values on the full pool (mean over GRU seeds or CV repetitions); bootstrap means are reported next to them. For the secondary subsampling estimator the point is the replicate mean (v0.3).
P7 (FPR at a fixed test TPR of 0.8, plan 3.8). This uses test labels to place the threshold (matched recall); it is a descriptive comparison, not a deployable operating point. The threshold is re-chosen inside each bootstrap replicate.
P8 (precision under negative mixes). Precision uses the pooled pool-E recall of the setting with the FPR of each negative mix. For AgentDojo pool E has benign negatives only; 'failed_only' (unsuccessful attacks as negatives) is added as an extra mix in every setting.
P9 (post hoc, m11). The pre-specified reading (P4: native-prevalence unlabeled sample, native R1 alert rate) does not reproduce the review's m11 figures (FPR 0.191-0.209). A second reading was added after seeing this: unlabeled traffic at 2% prevalence (4 attacked + 196 negative held-out traces per draw) with the R1 alert rate at 2% prevalence (0.02*TPR + 0.98*FPR at theta_r80, i.e. the alerts per 1,000 at 2% of Table IV / 1000). Both are reported ('primary_poolE', 'deployment_prev02'); the second is labelled post hoc.
P10 (m9 rounding). As instructed, scores are rounded to 12 decimals before every ranking. This changes no value in AgentDrift settings, but in R4 and WUT it merges distinct saturated scores below 5e-13 into ties at 0 (e.g. R4 LR-C AUROC 0.4733 raw vs 0.4697 rounded; see m9_rounding_check). For AgentDojo every rank-based M8 quantity is therefore also given with the raw (cache tie-exact) scores in M8_AgentDojo_raw_scores (same replicates), and the M9 subgroup AUROCs in AUROC_subgroups_raw_scores; the raw values match the cache validation files. Thresholded rates are unaffected (they use the cached predictions).
P11 (degenerate intervals). When a rate is exactly 0 or 1 in every repetition, the bootstrap interval collapses; an exact Clopper-Pearson 95% interval is added ('clopper_pearson_95'), and for precision with zero false positives a lower bound from the CP upper FPR limit and the bootstrap TPR lower limit ('lower_bound_if_FPR_at_CP_upper').
P12 (added in consolidation, 2026-09-27; restriction). The M9 negative-class and hijack-pattern breakdown (plan Sec. 4, 'for every detector and setting') is computed for R1, R2, the five R3 folds, R4 and WUT only (prevalence_ops.py M9_SETTINGS, line 51), not for R1wd_s0-s4, WIT, WUT_F or LOSO_*. See Part D X1. The m10 reading 'test FPR exceeds the budget' holds for the GRU and TF-IDF only (Part D X6).
Full numbers: revision/prevalence_ops_results.json (key "deviations" repeats this list); summary: revision/prevalence_ops_summary.md.

### Diagnostics and reproducibility (pre_ablation_audit.py, repro_checks.py, write_diagnostics_summary.py; review m5, m13, m14, m15) - logged 2026-09-27

D1 (rewrite of an existing file). The v0.3 pre_ablation_audit.py was truncated mid-line at line 44 and did not parse. It was
    rewritten in place, as the task allows, with the same purpose: coverage, world and label structure, and domain and
    world probes. A byte copy of the v0.3 file is kept in revision/pre_ablation_audit_v03_truncated.py.txt. The rewrite
    also covers the provenance extraction (m13) and a world-key search (m5), which the v0.3 file did not have.
D2 (probe design choices the plan leaves open; plan Sec. 4 'Diagnostics'). Probes are StandardScaler plus L2 multinomial
    LogisticRegression (lbfgs, max_iter 5000). C is chosen from logspace(-5, 3, 9) by inner StratifiedKFold(3) grid search on
    accuracy. The outer folds are StratifiedKFold(5) repeated with random_state 0, 1 and 2. The v0.3 text and docstring
    disagree (3-fold versus 5-fold), and the plan does not fix k. The primary metric is accuracy, as in v0.3; balanced
    accuracy is also stored. Intervals are paired trajectory-level bootstraps (B = 2000). A world-cluster bootstrap
    is not meaningful when the classes are the worlds.
D3 (additions not in the plan). Added: (a) a domain probe on benign trajectories only; (b) a 'domain x label' floor for the
    world probe; (c) a supplementary world probe on the four non-email domains (worlds with >= 10 trajectories there;
    301 classes; one outer repetition, for run time), because the main world probe population is 81% email; (d) an
    unregularized reconstruction of the v0.3 probe (C = 1, 3- and 5-fold, with and without shuffling), used only to
    reconcile the manuscript values. None of these replaces the pre-specified nested probe.
D4 (exploratory key search, m5). The search for the key behind '1,029 worlds' covers 143 keys, 6 populations and 2 labels
    (1,716 rows). It is exploratory, and with this many rows a match on a single criterion is expected by chance.
    Only joint matches are treated as evidence, and none exists. [Corrected in consolidation, Part D X5: one row
    (train+val, user+date+contacts[names_sorted]) jointly reproduces 1,029 worlds, purity 0.858 (0.8575), 54 worlds with
    >= 30 trajectories and majority baseline 0.558, though not the largest sizes 194/169/144.]
D5 (definitions of README regularities, m13). The README does not define 'injected observation', 'in-domain audit@' or
    'mention a transfer'. The recomputations use regex approximations (diagnostics_results.json
    m13_regularities_recomputed.definitions). They do not reproduce the README's 28.9% and 75%. [Consolidation, Part D X5: 28.9% is approximately reproduced by the second definition,
    0.281 [0.270, 0.291]; only the 75% is not.]
D6 (prevalence intervals). The per-domain prevalences are fixed by the corpus design. Their bootstrap intervals are
    reported as the task asks, but they describe trajectory resampling only.
Full numbers: revision/diagnostics_results.json; summary: revision/diagnostics_summary.md.

### R4 deep (r4_deep.py; review M6, M7, m1, m2, m6, m7, m15, m16, m17.3, Q2, Q4) - logged 2026-09-27

R4D1 (seeds of new within-AgentDojo GRU fits). gru_WIT, gru_WUT_F, gru_C1_UT, gru_C1_IT, gru_noDoS_UT and
    gru_WUT_noUNK follow gru_ladder.py (one training seed per CV repetition: fold seed = training seed = repetition 0-4;
    gru_ladder.stratified_val(train, repetition)), not the preregistered 3 seeds (same as cache deviation G2).
R4D2 (AUPRC@10% in the Case rule). R4_CONTROL.md Sec. 7 uses 'AUPRC@10%'. The plan procedure uses the plan's
    prevalence-weighted AP (point, cluster lower bound); the v0.3 procedure uses the v0.3 subsample estimator. Both are
    reported. For the within-model strata no subsample AP exists, so both procedures use the weighted AP there.
R4D3 (Case A/B transfer condition). 'The transfer is near chance' is applied to both P0 and C1 (the text does not
    say which population); the conclusion is the same with P0 only for every detector.
R4D4 (near-chance rule). The pre-registered rule (CI includes 0.5, or AUROC < 0.60) is applied literally, including
    its one-sided '< 0.60' clause (review M6(d)); every configuration flags whether near chance holds only through that
    clause. Under the ATTEMPT relabelling, transfer AUROCs below 0.5 with intervals excluding 0.5 count as 'near chance'
    by this rule.
R4D5 (Outcome 4 criteria). GRU_PREREGISTRATION.md does not operationalise 'large seed variance'. Criteria C1-C4 are
    defined here, post hoc (results m16.outcome4.criteria_definitions).
R4D6 (attempt target, not in the plan). M6 action 6 asks for the Case decision under an 'attempt' target; the
    ATTEMPT configurations relabel the 4,987 unsuccessful attacks as positives at evaluation only (models unchanged).
    Exploratory.
R4D7 (model-identity probe design, m15). Nested CV: outer stratified 5-fold x 5 repetitions (shuffle seeds 0-4),
    inner stratified 5-fold grid over C in {1e-3, 1e-2, 0.1, 1, 10, 100}, L2 logistic regression with class_weight
    balanced and StandardScaler, inner scoring AUROC; floors = majority (AUROC 0.5) and suite one-hot (the AgentDojo
    'domain'). Primary metric = pooled out-of-fold AUROC; fold-mean AUROC is stored as a secondary metric.
R4D8 (model-balanced cluster bootstrap). New: fixed full-pool model-balance weights with suite|user_task (or IT)
    clusters resampled. The r4_control.py convention (weighted positive draws, bootstrap-mean estimate) is also
    reported.
R4D9 (m6 exclusion rule). A P0 test trace is 'UNK-bearing' when any of its tools is outside the vocabulary of the
    fold model that scored it (cache cov_model < 1). The noUNK-train arm removes UNK-bearing traces from the 90% fit
    partition only; the 10% early-stopping partition is unchanged.
R4D10 (failed-attack flag-rate intervals). Trajectory-level bootstrap and a suite|user_task cluster bootstrap over
    the 4,987 unsuccessful attacks (94 clusters), B = 2000, seed 4242; one repetition / seed drawn per replicate.
R4D11 (DoS removal). Primary: the 101 DoS positives removed at evaluation (models unchanged). Sensitivity: within-UT
    models refitted without them (lin_noDoS_UT, gru_noDoS_UT). P0F without DoS also drops the 252 unsuccessful DoS runs.
R4D12 (Q2 design). Per-template transfer within gpt-4o-2024-05-13 uses that model's 92 benign traces as the
    negatives of every template; trajectory-level stratified intervals. The pooled template-restricted designs add a
    cluster bootstrap with positives by suite|user_task|injection_task (the review's clustering) and negatives by
    suite|user_task.
R4D13 (degenerate cluster draws). A cluster replicate that contains no positive or no negative returns NaN and is
    dropped from the percentiles; this occurs only for the claude-3-5-sonnet stratum (7 positives), which is outside the
    model-controlled analyses.
R4D14 (v0.3 values the v0.3 scripts never produced). v0.3 per-model LR-A and GRU transfer, and v0.3 WIT LR-A, are
    recomputed on archive_v0.3/agentdojo_traces.jsonl with the v0.3 code paths (ablation.fit; gru_detector.train_gru
    refitted deterministically, checked bit-identical against the cache on corrected P0). The GRU was not run under WIT
    in v0.3.
R4D15 (Holm p-values). Bootstrap two-sided p = 2 min(P(d <= 0), P(d >= 0)); Holm within the five representations of
    a configuration (plan 3.6 d) and within the 12 within-capable detectors for the all-detector tables.
Full numbers: revision/r4_deep_results.json (key "deviations"); summary: revision/r4_deep_summary.md.
