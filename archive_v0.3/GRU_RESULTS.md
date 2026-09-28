# GRU_RESULTS: does a learned sequence detector reproduce the evaluation pattern?

*25 Sep 2026. Implements `GRU_PREREGISTRATION.md` exactly — architecture, training, thresholds, and interpretation rules unchanged from that document. Scripts: `gru_detector.py` (model), `gru_ladder.py` (split audit + full ladder). Raw output: `gru_audit.json`, `gru_results.json`. Existing A/B/C results (`ablation_results.json`, `r4_control_results.json`) were not modified or re-run.*

---

## 0. Split audit (run before the full experiment; `gru_audit.json`)

| rung | n_train | n_val | n_test | train prev | test prev | tool vocab | tool_occ_cov | frac_zero_known |
|---|---|---|---|---|---|---|---|---|
| R1 | 8,775 | 1,880 | 1,881 | 0.442 | 0.442 | 103 | 1.000 | 0.000 |
| R2 | 9,081 | (10% of train) | 1,722 | 0.441 | 0.450 | 103 | 1.000 | 0.000 |
| R3 medical | 8,446 | (10% of train) | 2,210 | 0.436 | 0.462 | 83 | 0.000 | 1.000 |
| R3 banking | 8,840 | (10% of train) | 1,816 | 0.462 | 0.345 | 83 | 0.000 | 1.000 |
| R3 web | 8,446 | (10% of train) | 2,210 | 0.436 | 0.462 | 83 | 0.000 | 1.000 |
| R3 email | 8,446 | (10% of train) | 2,210 | 0.436 | 0.462 | 83 | 0.000 | 1.000 |
| R3 coding | 8,446 | (10% of train) | 2,210 | 0.436 | 0.462 | 80 | 0.000 | 1.000 |
| R4 transfer | 10,656 | (10% of train) | 2,686 | 0.442 | 0.854 | 103 | 0.075 | 0.698 |
| R1-world-disjoint | 8,762* | (10% of train) | 1,894* | 0.438* | 0.457* | 103 | 1.000 | 0.000 |

*seed-0 split reused from Experiment 1b; world overlap = 0 (confirmed).

**Seeds:** 3 training seeds {0,1,2} per fit; validation-split seed base 1000 (+ training seed); K=5 CV folds where applicable. All fixed in `GRU_PREREGISTRATION.md` §4, not tuned here.

Counts, prevalence, vocabulary size and coverage all match the corresponding audits from the A/B/C ablation and R4_CONTROL runs (same underlying splits), confirming the GRU pipeline reads identical data.

---

## 1. Reproduction / sanity checks
The combined logistic regression **C** is retrained inside `gru_ladder.py` as the paired comparator (same recipe as `ablation.py`/`r4_control.py`). Its native-prevalence AUROC reproduces the established values:

| | this run (C) | previously established |
|---|---|---|
| R1 | 0.879 | 0.879 |
| R4 transfer (P0) | 0.440 | 0.440 |
| R4 within-AgentDojo (P0) | 0.855 | 0.855 |

**All three anchors match exactly.** The GRU results below are read against a verified-consistent baseline.

## 2. R1 and world-disjoint R1

| | GRU AUROC | C AUROC | GRU AUPRC (native) | C AUPRC (native) | paired ΔAUROC (GRU−C) |
|---|---|---|---|---|---|
| R1 (standard) | **0.936 [0.923, 0.947]** | 0.879 [0.862, 0.895] | 0.936 [0.921, 0.950] | 0.876 [0.856, 0.894] | **+0.056 [+0.042, +0.071]**, detected |
| R1-world-disjoint | **0.956 [0.943, 0.967]** | 0.891 [0.875, 0.904] | 0.957 [0.946, 0.968] | 0.901 [0.885, 0.915] | **+0.065 [+0.051, +0.081]**, detected |

The GRU beats C in-distribution, and — like C in Experiment 1b — it does **not** degrade when test worlds are entirely unseen; if anything it is slightly higher (0.936 → 0.956). This is consistent with the earlier conclusion that R1 is not inflated by world reuse, now for a second detector family.

## 3. R2 (unseen tasks, same tool vocabulary)

| | GRU AUROC | C AUROC | paired ΔAUROC |
|---|---|---|---|
| R2 | 0.915 [0.898, 0.935] | 0.902 [0.886, 0.917] | +0.014 [−0.005, +0.033], **not detected** |

No meaningful GRU-vs-C difference. Consistent with the earlier finding that R2 shows no degradation for C either.

## 4. R3 — leave-one-domain-out (coverage 0% every fold; the GRU is **not** structurally constant here — see §7)

| held-out domain | GRU AUROC | C AUROC | paired ΔAUROC (GRU−C) | GRU AUPRC@10% |
|---|---|---|---|---|
| medical | 0.639 [0.606, 0.668] | 0.745 [0.722, 0.766] | **−0.107 [−0.130, −0.086]** | 0.172 [0.128, 0.231] |
| banking | 0.736 [0.697, 0.777] | 0.828 [0.808, 0.849] | **−0.093 [−0.134, −0.059]** | 0.289 [0.208, 0.382] |
| web | 0.726 [0.690, 0.765] | 0.862 [0.845, 0.877] | **−0.136 [−0.166, −0.100]** | 0.267 [0.184, 0.374] |
| email | 0.671 [0.625, 0.732] | 0.739 [0.718, 0.760] | **−0.068 [−0.106, −0.011]** | 0.217 [0.146, 0.312] |
| coding | 0.695 [0.653, 0.746] | 0.814 [0.796, 0.833] | **−0.119 [−0.154, −0.078]** | 0.202 [0.148, 0.269] |

**GRU is significantly *worse* than C in all five folds** (all paired CIs exclude 0, all favouring C). At the same time, **GRU stays well above chance in every fold** (AUROC 0.64–0.74, all CI lower bounds > 0.5), and every fold's GRU AUROC CI is entirely below its own R1 CI [0.923, 0.947] — a detected R1→R3 degradation in **5 of 5 folds** by non-overlapping intervals.

## 5. R4 — AgentDrift → AgentDojo (coverage 7.5%, near-OOV)

**Transfer, P0 (392 benign / 2,294 attacked):**
| | GRU AUROC | C AUROC | paired ΔAUROC (GRU−C) |
|---|---|---|---|
| transfer | 0.488 [0.458, 0.518] | 0.440 [0.406, 0.472] | **+0.049 [+0.038, +0.060]**, detected |

**Transfer, C1 (injection-task benigns removed):** GRU 0.486 [0.454, 0.519], C 0.444 [0.408, 0.478]; ΔAUROC +0.042 [+0.031, +0.053].

**Model-balanced (4 eligible models, n_neg 308):** GRU 0.579 [0.459, 0.695], C 0.527 [0.418, 0.635]. **Both CIs include 0.5.**

GRU transfer is nominally a few points higher than C's, and the paired difference is detected — but GRU transfer itself is **at chance** (P0 and C1 CIs straddle or sit almost exactly at 0.5; model-balanced CI spans 0.459–0.695, including 0.5). This is not a meaningful improvement in absolute detection ability; it is a small, real difference between two near-useless detectors.

## 6. Within-AgentDojo control (same logic as R4_CONTROL, now for GRU)

| population | GRU within AUROC | C within AUROC | GRU within−transfer Δ |
|---|---|---|---|
| P0 | 0.860 [0.825, 0.886] | 0.855 [0.830, 0.879] | **+0.371 [+0.329, +0.410]**, detected |
| C1 | 0.849 [0.808, 0.888] | 0.867 [0.841, 0.894] | **+0.363 [+0.315, +0.413]**, detected |
| model-balanced | 0.842 [0.758, 0.910] | 0.811 [0.725, 0.884] | **+0.263 [+0.129, +0.397]**, detected |

**The Case A pattern reproduces for the GRU.** Trained within AgentDojo, the GRU detects attacks about as well as C (0.84–0.86 AUROC in every population); trained on AgentDrift, it is at chance on the same traces. This is not a representation-inadequacy story for either detector family — both can learn AgentDojo when given AgentDojo.

## 7. Comparison with C — summary across the ladder

| rung | GRU vs C | direction |
|---|---|---|
| R1 | GRU better, detected | +0.056 |
| R1-world-disjoint | GRU better, detected | +0.065 |
| R2 | no difference detected | +0.014 (CI crosses 0) |
| R3, all 5 folds | **GRU worse, detected in all 5** | −0.07 to −0.14 |
| R4 transfer (P0, C1) | GRU nominally better, detected, but both near chance | +0.04 to +0.05 |
| R4 transfer (model-balanced) | no difference detected (both CIs include 0.5) | +0.05 |
| R4 within-AgentDojo | comparable (no strong claim either way; overlapping CIs) | ±0.01–0.03 |

**The GRU is not uniformly "stronger."** It wins clearly in-distribution and loses clearly under within-benchmark domain shift (R3) — a *new*, GRU-specific finding not anticipated by the four pre-specified outcomes in exactly this form (see §8).

## 8. Which pre-specified outcome was met?

Re-reading `GRU_PREREGISTRATION.md` §9 against the results:

- **R1 strong:** AUROC CI lower bound 0.923 > 0.7 ✓.
- **R1→R3 degradation detected in ≥3/5 folds:** met in **5/5** folds — every R3 AUROC CI sits strictly below the R1 CI ✓.
- **R4 transfer near chance:** P0 AUROC CI [0.458, 0.518] includes 0.5 ✓; model-balanced [0.459, 0.695] includes 0.5 ✓.

**→ These three conditions together satisfy Outcome 1 ("same qualitative pattern"): strong in-distribution performance, detected domain-shift degradation, near-chance cross-benchmark transfer.** This strengthens the argument that the evaluation regime — not the linear detector specifically — drives the pattern.

**Outcome 3 ("GRU also fails R4, strengthening the cross-benchmark-transfer-is-hard evidence") is also satisfied** as a sub-case of Outcome 1's R4 finding, for a second detector family.

**Outcome 2 (GRU substantially improves R3/R4) is explicitly NOT met.** GRU R4 transfer CI lower bounds (0.458 P0; 0.459 model-balanced) are far below the pre-specified 0.6 bar, and R3 shows detected degradation in 5/5 folds, not "no detected degradation in ≥3/5." **We do not claim robustness depends on architecture in the direction of the GRU being more robust** — if anything the opposite.

**Outcome 4 (inconsistent, no common narrative) is not the best description** — the pattern is consistent across seeds and folds (tight CIs, no sign flips), just not identical in magnitude to the linear detectors.

**An outcome the pre-registration did not anticipate, and states plainly per instruction:** **the GRU is significantly *less* robust to within-benchmark domain shift than the simpler linear combined detector C**, despite being stronger in-distribution and despite retaining above-chance signal via `<UNK>` + behavioural features (unlike linear tool-only A, which is structurally at chance under R3). This is reported as observed, not folded into any of the four pre-specified buckets, because none of them describes a *architecture-dependent reversal of the ranking* under domain shift specifically.

## 9. Limitations and what this experiment does NOT establish

- **Two detector families is not "detectors in general."** The Outcome-1 conclusion ("the evaluation regime matters beyond the specific linear detector") is supported for these two families; it is not evidence that *every* architecture would show the same pattern, and the new R3 finding shows architecture *does* matter for the magnitude and even the ranking under shift.
- **The GRU's R3 weakness relative to C is not diagnosed here.** We did not run a tool-only-vs-behavioural-only GRU ablation (explicitly excluded from this pre-registration), so we cannot yet say whether the GRU's `<UNK>`-driven sequence representation, its capacity, its training regime (early stopping on a small validation slice), or something else drives the gap. This is a candidate for a future, separately pre-registered ablation — not run now, per the instruction to add no additional detector variants.
- **`<UNK>` handling means GRU-vs-A comparisons under R3/R4 are not apples-to-apples** (§7 of the pre-registration): the GRU retains length/behavioural signal that linear tool-only A structurally cannot. We have therefore only compared GRU to C, as pre-specified, not to A.
- **R4 "detected" differences are small and both sides are near chance.** The GRU's statistically detected edge over C in R4 transfer (P0, C1) is not practically meaningful — neither detector is usable near 0.44–0.49 AUROC. The model-balanced arm, which controls the sharpest known confound, shows **no detected difference** between GRU and C, and both CIs include 0.5.
- **Training instability was not observed** (no NaN losses, no learning-rate fallback triggered across all ≈85 fits), so the pre-registered abort rule was never exercised.
- **No hyperparameter tuning was performed**, as pre-specified; the single fixed configuration is reported as-is, including where it under-performs C.

---

## Do not yet: manuscript and prior findings

Per instruction, `MANUSCRIPT_DRAFT.md` is unmodified, and no existing finding in `FINDINGS.md` has been revised based on this run. The new, load-bearing facts that would need to be folded in on the next pass are:
1. A learned sequence detector reproduces the paper's central pattern (Outcome 1): strong in-distribution, degraded under domain shift, at-chance under cross-benchmark transfer — evidence the pattern is not an artifact of using linear models.
2. A **new, unanticipated finding**: under within-benchmark domain shift (R3), the GRU is *less* robust than the simple linear combined detector, reversing the in-distribution ranking. This qualifies any future claim that "a stronger detector would fix generalization" — here, the stronger in-distribution detector is the *worse* generalizer.
3. World-disjointness continues to show no inflation of R1, now for a second detector.
