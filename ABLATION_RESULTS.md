# ABLATION_RESULTS: what does the current detector rely on?

> **Read `PRE_ABLATION_DESIGN.md` first.** It is the authoritative pre-registration and supersedes the outcome rules in Part 1 below where they differ (it adds the vocabulary-coverage conditioning, the B− arm, the domain/world diagnostics, and Decision B). Part 1 here remains as the feature-definition reference. Results (Part 2) are not yet computed.

*Part 1 (feature definitions, protocol, operational outcome rules) was written **before** any ablation result was computed. Part 2 will be appended after the run. Script: `ablation.py` (not yet finalized). Raw output: `ablation_results.json`. The evaluation framework is `evalkit.py`, unchanged.*

---

## PART 1: PRE-SPECIFIED DESIGN

### 1. Exact feature definitions (all from `featurise.vectorise`, unchanged)
**Behavioural features (10), per trajectory:**

| # | name | definition | depends on tool names? |
|---|---|---|---|
| 0 | len | number of steps | no |
| 1 | uniq | number of distinct tool names in the trace | only via name *equality* |
| 2 | diversity | uniq / len | only via name equality |
| 3 | repeats | number of immediate repetitions (tool_t == tool_{t−1}) | only via name equality |
| 4 | write_n | number of steps whose tool name contains a keyword in {send, delete, update, create, transfer, add, download, post, write, cancel, apply, convert, share, grant} | **yes, keyword match on names** |
| 5 | write_frac | write_n / len | **yes, keyword match on names** |
| 6 | max_args | max over steps of the argument-dict size | no |
| 7 | mean_args | mean argument-dict size | no |
| 8 | max_obs | max observation length in characters | no |
| 9 | mean_obs | mean observation length in characters | no |

**Tool-identity features:** a bag-of-tools count vector over the vocabulary of tool names seen in **benign training** traces. Unseen tools map to nothing.

**Ambiguity (documented, not resolved by inventing features).**
- Features 4 and 5 use tool-name *semantics* through a hand-written keyword list. They are vocabulary-independent, in the sense that they need no fixed vocabulary, but they are not schema-free.
- Features 1–3 use only name equality within a trace.
- Because of this, a sensitivity representation **B−** removes features 4 and 5.

**Representations** (same `LogisticRegression(max_iter=1000, class_weight="balanced")`, with `StandardScaler` fitted on the training data):
- **A: tool identity only.** Bag-of-tools only.
- **B: behavioural only.** The 10 features above. This is identical to "LOGREG-agnostic" in R4_CONTROL.
- **B−: behavioural without name-keyword features.** Features 0–3 and 6–9. This is a sensitivity arm only.
- **C: combined.** The 10 behavioural features plus bag-of-tools (the current full representation).

No representation-specific tuning.

### 2. Structural facts, measured before the run (they constrain interpretation)
- **R1:** 100% of test tool occurrences are in the training vocabulary.
- **R3 (each held-out domain):** **0%** coverage. No tool name is shared between any two AgentDrift domains, so every held-out trace has an all-zero tool vector.
  - **Consequence:** under R3, **A gives a constant score by construction (AUROC = 0.5, AUPRC = prevalence).**
  - Also, **C's ranking under R3 depends only on its behavioural weights**, because its tool part is constant. An R3 comparison of A vs B is therefore *determined by construction* and is not evidence of behavioural robustness.
  - The informative R3 questions are: (i) does B keep above-chance signal, and how much does it degrade from its own R1 level; (ii) does C (behavioural weights learned jointly with tool features) behave differently from B (learned alone)?
- **R4:** 7.5% of AgentDojo tool occurrences are in the AgentDrift benign vocabulary, and 69.8% of AgentDojo traces contain no known tool. A is therefore nearly constant on most R4 traces.
- **World identity:** there are 1,029 distinct `world` dicts (user, email, company, contacts, date). In-sample majority-label lookup by world gives 0.858 accuracy, against a majority baseline of 0.558. This is consistent with the AgentDrift paper's reported 86.1%.

### 3. Protocol (identical to the corrected framework; nothing tuned)
- **R1.** Train on AgentDrift `data/train`, test on `data/test`.
  - Pool E = benign + hard_negative + attacked (failed_attack excluded).
  - Stratified bootstrap with B = 1,000, seed 5000 (as in `rerun_r1r3.py`).
  - Prevalence: native, and 1/2/5/10%, with n_neg = |E_neg| and n_pos = round(p/(1−p)·n_neg).
- **R3.** Leave-one-domain-out over the pooled stratified train and test sets. Five folds, seeds 5000 + 17·ri with ri = 2..6 (as in `rerun_r1r3.py`). Same pools and prevalences.
- **R4.**
  - **Transfer:** train on AgentDrift train+test, score AgentDojo.
  - **Within-AgentDojo reference:** per representation, with UT and IT schemes, 5 CV repetitions (seeds 0–4). This is the established R4 control, needed to say whether a representation can detect AgentDojo attacks *at all*.
  - **Populations:** P0 (full); C1 (injection-task benigns removed); model-balanced (4 eligible models, positives reweighted to the benign model mix).
  - B = 2,000, seed 12345. Prevalence: native, 10%, 5%.
- **Thresholds.** τ_r80 = the threshold giving recall 0.80 on *training* positives (in-sample), plus τ = 0.5. FPR is reported at τ_r80.
- **Paired comparisons** (all representations scored on the same examples, in the same bootstrap call): B−A, C−A, C−B, B−(B−).
- **Degradation difference** (for example, whether B degrades less than C from R1 to R3): per replicate b,
  d_b = (X_shift[b] − Y_shift[b]) − (X_R1[b] − Y_R1[b]).
  Within-rung terms are paired; the R1 and shift bootstraps are independent.

**Note on reproducibility.** C and B are the same models as LOGREG and LOGREG-agnostic in the corrected runs. Their *native point estimates* must reproduce the earlier values (R1 AUROC 0.879; R4 transfer 0.440 and 0.447; R4 within-UT 0.855 and 0.798). Bootstrap CIs can differ slightly, because adding methods to a bootstrap call changes the random-number stream.

### 4. Operational outcome rules (fixed before the run)
Let "detected" mean a 95% CI that excludes 0.
- **Outcome 1: behavioural robustness is supported** if **both** hold:
  - (a) B keeps signal under shift: its AUROC CI lower bound is > 0.5 in ≥3 of 5 R3 folds **and** in R4 transfer, for P0 and model-balanced;
  - (b) B degrades less than C from R1 to shift: the degradation difference (B vs C) is detected as favouring B in ≥3 of 5 R3 folds, or in R4.
  - (A is not the comparator here, because A's collapse under R3/R4 is structural.)
- **Outcome 2: the behavioural-transfer hypothesis is withdrawn** if B loses signal under shift (AUROC CI includes 0.5 or sits below it) in R4, **or** B's degradation from R1 is statistically indistinguishable from C's in most rungs.
  - It may be withdrawn for cross-benchmark transfer and retained, in narrowed form, for within-benchmark domain shift, if the R3 and R4 results differ.
- **Outcome 3: tool identity performs better** if A > B under a shift where A's features are *not* structurally zero (only R4 is eligible, on the ~30% of traces with known tools), or A > B in R1.
- **Outcome 4: combined best in-distribution but less robust** if C > A and C > B in R1 (paired, detected), **and** C degrades more than B under shift (degradation difference detected in ≥3 of 5 R3 folds, or in R4).
- Outcomes can co-occur, and are reported as observed.

### 5. Leakage diagnostics (diagnostic probes only, not detectors)
- **D1: domain from features** (AgentDrift, 5 domains). 5-fold stratified CV, multinomial LogReg, for A, B and B−. Report accuracy vs the majority baseline.
- **D2: world from features** (AgentDrift worlds with ≥30 trajectories). Multiclass CV accuracy vs the majority baseline, and vs a domain-only baseline (predict the most frequent training world of the trace's domain).
- **D3: world→label leakage in the R1 split.** The share of R1 test traces whose world occurs in training, and on those traces, the AUROC of the train-world attack rate as a label predictor. This measures how much label information world identity *carries*, independent of our features.
- **D4: dataset identity** (AgentDrift vs AgentDojo, benign traces only, to remove the class confound). 5-fold CV AUROC for A, B and B−.

---

## PART 2: RESULTS (computed after Part 1; `ablation_results.json`; interpreted under PRE_ABLATION_DESIGN.md Rules 0–5)

Representations: **A** tool-only, **B** behavioural, **C** combined. Same LogReg, same corrected framework. Format: estimate [95% bootstrap CI]. **Reproduction anchors pass:** R1 C AUROC 0.879; R4 transfer_B 0.447, transfer_C 0.440; within_C 0.855.

### 3. R1 — same distribution (coverage 100%: this is the clean comparison)
n_neg 825, n_pos native 831 (prev 0.502).

| rep | AUROC | AUPRC native | AUPRC @10% | recall / FPR @τ_r80 |
|---|---|---|---|---|
| A tool-only | 0.777 [0.756, 0.799] | 0.794 [0.770, 0.818] | 0.389 [0.302, 0.488] | 0.744 / 0.335 |
| B behavioural | 0.830 [0.810, 0.848] | 0.820 [0.798, 0.841] | 0.407 [0.314, 0.505] | 0.795 / 0.284 |
| C combined | **0.879 [0.862, 0.895]** | 0.876 [0.856, 0.894] | **0.528 [0.434, 0.621]** | 0.771 / 0.192 |

Paired (all detected, P(Δ≤0)=0.00): **C−A +0.101 [+0.085, +0.117]; C−B +0.049 [+0.034, +0.064]; B−A +0.052 [+0.027, +0.078].**
→ At full coverage, C > B > A. Combining tool + behavioural improves in-distribution performance (**Q2: yes**). Behavioural alone already beats tool-only.

### 4. R3 — leave-one-domain-out (coverage 0% for every fold; A is constant by construction — a coverage result, not a representation-quality result)

Coverage every fold: tool-occurrence 0.0, distinct 0.0, traces with any unseen tool 1.0, zero known tools 1.0. **A therefore outputs one score for all traces → AUROC 0.500, AUPRC = prevalence, recall 0 at τ_r80.** Per Rule 1 this is information loss from unseen tools, not evidence that tool identity intrinsically fails.

AUROC [95% CI]:
| held-out domain (n_neg / n_pos) | A | B behavioural | C combined | C−B (paired) |
|---|---|---|---|---|
| medical (935/1020) | 0.500 | **0.808 [0.789, 0.827]** | 0.745 [0.724, 0.768] | **−0.063 [−0.073, −0.052]** |
| banking (935/626) | 0.500 | **0.854 [0.836, 0.873]** | 0.828 [0.808, 0.848] | −0.026 [−0.038, −0.015] |
| web (935/1020) | 0.500 | **0.896 [0.883, 0.909]** | 0.862 [0.847, 0.876] | −0.034 [−0.046, −0.023] |
| email (935/1020) | 0.499 | **0.792 [0.773, 0.812]** | 0.739 [0.718, 0.760] | −0.053 [−0.064, −0.044] |
| coding (935/1020) | 0.500 | **0.852 [0.836, 0.869]** | 0.814 [0.795, 0.833] | −0.037 [−0.048, −0.027] |

- **B retains strong signal with the tool vocabulary entirely gone** (AUROC 0.79–0.90; AUPRC@10% 0.35–0.57). **Q1: yes** — behavioural information retains useful detection signal when tool identity is out-of-vocabulary (within-benchmark domain shift).
- **C < B in all five folds** (C−B negative, P(Δ≤0)=1.00 everywhere). Combined is *less* robust to domain shift than behavioural-only, even though C beats B in-distribution. Under R3 C's tool block is all-zero, so this shows the behavioural sub-model that C learned *jointly with* tool features generalizes worse than a behavioural model trained alone. **This is Outcome 4, within-benchmark: tool-identity features raise in-distribution AUROC (+0.049 in R1) but reduce domain-shift robustness.**
- **Operating point:** at τ_r80 from training, B keeps FPR 0.15–0.42, but C's FPR blows up to 0.50–0.72 (recall 0.83–0.99). C's threshold transfers far worse than B's.
- **A vs B/C under R3 is not an equivalent-information comparison** (Rule 0): A has zero tool information here.

### 5. R4 — AgentDrift → AgentDojo (coverage 7.5%: near-OOV regime)
Transfer coverage: tool-occurrence 0.075, distinct 0.105, traces with any unseen tool 0.987, zero known tools 0.698.

**P0 (392 benign / 2,294 attacked):**
| rep | transfer AUROC | transfer AUPRC@10% | within-AgentDojo AUROC | within-transfer Δ AUROC |
|---|---|---|---|---|
| A | **0.566 [0.543, 0.586]** | 0.152 [0.104, 0.229] | 0.788 [0.759, 0.816] | +0.222 [+0.188, +0.257] |
| B | 0.447 [0.413, 0.479] | 0.090 [0.076, 0.113] | 0.798 [0.771, 0.825] | +0.351 [+0.305, +0.395] |
| C | 0.440 [0.406, 0.472] | 0.090 [0.075, 0.114] | 0.855 [0.830, 0.879] | +0.415 [+0.374, +0.458] |

**Model-balanced @10% (4 eligible models, n_neg 308):** transfer A 0.554 [0.478, 0.632], B 0.523 [0.418, 0.630], C 0.527 [0.418, 0.635] (all CIs include 0.5); within A 0.771, B 0.723, C 0.811.

- **Behavioural transfer is below chance** (B 0.447 < 0.5). Behavioural features do **not** transfer across benchmarks; they are benchmark-specific. **The within-benchmark claim from R3 does not extend to cross-benchmark** (Rule 3): the behavioural-transfer hypothesis is **withdrawn for cross-benchmark transfer**, retained only for within-benchmark domain shift.
- **Tool-only transfers slightly better than behavioural here** (A 0.566 vs B/C 0.44), the reverse of R3. With only 7 shared tools this is a weak signal on the ~30% of traces with a known tool; under model balancing it falls to 0.554 (CI includes 0.5). So it is marginal, not a robust "tool identity transfers" result (Rule 5 caveat).
- **Within-AgentDojo, all three representations detect attacks** (AUROC 0.79–0.86; C best). So R4's near-chance transfer is not because the features cannot separate AgentDojo attacks — trained on AgentDojo they can. This reinforces the R4_CONTROL Case A conclusion, now for every representation.

### 6. Paired comparisons — summary
- R1: C > B > A (all detected).
- R3: B > C > A in every fold (C−B detected negative in all 5; A structural).
- R4 within-AgentDojo: C > B ≈ A. R4 transfer: A > B ≈ C, but all near chance once model-balanced.

### 7. World / domain leakage diagnostic (from the pre-registration audit, carried here)
- Tool identity predicts domain at accuracy 1.000 (baseline 0.207); behavioural 0.455; behavioural−keywords 0.381.
- Neither family recovers individual world identity beyond the domain-only baseline (all ≈ 0.04–0.07).
- The R1 split reuses worlds (99.3% of test worlds seen in training) and world identity is highly label-predictive by lookup (AUROC 0.963). This is a split-design leakage *vector*; the features do not encode world, so the detector does not directly exploit it. The accessible shortcut is domain identity.
- **S2 check (domain base rates):** R3 native attacked prevalence is 0.522 for medical/web/email/coding and 0.401 for banking; R1 pool is 0.502. Base rates are similar across domains (banking modestly lower), so domain identity is not a strong label shortcut. The strong D1 domain signal is therefore mostly benign for detection, but see the R1 caveat below.

### 8. Interpretation
- **Q1 (does behavioural retain signal when tools disappear?)** Yes for within-benchmark domain shift (R3: B 0.79–0.90 while A is structurally 0.5). No for cross-benchmark transfer (R4: B below chance).
- **Q2 (does combining improve in-distribution?)** Yes (R1: C 0.879 > B 0.830 > A 0.777, all detected).
- **Q3 (what happens under shift once coverage is accounted?)** R3: A's collapse is pure 0% coverage; B is most robust; **C is significantly less robust than B** despite winning in-distribution (Outcome 4). R4: near-OOV; behavioural transfers below chance, tool-only marginally above but ~chance when model-balanced; within-AgentDojo all representations work.
- **R1 caveat (per instruction):** the R1 split heavily reuses worlds and world identity is label-predictive, so strong R1 numbers are **not** described as robust generalization across worlds. R3/R4 are the generalization evidence.

### 9. Hypotheses supported
- **Supported:** behavioural features retain detection signal under within-benchmark domain shift when tool identity is out-of-vocabulary (R3).
- **Supported (new):** combining tool + behavioural improves in-distribution AUROC but *reduces* robustness to domain shift relative to behavioural-only (Outcome 4; C < B in all five R3 folds).
- **Not supported / withdrawn:** that behavioural features transfer across benchmarks — R4 behavioural transfer is below chance.
- **Not supported:** that tool identity intrinsically fails under domain shift — its R3 collapse is a 0%-coverage artifact.

### 10. Previous claims to withdraw or revise (FINDINGS only; MANUSCRIPT not yet touched)
- **Withdraw the unqualified claim** "vocabulary-agnostic behavioural features transfer across benchmarks; tool-identity features don't." Corrected: behavioural features retain signal under *within-benchmark* domain shift (tool identity is simply unavailable there at 0% coverage); across *benchmarks*, behavioural features transfer below chance and tool-only is marginally (not robustly) better.
- **Add** the Outcome-4 finding: combined < behavioural under domain shift.
- **Sensitivity arms B− and A+UNK were not triggered:** the core interpretation does not depend on them (the C<B and behavioural-retention findings are clear without separating the keyword features or adding an OOV token). They remain optional.

---

## PART 3: World-disjoint R1 (Experiment 1b) — is standard R1 inflated by world reuse?

*Script `world_disjoint_r1.py`; raw `world_disjoint_r1_results.json`. Same detector, same A/B/C features, same evalkit framework, training-derived thresholds.*

**Split construction.** Pool = AgentDrift `data/train` + `data/test` (10,656 trajectories). Whole `world`s are assigned **globally** to train or test (a world goes entirely to one side, even if it appears in more than one domain), visiting worlds round-robin across domains so every domain stays on both sides. Target test fraction 0.176 (matching standard R1). 5 world-assignment seeds; full stratified bootstrap on seed 0.

**Split statistics (seed 0; representative).** train 8,762 / test 1,894; **world overlap 0** (1,027 train worlds, 106 test worlds); **tool-occurrence coverage 1.000**; all 5 domains in test; attacked fraction train 0.438 / test 0.457.

**Results (world-disjoint):**
| rep | AUROC 5-seed mean [min,max] | seed-0 bootstrap | AUPRC (seed 0) | recall/FPR @τ_r80 |
|---|---|---|---|---|
| A tool-only | 0.820 [0.815, 0.825] | 0.821 [0.801, 0.839] | 0.819 [0.798, 0.837] | 0.803 / 0.339 |
| B behavioural | 0.804 [0.797, 0.811] | 0.804 [0.785, 0.824] | 0.756 [0.729, 0.783] | 0.793 / 0.353 |
| C combined | 0.893 [0.889, 0.901] | 0.889 [0.874, 0.902] | 0.876 [0.860, 0.893] | 0.785 / 0.180 |

**Comparison with standard R1 (AUROC):** A 0.777 → 0.820 (+0.043); B 0.830 → 0.804 (−0.026); C 0.879 → 0.893 (+0.014).

**Interpretation (cautious).**
- **No performance drop under unseen worlds.** C is essentially unchanged; A rises slightly; B falls slightly; all changes ≤0.04. **Standard R1 is not substantially inflated by world reuse**, which *strengthens* the R1 result. This is consistent with the diagnostic that the features do not encode individual world identity.
- **Not attributed to leakage.** A genuine leakage-removal effect would lower performance; the largest change is A increasing. The small shifts are more consistent with split-composition variation than removed memorization.
- **Ordering.** C > A ≈ B under world-disjoint (C clearly top; A and B swap vs standard R1, CIs overlap). The standard-R1 ordering B > A does **not** survive; only C > {A, B} is robust.
