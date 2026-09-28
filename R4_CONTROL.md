# R4_CONTROL: is AgentDrift→AgentDojo a transfer failure or a representation failure?

*Part 1 (design and decision rules) was written **before** any within-AgentDojo result was computed. Part 2 (results) is appended after the runs. Scripts: `evalkit.py`, `r4_control.py`, `rerun_r1r3.py`.*

---

## PART 1: PRE-SPECIFIED DESIGN

### 1. Dataset populations
- **P0 (original R4 population).** All traces extracted by `agentdojo_extract.py` from the recorded runs of 5 no-defence models (gpt-4o-2024-05-13, claude-3-5-sonnet-20241022, gemini-1.5-pro-002, gpt-4-turbo-2024-04-09, gemini-2.0-flash-001) at AgentDojo commit `089ed468`. It contains 392 benign and 2,294 attacked traces (2,686 total). Failed attacks and traces with fewer than two tool calls were already excluded at extraction.
- **C1 (injection-task control).** P0 minus the 65 benign traces whose user task is an `injection_task_*` (the attacker's goal, run as the agent's own legitimate task). This leaves 327 benign and 2,294 attacked.
- **Model control (evaluation-level).**
  - (i) AUROC and AUPRC computed *within* each model that has ≥50 benign and ≥50 attacked traces, and their macro-average.
  - (ii) A model-balanced pooled evaluation. Positives are resampled with weights that make the model mix of positives equal to the model mix of benigns, over the models used in (i).
  - claude-3-5-sonnet has only 7 attacked traces, so it is excluded from the model-controlled analyses (not from P0).
- **Model-identity probe.** On benign traces only, 5-fold CV logistic regression with the same features predicts gpt-4o vs other models. A high AUROC would mean the representation encodes the generating model, so class/model confounding could drive within-AgentDojo detection.

### 2. Train/test construction
- **AgentDrift→AgentDojo (transfer).** Identical to the existing R4. Train on AgentDrift `data/train.jsonl` + `data/test.jsonl` (AgentDrift commit `014a514f`), then score every trace in P0 (or C1). No AgentDojo data is used in training.
- **AgentDojo→AgentDojo (within).** 5-fold cross-validation producing out-of-fold (OOF) scores for every trace in the population. Two grouping schemes; the same user task never appears in both train and test:
  - **UT (primary): user-task-disjoint.** Group = suite × user_task_id.
  - **IT (strict): injection-goal-disjoint.** Attacked traces are grouped by suite × injection_task_id and benign traces by suite × user_task_id. A benign run of `injection_task_k` shares its group with attacks whose goal is `injection_task_k`. Test folds therefore contain *unseen attack goals*.
  - Group-to-fold assignment is a seeded random permutation of groups with round-robin assignment to 5 folds.
- **Paired comparison.** Both methods produce a score for exactly the same traces, so every within-vs-transfer comparison is **paired**.

### 3. Feature representation (unchanged)
`featurise.vectorise`, used as in R4:
- LOGREG-full = 10 vocabulary-independent behavioural features + bag-of-tools over the benign vocabulary *of the training data*.
- LOGREG-agnostic = the 10 behavioural features only.

Detector: `LogisticRegression(max_iter=1000, class_weight="balanced")` after `StandardScaler` fitted on the training data. The architecture and hyperparameters are not changed.

### 4. Seed procedure
- **Within-AgentDojo:** 5 CV repetitions, fold-assignment seeds 0–4. LogReg is deterministic given its data.
- **Transfer:** one deterministic model.
- **Bootstrap:** each replicate picks one CV repetition uniformly at random, so fold-assignment variability is included in the intervals. Bootstrap master seed 12345.

### 5. Prevalence procedure
- **Native.** P0: 2,294/2,686 = 85.4% attacked. C1: 2,294/2,621 = 87.5%.
- **Controlled.** p ∈ {10%, 5%}. Keep n_neg = all benign (392 in P0, 327 in C1) and set n_pos = round(p/(1−p) · n_neg):
  - P0: 44 at 10%, 21 at 5%.
  - C1: 36 at 10%, 17 at 5%.
- **1% and 2% are not reported for AgentDojo.** They would need 4–8 positives against 392 benign, which is not statistically meaningful.

### 6. Bootstrap procedure
- **Stratified bootstrap over the final test predictions, B = 2,000.** Each replicate draws n_neg benign indices with replacement from the benign traces and n_pos attacked indices with replacement from *all* attacked traces, at native or controlled class counts. The model-balanced arm uses weighted draws for the attacked traces.
- **Estimates.**
  - Native: point estimate = metric on the full population (averaged over CV repetitions); CI = 2.5/97.5 percentiles of the replicates.
  - Controlled: estimate = mean over replicates; CI = 2.5/97.5 percentiles.
- **Paired differences.** Each replicate uses the same resampled indices for both methods. We report Δ = within − transfer with its percentile CI and the fraction of replicates with Δ ≤ 0.
- **Metrics.** AUROC; AUPRC (chance level = prevalence); precision, recall, FPR and FNR at two thresholds fixed *without* test data:
  - τ = 0.5, the default decision rule of balanced LogReg;
  - τ_r80, the threshold at which **training-data** recall on attacked traces is 0.80 (from in-sample training scores).

### 7. Pre-registered decision rules
- **"Reasonable within-AgentDojo detection"** for a population/scheme means **all** of the following:
  - AUROC ≥ 0.80;
  - lower 95% bound of AUROC ≥ 0.70;
  - AUPRC@10% ≥ 0.30 (≥ 3× chance);
  - lower 95% bound of AUPRC@10% > 0.10.
- **"Near chance"** means the AUROC CI includes 0.5, or AUROC < 0.60.
- **Case A.** Within detection is reasonable on the primary scheme (UT) **for P0 and for C1**, the transfer is near chance, and the paired ΔAUROC CI excludes 0.
- **Case B.** Within detection is not reasonable on UT for P0 or for C1, and the transfer is also poor.
- **Case C.** Within detection is reasonable only in controlled populations or strata (C1, within-model, model-balanced) and not in P0.
- **Mixed or other outcomes are reported as observed, without forcing a case.** This includes UT reasonable but IT not reasonable, which would mean detection works for seen attack goals but not unseen ones.

The 0.80 / 0.30 thresholds are judgement calls fixed before seeing the results. They are not standard values.

### 8. Experiment 2 (audit of R1–R3), same protocol
- **Evaluation pool E** per rung = benign + hard_negative + attacked traces of the test set (`failed_attack` excluded, as in the original subsampling). The exact counts and native prevalence are recorded for each rung.
- **AUROC and AUPRC are computed on the same pool E.** Previously AUROC used the full test set, including `failed_attack`.
- **Stratified bootstrap** (B = 1,000) at native prevalence and at p ∈ {1, 2, 5, 10}%, with n_neg = |E_neg| and n_pos = round(p/(1−p)·n_neg).
- **Thresholds.** τ_r80 comes from training data, replacing the previous threshold that was chosen on the test set. For LOGREG, τ = 0.5 is also reported.
- **Paired Δ** between detectors within a rung, since they share test examples.
- **Independent-bootstrap Δ** between rungs (R2−R1, R3fold−R1), since the test sets differ. These differences are **not** paired.
- The old and new numbers are compared, and a result is described as "changed" only if the corrected computation actually moves it.

---

## PART 2: RESULTS (computed after Part 1 was fixed; `r4_control_results.json`, `r1r3_corrected.json`)

**Format.** Estimate [95% bootstrap percentile CI].
- Native prevalence: the estimate is on the full population (for within-AgentDojo, the mean over the 5 CV repetitions).
- Controlled prevalence: the mean over replicates.
- Thresholds never use test data.

**Sanity check passed.** Transfer AUROC on P0 reproduces the earlier R4 values exactly (full 0.440, agnostic 0.447).

### 7–8. AgentDojo-only (within) vs AgentDrift→AgentDojo (transfer), LOGREG

**P0 (392 benign, 2,294 attacked; native prevalence 0.854)**

| method | AUROC (native) | AUPRC, native (chance 0.854) | AUPRC @10% (n_pos 44 / n_neg 392; chance 0.101) | AUPRC @5% (21/392; chance 0.051) |
|---|---|---|---|---|
| within UT full | **0.855 [0.830, 0.880]** | 0.964 [0.954, 0.973] | **0.428 [0.311, 0.582]** | 0.290 [0.171, 0.463] |
| within UT agnostic | 0.798 [0.770, 0.827] | 0.947 [0.936, 0.957] | 0.306 [0.213, 0.422] | 0.199 [0.106, 0.342] |
| within IT full (unseen attack goals) | 0.766 [0.731, 0.813] | 0.943 [0.929, 0.959] | 0.324 [0.218, 0.464] | 0.209 [0.112, 0.354] |
| within IT agnostic | 0.771 [0.729, 0.810] | 0.937 [0.920, 0.953] | 0.263 [0.179, 0.389] | 0.163 [0.092, 0.295] |
| transfer full | **0.440 [0.407, 0.471]** | 0.818 [0.804, 0.832] | **0.089 [0.074, 0.113]** | 0.048 [0.037, 0.072] |
| transfer agnostic | 0.447 [0.414, 0.479] | 0.816 [0.803, 0.831] | 0.089 [0.075, 0.111] | 0.048 [0.037, 0.070] |

**Operating points.** τ comes from training data. FPR and recall do not depend on prevalence; precision is shown at 10%.

| method | τ=0.5: recall / FPR / FNR | τ_r80: recall / FPR | precision @10% (τ_r80) |
|---|---|---|---|
| within UT full | 0.806 / 0.225 / 0.194 | 0.749 / 0.173 | 0.330 |
| transfer full | 0.284 / 0.347 / 0.716 | 0.283 / 0.347 | 0.084 |

**C1 (injection-task benigns removed: 327 benign, 2,294 attacked; native prevalence 0.875)**

| method | AUROC | AUPRC @10% (36/327) |
|---|---|---|
| within UT full | **0.867 [0.843, 0.894]** | **0.498 [0.344, 0.675]** |
| within IT full | 0.742 [0.677, 0.797] | 0.318 [0.199, 0.481] |
| transfer full | 0.444 [0.407, 0.478] | 0.088 [0.073, 0.115] |

1% and 2% prevalence are not reported for AgentDojo, because they would need 4–8 positives against 327–392 benign traces.

### 9. Paired differences, within − transfer (same traces, same bootstrap indices)

| population / comparison | ΔAUROC | ΔAUPRC @10% | P(Δ≤0) |
|---|---|---|---|
| P0: UT full − transfer full | **+0.415 [+0.374, +0.457]** | +0.339 [+0.217, +0.494] | 0.000 |
| P0: UT agnostic − transfer agnostic | +0.351 [+0.308, +0.394] | +0.216 [+0.120, +0.338] | 0.000 |
| P0: IT full − transfer full | +0.326 [+0.276, +0.382] | +0.235 [+0.122, +0.376] | 0.000 |
| C1: UT full − transfer full | +0.424 [+0.382, +0.468] | +0.410 [+0.246, +0.590] | 0.000 |
| C1: IT full − transfer full | +0.298 [+0.224, +0.366] | +0.230 [+0.108, +0.389] | 0.000 |

### 10. Confound analysis

**Injection-task benigns (65 of 392).** No data were removed from P0; C1 is reported alongside it.
- **Original (P0) vs controlled (C1).** Within UT: AUROC 0.855 → 0.867 and AUPRC@10% 0.428 → 0.498. Transfer: essentially unchanged (0.440 → 0.444).
- **What changes.** At τ=0.5 the within detector flags 35% of these 65 traces, compared with 21% of other benigns and 80% of attacks. They are partly confusable with attacks, as their definition implies. Including them makes the within result slightly *conservative*.
- **What does not change.** The within-vs-transfer gap (ΔAUROC +0.415 → +0.424) and the Case decision. The transfer model scores these traces like other benigns (flag rate 0.37 vs 0.34).

**Model/class confound (81% of attacks come from gpt-4o).**
- **Counts (benign / attacked).** gpt-4o 92 / 1,866; gpt-4-turbo 82 / 180; gemini-1.5-pro 79 / 107; gemini-2.0-flash 55 / 134; claude-3.5-sonnet 84 / 7. Claude is **excluded** from the model controls because it has only 7 positives.
- **Within-model evaluation (native prevalence):**

| model (benign/attacked) | within UT full AUROC | transfer full AUROC | paired Δ |
|---|---|---|---|
| gemini-1.5-pro (79/107) | 0.792 [0.716, 0.862] | 0.593 [0.509, 0.677] | +0.199 [+0.089, +0.307] |
| gemini-2.0-flash (55/134) | 0.803 [0.725, 0.877] | 0.640 [0.544, 0.729] | +0.165 [+0.052, +0.269] |
| gpt-4-turbo (82/180) | 0.849 [0.788, 0.906] | 0.485 [0.407, 0.562] | +0.366 [+0.269, +0.461] |
| gpt-4o (92/1,866) | 0.846 [0.805, 0.885] | 0.447 [0.383, 0.507] | +0.399 [+0.327, +0.469] |
| **macro average** | **0.823** | **0.542** | n/a |

- **Model-balanced pooled evaluation.** Positives are reweighted to the benign model mix, over the 4 eligible models (308 benign).
  - Within UT full: AUROC **0.812 [0.781, 0.843]**; AUPRC@10% (34/308) **0.358 [0.237, 0.524]**.
  - Transfer: AUROC **0.526 [0.495, 0.558]** (CI includes 0.5); AUPRC@10% 0.122 [0.087, 0.180].
  - Paired ΔAUROC: +0.286 [+0.242, +0.332].
- **Model-identity probe** (benign traces only, gpt-4o vs the rest, 92 vs 300). Full features: AUROC 0.448 [0.380, 0.517]. Agnostic features: 0.615 [0.551, 0.680]. The model-identity signal is weak.
- **What changes.** The pooled transfer AUROC being **below** 0.5 (0.440, CI excludes 0.5) is largely a model-composition effect:
  - under model balancing, transfer is at chance (0.526);
  - within individual models, transfer ranges from 0.447 (gpt-4o) to 0.640 (gemini-2.0-flash).
  - The earlier wording "at or below chance" should become: "near chance, with the sub-0.5 pooled value driven by the gpt-4o-dominated positive class".
- **What does not change.**
  - Within-AgentDojo detection stays at ≥0.79 AUROC in every eligible model, and at 0.812 when model-balanced.
  - The within-vs-transfer gap is positive in every model, and all CIs exclude 0.
- **Limitations.**
  - Training is still pooled across models; the control is at the evaluation level only. Per-model training was not attempted, because no model has more than 92 benign traces.
  - DoS-type attack goals (about 100 positives) were not separated in this run.

### 11. Interpretation under the pre-registered cases

- **Primary scheme (UT), P0 and C1.**
  - Within detection meets every "reasonable" criterion. P0: AUROC 0.855 (lower bound 0.830), AUPRC@10% 0.428 (lower bound 0.311). C1: 0.867 / 0.498.
  - Transfer is near chance (AUROC 0.440 < 0.60).
  - The paired ΔAUROC CI excludes 0 (+0.415 [+0.374, +0.457]).
  - **→ Case A.**
- **The Case A outcome survives the model controls.** Model-balanced within detection (0.812 / AUPRC@10% 0.358) still meets the criteria, and within-model AUROC is 0.79–0.85 in all four eligible models.
- **Mixed element, reported as observed.**
  - Under the strict scheme (IT, unseen attack goals), within detection does **not** meet the pre-registered criterion of AUROC ≥ 0.80 (P0 0.766, C1 0.742). It remains far above transfer (paired Δ +0.33 / +0.30; CIs exclude 0).
  - The agnostic-only within detector (UT) is also just below the AUROC criterion (0.798).

**Supported statement (Case A wording).**
> *Cross-benchmark transfer is poor under the tested distribution shift. The same representation and detector, trained on AgentDojo with user-task-disjoint cross-validation, reaches AUROC 0.855 [0.830, 0.880] on AgentDojo traces. The model trained on AgentDrift is near chance on the same traces (0.440 [0.407, 0.471]; paired Δ +0.415 [+0.374, +0.457]).*

The representation is therefore **not** inadequate for AgentDojo in general. Within AgentDojo it is moderately effective for seen attack goals, and weaker for unseen ones.

**Still not supported.**
- Which shift causes the transfer failure. Benchmark, tool ecosystem, data generation and synthetic→recorded are all confounded in R4.
- Anything about unseen attack *types*.
- Anything about published detectors.

---

## Experiment 2: corrected R1–R3 evaluation (`rerun_r1r3.py`, `r1r3_corrected.json`)

**Pools.** Pool E = benign + hard_negative + attacked traces of the test set (failed_attack excluded).

| rung | benign / hard_negative / attacked | native prevalence | n_neg | n_pos at 1 / 2 / 5 / 10% |
|---|---|---|---|---|
| R1 | 600 / 225 / 831 | 0.502 | 825 | 8 / 17 / 43 / 92 |
| R2 | 525 / 204 / 775 | 0.515 | 729 | 7 / 15 / 38 / 81 |
| R3, each held-out domain except banking | 680 / 255 / 1,020 | 0.522 | 935 | 9 / 19 / 49 / 104 |
| R3, banking | 680 / 255 / 626 | 0.401 | 935 | 9 / 19 / 49 / 104 |

**What changed compared with the earlier sweep:**
- **AUPRC means:** essentially unchanged (≤0.02 at 5–10%; ≤0.07 at 1–2%). Example: R1 LOGREG AUPRC@2% 0.251 → 0.261 [0.109, 0.456].
- **Intervals:** **wider**, because negatives are now resampled too. Example: R1 LOGREG AUPRC@10% [0.482, 0.605] → [0.434, 0.621].
- **AUROC** (now on the same pool as AUPRC): slightly higher. Examples: R1 LOGREG 0.876 → 0.879 [0.862, 0.895]; R3-banking 0.779 → 0.828; IForest +0.02 to +0.07.
- **Operating points:** now use training-derived thresholds. Previously the threshold was chosen on the test set.

**Rung differences** (LOGREG, independent bootstrap; the test sets differ, so these are NOT paired):

| comparison | ΔAUROC (native) | ΔAUPRC @2% | ΔAUPRC @10% |
|---|---|---|---|
| R2 − R1 | +0.023 [+0.001, +0.046] | +0.115 [−0.186, +0.393] | +0.093 [−0.041, +0.226] |
| R3 medical − R1 | −0.133 [−0.160, −0.104] | −0.176 [−0.379, −0.007] | −0.274 [−0.385, −0.160] |
| R3 banking − R1 | −0.050 [−0.075, −0.024] | −0.114 [−0.340, +0.078] | −0.136 [−0.277, +0.002] |
| R3 web − R1 | −0.016 [−0.038, +0.005] | +0.005 [−0.242, +0.256] | −0.020 [−0.152, +0.112] |
| R3 email − R1 | −0.139 [−0.166, −0.112] | −0.150 [−0.363, +0.056] | −0.253 [−0.371, −0.133] |
| R3 coding − R1 | −0.064 [−0.088, −0.039] | −0.096 [−0.316, +0.121] | −0.139 [−0.270, −0.014] |

- **Revised R3 statement.**
  - An AUROC decrease is detected in 4 of 5 held-out domains (not web).
  - An AUPRC@10% decrease is detected in 3 of 5 (medical, email, coding).
  - At 2% prevalence, only the medical decrease has a CI that excludes 0.
  - **The earlier claim that point estimates are below R1 "in all five folds at every prevalence" is no longer true** (web at 2%: +0.005).
- **R2.** No degradation is detected. AUROC is slightly higher (+0.023; the CI just excludes 0).
- **Detector comparisons** (paired, same examples). LOGREG beats MARKOV and IFOREST in every rung; all ΔAUROC CIs exclude 0.

**New verified observation: the threshold does not transfer under domain shift** (LOGREG, τ_r80 from training data):

| rung | recall | FPR | alerts per 1,000 @2% |
|---|---|---|---|
| R1 | 0.773 | 0.192 [0.166, 0.219] | 204 |
| R2 | 0.790 | 0.149 | 162 |
| R3 medical / banking / web / email / coding | 0.83 / 0.97 / 0.99 / 0.96 / 0.97 | 0.50 / 0.66 / 0.71 / 0.72 / 0.67 | 509 / 666 / 712 / 728 / 677 |

In the held-out domains, the training-derived threshold flags most traces (FPR 0.50–0.72). This happens even where ranking is largely preserved (web: AUROC 0.862 vs 0.879). It is a calibration / operating-point shift, distinct from the change in ranking (AUROC/AUPRC).
