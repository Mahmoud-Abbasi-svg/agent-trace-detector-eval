# GRU_PREREGISTRATION: one stronger sequence detector for the evaluation ladder

*Written 25 Sep 2026, before implementation. Pre-registers a single small GRU detector, its training and evaluation, and the interpretation rules — fixed before any GRU result is seen. Nothing is implemented here. `evalkit.py`, the A/B/C ablation, and all existing result files are unchanged.*

---

## 1. Motivation
Every result so far uses simple detectors (Markov, logistic regression, isolation forest). The paper's question is whether *conventional in-distribution evaluation predicts behaviour under imbalance and shift*. A reviewer will reasonably ask whether the pattern (strong R1, domain-shift degradation, near-chance cross-benchmark transfer) is a property of linear models or of the evaluation regime. We therefore add **one** learned sequence model — a small GRU over the tool-name sequence — and run it through the identical ladder. The goal is not a better detector; it is a robustness check on the findings. We do **not** reproduce DriftNet (no public code/weights; it needs a frozen sentence-transformer over step text, a different preprocessing pipeline — see the DriftNet availability report).

## 2. Exact architecture (fixed; no sweep)
Single model, chosen to have very few hyperparameters. All values below are fixed before running.
- **Token embedding:** `nn.Embedding(V+2, 32)` over a tool vocabulary of size V (benign-training tool names) plus two reserved ids: `<PAD>`=0, `<UNK>`=1.
- **Recurrent core:** `nn.GRU(input_size=32, hidden_size=64, num_layers=1, batch_first=True)`, unidirectional. Sequences are padded to length 16 (AgentDrift max 11, AgentDojo max ≈ 11; a trace longer than 16 is truncated to its **last** 16 steps, recorded if it ever occurs). Padding is masked via `pack_padded_sequence`, so `<PAD>` never contributes.
- **Trajectory representation:** the final (last non-pad) GRU hidden state, `h ∈ R^64`.
- **Behavioural features (explicit design choice):** the **same 10 vocabulary-independent behavioural features** used in representation B/C (`featurise.vectorise`), standardised with a `StandardScaler` fitted on training data, concatenated to `h` → `R^74`.
  - This makes the primary GRU the learned analogue of the combined detector **C** (tool sequence + behavioural), i.e. the strongest baseline, giving the fairest "stronger detector" comparison.
- **Classifier:** a single `Linear(74 → 1)` + sigmoid. (One linear layer, to keep parameters minimal.)
- **Parameter count:** ≈ (V+2)·32 + GRU(3·(32·64+64·64+64)) + 74. For V≈103 (R1), ≈ 3.4k (embed) + ≈ 18.8k (GRU) + 75 ≈ **22k trainable params**. Small by design.
- **Optional secondary (only if needed, §9):** `GRU-tool-only` — identical but classifier is `Linear(64→1)`, no behavioural concat. Run **only** if the primary reproduces the R3 combined-vs-behavioural trade-off and we need to attribute it. Not part of the primary comparison; still one fixed config, no sweep.

## 3. Input representation
Per trajectory: the ordered list of `tool` names → token ids (train-benign vocab; unseen → `<UNK>`; padded to 16). Plus the 10 behavioural features at trajectory level. **No** `thought`/`observation`/`args` text embedding, **no** pretrained language model, **no** step-level targets. Identical field usage to the A/B/C pipeline, so the four-rung ladder stays coherent.

## 4. Training procedure (fixed)
- **Loss:** weighted BCE, `pos_weight = n_neg_train / n_pos_train` (mirrors `class_weight="balanced"`).
- **Optimizer:** Adam, lr 1e-3, weight decay 1e-4. Batch size 64. Max 30 epochs.
- **Early stopping:** on **validation AUPRC**, patience 5, restore best epoch.
  - **Validation source:** R1 uses the provided `data/val.jsonl` (1,880 traces). Every other rung (R1-world-disjoint, R2, R3 folds, R4-transfer training, R4 within-AgentDojo folds) carves a **stratified 10% of that rung's training set** as validation, with a fixed seed; validation is never a test trace and never crosses a world/domain/task boundary that the rung holds out (i.e. the 10% is drawn from training only).
- **Seeds:** because NN training is stochastic, each (rung, split) is trained with **3 training seeds** {0,1,2}. The 3 resulting test-score vectors are the "repetitions" passed to `evalkit` (its bootstrap samples one repetition per replicate, so seed variance enters the CIs). Point estimates are the mean over the 3 seeds.
- **Determinism:** `torch.manual_seed`, CPU, `torch.use_deterministic_algorithms(True)` where supported.
- **No test labels touched in training or model selection.**

## 5. Evaluation protocol (reuse the corrected framework)
Run the identical ladder and pools already defined:
- **R1** standard stratified (train `data/train`, test `data/test`; pool E = benign+hard_negative+attacked).
- **R1-world-disjoint** (the Experiment 1b split; 5 world-assignment seeds × 3 training seeds; report across-seed AUROC and seed-0 bootstrap).
- **R2** task-disjoint (`data_taskdisjoint`).
- **R3** leave-one-domain-out (5 folds; each fold reported separately + aggregate).
- **R4** AgentDrift→AgentDojo transfer, plus the established **within-AgentDojo** UT reference (5 CV folds), on populations **P0**, **C1**, and **model-balanced**, exactly as in `r4_control.py`.
- Bootstrap via `evalkit.bootstrap` (B = 1000 for R1–R3, 2000 for R4), stratified over test predictions, at native prevalence and p ∈ {1,2,5,10}% for R1–R3 and {native,10,5}% for R4 (as in the existing runs; 1–2% omitted for AgentDojo).
- **Thresholds:** τ_r80 from **training** scores (recall-0.80 in-sample) and τ=0.5. Never from test.

## 6. Metrics
Per rung / representation / prevalence: AUROC, AUPRC (with `evalkit` bootstrap CIs), precision, recall, FPR, FNR at τ_r80 and τ=0.5, and exact n_pos / n_neg. For R3 and R4, report **tool-vocabulary coverage** (tool-occurrence coverage, distinct-tool coverage, % traces with any unseen tool, % with zero known tools) next to every result — coverage is 100% for R1/R1-wd, 0% for R3, ≈7.5% for R4 (already measured).

## 7. OOV handling (explicit)
Unseen tools map to the single shared `<UNK>` id (not to zero/ignored). Consequence, stated up front:
- Under **R3** (0% coverage) the GRU receives a sequence of `<UNK>` tokens of the trajectory's true length, so it **retains trajectory-length and position/repeat structure** plus the 10 behavioural features. It is therefore **not** structurally constant like the linear tool-only representation A (which collapses to AUROC 0.5). GRU R3 results are interpreted as "sequence + behavioural signal with tool identity masked", and are **not** directly compared to linear A's structural 0.5.
- Under **R4** (~7.5% coverage) most tokens are `<UNK>`; the model sees mostly length/position + behavioural signal plus the few known tools.
- `<UNK>` is part of the model's fixed design, not a separate sensitivity arm.

## 8. Comparison strategy
- **Primary comparator:** the combined logistic regression **C** (the strongest existing baseline), on the same test examples → **paired** `evalkit` ΔAUROC and ΔAUPRC (GRU − C) per rung and prevalence.
- **Secondary context:** report GRU alongside A and B for completeness, but the headline comparison is GRU vs C.
- **Shift comparison:** for the GRU alone, report independent-bootstrap ΔAUROC/ΔAUPRC of each shift rung vs its own R1 (R2−R1, each R3 fold − R1, R4 transfer − R1), exactly as done for LogReg.
- **A/B/C results are not modified.** The GRU is added as a fourth detector.

## 9. Pre-specified interpretation rules (fixed before running)
"Detected" = 95% bootstrap CI excludes 0 (differences) or 0.5 (AUROC level). Judge the GRU on the same coverage-aware rules as the ablation (PRE_ABLATION_DESIGN Rules 0–5): always report coverage; never read the R3/R4 tool-masking as intrinsic representation failure.

- **Outcome 1 — same qualitative pattern.** Strong R1 (GRU AUROC CI lower bound > 0.7), detected degradation R1→R3 in ≥3/5 folds, and near-chance R4 transfer (AUROC CI includes 0.5 or below on P0 and model-balanced). → *Strengthens the claim that the evaluation regime matters beyond the specific linear detector.*
- **Outcome 2 — GRU substantially improves R3/R4.** GRU R4 transfer AUROC CI lower bound > 0.6 (P0 **and** model-balanced), or R3 shows no detected degradation in ≥3/5 folds. → *Report that robustness depends on detector architecture; do not claim the shift effect is detector-independent.* Explicitly revise the manuscript's generality claim.
- **Outcome 3 — GRU also fails R4** (near chance) while strong in R1. → *Strengthens the evidence that cross-benchmark transfer is particularly hard under the tested shift, still without universal claims* (it is two detector families, not "all detectors").
- **Outcome 4 — inconsistent** (e.g. strong R3 but variable across folds/seeds, or large seed variance). → *Report the inconsistency and the seed variance; do not force a common narrative.*
- **Within-AgentDojo check (Case A logic).** If the GRU trained within AgentDojo detects attacks (AUROC CI lower bound > 0.7) while transfer is near chance, that reproduces the R4_CONTROL Case A reading for a learned model. If within-AgentDojo is also near chance, report that the GRU representation is inadequate for AgentDojo rather than a transfer effect.
- **Coverage caveat carried throughout:** because `<UNK>` lets the GRU keep length/behavioural signal under R3/R4, a GRU that beats linear A under shift is **not** evidence that "tool identity transfers"; it reflects the non-tool signal the GRU retains.

## 10. Computational budget and stopping rule
- **Environment:** PyTorch 2.11.0+cpu confirmed available (8 threads); ≈22k-param model; trajectories ≤16 steps; datasets ≤~12k traces. A single fit is seconds on CPU.
- **Total fits:** R1 (1) + R1-wd (5 splits) + R2 (1) + R3 (5 folds) + R4 transfer (1) + R4 within-AgentDojo (5 folds × {P0,C1,model-balanced} = up to 15), each × 3 training seeds ≈ **~85 fits**. Estimated wall-clock: well under an hour on CPU. Bootstrap reuses `evalkit` and is unchanged.
- **Stopping / abort rules (pre-specified):**
  - If a fit produces NaN/Inf loss: halve lr **once** (to 5e-4) and retry that fit; if it recurs, abort the run and report the instability rather than tuning further.
  - No hyperparameter search. The only fixed values are those in §2/§4; none is adjusted based on test results.
  - If total wall-clock exceeds ~2 hours (e.g. determinism flags slow it), reduce training seeds from 3 to 2 (recorded), keep everything else.
- **Reproducibility:** a `gru_detector.py` + `gru_ladder.py` will write `gru_results.json`; seeds, versions, and dataset commits recorded as in the other scripts. (To be created only after this pre-registration is approved.)

---
*On approval, implement exactly this specification. Any deviation forced by implementation reality will be recorded in the results file and here, not silently applied.*

---

## Correction note (v0.4 revision), 27 September 2026

This note is appended after the fact and is not part of the plan. The plan text above is unchanged. The file as it stood before this note, with its original modification time (2026-09-25 15:02:47), is kept as `archive_v0.3/GRU_PREREGISTRATION.md` (SHA-256 `2bba95363bf4e9c5c2b8482a575b49c1e0557787ed0b6d99c50c69f7383fc5b7`).

1. **Status of this document.** Despite its file name, this document is an internal pre-specified analysis plan. It was not registered with any external registry, and no independent timestamp exists for it. Its modification time shows only that it was last written after all linear results (LR-A, LR-B and LR-C in R1, R3 and R4, including LR-C ranking below LR-B in all five held-out domains) and before the GRU code and results.
2. **Deviations in the execution.** The executed GRU runs departed from this plan in the following ways.
   - R1 early stopping used a stratified 10% hold-out of the R1 training data, not `data/val.jsonl` (Section 4 of this plan).
   - Within AgentDojo the GRU was trained once per cross-validation repetition, with training seeds 0 to 4, not with three seeds per fit.
   - R1-wd was run on split seed 0 only, not on five split seeds with three training seeds each.
   - Every fit clips the gradient norm at 1.0, which the plan does not mention.
   - Independent-bootstrap shift deltas were replaced by the non-overlap of intervals.
   - The model-balanced population was evaluated only at 10% prevalence, not at native counts.
   - Outcome 4 was dismissed in `GRU_RESULTS.md` without the seed variance being reported.
   - A diagnostic variant with a linear head on the behavioral features (named GRU-B in v0.3) was added after the GRU results were known and is post hoc.
   - Contrary to the last line of this plan, these deviations were not recorded in the results file at the time.
3. **Where the deviations are documented.** Every item, with file and line references and its effect on the conclusions, is listed in `revision/DEVIATIONS.md` (Part A for what was known when this plan was written, Part B items V15 to V24 for the deviations, and Part D item X14 for gradient clipping), and in Table S3 of the v0.4 supplementary material. `GRU_RESULTS.md` carries a matching correction.
