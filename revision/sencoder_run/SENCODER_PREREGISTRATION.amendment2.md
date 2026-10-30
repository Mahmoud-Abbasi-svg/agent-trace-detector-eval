# SENCODER_PREREGISTRATION: a published-architecture detector for the evaluation ladder

*Written 8 Oct 2026, before implementation. Pre-registers a reimplementation of the DriftNet architecture [Pinjari and Saint-Germain, arXiv:2609.10892], its training, its evaluation and the interpretation rules, all fixed before any result is seen. Nothing is implemented here. `evalkit.py`, the A/B/C ablation, the GRU results and every existing result file are unchanged by this document.*

---

## 1. Motivation

Juan M. Núñez V. asked, in his review of 8 Oct 2026, for a comparison against at least one recent state-of-the-art detector. The v0.4 manuscript states plainly that its claims concern our own baselines and one content reference, not the published detectors of Section II. That limitation is real and it is the weakest point in the generality argument.

There is also a scientific reason to add this particular detector rather than any other, and it is stronger than the reviewer's reason.

Section VI-D explains the R3 ranking reversals by a mechanism. Unseen tools reach inputs that training never touched. For LR-C the tool block becomes an all-zero raw vector that the scaler maps to a constant, which adds an offset of +1.653 to +1.899 logits. For the GRU the untrained `<UNK>` embedding is shrunk by weight decay and pushes held-out negatives 6.291 logits higher than training negatives. Dropout, which trains those inputs, largely removes both effects.

**A frozen sentence encoder has no untrained input for an unseen tool.** It reads the tool name as text and places it somewhere in an embedding space that was fixed before our training began. If the mechanism in Section VI-D is the right explanation, this detector should not show the offset and should not show the reversal, while still losing accuracy for the ordinary reason that a new domain is harder. If it does show the reversal anyway, our mechanism account is incomplete and the manuscript must say so.

So this is a falsification test of our own central mechanism claim, and it answers the review comment as a by-product. We state the prediction before running it (Section 9, Outcome 1).

**What this is not.** It is not DriftNet. No code or weights were released (confirmed in the v0.3 availability check), so this is a reimplementation from the architecture description in the paper, with two declared departures (Section 11). Every mention in the manuscript must read as a reimplementation, never as DriftNet itself, and no number here may be presented as DriftNet's reported performance. We also do not tune it. A reimplementation that we tuned and our baselines that we did not would be an unfair comparison in our own favour.

## 2. Exact architecture (fixed, no sweep)

Per step, a 772-dimensional vector. Over a trajectory, a small transformer with two heads, as described in the source paper.

- **Step text.** For step $t$, the concatenation `"<tool name> | <argument keys and values> | <observation>"`, built from the `tool`, `args` and `obs` fields, in that order, joined by `" | "`. Arguments are rendered as `key=value` pairs in the order the record stores them. Missing fields contribute an empty string.
- **The `thought` field is deliberately excluded.** AgentDrift records it and AgentDojo does not. Including it would make the step text of the two benchmarks structurally different and would confound every transfer result in R4 with a missing-field artefact. Excluding it costs AgentDrift information and keeps the representation identical across benchmarks, which is the property R4 needs. This choice is fixed here, before any run, and is declared again as D6 in Section 11.
- **Frozen encoder.** `sentence-transformers/all-mpnet-base-v2`, mean pooling, output 768 dimensions, L2-normalised. **Weights are frozen and never updated.** Maximum sequence length left at the model default of 384 word pieces. Truncation beyond that is recorded and reported.
- **Four scalar features.** Appended to each step vector, giving 772. See Section 3 for which four and why.
- **Trajectory encoder.** `nn.TransformerEncoder` of 2 layers, `d_model=256`, 4 attention heads, feed-forward 512, dropout 0.1, batch first, with a learned linear projection from 772 to 256 and a learned positional embedding over 16 positions. Sequences are padded to 16 steps with a key-padding mask. A trajectory longer than 16 is truncated to its **last** 16 steps, and any occurrence is recorded.
- **Detection head.** Mean pooling over non-pad positions, then `Linear(256 -> 1)`. This produces the trajectory score used everywhere in the ladder.
- **Localisation head.** `Linear(256 -> 1)` applied per position, producing a step-level logit. Present because the source architecture is dual-head and its auxiliary loss may change detection behaviour. Its outputs are **not** evaluated in this study.
- **Parameter count.** Projection 772·256 + 256 ≈ 198k, positional 16·256 ≈ 4k, two encoder layers ≈ 1.58M, heads 514. Total ≈ **1.79M trainable parameters**, which matches the ≈1.8M reported for DriftNet. The frozen encoder's 110M parameters are not trained and are not counted.

## 3. Input representation and the four scalar features

The source paper states that each step vector is a frozen `all-mpnet-base-v2` embedding plus four hand-crafted features, and does not enumerate them. We cannot recover them, so we **substitute four of our own and declare it** (Section 11, D1).

The four are chosen to be vocabulary-independent and already defined in `featurise.py`, so that nothing new enters the pipeline.

1. Observation length in characters, log1p transformed.
2. Argument length in characters, log1p transformed.
3. Step index divided by trajectory length, in [0, 1].
4. The state-change keyword flag for the tool name.

All four are standardised with a `StandardScaler` fitted on the training steps of the rung in question, never on test.

**A caveat we commit to in advance.** Feature 1 is the observation-length cue that Section VII-B shows the attacker writes. Including it means this detector inherits that cue, exactly as LR-B does. We therefore also run **one** sensitivity arm, `SE-noobs`, identical but with feature 1 removed, on R1, R3 and R4 only. This arm is pre-specified here and is not an exploratory addition.

## 4. Embedding procedure and caching

- Embeddings are computed **once** for every step of AgentDrift (train, val, test, and the task-disjoint split) and of `agentdojo_traces.jsonl`, then cached to disk as float32 arrays keyed by trajectory id and step index.
- Approximately 71k AgentDrift steps and 14,079 AgentDojo steps, so roughly **85k step texts**.
- The cache is written under `revision/cache/sencoder/` with a manifest recording the model name, its revision hash from the Hugging Face hub, the pooling mode, the maximum sequence length, the truncation count and the SHA-256 of each input file.
- Because the encoder is frozen, the cache is computed before any split is chosen and is shared by every rung and seed. **This is not leakage.** The encoder saw no label and no split, and its weights were fixed by a third party before this study. The manifest exists so a reader can verify that claim.

## 5. Training procedure (fixed)

- **Loss.** Weighted BCE on the detection head with `pos_weight = n_neg_train / n_pos_train`, plus 0.3 times a weighted BCE on the localisation head where step labels exist. The 0.3 is taken as a fixed auxiliary weight and is **not** tuned.
- **Step labels.** AgentDrift ships step-level labels, so the auxiliary loss is active for every AgentDrift fit. AgentDojo has none, so **within-AgentDojo fits use the detection loss alone**, which is declared as D2 in Section 11.
- **Optimiser.** Adam, learning rate 1e-4, weight decay 1e-4, batch size 32, maximum 30 epochs.
- **Early stopping.** On validation AUPRC, patience 5, restore the best epoch. **Validation is a stratified 10% carve-out of the training set of the rung in question, at every rung including R1**, drawn with a fixed seed, never crossing a boundary the rung holds out and never touching a test trace.

  *Amended 8 Oct 2026, before any run.* The first version of this plan said the validation source was the one the GRU pre-registration fixes in its Section 4, which names `data/val.jsonl` for R1, so that the two learned detectors would be selected the same way. That was wrong about what the GRU actually did. `revision/build_gru_cache.py` carves a stratified 10% of R1 training instead, which is recorded as deviation V15 of the revision. Following the GRU plan here would therefore have made the two detectors differ at R1, and the head-to-head ΔAUROC against the GRU is the comparison this detector exists to support. We match the executed GRU pipeline rather than its written plan, and declare it as D7. The 10-seed GRU re-run follows the executed pipeline for the same reason.
- **Seeds.** **10 training seeds, 0 to 9**, following the decision of 8 Oct 2026 to raise every learned detector from 3 to 10. The 10 score vectors are the repetitions passed to `evalkit`, whose bootstrap draws one repetition per replicate, so seed variance enters the intervals. Point estimates are the mean over the 10 seeds.
- **Determinism.** `torch.manual_seed`, CPU, `torch.use_deterministic_algorithms(True)` where supported.
- **No test label is touched in training or in model selection.**

## 6. Evaluation protocol

The identical ladder and pools already defined, with no change to any of them.

- **R1** standard stratified, pool E.
- **R1 world-disjoint**, 5 world-assignment seeds.
- **R2** task-disjoint.
- **R3** leave-one-domain-out, 5 folds, each reported separately.
- **R4** AgentDrift to AgentDojo transfer, with the within-AgentDojo UT control, on populations P0, C1 and model-balanced, exactly as `r4_control.py` defines them.
- **Graded coverage inside R1**, the five levels of Fig. 3 at achieved coverages 1.000, 0.775, 0.483, 0.225 and 0.000. **Masking means something different here and the difference is the point.** For LR-C a masked tool zeroes a column and for the GRU it maps to `<UNK>`, whereas for this detector the tool name is simply replaced by the literal string `unknown_tool` in the step text before embedding, which is an ordinary point in the frozen space rather than an untrained input. The same drawn masks are reused so the comparison is paired.
- Bootstrap via `evalkit.bootstrap`, B = 1000 for R1 to R3 and 2000 for R4, at native prevalence and the same prevalence grid as the existing runs.
- **Thresholds.** $\theta_{r80}$ from training scores and $\theta = 0.5$. Never from test.

## 7. Metrics

Per rung, representation and prevalence: AUROC, AUPRC with bootstrap intervals, precision, recall, FPR and FNR at both thresholds, and exact positive and negative counts. Tool-vocabulary coverage is reported next to every result as before, even though this detector has no tool vocabulary of its own, because the comparison against LR-C and the GRU is what the coverage axis is for.

Two diagnostics specific to this detector, both pre-specified.

- **Offset analogue.** The mean detection logit on held-out negatives minus that on training negatives, per R3 fold and seed. This is the quantity that is +1.653 to +1.899 for LR-C and +6.291 for the GRU. For a frozen encoder we predict it to be small.
- **Embedding drift.** The mean cosine distance between held-out step embeddings and their nearest training step embedding, per rung, so that a loss of accuracy under shift can be separated from an input the model was never trained to handle.

## 8. Out-of-vocabulary handling, stated explicitly

There is none, and that is the experimental point. An unseen tool name is text, so the frozen encoder maps it into the same space as every seen tool name, usually near semantically similar names. The model therefore meets a **covariate shift in a trained input space** rather than an **untrained input**. Every comparison against LR-C and the GRU under R3, R4 and the graded sweep must be read with that asymmetry stated, in both directions. A sentence encoder that holds up under shift is not evidence that it is a better detector, only that its input representation degrades gracefully.

## 9. Pre-specified interpretation rules, fixed before running

"Detected" means the 95% bootstrap interval excludes 0 for a difference or 0.5 for an AUROC level. Holm correction over the five R3 folds, as elsewhere. The coverage-aware rules of `PRE_ABLATION_DESIGN` apply unchanged.

- **Outcome 1, the mechanism survives.** The offset analogue is small, under +0.5 logits in at least four of five R3 folds, **and** neither reversal rule is met against LR-B. Reading: the Section VI-D mechanism is supported, because removing the untrained input removes the effect. The manuscript keeps its mechanism claim and gains a third detector family as evidence.
- **Outcome 2, the mechanism is incomplete.** The reversal against LR-B is met in three or more R3 folds **despite** a small offset. Reading: untrained inputs are not the whole story, and Section VI-D must be revised to say so. This is the outcome that would cost us most, which is why it is written down first.
- **Outcome 3, transfer still fails.** R4 transfer AUROC includes 0.5 on P0 and under model balancing, while within-AgentDojo clears 0.70. Reading: cross-benchmark failure now holds for three detector families and a content reference, which strengthens the RQ4 claim without making it universal.
- **Outcome 4, transfer succeeds.** R4 transfer interval lower bound above 0.60 on P0 **and** model-balanced. Reading: the transfer claim is detector-dependent. The abstract, Section VII-A and Section X must be revised to say that a frozen pretrained representation transfers where our baselines do not. **We commit now to reporting this outcome prominently if it occurs**, because it is the result that most undermines the paper's headline.
- **Outcome 5, inconsistent across seeds.** With 10 seeds, if the across-seed AUROC range in any R3 fold exceeds 0.10, report the spread as the finding and do not average it away.

No outcome licenses a claim about DriftNet's own reported performance. Our numbers bound a reimplementation.

## 10. Computational budget and stopping rule

- **Environment.** torch 2.11.0+cpu, sentence-transformers 6.0.1, transformers 5.14.1, scikit-learn 1.7.2, all confirmed present on 8 Oct 2026. The Hugging Face hub was reachable on the same date, and `all-mpnet-base-v2` is not yet in the local cache, so it is downloaded once.
- **Embedding pass.** Roughly 85k step texts, once, estimated 20 to 40 minutes on CPU. Cached thereafter.
- **Fits.** About 28 rung and split configurations times 10 seeds, so roughly **280 fits**, at an estimated 1 to 3 minutes each, giving **5 to 14 hours** of CPU. The graded-coverage sweep adds 5 levels times 10 seeds at test time only, with no refitting, since masking happens in the step text and the model is not retrained.
- **Stopping rule.** If the embedding pass exceeds 2 hours or a single fit exceeds 10 minutes, stop and report the budget problem rather than reducing seeds silently. Any reduction in seeds is a deviation and goes in Section 11.
- **The GRU is re-run at 10 seeds under its existing pre-registration**, so that the two learned detectors are compared at equal seed counts. That re-run changes no GRU design choice and its results supersede the 3-seed values wherever they differ.

## 11. Deviations declared in advance

- **D1.** The four scalar features are ours, not DriftNet's, which the source paper does not enumerate. Any difference from DriftNet's reported results may be due to this alone.
- **D2.** Within-AgentDojo fits train the detection head only, because AgentDojo carries no step labels. AgentDrift fits use both heads.
- **D3.** We do not evaluate the localisation head, so none of DriftNet's localisation metrics, which are injection-point exact match and span IoU, are reproduced or compared.
- **D4.** No hyperparameter search. Every value in Sections 2 and 5 is fixed here and is reported as fixed, including the auxiliary loss weight of 0.3 and the learning rate of 1e-4.
- **D7.** Validation at R1 is a stratified 10% carve-out of training, matching the executed GRU pipeline (revision deviation V15), not `data/val.jsonl` as the GRU pre-registration states. This keeps the two learned detectors selected identically at every rung. It means neither learned detector uses the AgentDrift validation split as shipped, which should be stated wherever the GRU and this detector are compared.
- **D6.** The `thought` field is dropped from the step text, although AgentDrift provides it, because AgentDojo does not and R4 requires one representation. DriftNet, which was evaluated on AgentDrift alone, had no reason to drop it and may well have used it. Our AgentDrift numbers are therefore a lower bound relative to a version that reads thoughts.
- **D5.** The source paper trains on the AgentDrift task-disjoint split, which is our R2. We train on every rung of the ladder, so our R2 is the closest comparison point to its reported setting and should be the one cited when relating our numbers to theirs.
