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

*Amendment 3 (8 Oct 2026, written before any embedding or fit of this study) supersedes the values in this section where Section 12 says so. Section 12 lists every change and its source.*


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

*Amendment 3 (8 Oct 2026, written before any embedding or fit of this study) supersedes the values in this section where Section 12 says so. Section 12 lists every change and its source.*


The source paper states that each step vector is a frozen `all-mpnet-base-v2` embedding plus four hand-crafted features, and does not enumerate them. We cannot recover them, so we **substitute four of our own and declare it** (Section 11, D1).

The four are chosen to be vocabulary-independent and already defined in `featurise.py`, so that nothing new enters the pipeline.

1. Observation length in characters, log1p transformed.
2. Argument length in characters, log1p transformed.
3. Step index divided by trajectory length, in [0, 1].
4. The state-change keyword flag for the tool name.

All four are standardised with a `StandardScaler` fitted on the training steps of the rung in question, never on test.

**A caveat we commit to in advance.** Feature 1 is the observation-length cue that Section VII-B shows the attacker writes. Including it means this detector inherits that cue, exactly as LR-B does. We therefore also run **one** sensitivity arm, `SE-noobs`, identical but with feature 1 removed, on R1, R3 and R4 only. This arm is pre-specified here and is not an exploratory addition.

## 4. Embedding procedure and caching

*Amendment 3 (8 Oct 2026, written before any embedding or fit of this study) supersedes the values in this section where Section 12 says so. Section 12 lists every change and its source.*


- Embeddings are computed **once** for every step of AgentDrift (train, val, test, and the task-disjoint split) and of `agentdojo_traces.jsonl`, then cached to disk as float32 arrays keyed by trajectory id and step index.
- Approximately 71k AgentDrift steps and 14,079 AgentDojo steps, so roughly **85k step texts**.
- The cache is written under `revision/cache/sencoder/` with a manifest recording the model name, its revision hash from the Hugging Face hub, the pooling mode, the maximum sequence length, the truncation count and the SHA-256 of each input file.
- Because the encoder is frozen, the cache is computed before any split is chosen and is shared by every rung and seed. **This is not leakage.** The encoder saw no label and no split, and its weights were fixed by a third party before this study. The manifest exists so a reader can verify that claim.

## 5. Training procedure (fixed)

*Amendment 3 (8 Oct 2026, written before any embedding or fit of this study) supersedes the values in this section where Section 12 says so. Section 12 lists every change and its source.*


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

*Amendment 3 (8 Oct 2026, written before any embedding or fit of this study) supersedes the values in this section where Section 12 says so. Section 12 lists every change and its source.*


- **Environment.** torch 2.11.0+cpu, sentence-transformers 6.0.1, transformers 5.14.1, scikit-learn 1.7.2, all confirmed present on 8 Oct 2026. The Hugging Face hub was reachable on the same date, and `all-mpnet-base-v2` is not yet in the local cache, so it is downloaded once.
- **Embedding pass.** Roughly 85k step texts, once, estimated 20 to 40 minutes on CPU. Cached thereafter.
- **Fits.** About 28 rung and split configurations times 10 seeds, so roughly **280 fits**, at an estimated 1 to 3 minutes each, giving **5 to 14 hours** of CPU. The graded-coverage sweep adds 5 levels times 10 seeds at test time only, with no refitting, since masking happens in the step text and the model is not retrained.
- **Stopping rule.** If the embedding pass exceeds 2 hours or a single fit exceeds 10 minutes, stop and report the budget problem rather than reducing seeds silently. Any reduction in seeds is a deviation and goes in Section 11.
- **The GRU is re-run at 10 seeds under its existing pre-registration**, so that the two learned detectors are compared at equal seed counts. That re-run changes no GRU design choice and its results supersede the 3-seed values wherever they differ.

## 11. Deviations declared in advance

*Amendment 3 (8 Oct 2026, written before any embedding or fit of this study) supersedes the values in this section where Section 12 says so. Section 12 lists every change and its source.*


- **D1.** The four scalar features are ours, not DriftNet's, which the source paper does not enumerate. Any difference from DriftNet's reported results may be due to this alone.
- **D2.** Within-AgentDojo fits train the detection head only, because AgentDojo carries no step labels. AgentDrift fits use both heads.
- **D3.** We do not evaluate the localisation head, so none of DriftNet's localisation metrics, which are injection-point exact match and span IoU, are reproduced or compared.
- **D4.** No hyperparameter search. Every value in Sections 2 and 5 is fixed here and is reported as fixed, including the auxiliary loss weight of 0.3 and the learning rate of 1e-4.
- **D7.** Validation at R1 is a stratified 10% carve-out of training, matching the executed GRU pipeline (revision deviation V15), not `data/val.jsonl` as the GRU pre-registration states. This keeps the two learned detectors selected identically at every rung. It means neither learned detector uses the AgentDrift validation split as shipped, which should be stated wherever the GRU and this detector are compared.
- **D6.** The `thought` field is dropped from the step text, although AgentDrift provides it, because AgentDojo does not and R4 requires one representation. DriftNet, which was evaluated on AgentDrift alone, had no reason to drop it and may well have used it. Our AgentDrift numbers are therefore a lower bound relative to a version that reads thoughts.
- **D5.** The source paper trains on the AgentDrift task-disjoint split, which is our R2. We train on every rung of the ladder, so our R2 is the closest comparison point to its reported setting and should be the one cited when relating our numbers to theirs.

## 12. Amendment 3, values recovered from the source paper (8 Oct 2026, before any embedding or fit)

**Why this amendment exists.** Sections 2, 3 and 5 were reconstructed from a one-line description in our own related-work table. The DriftNet paper itself (arXiv:2609.10892v1, Sections V and VI, Eqs. 6 to 16, Tables IV and V, Listing 1, Algorithm 1) was read on 8 Oct 2026 after amendment 2 was hashed. Where the paper is specific and the earlier text was a guess, the paper now wins. Where an earlier choice was a deliberate design decision for this study, it stays and is named below. No embedding of this study and no fit had been produced when this section was written and hashed. The only sentence-encoder files on disk at that time were a partial CPU run of `sencoder_embed.py` in `revision/cache/sencoder/` in the version 1 step-text format, started by the author after amendment 1. They are used only for a GPU against CPU equivalence check and never for a result.

**Hard rule.** The published DriftNet numbers are now visible to us. Nothing is tuned to approach them. Every value below is fixed as written. A gap between our R2 and the paper's task-disjoint test result is a result, reported with the deviations that plausibly explain it. This remains a reimplementation in every mention, and no number of ours is presented as DriftNet's own.

**Configuration adopted.** The paper reports two configurations, a reference run (L = 2, peak learning rate 3e-4, dropout 0.2, weight decay 0.01) and a sweep-selected run (L = 3, peak learning rate 1e-3, dropout 0.3, weight decay 0.05, d = 256). Its published test numbers come from the sweep-selected run, so that configuration is adopted. We run no sweep of our own. Adopting the selection the authors made on their own validation data is not a search by us, and D4 continues to hold for this study.

### 12.1 Changes, each with its source

| Item | Version 1 and amendment 2 | Amendment 3 | Source |
|---|---|---|---|
| Step text | `<tool> \| k=v, ... \| <obs>` | `TOOL: {tool} \| ARGS: {args} \| OBS: {obs}` with args as `json.dumps(args, sort_keys=True)` | Sec. V-A and Listing 1. Listing 1 prints the keys of `banking_attacked_full_0203` step 4 in sorted order although the record stores them unsorted, so sorted keys are inferred |
| THOUGHT segment | dropped (D6) | still dropped, segment removed entirely | Paper includes `THOUGHT: {thought}`. Kept out by design, see D6 below |
| Four step features | substituted (D1), standardised | `args_has_email`, `args_has_external_email`, `args_external_email_frac`, `args_has_external_url`, concatenated raw (values in 0 to 1, no scaler) | Sec. V-A, Eq. 6. "External" follows AgentDrift's own rule, Eq. 11 and Listing 2 of arXiv:2609.06972 |
| Layers | 2 | 3 | Table IV, selected configuration |
| d_model, heads | 256, 4 | 256, 4 (unchanged) | Sec. V-B |
| Feed-forward | 512, activation unstated | 512, GELU | Sec. V-B, Fig. 1 |
| Positional encoding | learned, 16 positions, last 16 steps kept | sinusoidal (Eq. 8), maximum length 64, dropout applied to projection plus encoding, no truncation needed (AgentDrift maximum 11, AgentDojo maximum 29 steps) | Sec. V-B |
| Dropout | 0.1 | 0.3 in the input, every encoder layer and both head MLPs | Table IV, selected configuration |
| Pooling | mean over valid steps | masked mean over valid steps (unchanged) | Eq. 10 |
| Detection head | `Linear(256 -> 1)` | MLP 256 -> 128, GELU, dropout, 128 -> 1 | Sec. V-B |
| Localisation head | `Linear(256 -> 1)`, binary | MLP 256 -> 128, GELU, dropout, 128 -> 4, softmax over benign, injection_point, hijacked, failed_injection | Eq. 11 |
| Trajectory loss | weighted BCE, pos_weight n_neg/n_pos | unchanged rule. The paper's w+ = 1.27 equals n_neg/n_pos of the task-disjoint training part | Eq. 13 |
| Step loss and weight | 0.3 x weighted binary BCE | 1.0 x class-weighted 4-way cross entropy, normalised by the summed weights of valid steps. Class weights are inverse frequency over the rung's training steps, scaled to sum to 4, which reproduces the paper's (0.08, 0.76, 0.34, 2.81) | Eqs. 12, 14, 15 |
| Optimiser | Adam, 1e-4, weight decay 1e-4 | AdamW (decoupled weight decay 0.05), peak learning rate 1e-3, triangular schedule over the full budget S = 40 epochs x batches per epoch, linear warm-up over 0.1 S, then linear decay to 0 | Eq. 16, Table IV |
| Epochs, early stopping | 30, patience 5 on validation AUPRC | 40, patience 8 on validation trajectory F1 at threshold 0.5, keep the checkpoint with the best F1 | Sec. V-E, Algorithm 1 |
| Batch size | 32 | 32 (unchanged) | Sec. V-E |
| Gradient clipping | none | global norm 1.0 | Sec. V-E, Algorithm 1 |
| Parameters | stated as 1.79M | 1,845,637, equal to the reported count (1,318,533 for L = 2, also equal), see D10 | Sec. V-B |

Values the paper does not state are the PyTorch defaults of `nn.TransformerEncoderLayer` (post-norm, layer-norm epsilon 1e-5, default initialisation) and of AdamW (betas 0.9 and 0.999, epsilon 1e-8).

### 12.2 The four features, exactly as computed here

For a step, let `a = json.dumps(step["args"]).lower()`. Let `EMAIL_RE = [\w.+-]+@[\w-]+\.[\w.-]+`, as in AgentDrift Listing 2. Let E(W) be the lower-cased addresses of the user and contacts of the world W, and D(W) their domains. A matched address m is external when m is not in E(W) and its domain is not in D(W), exactly as in that listing.

1. `args_has_email` is 1 when `EMAIL_RE` finds at least one match in `a`.
2. `args_has_external_email` is 1 when at least one match is external.
3. `args_external_email_frac` is the number of external matches divided by the number of matches, and 0 when there is none. **The paper names this feature but gives no formula.** Its worked example in Listing 1 gives (1, 1, 0.33, 0) for a step whose arguments hold exactly one address, which is external. The literal reading of the name gives 1.0 there, not 0.33. Readings that do reproduce 0.33 exist, for example external matches divided by the number of contacts in W (three in that world) or divided by the number of argument fields plus one, but each is a guess fitted to one number. We keep the literal reading because it is the only one the name supports, and declare the inconsistency in D1.
4. `args_has_external_url` is 1 when `a` contains a URL, matched by `(?:https?://|www\.)([a-z0-9.-]+)`, whose host, after removing a leading `www.`, is neither in D(W) nor a subdomain of a domain in D(W). The paper and AgentDrift say only "analogously" for URLs, so the regex and the host rule are ours.

**AgentDojo has no W.** Following the author's decision of 8 Oct 2026, W is derived once per suite from the AgentDojo environment files at commit 089ed468 (`src/agentdojo/data/suites/<suite>/environment.yaml` and `workspace/include/*.yaml`, SHA-256 in the run manifest). The rule mirrors AgentDrift, whose contacts share the user's own domain in 12,478 of 12,536 worlds. E(W) is the user's own address where the environment defines one, and D(W) is its domain. For workspace and travel this is `emma.johnson@bluesparrowtech.com` and `bluesparrowtech.com`. Slack and banking define no user address, so E(W) and D(W) are empty and every address or URL found there counts as external. This is declared as D8.

### 12.3 Arms added and retired

- **SE-noobs is retired.** Its premise was an explicit observation-length input, and none of the paper's four features is one.
- **Padding test, replacing the question SE-noobs asked.** The observation still reaches this detector through the sentence embedding. Following Section VII-B of the manuscript and `revision/features_obslen.py`, the filler `" The quick brown fox jumps over the lazy dog."` repeated and cut to n characters is appended to the observation of every injection_point step of each attacked R1 test trajectory, for n = 100 and n = 1,200 (n = 0 is the baseline). Padded steps are re-embedded. The ten R1 models are scored at test time only, with no refit. We report recall at the training threshold and AUROC on pool E, with padded positives and unpadded negatives, next to the existing LR-C and GRU values. Padding can push a step past the 384 word-piece limit, and the number of truncated padded steps is reported.
- **SE-W0, zeroed-feature sensitivity arm for R4.** For transfer, the ten AgentDrift-trained R4 models are scored on AgentDojo with all four features set to 0, with no refit. For within-AgentDojo (UT on P0 with P0F and model-balanced derived from it, and on C1), the models are refit with the four features set to 0 in training and test. If the R4 conclusions barely move, D8 does not matter and we say so. If they move, that is a finding.

### 12.4 Deviations after amendment 3

D2, D3, D4, D5 and D7 stand as written. D1 and D6 are restated. D8 to D12 are new.

- **D1, restated.** Three of the four step features are now recovered from the source together with AgentDrift's rule for "external" and are no longer substitutes. The fourth is named in the source as `args_external_email_frac`, but no formula is given. Its worked example (1, 1, 0.33, 0) is inconsistent with the literal reading, which we use (Section 12.2, item 3). The URL regex and host rule of the fourth listed feature are also ours. Differences from DriftNet's reported results may still come from these choices.
- **D3, restated.** The localisation head is now the paper's 4-way head and its loss is part of training. Its outputs are still not evaluated.
- **D6, restated.** The source serialises `THOUGHT: {thought}` into every step, and we do not, because AgentDojo has no thought field and R4 needs one representation. This is the most likely single reason our R2 differs from the paper's task-disjoint result, together with D1, D7 (our R2 validation is a 10% carve-out of task-disjoint training rather than the shipped task-disjoint validation part), our ten seeds against their single seed 42, and CPU against GPU training. **Our R2 is therefore not directly comparable to the reported number, and no gap between them may be read as a reimplementation failure.**
- **D8.** World features on AgentDojo use a per-suite W derived from the environment files (Section 12.2), whereas AgentDrift provides W per trajectory. The four features are therefore built differently on the two benchmarks, which is the same class of risk that led to D6. In banking and travel the features are close to constant because their arguments carry almost no addresses or URLs. The share of AgentDojo traces per suite with any address or URL in their arguments is reported. SE-W0 measures how much R4 depends on this.
- **D9.** `agentdojo_failed.jsonl` (4,987 unsuccessful attacks) is embedded in addition to the sources of Section 4, so that the P0F population used elsewhere in the paper can be reported for this detector.
- **D10, no deviation.** With the head MLPs of Section 12.1 the trainable parameter count is 1,845,637 for L = 3 and 1,318,533 for L = 2, both equal to the counts the paper reports. A hand count in our working notes before this section was written gave 256 more, which was an arithmetic error in the head layers and not a property of the model. The count is checked by instantiating the model in the smoke test, before any fit. The version 1 architecture of Section 2 would have had 1,256,706 parameters, not the 1.79M stated there, because that section's count assumed a feed-forward width of 1,024.
- **D11.** The environment differs from Section 10. The embedding pass runs on one NVIDIA H100 with torch 2.11.0 (CUDA 12.8 wheel), transformers 5.14.1 and sentence-transformers 6.0.1 under Python 3.13.5, rather than on CPU under Python 3.10. A fixed sample of 1,000 texts is re-embedded on CPU in the same job, and the version 1 format pass over `agentdrift_train` is compared row by row with the author's local CPU file. Training runs on CPU with deterministic algorithms, as Section 5 requires.
- **D12.** SE-noobs is retired and replaced by the padding test, and SE-W0 is added (Section 12.3). Both were fixed here, before any result.

### 12.5 Budget

SE takes 230 fits (R1 10, R1-wd 50, R2 10, R3 50, R4 transfer 10, within-AgentDojo UT 50 and C1 50). SE-W0 takes 100 refits within AgentDojo. The GRU at 10 seeds takes 230 fits. The stopping rules of Section 10 are unchanged.
