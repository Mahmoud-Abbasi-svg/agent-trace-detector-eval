# GRU_R3_DIAGNOSTIC: why is the GRU worse than linear C under R3?

*25 Sep 2026. A narrow diagnostic that runs on R3 only. It does **not** modify `gru_detector.py`, `GRU_PREREGISTRATION.md`, `GRU_RESULTS.md`, `FINDINGS.md`, `MANUSCRIPT_DRAFT.md`, or any A/B/C result.*
- Script: `gru_r3_diagnostic.py`.
- Raw output: `gru_r3_diagnostic_results.json`.

---

## 1. Question and conditions

**Question.** Is the GRU's R3 deficit associated with its tool-sequence pathway, with its behavioural features, or with the combination of the two?

**Three input variants.** Architecture, seeds, validation and thresholds are the same as the pre-registered GRU where they apply.

| variant | input | model |
|---|---|---|
| **GRU-A** tool sequence only | tool tokens (`<PAD>`/`<UNK>` + training vocabulary) | Embedding(32) → GRU(64) → Linear(64→1). This is the pre-registered `tool_only` path. |
| **GRU-B** behavioural only | the 10 standardised behavioural features | Linear(10→1). **There is no sequence input, so this variant has no embedding and no recurrent layer.** It is a linear classifier trained with the GRU training procedure (weighted BCE, Adam, early stopping on validation AUPRC). |
| **GRU-C** combined | tool tokens + behavioural features | The pre-registered primary GRU, refit here as a consistency check. |
| *C (linear)* | behavioural + bag-of-tools | The combined logistic regression, refit here so it can be paired with the variants on the same test traces. |

**Held fixed:**
- training seeds {0, 1, 2};
- stratified 10%-of-train validation split and early stopping on validation AUPRC;
- threshold derived from training data only;
- the five R3 folds;
- `evalkit` stratified bootstrap (B = 1,000, seeds 5200 + fold index), with paired differences on the same test traces;
- no tuning, no new datasets, no R1/R2/R4 reruns.

## 2. Sanity checks
- **GRU-C reproduces `GRU_RESULTS.md` exactly** in all five folds: 0.639 / 0.736 / 0.726 / 0.671 / 0.695. Training is deterministic.
- **Linear C reproduces the ablation's R3 C values exactly** (0.745 / 0.828 / 0.862 / 0.739 / 0.814).
- **GRU-B ≈ linear B from the ablation**, within ≤0.011 AUROC in every fold (0.797 vs 0.808, 0.847 vs 0.854, 0.898 vs 0.896, 0.786 vs 0.792, 0.843 vs 0.852). This is expected, since both are linear models on the same 10 features. So "GRU-B" is **not** a recurrent model, and nothing here says anything about a *recurrent* behavioural pathway.

## 3. Test populations (per fold)
In every fold, tool coverage is **0%**: tool-occurrence coverage 0.000, distinct-tool coverage 0.000, and 100% of test traces contain zero known tools. Every test tool is therefore `<UNK>` for GRU-A and GRU-C.

| held-out domain | n_neg | n_pos | native prevalence | n_pos at 10% |
|---|---|---|---|---|
| medical | 935 | 1,020 | 0.522 | 104 |
| banking | 935 | 626 | 0.401 | 104 |
| web | 935 | 1,020 | 0.522 | 104 |
| email | 935 | 1,020 | 0.522 | 104 |
| coding | 935 | 1,020 | 0.522 | 104 |

## 4. Results

**AUROC (native prevalence), 95% bootstrap CI:**

| domain | GRU-A tool | GRU-B behav. | GRU-C combined | C linear |
|---|---|---|---|---|
| medical | 0.572 [0.546, 0.597] | **0.797** [0.775, 0.817] | 0.639 [0.605, 0.668] | 0.745 [0.723, 0.766] |
| banking | 0.568 [0.540, 0.593] | **0.847** [0.827, 0.866] | 0.736 [0.698, 0.777] | 0.828 [0.808, 0.849] |
| web | 0.606 [0.580, 0.630] | **0.898** [0.884, 0.910] | 0.726 [0.690, 0.766] | 0.862 [0.846, 0.876] |
| email | 0.567 [0.542, 0.593] | **0.786** [0.765, 0.807] | 0.671 [0.624, 0.732] | 0.739 [0.716, 0.760] |
| coding | 0.582 [0.556, 0.608] | **0.843** [0.826, 0.860] | 0.695 [0.654, 0.744] | 0.814 [0.796, 0.833] |
| **mean [range]** | **0.579** [0.567, 0.606] | **0.834** [0.786, 0.898] | **0.693** [0.639, 0.736] | **0.798** [0.739, 0.862] |

**AUPRC at 10% prevalence** (104 positives vs 935 negatives; chance = 0.100), mean [range] across folds:
- GRU-A: 0.117 [0.113, 0.122], close to chance;
- GRU-B: 0.433 [0.339, 0.600];
- GRU-C: 0.231 [0.171, 0.290];
- C linear: 0.363 [0.256, 0.505].

Per-fold CIs are in the JSON. Native-prevalence AUPRC shows the same ordering.

**Paired ΔAUROC** (same test traces). All 25 differences below are detected (CI excludes 0).

| comparison | medical | banking | web | email | coding | detected |
|---|---|---|---|---|---|---|
| GRU-A − GRU-B | −0.224 | −0.280 | −0.293 | −0.220 | −0.261 | 5/5 |
| GRU-C − GRU-A | +0.066 | +0.168 | +0.121 | +0.104 | +0.112 | 5/5 |
| GRU-C − GRU-B | −0.158 | −0.112 | −0.172 | −0.116 | −0.150 | 5/5 |
| GRU-C − C linear | −0.107 | −0.093 | −0.136 | −0.069 | −0.120 | 5/5 |
| GRU-B − C linear | +0.051 | +0.019 | +0.036 | +0.047 | +0.029 | 5/5 |

**The ordering is the same in every fold: GRU-B > C linear > GRU-C > GRU-A.**

## 5. One descriptive fact about the training inputs (no model trained)
The GRU vocabulary is built from *benign* training traces (pre-registered rule). So in training, `<UNK>` could only appear in non-benign traces. The count:

**In all five R3 training sets, `<UNK>` occurs in 0 steps of 0 traces, in every class** (attacked, benign, failed_attack, hard_negative; roughly 8,400–8,800 training traces per fold). Every tool used in any training trace also appears in some benign training trace.

Two consequences follow **by construction**:
1. The explanation "the GRU learned that `<UNK>` means attack" is **ruled out**. `<UNK>` never occurs in training, so it can carry no learned association with the label.
2. **The `<UNK>` embedding is never trained on labelled data.** At R3 test time every token is `<UNK>`, so the sequence pathway receives only inputs that never occurred during training. The pre-registration (§7) correctly said the GRU "retains length/position structure" under `<UNK>`. It did not anticipate that this structure would be carried by an embedding the model was never fitted on.

## 6. Interpretation against the pre-specified rules

| rule | observed | reading |
|---|---|---|
| GRU-A substantially worse than GRU-B → the tool-sequence component contributes to the R3 weakness | **Yes**: −0.22 to −0.29, 5/5 detected | Supported |
| GRU-B also weak → the issue is not simply tool vocabulary | **No**: GRU-B is the strongest variant (0.83 mean) and beats linear C in 5/5 folds | The behavioural features are **not** the source of the deficit |
| GRU-C substantially worse than *both* → interaction | **Not met as stated**: GRU-C is worse than GRU-B (5/5) but **better** than GRU-A (5/5) | See below |
| All three similar → architecture more plausible (no causal claim) | **No**: the variants are clearly and consistently separated | Not this case |

**What the data support (associational, not causal):**
- The GRU's R3 deficit relative to linear C is **associated with its sequence pathway under 0% coverage**, not with its behavioural features.
  - The behavioural pathway on its own is at least as good as linear C.
  - The sequence pathway on its own is weak: AUROC 0.58, and AUPRC near chance.
  - Adding the sequence pathway to the behavioural features pulls the combined model **down** from 0.83 to 0.69. A naive expectation would be that the combined model at least matches its better component.
- **The same direction of effect appears in the linear models, at a smaller size.**
  - Linear: C < B by 0.03–0.06 in the ablation.
  - GRU: GRU-C < GRU-B by 0.11–0.17 here.
  - So "combined < behavioural-only under domain shift" holds for both detector families, and it is roughly three times larger for the GRU.
- **A structural difference is consistent with this, but it was not tested.**
  - Under 0% coverage, linear C's tool block becomes a *constant* vector. A constant shifts every score equally, so it cannot change the ranking.
  - The GRU's sequence pathway does not become constant. It produces length- and position-dependent hidden states from an embedding that was never trained (§5), and the head was fitted to rely on those hidden states (the GRU is the stronger model in-distribution: R1 AUROC 0.936).
  - Out-of-distribution hidden states that still affect the ranking would plausibly hurt more than a constant offset. **We did not manipulate this directly**, for example by training with random tool→`<UNK>` replacement, or by neutralising the hidden state at test time. It is a hypothesis consistent with the data, not an established mechanism.

## 7. What this diagnostic does NOT establish
- **No causal mechanism.** It localises the deficit to the sequence pathway plus its combination with the behavioural features. It does not show *why* that pathway is harmful. The untrained-`<UNK>` account in §6 is the most plausible reading, but it is untested.
- **Nothing about recurrent behavioural modelling.** GRU-B contains no recurrence, so the result cannot separate "the GRU architecture" from "the sequence input". A recurrent model over per-step behavioural features was not run.
- **Nothing about tool identity transferring across domains.** At 0% coverage GRU-A sees only `<UNK>` sequences. Its above-chance AUROC (0.57–0.61) reflects length/position/repetition structure, not tool identity (pre-registration §7, and the coverage rules in PRE_ABLATION_DESIGN).
- **Nothing about R1, R2 or R4.** Those rungs were not rerun. Whether the same pathway-level pattern holds in R4 (7.5% coverage) is untested.
- **It does not change the pre-registered GRU results or outcome.** `GRU_RESULTS.md` (Outcome 1 met; the unanticipated GRU < C finding under R3) stands as written. This diagnostic only narrows where that deficit is located.

## 8. Conclusion
**What does this tell us about why the GRU is worse than C under R3?**
- **Localisation: fairly conclusive.**
  - The deficit is **not** in the behavioural features. On their own they match or beat linear C (+0.02 to +0.05).
  - The deficit is associated with the **tool-sequence pathway under 0% vocabulary coverage**, and with how that pathway combines with the behavioural features. The sequence-only model is weak (0.58), and adding it to the behavioural features lowers performance by 0.11–0.17 in every fold.
  - `<UNK>` never occurs in training, so this pathway runs at test time entirely on inputs the model was never fitted on.
- **Mechanism: not established.** A plausible explanation is that out-of-distribution hidden states from an untrained `<UNK>` embedding shift scores in a ranking-relevant way, whereas linear C's tool block collapses to a harmless constant. We did not test it, and it is reported as a hypothesis only.
- **The one confirmatory test** would be a separately pre-registered intervention on the sequence pathway, for example random tool→`<UNK>` replacement during training. It was not run.
