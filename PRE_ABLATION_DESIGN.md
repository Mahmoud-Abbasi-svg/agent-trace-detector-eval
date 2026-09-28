# PRE_ABLATION_DESIGN: making the feature ablation interpretable before running it

*25 Sep 2026. Written before running the detector ablation. All numbers here are diagnostics on AgentDrift/AgentDojo structure and on `world`, not detector results. This document is the pre-registration; `ABLATION_RESULTS.md` Part 1 lists the feature definitions and will hold the results.*

---

## 1. Data audit summary (measured)

### 1.1 Tool-vocabulary coverage (benign training vocabulary vs test tools)
| rung | tool-occurrence coverage | traces with zero known tools |
|---|---|---|
| R1 (same distribution) | 1.000 | 0.000 |
| R3 leave-one-domain-out — **every** held-out domain (medical, banking, web, email, coding) | **0.000** | **1.000** |
| R4 AgentDrift → AgentDojo | 0.075 | 0.698 |

AgentDrift tool sets are **disjoint across the five domains** (no tool name is shared by two domains). Therefore every R3 held-out trace has an all-zero bag-of-tools vector. For R4, 70% of AgentDojo traces contain no tool seen in AgentDrift, and only 7.5% of AgentDojo tool occurrences are covered.

### 1.2 `world` and the binary label
- **Number of worlds:** 1,029 distinct `world` dicts (fields: user, email, company, contacts, date) over 10,656 trajectories.
- **Samples per world:** highly uneven; largest worlds have 194 / 169 / 144 trajectories; 54 worlds have ≥30.
- **Positive (attacked) rate per world:** in-sample majority-label lookup by world classifies 0.858 of trajectories correctly.
- **Baselines:** overall attacked base rate 0.442; majority-class baseline 0.558.
- **Our diagnostic vs the paper's.** Our 0.858 is *in-sample majority-label purity by world*: assign each world its own most-common label and score on the same data. The AgentDrift paper reports **86.1% by world-identity lookup**; the two figures agree closely but are **not the same evaluation** — ours is a purity statistic on the pooled data, theirs is their own reported leakage probe. We do not restate their number as ours.

## 2. World / domain identity diagnostic in the features (measured)

*Diagnostic classifiers only — not detectors, not used in the paper's detection results.*

**D1 — predict domain (5-class; majority baseline 0.207), 3-fold CV accuracy:**
| representation | accuracy |
|---|---|
| A tool identity (bag-of-tools) | **1.000** |
| B behavioural (10 features) | 0.455 |
| B− behavioural without the two name-keyword features | 0.381 |

**D2 — predict world (54 worlds with ≥30 traj, n=2,724; majority baseline 0.044, domain-only baseline 0.097), 3-fold CV accuracy:**
| representation | accuracy |
|---|---|
| A | 0.066 |
| B | 0.046 |
| B− | 0.043 |

**D3 — world → label association in the R1 split.** 99.3% of R1 test traces have a `world` that also appears in training. On those, the training-world attack rate predicts the test label with AUROC **0.963** (base rate 0.441).

**Reading (careful terminology).**
- The features carry strong **domain-identity** signal — tool identity encodes domain perfectly (D1 = 1.000), behavioural features moderately (0.38–0.46, above the 0.207 baseline). "State-changing" keyword features add a little domain signal (B 0.455 vs B− 0.381).
- The features do **not** carry fine **world-identity** signal: D2 is at or barely above the domain-only baseline for all three.
- There **is** a strong world→label association in the split (D3, 0.963 by lookup), i.e. the split reuses worlds and worlds are label-predictive. But because the features cannot recover world identity (D2), the detector cannot directly exploit that per-world association. The accessible shortcut is **domain identity**, not world identity.
- **Terminology decision:** we call D1/D2 a **domain-identity signal** (a dataset/benchmark shortcut). We call D3 a **world→label association in the split-design** (a leakage *vector*), and we explicitly do **not** claim the detector exploits world-level "label leakage", because D2 shows the features do not encode world.
- **Caveat that this raises for R1:** domain identity is fully recoverable from tools. If the attacked base rate differed by domain, domain identity would be a label shortcut. AgentDrift is domain-stratified, so this needs a check (see §7, S3) before R1 is called leakage-free.

## 3. Tool-vocabulary coverage: implications for the ablation

- **Under R3, tool-only (A) is constant by construction.** With 0% coverage, every held-out trace has an all-zero tool vector, so A outputs one score for all traces → AUROC = 0.5, AUPRC = prevalence. This is **not** evidence that tool identity is a poor representation; it is total information loss from unseen tools.
- **Combined (C) under R3 is ranked only by its behavioural weights**, because its tool block is constant. So an R3 "A vs B" or "A vs C" gap is mechanical, not informative about representation quality.
- **Under R4, A is near-constant on ~70% of traces** and has non-zero tool information on ~30%. So R4 tool-only is "nearly information-empty by construction" for most traces, but not entirely — it can be reported, with the coverage stated.
- **The clean comparison is R1**, where coverage is 100% and A, B, C all see full information.

## 4. Exact interpretation rules (fixed before running)

Let "detected" mean a 95% bootstrap CI excluding 0 (for a difference) or excluding 0.5 (for an AUROC level).

**Rule 0 — always report coverage next to every A result.** Any statement about A under R3/R4 is annotated with tool coverage and zero-known-tool fraction.

**Rule 1 — do not infer representation quality from structural collapse.**
- Tool-only AUROC ≈ 0.5 under R3 is attributed to **information loss from unseen tools** (0% coverage), never to "tool identity is a poor representation".
- Under R4, distinguish:
  - (i) **poor transfer with substantial known-tool coverage** — not applicable here at the trace level (coverage 7.5%);
  - (ii) **near-zero information because most tools are unseen** — this is the R4 tool-only regime, and is how A's R4 result is described.

**Rule 2 — behavioural robustness (B).** Judged on whether B *retains* signal under shift, on its own terms, not by beating A:
- B "retains signal" under a shift if its AUROC CI lower bound > 0.5.
- B "degrades" from R1 if the independent-bootstrap ΔAUROC (or ΔAUPRC at matched prevalence) R1→shift is detected as negative.

**Rule 3 — the behavioural-transfer hypothesis (from R4_CONTROL S1) is:**
- **Supported for within-benchmark domain shift** if B retains signal in ≥3/5 R3 folds.
- **Supported for cross-benchmark transfer** only if B retains signal in R4 transfer (P0 *and* model-balanced). Given §1.1, this is a strong prior against: near-zero coverage means even a good representation has little to work with. If B fails R4 transfer, the hypothesis is **withdrawn for cross-benchmark** and can only be retained, if at all, for **within-benchmark domain shift**.

**Rule 4 — combined vs behavioural (Outcome 4).** If C > B in R1 (paired, detected) but C degrades more than B from R1→shift (degradation-difference detected favouring B in ≥3/5 R3 folds or in R4), report as "benchmark-specific tool information improves in-distribution scores while reducing robustness". The R3 version of this is partly structural (C's tool block zeroes out), so the **primary evidence for Outcome 4 is R4**, where both blocks are active.

**Rule 5 — tool identity better (Outcome 3).** Only creditable where A has real information: R1 (full coverage), or the ~30% of R4 traces with ≥1 known tool. A ">" comparison under R3 is not evidence for Outcome 3.

## 5. Decision: **B — proceed, but explicitly condition interpretation on vocabulary coverage**

- Not **A** (proceed unchanged): the R3/R4 tool-only numbers would be misread without the coverage conditioning above.
- Not **C** (modify before running): the ablation is scientifically interpretable as designed once Rules 0–5 are applied. The R1 comparison is clean; the R3/R4 A-collapse is understood and reportable as a coverage effect; B and C remain interpretable throughout. Redesigning the feature sets would change the detector rather than probe it.
- Therefore **B**: run the planned A/B/B−/C ablation on R1, R3 (per fold), R4 (transfer + within controls), and interpret strictly under Rules 0–5, always reporting coverage.

## 6. Final proposed ablation conditions
- **Representations:** A (tool identity), B (behavioural, = current "agnostic"), C (combined, = current full). **B−** (behavioural minus the two name-keyword features) as a sensitivity arm, since D1 shows those two features carry some domain signal.
- **Rungs:** R1; R3 five folds; R4 (transfer on P0 / C1 / model-balanced, and the within-AgentDojo UT and IT references established in R4_CONTROL).
- **Everything else identical to the corrected framework:** pools, prevalences, seeds, B, training-derived thresholds, `evalkit` bootstrap. Report AUROC and AUPRC with CIs, paired differences (B−A, C−A, C−B, C−B−), FPR at τ_r80, and exact counts. No per-representation tuning.
- **Reproduction anchors:** C must reproduce the corrected LOGREG native points (R1 AUROC 0.879; R4 transfer 0.440; within-UT 0.855); B must reproduce LOGREG-agnostic (R4 transfer 0.447; within-UT 0.798).

## 7. Additional sensitivity analyses required
- **S1 — `<UNK>` tool encoding (separate sensitivity arm; do NOT replace A).**
  - *Current encoding:* an unseen tool contributes to no bag-of-tools column, so a trace of all-unseen tools has an all-zero tool vector, and two traces with *different* unseen tools are identical in the tool block.
  - *What `<UNK>` adds:* a single shared column counting unseen-tool steps. This is informative (an all-unknown trace becomes distinguishable from a partly-known one), **but the information it adds is essentially an out-of-vocabulary rate — a coverage/behavioural feature, not tool identity.** It would let "tool-only" score above chance under R3/R4 partly by counting how many tools are unknown, which is not what "tool identity" is meant to test.
  - *Does it change the question?* Yes. The deployed detector used silent zero-mapping. `<UNK>` is a **new representation not part of the original detector**. It makes A a fairer test of "does any tool-derived structure survive vocabulary shift", but it conflates tool identity with OOV rate.
  - *Recommendation:* include `A+UNK` **only as a clearly separate sensitivity analysis**, reported apart from A, with the explicit note that its above-chance behaviour under shift reflects OOV counting rather than tool identity. It is optional; A (zero-mapping) remains the primary tool-only result because it matches the original detector.
- **S2 — domain-identity control for R1 (essential to state, cheap).** Report the attacked base rate per domain in the R1 pools. If base rates are near-equal across domains, domain identity is not a label shortcut in R1 and the D1 domain signal is benign for detection. If they differ, flag domain identity as a potential R1 shortcut.
- **S3 — world-disjoint R1 (already scheduled as experiment #1b in NEXT_EXPERIMENTS).** This directly tests whether R1 is inflated by the world→label association (D3). It is the definitive check for the leakage vector and is cross-referenced here; not run in the ablation itself.

**Not to be run now (per instruction):** stronger detectors, step-level localization, adaptive attacks, additional datasets. The `<UNK>` arm and S2 are cheap and may be run with the ablation; S3 is separate.
