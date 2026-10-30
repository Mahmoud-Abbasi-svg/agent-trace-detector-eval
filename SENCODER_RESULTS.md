# SENCODER_RESULTS: does a frozen sentence encoder escape the evaluation pattern?

*9 Oct 2026. Implements `SENCODER_PREREGISTRATION.md` as amended (amendment 6, SHA-256 `cef3e7677710e561496b8427a261d2e34737c7af7908e5ef5809ffb5a43a8493`, verified before the run and recorded in `revision/PLAN_MANIFEST.txt` before any embedding or fit existed). SE in this document always means our reimplementation of the DriftNet architecture. No code or weights were released, and no number here is DriftNet's own reported performance. Scripts are `revision/sencoder_text.py`, `revision/sencoder_embed_gpu.py`, `revision/sencoder_detector.py`, `revision/build_sencoder_cache.py`, `revision/sencoder_analysis.py` and `revision/sencoder_tables.py`. Raw output is `revision/sencoder_results.json`. No existing result file, the manuscript or the supplement was edited. The 3-seed GRU cache in `revision/cache/gru/` is untouched, and the 10-seed GRU is in `revision/cache/gru10/`.*

> **Correction, 9 Oct 2026 (added after this file was written, body unchanged).** Section 4 says that SE's P0F transfer interval, 0.572 [0.525, 0.647], excludes 0.5 as it does for LR-C and the GRU. That interval comes from the stratified bootstrap. Under the suite and user-task cluster bootstrap used for Table VI of the paper, the interval is 0.572 [0.490, 0.657] and includes 0.5 (`revision/sencoder_table6_row.json`, computed after the run). The paper reports the cluster interval and does not claim that SE's P0F transfer is above chance.

---

## 0. Run record

| Item | Value |
|---|---|
| Plan hash verified | `cef3e767…a8493` (amendment 6). Chain 0c6b5812 → 55005733 (author's amendment 2) → e1f01872 → e2e88258 → d8011a8f → cef3e767 |
| Encoder | `sentence-transformers/all-mpnet-base-v2`, hub revision `e8c3b32edf5434bc2275fc9bab85f82640a19130`, mean pooling, L2-normalised, 384 word pieces |
| Environment | cn001, Python 3.13.5, torch 2.11.0+cu130, transformers 5.14.1, sentence-transformers 6.0.1, scikit-learn 1.7.2, numpy 2.2.6 |
| Embedding pass | Slurm 2762, CPU with 32 threads, 1,418.7 s for 106,622 step texts plus the masked (17,914 unique texts) and padded R1 variants |
| Fits | Slurm 2762 (157 units, 60.5 min, stopped by the 10-minute rule, see D13) and Slurm 2797 (413 units, 63.1 min, exit 0). 570 of 570 units |
| Analysis | Slurm 2797, 390.2 s, exit 0 |
| GPU against CPU check | Slurm 2799, one H100, 1,000 fixed texts. Maximum absolute difference 2.5e-07, minimum cosine 0.99999988, no row bitwise identical |
| Failed or cancelled attempts | 2756 (GPU, cancelled while pending), 2757 and 2759 (exit 127, `/usr/bin/time` absent), 2758 (cancelled after 22 s), 2761 (offline model cache lacked `refs/main`), 2798 (GPU check, out of memory). None of them produced a fit |
| Smoke gate | 1,845,637 trainable parameters at L = 3 and 1,318,533 at L = 2, both equal to the published counts. Two runs of the same seed differ by 0.0. GRU R1 seed 0 matches the 3-seed cache to 2.3e-6 |

**Truncation at 384 word pieces.** AgentDrift train 3 of 49,750 steps, validation 0 of 10,722, test 0 of 10,552. AgentDojo 950 of 14,079 steps (6.7%) and the unsuccessful attacks 3,237 of 21,519 (15.0%). No masked R1 text was truncated. At 1,200 filler characters, 14 of the 831 padded injection steps were truncated, so for those steps part of the filler is not seen by the encoder.

## 1. Which outcome was met

Section 9 of the plan fixes five outcomes. The numbers decide them as follows.

| Outcome | Rule | Deciding numbers | Met |
|---|---|---|---|
| 1, mechanism survives | offset analogue under +0.5 in at least 4 of 5 R3 folds **and** no reversal against LR-B | offset under +0.5 in 0 of 5 folds (banking +0.524, email +0.692, medical +5.161, coding +5.795, web +7.420). Reversal against LR-B in 0 folds on AUROC and on AUPRC at 10% | **no** |
| 2, mechanism incomplete | reversal against LR-B in 3 or more folds despite a small offset | reversal in 0 folds, and the offset is not small | **no** |
| 3, transfer still fails | transfer AUROC interval includes 0.5 on P0 and model-balanced, within-AgentDojo clears 0.70 | transfer P0 0.481 [0.418, 0.555], model-balanced 0.477 [0.400, 0.551], within P0 0.956 [0.924, 0.977] | **yes**, on the point and on the lower bound |
| 4, transfer succeeds | transfer lower bound above 0.60 on P0 and model-balanced | lower bounds 0.418 and 0.400 | **no** |
| 5, inconsistent across seeds | across-seed AUROC range above 0.10 in any R3 fold | largest range 0.084 (web) | **no** |

**Outcome 3 is met. Outcomes 1, 2, 4 and 5 are not.** The R3 result falls outside the outcome set the plan anticipated, and it has to be read plainly. The plan predicted a small offset for a frozen encoder because it has no untrained input. That prediction failed in three of five folds. In medical, coding and web the held-out negatives sit 5.2 to 7.4 logits above the training negatives, which is the size of the GRU's +6.291, without any untrained input. At the same time the ranking reversal did not occur. SE stays above LR-B in every fold, by a detected margin in four folds after Holm correction, with web not detected.

So the half of the Section VI-D account that links the reversal to untrained inputs is not contradicted by this detector. The half that attributes the offset to untrained inputs is contradicted. An offset of this size appears under ordinary covariate shift in a trained input space. Outcome 2's reading, that untrained inputs are not the whole story, therefore applies to the offset even though Outcome 2's rule, written for the reversal, is not met. Section VI-D must be revised on this point. It should not say the offset is evidence of an untrained input. Two facts bound that revision. The +0.5 bar sits below this estimator's own in-distribution value, since SE's R1 offset is already +0.643. Banking and email are at that in-distribution level, while medical, coding and web are 8 to 12 times above it. The size of the offset follows the embedding drift of the held-out domain only loosely (Section 6).

## 2. R1, world-disjoint R1, R2 and R3 (AUROC, pool E, evalkit bootstrap B = 1,000)

| Rung | κ | SE | GRU (10 seeds) | LR-C | LR-B | SE − LR-B, paired | SE seed range |
|---|---|---|---|---|---|---|---|
| R1 | 1.000 | 0.998 [0.995, 0.999] | 0.932 [0.915, 0.946] | 0.879 [0.863, 0.894] | 0.830 [0.812, 0.850] | +0.167 [+0.148, +0.185] | 0.002 |
| R1-wd split 0 | 1.000 | 0.994 [0.989, 0.998] | 0.955 [0.942, 0.966] | 0.891 [0.874, 0.906] | 0.821 [0.799, 0.841] | +0.174 [+0.153, +0.196] | 0.005 |
| R1-wd split 1 | 1.000 | 0.997 [0.994, 0.999] | 0.948 [0.934, 0.960] | 0.889 [0.875, 0.904] | 0.803 [0.782, 0.821] | +0.194 [+0.175, +0.216] | 0.004 |
| R1-wd split 2 | 1.000 | 0.998 [0.992, 1.000] | 0.961 [0.951, 0.971] | 0.901 [0.887, 0.914] | 0.822 [0.801, 0.842] | +0.176 [+0.156, +0.198] | 0.005 |
| R1-wd split 3 | 1.000 | 0.994 [0.989, 0.998] | 0.958 [0.944, 0.970] | 0.888 [0.873, 0.903] | 0.818 [0.798, 0.838] | +0.176 [+0.156, +0.197] | 0.005 |
| R1-wd split 4 | 1.000 | 0.997 [0.992, 0.999] | 0.957 [0.945, 0.967] | 0.892 [0.876, 0.906] | 0.819 [0.800, 0.838] | +0.178 [+0.158, +0.198] | 0.006 |
| R2 | 1.000 | 0.996 [0.993, 0.998] | 0.915 [0.899, 0.931] | 0.902 [0.887, 0.916] | 0.849 [0.829, 0.868] | +0.147 [+0.127, +0.167] | 0.001 |
| R3 medical | 0.000 | 0.959 [0.940, 0.972] | 0.649 [0.609, 0.699] | 0.745 [0.723, 0.766] | 0.808 [0.788, 0.827] | +0.152 [+0.123, +0.177] | 0.023 |
| R3 banking | 0.000 | 0.917 [0.874, 0.940] | 0.721 [0.661, 0.792] | 0.828 [0.808, 0.849] | 0.854 [0.835, 0.874] | +0.062 [+0.016, +0.093] | 0.047 |
| R3 web | 0.000 | 0.904 [0.842, 0.935] | 0.718 [0.689, 0.755] | 0.862 [0.846, 0.876] | 0.896 [0.882, 0.910] | +0.007 [−0.058, +0.042] | 0.084 |
| R3 email | 0.000 | 0.950 [0.931, 0.967] | 0.656 [0.607, 0.721] | 0.739 [0.716, 0.760] | 0.792 [0.771, 0.812] | +0.158 [+0.133, +0.184] | 0.027 |
| R3 coding | 0.000 | 0.938 [0.922, 0.951] | 0.683 [0.648, 0.736] | 0.814 [0.796, 0.833] | 0.852 [0.835, 0.868] | +0.086 [+0.062, +0.108] | 0.019 |

κ is tool-vocabulary coverage. SE has no tool vocabulary of its own, but κ is reported next to every result as the plan requires. SE is far above every other detector in distribution (0.998 at R1) and degrades under domain shift. Its R3 minus domain-matched R1 change is detected in all five folds after Holm (Section 3). The size of that loss, −0.040 to −0.093 AUROC, is comparable to LR-C's (−0.024 to −0.155) and far smaller than the GRU's (−0.191 to −0.306). LR-B itself does not degrade (−0.012 to +0.023).

**Operating points, SE.**

| Rung | AUPRC native | AUPRC at 2% | AUPRC at 10% | recall at θr80 | FPR at θr80 | recall at 0.5 | FPR at 0.5 |
|---|---|---|---|---|---|---|---|
| R1 | 0.998 [0.996, 0.999] | 0.967 [0.877, 1.000] | 0.988 [0.968, 0.999] | 0.796 [0.764, 0.826] | 0.001 [0.000, 0.004] | 0.981 [0.965, 0.993] | 0.022 [0.007, 0.044] |
| R2 | 0.996 [0.994, 0.998] | 0.935 [0.830, 1.000] | 0.977 [0.952, 0.994] | 0.786 [0.746, 0.821] | 0.001 [0.000, 0.008] | 0.970 [0.950, 0.990] | 0.025 [0.012, 0.043] |
| R3 medical | 0.965 [0.943, 0.977] | 0.681 [0.390, 0.896] | 0.829 [0.706, 0.906] | 0.770 [0.688, 0.884] | 0.046 [0.018, 0.091] | 0.960 [0.917, 0.984] | 0.263 [0.170, 0.327] |
| R3 banking | 0.906 [0.867, 0.931] | 0.568 [0.342, 0.791] | 0.745 [0.644, 0.834] | 0.345 [0.281, 0.447] | 0.002 [0.000, 0.005] | 0.643 [0.527, 0.749] | 0.029 [0.004, 0.056] |
| R3 web | 0.916 [0.824, 0.948] | 0.509 [0.096, 0.792] | 0.678 [0.353, 0.818] | 0.732 [0.632, 0.809] | 0.106 [0.031, 0.255] | 0.936 [0.894, 0.957] | 0.396 [0.306, 0.513] |
| R3 email | 0.964 [0.953, 0.976] | 0.769 [0.576, 0.934] | 0.864 [0.801, 0.925] | 0.620 [0.575, 0.672] | 0.001 [0.000, 0.006] | 0.800 [0.754, 0.840] | 0.035 [0.015, 0.066] |
| R3 coding | 0.950 [0.930, 0.962] | 0.648 [0.323, 0.863] | 0.792 [0.655, 0.866] | 0.737 [0.611, 0.802] | 0.049 [0.016, 0.104] | 0.940 [0.893, 0.969] | 0.317 [0.172, 0.382] |

The thresholds fail under R3, in both directions. At θr80 the training threshold keeps FPR low in banking and email but recall falls to 0.345 and 0.620. In medical, coding and web the same threshold lets FPR rise from 0.001 to 0.046 to 0.106. At 0.5 the FPR in web is 0.396. These are the folds with the large offsets of Section 5.

## 3. The reversal rule against LR-B (revision plan 3.4, domain_shift estimator, B = 2,000, Holm over five folds)

| Held-out domain | R1dm SE − LR-B (AUROC) | R3 SE − LR-B (AUROC) | R3 SE − LR-B (AUPRC at 10%) | reversal established (AUROC / AUPRC10) | SE R3 − R1dm (AUROC) |
|---|---|---|---|---|---|
| medical | +0.206 [+0.161, +0.256] | +0.151 [+0.125, +0.177] | +0.481 [+0.355, +0.558] | no / no | −0.040 [−0.058, −0.027] |
| banking | +0.163 [+0.117, +0.213] | +0.063 [+0.023, +0.093] | +0.301 [+0.192, +0.394] | no / no | −0.077 [−0.116, −0.054] |
| web | +0.111 [+0.078, +0.148] | +0.008 [−0.056, +0.043] (not detected) | +0.114 [−0.225, +0.254] | no / no | −0.093 [−0.154, −0.061] |
| email | +0.195 [+0.151, +0.242] | +0.158 [+0.133, +0.181] | +0.508 [+0.450, +0.558] | no / no | −0.049 [−0.067, −0.033] |
| coding | +0.151 [+0.113, +0.193] | +0.086 [+0.063, +0.107] | +0.335 [+0.197, +0.409] | no / no | −0.059 [−0.075, −0.045] |

The estimator reproduces the published LR-B, LR-C and 3-seed GRU values in `domain_shift_results.json` exactly (maximum absolute difference 0.0). With the same code, LR-C against LR-B is established in web, email and coding on AUROC, as published. The 10-seed GRU against LR-C is established in medical, banking, email and coding on AUROC and in medical on AUPRC at 10%, which are the same folds as the 3-seed result.

## 4. R4, AgentDrift to AgentDojo (r4_control.py populations, evalkit bootstrap B = 2,000, seed 12345)

| Population | n pos / n neg | SE transfer | SE within (UT) | within − transfer | SE-W0 transfer | SE-W0 within | GRU transfer | GRU within | LR-C transfer |
|---|---|---|---|---|---|---|---|---|---|
| P0 | 2,294 / 392 | 0.481 [0.418, 0.555] | 0.956 [0.924, 0.977] | +0.475 [+0.395, +0.544] | 0.503 [0.413, 0.579] | 0.963 [0.916, 0.985] | 0.510 [0.474, 0.542] | 0.845 [0.800, 0.872] | 0.473 [0.440, 0.505] |
| C1 | 2,294 / 327 | 0.467 [0.403, 0.533] | 0.949 [0.875, 0.977] | +0.482 [+0.381, +0.561] | 0.491 [0.409, 0.559] | 0.952 [0.910, 0.979] | 0.505 [0.468, 0.542] | 0.850 [0.812, 0.890] | 0.472 [0.437, 0.509] |
| P0F | 2,294 / 5,379 | 0.572 [0.525, 0.647] | 0.669 [0.630, 0.704] | +0.098 [+0.004, +0.165] | 0.587 [0.525, 0.640] | 0.645 [0.597, 0.684] | 0.646 [0.632, 0.662] | 0.833 [0.799, 0.859] | 0.626 [0.613, 0.640] |
| model-balanced, native counts | 2,287 / 308 | 0.477 [0.400, 0.551] | 0.961 [0.924, 0.984] | +0.484 [+0.403, +0.566] | 0.500 [0.405, 0.578] | 0.968 [0.919, 0.990] | 0.532 [0.490, 0.571] | 0.834 [0.786, 0.864] | 0.486 [0.448, 0.524] |

Transfer is near chance on P0, C1 and the model-balanced population, while the within-AgentDojo control reaches 0.949 to 0.961. That is Outcome 3, and with this detector cross-benchmark failure now holds for three detector families and a content reference. With the unsuccessful attacks added as negatives (P0F), transfer is 0.572 [0.525, 0.647]. The interval excludes 0.5, as it does for LR-C (0.626) and the GRU (0.646). Within-AgentDojo SE falls to 0.669 on P0F, because it was trained on P0 rows only and never saw an unsuccessful attack.

**SE-W0, transfer and within with the four world features zeroed in training and test (amendment 4).**

| Contrast (P0, AUROC, paired) | Difference |
|---|---|
| SE transfer − SE-W0 transfer | −0.023 [−0.117, +0.083] |
| SE within − SE-W0 within | −0.007 [−0.043, +0.044] |
| SE transfer − LR-B transfer | −0.001 [−0.057, +0.068] |
| SE transfer − GRU transfer | −0.029 [−0.089, +0.042] |

Zeroing the features moves neither transfer nor within by a detectable amount. No R4 conclusion depends on the per-suite W of D8, so D8 does not matter for R4 and we say so. SE transfer is also not distinguishable from LR-B or the GRU.

**Support of the world features in AgentDojo (D8).**

| Suite | Traces (P0 + unsuccessful) | Any address in arguments | Any URL in arguments | Any non-zero world feature |
|---|---|---|---|---|
| banking | 1,450 | 0.000 | 0.000 | 0.000 |
| slack | 1,601 | 0.347 | 0.816 | 0.871 |
| travel | 2,083 | 0.067 | 0.000 | 0.067 |
| workspace | 2,539 | 0.545 | 0.005 | 0.545 |

In banking no trace has an address or URL in its arguments, so all four features are constant there and carry no signal. In travel only 6.7% of traces have any non-zero feature (16.5% of P0 traces and 5.1% of unsuccessful attacks), so the features are close to constant there as well.

**Cluster bootstrap over suite and user task (117 clusters), secondary.**

| Method (P0) | AUROC, suite and user-task cluster bootstrap |
|---|---|
| SE transfer | 0.481 [0.403, 0.560] |
| SE within | 0.956 [0.914, 0.983] |
| SE-W0 transfer | 0.503 [0.404, 0.583] |
| SE-W0 within | 0.963 [0.906, 0.990] |
| GRU transfer | 0.510 [0.457, 0.551] |
| GRU within | 0.845 [0.792, 0.886] |
| LR-C transfer | 0.473 [0.427, 0.512] |
| LR-C within | 0.797 [0.749, 0.845] |

**Feature-block offset probe (amendment 4).** This is reported separately and is not SE-W0. The ten AgentDrift-trained SE models are scored on AgentDojo with the features zeroed and no refit.

| Feature-block offset probe (P0, no refit, not SE-W0) | Mean over 10 seeds |
|---|---|
| logit shift, AgentDojo negatives | -0.181 |
| logit shift, AgentDojo positives | -0.481 |
| logit shift, AgentDrift training negatives | -0.071 |
| AUROC with features / zeroed | 0.481 / 0.464 |

The shift is small, a fraction of a logit, and AUROC moves from 0.481 to 0.464. The features are zero for most AgentDrift training steps too (means 0.07 to 0.11), so zero is not an input these models never saw. The probe therefore shows little of the offset mechanism in this block, and it is kept out of every SE-W0 statement.

## 5. Offset analogue and embedding drift (plan Sec. 7)

The offset is the mean detection logit on held-out pool-E negatives minus the mean on the fit set's training negatives, per fold and seed. The GRU column applies the same estimator to the 10-seed GRU. **It was added after the SE values were seen and is not pre-registered.** The published GRU value of +6.291 is a tool-pathway component, so it is not the same estimator as this column.

| Held-out domain | SE offset, mean over 10 seeds | SE range over seeds | SE, benign and hard-negative training negatives only | GRU, same estimator (post hoc) | embedding drift, mean cosine distance |
|---|---|---|---|---|---|
| medical | +5.161 | +3.899 to +6.689 | +4.767 | +5.920 | 0.296 |
| banking | +0.524 | -0.040 to +1.522 | +0.262 | +5.592 | 0.283 |
| web | +7.420 | +3.994 to +8.821 | +7.092 | +7.280 | 0.410 |
| email | +0.692 | +0.373 to +1.291 | +0.313 | +7.133 | 0.260 |
| coding | +5.795 | +4.012 to +7.084 | +5.497 | +7.382 | 0.383 |
| R1 (reference) | +0.643 | +0.281 to +0.984 | +0.201 | +0.661 | 0.041 |

Measured with the same full-logit estimator, the GRU is +5.6 to +7.4 in all five folds and +0.661 at R1. So SE's offset is as large as the GRU's in three folds and close to the in-distribution level in banking and email. It is not uniformly small.

**Embedding drift, mean cosine distance from each held-out step to its nearest training step.**

| Rung | Mean cosine distance to nearest training step | Median | 90th percentile | Share of steps with an exact match |
|---|---|---|---|---|
| R1 | 0.041 | 0.027 | 0.103 | 0.119 |
| R1-wd split 0 | 0.040 | 0.025 | 0.103 | 0.147 |
| R2 | 0.080 | 0.053 | 0.199 | 0.043 |
| R3 medical | 0.296 | 0.298 | 0.368 | 0.000 |
| R3 banking | 0.283 | 0.279 | 0.415 | 0.000 |
| R3 web | 0.410 | 0.423 | 0.536 | 0.000 |
| R3 email | 0.260 | 0.248 | 0.365 | 0.000 |
| R3 coding | 0.383 | 0.384 | 0.508 | 0.000 |
| R4, P0 | 0.320 | 0.304 | 0.471 | 0.000 |
| R4, unsuccessful attacks | 0.298 | 0.284 | 0.410 | 0.000 |

R1 and R1-wd are close to their training steps (0.04, and 12 to 15% of steps have an exact duplicate in training). R2 is twice as far, and R3 and R4 are 6 to 10 times as far, with no exact duplicates. The two folds with in-distribution-sized offsets, banking and email, have the smallest R3 drift (0.283 and 0.260). Web and coding, with the largest offsets, have the largest drift (0.410 and 0.383). Medical breaks the order, with a large offset at drift 0.296, so the association is loose and five folds cannot carry a correlation.

## 6. Graded coverage inside R1 (masks of oov_mechanism.py, paired)

The tool name is replaced by `unknown_tool` in the step text before embedding. The ten R1 models are scored at test time and are not refit.

| Level | Achieved coverage | SE AUROC | SE FPR at θr80 | SE recall at θr80 | SE − LR-B | LR-C (published) | GRU (published) | LR-B |
|---|---|---|---|---|---|---|---|---|
| 1.0 | 1.000 | 0.998 [0.996, 0.999] | 0.001 [0.000, 0.001] | 0.796 [0.772, 0.819] | +0.167 [+0.148, +0.187] | 0.879 | 0.936 | 0.830 |
| 0.75 | 0.775 | 0.997 [0.995, 0.998] | 0.000 [0.000, 0.001] | 0.774 [0.750, 0.797] | +0.167 [+0.147, +0.186] | 0.853 | 0.874 | 0.830 |
| 0.5 | 0.483 | 0.996 [0.994, 0.997] | 0.000 [0.000, 0.001] | 0.750 [0.725, 0.774] | +0.166 [+0.147, +0.186] | 0.825 | 0.792 | 0.830 |
| 0.25 | 0.225 | 0.995 [0.993, 0.997] | 0.000 [0.000, 0.001] | 0.726 [0.699, 0.751] | +0.165 [+0.146, +0.185] | 0.800 | 0.720 | 0.830 |
| 0.0 | 0.000 | 0.994 [0.991, 0.997] | 0.000 [0.000, 0.000] | 0.711 [0.683, 0.737] | +0.164 [+0.145, +0.184] | 0.787 | 0.674 | 0.830 |

SE barely moves as tool coverage falls from 1.0 to 0.0 (AUROC 0.998 to 0.994). Over the same masks LR-C falls from 0.879 to 0.787 and the GRU from 0.936 to 0.674. The SE minus LR-B advantage is constant at +0.16 to +0.17. A masked tool name becomes an ordinary point of the frozen space, and this sweep shows that such a point does not cost ranking performance. Two limits apply. Tool identity can still leak through arguments and observations, which the mask does not touch. Mean embedding drift of the R1 test steps also rises only from 0.041 to 0.126 at full masking, far below R3's 0.26 to 0.41. Per Section 8 of the plan, this shows that the input representation degrades gracefully, not that SE is a better detector.

## 7. Observation-length padding (replaces SE-noobs, amendment 3)

The filler of `features_obslen.py` is appended to every injection_point observation of attacked R1 test trajectories, and the padded steps are re-embedded. The ten R1 models are scored at test time with their training θr80 and are not refit. The published columns use the same protocol.

| Filler characters | SE recall at θr80 | SE AUROC | LR-C recall (published) | LR-B recall (published) | GRU recall (published) |
|---|---|---|---|---|---|
| 0 | 0.796 [0.772, 0.819] | 0.998 [0.996, 0.999] | 0.771 [0.744, 0.799] | 0.795 [0.768, 0.823] | 0.682 [0.646, 0.717] |
| 100 | 0.685 [0.655, 0.713] | 0.990 [0.987, 0.993] | 0.634 [0.602, 0.664] | 0.664 [0.632, 0.698] | 0.578 [0.543, 0.611] |
| 1,200 | 0.531 [0.498, 0.564] | 0.883 [0.868, 0.897] | 0.033 [0.021, 0.046] | 0.161 [0.137, 0.188] | 0.000 [0.000, 0.000] |

Padding hurts SE too, but much less than the detectors that read length as a scalar. At 1,200 characters SE recall falls by 0.265 [0.241, 0.290] to 0.531, while LR-C falls to 0.033 and the GRU to 0.000. AUROC falls by 0.115 to 0.883. The attacker-written observation still reaches SE through the embedding, so the surface-cue concern applies to all four detectors, at different strengths. Fourteen padded steps were truncated at 1,200 characters (Section 0).

## 8. R2 against the source paper's reported setting (D5, D6)

This is the closest point to DriftNet's own setting, using all 1,722 task-disjoint test trajectories at threshold 0.5. **It is not a like-for-like comparison.** We drop THOUGHT (D6), use a literal reading of the fourth feature (D1), carve validation from task-disjoint training rather than using the shipped validation part (D7), and train 10 CPU seeds where the source trains one GPU run with seed 42.

| Quantity at threshold 0.5 | Our reimplementation, mean over 10 seeds (min to max) | DriftNet as published, single run (not ours) |
|---|---|---|
| precision | 0.975 (0.965 to 0.982) | 0.982 |
| recall | 0.970 (0.955 to 0.986) | 0.985 |
| F1 | 0.973 (0.967 to 0.979) | 0.983 |
| flag rate, benign | 0.021 (0.015 to 0.027) | 0.015 |
| flag rate, failed attack | 0.005 (0.005 to 0.009) | 0.000 |
| flag rate, hard negative | 0.036 (0.020 to 0.059) | 0.029 |

Our F1 is 0.973 against the published 0.983. This gap is a result and is not to be closed. D6 is its most plausible single cause. Nothing was tuned in response.

## 9. GRU at 10 seeds (executed pipeline, `build_gru_cache.py` with seeds and repetitions 0 to 9)

| Setting | GRU AUROC, 3 seeds (paper) | GRU AUROC, 10 seeds | per-seed range, 10 seeds | seeds 0 to 2 reproduce cache |
|---|---|---|---|---|
| R1 | 0.936 | 0.932 | 0.923 to 0.938 | yes |
| R1wd s0 | 0.956 | 0.955 | 0.949 to 0.961 | yes |
| R1wd s1 | 0.949 | 0.948 | 0.939 to 0.953 | yes |
| R1wd s2 | 0.962 | 0.961 | 0.955 to 0.966 | yes |
| R1wd s3 | 0.959 | 0.958 | 0.950 to 0.965 | yes |
| R1wd s4 | 0.956 | 0.957 | 0.948 to 0.961 | yes |
| R2 | 0.915 | 0.915 | 0.909 to 0.924 | yes |
| R3 medical | 0.639 | 0.649 | 0.626 to 0.693 | yes |
| R3 banking | 0.736 | 0.721 | 0.671 to 0.786 | yes |
| R3 web | 0.726 | 0.718 | 0.704 to 0.752 | yes |
| R3 email | 0.671 | 0.656 | 0.616 to 0.715 | yes |
| R3 coding | 0.695 | 0.683 | 0.660 to 0.729 | yes |
| R4 | 0.513 | 0.510 | 0.490 to 0.522 | yes |
| WUT | 0.840 | 0.845 | 0.808 to 0.861 | yes |

Seeds 0 to 2 reproduce the 3-seed cache. Score differences are below 5e-5 at every setting (2.3e-6 for R1 seed 0), and the θr80 decisions are identical. The 10-seed means differ from the 3-seed means by 0.015 or less. The R3 reversal pattern against LR-C is unchanged (Section 3). These values supersede the 3-seed values wherever the paper reports a GRU point estimate, and the 3-seed files are kept.

## 10. Deviations from the plan

Declared in the plan (Sections 11 to 15, hashed before the run):

- **D1, restated.** Three of the four world features are recovered from the source together with AgentDrift's rule for an external address. `args_external_email_frac` is named in the source but has no formula. Its worked example (1, 1, 0.33, 0) is inconsistent with the literal reading, external matches over all matches, which we use because it is the only reading the name supports. The URL regex and host rule are ours.
- **D2.** Within-AgentDojo fits use the trajectory loss only, because AgentDojo has no step labels.
- **D3, restated.** The 4-way localisation head is trained but not evaluated.
- **D4.** No hyperparameter search. The source's sweep-selected configuration was adopted as published.
- **D5.** Our R2 is the closest point to the source's task-disjoint setting.
- **D6, restated.** THOUGHT is dropped so that both benchmarks share one representation. Our R2 is therefore not directly comparable to the source, and no gap may be read as a reimplementation failure.
- **D7 (author's amendment 2).** Validation is a stratified 10% carve-out at every rung, matching the executed GRU pipeline.
- **D8.** World features on AgentDojo use a per-suite W derived from the environment files at commit 089ed468. SE-W0 shows that no R4 conclusion depends on it.
- **D9.** `agentdojo_failed.jsonl` was embedded so that P0F can be reported.
- **D10.** No deviation. The parameter count equals the published one.
- **D11 (amendments 5 and 6).** Embedding ran on CPU on cn001 because the GPU queue was about four hours long. The GPU check on 1,000 texts agrees to 2.5e-7.
- **D12.** SE-noobs was retired and replaced by the padding test. SE-W0 and the separately labelled offset probe were added.

Arising during execution, recorded in `revision/sencoder_run/RUNTIME_DEVIATIONS.md` before any analysis:

- **D13.** The 10-minute stop rule fired once. SE R4 seed 1 took 630.1 s while 15 fits shared 32 CPUs, and its full 40 epochs would take roughly 300 s alone. With the author's agreement the remaining 413 units ran with 6 workers. Threads per fit, hyperparameters, seeds and the 157 finished units were unchanged, and the rule stayed active. The slowest later fit took 565 s. A strict reading of the rule would have ended the study at 157 units.

Implementation choices not stated in the plan, fixed before the run:

- The offset analogue uses all training negatives of the fit set. The benign and hard-negative variant is reported alongside.
- Reversal uses `domain_shift.py`'s estimator, imported unchanged. The ladder tables use `evalkit.bootstrap` as Section 6 of the plan requires.
- GRU C1 within-AgentDojo follows the `gru_C1_UT` logic of `r4_deep.py` with repetitions 0 to 9.
- The GRU column of Section 5 is post hoc, and is labelled so in the JSON.

## 11. What this does not establish

These numbers bound a reimplementation and say nothing about DriftNet's own performance. SE's in-distribution near-ceiling (0.998) is partly a property of AgentDrift. In R1, 12% of test steps have an exact duplicate among training steps. A frozen encoder that holds up under masking and shift is not by that fact a better detector, only one whose input representation degrades gracefully. The offset finding rests on five folds, and its association with drift is loose. The outcome set did not foresee a large offset without a reversal, so the reading in Section 1 is ours, stated against the plan rather than drawn from it.
