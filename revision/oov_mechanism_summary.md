# OOV mechanism of the R3 reversals (review M1, Q1, m6 <UNK> norm)

Generated from `revision/oov_mechanism_results.json` (script `revision/oov_mechanism.py`, SHA-256 `02460c7104c222a9...`). Every number below is a key of that JSON; the key path is given in brackets. Intervals are 95% percentile bootstrap intervals (B = 2000). 'Detected' means the interval excludes 0; 'Holm' means detected after Holm correction within the stated family. Thresholds are the cached training thresholds theta_r80; nothing was tuned on test data.

**Reproduction checks.** Weighted-bootstrap AUROC engine vs evalkit.auroc: max abs diff 0.0e+00 [meta.engine_check_max_abs_diff_vs_evalkit_auroc]. Refitted R1 GRU models vs cached scores (max abs diff per model): GRU_seed0 0.0e+00, GRU_seed1 0.0e+00, GRU_seed2 0.0e+00, GRU-drop0.2_seed0 0.0e+00, GRU-drop0.2_seed1 0.0e+00, GRU-drop0.2_seed2 0.0e+00 [checks_r1_gru_refit]. Graded-coverage level 1 vs cache: LR-C_level1_max_abs_diff_vs_cache 2.2e-16, LR-C_level1_tool_part_max_abs_diff_vs_cache 4.0e-15, LR-C-drop_level1_max_abs_diff_vs_cache 0.0e+00, LR-C-drop_level1_tool_part_max_abs_diff_vs_cache 3.6e-15, GRU_seed0_level1_max_abs_diff_vs_cache 0.0e+00, GRU_seed1_level1_max_abs_diff_vs_cache 0.0e+00, GRU_seed2_level1_max_abs_diff_vs_cache 0.0e+00, GRU-drop0.2_seed0_level1_max_abs_diff_vs_cache 0.0e+00, GRU-drop0.2_seed1_level1_max_abs_diff_vs_cache 0.0e+00, GRU-drop0.2_seed2_level1_max_abs_diff_vs_cache 0.0e+00 [s2_graded.checks].

## M1, action 1: LR-C tool-block offset w_A^T c per R3 fold

**Question.** At zero coverage every held-out trajectory has an all-zero raw tool block, which the training scaler maps to a constant vector c. How large is the resulting logit offset w_A^T c, and how unusual is it relative to the training trajectories?

| Fold | LR-C offset | share of training rows >= offset (all / negatives / positives) | LR-C-drop offset | LR-C-drop share >= offset (all / negatives) |
|---|---|---|---|---|
| medical | +1.794 | 0.150 / 0.052 / 0.277 | -0.057 | 0.506 / 0.323 |
| banking | +1.899 | 0.113 / 0.042 / 0.196 | -0.057 | 0.531 / 0.341 |
| web | +1.702 | 0.130 / 0.046 / 0.238 | -0.033 | 0.498 / 0.342 |
| email | +1.790 | 0.107 / 0.047 / 0.185 | -0.021 | 0.505 / 0.350 |
| coding | +1.653 | 0.129 / 0.053 / 0.228 | -0.032 | 0.515 / 0.354 |

[s1_lrc_offset.settings.R3_<fold>.<LR-C|LR-C-drop>.offset / share_train_ge_offset / share_train_neg_ge_offset / share_train_pos_ge_offset]. The LR-C training tool contributions average 0 by construction; on training negatives they average -0.637, -0.599, -0.510, -0.491, -0.507 (medical ... coding), so the offset sits +2.431, +2.498, +2.213, +2.281, +2.160 logits above the mean training negative [offset_minus_train_neg_mean]. In every R3 fold every test row carries exactly the offset [test_tool_all_equal_offset = True]. For comparison, the same quantity for the models trained on the AgentDojo suites (LOSO) is banking -0.358, slack -2.217, travel -0.433, workspace +1.558 [s1_lrc_offset.settings.LOSO_<suite>.LR-C.offset].

**Plain reading.** The review's offsets are reproduced: LR-C adds +1.65 to +1.90 logits to every held-out trajectory. Only 10.7% to 15.0% of training trajectories reach that tool contribution: 4.2% to 5.3% of training negatives and 18.5% to 27.7% of training positives [s1_lrc_offset.R3_range]. So an all-zero tool block looks more attack-like to LR-C than the tool profile of most attacked training trajectories, and at the training threshold this raises recall and FPR together (action 3). The offset is the same for every held-out row, so it cannot change the ranking within a fold. It explains the threshold behaviour, not the AUROC loss. At zero coverage LR-C ranks held-out trajectories with its behavioural weights only, and those weights were fitted jointly with an informative tool block (action 2 shows this directly). Tool-block dropout removes the offset almost entirely (-0.057 to -0.021): half of its fit rows have an all-zero tool block. For the AgentDojo suites held out at zero coverage, the fitted offset is negative (-0.358 for the model that holds out banking, -2.217 for slack). There an unseen-tool trajectory is pushed towards benign, and alert rates fall rather than rise (action 5a). This concerns thresholds only.

## M1, action 2: graded tool coverage inside R1

**Question.** Does the C-versus-B reversal (and the GRU's collapse) appear inside R1, with no change of domain, when test-time tool coverage alone is reduced? level L masks round((1 - L) * 103) tool names drawn at random (without replacement) from the R1 benign-training vocabulary; masked names have their bag-of-tools column set to 0 for LR-C / LR-C-drop and are mapped to <UNK> for the GRUs; behavioural features and all fits and thresholds are unchanged. Levels 1 and 0 admit a single mask. mean over draws (and over the 3 training seeds for the GRUs) of AUROC / FPR / recall at the cached theta_r80; 95% interval = stratified trajectory bootstrap (B = 2000, rng 20201) of that draw- and seed-averaged metric, conditional on the drawn masks; draw_min / draw_max give the spread of the per-draw point values.

| Level | achieved coverage mean (range) | LR-C AUROC | LR-C-drop AUROC | LR-B AUROC | GRU AUROC | GRU-drop0.2 AUROC |
|---|---|---|---|---|---|---|
| 1.0 | 1.000 (1.000-1.000) | 0.879 [0.863, 0.895] | 0.882 [0.866, 0.898] | 0.830 [0.810, 0.848] | 0.936 [0.925, 0.946] | 0.943 [0.933, 0.953] |
| 0.75 | 0.775 (0.757-0.788) | 0.853 [0.836, 0.869] | 0.862 [0.845, 0.877] | 0.830 [0.810, 0.848] | 0.874 [0.862, 0.887] | 0.923 [0.912, 0.933] |
| 0.5 | 0.483 (0.461-0.536) | 0.825 [0.807, 0.841] | 0.846 [0.829, 0.861] | 0.830 [0.810, 0.848] | 0.792 [0.778, 0.807] | 0.888 [0.875, 0.899] |
| 0.25 | 0.225 (0.195-0.268) | 0.800 [0.780, 0.818] | 0.827 [0.808, 0.843] | 0.830 [0.810, 0.848] | 0.720 [0.700, 0.739] | 0.842 [0.826, 0.857] |
| 0.0 | 0.000 (0.000-0.000) | 0.787 [0.765, 0.808] | 0.822 [0.801, 0.840] | 0.830 [0.810, 0.848] | 0.674 [0.650, 0.699] | 0.794 [0.772, 0.813] |

| Level | LR-C FPR@theta_r80 | LR-C-drop FPR | LR-B FPR | GRU FPR | GRU-drop0.2 FPR |
|---|---|---|---|---|---|
| 1.0 | 0.192 [0.166, 0.219] | 0.187 [0.160, 0.213] | 0.284 [0.253, 0.315] | 0.045 [0.034, 0.057] | 0.054 [0.041, 0.068] |
| 0.75 | 0.263 [0.237, 0.289] | 0.182 [0.160, 0.206] | 0.284 [0.253, 0.315] | 0.169 [0.152, 0.186] | 0.057 [0.046, 0.069] |
| 0.5 | 0.442 [0.416, 0.470] | 0.246 [0.223, 0.271] | 0.284 [0.253, 0.315] | 0.383 [0.362, 0.405] | 0.070 [0.060, 0.082] |
| 0.25 | 0.546 [0.519, 0.575] | 0.221 [0.198, 0.245] | 0.284 [0.253, 0.315] | 0.586 [0.560, 0.612] | 0.055 [0.045, 0.066] |
| 0.0 | 0.657 [0.627, 0.687] | 0.235 [0.207, 0.264] | 0.284 [0.253, 0.315] | 0.737 [0.710, 0.766] | 0.053 [0.040, 0.066] |

| Level | LR-C - LR-B | LR-C-drop - LR-B | GRU - LR-B | GRU-drop0.2 - LR-B | GRU - LR-C |
|---|---|---|---|---|---|
| 1.0 | +0.049 [+0.034, +0.064] | +0.051 [+0.036, +0.067] | +0.105 [+0.087, +0.125] | +0.113 [+0.096, +0.131] | +0.057 [+0.043, +0.071] |
| 0.75 | +0.023 [+0.009, +0.037] | +0.031 [+0.018, +0.045] | +0.044 [+0.026, +0.064] | +0.093 [+0.077, +0.110] | +0.021 [+0.008, +0.035] |
| 0.5 | -0.005 [-0.016, +0.006] | +0.016 [+0.006, +0.025] | -0.038 [-0.056, -0.020] | +0.058 [+0.044, +0.072] | -0.033 [-0.045, -0.021] |
| 0.25 | -0.030 [-0.041, -0.020] | -0.004 [-0.009, +0.001] | -0.110 [-0.131, -0.089] | +0.012 [+0.000, +0.023] | -0.080 [-0.095, -0.066] |
| 0.0 | -0.043 [-0.055, -0.031] | -0.009 [-0.011, -0.007] | -0.156 [-0.179, -0.131] | -0.037 [-0.046, -0.027] | -0.113 [-0.129, -0.096] |

[s2_graded.levels.<level>.detectors.<det>.AUROC / FPR@t_r80; .contrasts.<A>-minus-<B>.AUROC; per-draw values in .per_draw, recall in .recall@t_r80]. Mean tool-pathway contribution on R1 negatives, level 1 -> level 0: LR-C -0.461 -> +1.759; LR-C-drop -0.497 -> -0.040; GRU -3.772 -> +5.660; GRU-drop0.2 -2.278 -> -0.379 [s2_graded.levels.<level>.detectors.<det>.tool_pathway_mean_on_negatives].

**Plain reading.** Lowering tool coverage inside R1 reproduces the R3 pattern with no change of domain. LR-C falls from 0.879 to 0.787 (change -0.091 [-0.110, -0.074]) [s2_graded.change_level0_minus_level1.LR-C.AUROC]. Its FPR at the training threshold rises by +0.466 [+0.433, +0.497]. The C > B advantage is gone at coverage about 0.5 (LR-C - LR-B -0.005 [-0.016, +0.006]), and at zero coverage the sign has reversed (-0.043 [-0.055, -0.031]). The GRU falls by -0.261 [-0.288, -0.235] and ends at -0.156 [-0.179, -0.131] relative to LR-B and -0.113 [-0.129, -0.096] relative to LR-C, with FPR rising by +0.693 [+0.663, +0.722]. Dropout reduces the AUROC loss (LR-C-drop -0.060 [-0.076, -0.045], GRU-drop0.2 -0.149 [-0.169, -0.130]). It largely removes the FPR rise (LR-C-drop +0.049 [+0.022, +0.076], GRU-drop0.2 -0.001 [-0.015, +0.015]). It does not remove the loss: at zero coverage LR-C-drop is still -0.009 [-0.011, -0.007] below LR-B, GRU-drop0.2 is -0.037 [-0.046, -0.027] relative to LR-B and +0.006 [-0.009, +0.022] relative to LR-C. So removing tool identity at test time is enough to produce both reversals within one distribution. This does not show that nothing else changes under R3, where the behavioural distribution also shifts. The masks also remove tool names at random across domains, whereas R3 removes whole domains.

## M1, action 3: dropout arms in R1 and R3, paired contrasts, plan 3.4 rule

**Question.** With tool-block dropout (LR-C-drop) and token dropout (GRU-drop0.2 primary, GRU-drop0.1 sensitivity), do the in-distribution orderings still reverse under R3? stratified trajectory bootstrap, B = 2000, one pool per population (R1 pool E; R1 pool E restricted to one domain = domain-matched comparator, plan 3.2; R3 fold pool E); all detectors share the resampled rows (paired). GRU arms: one training seed drawn per replicate (pre-registered evalkit convention), point = mean over the 3 seeds; see s7 for the two-level and seed-averaged alternatives. reversal in fold d iff the domain-matched R1 difference is detected after Holm AND the R3 difference is detected after Holm with the opposite sign.

AUROC (pool E) by population [s3_dropout.populations.<pop>.detectors.<det>.AUROC]:

| Population | LR-B | LR-C | LR-C-drop | GRU | GRU-drop0.2 | GRU-drop0.1 |
|---|---|---|---|---|---|---|
| R1 | 0.830 [0.810, 0.848] | 0.879 [0.863, 0.895] | 0.882 [0.866, 0.898] | 0.936 [0.924, 0.947] | 0.943 [0.931, 0.954] | 0.944 [0.933, 0.954] |
| R1dm_medical | 0.793 [0.745, 0.841] | 0.786 [0.738, 0.833] | 0.797 [0.750, 0.841] | 0.916 [0.884, 0.944] | 0.912 [0.879, 0.941] | 0.923 [0.893, 0.950] |
| R1dm_banking | 0.832 [0.781, 0.878] | 0.853 [0.805, 0.896] | 0.854 [0.807, 0.898] | 0.915 [0.876, 0.951] | 0.928 [0.892, 0.961] | 0.924 [0.888, 0.957] |
| R1dm_web | 0.886 [0.853, 0.918] | 0.927 [0.900, 0.953] | 0.925 [0.897, 0.951] | 0.925 [0.892, 0.955] | 0.947 [0.919, 0.970] | 0.943 [0.915, 0.967] |
| R1dm_email | 0.804 [0.758, 0.850] | 0.894 [0.860, 0.926] | 0.898 [0.865, 0.930] | 0.966 [0.943, 0.985] | 0.969 [0.949, 0.985] | 0.969 [0.949, 0.985] |
| R1dm_coding | 0.845 [0.803, 0.882] | 0.894 [0.859, 0.924] | 0.897 [0.861, 0.927] | 0.944 [0.905, 0.969] | 0.948 [0.925, 0.968] | 0.948 [0.923, 0.969] |
| R3_medical | 0.808 [0.790, 0.827] | 0.745 [0.724, 0.766] | 0.797 [0.778, 0.816] | 0.639 [0.609, 0.668] | 0.747 [0.725, 0.769] | 0.722 [0.700, 0.745] |
| R3_banking | 0.854 [0.834, 0.873] | 0.828 [0.807, 0.849] | 0.853 [0.833, 0.871] | 0.736 [0.700, 0.774] | 0.848 [0.827, 0.866] | 0.834 [0.805, 0.858] |
| R3_web | 0.896 [0.882, 0.910] | 0.862 [0.846, 0.877] | 0.894 [0.880, 0.908] | 0.726 [0.690, 0.768] | 0.870 [0.850, 0.887] | 0.855 [0.838, 0.872] |
| R3_email | 0.792 [0.773, 0.812] | 0.739 [0.717, 0.761] | 0.777 [0.757, 0.797] | 0.671 [0.625, 0.733] | 0.766 [0.724, 0.797] | 0.749 [0.712, 0.789] |
| R3_coding | 0.852 [0.836, 0.868] | 0.814 [0.796, 0.832] | 0.847 [0.830, 0.863] | 0.695 [0.652, 0.744] | 0.811 [0.777, 0.837] | 0.798 [0.776, 0.821] |

Recall and FPR at theta_r80 [s3_dropout.populations.<pop>.detectors.<det>.recall@t_r80 / FPR@t_r80]:

| Population | LR-B rec / FPR | LR-C rec / FPR | LR-C-drop rec / FPR | GRU rec / FPR | GRU-drop0.2 rec / FPR | GRU-drop0.1 rec / FPR |
|---|---|---|---|---|---|---|
| R1 | 0.795 [0.768, 0.823] / 0.284 [0.253, 0.315] | 0.771 [0.741, 0.800] / 0.192 [0.166, 0.219] | 0.765 [0.735, 0.794] / 0.187 [0.160, 0.213] | 0.682 [0.649, 0.718] / 0.045 [0.028, 0.062] | 0.712 [0.677, 0.745] / 0.054 [0.036, 0.074] | 0.691 [0.655, 0.726] / 0.044 [0.029, 0.061] |
| R1dm_medical | 0.694 [0.628, 0.761] / 0.236 [0.170, 0.303] | 0.650 [0.583, 0.717] / 0.321 [0.248, 0.394] | 0.644 [0.578, 0.711] / 0.303 [0.236, 0.376] | 0.585 [0.489, 0.700] / 0.044 [0.012, 0.085] | 0.617 [0.522, 0.728] / 0.057 [0.018, 0.121] | 0.587 [0.511, 0.667] / 0.044 [0.012, 0.085] |
| R1dm_banking | 0.802 [0.730, 0.874] / 0.255 [0.188, 0.321] | 0.613 [0.522, 0.712] / 0.121 [0.073, 0.170] | 0.604 [0.513, 0.694] / 0.109 [0.067, 0.158] | 0.498 [0.378, 0.613] / 0.020 [0.000, 0.049] | 0.534 [0.441, 0.631] / 0.016 [0.000, 0.042] | 0.513 [0.405, 0.622] / 0.020 [0.000, 0.042] |
| R1dm_web | 0.861 [0.811, 0.911] / 0.255 [0.194, 0.321] | 0.906 [0.861, 0.944] / 0.164 [0.115, 0.218] | 0.883 [0.833, 0.928] / 0.182 [0.127, 0.236] | 0.733 [0.650, 0.806] / 0.071 [0.024, 0.121] | 0.787 [0.717, 0.856] / 0.079 [0.036, 0.127] | 0.748 [0.683, 0.817] / 0.061 [0.024, 0.103] |
| R1dm_email | 0.794 [0.733, 0.850] / 0.358 [0.285, 0.430] | 0.789 [0.728, 0.844] / 0.127 [0.079, 0.182] | 0.789 [0.728, 0.850] / 0.121 [0.073, 0.176] | 0.774 [0.689, 0.850] / 0.030 [0.006, 0.061] | 0.780 [0.711, 0.844] / 0.036 [0.012, 0.067] | 0.783 [0.722, 0.844] / 0.028 [0.006, 0.054] |
| R1dm_coding | 0.828 [0.772, 0.878] / 0.315 [0.248, 0.388] | 0.839 [0.783, 0.889] / 0.224 [0.164, 0.291] | 0.844 [0.789, 0.894] / 0.218 [0.158, 0.285] | 0.750 [0.644, 0.828] / 0.059 [0.024, 0.103] | 0.776 [0.706, 0.839] / 0.081 [0.030, 0.139] | 0.754 [0.683, 0.817] / 0.067 [0.018, 0.121] |
| R3_medical | 0.560 [0.530, 0.590] / 0.150 [0.127, 0.173] | 0.828 [0.805, 0.851] / 0.503 [0.471, 0.534] | 0.450 [0.421, 0.479] / 0.121 [0.101, 0.141] | 0.938 [0.895, 0.971] / 0.717 [0.641, 0.774] | 0.174 [0.081, 0.301] / 0.036 [0.004, 0.094] | 0.101 [0.066, 0.158] / 0.015 [0.003, 0.035] |
| R3_banking | 0.829 [0.799, 0.858] / 0.278 [0.249, 0.309] | 0.967 [0.952, 0.979] / 0.660 [0.631, 0.692] | 0.767 [0.733, 0.800] / 0.221 [0.196, 0.249] | 0.622 [0.002, 0.968] / 0.438 [0.000, 0.716] | 0.328 [0.133, 0.468] / 0.033 [0.005, 0.063] | 0.356 [0.129, 0.684] / 0.068 [0.007, 0.190] |
| R3_web | 0.825 [0.801, 0.848] / 0.197 [0.172, 0.223] | 0.990 [0.983, 0.996] / 0.707 [0.677, 0.735] | 0.786 [0.762, 0.813] / 0.168 [0.144, 0.191] | 0.917 [0.808, 0.997] / 0.623 [0.471, 0.774] | 0.230 [0.043, 0.522] / 0.024 [0.000, 0.080] | 0.317 [0.082, 0.555] / 0.042 [0.000, 0.110] |
| R3_email | 0.833 [0.811, 0.857] / 0.418 [0.388, 0.449] | 0.962 [0.949, 0.974] / 0.722 [0.693, 0.751] | 0.775 [0.750, 0.802] / 0.378 [0.346, 0.409] | 0.973 [0.941, 0.994] / 0.764 [0.690, 0.839] | 0.226 [0.182, 0.269] / 0.039 [0.026, 0.051] | 0.176 [0.129, 0.237] / 0.031 [0.014, 0.053] |
| R3_coding | 0.835 [0.813, 0.858] / 0.283 [0.256, 0.312] | 0.974 [0.963, 0.983] / 0.672 [0.641, 0.702] | 0.788 [0.765, 0.815] / 0.248 [0.221, 0.277] | 0.720 [0.260, 0.955] / 0.486 [0.079, 0.710] | 0.397 [0.288, 0.467] / 0.100 [0.049, 0.153] | 0.210 [0.133, 0.344] / 0.042 [0.016, 0.091] |

Paired AUROC differences, domain-matched R1 and R3, with Holm over the five folds [s3_dropout.reversal_rule_3_4.<A>-minus-<B>.per_fold.<fold>]:

| Contrast | Fold | R1 domain-matched | Holm p | R3 | Holm p | reversal (3.4) |
|---|---|---|---|---|---|---|
| LR-C-minus-LR-B | medical | -0.007 [-0.036, +0.024] | 0.6107 | -0.063 [-0.073, -0.052] | 0.0050 | no |
| LR-C-minus-LR-B | banking | +0.021 [-0.012, +0.054] | 0.4278 | -0.026 [-0.037, -0.014] | 0.0050 | no |
| LR-C-minus-LR-B | web | +0.041 [+0.016, +0.066] | 0.0050 | -0.034 [-0.046, -0.023] | 0.0050 | yes |
| LR-C-minus-LR-B | email | +0.090 [+0.047, +0.132] | 0.0050 | -0.053 [-0.063, -0.043] | 0.0050 | yes |
| LR-C-minus-LR-B | coding | +0.049 [+0.020, +0.080] | 0.0090 | -0.038 [-0.048, -0.026] | 0.0050 | yes |
| GRU-minus-LR-C | medical | +0.130 [+0.086, +0.172] | 0.0050 | -0.107 [-0.130, -0.087] | 0.0050 | yes |
| GRU-minus-LR-C | banking | +0.063 [+0.018, +0.108] | 0.0240 | -0.092 [-0.135, -0.061] | 0.0050 | yes |
| GRU-minus-LR-C | web | -0.002 [-0.030, +0.024] | 0.8975 | -0.137 [-0.167, -0.100] | 0.0050 | no |
| GRU-minus-LR-C | email | +0.071 [+0.042, +0.104] | 0.0050 | -0.069 [-0.106, -0.009] | 0.0070 | yes |
| GRU-minus-LR-C | coding | +0.049 [+0.009, +0.089] | 0.0340 | -0.119 [-0.154, -0.078] | 0.0050 | yes |
| LR-C-drop-minus-LR-B | medical | +0.003 [-0.025, +0.033] | 0.8496 | -0.011 [-0.013, -0.009] | 0.0050 | no |
| LR-C-drop-minus-LR-B | banking | +0.023 [-0.010, +0.058] | 0.3738 | -0.001 [-0.003, +0.001] | 0.2319 | no |
| LR-C-drop-minus-LR-B | web | +0.039 [+0.014, +0.065] | 0.0090 | -0.002 [-0.004, -0.001] | 0.0260 | yes |
| LR-C-drop-minus-LR-B | email | +0.094 [+0.051, +0.137] | 0.0050 | -0.015 [-0.017, -0.013] | 0.0050 | yes |
| LR-C-drop-minus-LR-B | coding | +0.052 [+0.023, +0.082] | 0.0080 | -0.005 [-0.007, -0.003] | 0.0050 | yes |
| GRU-drop0.2-minus-LR-C | medical | +0.126 [+0.085, +0.167] | 0.0050 | +0.001 [-0.017, +0.019] | 1.0000 | no |
| GRU-drop0.2-minus-LR-C | banking | +0.075 [+0.037, +0.116] | 0.0050 | +0.019 [+0.004, +0.033] | 0.0400 | no |
| GRU-drop0.2-minus-LR-C | web | +0.019 [-0.005, +0.043] | 0.1109 | +0.008 [-0.010, +0.021] | 1.0000 | no |
| GRU-drop0.2-minus-LR-C | email | +0.075 [+0.047, +0.106] | 0.0050 | +0.026 [-0.007, +0.055] | 1.0000 | no |
| GRU-drop0.2-minus-LR-C | coding | +0.054 [+0.025, +0.086] | 0.0050 | -0.003 [-0.036, +0.019] | 1.0000 | no |
| GRU-drop0.2-minus-LR-B | medical | +0.119 [+0.073, +0.165] | 0.0050 | -0.061 [-0.074, -0.048] | 0.0050 | yes |
| GRU-drop0.2-minus-LR-B | banking | +0.097 [+0.054, +0.140] | 0.0050 | -0.006 [-0.017, +0.004] | 0.2099 | no |
| GRU-drop0.2-minus-LR-B | web | +0.060 [+0.027, +0.095] | 0.0050 | -0.026 [-0.041, -0.015] | 0.0050 | yes |
| GRU-drop0.2-minus-LR-B | email | +0.165 [+0.120, +0.210] | 0.0050 | -0.027 [-0.059, -0.003] | 0.0320 | yes |
| GRU-drop0.2-minus-LR-B | coding | +0.103 [+0.066, +0.143] | 0.0050 | -0.041 [-0.069, -0.022] | 0.0050 | yes |
| GRU-drop0.1-minus-LR-C | medical | +0.137 [+0.098, +0.177] | 0.0050 | -0.023 [-0.040, -0.007] | 0.0150 | yes |
| GRU-drop0.1-minus-LR-C | banking | +0.071 [+0.031, +0.114] | 0.0050 | +0.005 [-0.020, +0.026] | 1.0000 | no |
| GRU-drop0.1-minus-LR-C | web | +0.016 [-0.008, +0.041] | 0.1959 | -0.007 [-0.022, +0.007] | 0.9565 | no |
| GRU-drop0.1-minus-LR-C | email | +0.075 [+0.046, +0.105] | 0.0050 | +0.010 [-0.020, +0.048] | 1.0000 | no |
| GRU-drop0.1-minus-LR-C | coding | +0.053 [+0.024, +0.087] | 0.0050 | -0.016 [-0.034, +0.004] | 0.4518 | no |
| GRU-drop0.1-minus-LR-B | medical | +0.130 [+0.086, +0.176] | 0.0050 | -0.086 [-0.099, -0.073] | 0.0050 | yes |
| GRU-drop0.1-minus-LR-B | banking | +0.092 [+0.048, +0.140] | 0.0050 | -0.021 [-0.044, -0.004] | 0.0130 | yes |
| GRU-drop0.1-minus-LR-B | web | +0.057 [+0.024, +0.090] | 0.0050 | -0.041 [-0.054, -0.029] | 0.0050 | yes |
| GRU-drop0.1-minus-LR-B | email | +0.165 [+0.119, +0.211] | 0.0050 | -0.043 [-0.072, -0.010] | 0.0100 | yes |
| GRU-drop0.1-minus-LR-B | coding | +0.102 [+0.064, +0.144] | 0.0050 | -0.053 [-0.069, -0.037] | 0.0050 | yes |
| GRU-drop0.2-minus-LR-C-drop | medical | +0.116 [+0.076, +0.155] | 0.0050 | -0.050 [-0.063, -0.037] | 0.0050 | yes |
| GRU-drop0.2-minus-LR-C-drop | banking | +0.074 [+0.035, +0.114] | 0.0050 | -0.005 [-0.016, +0.005] | 0.6217 | no |
| GRU-drop0.2-minus-LR-C-drop | web | +0.021 [-0.004, +0.046] | 0.0979 | -0.024 [-0.039, -0.013] | 0.0050 | no |
| GRU-drop0.2-minus-LR-C-drop | email | +0.071 [+0.043, +0.100] | 0.0050 | -0.012 [-0.044, +0.013] | 0.7946 | no |
| GRU-drop0.2-minus-LR-C-drop | coding | +0.051 [+0.023, +0.084] | 0.0050 | -0.036 [-0.064, -0.017] | 0.0050 | yes |
| GRU-minus-LR-B | medical | +0.123 [+0.072, +0.170] | 0.0050 | -0.169 [-0.198, -0.145] | 0.0050 | yes |
| GRU-minus-LR-B | banking | +0.084 [+0.035, +0.133] | 0.0060 | -0.118 [-0.157, -0.081] | 0.0050 | yes |
| GRU-minus-LR-B | web | +0.039 [+0.001, +0.076] | 0.0450 | -0.171 [-0.208, -0.128] | 0.0050 | yes |
| GRU-minus-LR-B | email | +0.161 [+0.116, +0.209] | 0.0050 | -0.121 [-0.163, -0.063] | 0.0050 | yes |
| GRU-minus-LR-B | coding | +0.098 [+0.050, +0.147] | 0.0060 | -0.157 [-0.198, -0.109] | 0.0050 | yes |
| LR-C-drop-minus-LR-C | medical | +0.010 [+0.005, +0.016] | 0.0050 | +0.051 [+0.042, +0.061] | 0.0050 | no |
| LR-C-drop-minus-LR-C | banking | +0.002 [-0.005, +0.008] | 0.6397 | +0.025 [+0.015, +0.035] | 0.0050 | no |
| LR-C-drop-minus-LR-C | web | -0.002 [-0.006, +0.002] | 0.6397 | +0.032 [+0.022, +0.043] | 0.0050 | no |
| LR-C-drop-minus-LR-C | email | +0.004 [+0.001, +0.008] | 0.0360 | +0.038 [+0.029, +0.047] | 0.0050 | no |
| LR-C-drop-minus-LR-C | coding | +0.003 [-0.001, +0.007] | 0.5667 | +0.033 [+0.022, +0.042] | 0.0050 | no |

Folds meeting the plan 3.4 reversal rule [s3_dropout.reversal_rule_3_4.<contrast>.folds_reversal_holm], and folds where the R3 difference is detected below / above zero after Holm:

- LR-C-minus-LR-B: reversal ['web', 'email', 'coding']; R3 below ['medical', 'banking', 'web', 'email', 'coding']; R3 above none; R1 domain-matched above ['web', 'email', 'coding']; R1 pooled +0.049 [+0.034, +0.064].
- GRU-minus-LR-C: reversal ['medical', 'banking', 'email', 'coding']; R3 below ['medical', 'banking', 'web', 'email', 'coding']; R3 above none; R1 domain-matched above ['medical', 'banking', 'email', 'coding']; R1 pooled +0.057 [+0.042, +0.072].
- GRU-minus-LR-B: reversal ['medical', 'banking', 'web', 'email', 'coding']; R3 below ['medical', 'banking', 'web', 'email', 'coding']; R3 above none; R1 domain-matched above ['medical', 'banking', 'web', 'email', 'coding']; R1 pooled +0.105 [+0.086, +0.125].
- LR-C-drop-minus-LR-B: reversal ['web', 'email', 'coding']; R3 below ['medical', 'web', 'email', 'coding']; R3 above none; R1 domain-matched above ['web', 'email', 'coding']; R1 pooled +0.051 [+0.036, +0.067].
- GRU-drop0.2-minus-LR-C: reversal none; R3 below none; R3 above ['banking']; R1 domain-matched above ['medical', 'banking', 'email', 'coding']; R1 pooled +0.064 [+0.050, +0.079].
- GRU-drop0.2-minus-LR-B: reversal ['medical', 'web', 'email', 'coding']; R3 below ['medical', 'web', 'email', 'coding']; R3 above none; R1 domain-matched above ['medical', 'banking', 'web', 'email', 'coding']; R1 pooled +0.113 [+0.095, +0.132].
- GRU-drop0.1-minus-LR-C: reversal ['medical']; R3 below ['medical']; R3 above none; R1 domain-matched above ['medical', 'banking', 'email', 'coding']; R1 pooled +0.065 [+0.051, +0.079].
- GRU-drop0.1-minus-LR-B: reversal ['medical', 'banking', 'web', 'email', 'coding']; R3 below ['medical', 'banking', 'web', 'email', 'coding']; R3 above none; R1 domain-matched above ['medical', 'banking', 'web', 'email', 'coding']; R1 pooled +0.113 [+0.096, +0.132].
- GRU-drop0.2-minus-LR-C-drop: reversal ['medical', 'coding']; R3 below ['medical', 'web', 'coding']; R3 above none; R1 domain-matched above ['medical', 'banking', 'email', 'coding']; R1 pooled +0.061 [+0.047, +0.076].

Threshold change, R3 minus domain-matched R1 at theta_r80 (independent bootstraps, Holm over five folds) [s3_dropout.threshold_change_R3_minus_R1dm.<det>.<recall@t_r80|FPR@t_r80>.<fold>]:

| Detector | metric | medical | banking | web | email | coding |
|---|---|---|---|---|---|---|
| LR-B | recall@t_r80 | -0.135 [-0.208, -0.064] H | +0.027 [-0.054, +0.111] | -0.037 [-0.090, +0.019] | +0.039 [-0.021, +0.101] | +0.007 [-0.053, +0.068] |
| LR-B | FPR@t_r80 | -0.087 [-0.154, -0.017] | +0.024 [-0.050, +0.094] | -0.058 [-0.124, +0.011] | +0.061 [-0.018, +0.141] | -0.032 [-0.109, +0.041] |
| LR-C | recall@t_r80 | +0.177 [+0.105, +0.248] H | +0.354 [+0.258, +0.439] H | +0.085 [+0.045, +0.130] H | +0.173 [+0.114, +0.236] H | +0.135 [+0.083, +0.194] H |
| LR-C | FPR@t_r80 | +0.181 [+0.104, +0.260] H | +0.539 [+0.479, +0.595] H | +0.543 [+0.478, +0.603] H | +0.595 [+0.534, +0.651] H | +0.447 [+0.374, +0.512] H |
| LR-C-drop | recall@t_r80 | -0.194 [-0.271, -0.121] H | +0.163 [+0.065, +0.258] H | -0.097 [-0.149, -0.043] H | -0.013 [-0.076, +0.056] | -0.056 [-0.111, +0.004] |
| LR-C-drop | FPR@t_r80 | -0.182 [-0.256, -0.110] H | +0.112 [+0.057, +0.164] H | -0.014 [-0.074, +0.045] | +0.256 [+0.196, +0.313] H | +0.030 [-0.041, +0.094] |
| GRU | recall@t_r80 | +0.353 [+0.230, +0.459] H | +0.123 [-0.578, +0.549] | +0.183 [+0.036, +0.324] H | +0.199 [+0.116, +0.286] H | -0.030 [-0.534, +0.292] |
| GRU | FPR@t_r80 | +0.672 [+0.581, +0.741] H | +0.418 [-0.042, +0.703] | +0.552 [+0.379, +0.726] H | +0.734 [+0.651, +0.818] H | +0.427 [+0.001, +0.663] |
| GRU-drop0.2 | recall@t_r80 | -0.442 [-0.609, -0.258] H | -0.206 [-0.458, -0.022] H | -0.557 [-0.784, -0.237] H | -0.554 [-0.632, -0.467] H | -0.379 [-0.521, -0.265] H |
| GRU-drop0.2 | FPR@t_r80 | -0.021 [-0.102, +0.063] | +0.017 [-0.025, +0.057] | -0.054 [-0.120, +0.022] | +0.003 [-0.033, +0.034] | +0.019 [-0.066, +0.098] |
| GRU-drop0.1 | recall@t_r80 | -0.486 [-0.580, -0.386] H | -0.158 [-0.438, +0.223] | -0.431 [-0.706, -0.161] H | -0.607 [-0.692, -0.513] H | -0.544 [-0.661, -0.379] H |
| GRU-drop0.1 | FPR@t_r80 | -0.029 [-0.070, +0.008] | +0.048 [-0.028, +0.173] | -0.019 [-0.091, +0.068] | +0.002 [-0.034, +0.037] | -0.024 [-0.090, +0.054] |

(H = detected after Holm.)

**Plain reading (plan 3.4 rule).** Without remedies, the C-versus-B reversal meets the rule in web, email, coding. It cannot be claimed in medical or banking, where no in-distribution C > B advantage is detected. The GRU-versus-C reversal meets the rule in medical, banking, email, coding; in web the in-distribution GRU advantage is not detected. (1) Tool-block dropout: LR-C-drop - LR-B still meets the rule in web, email, coding. Its R3 disadvantage is small, -0.015 to -0.001 across folds, against -0.063 to -0.026 for LR-C [s3_dropout.magnitudes]. It is detected because the two scores rank the held-out trajectories almost identically (Spearman rho >= 0.997 in every fold, against 0.875-0.922 for LR-C vs LR-B) [s3_dropout.magnitudes.spearman_R3_poolE], so the paired interval is very narrow. So under the letter of the rule the reversal survives tool-block dropout in these folds. In absolute size it is 5% to 28% of the LR-C gap in the same fold [s3_dropout.magnitudes.ratio_abs_R3_LR-C-drop-minus-LR-B_to_LR-C-minus-LR-B], a near tie in practice. (2) Token dropout, p = 0.2 (primary): GRU-drop0.2 - LR-C meets the rule in none and is detected above LR-C after Holm in banking; its R3 difference ranges -0.003 to +0.026. So the GRU-versus-C reversal does not survive. GRU-drop0.2 is detected below LR-B under R3 in medical, web, email, coding (R3 differences over the five folds -0.061 to -0.006), although it is +0.060 to +0.165 above LR-B in-distribution. That GRU-versus-B reversal meets the rule in medical, web, email, coding. (3) p = 0.1 (sensitivity): versus LR-C the rule is met in medical; versus LR-B in medical, banking, web, email, coding. Dropout changes the direction of the threshold failure rather than removing it [s3_dropout.threshold_change_counts]. Relative to the domain-matched R1 value (Holm), LR-C's recall rises in 5/5 folds and its FPR in 5/5. For GRU-drop0.2 recall falls in 5/5 and FPR changes in 0/5. For GRU-drop0.1 recall falls in 4/5 and FPR changes in 0/5. For LR-C-drop recall rises in 1 and falls in 2 folds, and FPR rises in 2 and falls in 1. LR-B changes in 1 (recall) and 0 (FPR) folds.

## M1, action 4: GRU tool-pathway offset over all folds and seeds; m6: <UNK> norm

**Question.** tool pathway = GRU head weights over the 64 hidden units dotted with the final hidden state (cache field tool_contrib; train_tool_contrib on the 90% fit partition). Training negatives = all label-0 rows of the fit partition (benign, hard_negative, failed_attack); held-out negatives = pool-E negatives of the test fold (benign, hard_negative); held-out positives = attacked test rows. offset = held-out-negative mean minus training-negative mean.

| Fold | detector | train negatives | held-out negatives | held-out positives | offset (held-out - train negatives), seed mean [95% CI] |
|---|---|---|---|---|---|
| R1 | GRU | -4.304 | -3.772 | +3.246 | +0.532 [+0.387, +0.678] |
| R1 | GRU-drop0.2 | -2.522 | -2.278 | +2.213 | +0.243 [+0.141, +0.343] |
| R1 | GRU-drop0.1 | -3.380 | -3.033 | +2.578 | +0.347 [+0.234, +0.473] |
| R3_medical | GRU | -3.510 | +4.039 | +5.162 | +7.549 [+7.439, +7.665] |
| R3_medical | GRU-drop0.2 | -2.945 | -0.525 | +0.033 | +2.420 [+2.359, +2.485] |
| R3_medical | GRU-drop0.1 | -3.106 | -0.397 | +0.238 | +2.709 [+2.640, +2.781] |
| R3_banking | GRU | -3.149 | +1.083 | +1.938 | +4.231 [+4.135, +4.327] |
| R3_banking | GRU-drop0.2 | -2.610 | -0.350 | +0.190 | +2.260 [+2.203, +2.322] |
| R3_banking | GRU-drop0.1 | -2.849 | -0.213 | +0.412 | +2.636 [+2.571, +2.699] |
| R3_web | GRU | -3.413 | +3.782 | +5.151 | +7.195 [+7.083, +7.309] |
| R3_web | GRU-drop0.2 | -2.359 | -0.079 | +0.314 | +2.280 [+2.225, +2.336] |
| R3_web | GRU-drop0.1 | -3.165 | -0.149 | +0.288 | +3.016 [+2.958, +3.075] |
| R3_email | GRU | -2.958 | +3.701 | +4.636 | +6.659 [+6.558, +6.762] |
| R3_email | GRU-drop0.2 | -2.316 | -0.542 | -0.060 | +1.774 [+1.717, +1.830] |
| R3_email | GRU-drop0.1 | -2.493 | -0.302 | +0.129 | +2.191 [+2.135, +2.246] |
| R3_coding | GRU | -3.940 | +1.881 | +2.891 | +5.821 [+5.710, +5.933] |
| R3_coding | GRU-drop0.2 | -2.618 | -0.459 | +0.051 | +2.159 [+2.102, +2.217] |
| R3_coding | GRU-drop0.1 | -3.152 | -0.685 | -0.234 | +2.468 [+2.406, +2.531] |
| R4 | GRU | -3.000 | +0.789 | +3.145 | +3.789 [+3.584, +4.011] |
| R4 | GRU-drop0.2 | -2.560 | -1.773 | -0.735 | +0.787 [+0.673, +0.906] |
| R4 | GRU-drop0.1 | -2.930 | -1.847 | -0.699 | +1.083 [+0.960, +1.204] |

[s4_gru_offset.settings.<setting>.<det>.mean_over_seeds; per seed in .per_seed]. Over the 15 R3 fold x seed cells:

- GRU: training negatives mean -3.394 (range -4.872 to -2.230); held-out negatives +2.897 (-3.595 to +5.229); held-out positives +3.956 (-3.160 to +6.859); offset +6.291 (-0.782 to +8.617); held-out negative mean above training negative mean in 14/15 cells [s4_gru_offset.summary_R3.GRU].
- GRU-drop0.2: training negatives mean -2.570 (range -3.123 to -2.192); held-out negatives -0.391 (-0.817 to +0.674); held-out positives +0.106 (-0.583 to +1.355); offset +2.179 (+1.633 to +2.891); held-out negative mean above training negative mean in 15/15 cells [s4_gru_offset.summary_R3.GRU-drop0.2].
- GRU-drop0.1: training negatives mean -2.953 (range -3.821 to -2.193); held-out negatives -0.349 (-1.258 to +0.739); held-out positives +0.167 (-0.975 to +1.406); offset +2.604 (+1.787 to +3.808); held-out negative mean above training negative mean in 15/15 cells [s4_gru_offset.summary_R3.GRU-drop0.1].

<UNK> embedding norm, R3 folds x seeds [s4_gru_offset.unk.summary_R3.<det>; per cell in s4_gru_offset.unk.<setting>.<det>]:

- GRU: initial norm 5.597-6.145 (reproduced by re-seeding: True); trained norm 0.113-2.332 (mean 1.228); median trained tool-embedding norm 4.284-4.824; median initial tool-embedding norm 5.548-5.644; <UNK> training steps (before dropout) max 0; Adam-with-weight-decay-only simulation reproduces the trained norm (largest difference 0.0e+00).
- GRU-drop0.2: initial norm 5.597-6.145 (reproduced by re-seeding: True); trained norm 4.887-5.630 (mean 5.221); median trained tool-embedding norm 3.911-4.277; median initial tool-embedding norm 5.548-5.644; <UNK> training steps (before dropout) max 0.
- GRU-drop0.1: initial norm 5.597-6.145 (reproduced by re-seeding: True); trained norm 4.837-5.562 (mean 5.176); median trained tool-embedding norm 4.115-4.523; median initial tool-embedding norm 5.548-5.644; <UNK> training steps (before dropout) max 0.

Example (R3 medical, seed 0): initial 6.137, trained 1.878, median trained tool norm 4.612; 2023 optimizer steps to the restored epoch; plain multiplicative decay (1 - lr*wd)^T would leave 6.136; Adam with the coupled L2 term alone gives 1.878 [s4_gru_offset.unk.R3_medical.GRU.0]. For the refitted R1 models the simulated <UNK> vector matches the trained one coordinate-wise to 0.0e+00, 0.0e+00, 0.0e+00 [s4_gru_offset.unk.R1.GRU.<seed>.adam_prediction_max_abs_coord_err_vs_trained_vector]. Within AgentDojo (WUT), <UNK> does occur in training: up to 334 steps in 327 traces per fold, trained norm 2.755-5.572 [s4_gru_offset.wut_unk].

**Plain reading.** The review's one-model observation holds across folds and seeds. In 14 of 15 R3 fold x seed cells, the pre-registered GRU's tool pathway gives held-out negatives a higher contribution than training negatives. The mean offset is +6.29 logits (range -0.78 to +8.62), against +0.53 in R1. So under the tool pathway, held-out benign traces look like attacked training traces (training positives average +3.66). The exception (banking seed 1: offset -0.78, FPR at theta_r80 0.000, recall 0.006) is the seed that raises almost no alerts in that fold. The <UNK> row gets no data gradient in AgentDrift training (0 <UNK> training steps). Its trained value is what Adam's coupled weight decay alone does to the random initialisation: the simulation matches the trained norm in every cell (largest |simulated - cached| norm difference 0.0e+00, i.e. bit-identical in float32); for the refitted R1 models the vectors match coordinate for coordinate (largest difference 0.0e+00). Adam normalises each coordinate's gradient, so the small L2 gradient moves every coordinate towards 0 by about lr per step. After 1,666-3,451 steps the row falls from 5.60-6.15 to 0.11-2.33. The trained tool embeddings have median norms 4.28-4.82 (initial medians 5.55-5.64). Plain multiplicative decay, (1 - lr*wd)^T, would have left the row almost unchanged (at least 5.59). The manuscript's 'changes only through weight decay' is literally true but misleading: the row is an untrained random vector, shrunk by the optimiser, and its norm depends on how many steps early stopping allowed. With token dropout the <UNK> row is trained (p = 0.2: norm 4.89-5.63). The offset falls to +1.63 to +2.89 (p = 0.1: +1.79 to +3.81). It remains positive in every cell, which fits the lower but still present R3 loss of the dropout arms. These means describe the mechanism. They do not show by themselves how much of the AUROC loss the offset causes, because the tool contribution also varies across held-out trajectories.

## M1, action 5a: AgentDojo leave-one-suite-out

**Question.** Does the C-versus-B reversal replicate on real traces when a whole suite (and most of its tools) is held out? cluster bootstrap over suite|user_task (plan 3.5), B = 2000; WUT out-of-fold scores (5-fold CV grouped by suite|user_task, 5 repetitions); one repetition drawn per replicate and shared by all detectors (paired by fold assignment); point = mean over repetitions; pooled over all suites and restricted to each suite (suite-matched).

| Held-out suite | pos / neg (clusters) | tool-occurrence coverage | traces with zero known tools | LR-B | LR-C | LR-C-drop |
|---|---|---|---|---|---|---|
| banking | 671 / 56 (22) | 0.000 | 1.000 | 0.898 [0.814, 0.964] | 0.840 [0.731, 0.929] | 0.913 [0.827, 0.975] |
| slack | 904 / 105 (24) | 0.000 | 1.000 | 0.701 [0.619, 0.799] | 0.744 [0.665, 0.834] | 0.720 [0.639, 0.817] |
| travel | 182 / 103 (24) | 0.084 | 0.537 | 0.613 [0.550, 0.706] | 0.661 [0.588, 0.761] | 0.626 [0.557, 0.720] |
| workspace | 537 / 128 (47) | 0.362 | 0.194 | 0.644 [0.574, 0.725] | 0.643 [0.570, 0.726] | 0.640 [0.570, 0.719] |

[s5_loso.suites.<suite>.coverage; .LOSO_cluster.<det>.AUROC]

| Held-out suite | LR-C - LR-B, LOSO (cluster) | Holm p | same, WUT suite-matched | Holm p | LR-C-drop - LR-B, LOSO | LR-C-drop - LR-C, LOSO |
|---|---|---|---|---|---|---|
| banking | -0.059 [-0.172, +0.034] | 0.5037 | +0.021 [-0.070, +0.141] | 1.0000 | +0.014 [+0.004, +0.025] | +0.073 [-0.017, +0.180] |
| slack | +0.044 [+0.016, +0.076] | 0.0120 | +0.029 [-0.075, +0.113] | 1.0000 | +0.019 [+0.005, +0.042] | -0.024 [-0.043, -0.003] |
| travel | +0.048 [+0.011, +0.084] | 0.0420 | +0.034 [-0.138, +0.174] | 1.0000 | +0.013 [-0.014, +0.038] | -0.035 [-0.085, +0.016] |
| workspace | -0.001 [-0.019, +0.019] | 0.9195 | +0.054 [-0.038, +0.153] | 1.0000 | -0.004 [-0.014, +0.005] | -0.003 [-0.022, +0.014] |

In distribution, pooled over all suites (WUT; the same resample as the P0 rows of s6_rule4): LR-C - LR-B +0.056 [-0.003, +0.111] (cluster), +0.056 [+0.014, +0.092] (stratified); LR-C-drop - LR-B +0.041 [-0.018, +0.094]; LR-C-drop - LR-C -0.015 [-0.039, +0.006] [s5_loso.WUT_pooled.contrasts]. LOSO stratified-bootstrap sensitivity for LR-C - LR-B: banking -0.059 [-0.104, -0.016], slack +0.044 [+0.030, +0.056], travel +0.048 [+0.022, +0.075], workspace -0.001 [-0.012, +0.011] [s5_loso.suites.<suite>.contrasts.LR-C-minus-LR-B.LOSO_stratified].

**Plain reading.** No held-out suite meets the reversal rule for LR-C versus LR-B (suites: none). With the cluster bootstrap, the in-distribution C > B advantage is not detected in any single suite, so the rule's first condition fails everywhere. Pooled over suites it is also not detected (+0.056 [-0.003, +0.111]); it is detected only with the trajectory-level bootstrap (+0.056 [+0.014, +0.092]). On the held-out suites, C - B is detected above zero after Holm in slack, travel and below zero in no suite. Banking has zero coverage and C - B = -0.059 [-0.172, +0.034]: not detected with clusters, but detected with the trajectory-level bootstrap (-0.059 [-0.104, -0.016]). Slack also has zero coverage and shows the opposite sign (+0.044 [+0.016, +0.076]). At zero coverage the ranking depends only on LR-C's behavioural weights, and whether they rank a new suite better or worse than LR-B's is not consistent. The zero-block offset is negative in AgentDojo (action 1), so at the training threshold LR-C raises fewer alerts than LR-B on those suites (banking: LR-C FPR 0.018 / recall 0.283, LR-B FPR 0.054 / recall 0.763; slack: LR-C FPR 0.276 / recall 0.781, LR-B FPR 0.514 / recall 0.807) [s5_loso.suites.<suite>.LOSO_cluster.<det>]. That is the opposite of the AgentDrift FPR rise. Tool-block dropout gives LR-C-drop - LR-B detected above zero in banking, slack. Overall the AgentDrift reversal does not replicate as a consistent pattern on real multi-suite traces. With 22-47 clusters per suite the intervals are wide, and a small reversal in single suites (for example banking) cannot be excluded.

## M1, action 5b: pre-registered Rule 4 in R4 (corrected data)

**Question.** PRE_ABLATION_DESIGN.md Rule 4: if C > B in R1 (paired, detected) but C degrades more than B (degradation difference detected favouring B in >= 3/5 R3 folds or in R4), report 'benchmark-specific tool information improves in-distribution scores while reducing robustness'; the R4 test is primary.

Precondition, R1 LR-C - LR-B: +0.049 [+0.034, +0.064] [s6_rule4.R1_C_minus_B].

| Population | bootstrap | transfer C - B | within C - B | degradation C (within - transfer) | degradation B | DD (within ref.) | DD (R1 ref.) |
|---|---|---|---|---|---|---|---|
| P0 | cluster | -0.008 [-0.014, -0.004] | +0.056 [-0.003, +0.111] | +0.324 [+0.271, +0.384] | +0.259 [+0.202, +0.323] | +0.064 [+0.005, +0.120] | +0.057 [+0.042, +0.074] |
| P0 | stratified | -0.008 [-0.011, -0.006] | +0.056 [+0.014, +0.092] | +0.324 [+0.279, +0.369] | +0.259 [+0.215, +0.305] | +0.064 [+0.022, +0.101] | +0.057 [+0.042, +0.073] |
| C1 | cluster | -0.008 [-0.014, -0.003] | +0.067 [+0.007, +0.119] | +0.331 [+0.273, +0.392] | +0.256 [+0.197, +0.321] | +0.075 [+0.016, +0.130] | +0.057 [+0.041, +0.074] |
| C1 | stratified | -0.008 [-0.011, -0.006] | +0.067 [+0.029, +0.102] | +0.331 [+0.283, +0.378] | +0.256 [+0.204, +0.302] | +0.075 [+0.036, +0.111] | +0.057 [+0.042, +0.073] |

[s6_rule4.R4.<P0|C1>.<cluster|stratified>]. AUROC levels (P0, cluster): transfer_LR-C 0.473 [0.427, 0.510], transfer_LR-B 0.482 [0.435, 0.517], within_LR-C 0.797 [0.747, 0.845], within_LR-B 0.741 [0.691, 0.790].

Supplementary R3 clause, DD = (R1_C - R3_C) - (R1_B - R3_B): pooled R1 medical +0.111 [+0.093, +0.130], banking +0.074 [+0.056, +0.093], web +0.083 [+0.064, +0.102], email +0.102 [+0.084, +0.120], coding +0.086 [+0.068, +0.105] (5/5 detected favouring B after Holm); domain-matched R1 medical +0.056 [+0.023, +0.086], banking +0.047 [+0.012, +0.083], web +0.075 [+0.048, +0.103], email +0.143 [+0.099, +0.186], coding +0.087 [+0.056, +0.120] (5/5) [s6_rule4.R3_supplementary].

**Plain reading.** The precondition holds: C > B in R1 is detected (+0.049 [+0.034, +0.064]). On the corrected data both detectors are near chance in transfer (LR-C 0.473 [0.427, 0.510], LR-B 0.482 [0.435, 0.517]). C - B in transfer is small but detected (-0.008 [-0.014, -0.004], cluster). The degradation difference with the within-AgentDojo reference is +0.064 [+0.005, +0.120] with the cluster bootstrap and +0.064 [+0.022, +0.101] with the pre-registered stratified bootstrap. With the R1 reference it is +0.057 [+0.042, +0.074]. All are positive and detected, so LR-C loses more than LR-B, and Rule 4's R4 clause is met (Outcome 4). The test is weak, as the review notes. Both transfer AUROCs are close to 0.5, so the within-AgentDojo degradation difference mostly restates the within-AgentDojo C - B gap, which the cluster bootstrap does not detect on its own (+0.056 [-0.003, +0.111]); the cluster interval of the degradation difference only just excludes 0. The R3 clause is also met in all five folds with either R1 reference, but the pre-registration calls it partly structural. Supported reading: the detector that uses more benchmark-specific information loses more when moved across benchmarks. It is not evidence that the behavioural representation transfers, since LR-B is itself at chance in R4.

## Q1: GRU interval estimand, per-seed R3 values and a two-level bootstrap

**Question.** Is the GRU interval that of a seed-averaged detector or of a single-seed deployment? each replicate draws 3 seeds with replacement, then resamples trajectories (stratified); statistic = mean over drawn seeds of AUROC_GRU(seed) - AUROC_LR-C; pre-registered evalkit convention: one seed per replicate (the interval is a mixture of single-seed sampling distributions); statistic = mean over the 3 seeds within each trajectory replicate (conditional on the 3 trained seeds). with only 3 seeds the seed level of the two-level bootstrap is coarse (10 distinct seed multisets).

| Fold | GRU AUROC seeds 0/1/2 | GRU - LR-C per seed | two-level | mixture (pre-registered) | seed-averaged |
|---|---|---|---|---|---|
| R1 | 0.936 / 0.935 / 0.935 | +0.058 / +0.057 / +0.056 | +0.057 [+0.043, +0.072] | +0.057 [+0.042, +0.072] | +0.057 [+0.043, +0.071] |
| R3_medical | 0.640 / 0.649 / 0.626 | -0.105 / -0.096 / -0.119 | -0.107 [-0.124, -0.090] | -0.107 [-0.130, -0.087] | -0.107 [-0.119, -0.094] |
| R3_banking | 0.733 / 0.718 / 0.758 | -0.096 / -0.111 / -0.070 | -0.092 [-0.120, -0.069] | -0.092 [-0.135, -0.061] | -0.092 [-0.104, -0.079] |
| R3_web | 0.752 / 0.718 / 0.707 | -0.110 / -0.144 / -0.155 | -0.137 [-0.162, -0.110] | -0.137 [-0.167, -0.100] | -0.137 [-0.151, -0.122] |
| R3_email | 0.644 / 0.715 / 0.654 | -0.096 / -0.025 / -0.085 | -0.069 [-0.101, -0.027] | -0.069 [-0.106, -0.009] | -0.069 [-0.082, -0.055] |
| R3_coding | 0.684 / 0.672 / 0.729 | -0.130 / -0.142 / -0.086 | -0.119 [-0.148, -0.087] | -0.119 [-0.154, -0.078] | -0.119 [-0.133, -0.105] |

[s7_q1.folds.<fold>.GRU]. Per-seed GRU - LR-C intervals (trajectory bootstrap) are in .minus_LR-C_per_seed. Dropout arms (sensitivity), two-level GRU-drop - LR-C: GRU-drop0.2: medical +0.001 [-0.016, +0.019], banking +0.019 [+0.005, +0.033], web +0.008 [-0.006, +0.020], email +0.026 [+0.001, +0.051], coding -0.003 [-0.025, +0.016]; GRU-drop0.1: medical -0.023 [-0.038, -0.009], banking +0.005 [-0.013, +0.022], web -0.007 [-0.020, +0.005], email +0.010 [-0.016, +0.037], coding -0.016 [-0.032, -0.000].

Plan 3.4 rule re-applied with the two-level and the seed-averaged bootstrap [s7_q1.reversal_two_level / reversal_seed_averaged.<contrast>.folds_reversal_holm]: GRU-minus-LR-C: two-level ['medical', 'banking', 'email', 'coding'], seed-averaged ['medical', 'banking', 'email', 'coding']; GRU-minus-LR-B: two-level ['medical', 'banking', 'web', 'email', 'coding'], seed-averaged ['medical', 'banking', 'web', 'email', 'coding']; GRU-drop0.2-minus-LR-C: two-level none, seed-averaged none; GRU-drop0.2-minus-LR-B: two-level ['medical', 'web', 'email', 'coding'], seed-averaged ['medical', 'web', 'email', 'coding']; GRU-drop0.1-minus-LR-C: two-level ['medical'], seed-averaged ['medical']; GRU-drop0.1-minus-LR-B: two-level ['medical', 'banking', 'web', 'email', 'coding'], seed-averaged ['medical', 'banking', 'web', 'email', 'coding'].

**Plain reading.** The per-seed email differences are -0.096, -0.025, -0.085, as the review reported. The sign is the same for every seed, but the size varies about fourfold. In every R3 fold each single seed's GRU - LR-C interval lies below zero, and the fold-level difference is detected below zero under all three estimands. So the GRU-versus-C conclusion in R3 does not depend on the estimand. What does depend on the estimand is the width of the intervals. The pre-registered interval samples one seed per replicate, so it describes a randomly drawn single-seed deployment, while its point estimate is the seed mean. The seed-averaged interval is the narrowest and describes the mean of these three trained models only. The two-level interval is the closest to 'a seed-averaged detector with training randomness included'. With three seeds its seed level is coarse, so it should be read as approximate. For the dropout arms, where differences to LR-C are small, the reversal decisions under the three estimands are listed above. The manuscript should state which estimand it reports and give the per-seed values.

## Deviations and limitations

- O1. Bootstrap size: B = 2000 in every analysis here (the v0.3 AgentDrift tables used B = 1000); fixed rng seeds are listed per analysis in the code.
- O2. Multi-seed GRU intervals in s3 follow the pre-registered evalkit convention (one training seed drawn per replicate; point = seed mean). s7 adds the two-level (seeds then trajectories) and seed-averaged intervals and re-applies the plan 3.4 rule under both.
- O3. Holm correction needs p-values; they are two-sided bootstrap p-values (add-one), not analytic. 'Detected' without Holm = 95% percentile interval excludes 0.
- O4. Graded coverage (plan Sec. 4 'Mechanism'): level L masks round((1-L)*103) randomly drawn tool NAMES of the R1 benign-training vocabulary (26, 52, 77 names; levels 1 and 0 have one deterministic mask), as in the plan's 'random 25/50/75/100% of tools'. The achieved tool-occurrence coverage varies by draw and is reported, not forced to the target. The interval is for the draw- (and seed-) averaged metric conditional on the drawn masks; the per-draw range is given separately.
- O5. The R1 GRU and GRU-drop0.2 models were not saved by the cache; they were refitted with gru_variants.train_gru_drop exactly as build_gru_cache.run_unit calls it (same seeds, validation split and data) and checked against the cached scores (checks_r1_gru_refit). GRU-drop0.1 is not part of the graded-coverage request and was not refitted.
- O6. Within-AgentDojo (WUT) references: one CV repetition drawn per replicate and shared by all detectors (paired by fold assignment); evalkit draws the repetition independently per method.
- O7. AgentDojo intervals (LOSO, WUT, R4 Rule 4) use the plan 3.5 cluster bootstrap over suite|user_task as primary. For Rule 4 the pre-registered design (PRE_ABLATION_DESIGN.md) used the stratified evalkit bootstrap; it is reported alongside. The LOSO stratified intervals are a sensitivity analysis.
- O8. GRU tool-pathway 'training negatives' are all label-0 rows of the 90% fit partition (benign, hard_negative, failed_attack), the negatives the network was trained on; the mean over benign + hard_negative rows only is also given (train_negpoolEcats_mean).
- O9. Rule 4's R3 clause (>= 3/5 folds) is reported as a supplementary analysis with pooled R1 (as pre-registered) and domain-matched R1 (plan 3.2) references; its R4 clause is primary.
- O10. LOSO has no GRU (none in the cache), and the m6 request for a within-AgentDojo GRU without <UNK>-bearing traces is not run here.
- O11. R3 fold pools contain every pooled trajectory of the held-out domain, including the R1 test trajectories of that domain, so R3 and domain-matched R1 estimates share rows. The R3-minus-R1 differences (s3 threshold_change and auroc_change, and the s6 R3 clause; the R1-versus-R4 comparison is unaffected) resample the two populations independently, which ignores the positive covariance and makes those intervals conservative.
- O12. Every population is resampled once (pool_W, fixed seed per population) and the same resample is used in every section, so an estimand that appears in two sections (for example R1 LR-C - LR-B, or WUT LR-C - LR-B in s5 and s6) has the identical interval. s7 reuses the s3 mixture results.
