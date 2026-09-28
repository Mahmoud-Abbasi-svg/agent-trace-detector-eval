# Features, observation length and the keyword flag (review M5 and m8)

Script `revision/features_obslen.py` (SHA-256 `0b2e21ca7485b695...`); results `revision/features_obslen_results.json` (generated 2026-09-27T02:34:52+02:00); annotation `revision/tool_state_annotation.json`. Every number below is read from the results JSON; the key path is given in brackets (`/` separates keys). Values are rounded to 3 decimals from the 4-decimal JSON values.

Conventions. Point estimate [95% bootstrap interval], B = 2000. AgentDrift (R1, R3): stratified trajectory bootstrap. AgentDojo (R4 transfer, WUT, P0 univariate): cluster bootstrap over the 117 suite|user_task groups (primary); stratified results are in the JSON as a sensitivity. 'Detected' means the 95% interval excludes zero (or 0.5 for a single AUROC). Holm = Holm-adjusted bootstrap p-value across the five R3 folds. Thresholds (t_r80) always come from training data. Pool E = benign + hard negatives + attacked (AgentDrift), P0 (AgentDojo). Deviations F1-F9 are at the end.

## M5.1 Per-feature attributions: coefficients, univariate AUROC, range overlap

**Question.** Which behavioural features carry LR-B and the behavioural block of LR-C, is their association with the label the same in AgentDrift and AgentDojo, and do the AgentDojo values fall inside the AgentDrift range?

Standardized coefficients (features z-scored on the training set). R1: cached fit with 500 stratified bootstrap refits [`M5_1_coefficients/R1_LR-B_bootstrap/<feature>`, `M5_1_coefficients/R1_LR-C_bootstrap/<feature>`]. WUT: mean (minimum, maximum) over the 25 fold models [`M5_1_coefficients/WUT_<det>_25fold/<feature>`].

| Feature | R1 LR-B | R1 LR-C (behavioural block) | WUT LR-B, mean (min, max) | WUT LR-C, mean (min, max) |
|---|---|---|---|---|
| len | -1.383 [-1.749, -1.026] | -1.196 [-1.554, -0.868] | -0.518 (-1.228, +1.004) | +0.067 (-0.701, +1.802) |
| uniq | 0.863 [0.490, 1.251] | 1.587 [1.108, 2.104] | +0.777 (-0.086, +1.096) | +2.093 (+0.849, +2.907) |
| diversity | -0.865 [-1.073, -0.671] | -1.157 [-1.408, -0.920] | -0.715 (-0.956, -0.510) | -1.073 (-1.728, -0.373) |
| repeats | 0.106 [0.035, 0.172] | 0.096 [0.008, 0.191] | -0.229 (-1.341, +0.262) | +0.238 (-0.967, +0.979) |
| write_n | 0.901 [0.686, 1.128] | 0.100 [-0.023, 0.242] | +0.604 (+0.393, +0.886) | +0.997 (+0.424, +1.850) |
| write_frac | -0.651 [-0.858, -0.459] | -0.258 [-0.519, -0.036] | +0.279 (+0.093, +0.445) | +0.310 (-0.501, +0.854) |
| max_args | -0.129 [-0.202, -0.045] | -0.006 [-0.107, 0.104] | +0.243 (-0.067, +0.581) | -0.772 (-1.420, -0.222) |
| mean_args | 0.385 [0.299, 0.473] | 0.291 [0.174, 0.413] | -0.010 (-0.310, +0.293) | -0.138 (-0.652, +0.552) |
| max_obs | 0.510 [0.443, 0.573] | 0.357 [0.288, 0.446] | +0.969 (+0.226, +1.583) | +1.949 (+1.038, +3.350) |
| mean_obs | -1.861 [-1.965, -1.770] | -2.008 [-2.183, -1.919] | -0.683 (-1.170, -0.009) | -0.705 (-2.650, +2.429) |

Univariate AUROC of each feature on its own (score = feature value; above 0.5 means larger in attacked or compromised runs) [`M5_1_univariate/populations/<population>/metrics/<feature>/AUROC`].

| Feature | R1 | R3 medical | R3 banking | R3 web | R3 email | R3 coding | AgentDojo P0 (cluster) |
|---|---|---|---|---|---|---|---|
| len | 0.597 [0.570, 0.625] | 0.572 [0.545, 0.596] | 0.568 [0.540, 0.595] | 0.606 [0.581, 0.629] | 0.567 [0.542, 0.593] | 0.582 [0.556, 0.606] | 0.675 [0.628, 0.724] |
| uniq | 0.527 [0.498, 0.554] | 0.543 [0.516, 0.568] | 0.517 [0.489, 0.546] | 0.529 [0.504, 0.555] | 0.491 [0.467, 0.518] | 0.495 [0.469, 0.520] | 0.669 [0.616, 0.721] |
| diversity | 0.345 [0.322, 0.368] | 0.422 [0.399, 0.447] | 0.390 [0.366, 0.414] | 0.324 [0.305, 0.343] | 0.315 [0.295, 0.336] | 0.311 [0.291, 0.332] | 0.409 [0.367, 0.450] |
| repeats | 0.586 [0.569, 0.602] | 0.527 [0.512, 0.543] | 0.577 [0.558, 0.596] | 0.569 [0.557, 0.582] | 0.625 [0.609, 0.640] | 0.623 [0.606, 0.639] | 0.583 [0.545, 0.625] |
| write_n | 0.553 [0.527, 0.578] | 0.637 [0.613, 0.660] | 0.627 [0.602, 0.654] | 0.562 [0.544, 0.580] | 0.533 [0.508, 0.557] | 0.607 [0.584, 0.629] | 0.742 [0.698, 0.787] |
| write_frac | 0.529 [0.504, 0.556] | 0.617 [0.592, 0.643] | 0.601 [0.574, 0.630] | 0.557 [0.539, 0.576] | 0.502 [0.477, 0.527] | 0.589 [0.565, 0.612] | 0.615 [0.566, 0.663] |
| max_args | 0.456 [0.429, 0.481] | 0.432 [0.408, 0.455] | 0.458 [0.433, 0.485] | 0.559 [0.538, 0.582] | 0.509 [0.487, 0.532] | 0.484 [0.461, 0.507] | 0.604 [0.549, 0.657] |
| mean_args | 0.465 [0.438, 0.493] | 0.473 [0.448, 0.499] | 0.408 [0.380, 0.437] | 0.555 [0.530, 0.580] | 0.546 [0.521, 0.572] | 0.473 [0.448, 0.499] | 0.583 [0.535, 0.636] |
| max_obs | 0.598 [0.570, 0.625] | 0.608 [0.582, 0.634] | 0.420 [0.390, 0.451] | 0.550 [0.524, 0.576] | 0.661 [0.635, 0.685] | 0.630 [0.605, 0.656] | 0.657 [0.615, 0.706] |
| mean_obs | 0.257 [0.234, 0.280] | 0.254 [0.233, 0.276] | 0.187 [0.167, 0.208] | 0.188 [0.169, 0.206] | 0.313 [0.291, 0.337] | 0.245 [0.224, 0.267] | 0.558 [0.518, 0.606] |

Detected direction reversals (interval above 0.5 in one benchmark, below 0.5 in the other) [`M5_1_univariate/reversals_vs_AgentDojo_P0/<population>/reversal_detected`]: R1 vs AgentDojo P0: max_args, mean_args, mean_obs; R3_medical vs AgentDojo P0: max_args, mean_args, mean_obs; R3_banking vs AgentDojo P0: max_args, mean_args, max_obs, mean_obs; R3_web vs AgentDojo P0: mean_obs; R3_email vs AgentDojo P0: mean_obs; R3_coding vs AgentDojo P0: mean_args, mean_obs; AgentDrift_pooled vs AgentDojo P0: max_args, mean_args, mean_obs. Within AgentDrift, R1 vs R3 [`M5_1_univariate/reversals_R1_vs_R3`]: medical: none; banking: max_obs; web: max_args, mean_args; email: mean_args; coding: none. For comparison with the review, mean_obs AUROC on the v0.3 AgentDojo extraction is 0.595 [`M5_1_univariate/v03_archive_P0_point_AUROC/mean_obs`].

Range overlap with the R4 training data (all 10,656 AgentDrift trajectories) [`M5_1_univariate/range_overlap/<feature>`]: max_obs AgentDrift min/median/max [36.0, 162.0, 706.0], AgentDojo P0 [2.0, 711.0, 31363.0], share of P0 traces above the AgentDrift maximum 0.541; mean_obs AgentDrift [27.8333, 87.5, 295.0], P0 [2.0, 315.55, 19670.3333], share above 0.520. Under the R4 scaler, the share of P0 traces with |z| > 3 is 0.908 for max_obs and 0.692 for mean_obs (largest |z| 448 and 649); for the other eight features it is at most 0.248.

Median observation length per step in AgentDrift training data [`M5_4_truncation/agentdrift_train_step_obs_length/<group>/p50`]: benign steps of benign trajectories 98, benign steps of attacked trajectories 59, hijacked steps 50, injection points 168, failed injections 283, benign steps of hard negatives 73 characters.

**Reading.** mean_obs has the largest standardized weight in R1 LR-B (-1.861 [-1.965, -1.770]) and in LR-C's behavioural block (-2.008 [-2.183, -1.919]). On its own its AUROC is below 0.5 in every AgentDrift population, that is, attacked trajectories have shorter mean observations (AUROC 0.257 [0.234, 0.280] in R1 and 0.187 to 0.313 in the five held-out domains). The shortening is not confined to the attack steps: the benign-labelled steps of attacked trajectories are also shorter than the steps of benign trajectories (medians above), which points to a property of the generator's attacked category rather than to the injection itself. In AgentDojo P0 the direction flips (mean_obs 0.558 [0.518, 0.606]); max_args and mean_args also reverse against R1. AgentDojo observation lengths also lie far outside the AgentDrift range, so a transferred linear model extrapolates on exactly these two features. Within AgentDojo the fitted models still give mean_obs a negative weight (WUT LR-B mean -0.683) but a larger positive weight to max_obs (+0.969, positive in all 25 fold models), so the sign of mean_obs in a multivariate model is not by itself the reversal. The numbers do not show which benchmark difference causes the reversal, and the univariate AUROCs say nothing about how the features combine.

## M5.2 R3, R4 and within-AgentDojo without the observation-length features

**Question.** How much of LR-B and LR-C's performance under zero tool coverage (R3), under transfer (R4) and within AgentDojo (WUT) depends on max_obs and mean_obs, and does the R3 advantage of LR-B over LR-C survive without them?

AUROC on pool E [`M5_2_without_observation_length/<setting>/primary/metrics/<detector>/AUROC`]. LR-obsonly = logistic regression on max_obs and mean_obs only (post hoc, F3).

| Setting | LR-B | LR-B-noobs | LR-C | LR-C-noobs | LR-obsonly |
|---|---|---|---|---|---|
| R1 | 0.830 [0.811, 0.849] | 0.699 [0.674, 0.724] | 0.879 [0.862, 0.894] | 0.805 [0.784, 0.824] | 0.790 [0.769, 0.812] |
| R3_medical | 0.808 [0.788, 0.826] | 0.629 [0.605, 0.654] | 0.745 [0.724, 0.767] | 0.588 [0.563, 0.614] | 0.802 [0.781, 0.821] |
| R3_banking | 0.854 [0.836, 0.873] | 0.666 [0.639, 0.693] | 0.828 [0.809, 0.847] | 0.645 [0.619, 0.670] | 0.847 [0.828, 0.865] |
| R3_web | 0.896 [0.882, 0.910] | 0.675 [0.651, 0.698] | 0.862 [0.847, 0.878] | 0.672 [0.648, 0.697] | 0.858 [0.842, 0.873] |
| R3_email | 0.792 [0.772, 0.812] | 0.680 [0.657, 0.705] | 0.739 [0.717, 0.761] | 0.621 [0.596, 0.647] | 0.739 [0.718, 0.761] |
| R3_coding | 0.852 [0.835, 0.868] | 0.740 [0.718, 0.761] | 0.814 [0.796, 0.832] | 0.682 [0.659, 0.704] | 0.808 [0.789, 0.827] |
| R4 | 0.482 [0.434, 0.522] | 0.630 [0.582, 0.680] | 0.473 [0.425, 0.513] | 0.657 [0.611, 0.706] | 0.469 [0.422, 0.509] |
| WUT | 0.741 [0.689, 0.793] | 0.740 [0.688, 0.790] | 0.797 [0.747, 0.842] | 0.795 [0.744, 0.838] | 0.708 [0.657, 0.762] |

Prevalence-weighted AUPRC at pi = 0.10 [`.../metrics/<detector>/wAUPRC@0.1`]:

| Setting | LR-B | LR-B-noobs | LR-C | LR-C-noobs | LR-obsonly |
|---|---|---|---|---|---|
| R1 | 0.403 [0.357, 0.461] | 0.190 [0.170, 0.225] | 0.523 [0.468, 0.588] | 0.383 [0.331, 0.455] | 0.324 [0.286, 0.369] |
| R3_medical | 0.348 [0.306, 0.399] | 0.161 [0.145, 0.183] | 0.249 [0.221, 0.290] | 0.127 [0.116, 0.144] | 0.278 [0.247, 0.323] |
| R3_banking | 0.442 [0.385, 0.516] | 0.183 [0.161, 0.214] | 0.381 [0.336, 0.443] | 0.154 [0.138, 0.180] | 0.393 [0.346, 0.452] |
| R3_web | 0.562 [0.513, 0.620] | 0.254 [0.221, 0.302] | 0.505 [0.459, 0.555] | 0.211 [0.185, 0.244] | 0.439 [0.395, 0.494] |
| R3_email | 0.356 [0.315, 0.407] | 0.209 [0.183, 0.249] | 0.272 [0.242, 0.311] | 0.149 [0.134, 0.173] | 0.294 [0.262, 0.338] |
| R3_coding | 0.459 [0.409, 0.518] | 0.225 [0.201, 0.266] | 0.384 [0.344, 0.435] | 0.181 [0.163, 0.212] | 0.329 [0.293, 0.377] |
| R4 | 0.090 [0.082, 0.100] | 0.133 [0.117, 0.164] | 0.089 [0.081, 0.098] | 0.156 [0.137, 0.208] | 0.088 [0.080, 0.097] |
| WUT | 0.228 [0.183, 0.315] | 0.220 [0.180, 0.303] | 0.298 [0.228, 0.463] | 0.314 [0.241, 0.489] | 0.199 [0.164, 0.274] |

Recall and FPR at t_r80 [`.../metrics/<detector>/recall@t_r80`, `FPR@t_r80`]:

| Setting | LR-B-noobs recall | LR-B-noobs FPR | LR-C-noobs recall | LR-C-noobs FPR |
|---|---|---|---|---|
| R1 | 0.792 [0.765, 0.818] | 0.531 [0.498, 0.565] | 0.759 [0.732, 0.787] | 0.292 [0.261, 0.325] |
| R3_medical | 0.896 [0.877, 0.915] | 0.706 [0.676, 0.735] | 0.996 [0.992, 0.999] | 0.848 [0.824, 0.871] |
| R3_banking | 0.773 [0.741, 0.805] | 0.575 [0.543, 0.605] | 0.990 [0.982, 0.998] | 0.835 [0.811, 0.858] |
| R3_web | 0.471 [0.440, 0.501] | 0.174 [0.150, 0.199] | 1.000 [1.000, 1.000] | 0.979 [0.969, 0.987] |
| R3_email | 0.872 [0.850, 0.893] | 0.633 [0.603, 0.663] | 0.990 [0.983, 0.995] | 0.870 [0.848, 0.890] |
| R3_coding | 0.810 [0.785, 0.833] | 0.534 [0.503, 0.565] | 0.994 [0.989, 0.998] | 0.826 [0.803, 0.850] |
| R4 | 0.591 [0.494, 0.688] | 0.398 [0.306, 0.490] | 0.802 [0.742, 0.858] | 0.523 [0.426, 0.613] |
| WUT | 0.777 [0.700, 0.846] | 0.427 [0.327, 0.524] | 0.722 [0.650, 0.784] | 0.248 [0.162, 0.349] |

Paired AUROC differences per R3 fold with Holm across the five folds [`M5_2_without_observation_length/R3_holm/<contrast>/AUROC/<domain>`]:

| Contrast | medical | banking | web | email | coding | Folds detected after Holm |
|---|---|---|---|---|---|---|
| LR-C-noobs minus LR-B-noobs | -0.042 [-0.054, -0.029], p_Holm 0.005 | -0.021 [-0.032, -0.009], p_Holm 0.005 | -0.003 [-0.024, +0.019], p_Holm 0.811 | -0.059 [-0.071, -0.048], p_Holm 0.005 | -0.059 [-0.070, -0.048], p_Holm 0.005 | 4 of 5 |
| LR-B minus LR-C | +0.063 [+0.052, +0.073], p_Holm 0.005 | +0.026 [+0.014, +0.037], p_Holm 0.005 | +0.034 [+0.022, +0.046], p_Holm 0.005 | +0.053 [+0.043, +0.063], p_Holm 0.005 | +0.038 [+0.027, +0.049], p_Holm 0.005 | 5 of 5 |
| LR-B-noobs minus LR-B | -0.178 [-0.200, -0.155], p_Holm 0.005 | -0.189 [-0.216, -0.162], p_Holm 0.005 | -0.221 [-0.246, -0.197], p_Holm 0.005 | -0.112 [-0.134, -0.091], p_Holm 0.005 | -0.112 [-0.133, -0.090], p_Holm 0.005 | 5 of 5 |
| LR-C-noobs minus LR-C | -0.157 [-0.177, -0.137], p_Holm 0.005 | -0.183 [-0.205, -0.162], p_Holm 0.005 | -0.190 [-0.210, -0.169], p_Holm 0.005 | -0.119 [-0.138, -0.099], p_Holm 0.005 | -0.132 [-0.151, -0.113], p_Holm 0.005 | 5 of 5 |
| LR-obsonly minus LR-B | -0.006 [-0.017, +0.005], p_Holm 0.394 | -0.007 [-0.018, +0.003], p_Holm 0.394 | -0.038 [-0.050, -0.026], p_Holm 0.005 | -0.053 [-0.065, -0.042], p_Holm 0.005 | -0.044 [-0.056, -0.033], p_Holm 0.005 | 3 of 5 |

The same C-noobs minus B-noobs contrast on wAUPRC at pi = 0.10 is detected in 5 of 5 folds after Holm [`M5_2_without_observation_length/R3_holm/LR-C-noobs minus LR-B-noobs/wAUPRC@0.1`]; with world-cluster resampling (sensitivity) the web fold AUROC contrast is -0.003 [-0.025, +0.020] [`M5_2_without_observation_length/R3_web/world_cluster_sensitivity/contrasts/...`].

R4 (transfer) paired AUROC: LR-B-noobs minus LR-B +0.149 [+0.103, +0.204]; LR-C-noobs minus LR-C +0.183 [+0.139, +0.237]; LR-C-noobs minus LR-B-noobs +0.026 [+0.005, +0.047]. WUT: LR-B-noobs minus LR-B -0.001 [-0.012, +0.019]; LR-C-noobs minus LR-C -0.002 [-0.014, +0.020]. Within (WUT) minus transfer (R4) on the same P0 rows, Holm over five detectors [`M5_2_without_observation_length/within_minus_transfer/contrasts/WUT:<d> minus R4:<d>/AUROC`]: LR-B +0.259 [+0.199, +0.326], p_Holm 0.005; LR-C +0.324 [+0.264, +0.385], p_Holm 0.005; LR-B-noobs +0.110 [+0.054, +0.170], p_Holm 0.005; LR-C-noobs +0.138 [+0.073, +0.199], p_Holm 0.005; LR-obsonly +0.239 [+0.185, +0.298], p_Holm 0.005.

**Reading.** In every R3 fold, removing max_obs and mean_obs costs LR-B between 0.112 and 0.221 AUROC (all detected after Holm), and the two observation features alone come within 0.053 AUROC of full LR-B (no detected difference after Holm in medical, banking). So most of LR-B's zero-coverage signal in AgentDrift is the observation-length cue. With the features included, LR-B is above LR-C in all five folds after Holm; without them LR-C-noobs is below LR-B-noobs in 4 of 5 folds after Holm, and the difference is not detected in web (-0.003 [-0.024, +0.019]). On AUROC, an 'all five folds' statement therefore depends on the observation-length features; on the plan's primary metric (prevalence-weighted AUPRC at pi = 0.10) LR-C-noobs is below LR-B-noobs in 5 of 5 folds after Holm (web -0.043 [-0.080, -0.012], p_Holm 0.005), so the count depends on the metric. Without the observation features LR-C-noobs's training threshold does not hold under R3: at t_r80 it flags 0.826 to 0.979 of the held-out negatives. In R4, dropping the two features raises transfer AUROC (LR-B 0.482 [0.434, 0.522] to 0.630 [0.582, 0.680]; LR-C to 0.657 [0.611, 0.706]), consistent with removing a reversed, out-of-range cue (M5.1). It does not close the gap to within-AgentDojo performance: the within-minus-transfer difference is still detected after Holm for every detector. Within AgentDojo removing the two features costs nothing detectable (WUT LR-B-noobs minus LR-B -0.001 [-0.012, +0.019]), although on their own they carry signal there (LR-obsonly 0.708 [0.657, 0.762]); the within-AgentDojo information in observation length is therefore redundant with the other features, not absent. The noobs subsets were chosen after the review had seen these results, so the R4 gain is a post hoc observation, not evidence that the remaining features transfer.

## M5.3 Padding test (attacker-controlled observation length)

**Question.** If the attacker lengthens the injected observation, how do recall at the training threshold t_r80 and AUROC change? We add n filler characters (a neutral repeated sentence) to the injection-point observation of every attacked R1 test trajectory (one step each; 831 trajectories) and rescore with the unchanged R1 models; negatives are not padded. We do not claim that padded injections would still succeed against an agent; the test only asks how the detectors' scores respond to a quantity the attacker writes.

R1, recall at t_r80 and AUROC on pool E [`M5_3_padding/R1_linear/eval/metrics/<det>|pad<n>/<metric>`, `M5_3_padding/R1_GRU/eval/metrics/GRU|pad<n>/<metric>`; GRU = mean over seeds 0-2]:

| Detector | Metric | +0 | +100 | +300 | +600 | +1200 |
|---|---|---|---|---|---|---|
| LR-B | recall@t_r80 | 0.795 [0.768, 0.823] | 0.664 [0.632, 0.698] | 0.448 [0.414, 0.480] | 0.274 [0.244, 0.306] | 0.161 [0.137, 0.188] |
| LR-B | AUROC | 0.830 [0.811, 0.849] | 0.761 [0.739, 0.783] | 0.616 [0.589, 0.643] | 0.435 [0.407, 0.464] | 0.259 [0.235, 0.284] |
| LR-C | recall@t_r80 | 0.771 [0.744, 0.799] | 0.634 [0.602, 0.664] | 0.360 [0.327, 0.392] | 0.170 [0.144, 0.195] | 0.033 [0.021, 0.046] |
| LR-C | AUROC | 0.879 [0.862, 0.894] | 0.809 [0.789, 0.829] | 0.632 [0.605, 0.658] | 0.382 [0.354, 0.408] | 0.132 [0.115, 0.149] |
| LR-B-noobs | recall@t_r80 | 0.792 [0.765, 0.818] | 0.792 [0.765, 0.818] | 0.792 [0.765, 0.818] | 0.792 [0.765, 0.818] | 0.792 [0.765, 0.818] |
| LR-B-noobs | AUROC | 0.699 [0.674, 0.724] | 0.699 [0.674, 0.724] | 0.699 [0.674, 0.724] | 0.699 [0.674, 0.724] | 0.699 [0.674, 0.724] |
| LR-C-noobs | recall@t_r80 | 0.759 [0.732, 0.787] | 0.759 [0.732, 0.787] | 0.759 [0.732, 0.787] | 0.759 [0.732, 0.787] | 0.759 [0.732, 0.787] |
| LR-C-noobs | AUROC | 0.805 [0.784, 0.824] | 0.805 [0.784, 0.824] | 0.805 [0.784, 0.824] | 0.805 [0.784, 0.824] | 0.805 [0.784, 0.824] |
| GRU | recall@t_r80 | 0.682 [0.646, 0.717] | 0.578 [0.543, 0.611] | 0.341 [0.300, 0.380] | 0.100 [0.069, 0.132] | 0.000 [0.000, 0.000] |
| GRU | AUROC | 0.936 [0.924, 0.947] | 0.907 [0.892, 0.921] | 0.814 [0.791, 0.837] | 0.593 [0.556, 0.633] | 0.159 [0.119, 0.199] |

Paired change from +0 to +100 and to +1,200 characters [`M5_3_padding/R1_*/eval/contrasts/<det>|pad<n> minus <det>|pad0/<metric>`]: LR-B: recall -0.131 [-0.155, -0.108] (+100), -0.634 [-0.667, -0.602] (+1,200); LR-C: recall -0.137 [-0.160, -0.113] (+100), -0.739 [-0.768, -0.709] (+1,200); GRU: recall -0.105 [-0.132, -0.082] (+100), -0.682 [-0.717, -0.646] (+1,200). LR-B-noobs and LR-C-noobs do not read observation lengths, so their scores are identical at every level (by construction).

GRU refit check [`M5_3_padding/R1_GRU/checks/seed<k>`]: the three seeds were retrained with gru_detector.train_gru on gru_ladder.stratified_val; t_r80 thresholds and t_r80 predictions equal the cache and pool-E AUROC is identical; 5 of the 5,643 scores (3 seeds x 1,881 rows) differ from the cache by at most 6.0e-08 (float32 rounding). Linear scores rebuilt from the cached coefficients equal the cache to 2.2e-16.

Within AgentDojo (WUT, LR-C where feasible, plus LR-B and LR-B-noobs). The injected step was located in 2285 of 2294 compromised P0 traces (99.6%) by matching the raw run's `injections` text in the observations [`M5_3_padding/WUT/location`]; the other 9 stay unpadded. All located injected steps of a trace are padded. The 25 WUT fold models were refitted with the cache recipe and reproduce the cached out-of-fold scores to 5.6e-16 with identical thresholds. Recall at t_r80 and AUROC on P0, cluster bootstrap [`M5_3_padding/WUT/eval/primary/metrics/<det>|pad<n>/<metric>`]:

| Detector | Metric | +0 | +100 | +300 | +600 | +1200 |
|---|---|---|---|---|---|---|
| LR-C | recall@t_r80 | 0.720 [0.642, 0.788] | 0.723 [0.645, 0.790] | 0.731 [0.656, 0.796] | 0.745 [0.675, 0.807] | 0.759 [0.693, 0.820] |
| LR-C | AUROC | 0.797 [0.747, 0.842] | 0.799 [0.749, 0.844] | 0.803 [0.752, 0.848] | 0.808 [0.757, 0.853] | 0.818 [0.767, 0.863] |
| LR-B | recall@t_r80 | 0.776 [0.705, 0.840] | 0.778 [0.708, 0.842] | 0.785 [0.713, 0.852] | 0.792 [0.719, 0.860] | 0.808 [0.732, 0.876] |
| LR-B | AUROC | 0.741 [0.689, 0.793] | 0.743 [0.690, 0.795] | 0.746 [0.695, 0.798] | 0.752 [0.701, 0.804] | 0.764 [0.713, 0.814] |
| LR-B-noobs | recall@t_r80 | 0.777 [0.700, 0.846] | 0.777 [0.700, 0.846] | 0.777 [0.700, 0.846] | 0.777 [0.700, 0.846] | 0.777 [0.700, 0.846] |
| LR-B-noobs | AUROC | 0.740 [0.688, 0.790] | 0.740 [0.688, 0.790] | 0.740 [0.688, 0.790] | 0.740 [0.688, 0.790] | 0.740 [0.688, 0.790] |

Paired change +0 to +1,200: LR-C recall +0.039 [+0.017, +0.077], AUROC +0.021 [+0.015, +0.035]; LR-B recall +0.031 [+0.006, +0.085]. Restricting to traces with a located injection gives the same picture [`M5_3_padding/WUT/eval/located_subset`].

**Reading.** In R1, every detector that reads observation lengths loses recall steeply as the injected observation grows, and the GRU degrades in the same way as the linear models: 100 extra characters (the median injection-point observation is 168 characters) already cost LR-C 0.137 recall and the GRU 0.105, and at +1,200 the GRU flags 0.000 of attacked trajectories with AUROC 0.159, well below 0.5. The in-distribution scores of these detectors therefore depend on a quantity that an indirect-injection attacker writes (the attacker needs to know only the direction of the cue, not the detector's parameters). Within AgentDojo the direction is reversed: lengthening the injected observation makes LR-C slightly more likely to flag the trace, so padding is not an evasion there; the evasive direction would be shortening, which we did not test. The review reported a larger within-AgentDojo rise in LR-C recall (0.779 to 0.848 to 0.924); that run used an earlier AgentDojo extraction and the review gives three values without mapping them to padding levels, so we do not reproduce its magnitude and have not diagnosed the difference. The test does not show that padded injections still succeed, that real attackers pad, or how a detector retrained on padded data would behave.

## M5.4 Truncation of AgentDojo observations

**Question.** How do observation lengths compare between the benchmarks, how many AgentDojo traces lie beyond the AgentDrift maximum, and which steps did the v0.3 cut at 2,000 characters affect?

- AgentDrift trajectory max_obs ranges from 36 to 706 characters in the pooled train+test data [`M5_4_truncation/agentdrift_max_obs_range_pooled`] (813 including data/val.jsonl); the largest trajectory mean_obs is 295. Median per-trajectory max_obs: AgentDrift 162, AgentDojo P0 711 (benign 542, compromised 722), unsuccessful attacks 1142; per-step 99th percentile AgentDrift 299, AgentDojo P0 4230, maximum 31363 [`M5_4_truncation/*_trajectory`, `M5_4_truncation/*_step_obs_length`].
- Share of AgentDojo P0 traces whose max_obs exceeds the AgentDrift maximum: 54.1% [44.6, 64.1] (benign 37.8% [28.6, 47.2], compromised 56.8% [46.3, 67.4], unsuccessful attacks 75.4%). Share whose mean_obs exceeds the largest AgentDrift mean_obs: 52.0% [42.5, 62.5]; whose mean_obs exceeds the AgentDrift maximum max_obs: 14.6% [8.8, 21.9] [`M5_4_truncation/agentdojo_share_beyond_agentdrift`]. On the v0.3 extraction the max_obs share is 48.8% and the mean_obs-above-largest-mean share 48.2%, which reproduces the review's 48.8% and 48.2% [`cross_references/review_reported_values`].
- v0.3 cut: 448 of 14079 P0 steps (3.18%) sit exactly at 2,000 characters, in 262 traces. In the corrected extraction 500 P0 steps (3.55%) in 300 traces (249 compromised, 51 benign) are longer than 2,000 characters and would be cut, and 1473 steps in 991 unsuccessful-attack traces (not extracted in v0.3). Step by step against the v0.3 file (same step counts in all 2,686 traces): 420 steps differ only by the cut and 1464 differ otherwise (the pairing fix); 436 of the capped v0.3 steps have a corrected observation longer than 2,000 and 12 were capped only because a mispaired observation was long; in 245 traces the cut is the only difference. Re-applying the cut to the corrected P0 changes max_obs and mean_obs of 300 traces [`M5_4_truncation/v03_cut`].
- Effect on results (copied from the linear cache check): within-AgentDojo LR-C AUROC 0.846 [0.820, 0.873] with the cut vs 0.797 [0.766, 0.832] without; LR-B 0.773 [0.744, 0.803] vs 0.741 [0.709, 0.773]; transfer LR-C 0.476 [0.444, 0.509] vs 0.473 [0.442, 0.506] [`cross_references/linear_cache_r4_truncation_check`; stratified bootstrap].

**Reading.** The v0.3 extractor did cut AgentDojo observations at 2,000 characters: 3.2% of P0 steps, in 262 of 2686 traces, and 300 corrected P0 traces (11.2%) contain a longer observation. The manuscript should disclose this and say that the revision stores observations uncut in both benchmarks (harmonized: neither is cut). The cut is not what makes the benchmarks incompatible: even with the cut, 48.8% of AgentDojo traces lay beyond the largest AgentDrift observation, and removing the cut widens the range gap. Removing the cut leaves transfer near 0.5 and lowers within-AgentDojo LR-C AUROC by 0.049. The numbers do not say which cut, if any, a deployed detector should use.

## m8 Keyword-based state-changing features

**Question.** How well does the keyword flag in featurise.py (a tool name containing send, delete, update, create, transfer, add, download, post, write, cancel, apply, convert, share or grant) identify state-changing tools, how many hijacked or compromised calls does it miss, and does LR-B change if the two write features use a state-change annotation instead?

The annotation labels all 103 AgentDrift tool names (50 state-changing, 13 marked ambiguous) and all 70 AgentDojo tool names (22 state-changing, 1 ambiguous) [`m8_keyword_flag/annotation_summary`] (`revision/tool_state_annotation.json`, criterion in its header: persistent write, transmission to another party, or command execution). Keyword flag against the annotation [`m8_keyword_flag/keyword_vs_annotation_<benchmark>`]:

| Benchmark | Tool level: precision (TP/(TP+FP)) | Tool level: recall (TP/(TP+FN)) | Call level: precision | Call level: recall |
|---|---|---|---|---|
| AgentDrift (pooled train+test) | 0.957 (22/23) | 0.440 (22/50) | 0.970 [0.966, 0.973] | 0.455 [0.448, 0.461] |
| AgentDojo (P0 + unsuccessful) | 0.833 (15/18) | 0.682 (15/22) | 0.867 [0.784, 0.944] | 0.831 [0.773, 0.887] |
| AgentDojo P0 | 0.882 (15/17) | 0.682 (15/22) | 0.974 [0.951, 0.991] | 0.830 [0.783, 0.881] |

Read-only tools the keyword flags: convert_currency (353 calls), get_car_rental_address (1 call), get_hotels_address (713 calls), get_restaurants_address (464 calls). State-changing tools it misses, most frequent: AgentDrift fill_form 1759, execute_command 1659, forward_email 1161, submit_form 844, move_email 836, open_issue 593, install_package 559, login_site 547 (28 tools in all); AgentDojo invite_user_to_slack 594, append_to_file 314, remove_user_from_slack 217, reserve_hotel 177, schedule_transaction 133, reschedule_calendar_event 86, reserve_restaurant 40. Excluding the 13 AgentDrift tools marked ambiguous, tool-level precision is 1.000 and recall 0.455.

Hijacked AgentDrift steps (pooled train+test, 10450 steps in 4706 attacked trajectories) [`m8_keyword_flag/agentdrift_hijacked_steps_pooled`]: keyword-flagged 36.5% [35.6, 37.5], so 63.5% [62.5, 64.5] are missed; state-changing by the annotation 75.6% [74.8, 76.4]; of those, the keyword misses 4101 of 7903 (51.9% [50.7, 53.1]). Most-missed state-changing tools: execute_command 1181, forward_email 808, fill_form 770, submit_form 415, move_email 119, submit_claim 102. 43.2% of attacked trajectories have no keyword-flagged hijacked step.

Compromised AgentDojo P0 traces (2294 traces, 12506 calls) [`m8_keyword_flag/agentdojo_compromised_calls`]: 4797 calls are state-changing by the annotation and the keyword misses 815 of them (17.0% [12.2, 21.9]): invite_user_to_slack 416, remove_user_from_slack 212, append_to_file 59, reserve_hotel 56, schedule_transaction 48, reschedule_calendar_event 20, reserve_restaurant 4. AgentDojo has no per-call hijack label; counting only calls after the first located injected observation gives 815 missed of 4754 [`m8_keyword_flag/agentdojo_compromised_calls/post_injection_proxy`].

Sensitivity: LR-B with the annotation-based count and fraction of state-changing calls (features 4 and 5) in place of the keyword ones (LR-B-annot), same recipe; LR-B-minus (no state-change features, cache) for reference. AUROC [`m8_LRB_annotation_features/<setting>/primary/metrics/<det>/AUROC`] and paired difference LR-B-annot minus LR-B [`.../contrasts/LR-B-annot minus LR-B/AUROC`; R3 Holm in `m8_LRB_annotation_features/R3_holm`]:

| Setting | LR-B | LR-B-annot | LR-B-minus | LR-B-annot minus LR-B | Holm (R3) |
|---|---|---|---|---|---|
| R1 | 0.830 [0.811, 0.849] | 0.826 [0.806, 0.846] | 0.827 [0.808, 0.846] | -0.004 [-0.010, +0.001] |  |
| R3_medical | 0.808 [0.788, 0.826] | 0.807 [0.788, 0.826] | 0.803 [0.783, 0.822] | -0.001 [-0.004, +0.003] | p 0.731 |
| R3_banking | 0.854 [0.836, 0.873] | 0.850 [0.831, 0.869] | 0.839 [0.820, 0.858] | -0.004 [-0.009, +0.001] | p 0.375 |
| R3_web | 0.896 [0.882, 0.910] | 0.899 [0.885, 0.912] | 0.899 [0.886, 0.911] | +0.003 [-0.002, +0.008] | p 0.528 |
| R3_email | 0.792 [0.772, 0.812] | 0.768 [0.747, 0.790] | 0.807 [0.787, 0.827] | -0.024 [-0.031, -0.016] | p 0.005 |
| R3_coding | 0.852 [0.835, 0.868] | 0.859 [0.842, 0.874] | 0.849 [0.832, 0.865] | +0.007 [+0.002, +0.012] | p 0.020 |
| R4 | 0.482 [0.434, 0.522] | 0.479 [0.431, 0.520] | 0.478 [0.431, 0.518] | -0.002 [-0.006, +0.001] |  |

R1 standardized coefficients of the two write features: keyword write_n +0.901, write_frac -0.651; annotation write_n +1.379, write_frac -0.864 [`m8_LRB_annotation_features/R1/coef_*`]. Recall and FPR at t_r80 for both variants are in the JSON (`.../metrics/<det>/recall@t_r80`).

**Reading.** The keyword flag is precise but incomplete: almost everything it flags changes state (the exceptions are address and currency lookups matched by 'add' and 'convert'), but it misses about half of the state-changing calls in AgentDrift, including the most frequent hijack actions (execute_command, forward_email, fill_form, submit_form), and about one in six in compromised AgentDojo traces (Slack invitations and removals). Describing the two features as the count and fraction of 'state-changing calls' (manuscript Sec. V-B) therefore overstates what they measure; 'calls whose tool name matches a write-like keyword' is accurate. Replacing them with the annotation changes LR-B's AUROC by at most 0.024 in any setting (detected after Holm in 2 of 5 R3 folds, in opposite directions) and does not rescue transfer, so the keyword errors do not drive LR-B's R1, R3 or R4 results. The annotation is one annotator's reading of tool names, not ground truth from the environments.

## Deviations from the plan and caveats

- F1. Bootstrap for AgentDojo quantities. Plan 3.5 prescribes suite|user_task cluster resampling for the within-AgentDojo UT analyses; it is used here as the primary interval for every AgentDojo P0 quantity (WUT, and also R4 transfer and the univariate AUROCs on P0, which use the same rows), with the stratified trajectory bootstrap as a sensitivity. AgentDrift R1/R3 use the stratified trajectory bootstrap (plan 3.5), with a world-cluster sensitivity for the observation-length ablations.
- F2. Tests and multiplicity. Two-sided percentile-bootstrap p-values, p = min(1, 2 min((1 + #{d <= 0})/(B + 1), (1 + #{d >= 0})/(B + 1))), so the smallest attainable value with B = 2000 is 0.001. Holm is applied across the five R3 folds separately for each contrast and metric, and, for within-minus-transfer, across the five detectors of that table only (plan family d restricted to the detectors computed here).
- F3. Added detector LR-obsonly (logistic regression on max_obs and mean_obs only, cache recipe). It is not in the plan's detector list; it reproduces the review's 'observation features alone' numbers and is reported as post hoc.
- F4. Coefficient intervals are not specified by the plan. R1: 500 refits on stratified resamples of data/train.jsonl (vocabulary fixed). WUT: mean, minimum and maximum over the 25 fold models, plus a descriptive model fitted on all P0 rows with 500 suite|user_task cluster-bootstrap refits.
- F5. Padding specifics the plan leaves open: the filler is the sentence ' The quick brown fox jumps over the lazy dog.' repeated and cut to exactly n characters, appended to the observation string; only positives are padded, negatives and training data are unchanged; recall uses the unchanged training threshold t_r80. In AgentDojo (WUT) the injected step(s) are located from the raw run's `injections` field; 9 of 2,294 compromised P0 traces have no located step and stay unpadded (results are also given on the located subset). Only lengthening is tested; in AgentDojo longer observations raise the attack score, so the evasive direction there would be shortening, which was not tested. The GRU is padded in R1 only (the task asked for WUT LR-C).
- F6. The three R1 GRU seeds were retrained for the padding test with gru_detector.train_gru and gru_ladder.stratified_val (as the cache). Thresholds and t_r80 predictions equal the cache exactly and pool-E AUROC is identical, but 1-2 of 1,881 scores per seed differ from the cache by at most 6e-8 (float32 rounding; not bitwise identical as the GRU cache README reports for its own refit). All padding levels use the refitted models.
- F7. Prevalence-weighted AUPRC (plan 3.1) is reported at pi = 0.01, 0.05 and 0.10 for the ablation tables; the plan does not fix the grid for this analysis. The t05 operating point is not reported here.
- F8. The state-change annotation (tool_state_annotation.json) was made by a single annotator (an AI assistant working for the authors) from tool names, example arguments and example observations. It was not blind: the keyword list was known and the listing also showed each tool's call count and step-label counts (how often it is a hijacked step or occurs in attacked runs). 13 AgentDrift labels and 1 AgentDojo label are marked ambiguous. No second rater, so no agreement statistic.
- F9. Bootstrap seeds aligned across parts (second run of this script, 2026-09-27). In the first run each part drew its own bootstrap resamples, so the same quantity (for example R4 LR-B AUROC) had slightly different intervals in the ablation, padding and annotation tables. The padding (R1 linear, R1 GRU, WUT) and annotation-sensitivity parts now use the resampling seed of the matching setting in the ablation part (9200 + setting index; 9300 + index for the sensitivity bootstrap), so identical quantities have identical intervals. Point estimates and fitted models are unchanged; only interval endpoints moved, by at most a few thousandths.

Checks [`checks`]: cached coefficients reproduce the R1 fits exactly; the 25 WUT fold models reproduce the cached out-of-fold scores (max difference below 1e-15) and thresholds; the refitted LR-B used in the annotation sensitivity equals the cached LR-B (below 1e-13) in all seven settings; the fast average-precision routine equals sklearn's.
