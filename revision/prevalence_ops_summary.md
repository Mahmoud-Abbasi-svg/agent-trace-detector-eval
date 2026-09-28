# Operating points and prevalence (revision stream `prevalence_ops`)

Script `revision/prevalence_ops.py` (SHA-256 `ce6e0dc6a7845bec...`); results `revision/prevalence_ops_results.json` (generated 2026-09-26T23:18:27+0200). Every number below is a value in that JSON. Key paths are given in brackets; `<pi>` is one of pi01, pi02, pi05, pi10 or native. Estimates are shown as point [95% bootstrap percentile interval], with B = 2000. The point is the plug-in value on the full pool; for the GRU it is the mean over its three seeds, and for within-AgentDojo detectors the mean over the five CV repetitions. Each bootstrap replicate draws one GRU seed or CV repetition. Scores are rounded to 12 decimals before ranking (m9); for AgentDojo, rank-based values from unrounded scores are also given (deviation P10). Thresholds come from training or validation data only. The one exception is the matched-recall quantity (FPR at a fixed test TPR of 0.8), which is labelled wherever it appears. 'Detected' means the 95% interval excludes zero. H means detected after Holm correction within the stated family, d means detected before correction but not after Holm, and nd means not detected. 'CP' is an exact Clopper-Pearson 95% interval, given when the bootstrap interval collapses. '>=' is a lower bound on precision when no negative was flagged.

Pools. R1 pool E has 831 attacked and 825 negative traces (600 benign, 225 hard negatives, so 27% hard negatives). Its 225 failed attacks are scored separately. AgentDojo P0 has 2,294 compromised and 392 benign traces. Its 4,987 unsuccessful attacks are scored separately.

## M8. AUPRC at controlled prevalence

**Question.** What is AUPRC at prevalence pi when it is estimated from all positives? The estimator is prevalence-weighted average precision (each positive weighted by pi/(1-pi) * n-/n+), with a stratified bootstrap of the full pool (plan 3.1). How does it compare with the v0.3 subsampling estimate?

**R1, prevalence-weighted AUPRC** [`M8_R1.detectors.<det>.wAUPRC.<pi>`; AUROC `M8_R1.detectors.<det>.AUROC`]

| Detector | AUROC | wAUPRC 1% | 2% | 5% | 10% | native (0.502) |
|---|---|---|---|---|---|---|
| LR-A | 0.777 [0.755, 0.799] | 0.089 [0.069, 0.164] | 0.140 [0.112, 0.214] | 0.255 [0.213, 0.327] | 0.382 [0.334, 0.445] | 0.794 [0.770, 0.816] |
| LR-B | 0.830 [0.810, 0.849] | 0.098 [0.076, 0.175] | 0.152 [0.119, 0.231] | 0.271 [0.224, 0.340] | 0.403 [0.354, 0.462] | 0.820 [0.796, 0.842] |
| LR-C | 0.879 [0.862, 0.895] | 0.156 [0.121, 0.261] | 0.232 [0.183, 0.330] | 0.380 [0.321, 0.462] | 0.523 [0.466, 0.590] | 0.876 [0.857, 0.895] |
| Markov | 0.724 [0.700, 0.748] | 0.024 [0.021, 0.042] | 0.047 [0.041, 0.069] | 0.113 [0.099, 0.144] | 0.210 [0.187, 0.250] | 0.694 [0.667, 0.726] |
| IForest | 0.588 [0.560, 0.615] | 0.012 [0.011, 0.019] | 0.023 [0.022, 0.031] | 0.058 [0.054, 0.067] | 0.115 [0.107, 0.129] | 0.540 [0.518, 0.567] |
| GRU | 0.936 [0.922, 0.947] | 0.265 [0.177, 0.442] | 0.382 [0.282, 0.537] | 0.564 [0.470, 0.671] | 0.699 [0.628, 0.770] | 0.936 [0.920, 0.950] |
| TFIDF | 0.999 [0.999, 1.000] | 0.975 [0.960, 0.991] | 0.984 [0.973, 0.994] | 0.992 [0.986, 0.997] | 0.996 [0.992, 0.998] | 0.999 [0.999, 1.000] |

Chance level of this estimator, as the mean [2.5th, 97.5th percentile] of 2,000 random scorers [`M8_R1.chance_wAUPRC_random_scorer.<pi>`]: 1%: 0.0114 [0.0093, 0.0164], 2%: 0.0216 [0.0187, 0.0272], 5%: 0.0519 [0.0467, 0.0593], 10%: 0.1024 [0.0935, 0.1139].

Plug-in point versus bootstrap mean for LR-C [`M8_R1.detectors.LR-C.wAUPRC.<pi>.point` / `.boot_mean`]: 1%: 0.156 vs 0.175, 2%: 0.232 vs 0.247, 5%: 0.380 vs 0.389, 10%: 0.523 vs 0.528.

**R1, paired differences in wAUPRC.** Same resampled indices for both detectors; Holm over the five comparisons within each pi. The AUROC column is prevalence-free [`M8_R1.paired.<pi>.<comparison>`, with `p_boot` and `p_holm`].

| Comparison | 1% | 2% | 5% | 10% | native AP | AUROC |
|---|---|---|---|---|---|---|
| LR-C-minus-Markov | +0.132 [+0.093, +0.235] H | +0.185 [+0.132, +0.280] H | +0.268 [+0.199, +0.348] H | +0.313 [+0.244, +0.380] H | +0.181 [+0.145, +0.212] H | +0.155 [+0.127, +0.182] H |
| LR-C-minus-LR-B | +0.058 [-0.016, +0.153] nd | +0.081 [+0.003, +0.168] H | +0.110 [+0.035, +0.187] H | +0.120 [+0.059, +0.182] H | +0.056 [+0.037, +0.076] H | +0.048 [+0.034, +0.064] H |
| LR-C-minus-LR-A | +0.067 [+0.037, +0.126] H | +0.092 [+0.057, +0.142] H | +0.126 [+0.087, +0.166] H | +0.141 [+0.106, +0.176] H | +0.082 [+0.068, +0.097] H | +0.101 [+0.086, +0.118] H |
| GRU-minus-LR-C | +0.109 [-0.016, +0.277] nd | +0.150 [+0.023, +0.296] H | +0.183 [+0.075, +0.289] H | +0.176 [+0.093, +0.253] H | +0.060 [+0.041, +0.078] H | +0.057 [+0.042, +0.072] H |
| TFIDF-minus-LR-C | +0.819 [+0.713, +0.859] H | +0.752 [+0.654, +0.802] H | +0.612 [+0.530, +0.672] H | +0.473 [+0.406, +0.529] H | +0.124 [+0.105, +0.143] H | +0.121 [+0.104, +0.137] H |

Comparisons detected after Holm, out of five [`M8_R1.n_detected_holm`]:

- weighted: 1% 3, 2% 5, 5% 5, 10% 5;
- world-cluster bootstrap: 1% 4, 2% 4, 5% 5, 10% 5;
- v0.3 subsampling: 1% 1, 2% 1, 5% 4, 10% 5.

**World-cluster sensitivity** (676 worlds, plan 3.5) [`M8_R1.world_cluster_sensitivity`]:

- LR-C wAUPRC: 1% 0.156 [0.120, 0.256], 2% 0.232 [0.182, 0.322], 5% 0.380 [0.322, 0.456], 10% 0.523 [0.468, 0.585].
- At 1%: LR-C-minus-Markov +0.132 [+0.090, +0.228] H, LR-C-minus-LR-B +0.058 [-0.023, +0.150] nd, LR-C-minus-LR-A +0.067 [+0.036, +0.122] H, GRU-minus-LR-C +0.109 [+0.014, +0.261] H, TFIDF-minus-LR-C +0.819 [+0.719, +0.859] H.
- At 2%: LR-C-minus-Markov +0.185 [+0.130, +0.271] H, LR-C-minus-LR-B +0.081 [-0.007, +0.165] nd, LR-C-minus-LR-A +0.092 [+0.054, +0.138] H, GRU-minus-LR-C +0.150 [+0.056, +0.284] H, TFIDF-minus-LR-C +0.752 [+0.662, +0.802] H.

**Secondary: the v0.3 subsampling estimator.** Each replicate draws n+ positives against 825 negatives. The replicate mean is the expected AP of a test set with that many positives, and the spread is the sampling distribution of such a small test set, not a confidence interval for AUPRC [`M8_R1.subsampling_v03_secondary.<pi>.AP.<det>`].

| Detector | 1% (n+=8) | 2% (n+=17) | 5% (n+=43) | 10% (n+=92) |
|---|---|---|---|---|
| LR-A | 0.127 [0.022, 0.361] | 0.166 [0.053, 0.346] | 0.270 [0.159, 0.403] | 0.389 [0.294, 0.491] |
| LR-B | 0.133 [0.023, 0.372] | 0.182 [0.061, 0.366] | 0.282 [0.173, 0.415] | 0.409 [0.323, 0.506] |
| LR-C | 0.197 [0.040, 0.473] | 0.259 [0.105, 0.466] | 0.392 [0.266, 0.537] | 0.528 [0.431, 0.627] |
| Markov | 0.041 [0.013, 0.141] | 0.063 [0.031, 0.131] | 0.126 [0.083, 0.184] | 0.221 [0.173, 0.284] |
| IForest | 0.018 [0.009, 0.053] | 0.030 [0.019, 0.060] | 0.064 [0.049, 0.089] | 0.121 [0.102, 0.148] |
| GRU | 0.324 [0.095, 0.647] | 0.414 [0.214, 0.639] | 0.574 [0.429, 0.717] | 0.705 [0.612, 0.792] |
| TFIDF | 0.975 [0.866, 1.000] | 0.985 [0.940, 1.000] | 0.992 [0.975, 1.000] | 0.996 [0.988, 1.000] |
| random scorer, expected AP | 0.0171 | 0.0275 | 0.0565 | 0.1066 |

Detectors rank in the same order under both estimators at every pi [`M8_R1.detector_order_by_estimator.<pi>.same_order`: True, True, True, True].

Paired differences under subsampling, with Holm within each pi [`M8_R1.subsampling_v03_secondary.<pi>.paired_AP` and `.paired_AUROC`]:

| Comparison | AP 1% | AP 2% | AP 5% | AP 10% | AUROC, 1% replicates | AUROC, 2% replicates |
|---|---|---|---|---|---|---|
| LR-C-minus-Markov | +0.157 [-0.015, +0.438] nd | +0.196 [+0.030, +0.403] d | +0.266 [+0.131, +0.416] H | +0.306 [+0.196, +0.414] H | +0.151 [-0.047, +0.364] nd | +0.156 [+0.022, +0.293] d |
| LR-C-minus-LR-B | +0.065 [-0.175, +0.318] nd | +0.078 [-0.104, +0.263] nd | +0.109 [-0.023, +0.241] nd | +0.118 [+0.027, +0.214] H | +0.046 [-0.068, +0.162] nd | +0.047 [-0.029, +0.126] nd |
| LR-C-minus-LR-A | +0.070 [-0.072, +0.249] nd | +0.094 [-0.020, +0.231] nd | +0.122 [+0.036, +0.211] H | +0.139 [+0.080, +0.200] H | +0.102 [-0.011, +0.253] nd | +0.102 [+0.018, +0.198] d |
| GRU-minus-LR-C | +0.126 [-0.149, +0.432] nd | +0.155 [-0.063, +0.378] nd | +0.182 [+0.033, +0.335] H | +0.177 [+0.073, +0.279] H | +0.058 [-0.047, +0.177] nd | +0.056 [-0.020, +0.136] nd |
| TFIDF-minus-LR-C | +0.777 [+0.497, +0.951] H | +0.726 [+0.520, +0.884] H | +0.601 [+0.458, +0.726] H | +0.468 [+0.369, +0.564] H | +0.122 [+0.032, +0.251] H | +0.121 [+0.054, +0.204] H |

**AgentDojo P0, prevalence-weighted AUPRC.** R4 is transfer from AgentDrift and WUT is within AgentDojo (user-task CV). The main interval is the stratified bootstrap of plan 3.1 [`M8_AgentDojo.schemes.traj_strat.settings.<S>.detectors.<det>.<pi>`]. The cluster column is the 117-cluster suite|user_task bootstrap of plan 3.5 [`...schemes.cluster_suite_user_task...`]. The unrounded column uses raw scores [`M8_AgentDojo_raw_scores.schemes.traj_strat...`].

| Setting | Detector | AUROC | AUROC unrounded | wAUPRC 5% | wAUPRC 10% | 10% cluster CI | 10% unrounded |
|---|---|---|---|---|---|---|---|
| R4 | LR-A | 0.565 [0.544, 0.588] | 0.565 | 0.074 [0.063, 0.123] | 0.138 [0.121, 0.178] | [0.114, 0.192] | 0.138 [0.121, 0.178] |
| R4 | LR-B | 0.480 [0.446, 0.511] | 0.482 | 0.045 [0.042, 0.049] | 0.090 [0.084, 0.098] | [0.082, 0.100] | 0.090 [0.084, 0.098] |
| R4 | LR-C | 0.470 [0.437, 0.501] | 0.473 | 0.044 [0.041, 0.048] | 0.088 [0.083, 0.096] | [0.081, 0.098] | 0.089 [0.083, 0.096] |
| R4 | GRU | 0.511 [0.482, 0.540] | 0.513 | 0.052 [0.048, 0.059] | 0.104 [0.096, 0.117] | [0.094, 0.118] | 0.104 [0.096, 0.117] |
| R4 | TFIDF | 0.520 [0.486, 0.554] | 0.520 | 0.052 [0.048, 0.059] | 0.100 [0.093, 0.111] | [0.090, 0.119] | 0.100 [0.093, 0.111] |
| WUT | LR-A | 0.788 [0.758, 0.816] | 0.788 | 0.177 [0.134, 0.266] | 0.303 [0.242, 0.402] | [0.231, 0.498] | 0.303 [0.242, 0.402] |
| WUT | LR-B | 0.741 [0.709, 0.771] | 0.741 | 0.129 [0.106, 0.175] | 0.228 [0.193, 0.284] | [0.185, 0.320] | 0.228 [0.193, 0.284] |
| WUT | LR-C | 0.797 [0.766, 0.831] | 0.797 | 0.172 [0.135, 0.257] | 0.297 [0.245, 0.388] | [0.227, 0.463] | 0.299 [0.246, 0.388] |
| WUT | GRU | 0.840 [0.795, 0.873] | 0.840 | 0.271 [0.192, 0.386] | 0.411 [0.325, 0.528] | [0.316, 0.618] | 0.411 [0.325, 0.528] |
| WUT | TFIDF | 0.996 [0.994, 0.998] | 0.996 | 0.967 [0.943, 0.986] | 0.980 [0.965, 0.991] | [0.955, 0.996] | 0.980 [0.965, 0.991] |

Chance level of the estimator on P0 [`M8_AgentDojo.chance_wAUPRC_random_scorer.<pi>`]: 5% 0.0531 [0.0469, 0.0635], 10% 0.1034 [0.0935, 0.1176].

Within minus transfer (WUT minus R4), wAUPRC at 10%, with Holm over five detectors [`M8_AgentDojo.schemes.<scheme>.within_minus_transfer.pi10`]:

- stratified bootstrap: LR-A +0.165 [+0.100, +0.256] H, LR-B +0.138 [+0.103, +0.193] H, LR-C +0.209 [+0.154, +0.301] H, GRU +0.307 [+0.221, +0.421] H, TFIDF +0.880 [+0.861, +0.894] H;
- cluster bootstrap: LR-A +0.165 [+0.084, +0.346] H, LR-B +0.138 [+0.096, +0.227] H, LR-C +0.209 [+0.139, +0.375] H, GRU +0.307 [+0.213, +0.515] H, TFIDF +0.880 [+0.849, +0.901] H.

Paired detector differences at 10%, with Holm over four comparisons [`M8_AgentDojo.schemes.<scheme>.settings.<S>.paired.pi10`; unrounded in `M8_AgentDojo_raw_scores`]:

| Setting | Comparison | stratified | cluster | stratified, unrounded |
|---|---|---|---|---|
| R4 | LR-C-minus-LR-B | -0.002 [-0.003, -0.001] H | -0.002 [-0.003, -0.001] H | -0.002 [-0.003, -0.001] H |
| R4 | LR-C-minus-LR-A | -0.049 [-0.089, -0.031] H | -0.049 [-0.104, -0.023] H | -0.049 [-0.089, -0.030] H |
| R4 | GRU-minus-LR-C | +0.016 [+0.011, +0.023] H | +0.016 [+0.009, +0.025] H | +0.016 [+0.011, +0.023] H |
| R4 | TFIDF-minus-LR-C | +0.012 [+0.006, +0.020] H | +0.012 [+0.002, +0.029] H | +0.012 [+0.006, +0.020] H |
| WUT | LR-C-minus-LR-B | +0.070 [+0.010, +0.149] d | +0.070 [-0.031, +0.220] nd | +0.071 [+0.012, +0.150] H |
| WUT | LR-C-minus-LR-A | -0.006 [-0.092, +0.073] nd | -0.006 [-0.151, +0.113] nd | -0.004 [-0.091, +0.074] nd |
| WUT | GRU-minus-LR-C | +0.113 [+0.008, +0.222] d | +0.113 [-0.035, +0.302] nd | +0.112 [+0.008, +0.222] d |
| WUT | TFIDF-minus-LR-C | +0.683 [+0.590, +0.737] H | +0.683 [+0.514, +0.759] H | +0.682 [+0.589, +0.736] H |

**'Reasonable within-AgentDojo detection' criteria of R4_CONTROL.md Sec. 7 on WUT (P0)** [`R4_CONTROL_within_criteria_WUT.detectors.<det>`]. This stream does not decide the Case, which needs C1 and IT.

| Detector | AUROC >= 0.80 | AUROC lower >= 0.70 | wAUPRC@10% >= 0.30 (stratified, rounded) | same, unrounded | v0.3 subsampling AP@10% >= 0.30 |
|---|---|---|---|---|---|
| LR-A | 0.788 no | 0.758 yes | 0.303 yes | 0.3030 yes | 0.328 yes |
| LR-B | 0.741 no | 0.709 yes | 0.228 no | 0.2279 no | 0.246 no |
| LR-C | 0.797 no | 0.766 yes | 0.297 no | 0.2985 no | 0.322 yes |
| GRU | 0.840 yes | 0.795 yes | 0.411 yes | 0.4108 yes | 0.430 yes |
| TFIDF | 0.996 yes | 0.994 yes | 0.980 yes | 0.9804 yes | 0.981 yes |

v0.3 subsampling on AgentDojo (secondary) [`M8_AgentDojo.subsampling_v03_secondary.<pi>.AP.<S>.<det>`], LR-C: at 10% (n+ = 44) R4 0.093 [0.079, 0.113] and WUT 0.322 [0.225, 0.453]; at 5% (n+ = 21) R4 0.049 [0.039, 0.066] and WUT 0.203 [0.116, 0.342]. A random scorer's expected AP is 0.1126 at 10% and 0.0638 at 5%.

**Plain reading.** Estimated from all 831 positives, LR-C's AUPRC at 2% is 0.232 [0.183, 0.330]. This interval is 2.5 times narrower than the v0.3 subsampling spread [0.105, 0.466]. The subsampling mean is higher than the weighted value at 1% (0.197 vs 0.156) and at 2% (0.259 vs 0.232), and the two agree at 10% (0.528 vs 0.523). Detectors rank in the same order under both estimators; what changes is power. With the weighted estimator, LR-C minus Markov is detected at every prevalence, 1% included. GRU minus LR-C and LR-C minus LR-B are detected from 2% upward, both close to the Holm threshold at 2%. Under subsampling, only TFIDF minus LR-C survives Holm at 1% and at 2%. The RQ1 message stands and is sharper. At 2% the best content-blind detector (GRU) has AUPRC 0.382, although its AUROC is 0.936. The Isolation Forest sits at the estimator's chance level, and only the content detector TFIDF is near 1 (0.984). On AgentDojo, reweighting removes the need for 21- or 44-positive subsamples. Transfer AUPRC at 10% is 0.088-0.104 for LR-B, LR-C, GRU and TFIDF, against a chance level of 0.103, and 0.138 for LR-A. Within AgentDojo it is 0.228-0.411 for the content-blind detectors and 0.980 for TFIDF. Within minus transfer is detected for every detector under both bootstraps. LR-C's within-AgentDojo wAUPRC at 10% (0.297; 0.2985 unrounded) falls just below the 0.30 'reasonable' threshold of R4_CONTROL.md. Its v0.3 subsampling estimate (0.322) passes that threshold, so the choice of estimator flips this criterion for LR-C. The numbers do not show what AUPRC a deployment would see. Reweighting assumes that the class-conditional score distributions of the test pool carry over, including its 27% hard-negative share. The intervals cover test-set sampling and seed or repetition choice, not retraining.

## m3 (prevalence family). Multiplicity at each prevalence

**Question.** Which of the paired prevalence comparisons remain detected after Holm correction within each prevalence (plan 3.6(e))?

The counts and decisions are in the M8 tables above [`M8_R1.paired.<pi>.<comparison>.p_holm` / `.detected_holm`; `M8_R1.n_detected_holm`]. The borderline cases at 2% are:

- LR-C-minus-LR-B: +0.081 [+0.003, +0.168], p_boot 0.0440, p_holm 0.0440 (H); under the world-cluster bootstrap +0.081 [-0.007, +0.165], p_holm 0.0710 (nd).
- GRU-minus-LR-C: +0.150 [+0.023, +0.296], p_boot 0.0200, p_holm 0.0400 (H); under the world-cluster bootstrap +0.150 [+0.056, +0.284], p_holm 0.0050 (H).
- AgentDojo WUT, LR-C-minus-LR-B at 10%: +0.070 [+0.010, +0.149], p_holm 0.0510 (d) with rounded scores; +0.071 [+0.012, +0.150], p_holm 0.0480 (H) with unrounded scores; +0.070 [-0.031, +0.220] (nd) under the cluster bootstrap.

**Plain reading.** With the weighted estimator, 3 of five comparisons survive Holm at 1% and 5, 5 and 5 at 2%, 5% and 10%. Under v0.3 subsampling the counts are 1, 1, 4 and 5. Two results at 2% are fragile: LR-C minus LR-B and GRU minus LR-C pass Holm with adjusted p just below 0.05. Under world clustering, LR-C minus LR-B at 2% is no longer detected. Within AgentDojo, LR-C minus LR-B at 10% passes Holm only with unrounded scores, and it is not detected under the cluster bootstrap. The paper should report LR-C > LR-B at 2% (R1) and at 10% (WUT) as borderline, and should not base a claim on any single comparison at 1%.

## m17 item 5. Each comparison at every prevalence

**Question.** Section VI-A shows each comparison only at the prevalence where it fails. What does each comparison show at every prevalence?

The M8 tables give all five comparisons at 1%, 2%, 5%, 10% and native prevalence, under both estimators. The review's example, LR-C minus Markov under v0.3 subsampling at 2%, is +0.196 [+0.030, +0.403] (d) [`M8_R1.subsampling_v03_secondary.pi02.paired_AP.LR-C-minus-Markov`]. The same comparison with the weighted estimator is +0.185 [+0.132, +0.280] (H), and at 1% +0.132 [+0.093, +0.235] (H).

**Plain reading.** Reported at a common prevalence, the comparisons do not support the Section VI-A contrast. Under subsampling, AUPRC and AUROC lose power together on the same small replicates (see the AUROC columns). With the weighted estimator, most AUPRC differences are detected even at 1% to 2%. Only LR-C minus LR-B and GRU minus LR-C remain undetected at 1%.

## M8 (2). Precision at theta_r80 by prevalence (Eq. 6) and alerts per 1,000 traces

**Question.** What precision and alert load does the training-derived threshold give at each prevalence? Both are computed from bootstrap TPR and FPR on pool E, using the pool-E negative mix.

R1 [`R1_operating_points.detectors.<det>.t_r80.<quantity>`]:

| Detector | TPR | FPR | Prec. 1% | Prec. 2% | Prec. 5% | Prec. 10% | Alerts/1k at 2% | Alerts/1k at 10% |
|---|---|---|---|---|---|---|---|---|
| LR-A | 0.744 [0.715, 0.773] | 0.335 [0.303, 0.366] | 0.022 [0.020, 0.024] | 0.043 [0.040, 0.048] | 0.105 [0.096, 0.115] | 0.198 [0.183, 0.216] | 343 [312, 374] | 375 [347, 404] |
| LR-B | 0.795 [0.768, 0.823] | 0.284 [0.253, 0.316] | 0.028 [0.025, 0.031] | 0.054 [0.048, 0.061] | 0.129 [0.116, 0.143] | 0.238 [0.217, 0.260] | 294 [264, 326] | 335 [307, 364] |
| LR-C | 0.771 [0.744, 0.799] | 0.192 [0.165, 0.217] | 0.039 [0.034, 0.045] | 0.076 [0.067, 0.088] | 0.175 [0.156, 0.199] | 0.309 [0.281, 0.344] | 203 [177, 228] | 249 [225, 272] |
| Markov | 0.810 [0.782, 0.836] | 0.482 [0.447, 0.515] | 0.017 [0.015, 0.018] | 0.033 [0.031, 0.036] | 0.081 [0.076, 0.088] | 0.157 [0.147, 0.169] | 489 [454, 520] | 515 [484, 543] |
| IForest | 0.785 [0.755, 0.812] | 0.623 [0.589, 0.655] | 0.013 [0.012, 0.013] | 0.025 [0.024, 0.027] | 0.062 [0.059, 0.066] | 0.123 [0.116, 0.130] | 626 [593, 657] | 639 [609, 668] |
| GRU | 0.682 [0.649, 0.717] | 0.045 [0.029, 0.062] | 0.134 [0.101, 0.195] | 0.238 [0.184, 0.328] | 0.446 [0.368, 0.557] | 0.629 [0.551, 0.727] | 58 [42, 74] | 109 [93, 124] |
| TFIDF | 0.763 [0.735, 0.791] | 0.000 [0.000, 0.000] (CP 0.000-0.004) | 1.000 [1.000, 1.000] (>= 0.625) | 1.000 [1.000, 1.000] (>= 0.771) | 1.000 [1.000, 1.000] (>= 0.897) | 1.000 [1.000, 1.000] (>= 0.948) | 15 [15, 16] | 76 [74, 79] |

**Plain reading.** At theta_r80, LR-C flags 0.771 of R1 attacks and 0.192 of negatives. Eq. (6) then gives a precision of 0.076 [0.067, 0.088] at 2%, with 203 alerts per 1,000 traces, or 12.2 false alerts per true alert. The GRU's lower FPR (0.045) gives 0.238 at 2% and 58 alerts per 1,000, at a recall of 0.682, below the 0.80 target. TFIDF flags no negative in pool E. Its bootstrap precision is therefore degenerate at 1, and the Clopper-Pearson bound only guarantees at least 0.771 at 2%. All values refer to the R1 pool-E negative mix (M9 shows how the mix changes them). Eq. (6) assumes that TPR and FPR do not change with prevalence.

## M9. Breakdown by negative class and hijack pattern

**Question.** At theta_r80, how do the operating-point rates split by negative class (benign, hard negative, failed attack) and by hijack pattern (full, partial, delayed)? How does precision at 2% and 10% depend on the negative mix?

The bootstrap is trajectory-level and stratified by category x pattern [`M9.<S>.schemes.traj_strat.detectors.<det>.t_r80.<quantity>`]. The table shows the main detectors. Every cached detector, theta = 0.5, and the GRU at theta_val80 are in the JSON. Precision uses the pooled pool-E recall together with the FPR of each negative mix. A precision resting on fewer than 10 flagged pool-E traces is marked with its alert count [`...t_r80.n_flagged_poolE`].

**R1** (rows: recall_poolE 831, FPR_poolE 825, FPR_benign 600, FPR_hard_negative 225, flag_failed_attack 225, recall_full 466, recall_partial 231, recall_delayed 134)

| Detector | FPR benign | FPR hard neg. | Failed flagged | Recall full | Recall partial | Recall delayed | Prec. 2%: benign-only / pool-E mix / hard-only | Prec. 10%: benign-only / pool-E mix / hard-only |
|---|---|---|---|---|---|---|---|---|
| LR-A | 0.295 [0.258, 0.333] | 0.440 [0.373, 0.502] | 0.351 [0.289, 0.413] | 0.893 [0.863, 0.921] | 0.502 [0.437, 0.567] | 0.642 [0.560, 0.716] | 0.049 [0.043, 0.056] / 0.043 [0.039, 0.048] / 0.033 [0.029, 0.039] | 0.219 [0.198, 0.243] / 0.198 [0.182, 0.215] / 0.158 [0.140, 0.181] |
| LR-B | 0.193 [0.163, 0.225] | 0.524 [0.458, 0.591] | 0.524 [0.462, 0.587] | 0.758 [0.717, 0.796] | 0.848 [0.801, 0.892] | 0.836 [0.776, 0.896] | 0.077 [0.067, 0.091] / 0.054 [0.049, 0.060] / 0.030 [0.027, 0.034] | 0.314 [0.282, 0.352] / 0.238 [0.220, 0.258] / 0.144 [0.130, 0.162] |
| LR-C | 0.118 [0.092, 0.147] | 0.387 [0.320, 0.449] | 0.218 [0.164, 0.271] | 0.830 [0.796, 0.863] | 0.727 [0.671, 0.784] | 0.642 [0.567, 0.724] | 0.117 [0.097, 0.145] / 0.076 [0.067, 0.087] / 0.039 [0.034, 0.047] | 0.420 [0.370, 0.481] / 0.309 [0.282, 0.341] / 0.181 [0.159, 0.211] |
| LR-C-drop | 0.115 [0.088, 0.142] | 0.378 [0.311, 0.440] | 0.227 [0.173, 0.280] | 0.833 [0.800, 0.867] | 0.719 [0.658, 0.775] | 0.612 [0.530, 0.694] | 0.120 [0.099, 0.149] / 0.077 [0.068, 0.089] / 0.040 [0.034, 0.048] | 0.425 [0.374, 0.487] / 0.313 [0.285, 0.347] / 0.184 [0.161, 0.214] |
| Markov | 0.412 [0.372, 0.452] | 0.671 [0.604, 0.729] | 0.618 [0.556, 0.680] | 0.775 [0.738, 0.811] | 0.827 [0.775, 0.874] | 0.903 [0.851, 0.948] | 0.039 [0.035, 0.043] / 0.033 [0.031, 0.036] / 0.024 [0.022, 0.027] | 0.179 [0.165, 0.195] / 0.157 [0.148, 0.168] / 0.118 [0.109, 0.130] |
| IForest | 0.628 [0.590, 0.667] | 0.609 [0.542, 0.676] | 0.791 [0.738, 0.844] | 0.811 [0.775, 0.843] | 0.697 [0.636, 0.758] | 0.843 [0.784, 0.903] | 0.025 [0.023, 0.027] / 0.025 [0.024, 0.027] / 0.026 [0.023, 0.029] | 0.122 [0.115, 0.130] / 0.123 [0.116, 0.130] / 0.125 [0.114, 0.139] |
| TFIDF | 0.000 [0.000, 0.000] (CP 0.000-0.006) | 0.000 [0.000, 0.000] (CP 0.000-0.016) | 0.000 [0.000, 0.000] (CP 0.000-0.016) | 0.974 [0.959, 0.987] | 0.541 [0.476, 0.606] | 0.410 [0.328, 0.493] | 1.000 [1.000, 1.000] (>= 0.711) / 1.000 [1.000, 1.000] (>= 0.772) / 1.000 [1.000, 1.000] (>= 0.481) | 1.000 [1.000, 1.000] (>= 0.931) / 1.000 [1.000, 1.000] (>= 0.948) / 1.000 [1.000, 1.000] (>= 0.835) |
| GRU | 0.024 [0.010, 0.040] | 0.101 [0.062, 0.142] | 0.043 [0.013, 0.084] | 0.722 [0.670, 0.768] | 0.606 [0.532, 0.675] | 0.674 [0.575, 0.761] | 0.373 [0.264, 0.580] / 0.238 [0.183, 0.328] / 0.122 [0.089, 0.184] | 0.762 [0.661, 0.883] / 0.629 [0.549, 0.727] / 0.430 [0.346, 0.551] |
| GRU-drop0.2 | 0.027 [0.013, 0.042] | 0.126 [0.076, 0.178] | 0.046 [0.018, 0.084] | 0.760 [0.719, 0.798] | 0.635 [0.567, 0.701] | 0.679 [0.597, 0.754] | 0.356 [0.260, 0.518] / 0.215 [0.166, 0.294] / 0.104 [0.075, 0.158] | 0.749 [0.657, 0.854] / 0.597 [0.521, 0.694] / 0.387 [0.307, 0.505] |

**R2** (rows: recall_poolE 775, FPR_poolE 729, FPR_benign 525, FPR_hard_negative 204, flag_failed_attack 218, recall_full 450, recall_partial 208, recall_delayed 117)

| Detector | FPR benign | FPR hard neg. | Failed flagged | Recall full | Recall partial | Recall delayed | Prec. 2%: benign-only / pool-E mix / hard-only | Prec. 10%: benign-only / pool-E mix / hard-only |
|---|---|---|---|---|---|---|---|---|
| LR-A | 0.381 [0.341, 0.423] | 0.407 [0.338, 0.471] | 0.431 [0.367, 0.500] | 0.873 [0.842, 0.902] | 0.639 [0.577, 0.702] | 0.641 [0.547, 0.726] | 0.040 [0.036, 0.045] / 0.039 [0.036, 0.043] / 0.037 [0.032, 0.044] | 0.184 [0.169, 0.203] / 0.182 [0.168, 0.196] / 0.175 [0.154, 0.202] |
| LR-B | 0.137 [0.109, 0.166] | 0.490 [0.422, 0.559] | 0.477 [0.408, 0.546] | 0.747 [0.704, 0.787] | 0.861 [0.812, 0.904] | 0.752 [0.675, 0.829] | 0.104 [0.087, 0.128] / 0.063 [0.057, 0.071] / 0.031 [0.028, 0.037] | 0.387 [0.341, 0.444] / 0.268 [0.246, 0.294] / 0.150 [0.134, 0.171] |
| LR-C | 0.095 [0.070, 0.120] | 0.289 [0.230, 0.348] | 0.234 [0.179, 0.294] | 0.827 [0.793, 0.862] | 0.803 [0.750, 0.856] | 0.632 [0.538, 0.718] | 0.145 [0.119, 0.188] / 0.097 [0.085, 0.114] / 0.053 [0.044, 0.066] | 0.480 [0.424, 0.558] / 0.370 [0.335, 0.413] / 0.233 [0.200, 0.278] |
| LR-C-drop | 0.090 [0.067, 0.112] | 0.299 [0.235, 0.363] | 0.271 [0.216, 0.330] | 0.833 [0.798, 0.867] | 0.808 [0.755, 0.856] | 0.598 [0.504, 0.684] | 0.153 [0.125, 0.197] / 0.098 [0.085, 0.116] / 0.051 [0.043, 0.064] | 0.495 [0.436, 0.573] / 0.372 [0.336, 0.416] / 0.227 [0.195, 0.272] |
| Markov | 0.562 [0.520, 0.606] | 0.667 [0.603, 0.730] | 0.706 [0.647, 0.766] | 0.720 [0.676, 0.762] | 0.865 [0.817, 0.909] | 0.923 [0.872, 0.966] | 0.028 [0.026, 0.030] / 0.027 [0.025, 0.028] / 0.024 [0.021, 0.026] | 0.135 [0.125, 0.145] / 0.129 [0.121, 0.137] / 0.116 [0.107, 0.128] |
| IForest | 0.691 [0.653, 0.731] | 0.676 [0.608, 0.740] | 0.743 [0.683, 0.798] | 0.769 [0.729, 0.807] | 0.707 [0.644, 0.765] | 0.872 [0.812, 0.932] | 0.022 [0.021, 0.024] / 0.022 [0.021, 0.024] / 0.023 [0.021, 0.025] | 0.110 [0.103, 0.117] / 0.110 [0.104, 0.116] / 0.112 [0.103, 0.123] |
| TFIDF | 0.000 [0.000, 0.000] (CP 0.000-0.007) | 0.000 [0.000, 0.000] (CP 0.000-0.018) | 0.000 [0.000, 0.000] (CP 0.000-0.017) | 0.973 [0.958, 0.987] | 0.510 [0.447, 0.577] | 0.214 [0.145, 0.291] | 1.000 [1.000, 1.000] (>= 0.675) / 1.000 [1.000, 1.000] (>= 0.742) / 1.000 [1.000, 1.000] (>= 0.448) | 1.000 [1.000, 1.000] (>= 0.919) / 1.000 [1.000, 1.000] (>= 0.940) / 1.000 [1.000, 1.000] (>= 0.815) |
| GRU | 0.027 [0.008, 0.048] | 0.100 [0.054, 0.152] | 0.067 [0.037, 0.101] | 0.700 [0.653, 0.742] | 0.601 [0.505, 0.683] | 0.584 [0.487, 0.684] | 0.351 [0.218, 0.632] / 0.227 [0.158, 0.346] / 0.120 [0.081, 0.199] | 0.737 [0.603, 0.903] / 0.611 [0.505, 0.742] / 0.425 [0.325, 0.576] |

The cluster bootstrap for LR-C (35 task clusters; plan 3.5) [`M9.R2.schemes.cluster_task.detectors.LR-C.t_r80`] gives: FPR benign 0.095 [0.049, 0.149]; failed flagged 0.234 [0.146, 0.329]; recall 0.791 [0.747, 0.832]; precision at 2% (pool-E mix) 0.097 [0.077, 0.133].

**R3_medical** (rows: recall_poolE 1020, FPR_poolE 935, FPR_benign 680, FPR_hard_negative 255, flag_failed_attack 255, recall_full 618, recall_partial 247, recall_delayed 155)

| Detector | FPR benign | FPR hard neg. | Failed flagged | Recall full | Recall partial | Recall delayed | Prec. 2%: benign-only / pool-E mix / hard-only | Prec. 10%: benign-only / pool-E mix / hard-only |
|---|---|---|---|---|---|---|---|---|
| LR-A | 0.000 [0.000, 0.000] (CP 0.000-0.005) | 0.000 [0.000, 0.000] (CP 0.000-0.014) | 0.000 [0.000, 0.000] (CP 0.000-0.014) | 0.000 [0.000, 0.000] (CP 0.000-0.006) | 0.000 [0.000, 0.000] (CP 0.000-0.015) | 0.000 [0.000, 0.000] (CP 0.000-0.024) | undefined (no alert) / undefined (no alert) / undefined (no alert) | undefined (no alert) / undefined (no alert) / undefined (no alert) |
| LR-B | 0.090 [0.069, 0.112] | 0.310 [0.255, 0.365] | 0.396 [0.337, 0.455] | 0.468 [0.427, 0.506] | 0.830 [0.781, 0.879] | 0.497 [0.419, 0.574] | 0.113 [0.092, 0.141] / 0.071 [0.062, 0.082] / 0.036 [0.030, 0.043] | 0.409 [0.356, 0.471] / 0.293 [0.264, 0.327] / 0.167 [0.144, 0.198] |
| LR-C | 0.451 [0.413, 0.488] | 0.639 [0.580, 0.694] | 0.576 [0.514, 0.635] | 0.759 [0.725, 0.791] | 0.951 [0.923, 0.976] | 0.903 [0.852, 0.948] | 0.036 [0.033, 0.039] / 0.033 [0.031, 0.035] / 0.026 [0.024, 0.028] | 0.169 [0.158, 0.182] / 0.155 [0.146, 0.164] / 0.126 [0.116, 0.137] |
| LR-C-drop | 0.063 [0.046, 0.082] | 0.275 [0.224, 0.329] | 0.271 [0.216, 0.325] | 0.333 [0.296, 0.372] | 0.761 [0.709, 0.814] | 0.419 [0.342, 0.503] | 0.127 [0.099, 0.167] / 0.071 [0.060, 0.084] / 0.032 [0.027, 0.040] | 0.442 [0.375, 0.523] / 0.293 [0.259, 0.333] / 0.154 [0.131, 0.185] |
| Markov | 1.000 [1.000, 1.000] (CP 0.995-1.000) | 1.000 [1.000, 1.000] (CP 0.986-1.000) | 1.000 [1.000, 1.000] (CP 0.986-1.000) | 1.000 [1.000, 1.000] (CP 0.994-1.000) | 1.000 [1.000, 1.000] (CP 0.985-1.000) | 1.000 [1.000, 1.000] (CP 0.976-1.000) | 0.020 [0.020, 0.020] / 0.020 [0.020, 0.020] / 0.020 [0.020, 0.020] | 0.100 [0.100, 0.100] / 0.100 [0.100, 0.100] / 0.100 [0.100, 0.100] |
| IForest | 0.000 [0.000, 0.000] (CP 0.000-0.005) | 0.000 [0.000, 0.000] (CP 0.000-0.014) | 0.000 [0.000, 0.000] (CP 0.000-0.014) | 0.000 [0.000, 0.000] (CP 0.000-0.006) | 0.004 [0.000, 0.012] | 0.000 [0.000, 0.000] (CP 0.000-0.024) | 1.000, unreliable (1 alert) / 1.000, unreliable (1 alert) / 1.000, unreliable (1 alert) | 1.000, unreliable (1 alert) / 1.000, unreliable (1 alert) / 1.000, unreliable (1 alert) |
| TFIDF | 0.000 [0.000, 0.000] (CP 0.000-0.005) | 0.000 [0.000, 0.000] (CP 0.000-0.014) | 0.000 [0.000, 0.000] (CP 0.000-0.014) | 0.984 [0.972, 0.992] | 0.211 [0.162, 0.263] | 0.071 [0.032, 0.116] | 1.000 [1.000, 1.000] (>= 0.708) / 1.000 [1.000, 1.000] (>= 0.769) / 1.000 [1.000, 1.000] (>= 0.478) | 1.000 [1.000, 1.000] (>= 0.930) / 1.000 [1.000, 1.000] (>= 0.948) / 1.000 [1.000, 1.000] (>= 0.833) |
| GRU | 0.712 [0.637, 0.771] | 0.728 [0.620, 0.808] | 0.637 [0.561, 0.718] | 0.911 [0.851, 0.955] | 0.969 [0.915, 0.996] | 0.998 [0.981, 1.000] | 0.026 [0.025, 0.028] / 0.026 [0.025, 0.028] / 0.026 [0.023, 0.029] | 0.128 [0.121, 0.137] / 0.127 [0.121, 0.136] / 0.126 [0.116, 0.141] |
| GRU-drop0.2 | 0.028 [0.004, 0.072] | 0.058 [0.000, 0.173] | 0.030 [0.000, 0.094] | 0.049 [0.005, 0.131] | 0.510 [0.296, 0.753] | 0.142 [0.019, 0.323] | 0.142 [0.073, 0.327] / 0.131 [0.057, 0.292] / 0.117 [0.032, 1.000] | 0.465 [0.300, 0.725] / 0.434 [0.248, 0.691] / 0.388 [0.151, 1.000] |

**R3_banking** (rows: recall_poolE 626, FPR_poolE 935, FPR_benign 680, FPR_hard_negative 255, flag_failed_attack 255, recall_full 217, recall_partial 260, recall_delayed 149)

| Detector | FPR benign | FPR hard neg. | Failed flagged | Recall full | Recall partial | Recall delayed | Prec. 2%: benign-only / pool-E mix / hard-only | Prec. 10%: benign-only / pool-E mix / hard-only |
|---|---|---|---|---|---|---|---|---|
| LR-A | 0.000 [0.000, 0.000] (CP 0.000-0.005) | 0.000 [0.000, 0.000] (CP 0.000-0.014) | 0.000 [0.000, 0.000] (CP 0.000-0.014) | 0.000 [0.000, 0.000] (CP 0.000-0.017) | 0.000 [0.000, 0.000] (CP 0.000-0.014) | 0.000 [0.000, 0.000] (CP 0.000-0.024) | undefined (no alert) / undefined (no alert) / undefined (no alert) | undefined (no alert) / undefined (no alert) / undefined (no alert) |
| LR-B | 0.150 [0.124, 0.176] | 0.620 [0.561, 0.678] | 0.737 [0.682, 0.788] | 0.659 [0.594, 0.719] | 0.958 [0.931, 0.981] | 0.852 [0.792, 0.906] | 0.101 [0.087, 0.121] / 0.057 [0.052, 0.063] / 0.027 [0.024, 0.029] | 0.380 [0.341, 0.427] / 0.249 [0.232, 0.268] / 0.129 [0.119, 0.141] |
| LR-C | 0.599 [0.562, 0.634] | 0.824 [0.776, 0.867] | 0.831 [0.784, 0.875] | 0.917 [0.876, 0.949] | 0.996 [0.988, 1.000] | 0.987 [0.966, 1.000] | 0.032 [0.030, 0.034] / 0.029 [0.028, 0.030] / 0.023 [0.022, 0.025] | 0.152 [0.144, 0.161] / 0.140 [0.134, 0.146] / 0.115 [0.110, 0.122] |
| LR-C-drop | 0.096 [0.075, 0.118] | 0.557 [0.498, 0.616] | 0.651 [0.592, 0.710] | 0.525 [0.461, 0.590] | 0.946 [0.915, 0.969] | 0.805 [0.738, 0.866] | 0.141 [0.116, 0.174] / 0.066 [0.060, 0.073] / 0.027 [0.024, 0.031] | 0.471 [0.417, 0.535] / 0.278 [0.257, 0.300] / 0.133 [0.120, 0.147] |
| Markov | 1.000 [1.000, 1.000] (CP 0.995-1.000) | 1.000 [1.000, 1.000] (CP 0.986-1.000) | 1.000 [1.000, 1.000] (CP 0.986-1.000) | 1.000 [1.000, 1.000] (CP 0.983-1.000) | 1.000 [1.000, 1.000] (CP 0.986-1.000) | 1.000 [1.000, 1.000] (CP 0.976-1.000) | 0.020 [0.020, 0.020] / 0.020 [0.020, 0.020] / 0.020 [0.020, 0.020] | 0.100 [0.100, 0.100] / 0.100 [0.100, 0.100] / 0.100 [0.100, 0.100] |
| IForest | 0.001 [0.000, 0.004] | 0.000 [0.000, 0.000] (CP 0.000-0.014) | 0.000 [0.000, 0.000] (CP 0.000-0.014) | 0.000 [0.000, 0.000] (CP 0.000-0.017) | 0.012 [0.000, 0.027] | 0.000 [0.000, 0.000] (CP 0.000-0.024) | 0.062, unreliable (4 alerts) / 0.084, unreliable (4 alerts) / 1.000, unreliable (4 alerts) | 0.266, unreliable (4 alerts) / 0.332, unreliable (4 alerts) / 1.000, unreliable (4 alerts) |
| TFIDF | 0.000 [0.000, 0.000] (CP 0.000-0.005) | 0.000 [0.000, 0.000] (CP 0.000-0.014) | 0.000 [0.000, 0.000] (CP 0.000-0.014) | 0.733 [0.677, 0.793] | 0.038 [0.019, 0.062] | 0.007 [0.000, 0.020] | 1.000 [1.000, 1.000] (>= 0.485) / 1.000 [1.000, 1.000] (>= 0.564) / 1.000 [1.000, 1.000] (>= 0.262) | 1.000 [1.000, 1.000] (>= 0.837) / 1.000 [1.000, 1.000] (>= 0.875) / 1.000 [1.000, 1.000] (>= 0.658) |
| GRU | 0.427 [0.000, 0.710] | 0.465 [0.000, 0.761] | 0.473 [0.000, 0.773] | 0.544 [0.000, 0.917] | 0.664 [0.004, 1.000] | 0.662 [0.000, 1.000] | 0.353 [0.027, 1.000] / 0.352 [0.027, 1.000] / 0.351 [0.025, 1.000] | 0.426 [0.130, 1.000] / 0.424 [0.129, 1.000] / 0.419 [0.121, 1.000] |
| GRU-drop0.2 | 0.022 [0.006, 0.038] | 0.064 [0.000, 0.149] | 0.064 [0.000, 0.133] | 0.065 [0.005, 0.129] | 0.568 [0.273, 0.762] | 0.293 [0.040, 0.497] | 0.231 [0.143, 0.365] / 0.189 [0.124, 0.362] / 0.206 [0.056, 1.000] | 0.620 [0.476, 0.758] / 0.552 [0.435, 0.755] / 0.498 [0.245, 1.000] |

**R3_web** (rows: recall_poolE 1020, FPR_poolE 935, FPR_benign 680, FPR_hard_negative 255, flag_failed_attack 255, recall_full 607, recall_partial 263, recall_delayed 150)

| Detector | FPR benign | FPR hard neg. | Failed flagged | Recall full | Recall partial | Recall delayed | Prec. 2%: benign-only / pool-E mix / hard-only | Prec. 10%: benign-only / pool-E mix / hard-only |
|---|---|---|---|---|---|---|---|---|
| LR-A | 0.000 [0.000, 0.000] (CP 0.000-0.005) | 0.000 [0.000, 0.000] (CP 0.000-0.014) | 0.000 [0.000, 0.000] (CP 0.000-0.014) | 0.000 [0.000, 0.000] (CP 0.000-0.006) | 0.000 [0.000, 0.000] (CP 0.000-0.014) | 0.000 [0.000, 0.000] (CP 0.000-0.024) | undefined (no alert) / undefined (no alert) / undefined (no alert) | undefined (no alert) / undefined (no alert) / undefined (no alert) |
| LR-B | 0.088 [0.069, 0.110] | 0.486 [0.424, 0.549] | 0.424 [0.361, 0.482] | 0.840 [0.811, 0.870] | 0.833 [0.787, 0.878] | 0.747 [0.673, 0.814] | 0.160 [0.133, 0.198] / 0.079 [0.071, 0.088] / 0.033 [0.030, 0.038] | 0.509 [0.454, 0.574] / 0.318 [0.293, 0.345] / 0.159 [0.143, 0.178] |
| LR-C | 0.632 [0.599, 0.669] | 0.906 [0.871, 0.941] | 0.745 [0.690, 0.800] | 0.990 [0.982, 0.997] | 0.992 [0.981, 1.000] | 0.987 [0.967, 1.000] | 0.031 [0.029, 0.033] / 0.028 [0.027, 0.029] / 0.022 [0.021, 0.023] | 0.148 [0.141, 0.155] / 0.135 [0.130, 0.139] / 0.108 [0.105, 0.112] |
| LR-C-drop | 0.071 [0.053, 0.090] | 0.427 [0.369, 0.486] | 0.341 [0.282, 0.396] | 0.796 [0.763, 0.827] | 0.806 [0.757, 0.856] | 0.713 [0.640, 0.787] | 0.185 [0.151, 0.234] / 0.087 [0.077, 0.098] / 0.036 [0.032, 0.042] | 0.553 [0.492, 0.625] / 0.342 [0.314, 0.371] / 0.170 [0.152, 0.191] |
| Markov | 1.000 [1.000, 1.000] (CP 0.995-1.000) | 1.000 [1.000, 1.000] (CP 0.986-1.000) | 1.000 [1.000, 1.000] (CP 0.986-1.000) | 1.000 [1.000, 1.000] (CP 0.994-1.000) | 1.000 [1.000, 1.000] (CP 0.986-1.000) | 1.000 [1.000, 1.000] (CP 0.976-1.000) | 0.020 [0.020, 0.020] / 0.020 [0.020, 0.020] / 0.020 [0.020, 0.020] | 0.100 [0.100, 0.100] / 0.100 [0.100, 0.100] / 0.100 [0.100, 0.100] |
| IForest | 0.000 [0.000, 0.000] (CP 0.000-0.005) | 0.000 [0.000, 0.000] (CP 0.000-0.014) | 0.000 [0.000, 0.000] (CP 0.000-0.014) | 0.000 [0.000, 0.000] (CP 0.000-0.006) | 0.000 [0.000, 0.000] (CP 0.000-0.014) | 0.000 [0.000, 0.000] (CP 0.000-0.024) | undefined (no alert) / undefined (no alert) / undefined (no alert) | undefined (no alert) / undefined (no alert) / undefined (no alert) |
| TFIDF | 0.000 [0.000, 0.000] (CP 0.000-0.005) | 0.000 [0.000, 0.000] (CP 0.000-0.014) | 0.000 [0.000, 0.000] (CP 0.000-0.014) | 0.837 [0.807, 0.867] | 0.274 [0.224, 0.327] | 0.020 [0.000, 0.047] | 1.000 [1.000, 1.000] (>= 0.675) / 1.000 [1.000, 1.000] (>= 0.740) / 1.000 [1.000, 1.000] (>= 0.439) | 1.000 [1.000, 1.000] (>= 0.919) / 1.000 [1.000, 1.000] (>= 0.939) / 1.000 [1.000, 1.000] (>= 0.810) |
| GRU | 0.594 [0.435, 0.750] | 0.702 [0.541, 0.867] | 0.586 [0.463, 0.698] | 0.901 [0.769, 0.998] | 0.919 [0.802, 1.000] | 0.978 [0.913, 1.000] | 0.031 [0.026, 0.038] / 0.030 [0.026, 0.035] / 0.026 [0.023, 0.030] | 0.149 [0.128, 0.175] / 0.142 [0.125, 0.163] / 0.128 [0.113, 0.145] |
| GRU-drop0.2 | 0.014 [0.000, 0.053] | 0.052 [0.000, 0.173] | 0.010 [0.000, 0.047] | 0.160 [0.003, 0.428] | 0.381 [0.129, 0.730] | 0.249 [0.027, 0.640] | 0.733 [0.162, 1.000] / 0.368 [0.112, 1.000] / 0.160 [0.056, 1.000] | 0.858 [0.512, 1.000] / 0.708 [0.407, 1.000] / 0.484 [0.244, 1.000] |

**R3_email** (rows: recall_poolE 1020, FPR_poolE 935, FPR_benign 680, FPR_hard_negative 255, flag_failed_attack 255, recall_full 605, recall_partial 264, recall_delayed 151)

| Detector | FPR benign | FPR hard neg. | Failed flagged | Recall full | Recall partial | Recall delayed | Prec. 2%: benign-only / pool-E mix / hard-only | Prec. 10%: benign-only / pool-E mix / hard-only |
|---|---|---|---|---|---|---|---|---|
| LR-A | 0.000 [0.000, 0.000] (CP 0.000-0.005) | 0.000 [0.000, 0.000] (CP 0.000-0.014) | 0.000 [0.000, 0.000] (CP 0.000-0.014) | 0.000 [0.000, 0.000] (CP 0.000-0.006) | 0.000 [0.000, 0.000] (CP 0.000-0.014) | 0.000 [0.000, 0.000] (CP 0.000-0.024) | undefined (no alert) / undefined (no alert) / undefined (no alert) | undefined (no alert) / undefined (no alert) / undefined (no alert) |
| LR-B | 0.357 [0.322, 0.394] | 0.580 [0.522, 0.639] | 0.424 [0.361, 0.486] | 0.820 [0.790, 0.851] | 0.864 [0.822, 0.902] | 0.834 [0.775, 0.887] | 0.045 [0.041, 0.050] / 0.039 [0.036, 0.042] / 0.028 [0.026, 0.032] | 0.206 [0.189, 0.224] / 0.181 [0.170, 0.194] / 0.138 [0.126, 0.151] |
| LR-C | 0.685 [0.650, 0.718] | 0.820 [0.773, 0.867] | 0.616 [0.557, 0.675] | 0.950 [0.932, 0.967] | 0.977 [0.958, 0.992] | 0.980 [0.954, 1.000] | 0.028 [0.027, 0.029] / 0.026 [0.025, 0.028] / 0.023 [0.022, 0.025] | 0.135 [0.129, 0.141] / 0.129 [0.124, 0.134] / 0.115 [0.110, 0.122] |
| LR-C-drop | 0.324 [0.290, 0.360] | 0.522 [0.463, 0.584] | 0.353 [0.294, 0.412] | 0.752 [0.717, 0.785] | 0.830 [0.784, 0.871] | 0.775 [0.709, 0.834] | 0.047 [0.042, 0.052] / 0.040 [0.037, 0.044] / 0.029 [0.026, 0.033] | 0.210 [0.192, 0.231] / 0.186 [0.173, 0.200] / 0.142 [0.129, 0.157] |
| Markov | 1.000 [1.000, 1.000] (CP 0.995-1.000) | 1.000 [1.000, 1.000] (CP 0.986-1.000) | 1.000 [1.000, 1.000] (CP 0.986-1.000) | 1.000 [1.000, 1.000] (CP 0.994-1.000) | 1.000 [1.000, 1.000] (CP 0.986-1.000) | 1.000 [1.000, 1.000] (CP 0.976-1.000) | 0.020 [0.020, 0.020] / 0.020 [0.020, 0.020] / 0.020 [0.020, 0.020] | 0.100 [0.100, 0.100] / 0.100 [0.100, 0.100] / 0.100 [0.100, 0.100] |
| IForest | 0.000 [0.000, 0.000] (CP 0.000-0.005) | 0.000 [0.000, 0.000] (CP 0.000-0.014) | 0.012 [0.000, 0.027] | 0.000 [0.000, 0.000] (CP 0.000-0.006) | 0.004 [0.000, 0.011] | 0.007 [0.000, 0.020] | 1.000, unreliable (2 alerts) / 1.000, unreliable (2 alerts) / 1.000, unreliable (2 alerts) | 1.000, unreliable (2 alerts) / 1.000, unreliable (2 alerts) / 1.000, unreliable (2 alerts) |
| TFIDF | 0.000 [0.000, 0.000] (CP 0.000-0.005) | 0.000 [0.000, 0.000] (CP 0.000-0.014) | 0.000 [0.000, 0.000] (CP 0.000-0.014) | 0.967 [0.950, 0.980] | 0.345 [0.288, 0.402] | 0.225 [0.159, 0.292] | 1.000 [1.000, 1.000] (>= 0.718) / 1.000 [1.000, 1.000] (>= 0.778) / 1.000 [1.000, 1.000] (>= 0.490) | 1.000 [1.000, 1.000] (>= 0.933) / 1.000 [1.000, 1.000] (>= 0.950) / 1.000 [1.000, 1.000] (>= 0.839) |
| GRU | 0.744 [0.666, 0.821] | 0.820 [0.725, 0.910] | 0.574 [0.388, 0.725] | 0.958 [0.916, 0.990] | 0.990 [0.955, 1.000] | 1.000 [1.000, 1.000] (CP 0.976-1.000) | 0.026 [0.024, 0.028] / 0.025 [0.023, 0.027] / 0.024 [0.022, 0.026] | 0.127 [0.118, 0.137] / 0.124 [0.116, 0.133] / 0.117 [0.108, 0.127] |
| GRU-drop0.2 | 0.036 [0.022, 0.053] | 0.046 [0.024, 0.075] | 0.024 [0.008, 0.047] | 0.124 [0.098, 0.152] | 0.465 [0.356, 0.561] | 0.216 [0.119, 0.318] | 0.113 [0.081, 0.171] / 0.106 [0.078, 0.150] / 0.091 [0.056, 0.170] | 0.409 [0.324, 0.529] / 0.392 [0.317, 0.490] / 0.354 [0.246, 0.526] |

**R3_coding** (rows: recall_poolE 1020, FPR_poolE 935, FPR_benign 680, FPR_hard_negative 255, flag_failed_attack 255, recall_full 597, recall_partial 259, recall_delayed 164)

| Detector | FPR benign | FPR hard neg. | Failed flagged | Recall full | Recall partial | Recall delayed | Prec. 2%: benign-only / pool-E mix / hard-only | Prec. 10%: benign-only / pool-E mix / hard-only |
|---|---|---|---|---|---|---|---|---|
| LR-A | 0.000 [0.000, 0.000] (CP 0.000-0.005) | 0.000 [0.000, 0.000] (CP 0.000-0.014) | 0.000 [0.000, 0.000] (CP 0.000-0.014) | 0.000 [0.000, 0.000] (CP 0.000-0.006) | 0.000 [0.000, 0.000] (CP 0.000-0.014) | 0.000 [0.000, 0.000] (CP 0.000-0.022) | undefined (no alert) / undefined (no alert) / undefined (no alert) | undefined (no alert) / undefined (no alert) / undefined (no alert) |
| LR-B | 0.213 [0.182, 0.244] | 0.471 [0.412, 0.533] | 0.565 [0.502, 0.627] | 0.834 [0.804, 0.863] | 0.853 [0.807, 0.896] | 0.811 [0.750, 0.866] | 0.074 [0.065, 0.085] / 0.057 [0.052, 0.063] / 0.035 [0.031, 0.040] | 0.303 [0.276, 0.337] / 0.247 [0.229, 0.267] / 0.165 [0.148, 0.185] |
| LR-C | 0.610 [0.575, 0.647] | 0.835 [0.792, 0.879] | 0.753 [0.698, 0.804] | 0.977 [0.963, 0.988] | 0.961 [0.938, 0.985] | 0.982 [0.957, 1.000] | 0.032 [0.030, 0.034] / 0.029 [0.027, 0.030] / 0.023 [0.022, 0.025] | 0.151 [0.143, 0.159] / 0.139 [0.133, 0.144] / 0.115 [0.109, 0.121] |
| LR-C-drop | 0.178 [0.149, 0.207] | 0.435 [0.376, 0.498] | 0.431 [0.373, 0.494] | 0.786 [0.750, 0.819] | 0.819 [0.772, 0.865] | 0.750 [0.683, 0.817] | 0.083 [0.072, 0.098] / 0.061 [0.055, 0.068] / 0.036 [0.031, 0.041] | 0.330 [0.296, 0.372] / 0.261 [0.240, 0.284] / 0.168 [0.150, 0.190] |
| Markov | 1.000 [1.000, 1.000] (CP 0.995-1.000) | 1.000 [1.000, 1.000] (CP 0.986-1.000) | 1.000 [1.000, 1.000] (CP 0.986-1.000) | 1.000 [1.000, 1.000] (CP 0.994-1.000) | 1.000 [1.000, 1.000] (CP 0.986-1.000) | 1.000 [1.000, 1.000] (CP 0.978-1.000) | 0.020 [0.020, 0.020] / 0.020 [0.020, 0.020] / 0.020 [0.020, 0.020] | 0.100 [0.100, 0.100] / 0.100 [0.100, 0.100] / 0.100 [0.100, 0.100] |
| IForest | 0.000 [0.000, 0.000] (CP 0.000-0.005) | 0.000 [0.000, 0.000] (CP 0.000-0.014) | 0.000 [0.000, 0.000] (CP 0.000-0.014) | 0.000 [0.000, 0.000] (CP 0.000-0.006) | 0.000 [0.000, 0.000] (CP 0.000-0.014) | 0.000 [0.000, 0.000] (CP 0.000-0.022) | undefined (no alert) / undefined (no alert) / undefined (no alert) | undefined (no alert) / undefined (no alert) / undefined (no alert) |
| TFIDF | 0.000 [0.000, 0.000] (CP 0.000-0.005) | 0.000 [0.000, 0.000] (CP 0.000-0.014) | 0.000 [0.000, 0.000] (CP 0.000-0.014) | 0.988 [0.980, 0.995] | 0.158 [0.116, 0.201] | 0.006 [0.000, 0.018] | 1.000 [1.000, 1.000] (>= 0.696) / 1.000 [1.000, 1.000] (>= 0.759) / 1.000 [1.000, 1.000] (>= 0.463) | 1.000 [1.000, 1.000] (>= 0.926) / 1.000 [1.000, 1.000] (>= 0.945) / 1.000 [1.000, 1.000] (>= 0.825) |
| GRU | 0.468 [0.057, 0.697] | 0.535 [0.114, 0.780] | 0.416 [0.000, 0.686] | 0.704 [0.213, 0.953] | 0.753 [0.367, 0.965] | 0.730 [0.165, 1.000] | 0.043 [0.027, 0.091] / 0.038 [0.026, 0.069] / 0.030 [0.024, 0.048] | 0.191 [0.130, 0.354] / 0.172 [0.128, 0.286] / 0.142 [0.118, 0.215] |
| GRU-drop0.2 | 0.080 [0.031, 0.129] | 0.153 [0.082, 0.239] | 0.031 [0.004, 0.067] | 0.320 [0.219, 0.414] | 0.609 [0.483, 0.737] | 0.339 [0.152, 0.506] | 0.100 [0.064, 0.172] / 0.079 [0.055, 0.115] / 0.051 [0.036, 0.076] | 0.372 [0.272, 0.530] / 0.316 [0.241, 0.413] / 0.227 [0.169, 0.309] |

**R4** (rows: recall_poolE 2294, FPR_poolE 392, FPR_benign 392, flag_failed_attack 4987)

| Detector | FPR benign (= pool E) | Unsuccessful attacks flagged | Recall | Prec. 2% | Prec. 10% | Prec. 2%, failed-only negatives | Prec. 10%, failed-only |
|---|---|---|---|---|---|---|---|
| LR-A | 0.043 [0.026, 0.066] | 0.031 [0.026, 0.036] | 0.078 [0.068, 0.090] | 0.036 [0.023, 0.063] | 0.167 [0.115, 0.267] | 0.049 [0.040, 0.060] | 0.219 [0.186, 0.258] |
| LR-B | 0.224 [0.186, 0.265] | 0.047 [0.042, 0.053] | 0.191 [0.176, 0.206] | 0.017 [0.014, 0.021] | 0.086 [0.073, 0.104] | 0.076 [0.067, 0.088] | 0.310 [0.280, 0.344] |
| LR-C | 0.224 [0.186, 0.265] | 0.056 [0.050, 0.062] | 0.225 [0.208, 0.242] | 0.020 [0.017, 0.025] | 0.100 [0.085, 0.120] | 0.076 [0.068, 0.086] | 0.310 [0.283, 0.340] |
| LR-C-drop | 0.222 [0.184, 0.263] | 0.039 [0.034, 0.045] | 0.120 [0.108, 0.134] | 0.011 [0.009, 0.014] | 0.057 [0.047, 0.070] | 0.059 [0.050, 0.070] | 0.254 [0.223, 0.290] |
| Markov | 1.000 [1.000, 1.000] (CP 0.991-1.000) | 1.000 [1.000, 1.000] (CP 0.999-1.000) | 1.000 [1.000, 1.000] (CP 0.998-1.000) | 0.020 [0.020, 0.020] | 0.100 [0.100, 0.100] | 0.020 [0.020, 0.020] | 0.100 [0.100, 0.100] |
| IForest | 0.283 [0.240, 0.329] | 0.342 [0.329, 0.356] | 0.394 [0.374, 0.413] | 0.028 [0.024, 0.033] | 0.134 [0.117, 0.156] | 0.023 [0.022, 0.024] | 0.113 [0.107, 0.119] |
| TFIDF | 0.000 [0.000, 0.000] (CP 0.000-0.009) | 0.000 [0.000, 0.000] (CP 0.000-0.001) | 0.000 [0.000, 0.000] (CP 0.000-0.002) | undefined (no alert) | undefined (no alert) | undefined (no alert) | undefined (no alert) |
| GRU | 0.168 [0.115, 0.230] | 0.073 [0.049, 0.094] | 0.256 [0.213, 0.296] | 0.030 [0.024, 0.040] | 0.145 [0.119, 0.184] | 0.068 [0.056, 0.087] | 0.284 [0.245, 0.343] |
| GRU-drop0.2 | 0.127 [0.087, 0.168] | 0.024 [0.018, 0.030] | 0.059 [0.034, 0.084] | 0.009 [0.006, 0.014] | 0.048 [0.031, 0.071] | 0.047 [0.031, 0.072] | 0.212 [0.148, 0.296] |

The cluster bootstrap for LR-C (118 suite|user_task clusters; plan 3.5) [`M9.R4.schemes.cluster_suite_user_task.detectors.LR-C.t_r80`] gives: FPR benign 0.224 [0.141, 0.308]; failed flagged 0.056 [0.033, 0.084]; recall 0.225 [0.144, 0.316]; precision at 2% (pool-E mix) 0.020 [0.015, 0.027].

**WUT** (rows: recall_poolE 2294, FPR_poolE 392, FPR_benign 392, flag_failed_attack 4987)

| Detector | FPR benign (= pool E) | Unsuccessful attacks flagged | Recall | Prec. 2% | Prec. 10% | Prec. 2%, failed-only negatives | Prec. 10%, failed-only |
|---|---|---|---|---|---|---|---|
| LR-A | 0.262 [0.214, 0.316] | 0.262 [0.231, 0.287] | 0.709 [0.676, 0.735] | 0.052 [0.044, 0.063] | 0.232 [0.202, 0.267] | 0.052 [0.047, 0.060] | 0.232 [0.213, 0.257] |
| LR-B | 0.428 [0.378, 0.477] | 0.460 [0.432, 0.482] | 0.776 [0.756, 0.798] | 0.036 [0.032, 0.040] | 0.168 [0.152, 0.186] | 0.033 [0.032, 0.035] | 0.158 [0.151, 0.166] |
| LR-C | 0.216 [0.138, 0.281] | 0.279 [0.253, 0.305] | 0.720 [0.686, 0.744] | 0.065 [0.051, 0.093] | 0.274 [0.225, 0.360] | 0.050 [0.046, 0.054] | 0.223 [0.209, 0.236] |
| LR-C-drop | 0.268 [0.217, 0.327] | 0.288 [0.271, 0.309] | 0.720 [0.687, 0.748] | 0.052 [0.044, 0.062] | 0.230 [0.200, 0.266] | 0.048 [0.045, 0.052] | 0.217 [0.205, 0.231] |
| Markov | 0.721 [0.661, 0.788] | 0.770 [0.735, 0.831] | 0.857 [0.836, 0.877] | 0.024 [0.022, 0.026] | 0.117 [0.109, 0.125] | 0.022 [0.021, 0.023] | 0.110 [0.104, 0.115] |
| IForest | 0.490 [0.434, 0.546] | 0.573 [0.549, 0.615] | 0.779 [0.741, 0.809] | 0.031 [0.028, 0.036] | 0.150 [0.136, 0.168] | 0.027 [0.026, 0.029] | 0.131 [0.125, 0.138] |
| TFIDF | 0.000 [0.000, 0.000] (CP 0.000-0.009) | 0.308 [0.286, 0.331] | 0.780 [0.758, 0.801] | 1.000 [1.000, 1.000] (>= 0.623) | 1.000 [1.000, 1.000] (>= 0.900) | 0.049 [0.046, 0.053] | 0.220 [0.208, 0.232] |
| GRU | 0.178 [0.115, 0.240] | 0.227 [0.173, 0.276] | 0.761 [0.719, 0.794] | 0.082 [0.059, 0.120] | 0.327 [0.256, 0.425] | 0.065 [0.055, 0.084] | 0.274 [0.239, 0.333] |

The cluster bootstrap for LR-C (118 suite|user_task clusters; plan 3.5) [`M9.WUT.schemes.cluster_suite_user_task.detectors.LR-C.t_r80`] gives: FPR benign 0.216 [0.126, 0.310]; failed flagged 0.279 [0.192, 0.376]; recall 0.720 [0.642, 0.784]; precision at 2% (pool-E mix) 0.065 [0.046, 0.102].

**AUROC by subgroup.** Each AUROC ranks the positives of one subgroup against the pool-E negatives, or all positives against one negative class [`M9.<S>.schemes.traj_strat.detectors.<det>.AUROC_subgroups`; paired LR-C minus LR-B in `M9.<S>.schemes.traj_strat.paired_AUROC_subgroups`].

| Setting | Detector | full | partial | delayed | vs benign | vs hard neg. | vs failed |
|---|---|---|---|---|---|---|---|
| R1 | LR-B | 0.806 [0.784, 0.828] | 0.872 [0.850, 0.896] | 0.842 [0.811, 0.869] | 0.881 [0.864, 0.898] | 0.695 [0.657, 0.731] | 0.721 [0.683, 0.756] |
| R1 | LR-C | 0.898 [0.881, 0.913] | 0.858 [0.833, 0.882] | 0.849 [0.819, 0.878] | 0.917 [0.903, 0.931] | 0.777 [0.744, 0.810] | 0.864 [0.838, 0.889] |
| R1 | GRU | 0.948 [0.933, 0.960] | 0.927 [0.909, 0.946] | 0.908 [0.873, 0.938] | 0.952 [0.941, 0.962] | 0.893 [0.868, 0.917] | 0.937 [0.917, 0.955] |
| R1 | LR-C minus LR-B | +0.092 [+0.075, +0.109] | -0.015 [-0.040, +0.009] | +0.007 [-0.022, +0.035] | +0.036 [+0.021, +0.051] | +0.082 [+0.054, +0.111] | +0.144 [+0.114, +0.175] |
| R3_medical | LR-B | 0.766 [0.743, 0.788] | 0.919 [0.900, 0.936] | 0.796 [0.763, 0.827] | 0.856 [0.838, 0.873] | 0.679 [0.644, 0.716] | 0.637 [0.598, 0.678] |
| R3_medical | LR-C | 0.682 [0.656, 0.706] | 0.884 [0.860, 0.906] | 0.777 [0.741, 0.810] | 0.787 [0.766, 0.808] | 0.633 [0.593, 0.672] | 0.671 [0.633, 0.711] |
| R3_medical | GRU | 0.578 [0.548, 0.607] | 0.755 [0.681, 0.835] | 0.696 [0.643, 0.748] | 0.652 [0.614, 0.685] | 0.602 [0.558, 0.648] | 0.716 [0.658, 0.768] |
| R3_medical | LR-C minus LR-B | -0.085 [-0.097, -0.072] | -0.035 [-0.046, -0.025] | -0.019 [-0.035, -0.003] | -0.069 [-0.080, -0.058] | -0.047 [-0.065, -0.028] | +0.034 [+0.020, +0.049] |
| R3_banking | LR-B | 0.743 [0.708, 0.775] | 0.941 [0.926, 0.954] | 0.866 [0.839, 0.892] | 0.906 [0.890, 0.922] | 0.715 [0.681, 0.747] | 0.641 [0.605, 0.677] |
| R3_banking | LR-C | 0.678 [0.641, 0.712] | 0.924 [0.908, 0.940] | 0.880 [0.854, 0.903] | 0.880 [0.863, 0.895] | 0.692 [0.655, 0.725] | 0.598 [0.559, 0.638] |
| R3_banking | GRU | 0.582 [0.511, 0.657] | 0.828 [0.772, 0.867] | 0.801 [0.680, 0.883] | 0.761 [0.728, 0.797] | 0.671 [0.609, 0.727] | 0.653 [0.606, 0.703] |
| R3_banking | LR-C minus LR-B | -0.065 [-0.087, -0.043] | -0.017 [-0.028, -0.005] | +0.014 [-0.004, +0.032] | -0.027 [-0.038, -0.016] | -0.024 [-0.050, +0.002] | -0.044 [-0.068, -0.020] |
| R3_web | LR-B | 0.900 [0.885, 0.914] | 0.905 [0.885, 0.922] | 0.866 [0.835, 0.893] | 0.942 [0.930, 0.952] | 0.776 [0.745, 0.805] | 0.794 [0.765, 0.822] |
| R3_web | LR-C | 0.856 [0.838, 0.872] | 0.874 [0.851, 0.896] | 0.870 [0.841, 0.894] | 0.915 [0.902, 0.927] | 0.722 [0.689, 0.754] | 0.771 [0.737, 0.804] |
| R3_web | GRU | 0.709 [0.671, 0.755] | 0.746 [0.703, 0.783] | 0.756 [0.700, 0.824] | 0.757 [0.716, 0.800] | 0.643 [0.596, 0.691] | 0.782 [0.725, 0.835] |
| R3_web | LR-C minus LR-B | -0.045 [-0.058, -0.032] | -0.031 [-0.046, -0.017] | +0.003 [-0.015, +0.022] | -0.027 [-0.037, -0.017] | -0.053 [-0.085, -0.024] | -0.023 [-0.045, -0.004] |
| R3_email | LR-B | 0.782 [0.759, 0.804] | 0.824 [0.796, 0.850] | 0.778 [0.743, 0.810] | 0.819 [0.798, 0.839] | 0.722 [0.689, 0.754] | 0.764 [0.727, 0.801] |
| R3_email | LR-C | 0.741 [0.718, 0.765] | 0.740 [0.709, 0.770] | 0.731 [0.694, 0.766] | 0.773 [0.751, 0.796] | 0.650 [0.613, 0.688] | 0.770 [0.733, 0.806] |
| R3_email | GRU | 0.631 [0.604, 0.658] | 0.755 [0.654, 0.896] | 0.683 [0.613, 0.782] | 0.682 [0.636, 0.742] | 0.642 [0.579, 0.714] | 0.820 [0.747, 0.913] |
| R3_email | LR-C minus LR-B | -0.041 [-0.051, -0.031] | -0.085 [-0.100, -0.068] | -0.048 [-0.067, -0.030] | -0.046 [-0.056, -0.036] | -0.072 [-0.091, -0.054] | +0.006 [-0.009, +0.022] |
| R3_coding | LR-B | 0.853 [0.833, 0.871] | 0.875 [0.852, 0.898] | 0.810 [0.779, 0.838] | 0.889 [0.873, 0.904] | 0.753 [0.720, 0.784] | 0.747 [0.714, 0.778] |
| R3_coding | LR-C | 0.806 [0.784, 0.826] | 0.836 [0.807, 0.863] | 0.811 [0.782, 0.839] | 0.855 [0.837, 0.873] | 0.706 [0.670, 0.740] | 0.743 [0.709, 0.775] |
| R3_coding | GRU | 0.687 [0.654, 0.723] | 0.703 [0.630, 0.793] | 0.710 [0.653, 0.777] | 0.715 [0.668, 0.771] | 0.641 [0.591, 0.692] | 0.805 [0.743, 0.879] |
| R3_coding | LR-C minus LR-B | -0.047 [-0.060, -0.036] | -0.040 [-0.055, -0.025] | +0.001 [-0.019, +0.021] | -0.034 [-0.045, -0.024] | -0.047 [-0.073, -0.024] | -0.004 [-0.025, +0.019] |

AgentDojo, AUROC of compromised against unsuccessful attacks [`M9.<R4|WUT>.schemes.traj_strat.detectors.<det>.AUROC_subgroups.poolE_vs_failed`]: R4: LR-A 0.587 [0.576, 0.598], LR-B 0.638 [0.625, 0.652], LR-C 0.639 [0.626, 0.653], GRU 0.658 [0.644, 0.673], TFIDF 0.634 [0.620, 0.648]; WUT: LR-A 0.780 [0.758, 0.797], LR-B 0.716 [0.693, 0.733], LR-C 0.775 [0.757, 0.800], GRU 0.827 [0.796, 0.861], TFIDF 0.808 [0.796, 0.820].

**Plain reading.** In R1, LR-C at theta_r80 flags 0.118 of benign traces, 0.387 of hard negatives and 0.218 of failed attacks. Its recall is 0.830 on full, 0.727 on partial and 0.642 on delayed hijacks. Its precision at 2% therefore ranges from 0.117 with benign-only negatives, through 0.076 with the pool-E mix, to 0.039 with hard negatives only. At fixed prevalence the negative mix changes precision by a factor of 3.0, so every precision figure must state its mix. In R1 every trained detector (LR-A, LR-B, LR-C, Markov, GRU) flags hard negatives more often than benign traces. LR-B ranks partial hijacks slightly better than LR-C (AUROC 0.872 vs 0.858; LR-C minus LR-B -0.015 [-0.040, +0.009], not detected). LR-C's overall advantage comes from full hijacks (+0.092 [+0.075, +0.109]), from hard negatives (+0.082 [+0.054, +0.111]) and from failed attacks (+0.144 [+0.114, +0.175]). Under domain shift, LR-B's full-hijack recall at theta_r80 is 0.468 (medical), 0.659 (banking), 0.840 (web), 0.820 (email), 0.834 (coding). In R4 transfer, LR-C flags 0.224 of AgentDojo benign traces, about as often as it flags compromised ones (0.225), and 0.056 of unsuccessful attacks. Within AgentDojo, TFIDF flags no benign trace but 0.308 of unsuccessful attacks. Its AUROC is 0.808 against unsuccessful attacks and 0.996 against benign traces. So it largely detects injected content rather than compromise: if unsuccessful attacks count as negatives, its precision at 2% is 0.049. The breakdown shows where alerts come from. It does not say which negative mix a deployment would have, and the AgentDrift hard negatives are synthetic.

## m4. Held-out GRU threshold, matched recall, and theta = 0.5

**Question.** How does the GRU perform at its validation threshold theta_val80? What is each detector's FPR at a fixed test TPR of 0.8? What does theta = 0.5 give?

R1 GRU on pool E [`m4_gru_t_val80.table.R1.GRU.<op>`; per-seed values are in `point_by_rep`]:

| Threshold | Recall (per seed) | FPR (per seed) | Precision 2%, pool-E mix (per seed) | Precision 10% |
|---|---|---|---|---|
| t_r80 | 0.682 [0.645, 0.716] (0.674, 0.693, 0.680) | 0.045 [0.028, 0.062] (0.046, 0.050, 0.039) | 0.238 [0.183, 0.328] (0.230, 0.222, 0.263) | 0.629 [0.549, 0.727] |
| t_val80 | 0.775 [0.745, 0.804] (0.771, 0.773, 0.781) | 0.073 [0.055, 0.092] (0.068, 0.074, 0.076) | 0.179 [0.147, 0.225] (0.188, 0.176, 0.173) | 0.542 [0.483, 0.613] |

**FPR at a fixed test TPR of 0.8.** The threshold is placed using test labels, so this is a matched-recall comparison, not an operating point [`M8_R1.detectors.<det>.FPR_at_test_TPR80.FPR_at_TPR80`]: LR-A 0.411 [0.364, 0.472], LR-B 0.288 [0.250, 0.335], LR-C 0.210 [0.175, 0.244], Markov 0.472 [0.417, 0.524], IForest 0.634 [0.594, 0.676], GRU 0.082 [0.061, 0.108], TFIDF 0.000 [0.000, 0.000]. For the GRU the per-seed values are 0.078, 0.087, 0.081.

**theta = 0.5**, recall / FPR on pool E [`m4_theta05.table.<S>.<det>.recall_poolE` / `.FPR_poolE`; failed-attack flag rates in `.flag_failed_attack`]:

| Setting | LR-A | LR-B | LR-C | GRU | TFIDF |
|---|---|---|---|---|---|
| R1 | 0.658 [0.628, 0.687] / 0.230 [0.202, 0.259] | 0.758 [0.730, 0.787] / 0.247 [0.219, 0.274] | 0.786 [0.759, 0.815] / 0.198 [0.172, 0.224] | 0.856 [0.819, 0.893] / 0.119 [0.084, 0.159] | 0.990 [0.983, 0.996] / 0.008 [0.002, 0.016] |
| R3_medical | 0.000 [0.000, 0.000] (CP 0.000-0.004) / 0.000 [0.000, 0.000] (CP 0.000-0.004) | 0.527 [0.497, 0.557] / 0.140 [0.120, 0.161] | 0.864 [0.842, 0.884] / 0.538 [0.507, 0.569] | 0.994 [0.988, 0.999] / 0.817 [0.788, 0.846] | 0.931 [0.917, 0.946] / 0.015 [0.007, 0.024] |
| R3_banking | 0.000 [0.000, 0.000] (CP 0.000-0.006) / 0.000 [0.000, 0.000] (CP 0.000-0.004) | 0.805 [0.776, 0.834] / 0.249 [0.226, 0.273] | 0.971 [0.957, 0.982] / 0.680 [0.651, 0.710] | 0.686 [0.070, 0.997] / 0.531 [0.003, 0.833] | 0.559 [0.529, 0.591] / 0.000 [0.000, 0.000] (CP 0.000-0.004) |
| R3_web | 0.000 [0.000, 0.000] (CP 0.000-0.004) / 0.000 [0.000, 0.000] (CP 0.000-0.004) | 0.762 [0.736, 0.788] / 0.144 [0.124, 0.166] | 0.990 [0.984, 0.996] / 0.706 [0.678, 0.735] | 0.992 [0.974, 1.000] / 0.763 [0.681, 0.831] | 0.945 [0.931, 0.958] / 0.145 [0.124, 0.168] |
| R3_email | 0.000 [0.000, 0.000] (CP 0.000-0.004) / 0.000 [0.000, 0.000] (CP 0.000-0.004) | 0.800 [0.775, 0.823] / 0.384 [0.353, 0.416] | 0.965 [0.953, 0.975] / 0.733 [0.705, 0.760] | 0.991 [0.984, 0.996] / 0.832 [0.804, 0.861] | 0.951 [0.938, 0.963] / 0.052 [0.040, 0.067] |
| R3_coding | 0.000 [0.000, 0.000] (CP 0.000-0.004) / 0.000 [0.000, 0.000] (CP 0.000-0.004) | 0.807 [0.783, 0.831] / 0.255 [0.227, 0.282] | 0.976 [0.968, 0.985] / 0.677 [0.649, 0.707] | 0.904 [0.705, 1.000] / 0.655 [0.370, 0.818] | 0.832 [0.814, 0.850] / 0.007 [0.002, 0.014] |
| R4 | 0.067 [0.058, 0.078] / 0.008 [0.000, 0.018] | 0.178 [0.163, 0.194] / 0.224 [0.186, 0.265] | 0.226 [0.209, 0.243] / 0.224 [0.186, 0.265] | 0.301 [0.260, 0.337] / 0.212 [0.161, 0.263] | 0.175 [0.160, 0.190] / 0.217 [0.179, 0.258] |

**Plain reading.** With its validation threshold, the GRU's recall rises from 0.682 to 0.775 and its FPR from 0.045 to 0.073. Precision at 2% falls from 0.238 to 0.179. The in-sample threshold was optimistic about recall and flattered the GRU's precision. At a matched test recall of 0.8, the GRU's FPR is 0.082 against 0.210 for LR-C, so its advantage in R1 does not depend on the threshold. At theta = 0.5 the GRU in R1 has FPR 0.119 and recall 0.856. Under domain shift the default cut fails like theta_r80: FPR 0.538-0.733 for LR-C and 0.531-0.832 for the GRU. LR-A flags nothing in any R3 fold. In R4, recall at 0.5 is at most 0.301 for every detector. The matched-recall numbers use test labels and are not achievable operating points.

## m10. FPR budgets set on training negatives; Eq. (6) at 0.1% and 0.01%

**Question.** Suppose the threshold flags at most 1% or 5% of training negatives. What TPR and FPR does it then give on the R1 test pool E? What precision does LR-C have at 0.1% and 0.01% prevalence?

The primary definition of training negatives is benign plus hard-negative training rows [`R1_operating_points.m10_fpr_budgets.detectors.<det>.benign_hard.<fpr01|fpr05>`]. The sensitivity uses all label-0 training rows, failed attacks included [`...all_label0...`].

| Detector | 1%: train FPR | 1%: test TPR | 1%: test FPR | 5%: train FPR | 5%: test TPR | 5%: test FPR | all label-0, 1%: TPR / FPR | all label-0, 5%: TPR / FPR |
|---|---|---|---|---|---|---|---|---|
| LR-A | 0.0099 | 0.173 [0.147, 0.199] | 0.013 [0.006, 0.022] | 0.0499 | 0.400 [0.367, 0.433] | 0.065 [0.048, 0.082] | 0.181 / 0.015 | 0.402 / 0.065 |
| LR-B | 0.0099 | 0.194 [0.166, 0.221] | 0.011 [0.005, 0.019] | 0.0499 | 0.373 [0.341, 0.407] | 0.067 [0.051, 0.085] | 0.140 / 0.006 | 0.326 / 0.047 |
| LR-C | 0.0099 | 0.276 [0.244, 0.307] | 0.017 [0.008, 0.027] | 0.0499 | 0.529 [0.495, 0.563] | 0.065 [0.048, 0.082] | 0.270 / 0.017 | 0.520 / 0.059 |
| Markov | 0.0099 | 0.054 [0.039, 0.070] | 0.017 [0.008, 0.027] | 0.0499 | 0.243 [0.214, 0.272] | 0.069 [0.052, 0.087] | 0.026 / 0.013 | 0.184 / 0.048 |
| IForest | 0.0099 | 0.006 [0.001, 0.012] | 0.013 [0.006, 0.022] | 0.0499 | 0.046 [0.032, 0.060] | 0.052 [0.038, 0.068] | 0.006 / 0.010 | 0.037 / 0.045 |
| GRU | 0.0098-0.0098 | 0.770 [0.717, 0.809] | 0.070 [0.050, 0.092] | 0.0498-0.0499 | 0.889 [0.866, 0.910] | 0.156 [0.128, 0.182] | 0.774 / 0.071 | 0.890 / 0.156 |
| TFIDF | 0.0099 | 0.998 [0.994, 1.000] | 0.033 [0.021, 0.045] | 0.0499 | 0.999 [0.996, 1.000] | 0.081 [0.063, 0.102] | 0.998 / 0.034 | 0.999 / 0.099 |

**Eq. (6) at very low prevalence.**

- LR-C at theta_r80: precision 0.0040 [0.0035, 0.0047] at 0.1% and 0.00040 [0.00035, 0.00047] at 0.01%, with 192 [165, 218] and 192 [165, 217] alerts per 1,000 [`R1_operating_points.detectors.LR-C.t_r80.precision_pi0.1pct` / `precision_pi0.01pct` / `alerts_per_1000_*`].
- LR-C at the 1% budget: 0.0160 [0.0101, 0.0311] at 0.1% and 0.00162 [0.00102, 0.00320] at 0.01% [`...m10_fpr_budgets.detectors.LR-C.benign_hard.fpr01.precision_pi0.1pct`].
- GRU at theta_r80, at 0.1%: 0.0151 [0.0110, 0.0234].
- TFIDF at theta_r80 flags no pool-E negative. With the CP upper FPR bound, its precision at 0.1% is at least 0.142 [`R1_operating_points.detectors.TFIDF.t_r80.precision_pi0.1pct.lower_bound_if_FPR_at_CP_upper`].

**Plain reading.** With thresholds that flag at most 1% (or 5%) of training negatives, LR-C catches 0.276 (0.529) of R1 attacks, at a test FPR of 0.017 (0.065). The test FPR exceeds the budget because training scores are in-sample. The overshoot is largest for the flexible models: at the 1% budget the GRU's test FPR is 0.070 and TFIDF's 0.033. Counting all label-0 training rows as negatives gives LR-C 0.270 / 0.017 and 0.520 / 0.059, which are the review's figures. At 0.1% and 0.01% prevalence, LR-C at theta_r80 has precision 0.0040 and 0.00040. The 1% budget only lifts this to 0.016 at 0.1%. The best content-blind value at 0.1% over all thresholds examined is 0.0225 (LR-B, all_label0.fpr01). These are extrapolations by Eq. (6) and assume that TPR and FPR stay fixed.

## m11. Label-free threshold on the held-out domain (plan 3.7)

**Question.** On each held-out R3 domain, set the threshold at the quantile of 200 unlabeled held-out traces that reproduces the R1 alert rate. What FPR and recall does this give on the remaining held-out traces?

Two readings of 'the R1 alert rate' are reported [`m11_label_free.R1_alert_rate_targets`]:

- (a) Pre-specified: the R1 alert rate at native prevalence on pool E, with a native-prevalence unlabeled sample. Targets: LR-C 0.482, LR-B 0.540, GRU 0.361, 0.373, 0.361.
- (b) Post hoc (P9): the R1 alert rate at 2% prevalence, with unlabeled traffic at 2% (4 attacked and 196 negative traces per draw). Targets: LR-C 0.203, LR-B 0.294, GRU 0.059, 0.063, 0.052.

Values are mean [min, max] over 5 draws; for the GRU, over 3 seeds x 5 draws [`m11_label_free.folds.<domain>.detectors.<det>.<primary_poolE|deployment_prev02>.summary_over_draws`]. theta_r80 in the same fold is shown for comparison [`...theta_r80_same_fold_full_poolE`].

| Domain | Detector | (a) FPR | (a) Recall | (b) FPR | (b) Recall | (b) Prec. 2% | theta_r80 FPR / recall |
|---|---|---|---|---|---|---|---|
| medical | LR-C | 0.312 [0.257, 0.342] | 0.631 [0.559, 0.689] | 0.180 [0.130, 0.200] | 0.489 [0.439, 0.512] | 0.053 [0.050, 0.065] | 0.503 / 0.827 |
| medical | LR-B | 0.303 [0.243, 0.348] | 0.742 [0.677, 0.776] | 0.271 [0.234, 0.299] | 0.731 [0.697, 0.755] | 0.052 [0.049, 0.057] | 0.150 / 0.560 |
| medical | GRU | 0.314 [0.269, 0.372] | 0.432 [0.367, 0.496] | 0.057 [0.026, 0.076] | 0.123 [0.092, 0.160] | 0.044 [0.035, 0.068] | 0.717 / 0.938 |
| banking | LR-C | 0.284 [0.248, 0.347] | 0.770 [0.737, 0.827] | 0.179 [0.153, 0.217] | 0.665 [0.640, 0.698] | 0.071 [0.062, 0.079] | 0.660 / 0.966 |
| banking | LR-B | 0.303 [0.271, 0.334] | 0.846 [0.817, 0.871] | 0.287 [0.235, 0.340] | 0.834 [0.801, 0.863] | 0.057 [0.049, 0.065] | 0.278 / 0.829 |
| banking | GRU | 0.243 [0.148, 0.300] | 0.553 [0.445, 0.618] | 0.050 [0.020, 0.092] | 0.273 [0.204, 0.352] | 0.107 [0.072, 0.170] | 0.438 / 0.622 |
| web | LR-C | 0.220 [0.182, 0.245] | 0.770 [0.744, 0.796] | 0.203 [0.173, 0.246] | 0.745 [0.714, 0.783] | 0.070 [0.061, 0.078] | 0.707 / 0.990 |
| web | LR-B | 0.221 [0.143, 0.264] | 0.846 [0.772, 0.877] | 0.276 [0.235, 0.323] | 0.889 [0.870, 0.912] | 0.062 [0.054, 0.070] | 0.197 / 0.825 |
| web | GRU | 0.250 [0.159, 0.314] | 0.495 [0.390, 0.613] | 0.077 [0.053, 0.143] | 0.250 [0.191, 0.340] | 0.065 [0.042, 0.094] | 0.623 / 0.917 |
| email | LR-C | 0.290 [0.226, 0.343] | 0.609 [0.514, 0.670] | 0.221 [0.172, 0.267] | 0.505 [0.441, 0.557] | 0.045 [0.041, 0.050] | 0.722 / 0.962 |
| email | LR-B | 0.295 [0.223, 0.356] | 0.721 [0.630, 0.781] | 0.309 [0.279, 0.376] | 0.738 [0.714, 0.769] | 0.047 [0.040, 0.050] | 0.418 / 0.833 |
| email | GRU | 0.257 [0.194, 0.331] | 0.440 [0.391, 0.527] | 0.065 [0.047, 0.096] | 0.194 [0.152, 0.267] | 0.058 [0.044, 0.069] | 0.764 / 0.973 |
| coding | LR-C | 0.217 [0.164, 0.277] | 0.666 [0.602, 0.728] | 0.192 [0.160, 0.241] | 0.634 [0.591, 0.677] | 0.064 [0.054, 0.070] | 0.672 / 0.974 |
| coding | LR-B | 0.244 [0.186, 0.323] | 0.785 [0.725, 0.849] | 0.293 [0.237, 0.357] | 0.836 [0.792, 0.869] | 0.056 [0.047, 0.064] | 0.283 / 0.835 |
| coding | GRU | 0.245 [0.190, 0.345] | 0.444 [0.380, 0.516] | 0.059 [0.035, 0.087] | 0.180 [0.135, 0.244] | 0.060 [0.051, 0.076] | 0.486 / 0.720 |

Sensitivity for reading (a), with failed attacks included in both the target and the draws [`...sensitivity_all_rows.summary_over_draws.FPR.mean`]. Mean FPR:

- medical: LR-C 0.293, LR-B 0.307, GRU 0.291.
- banking: LR-C 0.238, LR-B 0.305, GRU 0.209.
- web: LR-C 0.192, LR-B 0.220, GRU 0.245.
- email: LR-C 0.304, LR-B 0.326, GRU 0.258.
- coding: LR-C 0.210, LR-B 0.243, GRU 0.241.

**Plain reading.** Setting the threshold without labels lowers LR-C's held-out FPR from 0.503-0.722 at theta_r80. Under reading (a) the mean FPR across folds is 0.217-0.312, with recall 0.609-0.770. Under reading (b) it is 0.179-0.221, with recall 0.489-0.745, close to the review's FPR 0.191-0.209 and recall 0.483-0.728 (see the reproduction table). For the GRU, reading (b) gives FPR 0.050-0.077 but recall only 0.123-0.273. LR-B already had moderate FPR at theta_r80, and the remedy makes it worse in some folds (reading (b): 0.271-0.309). The remedy restores an alert budget; it does not restore ranking quality. Precision at 2% stays at 0.045-0.071 for LR-C. It also works only if the unlabeled sample has the prevalence that the target alert rate assumes. Reading (a) matches LR-C's native R1 alert rate (0.482) on held-out pools that are about half attacks. Reading (b) matches its 2%-prevalence alert rate (0.203) on traffic at 2% prevalence.

## m9. Finite-sample chance AP and floating-point ties

**Question.** What is the expected AP of a random scorer at the v0.3 subsample sizes? Does rounding scores to 12 decimals before ranking change any value?

Expected AP of a random scorer [`m9_chance_AP.values.<n+>/<n->`], as exact value (Monte Carlo mean of 20,000 random rankings; prevalence):

| n+ / n- | exact | Monte Carlo mean | prevalence |
|---|---|---|---|
| 8/825 | 0.0171 | 0.0171 | 0.0096 |
| 17/825 | 0.0275 | 0.0276 | 0.0202 |
| 43/825 | 0.0565 | 0.0564 | 0.0495 |
| 92/825 | 0.1066 | 0.1067 | 0.1003 |
| 44/392 | 0.1126 | 0.1126 | 0.1009 |
| 21/392 | 0.0638 | 0.0638 | 0.0508 |

**Rounding check** [`m9_rounding_check.<S>.<det>`], over 123 (setting, detector) pairs in R1, R2, R3, R4 and WUT:

- Rounding changes the ordering or tie structure of pool-E scores in 9 pairs, all in AgentDojo settings: R4 LR-B, R4 LR-C, R4 LR-B-minus, R4 LR-C-drop, R4 GRU, R4 GRU-drop0.2, R4 GRU-drop0.1, WUT LR-C, WUT LR-C-drop.
- In R4 the cause is saturated scores. Rows with nonzero scores below 5e-13 become tied at 0, e.g. 443 rows for LR-C. R4 LR-C AUROC moves from 0.4733 (raw, equal to the cache validation) to 0.4697 (rounded). R4 GRU moves from 0.5134, 0.5129, 0.5136 to 0.5109, 0.5117, 0.5107.
- No AgentDrift value changes: R1, R2 and every R3 fold have identical order and ties.
- Predictions of the new thresholds that differ between raw and rounded scores: 0 for the FPR budgets and 0 for the label-free thresholds.
- R3 LR-A and Markov AUROC are exactly 0.5000 in the cache, e.g. R3_email LR-A 0.5000 [`M9.R3_email.schemes.traj_strat.detectors.LR-A.AUROC_subgroups.poolE_vs_poolE_neg`].

**Plain reading.** A random scorer's expected AP is 0.0171 at 8 positives and 0.0275 at 17, not the prevalence (0.0096 and 0.0202). The chance lines of the v0.3 subsample figures should use these values. The Isolation Forest's v0.3 subsample APs (0.017 and 0.030) are at this chance level. For the weighted estimator the chance level is close to pi (M8). The caches already store tie-exact scores, so rounding is only needed for code that recomputes scores with float noise (v0.3 R3 LR-A 0.499). Applied to AgentDojo scores, 12-decimal rounding creates ties between distinct saturated probabilities. It changes point AUROC or wAUPRC by up to 0.0056 (WUT LR-C). For AgentDojo we therefore recommend the unrounded values, which are reported alongside.

## Reproduction of the review's figures

Each row sets a value quoted in REVIEW_REPORT.md next to ours [`review_reference_values.rows`; our key in `ours_key`].

| Comment | Quantity | Review | Ours |
|---|---|---|---|
| M8 | LR-C wAUPRC 1% | 0.1560 | 0.1563 |
| M8 | LR-C wAUPRC 2% | 0.2320 | 0.2322 |
| M8 | LR-C wAUPRC 5% | 0.3800 | 0.3804 |
| M8 | LR-C wAUPRC 10% | 0.5230 | 0.5231 |
| M8 | LR-C wAUPRC 2% lower | 0.1850 | 0.1832 |
| M8 | LR-C wAUPRC 2% upper | 0.3250 | 0.3301 |
| M8 | LR-C wAUPRC 2% bootstrap mean | 0.2450 | 0.2465 |
| M8 | LR-C minus Markov wAUPRC 1% | 0.1320 | 0.1323 |
| M8 | LR-C minus Markov wAUPRC 2% | 0.1850 | 0.1848 |
| M8 | LR-C minus LR-B wAUPRC 2% | 0.0820 | 0.0805 |
| M8 | LR-C minus LR-B wAUPRC 2% lower | 0.0030 | 0.0026 |
| M8 | GRU minus LR-C AUROC native | 0.0560 | 0.0568 |
| M8 | GRU minus LR-C AUROC on 2% replicates, lower | -0.0210 | -0.0196 |
| M8 | LR-C minus Markov AUROC on 1% replicates, lower | -0.0350 | -0.0470 |
| M9 | LR-C FPR benign | 0.1180 | 0.1183 |
| M9 | LR-C FPR hard negative | 0.3870 | 0.3867 |
| M9 | LR-C failed flagged | 0.2180 | 0.2178 |
| M9 | LR-C precision 2% benign-only | 0.1170 | 0.1174 |
| M9 | LR-C precision 2% pool-E mix | 0.0770 | 0.0760 |
| M9 | LR-C precision 2% hard-only | 0.0390 | 0.0391 |
| M9 | LR-C recall full | 0.8300 | 0.8305 |
| M9 | LR-C recall partial | 0.7270 | 0.7273 |
| M9 | LR-C recall delayed | 0.6420 | 0.6418 |
| M9 | LR-A recall partial | 0.5020 | 0.5022 |
| M9 | LR-B partial-hijack AUROC | 0.8720 | 0.8724 |
| M9 | LR-C partial-hijack AUROC | 0.8580 | 0.8578 |
| M9 | R3 medical LR-B full recall | 0.4700 | 0.4676 |
| M9 | R3 banking LR-B full recall | 0.6600 | 0.6590 |
| m4 | GRU t_val80 precision 2%, seed 0 | 0.1880 | 0.1883 |
| m4 | GRU FPR at test TPR 0.8, seed 1 | 0.0870 | 0.0873 |
| m4 | LR-C FPR at test TPR 0.8 | 0.2100 | 0.2097 |
| m4 | GRU R1 theta=0.5 FPR | 0.1190 | 0.1192 |
| m4 | GRU R1 theta=0.5 recall | 0.8560 | 0.8556 |
| m9 | chance AP 8/825 | 0.0170 | 0.0171 |
| m9 | chance AP 17/825 | 0.0275 | 0.0275 |
| m9 | chance AP 44/392 | 0.1127 | 0.1126 |
| m10 | LR-C precision 0.1% | 0.0040 | 0.0040 |
| m10 | LR-C precision 0.01% | 0.0004 | 0.0004 |
| m10 | LR-C 1% budget TPR (all label-0 negatives) | 0.2700 | 0.2696 |
| m10 | LR-C 1% budget FPR (all label-0 negatives) | 0.0170 | 0.0170 |
| m10 | LR-C 5% budget TPR (all label-0 negatives) | 0.5200 | 0.5199 |
| m10 | LR-C 5% budget FPR (all label-0 negatives) | 0.0590 | 0.0594 |
| m17.5 | LR-C minus Markov subsampled AP 2% | 0.1980 | 0.1961 |
| m11 | LR-C label-free FPR range over folds (reading b, fold means) | 0.191-0.209 | 0.179-0.221 |
| m11 | LR-C label-free recall range over folds (reading b, fold means) | 0.483-0.728 | 0.489-0.745 |

## Deviations and interpretation notes

- P1 (interpretation, AgentDojo and R2 intervals). Plan 3.1 asks for a stratified full-pool bootstrap for AUPRC at controlled prevalence; plan 3.5 asks for a (suite, user task) cluster bootstrap for the AgentDojo UT analyses and a task cluster bootstrap for R2. Both are reported: 'traj_strat' (plan 3.1, the scheme named in the task) and 'cluster_*' (plan 3.5). R1 and R3 use the trajectory-level bootstrap; R1 also has a world-cluster sensitivity.
- P2 (interpretation, m10 training negatives). The plan does not say which training rows are 'training negatives'. Primary: benign + hard_negative training rows (the classes of the test pool-E negatives); sensitivity: all label-0 training rows (incl. failed attacks). Threshold rule: flag s > t, t = (n-k)-th smallest training-negative score, k = floor(budget*n). The scores are in-sample for the supervised detectors (as for theta_r80).
- P3 (new fits). The cache stores no training scores for the linear detectors. The R1 linear detectors were refitted with the unmodified cache code (build_linear_cache.py functions, imported) only to recover in-sample training scores; the refit reproduces the cached test scores (max |diff| reported in meta.R1_train_score_refit) and t_r80. GRU training scores are reconstructed in float64 from the cached logit decomposition of the 90% fit partition; the reconstructed t_r80 differs from the cached float32 value by < 1e-7.
- P4 (interpretation, m11). The 'R1 alert rate' is the detector's alert rate at theta_r80 on the R1 test pool E (GRU: per seed). The unlabeled draws come from the held-out pool E; a sensitivity uses all rows (incl. failed attacks) both for the R1 alert rate and for the draws. LR-B and the GRU are added to LR-C as the task requests.
- P5 (M9 bootstrap strata). The M9 breakdown resamples each stratum category x hijack pattern (and the failed attacks) separately, so the class composition is fixed as in the test set; the M8 blocks stratify by class (pool E) as evalkit.
- P6 (point estimates). Point estimates are plug-in values on the full pool (mean over GRU seeds or CV repetitions); bootstrap means are reported next to them. For the secondary subsampling estimator the point is the replicate mean (v0.3).
- P7 (FPR at a fixed test TPR of 0.8, plan 3.8). This uses test labels to place the threshold (matched recall); it is a descriptive comparison, not a deployable operating point. The threshold is re-chosen inside each bootstrap replicate.
- P8 (precision under negative mixes). Precision uses the pooled pool-E recall of the setting with the FPR of each negative mix. For AgentDojo pool E has benign negatives only; 'failed_only' (unsuccessful attacks as negatives) is added as an extra mix in every setting.
- P9 (post hoc, m11). The pre-specified reading (P4: native-prevalence unlabeled sample, native R1 alert rate) does not reproduce the review's m11 figures (FPR 0.191-0.209). A second reading was added after seeing this: unlabeled traffic at 2% prevalence (4 attacked + 196 negative held-out traces per draw) with the R1 alert rate at 2% prevalence (0.02*TPR + 0.98*FPR at theta_r80, i.e. the alerts per 1,000 at 2% of Table IV / 1000). Both are reported ('primary_poolE', 'deployment_prev02'); the second is labelled post hoc.
- P10 (m9 rounding). As instructed, scores are rounded to 12 decimals before every ranking. This changes no value in AgentDrift settings, but in R4 and WUT it merges distinct saturated scores below 5e-13 into ties at 0 (e.g. R4 LR-C AUROC 0.4733 raw vs 0.4697 rounded; see m9_rounding_check). For AgentDojo every rank-based M8 quantity is therefore also given with the raw (cache tie-exact) scores in M8_AgentDojo_raw_scores (same replicates), and the M9 subgroup AUROCs in AUROC_subgroups_raw_scores; the raw values match the cache validation files. Thresholded rates are unaffected (they use the cached predictions).
- P11 (degenerate intervals). When a rate is exactly 0 or 1 in every repetition, the bootstrap interval collapses; an exact Clopper-Pearson 95% interval is added ('clopper_pearson_95'), and for precision with zero false positives a lower bound from the CP upper FPR limit and the bootstrap TPR lower limit ('lower_bound_if_FPR_at_CP_upper').

The R1 refit that recovers training scores reproduces the cached test scores exactly [`meta.R1_train_score_refit.verification`]: LR-A max |diff| 0.0e+00, LR-B max |diff| 0.0e+00, LR-C max |diff| 0.0e+00, Markov max |diff| 0.0e+00, IForest max |diff| 0.0e+00, TFIDF max |diff| 0.0e+00. The reconstructed GRU t_r80 differs from the cached value by 8.7e-09, 5.4e-10, 3.6e-08. The self-tests of the estimator code against sklearn and evalkit pass, with maximum error 6.7e-16 [`meta.self_tests_max_abs_error`].
