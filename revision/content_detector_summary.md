# Content-aware detector through the full ladder (review M2 and m13)

Script `revision/content_detector.py`; results `revision/content_detector_results.json`; content fits `revision/cache/content/`; log `revision/logs/content_detector.log`; deviations `revision/DEVIATIONS.md`, section "Content detector" (C1 to C10). Every number below is a value in the results file. Its key is given in brackets. `S` stands for a setting and `D` for a detector.

Conventions:
- All metrics are on pool E: AgentDrift benign, hard-negative and attacked rows, and AgentDojo P0. Unsuccessful attacks are scored but kept out of pool E, and their flag rate is reported separately.
- Each value is written as point [95% interval]. The point is the full-sample value; in WUT, WUT_F and WIT it is the mean of 5 repetitions. The interval is a percentile bootstrap with B = 2,000, resampling the plan 3.5 unit: trajectories (stratified) in R1, R1-wd and R3; tasks in R2; suite|user_task clusters in R4, WUT, WUT_F and LOSO. In WIT, positives are resampled by suite|injection goal and negatives by suite|user_task.
- AP@2% and AP@10% are the prevalence-weighted average precision of plan 3.1: each positive is weighted pi/(1-pi)*n_neg/n_pos. θr80 is the 20th percentile of in-sample training-positive scores; no threshold uses test data.
- "Detected" means the 95% interval of a difference excludes 0. "Holm" means it is still detected after Holm correction within the family named.
- TFIDF, TFIDF-nodigit, TFIDF-worldmask, LR-C and LR-B scores are read from the shared linear cache. Nothing was refitted for them. R1 refits of TFIDF and TFIDF-nodigit, used only to read the coefficients, reproduce the cached scores exactly: the maximum absolute difference is 0.0 [separability.R1_TFIDF.check_vs_cache]. The metric code matches sklearn and evalkit; the maximum absolute difference is 0.0 [checks.max_abs_diff].
- As plan Section 0 notes, the review had already run approximate versions of these analyses, so this is a confirmatory re-run, not a prospective test.

## M2. Only content-blind detectors are studied; a trivial content-reading baseline shows no R3 reversal and no FPR explosion

### M2.1 Does a content-reading detector change the conclusions across the ladder?

The question: the review asks for at least one content-aware detector to be run through every rung. Here that is a TF-IDF logistic regression over tool name, arguments and observation text, configured as in plan Section 2 (TFIDF), plus the same model with every digit removed (TFIDF-nodigit).

Table 1. TFIDF [settings.S.detectors.TFIDF.<AUROC | AP_w02 | AP_w10 | recall_t_r80 | FPR_t_r80>]

| Setting | Resampled unit | Pool E pos / neg | AUROC | AP@2% | AP@10% | Recall at θr80 | FPR at θr80 |
|---|---|---|---|---|---|---|---|
| R1 | trajectories | 831 / 825 | 0.999 [0.999, 1.000] | 0.984 [0.973, 0.994] | 0.996 [0.992, 0.998] | 0.763 [0.735, 0.791] | 0.000 [0.000, 0.000] |
| R1wd_s0 | trajectories | 865 / 786 | 1.000 [0.999, 1.000] | 0.990 [0.982, 0.997] | 0.998 [0.995, 0.999] | 0.773 [0.746, 0.799] | 0.000 [0.000, 0.000] |
| R1wd_s1 | trajectories | 924 / 787 | 1.000 [0.999, 1.000] | 0.991 [0.981, 0.998] | 0.998 [0.995, 0.999] | 0.792 [0.764, 0.819] | 0.000 [0.000, 0.000] |
| R1wd_s2 | trajectories | 899 / 802 | 1.000 [1.000, 1.000] | 0.994 [0.987, 0.999] | 0.999 [0.997, 1.000] | 0.786 [0.761, 0.812] | 0.000 [0.000, 0.000] |
| R1wd_s3 | trajectories | 863 / 811 | 1.000 [0.999, 1.000] | 0.990 [0.979, 0.998] | 0.997 [0.995, 0.999] | 0.768 [0.739, 0.796] | 0.000 [0.000, 0.000] |
| R1wd_s4 | trajectories | 893 / 788 | 1.000 [0.999, 1.000] | 0.987 [0.973, 0.998] | 0.997 [0.993, 0.999] | 0.761 [0.735, 0.788] | 0.000 [0.000, 0.000] |
| R2 | tasks | 775 / 729 | 0.998 [0.996, 1.000] | 0.957 [0.927, 0.985] | 0.987 [0.974, 0.996] | 0.734 [0.683, 0.781] | 0.000 [0.000, 0.000] |
| R3_medical | trajectories | 1020 / 935 | 0.996 [0.994, 0.997] | 0.926 [0.905, 0.949] | 0.973 [0.964, 0.981] | 0.658 [0.630, 0.687] | 0.000 [0.000, 0.000] |
| R3_banking | trajectories | 626 / 935 | 0.989 [0.985, 0.992] | 0.825 [0.783, 0.873] | 0.933 [0.913, 0.951] | 0.272 [0.236, 0.307] | 0.000 [0.000, 0.000] |
| R3_web | trajectories | 1020 / 935 | 0.980 [0.976, 0.984] | 0.849 [0.824, 0.877] | 0.920 [0.906, 0.936] | 0.572 [0.543, 0.603] | 0.000 [0.000, 0.000] |
| R3_email | trajectories | 1020 / 935 | 0.990 [0.987, 0.993] | 0.912 [0.888, 0.937] | 0.961 [0.950, 0.971] | 0.696 [0.668, 0.724] | 0.000 [0.000, 0.000] |
| R3_coding | trajectories | 1020 / 935 | 0.987 [0.983, 0.990] | 0.871 [0.851, 0.894] | 0.936 [0.923, 0.949] | 0.620 [0.589, 0.649] | 0.000 [0.000, 0.000] |
| R4 | suite|user_task | 2294 / 392 | 0.520 [0.471, 0.564] | 0.023 [0.019, 0.032] | 0.100 [0.089, 0.119] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| WUT | suite|user_task | 2294 / 392 | 0.996 [0.991, 0.999] | 0.943 [0.890, 0.992] | 0.980 [0.957, 0.996] | 0.780 [0.708, 0.837] | 0.000 [0.000, 0.000] |
| WUT_F | suite|user_task | 2294 / 392 | 0.941 [0.911, 0.966] | 0.349 [0.196, 0.785] | 0.663 [0.490, 0.879] | 0.755 [0.675, 0.822] | 0.049 [0.010, 0.097] |
| WIT | IT (goal / user task) | 2294 / 392 | 0.987 [0.972, 0.996] | 0.912 [0.844, 0.966] | 0.955 [0.911, 0.984] | 0.640 [0.450, 0.786] | 0.000 [0.000, 0.000] |
| LOSO_banking | suite|user_task | 671 / 56 | 0.998 [0.993, 1.000] | 0.989 [0.975, 0.999] | 0.994 [0.983, 0.999] | 0.921 [0.874, 0.977] | 0.000 [0.000, 0.000] |
| LOSO_slack | suite|user_task | 904 / 105 | 0.998 [0.995, 1.000] | 0.990 [0.980, 0.999] | 0.994 [0.987, 0.999] | 0.950 [0.922, 0.975] | 0.000 [0.000, 0.000] |
| LOSO_travel | suite|user_task | 182 / 103 | 0.989 [0.960, 1.000] | 0.989 [0.961, 1.000] | 0.990 [0.964, 1.000] | 0.489 [0.299, 0.659] | 0.000 [0.000, 0.000] |
| LOSO_workspace | suite|user_task | 537 / 128 | 0.991 [0.969, 1.000] | 0.991 [0.968, 1.000] | 0.992 [0.971, 1.000] | 0.263 [0.128, 0.397] | 0.000 [0.000, 0.000] |

Table 2. TFIDF-nodigit [settings.S.detectors.TFIDF-nodigit.<AUROC | AP_w02 | AP_w10 | recall_t_r80 | FPR_t_r80>]

| Setting | Resampled unit | Pool E pos / neg | AUROC | AP@2% | AP@10% | Recall at θr80 | FPR at θr80 |
|---|---|---|---|---|---|---|---|
| R1 | trajectories | 831 / 825 | 0.999 [0.997, 0.999] | 0.976 [0.963, 0.988] | 0.992 [0.987, 0.996] | 0.752 [0.723, 0.783] | 0.000 [0.000, 0.000] |
| R1wd_s0 | trajectories | 865 / 786 | 0.999 [0.999, 1.000] | 0.983 [0.974, 0.992] | 0.995 [0.992, 0.998] | 0.763 [0.735, 0.788] | 0.000 [0.000, 0.000] |
| R1wd_s1 | trajectories | 924 / 787 | 0.999 [0.999, 1.000] | 0.989 [0.982, 0.996] | 0.997 [0.994, 0.999] | 0.788 [0.762, 0.813] | 0.000 [0.000, 0.000] |
| R1wd_s2 | trajectories | 899 / 802 | 0.999 [0.999, 1.000] | 0.981 [0.971, 0.991] | 0.995 [0.991, 0.998] | 0.780 [0.752, 0.804] | 0.000 [0.000, 0.000] |
| R1wd_s3 | trajectories | 863 / 811 | 0.999 [0.998, 1.000] | 0.979 [0.967, 0.990] | 0.994 [0.990, 0.997] | 0.768 [0.740, 0.795] | 0.000 [0.000, 0.000] |
| R1wd_s4 | trajectories | 893 / 788 | 1.000 [0.999, 1.000] | 0.988 [0.977, 0.997] | 0.997 [0.994, 0.999] | 0.774 [0.747, 0.800] | 0.000 [0.000, 0.000] |
| R2 | tasks | 775 / 729 | 0.995 [0.990, 0.998] | 0.937 [0.903, 0.970] | 0.975 [0.956, 0.990] | 0.746 [0.698, 0.788] | 0.000 [0.000, 0.000] |
| R3_medical | trajectories | 1020 / 935 | 0.980 [0.975, 0.985] | 0.839 [0.814, 0.868] | 0.919 [0.904, 0.934] | 0.644 [0.616, 0.674] | 0.000 [0.000, 0.000] |
| R3_banking | trajectories | 626 / 935 | 0.943 [0.931, 0.954] | 0.753 [0.717, 0.789] | 0.844 [0.819, 0.867] | 0.288 [0.252, 0.321] | 0.000 [0.000, 0.000] |
| R3_web | trajectories | 1020 / 935 | 0.969 [0.963, 0.975] | 0.838 [0.815, 0.866] | 0.905 [0.890, 0.920] | 0.594 [0.566, 0.625] | 0.000 [0.000, 0.000] |
| R3_email | trajectories | 1020 / 935 | 0.974 [0.967, 0.980] | 0.878 [0.858, 0.901] | 0.930 [0.917, 0.943] | 0.672 [0.644, 0.700] | 0.000 [0.000, 0.000] |
| R3_coding | trajectories | 1020 / 935 | 0.958 [0.949, 0.966] | 0.814 [0.788, 0.839] | 0.884 [0.865, 0.901] | 0.617 [0.587, 0.646] | 0.000 [0.000, 0.000] |
| R4 | suite|user_task | 2294 / 392 | 0.470 [0.413, 0.524] | 0.028 [0.021, 0.040] | 0.103 [0.089, 0.125] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| WUT | suite|user_task | 2294 / 392 | 0.996 [0.992, 0.999] | 0.948 [0.901, 0.994] | 0.982 [0.961, 0.997] | 0.784 [0.712, 0.840] | 0.000 [0.000, 0.000] |
| WUT_F | suite|user_task | 2294 / 392 | 0.942 [0.913, 0.966] | 0.400 [0.249, 0.764] | 0.692 [0.537, 0.875] | 0.755 [0.672, 0.825] | 0.052 [0.011, 0.101] |
| WIT | IT (goal / user task) | 2294 / 392 | 0.988 [0.975, 0.996] | 0.921 [0.856, 0.972] | 0.960 [0.918, 0.986] | 0.652 [0.474, 0.788] | 0.000 [0.000, 0.000] |
| LOSO_banking | suite|user_task | 671 / 56 | 0.996 [0.991, 1.000] | 0.987 [0.973, 1.000] | 0.992 [0.980, 1.000] | 0.921 [0.874, 0.977] | 0.000 [0.000, 0.000] |
| LOSO_slack | suite|user_task | 904 / 105 | 0.998 [0.996, 1.000] | 0.991 [0.981, 1.000] | 0.995 [0.989, 1.000] | 0.949 [0.920, 0.974] | 0.000 [0.000, 0.000] |
| LOSO_travel | suite|user_task | 182 / 103 | 0.989 [0.960, 1.000] | 0.989 [0.961, 1.000] | 0.990 [0.964, 1.000] | 0.610 [0.406, 0.804] | 0.000 [0.000, 0.000] |
| LOSO_workspace | suite|user_task | 537 / 128 | 0.991 [0.968, 1.000] | 0.991 [0.968, 1.000] | 0.992 [0.971, 1.000] | 0.648 [0.515, 0.758] | 0.000 [0.000, 0.000] |

Table 3. LR-C (content-blind reference, same cache) [settings.S.detectors.LR-C.<AUROC | AP_w02 | AP_w10 | recall_t_r80 | FPR_t_r80>]

| Setting | Resampled unit | Pool E pos / neg | AUROC | AP@2% | AP@10% | Recall at θr80 | FPR at θr80 |
|---|---|---|---|---|---|---|---|
| R1 | trajectories | 831 / 825 | 0.879 [0.863, 0.895] | 0.232 [0.187, 0.326] | 0.523 [0.469, 0.586] | 0.771 [0.744, 0.799] | 0.192 [0.166, 0.218] |
| R1wd_s0 | trajectories | 865 / 786 | 0.891 [0.875, 0.906] | 0.334 [0.291, 0.400] | 0.591 [0.547, 0.644] | 0.785 [0.756, 0.813] | 0.173 [0.148, 0.198] |
| R1wd_s1 | trajectories | 924 / 787 | 0.889 [0.875, 0.904] | 0.308 [0.262, 0.389] | 0.577 [0.530, 0.633] | 0.806 [0.780, 0.833] | 0.187 [0.161, 0.213] |
| R1wd_s2 | trajectories | 899 / 802 | 0.901 [0.887, 0.915] | 0.383 [0.339, 0.447] | 0.628 [0.584, 0.679] | 0.810 [0.784, 0.834] | 0.165 [0.138, 0.191] |
| R1wd_s3 | trajectories | 863 / 811 | 0.888 [0.873, 0.903] | 0.312 [0.258, 0.408] | 0.585 [0.535, 0.644] | 0.774 [0.745, 0.801] | 0.158 [0.133, 0.182] |
| R1wd_s4 | trajectories | 893 / 788 | 0.892 [0.877, 0.907] | 0.337 [0.288, 0.414] | 0.597 [0.550, 0.651] | 0.801 [0.774, 0.828] | 0.178 [0.150, 0.203] |
| R2 | tasks | 775 / 729 | 0.902 [0.879, 0.924] | 0.355 [0.289, 0.441] | 0.615 [0.556, 0.682] | 0.791 [0.746, 0.833] | 0.150 [0.105, 0.199] |
| R3_medical | trajectories | 1020 / 935 | 0.745 [0.724, 0.766] | 0.071 [0.059, 0.095] | 0.249 [0.222, 0.288] | 0.827 [0.804, 0.849] | 0.503 [0.472, 0.535] |
| R3_banking | trajectories | 626 / 935 | 0.828 [0.809, 0.849] | 0.121 [0.097, 0.175] | 0.381 [0.334, 0.448] | 0.966 [0.952, 0.979] | 0.660 [0.630, 0.690] |
| R3_web | trajectories | 1020 / 935 | 0.862 [0.846, 0.877] | 0.244 [0.196, 0.329] | 0.505 [0.460, 0.555] | 0.990 [0.983, 0.996] | 0.707 [0.679, 0.736] |
| R3_email | trajectories | 1020 / 935 | 0.739 [0.716, 0.760] | 0.095 [0.077, 0.127] | 0.272 [0.241, 0.308] | 0.962 [0.950, 0.973] | 0.722 [0.693, 0.752] |
| R3_coding | trajectories | 1020 / 935 | 0.814 [0.797, 0.833] | 0.146 [0.121, 0.199] | 0.384 [0.343, 0.439] | 0.974 [0.963, 0.983] | 0.672 [0.643, 0.702] |
| R4 | suite|user_task | 2294 / 392 | 0.473 [0.428, 0.513] | 0.018 [0.016, 0.020] | 0.089 [0.081, 0.098] | 0.225 [0.137, 0.314] | 0.224 [0.144, 0.309] |
| WUT | suite|user_task | 2294 / 392 | 0.797 [0.747, 0.842] | 0.078 [0.054, 0.191] | 0.299 [0.229, 0.445] | 0.720 [0.645, 0.784] | 0.216 [0.125, 0.307] |
| WUT_F | suite|user_task | 2294 / 392 | 0.794 [0.749, 0.840] | 0.090 [0.059, 0.288] | 0.326 [0.246, 0.507] | 0.735 [0.660, 0.802] | 0.269 [0.175, 0.377] |
| WIT | IT (goal / user task) | 2294 / 392 | 0.687 [0.548, 0.823] | 0.060 [0.031, 0.175] | 0.242 [0.143, 0.441] | 0.591 [0.399, 0.779] | 0.257 [0.170, 0.344] |
| LOSO_banking | suite|user_task | 671 / 56 | 0.840 [0.733, 0.933] | 0.192 [0.118, 0.779] | 0.469 [0.307, 0.832] | 0.283 [0.113, 0.432] | 0.018 [0.000, 0.059] |
| LOSO_slack | suite|user_task | 904 / 105 | 0.744 [0.666, 0.836] | 0.102 [0.079, 0.290] | 0.261 [0.223, 0.467] | 0.781 [0.668, 0.888] | 0.276 [0.106, 0.463] |
| LOSO_travel | suite|user_task | 182 / 103 | 0.661 [0.585, 0.759] | 0.145 [0.065, 0.309] | 0.268 [0.178, 0.440] | 0.907 [0.851, 0.965] | 0.757 [0.589, 0.912] |
| LOSO_workspace | suite|user_task | 537 / 128 | 0.644 [0.565, 0.726] | 0.027 [0.022, 0.037] | 0.132 [0.107, 0.174] | 0.983 [0.958, 1.000] | 0.758 [0.623, 0.878] |

WUT_F is an extra setting (unsuccessful attacks added as training negatives). The task list does not include it.

Table 4. Paired TFIDF minus LR-C (same resampled rows) [settings.S.paired."TFIDF - LR-C".<metric>]. Holm is applied within the five R3 folds and within the four LOSO suites [families_holm.R3_folds, families_holm.LOSO_suites].

| Setting | ΔAUROC | ΔAP@2% | ΔAP@10% | ΔRecall at θr80 | ΔFPR at θr80 |
|---|---|---|---|---|---|
| R1 | +0.121 [+0.105, +0.136] | +0.752 [+0.657, +0.800] | +0.473 [+0.410, +0.526] | -0.008 [-0.046, +0.030] (not detected) | -0.192 [-0.218, -0.166] |
| R1wd_s0 | +0.109 [+0.094, +0.124] | +0.656 [+0.589, +0.699] | +0.406 [+0.354, +0.449] | -0.012 [-0.049, +0.024] (not detected) | -0.173 [-0.198, -0.148] |
| R1wd_s1 | +0.110 [+0.095, +0.125] | +0.683 [+0.603, +0.730] | +0.421 [+0.365, +0.468] | -0.014 [-0.049, +0.018] (not detected) | -0.187 [-0.213, -0.161] |
| R1wd_s2 | +0.099 [+0.085, +0.113] | +0.610 [+0.547, +0.655] | +0.370 [+0.320, +0.415] | -0.023 [-0.055, +0.008] (not detected) | -0.165 [-0.191, -0.138] |
| R1wd_s3 | +0.111 [+0.096, +0.126] | +0.677 [+0.581, +0.733] | +0.413 [+0.355, +0.463] | -0.006 [-0.041, +0.030] (not detected) | -0.158 [-0.182, -0.133] |
| R1wd_s4 | +0.108 [+0.093, +0.123] | +0.649 [+0.574, +0.699] | +0.400 [+0.346, +0.446] | -0.039 [-0.073, -0.003] | -0.178 [-0.203, -0.150] |
| R2 | +0.096 [+0.074, +0.119] | +0.602 [+0.514, +0.676] | +0.372 [+0.304, +0.435] | -0.057 [-0.108, -0.006] | -0.150 [-0.199, -0.105] |
| R3_medical | +0.250 [+0.230, +0.271] (Holm: detected) | +0.856 [+0.823, +0.879] (Holm: detected) | +0.724 [+0.682, +0.752] (Holm: detected) | -0.170 [-0.208, -0.129] (Holm: detected) | -0.503 [-0.535, -0.472] (Holm: detected) |
| R3_banking | +0.161 [+0.139, +0.181] (Holm: detected) | +0.704 [+0.628, +0.757] (Holm: detected) | +0.551 [+0.484, +0.604] (Holm: detected) | -0.695 [-0.735, -0.655] (Holm: detected) | -0.660 [-0.690, -0.630] (Holm: detected) |
| R3_web | +0.118 [+0.102, +0.136] (Holm: detected) | +0.605 [+0.518, +0.662] (Holm: detected) | +0.416 [+0.362, +0.464] (Holm: detected) | -0.419 [-0.449, -0.387] (Holm: detected) | -0.707 [-0.736, -0.679] (Holm: detected) |
| R3_email | +0.251 [+0.230, +0.273] (Holm: detected) | +0.818 [+0.775, +0.848] (Holm: detected) | +0.689 [+0.650, +0.722] (Holm: detected) | -0.266 [-0.296, -0.235] (Holm: detected) | -0.722 [-0.752, -0.693] (Holm: detected) |
| R3_coding | +0.172 [+0.153, +0.190] (Holm: detected) | +0.725 [+0.667, +0.758] (Holm: detected) | +0.551 [+0.496, +0.593] (Holm: detected) | -0.354 [-0.385, -0.323] (Holm: detected) | -0.672 [-0.702, -0.643] (Holm: detected) |
| R4 | +0.047 [-0.002, +0.096] (not detected) | +0.005 [+0.001, +0.014] | +0.012 [+0.001, +0.027] | -0.225 [-0.314, -0.137] | -0.224 [-0.309, -0.144] |
| WUT | +0.199 [+0.154, +0.249] | +0.865 [+0.755, +0.925] | +0.682 [+0.536, +0.752] | +0.060 [-0.037, +0.148] (not detected) | -0.216 [-0.307, -0.125] |
| WUT_F | +0.148 [+0.095, +0.195] | +0.259 [+0.040, +0.677] | +0.337 [+0.099, +0.560] | +0.019 [-0.084, +0.117] (not detected) | -0.220 [-0.330, -0.123] |
| WIT | +0.300 [+0.168, +0.440] | +0.852 [+0.736, +0.906] | +0.713 [+0.521, +0.808] | +0.048 [-0.158, +0.261] (not detected) | -0.257 [-0.344, -0.170] |
| LOSO_banking | +0.158 [+0.064, +0.264] (Holm: detected) | +0.797 [+0.210, +0.872] (Holm: detected) | +0.525 [+0.160, +0.686] (Holm: detected) | +0.638 [+0.455, +0.851] (Holm: detected) | -0.018 [-0.059, +0.000] (not detected) |
| LOSO_slack | +0.254 [+0.163, +0.332] (Holm: detected) | +0.888 [+0.701, +0.908] (Holm: detected) | +0.733 [+0.529, +0.768] (Holm: detected) | +0.169 [+0.077, +0.263] (Holm: detected) | -0.276 [-0.463, -0.106] (Holm: detected) |
| LOSO_travel | +0.328 [+0.223, +0.406] (Holm: detected) | +0.844 [+0.674, +0.925] (Holm: detected) | +0.722 [+0.548, +0.812] (Holm: detected) | -0.418 [-0.581, -0.267] (Holm: detected) | -0.757 [-0.912, -0.589] (Holm: detected) |
| LOSO_workspace | +0.347 [+0.267, +0.422] (Holm: detected) | +0.964 [+0.940, +0.977] (Holm: detected) | +0.860 [+0.811, +0.887] (Holm: detected) | -0.721 [-0.862, -0.581] (Holm: detected) | -0.758 [-0.878, -0.623] (Holm: detected) |

Table 5. Flag rate of unsuccessful attacks at θr80 (outside pool E) [settings.S.per_class_t_r80.D.flag_failed_attack.rate]. The interval resamples trajectories for AgentDrift and suite|user_task clusters for AgentDojo.

| Setting | n | TFIDF | TFIDF-nodigit | LR-C |
|---|---|---|---|---|
| R1 | 225 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.218 [0.164, 0.271] |
| R2 | 218 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.234 [0.183, 0.289] |
| R3_medical | 255 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.576 [0.514, 0.635] |
| R3_banking | 255 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.831 [0.784, 0.875] |
| R3_web | 255 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.745 [0.686, 0.796] |
| R3_email | 255 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.616 [0.557, 0.675] |
| R3_coding | 255 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.753 [0.698, 0.804] |
| R4 | 4987 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.056 [0.033, 0.086] |
| WUT | 4987 | 0.308 [0.238, 0.377] | 0.334 [0.266, 0.405] | 0.279 [0.187, 0.379] |
| WUT_F | 4987 | 0.055 [0.030, 0.089] | 0.059 [0.034, 0.090] | 0.227 [0.134, 0.320] |
| WIT | 4987 | 0.360 [0.284, 0.431] | 0.378 [0.300, 0.452] | 0.319 [0.221, 0.425] |
| LOSO_banking | 723 | 0.571 [0.427, 0.657] | 0.582 [0.445, 0.673] | 0.090 [0.013, 0.198] |
| LOSO_slack | 592 | 0.544 [0.485, 0.596] | 0.539 [0.480, 0.590] | 0.334 [0.141, 0.548] |
| LOSO_travel | 1798 | 0.433 [0.308, 0.549] | 0.487 [0.354, 0.606] | 0.882 [0.768, 0.969] |
| LOSO_workspace | 1874 | 0.253 [0.156, 0.353] | 0.479 [0.370, 0.579] | 0.926 [0.855, 0.977] |

Plain reading.
In distribution (R1, the five world-disjoint splits and R2), TFIDF is close to perfect. Its AUROC is 0.998 to 1.000 and its AP@2% is 0.957 to 0.994. At θr80 it flags none of the pool-E negatives. LR-C, by contrast, reaches AP@2% of 0.232 to 0.383. TFIDF minus LR-C in AUROC is 0.096 to 0.121 and is detected in all seven in-distribution settings. On the world-disjoint splits, TFIDF has a mean AUROC of 0.9997 (range 0.9996 to 0.9998) [R1wd_summary.TFIDF.AUROC], so its R1 performance does not come from shared worlds. Under domain shift (R3), TFIDF keeps an AUROC of 0.980 to 0.996 and an FPR at θr80 of exactly 0 in every fold. LR-C falls to 0.739 to 0.862, with FPR 0.503 to 0.722. TFIDF minus LR-C is detected after Holm in 5 of 5 folds for AUROC, AP@2% and AP@10%. Removing digits changes R3 AUROC by -0.011 to -0.046 (paired TFIDF-nodigit minus TFIDF, detected in 5 of 5 folds) [settings.R3_<d>.paired."TFIDF-nodigit - TFIDF".AUROC]. The largest change is in banking, -0.046 [-0.057, -0.036], where TFIDF-nodigit still reaches 0.943 [0.931, 0.954]. The review's claim holds: under this baseline there is no FPR explosion and no ranking reversal (M2.4). Across benchmarks, however, the content model fails like the content-blind ones. In R4 transfer its AUROC is 0.520 [0.471, 0.564], an interval that includes 0.5. At θr80 it flags no AgentDojo trajectory at all: recall 0, FPR 0, and a failed-attack flag rate of 0.000. TFIDF minus LR-C in R4 AUROC is +0.047 [-0.002, +0.096] (not detected). Within AgentDojo it is again near perfect: WUT 0.996 [0.991, 0.999], WIT 0.987 [0.972, 0.996], and LOSO 0.989 to 0.998. These numbers do not show that TF-IDF recognises hijacking. M2.5 and M2.6 show that it reads the attack and negative-class templates of each generator. They also do not show that a content detector transfers from synthetic to real traces; R4 shows that this one does not. The in-distribution precision numbers rest on zero false positives. In R1 no negative is flagged among 825; the exact 95% upper bound on the FPR is 0.0045, which implies a precision at 2% of at least 0.777 [zero_fp_bounds.R1.TFIDF]. The corresponding point for LR-C is 0.076 [0.067, 0.087] [settings.R1.detectors.LR-C.prec02_t_r80]; the review cites 0.077 for it.

Comparison with the review (approximate values, REVIEW_REPORT.md lines 191-195, stored under review_reported). The review reported R3 AUROC of 0.993/0.981/0.974/0.985/0.982. The plan configuration here gives 0.996/0.989/0.980/0.990/0.987, higher in every fold. The review reported R1 recall at θr80 of 0.783; here it is 0.763. The review's exact configuration is not known. The conclusions are the same.

### M2.2 Is the R1 result world-identity leakage? (world-mask sensitivity)

The question: does masking the world record change TFIDF in R1? TFIDF-worldmask is the plan's mask, from the linear cache. It replaces whole world string values. TFIDF-anon and TFIDF-anon-nodigit are added sensitivity variants (deviation C3). They follow the anonymisation the AgentDrift README recommends (lines 151-153): person names and their single name tokens, the company name and the world date become placeholders, and every e-mail address becomes an in-domain or external placeholder. The -nodigit variant also removes all digits, as README lines 157-158 advise (mask target domains and amounts).

Table 6. R1 [settings.R1.detectors.D.<metric>; paired: settings.R1.paired."D - TFIDF"]

| Detector | AUROC | AP@2% | Recall at θr80 | FPR at θr80 | ΔAUROC vs TFIDF | ΔAP@2% vs TFIDF |
|---|---|---|---|---|---|---|
| TFIDF | 0.999 [0.999, 1.000] | 0.984 [0.973, 0.994] | 0.763 [0.735, 0.791] | 0.000 [0.000, 0.000] | - | - |
| TFIDF-worldmask | 0.999 [0.998, 1.000] | 0.978 [0.962, 0.992] | 0.757 [0.728, 0.787] | 0.000 [0.000, 0.000] | -0.000 [-0.001, +0.000] (not detected) | -0.006 [-0.019, +0.002] (not detected) |
| TFIDF-anon | 0.999 [0.998, 1.000] | 0.981 [0.968, 0.993] | 0.765 [0.736, 0.794] | 0.000 [0.000, 0.000] | -0.000 [-0.000, +0.000] (not detected) | -0.003 [-0.010, +0.002] (not detected) |
| TFIDF-anon-nodigit | 0.999 [0.998, 0.999] | 0.980 [0.971, 0.989] | 0.758 [0.729, 0.787] | 0.000 [0.000, 0.000] | -0.001 [-0.001, -0.000] | -0.004 [-0.014, +0.005] (not detected) |
| TFIDF-nodigit | 0.999 [0.997, 0.999] | 0.976 [0.963, 0.988] | 0.752 [0.723, 0.783] | 0.000 [0.000, 0.000] | -0.001 [-0.002, -0.000] | -0.008 [-0.020, +0.004] (not detected) |

Residual identity: after the plan's world mask, a first- or last-name token of the user or a contact still occurs in 0.198 of R1 trajectories, and 0.441 of trajectories contain no world string at all. After the README-style anonymisation the residual is 0.006 [worldmask_info]. In R3, TFIDF-anon-nodigit has an AUROC of 0.964 to 0.987 [settings.R3_<d>.detectors.TFIDF-anon-nodigit.AUROC]. Its FPR at θr80 is 0 in four folds and 0.002 in email. Under world-cluster resampling (sensitivity; 676 worlds in the R1 test pool), TFIDF's R1 AUROC is 0.999 [0.999, 1.000] [settings.R1.sensitivity.world_cluster].

Plain reading. Neither the plan's world mask nor the stronger README anonymisation changes R1 in any way that matters: every variant keeps an AUROC of at least 0.998 and zero false positives at θr80. The R1 paired differences in AUROC are at most 0.001, and those in AP@2% are small; Table 6 flags which are detected. World-identity leakage therefore does not explain the content model's R1 result, which agrees with the world-disjoint result in M2.1. What this does not rule out is template memorisation. The masks remove identity strings, not the class-specific phrasing of the generator. M2.6 and m13 show that the phrasing carries the signal.

### M2.3 Threshold behaviour of TFIDF under R3, relative to domain-matched R1 (plan 3.2 and 3.3)

The question: does θr80, set on training positives, carry over to a held-out domain? Plan 3.3 defines threshold failure as a detected change in recall or FPR at θr80 relative to the R1 model scored on the same domain's R1 test rows. R3 minus R1 uses independent bootstraps, and Holm is applied over the five folds per detector and metric [domain_matched].

Table 7. Recall and FPR at θr80: domain-matched R1 (R1dm) against R3 [domain_matched.domains.<d>.R1dm|R3.D; change: domain_matched.domains.<d>.R3_minus_R1dm.D.<recall_t_r80 | FPR_t_r80>]

| Fold | Detector | R1dm recall | R3 recall | Δ recall | R1dm FPR | R3 FPR | Δ FPR | Threshold failure (Holm) |
|---|---|---|---|---|---|---|---|---|
| medical | TFIDF | 0.739 [0.672, 0.800] | 0.658 [0.630, 0.687] | -0.081 [-0.150, -0.008] (Holm: detected) | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] (not detected) | yes |
| medical | TFIDF-nodigit | 0.672 [0.600, 0.739] | 0.644 [0.616, 0.674] | -0.028 [-0.104, +0.048] (not detected) | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] (not detected) | no |
| medical | LR-C | 0.650 [0.583, 0.717] | 0.827 [0.804, 0.849] | +0.177 [+0.105, +0.252] (Holm: detected) | 0.321 [0.248, 0.394] | 0.503 [0.472, 0.535] | +0.181 [+0.108, +0.259] (Holm: detected) | yes |
| medical | LR-B | 0.694 [0.628, 0.761] | 0.560 [0.529, 0.588] | -0.135 [-0.211, -0.062] (Holm: detected) | 0.236 [0.170, 0.303] | 0.150 [0.128, 0.171] | -0.087 [-0.154, -0.016] (detected; not after Holm) | yes |
| banking | TFIDF | 0.685 [0.604, 0.766] | 0.272 [0.236, 0.307] | -0.413 [-0.508, -0.323] (Holm: detected) | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] (not detected) | yes |
| banking | TFIDF-nodigit | 0.703 [0.613, 0.784] | 0.288 [0.252, 0.321] | -0.415 [-0.510, -0.321] (Holm: detected) | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] (not detected) | yes |
| banking | LR-C | 0.613 [0.514, 0.703] | 0.966 [0.952, 0.979] | +0.354 [+0.262, +0.453] (Holm: detected) | 0.121 [0.073, 0.176] | 0.660 [0.630, 0.690] | +0.539 [+0.478, +0.593] (Holm: detected) | yes |
| banking | LR-B | 0.802 [0.721, 0.874] | 0.829 [0.799, 0.858] | +0.027 [-0.052, +0.111] (not detected) | 0.255 [0.188, 0.327] | 0.278 [0.249, 0.306] | +0.024 [-0.053, +0.096] (not detected) | no |
| web | TFIDF | 0.767 [0.706, 0.828] | 0.572 [0.543, 0.603] | -0.195 [-0.269, -0.129] (Holm: detected) | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] (not detected) | yes |
| web | TFIDF-nodigit | 0.783 [0.728, 0.844] | 0.594 [0.566, 0.625] | -0.189 [-0.258, -0.124] (Holm: detected) | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] (not detected) | yes |
| web | LR-C | 0.906 [0.861, 0.944] | 0.990 [0.983, 0.996] | +0.085 [+0.043, +0.128] (Holm: detected) | 0.164 [0.109, 0.218] | 0.707 [0.679, 0.736] | +0.543 [+0.481, +0.602] (Holm: detected) | yes |
| web | LR-B | 0.861 [0.811, 0.911] | 0.825 [0.800, 0.848] | -0.037 [-0.091, +0.019] (not detected) | 0.255 [0.194, 0.321] | 0.197 [0.172, 0.222] | -0.058 [-0.129, +0.010] (not detected) | no |
| email | TFIDF | 0.817 [0.761, 0.872] | 0.696 [0.668, 0.724] | -0.121 [-0.183, -0.058] (Holm: detected) | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] (not detected) | yes |
| email | TFIDF-nodigit | 0.817 [0.761, 0.872] | 0.672 [0.644, 0.700] | -0.145 [-0.208, -0.081] (Holm: detected) | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] (not detected) | yes |
| email | LR-C | 0.789 [0.728, 0.844] | 0.962 [0.950, 0.973] | +0.173 [+0.115, +0.235] (Holm: detected) | 0.127 [0.079, 0.182] | 0.722 [0.693, 0.752] | +0.595 [+0.537, +0.649] (Holm: detected) | yes |
| email | LR-B | 0.794 [0.733, 0.850] | 0.833 [0.812, 0.856] | +0.039 [-0.022, +0.104] (not detected) | 0.358 [0.285, 0.436] | 0.418 [0.387, 0.450] | +0.061 [-0.019, +0.140] (not detected) | no |
| coding | TFIDF | 0.778 [0.717, 0.839] | 0.620 [0.589, 0.649] | -0.158 [-0.225, -0.091] (Holm: detected) | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] (not detected) | yes |
| coding | TFIDF-nodigit | 0.767 [0.706, 0.828] | 0.617 [0.587, 0.646] | -0.150 [-0.215, -0.083] (Holm: detected) | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] (not detected) | yes |
| coding | LR-C | 0.839 [0.783, 0.889] | 0.974 [0.963, 0.983] | +0.135 [+0.083, +0.191] (Holm: detected) | 0.224 [0.164, 0.285] | 0.672 [0.643, 0.702] | +0.447 [+0.378, +0.516] (Holm: detected) | yes |
| coding | LR-B | 0.828 [0.772, 0.878] | 0.835 [0.813, 0.859] | +0.008 [-0.048, +0.070] (not detected) | 0.315 [0.242, 0.382] | 0.283 [0.255, 0.312] | -0.032 [-0.101, +0.044] (not detected) | no |

Number of folds with threshold failure, unadjusted / Holm [domain_matched.threshold_failure.D.n_folds_failure(_holm)]: TFIDF 5 / 5; TFIDF-nodigit 4 / 4; TFIDF-toolname 5 / 5; TFIDF-anon-nodigit 4 / 4; LR-C 5 / 5; LR-B 1 / 1.

Table 8. AUROC, R3 minus domain-matched R1 [domain_matched.domains.<d>.R3_minus_R1dm.D.AUROC]

| Fold | TFIDF R1dm | TFIDF R3 | TFIDF Δ | TFIDF-nodigit Δ | LR-C Δ | LR-B Δ |
|---|---|---|---|---|---|---|
| medical | 0.999 [0.998, 1.000] | 0.996 [0.994, 0.997] | -0.004 [-0.006, -0.002] (Holm: detected) | -0.017 [-0.023, -0.011] (Holm: detected) | -0.041 [-0.092, +0.007] (not detected) | +0.015 [-0.037, +0.067] (not detected) |
| banking | 0.999 [0.997, 1.000] | 0.989 [0.985, 0.992] | -0.010 [-0.014, -0.006] (Holm: detected) | -0.054 [-0.066, -0.043] (Holm: detected) | -0.024 [-0.073, +0.029] (not detected) | +0.023 [-0.028, +0.078] (not detected) |
| web | 0.998 [0.995, 1.000] | 0.980 [0.976, 0.984] | -0.018 [-0.023, -0.013] (Holm: detected) | -0.028 [-0.036, -0.020] (Holm: detected) | -0.065 [-0.096, -0.035] (Holm: detected) | +0.010 [-0.026, +0.049] (not detected) |
| email | 1.000 [1.000, 1.000] | 0.990 [0.987, 0.993] | -0.010 [-0.013, -0.007] (Holm: detected) | -0.026 [-0.033, -0.020] (Holm: detected) | -0.155 [-0.193, -0.112] (Holm: detected) | -0.012 [-0.059, +0.038] (not detected) |
| coding | 1.000 [0.999, 1.000] | 0.987 [0.983, 0.990] | -0.013 [-0.017, -0.010] (Holm: detected) | -0.041 [-0.050, -0.033] (Holm: detected) | -0.080 [-0.116, -0.040] (Holm: detected) | +0.006 [-0.034, +0.049] (not detected) |

Table 9. TFIDF recall at θr80 by hijack pattern [settings.S.per_class_t_r80.TFIDF.recall_attacked_<pattern>.rate], with the pool-E positive composition [settings.S.pattern_composition_poolE_pos]

| Setting | Full | Partial | Delayed | Positives full / partial / delayed |
|---|---|---|---|---|
| R1 | 0.974 [0.959, 0.987] | 0.541 [0.476, 0.602] | 0.410 [0.328, 0.493] | 466 / 231 / 134 |
| R3_medical | 0.984 [0.974, 0.992] | 0.211 [0.162, 0.259] | 0.071 [0.032, 0.116] | 618 / 247 / 155 |
| R3_banking | 0.733 [0.668, 0.788] | 0.038 [0.019, 0.062] | 0.007 [0.000, 0.020] | 217 / 260 / 149 |
| R3_web | 0.837 [0.807, 0.865] | 0.274 [0.221, 0.327] | 0.020 [0.000, 0.047] | 607 / 263 / 150 |
| R3_email | 0.967 [0.952, 0.980] | 0.345 [0.288, 0.405] | 0.225 [0.159, 0.298] | 605 / 264 / 151 |
| R3_coding | 0.988 [0.980, 0.997] | 0.158 [0.116, 0.201] | 0.006 [0.000, 0.018] | 597 / 259 / 164 |

Plain reading.
By the plan 3.3 definition, TFIDF's threshold fails in all five R3 folds, and only in the recall direction. Relative to the domain-matched R1 value, recall at θr80 changes by -0.081 (medical) to -0.413 (banking), and every change is detected after Holm. The FPR is 0 in both R1dm and R3 for every domain. By hijack pattern (Table 9), TFIDF still recovers 0.733 to 0.988 of full hijacks in R3 but only 0.038 to 0.345 of partial and 0.006 to 0.225 of delayed ones; the pooled R1 values are 0.974, 0.541 and 0.410 (these per-pattern values are not domain-matched). The largest drop is in banking, the fold with the smallest share of full hijacks (217 of 626 positives). There even full-hijack recall is lower (0.733 [0.668, 0.788]). Recall at θr80 therefore depends on the hijack mix of the fold, which differs across folds (m13). TFIDF's AUROC also changes by a detected amount in every fold, but the amount is small (-0.004 to -0.018). The content-blind detectors fail differently. LR-C's recall and FPR both rise; its FPR changes by 0.181 to 0.595. LR-B's threshold fails after Holm only in medical, and holds in the other folds. The numbers show that threshold non-transfer under domain shift is not confined to content-blind detectors. Its direction depends on the detector: missed attacks for TFIDF, false alarms for LR-C. They do not show that TFIDF's operating point is safe: at zero FPR its held-out recall at θr80 is only 0.272 to 0.696.

### M2.4 Does the content detector show a ranking reversal under R3? (plan 3.4)

Plan 3.4 claims a reversal only where (i) the in-distribution AUROC advantage is detected in the domain-matched R1 comparison and (ii) the R3 paired difference is detected with the opposite sign, both after Holm over the five folds [domain_matched.reversal_check_AUROC].

| Pair | R1dm Δ AUROC (5 folds) | R3 Δ AUROC (5 folds) | Reversals |
|---|---|---|---|
| TFIDF - LR-C | +0.213, +0.146, +0.071, +0.106, +0.106 | +0.250, +0.161, +0.118, +0.251, +0.172 | 0 |
| TFIDF - LR-B | +0.206, +0.167, +0.112, +0.196, +0.154 | +0.188, +0.135, +0.084, +0.198, +0.135 | 0 |
| TFIDF-nodigit - LR-C | +0.211, +0.144, +0.070, +0.106, +0.105 | +0.235, +0.115, +0.106, +0.235, +0.144 | 0 |
| TFIDF-toolname - LR-B | -0.052, -0.043, -0.017, +0.085, +0.013 | -0.288, -0.315, -0.310, -0.238, -0.189 | 1 |

Plain reading. TFIDF beats both LR-C and LR-B in every domain, in domain-matched R1 and in R3, and every one of these differences is detected after Holm [domain_matched.domains.<d>.R1dm_paired | R3_paired.<pair>.AUROC]. TFIDF shows no reversal. The in-distribution ranking of the best detector carries over to held-out domains. For this detector family, the paper's R3 lesson ("the in-distribution ranking reverses") does not hold. The tool-name variant has an in-distribution advantage over LR-B that survives Holm only in email (email R1dm +0.085 [+0.031, +0.141] (Holm: detected), R3 -0.238 [-0.267, -0.210] (Holm: detected)). There the R3 difference is detected with the opposite sign, so by the plan 3.4 definition it reverses in email. In the other folds there is no detected in-distribution advantage to reverse, although TFIDF-toolname is far worse than LR-B in R3 in all five folds (M2.7).

### M2.5 R4: does content rescue transfer to real AgentDojo traces?

Table 10. Within-AgentDojo minus transfer, paired on the same P0 rows [within_vs_transfer.<W>.deltas.D.<metric>]. Holm is applied over {TFIDF, TFIDF-nodigit} x {WUT, WIT}. WUT_F is an extra.

| Within scheme | Detector | Δ AUROC | Δ AP@2% | Δ Recall at θr80 |
|---|---|---|---|---|
| WUT | TFIDF | +0.476 [+0.429, +0.520] (Holm: detected) | +0.920 [+0.863, +0.969] (Holm: detected) | +0.780 [+0.707, +0.841] (Holm: detected) |
| WUT | TFIDF-nodigit | +0.526 [+0.472, +0.580] (Holm: detected) | +0.920 [+0.871, +0.967] (Holm: detected) | +0.784 [+0.714, +0.843] (Holm: detected) |
| WUT | LR-C | +0.324 [+0.269, +0.386] | +0.060 [+0.037, +0.179] | +0.495 [+0.373, +0.618] |
| WIT | TFIDF | +0.467 [+0.353, +0.601] (Holm: detected) | +0.890 [+0.824, +0.942] (Holm: detected) | +0.640 [+0.452, +0.786] (Holm: detected) |
| WIT | TFIDF-nodigit | +0.518 [+0.386, +0.671] (Holm: detected) | +0.893 [+0.834, +0.943] (Holm: detected) | +0.652 [+0.477, +0.788] (Holm: detected) |
| WIT | LR-C | +0.214 [+0.034, +0.398] | +0.042 [+0.013, +0.168] | +0.366 [+0.166, +0.590] |
| WUT_F | TFIDF | +0.421 [+0.379, +0.467] | +0.326 [+0.181, +0.768] | +0.755 [+0.675, +0.817] |
| WUT_F | TFIDF-nodigit | +0.472 [+0.414, +0.526] | +0.373 [+0.226, +0.753] | +0.755 [+0.673, +0.820] |
| WUT_F | LR-C | +0.321 [+0.267, +0.376] | +0.072 [+0.041, +0.257] | +0.510 [+0.387, +0.619] |

AgentDojo attack templates [separability.agentdojo_template_markers]. We checked five fixed attack strings: "<INFORMATION>", "ignore your previous", "TODO:", "IMPORTANT!!!" and "ignore all previous". At least one occurs in 0.996 of attacked P0 runs, in 0.947 of unsuccessful attacks and in 0.000 of benign P0 runs. Used as a fixed rule, with no fitting, marker presence gives an AUROC of 0.998 on P0. The largest coefficients of an in-sample TF-IDF fit on P0 are the words of the "important_instructions" template ("solve the", "the beginning", "gave you", "information") [separability.AgentDojo_P0_TFIDF_in_sample.coefs.top_positive].

Plain reading.
Content does not rescue cross-benchmark transfer. TFIDF's R4 AUROC is 0.520 [0.471, 0.564], and TFIDF-nodigit's is 0.470 [0.413, 0.524]. Neither interval excludes 0.5 under the plan's cluster bootstrap. With trajectory-level resampling (sensitivity), TFIDF-nodigit is 0.470 [0.439, 0.499], slightly below chance [settings.R4.sensitivity.trajectory_stratified]. The AgentDrift-trained threshold flags nothing on AgentDojo. The within-minus-transfer gaps are larger for TFIDF (+0.476 [+0.429, +0.520] in WUT) than for LR-C (+0.324 [+0.269, +0.386]), so for this detector too the R4 result is a failure to transfer, not a weak representation. The RQ4 conclusion generalises to the content detector. The within-AgentDojo scores should not be read as hijack detection. AgentDojo benign runs contain no injection, while almost every attacked run carries a fixed attack string. The content model therefore mostly detects that an injection is present. In WUT it flags 0.308 [0.238, 0.377] of unsuccessful attacks at θr80, against an FPR of 0.000 on benign runs. When unsuccessful attacks are added as training negatives (WUT_F), the failed-attack flag rate falls to 0.055 [0.030, 0.089]. AUROC then falls to 0.941 [0.911, 0.966], the benign FPR at θr80 rises to 0.049 [0.010, 0.097], and AP@2% falls to 0.349 [0.196, 0.785]. Separating success from attempt is much harder than separating attack from no attack.

### M2.6 Is AgentDrift nearly separable by surface content? (external-validity finding)

In-sample fit on R1 train [separability.R1_TFIDF.train_fit]: AUROC 0.99997, accuracy at 0.5 0.9974. R1 test AUROC is 0.999 [0.999, 1.000] (Table 1).

Table 11. The 25 largest positive and 25 largest negative R1 TF-IDF coefficients, with the share of R1 training trajectories of each category that contain the feature [separability.R1_TFIDF.coefs.top_positive | top_negative]

| Direction | Feature | Coefficient | Benign | Hard negative | Failed attack | Attacked |
|---|---|---|---|---|---|---|
| toward attacked | `correction` | +6.10 | 0.000 | 0.000 | 0.000 | 0.169 |
| toward attacked | `audit com` | +4.50 | 0.000 | 0.000 | 0.000 | 0.436 |
| toward attacked | `ext` | +4.29 | 0.000 | 0.000 | 0.001 | 0.403 |
| toward attacked | `compliance` | +4.29 | 0.000 | 0.021 | 0.164 | 0.467 |
| toward attacked | `retrieved` | +3.82 | 0.142 | 0.142 | 0.314 | 0.271 |
| toward attacked | `2024` | +3.77 | 0.066 | 0.342 | 0.130 | 0.369 |
| toward attacked | `instead` | +3.67 | 0.000 | 0.000 | 0.001 | 0.089 |
| toward attacked | `compliance review` | +3.53 | 0.000 | 0.005 | 0.002 | 0.393 |
| toward attacked | `ext audit` | +3.50 | 0.000 | 0.000 | 0.000 | 0.369 |
| toward attacked | `external` | +3.22 | 0.000 | 0.001 | 0.003 | 0.084 |
| toward attacked | `to compliance` | +2.93 | 0.000 | 0.000 | 0.008 | 0.381 |
| toward attacked | `review ext` | +2.77 | 0.000 | 0.000 | 0.000 | 0.330 |
| toward attacked | `policy` | +2.77 | 0.051 | 0.082 | 0.030 | 0.462 |
| toward attacked | `also` | +2.60 | 0.009 | 0.093 | 0.018 | 0.091 |
| toward attacked | `must` | +2.51 | 0.000 | 0.000 | 0.006 | 0.371 |
| toward attacked | `per` | +2.50 | 0.029 | 0.033 | 0.572 | 0.600 |
| toward attacked | `malicious` | +2.47 | 0.000 | 0.000 | 0.006 | 0.025 |
| toward attacked | `verification` | +2.47 | 0.001 | 0.030 | 0.058 | 0.203 |
| toward attacked | `to security` | +2.41 | 0.000 | 0.001 | 0.009 | 0.070 |
| toward attacked | `retrieved for` | +2.40 | 0.016 | 0.012 | 0.048 | 0.080 |
| toward attacked | `must be` | +2.36 | 0.000 | 0.000 | 0.004 | 0.353 |
| toward attacked | `all` | +2.33 | 0.119 | 0.098 | 0.164 | 0.567 |
| toward attacked | `review` | +2.30 | 0.122 | 0.397 | 0.056 | 0.526 |
| toward attacked | `correction also` | +2.24 | 0.000 | 0.000 | 0.000 | 0.052 |
| toward attacked | `instead of` | +2.21 | 0.000 | 0.000 | 0.001 | 0.046 |
| toward negative | `2026` | -9.68 | 0.742 | 0.000 | 0.848 | 0.046 |
| toward negative | `invoice` | -4.21 | 0.029 | 0.830 | 0.026 | 0.028 |
| toward negative | `5000` | -4.00 | 0.011 | 0.806 | 0.009 | 0.021 |
| toward negative | `to audit` | -3.87 | 0.000 | 0.772 | 0.001 | 0.033 |
| toward negative | `10` | -3.87 | 0.644 | 0.541 | 0.983 | 0.317 |
| toward negative | `on 2026` | -3.80 | 0.305 | 0.000 | 0.048 | 0.000 |
| toward negative | `id` | -3.41 | 0.494 | 0.350 | 0.167 | 0.271 |
| toward negative | `within` | -3.30 | 0.018 | 0.009 | 0.956 | 0.002 |
| toward negative | `minutes` | -3.28 | 0.028 | 0.019 | 0.990 | 0.013 |
| toward negative | `for invoice` | -3.26 | 0.002 | 0.562 | 0.000 | 0.001 |
| toward negative | `10 minutes` | -3.23 | 0.000 | 0.000 | 0.961 | 0.000 |
| toward negative | `please` | -3.06 | 0.144 | 0.803 | 0.077 | 0.170 |
| toward negative | `5000 to` | -2.87 | 0.000 | 0.511 | 0.001 | 0.001 |
| toward negative | `transfer 5000` | -2.80 | 0.000 | 0.521 | 0.001 | 0.000 |
| toward negative | `by` | -2.80 | 0.301 | 0.143 | 0.840 | 0.267 |
| toward negative | `avoid` | -2.72 | 0.009 | 0.016 | 0.846 | 0.010 |
| toward negative | `transfer` | -2.71 | 0.116 | 0.803 | 0.281 | 0.095 |
| toward negative | `to avoid` | -2.61 | 0.000 | 0.009 | 0.833 | 0.009 |
| toward negative | `most` | -2.55 | 0.231 | 0.001 | 0.006 | 0.001 |
| toward negative | `00` | -2.48 | 0.325 | 0.276 | 0.052 | 0.097 |
| toward negative | `most recent` | -2.45 | 0.224 | 0.000 | 0.006 | 0.000 |
| toward negative | `within 10` | -2.41 | 0.001 | 0.000 | 0.640 | 0.000 |
| toward negative | `immediately` | -2.41 | 0.001 | 0.000 | 0.636 | 0.002 |
| toward negative | `security alert` | -2.36 | 0.000 | 0.001 | 0.916 | 0.036 |
| toward negative | `2026 02` | -2.35 | 0.131 | 0.000 | 0.050 | 0.000 |

Table 12. Share of trajectories whose detector text (tool, arguments, observations) contains "2026" [separability.token_2026.<population>.<token_in_TFIDF_text | substring_in_TFIDF_text>.<category>.share]

| Population | Definition | Benign | Hard negative | Attacked | Failed attack |
|---|---|---|---|---|---|
| R1_train | TF-IDF token | 0.742 (2077/2800) | 0.000 (0/1050) | 0.046 (180/3875) | 0.848 (890/1050) |
| R1_train | substring | 0.754 (2110/2800) | 0.000 (0/1050) | 0.047 (181/3875) | 0.862 (905/1050) |
| R1_test | TF-IDF token | 0.760 (456/600) | 0.000 (0/225) | 0.047 (39/831) | 0.840 (189/225) |
| R1_test | substring | 0.767 (460/600) | 0.000 (0/225) | 0.047 (39/831) | 0.844 (190/225) |
| analysis_pool_train+test | TF-IDF token | 0.745 (2533/3400) | 0.000 (0/1275) | 0.047 (219/4706) | 0.846 (1079/1275) |
| analysis_pool_train+test | substring | 0.756 (2570/3400) | 0.000 (0/1275) | 0.047 (220/4706) | 0.859 (1095/1275) |
| full_corpus_train+val+test | TF-IDF token | 0.743 (2971/4000) | 0.000 (0/1500) | 0.048 (263/5536) | 0.847 (1270/1500) |
| full_corpus_train+val+test | substring | 0.753 (3012/4000) | 0.000 (0/1500) | 0.048 (264/5536) | 0.859 (1288/1500) |

By field (analysis pool): the token occurs in observations of 0.722 of benign, 0.000 of hard-negative, 0.047 of attacked and 0.842 of failed-attack trajectories. Every world record contains it (share 1.000 in each category), but the world is not part of the detector's text. It enters through generated dates of that year in benign observations and through protocol and reference codes that contain it in failed-attack poison. No hard negative contains it in the detector text. The review's figure of 75.6% of benign (line 195) is the substring share on the 10,656-trajectory analysis pool (0.756).

Table 13. Single-feature rules. The feature and its direction are chosen on R1 training data by univariate training AUROC and then evaluated on the R1 test pool E [separability.R1_TFIDF.single_token_rules]. Also shown is the README's fixed external-address rule [separability.readme_external_address_rule_R1_test].

| Rule | Direction | Training AUROC | Test AUROC (pool E) |
|---|---|---|---|
| `2026` | absence->attacked | 0.780 | 0.753 |
| `per` | presence->attacked | 0.727 | 0.780 |
| `all` | presence->attacked | 0.722 | 0.728 |
| `audit com` | presence->attacked | 0.718 | 0.718 |
| `compliance` | presence->attacked | 0.714 | 0.730 |
| `policy` | presence->attacked | 0.705 | 0.698 |
| external address in arguments (README lines 112, 117-119 and 127) | presence->attacked | n/a (no fitting) | 0.711 |

The external-address rule flags 0.522 of attacked, 0.103 of benign, 0.093 of hard-negative and 0.120 of failed-attack R1 test trajectories.

Plain reading.
Yes: AgentDrift is nearly separable by surface content, and the separating features are templates of the generator, not general signs of hijacking. The strongest negative feature is "2026" (coefficient -9.68). It occurs in 0.742 of benign and 0.848 of failed-attack training trajectories, in 0.000 of hard negatives and in 0.046 of attacked ones. The other large negative features spell out two negative-class templates. The hard-negative template is "invoice", "5000", "to audit", "for invoice", "5000 to", "transfer 5000": each feature occurs in 0.511 to 0.830 of hard negatives and in at most 0.033 of any other category. The failed-attack poison is "within", "minutes", "10 minutes", "to avoid", "security alert", "immediately", "within 10": each occurs in 0.636 to 0.990 of failed attacks and in at most 0.036 of any other category. The large positive features spell out the successful-injection template, "correction", "audit com", "ext", "ext audit", "compliance review", "to compliance", "review ext", "must", "must be": each occurs in 0.169 to 0.436 of attacked trajectories and in at most 0.008 of any other category (Table 11). No single feature separates the classes: the best single-token rule reaches a test AUROC of 0.780. A linear combination of the 50,000 features does separate them almost perfectly (training AUROC 0.99997). This supports reporting near-trivial content separability as an external-validity limitation of AgentDrift, as M2 action 4 asks: in-distribution AgentDrift results measure recovery of the generator's templates, and R4 shows that this recovery does not carry to real traces. It does not show that content-aware detection is trivial in deployment.

### M2.7 Do tool-name semantics carry across domains? (TF-IDF over tool names split on underscores)

The question: in R3 every held-out tool is unseen, so LR-A and the bag-of-tools block are uninformative. Do the words inside tool names, such as "send" or "delete", carry a signal across domains? TFIDF-toolname is a TF-IDF logistic regression with the plan configuration, over the tool names alone, with underscores replaced by spaces.

Table 14. TFIDF-toolname [settings.S.detectors.TFIDF-toolname.<metric>; toolname.S.coverage; paired: settings.S.paired."TFIDF-toolname - LR-B".AUROC, Holm over the five folds from families_holm.R3_folds."TFIDF-toolname - LR-B"]

| Setting | AUROC | AP@2% | Recall at θr80 | FPR at θr80 | Word-occurrence coverage | Features in >= 1% of test rows | Δ AUROC vs LR-B |
|---|---|---|---|---|---|---|---|
| R1 | 0.839 [0.821, 0.857] | 0.238 [0.194, 0.314] | 0.722 [0.692, 0.752] | 0.207 [0.179, 0.234] | 1.000 | 360 | +0.009 [-0.016, +0.033] (not detected) |
| R3_medical | 0.520 [0.494, 0.546] | 0.021 [0.019, 0.025] | 0.381 [0.351, 0.412] | 0.418 [0.387, 0.451] | 0.319 | 7 | -0.288 [-0.318, -0.257] (Holm: detected) |
| R3_banking | 0.540 [0.510, 0.569] | 0.027 [0.023, 0.037] | 0.104 [0.080, 0.128] | 0.094 [0.076, 0.113] | 0.330 | 5 | -0.315 [-0.348, -0.281] (Holm: detected) |
| R3_web | 0.587 [0.561, 0.611] | 0.024 [0.022, 0.029] | 0.027 [0.018, 0.037] | 0.061 [0.046, 0.076] | 0.271 | 9 | -0.310 [-0.338, -0.281] (Holm: detected) |
| R3_email | 0.554 [0.529, 0.580] | 0.026 [0.023, 0.031] | 0.649 [0.620, 0.677] | 0.591 [0.560, 0.624] | 0.374 | 9 | -0.238 [-0.267, -0.210] (Holm: detected) |
| R3_coding | 0.663 [0.638, 0.687] | 0.055 [0.045, 0.083] | 0.533 [0.503, 0.565] | 0.280 [0.252, 0.309] | 0.308 | 11 | -0.189 [-0.218, -0.160] (Holm: detected) |
| R4 | 0.700 [0.656, 0.739] | 0.089 [0.073, 0.132] | 0.719 [0.650, 0.782] | 0.449 [0.350, 0.547] | 0.443 | 40 | +0.218 [+0.158, +0.282] |
| LOSO_banking | 0.605 [0.439, 0.755] | 0.031 [0.020, 0.053] | 0.004 [0.000, 0.013] | 0.000 [0.000, 0.000] | 0.361 | 8 | -0.294 [-0.472, -0.129] (Holm: detected) |
| LOSO_slack | 0.449 [0.376, 0.528] | 0.055 [0.030, 0.097] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.431 | 7 | -0.252 [-0.374, -0.153] (Holm: detected) |
| LOSO_travel | 0.606 [0.501, 0.722] | 0.117 [0.046, 0.263] | 0.451 [0.318, 0.596] | 0.214 [0.073, 0.384] | 0.313 | 19 | -0.006 [-0.137, +0.107] (not detected) |
| LOSO_workspace | 0.703 [0.634, 0.775] | 0.120 [0.077, 0.459] | 0.166 [0.093, 0.254] | 0.062 [0.000, 0.155] | 0.572 | 25 | +0.058 [-0.042, +0.156] (not detected) |

Domain-matched comparison [domain_matched.domains.<d>.R1dm.TFIDF-toolname.AUROC and R3_minus_R1dm.TFIDF-toolname.AUROC]: medical R1dm 0.741 [0.690, 0.787], Δ -0.221 [-0.275, -0.164] (Holm: detected); banking R1dm 0.789 [0.731, 0.838], Δ -0.250 [-0.306, -0.186] (Holm: detected); web R1dm 0.869 [0.832, 0.904], Δ -0.282 [-0.326, -0.237] (Holm: detected); email R1dm 0.890 [0.854, 0.923], Δ -0.336 [-0.377, -0.291] (Holm: detected); coding R1dm 0.858 [0.819, 0.893], Δ -0.195 [-0.238, -0.149] (Holm: detected).

Words that occur in at least 1% of held-out rows, ranked by coefficient (the five highest) [toolname.S.top_positive_present_in_test]: R3_medical: add (+1.59), send (+1.19), submit (+0.40), search (+0.36), schedule (+0.20); R3_banking: delete (+2.43), add (+0.25), get (-0.24), check (-0.43), open (-1.04); R3_web: file (+1.28), file close (+0.81), search (+0.01), submit (-0.10), close (-0.36); R3_email: delete (+1.86), add (+1.10), search (+1.02), download (+0.73), send (+0.48); R3_coding: file (+1.11), delete (+0.58), read (+0.52), search (+0.41), create (+0.36).

Plain reading.
Only weakly. In R1 the tool-name words give an AUROC of 0.839 [0.821, 0.857], not distinguishable from LR-B (+0.009 [-0.016, +0.033] (not detected)). In R3 they give 0.520 to 0.663. Against the domain-matched R1 value that is a change of -0.195 to -0.336, detected after Holm in every fold, and TFIDF-toolname is worse than LR-B in all five folds after Holm. The reason is visible in the coverage. Only 0.271 to 0.374 of held-out tool-word occurrences were seen in training, and only 5 to 11 features occur in at least 1% of held-out rows. These are generic verbs such as get, add, send, delete, check and open. The model's strongest features are within-domain tool transitions, which never occur in the held-out domain [toolname.S.top_positive]. The threshold also fails in every fold [domain_matched.threshold_failure.TFIDF-toolname]. The hope that tool-name semantics make R3 informative is therefore not supported; this is consistent with the review's range of 0.476 to 0.679. The extras point the same way. On real traces (LOSO) the variant gives 0.449 to 0.703. In R4 it gives 0.700 [0.656, 0.739], better than chance but with a poor operating point (recall 0.719, FPR 0.449 at θr80). In R4, 7 of 67 AgentDojo tool names and 25 of 88 tool words occur in AgentDrift training. That is an exploratory observation, not a tested claim.

### M2.8 Which conclusions survive for the content detector

| v0.3 conclusion (content-blind detectors) | Content detector (TFIDF) | Evidence |
|---|---|---|
| In-distribution AUROC is moderate (LR-C 0.879 in R1) | Not for content: AUROC 0.998 to 1.000 in R1, R1-wd and R2 | Tables 1, 4 |
| R3 degrades AUROC | Detected but small changes against domain-matched R1 (-0.004 to -0.018) | Table 8 |
| R3 FPR explosion at θr80 | Absent: FPR 0 in every fold | Tables 1, 7 |
| R3 threshold failure | Present in 5 of 5 folds (Holm), in the recall direction | Table 7 |
| R3 ranking reversal | Absent: TFIDF beats LR-B and LR-C in every domain, in R1dm and in R3 (0 and 0 reversals) | M2.4 |
| R4 transfer fails; within-AgentDojo works | Holds: transfer AUROC 0.520 (interval includes 0.5); within near perfect but template-driven | Tables 1, 10; M2.5 |
| Real-trace domain shift (LOSO) | AUROC 0.989 to 0.998 in all suites, FPR 0 at θr80; recall at θr80 banking 0.921, slack 0.950, travel 0.489, workspace 0.263 | Table 1 |

## m13. Benchmark provenance and documented regularities: template concentration

The question: are the README's statements about template concentration true of the released data, and how do they bear on the content separability in M2? AgentDrift README (SHA-256 395266eb9cb2ad8d...). Line numbers [separability.readme_template_claims.readme_lines]: generator named at line 24; world-identity leakage at lines 146 to 155, with the anonymisation recommendation starting at line 151; template concentration at lines 156 to 158 (the ext-audit.com claim at line 156, the hard-negative claim at lines 156 to 157, the masking advice at lines 157 to 158); banking cut-off at lines 164 to 166; generation prompts and scripts not released at lines 177 to 178; the e-mail regex and known_sets at lines 112 and 117 to 119.

Table 15. README claims against the released records. The full corpus is 12,536 trajectories (train, val and test); the analysis pool is the 10,656 used in every setting [separability.readme_template_claims].

| Claim (README line) | README | Our operationalisation | Full corpus | Analysis pool |
|---|---|---|---|---|
| Poisoned observations naming ext-audit.com (156) | 0.289 | domain `ext-audit.com`, all poisoned observations (n = 7036 / 5981) | 0.281 | 0.278 |
| | | substring `ext-audit` (also .net, -team.com), all poisoned observations | 0.288 | 0.285 |
| | | substring `ext-audit`, trajectory level (attacked + failed) | 0.292 | 0.289 |
| | | domain, successful-injection observations only | 0.357 | 0.353 |
| | | domain, failed-injection observations only | 0.000 | 0.000 |
| Hard negatives mentioning a $5,000 amount, an in-domain audit@ address or a transfer (156-157) | 0.75 | any of the three | 0.985 | 0.985 |
| | | at least two of the three | 0.864 | 0.867 |
| | | all three | 0.509 | 0.511 |
| | | $5,000 amount / in-domain audit@ / "transfer" | 0.829 / 0.681 / 0.848 | 0.831 / 0.682 / 0.850 |
| Same markers in benign trajectories (comparison) | n/a | any of the three; in-domain audit@ | 0.161; 0.000 | 0.164; 0.000 |
| Same markers in attacked / failed trajectories | n/a | any of the three | 0.144 / 0.299 | 0.143 / 0.299 |

Per-fold hijack-pattern composition of the R3 pool-E positives (full / partial / delayed) [settings.R3_<d>.pattern_composition_poolE_pos]: medical 618 / 247 / 155; banking 217 / 260 / 149; web 607 / 263 / 150; email 605 / 264 / 151; coding 597 / 259 / 164. The review's counts (lines 789 to 791) are banking 217 / 260 / 149, medical 618 / 247 / 155 and email 605 / 264 / 151, equal to ours.

Plain reading.
Both README regularities are real, and the second is stronger than stated. For ext-audit.com, the README gives 0.289. On the full corpus, the four definitions we tried give 0.281 to 0.292: 0.281 for the exact domain and 0.288 for the substring, over all 7,036 poisoned observations. So the claim is reproduced approximately, not exactly. The README does not say that the address is confined to successful injections: it appears in 0.357 of successful-injection observations and in none of the failed-injection ones. So it is a class-specific template, and TF-IDF uses it ("audit com", "ext" and "ext audit" are among the largest positive coefficients, Table 11). For the hard negatives, 0.985 contain at least one of the three markers, 0.864 at least two and 0.509 all three, against 0.161 of benign trajectories for any of the three. No in-domain audit@ address occurs in any benign trajectory. The README's "75%" is therefore a lower bound under the "any" reading and does not match any definition we tried exactly. The README does not define the markers, so this is a discrepancy of definition, not evidence of an error. Beyond the README, the unsuccessful-attack poison has its own template ("within", "minutes", "10 minutes", "to avoid", "security alert", "immediately", "within 10", Table 11), and the token "2026" separates benign and failed-attack text from hard-negative text (Table 12). Taken together, each of the four categories carries a recognisable surface template. This is consistent with a bag-of-words model being near perfect in distribution and still holding up across AgentDrift domains, which share the generator's templates. The README's own advice to anonymise and mask amounts (lines 151-153, 157-158) is not enough to remove the signal: TFIDF-anon-nodigit keeps an R1 AUROC of 0.999 [0.998, 0.999] (M2.2). What these numbers do not establish is how the templates arose. The generation prompts are not released (README lines 177 to 178).

## Deviations, additions and limitations (logged in revision/DEVIATIONS.md, C1 to C10)

- C1. Resampling for R4 transfer and LOSO uses suite|user_task clusters. Plan 3.5 names clusters only for the UT and IT analyses. Trajectory-level intervals are given as sensitivity for R2 and the AgentDojo settings, and world-cluster intervals for R1, R1-wd and R3 [settings.S.sensitivity].
- C2. In WUT, WUT_F and WIT the point is the mean over the 5 repetitions. Each bootstrap replicate draws one repetition, shared by every detector, so paired differences use the same fold models.
- C3. Added detectors that are not in the plan: TFIDF-toolname (requested by this task; R1 and R3, plus R4 and LOSO as extras), and TFIDF-anon and TFIDF-anon-nodigit (README-style anonymisation; a sensitivity analysis stronger than the plan's world mask). WUT_F is an extra setting for TFIDF.
- C4. Not reported here: plan 3.8's TPR at FPR budgets of 1% and 5% set on training negatives, because the cache holds no training scores for the text detectors; the plan 3.1 secondary subsampling AP; and cluster bootstraps for the domain-matched differences (trajectory level, as plan 3.5 prescribes for R1 and R3).
- C5. Bootstrap p-values are two-sided: 2 x min(P(d <= 0), P(d >= 0)). Holm is applied within the families stated in each table. The within-minus-transfer family covers the content detectors only.
- C6. Extras: precision at 2% and 10% implied at θr80 (Eq. 6) [settings.S.detectors.D.prec02_t_r80], FPR at a fixed test TPR of 0.8 (a test-oracle operating point, plan 3.8) [FPR_at_testTPR80], and exact binomial upper bounds where no negative is flagged [zero_fp_bounds].
- C7. The README claims are checked under our own operationalisations, because the README defines none of its markers (separability.readme_template_claims.definitions).
- C8. "2026" is counted both as the TF-IDF analyzer token and as a substring. The review's 75.6% matches the substring share on the analysis pool.
- C9. TFIDF-toolname joins the steps with " || ", as TFIDF does, and the tokenizer drops the separator. Bigrams therefore span consecutive tools, and these transition bigrams are the model's strongest features.
- C10. Point estimates are full-sample values. evalkit.ci reports the bootstrap mean instead, so values here can differ from evalkit-style tables in the third decimal.
- Limitation: this is one bag-of-words detector. The review also asked for a frozen sentence encoder and a released component (PromptGuard 2, or an AlignmentCheck-style monitor); neither was run in this stream.
