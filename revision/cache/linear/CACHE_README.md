# Linear prediction cache (revision)

Built by `revision/build_linear_cache.py` (script SHA-256 `a4a4997543387841e271af9002baa9649b3311939c2870029c5ac371086d473f`), validated by `revision/validate_linear_cache.py` on 2026-09-26T21:57:20+0200. Every number below is read from `cache/linear/validation.json` (key given in brackets where useful). Scores are for the 20 settings of REVISION_ANALYSIS_PLAN.md Section 2; nothing was tuned on test data (all thresholds come from the training folds).

Overall: format checks all pass; in-process reproduction checks against the v0.3 code paths: 105 of 105 pass (largest score difference 3.9e-08); requested anchors reproduced: 16 of 16.

## Files

- `<setting>.pkl`: one dict per setting in the shared cache format (keys `setting`, `family`, `test_meta`, `y`, `pool_E`, `scores`, `pred`, `thr`, `info`, plus `lrc_parts` and `coefs` for single-fit settings and `folds`, `cov_reps` for cross-validated ones, `detectors`, `provenance`).
- `_parts/`: one file per CV repetition of WUT, WUT_F, WIT (assembled into the pickles).
- `_validation/<setting>.json`, `validation.json`: the numbers in this file.
- `r4_truncation_check.json`: diagnostic for the R4 values (from `revision/linear_r4_truncation_check.py`).
- `build_log.txt`: build log.

Row order within each setting is identical to the GRU cache (`cache/gru/<setting>.pkl`) wherever that pickle exists (checked: `settings.<s>.format_checks.row_order_equals_gru_cache`).

## Detectors

| Detector | Definition |
|---|---|
| LR-A, LR-B, LR-C | as ablation.py: StandardScaler on training rows, LogisticRegression(max_iter=1000, class_weight='balanced'); A = bag of tools (vocabulary from benign training trajectories), B = 10 behavioural features, C = B + A |
| LR-B-noobs, LR-C-noobs | B or C without behavioural features 8, 9 (max and mean observation length) |
| LR-B-minus | B without behavioural features 4, 5 (keyword state-changing count and fraction) |
| LR-C-drop | LR-C fitted on [X; X with the tool block zeroed], labels duplicated, scaler on the stacked matrix; t_r80 from the original training positives |
| Markov | baseline_drift.Markov fitted on benign training tool sequences, as rerun_r1r3.py |
| IForest | IsolationForest(200 trees, random_state=0) on standardized C features of benign training rows, as rerun_r1r3.py |
| TFIDF | TfidfVectorizer(lowercase, (1,2)-grams, min_df=2, max_features=50000, sublinear_tf) on training texts + LogisticRegression(max_iter=2000, class_weight='balanced', C=1); text = ' \|\| '.join(f'{tool} {str(args)} {obs}') |
| TFIDF-nodigit | TFIDF with all digits removed from the text |
| TFIDF-worldmask (R1 only) | TFIDF with every string value of the trajectory's world dict replaced by WORLDTOKEN |

Thresholds: `t_r80` = evalkit.thr_at_recall(in-sample training-positive scores, 0.80) (for CV settings one value per repetition and fold, `thr[d]['t_r80']` has shape (5, 5) and `folds` gives each row's fold); `t05` = 0.5 (a probability cut for LR-*/TFIDF*, sklearn's default outlier cut for IForest, and not a calibrated cut for Markov, whose score is a mean negative log-probability).

## Settings and sizes

| Setting | Scheme | n_train (pos) | Test pool E pos / neg | Failed attacks scored | Vocab | kappa |
|---|---|---|---|---|---|---|
| R1 | single | 8775 (3875) | 831 / 825 | 225 | 103 | 1.000 |
| R1wd_s0 | single | 8762 (3841) | 865 / 786 | 243 | 103 | 1.000 |
| R1wd_s1 | single | 8739 (3782) | 924 / 787 | 206 | 103 | 1.000 |
| R1wd_s2 | single | 8711 (3807) | 899 / 802 | 244 | 103 | 1.000 |
| R1wd_s3 | single | 8752 (3843) | 863 / 811 | 230 | 103 | 1.000 |
| R1wd_s4 | single | 8764 (3813) | 893 / 788 | 211 | 103 | 1.000 |
| R2 | single | 9081 (4008) | 775 / 729 | 218 | 103 | 1.000 |
| R3_medical | single | 8446 (3686) | 1020 / 935 | 255 | 83 | 0.000 |
| R3_banking | single | 8840 (4080) | 626 / 935 | 255 | 83 | 0.000 |
| R3_web | single | 8446 (3686) | 1020 / 935 | 255 | 83 | 0.000 |
| R3_email | single | 8446 (3686) | 1020 / 935 | 255 | 83 | 0.000 |
| R3_coding | single | 8446 (3686) | 1020 / 935 | 255 | 80 | 0.000 |
| R4 | single | 10656 (4706) | 2294 / 392 | 4987 | 103 | 0.075 |
| WUT | UT | 2148.8 (1835.2) mean | 2294 / 392 | 4987 | 57-64 | 0.976 |
| WUT_F | UT | 6138.4 (1835.2) mean | 2294 / 392 | 4987 | 57-64 | 0.976 |
| WIT | IT | 2148.8 (1835.2) mean | 2294 / 392 | 4987 | 59-63 | 0.980 |
| LOSO_banking | single | 1959 (1623) | 671 / 56 | 723 | 53 | 0.000 |
| LOSO_slack | single | 1677 (1390) | 904 / 105 | 592 | 53 | 0.000 |
| LOSO_travel | single | 2401 (2112) | 182 / 103 | 1798 | 44 | 0.084 |
| LOSO_workspace | single | 2021 (1757) | 537 / 128 | 1874 | 45 | 0.362 |

kappa = tool-occurrence coverage of the pool-E test rows by the benign training vocabulary (CV settings: mean over repetitions and folds). For CV settings n_train is the mean over the 25 fits.

## Reproduction anchors (v0.3)

AUROC on pool E, point estimate with 95% stratified bootstrap interval [`anchors`].

| Setting | Detector | v0.3 | Reproduced | Match (3 dp) |
|---|---|---|---|---|
| R1 | LR-A | 0.777 | 0.7774 [0.753, 0.799] | yes |
| R1 | LR-B | 0.830 | 0.8303 [0.812, 0.849] | yes |
| R1 | LR-C | 0.879 | 0.8788 [0.863, 0.895] | yes |
| R1 | Markov | 0.724 | 0.7238 [0.699, 0.748] | yes |
| R1 | IForest | 0.588 | 0.5877 [0.559, 0.614] | yes |
| R3_medical | LR-C | 0.745 | 0.7452 [0.724, 0.765] | yes |
| R3_banking | LR-C | 0.828 | 0.8283 [0.808, 0.849] | yes |
| R3_web | LR-C | 0.862 | 0.8623 [0.847, 0.876] | yes |
| R3_email | LR-C | 0.739 | 0.7393 [0.718, 0.760] | yes |
| R3_coding | LR-C | 0.814 | 0.8141 [0.795, 0.834] | yes |
| R3_medical | LR-B | 0.808 | 0.8079 [0.789, 0.827] | yes |
| R3_banking | LR-B | 0.854 | 0.8543 [0.835, 0.873] | yes |
| R3_web | LR-B | 0.896 | 0.8963 [0.881, 0.909] | yes |
| R3_email | LR-B | 0.792 | 0.7923 [0.773, 0.811] | yes |
| R3_coding | LR-B | 0.852 | 0.8518 [0.836, 0.868] | yes |
| R2 | LR-C | 0.902 | 0.9019 [0.886, 0.916] | yes |

Additional anchors (not requested): 31 of 34 equal the archived v0.3 value to 4 decimals (R2 and R3 Markov/IForest, R3 LR-A, R1-wd LR-A/B/C for 5 split seeds, R4 LR-A transfer and within-UT). The exceptions are listed here:

- R3_coding Markov (AUROC): v0.3 0.4995, cache 0.5000.
- R3_email LR-A (AUROC): v0.3 0.4989, cache 0.5000.
- R3_coding LR-A (AUROC): v0.3 0.4996, cache 0.5000.

These exceptions are tied scores: when every test trajectory has an unseen tool vocabulary, LR-A (and Markov) give all rows the same score mathematically, but the v0.3 code broke the tie with float noise (values such as 0.4989). The cache makes such ties bitwise exact, so the AUROC is exactly 0.5. The noisy value is not even stable across runs of the same v0.3 code: re-running it today gives the values in the next table, which differ from the archived ones.

The same float noise matters much more for Markov at low coverage, because its mean of identical log-probabilities differs in the last bits with trajectory length, so the ties are broken in a systematic, not random, order (in LOSO_slack every row is tied mathematically). AUROC on pool E, v0.3 code path versus tie-exact cache [`tie_check`] (settings with a difference of at least 0.0005):

| Setting | Detector | v0.3 code path (raw) | Cache (tie-exact) |
|---|---|---|---|
| R3_web | LR-A | 0.4990 | 0.5000 |
| R3_email | LR-A | 0.4995 | 0.5000 |
| R3_coding | Markov | 0.4995 | 0.5000 |
| R3_coding | LR-A | 0.4990 | 0.5000 |
| R4 | Markov | 0.5082 | 0.5902 |
| LOSO_banking | Markov | 0.5291 | 0.5000 |
| LOSO_slack | Markov | 0.6636 | 0.5000 |
| LOSO_travel | Markov | 0.4237 | 0.5498 |
| LOSO_workspace | Markov | 0.5138 | 0.4976 |

## R4 with the corrected AgentDojo extraction

These values differ from v0.3 because the extraction was corrected. Population: P0 (unsuccessful attacks are scored but excluded from AUROC) [`r4.transfer_R4`, `r4.withinUT_WUT`, `r4.withinIT_WIT`, `r4.withinUT_failed_as_negatives_WUT_F`].

| Detector | Transfer (R4) | Within-UT (WUT) | Within-UT, failed as negatives (WUT_F) | Within-IT (WIT) |
|---|---|---|---|---|
| LR-A | 0.566 [0.543, 0.588] | 0.788 [0.759, 0.815] | 0.790 [0.763, 0.817] | 0.690 [0.617, 0.767] |
| LR-B | 0.482 [0.449, 0.515] | 0.741 [0.709, 0.771] | 0.720 [0.687, 0.752] | 0.719 [0.674, 0.759] |
| LR-C | 0.473 [0.441, 0.506] | 0.797 [0.765, 0.831] | 0.794 [0.763, 0.828] | 0.688 [0.636, 0.737] |
| LR-B-noobs | 0.630 [0.597, 0.663] | 0.740 [0.712, 0.769] | 0.727 [0.696, 0.755] | 0.718 [0.675, 0.758] |
| LR-C-noobs | 0.657 [0.625, 0.688] | 0.795 [0.760, 0.827] | 0.799 [0.771, 0.828] | 0.701 [0.653, 0.767] |
| LR-B-minus | 0.478 [0.445, 0.512] | 0.701 [0.671, 0.730] | 0.682 [0.644, 0.718] | 0.669 [0.624, 0.715] |
| LR-C-drop | 0.475 [0.443, 0.509] | 0.781 [0.751, 0.814] | 0.784 [0.757, 0.811] | 0.666 [0.586, 0.726] |
| Markov | 0.590 [0.572, 0.608] | 0.575 [0.532, 0.618] | 0.575 [0.532, 0.619] | 0.564 [0.528, 0.601] |
| IForest | 0.647 [0.614, 0.680] | 0.692 [0.661, 0.721] | 0.692 [0.662, 0.722] | 0.690 [0.650, 0.722] |
| TFIDF | 0.520 [0.487, 0.552] | 0.996 [0.993, 0.998] | 0.941 [0.927, 0.955] | 0.987 [0.982, 0.991] |
| TFIDF-nodigit | 0.470 [0.442, 0.498] | 0.996 [0.994, 0.998] | 0.942 [0.928, 0.955] | 0.988 [0.984, 0.992] |

The LR-A/B/C point values equal the re-run v0.3 scripts (ablation.py, r4_control.py on the corrected data) to 4 decimals: 8 of 8 [`r4.cache_equals_rerun_v03_point_4dp`]. The within-UT intervals here come from this validator's bootstrap, so they differ slightly from ablation_results.json.

The review's corrected estimates used positional pairing but still truncated observations at 2,000 characters. The corrected file also removes that truncation (plan item 1.2). Re-applying the truncation to the corrected records and re-running the unmodified r4_control code paths (`r4_truncation_check.json`; 300 P0 traces have an observation longer than 2,000 characters):

| Quantity | Review | Corrected, truncated at 2,000 | Corrected, untruncated (this cache) |
|---|---|---|---|
| transfer_LR-C | 0.476 | 0.476 [0.444, 0.509] | 0.473 [0.442, 0.506] |
| transfer_LR-B | 0.483 | 0.483 [0.451, 0.516] | 0.482 [0.450, 0.515] |
| withinUT_LR-C | 0.846 | 0.846 [0.820, 0.873] | 0.797 [0.766, 0.832] |
| withinUT_LR-B | 0.773 | 0.773 [0.744, 0.803] | 0.741 [0.709, 0.773] |

So the review's numbers are reproduced once truncation is re-applied. Removing the truncation leaves transfer near 0.5 and lowers the within-AgentDojo AUROC of LR-C and LR-B.

## AUROC by setting (pool E)

Point estimate with 95% stratified bootstrap interval [`settings.<s>.metrics.<detector>.AUROC`]. The file also gives AUPRC, recall and FPR at t_r80 and t05, and failed-attack flag rates.

| Setting | LR-A | LR-B | LR-C | LR-C-drop | LR-B-noobs | LR-C-noobs | LR-B-minus | Markov | IForest | TFIDF | TFIDF-nodigit |
|---|---|---|---|---|---|---|---|---|---|---|---|
| R1 | 0.777 [0.753, 0.799] | 0.830 [0.812, 0.849] | 0.879 [0.863, 0.895] | 0.882 [0.866, 0.898] | 0.699 [0.674, 0.723] | 0.805 [0.784, 0.825] | 0.827 [0.808, 0.846] | 0.724 [0.699, 0.748] | 0.588 [0.559, 0.614] | 0.999 [0.999, 1.000] | 0.999 [0.997, 0.999] |
| R1wd_s0 | 0.821 [0.801, 0.840] | 0.821 [0.801, 0.840] | 0.891 [0.875, 0.906] | 0.894 [0.879, 0.908] | 0.692 [0.666, 0.718] | 0.844 [0.825, 0.861] | 0.820 [0.801, 0.839] | 0.762 [0.738, 0.785] | 0.563 [0.538, 0.590] | 1.000 [0.999, 1.000] | 0.999 [0.999, 1.000] |
| R1wd_s1 | 0.822 [0.800, 0.840] | 0.803 [0.783, 0.823] | 0.889 [0.873, 0.903] | 0.896 [0.880, 0.909] | 0.671 [0.645, 0.696] | 0.850 [0.831, 0.867] | 0.804 [0.784, 0.824] | 0.756 [0.731, 0.778] | 0.536 [0.508, 0.564] | 1.000 [0.999, 1.000] | 1.000 [0.999, 1.000] |
| R1wd_s2 | 0.823 [0.804, 0.842] | 0.822 [0.803, 0.842] | 0.901 [0.887, 0.916] | 0.905 [0.891, 0.919] | 0.696 [0.673, 0.722] | 0.852 [0.833, 0.870] | 0.826 [0.807, 0.845] | 0.769 [0.745, 0.793] | 0.572 [0.544, 0.599] | 1.000 [1.000, 1.000] | 0.999 [0.999, 1.000] |
| R1wd_s3 | 0.822 [0.802, 0.841] | 0.818 [0.797, 0.836] | 0.888 [0.871, 0.903] | 0.893 [0.877, 0.907] | 0.702 [0.677, 0.726] | 0.848 [0.829, 0.866] | 0.819 [0.797, 0.837] | 0.752 [0.728, 0.776] | 0.575 [0.547, 0.601] | 1.000 [0.999, 1.000] | 0.999 [0.998, 1.000] |
| R1wd_s4 | 0.817 [0.797, 0.834] | 0.819 [0.800, 0.839] | 0.892 [0.876, 0.906] | 0.897 [0.881, 0.911] | 0.696 [0.671, 0.722] | 0.845 [0.827, 0.862] | 0.820 [0.802, 0.840] | 0.742 [0.719, 0.766] | 0.545 [0.520, 0.574] | 1.000 [0.999, 1.000] | 1.000 [0.999, 1.000] |
| R2 | 0.788 [0.765, 0.809] | 0.849 [0.829, 0.866] | 0.902 [0.886, 0.916] | 0.904 [0.889, 0.918] | 0.714 [0.687, 0.738] | 0.821 [0.801, 0.840] | 0.841 [0.821, 0.860] | 0.652 [0.623, 0.681] | 0.558 [0.529, 0.590] | 0.998 [0.997, 0.999] | 0.995 [0.993, 0.997] |
| R3_medical | 0.500 [0.500, 0.500] | 0.808 [0.789, 0.827] | 0.745 [0.724, 0.765] | 0.797 [0.778, 0.816] | 0.629 [0.604, 0.652] | 0.588 [0.562, 0.612] | 0.803 [0.784, 0.821] | 0.500 [0.500, 0.500] | 0.605 [0.579, 0.629] | 0.996 [0.994, 0.997] | 0.980 [0.975, 0.984] |
| R3_banking | 0.500 [0.500, 0.500] | 0.854 [0.835, 0.873] | 0.828 [0.808, 0.849] | 0.853 [0.833, 0.872] | 0.666 [0.639, 0.693] | 0.645 [0.617, 0.673] | 0.839 [0.818, 0.859] | 0.500 [0.500, 0.500] | 0.644 [0.619, 0.672] | 0.989 [0.985, 0.992] | 0.943 [0.931, 0.954] |
| R3_web | 0.500 [0.500, 0.500] | 0.896 [0.881, 0.909] | 0.862 [0.847, 0.876] | 0.894 [0.879, 0.907] | 0.675 [0.649, 0.698] | 0.672 [0.648, 0.695] | 0.899 [0.885, 0.911] | 0.500 [0.500, 0.500] | 0.654 [0.630, 0.677] | 0.980 [0.976, 0.984] | 0.969 [0.962, 0.975] |
| R3_email | 0.500 [0.500, 0.500] | 0.792 [0.773, 0.811] | 0.739 [0.718, 0.760] | 0.777 [0.757, 0.796] | 0.680 [0.656, 0.704] | 0.621 [0.596, 0.645] | 0.807 [0.788, 0.825] | 0.500 [0.500, 0.500] | 0.628 [0.603, 0.650] | 0.990 [0.987, 0.993] | 0.974 [0.967, 0.980] |
| R3_coding | 0.500 [0.500, 0.500] | 0.852 [0.836, 0.868] | 0.814 [0.795, 0.834] | 0.847 [0.830, 0.863] | 0.740 [0.719, 0.762] | 0.682 [0.659, 0.707] | 0.849 [0.833, 0.865] | 0.500 [0.500, 0.500] | 0.627 [0.603, 0.654] | 0.987 [0.983, 0.990] | 0.958 [0.950, 0.966] |
| R4 | 0.566 [0.543, 0.588] | 0.482 [0.449, 0.515] | 0.473 [0.441, 0.506] | 0.475 [0.443, 0.509] | 0.630 [0.597, 0.663] | 0.657 [0.625, 0.688] | 0.478 [0.445, 0.512] | 0.590 [0.572, 0.608] | 0.647 [0.614, 0.680] | 0.520 [0.487, 0.552] | 0.470 [0.442, 0.498] |
| WUT | 0.788 [0.759, 0.815] | 0.741 [0.709, 0.771] | 0.797 [0.765, 0.831] | 0.781 [0.751, 0.814] | 0.740 [0.712, 0.769] | 0.795 [0.760, 0.827] | 0.701 [0.671, 0.730] | 0.575 [0.532, 0.618] | 0.692 [0.661, 0.721] | 0.996 [0.993, 0.998] | 0.996 [0.994, 0.998] |
| WUT_F | 0.790 [0.763, 0.817] | 0.720 [0.687, 0.752] | 0.794 [0.763, 0.828] | 0.784 [0.757, 0.811] | 0.727 [0.696, 0.755] | 0.799 [0.771, 0.828] | 0.682 [0.644, 0.718] | 0.575 [0.532, 0.619] | 0.692 [0.662, 0.722] | 0.941 [0.927, 0.955] | 0.942 [0.928, 0.955] |
| WIT | 0.690 [0.617, 0.767] | 0.719 [0.674, 0.759] | 0.688 [0.636, 0.737] | 0.666 [0.586, 0.726] | 0.718 [0.675, 0.758] | 0.701 [0.653, 0.767] | 0.669 [0.624, 0.715] | 0.564 [0.528, 0.601] | 0.690 [0.650, 0.722] | 0.987 [0.982, 0.991] | 0.988 [0.984, 0.992] |
| LOSO_banking | 0.500 [0.500, 0.500] | 0.898 [0.846, 0.940] | 0.840 [0.788, 0.884] | 0.913 [0.863, 0.953] | 0.878 [0.826, 0.920] | 0.827 [0.775, 0.873] | 0.858 [0.813, 0.898] | 0.500 [0.500, 0.500] | 0.767 [0.700, 0.825] | 0.998 [0.994, 1.000] | 0.996 [0.992, 0.999] |
| LOSO_slack | 0.500 [0.500, 0.500] | 0.701 [0.644, 0.756] | 0.744 [0.689, 0.799] | 0.720 [0.663, 0.776] | 0.700 [0.643, 0.755] | 0.742 [0.687, 0.796] | 0.698 [0.643, 0.751] | 0.500 [0.500, 0.500] | 0.690 [0.628, 0.750] | 0.998 [0.996, 1.000] | 0.999 [0.996, 1.000] |
| LOSO_travel | 0.653 [0.596, 0.711] | 0.613 [0.541, 0.681] | 0.661 [0.595, 0.726] | 0.626 [0.555, 0.692] | 0.599 [0.526, 0.668] | 0.649 [0.583, 0.716] | 0.620 [0.553, 0.689] | 0.550 [0.526, 0.574] | 0.708 [0.645, 0.772] | 0.989 [0.973, 1.000] | 0.989 [0.973, 1.000] |
| LOSO_workspace | 0.703 [0.658, 0.746] | 0.644 [0.581, 0.704] | 0.643 [0.580, 0.707] | 0.640 [0.577, 0.700] | 0.717 [0.669, 0.763] | 0.779 [0.734, 0.823] | 0.616 [0.554, 0.675] | 0.498 [0.452, 0.545] | 0.682 [0.633, 0.730] | 0.991 [0.982, 0.998] | 0.991 [0.982, 0.998] |

R1 TFIDF-worldmask: 0.999 [0.999, 1.000].

## Implementation notes (also logged in revision/DEVIATIONS.md)

- Fitting follows ablation.py / rerun_r1r3.py exactly; the cache only makes mathematically tied scores bitwise tied (LR-*, IForest and TFIDF* are scored once per distinct feature row or text and broadcast; Markov scores are rounded to 12 decimals, since a mean of identical log-probabilities depends on the trajectory length in floating point). Scores differ from the v0.3 code paths by at most 3.9e-08.
- Unsuccessful AgentDojo attacks are placed in the fold of their suite|user_task group. Groups that no P0 row defines are assigned with np.random.default_rng(10000 + seed), position mod 5: 1 group (1 row) in WUT and WUT_F; 11 groups (64 rows) in WIT, where P0 attacks are grouped by injection goal, so user tasks with no benign P0 run have no P0 group.
- WUT_F uses the WUT fold assignment; the unsuccessful attacks of the four training folds are added as label-0 training rows (they do not enter the benign tool vocabulary, the Markov model or the IForest).
- TFIDF-worldmask replaces whole world string values (user, email, company, contact name, email and relation, date), case-insensitively, as whole occurrences. Mean replacements per trajectory 2.00; 0.441 of trajectories contain no world string; a first- or last-name token of the user or a contact survives in 0.198 of trajectories (partial names are not world values).
- In the AgentDrift files `args` is a dict, not a string; the text uses str(args) in both benchmarks, and the v0.3 argument-size features (len(args)) therefore count argument keys in both.
- Coverage `cov` in test_meta for CV settings is the mean over repetitions (per-repetition values in `cov_reps`).
- Metric intervals in this file are trajectory-level stratified bootstraps (v0.3 style). The plan's cluster bootstraps and prevalence-weighted AUPRC are left to the downstream analyses.

## Problems and caveats

- Float noise in tied scores changes AUROC in the v0.3 code paths when coverage is low (tie_check.json; cache values are tie-exact): R3_web LR-A: v0.3 code path 0.4990, cache 0.5000; R3_email LR-A: v0.3 code path 0.4995, cache 0.5000; R3_coding Markov: v0.3 code path 0.4995, cache 0.5000; R3_coding LR-A: v0.3 code path 0.4990, cache 0.5000; R4 Markov: v0.3 code path 0.5082, cache 0.5902; LOSO_banking Markov: v0.3 code path 0.5291, cache 0.5000; LOSO_slack Markov: v0.3 code path 0.6636, cache 0.5000; LOSO_travel Markov: v0.3 code path 0.4237, cache 0.5498; LOSO_workspace Markov: v0.3 code path 0.5138, cache 0.4976.
- R4 values do not match the review's corrected estimates because the corrected extraction also removes the 2,000-character truncation (see the truncation check); they are not a reproduction failure.
- TFIDF reaches AUROC near 1 in R1 (0.999 [0.999, 1.000]) and TFIDF-worldmask barely changes it (0.999 [0.999, 1.000]); the content detector separates the AgentDrift classes almost perfectly in-distribution.
- When the held-out tool vocabulary is disjoint (R3 folds, LOSO_banking, LOSO_slack) LR-A and Markov give every test row the same score: AUROC is exactly 0.5, LR-A flags no trajectory at t_r80 or t05 (recall 0, FPR 0) and Markov flags every trajectory at t_r80 (recall 1, FPR 1) [settings.<s>.metrics.<d>.recall@t_r80 / FPR@t_r80].
