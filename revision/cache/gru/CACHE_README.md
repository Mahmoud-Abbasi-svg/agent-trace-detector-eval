# GRU prediction cache (family `gru`)

Generated 2026-09-26T21:19:54 by `revision/summarize_gru_cache.py` from the cache written by `revision/build_gru_cache.py` (training function in `revision/gru_variants.py`). Every number below is a key of `revision/gru_cache_results.json` (paths given in brackets).

## What is in the cache

One pickle per setting, `cache/gru/<setting>.pkl`, in the shared cache format (`setting`, `test_meta`, `y`, `pool_E`, `scores`, `pred`, `thr`, `info`). Detectors: `GRU` (token dropout p = 0, the pre-registered configuration), `GRU-drop0.2` (primary), `GRU-drop0.1` (sensitivity). Architecture, constants, optimiser, early stopping, vocabulary (benign training trajectories only) and seeds are those of `gru_detector.py`/`gru_ladder.py`, imported unmodified.

- Scores: single-fit settings `scores[det]` has shape (3, n_test), training seeds 0, 1, 2 (row r = seed r). WUT: (5, n_test) out-of-fold scores, repetition r uses fold-assignment seed r and training seed r (as `gru_ladder.py`).
- Validation (early stopping, `t_val80`): `gru_ladder.stratified_val` (10% of the setting's training set, stratified, rng seed 1000 + training seed), for every setting including R1 (as v0.3 `gru_ladder.py`; `data/val.jsonl` is not used).
- `pred[det]`: `t_r80` (20th percentile of in-sample training-positive scores, from `train_gru`), `t_val80` (20th percentile of validation-positive scores), `t05`; same shape as scores. `thr[det]` holds the values (single fit: arrays over seeds; WUT: arrays (rep, fold)).
- Test rows always include the unsuccessful attacks (AgentDrift `failed_attack`; AgentDojo `agentdojo_failed.jsonl`, category `attack_failed`), flagged `in_pool_E = False`; they are scored, never fitted.
- `test_meta[i]['cov']`: fraction of the trajectory's tool calls whose tool is in the benign tool vocabulary of the setting's full training set (identical to the linear cache). `cov_model[det]` uses the vocabulary the network actually had (90% fit partition). WUT: `cov` is the mean over repetitions of the out-of-fold value; `cov_by_rep` per repetition.
- Mechanism fields (single fit, per detector, per seed): `tool_contrib` (head weights over the 64 GRU hidden units dotted with the final hidden state), `beh_contrib` (head weights over the 10 standardised behavioural features dotted with them), `bias`, `logit` (= tool + beh + bias; max reconstruction error per unit in `train_diag`); `train_tool_contrib`/`train_beh_contrib` with `train_y`, `train_category`, `train_domain` on the fit partition (validation rows excluded); `unk_norm`, `unk_init_norm`, `tool_emb_norms` (+ `tool_emb_names`), `unk_train_steps` (UNK tokens among the encoded training steps, before dropout). WUT: `tool_contrib_oof`, `unk_norm`, `unk_train_steps`, `unk_train_traces(_pos)` as (rep, fold) arrays; per-unit training summaries in `_parts/WUT/GRU/`.
- `train_diag[det]`: one dict per unit (epochs run, best epoch, validation AP curve, thresholds, realised dropout rate, sizes, UNK counts, timings). `_parts/<setting>/<det>/` keeps every unit's raw output (resumable build).

## Settings, sizes and timings [settings.<S>, files, timing]

| setting | detectors | n_test | pool E (pos/neg) | not pool E | n_train (pos) | kappa | vocab | file MB | fits | fit s (mean) |
|---|---|---|---|---|---|---|---|---|---|---|
| R1 | GRU, GRU-drop0.2, GRU-drop0.1 | 1881 | 1656 (831/825) | 225 | 8775 (3875) | 1.0 | 103 | 7.82 | 9 | 28.2 |
| R1wd_s0 | GRU | 1894 | 1651 (865/786) | 243 | 8762 (3841) | 1.0 | 103 | 3.6 | 3 | 21.3 |
| R1wd_s1 | GRU | 1917 | 1711 (924/787) | 206 | 8739 (3782) | 1.0 | 103 | 3.63 | 3 | 25.0 |
| R1wd_s2 | GRU | 1945 | 1701 (899/802) | 244 | 8711 (3807) | 1.0 | 103 | 3.64 | 3 | 26.1 |
| R1wd_s3 | GRU | 1904 | 1674 (863/811) | 230 | 8752 (3843) | 1.0 | 103 | 3.62 | 3 | 21.7 |
| R1wd_s4 | GRU | 1892 | 1681 (893/788) | 211 | 8764 (3813) | 1.0 | 103 | 3.61 | 3 | 24.0 |
| R2 | GRU | 1722 | 1504 (775/729) | 218 | 9081 (4008) | 1.0 | 103 | 3.48 | 3 | 30.0 |
| R3_medical | GRU, GRU-drop0.2, GRU-drop0.1 | 2210 | 1955 (1020/935) | 255 | 8446 (3686) | 0.0 | 83 | 7.86 | 9 | 26.2 |
| R3_banking | GRU, GRU-drop0.2, GRU-drop0.1 | 1816 | 1561 (626/935) | 255 | 8840 (4080) | 0.0 | 83 | 7.72 | 9 | 26.9 |
| R3_web | GRU, GRU-drop0.2, GRU-drop0.1 | 2210 | 1955 (1020/935) | 255 | 8446 (3686) | 0.0 | 83 | 7.84 | 9 | 26.0 |
| R3_email | GRU, GRU-drop0.2, GRU-drop0.1 | 2210 | 1955 (1020/935) | 255 | 8446 (3686) | 0.0 | 83 | 8.07 | 9 | 25.0 |
| R3_coding | GRU, GRU-drop0.2, GRU-drop0.1 | 2210 | 1955 (1020/935) | 255 | 8446 (3686) | 0.0 | 80 | 7.92 | 9 | 25.6 |
| R4 | GRU, GRU-drop0.2, GRU-drop0.1 | 7673 | 2686 (2294/392) | 4987 | 10656 (4706) | 0.0751 | 103 | 10.81 | 9 | 28.9 |
| WUT | GRU | 7673 | 2686 (2294/392) | 4987 | 2149 (1835) | 0.9756 | 61 | 3.09 | 25 | 8.2 |

Total: 106 fits, 38.8 min of fitting (mean 22.0 s per fit; CPU, torch 2.11.0+cpu, 8 threads). WUT n_train/kappa/vocab are means over the 25 (rep, fold) units.

## Checks [checks]

- p = 0 copy vs `gru_detector.train_gru` on R1 seed 0: scores identical = True (max abs diff 0.0), parameters identical = True, threshold identical = True; batch order unchanged by dropout = True (realised p = 0.2 drop rate 0.1970).
- Cached scores equal a fresh unmodified `train_gru` refit: R1 seed 0 True, R4 seed 0 P0 True / failed True, WUT rep 0 fold 0 True (folds equal True); GRU-drop0.2 R1 seed 0 refit identical True.
- R4/WUT GRU point estimates equal those of `gru_ladder.py` re-run on the corrected extraction (`agent-ids/gru_results.json`): True.
- Same test rows, order, `test_meta`, `cov`, `y` and `pool_E` as the linear cache in every setting, and WUT folds equal the linear cache's: True (WUT: 1 suite|user_task groups occur only among failed attacks, 1 failed rows) [checks.consistency_with_linear_cache].

## Reproduction anchors (GRU, pool E AUROC, mean of 3 seeds) [anchors]

| setting | expected (v0.3) | reproduced [95% CI] | match |
|---|---|---|---|
| R1 | 0.936 (0.9356) | 0.9356 [0.923, 0.947] | True |
| R1wd_s0 | 0.956 (0.9559) | 0.9559 [0.944, 0.967] | True |
| R2 | 0.915 (0.9154) | 0.9154 [0.897, 0.934] | True |
| R3_medical | 0.639 (0.6386) | 0.6386 [0.607, 0.669] | True |
| R3_banking | 0.736 (0.7363) | 0.7363 [0.698, 0.776] | True |
| R3_web | 0.726 (0.7257) | 0.7257 [0.689, 0.767] | True |
| R3_email | 0.671 (0.6708) | 0.6708 [0.625, 0.730] | True |
| R3_coding | 0.695 (0.6948) | 0.6948 [0.653, 0.744] | True |

R1-wd across the five split seeds: 0.9559, 0.9492, 0.9619, 0.9589, 0.9561 (mean 0.9564).

## Pool-E results (point [95% stratified bootstrap CI], B = 2000) [settings.<S>.detectors.<det>]

| setting | detector | AUROC | AUPRC | recall@t_r80 | FPR@t_r80 | FPR@t_val80 | failed flagged@t_r80 |
|---|---|---|---|---|---|---|---|
| R1 | GRU | 0.936 [0.923, 0.947] | 0.936 [0.920, 0.949] | 0.682 [0.646, 0.716] | 0.045 [0.028, 0.062] | 0.073 [0.054, 0.093] | 0.043 [0.013, 0.089] |
| R1 | GRU-drop0.2 | 0.943 [0.931, 0.955] | 0.942 [0.925, 0.955] | 0.712 [0.676, 0.745] | 0.054 [0.035, 0.073] | 0.079 [0.056, 0.102] | 0.046 [0.013, 0.084] |
| R1 | GRU-drop0.1 | 0.944 [0.931, 0.954] | 0.943 [0.930, 0.956] | 0.691 [0.651, 0.727] | 0.044 [0.030, 0.062] | 0.071 [0.053, 0.092] | 0.038 [0.013, 0.071] |
| R1wd_s0 | GRU | 0.956 [0.944, 0.967] | 0.957 [0.947, 0.968] | 0.782 [0.748, 0.816] | 0.052 [0.032, 0.076] | 0.073 [0.053, 0.097] | 0.048 [0.021, 0.086] |
| R1wd_s1 | GRU | 0.949 [0.939, 0.959] | 0.952 [0.938, 0.963] | 0.723 [0.670, 0.777] | 0.047 [0.020, 0.071] | 0.061 [0.041, 0.083] | 0.063 [0.029, 0.102] |
| R1wd_s2 | GRU | 0.962 [0.953, 0.971] | 0.962 [0.950, 0.972] | 0.759 [0.723, 0.792] | 0.037 [0.024, 0.055] | 0.049 [0.034, 0.066] | 0.038 [0.016, 0.066] |
| R1wd_s3 | GRU | 0.959 [0.943, 0.970] | 0.961 [0.944, 0.974] | 0.749 [0.701, 0.794] | 0.035 [0.010, 0.059] | 0.064 [0.038, 0.100] | 0.026 [0.000, 0.070] |
| R1wd_s4 | GRU | 0.956 [0.941, 0.967] | 0.962 [0.950, 0.972] | 0.741 [0.707, 0.776] | 0.038 [0.023, 0.058] | 0.066 [0.048, 0.085] | 0.044 [0.014, 0.085] |
| R2 | GRU | 0.915 [0.897, 0.934] | 0.921 [0.899, 0.943] | 0.656 [0.619, 0.689] | 0.047 [0.026, 0.070] | 0.089 [0.048, 0.125] | 0.067 [0.037, 0.105] |
| R3_medical | GRU | 0.639 [0.607, 0.669] | 0.634 [0.600, 0.665] | 0.938 [0.896, 0.970] | 0.717 [0.641, 0.771] | 0.768 [0.717, 0.824] | 0.637 [0.557, 0.718] |
| R3_medical | GRU-drop0.2 | 0.747 [0.724, 0.767] | 0.748 [0.724, 0.771] | 0.174 [0.081, 0.302] | 0.036 [0.005, 0.095] | 0.084 [0.024, 0.183] | 0.030 [0.000, 0.094] |
| R3_medical | GRU-drop0.1 | 0.722 [0.699, 0.744] | 0.724 [0.699, 0.748] | 0.101 [0.065, 0.159] | 0.015 [0.004, 0.035] | 0.059 [0.011, 0.135] | 0.006 [0.000, 0.024] |
| R3_banking | GRU | 0.736 [0.698, 0.776] | 0.657 [0.609, 0.703] | 0.622 [0.002, 0.968] | 0.438 [0.000, 0.718] | 0.456 [0.000, 0.757] | 0.473 [0.000, 0.769] |
| R3_banking | GRU-drop0.2 | 0.848 [0.828, 0.868] | 0.783 [0.750, 0.816] | 0.328 [0.131, 0.468] | 0.033 [0.005, 0.064] | 0.054 [0.012, 0.090] | 0.064 [0.000, 0.133] |
| R3_banking | GRU-drop0.1 | 0.834 [0.805, 0.859] | 0.762 [0.721, 0.803] | 0.356 [0.129, 0.684] | 0.068 [0.009, 0.190] | 0.097 [0.009, 0.263] | 0.122 [0.000, 0.377] |
| R3_web | GRU | 0.726 [0.689, 0.767] | 0.727 [0.686, 0.778] | 0.917 [0.809, 0.997] | 0.623 [0.472, 0.772] | 0.697 [0.572, 0.812] | 0.586 [0.463, 0.702] |
| R3_web | GRU-drop0.2 | 0.870 [0.851, 0.887] | 0.872 [0.848, 0.891] | 0.230 [0.044, 0.522] | 0.024 [0.000, 0.081] | 0.073 [0.000, 0.221] | 0.011 [0.000, 0.047] |
| R3_web | GRU-drop0.1 | 0.855 [0.838, 0.872] | 0.859 [0.840, 0.876] | 0.317 [0.083, 0.556] | 0.042 [0.000, 0.109] | 0.155 [0.006, 0.314] | 0.011 [0.000, 0.035] |
| R3_email | GRU | 0.671 [0.625, 0.730] | 0.679 [0.637, 0.738] | 0.973 [0.940, 0.995] | 0.764 [0.691, 0.841] | 0.804 [0.768, 0.851] | 0.574 [0.384, 0.722] |
| R3_email | GRU-drop0.2 | 0.766 [0.726, 0.797] | 0.765 [0.728, 0.798] | 0.226 [0.183, 0.267] | 0.039 [0.027, 0.053] | 0.136 [0.082, 0.213] | 0.024 [0.008, 0.047] |
| R3_email | GRU-drop0.1 | 0.749 [0.713, 0.787] | 0.754 [0.717, 0.792] | 0.176 [0.129, 0.237] | 0.031 [0.014, 0.054] | 0.111 [0.040, 0.166] | 0.020 [0.004, 0.039] |
| R3_coding | GRU | 0.695 [0.653, 0.744] | 0.684 [0.645, 0.733] | 0.720 [0.258, 0.955] | 0.486 [0.078, 0.709] | 0.580 [0.218, 0.788] | 0.416 [0.000, 0.686] |
| R3_coding | GRU-drop0.2 | 0.811 [0.777, 0.837] | 0.797 [0.760, 0.826] | 0.397 [0.289, 0.469] | 0.100 [0.049, 0.155] | 0.155 [0.106, 0.199] | 0.031 [0.004, 0.067] |
| R3_coding | GRU-drop0.1 | 0.798 [0.777, 0.821] | 0.787 [0.763, 0.812] | 0.210 [0.134, 0.344] | 0.042 [0.015, 0.091] | 0.098 [0.049, 0.181] | 0.009 [0.000, 0.031] |
| R4 | GRU | 0.513 [0.485, 0.542] | 0.852 [0.839, 0.867] | 0.256 [0.213, 0.295] | 0.168 [0.115, 0.230] | 0.182 [0.120, 0.237] | 0.073 [0.049, 0.094] |
| R4 | GRU-drop0.2 | 0.500 [0.470, 0.531] | 0.840 [0.826, 0.856] | 0.059 [0.035, 0.084] | 0.127 [0.089, 0.166] | 0.138 [0.102, 0.173] | 0.024 [0.018, 0.030] |
| R4 | GRU-drop0.1 | 0.497 [0.465, 0.528] | 0.841 [0.825, 0.859] | 0.054 [0.034, 0.086] | 0.117 [0.082, 0.156] | 0.137 [0.102, 0.173] | 0.021 [0.016, 0.025] |
| WUT | GRU | 0.840 [0.794, 0.872] | 0.963 [0.951, 0.972] | 0.761 [0.721, 0.796] | 0.178 [0.115, 0.245] | 0.199 [0.128, 0.276] | 0.227 [0.174, 0.276] |

Dropout minus GRU, paired bootstrap on the same test rows [settings.<S>.paired]:

| setting | contrast | dAUROC (point; 95% CI) | dAUPRC | detected (AUROC) |
|---|---|---|---|---|
| R1 | GRU-drop0.2-minus-GRU | +0.007 [-0.002, +0.017] | +0.006 [-0.007, +0.017] | False |
| R1 | GRU-drop0.1-minus-GRU | +0.008 [+0.000, +0.016] | +0.008 [-0.001, +0.016] | True |
| R3_medical | GRU-drop0.2-minus-GRU | +0.108 [+0.080, +0.139] | +0.114 [+0.085, +0.147] | True |
| R3_medical | GRU-drop0.1-minus-GRU | +0.083 [+0.059, +0.112] | +0.090 [+0.065, +0.121] | True |
| R3_banking | GRU-drop0.2-minus-GRU | +0.112 [+0.075, +0.146] | +0.126 [+0.088, +0.169] | True |
| R3_banking | GRU-drop0.1-minus-GRU | +0.097 [+0.055, +0.135] | +0.105 [+0.060, +0.154] | True |
| R3_web | GRU-drop0.2-minus-GRU | +0.144 [+0.102, +0.181] | +0.145 [+0.090, +0.186] | True |
| R3_web | GRU-drop0.1-minus-GRU | +0.130 [+0.087, +0.167] | +0.132 [+0.081, +0.173] | True |
| R3_email | GRU-drop0.2-minus-GRU | +0.095 [+0.022, +0.151] | +0.086 [+0.017, +0.141] | True |
| R3_email | GRU-drop0.1-minus-GRU | +0.078 [+0.008, +0.143] | +0.075 [+0.007, +0.136] | True |
| R3_coding | GRU-drop0.2-minus-GRU | +0.116 [+0.057, +0.163] | +0.113 [+0.057, +0.158] | True |
| R3_coding | GRU-drop0.1-minus-GRU | +0.104 [+0.054, +0.148] | +0.103 [+0.057, +0.147] | True |
| R4 | GRU-drop0.2-minus-GRU | -0.013 [-0.021, -0.004] | -0.012 [-0.018, -0.005] | True |
| R4 | GRU-drop0.1-minus-GRU | -0.017 [-0.027, -0.008] | -0.011 [-0.020, -0.002] | True |

## R4 after the corrected AgentDojo extraction [R4_values]

- v0.3 (buggy extraction) GRU transfer AUROC on P0: 0.488 [0.458, 0.518]; within-AgentDojo UT: 0.860 [0.825, 0.886].
- Cache R4 GRU (transfer, P0): AUROC 0.513 [0.485, 0.542], AUPRC 0.852 [0.839, 0.867], recall@t_r80 0.256 [0.213, 0.295], FPR@t_r80 0.168 [0.115, 0.230].
- Cache R4 GRU-drop0.2 (transfer, P0): AUROC 0.500 [0.470, 0.531], AUPRC 0.840 [0.826, 0.856], recall@t_r80 0.059 [0.035, 0.084], FPR@t_r80 0.127 [0.089, 0.166].
- Cache R4 GRU-drop0.1 (transfer, P0): AUROC 0.497 [0.465, 0.528], AUPRC 0.841 [0.825, 0.859], recall@t_r80 0.054 [0.034, 0.086], FPR@t_r80 0.117 [0.082, 0.156].
- Cache WUT GRU (within-AgentDojo UT, P0): AUROC 0.840 [0.794, 0.872], AUPRC 0.963 [0.951, 0.972].
- Cluster bootstrap over 117 suite|user_task clusters (sensitivity), R4 GRU AUROC 0.513 [0.463, 0.551]; WUT GRU AUROC 0.840 [0.783, 0.885].
- Model-balanced (native counts, r4_control.py weighting), GRU AUROC: R4 0.536 [0.501, 0.569], WUT 0.829 [0.778, 0.860] (estimate = bootstrap mean, as r4_control.py).
- R4 saturation: the behavioural pathway reaches |logit| up to 974 on AgentDojo (observation lengths far outside the AgentDrift standardisation); fraction of P0 scores exactly 1.0 per seed [0.0004, 0.0, 0.0], exactly 0.0 [0.0432, 0.0372, 0.0432] [settings.R4.detectors.GRU.mechanism]. Ties are handled by evalkit's rank-based AUROC.

## Known limitations

- Intervals here are the v0.3 trajectory-level stratified bootstrap. The plan's primary estimands (prevalence-weighted AUPRC, cluster bootstraps, Holm families, domain-matched R1 comparators) are computed by the analysis scripts from this cache; the cluster intervals given here for R2/R4/WUT are a labelled sensitivity only.
- WUT: failed attacks whose suite|user_task group has no P0 row are assigned to folds by default_rng(10000 + rep) (same rule as the linear cache); any of the five models is out-of-fold for them because training uses P0 rows only.
- `unk_train_steps` counts UNK among the encoded steps (last 16 steps per trajectory); the untruncated count is in `checks.exact_refit_checks.unk_train_steps_untruncated` for R1, R3 and R4.
