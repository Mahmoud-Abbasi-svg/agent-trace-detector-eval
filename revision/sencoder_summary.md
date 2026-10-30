# Sentence-encoder detector (SE, a reimplementation of the DriftNet architecture): summary

*Generated 9 Oct 2026 from `revision/sencoder_results.json`. Plan `SENCODER_PREREGISTRATION.md` amendment 6, SHA-256 `cef3e7677710e561496b8427a261d2e34737c7af7908e5ef5809ffb5a43a8493`. Slurm jobs 2762 and 2797 (fits and analysis) and 2799 (GPU check). Full write-up in `SENCODER_RESULTS.md`. No number here is DriftNet's reported performance.*

## Outcome decision (plan Section 9)

| Outcome | Met | Deciding numbers |
|---|---|---|
| 1, mechanism survives | no | offset under +0.5 in 0 of 5 folds, reversal against LR-B in 0 folds |
| 2, mechanism incomplete | no | reversal against LR-B in 0 folds |
| 3, transfer still fails | yes | transfer P0 0.481 [0.418, 0.555], model-balanced 0.477 [0.400, 0.551], within P0 0.956 [0.924, 0.977] |
| 4, transfer succeeds | no | lower bounds 0.418 and 0.400 |
| 5, seed inconsistency | no | largest R3 across-seed AUROC range 0.084 (web) |

## Reading

Outcome 3 is met. The R3 result is a case the plan did not list. A frozen encoder has no untrained input, yet in medical, coding and web its held-out negatives sit 5.2 to 7.4 logits above its training negatives, as large as the GRU on the same estimator (post hoc). In banking and email the offset is at the in-distribution level (+0.52 and +0.69, against +0.64 at R1). No reversal against LR-B occurs in any fold. The offset part of the Section VI-D mechanism is therefore contradicted, and the reversal part is not. Section VI-D should stop presenting the offset as a signature of untrained inputs. Transfer to AgentDojo stays near chance while within-AgentDojo reaches 0.96, and zeroing the approximated world features changes neither (SE-W0). The graded tool-coverage sweep barely moves SE, from 0.998 to 0.994. Observation padding of 1,200 characters cuts SE recall from 0.796 to 0.531, much less than LR-C (to 0.033) and the GRU (to 0.000). The GRU at 10 seeds reproduces its 3-seed values within 0.015 AUROC and keeps the same R3 reversal folds against LR-C.

## Ladder, AUROC on pool E

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

## SE operating points

| Rung | AUPRC native | AUPRC at 2% | AUPRC at 10% | recall at θr80 | FPR at θr80 | recall at 0.5 | FPR at 0.5 |
|---|---|---|---|---|---|---|---|
| R1 | 0.998 [0.996, 0.999] | 0.967 [0.877, 1.000] | 0.988 [0.968, 0.999] | 0.796 [0.764, 0.826] | 0.001 [0.000, 0.004] | 0.981 [0.965, 0.993] | 0.022 [0.007, 0.044] |
| R2 | 0.996 [0.994, 0.998] | 0.935 [0.830, 1.000] | 0.977 [0.952, 0.994] | 0.786 [0.746, 0.821] | 0.001 [0.000, 0.008] | 0.970 [0.950, 0.990] | 0.025 [0.012, 0.043] |
| R3 medical | 0.965 [0.943, 0.977] | 0.681 [0.390, 0.896] | 0.829 [0.706, 0.906] | 0.770 [0.688, 0.884] | 0.046 [0.018, 0.091] | 0.960 [0.917, 0.984] | 0.263 [0.170, 0.327] |
| R3 banking | 0.906 [0.867, 0.931] | 0.568 [0.342, 0.791] | 0.745 [0.644, 0.834] | 0.345 [0.281, 0.447] | 0.002 [0.000, 0.005] | 0.643 [0.527, 0.749] | 0.029 [0.004, 0.056] |
| R3 web | 0.916 [0.824, 0.948] | 0.509 [0.096, 0.792] | 0.678 [0.353, 0.818] | 0.732 [0.632, 0.809] | 0.106 [0.031, 0.255] | 0.936 [0.894, 0.957] | 0.396 [0.306, 0.513] |
| R3 email | 0.964 [0.953, 0.976] | 0.769 [0.576, 0.934] | 0.864 [0.801, 0.925] | 0.620 [0.575, 0.672] | 0.001 [0.000, 0.006] | 0.800 [0.754, 0.840] | 0.035 [0.015, 0.066] |
| R3 coding | 0.950 [0.930, 0.962] | 0.648 [0.323, 0.863] | 0.792 [0.655, 0.866] | 0.737 [0.611, 0.802] | 0.049 [0.016, 0.104] | 0.940 [0.893, 0.969] | 0.317 [0.172, 0.382] |

## Reversal rule against LR-B

| Held-out domain | R1dm SE − LR-B (AUROC) | R3 SE − LR-B (AUROC) | R3 SE − LR-B (AUPRC at 10%) | reversal established (AUROC / AUPRC10) | SE R3 − R1dm (AUROC) |
|---|---|---|---|---|---|
| medical | +0.206 [+0.161, +0.256] | +0.151 [+0.125, +0.177] | +0.481 [+0.355, +0.558] | no / no | −0.040 [−0.058, −0.027] |
| banking | +0.163 [+0.117, +0.213] | +0.063 [+0.023, +0.093] | +0.301 [+0.192, +0.394] | no / no | −0.077 [−0.116, −0.054] |
| web | +0.111 [+0.078, +0.148] | +0.008 [−0.056, +0.043] (not detected) | +0.114 [−0.225, +0.254] | no / no | −0.093 [−0.154, −0.061] |
| email | +0.195 [+0.151, +0.242] | +0.158 [+0.133, +0.181] | +0.508 [+0.450, +0.558] | no / no | −0.049 [−0.067, −0.033] |
| coding | +0.151 [+0.113, +0.193] | +0.086 [+0.063, +0.107] | +0.335 [+0.197, +0.409] | no / no | −0.059 [−0.075, −0.045] |

## Offset analogue and embedding drift

| Held-out domain | SE offset, mean over 10 seeds | SE range over seeds | SE, benign and hard-negative training negatives only | GRU, same estimator (post hoc) | embedding drift, mean cosine distance |
|---|---|---|---|---|---|
| medical | +5.161 | +3.899 to +6.689 | +4.767 | +5.920 | 0.296 |
| banking | +0.524 | -0.040 to +1.522 | +0.262 | +5.592 | 0.283 |
| web | +7.420 | +3.994 to +8.821 | +7.092 | +7.280 | 0.410 |
| email | +0.692 | +0.373 to +1.291 | +0.313 | +7.133 | 0.260 |
| coding | +5.795 | +4.012 to +7.084 | +5.497 | +7.382 | 0.383 |
| R1 (reference) | +0.643 | +0.281 to +0.984 | +0.201 | +0.661 | 0.041 |

The GRU column is post hoc and not pre-registered.

## R4

| Population | n pos / n neg | SE transfer | SE within (UT) | within − transfer | SE-W0 transfer | SE-W0 within | GRU transfer | GRU within | LR-C transfer |
|---|---|---|---|---|---|---|---|---|---|
| P0 | 2,294 / 392 | 0.481 [0.418, 0.555] | 0.956 [0.924, 0.977] | +0.475 [+0.395, +0.544] | 0.503 [0.413, 0.579] | 0.963 [0.916, 0.985] | 0.510 [0.474, 0.542] | 0.845 [0.800, 0.872] | 0.473 [0.440, 0.505] |
| C1 | 2,294 / 327 | 0.467 [0.403, 0.533] | 0.949 [0.875, 0.977] | +0.482 [+0.381, +0.561] | 0.491 [0.409, 0.559] | 0.952 [0.910, 0.979] | 0.505 [0.468, 0.542] | 0.850 [0.812, 0.890] | 0.472 [0.437, 0.509] |
| P0F | 2,294 / 5,379 | 0.572 [0.525, 0.647] | 0.669 [0.630, 0.704] | +0.098 [+0.004, +0.165] | 0.587 [0.525, 0.640] | 0.645 [0.597, 0.684] | 0.646 [0.632, 0.662] | 0.833 [0.799, 0.859] | 0.626 [0.613, 0.640] |
| model-balanced, native counts | 2,287 / 308 | 0.477 [0.400, 0.551] | 0.961 [0.924, 0.984] | +0.484 [+0.403, +0.566] | 0.500 [0.405, 0.578] | 0.968 [0.919, 0.990] | 0.532 [0.490, 0.571] | 0.834 [0.786, 0.864] | 0.486 [0.448, 0.524] |

| Contrast (P0, AUROC, paired) | Difference |
|---|---|
| SE transfer − SE-W0 transfer | −0.023 [−0.117, +0.083] |
| SE within − SE-W0 within | −0.007 [−0.043, +0.044] |
| SE transfer − LR-B transfer | −0.001 [−0.057, +0.068] |
| SE transfer − GRU transfer | −0.029 [−0.089, +0.042] |

| Suite | Traces (P0 + unsuccessful) | Any address in arguments | Any URL in arguments | Any non-zero world feature |
|---|---|---|---|---|
| banking | 1,450 | 0.000 | 0.000 | 0.000 |
| slack | 1,601 | 0.347 | 0.816 | 0.871 |
| travel | 2,083 | 0.067 | 0.000 | 0.067 |
| workspace | 2,539 | 0.545 | 0.005 | 0.545 |

| Feature-block offset probe (P0, no refit, not SE-W0) | Mean over 10 seeds |
|---|---|
| logit shift, AgentDojo negatives | -0.181 |
| logit shift, AgentDojo positives | -0.481 |
| logit shift, AgentDrift training negatives | -0.071 |
| AUROC with features / zeroed | 0.481 / 0.464 |

## Graded coverage

| Level | Achieved coverage | SE AUROC | SE FPR at θr80 | SE recall at θr80 | SE − LR-B | LR-C (published) | GRU (published) | LR-B |
|---|---|---|---|---|---|---|---|---|
| 1.0 | 1.000 | 0.998 [0.996, 0.999] | 0.001 [0.000, 0.001] | 0.796 [0.772, 0.819] | +0.167 [+0.148, +0.187] | 0.879 | 0.936 | 0.830 |
| 0.75 | 0.775 | 0.997 [0.995, 0.998] | 0.000 [0.000, 0.001] | 0.774 [0.750, 0.797] | +0.167 [+0.147, +0.186] | 0.853 | 0.874 | 0.830 |
| 0.5 | 0.483 | 0.996 [0.994, 0.997] | 0.000 [0.000, 0.001] | 0.750 [0.725, 0.774] | +0.166 [+0.147, +0.186] | 0.825 | 0.792 | 0.830 |
| 0.25 | 0.225 | 0.995 [0.993, 0.997] | 0.000 [0.000, 0.001] | 0.726 [0.699, 0.751] | +0.165 [+0.146, +0.185] | 0.800 | 0.720 | 0.830 |
| 0.0 | 0.000 | 0.994 [0.991, 0.997] | 0.000 [0.000, 0.000] | 0.711 [0.683, 0.737] | +0.164 [+0.145, +0.184] | 0.787 | 0.674 | 0.830 |

## Padding

| Filler characters | SE recall at θr80 | SE AUROC | LR-C recall (published) | LR-B recall (published) | GRU recall (published) |
|---|---|---|---|---|---|
| 0 | 0.796 [0.772, 0.819] | 0.998 [0.996, 0.999] | 0.771 [0.744, 0.799] | 0.795 [0.768, 0.823] | 0.682 [0.646, 0.717] |
| 100 | 0.685 [0.655, 0.713] | 0.990 [0.987, 0.993] | 0.634 [0.602, 0.664] | 0.664 [0.632, 0.698] | 0.578 [0.543, 0.611] |
| 1,200 | 0.531 [0.498, 0.564] | 0.883 [0.868, 0.897] | 0.033 [0.021, 0.046] | 0.161 [0.137, 0.188] | 0.000 [0.000, 0.000] |

## Embedding drift by rung

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

## R2 against the published setting (not like for like, see D1, D6, D7)

| Quantity at threshold 0.5 | Our reimplementation, mean over 10 seeds (min to max) | DriftNet as published, single run (not ours) |
|---|---|---|
| precision | 0.975 (0.965 to 0.982) | 0.982 |
| recall | 0.970 (0.955 to 0.986) | 0.985 |
| F1 | 0.973 (0.967 to 0.979) | 0.983 |
| flag rate, benign | 0.021 (0.015 to 0.027) | 0.015 |
| flag rate, failed attack | 0.005 (0.005 to 0.009) | 0.000 |
| flag rate, hard negative | 0.036 (0.020 to 0.059) | 0.029 |

## GRU at 10 seeds

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

## Deviations

D1 to D12 are declared in the plan (Sections 11 to 15). D13, recorded at run time in `revision/sencoder_run/RUNTIME_DEVIATIONS.md`, covers the one fit that exceeded the 10-minute rule under 15-way CPU contention and the resumed run with 6 workers. The full list with reasons is in Section 10 of `SENCODER_RESULTS.md`.
