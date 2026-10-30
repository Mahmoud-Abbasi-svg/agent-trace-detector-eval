| Detector | medical | banking | web | email | coding | Folds failing (Holm / unadjusted) |
| --- | --- | --- | --- | --- | --- | --- |
| *Status relative to domain-matched R1 after Holm over five folds* |  |  |  |  |  |  |
| LR-A (tool identity) | FPR↓ rec↓ | FPR↓ rec↓ | FPR↓ rec↓ | FPR↓ rec↓ | FPR↓ rec↓ | 5 / 5 |
| LR-B (behavioral) | rec↓ | holds | holds | holds | holds | 1 / 1 |
| LR-B-minus | FPR↓ | holds | holds | holds | holds | 1 / 1 |
| LR-B-noobs | holds | holds | FPR↓ rec↓ | holds | holds | 1 / 2 |
| LR-C (combined) | FPR↑ rec↑ | FPR↑ rec↑ | FPR↑ rec↑ | FPR↑ rec↑ | FPR↑ rec↑ | 5 / 5 |
| LR-C-drop | FPR↓ rec↓ | FPR↑ rec↑ | rec↓ | FPR↑ | holds | 4 / 4 |
| LR-C-noobs | FPR↑ rec↑ | FPR↑ rec↑ | FPR↑ rec↑ | FPR↑ rec↑ | FPR↑ rec↑ | 5 / 5 |
| GRU | FPR↑ rec↑ | holds | FPR↑ rec↑ | FPR↑ rec↑ | holds | 3 / 4 |
| GRU, seed 0 | FPR↑ rec↑ | FPR↑ rec↑ | FPR↑ rec↑ | FPR↑ rec↑ | FPR↑ rec↑ | 5 / 5 |
| GRU, seed 1 | FPR↑ rec↑ | rec↓ | FPR↑ rec↑ | FPR↑ rec↑ | FPR↑ rec↑ | 5 / 5 |
| GRU, seed 2 | FPR↑ rec↑ | FPR↑ rec↑ | FPR↑ rec↑ | FPR↑ rec↑ | rec↓ | 5 / 5 |
| GRU at θval80 | FPR↑ rec↑ | holds | FPR↑ rec↑ | FPR↑ rec↑ | FPR↑ | 4 / 4 |
| GRU-drop0.2 | rec↓ | rec↓ | rec↓ | rec↓ | rec↓ | 5 / 5 |
| GRU-drop0.1 | rec↓ | holds | rec↓ | rec↓ | rec↓ | 4 / 4 |
| TF-IDF (content) | rec↓ | rec↓ | rec↓ | rec↓ | rec↓ | 5 / 5 |
| TF-IDF no digits | holds | rec↓ | rec↓ | rec↓ | rec↓ | 4 / 4 |
| Markov | FPR↑ rec↑ | FPR↑ rec↑ | FPR↑ rec↑ | FPR↑ rec↑ | FPR↑ rec↑ | 5 / 5 |
| IForest | FPR↓ rec↓ | FPR↓ rec↓ | FPR↓ rec↓ | FPR↓ rec↓ | FPR↓ rec↓ | 5 / 5 |
| *R3 minus domain-matched R1 at θr80* |  |  |  |  |  |  |
| LR-B (behavioral), Δ recall | −0.135* [−0.208, −0.059] | +0.027 [−0.048, +0.110] | −0.037 [−0.092, +0.023] | +0.039 [−0.025, +0.106] | +0.007 [−0.054, +0.069] |  |
| LR-B (behavioral), Δ FPR | −0.087 [−0.159, −0.018] | +0.024 [−0.052, +0.094] | −0.058 [−0.129, +0.013] | +0.061 [−0.018, +0.138] | −0.032 [−0.109, +0.046] |  |
| LR-C (combined), Δ recall | +0.177* [+0.102, +0.248] | +0.354* [+0.265, +0.447] | +0.085* [+0.045, +0.131] | +0.173* [+0.114, +0.235] | +0.135* [+0.083, +0.191] |  |
| LR-C (combined), Δ FPR | +0.181* [+0.105, +0.262] | +0.539* [+0.478, +0.594] | +0.543* [+0.479, +0.603] | +0.595* [+0.539, +0.652] | +0.447* [+0.379, +0.513] |  |
| GRU, Δ recall | +0.353* [+0.225, +0.466] | +0.123 [−0.575, +0.551] | +0.183* [+0.036, +0.326] | +0.199* [+0.117, +0.288] | −0.030 [−0.538, +0.283] |  |
| GRU, Δ FPR | +0.672* [+0.582, +0.740] | +0.418 [−0.042, +0.702] | +0.552* [+0.381, +0.724] | +0.734* [+0.649, +0.818] | +0.427 [+0.002, +0.668] |  |
| TF-IDF (content), Δ recall | −0.081* [−0.147, −0.009] | −0.413* [−0.503, −0.320] | −0.195* [−0.261, −0.127] | −0.121* [−0.184, −0.059] | −0.158* [−0.224, −0.089] |  |
| TF-IDF (content), Δ FPR | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |  |
