| Detector | Rate | medical | banking | web | email | coding | Failing folds (Holm) and direction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LR-B (behavioral) | Recall | 0.694 → 0.560* | 0.802 → 0.829 | 0.861 → 0.825 | 0.794 → 0.833 | 0.828 → 0.835 | 1 of 5 |
| LR-B (behavioral) | FPR | 0.236 → 0.150 | 0.255 → 0.278 | 0.255 → 0.197 | 0.358 → 0.418 | 0.315 → 0.283 | recall down (medical only) |
| LR-C (combined) | Recall | 0.650 → 0.828* | 0.613 → 0.967* | 0.906 → 0.990* | 0.789 → 0.962* | 0.839 → 0.974* | 5 of 5 |
| LR-C (combined) | FPR | 0.321 → 0.503* | 0.121 → 0.660* | 0.164 → 0.707* | 0.127 → 0.722* | 0.224 → 0.672* | over-alerting (FPR and recall up) |
| GRU | Recall | 0.585 → 0.938* | 0.498 → 0.622 | 0.733 → 0.917* | 0.774 → 0.973* | 0.750 → 0.720 | 3 of 5 |
| GRU | FPR | 0.044 → 0.717* | 0.020 → 0.438 | 0.071 → 0.623* | 0.030 → 0.764* | 0.059 → 0.486 | over-alerting (FPR and recall up) |
| GRU | Per seed | 3 of 3 seeds fail | 3 of 3 seeds fail | 3 of 3 seeds fail | 3 of 3 seeds fail | 3 of 3 seeds fail | 5 of 5 for every seed |
| TF-IDF (content) | Recall | 0.739 → 0.658* | 0.685 → 0.272* | 0.767 → 0.572* | 0.817 → 0.696* | 0.778 → 0.620* | 5 of 5 |
| TF-IDF (content) | FPR | 0.000 → 0.000 | 0.000 → 0.000 | 0.000 → 0.000 | 0.000 → 0.000 | 0.000 → 0.000 | under-alerting (recall down) |
