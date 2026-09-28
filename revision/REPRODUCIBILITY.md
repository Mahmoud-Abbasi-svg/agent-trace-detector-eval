# Reproducibility guide (review comment m14)

This file lists every script of the project in execution order, with its command line, the environment variables it
reads, its random seeds and bootstrap settings, and the model settings behind the reported numbers. File and line
references were checked against the scripts on 26 September 2026. The automated extraction in
`revision/repro_checks.json` (written by `revision/repro_checks.py`) records, for every script, its SHA-256, the
environment variables it reads, and every line that sets a seed, a bootstrap size or an estimator. Use it to check this
file after any script changes. Seven analysis-stream scripts (Section 4.4) were still being edited by their own
streams when this file was written, so their exact line numbers are in `repro_checks.json`, not here.
`repro_checks.py` was last re-run on 27 September 2026, 02:15.

## 1. Environment

| Item | Value |
|---|---|
| Machine | Windows 11 (build 26200), CPU only |
| Python | CPython 3.10.4 (`C:/Python310/python.exe`) |
| Third-party packages the code imports | numpy 2.2.6, scipy 1.15.3, scikit-learn 1.7.2, torch 2.11.0+cpu, matplotlib 3.10.8 (figures only) |
| Lock file | `revision/requirements-lock.txt`: the five packages above plus their 20 installed runtime dependencies, exact versions from `importlib.metadata`. Install with `pip install -r revision/requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cpu` |
| Other imports | Python standard library only. Every third-party import in every `.py` file was listed with `ast`, and nothing beyond the five packages appears |
| Determinism | `gru_detector.py:15` and `revision/gru_variants.py` call `torch.use_deterministic_algorithms(True)`. Every GRU fit calls `torch.manual_seed(seed)` and `np.random.seed(seed)` (`gru_detector.py:63`). The scikit-learn fits are deterministic: lbfgs, and IsolationForest with `random_state=0` |

## 2. Data and environment variables

| Variable | Points to | Read by (file:line) |
|---|---|---|
| `AGENTDRIFT` | clone of the AgentDrift repository: commit `014a514fa998b4ac4519579fceb8a5884b379bda` (2026-09-10), dataset release 1.0 of 2026-09-07 (`CHANGELOG.md:3`), record schema field `"version": "2.0"` | `rerun_r1r3.py:21`, `ablation.py:12`, `world_disjoint_r1.py:19`, `r4_control.py:12`, `gru_ladder.py:12`, `gru_r3_diagnostic.py:30`, `baseline_drift.py:26`, `sweep.py:18`, `cross_benchmark.py:18`, `pre_ablation_audit.py:49`, `featurise.py:68` (only when run as a script), `inspect_agentdrift.py:7` (optional, with a fallback path), `revision/build_linear_cache.py:54`, `revision/build_gru_cache.py:37`, `revision/linear_r4_truncation_check.py:22`, `revision/summarize_gru_cache.py:303` (`recheck` only), and the analysis streams of Section 5 |
| `AGENTDOJO` | clone of the AgentDojo repository (MIT licence): commit `089ed468cf3ed0322acc66b0211f26d9d90dbf60` (2026-06-02), with its `runs/` folder | `agentdojo_extract.py:25`, `tests/test_agentdojo_alignment.py:15`, `pre_ablation_audit.py:50` (optional; provenance section only), `revision/features_obslen.py:24`, `revision/r4_deep.py:1504, 1534, 1558` |
| `AUDIT_NJOBS` | optional; number of GridSearchCV workers in `pre_ablation_audit.py` (default 4; results do not depend on it) | `pre_ablation_audit.py:59` |

Several modules read `AGENTDRIFT` when they are imported: `gru_ladder.py`, `world_disjoint_r1.py`, `r4_control.py`,
`ablation.py`, `rerun_r1r3.py` and `revision/build_linear_cache.py`. Any script that imports one of them therefore
needs the variable set, even if it never loads AgentDrift itself. Examples are `revision/linear_tie_check.py` and
`revision/prevalence_ops.py`, through `build_linear_cache`.

Set both variables before running anything (Git Bash):

```bash
export AGENTDRIFT=<path>/AgentDrift      # commit 014a514f
export AGENTDOJO=<path>/agentdojo        # commit 089ed468
PY=C:/Python310/python.exe
cd <path>/agent-ids
```

## 3. Library modules (imported, not run)

| Module | Role | Settings |
|---|---|---|
| `featurise.py` | `load`, `build_vocab` (sorted tool names), `vectorise` (10 behavioural features + bag of tools), `tool_seq`, `label_pos` | state-changing keyword list at `featurise.py:43-46` |
| `evalkit.py` | AUROC (rank-based), AUPRC (`average_precision_score`), stratified bootstrap, paired and independent deltas, `thr_at_recall` | `bootstrap(..., B=2000, seed=12345)` defaults (`evalkit.py:58`). Each replicate resamples positives and negatives with replacement and draws one repetition per method |
| `baseline_drift.py` | **dependency of `rerun_r1r3.py:18`**, which imports `Markov` and `feats` from it; also imported by `revision/linear_tie_check.py`. It produces the Markov, LR-C (LOGREG) and Isolation Forest rows of Table IV | Markov smoothing `k_smooth = 0.5` (`baseline_drift.py:32`). `RNG = default_rng(42)` (`:27`) is defined but never used |
| `gru_detector.py` | pre-registered GRU (`GRU_PREREGISTRATION.md`) | `MAXLEN 16, EMB 32, HID 64, BEH_DIM 10` (`:12`); Adam `lr 1e-3`, weight decay `1e-4`, batch 64, at most 30 epochs, early stopping on validation AUPRC with patience 5 (`:13`); vocabulary from benign training trajectories (`:18-20`) |

## 4. Execution order

### 4.1 Data preparation and checks

| # | Command | Inputs | Outputs | Seeds |
|---|---|---|---|---|
| 1 | `$PY agentdojo_extract.py` | `$AGENTDOJO/runs/<5 models>/...` | `agentdojo_traces.jsonl` (P0: 392 benign, 2,294 attacked), `agentdojo_failed.jsonl` (4,987), `agentdojo_extract_manifest.json` | none (deterministic; runs with fewer than two tool calls are excluded) |
| 2 | `$PY -B tests/test_agentdojo_alignment.py` (or `$PY -B -m pytest -q -p no:cacheprovider tests/test_agentdojo_alignment.py`) | `$AGENTDOJO`, `agentdojo_traces.jsonl` | stdout | none. **Result (26 Sep 2026, re-run 27 Sep 2026): pass.** Script mode prints the known Gemini run OK, "positional == id pairing on 8151 runs with usable ids: OK" and "extracted file uses positional pairing: OK". Under pytest: "3 passed, 1 warning", where the warning says one test returns a value (`revision/repro_checks.json` → `alignment_test`) |
| 3 | `$PY inspect_agentdrift.py` (optional descriptive) | `$AGENTDRIFT/data*` | `summary.json` | none |
| 4 | `$PY -B pre_ablation_audit.py [--budget 300]`, repeated until it prints `ASSEMBLED` (about 15 min of work in total on this machine; `--status` lists the sections) | `$AGENTDRIFT`, `$AGENTDOJO`, `agentdojo_*.jsonl`, `revision/repro_checks.json` | `revision/diagnostics_parts/*.json`, `revision/diagnostics_results.json` | bootstrap B = 2000 with seeds `20260926 + offset` (base offsets in `SEEDS`, `pre_ablation_audit.py:61-63`, plus the fixed per-block increments written next to each call). Probes: outer `StratifiedKFold(5, shuffle=True, random_state=0,1,2)`, inner `StratifiedKFold(3, shuffle=True, random_state=100+10*rep+fold)` |
| 5 | `$PY -B revision/repro_checks.py` | all scripts, `tests/` | `revision/requirements-lock.txt`, `revision/repro_checks.json` | none |
| 6 | `$PY revision/write_diagnostics_summary.py` | `revision/diagnostics_results.json` | `revision/diagnostics_summary.md` | none |

`pre_ablation_audit.py` was truncated in v0.3: it ended mid-line at line 44 and did not parse. It was rewritten for
the revision; a byte copy of the v0.3 file is in `revision/pre_ablation_audit_v03_truncated.py.txt`. Every script in
`agent-ids/`, `paper/`, `tests/` and `revision/` now parses (`repro_checks.json` → `parse_failures` is empty).

### 4.2 v0.3 experiments (re-run unchanged on the corrected AgentDojo data)

All logistic regressions in this block are `LogisticRegression(max_iter=1000, class_weight="balanced")` on
`StandardScaler`-transformed features, with the scikit-learn 1.7.2 defaults for everything else: **L2 penalty, C = 1.0,
solver lbfgs, tol 1e-4, intercept fitted** (`get_params()` confirmed). Thresholds: `t_r80` is the 20th percentile of
in-sample training-positive scores (`evalkit.thr_at_recall`), and `t0.5` is 0.5.

| # | Command | Outputs | Detector settings (file:line) | Bootstrap and other seeds (file:line) |
|---|---|---|---|---|
| 7 | `$PY rerun_r1r3.py` | `r1r3_corrected.json` (Table IV) | LR `:34`; IsolationForest `n_estimators=200, random_state=0` on standardized features of benign training rows (`:38`), other defaults (`max_samples='auto'`, `contamination='auto'`, `max_features=1.0`, no bootstrap); Markov from `baseline_drift.py` | B = 1000 (`:22`); seed `5000 + 17*ri` (`:60`), with ri in the rung order R1_same 0 (5000), R2_newtask 1 (5017), R3 medical 2 (5034), banking 3 (5051), web 4 (5068), email 5 (5085), coding 6 (5102). The same seed is used at native prevalence and at every controlled prevalence of a rung |
| 8 | `$PY ablation.py` | `ablation_results.json` (Tables V, VI, VII) | LR `:31` | R1: B = 1000, seed 5000 (`:85`). R3: B = 1000, seed `5034 + 17*ri` with ri = 0..4 for medical, banking, web, email, coding (`:91`). R4 P0 and C1: B = 2000, seed 12345 (`:120`). R4 model-balanced at 10%: seed 999 (`:148`). Within-AgentDojo folds: seeds 0-4 (`:13`, `:108`, `:139`), groups shuffled with `default_rng(sd)`, fold = position mod 5 |
| 9 | `$PY world_disjoint_r1.py` | `world_disjoint_r1_results.json` (Table VI, R1-wd row) | LR `:39` | split seeds 0-4 (`:21`; `world_disjoint_split` shuffles worlds with `default_rng(seed)`, `:54`); B = 1000, seed 777, split seed 0 only (`:112`) |
| 10 | `$PY r4_control.py` | `r4_control_results.json` (Table VII) | LR `:29`; model-identity probe LR `:189` | B = 2000 (`:13`). Every bootstrap uses the evalkit default seed 12345, because no seed is passed: native, p10 and p05 (`:120`), per model (`:142`), **model-balanced native counts and p10 (`:158`)**, and the probe (`:191`). CV fold seeds 0-4 (`:13`, `:55`). Probe folds: `default_rng(0)` (`:181`) |
| 11 | `$PY gru_ladder.py full` (`audit` for the split audit only) | `gru_audit.json`, `gru_results.json` (Table VIII) | GRU from `gru_detector.py`; comparator LR-C `:41` | training seeds 0, 1, 2 (`:13`). Validation split: stratified 10% of the training set, `default_rng(1000 + seed)` (`:18`), for every rung including R1 (`data/val.jsonl` is not used; DEVIATIONS G1). Bootstrap: R1 5000 (`:108`), R2 5100 (`:110`), R3 `5200 + ri`, medical to coding 5200-5204 (`:113`), R1-wd 5300 (`:118`), all with B = 1000; R4 P0 and C1 12345 with B = 2000 (`:159`); model-balanced at 10% 999 (`:175`). Within AgentDojo: fold seed = training seed = 0..4, one GRU per repetition (`:136-152`; DEVIATIONS G2) |
| 12 | `$PY gru_r3_diagnostic.py` | `gru_r3_diagnostic_results.json` | GRU-A, GRU-B, GRU-C variants; LR-C via `gru_ladder.fitC` | training seeds 0, 1, 2; bootstrap B = 1000, seed `5200 + ri` (`:134`) |
| 13 | `cd paper; $PY gen_tables.py; $PY make_figures.py; $PY sync_tables.py; $PY build_manuscript.py; $PY md2tex.py; pdflatex main` | `paper/tables/*`, figures, manuscript | reads the result files only | none |

The re-run of rows 8, 10 and 11 on the corrected extraction is logged in `revision/rerun_v03_scripts.log`, which
ends with "gru_ladder exit 0". The v0.3 outputs are archived in `archive_v0.3/`.

Superseded exploratory scripts, which no manuscript number comes from: `baseline_drift.py` run as a script
(`baseline_results.json`; LR `:109`, IsolationForest `random_state=0` `:113`, test subsampling
`default_rng(seed=0)` `:68`), `sweep.py` (`sweep_results.json`; 20 subsampling seeds `default_rng(1000 + seed)`
`:20, :68`; IsolationForest `random_state=0` `:54`) and `cross_benchmark.py` (`cross_benchmark_results.json`;
20 seeds `default_rng(seed)` `:19, :70`).

### 4.3 Revision prediction caches

| # | Command | Outputs | Settings and seeds |
|---|---|---|---|
| 14 | `$PY revision/build_linear_cache.py --all [--budget 420]`, repeated until done (`--status`, `--setting S`, `--force`) | `revision/cache/linear/<setting>.pkl` (20 settings) | LR-A/B/C, -noobs, -minus and -drop as `ablation.py` (`:305`); IsolationForest `n_estimators=200, random_state=0` (`:355`); TF-IDF `TfidfVectorizer(lowercase=True, ngram_range=(1,2), min_df=2, max_features=50000, sublinear_tf=True)` (`:363`) + `LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)` (`:367`); Markov scores rounded to 12 decimals (`:277`). CV fold seeds 0-4 (`:59`); groups without a P0 row are assigned with `default_rng(10000 + seed)` (`:498`) |
| 15 | `$PY revision/validate_linear_cache.py [--budget 500]` | `cache/linear/validation.json`, `CACHE_README.md` | bootstrap B = 1000 (AgentDrift) or 2000 (AgentDojo), seed `20260926 + setting index` (`:103`); flag-rate bootstrap `seed + 1` (`:113`) |
| 16 | `$PY revision/linear_r4_truncation_check.py` | `cache/linear/r4_truncation_check.json` | B = 2000, seed 12345; CV seeds from `r4_control.SEEDS` |
| 17 | `$PY revision/linear_tie_check.py` | `cache/linear/tie_check.json` | no bootstrap |
| 18 | `$PY revision/build_gru_cache.py run <setting> <GRU / GRU-drop0.2 / GRU-drop0.1> [--budget 520]`, repeated until `COMPLETE`; also `verify`, `status` | `revision/cache/gru/<setting>.pkl` | training seeds 0, 1, 2 (`:42`); WUT repetitions 0-4 with fold seed = training seed = repetition (`:43`); groups without a P0 row `default_rng(10000 + rep)` (`:137`); validation split `default_rng(1000 + seed)` via `gru_ladder.stratified_val`; token dropout uses a separate `torch.Generator` seeded with `seed + 10007` (`revision/gru_variants.py`) |
| 19 | `$PY revision/summarize_gru_cache.py all` (or `summarize`, `recheck`, `assemble`) | `revision/gru_cache_results.json`, `cache/gru/CACHE_README.md` | B = 2000; seed `7000 + setting index`; flag rates `seed + 100`; cluster bootstrap `seed + 200` (R2, R4, WUT); R4 populations seed 12345 (model-balanced at the evalkit default 12345) |
| 20 | `bash revision/offline_driver.sh` | drives rows 14 and 18 in order | as rows 14 and 18 |

### 4.4 Revision analysis streams (read the caches)

These scripts belong to other analysis streams and were still changing when this file was written. Commands are
taken from their docstrings. Seeds are fixed and listed in the scripts. The per-line extraction, with SHA-256 at the
time of listing, is in `revision/repro_checks.json` → `scripts`.

| Script | Command(s) | Environment | Bootstrap and seeds |
|---|---|---|---|
| `revision/content_detector.py` | `fit --all [--budget 540]`, `analyze --all`, `assemble`, `summarize`, `status` | AGENTDRIFT | B = 2000; seeds `crc32("|".join(parts)) mod 2^31` per analysis (`seed_of`); TF-IDF configuration as row 14 |
| `revision/domain_shift.py` | `run [sections]`, `assemble`, `all`, `status` | AGENTDRIFT (section m12 only) | B = 2000; one seed per population, `crc32(name) & 0x7FFFFFFF` (`seed_of`); classifier two-sample test `StratifiedKFold(5, shuffle=True, random_state=0)` with `LogisticRegression(max_iter=5000, class_weight="balanced")` |
| `revision/features_obslen.py` | `status`, `<part>`, `all`, `assemble`, `summary` | AGENTDRIFT, AGENTDOJO | B = 2000; fixed seeds per block (8001-8024, 8101-8102, 9200-9903) |
| `revision/oov_mechanism.py` | `train`, `graded`, `analyze`, `summary`, `all` | AGENTDRIFT (and AGENTDOJO per its docstring) | B = 2000; `POOL_SEEDS` and `REP_SEEDS` dictionaries (20201-20224, 50500-50912); graded-coverage masks `default_rng(7100 + 100*level + draw)` |
| `revision/prevalence_ops.py` | `train_scores`, `analyze [--budget]`, `derive`, `summary`, `all` | AGENTDRIFT (through `build_linear_cache`) | B = 2000; seeds `20260926 + fixed offset` per block; IsolationForest `random_state=0` |
| `revision/r4_deep.py` | `status`, `fit <job/all> [--budget]`, `analyze <section/all>`, `assemble` | AGENTDRIFT, AGENTDOJO | stratified bootstrap seed 12345, cluster bootstrap seed 20260927, failed-attack flag rates seed 4242; B = 2000 (with B = 1000 and seeds 5000, 5200+i and 6000+seed in the v0.3-comparison blocks) |
| `revision/tool_state_labels.py` | imported (hand annotation of tool names) | none | none |

## 5. Bootstrap seeds at a glance (manuscript tables)

| Table or quantity | Script | B | Seed(s) |
|---|---|---|---|
| Table IV (R1-R3 reference baselines) | `rerun_r1r3.py` | 1000 | 5000 + 17 ri (5000, 5017, 5034, 5051, 5068, 5085, 5102) |
| Tables V and VI, A/B/C rows | `ablation.py` | 1000 (R1, R3), 2000 (R4) | R1 5000; R3 5034 + 17 ri; R4 12345; model-balanced at 10% 999 |
| Table VI, R1-wd row | `world_disjoint_r1.py` | 1000 | 777 (split seed 0); split seeds 0-4 |
| Table VII | `r4_control.py` | 2000 | 12345 everywhere, including model-balanced (evalkit default) |
| Table VIII (GRU) | `gru_ladder.py` | 1000 (R1-R3, R1-wd), 2000 (R4) | 5000, 5100, 5200-5204, 5300, 12345, 999 |
| GRU R3 diagnostic | `gru_r3_diagnostic.py` | 1000 | 5200-5204 |
| Sec. III-A / VI-B diagnostics | `pre_ablation_audit.py` (restored) | 2000 | 20260926 + offsets (1-7, 10-14, plus fixed per-block increments) |

Sec. IX of the manuscript gives seeds only for `rerun_r1r3.py` (5000 + 17i) and R4 (12345, and 999 for the
model-balanced evaluation). The 999 belongs to `ablation.py` and `gru_ladder.py`. `r4_control.py`, which produces the
model-balanced values in Table VII, uses 12345.

## 6. Numbers that were not traceable in v0.3

The world count, purity and probe numbers in Sec. III-A and Sec. VI-B came from the truncated audit script. They now
have keys in `revision/diagnostics_results.json`. Some reproduce and some do not; see `revision/diagnostics_summary.md`
and `manuscript_comparison` in the results file. In short:
- 99.3%, 0.963, 0.441, 0.558 and "54 worlds with at least 30 trajectories (2,724)" reproduce.
- 1,029 worlds, purity 0.858 and the largest worlds 194/169/144 do not reproduce together under any of the 143 world
  keys tried. The audit tried 143 keys on 6 populations with 2 labels, 1,716 rows in all. Four rows give 1,029 worlds
  by themselves: `contacts[names]` on the full corpus, and `user+date+contacts[names_sorted]` on train+val. One of them
  also gives purity 0.8575. None gives 194/169/144 or 54 worlds with 2,724 trajectories. Under the split's own key
  there are 1,133 worlds (purity 0.883) in the pool and 1,141 in the full corpus; the full-corpus count matches the
  AgentDrift README.
- The domain-probe values 1.000/0.455/0.381 reproduce with an unregularized 3-fold unshuffled probe.
- The world-probe values 0.066/0.046/0.043 lie within the fold-assignment spread of that probe but do not reproduce
  exactly.
