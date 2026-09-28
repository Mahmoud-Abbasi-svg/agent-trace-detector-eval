# Evaluating Baseline Detectors of Hijacked LLM Agents Under Rare Attacks and Distribution Shift

Code, derived data and result files for the paper by **Mahmoud Abbasi** and **Juan Manuel Núñez Velasco**
(BISITE Research Group, University of Salamanca).

The paper evaluates simple detectors that flag hijacked LLM agents from their tool-call traces. It covers
logistic regression on tool-identity, behavioral and combined features, a GRU sequence model and a TF-IDF content
reference. The evaluation runs at controlled attack prevalence and under world, task, domain and cross-benchmark
shifts on [AgentDrift](https://github.com/Asif-0209/AgentDrift) and
[AgentDojo](https://github.com/ethz-spylab/agentdojo).

## Repository layout

| Path | Contents |
|---|---|
| `featurise.py`, `evalkit.py`, `gru_detector.py`, `baseline_drift.py` | Shared modules: features, metrics and bootstrap, the GRU, the Markov and Isolation Forest references |
| `agentdojo_extract.py` | Extracts AgentDojo traces with positional pairing of tool calls and observations |
| `agentdojo_traces.jsonl`, `agentdojo_failed.jsonl` | Extracted AgentDojo traces (population P0, 392 benign and 2,294 compromised) and the 4,987 unsuccessful attacks |
| `rerun_r1r3.py`, `ablation.py`, `world_disjoint_r1.py`, `r4_control.py`, `gru_ladder.py`, `gru_r3_diagnostic.py`, `pre_ablation_audit.py` | Experiment scripts from the first version, re-run on the corrected extraction, with their result files (`*.json`) |
| `R4_CONTROL.md`, `PRE_ABLATION_DESIGN.md`, `GRU_PREREGISTRATION.md`, `REVISION_ANALYSIS_PLAN.md` | The four internal analysis plans (not externally registered; see Section IV-A and Table S3 of the paper) |
| `revision/` | Prediction-cache builders, the seven analysis streams, their result files and summaries, `DEVIATIONS.md`, `RESULTS_SUMMARY.md` and the reproducibility guide |
| `revision/cache/` | Validation and summary files of the prediction caches (the cached predictions themselves, about 300 MB, are not included and are rebuilt by the scripts) |
| `paper/` | Scripts that generate every table and figure of the paper from the result files, with their outputs |
| `archive_v0.3/` | Results and the earlier AgentDojo extraction of the first version, kept for the comparisons reported in Section S2 |
| `tests/` | Unit test of the AgentDojo extraction |

## Setup

The results were computed with CPython 3.10.4 on a CPU.

```bash
pip install -r revision/requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

Scripts that read the original benchmarks expect two environment variables pointing to local clones.

```bash
git clone https://github.com/Asif-0209/AgentDrift && git -C AgentDrift checkout 014a514f
git clone https://github.com/ethz-spylab/agentdojo && git -C agentdojo checkout 089ed468
export AGENTDRIFT=$PWD/AgentDrift
export AGENTDOJO=$PWD/agentdojo
```

## Reproducing the paper

Regenerating the tables and figures needs only the files in this repository.

```bash
cd paper
python gen_tables_v04.py      # writes paper/tables/v04_* and s_* (Markdown and LaTeX)
python make_figures_v04.py    # writes paper/fig_v04_*.pdf and .png
```

Re-running the experiments from the benchmarks follows `revision/REPRODUCIBILITY.md`, which lists every script in
execution order with its command line, inputs, outputs, seeds and bootstrap settings. In short:

1. `python agentdojo_extract.py`, then `python tests/test_agentdojo_alignment.py`
2. the first-version experiment scripts (Section 4.2 of the guide)
3. the prediction caches, `bash revision/offline_driver.sh`
4. the analysis streams in `revision/` (Section 4.4 of the guide)

## Notes

- Absolute paths of the machine the analyses ran on were replaced in the result files by `<repo>`, `$AGENTDRIFT`,
  `$AGENTDOJO` or `<local>`. No numeric result was changed. `revision/offline_driver.sh` was also made portable, so its SHA-256 in
  `revision/repro_checks.json` no longer matches. No Python script was modified.
- Comment identifiers such as M3 or m14 in the documentation refer to an internal pre-submission review.
- `GRU_PREREGISTRATION.md` keeps its file name from the first version. The plans were internal and not externally
  registered, and the paper describes what was known when each plan was written.

## Licenses

- Code in this repository is released under the MIT License (see `LICENSE`).
- `agentdojo_traces.jsonl`, `agentdojo_failed.jsonl` and `archive_v0.3/agentdojo_traces.jsonl` are derived from the
  AgentDojo recorded runs, which are released under the MIT License by their authors (see
  `THIRD_PARTY/AgentDojo-LICENSE.txt`).
- Files derived from AgentDrift, such as the tool annotation in `revision/tool_state_annotation.json`, are shared under
  CC BY 4.0 with attribution to Pinjari and Saint-Germain, *AgentDrift: A Step-Labeled Benchmark of
  Injection-Hijacked LLM Agent Trajectories* (2026), https://github.com/Asif-0209/AgentDrift.

## Citation

If you use this code or data, please cite the paper (see `CITATION.cff`). The arXiv identifier will be added once
the preprint is available.
