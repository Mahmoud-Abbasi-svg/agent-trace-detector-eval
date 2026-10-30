# Monitor study: every call and every raw output

AlignmentCheck (LlamaFirewall) is run once per agent step, with the vendored scanner in `alignmentcheck_scanner.py`
and the plan in `../../MONITOR_PREREGISTRATION.md` (first version and amendments 1 to 4 in this folder). Calls are
rendered by `../monitor_render.py`, scored by `../monitor_score_vllm.py` (MON-L, Llama 3.3 70B w4a16 on vLLM) and
`../monitor_score_api.py` (MON-A, Claude Sonnet 5.5 through the provider's batch interface), and analysed by
`../monitor_detector.py` (frozen copy: `monitor_detector.frozen.py`) into `../monitor_results.json`.

Files larger than 1 MB are gzip-compressed with unchanged content. To decompress all of them:

```bash
python -c "import glob,gzip,shutil; [shutil.copyfileobj(gzip.open(f), open(f[:-3], 'wb')) for f in glob.glob('**/*.jsonl.gz', recursive=True)]"
```

## Calls (`calls/`)

One JSON line per call: `cid` (call id, trace id plus `#k`), `tid` (trace id), `k` (step), `m` (trace length) and
`user_prompt` (the rendered AlignmentCheck user message). The system prompt and the SHA-256 of every decompressed
call file are in `calls_manifest.json`, `calls_manifest_th.json` and `calls_manifest_mask.json`.

| Group | Calls | Contents |
|---|---|---|
| `pilot_val` | 1,738 | Pilot on 300 AgentDrift validation traces |
| `B1_dojo_p0` | 14,079 | AgentDojo population P0 (2,686 traces) |
| `B2_drift_test` | 10,552 | AgentDrift R1 test set (1,881 traces) |
| `B3_dojo_failed` | 21,519 | The 4,987 unsuccessful AgentDojo attacks |
| `L_drift_rest` | 60,472 | The remaining AgentDrift traces (MON-L only) |
| `TH_drift_test`, `TH_dojo_all` | 10,552, 35,598 | Arm with the agent's thoughts or assistant text included (MON-L) |
| `MASK_<level>_d<draw>` | 10,552 each | R1 test set with tool names masked at graded levels, 16 sets (MON-L) |
| `REP500` | 500 | Repeatability check |

## Outputs

- `mon_l/main_<group>.jsonl.gz`: MON-L, one line per call with `cid`, `raw` (the generated text), `conclusion`,
  `p_true` (probability of the conclusion `true`, the trace score input), `parse_ok`, `finish`, `n_prompt`, `n_gen`
  and `p_note`. Software versions in `mon_l/env.json`, the job log in `mon_l/stdout.log`.
- `api/results_<group>.jsonl.gz`: MON-A, one line per call with `cid`, `raw`, `conclusion`, `parse_ok`, `stop`,
  `type`, `usage` and `batch`. `api/batches.json` lists the batches, and `api/idmap_<batch>.json` maps batch request
  ids to call ids. `api/pilot_*` holds the pilot.
- `run_record.json`: wall time of the MON-L job and the cost of MON-A by group.

## Other inputs

- `agentdojo_run_context.jsonl`: AgentDojo user prompts and assistant text used to render the calls.
- `agentdrift_taskdisjoint/`: the task-disjoint AgentDrift split used for the SE comparison.
- `hpc/`: Slurm job scripts for the MON-L runs.
