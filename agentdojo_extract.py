"""Extract AgentDojo recorded runs into featurise-compatible trajectory records.

Label semantics (verified against AgentDojo's own analysis notebook, where ASR = mean(security)):
security == True means the injection task was executed, so the run is COMPROMISED.
  benign        : attack_type is None                      (agent doing the legitimate user task)
  attacked      : attack_type not None AND security == True  (injection realised, positive class)
  attack_failed : attack_type not None AND security == False (written to agentdojo_failed.jsonl)
For denial-of-service attack variants AgentDojo defines security as "not utility"; these runs are
flagged with is_dos = True so that analyses can report results with and without them.

Steps come from assistant tool_calls (function, args). Each call is paired with the tool messages
that follow its assistant message, IN ORDER (positional pairing). Tool-call ids are not used,
because Gemini runs store empty ids and some GPT runs reuse ids (the v0.3 id-based lookup assigned
the wrong observation to steps in about 410 traces). Observations are stored without truncation.

Runs with fewer than two tool calls are excluded; exclusions are counted by class and model.
Outputs (next to this file):
  agentdojo_traces.jsonl   benign + compromised runs  (schema: id, agent=suite, category, steps, meta)
  agentdojo_failed.jsonl   unsuccessful attack runs   (same schema, category = attack_failed)
  agentdojo_extract_manifest.json  counts by class and model, exclusions, alignment statistics
Read-only over the cloned repository (environment variable AGENTDOJO); nothing is executed.
"""
import json, os, glob, collections

ADJ = os.environ["AGENTDOJO"]
MODELS = ["gpt-4o-2024-05-13", "claude-3-5-sonnet-20241022", "gemini-1.5-pro-002",
          "gpt-4-turbo-2024-04-09", "gemini-2.0-flash-001"]
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "agentdojo_traces.jsonl")
OUT_FAILED = os.path.join(HERE, "agentdojo_failed.jsonl")
OUT_MANIFEST = os.path.join(HERE, "agentdojo_extract_manifest.json")


def _fn_name(tc):
    f = tc.get("function")
    return f if isinstance(f, str) else (f or {}).get("name")


def steps_from(messages, stats=None):
    """Pair every assistant tool call with the tool messages that follow it, in order.

    Returns a list of {tool, args, obs}. If an assistant turn has more calls than following tool
    messages, the unmatched calls get an empty observation (counted in stats['unmatched_calls']).
    """
    steps = []
    i, n = 0, len(messages)
    while i < n:
        m = messages[i]
        if m.get("role") == "assistant" and (m.get("tool_calls") or []):
            calls = [tc for tc in m["tool_calls"] if _fn_name(tc)]
            # collect the consecutive tool messages that answer this assistant turn
            j = i + 1
            tool_msgs = []
            while j < n and messages[j].get("role") == "tool":
                tool_msgs.append(messages[j]); j += 1
            for k, tc in enumerate(calls):
                if k < len(tool_msgs):
                    tm = tool_msgs[k]
                    obs = tm.get("content") if not tm.get("error") else str(tm.get("error"))
                    if stats is not None:
                        tcall = tm.get("tool_call") or {}
                        if _fn_name(tcall) and _fn_name(tcall) != _fn_name(tc):
                            stats["name_mismatch"] += 1
                else:
                    obs = ""
                    if stats is not None:
                        stats["unmatched_calls"] += 1
                steps.append({"tool": _fn_name(tc), "args": tc.get("args", {}) or {}, "obs": str(obs if obs is not None else "")})
            if stats is not None and len(tool_msgs) > len(calls):
                stats["extra_tool_msgs"] += len(tool_msgs) - len(calls)
            i = j
        else:
            i += 1
    return steps


def main():
    counts = collections.Counter()
    by_model = collections.defaultdict(collections.Counter)
    stats = collections.Counter()
    with open(OUT, "w", encoding="utf-8") as w, open(OUT_FAILED, "w", encoding="utf-8") as wf:
        for model in MODELS:
            for f in sorted(glob.glob(os.path.join(ADJ, "runs", model, "*", "*", "*", "*.json"))):
                try:
                    d = json.load(open(f, encoding="utf-8"))
                except Exception:
                    counts["unreadable"] += 1; continue
                at = d.get("attack_type"); sec = d.get("security")
                if at is None:
                    cat = "benign"
                elif sec is True:
                    cat = "attacked"
                else:
                    cat = "attack_failed"
                steps = steps_from(d.get("messages", []), stats)
                by_model[model][f"{cat}_all"] += 1
                if len(steps) < 2:
                    by_model[model][f"{cat}_excluded_short"] += 1; counts[f"{cat}_excluded_short"] += 1
                    continue
                rid = f"{model}|{d.get('suite_name')}|{d.get('user_task_id')}|{d.get('injection_task_id')}|{at}"
                meta = {"model": model, "suite": d.get("suite_name"), "user_task": d.get("user_task_id"),
                        "injection_task": d.get("injection_task_id"), "attack_type": at,
                        "is_dos": bool(at and "dos" in str(at).lower()), "utility": d.get("utility"),
                        "security": sec, "n_steps": len(steps)}
                rec = {"id": rid, "agent": d.get("suite_name"), "category": cat, "source_category": cat,
                       "steps": steps, "meta": meta}
                (wf if cat == "attack_failed" else w).write(json.dumps(rec) + "\n")
                counts[cat] += 1; by_model[model][cat] += 1
    manifest = {"counts_kept": {k: v for k, v in counts.items() if not k.endswith("_short") and k != "unreadable"},
                "excluded_fewer_than_two_calls": {k: v for k, v in counts.items() if k.endswith("_short")},
                "unreadable": counts.get("unreadable", 0), "by_model": {m: dict(c) for m, c in by_model.items()},
                "alignment": dict(stats), "truncation": "none"}
    json.dump(manifest, open(OUT_MANIFEST, "w", encoding="utf-8"), indent=1)
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
