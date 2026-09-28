"""Extract AgentDojo recorded runs into featurise-compatible trajectory records.

Label semantics (VERIFIED against AgentDojo's own analysis notebook, where
ASR = mean(security)):  security==True  => the injection task was executed => COMPROMISED.
  benign   : attack_type is None                     (agent doing the legitimate user task)
  attacked : attack_type not None AND security==True  (injection realised = positive class)
  (attack_failed: attack_type not None AND security==False -> EXCLUDED from the test set)

Trajectory steps come from assistant tool_calls (function, args) paired with the following
tool message (observation). We pool several base models (no defense variants) for volume/diversity.
Output: agentdojo_traces.jsonl (schema: id, agent=suite, category, steps=[{tool,args,obs}]).
Read-only over the cloned repo; no execution.
"""
import json, os, glob

ADJ = os.environ["AGENTDOJO"]
MODELS = ["gpt-4o-2024-05-13", "claude-3-5-sonnet-20241022", "gemini-1.5-pro-002",
          "gpt-4-turbo-2024-04-09", "gemini-2.0-flash-001"]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agentdojo_traces.jsonl")


def steps_from(messages):
    # map tool_call_id -> observation
    obs = {}
    for m in messages:
        if m.get("role") == "tool":
            obs[m.get("tool_call_id")] = m.get("content") if not m.get("error") else str(m.get("error"))
    steps = []
    for m in messages:
        if m.get("role") == "assistant":
            for tc in (m.get("tool_calls") or []):
                fn = tc.get("function") if isinstance(tc.get("function"), str) else (tc.get("function") or {}).get("name")
                if not fn: continue
                steps.append({"tool": fn, "args": tc.get("args", {}) or {},
                              "obs": str(obs.get(tc.get("id"), ""))[:2000]})
    return steps


def main():
    n = {"benign": 0, "attacked": 0, "skipped_failed": 0, "empty": 0}
    with open(OUT, "w", encoding="utf-8") as w:
        for model in MODELS:
            for f in glob.glob(os.path.join(ADJ, "runs", model, "*", "*", "*", "*.json")):
                try:
                    d = json.load(open(f, encoding="utf-8"))
                except Exception:
                    continue
                at = d.get("attack_type"); sec = d.get("security")
                if at is None:
                    cat = "benign"
                elif sec is True:
                    cat = "attacked"
                else:
                    n["skipped_failed"] += 1; continue
                steps = steps_from(d.get("messages", []))
                if len(steps) < 2:
                    n["empty"] += 1; continue
                rid = f"{model}|{d.get('suite_name')}|{d.get('user_task_id')}|{d.get('injection_task_id')}|{at}"
                rec = {"id": rid, "agent": d.get("suite_name"), "category": cat,
                       "source_category": cat, "steps": steps}
                w.write(json.dumps(rec) + "\n"); n[cat] += 1
    print("AgentDojo extraction:", n)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
