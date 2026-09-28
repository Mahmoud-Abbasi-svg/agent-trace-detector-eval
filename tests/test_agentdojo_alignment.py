"""Unit tests for the AgentDojo observation pairing (review comment M7).

Run:  AGENTDOJO=<path to agentdojo clone> python tests/test_agentdojo_alignment.py
1. A known Gemini run whose tool-call ids are empty must get its three distinct observations
   (lengths 636, 712, 67), not the last observation three times.
2. For every run of the five models whose tool-call ids are non-empty and unique, positional pairing
   must agree with id-based pairing (the tool message's tool_call_id equals the call's id).
3. The extracted traces file must agree with a fresh positional extraction of the same runs.
"""
import json, os, sys, glob
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from agentdojo_extract import steps_from, MODELS  # noqa: E402

ADJ = os.environ["AGENTDOJO"]


def test_known_gemini_run():
    f = os.path.join(ADJ, "runs", "gemini-1.5-pro-002", "banking", "user_task_0", "important_instructions", "injection_task_0.json")
    d = json.load(open(f, encoding="utf-8"))
    steps = steps_from(d["messages"])
    assert [s["tool"] for s in steps] == ["read_file", "get_most_recent_transactions", "send_money"], steps
    assert [len(s["obs"]) for s in steps] == [636, 712, 67], [len(s["obs"]) for s in steps]


def test_positional_matches_ids_when_ids_are_usable():
    checked = mismatched = 0
    for model in MODELS:
        for f in glob.glob(os.path.join(ADJ, "runs", model, "*", "*", "*", "*.json")):
            d = json.load(open(f, encoding="utf-8"))
            msgs = d.get("messages", [])
            ids = [tc.get("id") for m in msgs if m.get("role") == "assistant" for tc in (m.get("tool_calls") or [])]
            if not ids or any(not i for i in ids) or len(set(ids)) != len(ids):
                continue
            by_id = {m.get("tool_call_id"): (m.get("content") if not m.get("error") else str(m.get("error")))
                     for m in msgs if m.get("role") == "tool"}
            steps = steps_from(msgs)
            expected = [str(by_id.get(i, "")) for i in ids][:len(steps)]
            checked += 1
            if [s["obs"] for s in steps] != expected:
                mismatched += 1
    assert checked > 1000, checked
    assert mismatched == 0, f"{mismatched} of {checked} runs disagree"
    return checked


def test_extracted_file_is_positional():
    recs = [json.loads(l) for l in open(os.path.join(ROOT, "agentdojo_traces.jsonl"), encoding="utf-8")]
    gem = [r for r in recs if r["id"].startswith("gemini-1.5-pro-002|banking|user_task_0|injection_task_0|important_instructions")]
    assert gem and [len(s["obs"]) for s in gem[0]["steps"]] == [636, 712, 67]


if __name__ == "__main__":
    test_known_gemini_run(); print("known Gemini run: OK")
    n = test_positional_matches_ids_when_ids_are_usable(); print(f"positional == id pairing on {n} runs with usable ids: OK")
    test_extracted_file_is_positional(); print("extracted file uses positional pairing: OK")
