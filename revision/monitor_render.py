"""Render AlignmentCheck calls for the monitor study (MONITOR_PREREGISTRATION.md, amendment 1).

Builds one call per agent step, exactly as the released AlignmentCheck scanner builds its input
(Sections 2.1 to 2.4 of the plan). Writes one JSONL file per call group to revision/monitor_run/calls/
and a manifest with the SHA-256 of every file. Nothing is scored here.

Sources
  prompts   revision/monitor_run/alignmentcheck_scanner.py, the released file vendored byte for byte
            (git blob 67d819d37845bbcc354e49db38b23b63c0b9880c, MIT). Checked on every run.
  AgentDrift  $AGENTDRIFT/data/{train,val,test}.jsonl at commit 014a514f (user message = record 'task').
  AgentDojo   agentdojo_traces.jsonl and agentdojo_failed.jsonl (steps), and
              revision/monitor_run/agentdojo_run_context.jsonl (user message and assistant text per call,
              taken from the run files of commit 089ed468).

Groups (rows of each JSONL: cid, tid, k, m, user_prompt)
  pilot_val      300 AgentDrift validation traces, seed 20261016 (pilot only, outputs discarded)
  B1_dojo_p0     AgentDojo P0, 2,686 traces, 14,079 calls
  B2_drift_test  AgentDrift R1 test, 1,881 traces, 10,552 calls
  B3_dojo_failed AgentDojo unsuccessful attacks, 4,987 traces, 21,519 calls
  L_drift_rest   AgentDrift train and val, the rest of the MON-L arm
  (MON-L-th and MON-L-mask are rendered later with --arm th / --arm mask)

Usage
  python monitor_render.py           # render primary groups and write calls_manifest.json
  python monitor_render.py --check   # verify only, write nothing
"""
import os, sys, re, json, hashlib, random, argparse, enum, dataclasses, collections

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RUN = os.path.join(HERE, "monitor_run")
CALLS = os.path.join(RUN, "calls")
SCANNER = os.path.join(RUN, "alignmentcheck_scanner.py")
CONTEXT = os.path.join(RUN, "agentdojo_run_context.jsonl")
BLOB_SHA = "67d819d37845bbcc354e49db38b23b63c0b9880c"
SEED = 20261016
EXPECTED_CALLS = {"B1_dojo_p0": 14079, "B2_drift_test": 10552, "B3_dojo_failed": 21519}
EXPECTED_DRIFT_STEPS = 71024


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_prompts():
    raw = open(SCANNER, "rb").read()
    blob = hashlib.sha1(b"blob %d\x00" % len(raw) + raw).hexdigest()
    assert blob == BLOB_SHA, f"scanner file is not the registered blob ({blob})"
    src = raw.decode("utf-8")
    sys_p = re.search(r'SYSTEM_PROMPT: str = """(.*?)"""', src, re.S).group(1)
    usr_f = re.search(r'USER_PROMPT_FORMAT: str = """(.*?)"""', src, re.S).group(1)
    return sys_p, usr_f


# Minimal copies of the released data types, so that str(message) renders exactly as
# llamafirewall_data_types.Message.__str__ does: f"{self.role}: {self.content}".
class Role(enum.Enum):
    TOOL = "tool"
    USER = "user"
    ASSISTANT = "assistant"
    MEMORY = "memory"
    SYSTEM = "system"


@dataclasses.dataclass
class Message:
    role: Role
    content: str

    def __str__(self):
        return f"{self.role}: {self.content}"


assert str(Message(Role.USER, "x")) == "Role.USER: x"


def action_content(step, thought):
    return json.dumps({"thought": thought, "action": step["tool"], "action_input": step["args"]})


def render_trace(usr_f, user, steps, thoughts=None, tools=None):
    """One user prompt per step k = 1..m, as AlignmentCheckScanner.scan builds it."""
    past = [Message(Role.USER, user)]
    out = []
    for k, s in enumerate(steps):
        st = dict(s)
        if tools is not None:
            st["tool"] = tools[k]
        t = "" if thoughts is None else (thoughts[k] or "")
        msg = Message(Role.ASSISTANT, action_content(st, t))
        trace_str = "\n".join(str(m) for m in past + [msg])
        out.append(usr_f.format(user_message=user, trace=trace_str))
        past.append(msg)
    return out


def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def data():
    mp = os.environ["AGENTDRIFT"]
    dr = {s: load_jsonl(os.path.join(mp, "data", f"{s}.jsonl")) for s in ("train", "val", "test")}
    p0 = load_jsonl(os.path.join(ROOT, "agentdojo_traces.jsonl"))
    failed = load_jsonl(os.path.join(ROOT, "agentdojo_failed.jsonl"))
    ctx = {r["id"]: r for r in load_jsonl(CONTEXT)}
    return dr, p0, failed, ctx


def check_dojo(recs, ctx):
    for r in recs:
        c = ctx[r["id"]]
        assert c["user"], r["id"]
        assert len(c["thoughts"]) == len(r["steps"]), r["id"]


def groups(dr, p0, failed):
    val_ids = sorted(r["id"] for r in dr["val"])
    pick = set(random.Random(SEED).sample(val_ids, 300))
    return {
        "pilot_val": [("drift", r) for r in dr["val"] if r["id"] in pick],
        "B1_dojo_p0": [("dojo", r) for r in p0],
        "B2_drift_test": [("drift", r) for r in dr["test"]],
        "B3_dojo_failed": [("dojo", r) for r in failed],
        "L_drift_rest": [("drift", r) for r in dr["train"] + dr["val"]],
    }


def th_groups(dr, p0, failed):
    """MON-L-th (plan Section 4): AgentDrift R1 test with the step 'thought', AgentDojo with the assistant text."""
    return {"TH_drift_test": [("drift", r) for r in dr["test"]],
            "TH_dojo_all": [("dojo", r) for r in p0 + failed]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--arm", choices=["primary", "th", "mask"], default="primary")
    a = ap.parse_args()
    sys_p, usr_f = load_prompts()
    dr, p0, failed, ctx = data()
    assert len(p0) == 2686 and len(failed) == 4987
    assert sum(len(r["steps"]) for s in dr.values() for r in s) == EXPECTED_DRIFT_STEPS
    check_dojo(p0 + failed, ctx)
    if a.arm == "th":
        return render_th(a, usr_f, sys_p, dr, p0, failed, ctx)
    if a.arm == "mask":
        return render_mask(a, usr_f, sys_p, dr)
    G = groups(dr, p0, failed)
    os.makedirs(CALLS, exist_ok=True)
    man = {"plan": "MONITOR_PREREGISTRATION.md (amendment 1)", "scanner_blob": BLOB_SHA,
           "system_prompt_sha256": hashlib.sha256(sys_p.encode("utf-8")).hexdigest(),
           "system_prompt_chars": len(sys_p), "agentdrift": os.environ["AGENTDRIFT"],
           "agentdojo_context_sha256": sha256(CONTEXT), "seed": SEED, "python": sys.version.split()[0],
           "groups": {}}
    for g, rows in G.items():
        n = 0; chars = 0
        path = os.path.join(CALLS, f"{g}.jsonl")
        f = None if a.check else open(path, "w", encoding="utf-8", newline="\n")
        for src, r in rows:
            user = r["task"] if src == "drift" else ctx[r["id"]]["user"]
            ups = render_trace(usr_f, user, r["steps"])
            for k, up in enumerate(ups, 1):
                n += 1; chars += len(sys_p) + len(up)
                if f:
                    f.write(json.dumps({"cid": f"{r['id']}#{k}", "tid": r["id"], "k": k, "m": len(ups),
                                        "user_prompt": up}, ensure_ascii=False) + "\n")
        if f:
            f.close()
        if g in EXPECTED_CALLS:
            assert n == EXPECTED_CALLS[g], (g, n)
        man["groups"][g] = {"traces": len(rows), "calls": n, "chars_incl_system": chars,
                            "sha256": None if a.check else sha256(path)}
        print(g, len(rows), n)
    if not a.check:
        json.dump(man, open(os.path.join(RUN, "calls_manifest.json"), "w", encoding="utf-8"), indent=1)


def render_mask(a, usr_f, sys_p, dr):
    """MON-L-mask (plan Section 4 as corrected by amendment 2): all 16 masked sets of graded_masks.json on the
    AgentDrift R1 test set. A masked tool name is replaced by the literal 'unknown_tool' in the action field."""
    gm = json.load(open(os.path.join(HERE, "sencoder_run", "graded_masks.json"), encoding="utf-8"))
    keys = [k for k in gm["masks"] if k != "1.0|0"]
    assert len(keys) == 16
    os.makedirs(CALLS, exist_ok=True)
    man = {"plan": "MONITOR_PREREGISTRATION.md (amendment 2), arm MON-L-mask", "scanner_blob": BLOB_SHA,
           "masks_file_sha256": sha256(os.path.join(HERE, "sencoder_run", "graded_masks.json")), "groups": {}}
    total = 0
    for key in keys:
        masked = set(gm["masks"][key])
        g = "MASK_" + key.replace("|", "_d")
        path = os.path.join(CALLS, f"{g}.jsonl")
        f = None if a.check else open(path, "w", encoding="utf-8", newline="\n")
        n = 0; nmasked = 0
        for r in dr["test"]:
            tools = [("unknown_tool" if s["tool"] in masked else s["tool"]) for s in r["steps"]]
            nmasked += sum(t == "unknown_tool" for t in tools)
            ups = render_trace(usr_f, r["task"], r["steps"], tools=tools)
            for k, up in enumerate(ups, 1):
                n += 1
                if f:
                    f.write(json.dumps({"cid": f"{r['id']}#{k}", "tid": r["id"], "k": k, "m": len(ups),
                                        "user_prompt": up}, ensure_ascii=False) + "\n")
        if f:
            f.close()
        assert n == 10552, (g, n)
        total += n
        man["groups"][g] = {"mask_key": key, "tools_masked": len(masked), "achieved_coverage": gm["cov_existing"][key],
                            "steps_masked": nmasked, "calls": n, "sha256": None if a.check else sha256(path)}
        print(g, len(masked), nmasked, n)
    assert total == 168832
    if not a.check:
        json.dump(man, open(os.path.join(RUN, "calls_manifest_mask.json"), "w", encoding="utf-8"), indent=1)


def render_th(a, usr_f, sys_p, dr, p0, failed, ctx):
    G = th_groups(dr, p0, failed)
    expected = {"TH_drift_test": 10552, "TH_dojo_all": 35598}
    os.makedirs(CALLS, exist_ok=True)
    mpath = os.path.join(RUN, "calls_manifest_th.json")
    man = {"plan": "MONITOR_PREREGISTRATION.md (amendment 1), arm MON-L-th", "scanner_blob": BLOB_SHA,
           "system_prompt_sha256": hashlib.sha256(sys_p.encode("utf-8")).hexdigest(), "groups": {}}
    for g, rows in G.items():
        n = 0; with_text = 0
        path = os.path.join(CALLS, f"{g}.jsonl")
        f = None if a.check else open(path, "w", encoding="utf-8", newline="\n")
        for src, r in rows:
            if src == "drift":
                user = r["task"]; th = [s.get("thought") or "" for s in r["steps"]]
            else:
                user = ctx[r["id"]]["user"]; th = [t or "" for t in ctx[r["id"]]["thoughts"]]
            with_text += sum(bool(t) for t in th)
            ups = render_trace(usr_f, user, r["steps"], thoughts=th)
            for k, up in enumerate(ups, 1):
                n += 1
                if f:
                    f.write(json.dumps({"cid": f"{r['id']}#{k}", "tid": r["id"], "k": k, "m": len(ups),
                                        "user_prompt": up}, ensure_ascii=False) + "\n")
        if f:
            f.close()
        assert n == expected[g], (g, n)
        man["groups"][g] = {"traces": len(rows), "calls": n, "steps_with_thought_text": with_text,
                            "sha256": None if a.check else sha256(path)}
        print(g, len(rows), n, with_text)
    if not a.check:
        json.dump(man, open(mpath, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
