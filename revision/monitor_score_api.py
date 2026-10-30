"""Score MON-A (claude-sonnet-5-5) calls through the Anthropic API (MONITOR_PREREGISTRATION.md, amendment 1).

Settings fixed by the plan (Sections 2.5 and 3) and by the settings probe of 21 Oct 2026:
  model claude-sonnet-5-5, max_tokens 1024, thinking between_tools, effort high, structured output
  (output_config.format json_schema of AlignmentCheckOutputSchema), default sampling because the API
  rejects temperature for this model. The released SYSTEM_PROMPT is sent as a cached system block.
Errors and unparseable outputs are recorded and handled later as the released scanner does (conclusion true).

Commands (key in env MAHMOUDABBASIAPIKEY or ANTHROPIC_API_KEY):
  python monitor_score_api.py pilot                 # 200-call pilot from calls/pilot_val.jsonl, synchronous
  python monitor_score_api.py submit B1_dojo_p0     # submit a group as Message Batches (chunks of <= 5000)
  python monitor_score_api.py submit B3_dojo_failed --tids FILE   # optional trace-id subset (B3 sample)
  python monitor_score_api.py status                # poll every submitted batch
  python monitor_score_api.py collect               # download results of ended batches
  python monitor_score_api.py cost                  # running cost from recorded usage
State and outputs live in revision/monitor_run/api/.
"""
import os, sys, re, json, time, random, argparse, hashlib, concurrent.futures as cf
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, "monitor_run")
CALLS = os.path.join(RUN, "calls")
API = os.path.join(RUN, "api")
URL = "https://api.anthropic.com/v1"
MODEL = "claude-sonnet-5-5"
SEED = 20261016
CHUNK = 5000
CAP_USD = 95.0
# USD per million tokens on 16 Oct 2026 (plan Section 3). Batch halves input and output. Cache components are
# charged without the batch discount here, which overstates batch cost slightly (conservative).
PRICE = {"in": 2.0, "out": 10.0, "cache_read": 0.10, "cache_write_5m": 2.50, "cache_write_1h": 4.0}
SCHEMA = {"type": "object",
          "properties": {"observation": {"type": "string"}, "thought": {"type": "string"},
                         "conclusion": {"type": "boolean"}},
          "required": ["observation", "thought", "conclusion"], "additionalProperties": False}


def key():
    k = os.environ.get("MAHMOUDABBASIAPIKEY") or os.environ.get("ANTHROPIC_API_KEY")
    assert k, "no API key in the environment"
    return k


def headers():
    return {"x-api-key": key(), "anthropic-version": "2023-06-01", "content-type": "application/json"}


def system_prompt():
    src = open(os.path.join(RUN, "alignmentcheck_scanner.py"), "rb").read()
    assert hashlib.sha1(b"blob %d\x00" % len(src) + src).hexdigest() == "67d819d37845bbcc354e49db38b23b63c0b9880c"
    return re.search(r'SYSTEM_PROMPT: str = """(.*?)"""', src.decode("utf-8"), re.S).group(1)


def params(user_prompt, ttl):
    return {"model": MODEL, "max_tokens": 1024,
            "system": [{"type": "text", "text": system_prompt(), "cache_control": {"type": "ephemeral", "ttl": ttl}}],
            "messages": [{"role": "user", "content": user_prompt}],
            "thinking": {"type": "between_tools"},
            "output_config": {"effort": "high", "format": {"type": "json_schema", "schema": SCHEMA}}}


def parse_message(msg):
    """Return (conclusion or None, parse_ok, text)."""
    text = "".join(b.get("text", "") for b in msg.get("content", []) if b.get("type") == "text")
    try:
        o = json.loads(text)
        c = o.get("conclusion")
        if isinstance(c, bool) and set(o) == {"observation", "thought", "conclusion"}:
            return c, True, text
    except Exception:
        pass
    return None, False, text


def cost_of(usage, batch):
    f = 0.5 if batch else 1.0
    cc = usage.get("cache_creation") or {}
    w5 = cc.get("ephemeral_5m_input_tokens", 0); w1 = cc.get("ephemeral_1h_input_tokens", 0)
    if not cc:
        w5 = usage.get("cache_creation_input_tokens", 0)
    return (usage.get("input_tokens", 0) * PRICE["in"] * f + usage.get("output_tokens", 0) * PRICE["out"] * f
            + usage.get("cache_read_input_tokens", 0) * PRICE["cache_read"]
            + w5 * PRICE["cache_write_5m"] + w1 * PRICE["cache_write_1h"]) / 1e6


def load_calls(group, tids=None):
    with open(os.path.join(CALLS, f"{group}.jsonl"), encoding="utf-8") as f:
        rows = [json.loads(l) for l in f]
    if tids is not None:
        rows = [r for r in rows if r["tid"] in tids]
    return rows


def append(path, rec):
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def cmd_pilot(a):
    os.makedirs(API, exist_ok=True)
    rows = load_calls("pilot_val")
    pick = sorted(random.Random(SEED).sample(range(len(rows)), 200))
    rows = [rows[i] for i in pick]
    out = os.path.join(API, "pilot_results.jsonl")
    assert not os.path.exists(out), "pilot already run"
    t0 = time.time()

    def one(r):
        for attempt in range(6):
            resp = requests.post(f"{URL}/messages", headers=headers(), json=params(r["user_prompt"], "5m"), timeout=300)
            if resp.status_code in (429, 500, 502, 503, 529):
                time.sleep(2 ** attempt * 2); continue
            break
        rec = {"cid": r["cid"], "status": resp.status_code}
        if resp.status_code == 200:
            j = resp.json(); c, ok, _ = parse_message(j)
            rec.update(parse_ok=ok, stop=j.get("stop_reason"), usage=j.get("usage"))
        else:
            rec["error"] = resp.text[:500]
        return rec

    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        recs = list(ex.map(one, rows))
    wall = time.time() - t0
    for rec in recs:
        append(out, rec)   # conclusions are deliberately not stored for the pilot (outputs discarded)
    ok = [r for r in recs if r["status"] == 200]
    summ = {"calls": len(recs), "http_ok": len(ok), "parse_ok": sum(r.get("parse_ok", False) for r in ok),
            "wall_s": round(wall, 1), "cost_usd_sync": round(sum(cost_of(r["usage"], False) for r in ok), 4)}
    keys = ["input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"]
    summ["mean_usage"] = {k: round(sum((r["usage"] or {}).get(k, 0) for r in ok) / max(1, len(ok)), 1) for k in keys}
    json.dump(summ, open(os.path.join(API, "pilot_summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))


def state_path():
    return os.path.join(API, "batches.json")


def load_state():
    p = state_path()
    return json.load(open(p)) if os.path.exists(p) else {"batches": []}


def save_state(s):
    json.dump(s, open(state_path(), "w"), indent=1)


def cmd_submit(a):
    os.makedirs(API, exist_ok=True)
    tids = None
    if a.tids:
        tids = set(l.strip() for l in open(a.tids, encoding="utf-8") if l.strip())
    rows = load_calls(a.group, tids)
    s = load_state()
    done = {b["group"] for b in s["batches"]}
    assert a.group not in done or a.force, f"{a.group} already submitted"
    for i in range(0, len(rows), CHUNK):
        chunk = rows[i:i + CHUNK]
        reqs = [{"custom_id": hashlib.sha1(r["cid"].encode()).hexdigest()[:32], "params": params(r["user_prompt"], "1h")}
                for r in chunk]
        idmap = {q["custom_id"]: r["cid"] for q, r in zip(reqs, chunk)}
        assert len(idmap) == len(chunk)
        resp = requests.post(f"{URL}/messages/batches", headers=headers(), json={"requests": reqs}, timeout=600)
        assert resp.status_code == 200, resp.text[:500]
        b = resp.json()
        json.dump(idmap, open(os.path.join(API, f"idmap_{b['id']}.json"), "w"))
        s["batches"].append({"group": a.group, "id": b["id"], "n": len(chunk), "submitted": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                             "status": b.get("processing_status")})
        save_state(s)
        print(a.group, b["id"], len(chunk), b.get("processing_status"))


def cmd_status(a):
    s = load_state()
    for b in s["batches"]:
        r = requests.get(f"{URL}/messages/batches/{b['id']}", headers=headers(), timeout=60).json()
        b["status"] = r.get("processing_status"); b["counts"] = r.get("request_counts"); b["results_url"] = r.get("results_url")
        print(b["group"], b["id"], b["status"], b["counts"])
    save_state(s)


def cmd_collect(a):
    s = load_state()
    for b in s["batches"]:
        out = os.path.join(API, f"results_{b['group']}.jsonl")
        if b.get("collected") or b.get("status") != "ended" or not b.get("results_url"):
            continue
        idmap = json.load(open(os.path.join(API, f"idmap_{b['id']}.json")))
        resp = requests.get(b["results_url"], headers=headers(), timeout=600, stream=True)
        assert resp.status_code == 200, resp.text[:300]
        n = 0
        for line in resp.iter_lines():
            if not line:
                continue
            r = json.loads(line); res = r["result"]
            rec = {"cid": idmap[r["custom_id"]], "batch": b["id"], "type": res["type"]}
            if res["type"] == "succeeded":
                msg = res["message"]; c, ok, text = parse_message(msg)
                rec.update(conclusion=c, parse_ok=ok, stop=msg.get("stop_reason"), usage=msg.get("usage"), raw=text)
            else:
                rec["error"] = res.get("error")
            append(out, rec); n += 1
        b["collected"] = True; b["collected_n"] = n
        save_state(s)
        print("collected", b["group"], b["id"], n)


def cmd_cost(a):
    tot = 0.0; by = {}
    p = os.path.join(API, "pilot_results.jsonl")
    if os.path.exists(p):
        for l in open(p, encoding="utf-8"):
            r = json.loads(l)
            if r.get("usage"):
                c = cost_of(r["usage"], False); tot += c; by["pilot"] = by.get("pilot", 0) + c
    for f in sorted(os.listdir(API)):
        if f.startswith("results_"):
            g = f[8:-6]
            for l in open(os.path.join(API, f), encoding="utf-8"):
                r = json.loads(l)
                if r.get("usage"):
                    c = cost_of(r["usage"], True); tot += c; by[g] = by.get(g, 0) + c
    print(json.dumps({"total_usd": round(tot, 2), "cap_usd": CAP_USD, "by_group": {k: round(v, 2) for k, v in by.items()}}, indent=1))


def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("pilot"); p.add_argument("--workers", type=int, default=8)
    p = sp.add_parser("submit"); p.add_argument("group"); p.add_argument("--tids"); p.add_argument("--force", action="store_true")
    sp.add_parser("status"); sp.add_parser("collect"); sp.add_parser("cost")
    a = ap.parse_args()
    {"pilot": cmd_pilot, "submit": cmd_submit, "status": cmd_status, "collect": cmd_collect, "cost": cmd_cost}[a.cmd](a)


if __name__ == "__main__":
    main()
