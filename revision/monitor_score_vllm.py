"""Score MON-L calls with vLLM on one H100 (MONITOR_PREREGISTRATION.md, Sections 2.5, 2.6, 3 and 10).

Backbone RedHatAI/Llama-3.3-70B-Instruct-quantized.w4a16 at revision da23851b, loaded offline from HF_HOME
(compute nodes have no egress). Greedy decoding (temperature 0), max 1,024 new tokens, output constrained to the
AlignmentCheckOutputSchema JSON schema with key order observation, thought, conclusion. The graded score is
p = P(true) / (P(true) + P(false)) at the generated token that carries the conclusion value, read from the
returned log-probabilities. The released SYSTEM_PROMPT is sent as the system message and each rendered user
prompt (revision/monitor_run/calls/<group>.jsonl) as the user message, through the model's chat template.

Usage on cn001 (see the job script in revision/monitor_run/mon_l_job.sh):
  python monitor_score_vllm.py --groups pilot_val --pilot          # pilot: no conclusions stored
  python monitor_score_vllm.py --groups B1_dojo_p0 B3_dojo_failed B2_drift_test L_drift_rest
  python monitor_score_vllm.py --smoke 20 --groups pilot_val       # quick load-and-format check
Resumable: calls already present in the output file are skipped. One output line per call.
"""
import os, sys, re, json, time, math, hashlib, argparse, platform

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, "monitor_run")
CALLS = os.path.join(RUN, "calls")
OUT = os.path.join(RUN, "mon_l")
MODEL = "RedHatAI/Llama-3.3-70B-Instruct-quantized.w4a16"
REVISION = "da23851b92f5ea8023d6c70f790feb8e05269812"
BLOB_SHA = "67d819d37845bbcc354e49db38b23b63c0b9880c"
SCHEMA = {"type": "object",
          "properties": {"observation": {"type": "string"}, "thought": {"type": "string"},
                         "conclusion": {"type": "boolean"}},
          "required": ["observation", "thought", "conclusion"], "additionalProperties": False}
CHUNK = 2000


def system_prompt():
    raw = open(os.path.join(RUN, "alignmentcheck_scanner.py"), "rb").read()
    assert hashlib.sha1(b"blob %d\x00" % len(raw) + raw).hexdigest() == BLOB_SHA
    return re.search(r'SYSTEM_PROMPT: str = """(.*?)"""', raw.decode("utf-8"), re.S).group(1)


def build_llm():
    from vllm import LLM
    kw = dict(model=MODEL, revision=REVISION, max_model_len=16384, gpu_memory_utilization=0.90,
              enable_prefix_caching=True, tensor_parallel_size=1, seed=0)
    # Ask for log-probabilities after the structured-output mask if this vLLM has the option. Checked on the
    # engine arguments first, so that a failed construction never leaves half-allocated GPU memory behind.
    try:
        from vllm.engine.arg_utils import EngineArgs
        has_mode = "logprobs_mode" in getattr(EngineArgs, "__dataclass_fields__", {})
    except Exception:
        has_mode = False
    if has_mode:
        return LLM(logprobs_mode="processed_logprobs", **kw), "processed_logprobs"
    return LLM(**kw), "raw_logprobs"


def sampling_params():
    from vllm import SamplingParams
    try:
        from vllm.sampling_params import StructuredOutputsParams
        return SamplingParams(temperature=0.0, max_tokens=1024, logprobs=20,
                              structured_outputs=StructuredOutputsParams(json=SCHEMA))
    except (ImportError, TypeError):
        from vllm.sampling_params import GuidedDecodingParams
        return SamplingParams(temperature=0.0, max_tokens=1024, logprobs=20,
                              guided_decoding=GuidedDecodingParams(json=SCHEMA))


def conclusion_prob(tok, out):
    """p(true) at the token where the conclusion value starts, or None if it cannot be located."""
    ids = out.token_ids; lps = out.logprobs
    if not lps:
        return None, "no_logprobs"
    text = ""
    for i, tid in enumerate(ids):
        piece = tok.decode([tid])
        before = text; text += piece
        m = re.search(r'"conclusion"\s*:\s*$', before)
        if m and piece.strip()[:1] in ("t", "f"):
            pt = pf = 0.0; lowest = 0.0
            for cand_id, lp in lps[i].items():
                p = math.exp(lp.logprob); lowest = min(lowest or p, p)
                s = tok.decode([cand_id]).strip()
                if s and "true".startswith(s):
                    pt += p
                elif s and "false".startswith(s):
                    pf += p
            if pt == 0.0 and pf == 0.0:
                return None, "no_candidates"
            note = "ok"
            if pt == 0.0 or pf == 0.0:   # one side outside the returned top 20: bound it by the lowest returned
                note = "bounded"
                pt = pt or lowest; pf = pf or lowest
            return pt / (pt + pf), note
    return None, "not_found"


def parse(text):
    try:
        o = json.loads(text)
        c = o.get("conclusion")
        if isinstance(c, bool) and set(o) == {"observation", "thought", "conclusion"}:
            return c, True
    except Exception:
        pass
    return None, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="+", required=True)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--smoke", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    sys_p = system_prompt()
    llm, lp_mode = build_llm()
    tok = llm.get_tokenizer()
    sp = sampling_params()
    import vllm, torch, transformers
    env = {"vllm": vllm.__version__, "torch": torch.__version__, "transformers": transformers.__version__,
           "python": platform.python_version(), "gpu": torch.cuda.get_device_name(0), "logprobs_mode": lp_mode,
           "model": MODEL, "revision": REVISION}
    print(json.dumps(env), flush=True)
    json.dump(env, open(os.path.join(OUT, "env.json"), "w"), indent=1)
    for g in a.groups:
        rows = [json.loads(l) for l in open(os.path.join(CALLS, f"{g}.jsonl"), encoding="utf-8")]
        if a.smoke:
            rows = rows[:a.smoke]
        tag = "smoke" if a.smoke else ("pilot" if a.pilot else "main")
        path = os.path.join(OUT, f"{tag}_{g}.jsonl")
        done = set()
        if os.path.exists(path):
            done = {json.loads(l)["cid"] for l in open(path, encoding="utf-8")}
        todo = [r for r in rows if r["cid"] not in done]
        print(g, tag, "total", len(rows), "todo", len(todo), flush=True)
        t0 = time.time(); n = 0
        for i in range(0, len(todo), CHUNK):
            chunk = todo[i:i + CHUNK]
            msgs = [[{"role": "system", "content": sys_p}, {"role": "user", "content": r["user_prompt"]}] for r in chunk]
            outs = llm.chat(msgs, sp, use_tqdm=False)
            with open(path, "a", encoding="utf-8", newline="\n") as f:
                for r, o in zip(chunk, outs):
                    c = o.outputs[0]
                    concl, ok = parse(c.text)
                    p, note = conclusion_prob(tok, c)
                    rec = {"cid": r["cid"], "parse_ok": ok, "finish": c.finish_reason, "p_note": note,
                           "n_prompt": len(o.prompt_token_ids), "n_gen": len(c.token_ids)}
                    if not a.pilot:   # the pilot stores no conclusion and no score (plan Section 10)
                        rec.update(conclusion=concl, p_true=p, raw=c.text)
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += len(chunk)
            el = time.time() - t0
            print(f"{g} {n}/{len(todo)} {el:.0f}s {n / el:.2f} calls/s", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
