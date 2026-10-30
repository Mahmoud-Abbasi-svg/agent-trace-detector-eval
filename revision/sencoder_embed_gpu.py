"""Embedding cache for the SE detector, SENCODER_PREREGISTRATION.md Sec. 4 as amended by Sec. 12 and 13 (amendments 3, 4).

Step text is sencoder_text.step_text (amendment 3 format, THOUGHT dropped per D6). Encoder, pooling, normalisation
and maximum sequence length are those of revision/sencoder_embed.py (MODEL_NAME, MAX_SEQ imported from it).

Outputs under $SENCODER_OUT (default revision/cache/sencoder_gpu/):
  <source>.npz        ids [n,2] (trace index, step index), emb float32 [n,768] L2-normalised, trace_ids [m]
                      sources agentdrift_train/val/test, agentdojo, agentdojo_failed (D9)
  masked_r1.npz       R1 test pool-E steps under every graded-coverage mask (tool -> 'unknown_tool')
  padded_r1.npz       R1 test pool-E steps with n filler characters appended to the observation of every
                      injection_point step of attacked trajectories, n in PADS (Sec. 12.3)
  manifest.json       versions, revision, truncation counts, SHA-256 of inputs, CPU check, task-disjoint check

Usage: python sencoder_embed_gpu.py [--device cuda] [--batch 128] [--cpu-check 1000]
"""
import argparse, json, os, sys, time, platform, hashlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import sencoder_embed as se  # noqa: E402  (unmodified; MODEL_NAME, MAX_SEQ, sha256, version 1 step_text)
import sencoder_text as stx  # noqa: E402

OUT = os.environ.get("SENCODER_OUT", os.path.join(HERE, "cache", "sencoder_gpu"))
AGENTDRIFT = se.AGENTDRIFT
BUDGET_S = 2 * 3600
PADS = [0, 100, 1200]
POOL_E = ("benign", "hard_negative", "attacked")


def read(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def flat(recs, text_fn):
    ids, texts = [], []
    for ti, r in enumerate(recs):
        for si, st in enumerate(r.get("steps") or []):
            ids.append((ti, si)); texts.append(text_fn(st))
    return np.asarray(ids, np.int64), texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--cpu-check", type=int, default=1000)
    ap.add_argument("--masks", default=os.path.join(HERE, "sencoder_run", "graded_masks.json"))
    a = ap.parse_args()
    T0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    srcs = {"agentdrift_train": os.path.join(AGENTDRIFT, "data", "train.jsonl"),
            "agentdrift_val": os.path.join(AGENTDRIFT, "data", "val.jsonl"),
            "agentdrift_test": os.path.join(AGENTDRIFT, "data", "test.jsonl"),
            "agentdojo": os.path.join(ROOT, "agentdojo_traces.jsonl"),
            "agentdojo_failed": os.path.join(ROOT, "agentdojo_failed.jsonl")}
    import torch, transformers, sentence_transformers
    from sentence_transformers import SentenceTransformer
    from huggingface_hub import snapshot_download
    if a.device == "cpu":
        torch.set_num_threads(max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))))
    model = SentenceTransformer(se.MODEL_NAME, device=a.device)
    model.max_seq_length = se.MAX_SEQ
    model.eval()
    tok = model.tokenizer
    snap = snapshot_download(se.MODEL_NAME, local_files_only=True)
    mods = [type(m).__name__ for m in model]
    assert "Pooling" in mods and "Normalize" in mods, mods
    _pm = [m for m in model if type(m).__name__ == "Pooling"][0]
    try:
        pool_mode = _pm.get_config_dict()
    except Exception:
        pool_mode = {k: v for k, v in vars(_pm).items() if k.startswith("pooling_mode") or k in ("word_embedding_dimension",)}
    pool_mode = json.loads(json.dumps(pool_mode, default=str))

    def enc(texts):
        return model.encode(texts, batch_size=a.batch, convert_to_numpy=True, normalize_embeddings=True,
                            show_progress_bar=False).astype(np.float32)

    def ntrunc(texts):
        over = 0
        for i in range(0, len(texts), 512):
            over += sum(1 for x in tok(texts[i:i + 512], truncation=False)["input_ids"] if len(x) > se.MAX_SEQ)
        return over

    man = {"model": se.MODEL_NAME, "revision": os.path.basename(snap), "pooling_config": pool_mode, "modules": mods,
           "normalised": True, "max_seq_length": se.MAX_SEQ, "text_format": "TOOL: {tool} | ARGS: {json, sorted keys} | OBS: {obs}",
           "thought_excluded": True, "device": a.device,
           "gpu": torch.cuda.get_device_name(0) if a.device.startswith("cuda") else None, "batch_size": a.batch, "torch_threads": torch.get_num_threads(),
           "torch": torch.__version__, "torch_cuda": torch.version.cuda, "transformers": transformers.__version__,
           "sentence_transformers": sentence_transformers.__version__, "python": platform.python_version(),
           "numpy": np.__version__, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
           "built": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "code_sha256": {f: se.sha256(os.path.join(HERE, f)) for f in ("sencoder_embed.py", "sencoder_text.py", "sencoder_embed_gpu.py")},
           "sources": {}}
    allt = {}
    R = {}
    for name, path in srcs.items():
        t0 = time.time()
        recs = read(path); R[name] = recs
        ids, texts = flat(recs, stx.step_text)
        emb = enc(texts)
        np.savez_compressed(os.path.join(OUT, f"{name}.npz"), ids=ids, emb=emb,
                            trace_ids=np.asarray([r["id"] for r in recs], dtype=object))
        allt[name] = texts
        over = ntrunc(texts)
        man["sources"][name] = {"path": path, "sha256": se.sha256(path), "n_traces": len(recs), "n_steps": len(texts),
                                "n_steps_truncated": over, "pct_truncated": round(100.0 * over / max(1, len(texts)), 3),
                                "seconds": round(time.time() - t0, 1)}
        print(f"[{name}] {len(recs)} traces, {len(texts)} steps, {over} truncated, {time.time()-t0:.0f}s", flush=True)
        if time.time() - T0 > BUDGET_S:
            man["stopped"] = "embedding pass exceeded 2 h (plan Sec. 10)"
            json.dump(man, open(os.path.join(OUT, "manifest.json"), "w"), indent=2); sys.exit("STOP: budget")

    # task-disjoint files: same trajectories re-split? (then the standard-split embeddings are reused)
    std = {r["id"]: [stx.step_text(s) for s in r["steps"]] for k in ("agentdrift_train", "agentdrift_val", "agentdrift_test") for r in R[k]}
    tdr = {"n_records": 0, "missing_in_standard": 0, "text_mismatch": 0, "files": {}}
    for k in ("train", "val", "test"):
        p = os.path.join(AGENTDRIFT, "data_taskdisjoint", f"{k}.jsonl")
        tdr["files"][k] = {"path": p, "sha256": se.sha256(p)}
        for r in read(p):
            tdr["n_records"] += 1
            if r["id"] not in std:
                tdr["missing_in_standard"] += 1
            elif std[r["id"]] != [stx.step_text(s) for s in r["steps"]]:
                tdr["text_mismatch"] += 1
    tdr["identical_to_standard"] = tdr["n_records"] == 12536 and tdr["missing_in_standard"] == 0 and tdr["text_mismatch"] == 0
    man["taskdisjoint"] = tdr
    print("taskdisjoint:", tdr, flush=True)
    assert tdr["identical_to_standard"], "task-disjoint records differ; embed them separately before fitting"

    teE = [r for r in R["agentdrift_test"] if r["category"] in POOL_E]
    steps = [(ti, si, st, r) for ti, r in enumerate(teE) for si, st in enumerate(r["steps"])]
    sids = np.asarray([(ti, si) for ti, si, _, _ in steps], np.int64)
    tids = np.asarray([r["id"] for r in teE], dtype=object)

    def variant_table(variants):
        """variants: {key: text list over `steps`} -> unique texts, index matrix."""
        uniq = {}
        idx = {}
        for k, tl in variants.items():
            idx[k] = np.asarray([uniq.setdefault(t, len(uniq)) for t in tl], np.int64)
        texts = [None] * len(uniq)
        for t, i in uniq.items():
            texts[i] = t
        return texts, idx

    # graded-coverage masks
    masks = json.load(open(a.masks, encoding="utf-8"))["masks"]
    var = {}
    for k, m in masks.items():
        ms = set(m)
        var[k] = [stx.step_text(st, stx.MASK_TOOL if st.get("tool") in ms else None) for _, _, st, _ in steps]
    mt, mi = variant_table(var)
    me = enc(mt)
    keys = list(mi)
    np.savez_compressed(os.path.join(OUT, "masked_r1.npz"), emb=me, ids=sids, trace_ids=tids,
                        mask_keys=np.asarray(keys, dtype=object), index=np.stack([mi[k] for k in keys]))
    man["masked_r1"] = {"mask_file_sha256": se.sha256(a.masks), "n_masks": len(keys), "n_steps_poolE": len(steps),
                        "n_unique_texts": len(mt), "n_unique_truncated": ntrunc(mt)}
    print("masked:", man["masked_r1"], flush=True)

    # padding test
    def padded(st, r, n):
        if n and r["category"] == "attacked" and st.get("label") == "injection_point":
            st = dict(st); st["obs"] = ("" if st.get("obs") is None else str(st["obs"])) + stx.filler(n)
        return stx.step_text(st)
    var = {f"pad{n}": [padded(st, r, n) for _, _, st, r in steps] for n in PADS}
    pt, pi = variant_table(var)
    pe = enc(pt)
    pk = list(pi)
    is_pad = np.array([r["category"] == "attacked" and st.get("label") == "injection_point" for _, _, st, r in steps])
    np.savez_compressed(os.path.join(OUT, "padded_r1.npz"), emb=pe, ids=sids, trace_ids=tids,
                        pad_keys=np.asarray(pk, dtype=object), index=np.stack([pi[k] for k in pk]), is_padded_step=is_pad)
    man["padded_r1"] = {"pads": PADS, "n_padded_steps": int(is_pad.sum()),
                        "n_padded_steps_truncated": {k: ntrunc([var[k][i] for i in np.where(is_pad)[0]]) for k in var}}
    print("padded:", man["padded_r1"], flush=True)

    # CPU equivalence check on a fixed random sample of amendment-3 texts
    if a.cpu_check and a.device != "cpu":
        rng = np.random.default_rng(20261008)
        pool = [(n, i) for n in allt for i in range(len(allt[n]))]
        pick = rng.choice(len(pool), size=min(a.cpu_check, len(pool)), replace=False)
        sample = [pool[j] for j in pick]
        G = np.stack([np.load(os.path.join(OUT, f"{n}.npz"), allow_pickle=True)["emb"][i] for n, i in sample])
        torch.set_num_threads(max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))))
        cm = SentenceTransformer(se.MODEL_NAME, device="cpu"); cm.max_seq_length = se.MAX_SEQ
        t0 = time.time()
        C = cm.encode([allt[n][i] for n, i in sample], batch_size=64, convert_to_numpy=True,
                      normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
        cos = np.sum(G * C, 1); d = np.abs(G - C)
        man["cpu_check"] = {"n": len(sample), "seed": 20261008, "max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()),
                            "min_cosine": float(cos.min()), "mean_cosine": float(cos.mean()),
                            "n_bitwise_identical_rows": int(np.sum(np.all(G == C, 1))), "cpu_seconds": round(time.time() - t0, 1),
                            "sample": [[n, int(i)] for n, i in sample]}
        print("cpu check:", {k: v for k, v in man["cpu_check"].items() if k != "sample"}, flush=True)
    man["total_seconds"] = round(time.time() - T0, 1)
    json.dump(man, open(os.path.join(OUT, "manifest.json"), "w"), indent=2)
    print("wrote manifest; total", man["total_seconds"], "s", flush=True)


if __name__ == "__main__":
    main()
