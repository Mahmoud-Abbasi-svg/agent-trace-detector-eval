"""Build the frozen all-mpnet-base-v2 step-embedding cache.

Implements Section 4 of SENCODER_PREREGISTRATION.md. The encoder is frozen and saw no
label and no split, so this runs once, before any split is chosen, and every rung and
seed reads the same cache.

Step text (Sec. 2 of the plan, with D6):   "<tool> | <k=v, ...> | <obs>"
The AgentDrift 'thought' field is deliberately excluded so that AgentDrift and AgentDojo
share one representation for R4.

Outputs, under revision/cache/sencoder/:
    <source>.npz          ids (trace_id, step_idx) + float32 [n_steps, 768], L2-normalised
    manifest.json         model, revision, pooling, max_seq_len, truncation counts, input SHA-256

Usage:   python sencoder_embed.py [--batch 64] [--limit N]
"""
import argparse, hashlib, json, os, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "cache", "sencoder")

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
MAX_SEQ = 384  # model default, fixed by the plan

AGENTDRIFT = os.environ.get(
    "AGENTDRIFT", r"J:\Post_Doc\Agent_id\crywolf\data\agentdrift\014a514fa998"
)
SOURCES = {
    "agentdrift_train": os.path.join(AGENTDRIFT, "data", "train.jsonl"),
    "agentdrift_val": os.path.join(AGENTDRIFT, "data", "val.jsonl"),
    "agentdrift_test": os.path.join(AGENTDRIFT, "data", "test.jsonl"),
    "agentdojo": os.path.join(ROOT, "agentdojo_traces.jsonl"),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def render_args(args):
    """Arguments as 'key=value' pairs, in the order the record stores them."""
    if not isinstance(args, dict):
        return "" if args is None else str(args)
    return ", ".join(f"{k}={v}" for k, v in args.items())


def step_text(step):
    tool = step.get("tool") or ""
    obs = step.get("obs") or ""
    return f"{tool} | {render_args(step.get('args'))} | {obs}"


def load_steps(path, limit=None):
    """-> (ids [n,2] int64 index into trace list, texts [n], trace_ids [m])."""
    trace_ids, texts, ids = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            ti = len(trace_ids)
            trace_ids.append(d["id"])
            for si, st in enumerate(d.get("steps") or []):
                ids.append((ti, si))
                texts.append(step_text(st))
            if limit and len(trace_ids) >= limit:
                break
    return np.asarray(ids, dtype=np.int64), texts, trace_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None, help="traces per source, for a smoke run")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    for name, path in SOURCES.items():
        if not os.path.exists(path):
            sys.exit(f"missing input for {name}: {path}")

    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(max(1, (os.cpu_count() or 4)))
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    model.max_seq_length = MAX_SEQ
    model.eval()
    tok = model.tokenizer

    manifest = {
        "model": MODEL_NAME,
        "revision": getattr(getattr(model, "model_card_data", None), "base_model_revision", None),
        "pooling": "mean",
        "normalised": True,
        "max_seq_length": MAX_SEQ,
        "dim": int(model.get_sentence_embedding_dimension()),
        "thought_excluded": True,
        "torch": torch.__version__,
        "built": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "sources": {},
    }

    for name, path in SOURCES.items():
        t0 = time.time()
        idx, texts, trace_ids = load_steps(path, args.limit)
        print(f"[{name}] {len(trace_ids)} traces, {len(texts)} steps", flush=True)

        # Truncation audit: tokenise without truncation and count texts over the limit.
        over = 0
        for i in range(0, len(texts), 512):
            lens = [len(x) for x in tok(texts[i : i + 512], truncation=False)["input_ids"]]
            over += sum(1 for n in lens if n > MAX_SEQ)

        emb = model.encode(
            texts,
            batch_size=args.batch,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)

        np.savez_compressed(
            os.path.join(OUT, f"{name}.npz"),
            ids=idx,
            emb=emb,
            trace_ids=np.asarray(trace_ids, dtype=object),
        )
        manifest["sources"][name] = {
            "path": path,
            "sha256": sha256(path),
            "n_traces": len(trace_ids),
            "n_steps": len(texts),
            "n_steps_truncated": over,
            "pct_truncated": round(100.0 * over / max(1, len(texts)), 2),
            "seconds": round(time.time() - t0, 1),
        }
        print(
            f"[{name}] done in {time.time()-t0:.0f}s, "
            f"{over} of {len(texts)} steps truncated at {MAX_SEQ} word pieces",
            flush=True,
        )

    with open(os.path.join(OUT, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print("wrote", os.path.join(OUT, "manifest.json"), flush=True)


if __name__ == "__main__":
    main()
