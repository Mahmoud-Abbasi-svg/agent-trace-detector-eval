"""GPU against CPU check of the embedding cache (SENCODER_PREREGISTRATION.md amendment 6, D11).

The main cache was built on CPU (revision/sencoder_embed_gpu.py --device cpu). This script draws a fixed random
sample of 1,000 amendment 3 step texts (numpy default_rng(20261008) over the pooled steps of the five sources, in
source order), re-embeds them on one GPU with the same model, revision, pooling, normalisation and maximum length,
and compares with the CPU rows. Writes gpu_check.json.
Usage: SENCODER_EMB=<cache dir> python sencoder_gpu_check.py [--n 1000]
"""
import argparse, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import sencoder_embed as se  # noqa: E402
import sencoder_text as stx  # noqa: E402

EMB = os.environ["SENCODER_EMB"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=1000); a = ap.parse_args()
    srcs = {"agentdrift_train": os.path.join(se.AGENTDRIFT, "data", "train.jsonl"),
            "agentdrift_val": os.path.join(se.AGENTDRIFT, "data", "val.jsonl"),
            "agentdrift_test": os.path.join(se.AGENTDRIFT, "data", "test.jsonl"),
            "agentdojo": os.path.join(ROOT, "agentdojo_traces.jsonl"),
            "agentdojo_failed": os.path.join(ROOT, "agentdojo_failed.jsonl")}
    texts = {}
    for n, p in srcs.items():
        recs = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
        texts[n] = [stx.step_text(s) for r in recs for s in r["steps"]]
    pool = [(n, i) for n in texts for i in range(len(texts[n]))]
    rng = np.random.default_rng(20261008)
    pick = rng.choice(len(pool), size=a.n, replace=False)
    sample = [pool[j] for j in pick]
    need = sorted({n for n, _ in sample})
    E = {}
    for n in need:
        with np.load(os.path.join(EMB, f"{n}.npz"), allow_pickle=True) as z:   # self-produced cache
            E[n] = z["emb"]
    C = np.stack([E[n][i] for n, i in sample]); del E
    import torch, transformers, sentence_transformers
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(se.MODEL_NAME, device="cuda"); m.max_seq_length = se.MAX_SEQ
    t0 = time.time()
    G = m.encode([texts[n][i] for n, i in sample], batch_size=128, convert_to_numpy=True, normalize_embeddings=True,
                 show_progress_bar=False).astype(np.float32)
    cos = np.sum(G * C, 1); d = np.abs(G - C)
    out = {"n": a.n, "seed": 20261008, "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "torch_cuda": torch.version.cuda, "transformers": transformers.__version__,
           "sentence_transformers": sentence_transformers.__version__, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
           "max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()), "min_cosine": float(cos.min()),
           "mean_cosine": float(cos.mean()), "n_bitwise_identical_rows": int(np.sum(np.all(G == C, 1))),
           "gpu_seconds": round(time.time() - t0, 2),
           "cpu_manifest_torch": json.load(open(os.path.join(EMB, "manifest.json"))).get("torch"),
           "by_source": {n: int(sum(1 for s, _ in sample if s == n)) for n in texts}}
    json.dump(out, open("gpu_check.json", "w"), indent=1)
    print(out)


if __name__ == "__main__":
    main()
