# Obsolete cache, do not use for any result

Built 8 Oct 2026 by a local CPU run of `sencoder_embed.py` **in the version 1 step-text format**
(`"<tool> | <k=v> | <obs>"`), under amendment 1 of `SENCODER_PREREGISTRATION.md`.

Amendment 3 replaced the step text with the DriftNet paper's format
(`"TOOL: … | ARGS: {json} | OBS: …"`), so these embeddings are built from different inputs and
**cannot** serve as the GPU versus CPU reference. The cluster CPU sample under amendment 3 is the
only valid reference.

Kept because `revision/PLAN_MANIFEST.txt` records its existence at the amendment 3 hashing time.
Delete only if that note is amended too.

## The one thing worth carrying forward

Truncation at the 384 word-piece limit is strongly asymmetric across the two benchmarks.

| Source | Traces | Steps | Steps truncated | Share |
|---|---:|---:|---:|---:|
| agentdrift_train | 8,775 | 49,750 | 3 | 0.006% |
| agentdrift_val | 1,880 | 10,722 | 0 | 0% |
| agentdrift_test | 1,881 | 10,552 | 0 | 0% |
| **agentdojo** | **2,686** | **14,079** | **912** | **6.48%** |

These counts are for the version 1 text. The amendment 3 text adds field prefixes and renders
arguments as JSON, so it is longer and truncation will be **worse** than shown here, not better.
