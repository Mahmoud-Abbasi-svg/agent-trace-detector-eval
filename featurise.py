"""Trace featureriser for the agent-trace IDS study.

Each trajectory -> (a) an ordered list of tool tokens (for sequence models / n-gram), and
(b) a fixed-length feature vector (for classical detectors: one-class SVM, isolation forest,
logistic regression). Trajectory-level label: attacked (category=="attacked") = positive class
(the thing an IDS must catch); benign / hard_negative / failed_attack = negative for the
detection task, but tracked separately so we can report false positives on hard negatives.

Read-only over the cloned AgentDrift; no attack generation.
"""
import json, os, collections

def load(p): return [json.loads(l) for l in open(p, encoding="utf-8")]

def tool_seq(rec): return [s.get("tool", "<none>") for s in rec["steps"]]

def label_pos(rec):
    """1 if this trajectory realises an injected goal (the IDS target), else 0."""
    return int(rec["category"] == "attacked")

def group(rec):
    return rec["category"]  # benign / attacked / failed_attack / hard_negative

def domain(rec):
    return rec["agent"]

def build_vocab(recs):
    toks = collections.Counter()
    for r in recs:
        for t in tool_seq(r): toks[t] += 1
    vocab = {t: i for i, t in enumerate(sorted(toks))}
    return vocab

def vectorise(rec, vocab):
    """Fixed-length feature vector, vocabulary-agnostic parts + bag-of-tools counts.
    Vocabulary-agnostic features make cross-benchmark transfer possible even when tool names differ."""
    seq = tool_seq(rec)
    n = len(seq)
    args_sizes = [len(s.get("args", {}) or {}) for s in rec["steps"]]
    obs_lens = [len(str(s.get("obs", ""))) for s in rec["steps"]]
    uniq = len(set(seq))
    # transition novelty: fraction of consecutive tool pairs that are repeats/backtracks
    repeats = sum(1 for i in range(1, n) if seq[i] == seq[i-1])
    write_tools = sum(1 for t in seq if any(k in t.lower() for k in
                      ("send", "delete", "update", "create", "transfer", "add", "download",
                       "post", "write", "cancel", "apply", "convert", "share", "grant")))
    agnostic = [
        n,                              # length
        uniq,                           # distinct tools
        uniq / n if n else 0,           # diversity
        repeats,                        # immediate repeats
        write_tools,                    # count of state-changing/egress tools
        write_tools / n if n else 0,    # fraction state-changing
        max(args_sizes) if args_sizes else 0,
        sum(args_sizes) / n if n else 0,
        max(obs_lens) if obs_lens else 0,
        sum(obs_lens) / n if n else 0,
    ]
    bag = [0.0] * len(vocab)
    for t in seq:
        if t in vocab: bag[vocab[t]] += 1.0
    return agnostic, bag

AGNOSTIC_NAMES = ["len", "uniq", "diversity", "repeats", "write_n", "write_frac",
                  "max_args", "mean_args", "max_obs", "mean_obs"]

if __name__ == "__main__":
    MP = os.environ.get("AGENTDRIFT")
    tr = load(os.path.join(MP, "data", "train.jsonl"))
    v = build_vocab(tr)
    a, b = vectorise(tr[0], v)
    print("vocab size:", len(v))
    print("agnostic features:", dict(zip(AGNOSTIC_NAMES, a)))
    print("bag length:", len(b), "nonzero:", sum(x > 0 for x in b))
