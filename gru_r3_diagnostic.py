"""Narrow diagnostic: WHY is the GRU worse than C under R3? (does not modify gru_detector.py,
GRU_PREREGISTRATION.md, GRU_RESULTS.md, or the primary A/B/C ablation).

Three GRU input variants, R3 only (5 held-out domains), same seeds/protocol as the primary run:
  GRU-A tool-sequence only:   embedding(<PAD>/<UNK>+tools) -> GRU(64) -> Linear(64,1)
                               (identical architecture to the primary model's tool_only=True path)
  GRU-B behavioural only:     NO embedding/GRU (there is no sequence to feed); the 10 standardised
                               behavioural features -> Linear(10,1). This is the natural
                               "same architecture where applicable" reading when there is no
                               sequence component at all.
  GRU-C combined (primary):   embedding -> GRU(64) concat 10 behavioural feats -> Linear(74,1)
                               (refit here, same recipe, as an internal consistency check against
                               GRU_RESULTS.md -- NOT a replacement for that file).

Same 3 training seeds {0,1,2}, same stratified 10%-of-train validation construction, same
early-stopping-on-val-AUPRC, same training-derived threshold, same evalkit bootstrap protocol
and R3 fold construction/seeds as gru_ladder.py. No tuning. R1/R2/R4 not rerun.
Run: AGENTDRIFT=<clone> python gru_r3_diagnostic.py
"""
import os, json, numpy as np, torch, torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from featurise import load
import evalkit as ek
from gru_detector import (MAXLEN, EMB, HID, BEH_DIM, LR, WD, BATCH, MAX_EPOCH, PATIENCE,
                          PAD, UNK, build_tool_vocab, encode_seq, beh_feats, ylab)
from gru_ladder import stratified_val, fitC, coverage, TRAIN_SEEDS

MP = os.environ["AGENTDRIFT"]; HERE = os.path.dirname(os.path.abspath(__file__))
B13 = 1000; PREVS = [None, 0.01, 0.02, 0.05, 0.10]
torch.use_deterministic_algorithms(True)


class Variant(nn.Module):
    def __init__(self, vocab_size, mode):
        super().__init__()
        self.mode = mode  # "tool" | "beh" | "combined"
        if mode in ("tool", "combined"):
            self.emb = nn.Embedding(vocab_size + 2, EMB, padding_idx=PAD)
            self.gru = nn.GRU(EMB, HID, num_layers=1, batch_first=True)
        in_dim = {"tool": HID, "beh": BEH_DIM, "combined": HID + BEH_DIM}[mode]
        self.head = nn.Linear(in_dim, 1)

    def forward(self, ids, lens, beh):
        if self.mode == "beh":
            return self.head(beh).squeeze(1)
        e = self.emb(ids)
        packed = pack_padded_sequence(e, lens.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed); h = h[-1]
        z = h if self.mode == "tool" else torch.cat([h, beh], 1)
        return self.head(z).squeeze(1)


def _tensors(recs, vocab, scaler):
    seqs, lens = zip(*[encode_seq(r, vocab) for r in recs])
    ids = torch.tensor(seqs, dtype=torch.long); ln = torch.tensor(lens, dtype=torch.long)
    beh = torch.tensor(scaler.transform(beh_feats(recs, vocab)), dtype=torch.float32)
    return ids, ln, beh


def train_variant(train, val, seed, mode):
    torch.manual_seed(seed); np.random.seed(seed)
    vocab = build_tool_vocab(train)
    scaler = StandardScaler().fit(beh_feats(train, vocab))
    Xtr = _tensors(train, vocab, scaler); ytr = torch.tensor(ylab(train))
    Xva = _tensors(val, vocab, scaler); yva = ylab(val)
    npos, nneg = float(ytr.sum()), float((ytr == 0).sum())
    posw = torch.tensor(nneg / max(npos, 1.0))
    model = Variant(len(vocab), mode)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    lossf = nn.BCEWithLogitsLoss(pos_weight=posw)
    n = len(train); best_ap, best_state, bad = -1, None, 0
    for ep in range(MAX_EPOCH):
        model.train(); perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad()
            out = model(Xtr[0][b], Xtr[1][b], Xtr[2][b])
            loss = lossf(out, ytr[b])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nan_loss mode={mode} seed={seed}")
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval()
        with torch.no_grad():
            sv = torch.sigmoid(model(*Xva)).numpy()
        ap = average_precision_score(yva, sv) if yva.sum() else 0.0
        if ap > best_ap:
            best_ap, best_state, bad = ap, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= PATIENCE: break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        s_tr = torch.sigmoid(model(*Xtr)).numpy()
    thr = float(np.quantile(s_tr[ytr.numpy() == 1], 0.20)) if (ytr == 1).any() else 0.5
    return model, vocab, scaler, thr


def score_variant(model, vocab, scaler, thr, recs):
    X = _tensors(recs, vocab, scaler)
    model.eval()
    with torch.no_grad():
        s = torch.sigmoid(model(*X)).numpy()
    return {"score": s, "pred": {"t_r80": s >= thr, "t0.5": s >= 0.5}}


def variant_reps(train, test, mode):
    reps = []
    for sd in TRAIN_SEEDS:
        tr, va = stratified_val(train, sd)
        m, voc, scl, thr = train_variant(tr, va, sd, mode)
        reps.append(score_variant(m, voc, scl, thr, test))
    return reps


def main():
    pool = load(os.path.join(MP, "data/train.jsonl")) + load(os.path.join(MP, "data/test.jsonl"))
    domains = ["medical", "banking", "web", "email", "coding"]
    R = {"meta": {"train_seeds": TRAIN_SEEDS, "note": "R3 only; refits its own C for a same-run paired comparison"}}
    for ri, d in enumerate(domains):
        train = [r for r in pool if r["agent"] != d]; test = [r for r in pool if r["agent"] == d]
        test = [r for r in test if r["category"] in ("benign", "hard_negative", "attacked")]
        y = ylab(test)
        methods = {"GRU-A_tool": variant_reps(train, test, "tool"),
                   "GRU-B_beh": variant_reps(train, test, "beh"),
                   "GRU-C_combined": variant_reps(train, test, "combined")}
        methods["C_linear"] = [fitC(train)(test)]
        cov = coverage(train, test)
        node = {"cov": cov, "n_neg": int((y == 0).sum()), "n_pos": int(y.sum()),
               "native_prev": round(float(y.mean()), 4), "by_prev": {}}
        for p in PREVS:
            npos = None if p is None else int(round(p / (1 - p) * (y == 0).sum()))
            bt = ek.bootstrap(y, methods, n_pos=npos, B=B13, seed=5200 + ri)
            key = "native" if p is None else f"p{int(p*100):02d}"
            rows = {}
            for m in methods:
                pt = ek.point(y, methods[m]) if p is None else None
                rows[m] = {k: [round(pt[k] if p is None else ek.ci(v)[0], 4)] + [round(x, 4) for x in ek.ci(v)[1:]]
                          for k, v in bt[m].items()}
            paired = {f"{a}-minus-{b}": {k: ek.paired_delta(bt, a, b, k) for k in ["AUROC", "AUPRC"]}
                      for a, b in [("GRU-A_tool", "GRU-B_beh"), ("GRU-C_combined", "GRU-A_tool"),
                                   ("GRU-C_combined", "GRU-B_beh"), ("GRU-C_combined", "C_linear"),
                                   ("GRU-A_tool", "C_linear"), ("GRU-B_beh", "C_linear")]}
            node["by_prev"][key] = {"n_pos": int(y.sum()) if npos is None else npos, "n_neg": int((y == 0).sum()),
                                    "metrics": rows, "paired": paired}
        R[d] = node
        print(f"done {d}: A={node['by_prev']['native']['metrics']['GRU-A_tool']['AUROC'][0]:.3f} "
              f"B={node['by_prev']['native']['metrics']['GRU-B_beh']['AUROC'][0]:.3f} "
              f"C_gru={node['by_prev']['native']['metrics']['GRU-C_combined']['AUROC'][0]:.3f} "
              f"C_lin={node['by_prev']['native']['metrics']['C_linear']['AUROC'][0]:.3f}", flush=True)
    json.dump(R, open(os.path.join(HERE, "gru_r3_diagnostic_results.json"), "w"), indent=1, default=float)
    print("wrote gru_r3_diagnostic_results.json")


if __name__ == "__main__":
    main()
