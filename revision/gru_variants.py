"""Token-dropout GRU variant for the revision (REVISION_ANALYSIS_PLAN.md Sec. 2, "GRU-drop").

`train_gru_drop(train, val, seed, p)` is a line-by-line copy of `gru_detector.train_gru` with ONE
addition: when p > 0, during training only, every non-padding token of every training mini-batch is
replaced by the <UNK> index (1) independently with probability p. The Bernoulli draws come from a
separate `torch.Generator` seeded with (seed + 10007), so the global torch random stream (weight
initialisation, `torch.randperm` batch order) is exactly the one `train_gru` uses. Validation inputs
(early stopping), the in-sample training scores used for the recall-0.80 threshold, and all test
inputs are never dropped. With p = 0 no draw is made and the function is numerically identical to
`gru_detector.train_gru` (checked by `build_gru_cache.py verify`: bit-identical R1 seed-0 scores).

The architecture, constants, vocabulary, tensors, loss, optimiser, early stopping and threshold are
imported from `gru_detector.py` (not modified). The function additionally returns diagnostics
(validation-derived threshold, epochs, <UNK> statistics) without touching any random stream.

`decompose(model, X)` splits the head's logit into the contribution of the GRU hidden state
(head weights over the 64 hidden units . final hidden state; the "tool pathway"), the contribution
of the 10 standardised behavioural features, and the bias.
"""
import os, sys
import numpy as np, torch, torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gru_detector import (GRUDet, _tensors, build_tool_vocab, beh_feats, ylab,  # noqa: E402
                          LR, WD, BATCH, MAX_EPOCH, PATIENCE, PAD, UNK, HID)

DROP_SEED_OFFSET = 10007
torch.use_deterministic_algorithms(True)


def train_gru_drop(train, val, seed, p=0.0, tool_only=False):
    """Copy of gru_detector.train_gru + training-only token dropout with probability p.
    Returns (model, vocab, scaler, thr, aux); (model, vocab, scaler, thr) are what train_gru returns."""
    torch.manual_seed(seed); np.random.seed(seed)
    vocab = build_tool_vocab(train)
    scaler = StandardScaler().fit(beh_feats(train, vocab))
    Xtr = _tensors(train, vocab, scaler); ytr = torch.tensor(ylab(train))
    Xva = _tensors(val, vocab, scaler); yva = ylab(val)
    npos, nneg = float(ytr.sum()), float((ytr == 0).sum())
    posw = torch.tensor(nneg / max(npos, 1.0))
    model = GRUDet(len(vocab), tool_only)
    unk_init_norm = float(model.emb.weight.detach()[UNK].norm())          # read-only, no RNG use
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    lossf = nn.BCEWithLogitsLoss(pos_weight=posw)
    # --- the one addition: a separate generator for the dropout mask (main stream untouched) ---
    gen = None
    if p > 0:
        gen = torch.Generator(); gen.manual_seed(seed + DROP_SEED_OFFSET)
    n_tok_seen, n_tok_dropped = 0, 0
    # ------------------------------------------------------------------------------------------
    n = len(train); best_ap, best_state, bad = -1, None, 0
    ep_run, best_ep, val_curve = 0, -1, []
    for ep in range(MAX_EPOCH):
        model.train(); perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad()
            ids = Xtr[0][b]
            if p > 0:                                                    # training batches only
                real = ids != PAD
                drop = real & (torch.rand(ids.shape, generator=gen) < p)
                ids = ids.masked_fill(drop, UNK)
                n_tok_seen += int(real.sum()); n_tok_dropped += int(drop.sum())
            out = model(ids, Xtr[1][b], Xtr[2][b])
            loss = lossf(out, ytr[b])
            if not torch.isfinite(loss):
                raise FloatingPointError("nan_loss")
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval()
        with torch.no_grad():
            sv = torch.sigmoid(model(*Xva)).numpy()
        ap = average_precision_score(yva, sv) if yva.sum() else 0.0
        ep_run = ep + 1; val_curve.append(float(ap))
        if ap > best_ap:
            best_ap, best_state, bad = ap, {k: v.clone() for k, v in model.state_dict().items()}, 0
            best_ep = ep + 1
        else:
            bad += 1
            if bad >= PATIENCE: break
    model.load_state_dict(best_state)
    # training-derived threshold (recall 0.80 in-sample on train positives), exactly as train_gru
    model.eval()
    with torch.no_grad():
        s_tr = torch.sigmoid(model(*Xtr)).numpy()
    thr = float(np.quantile(s_tr[ytr.numpy() == 1], 0.20)) if (ytr == 1).any() else 0.5
    # validation-derived threshold (20th percentile of validation-positive scores), no dropout
    with torch.no_grad():
        s_va = torch.sigmoid(model(*Xva)).numpy()
    thr_val = float(np.quantile(s_va[yva == 1], 0.20)) if (yva == 1).any() else 0.5
    ids_tr = Xtr[0]; real_tr = ids_tr != PAD
    unk_rows = ((ids_tr == UNK) & real_tr).any(1)
    aux = {"thr_val80": thr_val, "p": float(p), "epochs_run": ep_run, "best_epoch": best_ep,
           "best_val_ap": float(best_ap), "val_ap_curve": val_curve,
           "n_fit": len(train), "n_fit_pos": int(npos), "n_val": len(val), "n_val_pos": int(yva.sum()),
           "vocab_size_model": len(vocab),
           "unk_train_steps": int(((ids_tr == UNK) & real_tr).sum()),       # before dropout
           "unk_train_traces": int(unk_rows.sum()),
           "unk_train_traces_pos": int((unk_rows & (ytr == 1)).sum()),
           "train_tokens": int(real_tr.sum()),
           "drop_tokens_seen": n_tok_seen, "drop_tokens_dropped": n_tok_dropped,
           "drop_rate_realised": (n_tok_dropped / n_tok_seen) if n_tok_seen else 0.0,
           "unk_init_norm": unk_init_norm,
           "s_train": s_tr, "y_train": ytr.numpy().astype(np.int8), "Xtr": Xtr}
    return model, vocab, scaler, thr, aux


def decompose(model, X):
    """Head-logit decomposition: logit = tool (w_h . h) + beh (w_b . beh_std) + bias.
    Returns float32 arrays tool, beh, logit and the scalar bias."""
    ids, lens, beh = X
    model.eval()
    with torch.no_grad():
        e = model.emb(ids)
        packed = pack_padded_sequence(e, lens.cpu(), batch_first=True, enforce_sorted=False)
        _, h = model.gru(packed); h = h[-1]
        W = model.head.weight[0]; bias = float(model.head.bias[0])
        tool = (h @ W[:HID]).numpy()
        behc = (beh @ W[HID:]).numpy() if not model.tool_only else np.zeros(len(ids), np.float32)
        logit = model(ids, lens, beh).numpy()
    return tool.astype(np.float32), behc.astype(np.float32), bias, logit.astype(np.float32)


def embedding_norms(model, vocab):
    W = model.emb.weight.detach()
    inv = sorted(vocab.items(), key=lambda kv: kv[1])
    names = [t for t, _ in inv]; idx = [i for _, i in inv]
    return float(W[UNK].norm()), W[idx].norm(dim=1).numpy().astype(np.float32), names
