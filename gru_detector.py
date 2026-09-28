"""GRU detector exactly per GRU_PREREGISTRATION.md. No tuning; all constants fixed here.
tool-name embedding(32) + <PAD>=0/<UNK>=1, 1-layer GRU(hidden 64), final hidden state concat
10 behavioural features (standardised) -> Linear(74->1) -> sigmoid. Weighted BCE, Adam 1e-3 wd 1e-4,
batch 64, <=30 epochs, early stop on val AUPRC patience 5. CPU, deterministic. Tool-only variant optional.
"""
import numpy as np, torch, torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from featurise import vectorise

MAXLEN = 16; EMB = 32; HID = 64; BEH_DIM = 10
LR = 1e-3; WD = 1e-4; BATCH = 64; MAX_EPOCH = 30; PATIENCE = 5
PAD, UNK = 0, 1
torch.use_deterministic_algorithms(True)


def build_tool_vocab(train):
    toks = sorted({s["tool"] for r in train if r["category"] == "benign" for s in r["steps"]})
    return {t: i + 2 for i, t in enumerate(toks)}  # 0=PAD,1=UNK


def encode_seq(rec, vocab):
    ids = [vocab.get(s["tool"], UNK) for s in rec["steps"]][:MAXLEN]  # truncate to last 16 if longer
    if len(rec["steps"]) > MAXLEN:
        ids = [vocab.get(s["tool"], UNK) for s in rec["steps"]][-MAXLEN:]
    L = len(ids); ids = ids + [PAD] * (MAXLEN - L)
    return ids, max(L, 1)


def beh_feats(recs, vocab):
    # behavioural = first 10 of vectorise (vocabulary-independent); vocab arg unused for these
    return np.array([vectorise(r, {})[0] for r in recs], float)


def ylab(recs): return np.array([int(r["category"] == "attacked") for r in recs], float)


class GRUDet(nn.Module):
    def __init__(self, vocab_size, tool_only=False):
        super().__init__()
        self.tool_only = tool_only
        self.emb = nn.Embedding(vocab_size + 2, EMB, padding_idx=PAD)
        self.gru = nn.GRU(EMB, HID, num_layers=1, batch_first=True)
        self.head = nn.Linear(HID if tool_only else HID + BEH_DIM, 1)

    def forward(self, ids, lens, beh):
        e = self.emb(ids)
        packed = pack_padded_sequence(e, lens.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed); h = h[-1]                      # final hidden state (B,HID)
        z = h if self.tool_only else torch.cat([h, beh], 1)
        return self.head(z).squeeze(1)


def _tensors(recs, vocab, scaler):
    seqs, lens = zip(*[encode_seq(r, vocab) for r in recs])
    ids = torch.tensor(seqs, dtype=torch.long); ln = torch.tensor(lens, dtype=torch.long)
    beh = torch.tensor(scaler.transform(beh_feats(recs, vocab)), dtype=torch.float32)
    return ids, ln, beh


def train_gru(train, val, seed, tool_only=False):
    torch.manual_seed(seed); np.random.seed(seed)
    vocab = build_tool_vocab(train)
    scaler = StandardScaler().fit(beh_feats(train, vocab))
    Xtr = _tensors(train, vocab, scaler); ytr = torch.tensor(ylab(train))
    Xva = _tensors(val, vocab, scaler); yva = ylab(val)
    npos, nneg = float(ytr.sum()), float((ytr == 0).sum())
    posw = torch.tensor(nneg / max(npos, 1.0))
    model = GRUDet(len(vocab), tool_only)
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
                raise FloatingPointError("nan_loss")
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
    model.load_state_dict(best_state)
    # training-derived threshold (recall 0.80 in-sample on train positives)
    model.eval()
    with torch.no_grad():
        s_tr = torch.sigmoid(model(*Xtr)).numpy()
    thr = float(np.quantile(s_tr[ytr.numpy() == 1], 0.20)) if (ytr == 1).any() else 0.5
    return model, vocab, scaler, thr


def score_gru(model, vocab, scaler, thr, recs):
    model.eval()
    X = _tensors(recs, vocab, scaler)
    with torch.no_grad():
        s = torch.sigmoid(model(*X)).numpy()
    return {"score": s, "pred": {"t_r80": s >= thr, "t0.5": s >= 0.5}}
