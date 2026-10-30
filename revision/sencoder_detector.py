"""SE detector, a reimplementation of the DriftNet architecture, per SENCODER_PREREGISTRATION.md Sec. 12 (amendment 3).

No tuning. Every constant below is fixed by Section 12. Values not stated there are PyTorch defaults.
Step input x_t = [e_t || w_t] in R^772: frozen all-mpnet-base-v2 embedding (cache) and the four raw world features.
Trunk: Linear(772->256); h <- Dropout(h + sinusoidal PE) (max 64); TransformerEncoder(3 layers, 4 heads, FF 512,
GELU, dropout 0.3, post-norm) with key-padding mask.
Head 1: masked mean -> MLP 256->128 GELU dropout ->1.  Head 2: per step MLP 256->128 GELU dropout ->4.
Loss: BCE(w+ = n_neg/n_pos) + 1.0 * class-weighted 4-way CE over valid labelled steps (alpha = inverse frequency,
scaled to sum 4). AgentDojo fits: trajectory loss only (D2).
AdamW(lr peak 1e-3, wd 0.05), triangular schedule over S = 40 x batches (warm-up 0.1 S, then linear decay to 0),
batch 32, grad-norm clip 1.0, <= 40 epochs, early stop on validation trajectory F1 at 0.5, patience 8, keep best F1.
CPU, torch.use_deterministic_algorithms(True).
Thresholds: t_r80 = 20th percentile of in-sample training-positive scores, t05 = 0.5, t_val80 for reference.
"""
import math, time
import numpy as np
import torch
import torch.nn as nn

MAXLEN = 64; D_EMB = 768; D_W = 4; D_MODEL = 256; N_HEAD = 4; D_FF = 512; N_LAYERS = 3; DROPOUT = 0.3
LR_PEAK = 1e-3; WD = 0.05; BATCH = 32; MAX_EPOCH = 40; PATIENCE = 8; LAMBDA = 1.0; WARMUP = 0.1; CLIP = 1.0
STEP_CLASSES = ("benign", "injection_point", "hijacked", "failed_injection")
torch.use_deterministic_algorithms(True)


def sinusoidal(maxlen, d):
    pe = torch.zeros(maxlen, d)
    pos = torch.arange(maxlen, dtype=torch.float32)[:, None]
    div = torch.exp(torch.arange(0, d, 2, dtype=torch.float32) * (-math.log(10000.0) / d))
    pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
    return pe


class SEDet(nn.Module):
    def __init__(self, d_in=D_EMB + D_W):
        super().__init__()
        self.proj = nn.Linear(d_in, D_MODEL)
        self.register_buffer("pe", sinusoidal(MAXLEN, D_MODEL), persistent=False)
        self.drop = nn.Dropout(DROPOUT)
        layer = nn.TransformerEncoderLayer(D_MODEL, N_HEAD, D_FF, dropout=DROPOUT, activation="gelu", batch_first=True)
        self.enc = nn.TransformerEncoder(layer, N_LAYERS, enable_nested_tensor=False)
        self.head_traj = nn.Sequential(nn.Linear(D_MODEL, D_MODEL // 2), nn.GELU(), nn.Dropout(DROPOUT), nn.Linear(D_MODEL // 2, 1))
        self.head_step = nn.Sequential(nn.Linear(D_MODEL, D_MODEL // 2), nn.GELU(), nn.Dropout(DROPOUT), nn.Linear(D_MODEL // 2, 4))

    def forward(self, x, pad):
        L = x.shape[1]
        h = self.drop(self.proj(x) + self.pe[:L][None])
        h = self.enc(h, src_key_padding_mask=pad)
        keep = (~pad).unsqueeze(-1).to(h.dtype)
        p = (h * keep).sum(1) / keep.sum(1).clamp(min=1.0)
        return self.head_traj(p).squeeze(1), self.head_step(h)


def n_params(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


class Encoded:
    """Trajectories as lists of global step rows (into EMB / WF tables) plus labels."""

    def __init__(self, recs, rows_of):
        self.rows = []; self.lab = []; self.n_truncated = 0
        self.y = np.array([int(r["category"] == "attacked") for r in recs], np.float32)
        self.has_lab = np.zeros(len(recs), bool)
        for i, r in enumerate(recs):
            g = np.asarray(rows_of(r), np.int64)
            st = r["steps"]
            lab = np.array([STEP_CLASSES.index(s["label"]) for s in st], np.int64) if st and all("label" in s for s in st) else None
            if len(g) > MAXLEN:
                self.n_truncated += 1
                g = g[-MAXLEN:]; lab = None if lab is None else lab[-MAXLEN:]
            assert len(g) > 0
            self.rows.append(g); self.lab.append(lab); self.has_lab[i] = lab is not None
        self.n = len(recs)


def batch(enc, idx, EMB, WF, zero_w=False):
    L = max(len(enc.rows[i]) for i in idx)
    x = np.zeros((len(idx), L, D_EMB + D_W), np.float32)
    pad = np.ones((len(idx), L), bool)
    lab = np.full((len(idx), L), -100, np.int64)
    for j, i in enumerate(idx):
        g = enc.rows[i]; k = len(g)
        x[j, :k, :D_EMB] = EMB[g]
        if not zero_w:
            x[j, :k, D_EMB:] = WF[g]
        pad[j, :k] = False
        if enc.lab[i] is not None:
            lab[j, :k] = enc.lab[i]
    return torch.from_numpy(x), torch.from_numpy(pad), torch.from_numpy(lab)


@torch.no_grad()
def logits(model, enc, EMB, WF, zero_w=False, bs=256):
    model.eval()
    out = np.empty(enc.n, np.float64)
    order = np.argsort([len(r) for r in enc.rows], kind="stable")
    for i in range(0, enc.n, bs):
        idx = order[i:i + bs]
        x, pad, _ = batch(enc, idx, EMB, WF, zero_w)
        out[idx] = model(x, pad)[0].double().numpy()
    return out


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def f1_at_half(y, s):
    p = s >= 0.5
    tp = float(np.sum(p & (y == 1))); fp = float(np.sum(p & (y == 0))); fn = float(np.sum(~p & (y == 1)))
    return 2 * tp / (2 * tp + fp + fn) if tp > 0 else 0.0


def class_weights(enc):
    cnt = np.zeros(4)
    for lab in enc.lab:
        if lab is not None:
            cnt += np.bincount(lab, minlength=4)
    inv = np.where(cnt > 0, 1.0 / np.maximum(cnt, 1), 0.0)
    return (4.0 * inv / inv.sum()) if inv.sum() > 0 else np.ones(4), cnt


def train_se(enc_tr, enc_va, EMB, WF, seed, zero_w=False):
    torch.manual_seed(seed); np.random.seed(seed)
    model = SEDet()
    opt = torch.optim.AdamW(model.parameters(), lr=LR_PEAK, weight_decay=WD)
    n = enc_tr.n; nb = math.ceil(n / BATCH); S = MAX_EPOCH * nb; Sw = WARMUP * S
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(s / Sw, (S - s) / (S - Sw)) if s > 0 else 0.0)
    npos, nneg = float(enc_tr.y.sum()), float((enc_tr.y == 0).sum())
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(nneg / max(npos, 1.0)))
    step_on = bool(enc_tr.has_lab.any())
    alpha, cnt = class_weights(enc_tr)
    ce = nn.CrossEntropyLoss(weight=torch.tensor(alpha, dtype=torch.float32), ignore_index=-100)
    ytr = torch.from_numpy(enc_tr.y)
    best_f1, best_state, bad, best_ep = -1.0, None, 0, -1
    hist = []; t0 = time.time()
    for ep in range(MAX_EPOCH):
        model.train(); perm = torch.randperm(n).numpy(); tl = 0.0
        for i in range(0, n, BATCH):
            b = perm[i:i + BATCH]
            x, pad, lab = batch(enc_tr, b, EMB, WF, zero_w)
            opt.zero_grad()
            zt, zs = model(x, pad)
            loss = bce(zt, ytr[b])
            if step_on and (lab != -100).any():
                loss = loss + LAMBDA * ce(zs.reshape(-1, 4), lab.reshape(-1))
            if not torch.isfinite(loss):
                raise FloatingPointError("nan_loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step(); sched.step()
            tl += float(loss) * len(b)
        sv = sigmoid(logits(model, enc_va, EMB, WF, zero_w))
        f1 = f1_at_half(enc_va.y, sv)
        hist.append({"epoch": ep + 1, "train_loss": tl / n, "val_f1": f1, "lr_end": float(sched.get_last_lr()[0])})
        if f1 > best_f1:
            best_f1, best_state, bad, best_ep = f1, {k: v.clone() for k, v in model.state_dict().items()}, 0, ep + 1
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state); model.eval()
    z_tr = logits(model, enc_tr, EMB, WF, zero_w); s_tr = sigmoid(z_tr)
    thr = float(np.quantile(s_tr[enc_tr.y == 1], 0.20)) if (enc_tr.y == 1).any() else 0.5
    s_va = sigmoid(logits(model, enc_va, EMB, WF, zero_w))
    thr_val = float(np.quantile(s_va[enc_va.y == 1], 0.20)) if (enc_va.y == 1).any() else 0.5
    diag = {"epochs_run": len(hist), "best_epoch": best_ep, "best_val_f1": float(best_f1), "history": hist,
            "step_loss_active": step_on, "class_weights": alpha.tolist(), "step_class_counts": cnt.tolist(),
            "pos_weight": nneg / max(npos, 1.0), "n_params": n_params(model), "n_train": n, "n_train_pos": int(npos),
            "n_val": enc_va.n, "n_val_pos": int(enc_va.y.sum()), "train_truncated": enc_tr.n_truncated,
            "fit_seconds": time.time() - t0, "thr_r80": thr, "thr_val80": thr_val, "zero_w": zero_w,
            "train_logit": z_tr.astype(np.float32), "train_y": enc_tr.y.astype(np.int8)}
    return model, thr, thr_val, diag
