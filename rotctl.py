#!/usr/bin/env python3
"""Rotation control: is sign+top's advantage a coordinate-basis effect?

Conjugate the compressor by fixed random orthogonals:
    u_hat = Q^T C(Q u R^T) R
Low rank must be INVARIANT (singular values unchanged) -- that is the harness
self-test. Sign+top is not rotation-invariant if the coordinate story is right.

All arms norm-matched: u_hat rescaled to ||u|| globally after compression.
"""
import json, math, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, '/mnt/user-data/uploads')
from signtop_real import signtop_budget

torch.set_num_threads(os.cpu_count())

# ---- corpus (bind) ----------------------------------------------------
ids = json.load(open('/home/claude/work/bundle/corpora/bind.json'))
uniq = sorted(set(ids)); remap = {c: i for i, c in enumerate(uniq)}
ids = [remap[c] for c in ids]
VOCAB = len(uniq)
n = int(0.9 * len(ids))
train_t = torch.tensor(ids[:n]); val_t = torch.tensor(ids[n:])

D = 256; N_HEADS = 4; N_STU = 6; BATCH = 8; SEQ = 64; LR = 3e-4
STEPS = 140

class Attn(nn.Module):
    def __init__(s):
        super().__init__(); dh = D // N_HEADS
        s.WQ = nn.Linear(D, D, bias=False); s.WK = nn.Linear(D, D, bias=False)
        s.WV = nn.Linear(D, D, bias=False); s.op = nn.Linear(D, D, bias=False)
        s.ln = nn.LayerNorm(D); s.sc = math.sqrt(dh); s.nh = N_HEADS; s.dh = dh
        for w in [s.WQ, s.WK, s.WV, s.op]: nn.init.normal_(w.weight, std=0.02)
    def forward(s, h):
        B, S, _ = h.shape
        Q = s.WQ(h).view(B, S, s.nh, s.dh).transpose(1, 2)
        K = s.WK(h).view(B, S, s.nh, s.dh).transpose(1, 2)
        V = s.WV(h).view(B, S, s.nh, s.dh).transpose(1, 2)
        sc = Q @ K.transpose(-2, -1) / s.sc
        m = torch.triu(torch.ones(S, S), diagonal=1).bool()
        sc = sc.masked_fill(m.unsqueeze(0).unsqueeze(0), float('-inf'))
        return s.ln(h + s.op((F.softmax(sc, -1) @ V).transpose(1, 2).reshape(B, S, D)))

class FF(nn.Module):
    def __init__(s):
        super().__init__()
        s.g = nn.Linear(D, D*2, bias=False); s.v = nn.Linear(D, D*2, bias=False)
        s.o = nn.Linear(D*2, D, bias=False); s.n = nn.LayerNorm(D)
        for w in [s.g, s.v, s.o]: nn.init.normal_(w.weight, std=0.02)
    def forward(s, h): return s.n(h + s.o(F.silu(s.g(h)) * s.v(h)))

class Block(nn.Module):
    def __init__(s): super().__init__(); s.attn = Attn(); s.ff = FF()
    def forward(s, h): return s.ff(s.attn(h))

class LM(nn.Module):
    def __init__(s):
        super().__init__()
        s.te = nn.Embedding(VOCAB, D); s.pe = nn.Embedding(512, D)
        s.blocks = nn.ModuleList([Block() for _ in range(N_STU)])
        s.ln_f = nn.LayerNorm(D); s.head = nn.Linear(D, VOCAB, bias=False)
        s.head.weight = s.te.weight
        nn.init.normal_(s.te.weight, std=0.02); nn.init.normal_(s.pe.weight, std=0.02)
    def forward(s, x, y=None):
        h = s.te(x) + s.pe(torch.arange(x.shape[1]))
        for b in s.blocks: h = b(h)
        lg = s.head(s.ln_f(h))
        return lg, (F.cross_entropy(lg.view(-1, VOCAB), y.reshape(-1)) if y is not None else None)

def batches(seed, k):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(k):
        i = torch.randint(len(train_t) - SEQ - 1, (BATCH,), generator=g)
        out.append((torch.stack([train_t[j:j+SEQ] for j in i]),
                    torch.stack([train_t[j+1:j+SEQ+1] for j in i])))
    return out

@torch.no_grad()
def evalv(m, k=12):
    g = torch.Generator().manual_seed(999); tot = 0.
    m.eval()
    for _ in range(k):
        i = torch.randint(len(val_t) - SEQ - 1, (BATCH,), generator=g)
        x = torch.stack([val_t[j:j+SEQ] for j in i])
        y = torch.stack([val_t[j+1:j+SEQ+1] for j in i])
        tot += float(m(x, y)[1])
    m.train(); return tot / k

# ---- rotations --------------------------------------------------------
_ROT = {}
def rot(shape):
    if shape not in _ROT:
        g = torch.Generator().manual_seed(hash(shape) % (2**31))
        Q, _ = torch.linalg.qr(torch.randn(shape[0], shape[0], generator=g))
        R, _ = torch.linalg.qr(torch.randn(shape[1], shape[1], generator=g))
        _ROT[shape] = (Q, R)
    return _ROT[shape]

def st_tensor(u, k):
    fl = u.reshape(-1); out = torch.zeros_like(fl)
    tail = torch.ones_like(fl, dtype=torch.bool)
    if k > 0:
        idx = torch.topk(fl.abs(), min(k, fl.numel()), sorted=False).indices
        tail[idx] = False; out[idx] = fl[idx]
    if tail.any():
        out[tail] = torch.sign(fl[tail]) * float(fl[tail].abs().mean())
    return out.view_as(u)

def compress(u, grad, mode, r, conj):
    out = {}
    for nme, t in u.items():
        if t.dim() < 2: out[nme] = t.clone(); continue
        sh = tuple(t.shape)
        if conj:
            Q, R = rot(sh); tt = Q.T @ t @ R; gg = Q.T @ grad[nme] @ R
        else:
            tt, gg = t, grad[nme]
        if mode == 'signtop':
            h = st_tensor(tt, signtop_budget(sh, r))
        else:
            rr = min(r, min(sh))
            U, _, Vt = torch.linalg.svd(gg, full_matrices=False)
            P, W = U[:, :rr], Vt[:rr].T
            h = P @ (P.T @ tt @ W) @ W.T
        if conj:
            Q, R = rot(sh); h = Q @ h @ R.T
        out[nme] = h
    return out

def run(mode, r, conj, seed=0, log=None):
    torch.manual_seed(seed); m = LM(); m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=LR*5, betas=(0.9, 0.95), weight_decay=0.1)
    bs = batches(1234, STEPS)
    rets = []
    for s, (x, y) in enumerate(bs, 1):
        for pg in opt.param_groups:
            pg['lr'] = LR*5*min(1., s/10)
        _, l = m(x, y); opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        if mode == 'none':
            opt.step(); continue
        grad = {k: p.grad.detach().clone() for k, p in m.named_parameters()
                if p.dim() == 2}
        before = {k: p.data.clone() for k, p in m.named_parameters() if p.dim() == 2}
        opt.step()
        u = {k: p.data - before[k] for k, p in m.named_parameters() if k in before}
        uh = compress(u, grad, mode, r, conj)
        un = math.sqrt(sum(float((v*v).sum()) for v in u.values()))
        hn = math.sqrt(sum(float((v*v).sum()) for v in uh.values()))
        sc = un / hn if hn > 1e-30 else 1.0
        rets.append(hn / un if un > 1e-30 else 1.0)
        with torch.no_grad():
            for k, p in m.named_parameters():
                if k in uh: p.data.copy_(before[k] + uh[k] * sc)
    return evalv(m), (float(np.mean(rets)) if rets else 1.0)

if __name__ == '__main__':
    arms = [('none', 0, False), ('lowrank', 32, False), ('lowrank', 32, True),
            ('signtop', 32, False), ('signtop', 32, True),
            ('signtop', 4, False), ('signtop', 4, True),
            ('lowrank', 4, False), ('lowrank', 4, True)]
    print(f"{'arm':>22}{'final':>9}{'|uh|/|u|':>11}{'sec':>7}")
    res = {}
    for mode, r, conj in arms:
        t0 = time.time(); v, ret = run(mode, r, conj)
        nm = f"{mode} r={r}{' ROT' if conj else ''}" if mode != 'none' else 'uncompressed'
        res[nm] = v
        print(f"{nm:>22}{v:>9.4f}{ret:>11.3f}{time.time()-t0:>7.0f}", flush=True)
    json.dump(res, open('/home/claude/rot_results.json', 'w'), indent=1)
