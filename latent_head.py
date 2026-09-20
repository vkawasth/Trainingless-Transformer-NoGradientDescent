#!/usr/bin/env python3
"""LATENT_HEAD -- marginalise over the open constituent instead of naming it.

    python3 latent_head.py --steps 48000 --seed 1 --save M1.pt
    python3 latent_head.py --load M1.pt            # probes only
    python3 latent_head.py --no-latent ...         # matched control

THE DIAGNOSIS THIS ADDRESSES
----------------------------
The auxiliary head asked the model to NAME the level-3 symbol and reached
0.056. But that symbol has 2.29 nats of residual entropy given the prefix --
about 10 live candidates out of 64. Asking for a point estimate of something
genuinely uncertain gives a target that is mostly noise, which is a plausible
reason the supervision failed to install anything.

A marginalising head never asks the model to commit:

    P(x_t | x_<t)  =  sum_s  q(s | x_<t) * E(x_t | s)

  q(s | x_<t)   a posterior over the 64 internal symbols, from h[t]
  E(x_t | s)    an emission table, symbol -> leaf distribution, learned

Trained on plain next-token likelihood -- no auxiliary target, nothing to be
wrong about. The latent is structurally forced to exist because the output
factors through it. This is inside-outside's shape written as a layer.

WHAT COUNTS
-----------
  loss:  does `higher` move toward the 3.5527 floor
  probe: is level 3 decodable from q(s|x_<t) -- and from h[t] -- where the
         transformer got 0.056
A control arm (--no-latent) has the same parameter count with a plain head,
so any difference is the factorisation and not capacity.
"""
import json, math, argparse, time, random, os, sys
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/tmp")
ap.add_argument("--d-model", type=int, default=64)
ap.add_argument("--n-heads", type=int, default=4)
ap.add_argument("--layers", type=int, default=4)
ap.add_argument("--n-latent", type=int, default=0, help="0 = use nsym from meta")
ap.add_argument("--steps", type=int, default=48000)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--no-latent", action="store_true", help="control: plain head")
ap.add_argument("--save", default="")
ap.add_argument("--load", default="")
ap.add_argument("--eval-every", type=int, default=2000)
ap.add_argument("--n-probe", type=int, default=4000)
a = ap.parse_args()

M = json.load(open(f"{a.data}/rhm_meta.json"))
L, V, SEQ, NLEAF = M["depth"], M["nsym"], M["seq_len"], M["nleaf"]
KIND = M["pos_kind"]
R = {int(l): {int(i): [tuple(t) for t in v] for i, v in d.items()}
     for l, d in M["rules_all"].items()}
K = a.n_latent or V
rng = random.Random(a.seed); torch.manual_seed(a.seed)

def sample(nb):
    X = torch.zeros(nb, SEQ, dtype=torch.long)
    Y = {l: torch.zeros(nb, SEQ, dtype=torch.long) for l in range(1, L+1)}
    for b in range(nb):
        def go(s, l, off):
            if l > 0:
                for t in range(off, off + 2**l): Y[l][b, t] = s
            if l == 0: return [s]
            x, y = rng.choice(R[l][s])
            return go(x, l-1, off) + go(y, l-1, off + 2**(l-1))
        X[b] = torch.tensor(go(rng.randrange(V), L, 0))
    return X, Y

# ------------------------------------------------------------------ model
class Attn(nn.Module):
    def __init__(s, d, nh):
        super().__init__(); s.nh, s.dh = nh, d//nh
        s.q, s.k, s.v, s.o = (nn.Linear(d, d, bias=False) for _ in range(4))
    def forward(s, x):
        B, T, D = x.shape
        sp = lambda t: t.view(B, T, s.nh, s.dh).transpose(1, 2)
        q, k, v = sp(s.q(x)), sp(s.k(x)), sp(s.v(x))
        att = (q @ k.transpose(-2, -1)) / math.sqrt(s.dh)
        m = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
        att = att.masked_fill(m, float("-inf")).softmax(-1)
        return s.o((att @ v).transpose(1, 2).contiguous().view(B, T, D))

class Block(nn.Module):
    def __init__(s, d, nh):
        super().__init__()
        s.n1, s.at = nn.LayerNorm(d), Attn(d, nh)
        s.n2 = nn.LayerNorm(d)
        s.mlp = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
    def forward(s, x):
        x = x + s.at(s.n1(x)); return x + s.mlp(s.n2(x))

class LatentLM(nn.Module):
    """P(x_t) = sum_s q(s|h_t) E(x_t|s), or a plain linear head as control."""
    def __init__(s, d, nh, nl, K, latent=True):
        super().__init__()
        s.te = nn.Embedding(NLEAF, d); s.pe = nn.Embedding(SEQ + 4, d)
        s.blocks = nn.ModuleList([Block(d, nh) for _ in range(nl)])
        s.lnf = nn.LayerNorm(d)
        s.latent = latent
        if latent:
            s.qhead = nn.Linear(d, K)              # posterior over constituents
            s.emit = nn.Parameter(torch.randn(K, NLEAF) * 0.02)  # symbol -> leaf
        else:
            # same parameter count, no factorisation
            s.mid = nn.Linear(d, K); s.out = nn.Linear(K, NLEAF)
    def hidden(s, idx):
        h = s.te(idx) + s.pe(torch.arange(idx.shape[1], device=idx.device))[None]
        for b in s.blocks: h = b(h)
        return s.lnf(h)
    def forward(s, idx, targets=None):
        h = s.hidden(idx)
        if s.latent:
            logq = F.log_softmax(s.qhead(h), -1)              # B x T x K
            loge = F.log_softmax(s.emit, -1)                  # K x NLEAF
            # log sum_s q(s) E(x|s), in log space
            logp = torch.logsumexp(logq.unsqueeze(-1) + loge[None, None], dim=2)
        else:
            logp = F.log_softmax(s.out(F.gelu(s.mid(h))), -1)
            logq = None
        loss = None
        if targets is not None:
            loss = F.nll_loss(logp.reshape(-1, NLEAF), targets.reshape(-1),
                              ignore_index=-100)
        return logp, logq, h, loss

model = LatentLM(a.d_model, a.n_heads, a.layers, K, latent=not a.no_latent)
tag = "plain head (control)" if a.no_latent else f"latent head, K={K}"
print(f"  {a.layers}-layer  d={a.d_model}  {tag}  "
      f"params={sum(p.numel() for p in model.parameters()):,}  seed={a.seed}")
print(f"  floors: local 1.6034  higher 3.5527   "
      f"(transformer 48k: local 2.04  higher 3.77)")

if a.load:
    model.load_state_dict(torch.load(a.load, map_location="cpu")); model.eval()
else:
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95),
                            weight_decay=0.01)
    t0 = time.time()
    for step in range(1, a.steps + 1):
        X, _ = sample(a.batch)
        Y = torch.cat([X[:, 1:], torch.full((X.shape[0], 1), -100)], 1)
        _, _, _, loss = model(X, Y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % a.eval_every == 0 or step == 1:
            with torch.no_grad():
                Xv, _ = sample(256)
                Yv = torch.cat([Xv[:, 1:], torch.full((256, 1), -100)], 1)
                lp, _, _, _ = model(Xv)
                ce = F.nll_loss(lp.reshape(-1, NLEAF), Yv.reshape(-1),
                                ignore_index=-100, reduction='none').view(Xv.shape)
                k = (Yv != -100).double()
                lo = float((ce*k)[:, [t for t in range(SEQ-1)
                                      if KIND[t+1] == "local"]].mean())
                hi = float((ce*k)[:, [t for t in range(SEQ-1)
                                      if KIND[t+1] == "higher"]].mean())
            print(f"  {step:6d}  local={lo:.4f}  higher={hi:.4f}  "
                  f"[{time.time()-t0:.0f}s]")
    if a.save: torch.save(model.state_dict(), a.save); print(f"  saved {a.save}")

# --------------------------------------------------------------- probes
def probe(Hm, lab, nclass, epochs=300):
    n = Hm.shape[0]; k = int(0.7*n); idx = torch.randperm(n)
    tr, te = idx[:k], idx[k:]
    w = nn.Linear(Hm.shape[1], nclass)
    opt = torch.optim.Adam(w.parameters(), lr=3e-3, weight_decay=1e-4)
    for _ in range(epochs):
        opt.zero_grad(); F.cross_entropy(w(Hm[tr]), lab[tr]).backward(); opt.step()
    with torch.no_grad():
        return float((w(Hm[te]).argmax(-1) == lab[te]).float().mean())

model.eval()
with torch.no_grad():
    X, Y = sample(a.n_probe)
    _, logq, h, _ = model(X)

print(f"\n  DECODING THE LATENT PARSE   (chance = {1.0/V:.4f})")
tf = {(1,'local'):0.479,(1,'higher'):0.213,(2,'local'):0.072,(2,'higher'):0.064,
      (3,'local'):0.028,(3,'higher'):0.024,(4,'local'):0.020,(4,'higher'):0.018}
srcs = [("h[t]", h)] + ([("q(s|x)", logq)] if logq is not None else [])
print(f"  {'source':>8}{'level':>7}{'kind':>9}{'acc':>9}{'shuf':>8}   transformer")
for nm, Hsrc in srcs:
    for l in range(1, L+1):
        for kind in ("local", "higher"):
            ts = [t for t in range(SEQ) if KIND[t] == kind]
            f = torch.cat([Hsrc[:, t, :] for t in ts])
            lab = torch.tensor([int(Y[l][i, t]) for t in ts
                                for i in range(Hsrc.shape[0])], dtype=torch.long)
            acc = probe(f, lab, V)
            sh = probe(f, lab[torch.randperm(len(lab))], V)
            print(f"  {nm:>8}{l:>7}{kind:>9}{acc:>9.3f}{sh:>8.3f}"
                  f"   {tf[(l,kind)]:.3f}")

if logq is not None:
    with torch.no_grad():
        ent = float((-(logq.exp()*logq).sum(-1)).mean())
    print(f"\n  mean posterior entropy over the latent = {ent:.3f} nats "
          f"(chance {math.log(K):.3f})")
    print(f"  exact H(level-3 | prefix) at `higher` positions = 2.290 nats")
    print(f"  a posterior near the exact value means the head is doing the")
    print(f"  marginalisation the task requires; near chance means it collapsed")
