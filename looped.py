#!/usr/bin/env python3
"""LOOPED -- recurrent depth: one block, applied L times, shared weights.

    python3 looped.py --loops 4 --steps 48000 --seed 1 --save P4.pt
    python3 looped.py --loops 8 --steps 48000 --seed 1 --save P8.pt
    python3 looped.py --loops 4 --no-share --steps 48000 --save P4u.pt   # control

WHY THIS IS THE LAST ARCHITECTURE WORTH TRYING
----------------------------------------------
Inside-outside applies the SAME rule tensor at every level of the tree. That
is weight sharing across DEPTH. The depth sweep varied the number of DISTINCT
layers (2, 4, 8) and never gave one layer the chance to run four times, so the
one inductive bias that matches the algorithm was never tested.

A looped transformer runs a single block `loops` times. Parameter count stays
at one block; computation scales with the loop count. If iterated message
passing is what level 3 needs, this is the family that can express it.

--no-share is the matched control: the same number of applications with
independent weights, i.e. an ordinary deep transformer. A gap between them is
the weight sharing; no gap means loop count alone was doing the work.

Reference points: transformer 48k steps local 2.04 higher 3.77; floors 1.6034
and 3.5527; level-3 decoding pinned at 0.028-0.034 across eight configurations
against a shuffled-label floor of 0.025-0.034.
"""
import json, math, argparse, time, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/tmp")
ap.add_argument("--d-model", type=int, default=64)
ap.add_argument("--n-heads", type=int, default=4)
ap.add_argument("--loops", type=int, default=4)
ap.add_argument("--no-share", action="store_true")
ap.add_argument("--steps", type=int, default=48000)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--save", default=""); ap.add_argument("--load", default="")
ap.add_argument("--eval-every", type=int, default=2000)
ap.add_argument("--n-probe", type=int, default=4000)
a = ap.parse_args()

M = json.load(open(f"{a.data}/rhm_meta.json"))
L, V, SEQ, NLEAF = M["depth"], M["nsym"], M["seq_len"], M["nleaf"]
KIND = M["pos_kind"]
R = {int(l): {int(i): [tuple(t) for t in v] for i, v in d.items()}
     for l, d in M["rules_all"].items()}
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
        return s.o(((att.masked_fill(m, float("-inf")).softmax(-1)) @ v)
                   .transpose(1, 2).contiguous().view(B, T, D))

class Block(nn.Module):
    def __init__(s, d, nh):
        super().__init__()
        s.n1, s.at, s.n2 = nn.LayerNorm(d), Attn(d, nh), nn.LayerNorm(d)
        s.mlp = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
    def forward(s, x):
        x = x + s.at(s.n1(x)); return x + s.mlp(s.n2(x))

class Looped(nn.Module):
    def __init__(s, d, nh, loops, share=True):
        super().__init__()
        s.te = nn.Embedding(NLEAF, d); s.pe = nn.Embedding(SEQ+4, d)
        s.loops, s.share = loops, share
        s.blocks = nn.ModuleList([Block(d, nh)] if share else
                                 [Block(d, nh) for _ in range(loops)])
        # a per-iteration marker so the shared block can tell which round it is
        s.step_emb = nn.Embedding(loops, d) if share else None
        s.lnf = nn.LayerNorm(d); s.head = nn.Linear(d, NLEAF)
    def hidden(s, idx, keep=False):
        h = s.te(idx) + s.pe(torch.arange(idx.shape[1], device=idx.device))[None]
        states = []
        for i in range(s.loops):
            if s.share:
                h = h + s.step_emb(torch.tensor(i, device=idx.device))
                h = s.blocks[0](h)
            else:
                h = s.blocks[i](h)
            if keep: states.append(s.lnf(h))
        h = s.lnf(h)
        return (h, states) if keep else h
    def forward(s, idx, tgt=None):
        h = s.hidden(idx); lg = s.head(h)
        loss = None
        if tgt is not None:
            loss = F.cross_entropy(lg.reshape(-1, NLEAF), tgt.reshape(-1),
                                   ignore_index=-100)
        return lg, h, loss

model = Looped(a.d_model, a.n_heads, a.loops, share=not a.no_share)
print(f"  looped d={a.d_model} loops={a.loops} "
      f"{'SHARED (1 block)' if not a.no_share else 'unshared (control)'}  "
      f"params={sum(p.numel() for p in model.parameters()):,}  seed={a.seed}")
print(f"  floors: local 1.6034  higher 3.5527   "
      f"transformer 48k: local 2.04  higher 3.77")

if a.load:
    model.load_state_dict(torch.load(a.load, map_location="cpu")); model.eval()
else:
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95),
                            weight_decay=0.01)
    t0 = time.time()
    for step in range(1, a.steps+1):
        X, _ = sample(a.batch)
        Y = torch.cat([X[:, 1:], torch.full((X.shape[0], 1), -100)], 1)
        _, _, loss = model(X, Y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % a.eval_every == 0 or step == 1:
            with torch.no_grad():
                Xv, _ = sample(256)
                Yv = torch.cat([Xv[:, 1:], torch.full((256, 1), -100)], 1)
                lg, _, _ = model(Xv)
                ce = F.cross_entropy(lg.reshape(-1, NLEAF), Yv.reshape(-1),
                                     ignore_index=-100, reduction='none'
                                     ).view(Xv.shape)
                k = (Yv != -100).double()
                lo = float((ce*k)[:, [t for t in range(SEQ-1)
                                      if KIND[t+1]=="local"]].mean())
                hi = float((ce*k)[:, [t for t in range(SEQ-1)
                                      if KIND[t+1]=="higher"]].mean())
            print(f"  {step:6d}  local={lo:.4f}  higher={hi:.4f}  "
                  f"[{time.time()-t0:.0f}s]")
    if a.save: torch.save(model.state_dict(), a.save); print(f"  saved {a.save}")

def probe(Hm, lab, nc, ep=300):
    n = Hm.shape[0]; k = int(0.7*n); i = torch.randperm(n)
    tr, te = i[:k], i[k:]
    w = nn.Linear(Hm.shape[1], nc)
    o = torch.optim.Adam(w.parameters(), lr=3e-3, weight_decay=1e-4)
    for _ in range(ep):
        o.zero_grad(); F.cross_entropy(w(Hm[tr]), lab[tr]).backward(); o.step()
    with torch.no_grad():
        return float((w(Hm[te]).argmax(-1) == lab[te]).float().mean())

model.eval()
with torch.no_grad():
    X, Y = sample(a.n_probe); h, states = model.hidden(X, keep=True)
tf = {(1,'local'):0.479,(1,'higher'):0.213,(2,'local'):0.072,(2,'higher'):0.064,
      (3,'local'):0.028,(3,'higher'):0.024,(4,'local'):0.020,(4,'higher'):0.018}
print(f"\n  DECODING THE LATENT PARSE from the FINAL state  "
      f"(chance = {1.0/V:.4f})")
print(f"  {'level':>7}{'kind':>9}{'acc':>9}{'shuf':>8}   transformer")
for l in range(1, L+1):
    for kind in ("local", "higher"):
        ts = [t for t in range(SEQ) if KIND[t] == kind]
        f = torch.cat([h[:, t, :] for t in ts])
        lab = torch.tensor([int(Y[l][i, t]) for t in ts for i in range(h.shape[0])],
                           dtype=torch.long)
        print(f"  {l:>7}{kind:>9}{probe(f, lab, V):>9.3f}"
              f"{probe(f, lab[torch.randperm(len(lab))], V):>8.3f}"
              f"   {tf[(l,kind)]:.3f}")

# does iteration i carry level i? that is the signature of message passing
print(f"\n  PER-ITERATION: does round i encode level i? (`higher` positions)")
print(f"  {'round':>6}" + "".join(f"{'L'+str(l):>9}" for l in range(1, L+1)))
ts = [t for t in range(SEQ) if KIND[t] == "higher"]
for i, st in enumerate(states):
    row = []
    for l in range(1, L+1):
        f = torch.cat([st[:, t, :] for t in ts])
        lab = torch.tensor([int(Y[l][b, t]) for t in ts for b in range(st.shape[0])],
                           dtype=torch.long)
        row.append(probe(f, lab, V))
    print(f"  {i+1:>6}" + "".join(f"{x:>9.3f}" for x in row))
print(f"\n  a rising diagonal (round 1 -> L1, round 2 -> L2, ...) would be")
print(f"  iterated message passing. A flat table means the loop is just depth.")
