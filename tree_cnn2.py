#!/usr/bin/env python3
"""TREE_CNN2 -- dilated causal convolution with tree-matched receptive fields.

    python3 tree_cnn2.py --steps 48000 --seed 1 --save C2.pt
    python3 tree_cnn2.py --load C2.pt      # probes only

WHAT WAS WRONG WITH THE FIRST VERSION
-------------------------------------
It used stride-2 pooling and gave each position only the COMPLETED nodes
strictly to its left. At position 5 the level-1 node available covered leaves
2-3, not the partial node holding leaf 4, so the sibling arrived only as a raw
embedding and half of every open subtree was invisible. `local` stalled at
3.83 where the transformer reaches 2.04 -- the model was starved of the easy
information, so the run said nothing about hierarchy.

THIS VERSION
------------
Dilated causal convolutions, WaveNet form. Layer l has dilation 2^(l-1) and
kernel 2, so position t sees a contiguous window of 2^l leaves ending at t --
the same receptive fields as the grammar's levels, but with NOTHING dropped:

    layer 1  dilation 1   window 2    (sibling)
    layer 2  dilation 2   window 4
    layer 3  dilation 4   window 8
    layer 4  dilation 8   window 16

Left-padding makes it causal: output t depends on inputs <= t only. Weight
sharing across positions is the inductive bias under test; unlike the stride
version, the information available matches what the transformer gets.

THE TEST
--------
Transformer decodes level 3 at 0.028 (probe) / 0.056 (direct supervision).
  level 3 >> that  => inductive bias was the problem
  level 3 ~ that   => locality is not the issue; both compute bottom-up while
                      the grammar runs top-down, and no feedforward bottom-up
                      net recovers a latent needing marginalisation over parses
`local` must land near the transformer's 2.04 for the run to mean anything.
If it does not, the model is starved again and the level-3 number is void.
"""
import json, math, argparse, time, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/tmp")
ap.add_argument("--d-model", type=int, default=64)
ap.add_argument("--layers", type=int, default=4)
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

class DilatedBlock(nn.Module):
    def __init__(s, d, dil):
        super().__init__()
        s.dil = dil
        s.conv = nn.Conv1d(d, d, kernel_size=2, dilation=dil)
        s.n1 = nn.LayerNorm(d); s.n2 = nn.LayerNorm(d)
        s.mlp = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
    def forward(s, x):                       # x: B x T x d
        z = s.n1(x).transpose(1, 2)
        z = F.pad(z, (s.dil, 0))             # left pad => causal
        z = s.conv(z).transpose(1, 2)
        x = x + z
        return x + s.mlp(s.n2(x))

class TreeCNN(nn.Module):
    def __init__(s, d, nl):
        super().__init__()
        s.te = nn.Embedding(NLEAF, d); s.pe = nn.Embedding(SEQ+4, d)
        s.blocks = nn.ModuleList([DilatedBlock(d, 2**i) for i in range(nl)])
        s.lnf = nn.LayerNorm(d); s.head = nn.Linear(d, NLEAF)
    def hidden(s, idx):
        h = s.te(idx) + s.pe(torch.arange(idx.shape[1], device=idx.device))[None]
        for b in s.blocks: h = b(h)
        return s.lnf(h)
    def forward(s, idx, targets=None):
        h = s.hidden(idx); lg = s.head(h)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(lg.reshape(-1, NLEAF), targets.reshape(-1),
                                   ignore_index=-100)
        return lg, h, loss

model = TreeCNN(a.d_model, a.layers)
print(f"  dilated causal CNN  d={a.d_model} layers={a.layers} "
      f"(dilations {[2**i for i in range(a.layers)]}, windows "
      f"{[2**(i+1) for i in range(a.layers)]})  "
      f"params={sum(p.numel() for p in model.parameters()):,}")
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
                                      if KIND[t+1] == "local"]].mean())
                hi = float((ce*k)[:, [t for t in range(SEQ-1)
                                      if KIND[t+1] == "higher"]].mean())
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
    X, Y = sample(a.n_probe); h = model.hidden(X)
print(f"\n  DECODING THE LATENT PARSE   (chance = {1.0/V:.4f})")
tf = {(1,'local'):0.479,(1,'higher'):0.213,(2,'local'):0.072,(2,'higher'):0.064,
      (3,'local'):0.028,(3,'higher'):0.024,(4,'local'):0.020,(4,'higher'):0.018}
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
print(f"\n  check `local` first: near 2.04 means the model got the easy part and")
print(f"  the level-3 number is interpretable. Far above it means starved again.")
