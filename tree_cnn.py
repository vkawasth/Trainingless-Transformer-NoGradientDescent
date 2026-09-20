#!/usr/bin/env python3
"""TREE_CNN -- does convolution's inductive bias recover level 3?

    python3 tree_cnn.py --steps 48000 --seed 1 --save C1.pt
    python3 tree_cnn.py --load C1.pt          # probes only

THE TEST
--------
RHM is a hierarchical convolution: the SAME rule table applies at every
level-1 node, every level-2 node, and so on. A transformer must discover that
invariance; a stride-2 stack gets it by construction. Layer l has receptive
field 2^l, matching the grammar's level l exactly.

    layer 1: sibling pairs        (level 1)
    layer 2: pairs of pairs       (level 2)
    layer 3: 8 leaves             (level 3)
    layer 4: 16 leaves            (level 4)

The transformer decodes level 3 at 0.056 with direct supervision. If this
reaches, say, 0.3+, the failure was inductive bias: attention had to learn the
tree structure convolution assumes. If it also lands near 0.056, locality is
not the issue and the problem is inference DIRECTION -- the model computes
bottom-up while the generative process runs top-down, and no feedforward
bottom-up architecture recovers a latent that requires marginalising over
parses. That would close this architectural family.

Causality: a stride-2 conv at node n sees leaves [2^l n, 2^l(n+1)), which
includes leaves AFTER the position being predicted. For next-token prediction
each position is given only the completed nodes strictly to its left; the
decode probe is run on the same causal features.
"""
import json, math, argparse, time, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/tmp")
ap.add_argument("--d-model", type=int, default=64)
ap.add_argument("--steps", type=int, default=48000)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--seed", type=int, default=1)
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
rng = random.Random(a.seed)
torch.manual_seed(a.seed)

def sample(nb):
    """sequences plus the level-l symbol above every leaf"""
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

class TreeCNN(nn.Module):
    """stride-2 stack: layer l has receptive field exactly 2^l leaves."""
    def __init__(s, d, nlev=L, vocab=NLEAF):
        super().__init__()
        s.emb = nn.Embedding(vocab, d)
        s.convs = nn.ModuleList([nn.Conv1d(d, d, kernel_size=2, stride=2)
                                 for _ in range(nlev)])
        s.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(nlev)])
        s.mlps = nn.ModuleList([nn.Sequential(nn.Linear(d, 4*d), nn.GELU(),
                                              nn.Linear(4*d, d))
                                for _ in range(nlev)])
        s.head = nn.Linear(d, vocab)
        s.d, s.nlev = d, nlev
    def levels(s, idx):
        """returns [h0 (leaves), h1, ..., hL], h_l has SEQ/2^l nodes"""
        h = s.emb(idx)                                  # B x T x d
        outs = [h]
        for i, (c, n, m) in enumerate(zip(s.convs, s.norms, s.mlps)):
            z = c(outs[-1].transpose(1, 2)).transpose(1, 2)
            z = n(z); z = z + m(z)
            outs.append(z)
        return outs
    def causal_state(s, idx):
        """For predicting token t, gather only nodes COMPLETED strictly before
        t. Node n at level l covers [2^l n, 2^l(n+1)), so it is usable at t
        only if 2^l(n+1) <= t. Take, per level, the last such node."""
        outs = s.levels(idx)
        B, T = idx.shape
        feats = [torch.zeros(B, T, s.d, device=idx.device)]      # leaf t-1
        feats[0][:, 1:, :] = outs[0][:, :-1, :]
        for l in range(1, s.nlev + 1):
            w = 2 ** l
            f = torch.zeros(B, T, s.d, device=idx.device)
            for t in range(T):
                # node n covers [w*n, w*(n+1)); usable at t iff w*(n+1) <= t,
                # so the last completed node is floor(t/w) - 1.
                n = (t // w) - 1
                if n >= 0: f[:, t, :] = outs[l][:, n, :]
            feats.append(f)
        return torch.cat(feats, -1), outs

class Model(nn.Module):
    def __init__(s, d):
        super().__init__()
        s.net = TreeCNN(d)
        s.read = nn.Linear(d * (L + 1), NLEAF)
    def forward(s, idx, targets=None):
        st, outs = s.net.causal_state(idx)
        logits = s.read(st)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, NLEAF), targets.reshape(-1),
                                   ignore_index=-100)
        return logits, loss, st, outs

model = Model(a.d_model)
print(f"  tree-CNN  d={a.d_model} levels={L}  "
      f"params={sum(p.numel() for p in model.parameters()):,}  seed={a.seed}")

if a.load:
    model.load_state_dict(torch.load(a.load, map_location="cpu")); model.eval()
else:
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95),
                            weight_decay=0.01)
    t0 = time.time()
    for step in range(1, a.steps + 1):
        X, _ = sample(a.batch)
        Y = torch.cat([X[:, 1:], torch.full((X.shape[0], 1), -100)], 1)
        _, loss, _, _ = model(X, Y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % a.eval_every == 0 or step == 1:
            with torch.no_grad():
                Xv, _ = sample(256)
                Yv = torch.cat([Xv[:, 1:], torch.full((256, 1), -100)], 1)
                lg, _, _, _ = model(Xv, Yv)
                ce = F.cross_entropy(lg.reshape(-1, NLEAF), Yv.reshape(-1),
                                     ignore_index=-100, reduction='none'
                                     ).view(Xv.shape)
                keep = Yv != -100
                lo = float((ce * keep)[:, [t for t in range(SEQ-1)
                                           if KIND[(t+1) % SEQ] == "local"]].mean())
                hi = float((ce * keep)[:, [t for t in range(SEQ-1)
                                           if KIND[(t+1) % SEQ] == "higher"]].mean())
            print(f"  {step:6d}  local={lo:.4f}  higher={hi:.4f}  "
                  f"[{time.time()-t0:.0f}s]")
    if a.save: torch.save(model.state_dict(), a.save); print(f"  saved {a.save}")

# ---------------- the probe: can level l be decoded? ----------------------
@torch.no_grad()
def feats_and_labels(m, n):
    X, Y = sample(n)
    _, _, st, _ = m(X)
    return st, Y

def probe(H, lab, V_, epochs=300):
    n = H.shape[0]; k = int(0.7*n); idx = torch.randperm(n)
    tr, te = idx[:k], idx[k:]
    w = nn.Linear(H.shape[1], V_)
    opt = torch.optim.Adam(w.parameters(), lr=3e-3, weight_decay=1e-4)
    for _ in range(epochs):
        opt.zero_grad(); F.cross_entropy(w(H[tr]), lab[tr]).backward(); opt.step()
    with torch.no_grad():
        return float((w(H[te]).argmax(-1) == lab[te]).float().mean())

model.eval()
H, Y = feats_and_labels(model, a.n_probe)
print(f"\n  DECODING THE LATENT PARSE from the causal state  "
      f"(chance = {1.0/V:.4f})")
print(f"  {'level':>7}{'kind':>9}{'acc':>9}{'shuffled':>10}   transformer")
tf = {(1,'local'):0.479,(1,'higher'):0.213,(2,'local'):0.072,(2,'higher'):0.064,
      (3,'local'):0.028,(3,'higher'):0.024,(4,'local'):0.020,(4,'higher'):0.018}
for l in range(1, L+1):
    for kind in ("local", "higher"):
        ts = [t for t in range(SEQ) if KIND[t] == kind]
        f = torch.cat([H[:, t, :] for t in ts])
        lab = torch.tensor([int(Y[l][i, t]) for t in ts for i in range(H.shape[0])],
                           dtype=torch.long)
        acc = probe(f, lab, V)
        sh = probe(f, lab[torch.randperm(len(lab))], V)
        print(f"  {l:>7}{kind:>9}{acc:>9.3f}{sh:>10.3f}   {tf[(l,kind)]:.3f}")
print(f"\n  level 3 >> 0.056 => inductive bias was the problem")
print(f"  level 3 ~ 0.056  => locality is not the issue; the model computes")
print(f"                      bottom-up while the grammar runs top-down, and")
print(f"                      no feedforward bottom-up net recovers a latent")
print(f"                      that needs marginalising over parses")
