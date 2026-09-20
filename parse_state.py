#!/usr/bin/env python3
"""PARSE_STATE -- does the model know where it is in the parse?

    python3 parse_state.py --load L8.pt --layers 8

Every structural measurement so far read PAIR representations at an appended
PROBE position. None looked at what the model does while PREDICTING: at
position t, having seen x_0..x_t, which latent symbols are active above it?
That is the quantity `higher` depends on.

For each level l, the leaf at position t sits under exactly one level-l node.
This decodes that node's symbol from the residual stream at position t with a
linear probe, trained on some sequences and tested on held-out ones.

CONTROLS, because a probe on 64 dims with V classes overfits easily:
  shuffled labels   same features, symbol labels permuted -> chance
  random model      an untrained network of the same shape -> what the
                    architecture gives for free before any learning
Accuracy above BOTH is evidence the parse state is represented.

Chance is 1/V. Report per level and per position kind: if the model decodes
level 1 but not levels 3-4, it tracks siblings and not the hierarchy, which
is exactly the shape of the local/higher loss split.
"""
import json, math, argparse, os, sys, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--load", required=True)
ap.add_argument("--data", default="/tmp")
ap.add_argument("--layers", type=int, default=2)
ap.add_argument("--d-model", type=int, default=64)
ap.add_argument("--n-heads", type=int, default=4)
ap.add_argument("--n-seq", type=int, default=4000)
ap.add_argument("--epochs", type=int, default=300)
a = ap.parse_args()

sys.argv = [sys.argv[0], "--mlp", "--data", a.data]
_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rhm16.py")).read()
_src = _src.split("# ============================================================ PHASE 1")[0]
_ns = {"__name__": "_rhm"}; exec(compile(_src, "rhm16.py", "exec"), _ns)
LM2, META = _ns["LM2"], _ns["META"]
L, V, SEQ, NLEAF = META["depth"], META["nsym"], META["seq_len"], META["nleaf"]
KIND = META["pos_kind"]
R = {int(l): {int(i): [tuple(t) for t in v] for i, v in d.items()}
     for l, d in META["rules_all"].items()}
rng = random.Random(0)

def sample_with_parse():
    """returns leaves, and node[l][t] = level-l symbol above leaf t"""
    node = {l: [None]*SEQ for l in range(1, L+1)}
    def go(sym, l, off):
        if l == 0: return [sym]
        for t in range(off, off + 2**l): node[l][t] = sym
        x, y = rng.choice(R[l][sym])
        return go(x, l-1, off) + go(y, l-1, off + 2**(l-1))
    return go(rng.randrange(V), L, 0), node

def build(model):
    X, Y = [], {l: [] for l in range(1, L+1)}
    seqs, nodes = [], []
    for _ in range(a.n_seq):
        s, nd = sample_with_parse(); seqs.append(s); nodes.append(nd)
    ids = torch.tensor(seqs, dtype=torch.long)
    with torch.no_grad():
        h = model(ids, need_hidden=True)["h2"]          # B x T x d
    return h, nodes

def probe(H, lab, tag):
    """linear decode with a held-out split"""
    n = H.shape[0]; k = int(0.7*n)
    idx = torch.randperm(n)
    tr, te = idx[:k], idx[k:]
    w = nn.Linear(H.shape[1], V)
    opt = torch.optim.Adam(w.parameters(), lr=3e-3, weight_decay=1e-4)
    for _ in range(a.epochs):
        opt.zero_grad(); F.cross_entropy(w(H[tr]), lab[tr]).backward(); opt.step()
    with torch.no_grad():
        acc = float((w(H[te]).argmax(-1) == lab[te]).float().mean())
    return acc

torch.manual_seed(0)
models = {}
m = LM2(a.d_model, a.n_heads, nl=a.layers, mlp=True)
sd = torch.load(a.load, map_location="cpu")
# checkpoints trained with --aux-levels carry aux.N.* heads that a plain LM2
# does not define. Load non-strict and keep them aside: the aux head IS the
# model's own answer to "can level N be decoded from h2", so report it too.
aux_sd = {k: v for k, v in sd.items() if k.startswith("aux.")}
missing, unexpected = m.load_state_dict(
    {k: v for k, v in sd.items() if not k.startswith("aux.")}, strict=False)
if missing: print(f"  WARNING missing keys: {missing}")
m.eval()
if aux_sd:
    print(f"  checkpoint carries aux heads for levels "
          f"{sorted({k.split('.')[1] for k in aux_sd})}")
models["trained"] = m
torch.manual_seed(1)
models["random model"] = LM2(a.d_model, a.n_heads, nl=a.layers, mlp=True).eval()

print(f"  {a.load}  L={L} levels, v={V} symbols, chance = {1.0/V:.4f}")
print(f"  decoding the level-l symbol above position t from h2[t], "
      f"{a.n_seq} sequences, 70/30 split")
print(f"\n  {'model':<14}{'level':>7}{'kind':>9}{'acc':>9}{'shuffled':>10}")
for mname, mdl in models.items():
    H, nodes = build(mdl)
    for l in range(1, L+1):
        for kind in ("local", "higher"):
            ts = [t for t in range(SEQ) if KIND[t] == kind]
            feats = torch.cat([H[:, t, :] for t in ts])
            # feats stacks t-blocks: [t0 over all sequences, t1 over all, ...]
            # so labels must iterate t in the OUTER loop to match.
            labs = torch.tensor([nodes[i][l][t] for t in ts
                                 for i in range(H.shape[0])], dtype=torch.long)
            assert len(labs) == len(feats)
            acc = probe(feats, labs, f"{mname} L{l} {kind}")
            sh = labs[torch.randperm(len(labs))]
            accs = probe(feats, sh, "shuffled")
            print(f"  {mname:<14}{l:>7}{kind:>9}{acc:>9.3f}{accs:>10.3f}")
# the trained aux head itself, on freshly sampled data
if aux_sd:
    print("\n  AUX HEADS FROM THE CHECKPOINT (the model's own decoder)")
    H, nodes = build(models["trained"])
    for lv in sorted({int(k.split(".")[1]) for k in aux_sd}):
        w = nn.Linear(a.d_model, V)
        w.weight.data = aux_sd[f"aux.{lv}.weight"]; w.bias.data = aux_sd[f"aux.{lv}.bias"]
        for kind in ("local", "higher"):
            ts = [t for t in range(SEQ) if KIND[t] == kind]
            feats = torch.cat([H[:, t, :] for t in ts])
            labs = torch.tensor([nodes[i][lv][t] for t in ts
                                 for i in range(H.shape[0])], dtype=torch.long)
            with torch.no_grad():
                acc = float((w(feats).argmax(-1) == labs).float().mean())
            print(f"    level {lv} {kind:<7} aux-head acc = {acc:.3f}   "
                  f"(chance {1.0/V:.4f})")
    print("    high here + no loss improvement => the parse state IS represented")
    print("    and the readout cannot use it. At chance => the aux objective")
    print("    never succeeded, so the intervention was never actually applied.")

print(f"\n  acc >> shuffled AND >> random-model  => the parse state is there.")
print(f"  level 1 decodable but levels 3-4 not => the model tracks siblings")
print(f"  and not the hierarchy, which is the local/higher split itself.")
