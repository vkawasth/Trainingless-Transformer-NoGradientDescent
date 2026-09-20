#!/usr/bin/env python3
"""NULL_PROBE -- is the order probe evidence about te.weight, or about itself?

    python3 null_probe.py [--load s1.pt] [--data /tmp]

The probe reads comparability off te.weight at 0.98-1.00 on held-out hop pairs.
That was taken as evidence the closure lives in the embedding. But with 48
near-orthogonal vectors in 64 dimensions, ANY assignment of scalar ranks to
tokens is linearly realizable -- the probe can carry the order in its own
weights and never needs the embedding to encode it. Held-out hop pairs reuse
the same 48 tokens, so generalising across pairs is free.

Four matrices, same probe, same classes:
  trained      te.weight from the checkpoint
  random       a fresh init, never trained, no model forward pass
  orthonormal  48 exactly orthogonal rows -- the extreme case
  shuffled     trained te.weight with its ROWS permuted: same geometry,
               wrong token identities

and one destroyed-label control on each: positives/negatives relabelled at
random, which must collapse to chance or the fitting procedure is broken.

If `random` scores like `trained`, the probe measures the ambient geometry,
not a learned representation, and it is retired as evidence.
"""
import json, math, argparse, os, sys
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--load", default="")
ap.add_argument("--data", default="/tmp")
ap.add_argument("--d-model", type=int, default=64)
ap.add_argument("--n-heads", type=int, default=4)
ap.add_argument("--layers", type=int, default=2)
a = ap.parse_args()

sys.argv = [sys.argv[0], "--mlp", "--data", a.data]
_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "rhm15.py")).read()
_src = _src.split("# ============================================================ PHASE 1")[0]
_ns = {"__name__": "_rhm"}
exec(compile(_src, "rhm15.py", "exec"), _ns)
LM2, META = _ns["LM2"], _ns["META"]
_fit_probe, _feat, _hop_classes = _ns["_fit_probe"], _ns["_feat"], _ns["_hop_classes"]
NLEAF, VOCAB = _ns["NLEAF"], _ns["VOCAB"]

P = np.array(META["prec"], dtype=bool)
prod = set()
for v in META["rules_level1"].values():
    for x, y in v: prod.add((int(x), int(y)))
pos = [[x, y] for (x, y) in prod if P[x, y]]
neg = [list(q) for q in META["C_incomp"]] + [[y, x] for x, y in META["C_incomp"]]
cls = _hop_classes()

torch.manual_seed(0)
mats = {}
m0 = LM2(a.d_model, a.n_heads, nl=a.layers, mlp=True)
mats["random init"] = m0.te.weight.detach().clone()
Q, _ = torch.linalg.qr(torch.randn(a.d_model, VOCAB))
mats["orthonormal"] = Q.T[:VOCAB].contiguous()
if a.load:
    m1 = LM2(a.d_model, a.n_heads, nl=a.layers, mlp=True)
    m1.load_state_dict(torch.load(a.load, map_location="cpu"))
    E = m1.te.weight.detach().clone()
    mats["trained (" + os.path.basename(a.load) + ")"] = E
    perm = torch.randperm(E.shape[0])
    mats["trained, rows shuffled"] = E[perm].contiguous()

print(f"  vocab {VOCAB} tokens in d={a.d_model}   "
      f"probe trained on {len(pos)} ordered rules vs {len(neg)} incomparable")
print(f"  {'embedding':<28}{'in-samp':>9}" +
      "".join(f"{str(k)+'-hop':>10}" for k in sorted(cls)) + f"{'RANDLBL':>10}")

for name, E in mats.items():
    w = _fit_probe(E, pos, neg, iters=3000)
    with torch.no_grad():
        ins = float(((w(_feat(E, pos + neg)).squeeze(-1) > 0).float() ==
                     torch.cat([torch.ones(len(pos)), torch.zeros(len(neg))])
                     ).float().mean())
        accs = []
        for k in sorted(cls):
            pr = cls[k]
            accs.append(float((w(_feat(E, pr)).squeeze(-1) > 0).float().mean())
                        if len(pr) >= 8 else float("nan"))
    # destroyed-label control: same data, labels randomised
    g = torch.Generator().manual_seed(7)
    allp = pos + neg
    idx = torch.randperm(len(allp), generator=g)
    p2 = [allp[i] for i in idx[:len(pos)]]; n2 = [allp[i] for i in idx[len(pos):]]
    w2 = _fit_probe(E, p2, n2, iters=3000)
    with torch.no_grad():
        rl = float((w2(_feat(E, cls[min(cls)])).squeeze(-1) > 0).float().mean())
    print(f"  {name:<28}{ins:>9.3f}" +
          "".join(f"{x:>10.3f}" for x in accs) + f"{rl:>10.3f}")

print("\n  RANDLBL = probe fitted on randomised labels, scored on 2-hop pairs;")
print("  it should sit near 0.5 or the fitting procedure itself is suspect.")
print("  If 'random init' matches 'trained', the probe is reading the ambient")
print("  geometry of 48 near-orthogonal vectors, not a learned order, and it")
print("  is retired as evidence about te.weight.")
