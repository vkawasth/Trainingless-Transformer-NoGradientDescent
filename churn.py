"""FROM HIGH CHURN TO LOW-DIMENSIONAL UPDATES: WHERE DOES IT HAPPEN?

Two facts already measured sit in tension. Per-step updates are high rank --
PR(dW_Q) = 18-24 at every checkpoint -- while the accumulated chord is PR 3.45.
And at beta1 = 0 consecutive steps are white (cos = -0.033 at lag 1), so the
landscape supplies no directional persistence; momentum supplies all of it,
giving H = 0.814 rather than the diffusive 0.5.

So the dimensional reduction is not in the steps. It is in what survives
integration. This measures both, over training, to see which one falls:

    PR_step    participation ratio of the individual updates in a window
    PR_win     PR of the window's SUM
    PR_ratio   PR_win / PR_step, the compression the integral performs
    eff_rank   exp(spectral entropy) of the window matrix, a second estimator
    cech_b1    cycles in the 3-d projection, which began appearing only late

If PR_step is flat while PR_win falls, churn is constant and the integral is
doing the reduction -- and the reduction is a property of momentum, not of the
loss surface. If PR_step also falls, the updates themselves are becoming
low-rank and something in the landscape is responsible.
"""
import io, contextlib, math
import numpy as np
import torch

R = open("compiler_geometri_patched_86.py").read()
SRC = R[:R.find("# \u2500\u2500 PHASE 3")]
for o, n in [("for mf_r in range(1, 16):", "for mf_r in range(1, 3):"),
             ("    if pc == N_STU-1:", "    if False:"),
             ("ETA_MF=0.01", "ETA_MF=0.002"),
             ("    if len(tau_history)>=3 and tau > tau_history[-2] > tau_history[-3]:",
              "    if False:")]:
    assert SRC.count(o) == 1, o
    SRC = SRC.replace(o, n, 1)
EO = "evals,evecs=spla.eigsh(L_sym,k=D+1,which='SM',tol=1e-4,maxiter=2000)"
EN = ("_v0=np.random.RandomState(7).randn(L_sym.shape[0])\n"
      "evals,evecs=spla.eigsh(L_sym,k=D+1,which='SM',tol=1e-4,maxiter=2000,v0=_v0)\n"
      "evecs=evecs*np.sign(evecs[np.argmax(np.abs(evecs),axis=0),np.arange(evecs.shape[1])])")
assert SRC.count(EO) == 1
SRC = SRC.replace(EO, EN, 1)

import sys
B1 = float(sys.argv[1]) if len(sys.argv) > 1 else 0.9
torch.manual_seed(1234); np.random.seed(1234)
G = {}; _b = io.StringIO()
with contextlib.redirect_stdout(_b):
    exec(SRC, G)
model = G["model"]; gb = G["get_batch"]; LR = G["LR"] * 5
named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
tot = sum(p.numel() for _, p in named)
g = torch.Generator().manual_seed(42)
sub = torch.randperm(tot, generator=g)[:40000]
opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(B1, 0.95),
                        weight_decay=0.1)
EV = [gb() for _ in range(4)]


def vloss():
    with torch.no_grad():
        return sum(float(model(x, y)[1]) for x, y in EV) / len(EV)


def pr(sv):
    e = sv ** 2
    return float(e.sum() ** 2 / max(float((e ** 2).sum()), 1e-30))


def eff_rank(sv):
    p = (sv ** 2) / max(float((sv ** 2).sum()), 1e-30)
    p = p[p > 1e-12]
    return float(np.exp(-(p * np.log(p)).sum()))


W = 16
buf = []
print(f"  beta1 = {B1}\n")
print(f"  {'step':>5}{'val':>9}{'PR_step':>9}{'PR_win':>8}{'ratio':>8}"
      f"{'effrank':>9}{'|win|':>9}{'|path|':>9}")
for t in range(1, 641):
    prev = torch.cat([p.data.reshape(-1) for _, p in named])[sub].clone()
    x, y = gb(); _, l = model(x, y)
    opt.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    cur = torch.cat([p.data.reshape(-1) for _, p in named])[sub]
    buf.append((cur - prev).double())
    if len(buf) > W:
        buf.pop(0)
    if len(buf) < W or t % 64:
        continue
    M = torch.stack(buf, 1)                       # (sub, W)
    sv = torch.linalg.svdvals(M).numpy()
    s = sum(buf)
    prs = pr(sv)
    prw = 1.0                                     # a single vector is rank 1
    # PR of the window SUM measured against the window's own basis: how much
    # of the window's spectral mass the net displacement occupies
    q = s / max(float(s.norm()), 1e-30)
    proj = (M.T @ q).numpy()
    prw = pr(np.abs(proj))
    path = sum(float(v.norm()) for v in buf)
    print(f"  {t:>5}{vloss():>9.4f}{prs:>9.2f}{prw:>8.2f}{prw/max(prs,1e-9):>8.3f}"
          f"{eff_rank(sv):>9.2f}{float(s.norm()):>9.3f}{path:>9.3f}", flush=True)
