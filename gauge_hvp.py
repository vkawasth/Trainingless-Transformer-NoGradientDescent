"""KNOWN-ANSWER TEST FOR THE HESSIAN ESTIMATOR, USING THE GAUGE ORBIT.

The loss is exactly constant along the gauge orbit of the attention symmetry
W_Q -> W_Q A, W_K -> W_K A^{-T}, so for every X in gl(d_h) the directional
second derivative vanishes identically:

    v_X = (W_Q X, -W_K X^T)   =>   H v_X = 0,   v_X^T H v_X = 0.

That is a ground truth available with no measurement, and the curvature
estimator used in Phases 37-39 never had one. Its reliability gate returned
ceiling 0.948 with separation 0.062 at B=64 -- which established that it was
unreliable but not whether it was WRONG.

Reported as the batch count B grows:

    ||H v_g||           should be 0
    ||H v_r||           a random direction of MATCHED norm, restricted to the
                        same coordinates, for scale
    ratio               ||H v_g|| / ||H v_r||   -- the figure of merit
    q_g, q_r            v^T H v / ||v||^2 for each

An exact estimator gives ratio 0. A ratio near 1 means a measured curvature of
this magnitude is entirely estimator error. Anything in between quantifies what
fraction of a curvature number is real.

X is drawn at random per head, so this is not a special direction the estimator
might handle by accident -- it is a random direction inside a subspace where
the answer is known.
"""
import io, contextlib, math
import numpy as np, torch

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

torch.manual_seed(1234); np.random.seed(1234)
G = {}; _b = io.StringIO()
with contextlib.redirect_stdout(_b):
    exec(SRC, G)
model = G["model"]; gb = G["get_batch"]; LR = G["LR"] * 5
ps = [p for p in model.parameters() if p.requires_grad]
names = [n for n, p in model.named_parameters() if p.requires_grad]
idx = {n: i for i, n in enumerate(names)}
H = 4

opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95),
                        weight_decay=0.1)
for t in range(1, 201):
    x, y = gb(); _, l = model(x, y)
    opt.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
with torch.no_grad():
    _ev = [gb() for _ in range(6)]
    _v = sum(float(model(a, b)[1]) for a, b in _ev) / len(_ev)
print(f"  checkpoint: step 200, val {_v:.4f}\n")

def gauge_vec(seed):
    g = torch.Generator().manual_seed(seed)
    v = [torch.zeros_like(p) for p in ps]
    for n in names:
        if not n.endswith("WQ.weight"):
            continue
        nk = n.replace("WQ", "WK")
        if nk not in idx:
            continue
        Q = ps[idx[n]].data; K = ps[idx[nk]].data
        D = Q.shape[0]; dh = D // H
        for h in range(H):
            sl = slice(h * dh, (h + 1) * dh)
            X = torch.randn(dh, dh, generator=g)
            v[idx[n]][:, sl] = Q[:, sl] @ X
            v[idx[nk]][:, sl] = -K[:, sl] @ X.T
    return v

nrm = lambda v: math.sqrt(sum(float((a * a).sum()) for a in v))
dot = lambda a, b: sum(float((x * y).sum()) for x, y in zip(a, b))

def rand_like(v, seed):
    g = torch.Generator().manual_seed(seed)
    r = [torch.randn(x.shape, generator=g) if float(x.norm()) > 0
         else torch.zeros_like(x) for x in v]
    s = nrm(v); t = nrm(r)
    return [a * (s / max(t, 1e-30)) for a in r]

def hvp(v, B):
    acc = [torch.zeros_like(p) for p in ps]
    for _ in range(B):
        x, y = gb(); model.zero_grad()
        l = model(x, y)[1]
        gr = torch.autograd.grad(l, ps, create_graph=True)
        d = sum((a * b).sum() for a, b in zip(gr, v))
        hv = torch.autograd.grad(d, ps)
        for i, h_ in enumerate(hv):
            acc[i] += h_.detach()
    return [a / B for a in acc]

vg = gauge_vec(11); vr = rand_like(vg, 12)
print(f"  ||v_gauge|| {nrm(vg):.4f}   ||v_rand|| {nrm(vr):.4f}   (matched)\n")
print(f"  {'B':>4}{'||Hv_g||':>13}{'||Hv_r||':>13}{'ratio':>9}"
      f"{'q_gauge':>13}{'q_rand':>13}")
for B in (1, 4, 16, 64):
    hg = hvp(vg, B); hr = hvp(vr, B)
    ng, nr_ = nrm(hg), nrm(hr)
    qg = dot(vg, hg) / max(nrm(vg) ** 2, 1e-30)
    qr = dot(vr, hr) / max(nrm(vr) ** 2, 1e-30)
    print(f"  {B:>4}{ng:>13.4e}{nr_:>13.4e}{ng / max(nr_, 1e-30):>9.4f}"
          f"{qg:>13.4e}{qr:>13.4e}", flush=True)
print(f"\n  exact answer: ||Hv_g|| = 0 and q_gauge = 0.")
print(f"  ratio -> 0: estimator sound.  ratio -> 1: a curvature of this size")
print(f"  is entirely estimator error.")
