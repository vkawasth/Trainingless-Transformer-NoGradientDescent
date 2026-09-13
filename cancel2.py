"""IS THE CANCELLED MOTION NEEDED? A TWO-PASS TEST.

The previous attempt projected each interval's lift onto its OWN net direction.
That is the identity map -- a vector projected on itself is itself -- which is
why the cancellation column read 0.0000 and the shadow received nothing.

Cancellation happens ACROSS intervals, not within one. Writing the trajectory
as a sum of interval lifts L_1..L_N with chord C = sum L_i, the surviving
component of L_i is its projection on C:

    L_i^par = <L_i, u> u ,   u = C/||C||        sum_i L_i^par = C  exactly
    L_i^perp = L_i - L_i^par                    sum_i L_i^perp = 0  exactly

So the perpendicular parts cancel by construction, and the question is whether
they are NEEDED. This requires knowing C, i.e. hindsight, and is therefore a
test of necessity rather than a training method.

PASS 1  train normally, recording each interval's displacement split into
        four lanes kept separate throughout:
            LN    scale direction of LN-fed matrices   (early activity)
            QK    GL(d_h) orbit of (W_Q, W_K)          (late activity)
            LIFT  the functional remainder
            RES   biases, LN gains, embeddings
PASS 2  replay the recorded intervals into fresh shadows under four policies:
            full        every lane, in full          -- must reproduce pass 1
            no_rot      LN and QK lanes dropped
            no_cancel   LIFT replaced by L_i^par
            minimal     both
        evaluating after each interval so the divergence is visible over time.

The 'full' arm is the harness check: if it does not reproduce the real
trajectory to several decimals, nothing else in the table can be read.
"""
import io, contextlib, math, copy
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
named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
H, K, STEPS = 4, 20, 160
LNFED = ("attn.WV.weight", "attn.op.weight", "ff.g.weight", "ff.v.weight",
         "ff.o.weight")
EV = [gb() for _ in range(4)]
ENTRY = copy.deepcopy({n: p.data.clone() for n, p in named})

def vloss(m):
    with torch.no_grad():
        return sum(float(m(a, b)[1]) for a, b in EV) / len(EV)

def qk_gauge_part(Q, Kx, dQ, dK):
    D = Q.shape[0]; dh = D // H
    gQ = torch.zeros_like(dQ); gK = torch.zeros_like(dK)
    for h in range(H):
        s = slice(h * dh, (h + 1) * dh)
        q = Q[:, s].double(); k = Kx[:, s].double()
        A = q.T @ q; B = k.T @ k
        Rm = q.T @ dQ[:, s].double() - (k.T @ dK[:, s].double()).T
        a, U = torch.linalg.eigh(A); b, V = torch.linalg.eigh(B)
        X = U @ ((U.T @ Rm @ V) / (a[:, None] + b[None, :] + 1e-12)) @ V.T
        gQ[:, s] = (q @ X).to(dQ.dtype); gK[:, s] = (-k @ X.T).to(dK.dtype)
    return gQ, gK

# ---------------- PASS 1 ----------------
opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95),
                        weight_decay=0.1)
prev = {n: p.data.clone() for n, p in named}
REC = []
print(f"  PASS 1  training, K={K}\n")
print(f"  {'t':>4}{'val':>9}{'lnE':>8}{'qkE':>8}")
for t in range(1, STEPS + 1):
    x, y = gb(); _, l = model(x, y)
    opt.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    if t % K:
        continue
    cur = {n: p.data.clone() for n, p in named}
    D_ = {n: cur[n] - prev[n] for n, _ in named}
    rot, lift = {}, {}
    lnE = qkE = tot = 0.0
    for n, p in named:
        d = D_[n]; tot += float((d.double() ** 2).sum())
        g = torch.zeros_like(d)
        if n.endswith("attn.WQ.weight"):
            nk = n.replace("WQ", "WK")
            g, gk = qk_gauge_part(prev[n], prev[nk], d, D_[nk])
            qkE += float((g.double() ** 2).sum() + (gk.double() ** 2).sum())
        elif n.endswith("attn.WK.weight"):
            nq = n.replace("WK", "WQ")
            _, g = qk_gauge_part(prev[nq], prev[n], D_[nq], d)
        elif n.endswith(LNFED):
            w = prev[n].double().reshape(-1); dd = d.double().reshape(-1)
            c = float(dd @ w) / max(float(w @ w), 1e-30)
            g = (c * prev[n].double()).to(d.dtype)
            lnE += float((g.double() ** 2).sum())
        rot[n] = g; lift[n] = d - g
    REC.append((rot, lift))
    prev = cur
    print(f"  {t:>4}{vloss(model):>9.4f}{lnE/tot:>8.4f}{qkE/tot:>8.4f}", flush=True)
REAL = vloss(model)

# ---------------- chord and its per-interval projections ----------------
C = {n: sum(r[1][n] for r in REC) for n, _ in named}
cflat = torch.cat([C[n].double().reshape(-1) for n, _ in named])
cn = float(cflat.norm())
par = []
for rot, lift in REC:
    lf = torch.cat([lift[n].double().reshape(-1) for n, _ in named])
    coef = float(lf @ cflat) / max(cn * cn, 1e-30)
    par.append({n: (coef * C[n].double()).to(C[n].dtype) for n, _ in named})
chk = torch.cat([sum(p_[n].double().reshape(-1) for p_ in par)
                 for n, _ in named])
print(f"\n  chord ||C|| {cn:.4f}   sum of parallel parts reproduces it: "
      f"{float((chk - cflat).norm()) / cn:.2e} relative error")
perp = sum(float(((torch.cat([lift[n].double().reshape(-1) for n, _ in named])
                   - torch.cat([p_[n].double().reshape(-1) for n, _ in named]))
                  ** 2).sum()) for (rot, lift), p_ in zip(REC, par))
lifte = sum(float((torch.cat([lift[n].double().reshape(-1)
                              for n, _ in named]) ** 2).sum())
            for rot, lift in REC)
print(f"  cancelled fraction of lift energy: {perp/lifte:.4f}\n")

# ---------------- PASS 2 ----------------
POL = ["full", "no_rot", "no_cancel", "minimal"]
print(f"  PASS 2  replay")
print(f"  {'t':>4}" + "".join(f"{p_:>11}" for p_ in POL) + f"{'real':>10}")
sh = {}
for p_ in POL:
    with torch.no_grad():
        for n, q in named:
            q.data.copy_(ENTRY[n])
    sh[p_] = {n: ENTRY[n].clone() for n, _ in named}
for i, (rot, lift) in enumerate(REC, 1):
    row = []
    for p_ in POL:
        use_rot = p_ in ("full", "no_cancel")
        use_full_lift = p_ in ("full", "no_rot")
        for n, _ in named:
            sh[p_][n] = sh[p_][n] + (rot[n] if use_rot else 0) \
                        + (lift[n] if use_full_lift else par[i - 1][n])
        with torch.no_grad():
            for n, q in named:
                q.data.copy_(sh[p_][n])
        row.append(vloss(model))
    print(f"  {i*K:>4}" + "".join(f"{v:>11.4f}" for v in row)
          + (f"{REAL:>10.4f}" if i == len(REC) else ""), flush=True)
