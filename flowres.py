"""EXACT FLOW RESIDUAL ON A FIXED BATCH.

The residual has resisted every explanation: not gauge (2.4%), not curvature
(99% left after projecting out Hg), not dimensional (P(32)/P(3)=1.18), not the
nonlinearities (survives softmax and LayerNorm softening). Every one of those
measurements used stochastic minibatches, so sampling noise was never excluded.

This fixes the batch. L(theta) is then deterministic and the true gradient flow
dtheta/dt = -grad L is well defined, so the ground truth can be integrated with
RK4 and compared step by step against the discrete optimiser:

    R_t = dtheta_disc - dtheta_flow

decomposed along and across g_t. Three hypotheses, distinguished by behaviour:

  ||R|| -> 0 as dt -> 0        discretisation artefact
  R_perp dominates, isotropic  thermal noise on the loss surface
  ||R|| persists with lag-1    intrinsic geometric holonomy
  correlation

Step sizes swept over a decade so the dt -> 0 branch can be read from the slope.
The lag sweep on corr(R_t, R_t+k) is included because eta = 0.94 was measured
with stochastic batches and has not had the control that killed the defect
result -- a decay curve is memory, an isolated k=1 spike is construction.
"""
import json, subprocess, numpy as np, torch, gc, io, contextlib

subprocess.run(["python3", "/mnt/user-data/uploads/build_corpus.py", "--out", "/tmp",
                "--loops", "300"], check=True, capture_output=True)
RAW = open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
CUT = RAW.find("# \u2500\u2500 PHASE 3")
src = RAW[:CUT].replace("D=256; N_HEADS=4", "D=128; N_HEADS=4", 1)
src = src.replace("for mf_r in range(1, 16):", "for mf_r in range(1, 3):", 1)
src = src.replace("    if pc == N_STU-1:", "    if False:", 1)
src = src.replace("    if len(tau_history)>=3 and tau > tau_history[-2] > tau_history[-3]:",
                  "    if False:", 1)
G = {}
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    exec(src, G)
model = G["model"]; get_batch = G["get_batch"]; LR = G["LR"]
params = [p for _, p in model.named_parameters()]
P = sum(p.numel() for p in params)
for p in params:
    p.requires_grad_(True)

FX, FY = get_batch()          # THE fixed batch: the loss surface is now static


def flat():
    return torch.cat([p.data.flatten() for p in params]).clone()


def setth(t):
    with torch.no_grad():
        i = 0
        for p in params:
            q = p.numel(); p.data.copy_(t[i:i + q].view_as(p)); i += q


def grad_at(th):
    setth(th); model.zero_grad()
    _, l = model(FX, FY); l.backward()
    g = torch.cat([(p.grad.flatten() if p.grad is not None
                    else torch.zeros(p.numel())) for p in params]).clone()
    model.zero_grad(); return g, float(l)


def rk4(th, h):
    k1, _ = grad_at(th)
    k2, _ = grad_at(th - 0.5 * h * k1)
    k3, _ = grad_at(th - 0.5 * h * k2)
    k4, _ = grad_at(th - h * k3)
    return -(h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


# warm the model to a non-trivial point using the fixed batch
th = flat()
for _ in range(60):
    g, _ = grad_at(th); th = th - 0.05 * g / (g.norm() + 1e-12)
setth(th)
g0, l0 = grad_at(th)
print(f"  fixed batch, warmed: loss {l0:.5f}, |g| {g0.norm():.5f}\n")
print(f"  === Euler vs RK4 on the SAME flow: does R vanish as h -> 0? ===")
print(f"  {'h':>10}{'|R|/|dth|':>12}{'R_par frac':>12}{'R_perp frac':>13}{'slope':>9}")
prev = None
for h in (1e-1, 3e-2, 1e-2, 3e-3, 1e-3):
    g, _ = grad_at(th)
    du_e = -h * g
    du_r = rk4(th, h)
    R = du_e - du_r
    gn2 = max(float((g * g).sum()), 1e-30)
    par = float((R * g).sum()) ** 2 / gn2
    tot = max(float((R * R).sum()), 1e-30)
    rel = float(R.norm() / (du_r.norm() + 1e-30))
    sl = (np.log(rel) - np.log(prev[1])) / (np.log(h) - np.log(prev[0])) if prev else float('nan')
    print(f"  {h:>10.0e}{rel:>12.3e}{par/tot:>12.4f}{1-par/tot:>13.4f}{sl:>9.2f}")
    prev = (h, rel)
print(f"  (slope ~1 => first-order Euler error, the expected discretisation law)")

print(f"\n  === AdamW vs the true flow, at the actual training step size ===")
torch.manual_seed(17)
setth(th)
opt = torch.optim.AdamW(model.parameters(), lr=LR * 5, betas=(0.9, 0.95),
                        weight_decay=0.1)
Rs = []; rows = []
for t in range(80):
    th_t = flat()
    g, lv = grad_at(th_t)
    setth(th_t)
    opt.zero_grad()
    _, l = model(FX, FY); l.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    du_d = flat() - th_t
    hh = float(du_d.norm() / (g.norm() + 1e-30))     # matched arclength
    du_f = rk4(th_t, hh)
    setth(th_t + du_d)
    R = du_d - du_f
    gn2 = max(float((g * g).sum()), 1e-30)
    par = float((R * g).sum()) ** 2 / gn2
    tot = max(float((R * R).sum()), 1e-30)
    Rs.append((R / (R.norm() + 1e-30)).clone())
    rows.append(dict(t=t, loss=lv, rel=float(R.norm() / (du_d.norm() + 1e-30)),
                     parf=par / tot,
                     cosdf=float((du_d * du_f).sum() / (du_d.norm() * du_f.norm() + 1e-30))))
m = lambda k: float(np.mean([r[k] for r in rows[20:]]))
print(f"    |R|/|dtheta_disc|   {m('rel'):.4f}")
print(f"    R_parallel fraction {m('parf'):.4f}   R_perp {1-m('parf'):.4f}")
print(f"    cos(dtheta_disc, dtheta_flow) {m('cosdf'):+.4f}")
print(f"    loss {rows[0]['loss']:.5f} -> {rows[-1]['loss']:.5f}")
print(f"\n  === lag sweep on the residual direction ===")
print(f"  {'k':>4}{'corr':>10}")
for k in range(1, 9):
    c = [float((Rs[i] * Rs[i + k]).sum()) for i in range(len(Rs) - k) if i >= 20]
    print(f"  {k:>4}{np.mean(c):>10.4f}")
json.dump(rows, open("/home/claude/work/res_flow.json", "w"), indent=2)
print(f"\n  decay curve => memory;  isolated k=1 => construction artefact")
