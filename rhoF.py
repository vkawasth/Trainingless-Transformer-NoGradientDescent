"""THE FISHER RATIO rho_F ACROSS k, WITH THE CONTROL THAT MATTERS.

Measured: the projection of the update onto a few Fisher directions gives MORE
loss reduction than the full update, in 3 of 4 corpus/step cases. The scalar
form of that finding:

    rho_F = (-g^T P_F dtheta) / (-g^T dtheta)      share of first-order descent
    eta_F = (-g^T P_F dtheta) / ||P_F dtheta||^2   descent per unit energy

rho_F > 1 means the residual is actively cancelling useful descent.

THE TRAP: P_F is estimated FROM gradients, so -g^T P_F dtheta is large partly by
construction. Two controls, both required:
  INDEP   P_F built from batches DISJOINT from the g used to score it
  RANDOM  a random subspace of the same dimension k, scored identically

Swept over k rather than measured at one value, since the k-dependence is what
distinguishes a genuine low-dimensional descent geometry from a projection
artefact that grows smoothly with dimension.
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
KS = [1, 2, 3, 5, 8, 12]
NF = 40
torch.manual_seed(17)
opt = torch.optim.AdamW(model.parameters(), lr=LR * 5, betas=(0.9, 0.95),
                        weight_decay=0.1)


def flat():
    return torch.cat([p.data.flatten() for p in params]).clone()


def setth(t):
    with torch.no_grad():
        i = 0
        for p in params:
            q = p.numel(); p.data.copy_(t[i:i + q].view_as(p)); i += q


def gfl():
    return torch.cat([(p.grad.flatten() if p.grad is not None
                       else torch.zeros(p.numel())) for p in params]).clone()


def fvp(th, v, nb=NF):
    a = torch.zeros(P)
    for _ in range(nb):
        x, y = get_batch(); model.zero_grad(); _, l = model(x, y); l.backward()
        g = gfl(); a += g * float((g * v).sum()); setth(th)
    model.zero_grad(); return a / nb


def sheet(th, kmax, seed):
    gen = torch.Generator().manual_seed(seed)
    Y = torch.stack([fvp(th, torch.randn(P, generator=gen)) for _ in range(kmax)], 1)
    return torch.linalg.qr(Y)[0]


step = 0
rows = []
for W in (140, 260):
    while step < W:
        x, y = get_batch(); _, l = model(x, y); opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); step += 1
    th0 = flat()
    QA = sheet(th0, max(KS), 11)          # sheet A
    QB = sheet(th0, max(KS), 22)          # sheet B, independent draw
    setth(th0)
    # scoring gradient, from batches not used in either sheet
    gs = torch.zeros(P)
    for _ in range(20):
        x, y = get_batch(); model.zero_grad(); _, l = model(x, y); l.backward()
        gs += gfl(); setth(th0)
    gs /= 20; model.zero_grad()
    for _ in range(5):
        x, y = get_batch(); _, l = model(x, y); opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    u = flat() - th0; setth(th0)
    denom = -float((gs * u).sum())
    rg = torch.Generator().manual_seed(99)
    print(f"\n  === step {W}, full first-order descent -g.du = {denom:+.6f} ===")
    print(f"  {'k':>4}{'rho_F(A)':>11}{'rho_F(B indep)':>16}{'rho_rand':>11}"
          f"{'energy':>9}{'eta_F/eta_full':>16}")
    for k in KS:
        out = {}
        for nm, Q in (("A", QA[:, :k]), ("B", QB[:, :k])):
            pr = Q @ (Q.T @ u)
            out[nm] = (-float((gs * pr).sum()) / denom,
                       float((pr * pr).sum()) / float((u * u).sum()))
        R = torch.randn(P, k, generator=rg)
        R = torch.linalg.qr(R)[0]
        prr = R @ (R.T @ u)
        rr = -float((gs * prr).sum()) / denom
        prB = QB[:, :k] @ (QB[:, :k].T @ u)
        etaF = -float((gs * prB).sum()) / max(float((prB * prB).sum()), 1e-30)
        etaU = denom / max(float((u * u).sum()), 1e-30)
        rows.append(dict(step=W, k=k, rhoA=out["A"][0], rhoB=out["B"][0],
                         rhorand=rr, energy=out["B"][1], eta=etaF / max(etaU, 1e-30)))
        print(f"  {k:>4}{out['A'][0]:>11.3f}{out['B'][0]:>16.3f}{rr:>11.5f}"
              f"{out['B'][1]:>9.4f}{etaF / max(etaU, 1e-30):>16.1f}")
    setth(th0 + u)
json.dump(rows, open("/home/claude/work/res_rhoF.json", "w"), indent=2)
print("\n  rho > 1 => the residual cancels useful descent")
print("  rho(B) ~ rho(A) => not an artefact of fitting the sheet to the gradient")
print("  rho(random) ~ k/P => the effect is the Fisher direction, not dimension")
