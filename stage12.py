"""STAGE 1 (pin MF) + STAGE 2 (calibrate SPLIT).

Nothing downstream is interpretable until these two are settled.

STAGE 1. The MF pump stops adaptively -- on Phi_clean = N_STU-1, or on tau
rising twice. Measured round counts have varied 1,2,3,4,9 across runs at the
same capacity, and two Krylov runs at D=128 gave n_neg = 3-4 and n_neg = 0 with
MF3 and MF1 respectively. Since MF injects the energy the compiler works with,
it sets the starting geometry, so it is a live variable in every comparison made
so far. Here the loop bound is overridden to a FIXED count and the resulting
n_neg is measured as a function of that count -- which both pins the variable and
tests whether it explains the discrepancy.

STAGE 2. Two INDEPENDENT Krylov spaces at the SAME theta gave overlap 0.301 and
0.315 at m=6 with 2 HVP batches, while consecutive-checkpoint overlap ran
0.003-0.35. The instrument therefore cannot resolve transport at all: the signal
sits at or below its own floor. Sweep (m, n_hvp, n_drift) and find where SPLIT
becomes high enough that a transport measurement has room to be meaningful.
"""
import json, subprocess, numpy as np, torch, gc, io, contextlib, re

subprocess.run(["python3", "/mnt/user-data/uploads/build_corpus.py", "--out", "/tmp",
                "--loops", "300"], check=True, capture_output=True)
RAW = open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
CUT = RAW.find("# \u2500\u2500 PHASE 3")


def build(D, mf_fixed):
    """Compile with the MF pump pinned to exactly mf_fixed rounds."""
    src = RAW[:CUT].replace("D=256; N_HEADS=4", "D=%d; N_HEADS=4" % D, 1)
    src = src.replace("for mf_r in range(1, 16):",
                      "for mf_r in range(1, %d):" % (mf_fixed + 1), 1)
    # disable both adaptive stops so the count is exactly mf_fixed
    src = src.replace("    if pc == N_STU-1:", "    if False:", 1)
    src = src.replace("    if len(tau_history)>=3 and tau > tau_history[-2] > tau_history[-3]:",
                      "    if False:", 1)
    G = {}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(src, G)
    log = buf.getvalue()
    used = re.search(r"After MF(\d+): val=([\d.]+)", log)
    return G, (int(used.group(1)), float(used.group(2))) if used else (None, None)


def krylov_tools(G):
    model = G["model"]; get_batch = G["get_batch"]
    params = [p for _, p in model.named_parameters()]
    P = sum(p.numel() for p in params)
    for p in params:
        p.requires_grad_(True)

    def flat():
        return torch.cat([p.data.flatten() for p in params]).clone()

    def setth(t):
        with torch.no_grad():
            i = 0
            for p in params:
                k = p.numel(); p.data.copy_(t[i:i + k].view_as(p)); i += k

    def gfl():
        return torch.cat([(p.grad.flatten() if p.grad is not None
                           else torch.zeros(p.numel())) for p in params]).clone()

    def drift(th, n):
        a = torch.zeros(P)
        for _ in range(n):
            x, y = get_batch(); model.zero_grad(); _, l = model(x, y); l.backward()
            a += gfl(); setth(th)
        model.zero_grad(); return a / n

    def hvp(th, v, nb):
        a = torch.zeros(P)
        for _ in range(nb):
            x, y = get_batch(); model.zero_grad(); _, loss = model(x, y)
            gr = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
            gr = [t if t is not None else torch.zeros_like(p) for t, p in zip(gr, params)]
            g2 = torch.cat([t.flatten() for t in gr])
            hv = torch.autograd.grad((g2 * v).sum(), params, allow_unused=True)
            hv = [t if t is not None else torch.zeros_like(p) for t, p in zip(hv, params)]
            a += torch.cat([t.flatten() for t in hv]).detach(); setth(th)
        model.zero_grad(); return a / nb

    def kry(th, m, nb, nd):
        g = drift(th, nd); V = [g / (g.norm() + 1e-30)]
        for _ in range(m - 1):
            w = hvp(th, V[-1], nb)
            for u in V:
                w = w - float((w * u).sum()) * u
            if float(w.norm()) < 1e-12:
                break
            V.append(w / w.norm())
        Q = torch.stack(V, 1)
        HQ = torch.stack([hvp(th, Q[:, j], nb) for j in range(Q.shape[1])], 1)
        T = (Q.T @ HQ).numpy(); T = (T + T.T) / 2
        return Q, T

    return model, params, P, flat, setth, kry


def ov(A, B):
    k = min(A.shape[1], B.shape[1])
    return float((A[:, :k].T @ B[:, :k]).pow(2).sum() / k)


print("=== STAGE 1: n_neg as a function of PINNED MF rounds, D=128 ===")
print(f"  {'MF':>4}{'handoff val':>13}{'val@30':>9}{'n_neg':>7}{'eig(T)':>46}")
s1 = []
for mfr in (1, 2, 3, 4):
    G, used = build(128, mfr)
    model, params, P, flat, setth, kry = krylov_tools(G)
    LR = G["LR"]; ev = G["eval_val"]; get_batch = G["get_batch"]
    torch.manual_seed(17)
    opt = torch.optim.AdamW(model.parameters(), lr=LR * 5, betas=(0.9, 0.95),
                            weight_decay=0.1)
    for _ in range(30):
        x, y = get_batch(); _, l = model(x, y); opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    th = flat()
    Q, T = kry(th, 6, 2, 16)
    e = np.linalg.eigvalsh(T)
    v30 = float(ev(model, n=5)); model.train()
    s1.append(dict(mf=mfr, used=used, val30=v30, nneg=int((e < 0).sum()),
                   eig=[float(x) for x in e]))
    print(f"  {mfr:>4}{(used[1] if used else float('nan')):>13.4f}{v30:>9.4f}"
          f"{int((e<0).sum()):>7}  {' '.join(f'{x:>6.3f}' for x in e)}", flush=True)
    del G, model, params
    gc.collect()
json.dump(s1, open("/home/claude/work/res_stage1_mf.json", "w"), indent=2)

print("\n=== STAGE 2: SPLIT floor vs estimator resources, D=128, MF pinned to 2 ===")
print(f"  {'m':>4}{'nb':>4}{'ndrift':>8}{'SPLIT':>9}{'cost(HVP)':>11}")
G, used = build(128, 2)
model, params, P, flat, setth, kry = krylov_tools(G)
LR = G["LR"]; get_batch = G["get_batch"]
torch.manual_seed(17)
opt = torch.optim.AdamW(model.parameters(), lr=LR * 5, betas=(0.9, 0.95),
                        weight_decay=0.1)
for _ in range(30):
    x, y = get_batch(); _, l = model(x, y); opt.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
th = flat()
s2 = []
for (m, nb, nd) in [(6, 2, 16), (6, 6, 40), (4, 8, 60), (3, 12, 80), (2, 16, 100)]:
    A, _ = kry(th, m, nb, nd)
    B, _ = kry(th, m, nb, nd)
    sp = ov(A, B)
    cost = (2 * m - 1) * nb + nd
    s2.append(dict(m=m, nb=nb, nd=nd, split=sp, cost=cost))
    print(f"  {m:>4}{nb:>4}{nd:>8}{sp:>9.4f}{cost:>11}", flush=True)
    json.dump(s2, open("/home/claude/work/res_stage2_split.json", "w"), indent=2)
print("\n  SPLIT is the ceiling for any transport measurement: consecutive-checkpoint")
print("  overlap must be read against it, and cannot meaningfully exceed it.")
