"""RECONNAISSANCE: DO THE ROLES EXPAND TOGETHER OR INDEPENDENTLY?

Before building GPS, one question decides its design. If every role's effective
dimension expands on the same schedule, a single global trigger suffices. If they
expand independently, each block needs its own frame and its own recalibration.

Measured on the REAL compiler at D=128, P~4.3e6 -- not the tiny replica -- since
that is the object GPS would replace, and the per-block k90 <= 5 finding has
never been tested at this scale.

Five checkpoints, no intervention: 20, 40, 90, 120, 160.
Per role (EMB, LN, FF, W_Q, W_K, W_V, W_O), over a trailing window of updates:
  k90        directions for 90% of that role's update energy
  k90 / n    as a fraction of the block, since blocks differ in size by 60x
  E(top3)    energy in the leading three directions
  cos lag1   directional coherence
  drift      ||mu||^2 / mean||g||^2 restricted to that role -- the leading
             indicator, computed per block rather than globally

Coherence is then read off directly: do the k90 curves move together across
roles, and does each role's k90 track its OWN drift ratio?
"""
import json, subprocess, numpy as np, torch, io, contextlib, re

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
model = G["model"]; get_batch = G["get_batch"]; LR = G["LR"]; ev = G["eval_val"]
named = [(n, p) for n, p in model.named_parameters()]
params = [p for _, p in named]
P = sum(p.numel() for p in params)
for p in params:
    p.requires_grad_(True)


def role(nm):
    if nm.startswith("te") or nm.startswith("pe"):
        return "EMB"
    if "ln" in nm.lower():
        return "LN"
    if ".ff." in nm:
        return "FF"
    for pat, lab in (("WQ", "W_Q"), ("WK", "W_K"), ("WV", "W_V")):
        if pat in nm:
            return lab
    if "WO" in nm or "op" in nm:
        return "W_O"
    return "other"


ROLES = ["EMB", "LN", "FF", "W_Q", "W_K", "W_V", "W_O"]
span = {}; i = 0
for nm, p in named:
    span[nm] = (i, i + p.numel()); i += p.numel()
idx = {r: [] for r in ROLES}
for nm, (a, b) in span.items():
    if role(nm) in idx:
        idx[role(nm)].append(torch.arange(a, b))
idx = {r: torch.cat(v) for r, v in idx.items() if len(v)}
torch.manual_seed(17)
opt = torch.optim.AdamW(model.parameters(), lr=LR * 5, betas=(0.9, 0.95),
                        weight_decay=0.1)


def flat():
    return torch.cat([p.data.flatten() for p in params]).clone()


def setth(t):
    with torch.no_grad():
        j = 0
        for p in params:
            q = p.numel(); p.data.copy_(t[j:j + q].view_as(p)); j += q


def gvec():
    return torch.cat([(p.grad.flatten() if p.grad is not None
                       else torch.zeros(p.numel())) for p in params]).clone()


CKS = [20, 40, 90, 120, 160]
WIN = 18
print(f"  real compiler, D=128, P={P}")
print(f"  block sizes: " + "  ".join(f"{r}:{len(ii)}" for r, ii in idx.items()) + "\n")
hist = {r: [] for r in idx}
step = 0
res = {}
prev = flat()
for ck in CKS:
    while step < ck:
        x, y = get_batch(); _, l = model(x, y); opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); step += 1
        cur = flat(); du = cur - prev; prev = cur
        for r, ii in idx.items():
            hist[r].append(du[ii].clone())
        for r in hist:
            if len(hist[r]) > WIN:
                hist[r].pop(0)
    th = flat()
    # per-role drift ratio from a handful of minibatch gradients
    Gs = []
    for _ in range(12):
        x, y = get_batch(); model.zero_grad(); _, l = model(x, y); l.backward()
        Gs.append(gvec()); setth(th)
    model.zero_grad()
    Gm = torch.stack(Gs, 1)
    v = float(ev(model, n=6)); model.train()
    row = {}
    for r, ii in idx.items():
        A = torch.stack(hist[r], 1)
        s = torch.linalg.svdvals(A).numpy() ** 2
        c = np.cumsum(s) / s.sum()
        k90 = int(np.argmax(c >= 0.90) + 1)
        e3 = float(c[min(2, len(c) - 1)])
        cs = float(np.mean([float((A[:, t] * A[:, t + 1]).sum() /
                                  (A[:, t].norm() * A[:, t + 1].norm() + 1e-30))
                            for t in range(A.shape[1] - 1)]))
        Gr = Gm[ii]
        mu = Gr.mean(1)
        dr = float((mu @ mu) / max(float((Gr * Gr).sum(0).mean()), 1e-30))
        row[r] = dict(k90=k90, frac=k90 / len(ii), e3=e3, cos=cs, drift=dr)
    res[ck] = dict(val=v, roles=row)
    print(f"  === step {ck}, val {v:.4f} ===")
    print(f"  {'role':>6}{'n':>9}{'k90':>6}{'k90/n':>10}{'E(top3)':>10}"
          f"{'cos lag1':>10}{'drift':>9}")
    for r in idx:
        d = row[r]
        print(f"  {r:>6}{len(idx[r]):>9}{d['k90']:>6}{d['frac']:>10.2e}"
              f"{d['e3']:>10.3f}{d['cos']:>10.3f}{d['drift']:>9.4f}", flush=True)
    print()
json.dump({str(k): v for k, v in res.items()},
          open("/home/claude/work/res_recon.json", "w"), indent=2)
print("  === coherence across roles ===")
ks = np.array([[res[c]["roles"][r]["k90"] for r in idx] for c in CKS], dtype=float)
ds = np.array([[res[c]["roles"][r]["drift"] for r in idx] for c in CKS], dtype=float)
rs = list(idx)
print(f"  {'role':>6}" + "".join(f"{c:>7}" for c in CKS) + "   k90 trajectory")
for j, r in enumerate(rs):
    print(f"  {r:>6}" + "".join(f"{int(ks[i,j]):>7}" for i in range(len(CKS))))
cm = np.corrcoef(ks.T)
off = cm[~np.eye(len(rs), dtype=bool)]
print(f"\n  mean pairwise corr of k90 trajectories across roles: {np.nanmean(off):+.3f}")
print(f"  (near 1 => roles expand together, one global trigger suffices)")
print(f"  (near 0 => independent, each block needs its own frame)")
pc = [float(np.corrcoef(ks[:, j], ds[:, j])[0, 1]) for j in range(len(rs))]
print(f"  corr(k90, own drift) per role: "
      + "  ".join(f"{r}:{c:+.2f}" for r, c in zip(rs, pc)))
