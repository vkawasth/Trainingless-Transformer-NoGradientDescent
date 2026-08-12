"""IS THE SHEET FORMING, OR IS THE ESTIMATOR UNDERPOWERED?

Every SPLIT measurement asks whether two independent estimates at the same theta
agree. That presumes a fixed object estimated twice. If the sheet is FORMING,
the two disagree partly because the estimator is noisy and partly because there
is no single sheet yet -- and a single SPLIT number cannot separate those.

The discriminator is the TREND at fixed cost:

  formed, noisily measured  ->  SPLIT roughly stationary in training time; the
                                estimator's variance does not care where it is
  forming                   ->  SPLIT RISES; early the sheet is diffuse and any
                                two estimates land differently, later it
                                concentrates and they converge

Second, independent discriminator: the effective dimension of F itself. A
forming object should CONCENTRATE, so the participation ratio of the Fisher
spectrum should fall.

Both measured at eight checkpoints from early to late, at FIXED cost per
checkpoint so the trend is a property of the object rather than of the budget.
A random-subspace null is reported for scale.
"""
import json, subprocess, numpy as np, torch, io, contextlib

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
params = [p for _, p in model.named_parameters()]
P = sum(p.numel() for p in params)
for p in params:
    p.requires_grad_(True)
R = 4
NF = 80
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


def sheet_and_spec(th, seed):
    gen = torch.Generator().manual_seed(seed)
    Y = torch.stack([fvp(th, torch.randn(P, generator=gen)) for _ in range(R)], 1)
    Q = torch.linalg.qr(Y)[0]
    # Rayleigh-Ritz on the sampled sheet: spectrum of Q' F Q
    FQ = torch.stack([fvp(th, Q[:, j]) for j in range(R)], 1)
    T = (Q.T @ FQ).numpy(); T = (T + T.T) / 2
    e = np.abs(np.linalg.eigvalsh(T))
    pr = float(e.sum() ** 2 / (e ** 2).sum()) if e.sum() > 0 else float("nan")
    return Q, pr, e


CKS = [20, 40, 70, 110, 160, 220, 290, 370]
step = 0
rows = []
print(f"  r={R}, {NF} Fisher samples per probe, FIXED cost per checkpoint")
print(f"  random-subspace null for SPLIT: r/P = {R/P:.1e}\n")
print(f"  {'step':>6}{'val':>9}{'SPLIT_F':>10}{'PR(F|sheet)':>13}"
      f"{'lam1/lam_r':>12}{'|g|':>9}")
for ck in CKS:
    while step < ck:
        x, y = get_batch(); _, l = model(x, y); opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); step += 1
    th = flat()
    QA, prA, eA = sheet_and_spec(th, 100 + ck)
    QB, _, _ = sheet_and_spec(th, 900 + ck)
    sp = float((QA.T @ QB).pow(2).sum() / R)
    g = torch.zeros(P)
    for _ in range(10):
        x, y = get_batch(); model.zero_grad(); _, l = model(x, y); l.backward()
        g += gfl(); setth(th)
    model.zero_grad(); g /= 10
    v = float(ev(model, n=5)); model.train()
    rows.append(dict(ck=ck, val=v, split=sp, pr=prA,
                     ratio=float(eA.max() / max(eA.min(), 1e-30)),
                     gn=float(g.norm())))
    print(f"  {ck:>6}{v:>9.4f}{sp:>10.4f}{prA:>13.3f}"
          f"{rows[-1]['ratio']:>12.1f}{rows[-1]['gn']:>9.4f}", flush=True)
    setth(th)
json.dump(rows, open("/home/claude/work/res_forming.json", "w"), indent=2)
s = np.array([r["split"] for r in rows]); p = np.array([r["pr"] for r in rows])
t = np.arange(len(rows), dtype=float)
print(f"\n  trend in SPLIT_F : slope {np.polyfit(t,s,1)[0]:+.4f} per checkpoint, "
      f"corr {np.corrcoef(t,s)[0,1]:+.3f}")
print(f"  trend in PR      : slope {np.polyfit(t,p,1)[0]:+.4f} per checkpoint, "
      f"corr {np.corrcoef(t,p)[0,1]:+.3f}")
print(f"\n  SPLIT rising + PR falling => the sheet is forming")
print(f"  both flat                 => formed object, underpowered estimator")
