"""IS THERE A COARSER MANIFOLD ALONG FISHER?

The Fisher sheet is rank 3 out of P and carries -0.75 +- 0.30 of the total
descent at every checkpoint from val 3.66 down to 0.061. But it is estimated
fresh each time, and if it rotates like the update frame does (separation 2.475
-> 2.550 across a 16x window range, i.e. near-maximum) then it has to be
re-estimated constantly.

The question: do the sheets from DIFFERENT times lie inside one larger FIXED
subspace? If six rank-3 sheets span an effective rank of 4 or 5 rather than 18,
that span is a coarser manifold -- static, containing the moving sheet, and
usable without tracking.

    18 vectors spanning ~18    no manifold; each sheet is independent
    18 vectors spanning ~4-5   a coarse manifold exists

Three things measured:
  UNION RANK      stack all sheets, SVD, effective rank at 90% energy, against
                  the null of independently drawn 3-planes
  PAIRWISE        Grassmann distance between sheets at different times, against
                  the maximum sqrt(3) pi/2 = 2.72 and against random 3-planes
  RANK-1 SHARE    how much of each sheet's own descent the FIRST direction
                  carries -- if it is nearly all, the sheet is really rank 1 and
                  the coarse structure is coarser still

The Fisher sheet is built by the randomised range finder on gradient samples,
the same construction the pipeline uses.
"""
import json, subprocess, numpy as np, torch, io, contextlib, math

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
ps = [p for p in model.parameters() if p.requires_grad]
P = sum(p.numel() for p in ps)
opt = torch.optim.AdamW(model.parameters(), lr=LR * 5, betas=(0.9, 0.95),
                        weight_decay=0.1)
K, NPROBE = 3, 24
CKS = [20, 40, 60, 80, 110, 140]


def flat():
    return torch.cat([p.data.reshape(-1) for p in ps]).clone()


def setth(t):
    with torch.no_grad():
        j = 0
        for p in ps:
            q = p.numel(); p.data.copy_(t[j:j + q].view_as(p)); j += q


def gvec():
    return torch.cat([(p.grad.reshape(-1) if p.grad is not None
                       else torch.zeros(p.numel())) for p in ps]).clone()


def fisher_sheet():
    th = flat()
    Gs = []
    for _ in range(NPROBE):
        x, y = get_batch(); model.zero_grad(); _, l = model(x, y); l.backward()
        Gs.append(gvec()); setth(th)
    model.zero_grad()
    Gm = torch.stack(Gs, 1)
    Om = torch.randn(Gm.shape[1], K)
    return torch.linalg.qr(Gm @ Om)[0][:, :K], Gm


def dgr(Q1, Q2):
    sv = torch.linalg.svdvals(Q1.T @ Q2).numpy()
    return float(np.sqrt((np.arccos(np.clip(sv, 0, 1)) ** 2).sum()))


sheets, vals, share1 = [], [], []
step = 0
for ck in CKS:
    while step < ck:
        x, y = get_batch(); _, l = model(x, y); opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); step += 1
    v = float(ev(model, n=6)); model.train()
    Q, Gm = fisher_sheet()
    g = Gm.mean(1)
    # how much of the sheet's captured gradient does direction 1 carry
    c = Q.T @ g
    share1.append(float(c[0] ** 2) / max(float((c * c).sum()), 1e-30))
    sheets.append(Q); vals.append(v)
    print(f"  checkpoint {ck:>4}  val {v:.4f}   dir-1 share of sheet {share1[-1]:.3f}",
          flush=True)

print(f"\n  P={P:,}   {len(sheets)} sheets of rank {K} = {len(sheets)*K} vectors")
S = torch.cat(sheets, 1)
sv = torch.linalg.svdvals(S).numpy()
e = sv ** 2; cum = np.cumsum(e) / e.sum()
r90 = int(np.argmax(cum >= 0.90) + 1)
gen = torch.Generator().manual_seed(5)
RN = [torch.linalg.qr(torch.randn(P, K, generator=gen))[0] for _ in sheets]
svr = torch.linalg.svdvals(torch.cat(RN, 1)).numpy()
er = svr ** 2; cr = np.cumsum(er) / er.sum()
r90r = int(np.argmax(cr >= 0.90) + 1)
print(f"  union effective rank at 90% energy:  {r90}   NULL (random 3-planes) {r90r}")
print(f"  singular values: {np.round(sv/sv[0], 3)}")
print(f"\n  pairwise Grassmann distance between sheets  (max {math.sqrt(K)*math.pi/2:.3f})")
print(f"  {'':>6}" + "".join(f"{c:>8}" for c in CKS))
for i, a in enumerate(CKS):
    row = "".join(f"{dgr(sheets[i], sheets[j]):>8.3f}" if j != i else f"{'-':>8}"
                  for j in range(len(CKS)))
    print(f"  {a:>6}" + row)
rand = np.mean([dgr(RN[0], RN[i]) for i in range(1, len(RN))])
off = [dgr(sheets[i], sheets[j]) for i in range(len(CKS)) for j in range(i + 1, len(CKS))]
print(f"\n  mean pairwise {np.mean(off):.3f}   random 3-planes {rand:.3f}   "
      f"consecutive {np.mean([dgr(sheets[i],sheets[i+1]) for i in range(len(CKS)-1)]):.3f}")
print(f"  direction-1 share of each sheet: mean {np.mean(share1):.3f} "
      f"sd {np.std(share1):.3f}")
json.dump(dict(r90=r90, r90_null=r90r, sv=(sv / sv[0]).tolist(),
               pairwise=off, rand=float(rand), share1=share1, vals=vals),
          open("/home/claude/work/res_coarse.json", "w"), indent=2)
print(f"\n  union rank << {len(sheets)*K}  => a coarser fixed manifold contains all the sheets")
print(f"  union rank ~ {len(sheets)*K}   => each sheet is independent; no coarser structure")
