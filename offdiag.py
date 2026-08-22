"""
DOES THE OFF-DIAGONAL OF F_C BUY ANYTHING? A PARTIAL-OBSERVATION TEST.

Adam stores diag(F_C) as v and discards F_ij. The claim is that the discarded
relational structure is what the strand needs. That is testable as a
SUBSTITUTION scheme, which is also the programme's one remaining open question:

  compute sign(g) exactly on a cheap subset O of coordinates,
  predict sign(g) on the remaining target set T from the relational structure.

  arm A   diagonal disc only:  predict sign(g_i) by sign(m_i).
          This is what Adam has. It is batch-independent, so its accuracy is the
          compatibility law of sec:compatibility.
  arm B   diagonal disc + observed coordinates of the SAME batch, through a
          ridge map learned from F_off. Batch-dependent.
  arm C   observed coordinates only, no disc.
  null    shuffle the observed vector across batches -> destroys the relational
          signal while preserving all marginals.

  B >> A  -> the off-diagonal carries real, usable information; partial
             computation plus relational inference substitutes for part of the
             backward pass.
  B ~ A   -> the discarded off-diagonal does not help predict the strand, and
             the diagonal disc was not lossy in any way that matters here.

Sweep the observed fraction so the cost/benefit curve is visible.
"""
import json, subprocess, numpy as np, torch

subprocess.run(["python3", "/mnt/user-data/uploads/build_corpus.py", "--out", "/tmp",
                "--loops", "300"], check=True, capture_output=True)
SRC = open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
G = {}
exec(SRC[:SRC.find("# \u2500\u2500 PHASE 1")], G)
model = G["model"]; get_batch = G["get_batch"]; LR = G["LR"]
params = [p for _, p in model.named_parameters()]
P = sum(p.numel() for p in params)
NSUB, NB = 6000, 400

torch.manual_seed(17)
opt = torch.optim.AdamW(model.parameters(), lr=LR * 5, betas=(0.9, 0.95), weight_decay=0.1)
for _ in range(200):
    x, y = get_batch(); _, l = model(x, y)
    opt.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
theta = torch.cat([p.data.flatten() for p in params]).clone()
idx = torch.tensor(np.random.default_rng(17).choice(P, NSUB, replace=False))
m = torch.cat([opt.state[p]["exp_avg"].flatten() for p in params])[idx].numpy()
v = torch.cat([opt.state[p]["exp_avg_sq"].flatten() for p in params])[idx].numpy()
r = np.abs(m) / (np.sqrt(v) + 1e-12)
sm = np.sign(m)
print(f"  checkpoint; r50={np.percentile(r,50):.3f} r90={np.percentile(r,90):.3f}")

S = np.zeros((NB, NSUB), dtype=np.float32)
for b in range(NB):
    x, y = get_batch()
    model.zero_grad(); _, l = model(x, y); l.backward()
    S[b] = torch.sign(torch.cat([(p.grad.flatten() if p.grad is not None
            else torch.zeros(p.numel())) for p in params])[idx]).numpy()
    with torch.no_grad():
        i = 0
        for p in params:
            k = p.numel(); p.data.copy_(theta[i:i + k].view_as(p)); i += k
model.zero_grad()
print(f"  swept {NB} batches; mean |offdiag corr| gives the relational signal")

ntr = int(NB * 0.75)
rng = np.random.default_rng(0)
perm = rng.permutation(NSUB)
ALPHAS = np.logspace(-2, 5, 15)


def agree(pred, true):
    return float((np.sign(pred) == true)[true != 0].mean())


print(f"\n{'obs frac':>10}{'|O|':>7}{'A disc':>9}{'B disc+obs':>12}"
      f"{'C obs only':>12}{'null':>9}{'B - A':>9}")
out = []
for frac in [0.02, 0.05, 0.10, 0.25, 0.50]:
    nO = int(NSUB * frac)
    O, T = perm[:nO], perm[nO:]
    Xtr, Xte = S[:ntr][:, O], S[ntr:][:, O]
    Ytr, Yte = S[:ntr][:, T], S[ntr:][:, T]
    # arm A: batch-independent sign(m) on the target set
    A = agree(np.tile(sm[T], (NB - ntr, 1)), Yte)
    # arm B / C: ridge from observed to target, alpha tuned on a val split
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
    nv = max(20, int(ntr * 0.2))
    U, sv, Vt = np.linalg.svd(Ztr[:-nv], full_matrices=False)
    ym = Ytr[:-nv].mean(0)
    Ac = U.T @ (Ytr[:-nv] - ym)
    best = (-1, None)
    for a in ALPHAS:
        d = sv / (sv ** 2 + a)
        sc = agree((Ztr[-nv:] @ Vt.T * d) @ Ac + ym, Ytr[-nv:])
        if sc > best[0]: best = (sc, d)
    d = best[1]
    predC = (Zte @ Vt.T * d) @ Ac + ym
    C = agree(predC, Yte)
    # arm B: add the disc as an offset feature
    predB = predC + np.tile(sm[T], (NB - ntr, 1)) * 0.5
    B = agree(predB, Yte)
    # null: shuffle observed rows across batches
    Zn = Zte[rng.permutation(len(Zte))]
    N = agree((Zn @ Vt.T * d) @ Ac + ym, Yte)
    out.append(dict(frac=frac, nO=nO, A=A, B=B, C=C, null=N))
    print(f"{frac:>10.2f}{nO:>7}{A:>9.4f}{B:>12.4f}{C:>12.4f}{N:>9.4f}{B-A:>+9.4f}")

print("\n  arm A is batch-independent (sign of m). B and C see the same batch.")
print("  null preserves all marginals and destroys only the batch pairing.")
json.dump(out, open("/home/claude/work/res_offdiag.json", "w"), indent=1)
