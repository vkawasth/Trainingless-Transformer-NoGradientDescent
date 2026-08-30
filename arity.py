"""WHEN DOES EACH CONTEXT LENGTH START PAYING?

The corpus is ONE 1364-token sequence repeated 270x (val is the same sequence).
So there is no distribution over instances and no relations between objects.
What is measurable is MEMORISATION ORDER BY N-GRAM LENGTH.

    L_n(t) = loss predicting x_{i+1} given the n tokens before it
    G_n(t) = L_{n-1}(t) - L_n(t)      what the n-th token back is worth
    tau_n  = first checkpoint where G_n crosses threshold and stays

POSITION CONTROL. Truncating context shifts absolute positions, which would
confound context length with the positional embedding. Instead the probe window
is fixed at length W and the target always sits at position W; shorter contexts
drop the EARLIEST tokens, so every retained token keeps its absolute position.

NULL. The same measurement on shuffled probe windows: same tokens, same
positions, order destroyed. G_n(real) >> G_n(shuf) is the requirement.
"""
import io, contextlib, subprocess, sys, math, json
import numpy as np, torch

subprocess.run([sys.executable, "build_corpus.py", "--out", "/tmp", "--loops", "300"],
               check=True, capture_output=True)

RAW = open("compiler_geometri_patched_86.py").read()
SRC = RAW[:RAW.find("# \u2500\u2500 PHASE 3")]
for old, new in [("D=256; N_HEADS=4", "D=128; N_HEADS=4"),
                 ("for mf_r in range(1, 16):", "for mf_r in range(1, 3):"),
                 ("    if pc == N_STU-1:", "    if False:"),
                 ("    if len(tau_history)>=3 and tau > tau_history[-2] > tau_history[-3]:",
                  "    if False:")]:
    assert SRC.count(old) == 1, f"prelude anchor {old!r}"
    SRC = SRC.replace(old, new, 1)
EIG_OLD = "evals,evecs=spla.eigsh(L_sym,k=D+1,which='SM',tol=1e-4,maxiter=2000)"
EIG_NEW = ("_v0=np.random.RandomState(7).randn(L_sym.shape[0])\n"
           "evals,evecs=spla.eigsh(L_sym,k=D+1,which='SM',tol=1e-4,maxiter=2000,v0=_v0)\n"
           "evecs=evecs*np.sign(evecs[np.argmax(np.abs(evecs),axis=0),np.arange(evecs.shape[1])])")
assert SRC.count(EIG_OLD) == 1
SRC = SRC.replace(EIG_OLD, EIG_NEW, 1)
assert "v0=_v0" in SRC

torch.manual_seed(1234); np.random.seed(1234)
G = {}; _b = io.StringIO()
with contextlib.redirect_stdout(_b): exec(SRC, G)
model = G["model"]; get_batch = G["get_batch"]; LR = G["LR"]
VOCAB = G["VOCAB"]

ids = json.load(open("/tmp/train_ids.json"))
BASE = ids[:1364]
W = 8                      # probe window: target sits at position W
NPROBE = 256
rng = np.random.default_rng(7)
starts = rng.choice(len(BASE) - W - 1, size=NPROBE, replace=False)
PROBE = torch.tensor([[BASE[s + j] for j in range(W + 1)] for s in starts])   # (N, W+1)
# shuffled null: permute the CONTEXT tokens within each window, keep target
_p = torch.stack([torch.randperm(W, generator=torch.Generator().manual_seed(int(s)))
                  for s in starts])
PROBE_SH = PROBE.clone()
for i in range(NPROBE):
    PROBE_SH[i, :W] = PROBE[i, :W][_p[i]]

NS = [1, 2, 3, 5, 7]

@torch.no_grad()
def Ln(probe, n):
    """loss on the target given the n tokens immediately before it,
    with absolute positions preserved (target always at index W)."""
    x = probe[:, W - n:W]                      # (N, n) real context
    y = probe[:, W]
    pos = torch.arange(W - n, W)
    h = model.te(x) + model.pe(pos)
    for blk in model.blocks: h = blk(h)
    lg = model.head(model.ln_f(h))[:, -1, :]
    return float(torch.nn.functional.cross_entropy(lg, y))

CKPTS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160]
opt = torch.optim.AdamW(model.parameters(), lr=LR*5, betas=(0.9, 0.95), weight_decay=0.1)
rows = []
step = 0
for ck in CKPTS:
    while step < ck:
        x, y = get_batch(); _, l = model(x, y)
        opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); step += 1
    r = dict(step=ck,
             real={n: Ln(PROBE, n) for n in NS},
             shuf={n: Ln(PROBE_SH, n) for n in NS})
    rows.append(r)
    print(f"  step {ck:>3} done", flush=True)

def gains(d):
    return {NS[i]: d[NS[i-1]] - d[NS[i]] for i in range(1, len(NS))}

print(f"\n  L_n, real probes (target position fixed at {W})\n")
print(f"  {'step':>5}" + "".join(f"{'L'+str(n):>8}" for n in NS)
      + "   |" + "".join(f"{'G'+str(n):>8}" for n in NS[1:]))
for r in rows:
    g = gains(r["real"])
    print(f"  {r['step']:>5}" + "".join(f"{r['real'][n]:>8.3f}" for n in NS)
          + "   |" + "".join(f"{g[n]:>8.3f}" for n in NS[1:]))

print(f"\n  shuffled-context null (same tokens, same positions, order destroyed)\n")
print(f"  {'step':>5}" + "".join(f"{'G'+str(n):>8}" for n in NS[1:]))
for r in rows:
    g = gains(r["shuf"])
    print(f"  {r['step']:>5}" + "".join(f"{g[n]:>8.3f}" for n in NS[1:]))

THR = 0.05
print(f"\n  onset tau_n: first step with G_n > {THR} that never falls back below\n")
for n in NS[1:]:
    seq = [(r["step"], gains(r["real"])[n]) for r in rows]
    tau = None
    for i, (s, v) in enumerate(seq):
        if v > THR and all(w > THR for _, w in seq[i:]):
            tau = s; break
    print(f"    tau_{n} = {tau}")
json.dump(rows, open("res_arity.json", "w"), indent=1)
