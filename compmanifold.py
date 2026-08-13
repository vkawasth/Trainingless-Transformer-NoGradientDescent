"""DOES EACH COMPONENT TRAVEL IN ITS OWN CONSISTENT DIRECTION?

The claim to test: for each of EMB, LN, FF, W_Q, W_K, W_V, W_O, is ~90% of the
component's motion captured by a LOW-DIMENSIONAL matrix manifold -- a fixed
subspace of its own parameter block that the trajectory stays inside?

Two distinct questions, measured separately because they can disagree:

  DIRECTION   cos(du_i(t), du_i(t')) across checkpoints. Does the component keep
              moving the same way, or does it turn?
  SUBSPACE    cumulative energy of the component's update sequence in its own
              top-k singular directions. If k=3 captures 90%, the component's
              flow lives on a 3-dimensional manifold inside its block.

The subspace question is the one that matters for a matrix-manifold description,
and it is NOT implied by the direction question: a trajectory can turn steadily
while staying in a plane.

Reported per component, with the k needed for 50/90% energy, against a random
null of the same block size and sequence length -- for T updates in a block of
size n, random directions need ~0.9*T components, so the null is the ceiling.

Run at P=3672 where blocks are small enough that the SVDs are exact, and at
three windows spanning the regimes: 20-60 (early), 100-160 (mid), 240-300 (late).
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn, math, json
torch.manual_seed(0)
V, D, L, T = 40, 12, 2, 16


class Blk(nn.Module):
    def __init__(s):
        super().__init__()
        s.n1 = nn.LayerNorm(D); s.n2 = nn.LayerNorm(D)
        s.q = nn.Linear(D, D, bias=False); s.k = nn.Linear(D, D, bias=False)
        s.v = nn.Linear(D, D, bias=False); s.o = nn.Linear(D, D, bias=False)
        s.g = nn.Linear(D, 2 * D, bias=False); s.u = nn.Linear(D, 2 * D, bias=False)
        s.w = nn.Linear(2 * D, D, bias=False)

    def forward(s, x):
        h = s.n1(x); q, k, v = s.q(h), s.k(h), s.v(h)
        a = Fn.softmax(q @ k.transpose(-2, -1) / math.sqrt(D)
                       + torch.triu(torch.full((x.shape[1], x.shape[1]), -1e9), 1), dim=-1)
        x = x + s.o(a @ v); h = s.n2(x)
        return x + s.w(Fn.silu(s.g(h)) * s.u(h))


class M(nn.Module):
    def __init__(s):
        super().__init__()
        s.te = nn.Embedding(V, D); s.pe = nn.Embedding(T, D)
        s.b = nn.ModuleList([Blk() for _ in range(L)]); s.nf = nn.LayerNorm(D)

    def forward(s, x, y):
        h = s.te(x) + s.pe(torch.arange(x.shape[1]))
        for b in s.b:
            h = b(h)
        lo = s.nf(h) @ s.te.weight.T
        return lo, Fn.cross_entropy(lo.reshape(-1, V), y.reshape(-1))


rg = np.random.default_rng(1)
rules = {i: [(i * 3 + 1) % V, (i * 5 + 2) % V] for i in range(V)}
s_ = [0]
for _ in range(6000):
    c = s_[-1]
    s_.append(int(rules[c][0] if (len(s_) > 1 and s_[-2] % 2 == 0) else rules[c][1]))
seq = np.array(s_)


def batch(n=24):
    i = rg.integers(0, len(seq) - T - 1, n)
    return (torch.tensor(np.stack([seq[j:j + T] for j in i])),
            torch.tensor(np.stack([seq[j + 1:j + T + 1] for j in i])))


m = M(); named = [(n, p) for n, p in m.named_parameters()]
ps = [p for _, p in named]
P = sum(p.numel() for p in ps)


def role(nm):
    if nm.startswith("te") or nm.startswith("pe"):
        return "EMB"
    if "n1" in nm or "n2" in nm or nm.startswith("nf"):
        return "LN"
    for r, lab in (("\.q\.", "W_Q"), ("\.k\.", "W_K"), ("\.v\.", "W_V"),
                   ("\.o\.", "W_O")):
        import re
        if re.search(r, nm):
            return lab
    if ".g." in nm or ".u." in nm or ".w." in nm:
        return "FF"
    return "other"


ROLES = ["EMB", "LN", "W_Q", "W_K", "W_V", "W_O", "FF"]
span = {}; i = 0
for nm, p in named:
    span[nm] = (i, i + p.numel()); i += p.numel()
idx = {r: [] for r in ROLES}
for nm, (a, b) in span.items():
    r = role(nm)
    if r in idx:
        idx[r].append(np.arange(a, b))
idx = {r: np.concatenate(v) for r, v in idx.items() if v}
opt = torch.optim.AdamW(m.parameters(), lr=3e-3, betas=(0.9, 0.95), weight_decay=0.1)


def flat():
    return torch.cat([p.data.flatten() for p in ps]).clone()


WINDOWS = [(20, 60, "early"), (100, 160, "mid"), (240, 300, "late")]
step = 0; prev = flat(); U = {r: [] for r in idx}; marks = {}
allck = sorted({c for w in WINDOWS for c in range(w[0], w[1] + 1)})
for ck in range(1, 301):
    x, y = batch(); _, l = m(x, y); opt.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); step += 1
    cur = flat(); du = (cur - prev).numpy(); prev = cur
    for r, ii in idx.items():
        U[r].append(du[ii])
    marks[ck] = True


def energy_k(Mx, frac):
    s = np.linalg.svd(Mx, compute_uv=False) ** 2
    c = np.cumsum(s) / s.sum()
    return int(np.argmax(c >= frac) + 1), float(c[min(2, len(c) - 1)])


print(f"  P={P}, exact SVDs per block\n")
for lo, hi, lab in WINDOWS:
    print(f"  === {lab} (steps {lo}-{hi}, {hi-lo+1} updates) ===")
    print(f"  {'role':>6}{'block n':>9}{'k for 50%':>11}{'k for 90%':>11}"
          f"{'E(top3)':>10}{'null k90':>10}{'cos lag1':>10}")
    for r, ii in idx.items():
        A = np.stack(U[r][lo - 1:hi], 1)          # n x Tsteps
        k50, _ = energy_k(A, 0.50)
        k90, e3 = energy_k(A, 0.90)
        Rnd = rg.normal(size=A.shape)
        kn, _ = energy_k(Rnd, 0.90)
        cs = np.mean([float(A[:, t] @ A[:, t + 1] /
                            (np.linalg.norm(A[:, t]) * np.linalg.norm(A[:, t + 1]) + 1e-30))
                      for t in range(A.shape[1] - 1)])
        print(f"  {r:>6}{len(ii):>9}{k50:>11}{k90:>11}{e3:>10.3f}{kn:>10}{cs:>10.3f}")
    print()
print(f"  k90 << null k90 => the component's flow lives on a low-dim manifold")
print(f"  cos lag1 high   => it also keeps the same direction")
