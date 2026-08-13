"""FOUR-POINT TRAJECTORY TOPOLOGY PER COMPONENT.

For each role W in {EMB, LN, W_Q, W_K, W_V, W_O, FF} and four checkpoints
t1..t4, three families of measurement.

LIMIT (invariance / stationary anchor)
    overlap of the top-r eigenspaces of W^(t1) and W^(t4). ->1 means the
    subspace has solidified into an invariant frame. Compared against the
    overlap of two RANDOM r-planes in the same block, which is the floor set by
    block size alone.

COLIMIT (gluability)
    PR of the 4-step trajectory matrix T_W = [W1|W2|W3|W4]. PR->1 means the four
    states collapse onto a common low-rank manifold; PR->4 means they are
    mutually orthogonal and unglued. NOTE the raw weights are dominated by their
    common mean, which would force PR->1 trivially, so PR is computed on the
    CENTRED trajectory, and the raw value reported alongside for contrast.

PATH SHAPE
    kappa   ||W3-W2|| / (||W2-W1||*||W4-W3||), the geodesic defect
    cos     between (W2-W1) and (W4-W3): are late steps parallel to early ones?

Windows are matched to the regimes measured earlier: 20-60, 100-160, 240-300.
Run at P=3672 so every SVD is exact.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn, math, json, re
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


def role(nm):
    if nm.startswith("te"):
        return "EMB_tok"
    if nm.startswith("pe"):
        return "EMB_pos"
    if "n1" in nm or "n2" in nm or nm.startswith("nf"):
        return "LN"
    for pat, lab in ((r"\.q\.", "W_Q"), (r"\.k\.", "W_K"),
                     (r"\.v\.", "W_V"), (r"\.o\.", "W_O")):
        if re.search(pat, nm):
            return lab
    if ".g." in nm or ".u." in nm or ".w." in nm:
        return "FF"
    return "other"


ROLES = ["EMB_tok", "EMB_pos", "LN", "W_Q", "W_K", "W_V", "W_O", "FF"]
opt = torch.optim.AdamW(m.parameters(), lr=3e-3, betas=(0.9, 0.95), weight_decay=0.1)
WINDOWS = [(20, 60, "early"), (100, 160, "mid"), (240, 300, "late")]
snap_at = sorted({lo + int(k * (hi - lo) / 3) for lo, hi, _ in WINDOWS for k in range(4)})
snaps = {}
step = 0
while step < 301:
    x, y = batch(); _, l = m(x, y); opt.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); step += 1
    if step in snap_at:
        snaps[step] = {r: [] for r in ROLES}
        for nm, p in named:
            r = role(nm)
            if r in snaps[step]:
                snaps[step][r].append(p.data.clone().reshape(p.shape[0], -1)
                                      if p.dim() == 2 else p.data.clone().reshape(-1, 1))


def topmat(mats):
    """stack blocks of a role into one matrix; rows must agree, so blocks with
    differing row counts are transposed to the majority row count where possible
    and otherwise flattened into columns."""
    rows = [x.shape[0] for x in mats]
    tgt = max(set(rows), key=rows.count)
    cols = []
    for x in mats:
        if x.shape[0] == tgt:
            cols.append(x.reshape(tgt, -1))
        elif x.shape[1] == tgt:
            cols.append(x.T.reshape(tgt, -1))
        else:
            v = x.reshape(-1)
            k = (v.numel() // tgt) * tgt
            if k > 0:
                cols.append(v[:k].reshape(tgt, -1))
    return torch.cat(cols, 1)


def eigspace(Wm, r=3):
    U, S, Vt = torch.linalg.svd(Wm.float(), full_matrices=False)
    return U[:, :min(r, U.shape[1])]


def ov(A, B):
    k = min(A.shape[1], B.shape[1])
    return float((A[:, :k].T @ B[:, :k]).pow(2).sum() / k)


def PR(Mx):
    s = torch.linalg.svdvals(Mx.float()).numpy() ** 2
    return float(s.sum() ** 2 / (s ** 2).sum())


print("  P=3672, exact SVDs; PR on CENTRED trajectory (raw shown for contrast)\n")
for lo, hi, lab in WINDOWS:
    ts = sorted([s for s in snap_at if lo <= s <= hi])[:4]
    print(f"  === {lab}  checkpoints {ts} ===")
    print(f"  {'role':>6}{'ov(t1,t4)':>11}{'ov random':>11}{'PR cent':>9}"
          f"{'PR raw':>8}{'kappa':>9}{'cos(12,34)':>12}")
    for r in ROLES:
        Ws = [topmat(snaps[t][r]) for t in ts]
        E1, E4 = eigspace(Ws[0]), eigspace(Ws[-1])
        n = Ws[0].shape[0]
        g = torch.Generator().manual_seed(3)
        R1 = torch.linalg.qr(torch.randn(n, 3, generator=g))[0]
        R2 = torch.linalg.qr(torch.randn(n, 3, generator=g))[0]
        Tw = torch.stack([w.flatten() for w in Ws], 1)
        Tc = Tw - Tw.mean(1, keepdim=True)
        d1 = Ws[1].flatten() - Ws[0].flatten()
        d2 = Ws[2].flatten() - Ws[1].flatten()
        d3 = Ws[3].flatten() - Ws[2].flatten()
        kap = float(d2.norm() / max(float(d1.norm() * d3.norm()), 1e-30))
        cs = float((d1 * d3).sum() / (d1.norm() * d3.norm() + 1e-30))
        print(f"  {r:>6}{ov(E1,E4):>11.4f}{ov(R1,R2):>11.4f}{PR(Tc):>9.3f}"
              f"{PR(Tw):>8.3f}{kap:>9.2f}{cs:>12.4f}")
    print()
print("  ov(t1,t4)->1 vs random => the limit has solidified: an invariant frame")
print("  PR(centred)->1         => the colimit has glued: one shared manifold")
print("  cos(12,34)->1          => late steps parallel to early ones")
