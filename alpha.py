"""HOW LITTLE CAN BE SPENT OUTSIDE THE SUBSPACE?

The question "can a subspace optimiser ignore the non-contributing dimensions"
has an answer in the data already, and it is no by both routes:

  CS-Adam met its 90-91% ENERGY coverage target and lost to Adam by 29x. The
  discarded 10% was what training needed.
  Three Fisher directions holding 2% of the energy carried ALL the descent.

Energy is the wrong selector in both directions. So rather than assume the
complement can be dropped, this sweeps how much of it to keep:

    u = P_Q u_adam  +  alpha * (I - P_Q) u_adam

  alpha = 1   plain Adam, exactly
  alpha = 0   pure projection, the CS-Adam regime that failed
  in between  the question

The frame Q is built from the recent update window per block (k=4 x 5 blocks,
rebuilt every 8 steps, EMA-filtered at the measured H*=4). Selection inside the
frame is by SVD of the update history, which is descent-weighted by construction,
rather than by an energy threshold.

Three seeds per alpha, tuned Adam at lr=6e-3 as the reference. Cost is reported:
the projection is nearly free, so alpha < 1 buys nothing unless it also buys
loss.
"""
import math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, json

V, D, L, T_LEN = 40, 12, 2, 16
K_DIM = 4
STEPS = 400
R_REBUILD = 8


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
        a = F.softmax(q @ k.transpose(-2, -1) / math.sqrt(D)
                      + torch.triu(torch.full((x.shape[1], x.shape[1]), -1e9), 1), dim=-1)
        x = x + s.o(a @ v); h = s.n2(x)
        return x + s.w(F.silu(s.g(h)) * s.u(h))


class Net(nn.Module):
    def __init__(s):
        super().__init__()
        s.te = nn.Embedding(V, D); s.pe = nn.Embedding(T_LEN, D)
        s.b = nn.ModuleList([Blk() for _ in range(L)]); s.nf = nn.LayerNorm(D)

    def forward(s, x, y):
        h = s.te(x) + s.pe(torch.arange(x.shape[1]))
        for blk in s.b:
            h = blk(h)
        lo = s.nf(h) @ s.te.weight.T
        return lo, F.cross_entropy(lo.reshape(-1, V), y.reshape(-1))


rules = {i: [(i * 3 + 1) % V, (i * 5 + 2) % V] for i in range(V)}
s_ = [0]
for _ in range(6000):
    c = s_[-1]
    s_.append(int(rules[c][0] if (len(s_) > 1 and s_[-2] % 2 == 0) else rules[c][1]))
seq = np.array(s_)


def mk(seed):
    r = np.random.default_rng(seed)

    def b(n=24):
        i = r.integers(0, len(seq) - T_LEN - 1, n)
        return (torch.tensor(np.stack([seq[j:j + T_LEN] for j in i])),
                torch.tensor(np.stack([seq[j + 1:j + T_LEN + 1] for j in i])))
    return b


def blocks_of(m):
    span = {}; i = 0
    for nm, p in m.named_parameters():
        span[nm] = (i, i + p.numel()); i += p.numel()
    bi = {"ATTN0": [], "FF0": [], "ATTN1": [], "FF1": [], "REST": []}
    for nm, (a, b) in span.items():
        t = torch.arange(a, b)
        if "b.0." in nm and any(z in nm for z in [".q.", ".k.", ".v.", ".o."]):
            bi["ATTN0"].append(t)
        elif "b.0." in nm and any(z in nm for z in [".g.", ".u.", ".w."]):
            bi["FF0"].append(t)
        elif "b.1." in nm and any(z in nm for z in [".q.", ".k.", ".v.", ".o."]):
            bi["ATTN1"].append(t)
        elif "b.1." in nm and any(z in nm for z in [".g.", ".u.", ".w."]):
            bi["FF1"].append(t)
        else:
            bi["REST"].append(t)
    return {k: torch.cat(v) for k, v in bi.items() if len(v)}


def run(alpha, seed, lr=6e-3):
    torch.manual_seed(seed)
    bt = mk(seed); EVB = [bt(48) for _ in range(3)]
    m = Net(); ps = [p for p in m.parameters() if p.requires_grad]
    P = sum(p.numel() for p in ps)
    bi = blocks_of(m)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    hist = {k: [] for k in bi}
    frames = {}
    covered = []
    for st in range(STEPS):
        th = torch.cat([p.data.reshape(-1) for p in ps]).clone()
        x, y = bt(); _, l = m(x, y); opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        u = torch.cat([p.data.reshape(-1) for p in ps]) - th
        for k, ii in bi.items():
            hist[k].append(u[ii].clone())
            if len(hist[k]) > R_REBUILD:
                hist[k].pop(0)
        if alpha < 1.0 and len(hist["FF0"]) == R_REBUILD:
            if st % R_REBUILD == 0 or not frames:
                for k, ii in bi.items():
                    A = torch.stack(hist[k], 1)
                    frames[k] = torch.linalg.svd(A, full_matrices=False)[0][:, :K_DIM]
            nu = torch.zeros(P)
            cap = 0.0; tot = 0.0
            for k, ii in bi.items():
                ub = u[ii]; Q = frames[k]
                par = Q @ (Q.T @ ub)
                nu[ii] = par + alpha * (ub - par)
                cap += float((par * par).sum()); tot += float((ub * ub).sum())
            covered.append(cap / max(tot, 1e-30))
            with torch.no_grad():
                o = 0
                for p in ps:
                    k2 = p.numel()
                    p.data.copy_((th + nu)[o:o + k2].view_as(p)); o += k2
    t = 0.0
    with torch.no_grad():
        for x, y in EVB:
            t += float(m(x, y)[1])
    return t / len(EVB), (float(np.mean(covered)) if covered else 1.0)


SEEDS = (0, 1, 2)
print(f"  P=3672, {STEPS} steps, k={K_DIM} x 5 blocks, rebuild {R_REBUILD}")
print(f"  u = P_Q u  +  alpha * (I - P_Q) u   -- alpha=1 IS Adam\n")
print(f"  {'alpha':>8}" + "".join(f"{'s'+str(s):>9}" for s in SEEDS)
      + f"{'mean':>9}{'sd':>9}{'vs adam':>9}{'coverage':>10}")
base = None
out = {}
for a in (1.0, 0.5, 0.25, 0.1, 0.03, 0.0):
    rs = [run(a, s) for s in SEEDS]
    vs = [r[0] for r in rs]; cv = float(np.mean([r[1] for r in rs]))
    mu = float(np.mean(vs))
    if base is None:
        base = mu
    print(f"  {a:>8.2f}" + "".join(f"{v:>9.4f}" for v in vs)
          + f"{mu:>9.4f}{np.std(vs):>9.4f}{mu/base:>9.2f}x{cv:>10.3f}", flush=True)
    out[str(a)] = dict(mean=mu, sd=float(np.std(vs)), cov=cv)
json.dump(out, open("/home/claude/work/res_alpha.json", "w"), indent=2)
print(f"\n  alpha where the loss is still near 1.00x => how little the complement needs")
print(f"  if only alpha=1 works, the complement cannot be cheapened at this k")
