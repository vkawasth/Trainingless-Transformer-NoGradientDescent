"""A SMALL COMPILER WITH GPS-CONTROLLED PHASE 3.

Everything measured so far is assembled into a working system at P=3672, where
computation is exact and a result can be trusted before any attempt at scale.

Phase 3 is replaced by a controller rather than eliminated, because the
measurements say elimination is impossible: the descent is ~35-dimensional at any
instant and those 35 dimensions rotate (frozen frame -> val 1.62 vs 0.19
re-estimated). So the controller tracks the frame instead of precomputing it.

  detect        DETECT full-rank steps, estimate per-block top-k motion frames
  project       run confined to the frames
  monitor       every CHECK steps, one full gradient gives
                    gamma = 1 - ||P_Q g||^2 / ||g||^2
                energy-weighted across blocks, since LN is 330x smaller than FF
                and a max would let the smallest block drive the schedule
  recalibrate   when gamma > theta, WIN full-rank steps, re-estimate, resume

THE BUDGET IS CAPPED. The previous attempt at 4.3M spent 308 full steps against
a 200-step baseline and "won" by doing more work. Here total full-rank steps are
capped at the same number the uniform baseline gets, so the arms are comparable
by construction and the only question is PLACEMENT.

Arms: full, uniform (same cap, evenly spaced), gps@theta for a swept theta,
frozen (detect once). Threshold is swept rather than chosen, so the schedule is
a consequence of theta and not of hindsight.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn, math, re, json

V, D, L, T = 40, 12, 2, 16
K = 5
STEPS = 300
DETECT = 20
WIN = 8
CHECK = 5
CAP = 80          # total full-rank steps allowed to every non-full arm


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


def role(nm):
    if nm.startswith("te") or nm.startswith("pe"):
        return "EMB"
    if "n1" in nm or "n2" in nm or nm.startswith("nf"):
        return "LN"
    for pat, lab in ((r"\.q\.", "W_Q"), (r"\.k\.", "W_K"),
                     (r"\.v\.", "W_V"), (r"\.o\.", "W_O")):
        if re.search(pat, nm):
            return lab
    if re.search(r"\.(g|u|w)\.", nm):
        return "FF"
    return "other"


ROLES = ["EMB", "LN", "W_Q", "W_K", "W_V", "W_O", "FF"]


def run(mode, theta=None, seed=0):
    torch.manual_seed(seed)
    m = M(); named = [(n, p) for n, p in m.named_parameters()]
    ps = [p for _, p in named]
    P = sum(p.numel() for p in ps)
    span = {}; i = 0
    for nm, p in named:
        span[nm] = (i, i + p.numel()); i += p.numel()
    idx = {r: [] for r in ROLES}
    for nm, (a, b) in span.items():
        if role(nm) in idx:
            idx[role(nm)].append(torch.arange(a, b))
    idx = {r: torch.cat(v) for r, v in idx.items() if len(v)}
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, betas=(0.9, 0.95),
                            weight_decay=0.1)

    def flat():
        return torch.cat([p.data.flatten() for p in ps]).clone()

    def setth(t):
        with torch.no_grad():
            j = 0
            for p in ps:
                q = p.numel(); p.data.copy_(t[j:j + q].view_as(p)); j += q

    def gv():
        return torch.cat([(p.grad.flatten() if p.grad is not None
                           else torch.zeros(p.numel())) for p in ps]).clone()
    EV = [batch(48) for _ in range(3)]

    def loss():
        t = 0.0
        with torch.no_grad():
            for x, y in EV:
                t += float(m(x, y)[1])
        return t / len(EV)

    Q = None; recent = {r: [] for r in idx}
    used = 0; recals = []; curve = []; gam = []
    uni_every = max(1, (STEPS - DETECT) // max(1, CAP - DETECT))
    for st in range(STEPS):
        full = False
        if mode == "full":
            full = True
        elif st < DETECT:
            full = True
        elif mode == "uniform":
            full = (used < CAP) and (st % uni_every == 0)
        elif mode == "frozen":
            full = False
        elif mode == "gps" and Q is not None and st % CHECK == 0 and used < CAP:
            th = flat()
            x, y = batch(); m.zero_grad(); _, l = m(x, y); l.backward()
            g = gv(); m.zero_grad(); setth(th)
            num = den = 0.0
            for r, ii in idx.items():
                gi = g[ii]; e = float((gi * gi).sum())
                pj = Q[r] @ (Q[r].T @ gi)
                num += e * (1 - float((pj * pj).sum()) / max(e, 1e-30)); den += e
            gl = num / max(den, 1e-30); gam.append((st, gl))
            if gl > theta:
                recals.append(st); full = True
        th = flat()
        x, y = batch(); _, l = m(x, y); opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        du = flat() - th
        if full and (mode == "full" or used < CAP):
            if mode != "full":
                used += 1
            for r, ii in idx.items():
                recent[r].append(du[ii].clone())
                if len(recent[r]) > max(WIN, DETECT):
                    recent[r].pop(0)
            if mode != "full" and (st == DETECT - 1 or (st in recals)):
                Q = {r: torch.linalg.svd(torch.stack(recent[r], 1),
                                         full_matrices=False)[0][:, :K] for r in idx}
        else:
            nu = torch.zeros_like(du)
            for r, ii in idx.items():
                nu[ii] = Q[r] @ (Q[r].T @ du[ii])
            setth(th + nu)
        if (st + 1) % 100 == 0:
            curve.append((st + 1, loss()))
    return dict(val=loss(), curve=curve, used=used, recals=len(recals),
                first=recals[:6], P=P, gam=gam[:6])


print(f"  P=3672, k={K}/block x 7 = {7*K} dims ({7*K/3672*100:.2f}% of P)")
print(f"  budget cap {CAP} full-rank steps of {STEPS} for every non-full arm\n")
print(f"  {'arm':>11}{'full used':>11}{'val@100':>10}{'val@200':>10}"
      f"{'val@300':>10}{'recals':>8}")
out = {}
arms = [("full", "full", None), ("uniform", "uniform", None),
        ("frozen", "frozen", None),
        ("gps@0.3", "gps", 0.3), ("gps@0.5", "gps", 0.5),
        ("gps@0.7", "gps", 0.7), ("gps@0.9", "gps", 0.9)]
for lab, mode, th in arms:
    r = run(mode, theta=th)
    out[lab] = r
    c = dict(r["curve"])
    extra = f"   first {r['first']}" if r["recals"] else ""
    print(f"  {lab:>11}{r['used']:>11}{c.get(100,float('nan')):>10.4f}"
          f"{c.get(200,float('nan')):>10.4f}{c.get(300,float('nan')):>10.4f}"
          f"{r['recals']:>8}{extra}", flush=True)
json.dump({k: {kk: vv for kk, vv in v.items() if kk != "gam"} for k, v in out.items()},
          open("/home/claude/work/res_smallgps.json", "w"), indent=2)
print(f"\n  gps beats uniform at the SAME cap => placement matters, not spend")
print(f"  gps ~ uniform                      => only the budget matters")
print(f"  frozen far worse                   => the frame must be tracked")
