"""THE SPLIT-HALF GATE: IS THE ROTATION REAL, AND AT WHAT SMOOTHING HORIZON?

|dQ| ~ 1.7 rad between disjoint 4-step windows, nearly invariant to a 10x
learning-rate change. Two readings:

  REAL      the frame genuinely turns that fast
  NOISE     a 3-dim frame estimated from 4 stochastic gradients is simply not
            reproducible, and |dQ| measures estimation variance

The discriminator is a split-half on INTERLEAVED updates from the SAME window:
even-indexed steps build V_even, odd-indexed build V_odd. Both see the same
underlying motion, so any disagreement is estimator variance and nothing else.

    d_Gr(V_even, V_odd) ~ 0     the frame is resolved; rotation is motion
    d_Gr(V_even, V_odd) ~ |dQ|  the frame is noise; rotation was never measured

Run at each candidate smoothing horizon H = 1/(1-beta), so the smallest H that
passes the gate is a MEASUREMENT rather than a hyperparameter. The stream is
filtered first:

    m_t = beta m_{t-1} + (1-beta) g_t

and the frames are built from the filtered history. Reported alongside:
  - the raw (H=1) case, which is the previous measurement
  - the FILTERED frame's velocity between disjoint windows, which is the bend
    rate of the highway once the jitter is removed
  - a shuffled control: same vectors in random time order, which destroys the
    drift while preserving the noise, so a passing gate cannot be an artefact of
    smoothing alone
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn, math, re, json

V, D, L, T = 40, 12, 2, 16
KDIM = 3
STEPS = 420


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


def keyof(nm):
    if nm.startswith("te") or nm.startswith("pe"):
        return "EMB"
    mm = re.match(r"b\.(\d)\.", nm)
    if not mm:
        return None
    li = mm.group(1)
    if re.search(r"\.(q|k|v|o)\.", nm):
        return f"ATTN{li}"
    if re.search(r"\.(g|u|w)\.", nm):
        return f"FF{li}"
    return None


torch.manual_seed(0)
m = M(); named = [(n, p) for n, p in m.named_parameters()]
ps = [p for _, p in named]
P = sum(p.numel() for p in ps)
span = {}; i = 0
for nm, p in named:
    span[nm] = (i, i + p.numel()); i += p.numel()
idx = {}
for nm, (a, b) in span.items():
    k = keyof(nm)
    if k:
        idx.setdefault(k, []).append(torch.arange(a, b))
idx = {k: torch.cat(v) for k, v in idx.items()}
opt = torch.optim.AdamW(m.parameters(), lr=3e-3, betas=(0.9, 0.95), weight_decay=0.1)


def flat():
    return torch.cat([p.data.flatten() for p in ps]).clone()


# collect the raw update stream once; every horizon is evaluated on the same data
US = []
for st in range(STEPS):
    th = flat()
    x, y = batch(); _, l = m(x, y); opt.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    US.append((flat() - th).clone())
print(f"  P={P}, k={KDIM}, {STEPS} updates collected, {len(idx)} blocks\n")


def frame(cols, ii):
    A = torch.stack([c[ii] for c in cols], 1)
    return torch.linalg.svd(A, full_matrices=False)[0][:, :KDIM]


def dgr(Q1, Q2):
    sv = torch.linalg.svdvals(Q1.T @ Q2).numpy()
    return float(np.sqrt((np.arccos(np.clip(sv, 0, 1)) ** 2).sum()))


def ema_stream(beta):
    out = []; mstate = torch.zeros(P)
    for u in US:
        mstate = beta * mstate + (1 - beta) * u
        out.append(mstate.clone())
    return out


MAXD = math.sqrt(KDIM) * math.pi / 2
print(f"  max possible d_Gr for k={KDIM}: {MAXD:.3f} rad\n")
print(f"  {'H':>5}{'beta':>7}{'split-half':>12}{'shuffled':>11}"
      f"{'disjoint vel':>14}{'gate':>7}")
rows = []
WINL = 16
for H in (1, 4, 8, 16, 32, 64):
    beta = 0.0 if H == 1 else 1 - 1.0 / H
    S = US if H == 1 else ema_stream(beta)
    sh = []; vel = []; shuf = []
    start = 2 * H + 20
    for t in range(start, STEPS - 2 * WINL, WINL):
        w1 = S[t:t + WINL]; w2 = S[t + WINL:t + 2 * WINL]
        for k, ii in idx.items():
            Qe = frame(w1[0::2], ii); Qo = frame(w1[1::2], ii)
            sh.append(dgr(Qe, Qo))
            vel.append(dgr(frame(w1, ii), frame(w2, ii)))
            pm = list(np.random.default_rng(t).permutation(len(w1)))
            Qs = frame([w1[j] for j in pm[:len(pm)//2]], ii)
            Qs2 = frame([w1[j] for j in pm[len(pm)//2:]], ii)
            shuf.append(dgr(Qs, Qs2))
    a, b, c = np.mean(sh), np.mean(shuf), np.mean(vel)
    ok = "PASS" if a < 0.3 * MAXD else ""
    print(f"  {H:>5}{beta:>7.3f}{a:>12.3f}{b:>11.3f}{c:>14.3f}{ok:>7}", flush=True)
    rows.append(dict(H=H, split=float(a), shuf=float(b), vel=float(c)))
json.dump(rows, open("/home/claude/work/res_gate.json", "w"), indent=2)
print(f"\n  split-half -> 0 as H grows => the frame resolves; the residual")
print(f"  'disjoint vel' is then the true bend rate of the filtered highway")
print(f"  split-half ~ disjoint vel at all H => the frame is noise at every horizon")
