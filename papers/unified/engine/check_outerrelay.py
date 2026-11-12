"""Checks for outer reconciliation and the relay clock (module pce/outerrelay.py), on the exact world of Study 26.
Honest contexts h: the development contexts other than those the coalition controls. Decision g = P(Y=1 | M=1) - P(Y=1 | M=0) (Study 26),
threshold tau2 = g(P*) + 0.04; the coalition wants g >= tau2.
  OW1 IDENTITY  extension and its certificate (the honest configuration and 40 far-side ones, six outer contexts): a local section with
                outer tables extends iff no Farkas certificate exists; the honest configuration extends with every outer context; every
                certificate is a cover-additive function, non-negative on all 32 states, with expectation -1 under the reported tables
  OW2 IDENTITY  the outer layer catches every coherent far-side coalition configuration iff its separation margin rho_W > 0; when
                rho_W = 0 the margin's optimiser is a far-side configuration that extends (six candidate outer contexts, 60 configurations each)
  OW3 IDENTITY  a witness is not a culprit: on Study 26's pooled tables (z = 3) removing {M,C} (the corrupted table) and removing
                {M,C,Y} (an honest-plus-coalition table) each restore compatibility; every minimal conflict contains both
  OW4 FINDING   choosing what to inspect: margins rho_W and the noise bound eta (3 standard errors in l1, 3000 episodes) for the candidates;
                which of them would have protected Study 26's decision (rho_W > 2 eta)
  OW5 IDENTITY  the protection witness: the dual of the margin LP gives lambda on the inspected cells (|lambda| <= 1), mu on the honest
                tables and nu >= 0 with lambda(x_W) + mu(x_h) - nu g'(x) <= 0 on every state and lambda.q* + mu.h - nu tau' = rho_W (strong
                duality); every sampled wrong-side law matching h then has lambda.(M_W p - q*) <= -rho_W, hence ||M_W p - q*||_1 >= rho_W
                (g' = -g, tau' = -tau: the wrong side here is g >= tau)
  RY1 IDENTITY  activation times are invariant under swapping adjacent independent events (they depend only on the trace)
  RY2 IDENTITY  the clock is not a braid invariant: e e^-1 f and f have the same braid (normal form) but different activation; and the
                braid is sound for reach: everything reached along the reduced word is reached along the history (500 histories)
  RY3 IDENTITY  relay clock, one actor at the end of a chain 1-2-3: the flip takes exactly a_3 + ceil(D / delta_3) rounds, the bound T*,
                and exceeds the clock of Theorem 122 by the relay delay a_3 (100 histories)
  RY4 IDENTITY  relay clock, two actors (members 1 and 3 control {M,C,Y} and {R,H,Y}; decision g + P(Y=1 | R=1, H=1)): every simulated
                coherent attack takes at least T* rounds (60 histories x 3 budgets)
"""
import os, sys, json, itertools, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import warnings; warnings.filterwarnings("ignore")
from scipy.optimize import linprog
from scipy.stats import chi2
from pce import outerrelay as O
W = O.W
results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
rng = np.random.default_rng(271); P = W.truth(); NS = O.NS
DEV = [c for c, _, _ in W.CORE_ALLOC]; MCY = ("M", "C", "Y"); RHY = ("R", "H", "Y")
M1, M0, Y1 = O.ev(lambda v: v["M"] == 1), O.ev(lambda v: v["M"] == 0), O.ev(lambda v: v["Y"] == 1)
g = Y1 * M1 / (M1 @ P) - Y1 * M0 / (M0 @ P); tau = g @ P + 0.04
h_ctx = [c for c in DEV if c != MCY]; hblocks = [(c, O.ctxmat(c) @ P) for c in h_ctx]
CANDS = [("M", "Y"), ("M", "R", "Y"), ("M", "H", "Y"), ("M", "C", "R", "Y"), ("C", "Y"), MCY]

# OW1
ok = True; nfeas = ninf = 0
honest_cfg = [(MCY, O.ctxmat(MCY) @ P)]
for it in range(41):
    cfg = honest_cfg if it == 0 else O.far_side_config(h_ctx, [MCY], g, tau, P, rng)[0]   # the first is the honest configuration
    for Wc in CANDS:
        blocks = hblocks + cfg + [(Wc, O.ctxmat(Wc) @ P)]; feas = O.extends(blocks); y = O.certificate(blocks)
        ok &= feas == (y is None)
        if y is not None:
            M, d = O.stack(blocks); ok &= (M.T @ y).min() >= -1e-7 and abs(d @ y + 1) < 1e-7; ninf += 1
        else: nfeas += 1
check("OW1 IDENTITY extension fails iff a Farkas certificate exists; certificates are non-negative cover-additive functions", ok and nfeas == len(CANDS), f"{nfeas} extensions, {ninf} certified failures")

# OW2
ok = True; rows = {}
for Wc in CANDS:
    r, p0 = O.rho(h_ctx, [Wc], g, tau, P); caught = 0
    for _ in range(60):
        cfg, _p = O.far_side_config(h_ctx, [MCY], g, tau, P, rng); caught += not O.extends(hblocks + cfg + [(Wc, O.ctxmat(Wc) @ P)])
    if r > 1e-9: ok &= caught == 60
    else:
        cfg0 = [(MCY, O.ctxmat(MCY) @ p0)]; ok &= O.extends(hblocks + cfg0 + [(Wc, O.ctxmat(Wc) @ P)]) and g @ p0 >= tau - 1e-9
    rows["".join(Wc)] = (float(r), caught)
check("OW2 IDENTITY the outer layer catches every coherent far-side configuration iff rho_W > 0", ok,
      "; ".join(f"{k}: rho {v[0]:.4f}, caught {v[1]}/60" for k, v in rows.items()))

# OW3 (Study 26 pooled tables, same flagging and z = 3 rule)
dev, hold = W.generate(); core = [s for s in dev if s["ctx"] != ("L3",)]; by = {}
for s in core: by.setdefault(s["ctx"], []).append(s)
F = set()
for ctx, ss in by.items():
    for s in ss:
        o = sum(x["counts"] for x in ss if x is not s); e = s["n"] * o / o.sum(); m = e > 1e-9
        if chi2.sf(float((((s["counts"] - e) ** 2)[m] / e[m]).sum()), m.sum() - 1) < 0.001: F.add(s["id"])
pool = {}
for s in core:
    if s["id"] not in F: pool[s["ctx"]] = pool.get(s["ctx"], 0) + s["counts"]
def robust_feasible(ctxs):
    Aub, bub = [], []
    for c in ctxs:
        cnt = pool[c]; N = cnt.sum(); p = cnt / N; tol = 3 * np.sqrt(p * (1 - p) / N) + 1e-6; A = O.ctxmat(c); Aub += [A, -A]; bub += [p + tol, -p + tol]
    return linprog(np.zeros(NS), A_ub=np.vstack(Aub), b_ub=np.concatenate(bub), A_eq=np.ones((1, NS)), b_eq=[1.0], bounds=[(0, None)] * NS, method="highs").status == 0
allc = list(pool); minimal = []
for r in range(1, len(allc) + 1):
    for sub in itertools.combinations(allc, r):
        if any(set(m) <= set(sub) for m in minimal): continue
        if not robust_feasible(list(sub)): minimal.append(sub)
no_mc = robust_feasible([c for c in allc if c != ("M", "C")]); no_mcy = robust_feasible([c for c in allc if c != MCY]); full = robust_feasible(allc)
ok = (not full) and no_mc and no_mcy and all(("M", "C") in m and MCY in m for m in minimal)
check("OW3 IDENTITY a witness is not a culprit: removing the corrupted {M,C} or the honest {M,C,Y} each restores compatibility", ok,
      f"minimal conflicts: {['/'.join(''.join(c) for c in m) for m in minimal]}; without MC: {no_mc}; without MCY: {no_mcy}")

# OW4
info = []; prot = []
for Wc in CANDS:
    r = rows["".join(Wc)][0]; q = O.ctxmat(Wc) @ P; eta = 3 * np.sqrt(q * (1 - q) / 3000).sum(); info.append(f"{''.join(Wc)}: rho {r:.4f}, 2 eta {2 * eta:.4f}")
    if r > 2 * eta: prot.append("".join(Wc))
check("OW4 FINDING margins and noise for each candidate inspection (3000 episodes); which would protect the decision", True, "; ".join(info) + f"; protecting: {prot}")

# OW5: the protection witness (dual of the margin LP)
ok = True; info5 = []
Mh = np.vstack([O.ctxmat(c) for c in h_ctx]); mh = Mh @ P; gp, taup = -g, -tau
for Wc in CANDS:
    r = rows["".join(Wc)][0]
    if not (np.isfinite(r) and r > 1e-9): continue
    Mw = O.ctxmat(Wc); q = Mw @ P; k = Mw.shape[0]; nh = Mh.shape[0]
    # variables (lambda[k], mu[nh], nu); maximise lambda.q + mu.mh - nu taup
    c = -np.r_[q, mh, -taup]; Aub = np.hstack([Mw.T, Mh.T, -gp[:, None]]); bub = np.zeros(NS)
    res = linprog(c, A_ub=Aub, b_ub=bub, bounds=[(-1, 1)] * k + [(None, None)] * nh + [(0, None)], method="highs")
    lam, mu, nu = res.x[:k], res.x[k:k + nh], res.x[-1]; dual = -res.fun
    ok &= abs(dual - r) < 1e-7 and (Aub @ res.x).max() <= 1e-8
    worst = -np.inf
    for _ in range(100):
        cfg, pw = O.far_side_config(h_ctx, [MCY], g, tau, P, rng)
        worst = max(worst, lam @ (Mw @ pw - q) + r); ok &= np.abs(Mw @ pw - q).sum() >= r - 1e-8
    ok &= worst <= 1e-8; info5.append(f"{''.join(Wc)}: dual {dual:.4f} = rho, nu {nu:.3f}")
check("OW5 IDENTITY the protection witness: the dual certificate equals rho_W and separates every sampled wrong-side law", ok, "; ".join(info5))

# RY1, RY2
E3 = [(1, 2), (2, 3), (3, 4)]
def rword(n, edges): return [(edges[rng.integers(len(edges))], int(rng.choice([1, -1]))) for _ in range(n)]
ok = True
for _ in range(300):
    w = rword(rng.integers(4, 11), E3); a0 = O.activation(w, 1); nf0 = O.reduce(w)[0]
    for _ in range(20):
        i = rng.integers(len(w) - 1)
        if not O.dependent(w[i], w[i + 1]): w[i], w[i + 1] = w[i + 1], w[i]
    ok &= O.activation(w, 1) == a0 and O.reduce(w)[0] == nf0
check("RY1 IDENTITY activation depends only on the trace: invariant under swaps of independent events", ok, "300 histories, 20 swaps each")
w1 = [((1, 2), 1), ((1, 2), -1), ((2, 3), 1)]; w2 = [((2, 3), 1)]
same = O.reduce(w1)[0] == O.reduce(w2)[0]; a1, a2 = O.activation(w1, 1), O.activation(w2, 1)
sound = True; strict = 0
for _ in range(500):
    w = rword(rng.integers(3, 12), E3); red = O.reduce(w)[1]; Rw, Rr = set(O.activation(w, 1)), set(O.activation(red, 1))
    sound &= Rr <= Rw; strict += Rr < Rw
check("RY2 IDENTITY the clock is not a braid invariant, but the braid is sound for reach", same and 3 in a1 and 3 not in a2 and sound,
      f"same braid: {same}; activation {a1} vs {a2}; reduced-word reach inside history reach in 500/500, strictly smaller in {strict}")

# RY3, RY4: simulated coherent attacks
def Dfun_factory(h_ctx_, members, z, gv, tv):
    Mh = np.vstack([O.ctxmat(c) for c in h_ctx_]); mh = Mh @ P
    def D(A, z=z, want_p=False):
        fixed = [(c, z[j]) for j, c in members.items() if j not in A]; mov = [(j, c) for j, c in members.items() if j in A]
        k = sum(2 ** len(c) for _, c in mov); cost = np.r_[np.zeros(NS), np.ones(k)]
        Aeq = [np.hstack([Mh, np.zeros((len(mh), k))])]; beq = [mh]
        for c, t in fixed: Aeq.append(np.hstack([O.ctxmat(c), np.zeros((len(t), k))])); beq.append(t)
        Aub = [np.r_[-gv, np.zeros(k)][None]]; bub = [[-tv]]; off = NS
        for j, c in mov:
            Mc = O.ctxmat(c); n = Mc.shape[0]; Iblk = np.zeros((n, k)); Iblk[:, off - NS:off - NS + n] = np.eye(n)
            Aub += [np.hstack([Mc, -Iblk]), np.hstack([-Mc, -Iblk])]; bub += [z[j], -z[j]]; off += n
        r = linprog(cost, A_ub=np.vstack(Aub), b_ub=np.concatenate(bub), A_eq=np.vstack(Aeq), b_eq=np.concatenate(beq), bounds=[(0, None)] * (NS + k), method="highs")
        if r.status != 0: return (np.inf, None) if want_p else np.inf
        return (r.fun, r.x[:NS]) if want_p else r.fun
    return D, Mh, mh
def flipped(h_ctx_, members, z, gv, tv, Mh, mh):
    Aeq = [Mh] + [O.ctxmat(c) for j, c in members.items()]; beq = [mh] + [z[j] for j in members]
    r = linprog(gv, A_eq=np.vstack(Aeq), b_eq=np.concatenate(beq), bounds=[(0, None)] * NS, method="highs"); return r.status == 0 and r.fun >= tv - 1e-9
def simulate(h_ctx_, members, a, delta, gv, tv, cap=4000):
    z = {j: O.ctxmat(c) @ P for j, c in members.items()}; Dz, Mh, mh = Dfun_factory(h_ctx_, members, z, gv, tv)
    for rnd in range(1, cap):
        A = frozenset(j for j, t in a.items() if t < rnd and j in members and delta.get(j, 0) > 0)
        if A:
            Dval, pst = Dfun_factory(h_ctx_, members, z, gv, tv)[0](A, z=z, want_p=True)
            if np.isfinite(Dval) and Dval > 1e-12:
                tgt = {j: O.ctxmat(members[j]) @ pst for j in A}; dist = {j: np.abs(tgt[j] - z[j]).sum() for j in A}
                f = min([1.0] + [delta[j] / dist[j] for j in A if dist[j] > 1e-15])
                for j in A: z[j] = z[j] + f * (tgt[j] - z[j])
        if flipped(h_ctx_, members, z, gv, tv, Mh, mh): return rnd
    return np.inf
E2 = [(1, 2), (2, 3)]; ok = True; delays = []
members3 = {3: MCY}; z0 = {3: O.ctxmat(MCY) @ P}; D3, _, _ = Dfun_factory(h_ctx, members3, z0, g, tau); Dfull = D3(frozenset({3}))
for _ in range(100):
    while True:
        w = rword(rng.integers(3, 9), E2); a = O.activation(w, 1)
        if 3 in a: break
    d3 = float(rng.uniform(0.005, 0.03)); Tstar = O.flip_bound(a, {3: d3}, lambda A: D3(A)); Tsim = simulate(h_ctx, members3, a, {3: d3}, g, tau)
    ok &= Tstar == Tsim == a[3] + math.ceil(Dfull / d3 - 1e-12); delays.append(Tstar - math.ceil(Dfull / d3 - 1e-12))
check("RY3 IDENTITY relay clock with one actor at the end of the chain: flip time = a_3 + ceil(D / delta), the bound, exactly", ok,
      f"D = {Dfull:.4f}; relay delay over Theorem 122's clock: {min(delays)} to {max(delays)} rounds")
R1H1 = O.ev(lambda v: v["R"] == 1 and v["H"] == 1); g2 = g + Y1 * R1H1 / (R1H1 @ P); tau2b = g2 @ P + 0.06
h2 = [c for c in DEV if c not in (MCY, RHY)]; members = {1: MCY, 3: RHY}; z0 = {j: O.ctxmat(c) @ P for j, c in members.items()}
D2, _, _ = Dfun_factory(h2, members, z0, g2, tau2b); ok = True; gaps = []; Dvals = {"1": D2(frozenset({1})), "3": D2(frozenset({3})), "13": D2(frozenset({1, 3}))}
for _ in range(60):
    while True:
        w = rword(rng.integers(3, 9), E2); a = O.activation(w, 1)
        if 3 in a: break
    for d in ((0.01, 0.01), (0.005, 0.02), (0.02, 0.005)):
        delta = {1: d[0], 3: d[1]}; Tstar = O.flip_bound(a, delta, lambda A: D2(A)); Tsim = simulate(h2, members, a, delta, g2, tau2b)
        ok &= Tsim >= Tstar; gaps.append(Tsim - Tstar)
check("RY4 IDENTITY relay clock with two actors: no simulated coherent attack beats the bound T*", ok,
      f"D by active set: {', '.join(f'{k}: {v:.4f}' for k, v in Dvals.items())}; simulated minus bound: min {min(gaps)}, median {int(np.median(gaps))}, max {max(gaps)}; equal in {sum(x == 0 for x in gaps)}/{len(gaps)}")
json.dump(dict(rho={k: v[0] for k, v in rows.items()}, minimal=[["".join(c) for c in m] for m in minimal], D=Dvals, delays=[int(x) for x in delays], gaps=[int(x) for x in gaps]),
          open(os.path.join(HERE, "outerrelay_summary.json"), "w"), indent=1)
print(f"{sum(results)}/{len(results)} passed")
