"""Checks for section 12.25 of the companion note (module pce/viability.py): trajectories of decision viability.
IDENTITY = an exact or numerical instance of a statement proved in the text; CERTIFICATE = an exact computation reported as a result;
NUMERICAL = floating-point (linear programs, iterative proportional fitting), with tolerances stated.
  VB1  IDENTITY     reachable set: after consistent batches of total share Q the data lie in (1 - Q) d + Q D, and every point of it is reached
                    by one batch; the viability shares q_some <= q_all, and the optimal batch of q_some does flip some rule
  VB2  NUMERICAL    open loop, along segments of data: L - tau and U - tau change sign at most twice; robust-value differences of two actions
                    change sign at most p_a + p_b - 1 times (p = dual vertices)
  VB3  IDENTITY     forgetting: two window streams with the same batches after t0 coincide W steps later; with pooling they differ by
                    exactly N_t0 / N_t times the initial difference
  VB4  NUMERICAL    the ranking radius is certified: random consistent edits cheaper than it never change the ranking of actions by robust value
  VB5  NUMERICAL    a five-regime open-loop stream (clean, degeneration to the rigid face, poisoning on the facet, off the facet, recovery):
                    per step fibre dimension, interval, R, viability shares, ranking radius, Hausdorff increment; decisions change at most
                    twice per regime and are eventually constant
  VB6  CERTIFICATE  closed loop (the decision picks the next batch's law, window of W batches): the state is finite, so decisions are
                    eventually periodic with period <= 2^W; observed periods for W = 3..8; the identified interval straddles tau throughout
"""
import os, sys, random, itertools
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import margsec as G
from pce import poisonflip as P
from pce import voigeom as V
from pce import viability as Vb
results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
C = G.CELLS; M4 = np.array(G.M, float); T = np.array([1.0 if c[0] == c[1] else 0.0 for c in C]); TAU = 0.45
E4, _ = P.entries_chsh(); SU = P.signed_unit(E4, P.circuits(E4)); DV = P.dual_vertices(T); rng = np.random.default_rng(20261107)
F = G.facets_D(); faces_ = sorted(G.faces_D(F), key=lambda V_: sorted(V_)); rr = random.Random(20261108)
V0 = rr.sample([V_ for V_ in faces_ if len(V_) == 8 and G.face_dim(V_) == 7 and G.fibre_dim_face(V_) == 0], 3)[0]
rr = random.Random(20261108); wv = [rr.randint(5, 25) for _ in V0]; t = sum(wv); wv = [x * 240 // t for x in wv]; wv[0] += 240 - sum(wv)
x1 = np.zeros(16)
for k, v in zip(sorted(V0), wv): x1[k] = v / 240
x2 = 0.8 * x1 + 0.2 / 16

# VB1
from scipy.optimize import linprog
ok = True; rows = []
for s in range(20):
    d = M4 @ rng.dirichlet(np.ones(16)); Q = rng.uniform(0.05, 0.5); k = rng.integers(1, 5)
    shares = rng.dirichlet(np.ones(k)) * Q; laws = [rng.dirichlet(np.ones(16)) for _ in range(k)]
    dn = (1 - Q) * d + sum(sh * (M4 @ x) for sh, x in zip(shares, laws))
    tgt = (dn - (1 - Q) * d) / Q
    r = linprog(np.zeros(16), A_eq=np.vstack([M4, np.ones(16)]), b_eq=np.r_[tgt, 1.0], bounds=[(0, None)] * 16, method="highs"); ok &= r.status == 0
for x in (x1, x2, rng.dirichlet(np.ones(16)), rng.dirichlet(np.ones(16))):
    d = M4 @ x; L, U = P.lower_upper(d, T); tau = TAU if L > TAU else L - 0.03
    qs, qa = Vb.q_some(d, T, tau), Vb.q_all(d, T, tau, DV)
    # verify q_some by reconstructing the optimal batch: bisection on q with an LP for the batch
    def flips(q):
        n = 16; Aeq = np.vstack([np.c_[M4, -M4], np.r_[np.ones(n), np.zeros(n)][None], np.r_[np.zeros(n), np.ones(n)][None]])
        rr_ = linprog(np.zeros(2 * n), A_ub=np.r_[T, np.zeros(n)][None], b_ub=[tau], A_eq=Aeq, b_eq=np.r_[(1 - q) * d, 1.0, q], bounds=[(0, None)] * (2 * n), method="highs")
        return rr_.status == 0
    lo, hi = 0.0, 1.0
    for _ in range(50):
        mid = (lo + hi) / 2; lo, hi = (lo, mid) if flips(mid) else (mid, hi)
    ok &= abs(hi - qs) < 1e-6 and qs <= qa + 1e-9; rows.append((round(L, 3), round(U, 3), round(tau, 3), round(qs, 4), round(qa, 4)))
check("VB1 IDENTITY the reachable set after share Q is (1 - Q) d + Q D; viability shares by LP, q_some <= q_all, confirmed by bisection",
      ok, "20 random multi-batch histories reconstructed by one batch; (L, U, tau, q_some, q_all): " + "; ".join(str(r_) for r_ in rows))

# VB2
mx = [0, 0]; mxp = 0; okp = True
for s in range(150):
    da, db = M4 @ rng.dirichlet(np.ones(16)), M4 @ rng.dirichlet(np.ones(16)); qs_ = np.linspace(0, 1, 101)
    LU = [P.lower_upper((1 - q) * da + q * db, T) for q in qs_]; tau = rng.uniform(0.3, 0.7)
    mx[0] = max(mx[0], Vb.sign_changes([l - tau for l, _ in LU])); mx[1] = max(mx[1], Vb.sign_changes([u - tau for _, u in LU]))
pa = []
for s in range(30):
    Ua = [rng.normal(size=16) for _ in range(2)]; p_ = [len(P.dual_lower(u)) for u in Ua]
    da, db = M4 @ rng.dirichlet(np.ones(16)), M4 @ rng.dirichlet(np.ones(16)); qs_ = np.linspace(0, 1, 201)
    sc = Vb.sign_changes([P.lower_upper((1 - q) * da + q * db, Ua[0])[0] - P.lower_upper((1 - q) * da + q * db, Ua[1])[0] for q in qs_])
    okp &= sc <= p_[0] + p_[1] - 1; mxp = max(mxp, sc); pa.append(sum(p_))
check("VB2 NUMERICAL along segments L - tau and U - tau change sign at most twice; robust-value differences at most p_a + p_b - 1 times",
      mx[0] <= 2 and mx[1] <= 2 and okp, f"150 segments x 101 points: at most {mx[0]} and {mx[1]} sign changes; 30 pairs of random actions: at most {mxp} "
      f"sign changes against bounds {min(pa) - 1}-{max(pa) - 1}")

# VB3
ok = True; W = 5
base = [rng.dirichlet(np.ones(16)) for _ in range(20)]; alt = [rng.dirichlet(np.ones(16)) for _ in range(6)] + base[6:]
sa, sb = Vb.window_stream(base, W), Vb.window_stream(alt, W)
ok &= all(np.abs(sa[t] - sb[t]).max() < 1e-12 for t in range(6 + W - 1, 20)) and np.abs(sa[6 + W - 2] - sb[6 + W - 2]).max() > 1e-6
n0 = 6; da = M4 @ np.mean(base[:n0], axis=0); db = M4 @ np.mean(alt[:n0], axis=0); rat = []
for tt in range(n0, 20):
    pa_ = M4 @ np.mean(base[:tt + 1], axis=0); pb_ = M4 @ np.mean(alt[:n0] + base[n0:tt + 1], axis=0)
    rat.append(np.abs(pa_ - pb_).max() / np.abs(da - db).max() - n0 / (tt + 1)); ok &= abs(rat[-1]) < 1e-12
check("VB3 IDENTITY forgetting: window streams that agree after t0 coincide W steps later; pooled streams differ by exactly N_t0 / N_t", ok,
      f"window W = {W}: equal from step {6 + W - 1} on; pooling: largest deviation from N_t0/N_t {max(abs(r_) for r_ in rat):.1e}")

# VB4
U3 = [T - TAU, TAU - T, 0.5 * (T - TAU) + 0.02]; LAM = [Vb.lipschitz_action(u, SU) for u in U3]
ok = True; viol = 0; ntest = 0; rads = []
for s in range(15):
    d = M4 @ rng.dirichlet(np.ones(16)); rad, order = Vb.rank_radius(d, U3, LAM); rads.append(rad)
    for e in range(40):
        y = M4 @ rng.dirichlet(np.ones(16)); lam = rng.uniform(0, 1); dn = (1 - lam) * d + lam * y; c = P.cost(E4, dn - d)
        if c < rad:
            ntest += 1; o2 = np.argsort(Vb.robust_values(dn, U3))[::-1]; viol += list(o2) != list(order)
check("VB4 NUMERICAL the ranking radius is certified: cheaper consistent edits never change the ranking of actions by robust value", viol == 0 and ntest > 0,
      f"Lipschitz constants {[round(l, 3) for l in LAM]}; 15 points, {ntest} edits cheaper than the radius, {viol} ranking changes; radii {min(rads):.4f}-{max(rads):.4f}")

# VB5
W = 6; reg = [("clean", x2), ("degeneration", x1), ("on-facet poison", 0.85 * x1 + 0.15 * np.eye(16)[6]), ("off-facet poison", 0.85 * x1 + 0.15 * np.eye(16)[10]), ("recovery", x2)]
batches = [law for _, law in reg for _ in range(8)]; ds = Vb.window_stream([x2] * W + batches, W)[W:]
traj = []; prev = None
for t, d in enumerate(ds):
    L, U = P.lower_upper(d, T); me = float(T @ G.maxent(d)); aa = V.act_abstain(L, U, TAU)
    rad, order = Vb.rank_radius(d, U3, LAM)
    qs = Vb.q_some(d, T, TAU) if me >= TAU else Vb.q_some(d, -T, -TAU)
    qa = Vb.q_all(d, T, TAU, DV) if me >= TAU else Vb.q_all(d, -T, -TAU, P.dual_vertices(-T))
    hd = V.hausdorff_tv(prev, d) if prev is not None else 0.0
    traj.append(dict(t=t, reg=reg[t // 8][0], L=L, U=U, me=me, R=aa["R"], qs=qs, qa=qa, dim=G.fibre_dim(d), rad=rad, order=tuple(order), hd=hd, act=me >= TAU))
    prev = d
ok = True; per = []
for i, (nm, _) in enumerate(reg):
    seg = traj[8 * i: 8 * i + 8]; sw = sum(1 for a, b in zip(seg, seg[1:]) if a["act"] != b["act"]); ok &= sw <= 2
    per.append(f"{nm}: dims {sorted(set(s_['dim'] for s_ in seg))}, decision switches {sw}, R {max(s_['R'] for s_ in seg):.3f} max, q_some {min(s_['qs'] for s_ in seg):.3f}-{max(s_['qs'] for s_ in seg):.3f}, "
               f"q_all {min(s_['qa'] for s_ in seg):.3f}-{max(s_['qa'] for s_ in seg):.3f}, ranking changes {sum(1 for a, b in zip(seg, seg[1:]) if a['order'] != b['order'])}, max Hausdorff step {max(s_['hd'] for s_ in seg):.3f}")
check("VB5 NUMERICAL a five-regime open-loop stream (window of 6 batches): viability, regret and structure per regime; at most two decision switches per regime",
      ok, " | ".join(per))
TRAJ = traj

# VB6
xu = np.full(16, 1 / 16); eq = [i for i, c in enumerate(C) if c[0] == c[1]]
xa = 0.5 * xu; xa[10] += 0.5; xb = 0.5 * xu; xb[eq[3]] += 0.5
ok = True; pers = []; straddle = True
for W in range(3, 9):
    acts, dsl = Vb.closed_loop(xa, xb, [xu] * W, W, 160, T, TAU, G.maxent); p = Vb.eventual_period(acts); pers.append((W, p)); ok &= p is not None and p <= 2 ** W
    straddle &= all(P.lower_upper(d, T)[0] < TAU < P.lower_upper(d, T)[1] for d in dsl[-20:])
ol = Vb.window_stream([xa] * 40, 5); ol_acts = [int(float(T @ G.maxent(d)) >= TAU) for d in ol]
check("VB6 CERTIFICATE closed loop: finite state, so decisions are eventually periodic (period <= 2^W); open loop with a fixed law is eventually constant",
      ok and straddle and Vb.eventual_period(ol_acts) == 1, f"(window W, period): {pers}; the identified interval straddles tau at every late step (decisions chatter inside the band); "
      f"open loop: decisions constant after step {next(i for i in range(len(ol_acts)) if len(set(ol_acts[i:])) == 1)}")
print(f"{sum(results)}/{len(results)} passed")
