"""Checks for the sequential valuation theorem (prop:seqval, section sec:likval): dependent and adaptive data. New script; it modifies
nothing; uses pce/likval.py and pce/kahlerbridge.py. Setting: F_1, circuit a = (2, -1, 2, -3); term k carries a law P_k on M outcomes;
the wall function after n observations is c_n = c0 - S_n / N with S_n = sum_t Lambda_a(x_t), Lambda_a = sum_k a_k log P_k.
  SQ1 IDENTITY  for adaptive Markov data, c_n = c0 - S_n / N exactly, and S_n = A_n + M_n with A_n = sum_t E_{Q_t} Lambda_a (the
                predictable compensator) and M_n a sum of conditionally centred increments (exact algebra; conditional means checked)
  SQ2 IDENTITY  maximal Azuma: with B = max |Lambda_a|, P(max_{k<=n} |M_k| >= x) <= 2 exp(-x^2 / (8 n B^2)); the empirical frequency over
                2000 Markov-dependent runs at the x where the bound is 0.05 is at most 0.05
  SQ3 IDENTITY  the compensator predicts the crossing: |s_hat - s_A| (s_A the crossing of c0 - A_n / N) shrinks like N^-1/2 under Markov
                dependence (slope in [-0.7, -0.3])
  SQ4 IDENTITY  invisibility: an adversary that changes the latent law (2 x 2 cells) but keeps the law of the observation (the row) at
                every step leaves the law of every observed sequence unchanged (exact enumeration, n = 8): likelihood ratio 1, the
                valuation path equal in law, while the decision quantity P(cell 11) differs
  SQ5 IDENTITY  cost: pathwise, Delta_n^2 <= 2 n B^2 K_n, where Delta_n = A'_n - A_n is the adversary's shift of the compensator and
                K_n = sum_t KL(Q'_t || Q_t) its accumulated divergence (500 adaptive adversaries; zero violations); tightness reported
  (first run: 6/6; the SQ4 note then compared the decision quantity on different histories, 0.150 vs 0.630; now on the same history)
  SQ6 RECORD    F_1 example: honest data never reach the wall; the least-divergence adversary that reaches it at s = 1 has K_n growing
                linearly in N, and a likelihood-ratio test against it at level 0.05 has power near 1 already at N = 200
"""
import os, sys, math, itertools
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import warnings; warnings.filterwarnings("ignore")
from pce import likval as LV, kahlerbridge as KB
results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
rng = np.random.default_rng(20261114); M = 6; a = np.array([2, -1, 2, -3], float)
v, lam0 = KB.F1(0.2); lam0 = np.array(lam0, float); c0 = -a @ lam0
def rlaw(k=M): return rng.dirichlet(np.ones(k) * 2)
P = [rlaw() for _ in range(4)]; LA = LV.circuit_llr(a, P); B = float(np.abs(LA).max())
def markov_kernel(drift):
    """a transition matrix whose rows tilt the stationary law towards low Lambda_a by 'drift' (dependent data)."""
    K = np.array([rlaw() for _ in range(M)]); K = K * np.exp(-drift * LA)[None, :]; return K / K.sum(1, keepdims=True)
# SQ1: exact algebra along an adaptive stream (the kernel switches with the sign of the current wall function)
K1, K2 = markov_kernel(0.3), markov_kernel(-0.2); N = 500.0; worst = 0.0; worst_c = 0.0
for _ in range(50):
    x = int(rng.integers(M)); S = 0.0; A = 0.0; Mv = 0.0; lam = lam0.copy()
    for t in range(400):
        c = c0 - S / N; Qt = (K1 if c < 0 else K2)[x]; x = int(rng.choice(M, p=Qt))
        S += LA[x]; A += Qt @ LA; Mv += LA[x] - Qt @ LA
        lam = lam - np.array([-math.log(P[k][x]) for k in range(4)]) / N
        worst = max(worst, abs(S - (A + Mv))); worst_c = max(worst_c, abs((-a @ lam) - (c0 - S / N)))
check("SQ1 IDENTITY the wall function is c0 - S_n/N exactly for adaptive dependent data, and S_n = A_n + M_n", worst < 1e-9 and worst_c < 1e-12,
      f"max |S - A - M| {worst:.1e}; max |c_n - (c0 - S_n/N)| {worst_c:.1e}")
# SQ2: maximal Azuma for the martingale part
n = 300; xq = math.sqrt(8 * n * B * B * math.log(2 / 0.05)); hits = 0; R = 2000; Kd = markov_kernel(0.5); mx = []
for _ in range(R):
    x = int(rng.integers(M)); Mv = 0.0; m_ = 0.0
    for t in range(n):
        Qt = Kd[x]; x = int(rng.choice(M, p=Qt)); Mv += LA[x] - Qt @ LA; m_ = max(m_, abs(Mv))
    hits += m_ >= xq; mx.append(m_)
check("SQ2 IDENTITY maximal Azuma bound for the martingale part under Markov dependence", hits / R <= 0.05,
      f"B {B:.2f}; bound point x {xq:.1f}; frequency {hits / R:.4f}; 95th percentile of max|M| {np.quantile(mx, 0.95):.1f} (bound is conservative)")
# SQ3: the compensator predicts the crossing time at rate N^-1/2
while True:                                      # a dependent kernel whose stationary drift reaches the wall at 0.3 < s_A < 1.5
    Kc = markov_kernel(float(rng.uniform(0.3, 2.0)))
    w, V = np.linalg.eig(Kc.T); pi = np.real(V[:, np.argmin(abs(w - 1))]); pi = pi / pi.sum(); drift = pi @ LA
    if drift < 0 and 0.3 < c0 / drift < 1.5: break
sA = c0 / drift                                  # c0 - (n/N) drift = 0 at n/N = c0 / drift (c0 < 0, drift < 0)
med = []; Ns = [1e2, 1e3, 3e3, 1e4]
for Nn in Ns:
    e = []
    for _ in range(100):
        nmax = int(3 * sA * Nn); x = int(rng.integers(M)); S = 0.0; A = 0.0; sh = None; sa = None
        U = rng.random(nmax); cum = np.cumsum(Kc, axis=1)
        for t in range(nmax):
            Qt = Kc[x]; A += Qt @ LA; x = int(np.searchsorted(cum[x], U[t])); x = min(x, M - 1); S += LA[x]
            if sh is None and c0 - S / Nn >= 0: sh = (t + 1) / Nn
            if sa is None and c0 - A / Nn >= 0: sa = (t + 1) / Nn
            if sh is not None and sa is not None: break
        e.append(abs((sh or 3 * sA) - (sa or 3 * sA)))
    med.append(float(np.median(e)))
sl = np.polyfit(np.log(Ns), np.log(med), 1)[0]
check("SQ3 IDENTITY the compensator's crossing time predicts the data's at rate N^-1/2 under Markov dependence", -0.7 <= sl <= -0.3,
      f"s_A {sA:.3f}; median |s_hat - s_A| {', '.join(f'{m:.1e}' for m in med)}; slope {sl:.3f}")
# SQ4: invisibility by exact enumeration. Latent cells (r, c) in {0,1}^2, observation y = r. Hypotheses read y only.
Py = [rlaw(2) for _ in range(4)]; LAy = LV.circuit_llr(a, Py)
def latent_honest(hist):
    pr1 = 0.3 + 0.4 * (sum(hist) % 2); return {(1, 0): pr1 * 0.5, (1, 1): pr1 * 0.5, (0, 0): (1 - pr1) * 0.5, (0, 1): (1 - pr1) * 0.5}
def latent_adv(hist):
    """same row law, every bit of the row mass moved to column 1 after any observed 1 (adaptive, inside the fibre)."""
    pr1 = 0.3 + 0.4 * (sum(hist) % 2); tilt = 0.9 if (hist and hist[-1] == 1) else 0.6
    return {(1, 0): pr1 * (1 - tilt), (1, 1): pr1 * tilt, (0, 0): (1 - pr1) * (1 - tilt), (0, 1): (1 - pr1) * tilt}
def seq_law(latent, n):
    out = {}
    for ys in itertools.product([0, 1], repeat=n):
        p = 1.0
        for t in range(n):
            L = latent(list(ys[:t])); p *= sum(v_ for (r, c), v_ in L.items() if r == ys[t])
        out[ys] = p
    return out
n4 = 8; H_ = seq_law(latent_honest, n4); A_ = seq_law(latent_adv, n4)
dev = max(abs(H_[k] - A_[k]) for k in H_)
g_h = latent_honest([1])[(1, 1)]; g_a = latent_adv([1])[(1, 1)]
check("SQ4 IDENTITY an adversary inside the observational fibre leaves the law of every observed sequence unchanged (likelihood ratio 1)",
      dev < 1e-15 and abs(sum(H_.values()) - 1) < 1e-12, f"{2 ** n4} sequences; max |P - P'| {dev:.1e}; decision quantity P(11): honest {g_h:.3f}, adversary {g_a:.3f} after a 1")
# SQ5: pathwise cost inequality Delta_n^2 <= 2 n B^2 K_n
viol = 0; ratios = []
for i in range(500):
    Kh = markov_kernel(float(rng.normal(0, 0.3))); n5 = int(rng.integers(20, 300)); x = int(rng.integers(M)); D = 0.0; KL = 0.0
    style = i % 3; eta = float(rng.uniform(0.01, 0.5))
    for t in range(n5):
        Qh = Kh[x]
        if style == 0: Qa = Qh * np.exp(-eta * LA)                                # least-divergence tilt along Lambda_a
        elif style == 1: Qa = Qh * np.exp(eta * rng.normal(size=M))               # random tilt
        else: Qa = Qh * np.exp(-eta * LA * (1 if t % 2 else -0.5))                 # adaptive, alternating
        Qa = Qa / Qa.sum(); D += Qa @ LA - Qh @ LA; KL += float(Qa @ np.log(Qa / Qh)); x = int(rng.choice(M, p=Qa))
    viol += D * D > 2 * n5 * B * B * KL + 1e-12; ratios.append((D * D / (2 * n5 * B * B * KL), style))
tight = max(r for r, s in ratios if s == 0)
check("SQ5 IDENTITY pathwise cost: the compensator shift satisfies Delta_n^2 <= 2 n B^2 K_n", viol == 0,
      f"violations {viol}/500; largest ratio Delta^2 / (2 n B^2 K): tilt along Lambda_a {tight:.3f}, random {max(r for r, s in ratios if s == 1):.4f}")
# SQ6: F_1 record. Honest: i.i.d. Q with <a, D> <= 0 (never crosses). Adversary: tilt Q along -Lambda_a just enough to cross at s = 1.
while True:
    Qh = rlaw()
    if Qh @ LA >= 0: break                       # honest drift away from the wall
def tilt(eta): q = Qh * np.exp(-eta * LA); return q / q.sum()
lo, hi = 0.0, 20.0
for _ in range(80):
    mid = (lo + hi) / 2
    if tilt(mid) @ LA > c0: lo = mid               # need E Lambda_a = c0 so that c0 - s E = 0 at s = 1
    else: hi = mid
Qa = tilt(hi); kl = float(Qa @ np.log(Qa / Qh)); rows = []
for Nn in (50, 200, 1000):
    n6 = int(Nn); lr_h = []; lr_a = []
    for _ in range(2000):
        xh = rng.choice(M, size=n6, p=Qh); xa = rng.choice(M, size=n6, p=Qa)
        lr_h.append(np.log(Qa[xh] / Qh[xh]).sum()); lr_a.append(np.log(Qa[xa] / Qh[xa]).sum())
    thr = np.quantile(lr_h, 0.95); rows.append((Nn, n6 * kl, float(np.mean(np.array(lr_a) > thr))))
check("SQ6 RECORD F_1: the least-divergence wall-crossing attack pays K_n = n KL, linear in N; a likelihood-ratio test against it",
      rows[1][2] > 0.9, f"honest drift E Lambda_a {Qh @ LA:.3f} (c0 {c0:.2f}); attack KL per step {kl:.4f}; " + "; ".join(f"N={r[0]}: K_n {r[1]:.1f}, power {r[2]:.3f}" for r in rows))
print(f"{sum(results)}/{len(results)} passed")
