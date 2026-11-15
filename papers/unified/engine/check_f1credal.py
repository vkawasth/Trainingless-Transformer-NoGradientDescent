"""Worked example for sec:credalflow on F_1 (ex:F1): a credal Kahler class alpha in [0.36, 0.45], just above the outcome wall alpha = 1/3,
under evidence. New script; it modifies nothing; uses pce/kahlerbridge.py, pce/credalflow.py.
Four terms with circuit a = (2, -1, 2, -3); term k carries a law P_k on three outcomes; the posterior over terms after n observations is
p_k proportional to e^(N lambda_k) prod_t P_k(x_t), so evidence tilts log p by sum_t log P_k(x_t) (the rate form: -n H_k(Q)). The wall
function c = -<a, lambda_eff> equals -(1/N) <a, log p> (sum of a is 0): a log-linear payoff. P_1 = (0.6, 0.3, 0.1), P_2 = (0.2, 0.5, 0.3),
P_3 = (0.5, 0.2, 0.3), P_4 = (0.2, 0.3, 0.5); honest data Q = (0.5, 0.3, 0.2), drift d = E_Q Lambda_a = 0.872.
  FK1 IDENTITY  rigid motion: along any data stream every member's wall function moves by the same -S_n / N, so the interval of wall
                values keeps its width exactly (the Hilbert isometry, in lambda coordinates)
  FK2 IDENTITY  the wall decision is log-linear, so its transit is exactly linear in the rate limit: indeterminate for s = n/N in
                (c_min / d, c_max / d], duration (c_max - c_min) / d = 0.31; with data the entry and exit times converge at rate
                N^-1/2 (slope in [-0.7, -0.3])
  FK3 IDENTITY  a linear payoff on the same credal set (thm:crossing): g = p_4 - p_3 ('the posterior favours term 4 over term 3'), with
                the credal prior the hull of the two endpoint laws: at most one switch (Descartes), eventual sign that of the best-fitting
                term with non-zero payoff (term 3: soft evidence becomes hard), and the explicit bound of thm:crossing(iv) never beaten.
                First run: PASS, but with the bound 65.0 against an actual 1.92, because the levels included term 1, whose payoff is 0
                and whose likelihood is within 0.007 of term 3's. States with zero payoff do not enter the exponential sum, so the
                bound is now taken over the states with c_i != 0 (thm:crossing restated accordingly)
  FK4 IDENTITY  valuation limit (cor:tropcross): g's last indeterminate time equals the tropical transit time alpha_max / (H_4 - H_3)
                = 1.920 exactly at every N; the wall's transit (FK2) and g's are different decisions with different windows.
                First run, as specified ('tends to it at rate 1/N'): FAIL. The time was 1.9203 at N = 20, 80 and 320 (no convergence
                because there is no error), and N = 1280 underflowed. A payoff that compares two terms only has the sign of
                log p_4 - log p_3, which is log-linear: its crossing is exact at every N, like the wall's. The 1/N rate of
                cor:tropcross is for payoffs with three or more terms (CR10). Restated; N <= 320
  FK5 IDENTITY  cost (prop:seqval(iii)): an honest stream Q' = (0.3, 0.4, 0.3) keeps the set on the positive side (drift -0.439); the
                least-divergence tilt that makes the whole set negative by s = 1 pays K_n = n KL, at least the bound Delta_n^2 / (2 n B^2)
  FK6 RECORD    the example's numbers: transit window, crossing times by N, attack divergence per step and in total, ratio to the bound,
                and the power of a likelihood-ratio test against the attack at N = 200 and 1000
"""
import os, sys, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import kahlerbridge as KB, credalflow as CF
results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
rng = np.random.default_rng(20261120)
a = np.array([2, -1, 2, -3], float); P = np.array([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3], [0.5, 0.2, 0.3], [0.2, 0.3, 0.5]]); LP = np.log(P)
LA = a @ LP; B = float(np.abs(LA).max()); Q = np.array([0.5, 0.3, 0.2]); d = float(Q @ LA); H = -(LP @ Q)
AL = (0.36, 0.45); lam = {al: np.array(KB.F1(al)[1], float) for al in AL}; c0 = {al: float(-a @ lam[al]) for al in AL}
cmin, cmax = min(c0.values()), max(c0.values())
# FK1 rigid motion
worst = 0.0
for _ in range(200):
    N = 300.0; x = rng.choice(3, size=600, p=Q); S = np.cumsum(LA[x])
    w = [(c0[AL[1]] - S / N) - (c0[AL[0]] - S / N)]; worst = max(worst, float(np.abs(w[0] - (cmax - cmin)).max()))
    lam_eff = {al: lam[al][None, :] + np.cumsum(LP[:, x].T, axis=0) / N for al in AL}
    cc = {al: -(lam_eff[al] @ a) for al in AL}; worst = max(worst, float(np.abs((cc[AL[1]] - cc[AL[0]]) - (cmax - cmin)).max()))
check("FK1 IDENTITY the interval of wall values translates rigidly: its width is constant along every data stream", worst < 1e-12, f"200 streams; max width change {worst:.1e}; width {cmax - cmin:.3f}")
# FK2 exact transit in the rate limit; data streams converge at N^-1/2
s_in, s_out = cmin / d, cmax / d; errs = []
Ns = [100, 400, 1600, 6400]
for N in Ns:
    e = []
    for _ in range(200):
        n = int(1.5 * s_out * N) + 10; S = np.cumsum(LA[rng.choice(3, size=n, p=Q)]); s = np.arange(1, n + 1) / N
        enter = s[np.argmax(cmin - S / N < 0)]; leave = s[np.argmax(cmax - S / N < 0)]; e.append(max(abs(enter - s_in), abs(leave - s_out)))
    errs.append(float(np.median(e)))
sl = np.polyfit(np.log(Ns), np.log(errs), 1)[0]
check("FK2 IDENTITY the wall decision is indeterminate exactly for s in (c_min/d, c_max/d] in rates; data converge at N^-1/2", -0.7 <= sl <= -0.3,
      f"window ({s_in:.3f}, {s_out:.3f}], duration {s_out - s_in:.3f} = width {cmax - cmin:.2f} / drift {d:.3f}; median error {', '.join(f'{x:.4f}' for x in errs)}; slope {sl:.2f}")
# FK3 linear payoff g = p_4 - p_3 on the hull of the endpoint priors
cg = np.array([0.0, 0.0, -1.0, 1.0])
def prior(al, N): l = N * lam[al]; l -= l.max(); p = np.exp(l); return p / p.sum()
ok = True; info = []
for N in (5, 10, 20):
    V = [prior(al, N) for al in AL]; av = -N * H                                                # tilt per unit s
    for v in V:
        ok &= CF.sign_changes(CF.level_coeffs(v, cg, av)) <= 1 and CF.lex_sign(v, cg, av) == -1
        ts = np.linspace(0, 6, 3001); vals = [float(cg @ CF.tilt(v, av, t)) for t in ts]; ok &= CF.sign_changes(vals) <= 1
    Bd = CF.crossing_bound(V, -cg, av)                                                          # determinate negative = -g positive
    tl = CF.last_nonpositive(V, -cg, av, 8.0); ok &= Bd is not None and (tl is None or tl <= Bd + 1e-9); info.append(f"N={N}: last {tl:.3f} <= bound {Bd:.3f}")
check("FK3 IDENTITY thm:crossing on F1: at most one switch; eventual sign that of term 3 (hard evidence on the best fit); the bound holds", ok, "; ".join(info))
# FK4 valuation limit
t_trop = max(al for al in AL) / (H[3] - H[2]); win = (min(AL) / (H[3] - H[2]), t_trop); rows = []
for N in (5, 20, 80, 320):
    V = [prior(al, N) for al in AL]; tl = CF.last_nonpositive(V, -cg, -N * H, t_trop + 2, n=8000); rows.append((N, tl, abs(tl - t_trop)))
check("FK4 IDENTITY a two-term payoff is log-linear: g's crossing time equals the tropical transit time alpha_max / (H_4 - H_3) at every N", max(r[2] for r in rows) < 1e-6,
      f"tropical window ({win[0]:.3f}, {win[1]:.3f}]; " + "; ".join(f"N={r[0]}: {r[1]:.6f}" for r in rows) + f"; wall window ({s_in:.3f}, {s_out:.3f}]")
# FK5 cost of forcing the decision
Qh = np.array([0.3, 0.4, 0.3]); dh = float(Qh @ LA); target = cmax + 0.05                    # need E_Q' Lambda_a = target at s = 1
def tilt_to(m):
    lo, hi = 0.0, 10.0
    for _ in range(100):
        mid = (lo + hi) / 2; q = Qh * np.exp(mid * LA); q /= q.sum()
        if q @ LA < m: lo = mid
        else: hi = mid
    q = Qh * np.exp(hi * LA); return q / q.sum()
Qa = tilt_to(target); kl = float(Qa @ np.log(Qa / Qh)); Dstep = float(Qa @ LA - dh); boundstep = Dstep ** 2 / (2 * B * B)
check("FK5 IDENTITY forcing the whole set across the wall by s = 1 costs n KL(Q'||Q) >= Delta_n^2 / (2 n B^2) (prop:seqval(iii))", kl >= boundstep,
      f"honest drift {dh:.3f}; attack drift {Qa @ LA:.3f}; KL per step {kl:.4f} >= bound {boundstep:.4f} (ratio {kl / boundstep:.1f})")
pw = []
for N in (200, 1000):
    lh = [float(np.log(Qa[x] / Qh[x]).sum()) for x in (rng.choice(3, size=N, p=Qh) for _ in range(3000))]
    la_ = [float(np.log(Qa[x] / Qh[x]).sum()) for x in (rng.choice(3, size=N, p=Qa) for _ in range(3000))]
    pw.append((N, N * kl, float(np.mean(np.array(la_) > np.quantile(lh, 0.95)))))
check("FK6 RECORD F1 credal example: transit, crossing times, attack cost and detectability", True,
      f"Q' = ({', '.join(f'{x:.3f}' for x in Qa)}); " + "; ".join(f"N={N}: total divergence {K:.1f}, power {p:.3f}" for N, K, p in pw))
print(f"{sum(results)}/{len(results)} passed")
