"""Checks for 'Braids, credal sets and conditioning' (module pce/credalbraid.py).
  CP1 IDENTITY  projection and conditioning: for every law, Pi_*(P|C) and (Pi_*P)|Pi(C) = Pi_*(P|Chat) differ in total variation by exactly
                w TV(Pi_*(P|C), Pi_*(P|Chat \\ C)), w = P(Chat \\ C)/P(Chat); the credal sets are at Hausdorff distance at most max w; they
                coincide for every credal set when C is Pi-saturated or Pi(C) is one point, and otherwise a two-point law separates them
  CP2 IDENTITY  the braid as the projection (brokerage histories, Pi = triple invariant): conditioning on 'mu = 1' commutes with
                projecting; conditioning on 'visitor 1 opens first', which the braid does not determine, does not, within the bound
  CP3 IDENTITY  under a circular law the angle theta1 of the first difference is uniform and independent of the phase difference psi, so
                P(mu = 1) = E overlap(S1, S2 + psi); checked against the Goodman density and against braids read from regularised
                histories under laws with a fixed psi
  CP4 IDENTITY  the envelope of P(mu = 1) over all circular laws is the Frechet interval of the two sector probabilities, attained by laws
                with psi = 0 and psi = pi; the pairwise probabilities are the same for every circular law; real-correlation Gaussian laws
                give a strict sub-interval
  CP5 IDENTITY  shared nuisance: when honest and coordinated histories share the timing tilt lambda, the credal posterior of 'coordinated'
                given the full history equals that given the braid, history by history
  CP6 FINDING   separate nuisances: the braid gives narrower posterior intervals than untyped precedence, typed precedence or the full
                history; with the coalition's relay preference pinned (beta in [0.5, 2]) the braid dilates at fewer values than any finer
                statistic; unpinned (beta in [-1, 2]) every informative statistic dilates at every value
"""
import os, sys, json, itertools
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import warnings; warnings.filterwarnings("ignore")
from pce import credalbraid as CB, brokerage as BR, pseudoprob as PP, tiebraid as TB
results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
rng = np.random.default_rng(31)
tv = lambda x, y: 0.5 * np.abs(np.asarray(x) - np.asarray(y)).sum()

# CP1
ok_id = ok_bd = ok_sat = ok_sep = True; inst = 0; gaps = []
for _ in range(150):
    n, m = 12, 4; Pi = rng.integers(0, m, n).tolist(); V = rng.dirichlet(np.ones(n) * 0.7, 4)
    C = sorted(rng.choice(n, int(rng.integers(2, 7)), replace=False).tolist()); Ch = CB.saturate(C, Pi); extra = sorted(set(Ch) - set(C))
    A = CB.push(CB.cond(V, C), Pi, m); B = CB.push(CB.cond(V, Ch), Pi, m)
    for v, a, b in zip(V, A, B):
        w = v[extra].sum() / v[Ch].sum(); rest = CB.push(CB.cond(v, extra), Pi, m) if extra else None
        ok_id &= abs(tv(a, b) - (w * tv(a, rest[0]) if extra else 0.0)) < 1e-12
    d = CB.hausdorff(A, B); bd = CB.gap_bound(V, C, Pi); ok_bd &= d <= bd + 1e-9; gaps.append((d, bd))
    if len({Pi[h] for h in C}) == 1 or not extra: ok_sat &= d < 1e-9
    elif extra:
        h2 = extra[0]; h1 = next(h for h in C if Pi[h] != Pi[h2]) if any(Pi[h] != Pi[h2] for h in C) else None
        if h1 is not None:
            P2 = np.zeros(n); P2[h1] = P2[h2] = 0.5
            ok_sep &= tv(CB.push(CB.cond(P2, C), Pi, m)[0], CB.push(CB.cond(P2, Ch), Pi, m)[0]) > 0.4
    inst += 1
check("CP1 IDENTITY projection vs conditioning: exact per-law defect, Hausdorff bound, commutation iff saturated or one value", ok_id and ok_bd and ok_sat and ok_sep,
      f"{inst} random instances; largest ratio distance/bound {max(d / b for d, b in gaps if b > 0):.3f}")

# CP2
H = BR.interleavings(2); Pr = CB.projections(H); Pi, keys = Pr["braid"]; m = len(keys)
V = np.array([L for lam in (0.8, 1.25) for beta in (0.0, 2.0) for L in [CB.laws(H, lam, beta)[1]]])
C_mu = [i for i, h in enumerate(H) if BR.mu123(h) == 1]; C_open = [i for i, h in enumerate(H) if BR.stats(h)["first_open"] == "1"]
d_mu = CB.hausdorff(CB.push(CB.cond(V, C_mu), Pi, m), CB.push(CB.cond(V, CB.saturate(C_mu, Pi)), Pi, m))
d_op = CB.hausdorff(CB.push(CB.cond(V, C_open), Pi, m), CB.push(CB.cond(V, CB.saturate(C_open, Pi)), Pi, m)); b_op = CB.gap_bound(V, C_open, Pi)
check("CP2 IDENTITY braid projection: 'mu = 1' commutes; 'visitor 1 opens first' does not, within the bound", d_mu < 1e-9 and 0 < d_op <= b_op + 1e-9,
      f"distance {d_op:.3f}, bound {b_op:.3f}")

# CP3
Zv, a1, a2 = CB.arcs_broker()
def mc_mu(om):
    k = 0
    for o in om:
        w, init, _, _ = TB.regularised(Zv, o); lk = TB.linking(w, init, 3); k += all(v == 0 for v in lk) and TB.triple(w, init, 0, 1, 2) == 1
    return k / len(om)
S = np.array([[1, 0, 0.6], [0, 1, 0], [0.6, 0, 1]]); om = PP.sample_circular_gaussian(S, 20000, rng)
t1 = np.angle(om[:, 0] - om[:, 1]); t2 = np.angle(om[:, 1] - om[:, 2]); ps = np.angle(np.exp(1j * (t1 - t2)))
q1 = np.digitize(t1, np.linspace(-np.pi, np.pi, 5)[1:-1]); q2 = np.digitize(ps, np.linspace(-np.pi, np.pi, 5)[1:-1])
J = np.array([[np.mean((q1 == i) & (q2 == j)) for j in range(4)] for i in range(4)]); indep = np.abs(J - np.outer(J.sum(1), J.sum(0))).max()
cov = S[0, 1] - S[0, 2] - S[1, 1] + S[1, 2]; rho = cov / np.sqrt((S[0, 0] + S[1, 1] - 2 * S[0, 1]) * (S[1, 1] + S[2, 2] - 2 * S[1, 2]))
grid = np.linspace(-np.pi, np.pi, 721)[:-1] + np.pi / 720; ov = np.array([CB.overlap(a1, a2, p, 4000) for p in grid])
mix = float(np.sum(ov * PP.phase_diff_density(grid, rho)) * (2 * np.pi / 720)); dens = float(PP.joint_sector_prob(a1, a2, rho, n=800))
fixed = {}
for psi in (0.0, np.pi / 2, np.pi):
    z0 = np.array([0, -1, -1 - np.exp(-1j * psi)], complex)                 # z0-z1 = 1, z1-z2 = e^{-i psi}: theta1 - theta2 = psi
    phi = rng.uniform(0, 2 * np.pi, 1500); fixed[round(psi, 3)] = (mc_mu(np.exp(1j * phi)[:, None] * z0[None, :]), CB.overlap(a1, a2, psi))
ok = indep < 0.01 and abs(mix - dens) < 0.005 and all(abs(a - b) < 0.03 for a, b in fixed.values())
check("CP3 IDENTITY circular laws: theta1 independent of psi; P(mu=1) = E overlap(psi), against the density and against fixed-psi braids", ok,
      f"independence {indep:.4f}; mixture {mix:.4f} vs density {dens:.4f}; fixed psi (braid MC, overlap): { {k: (round(a, 3), round(b, 3)) for k, (a, b) in fixed.items()} }")

# CP4
s1, s2 = (a1[1] - a1[0]) / (2 * np.pi), (a2[1] - a2[0]) / (2 * np.pi); F = CB.frechet(s1, s2); lo, hi, plo, phi_ = CB.envelope_circular(a1, a2); ENV = [lo, hi]
pair = {}
for psi in (0.0, np.pi):
    z0 = np.array([0, -1, -1 - np.exp(-1j * psi)], complex); phi = rng.uniform(0, 2 * np.pi, 4000); om = np.exp(1j * phi)[:, None] * z0[None, :]
    pair[round(psi, 3)] = (float(np.mean([TB._sector(o[0] - o[1], TB.walls(Zv)[(0, 1)]) == (1 if a1[1] < np.pi else 0) for o in om])),)
cs = np.linspace(-0.95, 0.95, 39); gauss = []
for c in cs:
    S = np.array([[1, 0, c], [0, 1, 0], [c, 0, 1]]); cov = S[0, 1] - S[0, 2] - S[1, 1] + S[1, 2]
    r = cov / np.sqrt((S[0, 0] + S[1, 1] - 2 * S[0, 1]) * (S[1, 1] + S[2, 2] - 2 * S[1, 2])); gauss.append(float(PP.joint_sector_prob(a1, a2, r, n=400)))
ok = abs(lo - F[0]) < 0.002 and abs(hi - F[1]) < 0.002 and abs(fixed[0.0][0] - F[1]) < 0.03 and abs(fixed[round(np.pi, 3)][0] - F[0]) < 0.01
ok &= all(abs(v[0] - s1) < 0.03 for v in pair.values()) and F[0] + 0.01 < min(gauss) and max(gauss) < F[1] - 0.01
check("CP4 IDENTITY envelope over circular laws = Frechet interval, attained at psi = 0 and pi; pairwise precise; Gaussian family a strict sub-interval", ok,
      f"sector probabilities {s1:.4f}, {s2:.4f}; Frechet [{F[0]:.3f}, {F[1]:.3f}]; envelope [{lo:.3f}, {hi:.3f}]; real-correlation Gaussian [{min(gauss):.3f}, {max(gauss):.3f}]; pairwise under psi = 0, pi: {pair}")

# CP5
lams = np.linspace(0.8, 1.25, 7); betas = np.linspace(0.5, 2.0, 7); ok = True
for lam in lams:
    for beta in betas:
        p, q = CB.laws(H, lam, beta); mu = np.array([BR.mu123(h) for h in H])
        post_full = 0.5 * q / (0.5 * q + 0.5 * p)
        for y in set(mu):
            idx = mu == y; post_br = 0.5 * q[idx].sum() / (0.5 * q[idx].sum() + 0.5 * p[idx].sum()); ok &= np.allclose(post_full[idx], post_br)
check("CP5 IDENTITY shared nuisance: posterior given the full history = posterior given the braid, for every law", ok, f"{len(lams) * len(betas)} laws")

# CP6
res = {}
for name, br in (("pinned", np.linspace(0.5, 2, 7)), ("unpinned", np.linspace(-1, 2, 13))):
    res[name] = {}
    for k in ("untyped", "typed", "braid", "full"):
        Pi, keys = Pr[k]; lo, hi = CB.posterior_interval(H, Pi, len(keys), lams, lams, br)
        p_ref = CB.push(CB.laws(H, 1.0, 0.0)[0], Pi, len(keys))[0]                     # weight values by the uniform honest law
        res[name][k] = dict(values=len(keys), width=round(float(np.sum(p_ref * (hi - lo))), 3), dilated=int(np.sum((lo < 0.5 - 1e-9) & (hi > 0.5 + 1e-9))),
                            dilated_mass=round(float(np.sum(p_ref * ((lo < 0.5 - 1e-9) & (hi > 0.5 + 1e-9)))), 3))
R = res["pinned"]; U = res["unpinned"]
ok = all(R["braid"]["width"] < R[k]["width"] for k in ("untyped", "typed", "full")) and all(R["braid"]["dilated_mass"] < R[k]["dilated_mass"] for k in ("untyped", "typed", "full"))
ok &= all(U[k]["dilated"] == U[k]["values"] for k in U)
check("CP6 FINDING separate nuisances: the braid gives the narrowest intervals and least dilation; unpinned relay preference dilates everywhere", ok, f"{res}")
json.dump(dict(posterior=res, frechet=F, envelope=ENV, gauss=[min(gauss), max(gauss)], fixed_psi={str(k): v for k, v in fixed.items()}),
          open(os.path.join(HERE, "credalbraid_summary.json"), "w"), indent=1)
print(f"{sum(results)}/{len(results)} passed")
