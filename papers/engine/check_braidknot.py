"""Checks for 'What braids and knots carry' and 'The mathematical chain' (companion note; modules pce/braidknot.py, pce/bundlebraid.py,
pce/pdknot.py). IDENTITY = an exact instance of a statement proved or cited in the text; FINDING = a computed fact about a toy;
FAILURE = a check of a construction that the construction does not pass, recorded because the text reports it.
  BK1  IDENTITY  Milnor's invariants of the closures (Artin action and Magnus expansion): A B A^-1 B^-1 (A = sigma_1^2, B = sigma_2^2) has all
                 linking numbers 0 and second-order coefficient +-1, as the Borromean braid (sigma_1 sigma_2^-1)^3 does; B A B^-1 A^-1 reverses
                 the sign; A A^-1 B B^-1 and A B B^-1 A^-1 are trivial braids
  BK2  FINDING   in the planar realisation (full turns about midpoints), the signed area enclosed by the winding angles (theta_12, theta_23),
                 divided by (2 pi)^2, equals the second-order coefficient for all four orderings, and stays within 0.01 under jitter that
                 keeps the strands apart; net windings and path length are equal for all four
  BK3  IDENTITY  closure invariants (Jones by the state sum, Alexander and determinant by the reduced Burau formula): right trefoil, figure-eight
                 (det 5), Borromean (det 16), and the commutator's closure has the Borromean Jones and Alexander polynomials; (sigma_1 sigma_2^-1)^2
                 closes to 1 component and sigma_1 sigma_2^-2 sigma_1 to 3, with equal letter counts
  BK4  IDENTITY  the parity pair on the figure-eight pseudo-diagram: q_+ and q_- agree on every moment of order <= 3; P(knotted) differs by theta/2
                 and E[V] by 2 theta times the degree-4 Walsh coefficient of V; the writhe (degree 1) has equal expectations
  BK5  IDENTITY  coherent histories give the trivial bundle braid: drift, shock, matched drift and pump-and-retreat (every snapshot coherent)
                 give the empty word under every convention (ring 0.005-0.08, projection angle -0.02..0.05) and under noise
  BK6  FAILURE   the bundle braid is not determined by the data for the liar, framing and cascade histories: several braid elements and several
                 closure component counts across conventions; the data tie the strands (equal suspicion heights) at most steps
  BK7  FINDING   suspect sets (strands within 0.01 of the lowest height) are the same under every convention and under noise; liar: R2;
                 framing and cascade: R2, then {S3, R1}; lockstep coalitions: {S4} after the inspection; honest coherent histories: none
  BK8  FINDING   the central table: smallest explanation (<= 3, 4 = none), number of changes of suspect set, braid determined or not
  BK9  IDENTITY  protection theorem on the bridge: for the pair {S2, S3}, the anchor (X3, X5) leaves an anchored range excluding 1/2 and has
                 separation margin rho > 0; the anchor (X2, X4) has rho = 0; with delta = D / 7 the earliest flip is at step 7
  BK10 IDENTITY  noisy version: with inspection noise eta < rho / 2 and flag threshold kappa = rho / 2, the honest anchor distance is <= eta
                 and every coherent far-side configuration has anchor distance >= rho - eta > kappa (checked on random far-side laws)
  BK11 FINDING   co-revision network (time-aware, node-normalised, after Iannucci et al.): the group of sources that repeatedly revise
                 together is exactly the coalition {S2, S3, R1, R2} in the efficient, stealthy and pump-and-retreat histories, where suspect sets
                 name nobody or the honest S4; it names the same group in the matched honest drift (identical reports), all seven sources in the
                 honest drift, and nobody in the shock, the contextual world and the single-liar histories
"""
import os, sys, json, math, itertools
import numpy as np
import sympy as sp
from scipy.optimize import linprog
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import braidknot as BK, bundlebraid as BB, pdknot as PD, watchtoy as WT, bridgetoy as BT, tamper as T, miscal as MC
results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
A_, B_, a_, b_ = BK.A, BK.B, BK.AI, BK.BI
ORD = {"A B a b": A_ + B_ + a_ + b_, "B A b a": B_ + A_ + b_ + a_, "A a B b": A_ + a_ + B_ + b_, "A B b a": A_ + B_ + b_ + a_}

# BK1
M = {k: BK.milnor(w) for k, w in ORD.items()}; borr = BK.milnor([1, -2] * 3)
zero_lk = all(all(v == 0 for d in m[0].values() for v in d.values()) for m in list(M.values()) + [borr])
c = {k: m[1][3][0] for k, m in M.items()}
check("BK1 IDENTITY Milnor: zero linking, commutator +-1 like the Borromean braid, reorderings trivial",
      zero_lk and c["A B a b"] == borr[1][3][0] == 1 and c["B A b a"] == -1 and BK.same_braid(ORD["A a B b"], [], 3)
      and BK.same_braid(ORD["A B b a"], [], 3) and not BK.same_braid(ORD["A B a b"], [], 3),
      f"coefficient of X1 X2 in the longitude of strand 3: { {**c, 'Borromean': borr[1][3][0]} }; all longitudes {M['A B a b'][1]}")

# BK2
seqs = {k: [BK.TWIST[x] for x in k.split()] for k in ORD}
area, rob, same = {}, 0.0, True
for k, s in seqs.items():
    P = BK.commutator_path(s); w, ar, dmin = BK.winding_area(P); area[k] = round(ar, 4)
    same &= all(abs(v) < 1e-9 for v in w.values()) and dmin > 0.9
    for sd in range(5):
        Pj = BK.commutator_path(s, jitter=0.03, seed=sd); _, arj, dj = BK.winding_area(Pj); rob = max(rob, abs(arj - ar)); same &= dj > 0.5
plen = {k: round(float(np.abs(np.diff(BK.commutator_path(s), axis=0)).sum()), 3) for k, s in seqs.items()}
check("BK2 FINDING area of the winding angles = second-order coefficient; robust to jitter; coarse summaries equal",
      all(abs(area[k] - c[k]) < 1e-3 for k in ORD) and rob < 0.01 and same and len(set(plen.values())) == 1,
      f"area/(2pi)^2 {area}; largest change under jitter {rob:.4f}; path lengths {plen}")

# BK3
t = sp.Symbol("t")
J = {k: BB.jones_closure(w, n) for k, (w, n) in {"trefoil": ([1, 1, 1], 2), "fig8": ([1, -2, 1, -2], 3), "borr": ([1, -2] * 3, 3),
                                                 "comm": (ORD["A B a b"], 3), "memoB": ([1, -2, -2, 1], 3)}.items()}
alex_eq = sp.simplify(BB.alexander([1, -2] * 3, 3) / BB.alexander(ORD["A B a b"], 3))
ok = (PD.classify(J["trefoil"]) == "right trefoil" and PD.classify(J["fig8"]) == "figure-eight" and abs(BB.burau_det([1, -2, 1, -2], 3) - 5) < 1e-9
      and abs(BB.burau_det([1, -2] * 3, 3) - 16) < 1e-9 and sp.simplify(J["comm"] - J["borr"]) == 0 and alex_eq.is_number
      and len(BK.cycles(BK.permutation([1, -2, 1, -2], 3))) == 1 and len(BK.cycles(BK.permutation([1, -2, -2, 1], 3))) == 3)
check("BK3 IDENTITY closure invariants; the commutator's closure has the Borromean polynomials; memo braids 1 vs 3 components", ok,
      f"V(Borromean) = V(commutator) = {J['borr']}; Alexander ratio {alex_eq}; det fig8 5, Borromean 16; V(memo B) = {J['memoB']}")

# BK4
th = 0.6; qp, qm = BK.parity_laws(th)
low = max(abs(BK.moments(qp, S) - BK.moments(qm, S)) for k in range(4) for S in itertools.combinations(range(4), k))
Vs = {e: PD.jones(PD.FIG8, PD.resolve(PD.FIG8, e)) for e in BK.SIGNS4}
knot = np.array([0.0 if sp.simplify(Vs[e] - 1) == 0 else 1.0 for e in BK.SIGNS4])
top = sp.expand(sum(Vs[e] * np.prod(e) for e in BK.SIGNS4) / 16)
dV = sp.expand(sum(sp.Rational(2, 16) * sp.Rational(3, 5) * int(np.prod(e)) * Vs[e] for e in BK.SIGNS4))   # (q_+ - q_-)(e) = 2 theta prod(e) / 16, theta = 3/5
wr = np.array([sum(e) for e in BK.SIGNS4])                                   # writhe of the resolution = sum of its crossing signs
check("BK4 IDENTITY parity pair: moments of order <= 3 equal; P(knotted) differs by theta/2; E[V] by 2 theta Vhat(1234); writhe equal",
      low < 1e-12 and abs((qp - qm) @ knot - th / 2) < 1e-12 and sp.simplify(dV - 2 * sp.Rational(3, 5) * top) == 0 and abs(float(sum((a - b) * float(Vs[e].subs(sp.Symbol('t'), 2)) for a, b, e in zip(qp, qm, BK.SIGNS4))) - float(dV.subs(sp.Symbol('t'), 2))) < 1e-12 and abs((qp - qm) @ wr) < 1e-12 and top != 0,
      f"theta {th}; largest difference of lower moments {low:.1e}; P+(knotted) - P-(knotted) = {(qp - qm) @ knot:.3f}; Vhat(1234) = {top}")

# bundle features (the expensive part): ten histories, clean and with noise 0.003
W = WT.setup(); g = W["gB"]
NAMES = ["drift", "shock", "matched", "pump_retreat", "contextual", "redundant_liar", "framing", "cascade", "efficient", "stealthy"]
F = {}
for nm in NAMES:
    H = BB.history(nm, W); F[nm] = BB.features(H, g); F[nm + "@noise"] = BB.features(H, g, noise=0.003, seed=1)
CONV = [(e, p) for e in (0.005, 0.01, 0.02, 0.03, 0.04, 0.08) for p in (0.0, 0.01, 0.02, -0.02, 0.05)]
def key(w): img = BK.artin(w, 7); return tuple(tuple(img[j]) for j in sorted(img))
def suspects(S):
    out = []
    for s in S: m = s.min(); out.append(tuple(x for x, v in zip(BB.BASE, s) if v <= m + 0.01) if s.max() > 0.05 else ())
    return out
summ = {}
for nm in NAMES:
    S, V, rows = F[nm]; Sn, Vn, _ = F[nm + "@noise"]
    els = {key(BB.braid(BB.points(S, V, e), p)[0]) for e, p in CONV}; els_n = key(BB.braid(BB.points(Sn, Vn, 0.02), 0.0)[0])
    comps = sorted({len(BK.cycles(BK.permutation(BB.braid(BB.points(S, V, e), p)[0], 7))) for e, p in CONV})
    gen = all(BB.braid(BB.points(S, V, e), p)[2] for e, p in CONV)
    ties = sum(1 for s in S if any(abs(a - b) < 1e-6 and max(a, b) > 1e-6 for i, a in enumerate(s) for b in s[i + 1:]))
    sus = suspects(S); changes = sum(1 for a, b in zip(sus, sus[1:]) if a != b)
    summ[nm] = dict(elements=len(els), noise_same=els_n in els and len(els) == 1, components=comps, generic=gen, ties=ties,
                    suspects=sus, suspects_noise=suspects(Sn), changes=changes, min_expl=[r["min_expl"] for r in rows],
                    naive_wrong=[(r["naive"] >= 0.5) != (r["truth"] >= 0.5) for r in rows], trivial=els == {key([])})
json.dump({k: {kk: (vv if not isinstance(vv, np.generic) else vv.item()) for kk, vv in v.items()} for k, v in summ.items()},
          open(os.path.join(HERE, "bundlebraid_summary.json"), "w"), indent=1, default=lambda o: o if not isinstance(o, np.generic) else o.item())

# BK5
coh = ["drift", "shock", "matched", "pump_retreat"]
check("BK5 IDENTITY coherent histories give the trivial bundle braid under every convention and under noise",
      all(summ[n]["trivial"] and summ[n]["noise_same"] and summ[n]["generic"] for n in coh),
      f"{len(CONV)} conventions; empty word in every case for {coh}")

# BK6
bad = ["redundant_liar", "framing", "cascade"]
check("BK6 FAILURE the bundle braid is not determined by the data when strands tie (liar, framing, cascade)",
      all(summ[n]["elements"] >= 2 for n in bad) and all(summ[n]["ties"] >= 12 for n in bad),
      "; ".join(f"{n}: {summ[n]['elements']} braid elements, components {summ[n]['components']}, tied steps {summ[n]['ties']}/21" for n in bad))

# BK7
exp = {"redundant_liar": [(), ("R2",)], "framing": [(), ("R2",), ("S3", "R1")], "cascade": [(), ("R2",), ("S3", "R1")],
       "efficient": [(), ("S4",)], "stealthy": [(), ("S4",)], "drift": [()], "shock": [()], "matched": [()], "pump_retreat": [()]}
def runs(seq): return [a for i, a in enumerate(seq) if i == 0 or a != seq[i - 1]]
ok = all(runs(summ[n]["suspects"]) == v for n, v in exp.items()) and all(summ[n]["suspects"] == summ[n]["suspects_noise"] for n in NAMES)
check("BK7 FINDING suspect sets are convention-free and noise-stable; blame passes from the liar R2 to the honest {S3, R1} under framing",
      ok, "; ".join(f"{n}: {' -> '.join(str(list(x)) if x else 'none' for x in runs(summ[n]['suspects']))}" for n in NAMES))

# BK8
tab = {n: (sorted(set(summ[n]["min_expl"])), summ[n]["changes"], "trivial" if summ[n]["trivial"] else ("not determined" if summ[n]["elements"] > 1 else "one element"),
           sum(summ[n]["naive_wrong"])) for n in NAMES}
check("BK8 FINDING the central table (explanation sizes, suspect-set changes, braid, steps with the naive decision wrong)",
      tab["cascade"][0] == [0, 1, 3, 4] and tab["redundant_liar"][0] == [0, 1] and tab["contextual"][1] == 0 and tab["efficient"][2] == "trivial",
      f"{tab}")

# BK9
P, gB = W["P"], W["gB"]; hon = MC.reports(P); Hcells = [hon["S1"], hon["S4"], hon["S5"]]
def rho(q):
    Mq, keys = T.marg(5, q); y = Mq @ P; Mh, bh = T.system(5, Hcells); r = Mq.shape[0]
    cst = np.r_[np.zeros(32), np.ones(r)]
    Aub = np.vstack([np.hstack([Mq, -np.eye(r)]), np.hstack([-Mq, -np.eye(r)]), np.r_[gB, np.zeros(r)][None, :]])
    bub = np.r_[y, -y, 0.5]; Aeq = np.hstack([Mh, np.zeros((Mh.shape[0], r))])
    res = linprog(cst, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=bh, bounds=[(0, None)] * (32 + r), method="highs"); return res.fun
rho35, rho24 = rho((2, 4)), rho((1, 3))
I35 = T.interval(5, Hcells + [("Q", (2, 4), BT.table(P, (2, 4)))], gB); I24 = T.interval(5, Hcells + [("Q", (1, 3), BT.table(P, (1, 3)))], gB)
pth = MC.coalition_endpoint(P, ["S2", "S3"], Hcells, gB, 0.5)
D = sum(sum(abs(BT.table(pth, BT.CTX[s])[k] - BT.table(P, BT.CTX[s])[k]) for k in BT.table(P, BT.CTX[s])) for s in ("S2", "S3"))
check("BK9 IDENTITY protection on the bridge: (X3, X5) separates {S2, S3} (range excludes 1/2, rho > 0); (X2, X4) does not; clock 7 steps",
      I35[0] > 0.5 and rho35 > 0.05 and I24[0] <= 0.5 and rho24 < 1e-9 and math.ceil(D / (D / 7) - 1e-9) == 7,
      f"anchored range with (X3,X5) [{I35[0]:.3f}, {I35[1]:.3f}], rho = {rho35:.4f}; with (X2,X4) [{I24[0]:.3f}, {I24[1]:.3f}], rho = {rho24:.1e}; D = {D:.4f}")

# BK10
rng = np.random.default_rng(7); eta = 0.45 * rho35; kappa = rho35 / 2; Mq, _ = T.marg(5, (2, 4))
def anchor_dist(cells, qobs):
    Mh, bh = T.system(5, cells); r = Mq.shape[0]; cst = np.r_[np.zeros(32), np.ones(r)]
    Aub = np.vstack([np.hstack([Mq, -np.eye(r)]), np.hstack([-Mq, -np.eye(r)])]); bub = np.r_[qobs, -qobs]
    res = linprog(cst, A_ub=Aub, b_ub=bub, A_eq=np.hstack([Mh, np.zeros((Mh.shape[0], r))]), b_eq=bh, bounds=[(0, None)] * (32 + r), method="highs")
    return res.fun
worst_h, worst_f = 0.0, np.inf
for _ in range(200):
    nz = rng.standard_normal(4); nz -= nz.mean(); nz *= eta / np.abs(nz).sum(); qobs = Mq @ P + nz      # inspection noise of L1 size eta
    worst_h = max(worst_h, anchor_dist(list(hon.values()), qobs))
    z = rng.dirichlet(np.ones(32)); Mh, bh = T.system(5, Hcells)                                  # a random far-side law consistent with h
    r_ = linprog(rng.standard_normal(32), A_eq=np.vstack([Mh]), b_eq=bh, A_ub=gB[None, :], b_ub=[0.5], bounds=[(0, None)] * 32, method="highs")
    pf = r_.x; far = Hcells + [("S2", BT.CTX["S2"], BT.table(pf, BT.CTX["S2"])), ("S3", BT.CTX["S3"], BT.table(pf, BT.CTX["S3"]))]
    Ifar = T.interval(5, far, gB)
    if Ifar[1] <= 0.5 + 1e-9: worst_f = min(worst_f, anchor_dist(far, qobs))
check("BK10 IDENTITY noisy margin: honest anchor distance <= eta < kappa < rho - eta <= anchor distance of every far-side configuration",
      worst_h <= eta + 1e-9 and eta < kappa < rho35 - eta <= worst_f + 1e-9,
      f"rho {rho35:.4f}, eta {eta:.4f}, kappa {kappa:.4f}; honest max {worst_h:.4f}; far-side min {worst_f:.4f} (over random coherent far-side reports of S2, S3)")

# BK11
grp = {nm: BB.corevision(BB.history(nm, W))[1] for nm in NAMES}
coal = ("S2", "S3", "R1", "R2")
check("BK11 FINDING co-revision names the lockstep coalitions that suspect sets miss or misattribute; and, identically, the matched honest drift",
      all(grp[n] == coal for n in ("efficient", "stealthy", "pump_retreat", "matched")) and grp["drift"] == tuple(BB.BASE)
      and all(grp[n] == () for n in ("shock", "contextual", "redundant_liar", "framing", "cascade")), f"{grp}")
print(f"{sum(results)}/{len(results)} passed")
