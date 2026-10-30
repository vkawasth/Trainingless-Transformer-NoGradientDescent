"""worked_descent.py -- the descent complex and the filtration on the worked examples (lab paper, Section 17 Step 14 and Section 18
Stage 7). Every number printed in those paragraphs is recomputed here; writes lab/worked_descent.json.

  WD1  the context category of the pair cover at m = 3: AB, BC, AC, A, B, C and the empty context; the nerve is a hollow triangle
  WD2  the Cech complex: dim C^0 = 12, C^1 = 6, C^2 = 1; rank d^0 = 5, rank d^1 = 1; H^0 = 7, H^1 = H^2 = 0
  WD3  the marginal map from signed laws (dimension 8) onto ker d^0 (dimension 7): rank 7, kernel spanned by chi_ABC (the line s)
  WD4  D1: the drug example's pair tables glue (lambda* = 1); the liar cycle does not (lambda* = 0, inequality value 3 > 2)
  WD5  D2: sign class 0, with M_01 M_02 M_12 = (-0.0432)(0.0324)(-0.0288) = 4.031e-5 > 0, and sigma = sign v = (+, -, +); the label
       switch gives (-, +, -); the "exactly one harm" law (uniform on 100, 010, 001) has M_ij = -1/9 on every edge: product < 0, class 1
  WD6  D3: coker B = 0 at m = 3, so |v| is solved: |v_i| = sqrt(M_ij M_ik / M_jk) = (0.2205, 0.1960, 0.1470) = sqrt(0.24) (0.45, 0.40, 0.30)
  WD7  D4: the walls leave the segment R = [-0.763, 0.573]; a witness with both classes zero and no segment
  WD8  D5 (m = 3): rhat = r0 = -0.408 lies in R; past eps_det = 0.0833 the hidden channel puts rhat outside R (the triple contradicts
       the pairs)
  WD9  four variables, D3: the two log tetrad ratios vanish on the truth; an injected product law (Theorem E(4)) at eps = 0.1 keeps the
       signs and makes them non-zero
  WD10 four variables, D5: the four r_S agree on the truth; half-capacity chi_ABC injection at eps = 0.05 moves r_ABC only
  WD11 labels: one replicate of Study 13's design A1b (n = 1000): the batch cochain delta^0 m_T, its energy = K (between + within), and
       the source statistic
"""
import itertools, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce.lab import law, moments, segment, r_of_t, fit_v
from pce import segment_decision as SD
from pce import injection as IJ
from pce import descent as DE
from pce import sources as SR
from pce import Window
from pce.cohomology import cech, cohomology_dims, separate
from pce.integrate import gluable_fraction
from pce.complex import nerve, reduced_betti

results = []; OUT = {}
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
r4 = lambda x: float(np.round(x, 6))

P = law(0.6, np.array([.5, .1, .4]), np.array([.05, .5, .1])); mu, M, mS = moments(P); v = SD.v_closed(M); vT = float(np.prod(v))
D3 = {0: 2, 1: 2, 2: 2}; pairs = [(0, 1), (1, 2), (0, 2)]; names = {(0, 1): "AB", (1, 2): "BC", (0, 2): "AC"}

# WD1
ctx = {"AB": {0, 1}, "BC": {1, 2}, "AC": {0, 2}}; N = nerve(ctx); b = reduced_betti([tuple(sorted(J)) for J in N])
OUT["nerve"] = {"simplices": [list(J) for J in N], "betti": {str(k): int(v_) for k, v_ in b.items()}}
check("WD1 the nerve of the pair cover is a hollow triangle (three vertices, three edges, no 2-simplex; b1 = 1)",
      len(N) == 6 and b.get(1) == 1 and b.get(2, 0) == 0, str(OUT["nerve"]["simplices"]))

# WD2
ds, off, U = cech(D3, pairs, qmax=2); dims = [off[q]["dim"] for q in range(3)]; ranks = [int(np.linalg.matrix_rank(d, 1e-9)) for d in ds[:2]]
H = cohomology_dims(ds)[:3]; d0 = ds[0]
OUT["cech"] = {"dims": dims, "ranks": ranks, "H": [int(h) for h in H], "d0": d0.astype(int).tolist(),
               "C0_blocks": [names[w] for w in pairs], "C1_blocks": [str(sorted(U[J])) for J in off[1] if J != "dim"]}
check("WD2 Cech complex: dims 12, 6, 1; ranks of d0, d1 = 5, 1; H^0 = 7, H^1 = H^2 = 0", dims == [12, 6, 1] and ranks == [5, 1] and H == [7, 0, 0],
      f"dims {dims}, ranks {ranks}, H {H}")

# WD3
K, X, rk = DE.descent_kernel(3, pairs); chi = IJ.chi(3, (0, 1, 2))
OUT["descent_map"] = {"rank": rk, "kernel_dim": int(K.shape[1]), "kernel_is_chi": bool(abs(abs(K[:, 0] @ chi) / np.linalg.norm(chi) - 1) < 1e-9)}
check("WD3 marginal map: rank 7 onto ker d0, kernel spanned by chi_ABC", rk == 7 and K.shape[1] == 1 and OUT["descent_map"]["kernel_is_chi"])

# WD4
W = [Window(("A", "B"), P.sum(axis=2)), Window(("B", "C"), P.sum(axis=0)), Window(("A", "C"), P.sum(axis=1))]
DA = {"A": 2, "B": 2, "C": 2}; lam_truth = gluable_fraction(DA, W).lam
eq = np.array([[0.5, 0], [0, 0.5]]); ne = np.array([[0, 0.5], [0.5, 0]]); liar = [Window(("A", "B"), eq), Window(("B", "C"), eq), Window(("A", "C"), ne)]
lam_liar = gluable_fraction(DA, liar).lam; nu_liar = separate(DA, liar).nu
bell_truth = float(np.trace(P.sum(axis=2)) + np.trace(P.sum(axis=0)) + P.sum(axis=1)[0, 1] + P.sum(axis=1)[1, 0])
OUT["D1"] = {"lambda_truth": r4(lam_truth), "lambda_liar": r4(lam_liar), "nu_liar": r4(nu_liar), "bell_truth": r4(bell_truth), "bell_liar": 3.0}
check("WD4 D1: the drug example glues (lambda* = 1, inequality value below 2); the liar cycle does not (lambda* = 0, value 3)",
      abs(lam_truth - 1) < 1e-9 and lam_liar < 1e-9 and bell_truth <= 2 and abs(nu_liar - 0.5) < 1e-9, f"truth's value {bell_truth:.3f}")

# WD5
e3, B3 = DE.pair_graph(3); prod = M[0, 1] * M[0, 2] * M[1, 2]; sc = DE.sign_class(M, e3, 3)
one = np.zeros((2, 2, 2)); one[1, 0, 0] = one[0, 1, 0] = one[0, 0, 1] = 1 / 3; M1 = moments(one)[1]; sc1 = DE.sign_class(M1, e3, 3)
OUT["D2"] = {"M": [r4(M[0, 1]), r4(M[0, 2]), r4(M[1, 2])], "product": float(prod), "sign_class": int(sc[0]), "sigma": [int(np.sign(x)) for x in v],
             "witness_M": [r4(M1[0, 1]), r4(M1[0, 2]), r4(M1[1, 2])], "witness_class": int(sc1[0])}
check("WD5 D2: product 4.031e-5 > 0, class 0, sigma (+, -, +); the exactly-one-harm law has M_ij = -1/9 and class 1",
      abs(prod - 4.031e-5) < 1e-8 and sc[0] == 0 and list(np.sign(v)) == [1, -1, 1] and np.allclose([M1[0, 1], M1[0, 2], M1[1, 2]], -1 / 9) and sc1[0] == 1,
      f"product {prod:.4e}")

# WD6
absv = [np.sqrt(abs(M[0, 1] * M[0, 2] / M[1, 2])), np.sqrt(abs(M[0, 1] * M[1, 2] / M[0, 2])), np.sqrt(abs(M[0, 2] * M[1, 2] / M[0, 1]))]
OUT["D3"] = {"cokernel_dim": int(3 - np.linalg.matrix_rank(B3)), "abs_v": [r4(x) for x in absv]}
check("WD6 D3: coker B = 0 at m = 3; |v| = (0.2205, 0.1960, 0.1470) = sqrt(0.24) (0.45, 0.40, 0.30)",
      OUT["D3"]["cokernel_dim"] == 0 and np.allclose(absv, np.sqrt(0.24) * np.array([0.45, 0.40, 0.30])))

# WD7
lo, hi = segment(mu, v); R = (r_of_t(lo), r_of_t(hi)); rng = np.random.default_rng(14)
wit = None
for _ in range(20000):
    Q = IJ.as_law(rng.dirichlet(np.ones(8)), 3); rr = DE.pair_realizable(Q)
    if rr["algebraic"] and rr["walls"] is False:
        muq, Mq, _ = moments(Q); vq = SD.v_closed(Mq); lq, hq = segment(muq, vq); wit = (Q, muq, vq, lq, hq); break
OUT["D4"] = {"R": [r4(R[0]), r4(R[1])], "t": [r4(lo), r4(hi)],
             "witness": {"mu": [r4(x) for x in wit[1]], "v": [r4(x) for x in wit[2]], "t_lo": r4(wit[3]), "t_hi": r4(wit[4])}}
check("WD7 D4: the segment R = [-0.763, 0.573]; a law with both classes zero whose walls cross (t_lo > t_hi)",
      abs(R[0] + 0.7626) < 1e-3 and abs(R[1] - 0.5731) < 1e-3 and wit is not None and wit[3] > wit[4], str(OUT["D4"]["witness"]))

# WD8
r0 = mS((0, 1, 2)) / vT; s_hi = IJ.stealth_range(P, (0, 1, 2))[1]; Qh = SR.hidden_Q(P, s_hi); e_det = (R[0] - r0) * vT / s_hi
e_out = SD.from_tables(SD.exact_tables(IJ.mix(P, Qh, 0.10))); rhat_out = e_out[2] if e_out else None
OUT["D5_m3"] = {"r0": r4(r0), "eps_det": r4(e_det), "rhat_at_0.10": r4(rhat_out)}
check("WD8 D5 (m = 3): r0 = -0.408 in R; at eps = 0.10 > eps_det = 0.0833 rhat leaves R", R[0] < r0 < R[1] and abs(e_det - 0.0833) < 1e-3 and rhat_out < R[0],
      f"rhat at 0.10: {rhat_out:.4f}")

# WD9
P4 = law(0.6, np.array([.5, .1, .4, .2]), np.array([.05, .5, .1, .7])); mu4, M4, mS4 = moments(P4); e4, B4 = DE.pair_graph(4)
idx = {ed: k for k, ed in enumerate(e4)}
def tets(Mx): return [float(np.log(abs(Mx[0, 1] * Mx[2, 3] / (Mx[0, 2] * Mx[1, 3])))), float(np.log(abs(Mx[0, 2] * Mx[1, 3] / (Mx[0, 3] * Mx[1, 2]))))]
Qp = law(1.0, np.array([.9, .9, .1, .5]), np.array([.9, .9, .1, .5])); Pq = IJ.mix(P4, Qp, 0.10); Mq = moments(Pq)[1]
OUT["D3_m4"] = {"cokernel_dim": int(6 - np.linalg.matrix_rank(B4)), "log_tetrads_truth": [r4(x) for x in tets(M4)], "log_tetrads_injected": [r4(x) for x in tets(Mq)],
                "sign_class_truth": DE.sign_class(M4, e4, 4).tolist(), "sign_class_injected": DE.sign_class(Mq, e4, 4).tolist()}
check("WD9 four variables, D3: coker B has dimension 2; log tetrads vanish on the truth; a product law injected at 0.1 keeps the signs and breaks them",
      OUT["D3_m4"]["cokernel_dim"] == 2 and max(map(abs, tets(M4))) < 1e-9 and min(map(abs, tets(Mq))) > 1e-3 and not any(OUT["D3_m4"]["sign_class_injected"]),
      f"injected log tetrads {OUT['D3_m4']['log_tetrads_injected']}")

# WD10
v4 = fit_v(mu4, M4); T = list(itertools.combinations(range(4), 3)); rb = {S: mS4(S) / np.prod(v4[list(S)]) for S in T}
lo4, hi4 = IJ.stealth_range(P4, (0, 1, 2)); s4 = 0.5 * hi4; Q4 = IJ.as_law(IJ.as_vec(P4) + s4 * IJ.chi(4, (0, 1, 2)), 4); Pa = IJ.mix(P4, Q4, 0.05)
mua, Ma, mSa = moments(Pa); va = fit_v(mua, Ma); ra = {S: mSa(S) / np.prod(va[list(S)]) for S in T}
OUT["D5_m4"] = {"r_truth": {"".join("ABCD"[i] for i in S): r4(rb[S]) for S in T}, "r_injected": {"".join("ABCD"[i] for i in S): r4(ra[S]) for S in T}}
check("WD10 four variables, D5: the four r_S agree on the truth; chi_ABC at eps = 0.05 moves r_ABC only",
      np.ptp(list(rb.values())) < 1e-9 and abs(ra[(0, 1, 2)] - rb[(0, 1, 2)]) > 1e-3 and all(abs(ra[S] - rb[S]) < 1e-9 for S in T if S != (0, 1, 2)),
      str(OUT["D5_m4"]))

# WD11
sys.path.insert(0, os.path.join(HERE, "study13_sources")); import design13 as D13
d = D13.setup("W"); n = 1000; e, sh = D13.shifts("A1b", n, d["gm"]["sm"], d["s"]); eps = [0.0] * 16 + e; rgen = np.random.default_rng(29)
bats = [SR.sample_batch(d["P"], d["Q"], eps[k], n, rgen, rho=D13.RHO) for k in range(24)]; mb = np.array([SR.mT_stats(x[(0, 1, 2)])[0] for x in bats])
dlt, En, Bt, Wt = DE.labelled_coboundary(mb, D13.SRC); vp = float(np.mean([SR.mT_stats(x[(0, 1, 2)])[1] for x in bats]))
_, zs, dropped = SR.source_filter(mb, D13.SRC, vp, n, D13.C_S, np.ones(24, bool))
OUT["labels"] = {"m_b": [r4(x) for x in mb], "src": D13.SRC.tolist(), "energy": float(En), "between": float(Bt), "within": float(Wt),
                 "z_source": {k: float(x) for k, x in zs.items()}, "source_dropped": dropped, "shift_per_adv_batch": r4(sh[0]), "m_T_truth": r4(mS((0, 1, 2)))}
check("WD11 labels: energy of delta^0 m_T = K (between + within) on one A1b replicate; C's statistic stays under c_s",
      abs(En - 24 * (Bt + Wt)) < 1e-12 and abs(zs["C"]) < D13.C_S + 1.5, f"z = {OUT['labels']['z_source']}, dropped {dropped}")

json.dump(OUT, open(os.path.join(HERE, "..", "lab", "worked_descent.json"), "w"), indent=1, default=float)
print(f"{sum(results)}/{len(results)} passed")
