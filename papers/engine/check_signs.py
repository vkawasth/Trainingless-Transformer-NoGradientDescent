"""check_signs.py -- Phase III, Stage 3: real signs, parity, and how syzygies govern the sign tests (companion "Two latent types on
arbitrary covers"). Writes lab/signs.json.

  SG1  sheets: for covers with a finite fibre, the number of real solutions (v, r) found by multistart solving equals |G_2| = 2^dim,
       G_2 = ker P^T over F_2; the label swap (all signs flipped) always lies in G_2
  SG2  relative signs travel through r: two disjoint triangles, each with its triple, have their relative labels fixed although no window
       meets both (the shared r = t - 1/t aligns them); with one triple only, the second triangle's labels are free (4 solutions)
  SG3  the criterion is linear algebra over F_2, not the naive odd-meeting rule: a window meeting both components oddly fixes their
       relative sign; one meeting them evenly does not; and two windows each meeting one component (as in SG2) also do
  SG4  invisible sign classes: rank_Q A - rank_2 A equals the number of even invariant factors in the Smith form of A; for the triangle
       the negative-product sign pattern satisfies every relation (there are none) but has no real explanation; for the 4-cycle the
       wrong sign pattern violates the binomial c01 c23 = c12 c03
  SG5  for every pair graph in the zoo, b_1 = (relations: visible sign classes, = codim I_B) + (invisible: one per non-bipartite
       component, the odd-cycle class)
  SG6  data sign patterns: those passing every generator's sign test number 2^(n - codim); those realised by the model 2^(rank_2 A);
       the quotient is 2^(invisible) (enumeration on the Stage 1 covers)
  SG7  syzygies govern the sign tests: evaluated at the identity of the torus (all moments 1, a model point), the first syzygies are
       integer relations among the generators' moves u_j - w_j, of rank #gens - codim over Q and over F_2; so exactly codim of the
       generators' sign tests are independent, and the syzygies name the dependent ones
  SG8  the sign ladder: adding a face changes (rank_Q A, rank_2 A) by (1, 1) (absorbed: a sheet pair merges), (0, 0) (a new test), or
       (1, 0) (absorbed over Q but an invisible sign class is born), the last exactly when a face closes an odd cycle of the pair graph
"""
import itertools, json, os, sys
import numpy as np
from scipy.optimize import least_squares
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import signs as SG
from pce import syzygy as SZ
from pce import covers as CV

rng = np.random.default_rng(20261029); results = []; OUT = {}
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
pairs = lambda Ws: sorted({e for W in Ws for e in itertools.combinations(W, 2)})
def faces_of(m, E, W): return [tuple(e) for e in E] + [tuple(T) for T in W]

def solutions(m, faces, starts=300):
    v = rng.uniform(0.25, 0.7, m) * rng.choice([-1, 1], m); r = rng.uniform(0.3, 1.2) * rng.choice([-1, 1])
    target = np.array([np.prod(v[list(F)]) * CV.s_poly(len(F), r) if len(F) > 2 else np.prod(v[list(F)]) for F in faces])
    f = lambda z: np.array([np.prod(z[list(F)]) * CV.s_poly(len(F), z[m]) if len(F) > 2 else np.prod(z[list(F)]) for F in faces]) - target
    sols = []
    for _ in range(starts):
        rr = least_squares(f, rng.normal(size=m + 1), xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=2000)
        if np.abs(rr.fun).max() < 1e-11 and not any(np.allclose(rr.x, s, atol=1e-6) for s in sols): sols.append(rr.x)
    return sols, np.r_[v, r]

# SG1, SG2
COV = {"K3+012": (3, pairs([(0, 1, 2)]), [(0, 1, 2)]),
       "two triangles + both triples": (6, pairs([(0, 1, 2), (3, 4, 5)]), [(0, 1, 2), (3, 4, 5)]),
       "two triangles + one triple": (6, pairs([(0, 1, 2), (3, 4, 5)]), [(0, 1, 2)]),
       "bowtie": (5, pairs([(0, 1, 2), (2, 3, 4)]), [(0, 1, 2), (2, 3, 4)]),
       "K4 + triples": (4, list(itertools.combinations(range(4), 2)), list(itertools.combinations(range(4), 3))),
       "C4 + 012 + 123": (4, [(0, 1), (1, 2), (2, 3), (0, 3)], [(0, 1, 2), (1, 2, 3)])}
ok = True; tab = {}
for k, (m, E, W) in COV.items():
    F = faces_of(m, E, W); G = SG.sign_group(m, F); A, _ = SZ.config(m, E, W); finite = np.linalg.matrix_rank(A) == m + 1
    if finite:
        sols, truth = solutions(m, F); ok &= len(sols) == 2 ** G["dim"]
    else:
        sols = []
    ok &= G["swap_in"]; tab[k] = dict(dim=G["dim"], sheets=G["sheets"], finite=bool(finite), solutions=len(sols))
OUT["sheets"] = tab
check("SG1 sheets: on finite fibres the multistart real solutions number exactly |G_2| = 2^dim; the label swap is always in G_2", ok,
      "; ".join(f"{k}: |G_2| = {2 ** v['dim']}, solutions {v['solutions'] if v['finite'] else '(continuous fibre)'}" for k, v in tab.items()))
m = 6; F_both = faces_of(m, *COV["two triangles + both triples"][1:]); F_one = faces_of(m, *COV["two triangles + one triple"][1:])
rel_both = SG.determined(m, F_both, [0, 3]); rel_one = SG.determined(m, F_one, [0, 3])
check("SG2 relative labels travel through r: two disjoint triangles with both triples are aligned (2 solutions), with one triple they are not (4)",
      rel_both and not rel_one and tab["two triangles + both triples"]["solutions"] == 2 and tab["two triangles + one triple"]["solutions"] == 4)

# SG3: two disjoint edges {0,1}, {2,3} plus one window
E2 = [(0, 1), (2, 3)]
odd = SG.determined(4, faces_of(4, E2, []) + [(0, 1, 2)], [0, 2])                 # meets {0,1} twice (even), {2,3} once: no
odd2 = SG.determined(4, faces_of(4, E2, []) + [(0, 2), (0, 2, 3)][:1], [0, 2])     # a pair window 02 meets each once: yes
even = SG.determined(4, faces_of(4, E2, []) + [(0, 1, 2, 3)], [0, 2])              # four-window meets each twice: no
OUT["parity_rule"] = dict(window_012=odd, window_02=odd2, window_0123=even, through_r=rel_both)
check("SG3 F_2 criterion: a window meeting both components oddly (02) fixes the relative sign; even meetings (012 on {0,1}, 0123) do not; and windows meeting one component each can (SG2)",
      odd2 and not odd and not even and rel_both)

# SG4
A3 = SG.parity_matrix(3, [(0, 1), (1, 2), (0, 2)])[0][:3]; B3 = A3.astype(int)
inv3, rq3, r23 = SG.invisible_classes(B3); sm3 = SG.smith_invariants(B3)
neg_ok = CV.factorise(3, [(0, 1), (1, 2), (0, 2)], {(0, 1): -0.05, (1, 2): 0.04, (0, 2): 0.03})["ok"]
C4 = [(0, 1), (1, 2), (2, 3), (0, 3)]; c4bad = {(0, 1): -0.05, (1, 2): 0.04, (2, 3): 0.03, (0, 3): 0.0375}
binom = c4bad[(0, 1)] * c4bad[(2, 3)] - c4bad[(1, 2)] * c4bad[(0, 3)]; c4rel = abs(c4bad[(0, 1)] * c4bad[(2, 3)]) - abs(c4bad[(1, 2)] * c4bad[(0, 3)])
ok = True
for k, (m, E, W) in COV.items():
    A, _ = SZ.config(m, E, W); inv, rq, r2 = SG.invisible_classes(A); d = SG.smith_invariants(A); ok &= inv == sum(1 for x in d if x % 2 == 0)
check("SG4 invisible classes = even Smith invariants; the triangle's negative sign class satisfies every relation yet has no real explanation; the 4-cycle's wrong sign breaks its binomial",
      ok and inv3 == 1 and sm3 == [1, 1, 2] and not neg_ok and abs(c4rel) < 1e-12 and binom != 0, f"triangle Smith {sm3}; C4 binomial value {binom:.4f} with |terms| equal")

# SG5
ZOO = {"C3": (3, [(0, 1), (1, 2), (0, 2)]), "C4": (4, C4), "C4+diag": (4, C4 + [(0, 2)]), "C5": (5, [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)]),
       "P4": (4, [(0, 1), (1, 2), (2, 3)]), "two edges": (4, E2), "triangle+edge": (5, [(0, 1), (1, 2), (0, 2), (3, 4)]), "bowtie": (5, pairs([(0, 1, 2), (2, 3, 4)])),
       "K4": (4, list(itertools.combinations(range(4), 2))), "K5": (5, list(itertools.combinations(range(5), 2))), "two triangles": (6, pairs([(0, 1, 2), (3, 4, 5)]))}
ok = True; split = {}
for k, (m, E) in ZOO.items():
    s = SG.pair_split(m, E); split[k] = s; ok &= s["visible"] == s["codim"] and s["invisible"] == s["nonbipartite"] and s["b1"] == s["visible"] + s["invisible"]
OUT["pair_split"] = split
check("SG5 b_1 = visible (= codim I_B, the sign shadows of the relations) + invisible (one odd-cycle class per non-bipartite component)", ok,
      "; ".join(f"{k}: {v['b1']} = {v['visible']} + {v['invisible']}" for k, v in split.items()))

# SG6, SG7
STAGE1 = {"C4+012": (4, [(0, 1), (1, 2), (2, 3), (0, 3)], [(0, 1, 2)]), "K4 pairs": (4, list(itertools.combinations(range(4), 2)), []),
          "bowtie": (5, pairs([(0, 1, 2), (2, 3, 4)]), [(0, 1, 2), (2, 3, 4)]), "fan": (5, pairs([(0, 1, 2), (0, 2, 3), (0, 3, 4)]), [(0, 1, 2), (0, 2, 3), (0, 3, 4)]),
          "K4 triples": (4, list(itertools.combinations(range(4), 2)), list(itertools.combinations(range(4), 3))),
          "strip3": (7, pairs([(0, 1, 2), (2, 3, 4), (4, 5, 6)]), [(0, 1, 2), (2, 3, 4), (4, 5, 6)]), "K3 pairs": (3, [(0, 1), (1, 2), (0, 2)], [])}
ok6 = ok7 = True; info6 = {}; info7 = {}
for k, (m, E, W) in STAGE1.items():
    A, names = SZ.config(m, E, W); n = A.shape[1]; inv, rq, r2 = SG.invisible_classes(A)
    R = SZ.m2_ideal(m, E, W); codim = R["codim"]
    model = {tuple((A.T @ np.array(sig)) % 2) for sig in itertools.product((0, 1), repeat=m + 1)}
    if R["gens"]:
        G = SZ.parse_entries(R["diff"][0])[0]; U = SZ.binomial_moves(G, R["names"])
        passing = sum(1 for s in itertools.product((0, 1), repeat=n) if not np.any((U @ np.array(s)) % 2))
    else:
        G = []; U = np.zeros((0, n), int); passing = 2 ** n
    ok6 &= passing == 2 ** (n - codim) and len(model) == 2 ** r2 and passing // len(model) == 2 ** inv
    info6[k] = dict(n=n, codim=codim, passing=passing, model=len(model), invisible=inv)
    if len(G) > codim:
        S2 = SZ.parse_entries(R["diff"][1]); ones = {nm: 1 for nm in R["names"]}
        H = np.array([[int(round(SZ.eval_rel(x, ones))) for x in row] for row in S2])          # gens x syzygies, at the torus identity
        ok7 &= not np.any(U.T @ H) and np.linalg.matrix_rank(H) == len(G) - codim and SG.gf2_rank(H % 2) == len(G) - codim and SG.gf2_rank(U % 2) == codim
        info7[k] = dict(gens=len(G), codim=codim, rank_Q=int(np.linalg.matrix_rank(H)), rank_2=SG.gf2_rank(H % 2), tests_rank_2=SG.gf2_rank(U % 2))
OUT["sign_patterns"] = info6; OUT["syzygy_signs"] = info7
check("SG6 data sign patterns: 2^(n - codim) pass every relation's sign test, 2^(rank_2 A) are realised, the quotient is 2^(invisible)", ok6,
      "; ".join(f"{k}: {v['passing']} pass, {v['model']} realised" for k, v in info6.items()))
check("SG7 syzygies at the torus identity are integer relations among the moves u_j - w_j, of rank #gens - codim over Q and F_2; exactly codim sign tests are independent",
      ok7, "; ".join(f"{k}: {v['gens']} gens, {v['codim']} independent tests, syzygy rank {v['rank_Q']} (Q) = {v['rank_2']} (F_2)" for k, v in info7.items()))

# SG8
def Amat(m, faces):
    A = np.zeros((m + 1, len(faces)), int)
    for j, f in enumerate(faces): A[list(f), j] = 1; A[m, j] = 1 if len(f) == 3 else 0
    return A
counts = {}; ok = True; ex01 = None
for trial in range(300):
    m = int(rng.integers(3, 7)); faces = []; Eset = set()
    cand = [tuple(e) for e in itertools.combinations(range(m), 2)] + [tuple(T) for T in itertools.combinations(range(m), 3)]
    for kk in rng.permutation(len(cand))[:int(rng.integers(3, 10))]:
        F = cand[kk]
        if len(F) == 3:
            for e in itertools.combinations(F, 2):
                if e not in Eset: Eset.add(e); faces.append(e)
        if F in faces: continue
        before = list(faces); faces.append(F)
        if len(F) == 2: Eset.add(F)
        if not before: continue
        A0, A1 = Amat(m, before), Amat(m, faces); d = (int(np.linalg.matrix_rank(A1) - np.linalg.matrix_rank(A0)), SG.gf2_rank(A1) - SG.gf2_rank(A0))
        counts[str(d)] = counts.get(str(d), 0) + 1
        prevE = [f for f in before if len(f) == 2]
        closes_odd = False
        if len(F) == 2:
            comp = [C for C in CV.components(m, prevE) if F[0] in C][0]; side = CV.bipartition(comp, prevE)
            closes_odd = F[1] in comp and side is not None and side[F[0]] == side[F[1]]
        ok &= (d == (1, 0)) == closes_odd
        if d == (0, 1) and ex01 is None: ex01 = (before, F)
OUT["ladder"] = dict(counts=counts, example_01=str(ex01))
check("SG8 the sign ladder: (1,1) absorbed and a sign fixed; (0,0) a test; (1,0) absorbed with an invisible class born, exactly when an edge closes an odd cycle; (0,1) a test that also fixes a sign (e.g. a second triangle's triple)",
      ok and all(k in counts for k in ("(1, 1)", "(0, 0)", "(1, 0)", "(0, 1)")), f"{counts}; (0,1) example: adding {ex01[1] if ex01 else None}")

json.dump(OUT, open(os.path.join(HERE, "..", "lab", "signs.json"), "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else (bool(o) if isinstance(o, np.bool_) else str(o)))
print(f"{sum(results)}/{len(results)} passed")
