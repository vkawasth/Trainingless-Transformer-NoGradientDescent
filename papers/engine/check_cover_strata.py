"""check_cover_strata.py -- Phase III, Stage 4: strata (companion "Two latent types on arbitrary covers"). Writes lab/strata.json (module pce/cover_strata.py; renamed from check_strata.py, whose name belongs to the walls-and-chambers check).

  CS1  latent rank at a coincidence v_i = 0: rank of A on the faces avoiding i, plus one if some face contains i (308 random cases)
  CS2  the same formula for double coincidences Z = {i, j}: faces meeting Z twice vanish to second order, faces meeting it once give e_j
  CS3  tests at a coincidence: a generator whose A-degree in i is >= 2 vanishes to second order (zero gradient); the tests' degrees of
       freedom are the rank of the generators of degree <= 1 in i
  CS4  pair graphs: the tests lose exactly max(0, lobes_i - 1) degrees of freedom at v_i = 0, lobes_i = the number of components of G - i
       that are non-bipartite together with i (random graphs and named ones: bowtie, windmill, two 4-cycles, triangle + 4-cycle)
  CS5  the cut-variable conjecture is refined: two 4-cycles sharing a cut vertex lose nothing; the bowtie and the windmill lose at their
       centres; with triples (the strip) the drops sit at the shared variables 2 and 4
  CS6  r = 0 (equal weights): odd windows carry no sign, so labels aligned through r come apart: two disjoint triangles with both
       triples have 2 real explanations at r != 0 and 4 at r = 0; K4 with its triples keeps 2
  CS7  the merge v = 0 is the deepest stratum: latent rank 0 and every test gradient 0, for every cover tested
  CS8  the strata table of the bowtie with its triples (latent rank, hiding dimension, test df, sheets) over all strata
"""
import itertools, json, os, sys
import numpy as np
from scipy.optimize import least_squares
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import cover_strata as STR
from pce import syzygy as SZ
from pce import signs as SG
from pce import covers as CV

rng = np.random.default_rng(20261029); results = []; OUT = {}
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
pairs = lambda Ws: sorted({e for W in Ws for e in itertools.combinations(W, 2)})
def rand_cover():
    m = int(rng.integers(4, 7)); W = [T for T in itertools.combinations(range(m), 3) if rng.random() < 0.25]
    E = sorted({e for T in W for e in itertools.combinations(T, 2)} | {e for e in itertools.combinations(range(m), 2) if rng.random() < 0.3})
    return m, E, W

# CS1, CS2
tot = bad = tot2 = bad2 = 0
for _ in range(60):
    m, E, W = rand_cover(); F = [tuple(e) for e in E] + W; v = rng.uniform(0.2, 0.6, m) * rng.choice([-1, 1], m); r = rng.uniform(0.3, 1.0)
    for i in range(m):
        vz = v.copy(); vz[i] = 0; tot += 1; bad += STR.latent_rank(m, F, vz, r) != STR.latent_rank_pred(m, F, [i])
    for i, j in itertools.combinations(range(m), 2):
        vz = v.copy(); vz[[i, j]] = 0; tot2 += 1; bad2 += STR.latent_rank(m, F, vz, r) != STR.latent_rank_pred(m, F, [i, j])
check("CS1 latent rank at v_i = 0 = rank of A on faces avoiding i, + 1 if a face contains i", bad == 0, f"{tot} cases")
check("CS2 the same rule at double coincidences (faces meeting Z twice vanish to second order)", bad2 == 0, f"{tot2} cases")

# CS3
tot3 = bad3 = 0; zeroed = 0
for _ in range(15):
    m, E, W = rand_cover(); F = [tuple(e) for e in E] + W
    if len(F) > 12: continue
    R = SZ.m2_ideal(m, E, W)
    if not R["gens"]: continue
    G = SZ.parse_entries(R["diff"][0])[0]; v = rng.uniform(0.2, 0.6, m) * rng.choice([-1, 1], m); r = rng.uniform(0.3, 1.0)
    for i in range(m):
        vz = v.copy(); vz[i] = 0; d, J = STR.tests_df(G, R["names"], SZ.model_point(m, E, W, vz, r))
        keep = [k for k, g in enumerate(G) if max(STR.b_degree(g, R["names"], F, i)) <= 1]
        zero = [k for k, g in enumerate(G) if min(STR.b_degree(g, R["names"], F, i)) >= 2]; zeroed += len(zero)
        d2 = np.linalg.matrix_rank(J[keep], tol=1e-7) if keep else 0; tot3 += 1
        bad3 += d != d2 or (bool(zero) and np.abs(J[zero]).max() > 1e-8)
check("CS3 generators of A-degree >= 2 in i vanish to second order at v_i = 0; the tests' df is the rank of those of degree <= 1", bad3 == 0 and zeroed > 0,
      f"{tot3} cases, {zeroed} generators vanishing to second order")

# CS4, CS5
def df_graph(m, E, v, i=None):
    R = SZ.m2_ideal(m, E, [])
    if not R["gens"]: return 0, 0
    G = SZ.parse_entries(R["diff"][0])[0]; d0 = STR.tests_df(G, R["names"], SZ.model_point(m, E, [], v, 0.5))[0]
    if i is None: return d0, d0
    vz = v.copy(); vz[i] = 0; return d0, STR.tests_df(G, R["names"], SZ.model_point(m, E, [], vz, 0.5))[0]
named = {"bowtie": (5, [(0, 1), (1, 2), (0, 2), (2, 3), (3, 4), (2, 4)], 2), "windmill": (7, [(0, 1), (1, 2), (0, 2), (0, 3), (3, 4), (0, 4), (0, 5), (5, 6), (0, 6)], 0),
         "two 4-cycles at 0": (7, [(0, 1), (1, 2), (2, 3), (0, 3), (0, 4), (4, 5), (5, 6), (0, 6)], 0), "triangle + 4-cycle at 0": (6, [(0, 1), (1, 2), (0, 2), (0, 3), (3, 4), (4, 5), (0, 5)], 0)}
ok = True; nm_info = {}; tot4 = 0
for k, (m, E, c) in named.items():
    v = rng.uniform(0.2, 0.6, m) * rng.choice([-1, 1], m); d0, dc = df_graph(m, E, v, c); L = STR.lobes(m, E, c)
    nm_info[k] = dict(df=d0, df_at_centre=dc, lobes=L); ok &= dc == d0 - max(0, L - 1)
for _ in range(25):
    m = int(rng.integers(4, 7)); E = [e for e in itertools.combinations(range(m), 2) if rng.random() < 0.5]
    if len(E) < 4: continue
    v = rng.uniform(0.2, 0.6, m) * rng.choice([-1, 1], m)
    for i in range(m):
        d0, di = df_graph(m, E, v, i); tot4 += 1; ok &= di == d0 - max(0, STR.lobes(m, E, i) - 1)
OUT["lobes"] = nm_info
check("CS4 pair graphs: the tests lose max(0, lobes_i - 1) degrees of freedom at v_i = 0", ok,
      "; ".join(f"{k}: df {v['df']} -> {v['df_at_centre']} ({v['lobes']} lobes)" for k, v in nm_info.items()) + f"; {tot4} random cases")
check("CS5 the cut-variable conjecture refined: a cut vertex between bipartite lobes (two 4-cycles) loses nothing; between non-bipartite lobes it loses one per extra lobe",
      nm_info["two 4-cycles at 0"]["df_at_centre"] == nm_info["two 4-cycles at 0"]["df"] and nm_info["windmill"]["df"] - nm_info["windmill"]["df_at_centre"] == 2
      and nm_info["bowtie"]["df"] - nm_info["bowtie"]["df_at_centre"] == 1 and nm_info["triangle + 4-cycle at 0"]["df_at_centre"] == nm_info["triangle + 4-cycle at 0"]["df"])

# CS6
def count_solutions(m, F, v, r, starts=300):
    target = STR.phi(m, F, np.r_[v, r]); sols = []
    for _ in range(starts):
        rr = least_squares(lambda z: STR.phi(m, F, z) - target, rng.normal(size=m + 1), xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=3000)
        if np.abs(rr.fun).max() < 1e-11 and not any(np.allclose(rr.x, s, atol=1e-5) for s in sols): sols.append(rr.x)
    return len(sols)
TT = (6, pairs([(0, 1, 2), (3, 4, 5)]), [(0, 1, 2), (3, 4, 5)]); K4 = (4, list(itertools.combinations(range(4), 2)), list(itertools.combinations(range(4), 3)))
res6 = {}
for k, (m, E, W) in {"two triangles + triples": TT, "K4 + triples": K4}.items():
    F = [tuple(e) for e in E] + W; v = rng.uniform(0.25, 0.6, m) * rng.choice([-1, 1], m)
    res6[k] = dict(generic=count_solutions(m, F, v, 0.7), r0=count_solutions(m, F, v, 0.0), pred_generic=2 ** SG.sign_group(m, F)["dim"],
                   pred_r0=2 ** STR.sheets_at_r0(m, F)["dim"])
OUT["r0"] = res6
check("CS6 at r = 0 odd windows carry no sign: two disjoint triangles go from 2 to 4 real explanations (labels no longer aligned through r); K4 keeps 2",
      res6["two triangles + triples"]["generic"] == 2 and res6["two triangles + triples"]["r0"] == 4 and res6["K4 + triples"]["r0"] == 2
      and all(v["r0"] == v["pred_r0"] and v["generic"] == v["pred_generic"] for v in res6.values()), str(res6))

# CS7
ok = True
for _ in range(10):
    m, E, W = rand_cover(); F = [tuple(e) for e in E] + W; ok &= STR.latent_rank(m, F, np.zeros(m), 0.7) == 0 == STR.latent_rank_pred(m, F, list(range(m)))
    if len(F) <= 12:
        R = SZ.m2_ideal(m, E, W)
        if R["gens"]:
            G = SZ.parse_entries(R["diff"][0])[0]; ok &= STR.tests_df(G, R["names"], SZ.model_point(m, E, W, np.zeros(m), 0.7))[0] == 0
check("CS7 the merge is the deepest stratum: latent rank 0 and every test gradient 0", ok)

# CS8
m, E, W = 5, pairs([(0, 1, 2), (2, 3, 4)]), [(0, 1, 2), (2, 3, 4)]; F = [tuple(e) for e in E] + W; R = SZ.m2_ideal(m, E, W); G = SZ.parse_entries(R["diff"][0])[0]
v = np.array([0.4, -0.3, 0.5, 0.35, -0.45]); table = []
def row(name, vz, r):
    lr = STR.latent_rank(m, F, vz, r); d = STR.tests_df(G, R["names"], SZ.model_point(m, E, W, vz, r))[0]
    sh = 2 ** (STR.sheets_at_r0(m, F)["dim"] if r == 0 else SG.sign_group(m, F)["dim"]) // 2
    table.append(dict(stratum=name, latent_rank=lr, hiding_dim=m + 1 - lr, tests_df=d, sheets=sh))
row("generic", v, 0.7)
for i in range(m):
    vz = v.copy(); vz[i] = 0; row(f"v_{i} = 0", vz, 0.7)
vz = v.copy(); vz[[0, 2]] = 0; row("v_0 = v_2 = 0", vz, 0.7)
row("r = 0", v, 0.0); row("merge", np.zeros(m), 0.7)
OUT["bowtie_strata"] = table
check("CS8 the bowtie's strata table: generic (rank 6, df 2), a coincidence at the shared variable (df 1), r = 0, the merge (rank 0, df 0)",
      table[0]["latent_rank"] == 6 and table[0]["tests_df"] == 2 and table[3]["tests_df"] == 1 and table[-1]["latent_rank"] == 0 and table[-1]["tests_df"] == 0,
      "; ".join(f"{t['stratum']}: rank {t['latent_rank']}, dim {t['hiding_dim']}, df {t['tests_df']}, sheets {t['sheets']}" for t in table))

json.dump(OUT, open(os.path.join(HERE, "..", "lab", "strata.json"), "w"), indent=1, default=lambda o: int(o) if hasattr(o, "__int__") else str(o))
print(f"{sum(results)}/{len(results)} passed")
