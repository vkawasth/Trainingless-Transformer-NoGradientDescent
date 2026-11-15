"""The 3 x 3 test: four-dimensional coupling fibres (section sec:threebythree). New script; it modifies nothing; uses pce/cpl33.py.
Declared before the first run:
  E1 generic asymmetric margins f = (30, 20, 10)/60, g = (24, 21, 15)/60.
  E2 symmetric margins f = g = (1/3, 1/3, 1/3) (degenerate: subset sums coincide; included to show the collapse).
  E2' near-symmetric, nondegenerate f = (40, 43, 37)/120, g = (38, 41, 41)/120.
  Search: 600 margins with integer entries over 120 (Dirichlet(2) scaled and rounded; a remainder fixes the sums), seed 20261122,
  nondegenerate only (no proper row subset sum equals a column subset sum). E3 = the first searched margin whose dimension pattern
  (dimensions removed by successive levels, zeros dropped) is (1, 1, 1, 1); E4 = the first with a level removing 3 dimensions.
  Census: 2000 further margins (seed 20261123), the same distribution.
  (first run: 418 of 600 searched margins raised errors, because the facet test was infeasible for cells with f_i + g_j > 1, which can
  never be 0; these were code errors, not certificate failures. The Frechet bound now settles such cells exactly. The census then
  stopped on the same error.)
  TT1 IDENTITY  every linear programme in the leximin recursion, the facet tests and the search is certified in exact arithmetic
                (rational primal and dual, equal objectives); in every nondegenerate case the levels remove 4 dimensions in total
  TT2 RECORD    the declared examples: facets, levels h_j with their cells, dimensions removed r_j, vertices, smooth, Fano
  TT3 RECORD    selection depth: the distribution of dimension patterns over the search; E3 and E4
  TT4 IDENTITY  every nondegenerate sampled chamber is simple with unimodular vertex cones (Delzant); the symmetric margins are not simple
  TT5 RECORD    chambers: normal fans (maximal cones) up to row, column and transpose symmetry, and facet sets; how many fans are Fano
                (same vertex structure with support numbers 1, i.e. -K ample). First run classified by facet set only, which is coarser
                than the fan (one type per facet count); now by fan
  TT6 IDENTITY  spectral correspondence (thm:scales) on E1, E2', E3, E4: eigenvalues of Q/W at the positive critical point, 250 digits,
                L = 60, 120, 240, extrapolated in 1/L, equal h_j - h_1 with multiplicity r_j (tolerance 0.01). First run: FAIL, because
                the search found neither E3 nor E4, so only 2 examples ran against the declared minimum of 3 (both within 0.0026). Added
                after the first run and labelled so: E5 = the first searched margin with pattern (2, 2), E6 = the first with (4)
  TT7 IDENTITY  the first level (prop:firstlevel): h_1 = min(min_i f_i, min_j g_j) / 3, and the cells blocked at level 1 are exactly the
                rows or columns attaining it, so r_1 = 2 x (number of them); hence no level removes 3 and (1, 1, 1, 1) never occurs
                (all searched and census margins whose minimizing lines are facets)
"""
import os, sys, json, math, itertools
from collections import Counter
from fractions import Fraction as Fr
import numpy as np
import mpmath as mp
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import cpl33 as C
results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
def nondegenerate(f, g):
    fs = {sum(c) for k in (1, 2) for c in itertools.combinations(f, k)}; gs = {sum(c) for k in (1, 2) for c in itertools.combinations(g, k)}
    return not (fs & gs)
def rmarg(rng, den=120):
    v = rng.dirichlet([2, 2, 2]) * den; k = np.maximum(1, np.floor(v)).astype(int); k[int(np.argmax(k))] += den - k.sum()
    return [Fr(int(x), den) for x in k] if min(k) >= 1 else None
def analyse(f, g):
    F = C.exact_facets(f, g); lv, x = C.exact_leximin(f, g, F); r = C.r_pattern(lv)
    sm, V = C.smooth(f, g, F); fa = C.fano(f, g, F, V) if sm else None
    return dict(F=F, levels=lv, x=x, r=r, smooth=sm, fano=fa, nverts=len(V))
def tight_counts(f, g, F):
    """number of facets tight at each vertex (simple iff always 4): vertices as distinct points from the 4-subsets."""
    N = C.normals(); x0 = C.base_point(f, g); pts = {}
    import sympy as sp
    for Sq in C.vertex_sets(f, g, F):
        M = sp.Matrix([list(N[c]) for c in Sq]); u = M.LUsolve(sp.Matrix([-x0[c] for c in Sq])); key = tuple(u)
        pts[key] = sum(1 for c in F if sum(N[c][k] * u[k] for k in range(4)) == -x0[c])
    return list(pts.values())
EX = dict(E1=([Fr(30, 60), Fr(20, 60), Fr(10, 60)], [Fr(24, 60), Fr(21, 60), Fr(15, 60)]), E2=([Fr(1, 3)] * 3, [Fr(1, 3)] * 3),
          E2p=([Fr(40, 120), Fr(43, 120), Fr(37, 120)], [Fr(38, 120), Fr(41, 120), Fr(41, 120)]))
info = {}; cert_fail = 0; total_bad = 0
for nm, (f, g) in EX.items():
    try:
        a = analyse(f, g); a["tight"] = tight_counts(f, g, a["F"]); a["nondeg"] = nondegenerate(f, g); info[nm] = a
        total_bad += a["nondeg"] and sum(a["r"]) != 4
    except ValueError as e: cert_fail += 1; print("certificate failure", nm, e)
rng = np.random.default_rng(20261122); pats = Counter(); E3 = E4 = E5 = E6 = None; searched = 0; FL = []
while searched < 600:
    f = rmarg(rng); g = rmarg(rng)
    if f is None or g is None or not nondegenerate(f, g): continue
    searched += 1
    try:
        F = C.exact_facets(f, g); lv, x = C.exact_leximin(f, g, F); r = C.r_pattern(lv)
    except ValueError as e: cert_fail += 1; continue
    total_bad += sum(r) != 4; pat = tuple(k for k in r if k > 0); pats[pat] += 1
    if E3 is None and pat == (1, 1, 1, 1): E3 = (f, g)
    if E4 is None and 3 in pat: E4 = (f, g)
    if E5 is None and pat == (2, 2): E5 = (f, g)
    if E6 is None and pat == (4,): E6 = (f, g)
    FL.append((f, g, F, lv, r))
check("TT1 IDENTITY every linear programme is certified exactly; in every nondegenerate case the levels remove 4 dimensions in total",
      cert_fail == 0 and total_bad == 0, f"{searched} searched margins + 3 declared; certificate failures {cert_fail}; totals != 4: {total_bad}")
for nm, fg in (("E3", E3), ("E4", E4), ("E5", E5), ("E6", E6)):
    if fg is not None:
        a = analyse(*fg); a["tight"] = tight_counts(fg[0], fg[1], a["F"]); a["nondeg"] = True; info[nm] = a
fmt = lambda fr: "(" + ", ".join(str(x) for x in fr) + ")"
def desc(nm):
    a = info[nm]; f, g = EX.get(nm, None) or dict(E3=E3, E4=E4, E5=E5, E6=E6)[nm]
    return (f"{nm}: f={fmt(f)}, g={fmt(g)}, {len(a['F'])} facets, levels " + ", ".join(f"{h} [{len(S)} cells]" for h, S in a["levels"])
            + f", r={a['r']}, vertices {a['nverts']}, simple {set(a['tight']) == {4}}, smooth {a['smooth']}, Fano {a['fano']}")
check("TT2 RECORD declared examples (and the searched E3, E4)", True, " | ".join(desc(nm) for nm in info))
check("TT3 RECORD selection depth over the search: patterns of dimensions removed by successive levels", True,
      f"{searched} margins: " + ", ".join(f"{p}: {n}" for p, n in pats.most_common()) + f"; E3 {'found' if E3 else 'not found'}; E4 {'found' if E4 else 'not found'}")
rng2 = np.random.default_rng(20261123); n_ok = 0; n_tot = 0; types = Counter(); fano_by_type = {}; nf = Counter(); not_simple = 0; ftypes = Counter()
while n_tot < 2000:
    f = rmarg(rng2); g = rmarg(rng2)
    if f is None or g is None or not nondegenerate(f, g): continue
    n_tot += 1; F = C.exact_facets(f, g); sm, V = C.smooth(f, g, F); tc = tight_counts(f, g, F); simple = set(tc) == {4}
    n_ok += sm and simple; not_simple += not simple; key = C.canon_fan(V); types[key] += 1; nf[len(F)] += 1; ftypes[C.canon(F)] += 1
    if key not in fano_by_type: fano_by_type[key] = (C.fano(f, g, F, V), len(F), len(V))
sym = info["E2"]; sym_simple = set(sym["tight"]) == {4}
check("TT4 IDENTITY every nondegenerate sampled chamber is simple with unimodular vertex cones; the symmetric margins are not simple",
      n_ok == n_tot and not sym_simple, f"{n_tot} margins: Delzant {n_ok}; facets per chamber {dict(sorted(nf.items()))}; symmetric margins: tight facets per vertex {sorted(set(sym['tight']))}")
nF = sum(1 for k, v in fano_by_type.items() if v[0]); share = sum(types[k] for k, v in fano_by_type.items() if v[0]) / n_tot
check("TT5 RECORD chambers up to symmetry and the Fano ones", True,
      f"{len(types)} normal-fan types ({len(ftypes)} facet sets) in {n_tot} margins; Fano fan types {nF}; share of sampled margins in a Fano chamber {share:.2f}; "
      + "fan types (facets, vertices, Fano, count): " + "; ".join(f"({v[1]}, {v[2]}, {v[0]}, {types[k]})" for k, v in sorted(fano_by_type.items(), key=lambda kv: (kv[1][1], kv[1][2]))))
# TT6 spectral correspondence
mp.mp.dps = 250; N = C.normals()
def scales_at(L, F, xstar):
    rows = [[mp.mpf(int(N[c][i])) for i in range(4)] for c in F]; c0 = [mp.mpf(xstar[C.CELLS.index(c)].numerator) / xstar[C.CELLS.index(c)].denominator for c in F]
    u = [mp.mpf(0)] * 4
    for _ in range(200):
        e = [mp.e ** (-L * (c0[a] + mp.fsum(rows[a][i] * u[i] for i in range(4)))) for a in range(len(F))]
        gvec = mp.matrix([-L * mp.fsum(e[a] * rows[a][i] for a in range(len(F))) for i in range(4)]); H = mp.matrix(4, 4)
        for i in range(4):
            for j in range(4): H[i, j] = L * L * mp.fsum(e[a] * rows[a][i] * rows[a][j] for a in range(len(F)))
        step = mp.lu_solve(H, gvec); u = [u[i] - step[i] for i in range(4)]
        if mp.norm(step) < mp.mpf(10) ** (-150): break
    Fv = mp.fsum(e); ev = mp.eigsy(H / (L * L * Fv))[0]
    return sorted(float(-mp.log(abs(ev[i])) / L) for i in range(4))
spec = {}; worst = 0.0; done = 0
for nm in ("E1", "E2p", "E3", "E4", "E5", "E6"):
    if nm not in info: continue
    a = info[nm]; h1 = a["levels"][0][0]; target = sorted(float(h - h1) for (h, S), r in zip(a["levels"], a["r"]) for _ in range(r))
    Ls = [60, 120, 240]; vals = np.array([scales_at(L, a["F"], a["x"]) for L in Ls]); ext = [float(np.polyfit([1 / L for L in Ls], vals[:, i], 1)[1]) for i in range(4)]
    err = max(abs(x - y) for x, y in zip(sorted(ext), target)); worst = max(worst, err); done += 1
    spec[nm] = dict(target=target, extrapolated=sorted(ext), by_L={str(L): list(v) for L, v in zip(Ls, vals)})
check("TT6 IDENTITY eigenvalue valuations of Q/W at the positive critical point equal h_j - h_1 with multiplicity r_j in four dimensions",
      worst < 0.01 and done >= 3, f"{done} examples; worst extrapolation error {worst:.4f}; " + "; ".join(f"{k}: target {[round(t, 4) for t in v['target']]}" for k, v in spec.items()))
json.dump(dict(examples={nm: dict(F=[list(c) for c in a["F"]], levels=[[str(h), [list(c) for c in S]] for h, S in a["levels"]], r=a["r"], smooth=a["smooth"], fano=a["fano"],
               nverts=a["nverts"], tight=sorted(set(a["tight"]))) for nm, a in info.items()},
               margins={nm: [[str(x) for x in fg[0]], [str(x) for x in fg[1]]] for nm, fg in list(EX.items()) + [("E3", E3), ("E4", E4), ("E5", E5), ("E6", E6)] if fg is not None},
               patterns={str(k): v for k, v in pats.items()}, searched=searched, census=dict(n=n_tot, types=len(types), fano_types=nF, fano_share=share,
               facets=dict(nf)), spectral=spec), open(os.path.join(HERE, "threebythree_results.json"), "w"), indent=1)
# TT7 the first level
bad = 0; checked = 0
for f, g, F, lv, r in FL:
    h_pred = min(min(f), min(g)) / 3; lines = [("row", i) for i in range(3) if f[i] / 3 == h_pred] + [("col", j) for j in range(3) if g[j] / 3 == h_pred]
    cells = sorted({(i, j) for kind, k in lines for i in range(3) for j in range(3) if (i == k if kind == "row" else j == k)})
    if not all(c in F for c in cells): continue
    checked += 1; bad += lv[0][0] != h_pred or sorted(lv[0][1]) != cells or r[0] != 2 * len(lines)
check("TT7 IDENTITY first level: h_1 = min(f_min, g_min)/3, blocked cells = the minimizing rows/columns, r_1 = 2 x their number", bad == 0 and checked > 400,
      f"{checked} searched margins with facet minimizing lines; violations {bad}")
print(f"{sum(results)}/{len(results)} passed")
