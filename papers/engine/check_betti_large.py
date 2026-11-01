"""Checks for the generators and first Betti numbers of the two largest higher-window covers (companion "Two latent types on
arbitrary covers", Theorem "higher windows" (3)). Needs Macaulay2.
  BG1  graded-piece generators: for each v-exponent multidegree b, I_b is the null space of the coefficient matrix of the r-polynomials
       of all monomials with A u = b; minimal generators are I_b modulo sum_F m_F I_{b - e_F}. This reproduces 16 (full four-window) and
       26 (tetrahedron + triangle), and confirms 71 (two tetrahedra on an edge, none new for 8 <= |b| <= 14) and 159 (full five-window,
       none new for 9 <= |b| <= 12); every lifted generator is checked as an exact identity over Z
  BG2  Macaulay2 Betti tables from these generators (over Z/32003, minimalBetti) reproduce the published tables of the two smaller covers
  BG3  two tetrahedra on an edge: codimension 14, beta_1 = 71, beta_2 = 481 (minimal resolution truncated at length 2)
"""
import json, os, subprocess, sys, io, contextlib, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "betti_large"))
import graded_gens as GG
import make_m2_lib as MM
results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
covers = {"k4full": (4, [F for k in (2, 3, 4) for F in itertools.combinations(range(4), k)], 12),
          "tettri": (5, [F for W in ([0, 1, 2, 3], [2, 3, 4]) for k in (2, 3, 4) for F in itertools.combinations(W, k) if len(F) <= len(W)], 14),
          "twotet": (6, [F for W in ([0, 1, 2, 3], [0, 1, 4, 5]) for k in (2, 3, 4) for F in itertools.combinations(W, k)], 14),
          "five": (5, [F for k in (2, 3, 4, 5) for F in itertools.combinations(range(5), k)], 12)}
counts = {}; bydeg = {}
for name, (m, faces, Bmax) in covers.items():
    with contextlib.redirect_stdout(io.StringIO()): g = GG.run(name, m, faces, Bmax)
    counts[name] = sum(g.values()); bydeg[name] = dict(sorted(g.items()))
check("BG1 graded-piece generators: 16, 26, 71 and 159 as stated in the paper, every lifted generator exact over Z",
      counts == {"k4full": 16, "tettri": 26, "twotet": 71, "five": 159}, "; ".join(f"{k} {v} by |b| {bydeg[k]}" for k, v in counts.items()))
def m2(script):
    return subprocess.run(["M2", "--script", script], capture_output=True, text=True, timeout=3600).stdout
tot = {}
for name in ("k4full", "tettri"):
    path = MM.write(name, os.path.join(HERE, "betti_large"), mode="full"); out = m2(path)
    tot[name] = [l for l in out.splitlines() if l.startswith("TOTALS")][0].split(" ", 1)[1]
check("BG2 Betti tables from the extracted generators reproduce the published ones",
      tot == {"k4full": "(1,16,41,47,40,24,5)", "tettri": "(1,26,92,171,258,274,161,44,3)"}, str(tot))
path = MM.write("twotet", os.path.join(HERE, "betti_large"), mode="trunc2"); out = m2(path)
codim = [l for l in out.splitlines() if l.startswith("CODIM")][0].split()[1]; tr = [l for l in out.splitlines() if l.startswith("TRUNC2")][0].split(" ", 1)[1]
check("BG3 two tetrahedra on an edge: codimension 14, beta_1 = 71, beta_2 = 481", codim == "14" and tr == "(1,71,481)", f"codim {codim}, {tr}")
print(f"{sum(results)}/{len(results)} passed")
