"""Standalone verifier for the stratified ledger (engine/stratledger.json). It imports nothing from the project: it re-declares the
decision quantity from the paper (sec:ledger), rebuilds each table's margin system, solves the basis systems by exact Gaussian
elimination over the rationals (not the stored inverses), and checks every certificate from first principles.
  QL12 IDENTITY  for every table in the ledger and every band end (upper and lower, before and after evidence): the recorded basis is a
                 spanning tree of K_{r,c} (nonsingular, r + c - 1 cells), its vertex is feasible, the dual solution of n - lambda d on the
                 basis prices every other cell non-positively (upper end) or non-negatively (lower end), and the exact ratio equals the
                 recorded value to 1e-12; every listed table has a defined band (f0 + f1 + g0 + g1 > 1) and no table is both listed
                 and recorded as undefined; every stratum lists exactly the tables whose upper-end certificate is that basis, and its totals equal the sums of
                 the table records
Usage: python3 verify_ledger.py [path to stratledger.json]
"""
import json, sys
from fractions import Fraction as Fr
def solve(M, rhs):
    """exact Gaussian elimination; returns the solution or None if singular."""
    n = len(M); A = [list(M[i]) + [rhs[i]] for i in range(n)]
    for c in range(n):
        p = next((r for r in range(c, n) if A[r][c] != 0), None)
        if p is None: return None
        A[c], A[p] = A[p], A[c]
        for r in range(n):
            if r != c and A[r][c] != 0:
                f = A[r][c] / A[c][c]; A[r] = [x - f * y for x, y in zip(A[r], A[c])]
    return [A[i][n] / A[i][i] for i in range(n)]
def system(r, c):
    cells = [(i, j) for i in range(r) for j in range(c)]
    rows = [[Fr(1) if cl[0] == i else Fr(0) for cl in cells] for i in range(r)] + [[Fr(1) if cl[1] == j else Fr(0) for cl in cells] for j in range(c - 1)]
    return cells, rows
def verify(path):
    R = json.load(open(path)); bad = 0; checked = 0; report = []
    for tag, D in R.items():
        r, c = map(int, tag.split("x")); cells, A = system(r, c); m = len(A); ix = {cl: k for k, cl in enumerate(cells)}
        n0 = [Fr(1) if cl == (0, 0) else Fr(0) for cl in cells]; d0 = [Fr(1) if cl in ((0, 0), (0, 1), (1, 0), (1, 1)) else Fr(0) for cl in cells]
        Lk = [Fr(1) + Fr((i + 2 * j) % 3, 2) for (i, j) in cells]
        obj = dict(hi0=(n0, d0, 1), lo0=(n0, d0, -1), hi1=([a * b for a, b in zip(n0, Lk)], [a * b for a, b in zip(d0, Lk)], 1),
                   lo1=([a * b for a, b in zip(n0, Lk)], [a * b for a, b in zip(d0, Lk)], -1))
        libs = {k: [[ix[tuple(cl)] for cl in B] for B in v] for k, v in D["libraries"].items()}
        for tb in D["tables"]:
            f = [Fr(x) for x in tb["f"]]; g = [Fr(x) for x in tb["g"]]; b = f + g[:-1]
            if not (f[0] + f[1] + g[0] + g[1] > 1): bad += 1
            for key, (nv, dv, sense) in obj.items():
                B = libs[key][tb["certificates"][key]]; checked += 1
                if len(B) != m: bad += 1; continue
                AB = [[A[i][j] for j in B] for i in range(m)]; xB = solve(AB, b)
                if xB is None or any(x < 0 for x in xB): bad += 1; continue
                num = sum(nv[B[i]] * xB[i] for i in range(m)); den = sum(dv[B[i]] * xB[i] for i in range(m))
                if den <= 0: bad += 1; continue
                lam = num / den; cp = [nv[j] - lam * dv[j] for j in range(len(cells))]
                y = solve([[A[i][j] for i in range(m)] for j in B], [cp[j] for j in B])        # A_B^T y = c'_B
                if y is None: bad += 1; continue
                for j in range(len(cells)):
                    if j in B: continue
                    rc = cp[j] - sum(y[i] * A[i][j] for i in range(m))
                    if (sense > 0 and rc > 0) or (sense < 0 and rc < 0): bad += 1; break
                else:
                    bad += abs(float(lam) - tb[key]) > 1e-12
        ids_undef = {u["id"] for u in D["undefined"]}; ids_def = {tb["id"] for tb in D["tables"]}; bad += len(ids_undef & ids_def)
        byid = {tb["id"]: tb for tb in D["tables"]}
        for sid, s in D["strata"].items():
            mem = sorted(tb["id"] for tb in D["tables"] if tb["certificates"]["hi0"] == int(sid))
            bad += mem != sorted(s["tables"]) or s["count"] != len(mem) or abs(sum(byid[i]["drift"] for i in mem) - s["drift_total"]) > 1e-9
        report.append(f"{tag}: {len(D['tables'])} tables, {len(D['undefined'])} undefined, {len(D['strata'])} strata")
    return bad, checked, report
if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "stratledger.json"; bad, checked, report = verify(path)
    print(("PASS" if bad == 0 else "FAIL") + " QL12 IDENTITY standalone verification of every ledger certificate in exact arithmetic   [" + "; ".join(report) + f"; band ends verified {checked}; failures {bad}]")
    print(f"{int(bad == 0)}/1 passed")
