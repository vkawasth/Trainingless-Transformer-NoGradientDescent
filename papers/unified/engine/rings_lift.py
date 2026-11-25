"""EXPLORATORY (not a registered check): contextuality on binary n-cycles as the absence of a periodic lift.
defect(d) = min P(X_n != X_0) over gluings of the unrolled path X_0..X_n (context (n-1, n) carries the data of (n-1, 0));
CF(d) = contextual fraction = 1 - max total mass of a sub-probability global law dominated by the data on every context.
Compares the two on CHSH-type symmetric boxes and on random no-signalling n-cycle data."""
import itertools, json, numpy as np
from scipy.optimize import linprog
def tables_from(singles, cors):
    n = len(singles); T = []
    for k in range(n):
        a, b = singles[k], singles[(k + 1) % n]; c = cors[k]          # P(X_k=1)=a, P(X_{k+1}=1)=b, P(1,1)=c
        T.append({(1, 1): c, (1, 0): a - c, (0, 1): b - c, (0, 0): 1 - a - b + c})
    return T
def defect(T):
    n = len(T); V = list(itertools.product((0, 1), repeat=n + 1)); A = []; b = []
    for k in range(n):
        for (x, y), p in T[k].items(): A.append([1.0 if (v[k] == x and v[k + 1] == y) else 0.0 for v in V]); b.append(p)
    r = linprog([1.0 if v[0] != v[n] else 0.0 for v in V], A_eq=A, b_eq=b, bounds=(0, None), method="highs"); return r.fun
def cf(T):
    n = len(T); V = list(itertools.product((0, 1), repeat=n)); A = []; b = []
    for k in range(n):
        for (x, y), p in T[k].items(): A.append([1.0 if (v[k] == x and v[(k + 1) % n] == y) else 0.0 for v in V]); b.append(p)
    r = linprog(-np.ones(len(V)), A_ub=A, b_ub=b, bounds=(0, None), method="highs"); return 1 + r.fun
out = {"chsh": [], "random": {}}
for lam in (0.5, 0.6, 0.7071, 0.8, 0.9, 1.0):
    c = [(1 + lam) / 4] * 3 + [(1 - lam) / 4]; T = tables_from([0.5] * 4, c); out["chsh"].append((lam, defect(T), cf(T)))
rng = np.random.default_rng(20261124)
for n in (3, 4, 5):
    rows = []
    for _ in range(300):
        s = rng.uniform(0.15, 0.85, n); cs = []
        for k in range(n):
            a, b = s[k], s[(k + 1) % n]; lo, hi = max(0, a + b - 1), min(a, b); cs.append(rng.uniform(lo, hi))
        T = tables_from(s, cs); rows.append((defect(T), cf(T)))
    rows = np.array(rows); ctx = rows[:, 1] > 1e-9
    out["random"][n] = dict(n=len(rows), contextual=int(ctx.sum()), zero_defect_iff_noncontextual=bool(((rows[:, 0] < 1e-9) == (~ctx)).all()),
                           max_abs_diff=float(np.abs(rows[:, 0] - rows[:, 1]).max()), ratio_range=[float((rows[ctx, 0] / rows[ctx, 1]).min()) if ctx.any() else None, float((rows[ctx, 0] / rows[ctx, 1]).max()) if ctx.any() else None])
json.dump(out, open("rings_lift_output.json", "w"), indent=1); print(json.dumps(out, indent=1))
