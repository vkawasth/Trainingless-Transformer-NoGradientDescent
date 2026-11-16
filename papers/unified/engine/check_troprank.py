"""Study 40: the leximin point of a two-way table is tropical rank one; leximin by peeling (sec:troprank: prop:troprank1, cor:peel).
Pre-registered in study40_troprank/PREREG_study40.md, frozen by SHA-256 before any code for it ran. Reuses helpers of check_four6.py and the
Shape class of check_gkz.py (code before their check blocks); modifies nothing.
  RK1 IDENTITY tropical rank one    RK2 IDENTITY peeling = leximin    RK3 IDENTITY converse    RK4 IDENTITY walls = ratio ties
  RK5 RECORD off-fan changes        RK6 RECORD timing
Output: check_troprank_output.txt, study40_troprank/troprank.json.
First run (the only run): 6/6. RK4: 163 resolved changes, 0 violations; 42 bisections stalled on the wall and were not tested.
"""
import os, sys, json, time
from fractions import Fraction as Fr
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
SRC = open(os.path.join(HERE, "check_gkz.py")).read().split("# GZ1 ---")[0]; exec(SRC)
results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
OUT = {}

def peel(f, g):
    """Exact leximin of a two-way table by peeling. Returns table (dict cell -> Fraction) and steps (value, kind, index, gap)."""
    fr = [Fr(x) for x in f]; gc = [Fr(x) for x in g]; R = list(range(len(f))); C = list(range(len(g))); X = {}; steps = []
    while R and C:
        cand = sorted([(fr[i] / len(C), 0, i) for i in R] + [(gc[j] / len(R), 1, j) for j in C])
        v, kind, idx = cand[0]; gap = (cand[1][0] - v) if len(cand) > 1 else None
        steps.append((v, kind, idx, gap))
        if kind == 0:
            for j in C: X[(idx, j)] = v; gc[j] -= v
            R.remove(idx)
        else:
            for i in R: X[(i, idx)] = v; fr[i] -= v
            C.remove(idx)
    return X, steps
def lexf(cells, A, f, g):
    lv, x, _, _ = leximin_float(list(map(float, f)), list(map(float, g)), cells, A, cells); return lv, x
def partition_from_values(cells, vals, tol=1e-9):
    groups = []
    for k in np.argsort(vals):
        if groups and abs(vals[k] - groups[-1][0]) < tol: groups[-1][1].add(cells[k])
        else: groups.append([vals[k], {cells[k]}])
    return [frozenset(s) for _, s in groups]
SH = [(2, 3), (2, 5), (3, 3), (3, 4), (4, 6)]
rng = np.random.default_rng(4001)
w1 = w2 = w3 = 0.0; pm2 = 0; tl = tp = 0.0; nt = 0
for (r, c) in SH:
    cells, A, N, d = shape(r, c)
    for _ in range(50):
        f = rng.dirichlet([2] * r); g = rng.dirichlet([2] * c)
        t0 = time.time(); lv, x = lexf(cells, A, f, g); t1 = time.time(); X, _ = peel(f, g); t2 = time.time()
        if (r, c) == (4, 6): tl += t1 - t0; tp += t2 - t1; nt += 1
        T = np.zeros((r, c))
        for k, (i, j) in enumerate(cells): T[i, j] = x[k]
        w1 = max(w1, float(np.abs(np.minimum(T.max(1)[:, None], T.max(0)[None, :]) - T).max()))
        xp = np.array([float(X[cl]) for cl in cells]); w2 = max(w2, float(np.abs(xp - x).max()))
        pm2 += partition_from_values(cells, xp) != [frozenset(S_) for _, S_ in lv]
    for _ in range(50):
        a = rng.uniform(0.1, 1, r); b = rng.uniform(0.1, 1, c); T = np.minimum(a[:, None], b[None, :])
        lv, x = lexf(cells, A, T.sum(1), T.sum(0)); xt = np.array([T[i, j] for (i, j) in cells]); w3 = max(w3, float(np.abs(x - xt).max()))
check("RK1 IDENTITY prop:troprank1: the leximin point is tropical rank one, x_ij = min(row max, column max)", w1 < 1e-9,
      f"5 shapes x 50 tables; max deviation {w1:.1e}")
check("RK2 IDENTITY cor:peel: exact peeling gives the leximin point and its level partition", w2 < 1e-9 and pm2 == 0,
      f"250 tables; max difference {w2:.1e}; partition mismatches {pm2}")
check("RK3 IDENTITY prop:troprank1 (converse): every table min(a_i, b_j) is the leximin point of its own margins", w3 < 1e-9,
      f"250 random (a, b); max difference {w3:.1e}")
OUT["rk123"] = dict(w1=w1, w2=w2, pm2=pm2, w3=w3)
# RK4, RK5 ----------------------------------------------------------------
rng4 = np.random.default_rng(4002); info4 = []; ok4 = True; rec = {}
for name, S, samp, seg, _, length in SHAPES:
    n = v = off = off_first = stalls = 0; worst = 0.0
    for _ in range(40):
        p, q = seg(rng4, length); ts = np.linspace(0, 1, 41); sg = [S.lex(*at(p, q, t))[0] for t in ts]
        for k in range(40):
            if sg[k] == sg[k + 1]: continue
            a = sg[k]; lo_, hi_ = ts[k], ts[k + 1]; stalled = False
            while hi_ - lo_ > 1e-10:
                mid = (lo_ + hi_) / 2
                try: z = S.lex(*at(p, q, mid))[0]
                except RuntimeError:
                    try: z = S.lex(*at(p, q, mid + 1e-12))[0]
                    except RuntimeError: stalled = True; break
                if z == a: lo_ = mid
                else: hi_ = mid
            if stalled: stalls += 1; continue
            fm, gm = at(p, q, (lo_ + hi_) / 2); _, steps = peel(fm, gm)
            gaps = [(float(st[3]), s_i) for s_i, st in enumerate(steps) if st[3] is not None]
            gmin, step_at = min(gaps); worst = max(worst, gmin); n += 1; v += gmin >= 1e-7
            if S.circuit_dist(S.b_of(fm, gm)) >= 1e-7: off += 1; off_first += step_at == 0
    ok4 &= v == 0 and n > 0; rec[name] = dict(changes=n, violations=v, worst_gap=worst, stalls=stalls, off_fan=off, off_fan_at_first_step=off_first)
    info4.append(f"{name}: {n} changes, violations {v}, max tie gap {worst:.1e}, stalled bisections {stalls}")
check("RK4 IDENTITY cor:peel: every change of the leximin signature is a tie between two peeling ratios", ok4, "; ".join(info4))
check("RK5 RECORD changes off every circuit wall, and the peeling step of their tie", True,
      "; ".join(f"{k}: {r_['off_fan']} of {r_['changes']} off the secondary fan, {r_['off_fan_at_first_step']} of them at the first step" for k, r_ in rec.items()))
OUT["rk45"] = rec
check("RK6 RECORD time per 4x6 table", True, f"leximin_float {1000 * tl / nt:.1f} ms; exact peeling {1000 * tp / nt:.2f} ms")
OUT["rk6"] = dict(lp_ms=1000 * tl / nt, peel_ms=1000 * tp / nt)
json.dump(OUT, open(os.path.join(HERE, "study40_troprank", "troprank.json"), "w"), default=str)
print(f"{sum(results)}/{len(results)} passed")
