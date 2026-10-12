"""Collapse-approach tracker (PREANALYSIS_COLLAPSE.md, section 1). A trajectory of two-way systems traces a path of margins; the
coupling polytopes form a family over it (Bands Prop. 6.1). At each step: chamber and signed wall distances with bands; tear and break
margins, velocity, tear and break times (Bands Def. 3.1, Thm. 3.2, Def. 5.7); the Pythagorean split of the drift into interaction and
margin parts (Csiszar); the normal-fan type (dimension, vertices = maximal cones, facets = rays); a Dirichlet posterior over chambers."""
import itertools, os, sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
from monitor import bands, walls, event_range
LOG2 = np.log(2)

_SOLVERS = {}
def _solvers(n, m):
    if (n, m) not in _SOLVERS:
        cells = [(i, j) for i in range(n) for j in range(m)]; N = n * m
        A = np.array([[1.0 if c[0] == i else 0.0 for c in cells] for i in range(n)] + [[1.0 if c[1] == j else 0.0 for c in cells] for j in range(m)])
        S = []
        for Z in itertools.combinations(range(N), N - (n + m - 1)):
            M = np.vstack([A] + [np.eye(N)[z][None] for z in Z])
            if np.linalg.matrix_rank(M) == N: S.append(np.linalg.pinv(M)[:, :n + m])
        _SOLVERS[(n, m)] = (A, np.array(S))
    return _SOLVERS[(n, m)]
def vertices(f, g, tol=1e-9):
    n, m = len(f), len(g); A, S = _solvers(n, m); b = np.concatenate([f, g]); X = S @ b
    ok = (X.min(1) > -tol) & (np.abs(X @ A.T - b).max(1) < 1e-8)
    return np.unique(np.round(X[ok], 9), axis=0)
def fan_type(f, g):
    """(dimension, number of vertices = maximal cones of the normal fan, number of facets = rays)"""
    V = vertices(np.asarray(f, float), np.asarray(g, float))
    if len(V) == 1: return dict(dim=0, vertices=1, facets=0)
    def adim(P): return int(np.linalg.matrix_rank(P[1:] - P[0], tol=1e-6)) if len(P) > 1 else 0
    d = adim(V); fac = 0
    for c in range(V.shape[1]):
        on = V[np.abs(V[:, c]) < 1e-9]
        if 0 < len(on) < len(V) and adim(on) == d - 1: fac += 1
    return dict(dim=d, vertices=int(len(V)), facets=fac)

def wall_table(f, g, nf, ng):
    """signed wall distances f(I) - g(J) with intervals on the margin bands"""
    (flo, fhi), (glo, ghi) = bands(f, nf), bands(g, ng); out = []
    for I, J in walls(len(f), len(g)):
        v = f[list(I)].sum() - g[list(J)].sum(); alo, ahi = event_range(flo, fhi, I); blo, bhi = event_range(glo, ghi, J)
        lo, hi = alo - bhi, ahi - blo; out.append((v, lo, hi, 1 if lo > 0 else (-1 if hi < 0 else 0)))
    return np.array(out)

def ipf(p, f, g, it=2000):
    q = p.copy()
    for _ in range(it):
        q *= (f / np.maximum(q.sum(1), 1e-300))[:, None]; q *= (g / np.maximum(q.sum(0), 1e-300))[None, :]
        if np.abs(q.sum(1) - f).max() < 1e-13: break
    return q
def kl(p, q): m = p > 0; return float((p[m] * np.log(p[m] / q[m])).sum()) / LOG2
def drift_split(p_prev, p_now):
    """D(p_now || p_prev) = D(p_now || p*) [interaction] + D(p* || p_prev) [margin], p* = IPF of p_prev to p_now's margins (bits)"""
    ps = ipf(p_prev, p_now.sum(1), p_now.sum(0)); tot, inter, marg = kl(p_now, p_prev), kl(p_now, ps), kl(ps, p_prev)
    return dict(total=tot, interaction=inter, margin=marg, pythagoras_err=abs(tot - inter - marg))

def chamber_posterior(T, rng, draws=200):
    D = rng.dirichlet(T.ravel() + 0.5, draws).reshape(draws, *T.shape); S = []
    for P in D:
        f, g = P.sum(1), P.sum(0); S.append(np.sign([f[list(I)].sum() - g[list(J)].sum() for I, J in walls(len(f), len(g))]).astype(int))
    vals, cnt = np.unique(np.array(S), axis=0, return_counts=True); return vals[cnt.argmax()], float(cnt.max() / draws), np.array(S)

def track(tables, neff=None, rng=None, labels=None):
    """tables: list of count tables (or laws with neff given). Returns one record per step."""
    rng = rng or np.random.default_rng(0); out = []; prev = None
    for t, T in enumerate(tables):
        T = np.asarray(T, float); n = float(neff[t]) if neff is not None else T.sum(); P = T / T.sum(); f, g = P.sum(1), P.sum(0)
        W = wall_table(f, g, n, n); mT, mB = float(np.abs(W[:, 0]).min()), float(min(f.min(), g.min()))
        rec = dict(step=t if labels is None else labels[t], n=n, f=f.round(4).tolist(), g=g.round(4).tolist(), tear_margin=mT, break_margin=mB,
                   signs=W[:, 0].round(12).tolist(), decided=W[:, 3].astype(int).tolist(), wall_lo=W[:, 1].tolist(), wall_hi=W[:, 2].tolist(),
                   fan=fan_type(f, g))
        counts = T if neff is None else P * n
        modal, pmodal, S = chamber_posterior(counts, rng); point = np.sign(W[:, 0]).astype(int)
        rec['chamber_posterior_modal'] = pmodal; rec['wall_keep_prob'] = (S == point[None, :]).mean(0).round(3).tolist()
        if prev is not None:
            Pp, fp, gp = prev
            v = 0.5 * np.abs(f - fp).sum() + 0.5 * np.abs(g - gp).sum(); rec['velocity'] = float(v)
            rec['tear_time'] = mT / v if v > 0 else np.inf; rec['break_time'] = mB / v if v > 0 else np.inf
            sm = lambda Q, nn: (Q * nn + 0.5) / (Q * nn + 0.5).sum()          # +1/2 per cell before the split (reported)
            rec['drift'] = drift_split(sm(Pp, n), sm(P, n))
            dprev = np.array(out[-1]['decided']); rec['decided_flips'] = int(((dprev * W[:, 3]) < 0).sum())
            rec['point_flips'] = int((np.sign(out[-1]['signs']) != np.sign(W[:, 0])).sum())
            rec['fan_change'] = 'same' if rec['fan'] == out[-1]['fan'] else ('break' if rec['fan']['dim'] < out[-1]['fan']['dim'] else
                                 ('heal' if rec['fan']['dim'] > out[-1]['fan']['dim'] else ('blow-up' if rec['fan']['vertices'] > out[-1]['fan']['vertices'] else 'blow-down')))
        out.append(rec); prev = (P, f, g)
    return out

if __name__ == '__main__':      # planted checks
    import json
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'phase4')); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'open_clinical'))
    import geom, tower_icu as TI
    rng = np.random.default_rng(1); ok = {}
    errs = [drift_split(rng.dirichlet(np.ones(9)).reshape(3, 3), rng.dirichlet(np.ones(9)).reshape(3, 3))['pythagoras_err'] for _ in range(50)]
    ok['pythagoras'] = max(errs) < 1e-10
    ok['fan_3x3'] = all(fan_type(f, g)['vertices'] == geom.n_vertices(f, g) for f, g in ((rng.dirichlet(np.ones(3)), rng.dirichlet(np.ones(3))) for _ in range(40)))
    ok['fan_dim_3x3'] = all(fan_type(rng.dirichlet(np.ones(3)), rng.dirichlet(np.ones(3)))['dim'] == 4 for _ in range(100))   # added after a tolerance bug
    nmap = {'CP^2': 3, 'F_1 = Bl_1 CP^2': 4, 'CP^1 x CP^1': 4, 'Bl_2 CP^2': 5, 'Bl_3 CP^2 (dP6)': 6}
    ok['fan_2x3'] = all(fan_type(np.array([f1, 1 - f1]), g)['vertices'] == nmap[TI.toric(f1, g)[0]] and fan_type(np.array([f1, 1 - f1]), g)['facets'] == nmap[TI.toric(f1, g)[0]]
                        for f1, g in ((rng.uniform(.05, .95), rng.dirichlet(np.ones(3))) for _ in range(40)))
    # a planted path across the single wall f(0) = g(0) of a 2x2 system, with large n: one decided flip at the crossing step
    tabs = []
    for f0 in (0.30, 0.35, 0.45, 0.55, 0.60):
        f, g = np.array([f0, 1 - f0]), np.array([0.5, 0.5]); tabs.append(np.outer(f, g) * 20000)
    tr = track(tabs); flips = [r.get('decided_flips', 0) for r in tr]
    ok['wall_crossing'] = [int(x > 0) for x in flips] == [0, 0, 0, 1, 0]   # a 2x2 system with g = (1/2, 1/2) has two walls, crossed at the same step
    # a path towards a zero margin: break time shrinks to 0
    tabs = [np.outer([1 - e, e], [0.5, 0.5]) * 5000 for e in (0.3, 0.2, 0.1, 0.05, 0.0)]
    tr = track(tabs); bt = [r['break_time'] for r in tr[1:]]
    ok['break_time'] = all(a >= b - 1e-12 for a, b in zip(bt, bt[1:])) and bt[-1] == 0 and tr[-1]['fan']['dim'] == 0 and tr[-1]['fan_change'] == 'break'
    print(ok, 'PASS' if all(ok.values()) else 'FAIL')
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results'), exist_ok=True)
    open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'tracker_checks.txt'), 'w').write(('PASS ' if all(ok.values()) else 'FAIL ') + json.dumps({k: bool(v) for k, v in ok.items()}) + ' (check specs corrected before any real run: two walls in the 2x2 path; break time non-increasing. fan_dim_3x3 added after the first trajectory run exposed a rank-tolerance bug in the fan dimension; trajectories rerun)')
