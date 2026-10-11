"""Support floors, typed obstructions and auditable decision records (PREANALYSIS_SUPPORT.md).

Floor polytope. P_tau(f, g) = {pi in P(f, g): pi_xy >= tau_xy on designated cells D}. Proposition R: pi -> pi - tau 1_D maps it
bijectively onto P(f', g') with reduced margins f' = f - rowsums(tau 1_D), g' = g - colsums(tau 1_D). So feasibility, the normal fan
and the walls of the floored system are those of the reduced margins, and the headroom (the paper's max-min program) is a break margin
of the reduced margins, with a row or a column as the bottleneck.

Gluing. For a cover of contexts (sets of variables with local tables), a global section with floor tau exists iff
max { b'^T y : A^T y <= 0, -1 <= y <= 1 } = 0, where A maps global tables to the stacked local tables and b' = b - A tau.
A positive value is the smallest total (l1) relaxation of the local tables needed to glue, and the optimal y is a Farkas
certificate: y^T A pi' <= 0 for every pi' >= 0, but y^T b' > 0.

Transport. Floors push forward through any non-negative kernel K: pi >= tau implies K pi >= K tau.

Typed obstruction record O = (supp, glue, comp, evid); each component is None or a typed witness.
Records. A decision record holds its canonical inputs, its output and a SHA-256; replay() recomputes and compares.
"""
import os, sys, json, hashlib, itertools
import numpy as np
from scipy.optimize import linprog
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H); sys.path.insert(0, os.path.join(H, '..', 'phase6')); sys.path.insert(0, os.path.join(H, '..', '..'))
import fan_tracker as FT
from monitor import walls

# ------------------------------------------------------------------------------------------- floor polytopes
def mask(D, n, m):
    M = np.zeros((n, m))
    for x, y in D: M[x, y] = 1
    return M
def reduced_margins(f, g, tau, D):
    T = np.asarray(tau, float) * mask(D, len(f), len(g)); return np.asarray(f, float) - T.sum(1), np.asarray(g, float) - T.sum(0)

def floor_status(f, g, tau, D, nf=None, ng=None):
    """'feasible'/'infeasible' at the point margins, and decided status when the margins carry bands (effective sizes nf, ng)"""
    fr, gr = reduced_margins(f, g, tau, D); out = dict(point='feasible' if min(fr.min(), gr.min()) >= -1e-12 else 'infeasible', f_reduced=fr.tolist(), g_reduced=gr.tolist())
    if nf is not None:
        from monitor import bands
        (flo, fhi), (glo, ghi) = bands(np.asarray(f, float), nf), bands(np.asarray(g, float), ng); T = np.asarray(tau, float) * mask(D, len(f), len(g))
        if min((flo - T.sum(1)).min(), (glo - T.sum(0)).min()) >= 0: out['decided'] = 'feasible'
        elif min((fhi - T.sum(1)).min(), (ghi - T.sum(0)).min()) < 0: out['decided'] = 'infeasible'
        else: out['decided'] = 'undecided'
    return out

def headroom(f, g, tau, w, D):
    """closed form delta* = min over margins of (margin - floor sum) / weight sum; valid when >= 0. Returns value and bottleneck."""
    M = mask(D, len(f), len(g)); T, W = np.asarray(tau, float) * M, np.asarray(w, float) * M; cand = []
    for x in range(len(f)):
        if W[x].sum() > 0: cand.append(((f[x] - T[x].sum()) / W[x].sum(), ('row', x)))
    for y in range(len(g)):
        if W[:, y].sum() > 0: cand.append(((g[y] - T[:, y].sum()) / W[:, y].sum(), ('col', y)))
    v, b = min(cand); ties = [c[1] for c in cand if abs(c[0] - v) < 1e-9]
    return dict(delta=float(v), bottleneck=b, ties=ties, closed_form_valid=bool(v >= 0),
                certificate=f"{b[0]} {b[1]} carries {(f[b[1]] if b[0] == 'row' else g[b[1]]):.4f}; its floors demand {(T[b[1]].sum() if b[0] == 'row' else T[:, b[1]].sum()):.4f}")

def _transport_eq(n, m):
    A = np.vstack([np.kron(np.eye(n), np.ones(m)), np.kron(np.ones(n), np.eye(m))]); return A
def headroom_lp(f, g, tau, w, D):
    n, m = len(f), len(g); N = n * m; A = np.hstack([_transport_eq(n, m), np.zeros((n + m, 1))]); b = np.concatenate([f, g])
    Aub, bub = [], []
    for (x, y) in D: r = np.zeros(N + 1); r[x * m + y] = -1; r[-1] = w[x][y]; Aub.append(r); bub.append(-tau[x][y])
    c = np.zeros(N + 1); c[-1] = -1
    res = linprog(c, A_ub=Aub, b_ub=bub, A_eq=A, b_eq=b, bounds=[(0, None)] * N + [(None, None)], method='highs')
    return float(-res.fun), np.asarray(res.eqlin.marginals)

def floored_fan(f, g, tau, D):
    """normal-fan type and wall table of the floored system, via Proposition R (normalised reduced margins)"""
    fr, gr = reduced_margins(f, g, tau, D); s = fr.sum()
    if min(fr.min(), gr.min()) < -1e-12: return dict(feasible=False)
    fn, gn = np.clip(fr / s, 0, 1), np.clip(gr / s, 0, 1)
    sh = [(list(I), list(J), float(fr[list(I)].sum() - gr[list(J)].sum())) for I, J in walls(len(f), len(g))]
    return dict(feasible=True, fan=FT.fan_type(fn, gn), shifted_walls=sh, tear_margin=float(min(abs(v) for _, _, v in sh)))

def vertices_bounded(f, g, lower):
    """independent vertex enumeration of {pi in P(f,g): pi >= lower}: basic solutions with non-basic cells at their lower bound"""
    n, m = len(f), len(g); N = n * m; A = _transport_eq(n, m); b = np.concatenate([f, g]); lo = np.asarray(lower, float).ravel(); V = []
    for Z in itertools.combinations(range(N), N - (n + m - 1)):
        B = [i for i in range(N) if i not in Z]; AB = A[:, B]
        if np.linalg.matrix_rank(AB) < n + m - 1: continue
        rhs = b - A[:, list(Z)] @ lo[list(Z)]; xb, *_ = np.linalg.lstsq(AB, rhs, rcond=None)
        if np.abs(AB @ xb - rhs).max() > 1e-9: continue
        x = lo.copy(); x[B] = xb
        if (x >= lo - 1e-9).all(): V.append(np.round(x, 8))
    return np.unique(np.array(V), axis=0) if V else np.zeros((0, N))

# ------------------------------------------------------------------------------------------- gluing on a cover
def _marg_matrix(card, ctx):
    """rows: cells of the local table on variables ctx; columns: global cells"""
    gl = list(itertools.product(*[range(k) for k in card])); loc = list(itertools.product(*[range(card[v]) for v in ctx]))
    A = np.zeros((len(loc), len(gl)))
    for j, c in enumerate(gl): A[loc.index(tuple(c[v] for v in ctx)), j] = 1
    return A

def glue(card, family, tau=0.0):
    """family: list of (ctx tuple of variable indices, local table). tau: floor on every global cell (scalar or array).
    Returns status, obstruction (l1 relaxation needed), Farkas vector y, its verification and the contexts carrying it."""
    A = np.vstack([_marg_matrix(card, ctx) for ctx, _ in family]); b = np.concatenate([np.asarray(t, float).ravel() for _, t in family])
    G = A.shape[1]; tv = np.full(G, float(tau)) if np.ndim(tau) == 0 else np.asarray(tau, float).ravel(); bp = b - A @ tv
    res = linprog(-bp, A_ub=A.T, b_ub=np.zeros(G), bounds=[(-1, 1)] * len(b), method='highs'); val = float(-res.fun); y = res.x
    sizes = [np.asarray(t).size for _, t in family]; cut = np.cumsum([0] + sizes); region = [list(family[i][0]) for i in range(len(family)) if np.abs(y[cut[i]:cut[i + 1]]).max() > 1e-9]
    ok = val <= 1e-9
    out = dict(status='glues' if ok else 'fails', obstruction_l1=round(val, 10))
    if not ok: out.update(farkas=y.round(10).tolist(), verified=bool((A.T @ y <= 1e-9).all() and bp @ y > 1e-9), region=region)
    return out

def three_cycle_noncontextual(Exy, Eyz, Ezx):
    """Araujo et al. 2013 3-cycle inequalities for +-1 variables with consistent margins: sum gamma_i E_i <= 1, odd number of -1s"""
    E = (Exy, Eyz, Ezx)
    return all(sum(gm * e for gm, e in zip(gam, E)) <= 1 + 1e-9 for gam in itertools.product((1, -1), repeat=3) if gam.count(-1) % 2 == 1)

def chain_statuses(pxy, pyz, tau):
    """X - Y - Z chain, binary, floor tau on all 8 cells: closed forms for the three statuses"""
    py = pxy.sum(0); t_any = min(pxy.min(), pyz.min()) / 2; t_ci = min(pxy[x, y] * pyz[y, z] / py[y] for x in range(2) for y in range(2) for z in range(2))
    return dict(any_filler=True, floored_filler=bool(tau <= t_any + 1e-12), ci_filler_meets_floor=bool(tau <= t_ci + 1e-12), tau_any=float(t_any), tau_ci=float(t_ci))

# ------------------------------------------------------------------------------------------- transport and composition
def push(pi, tau, K):
    """floors push forward through a non-negative kernel K (rows: target cells, columns: source cells)"""
    K = np.asarray(K, float); return K @ np.asarray(pi, float).ravel(), K @ np.asarray(tau, float).ravel()
def holonomy(kernels):
    """composition obstruction: max total-variation displacement of a point mass carried around a loop of kernels"""
    P = np.eye(np.asarray(kernels[0]).shape[1])
    for K in kernels: P = np.asarray(K, float) @ P
    return float(0.5 * np.abs(P - np.eye(P.shape[0])).sum(0).max())

# ------------------------------------------------------------------------------------------- typed records and replay
def canon(o):
    if isinstance(o, dict): return {str(k): canon(v) for k, v in sorted(o.items(), key=lambda kv: str(kv[0]))}
    if isinstance(o, (list, tuple)): return [canon(v) for v in o]
    if isinstance(o, (np.floating, float)): return round(float(o), 10)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, np.ndarray): return canon(o.tolist())
    if isinstance(o, np.bool_): return bool(o)
    return o
def sha(o): return hashlib.sha256(json.dumps(canon(o), sort_keys=True).encode()).hexdigest()

def audit(problem):
    """problem: dict with optional parts 'decision' (sources, U, actions, kappa, floors), 'cover' (card, family, tau),
    'loop' (kernels, tolerance). Returns the typed obstruction record and each part's result."""
    import decide as DC
    O = dict(supp=None, glue=None, comp=None, evid=None); res = {}
    if 'decision' in problem:
        p = problem['decision']; S = [DC.Source(s['name'], s['counts'], s['r'], s.get('T')) for s in p['sources']]
        d = DC.decide(S, np.array(p['U']), p['actions'], kappa=p.get('kappa', 1.1), floors=p.get('floors'))
        res['decision'] = d; O['supp'] = d['obstructions']['supp']; O['evid'] = d['obstructions']['evid']
    if 'cover' in problem:
        c = problem['cover']; g0 = glue(c['card'], [(tuple(k), np.array(t)) for k, t in c['family']], 0.0)
        gt = glue(c['card'], [(tuple(k), np.array(t)) for k, t in c['family']], c.get('tau', 0.0)); res['cover'] = dict(unfloored=g0, floored=gt)
        if g0['status'] == 'fails': O['glue'] = dict(type='no global section (contextual)', witness=g0)
        elif gt['status'] == 'fails': O['supp'] = O['supp'] or dict(type='support-contextual: glues, but not with the floor', witness=gt)
    if 'loop' in problem:
        h = holonomy(problem['loop']['kernels']); res['loop'] = dict(holonomy=h)
        if h > problem['loop'].get('tolerance', 1e-9): O['comp'] = dict(type='nonzero holonomy around the loop', holonomy=h)
    return dict(obstructions=O, results=res)

def record(problem, version=None):
    inp = canon(problem); out = audit(inp); return dict(inputs=inp, version=version, output=canon(out), sha256=sha(out))
def replay(rec): return sha(audit(rec['inputs'])) == rec['sha256']

# ---------------------------------------------------------------------------------------------- planted checks
if __name__ == '__main__':
    import decide as DC
    ok = {}; rng = np.random.default_rng(2)
    f, g = np.array([.5, .3, .2]), np.array([.4, .35, .25]); tau = np.full((3, 3), .03); D = [(x, y) for x in range(3) for y in range(3)]
    fr, gr = reduced_margins(f, g, tau, D); V1 = vertices_bounded(f, g, tau); V2 = vertices_bounded(fr, gr, np.zeros((3, 3))) + tau.ravel()
    ok['prop_R_vertices'] = V1.shape == V2.shape and np.abs(np.unique(np.round(V1, 7), axis=0) - np.unique(np.round(V2, 7), axis=0)).max() < 1e-7
    h = headroom(f, g, tau, np.ones((3, 3)), D); ok['headroom_closed_form'] = abs(h['delta'] - headroom_lp(f, g, tau, np.ones((3, 3)), D)[0]) < 1e-9 and h['bottleneck'] == ('row', 2)
    # PR-like triangle: X=Y, Y=Z, Z != X perfectly -> no global section; certificate verified
    eq, ne = np.array([[.5, 0], [0, .5]]), np.array([[0, .5], [.5, 0]])
    gl = glue((2, 2, 2), [((0, 1), eq), ((1, 2), eq), ((2, 0), ne)]); ok['contextual_triangle'] = gl['status'] == 'fails' and gl['verified']
    # a gluable family that fails with a floor: marginals of a joint with two empty cells, floor 0.01
    J = rng.dirichlet(np.ones(8)).reshape(2, 2, 2); J[0, 0, 0] = 0; J[1, 1, 1] = 0; J /= J.sum()
    fam = [((0, 1), J.sum(2)), ((1, 2), J.sum(0)), ((2, 0), J.sum(1).T)]
    ok['support_contextual_possible'] = glue((2, 2, 2), fam, 0.0)['status'] == 'glues'
    ok['holonomy'] = holonomy([np.eye(3)[[1, 2, 0]].T]) > 0.99 and holonomy([np.eye(3), np.eye(3)]) == 0
    # decide with floors: a floor the sources allow is met; a floor above what they allow is a support obstruction
    S = [DC.Source('a', [600, 300, 100], 1.0)]; U, acts = DC.assert_actions('abc')
    d_ok = DC.decide(S, U, acts, floors={2: .05}); d_bad = DC.decide(S, U, acts, floors={2: .30})
    ok['decide_floors'] = d_ok['status'] == 'act' and d_ok['floor']['headroom'] > 0 and d_bad['status'] == 'support obstruction' and d_bad['obstructions']['supp']['bottleneck_floors'] == [2]
    rec = record(dict(decision=dict(sources=[dict(name='a', counts=[600, 300, 100], r=1.0)], U=U.tolist(), actions=acts, kappa=1.1, floors={2: .05})), 'v1')
    ok['replay'] = replay(rec)
    print(json.dumps({k: bool(v) for k, v in ok.items()}), 'PASS' if all(ok.values()) else 'FAIL')
    open(os.path.join(H, 'results', 'support_checks.txt'), 'w').write(('PASS ' if all(ok.values()) else 'FAIL ') + json.dumps({k: bool(v) for k, v in ok.items()}))
