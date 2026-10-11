"""Decision under uncertainty with a support threshold, a provenance trace and the weakest link.

Credal set. The true outcome law q (over m outcomes) is unknown. Each source c reports counts N_c over its own outcome space, has
reliability r_c (or an interval; the low end is used, which is conservative), and an optional known context map T_c (columns are
the source's outcome law for each true outcome; identity if the context does not change outcomes). Then P_c = r_c T_c q + (1-r_c) F_c
with F_c arbitrary, so with the band [lo_c, hi_c] on P_c (monitor.bands):
    r_c (T_c q)_k <= hi_c,k            (the true part cannot exceed what was seen)
    r_c (T_c q)_k >= lo_c,k - (1-r_c)  (the false part cannot hide more than its share)
Q = {q in the simplex satisfying all of these} is a polytope.

Support threshold. Action x is kappa-supported over y when, for every q in Q,
    E_q[U_x] >= kappa * E_q[U_y]        (the shifted binomial wall  <U_x - kappa U_y, q> = 0)
with an optional additive slack delta: E_q[U_x] - kappa E_q[U_y] >= delta. kappa = 1, delta = 0 is the plain wall (a tie); kappa = 1.1
asks that x be at least 10% better in the worst case. For "assert outcome" actions (U = identity) this is q_x >= kappa q_y on all of Q:
the toric binomial q_x - kappa q_y set to a positive constant ratio rather than to zero. Each check is one LP.

Trace. By LP duality the worst-case margin is a sum over constraints of (dual x right-hand side); grouping by source gives each
source's contribution, and the binding constraints say which source bounds, on which outcomes, produced the decision.
Weakest link. For each source, the critical reliability r_c* below which the decision no longer holds (others fixed); the source with
the smallest slack r_c - r_c* is the weakest link. The thinnest cell is the binding constraint with the fewest supporting counts.
Regional gluing failure. If Q is empty the sources contradict each other: an elastic LP finds the smallest total relaxation, which
names the conflicting sources and outcomes; their reliabilities are lowered to the largest value consistent with the rest (the
certified reliability), and this repair is recorded in the trace.
"""
import os, sys, json
import numpy as np
from scipy.optimize import linprog
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
from monitor import bands

class Source:
    def __init__(self, name, counts, reliability, T=None):
        self.name = name; self.N = np.asarray(counts, float); self.n = self.N.sum()
        r = reliability if np.ndim(reliability) else (reliability, reliability); self.r_stated = (float(r[0]), float(r[1])); self.r = float(r[0])
        self.T = None if T is None else np.asarray(T, float)
        self.lo, self.hi = bands(self.N / self.n, self.n)

_FLOOR = None          # policy floors on outcomes (array of length m), set inside decide() after the evidence repair

def _rows(sources, m, override=None):
    """inequality rows A q <= b with a tag per row (source index, kind, outcome k); source index -1 is the support policy"""
    A, b, tag = [], [], []
    if _FLOOR is not None:
        for k in range(m):
            if _FLOOR[k] > 0: r = np.zeros(m); r[k] = -1.0; A.append(r); b.append(-_FLOOR[k]); tag.append((-1, 'floor', k))
    for i, s in enumerate(sources):
        r = s.r if override is None or i not in override else override[i]
        if r <= 0: continue
        T = np.eye(m) if s.T is None else s.T
        for k in range(T.shape[0]):
            A.append(r * T[k]); b.append(s.hi[k]); tag.append((i, 'upper', k))
            lb = s.lo[k] - (1 - r)
            if lb > 0: A.append(-r * T[k]); b.append(-lb); tag.append((i, 'lower', k))
    return np.array(A).reshape(-1, m), np.array(b), tag

def _lp(cvec, A, b, m):
    return linprog(cvec, A_ub=A if len(A) else None, b_ub=b if len(b) else None, A_eq=np.ones((1, m)), b_eq=[1.0], bounds=[(0, 1)] * m, method='highs')

def feasible(sources, m, override=None):
    A, b, _ = _rows(sources, m, override); return _lp(np.zeros(m), A, b, m).status == 0

def conflicts(sources, m):
    """elastic LP: min sum s, A q <= b + s. Returns per-source total violation and the violated rows."""
    A, b, tag = _rows(sources, m); R = len(b)
    c = np.concatenate([np.zeros(m), np.ones(R)]); Aub = np.hstack([A, -np.eye(R)])
    res = linprog(c, A_ub=Aub, b_ub=b, A_eq=np.hstack([np.ones((1, m)), np.zeros((1, R))]), b_eq=[1.0], bounds=[(0, 1)] * m + [(0, None)] * R, method='highs')
    s = res.x[m:]; per = {}
    for v, t in zip(s, tag):
        if v > 1e-9: per.setdefault(sources[t[0]].name, []).append(dict(kind=t[1], outcome=t[2], violation=round(float(v), 4)))
    return float(res.fun), per

def repair(sources, m, log, mode='exclude'):
    """Regional gluing failure. Find which single sources, if excluded, make Q non-empty; exclude the one with the lowest stated
    reliability (mode 'exclude'), or lower it to the largest consistent reliability (mode 'certified'). Repeat while Q is empty.
    The trace records the conflict, every alternative resolution, and the certified (largest consistent) reliability."""
    for _ in range(len(sources)):
        if feasible(sources, m): return
        tot, per = conflicts(sources, m); live = [i for i, s in enumerate(sources) if s.r > 0]
        fixes = [i for i in live if feasible(sources, m, {i: 0.0})]
        cand = fixes or [[s.name for s in sources].index(max(per, key=lambda k: sum(v['violation'] for v in per[k])))]
        i = min(cand, key=lambda j: sources[j].r_stated[1])
        lo, hi = 0.0, sources[i].r
        for _ in range(40):
            mid = (lo + hi) / 2
            if feasible(sources, m, {i: mid}): lo = mid
            else: hi = mid
        log.append(dict(event='regional gluing failure', total_violation=round(tot, 4), conflicts=per,
                        resolvable_by_excluding=[sources[j].name for j in fixes], repaired_source=sources[i].name, mode=mode,
                        reliability_stated=sources[i].r_stated, reliability_certified_max=round(lo, 4)))
        sources[i].r = 0.0 if mode == 'exclude' else lo

def margin(sources, m, u, override=None, trace=False):
    """min over Q of <u, q>, with the dual decomposition by source when trace=True"""
    A, b, tag = _rows(sources, m, override); res = _lp(u, A, b, m)
    if res.status != 0: return None if not trace else (None, None)
    if not trace: return float(res.fun)
    lam = res.ineqlin.marginals if len(b) else np.array([]); contrib = {}; binding = []
    for l, bb, t in zip(lam, b, tag):
        if abs(l) > 1e-10:
            if t[0] < 0:
                contrib['support policy'] = contrib.get('support policy', 0.0) + float(l * bb)
                binding.append(dict(source='support policy', kind='floor', outcome=t[2], dual=round(float(l), 5), counts=None, band_width=None)); continue
            s = sources[t[0]]; contrib[s.name] = contrib.get(s.name, 0.0) + float(l * bb)
            binding.append(dict(source=s.name, kind=t[1], outcome=t[2], dual=round(float(l), 5), counts=float(s.N[t[2]]), band_width=round(float(s.hi[t[2]] - s.lo[t[2]]), 4)))
    contrib['simplex'] = float(res.eqlin.marginals[0]); contrib['bounds'] = float(res.upper.marginals.sum())
    return float(res.fun), dict(worst_case_q=np.round(res.x, 4).tolist(), contributions={k: round(v, 5) for k, v in contrib.items()},
                                 decomposition_error=abs(sum(contrib.values()) - res.fun), binding=binding)

def floor_headroom(sources, m, floors):
    """max delta with q_k >= floor_k + delta on the floored outcomes, q in the sources' credal set (floors not imposed as rows)"""
    A, b, tag = _rows(sources, m); F = np.asarray(floors, float); idx = [k for k in range(m) if F[k] > 0]
    Aub = [np.append(r, 0.0) for r in A]; bub = list(b)
    for k in idx: r = np.zeros(m + 1); r[k] = -1; r[-1] = 1; Aub.append(r); bub.append(-F[k])
    c = np.zeros(m + 1); c[-1] = -1
    res = linprog(c, A_ub=np.array(Aub), b_ub=np.array(bub), A_eq=np.append(np.ones(m), 0)[None], b_eq=[1.0], bounds=[(0, 1)] * m + [(None, 1)], method='highs')
    lam = res.ineqlin.marginals; binding = []
    for l, t in zip(lam[:len(b)], tag):
        if abs(l) > 1e-10: binding.append(dict(source=sources[t[0]].name, kind=t[1], outcome=t[2], dual=round(float(l), 5)))
    active = [k for k, l in zip(idx, lam[len(b):]) if abs(l) > 1e-10]
    return float(-res.fun), binding, active

def decide(sources, U, actions, kappa=1.1, delta=0.0, voi_factor=4.0, repair_mode='exclude', floors=None):
    global _FLOOR
    _FLOOR = None
    try:
        return _decide(sources, U, actions, kappa, delta, voi_factor, repair_mode, floors)
    finally:
        _FLOOR = None

def _decide(sources, U, actions, kappa, delta, voi_factor, repair_mode, floors):
    """returns a dict: status (act / hedge / conflict-repaired act or hedge), action(s), margins, trace, weakest link, thinnest cell, VOI"""
    U = np.asarray(U, float); m = U.shape[1]; log = []
    sources = [Source(s.name, s.N, s.r_stated, s.T) for s in sources]           # work on copies
    global _FLOOR
    repair(sources, m, log, repair_mode)                                        # evidence conflicts are settled before the support policy
    evid = dict(type='conflicting sources', repairs=log) if log else None; supp = None; floor_info = None
    if floors is not None:
        F = np.zeros(m)
        for k, v in (floors.items() if isinstance(floors, dict) else enumerate(floors)): F[int(k)] = float(v)
        hd, hb, act = floor_headroom(sources, m, F); floor_info = dict(floors=F.tolist(), headroom=round(hd, 6), bottleneck_floors=act, binding_sources=hb)
        if hd < -1e-12:
            supp = dict(type='support floor exceeds what the sources allow', shortfall=round(-hd, 6), **floor_info)
            return dict(status='support obstruction', action=None, hedge_set=None, robust_choice=None, kappa=kappa, delta=delta, strength_kappa_star=None,
                        closest_rival=None, margin_vs_closest=None, trace=dict(repairs=log), weakest_link=None, thinnest_cell=None, value_of_information=None,
                        floor=floor_info, obstructions=dict(supp=supp, glue=None, comp=None, evid=evid),
                        next_action='revise the floor, revise the evidence, or defer')
        _FLOOR = F
    def mxy(x, y, override=None): return margin(sources, m, U[x] - kappa * U[y], override)
    def supported(x, override=None): return all((mxy(x, y, override) or -1) - delta >= -1e-12 for y in range(len(U)) if y != x)
    win = [x for x in range(len(U)) if supported(x)]
    # strength: largest kappa at which each action is still supported over each rival (bisection on the LP)
    def kstar(x, y, override=None):
        lo, hi = 0.0, 1e3
        if margin(sources, m, U[x] - lo * U[y], override) is None: return None
        for _ in range(50):
            mid = (lo + hi) / 2
            if margin(sources, m, U[x] - mid * U[y], override) - delta >= 0: lo = mid
            else: hi = mid
        return lo
    lower = [margin(sources, m, U[x]) for x in range(len(U))]; best = int(np.argmax(lower))   # best worst case
    x = win[0] if win else best
    rivals = {actions[y]: round(kstar(x, y), 4) for y in range(len(U)) if y != x}; ry = min(rivals, key=rivals.get); y = actions.index(ry)
    mval, tr = margin(sources, m, U[x] - kappa * U[y], trace=True)
    # weakest link: critical reliability per source for the decision x over its closest rival
    wl = {}
    for i, s in enumerate(sources):
        if (mval or -1) - delta < 0: break
        if mxy(x, y, {i: 0.0}) - delta >= 0: wl[s.name] = dict(r=round(s.r, 4), r_critical=0.0, slack=round(s.r, 4), needed=False); continue
        lo, hi = 0.0, s.r
        for _ in range(40):
            mid = (lo + hi) / 2
            if (mxy(x, y, {i: mid}) or -1) - delta >= 0: hi = mid
            else: lo = mid
        wl[s.name] = dict(r=round(s.r, 4), r_critical=round(hi, 4), slack=round(s.r - hi, 4), needed=True)
    weakest = min((k for k in wl if wl[k]['needed']), key=lambda k: wl[k]['slack'], default=None)
    thin = min((b for b in tr['binding'] if b['counts'] is not None), key=lambda d: d['counts'], default=None) if tr else None
    # value of information: which source, with voi_factor times the data at the same proportions, most raises the margin
    voi = {}
    for i, s in enumerate(sources):
        keep = (s.lo, s.hi); s.lo, s.hi = bands(s.N / s.n, s.n * voi_factor)
        if s.r <= 0: voi[s.name] = 'excluded'
        else:
            mv = mxy(x, y); voi[s.name] = 'more data would expose a conflict' if mv is None else round(mv - (mval if mval is not None else 0), 5)
        s.lo, s.hi = keep
    status = 'act' if win else 'hedge'
    maximal = [actions[a] for a in range(len(U)) if not any(b != a and (mxy(b, a) or -1) - delta >= 0 for b in range(len(U)))]
    return dict(status=status + (' (after repair)' if log else ''), action=actions[x] if win else None, hedge_set=None if win else maximal,
                robust_choice=actions[best], kappa=kappa, delta=delta,
                strength_kappa_star=rivals, closest_rival=ry, margin_vs_closest=None if mval is None else round(mval, 5),
                trace=dict(repairs=log, **(tr or {})), weakest_link=dict(source=weakest, by_source=wl), thinnest_cell=thin,
                value_of_information=dict(factor=voi_factor, margin_gain=voi, best=max((k for k in voi if isinstance(voi[k], float)), key=voi.get, default=None)),
                floor=floor_info, obstructions=dict(supp=None, glue=None, comp=None, evid=evid))

def assert_actions(outcomes): return np.eye(len(outcomes)), [f'assert {o}' for o in outcomes]

# ---------------------------------------------------------------------------------------------- planted checks
if __name__ == '__main__':
    rng = np.random.default_rng(11); ok = {}
    O = list('ABCD'); U, acts = assert_actions(O)
    # 1. one fully reliable source, large n: acts on the argmax; kappa* close to the true ratio q_A / q_B
    q = np.array([.5, .3, .15, .05]); d = decide([Source('s', q * 1e6, 1.0)], U, acts, kappa=1.1)
    ok['argmax_and_ratio'] = d['status'] == 'act' and d['action'] == 'assert A' and abs(d['strength_kappa_star']['assert B'] - 5 / 3) < 0.02
    # 2. kappa above the true ratio -> no longer supported (hedge)
    ok['kappa_threshold'] = decide([Source('s', q * 1e6, 1.0)], U, acts, kappa=1.8)['status'] == 'hedge'
    # 3. duality: the per-source contributions sum to the margin
    S3 = [Source('a', rng.multinomial(3000, .7 * q + .3 * np.eye(4)[1]), .7), Source('b', rng.multinomial(3000, .7 * q + .3 * np.eye(4)[2]), .7)]
    d3 = decide(S3, U, acts, kappa=1.1); ok['dual_decomposition'] = d3['trace']['decomposition_error'] < 1e-7
    # 4. two contexts with different misconceptions decide A where neither alone does
    ok['gluing_helps'] = d3['status'] == 'act' and all(decide([s], U, acts, kappa=1.1)['status'] == 'hedge' for s in S3)
    # 5. weakest link: the source whose misconception sits on the closest rival is the one the decision leans on
    #    (removing the other leaves the rival bounded); a planted conflicting source is found and repaired
    bad = Source('liar', [0, 0, 0, 3000], 0.9)                       # claims 90% reliable but reports only D
    d5 = decide(S3 + [bad], U, acts, kappa=1.1)
    ok['conflict_found_and_repaired'] = bool(d5['trace']['repairs']) and d5['trace']['repairs'][0]['repaired_source'] == 'liar' and d5['action'] == 'assert A'
    # 6. context map: a source that reports on a coarsened outcome space (A+B vs C+D) through a known T
    T = np.array([[1, 1, 0, 0], [0, 0, 1, 1.]]); c6 = Source('coarse', rng.multinomial(5000, T @ q), 1.0, T)
    ok['context_map_constrains'] = abs(-margin([c6], 4, -np.array([0, 0, 1, 1.])) - c6.hi[1]) < 1e-9 and abs(margin([c6], 4, np.array([0, 0, 1, 1.])) - c6.lo[1]) < 1e-9   # max and min of q_C + q_D are the coarse band ends
    print(json.dumps({k: bool(v) for k, v in ok.items()}), 'PASS' if all(ok.values()) else 'FAIL')
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results'), exist_ok=True)
    open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'decide_checks.txt'), 'w').write(('PASS ' if all(ok.values()) else 'FAIL ') + json.dumps({k: bool(v) for k, v in ok.items()}))
