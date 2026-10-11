"""monitor.py -- a structural-change monitor for couplings under changing information.

A *system* is a pair of contents (A, B) measured jointly on the same units in each of several *contexts* (versions, wordings, years,
waves). In context c the observed joint table T_c of (A, B) has margins (f_c, g_c). The monitor reports, for a change c -> c':

  regime   bend / tear / break of the margins (f, g), judged with sampling bands [Bands, Thms 3.2, 3.3, Prop 3.4]
  stress   realized change TV(f, f') + TV(g, g') divided by the tear margin m_T(f, g) of the starting state
  context  Contextuality-by-Default for two contents in two contexts (a cyclic system of rank 2) [Bridges, Sec. 4]
  churn    when the same units are observed in both contexts: the observed coupling of a content with itself across contexts,
           against the maximal coupling CbD assumes; the excess is item-level change that the margins cannot show
  gluing   for three contexts on the same units, the gluing gap I(X1; X3 | X2) [Bands, Prop 4.8]
"""
import itertools, math, os, sys
import numpy as np
from scipy.stats import norm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cbd import lp_noncontextual, criterion, cntx   # noqa: E402

# ------------------------------------------------------------------ bands
def bands(p, n_eff, alpha=0.05):
    """Simultaneous elementwise band for a probability vector p estimated from n_eff observations: Wilson intervals at
    level alpha / k (Bonferroni over the k categories), then tightened so that the band is consistent with sum = 1."""
    p = np.asarray(p, float); k = len(p); z = norm.ppf(1 - alpha / (2 * k)); n = max(n_eff, 1.0)
    den = 1 + z * z / n; ctr = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    lo, hi = np.clip(ctr - half, 0, 1), np.clip(ctr + half, 0, 1)
    lo2 = np.maximum(lo, 1 - (hi.sum() - hi)); hi2 = np.minimum(hi, 1 - (lo.sum() - lo))
    return lo2, hi2

def event_range(lo, hi, I):
    I = list(I); Ic = [i for i in range(len(lo)) if i not in I]
    return max(lo[I].sum(), 1 - hi[Ic].sum()), min(hi[I].sum(), 1 - lo[Ic].sum())

# ------------------------------------------------------------------ walls, margins
def walls(n, m):
    """Walls f(I) = g(J) for I containing 0 (f(I^c) = g(J^c) is the same wall), I and J nonempty and proper."""
    Is = [I for r in range(1, n) for I in itertools.combinations(range(n), r) if 0 in I]
    Js = [J for r in range(1, m) for J in itertools.combinations(range(m), r)]
    return [(I, J) for I in Is for J in Js]

def wall_values(f, g):
    return np.array([np.asarray(f)[list(I)].sum() - np.asarray(g)[list(J)].sum() for I, J in walls(len(f), len(g))])

def tear_margin(f, g):
    return float(np.min(np.abs(wall_values(f, g))))

def break_margin(f, g):
    return float(min(np.min(f), np.min(g)))

def wall_signs(fb, gb):
    """Sign of every wall function that is certain on the band (interval excludes 0): +1, -1, or 0 when undecided."""
    (flo, fhi), (glo, ghi) = fb, gb; out = []
    for I, J in walls(len(flo), len(glo)):
        a_lo, a_hi = event_range(flo, fhi, I); b_lo, b_hi = event_range(glo, ghi, J)
        lo, hi = a_lo - b_hi, a_hi - b_lo
        out.append(1 if lo > 0 else (-1 if hi < 0 else 0))
    return np.array(out)

def tv(p, q): return 0.5 * float(np.abs(np.asarray(p, float) - np.asarray(q, float)).sum())

def H(p):
    p = np.asarray(p, float).ravel(); p = p[p > 0]; return float(-(p * np.log2(p)).sum())

# ------------------------------------------------------------------ state of a context, and the change between two
class State:
    def __init__(self, T, n_eff, alpha=0.05):
        self.T = np.asarray(T, float) / np.sum(T); self.n_eff = float(n_eff)
        self.f, self.g = self.T.sum(1), self.T.sum(0)
        self.fb, self.gb = bands(self.f, n_eff, alpha), bands(self.g, n_eff, alpha)
        self.signs = wall_signs(self.fb, self.gb)
        self.mT, self.mB = tear_margin(self.f, self.g), break_margin(self.f, self.g)
        self.MI = H(self.f) + H(self.g) - H(self.T)

def regime(A, B, break_eps=0.01):
    """bend: every wall that is decided in both states keeps its sign; tear: some wall decided in both changes sign;
    tear?: the point estimates change side of a wall that is undecided in at least one of the two states;
    break: some margin category whose band in A lies above break_eps has a band in B below break_eps."""
    brk = any((A.fb[0][i] > break_eps and B.fb[1][i] < break_eps) for i in range(len(A.f))) or \
          any((A.gb[0][j] > break_eps and B.gb[1][j] < break_eps) for j in range(len(A.g)))
    flips = [k for k in range(len(A.signs)) if A.signs[k] * B.signs[k] == -1]
    pA, pB = np.sign(wall_values(A.f, A.g)), np.sign(wall_values(B.f, B.g))
    soft = [k for k in range(len(A.signs)) if pA[k] * pB[k] < 0 and k not in flips]
    walls_list = walls(len(A.f), len(A.g))
    kind = "break" if brk else ("tear" if flips else ("tear?" if soft else "bend"))
    return dict(kind=kind, walls_crossed=[walls_list[k] for k in flips], walls_maybe=[walls_list[k] for k in soft],
                stress=(tv(A.f, B.f) + tv(A.g, B.g)) / A.mT if A.mT > 0 else float('inf'),
                change=tv(A.f, B.f) + tv(A.g, B.g), mT=A.mT, mB=A.mB, dMI=B.MI - A.MI)

# ------------------------------------------------------------------ CbD for two contents in two contexts
def cbd_rank2(T1, T2):
    """T1 = 2x2 law of (A, B) in context 1, T2 = 2x2 law of (A, B) in context 2 (rows A = 0/1, columns B = 0/1).
    As a cyclic system of rank 2: context 0 holds (V_0 = A_1, W_1 = B_1), context 1 holds (V_1 = B_2, W_0 = A_2)."""
    T1 = np.asarray(T1, float) / np.sum(T1); T2 = np.asarray(T2, float) / np.sum(T2)
    tabs = [T1, T2.T]
    c, s, D = criterion(tabs)
    return dict(contextual=bool(c), lp_contextual=not lp_noncontextual(tabs), s_odd=s, Delta=D, CNTX=cntx(tabs),
                direct_A=abs(T1.sum(1)[1] - T2.sum(1)[1]), direct_B=abs(T1.sum(0)[1] - T2.sum(0)[1]))

def splits(k):
    """All nontrivial dichotomizations of k categories, one per pair {S, S^c}: subsets S containing category 0."""
    return [S for r in range(1, k) for S in itertools.combinations(range(k), r) if 0 in S]

_SPLIT_CACHE = {}
def cuts(k):
    """Order-respecting dichotomizations of k ordered categories: {0..j} against the rest."""
    return [tuple(range(j + 1)) for j in range(k - 1)]

def _split_lp(n, m, ordinal=False):
    """Constraint matrix and connection indicators of the split LP for n x m tables (cached)."""
    sp = cuts if ordinal else splits
    if (n, m, ordinal) not in _SPLIT_CACHE:
        idx = np.array(list(itertools.product(range(n), range(m), range(n), range(m)))); rows = []
        for a, b in itertools.product(range(n), range(m)):
            rows.append(((idx[:, 0] == a) & (idx[:, 1] == b)).astype(float)); rows.append(((idx[:, 2] == a) & (idx[:, 3] == b)).astype(float))
        conA = [np.isin(idx[:, 0], S) == np.isin(idx[:, 2], S) for S in sp(n)]
        conB = [np.isin(idx[:, 1], S) == np.isin(idx[:, 3], S) for S in sp(m)]
        _SPLIT_CACHE[(n, m, ordinal)] = (np.array(rows), -(np.sum(conA, 0) + np.sum(conB, 0)).astype(float))
    return _SPLIT_CACHE[(n, m, ordinal)]

def cbd_split(T1, T2, ordinal=False):
    """Contextuality-by-Default for two contents with any finite number of outcomes in two contexts, in the canonical form of
    Dzhafarov, Cervantes and Kujala (2017): every content is replaced by all its dichotomizations, and the system is noncontextual
    iff there is a coupling of (A_1, B_1, A_2, B_2) with the observed context tables in which every dichotomization of A (and of B)
    is maximally coupled with itself across the two contexts. Solved as a linear program; returns the deficit
    (maximal minus attainable total agreement of the split connections), which is > 0 exactly when the system is contextual.
    ordinal=True uses only the order-respecting cuts (a sensitivity analysis for ordered answer scales). The contextuality threshold
    1e-7 sits above the solver tolerance; reported deficits are compared with it explicitly."""
    from scipy.optimize import linprog
    T1 = np.asarray(T1, float) / np.sum(T1); T2 = np.asarray(T2, float) / np.sum(T2); n, m = T1.shape
    Aeq, c = _split_lp(n, m, ordinal); sp = cuts if ordinal else splits
    beq = np.array([v for a, b in itertools.product(range(n), range(m)) for v in (T1[a, b], T2[a, b])])
    fA1, fA2, gB1, gB2 = T1.sum(1), T2.sum(1), T1.sum(0), T2.sum(0)
    mx = sum(1 - abs(fA1[list(S)].sum() - fA2[list(S)].sum()) for S in sp(n)) + sum(1 - abs(gB1[list(S)].sum() - gB2[list(S)].sum()) for S in sp(m))
    r = linprog(c, A_eq=Aeq, b_eq=beq, bounds=(0, None), method='highs')
    deficit = mx + r.fun
    return dict(contextual=bool(deficit > 1e-7), deficit=float(max(deficit, 0.0)), n_splits=len(sp(n)) + len(sp(m)))

# ------------------------------------------------------------------ churn: observed connection coupling vs maximal
def churn(x, y, cats=None):
    """x, y: the same content observed on the same units in two contexts. Returns the joint table, the observed agreement
    P(X = Y), the maximal agreement sum_i min(f_i, g_i) allowed by the margins (the maximal coupling of CbD), and the excess
    churn = maximal - observed agreement, which no comparison of margins can see."""
    x = np.asarray(x); y = np.asarray(y); cats = cats if cats is not None else sorted(set(x) | set(y))
    T = np.array([[np.sum((x == a) & (y == b)) for b in cats] for a in cats], float); T /= T.sum()
    f, g = T.sum(1), T.sum(0)
    obs = float(np.trace(T)); mx = float(np.minimum(f, g).sum())
    return dict(T=T, cats=cats, agree=obs, max_agree=mx, excess=mx - obs, tv_margins=tv(f, g))

# ------------------------------------------------------------------ gluing gap
def gluing_gap(x1, x2, x3):
    """I(X1; X3 | X2) in bits: zero exactly when the joint law of the three waves is the gluing of (1,2) and (2,3)."""
    x1, x2, x3 = map(np.asarray, (x1, x2, x3)); n = len(x1); I = 0.0
    for b in set(x2):
        m = x2 == b; nb = m.sum()
        for a in set(x1[m]):
            for c in set(x3[m]):
                pac = np.sum(m & (x1 == a) & (x3 == c)) / nb
                if pac == 0: continue
                pa = np.sum(m & (x1 == a)) / nb; pc = np.sum(m & (x3 == c)) / nb
                I += nb / n * pac * math.log2(pac / (pa * pc))
    return I

def bootstrap(fn, arrays, B=500, seed=0):
    rng = np.random.default_rng(seed); n = len(arrays[0]); vals = []
    for _ in range(B):
        idx = rng.integers(0, n, n); vals.append(fn(*[np.asarray(a)[idx] for a in arrays]))
    return np.percentile(vals, [2.5, 97.5])

if __name__ == "__main__":
    # unit tests
    rng = np.random.default_rng(1); ok = True
    # bands cover the truth at about the nominal rate (simultaneous, so at least 95%)
    p = np.array([0.2, 0.5, 0.3]); cover = 0
    for _ in range(2000):
        c = rng.multinomial(300, p) / 300; lo, hi = bands(c, 300); cover += np.all((lo <= p) & (p <= hi))
    print(f"band coverage {cover / 2000:.3f}"); ok &= cover / 2000 >= 0.95
    # tear margin equals the distance used in Bands (example of the architecture note: 3/100)
    f = np.array([0.4, 0.6]); g = np.array([0.3, 0.43, 0.27]); print("tear margin", round(tear_margin(f, g), 6)); ok &= abs(tear_margin(f, g) - 0.03) < 1e-12
    # regime: a reversal decided on both sides is a tear; a small shift is a bend
    A = State(np.outer([0.7, 0.3], [0.4, 0.6]), 2000); B = State(np.outer([0.3, 0.7], [0.4, 0.6]), 2000); C = State(np.outer([0.68, 0.32], [0.4, 0.6]), 2000)
    rAB, rAC = regime(A, B), regime(A, C); print("A->B", rAB['kind'], rAB['walls_crossed'], "| A->C", rAC['kind']); ok &= rAB['kind'] == 'tear' and rAC['kind'] == 'bend'
    # churn: identical columns have no churn; a random relabelling with the same margins has excess churn
    x = rng.integers(0, 2, 5000); y = x.copy(); y2 = rng.permutation(x)
    c1, c2 = churn(x, y), churn(x, y2); print("churn identical", round(c1['excess'], 4), "| permuted", round(c2['excess'], 3)); ok &= c1['excess'] == 0 and c2['excess'] > 0.4
    # gluing gap: a Markov chain has gap ~0; a non-Markov triple has a positive gap
    x1 = rng.integers(0, 2, 20000); x2 = np.where(rng.random(20000) < 0.8, x1, 1 - x1); x3 = np.where(rng.random(20000) < 0.8, x2, 1 - x2)
    x3b = np.where(rng.random(20000) < 0.8, x1, 1 - x1)
    print("gluing gap Markov", round(gluing_gap(x1, x2, x3), 4), "| non-Markov", round(gluing_gap(x1, x2, x3b), 3)); ok &= gluing_gap(x1, x2, x3) < 0.002 and gluing_gap(x1, x2, x3b) > 0.05
    # CbD rank 2: LP and criterion agree on random pairs of tables
    agree = sum((lambda r: r['contextual'] == r['lp_contextual'])(cbd_rank2(rng.dirichlet(np.ones(4)).reshape(2, 2), rng.dirichlet(np.ones(4)).reshape(2, 2))) for _ in range(300))
    print("CbD rank 2: LP = criterion on", agree, "of 300"); ok &= agree == 300
    # split (canonical) CbD reduces to rank 2 for binary contents
    ag2 = 0
    for _ in range(300):
        T1, T2 = rng.dirichlet(np.ones(4)).reshape(2, 2), rng.dirichlet(np.ones(4)).reshape(2, 2)
        ag2 += cbd_split(T1, T2)['contextual'] == cbd_rank2(T1, T2)['contextual']
    print("CbD split = rank 2 on binary contents:", ag2, "of 300"); ok &= ag2 == 300
    # a 3x3 system can be contextual in its splits while its 'first outcome vs rest' reduction is not
    hits = 0
    for _ in range(400):
        T1, T2 = rng.dirichlet(np.ones(9) * 0.5).reshape(3, 3), rng.dirichlet(np.ones(9) * 0.5).reshape(3, 3)
        b = lambda T: np.array([[T[0, 0], T[0, 1:].sum()], [T[1:, 0].sum(), T[1:, 1:].sum()]])
        hits += cbd_split(T1, T2)['contextual'] and not cbd_rank2(b(T1), b(T2))['contextual']
    print("3x3 systems contextual only in their splits:", hits, "of 400")
    print("ALL UNIT TESTS PASS" if ok else "UNIT TEST FAILURE")
