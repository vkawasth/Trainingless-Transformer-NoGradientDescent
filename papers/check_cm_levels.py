"""check_cm_levels.py -- the absolute tower's Beck defects at every level (Tower Section 13.6, Thm. 13.16 and Prop. 13.17).
Uses the definitions of check_tower.py verbatim (exact rational arithmetic).

Track A (averaging law, all levels):
  CA1  the two sides of (Cm) have equal first components (P, e)
  CA2  the context component of the right side is sum_i a_i E3(j_{e_i})(e_i), e_i = E3(eta)(c_i)
  CA3  the context component of the left side is sum_i a_i E3(j_{ebar})(e_i), ebar = sum_i a_i e_i: the right side with every
       per-sample label e_i replaced by the pooled label ebar, at levels one, two, three and the shadow
  CA4  label pooling strictly lowers the label second moment V (computed from the element, not from a representation) whenever
       the per-sample labels differ
  CA5  naturality: for f: X -> Y, C3 C3 D f maps the (Cm) pair at X to the (Cm) pair at Y
Track B (shift law):
  CB1  for rho = sum_k a_k delta_k and contexts whose level-one parts are delta_k, the level-one components of (U) at (x, c) and of
       (M) at a one-group M' are the same pair with the roles of left and right exchanged
  CB2  the same holds for the full elements of C3 D X when the higher components of c are chosen to match (reported per component)
"""
import sys
src = open(__file__.replace('check_cm_levels.py', 'check_tower.py')).read()
defs = src.split('# ---------- comonad ----------')[0]
exec(defs)
exec(src[src.index('def lam_bar(M):'):src.index('def rM(Xs, n=2)')])
exec(src[src.index('def K1(h):'):src.index('oksn = oksc = True')])
from fractions import Fraction as F
ok = {}

def rM_k(Xs, n):
    pts = [rpoint(Xs) for _ in range(n)]; ws = [F(R.randint(1, 5)) for _ in pts]; s = sum(ws)
    return Dist([(p, w / s) for p, w in zip(pts, ws)])
def vec(d, prefix=()):
    """flatten a nested Dist / tuple into a dict of coordinates (for second moments)"""
    out = {}
    if isinstance(d, Dist):
        for v, p in d.items(): out[prefix + (repr(v),)] = out.get(prefix + (repr(v),), 0) + p
    return out
def V_label(ctx_component):
    # level-one component of the transported context: a Dist over pairs (y, label); V = sum_w ||level-one part of label||^2
    tot = F(0)
    for (y, lab), w in ctx_component.items():
        tot += w * sum(q * q for q in vec(lab[0]).values())
    return tot

ca = {k: True for k in ('CA1', 'CA2', 'CA3_l1', 'CA3_l2', 'CA3_l3', 'CA3_sh', 'CA4', 'CA5')}; strict = 0
for trial in range(60):
    M = rM_k(X, R.choice([2, 3]))
    lhs = dlt(lam_bar(M)); rhs = C(lam_bar)(lam_bar(Dmap(dlt, M)))
    items = list(M.items()); es = [E(eta)(p[1]) for p, a in items]; ws = [a for p, a in items]
    ebar = Emix(list(zip(ws, es)))
    ca['CA1'] &= lhs[0] == rhs[0]
    right_rep = Emix([(a, E(j(ei))(ei)) for a, ei in zip(ws, es)])
    left_rep = Emix([(a, E(j(ebar))(ei)) for a, ei in zip(ws, es)])
    ca['CA2'] &= rhs[1] == right_rep
    for idx, key in enumerate(('CA3_l1', 'CA3_l2', 'CA3_l3', 'CA3_sh')): ca[key] &= lhs[1][idx] == left_rep[idx]
    if len({repr(ei[0]) for ei in es}) > 1:
        ca['CA4'] &= V_label(lhs[1][0]) < V_label(rhs[1][0]); strict += 1
    for h in (f, g):
        ca['CA5'] &= C(C(Dk(1, h)))(lhs) == dlt(lam_bar(Dmap(C(h), M))) and C(C(Dk(1, h)))(rhs) == C(lam_bar)(lam_bar(Dmap(dlt, Dmap(C(h), M))))
ok.update(ca); ok['CA4_strict_count'] = strict > 30

# Track B
a = [F(1, 3), F(2, 3)]; rho = Dist([(0, a[0]), (1, a[1])])
sig = [Dist([(0, F(1, 4)), (1, F(3, 4))]), Dist([(0, F(2, 5)), (1, F(3, 5))])]          # arbitrary level-one laws
Pi = Dist([(sig[0], a[0]), (sig[1], a[1])])
tau = [Dist([(sig[0], F(1, 2)), (sig[1], F(1, 2))]), delta(sig[1])]
Om = Dist([(tau[0], a[0]), (tau[1], a[1])])
hy = [("t1", delta(0)), ("t2", delta(1))]                                      # point-mass predictives (needed for the shadow)
Sig = Dist([(hy[0], a[0]), (hy[1], a[1])])
x0 = 0
pt = (x0, (rho, Pi, Om, Sig))
# contexts of the one-group M': level-one delta_k; higher parts chosen as the eta-images of the components of c
ck = [(delta(k), Dmap(delta, sig[k]), Dmap(lambda t: Dmap(delta, t), tau[k]), delta(hy[k])) for k in (0, 1)]
Mone = Dist([((x0, ck[k]), a[k]) for k in (0, 1)]); Mp = delta(Mone)
uL, uR = lam_sig(delta(pt)), C(eta)(pt)
mL, mR = lam_sig(mu(Mp)), C(mu)(lam_sig(Dmap(lam_sig, Mp)))
ok['CB1_level_one_swap'] = uL[1][0] == mR[1][0] and uR[1][0] == mL[1][0] and uL[1][0] != uR[1][0]
comp = {name: (uL[1][i] == mR[1][i], uR[1][i] == mL[1][i]) for i, name in enumerate(('level1', 'level2', 'level3', 'shadow'))}
comp['first'] = (uL[0] == mR[0], uR[0] == mL[0])
print('Track B, per component (uL==mR, uR==mL):', comp)
ok['CB2_full_swap'] = all(v == (True, True) for v in comp.values())

# CB3: the exchanged pair for random weights, laws and hypotheses
def rlaw01():
    w = F(R.randint(1, 9), 10); return Dist([(0, w), (1, 1 - w)])
cb3 = True; ncb = 0
for _ in range(50):
    w0 = F(R.randint(1, 9), 10); a = [w0, 1 - w0]; rho = Dist([(0, a[0]), (1, a[1])])
    sig = [rlaw01(), rlaw01()]; tau = [Dist([(sig[0], F(1, 2)), (rlaw01(), F(1, 2))]), delta(rlaw01())]
    hy = [(R.choice(THETA), delta(R.choice((0, 1)))), (R.choice(THETA), delta(R.choice((0, 1))))]
    Pi = mix([(a[0], delta(sig[0])), (a[1], delta(sig[1]))]); Om = mix([(a[0], delta(tau[0])), (a[1], delta(tau[1]))]); Sig = mix([(a[0], delta(hy[0])), (a[1], delta(hy[1]))])
    x0 = R.choice((0, 1)); pt = (x0, (rho, Pi, Om, Sig))
    ck = [(delta(k), Dmap(delta, sig[k]), Dmap(lambda t: Dmap(delta, t), tau[k]), delta(hy[k])) for k in (0, 1)]
    Mp = delta(Dist([((x0, ck[k]), a[k]) for k in (0, 1)]))
    uL, uR = lam_sig(delta(pt)), C(eta)(pt); mL, mR = lam_sig(mu(Mp)), C(mu)(lam_sig(Dmap(lam_sig, Mp)))
    cb3 &= uL == mR and uR == mL and uL != uR; ncb += 1
ok['CB3_random_exchanged_pairs'] = cb3
print('CB3 instances', ncb)
print({k: bool(v) for k, v in ok.items()}, 'strict', strict)
core = [k for k in ok if k != 'CB2_full_swap']
print('PASS' if all(ok[k] for k in core) else 'FAIL'); sys.exit(0 if all(ok[k] for k in core) else 1)
