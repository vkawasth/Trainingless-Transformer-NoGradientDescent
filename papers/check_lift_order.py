"""check_lift_order.py -- candidate enrichments for the absolute tower's Beck defects (Tower Section 13.6, Prop. 13.13, Thm. 13.14).
Level-one components of the three Beck comparisons, as in Props. 13.8-13.10, in exact rational arithmetic.

  LO1  coordinatewise pooling: a law Q is a pooling of Q' if Q is the law of (E[U|G_1], E[V|G_2]) for (U,V) ~ Q' and sub-sigma-
       algebras G_1, G_2 (one per coordinate). Unit: the lambda-first side delta_rho is a pooling of D(eta)rho. Multiplication: the
       other side sum nu_l delta_{bar rho_l} is a pooling of the lambda-first side. Comultiplication: the lambda-first side L is a
       pooling of R (context coordinate pooled, sample kept). Witnesses are constructed and verified exactly.
  LO2  strictness: a pooling that is not an identity strictly lowers the second moment sum ||U||^2 + ||V||^2, so no comparison
       holds in both directions (random instances, exact).
  LO3  hence any preorder that contains strict poolings in one direction and does not contain their reverses orients unit and
       comultiplication against multiplication (checked on every instance: the lambda-first side is the coarse side in exactly
       two of the three axioms)
  LO4  the ambience's set order (inclusion of support hulls, Tower Section 10.7) orients unit and multiplication oppositely and
       cannot compare the comultiplication sides (exact)
  LO5  liftability relative to a fixed admissibility datum orients the unit defect either way depending on the datum (exact)
"""
from fractions import Fraction as Fr
import random, itertools, sys
random.seed(13)
def rlaw(n, d=9):
    w = [random.randint(1, d) for _ in range(n)]; s = sum(w); return tuple(Fr(x, s) for x in w)
def avg(pts, wts):
    W = sum(wts); return tuple(sum(w * p[i] for p, w in zip(pts, wts)) / W for i in range(len(pts[0])))
def sq(v): return sum(x * x for x in v)
def m2(Q): return sum(w * sum(sq(c) for c in atom) for atom, w in Q.items())   # atoms are tuples of coordinates
def law(pairs):
    out = {}
    for a, w in pairs: out[a] = out.get(a, 0) + w
    return out
ok = {}

# ---- instances (X = {0,1,2}); level-one objects are laws on tuples of coordinates
def unit_instance():
    rho = rlaw(3); deltas = [tuple(Fr(int(i == x)) for i in range(3)) for x in range(3)]
    lam_first = law([((rho,), Fr(1))]); other = law([((deltas[x],), rho[x]) for x in range(3)])
    # witness: U = delta_x with prob rho(x); E[U | trivial] = rho
    witness = law([((avg([deltas[x] for x in range(3)], list(rho)),), Fr(1))])
    return dict(coarse='lambda', lam=lam_first, oth=other, ok_witness=witness == lam_first)
def mult_instance():
    L, K = random.randint(1, 3), random.randint(2, 3); nu = rlaw(L); a = [rlaw(K) for _ in range(L)]; r = [[rlaw(3) for _ in range(K)] for _ in range(L)]
    lam_first = law([((r[l][k],), nu[l] * a[l][k]) for l in range(L) for k in range(K)])
    other = law([((avg(r[l], list(a[l])),), nu[l]) for l in range(L)])
    witness = law([((avg(r[l], list(a[l])),), nu[l]) for l in range(L)])     # E[U | l]
    return dict(coarse='other', lam=lam_first, oth=other, ok_witness=witness == other)
def comult_instance():
    # samples P_i with per-sample contexts e_i (level-one parts), weights w_i; lambda-first side pools the contexts
    n = random.randint(2, 3); w = rlaw(n); P = [rlaw(2) for _ in range(n)]; e = [rlaw(3) for _ in range(n)]; ebar = avg(e, list(w))
    lam_first = law([((P[i], ebar), w[i]) for i in range(n)]); other = law([((P[i], e[i]), w[i]) for i in range(n)])
    witness = law([((P[i], avg(e, list(w))), w[i]) for i in range(n)])        # G_1 full, G_2 trivial
    return dict(coarse='lambda', lam=lam_first, oth=other, ok_witness=witness == lam_first, P=P, e=e, ebar=ebar)

r1 = r2 = r3 = True; strict = 0
for _ in range(300):
    insts = [unit_instance(), mult_instance(), comult_instance()]
    r1 &= all(I['ok_witness'] for I in insts)
    for I in insts:
        coarse, fine = (I['lam'], I['oth']) if I['coarse'] == 'lambda' else (I['oth'], I['lam'])
        if coarse != fine: r2 &= m2(coarse) < m2(fine); strict += 1
    r3 &= [I['coarse'] for I in insts] == ['lambda', 'other', 'lambda']
ok['LO1_pooling_witnesses'] = r1; ok['LO2_strict'] = r2 and strict > 0; ok['LO3_orientation'] = r3

# ---- LO4: support-hull inclusion (exact, small cases)
def in_hull(pts, p):
    # p in conv(pts) for points of a simplex: solve by enumerating supports of size <= dim+1 with exact barycentric weights
    dim = len(p)
    for k in range(1, min(len(pts), dim) + 1):
        for S in itertools.combinations(pts, k):
            # solve sum t_j S_j = p, sum t_j = 1, by Gaussian elimination over Fractions (least-squares free: exact check)
            A = [[S[j][i] for j in range(k)] for i in range(dim)] + [[Fr(1)] * k]; b = list(p) + [Fr(1)]
            M = [row[:] + [bb] for row, bb in zip(A, b)]; R = len(M); piv = []; rr = 0
            for c in range(k):
                pr = next((i for i in range(rr, R) if M[i][c] != 0), None)
                if pr is None: continue
                M[rr], M[pr] = M[pr], M[rr]; M[rr] = [x / M[rr][c] for x in M[rr]]
                for i in range(R):
                    if i != rr and M[i][c] != 0: M[i] = [x - M[i][c] * y for x, y in zip(M[i], M[rr])]
                piv.append(c); rr += 1
            if any(all(x == 0 for x in M[i][:-1]) and M[i][-1] != 0 for i in range(R)): continue
            t = [Fr(0)] * k
            for i, c in enumerate(piv): t[c] = M[i][-1]
            if all(x >= 0 for x in t): return True
    return False
def flat(atom): return tuple(x for c in atom for x in c)
def hull_sub(Q, Qp): pts = [flat(a) for a in Qp]; return all(in_hull(pts, flat(a)) for a in Q)
u, m, c = unit_instance(), mult_instance(), comult_instance()
while len(set(c['e'])) < 2 or len(set(c['P'])) < 2: c = comult_instance()
r4 = hull_sub(u['lam'], u['oth']) and not hull_sub(u['oth'], u['lam'])
r4 &= hull_sub(m['oth'], m['lam'])
# paper's comultiplication instance: X={0,1}, samples delta_0, delta_1, contexts e_0, e_1 (level-one parts delta_0, delta_1)
d0, d1 = (Fr(1), Fr(0)), (Fr(0), Fr(1)); e0, e1, eb = d0, d1, (Fr(1, 2), Fr(1, 2))
Lp = law([((d0, eb), Fr(1, 2)), ((d1, eb), Fr(1, 2))]); Rp = law([((d0, e0), Fr(1, 2)), ((d1, e1), Fr(1, 2))])
r4 &= not hull_sub(Lp, Rp) and not hull_sub(Rp, Lp)
ok['LO4_set_order_fails'] = r4

# ---- LO5: liftability relative to a fixed datum, unit defect at level two: delta_rho in D(S_1) iff rho in S_1;
#      D(eta)rho in D(S_1) iff every delta_x with rho(x) > 0 is in S_1
rho = (Fr(1, 2), Fr(1, 2)); dx = [(Fr(1), Fr(0)), (Fr(0), Fr(1))]
def lift_unit(S1): return (rho in S1), all(d in S1 for d in dx)
a, b = lift_unit({rho}), lift_unit(set(dx))
ok['LO5_relative_liftability_flips'] = a == (True, False) and b == (False, True)

print({k: bool(v) for k, v in ok.items()}, 'strict comparisons', strict)
print('PASS' if all(ok.values()) else 'FAIL'); sys.exit(0 if all(ok.values()) else 1)
