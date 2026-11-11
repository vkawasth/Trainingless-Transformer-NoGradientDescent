"""Checks for the strand-addition proposition (module pce/stranddefect.py). P_n is the pure braid group on n labelled sources; forget
deletes the newcomer; a section s puts a newcomer next to an old history; the defect of an enlarged history bp over b is bp s(b)^{-1},
defined only when forget(bp) = b.
  SD1 IDENTITY  forgetting a strand is well defined on braids and a homomorphism of pure braid groups (n + 1 = 3, 4, 5)
  SD2 IDENTITY  the three kinds of section (far away; behind everything at position k + 1; echo of source i) respect the braid relations
                and satisfy forget(s(b)) = b: each is a splitting of the Fadell-Neuwirth sequence
  SD3 IDENTITY  compatibility: if forget(bp) = b the defect lies in the kernel of forget (for every section); if not, it does not, so the
                condition cannot be dropped; the defect of s(b) itself is trivial
  SD4 IDENTITY  the kernel is free on the newcomer's loops A_{1,n+1}..A_{n,n+1}: the loop word of the defect is a homomorphism, sends
                A_{i,n+1} to the letter i, and the product of the letters gives back the defect (Artin's combing); distinct defects have
                distinct loop words
  SD5 IDENTITY  first layer: the exponent sums of the loop word are the linking numbers lk(i, newcomer) of bp (relative to s_far, which
                by SD7 is every boundary section); relative to the echo of i they drop by lk(i, j) of b; a newcomer that echoes i
                exactly has zero defect relative to e_i and defect (lk(i, j))_j relative to s_far
  SD6 IDENTITY  second layer: degree-2 Magnus coefficients satisfy M_ij + M_ji = d_i d_j; the commutator newcomer [A_13, A_23] has
                d = 0 and (M_12, M_21) = (1, -1); when all linking numbers vanish M_12 is Milnor's coefficient of the closure (as in
                braidknot.milnor); on the brokerage histories the antisymmetric part (M_12 - M_21)/2 of the third visitor's defect is the
                triple invariant mu_123 of Proposition 'brokerage'
  SD7 IDENTITY  dependence on the section: every boundary section (newcomer behind everything at position k + 1) becomes s_far once the
                newcomer is moved to the right end behind the others, so 'a newcomer from outside' is one section; an echo section is
                c_i . s_far with c_i a crossed homomorphism (c_i(uv) = c_i(u) s(u) c_i(v) s(u)^{-1}) whose first layer is lk(i, .).
                (A first version claimed boundary sections differ at the second layer; that was wrong and is withdrawn.)
  SD8 IDENTITY  regions: forgetting two newcomers in either order gives the same braid, and the defects compose by the chain rule
                Delta(bpp, b) = Delta(bpp, bp) s(Delta(bp, b))
"""
import os, sys, json, itertools
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import stranddefect as S, braidknot as BK, brokerage as BR
results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
inv, red, same = BK._inv, BK._red, BK.same_braid
rng = np.random.default_rng(2026)
def rpure(N, L):
    w = []
    for _ in range(L):
        i, j = sorted(rng.choice(np.arange(1, N + 1), 2, replace=False).tolist()); g = S.pure_gen(i, j)
        w += g if rng.random() < 0.5 else inv(g)
    return w
def rword(N, L): return [int(rng.integers(1, N)) * int(rng.choice([1, -1])) for _ in range(L)]
def scramble(w, N):
    """an equal braid written differently: insert r r^{-1} at random places and apply random braid relations"""
    w = list(w)
    for _ in range(4):
        k = int(rng.integers(0, len(w) + 1)); r = rword(N, 3); w = w[:k] + r + inv(r) + w[k:]
    for _ in range(30):
        k = int(rng.integers(0, max(1, len(w) - 2)))
        a, b, c = (w[k:k + 3] + [0, 0, 0])[:3]
        if a > 0 and c == a and b > 0 and abs(a - b) == 1: w[k:k + 3] = [b, a, b]
        elif a and b and abs(abs(a) - abs(b)) > 1: w[k], w[k + 1] = b, a
    return w
def trivial(w, N): return same(w, [], N)
SECS = lambda n: [("far",)] + [("back", k) for k in range(n + 1)] + [("echo", i) for i in range(1, n + 1)]

# SD1
ok_h = ok_w = True
for N in (3, 4, 5):
    for _ in range(60):
        u, v = rpure(N, 4), rpure(N, 4); k = int(rng.integers(1, N + 1))
        ok_h &= same(S.forget(u + v, N, k), S.forget(u, N, k) + S.forget(v, N, k), N - 1)
        ok_w &= same(S.forget(scramble(u, N), N, k), S.forget(u, N, k), N - 1)
check("SD1 IDENTITY forget is well defined on braids and a homomorphism", ok_h and ok_w, "180 pairs, 180 rewritten words")

# SD2
ok = True
for n in (2, 3, 4):
    rels = [([i, i + 1, i], [i + 1, i, i + 1]) for i in range(1, n - 1)] + [([i, j], [j, i]) for i in range(1, n) for j in range(i + 2, n)]
    for sec in SECS(n):
        ok &= all(same(S.section(a, n, sec), S.section(b, n, sec), n + 1) for a, b in rels)
        for _ in range(20):
            b = rpure(n, 4); ok &= same(S.forget(S.section(b, n, sec), n + 1, S.newcomer_label(n, sec)), b, n)
            ok &= same(S.section(scramble(b, n), n, sec), S.section(b, n, sec), n + 1)
check("SD2 IDENTITY far, behind-at-k and echo sections are splittings: braid relations kept, forget o s = id", ok)

# SD3
ok_in = ok_out = ok_triv = True; nbad = 0
for n in (2, 3, 4):
    for sec in SECS(n):
        lab = S.newcomer_label(n, sec)
        for _ in range(15):
            b = rpure(n, 4); g = rpure(n + 1, 3); bp = S.section(b, n, sec) + g          # an enlarged history
            bp = g + S.section(b, n, sec) if rng.random() < 0.5 else bp
            bfit = S.forget(bp, n + 1, lab); D = S.defect(bp, bfit, n, sec)
            ok_in &= trivial(S.forget(D, n + 1, lab), n)
            ok_triv &= trivial(S.defect(S.section(b, n, sec), b, n, sec), n + 1)
            c = rpure(n, 2)
            if not same(c, [], n):                                                          # a wrong old history
                Dw = list(bp) + inv(S.section(S.forget(bp, n + 1, lab) + c, n, sec)); ok_out &= not trivial(S.forget(Dw, n + 1, lab), n); nbad += 1
check("SD3 IDENTITY with forget(bp) = b the defect is in the kernel; without it, it is not; defect of s(b) is trivial", ok_in and ok_out and ok_triv,
      f"{nbad} incompatible pairs, none in the kernel")

# SD4
ok = True; distinct = 0; pairs = 0
for n in (2, 3, 4):
    ok &= all(S.loop_word(S.pure_gen(i, n + 1), n) == [i] for i in range(1, n + 1))
    Ds = []
    for _ in range(15):
        b = rpure(n, 2); bp = rpure(n + 1, 2) + S.section(b, n, ("far",)) + rpure(n + 1, 1); D = S.defect(bp, S.forget(bp, n + 1, n + 1), n)
        lw = S.loop_word(D, n); ok &= same(S.from_loop_word(lw, n), D, n + 1); Ds.append((D, lw))
    for (D1, l1), (D2, l2) in itertools.combinations(Ds, 2):
        pairs += 1; e = same(D1, D2, n + 1); ok &= (l1 == l2) == e; distinct += not e
    for _ in range(30):
        u, v = rpure(n + 1, 3), rpure(n + 1, 3)
        u = S.defect(u, S.forget(u, n + 1, n + 1), n); v = S.defect(v, S.forget(v, n + 1, n + 1), n)
        ok &= S.loop_word(u + v, n) == red(S.loop_word(u, n) + S.loop_word(v, n))
check("SD4 IDENTITY the defect is a word in the newcomer's loops A_{i,n+1}: homomorphism, combing, injective", ok, f"{pairs} pairs compared, {distinct} distinct")

# SD5
ok = True
def lk_new(word, n, lab): L = S.linking(word, n + 1); return {j: L.get((min(j, lab), max(j, lab)), 0) for j in range(1, n + 2) if j != lab}
for n in (2, 3, 4):
    for _ in range(25):
        b = rpure(n, 3); bp = rpure(n + 1, 2) + S.section(b, n, ("far",)) + rpure(n + 1, 2)
        bb = S.forget(bp, n + 1, n + 1); lw = S.loop_word(S.defect(bp, bb, n), n); d1 = S.magnus_coords(lw, n)[0]
        ok &= d1 == lk_new(bp, n, n + 1)
        for i in range(1, n + 1):                                               # echo of i: drops by lk(i, j) of b
            e = S.section(b, n, ("echo", i)); ok &= trivial(S.defect(e, b, n, ("echo", i)), n + 1)
            Lb = S.linking(b, n); lab = i + 1
            le = lk_new(e, n, lab)
            exp = {(j if j < lab else j): (Lb.get((min(i, j - (1 if j > lab else 0)), max(i, j - (1 if j > lab else 0))), 0) if j != i else 0) for j in range(1, n + 2) if j != lab}
            ok &= le == exp
check("SD5 IDENTITY first layer = linking numbers with the newcomer; an exact echo of i has zero defect relative to e_i and lk(i, j) relative to s_far", ok)

# SD6
ok = True
for n in (2, 3, 4):
    for _ in range(30):
        bp = rpure(n + 1, 5); D = S.defect(bp, S.forget(bp, n + 1, n + 1), n); d1, d2 = S.magnus_coords(S.loop_word(D, n), n)
        ok &= all(d2[(i, j)] + d2[(j, i)] == d1[i] * d1[j] for i in range(1, n + 1) for j in range(1, n + 1) if i != j)
com = S.pure_gen(1, 3) + S.pure_gen(2, 3) + inv(S.pure_gen(1, 3)) + inv(S.pure_gen(2, 3)); d1, d2 = S.magnus_coords(S.loop_word(com, 2), 2)
ok_c = d1 == {1: 0, 2: 0} and (d2[(1, 2)], d2[(2, 1)]) == (1, -1)
rel = set(); cnt = 0
for _ in range(400):
    a, b_, c = (S.pure_gen(*sorted(rng.choice([1, 2, 3], 2, replace=False).tolist())) for _ in range(3))
    w = a + b_ + inv(a) + inv(b_)
    if rng.random() < 0.5: w = c + w + inv(c)
    if any(S.linking(w, 3).values()): continue
    D = S.defect(w, S.forget(w, 3, 3), 2); _, d2 = S.magnus_coords(S.loop_word(D, 2), 2); lk, cc = BK.milnor(w)
    rel.add((d2[(1, 2)], cc[3][0])); cnt += 1
signs = {(x, y) for x, y in rel if x != 0}; ok_m = len(signs) > 0 and (all(x == y for x, y in rel) or all(x == -y for x, y in rel))
mu = []
for m in (1, 2):
    for h in BR.interleavings(m):
        w = BR.braid_word(h); D = S.defect(w, S.forget(w, 3, 3), 2); _, d2 = S.magnus_coords(S.loop_word(D, 2), 2)
        mu.append(((d2[(1, 2)] - d2[(2, 1)]) // 2, BR.mu123(h)))
ok_b = all(x == y for x, y in mu) or all(x == -y for x, y in mu)
sgn_m = "+" if all(x == y for x, y in rel) else "-"; sgn_b = "+" if all(x == y for x, y in mu) else "-"
check("SD6 IDENTITY second layer: shuffle identity, commutator newcomer, Milnor's coefficient, brokerage triple invariant", ok and ok_c and ok_m and ok_b,
      f"zero-linking samples {cnt}, M_12 = {sgn_m} Milnor coefficient on {sorted(rel)}; brokerage: (M_12 - M_21)/2 = {sgn_b} mu_123 on {len(mu)} histories, values {sorted(set(mu))}")

# SD7
ok_bd = ok_cocycle = ok_lk = True
for n in (2, 3):
    for k in range(n + 1):                                                      # boundary sections: one section up to relabelling
        pi = [-x for x in range(k + 1, n + 1)]                                  # move the newcomer from position k+1 to n+1, behind
        ok_bd &= all(same(inv(pi) + S.section_back(b, n, k) + pi, S.section_far(b, n), n + 1) for b in [rpure(n, 3) for _ in range(10)])
    for i in range(1, n + 1):                                                   # echo sections: s_echo = c_i . s_far, c_i a crossed hom
        pi = [-x for x in range(i + 1, n + 1)]
        def c(b): return inv(pi) + S.section_echo(b, n, i) + pi + inv(S.section_far(b, n))
        for _ in range(10):
            u, v = rpure(n, 2), rpure(n, 2)
            ok_cocycle &= same(c(u + v), c(u) + S.section_far(u, n) + c(v) + inv(S.section_far(u, n)), n + 1)
            d1 = S.magnus_coords(S.loop_word(c(u), n), n)[0]; Lu = S.linking(u, n)
            ok_lk &= d1 == {j: (Lu.get((min(i, j), max(i, j)), 0) if j != i else 0) for j in range(1, n + 1)}
check("SD7 IDENTITY all boundary sections are s_far after relabelling; an echo section is c_i . s_far with c_i a crossed homomorphism whose first layer is lk(i, .)",
      ok_bd and ok_cocycle and ok_lk)

# SD8
ok = True
for n in (2, 3):
    for _ in range(25):
        bpp = rpure(n + 2, 5)
        f1 = S.forget(S.forget(bpp, n + 2, n + 2), n + 1, n + 1); f2 = S.forget(S.forget(bpp, n + 2, n + 1), n + 1, n + 1)
        ok &= same(f1, f2, n)
        bp = S.forget(bpp, n + 2, n + 2); b = S.forget(bp, n + 1, n + 1)
        lhs = list(bpp) + inv(S.section_far(S.section_far(b, n), n + 1))
        rhs = S.defect(bpp, bp, n + 1) + S.section_far(S.defect(bp, b, n), n + 1)
        ok &= same(lhs, rhs, n + 2)
check("SD8 IDENTITY nested regions: forgetting in either order agrees; defects compose by the chain rule", ok)
json.dump(dict(milnor_relation=sorted(map(list, rel)), brokerage=sorted(set(mu))), open(os.path.join(HERE, "stranddefect_summary.json"), "w"), indent=1)
print(f"{sum(results)}/{len(results)} passed")
