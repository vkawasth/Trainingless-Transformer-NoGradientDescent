"""Exact primary decomposition check: intersect the component ideals by elimination (I n J = (t I + (1 - t) J) n k[x])."""
import itertools, time
import sympy as sp
def intersect(I, J, vars_):
    t = sp.Symbol("t_elim")
    G = sp.groebner([t * f for f in I] + [(1 - t) * g for g in J], t, *vars_, order="lex")
    return [g for g in G.exprs if not g.has(t)]
for m in (3, 4):
    w1 = sp.Symbol("w1"); d = sp.symbols(f"d1:{m+1}"); vars_ = list(d) + [w1]      # mu = c is a common linear factor; drop it
    comps = [[w1], [1 - w1]] + [[d[j] for j in range(m) if j != i] for i in range(m)]
    t0 = time.time(); acc = comps[0]
    for C in comps[1:]: acc = intersect(acc, C, vars_)
    I = [w1 * (1 - w1) * d[i] * d[j] for i, j in itertools.combinations(range(m), 2)]
    GA = sp.groebner(acc, *vars_, order="grevlex"); GI = sp.groebner(I, *vars_, order="grevlex")
    print(f"m={m}: intersection of the {len(comps)} components == (w1 w2 d_i d_j): {GA.exprs == GI.exprs}   ({time.time()-t0:.1f}s)")
