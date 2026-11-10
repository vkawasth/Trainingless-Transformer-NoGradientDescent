"""Checks for 'Co-revision over any graph' (companion note; module pce/raagbraid.py).
  RG1 IDENTITY  in six drawings (path, star, triangle, square, planar K4, K4 drawn with crossing diagonals) the twists of two edges commute
                exactly when the edges share no source and their segments do not cross
  RG2 IDENTITY  in the crossed drawing of K4 the independent diagonals do not commute, so the event-to-braid map is not a homomorphism
                from A_G (the planarity obstruction, Hanani-Tutte)
  RG3 IDENTITY  the triangle relation: with single twists, a_{12} commutes with a_{02} a_{01} (their product with a_{12} is the boundary twist
                of a disc around the three sources), so single twists are not faithful on a triangle; the same relation appears on an empty
                triangle of the planar K4
  RG4 FINDING   faithfulness on balls: single twists are injective on all words of length <= 5 (path, star) or <= 4 (square); squared twists
                are injective on the same balls for every planar drawing tested, including the triangle and K4
  RG5 IDENTITY  the partially commutative signature (independence = disjoint edges) separates exactly the elements of A_G on all words of
                length <= 4 for the star, the triangle and the square (word problem solved independently by cancellation and Cartier-Foata)
"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import raagbraid as RG
results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
NAMES = ["path4", "star3", "triangle", "square", "k4_planar", "k4_crossed"]; D = {n: RG.drawing(n) for n in NAMES}

# RG1
ok = True; info = {}
for n, d in D.items():
    m = len(d["E"]); bad = 0
    for a in range(m):
        for b in range(a + 1, m):
            com = RG.braid_key(d, [(a, 1), (b, 1)]) == RG.braid_key(d, [(b, 1), (a, 1)])
            exp = RG.independent(d, a, b) and not RG.segments_cross(d, a, b); bad += com != exp
    ok &= bad == 0; info[n] = m * (m - 1) // 2
check("RG1 IDENTITY twists commute iff the edges are disjoint and their segments do not cross", ok, f"pairs tested per drawing {info}")

# RG2
d = D["k4_crossed"]; a, b = d["E"].index((0, 2)), d["E"].index((1, 3))
check("RG2 IDENTITY crossing independent edges: the diagonals of the crossed K4 are independent but their twists do not commute",
      RG.independent(d, a, b) and RG.segments_cross(d, a, b) and RG.braid_key(d, [(a, 1), (b, 1)]) != RG.braid_key(d, [(b, 1), (a, 1)]))

# RG3
t = D["triangle"]; e01, e12, e02 = t["E"].index((0, 1)), t["E"].index((1, 2)), t["E"].index((0, 2))
lhs = RG.braid_key(t, [(e12, 1), (e02, 1), (e01, 1)]); rhs = RG.braid_key(t, [(e02, 1), (e01, 1), (e12, 1)])
prod = RG.braid_key(t, [(e02, 1), (e01, 1), (e12, 1)]); central = all(RG.braid_key(t, [(e02, 1), (e01, 1), (e12, 1), (x, 1)]) == RG.braid_key(t, [(x, 1), (e02, 1), (e01, 1), (e12, 1)]) for x in range(3))
nf_differ = RG.raag_normal_form(t, [(e12, 1), (e02, 1), (e01, 1)]) != RG.raag_normal_form(t, [(e02, 1), (e01, 1), (e12, 1)])
k = D["k4_planar"]; f = lambda u, v: k["E"].index((u, v))
sub = RG.braid_key(k, [(f(0, 1), 1), (f(1, 3), 1), (f(0, 3), 1)]) == RG.braid_key(k, [(f(0, 3), 1), (f(0, 1), 1), (f(1, 3), 1)])
check("RG3 IDENTITY the triangle relation (single twists): a12 commutes with a02 a01; the product is central; same on an empty triangle of K4",
      lhs == rhs and central and nf_differ and sub, "a12 (a02 a01) = (a02 a01) a12 in the braid group but not in the free group A_triangle")

# RG4
res = {}
for n, L in (("path4", 5), ("star3", 5), ("square", 4), ("triangle", 5), ("k4_planar", 4)):
    res[n] = {N: RG.faithful_on_ball(D[n], N, L)[:2] for N in (1, 2)}
ok = (all(res[n][1] == (True, True) for n in ("path4", "star3", "square")) and all(res[n][2] == (True, True) for n in res)
      and res["triangle"][1] == (True, False) and res["k4_planar"][1] == (True, False))
check("RG4 FINDING single twists faithful on balls for triangle-free drawings; squared twists faithful on every planar drawing tested", ok,
      f"(homomorphism, injective) by drawing and power: {res}")
json.dump({n: {str(N): list(v) for N, v in r.items()} for n, r in res.items()}, open(os.path.join(HERE, "raagbraid_summary.json"), "w"), indent=1)

# RG5
ok = True; info = {}
for n in ("star3", "triangle", "square"):
    d = D[n]; byk = {}; byn = {}
    for w in RG.words(len(d["E"]), 4):
        nf = RG.raag_normal_form(d, w); sk = RG.pcsig_key(d, w, 4)
        byn.setdefault(nf, set()).add(sk); byk.setdefault(sk, set()).add(nf)
    ok &= all(len(v) == 1 for v in byn.values()) and all(len(v) == 1 for v in byk.values()); info[n] = len(byn)
check("RG5 IDENTITY the partially commutative signature separates exactly the elements of A_G (words of length <= 4)", ok, f"elements on the ball: {info}")
print(f"{sum(results)}/{len(results)} passed")
