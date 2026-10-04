"""Posterior + shadow algebra (alternatives (+), joint factors (x), conditioning) and the context comonad: exact laws."""
import random
from fractions import Fraction as Fr
from amb_vigneaux import shadow as sh

X = ["a", "b", "c"]


def rmu(r):
    w = {x: Fr(r.randint(1, 4)) for x in X}; z = sum(w.values()); return {x: v / z for x, v in w.items()}


def rsp(r, tag):
    k = r.randint(1, 2); return sh.SP([(r.randint(1, 3), rmu(r), {"%s%d" % (tag, i)}) for i in range(k)])


def test_oplus_laws():
    for s in range(20):
        r = random.Random(s); a, b, c = rsp(r, "a"), rsp(r, "b"), rsp(r, "c"); l, m = Fr(r.randint(1, 4), 5), Fr(r.randint(1, 4), 5)
        assert sh.oplus(a, b, l) == sh.oplus(b, a, 1 - l)                                    # commutative
        assert sh.oplus(a, a, l) == a                                                       # idempotent
        lhs = sh.oplus(sh.oplus(a, b, l), c, m)                                             # associative with induced weights
        rhs = sh.oplus(a, sh.oplus(b, c, (1 - l) * m / (1 - l * m)), l * m)
        assert lhs == rhs


def test_otimes_commutative_associative():
    for s in range(20):
        r = random.Random(100 + s); a, b, c = rsp(r, "a"), rsp(r, "b"), rsp(r, "c")
        assert sh.otimes(a, b) == sh.otimes(b, a)
        assert sh.otimes(sh.otimes(a, b), c) == sh.otimes(a, sh.otimes(b, c))


def test_distributive_law_only_with_bayes_reweighting():
    fixed_fail = 0
    for s in range(20):
        r = random.Random(200 + s); a, b = rsp(r, "a"), rsp(r, "b"); l = Fr(r.randint(1, 4), 5)
        L = {x: Fr(r.randint(0, 3)) for x in X}; L["a"] += 1
        lhs = sh.cond(sh.oplus(a, b, l), L, "e")
        assert lhs == sh.oplus(sh.cond(a, L, "e"), sh.cond(b, L, "e"), sh.bayes_weight(a, b, l, L))   # Bayes: exact
        fixed_fail += lhs != sh.oplus(sh.cond(a, L, "e"), sh.cond(b, L, "e"), l)
    assert fixed_fail > 0                                                                   # fixed weights: fails


def test_U_is_a_homomorphism():
    for s in range(20):
        r = random.Random(300 + s); a, b = rsp(r, "a"), rsp(r, "b"); l = Fr(r.randint(1, 4), 5)
        Ua, Ub = a.U(), b.U()
        assert sh.oplus(a, b, l).U() == {x: l * Ua[x] + (1 - l) * Ub[x] for x in X}
        prod = {x: Ua[x] * Ub[x] for x in X}; z = sum(prod.values())
        assert sh.otimes(a, b).U() == {x: v / z for x, v in prod.items()}


def test_alternatives_stay_apart_and_get_bayes_weights():
    A = sh.SP.point({"a": Fr(3, 4), "b": Fr(1, 4)}, {"A"}); B = sh.SP.point({"a": Fr(1, 4), "b": Fr(3, 4)}, {"B"})
    AB = sh.oplus(A, B, Fr(1, 2)); C = sh.SP.point(AB.U(), {"C"})
    assert AB.U() == C.U() and AB != C and len(AB.branches()) == 2                          # Giry cannot tell; the shadow can
    post = sh.cond(AB, {"a": 1, "b": 0}, "saw a")
    w = {tuple(sorted(S - {"saw a"}))[0]: wt for wt, _, S in post.branches()}
    assert w == {"A": Fr(3, 4), "B": Fr(1, 4)}                                              # exact posterior hypothesis weights


def test_context_comonad_laws():
    for s in range(15):
        r = random.Random(400 + s); rho = rmu(r); x = r.choice(X); c = sh.ctx(x, rho, {"S%d" % s})
        assert sh.eps(sh.delta(c)) == c                                                     # counit (left)
        assert sh.cmap(sh.eps, sh.delta(c)) == c                                            # counit (right)
        assert sh.delta(sh.delta(c)) == sh.cmap(sh.delta, sh.delta(c))                       # coassociativity
        f = {"a": 0, "b": 1, "c": 1}.get                                                    # functoriality on a map X -> {0,1}
        assert sh.eps(sh.cmap(f, c)) == f(sh.eps(c))


def test_tower_paper_theorems_independent():
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples", "tower"))
    import check_tower_independent as T
    assert T.check_61(15) and T.check_98(15)


def test_problem6_collapsed_claim_passes_level_one_fails_level_two():
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples", "tower"))
    import problem6 as P
    res = P.part_b(trials=150, seed=1)
    b = res["by_claim"]
    assert b["collapsed"]["d1"] < 1e-9 and b["collapsed"]["d2"] > 1.5          # level-one admissible, level-two obstructed
    assert abs(b["collapsed"]["E0"] - b["bayes"]["E0"]) < 1e-12               # identical now
    assert b["collapsed"]["E1"] > 2 * b["bayes"]["E1"]                         # worse after updating
