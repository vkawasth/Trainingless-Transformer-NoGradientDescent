import random
from fractions import Fraction as Fr
from amb_vigneaux import constructive as C

rng = random.Random(0)


def rlist(X, n=None):
    return [(rng.choice(X), Fr(rng.randint(1, 9), rng.randint(1, 9))) for _ in range(n or rng.randint(1, 5))]


def test_generators_and_the_bijection_relation():
    assert C.equiv([("x", Fr(1)), ("x", Fr(1))], [("x", Fr(1))])          # merging: not a bijection of entries
    assert C.equiv([("x", Fr(1)), ("y", Fr(2))], [("y", Fr(6)), ("x", Fr(3))])   # permutation + scaling


def test_monad_laws_exactly_over_Q():
    X = list("abcd")
    for _ in range(200):
        l = rlist(X)
        assert C.equiv(C.mu([(l, Fr(1))]), l)                                # left unit: mu . eta_D = id
        assert C.equiv(C.mu([(C.eta(x), q) for x, q in l]), l)               # right unit: mu . D(eta) = id
        # associativity on a three-level list
        Psi = [([(rlist(X), Fr(rng.randint(1, 5))) for _ in range(rng.randint(1, 3))], Fr(rng.randint(1, 5))) for _ in range(rng.randint(1, 3))]
        lhs = C.mu(C.mu(Psi))                                                 # mu_X . mu_{D X}
        rhs = C.mu([(C.mu(Phi), p) for Phi, p in Psi])                        # mu_X . D(mu_X)
        assert C.equiv(lhs, rhs)
        # well-definedness of mu: replacing an inner list by an equivalent one does not change the class
        L = [(rlist(X), Fr(rng.randint(1, 5))) for _ in range(3)]
        L2 = [([(x, 3 * q) for x, q in reversed(i)], w) for i, w in L]          # permute and rescale each inner list
        assert C.equiv(C.mu(L), C.mu(L2))
        # naturality of eta and mu for f = upper-casing
        f = str.upper
        assert C.equiv(C.dmap(f, C.mu(L)), C.mu([(C.dmap(f, i), w) for i, w in L]))


def test_commutative_and_affine():
    X = list("abc")
    for _ in range(100):
        a, b = rlist(X), rlist(X)
        assert C.equiv(C.strength_pair(a, b), C.strength_pair_swapped(a, b))   # commutative
        assert C.equiv(C.dmap(lambda x: "*", a), [("*", Fr(1))])             # affine: D(1) = 1


def test_vigneaux_action_is_rational_and_satisfies_the_cocycle_for_tsallis2():
    # S_2(P) = 1 - sum p^2 ; Tsallis-2 cocycle: S_2(X,Y) = S_2(X) + (X . S_2(Y))_2 with the alpha-twisted action
    E = [(i, j) for i in range(2) for j in range(3)]
    l = [(e, Fr(rng.randint(1, 9))) for e in E]
    def S2(lst):
        return 1 - sum(q ** 2 for _, q in C.nf(lst))                          # nf weights are normalised
    w = sum(q for _, q in l); px = {}
    for (i, j), q in l:
        px[i] = px.get(i, Fr(0)) + q / w
    lhs = S2(l)
    rhs = (1 - sum(v ** 2 for v in px.values())) + sum(v ** 2 * S2(C.condition(l, lambda e, i=i: e[0] == i)) for i, v in px.items())
    assert lhs == rhs and isinstance(lhs, Fr)
