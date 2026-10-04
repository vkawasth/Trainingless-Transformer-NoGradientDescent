"""Structured-posterior monads on finite spaces: exact monad-law tests (fractions), expected failures, conditioning,
naturality of the forgetful map, and a floating-point implementation checked against the exact one."""
import random
from fractions import Fraction as Fr
import pytest
from amb_vigneaux import sp

X, Y, Z = [0, 1, 2], ["a", "b"], ["u", "v", "w"]
S3 = sp.Subspace(3)
MONOIDS = {"FactorSet": (sp.FactorSet, lambda r: sp.FactorSet.of(*r.sample("fghij", r.randint(0, 2)))),
           "FactorMultiset": (sp.FactorMultiset, lambda r: sp.FactorMultiset.of(*r.choices("fgh", k=r.randint(0, 2)))),
           "Trace": (sp.Trace, lambda r: sp.Trace.of(*r.choices("efg", k=r.randint(0, 2)))),
           "Subspace": (S3, lambda r: S3.of(*[[r.randint(-1, 1) for _ in range(3)] for _ in range(r.randint(0, 1))]))}


def rdist(r, keys):
    w = {k: Fr(r.randint(0, 3)) for k in keys}
    if sum(w.values()) == 0: w[keys[0]] = Fr(1)
    return sp.dist(w)


def per_branch_case(M, rec, seed):
    r = random.Random(seed)
    rho = {(x, rec(r)): p for x, p in rdist(r, X).items()}
    Kt = {x: {(y, rec(r)): p for y, p in rdist(r, Y).items()} for x in X}
    Lt = {y: {(z, rec(r)): p for z, p in rdist(r, Z).items()} for y in Y}
    return sp.PerBranch(M), rho, Kt.__getitem__, Lt.__getitem__


def global_case(M, rec, seed):
    r = random.Random(seed)
    rho = (rdist(r, X), rec(r)); Kt = {x: (rdist(r, Y), rec(r)) for x in X}; Lt = {y: (rdist(r, Z), rec(r)) for y in Y}
    return sp.Global(M), rho, Kt.__getitem__, Lt.__getitem__


@pytest.mark.parametrize("name", list(MONOIDS))
def test_per_branch_is_a_monad_for_every_monoid(name):
    M, rec = MONOIDS[name]
    for seed in range(30):
        T, rho, K, L = per_branch_case(M, rec, seed)
        assert all(sp.check_laws(T, rho, K, L, X).values()), (name, seed)


@pytest.mark.parametrize("name", ["FactorSet", "Subspace"])
def test_global_record_is_a_monad_for_semilattices(name):
    M, rec = MONOIDS[name]
    for seed in range(30):
        T, rho, K, L = global_case(M, rec, seed)
        assert all(sp.check_laws(T, rho, K, L, X).values()), (name, seed)


@pytest.mark.parametrize("name", ["FactorMultiset", "Trace"])
def test_global_record_breaks_associativity_without_idempotence(name):
    M, rec = MONOIDS[name]; fails = 0
    for seed in range(30):
        T, rho, K, L = global_case(M, rec, seed); law = sp.check_laws(T, rho, K, L, X)
        assert law["left_identity"] and law["right_identity"] and law["naturality_of_U"]
        fails += not law["associativity"]
    assert fails > 0                                      # a record reachable along two paths is counted twice


def test_minimal_counterexample_multiset():
    """x in {0, 1} both lead to y = 'a' whose kernel L records factor 'f': aggregated once vs twice"""
    T = sp.Global(sp.FactorMultiset); f = sp.FactorMultiset.of("f")
    rho = ({0: Fr(1, 2), 1: Fr(1, 2)}, sp.FactorMultiset.unit)
    K = lambda x: ({"a": Fr(1)}, sp.FactorMultiset.unit); L = lambda y: ({"u": Fr(1)}, f)
    lhs = T.bind(T.bind(rho, K), L); rhs = T.bind(rho, lambda x: T.bind(K(x), L))
    assert lhs[0] == rhs[0] and lhs[1] == f and rhs[1] == sp.FactorMultiset.of("f", "f")


def test_conditioning_changes_posterior_and_provenance():
    M = sp.Product(sp.FactorSet, sp.Trace); T = sp.PerBranch(M)
    rho = {(0, M.unit): Fr(1, 2), (1, M.unit): Fr(1, 2)}
    post = T.condition(rho, lambda x: Fr(3, 4) if x == 1 else Fr(1, 4), e="obs1", factor="L1", trace_slot=1, factor_slot=0)
    assert T.U(post) == {0: Fr(1, 4), 1: Fr(3, 4)}
    assert set(T.records(post)) == {(frozenset({"L1"}), ("obs1",))}
    G = sp.Global(M); gpost = G.condition(({0: Fr(1, 2), 1: Fr(1, 2)}, M.unit), lambda x: Fr(x), e="e", factor="F", trace_slot=1, factor_slot=0)
    assert gpost == ({1: Fr(1)}, (frozenset({"F"}), ("e",)))
    with pytest.raises(ZeroDivisionError):
        T.condition(rho, lambda x: 0)


def test_subspace_monoid_is_span():
    a = S3.of([1, 0, 0]); b = S3.of([0, 1, 0]); c = S3.of([1, 1, 0])
    assert S3.mul(a, b) == S3.mul(a, c) == S3.of([1, 0, 0], [0, 1, 0]) and S3.mul(a, a) == a


def test_floating_point_matches_exact():
    """the same constructions with float probabilities agree with the exact ones to rounding"""
    M, rec = MONOIDS["Trace"]
    for seed in range(10):
        T, rho, K, L = per_branch_case(M, rec, seed)
        fl = lambda d: {k: float(v) for k, v in d.items()}
        exact = T.bind(T.bind(rho, K), L); approx = T.bind(T.bind(fl(rho), lambda x: fl(K(x))), lambda y: fl(L(y)))
        assert set(exact) == set(approx) and all(abs(float(exact[k]) - approx[k]) < 1e-12 for k in exact)


def test_multimodal_stress_per_branch_keeps_hypotheses_apart():
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples", "sp"))
    import multimodal as mm
    PB, GL = sp.PerBranch(mm.M), sp.Global(mm.M)
    rho_pb = PB.bind({("A", mm.M.unit): Fr(1, 2), ("B", mm.M.unit): Fr(1, 2)}, lambda h: {(x, mm.REC[h]): p for x, p in mm.MU[h].items()})
    rho_gl = GL.bind(({"A": Fr(1, 2), "B": Fr(1, 2)}, mm.M.unit), lambda h: (mm.MU[h], mm.REC[h]))
    assert PB.U(rho_pb) == GL.U(rho_gl)                                   # Giry sees only the mixture
    assert all(PB.given_record(rho_pb, mm.REC[h]) == mm.MU[h] for h in "AB")   # per-branch recovers both components
    assert rho_gl[1][0] == mm.REC["A"][0] | mm.REC["B"][0]                # the global record is the union
    post = PB.condition(rho_pb, lambda x: Fr(int(x[1] == 1)))
    zA = sum(p for x, p in mm.MU["A"].items() if x[1] == 1); zB = sum(p for x, p in mm.MU["B"].items() if x[1] == 1)
    assert PB.records(post)[mm.REC["A"]] == zA / (zA + zB) == Fr(25, 41)  # exact Bayes model weight
