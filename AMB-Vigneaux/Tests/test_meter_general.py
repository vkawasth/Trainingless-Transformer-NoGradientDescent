from amb_vigneaux.meter_general import parse, run_general, build_units, arc_consistency


def kinds(res):
    return sorted(r.kind for r in res)


def test_implication_cycle_alone_is_not_a_contradiction():
    # "A->B, B->C, C->~A" is satisfiable (A false): the example in the attention proposal is not a contradiction
    cl = [parse("u1", "A -> B"), parse("u2", "B -> C"), parse("u3", "C -> ~A")]
    assert run_general(cl) == []
    cl.append(parse("u4", "A"))
    res = run_general(cl)
    assert kinds(res) == ["propagation"] and sorted(res[0].units) == ["u1", "u2", "u3", "u4"]


def test_self_and_overlap():
    res = run_general([parse("u", "A & ~A"), parse("v", "B"), parse("w", "~B")])
    assert kinds(res) == ["overlap", "self"]


def test_liar_parity_cycle_is_cyclic_type():
    cl = [parse("a", "A <-> B"), parse("b", "B <-> C"), parse("c", "C <-> ~A")]
    res = run_general(cl)
    assert kinds(res) == ["cyclic"] and not res[0].acyclic
    # arc consistency leaves every unit non-empty: d0 = 0 but no global section
    assert all(u.support for u in arc_consistency(build_units(cl)).values())


def test_logical_bell_bound_with_frequencies():
    cl = [parse("a", "A <-> B", 0.9), parse("b", "B <-> C", 0.9), parse("c", "C <-> ~A", 0.9)]
    res = run_general(cl)
    assert abs(res[0].cf_lower - 0.7) < 1e-12


def test_f2_fast_path_agrees_and_scales():
    import time
    from amb_vigneaux.meter_general import run_general_f2
    cl = [parse("a", "A <-> B"), parse("b", "B <-> C"), parse("c", "C <-> ~A"), parse("d", "D ^ E"), parse("e", "D")]
    slow, fast = run_general(cl), run_general_f2(cl)
    assert sorted(sorted(r.units) for r in slow) == sorted(sorted(r.units) for r in fast)
    assert run_general_f2([parse("a", "A -> B")]) is None          # not affine: no fast path
    L = 400                                                         # Liar ring on 400 propositions
    ring = [parse(f"u{i}", f"X{i} <-> X{i+1}") for i in range(L - 1)] + [parse("last", f"X{L-1} <-> ~X0")]
    t = time.time(); res = run_general_f2(ring)
    assert len(res) == 1 and len(res[0].units) == L and time.time() - t < 30


def test_f2_cores_are_minimal_on_random_parity_systems():
    import random
    from amb_vigneaux.meter_general import run_general_f2, build_units, satisfiable
    rnd = random.Random(0)
    for trial in range(40):
        cl = []
        for k in range(10):
            a, b = rnd.sample(range(7), 2)
            cl.append(parse(f"u{k}", f"X{a} <-> X{b}" if rnd.random() < .5 else f"X{a} <-> ~X{b}"))
        units = build_units(cl)
        res = run_general_f2(cl)
        assert (len(res) == 0) == satisfiable(units)
        for r in res:
            core = {k: units[k] for k in r.units}
            assert not satisfiable(core)
            for k in r.units:
                sub = {c: units[c] for c in r.units if c != k}
                assert not sub or satisfiable(sub)
