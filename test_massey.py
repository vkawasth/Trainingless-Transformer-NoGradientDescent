from amb_vigneaux.massey import Structure, surprisal_power, cup, delta, add, max_abs, massey_report


def test_massey_powers_of_surprisal_vanish():
    rep = massey_report(sizes=(2, 2, 2), N=4, trials=3)
    for k, v in rep.items():
        assert v < 1e-9, (k, v)


def test_cup_square_nonzero_but_exact():
    S = Structure((2, 3))
    s, u2 = surprisal_power(S, 1), surprisal_power(S, 2)
    assert max_abs(S, cup(s, s), 3) > 1e-2                      # s cup s is a non-zero cochain ...
    assert max_abs(S, add(delta(S, u2), cup(s, s)), 3) < 1e-9   # ... equal to -delta(s^2/2): exact
