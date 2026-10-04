"""Reduced checks of the claim-learning theorems (examples/tower/claims_theorems.py)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples", "tower"))
import claims_theorems as T


def test_lemma1_graded_distance_is_transport():
    assert T.check_lemma1(trials=80) < 1e-9


def test_theorem2_corollary3_bound_chain():
    viol, far, n = T.check_thm2_cor3(trials=60, n=3000)
    assert viol == 0 and far <= max(1, n // 50)


def test_theorem4_explicit_bound():
    rate, _ = T.check_thm4(trials=40, reps=20)
    assert rate <= 0.1


def test_prop5a_and_5b():
    v, used = T.check_prop5a(trials=120); assert used > 20 and v == 0
    c = T.collapsed_example(); assert c["d2W"] == 0 and abs(c["err"] - c["predicted"]) < 1e-12 and c["err"] > 0


def test_ties_break_the_lower_bound():
    t = T.tie_example(n=8000)
    assert t["kl_gap"] < 1e-15 and t["zero_hits"] > 0 and t["min_err_late"] < 1e-12 < t["half_s"] - 1e-9 <= t["max_err_late"]
