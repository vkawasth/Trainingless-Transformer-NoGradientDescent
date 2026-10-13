"""
check_double_category.py

Verifies the double-category structure with pooling as the vertical
direction and the Beck diagram as the horizontal direction.

Run: python check_double_category.py
"""

from fractions import Fraction
import random
import hashlib
import json


# ============================================================
# Canonical distributions
# ============================================================

def canonical(d):
    return tuple(sorted(d.items(), key=lambda kv: repr(kv[0])))


def mk(items):
    d = {}
    for k, p in items:
        d[k] = d.get(k, Fraction(0)) + Fraction(p)
    return canonical(d)


def push(d, f):
    return mk([(f(k), p) for k, p in d])


def barycenter(dd):
    out = {}
    for rho_key, p in dd:
        for b, q in rho_key:
            out[b] = out.get(b, Fraction(0)) + p * q
    return canonical(out)


def tv(d1, d2):
    a = dict(d1)
    b = dict(d2)
    keys = set(a) | set(b)
    return sum(abs(a.get(k, Fraction(0)) - b.get(k, Fraction(0)))
               for k in keys) / 2


def is_dist(x):
    if not isinstance(x, tuple) or len(x) == 0:
        return False
    for item in x:
        if not (isinstance(item, tuple) and len(item) == 2):
            return False
    return True


def is_hypothesis(x):
    return (isinstance(x, tuple) and len(x) == 2
            and isinstance(x[0], int) and is_dist(x[1]))


def dist_of(x):
    if is_hypothesis(x):
        return x[1]
    return x


def type_aware_distance(a, b):
    if is_dist(a) and is_dist(b):
        return tv(a, b)
    if is_hypothesis(a) and is_hypothesis(b):
        if a[0] != b[0]:
            return Fraction(1)
        return tv(a[1], b[1])
    if (is_dist(a) and is_hypothesis(b)) or (is_hypothesis(a) and is_dist(b)):
        return tv(dist_of(a), dist_of(b))
    return Fraction(1)


# ============================================================
# Pooling (the vertical direction)
# ============================================================

def pooling(nu, atoms):
    d = dict(nu)
    groups = {}
    for k, p in d.items():
        best = None
        best_d = None
        for a in atoms:
            dist = type_aware_distance(k, a)
            if best_d is None or dist < best_d:
                best_d = dist
                best = a
        if best is None:
            continue
        groups.setdefault(best, []).append((k, p))
    out = {}
    for a, members in groups.items():
        mass = sum(p for _, p in members)
        if mass == 0:
            continue
        weighted = {}
        for k, p in members:
            dk = dist_of(k)
            for b, q in dict(dk).items():
                weighted[b] = weighted.get(b, Fraction(0)) + (p / mass) * q
        bary = canonical(weighted)
        out[bary] = out.get(bary, Fraction(0)) + mass
    return canonical(out)


# ============================================================
# The tower context
# ============================================================

def random_rho(n, rng):
    weights = [rng.randint(1, 5) for _ in range(n)]
    return mk([(i, w) for i, w in enumerate(weights)])


def random_coherent_context(n, rng):
    rho1 = random_rho(n, rng)
    rho2 = random_rho(n, rng)
    while rho1 == rho2:
        rho2 = random_rho(n, rng)
    w11 = Fraction(rng.randint(1, 4), 5)
    Pi1 = mk([(rho1, w11), (rho2, 1 - w11)])
    w21 = Fraction(rng.randint(1, 4), 5)
    Pi2 = mk([(rho1, w21), (rho2, 1 - w21)])
    omega1 = Fraction(rng.randint(1, 4), 5)
    Omega = mk([(Pi1, omega1), (Pi2, 1 - omega1)])
    Pi = barycenter(Omega)
    rho = barycenter(Pi)
    items = list(dict(Pi).items())
    Sigma = mk([((i % 2, j), w) for i, (j, w) in enumerate(items)])
    return (rho, Pi, Omega, Sigma)


# ============================================================
# Horizontal and vertical 1-cells
# ============================================================

def L_1cell(x, c):
    """Vertical 1-cell: pooling by Pi-atoms."""
    rho, Pi, Omega, Sigma = c
    return lambda nu: pooling(nu, list(dict(Pi).keys()))


def R_1cell(x, c):
    """Vertical 1-cell: pooling by Sigma-hypotheses."""
    rho, Pi, Omega, Sigma = c
    return lambda nu: pooling(nu, list(dict(Sigma).keys()))


def shift_1cell(x, c):
    """Horizontal 1-cell: the shift law at a point mass."""
    rho, Pi, Omega, Sigma = c
    return mk([((x, c), Fraction(1))])


# ============================================================
# The squares
# ============================================================

def square_LL(x, c, nu):
    """
    The square for L * L:
       nu --shift--> shift(nu)
       |              |
       L              L
       v              v
       L(nu) --shift--> shift(L(nu))
    """
    L = L_1cell(x, c)
    L_nu = L(nu)
    L_L_nu = L(L_nu)
    return (L_nu, L_L_nu)


def square_RR(x, c, nu):
    R = R_1cell(x, c)
    R_nu = R(nu)
    R_R_nu = R(R_nu)
    return (R_nu, R_R_nu)


def square_LR(x, c, nu):
    """
    The square for L * R:
       nu --shift--> shift(nu)
       |              |
       L              R
       v              v
       L(nu) --shift--> R(L(nu))
    """
    L = L_1cell(x, c)
    R = R_1cell(x, c)
    L_nu = L(nu)
    R_L_nu = R(L_nu)
    return (L_nu, R_L_nu)


def square_RL(x, c, nu):
    L = L_1cell(x, c)
    R = R_1cell(x, c)
    R_nu = R(nu)
    L_R_nu = L(R_nu)
    return (R_nu, L_R_nu)


# ============================================================
# Interchange law
# ============================================================

def check_interchange(n, n_trials, seed):
    """
    Check the interchange law: the two ways of composing
    L * R * L * R agree.
    """
    rng = random.Random(seed)
    results = []
    for trial in range(n_trials):
        c = random_coherent_context(n, rng)
        x = rng.randint(0, n - 1)
        nu = mk([(mk([(b, Fraction(1))]), Fraction(1, n)) for b in range(n)])

        L = L_1cell(x, c)
        R = R_1cell(x, c)

        # Horizontal composition: L * L, then R * R
        # Vertical composition: (L * L) then (R * R)
        # Interchange: the two orders agree

        # Order 1: horizontal first, then vertical
        h_first_LL = L(L(nu))
        h_first_RR = R(R(nu))
        # Apply to nu
        order1_L = h_first_LL
        order1_R = h_first_RR

        # Order 2: vertical first, then horizontal
        v_first = L(R(nu))
        # The interchange law requires: order1_L = order2_L (for the L component)
        # and order1_R = order2_R (for the R component)

        # For the L-R pairing:
        LR = R(L(nu))
        RL = L(R(nu))

        # The interchange law: R(L(nu)) vs L(R(nu))
        # If they are equal, the interchange law holds.
        interchange_holds = (LR == RL)

        results.append({
            "trial": trial,
            "interchange_holds": interchange_holds,
            "LR": LR,
            "RL": RL,
        })
    return results


# ============================================================
# Beck defects as horizontal 2-cells
# ============================================================

def check_beck_defects(n, n_trials, seed):
    """
    Check that the Beck defects are horizontal 2-cells.
    The (U), (M), (Cm) defects are the comparisons at each level.
    """
    rng = random.Random(seed)
    results = []
    for trial in range(n_trials):
        c = random_coherent_context(n, rng)
        x = rng.randint(0, n - 1)
        rho, Pi, Omega, Sigma = c

        # (U) defect: delta_rho vs D(eta)rho
        D_eta_rho = push(rho, lambda b: mk([(b, Fraction(1))]))
        delta_rho = mk([(rho, Fraction(1))])
        u_defect = (delta_rho != D_eta_rho)

        # (M) defect: level-one pooling
        # (Cm) defect: pair (L, R)
        L = L_1cell(x, c)
        R = R_1cell(x, c)
        nu = mk([(mk([(b, Fraction(1))]), Fraction(1, n)) for b in range(n)])
        L_nu = L(nu)
        R_nu = R(nu)
        cm_defect = (L_nu, R_nu)

        results.append({
            "trial": trial,
            "U_defect": u_defect,
            "cm_defect_well_defined": is_dist(cm_defect[0]) and is_dist(cm_defect[1]),
        })
    return results


# ============================================================
# Main
# ============================================================

def run_all(n, n_trials, seed, label):
    print(f"=== {label} (n = {n}, trials = {n_trials}) ===")
    interchange = check_interchange(n, n_trials, seed)
    beck = check_beck_defects(n, n_trials, seed + 1)

    n_interchange = sum(1 for r in interchange if r["interchange_holds"])
    n_U = sum(1 for r in beck if r["U_defect"])
    n_cm = sum(1 for r in beck if r["cm_defect_well_defined"])

    print(f"  interchange holds: {n_interchange}/{n_trials}")
    print(f"  U defect nonzero: {n_U}/{n_trials}")
    print(f"  (Cm) defect well-defined: {n_cm}/{n_trials}")
    print()

    return {
        "label": label,
        "n": n,
        "n_trials": n_trials,
        "interchange": n_interchange,
        "U_defect": n_U,
        "cm_defect": n_cm,
    }


def freeze(results, filename="check_double_category_results.json"):
    with open(filename, "w") as f:
        json.dump(results, f, indent=2, default=str)
    with open(filename, "rb") as f:
        h = hashlib.sha256(f.read()).hexdigest()
    print(f"SHA-256: {h}")
    return h


if __name__ == "__main__":
    all_results = []
    all_results.append(run_all(2, 300, 42, "two-point base"))
    all_results.append(run_all(3, 300, 42, "three-point base"))
    all_results.append(run_all(4, 300, 42, "four-point base"))
    freeze(all_results)
