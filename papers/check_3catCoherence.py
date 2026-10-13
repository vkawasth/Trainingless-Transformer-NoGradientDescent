"""
check_3category.py

Verifies the 3-category structure of the tower:
- Option A: one shadow level Sh(Y), with the D-levels fixed.
- Option B: two shadow levels Sh(Y), Sh(DY), with the D-levels fixed.
- The 3-cell coherence between the two shadow levels.

Run: python check_3category.py
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
# Pooling (conditional expectation)
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
# Tower contexts
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
# Option A: one shadow level
# ============================================================

def option_A_shadows(n, c, nu):
    """
    Option A: one shadow level Sh(Y).
    L_A pools by Pi-atoms.
    R_A pools by Sigma-hypotheses.
    """
    rho, Pi, Omega, Sigma = c
    L = pooling(nu, list(dict(Pi).keys()))
    R = pooling(nu, list(dict(Sigma).keys()))
    return L, R


def check_option_A(n, n_trials, seed):
    """
    Option A: verify the double-category structure with one shadow level.
    """
    rng = random.Random(seed)
    results = []
    for trial in range(n_trials):
        c = random_coherent_context(n, rng)
        x = rng.randint(0, n - 1)
        nu = mk([(mk([(b, Fraction(1))]), Fraction(1, n)) for b in range(n)])

        L, R = option_A_shadows(n, c, nu)
        LR = pooling(L, list(dict(c[3]).keys()))
        RL = pooling(R, list(dict(c[1]).keys()))

        interchange_strict = (LR == RL)
        L_idem = (pooling(L, list(dict(c[1]).keys())) == L)
        R_idem = (pooling(R, list(dict(c[3]).keys())) == R)

        results.append({
            "trial": trial,
            "interchange": interchange_strict,
            "L_idempotent": L_idem,
            "R_idempotent": R_idem,
        })
    return results


# ============================================================
# Option B: two shadow levels
# ============================================================

def option_B_shadows(n, c, nu):
    """
    Option B: two shadow levels Sh(Y) and Sh(DY).
    L_4, R_4 at level 4 (Sh(Y)).
    L_5, R_5 at level 5 (Sh(DY)).
    """
    rho, Pi, Omega, Sigma = c

    # Level 4 shadows
    L4 = pooling(nu, list(dict(Pi).keys()))
    R4 = pooling(nu, list(dict(Sigma).keys()))

    # Level 5 shadows: push the level-4 shadows through the transport
    # For the check, we use the same atoms but on the pushforward.
    Pi_push = push(Pi, lambda r: mk([(r, Fraction(1))]))
    Sigma_push = push(Sigma, lambda h: (h[0], mk([(h[1], Fraction(1))])))

    # The level-5 atoms are the pushforwards
    L5_atoms = list(dict(Pi_push).keys())
    R5_atoms = list(dict(Sigma_push).keys())

    L5 = pooling(nu, L5_atoms)
    R5 = pooling(nu, R5_atoms)

    return (L4, R4, L5, R5)


def check_option_B(n, n_trials, seed):
    """
    Option B: verify the 3-category structure with two shadow levels.
    """
    rng = random.Random(seed)
    results = []
    for trial in range(n_trials):
        c = random_coherent_context(n, rng)
        x = rng.randint(0, n - 1)
        nu = mk([(mk([(b, Fraction(1))]), Fraction(1, n)) for b in range(n)])

        L4, R4, L5, R5 = option_B_shadows(n, c, nu)

        # Interchange at level 4
        inter_4 = (pooling(L4, list(dict(c[3]).keys())) ==
                   pooling(R4, list(dict(c[1]).keys())))

        # Interchange at level 5
        inter_5 = (L5 == R5)

        # Idempotence
        L4_idem = (pooling(L4, list(dict(c[1]).keys())) == L4)
        R4_idem = (pooling(R4, list(dict(c[3]).keys())) == R4)
        L5_idem = (pooling(L5, list(dict(c[1]).keys())) == L5)
        R5_idem = (pooling(R5, list(dict(c[3]).keys())) == R5)

        # 3-cell coherence: the two levels agree on the diagonal
        # (the composition of the two levels in either order)
        level_4_then_5 = pooling(L4, list(dict(c[3]).keys()))
        level_5_then_4 = pooling(R5, list(dict(c[1]).keys()))
        coherence = (level_4_then_5 == level_5_then_4)

        results.append({
            "trial": trial,
            "interchange_4": inter_4,
            "interchange_5": inter_5,
            "L4_idempotent": L4_idem,
            "R4_idempotent": R4_idem,
            "L5_idempotent": L5_idem,
            "R5_idempotent": R5_idem,
            "coherence": coherence,
        })
    return results


# ============================================================
# The 3-cell check
# ============================================================

def check_3cell(n, n_trials, seed):
    """
    Verify the 3-cell coherence between the two shadow levels.
    The 3-cell is the comparison between:
    - Composing at level 4 first, then level 5.
    - Composing at level 5 first, then level 4.
    """
    rng = random.Random(seed)
    results = []
    for trial in range(n_trials):
        c = random_coherent_context(n, rng)
        x = rng.randint(0, n - 1)
        nu = mk([(mk([(b, Fraction(1))]), Fraction(1, n)) for b in range(n)])

        rho, Pi, Omega, Sigma = c

        # Level 4 atoms and level 5 atoms
        atoms_4 = list(dict(Pi).keys())
        atoms_5 = list(dict(push(Pi, lambda r: mk([(r, Fraction(1))]))).keys())

        # Composition order 1: level 4, then level 5
        comp_45 = pooling(pooling(nu, atoms_4), atoms_5)

        # Composition order 2: level 5, then level 4
        comp_54 = pooling(pooling(nu, atoms_5), atoms_4)

        # 3-cell: the comparison between the two orders
        three_cell = (comp_45 == comp_54)

        results.append({
            "trial": trial,
            "three_cell_trivial": three_cell,
        })
    return results


# ============================================================
# Main
# ============================================================

def run_all(n, n_trials, seed, label):
    print(f"=== {label} (n = {n}, trials = {n_trials}) ===")
    A = check_option_A(n, n_trials, seed)
    B = check_option_B(n, n_trials, seed + 1)
    C = check_3cell(n, n_trials, seed + 2)

    n_A_inter = sum(1 for r in A if r["interchange"])
    n_A_L_idem = sum(1 for r in A if r["L_idempotent"])
    n_A_R_idem = sum(1 for r in A if r["R_idempotent"])

    n_B_inter_4 = sum(1 for r in B if r["interchange_4"])
    n_B_inter_5 = sum(1 for r in B if r["interchange_5"])
    n_B_L4_idem = sum(1 for r in B if r["L4_idempotent"])
    n_B_R4_idem = sum(1 for r in B if r["R4_idempotent"])
    n_B_L5_idem = sum(1 for r in B if r["L5_idempotent"])
    n_B_R5_idem = sum(1 for r in B if r["R5_idempotent"])
    n_B_coh = sum(1 for r in B if r["coherence"])

    n_C_3cell = sum(1 for r in C if r["three_cell_trivial"])

    print("Option A (one shadow):")
    print(f"  interchange: {n_A_inter}/{n_trials}")
    print(f"  L idempotent: {n_A_L_idem}/{n_trials}")
    print(f"  R idempotent: {n_A_R_idem}/{n_trials}")
    print()
    print("Option B (two shadows):")
    print(f"  interchange level 4: {n_B_inter_4}/{n_trials}")
    print(f"  interchange level 5: {n_B_inter_5}/{n_trials}")
    print(f"  L4 idempotent: {n_B_L4_idem}/{n_trials}")
    print(f"  R4 idempotent: {n_B_R4_idem}/{n_trials}")
    print(f"  L5 idempotent: {n_B_L5_idem}/{n_trials}")
    print(f"  R5 idempotent: {n_B_R5_idem}/{n_trials}")
    print(f"  coherence: {n_B_coh}/{n_trials}")
    print()
    print("3-cell (coherence between levels):")
    print(f"  3-cell trivial: {n_C_3cell}/{n_trials}")
    print()

    return {
        "label": label,
        "n": n,
        "n_trials": n_trials,
        "A_interchange": n_A_inter,
        "A_L_idempotent": n_A_L_idem,
        "A_R_idempotent": n_A_R_idem,
        "B_interchange_4": n_B_inter_4,
        "B_interchange_5": n_B_inter_5,
        "B_L4_idempotent": n_B_L4_idem,
        "B_R4_idempotent": n_B_R4_idem,
        "B_L5_idempotent": n_B_L5_idem,
        "B_R5_idempotent": n_B_R5_idem,
        "B_coherence": n_B_coh,
        "C_3cell_trivial": n_C_3cell,
    }


def freeze(results, filename="check_3category_results.json"):
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
