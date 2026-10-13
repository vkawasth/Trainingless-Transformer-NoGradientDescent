"""
check_noncoherent.py

Checks whether the interchange law holds strictly or only up to a 2-cell
on non-coherent contexts.

Run: python check_noncoherent.py
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
# Pooling
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
# Non-coherent context
# ============================================================

def random_rho(n, rng):
    weights = [rng.randint(1, 5) for _ in range(n)]
    return mk([(i, w) for i, w in enumerate(weights)])


def random_noncoherent_context(n, rng):
    """
    Non-coherent context: q(Sigma) != Pi.
    Sigma has hypotheses with the same distribution but different descriptors.
    """
    rho1 = random_rho(n, rng)
    rho2 = random_rho(n, rng)
    while rho1 == rho2:
        rho2 = random_rho(n, rng)

    # Pi has two atoms
    w = Fraction(rng.randint(1, 4), 5)
    Pi = mk([(rho1, w), (rho2, 1 - w)])

    # Sigma has hypotheses with the same distribution but different descriptors
    Sigma = mk([
        ((0, rho1), Fraction(1, 3)),
        ((1, rho1), Fraction(1, 3)),
        ((0, rho2), Fraction(1, 3)),
    ])

    rho = barycenter(Pi)
    # Omega = point mass at Pi (not used)
    Omega = mk([(Pi, Fraction(1))])

    return (rho, Pi, Omega, Sigma)


# ============================================================
# The two shadows
# ============================================================

def L_shadow(x, c, nu):
    rho, Pi, Omega, Sigma = c
    return pooling(nu, list(dict(Pi).keys()))


def R_shadow(x, c, nu):
    rho, Pi, Omega, Sigma = c
    return pooling(nu, list(dict(Sigma).keys()))


# ============================================================
# Inputs
# ============================================================

def make_dist_input(n):
    return mk([(mk([(b, Fraction(1))]), Fraction(1, n)) for b in range(n)])


# ============================================================
# Checks
# ============================================================

def check_different(n, n_trials, seed):
    rng = random.Random(seed)
    results = []
    for trial in range(n_trials):
        c = random_noncoherent_context(n, rng)
        x = rng.randint(0, n - 1)
        nu = make_dist_input(n)

        L = L_shadow(x, c, nu)
        R = R_shadow(x, c, nu)

        different = (L != R)

        results.append({
            "trial": trial,
            "different": different,
        })
    return results


def check_interchange(n, n_trials, seed):
    rng = random.Random(seed)
    results = []
    for trial in range(n_trials):
        c = random_noncoherent_context(n, rng)
        x = rng.randint(0, n - 1)
        nu = make_dist_input(n)

        LR = R_shadow(x, c, L_shadow(x, c, nu))
        RL = L_shadow(x, c, R_shadow(x, c, nu))

        interchange_strict = (LR == RL)
        interchange_fails = not interchange_strict

        results.append({
            "trial": trial,
            "interchange_strict": interchange_strict,
            "interchange_fails": interchange_fails,
        })
    return results


def check_double_composition(n, n_trials, seed):
    """
    Compose the (Cm) diagram with itself.
    The two orders of composition are:
    - Order 1: (L * R) * (L * R)
    - Order 2: (L * L) * (R * R)
    Check if they agree.
    """
    rng = random.Random(seed)
    results = []
    for trial in range(n_trials):
        c = random_noncoherent_context(n, rng)
        x = rng.randint(0, n - 1)
        nu = make_dist_input(n)

        L = lambda nu_: L_shadow(x, c, nu_)
        R = lambda nu_: R_shadow(x, c, nu_)

        # Order 1: (L * R) * (L * R) = L(R(L(R(nu))))
        order1 = L(R(L(R(nu))))

        # Order 2: (L * L) * (R * R) = L(L(R(R(nu))))
        order2 = L(L(R(R(nu))))

        agree = (order1 == order2)

        results.append({
            "trial": trial,
            "agree": agree,
        })
    return results


# ============================================================
# Main
# ============================================================

def run_all(n, n_trials, seed, label):
    print(f"=== {label} (n = {n}, trials = {n_trials}) ===")
    diff = check_different(n, n_trials, seed)
    inter = check_interchange(n, n_trials, seed + 1)
    comp = check_double_composition(n, n_trials, seed + 2)

    n_diff = sum(1 for r in diff if r["different"])
    n_inter_strict = sum(1 for r in inter if r["interchange_strict"])
    n_inter_fails = sum(1 for r in inter if r["interchange_fails"])
    n_comp = sum(1 for r in comp if r["agree"])

    print(f"  L different from R: {n_diff}/{n_trials}")
    print(f"  interchange strict: {n_inter_strict}/{n_trials}")
    print(f"  interchange fails: {n_inter_fails}/{n_trials}")
    print(f"  double composition agrees: {n_comp}/{n_trials}")
    print()

    return {
        "label": label,
        "n": n,
        "n_trials": n_trials,
        "different": n_diff,
        "interchange_strict": n_inter_strict,
        "interchange_fails": n_inter_fails,
        "double_composition": n_comp,
    }


def freeze(results, filename="check_noncoherent_results.json"):
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
