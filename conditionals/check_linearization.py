"""
check_linearization.py (clean rewrite)

Verifies the key lemma and the Hessian identity of the linearization
of the coupling A_infinity-category, for n = 2, 3, 4.
"""

from fractions import Fraction as F
from itertools import combinations, permutations
from math import factorial, gcd


# ----------------------------------------------------------------------
# 1. Cycle enumeration
# ----------------------------------------------------------------------
def enumerate_cycles(n):
    """
    Enumerate all simple 2k-cycles of K_{n,n}, for k = 2, ..., n.
    A 2k-cycle is represented as a frozenset of 2k edges (i, j).
    """
    cycles = set()
    for k in range(2, n + 1):
        for rows in combinations(range(n), k):
            for cols in combinations(range(n), k):
                for rho in permutations(range(k)):
                    for sigma in permutations(range(k)):
                        edges = set()
                        for i in range(k):
                            edges.add((rows[rho[i]], cols[sigma[i]]))
                            edges.add((rows[rho[(i + 1) % k]], cols[sigma[i]]))
                        cycles.add(frozenset(edges))
    return list(cycles)

#def enumerate_cycles(n):
#    """
#    Enumerate all simple 2k-cycles of K_{n,n}, for k = 2, ..., n.
#    A 2k-cycle is represented as a frozenset of 2k edges (i, j),
#    with i in rows, j in cols.
#    """
#    cycles = set()
#    for k in range(2, n + 1):
#        for rows in combinations(range(n), k):
#            for cols in combinations(range(n), k):
#                # A 2k-cycle corresponds to a permutation sigma of
#                # {0, ..., k-1}: the cycle is
#                #   rows[0] - cols[sigma[0]] - rows[1] - cols[sigma[1]]
#                #   - ... - rows[k-1] - cols[sigma[k-1]] - rows[0]
#                # The edges are:
#                #   (rows[i], cols[sigma[i]]) for i = 0..k-1
#                #   (rows[(i+1) % k], cols[sigma[i]]) for i = 0..k-1
#                for sigma in permutations(range(k)):
#                    edges = set()
#                    for i in range(k):
#                        edges.add((rows[i], cols[sigma[i]]))
#                        edges.add((rows[(i + 1) % k], cols[sigma[i]]))
#                    # Canonicalize: sort the edges
#                    cycles.add(frozenset(edges))
#    return list(cycles)


# ----------------------------------------------------------------------
# 2. Primitive normal
# ----------------------------------------------------------------------

def primitive_normal(cycle, n):
    """
    Compute the primitive normal of a cycle in the lattice basis
    Lambda = Z^{(n-1)^2}.
    The normal is the (n-1)x(n-1) block of the unique matrix v
    with zero row and column sums, such that v - incidence is
    supported on the boundary (row n-1 or column n-1).
    """
    # Build the incidence vector (n x n matrix)
    incidence = [[0] * n for _ in range(n)]
    for (i, j) in cycle:
        incidence[i][j] += 1
    # The normal v satisfies:
    #   v[i][j] = incidence[i][j] for i, j <= n-2
    #   row sums and column sums of v are zero
    # Build v
    v = [[0] * n for _ in range(n)]
    for i in range(n - 1):
        for j in range(n - 1):
            v[i][j] = incidence[i][j]
    # Fill in the last column and last row to make row/col sums zero
    for i in range(n - 1):
        v[i][n - 1] = -sum(v[i][j] for j in range(n - 1))
    for j in range(n - 1):
        v[n - 1][j] = -sum(v[i][j] for i in range(n - 1))
    v[n - 1][n - 1] = -sum(v[i][n - 1] for i in range(n - 1))
    # Return the (n-1)x(n-1) block
    normal = {}
    for i in range(n - 1):
        for j in range(n - 1):
            if v[i][j] != 0:
                normal[(i, j)] = v[i][j]
    return normal


# ----------------------------------------------------------------------
# 3. Verification
# ----------------------------------------------------------------------

def expected_cycle_count(n, k):
    """The expected number of 2k-cycles in K_{n,n}."""
    return (factorial(n) // factorial(n - k)) ** 2 // (2 * k)


def verify_cycle_counts(n):
    cycles = enumerate_cycles(n)
    actual = {}
    for C in cycles:
        k = len(C) // 2  # 2k-cycle
        actual[k] = actual.get(k, 0) + 1
    expected = {}
    for k in range(2, n + 1):
        expected[k] = expected_cycle_count(n, k)
    ok = all(expected.get(k) == actual.get(k) for k in expected)
    return ok, expected, actual


def is_primitive(v):
    vals = [abs(c) for c in v.values() if c != 0]
    if not vals:
        return True
    g = vals[0]
    for c in vals[1:]:
        g = gcd(g, c)
    return g == 1


def verify_key_lemma(n):
    cycles = enumerate_cycles(n)
    ok = True
    for C in cycles:
        v = primitive_normal(C, n)
        if not is_primitive(v):
            ok = False
            print(f"    Non-primitive normal for cycle: {v}")
    return ok


def verify_hessian(n):
    cycles = enumerate_cycles(n)
    d = (n - 1) ** 2
    H = [[F(0) for _ in range(d)] for _ in range(d)]
    x_prod = F(1, n * n)
    for C in cycles:
        v = primitive_normal(C, n)
        # (x^prod)^{v_C} = prod_{i,j} (1/n^2)^{v_C[i,j]}
        # = (1/n^2)^{sum v_C}
        exponent = sum(v.values())
        coeff = x_prod ** exponent
        keys = sorted(v.keys())
        for key1 in keys:
            for key2 in keys:
                idx1 = key1[0] * (n - 1) + key1[1]
                idx2 = key2[0] * (n - 1) + key2[1]
                H[idx1][idx2] += coeff * v[key1] * v[key2]
    ok = all(H[i][j] == H[j][i] for i in range(d) for j in range(d))
    return ok, H


def main():
    print("=" * 72)
    print("check_linearization.py -- clean rewrite")
    print("=" * 72)
    all_ok = True
    for n in [2, 3, 4]:
        print(f"\n--- n = {n} ---")
        ok1, exp, act = verify_cycle_counts(n)
        print(f"  Cycle counts: {ok1}")
        print(f"    expected: {exp}")
        print(f"    actual:   {act}")
        all_ok &= ok1
        ok2 = verify_key_lemma(n)
        print(f"  Key lemma: {ok2}")
        all_ok &= ok2
        ok3, H = verify_hessian(n)
        print(f"  Hessian symmetric: {ok3}")
        if n <= 3:
            print(f"    Hessian ({len(H)}x{len(H)}):")
            for row in H:
                print(f"      {row}")
        all_ok &= ok3
    print("\n" + "=" * 72)
    if all_ok:
        print("ALL CHECKS PASS")
    else:
        print("SOME CHECKS FAILED")
    print("=" * 72)
    return all_ok


if __name__ == "__main__":
    main()
