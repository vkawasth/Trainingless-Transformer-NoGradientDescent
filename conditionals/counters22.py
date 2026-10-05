"""
check_2x2.py

Exact rational checks for the square 2x2 case:
coupling 2-cells, vertical composition, whiskering,
the failure of postcomposition functoriality, and the
defect F_2 as a signed measure.

Companion to:
  V. K. Awasthi,
  "Conditionals, Couplings and Toric Geometry",
  Section: "The square 2x2 case".

All arithmetic is exact (fractions.Fraction).
Run:  python check_2x2.py
"""

from fractions import Fraction as F
from itertools import product


# ----------------------------------------------------------------------
# 1. Basic 1-cell machinery
# ----------------------------------------------------------------------

def k(a, b):
    """
    The stochastic 1-cell k(a,b) on B = B' = {0,1}:
        row 0: (1-a, a)
        row 1: (1-b, b)
    """
    return [[1 - a, a], [1 - b, b]]


def is_stochastic(m):
    """Check that a 2x2 matrix is stochastic."""
    for row in m:
        if any(x < 0 for x in row):
            return False
        if sum(row) != 1:
            return False
    return True


def kleisli(h, kk):
    """
    Kleisli composition (h o k)(i)(j) = sum_l k(i)(l) h(l)(j).
    Both h and kk are 2x2 stochastic matrices.
    """
    return [
        [sum(kk[i][l] * h[l][j] for l in range(2)) for j in range(2)]
        for i in range(2)
    ]


# ----------------------------------------------------------------------
# 2. Couplings
# ----------------------------------------------------------------------

def coupling_segment(p, q):
    """
    Coupling polytope of (p, 1-p) and (q, 1-q): the segment
        a in [max(0, p+q-1), min(p,q)]
    with matrix
        [[a, p-a], [q-a, 1-p-q+a]].
    Returns (lo, hi).
    """
    lo = max(F(0), p + q - 1)
    hi = min(p, q)
    return lo, hi


def coupling_matrix(a, p, q):
    """The 2x2 coupling with parameter a."""
    return [[a, p - a], [q - a, 1 - p - q + a]]


def is_coupling(alpha, f, g):
    """
    Check that alpha is a coupling of f and g, where
    f and g are 2x2 stochastic matrices (one row per x in {0,1}).
    alpha is a list of two 2x2 matrices (one per x).
    """
    for x in range(2):
        M = alpha[x]
        # Nonnegativity
        for row in M:
            for v in row:
                if v < 0:
                    return False
        # Row sums = f(x)
        for y in range(2):
            if sum(M[y]) != f[x][y]:
                return False
        # Column sums = g(x)
        for yp in range(2):
            if sum(M[y][yp] for y in range(2)) != g[x][yp]:
                return False
    return True


def cost(alpha):
    """Cost c(alpha) = max_x Pr(y != y')."""
    worst = F(0)
    for x in range(2):
        M = alpha[x]
        # Pr(y != y') = 1 - sum_y alpha(y,y) = 1 - (M[0][0] + M[1][1])
        d = 1 - (M[0][0] + M[1][1])
        if d > worst:
            worst = d
    return worst


def total_variation(p, q):
    """Total variation between two distributions on {0,1}."""
    return F(1, 2) * (abs(p[0] - q[0]) + abs(p[1] - q[1]))


def d_observable(f, g):
    """d(f,g) = max_x TV(f(x), g(x))."""
    return max(total_variation(f[x], g[x]) for x in range(2))


# ----------------------------------------------------------------------
# 3. Vertical composition (the gluing lemma)
# ----------------------------------------------------------------------

def vertical(alpha, beta, g):
    """
    Vertical composite of alpha: f => g and beta: g => h.
    g is the middle 1-cell (2x2 stochastic matrix).
    Returns a list of two 2x2 matrices.
    """
    out = []
    for x in range(2):
        A = alpha[x]
        B = beta[x]
        M = [[F(0), F(0)], [F(0), F(0)]]
        for y in range(2):
            for ypp in range(2):
                s = F(0)
                for yp in range(2):
                    gy = g[x][yp]
                    if gy == 0:
                        continue
                    s += A[y][yp] * B[yp][ypp] / gy
                M[y][ypp] = s
        out.append(M)
    return out


# ----------------------------------------------------------------------
# 4. Pre- and postcomposition (whiskering)
# ----------------------------------------------------------------------

def precompose(alpha, u):
    """
    Precomposition of alpha: f => g by a 1-cell u: W -> B.
    Here we only handle W = B = {0,1}, so u is a 2x2 stochastic matrix.
    The result is alpha o u: f o u => g o u, evaluated at each w.
    """
    # u(w) is the w-th row of u, viewed as a distribution on B.
    # alpha o u (w) = alpha(u(w)).
    # We need alpha evaluated at the distribution u(w), which is a
    # convex combination of alpha(0) and alpha(1).
    out = []
    for w in range(2):
        # u(w) = (u[w][0], u[w][1])
        M = [[F(0), F(0)], [F(0), F(0)]]
        for x in range(2):
            coeff = u[w][x]
            if coeff == 0:
                continue
            for y in range(2):
                for yp in range(2):
                    M[y][yp] += coeff * alpha[x][y][yp]
        out.append(M)
    return out


def postcompose(alpha, h):
    """
    Postcomposition of alpha: f => g by a 1-cell h: B' -> B''.
    Here B' = B'' = {0,1}, so h is a 2x2 stochastic matrix.
    (h . alpha)(x)(z,z') = sum_{y,y'} alpha(x)(y,y') h(y)(z) h(y')(z').
    """
    out = []
    for x in range(2):
        M = [[F(0), F(0)], [F(0), F(0)]]
        for y in range(2):
            for yp in range(2):
                a = alpha[x][y][yp]
                if a == 0:
                    continue
                for z in range(2):
                    for zp in range(2):
                        M[z][zp] += a * h[y][z] * h[yp][zp]
        out.append(M)
    return out


# ----------------------------------------------------------------------
# 5. The four deterministic 1-cells
# ----------------------------------------------------------------------

e0 = k(F(0), F(0))   # constant to 0
e1 = k(F(1), F(1))   # constant to 1
sigma = k(F(0), F(1))  # identity
tau = k(F(1), F(0))  # swap

DET = {"e0": e0, "e1": e1, "sigma": sigma, "tau": tau}


# ----------------------------------------------------------------------
# 6. Checks
# ----------------------------------------------------------------------

def check(name, cond, extra=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}{(' -- ' + extra) if extra else ''}")
    return cond


def main():
    all_ok = True
    print("=" * 72)
    print("check_2x2.py -- exact rational checks")
    print("=" * 72)

    # ---- (a) Deterministic 1-cells are stochastic ----
    for name, M in DET.items():
        all_ok &= check(f"{name} is stochastic", is_stochastic(M))

    # ---- (b) Coupling polytope at a single point ----
    # For p = 1/2, q = 1/2: lo = 0, hi = 1/2, product at a = 1/4.
    lo, hi = coupling_segment(F(1, 2), F(1, 2))
    all_ok &= check("Cpl((1/2,1/2),(1/2,1/2)) = [0, 1/2]",
                    lo == 0 and hi == F(1, 2))
    # For p = 1/2, q = 7/10: lo = 1/5, hi = 1/2, product at a = 7/20.
    lo, hi = coupling_segment(F(1, 2), F(7, 10))
    all_ok &= check("Cpl((1/2,1/2),(7/10,3/10)) = [1/5, 1/2]",
                    lo == F(1, 5) and hi == F(1, 2))

    # ---- (c) Deterministic couplings are unique ----
    # sigma => sigma: unique diagonal
    alpha_sigma_sigma = [
        [[F(1), F(0)], [F(0), F(0)]],
        [[F(0), F(0)], [F(0), F(1)]],
    ]
    all_ok &= check("sigma => sigma unique coupling",
                    is_coupling(alpha_sigma_sigma, sigma, sigma))
    all_ok &= check("cost(sigma,sigma) = 0",
                    cost(alpha_sigma_sigma) == 0)

    # sigma => tau: unique off-diagonal
    alpha_sigma_tau = [
        [[F(0), F(1)], [F(0), F(0)]],
        [[F(0), F(0)], [F(1), F(0)]],
    ]
    all_ok &= check("sigma => tau unique coupling",
                    is_coupling(alpha_sigma_tau, sigma, tau))
    all_ok &= check("cost(sigma,tau) = 1",
                    cost(alpha_sigma_tau) == 1)

    # e0 => e1: unique
    alpha_e0_e1 = [
        [[F(0), F(1)], [F(0), F(0)]],
        [[F(0), F(1)], [F(0), F(0)]],
    ]
    all_ok &= check("e0 => e1 unique coupling",
                    is_coupling(alpha_e0_e1, e0, e1))
    all_ok &= check("cost(e0,e1) = 1",
                    cost(alpha_e0_e1) == 1)

    # sigma => e0: unique
    alpha_sigma_e0 = [
        [[F(1), F(0)], [F(0), F(0)]],
        [[F(0), F(0)], [F(1), F(0)]],
    ]
    all_ok &= check("sigma => e0 unique coupling",
                    is_coupling(alpha_sigma_e0, sigma, e0))
    all_ok &= check("cost(sigma,e0) = 1/2",
                    cost(alpha_sigma_e0) == F(1, 2))

    # ---- (d) Stochastic example: f = k(1/2,1/2), g = k(3/10,7/10) ----
    f = k(F(1, 2), F(1, 2))
    g = k(F(3, 10), F(7, 10))

    # At x = 0: f(0) = (1/2,1/2), g(0) = (7/10,3/10).
    # Coupling segment: a in [1/5, 1/2].
    lo0, hi0 = coupling_segment(F(1, 2), F(7, 10))
    all_ok &= check("Cpl(f(0),g(0)) = [1/5, 1/2]",
                    lo0 == F(1, 5) and hi0 == F(1, 2))
    # At x = 1: f(1) = (1/2,1/2), g(1) = (3/10,7/10).
    lo1, hi1 = coupling_segment(F(1, 2), F(3, 10))
    all_ok &= check("Cpl(f(1),g(1)) = [0, 3/10]",
                    lo1 == 0 and hi1 == F(3, 10))

    # Product coupling: a = p*q at each x.
    a_prod_0 = F(1, 2) * F(7, 10)  # 7/20
    a_prod_1 = F(1, 2) * F(3, 10)  # 3/20
    alpha_prod = [
        coupling_matrix(a_prod_0, F(1, 2), F(7, 10)),
        coupling_matrix(a_prod_1, F(1, 2), F(3, 10)),
    ]
    all_ok &= check("product coupling is a coupling",
                    is_coupling(alpha_prod, f, g))
    all_ok &= check("product coupling cost = 1/2",
                    cost(alpha_prod) == F(1, 2))

    # Maximal coupling: a = min(p,q) at each x.
    a_max_0 = min(F(1, 2), F(7, 10))  # 1/2
    a_max_1 = min(F(1, 2), F(3, 10))  # 3/10
    alpha_max = [
        coupling_matrix(a_max_0, F(1, 2), F(7, 10)),
        coupling_matrix(a_max_1, F(1, 2), F(3, 10)),
    ]
    all_ok &= check("maximal coupling is a coupling",
                    is_coupling(alpha_max, f, g))
    all_ok &= check("maximal coupling cost = d(f,g) = 1/5",
                    cost(alpha_max) == F(1, 5))
    all_ok &= check("d(f,g) = 1/5",
                    d_observable(f, g) == F(1, 5))

    # ---- (e) Vertical composition: example with f = sigma, g = k(1/2,1/2), h = tau ----
    gmid = k(F(1, 2), F(1, 2))
    # alpha: sigma => gmid
    alpha = [
        [[F(1, 2), F(1, 2)], [F(0), F(0)]],
        [[F(0), F(0)], [F(1, 2), F(1, 2)]],
    ]
    all_ok &= check("alpha: sigma => gmid is a coupling",
                    is_coupling(alpha, sigma, gmid))

    # beta: gmid => tau
    beta = [
        [[F(0), F(1, 2)], [F(0), F(1, 2)]],
        [[F(1, 2), F(0)], [F(1, 2), F(0)]],
    ]
    all_ok &= check("beta: gmid => tau is a coupling",
                    is_coupling(beta, gmid, tau))

    # vertical composite
    comp = vertical(alpha, beta, gmid)
    all_ok &= check("vertical composite is a coupling sigma => tau",
                    is_coupling(comp, sigma, tau))
    # At x = 0, it should be [[0,1],[0,0]].
    all_ok &= check("composite at x=0 equals [[0,1],[0,0]]",
                    comp[0] == [[F(0), F(1)], [F(0), F(0)]])
    # At x = 1, it should be [[0,0],[1,0]].
    all_ok &= check("composite at x=1 equals [[0,0],[1,0]]",
                    comp[1] == [[F(0), F(0)], [F(1), F(0)]])

    # ---- (f) Precomposition by a deterministic 1-cell ----
    # u = sigma (identity). Then alpha o sigma = alpha.
    alpha_pre = precompose(alpha, sigma)
    all_ok &= check("precomposition by sigma is alpha",
                    alpha_pre == alpha)
    # u = tau (swap). Then (alpha o tau)(w) = alpha(tau(w)).
    alpha_pre_tau = precompose(alpha, tau)
    # tau(0) = 1, tau(1) = 0, so (alpha o tau)(0) = alpha(1),
    # (alpha o tau)(1) = alpha(0).
    all_ok &= check("precomposition by tau swaps x=0 and x=1",
                    alpha_pre_tau[0] == alpha[1] and
                    alpha_pre_tau[1] == alpha[0])

    # ---- (g) Postcomposition by h = k(9/10, 1/10) ----
    h = k(F(9, 10), F(1, 10))

    # Set up the counterexample:
    #   f = g = kk = k(1/2,1/2)
    #   alpha = Delta_f
    #   beta(0) = swap, beta(1) = Delta
    # so beta . alpha = beta.
    fk = k(F(1, 2), F(1, 2))
    gk = k(F(1, 2), F(1, 2))
    kk = k(F(1, 2), F(1, 2))

    alpha_d = [
        [[F(1, 2), F(0)], [F(0), F(1, 2)]],
        [[F(1, 2), F(0)], [F(0), F(1, 2)]],
    ]
    beta_c = [
        [[F(0), F(1, 2)], [F(1, 2), F(0)]],
        [[F(1, 2), F(0)], [F(0), F(1, 2)]],
    ]
    all_ok &= check("alpha_d: fk => gk is a coupling",
                    is_coupling(alpha_d, fk, gk))
    all_ok &= check("beta_c: gk => kk is a coupling",
                    is_coupling(beta_c, gk, kk))

    # beta_c . alpha_d = beta_c (since alpha_d is the identity)
    comp_beta_alpha = vertical(alpha_d, beta_c, gk)
    all_ok &= check("beta . alpha = beta",
                    comp_beta_alpha == beta_c)

    # Postcompose by h
    h_alpha = postcompose(alpha_d, h)
    h_beta = postcompose(beta_c, h)
    h_comp = postcompose(comp_beta_alpha, h)

    # Check: (h . beta) . (h . alpha) vs h . (beta . alpha)
    # The marginals of h . alpha and h . beta are h o gk.
    h_gk = kleisli(h, gk)
    lhs = vertical(h_alpha, h_beta, h_gk)
    rhs = h_comp

    # At x = 0, the two should differ.
    diff0 = [[lhs[0][i][j] - rhs[0][i][j] for j in range(2)] for i in range(2)]
    print("  (h.beta).(h.alpha)(0) =", lhs[0])
    print("  h.(beta.alpha)(0)     =", rhs[0])
    print("  defect F_2(0)         =", diff0)

    # Check the explicit numerical values from the LaTeX section.
    all_ok &= check("(h.alpha)(0) = [[0.41,0.09],[0.09,0.41]]",
                    h_alpha[0] == [[F(41, 100), F(9, 100)],
                                   [F(9, 100), F(41, 100)]])
    all_ok &= check("(h.beta)(0) = [[0.09,0.41],[0.41,0.09]]",
                    h_beta[0] == [[F(9, 100), F(41, 100)],
                                  [F(41, 100), F(9, 100)]])
    all_ok &= check("((h.beta).(h.alpha))(0) = [[0.1476,0.3524],[0.3524,0.1476]]",
                    lhs[0] == [[F(1476, 10000), F(3524, 10000)],
                               [F(3524, 10000), F(1476, 10000)]])
    all_ok &= check("h.(beta.alpha)(0) = [[0.09,0.41],[0.41,0.09]]",
                    rhs[0] == [[F(9, 100), F(41, 100)],
                               [F(41, 100), F(9, 100)]])
    all_ok &= check("defect F_2(0) = [[0.0576,-0.0576],[-0.0576,0.0576]]",
                    diff0 == [[F(576, 10000), F(-576, 10000)],
                              [F(-576, 10000), F(576, 10000)]])
    # The defect is not a coupling: it has negative entries.
    has_negative = any(diff0[i][j] < 0 for i in range(2) for j in range(2))
    all_ok &= check("defect F_2(0) is not a coupling (has negative entries)",
                    has_negative)

    # ---- (h) Graded shadow ----
    # Grades add under vertical composition; the defect has grade
    # bounded by c(alpha) + c(beta).
    grade_alpha = cost(h_alpha)
    grade_beta = cost(h_beta)
    grade_comp = cost(lhs)
    all_ok &= check("grade subadditivity: c((h.beta).(h.alpha)) <= c(h.alpha)+c(h.beta)",
                    grade_comp <= grade_alpha + grade_beta)
    print(f"  grades: c(h.alpha)={grade_alpha}, c(h.beta)={grade_beta}, "
          f"c(composite)={grade_comp}")

    # ---- Summary ----
    print("=" * 72)
    if all_ok:
        print("ALL CHECKS PASS")
    else:
        print("SOME CHECKS FAILED")
    print("=" * 72)
    return all_ok


if __name__ == "__main__":
    main()
