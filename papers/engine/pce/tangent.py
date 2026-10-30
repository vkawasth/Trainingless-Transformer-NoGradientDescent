"""A tangent structure for injected data (companion paper "A tangent structure for injected data"; lab paper, Section "Markov
categories", paragraph "Tangent categories"). New module; nothing frozen.

Points of T^k M are written in the Weil algebra W_k = R[e_1, ..., e_k]/(e_i^2) (Leung's presentation of tangent structures): a point is
a coefficient array c[S], S a subset of {1..k} (bitmask), meaning sum_S c[S] e^S. e_1 is the innermost T, e_k the outermost:
T^k M = T(T^{k-1} M) adds e_k.

  HD                            hyper-dual numbers (W_k-valued arrays): +, -, *, /, sqrt, sums; T^k(f) of a polynomial or smooth
                                map f is f evaluated on HD
  subst(c, images, k_new)       the algebra map sending e_i to the monomial images[i] (a bitmask; 0 kills e_i): every structure map
                                of the tangent structure is one of these
  proj(k, j), zero(k, j), lift(k, j), flip(k, j), plus(k, j)
                                T^a p T^b, T^a 0 T^b, T^a l T^b, T^a c T^b and T^a(+)T^b with b = j (the generators they act on)
  in_T(c, k)                    membership of a point in T^k Delta for the closed simplex (feasible-direction cones), by the
                                lexicographic lemma: for each cell and each set U of live generators, the coefficients on subsets of
                                U are lexicographically >= 0, monomials ordered by size (e_1 << e_2 << ... << e_k << 1)
  in_T_def(c, k)                the same, from the definition (T Y = pairs (y, u) with y + s u in Y for all small s >= 0), computed
                                exactly with rational small parameters; used to verify the lemma
  vertical_fibre(x, U, W)       the fibre of T(p) over the zero section inside T^2 Delta, against the image of the vertical lift
  tetrad_jets(P)                tetrads, their Jacobian J and their full second derivative H at P (exact, via HD on the polynomial)
  lift_pair(J, H, u, v)         the obstruction [H(u, v)] in coker J to completing (u, v) in T_2 V to a point of T^2 V
"""
from __future__ import annotations
import itertools
from fractions import Fraction
import numpy as np
from scipy.linalg import null_space


def popcount(S): return bin(S).count("1")


# ------------------------------------------------------------------------------------------------ Weil algebra arithmetic
class HD:
    """an array with coefficients in W_k: c has shape (2^k, *shape)"""
    __array_priority__ = 1000

    def __init__(self, c, k):
        self.c = np.asarray(c, dtype=float); self.k = k

    @staticmethod
    def var(x, dirs):
        """x + sum_i dirs[i] e_{i+1}  (dirs may include None for 0); higher coefficients zero"""
        x = np.asarray(x, float); k = len(dirs); c = np.zeros((2 ** k,) + x.shape); c[0] = x
        for i, d in enumerate(dirs):
            if d is not None: c[1 << i] = d
        return HD(c, k)

    @staticmethod
    def point(coeffs, k):
        return HD(np.asarray(coeffs, float), k)

    def _lift(self, o):
        if isinstance(o, HD): return o
        o = np.asarray(o, float); c = np.zeros((2 ** self.k,) + np.broadcast_shapes(o.shape, self.c.shape[1:])); c[0] = o; return HD(c, self.k)

    def __add__(self, o): o = self._lift(o); return HD(self.c + o.c, self.k)
    __radd__ = __add__
    def __neg__(self): return HD(-self.c, self.k)
    def __sub__(self, o): return self + (-self._lift(o))
    def __rsub__(self, o): return self._lift(o) - self

    def __mul__(self, o):
        o = self._lift(o); n = 2 ** self.k; shp = np.broadcast_shapes(self.c.shape[1:], o.c.shape[1:]); out = np.zeros((n,) + shp)
        for A in range(n):
            if not np.any(self.c[A]): continue
            for B in range(n):
                if A & B: continue
                out[A | B] = out[A | B] + self.c[A] * o.c[B]
        return HD(out, self.k)
    __rmul__ = __mul__

    def _apply(self, derivs):
        """f(x0 + n) = sum_j f^(j)(x0) n^j / j!, n nilpotent (n^(k+1) = 0); derivs(x0, j) gives f^(j)(x0)"""
        x0 = self.c[0]; nil = HD(self.c.copy(), self.k); nil.c[0] = 0; out = HD(np.zeros_like(self.c), self.k); out.c[0] = derivs(x0, 0)
        pw = HD(np.zeros_like(self.c), self.k); pw.c[0] = 1.0; fact = 1.0
        for j in range(1, self.k + 1):
            pw = pw * nil; fact *= j; out = out + pw * (derivs(x0, j) / fact)
        return out

    def recip(self):
        from math import factorial
        return self._apply(lambda x, j: (-1) ** j * factorial(j) / x ** (j + 1))

    def __truediv__(self, o):
        if isinstance(o, HD): return self * o.recip()
        return HD(self.c / np.asarray(o, float), self.k)

    def __rtruediv__(self, o): return self._lift(o) * self.recip()

    def sqrt(self):
        def d(x, j):
            coef = 1.0
            for i in range(j): coef *= (0.5 - i)
            return coef * x ** (0.5 - j)
        return self._apply(d)

    def __getitem__(self, idx):
        idx = idx if isinstance(idx, tuple) else (idx,); return HD(self.c[(slice(None),) + idx], self.k)

    def sum(self, axis=None):
        if axis is None: return HD(self.c.reshape(self.c.shape[0], -1).sum(axis=1), self.k)
        ax = tuple(a + 1 for a in (axis if isinstance(axis, tuple) else (axis,))); return HD(self.c.sum(axis=ax), self.k)

    @property
    def shape(self): return self.c.shape[1:]

    @property
    def real(self): return self.c[0]

    def coef(self, S): return self.c[S]


def hd_stack(items, k):
    return HD(np.stack([it.c if isinstance(it, HD) else np.broadcast_to(np.asarray(it, float), ()) for it in items], axis=1), k)


# ------------------------------------------------------------------------------------------------ structure maps as substitutions
def subst(c, images, k_new):
    """c: coefficient array (2^k, ...); images[i] = bitmask in the new algebra for e_{i+1} (0 means e_{i+1} -> 0). A monomial e^S maps
    to the product of its images (zero when two images share a generator)."""
    k = len(images); out = np.zeros((2 ** k_new,) + c.shape[1:])
    for S in range(2 ** k):
        if not np.any(c[S]): continue
        tgt = 0; ok = True
        for i in range(k):
            if S >> i & 1:
                im = images[i]
                if im == 0 or tgt & im: ok = False; break
                tgt |= im
        if ok: out[tgt] = out[tgt] + c[S]
    return out


def _bit(i): return 1 << i


def proj(k, j):
    """T^a p T^b on T^k M (a + b + 1 = k, j = b): kills generator j+1 (0-based j), renumbers the outer ones down"""
    return [0 if i == j else _bit(i if i < j else i - 1) for i in range(k)], k - 1


def zero(k, j):
    """T^a 0 T^b: T^k M -> T^{k+1} M, inserts a new generator at position j (0-based) with zero coefficients"""
    return [_bit(i if i < j else i + 1) for i in range(k)], k + 1


def lift(k, j):
    """T^a l T^b: T^k M -> T^{k+1} M, e_{j+1} -> e_{j+1} e_{j+2}, outer generators shifted up"""
    return [(_bit(j) | _bit(j + 1)) if i == j else _bit(i if i < j else i + 1) for i in range(k)], k + 1


def flip(k, j):
    """T^a c T^b: swaps generators j+1 and j+2"""
    im = [_bit(i) for i in range(k)]; im[j], im[j + 1] = im[j + 1], im[j]; return im, k


def plus(k, j):
    """T^a (+) T^b on T_2-points written with k+1 generators where generators j and j+1 (0-based) are the two summands:
    both map to the same new generator (their product dies)"""
    return [_bit(i) if i <= j else _bit(i - 1) for i in range(k + 1)], k


def apply(c, m):
    im, kn = m; return subst(c, im, kn)


def compose(*maps):
    """apply maps left to right: compose(f, g)(c) = g(f(c)); returns a function"""
    def F(c):
        for m in maps: c = apply(c, m)
        return c
    return F


# ------------------------------------------------------------------------------------------------ the lexicographic lemma
def lex_order(U):
    """the subsets of U (bitmask) in decreasing size of the monomial: e^S before e^T iff the lowest generator in S ^ T lies in T"""
    subs = [S for S in range(U + 1) if S & ~U == 0]
    def key(S):
        return tuple(-(S >> i & 1) for i in range(U.bit_length()))       # lowest index most significant; containing it = smaller
    return sorted(subs, key=key, reverse=True)


def lex_nonneg(seq, tol=0.0):
    for v in seq:
        if v > tol: return True
        if v < -tol: return False
    return True


def in_TH(c, k, tol=1e-12):
    """c: coefficients (2^k,) of one cell. Membership in T^k of the half-line, by the lemma"""
    for U in range(2 ** k):
        if not lex_nonneg([c[S] for S in lex_order(U)], tol): return False
    return True


def in_T(c, k, tol=1e-12):
    """c: (2^k, n) coefficients of a point over the n cells of the simplex: equalities and per-cell lemma"""
    c = np.asarray(c, float)
    if abs(c[0].sum() - 1) > 1e-9 or any(abs(c[S].sum()) > 1e-9 for S in range(1, 2 ** k)): return False
    return all(in_TH(c[:, x], k, tol) for x in range(c.shape[1]))


def in_T_def(c, k, scale=Fraction(1, 1000)):
    """membership in T^k of the half-line from the definition, exactly: T Y = {(y, u): y in Y and y + s u in Y for 0 < s <= s0}.
    'For all small s' is decided at one rational s per level, each far below the previous one (coefficients in {-1, 0, 1})."""
    c = [Fraction(v) for v in c]
    if k == 0: return c[0] >= 0
    half = 2 ** (k - 1); base = c[:half]; dirn = c[half:]; s = scale
    nxt = scale ** 3
    return in_T_def(base, k - 1, nxt) and in_T_def([b + s * d for b, d in zip(base, dirn)], k - 1, nxt * s)


# ------------------------------------------------------------------------------------------------ universality of the vertical lift
def feasible_cone(x, u, tol=1e-12):
    """u in C_x: zero-sum and u_y >= 0 where x_y = 0"""
    return abs(np.sum(u)) < 1e-9 and all(u[y] >= -tol for y in range(len(x)) if abs(x[y]) < tol)


def second_cone(x, u, w, tol=1e-12):
    """w in C_{x,u}: zero-sum and w_y >= 0 where x_y = u_y = 0 (the cone at the point 'x moved infinitesimally along u')"""
    return abs(np.sum(w)) < 1e-9 and all(w[y] >= -tol for y in range(len(x)) if abs(x[y]) < tol and abs(u[y]) < tol)


def T2_point(x, a, b, d):
    """coefficients of x + a e1 + b e2 + d e1 e2"""
    return np.array([x, a, b, d], float)


def vertical_fibre(x, U, W):
    """for candidate u in U, w in W (lists of zero-sum vectors): which (x, u, 0, w) lie in T^2 Delta (the fibre of T(p) over the zero
    section), and which lie in the image of the vertical lift nu(x, w, u) = (x, u, 0, w), i.e. u, w in C_x"""
    fib, img = [], []
    for u in U:
        for w in W:
            if in_T(T2_point(x, u, np.zeros_like(x), w), 2): fib.append((u, w))
            if feasible_cone(x, u) and feasible_cone(x, w): img.append((u, w))
    return fib, img


# ------------------------------------------------------------------------------------------------ lab maps on HD (interior category)
X3 = list(itertools.product((0, 1), repeat=3))


def hd_law(p, dirs):
    """p: 8-vector law of three binary variables; dirs: list of 8-vectors (or None); returns an HD 8-vector"""
    return HD.var(p, dirs)


def moments_hd(P):
    """P: HD 8-vector on X3. mu_i, M_ij (pair covariances), m_T (centred triple moment)"""
    xs = np.array(X3, float); mu = [(P * xs[:, i]).sum() for i in range(3)]
    def E(f): return (P * f).sum()
    M = {}
    for i, j in itertools.combinations(range(3), 2):
        M[(i, j)] = E(xs[:, i] * xs[:, j]) - mu[i] * mu[j]
    # centred triple moment: E[(x0-mu0)(x1-mu1)(x2-mu2)] expanded
    e012 = E(xs[:, 0] * xs[:, 1] * xs[:, 2]); e01, e02, e12 = E(xs[:, 0] * xs[:, 1]), E(xs[:, 0] * xs[:, 2]), E(xs[:, 1] * xs[:, 2])
    mT = e012 - mu[2] * e01 - mu[1] * e02 - mu[0] * e12 + 2 * mu[0] * mu[1] * mu[2]
    return mu, M, mT


def rhat_hd(P, sign=-1.0):
    """rhat = m_T / v_T with v_T = sign * sqrt(M01 M02 M12): smooth where M01 M02 M12 > 0 (off the coincidence strata)"""
    _, M, mT = moments_hd(P); prod = M[(0, 1)] * M[(0, 2)] * M[(1, 2)]; return mT / (prod.sqrt() * sign)


def poly_hd(r, coeffs):
    out = r * 0.0 + coeffs[-1]
    for a in reversed(coeffs[:-1]): out = out * r + a
    return out


def marginal_matrix3(window):
    """the 4 x 8 stochastic map from laws on X3 to the window's 2 x 2 table (affine: a Markov kernel)"""
    cells = list(itertools.product((0, 1), repeat=len(window)))
    return np.array([[1.0 if tuple(x[i] for i in window) == c else 0.0 for x in X3] for c in cells])


# ------------------------------------------------------------------------------------------------ the tetrad scheme (m = 4)
X4 = list(itertools.product((0, 1), repeat=4))


def tetrads_hd(P):
    xs = np.array(X4, float); mu = [(P * xs[:, i]).sum() for i in range(4)]
    def M(i, j): return (P * (xs[:, i] * xs[:, j])).sum() - mu[i] * mu[j]
    m01, m23, m02, m13, m03, m12 = M(0, 1), M(2, 3), M(0, 2), M(1, 3), M(0, 3), M(1, 2)
    return m01 * m23 - m02 * m13, m01 * m23 - m03 * m12


def tetrad_jets(p):
    """exact J (2 x 16) and H (2 x 16 x 16) of the tetrads at the law p (16-vector), from W_2-points p + e_i e1 + e_j e2"""
    n = 16; J = np.zeros((2, n)); H = np.zeros((2, n, n)); I = np.eye(n)
    for i in range(n):
        for j in range(i, n):
            T = tetrads_hd(HD.var(p, [I[i], I[j]]))
            for t in range(2):
                H[t, i, j] = H[t, j, i] = T[t].c[3]
                J[t, i] = T[t].c[1]
    return J, H


def lift_pair(J, H, u, v, B=None, tol=1e-9):
    """obstruction to completing (p, u, v) in T_2 V to (p, u, v, d) in T^2 V: the class of H(u, v) in the cokernel of J restricted to
    the simplex's tangent space (directions with sum 0); returns (norm of the class, a completing d or None)"""
    if B is None: B = null_space(np.ones((1, J.shape[1])))
    JB = J @ B; q = np.einsum("kij,i,j->k", H, u, v); Uu, sv, _ = np.linalg.svd(JB); rk = int(np.sum(sv > tol * max(sv[0], 1e-300)))
    ob = float(np.linalg.norm(Uu[:, rk:].T @ q)) if rk < JB.shape[0] else 0.0
    if ob > 1e-9: return ob, None
    z = np.linalg.lstsq(JB, -q, rcond=None)[0]; return ob, B @ z
