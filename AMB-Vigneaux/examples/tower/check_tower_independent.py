"""Independent exact checks of two results of 'The Three-Level Posterior Tower' (v4), written from the paper's definitions,
not from its scripts. Fractions throughout.
  Theorem 6.1  C_E(X) = X x E(X) is a comonad for any functor E, with delta(x, c) = ((x, c), E(j_{X,c})(c));
               checked for E = D o D (second-order distributions: the context is transported, not copied).
  Theorem 9.8  lambda^B(M) = (D(pr1) M, beta_B(D(pr2) M)) : D C^B => C^B D satisfies (U), (Co), (M), (Cm), with
               A_B = D(B) x D^2(B) and C^B(Y) = Y x A_B; coherent contexts (mu Pi = rho) are closed under beta_B."""
import random, itertools
from fractions import Fraction as Fr

def D(d):                       # a distribution as a hashable frozenset of (outcome, prob) with prob > 0
    acc = {}
    for k, v in d:
        if v: acc[k] = acc.get(k, 0) + Fr(v)
    return frozenset(acc.items())
def push(f, d): return D((f(x), p) for x, p in d)
def eta(x): return D([(x, 1)])
def mu(dd): return D((x, p * q) for d, p in dd for x, q in d)        # barycenter D^2 -> D
def rd(r, xs, k=None):
    xs = list(xs) if k is None else r.sample(list(xs), k); w = [r.randint(1, 3) for _ in xs]; z = sum(w)
    return D((x, Fr(a, z)) for x, a in zip(xs, w))

# ---------------- Theorem 6.1 with E = D o D
def E_map(f, c): return push(lambda d: push(f, d), c)                 # E(f) = D(D(f))
def j(c): return lambda y: (y, c)
def eps(w): return w[0]
def delta(w): x, c = w; return (w, E_map(j(c), c))
def Cmap(f, w): x, c = w; return (f(x), E_map(f, c))

def check_61(trials=40):
    X = [0, 1, 2]
    for t in range(trials):
        r = random.Random(t); c = rd(r, [rd(r, X, 2) for _ in range(2)]); w = (r.choice(X), c)
        assert eps(delta(w)) == w
        assert Cmap(eps, delta(w)) == w
        assert delta(delta(w)) == Cmap(delta, delta(w))
        f = {0: "a", 1: "b", 2: "b"}.get                                   # naturality of delta along a non-injective map
        assert delta(Cmap(f, w)) == Cmap(lambda v: Cmap(f, v), delta(w))
    return True

# ---------------- Theorem 9.8: base-relative law
B = [0, 1]
def beta(dA):                                                          # componentwise barycenter on A_B = D(B) x D^2(B)
    rho = mu(D((r_, p) for (r_, _), p in dA)); Pi = mu(D((P, p) for (_, P), p in dA)); return (rho, Pi)
def lam(M):                                                            # M in D(Y x A_B)
    return (push(lambda w: w[0], M), beta(push(lambda w: w[1], M)))
def CB(f, w): y, c = w; return (f(y), c)
def epsB(w): return w[0]
def deltaB(w): return (w, w[1])
def rcontext(r, coherent):
    S = [rd(r, B) for _ in range(2)]; Pi = rd(r, S); return (mu(Pi) if coherent else rd(r, B), Pi)

def check_98(trials=40):
    Y = ["u", "v"]
    for t in range(trials):
        r = random.Random(100 + t); ctxs = [rcontext(r, t % 2 == 0) for _ in range(2)]
        M = D(((r.choice(Y), ctxs[i]), Fr(1 + i, 3)) for i in range(2))
        y, c = r.choice(Y), ctxs[0]
        assert lam(eta((y, c))) == CB(eta, (y, c))                                       # (U)
        assert epsB(lam(M)) == push(epsB, M)                                             # (Co)
        Ms = [D(((r.choice(Y), rcontext(r, True)), Fr(1, 2)) for _ in range(2)) for _ in range(2)]
        MM = D((Ms[i], Fr(1 + i, 3)) for i in range(2))
        assert lam(mu(MM)) == CB(mu, lam(push(lam, MM)))                                 # (M)
        assert deltaB(lam(M)) == CB(lam, lam(push(deltaB, M)))                           # (Cm)
        cc = [rcontext(r, True) for _ in range(3)]; bc = beta(D((cc[i], Fr(1, 3)) for i in range(3)))
        assert mu(bc[1]) == bc[0]                                                        # coherence closed under beta
    return True

if __name__ == "__main__":
    print("Theorem 6.1 (E = D o D): comonad laws and naturality of delta:", check_61())
    print("Theorem 9.8: (U), (Co), (M), (Cm) and coherence closure:", check_98())
