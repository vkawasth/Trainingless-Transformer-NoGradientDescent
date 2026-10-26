"""scan_c5.py -- the 5-cycle (KCBS) representation schemes Rep_n(W_C5), n = 3 or 4 (note: Section 15).

For every class of classical points (direct sums of characters, up to the dihedral symmetry, sign flips of the observables and
reordering): singularity at second order, the block structure of nearby Hermitian models (eigenspaces of a random element of the
commutant), the share of nearby models violating a 5-cycle inequality, and the local maximum of the witness within distance 0.1.
Writes ../level2/c5_scan_n{n}.json. Run: python scan_c5.py 3 (about 2 minutes) or python scan_c5.py 4 (about an hour)."""
import sys, os, itertools, time, json, numpy as np, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pce.coxeter import analyse, diagonal_points
from scipy.optimize import least_squares, minimize
k = 5; G = ([str(i) for i in range(k)], [(i, (i + 1) % k) for i in range(k)])
GAM = [g for g in itertools.product((1, -1), repeat=k) if g.count(-1) % 2 == 1]
n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
iu = np.triu_indices(n); L = len(iu[0])
def witness(X):
    return max(np.linalg.eigvalsh((lambda B: (B + B.T) / 2)(sum(g[i] * X[i] @ X[(i + 1) % k] for i in range(k)))).max() for g in GAM) - (k - 2)
def unpack(z):
    out = []
    for v in range(k):
        M = np.zeros((n, n)); M[iu] = z[v * L:(v + 1) * L]; out.append(M + M.T - np.diag(np.diag(M)))
    return out
pack = lambda X: np.concatenate([x[iu] for x in X])
def F(z):
    X = unpack(z); r = [X[v] @ X[v] - np.eye(n) for v in range(k)] + [X[u] @ X[w] - X[w] @ X[u] for u, w in G[1]]
    return np.concatenate([q.ravel() for q in r])
def canon(key):
    best = None; cols = [tuple(int(x) for x in c) for c in key]
    for r in range(k):
        for refl in (False, True):
            perm = [((-i if refl else i) + r) % k for i in range(k)]
            for fl in itertools.product((1, -1), repeat=k):
                cc = tuple(sorted(tuple(fl[v] * c[perm[v]] for v in range(k)) for c in cols))
                if best is None or cc < best: best = cc
    return best
def blocks(X, rng):
    """block sizes of a Hermitian model: eigenspaces of a random Hermitian element of the commutant"""
    C = np.vstack([np.kron(x, np.eye(n)) - np.kron(np.eye(n), x.T) for x in X]); u, s, vt = np.linalg.svd(C)
    basis = vt[np.sum(s > 1e-7):]; Y = sum(rng.normal() * b.reshape(n, n) for b in basis); Y = (Y + Y.T) / 2
    ev = np.sort(np.linalg.eigvalsh(Y)); sizes = []; cur = 1
    for a, b in zip(ev[:-1], ev[1:]):
        if b - a < 1e-6: cur += 1
        else: sizes.append(cur); cur = 1
    sizes.append(cur); return tuple(sorted(sizes, reverse=True))
def local_max(X0, eps, starts, rng):
    z0 = pack(X0); best = -np.inf
    for s in range(starts):
        z = least_squares(F, z0 + eps * 0.5 * rng.normal(size=z0.size)).x
        cons = [{"type": "ineq", "fun": lambda zz: eps ** 2 - np.sum((zz - z0) ** 2)}]
        res = minimize(lambda zz: -witness(unpack(zz)) + 200 * np.sum(F(zz) ** 2), z, method="SLSQP", constraints=cons, options={"maxiter": 300})
        zz = least_squares(F, res.x).x
        if np.linalg.norm(F(zz)) < 1e-8 and np.sum((zz - z0) ** 2) <= (1.05 * eps) ** 2: best = max(best, witness(unpack(zz)))
    return best
if __name__ == "__main__":
    t = time.time(); reps = {}; sizes = {}
    for X, key in diagonal_points(G, n):
        c = canon(key); reps.setdefault(c, X); sizes[c] = sizes.get(c, 0) + 1
    print(f"n={n}: {sum(sizes.values())} classical points, {len(reps)} classes ({time.time()-t:.0f}s)", flush=True)
    rng = np.random.default_rng(11); rows = []
    for idx, (c, X) in enumerate(reps.items()):
        a = analyse(G, X, 3, rng); z0 = pack(X); decs = {}; ctx = 0; m = 0
        for eps in (0.03, 0.15):
            for _ in range(15):
                z = least_squares(F, z0 + eps * rng.normal(size=z0.size), xtol=1e-14, ftol=1e-14, gtol=1e-14).x
                if np.linalg.norm(F(z)) < 1e-9:
                    Xn = unpack(z); m += 1; d = blocks(Xn, rng); decs[str(d)] = decs.get(str(d), 0) + 1; ctx += witness(Xn) > 1e-6
        lm = local_max(X, 0.1, 3, rng)
        rows.append({"class": [list(col) for col in c], "size": sizes[c], "singular": bool(a["obstructed"]), "H1": a["H1"],
                     "nearby_blocks": decs, "random_ctx_share": ctx / max(m, 1), "local_max_witness": float(lm)})
        if idx % 20 == 0: print(idx, f"{time.time()-t:.0f}s", flush=True)
    json.dump(rows, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "level2", f"c5_scan_n{n}.json"), "w"))
    print("done", f"{time.time()-t:.0f}s", flush=True)
