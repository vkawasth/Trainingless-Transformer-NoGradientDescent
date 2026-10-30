"""check_quantum.py -- the lab as the commutative part of Ikeda's entanglement obstruction (lab paper, Proposition "the lab inside the
quantum marginal problem"; Ikeda, arXiv:2511.04326v2).

Ikeda's presheaf S of density operators on subsystems, with restriction = partial trace, has the map j_U : S(I) -> H^0(U, S); its
failures are (A) gluing failure (j_U not surjective) and (B) non-uniqueness (j_U not injective). The lab's K is j_U restricted to the
diagonal (classical) states in the product basis.

  QU1 dephasing commutes with partial trace: for random three-qubit states, the diagonal of Tr_C rho equals Tr_C of the diagonal of rho
      (and likewise for every subsystem), so dephasing is a retraction of the quantum marginal problem onto the classical one
  QU2 hence classical gluing = quantum gluing for diagonal families: a diagonal compatible family has a quantum global state iff it has a
      classical one; the liar cycle has neither (the dephased state of any candidate would be a classical global law)
  QU3 non-uniqueness grows: on the pair cover the classical kernel of K has dimension 1 (chi_ABC, which is -Z Z Z on the diagonal); the
      kernel of the quantum restriction to the three two-qubit windows is spanned by the 27 weight-3 Pauli strings: coherences add 26
      hidden directions that no window sees
  QU4 the characteristic number of the classical fibre: the line P + s chi has lattice length s_+ + |s_-| = 0.040 on the worked example
      (the symplectic area of the toric CP^1 over 2 pi); the hidden channel moves the point but leaves the length unchanged
"""
import itertools, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pce.lab import law
from pce import injection as IJ
from pce import sources as SR

rng = np.random.default_rng(20261029); results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)

def ptrace(rho, keep, n=3):
    """partial trace of an n-qubit density matrix onto the qubits in keep (sorted)"""
    r = rho.reshape([2] * (2 * n)); out = [i for i in range(n) if i not in keep]
    for k, i in enumerate(sorted(out, reverse=True)):
        m = n - k; r = np.trace(r, axis1=i, axis2=i + m)
    d = 2 ** len(keep); return r.reshape(d, d)

def rand_state(n=3):
    G = rng.normal(size=(2 ** n, 2 ** n)) + 1j * rng.normal(size=(2 ** n, 2 ** n)); R = G @ G.conj().T; return R / np.trace(R).real

# QU1
worst = 0.0
for _ in range(50):
    rho = rand_state(); D = np.diag(np.diag(rho))
    for keep in [(0, 1), (1, 2), (0, 2), (0,), (1,), (2,)]:
        worst = max(worst, np.abs(np.diag(ptrace(rho, keep)) - np.diag(ptrace(D, keep))).max(), np.abs(ptrace(D, keep) - np.diag(np.diag(ptrace(D, keep)))).max())
check("QU1 dephasing commutes with partial trace (a retraction of the quantum marginal problem onto the classical one)", worst < 1e-12, f"max deviation {worst:.1e}")

# QU2: for a random quantum state with diagonal pair marginals forced by dephasing, its dephased version is a classical law with the same
# diagonal marginals; for the liar cycle, no classical law exists (lambda* = 0), hence no quantum state with those marginals either
ok = True
for _ in range(30):
    rho = rand_state(); D = np.diag(np.diag(rho)).real; P = np.diag(D).reshape(2, 2, 2)
    for keep in [(0, 1), (1, 2), (0, 2)]:
        ok &= np.allclose(np.diag(ptrace(rho, keep)).real, P.sum(axis=tuple(i for i in range(3) if i not in keep)).ravel())
check("QU2 a quantum global state for diagonal marginals dephases to a classical law with the same marginals; so classical and quantum gluing agree on diagonal families, and the liar cycle glues in neither",
      ok)

# QU3
def herm_basis(d):
    B = []
    for i in range(d):
        E = np.zeros((d, d), complex); E[i, i] = 1; B.append(E)
    for i, j in itertools.combinations(range(d), 2):
        E = np.zeros((d, d), complex); E[i, j] = E[j, i] = 1; B.append(E)
        E = np.zeros((d, d), complex); E[i, j] = 1j; E[j, i] = -1j; B.append(E)
    return B
Hb = herm_basis(8)
A = np.array([np.concatenate([np.r_[ptrace(Hm, keep).real.ravel(), ptrace(Hm, keep).imag.ravel()] for keep in [(0, 1), (1, 2), (0, 2)]]) for Hm in Hb]).T
qk = 64 - np.linalg.matrix_rank(A, 1e-9)
from pce import descent as DE
ck = DE.descent_kernel(3, [(0, 1), (1, 2), (0, 2)])[0].shape[1]
X_ = np.array([[0, 1], [1, 0]], complex); Y_ = np.array([[0, -1j], [1j, 0]]); Z_ = np.diag([1, -1]).astype(complex)
w3 = [np.kron(np.kron(a, b), c) for a in (X_, Y_, Z_) for b in (X_, Y_, Z_) for c in (X_, Y_, Z_)]
killed = all(np.abs(ptrace(Pm / 8, keep)).max() < 1e-12 for Pm in w3 for keep in [(0, 1), (1, 2), (0, 2)])
zzz_is_chi = np.allclose(np.diag(np.kron(np.kron(Z_, Z_), Z_)).real, -IJ.chi(3, (0, 1, 2)))
check("QU3 non-uniqueness grows: the classical kernel of K is 1-dimensional (chi_ABC = -Z Z Z on the diagonal); the quantum kernel is spanned by the 27 weight-3 Pauli strings",
      ck == 1 and qk == 27 and killed and zzz_is_chi, f"classical kernel {ck}; quantum kernel {qk} (of 64)")

# QU4
P = law(0.6, np.array([.5, .1, .4]), np.array([.05, .5, .1])); lo, hi = IJ.stealth_range(P, (0, 1, 2)); L = hi - lo
Q = SR.hidden_Q(P, hi); Pe = IJ.mix(P, Q, 0.05); lo2, hi2 = IJ.stealth_range(Pe, (0, 1, 2))
check("QU4 the classical fibre's characteristic number: lattice length 0.040 (area of the toric CP^1 over 2 pi); the hidden channel moves the point (s shifts by eps s) but not the length",
      abs(L - 0.040) < 1e-12 and abs((hi2 - lo2) - L) < 1e-12 and abs(lo2 - (lo - 0.05 * hi)) < 1e-12, f"length {L:.3f}; after injection [{lo2:.5f}, {hi2:.5f}]")

print(f"{sum(results)}/{len(results)} passed")
