"""Markov-category readings of the lab (lab paper, Section "Markov categories"): the two-type model as a factorisation through a two-point
object with copying; sufficiency on the hiding line (Fisher information per window); de Finetti for batches (the mixing measure of the
batch laws and the monad multiplication that pools it); Blackwell's order of experiments on a three-point slice of the segment.

  factorise(w, a1, a2)                 the composite law  I --w--> Z --copy--> Z x Z x Z --(K_A x K_B x K_C)--> X_A x X_B x X_C
  fisher_on_line(P, window)            Fisher information about s carried by one window, for the family P + s chi_ABC at s = 0
  batch_mixing(P, Q, eps)              the empirical measure of batch laws (atoms and weights), its pool mu(pi), and the Dirac test
  exch_defect(Pb, Pc)                  max |P_b(x) P_c(y) - P_c(x) P_b(y)|: the failure of exchangeability across two batches
  bayes_value(E, U, prior)             max expected utility after observing the experiment E (rows: states, columns: signals)
  maxmin_value(E, U)                   ex-ante maxmin over randomized rules signal -> act (a linear programme)
  rule_value(E, U, rule)               the worst-case value of a fixed rule (signal -> act)
"""
from __future__ import annotations
import itertools
import numpy as np
from scipy.optimize import linprog
from . import injection as IJ

XS = list(itertools.product((0, 1), repeat=3))


def factorise(w, a1, a2):
    """returns the 8-cell law and the pieces of the diagram: the state w on Z, the kernels K_i (rows z, columns x_i)"""
    Wz = np.array([w, 1 - w]); K = [np.array([[1 - a1[i], a1[i]], [1 - a2[i], a2[i]]]) for i in range(3)]
    P = np.zeros((2, 2, 2))
    for x in XS:
        P[x] = sum(Wz[z] * K[0][z, x[0]] * K[1][z, x[1]] * K[2][z, x[2]] for z in range(2))
    return P, Wz, K


def fisher_on_line(P, window):
    """I(s) = sum over the window's cells of (d p / d s)^2 / p for the family P + s chi_ABC; marginals of chi on proper windows vanish"""
    chi = IJ.as_law(IJ.chi(3, (0, 1, 2)), 3); keep = tuple(window)
    ax = tuple(i for i in range(3) if i not in keep); dp = chi.sum(axis=ax) if ax else chi; p = P.sum(axis=ax) if ax else P
    return float(((dp ** 2) / p).sum())


def batch_mixing(P, Q, eps):
    """eps: list of batch shares. Atoms are the distinct batch laws; pi their weights; pool = mu(pi) = sum pi_k P_k"""
    laws = [IJ.mix(P, Q, e) for e in eps]; atoms = []; wts = []
    for L in laws:
        for k, A in enumerate(atoms):
            if np.allclose(A, L, atol=1e-15): wts[k] += 1; break
        else: atoms.append(L); wts.append(1)
    wts = np.array(wts, float) / len(laws); pool = sum(wv * A for wv, A in zip(wts, atoms))
    return atoms, wts, pool, len(atoms) == 1


def exch_defect(Pb, Pc):
    a = IJ.as_vec(Pb); b = IJ.as_vec(Pc); return float(np.abs(np.outer(a, b) - np.outer(b, a)).max())


def bayes_value(E, U, prior):
    """E: states x signals (rows sum to 1); U: acts x states; returns sum over signals of max_a sum_theta prior E U"""
    E = np.asarray(E, float); U = np.asarray(U, float); prior = np.asarray(prior, float)
    return float(sum(max(U[a] @ (prior * E[:, s]) for a in range(U.shape[0])) for s in range(E.shape[1])))


def maxmin_value(E, U):
    """max_q min_theta sum_s E[theta, s] sum_a q(a|s) U[a, theta] over randomized rules q (an LP); returns (value, q)"""
    E = np.asarray(E, float); U = np.asarray(U, float); nT, nS = E.shape; nA = U.shape[0]; nq = nS * nA
    c = np.zeros(nq + 1); c[-1] = -1.0
    A_ub = []; b_ub = []
    for th in range(nT):
        row = np.zeros(nq + 1); row[-1] = 1.0
        for s in range(nS):
            for a in range(nA): row[s * nA + a] = -E[th, s] * U[a, th]
        A_ub.append(row); b_ub.append(0.0)
    A_eq = []; b_eq = []
    for s in range(nS):
        row = np.zeros(nq + 1); row[s * nA:(s + 1) * nA] = 1.0; A_eq.append(row); b_eq.append(1.0)
    r = linprog(c, A_ub=np.array(A_ub), b_ub=b_ub, A_eq=np.array(A_eq), b_eq=b_eq, bounds=[(0, 1)] * nq + [(None, None)], method="highs")
    return float(-r.fun), r.x[:-1].reshape(nS, nA)


def rule_value(E, U, rule):
    E = np.asarray(E, float); U = np.asarray(U, float)
    return float(min(sum(E[th, s] * U[rule[s], th] for s in range(E.shape[1])) for th in range(E.shape[0])))
