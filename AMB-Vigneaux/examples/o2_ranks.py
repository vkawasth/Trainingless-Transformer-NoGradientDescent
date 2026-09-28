#!/usr/bin/env python3
"""O2 — is the merged model a minimal sufficient statistic?

For each level l of the TRUE grammar of a corpus built by build_corpus_syn.py:

  downward  R_l      V × c² child-row matrix (rows P(children | B = b))
            D_l      V × A^(2^l) law of the subtree's leaves given b;
                     rank via the Gram recursion G_l = R_l (G_{l-1} ⊗ G_{l-1}) R_lᵀ
  upward    P_l      V × |outside| joint P(B = b, outside leaves) for a node in the
                     LEFT or RIGHT child position; rank via the outside Gram
                     K_l[b,b'] = Σ K_{l+1}[p,p'] R[p,b,c] R[p',b',c'] G_l[c,c']
  minimal   r*_l     = rank P(subtree leaves, outside leaves) = rank(G^{1/2} K^{1/2}):
                     the dimension of the minimal sufficient statistic of the subtree
                     for the outside (P(O | x) spans an r*-dimensional space)

and the counts of DISTINCT child rows and distinct normalised upward rows,
which are what a partition (merge) can exploit.
"""
import json, sys, itertools
import numpy as np

data = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/hier/cH"
M = json.load(open(f"{data}/rhm_meta.json"))
L, V, A = M["depth"], M["nsym"], M["nleaf"]
card = lambda l: A if l == 1 else V
P = {}
for l in range(1, L + 1):
    t = np.zeros((V, card(l), card(l)))
    for s, r in M["rules_all"][str(l)].items():
        for (x, y) in r:
            t[int(s), x, y] += 1.0 / len(r)
    P[l] = t
prior = np.full(V, 1.0 / V)          # uniform root, as generated


def rank(X, tol=1e-10):
    s = np.linalg.svd(X, compute_uv=False)
    return int((s > tol * max(1.0, s.max())).sum())


def psd_sqrt(G):
    w, U = np.linalg.eigh((G + G.T) / 2)
    return U @ np.diag(np.sqrt(np.clip(w, 0, None))) @ U.T


def distinct_rows(X, tol=1e-9):
    reps = []
    for x in X:
        if not any(np.abs(x - r).max() < tol for r in reps):
            reps.append(x)
    return len(reps)


# downward Grams: G_0 = I_A (one-hot leaves), G_l = R (G⊗G) Rᵀ
G = {0: np.eye(A)}
for l in range(1, L + 1):
    R = P[l].reshape(V, -1)
    G[l] = R @ np.kron(G[l - 1], G[l - 1]) @ R.T

# symbol marginals at each level (prior pushed down) for the outside construction
marg = {L: prior}
for l in range(L, 1, -1):
    j = np.einsum("p,pxy->xy", marg[l], P[l])
    marg[l - 1] = (j.sum(1) + j.sum(0)) / 2.0

# outside Grams: K[b,b'] = Σ_o P(b,o) P(b',o) for the joint of B and the outside
# of ONE node at level l in a given position; the root has an empty outside.
K = {L: {"root": np.outer(prior, prior)}}
for l in range(L - 1, 0, -1):
    K[l] = {}
    Rp = P[l + 1]                                   # parent rule tensor p -> (left, right)
    Kpar = K[l + 1]["left"] if l + 1 < L else K[L]["root"]    # parent taken in the left position
    Gs = G[l]                                       # downward Gram of the sibling subtree
    # node = left child: P(b, o) = Σ_p P(p, o_p) Σ_c R[p,b,c] P(sib | c)
    K[l]["left"] = np.einsum("pq,pbc,qde,ce->bd", Kpar, Rp, Rp, Gs)
    K[l]["right"] = np.einsum("pq,pcb,qed,ce->bd", Kpar, Rp, Rp, Gs)

print(f"true grammar: L={L} V={V} A={A}  ({data})\n")
print(f"{'level':>5}{'pos':>7}{'V':>4}{'distinct child rows':>21}{'rank R':>8}{'rank D':>8}"
      f"{'rank upward':>13}{'distinct upward':>17}{'r* (minimal)':>14}")
rows = []
for l in range(1, L + 1):
    keep = marg[l] > 0 if l < L else np.ones(V, bool)      # drop unreachable symbols
    R = P[l].reshape(V, -1)[keep]
    Vr = int(keep.sum())
    dcr, rR, rD = distinct_rows(R), rank(R), rank(G[l][np.ix_(keep, keep)])
    for pos, Kl in K[l].items():
        Kl = Kl[np.ix_(keep, keep)]
        rU = rank(Kl)
        # distinct normalised upward rows: P(o | b) = P(b,o)/P(b); Gram of conditionals
        pb = np.sqrt(np.clip(np.diag(Kl), 1e-300, None))
        # rows b, b' of P(o|b) are equal iff their conditional Gram distance is 0
        m = marg[l][keep] if pos != "root" else prior
        Kc = Kl / np.outer(m, m)
        dist = np.diag(Kc)[:, None] + np.diag(Kc)[None, :] - 2 * Kc
        classes = []
        for b in range(Vr):
            for cl in classes:
                if abs(dist[b, cl[0]]) < 1e-9 * max(1, np.abs(Kc).max()):
                    cl.append(b); break
            else:
                classes.append([b])
        rstar = rank(psd_sqrt(G[l][np.ix_(keep, keep)]) @ psd_sqrt(Kl))
        rows.append((l, pos, dcr, rR, rD, rU, len(classes), rstar))
        print(f"{l:>5}{pos:>7}{Vr:>4}{dcr:>21}{rR:>8}{rD:>8}{rU:>13}{len(classes):>17}{rstar:>14}")

mu = np.einsum("p,pxy->xy", prior, P[L])
print(f"\nroot coupling μ(u,v) = Σ_b π_b R_L[b,u,v]: rank {rank(mu)}  (V = {V})")
json.dump({"rows": rows, "rank_mu": rank(mu)}, open(f"{data}/o2_ranks.json", "w"))
