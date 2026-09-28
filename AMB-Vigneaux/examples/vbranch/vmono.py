"""Arity-graded monodromy on Corpus V with three labelling sources.

Latent content: Corpus V trees (true grammar).  Level-1 chunks carry a true symbol b (8 values)
and an arity k (number of right children, 0..5).
Sources A, B, C label chunks with their own alphabets: label = pi_s(b) w.p. 1 - eps, else uniform.
Edges of the cover: the pairs (A,B), (B,C), (C,A), each compared on a DISJOINT third of the
documents -- so no chunk is labelled by all three and the cover has a genuine cycle.
Planted twist (the positive): on documents shared with A, source C applies tau (a cyclic shift of
its alphabet) to ODD-arity chunks only.  Null: no twist.
Transport on edge s->t in grade G: T[x, y] = P(label_t = y | label_s = x) from co-labelled chunks of
grade G.  Loop operator M_G = T_AB T_BC T_CA.  Read: nearest permutation (holonomy), its fixed
points (the permutation character), and the eigenvalues (moduli = magnitude, phases = holonomy).
Grades: pooled (all k), parity of k, each k.
"""
import json, sys, itertools, numpy as np
from vtree import VGrammar, sample

V = 8
D = json.load(open("cV/corpus.json")); g = VGrammar.from_json(D["grammar"])
rng0 = np.random.default_rng(7)
_, TREES = sample(g, 6000, rng0, return_trees=True)


def chunks(tree):
    out = []
    def walk(node, l):
        b, kids = node
        if l == 1:
            out.append((b, len(kids) - 1))
        else:
            for c in kids:
                walk(c, l - 1)
    walk(tree, g.L)
    return out


CH = [chunks(t) for t in TREES]
TAU = np.roll(np.arange(V), 1)                        # cyclic shift of the alphabet


def simulate(rng, eps, twist, docs):
    pis = {s: rng.permutation(V) for s in "ABC"}
    edges = {("A", "B"): [], ("B", "C"): [], ("C", "A"): []}
    order = rng.permutation(len(docs))
    thirds = np.array_split(order, 3)
    for (edge, idx) in zip(edges, thirds):
        for d in idx:
            for b, k in docs[d]:
                labs = []
                for s in edge:
                    lab = pis[s][b]
                    if twist and s == "C" and edge == ("C", "A") and k % 2 == 1:
                        lab = TAU[lab]
                    if rng.random() < eps:
                        lab = rng.integers(V)
                    labs.append(int(lab))
                edges[edge].append((labs[0], labs[1], k))
    return edges


MINC = 30


def counts(pairs, sel):
    N = np.zeros((V, V))
    for x, y, k in pairs:
        if sel(k):
            N[x, y] += 1
    return N


def loop(edges, sel):
    """Loop operator restricted to the grade's support: a label is ACTIVE if it is observed at
    least MINC times as the source side of every edge along the loop (A, then B, then C)."""
    Ns = [counts(edges[e], sel) for e in (("A", "B"), ("B", "C"), ("C", "A"))]
    Ts = [(N + 1e-9) / (N + 1e-9).sum(1, keepdims=True) for N in Ns]
    M = Ts[0] @ Ts[1] @ Ts[2]
    # noise floor: estimate eps from the POOLED A-B transport (every symbol well populated), then a
    # label is active in this grade if its count is at least 3x the count noise alone would give it
    P = counts(edges[("A", "B")], lambda k: True)
    rowmax = (P.max(1) / P.sum(1).clip(1)).mean()
    q_ab = float(np.clip((rowmax - 1 / V) / (1 - 1 / V), 1e-6, 1))    # P(both A and B label correctly)
    eps_hat = 1 - np.sqrt(q_ab)                                        # one source's noise (symmetric sources)
    rc = Ns[0].sum(1)
    act = (rc >= MINC) & (rc >= 2 * eps_hat * rc.sum() / V)
    return M, act


from scipy.optimize import linear_sum_assignment


def holonomy_perm(edges, sel):
    """Estimate each edge's alignment as a one-to-one assignment maximising co-label counts
    (Hungarian), then compose around the loop: h = pi_CA o pi_BC o pi_AB."""
    perms = []
    for e in (("A", "B"), ("B", "C"), ("C", "A")):
        N = counts(edges[e], sel)
        r, c = linear_sum_assignment(-N)
        pm = np.empty(V, dtype=int); pm[r] = c
        perms.append(pm)
    return perms[2][perms[1][perms[0]]]


def read(Mact, h=None):
    M, act = Mact
    idx = np.flatnonzero(act)
    if len(idx) == 0:
        return dict(active=0, fixed_points=0, nonfixed=0, trace=np.nan, second_modulus=np.nan)
    perm = h[idx] if h is not None else M[idx].argmax(1)
    fixed = int((perm == idx).sum())
    sub = M[np.ix_(idx, idx)]
    ev = np.linalg.eigvals(sub)
    mods = np.sort(np.abs(ev))[::-1]
    return dict(active=int(len(idx)), fixed_points=fixed, nonfixed=int(len(idx) - fixed),
                trace=float(np.trace(sub)) / len(idx), second_modulus=float(mods[1]) if len(mods) > 1 else np.nan)


GRADES = {"pooled": lambda k: True, "even k": lambda k: k % 2 == 0, "odd k": lambda k: k % 2 == 1}
GRADES.update({f"k={k}": (lambda k0: (lambda k: k == k0))(k) for k in range(6)})

if __name__ == "__main__":
    NDOC = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    out = {}
    for eps in (0.1, 0.3):
        for twist in (False, True):
            acc = {G: [] for G in GRADES}
            for r in range(REPS):
                rng = np.random.default_rng(1000 * r + int(10 * eps) + (500 if twist else 0))
                docs = [CH[i] for i in rng.choice(len(CH), NDOC, replace=False)]
                E = simulate(rng, eps, twist, docs)
                for G, sel in GRADES.items():
                    acc[G].append(read(loop(E, sel), holonomy_perm(E, sel)))
            key = f"eps={eps},{'twist' if twist else 'null'}"
            out[key] = {G: dict(detect=float(np.mean([a["nonfixed"] > 0 for a in v])),
                                active=float(np.mean([a["active"] for a in v])),
                                moved=float(np.mean([a["nonfixed"] for a in v])),
                                trace_per_label=float(np.nanmean([a["trace"] for a in v])),
                                second_modulus=float(np.nanmean([a["second_modulus"] for a in v])))
                        for G, v in acc.items()}
            print(f"\n{key}  (P[holonomy != id] | active labels | labels moved | trace/label | 2nd |eigenvalue|)")
            for G, v in out[key].items():
                print(f"  {G:>7}: detect {v['detect']:.2f} | active {v['active']:.1f} | moved {v['moved']:.2f} | "
                      f"trace/label {v['trace_per_label']:.3f} | |lambda_2| {v['second_modulus']:.3f}")
    json.dump(out, open(f"cV/vmono_N{NDOC}.json", "w"), indent=1)
