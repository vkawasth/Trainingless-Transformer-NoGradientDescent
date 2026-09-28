"""Synthetic benchmark for the contradiction meter.

World: 12 binary propositions with hidden truth values. Honest units assert parities
x_i XOR x_j of random pairs, each claim sampled m times with flip noise eps.
Scenarios (200 random instances each):
  honest        4 honest units                               -> nothing should fire
  self          + a unit asserting an odd (Liar) cycle        -> self-contradiction
  overlap       + a unit flipping an edge an honest unit asserts -> overlap disagreement
  cross         + a unit flipping an edge that closes an honest path -> cross-unit odd cycle, blamed
  fabrication   + a unit consistent with a DIFFERENT world, on unshared propositions -> invisible
"""
import json, numpy as np
from amb_vigneaux.meter import run_meter

NP = 12


def honest_units(rng, x, k=4, e=6, m=1, eps=0.0):
    claims, used = [], set()
    for u in range(k):
        for _ in range(e):
            i, j = rng.choice(NP - 4, 2, replace=False)          # propositions 0..7 shared pool
            used.add((min(i, j), max(i, j)))
            claims += sample(f"h{u}", i, j, x[i] ^ x[j], m, eps, rng)
    return claims, used


def sample(unit, i, j, r, m, eps, rng):
    return [(unit, int(i), int(j), int(r ^ (rng.random() < eps))) for _ in range(m)]


def path_between(used, a, b):
    """BFS path a->b in the honest graph (list of vertices) or None."""
    adj = {}
    for i, j in used:
        adj.setdefault(i, []).append(j); adj.setdefault(j, []).append(i)
    prev, q = {a: None}, [a]
    while q:
        u = q.pop(0)
        for w in adj.get(u, []):
            if w not in prev:
                prev[w] = u; q.append(w)
    if b not in prev:
        return None
    p = [b]
    while prev[p[-1]] is not None:
        p.append(prev[p[-1]])
    return p[::-1]


def instance(kind, rng, m, eps):
    x = rng.integers(0, 2, NP)
    claims, used = honest_units(rng, x, m=m, eps=eps)
    truth = None
    if kind == "self":
        cyc = [8, 9, 10]                                           # fresh propositions
        rs = [0, 0, 1]                                             # 8=9, 9=10, 10!=8 (Liar)
        for (a, b), r in zip(zip(cyc, cyc[1:] + cyc[:1]), rs):
            claims += sample("bad", a, b, r, m, eps, rng)
        truth = "self"
    elif kind == "overlap":
        i, j = sorted(next(iter(used)))
        claims += sample("bad", i, j, 1 ^ x[i] ^ x[j], m, eps, rng)
        truth = "overlap"
    elif kind == "cross":
        for _ in range(100):
            a, b = rng.choice(NP - 4, 2, replace=False)
            p = path_between(used, a, b)
            if p and len(p) >= 3 and (min(a, b), max(a, b)) not in used:
                break
        else:
            return None
        claims += sample("bad", a, b, 1 ^ x[a] ^ x[b], m, eps, rng)
        truth = ("cross", (min(a, b), max(a, b)))
    elif kind == "fabrication":
        y = rng.integers(0, 2, NP)
        for a, b in [(8, 9), (9, 10), (10, 11), (8, 11)]:
            claims += sample("bad", a, b, y[a] ^ y[b], m, eps, rng)
        truth = "fabrication"
    return claims, truth


rng = np.random.default_rng(0)
rows = {}
for m, eps, mode in [(1, 0.0, "pooled"), (5, 0.1, "worst"), (5, 0.1, "pooled"), (3, 0.1, "pooled"),
                     (15, 0.2, "worst"), (15, 0.2, "pooled"), (5, 0.2, "pooled")]:
    for kind in ("honest", "self", "overlap", "cross", "fabrication"):
        c = dict(n=0, self=0, overlap=0, cross=0, blamed_right=0, any_fire=0, bad_involved=0)
        for _ in range(200):
            inst = instance(kind, rng, m, eps)
            if inst is None:
                continue
            claims, truth = inst
            rep = run_meter(claims, trusted=[f"h{u}" for u in range(4)], deterministic=(eps == 0.0), noise=mode)
            selfc, conf = rep.flagged(0.05, bonferroni=True)
            c["n"] += 1
            c["self"] += any(selfc.values())
            c["overlap"] += bool(rep.overlap_disagreements)
            c["cross"] += bool(conf)
            c["any_fire"] += any(selfc.values()) or bool(rep.overlap_disagreements) or bool(conf)
            flagged_units = set(u for u, cs in selfc.items() if cs) | \
                set(u for d in rep.overlap_disagreements for u in d["units"]) | set().union(*[cy.units for cy in conf]) if (selfc or conf or rep.overlap_disagreements) else set()
            c["bad_involved"] += "bad" in flagged_units
            if isinstance(truth, tuple):
                c["blamed_right"] += any(cy.blamed == truth[1] for cy in conf)
        rows[f"m={m},eps={eps},{mode},{kind}"] = {k: (v / c["n"] if k != "n" else v) for k, v in c.items()}
        r = rows[f"m={m},eps={eps},{mode},{kind}"]
        print(f"m={m:>2} eps={eps:.1f} {mode:>6} {kind:>12}: fire {r['any_fire']:.2f} | self {r['self']:.2f} "
              f"overlap {r['overlap']:.2f} cross {r['cross']:.2f} | bad unit implicated {r['bad_involved']:.2f}"
              + (f" | blamed correct edge {r['blamed_right']:.2f}" if kind == "cross" else ""))
json.dump(rows, open("examples/meter_benchmark.json", "w"), indent=1)
