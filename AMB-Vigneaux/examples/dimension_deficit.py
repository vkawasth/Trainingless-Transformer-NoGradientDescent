"""Dimension deficit rank(M_loop − I) from estimated edge supports, for affine operation
systems on a 4-cycle over F_3 and F_5, with a threshold sweep and a parametric-bootstrap
null band.  Compared against γ over Z/p (and Z), logical contextuality, and the Z/n mode
holonomy (translations only)."""
import json, itertools, numpy as np
from amb_vigneaux.holonomy import Graph, discrete_class
from amb_vigneaux.scenario import Scenario
from amb_vigneaux.outcome import cohomological_obstruction, compatible_global_sections
from amb_vigneaux.magphase import *

G = Graph(4, [(0, 1), (1, 2), (2, 3), (3, 0)])
TAUS = (0.2, 0.3, 0.4, 0.5, 0.6)


def planted(p):
    return {"flat": [(1, 1), (1, 2), (1, 0), (1, p - 3)],            # b's sum to 0
            "translation h=1": [(1, 1), (1, 2), (1, 0), (1, p - 2)],  # a_loop = 1, b_loop = 1
            "multiplicative": [(2, 0), (1, 0), (1, 0), (1, 0)]}       # a_loop = 2 ≠ 1


def support_map(p, ops):
    outs = {f"c{v}": tuple(range(p)) for v in range(G.n_vertices)}
    ctx = tuple((f"c{a}", f"c{b}") for a, b in G.edges)
    return Scenario(outs, ctx), {C: [(x, (a * x + b) % p) for x in range(p)] for C, (a, b) in zip(ctx, ops)}


def layer3(p, ops):
    sc, supp = support_map(p, ops)
    C0 = sc.contexts[0]
    gs = compatible_global_sections(sc, supp)
    logical = any(not any(tuple(g[sc.measurements.index(m)] for m in C0) == s for g in gs) for s in supp[C0])
    gam = {r: any(not cohomological_obstruction(sc, supp, C0, s, ring=r).obstruction_vanishes for s in supp[C0])
           for r in ("Z", p)}
    return dict(n_global=len(gs), logical=logical, gamma_Z=gam["Z"], gamma_p=gam[p])


def eps_hat(tables, ops, p):
    off = sum(T.sum() - sum(T[x, (a * x + b) % p] for x in range(p)) for T, (a, b) in zip(tables, ops))
    tot = sum(T.sum() for T in tables)
    return float(np.clip(off / tot * p / (p - 1), 1e-3, 0.99))


def nearest_flat(ops, p):
    ops = list(ops)
    M = loop_matrix(G, ops, p, G.C[0])
    if rank_mod_p((M - np.eye(2, dtype=np.int64)) % p, p) == 0:
        return ops
    k = G.cotree[0]                          # re-solve the cotree edge so the loop closes
    for a, b in itertools.product(range(1, p), range(p)):
        trial = ops[:k] + [(a, b)] + ops[k + 1:]
        if rank_mod_p((loop_matrix(G, trial, p, G.C[0]) - np.eye(2, dtype=np.int64)) % p, p) == 0:
            return trial
    raise RuntimeError


rng = np.random.default_rng(1)
out = {}
for p in (3, 5):
    for name, ops in planted(p).items():
        truth = dimension_deficit(G, ops, p)
        L3 = layer3(p, ops)
        mode_h = int(G.holonomy([b for _, b in ops], p)[0]) if all(a == 1 for a, _ in ops) else None
        print(f"\nF_{p}  {name}: true deficit {truth[0]}, global sections {truth[1]} | "
              f"logical {L3['logical']}, γ_Z≠0 {L3['gamma_Z']}, γ_{p}≠0 {L3['gamma_p']}, translation holonomy {mode_h}")
        print(f"{'eps':>5}{'N':>5}{'tau':>6}{'recovered':>11}{'P(def>0)':>10}{'null P(def>0)':>15}{'P(glob=0)':>11}")
        for eps in (0.1, 0.3):
            for N in (30, 90, 300):
                R, B = 150, 60
                for tau in TAUS:
                    rec = det = glob0 = 0
                    null_det = null_tot = 0
                    for r in range(R):
                        T = sample_affine_edges(G, ops, p, N, eps, rng)
                        d, g, est = estimated_deficit(G, T, p, tau)
                        if d is None:
                            continue
                        rec += 1; det += d > 0; glob0 += g == 0
                        if r < 20:                                   # null band from nearest flat fit
                            f0, e0 = nearest_flat(est, p), eps_hat(T, est, p)
                            for _ in range(B // 20 * 1 or 1):
                                Tn = sample_affine_edges(G, f0, p, N, e0, rng)
                                dn, _, _ = estimated_deficit(G, Tn, p, tau)
                                if dn is not None:
                                    null_tot += 1; null_det += dn > 0
                    key = f"p{p}|{name}|eps{eps}|N{N}|tau{tau}"
                    out[key] = dict(recovered=rec / R, det=det / max(rec, 1), null=null_det / max(null_tot, 1),
                                    glob0=glob0 / max(rec, 1))
                    if tau in (0.3, 0.5):
                        v = out[key]
                        print(f"{eps:>5}{N:>5}{tau:>6}{v['recovered']:>11.2f}{v['det']:>10.2f}{v['null']:>15.2f}{v['glob0']:>11.2f}")
        out[f"p{p}|{name}|layer3"] = dict(L3, true_deficit=truth[0], true_global=truth[1], translation_h=mode_h)
json.dump(out, open("examples/dimension_deficit.json", "w"), indent=1)
