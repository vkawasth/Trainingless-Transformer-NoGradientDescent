"""Second-order structure: covers whose nerve has 2-cells, the path 2-groupoid, and instrumentation for drawing the
nerve as a (representational) simplicial complex.

Nerve N(U): vertices = contexts; a set of contexts spans a simplex iff they share a measurement. Its 1-skeleton is the
context graph; its 2-cells (triples of contexts with a common measurement) are the 2-cells of the path 2-groupoid:
homotopies between paths of overlaps. A loop of the context graph that bounds 2-cells is filled: holonomy around it
is forced trivial by the triple overlap, and any obstruction moves up to H^2 of the nerve (a closed surface).

  triangle cover {LC, CR, LR}          Betti (1, 1)     one open loop: first-order holonomy, 3-way parity free
  tetrahedral cover {abc, abd, acd, bcd}  Betti (1, 0, 1)  graph has 3 loops, all filled; obstruction on the sphere,
                                                         4-way parity free (dim ker R = 1)

Instrumentation (`complex_instrument`) returns a JSON-able description: vertex positions (a fixed, representational
2-D layout: Schlegel diagram for 4 vertices, circle otherwise), edges with their overlap and signalling (TV between the
two restrictions to the overlap), faces with their common measurement and the triple agreement, per-context weights of
the Bell certificate (dual of the CF programme), Betti numbers, and which graph loops are filled. `plot_complex` draws it.
"""
from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Sequence

import numpy as np

from .scenario import EmpiricalModel, Scenario
from .outcome import contextual_fraction


def nerve(contexts: Sequence[Sequence[str]], max_dim: int = 3) -> Dict[int, List[tuple]]:
    C = [set(c) for c in contexts]; n = len(C)
    return {k: [s for s in itertools.combinations(range(n), k + 1) if set.intersection(*[C[i] for i in s])]
            for k in range(min(max_dim, n - 1) + 1)}


def _boundary(simp, k):
    rows = {s: i for i, s in enumerate(simp[k - 1])}
    D = np.zeros((len(simp[k - 1]), len(simp[k])))
    for j, s in enumerate(simp[k]):
        for t in range(len(s)):
            D[rows[s[:t] + s[t + 1:]], j] = (-1) ** t
    return D


def betti(contexts) -> List[int]:
    S = nerve(contexts, max_dim=len(contexts) - 1); r = {}
    for k in range(1, len(S)):
        r[k] = int(np.linalg.matrix_rank(_boundary(S, k))) if S[k] and S[k - 1] else 0
    return [len(S[k]) - r.get(k, 0) - r.get(k + 1, 0) for k in range(len(S)) if S[k]]


def path_2groupoid(contexts) -> dict:
    """objects = contexts, 1-cells = overlaps (edges), 2-cells = triple overlaps (faces). Fundamental loops of the
    context graph (spanning-tree cycles) and whether each is filled (a boundary of 2-cells, over Q)."""
    S = nerve(contexts, max_dim=2); E = S.get(1, []); F = S.get(2, [])
    n = len(contexts); parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    tree, loops = [], []
    for e in E:
        a, b = find(e[0]), find(e[1])
        if a != b:
            parent[a] = b; tree.append(e)
        else:
            loops.append(e)
    adj = {i: [] for i in range(n)}
    for (u, v) in tree:
        adj[u].append(v); adj[v].append(u)
    def tree_path(u, v):
        prev = {u: None}; q = [u]
        while q:
            x = q.pop(0)
            for y in adj[x]:
                if y not in prev:
                    prev[y] = x; q.append(y)
        p = [v]
        while p[-1] != u:
            p.append(prev[p[-1]])
        return p[::-1]
    eidx = {e: i for i, e in enumerate(E)}
    D2 = _boundary(S, 2) if F else np.zeros((len(E), 0))
    out = []
    for (u, v) in loops:
        cyc = tree_path(v, u) + [v]            # v -> ... -> u, then the closing edge u -> v
        z = np.zeros(len(E))
        for a, b in zip(cyc[:-1], cyc[1:]):
            z[eidx[(min(a, b), max(a, b))]] += 1 if a < b else -1
        filled = bool(D2.shape[1]) and np.linalg.matrix_rank(np.column_stack([D2, z])) == np.linalg.matrix_rank(D2)
        out.append(dict(cycle=[tuple(contexts[i]) for i in cyc], filled=bool(filled)))
    return dict(objects=[tuple(c) for c in contexts], one_cells=[(tuple(contexts[a]), tuple(contexts[b])) for a, b in E],
                two_cells=[tuple(tuple(contexts[i]) for i in f) for f in F], loops=out,
                graph_beta1=len(loops), filled=sum(l["filled"] for l in out), betti=betti(contexts))


def _layout(n):
    if n == 4:     # Schlegel diagram of the tetrahedron: three outer vertices, one in the middle
        return [np.array([np.cos(a), np.sin(a)]) for a in (np.pi / 2, np.pi / 2 + 2 * np.pi / 3, np.pi / 2 + 4 * np.pi / 3)] + [np.zeros(2)]
    return [np.array([np.cos(np.pi / 2 + 2 * np.pi * i / n), np.sin(np.pi / 2 + 2 * np.pi * i / n)]) for i in range(n)]


def complex_instrument(model: EmpiricalModel, labels: Optional[Sequence[str]] = None) -> dict:
    sc = model.scenario; ctx = list(sc.contexts); S = nerve(ctx, max_dim=2); pos = _layout(len(ctx))
    cf = contextual_fraction(model)
    w = {C: float(np.abs(cf.bell_inequality[C]).sum()) for C in ctx}; tot = sum(w.values()) or 1.0
    V = [dict(id=i, context=list(C), label=(labels[i] if labels else "".join(C)), pos=pos[i].tolist(), bell_weight=w[C] / tot) for i, C in enumerate(ctx)]
    E = []
    for (i, j) in S.get(1, []):
        O = sc.overlap(ctx[i], ctx[j])
        a = sc.restriction_matrix(ctx[i], O) @ model.tables[ctx[i]]; b = sc.restriction_matrix(ctx[j], O) @ model.tables[ctx[j]]
        E.append(dict(u=i, v=j, overlap=list(O), signalling=0.5 * float(np.abs(a - b).sum())))
    F = []
    for (i, j, k) in S.get(2, []):
        O = sc.overlap(sc.overlap(ctx[i], ctx[j]), ctx[k])
        m = [sc.restriction_matrix(ctx[t], O) @ model.tables[ctx[t]] for t in (i, j, k)]
        F.append(dict(vertices=[i, j, k], common=list(O), agreement=1 - max(0.5 * float(np.abs(x - y).sum()) for x, y in itertools.combinations(m, 2))))
    g = path_2groupoid(ctx)
    return dict(vertices=V, edges=E, faces=F, betti=g["betti"], graph_beta1=g["graph_beta1"], filled_loops=g["filled"], CF=cf.value,
                outer_face_filled=len(ctx) == 4 and len(F) == 4)


def plot_complex(inst: dict, ax, title: str = "", face_alpha=0.18, cmap="Blues"):
    """representational drawing of the nerve: filled triangles for 2-cells, edges for overlaps (width = signalling),
    vertices sized by the context's share of the Bell certificate"""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    P = {v["id"]: np.array(v["pos"]) for v in inst["vertices"]}
    cm = plt.get_cmap(cmap)
    if inst.get("outer_face_filled"):
        ax.add_patch(Polygon([P[0], P[1], P[2]], closed=True, facecolor=cm(0.35), alpha=0.10, lw=0))
        outer = [f for f in inst["faces"] if 3 not in f["vertices"]]
        if outer:
            ax.text(1.05, 0.95, ",".join(outer[0]["common"]) + "  (outer face)", fontsize=6.5, color="#52514e", ha="left", va="center")
    for f in inst["faces"]:
        if inst.get("outer_face_filled") and 3 not in f["vertices"]:
            continue
        a, b, c = (P[i] for i in f["vertices"])
        ax.add_patch(Polygon([a, b, c], closed=True, facecolor=cm(0.25 + 0.6 * f["agreement"]), alpha=face_alpha, edgecolor="none"))
        cen = (a + b + c) / 3
        ax.text(cen[0], cen[1], ",".join(f["common"]), fontsize=6.5, color="#52514e", ha="center", va="center")
    for e in inst["edges"]:
        a, b = P[e["u"]], P[e["v"]]
        ax.plot([a[0], b[0]], [a[1], b[1]], color="#0b0b0b", lw=0.8 + 25 * e["signalling"], alpha=0.75)
    for v in inst["vertices"]:
        p = P[v["id"]]
        ax.scatter([p[0]], [p[1]], s=60 + 500 * v["bell_weight"], color="#eb6834", zorder=4, edgecolor="white", lw=0.8)
        off = p / (np.linalg.norm(p) + 1e-9) * 0.24 if np.linalg.norm(p) > 1e-6 else np.array([0, -0.2])
        ax.text(p[0] + off[0], p[1] + off[1], v["label"], fontsize=7.5, ha="center", va="center", color="#0b0b0b")
    b = inst["betti"]
    ax.set_title(title + f"\nBetti {tuple(b)}; graph loops {inst['graph_beta1']}, filled {inst['filled_loops']}; CF {inst['CF']:.2f}",
                 fontsize=8, loc="left", color="#0b0b0b")
    ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-1.4, 1.9); ax.set_ylim(-1.0, 1.35)
