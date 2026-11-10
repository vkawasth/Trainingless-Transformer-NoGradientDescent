"""Coherent miscalibration: synchronised deformations, coalition reach and anchor certificates (sec. 'Coherent miscalibration' of the
companion note). New module; it builds on pce/tamper.py and pce/bridgetoy.py and modifies nothing.
  ising(J, field)                 a law on the five-cycle with edge couplings J (5) and fields (5)
  reports(p)                      the five sources' tables under law p
  coalition_path(P, coal, anchors, target, T)   a straight-line coherent path: p_lam = (1 - lam) P + lam p1, where p1 matches the anchors
                                  exactly, has decision value 'target', and moves the coalition's tables as little as possible (L1); the
                                  coalition reports p_lam's tables, everyone else stays at P's
  world_path(J0, J1, field, T)    honest drift: the world itself moves, couplings interpolated linearly; every report follows the truth
  stats(path_reports, anchor_tables, g)   the decision maker's statistics along a path: naive margin, cumulative boundary-directed
                                  movement D_T, the largest per-report step (L1), snapshot residual without and with the anchor, and the set
                                  of reports that moved
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import linprog
from . import tamper as T
from . import bridgetoy as BT

N = 5; TAU = 0.5; A = T.atoms(N)


def ising(J, field):
    w = np.array([np.exp(sum(J[i] * x[i] * x[(i + 1) % 5] for i in range(5)) + np.dot(field, x)) for x in A]); return w / w.sum()


def reports(p): return {s: (s, BT.CTX[s], BT.table(p, BT.CTX[s])) for s in BT.LABELS}


def coalition_endpoint(P, coal, anchors, g, target):
    """min L1 change of the coalition's tables, subject to: a law p1, anchors exact, g(p1) = target"""
    NN = 2 ** N; Ms = []; bs = []
    for c in anchors: M, keys = T.marg(N, c[1]); Ms.append(M); bs.append(np.array([c[2][k] for k in keys]))
    Mc = []; bc = []
    for s in coal: M, keys = T.marg(N, BT.CTX[s]); Mc.append(M); bc.append(M @ P)
    Mc = np.vstack(Mc); bc = np.concatenate(bc); r = len(bc)
    cost = np.concatenate([np.zeros(NN), np.ones(2 * r)])
    Aeq = [np.concatenate([np.ones(NN), np.zeros(2 * r)])]; beq = [1.0]
    for M, b in zip(Ms, bs):
        for i in range(M.shape[0]): Aeq.append(np.concatenate([M[i], np.zeros(2 * r)])); beq.append(b[i])
    Aeq.append(np.concatenate([g, np.zeros(2 * r)])); beq.append(target)
    for i in range(r):
        row = np.concatenate([Mc[i], np.zeros(2 * r)]); row[NN + i] = -1; row[NN + r + i] = 1; Aeq.append(row); beq.append(bc[i])
    res = linprog(cost, A_eq=np.array(Aeq), b_eq=beq, bounds=[(0, None)] * (NN + 2 * r), method="highs")
    return None if res.status else res.x[:NN]


def coalition_path(P, coal, anchors, g, target, T_=20):
    p1 = coalition_endpoint(P, coal, anchors, g, target); hon = reports(P); out = []
    for lam in np.linspace(0, 1, T_ + 1):
        pl = (1 - lam) * P + lam * p1; rep = dict(hon)
        for s in coal: rep[s] = (s, BT.CTX[s], BT.table(pl, BT.CTX[s]))
        out.append(rep)
    return out, p1


def world_path(J0, J1, field, T_=20):
    laws = [ising((1 - lam) * np.asarray(J0) + lam * np.asarray(J1), field) for lam in np.linspace(0, 1, T_ + 1)]
    return [reports(p) for p in laws], laws


def naive_value(cells, g):
    res = T.residual(N, cells); I = T.interval(N, cells, g, tol=res + 1e-9 if res > 1e-9 else 0.0); return 0.5 * (I[0] + I[1]), res


def stats(path, anchor_tables, g, anchor_ctx):
    margins, res_no, res_an, steps = [], [], [], []
    for t, rep in enumerate(path):
        cells = list(rep.values()); v, r0 = naive_value(cells, g); margins.append(v - TAU); res_no.append(r0)
        res_an.append(T.residual(N, cells + [("anchor", anchor_ctx, anchor_tables[t])]))
        if t: steps.append(max(sum(abs(rep[s][2][k] - path[t - 1][s][2][k]) for k in rep[s][2]) for s in rep))
    moved = sorted(s for s in path[0] if sum(abs(path[-1][s][2][k] - path[0][s][2][k]) for k in path[0][s][2]) > 1e-3)
    dT = float(sum(max(0.0, -(margins[t + 1] - margins[t])) for t in range(len(margins) - 1)))
    disp = float(sum(sum(abs(path[-1][s][2][k] - path[0][s][2][k]) for k in path[0][s][2]) for s in path[0]))
    return dict(margin0=margins[0], margin1=margins[-1], D_T=dT, max_step=max(steps), displacement=disp,
                max_res_no_anchor=max(res_no), max_res_anchor=max(res_an), moved=moved)
