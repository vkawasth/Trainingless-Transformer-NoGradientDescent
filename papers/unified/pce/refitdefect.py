"""The transport/refitting defect of a coarsening. New module; it modifies nothing (uses pce/kerneltransport.py).

Fine space X in {a,b,c,d} x Y in {0,1}; coarsening g: X -> X' in {A, B}. Fine data J (tables on fine contexts); coarse data J' (the
coarse-measurable tables read on the coarse space). For a starting law Q (an updater's carried law):
  R  = Pi_J(Q)            refit on the fine space        g_* R   = 'refit, then coarsen'
  R' = Pi_J'(g_* Q)       refit on the coarse space      R'      = 'coarsen, then refit'
  defect(Q) = TV(g_* R, R')
  parts(Q) returns D(g_*R || R'), Delta_fine = D(R || Q) - D(R' || g_*Q), and within = E_{g_*R} D(R(.|c) || Q(.|c)), which satisfy
  D(g_*R || R') = Delta_fine - within (an identity, checked in check_refitdefect.py)
"""
from __future__ import annotations
import numpy as np
from . import kerneltransport as KT

M = KT.merge()


def kl(P, Q): P, Q = np.asarray(P), np.asarray(Q); m = P > 0; return float(np.sum(P[m] * np.log(P[m] / Q[m])))


def within(R, Q):
    """E_{g_*R} D(R(.|c, y) || Q(.|c, y)): the fine-given-coarse divergence, coarse cell = (class, y)"""
    tot = 0.0
    for j, (c, y) in enumerate(KT.COARSE):
        idx = [i for i, (x, yy) in enumerate(KT.FINE) if KT.cls(x) == c and yy == y]
        rc = R[idx].sum(); qc = Q[idx].sum()
        if rc > 0: tot += rc * kl(R[idx] / rc, Q[idx] / qc)
    return tot


def parts(Q, fine_tables, coarse_tables):
    R = KT.refit(Q, fine_tables, KT.FINE); Rp = KT.refit(Q @ M, coarse_tables, KT.COARSE); gR = R @ M
    return dict(R=R, Rp=Rp, gR=gR, defect=KT.tv(gR, Rp), kl=kl(gR, Rp), delta_fine=kl(R, Q) - kl(Rp, Q @ M), within=within(R, Q))
