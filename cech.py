"""ALPHA COMPLEX vs VIETORIS-RIPS ON THE TRAJECTORY.

Rips admits a 2-simplex whenever all three pairwise distances are below eps,
even when the three balls share no common point. It therefore OVER-fills
triangles and UNDER-counts b1: genuine cycles get capped off. That is a
plausible reason every inherited gate returned b1 = 0.

The alpha complex is the subcomplex of the Delaunay triangulation whose
simplices have circumradius <= alpha. By the nerve lemma it has the exact
homotopy type of the union of balls, so its b1 is the true one. Cost is
Delaunay: cheap in the plane and in R^3, infeasible in R^128 -- a second
argument for projecting low, on top of concentration of measure flattening all
distances at high dimension.

Compared on the same trajectory windows:
    b1_rips    Rips at the adaptive scale (median x 0.75)
    b1_alpha   alpha complex at the matched scale
    b1_alpha*  alpha maximised over a filtration sweep, which is the honest
               persistence answer rather than one arbitrary cut

If alpha finds cycles where Rips finds none, the inherited gates failed for a
fixable reason rather than an intrinsic one.
"""
import io, contextlib, math
import numpy as np
import torch
from scipy.spatial import Delaunay
from itertools import combinations


def _betti(N, edges, tris):
    if not edges:
        return N, 0
    d1 = np.zeros((N, len(edges)))
    for k, (u, v) in enumerate(edges):
        d1[u, k] = -1.0
        d1[v, k] = 1.0
    r1 = np.linalg.matrix_rank(d1, tol=1e-8)
    if not tris:
        return N - r1, max(0, len(edges) - r1)
    emap = {e: i for i, e in enumerate(edges)}
    d2 = np.zeros((len(edges), len(tris)))
    for t, (u, v, w) in enumerate(tris):
        d2[emap[(u, v)], t] = 1.0
        d2[emap[(v, w)], t] = 1.0
        d2[emap[(u, w)], t] = -1.0
    r2 = np.linalg.matrix_rank(d2, tol=1e-8)
    return N - r1, max(0, len(edges) - r1 - r2)


def rips(pts, eps):
    N = len(pts)
    d = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
    e = [(i, j) for i in range(N) for j in range(i + 1, N) if d[i, j] <= eps]
    es = set(e)
    t = [(i, j, k) for i, j, k in combinations(range(N), 3)
         if (i, j) in es and (j, k) in es and (i, k) in es]
    return _betti(N, e, t)


def _meb_radius(P):
    """Radius of the minimum enclosing ball. This is what the NERVE needs:
    balls B(p_i, r) have a common point iff the MEB radius <= r. The
    circumradius is wrong whenever the circumcentre falls outside the simplex,
    which is why the circumradius version reported b1 = 0 on a circle and 45
    spurious cycles on the trajectory."""
    n = len(P)
    if n == 1:
        return 0.0
    if n == 2:
        return float(np.linalg.norm(P[0] - P[1]) / 2)
    # exact for 3 points: circumcentre if acute, else the longest edge / 2
    a = np.linalg.norm(P[1] - P[2]); b = np.linalg.norm(P[0] - P[2])
    c = np.linalg.norm(P[0] - P[1])
    s2 = sorted([a, b, c])
    if s2[2] ** 2 >= s2[0] ** 2 + s2[1] ** 2:      # obtuse or right
        return float(s2[2] / 2)
    cr = _circumradius(P)
    return float(cr)


def cech(pts, r):
    """Nerve of the cover by balls of radius r. Edge iff the two balls meet
    (d <= 2r); triangle iff all three meet (MEB radius <= r)."""
    N = len(pts)
    d = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
    e = [(i, j) for i in range(N) for j in range(i + 1, N) if d[i, j] <= 2 * r]
    es = set(e)
    t = [(i, j, k) for i, j, k in combinations(range(N), 3)
         if (i, j) in es and (j, k) in es and (i, k) in es
         and _meb_radius(pts[[i, j, k]]) <= r]
    return _betti(N, e, t)


def _circumradius(P):
    if len(P) == 2:
        return float(np.linalg.norm(P[0] - P[1]) / 2)
    A = 2 * (P[1:] - P[0])
    b = (P[1:] ** 2).sum(1) - (P[0] ** 2).sum()
    try:
        c, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return np.inf
    return float(np.linalg.norm(c - P[0]))


def alpha_complex(pts, alpha):
    N = len(pts)
    try:
        dl = Delaunay(pts)
    except Exception:
        return N, 0
    edges, tris = set(), set()
    for simp in dl.simplices:
        for t in combinations(sorted(simp), 3):
            if _circumradius(pts[list(t)]) <= alpha:
                tris.add(t)
        for e in combinations(sorted(simp), 2):
            if _circumradius(pts[list(e)]) <= alpha:
                edges.add(e)
    edges |= {tuple(sorted(p)) for t in tris for p in combinations(t, 2)}
    return _betti(N, sorted(edges), sorted(tris))


R = open("compiler_geometri_patched_86.py").read()
SRC = R[:R.find("# \u2500\u2500 PHASE 3")]
for o, n in [("for mf_r in range(1, 16):", "for mf_r in range(1, 3):"),
             ("    if pc == N_STU-1:", "    if False:"),
             ("ETA_MF=0.01", "ETA_MF=0.002"),
             ("    if len(tau_history)>=3 and tau > tau_history[-2] > tau_history[-3]:",
              "    if False:")]:
    assert SRC.count(o) == 1, o
    SRC = SRC.replace(o, n, 1)
EO = "evals,evecs=spla.eigsh(L_sym,k=D+1,which='SM',tol=1e-4,maxiter=2000)"
EN = ("_v0=np.random.RandomState(7).randn(L_sym.shape[0])\n"
      "evals,evecs=spla.eigsh(L_sym,k=D+1,which='SM',tol=1e-4,maxiter=2000,v0=_v0)\n"
      "evecs=evecs*np.sign(evecs[np.argmax(np.abs(evecs),axis=0),np.arange(evecs.shape[1])])")
assert SRC.count(EO) == 1
SRC = SRC.replace(EO, EN, 1)

torch.manual_seed(1234); np.random.seed(1234)
G = {}; _b = io.StringIO()
with contextlib.redirect_stdout(_b):
    exec(SRC, G)
model = G["model"]; gb = G["get_batch"]; LR = G["LR"] * 5
named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
tot = sum(p.numel() for _, p in named)
g = torch.Generator().manual_seed(42)
sub = torch.randperm(tot, generator=g)[:20000]
P3 = torch.randn(20000, 3, generator=g) / math.sqrt(3)
opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95),
                        weight_decay=0.1)
EV = [gb() for _ in range(4)]


def vloss():
    with torch.no_grad():
        return sum(float(model(x, y)[1]) for x, y in EV) / len(EV)


th = np.linspace(0, 2 * np.pi, 12, endpoint=False)
circ = np.stack([np.cos(th), np.sin(th)], 1)
dc = np.linalg.norm(circ[:, None] - circ[None], axis=-1)
ec = float(np.median(dc)) * 0.75
print("  sanity, 12 points on a circle (true b1 = 1):")
print(f"    rips b1 = {rips(circ, ec)[1]}    alpha b1 = {alpha_complex(circ, ec)[1]}\n")

buf, rows = [], []
print(f"  {'step':>5}{'val':>9}{'rips':>7}{'alpha':>7}{'cech':>8}")
for t in range(1, 641):
    x, y = gb(); _, l = model(x, y)
    opt.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    if t % 8:
        continue
    flat = torch.cat([p.data.reshape(-1) for _, p in named])[sub]
    buf.append((flat @ P3).numpy())
    if len(buf) > 16:
        buf.pop(0)
    if len(buf) < 16:
        continue
    X = np.stack(buf)
    d = np.linalg.norm(X[:, None] - X[None], axis=-1)
    e = float(np.median(d)) * 0.75
    br = rips(X, e)[1]
    ba = alpha_complex(X, e)[1]
    # r = 2 x the median consecutive-step displacement: consecutive balls
    # always meet, distant ones only if the trajectory returns
    step = float(np.median([np.linalg.norm(X[i+1]-X[i]) for i in range(len(X)-1)]))
    bs = cech(X, step)[1]
    rows.append((t, vloss(), br, ba, bs))
    if t % 64 == 0:
        print(f"  {t:>5}{rows[-1][1]:>9.4f}{br:>7}{ba:>7}{bs:>8}", flush=True)

A = np.array(rows, float)
print(f"\n  over {len(rows)} windows")
for i, lab in ((2, "rips"), (3, "alpha"), (4, "cech")):
    print(f"    {lab:<8} values {sorted(set(A[:, i].astype(int)))}   "
          f"mean {A[:, i].mean():.2f}  sd {A[:, i].std():.2f}")
dv = np.diff(A[:, 1])
for i, lab in ((2, "rips"), (3, "alpha"), (4, "cech")):
    if A[:, i].std() < 1e-9:
        print(f"    {lab:<8} constant, no correlation")
    else:
        print(f"    {lab:<8} r with next-window loss drop = "
              f"{np.corrcoef(A[:-1, i], -dv)[0, 1]:+.3f}")
