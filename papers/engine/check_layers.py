"""check_layers.py -- layered restriction structure for corners (companion "A tangent structure for injected data", Section "Layers").
Writes lab/layers.json.

  LY1  strata: every relatively open face of Delta_3 and Delta_4, taken as its own object, has a universal vertical lift
  LY2  the corner failure is a change of layer: at every face point the fibre of T(p) over 0 is the union over entered faces G of
       {u with G(x, u) = G} x (cone of G), exactly (enumeration on Delta_3 and Delta_4)
  LY3  the layered vertical bundle is closed under addition: w in C_G(u), w' in C_G(u') imply w + w' in C_G(u + u')
  LY4  resolution at a coincidence: downstairs |T rhat| grows like 1/|v_2|; lifted to the parameters (w, a_1, a_2) the same estimator
       is r(w) = t - 1/t, with a bounded derivative and none in the a-directions
  LY5  resolution at the merge: two families of truths entering the same product law (rho -> 0) with different weights keep different
       constant values of rhat, so rhat has no limit at the merge downstairs; in the blown-up coordinates (mu, rho, theta, t) it is the
       smooth coordinate r = t - 1/t
  LY6  multi-map: the explanation correspondence of the triangle, with c_01 scaled by lam, is a segment in y = log t whose endpoints are
       piecewise linear in log lam (slopes +-1/2); it shrinks to a point at lam* and is empty below: its graph has a corner (a fold) at the edge of
       the domain, with a wedge as tangent cone
  LY7  branches: on two disjoint edges the relative-sign sheet shrinks (Chebyshev radius -> 0) and dies at w* = 0.6231, while the
       true sheet persists: a branch of the multi-map ends at a positivity wall
  LY8  the pipeline's domain {M01 M02 M12 > 0} is not convex: two laws in it whose mixtures leave it for a range of shares, so a pooling
       statement needs the restriction criterion along the whole chord
"""
import itertools, json, os, sys
import numpy as np
from scipy.optimize import brentq
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import layers as LY
from pce import tangent as TG
from pce.tangent import HD
from pce import injection as IJ
from pce import segment_decision as SD
from pce.lab import law

rng = np.random.default_rng(20261029); results = []; OUT = {}
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
def faces(n):
    for k in range(1, n + 1):
        for supp in itertools.combinations(range(n), k):
            x = np.zeros(n); x[list(supp)] = 1.0 / k; yield supp, x
grid = lambda n: [np.array(z, float) for z in itertools.product((-1, 0, 1), repeat=n) if sum(z) == 0]

# LY1, LY2, LY3
ok1 = ok2 = ok3 = True; table = {}
for n in (3, 4):
    G = grid(n)
    for supp, x in faces(n):
        u_ok, nf = LY.open_face_universal(x, G, G); ok1 &= u_ok
        fib, pred, layers = LY.fibre_by_layer(x, G, G); ok2 &= fib == pred; table[f"D{n} {supp}"] = {str(k): v for k, v in layers.items()}
        for _ in range(200):
            us = [u for u in G if TG.feasible_cone(x, u)]; u, u2 = us[rng.integers(len(us))], us[rng.integers(len(us))]
            Gu, Gu2, Gs = LY.entered_face(x, u), LY.entered_face(x, u2), LY.entered_face(x, u + u2)
            ws = [w for w in G if LY.face_cone_member(Gu, w, n)]; ws2 = [w for w in G if LY.face_cone_member(Gu2, w, n)]
            w, w2 = ws[rng.integers(len(ws))], ws2[rng.integers(len(ws2))]; ok3 &= LY.face_cone_member(Gs, w + w2, n)
OUT["layers"] = table
check("LY1 every relatively open face of Delta_3, Delta_4 is a tangent category on its own (universal lift)", ok1)
check("LY2 the corner failure is a change of layer: the fibre over 0 is the union over entered faces G of {u: G(x,u) = G} x cone(G)", ok2,
      f"e.g. edge of Delta_3: {table['D3 (0, 1)']}")
check("LY3 the layered vertical bundle is closed under addition (C_G grows with G, and G(x, u + u') contains G(x, u), G(x, u'))", ok3)

# LY4
a1 = np.array([.5, .1, .4]); a2b = np.array([.05, .5, .1]); w = 0.6; rows = []
for tv in (0.3, 0.1, 0.03, 0.01):
    a2 = a2b.copy(); a2[2] = a1[2] - tv * (a1[2] - a2b[2]); P = IJ.as_vec(law(w, a1, a2)); v2 = np.sqrt(w * (1 - w)) * (a1[2] - a2[2])
    down = max(abs(TG.rhat_hd(HD.var(P, [e - P])).c[1]) for e in np.eye(8))
    th = np.r_[w, a1, a2]; h = 1e-6
    rf = lambda z: SD.from_tables(SD.exact_tables(law(z[0], z[1:4], z[4:7])))[2]
    grad = np.array([(rf(th + h * e) - rf(th - h * e)) / (2 * h) for e in np.eye(7)])
    t = np.sqrt((1 - w) / w); drdw = (lambda ww: np.sqrt((1 - ww) / ww) - 1 / np.sqrt((1 - ww) / ww))
    rows.append(dict(v2=float(v2), down=float(down), up_norm=float(np.linalg.norm(grad)), up_w=float(grad[0]), up_a=float(np.abs(grad[1:]).max()),
                     drdw=float((drdw(w + h) - drdw(w - h)) / (2 * h)), rhat=float(rf(th)), r=float(t - 1 / t)))
OUT["coincidence"] = rows
check("LY4 lifting resolves the coincidence: |T rhat| ~ 1/|v_2| downstairs; upstairs rhat = r(w), derivative bounded and zero in the a-directions",
      rows[-1]["down"] > 20 * rows[0]["down"] and max(abs(r_["up_w"] - r_["drdw"]) for r_ in rows) < 1e-5 and max(r_["up_a"] for r_ in rows) < 1e-5
      and max(abs(r_["rhat"] - r_["r"]) for r_ in rows) < 1e-9, f"downstairs {[round(r_['down'], 1) for r_ in rows]}; upstairs dr/dw {rows[0]['up_w']:.3f}")

# LY5
mu = np.array([.32, .26, .28]); th0 = np.array([0.6, -0.5, 0.45]); th0 /= np.linalg.norm(th0); fam = {}
for wv in (0.6, 0.3):
    t = np.sqrt((1 - wv) / wv); vals = []
    for rho in (0.1, 0.01, 0.001):
        v = rho * th0; P = law(wv, mu + v * t, mu - v / t); vals.append(float(SD.from_tables(SD.exact_tables(P))[2]))
    fam[wv] = dict(t=float(t), r=float(t - 1 / t), rhat=vals)
OUT["merge"] = {str(k): v for k, v in fam.items()}
same_limit_law = np.allclose(law(0.6, mu, mu), law(0.3, mu, mu))
check("LY5 the merge is a cone point: two families entering the same product law keep rhat = r(w) (no limit downstairs); upstairs r is a smooth coordinate",
      same_limit_law and all(np.allclose(f["rhat"], f["r"], atol=1e-8) for f in fam.values()) and abs(fam[0.6]["r"] - fam[0.3]["r"]) > 0.5,
      "; ".join(f"w = {k}: rhat {[round(x, 4) for x in f['rhat']]} = r {f['r']:.4f}" for k, f in fam.items()))

# LY6
lams = np.exp(np.linspace(np.log(1.0), np.log(0.2), 161)); seg = [LY.triangle_family(l) for l in lams]
alive = [(l, s) for l, s in zip(lams, seg) if s is not None]; lam_star = min(l for l, s in alive)
lam_star = brentq(lambda l: (lambda s: (s[2] - s[1]) if s else -1e-3)(LY.triangle_family(l)), lams[[k for k, s in enumerate(seg) if s is None][0]], lam_star)
s_star = LY.triangle_family(lam_star * (1 + 1e-9))
ll = np.log([l for l, _ in alive]); lo_ = np.array([s[1] for _, s in alive]); hi_ = np.array([s[2] for _, s in alive])
sl_lo = np.round(np.diff(lo_) / np.diff(ll), 6); sl_hi = np.round(np.diff(hi_) / np.diff(ll), 6)
main = lambda x: sorted({v for v in x if abs(abs(v) - 0.5) < 1e-6 or abs(v) < 1e-6}); straddle = lambda x: sum(1 for v in x if not (abs(abs(v) - 0.5) < 1e-6 or abs(v) < 1e-6))
slopes_lo, slopes_hi = np.array(main(sl_lo)), np.array(main(sl_hi))
OUT["fold"] = dict(lam_star=float(lam_star), y_star=s_star[1] if s_star else None, slopes_lo=slopes_lo.tolist(), slopes_hi=slopes_hi.tolist(),
                   curve=[[float(l), s[1], s[2]] for l, s in alive])
pw = straddle(sl_lo) <= 2 and straddle(sl_hi) <= 2
check("LY6 the triangle's explanation segment has endpoints piecewise linear in log lam (slopes +-1/2, one kink), shrinks to a point at lam* and is empty below: a corner of the graph",
      pw and s_star is not None and abs(s_star[2] - s_star[1]) < 1e-6 and LY.triangle_family(lam_star * 0.99) is None,
      f"lam* = {lam_star:.4f}; slopes {slopes_lo.tolist()} / {slopes_hi.tolist()}")

# LY7
ws = np.linspace(0.45, 0.9, 46); rad = [LY.two_edge_sheet(wv) for wv in ws]
w_star = brentq(lambda wv: LY.two_edge_sheet(wv)[0] if LY.two_edge_sheet(wv)[0] > -0.5 else -1e-4, 0.6, 0.7)
OUT["branch"] = dict(w_star=float(w_star), w=ws.tolist(), radius_rel=[r[0] for r in rad], radius_true=[r[1] for r in rad])
check("LY7 a branch ends at a positivity wall: the relative-sign sheet of two disjoint edges shrinks to nothing at w* while the true sheet persists",
      abs(w_star - 0.6231) < 1e-3 and all(r[1] > 0.1 for r in rad) and rad[0][0] > 0.1 and rad[-1][0] < 0, f"w* = {w_star:.4f}")

# LY8
hole = None
for _ in range(40000):
    P = rng.dirichlet(np.ones(8)); Q = rng.dirichlet(np.ones(8))
    if LY.pipeline_domain(IJ.as_law(P, 3)) > 1e-4 and LY.pipeline_domain(IJ.as_law(Q, 3)) > 1e-4:
        es = np.linspace(0, 1, 401); d = np.array([LY.pipeline_domain(IJ.as_law((1 - e) * P + e * Q, 3)) for e in es])
        if d.min() < 0: hole = (P, Q, float(es[d < 0].min()), float(es[d < 0].max())); break
OUT["hole"] = dict(P=hole[0].tolist(), Q=hole[1].tolist(), shares=[hole[2], hole[3]]) if hole else None
check("LY8 the pipeline's domain is not convex: two laws inside it whose mixtures leave it for a range of shares", hole is not None,
      f"shares [{hole[2]:.3f}, {hole[3]:.3f}]" if hole else "")

json.dump(OUT, open(os.path.join(HERE, "..", "lab", "layers.json"), "w"), indent=1, default=float)
print(f"{sum(results)}/{len(results)} passed")
