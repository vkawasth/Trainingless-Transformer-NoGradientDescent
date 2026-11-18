"""Study 48 (steps 2-5): the 2x4 row-column wall (FL1-FL6). Pre-registered in study48_2x4/PREREG_study48.md, frozen by SHA-256 before
this code ran. Interval primitives (Arb, Krawczyk) from check_example45.py; braid reading and the Artin comparison from check_leadbraid46.py;
the n-dimensional families, the fold system and the tracker are written here. Checkpointed per item in study48_2x4/ckpt/.
Output: check_fold48_output.txt.
"""
import os, sys, json, runpy, time, math, itertools
from fractions import Fraction as Fr
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); D48 = os.path.join(HERE, "study48_2x4"); CK = os.path.join(D48, "ckpt"); os.makedirs(CK, exist_ok=True)
_argv = sys.argv; sys.argv = ["check_leadbraid46.py"]; M = runpy.run_path(os.path.join(HERE, "check_leadbraid46.py")); sys.argv = _argv
arb, acb, kraw, box, pt, mat_inv_acb, tball, ser, des, sgn = (M[k] for k in ("arb", "acb", "kraw", "box", "pt", "mat_inv_acb", "tball", "ser", "des", "sgn"))
read_word, artin, zrot = M["read_word"], M["artin"], M["zrot"]
OUT = open(os.path.join(HERE, "check_fold48_output.txt"), "a")
def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); OUT.write(s + "\n"); OUT.flush()
def check(name, cond, info=""):
    say(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else "")); return bool(cond)
def ckpt(name, fn):
    p = os.path.join(CK, name + ".json")
    if os.path.exists(p): return json.load(open(p))
    t0 = time.time(); r = fn(); r["seconds"] = round(time.time() - t0); json.dump(r, open(p, "w"), default=str)
    say(f"  [{name}] " + str({k: v for k, v in r.items() if k not in ("steps", "rows")})[:1500]); return r

# ---------------------------------------------------------------- the wall ------------------------------------------------------------------
NC = 4; DIM = NC - 1
CELLS = [(i, j) for i in range(2) for j in range(NC)]
def expo(i, j):
    if j == 0: return [1] * DIM if i == 0 else [-1] * DIM
    v = [0] * DIM; v[j - 1] = -1 if i == 0 else 1; return v
V = [expo(i, j) for i, j in CELLS]; Vf = np.array(V, float)
GQ = [Fr(3, 20), Fr(1, 4), Fr(7, 25), Fr(8, 25)]; FQ = [Fr(3, 10), Fr(7, 10)]; FS = [1, -1]
gf = np.array([float(x) for x in GQ]); DX = np.array([FS[i] * gf[j] for i, j in CELLS])
def x0f(t): return np.array([(float(FQ[i]) + FS[i] * t) * gf[j] for i, j in CELLS])
def Qb(fr): return arb(fr.numerator) / fr.denominator
GA = [Qb(x) for x in GQ]; FA = [Qb(x) for x in FQ]
def x0b(k, t): i, j = CELLS[k]; return (FA[i] + FS[i] * t) * GA[j]
EPS = [(-1) ** (((sum(v) - 1) // 2) % 2) for v in V]                          # i^{|v|} = i * eps
TAU_STAR = 2 * math.log(2 / math.sqrt(27)); VSTAR = 8 / 3 * 3 ** 0.25
def h1f(t): return (0.3 + t) / 4 if t <= 0 else 0.075

# ---------------------------------------------------------------- the fold system on the rho-fixed locus (real) --------------------------------
def R_terms_f(x, t, L): return np.array(EPS) * np.exp(-L * x0f(t) + Vf @ x)
def fold_F(z, L):
    x, t = z[:3], z[3]; E = R_terms_f(x, t, L); H = (Vf.T * E) @ Vf
    return np.array([*(Vf.T @ E), np.linalg.det(H)])
def fold_J(z, L, h=1e-7):
    J = np.zeros((4, 4))
    for c in range(4):
        e = np.zeros(4); e[c] = h * max(1, abs(z[c])); J[:, c] = (fold_F(z + e, L) - fold_F(z - e, L)) / (2 * e[c])
    return J
def det3(H): return (H[0][0] * (H[1][1] * H[2][2] - H[1][2] * H[2][1]) - H[0][1] * (H[1][0] * H[2][2] - H[1][2] * H[2][0]) + H[0][2] * (H[1][0] * H[2][1] - H[1][1] * H[2][0]))
def fold_Fb(z, L):
    x, t = z[:3], z[3]; E = [EPS[k] * (-L * x0b(k, t) + sum(V[k][m] * x[m] for m in range(3))).exp() for k in range(8)]
    H = [[sum(V[k][a] * V[k][b] * E[k] for k in range(8)) for b in range(3)] for a in range(3)]
    return [sum(V[k][m] * E[k] for k in range(8)) for m in range(3)] + [det3(H)]
def fold_Jb(z, L):
    """Exact Jacobian (balls) of the fold system in (x, t)."""
    x, t = z[:3], z[3]; E = [EPS[k] * (-L * x0b(k, t) + sum(V[k][m] * x[m] for m in range(3))).exp() for k in range(8)]
    dt = [-L * FS[CELLS[k][0]] * GA[CELLS[k][1]] for k in range(8)]
    H = [[sum(V[k][a] * V[k][b] * E[k] for k in range(8)) for b in range(3)] for a in range(3)]
    rows = []
    for m in range(3):
        rows.append([sum(V[k][m] * V[k][c] * E[k] for k in range(8)) for c in range(3)] + [sum(V[k][m] * dt[k] * E[k] for k in range(8))])
    last = []
    for c in range(4):
        wts = [V[k][c] if c < 3 else dt[k] for k in range(8)]
        dH = [[sum(V[k][a] * V[k][b] * wts[k] * E[k] for k in range(8)) for b in range(3)] for a in range(3)]
        # d det = sum over rows of det with that row replaced by its derivative
        tot = arb(0)
        for r_ in range(3):
            Hr = [list(row) for row in H]; Hr[r_] = dH[r_]; tot = tot + det3(Hr)
        last.append(tot)
    return rows + [last]
def ustar_row(t): h = (0.3 + t) / 4; return np.array([(0.3 + t) * gf[j] - h for j in range(1, 4)])
def fold_at(L):
    t0 = TAU_STAR / L                                                          # w = w_F - L u*, w_F = log|Y*| + i pi/2, so x = -ln(3)/4 - L u*
    x = np.full(3, -0.25 * math.log(3)) - L * ustar_row(t0)
    z = np.array([*x, t0])
    for _ in range(100):
        st = np.linalg.solve(fold_J(z, L), fold_F(z, L)); z = z - st
        if np.abs(st).max() < 1e-14 * (1 + np.abs(z).max()): break
    Yi = np.linalg.inv(fold_J(z, L)); Y = [[arb(float(Yi[a, b])) for b in range(4)] for a in range(4)]
    xm = [arb(float(v)) for v in z]; cert = None
    for r in (1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8):
        X = [arb(float(v), r * (1 + abs(v))) for v in z]; ok, K = kraw(lambda q: fold_Fb(q, L), lambda q: fold_Jb(q, L), xm, X, Y)
        if ok: cert = (r, X, K); break
    if cert is None: return dict(L=L, certified=False, approx=list(map(float, z)))
    K = cert[2]; x, t = K[:3], K[3]
    E = [EPS[k] * (-L * x0b(k, t) + sum(V[k][m] * x[m] for m in range(3))).exp() for k in range(8)]
    H = [[sum(V[k][a] * V[k][b] * E[k] for k in range(8)) for b in range(3)] for a in range(3)]
    minors = [H[a][c] * H[b][d] - H[a][d] * H[b][c] for (a, b) in itertools.combinations(range(3), 2) for (c, d) in itertools.combinations(range(3), 2)]
    rank2 = any(not m.contains(0) for m in minors)
    # kernel vector: cross product of two rows of H whose 2x2 minor (in some columns) is nonzero -> use rows 0, 1 if possible
    best = None
    for (a, b) in itertools.combinations(range(3), 2):
        kv = [H[a][1] * H[b][2] - H[a][2] * H[b][1], H[a][2] * H[b][0] - H[a][0] * H[b][2], H[a][0] * H[b][1] - H[a][1] * H[b][0]]
        if any(not c.contains(0) for c in kv): best = kv; break
    cubic = sum(E[k] * sum(V[k][m] * best[m] for m in range(3)) ** 3 for k in range(8)) if best else None
    R0 = sum(E); h1 = (arb(3) / 10 + t) / 4; scaled = R0 * (L * h1).exp()                 # value of W is i R0; scaled by e^{L h1}
    return dict(L=L, certified=True, rad=cert[0], t=ser(t), tau=ser(t * L), x=[ser(v) for v in x], rank2=bool(rank2),
                cubic=None if cubic is None else ser(cubic), cubic_nonzero=bool(cubic is not None and not cubic.contains(0)),
                value_im=ser(R0), scaled_value_im=ser(scaled))

# ---------------------------------------------------------------- n-dimensional family and tracker (complex tau paths) ----------------------
class WPath:
    """W at fixed L along tau = p(theta) (complex path), theta in [0, 1]; pf/pb float/ball path, dpf/dpb derivatives."""
    def __init__(self, L, pf, pb, dpf, dpb): self.L, self.pf, self.pb, self.dpf, self.dpb = L, pf, pb, dpf, dpb
    def c(self, th): return np.exp(-self.L * x0f(self.pf(th) / self.L))
    def E(self, w, th): return self.c(th) * np.exp(Vf @ w)
    def F(self, w, th): return Vf.T @ self.E(w, th)
    def J(self, w, th): E = self.E(w, th); return (Vf.T * E) @ Vf
    def s(self, w, th): return self.E(w, th).sum()
    def dwdth(self, w, th):
        E = self.E(w, th); dE = E * (-DX * self.dpf(th))            # d/dtheta of exp(-L x0(tau/L)) = -DX * dtau/dtheta * ...
        return -np.linalg.solve(self.J(w, th), Vf.T @ dE)
    def newton(self, w, th, it=80):
        w = np.array(w, complex)
        for _ in range(it):
            try: st = np.linalg.solve(self.J(w, th), self.F(w, th))
            except np.linalg.LinAlgError: return None
            w = w - st
            if not np.isfinite(w).all(): return None
            if np.abs(st).max() < 1e-15 * (1 + np.abs(w).max()): break
        E = self.E(w, th); return w if np.abs(Vf.T @ E).max() < 1e-11 * np.abs(E).max() else None
    def Eb(self, X, TH):
        T = self.pb(TH); return [(-self.L * x0b(k, T / self.L) + sum(V[k][m] * X[m] for m in range(3))).exp() for k in range(8)]
    def Fb(self, TH): return lambda X: (lambda e: [sum(V[k][m] * e[k] for k in range(8)) for m in range(3)])(self.Eb(X, TH))
    def Jb(self, TH): return lambda X: (lambda e: [[sum(V[k][a] * V[k][b] * e[k] for k in range(8)) for b in range(3)] for a in range(3)])(self.Eb(X, TH))
    def sb(self, X, TH): return sum(self.Eb(X, TH))
def distinct_mod3(A, B):
    for m in range(3):
        d = A[m] - B[m]
        if not d.real.contains(0): return True
        if not (d.imag / (2 * arb.pi())).contains_integer(): return True
    return False
def cert3(fam, w, TH, thm, rads):
    Y = mat_inv_acb(fam.J(w, thm)); xm = [pt(v) for v in w]
    for r in rads:
        X = [box(v, r) for v in w]; ok, K = kraw(fam.Fb(TH), fam.Jb(TH), xm, X, Y)
        if ok: return r, X, K
    return None
RAD = tuple(1e-12 * 1.25 ** k for k in range(110))
def track3(fam, ws, Bprev, h0=0.002, hmax=0.01, hmin=1e-9):
    th = 0.0; ws = [np.array(w) for w in ws]; h = h0; steps = []
    while th < 1.0:
        q = min(th + h, 1.0); m = (th + q) / 2; TH = tball(th, q); out = []; fail = False
        for i, w in enumerate(ws):
            wm = fam.newton(w + fam.dwdth(w, th) * (m - th), m)
            if wm is None: fail = True; break
            sc = 1 + np.abs(wm).max(); sp = np.abs(fam.dwdth(wm, m)).max() * (q - th) / 2
            ends = []
            for te in (th, q):
                we = fam.newton(wm, te); ce = None if we is None else cert3(fam, we, arb(te), te, tuple(r * sc for r in RAD[:40]))
                if ce is None: fail = True; break
                ends.append(ce)
            if fail: break
            rmin = max([1.5 * sp] + [1.3 * e[0] for e in ends] + ([1.3 * float(Bprev[i][0].real.rad())] if Bprev is not None else []))
            c = cert3(fam, wm, TH, m, tuple(r for r in (x * sc for x in RAD) if r > rmin)[:36])
            if c is None: fail = True; break
            r, X, K = c
            if not all(all(X[k].contains(e[1][k]) for k in range(3)) for e in ends) or (Bprev is not None and not all(X[k].contains(Bprev[i][k]) for k in range(3))): fail = True; break
            out.append(dict(wm=wm, X=X, Bb=ends[1][1], s=fam.sb(X, TH), sa=fam.sb(ends[0][2], arb(th)), sb=fam.sb(ends[1][2], arb(q))))
        if not fail:
            Xs = [o["X"] for o in out]; fail = not all(distinct_mod3(Xs[i], Xs[j]) for i in range(len(Xs)) for j in range(i + 1, len(Xs)))
        if fail:
            h /= 2
            if h < hmin: raise RuntimeError(f"tracking failed at theta = {th}")
            continue
        steps.append(dict(pa=th, pb=q, br=out)); Bprev = [o["Bb"] for o in out]; ws = [fam.newton(o["wm"], q) for o in out]; th = q; h = min(h * 1.5, hmax)
    return steps, Bprev, ws
def view(steps, sel=None):
    sel = range(len(steps[0]["br"])) if sel is None else sel
    return [dict(pa=s["pa"], pb=s["pb"], s=[s["br"][i]["s"] for i in sel], sa=[s["br"][i]["sa"] for i in sel], sb=[s["br"][i]["sb"] for i in sel]) for s in steps]
def refine(fam, steps, bad):
    out = []
    for si, st in enumerate(steps):
        if si not in bad: out.append(st); continue
        ws = [fam.newton(o["wm"], st["pa"]) for o in st["br"]]
        Ba = []
        for w in ws:
            c = cert3(fam, w, arb(st["pa"]), st["pa"], tuple(r * (1 + np.abs(w).max()) for r in RAD[:40])); Ba.append(c[1])
        sub_fam = WPath(fam.L, lambda u, a=st["pa"], b=st["pb"]: fam.pf(a + (b - a) * u), lambda U, a=st["pa"], b=st["pb"]: fam.pb(arb(a) + (arb(b) - arb(a)) * U),
                        lambda u, a=st["pa"], b=st["pb"]: fam.dpf(a + (b - a) * u) * (b - a), None)
        sub, _, _ = track3(sub_fam, ws, Ba, h0=1 / 8, hmax=1 / 8)
        for s_ in sub: s_["pa"] = st["pa"] + (st["pb"] - st["pa"]) * s_["pa"]; s_["pb"] = st["pa"] + (st["pb"] - st["pa"]) * s_["pb"]
        out += sub
    return out

def read_word48(steps, zmap, n):
    """read_word of Study 46 with one extension forced by the exact antipodal symmetry (s -> -s): crossings whose times cannot be separated
    are accepted when they involve disjoint pairs of strands that are adjacent at disjoint positions (the generators then commute, so their
    order does not matter)."""
    cur = None; word = []; bad = []
    for si, st in enumerate(steps):
        Z = [zmap(x) for x in st["s"]]; Za = [zmap(x) for x in st["sa"]]; Zb = [zmap(x) for x in st["sb"]]
        if cur is None:
            if any(sgn((Za[i] - Za[j]).real) == 0 for i, j in itertools.combinations(range(n), 2)): bad.append(si); return word, None, bad
            cur = sorted(range(n), key=lambda k: float(Za[k].real.mid()))
        evs = []; ok = True
        for i, j in itertools.combinations(range(n), 2):
            d = Z[i] - Z[j]
            if not d.real.contains(0): continue
            if d.imag.contains(0): ok = False; break
            ra, rb = (Za[i] - Za[j]).real, (Zb[i] - Zb[j]).real
            if sgn(ra) == 0 or sgn(rb) == 0: ok = False; break
            if sgn(ra) != sgn(rb): evs.append((ra / (ra - rb), i, j))
        if not ok: bad.append(si); continue
        evs.sort(key=lambda e: float(e[0].mid())); clusters = []
        for e in evs:
            if clusters and any(e[0].overlaps(f[0]) for f in clusters[-1]): clusters[-1].append(e)
            else: clusters.append([e])
        fail = False
        for cl in clusters:
            strands = [x for _, i, j in cl for x in (i, j)]
            if len(set(strands)) != len(strands): fail = True; break
            pos = []
            for _, i, j in cl:
                pi_, qi_ = cur.index(i), cur.index(j)
                if abs(pi_ - qi_) != 1: fail = True; break
                pos.append(min(pi_, qi_))
            if fail or any(abs(a - b) < 2 for a, b in itertools.combinations(pos, 2)): fail = True; break
            for k in sorted(pos):
                p_, q_ = cur[k], cur[k + 1]; above = (Z[p_] - Z[q_]).imag > 0
                word.append(dict(k=k + 1, e=1 if above else -1, left=p_, right=q_, step=si, p=[st["pa"], st["pb"]]))
            for k in pos: cur[k], cur[k + 1] = cur[k + 1], cur[k]
        if fail: bad.append(si); continue
    return word, cur, bad
def word_of(fam, steps, n=6):
    for _ in range(10):
        w, perm, bad = read_word48(view(steps), zrot, n)
        if not bad: return w, perm, steps
        steps = refine(fam, steps, set(bad))
    raise RuntimeError("braid reading undecided")

if __name__ == "__main__" and "--fl1" in sys.argv:
    for L in (40, 60, 80, 120):
        r = ckpt(f"FL1_L{L}", lambda L=L: fold_at(L))

# ---------------------------------------------------------------- FL2-FL4: detours around the fold ---------------------------------------------
def seg(L, a, b):
    return WPath(L, lambda th: a + (b - a) * th, lambda TH: acb(arb(a) + (arb(b) - arb(a)) * TH), lambda th: (b - a), None)
def arc(L, c, rho, ph0, ph1):
    cb = acb(c); rb = arb(rho); p0, p1 = arb(ph0), arb(ph1)
    return WPath(L, lambda th: c + rho * np.exp(1j * (ph0 + (ph1 - ph0) * th)),
                 lambda TH: cb + rb * (acb(0, 1) * acb(p0 + (p1 - p0) * TH)).exp(),
                 lambda th: rho * 1j * (ph1 - ph0) * np.exp(1j * (ph0 + (ph1 - ph0) * th)), None)
def leading_start(L, tau0=-4.0):
    b = math.exp(tau0 / 2); Ys = np.roots([1, 0, 0, 0, -1, 0, -b]); t0 = tau0 / L; us = ustar_row(t0); fam = seg(L, tau0, tau0 + 1)
    ws = [fam.newton(np.log(complex(Y)) * np.ones(3) - L * us, 0.0) for Y in Ys]
    cs = [cert3(fam, w, arb(0), 0.0, tuple(r * (1 + np.abs(w).max()) for r in RAD[:40])) for w in ws]
    Xs = [c[1] for c in cs]
    assert all(distinct_mod3(Xs[i], Xs[j]) for i in range(6) for j in range(i + 1, 6))
    return ws, Xs
def inv_word(w): return [dict(l, e=-l["e"]) for l in reversed(w)]
def detours(L, rho, tau_f):
    out = {}
    ws, B = leading_start(L)
    sA, BA, wA = track3(seg(L, -4.0, tau_f - rho), ws, B)
    sU, BU, wU = track3(arc(L, tau_f, rho, math.pi, 0.0), wA, BA)              # over the top: Im > 0
    sD, BD, wD = track3(arc(L, tau_f, rho, math.pi, 2 * math.pi), wA, BA)      # underneath: Im < 0
    sB, BB, wB = track3(seg(L, tau_f + rho, 4.0), wU, BU)
    sLp, BL, wL = track3(arc(L, tau_f, rho, 0.0, 2 * math.pi), wU, BU)         # loop based at tau_f + rho, counterclockwise
    words = {}
    for name, fam, st in (("A", seg(L, -4.0, tau_f - rho), sA), ("U", arc(L, tau_f, rho, math.pi, 0.0), sU), ("D", arc(L, tau_f, rho, math.pi, 2 * math.pi), sD),
                          ("B", seg(L, tau_f + rho, 4.0), sB), ("loop", arc(L, tau_f, rho, 0.0, 2 * math.pi), sLp)):
        w, perm, st2 = word_of(fam, st); words[name] = w
        if name == "loop": sLp = st2
        if name == "U": sU = st2
    # loop permutation (end boxes vs start boxes) and pair turns
    perm = [[j for j in range(6) if not distinct_mod3(BU[j], BL[i])] for i in range(6)]
    moved = [i for i in range(6) if perm[i] != [i]]
    pairs = sorted({tuple(sorted((i, perm[i][0]))) for i in moved if len(perm[i]) == 1})
    turns = {}
    for (a_, b_) in pairs:
        tot = arb(0); ok = True
        for st in sLp:
            za = st["br"][a_]["sa"] - st["br"][b_]["sa"]; zb = st["br"][a_]["sb"] - st["br"][b_]["sb"]; Z = st["br"][a_]["s"] - st["br"][b_]["s"]
            if not ((Z * za.conjugate()).real > 0): ok = False; break
            tot = tot + (zb / za).arg()
        turns[str((a_, b_))] = ser(tot / arb.pi()) if ok else None
    # values at the loop base: pairs nearest at the fold, discs, the predicted element
    base = [st for st in sLp][0]; vals = [base["br"][i]["sa"] for i in range(6)]
    nearest = sorted(itertools.combinations(range(6), 2), key=lambda p: float(abs(vals[p[0]] - vals[p[1]]).mid()))[:2]
    discs_free = all(all(float(abs(vals[k] - (vals[a_] + vals[b_]) / 2).lower()) > float(abs(vals[a_] - vals[b_]).upper()) / 2 for k in range(6) if k not in (a_, b_)) for (a_, b_) in pairs) and \
        all(float(abs((vals[a1] + vals[b1]) / 2 - (vals[a2] + vals[b2]) / 2).lower()) > float(abs(vals[a1] - vals[b1]).upper()) / 2 + float(abs(vals[a2] - vals[b2]).upper()) / 2
            for (a1, b1), (a2, b2) in itertools.combinations(pairs, 2))
    out.update(words={k: [(l["k"], l["e"]) for l in v] for k, v in words.items()}, perm=perm, pairs=[list(p) for p in pairs], nearest_pairs=[list(p) for p in nearest],
               turns_over_pi=turns, discs_free=bool(discs_free), base_values=[ser(v) for v in vals])
    # predicted loop element: both pairs rotated about their midpoints by 3 pi e (explicit motion, read with the same reader)
    pc = [complex(float(v.real.mid()), float(v.imag.mid())) for v in vals]
    def motion_word(e):
        N = 3000; frames = []
        for kk in range(N + 1):
            ph = e * 3 * math.pi * kk / N; P = list(pc)
            for (a_, b_) in pairs:
                m = (pc[a_] + pc[b_]) / 2; P[a_] = m + (pc[a_] - m) * np.exp(1j * ph); P[b_] = m + (pc[b_] - m) * np.exp(1j * ph)
            frames.append(P)
        stp = []
        for kk in range(N):
            A_, B_ = frames[kk], frames[kk + 1]
            S = [acb(complex(x).real, complex(x).imag).union(acb(complex(y).real, complex(y).imag)) for x, y in zip(A_, B_)]
            stp.append(dict(pa=kk / N, pb=(kk + 1) / N, s=S, sa=[acb(complex(x).real, complex(x).imag) for x in A_], sb=[acb(complex(y).real, complex(y).imag) for y in B_]))
        w, perm_, bad = read_word48(stp, zrot, 6)
        return None if bad else w
    loopA = artin(words["loop"], 6); pred = {}
    for e in (1, -1):
        mw = motion_word(e); pred[str(e)] = dict(word=None if mw is None else [(l["k"], l["e"]) for l in mw], equal=bool(mw is not None and artin(mw, 6) == loopA))
    out["predicted"] = pred
    lhs = artin(inv_word(words["A"] + words["U"] + words["B"]) + words["A"] + words["D"] + words["B"], 6)
    rhs = artin(inv_word(words["B"]) + words["loop"] + words["B"], 6)
    out["quotient_equals_transported_loop"] = bool(lhs == rhs)
    out["above"] = [(l["k"], l["e"]) for l in words["A"] + words["U"] + words["B"]]; out["below"] = [(l["k"], l["e"]) for l in words["A"] + words["D"] + words["B"]]
    return out
if __name__ == "__main__" and "--fl24" in sys.argv:
    for L in (40, 80):
        tf = float(des(json.load(open(os.path.join(CK, f"FL1_L{L}.json")))["tau"]).mid())
        for rho in (0.5, 0.25):
            ckpt(f"FL2_L{L}_rho{rho}", lambda L=L, rho=rho, tf=tf: detours(L, rho, tf))

# ---------------------------------------------------------------- FL6: capture (floats; a record) -----------------------------------------------
def _a(w, t, L): return -L * x0f(t) + Vf @ w
def newton_s(w, t, L, it=100):
    """Newton for grad W = 0 with the terms rescaled by their largest modulus (no overflow)."""
    w = np.array(w, complex)
    for _ in range(it):
        a = _a(w, t, L); E = np.exp(a - a.real.max()); F = Vf.T @ E; J = (Vf.T * E) @ Vf
        try: st = np.linalg.solve(J, F)
        except np.linalg.LinAlgError: return None
        w = w - st
        if not np.isfinite(w).all(): return None
        if np.abs(st).max() < 1e-13 * (1 + np.abs(w).max()): break
    a = _a(w, t, L); E = np.exp(a - a.real.max())
    return w if np.abs(Vf.T @ E).max() < 1e-9 * np.abs(E).max() else None
def logabs_s(w, t, L): a = _a(w, t, L); m = a.real.max(); return m + math.log(abs(np.exp(a - m).sum()) + 1e-300)
def wnorm3(w): return w.real + 1j * ((w.imag + np.pi) % (2 * np.pi) - np.pi)
def all12(t, L, seed=4801, starts=6000):
    rng = np.random.default_rng(seed); found = []
    for _ in range(starts):
        w = newton_s(-L * rng.uniform(-0.8, 0.8, 3) + 1j * rng.uniform(-np.pi, np.pi, 3), t, L)
        if w is None: continue
        w = wnorm3(w)
        if not any(np.abs(w.real - q.real).max() < 1e-7 * (1 + np.abs(w).max()) and np.abs(np.exp(1j * (w.imag - q.imag)) - 1).max() < 1e-7 for q in found): found.append(w)
        if len(found) == 12: break
    return found
def follow(ws, t0, t1, L, n=40):
    for k in range(1, n + 1):
        t = t0 + (t1 - t0) * k / n; ws = [newton_s(w, t, L) for w in ws]
        if any(w is None for w in ws): return None
    return ws
def capture_L(L):
    taus = np.linspace(-4, 4, 161); t0 = taus[0] / L; ws = all12(t0, L)
    if len(ws) != 12: return dict(L=L, n_found=len(ws))
    wm = [np.log(complex(Y)) * np.ones(3) - L * ustar_row(t0) for Y in np.roots([1, 0, 0, 0, -1, 0, -math.exp(-2)])]
    lead = sorted({int(np.argmin([np.abs(np.exp(1j * (w.imag - m.imag)) - 1).max() + np.abs(w.real - m.real).max() for w in ws])) for m in wm})
    entered = []; prev = t0
    for tau in taus:
        t = tau / L
        if tau > taus[0]:
            ws = follow(ws, prev, t, L)
            if ws is None: return dict(L=L, n_found=12, lead=lead, failed_at=float(tau))
        prev = t; h = h1f(t); ls = [L * h + logabs_s(w, t, L) for w in ws]               # log of the scaled modulus
        mx_lead = max(ls[i] for i in lead); far = [ls[i] for i in range(12) if i not in lead]
        if min(far) < mx_lead: entered.append(float(tau))
    return dict(L=L, n_found=12, lead=lead, entered_at=entered[:5], n_entered=len(entered))
def valuation_crossing(L=400):
    ts = np.linspace(-0.15, 0.15, 301); i0 = 150; ws = all12(ts[i0], L, starts=20000)
    if len(ws) != 12: return dict(L=L, n_found=len(ws))
    wm = [np.log(complex(Y)) * np.ones(3) - L * ustar_row(0.0) for Y in np.roots([1, 0, 0, 0, -1, 0, -1])]
    lead = sorted({int(np.argmin([np.abs(np.exp(1j * (w.imag - m.imag)) - 1).max() + np.abs(w.real - m.real).max() for w in ws])) for m in wm})
    gap = {}
    for direction in (1, -1):
        cur = ws; prev = ts[i0]; rng = range(i0, len(ts)) if direction == 1 else range(i0, -1, -1)
        for i in rng:
            t = ts[i]
            if i != i0:
                cur = follow(cur, prev, t, L, n=20)
                if cur is None: break
            prev = t; vals = [-logabs_s(w, t, L) / L for w in cur]
            gap[float(t)] = max(vals[k] for k in range(12) if k not in lead) - h1f(t)
    cross = sorted([t for t in gap if gap[t] >= 0], key=abs)
    return dict(L=L, lead=lead, t_c=(cross[0] if cross else None), gap_at_0=gap.get(0.0), n_t=len(gap))
if __name__ == "__main__" and "--fl6" in sys.argv:
    for L in range(10, 125, 5): ckpt(f"FL6_L{L}", lambda L=L: capture_L(L))
    ckpt("FL6_valuation", lambda: valuation_crossing(400))

def evaluate():
    say(f"==== evaluation {time.strftime('%Y-%m-%d %H:%M:%S')}")
    F1 = {L: json.load(open(os.path.join(CK, f"FL1_L{L}.json"))) for L in (40, 60, 80, 120)}
    d = [abs(float(des(F1[L]["tau"]).mid()) - TAU_STAR) for L in F1]; v120 = abs(float(des(F1[120]["scaled_value_im"]).mid()))
    check("FL1 HYPOTHESIS fold persistence: certified real fold with t real, rank 2, nonzero cubic, at every L; convergence to tau* and to 3.5095 i",
          all(r["certified"] and r["rank2"] and r["cubic_nonzero"] for r in F1.values()) and all(a > b for a, b in zip(d, d[1:])) and d[-1] < 0.05 and abs(v120 - VSTAR) < 0.05 * VSTAR,
          "; ".join(f"L = {L}: tau_f = {float(des(r['tau']).mid()):.6f}, scaled value {float(des(r['scaled_value_im']).mid()):+.5f} i, cubic {r['cubic'][:12]}" for L, r in F1.items()) + f"; tau* = {TAU_STAR:.6f}, |v*| = {VSTAR:.5f}")
    F2 = {(L, rho): json.load(open(os.path.join(CK, f"FL2_L{L}_rho{rho}.json"))) for L in (40, 80) for rho in (0.5, 0.25)}
    def turns(r): return [float(des(v).mid()) if v else None for v in r["turns_over_pi"].values()]
    ok2 = all(len(r["pairs"]) == 2 and sorted(map(tuple, r["pairs"])) == sorted(map(tuple, r["nearest_pairs"])) and all(t is not None and abs(abs(t) - 3) < 1e-6 for t in turns(r))
              and len({round(t) for t in turns(r)}) == 1 for (L, rho), r in F2.items() if rho == 0.5)
    check("FL2 HYPOTHESIS two separate local events: loop permutation = two transpositions (the nearest pairs at +-v_f), each pair turning +-3 pi, same sign",
          ok2, "; ".join(f"L = {L}, rho = {rho}: pairs {r['pairs']}, turns/pi {[round(t, 6) for t in turns(r)]}" for (L, rho), r in F2.items()))
    ok3 = all(r["discs_free"] and r["quotient_equals_transported_loop"] and any(v["equal"] for v in r["predicted"].values()) for (L, rho), r in F2.items() if rho == 0.5)
    check("FL3 HYPOTHESIS detour monodromy: loop braid = H_1^{3e} H_2^{3e}; beta(above)^-1 beta(below) = transported loop braid", ok3,
          "; ".join(f"L = {L}, rho = {rho}: loop {r['words']['loop']}, matches e = {[e for e, v in r['predicted'].items() if v['equal']]}, quotient ok {r['quotient_equals_transported_loop']}, discs free {r['discs_free']}" for (L, rho), r in F2.items()))
    def el(w): return artin([dict(k=k, e=e) for k, e in w], 6)
    ok4 = all(el(F2[(L, 0.5)]["above"]) == el(F2[(L, 0.25)]["above"]) and el(F2[(L, 0.5)]["below"]) == el(F2[(L, 0.25)]["below"]) for L in (40, 80))
    check("FL4 HYPOTHESIS stability under a smaller detour radius (as elements of B_6)", ok4,
          "; ".join(f"L = {L}: above {F2[(L, 0.5)]['above']}, below {F2[(L, 0.5)]['below']}" for L in (40, 80)))
    check("FL5 RECORD local Picard-Lefschetz data at each fold: A2 germ (splitting lemma, from FL1), vanishing 2-spheres meeting with intersection +-1; derived, not computed", True,
          "global classes in H_2 of the fibre not computed")
    caps = {L: json.load(open(os.path.join(CK, f"FL6_L{L}.json"))) for L in range(10, 125, 5)}; vc = json.load(open(os.path.join(CK, "FL6_valuation.json")))
    clean = [L for L, r in caps.items() if r.get("n_entered") == 0]; entered = {L: r["entered_at"][0] for L, r in caps.items() if r.get("n_entered")}
    Lcap = min(L for L in clean if all(caps[M].get("n_entered") == 0 for M in caps if M >= L)) if clean else None
    check("FL6 RECORD capture: first L from which no far value enters the leading range on the window; valuation crossing", True,
          f"float following failed for L <= 35 ({[L for L, r in caps.items() if 'failed_at' in r]}); entries (first tau): {entered}; no entry for L >= {Lcap}; "
          f"valuation crossing at L = 400: t_c = {vc.get('t_c')} (gap at t = 0: {vc.get('gap_at_0'):.4f}), 4/|t_c| = {4 / abs(vc['t_c']) if vc.get('t_c') else None}")
if __name__ == "__main__" and "--eval" in sys.argv: evaluate()
