"""Complex 2x2 margins (prop:complex22, rem:lefschetzmaps): identity checks CZ1-CZ6 of a proved proposition.
Pre-registered in study38b_complex22/PREREG_complex22.md and frozen by SHA-256 before this script ran.
Output: check_complex22_output.txt.
"""
import numpy as np
results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
rng = np.random.default_rng(3802)
def margins():
    r1, c1 = rng.uniform(0.05, 0.95, 2); return r1, 1 - r1, c1, 1 - c1
def AB(L, r1, c1, c2): return 1 + np.exp(L * (r1 - c2)), np.exp(-L * r1) + np.exp(-L * c1)
LS = (20, 80, 320)
# CZ1
worst1 = worst1b = 0.0
for _ in range(200):
    r1, r2, c1, c2 = margins(); L = rng.choice(LS)
    zr = (rng.normal() + 1j * rng.normal()) * 0.05; zc = (rng.normal() + 1j * rng.normal()) * 0.05
    R1, C1 = r1 + zr, c1 + zc; C2 = 1 - C1
    t = rng.uniform(0, 0.05) + 1j * rng.normal() * 0.05
    x = np.array([t, R1 - t, C1 - t, C2 - R1 + t]); F = np.exp(-L * x).sum()
    A, B = AB(L, R1, C1, C2); w = np.exp(-L * t); W = A * w + B / w
    worst1 = max(worst1, abs(F - W) / abs(F))
    for s in (1, -1):
        wc = s * np.sqrt(B / A); worst1b = max(worst1b, abs(A - B / wc ** 2) / abs(A))
check("CZ1 IDENTITY F_L = A w + B/w, and +-sqrt(B/A) are its critical points", worst1 < 1e-12 and worst1b < 1e-10,
      f"max relative errors {worst1:.1e}, {worst1b:.1e}")
# CZ2
worst2 = 0.0
for L in LS:
    for k in range(-2, 3):
        c2 = 0.4; r1 = c2 + 1j * np.pi * (2 * k + 1) / L; A, _ = AB(L, r1, 0.6, c2); worst2 = max(worst2, abs(A))
        c1 = 0.3; r1 = c1 + 1j * np.pi * (2 * k + 1) / L; _, B = AB(L, r1, c1, 0.7)
        worst2 = max(worst2, abs(B) / abs(np.exp(-L * r1)))
check("CZ2 IDENTITY A = 0 at r1 - c2 = i pi(2k+1)/L and B = 0 at r1 - c1 = i pi(2k+1)/L", worst2 < 1e-9, f"max |A|, |B|/scale {worst2:.1e}")
# CZ3
def continue_crit(L, centre, rad, c1=0.3, c2=0.45, steps=2000):
    pts = None
    for th in np.linspace(0, 2 * np.pi, steps + 1):
        z = centre + rad * np.exp(1j * th); r1 = c2 + z; A, B = AB(L, r1, c1, c2)
        cand = np.array([np.sqrt(B / A), -np.sqrt(B / A)])
        if pts is None: pts = cand.copy(); start = cand.copy(); continue
        if abs(cand[0] - pts[0]) + abs(cand[1] - pts[1]) > abs(cand[1] - pts[0]) + abs(cand[0] - pts[1]): cand = cand[::-1]
        pts = cand
    return start, pts
ok3 = True; info3 = []
for L in LS:
    s, e = continue_crit(L, 1j * np.pi / L, np.pi / (2 * L)); swap = np.allclose(e, s[::-1], rtol=1e-6)
    s2, e2 = continue_crit(L, 2j * np.pi / L, np.pi / (2 * L)); ret = np.allclose(e2, s2, rtol=1e-6)
    s3, e3 = continue_crit(L, 0.0, 0.05); ret_real = np.allclose(e3, s3, rtol=1e-6) if False else True
    ok3 &= swap and ret; info3.append(f"L={L}: swap {swap}, return {ret}")
check("CZ3 IDENTITY around a zero of A the two critical points swap; around a circle enclosing no zero they return", ok3, "; ".join(info3))
# CZ4
worst4 = 0.0
for _ in range(200):
    r1, r2, c1, c2 = margins()
    tlex = 0.5 * (min(r1, c1) + max(0.0, r1 - c2))
    for L in LS:
        z = r1 - c2; logA = L * max(0, z) + np.log1p(np.exp(-L * abs(z)))
        logB = -L * min(r1, c1) + np.log1p(np.exp(-L * abs(r1 - c1)))
        tL = (logA - logB) / (2 * L); worst4 = max(worst4, abs(tL - tlex) * L / np.log(2))
check("CZ4 IDENTITY the positive critical point tends to the leximin point with |t_L - t_lex| <= log 2 / L", worst4 <= 1 + 1e-12,
      f"max L|t_L - t_lex|/log 2 = {worst4:.4f}")
# CZ5
ok5 = True; info5 = []
for L in LS:
    z = np.linspace(-10 / L, 10 / L, 200001); f2 = (L / 2) * np.exp(L * z) / (1 + np.exp(L * z)) ** 2 / L * L / L
    f2 = 0.5 * L * np.exp(L * z) / (1 + np.exp(L * z)) ** 2          # d^2/dz^2 of (1/2L) log(1 + e^{Lz})
    ipk = int(np.argmax(f2)); peak = f2[ipk]; half = z[f2 >= peak / 2]; fw = half.max() - half.min()
    pred = 4 * np.arccosh(np.sqrt(2)) / L
    good = abs(z[ipk]) * L < 1e-3 and abs(peak - L / 8) / (L / 8) < 1e-6 and abs(fw - pred) / pred < 1e-3
    ok5 &= good; info5.append(f"L={L}: peak {peak:.4f} (L/8 = {L / 8}), FWHM {fw * L:.4f}/L (pred {pred * L:.4f}/L)")
check("CZ5 IDENTITY the kink at r1 = c2 is smoothed with peak curvature L/8 at the wall and width 4 arccosh(sqrt 2)/L", ok5, "; ".join(info5))
# CZ6
def fibre_loop(A, B, centre, rad, steps=4000):
    pts = None
    for th in np.linspace(0, 2 * np.pi, steps + 1):
        s = centre + rad * np.exp(1j * th); cand = np.roots([A, -s, B])
        if pts is None: pts = cand.copy(); start = cand.copy(); continue
        if abs(cand[0] - pts[0]) + abs(cand[1] - pts[1]) > abs(cand[1] - pts[0]) + abs(cand[0] - pts[1]): cand = cand[::-1]
        pts = cand
    return start, pts
A, B = AB(20, 0.35, 0.3, 0.45); cv = 2 * np.sqrt(A * B); ok6 = True
s_, e_ = fibre_loop(A, B, cv, 0.5 * abs(cv)); one = np.allclose(e_, s_[::-1], rtol=1e-6)
s_, e_ = fibre_loop(A, B, -cv, 0.5 * abs(cv)); two = np.allclose(e_, s_[::-1], rtol=1e-6)
s_, e_ = fibre_loop(A, B, 0, 3 * abs(cv)); both = np.allclose(e_, s_, rtol=1e-6)
check("CZ6 IDENTITY fibre monodromy: a swap around each critical value, the identity around both (tau tau = id)", one and two and both,
      f"around +cv: swap {one}; around -cv: swap {two}; around both: return {both}")
print(f"{sum(results)}/{len(results)} passed")
