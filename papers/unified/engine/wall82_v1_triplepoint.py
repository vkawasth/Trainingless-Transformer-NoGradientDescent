"""Study 82 (AMEND1): Study 81 pipeline output (wall81_bside.py caches, unchanged) with peg-free taut curves moved to the core of their
cylinder; then the Study 53/54 Floer routines and Study 76 signed triangles. Run as STUDY72_T=<t> python3 wall82.py <t>.
Output: study82_path/t<t>/b82.pkl (dims, degrees, ranks, signed products, stars, generator positions, lattice)"""
import os, sys, pickle, itertools
import numpy as np
import wall81_bside as W
t = sys.argv[1]; c, g, h = W.c, W.g, W.h
import signs76 as s76
D = os.path.join(c.HERE, "study82_path", f"t{t}"); os.makedirs(D, exist_ok=True)
def core(tg, T):
    """move a peg-free taut curve to the core of its cylinder; returns new dict and the transverse shift"""
    if any(q is not None for q in tg["pegs"]): return tg, 0.0
    lam = complex(tg["lam"]); u = lam / abs(lam); v = np.array(tg["verts"], complex)
    f = lambda z: (np.conj(u) * z).imag; hper = abs((np.conj(T.wa) * T.wb).imag) / abs(lam)
    s = np.array([(f(p) - f(v[0])) % hper for p in T.pun]); s = s[(s > 1e-12) & (s < hper - 1e-12)]
    mid = (s.min() + s.max() - hper) / 2; out = dict(tg); out["verts"] = list(v + 1j * u * mid); return out, float(mid)
if __name__ == "__main__":
    tc = W.taut_curves(W.transported()); T = W.T; cuts = W.cuts
    new, shifts, words_ok = [], [], True
    for q, tg in enumerate(tc["curves"]):
        nt, sh = core(tg, T); shifts.append(sh)
        w0 = h.canon(cuts.word(c.offset_curve(tg["verts"], tg["pegs"], tg["lam"], W.r0))); w1 = h.canon(cuts.word(c.offset_curve(nt["verts"], nt["pegs"], nt["lam"], W.r0)))
        words_ok &= w0 == w1; new.append(nt)
    cv0 = g.build(new, W.r0); cv = [cv0[q] for q in W.PERM]
    pr = h.floer_chain(cv, T, W.tol); tri = h.triangles_convex(cv, pr, T, W.tol); co = h.cohom_data(pr, tri)
    PAIRS = list(itertools.combinations(range(6), 2))
    stars = s76.star_params(cv, pr); CB = s76.signed_products(cv, pr, T, W.tol, stars)
    out = dict(t=float(t), perm=W.PERM, CLASS=c.CLASS, fits={e: float(f_[0]) for e, f_ in W.fits.items()}, shifts=shifts, words_ok=words_ok,
               dims={k: len(pr[k]["points"]) for k in PAIRS}, degms={k: sorted(set(int(x["D"]) for x in pr[k]["points"])) for k in PAIRS},
               mu1=all(not pr[k]["d"].any() for k in PAIRS), ranks=co["ranks"], CB=CB, stars=stars, wa=T.wa, wb=T.wb,
               pts={k: [complex(x["z"]) for x in pr[k]["points"]] for k in PAIRS}, words=[h.canon(cuts.word(c.offset_curve(n["verts"], n["pegs"], n["lam"], W.r0))) for n in new])
    if t == "-0.1":
        ci3 = pickle.load(open(os.path.join(c.HERE, "study75_cat", "ci3.pkl"), "rb")); cvs = g.build(ci3["curves"], W.r0); prs = h.floer_chain(cvs, T, W.tol)
        out["pts_ci3"] = {k: [complex(x["z"]) for x in prs[k]["points"]] for k in PAIRS}
    pickle.dump(out, open(os.path.join(D, "b82.pkl"), "wb")); print("b82 written", t, "core shifts", np.round(shifts, 5).tolist(), "words unchanged", words_ok)
