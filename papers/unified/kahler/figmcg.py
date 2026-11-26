"""Figures for the section on mapping class groups and the groups that drive the thimbles (sec:mcg).
figmcg_odds.png    (exact, 2x3): the positive real thimble in log coordinates and its IPF image, the credal fibre interior
figmcg_strands.png (calibration data, 3x4, Study 91 seed 9101): real critical values across margin ties
figmcg_fibre.png   (Study 86 cached representatives, 2x3): a colliding (disjoint) pair and a pair meeting once on the fibre
figmcg_hurwitz.png, figmcg_groups.png (schematic)"""
import os, sys, pickle, glob
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
HERE = os.path.dirname(os.path.abspath(__file__)); ENG = os.path.join(HERE, "..", "engine"); sys.path.insert(0, ENG)
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
INK, MUTED, C1, C2, C3, C4 = "#1f2937", "#6b7280", "#1D4ED8", "#C2185B", "#2BA58E", "#E07B2E"
# ------------------------------------------------------------------ 1. odds ratios, positive thimble, credal fibre (2x3)
def fig_odds():
    import mirror83 as M
    from scipy.optimize import minimize
    f, g, L = np.array([0.2, 0.8]), np.array([0.2, 0.35, 0.45]), 20.0
    V = np.array(M.V, float); CELL = M.CELL; a = M.coeffs(f, g, L)
    W = lambda w: float((a * np.exp(V @ w)).sum())
    def ipf(w, it=500):
        z = np.zeros((2, 3))
        for k, (i, j) in enumerate(CELL): z[i, j] = np.exp(V[k] @ w)
        x = z.copy()
        for _ in range(it): x *= (f / x.sum(1))[:, None]; x *= (g / x.sum(0))[None, :]
        return x
    ws = minimize(W, np.zeros(2), method="BFGS", options={"gtol": 1e-12}).x
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 4.3))
    R = 4.0; gx = np.linspace(-R, R, 241); X, Y = np.meshgrid(gx, gx); Z = np.vectorize(lambda p, q: W(np.array([p, q])))(X, Y)
    ax[0].contour(X, Y, np.log(Z), levels=14, colors=MUTED, linewidths=0.6)
    lines = np.linspace(-3, 3, 7); cols = plt.cm.viridis(np.linspace(0.1, 0.9, len(lines)))
    for c, s in zip(cols, lines):
        ax[0].plot([s, s], [-R, R], color=c, lw=1.2); ax[0].plot([-R, R], [s, s], color=c, lw=1.2, ls="--")
    ax[0].plot(*ws, "o", color=C2, ms=7, zorder=5); ax[0].plot(0, 0, "s", color=C1, ms=6, zorder=5)
    ax[0].set_title("positive real thimble $\\mathbb{R}_{>0}^2$ in log coordinates\n(level sets of $W$; one point = one odds-ratio pattern)")
    ax[0].set_xlabel("$\\log y_1$"); ax[0].set_ylabel("$\\log y_2$"); ax[0].set_aspect("equal")
    # fibre polygon in (x00, x01)
    import itertools
    pts = []
    for x00 in np.linspace(0, f[0], 801):
        lo = max(0, f[0] - x00 - g[2]); hi = min(g[1], f[0] - x00)
        if x00 <= g[0] and lo <= hi: pts.append((x00, lo, hi))
    P = np.array(pts); poly = np.vstack([np.c_[P[:, 0], P[:, 1]], np.c_[P[::-1, 0], P[::-1, 2]]])
    ax[1].fill(poly[:, 0], poly[:, 1], color="#eef2ff", ec=INK, lw=1)
    tt = np.linspace(-6, 6, 241)
    for c, s in zip(cols, lines):
        A = np.array([ipf(np.array([s, u]))[0, :2] for u in tt]); B = np.array([ipf(np.array([u, s]))[0, :2] for u in tt])
        ax[1].plot(A[:, 0], A[:, 1], color=c, lw=1.2); ax[1].plot(B[:, 0], B[:, 1], color=c, lw=1.2, ls="--")
    xs, x0 = ipf(ws), ipf(np.zeros(2))
    ax[1].plot(xs[0, 0], xs[0, 1], "o", color=C2, ms=7, zorder=5, label="image of the critical point of $W$")
    ax[1].plot(x0[0, 0], x0[0, 1], "s", color=C1, ms=6, zorder=5, label="odds ratios all 1: independence table $f_ig_j$")
    ax[1].set_title("its IPF image: the interior of the credal fibre\n(tables with margins $f=(0.2,0.8)$, $g=(0.2,0.35,0.45)$)")
    ax[1].set_xlabel("$x_{11}$ (seeded, above)"); ax[1].set_ylabel("$x_{12}$ (seeded, normal)"); ax[1].legend(loc="upper right", fontsize=7.5, frameon=False)
    ax[1].set_aspect("equal"); fig.tight_layout(); fig.savefig(os.path.join(HERE, "figmcg_odds.png"), dpi=200); plt.close(fig)
    return ws, xs
# ------------------------------------------------------------------ 2. 3x4 strands across ties (calibration walk)
def fig_strands():
    import lib91 as Q
    d = pickle.load(open(os.path.join(ENG, "study91_3x4", "calib", "walk_cal_20.pkl"), "rb")); T, V = d["T"], np.asarray(d["V"]).astype(complex)
    Fw, Gw = Q.gen(9101)[0]; ties = [(k + t, nm, p) for k in range(13) for t, nm, p in Q.ties_on(Fw[k], Gw[k], Fw[k + 1], Gw[k + 1])]
    lo, hi = 5.2, 5.9; sel = (T >= lo) & (T <= hi); Vs = V[sel]; Ts = T[sel]; rl = np.abs(Vs.imag) < 1e-8 * (1 + np.abs(Vs))
    fig, ax = plt.subplots(figsize=(9.6, 4.2))
    for q in range(96):
        y = np.where(rl[:, q], np.arcsinh(Vs[:, q].real), np.nan)
        if np.isfinite(y).any(): ax.plot(Ts, y, lw=0.7, color=plt.cm.tab20(q % 20))
    for t, nm, p in ties:
        if lo <= t <= hi: ax.axvline(t, color=INK, lw=0.8, ls=":"); ax.text(t, ax.get_ylim()[1] if False else 4.6, f"${'g' if nm == 'g' else 'f'}_{p[0]+1}={'g' if nm == 'g' else 'f'}_{p[1]+1}$", ha="center", fontsize=8)
    for e in d["ev"]:
        if lo <= e["t"] <= hi and e["kind"] == "collision":
            onT = min(abs(e["t"] - t) for t, _, _ in ties) < 1e-6; v = np.arcsinh(np.interp(e["t"], Ts, Vs[:, e["pair"][0]].real))
            ax.plot(e["t"], v, "o" if onT else "x", color=C2 if onT else C4, ms=3.5 if onT else 4, zorder=4)
    ax.set_xlim(lo, hi); ax.set_ylim(-4.2, 5.0); ax.set_xlabel("week of the season (path parameter $t$)"); ax.set_ylabel("real critical values, asinh scale")
    ax.set_title("3x4 strands (calibration path, seed 9101, L = 20): crossings at margin ties (dots) and off the ties (crosses)")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "figmcg_strands.png"), dpi=200); plt.close(fig)
# ------------------------------------------------------------------ 3. the fibre: a colliding pair and a pair meeting once (Study 86 cache)
def fig_fibre():
    tag = "8502_A3_L20_s4.729379"; os.environ["STUDY83_STATE"] = os.path.join(ENG, "study86_floer", "states", tag + ".pkl")
    import check_cat51 as c
    ST = pickle.load(open(os.environ["STUDY83_STATE"], "rb")); c.CK = os.path.join(ENG, "study86_floer", "ckpt_t86_" + tag); c.OUT = open(os.devnull, "w")
    c.CLASS = [tuple(int(v) for v in x) for x in ST["classes"]]
    s4 = c.S4(); s5 = c.S5(); T = c.Torus(s4["wa"], s4["wb"], s4["PUN"]); r0 = s5["r0"]; tc = pickle.load(open(os.path.join(c.CK, "taut83.pkl"), "rb"))
    b = pickle.load(open(os.path.join(ENG, "study86_floer", "runs", tag, "b83.pkl"), "rb")); PERM = list(b["perm"])
    import itertools
    P = list(itertools.combinations(range(6), 2)); dimsc = {(PERM[i], PERM[j]): b["dims"][(i, j)] for i, j in P}; dimsc.update({(q, p): v for (p, q), v in list(dimsc.items())})
    Minv = np.linalg.inv(np.array([[T.wa.real, T.wb.real], [T.wa.imag, T.wb.imag]]))
    def fold(z):
        ab = Minv @ np.array([z.real, z.imag]); return ab - np.floor(ab)
    def curve(k, reps=2):
        cv = tc["curves"][k]; z = c.offset_curve(cv["verts"], cv["pegs"], cv["lam"], r0); lam = complex(cv["lam"])
        return np.concatenate([z + q * lam for q in range(-reps, reps + 1)])
    pairs = [(0, 1, "colliding pair $V_1, V_2$ (bracket A3, week 4.73)"), None]
    one = [(p, q) for (p, q), v in dimsc.items() if p == 0 and q != 1 and v == 1]
    pairs[1] = (0, one[0][1], f"$V_1$ and $V_{one[0][1]+1}$")
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 4.6))
    for ip, (a_, (p, q, ttl)) in enumerate(zip(ax, pairs)):
        za, zb0 = curve(p, 3), curve(q, 3); cen = za[len(za) // 2]
        ref = za[None, :] if ip == 0 else za[None, len(za) // 2:len(za) // 2 + 1]
        sh = sorted(((np.abs(zb0[:, None] + m * T.wa + n * T.wb - ref).min(), m, n) for m in range(-4, 5) for n in range(-4, 5)))
        zb = np.concatenate([zb0 + m * T.wa + n * T.wb, [np.nan]] + [np.r_[zb0 + sh[k][1] * T.wa + sh[k][2] * T.wb, np.nan] for k in (1, 2)]) if False else None
        zb = np.concatenate([np.r_[zb0 + sh[k][1] * T.wa + sh[k][2] * T.wb, np.nan] for k in range(3 if ip == 0 else 1)])
        zb1 = zb0 + sh[0][1] * T.wa + sh[0][2] * T.wb; Dm = np.abs(zb1[:, None] - za[None, :]); ib, ia = np.unravel_index(np.argmin(Dm), Dm.shape)
        cen = (za[ia] + zb1[ib]) / 2
        if ip == 1:                                                         # centre on an actual crossing of the two polylines
            for k1 in range(len(za) - 1):
                for k2 in range(len(zb1) - 1):
                    p1, p2, q1, q2 = za[k1], za[k1 + 1], zb1[k2], zb1[k2 + 1]; d = p2 - p1; e = q2 - q1; den = (np.conj(d) * e).imag
                    if abs(den) < 1e-18: continue
                    tt = (np.conj(q1 - p1) * e).imag / den; uu = (np.conj(q1 - p1) * d).imag / den
                    if 0 <= tt <= 1 and 0 <= uu <= 1: cen = p1 + tt * d
        R = max(2.2 * abs(complex(tc["curves"][p]["lam"])), 2.5 * Dm.min()) if ip == 0 else 1.8 * abs(complex(tc["curves"][p]["lam"]))
        for z, col, nm in ((za, C1, f"$V_{p+1}$"), (zb, C2, f"$V_{q+1}$")): a_.plot(z.real, z.imag, color=col, lw=1.6, label=nm)
        PU = np.array([w + m * T.wa + n * T.wb for w in T.pun for m in range(-3, 4) for n in range(-3, 4)])
        a_.plot(PU.real, PU.imag, "x", color=INK, ms=6, mew=1.5, label="punctures")
        a_.set_xlim(cen.real - R, cen.real + R); a_.set_ylim(cen.imag - R, cen.imag + R); a_.set_aspect("equal"); a_.set_xticks([]); a_.set_yticks([])
        a_.set_title(ttl + f"\nFloer generators {dimsc[(p, q)]} one way, {dimsc[(q, p)]} the other"); a_.legend(fontsize=7.5, loc="lower right", frameon=False)
    fig.suptitle("vanishing cycles on the fibre (universal cover of the torus with 6 punctures), Study 86 cached representatives", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "figmcg_fibre.png"), dpi=200); plt.close(fig)
    return dimsc[(0, 1)], one[0]
# ------------------------------------------------------------------ 4. Hurwitz move and sigma^2 (schematic)
def fig_hurwitz():
    fig, ax = plt.subplots(1, 3, figsize=(9.8, 3.9))
    P = [(-0.45, 0.35), (0.0, 0.5), (0.45, 0.35)]; b = (0.0, -0.6)
    for k_, a_ in enumerate(ax):
        a_.set_xlim(-1.05, 1.05); a_.set_ylim(-1.05, 1.05); a_.set_aspect("equal"); a_.axis("off")
        if k_ == 2: continue
        a_.add_patch(Circle((0, 0), 0.95, fill=False, ec=INK)); a_.plot(*b, "o", color=INK, ms=4); a_.text(b[0], b[1] - 0.13, "$s_0$", ha="center")
        for k, p in enumerate(P): a_.plot(*p, "x", color=C2, ms=8, mew=2); a_.text(p[0], p[1] + 0.1, f"$v_{k+1}$", ha="center")
    for k, p in enumerate(P): ax[0].plot([b[0], p[0]], [b[1], p[1]], color=[C1, C3, C4][k], lw=1.6)
    ax[0].set_title("distinguished paths $\\gamma_1,\\gamma_2,\\gamma_3$\nvanishing cycles $V_1,V_2,V_3$")
    ax[1].plot([b[0], P[0][0]], [b[1], P[0][1]], color=C1, lw=1.6); ax[1].plot([b[0], P[2][0]], [b[1], P[2][1]], color=C4, lw=1.6)
    th = np.linspace(np.pi * 1.15, np.pi * 0.25, 60); arc = np.c_[P[0][0] + 0.22 * np.cos(th), P[0][1] + 0.22 * np.sin(th)]
    path = np.vstack([[b], [P[0][0] - 0.3, P[0][1] - 0.25], arc, [P[1]]]); ax[1].plot(path[:, 0], path[:, 1], color=C3, lw=1.6)
    ax[1].text(-0.86, 0.05, "$\\gamma_1\\gamma_2\\gamma_1^{-1}$", color=C3, fontsize=8); ax[1].text(-0.1, -0.1, "$\\gamma_1$", color=C1, fontsize=8)
    ax[1].set_title("$\\sigma_1$: $(\\gamma_1,\\gamma_2)\\mapsto(\\gamma_1\\gamma_2\\gamma_1^{-1},\\gamma_1)$\n$(T_1,T_2)\\mapsto(T_1T_2T_1^{-1},T_1)$  (Hurwitz move)")
    ax[2].text(0, 0.0, "$\\sigma_1^2(T_a,T_b)=(T_aT_b)\\,(T_a,T_b)\\,(T_aT_b)^{-1}$\n\n$a\\cap b=\\emptyset\\ \\Rightarrow\\ T_aT_b=T_bT_a$\n$\\Rightarrow\\ \\sigma_1^2$ acts trivially\n\n(collision wall: $\\sigma=\\sigma^{-1}$)", ha="center", va="center", fontsize=10)
    ax[2].set_title("why a collision is harmless\n(Lu--Wedeen--Zhang, p. 10)")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "figmcg_hurwitz.png"), dpi=200); plt.close(fig)
# ------------------------------------------------------------------ 5. the three groups (schematic)
def fig_groups():
    fig, ax = plt.subplots(figsize=(9.8, 3.9)); ax.set_xlim(0, 10); ax.set_ylim(0, 4.2); ax.axis("off")
    def box(x, y, w, h, title, body, col):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc="white", ec=col, lw=1.6))
        ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="top", fontsize=9, weight="bold", color=col); ax.text(x + w / 2, y + h - 0.62, body, ha="center", va="top", fontsize=7.8, color=INK)
    box(0.1, 2.25, 3.0, 1.8, "1. data symmetry", "Markov-move lattice $N$\n(zero-margin integer tables)\n$=$ cocharacters of the mirror torus\n$N\\otimes\\mathbb{C}^*$ = odds-ratio patterns", C3)
    box(3.5, 2.25, 3.0, 1.8, "2. description (gauge)", "$B_N$ on distinguished bases\n(Hurwitz moves = Stokes events)\nglobal conjugation in MCG(fibre)\n$\\Rightarrow$ mutations", C4)
    box(6.9, 2.25, 3.0, 1.8, "3. dynamics (monodromy)", "$\\pi_1$(complexified margins $\\setminus$ disc.)\nlarge-radius loop: $\\otimes$ line bundle\ncaustic: $\\sigma^3$ (braid relation)\ncollision: $\\sigma^2$ = id", C2)
    chain = ["$\\pi_1(M_{\\mathbb{C}}\\setminus\\Delta)$", "$B_N=\\mathrm{MCG}(D_N)$", "$\\mathrm{MCG}(\\text{fibre})$", "$\\mathrm{Aut}\\,D^b$ / mutations"]
    for k, s in enumerate(chain):
        x = 0.9 + 2.75 * k; ax.text(x, 0.75, s, ha="center", va="center", fontsize=9, bbox=dict(boxstyle="round", fc="#f3f4f6", ec=MUTED))
        if k < 3: ax.annotate("", (x + 1.55, 0.75), (x + 1.0, 0.75), arrowprops=dict(arrowstyle="->", color=INK))
    ax.text(5.0, 1.55, "monodromy of the family of mirrors, driven by the margins (3.) and recorded in a basis (2.) on the torus of odds ratios (1.)", ha="center", fontsize=8, color=MUTED)
    fig.savefig(os.path.join(HERE, "figmcg_groups.png"), dpi=200, bbox_inches="tight"); plt.close(fig)
if __name__ == "__main__":
    print("odds", fig_odds()); fig_strands(); print("fibre", fig_fibre()); fig_hurwitz(); fig_groups()
