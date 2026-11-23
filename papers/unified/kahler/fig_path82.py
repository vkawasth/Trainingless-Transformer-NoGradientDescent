"""Figure for Study 82 (fig_path82.pdf): the cylinder of a peg-free geodesic and the generator positions along the dense ladder."""
import os, pickle
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
E = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine")
TS = ["0.015", "0.01525", "0.0155", "0.01575", "0.016", "0.01625", "0.0165", "0.01675", "0.017", "0.01725", "0.0175"]
C_OLD, C_NEW, INK, MUTED = "#C2410C", "#1D4ED8", "#1F2937", "#6B7280"
def lat(z, wa, wb):
    if wb.real < 0: wa, wb = -wa, -wb                                       # canonical sign of the lattice basis (the pipeline may return -wa, -wb)
    B = np.array([[wa.real, wb.real], [wa.imag, wb.imag]]); return np.linalg.solve(B, [z.real, z.imag])
fig, (ax, bx) = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw=dict(width_ratios=[1, 1.35]))
# ---- left: one fundamental domain at t = 0.016, drawn in lattice coordinates (an affine image of the flat torus)
t0 = "0.016"; S4 = pickle.load(open(os.path.join(E, "study81_wall", f"ckpt_t{t0}", "S4_flat.pkl"), "rb")); wa, wb = S4["wa"], S4["wb"]
B81 = {t: pickle.load(open(os.path.join(E, "study81_wall", f"t{t}", "bside81.pkl"), "rb")) for t in ("0.01575", "0.016")}
pun = np.array([lat(p, wa, wb) % 1 for p in S4["PUN"]])
lam = B81[t0]["taut"][2]["lam"]; dl = lat(lam, wa, wb)                      # V3 direction in lattice coordinates
n = np.array([-dl[1], dl[0]]); n = n / np.linalg.norm(n)
def line_pts(c):
    """the closed geodesic through point c with direction dl, drawn as segments inside the unit square"""
    segs = []; pts = c + np.outer(np.linspace(0, 1, 4000), dl)
    pts = pts % 1; brk = np.nonzero(np.abs(np.diff(pts, axis=0)).max(1) > 0.5)[0] + 1
    return np.split(pts, brk)
v_old = {t: lat(complex(B81[t]["taut"][2]["verts"][0]), wa, wb) for t in B81}
# cylinder: transverse offsets of the punctures relative to the old position, period = 1/|dl| in the normal direction of the lattice picture
proj = lambda q: float(np.dot(q, n)); per = abs(1.0 / (dl[0] * 0 + 1)) if False else None
offs = []
for p in pun:
    for a in range(-4, 5):
        for b in range(-4, 5): offs.append(proj(p + np.array([a, b]) - v_old[t0]))
offs = np.array(offs); lo = offs[offs < 0].max(); hi = offs[offs > 0].min(); core = v_old[t0] + n * (lo + hi) / 2
for s in np.linspace(lo, hi, 25)[1:-1]:
    for seg in line_pts(v_old[t0] + n * s): ax.plot(seg[:, 0], seg[:, 1], color=C_NEW, alpha=0.06, lw=3)
for t, ls in (("0.01575", ":"), ("0.016", "--")):
    for seg in line_pts(v_old[t]): ax.plot(seg[:, 0], seg[:, 1], color=C_OLD, lw=1.2, ls=ls)
for seg in line_pts(core): ax.plot(seg[:, 0], seg[:, 1], color=C_NEW, lw=2)
ax.scatter(pun[:, 0], pun[:, 1], s=28, color=INK, zorder=5)
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xlabel("lattice coordinate 1", color=MUTED); ax.set_ylabel("lattice coordinate 2", color=MUTED)
ax.set_title("$V_3$ in its cylinder (t = 0.016)", fontsize=10, color=INK)
ax.plot([], [], color=C_NEW, alpha=0.3, lw=6, label="cylinder of parallel geodesics"); ax.plot([], [], color=C_NEW, lw=2, label="core (Study 82)")
ax.plot([], [], color=C_OLD, lw=1.2, ls="--", label="where the shrinking stopped (Study 81)"); ax.scatter([], [], s=28, color=INK, label="punctures")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=7.5, frameon=False, ncol=2)
for sp in ax.spines.values(): sp.set_color(MUTED)
ins = ax.inset_axes([0.58, 0.05, 0.4, 0.3]); c0 = core
for t, ls in (("0.01575", ":"), ("0.016", "--")):
    for seg in line_pts(v_old[t]): ins.plot(seg[:, 0], seg[:, 1], color=C_OLD, lw=1.4, ls=ls)
for seg in line_pts(core): ins.plot(seg[:, 0], seg[:, 1], color=C_NEW, lw=2)
cx, cy = 0.5, (core[1] + (0.5 - core[0]) * dl[1] / dl[0]) % 1
ins.set_xlim(cx - 0.03, cx + 0.03); ins.set_ylim(cy - 0.02, cy + 0.02); ins.set_xticks([]); ins.set_yticks([])
ins.set_title("zoom: two stopping places vs core", fontsize=6.5, color=MUTED)
ax.indicate_inset_zoom(ins, edgecolor=MUTED)
# ---- right: generators of (V3, V6') along the ladder
key = (2, 4)
def series(path_fn):
    xs, ys = [], []
    for t in TS:
        d = pickle.load(open(path_fn(t), "rb")); wa_, wb_ = d["wa"], d["wb"]
        for z in d["pts"][key]: xs.append(float(t)); ys.append(lat(z, wa_, wb_)[1] % 1)
    return np.array(xs), np.array(ys)
x1, y1 = series(lambda t: os.path.join(E, "study81_wall", f"t{t}", "points81.pkl"))
x2, y2 = series(lambda t: os.path.join(E, "study82_path", f"t{t}", "b82.pkl"))
bx.scatter(x1, y1, s=30, facecolors="none", edgecolors=C_OLD, linewidths=1.2, label="Study 81 pipeline (position left free)")
bx.scatter(x2, y2, s=22, color=C_NEW, label="Study 82 (cylinder core)")
bx.set_xlabel("margin parameter $t$", color=MUTED); bx.set_ylabel("generator position (lattice coordinate 2, mod 1)", color=MUTED)
bx.set_title("the three generators of $(V_3, V_{6'})$ along the dense ladder", fontsize=10, color=INK)
bx.grid(alpha=0.15); bx.legend(fontsize=8, frameon=False, loc="best")
for sp in bx.spines.values(): sp.set_color(MUTED)
fig.tight_layout(); fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig_path82.pdf")); fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig_path82.png"), dpi=130)
print("written")
