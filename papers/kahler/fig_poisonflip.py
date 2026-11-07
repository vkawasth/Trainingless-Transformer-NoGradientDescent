"""Figure for section 12.20 of the companion note (module pce/poisonflip.py).
  fig_poisonflip.pdf  (a) undetectable placements against the hidden structure (all 2721 faces of the 4-cycle's D); (b) how often 4 cells
                      suffice for the cheapest undetectable flip, n = 4..8; (c) a decision over time with one poisoned batch; (d) removing
                      each batch, one at a time, at the flipped times"""
import os, sys, collections, random
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "engine"))
from pce import margsec as G
from pce import poisonflip as P
BLUE, ORANGE, AQUA, VIOLET, ROSE, INK, INK2, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#c2417a", "#22211f", "#52514e", "#9a9a9a"
plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2})
CELLS = G.CELLS; M4 = np.array(G.M, float); W = np.array([1.0 if c[0] == c[1] else 0.0 for c in CELLS])
E4, _ = P.entries_chsh(); C4 = P.circuits(E4); SU4 = P.signed_unit(E4, C4); DV = P.dual_vertices(W); F = G.facets_D()
fig, ax = plt.subplots(2, 2, figsize=(12, 8.4))

# (a)
faces = G.faces_D(F); agg = collections.defaultdict(list)
for V in faces: agg[(G.face_dim(sorted(V)), G.fibre_dim_face(sorted(V)))].append(P.tangent_count(V, F, SU4))
a = ax[0, 0]; cols = {0: BLUE, 1: AQUA, 2: ORANGE, 3: VIOLET, 4: ROSE, 7: INK}; mk = {0: "o", 1: "^", 2: "s", 3: "D", 4: "v", 7: "*"}
for fb in sorted({k[1] for k in agg}):
    ks = sorted(k for k in agg if k[1] == fb); off = 0.09 * (fb - 1.5) if fb < 7 else 0
    a.plot([k[0] + off for k in ks], [np.mean(agg[k]) for k in ks], mk[fb], color=cols[fb], ms=8 if fb == 7 else 6.5, label=f"fibre dim {fb}")
a.annotate("CH facet\n(rigid)", (7 - 0.135, 252), xytext=(5.6, 300), fontsize=7.6, color=INK2, arrowprops=dict(arrowstyle="-", color=GREY, lw=0.7))
a.annotate("positivity facet", (7 + 0.225, 287), xytext=(4.6, 345), fontsize=7.6, color=INK2, arrowprops=dict(arrowstyle="-", color=GREY, lw=0.7))
a.set_xlabel("face dimension of the data (room to move undetected)"); a.set_ylabel("undetectable placements (of 376 signed circuits)")
a.legend(frameon=False, fontsize=7.6, loc="upper left", title="hidden freedom", title_fontsize=7.6)
a.set_title("(a) Where an attacker can edit without leaving D: monotone in the hidden\nstructure; dim + fibre dim = |S| - 1 splits it into data room and rule room", fontsize=9)

# (b)
b = ax[0, 1]; ns = range(4, 9); frac = []
for n in ns:
    cells, Mn, _ = G.ncycle(n); Mn = np.array(Mn, float); En, _ = P.entries_ncycle(n); SU = P.signed_unit(En, P.circuits_upto(En, 4))
    w = np.array([1.0 if c[0] == c[2] else 0.0 for c in cells]); r = np.random.default_rng(n); hit = tot = 0
    for s in range(30):
        x = r.dirichlet(np.ones(len(cells))); d = Mn @ x; L, U = P.lower_upper(d, w, M=Mn); tau = L - 0.01
        if tau <= 0: continue
        cs, _ = P.flip_some(d, w, tau, En, M=Mn); one = min([t for t in (P.along_some(d, c, w, tau, M=Mn) for c in SU) if t is not None] + [np.inf])
        tot += 1; hit += abs(cs - one) < 1e-7
    frac.append((hit, tot))
b.bar(list(ns), [h / t for h, t in frac], 0.6, color=BLUE)
for n, (h, t) in zip(ns, frac): b.text(n, h / t + 0.015, f"{h}/{t}", ha="center", fontsize=8, color=INK2)
b.set_xticks(list(ns)); b.set_xticklabels([f"n = {n}\n{4 * n} cells" for n in ns]); b.set_ylim(0, 1.12)
b.set_ylabel("share of random data where 4 cells suffice")
b.set_title("(b) The cheapest undetectable flip (gap 0.01) usually touches 4 cells,\nwhatever the number of variables: one circuit of the consistent-edit space", fontsize=9)

# (c), (d): the stream of check PF6-PF9
def s_faces():
    faces_ = sorted(G.faces_D(F), key=lambda V: sorted(V)); r = random.Random(20261108)
    return r.sample([V for V in faces_ if len(V) == 8 and G.face_dim(V) == 7 and G.fibre_dim_face(V) == 0], 3)
V0 = s_faces()[0]; r = random.Random(20261108); wv = [r.randint(5, 25) for _ in V0]; t_ = sum(wv); wv = [x * 240 // t_ for x in wv]; wv[0] += 240 - sum(wv)
P1 = [0] * 16
for k, v in zip(sorted(V0), wv): P1[k] = 5 * v
P2 = [4 * a // 5 + 15 for a in P1]; units = lambda wts: [CELLS[i] for i in range(16) for _ in range(wts[i])]
C_ALL = (0, 1, 2, 3); TAU = 0.45; T_END = 22; T_P = 10; S = P.Stream()
for t in range(T_END):
    B = list(range(len(S.counts)))
    if t == T_P:
        d, _ = P.data_of(P.pooled(S, B, C_ALL)); N = len(B) * 1200; q, R = P.poison_mix(d, W, TAU, DV); k = int(np.ceil(q * N / (1 - q)))
        while True:
            Sx = P.Stream(); Sx.counts = list(S.counts); Sx.add(P.units_from_law(R, k), C_ALL)
            if P.lower_upper(P.data_of(P.pooled(Sx, B + [len(B)], C_ALL))[0], W)[1] <= TAU + 1e-12: break
            k += 1
        S.add(P.units_from_law(R, k), C_ALL)
    else: S.add(units(P1 if t < 6 else P2), C_ALL)
f = lambda v: float(W @ P.maxent_rows(v, list(range(8))))
Ls, Us, Ms = [], [], []
for t in range(T_END):
    d, _ = P.data_of(P.pooled(S, range(t + 1), C_ALL)); L, U = P.lower_upper(d, W); Ls.append(L); Us.append(U); Ms.append(f(d))
c = ax[1, 0]; ts = np.arange(T_END)
c.fill_between(ts, Ls, Us, color=BLUE, alpha=0.18, lw=0, label="identified interval [L, U] (every rule lies here)")
c.plot(ts, Ms, "o-", color=BLUE, ms=3.5, lw=1.6, label="maximum-entropy rule"); c.axhline(TAU, color=ROSE, ls="--", lw=1, label=f"threshold {TAU}")
c.axvline(T_P, color=ORANGE, lw=1); c.text(T_P + 0.2, 0.55, f"one poisoned batch\n({k} units of one reading,\nagainst {N} clean)", fontsize=7.6, color=ORANGE)
c.axvline(6, color=GREY, lw=0.8, ls=":"); c.text(6.15, 0.36, "hidden structure\nchanges (face to\ninterior)", fontsize=7.4, color=INK2)
lo = [t for t in range(T_END) if Ms[t] < TAU]; c.axvspan(min(lo) - 0.5, max(lo) + 0.5, color=ROSE, alpha=0.08)
c.set_xlabel("time (batches)"); c.set_ylabel("P(A0 = A1), never observed"); c.set_ylim(0.33, 0.62); c.legend(frameon=False, fontsize=7.4, loc="upper left")
c.set_title(f"(c) The poisoned batch flips every rule for 1 step; the maximum-entropy\ndecision stays flipped for {len(lo)} steps (t = {min(lo)}-{max(lo)}), until clean data dilute it", fontsize=9)
dd = ax[1, 1]
for i, t in enumerate(lo):
    B = list(range(t + 1)); d, _ = P.data_of(P.pooled(S, B, C_ALL)); base = f(d); ch = []
    for bb in B:
        dm, _ = P.data_of(P.remove(S, P.pooled(S, B, C_ALL), B, [x for x in B if x != bb], C_ALL)); ch.append(f(dm) - base)
    dd.plot(B, ch, "o-" if i == 0 else "-", color=BLUE if i == 0 else GREY, ms=3.5, lw=1.4 if i == 0 else 0.8, alpha=1 if i == 0 else 0.6,
            label=f"t = {t}" if i in (0, len(lo) - 1) else None)
dd.axhline(TAU - Ms[lo[0]], color=ROSE, ls="--", lw=1, label=f"change needed to revert at t = {lo[0]}")
dd.axhline(0, color=INK2, lw=0.6); dd.set_xlabel("batch removed (one at a time)"); dd.set_ylabel("change of the maximum-entropy value")
dd.legend(frameon=False, fontsize=7.6, loc="upper left")
dd.set_title("(d) Removal explains the change: at every flipped time only removing\nthe poisoned batch reverts the decision; clean batches barely matter", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_poisonflip.pdf")); fig.savefig(os.path.join(HERE, "fig_poisonflip.png"), dpi=150); print("ok")
