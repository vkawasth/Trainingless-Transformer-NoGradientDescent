"""Figures for 'Co-revision over any graph', 'Brokerage and Milnor's invariant' and 'A probabilistic theory of pseudo-links':
  fig_graphs     the six drawings: edges, which pairs of independent edges commute as twists, and faithfulness of single / squared twists
  fig_broker     (a) relay, nested and sequential brokerage as interval diagrams with their triple invariants; (b) exact probability that the
                 triple invariant is 1 on interleaved visits, as a function of the correlation of the two visitors' biases, with Monte Carlo
                 points; (c) the three noise regimes for a tie of T steps"""
import os, sys, json
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "engine"))
import warnings; warnings.filterwarnings("ignore")
from pce import raagbraid as RG, pseudoprob as PP, tiebraid as TB
INK, MUTED, LINE, ACC, WARN = "#222222", "#666666", "#1f5f9e", "#c27c0e", "#b03a2e"
faith = json.load(open(os.path.join(HERE, "..", "engine", "raagbraid_summary.json")))

# graphs
names = [("path4", "path"), ("star3", "star"), ("triangle", "triangle"), ("square", "square"), ("k4_planar", "K4, planar"), ("k4_crossed", "K4, crossed")]
fig, ax = plt.subplots(1, 6, figsize=(13, 3.3))
for a, (n, title) in zip(ax, names):
    D = RG.drawing(n); P = D["P"]
    for k, (i, j) in enumerate(D["E"]):
        cross = any(RG.independent(D, k, l) and RG.segments_cross(D, k, l) for l in range(len(D["E"])) if l != k)
        a.plot([P[i].real, P[j].real], [P[i].imag, P[j].imag], color=WARN if cross else "#8a8a8a", lw=2.4 if cross else 1.8, zorder=1)
    a.scatter(P.real, P.imag, s=140, color="white", edgecolors=INK, linewidths=1.2, zorder=2)
    for i, z in enumerate(P): a.text(z.real, z.imag, str(i + 1), ha="center", va="center", fontsize=8, zorder=3)
    f = faith.get(n); sub = "not a homomorphism" if n == "k4_crossed" else (f"N=1: {'faithful' if f['1'][1] else 'relation'}\nN=2: {'faithful' if f['2'][1] else 'relation'}" if f else "")
    a.set_title(title, fontsize=9.5); a.text(0.5, -0.04, sub, transform=a.transAxes, ha="center", va="top", fontsize=8, color=MUTED)
    a.set_aspect("equal"); a.axis("off")
    cx, cy = P.real.mean(), P.imag.mean(); h = max(np.ptp(P.real), np.ptp(P.imag), 1.0) / 2 + 0.35
    a.set_xlim(cx - h, cx + h); a.set_ylim(cy - h, cy + h)
fig.text(0.5, 0.01, "Red: independent edges whose segments cross (their twists fail to commute). Faithfulness: on all words of length <= 4 or 5.", ha="center", fontsize=8.5, color=MUTED)
fig.tight_layout(rect=(0, 0.12, 1, 1)); fig.savefig(os.path.join(HERE, "fig_graphs.pdf")); fig.savefig(os.path.join(HERE, "fig_graphs.png"), dpi=150); plt.close(fig)

# broker
fig = plt.figure(figsize=(12.5, 3.9)); gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1, 1])
a = fig.add_subplot(gs[0])
pats = [("relay (1 to 3)", [(0, 2)], [(1, 3)], 1), ("nested", [(0, 3)], [(1, 2)], 0), ("sequential", [(0, 1)], [(2, 3)], 0)]
for r, (lab, I1, I3, mu) in enumerate(pats):
    y = 2.6 - 1.25 * r
    for (s, e) in I1: a.plot([s, e], [y + 0.18, y + 0.18], color=LINE, lw=5, solid_capstyle="butt")
    for (s, e) in I3: a.plot([s, e], [y - 0.18, y - 0.18], color=ACC, lw=5, solid_capstyle="butt")
    a.text(-0.25, y, lab, ha="right", va="center", fontsize=9); a.text(3.3, y, rf"$\mu_{{123}}={mu}$", va="center", fontsize=10)
a.text(0, 3.25, "source 1 with broker", color=LINE, fontsize=8); a.text(1.8, 3.25, "source 3 with broker", color=ACC, fontsize=8)
a.set_xlim(-2.2, 4.2); a.set_ylim(-0.2, 3.5); a.axis("off"); a.set_title("(a) relationships as intervals", fontsize=9.5, loc="left")
b = fig.add_subplot(gs[1]); Zv = PP.broker_visit("interleaved"); W = TB.walls(Zv); a01, a12 = W[(0, 1)], W[(1, 2)]
arc = lambda w: {1: (w[0], w[1]), 0: (w[1], w[0] + 2 * np.pi)}
rng = np.random.default_rng(5); cs = np.linspace(-0.95, 0.95, 15); ex = []; mcs = []
enc = set()                                                               # the enclosing sector pair, identified as in check PP4
for o in PP.sample_circular_gaussian(np.eye(3), 600, rng):
    w, init, _, _ = TB.regularised(Zv, o); lk = TB.linking(w, init, 3)
    if all(v == 0 for v in lk) and TB.triple(w, init, 0, 1, 2) == 1: enc.add((TB._sector(o[0] - o[1], a01), TB._sector(o[1] - o[2], a12)))
assert len(enc) == 1; (S1_, S2_), = enc
for c in cs:
    S = np.array([[1, 0, c], [0, 1, 0], [c, 0, 1]]); cov = S[0, 1] - S[0, 2] - S[1, 1] + S[1, 2]
    rho = cov / np.sqrt((S[0, 0] + S[1, 1] - 2 * S[0, 1]) * (S[1, 1] + S[2, 2] - 2 * S[1, 2])); ex.append(PP.joint_sector_prob(arc(a01)[S1_], arc(a12)[S2_], rho, n=800))
for c in cs[::3]:
    S = np.array([[1, 0, c], [0, 1, 0], [c, 0, 1]]); om = PP.sample_circular_gaussian(S, 1500, rng); k = 0
    for o in om:
        w, init, _, _ = TB.regularised(Zv, o); lk = TB.linking(w, init, 3); k += all(v == 0 for v in lk) and TB.triple(w, init, 0, 1, 2) == 1
    mcs.append(k / len(om))
b.plot(cs, ex, color=LINE, lw=1.8, label="exact (phase-difference density)"); b.plot(cs[::3], mcs, "o", color=INK, ms=4, label="Monte Carlo")
b.set_xlabel("correlation of the two visitors' biases", fontsize=9); b.set_ylabel(r"P($\mu_{123}=1$)", fontsize=9); b.legend(fontsize=8, frameon=False)
b.set_title("(b) the invariant under correlated biases", fontsize=9.5, loc="left")
c_ = fig.add_subplot(gs[2]); T = np.array([5, 10, 20, 50, 100, 200, 500, 1000])
c_.plot(T, np.sqrt((T - 1) / 12), color=WARN, lw=1.8, label="independent jitter: sd = sqrt((T-1)/12)")
c_.plot(T, np.log(T) / 2 / (2 * np.pi), color=ACC, lw=1.8, label="random walk: scale (1/2) log T (turns)")
c_.plot(T, np.zeros_like(T, float), color=LINE, lw=1.8, label="persistent bias: 0")
c_.set_xscale("log"); c_.set_xlabel("length of the tie T (steps)", fontsize=9); c_.set_ylabel("spread of the winding (turns)", fontsize=9)
c_.legend(fontsize=7.5, frameon=False, loc="upper left"); c_.set_title("(c) three regimes of perturbation", fontsize=9.5, loc="left")
for x in (b, c_): x.spines[["top", "right"]].set_visible(False); x.tick_params(labelsize=8)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_broker.pdf")); fig.savefig(os.path.join(HERE, "fig_broker.png"), dpi=150); plt.close(fig)
