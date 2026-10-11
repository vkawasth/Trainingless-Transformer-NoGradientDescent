"""Fano tubes of the PhysioNet tower (PREANALYSIS_TOWER_ICU.md, Part 2): the 2 x 3 system death x first-window pressure band.
Each tower gradation predicts the death margin f1; with the pressure margin g fixed, (f1, g) fixes the coupling polygon, the moment
polygon of a toric Fano surface. Writes figs/fig_fano_tubes.pdf."""
import os, sys, json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H); sys.path.insert(0, os.path.join(H, '..', 'bridges', 'fukaya'))
from toric_public import polygon
from monitor import bands
R = json.load(open(os.path.join(H, 'ext', 'open_clinical', 'results', 'tower_icu.json')))
X = json.load(open(os.path.join(H, 'ext', 'open_clinical', 'results', 'tower_icu_exploratory.json')))
COL = {'CP^2': '#2a78d6', 'CP^1 x CP^1': '#eb6834', 'F_1 = Bl_1 CP^2': '#1baf7a', 'Bl_2 CP^2': '#eda100', 'Bl_3 CP^2 (dP6)': '#e87ba4'}
LAB = {'CP^2': r'$\mathbb{CP}^2$', 'F_1 = Bl_1 CP^2': r'$\mathbb{F}_1$', 'CP^1 x CP^1': r'$\mathbb{CP}^1\times\mathbb{CP}^1$', 'Bl_2 CP^2': r'Bl$_2\mathbb{CP}^2$'}
INK, MUTED = '#1f1f1e', '#6b6a63'
ALPHAS = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]
plt.rcParams.update({'font.size': 8, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED})

rows = [('Same population (A $\\to$ B)', R['same_population'], None),
        ('Shift (surgical $\\to$ medical/coronary), floor = fit base rate', R['shift'], None),
        ('Shift, floor = evidence base rate (exploratory)', R['shift'], X['shift_evidence_floor'])]
fig = plt.figure(figsize=(7.2, 8.4))
for i, (title, arm, ex) in enumerate(rows):
    g = np.array(arm['g_H1_test']); tube = ex['tube'] if ex else [dict(n=t['n'], f1=t['f1_level1'], surface=t['surface_level1']) for t in arm['tube']]
    obs = X['shift' if i else 'same_population']['observed_rate']; nobs = X['shift' if i else 'same_population']['n']
    # --- (a) the tube: level-one polygon at each evidence count
    ax = fig.add_subplot(3, 2, 2 * i + 1, projection='3d')
    for t in tube[::4] + [tube[-1]]:
        P = polygon(t['f1'], g); verts = [[(t['n'], x, y) for x, y in P]]
        ax.add_collection3d(Poly3DCollection(verts, facecolor=COL[t['surface']], alpha=0.12, edgecolor=COL[t['surface']], linewidth=0.9))
    P = polygon(obs, g); nend = tube[-1]['n'] * 1.08
    ax.plot([nend] * (len(P) + 1), list(P[:, 0]) + [P[0, 0]], list(P[:, 1]) + [P[0, 1]], color=INK, linestyle='--', linewidth=1.0)
    ax.set_xlim(0, nend); ax.set_ylim(0, 0.16); ax.set_zlim(0, 0.16); ax.view_init(elev=18, azim=-62)
    ax.set_xlabel('evidence $n$', labelpad=-2); ax.set_ylabel(r'$\alpha_{11}$', labelpad=-4); ax.set_zlabel(r'$\alpha_{12}$', labelpad=-4)
    ax.tick_params(pad=-2, labelsize=6); ax.set_title(title, fontsize=8, color=INK, loc='left')
    # --- (b) the gradations on the f1 axis, with the wall f1 = g0
    ax = fig.add_subplot(3, 2, 2 * i + 2)
    lo, hi = 0.08, 0.22; wall = g[0]
    ax.axvspan(lo, wall, color=COL['CP^2'], alpha=0.10, lw=0); ax.axvspan(wall, hi, color=COL['F_1 = Bl_1 CP^2'], alpha=0.10, lw=0)
    ax.axvline(wall, color=MUTED, lw=1); ax.text(wall, 1.02, r'wall $f_1=g_0$', transform=ax.get_xaxis_transform(), ha='center', fontsize=7, color=MUTED)
    ax.text(lo + 0.003, 0.30, LAB['CP^2'], transform=ax.get_xaxis_transform(), color=INK, fontsize=9, va='top')
    ax.text(hi - 0.003, 0.30, LAB['F_1 = Bl_1 CP^2'], transform=ax.get_xaxis_transform(), color=INK, fontsize=9, va='top', ha='right')
    ns = [t['n'] for t in tube]; fs = [t['f1'] for t in tube]
    ax.plot(fs, ns, color=INK, lw=1.4); ax.text(fs[len(fs) // 2] + 0.003, ns[len(ns) // 2], 'level 1', fontsize=6.5, color=INK)
    f_ka = np.array(ex['f_ka'] if ex else arm['f_ka'])
    if ex: W = np.array(ex['weights'])
    else: W = np.zeros_like(f_ka); W[3] = np.array(arm['tube'][-1]['level3_alpha_weights'])   # posterior on M3 is 1.0 in both arms
    yk = nend = ns[-1] * 1.12
    for a in range(len(ALPHAS)):
        if W[3, a] > 1e-3: ax.scatter(f_ka[3, a], yk, s=8 + 200 * W[3, a], color=COL['CP^2'] if f_ka[3, a] < wall else COL['F_1 = Bl_1 CP^2'], edgecolor='white', linewidth=1, zorder=3)
    ax.text(f_ka[3].min(), yk * 1.04, r'level 3: M3 at each $\alpha$ (size = weight)', fontsize=6.5, color=MUTED)
    for k, m in enumerate(['M0', 'M1', 'M2', 'M3', 'M4']):
        ax.plot([f_ka[k, -1]], [ns[-1] * 1.25], marker='|', color=MUTED, ms=7)
    ax.text(lo + 0.002, ns[-1] * 1.29, r'level 2: models at $\alpha=1$', fontsize=6.5, color=MUTED)
    blo, bhi = bands([obs, 1 - obs], nobs); ax.axvspan(blo[0], bhi[0], ymax=0.06, color=INK, alpha=0.25, lw=0)
    ax.scatter([obs], [0], marker='*', s=70, color=INK, zorder=4); ax.text(obs, 0.075, 'observed', transform=ax.get_xaxis_transform(), ha='center', fontsize=6.5, color=INK)
    ax.set_xlim(lo, hi); ax.set_ylim(-ns[-1] * 0.05, ns[-1] * 1.36); ax.set_ylabel('evidence $n$'); ax.set_xlabel(r'predicted death rate $f_1$')
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
fig.tight_layout(h_pad=1.6)
fig.savefig(os.path.join(H, 'figs', 'fig_fano_tubes.pdf')); fig.savefig('/tmp/claude-0/fano.png', dpi=110); print('wrote figs/fig_fano_tubes.pdf')
