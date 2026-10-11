"""Phase 4 summary figure: left, does geometry forecast beyond the margins (WP1); right, the three reliability levels of the tower
across arms (WP4). Both are differences in test log loss with paired-bootstrap 95% intervals. Writes figs/fig_phase4.pdf."""
import os, json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
H = os.path.dirname(os.path.abspath(__file__)); R = lambda f: json.load(open(os.path.join(H, 'ext', 'phase4', 'results', f)))
INK, MUTED, GRID = '#1f1f1e', '#6b6a63', '#d9d8d2'; COL = {'T-fit': '#2a78d6', 'T-ev': '#eb6834', 'T-fam': '#1baf7a'}
plt.rcParams.update({'font.size': 8, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.6), gridspec_kw=dict(width_ratios=[1, 1.35]))
c, g = R('wp1_clinical.json'), R('wp1_gss.json')['result']
rows = [('ICU death\n(per-stay MAP $\\times$ HR)', c['primary_death']), ('ICU shock onset', c['secondary_shock_onset']), ('GSS chamber change\n(next survey year)', g)]
for i, (lab, r) in enumerate(rows):
    lo, hi = r['ci_B2_S2']; a1.plot([lo, hi], [i, i], color=INK, lw=2, solid_capstyle='round'); a1.scatter([r['delta_B2_S2']], [i], s=36, color=INK, zorder=3)
a1.axvline(0, color=MUTED, lw=1); a1.set_yticks(range(len(rows))); a1.set_yticklabels([r[0] for r in rows]); a1.invert_yaxis()
a1.set_xlabel('log-loss gain of geometry over\nflexible margins (B2 $-$ S2)'); a1.set_title('Geometry beyond the margins', loc='left', fontsize=8.5, color=INK)
t = R('wp4_tower.json')['arms']; names = list(t)
short = ['R0a same pop.', 'R0b shift', 'R1 same pop. (rev.)', 'R2 shift (rev.)', 'R3 opposite shift', 'R4 cross shift']
for i, k in enumerate(names):
    for j, v in enumerate(('T-fit', 'T-ev', 'T-fam')):
        r = t[k]['variants'][v]; y = i + (j - 1) * 0.24
        a2.plot(r['ci'], [y, y], color=COL[v], lw=2, solid_capstyle='round'); a2.scatter([r['delta']], [y], s=22, color=COL[v], edgecolor='white', linewidth=1, zorder=3, label=v if i == 0 else None)
a2.axvline(0, color=MUTED, lw=1); a2.set_yticks(range(len(names))); a2.set_yticklabels(short); a2.invert_yaxis()
a2.set_xlabel('log-loss gain over flat model averaging'); a2.set_title('Reliability level of the tower', loc='left', fontsize=8.5, color=INK)
from matplotlib.lines import Line2D
hd = [Line2D([0], [0], color=COL[v], lw=2, marker='o', ms=4, label=l) for v, l in (('T-fit', 'floor = fit base rate'), ('T-ev', 'floor = evidence base rate'), ('T-fam', 'family of floors'))]
fig.legend(handles=hd, frameon=False, loc='lower center', ncol=3, fontsize=7)
for a in (a1, a2):
    a.grid(axis='x', color=GRID, lw=0.6); a.set_axisbelow(True)
    for s in ('top', 'right'): a.spines[s].set_visible(False)
fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(os.path.join(H, 'figs', 'fig_phase4.pdf')); fig.savefig('/tmp/claude-0/phase4.png', dpi=120); print('wrote figs/fig_phase4.pdf')
