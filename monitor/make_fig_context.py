"""Phase 5 figure: the context-to-outcome map. Left: distance of each GSS item's wording x answer x party table from the invariance
variety (mean I(O;C|A) per year), with the Bonferroni decision. Right: toric type (vertex count of the answer x party coupling polytope)
of the welfare system in the two wordings over the years, observed and under the invariant map. Writes figs/fig_context_map.pdf."""
import os, json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
H = os.path.dirname(os.path.abspath(__file__)); R = json.load(open(os.path.join(H, 'ext', 'phase5', 'results', 'context_map.json')))
INK, MUTED, GRID, BLUE, ORANGE = '#1f1f1e', '#6b6a63', '#d9d8d2', '#2a78d6', '#eb6834'
plt.rcParams.update({'font.size': 8, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.3), gridspec_kw=dict(width_ratios=[1, 1.4]))
items = sorted([k for k in R['gss'] if not k.startswith('_')], key=lambda k: R['gss'][k]['I_bits_mean'])
for i, k in enumerate(items):
    v = R['gss'][k]; c = INK if v['rejected_bonferroni'] else MUTED
    a1.scatter(v['I_bits_mean'], i, s=34 if v['rejected_bonferroni'] else 22, color=c, marker='o' if v['rejected_bonferroni'] else 'o', facecolor=c if v['rejected_bonferroni'] else 'white', zorder=3)
a1.set_yticks(range(len(items))); a1.set_yticklabels(items); a1.set_xlabel(r'mean $I(O;C\mid A)$, bits')
a1.set_title('Distance from the invariance variety', loc='left', fontsize=8.5, color=INK)
a1.text(0.98, 0.02, 'filled: invariance rejected\n(Bonferroni over 11 items)', transform=a1.transAxes, ha='right', va='bottom', fontsize=6.5, color=MUTED)
by = R['gss']['welfare']['by_year']; yrs = sorted(int(y) for y in by)
for ctx, col, lab, dy in (('X', BLUE, 'wording "welfare"', -0.25), ('Y', ORANGE, 'wording "assistance to the poor"', 0.25)):
    obs = [by[str(y)]['toric'][ctx]['type_observed'] for y in yrs]; inv = [by[str(y)]['toric'][ctx]['type_invariant'] for y in yrs]
    a2.plot(yrs, np.array(obs) + dy, color=col, lw=1.4, marker='o', ms=3.5, label=lab + ', observed')
    a2.scatter(yrs, np.array(inv) + dy, s=26, facecolor='none', edgecolor=col, linewidth=1, label=lab + ', invariant map')
a2.set_yticks([9, 12, 15, 18]); a2.set_ylabel('vertices of the coupling polytope'); a2.set_xlabel('year')
a2.set_title('Welfare: toric type of answer $\\times$ party, by wording', loc='left', fontsize=8.5, color=INK)
fig.legend(*a2.get_legend_handles_labels(), frameon=False, loc='lower center', ncol=2, fontsize=6.8)
for a in (a1, a2):
    a.grid(axis='x' if a is a1 else 'y', color=GRID, lw=0.6); a.set_axisbelow(True)
    for s in ('top', 'right'): a.spines[s].set_visible(False)
fig.tight_layout(rect=(0, 0.1, 1, 1)); fig.savefig(os.path.join(H, 'figs', 'fig_context_map.pdf')); fig.savefig('/tmp/claude-0/context.png', dpi=120); print('ok')
