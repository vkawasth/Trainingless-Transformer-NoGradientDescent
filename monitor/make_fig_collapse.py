"""Phase 6 figures: fig_collapse_trajectories.pdf (the tracker on four real trajectories) and fig_recursive.pdf (recursive training)."""
import os, sys, json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
H = os.path.dirname(os.path.abspath(__file__)); P6 = os.path.join(H, 'ext', 'phase6'); sys.path.insert(0, P6); sys.path.insert(0, H)
INK, MUTED, GRID = '#1f1f1e', '#6b6a63', '#d9d8d2'; C4 = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
plt.rcParams.update({'font.size': 7.5, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED})
def tidy(a, gridaxis='y'):
    a.grid(axis=gridaxis, color=GRID, lw=0.6); a.set_axisbelow(True)
    for s in ('top', 'right'): a.spines[s].set_visible(False)
T = json.load(open(os.path.join(P6, 'results', 'trajectories.json')))
fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.4))
# (a) welfare x race, wording Y: signed distance of the nearest wall with its band interval; decided flip marked
import pandas as pd, fan_tracker as FT
from site_separation import YV, wtable
_d = pd.read_csv(os.path.join(H, 'data', 'gss', 'gss_extract.csv'), low_memory=False); _d = _d[_d.year >= 1985]; _tabs, _ne, _yr = [], [], []
for _y, _s in _d.groupby('year'):
    _T, _n, _m = wtable(_s, YV['natfare'], YV['natrace'])
    if _m >= 200: _tabs.append(_T); _ne.append(_n); _yr.append(int(_y))
tr = FT.track(_tabs, neff=_ne, rng=np.random.default_rng(0), labels=_yr); yrs = [r['step'] for r in tr]
assert [r.get('decided_flips') for r in tr] == [r.get('decided_flips') for r in T['gss']['Y|welfare|race']]   # same as the frozen run
k = int(np.argmin(np.abs(np.array(tr[yrs.index(2014)]['signs']))))       # the wall nearest in 2014
v = [r['signs'][k] for r in tr]
a = ax[0, 0]; a.axhline(0, color=MUTED, lw=1); a.fill_between(yrs, [r['wall_lo'][k] for r in tr], [r['wall_hi'][k] for r in tr], color=C4[0], alpha=0.15, lw=0)
a.plot(yrs, v, color=INK, lw=1.4, marker='o', ms=2.5)
for r in tr:
    if r.get('decided_flips', 0) > 0: a.scatter([r['step']], [r['signs'][k]], s=60, facecolor='none', edgecolor=C4[1], lw=1.4, zorder=4)
a.set_title('GSS welfare $\\times$ race (wording Y): nearest wall', loc='left', color=INK, fontsize=8); a.set_ylabel('signed wall value $f(I)-g(J)$'); tidy(a)
a.text(0.02, 0.04, 'circle: decided chamber change', transform=a.transAxes, color=MUTED, fontsize=6.5)
# (b) ICU decedents vs survivors: tear margin over 6-hour windows
a = ax[0, 1]; I = T['icu']
for i, g in enumerate(('survivors', 'decedents')):
    a.plot(range(8), [r['tear_margin'] for r in I[g]], color=C4[i], lw=1.4, marker='o', ms=3, label=g)
a.set_xticks(range(8)); a.set_xticklabels([f'{6*w}' for w in range(8)]); a.set_xlabel('hours since admission (window start)'); a.set_ylabel('tear margin $m_T$')
a.set_title('ICU MAP $\\times$ HR: distance to the nearest wall', loc='left', color=INK, fontsize=8); a.legend(frameon=False, fontsize=6.5); tidy(a)
# (c) tower paths: signed distance to the wall f1 = g0 along the evidence, with the interval from the pressure band
a = ax[1, 0]; a.axhline(0, color=MUTED, lw=1)
for i, (name, lab) in enumerate((('R0b shift A surgical -> B medical/coronary', 'R0b'), ('R2 reverse shift B surgical -> A medical/coronary', 'R2'),
                                 ('R4 cross shift A coronary+cardiac surgery -> B medical+surgical', 'R4'), ('R0a same population (A -> B)', 'R0a'))):
    p = T['tower'][name]['path']; n = [q['n'] for q in p]; d = [q['wall_f1_g0'] for q in p]
    a.fill_between(n, [q['wall_interval'][0] for q in p], [q['wall_interval'][1] for q in p], color=C4[i], alpha=0.12, lw=0)
    a.plot(n, d, color=C4[i], lw=1.3, label=lab)
a.set_xlabel('evidence $n$'); a.set_ylabel('$f_1 - g_0$ (tower level one)'); a.legend(frameon=False, fontsize=6.5, ncol=4)
a.set_title('Tower: hypotheses moving through chambers', loc='left', color=INK, fontsize=8); tidy(a)
a.text(0.98, 0.96, 'below 0: $\\mathbb{CP}^2$; above: $\\mathbb{F}_1$', transform=a.transAxes, ha='right', va='top', color=MUTED, fontsize=6.5)
# (d) LLM retraining: drift split
a = ax[1, 1]; L = T['llm']; tasks = list(L); xs = np.arange(len(tasks))
inter = [L[t][1]['drift']['interaction'] for t in tasks]; marg = [L[t][1]['drift']['margin'] for t in tasks]
a.bar(xs, marg, color=C4[0], width=0.7, label='margin drift'); a.bar(xs, inter, bottom=marg, color=C4[1], width=0.7, label='interaction drift')
a.set_yscale('symlog', linthresh=0.05); a.set_xticks(xs); a.set_xticklabels([t.replace('COUNTAHAPPYNUMBER', 'HAPPY').replace('LEETCODE_EASY', 'LEETCODE').replace('PRIME_FULL', 'PRIMES').replace('SENSITIVEQ', 'SENSITIVE') for t in tasks], rotation=40, ha='right')
a.set_ylabel('$D(p_{June}\\,\\|\\,p_{March})$, bits'); a.legend(frameon=False, fontsize=6.5); a.set_title('LLM retraining: margin vs interaction drift', loc='left', color=INK, fontsize=8); tidy(a)
fig.tight_layout(); fig.savefig(os.path.join(H, 'figs', 'fig_collapse_trajectories.pdf')); fig.savefig('/tmp/claude-0/traj.png', dpi=110)

# ---------------- recursive training
import recursive_training as R
RT = json.load(open(os.path.join(P6, 'results', 'recursive_training.json'))); rng = np.random.default_rng(R.SEED); Pxc, P0 = R.truth(rng); P0full = np.broadcast_to(P0, (R.C, R.X, R.Y)).copy()
fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.2)); g = np.arange(R.GEN + 1)
a = ax[0, 0]
for i, l in enumerate(R.LAMS): a.plot(g, RT['P4_invariance']['mean_cmi_bits'][str(l)], color=C4[i], lw=1.4, label=f'fresh {int(l*100)}%')
a.set_xlabel('generation'); a.set_ylabel('$I(Y;C\\mid X)$ of the model, bits'); a.set_title('Contexts leave the invariance variety', loc='left', color=INK, fontsize=8); a.legend(frameon=False, fontsize=6.5); tidy(a)
a = ax[0, 1]
for i, l in enumerate(R.LAMS):
    Qs = np.array([R.run_chain(P0full, l, None, np.random.default_rng(R.SEED + 10 * r + int(l * 100))) for r in range(R.REPS)])
    pos = np.broadcast_to(P0 > 0, Qs.shape[2:]); a.plot(g, (Qs[:, :, pos] == 0).mean((0, 2)), color=C4[i], lw=1.4, label=f'fresh {int(l*100)}%')
a.set_xlabel('generation'); a.set_ylabel('share of true categories at 0'); a.set_title('Breaks accumulate; fresh data slows them', loc='left', color=INK, fontsize=8); a.legend(frameon=False, fontsize=6.5); tidy(a)
a = ax[1, 0]; amax = P0full.argmax(-1); accs, certs, thrs = [], [], []
from monitor import bands
for r in range(R.REPS):
    Q = R.run_chain(P0full, 0.0, None, np.random.default_rng(R.SEED + 10 * r)); rr = np.random.default_rng(R.SEED + 1000 + r); H0 = R.sample(P0full, R.NHOLD, rr) / R.NHOLD
    lo, hi = np.zeros_like(H0), np.zeros_like(H0)
    for c in range(R.C):
        for x in range(R.X): lo[c, x], hi[c, x] = bands(H0[c, x], R.NHOLD)
    accs.append([(Q[k].argmax(-1) == amax).mean() for k in g]); certs.append([R.certificate(Q[k], lo, hi, H0) for k in g])
    nr = np.random.default_rng(R.SEED + 5000 + r); thrs.append(np.percentile([R.certificate(R.fit(R.sample(P0full, R.NCELL, nr)), lo, hi, H0) for _ in range(200)], 95))
a2 = a.twinx(); a.plot(g, np.mean(accs, 0), color=C4[0], lw=1.5, label='accuracy (argmax)'); a2.plot(g, np.mean(certs, 0), color=C4[1], lw=1.5, label='certificate')
a2.axhline(np.mean(thrs), color=C4[1], lw=0.8, ls='--'); a.set_xlabel('generation'); a.set_ylabel('accuracy', color=C4[0]); a2.set_ylabel('certificate vs held-out data', color=C4[1])
a.set_title('Certificate and accuracy, no fresh data', loc='left', color=INK, fontsize=8); tidy(a); a2.spines['top'].set_visible(False)
a = ax[1, 1]; P1 = RT['P1_contraction']
a.plot(g, P1['mean_D'], color=INK, lw=1.5, label='model (mean)'); a.plot(g, P1['bound_plan'], color=C4[0], lw=1, ls='--', label='planned bound ($\\eta(K)=1$)')
a.plot(g, P1['exploratory_sharper_bound']['bound'], color=C4[2], lw=1, ls=':', label='bound with $\\eta(K^g)$')
a.set_xlabel('generation'); a.set_ylabel('distinguishability of $x$ (TV)'); a.set_title('Information about $x$ under a paraphrase kernel', loc='left', color=INK, fontsize=8); a.legend(frameon=False, fontsize=6.5); tidy(a)
fig.tight_layout(); fig.savefig(os.path.join(H, 'figs', 'fig_recursive.pdf')); fig.savefig('/tmp/claude-0/rec.png', dpi=110); print('ok')
