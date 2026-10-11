"""Figures for the validation paper, all from the saved results and the public data (figs/*.pdf).
Palette: the reference categorical slots blue / orange / aqua (at most three series per panel), one-hue blue sequential, blue-red
diverging with a gray midpoint, recessive gray chrome; text in ink, never in series colors."""
import os, sys, json, itertools
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon as MPoly, FancyArrowPatch
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H)
from monitor import State, event_range, tear_margin
os.makedirs(os.path.join(H, 'figs'), exist_ok=True)
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK2, MUTED, GRID, AXIS, SURF = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'
NEUTRAL = '#f0efec'; RED = '#e34948'
SEQ = LinearSegmentedColormap.from_list('seq', ['#cde2fb', '#86b6ef', '#3987e5', '#256abf', '#104281'])
DIV = LinearSegmentedColormap.from_list('div', [BLUE, NEUTRAL, RED])
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.edgecolor': AXIS, 'axes.labelcolor': INK2, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.titlesize': 9, 'axes.titlecolor': INK, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
                     'figure.facecolor': 'white', 'axes.facecolor': SURF, 'axes.grid': False, 'grid.color': GRID, 'grid.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False, 'lines.linewidth': 2, 'legend.frameon': False, 'legend.fontsize': 8})
def save(fig, name): fig.savefig(os.path.join(H, 'figs', name), bbox_inches='tight'); plt.close(fig); print('wrote', name)
J = lambda f: json.load(open(os.path.join(H, f)))
T = J('toric_public.json'); S = J('site_separation.json'); PP = J('panel_paths.json'); G = J('gss_results.json'); PM = J('panel_models.json')
CX = J('ctx_allpairs.json')
d = pd.read_csv(os.path.join(H, 'data', 'gss', 'gss_extract.csv'), low_memory=False); d = d[d.year >= 1985]

# ------------------------------------------------------------------ Fig 1: coupling polygons with toric and Floer data
def cells(x1, x2, f1, g): return np.array([x1, x2, f1 - x1 - x2, g[0] - x1, g[1] - x2, g[2] - f1 + x1 + x2])
def frac(s): a, b = (s.split('/') + ['1'])[:2]; return int(a) / int(b)
def polypanel(ax, r, title):
    f1 = frac(r['f1']); g = [frac(x) for x in r['g']]; P = np.array(r['polygon'])
    ax.add_patch(MPoly(P, closed=True, facecolor='#eef4fc', edgecolor=AXIS, lw=1))
    xs = np.linspace(P[:, 0].min(), P[:, 0].max(), 300); ys = np.linspace(P[:, 1].min(), P[:, 1].max(), 300); X, Y = np.meshgrid(xs, ys)
    C = np.stack([X, Y, f1 - X - Y, g[0] - X, g[1] - Y, g[2] - f1 + X + Y]); ok = (C > 1e-9).all(0)
    Hh = np.where(ok, -(np.where(C > 0, C, 1) * np.log(np.where(C > 0, C, 1))).sum(0), np.nan)
    ax.contour(X, Y, Hh, levels=8, colors=GRID, linewidths=0.6)
    iso = np.array(r['isotopy']); ax.plot(iso[:, 0], iso[:, 1], color=BLUE, lw=1.2, ls=(0, (3, 2)))
    ax.plot(*r['product'], 'o', color=BLUE, ms=7, mec='white', mew=1.2, zorder=5)
    ax.plot(*r['leximin'], 'D', color=ORANGE, ms=11, mec='white', mew=1.2, zorder=5)
    if 'observed' in r: ax.plot(*r['observed'], 's', color=AQUA, ms=7, mec='white', mew=1.2, zorder=5)
    from collections import Counter
    for (u, v), k in Counter(tuple(b) for b in r['branes']).items():
        ax.plot(u, v, marker=(5, 1), color=INK, ms=7, zorder=6); ax.annotate(f'×{k}', (u, v), xytext=(-4, 7), textcoords='offset points', color=INK, fontsize=8, ha='right')
    pad = 0.08 * np.ptp(P, axis=0).max(); ax.set_xlim(P[:, 0].min() - pad, P[:, 0].max() + pad); ax.set_ylim(P[:, 1].min() - pad, P[:, 1].max() + pad)
    ax.set_xlabel(r'$\alpha_{11}$'); ax.set_ylabel(r'$\alpha_{12}$'); ax.set_title(title, fontsize=8.5)
    ax.text(0.98, 0.02, f"{r['surface'].split(' (')[0].replace('CP^2', 'CP²').replace('CP^1', 'CP¹')}\n{r['n_vertices']} vertices = {r['n_vertices']} branes",
            transform=ax.transAxes, ha='right', va='bottom', color=INK2, fontsize=7.5)
fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.6))
polypanel(axs[0], T['NHANES 2009_10'], 'NHANES 2009–10: diabetes × BMI class')
ins = axs[0].inset_axes([0.5, 0.5, 0.48, 0.46]); polypanel(ins, T['NHANES 2009_10'], '')
ins.set_xlim(0.026, 0.040); ins.set_ylim(0.030, 0.041); ins.set_xlabel(''); ins.set_ylabel(''); ins.tick_params(labelsize=6)
for t_ in list(ins.texts): t_.remove()
axs[0].indicate_inset_zoom(ins, edgecolor=MUTED)
polypanel(axs[1], T['health x education (X)']['2016'], 'GSS 2016: health × education')
polypanel(axs[2], T['poor x cities (Y)']['2024'], 'GSS 2024: the poor × big cities')
from matplotlib.lines import Line2D
hs = [Line2D([], [], marker='o', ls='', color=BLUE, mec='white', ms=7, label='product = maximum-entropy coupling (entropy-brane point)'),
      Line2D([], [], marker='D', ls='', color=ORANGE, mec='white', ms=7, label='leximin coupling'),
      Line2D([], [], marker='s', ls='', color=AQUA, mec='white', ms=7, label='observed coupling'),
      Line2D([], [], marker=(5, 1), ls='', color=INK, ms=7, label='nonzero torus branes (×multiplicity)'),
      Line2D([], [], color=BLUE, lw=1.2, ls=(0, (3, 2)), label=r'isotopy $\phi_t$, $t=0\to1$')]
fig.legend(handles=hs, loc='lower center', ncol=5, fontsize=7.5, bbox_to_anchor=(0.5, -0.02))
fig.suptitle('Coupling polygons as moment polygons of toric surfaces (gray: entropy level sets)', x=0.01, ha='left', color=INK, fontsize=10, fontweight='bold')
fig.tight_layout(rect=(0, 0.07, 1, 1)); save(fig, 'fig_toric.pdf')

# ------------------------------------------------------------------ Fig 2: the entropy brane meets the constraint brane (2 x 2 slices)
def wt22(sub, a, b):
    m = sub[a].isin([1, 2, 3]) & sub[b].isin([1, 2, 3]) & sub.wtssps.notna(); s = sub[m]; w = s.wtssps.values
    T_ = np.zeros((2, 2)); np.add.at(T_, ((s[a].values != 1).astype(int), (s[b].values != 1).astype(int)), w); return T_ / T_.sum()
fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.2))
ax = axs[0]
for yr, col in ((1985, BLUE), (2004, ORANGE), (2024, AQUA)):
    T_ = wt22(d[d.year == yr], 'natfare', 'natcity'); f, g = T_.sum(1), T_.sum(0)
    L, U = max(0, f[0] + g[0] - 1), min(f[0], g[0]); t = np.linspace(L + 1e-4, U - 1e-4, 400)
    p = np.stack([t, f[0] - t, g[0] - t, 1 - f[0] - g[0] + t])
    xi = -(np.log(p[0]) - np.log(p[1]) - np.log(p[2]) + np.log(p[3]))         # dH along the fibre direction (circulation +1,-1,-1,+1)
    ax.plot(t, xi, color=col, lw=1.6, label=str(yr)); ax.plot(f[0] * g[0], 0, 'o', color=col, ms=6, mec='white')
    ax.plot(T_[0, 0], -(np.log(T_[0, 0]) - np.log(T_[0, 1]) - np.log(T_[1, 0]) + np.log(T_[1, 1])), 's', color=col, ms=5, mec='white')
ax.axhline(0, color=INK2, lw=1); ax.set_ylim(-6, 6); ax.set_xlabel(r'coupling $t=\alpha_{11}$ on the fibre')
ax.set_ylabel(r'$dH$ along the fibre')
ax.text(0.02, 0.04, 'horizontal axis = conormal brane\ncircle = intersection (product)\nsquare = observed', transform=ax.transAxes, color=INK2, fontsize=7.5)
ax.legend(loc='upper right', title='welfare × cities, X', title_fontsize=7.5); ax.set_title('Entropy brane over the fibre')
L2 = pd.DataFrame(S['L2'])
ax = axs[1]
for (a, b, w), col, lab in ((('natfare', 'natcity', 'X'), BLUE, 'welfare × cities (X)'), (('natcrime', 'natdrug', 'X'), ORANGE, 'crime × drugs (X)'), (('nateduc', 'natheal', 'X'), AQUA, 'health × education (X)')):
    s = L2[(L2.a == a) & (L2.b == b) & (L2.w == w)].sort_values('year'); ax.plot(s.year, s.mi, color=col, lw=1.6, marker='o', ms=3, label=lab)
ax.set_ylabel(r'$I=D_{KL}(\mathrm{observed}\,\|\,\mathrm{max\text{-}ent})$, bits'); ax.legend(loc='upper left'); ax.set_title('Distance from the intersection point')
ax.grid(axis='y'); ax.set_ylim(bottom=0)
ax = axs[2]
for (a, b, w), col, lab in ((('natfare', 'natcity', 'X'), BLUE, 'welfare × cities (X)'), (('natcrime', 'natdrug', 'X'), ORANGE, 'crime × drugs (X)'), (('nateduc', 'natheal', 'X'), AQUA, 'health × education (X)')):
    s = L2[(L2.a == a) & (L2.b == b) & (L2.w == w)].sort_values('year'); ax.plot(s.year, s.steep, color=col, lw=1.6, marker='o', ms=3, label=lab)
ax.set_ylabel('largest Fisher eigenvalue on the fibre'); ax.set_yscale('log'); ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax.set_title('Steepness of the intersection'); ax.grid(axis='y')
fig.tight_layout(); save(fig, 'fig_entropy_brane.pdf')

# ------------------------------------------------------------------ Fig 3: Fisher-Bregman dual geometry on the simplex of answers
def tern(p): p = np.asarray(p, float); return np.array([p[..., 1] + 0.5 * p[..., 2], np.sqrt(3) / 2 * p[..., 2]]).T
def law(sub, v):
    s = sub[sub[v].isin([1, 2, 3]) & sub.wtssps.notna()]; return np.array([s.wtssps[s[v] == k].sum() for k in (1, 2, 3)]) / s.wtssps.sum()
q = law(d[d.year == 2024], 'natfare'); p = law(d[d.year == 2024], 'natfarey')
ps = p.copy(); ps[1:] = q[1:] / q[1:].sum() * (1 - p[0])           # I-projection of q onto M = {r : r_1 = p_1}
KL = lambda a, b: float((a * np.log2(a / b)).sum())
fig, ax = plt.subplots(figsize=(5.6, 4.9))
tri = tern(np.eye(3)); ax.add_patch(MPoly(tri, closed=True, facecolor=SURF, edgecolor=AXIS, lw=1))
n = 220; grid = np.array([[i / n, j / n, 1 - (i + j) / n] for i in range(1, n) for j in range(1, n - i)])
kq = (grid * np.log2(grid / q)).sum(1); xy = tern(grid)
cs = ax.tricontour(xy[:, 0], xy[:, 1], kq, levels=[0.05, 0.15, 0.3, 0.6, 1.0], colors=MUTED, linewidths=0.7)
ax.clabel(cs, fmt=lambda v: f'{v:g} bit', fontsize=6.5, colors=MUTED)
for c in np.linspace(-2, 2, 9):          # e-flat coordinate lines theta_1 = log(r1/r3) = c (curves), and m-flat lines r1 = const (straight)
    r3 = np.linspace(0.001, 0.999, 300); r1 = np.exp(c) * r3; ok = r1 + r3 < 1; r = np.stack([r1[ok], 1 - r1[ok] - r3[ok], r3[ok]], 1); ax.plot(*tern(r).T, color=GRID, lw=0.7)
for c in np.linspace(0.1, 0.9, 9):
    r = np.array([[c, 1 - c, 0], [c, 0, 1 - c]]); ax.plot(*tern(r).T, color=GRID, lw=0.7, ls=':')
M = np.array([[p[0], 1 - p[0], 0], [p[0], 0, 1 - p[0]]]); ax.plot(*tern(M).T, color=INK2, lw=1.2)
s_ = np.linspace(0, 1, 100)[:, None]; eg = q ** (1 - s_) * ps ** s_; eg /= eg.sum(1, keepdims=True); ax.plot(*tern(eg).T, color=BLUE, lw=2)
ax.plot(*tern(np.stack([p, ps])).T, color=ORANGE, lw=2)
for pt, col, lab, xyt in ((q, BLUE, 'q: "welfare", 2024', (0.62, 0.42)), (p, ORANGE, 'p: "assistance to the poor", 2024', (0.42, 0.06)), (ps, INK, 'p*: I-projection of q on M', (0.36, 0.20))):
    ax.plot(*tern(pt), 'o', color=col, ms=7, mec='white', mew=1.2, zorder=5)
    ax.annotate(lab, tern(pt), xytext=xyt, textcoords='data', fontsize=7.5, color=INK2, arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.7))
for k, lab in enumerate(['too little', 'about right', 'too much']): ax.annotate(lab, tern(np.eye(3)[k]), xytext=(0, -12 if k < 2 else 6), textcoords='offset points', ha='center', fontsize=8, color=INK2)
ax.text(0.5, -0.06, f'Pythagoras: KL(p‖q) = {KL(p, q):.3f} = KL(p‖p*) {KL(p, ps):.3f} + KL(p*‖q) {KL(ps, q):.3f} bits', transform=ax.transAxes, fontsize=7.8, color=INK, ha='center')
ax.text(0.5, -0.12, 'dark line M = {r : r(too little) = p(too little)}, m-flat;  blue: e-geodesic q→p*;  orange: m-geodesic p→p*;\ncontours: KL(·‖q);  faint curves: e-coordinate lines;  dotted: m-coordinate lines', transform=ax.transAxes, fontsize=6.8, color=INK2, ha='center', va='top')
ax.set_aspect('equal'); ax.axis('off'); ax.set_title('Dual affine (Fisher–Bregman) geometry of the welfare answer, GSS 2024')
save(fig, 'fig_fisher_bregman.pdf')
FB = dict(q=q.round(4).tolist(), p=p.round(4).tolist(), pstar=ps.round(4).tolist(), KLpq=KL(p, q), KLpps=KL(p, ps), KLpsq=KL(ps, q))

# ------------------------------------------------------------------ Fig 4: tearing timeline, welfare x race, wording Y
yrs = sorted(int(y) for y in G['welfare-race']['Y']); rows = []
for yr in yrs:
    sub = d[d.year == yr]; m = sub.natfarey.isin([1, 2, 3]) & sub.natracey.isin([1, 2, 3]) & sub.wtssps.notna(); s = sub[m]; w = s.wtssps.values
    T_ = np.zeros((3, 3)); np.add.at(T_, (s.natfarey.astype(int).values - 1, s.natracey.astype(int).values - 1), w); ne = w.sum() ** 2 / (w ** 2).sum()
    St = State(T_, ne); a_lo, a_hi = event_range(St.fb[0], St.fb[1], [0]); b_lo, b_hi = event_range(St.gb[0], St.gb[1], [1, 2])
    rows.append(dict(year=yr, f=St.f[0], flo=St.fb[0][0], fhi=St.fb[1][0], g=St.g[0], glo=St.gb[0][0], ghi=St.gb[1][0],
                     wall=St.f[0] - St.g[1:].sum(), wlo=a_lo - b_hi, whi=a_hi - b_lo))
R = pd.DataFrame(rows); steps = {(s['frm'], s['to']): s['kind'] for s in G['welfare-race']['steps']['Y']}
V33 = T['vertices33_series']
fig, axs = plt.subplots(3, 1, figsize=(8, 6.2), sharex=True, gridspec_kw=dict(height_ratios=[1.3, 1.3, 0.55]))
ax = axs[0]
ax.fill_between(R.year, R.flo, R.fhi, color=BLUE, alpha=0.18, lw=0); ax.plot(R.year, R.f, color=BLUE, lw=1.6, marker='o', ms=3)
ax.fill_between(R.year, R.glo, R.ghi, color=ORANGE, alpha=0.18, lw=0); ax.plot(R.year, R.g, color=ORANGE, lw=1.6, marker='o', ms=3)
ax.text(1985, R.f.iloc[0] + 0.05, '"assistance to the poor": too little', color=INK2, fontsize=7.5)
ax.text(1985, R.g.iloc[0] - 0.08, '"assistance to blacks": too little', color=INK2, fontsize=7.5)
ax.set_ylabel('share (95% band)'); ax.grid(axis='y'); ax.set_title('Margins with simultaneous bands, GSS wording Y')
ax = axs[1]
ax.fill_between(R.year, R.wlo, R.whi, color=BLUE, alpha=0.18, lw=0); ax.plot(R.year, R.wall, color=BLUE, lw=1.6, marker='o', ms=3)
ax.axhline(0, color=INK2, lw=1); ax.set_ylabel(r'$f_1 - (g_2+g_3)$')
ax.set_title(r'Wall $f(\mathrm{too\ little}) = g(\mathrm{about\ right}) + g(\mathrm{too\ much})$; the band interval decides the side')
for (a, b), k in steps.items():
    if k == 'tear':
        for ax_ in axs[:2]: ax_.axvline((a + b) / 2, color=INK, ls='--', lw=1)
        axs[1].annotate('decided tear\n2014→2016', ((a + b) / 2, R.whi.max()), xytext=(4, -18), textcoords='offset points', fontsize=7.5, color=INK)
    elif k == 'tear?': axs[1].plot((a + b) / 2, 0, '|', color=MUTED, ms=8)
ax.grid(axis='y')
ax = axs[2]
vc = [V33.get(f'{y}|natfare|natrace|Y') for y in yrs]
ax.step(yrs, vc, where='mid', color=AQUA, lw=1.6); ax.set_yticks([9, 12, 15, 18]); ax.set_ylim(10.5, 19.5); ax.set_ylabel('vertices'); ax.grid(axis='y')
ax.set_title('Vertices of the 3×3 coupling polytope (toric 4-fold; tears are blow-ups/downs)')
ax.set_xlabel('survey year'); fig.tight_layout(); save(fig, 'fig_tearing.pdf')

# ------------------------------------------------------------------ Fig 5: Contextuality-by-Default against the naive test
L3 = pd.DataFrame(S['L3'])
fig = plt.figure(figsize=(10.5, 3.6)); gs = fig.add_gridspec(1, 3, width_ratios=[0.8, 1.2, 1.3])
ax = fig.add_subplot(gs[0]); ax.axis('off'); ax.set_title('Cyclic system of rank 2')
pos = {'A_X': (0, 1), 'B_X': (1, 1), 'A_Y': (0, 0), 'B_Y': (1, 0)}
for k, (x, y) in pos.items():
    ax.add_patch(plt.Circle((x, y), 0.17, facecolor='white', edgecolor=INK2, lw=1.2)); ax.text(x, y, f'${k[0]}_{k[2]}$', ha='center', va='center', fontsize=10, color=INK)
for a, b in (('A_X', 'B_X'), ('A_Y', 'B_Y')): ax.plot([pos[a][0] + 0.17, pos[b][0] - 0.17], [pos[a][1]] * 2, color=BLUE, lw=2)
for a, b in (('A_X', 'A_Y'), ('B_X', 'B_Y')): ax.plot([pos[a][0]] * 2, [pos[a][1] - 0.17, pos[b][1] + 0.17], color=ORANGE, lw=2, ls=(0, (3, 2)))
ax.text(0.5, 1.25, 'context X (wording)', ha='center', color=INK2, fontsize=7.5); ax.text(0.5, -0.32, 'context Y', ha='center', color=INK2, fontsize=7.5)
ax.text(-0.45, 0.5, 'connection\n(direct influence)', ha='center', va='center', color=INK2, fontsize=7.2, rotation=90)
ax.set_xlim(-0.7, 1.4); ax.set_ylim(-0.5, 1.45); ax.set_aspect('equal')
ax = fig.add_subplot(gs[1])
cd = (L3.a == 'natcrime') & (L3.b == 'natdrug')
ax.scatter(L3.naive_cf[~cd], L3.deficit[~cd], s=8, color=BLUE, alpha=0.45, lw=0, label='54 other item pairs')
ax.scatter(L3.naive_cf[cd], L3.deficit[cd], s=16, color=ORANGE, lw=0, label='crime × drugs')
ax.set_xlabel('contextual fraction of a test with identical connections'); ax.set_ylabel('canonical CbD deficit'); ax.legend(loc='upper left')
ax.set_title(f'{len(L3)} system-years: the naive test is never 0'); ax.grid(axis='y')
ax = fig.add_subplot(gs[2])
pr = sorted(((v['deff1']['p'], k) for k, v in CX.items() if not k.startswith('_')))
lab = {'natfare': 'welfare', 'natcity': 'cities', 'natrace': 'race', 'natcrime': 'crime', 'natdrug': 'drugs', 'nateduc': 'education', 'natheal': 'health',
       'natenvir': 'environment', 'natspac': 'space', 'natarms': 'defense', 'nataid': 'aid'}
top = pr[:12]; yv = np.arange(len(top))[::-1]
ax.barh(yv, [-np.log10(p) for p, _ in top], color=[ORANGE if k == 'natcrime|natdrug' else BLUE for _, k in top], height=0.62)
ax.set_yticks(yv); ax.set_yticklabels([' × '.join(lab[x] for x in k.split('|')) for _, k in top], fontsize=7.5)
ax.axvline(-np.log10(CX['_summary']['bonferroni']), color=INK, ls='--', lw=1); ax.text(-np.log10(CX['_summary']['bonferroni']), len(top) - 0.6, ' Bonferroni\n (55 pairs)', fontsize=7, color=INK, va='top')
ax.set_xlabel(r'$-\log_{10}p$, boundary null combined over 24 years'); ax.set_title('Twelve smallest of 55 pairs (design effect 1)'); ax.grid(axis='x')
fig.tight_layout(); save(fig, 'fig_cbd.pdf')

# ------------------------------------------------------------------ Fig 6: the decorated nerve K_{2,11} and its holonomy class
ITEMS = S['items']; dp = np.array(S['L1_pooled']['d']); se = np.array(S['L1_pooled']['se']); cls = dp - dp.mean()
YV = {'natspac': 'natspacy', 'natenvir': 'natenviy', 'natheal': 'nathealy', 'natcity': 'natcityy', 'natcrime': 'natcrimy', 'natdrug': 'natdrugy',
      'nateduc': 'nateducy', 'natrace': 'natracey', 'natarms': 'natarmsy', 'nataid': 'nataidy', 'natfare': 'natfarey'}
kl = []
for it in ITEMS:
    a, b = law(d, it), law(d, YV[it]); kl.append(KL(a, b))
bX = np.mean([S['L1'][str(y)]['bX'] for y in S['years']], 0); bY = np.mean([S['L1'][str(y)]['bY'] for y in S['years']], 0)
fig = plt.figure(figsize=(10.5, 4.2)); gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1])
ax = fig.add_subplot(gs[0]); ax.axis('off'); ax.set_title('Decorated nerve of the bias site: wordings × items (pooled 1985–2024)')
k = len(ITEMS); xs = np.linspace(0, 10, k); vmax = max(abs(bX).max(), abs(bY).max())
for i in range(k):
    for (wx, wy), bv in (((5, 2.0), bX[i]), ((5, -2.0), bY[i])):
        ax.plot([wx, xs[i]], [wy, 0], color=DIV(0.5 + 0.5 * bv / vmax), lw=1.8, zorder=1)
for i in range(k):
    ax.scatter(xs[i], 0, s=40 + 900 * np.sqrt(kl[i]), color='white', edgecolor=INK2, lw=1.2, zorder=3)
    ax.text(xs[i], -0.42 if i % 2 else 0.36, lab[ITEMS[i]], ha='center', va='center', fontsize=7.5, color=INK)
for (wx, wy), t in (((5, 2.0), 'wording X'), ((5, -2.0), 'wording Y')):
    ax.scatter(wx, wy, s=260, color='white', edgecolor=INK, lw=1.4, zorder=3); ax.text(wx, wy + (0.38 if wy > 0 else -0.38), t, ha='center', fontsize=8, color=INK)
ax.text(-0.5, -2.65, 'edge colour: connection b(w, i) = logit P(too little)\nblue < 0 < red;  node size: KL(P_X ‖ P_Y) of the item\n'
        r'$\beta_1 = |E|-|V|+1 = 10$ independent loops', fontsize=7, color=INK2, ha='left', va='top')
ax.set_xlim(-0.6, 10.6); ax.set_ylim(-3.6, 2.6)
ax = fig.add_subplot(gs[1]); o = np.argsort(cls); yv = np.arange(k)
ax.barh(yv, cls[o], xerr=1.96 * se[o], color=[DIV(0.5 + 0.5 * c / abs(cls).max()) for c in cls[o]], height=0.62, error_kw=dict(ecolor=INK2, lw=1))
ax.set_yticks(yv); ax.set_yticklabels([lab[ITEMS[i]] for i in o], fontsize=7.5); ax.axvline(0, color=INK2, lw=1)
ax.set_xlabel(r'class coordinate $d_i-\bar d$, $d_i=b(X,i)-b(Y,i)$'); ax.grid(axis='x')
ax.set_title(f"Holonomy class in H¹ = R¹⁰: not flat in {S['L1_pooled']['years_flat_rejected']} of 24 years")
fig.tight_layout(); save(fig, 'fig_nerve.pdf')

# ------------------------------------------------------------------ Fig 7: panel composition and path dependence
fig = plt.figure(figsize=(10.5, 3.5)); gs = fig.add_gridspec(1, 4, width_ratios=[0.8, 0.8, 1.25, 1.25])
r = PP['06/natfare']; labs3 = ['little', 'right', 'much']
for k_, (key, ttl) in enumerate((('K13', r'$K_{1\to3}$ observed'), ('K12K23', r'$K_{1\to2}K_{2\to3}$ composite'))):
    ax = fig.add_subplot(gs[k_]); M_ = np.array(r[key]); ax.imshow(M_, cmap=SEQ, vmin=0, vmax=1)
    for i in range(3):
        for j in range(3): ax.text(j, i, f'{M_[i, j]:.2f}', ha='center', va='center', fontsize=7.5, color='white' if M_[i, j] > 0.5 else INK)
    ax.set_xticks(range(3)); ax.set_yticks(range(3)); ax.set_xticklabels(labs3, fontsize=7); ax.set_yticklabels(labs3 if k_ == 0 else [], fontsize=7)
    ax.set_title(ttl, fontsize=8.5); ax.set_facecolor('white')
    for s_ in ax.spines.values(): s_.set_visible(False)
fig.text(0.02, 0.02, 'welfare, 2006–2010 panel', fontsize=7.5, color=INK2)
ax = fig.add_subplot(gs[2]); keys = list(PM.keys()); xx = np.arange(len(keys))
ax.bar(xx, [PM[k_]['gap_observed'] for k_ in keys], color=BLUE, width=0.6, label='observed gap')
ax.plot(xx, [G['panels'][k_.split('/')[0]][k_.split('/')[1]]['gap_null95'] for k_ in keys], '_', color=INK, ms=14, mew=2, label='Markov null, 95th pct')
ax.plot(xx, [PM[k_]['markov']['gap_implied'] for k_ in keys], 'D', color=ORANGE, ms=5, label='latent Markov fit')
ax.plot(xx, [PM[k_]['stable']['gap_implied'] for k_ in keys], 'o', color=AQUA, ms=5, label='stable attitude fit')
ax.set_xticks(xx); ax.set_xticklabels([f"20{k_.split('/')[0]}\n{'welf.' if k_.endswith('fare') else 'poor'}" for k_ in keys], fontsize=7)
ax.set_ylabel(r'$I(X_1;X_3\mid X_2)$, bits'); ax.set_ylim(0, 0.15); ax.legend(fontsize=6.5, loc='upper left', ncol=2); ax.set_title('Gluing gap'); ax.grid(axis='y')
ax = fig.add_subplot(gs[3]); groups = ['vote', 'party']; w_ = 0.25
for j, (m_, col, l_) in enumerate((('bits_x1', BLUE, 'wave-1 answer'), ('bits_multiset', ORANGE, 'answers, order-free'), ('bits_path', AQUA, 'ordered path'))):
    vals = [PP['pooled_' + g_]['bits_series'] - PP['pooled_' + g_][m_] for g_ in groups]
    ax.bar(np.arange(2) + (j - 1) * w_, vals, width=w_ - 0.03, color=col, label=l_)
for i, g_ in enumerate(groups):
    ax.text(i, max(PP['pooled_' + g_]['bits_series'] - PP['pooled_' + g_][m] for m in ('bits_x1', 'bits_multiset', 'bits_path')) + 0.004,
            f"order: p = {PP['pooled_' + g_]['p_path_vs_multiset']:.2f}", ha='center', fontsize=7, color=INK)
ax.set_xticks([0, 1]); ax.set_xticklabels(['2008 vote', 'party ID, wave 3']); ax.axhline(0, color=INK2, lw=1)
ax.set_ylabel('bits gained (leave-one-out)'); ax.set_ylim(0, 0.16); ax.legend(fontsize=6.5, loc='upper right'); ax.set_title('Does the order of the path predict?'); ax.grid(axis='y')
fig.tight_layout(); save(fig, 'fig_panels.pdf')

# ------------------------------------------------------------------ Fig 8: toric types over forty years
fig, axs = plt.subplots(2, 1, figsize=(8, 3.8), sharex=True)
order = ['CP^2', 'F_1 = Bl_1 CP^2', 'CP^1 x CP^1', 'Bl_2 CP^2', 'Bl_3 CP^2 (dP6)']; nice = ['CP²', 'F₁', 'CP¹×CP¹', 'Bl₂CP²', 'Bl₃CP²']
for ax, (key, ttl) in zip(axs, (('health x education (X)', 'health (too little / not) × education, wording X'), ('poor x cities (Y)', '"the poor" (too little / not) × "big cities", wording Y'))):
    ser = T[key]; ys_ = sorted(int(y) for y in ser)
    yy = [order.index(ser[str(y)]['surface']) for y in ys_]; dec = [ser[str(y)]['chamber_decided'] for y in ys_]
    ax.scatter([y for y, k in zip(ys_, dec) if k], [v for v, k in zip(yy, dec) if k], s=30, color=BLUE, zorder=3, label='chamber decided on the bands')
    ax.scatter([y for y, k in zip(ys_, dec) if not k], [v for v, k in zip(yy, dec) if not k], s=30, facecolor='white', edgecolor=BLUE, lw=1.2, zorder=3, label='not decided')
    ax.set_yticks(range(5)); ax.set_yticklabels(nice, fontsize=7.5); ax.set_title(ttl, fontsize=8.5); ax.grid(axis='y')
axs[0].legend(loc='lower right', ncol=2, fontsize=7); axs[1].set_xlabel('survey year')
fig.suptitle('The toric surface of the 2×3 coupling polygon, year by year', x=0.01, ha='left', fontsize=10, fontweight='bold', color=INK)
fig.tight_layout(); save(fig, 'fig_toric_years.pdf')
json.dump(dict(fisher_bregman=FB, kl_items={i: round(k_, 4) for i, k_ in zip(ITEMS, kl)}), open(os.path.join(H, 'figure_numbers.json'), 'w'), indent=1)
