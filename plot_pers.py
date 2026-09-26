#!/usr/bin/env python3
"""PLOT_PERS -- barcodes and persistence diagrams for the two filtrations.

    python3 plot_pers.py --data /tmp/cD6 --load-P /tmp/cD6/fit.npz --out /tmp/pers
    python3 plot_pers.py --data /tmp/cH  --load-P /tmp/cH/sm.npz  --levels 1,2

Everything so far reported beta_k at a list of thresholds, which gives the
DIMENSION of the homology at each step but not the features themselves: it
cannot tell one long-lived cycle from several short ones. This computes actual
persistence pairs by standard GF(2) matrix reduction over the filtered complex,
so each feature gets a birth and a death, and draws:

  barcode            one horizontal bar per feature, H0 and H1
  persistence diagram   (birth, death) points, H0 and H1
  a row per level, both filtrations side by side

TWO FILTRATIONS, SAME LEVEL
---------------------------
  ROW SIDE    exact Bregman (KL) Cech on the V rule rows. A simplex enters at
              the radius of the smallest enclosing dual KL ball. H0 bars are
              symbols merging; the death of an H0 bar is the radius at which
              two symbol clusters become indistinguishable.
  BLOCK SIDE  Dowker complex of the hypergraph of subsets whose interaction
              information clears a threshold. Filtration runs from high |I| to
              low, so a decreasing threshold is the natural "time"; we negate
              it so that birth <= death as usual. H0 bars are constituents
              merging into larger ones.

The Rips filtration is drawn alongside the Cech one for the rows, because the
difference between them is the point of Edelsbrunner and Wagner's Result 2 and
is visible directly in the diagram.
"""
import json, math, argparse, itertools, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/tmp/cD6")
ap.add_argument("--load-P", default="")
ap.add_argument("--levels", default="")
ap.add_argument("--n", type=int, default=20000)
ap.add_argument("--iters", type=int, default=1500)
ap.add_argument("--out", default="/tmp/pers")
ap.add_argument("--chunk", type=int, default=4096)
a = ap.parse_args()

M = json.load(open(f"{a.data}/rhm_meta.json"))
L, V, SEQ, NLEAF = M["depth"], M["nsym"], M["seq_len"], M["nleaf"]
tr = np.array(json.load(open(f"{a.data}/train_ids.json")), dtype=np.int64)
X0 = tr[:(len(tr)//SEQ)*SEQ].reshape(-1, SEQ)[:a.n]
def card(l): return NLEAF if l == 1 else V
P = {}
for l in range(1, L+1):
    t = np.zeros((V, card(l), card(l)))
    for s, r in M["rules_all"][str(l)].items():
        for (x, y) in r: t[int(s), x, y] += 1.0/len(r)
    P[l] = t
src = "true grammar"
if a.load_P:
    z = np.load(a.load_P); P = {l: z[f"P{l}"] for l in range(1, L+1)}
    src = a.load_P.split("/")[-1]
levels = [int(x) for x in a.levels.split(",")] if a.levels else list(range(1, L+1))
EPS = 1e-12
LAST_PAIRW, LAST_NB = None, 0

# ============================================================ persistence
def persistence(simplices):
    """simplices: list of (value, tuple_of_vertices). Standard GF(2) reduction.
    Returns [(dim, birth, death)], death = inf for essential classes."""
    S = sorted(simplices, key=lambda x: (x[0], len(x[1]), x[1]))
    idx = {s: i for i, (_, s) in enumerate(S)}
    val = [v for v, _ in S]
    cols = []
    for i, (v, s) in enumerate(S):
        if len(s) == 1: cols.append(set())
        else: cols.append({idx[f] for f in itertools.combinations(s, len(s)-1)})
    low = {}; pairs = []
    for j in range(len(S)):
        col = cols[j]
        while col:
            l = max(col)
            if l not in low: break
            col ^= cols[low[l]]
        if col:
            l = max(col); low[l] = j; cols[j] = col
            if val[j] > val[l] + 1e-15:
                pairs.append((len(S[l][1])-1, val[l], val[j]))
        else:
            cols[j] = col
    paired = {p for p in low} | {j for j in low.values()}
    for j in range(len(S)):
        if j not in paired:
            pairs.append((len(S[j][1])-1, val[j], math.inf))
    # The complex is truncated at dimension 2, so every 2-simplex is unpaired
    # for want of a 3-simplex to kill it. Those are an artefact of truncation,
    # not topology: report H0 and H1 only.
    return [p for p in pairs if p[0] <= 1]

# ============================================================ row side
def kl(p, q):
    m = p > EPS
    return float((p[m]*np.log(p[m]/np.maximum(q[m], EPS))).sum())
def one_centre(pts):
    c = pts.mean(0); c /= c.sum(); best = max(kl(p, c) for p in pts)
    for i in range(1, a.iters+1):
        d = np.array([kl(p, c) for p in pts]); j = int(d.argmax())
        c = (1-1/(i+1))*c + (1/(i+1))*pts[j]; c = np.maximum(c, 0); c /= c.sum()
        r = max(kl(p, c) for p in pts)
        if r < best: best = r
    return best
def row_filtration(rows):
    n = len(rows); simp = [(0.0, (i,)) for i in range(n)]
    rho = {}
    for k in (2, 3):
        for S in itertools.combinations(range(n), k):
            rho[S] = one_centre(rows[list(S)])
    for S, v in rho.items(): simp.append((v, S))
    ed = {S: v for S, v in rho.items() if len(S) == 2}
    rips = [(0.0, (i,)) for i in range(n)]
    for S, v in ed.items(): rips.append((v, S))
    for S in (S for S in rho if len(S) == 3):
        rips.append((max(ed[(S[0], S[1])], ed[(S[0], S[2])], ed[(S[1], S[2])]), S))
    return simp, rips

# ============================================================ block side
def up(Pl, lo, hi):
    B, n, cx = lo.shape; cy = hi.shape[2]; p = Pl.shape[0]
    lo2, hi2 = lo.reshape(-1, cx), hi.reshape(-1, cy)
    Pt = Pl.transpose(1, 0, 2).reshape(cx, p*cy)
    out = np.empty((lo2.shape[0], p))
    for s in range(0, lo2.shape[0], a.chunk):
        t = (lo2[s:s+a.chunk] @ Pt).reshape(-1, p, cy)
        out[s:s+a.chunk] = (t*hi2[s:s+a.chunk, None, :]).sum(-1)
    return out.reshape(B, n, p)
def Hc(c):
    p = c/c.sum(); m = p > 0
    return float(-(p[m]*np.log(p[m])).sum())
def block_filtration(level):
    nb = SEQ >> level
    if nb < 3 or nb > 10: return None
    o = np.zeros(X0.shape + (NLEAF,)); np.put_along_axis(o, X0[..., None], 1.0, -1)
    t = o
    for l in range(1, level+1): t = up(P[l], t[:, 0::2], t[:, 1::2])
    codes = t.argmax(-1); n = len(codes)
    h1 = {i: Hc(np.bincount(codes[:, i], minlength=V).astype(float)) for i in range(nb)}
    h2 = {}
    for i, j in itertools.combinations(range(nb), 2):
        h2[(i, j)] = Hc(np.bincount(codes[:, i]*V+codes[:, j], minlength=V*V).astype(float))
    W = {}
    for i, j in itertools.combinations(range(nb), 2):
        W[(i, j)] = max(h1[i]+h1[j]-h2[(i, j)] - (V-1)**2/(2*n), 0.0)
    for i, j, k in itertools.combinations(range(nb), 3):
        h3 = Hc(np.bincount(codes[:, i]*V*V+codes[:, j]*V+codes[:, k],
                            minlength=V**3).astype(float))
        W[(i, j, k)] = abs(h1[i]+h1[j]+h1[k]-h2[(i, j)]-h2[(i, k)]-h2[(j, k)]+h3)
    global LAST_PAIRW, LAST_NB
    LAST_PAIRW = {k: v for k, v in W.items() if len(k) == 2}
    LAST_NB = nb
    # Dowker: a hyperedge contributes its full simplex, entering at -|I| so
    # that the filtration increases (high interaction = early)
    simp = [(-max(W.values()), (i,)) for i in range(nb)]
    face = {}
    for S, v in W.items():
        for k in (2, 3):
            for f in itertools.combinations(sorted(S), k):
                face[f] = max(face.get(f, -math.inf), v)
    for f, v in face.items(): simp.append((-v, f))
    return simp

# ====================================================== merge tree (dendrogram)
def merge_tree(n, weight_of_pair):
    """Single-linkage merge tree: repeatedly join the two components whose
    closest members are nearest. Persistence records WHEN components die; this
    records WHICH ones joined, which is where the tree actually lives."""
    order = sorted(weight_of_pair.items(), key=lambda kv: kv[1])
    par = list(range(n)); size = [1]*n
    node = list(range(n))            # current dendrogram node id per component
    nxt = n; merges = []             # (left_node, right_node, height, members)
    members = {i: [i] for i in range(n)}
    def find(x):
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    for (i, j), w in order:
        ri, rj = find(i), find(j)
        if ri == rj: continue
        li, lj = node[ri], node[rj]
        mem = members[li] + members[lj]
        merges.append((li, lj, float(w), mem))
        par[ri] = rj; node[rj] = nxt; members[nxt] = mem; nxt += 1
        if len(mem) == n: break
    return merges

def true_tree_pairs(nb, level):
    """under the grammar, blocks 2k and 2k+1 share a parent, then those pairs
    share a grandparent, and so on -- the positional tree over blocks"""
    lvl = {}
    for i in range(nb):
        for j in range(i+1, nb):
            d = 1
            while (i >> d) != (j >> d): d += 1
            lvl[(i, j)] = d
    return lvl

def draw_dendrogram(ax, merges, n, title, true_lvl=None):
    """merges[k] = (left_id, right_id, height, members); the k-th merge creates
    node id n+k. Leaves are laid out in the order the tree visits them, so the
    links do not cross."""
    kids = {}
    for k, (li, lj, h, mem) in enumerate(merges): kids[n+k] = (li, lj, h)
    roots = [n+len(merges)-1] if merges else list(range(n))
    leaf_order = []
    def visit(v):
        if v < n: leaf_order.append(v); return
        li, lj, _ = kids[v]; visit(li); visit(lj)
    for r in roots: visit(r)
    for v in range(n):
        if v not in leaf_order: leaf_order.append(v)
    xpos = {v: i for i, v in enumerate(leaf_order)}
    base = min((h for _, _, h, _ in merges), default=0.0)
    hgt = {v: base for v in range(n)}
    for k, (li, lj, h, mem) in enumerate(merges):
        xi, xj = xpos[li], xpos[lj]
        ax.plot([xi, xi], [hgt[li], h], color="#2b6cb0", lw=1.4)
        ax.plot([xj, xj], [hgt[lj], h], color="#2b6cb0", lw=1.4)
        ax.plot([xi, xj], [h, h], color="#2b6cb0", lw=1.4)
        xpos[n+k] = 0.5*(xi+xj); hgt[n+k] = h
    ok = tot = 0
    if true_lvl is not None:
        for (li, lj, h, mem) in merges:
            if len(mem) == 2:
                tot += 1
                if true_lvl.get(tuple(sorted(mem)), 99) == 1: ok += 1
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("block (tree order)", fontsize=8)
    ax.set_ylabel("merge height  (-|I|)", fontsize=8)
    ax.set_xticks(range(n)); ax.set_xticklabels([str(v) for v in leaf_order],
                                                fontsize=7)
    ax.tick_params(labelsize=7)
    if true_lvl is not None:
        ax.text(0.02, 0.96, f"sibling merges correct: {ok}/{tot}",
                transform=ax.transAxes, fontsize=8, va="top")
    return ok, tot

# ============================================================ drawing
def draw(ax_bar, ax_dia, pairs, title, xlabel):
    fin = [p for p in pairs if p[2] < math.inf]
    ess = [p for p in pairs if p[2] == math.inf]
    # the Dowker filtration runs over negative values (-|I_k|), so the axes
    # must come from the data rather than assume a range starting at zero
    lo = min([p[1] for p in pairs], default=0.0)
    hi = max([p[2] for p in fin], default=lo + 1.0)
    span = max(hi - lo, 1e-9)
    cap = hi + 0.15*span
    col = {0: "#2b6cb0", 1: "#c05621", 2: "#2f855a"}
    order = sorted(fin, key=lambda p: (p[0], p[1], p[2])) + \
            sorted(ess, key=lambda p: (p[0], p[1]))
    for y, (d, b, dd) in enumerate(order):
        end = cap if dd == math.inf else dd
        ax_bar.plot([b, end], [y, y], lw=2.2, color=col.get(d, "k"),
                    solid_capstyle="butt")
        if dd == math.inf:
            ax_bar.plot([end], [y], marker=">", ms=5, color=col.get(d, "k"))
    ax_bar.set_title(title, fontsize=9)
    ax_bar.set_xlabel(xlabel, fontsize=8)
    ax_bar.set_ylabel("feature", fontsize=8)
    ax_bar.set_xlim(lo - 0.03*span, cap)
    ax_bar.tick_params(labelsize=7)
    for d in sorted({p[0] for p in pairs}):
        pts = [(b, (cap if dd == math.inf else dd)) for dd_, b, dd in
               [(x[0], x[1], x[2]) for x in pairs if x[0] == d] for dd_ in [d]]
        if pts:
            ax_dia.scatter([p[0] for p in pts], [p[1] for p in pts], s=22,
                           color=col.get(d, "k"), label=f"H{d}",
                           edgecolors="white", linewidths=0.4, zorder=3)
    ax_dia.plot([lo, cap], [lo, cap], color="0.6", lw=0.8, zorder=1)
    ax_dia.axhline(cap, color="0.8", lw=0.8, ls=":", zorder=1)
    ax_dia.set_xlim(lo - 0.03*span, cap)
    ax_dia.set_ylim(lo - 0.03*span, cap + 0.05*span)
    ax_dia.set_xlabel("birth", fontsize=8); ax_dia.set_ylabel("death", fontsize=8)
    ax_dia.set_title("persistence diagram", fontsize=9)
    ax_dia.tick_params(labelsize=7); ax_dia.legend(fontsize=7, loc="lower right")

def summarise(name, pairs):
    fin = [p for p in pairs if p[2] < math.inf]
    for d in sorted({p[0] for p in pairs}):
        f = [p for p in fin if p[0] == d]
        e = sum(1 for p in pairs if p[0] == d and p[2] == math.inf)
        if f:
            longest = max(f, key=lambda p: p[2]-p[1])
            print(f"      {name:<12} H{d}: {len(f)} finite bars, {e} essential, "
                  f"longest {longest[2]-longest[1]:.4f} "
                  f"(birth {longest[1]:.4f}, death {longest[2]:.4f})")
        else:
            print(f"      {name:<12} H{d}: 0 finite bars, {e} essential")

print(f"  plot_pers: model = {src}, levels {levels}")
for l in levels:
    rows = P[l].reshape(V, -1)
    cech, rips = row_filtration(rows)
    pc, pr = persistence(cech), persistence(rips)
    blk = block_filtration(l)
    pb = persistence(blk) if blk else None
    has_tree = pb is not None and LAST_PAIRW is not None
    ncol = (4 if has_tree else 3) if pb else 2
    fig, axes = plt.subplots(2, ncol, figsize=(4.2*ncol, 6.4))
    draw(axes[0, 0], axes[1, 0], pc, f"level {l}: rows, Bregman-Cech", "KL radius")
    draw(axes[0, 1], axes[1, 1], pr, f"level {l}: rows, Rips", "KL radius")
    if pb:
        draw(axes[0, 2], axes[1, 2], pb, f"level {l}: blocks, Dowker", "-|I_k|")
    if has_tree:
        nb = LAST_NB
        # merge on DECREASING interaction: strongest coupling joins first
        wpair = {k: -v for k, v in LAST_PAIRW.items()}
        mg = merge_tree(nb, wpair)
        tl = true_tree_pairs(nb, l)
        ok, tot = draw_dendrogram(axes[0, 3], mg, nb,
                                  f"level {l}: block merge tree", tl)
        axes[1, 3].axis("off")
        lines = [f"recovered merge order (height = -|I|):"]
        for (li, lj, h, mem) in mg[:8]:
            lines.append(f"   join {sorted(mem)}  at {h:+.4f}")
        lines.append("")
        lines.append("true tree: blocks 2k, 2k+1 share a parent;")
        lines.append("those pairs share a grandparent, and so on.")
        lines.append(f"first-level sibling merges correct: {ok}/{tot}")
        axes[1, 3].text(0.02, 0.98, "\n".join(lines), va="top", fontsize=7.5,
                        family="monospace", transform=axes[1, 3].transAxes)
        print(f"      merge tree   first-level sibling merges correct: {ok}/{tot}")
    fig.suptitle(f"{src}  --  level {l}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fn = f"{a.out}_level{l}.png"; fig.savefig(fn, dpi=140); plt.close(fig)
    print(f"\n    level {l}  ->  {fn}")
    summarise("Cech rows", pc); summarise("Rips rows", pr)
    if pb: summarise("Dowker blk", pb)
print("\n  H0 bars are merges: a bar dies when its component joins another.")
print("  H1 bars are cycles. Comparing the Cech and Rips panels shows directly")
print("  what Edelsbrunner-Wagner Result 2 asserts: Rips fills triangles as soon")
print("  as their edges exist, so its H1 bars are shorter or absent.")
