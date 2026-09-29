"""Synthetic test of the hierarchy loop test with graded arity (amb_vigneaux.padic_loops).

Tree: depth 5, each internal node draws an arity in {2,3,4,5} (the variable arity of Corpus V); leaves carry
digit codes (a p-adic code with p = 5). Two sources report exact meeting-depth claims on random leaf pairs.
Each claim carries the source and the ARITY of the meeting node as the source sees it (observable grade).
  THEOREM CHECK  a node-wise distortion (all claims meeting at odd-arity nodes shifted one level deeper) -> 0 violations.
  PLANT          source B misplaces some leaves: at level L it moves the leaf to a sibling subtree, but only where
                 the node at level L has ODD arity (graded plant). Source A is correct.
  NOISE          every claim's depth is perturbed by +/-1 with probability eps (both sources), creating spurious loops.
Statistics: violations (total and by reported arity grade), fraction whose witness cycle touches a misplaced
leaf (localisation), detection rate vs the null (no misplacement) 95th percentile.
"""
import json, collections, warnings
warnings.filterwarnings('ignore')
import numpy as np
from amb_vigneaux.padic_loops import random_tree, lca_depth, hierarchy_violations

rng = np.random.default_rng(0)


def arity_at(arity, code, d):
    return arity.get(tuple(code[:d]), 0)


def make_claims(codes_true, codes_B, arity, leaves, n_claims, eps, rng, node_shift=False):
    claims = []
    for src, codes in (("A", codes_true), ("B", codes_B)):
        for _ in range(n_claims):
            i, j = rng.choice(len(leaves), 2, replace=False)
            a, b = codes[leaves[i]], codes[leaves[j]]
            d = lca_depth(a, b); k = arity_at(arity, a, d)
            if node_shift and k % 2 == 1:
                d += 1
            if eps and rng.random() < eps:
                d = max(0, d + int(rng.choice([-1, 1])))
            claims.append((int(i), int(j), int(d), (src, int(k))))
    return claims


def one_run(n_leaves=80, n_claims=200, n_mis=4, level=2, eps=0.0, graded=True, node_shift=False, seed=0):
    r = np.random.default_rng(seed)
    codes, arity = random_tree(5, r)
    idx = r.choice(len(codes), n_leaves, replace=False)
    leaves = list(range(n_leaves)); cmap = {i: codes[idx[i]] for i in leaves}
    codes_B = dict(cmap); mis = set()
    cand = [i for i in leaves if (not graded) or arity_at(arity, cmap[i], level) % 2 == 1]
    for i in r.permutation(cand)[:n_mis]:
        c = list(cmap[i]); k = arity_at(arity, c, level)
        c[level] = (c[level] + 1 + r.integers(k - 1)) % k          # a different child at that level
        # keep deeper digits valid in the new subtree
        pref = tuple(c[:level + 1])
        for d in range(level + 1, len(c)):
            kk = arity.get(pref, 2); c[d] = c[d] % kk; pref = pref + (c[d],)
        codes_B[i] = tuple(c); mis.add(int(i))
    claims = make_claims(cmap, codes_B, arity, leaves, n_claims, eps, r, node_shift)
    bad = hierarchy_violations(n_leaves, claims)
    grades = collections.Counter("odd" if c[3][1] % 2 else "even" for c in bad)
    touch = np.mean([bool((set(c[4]) | {c[0], c[1]}) & mis) for c in bad]) if bad and mis else float("nan")
    # localisation: rank leaves by violated claims whose witness cycle mixes the two sources
    src_of = collections.defaultdict(set)
    for c in claims:
        src_of[(min(c[0], c[1]), max(c[0], c[1]), c[2])].add(c[3][0])
    score = collections.Counter()
    for c in bad:
        path = c[4]
        srcs = {c[3][0]}
        for a, b in zip(path, path[1:]):
            for cc in claims:
                if {cc[0], cc[1]} == {a, b}:
                    srcs.add(cc[3][0]); break
        if len(srcs) == 2:
            for v in set(path) | {c[0], c[1]}:
                score[v] += 1
    top = [v for v, _ in score.most_common(max(1, len(mis)))]
    prec = len(set(top) & mis) / max(1, len(mis)) if mis else float("nan")
    return len(bad), grades, touch, prec


if __name__ == "__main__":
    out = {}
    v, g, _, _ = one_run(n_mis=0, node_shift=True)
    print(f"THEOREM CHECK node-wise distortion (odd-arity meeting nodes shifted): violations = {v}")
    out["node_shift_violations"] = v
    for eps in (0.0, 0.05, 0.10):
        out[f"eps={eps}"] = {}
        print(f"\neps = {eps}")
        for n_claims in (100, 200, 400):
            null = [one_run(n_claims=n_claims, n_mis=0, eps=eps, seed=s)[0] for s in range(100)]
            thr = np.percentile(null, 95)
            print(f"   claims/source {n_claims}: null violations mean {np.mean(null):.1f}, 95th pct {thr:.0f}")
            for level in (1, 3):
                runs = [one_run(n_claims=n_claims, level=level, eps=eps, seed=1000 + s) for s in range(100)]
                det = np.mean([r[0] > thr for r in runs])
                loc = np.nanmean([r[2] for r in runs])
                gr = sum((r[1] for r in runs), collections.Counter())
                odd = gr["odd"] / max(1, sum(gr.values()))
                prec = float(np.nanmean([r[3] for r in runs]))
                out[f"eps={eps}"][f"claims={n_claims},level={level}"] = dict(detect=float(det), localise=float(loc), frac_odd_grade=float(odd), null_95=float(thr), precision_top=prec)
                print(f"      misplacement level {level}: detect {det:.2f}  top-ranked leaves = misplaced {prec:.2f}  "
                      f"witness touches misplaced {loc:.2f}  violations in odd grade {odd:.2f}")
    # grade baseline: fraction of odd-grade claims overall (to compare localisation by grade)
    r = np.random.default_rng(5); codes, arity = random_tree(5, r)
    odd_nodes = np.mean([k % 2 for k in arity.values()])
    print(f"\nbaseline: fraction of internal nodes with odd arity = {odd_nodes:.2f}")
    out["odd_node_fraction"] = float(odd_nodes)
    json.dump(out, open("examples/hierarchy_loops.json", "w"), indent=1)
