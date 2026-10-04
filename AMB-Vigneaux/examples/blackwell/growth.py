"""Growing only frontier regions, exactly; a non-binary target; and the counterexample that says when not to.

Blocks of measurements are conditionally independent of each other given Theta (inside a block anything goes):
  parity3    X1, X2 fair coins; X3 = (Theta mod 2) xor X1 xor X2, flipped w.p. 0.1   (order-3 structure)
  copy-pair  X = signal of Theta (accuracy 0.65-0.75); X' = X flipped w.p. 0.1, cheaper
  direct     one signal of Theta (accuracy 0.6-0.8)
For k = 3 values of Theta, a 'signal' reports whether Theta = j for a block-specific j.
(1) Exactness: n = 12 (2 x [parity3, copy-pair, direct] ... ), all 4096 regions enumerated from the joint, against the
    block-wise frontier growth; for k = 2 and k = 3, three rewards each, every budget and cost price.
(2) Scaling: 12, 24, 36, 48 measurements (2^48 regions cannot be enumerated): frontier size, combinations formed, time;
    best value under a budget against myopic greedy growth.
(3) Counterexample: N fair noise, B a weak signal, C = Theta xor N. {N} <= {B} in Blackwell order, yet {N, C} reveals
    Theta and {B, C} does not. Treating every measurement as its own block (dropping N early) loses the optimum; putting
    N and C in one block (they are dependent given Theta) restores exactness.
"""
import os, sys, json, time, itertools
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "../.."))
import numpy as np
from amb_vigneaux import blackwell as B

rng = np.random.default_rng(41)


def sig_row(k, j, acc):
    """P(X = 1 | theta) for a signal of 'Theta == j' (k = 2: j = 1, the plain signal)"""
    return np.array([acc if t == j else 1 - acc for t in range(k)])


def block_table(kind, k, cost_scale=1.0):
    """returns (size, table P(x_block | theta) of shape (k, 2^size), per-measurement costs)"""
    if kind == "parity3":
        T = np.zeros((k, 8))
        for t in range(k):
            for idx, x in enumerate(itertools.product((0, 1), repeat=3)):
                par = (t % 2) ^ x[0] ^ x[1]
                T[t, idx] = 0.25 * (0.9 if x[2] == par else 0.1)
        return 3, T, [1.0, 1.0, 1.0]
    if kind == "copy-pair":
        j = int(rng.integers(1, k)) if k > 2 else 1; acc = float(rng.uniform(0.65, 0.75)); s = sig_row(k, j, acc)
        T = np.zeros((k, 4))
        for t in range(k):
            for idx, (x, xc) in enumerate(itertools.product((0, 1), repeat=2)):
                T[t, idx] = (s[t] if x else 1 - s[t]) * (0.9 if xc == x else 0.1)
        return 2, T, [1.0, 0.5]
    j = int(rng.integers(1, k)) if k > 2 else 1; acc = float(rng.uniform(0.6, 0.8)); s = sig_row(k, j, acc)
    return 1, np.stack([1 - s, s], 1), [float(rng.choice([1.0, 2.0, 3.0]))]


def block_candidates(size, T, costs, offset):
    out = []
    for r in range(size + 1):
        for S in itertools.combinations(range(size), r):
            k = T.shape[0]; tab = T.reshape((k,) + (2,) * size)
            drop = tuple(1 + i for i in range(size) if i not in S)
            M = (tab.sum(axis=drop) if drop else tab).reshape(k, -1)
            out.append((float(sum(costs[i] for i in S)), B.merge_columns(M), tuple(offset + i for i in S)))
    return out


def build(kinds, k):
    blocks, tables = [], []; off = 0
    for kd in kinds:
        size, T, costs = block_table(kd, k); blocks.append(block_candidates(size, T, costs, off)); tables.append((size, T, costs)); off += size
    return blocks, tables, off


def global_joint(tables, prior):
    k = len(prior); J = prior.copy().reshape(k, 1)
    for size, T, _ in tables:
        J = (J[:, :, None] * T[:, None, :]).reshape(k, -1)
    return J.ravel()


REW = {2: {"accuracy": np.eye(2), "asymmetric": np.array([[1.0, -2.0], [0.0, 1.0]]), "abstain": np.array([[1, 0], [0, 1], [0.7, 0.7]], float)},
       3: {"accuracy": np.eye(3), "costed": np.array([[1, -1, -2], [-0.5, 1, -0.5], [-2, -1, 1]], float), "abstain": np.vstack([np.eye(3), np.full((1, 3), 0.6)])}}


def exactness(k):
    kinds = ["parity3", "copy-pair", "direct"] * 2; blocks, tables, n = build(kinds, k)
    prior = np.full(k, 1.0 / k); J = global_joint(tables, prior); costs = np.concatenate([np.array(t[2]) for t in tables])
    t0 = time.time(); F, formed, checks = B.grow_frontier(blocks, prior); tg = time.time() - t0
    t0 = time.time(); allr = []
    for r in range(n + 1):
        for U in itertools.combinations(range(n), r):
            L, _ = B.experiment(J, U, n, k); allr.append((float(costs[list(U)].sum()), L, U))
    te = time.time() - t0
    ok = tests = 0
    for name, R in REW[k].items():
        vals_all = np.array([B.value_k(L, prior, R) for _, L, _ in allr]); c_all = np.array([c for c, _, _ in allr])
        vals_F = np.array([B.value_k(L, prior, R) for _, L, _ in F]); c_F = np.array([c for c, _, _ in F])
        for budget in np.arange(0, costs.sum() + 0.5, 0.5):
            tests += 1; ok += abs(vals_all[c_all <= budget + 1e-9].max() - vals_F[c_F <= budget + 1e-9].max()) < 1e-9
        for price in np.linspace(0, 0.1, 11):
            tests += 1; ok += abs((vals_all - price * c_all).max() - (vals_F - price * c_F).max()) < 1e-9
    print(f"(1) k={k}: n={n}, frontier by growth {len(F)} (formed {formed} combinations, {tg:.1f}s) vs {len(allr)} regions enumerated ({te:.1f}s);"
          f" optimum identical in {ok}/{tests} (reward, budget/price) cases")
    return dict(k=k, n=n, frontier=len(F), formed=formed, regions=len(allr), ok=ok, tests=tests, t_growth=tg, t_enum=te)


def scaling(k=2, reps=(1, 2, 3, 4, 5)):
    out = []
    for r in reps:
        kinds = ["parity3", "copy-pair", "direct", "direct"] * r * 1
        blocks, tables, n = build(kinds[: max(4, int(len(kinds)))], k)
        prior = np.full(k, 1.0 / k); t0 = time.time(); F, formed, checks = B.grow_frontier(blocks, prior); tg = time.time() - t0
        budget = 0.25 * sum(sum(t[2]) for t in tables); acc = REW[k]["accuracy"]
        best = max(B.value_k(L, prior, acc) for c, L, _ in F if c <= budget + 1e-9)
        # myopic greedy growth, one measurement at a time, using the block structure to evaluate exactly
        chosen = {b: () for b in range(len(blocks))}; spent = 0.0
        def region_value(ch):
            L = np.ones((k, 1))
            for b, sub in ch.items():
                Lb = next(c[1] for c in blocks[b] if c[2] == sub); L = B.product(L, Lb)
            return B.value_k(L, prior, acc)
        while True:
            opts = []
            for b, blk in enumerate(blocks):
                for c in blk:
                    if len(c[2]) == len(chosen[b]) + 1 and set(chosen[b]) <= set(c[2]):
                        extra = c[0] - next(x[0] for x in blk if x[2] == chosen[b])
                        if spent + extra <= budget + 1e-9:
                            ch = dict(chosen); ch[b] = c[2]; opts.append((region_value(ch), extra, b, c[2]))
            if not opts:
                break
            v, extra, b, sub = max(opts, key=lambda o: (o[0], -o[1])); chosen[b] = sub; spent += extra
        greedy = region_value(chosen)
        out.append(dict(n=n, blocks=len(blocks), frontier=len(F), formed=formed, seconds=tg, budget=budget, best=best, greedy=greedy))
        print(f"(2) n={n:2d} ({len(blocks)} blocks): frontier {len(F):4d}, combinations formed {formed:7d} (vs 2^{n} regions), {tg:5.1f}s;"
              f" budget {budget:.1f}: frontier best accuracy {best:.3f}, myopic greedy {greedy:.3f}", flush=True)
    return out


def counterexample():
    # variables: theta, N, Bw, C (binary); C = theta xor N exactly; Bw weak signal
    X = B.cells(4); p = np.zeros(16)
    for i, (t, nn, bw, c) in enumerate(X):
        p[i] = 0.5 * 0.5 * (0.6 if bw == t else 0.4) * (1.0 if c == (t ^ nn) else 0.0)
    prior = np.array([0.5, 0.5]); acc = np.eye(2)
    LN, _ = B.experiment(p, (0,), 3); LB, _ = B.experiment(p, (1,), 3); LNC, _ = B.experiment(p, (0, 2), 3); LBC, _ = B.experiment(p, (1, 2), 3)
    r = dict(N_le_B=B.leq_k(LN, LB, prior), NC_le_BC=B.leq_k(LNC, LBC, prior), BC_le_NC=B.leq_k(LBC, LNC, prior),
             val_N=B.value_k(LN, prior, acc), val_B=B.value_k(LB, prior, acc), val_NC=B.value_k(LNC, prior, acc), val_BC=B.value_k(LBC, prior, acc))
    # wrong blocks: each variable alone (N, B, C treated as conditionally independent) vs right blocks: {N, C}, {B}
    cost = {0: 1.0, 1: 1.0, 2: 1.0}; budget = 2.0
    def cand(vars_):
        out = []
        for rr in range(len(vars_) + 1):
            for S in itertools.combinations(vars_, rr):
                L, _ = B.experiment(p, S, 3); out.append((sum(cost[i] for i in S), L, S))
        return out
    wrongF, _, _ = B.grow_frontier([cand((0,)), cand((1,)), cand((2,))], prior)
    rightF, _, _ = B.grow_frontier([cand((0, 2)), cand((1,))], prior)
    # the 'wrong' frontier's region values must be recomputed from the true joint (its product formula is false)
    wrong_best = max(B.value_k(B.experiment(p, U, 3)[0], prior, acc) for c, L, U in wrongF if c <= budget)
    right_best = max(B.value_k(L, prior, acc) for c, L, U in rightF if c <= budget)
    r.update(wrong_blocks_best=wrong_best, right_blocks_best=right_best)
    print(f"(3) {{N}} <= {{B}}: {r['N_le_B']} (values {r['val_N']:.2f} <= {r['val_B']:.2f}); after adding C: {{N,C}} <= {{B,C}}: {r['NC_le_BC']},"
          f" {{B,C}} <= {{N,C}}: {r['BC_le_NC']} (values {r['val_NC']:.2f} vs {r['val_BC']:.2f}). Budget 2: growth with wrong blocks finds {wrong_best:.2f},"
          f" with N and C in one block {right_best:.2f}")
    return r


def plot(OUT):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ink, muted, blue, orange, green, grey = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834", "#1baf7a", "#b8b6ae"
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.4))
    for a in ax:
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        a.tick_params(colors=muted, labelsize=7)
    S = OUT["scaling"]; n = [s["n"] for s in S]
    ax[0].semilogy(n, [2.0 ** x for x in n], ":", color=grey, label="regions (2^n)")
    ax[0].semilogy(n, [s["formed"] for s in S], "-o", color=blue, ms=4, label="combinations formed by exact growth")
    ax[0].semilogy(n, [s["frontier"] for s in S], "-s", color=green, ms=4, label="final frontier")
    ax[0].set_xlabel("measurements n", fontsize=7.5, color=muted); ax[0].legend(fontsize=6.5, frameon=False)
    E = OUT["exactness"]
    ax[0].set_title("(a) exact frontier growth over conditionally independent blocks\n(optimum identical to full enumeration: " +
                    ", ".join(f"k={e['k']}: {e['ok']}/{e['tests']}" for e in E) + ")", fontsize=7.8, loc="left")
    ax[1].plot(n, [s["best"] for s in S], "-o", color=blue, ms=4, label="frontier (exact)")
    ax[1].plot(n, [s["greedy"] for s in S], "-o", color=orange, ms=4, label="myopic greedy growth")
    ax[1].set_xlabel("measurements n (budget = 25% of total cost)", fontsize=7.5, color=muted); ax[1].set_ylabel("accuracy", fontsize=7.5, color=muted)
    C = OUT["counterexample"]
    ax[1].legend(fontsize=6.5, frameon=False); ax[1].set_title(f"(b) best region under budget. Counterexample (dependent blocks):\nwrong blocks {C['wrong_blocks_best']:.2f} vs right blocks {C['right_blocks_best']:.2f}", fontsize=7.8, loc="left")
    fig.tight_layout(); fig.savefig(os.path.join(here, "growth_chart.pdf")); fig.savefig(os.path.join(here, "growth_chart.png"), dpi=150)


if __name__ == "__main__" and os.environ.get("PLOT_ONLY"):
    plot(json.load(open(os.path.join(here, "growth_results.json"))))
elif __name__ == "__main__":
    OUT = dict(exactness=[exactness(2), exactness(3)], scaling=scaling(), counterexample=counterexample())
    json.dump(json.loads(json.dumps(OUT, default=float)), open(os.path.join(here, "growth_results.json"), "w"), indent=1); plot(OUT)
