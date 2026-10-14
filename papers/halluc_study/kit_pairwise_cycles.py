"""kit_pairwise_cycles.py -- Study 3 of PREREG_hallucination.md (needs your model; nothing here calls a model).

Design: for triples of related binary claims (a, b, c), ask the model about each PAIR in a separate prompt (contexts ab, bc, ac),
n times each, and record the two binary answers. Input: JSONL lines {"triple": <id>, "context": "ab"|"bc"|"ac", "x": 0|1, "y": 0|1}.

For each triple the pairwise correlations E_ab, E_bc, E_ac (E = P(equal) - P(differ)) are estimated per context, and the four 3-cycle
inequalities sum gamma_ij E_ij <= 1 (gamma with an odd number of -1s; Conditionals Thm 4.8 at tau = 0) are evaluated. The family is
obstructed (no joint law: a hallucination in the sense of Section 16) iff one fails. Reported: the largest violation
V = max_gamma (sum gamma E - 1) with a 95% bootstrap CI (resampling answers within each context; B = 2000, seed 20261013).
For the symmetric model of Section 16, V = 2 (1 - 3 eps), twice the excess of Phi = sum P(X_i != X_j) over 2.

  python kit_pairwise_cycles.py score --answers answers.jsonl --out study3_results.json
  python kit_pairwise_cycles.py smoke     (synthetic pairwise answers from the symmetric model at eps = 0.25 and eps = 0.45)
"""
import json, sys, argparse, itertools
import numpy as np
SEED, B = 20261013, 2000
GAMMAS = [g for g in itertools.product((1, -1), repeat=3) if sum(1 for x in g if x == -1) % 2 == 1]

def E(pairs): pairs = np.asarray(pairs); return float(np.mean(np.where(pairs[:, 0] == pairs[:, 1], 1.0, -1.0)))
def violation(ctx):
    e = [E(ctx["ab"]), E(ctx["bc"]), E(ctx["ac"])]
    return max(sum(g * x for g, x in zip(gm, e)) - 1 for gm in GAMMAS), e
def score(rows):
    by = {}
    for r in rows: by.setdefault(r["triple"], {"ab": [], "bc": [], "ac": []})[r["context"]].append((r["x"], r["y"]))
    rng = np.random.default_rng(SEED); out = {}
    for t, ctx in by.items():
        v, e = violation(ctx); bs = []
        for _ in range(B):
            res = {k: [ctx[k][i] for i in rng.integers(0, len(ctx[k]), len(ctx[k]))] for k in ctx}; bs.append(violation(res)[0])
        out[str(t)] = dict(E=e, violation=v, lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)), obstructed=bool(v > 0))
    return out
def simulate(eps, n, rng, triple):
    rows = []
    for c in ("ab", "bc", "ac"):
        for _ in range(n):
            x = int(rng.random() < 0.5); y = x if rng.random() < eps else 1 - x
            rows.append({"triple": triple, "context": c, "x": x, "y": y})
    return rows
def smoke():
    rng = np.random.default_rng(3)
    rows = simulate(0.25, 4000, rng, "eps0.25") + simulate(0.45, 4000, rng, "eps0.45")
    r = score(rows); print(json.dumps(r, indent=1))
    assert r["eps0.25"]["obstructed"] and r["eps0.25"]["lo"] > 0 and abs(r["eps0.25"]["violation"] - 0.5) < 0.06
    assert not r["eps0.45"]["obstructed"]
    print("smoke test passed (V = 2(1-3eps): 0.5 at eps = 0.25; no violation at eps = 0.45)")
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--answers"); ap.add_argument("--out"); a = ap.parse_args()
    if a.cmd == "score":
        r = score([json.loads(l) for l in open(a.answers) if l.strip()]); json.dump(r, open(a.out, "w"), indent=1); print(r)
    elif a.cmd == "smoke": smoke()
