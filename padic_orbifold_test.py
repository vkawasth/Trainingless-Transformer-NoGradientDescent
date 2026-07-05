"""
padic_orbifold_test.py
=======================
Tests two hypotheses about Phase 3 basin structure:

1. P-ADIC STRUCTURE: Do token frequencies have p-adic structure
   that creates strata in the loss landscape?
   If yes: the optimizer could descend p-adic strata instead of
   blind gradient descent.

2. ORBIFOLD LOWER BOUND: Is the 120-step Phase 3 duration
   a topological invariant from the A_n Coxeter number?
   For A_4 (rank-3 monodromy, 6-layer transformer):
   Coxeter number h=5, |W(A_4)|=120.
   Minimum traversal: 120/h = 24 steps?

3. HOMOLOGY TEST: Does the loss landscape have nontrivial H_1
   (loops) that force the optimizer to wind around them?

Usage: python padic_orbifold_test.py
"""

import numpy as np
import json
import collections
import math

# ── 1. P-ADIC STRUCTURE ──────────────────────────────────────────────────────

def p_adic_valuation(n, p):
    """v_p(n) = largest k such that p^k divides n."""
    if n == 0: return float('inf')
    k = 0
    while n % p == 0:
        n //= p
        k += 1
    return k


def test_padic_structure(train_ids_path, vocab_path):
    print("="*60)
    print("  1. P-ADIC STRUCTURE OF CORPUS")
    print("="*60)

    with open(train_ids_path) as f:
        ids = list(map(int, json.load(f)))
    with open(vocab_path) as f:
        vocab = json.load(f)
    V = len(vocab) if isinstance(vocab, list) else len(vocab)

    freq = collections.Counter(ids)
    total = sum(freq.values())

    print(f"\n  Vocabulary: V={V}, total tokens={total}")
    print(f"  V = {V} = ", end="")
    # Factorize V
    n = V
    factors = []
    for p in range(2, 100):
        while n % p == 0:
            factors.append(p)
            n //= p
    if n > 1: factors.append(n)
    print(" × ".join(map(str, factors)))
    print()

    for p in [2, 3, 5, 7, 11]:
        valuations = [p_adic_valuation(freq.get(t, 1), p) for t in range(V)]
        strata = collections.Counter(valuations)
        total_v = sum(strata.values())
        H = -sum((c/total_v)*math.log2(c/total_v+1e-10)
                 for c in strata.values())

        # Expected entropy for uniform distribution over strata
        # If frequencies are random, stratum sizes should follow geometric distribution
        max_k = max(strata.keys()) if strata else 0
        H_geometric = -sum(
            (1-1/p)*(1/p)**k * math.log2((1-1/p)*(1/p)**k + 1e-10)
            for k in range(max_k+1))

        structured = H < H_geometric * 0.8
        print(f"  p={p:2d}: strata={dict(sorted(strata.items())[:5])}")
        print(f"       H={H:.3f} bits, H_geometric={H_geometric:.3f}  "
              f"{'STRUCTURED' if structured else 'random'}")

    print()
    print("  INTERPRETATION:")
    print("  If H << H_geometric: frequencies have p-adic structure")
    print("  → tokens cluster into p-adic strata")
    print("  → loss landscape may have p-adic separation")
    print("  → optimizer could descend strata instead of blind GD")


# ── 2. ORBIFOLD LOWER BOUND ──────────────────────────────────────────────────

def test_orbifold_lower_bound(N_layers=6, rank_monodromy=3):
    """
    For a transformer with N_layers and rank-r monodromy:
    The A_{n} Dynkin diagram has:
      n = rank_monodromy = 3 (from Test C: rank-3 effective dynamics)
      Coxeter number h(A_n) = n+1 = 4
      Weyl group order |W(A_n)| = (n+1)! = 24

    The orbifold Stab(F)/W(A_n) has:
      Minimum traversal = |W(A_n)| / h(A_n) = 24/4 = 6 steps?
      OR: minimum traversal = h × N_pairs = 4 × 5 = 20 steps?

    For A_4 (if we use N_layers-1=5 pairs):
      h(A_4) = 5, |W(A_4)| = 120
      Minimum traversal = 120/5 = 24 steps?
      OR: 120 steps total (= |W(A_4)|)?

    The 120-step Phase 3 matches |W(A_4)| = 5! = 120.
    This could be coincidence or topological lower bound.
    """
    print("="*60)
    print("  2. ORBIFOLD LOWER BOUND TEST")
    print("="*60)

    n_pairs = N_layers - 1  # 5 pairs for 6 layers

    # A_n Dynkin diagram data
    for n in range(2, 8):
        h = n + 1                    # Coxeter number of A_n
        W_order = math.factorial(n+1) # |W(A_n)| = (n+1)!
        h_check = W_order / math.factorial(n)  # should equal h
        print(f"  A_{n}: h={h}, |W|={W_order}, "
              f"min_traversal_1={W_order//h}, "
              f"min_traversal_2={h*n_pairs}")

    print()
    print(f"  Our setting: N_layers={N_layers}, n_pairs={n_pairs}")
    print(f"  Rank-3 monodromy → A_3 or A_4?")
    print()
    print(f"  A_3: h=4, |W|=24,  min_trav=6,  h×n_pairs=20")
    print(f"  A_4: h=5, |W|=120, min_trav=24, h×n_pairs=25")
    print(f"  A_5: h=6, |W|=720, min_trav=120, h×n_pairs=30")
    print()
    print(f"  OBSERVED: Phase 3 takes exactly 112-120 steps")
    print(f"  MATCH: |W(A_5)| = 720/6 = 120 ← A_5 min traversal")
    print(f"  OR:    |W(A_4)| = 120 ← A_4 Weyl group order")
    print()
    print(f"  HYPOTHESIS: 120 steps = topological lower bound from")
    print(f"  A_4 Weyl group acting on Stab(F).")
    print(f"  If true: Phase 3 CANNOT be reduced below ~24 steps")
    print(f"  (= |W(A_4)|/h(A_4) = 120/5 = 24)")
    print(f"  regardless of optimizer or learning rate.")

    print()
    print(f"  FALSIFICATION TEST:")
    print(f"  Run Phase 3 with very large LR (LR×50).")
    print(f"  If it still takes ~24+ steps: topological lower bound confirmed.")
    print(f"  If it converges in <10 steps: just a numerical artifact.")


# ── 3. HOMOLOGY TEST ─────────────────────────────────────────────────────────

def test_homology_structure():
    """
    The loss landscape H_1 (loops) can be detected by:
    - Running gradient descent from multiple random initializations
    - Checking if trajectories wind around a common center
    - If yes: there's a topological obstruction (H_1 nontrivial)

    Without running multiple trajectories, we can estimate from:
    - The 8 wall crossings in Phase 3 (sign changes of phases)
    - Each sign change corresponds to a loop crossing
    - 8 crossings over 112 steps = winding number ≈ 8/5 pairs ≈ 1.6

    If winding number ≈ integer: loop is topological
    If winding number ≠ integer: loop is just numerical noise
    """
    print("="*60)
    print("  3. H_1 LOOP STRUCTURE ESTIMATE")
    print("="*60)

    # From config A: 8 wall crossings over 112 steps, 5 layer pairs
    n_crossings = 8
    n_steps     = 112
    n_pairs     = 5

    winding_per_pair = n_crossings / n_pairs
    winding_total    = n_crossings / (2 * n_pairs)  # full loop = 2 crossings

    print(f"\n  Wall crossings in Phase 3 (config A): {n_crossings}")
    print(f"  Steps: {n_steps}, Layer pairs: {n_pairs}")
    print(f"  Crossings per pair: {winding_per_pair:.2f}")
    print(f"  Estimated winding number: {winding_total:.2f}")
    print()

    if abs(winding_total - round(winding_total)) < 0.3:
        print(f"  Winding number ≈ {round(winding_total)} (integer)")
        print(f"  → H_1 may be nontrivial: topological loop in landscape")
        print(f"  → Minimum path must wind around the loop once")
    else:
        print(f"  Winding number ≈ {winding_total:.2f} (non-integer)")
        print(f"  → H_1 likely trivial: crossings are numerical noise")

    print()
    print(f"  IMPLICATION:")
    print(f"  If H_1 ≠ 0: Phase 3 minimum length = loop circumference")
    print(f"  The loop circumference in step-space = steps per crossing")
    print(f"  = {n_steps}/{n_crossings} = {n_steps/n_crossings:.1f} steps per crossing")
    print(f"  Minimum Phase 3 = 1 full loop = {n_steps/n_crossings * 2:.0f} steps")
    print(f"  (if winding number = 1 means 2 crossings per loop)")


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    import sys, os

    train_path = '/tmp/train_ids.json'
    vocab_path = '/tmp/vocab.json'

    if os.path.exists(train_path) and os.path.exists(vocab_path):
        test_padic_structure(train_path, vocab_path)
    else:
        print("Corpus files not found — skipping p-adic test")
        print("(run from compiler environment with /tmp/train_ids.json)")

    print()
    test_orbifold_lower_bound(N_layers=6, rank_monodromy=3)
    print()
    test_homology_structure()

    print()
    print("="*60)
    print("  SUMMARY")
    print("="*60)
    print("""
  Three potential sources of Phase 3 structure:

  1. P-ADIC: token frequency structure → loss landscape strata
     Test: check if H(strata) << H_geometric for any prime p
     If yes: loss is p-adically separated, can exploit structure

  2. ORBIFOLD: A_n Weyl group → topological lower bound
     Test: run Phase 3 with LR×50, check if still takes 24+ steps
     If yes: 24 steps is irreducible (A_4 orbifold lower bound)
     If no: 120 steps is purely numerical

  3. HOMOLOGY: H_1 loops → minimum path length
     Test: run from multiple init points, check winding numbers
     If H_1 ≠ 0: loop is topological, path length is fixed

  The first experiment to run: Phase 3 with LR×50.
  If convergence in <10 steps: purely numerical, no topology.
  If convergence in 24+ steps: orbifold lower bound likely.
""")


if __name__ == '__main__':
    main()
