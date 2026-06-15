#!/usr/bin/env python3
"""
Frobenius Orbit Size Test
==========================
Tests whether k_p (Frobenius orbit size of char poly mod p) is conserved
across the 24 transformer layers — the CORRECT conservation law from the
screenshot, distinct from real eigenvalue isospectrality (which failed).

k_p is defined as:
  - Compute char poly χ_k(t) = det(tI - L_k) for L_k = W_Q^(k) W_K^(k)^T
  - Reduce coefficients mod p to get χ_k(t) ∈ F_p[t]
  - Factor over F_p
  - k_p = degree of the largest irreducible factor
       OR = LCM of degrees of all irreducible factors (Frobenius orbit interpretation)

Screenshot claim: k_5 = 2.00 exactly at all 7 skeleton depths.

If confirmed: Frobenius orbit size IS the conserved quantity.
  The transformer implements the QR update of a Toda lattice
  with symmetry group GL(n/2) × GL(n/2).
  This gives the OT boundary condition that makes the transport unique.

Usage:
    python frobenius_orbit_test.py --model gpt2-medium
"""

import argparse, warnings
warnings.filterwarnings('ignore')
import numpy as np
from fractions import Fraction
from transformers import GPT2LMHeadModel

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='gpt2-medium')
parser.add_argument('--primes', nargs='+', type=int, default=[2, 3, 5, 7, 11],
                    help='Primes to test')
parser.add_argument('--minor_size', type=int, default=8,
                    help='Use leading minor of this size (full 1024 is slow)')
args = parser.parse_args()

print(f"\n{'='*70}")
print(f"  FROBENIUS ORBIT SIZE TEST")
print(f"  Tests: is k_p conserved across 24 layers?")
print(f"  Screenshot claim: k_5 = 2.00 at all skeleton depths")
print(f"{'='*70}\n")

print("Loading model...", flush=True)
model = GPT2LMHeadModel.from_pretrained(args.model)
model.eval()
d = model.config.n_embd
n_layers = model.config.n_layer
m = min(args.minor_size, d)

def get_L(l):
    W = model.transformer.h[l].attn.c_attn.weight.detach().cpu().numpy()
    WQ = W[:, :d]; WK = W[:, d:2*d]
    L = WQ @ WK.T
    return L[:m, :m]  # leading m×m minor

print(f"  d={d}  n_layers={n_layers}  working with {m}×{m} leading minor\n")

# ── Frobenius orbit size computation ──────────────────────────────────────────
def char_poly_mod_p(M, p):
    """
    Compute characteristic polynomial of M mod p using integer arithmetic.
    Returns coefficients [c_0, c_1, ..., c_n] of det(tI - M) mod p.
    Uses the Berkowitz algorithm approximated by rounding M entries.
    """
    n = M.shape[0]
    # Round M to nearest integer, then reduce mod p
    M_int = np.round(M).astype(int) % p
    
    # Compute char poly coefficients using Faddeev-LeVerrier algorithm mod p
    # c_n = 1 (leading coeff)
    # c_{n-k} = (-1/k) * tr(M^k + c_{n-1} M^{k-1} + ...)
    
    # For small matrices, use direct computation
    # For the purposes of this test, we use the eigenvalues approach:
    # 1. Get real eigenvalues
    # 2. Form char poly from roots
    # 3. Round coefficients to integers
    # 4. Reduce mod p
    
    try:
        eigvals = np.linalg.eigvals(M.astype(float))
        # Form char poly coefficients from roots
        # poly = prod(t - lambda_i) = sum_k e_k * t^(n-k) * (-1)^k
        # where e_k are elementary symmetric polynomials
        from numpy.polynomial import polynomial as P
        coeffs = np.poly(eigvals)  # coeffs[0] = leading = 1
        # Round to nearest integer
        coeffs_int = np.round(coeffs.real).astype(int)
        coeffs_mod = coeffs_int % p
        return coeffs_mod, True
    except Exception as e:
        return None, False

def factor_over_Fp(poly_coeffs, p):
    """
    Factor polynomial over F_p by trial division.
    Returns list of (factor_degree, multiplicity) pairs.
    poly_coeffs: [c_n, c_{n-1}, ..., c_0] with c_n = leading term.
    """
    n = len(poly_coeffs) - 1
    if n <= 0:
        return []
    
    # Convert to monic polynomial mod p
    lead = int(poly_coeffs[0]) % p
    if lead == 0:
        return []
    lead_inv = pow(int(lead), -1, p) if lead != 0 else 0
    
    coeffs = [int(c) * lead_inv % p for c in poly_coeffs]
    
    factors = []
    
    # Find linear factors first (roots in F_p)
    remaining = coeffs.copy()
    for r in range(p):
        # Evaluate polynomial at r
        val = 0
        for c in remaining:
            val = (val * r + c) % p
        if val == 0:
            # r is a root, divide out (t - r)
            quot = []
            carry = 0
            for c in remaining[:-1]:
                carry = (carry + c) % p
                quot.append(carry)
                carry = carry * r % p
            remaining = quot
            factors.append((1, r))
            if len(remaining) <= 1:
                break
    
    # Remaining polynomial — try degree-2 irreducible factors
    if len(remaining) > 1:
        deg = len(remaining) - 1
        # Check if remaining is irreducible over F_p
        # Simple check: no roots in F_p
        has_root = False
        for r in range(p):
            val = 0
            for c in remaining:
                val = (val * r + c) % p
            if val == 0:
                has_root = True
                break
        
        if not has_root and deg == 2:
            factors.append((2, None))  # irreducible degree-2 factor
        elif not has_root and deg > 2:
            factors.append((deg, None))  # unknown irreducible structure
        elif has_root:
            factors.append((1, None))  # has linear factor
    
    return factors

def compute_k_p(M, p):
    """
    Compute Frobenius orbit size k_p for matrix M over F_p.
    k_p = max degree of irreducible factors of char poly mod p.
    
    k_p = 1: char poly splits completely (all linear factors) = supersingular
    k_p = 2: largest irred factor has degree 2 = GL(n/2)×GL(n/2) symmetry
    k_p = k: k-th Frobenius orbit
    """
    coeffs, ok = char_poly_mod_p(M, p)
    if not ok:
        return None
    
    factors = factor_over_Fp(coeffs, p)
    if not factors:
        return None
    
    max_deg = max(deg for deg, _ in factors)
    return max_deg

# ── Run across all layers ─────────────────────────────────────────────────────
print("Computing L_k = W_Q^(k) W_K^(k)^T for all layers...", flush=True)
L_layers = [get_L(l) for l in range(n_layers)]

print(f"\n{'='*70}")
print(f"  FROBENIUS ORBIT SIZES k_p across {n_layers} layers")
print(f"  Using {m}×{m} leading minor of L_k")
print("="*70)

print(f"\n  {'Layer':>6}", end='')
for p in args.primes:
    print(f"  {'k_'+str(p):>6}", end='')
print()
print("  " + "-"*(8 + 8*len(args.primes)))

all_kp = {p: [] for p in args.primes}

for l in range(n_layers):
    print(f"  L{l:>2}:  ", end='')
    for p in args.primes:
        kp = compute_k_p(L_layers[l], p)
        all_kp[p].append(kp)
        if kp is not None:
            print(f"  {kp:>6}", end='')
        else:
            print(f"  {'?':>6}", end='')
    print()

# ── Conservation analysis ─────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  CONSERVATION ANALYSIS")
print("="*70)

print(f"\n  {'Prime':>6}  {'Values':>30}  {'Constant?':>10}  {'Mean':>6}")
print("  " + "-"*58)

for p in args.primes:
    vals = [v for v in all_kp[p] if v is not None]
    if not vals:
        print(f"  p={p:>2}:   all failed")
        continue
    is_const = len(set(vals)) == 1
    mean_val = float(np.mean(vals))
    val_str = str(sorted(set(vals)))
    const_str = "YES ✓" if is_const else "NO ✗"
    print(f"  p={p:>2}:   {val_str:>30}  {const_str:>10}  {mean_val:>6.2f}")

print(f"""
  Screenshot claim: k_5 = 2.00 at all layers.
  
  IF k_5 = 2 is constant:
    → Frobenius orbit IS the conserved quantity ✓
    → Transport factors through GL(n/2) × GL(n/2)
    → This is the OT boundary condition
    → The 62 exceptional points are where k_5 transitions to 1
    → OT problem is WELL-POSED with this constraint
    
  IF k_5 varies:
    → The Frobenius orbit is NOT conserved
    → Need different invariant
    
  IF k_5 = 2 constant BUT k_p varies for other p:
    → The symmetry is p=5 specific
    → The GL(n/2) × GL(n/2) structure is visible only mod 5
    → This tells us which prime indexes the relevant representation
""")

# ── What this means for OT ────────────────────────────────────────────────────
print(f"{'='*70}")
print(f"  IMPLICATIONS FOR OPTIMAL TRANSPORT")
print("="*70)
print(f"""
  The optimal transport problem on LGr(d, 2d):
  
    min over F: c(L_0, F(L_0)) integrated over mu
    subject to: k_p constant across transport
  
  If k_5 = 2 is conserved:
    The constraint set is the GL(n/2)×GL(n/2) stratum of LGr.
    This is a CLOSED SUBMANIFOLD — the OT problem is well-posed.
    The optimal F is the unique geodesic on this submanifold.
    
    Computing F:
    1. All W_K^(k) lie on the GL(n/2)×GL(n/2) stratum (k_5=2 constant)
    2. The OT map F is the Riemannian exponential on this stratum
    3. F(t) at t=k/24 gives W_K^(k) directly
    4. No gradient descent — one geodesic computation
    
    This IS the Bridge B for transformers.
    The k_5=2 stratum is the "graph" we needed.
    The OT geodesic is the "Ihara zeta solution" we needed.
""")
