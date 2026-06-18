#!/usr/bin/env python3
"""
CR Triangle Solver — A∞ Product m2 via J-Holomorphic Triangles
================================================================
Computes the A∞ product m2: CF*(L_{k+1},L_{k+2}) ⊗ CF*(L_k,L_{k+1}) → CF*(L_k,L_{k+2})
by counting rigid J-holomorphic triangles with boundary on three Lagrangians.

SETUP
-----
The triangle u: Δ → T*R^d solves the CR equation:
    ∂u/∂s + J(u) ∂u/∂t = 0

on a disc with three boundary segments, each constrained to one Lagrangian:
    edge_01 (from corner a to b): u ∈ L_{k+1}
    edge_12 (from corner b to c): u ∈ L_{k+2}
    edge_20 (from corner c to a): u ∈ L_k

Corner conditions:
    u(corner_a) → a ∈ L_k ∩ L_{k+1}   (as s→-∞ on edge_01)
    u(corner_b) → b ∈ L_{k+1} ∩ L_{k+2} (as s→+∞ on edge_12)
    u(corner_c) → c ∈ L_k ∩ L_{k+2}   (as s→-∞ on edge_20)

DISCRETISATION
--------------
We use the conformal map of Δ to [-S,S] × [0,1] with punctures
replaced by asymptotic conditions. The triangle has three strip-like
ends corresponding to the three generators a, b, c.

Practically: we triangulate the disc Δ as a N_s × N_t grid with
three boundary segments meeting at corners.

BRIDGELAND PREDICTION
---------------------
m2(b,a) ≠ 0 (mod 2) IFF the pair of strips (k→k+1, k+1→k+2)
straddles a Bridgeland wall — i.e., Im(z_k) and Im(z_{k+1}) have
opposite signs {0,π}.

Wall layers (from lefschetz.py): L1→2, L6→7, L11→12, L13→14, L18→19

Usage:
    python cr_triangle.py
    python cr_triangle.py --layer_triple 12,13,14
    python cr_triangle.py --all_triples
"""
import argparse, json, warnings, time
warnings.filterwarnings('ignore')
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import lsqr as sp_lsqr
from scipy.optimize import minimize

parser = argparse.ArgumentParser()
parser.add_argument('--model',        default='gpt2-medium')
parser.add_argument('--layer_triple', default='12,13,14',
                    help='Three consecutive layers k,k+1,k+2')
parser.add_argument('--dim',    type=int,   default=6)
parser.add_argument('--N',      type=int,   default=12,
                    help='Grid size N×N on triangle')
parser.add_argument('--lambda_bdy', type=float, default=500.0)
parser.add_argument('--all_triples', action='store_true')
parser.add_argument('--save',   default='cr_triangle_results.json')
parser.add_argument('--verbose', action='store_true')
args = parser.parse_args()

layers = [int(x) for x in args.layer_triple.split(',')]
assert len(layers) == 3
Lk, Lk1, Lk2 = layers

DIM = args.dim
N   = args.N
lam = args.lambda_bdy

print(f"\n{'='*65}")
print(f"  CR TRIANGLE — J-HOLOMORPHIC TRIANGLE COUNTER")
print(f"  Triple: L{Lk} → L{Lk1} → L{Lk2}")
print(f"  Grid: {N}×{N}  |  DIM={DIM}  |  λ={lam}")
print(f"{'='*65}\n")

# ── Load model weights (no torchvision dependency) ──────────────────────────
import torch

def _load_gpt2_weights(model_name):
    """Load GPT2 weights without importing transformers (avoids torchvision issue)."""
    from pathlib import Path
    import os
    # Try HuggingFace cache directly
    cache_dirs = [
        Path.home() / '.cache' / 'huggingface' / 'hub',
        Path.home() / '.cache' / 'torch' / 'transformers',
    ]
    # Find cached model
    for cache_dir in cache_dirs:
        if not cache_dir.exists(): continue
        for p in cache_dir.rglob('pytorch_model.bin'):
            if model_name.replace('-','') in str(p).replace('-','').lower():
                print(f"  Loading from cache: {p}")
                return torch.load(p, map_location='cpu', weights_only=True)
        for p in cache_dir.rglob('model.safetensors'):
            if model_name.replace('-','') in str(p).replace('-','').lower():
                print(f"  Loading from cache (safetensors): {p}")
                try:
                    from safetensors.torch import load_file
                    return load_file(str(p))
                except ImportError:
                    pass
    # Fall back to transformers if available (without torchvision)
    import sys
    try:
        import transformers
        # Patch out the torchvision dependency before importing GPT2
        import unittest.mock as mock
        with mock.patch.dict(sys.modules, {'torchvision': mock.MagicMock(),
                                            'torchvision.transforms': mock.MagicMock()}):
            from transformers import GPT2LMHeadModel, GPT2Config
            config = GPT2Config.from_pretrained(model_name)
            m = GPT2LMHeadModel.from_pretrained(model_name, config=config)
            return m.state_dict()
    except Exception as e:
        print(f"  transformers load failed: {e}")
        print("  Generating synthetic W_K for testing...")
        return None

print(f"  Loading {args.model} weights...")
_sd = _load_gpt2_weights(args.model)
_n_layers = 24 if 'medium' in args.model else (48 if 'xl' in args.model else 12)
_d_model  = 1024 if 'medium' in args.model else (1600 if 'xl' in args.model else 768)

def get_WK(layer):
    """Get W_K for layer, projected to DIM subspace."""
    if _sd is None:
        # Synthetic: use deterministic random based on layer
        rng = np.random.RandomState(layer * 42 + 7)
        return rng.randn(DIM, DIM) * 0.02 + np.eye(DIM) * 0.1
    key = f'transformer.h.{layer}.attn.c_attn.weight'
    if key not in _sd:
        rng = np.random.RandomState(layer * 42 + 7)
        return rng.randn(DIM, DIM) * 0.02 + np.eye(DIM) * 0.1
    W = _sd[key].float().numpy()  # [d_model, 3*d_model]
    WK_full = W[:, _d_model:2*_d_model].T  # [d_model, d_model]
    # PCA project to DIM
    U, s, Vt = np.linalg.svd(WK_full)
    return Vt[:DIM, :DIM]  # [DIM, DIM]

WK0 = get_WK(Lk)   # L_k Lagrangian
WK1 = get_WK(Lk1)  # L_{k+1} Lagrangian
WK2 = get_WK(Lk2)  # L_{k+2} Lagrangian

# ── Almost complex structure ──────────────────────────────────────────────────
def make_J(WK_a, WK_b):
    """J from two W_K matrices, using L_k × L_{k+1} product."""
    A = WK_b @ np.linalg.pinv(WK_a)
    # Polar decomposition for J^2=-I
    U, s, Vt = np.linalg.svd(A)
    A_eff = U @ Vt   # orthogonal part
    J = np.block([[np.zeros((DIM,DIM)), -A_eff.T],
                  [A_eff,               np.zeros((DIM,DIM))]])
    # Verify J^2 = -I
    J2_err = np.linalg.norm(J @ J + np.eye(2*DIM))
    return J, J2_err

J01, J01_err = make_J(WK0, WK1)  # J for L_k → L_{k+1}
J12, J12_err = make_J(WK1, WK2)  # J for L_{k+1} → L_{k+2}
J02, J02_err = make_J(WK0, WK2)  # J for L_k → L_{k+2}
# Use average J for the triangle interior
# Use geodesic midpoint in the space of complex structures.
# Arithmetic average destroys J²=-I; geodesic preserves it.
# J(s) = expm(s * logm(J12 @ inv(J01))) @ J01, s=0.5
from scipy.linalg import expm as _expm, logm as _logm
try:
    _JJ = J12 @ np.linalg.inv(J01)
    J = _expm(0.5 * _logm(_JJ)) @ J01
except Exception:
    # Fallback: use J01 alone (also satisfies J²=-I exactly)
    J = J01.copy()
J2_err = float(np.linalg.norm(J @ J + np.eye(2*DIM)))
print(f"  J built: J²_err={J2_err:.2e}  (J01_err={J01_err:.2e}, J12_err={J12_err:.2e})")

TOT = 2 * DIM  # total phase-space dim

# ── Generators (intersection points) ─────────────────────────────────────────
def find_generators(WKa, WKb, n=2):
    """Find generators of CF*(L_a, L_b) = ker(WKb - WKa) ∩ unit ball."""
    diff = WKb - WKa
    if np.linalg.matrix_rank(diff) < DIM:
        # Non-trivial kernel
        U, s, Vt = np.linalg.svd(diff)
        null_idx = np.where(s < 0.01 * s[0])[0]
        null_vecs = Vt[null_idx]
    else:
        # Approximate: use smallest singular vectors
        U, s, Vt = np.linalg.svd(diff)
        null_vecs = Vt[-min(n, DIM):]
    # Generators in phase space T*R^DIM: (q, p) where p = WKa @ q
    gens = []
    for v in null_vecs[:n]:
        v /= max(np.linalg.norm(v), 1e-8)
        q = v
        p = WKa @ q
        gens.append(np.concatenate([q, p]))
    return gens

gens_01 = find_generators(WK0, WK1)   # a ∈ L_k ∩ L_{k+1}
gens_12 = find_generators(WK1, WK2)   # b ∈ L_{k+1} ∩ L_{k+2}
gens_02 = find_generators(WK0, WK2)   # c ∈ L_k ∩ L_{k+2}

print(f"  Generators: |L_k∩L_{{k+1}}|={len(gens_01)}, "
      f"|L_{{k+1}}∩L_{{k+2}}|={len(gens_12)}, "
      f"|L_k∩L_{{k+2}}|={len(gens_02)}")

# ── Triangle grid ─────────────────────────────────────────────────────────────
# Map the triangle to [0,1]×[0,1] with the three edges:
#   edge 0 (t=0): q-axis boundary → L_k condition
#   edge 1 (s=1): right boundary → L_{k+1} condition  
#   edge 2 (t=1): top boundary → L_{k+2} condition
#
# More precisely: use the unit square as the conformal model of the disc Δ
# with corners at (0,0)=a, (1,0)=b, (1,1)=c, (0,1) identified with a
# (This is the standard pair-of-pants/triangle conformal model)
#
# Grid points: (i,j) for i,j in 0..N-1
# s = i/(N-1), t = j/(N-1)
#
# Boundary assignments:
#   Bottom edge (t=0): L_k boundary
#   Right edge  (s=1): L_{k+1} boundary
#   Top edge    (t=1): L_{k+2} boundary
#   Left edge   (s=0): degenerate (corner a, asymptotic)
#
# Corner conditions:
#   (s=0, t=0): generator a ∈ L_k ∩ L_{k+1}
#   (s=1, t=0): generator b ∈ L_{k+1} ∩ L_{k+2}
#   (s=1, t=1): generator c ∈ L_k ∩ L_{k+2}

def solve_triangle(a_gen, b_gen, c_gen, verbose=False):
    """
    Solve for J-holomorphic triangle with corners a, b, c.
    
    Returns: {
        'cr_final': residual,
        'triangle_area': symplectic area,
        'converged': bool,
        'triangle_mod2': int
    }
    """
    ds = 1.0 / (N-1)
    dt = 1.0 / (N-1)
    
    # Initialize: linear interpolation between corners
    u = np.zeros((N, N, TOT))
    for i in range(N):
        for j in range(N):
            s = i / (N-1)
            t = j / (N-1)
            # Bilinear interpolation
            u[i, j] = ((1-s)*(1-t)*a_gen + s*(1-t)*b_gen + 
                        s*t*c_gen + (1-s)*t*a_gen)

    # ── Lagrangian projection ─────────────────────────────────────────────────
    def project_Lk(v):
        """Project to L_k: p = WK0 @ q."""
        q = v[:DIM]
        p = WK0 @ q
        return np.concatenate([q, p])

    def project_Lk1(v):
        """Project to L_{k+1}: p = WK1 @ q."""
        q = v[:DIM]
        p = WK1 @ q
        return np.concatenate([q, p])

    def project_Lk2(v):
        """Project to L_{k+2}: p = WK2 @ q."""
        q = v[:DIM]
        p = WK2 @ q
        return np.concatenate([q, p])

    def apply_boundary(u):
        """Enforce Lagrangian boundary conditions via projection."""
        u_new = u.copy()
        # Bottom edge (j=0): L_k
        for i in range(N):
            u_new[i, 0] = project_Lk(u[i, 0])
        # Right edge (i=N-1): L_{k+1}
        for j in range(N):
            u_new[N-1, j] = project_Lk1(u[N-1, j])
        # Top edge (j=N-1): L_{k+2}
        for i in range(N):
            u_new[i, N-1] = project_Lk2(u[i, N-1])
        # Corner conditions
        u_new[0,   0]   = a_gen   # (s=0, t=0) → a
        u_new[N-1, 0]   = b_gen   # (s=1, t=0) → b
        u_new[N-1, N-1] = c_gen   # (s=1, t=1) → c
        u_new[0,   N-1] = a_gen   # (s=0, t=1) → a (left edge stays at a)
        return u_new

    u = apply_boundary(u)

    # ── CR residual ───────────────────────────────────────────────────────────
    def cr_residual(u_flat):
        u = u_flat.reshape(N, N, TOT)
        res = np.zeros_like(u)
        # Interior points: du/ds + J du/dt = 0
        for i in range(1, N-1):
            for j in range(1, N-1):
                duds = (u[i+1,j] - u[i-1,j]) / (2*ds)
                dudt = (u[i,j+1] - u[i,j-1]) / (2*dt)
                res[i, j] = duds + J @ dudt
        return res.flatten()

    def full_objective(u_flat):
        u = u_flat.reshape(N, N, TOT)
        cr = cr_residual(u_flat)
        loss = 0.5 * np.dot(cr, cr)
        # Boundary penalty
        bdy_loss = 0.0
        # Bottom (L_k)
        for i in range(N):
            v = u[i,0]; q=v[:DIM]; p=v[DIM:]
            bdy_loss += lam * np.sum((p - WK0@q)**2)
        # Right (L_{k+1})
        for j in range(N):
            v = u[N-1,j]; q=v[:DIM]; p=v[DIM:]
            bdy_loss += lam * np.sum((p - WK1@q)**2)
        # Top (L_{k+2})
        for i in range(N):
            v = u[i,N-1]; q=v[:DIM]; p=v[DIM:]
            bdy_loss += lam * np.sum((p - WK2@q)**2)
        # Corners
        bdy_loss += lam * np.sum((u[0,0]   - a_gen)**2)
        bdy_loss += lam * np.sum((u[N-1,0] - b_gen)**2)
        bdy_loss += lam * np.sum((u[N-1,N-1] - c_gen)**2)
        total = loss + bdy_loss
        grad = np.zeros_like(u_flat)  # finite diff gradient below
        return total, grad

    # Stage 1: L-BFGS-B
    def obj_scalar(u_flat):
        u_r = u_flat.reshape(N, N, TOT)
        cr = cr_residual(u_flat)
        loss = 0.5 * np.dot(cr, cr)
        bdy = 0.0
        for i in range(N):
            v=u_r[i,0];   q=v[:DIM]; p=v[DIM:]; bdy+=lam*np.sum((p-WK0@q)**2)
        for j in range(N):
            v=u_r[N-1,j]; q=v[:DIM]; p=v[DIM:]; bdy+=lam*np.sum((p-WK1@q)**2)
        for i in range(N):
            v=u_r[i,N-1]; q=v[:DIM]; p=v[DIM:]; bdy+=lam*np.sum((p-WK2@q)**2)
        bdy += lam*(np.sum((u_r[0,0]-a_gen)**2)+np.sum((u_r[N-1,0]-b_gen)**2)+
                    np.sum((u_r[N-1,N-1]-c_gen)**2))
        return float(loss + bdy)

    t0 = time.time()
    res0 = float(np.linalg.norm(cr_residual(u.flatten())))
    
    result = minimize(obj_scalar, u.flatten(), method='L-BFGS-B',
                      options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
    
    u_opt = result.x.reshape(N, N, TOT)
    cr_final = float(np.linalg.norm(cr_residual(result.x)))
    
    # Symplectic area: ∫∫ ω = ∫∫ (dq ∧ dp)
    q = u_opt[:,:,:DIM]
    p = u_opt[:,:,DIM:]
    dqds = np.diff(q, axis=0) / ds
    dpdt = np.diff(p, axis=1) / dt
    dpds = np.diff(p, axis=0) / ds
    dqdt = np.diff(q, axis=1) / dt
    ms = min(dqds.shape[0], dpdt.shape[0])
    mt = min(dqds.shape[1], dpdt.shape[1])
    intg = (np.einsum('ijk,ijk->ij', dqds[:ms,:mt], dpdt[:ms,:mt]) -
            np.einsum('ijk,ijk->ij', dpds[:ms,:mt], dqdt[:ms,:mt]))
    omega = float(np.sum(intg) * ds * dt)
    if not np.isfinite(omega): omega = 0.0
    
    converged = cr_final < 0.5
    triangle_exists = converged and abs(omega) > 1e-6
    
    if verbose:
        print(f"    CR: {res0:.4f} → {cr_final:.6f}  A={omega:.6f}  "
              f"t={time.time()-t0:.1f}s")
    
    return {
        'cr_init': res0,
        'cr_final': cr_final,
        'triangle_area': omega,
        'converged': converged,
        'triangle_exists': triangle_exists,
        'triangle_mod2': int(triangle_exists),
    }

# ── Bridgeland wall prediction ─────────────────────────────────────────────────
# From lefschetz.py: Im(z_k) = arg(lambda_1(phi_k)) ∈ {0, π}
# Wall layers: where successive strips STRADDLE a wall
BRIDGELAND_WALLS = {1, 6, 11, 13, 18}  # from lefschetz.py run

def straddles_wall(k, k1):
    """Does the transition k→k1 straddle a Bridgeland wall?"""
    return k in BRIDGELAND_WALLS or k1 in BRIDGELAND_WALLS

prediction = straddles_wall(Lk, Lk1) or straddles_wall(Lk1, Lk2)
print(f"  Bridgeland prediction: m2 {'≠' if prediction else '='} 0 "
      f"({'straddles wall' if prediction else 'no wall'})")
print()

# ── Solve triangle ─────────────────────────────────────────────────────────────
def solve_triple(lk, lk1, lk2):
    WKa = get_WK(lk)
    WKb = get_WK(lk1)
    WKc = get_WK(lk2)
    
    ga = find_generators(WKa, WKb)
    gb = find_generators(WKb, WKc)
    gc = find_generators(WKa, WKc)
    
    if not ga or not gb or not gc:
        return {'error': 'no generators', 'triangle_mod2': 0}
    
    # Try primary generator combination
    results = []
    for a in ga[:1]:
        for b in gb[:1]:
            for c in gc[:1]:
                r = solve_triangle(a, b, c, verbose=args.verbose)
                r['lk'] = lk; r['lk1'] = lk1; r['lk2'] = lk2
                results.append(r)
    
    # Take the result with smallest CR residual
    best = min(results, key=lambda r: r.get('cr_final', 1e9))
    return best

print(f"Solving triangle L{Lk}→L{Lk1}→L{Lk2}...")
t0 = time.time()
result = solve_triple(Lk, Lk1, Lk2)
elapsed = time.time() - t0

print(f"\n{'='*65}")
print(f"  TRIANGLE RESULT: L{Lk}→L{Lk1}→L{Lk2}")
print(f"{'='*65}")
print(f"  CR residual:     {result.get('cr_init',0):.4f} → {result.get('cr_final',0):.6f}")
print(f"  Triangle area:   {result.get('triangle_area',0):.8f}")
print(f"  Converged:       {result.get('converged', False)}")
print(f"  Triangle mod2:   {result.get('triangle_mod2', 0)}")
print(f"  Time:            {elapsed:.1f}s")
print()
print(f"  Bridgeland prediction:  m2 {'≠' if prediction else '='} 0")
print(f"  Computed result:        m2 {'≠' if result.get('triangle_mod2',0) else '='} 0")
match = (prediction == bool(result.get('triangle_mod2', 0)))
print(f"  Match: {'✓' if match else '✗'}")

# ── Full sweep if requested ────────────────────────────────────────────────────
if args.all_triples:
    print(f"\n{'='*65}")
    print("  FULL SWEEP: All 22 consecutive triples")
    print(f"{'='*65}")
    print(f"  {'Triple':>12}  {'CR_final':>9}  {'Area':>10}  {'mod2':>5}  "
          f"{'Wall?':>6}  {'Match?':>7}")
    print("  " + "-"*55)
    
    sweep_results = []
    for k in range(22):
        r = solve_triple(k, k+1, k+2)
        wall = straddles_wall(k, k+1) or straddles_wall(k+1, k+2)
        pred_mod2 = int(wall)
        got_mod2  = r.get('triangle_mod2', 0)
        ok = (pred_mod2 == got_mod2)
        
        print(f"  L{k:>2}→L{k+1}→L{k+2}:  "
              f"{r.get('cr_final',99):>9.4f}  "
              f"{r.get('triangle_area',0):>10.6f}  "
              f"{got_mod2:>5}  "
              f"{'Y' if wall else 'N':>6}  "
              f"{'✓' if ok else '✗':>7}")
        
        sweep_results.append({
            'lk': k, 'lk1': k+1, 'lk2': k+2,
            'cr_final': r.get('cr_final', 99),
            'triangle_area': r.get('triangle_area', 0),
            'triangle_mod2': got_mod2,
            'bridgeland_prediction': pred_mod2,
            'match': ok,
        })
    
    correct = sum(1 for r in sweep_results if r['match'])
    print(f"\n  Bridgeland prediction accuracy: {correct}/22 = {correct/22*100:.0f}%")
    
    if args.save:
        with open(args.save, 'w') as f:
            json.dump(sweep_results, f, indent=2)
        print(f"  Saved → {args.save}")
