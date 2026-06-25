"""
cr_solver_bridge.py
====================
Runs the J-holomorphic CR equation on WK matrices from your model
(naming: blocks.N.attn.WK.weight) without requiring c_attn format.

Focuses on the problematic φ₂=1.36 rad layer pair (L2→L3) first,
then all pairs. Uses N=16 grid for better residual than N=8.

Usage:
  python cr_solver_bridge.py --safetensors basin_state.safetensors \
      --pairs 1,2 2,3 3,4 --N 16 --dim 3 --verbose
"""

import argparse, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
from scipy.linalg import svd
from scipy.optimize import minimize

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--safetensors', default='basin_state.safetensors')
    p.add_argument('--pairs', nargs='+', default=['1,2','2,3','3,4'],
                   help='Layer pairs to solve e.g. 1,2 2,3')
    p.add_argument('--N',   type=int, default=16, help='Grid points per edge')
    p.add_argument('--dim', type=int, default=3,  help='Subspace dimension')
    p.add_argument('--lam', type=float, default=100.0, help='Boundary penalty')
    p.add_argument('--max_iter', type=int, default=2000)
    p.add_argument('--verbose', action='store_true')
    return p.parse_args()

# ─── Load WK matrices ─────────────────────────────────────────────────────────

def load_wk(path):
    from safetensors.numpy import load_file
    t = load_file(path)
    wk = {}
    for k, v in t.items():
        if 'attn.WK.weight' in k:
            parts = k.split('.')
            layer = int(parts[1])
            wk[layer] = v.astype(np.float32)
    return [wk[i] for i in sorted(wk)]

# ─── Lagrangian from WK ───────────────────────────────────────────────────────

def lagrangian_basis(W, dim):
    """Top-dim left singular vectors of W = basis of Lagrangian L=graph(W)."""
    U, s, Vt = svd(W, full_matrices=False)
    return U[:, :dim], s[:dim]

def strip_energy(W0, W1, dim):
    U0, _ = lagrangian_basis(W0, dim)
    U1, _ = lagrangian_basis(W1, dim)
    M = U0.T @ U1
    sv = np.clip(svd(M, compute_uv=False), -1+1e-7, 1-1e-7)
    return float(np.sum(np.arccos(sv))), U0, U1

def bridgeland_phase(W0, W1):
    """φ = arg(λ₁(W1 W0⁻¹)) — should be in {0,π} in orbit."""
    M = W1 @ np.linalg.pinv(W0)
    evals = np.linalg.eigvals(M)
    dom = evals[np.argmax(np.abs(evals.real))]
    phi = float(np.arctan2(dom.imag, dom.real))
    if phi < 0: phi += 2*np.pi
    clean = (abs(phi) < 0.3 or abs(phi - np.pi) < 0.3
             or abs(phi - 2*np.pi) < 0.3)
    return phi, clean

# ─── CR equation ─────────────────────────────────────────────────────────────
# ∂u/∂s + J ∂u/∂t = 0  on [0,1]²
# u(s,0) ∈ L0,  u(s,1) ∈ L1  (boundary Lagrangians)
# Standard J = [[0,-I],[I,0]],  J²=-I exactly

def reduced_lagrangian_map(W0, W1, dim):
    """
    Compute the relative rotation between L0 and L1 in dim-space.
    
    For nearly-orthogonal Lagrangians (θ ≈ π/2), tan(θ) diverges.
    Use the rotation matrix R = Vm @ Um^T instead, which is always
    well-conditioned and encodes the relative orientation of the two
    Lagrangian subspaces.
    
    Boundary conditions:
      t=0: p = 0          (L0 = zero-section)
      t=1: p = R @ q      (L1 rotated relative to L0)
    """
    U0, _, _ = svd(W0, full_matrices=False)
    U1, _, _ = svd(W1, full_matrices=False)
    U0, U1 = U0[:, :dim], U1[:, :dim]

    M = U0.T @ U1          # (dim, dim) overlap matrix
    Um, sm, Vhm = svd(M)
    sm = np.clip(sm, -1+1e-7, 1-1e-7)
    theta = np.arccos(sm)  # principal angles

    # Rotation matrix: always ||R|| = sqrt(dim), never diverges
    # R encodes how to rotate from L0's frame to L1's frame
    R = Vhm.T @ Um.T       # (dim, dim), orthogonal

    return U0, U1, R, theta

def cr_residual_and_boundary(u_flat, R, dim, N, lam):
    """
    u: (2*dim, N, N) strip map in reduced R^{2*dim}.
    Boundary conditions:
      t=0: p = 0       (L0 = zero-section)
      t=1: p = R @ q   (L1 = graph of rotation R relative to L0)
    R is orthogonal (||R||=sqrt(dim)), always well-conditioned.
    """
    u = u_flat.reshape(2*dim, N, N)
    ds = 1.0 / (N - 1)
    dt = 1.0 / (N - 1)

    du_ds = np.gradient(u, ds, axis=1)
    du_dt = np.gradient(u, dt, axis=2)
    Jdu_dt = np.concatenate([-du_dt[dim:], du_dt[:dim]], axis=0)
    cr_loss = np.sum((du_ds + Jdu_dt)**2)

    # t=0: p = 0
    b0 = np.sum(u[dim:, :, 0]**2)

    # t=1: p = R @ q
    q1 = u[:dim,  :, N-1]        # (dim, N)
    p1 = u[dim:,  :, N-1]        # (dim, N)
    p1_target = R @ q1            # (dim, N)
    b1 = np.sum((p1 - p1_target)**2)

    # Normalization at t=0
    q0 = u[:dim, :, 0]
    norm_loss = np.sum((np.sum(q0**2, axis=0) - 1.0)**2)

    return cr_loss + lam * (b0 + b1) + lam * 0.1 * norm_loss


def sigmoid_init(R, dim, N, seed=0):
    """
    Initialize u: [0,1]² → R^{2*dim} satisfying boundary conditions exactly.
    
    Exact boundary satisfaction:
      t=0: q=e₁, p=0                (L0 boundary)
      t=1: q=e₁, p=R@e₁             (L1 boundary)
    Interior: bilinear interpolation + sinusoidal variation in s
    to give the optimizer non-trivial structure to work with.
    """
    rng = np.random.default_rng(seed)
    s_grid = np.linspace(0, 1, N)
    t_grid = np.linspace(0, 1, N)
    sig = 1.0 / (1.0 + np.exp(-6*(t_grid - 0.5)))  # (N,) sigmoid in t

    u = np.zeros((2*dim, N, N))
    e1 = np.zeros(dim); e1[0] = 1.0
    p1 = R @ e1   # target p at t=1

    for i, s in enumerate(s_grid):
        for j, t in enumerate(t_grid):
            # q: e1 + small sinusoidal variation in s (keeps |q|≈1)
            s_bump = 0.05 * np.sin(2*np.pi*s)
            q = e1.copy()
            if dim > 1:
                q[1] = s_bump
            q = q / (np.linalg.norm(q) + 1e-8)
            
            # p: interpolate 0→p1 via sigmoid
            p = sig[j] * (R @ q)
            
            u[:dim, i, j] = q
            u[dim:, i, j] = p

    return u.ravel()


def interpolate_grid(u_coarse, dim, N_fine):
    """Bilinear interpolation of u from coarse to fine grid."""
    from scipy.ndimage import zoom
    N_coarse = u_coarse.shape[1]
    scale = N_fine / N_coarse
    u_fine = np.zeros((2*dim, N_fine, N_fine))
    for c in range(2*dim):
        u_fine[c] = zoom(u_coarse[c], scale, order=1)
    return u_fine


def solve_cr_cascade(W0, W1, dim, lam, max_iter, verbose,
                     pair_label, n_restarts=8,
                     grid_cascade=(8, 16, 32)):
    """
    Multi-resolution CR solve: coarse→fine grid cascade.
    Warm-starts fine grid from coarse solution.
    """
    t_total = time.time()
    energy, U0, U1 = strip_energy(W0, W1, dim)
    phi, clean      = bridgeland_phase(W0, W1)
    _, _, R, theta  = reduced_lagrangian_map(W0, W1, dim)

    print(f"\n  Pair {pair_label}:")
    print(f"    Strip area A = {energy:.4f}")
    print(f"    θ = " + ", ".join(f"{t:.4f}" for t in theta))
    print(f"    φ = {phi:.3f} rad  ({'clean' if clean else '⚠ NOT CLEAN'})")

    best_r   = np.inf
    best_u   = None
    best_N   = grid_cascade[0]

    # ── Coarse grid: multiple random restarts ─────────────────────────────
    N0 = grid_cascade[0]
    print(f"    [N={N0}] Coarse grid, {n_restarts} restarts …")
    for restart in range(n_restarts):
        u0 = sigmoid_init(R, dim, N0, seed=restart)
        if restart > 0:
            rng = np.random.default_rng(restart * 137)
            u0 += 0.05 * rng.standard_normal(u0.shape)
        res = minimize(
            cr_residual_and_boundary, u0,
            args=(R, dim, N0, lam),
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-11,
                     'maxfun': max_iter * 20}
        )
        if verbose:
            print(f"      restart {restart}: {res.fun:.4f} ({res.nit} iters)")
        if res.fun < best_r:
            best_r = res.fun
            best_u = res.x.reshape(2*dim, N0, N0)
            best_N = N0
        if best_r < 0.1:
            break

    print(f"    [N={N0}] Best residual: {best_r:.6f}")

    # ── Refine on progressively finer grids ──────────────────────────────
    for N_fine in grid_cascade[1:]:
        if best_r < 0.05:   # already converged — skip finer grids
            break
        print(f"    [N={N_fine}] Refining …")
        u_warm = interpolate_grid(best_u, dim, N_fine).ravel()

        # Also try fresh inits at this resolution
        candidates = [u_warm]
        for restart in range(3):
            u_fresh = sigmoid_init(R, dim, N_fine, seed=restart + 100)
            candidates.append(u_fresh)

        for i, u0 in enumerate(candidates):
            res = minimize(
                cr_residual_and_boundary, u0,
                args=(R, dim, N_fine, lam),
                method='L-BFGS-B',
                options={'maxiter': max_iter * 2, 'ftol': 1e-15,
                         'gtol': 1e-11, 'maxfun': max_iter * 40}
            )
            label_s = "warm" if i == 0 else f"fresh{i}"
            if verbose:
                print(f"      [{label_s}] {res.fun:.6f} ({res.nit} iters)")
            if res.fun < best_r:
                best_r = res.fun
                best_u = res.x.reshape(2*dim, N_fine, N_fine)
                best_N = N_fine
            if best_r < 0.1:
                break

        print(f"    [N={N_fine}] Best residual: {best_r:.6f}")
        if best_r < 0.1:
            break

    converged = best_r < 0.1
    elapsed   = time.time() - t_total
    print(f"    FINAL: {best_r:.6f}  "
          f"({'✓ CONVERGED' if converged else '✗ not converged'})  "
          f"best_N={best_N}  [{elapsed:.1f}s]")

    return {
        "pair":             pair_label,
        "strip_area":       float(energy),
        "phase_rad":        float(phi),
        "phase_clean":      bool(clean),
        "principal_angles": theta.tolist(),
        "residual_final":   float(best_r),
        "best_N":           int(best_N),
        "converged":        bool(converged),
        "elapsed_s":        round(elapsed, 1),
    }

def solve_cr(W0, W1, dim, N, lam, max_iter, verbose, pair_label):
    """Solve CR equation for the strip L0→L1."""
    t0 = time.time()

    energy, U0, U1 = strip_energy(W0, W1, dim)
    phi, clean = bridgeland_phase(W0, W1)
    _, _, R, theta = reduced_lagrangian_map(W0, W1, dim)

    print(f"\n  Pair {pair_label}:")
    print(f"    Strip area A = {energy:.4f}")
    print(f"    Principal angles θ = " +
          ", ".join(f"{t:.4f}" for t in theta) + " rad")
    print(f"    Bridgeland φ = {phi:.3f} rad  "
          f"({'clean' if clean else '⚠ NOT CLEAN'})")
    print(f"    ||R|| = {np.linalg.norm(R):.4f}  "
          f"(rotation matrix, should be sqrt(dim)={np.sqrt(dim):.3f})")

    u0 = sigmoid_init(R, dim, N)
    r0 = cr_residual_and_boundary(u0, R, dim, N, lam)
    print(f"    CR residual (init):  {r0:.4f}")

    res = minimize(
        cr_residual_and_boundary,
        u0,
        args=(R, dim, N, lam),
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8,
                 'maxfun': max_iter*10, 'disp': verbose}
    )

    r_final = res.fun
    converged = r_final < 0.1
    elapsed = time.time() - t0

    print(f"    CR residual (final): {r_final:.6f}  "
          f"({'✓ < 0.1' if converged else f'✗ > 0.1'})")
    print(f"    Iterations: {res.nit}  [{elapsed:.1f}s]")
    if not converged and verbose:
        print(f"    Optimizer: {res.message}")

    return {
        "pair":             pair_label,
        "strip_area":       float(energy),
        "phase_rad":        float(phi),
        "phase_clean":      bool(clean),
        "principal_angles": theta.tolist(),
        "R_norm":           float(np.linalg.norm(R)),
        "residual_init":    float(r0),
        "residual_final":   float(r_final),
        "converged":        bool(converged),
        "n_iter":           int(res.nit),
        "elapsed_s":        round(elapsed, 2),
    }

# ─── m2 wall score ────────────────────────────────────────────────────────────

def m2_wall_score(areas, k):
    """m2 ≠ 0 iff |A_k - µ| + |A_{k+1} - µ| > 2·MAD"""
    mu  = np.mean(areas)
    mad = np.mean(np.abs(areas - mu))
    ws  = abs(areas[k] - mu) + abs(areas[k+1] - mu)
    return float(ws), float(2*mad), bool(ws > 2*mad)

# ─── A∞ check ─────────────────────────────────────────────────────────────────

def ainf_check(results_by_pair):
    """m2∘m2 = 0 mod 2: check for consecutive triples."""
    pairs = sorted(results_by_pair.keys())
    print(f"\n  A∞ associativity check (m2∘m2 = 0 mod 2):")
    for i in range(len(pairs)-1):
        k0, k1 = pairs[i]
        l0, l1 = pairs[i+1]
        if k1 != l0:
            continue
        m2_01 = 1 if results_by_pair[(k0,k1)]["residual_final"] < 0.5 else 0
        m2_12 = 1 if results_by_pair[(l0,l1)]["residual_final"] < 0.5 else 0
        prod  = (m2_01 * m2_12) % 2
        print(f"    m2(L{k0},L{k1},L{l1}): "
              f"m2_{k0}{k1}={m2_01} × m2_{k1}{l1}={m2_12} "
              f"= {prod} (mod 2)  {'✓' if prod == 0 else '✗'}")

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("=" * 60)
    print("  CR SOLVER — J-HOLOMORPHIC STRIP EQUATION")
    print(f"  ∂u/∂s + J ∂u/∂t = 0,  J²=−I exact")
    print("=" * 60)
    print(f"  Safetensors: {args.safetensors}")
    print(f"  N={args.N}  dim={args.dim}  λ={args.lam}")

    # Load
    wk_list = load_wk(args.safetensors)
    n_layers = len(wk_list)
    D = wk_list[0].shape[0]
    print(f"  Loaded {n_layers} WK matrices, D={D}")

    # All strip areas (for m2 wall score)
    areas = []
    for k in range(n_layers - 1):
        A, _, _ = strip_energy(wk_list[k], wk_list[k+1], args.dim)
        areas.append(A)
    areas = np.array(areas)
    mu  = float(np.mean(areas))
    mad = float(np.mean(np.abs(areas - mu)))
    std = float(np.std(areas))
    print(f"\n  Strip areas: " + ", ".join(f"{a:.4f}" for a in areas))
    print(f"  µ={mu:.4f}  std={std:.4f}  MAD={mad:.4f}")
    print(f"  Note: areas smaller than hessian_cvector_correlation.py because")
    print(f"  dim={args.dim} subspace (vs rank=6 there) — consistent.")
    print(f"  Bridgeland phases:")
    for k in range(n_layers - 1):
        phi, clean = bridgeland_phase(wk_list[k], wk_list[k+1])
        if k < len(areas) - 1:  # need areas[k] and areas[k+1]
            ws, thresh, m2nz = m2_wall_score(areas, k)
            print(f"    L{k}→L{k+1}: φ={phi:.3f}rad  "
                  f"{'clean' if clean else '⚠ NOT CLEAN':12s}  "
                  f"ws={ws:.4f}  m2={'≠0' if m2nz else '=0'}")
        else:
            print(f"    L{k}→L{k+1}: φ={phi:.3f}rad  "
                  f"{'clean' if clean else '⚠ NOT CLEAN':12s}  "
                  f"(terminal pair)")

    # Parse pairs
    pairs = []
    for p in args.pairs:
        k, l = map(int, p.split(','))
        pairs.append((k, l))

    # Solve CR for each pair
    results = {}
    print(f"\n  Solving CR equation for {len(pairs)} pair(s) "
          f"[N={args.N}, dim={args.dim}]:")
    print(f"  Target: residual < 0.1 (required for A∞ verification)")

    for k, l in pairs:
        label = f"L{k}→L{l}"
        r = solve_cr_cascade(wk_list[k], wk_list[l], args.dim,
                             args.lam, args.max_iter,
                             args.verbose, label,
                             n_restarts=8,
                             grid_cascade=(8, 16, 32))
        results[(k,l)] = r

    # A∞ check
    ainf_check(results)

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    n_conv = sum(1 for r in results.values() if r["converged"])
    print(f"  Converged (residual < 0.1): {n_conv}/{len(pairs)}")
    print(f"  Strip area std: {std:.4f}  "
          f"({'uniform — search regime' if std < 0.5 else 'differentiated — floor regime'})")
    for (k,l), r in sorted(results.items()):
        status = "✓" if r["converged"] else "✗"
        print(f"  {status} L{k}→L{l}: residual={r['residual_final']:.6f}  "
              f"φ={r['phase_rad']:.3f}rad  "
              f"{'clean' if r['phase_clean'] else '⚠ NOT CLEAN'}")

    # Save
    import json
    out = {
        "safetensors": args.safetensors,
        "config": {"N": args.N, "dim": args.dim, "lam": args.lam},
        "strip_areas": areas.tolist(),
        "strip_area_std": float(std),
        "results": {f"{k},{l}": v for (k,l),v in results.items()},
        "n_converged": n_conv,
    }
    with open("cr_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Results → cr_results.json")

if __name__ == "__main__":
    main()
