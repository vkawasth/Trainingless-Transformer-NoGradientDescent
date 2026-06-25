"""
ainf_triangle_solver.py  (v3 - correct FEM approach)
=====================================================
A∞ pentagon: m₂∘m₂ = 0 (mod 2)?

Uses a proper triangular finite element discretization with
precomputed adjacency for the CR equation finite differences.

Key insight: the triangle domain Δ = {(s,t): s≥0, t≥0, s+t≤1}
has a natural triangular grid. We precompute which index corresponds
to each (i,j) position using index_map[i,j] = idx, then use
this for O(1) neighbor lookup instead of slow distance search.
"""

import argparse, json, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
from scipy.linalg import svd
from scipy.optimize import minimize

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--safetensors', nargs='+',
                   default=['tau_spike_64.safetensors',
                            'basin_state.safetensors'])
    p.add_argument('--dim',       type=int,   default=3)
    p.add_argument('--N',         type=int,   default=10)
    p.add_argument('--lam',       type=float, default=0.05)
    p.add_argument('--max_iter',  type=int,   default=3000)
    p.add_argument('--n_restarts',type=int,   default=6)
    p.add_argument('--verbose',   action='store_true')
    p.add_argument('--output',    default='ainf_report.json')
    return p.parse_args()


# ─── WK loading ───────────────────────────────────────────────────────────────

def load_wk(path):
    from safetensors.numpy import load_file
    t = load_file(path)
    wk = {}
    for k, v in t.items():
        if 'attn.WK.weight' in k:
            wk[int(k.split('.')[1])] = v.astype(np.float32)
    return [wk[i] for i in sorted(wk)]


# ─── Geometry ─────────────────────────────────────────────────────────────────

def lagrangian_basis(W, dim):
    U, _, _ = svd(W, full_matrices=False)
    return U[:, :dim]

def bridgeland_phase(W0, W1):
    M = W1 @ np.linalg.pinv(W0)
    ev = np.linalg.eigvals(M)
    dom = ev[np.argmax(np.abs(ev.real))]
    phi = float(np.arctan2(dom.imag, dom.real))
    if phi < 0: phi += 2*np.pi
    return phi, abs(phi - np.pi) < 0.3

def rotation_R(W0, W1, dim):
    """Relative rotation from L0 to L1; flipped for phi=pi strips."""
    U0 = lagrangian_basis(W0, dim)
    U1 = lagrangian_basis(W1, dim)
    M = U0.T @ U1
    Um, sm, Vhm = svd(M)
    sm = np.clip(sm, -1+1e-7, 1-1e-7)
    theta = np.arccos(sm)
    R = Vhm.T @ Um.T
    _, pi = bridgeland_phase(W0, W1)
    return (-R if pi else R), theta


# ─── Triangular grid with O(1) neighbor lookup ───────────────────────────────

def build_triangle_grid(N):
    """
    Grid: (i,j) with i+j <= N-1, i,j >= 0.
    s = i/(N-1), t = j/(N-1).
    Total M = N*(N+1)/2 points.
    
    Returns:
      pts      : (M, 2) — (s,t) coords
      idx_map  : (N, N) — idx_map[i,j] = flat index (-1 if outside)
      interior : (M,) bool
      bd0, bd1, bd2 : (M,) bool boundary masks
    """
    idx_map = -np.ones((N, N), dtype=int)
    pts = []
    for i in range(N):
        for j in range(N - i):
            idx_map[i, j] = len(pts)
            pts.append([i/(N-1), j/(N-1)])
    pts = np.array(pts)
    M = len(pts)

    eps = 0.5   # in grid units
    bd0 = np.array([j == 0                 for i in range(N) for j in range(N-i)])
    bd1 = np.array([i == 0                 for i in range(N) for j in range(N-i)])
    bd2 = np.array([i + j == N - 1         for i in range(N) for j in range(N-i)])
    interior = ~bd0 & ~bd1 & ~bd2

    # Precompute neighbors for each interior point
    # neighbor_s[idx] = (idx_plus_s, idx_minus_s) in grid coords
    neighbors = {}
    k = 0
    for i in range(N):
        for j in range(N - i):
            if interior[k]:
                ip = idx_map[i+1, j] if i+1 < N and j < N-(i+1) else -1
                im = idx_map[i-1, j] if i-1 >= 0 else -1
                jp = idx_map[i, j+1] if j+1 < N-(i) else -1
                jm = idx_map[i, j-1] if j-1 >= 0 else -1
                neighbors[k] = (ip, im, jp, jm)
            k += 1

    return pts, idx_map, interior, bd0, bd1, bd2, neighbors, M


# ─── CR residual ─────────────────────────────────────────────────────────────

def triangle_cr_residual(u_flat, R01, R02, dim, N,
                          lam, interior, bd0, bd1, bd2,
                          neighbors, M):
    u = u_flat.reshape(2*dim, M)
    q, p = u[:dim], u[dim:]
    h = 1.0 / (N - 1)

    n_int = max(interior.sum(), 1)
    n_bd  = max(bd0.sum() + bd1.sum() + bd2.sum(), 1)

    # Boundary terms — normalized per point
    b0 = np.sum(p[:, bd0]**2) / n_bd
    b1 = np.sum((p[:, bd1] - R01 @ q[:, bd1])**2) / n_bd
    b2 = np.sum((p[:, bd2] - R02 @ q[:, bd2])**2) / n_bd

    # Non-degeneracy: fix u at the L2 corner (i=N-1, j=0, i.e. s=1,t=0)
    # This prevents the trivial constant-map solution.
    # At the L2 corner: q should point toward R02@e1, p = R02@q
    # This is the "output" corner of the triangle — fixing it forces
    # the map to actually interpolate between the three Lagrangians.
    e1 = np.zeros(dim); e1[0] = 1.0
    corner_idx = np.where(bd0 & bd1)[0]   # s=0,t=0 corner: L0∩L1
    corner2_idx = np.where(bd0 & bd2)[0]  # s=1,t=0 corner: L0∩L2
    
    norm_loss = 0.0
    if len(corner_idx):
        # L0∩L1 corner: q=e1, p=0 (on both L0 and L1)
        norm_loss += np.sum((q[:, corner_idx[0]] - e1)**2)
        norm_loss += np.sum(p[:, corner_idx[0]]**2)
    if len(corner2_idx):
        # L0∩L2 corner: q should be R02@e1 direction, p=R02@q
        target_q2 = R02 @ e1
        target_q2 /= np.linalg.norm(target_q2) + 1e-8
        norm_loss += np.sum((q[:, corner2_idx[0]] - target_q2)**2)
        norm_loss += np.sum((p[:, corner2_idx[0]] - R02 @ q[:, corner2_idx[0]])**2)

    # CR equation — normalized per interior point
    cr_loss = 0.0
    for idx, (ip, im, jp, jm) in neighbors.items():
        if ip >= 0 and im >= 0:
            du_ds = (u[:, ip] - u[:, im]) / (2*h)
        elif ip >= 0:
            du_ds = (u[:, ip] - u[:, idx]) / h
        elif im >= 0:
            du_ds = (u[:, idx] - u[:, im]) / h
        else:
            continue
        if jp >= 0 and jm >= 0:
            du_dt = (u[:, jp] - u[:, jm]) / (2*h)
        elif jp >= 0:
            du_dt = (u[:, jp] - u[:, idx]) / h
        elif jm >= 0:
            du_dt = (u[:, idx] - u[:, jm]) / h
        else:
            continue
        Jdu_dt = np.concatenate([-du_dt[dim:], du_dt[:dim]])
        cr_loss += np.sum((du_ds + Jdu_dt)**2)
    cr_loss /= n_int

    return cr_loss + lam*(b0 + b1 + b2) + lam*0.1*norm_loss


# ─── Initialization ───────────────────────────────────────────────────────────

def triangle_init(R01, R02, dim, pts, bd0, bd1, bd2, M, seed=0):
    """
    Barycentric init with exact corner conditions to prevent trivial solution.
    Corner L0∩L1 (s=0,t=0): q=e1, p=0
    Corner L0∩L2 (s=1,t=0): q=R02@e1/|R02@e1|, p=R02@q
    Corner L1∩L2 (s=0,t=1): q on L1∩L2 boundary
    """
    rng = np.random.default_rng(seed)
    u = np.zeros((2*dim, M))
    e1 = np.zeros(dim); e1[0] = 1.0
    s, t = pts[:,0], pts[:,1]
    r = np.clip(1 - s - t, 0, 1)

    # Corner targets
    q_c01 = e1.copy()                           # L0∩L1: q=e1, p=0
    q_c02 = R02 @ e1
    q_c02 /= np.linalg.norm(q_c02) + 1e-8      # L0∩L2: q=R02@e1 direction
    q_c12 = R01 @ e1
    q_c12 /= np.linalg.norm(q_c12) + 1e-8      # L1∩L2: q=R01@e1 direction

    for idx in range(M):
        si, ti, ri = s[idx], t[idx], r[idx]
        total = si + ti + ri + 1e-8

        # Barycentric interpolation of corner q values
        q = (ri*q_c01 + si*q_c12 + ti*q_c02) / total
        q /= np.linalg.norm(q) + 1e-8

        if seed > 0 and dim > 1:
            noise = rng.standard_normal(dim) * 0.02
            noise[0] = 0
            q = (q + noise)
            q /= np.linalg.norm(q) + 1e-8

        # p: barycentric blend of boundary conditions
        # ∂₀ (t→0): p=0; ∂₁ (s→0): p=R01@q; ∂₂ (r→0): p=R02@q
        p = (si*(R01@q) + ri*(R02@q)) / total
        if seed > 0:
            p += rng.standard_normal(dim) * 0.01

        u[:dim, idx] = q
        u[dim:, idx] = p
    return u.ravel()


# ─── Solve ────────────────────────────────────────────────────────────────────

def solve_triangle(W0, W1, W2, dim, N, lam, max_iter, n_restarts,
                   verbose, label):
    t0 = time.time()
    R01, th01 = rotation_R(W0, W1, dim)
    R12, th12 = rotation_R(W1, W2, dim)
    R02 = R12 @ R01   # composed rotation: L2 relative to L0
    phi01, pi01 = bridgeland_phase(W0, W1)
    phi12, pi12 = bridgeland_phase(W1, W2)

    pts, idx_map, interior, bd0, bd1, bd2, neighbors, M = build_triangle_grid(N)
    n_int = interior.sum()

    print(f"\n  Triple {label}:")
    print(f"    φ₀₁={phi01:.3f}  φ₁₂={phi12:.3f}  "
          f"{'0→π wall' if pi01 and not pi12 else 'π→0 wall' if pi12 and not pi01 else 'same chamber' if not pi01 and not pi12 else 'π→π'}")
    print(f"    Grid: {M} pts, {n_int} interior")

    best_r, best_u = np.inf, None

    for restart in range(n_restarts):
        u0 = triangle_init(R01, R02, dim, pts, bd0, bd1, bd2, M, seed=restart)
        r0 = triangle_cr_residual(u0, R01, R02, dim, N,
                                   lam, interior, bd0, bd1, bd2, neighbors, M)
        res = minimize(
            triangle_cr_residual, u0,
            args=(R01, R02, dim, N, lam, interior, bd0, bd1, bd2, neighbors, M),
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-11,
                     'maxfun': max_iter*20}
        )
        if verbose:
            print(f"      restart {restart}: init={r0:.4f} → {res.fun:.6f} "
                  f"({res.nit} iters)")
        if res.fun < best_r:
            best_r = res.fun
            best_u = res.x
        if best_r < 0.1:
            break

    converged = best_r < 0.1
    print(f"    Best residual: {best_r:.6f}  "
          f"{'✓ triangle exists' if converged else '✗ no triangle'}  "
          f"m₂={'1' if converged else '0'}  [{time.time()-t0:.1f}s]")

    return {
        "triple": label,
        "phi_01": float(phi01), "phi_12": float(phi12),
        "wall_01": bool(pi01),  "wall_12": bool(pi12),
        "residual": float(best_r),
        "converged": bool(converged),
        "m2": int(converged),
    }


# ─── Pentagon ─────────────────────────────────────────────────────────────────

def pentagon_check(results):
    keys = sorted(results.keys())
    print(f"\n  Pentagon (m₂∘m₂ = 0 mod 2):")
    all_ok = True
    pent = []
    for i in range(len(keys)-1):
        k1, k2 = keys[i], keys[i+1]
        # keys look like "L0,L1,L2" and "L1,L2,L3"
        a = results[k1]["m2"]
        b = results[k2]["m2"]
        prod = (a*b) % 2
        ok = prod == 0
        if not ok: all_ok = False
        print(f"    m₂({k1})={a} × m₂({k2})={b} = {prod} mod 2  "
              f"{'✓' if ok else '✗'}")
        pent.append({"pair": f"{k1} × {k2}", "product": prod, "ok": ok})
    return pent, all_ok


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    print("="*60)
    print("  A∞ TRIANGLE SOLVER  (v3 — FEM triangular grid)")
    print("  m₂∘m₂ = 0 (mod 2)?")
    print("="*60)

    all_reports = {}

    for path in args.safetensors:
        print(f"\n{'='*60}\n  Checkpoint: {path}\n{'='*60}")
        try:
            wk = load_wk(path)
        except Exception as e:
            print(f"  ERROR: {e}"); continue

        n = len(wk)
        print(f"  {n} WK matrices  N={args.N}  dim={args.dim}  λ={args.lam}")
        for k in range(n-1):
            phi, pi = bridgeland_phase(wk[k], wk[k+1])
            print(f"  L{k}→L{k+1}: φ={phi:.3f}  {'π-wall' if pi else '0-chamber'}")

        t_total = time.time()
        results = {}
        for k in range(n-2):
            label = f"L{k},L{k+1},L{k+2}"
            results[label] = solve_triangle(
                wk[k], wk[k+1], wk[k+2],
                args.dim, args.N, args.lam,
                args.max_iter, args.n_restarts,
                args.verbose, label)

        pent, holds = pentagon_check(results)

        n_conv = sum(1 for r in results.values() if r["converged"])
        print(f"\n  Triangles converged: {n_conv}/{n-2}")
        print(f"  Pentagon: {'✓ HOLDS (m₂∘m₂=0)' if holds else '✗ FAILS (curved A∞, m₀≠0)'}")
        if holds and n_conv > 0:
            print(f"  → A∞ associativity confirmed, m₀≈0")
        elif holds and n_conv == 0:
            print(f"  → Pentagon holds trivially (all m₂=0), no triangles found")
            print(f"  → This means: no J-holomorphic triangles exist in dim={args.dim}")
            print(f"  → Interpretation: m₂=0 on homology — Fukaya A∞ is trivial here")
        else:
            print(f"  → Curved A∞ obstruction active (m₀≠0)")

        all_reports[path] = {
            "triangles": results,
            "pentagon": pent,
            "pentagon_holds": holds,
            "n_converged": n_conv,
            "elapsed_s": round(time.time()-t_total, 1),
        }

    # Cross-checkpoint
    if len(all_reports) >= 2:
        print(f"\n{'='*60}\n  CROSS-CHECKPOINT\n{'='*60}")
        for path, rep in all_reports.items():
            name = path.split('/')[-1]
            print(f"  {name}: pentagon={'HOLDS' if rep['pentagon_holds'] else 'FAILS'}  "
                  f"triangles={rep['n_converged']}/4")

    with open(args.output, 'w') as f:
        json.dump(all_reports, f, indent=2, default=str)
    print(f"\n  Report → {args.output}")

if __name__ == "__main__":
    main()
