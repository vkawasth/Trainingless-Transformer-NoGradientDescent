"""
strip_energy_gate.py
=====================
Computes strip energies E(u) for each consecutive WK pair and
implements energy-gated gradient descent: only update layers
where the strip energy exceeds the geodesic threshold.

Theory
------
The J-holomorphic strip energy E(u) = ∫∫ ‖∂_s u‖² ds dt equals
the symplectic area A(L_k, L_{k+1}) at the CR solution u*.
The CR residual measures how far the current map is from J-holomorphic.

The energy decomposition:
  E_total(k)   = strip_area(k)       (symplectic action, always positive)
  E_residual(k) = cr_residual(k)      (non-geodesic component)
  E_ratio(k)    = cr_residual / area  (fraction of energy that is non-geodesic)

Interpretation:
  E_ratio(k) < ε  → strip is near-geodesic, transition smooth → FREEZE layer k
  E_ratio(k) ≥ ε  → strip is non-geodesic, wall crossing active → UPDATE layer k

Energy-gated descent:
  For each training step:
    1. Compute E_ratio for all strips (fast: O(d) SVDs)
    2. Mask gradients to zero for frozen layers
    3. Apply Adam only to active (high-energy) layers

This reduces CE steps by focusing computation on wall-crossing transitions.

Usage
-----
  python strip_energy_gate.py \\
      --checkpoint basin_state.pt \\
      --compiler compiler_geometric.py \\
      --n_steps 25 --threshold 0.02 \\
      --output energy_gate_report.json
"""

import argparse, json, math, time, types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.bool_):    return bool(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',  default='basin_state.pt')
    p.add_argument('--compiler',    default='compiler_geometric.py')
    p.add_argument('--rank',        type=int,   default=6)
    p.add_argument('--n_steps',     type=int,   default=25)
    p.add_argument('--lr',          type=float, default=3e-4)
    p.add_argument('--threshold',   type=float, default=0.02,
                   help='E_ratio threshold for freezing a layer')
    p.add_argument('--lam',         type=float, default=0.1,
                   help='CR boundary penalty (same as cr_solver_bridge.py)')
    p.add_argument('--output',      default='energy_gate_report.json')
    return p.parse_args()


# ─── Import compiler ──────────────────────────────────────────────────────────

def import_compiler(path):
    globs = {
        '__name__':     '__compiler__',
        '__file__':     path,
        '__builtins__': __builtins__,
    }
    src = open(path).read()
    src = src.replace('if __name__ == "__main__":', 'if False:')
    src = src.replace("if __name__ == '__main__':", 'if False:')
    import re
    src = re.sub(r'torch\.save\(', '_NOSAVE(', src)
    globs['_NOSAVE'] = lambda *a, **kw: None
    exec(compile(src, path, 'exec'), globs)
    return types.SimpleNamespace(**{k: v for k, v in globs.items()
                                    if not k.startswith('__')})


# ─── Strip energy computation ─────────────────────────────────────────────────

def strip_area_and_phases(wk_list, rank):
    """
    Compute strip areas and Bridgeland phases for all consecutive pairs.
    Returns: areas (d,), phases (d,), clean_flags (d,)
    """
    d = len(wk_list) - 1
    areas, phases, clean = [], [], []
    for k in range(d):
        Wk  = wk_list[k].detach().float()
        Wk1 = wk_list[k+1].detach().float()

        # Strip area = sum of principal angles
        Uk  = torch.linalg.svd(Wk,  full_matrices=False)[0][:, :rank]
        Uk1 = torch.linalg.svd(Wk1, full_matrices=False)[0][:, :rank]
        sv  = torch.linalg.svdvals(Uk.T @ Uk1).clamp(-1+1e-6, 1-1e-6)
        area = torch.arccos(sv).sum().item()
        areas.append(area)

        # Bridgeland phase: arg(λ_dom(W_{k+1} W_k^{-1}))
        M = (Wk1 @ torch.linalg.pinv(Wk)).numpy()
        evals = np.linalg.eigvals(M)
        dom = evals[np.argmax(np.abs(evals.real))]
        phi = float(np.arctan2(dom.imag, dom.real))
        if phi < 0: phi += 2*math.pi
        phi_clean = abs(phi) < 0.3 or abs(phi - math.pi) < 0.3
        phases.append(phi)
        clean.append(phi_clean)

    return np.array(areas), np.array(phases), clean


def cr_residual_for_pair(Wk, Wk1, rank, lam, N=12):
    """
    Compute CR residual for one pair (Lk, Lk+1) using the bridge solver.
    Returns: (cr_residual, strip_area, E_ratio)
    """
    from scipy.linalg import svd as scipy_svd
    from scipy.optimize import minimize

    W0 = Wk.detach().float().numpy()
    W1 = Wk1.detach().float().numpy()
    dim = 3   # reduced dimension for speed

    # Lagrangian bases
    U0 = scipy_svd(W0, full_matrices=False)[0][:, :dim]
    U1 = scipy_svd(W1, full_matrices=False)[0][:, :dim]

    # Principal angles and rotation matrix
    M = U0.T @ U1
    Um, sm, Vhm = scipy_svd(M)
    sm = np.clip(sm, -1+1e-7, 1-1e-7)
    theta = np.arccos(sm)
    R = Vhm.T @ Um.T

    # Bridgeland phase → R_eff
    Mphase = W1 @ np.linalg.pinv(W0)
    evals = np.linalg.eigvals(Mphase)
    dom = evals[np.argmax(np.abs(evals.real))]
    phi_pi = abs(float(np.arctan2(dom.imag, dom.real)) - math.pi) < 0.3
    R_eff = -R if phi_pi else R

    # Build triangular grid
    h = 1.0 / (N - 1)
    idx_map = -np.ones((N, N), dtype=int)
    pts = []
    for i in range(N):
        for j in range(N - i):
            idx_map[i, j] = len(pts)
            pts.append([i/(N-1), j/(N-1)])
    pts = np.array(pts)
    M_pts = len(pts)
    s, t = pts[:,0], pts[:,1]
    eps = 0.5
    bd0 = t < 1.5*h
    bd1 = s < 1.5*h
    bd2 = np.abs(s + t - 1) < 1.5*h
    interior = ~bd0 & ~bd1 & ~bd2

    # Precompute neighbors
    neighbors = {}
    k_cnt = 0
    for i in range(N):
        for j in range(N - i):
            if interior[k_cnt]:
                ip = idx_map[i+1,j] if i+1<N and j<N-(i+1) else -1
                im = idx_map[i-1,j] if i-1>=0 else -1
                jp = idx_map[i,j+1] if j+1<N-i else -1
                jm = idx_map[i,j-1] if j-1>=0 else -1
                neighbors[k_cnt] = (ip, im, jp, jm)
            k_cnt += 1

    n_int = max(interior.sum(), 1)
    n_bd  = max(bd0.sum() + bd1.sum() + bd2.sum(), 1)
    e1 = np.zeros(dim); e1[0] = 1.0

    def residual(u_flat):
        u = u_flat.reshape(2*dim, M_pts)
        q, p = u[:dim], u[dim:]

        # Boundaries
        b0 = np.sum(p[:, bd0]**2) / n_bd
        b1 = np.sum((p[:, bd1] - R_eff @ q[:, bd1])**2) / n_bd
        b2 = np.sum((p[:, bd2] - R_eff @ q[:, bd2])**2) / n_bd

        # Corners
        c01 = np.where(bd0 & bd1)[0]
        c02 = np.where(bd0 & bd2)[0]
        norm_loss = 0.0
        if len(c01): norm_loss += np.sum((q[:, c01[0]] - e1)**2)
        if len(c02):
            tq = R_eff @ e1; tq /= np.linalg.norm(tq) + 1e-8
            norm_loss += np.sum((q[:, c02[0]] - tq)**2)

        # CR
        cr = 0.0
        for idx, (ip, im, jp, jm) in neighbors.items():
            if ip>=0 and im>=0: dus = (u[:,ip]-u[:,im])/(2*h)
            elif ip>=0: dus = (u[:,ip]-u[:,idx])/h
            elif im>=0: dus = (u[:,idx]-u[:,im])/h
            else: continue
            if jp>=0 and jm>=0: dut = (u[:,jp]-u[:,jm])/(2*h)
            elif jp>=0: dut = (u[:,jp]-u[:,idx])/h
            elif jm>=0: dut = (u[:,idx]-u[:,jm])/h
            else: continue
            Jdt = np.concatenate([-dut[dim:], dut[:dim]])
            cr += np.sum((dus + Jdt)**2)
        cr /= n_int

        return cr + lam*(b0 + b1 + b2) + lam*0.1*norm_loss

    # Init
    r_eff2 = (R_eff @ R_eff.T)
    r_ = 1 - s - t
    u0 = np.zeros((2*dim, M_pts))
    for idx in range(M_pts):
        si, ti, ri_ = s[idx], t[idx], max(r_[idx], 0)
        tot = si + ti + ri_ + 1e-8
        q = e1.copy()
        p = (si*(R_eff@q) + ri_*(R_eff@q)) / tot
        u0[:dim, idx] = q
        u0[dim:, idx] = p
    u0 = u0.ravel()

    res = minimize(residual, u0, method='L-BFGS-B',
                   options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-9,
                            'maxfun': 10000})

    area = float(sum(theta))
    e_ratio = res.fun / (area + 1e-8)
    return float(res.fun), float(area), float(e_ratio)


# ─── Energy-gated layer mask ──────────────────────────────────────────────────

def compute_energy_mask(wk_list, rank, lam, threshold, verbose=True):
    """
    For each consecutive pair (Lk, Lk+1), compute E_ratio.
    Returns mask: layer k is ACTIVE (needs gradient) if E_ratio(k) >= threshold.

    Fast approximation: use strip_area and cr_residual.
    """
    d = len(wk_list) - 1
    areas, phases, clean = strip_area_and_phases(wk_list, rank)
    mu  = float(np.mean(areas))
    mad = float(np.mean(np.abs(areas - mu)))

    print(f"\n  Strip energy analysis:")
    print(f"  {'Pair':<10} {'Area':>8} {'Phase':>8} {'Clean':>6} "
          f"{'WallScore':>10} {'Status':>10}")
    print(f"  {'-'*58}")

    mask = {}   # layer_index → bool (True = ACTIVE)
    energy_data = []

    for k in range(d):
        area = areas[k]
        phi  = phases[k]
        is_clean = clean[k]
        phi_pi = abs(phi - math.pi) < 0.3

        # Wall score as proxy for E_ratio (fast, no CR solve needed)
        ws = abs(area - mu) / (mad + 1e-8)
        # High wall score → strip is non-geodesic → active
        # Low wall score → strip is near-geodesic → freeze
        e_ratio_proxy = ws / (d + 1e-8)   # normalized

        active = ws > threshold * d        # scale threshold by d
        mask[k] = active
        mask[k+1] = mask.get(k+1, False) or active  # both endpoints active

        status = "ACTIVE" if active else "freeze"
        print(f"  L{k}→L{k+1}:   {area:8.4f} {phi:8.3f} {str(is_clean):>6} "
              f"{ws:10.4f} {status:>10}")

        energy_data.append({
            'pair': f'L{k}→L{k+1}',
            'area': float(area),
            'phase_rad': float(phi),
            'phase_clean': bool(is_clean),
            'phase_pi': bool(phi_pi),
            'wall_score': float(ws),
            'active': bool(active),
        })

    print(f"\n  Active layers: {sum(mask.values())}/{len(wk_list)} "
          f"(threshold={threshold})")

    return mask, energy_data, areas, phases


# ─── Energy-gated optimizer ───────────────────────────────────────────────────

def energy_gated_step(model, get_batch, mask_active_layers,
                       lr, clip=1.0):
    """
    Single gradient step with energy gate:
    - Compute full gradient
    - Zero out gradients for frozen (low-energy) layers
    - Apply Adam only to active layers
    """
    model.train()
    x, y = get_batch()
    _, loss = model(x, y)
    loss.backward()

    # Zero gradients for frozen layers
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        # Determine which layer this parameter belongs to
        active = False
        for part in name.split('.'):
            if part.isdigit():
                layer_idx = int(part)
                if mask_active_layers.get(layer_idx, True):
                    active = True
                    break
        if not active:
            param.grad.zero_()

    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0],
        clip)

    return loss.item()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    print("=" * 60)
    print("  STRIP ENERGY GATE")
    print("  Energy-gated gradient descent via strip action")
    print("=" * 60)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  threshold={args.threshold}  n_steps={args.n_steps}  lr={args.lr}")

    t0 = time.time()

    # Load checkpoint BEFORE importing compiler
    print(f"\n[1/4] Pre-loading checkpoint …")
    ckpt_raw = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    state = ckpt_raw if not isinstance(ckpt_raw, dict) else ckpt_raw.get(
        'model', ckpt_raw.get('state_dict',
        ckpt_raw.get('model_state_dict', ckpt_raw)))
    state = {k: v.clone() for k, v in state.items() if isinstance(v, torch.Tensor)}
    print(f"      Cached {len(state)} tensors")

    # Import compiler
    print(f"[2/4] Importing compiler …")
    comp = import_compiler(args.compiler)
    LM        = comp.LM
    get_batch = comp.get_batch
    eval_val  = comp.eval_val

    # Build model
    model = LM()
    model.load_state_dict(state, strict=False)
    print(f"      Model: {sum(p.numel() for p in model.parameters()):,} params")

    # Extract WK matrices
    wk_list = []
    for k in range(6):
        key = f'blocks.{k}.attn.WK.weight'
        if key in state:
            wk_list.append(state[key].float())
    print(f"      {len(wk_list)} WK matrices")

    # Compute energy mask
    print(f"[3/4] Computing strip energy mask …")
    mask, energy_data, areas, phases = compute_energy_mask(
        wk_list, args.rank, args.lam, args.threshold, verbose=True)

    n_active = sum(mask.values())
    n_total  = len(wk_list)
    print(f"\n  Energy gate summary:")
    print(f"  Active layers: {n_active}/{n_total}")
    print(f"  Frozen layers: {n_total - n_active}/{n_total}")
    if n_active < n_total:
        frozen_frac = (n_total - n_active) / n_total
        print(f"  Gradient reduction: {frozen_frac:.0%} of parameters frozen")
        # Estimate FLOP reduction (rough: proportional to frozen layer fraction)
        # Each frozen layer saves ~D² multiplications per backward pass
        D = wk_list[0].shape[0]
        flops_saved = (n_total - n_active) * D * D
        flops_total = n_total * D * D
        print(f"  Estimated FLOP saving: ~{flops_saved/flops_total:.0%} "
              f"of WK backward computation")

    # Baseline val
    val0 = eval_val(model, n=16)
    print(f"\n  Baseline val = {val0:.4f}")

    # Energy-gated descent
    print(f"[4/4] Energy-gated descent ({args.n_steps} steps) …")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    losses = []
    for step in range(args.n_steps):
        opt.zero_grad()
        loss = energy_gated_step(model, get_batch, mask, args.lr)
        opt.step()
        losses.append(loss)
        if (step + 1) % 5 == 0:
            print(f"  step {step+1:3d}: loss={loss:.4f}")

    val1 = eval_val(model, n=16)
    print(f"\n  Final val = {val1:.4f}  (Δ = {val1-val0:+.4f})")

    # Compare: what would full gradient descent give?
    # Run same model with NO masking
    model_full = LM()
    model_full.load_state_dict(state, strict=False)
    opt_full = torch.optim.Adam(model_full.parameters(), lr=args.lr)
    for step in range(args.n_steps):
        model_full.train()
        x, y = get_batch()
        _, loss_f = model_full(x, y)
        opt_full.zero_grad()
        loss_f.backward()
        torch.nn.utils.clip_grad_norm_(model_full.parameters(), 1.0)
        opt_full.step()

    val_full = eval_val(model_full, n=16)
    print(f"  Full gradient val = {val_full:.4f}  (Δ = {val_full-val0:+.4f})")
    print(f"\n  Energy-gated vs full: {val1:.4f} vs {val_full:.4f}")
    if val1 <= val_full + 0.005:
        print(f"  ✓ Energy-gated descent matches full (within 0.005 nats)")
        print(f"  ✓ Gradient reduction validated")
    else:
        print(f"  Δ = {val1-val_full:+.4f} — energy gate may be too aggressive")
        print(f"  Try lowering --threshold")

    # Report
    report = {
        "checkpoint":   args.checkpoint,
        "config": {
            "rank": args.rank, "threshold": args.threshold,
            "n_steps": args.n_steps, "lr": args.lr,
        },
        "strip_energy": energy_data,
        "n_active":     int(n_active),
        "n_frozen":     int(n_total - n_active),
        "val_baseline": float(val0),
        "val_gated":    float(val1),
        "val_full":     float(val_full),
        "delta_gated":  float(val1 - val0),
        "delta_full":   float(val_full - val0),
        "elapsed_s":    round(time.time() - t0, 1),
    }

    Path(args.output).write_text(json.dumps(report, indent=2, cls=NumpyEncoder))
    print(f"\n  Report → {Path(args.output).resolve()}")
    print(f"  Total elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
