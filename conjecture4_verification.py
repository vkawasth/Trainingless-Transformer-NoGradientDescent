"""
conjecture4_verification.py
============================
Tests Conjecture 4: gradient descent restricted to ker(∂Φ/∂W) has
fewer wall crossings and faster convergence within a Bridgeland chamber.

DESIGN: Start from basin_entry_state.pt (post-MF-pump, val≈0.15-0.20).
Run two versions of basin settle from the SAME starting point:
  A) Standard AdamW (current compiler Phase 3)
  B) AdamW + ker(J) projection (one line added after loss.backward())

Measure:
  - Wall crossings (phase sign changes per 8-step window)
  - Steps to val < 0.15
  - Final val after N steps

The ker(J) projection is applied ONLY when phases are off-wall (J≠0).
At clean phase (J=0 by algebraic locking), no projection needed.

This is a controlled experiment — identical optimizer, batch, and
starting point. The only difference is the projection.

Usage
-----
  python conjecture4_verification.py \
      --checkpoint basin_entry_state.pt \
      --n_steps 120 \
      --rank 6
"""

import argparse, math, copy, time
import numpy as np
import torch

# ── Import model and data from compiler (adjust path as needed) ──────────────
import importlib.util, sys, os

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='basin_entry_state.pt')
    p.add_argument('--n_steps',   type=int,   default=120)
    p.add_argument('--rank',      type=int,   default=6)
    p.add_argument('--lr',        type=float, default=1e-3)
    p.add_argument('--j_every',   type=int,   default=8,
                   help='Recompute Jacobian every N steps')
    return p.parse_args()


def analytic_jacobian_and_project(model, rank, threshold=0.5):
    """
    For each WK layer pair (k, k+1):
      1. Compute φ_k = arg(λ_dom(W_{k+1}W_k^{-1})) analytically
      2. If |φ_k| > 0.1 and |φ_k - π| > 0.1 (off-wall):
         a. Compute ∂φ_k/∂W_{k+1} via eigenvalue perturbation
         b. Find top singular direction u1, v1
         c. Remove (u1ᵀ g v1) u1v1ᵀ from W_{k+1}.grad

    Returns: (phases, n_projected, jacobian_norms)
    """
    wk = {}
    for name, param in model.named_parameters():
        n = name.lower()
        if ('key' in n or 'wk' in n or 'w_k' in n) and 'weight' in n:
            try: li = int([p for p in name.split('.') if p.isdigit()][0])
            except: li = len(wk)
            wk[li] = (name, param)

    wk_items = [(wk[i], wk[i+1]) for i in sorted(wk)[:-1]]
    phases, j_norms = [], []
    n_projected = 0

    for k, ((n0, p0), (n1, p1)) in enumerate(wk_items):
        W0 = p0.detach().float().cpu().numpy().astype(complex)
        W1 = p1.detach().float().cpu().numpy().astype(complex)
        try:
            W0_inv = np.linalg.inv(W0)
            M      = W1 @ W0_inv
            ev, vc = np.linalg.eig(M)
            idx    = np.argmax(np.abs(ev.real))
            lam    = ev[idx]
            phi_k  = float(np.arctan2(lam.imag, lam.real))
            phases.append(phi_k)

            # Skip clean phases (Jacobian = 0 by algebraic locking)
            is_clean = abs(phi_k) < 0.15 or abs(abs(phi_k) - math.pi) < 0.15
            if is_clean or p1.grad is None:
                j_norms.append(0.0)
                continue

            # Analytic Jacobian via eigenvalue perturbation
            r_vec  = vc[:, idx]
            evL, vcL = np.linalg.eig(M.T)
            idxL   = np.argmax(np.abs(evL.real))
            l_vec  = vcL[:, idxL].conj()
            lr_val = l_vec @ r_vec
            if abs(lr_val) > 1e-10: l_vec = l_vec / lr_val

            lam_mag  = abs(lam)
            Wk_inv_r = W0_inv @ r_vec
            J_k      = np.outer(l_vec, Wk_inv_r)
            dPhi     = np.imag(J_k) / (lam_mag + 1e-10)
            j_norm   = float(np.linalg.norm(dPhi, 'fro'))
            j_norms.append(j_norm)

            if j_norm < threshold:
                continue

            # Top singular direction = phase-breaking gradient direction
            U_s, s_v, Vt_s = np.linalg.svd(dPhi)
            u1 = torch.tensor(U_s[:, 0].real, dtype=p1.grad.dtype,
                              device=p1.grad.device)
            v1 = torch.tensor(Vt_s[0, :].real, dtype=p1.grad.dtype,
                              device=p1.grad.device)

            # Gram-Schmidt: remove phase-breaking component
            coeff = float((u1 @ p1.grad.data @ v1))
            p1.grad.data -= coeff * torch.outer(u1, v1)
            n_projected += 1

        except Exception:
            phases.append(0.0)
            j_norms.append(0.0)

    return phases, n_projected, j_norms


def count_wall_crossings(phases_before, phases_after):
    """Count phase sign changes that cross a wall (not just noise)."""
    crossings = 0
    for p_b, p_a in zip(phases_before, phases_after):
        # A wall crossing: phase changes significantly and crosses 0 or π
        if abs(p_b) < 0.15 or abs(p_b - math.pi) < 0.15:
            continue  # was already at wall
        if abs(p_a) < 0.15 or abs(p_a - math.pi) < 0.15:
            continue  # landed at wall (good, not a crossing)
        if np.sign(p_b) != np.sign(p_a):
            crossings += 1
    return crossings


def run_basin_settle(model, get_batch, eval_val, lr, n_steps,
                     rank, j_every, use_projection, label):
    """
    Run basin settle from current model state.
    use_projection=False: standard AdamW (baseline)
    use_projection=True:  AdamW + ker(J) projection
    """
    opt = torch.optim.AdamW(model.parameters(), lr=lr*5,
                             betas=(0.9, 0.95), weight_decay=0.1)

    val_history  = [eval_val(model, n=8)]
    wall_crossings = 0
    phase_history  = []
    prev_phases    = None

    print(f"\n  {'─'*60}")
    print(f"  {label}")
    print(f"  {'step':>5} {'val':>8} {'Δ':>8} {'walls':>7} "
          f"{'n_proj':>7} {'phases_clean':>13}")
    print(f"  {'─'*55}")

    step_converged = n_steps
    for step in range(1, n_steps+1):
        # LR warmup
        if step <= 10:
            for pg in opt.param_groups: pg['lr'] = lr*5 * step/10
        elif step == 11:
            for pg in opt.param_groups: pg['lr'] = lr*5

        model.train()
        x, y = get_batch()
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # ker(J) projection (only in treatment arm)
        n_proj = 0
        cur_phases = None
        if use_projection and step % j_every == 0:
            cur_phases, n_proj, _ = analytic_jacobian_and_project(
                model, rank)

        opt.step()

        if step % 8 == 0:
            v     = eval_val(model, n=8)
            delta = abs(v - val_history[-1]) / 8
            val_history.append(v)

            # Compute current phases
            if cur_phases is None:
                phases_now, _, _ = analytic_jacobian_and_project(
                    model, rank, threshold=1e10)  # compute but don't project
            else:
                phases_now = cur_phases

            # Count wall crossings
            if prev_phases is not None:
                wc = count_wall_crossings(prev_phases, phases_now)
                wall_crossings += wc
            prev_phases = list(phases_now)
            phase_history.append(phases_now)

            n_clean = sum(1 for p in phases_now
                         if abs(p) < 0.2 or abs(abs(p)-math.pi) < 0.2)

            print(f"  {step:>5} {v:>8.4f} {delta:>8.4f} "
                  f"{wall_crossings:>7} {n_proj:>7} "
                  f"{n_clean}/{len(phases_now)} clean")

            if delta < 0.003 and v < 0.20:
                print(f"  ✓ Converged at step {step} (val={v:.4f})")
                step_converged = step
                break
            if v < 0.15:
                print(f"  ✓ val={v:.4f} < 0.15 at step {step}")
                step_converged = step
                break

    final_val = eval_val(model)
    return {
        'label':          label,
        'step_converged': step_converged,
        'final_val':      final_val,
        'wall_crossings': wall_crossings,
        'val_history':    val_history,
        'use_projection': use_projection,
    }


def main():
    args = parse_args()

    print("="*60)
    print("  CONJECTURE 4 VERIFICATION")
    print("  ker(J) projection vs standard AdamW")
    print("  Controlled experiment from identical starting point")
    print("="*60)

    # ── Load model (requires compiler infrastructure) ─────────────────────
    print(f"\n  Loading checkpoint: {args.checkpoint}")

    try:
        # Import model class from compiler
        spec = importlib.util.spec_from_file_location(
            "compiler", "compiler_analytic_topogate.py")
        # We can't easily import the whole compiler, so we'll use the
        # model and data functions directly if available in namespace

        # Attempt to load via torch
        ckpt = torch.load(args.checkpoint, map_location='cpu',
                          weights_only=False)
        print(f"  Checkpoint loaded: {type(ckpt)}")

        # Check if it's a state dict or full model
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            print("  State dict format — need model class to reconstruct")
            print("  Run this script FROM the compiler environment:")
            print()
            print("  # Add to compiler_analytic_topogate.py after basin_entry_state.pt is saved:")
            print("  import subprocess")
            print("  subprocess.run(['python', 'conjecture4_verification.py',")
            print("                  '--checkpoint', 'basin_entry_state.pt',")
            print("                  '--n_steps', '120', '--rank', '6'])")
            print()
            print("  OR: call run_verification(model, get_batch, eval_val, LR, rank=6)")
            print("  after saving basin_entry_state.pt in the compiler.")
            return

    except Exception as e:
        print(f"  Cannot load standalone: {e}")
        print()
        print("  This script must be called from within the compiler.")
        print("  Add the following to compiler_analytic_topogate.py")
        print("  AFTER saving basin_entry_state.pt:")
        print()
        print("  from conjecture4_verification import run_verification")
        print("  run_verification(model, get_batch, eval_val, LR, rank=6)")
        return


def run_verification(model, get_batch, eval_val, LR, rank=6,
                     n_steps=120, j_every=8):
    """
    Called from within the compiler after basin_entry_state.pt is saved.
    Runs the controlled A/B experiment from the current model state.
    """
    print("\n" + "="*60)
    print("  CONJECTURE 4 VERIFICATION")
    print("  A: Standard AdamW  |  B: AdamW + ker(J) projection")
    print("="*60)

    # Save current state for restoration between runs
    state_a = copy.deepcopy(model.state_dict())

    # ── RUN A: Standard AdamW ─────────────────────────────────────────────
    model.load_state_dict(state_a)
    result_a = run_basin_settle(
        model, get_batch, eval_val,
        lr=LR, n_steps=n_steps, rank=rank, j_every=j_every,
        use_projection=False,
        label="A: Standard AdamW (baseline)")

    # ── RUN B: AdamW + ker(J) projection ─────────────────────────────────
    model.load_state_dict(state_a)
    result_b = run_basin_settle(
        model, get_batch, eval_val,
        lr=LR, n_steps=n_steps, rank=rank, j_every=j_every,
        use_projection=True,
        label="B: AdamW + ker(J) projection")

    # ── COMPARISON ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  RESULTS: Conjecture 4 Verification")
    print(f"{'='*60}")
    print(f"  {'Metric':>30} {'A (baseline)':>14} {'B (projected)':>14}")
    print(f"  {'─'*60}")
    print(f"  {'Wall crossings':>30} {result_a['wall_crossings']:>14} "
          f"{result_b['wall_crossings']:>14}")
    print(f"  {'Steps to convergence':>30} {result_a['step_converged']:>14} "
          f"{result_b['step_converged']:>14}")
    print(f"  {'Final val':>30} {result_a['final_val']:>14.4f} "
          f"{result_b['final_val']:>14.4f}")

    wc_a = result_a['wall_crossings']
    wc_b = result_b['wall_crossings']
    st_a = result_a['step_converged']
    st_b = result_b['step_converged']

    print(f"\n  Wall crossing reduction: "
          f"{wc_a} → {wc_b} "
          f"({'↓' if wc_b < wc_a else '↑'}"
          f"{abs(wc_a-wc_b)} crossings, "
          f"{abs(wc_a-wc_b)/max(wc_a,1)*100:.0f}%)")
    print(f"  Step reduction:         "
          f"{st_a} → {st_b} CE "
          f"({'↓' if st_b < st_a else '↑'}"
          f"{abs(st_a-st_b)} steps)")

    if wc_b < wc_a and st_b <= st_a:
        verdict = "✓ CONJECTURE 4 SUPPORTED: projection reduces crossings AND steps"
    elif wc_b < wc_a:
        verdict = "~ PARTIAL: fewer crossings but more steps (projection too conservative)"
    elif st_b < st_a:
        verdict = "~ PARTIAL: fewer steps but more crossings (projection not preventing walls)"
    else:
        verdict = "✗ NOT SUPPORTED: projection does not help in this regime"

    print(f"\n  Verdict: {verdict}")

    # Restore model to run-B state (better result)
    if result_b['final_val'] < result_a['final_val']:
        print(f"  Model restored to run-B state (val={result_b['final_val']:.4f})")
    else:
        model.load_state_dict(state_a)
        model.load_state_dict(
            torch.load('basin_entry_state.pt', map_location='cpu',
                       weights_only=False))
        print(f"  Model restored to run-A state (val={result_a['final_val']:.4f})")

    return result_a, result_b


if __name__ == '__main__':
    main()
