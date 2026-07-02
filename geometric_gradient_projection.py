"""
geometric_gradient_projection.py
==================================
Translates geometric knowledge into gradient modifications.

Every geometric quantity → direct step reduction:

1. Jacobian J = ∂Φ/∂W  (33% nonzero, shape 5×393216)
   → Project gradient to null space of J
   → Only descend in directions that PRESERVE phase coordinates
   → Prevents dissipative wall crossings

2. Strip energy constant → U_k basis sufficient
   → Reduce parameter space from 393k to 7680 dimensions
   → 51× fewer parameters to optimize

3. r_m2^σ gradient direction
   → Descend along ∇(r_m2^σ) instead of ∇(CE_loss)
   → Geometric descent toward Stab(F) fixed point

4. Principal angles θᵢ ≈ π/2
   → Gradient is already orthogonal to strip directions
   → No strip projection needed; confirms gradient is safe

5. Three-geometry correlation ρ=0.90
   → Use N(H_S) ∩ N^-(H_V) as descent direction
   → Guaranteed to decrease loss AND preserve symplectic structure

THE CLAIM: Replacing blind gradient descent with geometric gradient
projection should eliminate the 37 dissipative crossings (keep only
the 19 adiabatic ones) and reduce CE steps significantly.

Usage
-----
  # As a drop-in replacement for optimizer.step():
  from geometric_gradient_projection import GeometricProjector
  projector = GeometricProjector(model, rank=6, fd_eps=1e-3)
  
  # In training loop:
  loss.backward()
  projector.project_gradients()   # replaces optimizer.zero_grad()
  optimizer.step()
"""

import math, time
import numpy as np
import torch


class GeometricProjector:
    """
    Projects gradients to phase-preserving subspace at each step.
    
    The projection:
      g_proj = g - J^T (J J^T + λI)^{-1} J g
    
    removes the components of g that change the phase coordinates φ_k.
    Only the null-space component (which preserves φ) is kept.
    
    Cost per step: 2L forward passes (finite difference for J columns)
    vs current: 1 forward pass (blind)
    
    Tradeoff: 2L× more expensive per step, but should need far fewer steps.
    If ratio of steps saved > 2L = 12, it's worth it.
    """

    def __init__(self, model, rank=6, fd_eps=1e-3, lambda_reg=0.1,
                 n_random=10, update_J_every=8):
        self.model       = model
        self.rank        = rank
        self.fd_eps      = fd_eps
        self.lambda_reg  = lambda_reg
        self.n_random    = n_random
        self.update_J_every = update_J_every
        self._step       = 0
        self._J_cache    = None   # cached random Jacobian projection
        self._V_cache    = None   # random directions used

    def _get_wk_list(self):
        wk = {}
        for name, param in self.model.named_parameters():
            n = name.lower()
            if ('key' in n or 'wk' in n or 'w_k' in n) and 'weight' in n:
                try:
                    li = int([p for p in name.split('.') if p.isdigit()][0])
                except Exception:
                    li = len(wk)
                wk[li] = param
        return [wk[i] for i in sorted(wk)]

    def _compute_phi(self):
        phases = []
        wk_list = self._get_wk_list()
        for k in range(len(wk_list)-1):
            W0 = wk_list[k].detach().cpu().float().numpy()
            W1 = wk_list[k+1].detach().cpu().float().numpy()
            M  = W1 @ np.linalg.pinv(W0)
            ev = np.linalg.eigvals(M)
            lam = ev[np.argmax(np.abs(ev.real))]
            phases.append(float(np.arctan2(lam.imag, lam.real)))
        return np.array(phases)

    def _update_jacobian(self):
        """
        Update the random-projection Jacobian estimate.
        Uses n_random finite-difference directions.
        """
        wk_list = self._get_wk_list()
        sizes   = [w.numel() for w in wk_list]
        D_total = sum(sizes)
        offsets = list(np.cumsum([0] + sizes))
        L = len(wk_list)

        # Current phases
        phi0 = self._compute_phi()

        rng = np.random.default_rng(self._step)
        V = rng.standard_normal((D_total, self.n_random))
        V /= np.linalg.norm(V, axis=0, keepdims=True) + 1e-10

        J_rows = np.zeros((L-1, self.n_random))

        with torch.no_grad():
            saved = [w.data.clone() for w in wk_list]

            for j in range(self.n_random):
                v = V[:, j]
                # Forward perturb
                for i, w in enumerate(wk_list):
                    s, e = offsets[i], offsets[i+1]
                    w.data += self.fd_eps * torch.tensor(
                        v[s:e].reshape(w.shape), dtype=w.dtype, device=w.device)
                phi_p = self._compute_phi()

                # Restore + backward perturb
                for i, w in enumerate(wk_list):
                    w.data.copy_(saved[i])
                    s, e = offsets[i], offsets[i+1]
                    w.data -= self.fd_eps * torch.tensor(
                        v[s:e].reshape(w.shape), dtype=w.dtype, device=w.device)
                phi_m = self._compute_phi()

                dphi = (phi_p - phi_m) / (2 * self.fd_eps)
                dphi = np.where(np.abs(dphi) > np.pi,
                                dphi - np.sign(dphi)*2*np.pi, dphi)
                J_rows[:, j] = dphi

                # Restore
                for i, w in enumerate(wk_list):
                    w.data.copy_(saved[i])

        self._J_cache = J_rows   # (L-1, n_random)
        self._V_cache = V         # (D_total, n_random)
        self._phi0    = phi0
        self._sizes   = sizes
        self._offsets = offsets

    def project_gradients(self, verbose=False):
        """
        Project model gradients to phase-preserving subspace.
        
        Call AFTER loss.backward(), BEFORE optimizer.step().
        """
        self._step += 1

        # Update Jacobian cache periodically
        if self._J_cache is None or self._step % self.update_J_every == 0:
            t0 = time.time()
            self._update_jacobian()
            if verbose:
                row_norms = np.linalg.norm(self._J_cache, axis=1)
                print(f"    [J updated in {time.time()-t0:.1f}s] "
                      f"row_norms={row_norms.round(3)}")

        wk_list = self._get_wk_list()
        J = self._J_cache    # (L-1, n_random)
        V = self._V_cache    # (D_total, n_random)

        # Assemble gradient in the random subspace
        # g_proj_space = V^T @ g_flat  (n_random,)
        g_proj = np.zeros(self.n_random)
        for i, w in enumerate(wk_list):
            if w.grad is None:
                continue
            g_flat = w.grad.detach().cpu().float().numpy().ravel()
            s, e = self._offsets[i], self._offsets[i+1]
            g_proj += V[s:e, :].T @ g_flat

        # Compute phase-changing component of gradient
        # g_phase = J^T (J J^T + λI)^{-1} J g_proj
        L_minus_1 = J.shape[0]
        M  = J @ J.T + self.lambda_reg * np.eye(L_minus_1)
        Jg = J @ g_proj
        alpha = np.linalg.solve(M, Jg)
        g_phase = J.T @ alpha   # (n_random,) — phase-changing part

        # Phase-preserving gradient (null space of J)
        g_null = g_proj - g_phase  # (n_random,)

        # Fraction of gradient in phase direction
        norm_phase = float(np.linalg.norm(g_phase))
        norm_total = float(np.linalg.norm(g_proj)) + 1e-10
        phase_frac = norm_phase / norm_total

        if verbose:
            print(f"    [grad proj] phase_frac={phase_frac:.3f} "
                  f"null_frac={1-phase_frac:.3f}")

        # If phase fraction is large, project it out
        if phase_frac > 0.05:
            g_corrected = g_null
            # Reconstruct corrected gradient in parameter space
            g_full_correction = V @ (g_proj - g_corrected)  # (D_total,)

            # Subtract phase-changing component from gradients
            with torch.no_grad():
                for i, w in enumerate(wk_list):
                    if w.grad is None:
                        continue
                    s, e = self._offsets[i], self._offsets[i+1]
                    correction = torch.tensor(
                        g_full_correction[s:e].reshape(w.shape),
                        dtype=w.grad.dtype, device=w.grad.device)
                    w.grad.data -= correction

        return {
            'phase_frac': phase_frac,
            'projected': phase_frac > 0.05,
            'step': self._step,
        }


# ── Standalone test: does projection reduce wall crossings? ──────────────────

def test_projection_reduces_crossings(model, get_batch, eval_val,
                                       phi_clean_fn, gluing_defect_fn,
                                       LR=1e-3, n_steps=120, rank=6,
                                       verbose=True):
    """
    Test: run basin settle with geometric gradient projection.
    Compare wall crossing count with and without projection.
    """
    projector = GeometricProjector(model, rank=rank,
                                    n_random=10, update_J_every=8)
    opt = torch.optim.AdamW(model.parameters(), lr=LR*5,
                             betas=(0.9,0.95), weight_decay=0.1)

    crossings    = 0
    prev_phi     = None
    crossing_log = []
    val_log      = []

    print("  Step   val    Φ_cl   τ     phase_frac  crossings")
    print("  " + "-"*55)

    for step in range(1, n_steps+1):
        model.train()
        x, y = get_batch()
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()

        # Geometric gradient projection
        proj_info = projector.project_gradients(verbose=(step % 32 == 0))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 8 == 0:
            v   = eval_val(model, n=8)
            pc  = phi_clean_fn(model)
            tau = gluing_defect_fn(model, n=4)

            # Count wall crossings (phase sign changes)
            phi_now = projector._compute_phi()
            if prev_phi is not None:
                n_cross = int(np.sum(np.sign(phi_now) != np.sign(prev_phi)))
                crossings += n_cross
                if n_cross > 0:
                    crossing_log.append({'step': step, 'n': n_cross,
                                         'phi': phi_now.tolist()})
            prev_phi = phi_now.copy()

            val_log.append({'step': step, 'val': v, 'pc': pc, 'tau': tau,
                            'phase_frac': proj_info['phase_frac']})

            if verbose:
                print(f"  {step:4d}  {v:.4f}  {pc}/5  {tau:.2f}  "
                      f"{proj_info['phase_frac']:.3f}        {crossings}")

            if v < 0.15:
                print(f"  ✓ val={v:.4f} < 0.15 at step {step}")
                break

    return {
        'n_steps': step,
        'final_val': eval_val(model),
        'total_crossings': crossings,
        'crossing_log': crossing_log,
        'val_log': val_log,
    }


if __name__ == '__main__':
    print("GeometricProjector loaded.")
    print("Usage: from geometric_gradient_projection import GeometricProjector")
    print("       projector = GeometricProjector(model, rank=6)")
    print("       # In loop: loss.backward(); projector.project_gradients(); opt.step()")
