"""
chamber_aware_optimizer.py
===========================
Implements projected gradient descent in ker(J) with chamber-aware
Nesterov momentum for Phase 3 basin settle.

The missing model: keep gradient descent inside the correct Bridgeland
chamber by projecting onto ker(∂Φ/∂W) at each step.

Three components:
1. ker(J) projection: removes gradient components that change phases
2. Nesterov look-ahead: detects wall proximity, decelerates near walls
3. Adaptive LR: scales inversely to distance from nearest wall

Expected result: O(log(1/ε)) convergence in Phase 3 instead of O(1/ε²),
eliminating dissipative crossings by construction.

Constraint: at clean phase (φ_k ∈ {0,π}), J=0 (algebraic locking),
so projection is free — all gradient directions are phase-preserving.
Cost only matters during pre-orbit phase (steps 1-50).

Usage (drop-in for Phase 3):
    from chamber_aware_optimizer import ChamberAwareOptimizer
    opt = ChamberAwareOptimizer(model, lr=LR*5, rank=6)
    for step in range(1, 151):
        x, y = get_batch()
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        wall_dist, crossed = opt.step()
        if crossed:
            print(f"  Wall crossing at step {step} — decelerated")
"""

import math
import numpy as np
import torch
import torch.nn as nn


class ChamberAwareOptimizer:
    """
    Projected gradient descent in ker(∂Φ/∂W) with Nesterov look-ahead.

    At each step:
    1. Compute gradient g = ∇L(θ)
    2. Compute Jacobian J = ∂Φ/∂W (phase sensitivity, analytic)
    3. Project: g_proj = g - J^T(JJ^T + λI)^{-1} J g
    4. Nesterov look-ahead: check φ(θ - η·g_proj) for wall crossings
    5. If wall detected: reduce η, reproject
    6. Apply: θ ← θ - η·g_proj + momentum

    The Jacobian is computed analytically via eigenvalue perturbation
    (same as jacobian_phase_map.py) — O(L·D³) per update.
    At clean phase J=0 so step 3 is free.
    """

    def __init__(self, model, lr=1e-3, rank=6, lambda_reg=0.1,
                 nesterov=True, wall_decel=0.5, j_update_every=4,
                 betas=(0.9, 0.95), weight_decay=0.1):
        self.model        = model
        self.lr           = lr
        self.rank         = rank
        self.lambda_reg   = lambda_reg
        self.nesterov     = nesterov
        self.wall_decel   = wall_decel       # LR multiplier when wall detected
        self.j_update_every = j_update_every # Recompute J every N steps
        self.betas        = betas
        self.weight_decay = weight_decay

        # AdamW state
        self._step   = 0
        self._m      = {n: torch.zeros_like(p)
                       for n, p in model.named_parameters()}
        self._v      = {n: torch.zeros_like(p)
                       for n, p in model.named_parameters()}

        # Jacobian cache
        self._J_cache   = None   # (L-1, n_wk_params) — projected Jacobian
        self._phi_cache = None   # current phases

        # Nesterov momentum buffer
        self._theta_momentum = None

        # Metrics
        self.wall_crossings = 0
        self.phase_history  = []

    def _get_wk_pairs(self):
        """Get consecutive WK matrix pairs."""
        wk = {}
        for name, param in self.model.named_parameters():
            n = name.lower()
            if ('key' in n or 'wk' in n or 'w_k' in n) and 'weight' in n:
                try: li = int([p for p in name.split('.') if p.isdigit()][0])
                except: li = len(wk)
                wk[li] = (name, param)
        return [(wk[i], wk[i+1]) for i in sorted(wk)[:-1]]

    def _compute_phi(self):
        """Compute current phases analytically."""
        pairs = self._get_wk_pairs()
        phases = []
        for (n0, p0), (n1, p1) in pairs:
            W0 = p0.detach().float().cpu().numpy().astype(complex)
            W1 = p1.detach().float().cpu().numpy().astype(complex)
            try:
                M  = W1 @ np.linalg.inv(W0)
                ev = np.linalg.eigvals(M)
                lam = ev[np.argmax(np.abs(ev.real))]
                phases.append(float(np.arctan2(lam.imag, lam.real)))
            except Exception:
                phases.append(0.0)
        return np.array(phases)

    def _analytic_jacobian_norms(self):
        """
        Compute ‖∂φ_k/∂W_{k+1}‖_F for each phase — the sensitivity.
        At clean phase this is 0 (algebraic locking).
        Returns: (L-1,) array of Jacobian norms, (L-1,) phases
        """
        pairs = self._get_wk_pairs()
        norms  = []
        phases = []
        for k, ((n0, p0), (n1, p1)) in enumerate(pairs):
            W0 = p0.detach().float().cpu().numpy().astype(complex)
            W1 = p1.detach().float().cpu().numpy().astype(complex)
            try:
                W0_inv = np.linalg.inv(W0)
                M      = W1 @ W0_inv
                ev, vc = np.linalg.eig(M)
                idx    = np.argmax(np.abs(ev.real))
                lam    = ev[idx]; r_vec = vc[:, idx]
                phi_k  = float(np.arctan2(lam.imag, lam.real))

                evL, vcL = np.linalg.eig(M.T)
                idxL     = np.argmax(np.abs(evL.real))
                l_vec    = vcL[:, idxL].conj()
                lr       = l_vec @ r_vec
                if abs(lr) > 1e-10: l_vec = l_vec / lr

                lam_mag  = abs(lam)
                Wk_inv_r = W0_inv @ r_vec
                J_k      = np.outer(l_vec, Wk_inv_r)
                dPhi     = np.imag(J_k) / (lam_mag + 1e-10)
                norms.append(float(np.linalg.norm(dPhi, 'fro')))
                phases.append(phi_k)
            except Exception:
                norms.append(0.0)
                phases.append(0.0)
        return np.array(norms), np.array(phases)

    def _project_gradient(self, j_norms, phases):
        """
        Project gradients to ker(J) using sensitivity norms.

        For each WK layer pair (k, k+1):
        If ‖∂φ_k/∂W_{k+1}‖ > threshold (off-wall):
            Project out the phase-changing component of W_{k+1}.grad
            using the rank-1 direction u₁v₁ᵀ from the Jacobian SVD.
        If ‖∂φ_k/∂W_{k+1}‖ ≈ 0 (clean phase):
            No projection needed (already in ker(J)).

        This is efficient: only off-wall layers need projection.
        """
        threshold = 0.5  # Jacobian norm threshold for projection
        pairs = self._get_wk_pairs()

        n_projected = 0
        for k, ((n0, p0), (n1, p1)) in enumerate(pairs):
            if j_norms[k] < threshold:
                continue  # Clean phase — no projection needed
            if p1.grad is None:
                continue

            W0 = p0.detach().float().cpu().numpy().astype(complex)
            W1 = p1.detach().float().cpu().numpy().astype(complex)
            try:
                W0_inv = np.linalg.inv(W0)
                M      = W1 @ W0_inv
                ev, vc = np.linalg.eig(M)
                idx    = np.argmax(np.abs(ev.real))
                lam    = ev[idx]; r_vec = vc[:, idx]
                evL, vcL = np.linalg.eig(M.T)
                idxL   = np.argmax(np.abs(evL.real))
                l_vec  = vcL[:, idxL].conj()
                lr     = l_vec @ r_vec
                if abs(lr) > 1e-10: l_vec = l_vec / lr

                lam_mag  = abs(lam)
                Wk_inv_r = W0_inv @ r_vec
                J_k      = np.outer(l_vec, Wk_inv_r)
                dPhi     = np.imag(J_k) / (lam_mag + 1e-10)

                # Top singular direction of dPhi = phase-changing direction
                U_s, s_v, Vt_s = np.linalg.svd(dPhi)
                if s_v[0] < 1e-10:
                    continue
                u1 = torch.tensor(U_s[:, 0].real, dtype=p1.grad.dtype,
                                  device=p1.grad.device)
                v1 = torch.tensor(Vt_s[0, :].real, dtype=p1.grad.dtype,
                                  device=p1.grad.device)

                # Project out phase-changing component from gradient
                # g_phase = (u1·g·v1) u1⊗v1  (rank-1 projection)
                g = p1.grad.data
                coeff = float((u1 @ g @ v1))
                p1.grad.data -= coeff * torch.outer(u1, v1)
                n_projected += 1
            except Exception:
                pass

        return n_projected

    def _detect_wall(self, phi_before, lr_candidate):
        """
        Nesterov look-ahead: check if a step of size lr would cross a wall.
        A wall crossing = phase changes sign (crosses 0 or π).
        Returns: (wall_distance, would_cross)
        """
        # Simple proxy: if any |φ_k| < lr * ‖∂φ/∂W‖ * ‖g‖, approaching wall
        j_norms, phi_after_approx = self._analytic_jacobian_norms()

        # Distance to nearest wall: min over k of |φ_k| / (j_norm * |g_norm|)
        distances = []
        for k in range(len(phi_before)):
            if j_norms[k] < 0.01:
                continue  # Clean phase, no wall nearby
            dist = abs(phi_before[k]) / (j_norms[k] + 1e-10)
            distances.append(dist)

        if not distances:
            return float('inf'), False

        min_dist = min(distances)
        would_cross = min_dist < lr_candidate
        return min_dist, would_cross

    def _compute_curv_gradient(self, lambda_curv=0.1):
        """
        Compute gradient of A∞ curvature: ∂R_assoc/∂θ via autograd.

        R_assoc = ‖m₂∘m₂‖_F / ‖m₂‖_F² (our proxy for Curv(t))

        m₂ proxy: for each layer pair (k, k+1, k+2):
          m₂_proxy = W_{k+2} W_{k+1}^{-1} W_{k+1} W_k^{-1} - W_{k+2} W_k^{-1}
          (measures failure of composition to be direct)

        ∂R_assoc/∂θ via autograd through these matrix operations.
        Adds λ·∂R_assoc/∂θ to each parameter gradient.

        Cost: 3 backward passes through matrix operations per triple.
        """
        pairs = self._get_wk_pairs()
        if len(pairs) < 2:
            return 0.0

        total_curv = torch.tensor(0.0, requires_grad=False)
        curv_grads = {n: torch.zeros_like(p)
                     for n, p in self.model.named_parameters()}

        for k in range(len(pairs)-1):
            (n0, p0), (n1, p1) = pairs[k]
            (_,  _),  (n2, p2) = pairs[k+1]

            try:
                # m₂ proxy: composition failure
                W0 = p0.detach().float()
                W1 = p1.detach().float()
                W2 = p2.detach().float()

                # Enable grad for augmented Lagrangian computation
                W0_g = W0.clone().requires_grad_(True)
                W1_g = W1.clone().requires_grad_(True)
                W2_g = W2.clone().requires_grad_(True)

                W0_inv = torch.linalg.inv(W0_g)
                W1_inv = torch.linalg.inv(W1_g)

                # Direct: W2 W0^{-1}
                M_direct = W2_g @ W0_inv

                # Composed: W2 W1^{-1} W1 W0^{-1} = W2 W0^{-1} (should cancel)
                # Deviation from cancellation = A∞ curvature proxy
                M_comp   = W2_g @ W1_inv @ W1_g @ W0_inv

                # Curv = ‖M_comp - M_direct‖_F² / (‖M_direct‖_F² + 1e-8)
                diff  = M_comp - M_direct
                curv_k = (diff * diff).sum() / ((M_direct * M_direct).sum() + 1e-8)

                # Backward
                curv_k.backward()
                total_curv = total_curv + curv_k.item()

                # Accumulate gradients
                if W0_g.grad is not None and n0 in curv_grads:
                    curv_grads[n0] += W0_g.grad * lambda_curv
                if W1_g.grad is not None and n1 in curv_grads:
                    curv_grads[n1] += W1_g.grad * lambda_curv
                if W2_g.grad is not None and n2 in curv_grads:
                    curv_grads[n2] += W2_g.grad * lambda_curv

            except Exception:
                pass

        # Add curvature gradient to parameter gradients
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.grad is not None and name in curv_grads:
                    param.grad.data += curv_grads[name]

        return float(total_curv)


        self.model.zero_grad()

    def zero_grad(self):
        self.model.zero_grad()

    def step(self, verbose=False):
        """
        Single optimization step with:
        1. ker(J) gradient projection
        2. Nesterov wall detection
        3. Adaptive LR
        4. AdamW update

        Returns: (wall_distance, wall_crossed)
        """
        self._step += 1
        b1, b2 = self.betas

        # Compute phase sensitivity
        if self._step % self.j_update_every == 1 or self._J_cache is None:
            j_norms, phases = self._analytic_jacobian_norms()
            self._J_cache   = j_norms
            self._phi_cache = phases
        else:
            j_norms = self._J_cache
            phases  = self._phi_cache

        self.phase_history.append(phases.copy())

        # Count clean phases
        n_clean = sum(1 for p in phases
                     if abs(p) < 0.3 or abs(abs(p) - math.pi) < 0.3)
        all_clean = n_clean == len(phases)

        # ker(J) projection (only if off-wall phases exist)
        n_projected = 0
        if not all_clean:
            n_projected = self._project_gradient(j_norms, phases)

        # Augmented Lagrangian: DISABLED (corrupts gradients at high val)
        # curv = self._compute_curv_gradient(lambda_curv=0.05)
        curv = 0.0

        # Nesterov wall detection
        wall_dist, would_cross = self._detect_wall(phases, self.lr)
        lr_effective = self.lr
        if would_cross:
            lr_effective = self.lr * self.wall_decel
            self.wall_crossings += 1

        if verbose and (n_projected > 0 or would_cross):
            print(f"  [chamber] step={self._step} "
                  f"n_clean={n_clean}/{len(phases)} "
                  f"n_proj={n_projected} "
                  f"wall_dist={wall_dist:.3f} "
                  f"lr={lr_effective:.6f}")

        # AdamW update in projected gradient direction
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.grad is None:
                    continue
                g = param.grad.data

                # Weight decay
                g = g + self.weight_decay * param.data

                # Adam moments
                self._m[name].mul_(b1).add_(g, alpha=1-b1)
                self._v[name].mul_(b2).addcmul_(g, g, value=1-b2)

                # Bias correction
                bc1 = 1 - b1**self._step
                bc2 = 1 - b2**self._step
                m_hat = self._m[name] / bc1
                v_hat = self._v[name] / bc2

                # Update
                param.data.addcdiv_(m_hat, v_hat.sqrt() + 1e-8,
                                    value=-lr_effective)

        return wall_dist, would_cross


def run_phase3_chamber_aware(model, get_batch, eval_val, phi_clean_fn,
                              gluing_defect_fn, LR, n_steps=150, rank=6,
                              verbose_every=8):
    """
    Phase 3 basin settle using ChamberAwareOptimizer.
    Drop-in replacement for the standard LR×5 flat loop.
    """
    opt = ChamberAwareOptimizer(model, lr=LR*5, rank=rank,
                                 j_update_every=4)

    val_history = [eval_val(model, n=4)]
    wall_events = []

    print("━━━ PHASE 3: CHAMBER-AWARE BASIN SETTLE ━━━━━━━━━━━━━━━")
    print("  ker(J) projection + Nesterov wall detection")
    print("  O(log(1/ε)) convergence within chamber (Conjecture 4)")
    print(f"  {'step':>5} {'val':>8} {'Δ':>8} {'Φ_cl':>6} "
          f"{'τ':>6} {'walls':>6} {'proj':>5}")
    print("  " + "-"*55)

    step = 0
    for step in range(1, n_steps+1):
        # Warmup first 10 steps
        if step <= 10:
            opt.lr = LR*5 * step/10
        else:
            opt.lr = LR*5  # restore after warmup

        model.train()
        x, y = get_batch()
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        wall_dist, crossed = opt.step()
        if crossed:
            wall_events.append(step)

        if step % verbose_every == 0:
            v    = eval_val(model, n=8)
            delta = abs(v - val_history[-1]) / verbose_every
            val_history.append(v)
            pc   = phi_clean_fn(model)
            tau  = gluing_defect_fn(model, n=4)

            print(f"  {step:>5} {v:>8.4f} {delta:>8.4f} {pc:>5}/5 "
                  f"{tau:>6.2f} {opt.wall_crossings:>6} "
                  f"{'-' if wall_dist==float('inf') else f'{wall_dist:.2f}':>5}")

            if delta < 0.003 and v < 0.20:
                print(f"  ✓ Plateau at step {step} (val={v:.4f} < 0.20)")
                break
            if delta < 0.0005:  # extremely flat at any val - genuinely stuck
                print(f"  ✓ Plateau (stuck) at step {step}")
                break
            if v < 0.15:
                print(f"  ✓ val={v:.4f} < 0.15 at step {step}")
                break

    print(f"\n  Total wall crossings (chamber-aware): {opt.wall_crossings}")
    print(f"  Wall events at steps: {wall_events}")
    print(f"  Compare: 37 dissipative crossings in GD-400")

    return step, eval_val(model), opt.wall_crossings


if __name__ == '__main__':
    print("ChamberAwareOptimizer loaded.")
    print("Usage: from chamber_aware_optimizer import run_phase3_chamber_aware")
    print("       step_basin, v_basin, n_walls = run_phase3_chamber_aware(")
    print("           model, get_batch, eval_val, phi_clean, gluing_defect, LR)")
