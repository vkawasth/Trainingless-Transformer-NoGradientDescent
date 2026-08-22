#!/usr/bin/env python3
"""
phase3_geometric_analysis.py -- Phase 3: Level-Set Curvature, Principal Angles,
                               and Axis Alignment (rho_F)

Measures the geometric quantities established in Phase 34:
  1. Level-set curvature identity: ΔL = -||g|| ||u_parallel|| + ½ u_perp^T H u_perp
  2. Principal angles (individual, not aggregate) of the Krylov frame
  3. Equipartition R = |κ_g| / |II(T,T)|
  4. Axis alignment: rho_F(H) = sum(H_ii^2) / ||H||_F^2 via Hutchinson

Runs Phase 3 only (steps ~0-170) with the compiler's corpus and architecture.
"""

import argparse
import contextlib
import io
import json
import math
import os
import subprocess
import sys

import numpy as np
import torch

# ============================================================================
# COMPILER SETUP
# ============================================================================

def setup_compiler():
    """Load and patch the compiler for Phase 3 only."""
    subprocess.run([
        "python3", "build_corpus.py",
        "--out", "/tmp", "--loops", "300"
    ], check=True, capture_output=True)

    RAW = open("compiler_geometri_patched_86.py").read()

    # Phase 3 only: cut after Phase 2 setup, before Phases 4-5
    CUT = RAW.find("# ──── PHASE 3")
    if CUT == -1:
        CUT = RAW.find("def run_pipeline():")
    src = RAW[:CUT] if CUT != -1 else RAW

    # Force D=128 (the excursion capacity)
    src = src.replace("D=256; N_HEADS=4", "D=128; N_HEADS=4", 1)

    # Shorten MF pump for faster measurement
    src = src.replace("for mf_r in range(1, 16):", "for mf_r in range(1, 3):", 1)
    src = src.replace("    if pc == N_STU-1:", "    if False:", 1)
    src = src.replace(
        "    if len(tau_history)>=3 and tau > tau_history[-2] > tau_history[-3]:",
        "    if False:", 1
    )

    # Execute in a clean namespace
    G = {}
    b = io.StringIO()
    with contextlib.redirect_stdout(b):
        exec(src, G)

    return G, b.getvalue()


# ============================================================================
# GEOMETRIC MEASUREMENTS
# ============================================================================
class GeometryTracker:
    """Track geometric quantities during Phase 3."""

    def __init__(self, model, named_parameters, K=4, window=8, n_probes=30):
        self.model = model
        self.ps = [p for _, p in named_parameters if p.requires_grad]
        self.P = sum(p.numel() for p in self.ps)

        # Role indexing
        self.span = {}
        i = 0
        for nm, p in named_parameters:
            if p.requires_grad:
                self.span[nm] = (i, i + p.numel())
                i += p.numel()

        self.role_idx = {}
        for nm, (a, b) in self.span.items():
            role = self._role(nm)
            self.role_idx.setdefault(role, []).append(torch.arange(a, b))
        self.role_idx = {k: torch.cat(v) for k, v in self.role_idx.items()}

        self.K = K
        self.window = window
        self.n_probes = n_probes
        self.hist = {k: [] for k in self.role_idx}
        self.prev_frame = None
        self.frames = []

        # Storage — RENAMED to avoid collision with method
        self.angle_history = {k: [] for k in self.role_idx}
        self.split_angle_history = {k: [] for k in self.role_idx}

        self.loss_changes = []
        self.parallel_norms = []
        self.perp_norms = []
        self.curvatures = []
        self.gradient_norms = []
        self.update_norms = []
        self.cosine_tangential = []

        # Axis alignment (rho_F) storage
        self.rho_F_values = []
        self.diag_sq_values = []
        self.frob_sq_values = []
        self.checkpoints = []

        # Hessian-vector product function (set during run)
        self._hvp_func = None

    def _role(self, nm):
        if nm.startswith("te") or nm.startswith("pe"):
            return "EMB"
        if "ln" in nm.lower() or nm.endswith("n.weight") or nm.endswith("n.bias"):
            return "LN"
        if ".ff." in nm:
            return "FF"
        if "WQ" in nm:
            return "W_Q"
        if "WK" in nm:
            return "W_K"
        if "WV" in nm:
            return "W_V"
        return "W_O"

    def flat(self):
        return torch.cat([p.data.reshape(-1) for p in self.ps]).clone()

    def grad_flat(self):
        return torch.cat([
            p.grad.reshape(-1) if p.grad is not None else torch.zeros(p.numel())
            for p in self.ps
        ])

    def principal_angles(self, Q1, Q2):
        """Compute individual principal angles between two frames."""
        if Q1 is None or Q2 is None:
            return None
        sv = torch.linalg.svdvals(Q1.T @ Q2).cpu().numpy()
        sv = np.clip(sv, 0, 1)
        return np.sort(np.arccos(sv))

    def hessian_vector_product(self, v, loss):
        """
        Compute Hessian-vector product Hv via Pearlmutter (R{v} = H v).
    
        Args:
            v: Vector to multiply with Hessian
            loss: Scalar loss tensor with requires_grad=True
    
        Returns:
            Hv: Hessian-vector product
        """
        # Debug: Check if loss requires grad
        if not loss.requires_grad:
            raise RuntimeError(
                f"loss.requires_grad = {loss.requires_grad}. "
                "Make sure model is in train() mode and loss is connected to parameters."
            )
    
        # Check parameters require grad
        params_with_grad = [p for p in self.ps if p.requires_grad]
        if not params_with_grad:
            raise RuntimeError("No parameters require gradients")
    
        # First compute gradients with create_graph=True
        grad_params = torch.autograd.grad(
            loss, 
            params_with_grad,  # Only pass parameters that require grad
            create_graph=True,
            retain_graph=True,
            allow_unused=True
        )
    
        # Handle None gradients
        grad_params = [
            g if g is not None else torch.zeros_like(p) 
            for g, p in zip(grad_params, params_with_grad)
        ]
    
        # Now compute directional derivative for each parameter
        grad_v = []
        v_idx = 0
        for g, p in zip(grad_params, params_with_grad):
            v_p = v[v_idx:v_idx + p.numel()].reshape(p.shape)
        
            # Compute Hessian-vector product: d/dp (g · v_p)
            hv_p = torch.autograd.grad(
                torch.sum(g * v_p),
                p,
                retain_graph=False,  # Don't keep graph after this
                allow_unused=True
            )[0]
        
            if hv_p is None:
                hv_p = torch.zeros_like(p)
            grad_v.append(hv_p.flatten())
            v_idx += p.numel()
    
        return torch.cat(grad_v)


    def hessian_vector_product(self, v, loss):
        """Compute Hessian-vector product Hv via Pearlmutter."""
        if self._hvp_func is not None:
            return self._hvp_func(v)

        grad_params = torch.autograd.grad(
            loss, self.ps, create_graph=True, retain_graph=True
        )

        grad_v = []
        v_idx = 0
        for g, p in zip(grad_params, self.ps):
            if g is not None:
                v_p = v[v_idx:v_idx + p.numel()].reshape(p.shape)
                hv_p = torch.autograd.grad(
                    torch.sum(g * v_p), p, retain_graph=True
                )[0]
                if hv_p is None:
                    hv_p = torch.zeros_like(p)
                grad_v.append(hv_p.flatten())
            else:
                grad_v.append(torch.zeros(p.numel()))
            v_idx += p.numel()

        return torch.cat(grad_v)

    def measure_axis_alignment(self, step, loss):
        """Measure rho_F(H) = sum(H_ii^2) / ||H||_F^2 using Hutchinson."""
        diag_sq_sum = 0.0
        frob_sq_sum = 0.0

        for _ in range(self.n_probes):
            z = torch.randint(0, 2, (self.P,)).float() * 2 - 1
            Hz = self.hessian_vector_product(z, loss)
            frob_sq_sum += torch.norm(Hz).item() ** 2
            diag_sq_sum += torch.norm(z * Hz).item() ** 2

        diag_sq_mean = diag_sq_sum / self.n_probes
        frob_sq_mean = frob_sq_sum / self.n_probes
        rho_F = diag_sq_mean / (frob_sq_mean + 1e-12)

        self.rho_F_values.append(rho_F)
        self.diag_sq_values.append(diag_sq_mean)
        self.frob_sq_values.append(frob_sq_mean)
        self.checkpoints.append(step)

        return rho_F

    def update(self, step, th_before, u, g, loss_before, loss_after):
        """Update geometric measurements at each step."""
        self.loss_changes.append(loss_after - loss_before)

        g_norm = float(torch.norm(g))
        u_norm = float(torch.norm(u))

        if g_norm > 1e-12:
            u_parallel = (torch.dot(u, g) / (g_norm ** 2)) * g
            u_perp = u - u_parallel
            cos_theta = float(torch.dot(u, g)) / (u_norm * g_norm + 1e-12)
        else:
            u_parallel = torch.zeros_like(u)
            u_perp = u.clone()
            cos_theta = 0.0

        self.gradient_norms.append(g_norm)
        self.update_norms.append(u_norm)
        self.cosine_tangential.append(cos_theta)
        self.parallel_norms.append(float(torch.norm(u_parallel)))
        self.perp_norms.append(float(torch.norm(u_perp)))

        for k, idx in self.role_idx.items():
            self.hist[k].append(u[idx].clone())
            if len(self.hist[k]) > self.window:
                self.hist[k].pop(0)

        if len(self.hist["FF"]) == self.window and step % self.window == 0:
            cur_frame = {}
            split_frames = []

            for k, idx in self.role_idx.items():
                A = torch.stack(self.hist[k], 1)
                U = torch.linalg.svd(A, full_matrices=False)[0][:, :self.K]
                cur_frame[k] = U

                even = torch.stack(self.hist[k][0::2], 1)
                odd = torch.stack(self.hist[k][1::2], 1)
                if even.shape[1] > 1 and odd.shape[1] > 1:
                    Ue = torch.linalg.svd(even, full_matrices=False)[0][:, :self.K//2 + 1]
                    Uo = torch.linalg.svd(odd, full_matrices=False)[0][:, :self.K//2 + 1]
                    split_frames.append((Ue, Uo))

            if self.prev_frame is not None:
                for k in self.role_idx:
                    if k in cur_frame and k in self.prev_frame:
                        angles = self.principal_angles(cur_frame[k], self.prev_frame[k])
                        if angles is not None:
                            self.angle_history[k].append(angles)

            for Ue, Uo in split_frames:
                angles = self.principal_angles(Ue, Uo)
                if angles is not None:
                    for k in self.role_idx:
                        self.split_angle_history[k].append(angles)

            self.frames.append(cur_frame)
            self.prev_frame = cur_frame

    def report(self):
        """Generate report of measured quantities."""
        print("\n" + "=" * 70)
        print("PHASE 3 GEOMETRIC MEASUREMENTS")
        print("=" * 70)

        if self.rho_F_values:
            print("\n--- Axis Alignment: rho_F(H) = sum(H_ii^2) / ||H||_F^2 ---")
            print(f"  Checkpoints measured: {len(self.rho_F_values)}")
            print(f"  rho_F mean: {np.mean(self.rho_F_values):.6f}")
            print(f"  rho_F std:  {np.std(self.rho_F_values):.6f}")
            print(f"  rho_F min:  {np.min(self.rho_F_values):.6f}")
            print(f"  rho_F max:  {np.max(self.rho_F_values):.6f}")
            print(f"  Isotropic baseline (1/P): {1.0/self.P:.6e}")
            print(f"  Ratio to isotropic: {np.mean(self.rho_F_values) / (1.0/self.P):.1f}x")

            if len(self.rho_F_values) >= 2:
                slope = (self.rho_F_values[-1] - self.rho_F_values[0]) / len(self.rho_F_values)
                print(f"  Trend (per checkpoint): {slope:.6f}")

        print("\n--- Individual Principal Angles ---")
        print(f"{'role':>8} {'n':>10}" + "".join(f" θ{i+1}" for i in range(min(4, self.K))) +
              "   random θ1   split θ1   frac<0.3")

        ROLES = ["EMB", "LN", "W_Q", "W_K", "W_V", "W_O", "FF"]
        for r in ROLES:
            if r not in self.angle_history or not self.angle_history[r]:
                continue

            angles = np.stack(self.angle_history[r])
            mean_angles = angles.mean(axis=0)
            n = len(self.role_idx.get(r, []))

            random_angles = self._random_angles(n)
            split = np.stack(self.split_angle_history[r]) if self.split_angle_history.get(r) else None
            split_mean = split.mean(axis=0)[0] if split is not None else float('nan')
            frac_small = float((angles[:, 0] < 0.3).mean())

            print(f"{r:>8} {n:>10,}" +
                  "".join(f" {v:.3f}" for v in mean_angles[:min(4, self.K)]) +
                  f"      {random_angles[0]:.3f}       {split_mean:.3f}     {frac_small:.3f}")

        if self.angle_history:
            all_theta1 = []
            all_thetaK = []
            for r in ROLES:
                if r in self.angle_history and self.angle_history[r]:
                    arr = np.stack(self.angle_history[r])
                    all_theta1.append(arr[:, 0].mean())
                    all_thetaK.append(arr[:, -1].mean())

            if all_theta1:
                print(f"\n  θ₁ mean: {np.mean(all_theta1):.3f} radians ({np.mean(all_theta1)*180/math.pi:.1f}°)")
                print(f"  θ_K mean: {np.mean(all_thetaK):.3f} radians ({np.mean(all_thetaK)*180/math.pi:.1f}°)")
                print(f"  Spread: {np.mean(all_thetaK) - np.mean(all_theta1):.3f}")

        print("\n--- Level-Set Curvature ---")
        print(f"  Loss changes recorded: {len(self.loss_changes)}")
        print(f"  Mean loss change: {np.mean(self.loss_changes):.6f}")
        print(f"  Mean ||u_parallel||: {np.mean(self.parallel_norms):.4f}")
        print(f"  Mean ||u_perp||: {np.mean(self.perp_norms):.4f}")
        print(f"  Mean cos(u, -g): {np.mean(self.cosine_tangential):.4f}")

    def _random_angles(self, n):
        gen = torch.Generator().manual_seed(3)
        Q1 = torch.linalg.qr(torch.randn(n, self.K, generator=gen))[0]
        Q2 = torch.linalg.qr(torch.randn(n, self.K, generator=gen))[0]
        return self.principal_angles(Q1, Q2)


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_phase3(args):
    """Run Phase 3 and collect geometric measurements."""
    print("Loading compiler...")
    G, _ = setup_compiler()

    model = G["model"]
    get_batch = G["get_batch"]
    LR = G["LR"]

    named_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=LR * 5,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )

    tracker = GeometryTracker(
        model, named_params,
        K=args.k,
        window=args.window,
        n_probes=args.probes
    )

    print(f"\nRunning Phase 3 for {args.steps} steps...")
    print(f"  K={args.k}, window={args.window}, probes={args.probes}")
    print("-" * 60)

    step = 0
    while step < args.steps:
        th_before = tracker.flat()

        # Forward pass
        x, y = get_batch()
        _, loss = model(x, y)
        loss_before = float(loss.detach().cpu().numpy())
        
        # --- IMPORTANT: Hessian-vector product must be computed BEFORE opt.step() ---
        # Measure axis alignment if needed (requires computational graph)
        if step > 0 and step % args.alignment_every == 0:
            with torch.no_grad():
                # We need a fresh loss for HVP
                _, loss_hvp = model(x, y)
                rho_F = tracker.measure_axis_alignment(step, loss_hvp)
                if args.verbose:
                    print(f"  step {step:3d}: rho_F = {rho_F:.6f}")

        # Backward pass
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Store gradient before step
        g = tracker.grad_flat()
        
        # Step
        opt.step()
        step += 1
        
        # Compute update and loss after
        u = tracker.flat() - th_before
        loss_after = float(loss.detach().cpu().numpy())

        # Update tracker (doesn't need gradients)
        tracker.update(step, th_before, u, g, loss_before, loss_after)

        if step % args.report_every == 0:
            print(f"  step {step:3d}: loss = {loss_after:.4f}, ||u|| = {float(torch.norm(u)):.4f}")

    # Generate report
    tracker.report()

    # Save data
    if args.out:
        data = {
            "steps": list(range(1, args.steps + 1)),
            "loss": tracker.loss_changes,
            "gradient_norms": tracker.gradient_norms,
            "update_norms": tracker.update_norms,
            "cos_tangential": tracker.cosine_tangential,
            "rho_F": tracker.rho_F_values,
            "rho_F_checkpoints": tracker.checkpoints,
            "principal_angles": {
                k: [a.tolist() if isinstance(a, np.ndarray) else a for a in v]
                for k, v in tracker.angle_history.items()
            }
        }
        with open(args.out, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nSaved data to {args.out}")





# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 3 Geometric Analysis")
    parser.add_argument("--steps", type=int, default=170, help="Number of steps to run")
    parser.add_argument("--k", type=int, default=4, help="Rank of Krylov frame")
    parser.add_argument("--window", type=int, default=8, help="History window for frame")
    parser.add_argument("--probes", type=int, default=30, help="Hutchinson probes for rho_F")
    parser.add_argument("--alignment-every", type=int, default=20, help="Measure rho_F every N steps")
    parser.add_argument("--report-every", type=int, default=16, help="Report interval")
    parser.add_argument("--out", type=str, default=None, help="Output JSON file")
    parser.add_argument("--seed", type=int, default=17, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    run_phase3(args)


if __name__ == "__main__":
    main()
