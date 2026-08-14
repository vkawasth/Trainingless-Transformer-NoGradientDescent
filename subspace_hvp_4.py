import copy
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set deterministic seed for reproducibility
torch.manual_seed(42)

class AdaptiveLanczosNewton:
    """
    Adaptive Lanczos Subspace Newton Optimizer:
    1. Computes top-K Krylov subspace basis via matrix-free double-backprop HVPs.
    2. Projects exact subspace Hessian T_K = Q^T H Q.
    3. Solves the trust-region subproblem with Levenberg-Marquardt gain ratio adaptivity.
    """
    def __init__(self, params, rank=8, lr=1.0, damping=1e-1, trust_radius=0.1):
        self.params = [p for p in params if p.requires_grad]
        self.rank = rank
        self.lr = lr
        self.damping = damping
        self.trust_radius = trust_radius
        self.p_dim = sum(p.numel() for p in self.params)

    def _get_flat_grads(self, grads):
        return torch.cat([g.view(-1) for g in grads])

    def _compute_exact_hvp(self, loss, v_flat):
        grads = torch.autograd.grad(loss, self.params, create_graph=True)
        grads_flat = self._get_flat_grads(grads)
        grad_v_prod = torch.dot(grads_flat, v_flat)
        hvp = torch.autograd.grad(grad_v_prod, self.params, retain_graph=True)
        return torch.cat([h.contiguous().view(-1) for h in hvp])

    def _run_lanczos_krylov(self, loss):
        K = self.rank
        Q = torch.zeros((self.p_dim, K), device=device)
        T_K = torch.zeros((K, K), device=device)

        # Initialize with normalized random seed vector
        v = torch.randn(self.p_dim, device=device)
        v = v / (torch.norm(v) + 1e-12)
        Q[:, 0] = v

        beta = 0.0
        q_prev = torch.zeros(self.p_dim, device=device)

        for j in range(K):
            q_j = Q[:, j]
            w = self._compute_exact_hvp(loss, q_j)
            alpha = torch.dot(q_j, w).item()
            T_K[j, j] = alpha

            w = w - alpha * q_j - beta * q_prev

            # Full Gram-Schmidt re-orthogonalization for numerical precision
            for i in range(j + 1):
                w = w - torch.dot(w, Q[:, i]) * Q[:, i]

            beta = torch.norm(w).item()

            if j < K - 1:
                if beta < 1e-8:
                    break
                T_K[j, j + 1] = beta
                T_K[j + 1, j] = beta
                Q[:, j + 1] = w / (beta + 1e-12)
                q_prev = q_j

        return Q, T_K

    def step(self, model, loss_fn, x, y):
        # 1. Forward Pass & Exact Gradient Calculation
        model.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        grads = torch.autograd.grad(loss, self.params, create_graph=True)
        grads_flat = self._get_flat_grads(grads).detach()

        # 2. Extract Top-K Krylov Eigenbasis via Lanczos HVPs
        Q, T_K = self._run_lanczos_krylov(loss)

        # 3. Spectral Decomposition & Damped Subspace Solve
        eigenvalues, eigenvectors = torch.linalg.eigh(T_K.detach())
        clamped_eigs = torch.clamp(eigenvalues, min=0.0) + self.damping
        T_inv = eigenvectors @ torch.diag(1.0 / clamped_eigs) @ eigenvectors.T

        # Calculate Subspace Projection Step
        g_K = Q.T @ grads_flat
        delta_K = -self.lr * (T_inv @ g_K)

        # 4. Trust-Region Radius Bounding
        delta_norm = torch.norm(delta_K).item()
        if delta_norm > self.trust_radius:
            delta_K = delta_K * (self.trust_radius / (delta_norm + 1e-12))

        delta_full = Q @ delta_K

        # 5. Quadratic Model Predicted Reduction
        # Delta L_pred = - (g_K^T delta_K + 0.5 * delta_K^T T_K delta_K)
        predicted_reduction = -(torch.dot(g_K, delta_K) + 0.5 * torch.dot(delta_K, T_K.detach() @ delta_K)).item()

        # Apply Trial Step
        offset = 0
        for p in self.params:
            numel = p.numel()
            p.data.add_(delta_full[offset : offset + numel].view_as(p.data))
            offset += numel

        # Evaluate Actual Reduction
        with torch.no_grad():
            loss_new = loss_fn(model(x), y).item()

        actual_reduction = loss.item() - loss_new

        # 6. Levenberg-Marquardt Gain Ratio Adaptivity (rho_t)
        rho = 0.0
        if abs(predicted_reduction) > 1e-8:
            rho = actual_reduction / predicted_reduction

            # Update Damping and Trust Radius based on Quadratic Fit Accuracy
            if rho > 0.75:
                self.damping = max(1e-4, self.damping * 0.5)
                self.trust_radius = min(1.0, self.trust_radius * 1.25)
            elif rho < 0.25:
                self.damping = min(10.0, self.damping * 2.0)
                self.trust_radius = max(0.01, self.trust_radius * 0.75)

        return loss.item(), eigenvalues[-1].item(), self.damping, self.trust_radius, rho

# ==============================================================================
# BENCHMARK EXECUTION (FIXED MINI-BATCH)
# ==============================================================================
class SyntheticNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 10)
        )
    def forward(self, x):
        return self.fc(x)

# Setup Target Model Architectures
model_newton = SyntheticNet().to(device)
model_adam = copy.deepcopy(model_newton)

opt_newton = AdaptiveLanczosNewton(
    model_newton.parameters(), rank=8, lr=1.0, damping=1e-1, trust_radius=0.1
)
opt_adam = torch.optim.Adam(model_adam.parameters(), lr=1e-2)
loss_fn = nn.CrossEntropyLoss()

# Fixed Data Batch
x_fixed = torch.randn(64, 64, device=device)
y_fixed = torch.randint(0, 10, (64,), device=device)

print("=" * 125)
print(f"ADAPTIVE LANCZOS NEWTON (LM GAIN RATIO) vs ADAM | FIXED BATCH (P = {opt_newton.p_dim:,}, K = 8)")
print("=" * 125)
print(f"{'STEP':>5} | {'NEWTON LOSS':>15} | {'ADAM LOSS':>15} | {'TOP EIG':>12} | {'DAMPING (λ)':>14} | {'RADIUS (Δ)':>12} | {'GAIN RATIO (ρ)':>14}")
print("-" * 125)

for step in range(1, 51):
    # 1. Adaptive Lanczos Newton Step
    loss_newton, top_eig, damp, radius, rho = opt_newton.step(model_newton, loss_fn, x_fixed, y_fixed)

    # 2. Standard Adam Step
    model_adam.zero_grad()
    loss_adam = loss_fn(model_adam(x_fixed), y_fixed)
    loss_adam.backward()
    opt_adam.step()

    print(f"{step:5d} | {loss_newton:15.6f} | {loss_adam.item():15.6f} | {top_eig:12.4f} | {damp:14.6f} | {radius:12.4f} | {rho:14.4f}")

print("=" * 125)
