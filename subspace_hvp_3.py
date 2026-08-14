import copy
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LanczosSubspaceNewtonCSAdam:
    """
    Lanczos Subspace Newton:
    1. Uses matrix-free Hessian-Vector Products (exact autograd HVP) to execute
       a K-step Lanczos Krylov iteration.
    2. Builds an orthonormal basis Q_t spanning the dominant eigenspace of H.
    3. Solves the exact Newton system inside the Lanczos Krylov subspace.
    """
    def __init__(self, params, rank=8, lr=1e-2, damping=1e-1, max_trust_radius=0.1):
        self.params = [p for p in params if p.requires_grad]
        self.rank = rank
        self.lr = lr
        self.damping = damping
        self.max_trust_radius = max_trust_radius
        self.p_dim = sum(p.numel() for p in self.params)

    def _get_flat_grads(self):
        return torch.cat([p.grad.view(-1) for p in self.params])

    def _compute_exact_hvp(self, loss, v_flat):
        """
        Computes exact Hessian-vector product H @ v via Pearlmutter double-backpropagation.
        """
        grads = torch.autograd.grad(loss, self.params, create_graph=True)
        grads_flat = torch.cat([g.view(-1) for g in grads])
        
        # Inner product g^T @ v
        grad_v_prod = torch.dot(grads_flat, v_flat)
        
        # Second derivative pass w.r.t params
        hvp = torch.autograd.grad(grad_v_prod, self.params, retain_graph=True)
        return torch.cat([h.contiguous().view(-1) for h in hvp])

    def _run_lanczos_krylov(self, loss):
        """
        Executes K steps of Lanczos iteration using exact HVPs.
        Returns orthonormal basis Q_t (P x K) and tridiagonal subspace Hessian T_K (K x K).
        """
        K = self.rank
        Q = torch.zeros((self.p_dim, K), device=device)
        T_K = torch.zeros((K, K), device=device)

        # Initialize Lanczos with a normalized random seed vector
        v = torch.randn(self.p_dim, device=device)
        v = v / torch.norm(v)
        Q[:, 0] = v

        beta = 0.0
        q_prev = torch.zeros(self.p_dim, device=device)

        for j in range(K):
            q_j = Q[:, j]
            
            # Compute exact Hessian-vector product w = H @ q_j
            w = self._compute_exact_hvp(loss, q_j)

            # Rayleigh quotient alpha_j = q_j^T H q_j
            alpha = torch.dot(q_j, w).item()
            T_K[j, j] = alpha

            # Orthogonalize against previous basis vectors
            w = w - alpha * q_j - beta * q_prev

            # Full Re-orthogonalization (Gram-Schmidt) for numerical stability
            for i in range(j + 1):
                w = w - torch.dot(w, Q[:, i]) * Q[:, i]

            beta = torch.norm(w).item()

            if j < K - 1:
                if beta < 1e-8:
                    # Krylov subspace converged early
                    break
                T_K[j, j + 1] = beta
                T_K[j + 1, j] = beta
                Q[:, j + 1] = w / beta
                q_prev = q_j

        return Q, T_K

    def _apply_flat_update(self, delta_t):
        offset = 0
        for p in self.params:
            if p.grad is None:
                continue
            numel = p.numel()
            p.data.add_(delta_t[offset : offset + numel].view_as(p.data))
            offset += numel

    def step(self, model, loss_fn, x, y):
        # 1. Compute Loss & Gradients (Enable graph for HVP backprop)
        model.zero_grad()
        loss = loss_fn(model(x), y)
        grads = torch.autograd.grad(loss, self.params, create_graph=True)
        grads_flat = torch.cat([g.view(-1) for g in grads]).detach()

        # 2. RUN LANCZOS ITERATION TO COMPUTE EIGENBASIS Q_t AND SUBSPACE HESSIAN T_K
        Q, T_K = self._run_lanczos_krylov(loss)

        # 3. SPECTRAL CLAMPING ON SUBSPACE HESSIAN
        eigenvalues, eigenvectors = torch.linalg.eigh(T_K.detach())
        clamped_eigenvalues = torch.clamp(eigenvalues, min=0.0) + self.damping
        
        # Inverse subspace Hessian via Rayleigh-Ritz decomposition
        T_inv = eigenvectors @ torch.diag(1.0 / clamped_eigenvalues) @ eigenvectors.T

        # 4. SOLVE SUBSPACE NEWTON SYSTEM
        g_K = Q.T @ grads_flat
        delta_K = -T_inv @ g_K

        # Trust-Region Step Bounding
        delta_norm = torch.norm(delta_K).item()
        if delta_norm > self.max_trust_radius:
            delta_K = delta_K * (self.max_trust_radius / delta_norm)

        u_newton = Q @ delta_K

        # Apply update to parameters
        self._apply_flat_update(u_newton)

        return loss.item(), delta_norm, f"Lanczos Newton (top eigs={eigenvalues[-1].item():.2f})"

# ==============================================================================
# BENCHMARK COMPARISON
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

model_newton = SyntheticNet().to(device)
model_adam = copy.deepcopy(model_newton)

opt_newton = LanczosSubspaceNewtonCSAdam(model_newton.parameters(), rank=8, damping=1e-1, max_trust_radius=0.1)
opt_adam = torch.optim.Adam(model_adam.parameters(), lr=1e-2)
loss_fn = nn.CrossEntropyLoss()

print("=" * 115)
print(f"LANCZOS SUBSPACE NEWTON vs. STANDARD ADAM (P = {opt_newton.p_dim:,}, Krylov Rank K=8)")
print("=" * 115)
print(f"{'STEP':>5} | {'LANCZOS NEWTON LOSS':>20} | {'STANDARD ADAM LOSS':>20} | {'TOP EIGENVALUE':>35}")
print("-" * 115)

torch.manual_seed(101)
for step in range(1, 26):
    x = torch.randn(32, 64, device=device)
    y = torch.randint(0, 10, (32,), device=device)

    # 1. Lanczos Subspace Newton Step
    loss_newton_val, d_norm, mode = opt_newton.step(model_newton, loss_fn, x, y)

    # 2. Standard Adam Step
    model_adam.zero_grad()
    out_adam = model_adam(x)
    loss_adam = loss_fn(out_adam, y)
    loss_adam.backward()
    opt_adam.step()

    print(f"{step:5d} | {loss_newton_val:20.4f} | {loss_adam.item():20.4f} | {mode:>35}")

print("=" * 115)
