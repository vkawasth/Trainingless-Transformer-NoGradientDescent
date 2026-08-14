import torch
import torch.nn as nn
import copy
from torch.optim import Adam

# Set device and seed for exact reproducibility
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)


# =====================================================================
# 1. OPTIMIZER DEFINITIONS (GRADIENT-SEEDED LANCZOS)
# =====================================================================

class HybridLanczosNewton:
    """Hybrid Lanczos-Newton with gradient-seeded Krylov subspace."""
    def __init__(self, params, rank=8, lr_newton=0.1, lr_ortho=0.01, damping=1e-2, trust_radius=0.1):
        self.params = [p for p in params if p.requires_grad]
        self.rank = rank
        self.lr_newton = lr_newton
        self.lr_ortho = lr_ortho
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
        return torch.cat([h.contiguous().view(-1) for h in hvp]).detach()

    def _run_lanczos_krylov(self, loss, grads_flat):
        K = self.rank
        Q = torch.zeros((self.p_dim, K), device=device)
        T_K = torch.zeros((K, K), device=device)

        # Seed Lanczos with the normalized gradient vector v1 = g / ||g||
        g_norm = torch.norm(grads_flat)
        if g_norm > 1e-12:
            v = grads_flat / g_norm
        else:
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

            for _ in range(2):
                for i in range(j + 1):
                    w = w - torch.dot(w, Q[:, i]) * Q[:, i]

            beta = torch.norm(w).item()

            if j < K - 1:
                if beta < 1e-8 or not torch.isfinite(torch.tensor(beta)):
                    Q = Q[:, : j + 1]
                    T_K = T_K[: j + 1, : j + 1]
                    break
                T_K[j, j + 1] = beta
                T_K[j + 1, j] = beta
                Q[:, j + 1] = w / (beta + 1e-12)
                q_prev = q_j

        T_K = 0.5 * (T_K + T_K.T)
        return Q, T_K

    def step(self, model, loss_fn, x, y):
        model.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)

        grads = torch.autograd.grad(loss, self.params, create_graph=True)
        grads_flat = self._get_flat_grads(grads).detach()

        Q, T_K = self._run_lanczos_krylov(loss, grads_flat)

        T_K_clean = torch.nan_to_num(T_K.detach(), nan=0.0, posinf=1e3, neginf=-1e3)
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(T_K_clean)
        except torch._C._LinAlgError:
            eigenvalues, eigenvectors = torch.linalg.eig(T_K_clean)
            eigenvalues = eigenvalues.real
            eigenvectors = eigenvectors.real

        clamped_eigs = torch.clamp(eigenvalues, min=0.0) + self.damping
        T_inv = eigenvectors @ torch.diag(1.0 / clamped_eigs) @ eigenvectors.T

        g_K = Q.T @ grads_flat
        delta_K = -self.lr_newton * (T_inv @ g_K)
        delta_newton = Q @ delta_K

        m_parallel = Q @ (Q.T @ grads_flat)
        m_perp = grads_flat - m_parallel
        delta_ortho = -self.lr_ortho * m_perp
        delta_ortho = delta_ortho - Q @ (Q.T @ delta_ortho)

        delta_combined = delta_newton + delta_ortho
        combined_norm = torch.norm(delta_combined).item()

        if combined_norm > self.trust_radius:
            delta_combined = delta_combined * (self.trust_radius / (combined_norm + 1e-12))

        offset = 0
        for p in self.params:
            numel = p.numel()
            p.data.add_(delta_combined[offset : offset + numel].view_as(p.data))
            offset += numel

        return loss.item()


class AdaptiveHybridLanczosNewton:
    """Adaptive Hybrid Lanczos-Newton with gradient-seeded Krylov subspace."""
    def __init__(self, params, rank=8, lr_newton=0.1, lr_ortho=0.01, betas=(0.9, 0.999), eps=1e-4, damping=1e-2, trust_radius=0.1):
        self.params = [p for p in params if p.requires_grad]
        self.rank = rank
        self.lr_newton = lr_newton
        self.lr_ortho = lr_ortho
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.damping = damping
        self.trust_radius = trust_radius
        self.p_dim = sum(p.numel() for p in self.params)

        self.step_num = 0
        self.m_t = torch.zeros(self.p_dim, device=device)
        self.v_t = torch.zeros(self.p_dim, device=device)

    def _get_flat_grads(self, grads):
        return torch.cat([g.view(-1) for g in grads])

    def _compute_exact_hvp(self, loss, v_flat):
        grads = torch.autograd.grad(loss, self.params, create_graph=True)
        grads_flat = self._get_flat_grads(grads)
        grad_v_prod = torch.dot(grads_flat, v_flat)
        hvp = torch.autograd.grad(grad_v_prod, self.params, retain_graph=True)
        return torch.cat([h.contiguous().view(-1) for h in hvp]).detach()

    def _run_lanczos_krylov(self, loss, grads_flat):
        K = self.rank
        Q = torch.zeros((self.p_dim, K), device=device)
        T_K = torch.zeros((K, K), device=device)

        # Seed Lanczos with the normalized gradient vector v1 = g / ||g||
        g_norm = torch.norm(grads_flat)
        if g_norm > 1e-12:
            v = grads_flat / g_norm
        else:
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

            for _ in range(2):
                for i in range(j + 1):
                    w = w - torch.dot(w, Q[:, i]) * Q[:, i]

            beta = torch.norm(w).item()

            if j < K - 1:
                if beta < 1e-8 or not torch.isfinite(torch.tensor(beta)):
                    Q = Q[:, : j + 1]
                    T_K = T_K[: j + 1, : j + 1]
                    break
                T_K[j, j + 1] = beta
                T_K[j + 1, j] = beta
                Q[:, j + 1] = w / (beta + 1e-12)
                q_prev = q_j

        T_K = 0.5 * (T_K + T_K.T)
        return Q, T_K

    def step(self, model, loss_fn, x, y):
        self.step_num += 1

        model.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)

        grads = torch.autograd.grad(loss, self.params, create_graph=True)
        grads_flat = self._get_flat_grads(grads).detach()

        self.m_t = self.beta1 * self.m_t + (1 - self.beta1) * grads_flat
        self.v_t = self.beta2 * self.v_t + (1 - self.beta2) * (grads_flat**2)

        m_hat = self.m_t / (1 - self.beta1**self.step_num)
        v_hat = self.v_t / (1 - self.beta2**self.step_num)

        Q, T_K = self._run_lanczos_krylov(loss, grads_flat)

        T_K_clean = torch.nan_to_num(T_K.detach(), nan=0.0, posinf=1e3, neginf=-1e3)
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(T_K_clean)
        except torch._C._LinAlgError:
            eigenvalues, eigenvectors = torch.linalg.eig(T_K_clean)
            eigenvalues = eigenvalues.real
            eigenvectors = eigenvectors.real

        clamped_eigs = torch.clamp(eigenvalues, min=0.0) + self.damping
        T_inv = eigenvectors @ torch.diag(1.0 / clamped_eigs) @ eigenvectors.T

        g_K = Q.T @ grads_flat
        delta_K = -self.lr_newton * (T_inv @ g_K)
        delta_newton = Q @ delta_K

        m_parallel = Q @ (Q.T @ m_hat)
        m_perp = m_hat - m_parallel

        v_parallel = Q @ (Q.T @ v_hat)
        v_perp = torch.clamp(v_hat - v_parallel, min=0.0)

        adaptive_scale = torch.clamp(1.0 / (torch.sqrt(v_perp) + self.eps), max=100.0)
        delta_ortho = -self.lr_ortho * (m_perp * adaptive_scale)
        delta_ortho = delta_ortho - Q @ (Q.T @ delta_ortho)

        delta_combined = delta_newton + delta_ortho
        combined_norm = torch.norm(delta_combined).item()

        if combined_norm > self.trust_radius:
            delta_combined = delta_combined * (self.trust_radius / (combined_norm + 1e-12))

        offset = 0
        for p in self.params:
            numel = p.numel()
            p.data.add_(delta_combined[offset : offset + numel].view_as(p.data))
            offset += numel

        return loss.item()


# =====================================================================
# 2. BENCHMARK EXECUTION
# =====================================================================

class SyntheticLandscapeModel(nn.Module):
    def __init__(self, in_features=10, hidden_dim=20, out_features=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_features),
        )

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    print("=" * 70)
    print("BENCHMARK (GRADIENT-SEEDED): ADAM vs HYBRID vs ADAPTIVE")
    print("=" * 70)

    base_model = SyntheticLandscapeModel().to(device)
    loss_fn = nn.MSELoss()

    scales = torch.tensor([1e-2, 1e-1, 1.0, 5.0, 10.0, 50.0, 100.0, 250.0, 500.0, 1000.0], device=device)
    X = torch.randn(256, 10, device=device) * scales
    y = torch.sin(X[:, :1]) * 2.0 + torch.randn(256, 1, device=device) * 0.05

    model_adam = copy.deepcopy(base_model)
    model_hybrid = copy.deepcopy(base_model)
    model_adaptive = copy.deepcopy(base_model)

    opt_adam = Adam(model_adam.parameters(), lr=1e-3)
    opt_hybrid = HybridLanczosNewton(model_hybrid.parameters(), rank=8, lr_newton=0.1, lr_ortho=0.01)
    opt_adaptive = AdaptiveHybridLanczosNewton(model_adaptive.parameters(), rank=8, lr_newton=0.1, lr_ortho=0.01)

    steps = 25
    print(f"Device: {device}")
    print(f"Total Parameters: {sum(p.numel() for p in base_model.parameters())}")
    print(f"Krylov Rank K: 8 | lr_ortho: 0.01")
    print("-" * 70)
    print(f"{'STEP':<6} | {'ADAM LOSS':<16} | {'HYBRID LOSS':<16} | {'ADAPTIVE HYBRID LOSS':<18}")
    print("-" * 70)

    for step in range(1, steps + 1):
        opt_adam.zero_grad()
        out_adam = model_adam(X)
        loss_a = loss_fn(out_adam, y)
        loss_a.backward()
        opt_adam.step()

        loss_h = opt_hybrid.step(model_hybrid, loss_fn, X, y)
        loss_ad = opt_adaptive.step(model_adaptive, loss_fn, X, y)

        print(f"{step:<6d} | {loss_a:<16.6f} | {loss_h:<16.6f} | {loss_ad:<18.6f}")

    print("-" * 70)
