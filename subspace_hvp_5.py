import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)


class AdaptiveHybridLanczosNewton:
    def __init__(
        self,
        params,
        rank=8,
        lr_newton=0.1,
        lr_ortho=1e-3,
        betas=(0.9, 0.999),
        eps=1e-4,
        damping=1e-2,
        trust_radius=0.1,
    ):
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

    def _run_lanczos_krylov(self, loss):
        K = self.rank
        Q = torch.zeros((self.p_dim, K), device=device)
        T_K = torch.zeros((K, K), device=device)

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

            # Double Gram-Schmidt Re-orthogonalization
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

        # Update moment vectors
        self.m_t = self.beta1 * self.m_t + (1 - self.beta1) * grads_flat
        self.v_t = self.beta2 * self.v_t + (1 - self.beta2) * (grads_flat**2)

        m_hat = self.m_t / (1 - self.beta1**self.step_num)
        v_hat = self.v_t / (1 - self.beta2**self.step_num)

        # Build Krylov Subspace
        Q, T_K = self._run_lanczos_krylov(loss)

        # Robust Eigendecomposition
        T_K_clean = torch.nan_to_num(
            T_K.detach(), nan=0.0, posinf=1e3, neginf=-1e3
        )
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

        # Adaptive Orthogonal Complement Step
        m_parallel = Q @ (Q.T @ m_hat)
        m_perp = m_hat - m_parallel

        v_parallel = Q @ (Q.T @ v_hat)
        v_perp = torch.clamp(v_hat - v_parallel, min=0.0)

        # Stabilized division with higher eps and clipped update vector
        adaptive_scale = torch.clamp(1.0 / (torch.sqrt(v_perp) + self.eps), max=100.0)
        delta_ortho = -self.lr_ortho * (m_perp * adaptive_scale)
        delta_ortho = delta_ortho - Q @ (Q.T @ delta_ortho)

        # Combined Step with Global Trust Region Cap
        delta_combined = delta_newton + delta_ortho
        combined_norm = torch.norm(delta_combined).item()

        if combined_norm > self.trust_radius:
            delta_combined = delta_combined * (self.trust_radius / (combined_norm + 1e-12))

        # Apply parameters update
        offset = 0
        for p in self.params:
            numel = p.numel()
            p.data.add_(delta_combined[offset : offset + numel].view_as(p.data))
            offset += numel

        return loss.item(), eigenvalues[-1].item(), torch.norm(m_perp).item()


class BenchmarkModel(nn.Module):
    def __init__(self, in_features=10, hidden_dim=20, out_features=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_features),
        )

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    print("=" * 65)
    print("RUNNING ADAPTIVE HYBRID LANCZOS NEWTON BENCHMARK (STABILIZED)")
    print("=" * 65)

    model = BenchmarkModel().to(device)
    p_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    loss_fn = nn.MSELoss()

    X = torch.randn(256, 10, device=device) * 2.0
    y = torch.sin(X[:, :1]) * 2.0 + torch.randn(256, 1, device=device) * 0.1

    optimizer = AdaptiveHybridLanczosNewton(
        model.parameters(),
        rank=8,
        lr_newton=0.1,
        lr_ortho=1e-3,
        damping=1e-2,
        trust_radius=0.1,
    )

    print(f"Device: {device}")
    print(f"Total Model Parameters (P): {p_total}")
    print(f"Krylov Subspace Rank (K): 8")
    print("-" * 65)
    print(
        f"{'STEP':<6} | {'LOSS':<12} | {'TOP EIG (λ_max)':<18} | {'||m_perp||':<12}"
    )
    print("-" * 65)

    for step in range(1, 26):
        loss_val, top_eig, m_perp_norm = optimizer.step(model, loss_fn, X, y)
        print(
            f"{step:<6d} | {loss_val:<12.6f} | {top_eig:<18.4f} | {m_perp_norm:<12.4f}"
        )

    print("-" * 65)
    print("Optimization run complete.")
