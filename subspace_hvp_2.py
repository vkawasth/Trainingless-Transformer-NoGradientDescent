import copy
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SpectralSubspaceNewtonCSAdam:
    """
    Spectral Damped Subspace Newton:
    1. Eigendecomposition of H_K with negative eigenvalue clamping (GNS / LM regularization).
    2. Trust-region norm bounding on delta_K.
    3. Coordinate-transformed EMA history for Hessian stability.
    """
    def __init__(self, params, rank=8, lr=1e-2, damping=1e-1, max_trust_radius=0.1, 
                 beta_h=0.9, fd_eps=1e-4, betas=(0.9, 0.999), eps=1e-8):
        self.params = [p for p in params if p.requires_grad]
        self.rank = rank
        self.lr = lr
        self.damping = damping
        self.max_trust_radius = max_trust_radius
        self.beta_h = beta_h
        self.fd_eps = fd_eps
        self.beta1, self.beta2 = betas
        self.eps = eps
        
        self.p_dim = sum(p.numel() for p in self.params)
        self.m = torch.zeros(self.p_dim, device=device)
        self.v = torch.zeros(self.p_dim, device=device)
        self.step_num = 0
        self.Q = None
        self.H_K_ema = None

    def _get_flat_grads(self):
        return torch.cat([p.grad.view(-1) for p in self.params])

    def _get_adam_direction(self, grads):
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads ** 2)
        
        m_hat = self.m / (1 - self.beta1 ** self.step_num)
        v_hat = self.v / (1 - self.beta2 ** self.step_num)
        
        return -self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)

    def _compute_batch_subspace_hessian(self, model, loss_fn, x, y):
        K = self.rank
        H_batch = torch.zeros((K, K), device=device)
        orig_weights = [p.data.clone() for p in self.params]

        for j in range(K):
            q_j = self.Q[:, j]
            
            offset = 0
            for p in self.params:
                numel = p.numel()
                p.data.add_(self.fd_eps * q_j[offset : offset + numel].view_as(p.data))
                offset += numel

            model.zero_grad()
            loss_pos = loss_fn(model(x), y)
            loss_pos.backward()
            g_pos = self._get_flat_grads()

            offset = 0
            for p in self.params:
                numel = p.numel()
                p.data.sub_(2 * self.fd_eps * q_j[offset : offset + numel].view_as(p.data))
                offset += numel

            model.zero_grad()
            loss_neg = loss_fn(model(x), y)
            loss_neg.backward()
            g_neg = self._get_flat_grads()

            for p, w_orig in zip(self.params, orig_weights):
                p.data.copy_(w_orig)

            hvp_j = (g_pos - g_neg) / (2 * self.fd_eps)
            H_batch[:, j] = self.Q.T @ hvp_j

        return 0.5 * (H_batch + H_batch.T)

    def _apply_flat_update(self, delta_t):
        offset = 0
        for p in self.params:
            if p.grad is None:
                continue
            numel = p.numel()
            p.data.add_(delta_t[offset : offset + numel].view_as(p.data))
            offset += numel

    def step(self, model, loss_fn, x, y):
        self.step_num += 1
        
        model.zero_grad()
        out = loss_fn(model(x), y)
        out.backward()
        grads = self._get_flat_grads()
        
        u_adam = self._get_adam_direction(grads)
        u_norm = torch.norm(u_adam).item()

        # Warmup Phase
        if self.step_num <= self.rank:
            if self.Q is None:
                self.Q = torch.zeros((self.p_dim, self.rank), device=device)
            
            self.Q[:, self.step_num - 1] = u_adam / (u_norm + 1e-12)

            for k in range(self.step_num):
                for j in range(k):
                    self.Q[:, k] -= torch.dot(self.Q[:, k], self.Q[:, j]) * self.Q[:, j]
                self.Q[:, k] /= (torch.norm(self.Q[:, k]) + 1e-12)

            self._apply_flat_update(u_adam)
            return out.item(), u_norm, "Warmup (Basis Building)"

        # Save Q before rotation
        Q_old = self.Q.clone()

        # 1. Rotate basis with innovation direction
        u_perp = u_adam - (self.Q @ (self.Q.T @ u_adam))
        q_new = u_perp / (torch.norm(u_perp) + 1e-12)

        e = (self.Q.T @ self.m) ** 2
        evict_idx = torch.argmin(e).item()

        if evict_idx != self.rank - 1:
            self.Q[:, [evict_idx, self.rank - 1]] = self.Q[:, [self.rank - 1, evict_idx]]

        self.Q[:, self.rank - 1] = q_new
        q_last = self.Q[:, self.rank - 1]
        for j in range(self.rank - 1):
            q_last -= torch.dot(q_last, self.Q[:, j]) * self.Q[:, j]
        self.Q[:, self.rank - 1] = q_last / (torch.norm(q_last) + 1e-12)

        # Basis transition matrix R
        R = Q_old.T @ self.Q

        # 2. Accumulate H_K via EMA with coordinate frame rotation
        H_batch = self._compute_batch_subspace_hessian(model, loss_fn, x, y)

        if self.H_K_ema is None:
            self.H_K_ema = H_batch
        else:
            H_ema_rotated = R.T @ self.H_K_ema @ R
            self.H_K_ema = self.beta_h * H_ema_rotated + (1.0 - self.beta_h) * H_batch

        self.H_K_ema = 0.5 * (self.H_K_ema + self.H_K_ema.T)

        # 3. SPECTRAL DAMPING: Clamp eigenvalues to force positive definiteness
        eigenvalues, eigenvectors = torch.linalg.eigh(self.H_K_ema)
        clamped_eigenvalues = torch.clamp(eigenvalues, min=0.0) + self.damping
        
        # Reconstruct inverse via spectral decomposition: H_inv = V * diag(1 / lambda_clamped) * V^T
        H_inv = eigenvectors @ torch.diag(1.0 / clamped_eigenvalues) @ eigenvectors.T

        # 4. Compute Subspace Newton Step
        g_K = self.Q.T @ grads
        delta_K = -H_inv @ g_K

        # Trust-Region Bounding on parameter displacement
        delta_norm = torch.norm(delta_K).item()
        if delta_norm > self.max_trust_radius:
            delta_K = delta_K * (self.max_trust_radius / delta_norm)

        u_newton = self.Q @ delta_K

        # Apply update
        self._apply_flat_update(u_newton)

        return out.item(), u_norm, f"Spectral Newton (radius={min(delta_norm, self.max_trust_radius):.3f})"

# ==============================================================================
# VERIFICATION BENCHMARK
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

opt_newton = SpectralSubspaceNewtonCSAdam(model_newton.parameters(), rank=8, lr=1e-2, damping=1e-1, max_trust_radius=0.1)
opt_adam = torch.optim.Adam(model_adam.parameters(), lr=1e-2)
loss_fn = nn.CrossEntropyLoss()

print("=" * 115)
print(f"SPECTRAL DAMPED NEWTON vs. STANDARD ADAM (P = {opt_newton.p_dim:,}, Rank K=8, Damping=0.1)")
print("=" * 115)
print(f"{'STEP':>5} | {'SPECTRAL NEWTON LOSS':>20} | {'STANDARD ADAM LOSS':>20} | {'MODE':>35}")
print("-" * 115)

torch.manual_seed(101)
for step in range(1, 26):
    x = torch.randn(32, 64, device=device)
    y = torch.randint(0, 10, (32,), device=device)

    loss_newton_val, u_norm, mode = opt_newton.step(model_newton, loss_fn, x, y)

    model_adam.zero_grad()
    out_adam = model_adam(x)
    loss_adam = loss_fn(out_adam, y)
    loss_adam.backward()
    opt_adam.step()

    print(f"{step:5d} | {loss_newton_val:20.4f} | {loss_adam.item():20.4f} | {mode:>35}")

print("=" * 115)
