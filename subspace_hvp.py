import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SubspaceNewtonCSAdam:
    """
    Subspace Newton CS-Adam: Uses Adam to maintain an optimal basis Q_t, 
    then solves an EXACT local Newton step (H_K + damping)^(-1) g_K inside R^K.
    """
    def __init__(self, params, rank=16, lr=1e-2, damping=1e-2, fd_eps=1e-4, betas=(0.9, 0.999), eps=1e-8):
        self.params = [p for p in params if p.requires_grad]
        self.rank = rank
        self.lr = lr
        self.damping = damping
        self.fd_eps = fd_eps
        self.beta1, self.beta2 = betas
        self.eps = eps
        
        self.p_dim = sum(p.numel() for p in self.params)
        self.m = torch.zeros(self.p_dim, device=device)
        self.v = torch.zeros(self.p_dim, device=device)
        self.step_num = 0
        self.Q = None

    def _get_flat_grads(self):
        return torch.cat([p.grad.view(-1) for p in self.params])

    def _get_adam_direction(self, grads):
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads ** 2)
        
        m_hat = self.m / (1 - self.beta1 ** self.step_num)
        v_hat = self.v / (1 - self.beta2 ** self.step_num)
        
        return -self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)

    def _compute_subspace_hessian(self, model, loss_fn, x, y):
        """
        Computes H_K = Q^T H Q (K x K) using directional finite differences of gradients.
        Requires K forward/backward evaluations across the subspace directions.
        """
        K = self.rank
        H_K = torch.zeros((K, K), device=device)
        
        # Save current weights
        orig_weights = [p.data.clone() for p in self.params]

        for j in range(K):
            q_j = self.Q[:, j]
            
            # Step forward along basis direction q_j
            offset = 0
            for p in self.params:
                numel = p.numel()
                p.data.add_(self.fd_eps * q_j[offset : offset + numel].view_as(p.data))
                offset += numel

            # Forward + Backward at +eps
            model.zero_grad()
            loss_pos = loss_fn(model(x), y)
            loss_pos.backward()
            g_pos = self._get_flat_grads()

            # Step backward along basis direction q_j
            offset = 0
            for p in self.params:
                numel = p.numel()
                p.data.sub_(2 * self.fd_eps * q_j[offset : offset + numel].view_as(p.data))
                offset += numel

            # Forward + Backward at -eps
            model.zero_grad()
            loss_neg = loss_fn(model(x), y)
            loss_neg.backward()
            g_neg = self._get_flat_grads()

            # Restore original weights
            for p, w_orig in zip(self.params, orig_weights):
                p.data.copy_(w_orig)

            # Finite difference HVP: H @ q_j
            hvp_j = (g_pos - g_neg) / (2 * self.fd_eps)

            # Project HVP onto subspace: Col j of H_K = Q^T @ (H @ q_j)
            H_K[:, j] = self.Q.T @ hvp_j

        # Symmetrize H_K to eliminate numerical asymmetry
        H_K = 0.5 * (H_K + H_K.T)
        return H_K

    def step(self, model, loss_fn, x, y):
        self.step_num += 1
        
        # Standard gradient at current point
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

            # Apply standard step during warmup
            self._apply_flat_update(u_adam)
            return out.item(), u_norm, "Warmup (Basis Building)"

        # ----------------------------------------------------------------------
        # 1. COMPUTE SUBSPACE HESSIAN H_K = Q^T H Q  (K x K)
        # ----------------------------------------------------------------------
        H_K = self._compute_subspace_hessian(model, loss_fn, x, y)

        # 2. Project current gradient into subspace: g_K = Q^T g
        g_K = self.Q.T @ grads

        # 3. Solve Subspace Newton System: delta_K = -(H_K + damping * I)^(-1) @ g_K
        H_damped = H_K + self.damping * torch.eye(self.rank, device=device)
        try:
            delta_K = -torch.linalg.solve(H_damped, g_K)
        except RuntimeError:
            # Fallback if singular
            delta_K = -0.01 * g_K

        # 4. Lift Subspace Newton Step back to Full Dimension R^P
        u_newton = self.Q @ delta_K

        # Scale Newton step magnitude relative to Adam step for stability
        u_newton_norm = torch.norm(u_newton).item()
        if u_newton_norm > 0:
            u_newton = u_newton * (u_norm / u_newton_norm)

        # ----------------------------------------------------------------------
        # 5. ROTATE BASIS WITH ADAM'S INNOVATION VECTOR
        # ----------------------------------------------------------------------
        u_perp = u_adam - (self.Q @ (self.Q.T @ u_adam))
        residual_norm = torch.norm(u_perp).item()
        q_new = u_perp / (residual_norm + 1e-12)

        e = (self.Q.T @ self.m) ** 2
        evict_idx = torch.argmin(e).item()

        if evict_idx != self.rank - 1:
            self.Q[:, [evict_idx, self.rank - 1]] = self.Q[:, [self.rank - 1, evict_idx]]

        self.Q[:, self.rank - 1] = q_new
        q_last = self.Q[:, self.rank - 1]
        for j in range(self.rank - 1):
            q_last -= torch.dot(q_last, self.Q[:, j]) * self.Q[:, j]
        self.Q[:, self.rank - 1] = q_last / (torch.norm(q_last) + 1e-12)

        # 6. Apply Subspace Newton Update
        self._apply_flat_update(u_newton)

        return out.item(), u_norm, "Subspace Newton Step"

    def _apply_flat_update(self, delta_t):
        offset = 0
        for p in self.params:
            if p.grad is None:
                continue
            numel = p.numel()
            p.data.add_(delta_t[offset : offset + numel].view_as(p.data))
            offset += numel

# ==============================================================================
# VERIFICATION RUN: SUBSPACE NEWTON vs STANDARD ADAM
# ==============================================================================
class SyntheticNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.Tanh(),  # Non-zero second derivatives for Hessian benchmark
            nn.Linear(32, 10)
        )
    def forward(self, x):
        return self.fc(x)

model_newton = SyntheticNet().to(device)
model_adam = copy.deepcopy(model_newton)

opt_newton = SubspaceNewtonCSAdam(model_newton.parameters(), rank=8, lr=1e-2, damping=1e-2)
opt_adam = torch.optim.Adam(model_adam.parameters(), lr=1e-2)
loss_fn = nn.CrossEntropyLoss()

print("=" * 110)
print(f"CURVATURE REDUCTION: SUBSPACE NEWTON vs. STANDARD ADAM (P = {opt_newton.p_dim:,}, Rank K=8)")
print("=" * 110)
print(f"{'STEP':>5} | {'SUBSPACE NEWTON LOSS':>22} | {'STANDARD ADAM LOSS':>20} | {'MODE':>25}")
print("-" * 110)

torch.manual_seed(101)
for step in range(1, 26):
    x = torch.randn(32, 64, device=device)
    y = torch.randint(0, 10, (32,), device=device)

    # 1. Subspace Newton Step (Inverts K x K Subspace Hessian H_K)
    loss_newton_val, u_norm, mode = opt_newton.step(model_newton, loss_fn, x, y)

    # 2. Standard Adam Step
    model_adam.zero_grad()
    out_adam = model_adam(x)
    loss_adam = loss_fn(out_adam, y)
    loss_adam.backward()
    opt_adam.step()

    print(f"{step:5d} | {loss_newton_val:22.4f} | {loss_adam.item():20.4f} | {mode:>25}")

print("=" * 110)
