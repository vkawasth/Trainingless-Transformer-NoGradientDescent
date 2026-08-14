import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

# Reproducibility
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# 1. DYNAMIC ENERGY CS-ADAM OPTIMIZER
# ==============================================================================
class DynamicEnergyCSAdam:
    def __init__(self, params, target_coverage=0.90, margin=0.02, min_rank=4, max_rank=32, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.params = [p for p in params if p.requires_grad]
        self.gamma = target_coverage
        self.margin = margin
        self.min_rank = min_rank
        self.max_rank = max_rank
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        
        self.p_dim = sum(p.numel() for p in self.params)
        self.m = torch.zeros(self.p_dim, device=device)
        self.v = torch.zeros(self.p_dim, device=device)
        self.step_num = 0
        self.k_curr = min_rank
        self.Q = None

    def _get_adam_update_and_momentum(self):
        grads = torch.cat([p.grad.view(-1) for p in self.params])
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads ** 2)
        
        m_hat = self.m / (1 - self.beta1 ** self.step_num)
        v_hat = self.v / (1 - self.beta2 ** self.step_num)
        
        u_t = -self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
        return u_t, self.m

    def step(self):
        self.step_num += 1
        u_t, m_t = self._get_adam_update_and_momentum()
        u_norm = torch.norm(u_t).item()
        m_sq_total = torch.dot(m_t, m_t).item() + 1e-12

        # Warmup Phase
        if self.step_num <= self.k_curr:
            if self.Q is None:
                self.Q = torch.zeros((self.p_dim, self.k_curr), device=device)
            
            q_in = u_t / (u_norm + 1e-12)
            self.Q[:, self.step_num - 1] = q_in

            for k in range(self.step_num):
                for j in range(k):
                    proj = torch.dot(self.Q[:, k], self.Q[:, j])
                    self.Q[:, k] -= proj * self.Q[:, j]
                self.Q[:, k] /= (torch.norm(self.Q[:, k]) + 1e-12)

            delta_t = u_t
            return u_norm, 0.0, 1.0, f"Warmup (K={self.k_curr})"

        # Subspace Projection
        r_t = self.Q.T @ u_t
        u_parallel = self.Q @ r_t
        u_perp = u_t - u_parallel
        residual_norm = torch.norm(u_perp).item()

        # Energy Coverage Calculation
        proj_m = self.Q.T @ m_t
        e = proj_m ** 2
        sorted_e, sorted_idx = torch.sort(e, descending=True)
        coverage = torch.sum(sorted_e).item() / m_sq_total

        q_new = u_perp / (residual_norm + 1e-12)

        # Dynamic Adaptation
        if coverage < (self.gamma - self.margin) and self.k_curr < self.max_rank:
            # EXPAND
            self.Q = torch.cat([self.Q, q_new.unsqueeze(1)], dim=1)
            q_last = self.Q[:, self.k_curr]
            for j in range(self.k_curr):
                q_j = self.Q[:, j]
                q_last -= torch.dot(q_last, q_j) * q_j
            self.Q[:, self.k_curr] = q_last / (torch.norm(q_last) + 1e-12)
            
            self.k_curr += 1
            action = f"EXPAND -> K={self.k_curr}"

        elif (torch.sum(sorted_e[:-1]).item() / m_sq_total) >= self.gamma and self.k_curr > self.min_rank:
            # CONTRACT
            evict_slot = sorted_idx[-1].item()
            keep_slots = [i for i in range(self.k_curr) if i != evict_slot]
            self.Q = self.Q[:, keep_slots]
            self.k_curr -= 1
            action = f"CONTRACT -> K={self.k_curr}"

        else:
            # ROTATE
            evict_slot = sorted_idx[-1].item()
            if evict_slot != self.k_curr - 1:
                self.Q[:, [evict_slot, self.k_curr - 1]] = self.Q[:, [self.k_curr - 1, evict_slot]]

            self.Q[:, self.k_curr - 1] = q_new
            q_last = self.Q[:, self.k_curr - 1]
            for j in range(self.k_curr - 1):
                q_j = self.Q[:, j]
                q_last -= torch.dot(q_last, q_j) * q_j
            self.Q[:, self.k_curr - 1] = q_last / (torch.norm(q_last) + 1e-12)
            action = f"ROTATE (K={self.k_curr})"

        # Apply update
        r_updated = self.Q.T @ u_t
        delta_t = self.Q @ r_updated
        delta_t = delta_t * (u_norm / (torch.norm(delta_t) + 1e-12))

        offset = 0
        for p in self.params:
            if p.grad is None:
                continue
            numel = p.numel()
            p.data.add_(delta_t[offset : offset + numel].view_as(p.data))
            offset += numel

        return u_norm, residual_norm, coverage, action

# ==============================================================================
# 2. SYNTHETIC VERIFICATION RUN
# ==============================================================================
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    def forward(self, x):
        return self.fc(x)

# Setup models
model_cs = SimpleNet().to(device)
model_adam = copy.deepcopy(model_cs)

opt_cs = DynamicEnergyCSAdam(model_cs.parameters(), target_coverage=0.85, min_rank=4, max_rank=16, lr=1e-2)
opt_adam = torch.optim.Adam(model_adam.parameters(), lr=1e-2)

print("=" * 125)
print(f"RUNNING FAST VERIFICATION BENCHMARK ({device.type.upper()}) | Total Parameters P = {opt_cs.p_dim:,}")
print("=" * 125)
print(f"{'STEP':>5} | {'CS-ADAM LOSS':>13} | {'STD ADAM LOSS':>13} | {'||u_t||_2':>10} | {'ENERGY COV %':>13} | {'ACTION / STATE':>25}")
print("-" * 125)

for step in range(1, 31):
    # Synthetic batch
    x = torch.randn(32, 128, device=device)
    y = torch.randint(0, 10, (32,), device=device)

    # CS-Adam Step
    model_cs.zero_grad()
    out_cs = model_cs(x)
    loss_cs = F.cross_entropy(out_cs, y)
    loss_cs.backward()
    u_norm, res_norm, coverage, action = opt_cs.step()

    # Standard Adam Step
    model_adam.zero_grad()
    out_adam = model_adam(x)
    loss_adam = F.cross_entropy(out_adam, y)
    loss_adam.backward()
    opt_adam.step()

    print(f"{step:5d} | {loss_cs.item():13.4f} | {loss_adam.item():13.4f} | {u_norm:10.4f} | {coverage*100:12.1f}% | {action:>25}")

print("=" * 125)
