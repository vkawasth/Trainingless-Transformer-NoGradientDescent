import torch
import torch.nn as nn
import torch.nn.functional as F

# Standard reproducibility setup
torch.manual_seed(42)

class RotationalCSAdam:
    def __init__(self, params, rank=16, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.params = [p for p in params if p.requires_grad]
        self.rank = rank
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        
        # Total parameters dimension
        self.p_dim = sum(p.numel() for p in self.params)
        
        # Adam states
        self.m = torch.zeros(self.p_dim)
        self.v = torch.zeros(self.p_dim)
        self.step_num = 0
        
        # Subspace Basis Matrix Q in R^(P x K)
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

        # Phase 1: Warmup phase to seed initial Q with K orthogonal vectors
        if self.step_num <= self.rank:
            if self.Q is None:
                self.Q = torch.zeros((self.p_dim, self.rank), device=u_t.device)
            
            q_in = u_t / (u_norm + 1e-12)
            self.Q[:, self.step_num - 1] = q_in

            # Gram-Schmidt sweep up to current rank slot
            for k in range(self.step_num):
                for j in range(k):
                    proj = torch.dot(self.Q[:, k], self.Q[:, j])
                    self.Q[:, k] -= proj * self.Q[:, j]
                self.Q[:, k] /= (torch.norm(self.Q[:, k]) + 1e-12)

            delta_t = u_t
            residual_norm = 0.0
            evicted_idx = -1
        
        # Phase 2: Rotational Subspace Eviction & Single-Vector GS Pass
        else:
            # 1. Measure innovation w.r.t current subspace Q
            r_t = self.Q.T @ u_t
            u_parallel = self.Q @ r_t
            u_perp = u_t - u_parallel
            residual_norm = torch.norm(u_perp).item()

            q_new = u_perp / (residual_norm + 1e-12)

            # 2. Score alignment with momentum vector m_t
            m_norm = m_t / (torch.norm(m_t) + 1e-12)
            energy_scores = torch.abs(self.Q.T @ m_norm)
            evicted_idx = torch.argmin(energy_scores).item()

            # 3. Permute evicted column to slot K-1 to isolate non-evicted vectors 0..K-2
            if evicted_idx != self.rank - 1:
                self.Q[:, [evicted_idx, self.rank - 1]] = self.Q[:, [self.rank - 1, evicted_idx]]

            # 4. Insert q_new directly at slot K-1
            self.Q[:, self.rank - 1] = q_new

            # 5. Single-vector Gram-Schmidt pass: O(P*K)
            # Columns 0..K-2 remain strictly untouched & orthogonal
            q_last = self.Q[:, self.rank - 1]
            for j in range(self.rank - 1):
                q_j = self.Q[:, j]
                q_last -= torch.dot(q_last, q_j) * q_j
            self.Q[:, self.rank - 1] = q_last / (torch.norm(q_last) + 1e-12)

            # 6. Project parameter update through rotated subspace
            r_updated = self.Q.T @ u_t
            delta_t = self.Q @ r_updated
            delta_t = delta_t * (u_norm / (torch.norm(delta_t) + 1e-12))

        # Apply update
        offset = 0
        for p in self.params:
            if p.grad is None:
                continue
            numel = p.numel()
            p.data.add_(delta_t[offset : offset + numel].view_as(p.data))
            offset += numel

        return u_norm, residual_norm, evicted_idx

# Verify execution on synthetic regression landscape
class SyntheticMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

model = SyntheticMLP()
optimizer = RotationalCSAdam(model.parameters(), rank=16, lr=1e-2)
x_dummy = torch.randn(128, 32)
y_dummy = torch.randn(128, 1)

print("=" * 125)
print("CORRECTED ROTATIONAL CS-ADAM (Single-Vector GS Pass at K-1)")
print("=" * 125)
print(f"{'STEP':>5} | {'LOSS':>8} | {'||u_t||_2':>10} | {'||u_perp||_2':>11} | {'INNOVATION %':>13} | {'EVICTED BASIS COL':>18}")
print("-" * 125)

for step_idx in range(1, 201):
    optimizer.params[0].grad = None
    output = model(x_dummy)
    loss = F.mse_loss(output, y_dummy)
    
    model.zero_grad()
    loss.backward()
    
    u_norm, residual_norm, evicted_idx = optimizer.step()
    
    if step_idx == 1 or step_idx % 10 == 0:
        innovation_pct = (residual_norm / (u_norm + 1e-12)) * 100.0
        evicted_str = f"Col #{evicted_idx}" if evicted_idx != -1 else "Warmup (None)"
        print(f"{step_idx:5d} | {loss.item():8.4f} | {u_norm:10.4f} | {residual_norm:11.4f} | {innovation_pct:12.2f}% | {evicted_str:>18}")

print("=" * 125)
