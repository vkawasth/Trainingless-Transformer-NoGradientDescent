import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Reproducibility
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# 1. ROTATIONAL CS-ADAM OPTIMIZER (WITH NOISE-THRESHOLD GUARD)
# ==============================================================================
# Changed rescaling to subspace.
class RotationalCSAdam:
    def __init__(self, params, rank=16, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, rescale=True, norm_threshold=1e-4):
    #def __init__(self, params, rank=32, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, norm_threshold=1e-4):
        self.params = [p for p in params if p.requires_grad]
        self.rank = rank
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.norm_threshold = norm_threshold
        
        self.p_dim = sum(p.numel() for p in self.params)
        self.m = torch.zeros(self.p_dim, device=device)
        self.v = torch.zeros(self.p_dim, device=device)
        self.step_num = 0
        self.Q = None
        self.rescale = rescale

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

        # Threshold guard to avoid eviction on numerical noise
        if u_norm < self.norm_threshold:
            delta_t = u_t
            residual_norm = 0.0
            evicted_idx = -1

        elif self.step_num <= self.rank:
            if self.Q is None:
                self.Q = torch.zeros((self.p_dim, self.rank), device=device)
            
            q_in = u_t / (u_norm + 1e-12)
            self.Q[:, self.step_num - 1] = q_in

            for k in range(self.step_num):
                for j in range(k):
                    proj = torch.dot(self.Q[:, k], self.Q[:, j])
                    self.Q[:, k] -= proj * self.Q[:, j]
                self.Q[:, k] /= (torch.norm(self.Q[:, k]) + 1e-12)

            delta_t = u_t
            residual_norm = 0.0
            evicted_idx = -1
        
        else:
            # Measure innovation against current subspace Q
            r_t = self.Q.T @ u_t
            u_parallel = self.Q @ r_t
            u_perp = u_t - u_parallel
            residual_norm = torch.norm(u_perp).item()

            q_new = u_perp / (residual_norm + 1e-12)

            # Alignment score with momentum m_t
            m_norm = m_t / (torch.norm(m_t) + 1e-12)
            energy_scores = torch.abs(self.Q.T @ m_norm)
            evicted_idx = torch.argmin(energy_scores).item()

            # Swap evicted column to last slot K-1
            if evicted_idx != self.rank - 1:
                self.Q[:, [evicted_idx, self.rank - 1]] = self.Q[:, [self.rank - 1, evicted_idx]]

            self.Q[:, self.rank - 1] = q_new

            # Single-vector GS Pass at slot K-1
            q_last = self.Q[:, self.rank - 1]
            for j in range(self.rank - 1):
                q_j = self.Q[:, j]
                q_last -= torch.dot(q_last, q_j) * q_j
            self.Q[:, self.rank - 1] = q_last / (torch.norm(q_last) + 1e-12)

            # Project parameter update back through rotated Q
            r_updated = self.Q.T @ u_t
            delta_t = self.Q @ r_updated
            if self.rescale:
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

# ==============================================================================
# 2. CONVOLUTIONAL BENCHMARK MODEL (CIFAR-10)
# ==============================================================================
class CIFARConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# Data Preparation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

# Instantiation
model = CIFARConvNet().to(device)
optimizer = RotationalCSAdam(model.parameters(), rank=32, lr=1e-3)

print("=" * 125)
print(f"BENCHMARKING ROTATIONAL CS-ADAM ON CIFAR-10 (Rank K=32, Params P={optimizer.p_dim:,})")
print("=" * 125)
print(f"{'STEP':>5} | {'LOSS':>8} | {'||u_t||_2':>10} | {'||u_perp||_2':>11} | {'INNOVATION %':>13} | {'EVICTED BASIS COL':>18}")
print("-" * 125)

global_step = 0
for epoch in range(1, 3):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        model.zero_grad()
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        
        u_norm, residual_norm, evicted_idx = optimizer.step()
        global_step += 1

        if global_step == 1 or global_step % 20 == 0:
            innovation_pct = (residual_norm / (u_norm + 1e-12)) * 100.0
            evicted_str = f"Col #{evicted_idx}" if evicted_idx != -1 else "Warmup (None)"
            print(f"{global_step:5d} | {loss.item():8.4f} | {u_norm:10.4f} | {residual_norm:11.4f} | {innovation_pct:12.2f}% | {evicted_str:>18}")

        if global_step >= 200:
            break
    if global_step >= 200:
        break

print("=" * 125)
