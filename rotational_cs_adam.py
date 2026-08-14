import copy
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
class RotationalCSAdam:
    def __init__(self, params, rank=32, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, norm_threshold=1e-4):
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
            r_t = self.Q.T @ u_t
            u_parallel = self.Q @ r_t
            u_perp = u_t - u_parallel
            residual_norm = torch.norm(u_perp).item()

            q_new = u_perp / (residual_norm + 1e-12)

            m_norm = m_t / (torch.norm(m_t) + 1e-12)
            energy_scores = torch.abs(self.Q.T @ m_norm)
            evicted_idx = torch.argmin(energy_scores).item()

            if evicted_idx != self.rank - 1:
                self.Q[:, [evicted_idx, self.rank - 1]] = self.Q[:, [self.rank - 1, evicted_idx]]

            self.Q[:, self.rank - 1] = q_new

            q_last = self.Q[:, self.rank - 1]
            for j in range(self.rank - 1):
                q_j = self.Q[:, j]
                q_last -= torch.dot(q_last, q_j) * q_j
            self.Q[:, self.rank - 1] = q_last / (torch.norm(q_last) + 1e-12)

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

        return u_norm, residual_norm, evicted_idx

# ==============================================================================
# 2. BENCHMARK CONVOLUTIONAL MODEL
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

# Setup Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=False)

# Initialize Identical Dual Models
model_cs = CIFARConvNet().to(device)
model_adam = copy.deepcopy(model_cs)

opt_cs = RotationalCSAdam(model_cs.parameters(), rank=32, lr=1e-3)
opt_adam = torch.optim.Adam(model_adam.parameters(), lr=1e-3)

print("=" * 145)
print(f"COMPARATIVE BENCHMARK: ROTATIONAL CS-ADAM (Rank K=32) vs. STANDARD ADAM (CIFAR-10, P={opt_cs.p_dim:,})")
print("=" * 145)
print(f"{'STEP':>5} | {'CS-ADAM LOSS':>13} | {'STD ADAM LOSS':>13} | {'||u_t||_2':>10} | {'||u_perp||_2':>11} | {'INNOVATION %':>13} | {'EVICTED BASIS COL':>18}")
print("-" * 145)

global_step = 0
for images, labels in train_loader:
    images, labels = images.to(device), labels.to(device)
    
    # --- 1. Step Rotational CS-Adam ---
    model_cs.zero_grad()
    out_cs = model_cs(images)
    loss_cs = F.cross_entropy(out_cs, labels)
    loss_cs.backward()
    u_norm, residual_norm, evicted_idx = opt_cs.step()

    # --- 2. Step Standard Adam ---
    model_adam.zero_grad()
    out_adam = model_adam(images)
    loss_adam = F.cross_entropy(out_adam, labels)
    loss_adam.backward()
    opt_adam.step()

    global_step += 1

    if global_step == 1 or global_step % 20 == 0:
        innovation_pct = (residual_norm / (u_norm + 1e-12)) * 100.0
        evicted_str = f"Col #{evicted_idx}" if evicted_idx != -1 else "Warmup (None)"
        print(f"{global_step:5d} | {loss_cs.item():13.4f} | {loss_adam.item():13.4f} | {u_norm:10.4f} | {residual_norm:11.4f} | {innovation_pct:12.2f}% | {evicted_str:>18}")

    if global_step >= 200:
        break

print("=" * 145)
