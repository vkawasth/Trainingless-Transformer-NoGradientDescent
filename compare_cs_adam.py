import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Ensure exact reproducibility
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from rotational_cs_adam_2 import RotationalCSAdam, CIFARConvNet

# Standard data loader (shuffle=False for identical mini-batches)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=False)

# Three identical model initializations
model_rescaled = CIFARConvNet().to(device)
model_honest = copy.deepcopy(model_rescaled)
model_adam = copy.deepcopy(model_rescaled)

# Optimizers
opt_rescaled = RotationalCSAdam(model_rescaled.parameters(), rank=32, lr=1e-3)
opt_honest = RotationalCSAdam(model_honest.parameters(), rank=32, lr=1e-3)
opt_adam = torch.optim.Adam(model_adam.parameters(), lr=1e-3)

# Patch the honest optimizer step to skip norm scaling in-place
def step_honest(self):
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

        # HONEST STEP: Pure projection without norm rescaling
        r_updated = self.Q.T @ u_t
        delta_t = self.Q @ r_updated  # <--- Rescale omitted

    offset = 0
    for p in self.params:
        if p.grad is None:
            continue
        numel = p.numel()
        p.data.add_(delta_t[offset : offset + numel].view_as(p.data))
        offset += numel

    return u_norm, residual_norm, evicted_idx

# Bind patched method
opt_honest.step = step_honest.__get__(opt_honest, RotationalCSAdam)

print("=" * 110)
print("THREE-WAY BENCHMARK: ADAM vs. CS-ADAM (RESCALED) vs. CS-ADAM (HONEST)")
print("=" * 110)
print(f"{'STEP':>5} | {'ADAM LOSS':>11} | {'RESCALED CS':>13} | {'HONEST CS':>11} | {'INNOVATION %':>13}")
print("-" * 110)

global_step = 0
for epoch in range(1, 3):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # 1. Adam
        model_adam.zero_grad()
        loss_adam = F.cross_entropy(model_adam(images), labels)
        loss_adam.backward()
        opt_adam.step()

        # 2. Rescaled CS-Adam
        model_rescaled.zero_grad()
        loss_rescaled = F.cross_entropy(model_rescaled(images), labels)
        loss_rescaled.backward()
        u_norm, residual_norm, _ = opt_rescaled.step()

        # 3. Honest CS-Adam
        model_honest.zero_grad()
        loss_honest = F.cross_entropy(model_honest(images), labels)
        loss_honest.backward()
        opt_honest.step()

        global_step += 1

        if global_step == 1 or global_step % 20 == 0:
            inn_pct = (residual_norm / (u_norm + 1e-12)) * 100.0
            print(
                f"{global_step:5d} | "
                f"{loss_adam.item():11.4f} | "
                f"{loss_rescaled.item():13.4f} | "
                f"{loss_honest.item():11.4f} | "
                f"{inn_pct:12.2f}%"
            )

        if global_step >= 200:
            break
    if global_step >= 200:
        break
print("=" * 110)
