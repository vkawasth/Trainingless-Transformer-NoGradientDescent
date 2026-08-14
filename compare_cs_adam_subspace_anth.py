"""
HEAD-TO-HEAD: ROTATIONAL CS-ADAM vs STANDARD ADAM ON CIFAR-10
=============================================================

Self-contained -- no imports from other project files.

Three arms, identical initial weights, identical batch sequence:

  adam         standard Adam, the baseline
  cs-rescaled  the projected step renormalised back to ||u_t||. This is the
               version that produced the head-to-head table where CS-Adam and
               Adam tracked to 3-4 decimal places.
  cs-honest    the SAME projection WITHOUT the renormalisation, so the step keeps
               its own (smaller) norm. This is a genuine subspace restriction.

Why the third arm decides something. With the rescale, CS-Adam takes an
Adam-MAGNITUDE step in a direction that (per the innovation column) is roughly
50-70% aligned with Adam's. On a smooth landscape a well-scaled 60%-aligned step
descends about as well as the original, so the identity in the two-arm table may
be a magnitude effect rather than evidence that the subspace carries the descent.

  cs-honest tracks Adam  ->  the rank-32 frame genuinely carries the descent
  cs-honest degrades     ->  the rescale was doing the work

For reference, on a small transformer (P=3672) pure projection without rescale
cost 73% against tuned Adam, and keeping a quarter of the orthogonal complement
cost 6%. Whether that holds at P=545K with much better frame capture is the open
question this run answers.

Requires: torch, torchvision. Downloads CIFAR-10 on first run.
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================================================================
# OPTIMIZER
# ==============================================================================
class RotationalCSAdam:
    """Adam update projected onto a rank-K subspace that rotates by eviction.

    rescale=True  : the projected step is scaled back to ||u_t||
    rescale=False : the projected step keeps its own norm
    alpha         : fraction of the orthogonal complement to retain (0 = none).
                    Only used when rescale=False.
    """

    def __init__(self, params, rank=32, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 norm_threshold=1e-4, rescale=True, alpha=0.0):
        self.params = [p for p in params if p.requires_grad]
        self.rank = rank
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.norm_threshold = norm_threshold
        self.rescale = rescale
        self.alpha = alpha

        self.p_dim = sum(p.numel() for p in self.params)
        self.m = torch.zeros(self.p_dim, device=device)
        self.v = torch.zeros(self.p_dim, device=device)
        self.step_num = 0
        self.Q = None

    def _adam_update(self):
        grads = torch.cat([
            (p.grad.reshape(-1) if p.grad is not None
             else torch.zeros(p.numel(), device=device))
            for p in self.params])
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads ** 2)
        m_hat = self.m / (1 - self.beta1 ** self.step_num)
        v_hat = self.v / (1 - self.beta2 ** self.step_num)
        u_t = -self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
        return u_t, self.m

    def step(self):
        self.step_num += 1
        u_t, m_t = self._adam_update()
        u_norm = torch.norm(u_t).item()

        if u_norm < self.norm_threshold or self.step_num <= self.rank:
            # warmup: fill the basis one column per step, full Adam step applied
            if self.Q is None:
                self.Q = torch.zeros((self.p_dim, self.rank), device=device)
            if self.step_num <= self.rank and u_norm >= self.norm_threshold:
                q_in = u_t / (u_norm + 1e-12)
                for j in range(self.step_num - 1):
                    q_in = q_in - torch.dot(q_in, self.Q[:, j]) * self.Q[:, j]
                self.Q[:, self.step_num - 1] = q_in / (torch.norm(q_in) + 1e-12)
            delta_t = u_t
            residual_norm = 0.0
            evicted_idx = -1
        else:
            r_t = self.Q.T @ u_t
            u_parallel = self.Q @ r_t
            u_perp = u_t - u_parallel
            residual_norm = torch.norm(u_perp).item()
            q_new = u_perp / (residual_norm + 1e-12)

            # evict the basis column least aligned with momentum
            m_norm = m_t / (torch.norm(m_t) + 1e-12)
            energy_scores = torch.abs(self.Q.T @ m_norm)
            evicted_idx = int(torch.argmin(energy_scores))

            if evicted_idx != self.rank - 1:
                self.Q[:, [evicted_idx, self.rank - 1]] = \
                    self.Q[:, [self.rank - 1, evicted_idx]]

            # insert at the last slot and re-orthogonalise only that column
            self.Q[:, self.rank - 1] = q_new
            q_last = self.Q[:, self.rank - 1].clone()
            for j in range(self.rank - 1):
                q_last = q_last - torch.dot(q_last, self.Q[:, j]) * self.Q[:, j]
            self.Q[:, self.rank - 1] = q_last / (torch.norm(q_last) + 1e-12)

            delta_t = self.Q @ (self.Q.T @ u_t)

            if self.rescale:
                delta_t = delta_t * (u_norm / (torch.norm(delta_t) + 1e-12))
            elif self.alpha > 0.0:
                delta_t = delta_t + self.alpha * (u_t - self.Q @ (self.Q.T @ u_t))

        offset = 0
        for p in self.params:
            numel = p.numel()
            p.data.add_(delta_t[offset:offset + numel].view_as(p.data))
            offset += numel

        return u_norm, residual_norm, evicted_idx


# ==============================================================================
# MODEL
# ==============================================================================
class CIFARConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128), nn.ReLU(), nn.Linear(128, 10))

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x.reshape(x.size(0), -1))


# ==============================================================================
# DATA
# ==============================================================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

train_dataset = datasets.CIFAR10(root="./data", train=True, download=True,
                                 transform=transform)
test_dataset = datasets.CIFAR10(root="./data", train=False, download=True,
                                transform=transform)
# shuffle=True with a fixed generator: every arm sees the same sequence, but the
# sequence is not the class-ordered file order
gen = torch.Generator().manual_seed(1234)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, generator=gen)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)


@torch.no_grad()
def evaluate(model):
    model.eval()
    tot, n = 0.0, 0
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        tot += float(F.cross_entropy(model(xb), yb, reduction="sum"))
        n += yb.numel()
    model.train()
    return tot / n


# ==============================================================================
# BENCHMARK
# ==============================================================================
STEPS = 400
STEPS = 3000
EVAL_EVERY = 100

base = CIFARConvNet().to(device)
P = sum(p.numel() for p in base.parameters())

model_adam = copy.deepcopy(base)
model_resc = copy.deepcopy(base)
model_hon = copy.deepcopy(base)

opt_adam = torch.optim.Adam(model_adam.parameters(), lr=1e-3)
opt_resc = RotationalCSAdam(model_resc.parameters(), rank=32, lr=1e-3, rescale=True)
opt_hon = RotationalCSAdam(model_hon.parameters(), rank=32, lr=1e-3, rescale=False)

print("=" * 108)
print(f"CIFAR-10  |  P = {P:,}  |  rank 32 = {32 / P:.1e} of P  |  {STEPS} steps  |  {device}")
print("=" * 108)
print(f"{'STEP':>5} | {'ADAM':>9} | {'CS-RESCALED':>12} | {'CS-HONEST':>10} | "
      f"{'INNOV %':>8} | {'||d_hon||/||u||':>15}")
print("-" * 108)

step = 0
done = False
while not done:
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        model_adam.zero_grad()
        la = F.cross_entropy(model_adam(images), labels)
        la.backward()
        opt_adam.step()

        model_resc.zero_grad()
        lr_ = F.cross_entropy(model_resc(images), labels)
        lr_.backward()
        opt_resc.step()

        model_hon.zero_grad()
        lh = F.cross_entropy(model_hon(images), labels)
        lh.backward()
        un, rn, _ = opt_hon.step()

        step += 1
        if step == 1 or step % 25 == 0:
            innov = 100.0 * rn / (un + 1e-12)
            keep = (max(0.0, 1.0 - (rn / (un + 1e-12)) ** 2)) ** 0.5 if un > 0 else 0.0
            print(f"{step:5d} | {la.item():9.4f} | {lr_.item():12.4f} | "
                  f"{lh.item():10.4f} | {innov:7.2f}% | {keep:15.3f}")
        if step % EVAL_EVERY == 0:
            print(f"      | {'TEST':>9} | {evaluate(model_adam):9.4f} "
                  f"| {evaluate(model_resc):12.4f} | {evaluate(model_hon):10.4f} |")
        if step >= STEPS:
            done = True
            break

print("=" * 108)
print(f"FINAL TEST LOSS   adam {evaluate(model_adam):.4f}   "
      f"cs-rescaled {evaluate(model_resc):.4f}   cs-honest {evaluate(model_hon):.4f}")
print()
print("cs-honest tracks adam  -> the rank-32 frame genuinely carries the descent")
print("cs-honest degrades     -> the rescale was doing the work")
