"""
CALIBRATED SUBSPACE ADAM (CS-Adam) - WORKING PRODUCTION ENGINE

Fixes:
1. Replaces pure i.i.d. noise with a learnable synthetic sequence task.
2. Incorporates residual exploration energy to prevent manifold lock-in.
3. Preserves exact Adam update scale in the active subspace.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as Fn

# =====================================================================
# 1. CS-Adam Optimizer Class
# =====================================================================

class CSAdam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01,
                 subspace_rank=16, window_size=30, warmup_steps=20, recal_interval=10,
                 residual_weight=0.1):
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

        self.K = subspace_rank
        self.window_size = window_size
        self.warmup_steps = warmup_steps
        self.recal_interval = recal_interval
        self.gamma = residual_weight  # Prevents subspace lock-in

        self.p_dim = sum(p.numel() for p in self.params)
        self.m = torch.zeros(self.p_dim, dtype=torch.float32)
        self.v = torch.zeros(self.p_dim, dtype=torch.float32)

        self.step_count = 0
        self.history = []
        self.Q = None

        self.last_cos_sim = 1.0
        self.last_topk_energy = 100.0
        self.last_spectrum = []

    def _get_flat_params_and_grads(self):
        p_list, g_list = [], []
        for p in self.params:
            p_list.append(p.data.flatten())
            if p.grad is not None:
                g_list.append(p.grad.flatten())
            else:
                g_list.append(torch.zeros_like(p.data.flatten()))
        return torch.cat(p_list), torch.cat(g_list)

    def _set_flat_params(self, flat_p):
        idx = 0
        with torch.no_grad():
            for p in self.params:
                num = p.numel()
                p.data.copy_(flat_p[idx : idx + num].view_as(p))
                idx += num

    def step(self):
        self.step_count += 1
        t = self.step_count

        theta, g = self._get_flat_params_and_grads()

        if self.weight_decay != 0:
            g = g + self.weight_decay * theta

        # 1. Update full-space Adam moments
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * g
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (g * g)

        m_hat = self.m / (1.0 - self.beta1 ** t)
        v_hat = self.v / (1.0 - self.beta2 ** t)

        # 2. Preconditioned full step vector
        u_t = -self.lr * (m_hat / (torch.sqrt(v_hat) + self.eps))

        # 3. Maintain raw step history buffer (uncontaminated by projections)
        self.history.append(u_t.clone())
        if len(self.history) > self.window_size:
            self.history.pop(0)

        # 4. SVD Subspace Recalibration
        should_recalibrate = (t == self.warmup_steps) or (
            t > self.warmup_steps and (t - self.warmup_steps) % self.recal_interval == 0
        )

        if should_recalibrate and len(self.history) >= self.K:
            H = torch.stack(self.history, dim=1)  # (P x W)
            U, S, _ = torch.linalg.svd(H, full_matrices=False)
            self.Q = U[:, :self.K]  # Top-K basis

            energy_sq = S ** 2
            total_e = energy_sq.sum().item()
            if total_e > 0:
                self.last_topk_energy = (energy_sq[:self.K].sum().item() / total_e) * 100.0
                self.last_spectrum = [(e.item() / total_e) * 100.0 for e in energy_sq[:5]]

        # 5. Combined Subspace-Residual Step
        if t <= self.warmup_steps or self.Q is None:
            delta = u_t
            self.last_cos_sim = 1.0
        else:
            proj_u = self.Q @ (self.Q.T @ u_t)
            res_u = u_t - proj_u
            
            # Combine rank-K subspace update with exploration energy
            delta = proj_u + self.gamma * res_u

            norm_u = torch.norm(u_t).item()
            norm_d = torch.norm(delta).item()
            if norm_u > 0 and norm_d > 0:
                self.last_cos_sim = (torch.dot(u_t, delta) / (norm_u * norm_d)).item()

        self._set_flat_params(theta + delta)
        return delta


# =====================================================================
# 2. Transformer & Learnable Data Setup
# =====================================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.q(x), self.k(x), self.v(x)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(C))
        mask = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)
        att = Fn.softmax(att + mask, dim=-1)
        return self.o(att @ v)


class MLP(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.g = nn.Linear(d_model, 4 * d_model, bias=False)
        self.u = nn.Linear(d_model, 4 * d_model, bias=False)
        self.w = nn.Linear(4 * d_model, d_model, bias=False)

    def forward(self, x):
        return self.w(Fn.silu(self.g(x)) * self.u(x))


class TransformerBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MiniTransformer(nn.Module):
    def __init__(self, vocab_size=64, d_model=32, n_layers=2, max_seq_len=16):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = Fn.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


def get_structured_dataset(vocab_size=64, seq_len=16, dataset_size=512):
    """Generates a deterministic sequence task with local Markov dependencies."""
    torch.manual_seed(42)
    transition = torch.randint(0, vocab_size, (vocab_size,))
    
    X_list, Y_list = [], []
    for _ in range(dataset_size):
        seq = [torch.randint(0, vocab_size, (1,)).item()]
        for _ in range(seq_len):
            next_tok = (transition[seq[-1]].item() + torch.randint(0, 3, (1,)).item()) % vocab_size
            seq.append(next_tok)
        X_list.append(seq[:-1])
        Y_list.append(seq[1:])
        
    return torch.tensor(X_list, dtype=torch.long), torch.tensor(Y_list, dtype=torch.long)


# =====================================================================
# 3. Execution Loop
# =====================================================================

def run_benchmark():
    torch.manual_seed(1337)
    vocab_size, d_model, n_layers, seq_len = 64, 32, 2, 16
    model = MiniTransformer(vocab_size, d_model, n_layers, seq_len)
    
    # Load structured learnable task
    X, Y = get_structured_dataset(vocab_size, seq_len, dataset_size=1024)
    batch_size = 32
    num_batches = len(X) // batch_size

    opt = CSAdam(
        model.parameters(),
        lr=3e-3,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.01,
        subspace_rank=16,
        window_size=30,
        warmup_steps=20,
        recal_interval=10,
        residual_weight=0.1
    )

    print("=" * 110)
    print(f"{'STEP':>5} | {'LOSS':>8} | {'TYPE':>6} | {'ALIGNMENT (cos θ)':>18} | {'TOP-16 ENERGY':>13} | {'TOP 5 SPECTRUM'}")
    print("-" * 110)

    for step in range(1, 201):
        b_idx = (step - 1) % num_batches
        x_batch = X[b_idx * batch_size : (b_idx + 1) * batch_size]
        y_batch = Y[b_idx * batch_size : (b_idx + 1) * batch_size]

        model.zero_grad()
        _, loss = model(x_batch, y_batch)
        loss.backward()

        opt.step()

        if step % 10 == 0 or step == 1:
            step_type = "FULL" if step <= opt.warmup_steps else "SUB"
            cos_str = f"{opt.last_cos_sim:.4f}" if step_type == "SUB" else "1.0000 (Warmup)"
            energy_str = f"{opt.last_topk_energy:.2f}%" if opt.last_spectrum else "N/A"
            spec_str = "[" + ", ".join(f"{p:.1f}%" for p in opt.last_spectrum) + "]" if opt.last_spectrum else "N/A"

            print(f"{step:>5d} | {loss.item():>8.4f} | {step_type:>6} | {cos_str:>18} | {energy_str:>13} | {spec_str}")

    print("=" * 110)

if __name__ == "__main__":
    run_benchmark()
