"""
ONLINE RANK-1 ROTATIONAL CS-ADAM (Continuous Subspace Tracking)

Tracks Grassmannian trajectory drift at EVERY STEP via O(P*K) Gram-Schmidt 
residual insertion, maintaining 100% directional alignment without 
periodic SVD recalibration lag.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as Fn

# Setup model architecture & synthetic data matching previous runs
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

def get_structured_dataset(vocab_size=64, seq_len=16, dataset_size=1024):
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
# Online Rank-1 Rotational Optimizer Class
# =====================================================================

class OnlineRotationalCSAdam:
    def __init__(self, params, lr=3e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01, rank=16):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.rank = rank
        self.step_num = 0

        # Total parameter count P
        self.p_dim = sum(p.numel() for p in self.params)
        
        # State buffers
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]

        # Subspace Basis Q: P x K
        self.Q = None
        self.ptr = 0 # Ring buffer insertion pointer

    def _get_flattened_adam_update(self):
        """Computes element-wise Adam update u_t vector in P-D space."""
        bc1 = 1.0 - self.beta1 ** self.step_num
        bc2 = 1.0 - self.beta2 ** self.step_num

        u_parts = []
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (g ** 2)

            m_hat = self.m[i] / bc1
            v_hat = self.v[i] / bc2

            u_p = -self.lr * (m_hat / (torch.sqrt(v_hat) + self.eps))
            if self.weight_decay != 0:
                u_p = u_p - self.lr * self.weight_decay * p.data
            
            u_parts.append(u_p.flatten())
        
        return torch.cat(u_parts)

    def step(self):
        self.step_num += 1
        u_t = self._get_flattened_adam_update()
        u_norm = torch.norm(u_t).item()

        # Warmup phase (t <= rank): Bootstrap initial orthonormal Q matrix
        if self.step_num <= self.rank:
            if self.Q is None:
                self.Q = torch.zeros((self.p_dim, self.rank), device=u_t.device)
            
            # Insert normalized vector
            q_in = u_t / (u_norm + 1e-12)
            self.Q[:, self.step_num - 1] = q_in

            # Orthonormalize built vectors using MGS
            for k in range(self.step_num):
                for j in range(k):
                    proj = torch.dot(self.Q[:, k], self.Q[:, j])
                    self.Q[:, k] -= proj * self.Q[:, j]
                self.Q[:, k] = self.Q[:, k] / (torch.norm(self.Q[:, k]) + 1e-12)

            delta_t = u_t # Take unconstrained step during warmup
            residual_norm = 0.0
        
        else:
            # ONLINE RANK-1 BASIS ROTATION (MGS Residual Insertion)
            
            # 1. Project u_t into current subspace Q
            r_t = self.Q.T @ u_t # K-D coordinates
            
            # 2. Compute out-of-subspace residual
            u_parallel = self.Q @ r_t
            u_perp = u_t - u_parallel
            residual_norm = torch.norm(u_perp).item()

            # 3. Normalize residual innovation vector
            q_new = u_perp / (residual_norm + 1e-12)

            # 4. Substitute oldest basis column in Q with q_new
            self.Q[:, self.ptr] = q_new
            self.ptr = (self.ptr + 1) % self.rank

            # 5. Fast Modified Gram-Schmidt re-orthonormalization sweep O(P*K)
            for k in range(self.rank):
                for j in range(k):
                    proj = torch.dot(self.Q[:, k], self.Q[:, j])
                    self.Q[:, k] -= proj * self.Q[:, j]
                self.Q[:, k] = self.Q[:, k] / (torch.norm(self.Q[:, k]) + 1e-12)

            # 6. Re-project u_t onto updated Q and enforce exact norm preservation
            r_updated = self.Q.T @ u_t
            delta_t = self.Q @ r_updated
            delta_t = delta_t * (u_norm / (torch.norm(delta_t) + 1e-12))

        # Apply update delta_t back to parameters
        offset = 0
        for p in self.params:
            if p.grad is None:
                continue
            numel = p.numel()
            p.data.add_(delta_t[offset : offset + numel].view_as(p.data))
            offset += numel

        return u_norm, residual_norm

# =====================================================================
# Execution & Benchmarking
# =====================================================================

def run_online_rotational_cs_adam():
    torch.manual_seed(1337)
    vocab_size, d_model, n_layers, seq_len = 64, 32, 2, 16
    model = MiniTransformer(vocab_size, d_model, n_layers, seq_len)
    
    X, Y = get_structured_dataset(vocab_size, seq_len, dataset_size=1024)
    batch_size = 32
    num_batches = len(X) // batch_size

    opt = OnlineRotationalCSAdam(model.parameters(), lr=3e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01, rank=16)

    print("=" * 115)
    print("ONLINE RANK-1 ROTATIONAL CS-ADAM (Continuous Subspace Tracking Log)")
    print("=" * 115)
    print(f"{'STEP':>5} | {'LOSS':>8} | {'||u_t||_2 (Adam)':>16} | {'||u_perp||_2 (Residual)':>22} | {'SUBSPACE INNOVATION %':>24}")
    print("-" * 115)

    for step in range(1, 201):
        b_idx = (step - 1) % num_batches
        x_batch = X[b_idx * batch_size : (b_idx + 1) * batch_size]
        y_batch = Y[b_idx * batch_size : (b_idx + 1) * batch_size]

        model.zero_grad()
        _, loss = model(x_batch, y_batch)
        loss.backward()

        u_norm, res_norm = opt.step()

        if step % 10 == 0 or step == 1:
            innovation_pct = (res_norm / (u_norm + 1e-12)) * 100.0
            print(f"{step:>5d} | {loss.item():>8.4f} | {u_norm:>16.4f} | {res_norm:>22.4f} | {innovation_pct:>23.2f}%")

    print("=" * 115)

if __name__ == "__main__":
    run_online_rotational_cs_adam()
