"""
SUBSPACE ROTATION & TURNING DETECTOR

Tracks how standard Adam's subspace Q_t rotates on the Grassmannian manifold Gr(K, P)
using QR decomposition (no SVD required).
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
# Subspace Rotation Tracking
# =====================================================================

def track_subspace_rotation():
    torch.manual_seed(1337)
    vocab_size, d_model, n_layers, seq_len = 64, 32, 2, 16
    model = MiniTransformer(vocab_size, d_model, n_layers, seq_len)
    
    X, Y = get_structured_dataset(vocab_size, seq_len, dataset_size=1024)
    batch_size = 32
    num_batches = len(X) // batch_size

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01)

    window_size = 16
    history_buffer = []

    prev_Q = None
    Q_at_last_recal = None

    print("=" * 120)
    print("SUBSPACE ROTATION DETECTOR (Grassmannian Drift Tracking)")
    print("=" * 120)
    print(f"{'STEP':>5} | {'LOSS':>8} | {'1-STEP Q OVERLAP':>18} | {'10-STEP Q OVERLAP':>18} | {'MAX TURN ANG (θ_max)':>22} | {'GRASSMANNIAN DIST':>20}")
    print("-" * 120)

    for step in range(1, 201):
        b_idx = (step - 1) % num_batches
        x_batch = X[b_idx * batch_size : (b_idx + 1) * batch_size]
        y_batch = Y[b_idx * batch_size : (b_idx + 1) * batch_size]

        optimizer.zero_grad()
        _, loss = model(x_batch, y_batch)
        loss.backward()

        optimizer.step()

        # Reconstruct full u_t vector from Adam state
        u_t_parts = []
        with torch.no_grad():
            for param in model.parameters():
                state = optimizer.state[param]
                if "exp_avg" in state and "exp_avg_sq" in state:
                    m = state["exp_avg"] / (1.0 - 0.9 ** step)
                    v = state["exp_avg_sq"] / (1.0 - 0.95 ** step)
                    u_p = -0.003 * (m / (torch.sqrt(v) + 1e-8)) - 0.003 * 0.01 * param.data
                    u_t_parts.append(u_p.flatten())

        u_t = torch.cat(u_t_parts)

        # Maintain sliding history buffer H_t
        history_buffer.append(u_t)
        if len(history_buffer) > window_size:
            history_buffer.pop(0)

        # Build orthonormal basis Q_t using QR decomposition on H_t (Fast, no SVD needed)
        if len(history_buffer) == window_size:
            H = torch.stack(history_buffer, dim=1) # P x K
            Q_t, _ = torch.linalg.qr(H, mode='reduced') # P x K

            # Measure rotation relative to previous step Q_{t-1}
            overlap_1step = 0.0
            overlap_10step = 0.0
            max_turn_deg = 0.0
            g_dist = 0.0

            if prev_Q is not None:
                # Frame overlap matrix M = Q_{t-1}^T @ Q_t
                M = prev_Q.T @ Q_t
                # Singular values of M yield cos(principal angles)
                S = torch.linalg.svdvals(M)
                S = torch.clamp(S, -1.0, 1.0)
                overlap_1step = S.mean().item()

            if Q_at_last_recal is not None:
                M_10 = Q_at_last_recal.T @ Q_t
                S_10 = torch.linalg.svdvals(M_10)
                S_10 = torch.clamp(S_10, -1.0, 1.0)
                overlap_10step = S_10.mean().item()

                principal_angles = torch.arccos(S_10)
                max_turn_deg = math.degrees(principal_angles.max().item())
                g_dist = torch.norm(principal_angles).item()

            if step % 10 == 0:
                Q_at_last_recal = Q_t.clone()

            prev_Q = Q_t.clone()

            if step % 10 == 0:
                print(f"{step:>5d} | {loss.item():>8.4f} | {overlap_1step:>18.4f} | {overlap_10step:>18.4f} | {max_turn_deg:>20.2f}° | {g_dist:>20.4f}")

    print("=" * 120)

if __name__ == "__main__":
    track_subspace_rotation()
