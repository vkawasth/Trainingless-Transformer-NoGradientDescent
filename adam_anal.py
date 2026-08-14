"""
FULL-SPACE ADAM DECISION MATRIX TRACKER

Instruments standard Adam on the exact same seed/model/data 
to extract layer-wise norms, directional momentum shifts, 
and effective coordinate updates.
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
# Decision Matrix Tracking Execution
# =====================================================================

def track_adam_decisions():
    torch.manual_seed(1337)
    vocab_size, d_model, n_layers, seq_len = 64, 32, 2, 16
    model = MiniTransformer(vocab_size, d_model, n_layers, seq_len)
    
    X, Y = get_structured_dataset(vocab_size, seq_len, dataset_size=1024)
    batch_size = 32
    num_batches = len(X) // batch_size

    # Standard PyTorch AdamW
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01)

    # Track decision variables across step t and t-1
    prev_u_t = None

    print("=" * 135)
    print("STANDARD ADAM DECISION MATRIX LOG")
    print("=" * 135)
    print(f"{'STEP':>5} | {'LOSS':>8} | {'||u_t||_2':>10} | {'STEP ANG (cos θ_t,t-1)':>22} | {'EMB UPDATE':>12} | {'ATTN UPDATE':>12} | {'MLP UPDATE':>12} | {'LN UPDATE':>12}")
    print("-" * 135)

    for step in range(1, 201):
        b_idx = (step - 1) % num_batches
        x_batch = X[b_idx * batch_size : (b_idx + 1) * batch_size]
        y_batch = Y[b_idx * batch_size : (b_idx + 1) * batch_size]

        optimizer.zero_grad()
        _, loss = model(x_batch, y_batch)
        loss.backward()

        # Capture state before step
        param_grads = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_grads[name] = param.grad.detach().clone()

        optimizer.step()

        # Reconstruction of u_t vector & layer-wise breakdown
        u_t_parts = []
        layer_norms = {"emb": [], "attn": [], "mlp": [], "ln": []}

        with torch.no_grad():
            for name, param in model.named_parameters():
                state = optimizer.state[param]
                if "exp_avg" in state and "exp_avg_sq" in state:
                    m = state["exp_avg"]
                    v = state["exp_avg_sq"]
                    bias_correction1 = 1.0 - 0.9 ** step
                    bias_correction2 = 1.0 - 0.95 ** step
                    
                    m_hat = m / bias_correction1
                    v_hat = v / bias_correction2
                    
                    # Exact Adam preconditioned step component
                    u_p = -0.003 * (m_hat / (torch.sqrt(v_hat) + 1e-8))
                    
                    if 0.01 != 0:
                        u_p = u_p - 0.003 * 0.01 * param.data

                    u_t_parts.append(u_p.flatten())

                    u_norm = torch.norm(u_p).item()
                    if "emb" in name:
                        layer_norms["emb"].append(u_norm)
                    elif "attn" in name:
                        layer_norms["attn"].append(u_norm)
                    elif "mlp" in name:
                        layer_norms["mlp"].append(u_norm)
                    elif "ln" in name:
                        layer_norms["ln"].append(u_norm)

        u_t = torch.cat(u_t_parts)
        total_norm = torch.norm(u_t).item()

        # Compute turn angle relative to PREVIOUS step
        if prev_u_t is not None:
            cos_turn = (torch.dot(u_t, prev_u_t) / (total_norm * torch.norm(prev_u_t))).item()
        else:
            cos_turn = 1.0

        prev_u_t = u_t.clone()

        if step % 10 == 0 or step == 1:
            emb_sum = math.sqrt(sum(x**2 for x in layer_norms["emb"]))
            attn_sum = math.sqrt(sum(x**2 for x in layer_norms["attn"]))
            mlp_sum = math.sqrt(sum(x**2 for x in layer_norms["mlp"]))
            ln_sum = math.sqrt(sum(x**2 for x in layer_norms["ln"]))

            print(f"{step:>5d} | {loss.item():>8.4f} | {total_norm:>10.4f} | {cos_turn:>22.4f} | {emb_sum:>12.4f} | {attn_sum:>12.4f} | {mlp_sum:>12.4f} | {ln_sum:>12.4f}")

    print("=" * 135)

if __name__ == "__main__":
    track_adam_decisions()
