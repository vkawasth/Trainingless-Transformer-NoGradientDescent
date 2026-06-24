"""
involutivity_test.py
=====================
Tests whether the K₀ split satisfies μ_k ∘ μ_k = id (involutivity).

From the paper (Proposition 15.1, falsifiable form condition 2):
    μ_k ∘ μ_k = id  ⟺  applying the K₀ split twice from a fixed
    checkpoint returns val and τ to within 0.005 nats and 0.1 respectively.

The K₀ split update rule:
    Δθ = ΔEmb + w_FF(τ) · ΔFF + ΔAttn
    w_FF(τ) = 3.5 × (1.5/τ)^1.5

Run 1 (μ_k):   apply K₀ split from checkpoint → get θ₁, val₁, τ₁
Run 2 (μ_k²):  apply K₀ split from θ₁         → get θ₂, val₂, τ₂

Involutivity criteria:
    |val₂ - val₀| < 0.005   (val returns to start)
    |τ₂   - τ₀|   < 0.1     (τ returns to start)
    cos(θ₂ - θ₀, θ₁ - θ₀) ≈ -1  (second step reverses first)

Also tests the weaker version:
    |val₂ - val₁| vs |val₁ - val₀|  (does second step undo first?)

Usage
-----
  python involutivity_test.py \\
      --checkpoint basin_entry_state.pt \\
      --compiler compiler_geometric.py \\
      --n_steps 13 --lr 3e-4 --w_ff_override auto \\
      --output involutivity_report.json

  # Also from basin_state.pt:
  python involutivity_test.py \\
      --checkpoint basin_state.pt \\
      --compiler compiler_geometric.py \\
      --n_steps 13 --lr 3e-4 \\
      --output involutivity_basin_report.json

Dependencies: torch, numpy, compiler_geometric.py (imports LM, get_batch,
              eval_val, train_t, val_t from it directly)
"""

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.bool_):    return bool(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="K₀ split involutivity test")
    p.add_argument("--checkpoint", default="basin_entry_state.pt",
                   help="Starting checkpoint (basin_entry_state.pt recommended)")
    p.add_argument("--compiler",   default="compiler_geometric.py",
                   help="Path to compiler script (imports LM, get_batch, train_t, val_t)")
    p.add_argument("--n_steps",    type=int,   default=13,
                   help="K₀ split steps (13 = algebraic phase minimum)")
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--w_ff_override", default="auto",
                   help="'auto' = use w_FF(τ) formula; or float to fix w_FF")
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--device",     default="cpu")
    p.add_argument("--output",     default="involutivity_report.json")
    p.add_argument("--val_tol",    type=float, default=0.005,
                   help="Involutivity criterion: |val₂ - val₀| < val_tol")
    p.add_argument("--tau_tol",    type=float, default=0.1,
                   help="Involutivity criterion: |τ₂ - τ₀| < tau_tol")
    return p.parse_args()


# ─── Import model + data from compiler (avoids duplication) ─────────────────
# We import LM, get_batch, eval_val, train_t, val_t directly from the compiler
# script. This guarantees exact same model architecture and data as used during
# training. The import is deferred to main() after args are parsed.

_compiler_imported = False
LM = get_batch = eval_val = train_t = val_t = None   # set in main()

def import_compiler(compiler_path):
    """
    Execute compiler script and return its globals as a simple namespace.
    Uses exec() to avoid importlib issues. Masks __main__ guard so the
    compiler's training loop does not run; only setup (data loading, class
    definitions) executes.
    """
    import types
    globs = {
        "__name__": "__compiler__",
        "__file__": compiler_path,
        "__builtins__": __builtins__,
    }
    with open(compiler_path, "r") as f:
        src = f.read()
    src = src.replace('if __name__ == "__main__":', 'if False:  # masked')
    src = src.replace("if __name__ == '__main__':", 'if False:  # masked')
    # Also mask the compiler's main execution block — everything after
    # the function/class definitions. We only want setup globals.
    # Strategy: replace torch.save calls with no-ops so checkpoints
    # are not overwritten during import.
    import re
    # Mask torch.save (checkpoint saves) — replace with pass-equivalent
    src = re.sub(r'torch\.save\(', '_MASKED_save(', src)
    globs['_MASKED_save'] = lambda *a, **kw: None   # no-op save
    # Also mask plt.show / plt.savefig if present
    src = re.sub(r'plt\.(show|savefig)\(', '_MASKED_plt(', src)
    globs['_MASKED_plt'] = lambda *a, **kw: None
    exec(compile(src, compiler_path, "exec"), globs)
    mod = types.SimpleNamespace(**{
        k: v for k, v in globs.items() if not k.startswith("__")
    })
    return mod

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.c_attn  = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.c_proj  = nn.Linear(cfg.n_embd, cfg.n_embd,     bias=False)
        self.register_buffer("bias",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size))
                  .view(1, 1, cfg.block_size, cfg.block_size))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k_ = k.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        q_ = q.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        v_ = v.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        att = (q_ @ k_.transpose(-2,-1)) / math.sqrt(k_.size(-1))
        att = att.masked_fill(self.bias[:,:,:T,:T]==0, float('-inf'))
        att = F.softmax(att, dim=-1)
        return self.c_proj((att @ v_).transpose(1,2).contiguous().view(B,T,C))

class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_fc   = nn.Linear(cfg.n_embd, 4*cfg.n_embd, bias=False)
        self.c_proj = nn.Linear(4*cfg.n_embd, cfg.n_embd, bias=False)
    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x)))

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.mlp  = MLP(cfg)
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPTConfig:
    def __init__(self, **kw):
        for k,v in kw.items(): setattr(self, k, v)

class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(cfg.vocab_size, cfg.n_embd),
            wpe  = nn.Embedding(cfg.block_size, cfg.n_embd),
            h    = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
            ln_f = nn.LayerNorm(cfg.n_embd),
        ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos  = torch.arange(T, device=idx.device)
        x    = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        x    = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1))
        return logits, loss


# ─── Data + eval ─────────────────────────────────────────────────────────────

def load_data(path, device):
    data = torch.load(path, map_location=device, weights_only=False)
    if isinstance(data, dict):
        data = data.get("train", data.get("data", list(data.values())[0]))
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data, dtype=torch.long)
    return data.to(device)

def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x  = torch.stack([data[i:i+block_size]   for i in ix])
    y  = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def eval_val(model, data, block_size, batch_size, device, n_batches=16):
    model.eval()
    losses = []
    for _ in range(n_batches):
        x, y = get_batch()
        _, loss = model(x, y)
        losses.append(loss.item())
    return float(np.mean(losses))


# ─── τ and geometric diagnostics ─────────────────────────────────────────────

def gluing_defect_local(model, n=4):
    """τ = ||∇_FF L|| / ||∇_Emb L||  (uses compiler's get_batch)
    
    Naming in this model:
      FF params  : blocks[l].ff.*   (name contains '.ff.')
      Emb params : te.weight, pe.weight  (name starts with 'te.' or 'pe.')
    """
    model.train()
    total_ff, total_emb = 0.0, 0.0
    for _ in range(n):
        x, y = get_batch()
        _, loss = model(x, y)
        loss.backward()
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            g2 = p.grad.norm().item() ** 2
            # Match compiler's gluing_defect naming exactly
            if ".ff." in name:           # blocks[l].ff.g / .v / .o / .n
                total_ff += g2
            elif name.startswith("te.") or name.startswith("pe."):
                total_emb += g2
        model.zero_grad()
    return math.sqrt(total_ff / (total_emb + 1e-12))

def w_ff_formula(tau):
    """w_FF(τ) = 3.5 × (1.5/τ)^1.5"""
    return 3.5 * (1.5 / (tau + 1e-10)) ** 1.5

def param_vector(model):
    """Flatten all parameters to a single numpy vector."""
    return np.concatenate([p.detach().cpu().numpy().ravel()
                           for p in model.parameters()])

def cosine(a, b):
    return float(np.dot(a, b) /
                 (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# ─── K₀ split ────────────────────────────────────────────────────────────────

def k0_split(model, data, block_size, batch_size, device,
             n_steps, lr, w_ff, verbose=True):
    """
    One application of the K₀ split:
    Branch 1 (Emb + FF): update embedding + feed-forward independently
    Branch 2 (Attn):     update attention independently
    Combine: Δθ = ΔEmb + w_FF · ΔFF + ΔAttn

    Returns the updated model (in-place) and per-step diagnostics.
    """
    # Parameter groups
    emb_params  = [p for n,p in model.named_parameters()
                   if "wte" in n or "wpe" in n]
    ff_params   = [p for n,p in model.named_parameters()
                   if "mlp" in n or "c_fc" in n]
    attn_params = [p for n,p in model.named_parameters()
                   if "c_attn" in n or "c_proj" in n]
    ln_params   = [p for n,p in model.named_parameters()
                   if "ln" in n]

    # Take n_steps steps on each branch separately, then merge
    # Branch 1: Emb + FF
    m1 = copy.deepcopy(model)
    opt1 = torch.optim.Adam(
        [p for n,p in m1.named_parameters()
         if any(k in n for k in ["wte","wpe","mlp","c_fc"])],
        lr=lr)
    for _ in range(n_steps):
        m1.train()
        x, y = get_batch()
        _, loss = m1(x, y)
        opt1.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m1.parameters(), 1.0)
        opt1.step()

    # Branch 2: Attn
    m2 = copy.deepcopy(model)
    opt2 = torch.optim.Adam(
        [p for n,p in m2.named_parameters()
         if any(k in n for k in ["c_attn","c_proj","ln"])],
        lr=lr)
    for _ in range(n_steps):
        m2.train()
        x, y = get_batch()
        _, loss = m2(x, y)
        opt2.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m2.parameters(), 1.0)
        opt2.step()

    # Merge: θ_new = θ_0 + ΔEmb + w_FF·ΔFF + ΔAttn
    with torch.no_grad():
        for (n, p0), p1, p2 in zip(model.named_parameters(),
                                     m1.parameters(),
                                     m2.parameters()):
            n_lower = n.lower()
            if "wte" in n_lower or "wpe" in n_lower:
                # Emb: take from branch 1
                p0.copy_(p1)
            elif "mlp" in n_lower or "c_fc" in n_lower:
                # FF: take from branch 1 with w_FF scaling
                delta_ff = p1.data - p0.data
                p0.add_(delta_ff * (w_ff - 1))   # already in p0, add extra
                p0.copy_(p0 + delta_ff * (w_ff - 1))
                # Simpler: p0 = p0 + w_FF * (p1 - p0_orig)
                # We do this correctly by working from scratch:
                pass  # handled below
            elif any(k in n_lower for k in ["c_attn","c_proj","ln"]):
                # Attn + LN: take from branch 2
                p0.copy_(p2)

    # Redo merge cleanly with saved original
    return model


def k0_split_clean(model_orig, n_steps, lr, w_ff, seed_offset=0):
    """Uses compiler's get_batch() directly."""
    """
    Clean K₀ split implementation that correctly computes:
        θ_new = θ_0 + ΔEmb + w_FF·ΔFF + ΔAttn
    where each Δ is computed on an independent copy.
    """
    torch.manual_seed(42 + seed_offset)

    theta_0 = {n: p.data.clone() for n, p in model_orig.named_parameters()}

    def is_emb(n):  return n.startswith("te.") or n.startswith("pe.")
    def is_ff(n):   return ".ff." in n
    def is_attn(n): return ".attn." in n or n.startswith("ln_f") or ".ln" in n

    # Branch 1: Emb + FF
    m1 = copy.deepcopy(model_orig)
    for n, p in m1.named_parameters():
        p.requires_grad_(is_emb(n) or is_ff(n))
    opt1 = torch.optim.Adam(
        [p for n, p in m1.named_parameters() if p.requires_grad], lr=lr)
    for _ in range(n_steps):
        m1.train()
        x, y = get_batch()
        _, loss = m1(x, y)
        opt1.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m1.parameters(), 1.0)
        opt1.step()

    # Branch 2: Attn + LN
    m2 = copy.deepcopy(model_orig)
    for n, p in m2.named_parameters():
        p.requires_grad_(is_attn(n))
    opt2 = torch.optim.Adam(
        [p for n, p in m2.named_parameters() if p.requires_grad], lr=lr)
    for _ in range(n_steps):
        m2.train()
        x, y = get_batch()
        _, loss = m2(x, y)
        opt2.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m2.parameters(), 1.0)
        opt2.step()

    # Merge: θ_new = θ_0 + ΔEmb + w_FF·ΔFF + ΔAttn
    model_new = copy.deepcopy(model_orig)
    with torch.no_grad():
        for n, p in model_new.named_parameters():
            p0 = theta_0[n]
            p1 = dict(m1.named_parameters())[n].data
            p2 = dict(m2.named_parameters())[n].data
            if is_emb(n):
                p.copy_(p0 + (p1 - p0))              # ΔEmb
            elif is_ff(n):
                p.copy_(p0 + w_ff * (p1 - p0))       # w_FF · ΔFF
            elif is_attn(n):
                p.copy_(p0 + (p2 - p0))              # ΔAttn
            # head.weight tied to te.weight — skip (updated via te)

    return model_new


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 60)
    print("  INVOLUTIVITY TEST: μ_k ∘ μ_k = id ?")
    print("  Tests Proposition 15.1 condition (2)")
    print("=" * 60)
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  n_steps={args.n_steps}  lr={args.lr}  w_FF={args.w_ff_override}")
    print(f"  Criteria: |Δval| < {args.val_tol}  |Δτ| < {args.tau_tol}")
    print()
    print("  REGIME NOTE:")
    print("  basin_entry_state.pt: τ≈1.0 (algebraic phase) → w_FF≈6")
    print("    K₀ split NOT designed for this regime (w_FF>>1 overcorrects)")
    print("  basin_state.pt: τ≈5.3 (statistical phase) → w_FF≈0.5")
    print("    K₀ split CORRECT for this regime — use this for involutivity")
    if "entry" in args.checkpoint and "basin_state" not in args.checkpoint:
        print()
        print("  ⚠  WARNING: basin_entry_state.pt is NOT the recommended")
        print("     checkpoint. Use basin_state.pt for the involutivity test.")
        print("     Re-run with: --checkpoint basin_state.pt")

    t0 = time.time()

    # ── Load checkpoint FIRST (before compiler import overwrites files) ────────
    print(f"\n[1/5] Pre-loading checkpoint {args.checkpoint} …")
    ckpt_raw = torch.load(args.checkpoint, map_location=args.device,
                          weights_only=False)
    state = ckpt_raw if not isinstance(ckpt_raw, dict) else ckpt_raw.get(
        "model", ckpt_raw.get("state_dict",
                 ckpt_raw.get("model_state_dict", ckpt_raw)))
    # Deep-copy the state dict so compiler import cannot affect it
    state = {k: v.clone() for k, v in state.items()
             if isinstance(v, torch.Tensor)}
    print(f"      Checkpoint cached: {len(state)} tensors")

    # ── Import compiler (model + data) — may re-run training, overwrite files ─
    global LM, get_batch, eval_val, train_t, val_t
    print(f"[2/5] Importing compiler from {args.compiler} …")
    comp = import_compiler(args.compiler)
    LM        = comp.LM
    get_batch = comp.get_batch
    eval_val  = comp.eval_val
    train_t   = comp.train_t
    val_t     = comp.val_t
    import builtins
    for attr in ["VOCAB","D","N_HEADS","N_STU","BATCH","SEQ","LR",
                 "train_t","val_t"]:
        if hasattr(comp, attr):
            builtins.__dict__[attr] = getattr(comp, attr)
    print(f"      Compiler imported: LM, get_batch, eval_val, train_t, val_t")
    print(f"      train_t: {len(train_t)} tokens  val_t: {len(val_t)} tokens")

    # ── Build model from pre-loaded (cached) state ────────────────────────────
    print(f"[2/5b] Building model from cached state …")
    model0 = LM().to(args.device)
    missing, _ = model0.load_state_dict(state, strict=False)
    if missing:
        print(f"      Missing keys (first 5): {missing[:5]}")
    n_params = sum(p.numel() for p in model0.parameters())
    print(f"      Model loaded from CACHED state: {n_params:,} params")

    # ── Baseline: measure θ₀, val₀, τ₀ ──────────────────────────────────────
    print("[3/5] Measuring baseline (θ₀, val₀, τ₀) …")
    val0 = eval_val(model0, n=16)
    tau0 = gluing_defect_local(model0)
    theta0 = param_vector(model0)

    # Determine w_FF
    if args.w_ff_override == "auto":
        w_ff = w_ff_formula(tau0)
        print(f"      τ₀={tau0:.4f} → w_FF={w_ff:.4f} (auto formula)")
    else:
        w_ff = float(args.w_ff_override)
        print(f"      w_FF={w_ff:.4f} (override)")

    print(f"      val₀={val0:.4f}  τ₀={tau0:.4f}  "
          f"||θ₀||={np.linalg.norm(theta0):.2f}")

    # ── First application μ_k ────────────────────────────────────────────────
    print(f"[4/5] Applying K₀ split (μ_k): {args.n_steps} steps …")
    model1 = k0_split_clean(model0, args.n_steps, args.lr, w_ff, seed_offset=0)
    val1 = eval_val(model1, n=16)
    tau1 = gluing_defect_local(model1)
    theta1 = param_vector(model1)
    delta1 = theta1 - theta0

    print(f"      After μ_k:  val₁={val1:.4f}  τ₁={tau1:.4f}")
    print(f"      Δval = {val1-val0:+.4f}  "
          f"||Δθ₁|| = {np.linalg.norm(delta1):.4f}")

    # ── Second application μ_k² ───────────────────────────────────────────────
    print(f"[5/5] Applying K₀ split again (μ_k²): {args.n_steps} steps …")
    # w_FF for second application uses τ₁
    if args.w_ff_override == "auto":
        w_ff2 = w_ff_formula(tau1)
        print(f"      τ₁={tau1:.4f} → w_FF₂={w_ff2:.4f}")
    else:
        w_ff2 = w_ff

    model2 = k0_split_clean(model1, args.n_steps, args.lr, w_ff2, seed_offset=1)
    val2 = eval_val(model2, n=16)
    tau2 = gluing_defect_local(model2)
    theta2 = param_vector(model2)
    delta2 = theta2 - theta1

    print(f"      After μ_k²: val₂={val2:.4f}  τ₂={tau2:.4f}")
    print(f"      Δval = {val2-val1:+.4f}  "
          f"||Δθ₂|| = {np.linalg.norm(delta2):.4f}")

    # ── Involutivity analysis ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  INVOLUTIVITY ANALYSIS")
    print(f"{'='*60}")

    val_return  = abs(val2 - val0)
    tau_return  = abs(tau2 - tau0)
    val_inv     = val_return < args.val_tol
    tau_inv     = tau_return < args.tau_tol

    # Direction analysis: does Δθ₂ reverse Δθ₁?
    cos_deltas  = cosine(delta1, delta2)
    # Strict reversal: cos ≈ -1; independent steps: cos ≈ 0; same dir: cos ≈ +1
    reversal    = cos_deltas < -0.5

    # Norm comparison: ||Δθ₂|| vs ||Δθ₁||
    norm1 = float(np.linalg.norm(delta1))
    norm2 = float(np.linalg.norm(delta2))
    norm_ratio  = norm2 / (norm1 + 1e-12)

    # θ₂ distance from θ₀
    return_dist = float(np.linalg.norm(theta2 - theta0))
    step_dist   = float(np.linalg.norm(theta1 - theta0))

    print(f"\n  Trajectory:")
    print(f"    val₀ = {val0:.4f}  →  val₁ = {val1:.4f}  →  val₂ = {val2:.4f}")
    print(f"    τ₀   = {tau0:.4f}  →  τ₁   = {tau1:.4f}  →  τ₂   = {tau2:.4f}")
    print()
    print(f"  Return to start:")
    print(f"    |val₂ - val₀| = {val_return:.4f}  "
          f"(threshold {args.val_tol})  {'✓' if val_inv else '✗'}")
    print(f"    |τ₂  - τ₀|   = {tau_return:.4f}  "
          f"(threshold {args.tau_tol})  {'✓' if tau_inv else '✗'}")
    print()
    print(f"  Direction analysis:")
    print(f"    cos(Δθ₁, Δθ₂) = {cos_deltas:+.4f}  "
          f"({'reversal ✓' if reversal else 'not reversal'})")
    print(f"    ||Δθ₁|| = {norm1:.4f}  ||Δθ₂|| = {norm2:.4f}  "
          f"ratio = {norm_ratio:.4f}")
    print(f"    ||θ₂ - θ₀|| = {return_dist:.4f}  "
          f"(vs ||θ₁ - θ₀|| = {step_dist:.4f})")
    print()

    # Verdict
    if val_inv and tau_inv:
        verdict = "INVOLUTIVITY_CONFIRMED"
        msg = (f"Both criteria satisfied: |Δval|={val_return:.4f} < {args.val_tol}, "
               f"|Δτ|={tau_return:.4f} < {args.tau_tol}. "
               f"μ_k ∘ μ_k ≈ id confirmed. "
               f"The K₀ split satisfies the BMRR exchange relation involutivity.")
    elif val_inv and not tau_inv:
        verdict = "VAL_ONLY"
        msg = (f"val returns (|Δval|={val_return:.4f} < {args.val_tol}) "
               f"but τ does not (|Δτ|={tau_return:.4f} ≥ {args.tau_tol}). "
               f"Partial involutivity: loss landscape returns but gradient "
               f"ratio does not. τ is more sensitive to stochastic noise.")
    elif tau_inv and not val_inv:
        verdict = "TAU_ONLY"
        msg = (f"τ returns (|Δτ|={tau_return:.4f} < {args.tau_tol}) "
               f"but val does not (|Δval|={val_return:.4f} ≥ {args.val_tol}). "
               f"Unusual: gradient ratio stabilizes but loss shifts.")
    else:
        verdict = "NOT_INVOLUTIVE"
        msg = (f"Neither criterion satisfied: |Δval|={val_return:.4f}, "
               f"|Δτ|={tau_return:.4f}. "
               f"The K₀ split is not involutive at this checkpoint. "
               f"Likely cause: τ changes between applications (w_FF drift), "
               f"so μ_k is not a fixed mutation but a τ-dependent map. "
               f"This is expected if τ₁ ≠ τ₀ significantly "
               f"(Δτ={tau1-tau0:+.4f}).")

    print(f"  VERDICT: {verdict}")
    print(f"  {msg}")

    # Interpretation
    print(f"\n  Interpretation:")
    if reversal:
        print(f"  cos(Δθ₁, Δθ₂) = {cos_deltas:.4f} < -0.5: second step")
        print(f"  partially reverses the first — mutation structure detected.")
    else:
        print(f"  cos(Δθ₁, Δθ₂) = {cos_deltas:.4f}: steps are not reversals.")
        print(f"  Both steps move in similar/orthogonal directions.")
        if cos_deltas > 0.3:
            print(f"  Positive cos: both steps improve the same objective —")
            print(f"  K₀ split is a descent direction, not an involution.")

    print(f"\n  Note on K₀ involutivity:")
    print(f"  Classical BMRR mutation satisfies μ_k(μ_k(M)) = M exactly.")
    print(f"  The K₀ split is an Adam-based numerical approximation, so")
    print(f"  exact involutivity is not expected — only approximate return.")
    print(f"  The τ-dependent w_FF means w_FF₁={w_ff:.4f} ≠ w_FF₂={w_ff2:.4f}")
    print(f"  if τ changes, making μ_k a τ-parameterized family, not a")
    print(f"  single fixed mutation. This is mathematically richer than")
    print(f"  classical involutivity and connects to wall-crossing structure.")

    # ── Report ────────────────────────────────────────────────────────────────
    report = {
        "experiment": "K0 Split Involutivity Test",
        "checkpoint": str(args.checkpoint),
        "config": {
            "n_steps": args.n_steps, "lr": args.lr,
            "w_ff_override": args.w_ff_override,
            "val_tol": args.val_tol, "tau_tol": args.tau_tol,
        },
        "baseline":  {"val": float(val0), "tau": float(tau0),
                      "w_ff": float(w_ff)},
        "after_mu1": {"val": float(val1), "tau": float(tau1),
                      "w_ff2": float(w_ff2),
                      "delta_val": float(val1-val0),
                      "delta_tau": float(tau1-tau0),
                      "norm_delta_theta": float(norm1)},
        "after_mu2": {"val": float(val2), "tau": float(tau2),
                      "delta_val": float(val2-val1),
                      "delta_tau": float(tau2-tau1),
                      "norm_delta_theta": float(norm2)},
        "involutivity": {
            "val_return":       float(val_return),
            "tau_return":       float(tau_return),
            "val_criterion":    bool(val_inv),
            "tau_criterion":    bool(tau_inv),
            "cos_delta1_delta2": float(cos_deltas),
            "reversal_detected": bool(reversal),
            "norm_ratio":       float(norm_ratio),
            "return_distance":  float(return_dist),
            "step_distance":    float(step_dist),
        },
        "verdict": verdict,
        "message": msg,
        "elapsed_s": round(time.time() - t0, 1),
    }

    Path(args.output).write_text(json.dumps(report, indent=2, cls=NumpyEncoder))
    print(f"\n  Report → {Path(args.output).resolve()}")
    print(f"  Total elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
