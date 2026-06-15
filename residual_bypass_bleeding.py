#!/usr/bin/env python3
"""
ACTIVATION COVARIANCE ALIGNMENT WITH RESIDUAL BYPASS BLEEDING
============================================================
Injects a structural residual bypass from the mature embedding space 
straight to the alignment engine to prevent untrained transformer blocks 
from triggering a supersingular coordinate collapse.

Mathematical Formulation:
  H_stabilized = alpha * H_blocks + (1 - alpha) * H_embeddings
  Cov(H) = H^T * H / N -> Yields stable, non-collapsed empirical singular values.
"""
import json, math, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_LAYERS_TEACHER=24; BATCH=8; SEQ=64; LR=3e-4
PROJ=48; HEAD_STEPS=100
GAMMA=0.85     # Structural context decay factor
ALPHA=0.20     # Residual bleeding blend factor (20% blocks, 80% raw embedding)

print(f"\n{'='*75}")
print(f"  STABILIZED RESIDUAL ACTIVATION ALIGNMENT ENGINE")
print(f"  High-Rank Zero-Shot Space Injection (\u03b3 = {GAMMA}, \u03b1 = {ALPHA})")
print(f"{'='*75}\n")

with open('/tmp/train_ids.json') as f: train_ids=list(json.load(f))
with open('/tmp/val_ids.json')   as f: val_ids=list(json.load(f))
with open('/tmp/vocab.json')     as f: vocab=json.load(f)
VOCAB=len(vocab)
train_t=torch.tensor(train_ids,dtype=torch.long)
val_t  =torch.tensor(val_ids,  dtype=torch.long)

def get_batch(split='train'):
    data=train_t if split=='train' else val_t
    ix=torch.randint(0,len(data)-SEQ-1,(BATCH,))
    return (torch.stack([data[i:i+SEQ] for i in ix]),
            torch.stack([data[i+1:i+SEQ+1] for i in ix]))

class Attn(nn.Module):
    def __init__(self,d,nh):
        super().__init__()
        self.nh=nh; self.dh=d//nh; self.sc=math.sqrt(d//nh)
        self.WQ=nn.Linear(d,d,bias=False); self.WK=nn.Linear(d,d,bias=False)
        self.WV=nn.Linear(d,d,bias=False); self.op=nn.Linear(d,d,bias=False)
        self.ln=nn.LayerNorm(d)
        for w in [self.WQ,self.WK,self.WV,self.op]: nn.init.normal_(w.weight,std=0.02)
    def forward(self,h):
        B,S,D_=h.shape; H=self.nh; dh=self.dh
        Q=self.WQ(h).view(B,S,H,dh).transpose(1,2)
        K=self.WK(h).view(B,S,H,dh).transpose(1,2)
        V=self.WV(h).view(B,S,H,dh).transpose(1,2)
        sc=Q@K.transpose(-2,-1)/self.sc
        mask=torch.triu(torch.ones(S,S,device=h.device),diagonal=1).bool()
        sc=sc.masked_fill(mask.unsqueeze(0).unsqueeze(0),float('-inf'))
        out=(F.softmax(sc,dim=-1)@V).transpose(1,2).reshape(B,S,D_)
        return self.ln(h+self.op(out))

class FF(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.g=nn.Linear(d,d*2,bias=False); self.v=nn.Linear(d,d*2,bias=False)
        self.o=nn.Linear(d*2,d,bias=False); self.n=nn.LayerNorm(d)
        for w in [self.g,self.v,self.o]: nn.init.normal_(w.weight,std=0.02)
    def forward(self,h): return self.n(h+self.o(F.silu(self.g(h))*self.v(h)))

class Block(nn.Module):
    def __init__(self,d,nh): super().__init__(); self.attn=Attn(d,nh); self.ff=FF(d)
    def forward(self,h): return self.ff(self.attn(h))

class LM(nn.Module):
    def __init__(self,d,nh,nl):
        super().__init__()
        self.te=nn.Embedding(VOCAB,d); self.pe=nn.Embedding(512,d)
        self.blocks=nn.ModuleList([Block(d,nh) for _ in range(nl)])
        self.ln_f=nn.LayerNorm(d)
        self.head=nn.Linear(d,VOCAB,bias=False); self.head.weight=self.te.weight
    def forward(self,x,y=None):
        h_emb = self.te(x) + self.pe(torch.arange(x.shape[1]))
        h = h_emb
        for b in self.blocks: h = b(h)
        # Apply structural residual bleeding to save high-rank representation
        h_bleed = ALPHA * h + (1.0 - ALPHA) * h_emb
        logits = self.head(self.ln_f(h_bleed))
        return logits,(F.cross_entropy(logits.view(-1,VOCAB),y.view(-1)) if y is not None else None)
        
    def get_hidden(self,x):
        h_emb = self.te(x) + self.pe(torch.arange(x.shape[1]))
        h = h_emb
        for b in self.blocks: h = b(h)
        # Preserve rank continuity prior to coordinate projection
        h_bleed = ALPHA * h + (1.0 - ALPHA) * h_emb
        return self.ln_f(h_bleed)

# ── Step 1: Baseline 24L Teacher Calibration Space ────────────────────────────
print("Step 1: Training 24L Teacher to calibrate embedding manifold...")
torch.manual_seed(42)
teacher = LM(D, N_HEADS, N_LAYERS_TEACHER)
opt_t = torch.optim.AdamW(teacher.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1)

for step in range(1, 201):
    teacher.train(); x, y = get_batch(); _, loss = teacher(x, y)
    opt_t.zero_grad(); loss.backward(); opt_t.step()
teacher.eval()
print("  Teacher workspace locked.")

# ── Step 2: Analytical High-Rank Context Cocycle Compiler ─────────────────────
print("\nStep 2: Compiling high-rank context cocycle from raw text data...")
def compute_unrolled_cocycle_spectrum(train_ids, vocab_size, seq_len=64, proj_dim=256, gamma=0.85):
    ids = np.array(train_ids, dtype=np.int32)
    N = len(ids)
    M_cocycle = np.zeros((vocab_size, vocab_size), dtype=np.float32)
    weights = np.array([math.pow(gamma, k) for k in range(1, seq_len + 1)], dtype=np.float32)
    
    for k in range(1, seq_len + 1):
        w = weights[k - 1]
        source_tokens = ids[:N - k]
        target_tokens = ids[k:]
        for i, j in zip(source_tokens, target_tokens):
            if i < vocab_size and j < vocab_size:
                M_cocycle[i, j] += w
                
    row_sums = M_cocycle.sum(axis=1, keepdims=True)
    M_cocycle = np.divide(M_cocycle, row_sums, out=np.zeros_like(M_cocycle), where=row_sums!=0)
    
    M_subspace = M_cocycle[:proj_dim, :proj_dim]
    Analytical_Lambda = np.linalg.svd(M_subspace, compute_uv=False)
    
    # Normalize analytical scale to unit energy bounds
    Analytical_Lambda = Analytical_Lambda / (Analytical_Lambda[0] + 1e-8)
    return Analytical_Lambda

high_rank_analytical_sv = compute_unrolled_cocycle_spectrum(train_ids, VOCAB, SEQ, D, GAMMA)
print(f"  Normalized Analytical Spectrum (Top 4 \u039b): {high_rank_analytical_sv[:4].round(4)}")

# ── Step 3: Train 2L Student with Transferred Teacher Seeding ─────────────────
print("\nStep 3: Training 2L student with mature teacher embedding matrices...")
torch.manual_seed(99)
student2 = LM(D, N_HEADS, 2)

for attr in ['te', 'pe', 'ln_f']:
    src = getattr(teacher, attr); dst = getattr(student2, attr)
    if hasattr(src, 'weight'): dst.weight.data.copy_(src.weight.data)
    if hasattr(src, 'bias') and src.bias is not None: dst.bias.data.copy_(src.bias.data)

opt_s = torch.optim.AdamW(student2.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1)
for step in range(1, 201):
    student2.train(); x, y = get_batch(); _, loss = student2(x, y)
    opt_s.zero_grad(); loss.backward(); opt_s.step()
student2.eval()

# Directly measure the Activation Spectrum protected by the bleeding bypass
with torch.no_grad():
    x_val, _ = get_batch('val')
    h_base = student2.get_hidden(x_val) 
    h_flat = h_base.reshape(-1, D)      
    cov_matrix = (h_flat.T @ h_flat) / h_flat.shape[0]
    sv_empirical = torch.linalg.svdvals(cov_matrix).cpu().numpy()

print(f"  Extracted Activation Spectrum (Top 4 \u039b): {sv_empirical[:4].round(4)}")

# ── Step 4: Activation Deflection Subspace Engine ────────────────────────────
def compute_spectral_edit_operator(cov_mat_torch, target_sv_np):
    U, Sigma, Vt = torch.linalg.svd(cov_mat_torch)
    Sigma_np = Sigma.cpu().numpy()
    
    # Scale target spectrum to match the absolute baseline variance scale
    t_sv = target_sv_np * Sigma_np[0]
    
    # Compute continuous scaling ratios across the non-collapsed variety
    ratio_vector = t_sv / (Sigma_np + 1e-8)
    # Clip extreme updates to preserve structural continuity
    ratio_vector = np.clip(ratio_vector, 0.1, 10.0)
    
    Sigma_edited = Sigma_np * ratio_vector
    M_edit = U @ torch.diag(torch.tensor(Sigma_edited, dtype=torch.float32)) @ Vt
    return M_edit

M_deflection = compute_spectral_edit_operator(cov_matrix, high_rank_analytical_sv)

def evaluate_calibrated_head(student_model, T_edit_matrix, steps=HEAD_STEPS):
    head_fresh = nn.Linear(D, VOCAB, bias=False)
    head_fresh.weight.data.copy_(teacher.te.weight.data)
    
    for p in student_model.parameters(): p.requires_grad_(False)
    head_fresh.weight.requires_grad_(True)
    opt_h = torch.optim.AdamW([head_fresh.weight], lr=LR*3, betas=(0.9, 0.95), weight_decay=0.01)

    def forward_corrected(x, y):
        with torch.no_grad():
            h = student_model.get_hidden(x)
            B_, S_, D_ = h.shape
            h_flat = h.reshape(-1, D_)
            if T_edit_matrix is not None:
                h_flat = h_flat @ T_edit_matrix
            h_out = h_flat.reshape(B_, S_, D_)
        logits = head_fresh(h_out)
        return F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))

    for _ in range(steps):
        student_model.eval(); x, y = get_batch()
        loss = forward_corrected(x, y)
        opt_h.zero_grad(); loss.backward(); opt_h.step()

    student_model.eval(); ls = []
    with torch.no_grad():
        for _ in range(40):
            x, y = get_batch('val')
            loss = forward_corrected(x, y)
            ls.append(loss.item())
            
    for p in student_model.parameters(): p.requires_grad_(True)
    return float(np.mean(ls))

# ── Step 5: Run Evaluation Sweep ──────────────────────────────────────────────
print("\nStep 4: Executing zero-shot activation spectrum injection...")

val_control = evaluate_calibrated_head(student2, T_edit_matrix=None)
print(f"  [Control] 2L Operator (Unedited Activation Space) ──► Val: {val_control:.4f}")

val_analytical = evaluate_calibrated_head(student2, T_edit_matrix=M_deflection)
print(f"  [Injected] 2L Operator + Analytical Spectrum     ──► Val: {val_analytical:.4f}")

print(f"\n{'='*75}")
print(f"  HIGH-RANK ZERO-SHOT CONTEXT COCYCLE RESULTS SUMMARY")
print(f"{'='*75}")
print(f"  2L Control Baseline (Empirical Space):       {val_control:.4f}")
print(f"  2L Model with High-Rank Activation Cocycle:  {val_analytical:.4f}")
print(f"{'='*75}\n")