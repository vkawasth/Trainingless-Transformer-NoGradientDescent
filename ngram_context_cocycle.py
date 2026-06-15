#!/usr/bin/env python3
"""
CONTEXT COCYCLE HOLONOMY EXTRACTOR WITH UNROLLED HISTORY TENSORS
================================================================
Bypasses the first-order Markov power fallacy by unrolling the multi-scale 
context history directly from token co-occurrence windows.

Mathematical Formulation:
  M_ij = \sum_{sequences} \sum_{k=1}^{SEQ} \mathbf{1}[x_t = i \text{ and } x_{t+k} = j] \cdot \gamma^k
  
This constructs a fat-tailed, high-rank analytical spectrum matrix (Cartan frame)
without training a model or experiencing ergodic collapse into a rank-1 vector.
"""
import json, math, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; BATCH=8; SEQ=64; LR=3e-4
PROJ=48; HEAD_STEPS=100
GAMMA=0.85  # Context decay factor: balances localized vs global token history

print(f"\n{'='*75}")
print(f"  UNROLLED CONTEXT COCYCLE HOLONOMY EXTRACTOR")
print(f"  High-Rank Zero-Shot Spectrum Injection Engine (\u03b3 = {GAMMA})")
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
        h=self.te(x)+self.pe(torch.arange(x.shape[1]))
        for b in self.blocks: h=b(h)
        logits=self.head(self.ln_f(h))
        return logits,(F.cross_entropy(logits.view(-1,VOCAB),y.view(-1)) if y is not None else None)
    def hidden_states(self,x):
        hs=[]; h=self.te(x)+self.pe(torch.arange(x.shape[1])); hs.append(h.detach())
        for b in self.blocks: h=b(h); hs.append(h.detach())
        return hs
    def get_hidden(self,x):
        h=self.te(x)+self.pe(torch.arange(x.shape[1]))
        for b in self.blocks: h=b(h)
        return self.ln_f(h)

def layer_jac(block,h_in,pos,m):
    seq,d_=h_in.shape; m=min(m,seq,d_)
    _,_,Vt=torch.linalg.svd(h_in,full_matrices=False)
    U=Vt[:m,:].T.detach(); J=np.zeros((m,m))
    with torch.enable_grad():
        for i in range(m):
            hh=h_in.clone().unsqueeze(0).detach().requires_grad_(True)
            ho=block(hh)
            v=(ho[0,pos,:] if ho.dim()==3 else ho[pos,:])
            (v*U[:,i]).sum().backward()
            g=hh.grad; g=(g[0,pos,:] if g.dim()==3 else g[pos,:]).detach()
            J[:,i]=(U.T@g).numpy()
    return J.T,U.detach().numpy(),m

# ── Step 1: Analytical Unrolled History Tensor Extraction ───────────────────
print("Step 1: Compiling high-rank context cocycle from raw data topology...")
def compute_unrolled_cocycle_spectrum(train_ids, vocab_size, seq_len=64, proj_dim=48, gamma=0.85):
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
    
    # Scale to base student footprint scale
    Analytical_Lambda = (Analytical_Lambda / (Analytical_Lambda[0] + 1e-8)) * 25.0
    return Analytical_Lambda

high_rank_analytical_sv = compute_unrolled_cocycle_spectrum(train_ids, VOCAB, SEQ, PROJ, GAMMA)
print(f"  Unrolled Cocycle Spectrum (Top 4 \u039b): {high_rank_analytical_sv[:4].round(4)}")


# ── Step 2: Initialize & Train Base 2L Student Target ───────────────────────
print("\nStep 2: Activating 2L Student Realization Sequence...")
torch.manual_seed(99)
student2 = LM(D, N_HEADS, 2)
opt_s = torch.optim.AdamW(student2.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1)

for s in range(1, 201):
    student2.train(); x, y = get_batch(); _, loss = student2(x, y)
    opt_s.zero_grad(); loss.backward(); opt_s.step()
student2.eval()

x_ref, _ = get_batch('val'); x_ref = x_ref[0:1]
pos = SEQ // 2; m = min(PROJ, SEQ, D)
with torch.no_grad():
    hs = student2.hidden_states(x_ref); hs = [h[0] for h in hs]

Js = []; U0_2l = None; ma = None
for l in range(2):
    J, U, m_ = layer_jac(student2.blocks[l], hs[l], pos, m)
    Js.append(J)
    if U0_2l is None: U0_2l = U; ma = m_

M_2l = np.eye(ma)
for J in reversed(Js): M_2l = J @ M_2l
sv_2l = np.linalg.svd(M_2l, compute_uv=False)
print(f"  Extracted Empirical 2L Spectrum  (Top 4 \u039b): {sv_2l[:4].round(4)}")


# ── Step 3: Cartan Subalgebra Lie Deflection Engine ──────────────────────────
def execute_lie_cartan_edit(M_base, target_sv, current_sv):
    U, Sigma, Vt = np.linalg.svd(M_base)
    t_sv = np.zeros_like(Sigma)
    trunc = min(len(Sigma), len(target_sv))
    t_sv[:trunc] = target_sv[:trunc]
    
    ratio_vector = t_sv / (current_sv + 1e-8)
    Sigma_edited = Sigma * ratio_vector
    return U @ np.diag(Sigma_edited) @ Vt

def evaluate_calibrated_head(student_model, M_edited_np, U0_np, steps=HEAD_STEPS):
    head_fresh = nn.Linear(D, VOCAB, bias=False)
    U0_t = torch.tensor(U0_np, dtype=torch.float32)
    M_t = torch.tensor(M_edited_np, dtype=torch.float32)

    for p in student_model.parameters(): p.requires_grad_(False)
    head_fresh.weight.requires_grad_(True)
    opt_h = torch.optim.AdamW([head_fresh.weight], lr=LR*3, betas=(0.9, 0.95), weight_decay=0.01)

    def forward_corrected(x, y):
        with torch.no_grad():
            h = student_model.get_hidden(x)
            B_, S_, D_ = h.shape; h_flat = h.reshape(-1, D_)
            h_proj = h_flat @ U0_t; h_ref = h_proj @ M_t; h_lift = h_ref @ U0_t.T
            h_orth = h_flat - h_flat @ U0_t @ U0_t.T
            h_out = (h_lift + h_orth).reshape(B_, S_, D_)
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
            h = student_model.get_hidden(x)
            B_, S_, D_ = h.shape; h_flat = h.reshape(-1, D_)
            h_proj = h_flat @ U0_t; h_ref = h_proj @ M_t; h_lift = h_ref @ U0_t.T
            h_orth = h_flat - h_flat @ U0_t @ U0_t.T
            h_out = (h_lift + h_orth).reshape(B_, S_, D_)
            logits = head_fresh(h_out)
            ls.append(F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1)).item())
            
    for p in student_model.parameters(): p.requires_grad_(True)
    return float(np.mean(ls))


# ── Step 4: Run Injection Evaluation ──────────────────────────────────────────
print("\nStep 4: Executing high-rank zero-shot spectrum injection...")

val_control = evaluate_calibrated_head(student2, M_2l, U0_2l)
print(f"  [Control] 2L Operator (Unedited Spectrum)    ──► Val: {val_control:.4f}")

M_analytical = execute_lie_cartan_edit(M_2l, high_rank_analytical_sv, sv_2l)
val_analytical = evaluate_calibrated_head(student2, M_analytical, U0_2l)
print(f"  [Injected] 2L Operator + Analytical Spectrum ──► Val: {val_analytical:.4f}")


# ── Final Verdict Analysis ───────────────────────────────────────────────────
print(f"\n{'='*75}")
print(f"  HIGH-RANK ZERO-SHOT CONTEXT COCYCLE RESULTS SUMMARY")
print(f"{'='*75}")
print(f"  2L Control Baseline (Empirical Spectrum):    {val_control:.4f}")
print(f"  2L Model with High-Rank Cocycle Spectrum:   {val_analytical:.4f}")
print(f"{'='*75}\n")