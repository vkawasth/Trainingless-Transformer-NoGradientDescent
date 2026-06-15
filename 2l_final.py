#!/usr/bin/env python3
"""
STRUCTURAL RESIDUAL MANIFOLD THREE-WAY COMPARISON HARNESS
========================================================
The definitive evaluation suite for the Context Algebra Programme.
Evaluates three distinct topological manifold configurations side-by-side 
using independent, calibrated linear decoding fields.

Configurations:
  1. 24L Teacher Baseline (Gold Standard Calibrated Space)
  2. 2L Student Control   (Standard Sequential Topology)
  3. 2L Bleeding Engine   (Structural Residual Bypass, alpha = 0.2)
"""
import json, math, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_LAYERS_TEACHER=24; BATCH=8; SEQ=64; LR=3e-4
ALPHA=0.20       # Structural residual bleeding blend factor
HEAD_STEPS=100

print(f"\n{'='*75}")
print(f"  STRUCTURAL RESIDUAL MANIFOLD THREE-WAY HARNESS")
print(f"  Parallel Evaluation Horizon (\u03b1 = {ALPHA})")
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
        
    def forward(self,x,y=None,use_bleed=False):
        h_emb = self.te(x) + self.pe(torch.arange(x.shape[1]))
        h = h_emb
        for b in self.blocks: h = b(h)
        h_final = (ALPHA * h + (1.0 - ALPHA) * h_emb) if use_bleed else h
        logits = self.head(self.ln_f(h_final))
        return logits,(F.cross_entropy(logits.view(-1,VOCAB),y.view(-1)) if y is not None else None)
        
    def get_hidden(self,x,use_bleed=False):
        h_emb = self.te(x) + self.pe(torch.arange(x.shape[1]))
        h = h_emb
        for b in self.blocks: h = b(h)
        h_final = (ALPHA * h + (1.0 - ALPHA) * h_emb) if use_bleed else h
        return self.ln_f(h_final)

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

# ── Step 2: Initialize 2L Student with Transferred Teacher Seeding ─────────────
print("\nStep 2: Initializing 2L Student with mature teacher embeddings...")
torch.manual_seed(99)
student2 = LM(D, N_HEADS, 2)

for attr in ['te', 'pe', 'ln_f']:
    src = getattr(teacher, attr); dst = getattr(student2, attr)
    if hasattr(src, 'weight'): dst.weight.data.copy_(src.weight.data)
    if hasattr(src, 'bias') and src.bias is not None: dst.bias.data.copy_(src.bias.data)

# Extract and monitor the structural profiles across all spaces
with torch.no_grad():
    x_val, _ = get_batch('val')
    
    # 24L Teacher Matrix Profile
    h_teacher = teacher.get_hidden(x_val, use_bleed=False).reshape(-1, D)
    sv_teacher = torch.linalg.svdvals((h_teacher.T @ h_teacher) / h_teacher.shape[0]).cpu().numpy()
    
    # 2L Student Control Matrix Profile
    h_control = student2.get_hidden(x_val, use_bleed=False).reshape(-1, D)
    sv_control = torch.linalg.svdvals((h_control.T @ h_control) / h_control.shape[0]).cpu().numpy()
    
    # 2L Bleeding Engine Matrix Profile
    h_bleed = student2.get_hidden(x_val, use_bleed=True).reshape(-1, D)
    sv_bleed = torch.linalg.svdvals((h_bleed.T @ h_bleed) / h_bleed.shape[0]).cpu().numpy()

print(f"  Teacher 24L Spectrum    (Top 4 \u039b): {sv_teacher[:4].round(4)}")
print(f"  Student 2L Control Spec (Top 4 \u039b): {sv_control[:4].round(4)}")
print(f"  Student 2L Bleeding Spec(Top 4 \u039b): {sv_bleed[:4].round(4)}")

# ── Step 3: Calibrate and Evaluate Decoupled Operators ───────────────────────
print("\nStep 3: Calibrating isolated decoding heads for parallel evaluation...")

def evaluate_manifold_state(model, use_bleed, steps=HEAD_STEPS):
    head_fresh = nn.Linear(D, VOCAB, bias=False)
    
    for p in model.parameters(): p.requires_grad_(False)
    head_fresh.weight.requires_grad_(True)
    opt_h = torch.optim.AdamW([head_fresh.weight], lr=LR*3, betas=(0.9, 0.95), weight_decay=0.01)

    def forward_corrected(x, y):
        with torch.no_grad():
            h = model.get_hidden(x, use_bleed=use_bleed)
        logits = head_fresh(h)
        return F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))

    for _ in range(steps):
        model.eval(); x, y = get_batch()
        loss = forward_corrected(x, y)
        opt_h.zero_grad(); loss.backward(); opt_h.step()

    model.eval(); ls = []
    with torch.no_grad():
        for _ in range(40):
            x, y = get_batch('val')
            loss = forward_corrected(x, y)
            ls.append(loss.item())
            
    for p in model.parameters(): p.requires_grad_(True)
    return float(np.mean(ls))

# ── Step 4: Execute Evaluation Sweep ──────────────────────────────────────────
print("Step 4: Compiling validation metrics...")

val_teacher  = evaluate_manifold_state(teacher,  use_bleed=False)
print(f"  [Gold Standard] 24L Teacher Space        ──► Val: {val_teacher:.4f}")

val_control  = evaluate_manifold_state(student2, use_bleed=False)
print(f"  [Control]       2L Standard Student      ──► Val: {val_control:.4f}")

val_bleeding = evaluate_manifold_state(student2, use_bleed=True)
print(f"  [Bleeding]      2L Student + Bypass (\u03b1) ──► Val: {val_bleeding:.4f}")

print(f"\n{'='*75}")
print(f"  CONTEXT ALGEBRA THREE-WAY RESULTS SUMMARY")
print(f"{'='*75}")
print(f"  Configuration 1: 24L Teacher Baseline:        {val_teacher:.4f}")
print(f"  Configuration 2: 2L Student Control Space:    {val_control:.4f}")
print(f"  Configuration 3: 2L Residual Bleeding Engine: {val_bleeding:.4f}")
print(f"{'='*75}\n")