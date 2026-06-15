#!/usr/bin/env python3
"""
GRADIENT-DRIVEN MANIFOLD REALIZATION (OPTION 2)
==============================================
Abandons manual matrix tampering. Uses an extended optimization horizon 
to let the data gradient naturally evolve the empirical activation spectrum, 
allowing the 2L model to find its true structural scale.

Mathematical Formulation:
  \Delta W = -\eta \nabla_W \mathcal{L}_{CE}
   Evolved Cov(H) -> Aligns to data topology organically via gradient descent.
"""
import json, math, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_LAYERS_TEACHER=24; BATCH=8; SEQ=64; LR=3e-4
ALPHA=0.20       # Maintain the high-rank residual bleeding bypass
EXTENDED_STEPS=2000

print(f"\n{'='*75}")
print(f"  DEEP GRADIENT-DRIVEN REALIZATION ENGINE")
print(f"  Extended Horizon Optimization Sweep (Steps: {EXTENDED_STEPS}, \u03b1 = {ALPHA})")
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
        h_bleed = ALPHA * h + (1.0 - ALPHA) * h_emb
        logits = self.head(self.ln_f(h_bleed))
        return logits,(F.cross_entropy(logits.view(-1,VOCAB),y.view(-1)) if y is not None else None)
        
    def get_hidden(self,x):
        h_emb = self.te(x) + self.pe(torch.arange(x.shape[1]))
        h = h_emb
        for b in self.blocks: h = b(h)
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

# ── Step 2: Initialize Student with Transferred Teacher Seeding ───────────────
print("\nStep 2: Initializing 2L student with mature teacher matrices...")
torch.manual_seed(99)
student2 = LM(D, N_HEADS, 2)

for attr in ['te', 'pe', 'ln_f']:
    src = getattr(teacher, attr); dst = getattr(student2, attr)
    if hasattr(src, 'weight'): dst.weight.data.copy_(src.weight.data)
    if hasattr(src, 'bias') and src.bias is not None: dst.bias.data.copy_(src.bias.data)

# Measure baseline activation spectrum at initialization (Step 201 baseline)
with torch.no_grad():
    x_val, _ = get_batch('val')
    h_init = student2.get_hidden(x_val).reshape(-1, D)
    cov_init = (h_init.T @ h_init) / h_init.shape[0]
    sv_init = torch.linalg.svdvals(cov_init).cpu().numpy()
print(f"  Initial Activation Spectrum (Top 4 \u039b): {sv_init[:4].round(4)}")

# ── Step 3: Run Extended Horizon Gradient Training ───────────────────────────
print(f"\nStep 3: Optimizing student over extended horizon ({EXTENDED_STEPS} steps)...")
opt_s = torch.optim.AdamW(student2.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1)

t0 = time.time()
for step in range(1, EXTENDED_STEPS + 1):
    student2.train()
    x, y = get_batch()
    _, loss = student2(x, y)
    opt_s.zero_grad()
    loss.backward()
    opt_s.step()
    
    if step % 500 == 0 or step == 100:
        print(f"  Step {step:4d}/{EXTENDED_STEPS} | Training Loss: {loss.item():.4f}")
print(f"  Optimization complete in {time.time() - t0:.2f}s.")
student2.eval()

# ── Step 4: Extract Evolved Manifold Characteristics ─────────────────────────
print("\nStep 4: Extracting evolved activation manifold profile...")
with torch.no_grad():
    h_final = student2.get_hidden(x_val).reshape(-1, D)
    cov_final = (h_final.T @ h_final) / h_final.shape[0]
    sv_final = torch.linalg.svdvals(cov_final).cpu().numpy()
print(f"  Evolved Activation Spectrum (Top 4 \u039b): {sv_final[:4].round(4)}")

# ── Step 5: Evaluate Final Realized System Losses ────────────────────────────
print("\nStep 5: Evaluating final generative loss metrics...")
student2.eval()
val_losses = []
with torch.no_grad():
    for _ in range(50):
        x, y = get_batch('val')
        _, loss = student2(x, y)
        val_losses.append(loss.item())
final_val_loss = float(np.mean(val_losses))

print(f"\n{'='*75}")
print(f"  DEEP GRADIENT REALIZATION RESULTS SUMMARY")
print(f"{'='*75}")
print(f"  Final Evolved 2L Validation Loss: {final_val_loss:.4f}")
print(f"  Spectrum Trajectory: {sv_init[0]:.2f} (Init) ──► {sv_final[0]:.2f} (Evolved)")
print(f"{'='*75}\n")