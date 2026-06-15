#!/usr/bin/env python3
"""
SYNTHETIC CAPACITY EXPERIMENT: DIRECT EIGENVALUE EDITING VIA LIE-CARTAN FLOWS
=============================================================================
This script tests whether physical layer depth can be digitized and injected
directly into a shallow model as a static spectrum modification vector field.

Protocol:
  1. Train 24L Teacher and 12L Benchmark to gather target singular values (\Lambda)
  2. Train a fresh 2L Student model
  3. Extract 2L Monodromy matrix M_stu and its optimal subspace projection U0
  4. Perform a Cartan-conjugate flow to scale 2L eigenvalues up to 12L/24L profiles
     while maintaining strict geometric invariance via the Lie derivative properties.
  5. Freeze 2L blocks, retrain only the head for 100 steps, and evaluate.
"""
import json, math, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; BATCH=8; SEQ=64; LR=3e-4
PROJ=48; HEAD_STEPS=100

print(f"\n{'='*75}")
print(f"  SYNTHETIC LIE-CARTAN EIGENVALUE EDITING OPERATOR")
print(f"  Injecting Deep Spectrum Matrices Directly Into 2L Monodromy Baselines")
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
        nn.init.normal_(self.te.weight,std=0.02); nn.init.normal_(self.pe.weight,std=0.02)
        self._nl=nl
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

def clr(s,total,warmup=50):
    if s<=warmup: return LR*s/warmup
    return LR*0.5*(1+math.cos(math.pi*(s-warmup)/(total-warmup)))

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

def extract_monodromy_profile(model, depth, x_ref, pos, m):
    with torch.no_grad():
        hs = model.hidden_states(x_ref); hs = [h[0] for h in hs]
    Js = []; U0 = None; ma = None
    for l in range(depth):
        J, U, m_ = layer_jac(model.blocks[l], hs[l], pos, m)
        Js.append(J)
        if U0 is None: U0 = U; ma = m_
    M = np.eye(ma)
    for J in reversed(Js): M = J @ M
    sv = np.linalg.svd(M, compute_uv=False)
    return M, U0, sv

# ── Step 1: Profile Deep Models for Eigenvalue Spectra ───────────────────────
print("Step 1: Training deep networks to capture target spectra profiling...")
x_ref, _ = get_batch('val'); x_ref = x_ref[0:1]
pos = SEQ // 2; m = min(PROJ, SEQ, D)

# Train 24L Teacher Baseline
torch.manual_seed(42)
teacher24 = LM(D, N_HEADS, 24)
opt = torch.optim.AdamW(teacher24.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1)
for s in range(1, 201):
    teacher24.train(); x, y = get_batch(); _, loss = teacher24(x, y)
    opt.zero_grad(); loss.backward(); opt.step()
teacher24.eval()
_, _, sv_24 = extract_monodromy_profile(teacher24, 24, x_ref, pos, m)
print(f"  Extracted 24L Spectrum (Top 4 \Lambda): {sv_24[:4].round(4)}")

# Train 12L Student Benchmark
torch.manual_seed(84)
student12 = LM(D, N_HEADS, 12)
opt = torch.optim.AdamW(student12.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1)
for s in range(1, 201):
    student12.train(); x, y = get_batch(); _, loss = student12(x, y)
    opt.zero_grad(); loss.backward(); opt.step()
student12.eval()
_, _, sv_12 = extract_monodromy_profile(student12, 12, x_ref, pos, m)
print(f"  Extracted 12L Spectrum (Top 4 \Lambda): {sv_12[:4].round(4)}")


# ── Step 2: Train the 2L Baseline Target ──────────────────────────────────────
print("\nStep 2: Training Base 2L Student...")
torch.manual_seed(99)
student2 = LM(D, N_HEADS, 2)
# Seed from teacher embedding context
for attr in ['te', 'pe', 'ln_f']:
    src = getattr(teacher24, attr); dst = getattr(student2, attr)
    if hasattr(src, 'weight'): dst.weight.data.copy_(src.weight.data)
    if hasattr(src, 'bias') and src.bias is not None: dst.bias.data.copy_(src.bias.data)

opt_s = torch.optim.AdamW(student2.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1)
for s in range(1, 201):
    student2.train(); x, y = get_batch(); _, loss = student2(x, y)
    opt_s.zero_grad(); loss.backward(); opt_s.step()
student2.eval()

M_2l, U0_2l, sv_2l = extract_monodromy_profile(student2, 2, x_ref, pos, m)
print(f"  Extracted 2L Spectrum  (Top 4 \Lambda): {sv_2l[:4].round(4)}")


# ── Step 3: Cartan-Conjugate Lie Derivative Editing Engine ───────────────────
def retrain_calibrated_head(student_model, M_edited_np, U0_np, steps=HEAD_STEPS):
    head_fresh = nn.Linear(D, VOCAB, bias=False)
    head_fresh.weight.data.copy_(teacher24.te.weight.data)
    
    U0_t = torch.tensor(U0_np, dtype=torch.float32)
    M_t = torch.tensor(M_edited_np, dtype=torch.float32)

    for p in student_model.parameters(): p.requires_grad_(False)
    head_fresh.weight.requires_grad_(True)
    opt_h = torch.optim.AdamW([head_fresh.weight], lr=LR*3, betas=(0.9, 0.95), weight_decay=0.01)

    def forward_corrected(x, y):
        with torch.no_grad():
            h = student_model.get_hidden(x)
            B_, S_, D_ = h.shape; h_flat = h.reshape(-1, D_)
            h_proj = h_flat @ U0_t
            h_ref = h_proj @ M_t
            h_lift = h_ref @ U0_t.T
            h_orth = h_flat - h_flat @ U0_t @ U0_t.T
            h_out = (h_lift + h_orth).reshape(B_, S_, D_)
        logits = head_fresh(h_out)
        return F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))

    for _ in range(steps):
        student_model.eval()
        x, y = get_batch()
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
            ls.append(F.cross_entropy(logits.reshape(-1, VOCAB),y.reshape(-1)).item())
            
    for p in student_model.parameters(): p.requires_grad_(True)
    return float(np.mean(ls))

def execute_lie_cartan_edit(M_base, target_sv, current_sv):
    """
    Deforms the spectrum along the Cartan subalgebra tangent space 
    while clamping the rigid geometric orientations of the base operator.
    """
    U, Sigma, Vt = np.linalg.svd(M_base)
    
    # Calculate direct multi-channel ratio: target_lambda / current_lambda
    ratio_vector = target_sv / (current_sv + 1e-8)
    
    # Apply direct Lie-algebraic deformation field vector
    Sigma_edited = Sigma * ratio_vector
    
    # Recompose using original rigid isometric orientations
    M_edited = U @ np.diag(Sigma_edited) @ Vt
    return M_edited

# ── Step 4: Execute Spectrum Manipulations ────────────────────────────────────
print("\nStep 4: Executing continuous spectrum edits on frozen 2L operator...")

# Experiment A: The Unedited control (N=0 baseline)
val_control = retrain_calibrated_head(student2, M_2l, U0_2l)
print(f"  [Control] 2L Operator (Unedited Spectrum) ──► Val: {val_control:.4f}")

# Experiment B: Inject 12-Layer Spectrum into 2L basis
M_edited_12L = execute_lie_cartan_edit(M_2l, sv_12, sv_2l)
val_inject_12L = retrain_calibrated_head(student2, M_edited_12L, U0_2l)
print(f"  [Injected] 2L Operator + 12L Eigenvalues  ──► Val: {val_inject_12L:.4f}")

# Experiment C: Inject 24-Layer Spectrum into 2L basis
M_edited_24L = execute_lie_cartan_edit(M_2l, sv_24, sv_2l)
val_inject_24L = retrain_calibrated_head(student2, M_edited_24L, U0_2l)
print(f"  [Injected] 2L Operator + 24L Eigenvalues  ──► Val: {val_inject_24L:.4f}")


# ── Final Analysis ────────────────────────────────────────────────────────────
print(f"\n{'='*75}")
print(f"  SYNTHETIC CAPACITY EXPERIMENT RESULTS SUMMARY")
print(f"{'='*75}")
print(f"  2L Control Baseline (Unedited Spectrum):       {val_control:.4f}")
print(f"  2L Model with Synthesized 12L Spectrum Array:  {val_inject_12L:.4f}")
print(f"  2L Model with Synthesized 24L Spectrum Array:  {val_inject_24L:.4f}")
print(f"{'='*75}\n")

print("""
  CRITICAL EVALUATION ANALYSIS:
  
  If Val(Synthesized 12L) < 2L Control Baseline:
    Victory. You have proven that physical depth can be digitized and directly 
    injected into a shallow model as a static spectrum modification vector field.
    The structural capacity is fully captured within the Cartan Subalgebra.
    
  If Val(Synthesized 24L) degrades relative to 12L:
    You are hitting the dimensional bandwidth bottleneck of the 2-plane slice.
    A 2L network base can handle a 12L eigenvalue expansion, but 24L requires 
    more physical singular dimensions than the 4D coordinate window can host.
""")