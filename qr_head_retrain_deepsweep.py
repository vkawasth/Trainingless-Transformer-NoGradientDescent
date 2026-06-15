#!/usr/bin/env python3
"""
MONODROMY DEPTH SWEEP WITH HEAD RETRAINING
==========================================
Evaluates the true, unmasked representation capacity of compressed students 
across depths: 2L, 4L, 6L, 8L, and 12L blocks.

For each depth configuration:
  1. Trains the student blocks with teacher embeddings (200 CE steps)
  2. Extracts the depth-specific student monodromy operator M_stu
  3. Sweeps QR refinement steps N in {0, 4, 12, 22}
  4. Freezes student blocks, swaps a clean linear head, and runs a 100-step 
     calibration pass to let the head adapt to the canonical QR frame.

This decouples structural/eigenvalue capacity from coordinate gauge errors, 
mapping the true geometric crystallization curve against Bott periodicity boundaries.
"""
import json, math, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_LAYERS_TEACHER=24; BATCH=8; SEQ=64; LR=3e-4
L_ATT=14; PROJ=48; HEAD_STEPS=100

print(f"\n{'='*75}")
print(f"  AUTOMATED MONODROMY DEPTH SWEEP + HOMOTOPY REFINEMENT")
print(f"  Mapping Bott Periodicity and Capacity vs. Orientation Gates")
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

def qr_refine(M,steps):
    M_r=M.copy()
    for _ in range(steps):
        Q,R=np.linalg.qr(M_r); M_r=R@Q
    return M_r

# ── Step 1: Baseline 24L Teacher Training ─────────────────────────────────────
print("Step 1: Training 24L Teacher Core Attractor Workspace...")
torch.manual_seed(42)
teacher=LM(D,N_HEADS,N_LAYERS_TEACHER)
opt=torch.optim.AdamW(teacher.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
for step in range(1,301):
    for pg in opt.param_groups: pg['lr']=clr(step,300,100)
    teacher.train(); x,y=get_batch(); _,loss=teacher(x,y)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(teacher.parameters(),1.0); opt.step()
teacher.eval()
with torch.no_grad():
    teacher_final_val=float(np.mean([teacher(*get_batch('val'))[1].item() for _ in range(60)]))
print(f"  Teacher baseline structural lock: val={teacher_final_val:.4f}\n")

# ── Calibration Head-Retraining Architecture ──────────────────────────────────
def retrain_head(student_model, M_corr, U0_t, steps=HEAD_STEPS):
    head_fresh=nn.Linear(D,VOCAB,bias=False)
    head_fresh.weight.data.copy_(teacher.te.weight.data)
    M_t=torch.tensor(M_corr,dtype=torch.float32)

    for p in student_model.parameters(): p.requires_grad_(False)
    head_fresh.weight.requires_grad_(True)
    opt_h=torch.optim.AdamW([head_fresh.weight],lr=LR*3,betas=(0.9,0.95),weight_decay=0.01)

    def forward_corrected(x,y):
        with torch.no_grad():
            h=student_model.get_hidden(x)
            B_,S_,D_=h.shape; h_flat=h.reshape(-1,D_)
            h_proj=h_flat@U0_t
            h_ref=h_proj@M_t
            h_lift=h_ref@U0_t.T
            h_orth=h_flat - h_flat@U0_t@U0_t.T
            h_out=(h_lift+h_orth).reshape(B_,S_,D_)
        logits=head_fresh(h_out)
        return F.cross_entropy(logits.reshape(-1,VOCAB),y.reshape(-1))

    for _ in range(steps):
        student_model.eval()
        x,y=get_batch()
        loss=forward_corrected(x,y)
        opt_h.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_([head_fresh.weight],1.0)
        opt_h.step()

    student_model.eval(); ls=[]
    with torch.no_grad():
        for _ in range(40):
            x,y=get_batch('val')
            h=student_model.get_hidden(x)
            B_,S_,D_=h.shape; h_flat=h.reshape(-1,D_)
            h_proj=h_flat@U0_t; h_ref=h_proj@M_t; h_lift=h_ref@U0_t.T
            h_orth=h_flat-h_flat@U0_t@U0_t.T
            h_out=(h_lift+h_orth).reshape(B_,S_,D_)
            logits=head_fresh(h_out)
            ls.append(F.cross_entropy(logits.reshape(-1,VOCAB),y.reshape(-1)).item())

    for p in student_model.parameters(): p.requires_grad_(True)
    return float(np.mean(ls))

# ── Step 2 & 3: Depth Sweep Execution Loop ───────────────────────────────────
DEPTH_SWEEP_CONFIGS = [2, 4, 6, 8, 12]
QR_SWEEP_STEPS = [0, 4, 12, 22]

# Structure storage matrix for the final sweep analysis
sweep_matrix = {d: {} for d in DEPTH_SWEEP_CONFIGS}
raw_student_baselines = {}

print(f"Step 2: Launching Layer Depth Sweep Array...")
for depth in DEPTH_SWEEP_CONFIGS:
    print(f"\n{"-"*50}\n  EVALUATING ARCHITECTURAL DEPTH: {depth} LAYERS\n{"-"*50}")
    torch.manual_seed(99 + depth)
    student = LM(D, N_HEADS, depth)
    
    # Direct alignment of anchor embedding matrices
    for attr in ['te', 'pe', 'ln_f']:
        src = getattr(teacher, attr); dst = getattr(student, attr)
        if hasattr(src, 'weight'): dst.weight.data.copy_(src.weight.data)
        if hasattr(src, 'bias') and src.bias is not None: dst.bias.data.copy_(src.bias.data)

    # Core Block Optimization Pass
    opt_s = torch.optim.AdamW(student.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1)
    for step in range(1, 201):
        for pg in opt_s.param_groups: pg['lr'] = clr(step, 200, 100)
        student.train(); x,y = get_batch(); _, loss = student(x,y)
        opt_s.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0); opt_s.step()
        
    student.eval()
    with torch.no_grad():
        val_raw_student = float(np.mean([student(*get_batch('val'))[1].item() for _ in range(40)]))
    raw_student_baselines[depth] = val_raw_student
    print(f"  Unrefined student block baseline: val={val_raw_student:.4f}")

    # Extract Monodromy Operator for this specific depth geometry
    x_ref, _ = get_batch('val'); x_ref = x_ref[0:1]
    with torch.no_grad():
        hs = student.hidden_states(x_ref); hs = [h[0] for h in hs]
    pos = SEQ // 2; m = min(PROJ, SEQ, D)

    Js = []; U0 = None; ma = None
    for l in range(depth):
        J, U, m_ = layer_jac(student.blocks[l], hs[l], pos, m)
        Js.append(J)
        if U0 is None: U0 = U; ma = m_

    M_stu = np.eye(ma)
    for J in reversed(Js): M_stu = J @ M_stu
    U0_t = torch.tensor(U0, dtype=torch.float32)

    # Sweep Discrete Toda Refinement Pages within this Depth Block
    for N in QR_SWEEP_STEPS:
        M_ref = qr_refine(M_stu, N)
        val_refined = retrain_head(student, M_ref, U0_t, steps=HEAD_STEPS)
        sweep_matrix[depth][N] = val_refined
        print(f"    Refinement N={N:>2} QR Steps ──► Calibrated Val Loss: {val_refined:.4f}")

# ── Step 4: Summary Analysis Report ───────────────────────────────────────────
print(f"\n\n{'='*75}")
print(f"  FINAL SUMMARY DEEP DEPTH SWEEP MATRIX")
print(f"  Teacher Baseline (24L): {teacher_final_val:.4f}")
print(f"{'='*75}")
print(f"  {'Depth':<6} | {'Raw Stud':<10} | " + " | ".join([f"N={n:<4}" for n in QR_SWEEP_STEPS]))
print(f"  {'-'*6}─┼─{'-'*10}─┼─" + "─┼─".join([f"{'-'*6}" for _ in QR_SWEEP_STEPS]))

for depth in DEPTH_SWEEP_CONFIGS:
    row_str = f"  {depth:<6} | {raw_student_baselines[depth]:<10.4f} | "
    row_str += " | ".join([f"{sweep_matrix[depth][n]:<.4f}" for n in QR_SWEEP_STEPS])
    print(row_str)
print(f"{'='*75}\n")

print("""
  THEORETICAL EVALUATION PROTOCOL:
  
  1. THE INTER-LAYER CAPACITY GAIN (Vertical Analysis):
     Look down the column for N=12. If val drops sharply between 4L and 8L, 
     then physical depth introduces higher-order topological invariants 
     (e.g. Bott Periodicity boundaries) that cannot be simulated entirely 
     via flat 2D matrix transformation.
     
  2. THE EQUIVALENCE THEOREM (Horizontal Analysis):
     Compare a shallow model's refined score (e.g. 2L with N=12) to a raw deep 
     model's baseline (e.g. 8L Raw Stud). If the refined shallow model matches 
     or beats the raw deeper model, the discrete Toda flow functions as a 
     direct mathematical replacement for physical layer allocation.
""")