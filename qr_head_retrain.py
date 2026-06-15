#!/usr/bin/env python3
"""
QR Refinement with Head Retraining
=====================================
The post-hoc QR correction failed because the output head was trained
on the student's orientation. M_delta rotated h away from what the
head was trained to decode.

FIX: After applying QR correction to h, retrain ONLY the head.
  Blocks frozen. Head learns to decode the QR-refined representation.
  
PROTOCOL:
  For N in {0,1,2,4,8,12,22}:
    1. Apply N QR steps to M_stu → M_ref
    2. Freeze student blocks
    3. Replace head with fresh linear layer
    4. Train head for 100 steps on h_corrected = h_stu @ M_ref @ U^T
    5. Evaluate val
    
  If val(N) decreases with N: QR steps ARE spectral pages.
  If val(N) stays flat: the orientation is irrelevant, capacity is the limit.
"""
import json, math, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_LAYERS=24; BATCH=8; SEQ=64; LR=3e-4
L_ATT=14; PROJ=48; HEAD_STEPS=100

print(f"\n{'='*65}")
print(f"  QR REFINEMENT + HEAD RETRAINING")
print(f"  Freeze blocks, retrain head on QR-corrected hidden states")
print(f"{'='*65}\n")

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
        return self.ln_f(h)  # [B,S,D]

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

# ── Train teacher ─────────────────────────────────────────────────────────────
print("Step 1: Train 24L teacher (300 steps)...")
torch.manual_seed(42)
teacher=LM(D,N_HEADS,N_LAYERS)
opt=torch.optim.AdamW(teacher.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
t0=time.time()
for step in range(1,301):
    for pg in opt.param_groups: pg['lr']=clr(step,300,100)
    teacher.train(); x,y=get_batch(); _,loss=teacher(x,y)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(teacher.parameters(),1.0); opt.step()
    if step%100==0:
        teacher.eval()
        with torch.no_grad():
            vl=float(np.mean([teacher(*get_batch('val'))[1].item() for _ in range(10)]))
        print(f"  step {step}  val={vl:.4f}  t={time.time()-t0:.0f}s")
        teacher.train()
teacher.eval()
with torch.no_grad():
    val_t=float(np.mean([teacher(*get_batch('val'))[1].item() for _ in range(60)]))
print(f"  Teacher val={val_t:.4f}\n")

# ── Train 2L student ──────────────────────────────────────────────────────────
print("Step 2: Train 2L student with teacher embeddings (200 steps)...")
torch.manual_seed(99)
student=LM(D,N_HEADS,2)
for attr in ['te','pe','ln_f']:
    src=getattr(teacher,attr); dst=getattr(student,attr)
    if hasattr(src,'weight'): dst.weight.data.copy_(src.weight.data)
    if hasattr(src,'bias') and src.bias is not None:
        dst.bias.data.copy_(src.bias.data)

opt_s=torch.optim.AdamW(student.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
t0=time.time()
for step in range(1,201):
    for pg in opt_s.param_groups: pg['lr']=clr(step,200,100)
    student.train(); x,y=get_batch(); _,loss=student(x,y)
    opt_s.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(student.parameters(),1.0); opt_s.step()
    if step%50==0:
        student.eval()
        with torch.no_grad():
            vl=float(np.mean([student(*get_batch('val'))[1].item() for _ in range(20)]))
        print(f"  step {step}  val={vl:.4f}")
        student.train()
student.eval()
with torch.no_grad():
    val_s=float(np.mean([student(*get_batch('val'))[1].item() for _ in range(60)]))
print(f"  Student val={val_s:.4f}\n")

# ── Extract student monodromy ─────────────────────────────────────────────────
print("Step 3: Extract M_stu and projection basis U0...")
x_ref,_=get_batch('val'); x_ref=x_ref[0:1]
with torch.no_grad():
    hs=student.hidden_states(x_ref); hs=[h[0] for h in hs]
pos=SEQ//2; m=min(PROJ,SEQ,D)

Js=[]; U0=None; ma=None
for l in range(2):
    J,U,m_=layer_jac(student.blocks[l],hs[l],pos,m)
    Js.append(J)
    if U0 is None: U0=U; ma=m_

M_stu=np.eye(ma)
for J in reversed(Js): M_stu=J@M_stu
sv=np.linalg.svd(M_stu,compute_uv=False)
print(f"  M_stu sv={sv[:4].round(3)}\n")

U0_t=torch.tensor(U0,dtype=torch.float32)  # [D, ma]

# ── Head retraining function ──────────────────────────────────────────────────
def retrain_head(student, M_corr, U0_t, steps=HEAD_STEPS):
    """
    Freeze all student params. Add a fresh linear head.
    Train head on h_corrected = ln_f(h_blocks) @ U0 @ M_corr @ U0^T
    plus the orthogonal complement (unchanged).
    Returns val loss after retraining.
    """
    # Fresh head (untied from embeddings for this experiment)
    head_fresh=nn.Linear(D,VOCAB,bias=False)
    # Init from teacher embedding (same semantic space)
    head_fresh.weight.data.copy_(teacher.te.weight.data)

    M_t=torch.tensor(M_corr,dtype=torch.float32)  # [ma, ma]

    # Freeze student
    for p in student.parameters(): p.requires_grad_(False)
    head_fresh.weight.requires_grad_(True)

    opt_h=torch.optim.AdamW([head_fresh.weight],lr=LR*3,
                              betas=(0.9,0.95),weight_decay=0.01)

    def forward_corrected(x,y):
        with torch.no_grad():
            h=student.get_hidden(x)          # [B,S,D]
            B_,S_,D_=h.shape
            h_flat=h.reshape(-1,D_)
            # Project → apply M_corr → lift back
            h_proj=h_flat@U0_t               # [B*S, ma]
            h_ref=h_proj@M_t                 # [B*S, ma]
            h_lift=h_ref@U0_t.T              # [B*S, D]
            # Keep orthogonal component unchanged
            h_orth=h_flat - h_flat@U0_t@U0_t.T
            h_out=(h_lift+h_orth).reshape(B_,S_,D_)
        logits=head_fresh(h_out)
        loss=F.cross_entropy(logits.reshape(-1,VOCAB),y.reshape(-1))
        return loss

    for step in range(1,steps+1):
        student.eval()
        x,y=get_batch()
        loss=forward_corrected(x,y)
        opt_h.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_([head_fresh.weight],1.0)
        opt_h.step()

    # Evaluate
    student.eval(); ls=[]
    with torch.no_grad():
        for _ in range(60):
            x,y=get_batch('val')
            h=student.get_hidden(x)
            B_,S_,D_=h.shape; h_flat=h.reshape(-1,D_)
            h_proj=h_flat@U0_t; h_ref=h_proj@M_t; h_lift=h_ref@U0_t.T
            h_orth=h_flat-h_flat@U0_t@U0_t.T
            h_out=(h_lift+h_orth).reshape(B_,S_,D_)
            logits=head_fresh(h_out)
            loss=F.cross_entropy(logits.reshape(-1,VOCAB),y.reshape(-1))
            ls.append(loss.item())

    # Unfreeze student
    for p in student.parameters(): p.requires_grad_(True)
    return float(np.mean(ls))

# ── Sweep QR steps ────────────────────────────────────────────────────────────
print(f"Step 4: QR refinement sweep with head retraining ({HEAD_STEPS} steps each)...")
print(f"  Teacher val={val_t:.4f}  Student val={val_s:.4f}\n")

QR_STEPS=[0,1,2,4,8,12,22]
results=[]

for N in QR_STEPS:
    t0=time.time()
    M_ref=qr_refine(M_stu,N)
    vl=retrain_head(student,M_ref,U0_t,steps=HEAD_STEPS)
    results.append((N,vl))
    sv_r=np.linalg.svd(M_ref,compute_uv=False)
    # Measure how far M_ref is from M_stu (should grow with N → triangular)
    off_diag=np.linalg.norm(np.tril(M_ref,-1))/np.linalg.norm(M_ref)
    print(f"  N={N:>2} QR: val={vl:.4f}  "
          f"off_diag={off_diag:.4f}  "
          f"sv1={sv_r[0]:.3f}  t={time.time()-t0:.0f}s")

# Also test: apply teacher monodromy directly
print(f"\n  Extracting teacher M_fwd for comparison...")
with torch.no_grad():
    hs_t=teacher.hidden_states(x_ref); hs_t=[h[0] for h in hs_t]
Js_t=[]
for l in range(L_ATT+1):
    J,_,_=layer_jac(teacher.blocks[l],hs_t[l],pos,m)
    Js_t.append(J)
    if (l+1)%8==0: print(f"    L{l+1}...",flush=True)
M_teach=np.eye(ma)
for J in reversed(Js_t): M_teach=J@M_teach

t0=time.time()
vl_teach=retrain_head(student,M_teach,U0_t,steps=HEAD_STEPS)
results.append(('teacher',vl_teach))
print(f"  Teacher M_fwd: val={vl_teach:.4f}  t={time.time()-t0:.0f}s")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  RESULTS: QR REFINEMENT + HEAD RETRAINING")
print("="*65)
print(f"\n  Teacher (24L):              val={val_t:.4f}")
print(f"  Student (2L, original head): val={val_s:.4f}")
print(f"\n  After {HEAD_STEPS}-step head retrain:")
print(f"  {'N QR steps':>12}  {'val':>8}  {'vs N=0':>10}")
print("  "+"-"*34)
base_val=results[0][1]
for row in results:
    N,vl=row
    delta=vl-base_val
    print(f"  {str(N):>12}  {vl:>8.4f}  {delta:>+10.4f}")

best=min((r for r in results if isinstance(r[0],int)),key=lambda x:x[1])
print(f"""
  Best: N={best[0]} QR steps → val={best[1]:.4f}
  Gap from teacher: {best[1]-val_t:.4f} nats
  Gap from student: {val_s-best[1]:.4f} nats closed

  KEY READING:
  If val DECREASES with N (even slightly):
    QR steps are adding spectral refinement to the representation.
    The crystallization IS happening through QR iteration.
    Each step = one spectral sequence page, at O(m^2) cost.
    
  If val is FLAT across N:
    The QR steps rotate the subspace isospectally without
    adding new structural information. The representation
    quality is determined by the eigenvalues (universal)
    not the eigenvectors (seed-dependent).
    Depth is the only path to closing the gap.
    
  If teacher M_fwd < N=0:
    The teacher's monodromy carries genuine extra information
    that the student's monodromy lacks. This is the orientation
    gap — the teacher and student find different attractors.
""")
