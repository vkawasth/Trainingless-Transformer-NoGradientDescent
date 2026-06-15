#!/usr/bin/env python3
"""
Monodromy Student: Hidden States at 13x Lower Cost
====================================================
The operator distillation confirmed cos=0.984 on hidden states (13x compression).
The missing piece: can a 2-layer student trained on the monodromy transport
achieve comparable LANGUAGE MODELING quality, not just geometric similarity?

TWO-STAGE PIPELINE:
  Stage 1: Train 24-layer teacher (standard SGD)
  Stage 2: Train 2-layer student with joint loss:
    L = λ * L_CE(logits, y)          (language modeling quality)
        + (1-λ) * L_mono(h_stu, M*)  (monodromy transport matching)
  
  L_mono = ||h_stu @ h_stu^T - M_fwd||_F  (match the forward monodromy)
  
  The monodromy M_fwd = J_14 @ ... @ J_0 encodes the teacher's
  hidden state transformation in closed form.
  
  λ anneals from 0→1: start with pure monodromy alignment,
  finish with pure language modeling. The geometric target
  pulls the student into the right subspace early,
  then the CE loss refines token predictions.

Why this differs from failed hidden-state matching:
  - Previous: match h_24 directly (teacher's specific orientation)
  - Now: match M_fwd (the transport operator, seed-independent)
  - The student finds its own orientation while executing the same transport
"""
import json, math, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from scipy.linalg import sqrtm as scipy_sqrtm

D=256; N_HEADS=4; N_LAYERS=24; BATCH=8; SEQ=64; LR=3e-4
L_ATT=14; PROJ=48

print(f"\n{'='*65}")
print(f"  MONODROMY STUDENT: HIDDEN STATES AT 13x LOWER COST")
print(f"  2-layer student trained on monodromy transport operator")
print(f"  d={D}  teacher={N_LAYERS}L  student=2L")
print(f"{'='*65}\n")

with open('/tmp/train_ids.json') as f: train_ids=json.load(f)
with open('/tmp/val_ids.json')   as f: val_ids=json.load(f)
with open('/tmp/vocab.json')     as f: vocab=json.load(f)
VOCAB=len(vocab)
train_t=torch.tensor(train_ids,dtype=torch.long)
val_t  =torch.tensor(val_ids,  dtype=torch.long)

def get_batch(split='train'):
    data=train_t if split=='train' else val_t
    ix=torch.randint(0,len(data)-SEQ-1,(BATCH,))
    return (torch.stack([data[i:i+SEQ]   for i in ix]),
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
    def hidden_out(self,x):
        h=self.te(x)+self.pe(torch.arange(x.shape[1]))
        for b in self.blocks: h=b(h)
        return self.ln_f(h)   # [B,S,D]
    def hidden_states(self,x):
        hs=[]; h=self.te(x)+self.pe(torch.arange(x.shape[1])); hs.append(h.detach())
        for b in self.blocks: h=b(h); hs.append(h.detach())
        return hs

def eval_val(model,n=60):
    model.eval(); ls=[]
    with torch.no_grad():
        for _ in range(n):
            x,y=get_batch('val'); _,l=model(x,y); ls.append(l.item())
    return float(np.mean(ls))

def cosine_sim(mA,mB,n=30):
    mA.eval(); mB.eval(); sims=[]
    with torch.no_grad():
        for _ in range(n):
            x,_=get_batch('val')
            hA=mA.hidden_out(x); hB=mB.hidden_out(x)
            hA=hA.reshape(-1,D); hB=hB.reshape(-1,D)
            sims.append(F.cosine_similarity(hA,hB,dim=-1).mean().item())
    return float(np.mean(sims))

def clr(s,total,warmup=100):
    if s<=warmup: return LR*s/warmup
    return LR*0.5*(1+math.cos(math.pi*(s-warmup)/(total-warmup)))

# ── Stage 1: Train 24-layer teacher ──────────────────────────────────────────
print("Stage 1: Train 24-layer teacher (300 steps)...")
torch.manual_seed(42)
teacher=LM(D,N_HEADS,N_LAYERS)
opt=torch.optim.AdamW(teacher.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
t0=time.time()
for step in range(1,301):
    for pg in opt.param_groups: pg['lr']=clr(step,300)
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
val_teacher=eval_val(teacher)
t_teacher=time.time()-t0
print(f"  Teacher: val={val_teacher:.4f}  params={sum(p.numel() for p in teacher.parameters()):,}  t={t_teacher:.0f}s\n")

# ── Extract monodromy M_fwd (approach to L14) ────────────────────────────────
print(f"Extracting monodromy M_fwd (L0→L{L_ATT})...")
x_ref,_=get_batch('val'); x_ref=x_ref[0:1]
with torch.no_grad():
    hs_b=teacher.hidden_states(x_ref); hs=[h[0] for h in hs_b]
pos=SEQ//2; m=min(PROJ,SEQ,D)

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

print("  Computing Jacobians...",flush=True)
Js=[]; U0=None; ma=None
for l in range(N_LAYERS):
    J,U,m_=layer_jac(teacher.blocks[l],hs[l],pos,m)
    Js.append(J)
    if U0 is None: U0=U; ma=m_
    if (l+1)%8==0: print(f"    L{l+1}...",flush=True)

M_fwd=np.eye(ma)
for l in range(L_ATT+1): M_fwd=Js[l]@M_fwd
sqM=np.real(scipy_sqrtm(M_fwd))
err=np.linalg.norm(sqM@sqM-M_fwd)/max(np.linalg.norm(M_fwd),1e-8)
sv=np.linalg.svd(M_fwd,compute_uv=False)
print(f"  M_fwd: sv={sv[:4].round(3)}  sqrtm_err={err:.4f}")

# Lift sqrtm(M_fwd) to d-space as a target transport operator
M_fwd_t=torch.tensor(M_fwd,dtype=torch.float32)   # [ma,ma] in projected space
U0_t=torch.tensor(U0,dtype=torch.float32)           # [D,ma]

def monodromy_loss(h_stu, M_fwd_t, U0_t):
    """
    Match student hidden state transport to teacher monodromy.
    h_stu: [B,S,D] student final hidden state
    Compute: h_stu_proj = h_stu @ U0  [B,S,ma]
    Target:  h_stu_proj @ h_stu_proj^T ≈ M_fwd  (the forward transport)
    Loss: ||mean_over_batch(h_proj h_proj^T) - M_fwd||_F^2
    """
    B,S,D_=h_stu.shape
    # Project to ma-dimensional subspace
    h_proj=h_stu.reshape(-1,D_)@U0_t  # [B*S, ma]
    # Empirical transport: h^T h / (B*S)
    T_stu=h_proj.T@h_proj / (B*S)     # [ma, ma]
    return ((T_stu-M_fwd_t)**2).mean()

# ── Stage 2a: Pure CE baseline (2-layer from scratch) ────────────────────────
print("\nStage 2a: 2-layer student — pure CE baseline (200 steps)...")
torch.manual_seed(99)
stu_ce=LM(D,N_HEADS,2)
# Copy embeddings from teacher (the semantics)
stu_ce.te.weight.data.copy_(teacher.te.weight.data)
stu_ce.pe.weight.data.copy_(teacher.pe.weight.data)
stu_ce.ln_f.weight.data.copy_(teacher.ln_f.weight.data)
stu_ce.ln_f.bias.data.copy_(teacher.ln_f.bias.data)

opt_ce=torch.optim.AdamW(stu_ce.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
t0=time.time()
for step in range(1,201):
    for pg in opt_ce.param_groups: pg['lr']=clr(step,200)
    stu_ce.train(); x,y=get_batch(); _,loss=stu_ce(x,y)
    opt_ce.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(stu_ce.parameters(),1.0); opt_ce.step()
    if step%50==0:
        vl=eval_val(stu_ce,n=20)
        print(f"  step {step}  val={vl:.4f}  t={time.time()-t0:.0f}s")
val_ce=eval_val(stu_ce); t_ce=time.time()-t0
cos_ce=cosine_sim(teacher,stu_ce)
print(f"  CE student: val={val_ce:.4f}  cos={cos_ce:.4f}  t={t_ce:.0f}s")

# ── Stage 2b: Monodromy-aligned student (λ annealing 0→1) ────────────────────
print("\nStage 2b: 2-layer student — monodromy alignment + CE (200 steps)...")
torch.manual_seed(99)
stu_mono=LM(D,N_HEADS,2)
# Copy embeddings from teacher
stu_mono.te.weight.data.copy_(teacher.te.weight.data)
stu_mono.pe.weight.data.copy_(teacher.pe.weight.data)
stu_mono.ln_f.weight.data.copy_(teacher.ln_f.weight.data)
stu_mono.ln_f.bias.data.copy_(teacher.ln_f.bias.data)

opt_mono=torch.optim.AdamW(stu_mono.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
t0=time.time()

def lam_schedule(step,total=200):
    """λ: CE weight. Anneals 0→1. Start pure monodromy, end pure CE."""
    return min(1.0, step/total)

for step in range(1,201):
    for pg in opt_mono.param_groups: pg['lr']=clr(step,200)
    lam=lam_schedule(step)
    stu_mono.train(); x,y=get_batch()
    logits,loss_ce=stu_mono(x,y)
    h_out=stu_mono.hidden_out(x)
    loss_m=monodromy_loss(h_out,M_fwd_t,U0_t)
    loss=lam*loss_ce+(1-lam)*loss_m*10  # scale monodromy loss
    opt_mono.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(stu_mono.parameters(),1.0); opt_mono.step()
    if step%50==0:
        vl=eval_val(stu_mono,n=20); vm=float(loss_m.item())
        print(f"  step {step}  val={vl:.4f}  L_mono={vm:.4f}  λ={lam:.2f}  t={time.time()-t0:.0f}s")
val_mono=eval_val(stu_mono); t_mono=time.time()-t0
cos_mono=cosine_sim(teacher,stu_mono)
print(f"  Mono student: val={val_mono:.4f}  cos={cos_mono:.4f}  t={t_mono:.0f}s")

# ── Stage 2c: Monodromy init + CE fine-tune ──────────────────────────────────
print("\nStage 2c: Monodromy weight-set init + CE fine-tune (200 steps)...")
torch.manual_seed(99)
stu_init=LM(D,N_HEADS,2)
# Set weights from sqrtm(M_fwd) — the operator distillation result
dJ=sqM-np.eye(ma)
W_op_d=U0@(SEQ*dJ)@U0.T+np.eye(D)-U0@U0.T
W_v_d=U0@U0.T+np.eye(D)-U0@U0.T
for blk in stu_init.blocks:
    with torch.no_grad():
        blk.attn.op.weight.copy_(torch.tensor(W_op_d,dtype=torch.float32))
        blk.attn.WV.weight.copy_(torch.tensor(W_v_d,dtype=torch.float32))
        blk.attn.WQ.weight.copy_(teacher.blocks[L_ATT].attn.WQ.weight)
        blk.attn.WK.weight.copy_(teacher.blocks[L_ATT].attn.WK.weight)
        blk.attn.ln.weight.copy_(teacher.blocks[L_ATT].attn.ln.weight)
        blk.attn.ln.bias.copy_(teacher.blocks[L_ATT].attn.ln.bias)
        blk.ff.g.weight.copy_(teacher.blocks[L_ATT].ff.g.weight)
        blk.ff.v.weight.copy_(teacher.blocks[L_ATT].ff.v.weight)
        blk.ff.o.weight.copy_(teacher.blocks[L_ATT].ff.o.weight)
        blk.ff.n.weight.copy_(teacher.blocks[L_ATT].ff.n.weight)
        blk.ff.n.bias.copy_(teacher.blocks[L_ATT].ff.n.bias)
stu_init.te.weight.data.copy_(teacher.te.weight.data)
stu_init.pe.weight.data.copy_(teacher.pe.weight.data)
stu_init.ln_f.weight.data.copy_(teacher.ln_f.weight.data)
stu_init.ln_f.bias.data.copy_(teacher.ln_f.bias.data)

val_init_0=eval_val(stu_init,n=30)
cos_init_0=cosine_sim(teacher,stu_init)
print(f"  Before fine-tune: val={val_init_0:.4f}  cos={cos_init_0:.4f}")

opt_init=torch.optim.AdamW(stu_init.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
t0=time.time()
for step in range(1,201):
    for pg in opt_init.param_groups: pg['lr']=clr(step,200)
    stu_init.train(); x,y=get_batch(); _,loss=stu_init(x,y)
    opt_init.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(stu_init.parameters(),1.0); opt_init.step()
    if step%50==0:
        vl=eval_val(stu_init,n=20)
        print(f"  step {step}  val={vl:.4f}  t={time.time()-t0:.0f}s")
val_init=eval_val(stu_init); t_init=time.time()-t0
cos_init=cosine_sim(teacher,stu_init)
print(f"  After fine-tune: val={val_init:.4f}  cos={cos_init:.4f}  t={t_init:.0f}s")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  RESULTS: HIDDEN STATES AT 13x LOWER COST")
print("="*65)

n_teacher=sum(p.numel() for p in teacher.parameters())
n_student=sum(p.numel() for p in stu_ce.parameters())

print(f"""
  TEACHER (24-layer, {t_teacher:.0f}s training):
    val={val_teacher:.4f}  params={n_teacher:,}

  STUDENT 2a — CE only (same embeddings, 200 steps, {t_ce:.0f}s):
    val={val_ce:.4f}  cos(hidden)={cos_ce:.4f}  params={n_student:,}
    compute ratio: {t_teacher/t_ce:.1f}x cheaper

  STUDENT 2b — Monodromy+CE annealed (200 steps, {t_mono:.0f}s):
    val={val_mono:.4f}  cos(hidden)={cos_mono:.4f}
    compute ratio: {t_teacher/t_mono:.1f}x cheaper

  STUDENT 2c — Monodromy init + CE fine-tune (200 steps, {t_init:.0f}s):
    val before={val_init_0:.4f}  cos before={cos_init_0:.4f}
    val after ={val_init:.4f}  cos after ={cos_init:.4f}
    compute ratio: {t_teacher/t_init:.1f}x cheaper

  PARAM REDUCTION: {n_teacher/n_student:.1f}x  ({n_teacher:,} → {n_student:,})

  KEY QUESTION:
    Does any 2-layer student approach teacher val={val_teacher:.4f}?
    Does monodromy alignment (2b/2c) beat pure CE (2a)?
    
    If 2b or 2c beats 2a:
      The monodromy transport carries information that CE alone misses.
      The Grassmannian alignment IS doing useful work.
      Hidden states at 13x lower cost is achievable.
    
    If all students similar:
      The embedding transfer (teacher→student) dominates.
      Monodromy alignment adds nothing beyond embeddings.
""")
