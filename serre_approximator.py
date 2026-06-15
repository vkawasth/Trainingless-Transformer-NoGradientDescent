#!/usr/bin/env python3
"""
Serre Cascade Approximator
===========================
6-layer student initialized from the Serre cascade of the teacher's
attractor Jacobian J_14.

Layer l is initialized so its Jacobian approximates ad(J_14)^l:
  W_l ← projection of ad(J_14)^l onto the active subspace U*

This is the closed-form initialization derived from the Kac-Moody
structure of the transformer's Jacobian Lie algebra.

HYPOTHESIS: A 6-layer student initialized this way should:
  1. Close the crystallization gap (val: 0.863 → ~0.3?)
  2. Outperform random 6L initialization + CE training
  3. Require fewer CE fine-tuning steps (the Serre cascade
     puts the student near the correct algebraic point)

COMPARISON:
  A: Teacher (24L, trained)
  B: 2L student + teacher embeddings + 200 CE (baseline)
  C: 6L random + teacher embeddings + 200 CE
  D: 6L Serre-initialized + teacher embeddings + 200 CE  ← the claim
  E: 6L Serre-initialized + teacher embeddings + 0 CE (pure cascade)
"""
import json, math, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_LAYERS=24; N_SERRE=6; BATCH=8; SEQ=64; LR=3e-4; PROJ=48
L_ATT=14  # attractor layer

print(f"\n{'='*65}")
print(f"  SERRE CASCADE APPROXIMATOR")
print(f"  6-layer student initialized from ad(J_14)^l")
print(f"  Closing the crystallization gap via Kac-Moody structure")
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
    def forward(self,x,y=None):
        h=self.te(x)+self.pe(torch.arange(x.shape[1]))
        for b in self.blocks: h=b(h)
        logits=self.head(self.ln_f(h))
        return logits,(F.cross_entropy(logits.view(-1,VOCAB),y.view(-1)) if y is not None else None)
    def hidden_states(self,x):
        hs=[]; h=self.te(x)+self.pe(torch.arange(x.shape[1])); hs.append(h.detach())
        for b in self.blocks: h=b(h); hs.append(h.detach())
        return hs

def clr(s,total,warmup=50):
    if s<=warmup: return LR*s/warmup
    return LR*0.5*(1+math.cos(math.pi*(s-warmup)/(total-warmup)))

def eval_val(model,n=60):
    model.eval(); ls=[]
    with torch.no_grad():
        for _ in range(n):
            x,y=get_batch('val'); _,l=model(x,y); ls.append(l.item())
    return float(np.mean(ls))

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

def comm(A,B): return A@B-B@A
def ad_k(A,B,k):
    r=B
    for _ in range(k): r=comm(A,r)
    return r

# ── Train teacher ─────────────────────────────────────────────────────────────
print("Stage 1: Train 24L teacher (300 steps)...")
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
val_teacher=eval_val(teacher)
print(f"  Teacher val={val_teacher:.4f}\n")

# ── Extract attractor Jacobian J_14 ──────────────────────────────────────────
print(f"Stage 2: Extract attractor Jacobian J_{{L_ATT}} and Serre cascade...")
torch.manual_seed(0); pos=SEQ//2; m=min(PROJ,SEQ,D)

# Average over 5 reference inputs for stability
J14_list=[]; U14_list=[]; ma=None
for _ in range(5):
    x_ref,_=get_batch('val'); x_ref=x_ref[0:1]
    with torch.no_grad():
        hs=teacher.hidden_states(x_ref); hs=[h[0] for h in hs]
    J,U,m_=layer_jac(teacher.blocks[L_ATT],hs[L_ATT],pos,m)
    J14_list.append(J); U14_list.append(U)
    if ma is None: ma=m_

J14=np.mean(J14_list,axis=0)   # [ma, ma] — attractor Jacobian
U14=np.mean(U14_list,axis=0)   # [D, ma]  — active subspace basis
dJ14=J14-np.eye(ma)            # perturbation

print(f"  J14 shape: {J14.shape}  ||δJ14||={np.linalg.norm(dJ14):.4f}")

# Compute Serre cascade: ad(dJ14)^l for l=1..N_SERRE
serre_cascade=[]
for l in range(1, N_SERRE+1):
    # Use dJ14 as the base element (the perturbation, not the full Jacobian)
    # ad(dJ14)^l(dJ14) = [dJ14,[dJ14,...[dJ14,dJ14]...]] — but [A,A]=0
    # Instead: ad(J14)^l applied to a probe = J14-neighbor direction
    # Use the next-layer Jacobian as the probe
    x_ref,_=get_batch('val'); x_ref=x_ref[0:1]
    with torch.no_grad():
        hs=teacher.hidden_states(x_ref); hs=[h[0] for h in hs]
    J_next,_,_=layer_jac(teacher.blocks[min(L_ATT+l,N_LAYERS-1)],
                          hs[min(L_ATT+l,N_LAYERS-1)],pos,m)
    # Serre cascade level l: ad(J14)^l(J_{14+l})
    cascade_l=ad_k(J14, J_next, l)
    # Normalize
    norm_l=float(np.linalg.norm(cascade_l))
    if norm_l>1e-8: cascade_l=cascade_l/norm_l
    serre_cascade.append(cascade_l)
    print(f"  Level {l}: ||ad(J14)^{l}(J_{{14+{l}}})|| = {norm_l:.4f}")

print(f"  Serre cascade extracted: {len(serre_cascade)} levels\n")

# ── Lift cascade to d-space weight matrices ───────────────────────────────────
def lift_to_weight(C_ma, U_basis, scale=0.02):
    """
    Lift [ma×ma] cascade matrix to [d×d] weight matrix.
    W = U @ C @ U^T + (I - UU^T) * scale  (keep orthogonal complement small)
    """
    UU=U_basis@U_basis.T  # [D,D] projection
    W=U_basis@C_ma@U_basis.T + (np.eye(D)-UU)*scale
    return W.astype(np.float32)

# ── Build Serre-initialized 6L student ───────────────────────────────────────
print("Stage 3: Build 6L Serre-initialized student...")
torch.manual_seed(99)
stu_serre=LM(D,N_HEADS,N_SERRE)

# Transfer teacher embeddings
for attr in ['te','pe','ln_f']:
    src=getattr(teacher,attr); dst=getattr(stu_serre,attr)
    if hasattr(src,'weight'): dst.weight.data.copy_(src.weight.data)
    if hasattr(src,'bias') and src.bias is not None:
        dst.bias.data.copy_(src.bias.data)

# Initialize each block from the Serre cascade
with torch.no_grad():
    for l in range(N_SERRE):
        C=serre_cascade[l]   # [ma,ma] cascade level l+1
        W_d=lift_to_weight(C,U14,scale=0.01)
        W_t=torch.tensor(W_d,dtype=torch.float32)
        # Set WK and WQ from the cascade (attention keys and queries)
        stu_serre.blocks[l].attn.WK.weight.copy_(W_t)
        stu_serre.blocks[l].attn.WQ.weight.copy_(W_t.T)
        # WV and output: use teacher's attractor layer weights
        stu_serre.blocks[l].attn.WV.weight.copy_(
            teacher.blocks[L_ATT].attn.WV.weight)
        stu_serre.blocks[l].attn.op.weight.copy_(
            teacher.blocks[L_ATT].attn.op.weight)
        stu_serre.blocks[l].ff.g.weight.copy_(
            teacher.blocks[L_ATT].ff.g.weight)
        stu_serre.blocks[l].ff.v.weight.copy_(
            teacher.blocks[L_ATT].ff.v.weight)
        stu_serre.blocks[l].ff.o.weight.copy_(
            teacher.blocks[L_ATT].ff.o.weight)

val_serre_0=eval_val(stu_serre)
print(f"  Serre-init 6L before fine-tune: val={val_serre_0:.4f}\n")

# ── Fine-tune all four students ───────────────────────────────────────────────
def finetune(model,steps=200,label=""):
    opt=torch.optim.AdamW(model.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
    t0=time.time()
    for step in range(1,steps+1):
        for pg in opt.param_groups: pg['lr']=clr(step,steps)
        model.train(); x,y=get_batch(); _,loss=model(x,y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if step%(steps//4)==0:
            vl=eval_val(model,n=20)
            print(f"  [{label}] step {step}  val={vl:.4f}")
    model.eval()
    return eval_val(model)

# B: 2L random + teacher embeddings
print("Stage 4B: 2L random + teacher embeddings (200 CE steps)...")
torch.manual_seed(99)
stu_2l=LM(D,N_HEADS,2)
for attr in ['te','pe','ln_f']:
    src=getattr(teacher,attr); dst=getattr(stu_2l,attr)
    if hasattr(src,'weight'): dst.weight.data.copy_(src.weight.data)
    if hasattr(src,'bias') and src.bias is not None:
        dst.bias.data.copy_(src.bias.data)
val_2l=finetune(stu_2l,200,"2L-random")
print(f"  2L random final: val={val_2l:.4f}\n")

# C: 6L random + teacher embeddings
print("Stage 4C: 6L random + teacher embeddings (200 CE steps)...")
torch.manual_seed(99)
stu_6l=LM(D,N_HEADS,N_SERRE)
for attr in ['te','pe','ln_f']:
    src=getattr(teacher,attr); dst=getattr(stu_6l,attr)
    if hasattr(src,'weight'): dst.weight.data.copy_(src.weight.data)
    if hasattr(src,'bias') and src.bias is not None:
        dst.bias.data.copy_(src.bias.data)
val_6l=finetune(stu_6l,200,"6L-random")
print(f"  6L random final: val={val_6l:.4f}\n")

# D: 6L Serre-initialized + teacher embeddings
print("Stage 4D: 6L Serre-initialized (200 CE steps)...")
val_serre=finetune(stu_serre,200,"6L-Serre")
print(f"  6L Serre final: val={val_serre:.4f}\n")

# ── Summary ───────────────────────────────────────────────────────────────────
n_teacher=sum(p.numel() for p in teacher.parameters())
n_2l=sum(p.numel() for p in stu_2l.parameters())
n_6l=sum(p.numel() for p in stu_6l.parameters())

print(f"\n{'='*65}")
print(f"  SERRE APPROXIMATOR RESULTS")
print("="*65)
print(f"""
  Teacher (24L):                val={val_teacher:.4f}  params={n_teacher:,}

  A: 2L random + emb (200 CE):  val={val_2l:.4f}  params={n_2l:,}
  B: 6L random + emb (200 CE):  val={val_6l:.4f}  params={n_6l:,}
  C: 6L Serre-init (0 CE):      val={val_serre_0:.4f}  (zero-shot)
  D: 6L Serre-init (200 CE):    val={val_serre:.4f}  params={n_6l:,}

  Crystallization gap (teacher→2L): {val_2l-val_teacher:.4f} nats
  Gap with 6L random:               {val_6l-val_teacher:.4f} nats
  Gap with 6L Serre:                {val_serre-val_teacher:.4f} nats

  Serre init advantage over random 6L: {val_6l-val_serre:.4f} nats

  READING:
  If Serre-init < random 6L:
    The cascade initialization carries genuine algebraic signal.
    The Kac-Moody structure of J14 encodes the correct subspace
    for the approximator to find the teacher's representation.

  If Serre-init ≈ random 6L:
    The cascade does not help beyond random.
    The closed-form initialization is not more informative
    than the data distribution alone.

  If 6L random >> 2L random:
    Depth (not initialization) closes the crystallization gap.
    Each additional layer adds one Serre level.
    6 layers implements Serre levels 1-6 via gradient descent.
""")
