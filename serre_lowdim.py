#!/usr/bin/env python3
"""
Serre Approximator with Low-Dimensional Embedding
===================================================
The three theories failed because the cascade was frozen while the
embedding trained. The fix: train EVERYTHING jointly (E = C @ G,
blocks initialized from Serre cascade) just like the successful
Serre approximator — but with E constrained to a low-dim subspace.

The cascade adapts to the embedding. The embedding adapts to the cascade.
They find each other through joint gradient descent.

ARCHITECTURES TESTED:
  A: Full embedding [V x D] + Serre cascade (baseline, val=0.187)
  B: E = C @ G_output  [V x 3] + Serre cascade, joint train
  C: E = C @ G_attractor [V x 48] + Serre cascade, joint train
  D: E = C @ G_laplacian [V x D_lap] + Serre cascade, joint train

G SOURCES:
  G_output:    3 output Floer generators from L21-L23
  G_attractor: M_fwd^{-1} @ U14^T  (48 attractor pullback dirs)
  G_laplacian: Top-D Laplacian eigenvectors (full D, but initialized well)

KEY DIFFERENCE from previous:
  Previous: inject cascade, freeze blocks, train embedding
  This:     inject cascade, train ALL params jointly (E = C @ G)
  The cascade updates via backprop through C @ G.
"""
import json, math, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_LAYERS=24; N_STU=6; BATCH=8; SEQ=64; LR=3e-4; PROJ=48
L_ATT=14

print(f"\n{'='*65}")
print(f"  SERRE APPROXIMATOR + LOW-DIM EMBEDDING")
print(f"  Joint training: cascade adapts to embedding")
print(f"{'='*65}\n")

with open('/tmp/train_ids.json') as f: train_ids=list(map(int,json.load(f)))
with open('/tmp/val_ids.json')   as f: val_ids  =list(map(int,json.load(f)))
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

# Standard LM (for teacher and baseline)
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

# Low-dim LM: E = C @ G, JOINT training (blocks NOT frozen)
class LowDimLM(nn.Module):
    def __init__(self,d,nh,nl,G_fixed,init_C=None):
        super().__init__()
        ng=G_fixed.shape[0]
        if init_C is not None:
            self.C=nn.Parameter(torch.tensor(init_C,dtype=torch.float32))
        else:
            self.C=nn.Parameter(torch.randn(VOCAB,ng)*0.02)
        # G is a learnable parameter too — let it adapt
        self.G=nn.Parameter(torch.tensor(G_fixed,dtype=torch.float32))
        self.pe=nn.Embedding(512,d)
        self.blocks=nn.ModuleList([Block(d,nh) for _ in range(nl)])
        self.ln_f=nn.LayerNorm(d)
        nn.init.normal_(self.pe.weight,std=0.02)
    def get_E(self): return self.C@self.G  # [V, D]
    def forward(self,x,y=None):
        E=self.get_E()
        h=E[x]+self.pe(torch.arange(x.shape[1]))
        for b in self.blocks: h=b(h)
        h=self.ln_f(h)
        logits=h@E.T
        loss=F.cross_entropy(logits.reshape(-1,VOCAB),y.reshape(-1)) if y is not None else None
        return logits,loss
    def hidden_states(self,x):
        hs=[]; E=self.get_E()
        h=E[x]+self.pe(torch.arange(x.shape[1])); hs.append(h.detach())
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
def lift_to_d(C,U,scale=0.01):
    UU=U@U.T
    return (U@C@U.T+(np.eye(D)-UU)*scale).astype(np.float32)

# ════════════════════════════════════════════════════════
# STAGE 0: Train teacher
# ════════════════════════════════════════════════════════
print("Stage 0: Train teacher (300 steps)...")
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
E_teacher=teacher.te.weight.data.numpy().copy()
teacher_norm=float(np.linalg.norm(E_teacher,'fro'))
print(f"  Teacher val={val_teacher:.4f}\n")

# ════════════════════════════════════════════════════════
# STAGE 1: Extract invariants + cascade + generators
# ════════════════════════════════════════════════════════
print("Stage 1: Extract invariants...")
torch.manual_seed(0); pos=SEQ//2; m=min(PROJ,SEQ,D); ma=None
J_acc=[[] for _ in range(N_LAYERS)]; U_acc=[[] for _ in range(N_LAYERS)]
for ref in range(5):
    x_ref,_=get_batch('val'); x_ref=x_ref[0:1]
    with torch.no_grad(): hs=teacher.hidden_states(x_ref); hs=[h[0] for h in hs]
    for l in range(N_LAYERS):
        J,U,m_=layer_jac(teacher.blocks[l],hs[l],pos,m)
        J_acc[l].append(J); U_acc[l].append(U)
        if ma is None: ma=m_
    if (ref+1)%3==0: print(f"  ref {ref+1}/5...",flush=True)
Js=[np.mean(J_acc[l],axis=0) for l in range(N_LAYERS)]
Us=[np.mean(U_acc[l],axis=0) for l in range(N_LAYERS)]
J14=Js[L_ATT]; U14=Us[L_ATT]

# Serre cascade
cascade=[]
for l in range(1,N_STU+1):
    C_l=ad_k(J14,Js[min(L_ATT+l,N_LAYERS-1)],l)
    n=float(np.linalg.norm(C_l))
    if n>1e-8: C_l=C_l/n
    cascade.append(C_l)

# M_fwd and pullback basis
M_fwd=np.eye(ma)
for l in range(L_ATT+1): M_fwd=Js[l]@M_fwd
M_fwd_D=U14@M_fwd@U14.T+(np.eye(D)-U14@U14.T)
M_inv=np.linalg.inv(M_fwd_D+1e-3*np.eye(D))
G_att=(M_inv@U14).T  # [ma, D]
G_att_q,_=np.linalg.qr(G_att.T); G_att=G_att_q.T[:ma,:].astype(np.float32)

# Output generators from L21-L23
gen_vecs=[]
for l in range(21,N_LAYERS):
    dJ=Js[l]-np.eye(ma)
    Usv,_,_=np.linalg.svd(dJ)
    for k in range(3): gen_vecs.append(Us[l]@Usv[:,k])
_,_,Vg=np.linalg.svd(np.stack(gen_vecs),full_matrices=False)
G_out_raw=Vg[:3,:].astype(np.float32)
G_out_q,_=np.linalg.qr(G_out_raw.T); G_out=G_out_q.T[:3,:].astype(np.float32)

# Laplacian embedding as G (full D, used as initialization)
print("  Computing Laplacian...")
bigram=np.zeros((VOCAB,VOCAB),dtype=np.float32)
for k in range(len(train_ids)-1):
    a,b=train_ids[k],train_ids[k+1]
    if 0<=a<VOCAB and 0<=b<VOCAB: bigram[a,b]+=1
A=(bigram+bigram.T)/2
deg=np.maximum(A.sum(1),1e-10)
D_inv_sqrt=sp.diags(1/np.sqrt(deg))
L_norm=sp.eye(VOCAB)-D_inv_sqrt@sp.csr_matrix(A)@D_inv_sqrt
vals,vecs=spla.eigsh(L_norm,k=min(D+2,VOCAB-1),which='SM')
idx=np.argsort(vals); vals=vals[idx]; vecs=vecs[:,idx]
skip=int(np.sum(vals<1e-8))
E_lap=(vecs[:,skip:skip+D]*float(teacher_norm/max(np.linalg.norm(vecs[:,skip:skip+D],'fro'),1e-8))).astype(np.float32)
print(f"  Done. Cascade: {N_STU} levels\n")

def inject_cascade_to(model):
    """Inject Serre cascade into any model with .blocks attribute."""
    with torch.no_grad():
        model.pe.weight.copy_(teacher.pe.weight)
        model.ln_f.weight.copy_(teacher.ln_f.weight)
        model.ln_f.bias.copy_(teacher.ln_f.bias)
        for l in range(N_STU):
            W_d=lift_to_d(cascade[l],U14,scale=0.01)
            W_t=torch.tensor(W_d,dtype=torch.float32)
            model.blocks[l].attn.WK.weight.copy_(W_t)
            model.blocks[l].attn.WQ.weight.copy_(W_t.T)
            model.blocks[l].attn.WV.weight.copy_(teacher.blocks[L_ATT].attn.WV.weight)
            model.blocks[l].attn.op.weight.copy_(teacher.blocks[L_ATT].attn.op.weight)
            model.blocks[l].ff.g.weight.copy_(teacher.blocks[L_ATT].ff.g.weight)
            model.blocks[l].ff.v.weight.copy_(teacher.blocks[L_ATT].ff.v.weight)
            model.blocks[l].ff.o.weight.copy_(teacher.blocks[L_ATT].ff.o.weight)

def joint_train(model, steps=200, label=""):
    """Joint training — ALL parameters update including blocks."""
    opt=torch.optim.AdamW(model.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
    t0=time.time()
    checkpoints={}
    for step in range(1,steps+1):
        for pg in opt.param_groups: pg['lr']=clr(step,steps,50)
        model.train(); x,y=get_batch(); _,loss=model(x,y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if step in [50,100,150,200]:
            vl=eval_val(model,n=20)
            checkpoints[step]=vl
            print(f"  [{label}] step {step}  val={vl:.4f}  t={time.time()-t0:.0f}s")
    return eval_val(model), checkpoints

# ════════════════════════════════════════════════════════
# BASELINE A: Full Serre approximator (teacher emb)
# ════════════════════════════════════════════════════════
print("="*65)
print("BASELINE A: Full Serre + teacher embedding (joint train)")
print("="*65)
torch.manual_seed(99)
mA=LM(D,N_HEADS,N_STU)
mA.te.weight.data.copy_(teacher.te.weight.data)
inject_cascade_to(mA)
vA,ckA=joint_train(mA,200,"A-full")
print(f"  Baseline A final: val={vA:.4f}\n")

# ════════════════════════════════════════════════════════
# MODEL B: Output generators (3-dim) + Serre, joint train
# ════════════════════════════════════════════════════════
print("="*65)
print("MODEL B: Output generators (3-dim) + Serre, JOINT train")
print(f"  G shape: {G_out.shape}  params C: {3*VOCAB:,}")
print("="*65)
# Init C from teacher projection onto output generators
C_B_init=(E_teacher@G_out.T).astype(np.float32)
print(f"  C init from teacher projection, range: {C_B_init.min():.3f} to {C_B_init.max():.3f}")
torch.manual_seed(99)
mB=LowDimLM(D,N_HEADS,N_STU,G_out,init_C=C_B_init)
inject_cascade_to(mB)
v0B=eval_val(mB,n=20); print(f"  Zero-shot: val={v0B:.4f}")
vB,ckB=joint_train(mB,200,"B-3dim")
print(f"  Model B final: val={vB:.4f}\n")

# ════════════════════════════════════════════════════════
# MODEL C: Attractor pullback (48-dim) + Serre, joint train
# ════════════════════════════════════════════════════════
print("="*65)
print("MODEL C: Attractor pullback (48-dim) + Serre, JOINT train")
print(f"  G shape: {G_att.shape}  params C: {ma*VOCAB:,}")
print("="*65)
C_C_init=(E_teacher@G_att.T).astype(np.float32)
print(f"  C init from teacher projection, range: {C_C_init.min():.3f} to {C_C_init.max():.3f}")
torch.manual_seed(99)
mC=LowDimLM(D,N_HEADS,N_STU,G_att,init_C=C_C_init)
inject_cascade_to(mC)
v0C=eval_val(mC,n=20); print(f"  Zero-shot: val={v0C:.4f}")
vC,ckC=joint_train(mC,200,"C-48dim")
print(f"  Model C final: val={vC:.4f}\n")

# ════════════════════════════════════════════════════════
# MODEL D: Laplacian init (full D) + Serre, joint train
# ════════════════════════════════════════════════════════
print("="*65)
print("MODEL D: Laplacian init (full D=256) + Serre, JOINT train")
print(f"  G=I (full D), C=E_lap  params: {D*VOCAB:,}")
print("="*65)
# Use identity G — C is the full D-dim embedding, initialized from Laplacian
G_identity=np.eye(D,dtype=np.float32)
torch.manual_seed(99)
mD=LowDimLM(D,N_HEADS,N_STU,G_identity,init_C=E_lap)
inject_cascade_to(mD)
v0D=eval_val(mD,n=20); print(f"  Zero-shot: val={v0D:.4f}")
vD,ckD=joint_train(mD,200,"D-Laplacian")
print(f"  Model D final: val={vD:.4f}\n")

# ════════════════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"  SERRE + LOW-DIM EMBEDDING — JOINT TRAINING RESULTS")
print("="*65)

print(f"\n  {'Model':>25}  {'dim':>5}  {'params_C':>10}  "
      f"{'step50':>8}  {'step100':>8}  {'step200':>8}")
print("  "+"-"*72)

rows=[
    ("A: Full (teacher emb)",D,D*VOCAB,ckA),
    ("B: Output gen (3-dim)",3,3*VOCAB,ckB),
    ("C: Attractor PB (48-dim)",ma,ma*VOCAB,ckC),
    ("D: Laplacian (256-dim)",D,D*VOCAB,ckD),
]
for name,dim,np_,ck in rows:
    s50=ck.get(50,99); s100=ck.get(100,99); s200=ck.get(200,99)
    print(f"  {name:>25}  {dim:>5}  {np_:>10,}  "
          f"{s50:>8.4f}  {s100:>8.4f}  {s200:>8.4f}")

print(f"""
  Teacher (24L oracle):        val={val_teacher:.4f}
  Prior best (Serre+teacher):  val=0.1865

  KEY READING:
  Joint training lets cascade and embedding co-adapt.
  
  If B (3-dim output gen) approaches A (full):
    The 3 Floer generators ARE the sufficient basis.
    Training reduces to learning 3 coords per token (2,025 params).
    85x parameter reduction with same quality.

  If C (48-dim attractor) approaches A:
    The attractor pullback basis is sufficient.
    Training reduces to 32,400 params (81% fewer).

  If D (Laplacian) approaches A:
    The Laplacian initialization with joint training
    reaches teacher quality — potential for training
    reduction via better initialization.

  If only A succeeds:
    The full embedding is necessary.
    No low-dimensional basis exists.
    Minimum training = 13.5% of params for ~120 steps.
""")
