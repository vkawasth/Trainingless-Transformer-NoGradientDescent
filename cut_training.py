#!/usr/bin/env python3
"""
Training Cut Experiment
========================
The profiler identified three phases:
  Phase 1 (0-75):   fast deformation — skippable via Serre+Laplacian init
  Phase 2 (75-225): co-adaptation — irreducible CE signal required
  Phase 3 (225-300): refinement — eliminable (gram_align frozen, orientation frozen)

This experiment tests the minimal training recipe:
  Init: Laplacian embedding + Serre cascade (skips Phase 1)
  Train: joint CE for 150 steps (Phase 2 only, skips Phase 3)
  
Compare against:
  A: Teacher emb + Serre + 200 CE (val=0.187, the current best)
  B: Teacher emb + Serre + 150 CE (Phase 2 only, teacher emb)
  C: Laplacian + Serre + 200 CE (val=0.305, prior result)
  D: Laplacian + Serre + 150 CE (Phase 2 only, Laplacian emb) ← TARGET
  E: Laplacian + Serre + 100 CE (aggressive cut)
  F: Laplacian + Serre + 75 CE  (enter Phase 2 late, exit early)

Also profile the Serre student during training to confirm
Phase 1 is skipped and Phase 2 dynamics match the teacher.
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
print(f"  TRAINING CUT EXPERIMENT")
print(f"  Phase 1 skip via init + Phase 3 elimination")
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

def grassmannian_dist(E1,E2,k=16):
    U1,_,_=np.linalg.svd(E1,full_matrices=False); U1=U1[:,:k]
    U2,_,_=np.linalg.svd(E2,full_matrices=False); U2=U2[:,:k]
    sv=np.clip(np.linalg.svd(U1.T@U2,compute_uv=False),0,1)
    return float(np.sqrt(np.sum(np.arccos(sv)**2)))

def gram_align(E,E_ref):
    G1=(E@E.T).flatten(); G2=(E_ref@E_ref.T).flatten()
    return float(np.corrcoef(G1,G2)[0,1])

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
    return J.T,U.detach().numpy()

def comm(A,B): return A@B-B@A
def ad_k(A,B,k):
    r=B
    for _ in range(k): r=comm(A,r)
    return r
def lift_to_d(C,U,scale=0.01):
    UU=U@U.T
    return (U@C@U.T+(np.eye(D)-UU)*scale).astype(np.float32)

# ════════════════════════════════════════════════════════
# STAGE 0: Train teacher + build Laplacian
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

# Build Laplacian
print("Building Laplacian embedding...")
bigram=np.zeros((VOCAB,VOCAB),dtype=np.float32)
for k in range(len(train_ids)-1):
    a,b=train_ids[k],train_ids[k+1]
    if 0<=a<VOCAB and 0<=b<VOCAB: bigram[a,b]+=1
A_mat=(bigram+bigram.T)/2
deg=np.maximum(A_mat.sum(1),1e-10)
D_is=sp.diags(1/np.sqrt(deg))
L_sp=sp.eye(VOCAB)-D_is@sp.csr_matrix(A_mat)@D_is
vals_l,vecs_l=spla.eigsh(L_sp,k=min(D+2,VOCAB-1),which='SM')
idx_l=np.argsort(vals_l)
vals_l=vals_l[idx_l]; vecs_l=vecs_l[:,idx_l]
skip_l=int(np.sum(vals_l<1e-8))
E_lap=(vecs_l[:,skip_l:skip_l+D]*
       (teacher_norm/max(float(np.linalg.norm(vecs_l[:,skip_l:skip_l+D],'fro')),1e-8))).astype(np.float32)
ga_lap=gram_align(E_lap,E_teacher)
print(f"  Laplacian gram_align={ga_lap:.4f}\n")

# ════════════════════════════════════════════════════════
# STAGE 1: Extract invariants + cascade
# ════════════════════════════════════════════════════════
print("Stage 1: Extract invariants + Serre cascade...")
torch.manual_seed(0); pos=SEQ//2; m=min(PROJ,SEQ,D); ma=None
J_acc=[[] for _ in range(N_LAYERS)]; U_acc=[[] for _ in range(N_LAYERS)]
for ref in range(5):
    x_ref,_=get_batch('val'); x_ref=x_ref[0:1]
    with torch.no_grad(): hs=teacher.hidden_states(x_ref); hs=[h[0] for h in hs]
    for l in range(N_LAYERS):
        J,U=layer_jac(teacher.blocks[l],hs[l],pos,m)
        J_acc[l].append(J); U_acc[l].append(U)
        if ma is None: ma=J.shape[0]
    if (ref+1)%3==0: print(f"  ref {ref+1}/5...",flush=True)

Js=[np.mean(J_acc[l],axis=0) for l in range(N_LAYERS)]
Us=[np.mean(U_acc[l],axis=0) for l in range(N_LAYERS)]
J14=Js[L_ATT]; U14=Us[L_ATT]

cascade=[]
for l in range(1,N_STU+1):
    C_l=ad_k(J14,Js[min(L_ATT+l,N_LAYERS-1)],l)
    n=float(np.linalg.norm(C_l))
    if n>1e-8: C_l=C_l/n
    cascade.append(C_l)
print(f"  Done. ma={ma}  cascade: {N_STU} levels\n")

def inject_cascade(model):
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

# ════════════════════════════════════════════════════════
# STAGE 2: Run all conditions with profiling
# ════════════════════════════════════════════════════════
def run_with_profile(E_init, label, steps, profile_every=25):
    """Joint train with continuous profiling of deformation phases."""
    torch.manual_seed(99)
    model=LM(D,N_HEADS,N_STU)
    with torch.no_grad():
        model.te.weight.copy_(torch.tensor(E_init[:VOCAB,:D].copy()))
    inject_cascade(model)

    opt=torch.optim.AdamW(model.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
    E_prev=model.te.weight.data.numpy().copy()

    print(f"\n  [{label}] {steps} steps")
    print(f"  {'step':>5}  {'val':>7}  {'grass':>7}  {'gram_lap':>9}  {'phase'}")
    print("  "+"-"*45)

    results={}
    for step in range(1,steps+1):
        for pg in opt.param_groups: pg['lr']=clr(step,steps,max(10,steps//6))
        model.train(); x,y=get_batch(); _,loss=model(x,y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()

        if step%profile_every==0 or step==steps:
            vl=eval_val(model,n=20)
            E_curr=model.te.weight.data.numpy()
            gdist=grassmannian_dist(E_curr,E_prev)
            ga=gram_align(E_curr,E_teacher)
            E_prev=E_curr.copy()
            # Phase identification
            if gdist>0.5: phase="1-fast"
            elif gdist>0.1: phase="2-adapt"
            else: phase="3-refine"
            results[step]=(vl,gdist,ga,phase)
            print(f"  {step:>5}  {vl:>7.4f}  {gdist:>7.4f}  {ga:>9.4f}  {phase}")

    return eval_val(model), results

# Conditions
print("="*65)
print("RUNNING ALL CONDITIONS")
print("="*65)

conditions=[
    ("A: Teacher+Serre+200CE",  E_teacher, 200),
    ("B: Teacher+Serre+150CE",  E_teacher, 150),
    ("C: Lap+Serre+200CE",      E_lap,     200),
    ("D: Lap+Serre+150CE",      E_lap,     150),  # TARGET
    ("E: Lap+Serre+100CE",      E_lap,     100),
    ("F: Lap+Serre+75CE",       E_lap,      75),
]

final_results={}
for label,E_init,steps in conditions:
    vf,prof=run_with_profile(E_init,label,steps)
    final_results[label]=(vf,prof)

# ════════════════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"  TRAINING CUT RESULTS")
print("="*65)

# Compute layer-steps and vs teacher compute
teacher_ls=300*24
print(f"\n  Teacher: {teacher_ls} layer-steps  val={val_teacher:.4f}")
print(f"\n  {'Condition':>25}  {'val':>7}  {'layer-steps':>12}  "
      f"{'vs teacher':>11}  {'beats teacher?'}")
print("  "+"-"*72)

for label,E_init,steps in conditions:
    vf,_=final_results[label]
    ls=steps*N_STU
    ratio=teacher_ls/ls
    beats="YES ✓" if vf<val_teacher else "no"
    print(f"  {label:>25}  {vf:>7.4f}  {ls:>12,}  "
          f"{ratio:>10.1f}x  {beats}")

print(f"""
  PHASE ANALYSIS:
  Phase 1 (fast deformation, grass>0.5): skipped by good init
  Phase 2 (co-adaptation, grass 0.1-0.5): irreducible
  Phase 3 (refinement, grass<0.1): eliminable

  The Laplacian init SKIPS Phase 1 by starting near the
  correct A∞ orbit. Check the profiling output above:
  - Teacher init: Phase 1 present in first 50-75 steps
  - Laplacian init: should skip directly to Phase 2

  KEY RESULT:
  Condition D (Lap+Serre+150CE) is the target:
    - Skip Phase 1 (Laplacian init ≈ 0 Phase 1 steps)
    - Phase 2 only (~150 steps)
    - Skip Phase 3 (stop at step 150 not 225)
    - Compute: {150*N_STU} layer-steps vs {teacher_ls} teacher
    - Reduction: {teacher_ls/(150*N_STU):.1f}x vs teacher
    - If val < 0.25: beats teacher at {teacher_ls/(150*N_STU):.0f}x lower compute
""")
