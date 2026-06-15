#!/usr/bin/env python3
"""
Structure Formation Profiler
==============================
Profile WHAT structures form during the 200 training steps
of the Serre approximator. Extract at every 10 steps:

1. Which Serre levels have crystallized
   (cascade alignment: how well does layer l match ad(J14)^l?)

2. Which filtration pages have appeared
   (rank profile: does it match [14,20,20,12,...,3]?)

3. Lie bracket decay at each layer
   (||[J_l, J_{l+1}]|| → 0 means l_2 vanishing: MC element forming)

4. Cohomology class formation
   (H^k = ker(d_k)/im(d_{k-1}) at each stalk)

5. Property T spectral gap crystallization
   (commutator graph gap: when does it lock in?)

The goal: identify which structures form SIMULTANEOUSLY
and can therefore be built in a single algebraic operation.
"""
import json, math, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_LAYERS_T=24; N_STU=6; BATCH=8; SEQ=64; LR=3e-4; PROJ=48
L_ATT=14; PROFILE_EVERY=10

print(f"\n{'='*65}")
print(f"  STRUCTURE FORMATION PROFILER")
print(f"  What crystallizes when during Serre approximator training?")
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

def eval_val(model,n=30):
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
    return J.T,U.detach().numpy()

def comm(A,B): return A@B-B@A
def ad_k(A,B,k):
    r=B
    for _ in range(k): r=comm(A,r)
    return r
def lift_to_d(C,U,scale=0.01):
    UU=U@U.T
    return (U@C@U.T+(np.eye(D)-UU)*scale).astype(np.float32)

# ════════════════════════════════════════════════════
# Train teacher + extract cascade
# ════════════════════════════════════════════════════
print("Training teacher (300 steps)...")
torch.manual_seed(42)
teacher=LM(D,N_HEADS,N_LAYERS_T)
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
print(f"  Teacher val={eval_val(teacher):.4f}\n")

print("Extracting teacher Jacobians...",flush=True)
torch.manual_seed(0); pos=SEQ//2; m=min(PROJ,SEQ,D); ma=None
J_acc=[[] for _ in range(N_LAYERS_T)]; U_acc=[[] for _ in range(N_LAYERS_T)]
for ref in range(5):
    x_ref,_=get_batch('val'); x_ref=x_ref[0:1]
    with torch.no_grad(): hs=teacher.hidden_states(x_ref); hs=[h[0] for h in hs]
    for l in range(N_LAYERS_T):
        J,U=layer_jac(teacher.blocks[l],hs[l],pos,m)
        J_acc[l].append(J); U_acc[l].append(U)
        if ma is None: ma=J.shape[0]
    if (ref+1)%3==0: print(f"  ref {ref+1}/5...",flush=True)

Js_t=[np.mean(J_acc[l],axis=0) for l in range(N_LAYERS_T)]
Us_t=[np.mean(U_acc[l],axis=0) for l in range(N_LAYERS_T)]
J14=Js_t[L_ATT]; U14=Us_t[L_ATT]

# Target Serre cascade levels for comparison
cascade_targets=[]
for l in range(1,N_STU+1):
    C=ad_k(J14,Js_t[min(L_ATT+l,N_LAYERS_T-1)],l)
    n=float(np.linalg.norm(C))
    if n>1e-8: C=C/n
    cascade_targets.append(C)

# Target rank profile from teacher
target_ranks=[int(np.sum(np.linalg.svd(Js_t[l]-np.eye(ma),
              compute_uv=False)>0.1)) for l in range(N_LAYERS_T)]

cascade=cascade_targets  # same object
print(f"  Done. ma={ma}\n")

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

# ════════════════════════════════════════════════════
# Build Serre student + profile structure formation
# ════════════════════════════════════════════════════
print("Building Serre student + profiling structure formation...")
torch.manual_seed(99)
student=LM(D,N_HEADS,N_STU)
student.te.weight.data.copy_(teacher.te.weight.data)
inject_cascade(student)

opt_s=torch.optim.AdamW(student.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)

x_ref2,_=get_batch('val'); x_ref2=x_ref2[0:1]

def extract_structures(model, step):
    """Extract all algebraic structures at current training state."""
    model.eval()
    with torch.no_grad(): hs=model.hidden_states(x_ref2); hs=[h[0] for h in hs]

    Js=[]; Us=[]
    for l in range(N_STU):
        J,U=layer_jac(model.blocks[l],hs[l],pos,m)
        Js.append(J); Us.append(U)

    s={}

    # 1. SERRE LEVEL ALIGNMENT
    # How well does each student layer match its cascade target?
    serre_align=[]
    for l in range(N_STU):
        dJ=Js[l]-np.eye(ma)
        target=cascade_targets[l]
        # Normalize both
        dJ_n=dJ/max(float(np.linalg.norm(dJ)),1e-8)
        target_n=target/max(float(np.linalg.norm(target)),1e-8)
        # Frobenius inner product = alignment
        align=float(np.sum(dJ_n*target_n))
        serre_align.append(align)
    s['serre_align']=serre_align

    # 2. RANK PROFILE (filtration pages)
    ranks=[]
    for l in range(N_STU):
        sv=np.linalg.svd(Js[l]-np.eye(ma),compute_uv=False)
        ranks.append(int(np.sum(sv>sv[0]*0.1)) if sv[0]>1e-8 else 0)
    s['ranks']=ranks

    # 3. LIE BRACKET DECAY (mu2 vanishing → MC element forming)
    brackets=[]
    for l in range(N_STU-1):
        b=float(np.linalg.norm(comm(Js[l+1],Js[l])))
        brackets.append(b)
    s['brackets']=brackets
    s['bracket_mean']=float(np.mean(brackets)) if brackets else 0

    # 4. MONODROMY FORMATION
    Mf=np.eye(ma)
    for l in range(N_STU): Mf=Js[l]@Mf
    sv_mf=np.linalg.svd(Mf,compute_uv=False)
    s['sv_mfwd']=float(sv_mf[0])

    # 5. COMMUTATOR GRAPH GAP (Property T)
    # Fast approximation: 3x3 commutator matrix for speed
    A3=np.zeros((N_STU,N_STU))
    for i in range(N_STU):
        for j in range(N_STU):
            if i!=j: A3[i,j]=float(np.linalg.norm(comm(Js[i],Js[j])))
    A3=(A3+A3.T)/2
    eigs3=np.linalg.eigvalsh(A3)[::-1]
    s['gap']=float(eigs3[0]-eigs3[1]) if len(eigs3)>1 else 0

    # 6. COHOMOLOGY: H^k = ker(d_k) / im(d_{k-1})
    # Approximate: rank of d_k (kernel size = ma - rank)
    coh_dims=[]
    for l in range(N_STU):
        dJ=Js[l]-np.eye(ma)
        sv=np.linalg.svd(dJ,compute_uv=False)
        rank=int(np.sum(sv>sv[0]*0.05)) if sv[0]>1e-8 else 0
        kernel_dim=ma-rank
        coh_dims.append(kernel_dim)
    s['coh_dims']=coh_dims

    return s

# Profile header
print(f"\n  Profiling every {PROFILE_EVERY} steps...")
print(f"\n  {'step':>5}  {'val':>7}  "
      f"{'S1':>5}  {'S2':>5}  {'S3':>5}  {'S4':>5}  {'S5':>5}  {'S6':>5}  "
      f"{'bracket':>8}  {'gap':>6}  {'sv_mf':>7}  "
      f"{'R1':>3}{'R2':>3}{'R3':>3}{'R4':>3}{'R5':>3}{'R6':>3}")
print("  "+"-"*95)

all_profiles=[]
E_prev=student.te.weight.data.numpy().copy()

for step in range(0,201):
    if step>0:
        for pg in opt_s.param_groups: pg['lr']=clr(step,200,50)
        student.train(); x,y=get_batch(); _,loss=student(x,y)
        opt_s.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(),1.0); opt_s.step()

    if step%PROFILE_EVERY==0:
        student.eval()
        vl=eval_val(student,n=20)
        s=extract_structures(student,step)
        s['step']=step; s['val']=vl

        # Grassmannian distance
        E_curr=student.te.weight.data.numpy()
        U1,_,_=np.linalg.svd(E_curr,full_matrices=False); U1=U1[:,:16]
        U2,_,_=np.linalg.svd(E_prev,full_matrices=False); U2=U2[:,:16]
        sv_c=np.clip(np.linalg.svd(U1.T@U2,compute_uv=False),0,1)
        s['grass']=float(np.sqrt(np.sum(np.arccos(sv_c)**2)))
        E_prev=E_curr.copy()

        all_profiles.append(s)

        sa=s['serre_align']; r=s['ranks']
        print(f"  {step:>5}  {vl:>7.4f}  "
              f"{sa[0]:>5.2f}  {sa[1]:>5.2f}  {sa[2]:>5.2f}  "
              f"{sa[3]:>5.2f}  {sa[4]:>5.2f}  {sa[5]:>5.2f}  "
              f"{s['bracket_mean']:>8.4f}  {s['gap']:>6.2f}  {s['sv_mfwd']:>7.3f}  "
              f"{r[0]:>3}{r[1]:>3}{r[2]:>3}{r[3]:>3}{r[4]:>3}{r[5]:>3}")

# ════════════════════════════════════════════════════
# ANALYSIS: What forms when?
# ════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"  STRUCTURE FORMATION ANALYSIS")
print("="*65)

# Find crystallization step for each structure
def crystallize_step(values, threshold_fn, profile_steps):
    """Find step where structure first stabilizes."""
    for i in range(1,len(values)):
        if threshold_fn(values[i], values[i-1]):
            return profile_steps[i]
    return None

steps=[p['step'] for p in all_profiles]
vals=[p['val'] for p in all_profiles]

# Serre level crystallization: align > 0.3 and stable
print(f"\n  SERRE CASCADE LEVELS — when does each level crystallize?")
print(f"  (crystallized = alignment > 0.3 with target)")
for l in range(N_STU):
    aligns=[p['serre_align'][l] for p in all_profiles]
    cryst=next((steps[i] for i in range(len(aligns)) if aligns[i]>0.3), None)
    print(f"  Level {l+1}: first align>0.3 at step {cryst}  "
          f"(final={aligns[-1]:.3f})")

print(f"\n  LIE BRACKET DECAY — when does mu2 vanish?")
brackets=[p['bracket_mean'] for p in all_profiles]
b0=brackets[0]
for i,b in enumerate(brackets):
    if b<b0*0.5:
        print(f"  Bracket decays to 50% at step {steps[i]}  ({b:.4f} vs {b0:.4f})")
        break

print(f"\n  PROPERTY T GAP — when does expander structure form?")
gaps=[p['gap'] for p in all_profiles]
for i,g in enumerate(gaps):
    if g>1.0:
        print(f"  Gap > 1.0 at step {steps[i]}  (gap={g:.3f})")
        break

print(f"\n  COHOMOLOGY — kernel dimensions at each stalk:")
print(f"  {'step':>5}  {'ker_L1':>7}  {'ker_L2':>7}  {'ker_L3':>7}  "
      f"{'ker_L4':>7}  {'ker_L5':>7}  {'ker_L6':>7}")
print("  "+"-"*52)
for p in all_profiles[::2]:
    coh=p['coh_dims']
    print(f"  {p['step']:>5}  "+"  ".join(f"{c:>7}" for c in coh))

print(f"\n  STRUCTURES FORMING SIMULTANEOUSLY:")
print(f"  (structures that crystallize within the same 10-step window)")
print()

# Group structures by crystallization step
cryst_steps={}
for l in range(N_STU):
    aligns=[p['serre_align'][l] for p in all_profiles]
    s=next((steps[i] for i in range(len(aligns)) if aligns[i]>0.3), 999)
    cryst_steps[f'Serre_L{l+1}']=s

# Bracket decay
for i in range(1,len(brackets)):
    if brackets[i]<brackets[0]*0.5:
        cryst_steps['LieBracket']=steps[i]; break

# Gap formation
for i,g in enumerate(gaps):
    if g>1.0:
        cryst_steps['PropertyT']=steps[i]; break

# Val threshold
for i,v in enumerate(vals):
    if v<1.0:
        cryst_steps['val<1.0']=steps[i]; break
    if v<0.5:
        cryst_steps['val<0.5']=steps[i]; break
    if v<0.25:
        cryst_steps['val<0.25']=steps[i]; break

# Group by step window
from collections import defaultdict
groups=defaultdict(list)
for name,step in sorted(cryst_steps.items(),key=lambda x:x[1]):
    window=(step//20)*20
    groups[window].append(name)

print(f"  Step window  Structures forming together")
print("  "+"-"*50)
for window,names in sorted(groups.items()):
    print(f"  {window:>4}-{window+20:<4}   {', '.join(names)}")

print(f"""
  KEY INSIGHT:
  Structures that form in the SAME window can potentially be
  built in a SINGLE algebraic operation via homological algebra.

  The question for building them in one go:
  Can we compute the final state of these simultaneous structures
  directly from:
    - The cascade targets {{S_l}} (known algebraically)
    - The teacher embeddings (known from teacher training)
    - The corpus statistics (known without training)
  
  without running the gradient descent that produces them?

  If the Serre levels, Lie brackets, and Property T gap all
  crystallize in the same window, they are LINKED — one algebraic
  object determines all three. Finding that object gives us the
  single-step construction.
""")
