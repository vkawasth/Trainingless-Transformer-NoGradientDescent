#!/usr/bin/env python3
"""
Teacher A_inf Profiler
========================
Profile which H^k obstructions are being resolved at each training step.

REVERSE HIRONAKA PICTURE:
  Start: random weights (all H^k nonzero, complex maximally singular)
  Goal:  trained teacher (H^k decreasing, path navigates obstructions)
  
  We don't need the optimal path — any path works.
  The path is the 200 CE steps navigating H^2 -> H^3 -> H^5 -> H^6 -> H^7 -> H^8

FUNCTOR CONSTRUCTION:
  Phi: C_corpus -> C_ctx
  At each training step, the functor changes.
  Track: when does each H^k obstruction get resolved?
  
  Resolution criterion:
    ||mu_k(step)|| / ||mu_k(step=0)|| < 0.5  (50% reduction)
  
  The step at which H^k resolves = the training step that crosses
  the obstruction at level k.

PATCH DIAGNOSTICS:
  For each prime path patch P:
    Track the mu6(P, step) norm during training.
    When does the patch "crystallize" (mu6 stops changing)?
    This is the step when the local section s_P is determined.
"""
import json, math, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_LAYERS_T=24; BATCH=8; SEQ=64; LR=3e-4; PROJ=48
L_ATT=14; PROFILE_EVERY=25

print(f"\n{'='*65}")
print(f"  TEACHER A_INF PROFILER")
print(f"  Tracking H^k obstruction resolution during training")
print(f"  Reverse Hironaka: when does each obstruction resolve?")
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

def clr(s,total=300,warmup=100):
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
def l3(a,b,c): return comm(comm(a,b),c)-comm(a,comm(b,c))
def N(A): return float(np.linalg.norm(A))

def mu4(a,b,c,d):
    return -(comm(l3(a,b,c),d)-l3(comm(a,b),c,d)+l3(a,comm(b,c),d)
            -l3(a,b,comm(c,d))+comm(a,l3(b,c,d)))
def mu5(a,b,c,d,e):
    return -(l3(l3(a,b,c),d,e)-l3(a,l3(b,c,d),e)+l3(a,b,l3(c,d,e))
            +comm(mu4(a,b,c,d),e)+comm(a,mu4(b,c,d,e))
            -mu4(comm(a,b),c,d,e)+mu4(a,b,c,comm(d,e)))
def mu6(a,b,c,d,e,f):
    m5ab=mu5(a,b,c,d,e); m5bc=mu5(b,c,d,e,f)
    m4ab=mu4(a,b,c,d); m4bc=mu4(b,c,d,e); m4cd=mu4(c,d,e,f)
    m3ab=l3(a,b,c); m3bc=l3(b,c,d); m3cd=l3(c,d,e); m3de=l3(d,e,f)
    return -(comm(m5ab,f)-comm(a,m5bc)+l3(m4ab,e,f)-l3(a,m4bc,f)
            +l3(a,b,m4cd)+mu4(m3ab,d,e,f)-mu4(a,m3bc,e,f)
            +mu4(a,b,m3cd,f)-mu4(a,b,c,m3de))
def mu7(a,b,c,d,e,f,g):
    total=np.zeros_like(a)
    total+=comm(mu6(a,b,c,d,e,f),g); total-=comm(a,mu6(b,c,d,e,f,g))
    total+=l3(mu5(a,b,c,d,e),f,g); total-=l3(a,mu5(b,c,d,e,f),g)
    total+=l3(a,b,mu5(c,d,e,f,g))
    total+=mu4(mu4(a,b,c,d),e,f,g); total-=mu4(a,mu4(b,c,d,e),f,g)
    total+=mu4(a,b,mu4(c,d,e,f),g); total-=mu4(a,b,c,mu4(d,e,f,g))
    total+=mu5(l3(a,b,c),d,e,f,g); total-=mu5(a,l3(b,c,d),e,f,g)
    total+=mu5(a,b,l3(c,d,e),f,g); total-=mu5(a,b,c,l3(d,e,f),g)
    total+=mu5(a,b,c,d,l3(e,f,g))
    total+=mu6(comm(a,b),c,d,e,f,g); total-=mu6(a,comm(b,c),d,e,f,g)
    total+=mu6(a,b,comm(c,d),e,f,g); total-=mu6(a,b,c,comm(d,e),f,g)
    total+=mu6(a,b,c,d,comm(e,f),g); total-=mu6(a,b,c,d,e,comm(f,g))
    return -total

def extract_ainf_snapshot(model, x_ref, pos, m):
    """Extract mu2..mu7 at current training state."""
    model.eval()
    with torch.no_grad(): hs=model.hidden_states(x_ref); hs=[h[0] for h in hs]
    Js=[]; ma=None
    for l in range(N_LAYERS_T):
        J,_=layer_jac(model.blocks[l],hs[l],pos,m)
        Js.append(J)
        if ma is None: ma=J.shape[0]

    # Use attractor neighborhood L11-L17 (7 layers for mu7)
    att=[Js[min(L_ATT+l,N_LAYERS_T-1)] for l in range(-3,4)]
    a,b,c,d,e,f,g=att

    return {
        'mu2': N(comm(a,b)),
        'mu3': N(l3(a,b,c)),
        'mu4': N(mu4(a,b,c,d)),
        'mu5': N(mu5(a,b,c,d,e)),
        'mu6': N(mu6(a,b,c,d,e,f)),
        'mu7': N(mu7(a,b,c,d,e,f,g)),
        # Patch-specific mu6 for top prime paths
        'mu6_patch1': N(mu6(Js[min(10,N_LAYERS_T-1)],
                            Js[min(13,N_LAYERS_T-1)],
                            Js[min(14,N_LAYERS_T-1)],
                            Js[min(15,N_LAYERS_T-1)],
                            Js[min(16,N_LAYERS_T-1)],
                            Js[min(17,N_LAYERS_T-1)])),
        'Js': Js,
    }

# ════════════════════════════════════════════════════
# Profile teacher training
# ════════════════════════════════════════════════════
print("Training teacher with A_inf profiling every 25 steps...")
torch.manual_seed(42)
teacher=LM(D,N_HEADS,N_LAYERS_T)
opt=torch.optim.AdamW(teacher.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
pos=SEQ//2; m=min(PROJ,SEQ,D)

x_ref,_=get_batch('val'); x_ref=x_ref[0:1]
torch.manual_seed(0)

snapshots=[]

# Print header
print(f"\n  {'step':>5}  {'val':>7}  {'mu2':>7}  {'mu3':>7}  "
      f"{'mu4':>7}  {'mu5':>7}  {'mu6':>7}  {'mu7':>7}  {'patch1':>8}")
print("  "+"-"*72)

t0=time.time()
for step in range(0,301):
    if step>0:
        for pg in opt.param_groups: pg['lr']=clr(step)
        teacher.train(); x,y=get_batch(); _,loss=teacher(x,y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(teacher.parameters(),1.0); opt.step()

    if step%PROFILE_EVERY==0:
        vl=eval_val(teacher,n=20)
        snap=extract_ainf_snapshot(teacher,x_ref,pos,m)
        snap['step']=step; snap['val']=vl
        snapshots.append(snap)
        print(f"  {step:>5}  {vl:>7.4f}  "
              f"{snap['mu2']:>7.4f}  {snap['mu3']:>7.4f}  "
              f"{snap['mu4']:>7.4f}  {snap['mu5']:>7.4f}  "
              f"{snap['mu6']:>7.4f}  {snap['mu7']:>7.4f}  "
              f"{snap['mu6_patch1']:>8.5f}")

print(f"\n  Training complete: {time.time()-t0:.0f}s")

# ════════════════════════════════════════════════════
# ANALYSIS: Obstruction resolution order
# ════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("  REVERSE HIRONAKA: OBSTRUCTION RESOLUTION ORDER")
print("="*65)

initial=snapshots[0]
final=snapshots[-1]

print(f"\n  H^k reduction from step 0 to step 300:")
print(f"  {'k':>3}  {'initial':>9}  {'final':>9}  {'reduction':>11}  "
      f"{'50% step':>10}")
print("  "+"-"*48)

for k in ['mu2','mu3','mu4','mu5','mu6','mu7']:
    v0=initial[k]; vf=final[k]
    reduction=v0/max(vf,1e-10)
    # Find step where 50% reduction achieved
    half_step=next((s['step'] for s in snapshots
                    if s[k]<v0*0.5), None)
    print(f"  H^{k[-1]}  {v0:>9.5f}  {vf:>9.5f}  {reduction:>10.2f}x  "
          f"{'step '+str(half_step) if half_step else 'not reached':>10}")

print(f"\n  PATCH CRYSTALLIZATION (mu6_patch1 stabilization):")
patch_vals=[s['mu6_patch1'] for s in snapshots]
steps=[s['step'] for s in snapshots]
for i in range(1,len(patch_vals)):
    if abs(patch_vals[i]-patch_vals[i-1])<0.00001:
        print(f"  Patch 1 crystallizes at step {steps[i]} "
              f"(change < 1e-5)")
        break

print(f"\n  FUNCTOR PATH (which H^k resolves at which step):")
print(f"  This is the Reverse Hironaka blow-down sequence.")
print(f"  Starting from singularities, the functor Phi resolves them in order:")
print()

resolution_steps={}
for k in ['mu2','mu3','mu5','mu6','mu7']:
    v0=initial[k]
    for s in snapshots:
        if s[k]<v0*0.5:
            resolution_steps[k]=s['step']
            break

for k,step in sorted(resolution_steps.items(),key=lambda x:x[1]):
    print(f"  Step {step:>4}: H^{k[-1]} resolves ({k})")

print(f"""
  INTERPRETATION:
  The functor Phi: C_corpus -> C_ctx navigates the obstructions
  in the order shown above. Each step is a "blow-down" in the
  Reverse Hironaka sense: a local singularity being resolved.

  Gradient descent does not need to know this order in advance.
  It discovers the path by following the CE loss gradient.
  
  The REVERSE SHEAF CONSTRUCTION can precompute this order:
    1. Given the architecture, predict which H^k resolves when
       (from the Koszul structure and prime path data)
    2. Initialize the functor at the H^6/H^7 level (prime paths)
    3. Flow gradient only to resolve H^7 and H^8
    4. The earlier obstructions (H^2, H^3, H^5, H^6) are
       already pre-resolved by the prime path initialization

  The remaining irreducible cost = steps needed to resolve H^7+H^8
  with corpus data. This is the true minimum training compute.
""")
