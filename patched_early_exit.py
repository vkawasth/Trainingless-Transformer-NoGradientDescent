#!/usr/bin/env python3
"""
Patched Early Exit + Reverse Hironaka Helper
==============================================
ARCHITECTURE:
  - Layers 0-L_ATT:   Set algebraically by Serre/prime cascade
                       (H^2, H^3 obstructions pre-resolved)
  - Layers L_ATT-L20: Prime path patches handle H^5, H^6
  - Layer L20:        Early exit point — sheaf locally flat here
  - Layers L20-L23:   H^7, H^8 obstructions — gradient flows HERE ONLY

HELPER STRATEGY:
  1. Initialize ALL layers from cascade (algebraic, no gradient)
  2. FREEZE layers 0 to L_freeze (already flat, from cascade)
  3. Only train L_freeze to N_STU (the gluing failure region)
  4. Use early exit loss at intermediate layers as auxiliary signal
     (helps gradient know when local flatness is achieved)
  5. Stop when early exit quality plateaus (orientation freeze)

This is gradient descent as a helper — minimal work:
  - Frozen: layers where sheaf is already consistent
  - Active: only the H^7/H^8 resolution region
  - Early exit: diagnostic that tells gradient when to stop

In the 6-layer student:
  - Freeze blocks 0-3 (cascade handles H^2-H^6)
  - Train only blocks 4-5 (H^7, H^8 resolution)
  - Early exit after block 3 as auxiliary loss weight

PREDICTION:
  Full freeze (blocks 0-3) + train (blocks 4-5) = ~50 CE steps
  vs standard 200 CE steps = 4x further reduction
  Combined with prime cascade: 6x × 4x = 24x total reduction
"""
import json, math, time, warnings, itertools
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_LAYERS_T=24; N_STU=6; BATCH=8; SEQ=64; LR=3e-4; PROJ=48
L_ATT=14

print(f"\n{'='*65}")
print(f"  PATCHED EARLY EXIT + REVERSE HIRONAKA HELPER")
print(f"  Minimal gradient descent on H^7/H^8 only")
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

class EarlyExitLM(nn.Module):
    """
    6-layer student with early exit heads at each block.
    - Blocks 0..freeze-1: FROZEN (cascade initialised, H^2-H^6 resolved)
    - Blocks freeze..N-1: TRAINABLE (H^7, H^8 resolution)
    - Early exit at block freeze-1: auxiliary loss signal
    """
    def __init__(self,d,nh,nl,freeze_after=3):
        super().__init__()
        self.te=nn.Embedding(VOCAB,d); self.pe=nn.Embedding(512,d)
        self.blocks=nn.ModuleList([Block(d,nh) for _ in range(nl)])
        self.ln_f=nn.LayerNorm(d)
        self.head=nn.Linear(d,VOCAB,bias=False); self.head.weight=self.te.weight
        # Early exit head at freeze point
        self.exit_ln=nn.LayerNorm(d)
        self.exit_head=nn.Linear(d,VOCAB,bias=False)
        self.freeze_after=freeze_after
        nn.init.normal_(self.te.weight,std=0.02)
        nn.init.normal_(self.pe.weight,std=0.02)

    def forward(self,x,y=None,alpha_exit=0.1):
        h=self.te(x)+self.pe(torch.arange(x.shape[1]))
        h_exit=None
        for i,b in enumerate(self.blocks):
            h=b(h)
            if i==self.freeze_after-1:
                h_exit=h.detach()  # detach: early exit doesn't flow into frozen layers
        logits=self.head(self.ln_f(h))
        if y is None:
            return logits,None
        loss_main=F.cross_entropy(logits.view(-1,VOCAB),y.view(-1))
        if h_exit is not None and alpha_exit>0:
            # Early exit auxiliary loss — measures H^6 resolution quality
            logits_exit=self.exit_head(self.exit_ln(h_exit))
            loss_exit=F.cross_entropy(logits_exit.view(-1,VOCAB),y.view(-1))
            loss=loss_main+alpha_exit*loss_exit
        else:
            loss=loss_main
        return logits,loss,loss_main if h_exit is not None else loss

    def freeze_blocks(self,n_freeze):
        """Freeze first n_freeze blocks."""
        for i in range(min(n_freeze,len(self.blocks))):
            for p in self.blocks[i].parameters():
                p.requires_grad_(False)

    def unfreeze_all(self):
        for p in self.parameters(): p.requires_grad_(True)

    def hidden_states(self,x):
        hs=[]; h=self.te(x)+self.pe(torch.arange(x.shape[1])); hs.append(h.detach())
        for b in self.blocks: h=b(h); hs.append(h.detach())
        return hs

def clr(s,total=300,warmup=50):
    if s<=warmup: return LR*s/warmup
    return LR*0.5*(1+math.cos(math.pi*(s-warmup)/(total-warmup)))

def eval_val(model,n=60):
    model.eval(); ls=[]
    with torch.no_grad():
        for _ in range(n):
            x,y=get_batch('val')
            out=model(x,y,alpha_exit=0.0)
            l=out[2] if len(out)==3 else out[1]
            ls.append(float(l))
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
def lift_to_d(C,U,scale=0.01):
    UU=U@U.T
    return (U@C@U.T+(np.eye(D)-UU)*scale).astype(np.float32)
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

# ════════════════════════════════════════════════════
# Train teacher + extract cascade
# ════════════════════════════════════════════════════
print("Training teacher (300 steps)...")
torch.manual_seed(42)
from torch.nn import Module
teacher_lm=EarlyExitLM(D,N_HEADS,N_LAYERS_T,freeze_after=N_LAYERS_T)
# Use standard LM for teacher training
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
val_t_=float(np.mean([teacher(*get_batch('val'))[1].item() for _ in range(30)]))
print(f"  Teacher val={val_t_:.4f}\n")

# Extract Jacobians
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
Js=[np.mean(J_acc[l],axis=0) for l in range(N_LAYERS_T)]
Us=[np.mean(U_acc[l],axis=0) for l in range(N_LAYERS_T)]
J14=Js[L_ATT]; U14=Us[L_ATT]

# Prime cascade
att_basin=[l for l in range(8,21) if N(Js[l]-np.eye(ma))<0.75]
combos=list(itertools.combinations(att_basin,6))
scored=sorted([(c,N(mu6(*[Js[i] for i in c]))) for c in combos],key=lambda x:-x[1])
cascade_prime=[mu6(*[Js[i] for i in c])/max(N(mu6(*[Js[i] for i in c])),1e-8)
               for c,_ in scored[:N_STU]]

# Serre cascade
cascade_serre=[]
for l in range(1,N_STU+1):
    C=Js[min(L_ATT+l,N_LAYERS_T-1)].copy()
    for _ in range(l): C=comm(J14,C)
    cascade_serre.append(C/max(N(C),1e-8))

def inject_cascade(stu, cascade):
    with torch.no_grad():
        stu.te.weight.copy_(teacher.te.weight)
        stu.pe.weight.copy_(teacher.pe.weight)
        stu.ln_f.weight.copy_(teacher.ln_f.weight)
        stu.ln_f.bias.copy_(teacher.ln_f.bias)
        # Exit head: copy final layer norm + head
        stu.exit_ln.weight.data.fill_(1.0)
        stu.exit_ln.bias.data.fill_(0.0)
        stu.exit_head.weight.copy_(teacher.head.weight)
        for l in range(N_STU):
            W_d=lift_to_d(cascade[l],U14,scale=0.01)
            W_t=torch.tensor(W_d,dtype=torch.float32)
            stu.blocks[l].attn.WK.weight.copy_(W_t)
            stu.blocks[l].attn.WQ.weight.copy_(W_t.T)
            stu.blocks[l].attn.WV.weight.copy_(teacher.blocks[L_ATT].attn.WV.weight)
            stu.blocks[l].attn.op.weight.copy_(teacher.blocks[L_ATT].attn.op.weight)
            stu.blocks[l].ff.g.weight.copy_(teacher.blocks[L_ATT].ff.g.weight)
            stu.blocks[l].ff.v.weight.copy_(teacher.blocks[L_ATT].ff.v.weight)
            stu.blocks[l].ff.o.weight.copy_(teacher.blocks[L_ATT].ff.o.weight)

# ════════════════════════════════════════════════════
# STUDENT EXPERIMENTS
# ════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("STUDENT EXPERIMENTS")
print("  A: Serre + 200CE (baseline)")
print("  B: Prime + 200CE (standard)")
print("  C: Prime + freeze(0-3) + train(4-5) only [H^7/H^8]")
print("  D: Prime + freeze(0-3) + exit loss + train(4-5) [with helper]")
print("  E: Prime + freeze(0-3) + 50CE only [minimal]")
print("="*65)

def run_standard(cascade, label, steps=200):
    """Standard training without freeze."""
    torch.manual_seed(99)
    stu=EarlyExitLM(D,N_HEADS,N_STU,freeze_after=N_STU)
    inject_cascade(stu,cascade)
    v0=eval_val(stu,n=20); print(f"\n  [{label}] zero-shot={v0:.4f}")
    opt_s=torch.optim.AdamW(stu.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
    ck={}
    for step in range(1,steps+1):
        for pg in opt_s.param_groups: pg['lr']=clr(step,steps)
        stu.train(); x,y=get_batch()
        _,loss,_=stu(x,y,alpha_exit=0.0)
        opt_s.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(stu.parameters(),1.0); opt_s.step()
        if step in [25,50,75,100,125,150,200]:
            v=eval_val(stu,n=20); ck[step]=v
            print(f"  [{label}] step {step:>4}  val={v:.4f}"
                  f"{' ✓' if v<val_t_ else ''}")
    return eval_val(stu),ck

def run_frozen(cascade, label, freeze_n=3, alpha_exit=0.0, steps=200):
    """Train only blocks freeze_n..N-1. Gradient helper on H^7/H^8."""
    torch.manual_seed(99)
    stu=EarlyExitLM(D,N_HEADS,N_STU,freeze_after=freeze_n)
    inject_cascade(stu,cascade)
    stu.freeze_blocks(freeze_n)
    v0=eval_val(stu,n=20)
    n_trainable=sum(p.numel() for p in stu.parameters() if p.requires_grad)
    n_total=sum(p.numel() for p in stu.parameters())
    print(f"\n  [{label}] zero-shot={v0:.4f}  "
          f"trainable={n_trainable}/{n_total} params "
          f"({100*n_trainable/n_total:.1f}%)")
    opt_s=torch.optim.AdamW(
        [p for p in stu.parameters() if p.requires_grad],
        lr=LR,betas=(0.9,0.95),weight_decay=0.1)
    ck={}
    for step in range(1,steps+1):
        for pg in opt_s.param_groups: pg['lr']=clr(step,steps)
        stu.train(); x,y=get_batch()
        _,loss,loss_main=stu(x,y,alpha_exit=alpha_exit)
        opt_s.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(stu.parameters(),1.0); opt_s.step()
        if step in [25,50,75,100,125,150,200]:
            v=eval_val(stu,n=20); ck[step]=v
            print(f"  [{label}] step {step:>4}  val={v:.4f}"
                  f"{' ✓' if v<val_t_ else ''}")
    return eval_val(stu),ck

vA,ckA=run_standard(cascade_serre,"A-Serre-std")
vB,ckB=run_standard(cascade_prime,"B-Prime-std")
vC,ckC=run_frozen(cascade_prime,"C-Prime-freeze3-noExit",
                   freeze_n=3,alpha_exit=0.0,steps=200)
vD,ckD=run_frozen(cascade_prime,"D-Prime-freeze3-withExit",
                   freeze_n=3,alpha_exit=0.2,steps=200)
vE,ckE=run_frozen(cascade_prime,"E-Prime-freeze3-50CE",
                   freeze_n=3,alpha_exit=0.0,steps=50)

print(f"\n{'='*65}")
print("  PATCHED EARLY EXIT RESULTS")
print("="*65)
print(f"\n  CONVERGENCE:")
print(f"  {'step':>6}  {'A-Serre':>8}  {'B-Prime':>8}  "
      f"{'C-frz3':>8}  {'D-frz+ex':>9}  {'E-50CE':>8}")
for s in [25,50,75,100,125,150,200]:
    a=ckA.get(s); b=ckB.get(s); c=ckC.get(s)
    d=ckD.get(s); e=ckE.get(s)
    row=f"  {s:>6}"
    for v in [a,b,c,d,e]:
        row+=f"  {v:>8.4f}" if v else f"  {'---':>8}"
    best=min((v for v in [b,c,d,e] if v),default=99)
    if a and best<a-0.005: row+=" ←"
    print(row)

# Compute layer-steps for each
print(f"""
  FINAL + COMPUTE:
    Teacher:                 val={val_t_:.4f}  ({300*N_LAYERS_T} layer-steps)
    A (Serre+200CE):         val={vA:.4f}  (1200 layer-steps, 6x)
    B (Prime+200CE):         val={vB:.4f}  (1200 layer-steps, 6x)
    C (Prime+frz3+200CE):    val={vC:.4f}  ({200*(N_STU-3)} active layer-steps)
    D (Prime+frz3+exit+200): val={vD:.4f}  ({200*(N_STU-3)} active layer-steps)
    E (Prime+frz3+50CE):     val={vE:.4f}  ({50*(N_STU-3)} active layer-steps)

  ACTIVE LAYER-STEPS (trainable blocks only):
    C/D: {200*(N_STU-3)} = {200} steps × {N_STU-3} active blocks
    E:   {50*(N_STU-3)}  = {50} steps × {N_STU-3} active blocks

  IF C ≈ B: freezing pre-resolved layers costs nothing in quality.
    Only H^7/H^8 (last 2 blocks) need gradient.
    Active compute = {200*(N_STU-3)} vs 1200 = {1200//(200*(N_STU-3))}x further reduction.

  IF D < C: early exit auxiliary loss helps the gradient navigate H^7/H^8.
    The exit signal tells the gradient when the early layers are flat enough.

  IF E beats teacher: 50 steps on 3 blocks = {50*3} layer-steps.
    Combined with 6x cascade reduction: total = {300*N_LAYERS_T//50//3}x reduction.
""")
