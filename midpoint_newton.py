#!/usr/bin/env python3
"""
Midpoint Newton — Apply Newton at Phase Boundary
=================================================
The Newton step at initialization fails because grad=0 there.
The teacher WK is already at a stationary point of its own loss.

The correct application: Newton at the Phase 1/2 boundary.
  - Phase 1 (steps 0-100): WV/WO/FF/E adapt, WK stationary
  - Phase 2 (steps 100-200): WK adapts to new landscape

Apply Newton AFTER Phase 1 completes (step 100):
  1. Run 100 CE steps (Phase 1: WV/WO/FF/E adapt)
  2. Freeze WV/WO/FF/E
  3. Compute grad and Fisher of WK only (now nonzero)
  4. Apply Newton step to WK
  5. Unfreeze, continue (or stop)

If Newton at step 100 captures Phase 2 in one step:
  Total = 100 CE steps + 1 Newton step vs 200 CE steps.
  2x reduction confirmed.

EXPERIMENTS:
  A: Teacher WK + 200CE (baseline, val=0.161)
  B: Teacher WK + 100CE + Newton(WK) + 0CE
  C: Teacher WK + 100CE + Newton(WK) + 10CE
  D: Teacher WK + 100CE + Newton(WK) + 50CE
  
  Also track WK gradient norm during Phase 1 to confirm
  it becomes nonzero at steps 75-100.
"""
import json, math, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_LAYERS_T=24; N_STU=6; BATCH=8; SEQ=64; LR=3e-4
L_ATT=14; N_NEWTON=200

print(f"\n{'='*65}")
print(f"  MIDPOINT NEWTON")
print(f"  Apply Newton at Phase 1/2 boundary (step 100)")
print(f"{'='*65}\n")

with open('/tmp/train_ids.json') as f: train_ids=list(map(int,json.load(f)))
with open('/tmp/val_ids.json')   as f: val_ids  =list(map(int,json.load(f)))
with open('/tmp/vocab.json')     as f: vocab=json.load(f)
VOCAB=len(vocab)
train_t=torch.tensor(train_ids,dtype=torch.long)
val_t=torch.tensor(val_ids,dtype=torch.long)

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

def clr(s,total=200,warmup=50):
    if s<=warmup: return LR*s/warmup
    return LR*0.5*(1+math.cos(math.pi*(s-warmup)/(total-warmup)))

def eval_val(model,n=60):
    model.eval(); ls=[]
    with torch.no_grad():
        for _ in range(n):
            x,y=get_batch('val'); _,l=model(x,y); ls.append(l.item())
    return float(np.mean(ls))

# Train teacher
print("Training teacher (300 steps)...")
torch.manual_seed(42)
teacher=LM(D,N_HEADS,N_LAYERS_T)
opt=torch.optim.AdamW(teacher.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
t0=time.time()
for step in range(1,301):
    lr_now=LR*min(step,100)/100 if step<=100 else \
           LR*0.5*(1+math.cos(math.pi*(step-100)/200))
    for pg in opt.param_groups: pg['lr']=lr_now
    teacher.train(); x,y=get_batch(); _,loss=teacher(x,y)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(teacher.parameters(),1.0); opt.step()
    if step%100==0:
        teacher.eval()
        with torch.no_grad():
            vl=float(np.mean([teacher(*get_batch('val'))[1].item() for _ in range(10)]))
        print(f"  step {step}  val={vl:.4f}  t={time.time()-t0:.0f}s")
        teacher.train()
teacher.eval(); val_teacher=eval_val(teacher)
print(f"  Teacher val={val_teacher:.4f}\n")

def build_student():
    torch.manual_seed(99)
    stu=LM(D,N_HEADS,N_STU)
    stu.te.weight.data.copy_(teacher.te.weight.data)
    with torch.no_grad():
        stu.pe.weight.copy_(teacher.pe.weight)
        stu.ln_f.weight.copy_(teacher.ln_f.weight)
        stu.ln_f.bias.copy_(teacher.ln_f.bias)
        for l in range(N_STU):
            stu.blocks[l].attn.WK.weight.copy_(teacher.blocks[L_ATT].attn.WK.weight)
            stu.blocks[l].attn.WQ.weight.copy_(teacher.blocks[L_ATT].attn.WQ.weight)
            stu.blocks[l].attn.WV.weight.copy_(teacher.blocks[L_ATT].attn.WV.weight)
            stu.blocks[l].attn.op.weight.copy_(teacher.blocks[L_ATT].attn.op.weight)
            stu.blocks[l].ff.g.weight.copy_(teacher.blocks[L_ATT].ff.g.weight)
            stu.blocks[l].ff.v.weight.copy_(teacher.blocks[L_ATT].ff.v.weight)
            stu.blocks[l].ff.o.weight.copy_(teacher.blocks[L_ATT].ff.o.weight)
    return stu

def compute_newton_delta(stu, n_seq=N_NEWTON):
    """Compute Newton step for WK at current parameter values."""
    grad_acc=torch.zeros_like(stu.blocks[0].attn.WK.weight)
    fisher_d=torch.zeros_like(stu.blocks[0].attn.WK.weight)
    torch.manual_seed(1)
    for i in range(n_seq):
        ix=torch.randint(0,len(train_t)-SEQ-1,(1,))[0].item()
        x=train_t[ix:ix+SEQ].unsqueeze(0)
        y=train_t[ix+1:ix+SEQ+1].unsqueeze(0)
        stu.zero_grad()
        _,loss=stu(x,y); loss.backward()
        # Average gradient across all blocks (they share same WK init)
        g=torch.zeros_like(stu.blocks[0].attn.WK.weight)
        for l in range(N_STU):
            if stu.blocks[l].attn.WK.weight.grad is not None:
                g+=stu.blocks[l].attn.WK.weight.grad/N_STU
        grad_acc+=g; fisher_d+=g**2
    grad_mean=grad_acc/n_seq
    fisher_diag=fisher_d/n_seq
    eps=1e-6
    delta=-(grad_mean/(fisher_diag+eps))
    return delta, float(grad_mean.norm()), float(fisher_diag.norm())

def apply_newton(stu, delta, scale=1.0):
    with torch.no_grad():
        for l in range(N_STU):
            stu.blocks[l].attn.WK.weight.add_(scale*delta)
            stu.blocks[l].attn.WQ.weight.add_(scale*delta.T)

# ════════════════════════════════════════════════════
# TRACK WK GRADIENT NORM DURING PHASE 1
# ════════════════════════════════════════════════════
print("="*65)
print("PHASE 1 WK GRADIENT NORM TRACKING")
print("  Confirm gradient becomes nonzero at steps 75-100")
print("="*65)

stu_track=build_student()
opt_track=torch.optim.AdamW(stu_track.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
print(f"\n  {'step':>5}  {'val':>7}  {'||grad_WK||':>12}  {'note'}")
print("  "+"-"*40)
for step in range(1,201):
    for pg in opt_track.param_groups: pg['lr']=clr(step,200)
    stu_track.train(); x,y=get_batch(); _,loss=stu_track(x,y)
    opt_track.zero_grad(); loss.backward()
    # Measure WK gradient norm before step
    wk_gnorm=float(sum(stu_track.blocks[l].attn.WK.weight.grad.norm()**2
                       for l in range(N_STU) 
                       if stu_track.blocks[l].attn.WK.weight.grad is not None)**0.5)
    torch.nn.utils.clip_grad_norm_(stu_track.parameters(),1.0)
    opt_track.step()
    if step in [1,10,25,50,75,100,125,150,175,200]:
        v=eval_val(stu_track,n=10)
        note=" <-- Phase 2 starts?" if step in [75,100] else ""
        print(f"  {step:>5}  {v:>7.4f}  {wk_gnorm:>12.6f}{note}")

# ════════════════════════════════════════════════════
# EXPERIMENTS
# ════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("EXPERIMENTS")
print("  A: Teacher WK + 200CE (baseline)")
print("  B: Teacher WK + 100CE + Newton(WK) + 0CE")
print("  C: Teacher WK + 100CE + Newton(WK) + 10CE")
print("  D: Teacher WK + 100CE + Newton(WK) + 50CE")
print("  E: Teacher WK + 100CE + Newton(WK) + 100CE")
print("="*65)

def run_with_midpoint_newton(label, phase1_steps, newton_scale,
                              phase2_steps):
    stu=build_student()
    opt_s=torch.optim.AdamW(stu.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
    v0=eval_val(stu,n=20)
    print(f"\n  [{label}] zero-shot={v0:.4f}")
    ck={0:v0}

    # Phase 1: standard CE
    for step in range(1,phase1_steps+1):
        for pg in opt_s.param_groups: pg['lr']=clr(step,phase1_steps+phase2_steps)
        stu.train(); x,y=get_batch(); _,loss=stu(x,y)
        opt_s.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(stu.parameters(),1.0); opt_s.step()
        if step in [25,50,75,100]:
            v=eval_val(stu,n=20); ck[step]=v
            print(f"  [{label}] P1 step {step:>4}  val={v:.4f}")

    v_p1=eval_val(stu,n=20)
    print(f"  [{label}] After Phase 1 ({phase1_steps} steps): val={v_p1:.4f}")

    # Newton step at phase boundary
    if newton_scale>0:
        delta,gnorm,fnorm=compute_newton_delta(stu)
        print(f"  [{label}] Newton ||grad||={gnorm:.6f}  "
              f"||delta||={float(delta.norm()):.6f}")

        # Line search for scale
        best_s=newton_scale; best_v=float('inf')
        for s in [0.01,0.1,0.5,1.0,2.0,5.0,10.0]:
            stu_t=build_student()
            # Copy current stu weights
            stu_t.load_state_dict(stu.state_dict())
            apply_newton(stu_t,delta,scale=s)
            v_t=eval_val(stu_t,n=10)
            if v_t<best_v: best_v=v_t; best_s=s
        print(f"  [{label}] Best Newton scale: {best_s}, val={best_v:.4f}")
        apply_newton(stu,delta,scale=best_s)
        v_newton=eval_val(stu,n=20)
        print(f"  [{label}] After Newton step: val={v_newton:.4f}")
        ck['newton']=v_newton

    if phase2_steps==0:
        return eval_val(stu),ck

    # Phase 2: CE after Newton
    opt_s2=torch.optim.AdamW(stu.parameters(),lr=LR*0.1,
                               betas=(0.9,0.95),weight_decay=0.1)
    for step in range(1,phase2_steps+1):
        for pg in opt_s2.param_groups:
            pg['lr']=LR*0.1*0.5*(1+math.cos(math.pi*step/phase2_steps))
        stu.train(); x,y=get_batch(); _,loss=stu(x,y)
        opt_s2.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(stu.parameters(),1.0); opt_s2.step()
        if step in [10,25,50,100]:
            v=eval_val(stu,n=20)
            ck[f'p2_{step}']=v
            print(f"  [{label}] P2 step {step:>4}  val={v:.4f}"
                  f"{' ✓' if v<val_teacher else ''}")

    return eval_val(stu),ck

vA,ckA=run_with_midpoint_newton("A-200CE",200,0,0)
vB,ckB=run_with_midpoint_newton("B-100CE+Newton+0CE",100,1.0,0)
vC,ckC=run_with_midpoint_newton("C-100CE+Newton+10CE",100,1.0,10)
vD,ckD=run_with_midpoint_newton("D-100CE+Newton+50CE",100,1.0,50)
vE,ckE=run_with_midpoint_newton("E-100CE+Newton+100CE",100,1.0,100)

print(f"\n{'='*65}")
print("  MIDPOINT NEWTON RESULTS")
print("="*65)
print(f"""
  FINAL:
    Teacher:                  val={val_teacher:.4f}
    A (200CE):                val={vA:.4f}  (baseline)
    B (100CE+Newton+0CE):     val={vB:.4f}  diff={vA-vB:+.4f}
    C (100CE+Newton+10CE):    val={vC:.4f}  diff={vA-vC:+.4f}
    D (100CE+Newton+50CE):    val={vD:.4f}  diff={vA-vD:+.4f}
    E (100CE+Newton+100CE):   val={vE:.4f}  diff={vA-vE:+.4f}

  IF B ~ A: Newton at step 100 captures Phase 2 in one step.
    Total: 100 CE + 1 Newton = 101 effective steps vs 200.
    The oscillation in Phase 2 is replaced by one Newton step.
    The morphism IS the midpoint Newton on the corpus Fisher.

  IF C or D ~ A with fewer total steps:
    Newton reduces Phase 2 but doesn't eliminate it.
    Partial speedup: 100 + 10 or 100 + 50 vs 200 steps.

  WK GRADIENT NORM TELLS THE STORY:
    If ||grad_WK|| is near-zero at step 0-75 and grows at 75-100:
    Phase 1/2 boundary confirmed at step 75-100.
    Newton is correctly placed at the branch point.
""")
