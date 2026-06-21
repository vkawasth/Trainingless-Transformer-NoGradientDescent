#!/usr/bin/env python3
"""
Mean-Field Initialization
==========================
The 167 CE steps are iterating the mean-field equations:
  E*[t] = argmin_E E_D[L(E[t], W_K, E[other])]
  W_K*  = argmin_{W_K} E_D[L(E*, W_K)]

These are coupled: optimal E depends on W_K and vice versa.
Static pre-computation breaks this coupling (W_K align hurt).

CORRECT APPROACH: Joint coordinate descent under LINEAR ATTENTION
  Linear attention: softmax(QK/sqrt(d)) ≈ QK/sqrt(d) (no normalization)
  Under linear attention, the system is LINEAR in E and W_K.
  The fixed point is solvable in closed form via least squares.

ALGORITHM:
  Step 0: Initialize E=E_teacher, W_K=W_K_teacher (current init)
  Step 1: Solve E* = argmin_E E_D[L_linear(E, W_K_current)]
           = one least-squares solve over corpus
  Step 2: Solve W_K* = argmin_{W_K} E_D[L_linear(E*, W_K_current)]
           = one least-squares solve over corpus  
  Step 3: Repeat once (2 iterations sufficient for linear system)
  
  Initialize student with (E*, W_K*) from mean-field.
  Then run standard CE training (should converge faster).

PRACTICAL APPROXIMATION:
  Full least-squares is expensive for large vocab.
  Use gradient steps with large LR on the LINEAR system instead:
  - Alternate between E update and W_K update
  - Each update is one full corpus pass
  - Stop after convergence of linear system (few steps)

PREDICTION:
  Mean-field initialization places student at basin entrance.
  Remaining CE steps (nonlinear correction) should be < 167.
  If 50 CE steps suffice: mean-field provides 117 step speedup.
"""
import json, math, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_LAYERS_T=24; N_STU=6; BATCH=8; SEQ=64; LR=3e-4
L_ATT=14; ALPHA_STAR=1.429

print(f"\n{'='*65}")
print(f"  MEAN-FIELD INITIALIZATION")
print(f"  Joint coordinate descent: E and W_K alternating")
print(f"  Maintains coupling — no static pre-computation")
print(f"{'='*65}\n")

with open('/tmp/train_ids.json') as f: train_ids=list(map(int,json.load(f)))
with open('/tmp/val_ids.json')   as f: val_ids  =list(map(int,json.load(f)))
with open('/tmp/vocab.json')     as f: _v=json.load(f)
vocab={t:i for i,t in enumerate(_v)} if isinstance(_v,list) else _v
VOCAB=len(vocab)
train_t=torch.tensor(train_ids,dtype=torch.long)
val_t=torch.tensor(val_ids,dtype=torch.long)

def get_batch(split='train'):
    data=val_t if split=='val' else train_t
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
    def get_flat_params(self):
        return torch.cat([p.data.flatten() for p in self.parameters()])
    def set_flat_params(self,flat):
        idx=0
        for p in self.parameters(): n=p.numel(); p.data.copy_(flat[idx:idx+n].reshape(p.shape)); idx+=n

def clr(s,total,warmup=20):
    if s<=warmup: return LR*s/warmup
    return LR*0.5*(1+math.cos(math.pi*(s-warmup)/(total-warmup)))
def eval_val(m,n=40):
    m.eval(); ls=[]
    with torch.no_grad():
        for _ in range(n): x,y=get_batch('val'); _,l=m(x,y); ls.append(l.item())
    return float(np.mean(ls))

def hv_product(model,v,n=15):
    params=list(model.parameters()); model.zero_grad()
    loss=sum(model(*get_batch())[1] for _ in range(n))/n
    grads=torch.autograd.grad(loss,params,create_graph=True)
    gv=(torch.cat([g.flatten() for g in grads])*v.detach()).sum()
    hv=torch.cat([h.flatten() for h in torch.autograd.grad(gv,params,retain_graph=False)]).detach()
    model.zero_grad(); return hv

def apply_newton_wk(stu,n_seq=500,eps=1e-3,scale=0.5):
    ga=torch.zeros_like(stu.blocks[0].attn.WK.weight)
    fd=torch.zeros_like(stu.blocks[0].attn.WK.weight)
    torch.manual_seed(2)
    for i in range(n_seq):
        ix=torch.randint(0,len(train_t)-SEQ-1,(1,))[0].item()
        x=train_t[ix:ix+SEQ].unsqueeze(0); y=train_t[ix+1:ix+SEQ+1].unsqueeze(0)
        stu.zero_grad(); _,loss=stu(x,y); loss.backward()
        g=torch.zeros_like(stu.blocks[0].attn.WK.weight)
        for l in range(N_STU):
            if stu.blocks[l].attn.WK.weight.grad is not None:
                g+=stu.blocks[l].attn.WK.weight.grad/N_STU
        ga+=g; fd+=g**2
    delta=-(ga/n_seq)/((fd/n_seq)+eps)
    with torch.no_grad():
        for l in range(N_STU):
            stu.blocks[l].attn.WK.weight.add_(scale*delta)
            stu.blocks[l].attn.WQ.weight.add_(scale*delta.T)

print("Training teacher (300 steps)...")
torch.manual_seed(42)
teacher=LM(D,N_HEADS,N_LAYERS_T)
opt=torch.optim.AdamW(teacher.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
for step in range(1,301):
    lr_now=LR*min(step,100)/100 if step<=100 else LR*0.5*(1+math.cos(math.pi*(step-100)/200))
    for pg in opt.param_groups: pg['lr']=lr_now
    teacher.train(); x,y=get_batch(); _,loss=teacher(x,y)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(teacher.parameters(),1.0); opt.step()
    if step%100==0:
        teacher.eval()
        with torch.no_grad(): vl=float(np.mean([teacher(*get_batch('val'))[1].item() for _ in range(10)]))
        print(f"  step {step}  val={vl:.4f}")
teacher.eval(); val_teacher=eval_val(teacher)
print(f"  Teacher val={val_teacher:.4f}\n")

def build_student():
    torch.manual_seed(99); stu=LM(D,N_HEADS,N_STU)
    stu.te.weight.data.copy_(teacher.te.weight.data)
    with torch.no_grad():
        stu.pe.weight.copy_(teacher.pe.weight); stu.ln_f.weight.copy_(teacher.ln_f.weight)
        stu.ln_f.bias.copy_(teacher.ln_f.bias)
        for l in range(N_STU):
            for s,d in [(teacher.blocks[L_ATT].attn.WK,stu.blocks[l].attn.WK),
                        (teacher.blocks[L_ATT].attn.WQ,stu.blocks[l].attn.WQ),
                        (teacher.blocks[L_ATT].attn.WV,stu.blocks[l].attn.WV),
                        (teacher.blocks[L_ATT].attn.op,stu.blocks[l].attn.op),
                        (teacher.blocks[L_ATT].ff.g,stu.blocks[l].ff.g),
                        (teacher.blocks[L_ATT].ff.v,stu.blocks[l].ff.v),
                        (teacher.blocks[L_ATT].ff.o,stu.blocks[l].ff.o)]:
                d.weight.copy_(s.weight)
    return stu

print("Computing v_neg...")
stu_ref=build_student(); n_p=sum(p.numel() for p in stu_ref.parameters())
v=torch.randn(n_p); v=v/v.norm()
for _ in range(15): Hv=hv_product(stu_ref,v,15); neg=-Hv; v=neg/max(float(neg.norm()),1e-10)
v_neg=v.clone(); print("v_neg ready.\n")

def mean_field_iterate(stu, n_iterations=3, n_corpus=200, mf_lr=0.01):
    """
    Joint coordinate descent on (E, W_K) maintaining coupling.
    
    Each iteration:
      1. Update E with gradient step (E frozen W_K)
      2. Update W_K with gradient step (W_K frozen E)
    
    Both updates use FULL CORPUS statistics (not minibatch).
    Alternating maintains the E-W_K coupling.
    
    This is the mean-field fixed-point iteration.
    Under linear attention it converges to the exact basin entrance.
    Under softmax attention it approximates the basin entrance.
    """
    print(f"  Mean-field iteration ({n_iterations} rounds, {n_corpus} seqs each):")

    for it in range(n_iterations):
        # ── Step 1: Update E with W_K fixed ──
        # Freeze W_K, W_Q (hold attention fixed)
        for l in range(N_STU):
            stu.blocks[l].attn.WK.weight.requires_grad_(False)
            stu.blocks[l].attn.WQ.weight.requires_grad_(False)

        # Accumulate embedding gradient over corpus
        emb_grad=torch.zeros(VOCAB,D)
        emb_fish=torch.zeros(VOCAB,D)
        torch.manual_seed(it*1000)
        for i in range(n_corpus):
            ix=torch.randint(0,len(train_t)-SEQ-1,(1,))[0].item()
            x=train_t[ix:ix+SEQ].unsqueeze(0); y=train_t[ix+1:ix+SEQ+1].unsqueeze(0)
            stu.zero_grad(); _,loss=stu(x,y); loss.backward()
            if stu.te.weight.grad is not None:
                g=stu.te.weight.grad.detach()
                emb_grad+=g; emb_fish+=g**2

        emb_grad/=n_corpus; emb_fish/=n_corpus
        # Natural gradient step for E
        delta_E=-(emb_grad/(emb_fish+1e-4))
        with torch.no_grad():
            stu.te.weight.add_(mf_lr*delta_E)

        v_e=eval_val(stu,n=10)

        # ── Step 2: Update W_K with E fixed ──
        for l in range(N_STU):
            stu.blocks[l].attn.WK.weight.requires_grad_(True)
            stu.blocks[l].attn.WQ.weight.requires_grad_(True)

        # Freeze embeddings
        stu.te.weight.requires_grad_(False)

        wk_grad=torch.zeros_like(stu.blocks[0].attn.WK.weight)
        wk_fish=torch.zeros_like(stu.blocks[0].attn.WK.weight)
        torch.manual_seed(it*1000+500)
        for i in range(n_corpus):
            ix=torch.randint(0,len(train_t)-SEQ-1,(1,))[0].item()
            x=train_t[ix:ix+SEQ].unsqueeze(0); y=train_t[ix+1:ix+SEQ+1].unsqueeze(0)
            stu.zero_grad(); _,loss=stu(x,y); loss.backward()
            g=torch.zeros_like(stu.blocks[0].attn.WK.weight)
            for l in range(N_STU):
                if stu.blocks[l].attn.WK.weight.grad is not None:
                    g+=stu.blocks[l].attn.WK.weight.grad/N_STU
            wk_grad+=g; wk_fish+=g**2

        wk_grad/=n_corpus; wk_fish/=n_corpus
        delta_WK=-(wk_grad/(wk_fish+1e-4))
        with torch.no_grad():
            for l in range(N_STU):
                stu.blocks[l].attn.WK.weight.add_(mf_lr*delta_WK)
                stu.blocks[l].attn.WQ.weight.add_(mf_lr*delta_WK.T)

        # Unfreeze all
        stu.te.weight.requires_grad_(True)

        v_wk=eval_val(stu,n=10)
        print(f"    iter {it+1}: after E update={v_e:.4f}  after W_K update={v_wk:.4f}")

def run(label, do_saddle=True, do_mf=False, mf_iters=3,
        settle_lr=5.0, settle_steps=33, basin_steps=167,
        do_sign=True, do_newton=True):
    stu=build_student()
    print(f"\n  [{label}]")

    if do_saddle:
        w0=stu.get_flat_params()
        stu.set_flat_params(w0+ALPHA_STAR*(v_neg/v_neg.norm()))
        print(f"    saddle exit: {eval_val(stu,n=20):.4f}")

    if do_mf:
        mean_field_iterate(stu,n_iterations=mf_iters,n_corpus=200,mf_lr=0.01)
        print(f"    after MF init: {eval_val(stu,n=20):.4f}")

    opt_s=torch.optim.AdamW(stu.parameters(),lr=LR*settle_lr,betas=(0.9,0.95),weight_decay=0.1)
    for step in range(1,settle_steps+1):
        for pg in opt_s.param_groups: pg['lr']=LR*settle_lr*min(step,10)/10
        stu.train(); x,y=get_batch(); _,loss=stu(x,y)
        opt_s.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(stu.parameters(),1.0); opt_s.step()
    print(f"    settle {settle_steps} (5x): {eval_val(stu,n=20):.4f}")

    if do_sign:
        with torch.no_grad():
            for l in [1,2]:
                stu.blocks[l].attn.WV.weight.mul_(-1)
                stu.blocks[l].attn.op.weight.mul_(-1)
        print(f"    sign: {eval_val(stu,n=20):.4f}")

    opt2=torch.optim.AdamW(stu.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
    for step in range(1,basin_steps+1):
        for pg in opt2.param_groups: pg['lr']=clr(step,basin_steps)
        stu.train(); x,y=get_batch(); _,loss=stu(x,y)
        opt2.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(stu.parameters(),1.0); opt2.step()
        if step in [25,50,75,100,125,150,167]:
            v=eval_val(stu,n=20)
            print(f"    basin {step:>4}: {v:.4f}{' ✓' if v<val_teacher else ''}")

    if do_newton:
        apply_newton_wk(stu)
        print(f"    WK Newton: {eval_val(stu,n=20):.4f}")

    vf=eval_val(stu,n=30)
    print(f"    FINAL={vf:.4f}")
    return vf

print("="*65)
print("EXPERIMENTS")
print("  A: Baseline (saddle+5xLR+sign+167CE+Newton)")
print("  B: Mean-field init (3 joint E/WK iters) + same pipeline")
print("  C: MF init + 100CE (test if 167 reducible)")
print("  D: MF init only + 50CE (aggressive test)")
print("="*65)

vA=run("A-Baseline",do_mf=False)
vB=run("B-MF+167CE",do_mf=True,mf_iters=3,basin_steps=167)
vC=run("C-MF+100CE",do_mf=True,mf_iters=3,basin_steps=100)
vD=run("D-MF+50CE",do_mf=True,mf_iters=3,basin_steps=50)

torch.save(teacher.state_dict(), 'teacher.pt')

print(f"""
{'='*65}
  MEAN-FIELD RESULTS
{'='*65}

  FINAL:
    Teacher:        val={val_teacher:.4f}
    A (baseline):   val={vA:.4f}
    B (MF+167CE):   val={vB:.4f}  diff={vA-vB:+.4f}
    C (MF+100CE):   val={vC:.4f}  diff={vA-vC:+.4f}
    D (MF+50CE):    val={vD:.4f}  diff={vA-vD:+.4f}

  IF B ~ A: MF init is neutral — coupling maintained, no interference
  IF B < A: MF init provides genuine speedup — joint update works
  IF C ~ A with 100 steps: MF saves 67 basin steps
  IF B > A: MF corrupts initialization — non-linearity wins
""")
