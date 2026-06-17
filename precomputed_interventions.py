#!/usr/bin/env python3
"""
Pre-computed Interventions — Using What We Know
=================================================
Three interventions derived from pre-computed quantities:

1. WK ALIGNMENT: Rotate W_K to align with embedding covariance
   W_K_aligned = V_E @ V_E.T @ W_K  (project onto top-k emb directions)
   This fixes the W_K-embedding misalignment (alignment 0.05 -> higher)
   Expected: reduces/eliminates 33-step settle phase

2. RARE TOKEN LR SCALING: Weight embedding LR by 1/P(t)
   High-LR for rare tokens (frequency 0.001) that define valley 2
   Low-LR for common tokens (already well-represented)
   Expected: rare tokens reach valley 2 embedding values faster

3. COMBINED: WK alignment + rare token LR scaling + saddle exit
   All three pre-computed interventions together
   Expected: reduces total CE steps needed

ATTRACTOR TARGETING RESULT (from attractor_targeting.py):
  B (0 basin steps + Newton): val=0.217 — Newton invalid at entrance
  D (50 basin steps + Newton): val=0.093
  E (100 basin steps + Newton): val=0.053
  A (167 basin steps + Newton): val=0.033 — best

The 167 basin steps integrate rare token statistics.
If we can provide rare token information algebraically,
fewer steps should suffice.
"""
import json, math, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_LAYERS_T=24; N_STU=6; BATCH=8; SEQ=64; LR=3e-4
L_ATT=14; ALPHA_STAR=1.429

print(f"\n{'='*65}")
print(f"  PRE-COMPUTED INTERVENTIONS")
print(f"  Using embedding geometry to replace gradient descent")
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
def apply_newton(stu,n_seq=500,eps=1e-3,scale=0.5):
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

# ════════════════════════════════════════════
# PRE-COMPUTE EVERYTHING
# ════════════════════════════════════════════
print("="*65)
print("PRE-COMPUTATION (no gradient descent)")
print("="*65)

# Token frequencies
token_freq=torch.zeros(VOCAB)
torch.manual_seed(0)
for _ in range(500): x,y=get_batch(); [token_freq.__setitem__(t,token_freq[t]+1) for t in y.flatten()]
token_prob=token_freq/token_freq.sum()

# Embedding matrix (teacher, tied to head)
E=teacher.te.weight.data.clone()  # (VOCAB, D)

# Corpus-weighted embedding SVD
sqrt_p=token_prob.sqrt().unsqueeze(1)
E_weighted=(sqrt_p*E).numpy()
U_emb,sv_emb,Vt_emb=np.linalg.svd(E_weighted,full_matrices=False)
V_emb=torch.tensor(Vt_emb,dtype=torch.float32)  # (D, D) — embedding directions

print(f"\n  Embedding SVD: top sv = {sv_emb[:5].round(4)}")
print(f"  Token freq range: min={float(token_prob.min()):.6f} max={float(token_prob.max()):.6f}")

# Rare tokens: frequency < 0.002 (below 2x average)
avg_prob=1.0/VOCAB
rare_threshold=2.0*avg_prob
rare_mask=token_prob<rare_threshold
n_rare=rare_mask.sum().item()
print(f"  Rare tokens (P < {rare_threshold:.4f}): {n_rare}/{VOCAB}")
print(f"  Average frequency: {avg_prob:.6f}")

# Inverse frequency weights for rare token LR scaling
# lr_scale[t] = min(avg_prob/P(t), max_scale) — boost rare tokens
max_lr_scale=50.0
lr_scale=torch.clamp(avg_prob/token_prob.clamp(min=1e-8), max=max_lr_scale)
print(f"  LR scale range: {float(lr_scale.min()):.1f}x to {float(lr_scale.max()):.1f}x")

# Saddle exit v_neg (recompute)
def hv_product(model,v,n=15):
    params=list(model.parameters()); model.zero_grad()
    loss=sum(model(*get_batch())[1] for _ in range(n))/n
    grads=torch.autograd.grad(loss,params,create_graph=True)
    gv=(torch.cat([g.flatten() for g in grads])*v.detach()).sum()
    hv=torch.cat([h.flatten() for h in torch.autograd.grad(gv,params,retain_graph=False)]).detach()
    model.zero_grad(); return hv

def build_student_base():
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

print("\n  Computing v_neg (saddle exit direction)...")
stu_ref=build_student_base()
n_p=sum(p.numel() for p in stu_ref.parameters())
v=torch.randn(n_p); v=v/v.norm()
for _ in range(15): Hv=hv_product(stu_ref,v,15); neg=-Hv; v=neg/max(float(neg.norm()),1e-10)
v_neg=v.clone()
print("  v_neg computed.\n")

# INTERVENTION 1: Align W_K with top-k embedding directions
def align_WK_with_embeddings(stu, k=32):
    """Project W_K onto top-k embedding directions."""
    Vk=V_emb[:k,:]  # (k, D) top-k embedding directions
    with torch.no_grad():
        for l in range(N_STU):
            WK=stu.blocks[l].attn.WK.weight.data  # (D, D)
            # Project each row of W_K onto embedding subspace
            WK_aligned=Vk.T@(Vk@WK.T)  # (D, D) — projection
            # Blend: keep some original structure
            stu.blocks[l].attn.WK.weight.data.copy_(0.5*WK+0.5*WK_aligned.T)
            stu.blocks[l].attn.WQ.weight.data.copy_(0.5*WK.T+0.5*WK_aligned)

# INTERVENTION 2: Inverse frequency embedding LR during basin descent
class InvFreqAdam(torch.optim.AdamW):
    """AdamW with per-token LR scaling for embeddings."""
    def __init__(self,params,lr_scale_emb,**kwargs):
        super().__init__(params,**kwargs)
        self.lr_scale_emb=lr_scale_emb  # (VOCAB,) per-token LR scale

    def step(self,closure=None):
        # Scale embedding gradient before step
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None and p.shape==(VOCAB,D):
                    # This is the embedding matrix
                    p.grad.data*=self.lr_scale_emb.unsqueeze(1)
        return super().step(closure)

def run(label, do_saddle=True, align_wk=False, inv_freq_lr=False,
        settle_lr=5.0, settle_steps=33, basin_steps=167,
        do_sign=True, do_newton=True):
    stu=build_student_base()

    # INTERVENTION 1: W_K alignment
    if align_wk:
        align_WK_with_embeddings(stu,k=32)
        v_align=eval_val(stu,n=20)
        print(f"\n  [{label}]  after W_K alignment: val={v_align:.4f}")
    else:
        v0=eval_val(stu,n=20)
        print(f"\n  [{label}]  zero-shot: val={v0:.4f}")

    # Saddle exit
    if do_saddle:
        w0=stu.get_flat_params()
        stu.set_flat_params(w0+ALPHA_STAR*(v_neg/v_neg.norm()))
        vs=eval_val(stu,n=20)
        print(f"    after saddle exit: val={vs:.4f}")

    # Settle phase
    if inv_freq_lr:
        opt_s=InvFreqAdam(stu.parameters(),lr_scale_emb=lr_scale,
                           lr=LR*settle_lr,betas=(0.9,0.95),weight_decay=0.1)
    else:
        opt_s=torch.optim.AdamW(stu.parameters(),lr=LR*settle_lr,betas=(0.9,0.95),weight_decay=0.1)

    for step in range(1,settle_steps+1):
        for pg in opt_s.param_groups: pg['lr']=LR*settle_lr*min(step,10)/10
        stu.train(); x,y=get_batch(); _,loss=stu(x,y)
        opt_s.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(stu.parameters(),1.0); opt_s.step()
    v_settle=eval_val(stu,n=20)
    print(f"    after {settle_steps} settle steps: val={v_settle:.4f}")

    # Sign correction
    if do_sign:
        with torch.no_grad():
            for l in [1,2]:
                stu.blocks[l].attn.WV.weight.mul_(-1)
                stu.blocks[l].attn.op.weight.mul_(-1)
        v_sign=eval_val(stu,n=20)
        print(f"    after sign correction: val={v_sign:.4f}")

    # Basin descent
    if inv_freq_lr:
        opt2=InvFreqAdam(stu.parameters(),lr_scale_emb=lr_scale,
                          lr=LR,betas=(0.9,0.95),weight_decay=0.1)
    else:
        opt2=torch.optim.AdamW(stu.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)

    for step in range(1,basin_steps+1):
        for pg in opt2.param_groups: pg['lr']=clr(step,basin_steps)
        stu.train(); x,y=get_batch(); _,loss=stu(x,y)
        opt2.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(stu.parameters(),1.0); opt2.step()
        if step in [25,50,75,100,125,150,167]:
            v=eval_val(stu,n=20)
            print(f"    basin step {step:>4}: val={v:.4f}{' ✓' if v<val_teacher else ''}")

    if do_newton:
        apply_newton(stu)
        v_n=eval_val(stu); print(f"    Newton: val={v_n:.4f}")

    vf=eval_val(stu,n=30); print(f"    FINAL={vf:.4f}")
    return vf

print("="*65)
print("EXPERIMENTS")
print("  A: Baseline (saddle+5xLR+sign+167CE+Newton)")
print("  B: + W_K alignment with embedding directions (k=32)")
print("  C: + Inverse frequency LR for embeddings")
print("  D: + Both W_K alignment AND inverse frequency LR")
print("  E: WK align + inv-freq + saddle + 100CE + Newton (fewer steps)")
print("="*65)

vA=run("A-Baseline")
vB=run("B-WKalign",align_wk=True)
vC=run("C-InvFreqLR",inv_freq_lr=True)
vD=run("D-Both",align_wk=True,inv_freq_lr=True)
vE=run("E-Both+100CE",align_wk=True,inv_freq_lr=True,basin_steps=100)

print(f"""
{'='*65}
  PRE-COMPUTED INTERVENTIONS RESULTS
{'='*65}

  FINAL:
    Teacher:           val={val_teacher:.4f}
    A (baseline):      val={vA:.4f}
    B (WK align):      val={vB:.4f}  diff={vA-vB:+.4f}
    C (inv-freq LR):   val={vC:.4f}  diff={vA-vC:+.4f}
    D (both):          val={vD:.4f}  diff={vA-vD:+.4f}
    E (both+100CE):    val={vE:.4f}  diff={vA-vE:+.4f}

  DECISIVE QUESTIONS:
    IF B < A: W_K-embedding alignment reduces basin steps needed
      The misalignment was causing unnecessary saddle wandering
      
    IF C < A: Rare token LR boosting accelerates valley 2 access
      The 167 steps were spending time on rare tokens unnecessarily
      
    IF E ~ A with 100 instead of 167 steps:
      Pre-computed interventions save 67 basin steps (40% reduction)
      The interventions are using what we pre-computed
""")
