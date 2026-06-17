#!/usr/bin/env python3
"""
Precomputed Sheet Assignment
=============================
The endpoint of the 33-step accelerated cobordism traversal
is estimable from the teacher's Im(z_l) values alone.

THEORY:
  After 33 CE steps at 5x LR, blocks with |Im(z_l^teacher)| < threshold
  will have flipped to negative Im(z_l). The sign correction (flip W_V, W_O)
  then restores the correct chamber.

  If we can predict which blocks flip, we can apply the sign correction
  at step 0 — before any CE training — and run 200 CE steps in the
  correct chamber from the start.

ESTIMATE:
  threshold ~ 5 * eta * ||g|| * T_settle / Dehn_gap
  From data: blocks 1 (Im=1.01) and 2 (Im=0.40) flipped.
  Threshold is between 0.40 and 1.01 — approximately 0.7.
  
  Predict: blocks where |Im(z_l^teacher)| < threshold flip.
  Apply W_V, W_O sign correction for those blocks at step 0.

EXPERIMENTS:
  A: C from etale_sheet_settler (33CE 5xLR + sign + 167CE + Newton) — BEST
  B: Precomputed sign correction at step 0, threshold=0.5, 200CE+Newton
  C: Precomputed sign correction at step 0, threshold=1.1, 200CE+Newton
  D: Sign correction at step 0 for ALL blocks, 200CE+Newton
  E: No sign correction (standard baseline)

If B or C achieves val ~ 0.038 without the 33-step settling phase:
  The sheet assignment is predictable from teacher Im(z_l).
  The entire 33-step phase is replaceable by algebraic precomputation.
  Total pipeline: 0 settle + sign + 200CE + Newton = same flops as E.
"""
import json, math, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_LAYERS_T=24; N_STU=6; BATCH=8; SEQ=64; LR=3e-4
L_ATT=14; M_FAST=16

print(f"\n{'='*65}")
print(f"  PRECOMPUTED SHEET ASSIGNMENT")
print(f"  Predict endpoint of cobordism from teacher Im(z_l)")
print(f"  Apply sign correction at step 0 — no settling needed")
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
    def hidden_states_all(self,x):
        hs=[]; h=self.te(x)+self.pe(torch.arange(x.shape[1])); hs.append(h.detach())
        for b in self.blocks: h=b(h); hs.append(h.detach())
        return hs

def clr(s,total=200,warmup=50):
    if s<=warmup: return LR*s/warmup
    return LR*0.5*(1+math.cos(math.pi*(s-warmup)/(total-warmup)))

def eval_val(model,n=60):
    model.eval(); ls=[]
    with torch.no_grad():
        for _ in range(n):
            x,y=get_batch('val'); _,l=model(x,y); ls.append(l.item())
    return float(np.mean(ls))

def get_Im_z_blocks(model, x_ref, m=M_FAST):
    model.eval()
    with torch.no_grad():
        hs=model.hidden_states_all(x_ref); hs=[h[0] for h in hs]
    Im_z=[]; theta=0.0; prev_u1=None
    for l in range(len(model.blocks)):
        h_l=hs[l]; pos=h_l.shape[0]//2
        _,_,Vt=torch.linalg.svd(h_l,full_matrices=False)
        U=Vt[:m,:].T.detach().numpy()
        J=np.zeros((m,m))
        with torch.enable_grad():
            for i in range(m):
                hh=h_l.clone().unsqueeze(0).detach().requires_grad_(True)
                ho=model.blocks[l](hh)
                v=ho[0,pos,:]
                (v*torch.tensor(U[:,i],dtype=torch.float32)).sum().backward()
                J[:,i]=U.T@hh.grad[0,pos,:].detach().numpy()
        J=J.T
        Ul,sv,_=np.linalg.svd(J,full_matrices=False)
        u1=Ul[:,0]; sv1=sv[0]
        if prev_u1 is not None:
            ct=float(np.clip(prev_u1@u1,-1,1))
            dt=math.acos(abs(ct))
            if prev_u1@u1<0: dt=-dt
            theta+=dt
        Im_z.append(sv1*math.sin(theta))
        prev_u1=u1
    return Im_z

def apply_newton_step(stu, n_seq=500, eps=1e-3, scale=0.5):
    grad_acc=torch.zeros_like(stu.blocks[0].attn.WK.weight)
    fisher_d=torch.zeros_like(stu.blocks[0].attn.WK.weight)
    torch.manual_seed(2)
    for i in range(n_seq):
        ix=torch.randint(0,len(train_t)-SEQ-1,(1,))[0].item()
        x=train_t[ix:ix+SEQ].unsqueeze(0); y=train_t[ix+1:ix+SEQ+1].unsqueeze(0)
        stu.zero_grad(); _,loss=stu(x,y); loss.backward()
        g=torch.zeros_like(stu.blocks[0].attn.WK.weight)
        for l in range(N_STU):
            if stu.blocks[l].attn.WK.weight.grad is not None:
                g+=stu.blocks[l].attn.WK.weight.grad/N_STU
        grad_acc+=g; fisher_d+=g**2
    delta=-(grad_acc/n_seq)/((fisher_d/n_seq)+eps)
    with torch.no_grad():
        for l in range(N_STU):
            stu.blocks[l].attn.WK.weight.add_(scale*delta)
            stu.blocks[l].attn.WQ.weight.add_(scale*delta.T)
    return float((grad_acc/n_seq).norm())

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

torch.manual_seed(0); x_ref,_=get_batch('val'); x_ref=x_ref[0:1]

# Measure teacher Im(z_l) — the ground truth for sheet prediction
print("="*65)
print("TEACHER Im(z_l) — sheet prediction ground truth")
print("="*65)
Im_z_teacher=get_Im_z_blocks(teacher,x_ref)
print(f"\n  Teacher Im(z_l): {[f'{z:.3f}' for z in Im_z_teacher]}")
print(f"  All positive: {all(z>0 for z in Im_z_teacher)}")
print(f"\n  Blocks by |Im(z)|:")
for l,z in enumerate(Im_z_teacher):
    bar='*'*int(abs(z)*20)
    print(f"  Block {l}: Im={z:>7.4f}  {bar}")

# Threshold analysis from experiment C result:
# Blocks 1 (Im=1.01) and 2 (Im=0.40) flipped after 33CE at 5xLR
# So threshold is somewhere between 0.40 and 1.01
# More precisely: Im=0.40 flipped, Im=1.01 also flipped
# → threshold > 1.01, all blocks with Im < 1.01 flip
# But block 0 (Im=0.00) didn't show as "flipped" in the sign correction
# (block 0 is always on real axis — Im=0 by construction)
# So effectively: blocks 1,2 flipped (Im=1.01, 0.40 both < threshold ~1.1)

print(f"\n  From experiment C: blocks 1,2 flipped (Im=1.01, 0.40)")
print(f"  Threshold estimate: ~1.1 (blocks with |Im| < 1.1 flip)")
print(f"  Predicted flip pattern: {[l for l,z in enumerate(Im_z_teacher) if 0<abs(z)<1.1]}")

def build_student(flip_blocks=None):
    """Build student with optional W_V/W_O sign correction."""
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
        # Apply precomputed sign correction
        if flip_blocks:
            for l in flip_blocks:
                stu.blocks[l].attn.WV.weight.mul_(-1)
                stu.blocks[l].attn.op.weight.mul_(-1)
    return stu

def run(label, flip_blocks=None, steps=200, do_newton=True,
        settle_steps=0, settle_lr_mult=1.0):
    stu=build_student(flip_blocks)
    v0=eval_val(stu,n=20)
    Im_z0=get_Im_z_blocks(stu,x_ref)
    neg0=sum(1 for z in Im_z0 if z<0)
    print(f"\n  [{label}]")
    print(f"    zero-shot={v0:.4f}  neg_blocks={neg0}/6")
    print(f"    flip_blocks={flip_blocks}  Im(z): {[f'{z:.2f}' for z in Im_z0]}")

    opt_s=torch.optim.AdamW(stu.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
    ck={0:v0}; total_steps=0

    # Optional settle phase
    if settle_steps>0:
        settle_lr=LR*settle_lr_mult
        for step in range(1,settle_steps+1):
            for pg in opt_s.param_groups: pg['lr']=settle_lr*min(step,10)/10
            stu.train(); x,y=get_batch(); _,loss=stu(x,y)
            opt_s.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(stu.parameters(),1.0); opt_s.step()
        total_steps+=settle_steps
        v_s=eval_val(stu,n=10)
        Im_z_s=get_Im_z_blocks(stu,x_ref)
        neg_s=sum(1 for z in Im_z_s if z<0)
        print(f"    After {settle_steps} settle steps (lr={settle_lr:.4f}): "
              f"val={v_s:.4f}  neg={neg_s}")
        # Sign correct remaining negatives
        if neg_s>0:
            with torch.no_grad():
                for l in range(N_STU):
                    if Im_z_s[l]<0:
                        stu.blocks[l].attn.WV.weight.mul_(-1)
                        stu.blocks[l].attn.op.weight.mul_(-1)
            v_corr=eval_val(stu,n=10)
            print(f"    After sign correction: val={v_corr:.4f}")

    # Main CE training
    for step in range(1,steps+1):
        total_step=total_steps+step
        for pg in opt_s.param_groups: pg['lr']=clr(total_step,total_steps+steps)
        stu.train(); x,y=get_batch(); _,loss=stu(x,y)
        opt_s.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(stu.parameters(),1.0); opt_s.step()
        if total_step in [33,50,75,100,125,150,175,200,233]:
            v=eval_val(stu,n=20); ck[total_step]=v
            print(f"    step {total_step:>4}  val={v:.4f}"
                  f"{' ✓' if v<val_teacher else ''}")

    if do_newton:
        gnorm=apply_newton_step(stu)
        v_n=eval_val(stu); ck['newton']=v_n
        print(f"    Newton: val={v_n:.4f}  ||g||={gnorm:.6f}")

    vf=eval_val(stu,n=30)
    print(f"    FINAL={vf:.4f}")
    return vf,ck

print(f"\n{'='*65}")
print("EXPERIMENTS")
print("  A: Best known (33CE 5xLR + sign + 167CE + Newton) [reference]")
print("  B: Precomputed flip [1,2] at step0, threshold=1.1, 200CE+Newton")
print("  C: Precomputed flip [2] at step0, threshold=0.5, 200CE+Newton")
print("  D: Flip ALL blocks at step0, 200CE+Newton")
print("  E: No flip (standard baseline) 200CE+Newton")
print("  F: Precomputed flip [1,2] + 33CE settle (1x) + 167CE + Newton")
print("="*65)

# Reference: reproduce C from etale_sheet_settler
vA,ckA=run("A-Reference-33CE5x+sign+167CE+Newton",
            flip_blocks=None,steps=167,do_newton=True,
            settle_steps=33,settle_lr_mult=5.0)

# Key test: precomputed flip at step 0
vB,ckB=run("B-Preflip[1,2]+200CE+Newton",
            flip_blocks=[1,2],steps=200,do_newton=True)

vC,ckC=run("C-Preflip[2]+200CE+Newton",
            flip_blocks=[2],steps=200,do_newton=True)

vD,ckD=run("D-FlipAll+200CE+Newton",
            flip_blocks=list(range(N_STU)),steps=200,do_newton=True)

vE,ckE=run("E-NoFlip+200CE+Newton",
            flip_blocks=None,steps=200,do_newton=True)

vF,ckF=run("F-Preflip[1,2]+33CE+167CE+Newton",
            flip_blocks=[1,2],steps=167,do_newton=True,
            settle_steps=33,settle_lr_mult=1.0)

print(f"\n{'='*65}")
print("  PRECOMPUTED SHEET RESULTS")
print("="*65)
print(f"\n  Teacher Im(z): {[f'{z:.2f}' for z in Im_z_teacher]}")
print(f"  Predicted flip blocks (|Im|<1.1): "
      f"{[l for l,z in enumerate(Im_z_teacher) if 0<abs(z)<1.1]}")

print(f"\n  CONVERGENCE:")
print(f"  {'step':>6}  {'A-ref':>7}  {'B-f12':>7}  {'C-f2':>7}  "
      f"{'D-all':>7}  {'E-none':>7}  {'F-f12s':>7}")
for s in [33,50,75,100,125,150,175,200,'newton']:
    row=f"  {str(s):>6}"
    for ck in [ckA,ckB,ckC,ckD,ckE,ckF]:
        v=ck.get(s)
        row+=f"  {v:>7.4f}" if v else f"  {'---':>7}"
    print(row)

best=min(vA,vB,vC,vD,vE,vF)
print(f"""
  FINAL:
    Teacher:          val={val_teacher:.4f}
    A (33+sign+167):  val={vA:.4f}  [reference best]
    B (flip[1,2]):    val={vB:.4f}  diff vs A={vA-vB:+.4f}
    C (flip[2]):      val={vC:.4f}  diff vs A={vA-vC:+.4f}
    D (flip all):     val={vD:.4f}  diff vs A={vA-vD:+.4f}
    E (no flip):      val={vE:.4f}  diff vs A={vA-vE:+.4f}
    F (flip+settle):  val={vF:.4f}  diff vs A={vA-vF:+.4f}

  KEY QUESTION:
    IF B ~ A: The sheet endpoint is predictable from teacher Im(z_l).
      Precomputed flip [1,2] at step 0 replaces 33-step settling.
      Same quality, no settling phase needed.
      Pipeline: flip + 200CE + Newton (no LR scheduling needed).

    IF B < A: Precomputed flip is even better than 33CE settling.
      The settling was imprecise — precomputed flip is exact.
      The sheet endpoint is fully determined by teacher Im(z_l).

    IF B >> A: The sheet endpoint is NOT predictable.
      The 33-step settling phase discovers something not in Im(z^teacher).
      The cobordism traversal has intrinsic information content.
""")
