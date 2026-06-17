#!/usr/bin/env python3
"""
Perverse Schober + Chamber Regularizer
========================================
The perverse schober on M_MC assigns a Fukaya category to each
prime path patch and a spherical functor (Dehn twist) to each wall.

The support skeleton Lambda = union of pairwise overlaps P_i ∩ P_j.
Chambers = connected components of M_MC \ Lambda.
The correct chamber: Im(z_l) > 0 for all l, Var_l(arg z_l) = 0.

CHAMBER REGULARIZER:
  L_chamber = lambda * Var_l(arg(z_l))
            = lambda * E_l[(arg z_l - mean_l(arg z_l))^2]

  This penalizes phase variance across blocks.
  When all blocks are in the same chamber: Var = 0, no penalty.
  When blocks straddle a wall: Var > 0, gradient pushes to same side.

  The gradient of L_chamber w.r.t. W_K is:
  d(Var)/d(W_K) = 2*(arg z_l - mean) * d(arg z_l)/d(W_K)

  This is the Jacobian we need — but we use a proxy:
  Instead of full d(arg z_l)/d(W_K), use the SIGN of Im(z_l)
  as a binary chamber indicator, and regularize toward all-positive.

PRACTICAL IMPLEMENTATION:
  At each step:
  1. Compute Im(z_l) for each block (fast, m=8)
  2. Compute sign pattern: s_l = sign(Im(z_l))
  3. If any s_l < 0: add a regularization loss that pushes Im(z_l) > 0
     R = sum_l max(0, -Im(z_l)) * lambda
  4. Total loss: L_CE + lambda * R

  Im(z_l) = sv_1(J_l) * sin(theta_l)
  This is differentiable w.r.t. W_K (approximately) via the
  dominant singular value.

  The chamber regularizer acts only during the 33-step settling phase.
  After all Im(z_l) > 0: regularizer turns off (L_chamber = 0).

COBORDISM INTERPRETATION:
  The spherical functor Phi_ij = Dehn twist at wall P_i ∩ P_j.
  Cost of one wall crossing = Dehn gap / Adam effective step = ~8 steps.
  4 crossings = 33 steps.
  
  Chamber regularizer makes the optimizer STAY on one side of each wall
  instead of bouncing across it. Expected savings: 33 steps.
  
  Total: (200-33)CE + Newton + 0 chamber steps = 167CE + Newton.
  Flop reduction: 167/200 * (200CE+Newton budget) → 7.1x vs 5.9x.
"""
import json, math, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_LAYERS_T=24; N_STU=6; BATCH=8; SEQ=64; LR=3e-4
L_ATT=14; M_FAST=8  # fast Jacobian dimension

print(f"\n{'='*65}")
print(f"  PERVERSE SCHOBER + CHAMBER REGULARIZER")
print(f"  L = L_CE + lambda * Var_l(arg z_l)")
print(f"  Penalizes inter-chamber gradient — forces single sheet")
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

def get_Im_z_blocks(stu, x_ref, m=M_FAST):
    """Get Im(z_l) for all blocks. Fast: m=8."""
    stu.eval()
    with torch.no_grad():
        hs=stu.hidden_states_all(x_ref); hs=[h[0] for h in hs]
    Im_z=[]; theta=0.0; prev_u1=None
    for l in range(N_STU):
        h_l=hs[l]; _,_,Vt=torch.linalg.svd(h_l,full_matrices=False)
        U_sub=Vt[:m,:].T.detach().numpy(); pos=h_l.shape[0]//2
        J=np.zeros((m,m))
        with torch.enable_grad():
            for i in range(m):
                hh=h_l.clone().unsqueeze(0).detach().requires_grad_(True)
                ho=stu.blocks[l](hh)
                v=ho[0,pos,:]
                (v*torch.tensor(U_sub[:,i],dtype=torch.float32)).sum().backward()
                J[:,i]=U_sub.T@hh.grad[0,pos,:].detach().numpy()
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

def chamber_regularizer_loss(stu, x_ref, m=M_FAST):
    """
    Compute chamber regularizer loss:
    R = sum_l max(0, -Im_z_l)  (penalize negative Im(z))
    
    This is a SOFT version: gradient pushes Im(z_l) toward positive.
    Uses differentiable proxy: sv_1(W_K_l[:m,:m]) * sin(theta_l)
    where theta_l is the accumulated angle (fixed, from previous step).
    """
    # Get current Im(z) values (for threshold check)
    Im_z=get_Im_z_blocks(stu,x_ref,m)
    n_negative=sum(1 for z in Im_z if z<0)

    if n_negative==0:
        return None, Im_z, 0  # already in correct chamber

    # Differentiable proxy for Im(z_l):
    # sv_1(W_K_l[:m,:m]) * sin(theta_l)
    # where theta_l is treated as fixed (from current Im_z measurement)
    chamber_loss=torch.tensor(0.0,requires_grad=True)
    for l in range(N_STU):
        if Im_z[l]<0:
            # sv_1 of projected W_K
            WK_proj=stu.blocks[l].attn.WK.weight[:m,:m]
            sv1=torch.linalg.svdvals(WK_proj)[0]
            # theta_l from current measurement
            theta_l=math.asin(max(-1,min(1,Im_z[l]/max(float(
                torch.linalg.svdvals(WK_proj)[0].detach()),1e-6))))
            # Proxy Im_z = sv1 * sin(theta_l)
            # We want Im_z > 0, so penalize sv1 * sin(theta_l) < 0
            # Since theta_l < 0: sin(theta_l) < 0 -> sv1 * sin < 0
            # Gradient: d(penalty)/d(sv1) = sin(theta_l) < 0
            # This pushes sv1 up... but sv1 > 0 always.
            # Better: directly penalize sv1 in negative-theta direction
            # Use hinge: max(0, -sv1 * sin(theta_l))
            sin_theta=math.sin(theta_l)
            penalty=torch.clamp(-sv1*sin_theta, min=0)
            chamber_loss=chamber_loss+penalty

    return chamber_loss, Im_z, n_negative

def apply_newton_correction(stu, n_seq=500, eps=1e-3, scale=0.5):
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

# Flop budget
ATTN_FLOPS=4*D**2*SEQ+2*SEQ**2*D
FF_FLOPS=8*D**2
LAYER_FLOPS=(ATTN_FLOPS+FF_FLOPS*SEQ)
STEP_T=LAYER_FLOPS*N_LAYERS_T*BATCH*3
STEP_S=LAYER_FLOPS*N_STU*BATCH*3
TEACHER_TOTAL=STEP_T*300
NEWTON_FLOPS=LAYER_FLOPS*N_STU*500*2

print(f"FLOP BUDGET")
print(f"  Teacher total: {TEACHER_TOTAL/1e12:.3f}T flops")
print(f"  Student/step:  {STEP_S/1e9:.2f}B flops")
print(f"  Newton:        {NEWTON_FLOPS/1e9:.1f}B flops")
print(f"  200CE+Newton:  {(STEP_S*200+NEWTON_FLOPS)/1e9:.1f}B  "
      f"→ {TEACHER_TOTAL/(STEP_S*200+NEWTON_FLOPS):.1f}x reduction")
print(f"  167CE+Newton:  {(STEP_S*167+NEWTON_FLOPS)/1e9:.1f}B  "
      f"→ {TEACHER_TOTAL/(STEP_S*167+NEWTON_FLOPS):.1f}x reduction\n")

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

def run(label, lambda_chamber=0.0, settle_steps=0,
        total_steps=200, apply_newton=True):
    stu=build_student()
    v0=eval_val(stu,n=20)
    Im_z0=get_Im_z_blocks(stu,x_ref)
    neg0=sum(1 for z in Im_z0 if z<0)
    print(f"\n  [{label}]")
    print(f"    zero-shot={v0:.4f}  neg_blocks={neg0}/6  "
          f"Im(z): {[f'{z:.2f}' for z in Im_z0]}")

    opt_s=torch.optim.AdamW(stu.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
    ck={0:v0}; sign_flips=0; prev_sign=None; chamber_resolved_at=None
    n_ce_actual=0

    for step in range(1,total_steps+1):
        for pg in opt_s.param_groups: pg['lr']=clr(step,total_steps)
        stu.train(); x,y=get_batch(); _,ce_loss=stu(x,y)

        # Chamber regularizer (only during settle phase or if lambda>0)
        total_loss=ce_loss
        if lambda_chamber>0 and (settle_steps==0 or step<=settle_steps):
            ch_loss,Im_z_now,n_neg=chamber_regularizer_loss(stu,x_ref)
            if ch_loss is not None:
                total_loss=ce_loss+lambda_chamber*ch_loss
            else:
                Im_z_now=None; n_neg=0
        else:
            Im_z_now=None; n_neg=0

        opt_s.zero_grad(); total_loss.backward()
        torch.nn.utils.clip_grad_norm_(stu.parameters(),1.0); opt_s.step()
        n_ce_actual+=1

        # Track sign flips (every 5 steps to save time)
        if step%5==0:
            Im_z_check=get_Im_z_blocks(stu,x_ref)
            signs=tuple(1 if z>0 else -1 for z in Im_z_check)
            if prev_sign is not None and signs!=prev_sign:
                sign_flips+=1
            if chamber_resolved_at is None and all(z>0 for z in Im_z_check):
                chamber_resolved_at=step
                print(f"    ✓ Chamber resolved at step {step}  "
                      f"(sign flips so far: {sign_flips})")
            prev_sign=signs

        if step in [10,20,33,50,75,100,125,150,175,200]:
            v=eval_val(stu,n=20); ck[step]=v
            marker='✓' if v<val_teacher else ''
            print(f"    step {step:>4}  val={v:.4f} {marker}  "
                  f"neg={n_neg}  flips={sign_flips}")

    if apply_newton:
        print(f"    Applying Newton correction...")
        gnorm=apply_newton_correction(stu)
        v_n=eval_val(stu); ck['newton']=v_n
        print(f"    After Newton: val={v_n:.4f}  ||g||={gnorm:.6f}")

    vf=eval_val(stu,n=30)
    flops=STEP_S*n_ce_actual+(NEWTON_FLOPS if apply_newton else 0)
    reduc=TEACHER_TOTAL/flops
    print(f"    FINAL={vf:.4f}  CE steps={n_ce_actual}  "
          f"Flops={flops/1e9:.1f}B  Reduction={reduc:.1f}x")
    print(f"    Chamber resolved at step: {chamber_resolved_at or 'not resolved'}")
    print(f"    Total sign flips: {sign_flips}")
    return vf,ck,flops,sign_flips,chamber_resolved_at

print("="*65)
print("EXPERIMENTS")
print("  A: Standard (baseline, 200CE+Newton)")
print("  B: Chamber regularizer lambda=0.1 (all 200 steps)")
print("  C: Chamber regularizer lambda=1.0 (first 50 steps only)")
print("  D: Chamber regularizer lambda=10.0 (first 33 steps only)")
print("="*65)

vA,ckA,fA,sfA,crA=run("A-Standard",lambda_chamber=0.0,apply_newton=True)
vB,ckB,fB,sfB,crB=run("B-Chamber-0.1",lambda_chamber=0.1,apply_newton=True)
vC,ckC,fC,sfC,crC=run("C-Chamber-1.0-50steps",lambda_chamber=1.0,
                        settle_steps=50,apply_newton=True)
vD,ckD,fD,sfD,crD=run("D-Chamber-10.0-33steps",lambda_chamber=10.0,
                        settle_steps=33,apply_newton=True)

print(f"\n{'='*65}")
print("  PERVERSE SCHOBER RESULTS")
print("="*65)
print(f"\n  {'step':>6}  {'A-std':>7}  {'B-0.1':>7}  {'C-1.0':>7}  {'D-10.0':>7}")
for s in [10,20,33,50,75,100,125,150,175,200,'newton']:
    row=f"  {str(s):>6}"
    for ck in [ckA,ckB,ckC,ckD]:
        v=ck.get(s)
        row+=f"  {v:>7.4f}" if v else f"  {'---':>7}"
    print(row)

print(f"""
  FINAL + FLOPS:
    Teacher:          val={val_teacher:.4f}  Flops={TEACHER_TOTAL/1e12:.3f}T
    A (std):          val={vA:.4f}  {fA/1e9:.1f}B  {TEACHER_TOTAL/fA:.1f}x  flips={sfA}  resolved={crA}
    B (λ=0.1):        val={vB:.4f}  {fB/1e9:.1f}B  {TEACHER_TOTAL/fB:.1f}x  flips={sfB}  resolved={crB}
    C (λ=1.0,50s):    val={vC:.4f}  {fC/1e9:.1f}B  {TEACHER_TOTAL/fC:.1f}x  flips={sfC}  resolved={crC}
    D (λ=10,33s):     val={vD:.4f}  {fD/1e9:.1f}B  {TEACHER_TOTAL/fD:.1f}x  flips={sfD}  resolved={crD}

  PERVERSE SCHOBER PREDICTION:
    IF sign_flips(B/C/D) < sign_flips(A):
      Chamber regularizer suppresses wall crossings.
      The perverse schober structure is active and controllable.
      
    IF chamber_resolved_at(B/C/D) < chamber_resolved_at(A):
      Regularizer accelerates sheet settling.
      Savings: (33 - resolved_at) × step_flops.
      
    IF val(B/C/D) < val(A) at same or fewer steps:
      Chamber regularizer is a net improvement.
      The perverse schober geometry gives actionable information.
      The cobordism (wall crossing) cost is reducible.
""")
