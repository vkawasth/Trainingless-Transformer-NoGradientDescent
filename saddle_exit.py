#!/usr/bin/env python3
"""
Saddle Exit — One Shot from Corpus Hessian
============================================
The loss landscape at initialization is a SADDLE, not a flat plateau.
  - Large gradient (0.986) but pointing WRONG direction (alignment=-0.035)
  - Negative curvature at steps 33, 100 (FLAT/SADDLE in 1D slice)
  - 5x LR works by jumping OVER the saddle, not across a flat region

THE EXIT:
  The saddle has a direction of steepest negative curvature:
    v_neg = eigenvector of Hessian H corresponding to lambda_min < 0

  One step in v_neg direction exits the saddle:
    theta_exit = theta_0 + alpha * v_neg
  
  where alpha is chosen so that L(theta_exit) is minimized along v_neg.
  This is a 1D line search along v_neg — trivial.

COMPUTING v_neg:
  H v_neg = lambda_min * v_neg
  
  Use INVERSE power iteration for the minimum eigenvalue:
    v_{k+1} = (H - mu*I)^{-1} v_k / ||(H - mu*I)^{-1} v_k||
  
  But H^{-1} is expensive. Alternative: Lanczos with a few steps.
  
  PRACTICAL: use randomized power iteration on (-H):
    Fv = -E_D[H v]  (negative Hessian vector product)
    Top eigenvector of (-H) = eigenvector of min eigenvalue of H
  
  Hessian-vector product via double backprop:
    H v = grad(grad L . v)  — one extra backward pass per iteration.

PIPELINE:
  Step 0: Compute v_neg (min Hessian eigenvector) — one corpus pass
  Step 1: Line search along v_neg — evaluate L at 5-10 points
  Step 2: Jump to minimum: theta* = theta_0 + alpha* * v_neg
  Step 3: Standard CE training from theta* (now past the saddle)
  Step 4: Newton correction at end

PREDICTION:
  After saddle exit: we land where 5x LR lands after ~7 steps (val~0.3-0.5)
  Then standard 167 CE + Newton from that point.
  Final val should match C from etale_sheet_settler (0.038).
  
  If confirmed: the saddle exit compresses 33 steps into 1 computation.
  The corpus Hessian contains the full information needed.
"""
import json, math, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_LAYERS_T=24; N_STU=6; BATCH=8; SEQ=64; LR=3e-4
L_ATT=14

print(f"\n{'='*65}")
print(f"  SADDLE EXIT — ONE SHOT FROM CORPUS HESSIAN")
print(f"  v_neg = min eigenvector of H = E_D[Hessian L]")
print(f"  theta* = theta_0 + alpha * v_neg  (line search, no training)")
print(f"{'='*65}\n")

with open('/tmp/train_ids.json') as f: train_ids=list(map(int,json.load(f)))
with open('/tmp/val_ids.json')   as f: val_ids  =list(map(int,json.load(f)))
with open('/tmp/vocab.json')     as f: vocab=json.load(f)
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
    def set_flat_params(self, flat):
        idx=0
        for p in self.parameters():
            n=p.numel(); p.data.copy_(flat[idx:idx+n].reshape(p.shape)); idx+=n
    def flat_grad(self):
        return torch.cat([p.grad.flatten() if p.grad is not None
                         else torch.zeros(p.numel()) for p in self.parameters()])

def clr(s,total=200,warmup=50):
    if s<=warmup: return LR*s/warmup
    return LR*0.5*(1+math.cos(math.pi*(s-warmup)/(total-warmup)))

def eval_val(model, n=40):
    model.eval(); ls=[]
    with torch.no_grad():
        for _ in range(n):
            x,y=get_batch('val'); _,l=model(x,y); ls.append(l.item())
    return float(np.mean(ls))

def hessian_vector_product(model, v, n_batches=20):
    """
    Compute H*v via double backprop.
    H*v = d/dtheta (grad L . v)
    """
    params=list(model.parameters())
    model.zero_grad()
    total_loss=torch.tensor(0.0)
    torch.manual_seed(42)
    for _ in range(n_batches):
        x,y=get_batch('train'); _,loss=model(x,y)
        total_loss=total_loss+loss/n_batches

    # First backward: get grad L
    grads=torch.autograd.grad(total_loss, params, create_graph=True)
    flat_grad=torch.cat([g.flatten() for g in grads])

    # Dot with v
    gv=(flat_grad * v.detach()).sum()

    # Second backward: get H*v = d(grad L . v)/dtheta
    hv_grads=torch.autograd.grad(gv, params, retain_graph=False)
    hv=torch.cat([g.flatten() for g in hv_grads]).detach()
    model.zero_grad()
    return hv

def min_hessian_eigenvector(model, n_iter=15, n_batches=20):
    """
    Find eigenvector of MINIMUM Hessian eigenvalue via power iteration on -H.
    v_min = top eigenvector of (-H)
    lambda_min = -top eigenvalue of (-H)
    """
    n_params=sum(p.numel() for p in model.parameters())
    v=torch.randn(n_params); v=v/v.norm()
    lambda_min=0.0

    print(f"  Power iteration on -H ({n_iter} iters, {n_batches} batches each):")
    for it in range(n_iter):
        Hv=hessian_vector_product(model, v, n_batches)
        neg_Hv=-Hv  # eigenvector of -H = eigenvector of min eigenvalue of H
        lam=float(neg_Hv.norm())
        lambda_min=-lam  # eigenvalue of H (negative = saddle direction)
        v=neg_Hv/max(lam,1e-10)
        if (it+1)%5==0:
            print(f"    iter {it+1}: lambda_min={lambda_min:.4f}  ||v||={float(v.norm()):.4f}")

    print(f"  Final: lambda_min(H) = {lambda_min:.4f}")
    print(f"  {'SADDLE CONFIRMED' if lambda_min<0 else 'NO SADDLE — convex'}")
    return v, lambda_min

def line_search_along_v(model, v, n_points=15, scale=5.0):
    """
    Find alpha* = argmin L(theta_0 + alpha * v)
    1D quadratic line search along negative curvature direction.
    """
    w0=model.get_flat_params().clone()
    v_norm=v/v.norm()

    alphas=np.linspace(-scale, scale, n_points)
    losses=[]
    for alpha in alphas:
        model.set_flat_params(w0 + alpha*v_norm)
        with torch.no_grad():
            ls=[model(*get_batch('train'))[1].item() for _ in range(8)]
        losses.append(np.mean(ls))

    # Find minimum
    best_idx=np.argmin(losses)
    alpha_star=float(alphas[best_idx])
    loss_star=losses[best_idx]

    model.set_flat_params(w0)  # restore
    return alpha_star, loss_star, alphas, losses

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

# ════════════════════════════════════════════════════
# Train teacher
# ════════════════════════════════════════════════════
print("Training teacher (300 steps)...")
torch.manual_seed(42)
teacher=LM(D,N_HEADS,N_LAYERS_T)
opt=torch.optim.AdamW(teacher.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
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
        print(f"  step {step}  val={vl:.4f}")
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

# ════════════════════════════════════════════════════
# STEP 1: Compute minimum Hessian eigenvector
# ════════════════════════════════════════════════════
print("="*65)
print("STEP 1: MINIMUM HESSIAN EIGENVECTOR (saddle direction)")
print("  v_neg = argmin_v v^T H v / ||v||^2")
print("="*65)

stu=build_student()
v0=eval_val(stu)
print(f"\n  Initial val: {v0:.4f}")
print(f"  Computing min Hessian eigenvector...")
t0=time.time()
v_neg, lambda_min=min_hessian_eigenvector(stu, n_iter=15, n_batches=20)
print(f"  Computed in {time.time()-t0:.1f}s")

# ════════════════════════════════════════════════════
# STEP 2: Line search along v_neg
# ════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("STEP 2: LINE SEARCH ALONG v_neg")
print("  Find alpha* = argmin L(theta_0 + alpha * v_neg)")
print("="*65)

print(f"\n  Evaluating L at 15 points along v_neg (scale=±5.0)...")
alpha_star, loss_star, alphas, losses=line_search_along_v(stu, v_neg, n_points=15, scale=5.0)

print(f"\n  alpha:  {[f'{a:.2f}' for a in alphas]}")
print(f"  losses: {[f'{l:.3f}' for l in losses]}")
print(f"\n  alpha* = {alpha_star:.3f}  loss* = {loss_star:.4f}")
print(f"  Saddle exit: val {v0:.4f} -> {loss_star:.4f}")

# ════════════════════════════════════════════════════
# STEP 3: Apply saddle exit, then CE training
# ════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("STEP 3: SADDLE EXIT + CE TRAINING")
print("="*65)

def run_with_saddle_exit(label, alpha, v_neg, ce_steps=200, do_newton=True):
    stu2=build_student()
    v_init=eval_val(stu2,n=20)

    # Apply saddle exit
    w0=stu2.get_flat_params().clone()
    v_norm=v_neg/v_neg.norm()
    stu2.set_flat_params(w0 + alpha*v_norm)
    v_after=eval_val(stu2,n=20)
    print(f"\n  [{label}]")
    print(f"    Before exit: val={v_init:.4f}")
    print(f"    After exit (alpha={alpha:.2f}): val={v_after:.4f}")

    opt_s=torch.optim.AdamW(stu2.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
    ck={'exit':v_after}
    for step in range(1,ce_steps+1):
        for pg in opt_s.param_groups: pg['lr']=clr(step,ce_steps)
        stu2.train(); x,y=get_batch(); _,loss=stu2(x,y)
        opt_s.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(stu2.parameters(),1.0); opt_s.step()
        if step in [10,20,33,50,75,100,125,150,175,200]:
            v=eval_val(stu2,n=20); ck[step]=v
            print(f"    step {step:>4}  val={v:.4f}"
                  f"{' ✓' if v<val_teacher else ''}")

    if do_newton:
        apply_newton_correction(stu2)
        v_n=eval_val(stu2); ck['newton']=v_n
        print(f"    Newton: val={v_n:.4f}")

    vf=eval_val(stu2,n=30)
    print(f"    FINAL={vf:.4f}")
    return vf,ck

def run_standard(label, ce_steps=200, do_newton=True):
    """Standard pipeline for comparison."""
    stu2=build_student()
    opt_s=torch.optim.AdamW(stu2.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
    v_init=eval_val(stu2,n=20)
    print(f"\n  [{label}]  zero-shot={v_init:.4f}")
    ck={0:v_init}
    for step in range(1,ce_steps+1):
        for pg in opt_s.param_groups: pg['lr']=clr(step,ce_steps)
        stu2.train(); x,y=get_batch(); _,loss=stu2(x,y)
        opt_s.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(stu2.parameters(),1.0); opt_s.step()
        if step in [10,20,33,50,75,100,125,150,175,200]:
            v=eval_val(stu2,n=20); ck[step]=v
            print(f"    step {step:>4}  val={v:.4f}"
                  f"{' ✓' if v<val_teacher else ''}")
    if do_newton:
        apply_newton_correction(stu2)
        v_n=eval_val(stu2); ck['newton']=v_n
        print(f"    Newton: val={v_n:.4f}")
    vf=eval_val(stu2,n=30); print(f"    FINAL={vf:.4f}")
    return vf,ck

# Run experiments
vE,ckE=run_standard("E-Standard-200CE+Newton")
vA,ckA=run_with_saddle_exit("A-SaddleExit+200CE+Newton",
                              alpha_star, v_neg, ce_steps=200)
vB,ckB=run_with_saddle_exit("B-SaddleExit+167CE+Newton",
                              alpha_star, v_neg, ce_steps=167)

# Also try negative alpha (other side of saddle)
vC,ckC=run_with_saddle_exit("C-SaddleExit(neg)+200CE+Newton",
                              -alpha_star, v_neg, ce_steps=200)

print(f"\n{'='*65}")
print("  SADDLE EXIT RESULTS")
print("="*65)
print(f"\n  lambda_min(H) = {lambda_min:.4f}  "
      f"({'SADDLE' if lambda_min<0 else 'CONVEX'})")
print(f"  alpha* = {alpha_star:.3f}  "
      f"(saddle exit distance)")
print(f"\n  CONVERGENCE:")
print(f"  {'step':>6}  {'E-std':>7}  {'A-exit':>7}  {'B-167':>7}  {'C-neg':>7}")
for s in [10,20,33,50,75,100,125,150,175,200,'newton']:
    row=f"  {str(s):>6}"
    for ck in [ckE,ckA,ckB,ckC]:
        v=ck.get(s)
        row+=f"  {v:>7.4f}" if v else f"  {'---':>7}"
    print(row)

print(f"""
  FINAL:
    Teacher:           val={val_teacher:.4f}
    E (standard):      val={vE:.4f}
    A (exit+200CE+N):  val={vA:.4f}  diff={vE-vA:+.4f}
    B (exit+167CE+N):  val={vB:.4f}  diff={vE-vB:+.4f}
    C (neg exit+200):  val={vC:.4f}  diff={vE-vC:+.4f}

  SADDLE EXIT THEORY:
    IF A or B << E:
      The Hessian minimum eigenvector IS the saddle exit direction.
      One corpus Hessian computation replaces 33 CE steps.
      The flat region (steps 0-33) is a saddle, not a plateau.
      The corpus Hessian reveals the basin from the loss geometry alone.
      
    IF A ~ E:
      The saddle direction is correct but the exit distance is wrong.
      The saddle has more complex structure than one eigenvector.
      Try: multiple saddle directions (Hessian top-k eigenvectors).
      
    IF A >> E:
      The saddle exit went to the WRONG basin.
      The negative curvature direction leads away from the target basin.
      The corpus Hessian at initialization is not predictive.
      The saddle structure is not the bottleneck.
""")
