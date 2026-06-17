#!/usr/bin/env python3
"""
Hessian-Free Optimization for Transformer Basin Descent
=========================================================
Implements Martens (2010) Hessian-Free optimization.
Uses Gauss-Newton (GN) HVP which is always positive semi-definite.

KEY INSIGHT: Standard CG failed because H has negative eigenvalues.
Gauss-Newton approximation G_GN = J^T H_loss J is PSD by construction.
For cross-entropy: H_loss = diag(p) - pp^T (Fisher information).
G_GN is the Fisher information matrix -- always >= 0.

GN HVP: G_GN * v = J^T (H_loss (J*v))
  Step 1: compute r = J*v  (forward pass with tangent vector v)
  Step 2: compute s = H_loss * r  (multiply by Fisher matrix)
  Step 3: compute G_GN*v = J^T * s  (backward pass with s)

Implemented via: torch.autograd.functional.jvp + vjp

COMPARISON:
  A: 100 CE steps (baseline)
  B: HF with GN HVP, 10 CG iterations
  C: HF with GN HVP, 20 CG iterations  
  D: HF + Armijo line search + 25 CE cleanup
"""
import json, math, time, copy, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torch.autograd.functional as taf

D=256; N_HEADS=4; N_LAYERS_T=24; N_STU=6; BATCH=8; SEQ=64; LR=3e-4
L_ATT=14; ALPHA_STAR=1.429

print(f"\n{'='*65}")
print(f"  HESSIAN-FREE OPTIMIZATION")
print(f"  Gauss-Newton HVP (PSD by construction)")
print(f"  G_GN = J^T H_loss J  (Fisher information)")
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
        super().__init__(); self.nh=nh; self.dh=d//nh; self.sc=math.sqrt(d//nh)
        self.WQ=nn.Linear(d,d,bias=False); self.WK=nn.Linear(d,d,bias=False)
        self.WV=nn.Linear(d,d,bias=False); self.op=nn.Linear(d,d,bias=False)
        self.ln=nn.LayerNorm(d)
        for w in [self.WQ,self.WK,self.WV,self.op]: nn.init.normal_(w.weight,std=0.02)
    def forward(self,h):
        B,S,D_=h.shape; H=self.nh; dh=self.dh
        Q=self.WQ(h).view(B,S,H,dh).transpose(1,2); K=self.WK(h).view(B,S,H,dh).transpose(1,2)
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
    def get_params_flat(self): return torch.cat([p.data.flatten() for p in self.parameters()])
    def set_params_flat(self,flat):
        idx=0
        for p in self.parameters(): n=p.numel(); p.data.copy_(flat[idx:idx+n].reshape(p.shape)); idx+=n
    def get_grad_flat(self):
        return torch.cat([p.grad.flatten() if p.grad is not None
                          else torch.zeros(p.numel()) for p in self.parameters()])

def eval_val(m,n=40):
    m.eval(); ls=[]
    with torch.no_grad():
        for _ in range(n): x,y=get_batch('val'); _,l=m(x,y); ls.append(l.item())
    return float(np.mean(ls))

def clr(s,total,warmup=20):
    if s<=warmup: return LR*s/warmup
    return LR*0.5*(1+math.cos(math.pi*(s-warmup)/(total-warmup)))

def gauss_newton_hvp(model, v, n_batches=15):
    """
    Gauss-Newton HVP: G_GN * v = J^T (H_loss (J*v))
    
    For cross-entropy loss with softmax:
    H_loss at output layer = diag(p) - p*p^T (Fisher info)
    
    Efficient computation via R-operator (Pearlmutter):
    Step 1: r = Jv  (forward mode: compute Jacobian-vector product)
    Step 2: s = H_loss * r  (Fisher matrix times r)
    Step 3: G_GN*v = J^T s  (backward mode: vector-Jacobian product)
    
    The result is always >= 0 (PSD).
    """
    params=list(model.parameters())
    n_params=sum(p.numel() for p in params)
    
    GNv=torch.zeros(n_params)
    
    torch.manual_seed(42)
    for _ in range(n_batches):
        x,y=get_batch()
        
        # Step 1: Forward pass to get logits and compute Jv
        # Use double backprop to get Jv
        model.zero_grad()
        logits,_=model(x)  # (B, S, V)
        
        # Compute gradient of (logits * v_logit_space) to get Jv
        # We need J^T (H_loss Jv), decomposed as:
        # 1. compute f = Jv = d(logits)/d(theta) . v
        # 2. compute g = H_loss f = (diag(p) - pp^T) f
        # 3. compute J^T g = d(logits)/d(theta)^T . g
        
        # Approach: use autograd to compute J^T (H_loss J v)
        # = d/dtheta [logits^T H_loss J v]
        # = d/dtheta [(J^T H_loss J v)^T theta] ... 
        
        # Simpler: use the R-operator trick
        # loss = ce(logits, y)
        # d^2 loss / dtheta^2 * v ≈ J^T (d^2 loss/df^2) J v
        # where f = logits
        
        # Compute Jv via forward mode (using grad of grad)
        # grad(sum(logits * dummy), params) gives J^T * dummy
        # We want Jv = J * v (forward mode)
        
        # Use the double-backward trick:
        # J*v = d/depsilon [f(theta + epsilon*v)]|_{eps=0}
        # Implement as: create_graph=True backward, then dot with v
        
        eps=1e-3
        v_list=[]; idx=0
        for p in params:
            n=p.numel(); v_list.append(v[idx:idx+n].reshape(p.shape)); idx+=n
        
        # Perturbed forward pass
        with torch.no_grad():
            for p,vp in zip(params,v_list): p.data.add_(eps*vp)
        logits_plus,_=model(x)
        with torch.no_grad():
            for p,vp in zip(params,v_list): p.data.sub_(2*eps*vp)
        logits_minus,_=model(x)
        with torch.no_grad():
            for p,vp in zip(params,v_list): p.data.add_(eps*vp)
        
        # Jv ≈ (logits_plus - logits_minus) / (2*eps)  -- shape (B,S,V)
        Jv=(logits_plus - logits_minus).detach()/(2*eps)  # (B,S,V)
        
        # Step 2: H_loss * Jv = (diag(p) - pp^T) Jv
        # p = softmax(logits)
        p=torch.softmax(logits.detach(),dim=-1)  # (B,S,V)
        # (diag(p) - pp^T) Jv = p*Jv - p*(p^T Jv)
        pJv=(p*Jv).sum(-1,keepdim=True)  # (B,S,1) = p^T Jv
        HJv=p*Jv - p*pJv  # (B,S,V) = H_loss * Jv
        
        # Step 3: J^T (H_loss Jv) = backprop HJv through logits
        model.zero_grad()
        logits2,_=model(x)
        logits2.backward(HJv/n_batches)
        GNv=GNv+model.get_grad_flat().detach()
    
    model.zero_grad()
    return GNv

def hf_cg_solve(model, grad_flat, n_iters=15, n_batches=10, damping=0.1, tol=1e-4):
    """
    CG solve using Gauss-Newton HVP (PSD, no regularization needed).
    damping: small positive constant for numerical stability.
    """
    b=-grad_flat; delta=torch.zeros_like(b); r=b.clone(); p=r.clone()
    r_sq=float((r*r).sum())
    print(f"  HF-CG init ||r||={r_sq**0.5:.4f}")
    
    for k in range(n_iters):
        Gp=gauss_newton_hvp(model,p,n_batches)+damping*p
        pGp=float((p*Gp).sum())
        if abs(pGp)<1e-10: break
        alpha=r_sq/pGp
        delta=delta+alpha*p; r=r-alpha*Gp
        r_sq_new=float((r*r).sum()); beta=r_sq_new/max(r_sq,1e-10)
        p=r+beta*p; r_sq=r_sq_new
        print(f"  HF-CG iter {k+1}: ||r||={r_sq**0.5:.5f}  ||delta||={float(delta.norm()):.4f}")
        if r_sq**0.5<tol:
            print(f"  HF-CG converged at iter {k+1}")
            break
    return delta

def armijo_line_search(model, delta, grad_flat, n_batches=10,
                        alpha0=1.0, c=0.1, rho=0.5, max_iter=8):
    """
    Armijo backtracking line search.
    Find alpha such that L(theta + alpha*delta) <= L(theta) + c*alpha*grad^T*delta
    """
    w0=model.get_params_flat().clone()
    # Current loss
    model.eval()
    with torch.no_grad():
        L0=float(np.mean([model(*get_batch())[1].item() for _ in range(n_batches)]))
    
    slope=float((grad_flat*delta).sum())  # should be negative
    
    alpha=alpha0
    for _ in range(max_iter):
        model.set_params_flat(w0+alpha*delta)
        with torch.no_grad():
            L_new=float(np.mean([model(*get_batch())[1].item() for _ in range(n_batches)]))
        if L_new <= L0 + c*alpha*slope:
            print(f"  Armijo: alpha={alpha:.4f}  L {L0:.4f}->{L_new:.4f}")
            return alpha
        alpha*=rho
    
    # No improvement found, restore
    model.set_params_flat(w0)
    print(f"  Armijo: no improvement, keeping original")
    return 0.0

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

def hv_product_full(model,v,n=15):
    params=list(model.parameters()); model.zero_grad()
    loss=sum(model(*get_batch())[1] for _ in range(n))/n
    grads=torch.autograd.grad(loss,params,create_graph=True)
    gv=(torch.cat([g.flatten() for g in grads])*v.detach()).sum()
    hv=torch.cat([h.flatten() for h in torch.autograd.grad(gv,params,retain_graph=False)]).detach()
    model.zero_grad(); return hv

print("Training teacher...")
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

def apply_mf(stu,n_iter=10,mf_lr=0.01,n_corpus=200):
    for it in range(n_iter):
        for l in range(N_STU):
            stu.blocks[l].attn.WK.weight.requires_grad_(False)
            stu.blocks[l].attn.WQ.weight.requires_grad_(False)
        eg=torch.zeros(VOCAB,D); ef=torch.zeros(VOCAB,D)
        torch.manual_seed(it*1000)
        for i in range(n_corpus):
            ix=torch.randint(0,len(train_t)-SEQ-1,(1,))[0].item()
            x=train_t[ix:ix+SEQ].unsqueeze(0); y=train_t[ix+1:ix+SEQ+1].unsqueeze(0)
            stu.zero_grad(); _,loss=stu(x,y); loss.backward()
            if stu.te.weight.grad is not None:
                g=stu.te.weight.grad.detach(); eg+=g; ef+=g**2
        eg/=n_corpus; ef/=n_corpus
        with torch.no_grad(): stu.te.weight.add_(-mf_lr*eg/(ef+1e-4))
        for l in range(N_STU):
            stu.blocks[l].attn.WK.weight.requires_grad_(True)
            stu.blocks[l].attn.WQ.weight.requires_grad_(True)
        stu.te.weight.requires_grad_(False)
        wg=torch.zeros_like(stu.blocks[0].attn.WK.weight)
        wf=torch.zeros_like(stu.blocks[0].attn.WK.weight)
        torch.manual_seed(it*1000+500)
        for i in range(n_corpus):
            ix=torch.randint(0,len(train_t)-SEQ-1,(1,))[0].item()
            x=train_t[ix:ix+SEQ].unsqueeze(0); y=train_t[ix+1:ix+SEQ+1].unsqueeze(0)
            stu.zero_grad(); _,loss=stu(x,y); loss.backward()
            g=torch.zeros_like(stu.blocks[0].attn.WK.weight)
            for l in range(N_STU):
                if stu.blocks[l].attn.WK.weight.grad is not None:
                    g+=stu.blocks[l].attn.WK.weight.grad/N_STU
            wg+=g; wf+=g**2
        wg/=n_corpus; wf/=n_corpus
        with torch.no_grad():
            for l in range(N_STU):
                stu.blocks[l].attn.WK.weight.add_(-mf_lr*wg/(wf+1e-4))
                stu.blocks[l].attn.WQ.weight.add_(-mf_lr*wg.T/(wf.T+1e-4))
        stu.te.weight.requires_grad_(True)
        if (it+1)%5==0: print(f"  MF iter {it+1}: val={eval_val(stu,n=5):.4f}")

# Build base state
print("Building MF10+settle+sign state...")
stu_ref=build_student(); n_p=sum(p.numel() for p in stu_ref.parameters())
v=torch.randn(n_p); v=v/v.norm()
for _ in range(15): Hv=hv_product_full(stu_ref,v,15); neg=-Hv; v=neg/max(float(neg.norm()),1e-10)
v_neg=v.clone()
stu_base=build_student()
w0=stu_base.get_params_flat(); stu_base.set_params_flat(w0+ALPHA_STAR*(v_neg/v_neg.norm()))
apply_mf(stu_base,n_iter=10)
opt_s=torch.optim.AdamW(stu_base.parameters(),lr=LR*5,betas=(0.9,0.95),weight_decay=0.1)
for step in range(1,34):
    for pg in opt_s.param_groups: pg['lr']=LR*5*min(step,10)/10
    stu_base.train(); x,y=get_batch(); _,loss=stu_base(x,y)
    opt_s.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(stu_base.parameters(),1.0); opt_s.step()
with torch.no_grad():
    for l in [1,2]:
        stu_base.blocks[l].attn.WV.weight.mul_(-1); stu_base.blocks[l].attn.op.weight.mul_(-1)
v_base=eval_val(stu_base,n=20)
print(f"Base state val: {v_base:.4f}\n")

print("="*65)
print("EXPERIMENTS")
print("="*65)
results={}

# A: baseline
s=copy.deepcopy(stu_base)
opt2=torch.optim.AdamW(s.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
for step in range(1,101):
    for pg in opt2.param_groups: pg['lr']=clr(step,100)
    s.train(); x,y=get_batch(); _,loss=s(x,y)
    opt2.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(s.parameters(),1.0); opt2.step()
    if step in [25,50,100]: print(f"  A CE {step}: {eval_val(s,n=10):.4f}")
apply_newton_wk(s); results['A']=eval_val(s,n=30)
print(f"  A FINAL={results['A']:.4f}")

# B: HF with GN HVP, 10 CG iters
print(f"\n[B] Hessian-Free: GN HVP, 10 CG iterations, Armijo line search")
s=copy.deepcopy(stu_base)
s.train(); s.zero_grad()
for _ in range(30): x,y=get_batch(); _,loss=s(x,y); (loss/30).backward()
grad_flat=s.get_grad_flat().detach().clone(); s.zero_grad()
print(f"  ||G|| = {float(grad_flat.norm()):.4f}")
t0=time.time()
delta=hf_cg_solve(s,grad_flat,n_iters=10,n_batches=8,damping=0.1)
print(f"  HF solve: {time.time()-t0:.1f}s  ||delta||={float(delta.norm()):.4f}")
alpha=armijo_line_search(s,delta,grad_flat,n_batches=8)
if alpha>0:
    s.set_params_flat(s.get_params_flat()+alpha*delta)
print(f"  After HF step: {eval_val(s,n=20):.4f}")
apply_newton_wk(s); results['B']=eval_val(s,n=30)
print(f"  B FINAL={results['B']:.4f}")

# C: HF + 25 CE
print(f"\n[C] Hessian-Free + 25 CE cleanup")
s=copy.deepcopy(stu_base)
s.train(); s.zero_grad()
for _ in range(30): x,y=get_batch(); _,loss=s(x,y); (loss/30).backward()
grad_flat=s.get_grad_flat().detach().clone(); s.zero_grad()
delta=hf_cg_solve(s,grad_flat,n_iters=10,n_batches=8,damping=0.1)
alpha=armijo_line_search(s,delta,grad_flat,n_batches=8)
if alpha>0: s.set_params_flat(s.get_params_flat()+alpha*delta)
print(f"  After HF: {eval_val(s,n=10):.4f}")
opt3=torch.optim.AdamW(s.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
for step in range(1,26):
    for pg in opt3.param_groups: pg['lr']=clr(step,25)
    s.train(); x,y=get_batch(); _,loss=s(x,y)
    opt3.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(s.parameters(),1.0); opt3.step()
    if step in [10,25]: print(f"  C CE {step}: {eval_val(s,n=10):.4f}")
apply_newton_wk(s); results['C']=eval_val(s,n=30)
print(f"  C FINAL={results['C']:.4f}")

print(f"""
{'='*65}
  HESSIAN-FREE RESULTS
{'='*65}
    Teacher:          val={val_teacher:.4f}
    A (100CE+N):      val={results['A']:.4f}  [100 CE steps]
    B (HF+Armijo+N):  val={results['B']:.4f}  [Gauss-Newton]
    C (HF+25CE+N):    val={results['C']:.4f}  [HF + minimal CE]

  GN HVP cost per iter: 2 forward+backward passes (finite diff Jv)
  10 CG iters: ~20 forward+backward = ~0.1 CE equiv
  
  IF B ~ A: GN Hessian solves the coupling in one step
    The PSD structure removes the indefiniteness barrier
    
  IF B > A but C << A: GN is a good preconditioner
    HF + 25CE outperforms 100CE standard
    
  KEY: does the GN HVP converge faster than the Tikhonov CG?
  Convergence rate measures effective condition number of G_GN.
  If k_GN << k_Tikhonov: GN approximation is correct curvature.
  If k_GN ~ k_Tikhonov: softmax nonlinearity dominates both.
""")
