#!/usr/bin/env python3
"""
Homotopy Continuation v2 — Temperature Annealing with Newton Corrector
=======================================================================
The correct homotopy parameter is the softmax TEMPERATURE τ.

H(θ, τ) = ∇_θ L(θ; softmax(QK^T / (τ·√d))) = 0

At τ=τ_max (high temp): attention → uniform, WK/WQ decouple from loss.
  Known solution: spectral embedding E_0, random WK/WQ (they don't matter).

At τ=1 (standard): full attention coupling.
  Unknown solution: θ* (what 167 CE steps find).

The path θ*(τ) is smooth because softmax temperature varies continuously.
No discontinuity. No incompatible loss functions.

PREDICTOR-CORRECTOR:
  For each τ step from τ_max → 1:
  Predictor: gradient step on L(θ; τ_new)   — moves along the path tangent
  Corrector: LM Newton step on L(θ; τ_new)  — snaps back to solution manifold
             (H(θ,τ_new) + μI) d = -∇L(θ; τ_new)
             Accept if ||∇L|| decreases

This is the EXACT structure of HomotopyContinuation.jl's parameter homotopy,
with τ as the parameter instead of polynomial coefficients.
"""
import json, math, warnings, collections, os, copy, sys
warnings.filterwarnings('ignore')
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_STU=6; BATCH=8; SEQ=64; LR=3e-4

for f in ['/tmp/train_ids.json','/tmp/val_ids.json','/tmp/vocab.json']:
    if not os.path.exists(f):
        print(f"ERROR: {f} missing."); sys.exit(1)

with open('/tmp/train_ids.json') as f: train_ids=list(map(int,json.load(f)))
with open('/tmp/val_ids.json')   as f: val_ids  =list(map(int,json.load(f)))
with open('/tmp/vocab.json')     as f: _v=json.load(f)
VOCAB=len(_v) if isinstance(_v,list) else len(_v)
train_t=torch.tensor(train_ids,dtype=torch.long)
val_t  =torch.tensor(val_ids,  dtype=torch.long)
print(f"VOCAB={VOCAB}, train={len(train_ids)} ({len(train_ids)//1364} loops)")

def get_batch(split='train'):
    data=val_t if split=='val' else train_t
    ix=torch.randint(0,len(data)-SEQ-1,(BATCH,))
    return (torch.stack([data[i:i+SEQ] for i in ix]),
            torch.stack([data[i+1:i+SEQ+1] for i in ix]))

# ── Temperature-parameterised model ──────────────────────────────────────────
class AttnTemp(nn.Module):
    """Attention with adjustable temperature on the softmax."""
    def __init__(self,d,nh):
        super().__init__(); self.nh=nh; self.dh=d//nh; self.sc=math.sqrt(d//nh)
        self.WQ=nn.Linear(d,d,bias=False); self.WK=nn.Linear(d,d,bias=False)
        self.WV=nn.Linear(d,d,bias=False); self.op=nn.Linear(d,d,bias=False)
        self.ln=nn.LayerNorm(d)
        for w in [self.WQ,self.WK,self.WV,self.op]: nn.init.normal_(w.weight,std=0.02)
    def forward(self,h,tau=1.0):
        B,S,D_=h.shape; H=self.nh; dh=self.dh
        Q=self.WQ(h).view(B,S,H,dh).transpose(1,2)
        K=self.WK(h).view(B,S,H,dh).transpose(1,2)
        V=self.WV(h).view(B,S,H,dh).transpose(1,2)
        # Temperature: divide scores by tau (high tau = flat/uniform attention)
        sc = Q@K.transpose(-2,-1) / (tau * self.sc)
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

class BlockTemp(nn.Module):
    def __init__(self,d,nh): super().__init__(); self.attn=AttnTemp(d,nh); self.ff=FF(d)
    def forward(self,h,tau=1.0): return self.ff(self.attn(h,tau))

class LMTemp(nn.Module):
    def __init__(self,d,nh,nl):
        super().__init__()
        self.te=nn.Embedding(VOCAB,d); self.pe=nn.Embedding(512,d)
        self.blocks=nn.ModuleList([BlockTemp(d,nh) for _ in range(nl)])
        self.ln_f=nn.LayerNorm(d)
        self.head=nn.Linear(d,VOCAB,bias=False); self.head.weight=self.te.weight
        nn.init.normal_(self.te.weight,std=0.02); nn.init.normal_(self.pe.weight,std=0.02)
    def forward(self,x,y=None,tau=1.0):
        h=self.te(x)+self.pe(torch.arange(x.shape[1]))
        for b in self.blocks: h=b(h,tau)
        logits=self.head(self.ln_f(h))
        return logits,(F.cross_entropy(logits.view(-1,VOCAB),y.view(-1)) if y is not None else None)
    def flat_params(self):
        return torch.cat([p.data.flatten() for p in self.parameters()])
    def set_flat(self,f):
        idx=0
        for p in self.parameters():
            n=p.numel(); p.data.copy_(f[idx:idx+n].reshape(p.shape)); idx+=n

def eval_val(m,n=20,tau=1.0):
    m.eval(); ls=[]
    with torch.no_grad():
        for _ in range(n):
            x,y=get_batch('val'); _,l=m(x,y,tau=tau); ls.append(l.item())
    return float(np.mean(ls))

def grad_at_tau(model, tau, n_batches=8):
    """Gradient and loss at temperature tau."""
    model.zero_grad()
    ls=[]
    for _ in range(n_batches):
        x,y=get_batch(); _,l=model(x,y,tau=tau); ls.append(l)
    loss=torch.stack(ls).mean(); loss.backward()
    g=torch.cat([p.grad.flatten() if p.grad is not None
                 else torch.zeros(p.numel())
                 for p in model.parameters()]).detach()
    model.zero_grad()
    return g, float(loss)

def hvp_at_tau(model, v, tau, n_batches=4):
    """HVP at temperature tau."""
    model.zero_grad()
    ls=[]
    for _ in range(n_batches):
        x,y=get_batch(); _,l=model(x,y,tau=tau); ls.append(l)
    loss=torch.stack(ls).mean()
    grads=torch.autograd.grad(loss,list(model.parameters()),create_graph=True)
    gflat=torch.cat([g.flatten() for g in grads])
    gv=(gflat*v.detach()).sum()
    hv=torch.autograd.grad(gv,list(model.parameters()),retain_graph=False)
    model.zero_grad()
    return torch.cat([h.flatten() for h in hv]).detach()

def lm_corrector(model, tau, n_cg=6, mu=0.1, n_batches=8):
    """
    LM Newton corrector at fixed tau.
    Solves (H(τ) + μI)d = -∇L(θ; τ), applies with line search.
    Returns (new_loss, accepted).
    """
    g, l0 = grad_at_tau(model, tau, n_batches)
    g_norm = float(g.norm())

    # CG solve
    d=torch.zeros_like(g); r=-g.clone(); p_cg=r.clone()
    rr=float((r*r).sum())
    for _ in range(n_cg):
        Hp=hvp_at_tau(model,p_cg,tau,n_batches=3)+mu*p_cg
        alpha=rr/max(float((p_cg*Hp).sum()),1e-10)
        d=d+alpha*p_cg; r=r-alpha*Hp
        rr_new=float((r*r).sum())
        p_cg=r+(rr_new/max(rr,1e-10))*p_cg; rr=rr_new

    theta0=model.flat_params()
    for scale in [1.0, 0.5, 0.25, 0.1, 0.05]:
        model.set_flat(theta0 + scale*d)
        g_new, l_new = grad_at_tau(model, tau, n_batches=4)
        # Accept if GRADIENT NORM decreases (not just loss)
        if float(g_new.norm()) < g_norm * 0.99:
            return l_new, True, float(g_new.norm())
    model.set_flat(theta0)
    return l0, False, g_norm

# ── Build spectral init ───────────────────────────────────────────────────────
print("\n[OFFLINE] Spectral embedding...")
bigram=collections.Counter()
for i in range(len(train_ids)-1):
    a,b=train_ids[i],train_ids[i+1]
    if a<VOCAB and b<VOCAB: bigram[(a,b)]+=1
rows,cols,vals=[],[],[]
for (a,b),cnt in bigram.items():
    rows.append(a); cols.append(b); vals.append(float(cnt))
W_sp=sp.csr_matrix((vals,(rows,cols)),shape=(VOCAB,VOCAB),dtype=np.float32)
W_sp=W_sp+W_sp.T
d_inv=np.array(1.0/(W_sp.sum(1)+1e-8)).flatten()
Dsi=sp.diags(np.sqrt(d_inv))
L_sym=sp.eye(VOCAB)-Dsi@W_sp@Dsi
evals,evecs=spla.eigsh(L_sym,k=D+1,which='SM',tol=1e-4,maxiter=2000)
idx_s=np.argsort(evals)
evecs=evecs[:,idx_s][:,1:D+1]
scales=1.0/(np.sqrt(evals[idx_s[1:D+1]])+1e-8)
E_0=(evecs*scales[np.newaxis,:]).astype(np.float32)
E_0=(E_0/(E_0.std()+1e-8)*0.02)

torch.manual_seed(99)
model=LMTemp(D,N_HEADS,N_STU)
model.te.weight.data.copy_(torch.tensor(E_0))

# ── Verify start system ───────────────────────────────────────────────────────
print("\n[VERIFY] Start system at τ=100 (near-uniform attention):")
TAU_START = 20.0  # high temperature = near-uniform attention
TAU_END   = 1.0   # standard temperature

v_tau_start = eval_val(model, tau=TAU_START)
v_tau_end   = eval_val(model, tau=TAU_END)
g_start, l_start = grad_at_tau(model, TAU_START, n_batches=8)

# Check WK/WQ gradients at high tau
idx_c=0; gnorms_tau={}
for name,param in model.named_parameters():
    n=param.numel(); g_slice=g_start[idx_c:idx_c+n]; idx_c+=n
    pt=('WQ' if '.attn.WQ.' in name else 'WK' if '.attn.WK.' in name else
        'WV' if '.attn.WV.' in name else 'WO' if '.attn.op.' in name else
        'Emb' if 'te.weight' in name else 'FF' if '.ff.' in name else 'other')
    gnorms_tau[pt]=gnorms_tau.get(pt,0)+float(g_slice.norm()**2)
gnorms_tau={k:math.sqrt(v) for k,v in gnorms_tau.items()}

print(f"  val at τ={TAU_START}: {v_tau_start:.4f}")
print(f"  val at τ={TAU_END}:   {v_tau_end:.4f}")
print(f"  Gradients at τ={TAU_START}:")
for k,v in sorted(gnorms_tau.items(),key=lambda x:-x[1]):
    print(f"    |g_{k}| = {v:.5f}")
wk_ratio = gnorms_tau.get('WK',1e-8)
emb_ratio = gnorms_tau.get('Emb',0)
print(f"  |gEmb|/|gWK| at τ={TAU_START}: {emb_ratio/max(wk_ratio,1e-8):.1f}×")
print(f"  (at τ=1 standard this is 268×, at high τ WK should be suppressed more)")

# ── Homotopy path: τ from TAU_START → TAU_END ────────────────────────────────
print(f"\n{'='*60}")
print(f"HOMOTOPY PATH: τ = {TAU_START} → {TAU_END}")
print(f"{'='*60}")

N_STEPS = 20
tau_schedule = np.exp(np.linspace(np.log(TAU_START), np.log(TAU_END), N_STEPS+1))

print(f"\n  {'τ':>7}  {'val(τ=1)':>9}  {'|∇L|':>8}  {'corrector':>10}")
print("  " + "-"*45)

for step_idx, tau in enumerate(tau_schedule[1:]):
    # PREDICTOR: gradient step at new (lower) tau
    # Use moderately larger step than standard
    eta = LR * 3.0
    g_pred, l_pred = grad_at_tau(model, tau, n_batches=6)
    g_pred_norm = float(g_pred.norm())

    with torch.no_grad():
        theta_pred = model.flat_params()
        # Normalized gradient step
        model.set_flat(theta_pred - eta * g_pred / max(g_pred_norm, 1e-8) * g_pred_norm)

    # CORRECTOR: LM Newton step at current tau
    n_accepted = 0
    g_norm_before = g_pred_norm
    for _ in range(3):
        l_new, accepted, g_norm_new = lm_corrector(model, tau, n_cg=6, mu=0.1)
        if accepted:
            n_accepted += 1
            g_norm_before = g_norm_new

    # Evaluate at standard temperature (τ=1) to track real val
    v_real = eval_val(model, n=8, tau=1.0)
    print(f"  τ={tau:>6.3f}  val={v_real:>9.4f}  |∇L|={g_norm_before:>8.4f}  "
          f"corr={n_accepted}/3  {'✓' if n_accepted>0 else '~'}")

v_hc = eval_val(model, n=30, tau=1.0)
print(f"\n  Homotopy final val (τ=1): {v_hc:.4f}")

# ── Reference ─────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("REFERENCE: 167 CE steps from same spectral init")
print(f"{'='*60}")
torch.manual_seed(99)
model_ref=LMTemp(D,N_HEADS,N_STU)
model_ref.te.weight.data.copy_(torch.tensor(E_0))
opt=torch.optim.AdamW(model_ref.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
for s in range(1,168):
    model_ref.train(); x,y=get_batch(); _,l=model_ref(x,y,tau=1.0)
    opt.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(model_ref.parameters(),1.0); opt.step()
    if s in [25,50,100,167]:
        print(f"  CE {s}: val={eval_val(model_ref,n=8):.4f}")
v_ref=eval_val(model_ref,n=40)

# ── Homotopy + residual ────────────────────────────────────────────────────────
model_res=copy.deepcopy(model)
opt_r=torch.optim.AdamW(model_res.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
for s in range(1,26):
    model_res.train(); x,y=get_batch(); _,l=model_res(x,y,tau=1.0)
    opt_r.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(model_res.parameters(),1.0); opt_r.step()
    if s in [5,10,25]:
        print(f"  HC+CE {s}: val={eval_val(model_res,n=8):.4f}")
v_res=eval_val(model_res,n=30)

print(f"""
{'='*60}
HOMOTOPY v2 RESULTS
{'='*60}
  τ_start={TAU_START} (near-uniform), τ_end=1.0 (standard)
  Path steps: {N_STEPS}, corrector: LM Newton on standard loss

  Spectral init (τ=1):        val={eval_val(model,n=10,tau=1.0) if False else v_tau_end:.4f}
  Homotopy ({N_STEPS} steps):        val={v_hc:.4f}
  Homotopy + 25 CE:           val={v_res:.4f}
  Reference (167 CE):         val={v_ref:.4f}

  KEY: does |gWK| decrease faster than at τ=1?
       If correctors accept → temperature path is smooth ✓
       If correctors reject → curvature still too high, reduce step size
""")
