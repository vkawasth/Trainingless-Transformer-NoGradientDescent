#!/usr/bin/env python3
"""
Natural Gradient Descent vs Standard Adam/SGD
===============================================
The Kähler/Fisher metric proposal is Natural Gradient Descent (NGD).
Theory: NGD follows geodesics on the statistical manifold MF.
It should converge faster than SGD by preconditioning with FIM^{-1}.

PRACTICAL APPROXIMATION (K-FAC style):
  Full FIM is n×n (n=params) — intractable.
  Approximate: block-diagonal FIM, one block per layer.
  For layer l: G_l ≈ E[∇_W L ∇_W L^T]  (empirical Fisher)
  
  Preconditioned gradient: g_nat = (G_l + λI)^{-1} g_l
  
  This IS what Adam approximates via v_t = β₂v_{t-1} + (1-β₂)g²
  Adam diagonal ≈ diag(FIM). NGD uses the full block.

THREE CONDITIONS:
  A: Standard AdamW (baseline — already a diagonal FIM approximation)
  B: SGD + Nesterov (no FIM preconditioning)
  C: SGD + block-diagonal empirical Fisher preconditioning (true NGD approx)

The question: does full block Fisher give measurably faster convergence
than Adam's diagonal approximation?
"""
import json, math, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

D=128; N_HEADS=4; N_LAYERS=8; BATCH=8; SEQ=64
LR_ADAM=3e-4; LR_SGD=0.05; TARGET=4.0; MAX_STEPS=300; LOG=25

print(f"\n{'='*65}")
print(f"  NATURAL GRADIENT vs ADAM vs SGD")
print(f"  Does FIM preconditioning beat Adam's diagonal approximation?")
print(f"  d={D}  layers={N_LAYERS}")
print(f"{'='*65}\n")

with open('/tmp/train_ids.json') as f: train_ids=json.load(f)
with open('/tmp/val_ids.json')   as f: val_ids=json.load(f)
with open('/tmp/vocab.json')     as f: vocab=json.load(f)
VOCAB=len(vocab)
train_t=torch.tensor(train_ids,dtype=torch.long)
val_t  =torch.tensor(val_ids,  dtype=torch.long)

def get_batch(split='train'):
    data=train_t if split=='train' else val_t
    ix=torch.randint(0,len(data)-SEQ-1,(BATCH,))
    return (torch.stack([data[i:i+SEQ]   for i in ix]),
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
        self._nl=nl
    def forward(self,x,y=None):
        h=self.te(x)+self.pe(torch.arange(x.shape[1]))
        for b in self.blocks: h=b(h)
        logits=self.head(self.ln_f(h))
        return logits,(F.cross_entropy(logits.view(-1,VOCAB),y.view(-1)) if y is not None else None)

def eval_val(model,n=40):
    model.eval(); ls=[]
    with torch.no_grad():
        for _ in range(n):
            x,y=get_batch('val'); _,l=model(x,y); ls.append(l.item())
    return float(np.mean(ls))

def clr_adam(s,total=MAX_STEPS,warmup=100):
    if s<=warmup: return LR_ADAM*s/warmup
    return LR_ADAM*0.5*(1+math.cos(math.pi*(s-warmup)/(total-warmup)))

def clr_sgd(s,total=MAX_STEPS,warmup=50):
    if s<=warmup: return LR_SGD*s/warmup
    return LR_SGD*0.5*(1+math.cos(math.pi*(s-warmup)/(total-warmup)))

# ── Block-diagonal empirical Fisher preconditioning ───────────────────────────
class FisherPreconditioner:
    """
    Maintains a running EMA of the per-layer empirical Fisher (diagonal blocks).
    For each weight matrix W [d_out, d_in]:
      G_W ≈ E[g_W g_W^T]  (d_out × d_out) — left Kronecker factor
      A_W ≈ E[a a^T]       (d_in  × d_in)  — right Kronecker factor (input cov)
    
    K-FAC approximation: G ≈ G_W ⊗ A_W
    Preconditioned grad: vec(g_nat) = (G_W ⊗ A_W + λI)^{-1} vec(g)
    Which simplifies to: g_nat = G_W^{-1} g A_W^{-1}
    
    We use diagonal approximation for tractability:
    G_W ≈ diag(E[g²]) per output unit — this IS Adam's v_t.
    
    Full version (expensive but correct):
    For d=128: G_W is 128×128, A_W is 128×128.
    Update: G_nat = G_W^{-½} g A_W^{-½}
    """
    def __init__(self, model, damping=1e-3, ema=0.95, update_freq=10):
        self.damping=damping; self.ema=ema; self.freq=update_freq
        self.G={}; self.A={}  # per-param Fisher factors
        self._step=0
        # Register hooks to capture input activations
        self._inputs={}
        for name,mod in model.named_modules():
            if isinstance(mod,nn.Linear) and mod.weight.requires_grad:
                mod._fisher_name=name
                mod.register_forward_hook(self._save_input)

    def _save_input(self,mod,inp,out):
        if inp[0] is not None:
            self._inputs[mod._fisher_name]=inp[0].detach()

    def update(self, model):
        """Update Fisher estimates from current gradients and saved inputs."""
        self._step+=1
        if self._step % self.freq != 0: return
        for name,mod in model.named_modules():
            if isinstance(mod,nn.Linear) and mod.weight.grad is not None:
                g=mod.weight.grad.data   # [d_out, d_in]
                # Left factor: gradient outer product (output space)
                G_new=(g@g.T)/g.shape[1]   # [d_out, d_out]
                # Right factor: input covariance
                if name in self._inputs:
                    a=self._inputs[name]
                    if a.dim()==3: a=a.reshape(-1,a.shape[-1])
                    A_new=(a.T@a)/a.shape[0]   # [d_in, d_in]
                    A_new=A_new.numpy()
                else:
                    A_new=np.eye(g.shape[1])
                G_new=G_new.numpy()
                # EMA update
                if name in self.G:
                    self.G[name]=self.ema*self.G[name]+(1-self.ema)*G_new
                    self.A[name]=self.ema*self.A[name]+(1-self.ema)*A_new
                else:
                    self.G[name]=G_new; self.A[name]=A_new

    def precondition(self, model):
        """Apply K-FAC preconditioning to all gradients."""
        for name,mod in model.named_modules():
            if isinstance(mod,nn.Linear) and mod.weight.grad is not None:
                if name not in self.G: continue
                g=mod.weight.grad.data.numpy()   # [d_out, d_in]
                G=self.G[name]; A=self.A[name]
                d_=self.damping
                # Tikhonov-regularised inverse: (G + λI)^{-1}
                try:
                    Ginv=np.linalg.inv(G+d_*np.eye(G.shape[0]))
                    Ainv=np.linalg.inv(A+d_*np.eye(A.shape[0]))
                    g_nat=Ginv@g@Ainv
                    # Rescale to match gradient norm
                    scale=np.linalg.norm(g)/(np.linalg.norm(g_nat)+1e-8)
                    mod.weight.grad.data=torch.tensor(
                        g_nat*scale, dtype=torch.float32)
                except np.linalg.LinAlgError:
                    pass  # skip if singular

# ── Training runs ─────────────────────────────────────────────────────────────
def run(name, use_adam=True, use_ngd=False, seed=42):
    torch.manual_seed(seed)
    model=LM(D,N_HEADS,N_LAYERS); model._nl=N_LAYERS

    if use_adam:
        opt=torch.optim.AdamW(model.parameters(),lr=LR_ADAM,
                               betas=(0.9,0.95),weight_decay=0.1)
    else:
        opt=torch.optim.SGD(model.parameters(),lr=LR_SGD,
                             momentum=0.9,nesterov=True)

    fisher=FisherPreconditioner(model) if use_ngd else None
    stt=None; vals=[]; t0=time.time()
    print(f"\n  [{name}]")

    for step in range(1,MAX_STEPS+1):
        lr=clr_adam(step) if use_adam else clr_sgd(step)
        for pg in opt.param_groups: pg['lr']=lr

        model.train(); x,y=get_batch(); _,loss=model(x,y)
        opt.zero_grad(); loss.backward()

        if fisher:
            fisher.update(model)
            fisher.precondition(model)

        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step()

        if step%LOG==0 or step==1:
            vl=eval_val(model,n=20); vals.append((step,vl))
            if vl<TARGET and stt is None:
                stt=step; print(f"    *** TARGET at step {step} ***")
            print(f"    {step:>4}/{MAX_STEPS}  val={vl:.4f}  t={time.time()-t0:.0f}s")

    fval=eval_val(model,n=100)
    return stt, vals, time.time()-t0, fval

# ── Run ───────────────────────────────────────────────────────────────────────
print("A: Standard AdamW (diagonal FIM approximation)...")
stt_A,vals_A,t_A,fval_A=run("AdamW", use_adam=True,  use_ngd=False)

print("\nB: SGD + Nesterov (no FIM)...")
stt_B,vals_B,t_B,fval_B=run("SGD+Nesterov", use_adam=False, use_ngd=False)

print("\nC: SGD + K-FAC block Fisher preconditioning (natural gradient)...")
stt_C,vals_C,t_C,fval_C=run("SGD+NGD(KFAC)", use_adam=False, use_ngd=True)

print("\nD: AdamW + K-FAC (full preconditioned)...")
stt_D,vals_D,t_D,fval_D=run("AdamW+NGD(KFAC)", use_adam=True, use_ngd=True)

# ── Results ───────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  RESULTS  (target val<{TARGET})")
print("="*65)

def fmt(s): return str(s) if s else f">{MAX_STEPS}"
rows=[
    ("AdamW (diagonal FIM)",       stt_A,fval_A,t_A),
    ("SGD+Nesterov (no FIM)",      stt_B,fval_B,t_B),
    ("SGD+NGD K-FAC",              stt_C,fval_C,t_C),
    ("AdamW+NGD K-FAC",            stt_D,fval_D,t_D),
]
print(f"\n  {'Method':28}  {'Steps→<4':>9}  {'Final val':>10}  {'Time':>7}")
print("  "+"-"*58)
for name,stt,fval,t in rows:
    print(f"  {name:28}  {fmt(stt):>9}  {fval:>10.4f}  {t:>6.1f}s")

print(f"\n  Loss curves:")
print(f"  {'step':>5}  {'AdamW':>8}  {'SGD':>8}  {'SGD+NGD':>9}  {'Adam+NGD':>10}")
print("  "+"-"*44)
sA={s:v for s,v in vals_A}; sB={s:v for s,v in vals_B}
sC={s:v for s,v in vals_C}; sD={s:v for s,v in vals_D}
for s in sorted(sA):
    print(f"  {s:>5}  {sA.get(s,0):>8.4f}  {sB.get(s,0):>8.4f}"
          f"  {sC.get(s,0):>9.4f}  {sD.get(s,0):>10.4f}")

print(f"""
INTERPRETATION:
  Adam IS already a diagonal Fisher preconditioner.
  If AdamW+NGD (K-FAC) beats AdamW alone:
    The off-diagonal Fisher structure (block K-FAC) adds real signal.
    Natural gradient on MF genuinely accelerates convergence.
    The Kähler manifold picture is operationally correct.
  
  If AdamW+NGD ≈ AdamW:
    Adam's diagonal approximation captures the FIM signal sufficiently.
    The full block structure adds noise but not signal.
    The Kähler geometry describes the endpoint but not the optimal path.
  
  If SGD+NGD beats SGD by the same margin Adam beats SGD:
    The FIM preconditioning is doing the same work as Adam.
    Natural gradient = Adam (already known in the literature).
    No additional advantage from the full block structure.
""")
