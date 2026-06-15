#!/usr/bin/env python3
"""
Kac-Moody Sweep: Find the True Serre Exponent
===============================================
Extend the Serre check to k=2..8.
If residual → 0 at finite k: finite-dimensional algebra, k identifies the type.
If residual decays but never reaches 0: Kac-Moody (affine/hyperbolic).

The Serre exponent k determines the approximator architecture:
  k=2 → A2 (sl3): 2-layer student suffices, 8-dim fundamental rep
  k=3 → B2/C2: 3-layer minimum, need symplectic structure
  k=4 → G2/F4: 4-layer minimum, exceptional structure
  k=∞ → Kac-Moody: depth = number of Serre levels approximated

APPROXIMATOR DESIGN:
Once k is known, build a k-layer student where:
  Layer l implements ad(e)^l — the l-th level of the Serre cascade
  Each layer is initialized from the teacher's Jacobian at the
  corresponding cascade level, not from random weights.
"""
import json, math, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_LAYERS=24; BATCH=8; SEQ=64; LR=3e-4; PROJ=48

print(f"\n{'='*65}")
print(f"  KAC-MOODY SWEEP: SERRE EXPONENT k=2..8")
print(f"  Finding the algebra type for the approximator")
print(f"{'='*65}\n")

with open('/tmp/train_ids.json') as f: train_ids=list(json.load(f))
with open('/tmp/val_ids.json')   as f: val_ids=list(json.load(f))
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

def clr(s,total=300,warmup=100):
    if s<=warmup: return LR*s/warmup
    return LR*0.5*(1+math.cos(math.pi*(s-warmup)/(total-warmup)))

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
    return J.T,U.detach().numpy(),m

def comm(A,B): return A@B-B@A
def ad_k(A,B,k):
    r=B
    for _ in range(k): r=comm(A,r)
    return r
def serre_res(e1,e2,k):
    c=comm(e1,e2); n=float(np.linalg.norm(c))
    if n<1e-10: return float('nan')
    return float(np.linalg.norm(ad_k(e1,e2,k))/n)

# ── Train ─────────────────────────────────────────────────────────────────────
print("Training (seed=42, 300 steps)...")
torch.manual_seed(42)
model=LM(D,N_HEADS,N_LAYERS)
opt=torch.optim.AdamW(model.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
t0=time.time()
for step in range(1,301):
    for pg in opt.param_groups: pg['lr']=clr(step)
    model.train(); x,y=get_batch(); _,loss=model(x,y)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    if step%100==0:
        model.eval()
        with torch.no_grad():
            vl=float(np.mean([model(*get_batch('val'))[1].item() for _ in range(10)]))
        print(f"  step {step}  val={vl:.4f}  t={time.time()-t0:.0f}s")
        model.train()
model.eval()
print()

# ── Extract Jacobians (5 refs) ────────────────────────────────────────────────
print("Extracting Jacobians...",flush=True)
torch.manual_seed(0); pos=SEQ//2; m=min(PROJ,SEQ,D)
all_Js=[]
for _ in range(5):
    x_ref,_=get_batch('val'); x_ref=x_ref[0:1]
    with torch.no_grad():
        hs=model.hidden_states(x_ref); hs=[h[0] for h in hs]
    Js=[]
    for l in range(N_LAYERS):
        J,U,m_=layer_jac(model.blocks[l],hs[l],pos,m)
        Js.append(J)
    all_Js.append(Js)
Js_mean=[np.mean([all_Js[r][l] for r in range(5)],axis=0) for l in range(N_LAYERS)]
print(f"  Done. m={m_}\n")

# ── Random baseline ───────────────────────────────────────────────────────────
np.random.seed(42)
rand_baselines={}
for k in range(2,9):
    vals=[]
    for _ in range(20):
        R1=np.random.randn(m_,m_)*0.3
        R2=np.random.randn(m_,m_)*0.3
        vals.append(serre_res(R1,R2,k))
    rand_baselines[k]=float(np.nanmean(vals))

print("Random baselines:")
for k,v in rand_baselines.items():
    print(f"  k={k}: {v:.4f}")
print()

# ── Sweep k=2..8 across all layer pairs ──────────────────────────────────────
print(f"{'='*65}")
print(f"  SERRE EXPONENT SWEEP k=2..8")
print("="*65)

K_MAX=8
results={}  # k → list of residuals

for k in range(2,K_MAX+1):
    residuals=[]
    for l in range(1,N_LAYERS-1):
        e1=Js_mean[l]; e2=Js_mean[l+1]
        r=serre_res(e1,e2,k)
        if not np.isnan(r): residuals.append(r)
    results[k]=residuals

# Print summary table
print(f"\n  {'k':>4}  {'mean':>8}  {'min':>8}  {'max':>8}  "
      f"{'%random':>9}  {'algebra?':>15}  note")
print("  "+"-"*70)

# Known algebra types by Serre exponent
algebra_names={
    2:"A_n (sl_{n+1})",
    3:"B_n/C_n",
    4:"G2 / F4",
    5:"E6",
    6:"E7",
    7:"E8",
    8:"affine/hyp KM"
}

prev_mean=None
for k in range(2,K_MAX+1):
    res=results[k]
    mean_r=float(np.nanmean(res))
    min_r=float(np.nanmin(res))
    max_r=float(np.nanmax(res))
    pct=mean_r/rand_baselines[k]*100
    alg=algebra_names.get(k,"?")
    # Convergence rate
    rate=f"↓{(prev_mean-mean_r)/prev_mean*100:.0f}%" if prev_mean else ""
    print(f"  k={k}  {mean_r:>8.4f}  {min_r:>8.4f}  {max_r:>8.4f}  "
          f"{pct:>8.1f}%  {alg:>15}  {rate}")
    prev_mean=mean_r

# ── Find the Serre exponent ───────────────────────────────────────────────────
print(f"\n  Log-residual decay (checking if exponential or hits zero):")
ks=list(range(2,K_MAX+1))
means=[float(np.nanmean(results[k])) for k in ks]
log_means=[np.log(m) for m in means]

# Fit linear to log (exponential decay)
coeffs=np.polyfit(ks,log_means,1)
decay_rate=coeffs[0]  # slope of log-residual vs k
r_sq=np.corrcoef(ks,log_means)[0,1]**2

print(f"  Log-linear fit: log(residual) = {coeffs[0]:.3f}*k + {coeffs[1]:.3f}")
print(f"  Decay rate per k: {decay_rate:.3f} (more negative = faster decay)")
print(f"  R² of log-linear fit: {r_sq:.4f}")
print(f"  Extrapolated k for residual<0.01: "
      f"k={int((np.log(0.01)-coeffs[1])/coeffs[0])}")

# Check if any k hits near-zero
near_zero_k=None
for k in ks:
    if float(np.nanmean(results[k]))<0.05:
        near_zero_k=k; break

# ── Per-layer breakdown at key k values ──────────────────────────────────────
print(f"\n  Per-layer residuals at k=4 (G2 candidate) and k=6 (E7):")
print(f"  {'L→L+1':>8}  {'k=4':>8}  {'k=5':>8}  {'k=6':>8}  {'k=7':>8}")
print("  "+"-"*42)
for l in range(1,N_LAYERS-1,3):
    e1=Js_mean[l]; e2=Js_mean[l+1]
    row=[serre_res(e1,e2,k) for k in [4,5,6,7]]
    att="←L14" if l==14 else ""
    print(f"  L{l:>2}→L{l+1:<2}  "+"  ".join(f"{r:>8.4f}" for r in row)+f"  {att}")

# ── Decision: what is the algebra? ───────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  DECISION: ALGEBRA TYPE AND APPROXIMATOR DEPTH")
print("="*65)

final_k4=float(np.nanmean(results[4]))
final_k5=float(np.nanmean(results[5])) if 5 in results else None
final_k6=float(np.nanmean(results[6])) if 6 in results else None

print(f"""
  Serre residual decay:
    k=2: {float(np.nanmean(results[2])):.4f}
    k=3: {float(np.nanmean(results[3])):.4f}
    k=4: {float(np.nanmean(results[4])):.4f}
    k=5: {float(np.nanmean(results[5])):.4f}
    k=6: {float(np.nanmean(results[6])):.4f}
    k=7: {float(np.nanmean(results[7])):.4f}
    k=8: {float(np.nanmean(results[8])):.4f}

  Log-linear decay rate: {decay_rate:.3f} per unit k
  R² = {r_sq:.4f} {'(clean exponential decay)' if r_sq>0.95 else '(not clean exponential)'}
""")

if near_zero_k:
    print(f"  FINITE-DIMENSIONAL: residual hits <0.05 at k={near_zero_k}")
    print(f"  Algebra type: {algebra_names.get(near_zero_k,'?')}")
    print(f"  APPROXIMATOR DEPTH: {near_zero_k} layers minimum")
elif r_sq > 0.95 and decay_rate < -0.3:
    extr_k=int((np.log(0.01)-coeffs[1])/coeffs[0])
    print(f"  KAC-MOODY: clean exponential decay, never hits zero")
    print(f"  Extrapolated zero at k≈{extr_k} (beyond finite algebra)")
    print(f"  APPROXIMATOR DEPTH: determined by desired precision")
    print(f"  At k=4: {final_k4:.3f} residual = "
          f"{(1-final_k4/rand_baselines[4])*100:.0f}% structure captured")
    print(f"  At k=6: {final_k6:.3f} residual = "
          f"{(1-final_k6/rand_baselines[6])*100:.0f}% structure captured")
else:
    print(f"  AMBIGUOUS: decay not clean exponential")
    print(f"  Need more data to classify")

print(f"""
  APPROXIMATOR ARCHITECTURE IMPLICATION:

  The approximator needs k layers where layer l implements
  the l-th level of the Serre cascade ad(e)^l.

  Student layer l is initialized from:
    W_l = projection of ad(J_{{att}})^l onto the active subspace
  where J_att is the teacher Jacobian at the attractor (L14).

  This is the correct closed-form initialization —
  not from random, not from weight copying,
  but from the Serre cascade of the attractor Jacobian.
""")
