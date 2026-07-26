"""
DOES A LOW-RANK / SKETCHED BACKWARD PASS RECOVER THE UPDATE SIGN?
Structured (correlated) error, unlike i.i.d. noise. Two cheap-backward proxies:
 (A) rank-k sketch of the gradient: g_hat = U U^T g, U random orthonormal kxP.
     (emulates propagating a k-dim sketch instead of full P-dim signal.)
 (B) top-k SVD truncation of g per tensor (structured low-rank error).
Measure sign agreement vs full g, and end-to-end val training on sign(g_hat).
The ii.d.-noise test already passed; this is the STRUCTURED-error test that
decides whether a genuinely cheaper backward algorithm keeps the signs.
"""
import time, gc, numpy as np, torch
t0=time.time()
g_={}; src=open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
exec(src[:src.find("# ── PHASE 1")], g_)
model=g_["model"]; get_batch=g_["get_batch"]; eval_val=g_["eval_val"]; LR=g_["LR"]
named=list(model.named_parameters())
def flat(): return torch.cat([p.data.flatten() for _,p in named]).clone()
def gradv(): return torch.cat([(p.grad.flatten() if p.grad is not None else torch.zeros(p.numel())) for _,p in named]).clone()
def newopt(): return torch.optim.AdamW(model.parameters(), lr=LR*5, betas=(0.9,0.95), weight_decay=0.1)
def V(n=16): return float(eval_val(model,n=n))
P=flat().numel()
# (A) sign agreement of a rank-k projection of the flattened gradient
# use a structured sketch: block-diagonal random projection, retaining fraction rho of a random subspace
model.load_state_dict(torch.load("init.pt")); torch.manual_seed(17); o=newopt()
RHOS=[0.5,0.25,0.1,0.05]
gen=torch.Generator().manual_seed(1)
agree={r:[] for r in RHOS}
for s in range(40):
    model.train(); x,y=get_batch(); _,l=model(x,y)
    o.zero_grad(); l.backward(); g=gradv()
    for rho in RHOS:
        k=max(1,int(P*rho))
        # random coordinate sketch: keep k random coords exact, low-rank-fill the rest
        # emulate rank-k linear sketch g_hat = S^T (S g) with S k random rows (Gaussian)
        idx=torch.randperm(P,generator=gen)[:k]
        # cheap reconstruction: project g onto span of k random sparse vectors ~ subsample+scale
        gh=torch.zeros_like(g); gh[idx]=g[idx]        # rank-k coordinate sketch (structured, sparse)
        agree[rho].append(float((torch.sign(gh)==torch.sign(g))[g!=0].double().mean()))
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); o.step()
print("="*72); print("  (A) COORDINATE-SKETCH SIGN AGREEMENT (structured, sparse error)"); print("="*72)
for rho in RHOS: print(f"  keep {int(100*rho):>3}% coords exact: sign agreement over nonzero = {100*np.mean(agree[rho]):.1f}%")
print("  (this is trivially rho on random coords; included as the structured-sparse baseline)")
# (B) the meaningful one: true low-rank truncation of each weight-gradient matrix
import re
off=0; SPAN={}
for n,p in named: SPAN[n]=(off,off+p.numel()); off+=p.numel()
MATS=[(n,tuple(p.shape)) for n,p in named if p.dim()==2]
def lowrank_grad(rank_frac):
    """replace each 2-D grad matrix by its top-r SVD; return flattened full-P vector"""
    out=flat()*0
    for n,p in named:
        a,b=SPAN[n]
        if p.grad is None: continue
        G=p.grad.detach()
        if G.dim()==2:
            r=max(1,int(min(G.shape)*rank_frac))
            U,S,Vt=torch.linalg.svd(G,full_matrices=False)
            Gr=(U[:,:r]*S[:r])@Vt[:r]
            out[a:b]=Gr.flatten()
        else:
            out[a:b]=G.flatten()
    return out
print("\n"+"="*72); print("  (B) TRUE LOW-RANK GRADIENT: sign agreement & end-to-end val"); print("="*72)
model.load_state_dict(torch.load("init.pt")); torch.manual_seed(17); o=newopt()
ag={rf:[] for rf in (0.5,0.25,0.1)}
for s in range(30):
    model.train(); x,y=get_batch(); _,l=model(x,y)
    o.zero_grad(); l.backward(); g=gradv()
    for rf in ag:
        gh=lowrank_grad(rf)
        ag[rf].append(float((torch.sign(gh)==torch.sign(g))[g!=0].double().mean()))
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); o.step()
for rf in ag: print(f"  rank {int(100*rf)}% of each matrix: sign agreement = {100*np.mean(ag[rf]):.1f}%")
def run(rf):
    model.load_state_dict(torch.load("init.pt")); torch.manual_seed(17); o=newopt()
    for s in range(120):
        model.train(); x,y=get_batch(); _,l=model(x,y)
        o.zero_grad(); l.backward()
        if rf<1.0:
            gh=lowrank_grad(rf); i=0
            with torch.no_grad():
                for _,p in named:
                    k=p.numel()
                    if p.grad is not None: p.grad.copy_(gh[i:i+k].view_as(p))
                    i+=k
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); o.step()
    return V()
vb=run(1.0)
print(f"\n  end-to-end training on the low-rank gradient:")
print(f"  {'full (baseline)':>18}: val {vb:.4f}")
import sys
for rf in [float(a) for a in (sys.argv[1:] or ["0.25"])]:
    v=run(rf); print(f"  rank {int(100*rf)}%: val {v:.4f}  ({v/vb:.2f}x)  [full {vb:.4f}]", flush=True)
print("\n  if signs survive TRUE low-rank truncation and training holds, a cheaper")
print("  (low-rank) backward pass is viable. if signs break, backprop is irreducible.")
print(f"\n  time {time.time()-t0:.0f}s")
