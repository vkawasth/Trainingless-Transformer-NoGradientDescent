"""
THE LAST DOOR: does a CHEAP backward signal preserve the update SIGN?
Feedback alignment (FA): replace the transpose weights in the backward pass with
FIXED RANDOM matrices. delta is then computed with random backward weights ->
no weight-symmetric backprop. Question: sign(update_FA) vs sign(update_true),
and does training on FA-signs hold?
If FA preserves signs -> the wall has a door (backward is substitutable).
If FA breaks signs   -> delta is irreducible even to substitution.
We emulate FA's effect on delta by corrupting the backward pass with a fixed
random linear map per layer (the structured error FA introduces), then compare.
"""
import time, gc, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
t0=time.time()
g_={}; src=open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
exec(src[:src.find("# ── PHASE 1")], g_)
model=g_["model"]; get_batch=g_["get_batch"]; eval_val=g_["eval_val"]; LR=g_["LR"]
named=list(model.named_parameters())
def flat(): return torch.cat([p.data.flatten() for _,p in named]).clone()
def gradv(): return torch.cat([(p.grad.flatten() if p.grad is not None else torch.zeros(p.numel())) for _,p in named]).clone()
def newopt(): return torch.optim.AdamW(model.parameters(), lr=LR*5, betas=(0.9,0.95), weight_decay=0.1)
def V(n=16): return float(eval_val(model,n=n))
# We can't easily rewire autograd here; instead we test the KEY quantity:
# FA replaces J^T (exact) with B (fixed random) in delta propagation. The net
# effect on a weight-gradient is that delta_out is passed through a random
# rotation. Emulate: g_FA = g_true perturbed by a FIXED random per-layer linear
# map acting on the OUTPUT-neuron index (the delta axis), constant across steps.
import re
off=0; SPAN={}
for n,p in named: SPAN[n]=(off,off+p.numel()); off+=p.numel()
MATS={n:tuple(p.shape) for n,p in named if p.dim()==2}
gen=torch.Generator().manual_seed(7)
# fixed random mixing on the output (row) axis of each weight matrix, strength alpha
def make_B(R,alpha):
    Bm=torch.eye(R)+alpha*torch.randn(R,R,generator=gen)/np.sqrt(R)
    return Bm
BMAP={}
def fa_grad(alpha):
    out=flat()*0
    for n,p in named:
        a,b=SPAN[n]
        if p.grad is None: continue
        G=p.grad.detach()
        if G.dim()==2 and alpha>0:
            R=G.shape[0]
            if (n,R) not in BMAP: BMAP[(n,R)]=make_B(R,alpha)
            out[a:b]=(BMAP[(n,R)]@G).flatten()      # random mix on delta (row) axis
        else:
            out[a:b]=G.flatten()
    return out
model.load_state_dict(torch.load("init.pt")); torch.manual_seed(17); o=newopt()
ALPHAS=[0.1,0.3,0.6,1.0]
ag={a:[] for a in ALPHAS}
for s in range(40):
    model.train(); x,y=get_batch(); _,l=model(x,y)
    o.zero_grad(); l.backward(); g=gradv()
    for al in ALPHAS:
        gf=fa_grad(al)
        ag[al].append(float((torch.sign(gf)==torch.sign(g))[g!=0].double().mean()))
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); o.step()
print("="*72); print("  (1) FEEDBACK-ALIGNMENT SIGN AGREEMENT (fixed random delta-mix)"); print("="*72)
for al in ALPHAS: print(f"  FA strength alpha={al}: sign agreement = {100*np.mean(ag[al]):.1f}%")
# (2) train using FA gradient
def run(alpha):
    model.load_state_dict(torch.load("init.pt")); torch.manual_seed(17); o=newopt()
    for s in range(120):
        model.train(); x,y=get_batch(); _,l=model(x,y)
        o.zero_grad(); l.backward()
        if alpha>0:
            gf=fa_grad(alpha); i=0
            with torch.no_grad():
                for _,p in named:
                    k=p.numel()
                    if p.grad is not None: p.grad.copy_(gf[i:i+k].view_as(p))
                    i+=k
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); o.step()
    return V()
vb=run(0.0)
print("\n"+"="*72); print("  (2) TRAINING ON THE FA GRADIENT"); print("="*72)
print(f"  {'exact backward':>22}: val {vb:.4f}")
import sys
for al in [float(a) for a in (sys.argv[1:] or ["0.3"])]:
    v=run(al); print(f"  {'FA alpha='+str(al):>22}: val {v:.4f}  ({v/vb:.2f}x)", flush=True)
print("\n  FA is genuinely cheaper only if the random B removes real compute;")
print("  here it tests whether a NON-weight-symmetric delta keeps the sign.")
print("  high agreement + good val => backward signal is substitutable (door open).")
print("  broken signs => delta is irreducible even to substitution (door shut).")
print(f"\n  time {time.time()-t0:.0f}s")
