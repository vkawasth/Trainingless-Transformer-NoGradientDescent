"""
DOES K_r HAVE NONTRIVIAL OPERATOR STRUCTURE, OR DOES IT FACTORIZE?
If the sign process is a product of independent per-coordinate 2-state chains
with flip prob h(r_i), then K_r is trivial (spectrum known, no coupling).
The operator theory needs coupling that SURVIVES conditioning on r.
Test: for coordinate pairs matched on (r_i, r_j), is
   P(flip_i AND flip_j) == P(flip_i) P(flip_j) ?
Excess joint-flip probability over the independent prediction, conditioned on r,
is the only thing that makes K_r more than a product of 2x2 blocks.
"""
import time, gc, numpy as np, torch
t0=time.time()
g_={}; src=open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
exec(src[:src.find("# ── PHASE 1")], g_)
model=g_["model"]; get_batch=g_["get_batch"]; LR=g_["LR"]
named=list(model.named_parameters())
def flat(): return torch.cat([p.data.flatten() for _,p in named]).clone()
def state(o,k):
    out=[]
    for _,p in named:
        s=o.state.get(p,{}); out.append(s[k].flatten() if k in s else torch.zeros(p.numel()))
    return torch.cat(out)
P=flat().numel()
# one weight matrix, so neighbours are meaningful
import re
off=0; SPAN={}
for n,p in named: SPAN[n]=(off,off+p.numel()); off+=p.numel()
a0,_=SPAN["blocks.2.ff.g.weight"]; R,C=512,256
IDX=torch.arange(a0,a0+R*C)
model.load_state_dict(torch.load("init.pt")); torch.manual_seed(17)
o=torch.optim.AdamW(model.parameters(), lr=LR*5, betas=(0.9,0.95), weight_decay=0.1)
prev=flat(); sp=None
FL=[]; RB=[]
NB=8
for s in range(1,121):
    model.train(); x,y=get_batch(); _,l=model(x,y)
    o.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    b4=flat(); o.step(); af=flat(); d=af-b4; sg=torch.sign(d)[IDX]
    m=state(o,"exp_avg").abs()[IDX]; v=state(o,"exp_avg_sq").sqrt()[IDX]; r=m/(v+1e-12)
    if sp is not None:
        FL.append((sg!=sp).view(R,C).numpy().astype(np.int8))
        RB.append(r.view(R,C).numpy())
    sp=sg; prev=af; del b4,af,d,m,v,r
    if s%40==0: gc.collect(); print(f"    step {s} ({time.time()-t0:.0f}s)", flush=True)
FL=np.stack(FL); RB=np.stack(RB)                # (T,R,C)
T=FL.shape[0]
# r-bin every coordinate every step
rflat=RB.reshape(T,-1)
edges=np.quantile(rflat, np.linspace(0,1,NB+1)[1:-1])
rbin=np.digitize(RB, edges)                     # (T,R,C) in 0..NB-1
# marginal flip prob per r-bin
pflip=np.zeros(NB); cnt=np.zeros(NB)
for b in range(NB):
    m=rbin==b; pflip[b]=FL[m].mean(); cnt[b]=m.sum()
print("\n"+"="*76); print("  DOES THE SIGN PROCESS FACTORIZE GIVEN r?"); print("="*76)
print("  test: joint flip prob of ADJACENT coords vs independent prediction,")
print("  conditioned on BOTH coords' r-bin.")
# adjacent pairs along columns (same neuron, neighbouring input)
obs=0.0; ind=0.0; npair=0
exc=[]
for b1 in range(NB):
    for b2 in range(b1,NB):
        # pairs (i,j) horizontally adjacent with rbin i=b1, j=b2
        A=FL[:,:,:-1]; B=FL[:,:,1:]
        ba=rbin[:,:,:-1]; bb=rbin[:,:,1:]
        msk=((ba==b1)&(bb==b2))
        if msk.sum()<2000: continue
        pj=(A[msk]&B[msk]).mean()               # joint flip
        pa=A[msk].mean(); pb=B[msk].mean()
        exc.append((pj-pa*pb, pa*pb, msk.sum()))
exc=np.array(exc)
wexc=np.average(exc[:,0], weights=exc[:,2])
wind=np.average(exc[:,1], weights=exc[:,2])
print(f"\n  adjacent (same neuron):")
print(f"    mean joint-flip excess over independence = {wexc:+.5f}")
print(f"    mean independent prediction              = {wind:.5f}")
print(f"    relative excess                          = {100*wexc/max(wind,1e-9):+.1f}%")
# control: random non-adjacent pairs matched on r-bins
rng=np.random.default_rng(0)
exc2=[]
flatFL=FL.reshape(T,-1); flatB=rbin.reshape(T,-1); N=flatFL.shape[1]
for _ in range(400):
    i,j=rng.integers(0,N,2)
    if abs(i-j)<10: continue
    a=flatFL[:,i]; b=flatFL[:,j]
    pj=(a&b).mean(); exc2.append(pj-a.mean()*b.mean())
print(f"\n  random distant pairs:")
print(f"    mean joint-flip excess = {np.mean(exc2):+.5f}")
print(f"\n  if adjacent excess >> random excess AND >0 after r-conditioning,")
print(f"  K_r has genuine coupling (nontrivial operator). If ~0, K_r factorizes")
print(f"  into independent 2-state chains and the operator theory is trivial.")
print(f"\n  time {time.time()-t0:.0f}s")
