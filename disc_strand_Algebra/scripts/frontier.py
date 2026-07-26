"""
IS THE ACTIVE FRONTIER STATIC OR MIGRATING?
Active set A_t = lowest-r coordinates (the ~ fraction that actually flip/move).
 (1) Jaccard overlap J(A_t, A_{t+k}) vs lag k. High=static working set,
     decaying=migrating frontier.
 (2) Does the active set track corpus? corr(coordinate activity, token freq) for
     the embedding rows (only place with direct token correspondence).
 (3) path/displacement L/D of the active set alone vs full.
"""
import time, gc, numpy as np, torch, json
t0=time.time()
g_={}; src=open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
exec(src[:src.find("# ── PHASE 1")], g_)
model=g_["model"]; get_batch=g_["get_batch"]; LR=g_["LR"]
named=list(model.named_parameters())
def flat(): return torch.cat([p.data.flatten() for _,p in named]).clone()
def st(o,k):
    out=[]
    for _,p in named:
        s=o.state.get(p,{}); out.append(s[k].flatten() if k in s else torch.zeros(p.numel()))
    return torch.cat(out)
P=flat().numel()
model.load_state_dict(torch.load("init.pt")); torch.manual_seed(17)
o=torch.optim.AdamW(model.parameters(), lr=LR*5, betas=(0.9,0.95), weight_decay=0.1)
ACT=[]; snaps={}; prev=flat()
FRAC=0.106      # ~460k/4.33M
for s in range(1,161):
    model.train(); x,y=get_batch(); _,l=model(x,y)
    o.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    o.step()
    r=(st(o,"exp_avg").abs()/(st(o,"exp_avg_sq").sqrt()+1e-12))
    thr=torch.quantile(r[torch.randperm(P)[:200000]],FRAC)
    active=(r<thr)                         # low-r = active frontier
    ACT.append(torch.nonzero(active,as_tuple=True)[0].numpy())
    del r
    if s%40==0: gc.collect(); print(f"    step {s} ({time.time()-t0:.0f}s)", flush=True)
def jac(a,b):
    sa,sb=set(a.tolist()),set(b.tolist()); return len(sa&sb)/len(sa|sb)
print("="*72); print("  (1) ACTIVE-FRONTIER OVERLAP  (static vs migrating)"); print("="*72)
print(f"  active fraction {FRAC:.3f} (~{int(FRAC*P/1000)}k coords)")
print(f"  {'lag k':>8}{'Jaccard':>10}")
for k in (1,5,10,20,40,80):
    js=[jac(ACT[t],ACT[t+k]) for t in range(20,len(ACT)-k,3)]
    print(f"  {k:>8}{np.mean(js):>10.3f}")
# baseline: overlap of two random sets of same size
n=len(ACT[40]); rng=np.random.default_rng(0)
jr=np.mean([jac(rng.choice(P,n,False),rng.choice(P,n,False)) for _ in range(5)])
print(f"  random-set baseline: {jr:.3f}")
print("\n  high & flat => static working set.  decaying toward baseline => migrating.")
# (2) core: coords active at EVERY step (persistent core) vs churn
core=set(ACT[20].tolist())
for t in range(21,len(ACT)): core &= set(ACT[t].tolist())
union=set()
for t in range(20,len(ACT)): union|=set(ACT[t].tolist())
print(f"\n  persistent core (active ALL steps 20-160): {len(core):,}")
print(f"  ever-active union: {len(union):,}  ({len(union)/P*100:.1f}% of all params)")
print(f"  core / typical active-set size = {len(core)/n:.3f}")
print(f"  => if core<<active-set, the frontier MIGRATES; a small core stays hot.")
print(f"\n  time {time.time()-t0:.0f}s")
