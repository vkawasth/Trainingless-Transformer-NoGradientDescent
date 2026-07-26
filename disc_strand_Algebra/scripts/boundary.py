"""
CAN STALE r PREDICT THE NEXT BACKWARD BOUNDARY?
r_t is free (optimizer state), built from PAST gradients. To skip backprop on
frozen coordinates we need: r_{t-1} predicts which coords flip at t, well enough
that freezing the high-r set loses nothing.
 (1) AUC of r_{t-1} (stale) vs r_t (fresh) for predicting flip_t.
 (2) The actual algorithm: each step, compute the TRUE update only on the
     low-r fraction f (by stale r); freeze the sign of the rest (reuse). Measure
     val vs f, against the compute it saves. This is 'backprop only on boundary'
     made concrete (gradient still full here, but it bounds the achievable).
 (3) leakage: of coords that flip at t, what fraction were in the high-r (frozen)
     set by stale r?  That is the error the boundary tracker would make.
"""
import time, gc, numpy as np, torch
t0=time.time()
g_={}; src=open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
exec(src[:src.find("# ── PHASE 1")], g_)
model=g_["model"]; get_batch=g_["get_batch"]; eval_val=g_["eval_val"]; LR=g_["LR"]
named=list(model.named_parameters())
def flat(): return torch.cat([p.data.flatten() for _,p in named]).clone()
def setflat(v):
    i=0
    with torch.no_grad():
        for _,p in named:
            k=p.numel(); p.data.copy_(v[i:i+k].view_as(p)); i+=k
def st(o,k):
    out=[]
    for _,p in named:
        s=o.state.get(p,{}); out.append(s[k].flatten() if k in s else torch.zeros(p.numel()))
    return torch.cat(out)
def newopt(): return torch.optim.AdamW(model.parameters(), lr=LR*5, betas=(0.9,0.95), weight_decay=0.1)
def V(n=16): return float(eval_val(model,n=n))
P=flat().numel()
# ---- (1) stale vs fresh r as flip predictor ----
model.load_state_dict(torch.load("init.pt")); torch.manual_seed(17); o=newopt()
prev=flat(); r_prev=None; sp=None; auc_stale=[]; auc_fresh=[]; leak=[]
SUB=torch.randperm(P)[:300000]
for s in range(1,121):
    model.train(); x,y=get_batch(); _,l=model(x,y)
    o.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    b4=flat(); o.step(); af=flat(); d=af-b4; sg=torch.sign(d)
    r_fresh=(st(o,"exp_avg").abs()/(st(o,"exp_avg_sq").sqrt()+1e-12))
    if sp is not None and r_prev is not None and s>10:
        fl=(sg!=sp)[SUB].double()
        def auc(score):
            sc=(-score[SUB]).numpy(); lab=fl.numpy(); n1=lab.sum(); n0=len(lab)-n1
            if n1==0 or n0==0: return np.nan
            rk=np.argsort(np.argsort(sc))+1
            return (rk[lab>0].sum()-n1*(n1+1)/2)/(n1*n0)
        auc_stale.append(auc(r_prev)); auc_fresh.append(auc(r_fresh))
        # leakage at freeze-fraction 0.7 (freeze top-70% r by STALE r)
        thr=torch.quantile(r_prev[SUB],0.30)     # keep lowest 30% active
        frozen=(r_prev>=thr)
        flipped=(sg!=sp)
        leak.append(float((flipped & frozen).sum())/max(float(flipped.sum()),1))
    sp=sg; r_prev=r_fresh; prev=af; del b4,af,d
    if s%40==0: gc.collect(); print(f"    step {s} ({time.time()-t0:.0f}s)", flush=True)
print("="*74); print("  (1) DOES STALE r PREDICT THE NEXT FLIP?"); print("="*74)
print(f"  AUC, stale r_(t-1) -> flip_t : {np.nanmean(auc_stale):.3f}")
print(f"  AUC, fresh r_t     -> flip_t : {np.nanmean(auc_fresh):.3f}")
print(f"  leakage @ freeze 70%: flips landing in the frozen set = {100*np.nanmean(leak):.1f}%")
# ---- (2) the actual boundary algorithm ----
print("\n"+"="*74); print("  (2) BACKPROP-ON-BOUNDARY: update low-r active set, freeze rest"); print("="*74)
def run(active_frac):
    model.load_state_dict(torch.load("init.pt")); torch.manual_seed(17); o=newopt()
    prev=flat(); r_prev=None
    for s in range(120):
        model.train(); x,y=get_batch(); _,l=model(x,y)
        o.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        b4=flat(); o.step(); af=flat(); d=af-b4
        if r_prev is not None and active_frac<1.0:
            thr=torch.quantile(r_prev[torch.randperm(P)[:200000]], 1-active_frac)
            active=(r_prev<thr).float()          # low-r coords move; high-r frozen
            setflat(b4 + d*active)
        r_prev=(st(o,"exp_avg").abs()/(st(o,"exp_avg_sq").sqrt()+1e-12))
        prev=flat()
    return V()
vb=run(1.0)
print(f"  {'active fraction':>16}{'val':>10}{'vs full':>10}")
print(f"  {'1.00 (baseline)':>16}{vb:>10.4f}{'1.00x':>10}")
for f in (0.5,0.3,0.1):
    v=run(f); print(f"  {f:>16.2f}{v:>10.4f}{v/vb:>9.2f}x", flush=True)
print("\n  if val holds at small active fraction, the boundary IS sparse and stale-r")
print("  finds it. if val degrades, determining the boundary needs the full gradient.")
print(f"\n  time {time.time()-t0:.0f}s")
