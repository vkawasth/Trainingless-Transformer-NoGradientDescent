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
P=flat().numel(); NT=1024; seg=(torch.arange(P)*NT//P).long(); cnt=torch.bincount(seg,minlength=NT).float()
SUB=torch.randperm(P)[:200000]                 # fixed coordinate subset for cheap cosines
model.load_state_dict(torch.load("init.pt")); torch.manual_seed(17)
o=torch.optim.AdamW(model.parameters(), lr=LR*5, betas=(0.9,0.95), weight_decay=0.1)
Es=[]; Gs=[]; Ss=[]; ce=[]; cf=[]      # Es/Gs/Ss are SUBSET vectors only
prev=flat()
for s in range(1,121):
    model.train(); x,y=get_batch(); _,l=model(x,y)
    o.zero_grad(); l.backward()
    gv=torch.cat([(p.grad.flatten() if p.grad is not None else torch.zeros(p.numel())) for _,p in named])
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    b4=flat(); o.step(); af=flat(); u=af-b4; sg=torch.sign(u)
    num=torch.zeros(NT).index_add_(0,seg,u*sg); den=torch.zeros(NT).index_add_(0,seg,sg*sg)
    eps=u-(num/(den+1e-30))[seg]*sg
    m=state(o,"exp_avg").abs(); v=state(o,"exp_avg_sq").sqrt(); r=m/(v+1e-12)
    rt=(torch.zeros(NT).index_add_(0,seg,r)/cnt).numpy()
    et=(torch.zeros(NT).index_add_(0,seg,eps.abs())/cnt).numpy()
    ce.append(np.corrcoef(et,rt)[0,1])
    Es.append(eps[SUB].clone()); Gs.append(gv[SUB].clone()); Ss.append(sg.clone())
    if len(Ss)>2: Ss.pop(0)                     # keep only last 2 full sign vectors
    # future-flip corr, tile level
    if len(Es)>1:
        flk=(torch.zeros(NT).index_add_(0,seg,(sg!=prev_sg).float())/cnt).numpy()
        cf.append(np.corrcoef(prev_et,flk)[0,1])
    prev_sg=sg.clone(); prev_et=et; prev=af
    del b4,af,u,eps,m,v,r,gv
    if len(Es)>12: Es.pop(0); Gs.pop(0)          # rolling buffer for autocorr
    if s%40==0: gc.collect(); print(f"    step {s} ({time.time()-t0:.0f}s)", flush=True)
def nc(a,b): return float(a@b/(a.norm()*b.norm()+1e-30))
print("\n"+"="*70); print("  RESIDUAL CHARACTER"); print("="*70)
print("  (1) temporal autocorrelation (subset):")
for k in (1,2,5,10):
    v=[nc(Es[t],Es[t+k]) for t in range(len(Es)-k)]
    print(f"      <eps_t,eps_t+{k}> = {np.mean(v):+.4f}")
print("  (2) residual vs future gradient:")
for k in (0,1,5):
    v=[nc(Es[t],Gs[t+k]) for t in range(len(Es)-k)]
    print(f"      <eps_t,g_t+{k}> = {np.mean(v):+.4f}")
print("  (3) tile-level:")
print(f"      corr(|eps|,r)          = {np.nanmean(ce):+.3f}")
print(f"      corr(|eps|,next flips) = {np.nanmean(cf):+.3f}")
print(f"\n  time {time.time()-t0:.0f}s")
