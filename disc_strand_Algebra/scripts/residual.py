"""
CHARACTERISE eps = u - a*sign(u), then the timescale-resolved trajectory gap.
 (1) temporal autocorr <eps_t, eps_{t+k}>
 (2) corr(eps_t, g_{t+k})  -- does the residual predict future gradients?
 (3) corr(|eps|_tile, r_tile) and corr(|eps|, future flips)
 (4) TRAJECTORY GAP: run a*sign(u) forward from a shared point; measure
     cos(true, recon) and val gap vs horizon k.  Flat=null, opens-then-closes=compensation.
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
def state(o,k):
    out=[]
    for _,p in named:
        s=o.state.get(p,{}); out.append(s[k].flatten() if k in s else torch.zeros(p.numel()))
    return torch.cat(out)
P=flat().numel(); NT=1024; seg=(torch.arange(P)*NT//P).long(); cnt=torch.bincount(seg,minlength=NT).float()
model.load_state_dict(torch.load("init.pt")); torch.manual_seed(17)
o=torch.optim.AdamW(model.parameters(), lr=LR*5, betas=(0.9,0.95), weight_decay=0.1)
EPS=[]; GRAD=[]; RT=[]; SG=[]
prev=flat()
for s in range(1,121):
    model.train(); x,y=get_batch(); _,l=model(x,y)
    o.zero_grad(); l.backward()
    gv=torch.cat([(p.grad.flatten() if p.grad is not None else torch.zeros(p.numel())) for _,p in named]).clone()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    b4=flat(); o.step(); af=flat(); u=af-b4; sg=torch.sign(u)
    num=torch.zeros(NT).index_add_(0,seg,u*sg); den=torch.zeros(NT).index_add_(0,seg,sg*sg)
    eps=u-(num/(den+1e-30))[seg]*sg
    m=state(o,"exp_avg").abs(); v=state(o,"exp_avg_sq").sqrt()
    EPS.append(eps); GRAD.append(gv); SG.append(sg)
    RT.append((torch.zeros(NT).index_add_(0,seg,(m/(v+1e-12)))/cnt))
    prev=af; del b4,af,u
    if s%40==0: gc.collect(); print(f"    step {s} ({time.time()-t0:.0f}s)", flush=True)
def ncos(a,b): return float(a@b/(a.norm()*b.norm()+1e-30))
print("\n"+"="*74); print("  (1) RESIDUAL TEMPORAL AUTOCORRELATION"); print("="*74)
for k in (1,2,5,10):
    c=np.mean([ncos(EPS[t],EPS[t+k]) for t in range(0,len(EPS)-k)])
    print(f"    <eps_t, eps_t+{k}>  cos = {c:+.4f}")
print("  near zero => no persistence (favours noise/compensation over structured)")
print("\n"+"="*74); print("  (2) DOES THE RESIDUAL PREDICT FUTURE GRADIENTS?"); print("="*74)
for k in (0,1,5):
    c=np.mean([ncos(EPS[t],GRAD[t+k]) for t in range(0,len(EPS)-k)])
    print(f"    <eps_t, g_t+{k}>  cos = {c:+.4f}")
print("\n"+"="*74); print("  (3) RESIDUAL vs r AND FUTURE FLIPS (per tile)"); print("="*74)
ce=[]; cf=[]
for t in range(len(EPS)-1):
    et=(torch.zeros(NT).index_add_(0,seg,EPS[t].abs())/cnt).numpy()
    fl=(torch.zeros(NT).index_add_(0,seg,(SG[t+1]!=SG[t]).float())/cnt).numpy()
    ce.append(np.corrcoef(et,RT[t].numpy())[0,1]); cf.append(np.corrcoef(et,fl)[0,1])
print(f"    corr(|eps|_tile, r_tile)          = {np.nanmean(ce):+.3f}")
print(f"    corr(|eps|_tile, next flip frac)  = {np.nanmean(cf):+.3f}")
import sys; sys.exit(0)
print("\n"+"="*74); print("  (4) TRAJECTORY GAP vs HORIZON  (recon a*sign(u) vs true)"); print("="*74)
def traj(recon, K):
    model.load_state_dict(torch.load("init.pt")); torch.manual_seed(17)
    oo=torch.optim.AdamW(model.parameters(), lr=LR*5, betas=(0.9,0.95), weight_decay=0.1)
    prev=flat(); cosk=[]
    for s in range(K):
        model.train(); x,y=get_batch(); _,l=model(x,y)
        oo.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        b4=flat(); oo.step(); af=flat(); u=af-b4
        if recon:
            sg=torch.sign(u); num=torch.zeros(NT).index_add_(0,seg,u*sg)
            den=torch.zeros(NT).index_add_(0,seg,sg*sg); u=(num/(den+1e-30))[seg]*sg
            setflat(b4+u); prev=flat()
        else: prev=af
    return flat(), float(eval_val(model,n=16))
th_true={}; 
base=torch.load("init.pt")
for K in (10,30,60,100):
    tT,vT=traj(False,K); tR,vR=traj(True,K)
    print(f"    K={K:>4}  cos(true,recon)={ncos(tT-flat()*0, tR):.4f}"
          f"   val true {vT:.4f}  recon {vR:.4f}  gap {vR-vT:+.4f}", flush=True)
print("\n  gap flat in K => residual null;  gap grows then shrinks => compensation")
print(f"\n  time {time.time()-t0:.0f}s")
