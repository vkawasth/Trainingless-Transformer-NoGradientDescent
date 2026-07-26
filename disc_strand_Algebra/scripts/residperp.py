"""
Is the residual eps = u - a*sign(u) PERPENDICULAR to the gradient?
If eps_par/eps is tiny, the 58% Fisher energy has ~0 first-order effect: <g,eps>~0.
Measure in Euclidean and in the diagonal-Fisher metric G^1/2.
Also: eps energy split by r-decile, and <g,u> vs <g, a*sign(u)> (descent retained).
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
import torch.nn.functional as Fn
P=flat().numel(); NT=1024; seg=(torch.arange(P)*NT//P).long(); cnt=torch.bincount(seg,minlength=NT).float()
model.load_state_dict(torch.load("init.pt")); torch.manual_seed(17)
o=torch.optim.AdamW(model.parameters(), lr=LR*5, betas=(0.9,0.95), weight_decay=0.1)
for s in range(100):
    model.train(); x,y=get_batch(); _,l=model(x,y)
    o.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); o.step()
G=torch.zeros(P)
for _ in range(8):
    x,_=get_batch(); lo,_=model(x); lp=Fn.log_softmax(lo.reshape(-1,lo.shape[-1]),1)
    ys=torch.multinomial(lp.exp(),1).squeeze(1); model.zero_grad(set_to_none=True); Fn.nll_loss(lp,ys).backward()
    G+=torch.cat([(p.grad.flatten() if p.grad is not None else torch.zeros(p.numel())) for _,p in named])**2
G/=8; Gh=G.sqrt()
prev=flat()
PAR=[];PARF=[];GE=[];GU=[];GS=[]; edec=np.zeros(10); rdec=np.zeros(10)
for s in range(40):
    model.train(); x,y=get_batch(); _,l=model(x,y)
    o.zero_grad(); l.backward()
    gv=torch.cat([(p.grad.flatten() if p.grad is not None else torch.zeros(p.numel())) for _,p in named]).clone()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    b4=flat(); o.step(); af=flat(); u=af-b4; sg=torch.sign(u)
    num=torch.zeros(NT).index_add_(0,seg,u*sg); den=torch.zeros(NT).index_add_(0,seg,sg*sg)
    rec=(num/(den+1e-30))[seg]*sg; eps=u-rec
    # parallel fraction wrt gradient
    par=float((eps@gv)**2/((gv@gv)*(eps@eps)+1e-30))       # cos^2(eps,g)
    ge=Gh*eps; gg=Gh*gv
    parf=float((ge@gg)**2/((gg@gg)*(ge@ge)+1e-30))
    PAR.append(par); PARF.append(parf)
    GE.append(float(gv@eps)); GU.append(float(gv@u)); GS.append(float(gv@rec))
    r=(state(o,"exp_avg").abs()/(state(o,"exp_avg_sq").sqrt()+1e-12))
    q=torch.quantile(r[torch.randperm(P)[:100000]],torch.linspace(0,1,11)[1:-1].double().float())
    b=torch.bucketize(r,q)
    for i in range(10):
        m=(b==i); edec[i]+=float((eps[m]**2).sum()); rdec[i]+=float((u[m]**2).sum())
    prev=af; del b4,af,u,eps,gv
    if s%20==19: gc.collect()
print("="*72); print("  IS THE RESIDUAL PERPENDICULAR TO THE GRADIENT?"); print("="*72)
print(f"  cos^2(eps, g)  Euclidean = {np.mean(PAR):.4f}   => {100*np.mean(PAR):.2f}% of eps energy is parallel")
print(f"  cos^2(eps, g)  Fisher    = {np.mean(PARF):.4f}")
print(f"\n  <g,eps> / <g,u>          = {np.mean(GE)/np.mean(GU):+.4f}")
print(f"  <g,u>      (full update) = {np.mean(GU):.5f}")
print(f"  <g,a*sign> (reconstruct) = {np.mean(GS):.5f}   retained {100*np.mean(GS)/np.mean(GU):.1f}%")
print(f"\n  => if <g,eps> ~ 0, the discarded 58% Fisher energy has ~no first-order")
print(f"     effect on the loss: the residual is (first-order) trajectory-null.")
print("\n  residual energy by r-decile (low r = active, high r = frozen):")
et=edec/edec.sum()
print("   decile:   " + " ".join(f"{i+1:>5}" for i in range(10)))
print("   |eps|^2%: " + " ".join(f"{100*et[i]:>5.1f}" for i in range(10)))
print(f"\n  time {time.time()-t0:.0f}s")
