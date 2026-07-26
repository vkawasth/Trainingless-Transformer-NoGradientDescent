"""
STALENESS-FIELD LAW + SYNCHRONIZATION-DOMAIN TEST.
 (1) Law: does coupling-weighted age-mismatch  sum_ij C_ij |a_i - a_j|  predict
     one-step loss degradation? Compare uniform vs adaptive staleness patterns.
 (2) Escape hatch: does the neuron coupling matrix C have low-dimensional block
     (synchronization) structure that a scheduler could exploit? Participation
     ratio of its spectrum + best 2-block cut vs random cut.
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
def stt(o,k):
    out=[]
    for _,p in named:
        s=o.state.get(p,{}); out.append(s[k].flatten() if k in s else torch.zeros(p.numel()))
    return torch.cat(out)
def newopt(): return torch.optim.AdamW(model.parameters(), lr=LR*5, betas=(0.9,0.95), weight_decay=0.1)
def L():
    model.eval()
    with torch.no_grad():
        tot=0
        for _ in range(8): x,y=get_batch(); _,l=model(x,y); tot+=float(l)
    model.train(); return tot/8
off=0; SPAN={}
for n,p in named: SPAN[n]=(off,off+p.numel()); off+=p.numel()
a0,_=SPAN["blocks.2.ff.g.weight"]; R,C=512,256; IDX=torch.arange(a0,a0+R*C)
# build neuron coupling C from r-conditioned flip correlation (as before)
model.load_state_dict(torch.load("init.pt")); torch.manual_seed(17); o=newopt()
NF=[]; RB=[]; prev=flat(); sp=None
for s in range(80):
    model.train(); x,y=get_batch(); _,l=model(x,y)
    o.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    b4=flat(); o.step(); af=flat(); d=af-b4
    r=(stt(o,"exp_avg").abs()/(stt(o,"exp_avg_sq").sqrt()+1e-12))
    sg=torch.sign(d[IDX])
    if sp is not None:
        NF.append((sg!=sp).view(R,C).float().mean(1).numpy()); RB.append(r[IDX].view(R,C).mean(1).numpy())
    sp=sg; prev=af; del d,r
NF=np.stack(NF); RB=np.stack(RB)
res=np.zeros_like(NF)
for j in range(R):
    b=np.polyfit(RB[:,j],NF[:,j],1); res[:,j]=NF[:,j]-np.polyval(b,RB[:,j])
Cc=np.abs(np.corrcoef(res.T)); np.fill_diagonal(Cc,0); Cc=np.nan_to_num(Cc)
print("="*68); print("  (2) SYNCHRONIZATION-DOMAIN STRUCTURE OF COUPLING C"); print("="*68)
ev=np.linalg.eigvalsh(Cc); ev=np.sort(np.abs(ev))[::-1]
pr=(ev.sum()**2)/(ev**2).sum()          # participation ratio = effective # modes
print(f"  coupling spectrum participation ratio: {pr:.0f} of {R} neurons")
print(f"  (few => low-dim synchronization structure; ~R => diffuse, no domains)")
# best 2-block: Fiedler cut vs random; measure within/between coupling ratio
W=Cc; Dg=np.diag(W.sum(1)); Lap=Dg-W
evals,evecs=np.linalg.eigh(Lap); fied=evecs[:,1]; grp=fied>np.median(fied)
def wb_ratio(g):
    within=W[np.ix_(g,g)].sum()+W[np.ix_(~g,~g)].sum(); between=2*W[np.ix_(g,~g)].sum()
    return within/max(between,1e-9)
rng=np.random.default_rng(0); rg=rng.random(R)>0.5
print(f"  within/between coupling ratio  Fiedler cut: {wb_ratio(grp):.2f}   random cut: {wb_ratio(rg):.2f}")
print(f"  (Fiedler >> random => separable domains exist; Fiedler ~ random => none)")
# (3) the law: age-mismatch predicts degradation
print("\n"+"="*68); print("  (1) DOES COUPLING-WEIGHTED AGE-MISMATCH PREDICT DEGRADATION?"); print("="*68)
def deg_for(pattern):
    """apply one stale step with a given per-neuron age pattern; return (mismatch, dL)"""
    model.load_state_dict(torch.load("init.pt")); torch.manual_seed(17); oo=newopt()
    for _ in range(40):
        x,y=get_batch(); _,l=model(x,y); oo.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); oo.step()
    L0=L()
    x,y=get_batch(); _,l=model(x,y); oo.zero_grad(); l.backward()
    b4=flat(); oo.step(); af=flat(); u=af-b4
    # age pattern over the R neurons of this matrix: 0=fresh, 1=stale(persist)
    ages=pattern.astype(float)
    mism=float((Cc*np.abs(ages[:,None]-ages[None,:])).sum())
    stale=(ages[:,None]*np.ones((1,C))).reshape(-1)
    uW=u[IDX].clone(); s_old=torch.sign(uW)   # (using its own sign as 'stale' proxy)
    keep=torch.tensor(1-stale,dtype=torch.bool)
    uW2=torch.where(keep,uW, torch.sign(uW)*float(uW.abs().mean()))
    setflat(torch.cat([af[:a0], (b4[IDX]+uW2), af[a0+R*C:]]) if False else af)  # apply full; measure counterfactual
    # counterfactual loss: set weights with the stale pattern
    full=b4.clone(); full[IDX]=b4[IDX]+uW2
    setflat(full); L1=L()
    return mism, L1-L0
pats={"uniform fresh":np.zeros(R),"uniform stale":np.ones(R),
      "random half":(rng.random(R)>0.5).astype(float),
      "r-adaptive":(RB.mean(0)>np.median(RB.mean(0))).astype(float),
      "Fiedler-aligned":grp.astype(float)}
print(f"  {'pattern':>18}{'age-mismatch':>15}{'dL':>10}")
MM=[];DL=[]
for nm,pt in pats.items():
    mm,dl=deg_for(pt); MM.append(mm); DL.append(dl)
    print(f"  {nm:>18}{mm:>15.1f}{dl:>10.4f}")
print(f"\n  corr(age-mismatch, dL) across patterns = {np.corrcoef(MM,DL)[0,1]:+.3f}")
print(f"  positive => the law holds: coupling-weighted age-mismatch drives damage.")
print(f"\n  time {time.time()-t0:.0f}s")
