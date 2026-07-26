"""
FLIP HAZARD FIELD.
 (1) fit h(r), r=|m|/sqrt(v), on fine quantile bins; compare exp vs logistic
 (2) tile features (mean r, var r, frac below tau, delta mean r) -> predict
     tile flip activity k steps AHEAD.  AUC vs a shuffled-tile null.
 (3) temporal persistence of tile flip activity.
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
model.load_state_dict(torch.load("init.pt")); torch.manual_seed(17)
o=torch.optim.AdamW(model.parameters(), lr=LR*5, betas=(0.9,0.95), weight_decay=0.1)
P=flat().numel(); NT=1024
seg=(torch.arange(P)*NT//P).long(); cnt=torch.bincount(seg,minlength=NT).float()
NBIN=20; hb=np.zeros(NBIN); hc=np.zeros(NBIN); edges=None
FEA=[]; FLIP=[]
prev=flat(); sp=None
for s in range(1,161):
    model.train(); x,y=get_batch(); _,l=model(x,y)
    o.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    b4=flat(); o.step(); af=flat(); d=af-b4; sg=torch.sign(d)
    m=state(o,"exp_avg").abs(); v=state(o,"exp_avg_sq").sqrt()
    r=m/(v+1e-12)
    tm=torch.zeros(NT).index_add_(0,seg,r)/cnt
    tv=torch.zeros(NT).index_add_(0,seg,(r-tm[seg])**2)/cnt
    tau=float(torch.quantile(r[torch.randperm(P)[:200000]],0.2))
    tf=torch.zeros(NT).index_add_(0,seg,(r<tau).float())/cnt
    FEA.append(torch.stack([tm,tv.sqrt(),tf]).numpy())
    if sp is not None:
        fl=(sg!=sp).float()
        FLIP.append((torch.zeros(NT).index_add_(0,seg,fl)/cnt).numpy())
        if s>20:
            sub=torch.randperm(P)[:300000]
            rr=r[sub]; ff=fl[sub]
            if edges is None:
                edges=torch.quantile(rr, torch.linspace(0,1,NBIN+1)[1:-1])
            b=torch.bucketize(rr,edges)
            for i in range(NBIN):
                k=(b==i)
                if k.any(): hb[i]+=float(ff[k].sum()); hc[i]+=float(k.sum())
    sp=sg; prev=af; del b4,af,d,m,v,r
    if s%40==0: gc.collect(); print(f"    step {s} ({time.time()-t0:.0f}s)", flush=True)
F=np.stack(FEA[1:]); Y=np.stack(FLIP)          # aligned: features at t, flips at t
print("\n"+"="*80); print("  (1) HAZARD CURVE h(r)"); print("="*80)
rate=hb/np.maximum(hc,1)
print(f"  {'bin':>5}{'flip rate':>12}{'log(rate)':>12}")
for i in range(0,NBIN,2):
    print(f"  {i+1:>5}{100*rate[i]:>11.2f}%{np.log(max(rate[i],1e-9)):>12.3f}")
xb=np.arange(NBIN); lg=np.log(np.maximum(rate,1e-9))
A=np.polyfit(xb,lg,1); pred=np.exp(np.polyval(A,xb))
r2=1-((rate-pred)**2).sum()/((rate-rate.mean())**2).sum()
print(f"\n  exponential fit in bin index: R^2 = {r2:.3f}   decay {A[0]:.3f}/bin"
      f"   ratio bin1/bin20 = {rate[0]/max(rate[-1],1e-12):.0f}x")
print("\n"+"="*80); print("  (2) PREDICT TILE FLIP ACTIVITY k STEPS AHEAD"); print("="*80)
def auc(sc,lab):
    n1=lab.sum(); n0=len(lab)-n1
    if n1==0 or n0==0: return np.nan
    rk=np.argsort(np.argsort(sc))+1
    return (rk[lab>0].sum()-n1*(n1+1)/2)/(n1*n0)
print(f"  {'lead k':>8}{'AUC mean r':>13}{'AUC frac<tau':>15}{'AUC 3-feat':>13}{'AUC shuffled':>14}")
T=len(Y)
for k in (1,5,10,20):
    A1=[];A2=[];A3=[];A4=[]
    for t in range(20,T-k,5):
        lab=(Y[t+k]>np.median(Y[t+k])).astype(float)
        A1.append(auc(-F[t][0],lab)); A2.append(auc(F[t][2],lab))
        X=np.stack([F[t][0],F[t][1],F[t][2]],1)
        w=np.linalg.lstsq(np.hstack([X,np.ones((NT,1))]),Y[t+k],rcond=None)[0]
        A3.append(auc(np.hstack([X,np.ones((NT,1))])@w,lab))
        A4.append(auc(np.random.permutation(-F[t][0]),lab))
    print(f"  {k:>8}{np.nanmean(A1):>13.3f}{np.nanmean(A2):>15.3f}"
          f"{np.nanmean(A3):>13.3f}{np.nanmean(A4):>14.3f}")
print("\n"+"="*80); print("  (3) TEMPORAL PERSISTENCE OF TILE ACTIVITY"); print("="*80)
ac=[np.mean([np.corrcoef(Y[t],Y[t+k])[0,1] for t in range(20,T-k,5)]) for k in (1,5,10,20,40)]
print("  autocorr of tile flip fraction:  " + "  ".join(f"k={k}:{a:+.3f}" for k,a in zip((1,5,10,20,40),ac)))
print(f"\n  time {time.time()-t0:.0f}s")
