"""
PREDICTIVE FISHER SUFFICIENCY.
Does the residual eps = u - a*sign(u) carry information about the FUTURE that
(r, sign u) does not?  Predict dL_{t+k} from three feature sets:
  (1) strand state: block summaries of r and sign-agreement
  (2) residual   : block summaries of |eps|
  (3) both
If residual adds ~0 over strand, dynamics factor through the strand state.
Control: also predict from the full update's block summaries (ceiling).
"""
import time, gc, numpy as np, torch
from numpy.linalg import lstsq
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
def vloss(n=8): 
    model.eval(); L=[]
    with torch.no_grad():
        for _ in range(n): x,y=get_batch(); _,l=model(x,y); L.append(float(l))
    model.train(); return float(np.mean(L))
P=flat().numel(); NT=64; seg=(torch.arange(P)*NT//P).long(); cnt=torch.bincount(seg,minlength=NT).float()
model.load_state_dict(torch.load("init.pt")); torch.manual_seed(17)
o=torch.optim.AdamW(model.parameters(), lr=LR*5, betas=(0.9,0.95), weight_decay=0.1)
STR=[]; RES=[]; FUL=[]; LOS=[]
prev=flat(); sp=None
for s in range(1,161):
    model.train(); x,y=get_batch(); _,l=model(x,y)
    o.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    b4=flat(); o.step(); af=flat(); u=af-b4; sg=torch.sign(u)
    m=state(o,"exp_avg").abs(); v=state(o,"exp_avg_sq").sqrt(); r=m/(v+1e-12)
    # a per tile, residual
    num=torch.zeros(NT).index_add_(0,seg,u*sg); den=torch.zeros(NT).index_add_(0,seg,sg*sg)
    a=(num/(den+1e-30)); rec=a[seg]*sg; eps=u-rec
    rt=torch.zeros(NT).index_add_(0,seg,r)/cnt
    agree=torch.zeros(NT); 
    if sp is not None: agree=torch.zeros(NT).index_add_(0,seg,(sg==sp).float())/cnt
    et=torch.zeros(NT).index_add_(0,seg,eps.abs())/cnt
    ut=torch.zeros(NT).index_add_(0,seg,u.abs())/cnt
    STR.append(torch.cat([rt,agree]).numpy()); RES.append(et.numpy()); FUL.append(ut.numpy())
    LOS.append(vloss(6) if s%1==0 else np.nan)
    sp=sg; prev=af; del b4,af,u,eps,m,v,r
    if s%40==0: gc.collect(); print(f"    step {s} ({time.time()-t0:.0f}s)", flush=True)
STR=np.stack(STR); RES=np.stack(RES); FUL=np.stack(FUL); LOS=np.array(LOS)
def cvr2(X,y,k):
    # predict y[t+k] from X[t]; 5-fold blocked CV R^2
    Xt=X[:-k]; yt=y[k:]; n=len(yt); f=n//5; r2=[]
    for i in range(5):
        te=np.arange(i*f,(i+1)*f); tr=np.setdiff1d(np.arange(n),te)
        A=np.hstack([Xt[tr],np.ones((len(tr),1))]); w=lstsq(A,yt[tr],rcond=None)[0]
        pr=np.hstack([Xt[te],np.ones((len(te),1))])@w
        ss=((yt[te]-yt[tr].mean())**2).sum()
        r2.append(1-((yt[te]-pr)**2).sum()/max(ss,1e-30))
    return np.mean(r2)
print("\n"+"="*80); print("  PREDICT dL_{t+k}: does the residual add over the strand state?"); print("="*80)
dL=np.diff(LOS,prepend=LOS[0])
print(f"  {'k':>4}{'strand(r,agree)':>18}{'residual':>11}{'strand+res':>13}{'full update':>14}")
for k in (1,3,5,10):
    rs=cvr2(STR,dL,k); rr=cvr2(RES,dL,k)
    rb=cvr2(np.hstack([STR,RES]),dL,k); rf=cvr2(FUL,dL,k)
    print(f"  {k:>4}{rs:>18.3f}{rr:>11.3f}{rb:>13.3f}{rf:>14.3f}", flush=True)
print("\n  if strand+res ~ strand, residual is trajectory-null and dynamics")
print("  factor through the strand state (r, sign).")
print(f"\n  time {time.time()-t0:.0f}s")
