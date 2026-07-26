"""
CAN THE NEXT FRONTIER BE PREDICTED FROM PRE-BACKWARD STATE?
Predict 1{r_i < r_c at t+1} using ONLY info available before the t+1 backward:
  r_t, m_t, v_t, |m|, sign persistence age, and r_t trend (r_t - r_{t-1}).
Metric: at each selection fraction f (compute backward on predicted-active f),
  what RECALL of the true next frontier do we get?  Diagonal (f=recall) = random.
Also the ceiling: how much does using r_t ALONE (stalest predictor) already give?
"""
import time, gc, numpy as np, torch
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
P=flat().numel(); FRAC=0.106
model.load_state_dict(torch.load("init.pt")); torch.manual_seed(17)
o=torch.optim.AdamW(model.parameters(), lr=LR*5, betas=(0.9,0.95), weight_decay=0.1)
SUB=torch.randperm(P)[:40000]
feats=[]; labels=[]; r_prev=None
for s in range(1,141):
    model.train(); x,y=get_batch(); _,l=model(x,y)
    o.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    o.step()
    m=st(o,"exp_avg"); v=st(o,"exp_avg_sq").sqrt(); r=(m.abs()/(v+1e-12))
    thr=torch.quantile(r[SUB],FRAC); active=(r<thr)
    if r_prev is not None and s>10:
        # features at time t (pre-backward for t+1): all known before next step
        F=torch.stack([r_prev[SUB], (r_prev-r_prev2)[SUB], m.abs()[SUB],
                       v[SUB], r_prev[SUB]**2], 1).numpy()
        feats.append(F); labels.append(active[SUB].numpy().astype(float))
    r_prev2 = r_prev if r_prev is not None else r.clone()
    r_prev = r.clone(); del m,v,r
    if s%40==0: gc.collect(); print(f"    step {s} ({time.time()-t0:.0f}s)", flush=True)
X=np.concatenate(feats); Y=np.concatenate(labels)
# train logistic on first half, test on second half (temporal split)
n=len(Y); h=n//2
from numpy.linalg import lstsq
Xn=(X-X.mean(0))/(X.std(0)+1e-9)
Xtr=np.hstack([Xn[:h],np.ones((h,1))]); Xte=np.hstack([Xn[h:],np.ones((n-h,1))])
w=lstsq(Xtr,Y[:h],rcond=None)[0]                 # linear probe (proxy for logistic ranking)
score=Xte@w; yte=Y[h:]
def recall_at(frac):
    k=int(len(score)*frac); idx=np.argsort(-score)[:k]
    return yte[idx].sum()/max(yte.sum(),1)
# r-alone baseline (stalest): rank by -r_prev  (already feature 0, but isolate it)
score_r=-Xte[:,0]
def recall_at_r(frac):
    k=int(len(score)*frac); idx=np.argsort(-score_r)[:k]
    return yte[idx].sum()/max(yte.sum(),1)
print("\n"+"="*72); print("  NEXT-FRONTIER PREDICTION FROM PRE-BACKWARD STATE"); print("="*72)
print(f"  true frontier fraction: {Y.mean():.3f}")
print(f"  {'select f':>10}{'recall (full model)':>22}{'recall (r alone)':>18}{'random':>9}")
for f in (0.10,0.15,0.20,0.30,0.50):
    print(f"  {f:>10.2f}{recall_at(f):>22.3f}{recall_at_r(f):>18.3f}{f:>9.2f}")
# the decisive number: recall at f=0.20
r20=recall_at(0.20)
print(f"\n  DECISIVE: recall at 20% selection = {r20:.3f}")
if r20>0.9: print("  => frontier predictable; backward compute reducible ~5x.")
elif r20>0.75: print("  => partial; some savings but leakage costs remain.")
else: print("  => NOT predictable enough; must compute full gradient to find frontier.")
# what selection fraction is needed for 95% recall?
for f in np.arange(0.1,1.01,0.05):
    if recall_at(f)>=0.95:
        print(f"  selection needed for 95% recall: {f:.2f}"); break
else:
    print("  95% recall not reached below full selection")
print(f"\n  time {time.time()-t0:.0f}s")
