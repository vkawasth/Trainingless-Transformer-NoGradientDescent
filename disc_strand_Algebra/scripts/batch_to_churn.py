"""
DOES THE CURRENT BATCH PREDICT THE NEXT FRONTIER, beyond optimizer state?
Add FORWARD-PASS features of B_t (available before the backward pass) to the
predictor and re-measure frontier recall at fixed budget.
For a weight W in FF: its gradient is (delta_out^T @ act_in). Before the backward
we HAVE act_in (forward) and can get a cheap proxy for which weights this batch
will touch: rows/cols with large forward activation. Test whether
  activation-magnitude features + r  >>  r alone.
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
off=0; SPAN={}
for n,p in named: SPAN[n]=(off,off+p.numel()); off+=p.numel()
# focus one FF weight; capture its input activation via forward hook
TN="blocks.2.ff.g.weight"; a0,_=SPAN[TN]
lin=dict(model.named_modules())["blocks.2.ff.g"]
R,C=model.get_parameter(TN).shape   # (2*DM, DM) = (512,256): out=R, in=C
cache={}
def hook(m,i,o): cache["ain"]=i[0].detach()
lin.register_forward_hook(hook)
model.load_state_dict(torch.load("init.pt")); torch.manual_seed(17)
o=torch.optim.AdamW(model.parameters(), lr=LR*5, betas=(0.9,0.95), weight_decay=0.1)
FRAC=0.106
Rprev=None; Feat=[]; Lab=[]
for s in range(1,141):
    model.train(); x,y=get_batch(); _,l=model(x,y)
    # forward activation feature for this weight: mean |input| per input-dim -> broadcast to cols
    ain=cache["ain"].reshape(-1,C).abs().mean(0)          # (C,) per input feature
    o.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); o.step()
    r=(st(o,"exp_avg").abs()/(st(o,"exp_avg_sq").sqrt()+1e-12))
    rW=r[a0:a0+R*C].view(R,C)
    thr=torch.quantile(r[torch.randperm(len(r))[:100000]],FRAC)
    actW=(rW<thr)                                         # this weight's frontier
    if Rprev is not None and s>10:
        # features per weight (i,j): r_prev, |input act_j| (batch signal), r_prev*act
        rp=Rprev
        actcol=ain.view(1,C).expand(R,C)                 # forward activation of input j
        F=torch.stack([rp.flatten(), actcol.flatten(),
                       (rp*actcol).flatten(), rp.flatten()**2], 1).numpy()
        Feat.append(F); Lab.append(actW.flatten().numpy().astype(float))
    Rprev=rW.clone(); del r
    if s%40==0: gc.collect(); print(f"    step {s} ({time.time()-t0:.0f}s)", flush=True)
X=np.concatenate(Feat); Y=np.concatenate(Lab)
n=len(Y); h=n//2
Xn=(X-X.mean(0))/(X.std(0)+1e-9)
from numpy.linalg import lstsq
def recall_curve(cols,label):
    Xtr=np.hstack([Xn[:h][:,cols],np.ones((h,1))]); Xte=np.hstack([Xn[h:][:,cols],np.ones((n-h,1))])
    w=lstsq(Xtr,Y[:h],rcond=None)[0]; sc=Xte@w; yte=Y[h:]
    out=[]
    for f in (0.10,0.20,0.30,0.50):
        k=int(len(sc)*f); idx=np.argsort(-sc)[:k]; out.append(yte[idx].sum()/max(yte.sum(),1))
    print(f"  {label:>34}: " + "  ".join(f"{f*100:.0f}%:{rc:.3f}" for f,rc in zip((.1,.2,.3,.5),out)))
    return out
print("\n"+"="*80); print("  FRONTIER RECALL: r-only  vs  r+batch-activation"); print("="*80)
print(f"  frontier base rate {Y.mean():.3f}   (weight {TN})")
recall_curve([0],          "r_prev only")
recall_curve([1],          "batch activation only")
recall_curve([0,1,2,3],    "r + batch activation + interaction")
print("\n  if r+batch >> r-only at 20%, the batch carries the missing frontier signal.")
print("  if equal, optimizer state already contains what the batch would add.")
print(f"\n  time {time.time()-t0:.0f}s")
