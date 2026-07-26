"""
THE MISSING CONTROL FOR STALE-SIGN.
k=2 stale sign: 200 parameter updates from 100 backward passes -> val 0.1670.
Fair comparison is NOT 200 full steps (0.0677) but 100 full steps: same compute.
Also: sign-flip rate per step, to test the 'sparse event detection' idea.
"""
import time, numpy as np, torch
t0=time.time()
g_={}; src=open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
exec(src[:src.find("# ── PHASE 1")], g_)
model=g_["model"]; get_batch=g_["get_batch"]; eval_val=g_["eval_val"]; LR=g_["LR"]
named=list(model.named_parameters())
def flat(): return torch.cat([p.data.flatten() for _,p in named]).clone()
def newopt(): return torch.optim.AdamW(model.parameters(), lr=LR*5, betas=(0.9,0.95), weight_decay=0.1)
def full(n):
    model.load_state_dict(torch.load("init.pt")); torch.manual_seed(17); o=newopt()
    P=flat().numel(); prev=flat(); flips=[]; sp=None
    for s in range(n):
        model.train(); x,y=get_batch(); _,l=model(x,y)
        o.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); o.step()
        cur=flat(); d=cur-prev; prev=cur; sg=torch.sign(d)
        if sp is not None: flips.append(float((sg!=sp).double().mean()))
        sp=sg
    return float(eval_val(model,n=16)), np.array(flips), P
print("="*80); print("  COMPUTE-MATCHED CONTROL FOR STALE-SIGN"); print("="*80)
res={}
for n in (50,100,133,200):
    v,fl,P=full(n); res[n]=v
    print(f"  {n:>4} full AdamW steps  ({n} backward passes)   val {v:.4f}", flush=True)
print(f"\n  stale-sign k=2: 200 updates from 100 backward passes -> val 0.1670")
print(f"  compute-matched baseline (100 full steps)            -> val {res[100]:.4f}")
if res[100] < 0.1670:
    print(f"  => the saving is ILLUSORY: 100 honest steps beat it by "
          f"{100*(0.1670-res[100])/0.1670:.0f}%")
else:
    print(f"  => stale-sign genuinely beats equal compute by "
          f"{100*(res[100]-0.1670)/res[100]:.0f}%")
print(f"\n  stale-sign k=4: 200 updates from 50 backward passes -> val 0.6606")
print(f"  compute-matched baseline (50 full steps)             -> val {res[50]:.4f}")
_,fl,P=full(200)
print("\n"+"="*80); print("  IS SIGN-FLIPPING A SPARSE EVENT?"); print("="*80)
print(f"  mean flip rate per step: {100*fl.mean():.1f}%  of {P:,} coordinates")
print(f"  => {int(fl.mean()*P):,} sign flips PER STEP")
print(f"  first 50 steps {100*fl[:50].mean():.1f}%   last 50 {100*fl[-50:].mean():.1f}%"
      f"   max {100*fl.max():.1f}%")
print(f"\n  the trajectory crosses ~{int(fl.mean()*P):,} sign hyperplanes each step,")
print("  so the sign pattern is not a sparse event stream and the orthant label")
print("  changes completely at every step.")
print(f"\n  time {time.time()-t0:.0f}s")
