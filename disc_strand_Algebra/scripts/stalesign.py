"""
CAN THE SIGN PATTERN BE REUSED ACROSS STEPS?
If so, backward passes can be SKIPPED -- a compute saving, not just bandwidth.
Arm k: real forward+backward every k steps (caching sign(d) and c=mean|d|);
       on the other k-1 steps apply  theta += sign_cached * c  with NO backward.
Also measures the raw step-to-step sign agreement rate.
"""
import sys, time, numpy as np, torch
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
def V(n=16): return float(eval_val(model,n=n))
def newopt(): return torch.optim.AdamW(model.parameters(), lr=LR*5, betas=(0.9,0.95), weight_decay=0.1)
model.load_state_dict(torch.load("init.pt")); v0=V()
# ---- baseline + sign agreement ----
model.load_state_dict(torch.load("init.pt")); torch.manual_seed(17)
o=newopt(); prev=flat(); sprev=None; agree=[]
for s in range(200):
    model.train(); x,y=get_batch(); _,l=model(x,y)
    o.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); o.step()
    cur=flat(); d=cur-prev; prev=cur; sg=torch.sign(d)
    if sprev is not None: agree.append(float((sg==sprev).double().mean()))
    sprev=sg
vb=V(); A=np.array(agree)
print("="*80); print("  SIGN PERSISTENCE ACROSS STEPS"); print("="*80)
print(f"  baseline val {vb:.4f}  (init {v0:.4f})")
print(f"  step-to-step sign agreement: mean {100*A.mean():.1f}%  "
      f"first50 {100*A[:50].mean():.1f}%  last50 {100*A[-50:].mean():.1f}%  "
      f"min {100*A.min():.1f}%")
pct=lambda v: 100*(v0-v)/max(v0-vb,1e-12)
def run(k):
    model.load_state_dict(torch.load("init.pt")); torch.manual_seed(17)
    o=newopt(); prev=flat(); cs=None; cc=None; nb=0
    for s in range(200):
        if s % k == 0:
            model.train(); x,y=get_batch(); _,l=model(x,y)
            o.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            o.step(); nb+=1
            cur=flat(); d=cur-prev
            cs=torch.sign(d); cc=float(d.abs().mean()); prev=cur
        else:
            with torch.no_grad():
                nv=prev+cs*cc
            setflat(nv); prev=nv
    return V(), nb
print(f"\n  {'k':>4}{'backward passes':>18}{'saved':>9}{'val':>10}{'% of improvement':>19}")
print(f"  {1:>4}{200:>18}{'0%':>9}{vb:>10.4f}{100.0:>18.1f}%")
for k in [int(a) for a in (sys.argv[1:] or ["2","4"])]:
    v,nb=run(k)
    print(f"  {k:>4}{nb:>18}{100*(1-nb/200):>8.0f}%{v:>10.4f}{pct(v):>18.1f}%", flush=True)
print(f"\n  time {time.time()-t0:.0f}s")
