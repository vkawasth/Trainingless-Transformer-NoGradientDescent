"""
IS MOMENTUM THE TRANSPORT PROJECTION?
beta1 is a low-pass filter on the update. If the oscillatory component is
separable and removable, raising beta1 should cut cancellation and shorten the
path. The question is whether progress survives.
"""
import time, gc, numpy as np, torch
t0=time.time()
g_={}; src=open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
exec(src[:src.find("# ── PHASE 1")], g_)
model=g_["model"]; get_batch=g_["get_batch"]; eval_val=g_["eval_val"]; LR=g_["LR"]
named=list(model.named_parameters())
def flat(): return torch.cat([p.data.flatten() for _,p in named]).clone()
def run(b1, T=200):
    model.load_state_dict(torch.load("init.pt")); torch.manual_seed(17)
    o=torch.optim.AdamW(model.parameters(), lr=LR*5, betas=(b1,0.95), weight_decay=0.1)
    prev=flat(); net=torch.zeros_like(prev); path=torch.zeros_like(prev)
    tot=0.0; cs=[]; dp=None; flips=[]; sp=None
    for s in range(T):
        model.train(); x,y=get_batch(); _,l=model(x,y)
        o.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); o.step()
        cur=flat(); d=cur-prev; prev=cur
        net+=d; path+=d.abs(); tot+=float(d.norm())
        if dp is not None: cs.append(float(d@dp)/max(float(d.norm()*dp.norm()),1e-30))
        sg=torch.sign(d)
        if sp is not None: flips.append(float((sg!=sp).double().mean()))
        dp=d.clone(); sp=sg
        if s%50==49: gc.collect()
    return dict(val=float(eval_val(model,n=16)), l1=float(net.abs().sum()/path.sum()),
                chord=float(net.norm()), path=tot, cons=np.mean(cs), flip=np.mean(flips))
print("="*88); print("  IS MOMENTUM THE PROJECTION THAT ISOLATES TRANSPORT?"); print("="*88)
print(f"  {'beta1':>7}{'val':>10}{'cancel %':>11}{'chord':>9}{'path':>9}"
      f"{'chord/path':>12}{'cos(d,d+1)':>12}{'flip %':>9}")
R={}
import sys
for b1 in [float(a) for a in (sys.argv[1:] or ['0.9'])]:
    r=run(b1); R[b1]=r
    print(f"  {b1:>7.2f}{r['val']:>10.4f}{100*(1-r['l1']):>10.1f}%{r['chord']:>9.1f}"
          f"{r['path']:>9.1f}{r['chord']/r['path']:>12.4f}{r['cons']:>12.3f}"
          f"{100*r['flip']:>8.1f}%", flush=True)
b=R.get(0.9, list(R.values())[0])
print(f"\n  relative to beta1=0.9 (default):")
for b1,r in R.items():
    if r is b: continue
    print(f"    beta1={b1:<5} val {r['val']/b['val']:>6.2f}x   path {r['path']/b['path']:>5.2f}x"
          f"   cancellation {100*(1-r['l1']):>5.1f}% vs {100*(1-b['l1']):.1f}%")
print("\n  If higher beta1 cuts cancellation AND keeps val, oscillation is removable.")
print("  If cancellation falls but val degrades, the oscillation is load-bearing.")
print(f"\n  time {time.time()-t0:.0f}s")
