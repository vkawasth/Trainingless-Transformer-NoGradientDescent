"""
FISHER RECONSTRUCTION OF THE SIGN FACTORISATION.
u ~ a . sign(u).  How much of u does this capture, in the Fisher metric, at
increasing disc resolution?  Contrast with the known fact that downstream loss
is unaffected even at resolution 1.
"""
import re, time, gc, numpy as np, torch, torch.nn.functional as Fn
t0=time.time()
g_={}; src=open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
exec(src[:src.find("# ── PHASE 1")], g_)
model=g_["model"]; get_batch=g_["get_batch"]; LR=g_["LR"]
named=list(model.named_parameters())
def flat(): return torch.cat([p.data.flatten() for _,p in named]).clone()
P=flat().numel()
model.load_state_dict(torch.load("init.pt")); torch.manual_seed(17)
o=torch.optim.AdamW(model.parameters(), lr=LR*5, betas=(0.9,0.95), weight_decay=0.1)
for s in range(100):
    model.train(); x,y=get_batch(); _,l=model(x,y)
    o.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); o.step()
# diagonal Fisher, labels sampled from the model
G=torch.zeros(P); model.eval()
for b in range(8):
    x,_=get_batch(); lo,_=model(x)
    lp=Fn.log_softmax(lo.reshape(-1,lo.shape[-1]),dim=-1)
    ys=torch.multinomial(lp.exp(),1).squeeze(1)
    model.zero_grad(set_to_none=True); Fn.nll_loss(lp,ys).backward()
    gv=torch.cat([(p.grad.flatten() if p.grad is not None else torch.zeros(p.numel()))
                  for _,p in named]); G+=gv*gv
G/=8; model.train()
print(f"  Fisher trace {float(G.sum()):.4e}   ({time.time()-t0:.0f}s)", flush=True)
def lay(n):
    m=re.match(r"blocks\.(\d+)\.",n); return f"L{m.group(1)}" if m else "EMB"
off=0; SEG={}
LS=[]
for n,p in named: LS+= [lay(n)]*p.numel()
KEYS=["EMB"]+[f"L{i}" for i in range(6)]
segL=torch.tensor([KEYS.index(k) for k in LS])
GRAN={"global (1)":torch.zeros(P,dtype=torch.long),
      "per layer (7)":segL,
      "per tile (1024)":(torch.arange(P)*1024//P).long(),
      "per tile (16384)":(torch.arange(P)*16384//P).long()}
ACC={k:{"g":[], "e":[]} for k in GRAN}
prev=flat()
for s in range(60):
    model.train(); x,y=get_batch(); _,l=model(x,y)
    o.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    b4=flat(); o.step(); af=flat(); u=af-b4; sg=torch.sign(u)
    for k,seg in GRAN.items():
        n=int(seg.max())+1
        num=torch.zeros(n).index_add_(0,seg,G*u*sg)      # Fisher-optimal a per group
        den=torch.zeros(n).index_add_(0,seg,G*sg*sg)
        a=num/(den+1e-30)
        rec=a[seg]*sg; res=u-rec
        ACC[k]["g"].append(1-float((G*res*res).sum()/max(float((G*u*u).sum()),1e-30)))
        # euclidean version
        num2=torch.zeros(n).index_add_(0,seg,u*sg); den2=torch.zeros(n).index_add_(0,seg,sg*sg)
        rec2=(num2/(den2+1e-30))[seg]*sg; r2=u-rec2
        ACC[k]["e"].append(1-float((r2*r2).sum()/max(float((u*u).sum()),1e-30)))
    prev=af; del b4,af,u,sg
    if s%20==19: gc.collect()
print("\n"+"="*80); print("  RECONSTRUCTION  u ~ a . sign(u)   (60 steps, from step 100)"); print("="*80)
print(f"  {'disc resolution':>20}{'R (Fisher)':>14}{'R (euclidean)':>16}")
for k in GRAN:
    print(f"  {k:>20}{np.mean(ACC[k]['g']):>14.4f}{np.mean(ACC[k]['e']):>16.4f}")
print("\n  For contrast, the same resolutions in the TRAINING experiment gave:")
print("    resolution 1     -> 100.1% of the loss improvement")
print("    resolution 64    -> 100.2%")
print("    resolution 1024  -> 100.2%")
print(f"\n  time {time.time()-t0:.0f}s")
