"""
2x3 DISC per strand: columns = [mean|d|, max|d|, Fisher-optimal a], row = sign.
Fisher-optimal a_T minimises ||G^{1/2}(u - a sign(u))||^2 on strand T:
   a_T = sum_i G_i |u_i| / sum_i G_i        (since sign*u = |u|, sign^2 = 1)
Compare training retention using each column as the disc magnitude.
"""
import time, gc, numpy as np, torch, torch.nn.functional as Fn
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
def newopt(): return torch.optim.AdamW(model.parameters(), lr=LR*5, betas=(0.9,0.95), weight_decay=0.1)
P=flat().numel(); NS=1024; e=np.linspace(0,P,NS+1).astype(int)
seg=torch.zeros(P,dtype=torch.long)
for i in range(NS): seg[e[i]:e[i+1]]=i
cnt=torch.bincount(seg,minlength=NS).double()
# diagonal Fisher once, at a trained checkpoint
model.load_state_dict(torch.load("init.pt")); torch.manual_seed(17); o=newopt()
for s in range(100):
    model.train(); x,y=get_batch(); _,l=model(x,y)
    o.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); o.step()
G=torch.zeros(P)
for _ in range(8):
    x,_=get_batch(); lo,_=model(x); lp=Fn.log_softmax(lo.reshape(-1,lo.shape[-1]),1)
    ys=torch.multinomial(lp.exp(),1).squeeze(1); model.zero_grad(set_to_none=True); Fn.nll_loss(lp,ys).backward()
    G+=torch.cat([(p.grad.flatten() if p.grad is not None else torch.zeros(p.numel())) for _,p in named])**2
G/=8; Gd=G.double()
print(f"  Fisher ready ({time.time()-t0:.0f}s)", flush=True)
def run(col):
    model.load_state_dict(torch.load("init.pt")); torch.manual_seed(17); ob=newopt(); prev=flat()
    for s in range(200):
        model.train(); x,y=get_batch(); _,l=model(x,y)
        ob.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); ob.step()
        cur=flat(); d=cur-prev; ad=d.abs().double(); sg=torch.sign(d)
        if col=="mean":
            a=(torch.zeros(NS,dtype=torch.float64).index_add_(0,seg,ad)/cnt)
        elif col=="max":
            a=torch.zeros(NS,dtype=torch.float64).scatter_reduce(0,seg,ad,reduce="amax",include_self=False)
        elif col=="fisher":
            num=torch.zeros(NS,dtype=torch.float64).index_add_(0,seg,Gd*ad)
            den=torch.zeros(NS,dtype=torch.float64).index_add_(0,seg,Gd)
            a=num/(den+1e-30)
        d2=sg*a[seg].float()
        setflat(prev+d2); prev=flat()
    return float(eval_val(model,n=16))
model.load_state_dict(torch.load("init.pt")); torch.manual_seed(17); ob=newopt(); prev=flat()
for s in range(200):
    model.train(); x,y=get_batch(); _,l=model(x,y)
    ob.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); ob.step()
vb=float(eval_val(model,n=16)); v0=4.4604
pct=lambda v:100*(v0-v)/max(v0-vb,1e-12)
print("="*66); print("  2x3 DISC: which magnitude column?"); print("="*66)
print(f"  baseline (full update) val {vb:.4f}")
print(f"  {'disc column':>16}{'val':>10}{'% of improvement':>19}")
import sys
for col in (sys.argv[1:] or ["fisher"]):
    v=run(col); print(f"  {col:>16}{v:>10.4f}{pct(v):>18.1f}%", flush=True)
print(f"\n  time {time.time()-t0:.0f}s")
