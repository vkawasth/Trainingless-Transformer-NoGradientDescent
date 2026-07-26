"""
THE ONLY WELL-POSED fwd/bwd COMBINATION:  dL = <g, d>, additive over strands.
Measures (i) how well first order predicts the actual loss change,
         (ii) how the loss change distributes over strands,
         (iii) how many strands contribute NEGATIVELY (work against the step).
"""
import re, time, numpy as np, torch
t0=time.time()
g_={}; src=open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
exec(src[:src.find("# ── PHASE 1")], g_)
model=g_["model"]; get_batch=g_["get_batch"]; LR=g_["LR"]
named=list(model.named_parameters())
def flat(): return torch.cat([p.data.flatten() for _,p in named]).clone()
def setflat(v):
    i=0
    with torch.no_grad():
        for _,p in named:
            k=p.numel(); p.data.copy_(v[i:i+k].view_as(p)); i+=k
off=0; SPAN={}
for n,p in named: SPAN[n]=(off,off+p.numel()); off+=p.numel()
P=off
def lay(n):
    m=re.match(r"blocks\.(\d+)\.",n); return f"L{m.group(1)}" if m else "EMB"
GI={}
for n,_ in named: GI.setdefault(lay(n),[]).append(torch.arange(*SPAN[n]))
GI={k:torch.cat(v) for k,v in GI.items()}; KEYS=["EMB"]+[f"L{i}" for i in range(6)]
NS=1024
e=np.linspace(0,P,NS+1).astype(int)
model.load_state_dict(torch.load("init.pt")); torch.manual_seed(17)
opt=torch.optim.AdamW(model.parameters(), lr=LR*5, betas=(0.9,0.95), weight_decay=0.1)
print("="*80); print("  dL = <g,d>  :  FIRST-ORDER PAIRING OF THE TWO SIDES"); print("="*80)
print(f"  {'step':>6}{'dL actual':>12}{'<g,d>':>12}{'ratio':>8}{'|neg strands|':>15}"
      f"{'top10% share':>14}{'Gini':>7}")
for s in range(1,161):
    model.train(); x,y=get_batch(); _,l0=model(x,y)
    opt.zero_grad(); l0.backward()
    gv=torch.cat([(p.grad.flatten() if p.grad is not None else torch.zeros(p.numel()))
                  for _,p in named]).clone()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    b4=flat(); opt.step(); af=flat(); d=af-b4
    if s in (5,20,40,80,120,160):
        model.eval()
        with torch.no_grad(): _,l1=model(x,y)
        model.train()
        dL=float(l1)-float(l0); gd=float(gv@d)
        contrib=np.array([float(gv[e[i]:e[i+1]]@d[e[i]:e[i+1]]) for i in range(NS)])
        neg=100*float((contrib>0).mean())          # >0 means it RAISED the loss
        a=np.abs(contrib); srt=np.sort(a)[::-1]; top=100*srt[:NS//10].sum()/a.sum()
        cs=np.cumsum(np.sort(a))/a.sum(); gini=1-2*np.trapezoid(cs,dx=1/NS)
        print(f"  {s:>6}{dL:>12.5f}{gd:>12.5f}{dL/gd if gd!=0 else float('nan'):>8.3f}"
              f"{neg:>14.1f}%{top:>13.1f}%{gini:>7.3f}", flush=True)
print("\n  ratio ~1 => first order dominates; <g,d> is the exact fwd/bwd pairing")
print("  'neg strands' = % of strands whose local pairing INCREASED the loss")
print(f"\n  time {time.time()-t0:.0f}s")
