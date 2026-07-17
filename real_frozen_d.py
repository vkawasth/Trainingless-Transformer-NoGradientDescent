import math, copy
from collections import defaultdict
import numpy as np, torch

g_ = {}
#src = open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
src = open("compiler_geometri_patched_86.py").read()
exec(src[:src.find("# ── PHASE 1")], g_)
model=g_["model"]; get_batch=g_["get_batch"]; eval_val=g_["eval_val"]; LR=g_["LR"]

def group_of(name):
    n=name.lower()
    if ".ln." in n or ".n." in n or n.startswith("ln_f"): return "LayerNorm"
    if n.startswith("te") or n.startswith("pe") or n.startswith("head"): return "Emb"
    if ".ff." in n: return "FF"
    if "wk" in n: return "W_K"
    if "wq" in n: return "W_Q"
    if "wv" in n: return "W_V"
    if ".op." in n: return "W_O"
    return "other"

class FrozenD:
    """Adam with the second moment FROZEN: fixed diagonal preconditioner D +
    live momentum. Tests whether Q/K need the SHAPE of D (freezable) or its
    ongoing ADAPTATION (not)."""
    def __init__(self, params, Ds, lr, b1=0.9, wd=0.1):
        self.params=params; self.D=Ds; self.lr=lr; self.b1=b1; self.wd=wd
        self.m=[torch.zeros_like(p) for p in params]; self.t=0
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None: p.grad=None
    def step(self):
        self.t+=1
        for p,D,m in zip(self.params,self.D,self.m):
            if p.grad is None: continue
            m.mul_(self.b1).add_(p.grad, alpha=1-self.b1)
            mhat=m/(1-self.b1**self.t)
            p.data.add_(mhat*D, alpha=-self.lr).add_(p.data, alpha=-self.lr*self.wd)

def warm_and_snapshot(steps=110):
    opt=torch.optim.AdamW(model.parameters(), lr=LR*5, betas=(0.9,0.95), weight_decay=0.1)
    for _ in range(steps):
        model.train(); x,y=get_batch(); _,l=model(x,y)
        opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    # frozen D per param from branch v
    b1,b2=opt.param_groups[0]["betas"]; st=opt.state; eps=1e-8
    Dfroz={}
    for name,p in model.named_parameters():
        if p in st and "exp_avg_sq" in st[p]:
            step=st[p]["step"]; step=float(step.item()) if torch.is_tensor(step) else float(step)
            vhat=st[p]["exp_avg_sq"]/(1-b2**max(step,1))
            Dfroz[name]=(1.0/(vhat.sqrt()+eps)).clone()
    return copy.deepcopy(model.state_dict()), Dfroz, eval_val(model,n=8)

def run(kind, branch, Dfroz, froz_groups, steps=100):
    model.load_state_dict(copy.deepcopy(branch))
    named=list(model.named_parameters())
    froz_p=[p for n,p in named if group_of(n) in froz_groups]
    froz_D=[Dfroz[n] for n,p in named if group_of(n) in froz_groups]
    keep_p=[p for n,p in named if group_of(n) not in froz_groups]
    opt_a=torch.optim.AdamW(keep_p, lr=LR*5, betas=(0.9,0.95), weight_decay=0.1) if keep_p else None
    opt_f=FrozenD(froz_p, froz_D, lr=LR*5) if froz_p else None
    traj=[]
    for s in range(1,steps+1):
        model.train(); x,y=get_batch(); _,l=model(x,y)
        if opt_a: opt_a.zero_grad()
        if opt_f: opt_f.zero_grad()
        l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        if opt_a: opt_a.step()
        if opt_f: opt_f.step()
        if s%20==0: traj.append((s, eval_val(model,n=8)))
    return traj

def main():
    P=sum(p.numel() for p in model.parameters())
    branch, Dfroz, vB = warm_and_snapshot(110)
    print("="*76); print("  FROZEN-D TEST ON THE CORE (real model)"); print("="*76)
    print(f"  branch: step 110  val={vB:.4f}   P={P:,}")
    print("  Q: does Q/K adaptivity need to keep ADAPTING, or just the frozen shape?\n")
    arms={
      "baseline AdamW":      set(),
      "freeze W_Q,W_K":      {"W_Q","W_K"},
      "freeze Q,K,V,O":      {"W_Q","W_K","W_V","W_O"},
      "freeze core+Emb+FF":  {"W_Q","W_K","W_V","W_O","FF","Emb"},
    }
    res={}
    for name,fg in arms.items():
        res[name]=run(name, branch, Dfroz, fg)
        froz=sum(p.numel() for n,p in model.named_parameters() if group_of(n) in fg)
        print(f"  {name:<22} frozen {100*froz/P:>3.0f}% of params  ->  "
              f"val@100 = {res[name][-1][1]:.4f}")
    base=res["baseline AdamW"][-1][1]
    print("\n  "+"-"*72)
    for name in arms:
        d=100*(res[name][-1][1]-base)/base
        print(f"  {name:<22} {d:+6.1f}% vs AdamW  "
              f"{'<- frozen shape suffices' if d<3 else '<- needs live adaptation'}")
    print("\n  If freezing Q/K stays within noise, the dispersed preconditioner is")
    print("  a SHAPE you can snapshot -- cull the v-update, keep the diagonal. That")
    print("  is the only lever that reaches past the 18% V/O ceiling into the core.")

if __name__=="__main__": main()
