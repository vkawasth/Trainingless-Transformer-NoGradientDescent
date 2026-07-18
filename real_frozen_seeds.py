"""
real_frozen_seeds.py
====================
Pin the ONE number that decides the past-18% lever: is freezing Q/K's second
moment free, or does it cost?  Two single runs disagreed (+0.7% vs +8.2%), and
the effect is ~the size of init/eval noise.  This settles it properly.

Design (paired + noise floor):
  for each seed:
    - warm to the branch with AdamW (same init drives both arms)
    - snapshot branch model state + frozen D for Q/K
    - arm BASE   : continue AdamW from branch
    - arm FROZEN : continue, Q/K on frozen D (v-update culled), rest AdamW
    - arm BASE2  : continue AdamW from branch with a DIFFERENT data order
                   -> baseline-vs-BASE2 gives the NOISE FLOOR the frozen delta
                      must beat to count as real.
  report per-seed deltas, mean +/- std, and whether frozen delta exceeds floor.

Run:  python real_frozen_seeds.py --seeds 5 --warm 110 --post 100
"""
import argparse, math, copy
from collections import defaultdict
import numpy as np, torch

g_={}; src=open("compiler_geometri_patched_86.py").read()
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
            m.mul_(self.b1).add_(p.grad, alpha=1-self.b1); mhat=m/(1-self.b1**self.t)
            p.data.add_(mhat*D, alpha=-self.lr).add_(p.data, alpha=-self.lr*self.wd)

def warm(steps, seed):
    torch.manual_seed(seed)
    opt=torch.optim.AdamW(model.parameters(), lr=LR*5, betas=(0.9,0.95), weight_decay=0.1)
    for _ in range(steps):
        model.train(); x,y=get_batch(); _,l=model(x,y)
        opt.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    b2=0.95; st=opt.state; eps=1e-8; Dfroz={}
    for name,p in model.named_parameters():
        if p in st and "exp_avg_sq" in st[p]:
            step=st[p]["step"]; step=float(step.item()) if torch.is_tensor(step) else float(step)
            vhat=st[p]["exp_avg_sq"]/(1-b2**max(step,1)); Dfroz[name]=(1.0/(vhat.sqrt()+eps)).clone()
    return copy.deepcopy(model.state_dict()), Dfroz

def cont(branch, Dfroz, froz_groups, post, dorder_seed):
    model.load_state_dict(copy.deepcopy(branch)); torch.manual_seed(dorder_seed)
    named=list(model.named_parameters())
    fp=[p for n,p in named if group_of(n) in froz_groups]
    fD=[Dfroz[n] for n,p in named if group_of(n) in froz_groups]
    kp=[p for n,p in named if group_of(n) not in froz_groups]
    oa=torch.optim.AdamW(kp, lr=LR*5, betas=(0.9,0.95), weight_decay=0.1) if kp else None
    of=FrozenD(fp,fD,lr=LR*5) if fp else None
    for _ in range(post):
        model.train(); x,y=get_batch(); _,l=model(x,y)
        if oa: oa.zero_grad()
        if of: of.zero_grad()
        l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        if oa: oa.step()
        if of: of.step()
    return eval_val(model, n=16)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--warm", type=int, default=110)
    ap.add_argument("--post", type=int, default=100)
    a=ap.parse_args()
    print("="*72); print(f"  FROZEN-Q/K vs NOISE FLOOR  ({a.seeds} seeds, paired)"); print("="*72)
    print(f"  {'seed':>4}{'base':>9}{'frozen':>9}{'base2':>9}"
          f"{'froz-Δ':>9}{'noise-Δ':>9}")
    print("  "+"-"*54)
    fd, nd = [], []
    for s in range(a.seeds):
        branch, Dfroz = warm(a.warm, seed=100+s)
        vb  = cont(branch, Dfroz, set(),          a.post, dorder_seed=200+s)
        vf  = cont(branch, Dfroz, {"W_Q","W_K"},  a.post, dorder_seed=200+s)
        vb2 = cont(branch, Dfroz, set(),          a.post, dorder_seed=999+s)
        fdelta=100*(vf-vb)/vb; ndelta=100*(vb2-vb)/vb
        fd.append(fdelta); nd.append(ndelta)
        print(f"  {s:>4}{vb:>9.4f}{vf:>9.4f}{vb2:>9.4f}{fdelta:>+8.1f}%{ndelta:>+8.1f}%")
    fd=np.array(fd); nd=np.array(nd)
    print("  "+"-"*54)
    print(f"  frozen-Q/K delta : {fd.mean():+.1f}% ± {fd.std():.1f}")
    print(f"  noise floor      : {np.abs(nd).mean():.1f}% ± {nd.std():.1f}  (baseline reorder)")
    verdict = ("FREE within noise" if abs(fd.mean()) < 2*max(np.abs(nd).mean(),1e-9)
               else "REAL cost above noise")
    print(f"\n  VERDICT: frozen-Q/K is {verdict}.")
    print("  If FREE: cull Q/K's v-update too -> live-Adam footprint past the 18%")
    print("  V/O ceiling.  If REAL: 18% (V/O scalar-SGD) is the honest hard ceiling.")

if __name__=="__main__": main()
