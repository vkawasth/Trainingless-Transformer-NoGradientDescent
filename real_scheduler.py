"""
real_scheduler.py
=================
One adaptive scheduler that watches graph drift and applies every cull we
validated, on the real model:
  - all blocks start on live AdamW (graph is being built)
  - when drift settles: migrate V/O to scalar SGD (lr*median D)  [~free]
  - optionally freeze Q/K's second moment (frozen D)             [~8% val]
  - on a sustained low-drift plateau: early-stop                 [cull steps]
Compared against plain AdamW at equal step budget.
"""
import math
from collections import defaultdict, deque
import numpy as np, torch

g_={}; src=open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
exec(src[:src.find("# ── PHASE 1")], g_)
model=g_["model"]; get_batch=g_["get_batch"]; eval_val=g_["eval_val"]; LR=g_["LR"]
GROUPS=["Emb","W_Q","W_K","W_V","W_O","FF","LayerNorm"]

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

def block_profile(named):
    prof=defaultdict(float)
    for name,p in named:
        if p.grad is not None: prof[group_of(name)]+=float(p.grad.pow(2).sum())
    v=np.array([math.sqrt(prof.get(k,0.0)) for k in GROUPS]); return v/(v.sum()+1e-12)

class UnifiedOpt:
    """Hand-rolled AdamW that can switch any param to scalar-SGD or frozen-D
    on the fly -- the substrate a graph scheduler needs."""
    def __init__(self, named, lr, b1=0.9, b2=0.95, wd=0.1, eps=1e-8):
        self.named=named; self.lr=lr; self.b1=b1; self.b2=b2; self.wd=wd; self.eps=eps
        self.S={p:{"m":torch.zeros_like(p),"v":torch.zeros_like(p),"t":0,
                   "mode":"adam","slr":None,"D":None} for _,p in named}
    def zero_grad(self):
        for _,p in self.named:
            if p.grad is not None: p.grad=None
    def migrate(self, groups, mode):
        for name,p in self.named:
            if group_of(name) in groups:
                s=self.S[p]
                vhat=s["v"]/(1-self.b2**max(s["t"],1)); D=1.0/(vhat.sqrt()+self.eps)
                if mode=="sgd": s["slr"]=self.lr*float(D.median()); s["mode"]="sgd"
                elif mode=="frozen": s["D"]=D.clone(); s["mode"]="frozen"
    def n_live_adam(self):
        return sum(p.numel() for _,p in self.named if self.S[p]["mode"]=="adam")
    def step(self):
        for _,p in self.named:
            if p.grad is None: continue
            s=self.S[p]; g=p.grad; s["t"]+=1; b1=self.b1
            s["m"].mul_(b1).add_(g, alpha=1-b1); mhat=s["m"]/(1-b1**s["t"])
            if s["mode"]=="adam":
                s["v"].mul_(self.b2).addcmul_(g,g,value=1-self.b2)
                vhat=s["v"]/(1-self.b2**s["t"]); upd=mhat/(vhat.sqrt()+self.eps)
                p.data.add_(upd, alpha=-self.lr)
            elif s["mode"]=="sgd":
                p.data.add_(mhat, alpha=-s["slr"])
            else:  # frozen D
                p.data.add_(mhat*s["D"], alpha=-self.lr)
            p.data.add_(p.data, alpha=-self.lr*self.wd)

def run_baseline(N):
    opt=UnifiedOpt(list(model.named_parameters()), LR*5)
    for _ in range(N):
        model.train(); x,y=get_batch(); _,l=model(x,y)
        opt.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    return eval_val(model,n=8), N

def run_scheduled(N, thr_switch=0.03, thr_stop=0.013, freeze_qk=False):
    named=list(model.named_parameters()); P=sum(p.numel() for _,p in named)
    opt=UnifiedOpt(named, LR*5)
    ph=deque(maxlen=20); dr=deque(maxlen=15)
    switched=False; log=[]; stop_run=0
    for s in range(1,N+1):
        model.train(); x,y=get_batch(); _,l=model(x,y)
        opt.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        ph.append(block_profile(named))
        if len(ph)==20:
            d=float(np.linalg.norm(ph[-1]-ph[0])); dr.append(d)
            md=np.mean(dr) if len(dr)==15 else 1.0
            if not switched and md<thr_switch:
                opt.migrate({"W_V","W_O"},"sgd")
                if freeze_qk: opt.migrate({"W_Q","W_K"},"frozen")
                switched=True; log.append((s,"cull V/O->SGD"+(" + freeze Q/K" if freeze_qk else ""),md))
            if switched and md<thr_stop:
                stop_run+=1
                if stop_run>=25:
                    log.append((s,"early-stop (plateau)",md)); 
                    return eval_val(model,n=8), s, log, 100*opt.n_live_adam()/P
            else: stop_run=0
    return eval_val(model,n=8), N, log, 100*opt.n_live_adam()/P

def main():
    P=sum(p.numel() for p in model.parameters()); N=150
    print("="*74); print("  ADAPTIVE GRAPH-DRIFT SCHEDULER vs PLAIN ADAMW (real model)"); print("="*74)
    import copy; torch.manual_seed(0)
    base_state=copy.deepcopy(model.state_dict())
    vb,nb=run_baseline(N)
    print(f"  baseline AdamW      : val {vb:.4f}  steps {nb}  100% params live-Adam")
    for fq in [False]:
        model.load_state_dict(copy.deepcopy(base_state))
        vs,ns,log,live=run_scheduled(N, freeze_qk=fq)
        tag="scheduler+freezeQK" if fq else "scheduler        "
        print(f"\n  {tag} : val {vs:.4f}  steps {ns}  {live:.0f}% params live-Adam")
        for st,ev,md in log: print(f"      step {st:>3}: {ev}   (drift {md:.3f})")
        print(f"      -> steps culled {100*(nb-ns)/nb:.0f}%, "
              f"live-Adam footprint {live:.0f}% (was 100%), "
              f"val {100*(vs-vb)/vb:+.1f}% vs baseline")

if __name__=="__main__": main()
