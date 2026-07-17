# End-to-end scheduler on the fast stand-in model (validated components run on
# the real 4.3M model separately). Shows switch + stop firing and savings.
import math, copy
from collections import defaultdict, deque
import numpy as np, torch
from step_utility import LM, Corpus, group_of, block_profile, LR

GROUPS=["Emb","W_Q","W_K","W_V","W_O","FF","LayerNorm"]

class UnifiedOpt:
    def __init__(self, named, lr, b1=0.9,b2=0.95,wd=0.1,eps=1e-8):
        self.named=named; self.lr=lr; self.b1=b1; self.b2=b2; self.wd=wd; self.eps=eps
        self.S={p:{"m":torch.zeros_like(p),"v":torch.zeros_like(p),"t":0,"mode":"adam","slr":None,"D":None} for _,p in named}
    def zero_grad(self):
        for _,p in self.named:
            if p.grad is not None: p.grad=None
    def migrate(self, groups, mode):
        for name,p in self.named:
            if group_of(name) in groups and self.S[p]["mode"]=="adam":
                s=self.S[p]; vhat=s["v"]/(1-self.b2**max(s["t"],1)); D=1.0/(vhat.sqrt()+self.eps)
                if mode=="sgd": s["slr"]=self.lr*float(D.median()); s["mode"]="sgd"
                else: s["D"]=D.clone(); s["mode"]="frozen"
    def live(self): return sum(p.numel() for _,p in self.named if self.S[p]["mode"]=="adam")
    def step(self):
        for _,p in self.named:
            if p.grad is None: continue
            s=self.S[p]; g=p.grad; s["t"]+=1
            s["m"].mul_(self.b1).add_(g,alpha=1-self.b1); mhat=s["m"]/(1-self.b1**s["t"])
            if s["mode"]=="adam":
                s["v"].mul_(self.b2).addcmul_(g,g,value=1-self.b2); vhat=s["v"]/(1-self.b2**s["t"])
                p.data.add_(mhat/(vhat.sqrt()+self.eps),alpha=-self.lr)
            elif s["mode"]=="sgd": p.data.add_(mhat,alpha=-s["slr"])
            else: p.data.add_(mhat*s["D"],alpha=-self.lr)
            p.data.add_(p.data,alpha=-self.lr*self.wd)

def run(model, corpus, scheduled, N=300, thr_switch=0.03, thr_stop=0.013):
    named=list(model.named_parameters()); P=sum(p.numel() for _,p in named)
    opt=UnifiedOpt(named, LR*5); ph=deque(maxlen=20); dr=deque(maxlen=15)
    switched=False; stop_run=0; log=[]
    for s in range(1,N+1):
        model.train(); x,y=corpus.get_batch(); _,l=model(x,y)
        opt.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        ph.append(block_profile(model))
        if scheduled and len(ph)==20:
            dr.append(float(np.linalg.norm(ph[-1]-ph[0]))); md=np.mean(dr) if len(dr)==15 else 1.0
            if not switched and md<thr_switch:
                opt.migrate({"W_V","W_O"},"sgd"); switched=True; log.append((s,"V/O->SGD",md))
            if switched and md<thr_stop:
                stop_run+=1
                if stop_run>=25: log.append((s,"early-stop",md)); return corpus.eval_val(model),s,log,100*opt.live()/P
            else: stop_run=0
    return corpus.eval_val(model), N, log, 100*opt.live()/P

def main():
    print("="*68); print("  END-TO-END SCHEDULER (stand-in) : switch + stop + savings"); print("="*68)
    torch.manual_seed(99); mb=LM(); torch.manual_seed(99); ms=LM()
    vb,nb,_,_=run(mb, Corpus(seed=3), scheduled=False)
    vs,ns,log,live=run(ms, Corpus(seed=3), scheduled=True)
    print(f"  baseline AdamW : val {vb:.4f}  steps {nb}  100% live-Adam")
    print(f"  scheduler      : val {vs:.4f}  steps {ns}  {live:.0f}% live-Adam")
    for st,ev,md in log: print(f"      step {st}: {ev} (drift {md:.3f})")
    print(f"\n  culled {100*(nb-ns)/nb:.0f}% of steps, dropped live-Adam to {live:.0f}%, "
          f"val {100*(vs-vb)/vb:+.1f}% vs baseline")

if __name__=="__main__": main()
