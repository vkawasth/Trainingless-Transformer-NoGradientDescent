import math
from collections import defaultdict, deque
import numpy as np, torch

#g_={}; src=open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
g_={}; src=open("compiler_geometri_patched_86.py").read()
exec(src[:src.find("# ── PHASE 1")], g_)
model=g_["model"]; get_batch=g_["get_batch"]; eval_val=g_["eval_val"]; LR=g_["LR"]
GROUPS=["Emb","W_Q","W_K","W_V","W_O","FF","LayerNorm"]; ATTN={"W_Q","W_K","W_V","W_O"}

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

def masks():
    idx=0; segs=defaultdict(list)
    for name,p in model.named_parameters():
        segs[group_of(name)].append((idx,idx+p.numel())); idx+=p.numel()
    P=idx; out={}
    for gp in GROUPS:
        m=torch.zeros(P,dtype=torch.bool)
        for a,b in segs.get(gp,[]): m[a:b]=True
        out[gp]=m
    return out,P

def attn_ff_coupling(probes=3):
    x,y=get_batch(); _,l=model(x,y)
    params=[p for _,p in model.named_parameters()]
    grads=torch.autograd.grad(l,params,create_graph=True)
    gflat=torch.cat([g.reshape(-1) for g in grads])
    M,P=masks()
    # ||g_attn|| / ||g_ff||
    ga=torch.cat([gflat[M[a]] for a in ATTN]).norm().item()
    gf=gflat[M["FF"]].norm().item()
    # cross-Hessian attn<->FF via HVP: v on FF, measure (Hv) on attn
    gen=torch.Generator().manual_seed(0); acc=0.0
    for _ in range(probes):
        v=torch.zeros(P); vb=torch.randn(int(M["FF"].sum()),generator=gen); vb/=vb.norm()+1e-12
        v[M["FF"]]=vb
        gv=(gflat*v).sum()
        Hv=torch.autograd.grad(gv,params,retain_graph=True)
        Hv=torch.cat([h.reshape(-1) for h in Hv])
        acc+=torch.cat([Hv[M[a]] for a in ATTN]).norm().item()
    return acc/probes, ga/(gf+1e-12)

def main():
    opt=torch.optim.AdamW(model.parameters(), lr=LR*5, betas=(0.9,0.95), weight_decay=0.1)
    print("="*70); print("  ATTENTION RECRUITMENT ON THE REAL (looped) CORPUS"); print("="*70)
    print(f"  {'step':>5}{'val':>9}{'attn-FF coupling':>18}{'r=|gA|/|gF|':>13}")
    print("  "+"-"*54)
    ckpts=[5,20,50,90,140,200,260]; step=0
    for tgt in ckpts:
        while step<tgt:
            model.train(); x,y=get_batch(); _,l=model(x,y)
            opt.zero_grad(); l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); step+=1
        coup,r=attn_ff_coupling()
        v=eval_val(model,n=6)
        print(f"  {step:>5}{v:>9.4f}{coup:>18.4f}{r:>13.3f}")
    print("\n  If coupling/r RISE (vs the markov stand-in where they fell), attention")
    print("  is recruited by the looped corpus -- induction heads. The original")
    print("  'FF-first, then densify' intuition holds on the task that needs it.")

if __name__=="__main__": main()
