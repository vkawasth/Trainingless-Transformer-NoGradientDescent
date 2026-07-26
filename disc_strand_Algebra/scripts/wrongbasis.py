"""
ARE THE LOW-RANK SIGN ERRORS LOCALIZED (wrong basis) OR SPREAD (intrinsic)?
Truncate each weight-grad matrix to rank r; find coords whose sign flips.
Measure whether flips concentrate by row(neuron), column(input), layer, tensor.
If localized -> a neuron/graph-adapted basis could preserve signs (wrong basis).
If uniform  -> spatial incompressibility is intrinsic.
Also test a NEURON-ADAPTED alternative: per-row (neuron) mean-removal + low rank.
"""
import re, time, gc, numpy as np, torch
t0=time.time()
g_={}; src=open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
exec(src[:src.find("# ── PHASE 1")], g_)
model=g_["model"]; get_batch=g_["get_batch"]; LR=g_["LR"]
named=list(model.named_parameters())
def flat(): return torch.cat([p.data.flatten() for _,p in named]).clone()
def gradv(): return torch.cat([(p.grad.flatten() if p.grad is not None else torch.zeros(p.numel())) for _,p in named]).clone()
def newopt(): return torch.optim.AdamW(model.parameters(), lr=LR*5, betas=(0.9,0.95), weight_decay=0.1)
off=0; SPAN={}
for n,p in named: SPAN[n]=(off,off+p.numel()); off+=p.numel()
model.load_state_dict(torch.load("init.pt")); torch.manual_seed(17); o=newopt()
for _ in range(60):
    model.train(); x,y=get_batch(); _,l=model(x,y)
    o.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); o.step()
# one representative matrix, analyse flip localization
model.train(); x,y=get_batch(); _,l=model(x,y); o.zero_grad(); l.backward()
def analyse(nm):
    p=dict(named)[nm]; G=p.grad.detach()
    R,C=G.shape
    for rf in (0.25,0.1):
        r=max(1,int(min(R,C)*rf))
        U,S,Vt=torch.linalg.svd(G,full_matrices=False)
        Gr=(U[:,:r]*S[:r])@Vt[:r]
        flip=(torch.sign(Gr)!=torch.sign(G))
        # localization: variance of per-row flip rate vs binomial
        pr=flip.float().mean(1); pc=flip.float().mean(0); pb=float(flip.float().mean())
        row_od=float(pr.std()/max(np.sqrt(pb*(1-pb)/C),1e-9))
        col_od=float(pc.std()/max(np.sqrt(pb*(1-pb)/R),1e-9))
        # neuron-adapted alt: remove per-row mean, then same rank
        Gc=G-G.mean(1,keepdim=True)
        U2,S2,V2=torch.linalg.svd(Gc,full_matrices=False)
        Gr2=(U2[:,:r]*S2[:r])@V2[:r]+G.mean(1,keepdim=True)
        agree_std=float((torch.sign(Gr)==torch.sign(G)).float().mean())
        agree_adp=float((torch.sign(Gr2)==torch.sign(G)).float().mean())
        print(f"  {nm:>22} rank{int(100*rf)}%  flip {100*pb:4.1f}%"
              f"  row-OD {row_od:4.1f}x  col-OD {col_od:4.1f}x"
              f"  | global {100*agree_std:.1f}%  neuron-adapted {100*agree_adp:.1f}%")
print("="*96); print("  LOW-RANK SIGN-FLIP LOCALIZATION  (OD=1 uniform, >>1 concentrated)"); print("="*96)
for nm in ["blocks.2.ff.g.weight","blocks.2.attn.WQ.weight","blocks.0.ff.g.weight",
           "blocks.5.ff.g.weight","te.weight"]:
    analyse(nm)
print("\n  row-OD >> 1 => flips concentrate on specific neurons (wrong-basis, fixable).")
print("  row-OD ~ 1 & neuron-adapted ~ global => intrinsic spatial incompressibility.")
# cross-layer: flip rate by tensor at fixed rank
print("\n"+"="*96); print("  FLIP RATE BY TENSOR (rank 25%)"); print("="*96)
for nm in ["blocks.2.ff.g.weight","blocks.2.attn.WQ.weight","blocks.2.attn.WV.weight",
           "blocks.0.ff.g.weight","blocks.5.ff.g.weight","te.weight"]:
    p=dict(named)[nm]; G=p.grad.detach(); R,C=G.shape; r=max(1,int(min(R,C)*0.25))
    U,S,Vt=torch.linalg.svd(G,full_matrices=False); Gr=(U[:,:r]*S[:r])@Vt[:r]
    print(f"  {nm:>22}: {100*float((torch.sign(Gr)!=torch.sign(G)).float().mean()):.1f}% flips")
print(f"\n  time {time.time()-t0:.0f}s")
