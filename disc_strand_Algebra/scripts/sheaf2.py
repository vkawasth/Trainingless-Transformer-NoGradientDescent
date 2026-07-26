import numpy as np
NODES=["EMB","LN","FF","W_Q","W_K","W_V","W_O"]
SK=np.load("SK.npy"); CELLS=[(0,20),(20,40),(40,60),(60,80)]
LAB=["120-140","140-160","160-180","180-200"]
G=np.concatenate([SK[i] for i in range(7)],axis=1)
def bas(a,b,k): 
    u,s,vt=np.linalg.svd(G[a:b],full_matrices=False); return vt[:k].T
print("="*76); print("  HONEST H^0: dim of the subspace common to ALL FOUR cells"); print("="*76)
print("  (edge stalk = ambient; restriction = inclusion; global sections = intersection)")
print(f"\n  {'k':>4}{'sum of proj. eigenvalues near 1':>34}   dim H^0 at cos>0.9 / >0.7 / >0.5")
for k in [3,6,10,20]:
    Ps=[]
    for (a,b) in CELLS:
        B=bas(a,b,k); Ps.append(B@B.T)
    Msum=sum(Ps)/len(Ps)
    ev=np.linalg.eigvalsh(Msum)[::-1][:k+2]
    d9=int((ev>0.9).sum()); d7=int((ev>0.7).sum()); d5=int((ev>0.5).sum())
    print(f"  {k:>4}{'  '.join(f'{x:.3f}' for x in ev[:6]):>34}      {d9}  /  {d7}  /  {d5}")
print("\n  a direction present in all 4 cells gives eigenvalue 1.0 of the mean projector")
print("\n"+"="*76); print("  IS THE SHARED DIRECTION THE CHORD?"); print("="*76)
import torch
seg=torch.load("seg.pt"); DT=seg["DT"]
# sketch the chord the same way
rg=np.random.default_rng(0)
off=0; sl={}
import json
# rebuild index map exactly as exp3
src=open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
g_={}; exec(src[:src.find("# ── PHASE 1")], g_)
named=list(g_["model"].named_parameters())
def grp(n):
    n=n.lower()
    if ".ln." in n or ".n." in n or n.startswith("ln_f"): return "LN"
    if n.startswith("te") or n.startswith("pe") or n.startswith("head"): return "EMB"
    if ".ff." in n: return "FF"
    if "wk" in n: return "W_K"
    if "wq" in n: return "W_Q"
    if "wv" in n: return "W_V"
    if ".op." in n: return "W_O"
    return "other"
sl={g:[] for g in NODES}
for n,p in named: sl[grp(n)].append((off,off+p.numel())); off+=p.numel()
IDX={g: torch.cat([torch.arange(a,b) for a,b in sl[g]]) for g in NODES}
M=512; rg=np.random.default_rng(0)
HSH={g: torch.tensor(rg.integers(0,M,size=len(IDX[g])),dtype=torch.long) for g in NODES}
SGN={g: torch.tensor(rg.choice([-1.0,1.0],size=len(IDX[g])),dtype=torch.float32) for g in NODES}
ch=[]
for g in NODES:
    v=DT[IDX[g]]*SGN[g]; b=torch.zeros(M); b.index_add_(0,HSH[g],v); ch.append(b.numpy())
ch=np.concatenate(ch); ch=ch/np.linalg.norm(ch)
for k in [1,3,6]:
    for (a,b),lab in zip(CELLS,LAB):
        B=bas(a,b,k); print(f"    k={k}  cell {lab}: ||proj of chord|| = {np.linalg.norm(B.T@ch):.3f}", end="")
        print()
    print()
