"""
IS THE BATCH->STRAND MAP LOW INTRINSIC DIMENSION (curved manifold) or high-dim?
Freeze theta. For many batches: batch embedding z_B (mean token embedding +
token histogram) -> strand s_B = sign(grad). 
 (1) linear predict s_B from z_B  (baseline; PR said this is limited)
 (2) nonlinear (RBF/random-feature) predict s_B from z_B
 (3) intrinsic dim of the batch->strand map: does strand similarity follow a
     low-dim function of batch distance? (kNN-regression sign agreement vs k)
If nonlinear >> linear and kNN predicts well, the map is a low-dim curved manifold.
If nonlinear ~ linear ~ chance, the strand is high-dim in the batch too.
"""
import time, gc, numpy as np, torch
t0=time.time()
g_={}; src=open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
exec(src[:src.find("# ── PHASE 1")], g_)
model=g_["model"]; get_batch=g_["get_batch"]; LR=g_["LR"]
named=list(model.named_parameters())
def flat(): return torch.cat([p.data.flatten() for _,p in named]).clone()
def gradv(): return torch.cat([(p.grad.flatten() if p.grad is not None else torch.zeros(p.numel())) for _,p in named]).clone()
def newopt(): return torch.optim.AdamW(model.parameters(), lr=LR*5, betas=(0.9,0.95), weight_decay=0.1)
P=flat().numel(); SUB=torch.randperm(P)[:40000]
VOC=int(model.te.weight.shape[0])
model.load_state_dict(torch.load("init.pt")); torch.manual_seed(17); o=newopt()
for _ in range(80):
    x,y=get_batch(); _,l=model(x,y); o.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); o.step()
theta=flat()
NB=200; Z=[]; S=[]
emb=model.te.weight.detach()
for _ in range(NB):
    x,y=get_batch()
    model.zero_grad(); _,l=model(x,y); l.backward()
    S.append(torch.sign(gradv()[SUB]).numpy().astype(np.int8))
    # batch embedding: mean token embedding + normalized token histogram (top dims)
    hist=torch.bincount(x.flatten(),minlength=VOC).float(); hist/=hist.sum()
    ze=emb[x.flatten()].mean(0)
    Z.append(torch.cat([ze, hist]).numpy())
    with torch.no_grad():
        i=0
        for _,p in named:
            k=p.numel(); p.data.copy_(theta[i:i+k].view_as(p)); i+=k
Z=np.stack(Z); S=np.stack(S).astype(float)
Z=(Z-Z.mean(0))/(Z.std(0)+1e-9)
h=NB*3//4
from numpy.linalg import lstsq
def agree(pred,true): return float((np.sign(pred)==np.sign(true))[true!=0].mean())
# (1) linear z->s
W=lstsq(np.hstack([Z[:h],np.ones((h,1))]),S[:h],rcond=None)[0]
pl=np.hstack([Z[h:],np.ones((NB-h,1))])@W
print("="*70); print("  BATCH -> STRAND MAP: linear vs nonlinear vs kNN"); print("="*70)
print(f"  strand base rate (sign +): {(S>0).mean():.3f}")
print(f"  (1) linear z->s     : sign agreement {agree(pl,S[h:]):.3f}")
# (2) random-feature nonlinear (RBF approx)
rng=np.random.default_rng(0); D=800
Wr=rng.normal(size=(Z.shape[1],D))*0.5; br=rng.uniform(0,2*np.pi,D)
Phi=np.cos(Z@Wr+br)
Wn=lstsq(np.hstack([Phi[:h],np.ones((h,1))]),S[:h],rcond=None)[0]
pn=np.hstack([Phi[h:],np.ones((NB-h,1))])@Wn
print(f"  (2) nonlinear z->s  : sign agreement {agree(pn,S[h:]):.3f}")
# (3) kNN: predict test strand from nearest training batches
from numpy import argsort
def knn(k):
    ag=[]
    for i in range(h,NB):
        d=((Z[:h]-Z[i])**2).sum(1); nn=argsort(d)[:k]
        pred=np.sign(S[nn].mean(0)); ag.append(agree(pred,S[i]))
    return np.mean(ag)
print(f"  (3) kNN k=1         : sign agreement {knn(1):.3f}")
print(f"      kNN k=5         : sign agreement {knn(5):.3f}")
print(f"      kNN k=15        : sign agreement {knn(15):.3f}")
# intrinsic-dim signal: does strand sim decay smoothly with batch distance?
ds=[]; ss=[]
for i in range(NB):
    for j in range(i+1,NB):
        ds.append(np.sqrt(((Z[i]-Z[j])**2).sum())); ss.append((S[i]==S[j]).mean())
print(f"\n  corr(batch distance, strand dissimilarity) = {np.corrcoef(ds,1-np.array(ss))[0,1]:+.3f}")
print(f"  (strong negative dist->sim => smooth manifold; ~0 => no manifold)")
print("\n  nonlinear >> linear AND kNN high => low-dim curved manifold (compressible).")
print("  nonlinear ~ linear ~ base rate => strand high-dim in batch too (not compressible).")
print(f"\n  time {time.time()-t0:.0f}s")
