"""
FRONTIER DYNAMICS.
 (1) velocity V(t) = |A(t+1) \ A(t)| : coords entering the frontier per step.
     correlate with loss and gradient variance; does it decay over training?
 (2) activity autocorrelation C(k)=corr(A(t),A(t+k)) -> persistent latent structure?
 (3) interaction graph: is the flip-coupling graph sparse/modular? Measure the
     degree distribution and whether Laplacian eigenvectors beat SVD at sign
     preservation on one matrix (linear-basis comparison, honestly labelled).
"""
import time, gc, numpy as np, torch
t0=time.time()
g_={}; src=open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
exec(src[:src.find("# ── PHASE 1")], g_)
model=g_["model"]; get_batch=g_["get_batch"]; LR=g_["LR"]
named=list(model.named_parameters())
def flat(): return torch.cat([p.data.flatten() for _,p in named]).clone()
def gradv(): return torch.cat([(p.grad.flatten() if p.grad is not None else torch.zeros(p.numel())) for _,p in named]).clone()
def st(o,k):
    out=[]
    for _,p in named:
        s=o.state.get(p,{}); out.append(s[k].flatten() if k in s else torch.zeros(p.numel()))
    return torch.cat(out)
P=flat().numel(); FRAC=0.106
model.load_state_dict(torch.load("init.pt")); torch.manual_seed(17)
o=torch.optim.AdamW(model.parameters(), lr=LR*5, betas=(0.9,0.95), weight_decay=0.1)
A=[]; LOSS=[]; GV=[]; prev=flat()
SUB=torch.randperm(P)[:300000]
for s in range(1,161):
    model.train(); x,y=get_batch(); _,l=model(x,y)
    o.zero_grad(); l.backward(); LOSS.append(float(l))
    g=gradv(); GV.append(float(g.var()))
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); o.step()
    r=(st(o,"exp_avg").abs()/(st(o,"exp_avg_sq").sqrt()+1e-12))
    thr=torch.quantile(r[torch.randperm(P)[:200000]],FRAC)
    A.append((r[SUB]<thr).numpy().astype(np.int8))
    del r,g
    if s%40==0: gc.collect(); print(f"    step {s} ({time.time()-t0:.0f}s)", flush=True)
A=np.stack(A); LOSS=np.array(LOSS); GV=np.array(GV)
print("="*72); print("  (1) FRONTIER VELOCITY V(t) = coords entering per step"); print("="*72)
V=np.array([np.sum((A[t+1]==1)&(A[t]==0)) for t in range(len(A)-1)])/len(SUB)
print(f"  mean turnover per step: {100*V.mean():.2f}% of the subset")
print(f"  V early (10-40): {100*V[10:40].mean():.2f}%   V late (120-159): {100*V[120:].mean():.2f}%")
print(f"  corr(V, loss)          = {np.corrcoef(V,LOSS[1:len(V)+1])[0,1]:+.3f}")
print(f"  corr(V, grad variance) = {np.corrcoef(V,GV[1:len(V)+1])[0,1]:+.3f}")
print(f"  corr(V, |dloss|)       = {np.corrcoef(V,np.abs(np.diff(LOSS))[:len(V)])[0,1]:+.3f}")
print("\n"+"="*72); print("  (2) ACTIVITY AUTOCORRELATION C(k)"); print("="*72)
Am=A-A.mean(0)
def Ck(k): 
    num=np.mean([np.corrcoef(A[t],A[t+k])[0,1] for t in range(20,len(A)-k,3)])
    return num
for k in (1,5,10,20,40,80): print(f"  C({k:>2}) = {Ck(k):+.3f}")
print("  approaches a positive constant => persistent latent activity structure.")
# per-coordinate activation frequency: is it bimodal (core vs churn)?
freq=A.mean(0)
print(f"\n  activation-frequency distribution over coords:")
for lo,hi in [(0,0.05),(0.05,0.2),(0.2,0.5),(0.5,0.9),(0.9,1.0)]:
    print(f"    freq [{lo:.2f},{hi:.2f}): {100*np.mean((freq>=lo)&(freq<hi)):.1f}% of coords")
print("\n"+"="*72); print("  (3) INTERACTION GRAPH: LAPLACIAN vs SVD basis (one matrix)"); print("="*72)
# build flip-coupling on a small matrix, compare Laplacian-eigvec truncation to SVD
import re
off=0; SPAN={}
for n,p in named: SPAN[n]=(off,off+p.numel()); off+=p.numel()
a0,_=SPAN["blocks.2.ff.g.weight"]; R,C=512,256
model.train(); x,y=get_batch(); _,l=model(x,y); o.zero_grad(); l.backward()
G=dict(named)["blocks.2.ff.g.weight"].grad.detach()
r=max(1,int(min(R,C)*0.25))
U,S,Vt=torch.linalg.svd(G,full_matrices=False)
Gsvd=(U[:,:r]*S[:r])@Vt[:r]
ag_svd=float((torch.sign(Gsvd)==torch.sign(G)).float().mean())
# graph basis: coupling = correlation of flip history across the 256 columns
flips=(A[:, :C] if C<=A.shape[1] else A)  # proxy, columns as nodes not exact -- report caveat
print(f"  SVD rank-25% sign agreement (reference): {100*ag_svd:.1f}%")
print(f"  (graph-Laplacian basis is still a linear transform; expected to at best")
print(f"   modestly beat SVD, not change the verdict. Full test needs the coupling")
print(f"   graph estimated over training, deferred.)")
print(f"\n  time {time.time()-t0:.0f}s")
