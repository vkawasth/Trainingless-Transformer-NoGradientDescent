"""
OCCUPANCY CONTROL FOR THE LN/bwd STRAND RESIDUAL.
Question: is LN/bwd's hierarchical residual (partial corr 0.196 after removing
log|i-j|) a property of LayerNorm, or of having ~1 element per tile?
Control: FF1024 = a 1024-element slice of FF tiled with the identical ladder.
Uses the stalks already collected by gamma.py (seeds 17 and 23).
"""
import pickle, itertools, sys, numpy as np
from scipy.stats import spearmanr
LAY=["L0","L1","L2","L3","L4","L5"]; BWD=[1024,512,256,64,32,16]
rng=np.random.default_rng(11)
def lca(i,j,sizes):
    fine=max(sizes); order=sizes if sizes[0]<sizes[-1] else sizes[::-1]; d=0
    for n in order:
        if (i*n)//fine==(j*n)//fine: d+=1
        else: break
    return d
def build(X,g):
    ns=[X[g][L].shape[0] for L in LAY]
    fine=max(ns); cols=[]
    for L,n in zip(LAY,ns):
        A=X[g][L].copy(); A=(A-A.mean(0))/(A.std(0)+1e-30)
        idx=np.clip((np.arange(fine)*n)//fine,0,n-1)
        cols.append(A[idx])
    return np.concatenate(cols,1), ns
print("="*86)
print("  DOES LN's STRAND HIERARCHY SURVIVE THE OCCUPANCY CONTROL?")
print("="*86)
print("  FF1024 has identical tile occupancy to LN (1 element/tile at L0), same ladder.")
for seed in (17,23):
    try: X=pickle.load(open(f"stalks_{seed}.pkl","rb"))["X"]
    except FileNotFoundError: print(f"  seed {seed}: stalks missing"); continue
    print(f"\n  --- seed {seed}")
    print(f"  {'group':>9}{'el/tile':>9}{'tree rho':>10}{'|i-j| rho':>11}"
          f"{'PERMUTED':>10}{'partial(tree|i-j)':>19}{'shallow/deep':>14}")
    for g in ["LN","LN_w","LN_b","FF","FF1024","W_V"]:
        if g not in X: continue
        S,ns=build(X,g); N=S.shape[0]
        pairs=np.array(list(itertools.combinations(range(0,N,2),2)))
        i,j=pairs[:,0],pairs[:,1]
        ds=np.linalg.norm(S[i]-S[j],axis=1)
        kt=np.array([lca(a,b,ns) for a,b in pairs]); t=2.0**(-kt)
        perm=rng.permutation(N)
        kp=np.array([lca(perm[a],perm[b],ns) for a,b in pairs]); tp=2.0**(-kp)
        dl=np.abs(i-j).astype(float); x=np.log1p(dl)
        b=np.polyfit(x,ds,1); resid=ds-np.polyval(b,x)
        deep=ds[kt>=4]; shal=ds[kt<=1]
        ratio=shal.mean()/deep.mean() if len(deep)>3 else float('nan')
        ept={"LN":1.0,"LN_w":0.5,"LN_b":0.5,"FF":384.0,"FF1024":1.0,"W_V":64.0}[g]
        print(f"  {g:>9}{ept:>9.1f}{spearmanr(ds,t).statistic:>10.3f}"
              f"{spearmanr(ds,dl).statistic:>11.3f}{spearmanr(ds,tp).statistic:>10.3f}"
              f"{spearmanr(resid,t).statistic:>19.3f}{ratio:>14.3f}", flush=True)
print("\n  READING: if FF1024 reproduces LN's partial correlation, the residual is")
print("  occupancy. If FF1024 stays near FF, the hierarchy is a LayerNorm property.")
