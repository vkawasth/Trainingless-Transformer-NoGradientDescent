"""
STRANDS: root-to-leaf paths through the refinement ladder.
 fwd ladder  L0..L5 = 16,32,64,256,512,1024  (branchings 2,2,4,2,2)
 bwd ladder  L0..L5 = 1024,512,256,64,32,16  (mirrored; fine end at L0)
Strand vector = concatenated (alpha,beta) along the path, z-scored per layer -> R^12.
Question: does distance in strand space track the ultrametric of the tree?
Controls: 3-adic tree on the index (a tree that is NOT the refinement),
          2-adic tree on the index, and plain |i-j|.
"""
import pickle, itertools, numpy as np
from scipy.stats import spearmanr
SH=pickle.load(open("sheaf2d.pkl","rb"))
LAY=["L0","L1","L2","L3","L4","L5"]
FWD=[16,32,64,256,512,1024]; BWD=[1024,512,256,64,32,16]

def strand_matrix(gk, side):
    res=SH[(gk,side)][0]
    sizes=FWD if side=="fwd" else BWD
    order=list(zip(LAY,sizes))
    fine=max(sizes); L_fine=[L for L,n in order if n==fine][0]
    cols=[]
    for L,n in order:
        if L not in res: return None,None
        X=res[L]["X"].copy()
        X=(X-X.mean(0))/ (X.std(0)+1e-30)             # z-score per layer
        idx=(np.arange(fine)*n)//fine                  # ancestor index of each leaf
        cols.append(X[idx])
    return np.concatenate(cols,1), sizes

def lca_depth_true(i,j,sizes):
    """number of layers (from root) at which two leaves agree"""
    fine=max(sizes); order=sizes if sizes[0]<sizes[-1] else sizes[::-1]
    d=0
    for n in order:
        if (i*n)//fine == (j*n)//fine: d+=1
        else: break
    return d

def padic_depth(i,j,p,ndig):
    d=0
    for k in range(ndig-1,-1,-1):
        if (i//p**k)%p == (j//p**k)%p: d+=1
        else: break
    return d

for gk in ("FF","LN"):
    for side in ("fwd","bwd"):
        S,sizes=strand_matrix(gk,side)
        if S is None: continue
        N=S.shape[0]
        sub=np.arange(0,N,1) if N<=1024 else np.arange(1024)
        pairs=np.array(list(itertools.combinations(range(0,N,2),2)))   # subsample for speed
        i,j=pairs[:,0],pairs[:,1]
        ds=np.linalg.norm(S[i]-S[j],axis=1)
        srt=sizes if sizes[0]<sizes[-1] else sizes[::-1]
        kt=np.array([lca_depth_true(a,b,sizes) for a,b in pairs])
        d2=2.0**(-kt); d3=3.0**(-kt)
        k2=np.array([padic_depth(a,b,2,10) for a,b in pairs]); dd2=2.0**(-k2)
        k3=np.array([padic_depth(a,b,3,7)  for a,b in pairs]); dd3=3.0**(-k3)
        dl=np.abs(i-j).astype(float)
        print("="*76); print(f"  {gk} / {side}   strands = {N}, dim = {S.shape[1]}, pairs = {len(pairs):,}")
        print("="*76)
        print(f"  {'metric':>34}{'Pearson':>10}{'Spearman':>11}")
        for nm,dm in [("TRUE tree, 2-adic  2^-lca",d2),
                      ("TRUE tree, 3-adic  3^-lca",d3),
                      ("control: 2-adic on index",dd2),
                      ("control: 3-adic on index",dd3),
                      ("control: |i-j|",dl)]:
            print(f"  {nm:>34}{np.corrcoef(ds,dm)[0,1]:>10.3f}{spearmanr(ds,dm).statistic:>11.3f}")
        print(f"\n  mean strand distance by lca depth (0 = different root):")
        print(f"  {'lca depth':>12}{'n pairs':>10}{'mean ||ds||':>14}{'sd':>9}")
        for d in range(0,7):
            m=kt==d
            if m.sum()>3:
                print(f"  {d:>12}{int(m.sum()):>10}{ds[m].mean():>14.3f}{ds[m].std():>9.3f}")
        deep=ds[kt>=4]; shal=ds[kt<=1]
        if len(deep)>3 and len(shal)>3:
            print(f"\n  deep-ancestor pairs (lca>=4): {deep.mean():.3f}   "
                  f"shallow (lca<=1): {shal.mean():.3f}   ratio {shal.mean()/deep.mean():.3f}")
