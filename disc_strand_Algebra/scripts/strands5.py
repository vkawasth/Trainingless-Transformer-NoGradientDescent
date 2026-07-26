import pickle, itertools, numpy as np
from scipy.stats import spearmanr
SH=pickle.load(open("sheaf2d.pkl","rb"))
LAY=["L0","L1","L2","L3","L4","L5"]; FWD=[16,32,64,256,512,1024]; BWD=[1024,512,256,64,32,16]
def strandM(gk,side):
    res=SH[(gk,side)][0]; sizes=FWD if side=="fwd" else BWD
    fine=max(sizes); cols=[]
    for L,n in zip(LAY,sizes):
        if L not in res: return None,None
        X=res[L]["X"].copy(); X=(X-X.mean(0))/(X.std(0)+1e-30)
        cols.append(X[(np.arange(fine)*n)//fine])
    return np.concatenate(cols,1), sizes
def lca_true(i,j,sizes):
    fine=max(sizes); order=sizes if sizes[0]<sizes[-1] else sizes[::-1]; d=0
    for n in order:
        if (i*n)//fine==(j*n)//fine: d+=1
        else: break
    return d
def padic(i,j,p,nd):
    d=0
    for k in range(nd-1,-1,-1):
        if (i//p**k)%p==(j//p**k)%p: d+=1
        else: break
    return d
rng=np.random.default_rng(11)
print("="*84); print("  5-ADIC, AND THE DECISIVE CONTROL: does the HIERARCHY matter, or only CONTIGUITY?")
print("="*84)
for gk in ("FF","LN"):
    for side in ("fwd","bwd"):
        S,sizes=strandM(gk,side)
        if S is None: continue
        N=S.shape[0]
        pairs=np.array(list(itertools.combinations(range(0,N,2),2)))
        i,j=pairs[:,0],pairs[:,1]
        ds=np.linalg.norm(S[i]-S[j],axis=1)
        kt=np.array([lca_true(a,b,sizes) for a,b in pairs])
        perm=rng.permutation(N)                      # shuffle leaf<->position alignment
        kp=np.array([lca_true(perm[a],perm[b],sizes) for a,b in pairs])
        rows=[("TRUE tree (any p: 2,3,5 identical)",2.0**(-kt)),
              ("2-adic on index",2.0**(-np.array([padic(a,b,2,10) for a,b in pairs]))),
              ("3-adic on index",3.0**(-np.array([padic(a,b,3,7)  for a,b in pairs]))),
              ("5-adic on index",5.0**(-np.array([padic(a,b,5,5)  for a,b in pairs]))),
              ("|i-j| (plain contiguity)",np.abs(i-j).astype(float)),
              ("PERMUTED tree (shape kept, alignment broken)",2.0**(-kp))]
        print(f"\n  {gk}/{side}   N={N}, pairs={len(pairs):,}")
        print(f"  {'metric':>45}{'Pearson':>10}{'Spearman':>11}")
        for nm,dm in rows:
            print(f"  {nm:>45}{np.corrcoef(ds,dm)[0,1]:>10.3f}{spearmanr(ds,dm).statistic:>11.3f}")
        # partial: does the tree add anything beyond |i-j|?
        from numpy.polynomial import polynomial as Pn
        x=np.log1p(np.abs(i-j).astype(float)); y=ds
        b=np.polyfit(x,y,1); resid=y-np.polyval(b,x)
        t=2.0**(-kt)
        print(f"  {'partial corr(tree | after removing log|i-j|)':>45}"
              f"{np.corrcoef(resid,t)[0,1]:>10.3f}{spearmanr(resid,t).statistic:>11.3f}")
