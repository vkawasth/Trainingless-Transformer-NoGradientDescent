import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../..'))
exec(open('/tmp/claude-0/-home-claude/bc7a5435-f2c5-5a55-9ebf-abe395f9a702/scratchpad/proto.py').read().split('res=[]')[0])
def krr_loo(K, y, lams):
    """kernel ridge with closed-form leave-one-out (y centred); returns best lam and alpha"""
    ev,V=np.linalg.eigh(K); yt=V.T@y; best=None
    for lam in lams:
        d=ev/(ev+lam); H_diag=(V**2)@d; fit=V@(d*yt); loo=(y-fit)/(1-H_diag)
        e=np.mean(loo**2)
        if best is None or e<best[0]: best=(e,lam)
    lam=best[1]; alpha=V@(yt/(ev+lam)); return lam, alpha, best[0]
def rbf(A,B,ls): 
    d=(A**2).sum(1)[:,None]+(B**2).sum(1)[None,:]-2*A@B.T; return np.exp(-d/(2*ls*ls))
res=[]
for seed in range(8):
    W=r.world(seed); c,q=W['ctx'],W['qry']; X=W['X']; T=W['T'][c].astype(float); Y=W['Y'][c].astype(float)
    qm=W['q'].mean(1); s_or=qm[:,1,1]+qm[:,0,0]; rm=lambda s: np.sqrt(np.mean((s-s_or)**2))
    yb=(Y==T).astype(float); m=yb.mean(); yc=yb-m
    out=[]
    best=None
    for name,kf in [('lin',lambda A,B: A@B.T),('poly2',lambda A,B: (1+A@B.T/50)**2)]+[(f'rbf{ls}',(lambda ls: (lambda A,B: rbf(A,B,ls)))(ls)) for ls in (5,7,10,14,20)]:
        K=kf(X[c],X[c]); lam,alpha,loo=krr_loo(K,yc,(1,3,10,30,100,300,1000,3000))
        s=np.clip(m+kf(X[q],X[c])@alpha,0,1); out.append((name,rm(s),loo))
        if best is None or loo<best[2]: best=(name,rm(s),loo)
    res.append([o[1] for o in out]+[best[1]])
    print(seed,W['family'][:10],' '.join(f"{o[0]}:{o[1]:.3f}" for o in out),'| LOO-selected',best[0],round(best[1],3),flush=True)
print('mean',np.round(np.mean(res,0),4))
