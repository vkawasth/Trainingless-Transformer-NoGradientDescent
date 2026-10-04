import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../..'))
exec(open('/tmp/claude-0/-home-claude/bc7a5435-f2c5-5a55-9ebf-abe395f9a702/scratchpad/helpers.py').read())
S=r.S
def unit(v): return v/np.linalg.norm(v)
res=[]
for seed in range(8):
    W=r.world(seed); c,q=W['ctx'],W['qry']; X=W['X']; T=W['T'][c].astype(float); Y=W['Y'][c].astype(float)
    qm=W['q'].mean(1); s_or=qm[:,1,1]+qm[:,0,0]; rm=lambda s: np.sqrt(np.mean((s-s_or)**2))
    fam=S.CATE_FAMILIES[seed%5]; cp=S._cate_parameters(seed,r.D,fam)
    yb=(Y==T).astype(float); yc=yb-yb.mean(); Xc=X[c]
    # first-moment direction (linear) and principal Hessian directions (pHd)
    d1=unit(Xc.T@yc)
    M=(Xc*yc[:,None]).T@Xc/len(yc); ev,V=np.linalg.eigh(M); o=np.argsort(-np.abs(ev)); D=np.column_stack([d1,V[:,o[0]],V[:,o[1]]])
    D,_=np.linalg.qr(D)
    cos_a=np.abs(D.T@unit(cp['a'])).max(); cos_b=np.abs(D.T@unit(cp['b'])).max()
    Uc=Xc@D; Uq=X[q]@D; m=yb.mean(); best=None
    for k in (1,2,3):
        for ls in (0.5,1,1.5,2,3):
            K=rbf(Uc[:,:k],Uc[:,:k],ls); lam,alpha,loo=krr_loo(K,yc,(1,3,10,30,100,300,1000))
            if best is None or loo<best[0]: best=(loo,k,ls,lam,alpha)
    loo,k,ls,lam,alpha=best; s=np.clip(m+rbf(Uq[:,:k],Uc[:,:k],ls)@alpha,0,1)
    res.append((rm(s),cos_a,cos_b)); print(seed,fam[:10],'pHd+KRR %.3f (k=%d ls=%.1f) | max cos with true a %.2f b %.2f'%(rm(s),k,ls,cos_a,cos_b),flush=True)
print('mean',np.round(np.mean(res,0),4))
