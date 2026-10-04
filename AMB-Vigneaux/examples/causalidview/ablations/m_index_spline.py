import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../..'))
exec(open('/tmp/claude-0/-home-claude/bc7a5435-f2c5-5a55-9ebf-abe395f9a702/scratchpad/proto.py').read().split('res=[]')[0])
LAMS=(0.3,1,3,10,30,100,300,1000)
def fitp(Z,y,Zq,lams=LAMS):
    b,l=cv_fit(Z,y,lams=lams); return sigm(Zq@b), b, l
def basis(Zi, knots):
    cols=[np.ones((len(Zi),1)), Zi]
    for j in range(Zi.shape[1]):
        for kn in knots[j]: cols.append(np.maximum(Zi[:,[j]]-kn,0)**2)
    for a in range(Zi.shape[1]):
        for b_ in range(a+1,Zi.shape[1]): cols.append(Zi[:,[a]]*Zi[:,[b_]])
    return np.hstack(cols)
res=[]
for seed in range(8):
    W=r.world(seed); c,q=W['ctx'],W['qry']; X=W['X']; T=W['T'][c].astype(float); Y=W['Y'][c].astype(float)
    qm=W['q'].mean(1); s_or=qm[:,1,1]+qm[:,0,0]; rm=lambda s: np.sqrt(np.mean((s-s_or)**2))
    one=lambda n: np.ones((n,1)); Zc=np.hstack([one(len(c)),X[c]]); Zq=np.hstack([one(len(q)),X[q]])
    _,bT,_=fitp(Zc,T,Zq)                                  # treatment index
    ZY=np.hstack([Zc,T[:,None],X[c]*(T[:,None]-0.5)]); _,bY,_=fitp(ZY,Y,np.hstack([Zq,np.zeros((len(q),1)),X[q]*0]))
    dirs=[bT[1:], bY[1:51], bY[52:]]
    Dm=np.stack([d/np.linalg.norm(d) for d in dirs],1)
    Zi=X[c]@Dm; Zqi=X[q]@Dm; sd=Zi.std(0); Zi/=sd; Zqi/=sd
    knots=[np.quantile(Zi[:,j],[0.25,0.5,0.75]) for j in range(Zi.shape[1])]
    Bc=basis(Zi,knots); Bq=basis(Zqi,knots)
    yb=(Y==T).astype(float)
    s_idx,_,l=fitp(Bc,yb,Bq)
    # factorised with index features
    Bc_full=np.hstack([Bc]); e,_,_=fitp(Bc_full,T,Bq); m1,_,_=fitp(Bc[T==1],Y[T==1],Bq); m0,_,_=fitp(Bc[T==0],Y[T==0],Bq)
    s_fac=e*m1+(1-e)*(1-m0)
    s_lin,_,_=fitp(Zc,yb,Zq)
    res.append((rm(s_lin),rm(s_idx),rm(s_fac),rm(0.5*(s_idx+s_fac)))); print(seed,W['family'][:10],[round(v,3) for v in res[-1]],l,flush=True)
print('mean lin, index-direct, index-factorised, blend',np.round(np.mean(res,0),4))
