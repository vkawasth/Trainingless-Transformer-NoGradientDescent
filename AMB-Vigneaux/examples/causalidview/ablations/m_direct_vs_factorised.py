import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../..'))
exec(open('/tmp/claude-0/-home-claude/bc7a5435-f2c5-5a55-9ebf-abe395f9a702/scratchpad/proto.py').read().split('res=[]')[0])
LAMS=(1,3,10,30,100,300,1000,3000)
def fitp(Z,y,Zq):
    b,l=cv_fit(Z,y,lams=LAMS); return sigm(Zq@b), l
res=[]
for seed in range(8):
    W=r.world(seed); c,q=W['ctx'],W['qry']; X=W['X']; T=W['T'][c].astype(float); Y=W['Y'][c].astype(float); I=W['I'][c].astype(float)
    qm=W['q'].mean(1); s_or=qm[:,1,1]+qm[:,0,0]
    rm=lambda s: np.sqrt(np.mean((s-s_or)**2))
    one=lambda n: np.ones((n,1))
    Zc=np.hstack([one(len(c)),X[c]]); Zq=np.hstack([one(len(q)),X[q]])
    sd,_=fitp(Zc,(Y==T).astype(float),Zq)
    e,le=fitp(Zc,T,Zq)
    m1,l1=fitp(Zc[T==1],Y[T==1],Zq); m0,l0=fitp(Zc[T==0],Y[T==0],Zq)
    sf=e*m1+(1-e)*(1-m0)
    # treatment model with I (marginalised over I at prediction: average j)
    ZcI=np.hstack([Zc,I[:,None]]); eI=0.5*(fitp(ZcI,T,np.hstack([Zq,np.zeros((len(q),1))]))[0]+fitp(ZcI,T,np.hstack([Zq,np.ones((len(q),1))]))[0])
    sfI=eI*m1+(1-eI)*(1-m0)
    # blend direct and factorised
    sb=0.5*(sd+sf)
    res.append((rm(sd),rm(sf),rm(sfI),rm(sb))); print(seed,[round(v,3) for v in res[-1]],le,l1,l0,flush=True)
print('mean direct, factorised, factorised+I, blend',np.round(np.mean(res,0),4))
