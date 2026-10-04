import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../..'))
exec(open('/tmp/claude-0/-home-claude/bc7a5435-f2c5-5a55-9ebf-abe395f9a702/scratchpad/helpers.py').read())
from scipy.optimize import nnls
LAMS=(1,3,10,30,100,300,1000,3000)
def lr_pred(Ztr,y,Zte):
    b,l=cv_fit(Ztr,y,lams=LAMS); return sigm(Zte@b)
def models(Xtr,T,Y,Xte):
    one=lambda n: np.ones((n,1)); Zc=np.hstack([one(len(Xtr)),Xtr]); Zq=np.hstack([one(len(Xte)),Xte]); yb=(Y==T).astype(float)
    out={}
    out['lr']=lr_pred(Zc,yb,Zq)
    e=lr_pred(Zc,T,Zq); m1=lr_pred(Zc[T==1],Y[T==1],Zq); m0=lr_pred(Zc[T==0],Y[T==0],Zq); out['fac']=e*m1+(1-e)*(1-m0)
    m=yb.mean()
    for name,kf in (('krr_lin',lambda A,B: A@B.T),('krr_rbf',lambda A,B: rbf(A,B,14)),('krr_poly',lambda A,B: (1+A@B.T/50)**2)):
        K=kf(Xtr,Xtr); lam,alpha,_=krr_loo(K,yb-m,(1,3,10,30,100,300,1000,3000)); out[name]=np.clip(m+kf(Xte,Xtr)@alpha,0,1)
    out['const']=np.full(len(Xte),m)
    return out
res=[]
for seed in range(8):
    W=r.world(seed); c,q=W['ctx'],W['qry']; X=W['X']; T=W['T'][c].astype(float); Y=W['Y'][c].astype(float)
    qm=W['q'].mean(1); s_or=qm[:,1,1]+qm[:,0,0]; rm=lambda s: np.sqrt(np.mean((s-s_or)**2))
    yb=(Y==T).astype(float)
    # out-of-fold predictions for stacking
    rng=np.random.default_rng(seed); f=rng.integers(0,5,len(c)); oof={}
    for k in range(5):
        tr,te=f!=k,f==k; P=models(X[c][tr],T[tr],Y[tr],X[c][te])
        for n,v in P.items(): oof.setdefault(n,np.zeros(len(c)))[te]=v
    names=list(oof); A=np.stack([oof[n] for n in names],1); w,_=nnls(A,yb); w=w/w.sum()
    P=models(X[c],T,Y,X[q]); s_st=np.stack([P[n] for n in names],1)@w
    res.append([rm(P[n]) for n in names]+[rm(s_st)]); print(seed,' '.join(f"{n}:{rm(P[n]):.3f}" for n in names),'| stack %.3f'%rm(s_st),dict(zip(names,np.round(w,2))),flush=True)
print('mean',names+['stack'],np.round(np.mean(res,0),4))
