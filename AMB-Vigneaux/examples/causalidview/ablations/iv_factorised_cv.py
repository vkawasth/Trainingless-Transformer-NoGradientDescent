import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../..'))
exec(open('/tmp/claude-0/-home-claude/bc7a5435-f2c5-5a55-9ebf-abe395f9a702/scratchpad/proto.py').read().split('res=[]')[0])
LAMS=(1,3,10,30,100,300,1000,3000)
def fitb(Z,y,Zqs):
    b,l=cv_fit(Z,y,lams=LAMS); return [sigm(Zq@b) for Zq in Zqs]
res=[]
for seed in range(8):
    W=r.world(seed); c,q=W['ctx'],W['qry']; X=W['X']; T=W['T'][c].astype(float); Y=W['Y'][c].astype(float); I=W['I'][c].astype(float)
    nq=len(q); one=lambda n: np.ones((n,1))
    ivo=np.array([r.iv_bounds(W['q'][i]) for i in range(nq)])
    # baseline: 4-class IRLS with I
    f=r.fit_irls(np.hstack([X[c],I[:,None]]),(2*T+Y).astype(int)); pj0=np.stack([f(np.hstack([X[q],np.full((nq,1),j)])) for j in (0,1)],1).reshape(-1,2,2,2)
    # factorised, CV ridge: treatment P(T=1|X,I), outcome P(Y=1|T=t,X,I) per arm
    Zc=np.hstack([one(len(c)),X[c],I[:,None]]); Zq=[np.hstack([one(nq),X[q],np.full((nq,1),j)]) for j in (0,1)]
    e=fitb(Zc,T,Zq)
    mu={t: fitb(Zc[T==t],Y[T==t],Zq) for t in (0,1)}
    pj=np.zeros((nq,2,2,2))
    for j in (0,1):
        for t in (0,1):
            pt=e[j] if t else 1-e[j]
            pj[:,j,t,1]=pt*mu[t][j]; pj[:,j,t,0]=pt*(1-mu[t][j])
    # direct 4-class per (j) with CV ridge: via two binary chains is what we did; also Manski s
    out=[]
    for P in (pj0,pj):
        iv=np.array([r.iv_bounds(P[i]) for i in range(nq)])
        out.append(np.sqrt(0.5*np.mean((iv[:,0]-ivo[:,0])**2+(iv[:,1]-ivo[:,1])**2)))
        out.append(np.mean(iv[:,2]>1e-7))
    res.append(out); print(seed,W['family'][:10],'baseline IV %.3f (infeas %.2f) | factorised CV %.3f (infeas %.2f)'%tuple(out),flush=True)
print('mean',np.round(np.mean(res,0),4))
