import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../..'))
exec(open('/tmp/claude-0/-home-claude/bc7a5435-f2c5-5a55-9ebf-abe395f9a702/scratchpad/helpers.py').read())
S=r.S
def unit(v): return v/np.linalg.norm(v)
res=[]
for seed in range(8):
    W=r.world(seed); c,q=W['ctx'],W['qry']; X=W['X']; T=W['T'][c].astype(float); Y=W['Y'][c].astype(float); I=W['I'][c].astype(float)
    qm=W['q'].mean(1); s_or=qm[:,1,1]+qm[:,0,0]; rm=lambda s: np.sqrt(np.mean((s-s_or)**2))
    rng=S._rng(seed,"nuisance_parameters"); mu_w=S._unit_vector(rng,r.D,14); vT=S._unit_vector(rng,r.D,14)
    one=lambda n: np.ones((n,1)); Zc=np.hstack([one(len(c)),X[c]])
    bT,_=cv_fit(np.hstack([Zc,I[:,None]]),T,lams=(1,3,10,30,100)); vh=unit(bT[1:51])
    bY,_=cv_fit(np.hstack([Zc,T[:,None]]),Y,lams=(1,3,10,30,100,300)); mh=unit(bY[1:51])
    cosT=abs(vh@vT); cosM=abs(mh@mu_w)
    def feats(Z, a_dir, m_dir):
        a=Z@a_dir; bq=Z@m_dir
        return np.column_stack([np.ones(len(Z)),a,np.tanh(0.7*bq+0.8),np.tanh(0.7*bq-0.8),a*np.tanh(0.7*bq+0.8),a*np.tanh(0.7*bq-0.8),a**2,bq])
    yb=(Y==T).astype(float)
    b,l=cv_fit(feats(X[c],vh,mh),yb,lams=(0.01,0.1,1,3,10,30)); s_est=sigm(feats(X[q],vh,mh)@b)
    b2,_=cv_fit(feats(X[c],vT,mu_w),yb,lams=(0.01,0.1,1,3,10,30)); s_tru=sigm(feats(X[q],vT,mu_w)@b2)
    res.append((rm(s_est),rm(s_tru),cosT,cosM)); print(seed,'estimated-dirs %.3f | true-dirs(no tau) %.3f | cos vT %.2f mu %.2f'%res[-1],flush=True)
print('mean',np.round(np.mean(res,0),4))
