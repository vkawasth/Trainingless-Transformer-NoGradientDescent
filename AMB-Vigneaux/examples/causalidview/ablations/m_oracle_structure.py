import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../..'))
exec(open('/tmp/claude-0/-home-claude/bc7a5435-f2c5-5a55-9ebf-abe395f9a702/scratchpad/helpers.py').read())
S=r.S
res=[]
for seed in range(8):
    W=r.world(seed); c,q=W['ctx'],W['qry']; X=W['X']; T=W['T'][c].astype(float); Y=W['Y'][c].astype(float); I=W['I'][c].astype(float)
    qm=W['q'].mean(1); s_or=qm[:,1,1]+qm[:,0,0]; rm=lambda s: np.sqrt(np.mean((s-s_or)**2)); nq=len(q)
    fam=S.CATE_FAMILIES[seed%5]; xc=S._rng(seed,"X_calibration").normal(size=(r.NCAL,r.D)); cp=S._cate_parameters(seed,r.D,fam); rc=S._raw_cate(xc,fam,cp)
    qf=lambda Z: (S._raw_cate(Z,fam,cp)-rc.mean())/rc.std()
    rng=S._rng(seed,"nuisance_parameters"); mu_w=S._unit_vector(rng,r.D,14); vT=S._unit_vector(rng,r.D,14)
    fb=lambda Z: 0.7*(Z@mu_w)+0.25*np.sin(Z@np.roll(mu_w,1))
    def feats(Z):
        a,bq,cc=Z@vT,fb(Z),np.tanh(qf(Z))
        return np.column_stack([np.ones(len(Z)),a,np.tanh(bq+0.8),np.tanh(bq-0.8),cc,a*cc,a*np.tanh(bq+0.8),a*np.tanh(bq-0.8),a**2])
    yb=(Y==T).astype(float)
    b,l=cv_fit(feats(X[c]),yb,lams=(0.01,0.1,1,3,10,30)); s=sigm(feats(X[q])@b)
    res.append(rm(s)); print(seed,fam[:10],'oracle-structure features: M %.3f'%rm(s),flush=True)
print('mean',np.mean(res))
