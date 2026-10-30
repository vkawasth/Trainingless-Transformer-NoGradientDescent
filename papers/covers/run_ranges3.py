"""Inner bounds for the decision ranges of Figure "higher"(c): general three-type explanations of the worked two-type law (m = 5),
by constrained optimisation from 40 random fibre points per cover. Writes lab/ranges3.json. About 15 minutes."""
import sys, itertools, numpy as np, time, json
sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '..', 'engine'))
from pce import threetype as TT
rng=np.random.default_rng(5)
m=5; w=0.6; a1=np.array([.5,.1,.4,.7,.3]); a2=np.array([.05,.5,.1,.2,.6])
pairs=[(i,j) for i,j in itertools.combinations(range(5),2)]; tri=list(itertools.combinations(range(5),3))
X=list(itertools.product((0,1),repeat=5))
ell3=np.array([1.0 if x[0]==x[1]==x[2]==1 else 0.0 for x in X]); ell4=np.array([1.0 if x[0]==x[1]==x[2]==x[3]==1 else 0.0 for x in X])
th0=TT.pack(np.array([w,(1-w)/2,(1-w)/2]), np.vstack([a1,a2,a2]))
out={}
for name,F in [("pairs",pairs),("pairs+012",pairs+[(0,1,2)]),("pairs+triples",pairs+tri)]:
    t=time.time(); pts=TT.fibre_points(th0,m,3,F,40,rng); target=TT.moments_obs(th0,m,F,3)
    res={}
    for en,ell in (("E3",ell3),("E4",ell4)):
        lo=TT.range_from(ell,pts,m,3,F,target,-1); hi=TT.range_from(ell,pts,m,3,F,target,1)
        res[en]=(lo[0] if lo else None, hi[0] if hi else None)
    out[name]=dict(npts=len(pts), **res); print(name, len(pts), res, "%.0fs"%(time.time()-t), flush=True)
json.dump(out,open(''+__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)),'..','lab','ranges3.json')+'','w'))
