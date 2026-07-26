"""
IS THE TILE DATA A VECTOR BUNDLE?
 (1) LINEAR (not affine): does tile_{i+1} = A tile_i need an intercept?
 (2) LOCAL PREDICTABILITY: R^2 of neighbour prediction, and decay with lag
 (3) COCYCLE along a layer: A_lag-k  ==  (A_lag-1)^k ?
 (4) FLATNESS on plaquettes: M . A_coarse  ==  (A_fine)^r . M ?
Run for fwd and bwd ladders.
"""
import pickle, numpy as np
SH=pickle.load(open("sheaf2d.pkl","rb"))
LAY=["L0","L1","L2","L3","L4","L5"]; FWD=[16,32,64,256,512,1024]; BWD=[1024,512,256,64,32,16]
def R2(Y,P): 
    ss=((Y-Y.mean(0))**2).sum(); return 1-((Y-P)**2).sum()/max(ss,1e-30)
def fit(Xa,Xb,affine=False):
    A=np.hstack([Xa,np.ones((len(Xa),1))]) if affine else Xa
    W,_,_,_=np.linalg.lstsq(A,Xb,rcond=None)
    return W, R2(Xb,A@W)
print("="*90)
print("  (1)-(2) LINEARITY AND NEIGHBOUR PREDICTABILITY   tile_i -> tile_{i+1}")
print("="*90)
print(f"  {'group/side':>11}{'layer':>6}{'n':>6}{'R2 linear':>11}{'R2 affine':>11}"
      f"{'gain':>7}{'R2 lag2':>9}{'R2 lag4':>9}{'R2 lag16':>10}")
A1={}
for (gk,side),(res,Ms,info,order) in SH.items():
    if gk=="W_K": continue
    for L in LAY:
        if L not in res: continue
        X=res[L]["X"]; n=X.shape[0]
        if n<8: continue
        Wl,r2l=fit(X[:-1],X[1:]); Wa,r2a=fit(X[:-1],X[1:],affine=True)
        A1[(gk,side,L)]=Wl.T
        r2k=[]
        for k in (2,4,16):
            r2k.append(fit(X[:-k],X[k:])[1] if n>k+4 else float('nan'))
        print(f"  {gk+'/'+side:>11}{L:>6}{n:>6}{r2l:>11.4f}{r2a:>11.4f}{r2a-r2l:>7.4f}"
              f"{r2k[0]:>9.4f}{r2k[1]:>9.4f}{r2k[2]:>10.4f}")
print("\n  gain = R2(affine) - R2(linear).  ~0 => linear, summable: a VECTOR bundle,")
print("  not merely an affine one.")
print("\n"+"="*90); print("  (3) COCYCLE ALONG A LAYER:   A_k  vs  (A_1)^k"); print("="*90)
print(f"  {'group/side':>11}{'layer':>6}{'||A2-A1^2||/||A2||':>21}{'||A4-A1^4||/||A4||':>21}")
for (gk,side),(res,Ms,info,order) in SH.items():
    if gk=="W_K": continue
    for L in ("L0","L2","L5"):
        if L not in res: continue
        X=res[L]["X"]; n=X.shape[0]
        if n<24: continue
        A_1=A1[(gk,side,L)]
        out=[]
        for k in (2,4):
            Ak=fit(X[:-k],X[k:])[0].T
            P=np.linalg.matrix_power(A_1,k)
            out.append(np.linalg.norm(Ak-P)/max(np.linalg.norm(Ak),1e-30))
        print(f"  {gk+'/'+side:>11}{L:>6}{out[0]:>21.4f}{out[1]:>21.4f}")
print("\n"+"="*90); print("  (4) FLATNESS:  refine-then-translate  vs  translate-then-refine"); print("="*90)
print("     M . A_coarse   vs   (A_fine)^r . M     (M = vertical restriction map)")
print(f"  {'group/side':>11}{'rung':>11}{'r':>4}{'rel. holonomy':>16}{'||MA||':>10}")
for (gk,side),(res,Ms,info,order) in SH.items():
    if gk=="W_K": continue
    sizes=FWD if side=="fwd" else BWD
    sz=dict(zip(LAY,sizes))
    for (a,b),M in Ms.items():
        if (gk,side,a) not in A1 or (gk,side,b) not in A1: continue
        r=sz[b]//sz[a]
        Ac=A1[(gk,side,a)]; Af=A1[(gk,side,b)]
        P1=M@Ac; P2=np.linalg.matrix_power(Af,r)@M
        rel=np.linalg.norm(P1-P2)/max(np.linalg.norm(P1),1e-30)
        print(f"  {gk+'/'+side:>11}{a+'->'+b:>11}{r:>4}{rel:>16.4f}{np.linalg.norm(P1):>10.3f}")
print("\n  rel. holonomy ~0 => the connection is FLAT and the two directions commute:")
print("  observing one tile determines its neighbours consistently by either route.")
