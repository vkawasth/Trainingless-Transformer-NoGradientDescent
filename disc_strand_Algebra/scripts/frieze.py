import pickle, numpy as np
SH=pickle.load(open("sheaf2d.pkl","rb"))
print("="*78); print("  IS THE REFINEMENT UNIMODULAR?   frieze/cluster needs det M = 1"); print("="*78)
print(f"  {'group/side':>11}{'rung':>10}{'det M':>10}{'|det-1|':>10}{'trace':>9}{'ev':>22}")
allde=[]
for (gk,side),(res,Ms,info,order) in SH.items():
    for (a,b),M in Ms.items():
        d=float(np.linalg.det(M)); t=float(np.trace(M)); w=np.linalg.eigvals(M)
        allde.append((gk,side,f"{a}->{b}",d))
        if gk in ("FF","LN"):
            print(f"  {gk+'/'+side:>11}{a+'->'+b:>10}{d:>10.4f}{abs(d-1):>10.4f}{t:>9.4f}"
                  f"   ({np.real(w[0]):+.3f},{np.real(w[1]):+.3f})")
D=np.array([x[3] for x in allde])
print(f"\n  all {len(D)} rungs:  mean det {D.mean():.4f}   median {np.median(D):.4f}"
      f"   |det-1| mean {np.abs(D-1).mean():.4f}")
print(f"  fraction within 10% of unimodular: {100*np.mean(np.abs(D-1)<0.1):.1f}%")
print("\n  by branch:")
for gk in ("FF","LN","W_K"):
    for side in ("fwd","bwd"):
        sel=np.array([x[3] for x in allde if x[0]==gk and x[1]==side])
        if len(sel)==0: continue
        print(f"    {gk+'/'+side:<11} det range [{sel.min():.3f}, {sel.max():.3f}]"
              f"   mean |det-1| = {np.abs(sel-1).mean():.3f}")
print("\n"+"="*78); print("  DIAMOND / CONWAY-COXETER TEST on the tile grid"); print("="*78)
print("  for adjacent tiles (i,i+1) at layer l and their children at l+1:")
print("  frieze rule requires  a*d - b*c = 1  on the value grid")
TT=pickle.load(open("tiletree.pkl","rb")); S=TT["series"]
for gk in ("FF","LN"):
    for side in ("bwd",):
        order=["L5","L4","L3","L2","L1","L0"] if side=="bwd" else ["L0","L1","L2","L3","L4","L5"]
        res=SH[(gk,side)][0]
        outs=[]
        for a,b in zip(order[:-1],order[1:]):
            if a not in res or b not in res: continue
            na,nb=res[a]["n"],res[b]["n"]
            if nb%na: continue
            r=nb//na
            # normalise each layer's beta to unit mean magnitude (friezes are scale-fixed)
            Ba=res[a]["X"][:,1]; Bb=res[b]["X"][:,1]
            Ba=Ba/np.abs(Ba).mean(); Bb=Bb/np.abs(Bb).mean()
            dets=[]
            for i in range(na-1):
                A=Ba[i]; B=Ba[i+1]; C=Bb[i*r]; Dd=Bb[(i+1)*r]
                dets.append(A*Dd-B*C)
            dets=np.array(dets)
            outs.append((f"{a}->{b}", dets.mean(), dets.std(), np.abs(dets-1).mean()))
        print(f"\n  {gk}/{side}")
        print(f"    {'rung':>10}{'mean(ad-bc)':>14}{'sd':>9}{'mean|ad-bc-1|':>16}")
        for o in outs: print(f"    {o[0]:>10}{o[1]:>14.4f}{o[2]:>9.4f}{o[3]:>16.4f}")
