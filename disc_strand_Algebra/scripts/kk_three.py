"""
(i)  global beta-line   : is the eigenvalue-1 eigenvector shared across groups? [sheaf]
(ii) attenuation w_l    : fixes the weighted grading J_w, J_w^2 = w        [geometry]
(iii)Fisher faithfulness: any tile of zero weight => order ideal           [operator algebra]
(iv) inductive limit    : does the refinement composite collapse to beta?  [Perron-Frobenius]
"""
import pickle, numpy as np
np.set_printoptions(precision=4, suppress=True)
SH=pickle.load(open("sheaf2d.pkl","rb"))     # (gk,side) -> (res, Ms, info, order)
FIS=pickle.load(open("fisher.pkl","rb"))
TT=pickle.load(open("tiletree.pkl","rb")); S=TT["series"]
LAY=["L0","L1","L2","L3","L4","L5"]

print("="*82); print("  (i)  THE BETA LINE: eigenvalue-1 eigenvector of the restriction system"); print("="*82)
print(f"  {'group/side':>12}{'rung':>10}{'ev1':>9}{'ev2':>9}{'v1=(a,b) normalised':>26}")
V1={}
for (gk,side),(res,Ms,info,order) in SH.items():
    vs=[]
    for (a,b),M in Ms.items():
        w,V=np.linalg.eig(M)
        i=int(np.argmin(np.abs(np.abs(w)-1.0)))
        v=np.real(V[:,i]); v=v/np.linalg.norm(v); v*= np.sign(v[1] if abs(v[1])>1e-12 else 1)
        vs.append(v)
        if (a,b)==list(Ms.keys())[0] or (a,b)==list(Ms.keys())[-1]:
            print(f"  {gk+'/'+side:>12}{a+'->'+b:>10}{np.real(w[i]):>9.4f}"
                  f"{np.real(w[1-i]):>9.4f}   ({v[0]:+.4f},{v[1]:+.4f})")
    V1[(gk,side)]=np.array(vs)
print("\n  consistency of the fixed line WITHIN a group (cos between rungs):")
for k,vs in V1.items():
    c=[abs(float(vs[i]@vs[j])) for i in range(len(vs)) for j in range(i+1,len(vs))]
    print(f"    {k[0]+'/'+k[1]:<12} mean cos {np.mean(c):.4f}   min {np.min(c):.4f}")
print("\n  consistency ACROSS groups (cos of mean fixed line):")
ks=list(V1.keys()); mv={k:(V1[k].mean(0)/np.linalg.norm(V1[k].mean(0))) for k in ks}
print(f"  {'':>12}"+"".join(f"{k[0]+'/'+k[1]:>12}" for k in ks))
for k in ks:
    print(f"  {k[0]+'/'+k[1]:>12}"+"".join(f"{abs(float(mv[k]@mv[j])):>12.4f}" for j in ks))

print("\n"+"="*82); print("  (ii) ATTENUATION PROFILE w_l   (backward reach: J_w^2 = w)"); print("="*82)
print(f"  {'group':>7}" + "".join(f"{L:>10}" for L in LAY) + f"{'L5/L0':>9}")
for gk in ("FF","LN","W_K","W_V","W_O","W_Q"):
    row=[]
    for L in LAY:
        k=f"{L}|{gk}|bwd"
        row.append(np.abs(S[k]["max"]).mean() if k in S else np.nan)
    row=np.array(row)
    if np.all(np.isnan(row)): continue
    print(f"  {gk:>7}" + "".join(f"{x:>10.3e}" for x in row) + f"{row[-1]/row[0]:>9.3f}")
print("\n  normalised w_l (FF, backward magnitude relative to L5):")
r=np.array([np.abs(S[f"{L}|FF|bwd"]["max"]).mean() for L in LAY]); r=r/r[-1]
print("    " + "  ".join(f"{L}:{x:.3f}" for L,x in zip(LAY,r)))
print(f"    => J_w is a contraction toward the input, factor {r[0]:.3f} at L0")

print("\n"+"="*82); print("  (iii) FISHER FAITHFULNESS on the tile algebra"); print("="*82)
print(f"  {'group':>7}{'tiles':>8}{'zeros':>7}{'min w':>12}{'max/min':>12}{'entropy/max':>13}")
anyzero=0
for gk in ("FF","LN","W_K","W_V","W_O","W_Q"):
    tot=[]; z=0; n=0
    for L in LAY:
        key=f"{L}|{gk}"
        if key not in FIS: continue
        fv=FIS[key]; nt=S[f"{L}|{gk}|fwd"]["nt"]
        e=np.linspace(0,len(fv),nt+1).astype(int)
        w=np.array([fv[e[i]:e[i+1]].sum() for i in range(nt)])
        tot.append(w); z+=int((w<=0).sum()); n+=nt
    if not tot: continue
    W=np.concatenate(tot); anyzero+=z
    p=W/W.sum(); H=-(p[p>0]*np.log(p[p>0])).sum()
    print(f"  {gk:>7}{n:>8}{z:>7}{W.min():>12.3e}{W.max()/max(W.min(),1e-300):>12.3e}"
          f"{H/np.log(len(W)):>13.4f}")
print(f"\n  total zero-weight tiles: {anyzero}"
      f"  => Fisher state is {'NOT faithful (order ideal exists)' if anyzero else 'FAITHFUL (full support)'}")

print("\n"+"="*82); print("  (iv) INDUCTIVE LIMIT: composite of all restriction maps"); print("="*82)
print(f"  {'group/side':>12}{'ev(composite)':>26}{'rank@1%':>9}{'cos(top ev, beta-line)':>24}")
for (gk,side),(res,Ms,info,order) in SH.items():
    Mlist=[Ms[k] for k in Ms]
    C=np.eye(2)
    for M in Mlist: C=M@C
    w,V=np.linalg.eig(C); i=int(np.argmax(np.abs(w)))
    v=np.real(V[:,i]); v=v/np.linalg.norm(v); v*=np.sign(v[1] if abs(v[1])>1e-12 else 1)
    rk=int((np.abs(w)>0.01*np.abs(w).max()).sum())
    bl=mv[(gk,side)]
    print(f"  {gk+'/'+side:>12}   ev=({np.real(w[0]):+.4f},{np.real(w[1]):+.4f})"
          f"{rk:>9}{abs(float(v@bl)):>24.4f}")
print("\n  ratio |ev2/ev1| of the composite (contraction onto the beta line):")
for (gk,side),(res,Ms,info,order) in SH.items():
    C=np.eye(2)
    for k in Ms: C=Ms[k]@C
    w=np.abs(np.linalg.eigvals(C)); w=np.sort(w)[::-1]
    print(f"    {gk+'/'+side:<12} |ev2|/|ev1| = {w[1]/max(w[0],1e-300):.5f}"
          f"   ({'rank-1 in the limit' if w[1]/max(w[0],1e-300)<0.05 else 'both survive'})")
