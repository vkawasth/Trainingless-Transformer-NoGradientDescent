"""
SPLIT-HALF RELIABILITY OF (alpha,beta).
If M's eigenvalues merely reproduce the reliability of each coordinate,
the 'Perron-Frobenius collapse onto beta' is regression attenuation.
Test: fit (alpha,beta) independently on odd/even steps; correlate across tiles.
Reliability via Spearman-Brown from the split-half correlation.
"""
import pickle, numpy as np
D=pickle.load(open("tiletree.pkl","rb")); S=D["series"]; META=D["meta"]
H=0.062
loss=np.array([m["loss"] for m in META]); LAY=["L0","L1","L2","L3","L4","L5"]

def fit(A, st, mask):
    ls=loss[st[mask]-1]-H
    ok=ls>1e-6
    x=np.log(ls[ok]); Y=np.log(np.maximum(A[mask][ok],1e-30))
    xm=x-x.mean(); den=float((xm*xm).sum())
    al=(Y-Y.mean(0)).T@xm/den
    be=Y.mean(0)-al*x.mean()
    return al, be

def sb(r):                      # Spearman-Brown: half-test r -> full-test reliability
    r=max(min(r,0.999999),-0.999999); return 2*r/(1+r)

print("="*88)
print("  SPLIT-HALF RELIABILITY  (odd vs even steps, independent fits)")
print("="*88)
print("  reliability = Spearman-Brown corrected split-half correlation across tiles")
print(f"\n  {'group/side':>11}{'layer':>6}{'tiles':>7}{'r_split(a)':>12}{'rel(a)':>9}"
      f"{'r_split(b)':>12}{'rel(b)':>9}{'M ev_a':>9}")
SH=pickle.load(open("sheaf2d.pkl","rb"))
EVA={}
for (gk,side),(res,Ms,info,order) in SH.items():
    evs={}
    for (a,b),M in Ms.items():
        w=np.linalg.eigvals(M); evs[b]=float(np.min(np.abs(w)))
    EVA[(gk,side)]=evs
rows=[]
for gk in ("FF","LN","W_K"):
    for side in ("fwd","bwd"):
        order=LAY if side=="fwd" else LAY[::-1]
        for L in order:
            k=f"{L}|{gk}|{side}"
            if k not in S: continue
            A=np.abs(S[k]["max"]).astype(np.float64)
            f0=S[k]["first_step"]; st=np.arange(f0,f0+A.shape[0])
            m1=(np.arange(len(st))%2==0); m2=~m1
            a1,b1=fit(A,st,m1); a2,b2=fit(A,st,m2)
            ra=np.corrcoef(a1,a2)[0,1] if a1.std()>0 and a2.std()>0 else np.nan
            rb=np.corrcoef(b1,b2)[0,1] if b1.std()>0 and b2.std()>0 else np.nan
            ev=EVA[(gk,side)].get(L,np.nan)
            rows.append((gk,side,L,len(a1),ra,sb(ra),rb,sb(rb),ev))
            print(f"  {gk+'/'+side:>11}{L:>6}{len(a1):>7}{ra:>12.3f}{sb(ra):>9.3f}"
                  f"{rb:>12.3f}{sb(rb):>9.3f}{ev:>9.3f}")
print("\n"+"="*88); print("  THE DECISIVE COMPARISON"); print("="*88)
R=[r for r in rows if not np.isnan(r[8])]
ra=np.array([r[5] for r in R]); rb=np.array([r[7] for r in R]); ev=np.array([r[8] for r in R])
print(f"  n = {len(R)} rungs with a fitted restriction map")
print(f"  mean reliability(alpha) = {ra.mean():.3f}   mean reliability(beta) = {rb.mean():.3f}")
print(f"  mean contracted eigenvalue of M = {ev.mean():.3f}")
print(f"\n  ATTENUATION NULL predicts:  ev_alpha ~ reliability(alpha)")
print(f"    corr( rel(alpha), ev_alpha ) = {np.corrcoef(ra,ev)[0,1]:+.3f}")
print(f"    mean |ev_alpha - rel(alpha)| = {np.abs(ev-ra).mean():.3f}")
print(f"    mean ev_alpha / rel(alpha)   = {np.mean(ev/np.maximum(ra,1e-6)):.3f}")
print(f"\n  {'verdict':>11}")
if np.abs(ev-ra).mean()<0.15 and np.corrcoef(ra,ev)[0,1]>0.5:
    print("   => CONSISTENT WITH ATTENUATION: the collapse tracks measurement reliability")
elif ra.mean()>0.7 and ev.mean()<0.4:
    print("   => REAL: alpha is reliably measured yet still contracted by refinement")
else:
    print("   => MIXED: see per-group table")
print("\n  per-group summary:")
print(f"  {'group/side':>11}{'rel(a)':>9}{'rel(b)':>9}{'mean ev_a':>11}{'ev_a - rel(a)':>15}")
for gk in ("FF","LN","W_K"):
    for side in ("fwd","bwd"):
        sel=[r for r in R if r[0]==gk and r[1]==side]
        if not sel: continue
        A=np.array([x[5] for x in sel]); B=np.array([x[7] for x in sel]); E=np.array([x[8] for x in sel])
        print(f"  {gk+'/'+side:>11}{A.mean():>9.3f}{B.mean():>9.3f}{E.mean():>11.3f}{(E-A).mean():>+15.3f}")
