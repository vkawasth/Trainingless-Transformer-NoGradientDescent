import json, numpy as np
H=json.load(open("hist.json")); GR=["EMB","LN","FF","W_Q","W_K","W_V","W_O"]
s=np.array([h["s"] for h in H]); nd=np.array([h["nd"] for h in H])
cos1=np.array([h["cos1"] for h in H]); kap=np.array([h["kap"] for h in H])
gn=np.array([h["gn"] for h in H]); th=np.array([h["theta"] for h in H])
V={h["s"]:h["val"] for h in H if "val" in h}
prof=np.array([h["prof"] for h in H])
print("="*74); print("  TRAJECTORY + CURVATURE, 200 STEPS"); print("="*74)
print(f"  {'win':>9}{'val':>9}{'|grad|':>9}{'|d|':>8}{'cos1':>8}{'kappa':>9}{'R=1/k':>9}{'R/|th|':>8}")
print("  "+"-"*66)
for a in range(0,200,20):
    m=(s>a)&(s<=a+20)
    vv=[V[k] for k in V if a<k<=a+20]
    print(f"  {a+1:>4}-{a+20:<4}{np.mean(vv):>9.4f}{gn[m].mean():>9.4f}{nd[m].mean():>8.4f}"
          f"{np.nanmean(cos1[m]):>8.3f}{np.nanmean(kap[m]):>9.4f}{1/np.nanmean(kap[m]):>9.3f}"
          f"{(1/np.nanmean(kap[m]))/th[m].mean():>8.3f}")
# drift of block profile (geo_optimizer style), window 20
print("\n  block-profile drift (||p_t - p_{t-20}||):")
dr=[]
for i in range(20,len(prof)):
    dr.append((s[i], float(np.linalg.norm(prof[i]-prof[i-20]))))
for a in range(0,200,20):
    w=[d for t,d in dr if a<t<=a+20]
    if w: print(f"    {a+1:>4}-{a+20:<4} drift {np.mean(w):.4f}")
print("\n  gradient share by block (mean over window):")
print(f"  {'win':>9}" + "".join(f"{g:>8}" for g in GR))
for a in [0,20,40,100,180]:
    m=(s>a)&(s<=a+20)
    print(f"  {a+1:>4}-{a+20:<4}" + "".join(f"{100*prof[m,i].mean():>7.1f}%" for i in range(7)))
print("\n  lag-cosines of unit tangent (rotation time constant):")
for h in H:
    if "lag" in h and h["s"] in (72,104,136,168,200):
        L=h["lag"]; print(f"    step {h['s']:>3}  " + "  ".join(f"k={k}:{L[k]:+.3f}" for k in ["1","4","8","16","32","64"]))
# Lojasiewicz fit on phase 3: dDelta/dt = -c Delta^(2theta)
ks=sorted(V); FLOOR=0.062
print("\n"+"="*74); print("  LOJASIEWICZ FIT  dD/dt = -c D^p   (D = val - floor)"); print("="*74)
for lo,hi in [(60,200),(100,200),(120,200),(140,200)]:
    kk=[k for k in ks if lo<=k<=hi]
    D=np.array([V[k]-FLOOR for k in kk]); t=np.array(kk,dtype=float)
    ok=D>1e-4; D=D[ok]; t=t[ok]
    if len(D)<5: continue
    dD=np.gradient(D,t); m=(dD<0)
    if m.sum()<5: print(f"  window {lo}-{hi}: too few decreasing points ({m.sum()})"); continue
    A=np.polyfit(np.log(D[m]), np.log(-dD[m]), 1)
    print(f"  window {lo:>4}-{hi:<4} n={m.sum():>3}  p = {A[0]:.3f}  (theta={A[0]/2:.3f})  c={np.exp(A[1]):.4g}")
print("\n  val trace:", " ".join(f"{k}:{V[k]:.4f}" for k in ks if k%20==0))
