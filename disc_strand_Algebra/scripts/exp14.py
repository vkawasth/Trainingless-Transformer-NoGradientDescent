import time, itertools, numpy as np, torch
t0=time.time()
g_={}; src=open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
exec(src[:src.find("# ── PHASE 1")], g_)
model=g_["model"]; get_batch=g_["get_batch"]; LR=g_["LR"]
NODES=["EMB","LN","FF","W_Q","W_K","W_V","W_O"]
def grp(n):
    n=n.lower()
    if ".ln." in n or ".n." in n or n.startswith("ln_f"): return "LN"
    if n.startswith("te") or n.startswith("pe") or n.startswith("head"): return "EMB"
    if ".ff." in n: return "FF"
    if "wk" in n: return "W_K"
    if "wq" in n: return "W_Q"
    if "wv" in n: return "W_V"
    if ".op." in n: return "W_O"
torch.manual_seed(3); XB=[get_batch() for _ in range(2)]
model=model.double()
named=list(model.named_parameters())
sl={g:[] for g in NODES}; off=0
for n,p in named: sl[grp(n)].append((off,off+p.numel())); off+=p.numel()
def setflat(v):
    i=0
    with torch.no_grad():
        for _,p in named:
            k=p.numel(); p.data.copy_(v[i:i+k].view_as(p)); i+=k
def sub(v,gs):
    m=torch.zeros_like(v)
    for g in gs:
        for a,b in sl[g]: m[a:b]=v[a:b]
    return m
seg=torch.load("seg.pt"); th=seg["th120"].double(); DT=seg["DT"].double()
ETA=float(LR*5); NEV=[0]
def G(v):                      # gradient at th+v  (theta-terms never enter the differences)
    NEV[0]+=1
    setflat(th+v); model.train(); model.zero_grad(set_to_none=True)
    for x,y in XB:
        _,l=model(x,y); (l/len(XB)).backward()
    g=torch.cat([(p.grad.flatten() if p.grad is not None else torch.zeros(p.numel(),dtype=torch.float64))
                 for _,p in named]).clone()
    model.zero_grad(set_to_none=True); return g
Z=torch.zeros_like(th); G0=G(Z)
m0=-ETA*G0
def m1(v):   return -ETA*(G(v)-G0)
def m2(u,v): return -ETA*(G(u+v)-G(u)-G(v)+G0)
def m3(u,v,w):
    return -ETA*(G(u+v+w)-G(u+v)-G(u+w)-G(v+w)+G(u)+G(v)+G(w)-G0)
T=0.10; D=[T*sub(DT,[g]) for g in NODES]
print("="*78); print("  FLOW-DEFINED CURVED A-INF  (float64, gradient differences)"); print("="*78)
print(f"  eta={ETA:.3g}  ||m0||={float(m0.norm()):.4g}  ||d_i||="
      +" ".join(f"{float(D[i].norm()):.2f}" for i in range(7)))
i,j,k=0,2,3
r2=float(m2(D[i],D[j]).norm())/max(float(m2(.5*D[i],.5*D[j]).norm()),1e-300)
r3=float(m3(D[i],D[j],D[k]).norm())/max(float(m3(.5*D[i],.5*D[j],.5*D[k]).norm()),1e-300)
print(f"  VALIDITY  ||m2(t)||/||m2(t/2)|| = {r2:.2f} (expect 4)   "
      f"||m3(t)||/||m3(t/2)|| = {r3:.2f} (expect 8)")
print("\n"+"-"*78)
print("  RELATION n=1 (curved):  m1(m1 x) + m2(m0,x) + m2(x,m0) = 0")
print(f"  {'x':>8}{'||m1^2 x||':>14}{'||2m2(m0,x)||':>16}{'residual':>13}{'rel':>8}")
for idx in [0,2,3,5]:
    x=D[idx]; A=m1(m1(x)); B=2*m2(m0,x); R=A+B
    sc=max(float(A.norm()),float(B.norm()),1e-300)
    print(f"  {NODES[idx]:>8}{float(A.norm()):>14.4g}{float(B.norm()):>16.4g}"
          f"{float(R.norm()):>13.4g}{float(R.norm())/sc:>8.3f}", flush=True)
print("\n"+"-"*78)
print("  RELATION n=3 (Stasheff):  Assoc  vs  d(m3)")
print(f"  {'triple':>16}{'||Assoc||':>13}{'||d m3||':>13}{'cos':>9}{'best rel':>11}")
for (a,b,c) in [(0,2,3),(3,4,5),(2,5,6)]:
    x,y,z=D[a],D[b],D[c]
    As=m2(m2(x,y),z)-m2(x,m2(y,z))
    dm=m1(m3(x,y,z))-m3(m1(x),y,z)-m3(x,m1(y),z)-m3(x,y,m1(z))
    cs=float(As@dm/(As.norm()*dm.norm()+1e-300))
    rr=min(float((As+dm).norm()),float((As-dm).norm()))/max(float(As.norm()),1e-300)
    print(f"  {'-'.join(NODES[q] for q in (a,b,c)):>16}{float(As.norm()):>13.4g}"
          f"{float(dm.norm()):>13.4g}{cs:>9.3f}{rr:>11.3f}", flush=True)
print(f"\n  ({NEV[0]} gradient evaluations, {time.time()-t0:.0f}s)")
