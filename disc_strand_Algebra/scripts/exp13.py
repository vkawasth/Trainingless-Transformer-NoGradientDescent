import time, itertools, numpy as np, torch
t0=time.time()
g_={}; src=open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
exec(src[:src.find("# ── PHASE 1")], g_)
model=g_["model"]; get_batch=g_["get_batch"]; LR=g_["LR"]
named=list(model.named_parameters())
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
seg=torch.load("seg.pt"); th=seg["th120"]; DT=seg["DT"]
torch.manual_seed(3); XB=[get_batch() for _ in range(2)]
ETA=float(LR*5)
NEV=[0]
def PHI(v):                       # flow map: one deterministic gradient step
    NEV[0]+=1
    setflat(v); model.train(); model.zero_grad(set_to_none=True)
    for x,y in XB:
        _,l=model(x,y); (l/len(XB)).backward()
    g=torch.cat([(p.grad.flatten() if p.grad is not None else torch.zeros(p.numel()))
                 for _,p in named]).clone()
    model.zero_grad(set_to_none=True)
    return v - ETA*g
P0=PHI(th)
m0 = P0-th                                    # curvature term
def m1(v):  return PHI(th+v)-P0-v             # reduced linearisation of the flow
def m2(u,v):return PHI(th+u+v)-PHI(th+u)-PHI(th+v)+P0
def m3(u,v,w):
    return (PHI(th+u+v+w)-PHI(th+u+v)-PHI(th+u+w)-PHI(th+v+w)
            +PHI(th+u)+PHI(th+v)+PHI(th+w)-P0)
T=0.10
D=[T*sub(DT,[g]) for g in NODES]
print("="*78); print("  FLOW-DEFINED CURVED A-INF DATA   Phi(th)=th-eta*grad,  eta=%.2g"%ETA); print("="*78)
print(f"  ||m0|| = {float(m0.norm()):.4g}   ||d_i|| = "
      + " ".join(f"{float(D[i].norm()):.3g}" for i in range(7)))
# scaling check: m2 ~ t^2, m3 ~ t^3
i,j,k=0,2,3
a2=float(m2(D[i],D[j]).norm()); b2=float(m2(0.5*D[i],0.5*D[j]).norm())
a3=float(m3(D[i],D[j],D[k]).norm()); b3=float(m3(0.5*D[i],0.5*D[j],0.5*D[k]).norm())
print(f"  scaling: ||m2(t)||/||m2(t/2)|| = {a2/max(b2,1e-30):.2f} (expect 4)   "
      f"||m3(t)||/||m3(t/2)|| = {a3/max(b3,1e-30):.2f} (expect 8)")
print("\n"+"-"*78)
print("  RELATION n=1 (curved):   m1(m1(x)) + m2(m0,x) + m2(x,m0) = 0")
print(f"  {'x':>8}{'||m1^2 x||':>14}{'||2 m2(m0,x)||':>17}{'residual':>13}{'rel':>9}")
for idx in [0,2,3,5]:
    x=D[idx]; A=m1(m1(x)); B=2*m2(m0,x); R=A+B
    sc=max(float(A.norm()),float(B.norm()),1e-30)
    print(f"  {NODES[idx]:>8}{float(A.norm()):>14.4g}{float(B.norm()):>17.4g}"
          f"{float(R.norm()):>13.4g}{float(R.norm())/sc:>9.3f}")
print("\n"+"-"*78)
print("  RELATION n=3 (Stasheff):  Assoc(x,y,z) = -[ m1(m3) - m3(m1x,y,z) - m3(x,m1y,z) - m3(x,y,m1z) ]")
print(f"  {'triple':>16}{'||Assoc||':>13}{'||d m3||':>13}{'cos':>9}{'rel resid':>12}")
for (a,b,c) in [(0,2,3),(0,2,5),(3,4,5),(2,5,6),(0,3,4)]:
    x,y,z=D[a],D[b],D[c]
    As = m2(m2(x,y),z) - m2(x,m2(y,z))
    dm = m1(m3(x,y,z)) - m3(m1(x),y,z) - m3(x,m1(y),z) - m3(x,y,m1(z))
    cs = float(As@dm/(As.norm()*dm.norm()+1e-30))
    r1 = float((As+dm).norm())/max(float(As.norm()),1e-30)
    r2 = float((As-dm).norm())/max(float(As.norm()),1e-30)
    print(f"  {'-'.join(NODES[q] for q in (a,b,c)):>16}{float(As.norm()):>13.4g}"
          f"{float(dm.norm()):>13.4g}{cs:>9.3f}{min(r1,r2):>12.3f}", flush=True)
print(f"\n  ({NEV[0]} flow evaluations, {time.time()-t0:.0f}s)")
