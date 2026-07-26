"""
MONODROMY ON A DELIBERATELY CLOSED LOOP.
The block response direction u_l is defined up to sign, so it is a section of a
REAL LINE BUNDLE over the loop. Its holonomy is Z/2: after transporting once
around, u_l returns as +u_l (orientable, w1=0) or -u_l (Moebius, w1=1).
Loop: an exact circle in parameter space,  theta(t) = theta_c + r(cos t v1 + sin t v2),
so closure is guaranteed by construction rather than hoped for.
"""
import re, sys, time, gc, numpy as np, torch
t0=time.time()
R_FRAC=float(sys.argv[1]) if len(sys.argv)>1 else 0.02
N=int(sys.argv[2]) if len(sys.argv)>2 else 24
g_={}; src=open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
exec(src[:src.find("# ── PHASE 1")], g_)
model=g_["model"]; get_batch=g_["get_batch"]; LR=g_["LR"]
torch.manual_seed(3); XB=[get_batch() for _ in range(2)]
model=model.double(); named=list(model.named_parameters())
def lay(n):
    m=re.match(r"blocks\.(\d+)\.",n); return f"L{m.group(1)}" if m else "EMB"
off=0; GI={}
for n,p in named: GI.setdefault(lay(n),[]).append(torch.arange(off,off+p.numel())); off+=p.numel()
GI={k:torch.cat(v) for k,v in GI.items()}; KEYS=["EMB"]+[f"L{i}" for i in range(6)]
IDX=[GI[k] for k in KEYS]; K=len(KEYS); P=off
def setflat(v):
    i=0
    with torch.no_grad():
        for _,p in named:
            k=p.numel(); p.data.copy_(v[i:i+k].view_as(p)); i+=k
def flat(): return torch.cat([p.data.flatten() for _,p in named]).clone()
def G(v):
    setflat(v); model.train(); model.zero_grad(set_to_none=True)
    for x,y in XB:
        _,l=model(x,y); (l/len(XB)).backward()
    g=torch.cat([(p.grad.flatten() if p.grad is not None else torch.zeros(p.numel(),dtype=torch.float64))
                 for _,p in named]).clone()
    model.zero_grad(set_to_none=True); return g
def mk(v,i):
    o=torch.zeros_like(v); o[IDX[i]]=v[IDX[i]]; return o
# --- reach a checkpoint and take two trajectory directions for the plane ---
ck=torch.load("J120.pt"); thc=ck["th"].double()
model.load_state_dict({k:v.double() for k,v in ck["sd"].items()})
opt=torch.optim.AdamW(model.parameters(), lr=LR*5, betas=(0.9,0.95), weight_decay=0.1)
prev=flat(); ds=[]
for _ in range(3):
    model.train(); x,y=get_batch(); _,l=model(x,y)
    opt.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    cur=flat(); ds.append((cur-prev).clone()); prev=cur
v1=ds[0]/ds[0].norm()
v2=ds[2]-(ds[2]@v1)*v1; v2=v2/v2.norm()
r=R_FRAC*float(thc.norm())
print(f"  loop radius {r:.3f} = {R_FRAC:.3f}*||theta||   N={N} points   "
      f"v1.v2={float(v1@v2):.2e}", flush=True)
H=1e-4
def u_at(th):
    g0=G(th); nV=[float(g0[I].norm()) for I in IDX]
    A=[]
    for m in range(K):
        d=torch.zeros_like(th); d[IDX[m]]=-g0[IDX[m]]/max(nV[m],1e-30)
        A.append((G(th+H*d)-g0)/H)
    out=[]
    for l in range(K):
        M=torch.stack([A[m][IDX[l]] for m in range(K)])
        _,S,Vt=torch.linalg.svd(M,full_matrices=False)
        out.append((Vt[0].clone(), float(S[1]/S[0])))
    del A,g0; gc.collect(); return out
U=[]; GAP=[]
for k in range(N+1):
    t=2*np.pi*k/N
    th=thc + r*(np.cos(t)*v1 + np.sin(t)*v2)
    o=u_at(th); U.append([x[0] for x in o]); GAP.append(np.mean([x[1] for x in o]))
    if k%6==0: print(f"    point {k}/{N} ({time.time()-t0:.0f}s)", flush=True)
print("\n"+"="*80); print("  HOLONOMY OF THE BLOCK-RESPONSE LINE BUNDLE"); print("="*80)
print(f"  mean singular gap S2/S1 along the loop = {np.mean(GAP):.3f}"
      f"  (small => u_l well defined)")
print(f"\n  {'block':>7}{'min |cos| step':>16}{'accum. angle':>15}{'sign after loop':>18}{'w1':>5}")
for l in range(K):
    sgn=1.0; mn=1.0; acc=0.0
    prevv=U[0][l]
    for k in range(1,N+1):
        c=float(prevv@U[k][l]); s=np.sign(c) if c!=0 else 1.0
        U[k][l]=U[k][l]*s          # continuous sign choice
        cc=abs(c); mn=min(mn,cc); acc+=np.degrees(np.arccos(min(1,cc)))
        sgn*=s; prevv=U[k][l]
    close=float(U[0][l]@U[N][l])
    print(f"  {KEYS[l]:>7}{mn:>16.3f}{acc:>14.1f}d{('+' if close>0 else '-')+f'{abs(close):.3f}':>18}"
          f"{(0 if close>0 else 1):>5}")
print("\n  w1=1 (sign flip) => non-orientable line bundle: genuine Z/2 monodromy.")
print("  w1=0 for every block => trivial holonomy, no winding.")
print(f"\n  time {time.time()-t0:.0f}s")
