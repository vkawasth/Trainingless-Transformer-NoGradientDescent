import time, itertools, gc, numpy as np, torch
t0=time.time()
g_={}; src=open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
exec(src[:src.find("# ── PHASE 1")], g_)
model=g_["model"]; get_batch=g_["get_batch"]
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
model=model.double(); named=list(model.named_parameters())
sl={g:[] for g in NODES}; off=0
for n,p in named: sl[grp(n)].append((off,off+p.numel())); off+=p.numel()
P=off
IDX=[torch.cat([torch.arange(a,b) for a,b in sl[g]]) for g in NODES]
NP=[len(I) for I in IDX]
def setflat(v):
    i=0
    with torch.no_grad():
        for _,p in named:
            k=p.numel(); p.data.copy_(v[i:i+k].view_as(p)); i+=k
NEV=[0]
def G(v):
    NEV[0]+=1
    setflat(v); model.train(); model.zero_grad(set_to_none=True)
    for x,y in XB:
        _,l=model(x,y); (l/len(XB)).backward()
    g=torch.cat([(p.grad.flatten() if p.grad is not None else torch.zeros(p.numel(),dtype=torch.float64))
                 for _,p in named]).clone()
    model.zero_grad(set_to_none=True); return g
def mk(v,i):
    o=torch.zeros_like(v); o[IDX[i]]=v[IDX[i]]; return o
def bip(a,b,j): return float(a[IDX[j]]@b[IDX[j]])      # block-j inner product
th=torch.load("J120.pt")["th"].double()
H=1e-4
g0=G(th); nV=[float(g0[IDX[i]].norm()) for i in range(7)]
Dg=[(G(th+H*(-mk(g0,i))/nV[i])-g0)/H for i in range(7)]
print(f"  level-1 built, {NEV[0]} evals ({time.time()-t0:.0f}s)", flush=True)
# ---- level-1 basis: V_i = -P_i g0 (7)  and  Bm[j][i] = P_j Dg_i (49) ----
# Gram is block-diagonal: in block j the vectors are V_j and Bm[j][i], i=0..6
def blockbasis(j):                       # returns Gram (8x8) and a projector helper
    vecs=[-g0]+[Dg[i] for i in range(7)]     # V_j = -P_j g0 ; Bm[j][i]=P_j Dg_i
    Gm=np.array([[bip(a,b,j) for b in vecs] for a in vecs])
    return vecs,Gm
BB=[blockbasis(j) for j in range(7)]
def inspan_full(Z):                      # projection onto span{V} + span{P_j Dg_i}
    num=0.0
    for j in range(7):
        vecs,Gm=BB[j]
        c=np.array([bip(Z,v,j) for v in vecs])
        num+=float(c@np.linalg.pinv(Gm)@c)
    return num/max(float(Z@Z),1e-300)
def inspan_V(Z):
    num=sum(bip(Z,-g0,j)**2/max(nV[j]**2,1e-300) for j in range(7))
    return num/max(float(Z@Z),1e-300)
def Bfield(i,j,gg,dgi,dgj,ni,nj):        # bracket at a point, given local data
    return -ni*mk(dgi,j)+nj*mk(dgj,i)
B0={}
for (i,j) in itertools.combinations(range(7),2):
    B0[(i,j)]=Bfield(i,j,g0,Dg[i],Dg[j],nV[i],nV[j])
DIMfull=7+49; DIMV=7
print("="*76); print("  LEVEL-2 CLOSURE:  does [V_k,[V_i,V_j]] stay in span(level 1)?"); print("="*76)
print(f"  basis: span{{V}} = {DIMV} dims ; span{{V, P_j Dg_i}} = {DIMfull} dims (superset of span{{V,B}})")
print(f"  nulls: {100*DIMV/P:.6f}%  and  {100*DIMfull/P:.6f}%")
KS=[0,2,3]; PAIRS=[(3,4),(5,6),(0,2)]
need=sorted(set([i for p in PAIRS for i in p]))
print(f"\n  {'[V_k,[V_i,V_j]]':>22}{'||.||':>12}{'in span V':>12}{'in span L1':>12}{'vs null':>10}")
rows=[]
for k in KS:
    u=(-mk(g0,k))/nV[k]
    gk=G(th+H*u)
    nVk=[float(gk[IDX[i]].norm()) for i in range(7)]
    Dgk={}
    for i in need:
        Dgk[i]=(G(th+H*u+H*(-mk(gk,i))/nVk[i])-gk)/H
    for (i,j) in PAIRS:
        Bsh=Bfield(i,j,gk,Dgk[i],Dgk[j],nVk[i],nVk[j])
        DB_Vk=(Bsh-B0[(i,j)])/H*nV[k]                      # (DB)V_k
        b=B0[(i,j)]; nb=float(b.norm())
        HB=(G(th+H*b/nb)-g0)/H*nb                           # H B
        DVk_B=-mk(HB,k)                                     # (DV_k)B = -P_k H B
        Z=DB_Vk-DVk_B
        fV=inspan_V(Z); fL=inspan_full(Z)
        rows.append((k,i,j,float(Z.norm()),fV,fL))
        print(f"  {'['+NODES[k]+',['+NODES[i]+','+NODES[j]+']]':>22}{float(Z.norm()):>12.4g}"
              f"{100*fV:>11.1f}%{100*fL:>11.1f}%{fL/(DIMfull/P):>9.0f}x", flush=True)
    del Dgk,gk; gc.collect()
R=np.array([[r[4],r[5]] for r in rows])
print(f"\n  mean in-span: span{{V}} {100*R[:,0].mean():.1f}%   span(level 1) {100*R[:,1].mean():.1f}%")
print(f"  mean vs null (level-1 basis): {np.mean(R[:,1])/(DIMfull/P):.0f}x")
print(f"  level-1 brackets closed at 53.3% on span{{V}} (7 dims) for comparison")
print(f"\n  ({NEV[0]} gradient evaluations, {time.time()-t0:.0f}s)")
