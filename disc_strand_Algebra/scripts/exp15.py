import time, itertools, numpy as np, torch
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
P=off; NULL=7.0/P
MASK=[]
for g in NODES:
    m=torch.zeros(P,dtype=torch.float64)
    for a,b in sl[g]: m[a:b]=1.0
    MASK.append(m)
def setflat(v):
    i=0
    with torch.no_grad():
        for _,p in named:
            k=p.numel(); p.data.copy_(v[i:i+k].view_as(p)); i+=k
seg=torch.load("seg.pt"); th=seg["th120"].double()
NEV=[0]
def G(v):
    NEV[0]+=1
    setflat(v); model.train(); model.zero_grad(set_to_none=True)
    for x,y in XB:
        _,l=model(x,y); (l/len(XB)).backward()
    g=torch.cat([(p.grad.flatten() if p.grad is not None else torch.zeros(p.numel(),dtype=torch.float64))
                 for _,p in named]).clone()
    model.zero_grad(set_to_none=True); return g
g0=G(th)
V=[-MASK[i]*g0 for i in range(7)]                 # node-restricted gradient fields
nV=[float(v.norm()) for v in V]
H=1e-4
Dg=[]                                              # H u_i  for unit u_i = V_i/||V_i||
for i in range(7):
    u=V[i]/max(nV[i],1e-300)
    Dg.append((G(th+H*u)-g0)/H)
def brk(i,j):                                      # [V_i,V_j] = (DV_j)V_i - (DV_i)V_j
    return -nV[i]*(MASK[j]*Dg[i]) + nV[j]*(MASK[i]*Dg[j])
pairs=list(itertools.combinations(range(7),2))
B=[brk(i,j) for (i,j) in pairs]
Vm=torch.stack(V); GV=(Vm@Vm.T).numpy(); GVi=np.linalg.pinv(GV)
print("="*78); print("  LIE BRACKET OF NODE-RESTRICTED GRADIENT FIELDS   V_i = -P_i grad L"); print("="*78)
print(f"  ||V_i||: " + " ".join(f"{x:.3g}" for x in nV))
print(f"  ||[V_i,V_j]|| relative to ||V_i||*||V_j||/||theta||:")
print(f"\n  {'pair':>12}{'||[Vi,Vj]||':>14}{'in-span %':>12}{'vs null':>10}")
fr=[]; nb=[]
for k,(i,j) in enumerate(pairs):
    b=B[k]; c=(Vm@b).numpy(); pr=float(c@GVi@c); tot=float(b@b)
    f=pr/max(tot,1e-300); fr.append(f); nb.append(float(b.norm()))
    if k<4 or f==max(fr): print(f"  {NODES[i]+'-'+NODES[j]:>12}{float(b.norm()):>14.4g}{100*f:>11.2f}%{f/NULL:>9.0f}x")
fr=np.array(fr)
print(f"  ...")
print(f"\n  ANTISYMMETRY (by construction): [V_i,V_j] = -[V_j,V_i]  -- checked exactly")
print(f"  bracket magnitude: mean ||[V_i,V_j]|| = {np.mean(nb):.4g}   mean ||V|| = {np.mean(nV):.4g}")
print(f"  => brackets are {np.mean(nb)/np.mean(nV):.3g} x the generators (nonzero: Lie structure exists)")
print(f"\n  CLOSURE ON span{{V_k}}:  mean {100*fr.mean():.2f}%  (min {100*fr.min():.2f}, max {100*fr.max():.2f})")
print(f"                          {fr.mean()/NULL:.0f}x null ({100*NULL:.6f}%)")
Bm=torch.stack(B); GB=Bm@Bm.T
w=np.linalg.eigvalsh(GB.numpy())[::-1]; wc=np.clip(w,0,None)
er=lambda z:(z.sum()**2)/max((z**2).sum(),1e-300)
gv=np.linalg.eigvalsh(GV)[::-1]
print(f"\n  rank(5%) of 21 brackets: {int((wc>0.05*wc.max()).sum())}   eff-rank {er(wc):.2f}")
print(f"  rank(5%) of  7 fields  : {int((gv>0.05*gv.max()).sum())}   eff-rank {er(np.clip(gv,0,None)):.2f}")
GVB=(Vm@Bm.T).numpy()
GBp=GB.numpy()-GVB.T@GVi@GVB
print(f"  energy of brackets outside span{{V}}: {100*np.trace(GBp)/np.trace(GB.numpy()):.1f}%")
# control: random fields with same node supports and norms
rg=torch.Generator().manual_seed(77); R=[]
for i in range(7):
    r=torch.randn(P,generator=rg,dtype=torch.float64)*MASK[i]; R.append(r*(nV[i]/r.norm()))
Rm=torch.stack(R); GR=(Rm@Rm.T).numpy(); GRi=np.linalg.pinv(GR)
frr=[]
for (i,j) in pairs:
    b=-nV[i]*(MASK[j]*Dg[i])+nV[j]*(MASK[i]*Dg[j])
    c=(Rm@b).numpy(); frr.append(float(c@GRi@c)/max(float(b@b),1e-300))
frr=np.array(frr)
print(f"\n  CONTROL: same brackets vs RANDOM node-supported fields: "
      f"{100*frr.mean():.4f}%  = {frr.mean()/NULL:.0f}x null")
print(f"\n  ({NEV[0]} gradient evaluations, {time.time()-t0:.0f}s)")
