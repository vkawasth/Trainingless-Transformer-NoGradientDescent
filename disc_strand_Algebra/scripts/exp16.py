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
P=off; NPAR=[sum(b-a for a,b in sl[g]) for g in NODES]
def setflat(v):
    i=0
    with torch.no_grad():
        for _,p in named:
            k=p.numel(); p.data.copy_(v[i:i+k].view_as(p)); i+=k
seg=torch.load("seg.pt"); th=seg["th120"].double()
def G(v):
    setflat(v); model.train(); model.zero_grad(set_to_none=True)
    for x,y in XB:
        _,l=model(x,y); (l/len(XB)).backward()
    g=torch.cat([(p.grad.flatten() if p.grad is not None else torch.zeros(p.numel(),dtype=torch.float64))
                 for _,p in named]).clone()
    model.zero_grad(set_to_none=True); return g
g0=G(th)
IDX=[torch.cat([torch.arange(a,b) for a,b in sl[g]]) for g in NODES]
def maskmul(v,i):
    out=torch.zeros_like(v); out[IDX[i]]=v[IDX[i]]; return out
V=[-maskmul(g0,i) for i in range(7)]; nV=[float(v.norm()) for v in V]
H=1e-4; Dg=[]
for i in range(7):
    Dg.append(((G(th+H*V[i]/nV[i])-g0)/H).float())
del g0; gc.collect()
Vf=torch.stack([v.float() for v in V]); del V; gc.collect()
GV=(Vf.double()@Vf.double().T).numpy(); GVi=np.linalg.pinv(GV)
pairs=list(itertools.combinations(range(7),2))
def brk(i,j): return -nV[i]*maskmul(Dg[i],j) + nV[j]*maskmul(Dg[j],i)
B=torch.stack([brk(i,j) for (i,j) in pairs])       # float32, 21 x P
del Dg; gc.collect()
Bd=B.double()
GB=(Bd@Bd.T).numpy(); GVB=(Vf.double()@Bd.T).numpy()
del Bd; gc.collect()
er=lambda z:(np.clip(z,0,None).sum()**2)/max((np.clip(z,0,None)**2).sum(),1e-300)
print("="*78); print("  LIE BRACKET CLOSURE  --  full accounting"); print("="*78)
fr=np.diag(GVB.T@GVi@GVB)/np.clip(np.diag(GB),1e-300,None)
print(f"  {'pair':>12}{'||[Vi,Vj]||':>13}{'in-span':>10}{'proper null':>13}{'vs null':>11}")
for k,(i,j) in enumerate(pairs):
    if k%4: continue
    nul=2.0/(NPAR[i]+NPAR[j])
    print(f"  {NODES[i]+'-'+NODES[j]:>12}{np.sqrt(GB[k,k]):>13.4g}{100*fr[k]:>9.1f}%"
          f"{100*nul:>12.5f}%{fr[k]/nul:>10.0f}x")
nulls=np.array([2.0/(NPAR[i]+NPAR[j]) for (i,j) in pairs])
print(f"\n  mean in-span {100*fr.mean():.1f}%   mean proper null {100*nulls.mean():.6f}%"
      f"   ratio {np.mean(fr/nulls):.0f}x")
wb=np.linalg.eigvalsh(GB)[::-1]; wv=np.linalg.eigvalsh(GV)[::-1]
GBp=GB-GVB.T@GVi@GVB; wp=np.linalg.eigvalsh(GBp)[::-1]
print(f"\n  7 fields   : rank(5%) {int((wv>0.05*wv.max()).sum())}  eff-rank {er(wv):.2f}")
print(f"  21 brackets: rank(5%) {int((np.clip(wb,0,None)>0.05*wb.max()).sum())}  eff-rank {er(wb):.2f}")
print(f"  brackets mod fields: eff-rank {er(wp):.2f}   "
      f"energy outside span{{V}}: {100*np.trace(GBp)/np.trace(GB):.1f}%")
# CONTROL: random node-supported fields of matched norm
rg=torch.Generator().manual_seed(77)
Rf=torch.zeros(7,P)
for i in range(7):
    r=torch.randn(len(IDX[i]),generator=rg); Rf[i,IDX[i]]=r/r.norm()*nV[i]
GR=(Rf.double()@Rf.double().T).numpy(); GRi=np.linalg.pinv(GR)
GRB=(Rf.double()@B.double().T).numpy()
frr=np.diag(GRB.T@GRi@GRB)/np.clip(np.diag(GB),1e-300,None)
print(f"\n  CONTROL  same brackets projected on RANDOM node-supported fields:")
print(f"    mean in-span {100*frr.mean():.5f}%   = {np.mean(frr/nulls):.1f}x proper null")
print(f"    signal/control ratio: {fr.mean()/max(frr.mean(),1e-30):.0f}x")
print(f"\n  time %.0fs"%(time.time()-t0))
