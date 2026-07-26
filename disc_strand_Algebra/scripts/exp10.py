import time, itertools, gc, numpy as np, torch
t0=time.time()
g_={}; src=open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
exec(src[:src.find("# ── PHASE 1")], g_)
model=g_["model"]; get_batch=g_["get_batch"]
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
def logits(vec):
    setflat(vec); model.eval(); out=[]
    with torch.no_grad():
        for x,y in XB: lo,_=model(x,y); out.append(lo.reshape(-1).clone())
    return torch.cat(out)
L0=logits(th); D=[sub(DT,[g]) for g in NODES]
pairs=list(itertools.combinations(range(7),2))
trip =list(itertools.combinations(range(7),3))
def F(vec_list, t):                      # df(t*sum of listed nodes)
    v=th.clone()
    for i in vec_list: v=v+t*D[i]
    return logits(v)-L0
f1=[F([i],1.0) for i in range(7)]
fh=[F([i],0.5) for i in range(7)]
print(f"  singles done ({time.time()-t0:.0f}s)", flush=True)
# grade-2 at t=1 and t=0.5, Richardson: H = 2*A(t/2) - A(t), A(t)=m_t/t^2
M1=[]; MH=[]
for (i,j) in pairs:
    M1.append(F([i,j],1.0)-f1[i]-f1[j])
    MH.append(F([i,j],0.5)-fh[i]-fh[j])
H=[ (8.0*MH[k] - M1[k]) for k in range(len(pairs)) ]     # pure 2nd-order part
print(f"  pairs done ({time.time()-t0:.0f}s)", flush=True)
# connected triple (inclusion-exclusion) = the obstruction term
Tt=[]
for (i,j,k) in trip:
    v=F([i,j,k],1.0)-F([i,j],1.0)-F([i,k],1.0)-F([j,k],1.0)+f1[i]+f1[j]+f1[k]
    Tt.append(v)
print(f"  triples done ({time.time()-t0:.0f}s)", flush=True)
def gram(A,B):
    return np.array([[float(a.double()@b.double()) for b in B] for a in A])
G1=gram(f1,f1); GH=gram(H,H); G1H=gram(f1,H)
GM=gram(M1,M1); G1M=gram(f1,M1); GT=gram(Tt,Tt); GHT=gram(H,Tt)
del Tt; gc.collect()
def erank(w): w=np.clip(w,0,None); return (w.sum()**2)/max((w**2).sum(),1e-30)
def rk(w,tol=0.05): w=np.clip(w,0,None); return int((w>tol*w.max()).sum())
def closure(Gxx,Gax,Gaa):
    P=np.diag(Gax.T@np.linalg.pinv(Gaa)@Gax)
    return P/np.clip(np.diag(Gxx),1e-30,None)
print("\n"+"="*74); print("  CURVATURE CORRECTION: raw product vs pure 2nd-order part"); print("="*74)
cm=closure(GM,G1M,G1); ch=closure(GH,G1H,G1)
print(f"  {'':<26}{'raw m (t=1)':>14}{'Richardson H':>15}")
print(f"  {'mean in-span fraction':<26}{100*cm.mean():>13.1f}%{100*ch.mean():>14.1f}%")
print(f"  {'  min / max':<26}{f'{100*cm.min():.1f} / {100*cm.max():.1f}':>14}"
      f"{f'{100*ch.min():.1f} / {100*ch.max():.1f}':>15}")
wm=np.linalg.eigvalsh(GM)[::-1]; wh=np.linalg.eigvalsh(GH)[::-1]
print(f"  {'rank(5%) of grade-2':<26}{rk(wm):>14}{rk(wh):>15}")
print(f"  {'eff-rank of grade-2':<26}{erank(wm):>14.2f}{erank(wh):>15.2f}")
print(f"  {'||grade-2|| / ||grade-1||':<26}{np.sqrt(np.trace(GM)/np.trace(G1)):>14.3f}"
      f"{np.sqrt(np.trace(GH)/np.trace(G1)):>15.3f}")
print("\n"+"="*74); print("  OBSTRUCTION: connected triple term  m_3"); print("="*74)
wt=np.linalg.eigvalsh(GT)[::-1]
print(f"  ||m_3|| / ||m_2||  = {np.sqrt(np.trace(GT)/np.trace(GH)):.3f}"
      f"      ||m_3|| / ||m_1|| = {np.sqrt(np.trace(GT)/np.trace(G1)):.3f}")
print(f"  rank(5%) of the 35 triples = {rk(wt)}   eff-rank {erank(wt):.2f}")
ct=closure(GT,GHT,GH)
print(f"  fraction of m_3 lying inside span(m_2): {100*ct.mean():.1f}%  "
      f"(min {100*ct.min():.1f}, max {100*ct.max():.1f})")
print("\n  interpretation: if ||m_3||/||m_2|| is small the expansion is converging;")
print("  if m_3 lies inside span(m_2) the tower closes at order 2.")
print("\n  time %.1fs"%(time.time()-t0))
