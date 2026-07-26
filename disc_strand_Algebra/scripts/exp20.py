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
P=off; NULL=7.0/P
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
def lg(v):
    setflat(v); model.eval(); out=[]
    with torch.no_grad():
        for x,y in XB: lo,_=model(x,y); out.append(lo.reshape(-1).clone())
    return torch.cat(out)
def Jt(v):
    setflat(th); model.eval(); model.zero_grad(set_to_none=True); i=0
    for x,y in XB:
        lo,_=model(x,y); n=lo.numel(); (lo.reshape(-1)*v[i:i+n]).sum().backward(); i+=n
    g=torch.cat([(p.grad.flatten() if p.grad is not None else torch.zeros(p.numel(),dtype=torch.float64))
                 for _,p in named]).clone()
    model.zero_grad(set_to_none=True); return g
D=[sub(DT,[g]) for g in NODES]
Dm=torch.stack(D); GD=(Dm@Dm.T).numpy(); GDi=np.linalg.pinv(GD)
L0=lg(th); pairs=list(itertools.combinations(range(7),2))
TS=[1.0,0.5,0.25]
M={}
for t in TS:
    f1=[lg(th+t*D[i])-L0 for i in range(7)]
    for (i,j) in pairs:
        M[(t,i,j)]=lg(th+t*D[i]+t*D[j])-L0-f1[i]-f1[j]
    print(f"  t={t} done ({time.time()-t0:.0f}s)", flush=True)
def inspan(mv):
    g=Jt(mv); c=(Dm@g).numpy()
    return float(c@GDi@c)/max(float(g@g),1e-300)
def report(lab,get):
    fr=np.array([inspan(get(i,j)) for (i,j) in pairs])
    print(f"  {lab:<34}{100*fr.mean():>9.2f}%{100*fr.min():>8.2f}{100*fr.max():>8.2f}{fr.mean()/NULL:>10.0f}x", flush=True)
    return fr
print("="*78); print("  THREE-POINT RICHARDSON: is the second-order verdict stable?"); print("="*78)
print(f"  A(t)=m_t/t^2 ;  2-pt: H=2A(t/2)-A(t) ;  3-pt: H=(1/3)A(t)-2A(t/2)+(8/3)A(t/4)")
print(f"  null = {100*NULL:.6f}%\n")
print(f"  {'estimator':<34}{'mean':>9}{'min':>8}{'max':>8}{'vs null':>10}")
report("raw m at t=1",            lambda i,j: M[(1.0,i,j)])
report("raw m at t=0.5",          lambda i,j: M[(0.5,i,j)])
report("raw m at t=0.25",         lambda i,j: M[(0.25,i,j)])
report("2-point Richardson",      lambda i,j: 8*M[(0.5,i,j)]-M[(1.0,i,j)])
report("3-point Richardson",      lambda i,j: (1/3)*M[(1.0,i,j)]-2*4*M[(0.5,i,j)]+(8/3)*16*M[(0.25,i,j)])
print("\n  scaling check (should be 4 for a pure 2nd-order term):")
for (i,j) in [(0,2),(3,4),(5,6)]:
    r1=float(M[(1.0,i,j)].norm()/M[(0.5,i,j)].norm()); r2=float(M[(0.5,i,j)].norm()/M[(0.25,i,j)].norm())
    print(f"    {NODES[i]}-{NODES[j]:<6} ||m(1)||/||m(.5)|| = {r1:.2f}   ||m(.5)||/||m(.25)|| = {r2:.2f}")
print("\n  time %.0fs"%(time.time()-t0))
