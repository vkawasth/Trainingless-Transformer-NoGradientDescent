import time, itertools, numpy as np, torch
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
P=off
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
def lg(vec):
    setflat(vec); model.eval(); out=[]
    with torch.no_grad():
        for x,y in XB: lo,_=model(x,y); out.append(lo.reshape(-1).clone())
    return torch.cat(out)
def Jt(v):
    setflat(th); model.eval(); model.zero_grad(set_to_none=True); i=0
    for x,y in XB:
        lo,_=model(x,y); n=lo.numel(); (lo.reshape(-1)*v[i:i+n]).sum().backward(); i+=n
    g=torch.cat([(p.grad.flatten() if p.grad is not None else torch.zeros(p.numel()))
                 for _,p in named]).clone()
    model.zero_grad(set_to_none=True); return g
L0=lg(th)
pairs=list(itertools.combinations(range(7),2))
NULL=7.0/P
def run(gens,label):
    Dm=torch.stack(gens); GD=(Dm@Dm.T).double().numpy(); GDi=np.linalg.pinv(GD)
    f1=[lg(th+gens[i])-L0 for i in range(7)]
    fr=[]
    for (i,j) in pairs:
        m_ij=lg(th+gens[i]+gens[j])-L0-f1[i]-f1[j]
        g=Jt(m_ij); c=(Dm@g).double().numpy()
        fr.append(float(c@GDi@c)/max(float(g.double()@g.double()),1e-30))
    fr=np.array(fr)
    print(f"  {label:<34}{100*fr.mean():>8.3f}%{100*fr.min():>9.3f}{100*fr.max():>9.3f}"
          f"{fr.mean()/NULL:>12.0f}x", flush=True)
    return fr
print("="*80)
print("  CONTROL: does m~ stay aligned with INDEPENDENT generators?")
print("="*80)
print(f"  null (random direction, 7-dim span in R^{P:,}) = {100*NULL:.6f}%")
print(f"\n  {'generator set':<34}{'mean':>8} {'min':>8} {'max':>8}{'vs null':>12}")
D0=[sub(DT,[g]) for g in NODES]
run(D0,"A) node cuts of DT (original)")
rg=torch.Generator().manual_seed(20)
Dr=[]
for g in NODES:
    v=torch.zeros(P)
    for a,b in sl[g]: v[a:b]=torch.randn(b-a,generator=rg)
    Dr.append(v*(sub(DT,[g]).norm()/v.norm()))
run(Dr,"B) random dirs per node, |d| matched")
rg2=torch.Generator().manual_seed(21)
Dr2=[]
for g in NODES:
    v=torch.zeros(P)
    for a,b in sl[g]: v[a:b]=torch.randn(b-a,generator=rg2)
    Dr2.append(v*(sub(DT,[g]).norm()/v.norm()))
run(Dr2,"C) random dirs, second seed")
# independent displacement: 40->120 segment restricted per node
try:
    J40=torch.load("J40.pt"); J120=torch.load("J120.pt")
    D2=[sub(J120["th"]-J40["th"],[g]) for g in NODES]
    run(D2,"D) node cuts of an independent seg")
except Exception as e:
    print("  D) skipped:",e)
print("\n  time %.1fs"%(time.time()-t0))
