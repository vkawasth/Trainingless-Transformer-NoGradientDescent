import time, numpy as np, torch
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
torch.manual_seed(3); XB=[get_batch() for _ in range(6)]
def logits(vec):
    setflat(vec); model.eval(); out=[]
    with torch.no_grad():
        for x,y in XB: lo,_=model(x,y); out.append(lo.reshape(-1).clone())
    return torch.cat(out)
L0=logits(th)
DF={g: logits(th+sub(DT,[g]))-L0 for g in NODES}
DFall=logits(th+DT)-L0
S=sum(DF[g] for g in NODES)
print("="*74); print("  IS THE NODE DECOMPOSITION LINEAR IN FUNCTION SPACE?"); print("="*74)
print(f"  ||df(full jump)||            = {float(DFall.norm()):.3f}")
print(f"  ||sum_g df(node g)||         = {float(S.norm()):.3f}")
print(f"  ||df(full) - sum df(node)||  = {float((DFall-S).norm()):.3f}"
      f"   = {100*float((DFall-S).norm()/DFall.norm()):.1f}% of ||df(full)||")
print(f"  cos(df(full), sum df(node))  = {float(DFall@S/(DFall.norm()*S.norm())):.4f}")
# half-scale linearity check (is a single node's own action linear?)
print("\n  per-node linearity: ||df(1.0*d_g)|| vs 2*||df(0.5*d_g)||")
print(f"  {'node':>6}{'||df(1.0)||':>13}{'2||df(0.5)||':>14}{'ratio':>8}")
for g in NODES:
    a=float(DF[g].norm())
    h=logits(th+0.5*sub(DT,[g]))-L0
    print(f"  {g:>6}{a:>13.3f}{2*float(h.norm()):>14.3f}{a/(2*float(h.norm())+1e-9):>8.3f}")
print("\n"+"="*74); print("  GRAM MATRIX OF FUNCTIONAL CHANGES  cos(df_i, df_j)"); print("="*74)
C=np.zeros((7,7))
for i,a in enumerate(NODES):
    for j,b in enumerate(NODES):
        C[i,j]=float(DF[a]@DF[b]/(DF[a].norm()*DF[b].norm()+1e-12))
print("        "+"".join(f"{g:>8}" for g in NODES))
for i,g in enumerate(NODES):
    print(f"  {g:>6}"+"".join(f"{C[i,j]:>8.3f}" for j in range(7)))
w=np.linalg.eigvalsh(C)[::-1]
print(f"\n  eigenvalues of the cosine Gram: {' '.join(f'{x:.3f}' for x in w)}")
print(f"  effective rank = {(w.sum()**2)/(w**2).sum():.2f} out of 7")
print(f"  ||df_g|| per node: " + "  ".join(f"{g}:{float(DF[g].norm()):.2f}" for g in NODES))
SYN=np.load("SYN.npy")
off=np.triu_indices(7,1)
r=np.corrcoef(C[off],SYN[off])[0,1]
print(f"\n  corr( functional overlap cos(df_i,df_j) , recovery synergy S_ij ) = {r:+.3f}")
print("  time %.1fs"%(time.time()-t0))
