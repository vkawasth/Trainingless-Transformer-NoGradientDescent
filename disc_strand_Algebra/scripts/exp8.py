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
D={g: sub(DT,[g]) for g in NODES}
df={g: (logits(th+D[g])-L0).double() for g in NODES}
B=torch.stack([df[g] for g in NODES])            # (7,N)
G=(B@B.T).numpy()                                 # Gram, real inner products
nrm=np.sqrt(np.diag(G))
print("="*76); print("  PANTS PRODUCT  m(i,j) = df(d_i+d_j) - df(d_i) - df(d_j)"); print("="*76)
X={}; 
for i,j in itertools.combinations(range(7),2):
    a,b=NODES[i],NODES[j]
    X[(i,j)]=(logits(th+D[a]+D[b])-L0).double()-df[a]-df[b]
    X[(j,i)]=X[(i,j)]
print("  collected 21 pair cross-terms  (%.0fs)"%(time.time()-t0), flush=True)
# --- does the product close on span{df_k}? ---
Gi=np.linalg.pinv(G)
print("\n  (1) CLOSURE: is m(i,j) inside span{df_k}?")
print(f"  {'pair':>12}{'||m||':>10}{'||m|| / ||df_i||||df_j||^0.5':>14}{'in-span %':>12}")
C=np.zeros((7,7,7)); clos=[]
for i,j in itertools.combinations(range(7),2):
    x=X[(i,j)]; bx=(B@x).numpy()
    c=Gi@bx; C[i,j]=c; C[j,i]=c
    proj=float(np.dot(c,bx)); tot=float(x@x)
    frac=proj/max(tot,1e-30); clos.append(frac)
    if (i,j) in [(3,4),(5,6),(0,1),(2,3),(0,5)]:
        print(f"  {NODES[i]+'-'+NODES[j]:>12}{float(x.norm()):>10.1f}"
              f"{float(x.norm())/np.sqrt(nrm[i]*nrm[j]):>14.2f}{100*frac:>11.1f}%")
print(f"  mean in-span fraction over all 21 pairs: {100*np.mean(clos):.1f}%  "
      f"(min {100*min(clos):.1f}%, max {100*max(clos):.1f}%)")
# --- bilinearity of the product ---
print("\n  (2) BILINEARITY: m(td_i,td_j) should scale as t^2")
for (i,j) in [(3,4),(5,6),(0,2)]:
    a,b=NODES[i],NODES[j]
    xh=(logits(th+0.5*D[a]+0.5*D[b])-L0).double()-(logits(th+0.5*D[a])-L0).double()-(logits(th+0.5*D[b])-L0).double()
    r=float(X[(i,j)].norm())/max(4*float(xh.norm()),1e-9)
    print(f"    {a}-{b}: ||m(1,1)|| / 4||m(.5,.5)|| = {r:.3f}   (1.000 = exactly bilinear)")
# --- Frobenius condition <m(i,j),k> == <i,m(j,k)> ---
print("\n  (3) FROBENIUS CONDITION  <m(i,j),e_k> == <e_i,m(j,k)>")
L=[];R=[]
for i,j,k in itertools.permutations(range(7),3):
    if i<k:
        lhs=float(X[(i,j)]@df[NODES[k]])/(nrm[i]*nrm[j]*nrm[k])
        rhs=float(df[NODES[i]]@X[(j,k)])/(nrm[i]*nrm[j]*nrm[k])
        L.append(lhs); R.append(rhs)
L=np.array(L); R=np.array(R)
rel=np.linalg.norm(L-R)/max(np.linalg.norm(L),1e-30)
print(f"    n = {len(L)} triples;  corr = {np.corrcoef(L,R)[0,1]:+.4f}")
print(f"    relative residual ||LHS-RHS||/||LHS|| = {rel:.4f}")
print(f"    scale: mean|LHS| = {np.mean(np.abs(L)):.4f}, mean|LHS-RHS| = {np.mean(np.abs(L-R)):.4f}")
# --- associativity from structure constants ---
print("\n  (4) ASSOCIATIVITY of the structure constants")
A1=np.einsum('ijm,mkl->ijkl',C,C); A2=np.einsum('jkm,iml->ijkl',C,C)
print(f"    ||assoc defect|| / ||product|| = {np.linalg.norm(A1-A2)/max(np.linalg.norm(A1),1e-30):.4f}")
# --- semisimplicity: block idempotent structure ---
print("\n  (5) BLOCK / IDEMPOTENT STRUCTURE  (unit-normalised Gram eigenbasis)")
Gc=G/np.outer(nrm,nrm)
w,U=np.linalg.eigh(Gc); idx=np.argsort(w)[::-1]; w=w[idx]; U=U[:,idx]
print("    Gram eigenvalues:", " ".join(f"{x:.3f}" for x in w))
for r in range(4):
    v=U[:,r]; dom=np.argsort(-np.abs(v))[:3]
    print(f"    mode {r}: theta={w[r]:.3f}  " + "  ".join(f"{NODES[d]}:{v[d]:+.2f}" for d in dom))
print("\n  time %.1fs"%(time.time()-t0))
