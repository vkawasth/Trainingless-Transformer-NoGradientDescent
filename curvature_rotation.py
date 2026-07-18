"""
Does CURVATURE decide the rotation of the drift?
For gradient flow  v_dot = -H v.  Split  Hv = lambda*v + r,  r perp v:
   lambda = <v,Hv>/||v||^2   (curvature ALONG drift -> speed change)
   omega  = ||r||/||v||      (TRANSVERSE curvature -> rotation rate)
Prediction: blocks with larger omega/|lambda| decohere faster (more negative alpha).
"""
import numpy as np, torch, json
g_={}; src=open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
exec(src[:src.find("# ── PHASE 1")], g_)
model=g_["model"]; get_batch=g_["get_batch"]; eval_val=g_["eval_val"]; LR=g_["LR"]
def grp(n):
    n=n.lower()
    if ".ln." in n or ".n." in n or n.startswith("ln_f"): return "LayerNorm"
    if n.startswith("te") or n.startswith("pe") or n.startswith("head"): return "Emb"
    if ".ff." in n: return "FF"
    if "wk" in n: return "W_K"
    if "wq" in n: return "W_Q"
    if "wv" in n: return "W_V"
    if ".op." in n: return "W_O"
    return "other"
GR=["Emb","W_Q","W_K","W_V","W_O","FF","LayerNorm"]
named=[(n,p) for n,p in model.named_parameters()]
params=[p for _,p in named]
sizes=[p.numel() for p in params]
masks={}
off=0; idxmap=[]
for (n,p) in named: idxmap.append((grp(n),off,off+p.numel())); off+=p.numel()
P=off
for g in GR:
    m=torch.zeros(P,dtype=torch.bool)
    for gg,a,b in idxmap:
        if gg==g: m[a:b]=True
    masks[g]=m

opt=torch.optim.AdamW(model.parameters(), lr=LR*5, betas=(0.9,0.95), weight_decay=0.1)
for _ in range(100):
    model.train(); x,y=get_batch(); _,l=model(x,y)
    opt.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
print(f"warmed, val={eval_val(model,n=4):.4f}", flush=True)

# drift direction = mean displacement over next 10 steps
prev=torch.cat([p.data.flatten() for p in params]).clone()
for _ in range(10):
    model.train(); x,y=get_batch(); _,l=model(x,y)
    opt.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
v=(torch.cat([p.data.flatten() for p in params])-prev)
v=v/ (v.norm()+1e-12)
print("drift direction captured", flush=True)

# Hessian-vector product H v  (averaged over a few batches)
def hvp(vec, nb=3):
    acc=torch.zeros(P)
    for _ in range(nb):
        x,y=get_batch(); _,l=model(x,y)
        gr=torch.autograd.grad(l, params, create_graph=True)
        gf=torch.cat([g.reshape(-1) for g in gr])
        gv=(gf*vec).sum()
        Hg=torch.autograd.grad(gv, params, retain_graph=False)
        acc+=torch.cat([h.reshape(-1) for h in Hg]).detach()
    model.zero_grad(set_to_none=True)
    return acc/nb
Hv=hvp(v)
print("HVP computed", flush=True)

alpha_meas={"Emb":-0.13,"LayerNorm":-0.16,"W_Q":-0.21,"W_K":-0.21,
            "W_V":-0.25,"W_O":-0.25,"FF":-0.27}
print("\n"+"="*78); print("  DOES CURVATURE DECIDE THE ROTATION?"); print("="*78)
print(f"  {'block':<11}{'lambda':>11}{'omega':>11}{'omega/|lam|':>13}{'alpha(meas)':>13}")
print("  "+"-"*66)
rows={}
for g in GR:
    m=masks[g]; vg=v[m]; hg=Hv[m]
    nv=float(vg.norm())
    if nv<1e-12: continue
    lam=float((vg*hg).sum()/(nv**2))
    r=hg-lam*vg
    om=float(r.norm()/nv)
    rows[g]=dict(lam=lam,omega=om,ratio=om/(abs(lam)+1e-12),alpha=alpha_meas[g])
    print(f"  {g:<11}{lam:>11.2f}{om:>11.2f}{om/(abs(lam)+1e-12):>13.2f}{alpha_meas[g]:>13.2f}")

xs=np.array([rows[g]["ratio"] for g in rows]); ys=np.array([rows[g]["alpha"] for g in rows])
from scipy.stats import spearmanr, pearsonr
rs,ps=spearmanr(xs,ys); rp,pp=pearsonr(np.log(xs+1e-9),ys)
print(f"\n  Spearman(omega/|lambda| , alpha) = {rs:+.3f}  (p={ps:.3f})")
print(f"  Pearson(log ratio, alpha)        = {rp:+.3f}  (p={pp:.3f})")
print("  prediction: MORE transverse curvature -> MORE negative alpha -> rho < 0")
json.dump(rows, open("curv_rot.json","w"), indent=1, default=float)
