"""
Is Phase 3 motion BALLISTIC (integrating toward something) or DIFFUSIVE
(low-pass filtered noise with a drift)?

  C(W) = ||sum_{t in W} d_t|| / sum_{t in W} ||d_t||
  ballistic drift  -> C(W) ~ const
  diffusive        -> C(W) ~ W^{-1/2}

Computed for all window lengths via the Gram matrix G[t,s]=<d_t,d_s>, using a
fixed random coordinate subsample per block (unbiased for inner products).
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
opt=torch.optim.AdamW(model.parameters(), lr=LR*5, betas=(0.9,0.95), weight_decay=0.1)
WARM,N,K=100,64,20000
for _ in range(WARM):
    model.train(); x,y=get_batch(); _,l=model(x,y)
    opt.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
print(f"warmed {WARM}, val={eval_val(model,n=4):.4f}", flush=True)

dims={g:sum(p.numel() for n,p in named if grp(n)==g) for g in GR}
rng=np.random.default_rng(0)
idx={g:torch.tensor(rng.choice(dims[g], size=min(K,dims[g]), replace=False)) for g in GR}
D={g:[] for g in GR}
prev={n:p.data.flatten().clone() for n,p in named}
for s in range(N):
    model.train(); x,y=get_batch(); _,l=model(x,y)
    opt.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    cat={g:[] for g in GR}
    for n,p in named:
        d=p.data.flatten()-prev[n]; cat[grp(n)].append(d); prev[n]=p.data.flatten().clone()
    for g in GR:
        v=torch.cat(cat[g]); D[g].append(v[idx[g]].numpy().copy())
print("recorded", N, "steps", flush=True)

def C_of_W(G,W):
    n=G.shape[0]; out=[]
    for a in range(0,n-W+1):
        sub=G[a:a+W,a:a+W]
        num=np.sqrt(max(sub.sum(),0)); den=np.sqrt(np.diag(sub)).sum()
        out.append(num/(den+1e-12))
    return float(np.mean(out))

Ws=[2,4,8,16,32,64]
print("\n"+"="*74); print("  BALLISTIC OR DIFFUSIVE?   C(W) vs window length"); print("="*74)
print(f"  {'block':<11}" + "".join(f"{'W='+str(w):>9}" for w in Ws) + f"{'alpha':>9}  verdict")
print("  "+"-"*72)
res={}
for g in GR:
    X=np.stack(D[g]); G=X@X.T
    cs=[C_of_W(G,w) for w in Ws]
    a,b=np.polyfit(np.log(Ws), np.log(cs), 1)      # C ~ W^a
    verdict = "BALLISTIC" if a>-0.15 else ("diffusive" if a<-0.4 else "intermediate")
    res[g]=dict(C=cs,alpha=float(a))
    print(f"  {g:<11}" + "".join(f"{c:>9.3f}" for c in cs) + f"{a:>9.2f}  {verdict}")
print(f"\n  reference: pure diffusion alpha=-0.50   pure drift alpha=0.00")
json.dump(res, open("ballistic.json","w"), indent=1)
