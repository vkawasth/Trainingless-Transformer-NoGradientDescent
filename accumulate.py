"""
What does one AdamW step actually SET in Phase 3?
Decompose each step into the part that accumulates and the part that cancels.

  coherence C = ||sum_t delta_t|| / sum_t ||delta_t||     (1 = pure drift, 0 = pure jitter)
  effective useful steps  N_eff = C^2 * N   (random walk: C ~ 1/sqrt(N))
Per block, and AdamW update vs raw gradient.
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
WARM=110; N=60
for _ in range(WARM):
    model.train(); x,y=get_batch(); _,l=model(x,y)
    opt.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
print(f"warmed {WARM}, val={eval_val(model,n=4):.4f}", flush=True)

sumd={g:None for g in GR}; sumn={g:0.0 for g in GR}      # AdamW displacement
sumg={g:None for g in GR}; sumgn={g:0.0 for g in GR}     # raw gradient
prev={n:p.data.flatten().clone() for n,p in named}
for s in range(N):
    model.train(); x,y=get_batch(); _,l=model(x,y)
    opt.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    gcat={g:[] for g in GR}
    for n,p in named:
        if p.grad is not None: gcat[grp(n)].append(p.grad.flatten().clone())
    opt.step()
    dcat={g:[] for g in GR}
    for n,p in named:
        d=p.data.flatten()-prev[n]; dcat[grp(n)].append(d.clone()); prev[n]=p.data.flatten().clone()
    for g in GR:
        if dcat[g]:
            dv=torch.cat(dcat[g]); sumd[g]=dv if sumd[g] is None else sumd[g]+dv; sumn[g]+=float(dv.norm())
        if gcat[g]:
            gv=torch.cat(gcat[g]); sumg[g]=gv if sumg[g] is None else sumg[g]+gv; sumgn[g]+=float(gv.norm())

print("\n"+"="*76)
print(f"  WHAT ACCUMULATES?  {N} AdamW steps from a settled checkpoint")
print("="*76)
print(f"  {'block':<11}{'params':>10}{'C(update)':>11}{'C(grad)':>10}{'N_eff':>8}{'ratio':>8}")
print("  "+"-"*60)
rows={}
for g in GR:
    if sumd[g] is None: continue
    C=float(sumd[g].norm())/ (sumn[g]+1e-12)
    Cg=float(sumg[g].norm())/(sumgn[g]+1e-12)
    npar=sum(p.numel() for n,p in named if grp(n)==g)
    rows[g]=dict(C=C,Cg=Cg,n=npar,Neff=C*C*N)
    print(f"  {g:<11}{npar:>10,}{C:>11.3f}{Cg:>10.3f}{C*C*N:>8.1f}{C/(Cg+1e-12):>8.2f}")
Ctot=float(torch.cat([sumd[g] for g in GR if sumd[g] is not None]).norm())/ (sum(sumn.values())+1e-12)
print(f"\n  WHOLE MODEL coherence C = {Ctot:.4f}   -> N_eff = {Ctot**2*N:.2f} of {N} steps")
print(f"  random-walk null C = 1/sqrt(N) = {1/np.sqrt(N):.4f}")
json.dump({g:rows[g] for g in rows}, open("accum.json","w"), indent=1, default=float)
