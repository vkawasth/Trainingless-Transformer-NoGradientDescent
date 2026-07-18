"""
manifold_rank.py
================
Does a LOW-DIMENSIONAL manifold carry the descent? (strong sense: closed-form-able)
Test the trajectory directly, no Hessian:
  1. from a basin checkpoint, run T GD steps, store displacements d_t = θ_t-θ_{t-1}
  2. effective rank of the T×P displacement matrix (SVD via the T×T Gram)
  3. closed-form-jump test: θ_0 + P_k(Δθ) for small k -- does a k-dim jump reach
     the same loss as the full T-step descent?
STRONG manifold  -> tiny eff-rank, small-k jump recovers the loss  (GD dispensable)
WEAK/isotropic   -> eff-rank ~ T, need many k                      (GD required)
"""
import math
import numpy as np, torch
g_={}; src=open("compiler_geometri_patched_86.py").read()
exec(src[:src.find("# ── PHASE 1")], g_)
model=g_["model"]; get_batch=g_["get_batch"]; eval_val=g_["eval_val"]; LR=g_["LR"]

def flat():  return torch.cat([p.data.flatten() for p in model.parameters()]).clone()
def val(n=8): return float(eval_val(model,n=n))

def main(warm=60, T=40):
    opt=torch.optim.AdamW(model.parameters(), lr=LR*5, betas=(0.9,0.95), weight_decay=0.1)
    print("="*74); print("  DOES A LOW-DIM MANIFOLD CARRY THE DESCENT? (trajectory rank)"); print("="*74)
    for _ in range(warm):
        model.train(); x,y=get_batch(); _,l=model(x,y)
        opt.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    theta0=flat(); v0=val()
    P=theta0.numel()
    print(f"  checkpoint: warm={warm}  P={P:,}  val0={v0:.4f}")
    # collect T displacements
    D=[]
    prev=theta0.clone()
    for t in range(T):
        model.train(); x,y=get_batch(); _,l=model(x,y)
        opt.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        cur=flat(); D.append((cur-prev).cpu()); prev=cur
    vT=val(); thetaT=flat()
    Dm=torch.stack(D)                      # (T,P) on CPU
    Delta=(thetaT.cpu()-theta0.cpu())      # total displacement

    # eff rank via Gram
    G=(Dm@Dm.T).double().numpy()           # (T,T)
    w,V=np.linalg.eigh(G); w=np.clip(w[::-1],0,None); V=V[:,::-1]
    s=np.sqrt(w)                            # singular values of Dm
    p=w/ (w.sum()+1e-30)
    eff=float(np.exp(-(p[p>0]*np.log(p[p>0])).sum()))
    c90=int(np.searchsorted(np.cumsum(p),0.90)+1)
    print(f"\n  step displacements: T={T}")
    print(f"  effective rank of the descent = {eff:.1f} / {T}   "
          f"({c90} directions carry 90% of the step energy)")
    print(f"  singular spectrum (norm): "+" ".join(f"{si/s[0]:.2f}" for si in s[:8])+" ...")

    # right singular vectors in P-space handled implicitly:
    #   U_i = Dm^T V_i / s_i ;  a_i = Delta·U_i = (Dm Delta)·V_i / s_i
    #   P_k Delta = Dm^T ( sum_{i<k} (a_i/s_i) V_i )   -- all in T-space, memory-lean
    b=(Dm@Delta).double().numpy()          # (T,)
    Vt=V; st=s
    print(f"\n  {'k':>3}{'||P_kD||/||D||':>14}{'jump val':>10}{'loss recovered':>16}")
    print("  "+"-"*46)
    for k in [1,2,3,5,10,min(20,T),T]:
        c=np.zeros(T)
        for i in range(k):
            if st[i]<1e-9: break
            a_i=float(Vt[:,i]@b)/st[i]
            c+=(a_i/st[i])*Vt[:,i]
        PkD=(Dm.T@torch.tensor(c,dtype=torch.float32)).float()   # (P,)
        frac=float(PkD.norm()/(Delta.norm()+1e-12))
        newp=(theta0.cpu()+PkD)
        i0=0
        for pm in model.parameters():
            n=pm.numel(); pm.data.copy_(newp[i0:i0+n].view_as(pm).to(pm.device)); i0+=n
        vk=val()
        rec=(v0-vk)/(v0-vT+1e-12)
        print(f"  {k:>3}{frac:>14.3f}{vk:>10.4f}{100*rec:>15.0f}%")
    # restore
    i0=0
    for pm in model.parameters():
        n=pm.numel(); pm.data.copy_(thetaT[i0:i0+n].view_as(pm)); i0+=n

    print(f"\n  full T-step descent: val {v0:.4f} -> {vT:.4f}")
    print("\n  READING: if a small k recovers ~100% of the loss drop, the descent")
    print("  lives on a low-dim manifold and is closed-form-jumpable. If recovery")
    print("  needs k ~ T, the descent is high-dim and GD is not dispensable.")

if __name__=="__main__":
    import sys
    main(warm=int(sys.argv[1]) if len(sys.argv)>1 else 60,
         T=int(sys.argv[2]) if len(sys.argv)>2 else 40)
