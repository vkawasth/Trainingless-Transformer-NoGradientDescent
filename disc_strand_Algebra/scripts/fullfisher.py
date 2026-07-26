"""
FULL-FISHER RESIDUAL: is the 36% missed energy an artifact of diagonal Fisher?
Fisher energy via JVPs (no diagonal approx):  u^T F u = (1/B) sum ||J u||^2
where J is the per-sample Jacobian of logits. Compare, for u vs reconstruction
a*sign(u), the fraction of TRUE Fisher energy retained.
"""
import time, gc, numpy as np, torch, torch.nn.functional as Fn
t0=time.time()
g_={}; src=open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
exec(src[:src.find("# ── PHASE 1")], g_)
model=g_["model"]; get_batch=g_["get_batch"]; LR=g_["LR"]
named=list(model.named_parameters()); params=[p for _,p in named]
def flat(): return torch.cat([p.data.flatten() for _,p in named]).clone()
P=flat().numel()
model.load_state_dict(torch.load("init.pt")); torch.manual_seed(17)
o=torch.optim.AdamW(model.parameters(), lr=LR*5, betas=(0.9,0.95), weight_decay=0.1)
for s in range(100):
    model.train(); x,y=get_batch(); _,l=model(x,y)
    o.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); o.step()
def unflat(v):
    out=[]; i=0
    for p in params:
        k=p.numel(); out.append(v[i:i+k].view_as(p)); i+=k
    return out
def fisher_energy(vec, nb=6):
    """(1/B) sum_over_samples ||sqrt(p) (J vec - mean)||^2 : true Fisher quadratic form.
       Uses JVP of log-softmax; Fisher = J^T (diag(p)-pp^T) J."""
    vs=unflat(vec); tot=0.0; cnt=0
    for _ in range(nb):
        x,_=get_batch()
        def f(*pl):
            i=0
            for p,pp in zip(params,pl): pass
            lo,_=model(x); return Fn.log_softmax(lo.reshape(-1,lo.shape[-1]),dim=-1)
        # jvp of logits wrt params along vec
        lo,_=model(x); logits=lo.reshape(-1,lo.shape[-1])
        p=Fn.softmax(logits,dim=-1).detach()
        (jvp,)=torch.autograd.functional.jvp(
            lambda *pl: (lambda L: L.reshape(-1,L.shape[-1]))(model_forward(pl,x)),
            tuple(params), tuple(vs), create_graph=False, strict=False)
        Jv=jvp                              # (T, V)
        mean=(p*Jv).sum(1,keepdim=True)
        quad=(p*(Jv-mean)**2).sum(1)        # per-token Fisher quadratic form
        tot+=float(quad.sum()); cnt+=1
    return tot/max(cnt,1)
def model_forward(pl,x):
    old=[p.data.clone() for p in params]
    with torch.no_grad():
        for p,v in zip(params,pl): p.data.copy_(v.data if hasattr(v,'data') else v)
    lo,_=model(x)
    with torch.no_grad():
        for p,v in zip(params,old): p.data.copy_(v)
    return lo
# simpler: finite-difference JVP of logits (avoids functional API issues)
def fisher_quad(vec, nb=8, eps=1e-3):
    tot=0.0
    th=flat()
    for _ in range(nb):
        x,_=get_batch()
        model.eval()
        with torch.no_grad():
            lo0,_=model(x); l0=Fn.log_softmax(lo0.reshape(-1,lo0.shape[-1]),1)
            p=l0.exp()
            i=0
            for pp in params:
                k=pp.numel(); pp.data.add_(eps*vec[i:i+k].view_as(pp)); i+=k
            lo1,_=model(x); l1=Fn.log_softmax(lo1.reshape(-1,lo1.shape[-1]),1)
            i=0
            for pp in params:
                k=pp.numel(); pp.data.copy_(th[i:i+k].view_as(pp)); i+=k
            dlog=(l1-l0)/eps                    # d log p along vec
            quad=(p*dlog*dlog).sum(1)-((p*dlog).sum(1))**2
            tot+=float(quad.sum())
        model.train()
    return tot/nb
# collect updates, measure true-Fisher retention of a*sign(u)
NT=1024; seg=(torch.arange(P)*NT//P).long(); cnt=torch.bincount(seg,minlength=NT).float()
prev=flat(); Rf=[]; Rd=[]
Gdiag=torch.zeros(P)
for _ in range(8):
    x,_=get_batch(); lo,_=model(x); lp=Fn.log_softmax(lo.reshape(-1,lo.shape[-1]),1)
    ys=torch.multinomial(lp.exp(),1).squeeze(1); model.zero_grad(set_to_none=True)
    Fn.nll_loss(lp,ys).backward()
    Gdiag+=torch.cat([(p.grad.flatten() if p.grad is not None else torch.zeros(p.numel())) for p in params])**2
Gdiag/=8
for s in range(20):
    model.train(); x,y=get_batch(); _,l=model(x,y)
    o.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    b4=flat(); o.step(); af=flat(); u=af-b4; sg=torch.sign(u)
    num=torch.zeros(NT).index_add_(0,seg,u*sg); den=torch.zeros(NT).index_add_(0,seg,sg*sg)
    rec=(num/(den+1e-30))[seg]*sg
    Fu=fisher_quad(u); Fr=fisher_quad(rec)
    Rf.append(Fr/max(Fu,1e-30))
    numd=float((Gdiag*rec*rec).sum()); dend=float((Gdiag*u*u).sum())
    Rd.append(numd/max(dend,1e-30))
    prev=af; del b4,af,u
    if s%5==4: gc.collect(); print(f"    step {s+1} ({time.time()-t0:.0f}s)", flush=True)
print("\n"+"="*72); print("  TRUE (full) FISHER vs DIAGONAL FISHER RETENTION of a*sign(u)"); print("="*72)
print(f"  full-Fisher   retention R = {np.mean(Rf):.4f}  +/- {np.std(Rf):.4f}")
print(f"  diagonal-Fisher retention = {np.mean(Rd):.4f}  +/- {np.std(Rd):.4f}")
print(f"\n  earlier diagonal measurement was 0.636.")
print(f"  if full ~ diagonal, the missing 36% is NOT a diagonal artifact (Possibility A dead).")
print(f"  if full >> diagonal, the residual lives in correlated directions.")
print(f"\n  time {time.time()-t0:.0f}s")
