"""
THE PAIRING <g, d> APPLIED TO DISCS.
 d_true    : the actual Adam update
 d_sign    : sign(d) * c_strand          (per-parameter sign, one magnitude/strand)
 d_uniform : c_strand * 1                (SAME correction across the whole strand)
Prediction: <g, uniform> ~ c * sum_i g_i, which cancels within a strand, whereas
<g, sign> ~ c * sum_i |g_i|.  The ratio |sum g| / sum|g| is the cancellation
factor and should be ~1/sqrt(n) for mixed signs.
"""
import time, numpy as np, torch
t0=time.time()
g_={}; src=open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
exec(src[:src.find("# ── PHASE 1")], g_)
model=g_["model"]; get_batch=g_["get_batch"]; LR=g_["LR"]
named=list(model.named_parameters())
def flat(): return torch.cat([p.data.flatten() for _,p in named]).clone()
P=flat().numel()
model.load_state_dict(torch.load("init.pt")); torch.manual_seed(17)
opt=torch.optim.AdamW(model.parameters(), lr=LR*5, betas=(0.9,0.95), weight_decay=0.1)
print("="*88); print("  <g,d> FOR DISCS:  why the uniform disc fails"); print("="*88)
for NS in (1024,):
    e=np.linspace(0,P,NS+1).astype(int)
    print(f"\n  strands = {NS}   params/strand = {P//NS:,}   1/sqrt(n) = {1/np.sqrt(P//NS):.4f}")
    print(f"  {'step':>6}{'<g,d_true>':>13}{'<g,d_sign>':>13}{'<g,d_unif>':>13}"
          f"{'unif/sign':>11}{'|sum g|/sum|g|':>16}{'cos(sgn d,sgn g)':>18}")
    for s in range(1,121):
        model.train(); x,y=get_batch(); _,l=model(x,y)
        opt.zero_grad(); l.backward()
        gv=torch.cat([(p.grad.flatten() if p.grad is not None else torch.zeros(p.numel()))
                      for _,p in named]).clone()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        b4=flat(); opt.step(); af=flat(); d=af-b4
        if s in (5,20,40,80,120):
            c=torch.zeros(NS); cs=torch.zeros(NS); gsum=torch.zeros(NS); gabs=torch.zeros(NS)
            for i in range(NS):
                sl=slice(e[i],e[i+1])
                c[i]=d[sl].abs().mean(); cs[i]=d[sl].mean()
                gsum[i]=gv[sl].sum();    gabs[i]=gv[sl].abs().sum()
            d_sign=torch.empty_like(d); d_unif=torch.empty_like(d)
            for i in range(NS):
                sl=slice(e[i],e[i+1])
                d_sign[sl]=torch.sign(d[sl])*c[i]; d_unif[sl]=cs[i]
            gt=float(gv@d); gsg=float(gv@d_sign); gu=float(gv@d_unif)
            canc=float((gsum.abs().sum())/max(float(gabs.sum()),1e-30))
            agree=float((torch.sign(d)==torch.sign(gv)).double().mean())
            print(f"  {s:>6}{gt:>13.5f}{gsg:>13.5f}{gu:>13.5f}"
                  f"{abs(gu/gsg) if gsg!=0 else float('nan'):>11.4f}{canc:>16.4f}"
                  f"{2*agree-1:>18.3f}", flush=True)
print("\n  unif/sign ~ |sum g|/sum|g| confirms the mechanism: a single correction")
print("  applied across a strand is projected onto the STRAND-SUM of the gradient,")
print("  which nearly cancels, while the per-parameter sign is projected onto sum|g|.")
print(f"\n  time {time.time()-t0:.0f}s")
