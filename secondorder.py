"""IS THE ASCENT LEVEL-SET CURVATURE? A ZERO-PARAMETER TEST.

u_perp is orthogonal to g BY CONSTRUCTION, so the first-order term vanishes
exactly:  <g, u_perp> = 0.  The measured dL_perp = +0.0059, positive at all six
checkpoints and 68x a norm-matched random direction, therefore cannot be a
first-order effect. It must be second order:

    dL_perp  ~  (1/2) u_perp^T H u_perp

Geometrically: u_perp is tangent to the level set {L = const}, which is a
codimension-1 submanifold. The loss rises because that submanifold CURVES -- the
second fundamental form along the direction of travel. Positive dL_perp means
positive curvature: the trajectory rides a valley wall.

This is the first equation in the programme with NO FITTED COEFFICIENTS. Every
earlier attempt -- exponential, two-term Fisher, replicator, fuel depletion --
had a constant standing in for exactly this curvature, which is why their
coefficients ran from -362 to +90 and one predicted a negative loss.

BOTH SIDES ARE MEASURED:
  left   dL_perp by ablation, on a FIXED eval set so L() is deterministic
  right  u_perp^T H u_perp by Pearlmutter double-backward, no approximation

CONTROLS
  <g,u_perp>/(|g||u_perp|)   must be ~0, or the split is wrong and there is a
                             first-order term contaminating the comparison
  the same test on u_par     where the first-order term DOES dominate, so the
                             ratio should be far from 1 -- confirming the test
                             can tell the two cases apart
  a random direction of the same norm, for the curvature scale
"""
import json, subprocess, numpy as np, torch, io, contextlib, math
subprocess.run(["python3","/mnt/user-data/uploads/build_corpus.py","--out","/tmp",
                "--loops","300"],check=True,capture_output=True)
RAW=open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
CUT=RAW.find("# \u2500\u2500 PHASE 3")
src=RAW[:CUT].replace("D=256; N_HEADS=4","D=128; N_HEADS=4",1)
src=src.replace("for mf_r in range(1, 16):","for mf_r in range(1, 3):",1)
src=src.replace("    if pc == N_STU-1:","    if False:",1)
src=src.replace("    if len(tau_history)>=3 and tau > tau_history[-2] > tau_history[-3]:","    if False:",1)
G={}; b=io.StringIO()
with contextlib.redirect_stdout(b): exec(src,G)
model=G["model"]; get_batch=G["get_batch"]; LR=G["LR"]
ps=[p for p in model.parameters() if p.requires_grad]
P=sum(p.numel() for p in ps)
EV=[get_batch() for _ in range(12)]          # FIXED -> deterministic L and H
def flat(): return torch.cat([p.data.reshape(-1) for p in ps]).clone()
def setth(t):
    with torch.no_grad():
        j=0
        for p in ps:
            q=p.numel(); p.data.copy_(t[j:j+q].view_as(p)); j+=q
def L():
    t=0.0
    with torch.no_grad():
        for x,y in EV: t+=float(model(x,y)[1])
    return t/len(EV)
def gfull():
    acc=torch.zeros(P)
    for x,y in EV:
        model.zero_grad(); _,l=model(x,y); l.backward()
        acc+=torch.cat([(p.grad.reshape(-1) if p.grad is not None
                         else torch.zeros(p.numel())) for p in ps])
    model.zero_grad()
    return acc/len(EV)
def quad(v):
    """v^T H v by Pearlmutter, on the same fixed set"""
    acc=0.0
    for x,y in EV:
        model.zero_grad()
        _,l=model(x,y)
        gr=torch.autograd.grad(l,ps,create_graph=True)
        gf=torch.cat([t.reshape(-1) for t in gr])
        hv=torch.autograd.grad((gf*v).sum(),ps,allow_unused=True)
        hvf=torch.cat([(t if t is not None else torch.zeros_like(p)).reshape(-1)
                       for t,p in zip(hv,ps)]).detach()
        acc+=float(hvf@v)
    model.zero_grad()
    return acc/len(EV)
opt=torch.optim.AdamW(model.parameters(),lr=LR*5,betas=(0.9,0.95),weight_decay=0.1)
print(f"  P={P:,}   deterministic eval and Hessian on {len(EV)} fixed batches\n")
print(f"  {'step':>5}{'L':>8}{'cos(g,uperp)':>14}{'dL_perp':>11}"
      f"{'0.5 uHu':>11}{'ratio':>8}{'dL_par':>11}{'0.5 uHu par':>13}{'ratio':>8}")
step=0; rows=[]
for ck in (24,48,72,96,120):
    while step<ck:
        x,y=get_batch(); _,l=model(x,y); opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); step+=1
    th0=flat(); L0=L()
    g=gfull(); setth(th0)
    gh=-g/(g.norm()+1e-30)
    x,y=get_batch(); _,l=model(x,y); opt.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    u=flat()-th0; setth(th0)
    upar=float(u@gh)*gh; uperp=u-upar
    cs=float(g@uperp)/(float(g.norm())*float(uperp.norm())+1e-30)
    setth(th0+uperp); dperp=L()-L0; setth(th0)
    setth(th0+upar);  dpar =L()-L0; setth(th0)
    q_perp=0.5*quad(uperp); setth(th0)
    q_par =0.5*quad(upar);  setth(th0)
    gen=torch.Generator().manual_seed(step)
    r=torch.randn(P,generator=gen); r=r/r.norm()*uperp.norm()
    setth(th0+r); drand=L()-L0; setth(th0)
    q_rand=0.5*quad(r); setth(th0)
    rp=dperp/q_perp if abs(q_perp)>1e-12 else float('nan')
    ra=dpar/q_par if abs(q_par)>1e-12 else float('nan')
    print(f"  {ck:>5}{L0:>8.4f}{cs:>14.2e}{dperp:>11.6f}{q_perp:>11.6f}{rp:>8.3f}"
          f"{dpar:>11.6f}{q_par:>13.6f}{ra:>8.3f}",flush=True)
    rows.append(dict(ck=ck,L=L0,cos=cs,dperp=dperp,q_perp=q_perp,
                     dpar=dpar,q_par=q_par,drand=drand,q_rand=q_rand))
json.dump(rows,open("/home/claude/work/res_secondorder.json","w"),indent=2)
rp=np.array([r["dperp"]/r["q_perp"] for r in rows if abs(r["q_perp"])>1e-12])
ra=np.array([r["dpar"]/r["q_par"] for r in rows if abs(r["q_par"])>1e-12])
print(f"\n  PERP  ratio dL / (0.5 uHu):  mean {rp.mean():.3f}  sd {rp.std():.3f}")
print(f"  PAR   ratio dL / (0.5 uHu):  mean {ra.mean():.3f}  sd {ra.std():.3f}")
print(f"  random: dL {np.mean([r['drand'] for r in rows]):+.6f}  "
      f"0.5uHu {np.mean([r['q_rand'] for r in rows]):+.6f}")
print(f"\n  perp ratio ~ 1 => the ascent IS level-set curvature, zero parameters")
print(f"  par ratio far from 1 => the test discriminates: there the first-order")
print(f"     term dominates, exactly as it should")
