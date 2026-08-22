"""DETERMINISTIC CURVATURE: SAME BATCH FOR BOTH CONSECUTIVE UPDATES.

The previous run estimated dT/dt from two consecutive STOCHASTIC updates, so it
inherited batch noise on both. The consequences were visible: cos(u,-g) jumped
to 0.3035 at step 190 against ~0.10 elsewhere, the ratio |kappa_g|/|II| scattered
1.15, 1.54, 1.47, 1.18, 1.19 with no trend, and a perturbation control with ZERO
displacement moved as much as the kick (1.17..1.87 against 0.68..1.84). Nothing
could be read from that.

Here the two consecutive updates use the SAME frozen batch, and so does the
Hessian. Then the difference between them is trajectory curvature, not two
different gradient samples. Everything is deterministic given theta.

    T   = u1 / |u1|          u1 from batch B at theta_0
    T'  = u2 / |u2|          u2 from batch B at theta_1 = theta_0 + u1
    dT  = T' - T
    kappa_g = tangential part of dT, with the along-track component removed
    II(T,T) = T_tan^T H T_tan / |g|,  H and g also on B

WHAT THE RATIO DISTINGUISHES
    R >> 1   active steering: the trajectory bends across the level set
    R ~ 1    equipartition: intrinsic and extrinsic bending are equal
    R -> 0   sliding: kappa_g vanishes and the path is a geodesic of the level
             set, driven only by the surface

CONTROLS
    the same measurement with a DIFFERENT batch for the second update, which
    should reproduce the earlier scatter and so demonstrate that the noise was
    the cause rather than the geometry
    II for a random tangent, for scale
    cos(u,-g) reported alongside, since its stability is the tell that the
    estimate is clean
"""
import json, subprocess, numpy as np, torch, io, contextlib, math, copy
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
def flat(): return torch.cat([p.data.reshape(-1) for p in ps]).clone()
def setth(t):
    with torch.no_grad():
        j=0
        for p in ps:
            q=p.numel(); p.data.copy_(t[j:j+q].view_as(p)); j+=q
def gB(B):
    acc=torch.zeros(P)
    for x,y in B:
        model.zero_grad(); _,l=model(x,y); l.backward()
        acc+=torch.cat([(p.grad.reshape(-1) if p.grad is not None
                         else torch.zeros(p.numel())) for p in ps])
    model.zero_grad(); return acc/len(B)
def HvB(v,B):
    acc=torch.zeros(P)
    for x,y in B:
        model.zero_grad(); _,l=model(x,y)
        gr=torch.autograd.grad(l,ps,create_graph=True)
        gf=torch.cat([t.reshape(-1) for t in gr])
        hv=torch.autograd.grad((gf*v).sum(),ps,allow_unused=True)
        acc+=torch.cat([(t if t is not None else torch.zeros_like(p)).reshape(-1)
                        for t,p in zip(hv,ps)]).detach()
    model.zero_grad(); return acc/len(B)
opt=torch.optim.AdamW(model.parameters(),lr=LR*5,betas=(0.9,0.95),weight_decay=0.1)
def step_on(B):
    """one AdamW step using batch set B; returns the realised update"""
    th=flat()
    model.zero_grad()
    tot=None
    for x,y in B:
        _,l=model(x,y); (l/len(B)).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    opt.step()
    return flat()-th
def curvature(B,B2=None):
    """kappa_g and II with B for step 1 and B2 (default B) for step 2"""
    th0=flat()
    g=gB(B); setth(th0); gn=float(g.norm()); nh=-g/(gn+1e-30)
    sd=copy.deepcopy(opt.state_dict())
    u1=step_on(B); th1=flat()
    u2=step_on(B2 if B2 is not None else B)
    setth(th0); opt.load_state_dict(sd)
    T=u1/(u1.norm()+1e-30); cn=float(T@nh)
    Tt=T-cn*nh; Tt=Tt/(Tt.norm()+1e-30)
    dT=(u2/(u2.norm()+1e-30))-T
    kg=dT-float(dT@nh)*nh; kg=kg-float(kg@Tt)*Tt
    hT=HvB(Tt,B); setth(th0)
    II=float(hT@Tt)/(gn+1e-30)
    gen=torch.Generator().manual_seed(int(gn*1e6)%10000)
    r=torch.randn(P,generator=gen); r=r-float(r@nh)*nh; r=r/(r.norm()+1e-30)
    hr=HvB(r,B); setth(th0)
    return float(kg.norm()),II,cn,gn,float(hr@r)/(gn+1e-30)
NB=6
FROZEN=[get_batch() for _ in range(NB)]
print(f"  P={P:,}   frozen batch set of {NB} for BOTH consecutive updates\n")
print(f"  {'step':>5}{'|g|':>9}{'cos(u,-g)':>11}{'|kap_g|':>11}{'II(T,T)':>11}"
      f"{'R det':>8}{'R stoch':>9}{'rand II':>10}")
step=0; rows=[]
for ck in (40,70,100,130,160,190):
    while step<ck:
        x,y=get_batch(); _,l=model(x,y); opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); step+=1
    kg,II,cn,gn,iir=curvature(FROZEN)
    kg2,II2,_,_,_=curvature(FROZEN,[get_batch() for _ in range(NB)])
    R=kg/max(abs(II),1e-12); Rs=kg2/max(abs(II2),1e-12)
    print(f"  {ck:>5}{gn:>9.4f}{cn:>11.4f}{kg:>11.5f}{II:>11.5f}{R:>8.3f}"
          f"{Rs:>9.3f}{iir:>10.5f}",flush=True)
    rows.append(dict(ck=ck,gn=gn,cos=cn,kg=kg,II=II,R=R,Rs=Rs,iir=iir))
json.dump(rows,open("/home/claude/work/res_geodet.json","w"),indent=2)
R=np.array([r["R"] for r in rows]); Rs=np.array([r["Rs"] for r in rows])
c=np.array([r["cos"] for r in rows])
print(f"\n  deterministic R: mean {R.mean():.3f} sd {R.std():.3f}  "
      f"spread {100*(R.max()-R.min())/R.mean():.0f}%")
print(f"  stochastic    R: mean {Rs.mean():.3f} sd {Rs.std():.3f}  "
      f"spread {100*(Rs.max()-Rs.min())/Rs.mean():.0f}%")
print(f"  cos(u,-g): {c.round(4)}  sd {c.std():.4f}  (earlier run jumped to 0.3035)")
print(f"\n  R falling toward 0 => sliding: a geodesic of the level set")
print(f"  R ~ 1 and steady    => equipartition")
print(f"  det sd << stoch sd  => the earlier scatter was batch noise")
