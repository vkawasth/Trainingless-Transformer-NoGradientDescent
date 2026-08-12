"""WHERE DOES THE FLOW RESIDUAL SIT? PER TILE, 200 STEPS, FIXED BATCH.

Established on a fixed batch: R = dtheta_disc - dtheta_flow has |R|/|dtheta| =
1.36, cos(disc, flow) = +0.25, R_perp = 0.76, and a smooth decay curve
0.965 -> 0.744 over eight lags. Euler-vs-RK4 converges at slope 1.00, so this is
not discretisation and not sampling noise -- Adam's field simply differs from
the gradient-flow field.

The open question is WHERE. Decomposed per component and per layer over 200
steps, all on the same fixed batch:

  rel_i    |R_i| / |dtheta_i|                 how far that tile is from flow
  cos_i    cos(dtheta_i^disc, dtheta_i^flow)  directional agreement
  par_i    R_par / |R|^2 for that tile        loss-active share
  lag1_i   corr of R_i direction, consecutive persistence per tile

If the residual is uniform across tiles, it is a global property of the update
rule. If it concentrates, the concentrating tiles are where Adam's field departs
most from descent.
"""
import json, subprocess, numpy as np, torch, gc, io, contextlib
subprocess.run(["python3","/mnt/user-data/uploads/build_corpus.py","--out","/tmp",
                "--loops","300"],check=True,capture_output=True)
RAW=open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
CUT=RAW.find("# \u2500\u2500 PHASE 3")
src=RAW[:CUT].replace("D=256; N_HEADS=4","D=128; N_HEADS=4",1)
src=src.replace("for mf_r in range(1, 16):","for mf_r in range(1, 3):",1)
src=src.replace("    if pc == N_STU-1:","    if False:",1)
src=src.replace("    if len(tau_history)>=3 and tau > tau_history[-2] > tau_history[-3]:",
                "    if False:",1)
G={}; buf=io.StringIO()
with contextlib.redirect_stdout(buf): exec(src,G)
model=G["model"]; get_batch=G["get_batch"]; LR=G["LR"]
named=[(n,p) for n,p in model.named_parameters()]; params=[p for _,p in named]
P=sum(p.numel() for p in params); nL=len(model.blocks)
for p in params: p.requires_grad_(True)
FX,FY=get_batch()
def flat(): return torch.cat([p.data.flatten() for p in params]).clone()
def setth(t):
    with torch.no_grad():
        i=0
        for p in params:
            q=p.numel(); p.data.copy_(t[i:i+q].view_as(p)); i+=q
def grad_at(th):
    setth(th); model.zero_grad(); _,l=model(FX,FY); l.backward()
    g=torch.cat([(p.grad.flatten() if p.grad is not None else
        torch.zeros(p.numel())) for p in params]).clone()
    model.zero_grad(); return g,float(l)
def rk4(th,h):
    k1,_=grad_at(th); k2,_=grad_at(th-0.5*h*k1)
    k3,_=grad_at(th-0.5*h*k2); k4,_=grad_at(th-h*k3)
    return -(h/6.0)*(k1+2*k2+2*k3+k4)
def grp(nm):
    if nm.startswith("te") or nm.startswith("pe"): return "EMB"
    if nm.startswith("blocks."):
        l=nm.split(".")[1]
        if "ln" in nm.lower(): return "LN"
        if ".ff." in nm: return f"FF{l}"
        return f"AT{l}"
    return "LN"
GS=["EMB","LN"]+[f"FF{l}" for l in range(nL)]+[f"AT{l}" for l in range(nL)]
span={}; i=0
for nm,p in named: span[nm]=(i,i+p.numel()); i+=p.numel()
masks={}
for g_ in GS:
    m=torch.zeros(P,dtype=torch.bool)
    for nm,(a,b) in span.items():
        if grp(nm)==g_: m[a:b]=True
    if m.sum()>0: masks[g_]=m
GS=[g_ for g_ in GS if g_ in masks]
torch.manual_seed(17)
opt=torch.optim.AdamW(model.parameters(),lr=LR*5,betas=(0.9,0.95),weight_decay=0.1)
acc={g_:{"rel":[],"cos":[],"par":[],"R":[]} for g_ in GS}
tot={"rel":[],"cos":[]}
for t in range(200):
    th=flat(); g,lv=grad_at(th); setth(th)
    opt.zero_grad(); _,l=model(FX,FY); l.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    dd=flat()-th
    hh=float(dd.norm()/(g.norm()+1e-30))
    df=rk4(th,hh); setth(th+dd)
    if t>=100:
        R=dd-df
        tot["rel"].append(float(R.norm()/(dd.norm()+1e-30)))
        tot["cos"].append(float((dd*df).sum()/(dd.norm()*df.norm()+1e-30)))
        for g_ in GS:
            m=masks[g_]; Ri=R[m]; di=dd[m]; fi=df[m]; gi=g[m]
            acc[g_]["rel"].append(float(Ri.norm()/(di.norm()+1e-30)))
            acc[g_]["cos"].append(float((di*fi).sum()/(di.norm()*fi.norm()+1e-30)))
            pa=float((Ri*gi).sum())**2/max(float((gi*gi).sum()),1e-30)
            acc[g_]["par"].append(pa/max(float((Ri*Ri).sum()),1e-30))
            acc[g_]["R"].append((Ri/(Ri.norm()+1e-30)).clone())
print(f"  fixed batch, 200 steps, stats over steps 100-199")
print(f"  GLOBAL: |R|/|dth| {np.mean(tot['rel']):.3f}   "
      f"cos(disc,flow) {np.mean(tot['cos']):+.3f}\n")
print(f"  {'tile':>6}{'|R|/|dth|':>11}{'cos':>9}{'R_par':>8}{'lag1':>8}{'lag4':>8}")
for g_ in GS:
    a=acc[g_]; Rs=a["R"]
    l1=np.mean([float((Rs[i]*Rs[i+1]).sum()) for i in range(len(Rs)-1)])
    l4=np.mean([float((Rs[i]*Rs[i+4]).sum()) for i in range(len(Rs)-4)])
    print(f"  {g_:>6}{np.mean(a['rel']):>11.3f}{np.mean(a['cos']):>+9.3f}"
          f"{np.mean(a['par']):>8.4f}{l1:>8.4f}{l4:>8.4f}")
json.dump({g_:{k:float(np.mean(v)) for k,v in acc[g_].items() if k!="R"} for g_ in GS},
          open("/home/claude/work/res_tilewise.json","w"),indent=2)
