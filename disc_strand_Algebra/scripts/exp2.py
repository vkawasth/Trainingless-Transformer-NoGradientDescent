import time, copy, json, numpy as np, torch
t0=time.time()
g_={}; src=open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
exec(src[:src.find("# ── PHASE 1")], g_)
model=g_["model"]; get_batch=g_["get_batch"]; eval_val=g_["eval_val"]; LR=g_["LR"]
named=list(model.named_parameters())
def grp(n):
    n=n.lower()
    if ".ln." in n or ".n." in n or n.startswith("ln_f"): return "LN"
    if n.startswith("te") or n.startswith("pe") or n.startswith("head"): return "EMB"
    if ".ff." in n: return "FF"
    return "ATTN"
NODES=["EMB","LN","FF","ATTN"]
sl={}; off=0
for n,p in named:
    sl.setdefault(grp(n),[]).append((off,off+p.numel())); off+=p.numel()
P=off
def flat(): return torch.cat([p.data.flatten() for _,p in named]).clone()
def setflat(v):
    i=0
    with torch.no_grad():
        for _,p in named:
            k=p.numel(); p.data.copy_(v[i:i+k].view_as(p)); i+=k
def sub(v,g):
    m=torch.zeros_like(v)
    for a,b in sl[g]: m[a:b]=v[a:b]
    return m
def V(n=12): return float(eval_val(model,n=n))
def steps(k,seed=None):
    if seed is not None: torch.manual_seed(seed)
    for _ in range(k):
        model.train(); x,y=get_batch(); _,l=model(x,y)
        opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()

# ---------- truth: 0->40->120 ----------
model.load_state_dict(torch.load("init.pt")); torch.manual_seed(17)
opt=torch.optim.AdamW(model.parameters(), lr=LR*5, betas=(0.9,0.95), weight_decay=0.1)
steps(40)
th40=flat(); S40=copy.deepcopy(model.state_dict()); O40=copy.deepcopy(opt.state_dict())
v40=V()
# path length + net over 40->120
prev=flat(); path=torch.zeros_like(prev); torch.manual_seed(9)
for _ in range(80):
    model.train(); x,y=get_batch(); _,l=model(x,y)
    opt.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    cur=flat(); path+=(cur-prev).abs(); prev=cur
th120=flat(); v120=V(); DT=th120-th40
print("="*74); print("  SKELETON GRAPH + 40->120 JUMP"); print("="*74)
print(f"  truth: val(40)={v40:.4f}   val(120)={v120:.4f}   ||Delta||={float(DT.norm()):.3f}")
print(f"  path length (L1 per-param sum of |steps|) vs |net|: "
      f"{float(path.sum()):.1f} vs {float(DT.abs().sum()):.1f}  -> {100*(1-float(DT.abs().sum()/path.sum())):.1f}% cancels")
print(f"  chord/step-sum (L2): {float(DT.norm()):.3f} / {sum(1 for _ in range(1)) and 'see below'}")
print("\n  NODE DECOMPOSITION of the true 40->120 displacement:")
print(f"  {'node':>6}{'params':>11}{'%params':>9}{'||d_node||':>12}{'% of ||D||^2':>14}")
tot=float(DT.norm())**2
for g in NODES:
    dv=sub(DT,g); npar=sum(b-a for a,b in sl[g])
    print(f"  {g:>6}{npar:>11,}{100*npar/P:>8.1f}%{float(dv.norm()):>12.3f}{100*float(dv.norm())**2/tot:>13.1f}%")

def restore():
    model.load_state_dict(copy.deepcopy(S40)); opt.load_state_dict(copy.deepcopy(O40))
def jump(vec, sc=1.0, corr=0):
    restore(); setflat(th40+sc*vec)
    v=V()
    if corr:
        torch.manual_seed(9); steps(corr); v=V()
    return v
rec=lambda v: (v, 100*(v40-v)/(v40-v120+1e-12))
print("\n  "+"-"*70)
print("  A) ORACLE full jump (exact true displacement), scale sweep")
print(f"  {'scale':>7}{'val':>10}{'recovery':>11}")
for sc in [0.25,0.5,0.75,1.0,1.25]:
    v,r=rec(jump(DT,sc)); print(f"  {sc:>7.2f}{v:>10.4f}{r:>10.0f}%")
print("\n  B) ORACLE partial jumps: which nodes carry the progress?")
print(f"  {'jumped':>22}{'val':>10}{'recovery':>11}")
for lab,gs in [("EMB only",["EMB"]),("LN only",["LN"]),("FF only",["FF"]),
               ("ATTN only",["ATTN"]),("SKELETON EMB+LN+FF",["EMB","LN","FF"]),
               ("ATTN complement",["ATTN"])]:
    vec=sum(sub(DT,g) for g in gs)
    v,r=rec(jump(vec)); print(f"  {lab:>22}{v:>10.4f}{r:>10.0f}%")
print("\n  C) NO-ORACLE probe jump: 10 real steps -> direction, scale sweep")
restore(); torch.manual_seed(9); steps(10); dp=flat()-th40
print(f"     cos(10-step probe, true 80-step displacement) = {float((dp@DT)/(dp.norm()*DT.norm())):.4f}")
print(f"  {'scale':>7}{'val':>10}{'recovery':>11}")
for sc in [1,2,4,float(DT.norm()/dp.norm()),8]:
    v,r=rec(jump(dp,sc)); print(f"  {sc:>7.2f}{v:>10.4f}{r:>10.0f}%")
print("\n  D) BALLISTIC: direction from the 0->40 segment")
d040=th40-torch.cat([p.flatten() for p in torch.load("init.pt").values() if p.dtype==torch.float32][:0]) if False else None
print("\n  E) CORRECTOR BUDGET after skeleton-only oracle jump")
vec=sum(sub(DT,g) for g in ["EMB","LN","FF"])
print(f"  {'corrector steps':>18}{'val':>10}{'recovery':>11}")
for c in [0,5,10,20,40]:
    v,r=rec(jump(vec,1.0,corr=c)); print(f"  {c:>18}{v:>10.4f}{r:>10.0f}%")
print("\n  F) CORRECTOR BUDGET after full oracle jump")
for c in [0,5,10,20]:
    v,r=rec(jump(DT,1.0,corr=c)); print(f"  {c:>18}{v:>10.4f}{r:>10.0f}%")
print("\n  G) CONTROL: 80 plain steps from 40 (= truth), and 20/40 plain steps")
for c in [20,40,80]:
    restore(); torch.manual_seed(9); steps(c); v,r=rec(V()); print(f"  {c:>18}{v:>10.4f}{r:>10.0f}%")
print("\ndone %.1fs"%(time.time()-t0))
