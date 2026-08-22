"""INDIVIDUAL PRINCIPAL ANGLES, NOT THEIR AGGREGATE.

Every rotation number in this programme is d_Gr = sqrt(sum theta_i^2) -- ONE
number summarising r angles. If one angle stays near zero while the others reach
pi/2, there is a persistent one-dimensional piece inside a frame that otherwise
turns freely, and every aggregate measurement would have hidden it.

That is not idle: EMB's k90 collapses to 1.00 and holds for 60 steps, so a
persistent single direction is exactly what one role is known to have.

The SVD already computes all r singular values of Q_t' Q_{t+1}; I have been
collapsing them. Reported per role:

    theta_1 .. theta_k    individual principal angles, ascending
    theta_1               the SMALLEST -- the most persistent direction
    frac(theta_1 < 0.3)   how often the leading direction is nearly fixed

CONTROLS
    random     angles between two random k-planes in R^n. In high dimension all
               of them concentrate near pi/2, so a small theta_1 is meaningful
               only against this.
    split-half angles between frames from interleaved updates at the SAME point,
               which is the estimator's own floor -- the aggregate version of
               this measured 0.283 against a rotation of 1.72.
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
named=[(n,p) for n,p in model.named_parameters()]; ps=[p for _,p in named]
P=sum(p.numel() for p in ps)
span={}; i=0
for nm,p in named: span[nm]=(i,i+p.numel()); i+=p.numel()
def role(nm):
    if nm.startswith("te") or nm.startswith("pe"): return "EMB"
    if "ln" in nm.lower() or nm.endswith("n.weight") or nm.endswith("n.bias"): return "LN"
    if ".ff." in nm: return "FF"
    if "WQ" in nm: return "W_Q"
    if "WK" in nm: return "W_K"
    if "WV" in nm: return "W_V"
    return "W_O"
ROLES=["EMB","LN","W_Q","W_K","W_V","W_O","FF"]
bi={}
for nm,(a,b) in span.items(): bi.setdefault(role(nm),[]).append(torch.arange(a,b))
bi={k:torch.cat(v) for k,v in bi.items()}
K,WINL=4,8
def flat(): return torch.cat([p.data.reshape(-1) for p in ps]).clone()
def ang(Q1,Q2):
    sv=torch.linalg.svdvals(Q1.T@Q2).numpy()
    return np.sort(np.arccos(np.clip(sv,0,1)))
torch.manual_seed(17)
opt=torch.optim.AdamW(model.parameters(),lr=LR*5,betas=(0.9,0.95),weight_decay=0.1)
hist={k:[] for k in bi}; prev=None; step=0
acc={k:[] for k in bi}; sh={k:[] for k in bi}
while step<170:
    th=flat()
    x,y=get_batch(); _,l=model(x,y); opt.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); step+=1
    u=flat()-th
    for k,ii in bi.items():
        hist[k].append(u[ii].clone())
        if len(hist[k])>WINL: hist[k].pop(0)
    if len(hist["FF"])==WINL and step%WINL==0:
        cur={}
        for k,ii in bi.items():
            A=torch.stack(hist[k],1)
            cur[k]=torch.linalg.svd(A,full_matrices=False)[0][:,:K]
            e=torch.linalg.svd(torch.stack(hist[k][0::2],1),full_matrices=False)[0][:,:K//2+1]
            o=torch.linalg.svd(torch.stack(hist[k][1::2],1),full_matrices=False)[0][:,:K//2+1]
            sh[k].append(ang(e,o))
        if prev is not None:
            for k in bi: acc[k].append(ang(prev[k],cur[k]))
        prev=cur
gen=torch.Generator().manual_seed(3)
print(f"  P={P:,}   k={K}   angles in radians, pi/2 = {math.pi/2:.3f}\n")
print(f"  {'role':>6}{'n':>9}" + "".join(f"{'th'+str(j+1):>8}" for j in range(K))
      + f"{'random th1':>12}{'split th1':>11}{'frac<0.3':>10}")
out={}
for r in ROLES:
    if r not in bi or not acc[r]: continue
    A=np.stack(acc[r]); m=A.mean(0)
    n=len(bi[r])
    R=[]
    for _ in range(3):
        a=torch.linalg.qr(torch.randn(n,K,generator=gen))[0]
        b=torch.linalg.qr(torch.randn(n,K,generator=gen))[0]
        R.append(ang(a,b))
    rnd=np.stack(R).mean(0)
    s1=np.stack(sh[r]).mean(0)
    fr=float((A[:,0]<0.3).mean())
    print(f"  {r:>6}{n:>9,}"+"".join(f"{v:>8.3f}" for v in m)
          +f"{rnd[0]:>12.3f}{s1[0]:>11.3f}{fr:>10.3f}")
    out[r]=dict(mean=m.tolist(),rand=rnd.tolist(),split=s1.tolist(),frac=fr)
json.dump(out,open("/home/claude/work/res_angles.json","w"),indent=2)
th1=np.array([out[r]["mean"][0] for r in out]); rn=np.array([out[r]["rand"][0] for r in out])
thk=np.array([out[r]["mean"][-1] for r in out])
print(f"\n  theta_1 mean {th1.mean():.3f} vs random {rn.mean():.3f}   "
      f"theta_{K} mean {thk.mean():.3f}")
print(f"  spread within a frame: theta_{K} - theta_1 = {(thk-th1).mean():.3f}")
print(f"\n  theta_1 << random and << theta_k => a persistent direction hidden")
print(f"  inside a frame that otherwise turns freely")
print(f"  all angles similar => the frame turns as a rigid whole")
