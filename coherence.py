"""Is r a bounded coherence coordinate, and is sqrt(v) the noise scale?

GFDS proposes  D_mu = |E[g]| ~ |m|   and   D_sigma = sqrt(E[g^2]) ~ sqrt(v),
with r = D_mu/D_sigma. But E[g^2] = Var[g] + E[g]^2, so sqrt(v) is the SECOND
MOMENT, not the noise scale. Consequences, both testable:
  (a) r = |m|/sqrt(v) <= 1 identically (Jensen), so r is a bounded coherence
      coordinate on [0,1], not a drift/noise ratio that can diverge.
  (b) the actual drift/noise ratio is  rho = |m| / sqrt(v - m^2),  unbounded.
"""
import json, subprocess, numpy as np, torch
subprocess.run(["python3","/mnt/user-data/uploads/build_corpus.py","--out","/tmp",
                "--loops","300"], check=True, capture_output=True)
SRC=open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
g={}; exec(SRC[:SRC.find("# \u2500\u2500 PHASE 1")], g)
model=g["model"]; get_batch=g["get_batch"]; LR=g["LR"]; eval_val=g["eval_val"]
named=list(model.named_parameters()); P=sum(p.numel() for _,p in named)
blk=np.zeros(P,bool); i=0
for nm,p in named:
    if nm.startswith("blocks."): blk[i:i+p.numel()]=True
    i+=p.numel()
torch.manual_seed(17)
opt=torch.optim.AdamW(model.parameters(),lr=LR*5,betas=(0.9,0.95),weight_decay=0.1)
idx=np.random.default_rng(17).choice(np.flatnonzero(blk),40000,replace=False)
print(f"{'step':>6}{'val':>9}{'r_med':>9}{'r_max':>9}{'%r>1':>9}"
      f"{'rho_med':>10}{'rho_p99':>10}{'rho_max':>10}")
done=0
for ck in [10,20,50,100,200]:
    while done<ck:
        x,y=get_batch(); _,l=model(x,y); opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); done+=1
    m=[];v=[]
    for p in model.parameters():
        st=opt.state[p]; m.append(st["exp_avg"].flatten()); v.append(st["exp_avg_sq"].flatten())
    m=torch.cat(m).numpy()[idx]; v=torch.cat(v).numpy()[idx]
    r=np.abs(m)/(np.sqrt(v)+1e-12)
    var=np.maximum(v-m**2,1e-20); rho=np.abs(m)/np.sqrt(var)
    val=float(eval_val(model,n=20)); model.train()
    print(f"{ck:>6}{val:>9.3f}{np.median(r):>9.4f}{r.max():>9.4f}{100*(r>1).mean():>9.4f}"
          f"{np.median(rho):>10.4f}{np.percentile(rho,99):>10.3f}{rho.max():>10.2f}")
