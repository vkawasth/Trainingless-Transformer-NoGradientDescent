"""IS kappa RELIABLE? It was measured at B=1, where the h field's ceiling is 0.46.

kappa(v) = v^T H v / |v|^2 is a scalar, so it should be far more stable than the
field -- but that has not been checked, and kappa_relearn=0.289 vs
kappa_learn=0.0002 is the one curvature result still standing.
Ten independent draws at each B, at C3 and at a displaced state.
"""
import io,contextlib,subprocess,sys,json
from collections import defaultdict
import numpy as np, torch, torch.nn.functional as F
subprocess.run([sys.executable,"build_corpus.py","--out","/tmp","--loops","300"],
               check=True,capture_output=True)
RAW=open("compiler_geometri_patched_86.py").read()
SRC=RAW[:RAW.find("# \u2500\u2500 PHASE 3")]
for o,n in [("D=256; N_HEADS=4","D=128; N_HEADS=4"),
            ("for mf_r in range(1, 16):","for mf_r in range(1, 3):"),
            ("    if pc == N_STU-1:","    if False:"),
            ("    if len(tau_history)>=3 and tau > tau_history[-2] > tau_history[-3]:","    if False:")]:
    assert SRC.count(o)==1; SRC=SRC.replace(o,n,1)
EO="evals,evecs=spla.eigsh(L_sym,k=D+1,which='SM',tol=1e-4,maxiter=2000)"
EN=("_v0=np.random.RandomState(7).randn(L_sym.shape[0])\n"
    "evals,evecs=spla.eigsh(L_sym,k=D+1,which='SM',tol=1e-4,maxiter=2000,v0=_v0)\n"
    "evecs=evecs*np.sign(evecs[np.argmax(np.abs(evecs),axis=0),np.arange(evecs.shape[1])])")
assert SRC.count(EO)==1; SRC=SRC.replace(EO,EN,1)
torch.manual_seed(1234); np.random.seed(1234)
G={}; b=io.StringIO()
with contextlib.redirect_stdout(b): exec(SRC,G)
model=G["model"]; gb=G["get_batch"]; LR=G["LR"]
ids=json.load(open("/tmp/train_ids.json")); BASE=ids[:1364]; W=8
nxt=defaultdict(set)
for od in range(1,6):
    for i in range(len(BASE)-od): nxt[(od,tuple(BASE[i:i+od]))].add(BASE[i+od])
def bk(i):
    for od in range(1,6):
        if i-od>=0 and len(nxt[(od,tuple(BASE[i-od:i]))])==1: return od
    return 6
POS=list(range(W,len(BASE)-1)); np.random.default_rng(3).shuffle(POS)
ALL=defaultdict(list)
for i in POS: ALL[bk(i)].append(i)
F3=ALL[3][:13]
Xf=torch.tensor([[BASE[i-W+j] for j in range(W)] for i in F3])
Yf=torch.tensor([BASE[i] for i in F3])
ps=[p for p in model.parameters() if p.requires_grad]
P=sum(p.numel() for p in ps)
torch.manual_seed(17)
opt=torch.optim.AdamW(model.parameters(),lr=LR*5,betas=(0.9,0.95),weight_decay=0.1)
for _ in range(160):
    x,y=gb(); _,l=model(x,y); opt.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
model.zero_grad(); F.cross_entropy(model(Xf)[0][:,-1,:],Yf).backward()
v=torch.cat([(p.grad.reshape(-1) if p.grad is not None else torch.zeros(p.numel()))
             for p in ps]).clone(); v=v/max(float(v.norm()),1e-30)
def kap(B):
    acc=0.0
    for _ in range(B):
        x,y=gb(); model.zero_grad(); l=model(x,y)[1]
        gr=torch.autograd.grad(l,ps,create_graph=True)
        gf=torch.cat([q.reshape(-1) for q in gr])
        hh=torch.autograd.grad((gf*v).sum(),ps)
        acc+=float(torch.cat([q.reshape(-1) for q in hh]).detach()@v)
    return acc/B
print(f"  kappa = g3^T H g3 / |g3|^2 at C3, 10 independent draws per B\n")
print(f"  {'B':>4}{'mean':>10}{'sd':>10}{'sd/|mean|':>12}")
for B in (1,4,16,64):
    s=[kap(B) for _ in range(10)]
    print(f"  {B:>4}{np.mean(s):>+10.4f}{np.std(s):>10.4f}"
          f"{np.std(s)/max(abs(np.mean(s)),1e-9):>12.2f}",flush=True)
print(f"\n  reference: kappa_learn 0.0002, kappa_relearn 0.289 -- both at B=1")
