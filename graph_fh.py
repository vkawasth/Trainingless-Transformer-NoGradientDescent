"""INTERVENTION GRAPH WITH F AND H ATTACHED PER NODE -- NOT USED TO DEFINE IT.

Three fixes to longgraph.py plus the piece it dropped.

FIX 1  A was dominated by scale. rho(A_U, A_160) = +0.988 against a
       reliability ceiling of 0.50 -- more similar than the state is to its own
       re-measurement, which means A was tracking overall activation magnitude
       rather than the interaction PATTERN. Each source row of A is now
       standardised (zero mean, unit norm) before comparison, so rho reflects
       which targets respond, not how loud the layer is.

FIX 2  P~ exceeded 1 everywhere (1.45-1.98). Persistence compared two states
       measured with the SAME probe batch while reliability varied the batch,
       so numerator and denominator had different noise structure. Persistence
       is now cross-batch: rho(A_t^a, A_{t+D}^b).

FIX 3  Reliability is reported per checkpoint, not assumed from step 160.

ADDED, per the plan: F and H attached to every NODE, measured alongside and
never used to define an edge.
    F_i = sum of diag Fisher over source row i's parameters
    h_i = sum of |H g3| over the same, the curvature field along the O3
          gradient
Then the question the attachment exists to answer:
    corr(F_i, outdeg_i) and corr(h_i, outdeg_i)
where outdeg_i = ||A_i||. If the geometric fields predict which units have
strong causal reach, F/H are coordinates for the graph. If not, they are local
diagnostics beside it -- which is what the session's evidence so far suggests.
"""
import io, contextlib, subprocess, sys, json, time
from collections import defaultdict
import numpy as np, torch, torch.nn.functional as F

OUT="res_graph_fh.json"; RES={}
def flush(): json.dump(RES,open(OUT,"w"),indent=1,default=float)
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
def build():
    torch.manual_seed(1234); np.random.seed(1234)
    G={}; b=io.StringIO()
    with contextlib.redirect_stdout(b): exec(SRC,G)
    return G
ids=json.load(open("/tmp/train_ids.json")); BASE=ids[:1364]; W=8
nxt=defaultdict(set)
for od in range(1,6):
    for i in range(len(BASE)-od): nxt[(od,tuple(BASE[i:i+od]))].add(BASE[i+od])
def bkt(i):
    for od in range(1,6):
        if i-od>=0 and len(nxt[(od,tuple(BASE[i-od:i]))])==1: return od
    return 6
POS=list(range(W,len(BASE)-1)); np.random.default_rng(3).shuffle(POS)
ALL=defaultdict(list)
for i in POS: ALL[bkt(i)].append(i)
FIT3=ALL[3][:13]; HELD3=ALL[3][len(ALL[3])//2:][:60]
def tens(idx):
    return (torch.tensor([[BASE[i-W+j] for j in range(W)] for i in idx]),
            torch.tensor([BASE[i] for i in idx]))
CKS=[20,40,60,80,100,120,160]; NU=24; PMAG=0.5; EPS_ASC,CAP,POLL=0.02,900,5

def run(seed,tag):
    t0=time.time(); G=build(); model=G["model"]; gb=G["get_batch"]; LR=G["LR"]
    torch.manual_seed(seed)
    ps=[p for p in model.parameters() if p.requires_grad]
    ff=model.blocks[0].ff; Wg=ff.g.weight
    # index range of Wg inside the flat parameter vector
    off=0; wg_off=None
    for p in ps:
        if p is Wg: wg_off=off
        off+=p.numel()
    bx1,_=gb(); bx2,_=gb()
    X3,Y3=tens(HELD3); Xf,Yf=tens(FIT3)
    @torch.no_grad()
    def resp(x):
        h=model.te(x)+model.pe(torch.arange(x.shape[1]))
        h=model.blocks[0].attn(h)
        return ff.n(h+ff.o(F.silu(ff.g(h))*ff.v(h)))
    @torch.no_grad()
    def A(x):
        base=resp(x).abs().mean(dim=(0,1))
        M=np.zeros((NU,base.shape[0])); gg=torch.Generator().manual_seed(5150)
        for i in range(NU):
            row=Wg[i].clone()
            d=torch.randn(row.shape,generator=gg); d=d/d.norm()*PMAG*row.norm()
            Wg[i].add_(d); M[i]=(resp(x).abs().mean(dim=(0,1))-base).numpy()
            Wg[i].copy_(row)
        return M
    def std_rows(M):                      # FIX 1: pattern, not magnitude
        Z=M-M.mean(1,keepdims=True)
        return (Z/(np.linalg.norm(Z,axis=1,keepdims=True)+1e-12)).reshape(-1)
    @torch.no_grad()
    def acc3():
        model.eval(); r=float((model(X3)[0][:,-1,:].argmax(-1)==Y3).float().mean())
        model.train(); return r
    def node_fields():
        nrow=Wg.shape[1]
        Fd=torch.zeros(sum(p.numel() for p in ps))
        for _ in range(8):
            x,y=gb(); model.zero_grad(); model(x,y)[1].backward()
            Fd+=torch.cat([(p.grad.reshape(-1) if p.grad is not None
                            else torch.zeros(p.numel())) for p in ps])**2
        Fd/=8
        model.zero_grad(); F.cross_entropy(model(Xf)[0][:,-1,:],Yf).backward()
        v=torch.cat([(p.grad.reshape(-1) if p.grad is not None else
                      torch.zeros(p.numel())) for p in ps]).clone()
        v=v/max(float(v.norm()),1e-30)
        x,y=gb(); model.zero_grad()
        l=model(x,y)[1]
        gr=torch.autograd.grad(l,ps,create_graph=True)
        gf=torch.cat([q.reshape(-1) for q in gr])
        hh=torch.autograd.grad((gf*v).sum(),ps)
        h=torch.cat([q.reshape(-1) for q in hh]).detach()
        Fi=[float(Fd[wg_off+i*nrow:wg_off+(i+1)*nrow].sum()) for i in range(NU)]
        hi=[float(h[wg_off+i*nrow:wg_off+(i+1)*nrow].abs().sum()) for i in range(NU)]
        return Fi,hi
    opt=torch.optim.AdamW(model.parameters(),lr=LR*5,betas=(0.9,0.95),weight_decay=0.1)
    st=0
    for T in CKS:
        while st<T:
            x,y=gb(); _,l=model(x,y); opt.zero_grad(); l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); st+=1
        Ma,Mb=A(bx1),A(bx2)
        Fi,hi=node_fields()
        RES[f"a_{tag}_{T}"]=std_rows(Ma).tolist(); RES[f"b_{tag}_{T}"]=std_rows(Mb).tolist()
        RES[f"deg_{tag}_{T}"]=np.linalg.norm(Ma,axis=1).tolist()
        RES[f"F_{tag}_{T}"]=Fi; RES[f"h_{tag}_{T}"]=hi
        RES[f"o3_{tag}_{T}"]=acc3(); flush()
        print(f"  {tag} t={T}: O3 {acc3():.3f} ({time.time()-t0:.0f}s)",flush=True)
    if tag=="s17":
        lo=RES[f"o3_{tag}_60"]; n=0
        while n<CAP:
            if n%POLL==0 and acc3()<=lo: break
            model.zero_grad(); F.cross_entropy(model(Xf)[0][:,-1,:],Yf).backward()
            g=torch.cat([(p.grad.reshape(-1) if p.grad is not None else
                          torch.zeros(p.numel())) for p in ps]); gn=float(g.norm())
            with torch.no_grad():
                for p in ps:
                    if p.grad is not None: p.data.add_(p.grad*(EPS_ASC/max(gn,1e-30)))
            n+=1
        Ma,Mb=A(bx1),A(bx2)
        RES[f"a_{tag}_U"]=std_rows(Ma).tolist(); RES[f"b_{tag}_U"]=std_rows(Mb).tolist()
        RES[f"deg_{tag}_U"]=np.linalg.norm(Ma,axis=1).tolist()
        RES[f"o3_{tag}_U"]=acc3(); flush()
        print(f"  {tag} unlearned: O3 {acc3():.3f}",flush=True)
    del G,model; import gc; gc.collect()

run(17,"s17"); run(23,"s23")
c=lambda a,b: float(np.corrcoef(a,b)[0,1])
g=lambda k: np.array(RES[k])
print(f"\n  row-standardised A. persistence is CROSS-BATCH, so P~ <= 1.\n")
print(f"  {'t':>5}{'O3':>7}{'R_t':>8}{'R_t s23':>9}{'S_t':>8}"
      f"{'P(t,nx)':>10}{'P~':>8}{'cF':>8}{'ch':>8}")
for i,T in enumerate(CKS):
    R1=c(g(f"a_s17_{T}"),g(f"b_s17_{T}")); R2=c(g(f"a_s23_{T}"),g(f"b_s23_{T}"))
    S=c(g(f"a_s17_{T}"),g(f"a_s23_{T}"))
    cF=c(g(f"F_s17_{T}"),g(f"deg_s17_{T}")); ch=c(g(f"h_s17_{T}"),g(f"deg_s17_{T}"))
    if i+1<len(CKS):
        T2=CKS[i+1]
        R1b=c(g(f"a_s17_{T2}"),g(f"b_s17_{T2}"))
        P=c(g(f"a_s17_{T}"),g(f"b_s17_{T2}"))
        Pt=P/np.sqrt(max(R1,1e-6)*max(R1b,1e-6))
        ps_,pts=f"{P:>10.3f}",f"{Pt:>8.3f}"
    else: ps_,pts=f"{'--':>10}",f"{'--':>8}"
    print(f"  {T:>5}{RES[f'o3_s17_{T}']:>7.3f}{R1:>8.3f}{R2:>9.3f}{S:>8.3f}"
          f"{ps_}{pts}{cF:>+8.3f}{ch:>+8.3f}")
if "a_s17_U" in RES:
    RU=c(g("a_s17_U"),g("b_s17_U"))
    print(f"\n  unlearned: O3 {RES['o3_s17_U']:.3f}  R_U {RU:.3f}")
    for T in (60,160):
        P=c(g("a_s17_U"),g(f"b_s17_{T}")); Rt=c(g(f"a_s17_{T}"),g(f"b_s17_{T}"))
        print(f"    rho(A_U, A_{T}) {P:+.3f}   corrected "
              f"{P/np.sqrt(max(RU,1e-6)*max(Rt,1e-6)):+.3f}")
rng=np.random.default_rng(0)
sh=[c(rng.permutation(g("a_s17_160")),g("b_s17_160")) for _ in range(200)]
print(f"\n  null {np.mean(sh):+.4f} +- {np.std(sh):.4f}")
print(f"  cF/ch = corr of node Fisher / curvature with out-degree")
flush()
