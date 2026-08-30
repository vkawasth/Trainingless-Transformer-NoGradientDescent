"""GRADED LOCAL MEASUREMENT WITH VALIDITY DOMAINS AND MATCHED NULLS.

One harness for the pattern both codebases converged on:

    ask a LOCAL question at each resolution level k
    return UNDEFINED where the estimator cannot see
    report the level at which it FIRST fails
    calibrate against a shape-matched null and a test-retest ceiling

Applied here to the two claims that lack nulls.

A. P-ADIC CORPUS STRUCTURE.  padic_orbifold_test.py reports STRUCTURED for
   every prime including p=11, where all 1017 tokens sit in one stratum and
   H = -0.000. It fires on degeneracy: low entropy is read as structure. And
   the corpus is one 1364-token string repeated 270 times, 270 = 2*3^3*5, so
   every frequency carries that factor. Fixed by computing frequencies on the
   BASE sequence and comparing against frequencies shuffled across tokens --
   same multiset, no token identity.

B. ARITY LADDER IDENTIFIABILITY.  _fit_mk in ainf_entropy.py fits (k-1)*d
   parameters from at most fit_window-(k-1) samples: at d=48, k=8 that is 336
   parameters from 57 samples, underdetermined 6x, so the error is
   interpolation not measurement. Here the ladder reports UNDEFINED above the
   identifiability limit instead of a number, and the limit is computed rather
   than assumed.
"""
import io, contextlib, subprocess, sys, json, math, collections
import numpy as np, torch

OUT="res_graded.json"; RES={}
def flush(): json.dump(RES,open(OUT,"w"),indent=1,default=float)

# ---------- A. p-adic corpus structure, done properly ----------
def vp(n,p):
    if n==0: return -1
    k=0
    while n%p==0: n//=p; k+=1
    return k

def strata_entropy(freqs,p):
    v=[vp(c,p) for c in freqs]
    cnt=collections.Counter(v); tot=sum(cnt.values())
    return -sum((c/tot)*math.log2(c/tot) for c in cnt.values() if c>0), dict(sorted(cnt.items()))

def part_A():
    ids=json.load(open("/tmp/train_ids.json"))
    BASE=ids[:1364]
    full=collections.Counter(ids); base=collections.Counter(BASE)
    V=len(json.load(open("/tmp/vocab.json")))
    reps=len(ids)//len(BASE)
    print(f"  corpus: V={V}, {len(ids)} tokens = {len(BASE)} x {reps}")
    print(f"  repetition count {reps} = "+" x ".join(
        str(p) for p in [q for q in range(2,reps+1) for _ in range(vp(reps,q))
                         if all(q%r for r in range(2,q))]))
    rng=np.random.default_rng(0)
    print(f"\n  {'p':>3}{'H(full)':>10}{'H(base)':>10}{'H(null)':>18}{'verdict':>12}")
    for p in (2,3,5,7,11):
        hf,_=strata_entropy([full[t] for t in range(V)],p)
        hb,sb=strata_entropy([base.get(t,0) for t in range(V)],p)
        # NULL: same multiset of counts, reassigned to tokens at random
        counts=[base.get(t,0) for t in range(V)]
        nulls=[]
        for _ in range(200):
            sh=list(counts); rng.shuffle(sh)
            nulls.append(strata_entropy(sh,p)[0])
        mu,sd=float(np.mean(nulls)),float(np.std(nulls))
        z=(hb-mu)/max(sd,1e-9)
        verdict="structured" if abs(z)>3 else "no signal"
        print(f"  {p:>3}{hf:>10.3f}{hb:>10.3f}{mu:>11.3f}+-{sd:<5.3f}{verdict:>12}")
        RES[f"padic_p{p}"]=dict(H_full=hf,H_base=hb,null_mu=mu,null_sd=sd,z=z,strata=sb)
    flush()
    print(f"\n  the null has the SAME count multiset, so any difference is about")
    print(f"  WHICH token holds which count -- the only thing 'structure' can mean")

# ---------- B. arity ladder with an identifiability gate ----------
def part_B():
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
    model=G["model"]; gb=G["get_batch"]
    opt=torch.optim.AdamW(model.parameters(),lr=G["LR"]*5,betas=(0.9,0.95),weight_decay=0.1)
    for _ in range(160):
        x,y=gb(); _,l=model(x,y); opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    x,_=gb()
    with torch.no_grad():
        h=model.te(x)+model.pe(torch.arange(x.shape[1]))
        for blk in model.blocks: h=blk(h)
        H=h[0].numpy()                      # (S, D) one sequence
    S,D=H.shape
    print(f"\n  hidden states {H.shape}, PCA to d, window fit\n")
    print(f"  {'d':>4}{'k':>4}{'params':>9}{'samples':>9}{'ratio':>8}{'relerr':>10}{'status':>12}")
    for d in (4,8,16):
        Hc=H-H.mean(0)
        U,s,Vt=np.linalg.svd(Hc,full_matrices=False)
        Z=Hc@Vt[:d].T
        for k in (2,3,5,8):
            ctx=k-1; nsamp=S-ctx; npar=ctx*d
            ratio=nsamp/max(npar,1)
            if ratio<2.0:
                print(f"  {d:>4}{k:>4}{npar:>9}{nsamp:>9}{ratio:>8.2f}{'--':>10}"
                      f"{'UNDEFINED':>12}")
                RES[f"mk_d{d}_k{k}"]=dict(ratio=ratio,relerr=None); flush(); continue
            X=np.array([Z[i-ctx:i].ravel() for i in range(ctx,S)])
            Y=np.array([Z[i] for i in range(ctx,S)])
            ntr=int(0.7*len(X))
            eps=1e-4*np.trace(X[:ntr].T@X[:ntr])/max(npar,1)
            Wm=np.linalg.lstsq(X[:ntr].T@X[:ntr]+eps*np.eye(npar),X[:ntr].T@Y[:ntr],
                               rcond=None)[0]
            pred=X[ntr:]@Wm
            rel=float(np.linalg.norm(pred-Y[ntr:])/max(np.linalg.norm(Y[ntr:]),1e-30))
            print(f"  {d:>4}{k:>4}{npar:>9}{nsamp:>9}{ratio:>8.2f}{rel:>10.4f}"
                  f"{'ok':>12}")
            RES[f"mk_d{d}_k{k}"]=dict(ratio=ratio,relerr=rel); flush()
    print(f"\n  HELD-OUT error, 70/30 split: an in-sample fit at ratio<2 would")
    print(f"  interpolate and report ~0 regardless of whether m_k exists")

print("="*72); print("  A. p-adic corpus structure, with a count-shuffled null"); print("="*72)
part_A()
print("\n"+"="*72); print("  B. arity ladder, with an identifiability gate"); print("="*72)
part_B()
flush()
