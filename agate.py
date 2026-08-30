"""RELIABILITY GATE FOR THE INTERVENTION GRAPH -- RUN BEFORE THE GRAPH.

Phi died because its test-retest ceiling (+0.07) sat below its shuffled null's
spread and below the different-seed floor (+0.27). The lesson, applied here
before any graph is built: establish what the edge observable is worth when
the answer is known.

A_ij = effect at unit j of a norm-matched perturbation of unit i.
Units are rows of block-0 FF's first matrix; j indexes the same layer's output
units. The perturbation is applied to row i only, magnitude fixed, and the
response is the change in mean |activation| at each output unit, on a FIXED
probe batch external to every state.

THREE NUMBERS, exactly as for Phi:
  ceiling   corr(A(theta), A'(theta))  same state, two independent probe
            batches. The instrument's own reproducibility.
  floor     corr(A(seed17), A(seed23)) same checkpoint, independent training.
  null      corr(A, shuffled A).

  ceiling >> floor >> null   -> the graph is measurable and partly
                                realisation-independent; proceed.
  ceiling ~ null             -> discard, as Phi was discarded. Do not build.

Nothing is interpreted in this run. It only decides whether the next one is
worth doing.
"""
import io, contextlib, subprocess, sys, json, time
import numpy as np, torch

OUT="res_agate.json"; RES={}
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
NU=24; PMAG=0.5

def run(seed,tag):
    t0=time.time(); G=build(); model=G["model"]; gb=G["get_batch"]; LR=G["LR"]
    torch.manual_seed(seed)
    opt=torch.optim.AdamW(model.parameters(),lr=LR*5,betas=(0.9,0.95),weight_decay=0.1)
    for _ in range(160):
        x,y=gb(); _,l=model(x,y); opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    ff=model.blocks[0].ff
    Wg=ff.g.weight            # (2D, D)
    bx1,_=gb(); bx2,_=gb()    # two independent probe batches
    @torch.no_grad()
    def resp(x):
        h=model.te(x)+model.pe(torch.arange(x.shape[1]))
        h=model.blocks[0].attn(h)
        return ff.n(h+ff.o(torch.nn.functional.silu(ff.g(h))*ff.v(h)))
    @torch.no_grad()
    def A(x):
        base=resp(x).abs().mean(dim=(0,1))            # (D,) per output unit
        M=np.zeros((NU,base.shape[0]))
        gg=torch.Generator().manual_seed(5150)
        for i in range(NU):
            row=Wg[i].clone()
            d=torch.randn(row.shape,generator=gg); d=d/d.norm()*PMAG*row.norm()
            Wg[i].add_(d)
            M[i]=(resp(x).abs().mean(dim=(0,1))-base).numpy()
            Wg[i].copy_(row)
        return M.reshape(-1)
    RES[f"A1_{tag}"]=A(bx1).tolist(); flush()
    RES[f"A2_{tag}"]=A(bx2).tolist(); flush()
    print(f"  {tag} done ({time.time()-t0:.0f}s)",flush=True)
    del G,model; import gc; gc.collect()

run(17,"s17"); run(23,"s23")
c=lambda a,b: float(np.corrcoef(a,b)[0,1])
a1,a2=RES["A1_s17"],RES["A2_s17"]; b1=RES["A1_s23"]
rng=np.random.default_rng(0)
sh=[c(rng.permutation(np.array(a1)),a2) for _ in range(200)]
print(f"\n  intervention-graph observable, {NU} source units x {len(a1)//NU} targets\n")
print(f"  ceiling  same state, two probe batches : {c(a1,a2):+.4f}")
print(f"           (seed 23)                     : {c(RES['A1_s23'],RES['A2_s23']):+.4f}")
print(f"  floor    step 160, different seed      : {c(a1,b1):+.4f}")
print(f"  null     shuffled                      : {np.mean(sh):+.4f} +- {np.std(sh):.4f}")
print(f"\n  ceiling >> null -> the graph is measurable; proceed")
print(f"  ceiling ~= null -> discard, as Phi was")
flush()
