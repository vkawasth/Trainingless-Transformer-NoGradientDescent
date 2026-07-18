"""
brane_test.py -- TEST THE BRANE PREDICTION: is densification a WALL-CROSSING?

A wall-crossing is a DISCONTINUOUS change in the set of stable objects.
The paper's brane section claims densification is one. But the attention->FF
coupling was previously measured to grow SMOOTHLY (phase transition withdrawn
as a threshold artifact). So the claim rests on whether the W_Q^T W_K rank
COLLAPSE is abrupt or gradual -- currently unknown (only 2 time points).

Measure QK/OV effective rank on a fine grid. Discriminator:
   concentration = (largest single-interval drop) / (total drop)
   smooth null   ~ 1/n_intervals ;  wall-crossing >> that.
"""
import json, math
import numpy as np, torch
g_={}; src=open("compiler_geometri_patched_86.py").read()
exec(src[:src.find("# ── PHASE 1")], g_)
model=g_["model"]; get_batch=g_["get_batch"]; eval_val=g_["eval_val"]; LR=g_["LR"]
blocks=[m for n,m in model.named_modules() if n.count(".")==1 and n.startswith("blocks.")]

def eff_rank(sv):
    sv=np.asarray(sv,dtype=np.float64); e=sv**2/((sv**2).sum()+1e-30)
    e=e[e>0]; return float(np.exp(-(e*np.log(e)).sum()))

def ranks():
    qk,ov=[],[]
    for b in blocks:
        a=b.attn
        qk.append(eff_rank(torch.linalg.svdvals(a.WQ.weight.detach().T@a.WK.weight.detach()).cpu().numpy()))
        ov.append(eff_rank(torch.linalg.svdvals(a.op.weight.detach()@a.WV.weight.detach()).cpu().numpy()))
    return float(np.mean(qk)), float(np.mean(ov)), qk

def main(N=200, every=8):
    opt=torch.optim.AdamW(model.parameters(), lr=LR*5, betas=(0.9,0.95), weight_decay=0.1)
    log=[]
    q,o,per=ranks(); log.append({"step":0,"val":float(eval_val(model,n=4)),"qk":q,"ov":o,"per":per})
    for s in range(1,N+1):
        model.train(); x,y=get_batch(); _,l=model(x,y)
        opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if s%every==0:
            q,o,per=ranks()
            log.append({"step":s,"val":float(eval_val(model,n=4)),"qk":q,"ov":o,"per":per})
    json.dump(log, open("brane_test.json","w"), indent=1, default=float)

    print("="*70); print("  IS DENSIFICATION A WALL-CROSSING? (QK rank on a fine grid)"); print("="*70)
    print(f"  {'step':>5}{'val':>8}{'QK rank':>9}{'OV rank':>9}{'dQK':>8}   {'L0':>6}")
    print("  "+"-"*50)
    prev=None
    for r in log:
        d = (r["qk"]-prev) if prev is not None else 0.0
        print(f"  {r['step']:>5}{r['val']:>8.3f}{r['qk']:>9.1f}{r['ov']:>9.1f}{d:>8.1f}   {r['per'][0]:>6.1f}")
        prev=r["qk"]
    qs=np.array([r["qk"] for r in log]); steps=[r["step"] for r in log]
    drops=-np.diff(qs); total=qs[0]-qs[-1]
    n_int=len(drops)
    conc=float(drops.max()/ (total+1e-12))
    where=steps[int(np.argmax(drops))+1]
    print("\n  "+"-"*50)
    print(f"  total QK rank drop      : {total:.1f}  ({qs[0]:.1f} -> {qs[-1]:.1f})")
    print(f"  largest single interval : {drops.max():.1f} at step {where}")
    print(f"  concentration           : {100*conc:.0f}% of the drop in 1 of {n_int} intervals")
    print(f"  smooth null             : {100/n_int:.0f}%  (uniform slide)")
    ratio=conc*n_int
    print(f"  ratio vs smooth null    : {ratio:.1f}x")
    print()
    if ratio>4:
        print("  => DISCONTINUOUS. The collapse is concentrated: wall-crossing language")
        print("     is earned (a sharp change in stable structure), even though the")
        print("     COUPLING grows smoothly.")
    elif ratio>2:
        print("  => INTERMEDIATE. Faster than uniform but not a clean jump; 'rapid")
        print("     crossover' is defensible, 'wall-crossing' overstates it.")
    else:
        print("  => SMOOTH. The rank slides. The wall-crossing claim is NOT supported")
        print("     and the brane paragraph must be weakened to a crossover.")

if __name__=="__main__":
    import sys
    main(N=int(sys.argv[1]) if len(sys.argv)>1 else 200,
         every=int(sys.argv[2]) if len(sys.argv)>2 else 8)
