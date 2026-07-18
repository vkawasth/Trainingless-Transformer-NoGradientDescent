"""
coupling_gated.py
=================
Backbone-first schedule with the probe SWAP + full geometric instrumentation.

  - onset gate = frozen-attention cross-Hessian coupling I(att->FF) (leads;
    observable without training attention), replacing the lagging r-onset.
  - QK/OV effective ranks logged over training.
  - all SIX geometric coordinates (phi_k/Phi_cl, tau, E, r_m2, cos_align)
    snapshotted at every TRIGGER event, so post-hoc you can see which
    coordinate co-moves with which transition.  E is logged but is the known
    construction artifact -- if E is the only thing that moves at a trigger,
    that trigger is measuring the mould, not the motion.

Writes coupling_gated.json (a list of trigger events with full geo state).
"""
import json, math
import numpy as np, torch

# ---- load real model + prefix helpers (g_floor, phi_clean, gluing_defect,
#      gradient_alignment, sheet_angles, eval_val, get_batch, LR) ------------
g_={}; src=open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
exec(src[:src.find("# ── PHASE 1")], g_)
# pull compute_rm2_sigma_inline verbatim from source (defined after PHASE 1);
# slice by indentation so we don't drag in Phase-3 top-level code
_lines=src.splitlines()
_s=next(i for i,l in enumerate(_lines) if l.startswith("def compute_rm2_sigma_inline"))
_e=_s+1
while _e<len(_lines) and (_lines[_e].strip()=="" or _lines[_e].startswith((" ","\t"))): _e+=1
exec("\n".join(_lines[_s:_e]), g_)
model=g_["model"]; get_batch=g_["get_batch"]; eval_val=g_["eval_val"]; LR=g_["LR"]
g_floor=g_["g_floor"]; phi_clean=g_["phi_clean"]; sheet_angles=g_["sheet_angles"]
gluing_defect=g_["gluing_defect"]; gradient_alignment=g_["gradient_alignment"]
rm2_fn=g_["compute_rm2_sigma_inline"]
blocks=[m for n,m in model.named_modules() if n.count(".")==1 and n.startswith("blocks.")]

class PassThrough(torch.nn.Module):
    def forward(self,h,*a,**k): return h
def backbone_only(on):
    for b in blocks:
        if on:
            if not hasattr(b,"_sa"): b._sa=b.attn
            b.attn=PassThrough()
        elif hasattr(b,"_sa"): b.attn=b._sa; del b._sa

def gname(n):
    n=n.lower()
    if "wq" in n or "wk" in n or "wv" in n or ".op." in n: return "att"
    if ".ff." in n: return "ff"
    return "other"

# ---- frozen-attention cross-Hessian coupling I(att->FF) -------------------
def coupling_att_ff(probes=3, seed=0):
    backbone_only(False)                       # attention must be in the graph
    x,y=get_batch(); out=model(x,y); loss=out[1]
    params=[p for _,p in model.named_parameters()]
    grads=torch.autograd.grad(loss, params, create_graph=True)
    gflat=torch.cat([g.reshape(-1) for g in grads])
    idx=0; att_mask=[]; ff_mask=[]
    for n,p in model.named_parameters():
        m=gname(n); att_mask.append(m=="att"); ff_mask.append(m=="ff")
    # boolean over flat vector
    sizes=[p.numel() for p in params]
    amask=torch.cat([torch.full((s,), a, dtype=torch.bool) for s,a in zip(sizes,att_mask)])
    fmask=torch.cat([torch.full((s,), f, dtype=torch.bool) for s,f in zip(sizes,ff_mask)])
    P=amask.numel(); gen=torch.Generator().manual_seed(seed); acc=0.0
    for _ in range(probes):
        v=torch.zeros(P); vb=torch.randn(int(fmask.sum()),generator=gen); vb/=vb.norm()+1e-12
        v[fmask]=vb
        gv=(gflat*v).sum()
        Hv=torch.autograd.grad(gv, params, retain_graph=True)
        Hv=torch.cat([h.reshape(-1) for h in Hv])
        acc+=float(Hv[amask].norm())
    model.zero_grad(set_to_none=True)
    return acc/probes

# ---- QK / OV effective ranks ----------------------------------------------
def eff_rank(sv):
    sv=np.asarray(sv,dtype=np.float64); e=sv**2/(sv**2).sum()+1e-30
    return float(np.exp(-(e*np.log(e)).sum()))
def qk_ov_ranks():
    qk,ov=[],[]
    for b in blocks:
        a=b._sa if hasattr(b,"_sa") else b.attn
        Wq,Wk,Wv,Wo=(a.WQ.weight.detach(),a.WK.weight.detach(),
                     a.WV.weight.detach(),a.op.weight.detach())
        qk.append(eff_rank(torch.linalg.svdvals(Wq.T@Wk).cpu().numpy()))
        ov.append(eff_rank(torch.linalg.svdvals(Wo@Wv).cpu().numpy()))
    return {"qk_mean":float(np.mean(qk)),"ov_mean":float(np.mean(ov)),
            "qk_per_layer":qk,"ov_per_layer":ov}

# ---- E (strip energy), same rank-6 W_K convention as rm2 -- the ARTIFACT ---
def strip_energy_E(rank=6):
    Ws=[p.detach().float().cpu().numpy() for n,p in model.named_parameters()
        if "wk" in n.lower() and "weight" in n.lower() and p.ndim>=2]
    Ws.sort(key=lambda w:w.shape[0]); tot=0.0
    for k in range(len(Ws)-1):
        U0,_,_=np.linalg.svd(Ws[k],full_matrices=False)
        U1,_,_=np.linalg.svd(Ws[k+1],full_matrices=False)
        r=min(rank,U0.shape[1],U1.shape[1])
        sv=np.linalg.svd(U0[:,:r].T@U1[:,:r],compute_uv=False)
        tot+=float(np.arccos(np.clip(sv,-1,1)).sum())
    return tot

# ---- the six geometric coordinates ---------------------------------------
def geo_coords():
    backbone_only(False)
    def safe(f,*a):
        try: return f(*a)
        except Exception as e: return None
    return {
        "tau":        safe(gluing_defect, model),
        "Phi_cl":     safe(phi_clean, model),
        "phi_k":      safe(sheet_angles, model),
        "E_artifact": safe(strip_energy_E),
        "r_m2":       safe(rm2_fn, model),
        "cos_align":  safe(gradient_alignment, model, g_floor),
    }

def snapshot(event, step):
    rec={"event":event,"step":step,"val":float(eval_val(model,n=6)),
         "coupling_att_ff":coupling_att_ff(),"ranks":qk_ov_ranks(),"geo":geo_coords()}
    return rec

# ---- schedule: coupling-gated backbone-first, logging at each trigger ------
def run(N=90, probe_every=15, coupling_floor=None):
    opt=torch.optim.AdamW(model.parameters(), lr=LR*5, betas=(0.9,0.95), weight_decay=0.1)
    log=[]
    log.append(snapshot("init", 0))
    c0=log[0]["coupling_att_ff"]
    # gate: if coupling already lifted off at init, don't enter backbone-only
    floor = coupling_floor if coupling_floor is not None else c0*1.5
    on = c0 > (floor if coupling_floor is not None else 0)  # decided per-probe below
    on = False
    if coupling_floor is not None and c0 > coupling_floor:
        on=True; log.append(snapshot("gate:full-from-start (coupling already high)",0))
    backbone_only(not on)
    for step in range(1,N+1):
        if not on and step%probe_every==0:
            c=coupling_att_ff()
            if c > c0*1.5:                       # coupling lifted off -> onset
                on=True; backbone_only(False)
                log.append(snapshot(f"ONSET coupling {c0:.2e}->{c:.2e}", step))
            else:
                log.append(snapshot(f"probe (coupling {c:.2e}, still low)", step))
        backbone_only(not on)
        x,y=get_batch(); out=model(x,y); loss=out[1]
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    backbone_only(False)
    log.append(snapshot("end", N))
    json.dump(log, open("coupling_gated.json","w"), indent=2, default=float)
    return log

if __name__=="__main__":
    import sys
    N=int(sys.argv[1]) if len(sys.argv)>1 else 90
    log=run(N=N)
    print("="*78); print("  COUPLING-GATED BACKBONE-FIRST + GEO INSTRUMENTATION"); print("="*78)
    hdr=f"  {'event':<34}{'step':>5}{'val':>7}{'I(att,ff)':>11}{'QKrank':>7}{'OVrank':>7}"
    print(hdr+f"{'tau':>6}{'Phi':>4}{'E*':>7}{'rm2':>7}{'cos':>7}")
    for r in log:
        g=r["geo"]
        def f(x,d=2): return ("%.*f"%(d,x)) if isinstance(x,(int,float)) else "  -"
        print(f"  {r['event'][:33]:<34}{r['step']:>5}{r['val']:>7.3f}"
              f"{r['coupling_att_ff']:>11.2e}{r['ranks']['qk_mean']:>7.1f}{r['ranks']['ov_mean']:>7.1f}"
              f"{f(g['tau']):>6}{str(g['Phi_cl']):>4}{f(g['E_artifact'],1):>7}"
              f"{f(g['r_m2']):>7}{f(g['cos_align']):>7}")
    print("\n  wrote coupling_gated.json  (full phi_k lists + per-layer ranks inside)")
