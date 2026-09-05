#!/usr/bin/env python3
"""ZONE-MATCHED OPTIMISER PATCH FOR THE GEOMETRY COMPILER.

  python3 patch_zoneadam.py                 # writes compiler_zoneadam.py
  python3 patch_zoneadam.py --run           # writes and runs it
  python3 compiler_zoneadam.py              # or run the output directly

Needs compiler_geometri_patched_86.py and build_corpus.py in the same
directory, and /tmp/{train_ids,val_ids,vocab}.json (the patched compiler builds
them if missing).

WHY THIS SHAPE
--------------
Phase 3 geo-stops around step 112 at val ~0.22, well short of the val ~0.09
that 200 steps of plain Adam reaches. So Phase 3 is entirely the assimilation
regime -- it never enters the late zone, and a stabilisation trigger inside it
never fires. The late regime is downstream: after the Snapper jump onto the
flat surface, in Phase 5's K0 split and joint descents.

So the two regimes get different optimisers:

  PHASE 3   CompressedAdam
            m at 4 bits per coordinate    (measured L 0.3755 vs Adam 0.3826)
            v at 4 bits per row           (measured L 0.3945, 3% cost)
            refreshed every step          (any staleness diverged: refresh-5
                                           gave cum 3128 against Adam's 65.8)
            ~16x optimiser-state reduction, no compute saving.

  PHASE 5   CompressedAdam(reduced=True)
            additionally replaces the momentum magnitude by its global mean,
            keeping only sgn(mhat). In the zone-switched experiment this was
            the best of five late-zone mechanisms (L 0.0903 against fp32's
            0.0879), and applied at a detected stabilisation point it gave
            0.0843 -- below fp32.

A StabDetector still runs in Phase 3, but as a MONITOR: it prints Sstab so the
assimilation reading can be checked on any corpus, and does not switch anything.

WHAT IS NOT CLAIMED
-------------------
Everything above was measured at D=128 on one corpus -- a single 1364-token
string repeated, with val identical to train -- one seed, and margins of 3-8%.
Several claims in this programme reversed under learning-rate recalibration at
exactly that scale. The Phase 5 reduction in particular has been validated as a
late-zone mechanism in a standalone loop, NOT inside Phase 5, whose K0 split
trains disjoint parameter subsets under a cosine schedule. Treat the Phase 5
arm as untested until its own before/after numbers exist.
"""
import argparse, subprocess, sys

OPT_CLASS = '''# ── ZONE-MATCHED OPTIMISERS (patched) ─────────────────────────
class CompressedAdam:
    """m 4 bits/coord, v 4 bits/row, refreshed every step.

    reduced=True additionally drops the momentum magnitude, keeping
    sgn(mhat) * mean|mhat| -- the late-zone mechanism.

    State is held dequantised for clarity; what was measured is that the
    quantised VALUES suffice, so a memory-constrained deployment would store
    the 4-bit codes and dequantise on read.
    """
    def __init__(self, params, lr, betas=(0.9,0.95), eps=1e-8,
                 weight_decay=0.1, mbits=4, vbits=4, vrow=False, reduced=False):
        self.p=[q for q in params if q.requires_grad]
        self.lr=lr; self.b1,self.b2=betas; self.eps=eps; self.wd=weight_decay
        self.mbits=mbits; self.vbits=vbits; self.vrow=vrow; self.reduced=reduced
        self.m=[torch.zeros_like(q) for q in self.p]
        self.v=[torch.zeros_like(q) for q in self.p]
        self.t=0; self.cum=0.0; self._vfrozen=None
        self._pg=[{"lr":lr}]
    @staticmethod
    def _q(x,bits):
        lv=torch.log(x.clamp_min(1e-20)); lo,hi=float(lv.min()),float(lv.max())
        if hi-lo<1e-12: return x
        s=(hi-lo)/(2**bits-1)
        return torch.exp(torch.round((lv-lo)/s).clamp(0,2**bits-1)*s+lo)
    def fire(self): self.reduced=True
    def freeze_v(self):
        """Latch vhat at its current value. From here the second moment is a
        stored constant: no EMA update is applied to it and no bias correction.

        Evidence: freezing vhat at step k and running to 120 gives val
        6.956, 1.417, 0.543, 0.404, 0.419, 0.411, 0.418 for k = 5,10,20,30,40,
        60,80 against a live-vhat reference of 0.4118. Loss is finished by
        k=30. Direction keeps improving well past it (cos to the true chord
        0.494 -> 0.960 over k=20..80) and scale lags furthest (overshoot ratio
        2.04 -> 1.17), so what is finished at k=30 is what the LOSS needs, not
        the geometry. Below k=20 the denominator is not usable at all: at k=10
        the chord overshoots 37x.
        """
        b2=1-self.b2**max(self.t,1)
        self._vfrozen=[(x/b2).clone() for x in self.v]
    @property
    def param_groups(self): return self._pg
    def zero_grad(self, set_to_none=False):
        for q in self.p:
            if q.grad is not None:
                if set_to_none: q.grad=None
                else: q.grad.detach_(); q.grad.zero_()
    @torch.no_grad()
    def step(self):
        lr=self._pg[0]["lr"]; self.t+=1
        b1=1-self.b1**self.t; b2=1-self.b2**self.t; tot=0.0
        for i,q in enumerate(self.p):
            g=q.grad if q.grad is not None else torch.zeros_like(q)
            self.m[i].mul_(self.b1).add_(g,alpha=1-self.b1)
            self.v[i].mul_(self.b2).addcmul_(g,g,value=1-self.b2)
            vh=(self._vfrozen[i] if getattr(self,'_vfrozen',None) is not None
                else self.v[i]/b2)
            mh=self.m[i]/b1
            if q.dim()==2 and q.shape[0]>1:
                if self.vrow:
                    vh=vh.mean(dim=-1,keepdim=True).expand_as(vh).contiguous()
                vh=self._q(vh,self.vbits)
                mh=(torch.sign(mh)*float(mh.abs().mean())) if self.reduced \\
                   else torch.sign(mh)*self._q(mh.abs(),self.mbits)
            u=-lr*(mh/(vh.sqrt()+self.eps)+self.wd*q.data)
            tot+=float((u*u).sum()); q.data.add_(u)
        self.cum+=tot**0.5

class StabDetector:
    """MONITOR ONLY. Sstab = agree(sgn(W_t-W_0), sgn(W_{t-D}-W_0)) on a
    subsample of one layer. On plain Adam it climbed 0.863 -> 0.957 over 200
    steps and crossed 0.95 at t=140 -- after Phase 3's geo-stop, which is the
    evidence that Phase 3 is assimilation-only."""
    def __init__(self, model, layer=3, sub=40000, delta=16):
        pre=f"blocks.{layer}."
        self.ref=[(n,p) for n,p in model.named_parameters()
                  if n.startswith(pre) and p.requires_grad]
        tot=sum(p.numel() for _,p in self.ref)
        g=torch.Generator().manual_seed(9)
        self.idx=torch.randperm(tot,generator=g)[:min(sub,tot)]
        self.W0=torch.cat([p.data.reshape(-1) for _,p in self.ref])[self.idx].clone()
        self.hist={}; self.delta=delta; self.trace=[]
    @torch.no_grad()
    def check(self, step):
        cur=torch.cat([p.data.reshape(-1) for _,p in self.ref])[self.idx]
        S=torch.sign(cur-self.W0); self.hist[step]=S
        prev=self.hist.get(step-self.delta)
        if prev is None: return None
        ss=float((S==prev).float().mean()); self.trace.append((step,ss)); return ss

'''

P3_OLD = """opt_b = torch.optim.AdamW(model.parameters(), lr=LR*5,
                           betas=(0.9,0.95), weight_decay=0.1)"""
P3_NEW = """VFREEZE_AT   = __VFA__
VFREEZE_GATE = None
opt_b = CompressedAdam(list(model.parameters()), lr=LR*5,
                       betas=(0.9,0.95), weight_decay=0.1)
_stab = StabDetector(model, layer=3)
print("  [zoneadam] Phase 3: CompressedAdam  m 4b/coord, v 4b/row, "
      "refresh every step")
print("  [zoneadam] Sstab monitor armed (reports only; Phase 3 is "
      "assimilation-regime)")
print(f"  [zoneadam] vhat freeze scheduled at step {VFREEZE_AT}"
      if VFREEZE_AT else "  [zoneadam] vhat freeze: off")"""

PROBE_OLD = """        rm2 = compute_rm2_sigma_inline(model)
"""
PROBE_NEW = """        rm2 = compute_rm2_sigma_inline(model)
        _ss = _stab.check(step)
        if _ss is not None:
            print(f"  [zoneadam] Sstab={_ss:.4f} at step {step}")
        # vhat freeze, evaluated at the existing probe so it costs nothing.
        # NOTE: the trigger is a STEP COUNT, not a geometric sensor. No sensor
        # in this compiler has been shown to detect vhat convergence -- around
        # the relevant window Phi_cl and tau show no distinctive signature --
        # so gating on one would be a step count in disguise. VFREEZE_GATE is
        # provided for when a validated sensor exists.
        if VFREEZE_AT and opt_b._vfrozen is None and step >= VFREEZE_AT \
           and (VFREEZE_GATE is None or VFREEZE_GATE(locals())):
            opt_b.freeze_v()
            print(f"  [zoneadam] vhat FROZEN at step {step}"
                  + (f" (Sstab={_ss:.4f})" if _ss is not None else "")
                  + " -- second moment is now a stored constant, no EMA update")
"""

Z1_CONST = "ZONE1_BOOST = __Z1B__\nZONE1_END   = __Z1E__\n"
Z1_OLD = """    if step <= 10:
        for pg in opt_b.param_groups:
            pg['lr'] = LR*5*step/10"""
Z1_NEW = """    # ZONE-I BOOST (patched). Standalone: 20 Adam steps at 4x LR reached
    # val 1.960 where 40 steps at 1x reached 2.226; 20 steps at 1x gave 3.707,
    # so the gain is the larger step, not the shorter walk. Composed with the
    # existing warmup ramp; decays linearly to 1x by ZONE1_END.
    if step == 1:
        print(f"  [zoneadam] Zone-I boost {ZONE1_BOOST}x -> 1x by step {ZONE1_END}")
    _z1 = 1.0 + (ZONE1_BOOST-1.0)*max(0.0, 1.0-(step-1)/max(ZONE1_END-1,1))
    for pg in opt_b.param_groups:
        pg['lr'] = (LR*5*step/10 if step <= 10 else LR*5) * _z1"""

P5_OLD_1 = "        opt1=torch.optim.AdamW(p1,lr=lr_emb_ff,betas=(0.9,0.95),weight_decay=0.1)"
P5_NEW_1 = ("        opt1=CompressedAdam(p1,lr=lr_emb_ff,betas=(0.9,0.95),\n"
            "                            weight_decay=0.1,reduced=True)")
P5_OLD_2 = "        opt2=torch.optim.AdamW(p2,lr=lr_attn,betas=(0.9,0.95),weight_decay=0.1)"
P5_NEW_2 = ("        opt2=CompressedAdam(p2,lr=lr_attn,betas=(0.9,0.95),\n"
            "                            weight_decay=0.1,reduced=True)")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--in",dest="src",default="compiler_geometri_patched_86.py")
    ap.add_argument("--out",dest="dst",default="compiler_zoneadam.py")
    ap.add_argument("--phase5",action="store_true",
                    help="ALSO patch Phase 5 (UNVALIDATED: the arm has never\n                         run to completion here -- k0_split_fn holds three\n                         model copies and exceeded memory)")
    ap.add_argument("--no-zone1",action="store_true")
    ap.add_argument("--zone1-boost",type=float,default=4.0)
    ap.add_argument("--zone1-end",type=int,default=20)
    ap.add_argument("--vfreeze",type=int,default=40,
                    help="freeze vhat at this step; 0 disables")
    ap.add_argument("--run",action="store_true")
    a=ap.parse_args()
    s=open(a.src).read()

    edits=[("phase3 optimiser",P3_OLD,P3_NEW),("phase3 probe",PROBE_OLD,PROBE_NEW)]
    if not a.no_zone1:
        edits.append(("zone1 boost",Z1_OLD,Z1_NEW))
        edits.append(("zone1 const","opt_b = CompressedAdam",
                      Z1_CONST+"opt_b = CompressedAdam"))
    if a.phase5:
        edits+=[("phase5 k0 embff",P5_OLD_1,P5_NEW_1),
                ("phase5 k0 attn", P5_OLD_2,P5_NEW_2)]
    # every anchor asserted: a silent replace miss yields a compiler that runs
    # unpatched while printing the full banner
    for name,pat,_ in edits:
        n=s.count(pat)
        if n!=1: sys.exit(f"anchor {name!r}: found {n}, expected 1 -- ABORT")
    # insert the classes just above Phase 3
    marker="# \u2500\u2500 PHASE 3"
    if s.count(marker)!=1: sys.exit("PHASE 3 marker not unique -- ABORT")
    s=s.replace(marker,OPT_CLASS+marker,1)
    for _,pat,rep in edits: s=s.replace(pat,rep,1)
    s=s.replace("__VFA__",repr(int(a.vfreeze)))
    s=s.replace("__Z1B__",repr(float(a.zone1_boost))).replace("__Z1E__",repr(int(a.zone1_end)))
    for probe in ("class CompressedAdam","_stab.check","CompressedAdam(list(model"):
        if probe not in s: sys.exit(f"patch did not land ({probe}) -- ABORT")
    if a.phase5 and "reduced=True" not in s:
        sys.exit("phase5 reduction did not land -- ABORT")
    open(a.dst,"w").write(s)
    print(f"wrote {a.dst}")
    print("  Phase 3 -> CompressedAdam (assimilation regime)")
    print("  Phase 5 -> CompressedAdam(reduced=True) [UNVALIDATED]"
          if a.phase5 else "  Phase 5 -> unchanged (stock AdamW)")
    print("  Phases 1, 2, 4 untouched")
    if a.run: sys.exit(subprocess.call([sys.executable,a.dst]))

if __name__=="__main__":
    main()
