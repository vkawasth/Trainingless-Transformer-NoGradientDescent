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
                 weight_decay=0.1, mbits=4, vbits=4, vrow=False, reduced=False,
                 names=None):
        if names is not None:
            pn=[(n,q) for n,q in zip(names,params) if q.requires_grad]
            self.names=[n for n,_ in pn]; self.p=[q for _,q in pn]
        else:
            self.p=[q for q in params if q.requires_grad]
            self.names=[f"p{i}" for i in range(len(self.p))]
        self.lr=lr; self.b1,self.b2=betas; self.eps=eps; self.wd=weight_decay
        self.mbits=mbits; self.vbits=vbits; self.vrow=vrow; self.reduced=reduced
        self.m=[torch.zeros_like(q) for q in self.p]
        self.v=[torch.zeros_like(q) for q in self.p]
        self.t=0; self.cum=0.0; self._vfrozen=None
        self.bidx={}
        for _i,_nm in enumerate(self.names):
            self.bidx.setdefault(self.bucket_of(_nm),[]).append(_i)
        self.buckets=sorted(self.bidx)
        self.hist={k:[] for k in self.buckets}
        self.sub={k:None for k in self.buckets}
        self.track=True
        self._pg=[{"lr":lr}]
    @staticmethod
    def _q(x,bits):
        lv=torch.log(x.clamp_min(1e-20)); lo,hi=float(lv.min()),float(lv.max())
        if hi-lo<1e-12: return x
        s=(hi-lo)/(2**bits-1)
        return torch.exp(torch.round((lv-lo)/s).clamp(0,2**bits-1)*s+lo)
    def fire(self): self.reduced=True
    @staticmethod
    def bucket_of(name):
        """K=6 buckets. Measured freeze-readiness differs sharply between them:
        attention-only freeze at 40 gave 0.3890 against 0.4126 for no freeze and
        0.4261 for skeleton-only, i.e. freezing the skeleton is WORSE than not
        freezing at all. A coherence gate built to discover this schedule lost
        to its own reversed control (0.4245 vs 0.3966), so the schedule below is
        a fixed step, not a detected one."""
        if name.startswith("te"): return "EMB_tok"
        if name.startswith("pe"): return "EMB_pos"
        if "ln" in name.lower():  return "LN"
        if ".ff." in name:        return "FF"
        if ".WQ." in name or ".WK." in name: return "ATT_QK"
        return "ATT_VO"
    def freeze_buckets(self, buckets):
        """Latch vhat for the named buckets only; the rest stay live."""
        b2=1-self.b2**max(self.t,1); n=0
        if self._vfrozen is None: self._vfrozen=[None]*len(self.p)
        for i,nm in enumerate(self.names):
            if self.bucket_of(nm) in buckets and self._vfrozen[i] is None:
                self._vfrozen[i]=(self.v[i]/b2).clone(); n+=self.p[i].numel()
        return n
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
        """Explicit bucket form.

            g^(k)     = grad restricted to bucket k
            m^(k)     = b1 m + (1-b1) g
            v^(k)     = b2 v + (1-b2) g^2
            alpha_t   = lr * sqrt(1-b2^t)/(1-b1^t)     ONE global scalar
            eps_t     = eps * sqrt(1-b2^t)             exactness of the
                                                       absorbed form
            theta^(k) -= alpha_t * m^(k)/(sqrt(v^(k)) + eps_t)

        The absorbed correction replaces two per-coordinate divisions with one
        scalar per step. It equals the standard form only with eps_t as given:
        with a constant eps the effective epsilon differs 4.5x at t=1 and 1.07x
        at t=40. Measured difference on this pipeline: 0.4126 vs 0.4124.
        """
        lr=self._pg[0]["lr"]; self.t+=1
        b1=1-self.b1**self.t; b2=1-self.b2**self.t
        alpha=lr*math.sqrt(b2)/b1; eps_t=self.eps*math.sqrt(b2)
        tot=0.0
        for k in self.buckets:
            acc=[]
            for i in self.bidx[k]:
                q=self.p[i]
                g=q.grad if q.grad is not None else torch.zeros_like(q)
                self.m[i].mul_(self.b1).add_(g,alpha=1-self.b1)
                self.v[i].mul_(self.b2).addcmul_(g,g,value=1-self.b2)
                _f=self._vfrozen
                vk=(_f[i] if (_f is not None and _f[i] is not None) else self.v[i])
                mk=self.m[i]
                if q.dim()==2 and q.shape[0]>1:
                    if self.vrow:
                        vk=vk.mean(dim=-1,keepdim=True).expand_as(vk).contiguous()
                    vk=self._q(vk,self.vbits)
                    mk=(torch.sign(mk)*float(mk.abs().mean())) if self.reduced \\
                       else torch.sign(mk)*self._q(mk.abs(),self.mbits)
                uk=mk/(vk.sqrt()+eps_t)
                if self.track: acc.append(uk.reshape(-1))
                d=-alpha*uk - lr*self.wd*q.data
                tot+=float((d*d).sum()); q.data.add_(d)
            if self.track and acc: self._push(k,torch.cat(acc))
        self.cum+=tot**0.5

    def _push(self,k,u):
        if self.sub[k] is None:
            g=torch.Generator().manual_seed(9)
            self.sub[k]=torch.randperm(u.numel(),generator=g)[:min(4000,u.numel())]
        h=self.hist[k]; h.append(u[self.sub[k]].clone())
        if len(h)>16: h.pop(0)

    def geometry(self):
        """Per-bucket trajectory coordinates: coherence, step angle, top-4 share.

        Measured at step 120 on this pipeline the buckets are geometrically
        distinct -- LN 0.87/13.3/0.973 moves nearly straight, EMB_pos
        0.63/45.3/0.727 turns 45 deg per step. REPORTED, NOT ACTED ON: a
        coherence gate built from these lost to its own reversed control
        (0.4245 vs 0.3966), and a gradient-norm gate tied its random-matched
        control (0.4226 vs 0.4221 +- 0.0019).
        """
        out={}
        for k in self.buckets:
            h=self.hist[k]
            if len(h)<3: continue
            H=torch.stack(h,1)
            coh=float(H.sum(1).norm())/max(float(sum(x.norm() for x in h)),1e-30)
            sv=torch.linalg.svdvals(H); e=sv**2
            R4=float(e[:4].sum())/max(float(e.sum()),1e-30)
            th=float(np.degrees(np.arccos(np.clip(float(
                (h[-1]@h[-2])/(h[-1].norm()*h[-2].norm()+1e-30)),-1,1))))
            out[k]=(coh,th,R4)
        return out

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
MF_OLD = "ETA_MF=0.01; N_SUB=200"
MF_NEW = "ETA_MF=__MFE__; N_SUB=200"
P3CAP_OLD = "for step in range(1, 151):"
P3CAP_NEW = "for step in range(1, __P3CAP__+1):"
PLAT_OLD = """        if delta < 0.003:
            print(f"  \u2713 Plateau (loss)"); break
        if v < 0.15:
            print(f"  \u2713 val={v:.4f} < 0.15"); break"""
PLAT_NEW = """        _small = delta < P3PLAT
        if step >= P3MIN and _small and _prev_small:
            print(f"  \u2713 Plateau (2 consecutive < {P3PLAT})"); break
        _prev_small = _small
        if P3VALSTOP > 0 and v < P3VALSTOP:
            print(f"  \u2713 val={v:.4f} < {P3VALSTOP}"); break"""
SNAP_OLD = 'direction = hessian_smallest_eigenvector(model)'
SNAP_NEW = """# SNAPPER DISABLED. Measured over 1200 Adam steps on a decorrelated
# corpus (val 4.81 -> 2.92, floor 2.238) the top Hessian direction is UPHILL
# at every probe distance 0.25-4.0 and every checkpoint; the quadratic
# coefficient decays 0.445 -> 0.168 rather than turning positive. There is no
# convex basin to jump into. On the degenerate corpus it was +15.5 with a real
# minimum, which is what the method was built for.
#
# It is skipped outright rather than no-opped, because the power iteration
# calls autograd.grad over all parameters and fails once --prune-attn has set
# requires_grad=False on a block's W_Q/W_K.
print("  [zoneadam] SNAPPER SKIPPED -- no convex basin on this corpus")
SNAPPER_SKIPPED = True
direction = torch.zeros(model.flat_params().numel())"""

TG_OLD = '    print("\u2501\u2501\u2501 PHASE 4: TOPOGATE (geometry-checked) \u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501")'
TG_NEW = """    _TG_SNAP = {n: p.data.clone() for n, p in model.named_parameters()}
    print("\u2501\u2501\u2501 PHASE 4: TOPOGATE (geometry-checked) \u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501")"""
TG2_OLD = '    print("\u2501\u2501\u2501 PHASE 5: ALIGNMENT'
TG2_NEW = """    # TopoGate flips discrete signs from Phi. Under the structural loss it
    # cost 0.19 nats (2.8416 -> 3.0337) against 0.016 under plain CE, because
    # the boundary term reorganises exactly the structure Phi measures.
    with torch.no_grad():
        for _n, _p in model.named_parameters(): _p.data.copy_(_TG_SNAP[_n])
    print("  [zoneadam] TOPOGATE REVERTED")
    print("\u2501\u2501\u2501 PHASE 5: ALIGNMENT"""

SL_OLD = """    model.train(); x, y = get_batch(); _, l = model(x, y)\n    opt_b.zero_grad(); l.backward()"""
SL_NEW = """    model.train(); x, y = get_batch(); _lg, l = model(x, y)\n    if STRUCT_LOSS: l = struct_loss(_lg, x, y, step)\n    opt_b.zero_grad(); l.backward()"""

P3_NEW = """STRUCT_LOSS     = __STRUCT__
if STRUCT_LOSS:
    # STRUCTURAL OBJECTIVE. Cross-entropy's gradient on a non-successor is
    # dL/dz_k = p(k|x) ~ 1e-4, so it cannot clear a tail it has already made
    # small. Measured under plain CE: precision 0.059, margin
    # min(z_valid)-max(z_invalid) = -0.91, i.e. the best invalid token
    # outscores the average valid one. Not a logit-scale problem -- T=1 is
    # already optimal, KL rises at 0.75 and at 1.5.
    #
    #   L = CE + lambda * softplus(gamma - [min_{i in S} z_i - LSE_{k not in S} z_k])
    #           + lambda_col * (the same on the columns of J(x,y)=p(y|x)P(x))
    #
    # LSE aggregates the whole tail so its gradient does not vanish as
    # individual tail masses shrink (max-margin constrains ONE token and stalls
    # precision at 0.289). The column term is the gluing constraint: forward
    # and reverse are two normalisations of one joint, so one weight set must
    # satisfy both; it improves the rows as well as the columns.
    #
    # S is a running accumulator of OBSERVED successors -- no oracle, 0 false
    # positives by construction, 98% coverage by step 700.
    #
    # lambda is released once the margin is comfortably satisfied: held fixed,
    # precision reaches 0.911 but KL rises 0.906 -> 1.001. Three controllers
    # (decay, hysteresis, pulsed) land on the same frontier, so the release
    # rate is the only knob that matters.
    #
    # Measured vs plain CE at step 700: KL 1.8336 -> 0.7811, precision
    # 0.059 -> 0.756, reverse recovery 0.542 -> 0.950.
    import torch.nn.functional as _F
    SL_GAMMA, SL_LAMBDA = 10.0, 0.5
    SL_FORK, SL_RELEASE, SL_RELTHR = 140, 0.3, 2.0
    _OBS = torch.zeros(VOCAB, VOCAB, dtype=torch.bool)

    def struct_loss(logits, x, y, step):
        lg = logits.reshape(-1, VOCAB)
        tgt = y.reshape(-1); ctx = x.reshape(-1)
        ce = _F.cross_entropy(lg, tgt)
        if step < SL_FORK:
            return ce
        _OBS[ctx, tgt] = True
        M = _OBS[ctx]
        minv = torch.where(M, lg, torch.full_like(lg, float("inf"))).min(-1).values
        lse = torch.logsumexp(torch.where(M, torch.full_like(lg, -1e30), lg), -1)
        ok = torch.isfinite(minv) & torch.isfinite(lse)
        if not ok.any():
            return ce
        ach = float((minv[ok] - lse[ok]).mean())
        lam = SL_LAMBDA / (1.0 + SL_RELEASE * max(0.0, ach - SL_RELTHR))
        return ce + lam * _F.softplus(SL_GAMMA - (minv[ok] - lse[ok])).mean()
    print(f"  [zoneadam] STRUCTURAL LOSS: LSE boundary on observed support, "
          f"gamma={SL_GAMMA} lambda={SL_LAMBDA} fork={SL_FORK}")
P3MIN           = __P3MIN__
P3PLAT          = __P3PLAT__
P3VALSTOP       = __P3VALSTOP__
_prev_small     = False
PRUNE_ATTN      = __PRUNE__
if PRUNE_ATTN:
    # ATTENTION-BLOCK PRUNING.
    #
    # A block whose attention map has collapsed to the causal-uniform
    # distribution is a static prefix average. The residual stream already
    # carries its input forward, so the loss is insensitive to P, dL/dP -> 0,
    # and W_Q, W_K receive no gradient -- while Adam's 1/sqrt(v) manufactures a
    # full-sized step out of that null signal.
    #
    # The floor is analytic, from the causal mask alone: row i attends to i
    # positions, so uninformative logits give p_max(i)=1/i and H=ln(i).
    # Averaging:  p_max = H_N/N,  ent = ln(N!)/(N ln N).
    # At N=64 these are 0.0741 and 0.7708; measured on a frozen block 0,
    # 0.0741 and 0.7708 -- four decimals, nothing fitted.
    #
    # Flag only if BOTH margins are small: the map is at the floor AND the
    # gradient is starved beyond what the softmax Jacobian accounts for.
    # Measured separation at D=256: block 0 m_p=0.002 m_r=0.02; blocks 1-5
    # m_p=0.47-1.16 m_r=6.0-8.0. Three orders on m_r, two on m_p, stable from
    # t=40, and the thresholds are not tuned.
    #
    # Verified by three independent ablations at t=400, agreeing within 0.004:
    #   baseline 3.0531, output zeroed 3.0492, W_Q/W_K frozen 3.0528.
    # Xavier instead of the spectral initialiser stays at the floor (0.0801),
    # so the redundancy is structural, not an initialisation artefact.
    import math as _math
    def _attn_probe(_mdl, _H=4):
        _x, _ = get_batch()
        _out = []
        with torch.no_grad():
            _h = _mdl.te(_x) + _mdl.pe(torch.arange(_x.shape[1]))
            for _bm in _mdl.blocks:
                _B, _T, _Dm = _h.shape; _dh = _Dm // _H
                _q = _bm.attn.WQ(_h).view(_B,_T,_H,_dh).transpose(1,2)
                _k = _bm.attn.WK(_h).view(_B,_T,_H,_dh).transpose(1,2)
                _s = (_q @ _k.transpose(-2,-1)) / _math.sqrt(_dh)
                _s = _s.masked_fill(torch.triu(torch.ones(_T,_T),1).bool(),
                                    float("-inf"))
                _P = _s.softmax(-1).clamp_min(1e-12)
                _s2 = (_P**2).sum(-1); _s3 = (_P**3).sum(-1)
                _jf = float(torch.sqrt((_s2-2*_s3+_s2**2).clamp_min(0)).mean())
                _out.append((float(_P.max(-1).values.mean()),
                             _jf/_math.sqrt(_dh), _T))
                _h = _bm(_h)
        return _out
    model.zero_grad()
    _xx,_yy = get_batch(); model(_xx,_yy)[1].backward()
    _pr = _attn_probe(model)
    _N = _pr[0][2]
    _floor = sum(1.0/_i for _i in range(1,_N+1))/_N
    _pd = dict(model.named_parameters())
    _pruned = []
    for _bi,(_pm,_pred,_) in enumerate(_pr):
        _gq = _pd[f"blocks.{_bi}.attn.WQ.weight"].grad
        _gv = _pd[f"blocks.{_bi}.attn.WV.weight"].grad
        _r = (float(_gq.norm()) if _gq is not None else 0.0) / max(
              float(_gv.norm()) if _gv is not None else 1.0, 1e-30)
        _mp = (_pm-_floor)/_floor; _mr = _r/max(_pred,1e-30)
        if _mp < 0.10 and _mr < 1.0: _pruned.append(_bi)
    model.zero_grad()
    if _pruned:
        # Zero the gradient with a hook rather than setting requires_grad=False.
        # The parameter stays differentiable, so autograd.grad over
        # list(model.parameters()) still works -- the Snapper HVP, the Phase 5
        # LM step and the Lanczos projection all do that, and requires_grad=False
        # makes every one of them raise "One of the differentiated Tensors does
        # not require grad". With a zero gradient, m and v stay at zero and
        # Adam's m/(sqrt(v)+eps) is exactly zero, so the weight never moves.
        _np_ = 0
        for _bi in _pruned:
            for _W in (model.blocks[_bi].attn.WQ.weight,
                       model.blocks[_bi].attn.WK.weight):
                _np_ += _W.numel()
                _W.register_hook(lambda g: torch.zeros_like(g))
        print(f"  [zoneadam] PRUNED attention QK in blocks {_pruned} "
              f"({_np_:,} params frozen; floor p_max={_floor:.4f})")
    else:
        print(f"  [zoneadam] attention pruner: no block at the causal-uniform "
              f"floor ({_floor:.4f})")
NO_GEOSTOP      = __NGS__
VFREEZE_AT      = __VFA__
VFREEZE_BUCKETS = __VFB__
_vfroz_done     = False
opt_b = CompressedAdam(list(model.parameters()), lr=LR*5,
                       betas=(0.9,0.95), weight_decay=0.1,
                       names=[n for n,_ in model.named_parameters()])
_stab = StabDetector(model, layer=3)
print("  [zoneadam] Phase 3: CompressedAdam  m 4b/coord, v 4b/row, "
      "refresh every step")
print("  [zoneadam] Sstab monitor armed (reports only; Phase 3 is "
      "assimilation-regime)")
print(f"  [zoneadam] vhat freeze: {sorted(VFREEZE_BUCKETS)} at step {VFREEZE_AT}"
      if VFREEZE_AT else "  [zoneadam] vhat freeze: off")"""

PROBE_OLD = """        rm2 = compute_rm2_sigma_inline(model)
"""
PROBE_NEW = """        rm2 = compute_rm2_sigma_inline(model)
        _ss = _stab.check(step)
        if _ss is not None:
            _g=opt_b.geometry()
            print(f"  [zoneadam] Sstab={_ss:.4f}  bucket coh/theta/R4: "
                  + "  ".join(f"{k}={c:.2f}/{t_:.0f}/{r:.2f}"
                              for k,(c,t_,r) in sorted(_g.items())))
        # vhat freeze, evaluated at the existing probe so it costs nothing.
        # NOTE: the trigger is a STEP COUNT, not a geometric sensor. No sensor
        # in this compiler has been shown to detect vhat convergence -- around
        # the relevant window Phi_cl and tau show no distinctive signature --
        # so gating on one would be a step count in disguise. VFREEZE_GATE is
        # provided for when a validated sensor exists.
        if VFREEZE_AT and not _vfroz_done and step >= VFREEZE_AT:
            _n=opt_b.freeze_buckets(VFREEZE_BUCKETS); _vfroz_done=True
            print(f"  [zoneadam] vhat FROZEN at step {step} for "
                  f"{sorted(VFREEZE_BUCKETS)} ({_n:,} params)"
                  + (f"  Sstab={_ss:.4f}" if _ss is not None else ""))
"""

Z1_CONST = "ZONE1_BOOST = __Z1B__\nZONE1_END   = __Z1E__\n"
GEO_OLD = "        geo_ok = (pc >= 4 and 5.0 <= tau <= 7.5 and rm2 >= 0.65)"
GEO_NEW = GEO_OLD + """
        if NO_GEOSTOP:
            # Geo-stop disabled for controlled comparison. In a single
            # smoothly converging run (val 3.65 -> 0.41) Phi_clean reads
            # 3,4,3,4,4,5 and rm2sigma swings 0.740,0.731,0.706,0.789,
            # 0.692,0.682, so this condition flips on and off while
            # nothing is converging or diverging. Identical settings
            # geo-stopped at 64, at 96, and not at all -- a 2x swing in
            # the final value sitting on top of any optimiser change.
            # With it off Phase 3 always exits on the loss plateau into
            # the tau-retry, so arms differ only in the optimiser.
            geo_ok = False"""

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
    ap.add_argument("--eta-mf",type=float,default=None)
    ap.add_argument("--p3cap",type=int,default=150)
    ap.add_argument("--p3min",type=int,default=0)
    ap.add_argument("--p3plat",type=float,default=0.003)
    ap.add_argument("--p3valstop",type=float,default=0.15)
    ap.add_argument("--no-snapper",action="store_true")
    ap.add_argument("--no-topogate",action="store_true",
                    help="revert Phase 4; it costs 0.19 nats "
                         "alongside --struct-loss")
    ap.add_argument("--struct-loss",action="store_true",
                    help="LSE boundary on the observed-successor "
                         "support; KL 1.83 -> 0.78 measured")
    ap.add_argument("--prune-attn",action="store_true",
                    help="freeze W_Q,W_K of any block at the "
                         "causal-uniform floor (analytic H_N/N)")
    ap.add_argument("--no-geostop",action="store_true")
    ap.add_argument("--vbuckets",default="ATT_QK,ATT_VO",
                    help="buckets to freeze; measured best is attention-only")
    ap.add_argument("--vfreeze",type=int,default=40,
                    help="freeze vhat at this step; 0 disables")
    ap.add_argument("--run",action="store_true")
    a=ap.parse_args()
    s=open(a.src).read()

    edits=[("phase3 optimiser",P3_OLD,P3_NEW),("phase3 probe",PROBE_OLD,PROBE_NEW)]
    if a.eta_mf is not None: edits.append(("mf",MF_OLD,MF_NEW))
    if a.p3cap!=150: edits.append(("p3cap",P3CAP_OLD,P3CAP_NEW))
    if a.p3min>0: edits.append(("plateau",PLAT_OLD,PLAT_NEW))
    if a.no_snapper: edits.append(("snapper",SNAP_OLD,SNAP_NEW))
    if a.struct_loss: edits.append(("struct loss",SL_OLD,SL_NEW))
    if a.no_topogate:
        edits.append(("topogate snap",TG_OLD,TG_NEW))
        edits.append(("topogate revert",TG2_OLD,TG2_NEW))
    if a.no_geostop: edits.append(("geostop",GEO_OLD,GEO_NEW))
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
    s=s.replace("__MFE__",repr(float(a.eta_mf)) if a.eta_mf is not None else "0.01")
    s=s.replace("__P3CAP__",repr(int(a.p3cap)))
    s=s.replace("__P3MIN__",repr(int(a.p3min)))
    s=s.replace("__P3PLAT__",repr(float(a.p3plat)))
    s=s.replace("__P3VALSTOP__",repr(float(a.p3valstop)))
    s=s.replace("__STRUCT__",repr(bool(a.struct_loss)))
    s=s.replace("__PRUNE__",repr(bool(a.prune_attn)))
    s=s.replace("__NGS__",repr(bool(a.no_geostop)))
    s=s.replace("__VFA__",repr(int(a.vfreeze)))
    s=s.replace("__VFB__",repr(set(x for x in a.vbuckets.split(",") if x)))
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
