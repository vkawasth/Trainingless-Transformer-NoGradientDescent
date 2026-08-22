#!/usr/bin/env python3
"""
patch_3456789.py -- PHASE 3: COMPRESSION + LEAKAGE + SIGN + FISHER-WEIGHTED COMPLEMENT
====================================================================================

Four mechanisms, each switchable, applied to Phase 3 only. Phases 1, 2, 4 and 5
are byte-identical -- the corpus-specific spectral init, the MF pump, TopoGate and
the K0 split descent are untouched.

  (3) COMPRESSION
        u' = P_Q u + alpha_role * (I - P_Q) u
      rank k per role x 7 roles, frames rebuilt every 2 steps from an update
      history window. NOTE: an n-column history yields at most n singular
      vectors, so the window must be >= the rank or the frame is silently
      capped; the patcher raises it and reports the ACTUAL width.

  (4) LEAKAGE TRIGGER
      gamma = 1 - ||P_Q g||^2 / ||g||^2, energy weighted across roles. While
      gamma < theta the step COASTS -- no forward or backward pass.

  (5) SIGN TRANSPORT
      The coast step is a_t * sign(g) per role rather than Q @ coef. Draws on the
      measurement that dtheta ~ a_t sign(g) retains 100.1% of the loss reduction
      with ONE SCALAR per step, so a coast costs a multiply instead of a matmul.

  (13) TWO-TERM ODE PREDICTION               <-- new in this patch
      Fit  dL/dt = -alpha e_F |u|^2 + beta (1-e_F) |u|^2  on steps 0..PRED,
      then predict forward with no refitting. Tests whether the Fisher/complement
      split maps the geometry to the loss. The residual changes sign near step
      50 -- helping before, hurting after -- so this covers phases 2 and 3 of
      Phase 3, not the saddle exit.

  (8) PHASE-3 STEP CAP
      Phase 3 has three measured segments, and they differ in character:
        1-40    leaving the saddle (phi 128 -> 57 deg within 16 steps) and most
                of the loss drop, val 10 -> 2.5. Capture 0.886 and rising.
        40-100  the stable window. Capture peaks at 0.913, k90 bottoms at 1.62,
                flip rate and drift both fall. This is where projection is
                nearly free.
        100-170 loosening. Capture falls to 0.799, k90 climbs back to 2.19,
                rotation rises. Buys the last ~0.15 nats -- which the tau-retry
                at LR x2 largely duplicates anyway.
      --maxstep cuts the third segment. Boundaries are from ONE run and located
      to about +-12 steps, so treat 120 and 100 as a sweep rather than settings.

  (7) SIGN COMPLEMENT
      The orthogonal complement is applied as a scalar times its sign rather
      than as the scaled vector:  alpha * ||d||/sqrt(n) * sign(d).
      Norm preserved per block, so this is a DIRECTION-only change and cannot be
      a disguised step-size increase. Tests whether the complement's magnitudes
      are redundant: dtheta ~ a_t sign(g) was measured to retain 100.1% of the
      loss reduction with one scalar per step.

  (6) FISHER-WEIGHTED COMPLEMENT
      alpha is currently a fixed per-role constant, so the discarded complement
      is thrown away uniformly. But the complement is not uniform: three Fisher
      directions out of 4.3e6 were measured to deliver MORE loss reduction than
      the full update, on two corpora, while the residual INCREASED loss. One
      Fisher direction was 9.8-17x more efficient per unit motion than average.

      So (6) splits the complement instead of scaling it:
          u' = P_Q u + beta * P_F (I - P_Q) u + alpha_role * (I - P_F)(I - P_Q) u
      keeping the part that overlaps the Fisher sheet at weight beta (default
      1.0, i.e. kept in full) and the rest at alpha_role.

      The Fisher sheet is estimated by a randomised range finder on r extra
      gradients, refreshed every --fisher-every steps. It is used ONLY to
      weight what is already being discarded -- never as the frame itself.
      That matters: SPLIT_F was measured at 0.21-0.31 at this scale, so two
      independent Fisher sheets at the same theta agree only about a quarter of
      the time. A frame built on it would be unstable; a weighting is not, since
      an unreliable sheet degrades gracefully toward the uniform-alpha case.

VERIFIED CONFIGURATION

    python3 patch_3456789.py --no-trigger --k 8 --win 24 --geoval 0.6 --run
        final 0.0766 vs GD-400's 0.0914, 187 CE vs 400, gap to floor 0.0146 vs
        0.0294. Phase 3 ran 162 CE to plateau, capture 0.901, gamma 0.952.

    --geoval MATTERS. Without it the same configuration geo-stopped at step 32
    at val 2.99 and finished at 0.512: a better projection satisfied the orbit
    criteria before the loss came down, and the criteria alone cannot tell
    "orbit is clean" from "orbit is clean AND near the basin".

    k=4 also works. k=2 does not reach the basin (0.379).

WHAT IS UNTESTED

    (4) and (5) have never fired: gamma runs 0.92-0.98 and the old theta default
    of 0.20 was set from CAPTURE, which measures the update, not the gradient.
    Use --theta 0.96 to exercise them, and read the gamma trace under --verbose:
    if gamma is flat, the trigger is a fixed schedule rather than a geometric
    criterion, whatever the loss does.

    (6) is new and unmeasured here.

USAGE
  MEASURED SO FAR (all with --no-trigger --geoval 0.6, k=8 win 24)
      plain                       0.0766     gap +0.0146
      --fisher                    0.0540     gap -0.0080   <- best
      --fisher --frank 6          0.0549     gap -0.0071   (rank 3 is enough)
      --fisher --beta 2.0         0.1274     gap +0.0654   (over-weighting exits early:
                                                            Phase 3 ran 118 CE not 161)
      GD-400 reference            0.0914     gap +0.0294

    python3 patch_3456789.py --no-trigger --k 8 --win 24 --geoval 0.6 --run      # verified
    python3 patch_3456789.py --no-trigger --k 8 --win 24 --geoval 0.6 --fisher --run
    python3 patch_3456789.py --k 8 --win 24 --geoval 0.6 --theta 0.96 --verbose --run
    python3 patch_3456789.py --no-trigger --k 8 --win 24 --geoval 0.6 --fisher --beta 0.5 --run

Needs compiler_geometri_patched_86.py and build_corpus.py alongside.
"""

import argparse
import os
import subprocess
import sys

SNAP_OLD = """SNAPPER_STEP = 0.1
SNAPPER_POINTS = 5
t_vals = np.array([i * SNAPPER_STEP for i in range(SNAPPER_POINTS)])"""

SNAP_NEW = """SNAPPER_STEP = 0.1
SNAPPER_POINTS = 5
# (9) t* lands at 0.040-0.054 in every run measured, i.e. inside the FIRST gap of
# a uniform 0.1 grid -- the quartic interpolates a minimum it never sampled, and
# the jump then lands worse than t=0 (0.0946 -> 0.0955, 0.1191 -> 0.1242).
# Concentrate points where the minimum actually is.
t_vals = np.array(__TVALS__)"""

ANCHOR = """opt_b = torch.optim.AdamW(model.parameters(), lr=LR*5,
                           betas=(0.9,0.95), weight_decay=0.1)"""

GEO2_OLD = """        geo_ok = (pc >= 4 and 5.0 <= tau <= 7.5 and rm2 >= 0.65)"""
GEO2_NEW = """        geo_ok = (pc >= 4 and 5.0 <= tau <= 7.5 and rm2 >= 0.65
                  and _sub_ready(step, v))"""

OLD_STEP = """    model.train(); x, y = get_batch(); _, l = model(x, y)
    opt_b.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt_b.step()

    if step % 8 == 0:"""

NEW_STEP = """    if _SUB_MAXSTEP and step >= _SUB_MAXSTEP:
        print(f"  \u2713 step cap ({_SUB_MAXSTEP}) \u2014 handing off to Phase 4")
        break
    if _sub_should_skip(step):
        _sub_coast(step)
    else:
        model.train(); x, y = get_batch(); _, l = model(x, y)
        _sub_th = _sub_flat()
        opt_b.zero_grad(); l.backward()
        _sub_note_grad()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt_b.step()
        _sub_project(_sub_th, step)

    if step % 8 == 0:"""

PATCH = '''
# ─────────────────────────────────────────────────────────────
# PATCH 3+4+5+6+7+8+9 -- PHASE 3 ONLY
#   (3) compression : u' = P_Q u + alpha (I - P_Q) u,  __DIMS__ dims
#   (4) leakage     : coast while gamma < theta, no backward pass
#   (5) sign        : coast step is a_t * sign(g) per role
#   (6) fisher      : keep the Fisher-aligned part of the complement at beta
# ─────────────────────────────────────────────────────────────
_SUB_K, _SUB_WIN, _SUB_REBUILD = __K__, __WIN__, __REBUILD__
_SUB_ALPHA    = __ALPHA__
_SUB_TRIGGER  = __TRIGGER__
_SUB_SIGN     = __SIGN__
_SUB_ACCUM    = __ACCUM__
_SUB_RESCALE  = __RESCALE__
_SUB_FISHER   = __FISHER__
_SUB_BETA     = __BETA__
_SUB_FR       = __FRANK__
_SUB_FEVERY   = __FEVERY__
_SUB_THETA    = __THETA__
_SUB_MAXCOAST = __MAXCOAST__
_SUB_VERBOSE  = __VERBOSE__
_SUB_GEOVAL   = __GEOVAL__
_SUB_SIGNCOMP = __SIGNCOMP__   # (7) complement applied as scalar x sign
_SUB_MAXSTEP  = __MAXSTEP__    # (8) hard cap on Phase 3's main loop
_SUB_DELTASTOP= __DELTASTOP__  # (10) project steps-to-target instead of a cap
_SUB_TARGET   = __TARGET__
_SUB_BUDGET   = __BUDGET__
_SUB_RHO      = __RHO__        # (11) hand off when rho = |dATTN|^2/|dFF|^2 >= this
_SUB_PL       = __PL__         # (12) PL stop: hand off at this fraction of the fitted floor
_SUB_PRED     = __PRED__       # (13) fit the two-term ODE on 0..PRED, predict after

def _sub_role(nm):
    if nm.startswith("te") or nm.startswith("pe"): return "EMB"
    if "ln" in nm.lower():                         return "LN"
    if ".ff." in nm:                               return "FF"
    if "WQ" in nm:                                 return "W_Q"
    if "WK" in nm:                                 return "W_K"
    if "WV" in nm:                                 return "W_V"
    if "WO" in nm or "op" in nm:                   return "W_O"
    return "EMB"

_sub_ps = [p for p in model.parameters() if p.requires_grad]
_sub_P  = sum(p.numel() for p in _sub_ps)
_sub_idx, _o = {}, 0
for _nm, _p in model.named_parameters():
    if _p.requires_grad:
        _sub_idx.setdefault(_sub_role(_nm), []).append(torch.arange(_o, _o + _p.numel()))
    _o += _p.numel()
_sub_idx   = {k: torch.cat(v) for k, v in _sub_idx.items()}
_sub_hist  = {k: [] for k in _sub_idx}
_sub_frame = {}
_sub_QF    = None          # Fisher sheet, global
_sub_gbuf  = []            # recent gradients, for the range finder
_sub_coef  = None
_sub_sign  = None
_sub_scale = {}
_sub_bank  = torch.zeros(_sub_P)
_sub_unrm  = {}
_sub_last_u = {}
_sub_stats = {"bwd": 0, "coast": 0, "fold": 0, "cap": [], "gamma": [], "run": 0,
              "unorm": [], "dims": 0, "rs": [], "fshare": []}

def _sub_flat():
    return torch.cat([p.data.reshape(-1) for p in _sub_ps]).clone()

def _sub_gflat():
    return torch.cat([(p.grad.reshape(-1) if p.grad is not None
                       else torch.zeros(p.numel())) for p in _sub_ps])

def _sub_note_grad():
    """energy-weighted leakage of the current gradient against the frame in force,
    and a gradient sample for the Fisher range finder.
    Energy weighted rather than max: LN has far fewer params than FF and a max
    would let the smallest block dictate the schedule."""
    g = _sub_gflat()
    if _SUB_FISHER:
        _sub_gbuf.append(g.detach().clone())
        if len(_sub_gbuf) > max(_SUB_FR * 2, 8):
            _sub_gbuf.pop(0)
    if not _sub_frame:
        return
    num = den = 0.0
    for k, ii in _sub_idx.items():
        gb = g[ii]; e = float((gb * gb).sum())
        if e <= 0.0:
            continue
        Q = _sub_frame[k]; pj = Q @ (Q.T @ gb)
        num += e * (1.0 - float((pj * pj).sum()) / e); den += e
    if den > 0.0:
        _sub_stats["gamma"].append(num / den)

def _sub_refresh_fisher():
    """randomised range finder on the recent gradient buffer: Y = G Omega, QR.
    Cheap and reuses gradients already computed. Used ONLY to weight the
    complement -- never as the frame."""
    global _sub_QF
    if len(_sub_gbuf) < _SUB_FR:
        return
    Gm = torch.stack(_sub_gbuf[-max(_SUB_FR * 2, 8):], 1)
    Om = torch.randn(Gm.shape[1], _SUB_FR)
    _sub_QF = torch.linalg.qr(Gm @ Om)[0][:, :_SUB_FR]

def _sub_comp(d):
    """(7) the complement as a scalar times its sign, norm preserved.
    dtheta ~ a_t sign(g) was measured to retain 100.1% of the loss reduction with
    ONE scalar, so the complement's magnitudes may be largely redundant. a is set
    to preserve the block's norm: a = ||d|| / sqrt(n), so this changes DIRECTION
    only and cannot be a disguised step-size change."""
    if not _SUB_SIGNCOMP:
        return d
    n = d.numel()
    if n == 0:
        return d
    return torch.sign(d) * (float(d.norm()) / (n ** 0.5))

def _sub_should_skip(step):
    if not _SUB_TRIGGER or not _sub_frame:
        return False
    if _sub_coef is None and _sub_sign is None:
        return False
    if _sub_stats["run"] >= _SUB_MAXCOAST or not _sub_stats["gamma"]:
        return False
    return _sub_stats["gamma"][-1] < _SUB_THETA

def _sub_coast(step):
    _sub_stats["coast"] += 1; _sub_stats["run"] += 1
    nu = torch.zeros(_sub_P)
    if _SUB_SIGN and _sub_sign is not None:
        for k, ii in _sub_idx.items():
            nu[ii] = _sub_scale.get(k, 0.0) * _sub_sign[ii]
    else:
        for k, ii in _sub_idx.items():
            nu[ii] = _sub_frame[k] @ _sub_coef[k]
    with torch.no_grad():
        i = 0
        for p in _sub_ps:
            n = p.numel(); p.data.add_(nu[i:i + n].view_as(p)); i += n

def _sub_project(th_before, step):
    global _sub_coef, _sub_sign, _sub_bank
    _sub_stats["bwd"] += 1; _sub_stats["run"] = 0
    u = _sub_flat() - th_before
    _sub_stats["unorm"].append(float(u.norm()))
    for k, ii in _sub_idx.items():
        _sub_hist[k].append(u[ii].clone())
        if len(_sub_hist[k]) > max(_SUB_WIN, _SUB_K):
            _sub_hist[k].pop(0)
    if len(_sub_hist["FF"]) < max(_SUB_WIN, _SUB_K):
        return                                     # warm-up: keep the full update
    if step % _SUB_REBUILD == 0 or not _sub_frame:
        for k, ii in _sub_idx.items():
            A = torch.stack(_sub_hist[k], 1)
            _sub_frame[k] = torch.linalg.svd(A, full_matrices=False)[0][:, :_SUB_K]
        _sub_stats["dims"] = sum(f.shape[1] for f in _sub_frame.values())
    if _SUB_FISHER and (step % _SUB_FEVERY == 0 or _sub_QF is None):
        _sub_refresh_fisher()

    _sub_sign = torch.sign(u)
    for k, ii in _sub_idx.items():
        s = _sub_sign[ii]
        _sub_scale[k] = float((u[ii] * s).sum()) / max(float((s * s).sum()), 1e-30)

    # (6) split the complement: Fisher-aligned part at beta, the rest at alpha
    drop_full = torch.zeros(_sub_P)
    nu = torch.zeros_like(u); coef = {}
    cap = tot = 0.0
    for k, ii in _sub_idx.items():
        Q, ub = _sub_frame[k], u[ii]
        c = Q.T @ ub
        par = Q @ c
        coef[k] = c
        drop_full[ii] = ub - par
        nu[ii] = par
        cap += float((par * par).sum()); tot += float((ub * ub).sum())
        _sub_unrm[k] = float((ub * ub).sum())
        _sub_last_u[k] = ub.clone()
    _ea = sum(_sub_unrm.get(z, 0.0) for z in ("W_Q", "W_K", "W_V", "W_O"))
    _ef = _sub_unrm.get("FF", 0.0)
    _sub_rho_val[0] = _ea / max(_ef, 1e-30)
    _sub_rhohist.append(_sub_rho_val[0])
    if _SUB_FISHER and _sub_QF is not None:
        dF = _sub_QF @ (_sub_QF.T @ drop_full)     # Fisher-aligned part of the drop
        dR = drop_full - dF
        dn = float(drop_full.norm()) ** 2
        if dn > 0.0:
            _sub_stats["fshare"].append(float((dF * dF).sum()) / dn)
        nu = nu + _SUB_BETA * dF
        for k, ii in _sub_idx.items():
            nu[ii] = nu[ii] + _SUB_ALPHA.get(k, 0.1) * _sub_comp(dR[ii])
        if _SUB_ACCUM:
            for k, ii in _sub_idx.items():
                _sub_bank[ii] += (1.0 - _SUB_ALPHA.get(k, 0.1)) * dR[ii]
    else:
        for k, ii in _sub_idx.items():
            nu[ii] = nu[ii] + _SUB_ALPHA.get(k, 0.1) * _sub_comp(drop_full[ii])
            if _SUB_ACCUM:
                _sub_bank[ii] += (1.0 - _SUB_ALPHA.get(k, 0.1)) * drop_full[ii]
    _sub_coef = coef
    _sub_stats["cap"].append(cap / max(tot, 1e-30))

    if _SUB_RESCALE:
        _n_u, _n_nu = float(u.norm()), float(nu.norm())
        if _n_nu > 1e-30:
            nu = nu * (_n_u / _n_nu)
            _sub_stats["rs"].append(_n_u / _n_nu)

    if _SUB_ACCUM and _sub_stats["unorm"]:
        import numpy as _np
        if float(_sub_bank.norm()) > 0.5 * _np.mean(_sub_stats["unorm"][-8:]):
            nu = nu + _sub_bank
            _sub_bank = torch.zeros(_sub_P)
            _sub_stats["fold"] += 1

    with torch.no_grad():
        i = 0
        for p in _sub_ps:
            n = p.numel(); p.data.copy_((th_before + nu)[i:i + n].view_as(p)); i += n

    if _SUB_VERBOSE and step % 16 == 0:
        import numpy as _np
        gm = _np.mean(_sub_stats["gamma"][-16:]) if _sub_stats["gamma"] else float("nan")
        fs = (f"  fisher-share {_np.mean(_sub_stats['fshare'][-16:]):.3f}"
              if _sub_stats["fshare"] else "")
        print(f"    [3456789] step {step:3d}  capture {_np.mean(_sub_stats['cap'][-16:]):.3f}"
              f"  gamma {gm:.3f}  bwd {_sub_stats['bwd']}  coast {_sub_stats['coast']}{fs}")

_sub_vhist = []
_sub_rhohist = []
_sub_rho_val = [0.0]
def _sub_rho():
    """(11) rho = ||dtheta ATTN||^2 / ||dtheta FF||^2, the FF -> attention handover.

    Measured on the UPDATE, not the gradient. Two earlier versions were wrong:

      v1 read p.grad at the geo_ok check, but the geometry probes (phi_clean,
         gluing_defect) run their own backwards and clear it -- rho printed
         0.000 at every check.
      v2 took its own backward, which fixed the zeros but (a) consumed batches
         from the training stream, shifting the whole trajectory, and (b) still
         measured the GRADIENT.

    (b) is the substantive error. The handover was measured in update energy:
        steps    8-24   24-40   40-80  80-120  120-170
        rho     0.374   0.240   0.551   1.214    1.028
    Update and gradient differ by exactly Adam's preconditioner, and the
    gradient version ran 1.691 -> 0.36 -> 0.51, never crossing 1.

    This version reads the per-role update norms already computed in
    _sub_project for the capture statistic. No extra backward, no batches
    consumed, and it measures the quantity the handover actually lives in."""
    return _sub_rho_val[0]

_sub_obs = []          # (step, val, e_F, |u|^2)
_sub_fit = [None]
def _sub_predict(step, v):
    """(13) fit dL/dt = -alpha e_F |u|^2 + beta (1-e_F) |u|^2, then PREDICT.

    Measured by ablation on the real pipeline at six checkpoints:

        step   val    e_F     dL Fisher   dL residual
          20  3.662  0.132     -0.244      -0.0486
          40  2.045  0.151     -0.174      -0.0027
          60  1.160  0.060     -0.029      +0.0051
          80  0.684  0.094     -0.061      +0.0291
         120  0.265  0.099     -0.037      +0.0167
         240  0.061  0.049     -0.008      +0.0023

    The Fisher sheet descends at every checkpoint -- 5-15% of the update energy
    delivering 56-105% of the loss reduction. But the RESIDUAL changes sign near
    step 50: before it, the complement helps; after, it hurts. So beta > 0 only
    holds past the backbone window, and the two-term form describes phases 2 and
    3 of Phase 3, not phase 1.

    This fits alpha and beta on steps 0..PRED and then predicts forward without
    refitting. It is a SEMI-EMPIRICAL test: e_F and |u|^2 are measured at each
    step and fed in, so what is being tested is whether the two-term form maps
    the geometry to the loss -- not whether the loss can be predicted from
    nothing. A closed prediction would also have to predict e_F, which nothing
    here can do (e_F is non-monotone: 0.132, 0.151, 0.060, 0.094, 0.099, 0.049).

    Printed each check past PRED: predicted vs actual, and the running error."""
    import numpy as _np
    if _sub_QF is None or not _sub_unrm:
        return
    u2 = sum(_sub_unrm.values())
    if u2 <= 0:
        return
    # e_F: fraction of the update inside the Fisher sheet
    _u = torch.zeros(_sub_P)
    for k, ii in _sub_idx.items():
        _u[ii] = _sub_last_u.get(k, torch.zeros(len(ii)))
    pj = _sub_QF @ (_sub_QF.T @ _u)
    eF = float((pj * pj).sum()) / max(float((_u * _u).sum()), 1e-30)
    _sub_obs.append((step, float(v), eF, u2))
    if len(_sub_obs) < 3:
        return
    if step <= _SUB_PRED:
        # least squares for alpha, beta on the observed dL so far
        A, b = [], []
        for i in range(1, len(_sub_obs)):
            s0, v0, e0, q0 = _sub_obs[i - 1]
            s1, v1, e1, q1 = _sub_obs[i]
            dt = max(s1 - s0, 1)
            em, qm = (e0 + e1) / 2, (q0 + q1) / 2
            A.append([-em * qm * dt, (1 - em) * qm * dt])
            b.append(v1 - v0)
        A = _np.array(A); b = _np.array(b)
        try:
            c, *_ = _np.linalg.lstsq(A, b, rcond=None)
            _sub_fit[0] = (float(c[0]), float(c[1]), float(_sub_obs[-1][1]),
                           _sub_obs[-1][0])
        except Exception:
            pass
        if _SUB_VERBOSE and _sub_fit[0]:
            al, be, _, _ = _sub_fit[0]
            print(f"    [ode] step {step} FIT  alpha {al:+.4f}  beta {be:+.5f}"
                  f"  a/b {al/be if abs(be)>1e-9 else float('nan'):+.1f}  e_F {eF:.4f}")
        return
    if _sub_fit[0] is None:
        return
    al, be, L0, s0 = _sub_fit[0]
    # integrate forward from the anchor using MEASURED e_F and |u|^2
    Lp = L0
    for i in range(1, len(_sub_obs)):
        sa, _, ea, qa = _sub_obs[i - 1]
        sb, _, eb, qb = _sub_obs[i]
        if sb <= s0:
            continue
        dt = max(sb - sa, 1)
        em, qm = (ea + eb) / 2, (qa + qb) / 2
        Lp += dt * (-al * em * qm + be * (1 - em) * qm)
    err = (Lp - v) / max(v, 1e-9)
    print(f"    [ode] step {step}  predicted {Lp:.4f}  actual {v:.4f}  "
          f"err {100*err:+.1f}%   e_F {eF:.4f}")

def _sub_pl(step, v):
    """(12) the loss half of the stopping rule, fitted rather than assumed.

    Phase 3's loss is exponential with a floor, measured over 17 points spanning
    two orders of magnitude:

        L(t) = 0.120 + 12.63 exp(-0.0434 t)        R2 = 0.974
        plain exponential                          log-R2 = 0.986
        power law                                  R2 = 0.185  (refuted)

    dL/dt = -k (L - L*) means ||grad L||^2 is proportional to (L - L*), which is
    the Polyak-Lojasiewicz condition holding with EQUALITY at k = 0.043/step.
    PL is the standard assumption under which gradient descent converges linearly
    without convexity; here it is measured.

    The fitted floor 0.120 sits within a factor of two of the compiler's
    independently measured val_floor = 0.062. Phase 3 decays toward a barrier it
    cannot cross -- which is what Phases 4 and 5 are for.

    So: fit L* and k online from the observed history, and hand off once L is
    within _SUB_PL of the fitted floor. Unlike a fixed val ceiling this adapts to
    where the barrier actually is on this run.

    Caveat: the instantaneous rate runs 0.078 early to 0.015 late, a 183%
    spread, so the exponential is a good GLOBAL fit with a systematic early-fast
    deviation. Early fits will overestimate k."""
    import numpy as _np
    H = [(a, b) for a, b in _sub_vhist if b > 0]
    if len(H) < 6:
        return False
    t = _np.array([a for a, b in H], float)
    y = _np.array([b for a, b in H], float)
    best = None
    for Ls in _np.linspace(0.0, 0.9 * y.min(), 25):
        m = y > Ls + 1e-4
        if m.sum() < 5:
            continue
        c = _np.polyfit(t[m], _np.log(y[m] - Ls), 1)
        if c[0] >= 0:
            continue
        pr = Ls + _np.exp(_np.polyval(c, t))
        ss = 1 - ((y - pr) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-30)
        if best is None or ss > best[0]:
            best = (ss, Ls, -c[0])
    if best is None:
        return False
    ss, Ls, k = best
    close = (v - Ls) / max(v, 1e-9)
    if _SUB_VERBOSE:
        print(f"    [PL] step {step} val {v:.4f}  fit L*={Ls:.4f} k={k:.4f} "
              f"R2={ss:.3f}  gap {close:.3f} vs {_SUB_PL}")
    return ss > 0.8 and close < _SUB_PL

def _sub_ready(step, v):
    """(10) the loss half of the stopping rule.

    The orbit criteria say the geometry has converged; they say nothing about
    how far the basin still is. A better projection satisfied them at step 32 at
    val 2.99 and the pipeline finished at 0.512 instead of 0.077.

    --geoval fixes that with a hardcoded ceiling. This is the adaptive form: fit
    the decay of the loss decrement and project how many steps remain to the
    target. Hand off when the geometry is ready AND the projection says the
    remaining descent is expensive -- which is the actual condition for Phase 4
    being the better instrument.

    Falls back to the fixed ceiling when --deltastop is off, and to the ceiling
    alone until there are enough points to fit."""
    _sub_vhist.append((step, float(v)))
    if _SUB_PRED > 0:
        _sub_predict(step, v)
    if _SUB_PL > 0:
        return _sub_pl(step, v) and v <= _SUB_GEOVAL
    if _SUB_RHO > 0:
        r = _sub_rho()
        ok = (r >= _SUB_RHO) and (v <= _SUB_GEOVAL)
        if _SUB_VERBOSE:
            print(f"    [rho] step {step} val {v:.4f} rho {r:.3f} "
                  f"vs {_SUB_RHO}  -> {'HANDOFF' if ok else 'build'}")
        return ok
    if not _SUB_DELTASTOP:
        return v <= _SUB_GEOVAL
    if v > _SUB_GEOVAL:
        return False                      # never hand off above the ceiling
    H = _sub_vhist[-5:]
    if len(H) < 4:
        return False
    import numpy as _np
    d = [(H[i+1][0] - H[i][0], H[i][1] - H[i+1][1]) for i in range(len(H)-1)]
    rate = [max(dv, 1e-9) / max(ds, 1) for ds, dv in d]
    if min(rate) <= 0:
        return True                       # not descending: nothing left here
    y = _np.log(_np.array(rate))
    x = _np.arange(len(y), dtype=float)
    sl = _np.polyfit(x, y, 1)[0]          # log-rate slope per check interval
    r_now = rate[-1]
    gap = float(v) - _SUB_TARGET
    if gap <= 0:
        return True
    if sl >= -1e-6:                       # rate not decaying: linear projection
        proj = gap / max(r_now, 1e-9)
    else:                                 # geometric decay, summed
        per = _np.mean([H[i+1][0]-H[i][0] for i in range(len(H)-1)])
        k = _np.exp(sl)
        tot = r_now * per / max(1.0 - k, 1e-9)
        proj = float("inf") if tot < gap else gap / max(r_now, 1e-9)
    if _SUB_VERBOSE:
        print(f"    [ready] step {step} val {v:.4f} rate {r_now:.5f}/step "
              f"slope {sl:+.3f} -> proj {proj:.0f} steps vs budget {_SUB_BUDGET}")
    return proj > _SUB_BUDGET

def _sub_report():
    import numpy as _np
    b, c = _sub_stats["bwd"], _sub_stats["coast"]
    cap = _np.mean(_sub_stats["cap"]) if _sub_stats["cap"] else float("nan")
    gm  = _np.mean(_sub_stats["gamma"]) if _sub_stats["gamma"] else float("nan")
    mode = ("sign" if (_SUB_TRIGGER and _SUB_SIGN) else
            "frame" if _SUB_TRIGGER else "off")
    extra = ""
    if _SUB_ACCUM:
        extra += f"  folds {_sub_stats['fold']}"
    if _SUB_RESCALE and _sub_stats["rs"]:
        extra += f"  rescale x{_np.mean(_sub_stats['rs']):.3f}"
    if _sub_rhohist:
        extra += f"  rho {_np.mean(_sub_rhohist[-16:]):.3f}"
    if _sub_stats["fshare"]:
        extra += f"  fisher-share {_np.mean(_sub_stats['fshare']):.3f}"
    print(f"  [3456789] dims {_sub_stats['dims']} (asked __DIMS__)/{_sub_P}  coast:{mode}  "
          f"backward {b}  coasted {c}  skip {c/max(b+c,1):.0%}  "
          f"capture {cap:.3f}  gamma {gm:.3f}{extra}")
# ─────────────────────── end patch 3+4+5+6+7+8+9 ───────────────────────
'''

REPORT_ANCHOR = """step_basin = step"""
REPORT_NEW = """_sub_report()
step_basin = step"""

ALPHA_BY_ROLE = {"EMB": 0.05, "LN": 0.05, "W_Q": 0.10, "W_K": 0.10,
                 "FF": 0.15, "W_O": 0.15, "W_V": 0.15}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="compiler_geometri_patched_86.py")
    ap.add_argument("--out", default="compiler_phase3456789.py")
    ap.add_argument("--k", type=int, default=8, help="rank per role (7 roles)")
    ap.add_argument("--win", type=int, default=24, help="update history length")
    ap.add_argument("--rebuild", type=int, default=2)
    ap.add_argument("--alpha", type=float, default=None,
                    help="single alpha for all roles; omit for per-role defaults")
    ap.add_argument("--geoval", type=float, default=float("inf"),
                    help="geo-stop additionally requires val <= this. Without it a "
                         "better projection satisfies the orbit criteria while the "
                         "loss is still high. 0.6 is the verified setting; 0 "
                         "disables geo-stop entirely.")
    ap.add_argument("--theta", type=float, default=0.20,
                    help="leakage threshold. Measured gamma is 0.92-0.98, so the "
                         "trigger never fires below that -- use ~0.96 to exercise it.")
    ap.add_argument("--maxcoast", type=int, default=3)
    ap.add_argument("--fisher", action="store_true",
                    help="(6) keep the Fisher-aligned part of the complement at beta")
    ap.add_argument("--beta", type=float, default=1.0,
                    help="weight on the Fisher-aligned complement (1.0 = keep in full)")
    ap.add_argument("--frank", type=int, default=3, help="Fisher sheet rank")
    ap.add_argument("--fisher-every", type=int, default=8, dest="fevery",
                    help="steps between Fisher sheet refreshes")
    ap.add_argument("--no-trigger", action="store_true", help="disable (4)")
    ap.add_argument("--no-sign", action="store_true", help="disable (5)")
    ap.add_argument("--snapn", type=int, default=4,
                    help="(9) eval batches per Snapper point. The default 4 gives "
                         "a 66%% swing between adjacent t (0.0571 -> 0.0949 -> "
                         "0.0687), so the quartic fits noise, not curvature -- it "
                         "placed t* at 0.0425 predicting 0.0756 while a SAMPLED "
                         "point at t=0.025 measured 0.0571. Raise to 16 or 32.")
    ap.add_argument("--tvals", default="0,0.025,0.05,0.075,0.1,0.2,0.4",
                    help="(9) Snapper sample points. Default concentrates near "
                         "t=0.05 where the minimum was measured to land in every "
                         "run; the stock uniform 0,.1,.2,.3,.4 never samples it.")
    ap.add_argument("--predict", type=int, default=0, dest="pred",
                    help="(13) fit the two-term Fisher ODE on steps up to this, "
                         "then predict the loss forward without refitting. "
                         "Needs --fisher. The residual changes sign near step 50, "
                         "so fitting past 50 is fitting the regime where beta>0 "
                         "holds; fitting before it mixes two regimes. Try 50, 80.")
    ap.add_argument("--pl", type=float, default=0.0,
                    help="(12) hand off when the loss is within this FRACTION of "
                         "its own fitted floor. L = L* + A exp(-k t) is fitted "
                         "online (measured R2 0.974 offline, k = 0.043/step, "
                         "L* = 0.120 against the compiler's val_floor 0.062). "
                         "0.5 means hand off when half the remaining gap to the "
                         "barrier is gone. 0 = off.")
    ap.add_argument("--rho", type=float, default=0.0,
                    help="(11) hand off when rho = |grad ATTN|^2/|grad FF|^2 "
                         "reaches this AND val <= --geoval. rho tracks the "
                         "FF->attention handover and rises monotonically "
                         "(0.240 -> 0.551 -> 1.214) where Phi_cl oscillates. "
                         "1.0 is where attention overtook FF. 0 = off.")
    ap.add_argument("--deltastop", action="store_true",
                    help="(10) replace the fixed --geoval ceiling with a "
                         "projection: fit the decay of the loss decrement and "
                         "hand off when geometry is ready AND the projected "
                         "steps to --target exceed --budget.")
    ap.add_argument("--target", type=float, default=0.15,
                    help="(10) loss the projection aims at")
    ap.add_argument("--budget", type=int, default=60,
                    help="(10) steps Phase 3 is willing to spend reaching it")
    ap.add_argument("--maxstep", type=int, default=0,
                    help="(8) cap Phase 3's main loop at this many steps, then "
                         "hand off. 0 = no cap. Measured phase structure: steps "
                         "1-40 leave the saddle and carry most of the loss drop "
                         "(val 10 -> 2.5); 40-100 is the stable window with "
                         "capture peaking at 0.913; past 100 capture falls to "
                         "0.799 while k90 climbs back, and the segment buys the "
                         "last ~0.15 nats that the tau-retry largely duplicates. "
                         "Try 120, then 100.")
    ap.add_argument("--signcomp", action="store_true",
                    help="(7) apply the orthogonal complement as scalar x sign on "
                         "REAL steps, norm preserved. Tests whether the "
                         "complement's magnitudes are redundant -- dtheta ~ a_t "
                         "sign(g) retained 100.1%% of the loss reduction with one "
                         "scalar. Direction-only change: the block norm is held.")
    ap.add_argument("--accum", action="store_true", help="residual banking")
    ap.add_argument("--rescale", action="store_true",
                    help="restore the projected step to Adam's norm")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()

    if not os.path.exists(a.src):
        sys.exit(f"not found: {a.src}")
    src = open(a.src).read()
    for anc in (ANCHOR, OLD_STEP, REPORT_ANCHOR):
        if anc not in src:
            sys.exit("anchors not found -- this patcher targets "
                     "compiler_geometri_patched_86.py")

    if a.win < a.k:
        print(f"  note: window {a.win} < rank {a.k}; raising window to {2*a.k} "
              f"(an n-column history yields at most n singular vectors)")
        a.win = 2 * a.k
    alpha = ALPHA_BY_ROLE if a.alpha is None else {k: a.alpha for k in ALPHA_BY_ROLE}
    dims = a.k * 7
    inf = float("inf")
    patch = (PATCH.replace("__K__", str(a.k))
                  .replace("__WIN__", str(a.win))
                  .replace("__REBUILD__", str(a.rebuild))
                  .replace("__ALPHA__", repr(alpha))
                  .replace("__TRIGGER__", str(not a.no_trigger))
                  .replace("__SIGN__", str(not a.no_sign))
                  .replace("__ACCUM__", str(bool(a.accum)))
                  .replace("__RESCALE__", str(bool(a.rescale)))
                  .replace("__FISHER__", str(bool(a.fisher)))
                  .replace("__BETA__", repr(a.beta))
                  .replace("__FRANK__", str(a.frank))
                  .replace("__FEVERY__", str(a.fevery))
                  .replace("__THETA__", str(a.theta))
                  .replace("__MAXCOAST__", str(a.maxcoast))
                  .replace("__VERBOSE__", str(bool(a.verbose)))
                  .replace("__SIGNCOMP__", str(bool(a.signcomp)))
                  .replace("__MAXSTEP__", str(a.maxstep))
                  .replace("__DELTASTOP__", str(bool(a.deltastop)))
                  .replace("__TARGET__", repr(a.target))
                  .replace("__BUDGET__", str(a.budget))
                  .replace("__RHO__", repr(a.rho))
                  .replace("__PL__", repr(a.pl))
                  .replace("__PRED__", str(a.pred))
                  .replace("__GEOVAL__", "float('inf')" if a.geoval == inf else repr(a.geoval))
                  .replace("__DIMS__", str(dims)))

    out = (src.replace(ANCHOR, ANCHOR + "\n" + patch, 1)
              .replace(OLD_STEP, NEW_STEP, 1)
              .replace(REPORT_ANCHOR, REPORT_NEW, 1))
    tv = "[" + a.tvals + "]"
    SN_OLD = "    v = eval_val(model, n=4)\n    loss_vals.append(v)"
    SN_NEW = f"    v = eval_val(model, n={a.snapn})\n    loss_vals.append(v)"
    if SN_OLD in out:
        out = out.replace(SN_OLD, SN_NEW, 1)
    elif a.snapn != 4:
        print("  warning: Snapper eval anchor not found, n unchanged")
    if SNAP_OLD in out:
        out = out.replace(SNAP_OLD, SNAP_NEW.replace("__TVALS__", tv), 1)
    else:
        print("  warning: Snapper grid anchor not found, grid unchanged")
    if a.geoval != inf or a.deltastop or a.rho > 0 or a.pl > 0 or a.pred > 0:
        if GEO2_OLD not in out:
            sys.exit("geo-stop anchor not found")
        out = out.replace(GEO2_OLD, GEO2_NEW, 1)
    open(a.out, "w").write(out)

    print(f"wrote {a.out}")
    print(f"  (3) compression : rank {a.k}/role x 7 = {dims} dims, window {a.win}, "
          f"rebuild every {a.rebuild}")
    print(f"      alpha {alpha}")
    print(f"  (4) leakage     : {'OFF' if a.no_trigger else f'theta {a.theta}, max coast {a.maxcoast}'}")
    print(f"  (5) sign coast  : {'OFF (coast on frame)' if a.no_sign else 'ON'}")
    print(f"  (6) fisher      : {f'ON  beta {a.beta}, rank {a.frank}, refresh {a.fevery}' if a.fisher else 'OFF'}")
    print(f"  (7) sign comp   : {'ON -- complement as scalar x sign' if a.signcomp else 'OFF'}")
    print(f"  (13) ODE predict: {f'fit to step {a.pred}, then forecast' if a.pred else 'OFF'}")
    print(f"  (12) PL stop    : {f'handoff within {a.pl} of fitted floor' if a.pl else 'OFF'}")
    print(f"  (11) rho stop   : {f'handoff at rho >= {a.rho}' if a.rho else 'OFF'}")
    print(f"  (9) snapper     : t = {a.tvals}   n={a.snapn} batches/point")
    print(f"  (10) deltastop  : {f'target {a.target}, budget {a.budget}' if a.deltastop else 'OFF (fixed geoval)'}")
    print(f"  (8) step cap    : {a.maxstep if a.maxstep else 'none'}")
    print(f"  (R) accumulate  : {'ON' if a.accum else 'OFF'}")
    print(f"  (N) rescale     : {'ON' if a.rescale else 'OFF (honest)'}")
    print(f"  geo-stop val    : {'unchanged' if a.geoval == inf else f'also requires val <= {a.geoval}'}")
    print(f"  phases 1, 2, 4, 5 unchanged")

    if a.run:
        if not os.path.exists("/tmp/train_ids.json"):
            print("building corpus...")
            subprocess.run([sys.executable, "build_corpus.py", "--out", "/tmp",
                            "--loops", "300"], check=True)
        print()
        subprocess.run([sys.executable, a.out], check=False)


if __name__ == "__main__":
    main()
