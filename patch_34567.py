#!/usr/bin/env python3
"""
patch_34567.py -- PHASE 3: COMPRESSION + LEAKAGE + SIGN + FISHER-WEIGHTED COMPLEMENT
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

  (7) SIGN COMPLEMENT                        <-- new in this patch
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

    python3 patch_34567.py --no-trigger --k 8 --win 24 --geoval 0.6 --run
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

    python3 patch_34567.py --no-trigger --k 8 --win 24 --geoval 0.6 --run      # verified
    python3 patch_34567.py --no-trigger --k 8 --win 24 --geoval 0.6 --fisher --run
    python3 patch_34567.py --k 8 --win 24 --geoval 0.6 --theta 0.96 --verbose --run
    python3 patch_34567.py --no-trigger --k 8 --win 24 --geoval 0.6 --fisher --beta 0.5 --run

Needs compiler_geometri_patched_86.py and build_corpus.py alongside.
"""

import argparse
import os
import subprocess
import sys

ANCHOR = """opt_b = torch.optim.AdamW(model.parameters(), lr=LR*5,
                           betas=(0.9,0.95), weight_decay=0.1)"""

GEO_OLD = """        geo_ok = (pc >= 4 and 5.0 <= tau <= 7.5 and rm2 >= 0.65)"""
GEO_NEW = """        geo_ok = (pc >= 4 and 5.0 <= tau <= 7.5 and rm2 >= 0.65
                  and v <= _SUB_GEOVAL)"""

OLD_STEP = """    model.train(); x, y = get_batch(); _, l = model(x, y)
    opt_b.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt_b.step()

    if step % 8 == 0:"""

NEW_STEP = """    if _sub_should_skip(step):
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
# PATCH 3+4+5+6+7 -- PHASE 3 ONLY
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
        print(f"    [34567] step {step:3d}  capture {_np.mean(_sub_stats['cap'][-16:]):.3f}"
              f"  gamma {gm:.3f}  bwd {_sub_stats['bwd']}  coast {_sub_stats['coast']}{fs}")

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
    if _sub_stats["fshare"]:
        extra += f"  fisher-share {_np.mean(_sub_stats['fshare']):.3f}"
    print(f"  [34567] dims {_sub_stats['dims']} (asked __DIMS__)/{_sub_P}  coast:{mode}  "
          f"backward {b}  coasted {c}  skip {c/max(b+c,1):.0%}  "
          f"capture {cap:.3f}  gamma {gm:.3f}{extra}")
# ─────────────────────── end patch 3+4+5+6+7 ───────────────────────
'''

REPORT_ANCHOR = """step_basin = step"""
REPORT_NEW = """_sub_report()
step_basin = step"""

ALPHA_BY_ROLE = {"EMB": 0.05, "LN": 0.05, "W_Q": 0.10, "W_K": 0.10,
                 "FF": 0.15, "W_O": 0.15, "W_V": 0.15}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="compiler_geometri_patched_86.py")
    ap.add_argument("--out", default="compiler_phase34567.py")
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
                  .replace("__GEOVAL__", "float('inf')" if a.geoval == inf else repr(a.geoval))
                  .replace("__DIMS__", str(dims)))

    out = (src.replace(ANCHOR, ANCHOR + "\n" + patch, 1)
              .replace(OLD_STEP, NEW_STEP, 1)
              .replace(REPORT_ANCHOR, REPORT_NEW, 1))
    if a.geoval != inf:
        if GEO_OLD not in out:
            sys.exit("geo-stop anchor not found")
        out = out.replace(GEO_OLD, GEO_NEW, 1)
    open(a.out, "w").write(out)

    print(f"wrote {a.out}")
    print(f"  (3) compression : rank {a.k}/role x 7 = {dims} dims, window {a.win}, "
          f"rebuild every {a.rebuild}")
    print(f"      alpha {alpha}")
    print(f"  (4) leakage     : {'OFF' if a.no_trigger else f'theta {a.theta}, max coast {a.maxcoast}'}")
    print(f"  (5) sign coast  : {'OFF (coast on frame)' if a.no_sign else 'ON'}")
    print(f"  (6) fisher      : {f'ON  beta {a.beta}, rank {a.frank}, refresh {a.fevery}' if a.fisher else 'OFF'}")
    print(f"  (7) sign comp   : {'ON -- complement as scalar x sign' if a.signcomp else 'OFF'}")
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
