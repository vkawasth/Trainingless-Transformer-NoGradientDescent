#!/usr/bin/env python3
"""
patch_345.py -- PHASE 3: COMPRESSION (3) + LEAKAGE TRIGGER (4) + SIGN TRANSPORT (5)
===================================================================================

Three mechanisms, each switchable, applied to Phase 3 only. Phases 1, 2, 4 and 5
are byte-identical -- the corpus-specific spectral init, the MF pump, TopoGate
and the K0 split descent are untouched.

  (3) COMPRESSION -- 14 dimensions
        u' = P_Q u + alpha_role * (I - P_Q) u
      rank 4 per role x 7 roles = 28 dims, frames rebuilt every 2 steps.
  k=4 VERIFIED WORKING at P=4.3e6. k=2 does not reach the basin (final 0.379).
  NOTE: an n-column update history yields at most n singular vectors, so the
  window must be >= the rank or the frame is silently capped. The patcher
  now raises the window automatically and reports the ACTUAL frame width.

  (4) LEAKAGE TRIGGER -- skipped backward passes
      gamma = 1 - ||P_Q g||^2 / ||g||^2, energy weighted across roles. While
      gamma < theta the step COASTS: no backward pass, no forward pass.

  (5) SIGN TRANSPORT -- scalar-times-sign coasting
      The coast step is a_t * sign(g) per role rather than Q @ coef. This draws
      on the single strongest measurement in the programme: dtheta ~ a_t sign(g)
      retains 100.1% of the loss reduction with ONE SCALAR per step. Coasting
      then costs a multiply instead of a P x K matmul.

      Honest limit: the sign-transport RESIDUAL is 78% of the update by norm,
      with lag-1 correlation 0.94 and ~90% of it transverse to the gradient. The
      scalar-times-sign step recovers the DESCENT while missing most of the
      vector. That is acceptable for Phase 3, whose job is the journey to the
      basin, and it is why (5) is used only for coasting -- never for the real
      steps, which stay full AdamW.

  (R) RESIDUAL ACCUMULATION -- optional, --accum
      The discarded (1-alpha)(I - P_Q) u is banked rather than thrown away, and
      folded back when its norm exceeds a fraction of the mean update norm. This
      preserves descent that compression drops, without extra backward passes.

WHAT IS MEASURED AND WHAT IS NOT

Measured on this pipeline:
    frame capture by age:   age 1 -> 0.873   age 4 -> 0.545   age 7 -> 0.335
    effective dimension:    EMB 1.52  LN 1.76  W_Q/W_K 2.00  W_V 2.19  W_O 2.14  FF 2.05
    orthogonal share @1:    EMB .077  LN .070  W_Q .112  W_K .117  FF .146  W_O .154  W_V .161
    capture by step:        0.89 (1-25) 0.93 (25-50) 0.90 (50-100) 0.80 (100-125)
    compression alone:      final val 0.0467 vs baseline 0.0460, 203 CE vs 187 -- no gain

Measured elsewhere, NOT reproduced in this pipeline:
    leakage-triggered placement beat uniform placement 0.056 vs 0.292 (5.2x, less budget)
    sign transport retained 100.1% of loss reduction with one scalar

Mechanisms (4) and (5) are therefore UNTESTED HERE. Run --no-trigger and
--no-sign arms to separate their contributions before believing any of it.

USAGE
    python3 patch_345.py --no-trigger --run        # k=4 compression: VERIFIED WORKING
    python3 patch_345.py --run                     # all three
    python3 patch_345.py --no-sign --run           # (3)+(4): coast on the frame
    python3 patch_345.py --no-trigger --run        # (3) only: compression
    python3 patch_345.py --accum --run             # add residual banking
    python3 patch_345.py --theta 0.30 --run        # looser trigger, more skips
    python3 patch_345.py --verbose --run

Phase 3 ends with a line reporting backward passes, coasted steps, skip
fraction, mean capture and mean gamma. Compare final val AND Phi_cl at handoff
against the unpatched compiler -- Phase 3's own loss is an intermediate.

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
# PATCH 3+4+5 -- PHASE 3 ONLY
#   (3) compression : u' = P_Q u + alpha_role (I - P_Q) u,  __DIMS__ dims
#   (4) leakage     : coast while gamma < theta, no backward pass
#   (5) sign        : coast step is a_t * sign(g) per role
# ─────────────────────────────────────────────────────────────
_SUB_K, _SUB_WIN, _SUB_REBUILD = __K__, __WIN__, __REBUILD__
_SUB_ALPHA    = __ALPHA__
_SUB_TRIGGER  = __TRIGGER__
_SUB_SIGN     = __SIGN__
_SUB_ACCUM    = __ACCUM__
_SUB_RESCALE  = __RESCALE__
_SUB_THETA    = __THETA__
_SUB_MAXCOAST = __MAXCOAST__
_SUB_VERBOSE  = __VERBOSE__
_SUB_GEOVAL   = __GEOVAL__   # geo-stop also requires val <= this

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
_sub_idx  = {k: torch.cat(v) for k, v in _sub_idx.items()}
_sub_hist = {k: [] for k in _sub_idx}
_sub_frame = {}
_sub_coef  = None                       # in-frame coefficients, for frame coasting
_sub_sign  = None                       # sign field + per-role scalar, for (5)
_sub_scale = {}
_sub_bank  = torch.zeros(_sub_P)        # residual accumulator, for (R)
_sub_stats = {"bwd": 0, "coast": 0, "fold": 0, "cap": [], "gamma": [], "run": 0,
              "unorm": [], "dims": 0, "rs": []}

def _sub_flat():
    return torch.cat([p.data.reshape(-1) for p in _sub_ps]).clone()

def _sub_gflat():
    return torch.cat([(p.grad.reshape(-1) if p.grad is not None
                       else torch.zeros(p.numel())) for p in _sub_ps])

def _sub_note_grad():
    """energy-weighted leakage of the current gradient against the frame in force.
    Energy weighted rather than max: LN has 1792 params against FF's 591360, and
    a max would let the smallest block dictate the schedule."""
    if not _sub_frame:
        return
    g = _sub_gflat()
    num = den = 0.0
    for k, ii in _sub_idx.items():
        gb = g[ii]; e = float((gb * gb).sum())
        if e <= 0.0:
            continue
        Q = _sub_frame[k]; pj = Q @ (Q.T @ gb)
        num += e * (1.0 - float((pj * pj).sum()) / e); den += e
    if den > 0.0:
        _sub_stats["gamma"].append(num / den)

def _sub_should_skip(step):
    if not _SUB_TRIGGER or not _sub_frame:
        return False
    if _sub_coef is None and _sub_sign is None:
        return False
    if _sub_stats["run"] >= _SUB_MAXCOAST or not _sub_stats["gamma"]:
        return False
    return _sub_stats["gamma"][-1] < _SUB_THETA

def _sub_coast(step):
    """advance without a forward or backward pass"""
    _sub_stats["coast"] += 1; _sub_stats["run"] += 1
    nu = torch.zeros(_sub_P)
    if _SUB_SIGN and _sub_sign is not None:
        # (5) scalar x sign, per role
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

    # (5) sign field and per-role scalar a_t = <u, sign(u)> / <sign, sign>
    _sub_sign = torch.sign(u)
    for k, ii in _sub_idx.items():
        s = _sub_sign[ii]
        _sub_scale[k] = float((u[ii] * s).sum()) / max(float((s * s).sum()), 1e-30)

    nu = torch.zeros_like(u); coef = {}
    cap = tot = 0.0
    for k, ii in _sub_idx.items():
        Q, ub = _sub_frame[k], u[ii]
        c = Q.T @ ub
        par = Q @ c
        coef[k] = c
        drop = ub - par
        nu[ii] = par + _SUB_ALPHA.get(k, 0.1) * drop
        if _SUB_ACCUM:
            _sub_bank[ii] += (1.0 - _SUB_ALPHA.get(k, 0.1)) * drop
        cap += float((par * par).sum()); tot += float((ub * ub).sum())
    _sub_coef = coef
    _sub_stats["cap"].append(cap / max(tot, 1e-30))

    # (N) optional norm restoration: same direction, Adam's magnitude.
    # Without this the step is sqrt(capture) of Adam's norm -- 0.97 at capture
    # 0.94, 0.92 at capture 0.84 -- and that shortfall compounds over the run.
    # With it, any remaining gap is DIRECTION rather than magnitude.
    if _SUB_RESCALE:
        _n_u, _n_nu = float(u.norm()), float(nu.norm())
        if _n_nu > 1e-30:
            nu = nu * (_n_u / _n_nu)
            _sub_stats["rs"].append(_n_u / _n_nu)

    # (R) fold the bank back once it is worth a step
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
        print(f"    [345] step {step:3d}  capture {_np.mean(_sub_stats['cap'][-16:]):.3f}"
              f"  gamma {gm:.3f}  bwd {_sub_stats['bwd']}  coast {_sub_stats['coast']}"
              + (f"  fold {_sub_stats['fold']}" if _SUB_ACCUM else ""))

def _sub_report():
    import numpy as _np
    b, c = _sub_stats["bwd"], _sub_stats["coast"]
    cap = _np.mean(_sub_stats["cap"]) if _sub_stats["cap"] else float("nan")
    gm  = _np.mean(_sub_stats["gamma"]) if _sub_stats["gamma"] else float("nan")
    mode = ("sign" if (_SUB_TRIGGER and _SUB_SIGN) else
            "frame" if _SUB_TRIGGER else "off")
    print(f"  [345] dims {_sub_stats['dims']} (asked __DIMS__)/{_sub_P}  coast:{mode}  backward {b}  coasted {c}  "
          f"skip {c/max(b+c,1):.0%}  capture {cap:.3f}  gamma {gm:.3f}"
          + (f"  folds {_sub_stats['fold']}" if _SUB_ACCUM else "")
          + (f"  rescale x{_np.mean(_sub_stats['rs']):.3f}"
             if (_SUB_RESCALE and _sub_stats['rs']) else ""))
# ────────────────────────── end patch 3+4+5 ──────────────────────────
'''

REPORT_ANCHOR = """step_basin = step"""
REPORT_NEW = """_sub_report()
step_basin = step"""

ALPHA_BY_ROLE = {"EMB": 0.05, "LN": 0.05, "W_Q": 0.10, "W_K": 0.10,
                 "FF": 0.15, "W_O": 0.15, "W_V": 0.15}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="compiler_geometri_patched_86.py")
    ap.add_argument("--out", default="compiler_phase345.py")
    ap.add_argument("--k", type=int, default=4,
                    help="rank per role (7 roles). k=4 is the verified working "
                         "setting; k=2 under-ranks and the pipeline does not "
                         "reach the basin.")
    ap.add_argument("--win", type=int, default=8)
    ap.add_argument("--rebuild", type=int, default=2)
    ap.add_argument("--alpha", type=float, default=None,
                    help="single alpha for all roles; omit for per-role defaults")
    ap.add_argument("--theta", type=float, default=0.20,
                    help="leakage threshold; coast while gamma is below it")
    ap.add_argument("--maxcoast", type=int, default=3)
    ap.add_argument("--no-trigger", action="store_true", help="disable (4)")
    ap.add_argument("--no-sign", action="store_true",
                    help="disable (5); coast on the frame instead")
    ap.add_argument("--accum", action="store_true", help="enable residual banking")
    ap.add_argument("--rescale", action="store_true",
                    help="restore the projected step to Adam's norm; separates a "
                         "direction deficit from a magnitude one")
    ap.add_argument("--geoval", type=float, default=float("inf"),
                    help="geo-stop additionally requires val <= this. A better "
                         "projection can satisfy the orbit criteria while the loss "
                         "is still high -- k=8/win24 geo-stopped at step 32 at val "
                         "2.99 and the pipeline never recovered. Set e.g. 0.6 to "
                         "require the loss to have come down first, or 0 to "
                         "disable geo-stop entirely (Phase 3 then runs to plateau).")
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

    alpha = ALPHA_BY_ROLE if a.alpha is None else {k: a.alpha for k in ALPHA_BY_ROLE}
    if a.win < a.k:
        print(f"  note: window {a.win} < rank {a.k}; raising window to {2*a.k} "
              f"(an n-column history yields at most n singular vectors)")
        a.win = 2 * a.k
    dims = a.k * 7
    patch = (PATCH.replace("__K__", str(a.k))
                  .replace("__WIN__", str(a.win))
                  .replace("__REBUILD__", str(a.rebuild))
                  .replace("__ALPHA__", repr(alpha))
                  .replace("__TRIGGER__", str(not a.no_trigger))
                  .replace("__SIGN__", str(not a.no_sign))
                  .replace("__ACCUM__", str(bool(a.accum)))
                  .replace("__RESCALE__", str(bool(a.rescale)))
                  .replace("__THETA__", str(a.theta))
                  .replace("__MAXCOAST__", str(a.maxcoast))
                  .replace("__VERBOSE__", str(bool(a.verbose)))
                  .replace("__GEOVAL__", "float('inf')" if a.geoval == float("inf") else repr(a.geoval))
                  .replace("__DIMS__", str(dims)))

    out = (src.replace(ANCHOR, ANCHOR + "\n" + patch, 1)
              .replace(OLD_STEP, NEW_STEP, 1)
              .replace(REPORT_ANCHOR, REPORT_NEW, 1))
    if a.geoval != float("inf"):
        if GEO_OLD not in out:
            sys.exit("geo-stop anchor not found")
        out = out.replace(GEO_OLD, GEO_NEW, 1)
    open(a.out, "w").write(out)
    print(f"wrote {a.out}")
    print(f"  (3) compression : rank {a.k}/role x 7 = {dims} dims, "
          f"window {a.win}, rebuild every {a.rebuild}, alpha {alpha}")
    print(f"  (4) leakage     : {'OFF' if a.no_trigger else f'theta {a.theta}, max coast {a.maxcoast}'}")
    print(f"  (5) sign coast  : {'OFF (coast on frame)' if a.no_sign else 'ON'}")
    print(f"  (R) accumulate  : {'ON' if a.accum else 'OFF'}")
    print(f"  (N) rescale     : {'ON -- step restored to Adam norm' if a.rescale else 'OFF (honest)'}")
    print(f"  geo-stop val    : {'unchanged' if a.geoval == float('inf') else f'also requires val <= {a.geoval}'}")
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
