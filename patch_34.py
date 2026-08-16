#!/usr/bin/env python3
"""
patch_34.py -- PHASE 3 SUBSPACE COMPRESSION (3) + LEAKAGE TRIGGER (4)
=====================================================================

Two mechanisms, deliberately separable, applied to Phase 3 only. Phases 1, 2, 4
and 5 are byte-identical: the corpus-specific spectral init, the MF pump,
TopoGate and the K0 split descent are untouched.

  (3) COMPRESSION -- 14 dimensions
      The realised update is confined to per-role frames:
          u' = P_Q u + alpha_role * (I - P_Q) u
      rank 2 per role x 7 roles = 14 dimensions of P = 1.18e6.

  (4) LEAKAGE TRIGGER -- skipped backward passes
      Compute gamma = 1 - ||P_Q g||^2 / ||g||^2 on the last gradient, energy
      weighted across roles. While gamma stays below theta, step along the
      existing frame using the previous coefficients WITHOUT a backward pass.
      When gamma crosses theta, take a real full-rank AdamW step and rebuild.

      (3) alone costs compute -- the projection is extra work on top of a full
      gradient. Only (4) skips backward passes, and that is where the saving is.

WHY THESE SETTINGS

Capture of a fresh update by a frame of age a, measured on this pipeline:
    age 1 -> 0.873    age 4 -> 0.545    age 7 -> 0.335
so the rebuild interval matters far more than the rank.

Effective dimension per role is 1.5-2.2 (EMB 1.52, LN 1.76, W_Q 2.00, W_K 2.00,
W_V 2.19, W_O 2.14, FF 2.05) with three directions holding 96-98% of the energy,
so rank 2 per role is at the measured dimension and rank 4 has headroom.

Per-role orthogonal share at age 1, which sets alpha:
    EMB 0.077  LN 0.070  W_Q 0.112  W_K 0.117  FF 0.146  W_O 0.154  W_V 0.161

Frame quality by step, one run: capture 0.89 (1-25), 0.93 (25-50), 0.90 (50-75),
0.90 (75-100), 0.80 (100-125). The projectable window is roughly steps 25-100.

WHAT IS AND IS NOT MEASURED

Measured: compression alone (patch_phase3.py) gives final val 0.0467 against a
baseline 0.0460, on 203 CE against 187 -- no gain, and it demonstrates only that
Phase 3 tolerates 14 dimensions. One seed per arm.

Measured elsewhere: leakage-triggered placement of a fixed full-rank budget beat
uniform placement 0.056 vs 0.292 on a small transformer -- 5.2x on less budget.
That result is what mechanism (4) is trying to carry into the pipeline, and it
has NOT been reproduced here. Treat --trigger as untested until you run it.

USAGE
    python3 patch_34.py --run                      # both mechanisms
    python3 patch_34.py --no-trigger --run         # (3) only: compression
    python3 patch_34.py --theta 0.30 --run         # looser trigger, more skips
    python3 patch_34.py --k 4 --alpha 0.0 --run    # ablations
    python3 patch_34.py --verbose --run            # capture, skips, gamma

Reports at the end of Phase 3: backward passes taken, steps skipped, mean
capture, and the skip fraction. Compare final val AND Phi_cl at handoff against
the unpatched compiler -- Phase 3's own loss is an intermediate.

Needs compiler_geometri_patched_86.py and build_corpus.py alongside.
"""

import argparse
import os
import subprocess
import sys

ANCHOR = """opt_b = torch.optim.AdamW(model.parameters(), lr=LR*5,
                           betas=(0.9,0.95), weight_decay=0.1)"""

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
# PATCH 3+4 -- PHASE 3 ONLY
#   (3) compression : u' = P_Q u + alpha_role * (I - P_Q) u,  __DIMS__ dims
#   (4) leakage     : coast on the frame while gamma < theta, no backward pass
# ─────────────────────────────────────────────────────────────
_SUB_K, _SUB_WIN, _SUB_REBUILD = __K__, __WIN__, __REBUILD__
_SUB_ALPHA   = __ALPHA__
_SUB_TRIGGER = __TRIGGER__
_SUB_THETA   = __THETA__
_SUB_MAXCOAST = __MAXCOAST__
_SUB_VERBOSE = __VERBOSE__

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
_sub_idx, _o = {}, 0
for _nm, _p in model.named_parameters():
    if _p.requires_grad:
        _sub_idx.setdefault(_sub_role(_nm), []).append(torch.arange(_o, _o + _p.numel()))
    _o += _p.numel()
_sub_idx = {k: torch.cat(v) for k, v in _sub_idx.items()}
_sub_hist  = {k: [] for k in _sub_idx}
_sub_frame = {}
_sub_coef  = None          # last in-frame coefficients, for coasting
_sub_stats = {"bwd": 0, "coast": 0, "cap": [], "gamma": [], "run": 0}

def _sub_flat():
    return torch.cat([p.data.reshape(-1) for p in _sub_ps]).clone()

def _sub_note_grad():
    """energy-weighted leakage of the CURRENT gradient against the frame in force.
    Energy weighted, not max: LN is 1792 params against FF's 591360, and a max
    would let the smallest block drive the schedule."""
    if not _sub_frame:
        return
    num = den = 0.0
    for k, ii in _sub_idx.items():
        g = torch.cat([(p.grad.reshape(-1) if p.grad is not None
                        else torch.zeros(p.numel())) for p in _sub_ps])[ii]
        e = float((g * g).sum())
        if e <= 0.0:
            continue
        Q = _sub_frame[k]; pj = Q @ (Q.T @ g)
        num += e * (1.0 - float((pj * pj).sum()) / e); den += e
    if den > 0.0:
        _sub_stats["gamma"].append(num / den)

def _sub_should_skip(step):
    if not _SUB_TRIGGER or _sub_coef is None or not _sub_frame:
        return False
    if not _sub_stats["gamma"]:
        return False
    if _sub_stats["run"] >= _SUB_MAXCOAST:
        return False
    return _sub_stats["gamma"][-1] < _SUB_THETA

def _sub_coast(step):
    """step along the existing frame with the last coefficients -- no backward pass"""
    global _sub_coef
    _sub_stats["coast"] += 1; _sub_stats["run"] += 1
    with torch.no_grad():
        i = 0
        nu = torch.zeros(sum(p.numel() for p in _sub_ps))
        for k, ii in _sub_idx.items():
            nu[ii] = _sub_frame[k] @ _sub_coef[k]
        for p in _sub_ps:
            n = p.numel(); p.data.add_(nu[i:i + n].view_as(p)); i += n

def _sub_project(th_before, step):
    global _sub_coef
    _sub_stats["bwd"] += 1; _sub_stats["run"] = 0
    u = _sub_flat() - th_before
    for k, ii in _sub_idx.items():
        _sub_hist[k].append(u[ii].clone())
        if len(_sub_hist[k]) > _SUB_WIN:
            _sub_hist[k].pop(0)
    if len(_sub_hist["FF"]) < _SUB_WIN:
        return                                     # warm-up: keep the full update
    if step % _SUB_REBUILD == 0 or not _sub_frame:
        for k, ii in _sub_idx.items():
            A = torch.stack(_sub_hist[k], 1)
            _sub_frame[k] = torch.linalg.svd(A, full_matrices=False)[0][:, :_SUB_K]
    nu = torch.zeros_like(u); coef = {}
    cap = tot = 0.0
    for k, ii in _sub_idx.items():
        Q, ub = _sub_frame[k], u[ii]
        c = Q.T @ ub
        par = Q @ c
        coef[k] = c
        nu[ii] = par + _SUB_ALPHA.get(k, 0.1) * (ub - par)
        cap += float((par * par).sum()); tot += float((ub * ub).sum())
    _sub_coef = coef
    _sub_stats["cap"].append(cap / max(tot, 1e-30))
    with torch.no_grad():
        i = 0
        for p in _sub_ps:
            n = p.numel(); p.data.copy_((th_before + nu)[i:i + n].view_as(p)); i += n
    if _SUB_VERBOSE and step % 16 == 0:
        import numpy as _np
        gm = _np.mean(_sub_stats["gamma"][-16:]) if _sub_stats["gamma"] else float("nan")
        print(f"    [3+4] step {step:3d}  capture {_np.mean(_sub_stats['cap'][-16:]):.3f}"
              f"  gamma {gm:.3f}  bwd {_sub_stats['bwd']}  coast {_sub_stats['coast']}")

def _sub_report():
    import numpy as _np
    b, c = _sub_stats["bwd"], _sub_stats["coast"]
    cap = _np.mean(_sub_stats["cap"]) if _sub_stats["cap"] else float("nan")
    gm  = _np.mean(_sub_stats["gamma"]) if _sub_stats["gamma"] else float("nan")
    print(f"  [3+4] dims {__DIMS__}/{sum(p.numel() for p in _sub_ps)}  "
          f"backward {b}  coasted {c}  skip {c/max(b+c,1):.0%}  "
          f"capture {cap:.3f}  gamma {gm:.3f}")
# ───────────────────────── end patch 3+4 ─────────────────────────
'''

REPORT_ANCHOR = """step_basin = step"""
REPORT_NEW = """_sub_report()
step_basin = step"""

ALPHA_BY_ROLE = {"EMB": 0.05, "LN": 0.05, "W_Q": 0.10, "W_K": 0.10,
                 "FF": 0.15, "W_O": 0.15, "W_V": 0.15}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="compiler_geometri_patched_86.py")
    ap.add_argument("--out", default="compiler_phase34.py")
    ap.add_argument("--k", type=int, default=2, help="rank per role (7 roles)")
    ap.add_argument("--win", type=int, default=8)
    ap.add_argument("--rebuild", type=int, default=2)
    ap.add_argument("--alpha", type=float, default=None,
                    help="single alpha for all roles; omit for per-role defaults")
    ap.add_argument("--theta", type=float, default=0.20,
                    help="leakage threshold; coast while gamma is below it")
    ap.add_argument("--maxcoast", type=int, default=3,
                    help="hard cap on consecutive skipped steps")
    ap.add_argument("--no-trigger", action="store_true",
                    help="compression only, no backward-pass skipping")
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
    dims = a.k * 7
    patch = (PATCH.replace("__K__", str(a.k))
                  .replace("__WIN__", str(a.win))
                  .replace("__REBUILD__", str(a.rebuild))
                  .replace("__ALPHA__", repr(alpha))
                  .replace("__TRIGGER__", str(not a.no_trigger))
                  .replace("__THETA__", str(a.theta))
                  .replace("__MAXCOAST__", str(a.maxcoast))
                  .replace("__VERBOSE__", str(bool(a.verbose)))
                  .replace("__DIMS__", str(dims)))

    out = (src.replace(ANCHOR, ANCHOR + "\n" + patch, 1)
              .replace(OLD_STEP, NEW_STEP, 1)
              .replace(REPORT_ANCHOR, REPORT_NEW, 1))
    open(a.out, "w").write(out)
    print(f"wrote {a.out}")
    print(f"  (3) compression: rank {a.k}/role x 7 = {dims} dims, "
          f"window {a.win}, rebuild every {a.rebuild}")
    print(f"      alpha {alpha}")
    print(f"  (4) leakage trigger: {'OFF' if a.no_trigger else f'theta {a.theta}, max coast {a.maxcoast}'}")
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
