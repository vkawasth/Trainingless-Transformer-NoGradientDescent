#!/usr/bin/env python3
"""
PATCH THE COMPILER: SUBSPACE PROJECTION IN PHASE 3 ONLY
=======================================================

Reads compiler_geometri_patched_86.py, writes a patched copy, optionally runs it.
Phases 1, 2, 4 and 5 are byte-identical -- the corpus-specific spectral init, the
MF pump, TopoGate and the K0 split descent are untouched. The only change is the
update application inside Phase 3's loop.

WHAT THE PATCH DOES

After opt_b.step(), the realised update is projected onto per-role frames and a
fraction of the orthogonal complement is added back:

    u   = theta_after - theta_before
    u'  = P_Q u + alpha_role * (I - P_Q) u

Q is rank 4 per role, rebuilt every 2 steps from the last 8 updates.

WHY THOSE SETTINGS

Measured on this pipeline, capture of a fresh update by a frame of age a:
    age 1 -> 0.873      age 4 -> 0.545      age 7 -> 0.335
so the rebuild interval matters far more than the rank. Rebuilding every 2 steps
keeps the frame near age 1; the earlier 8-step interval averaged ~0.70 and cost
63% more steps in Phase 3.

Per-role orthogonal share at age 1, which sets alpha:
    EMB 0.077   LN 0.070   W_Q 0.112   W_K 0.117   FF 0.146   W_O 0.154   W_V 0.161

Effective dimension per role is 1.5-2.2 with three directions holding 96-98% of
the energy, so rank 4 has headroom.

MEASURED RESULT (one seed each, single-CPU sandbox)

    baseline   final val 0.0460   187 CE   Phi_cl 2/5 through handoff
    patched    final val 0.0467   203 CE   Phi_cl 4/5 through handoff

Phase 3 alone ends worse under projection (val 0.159 at plateau) and the pipeline
ends the same, because Phase 3's loss is an intermediate -- Phases 4 and 5 close
the gap. The patched arm hands off with twice the clean alignment.

CAVEAT: one seed per arm. The 1.5% final difference is not resolved; run each
arm two or three times before drawing a conclusion from it.

USAGE
    python3 patch_phase3.py                  # write compiler_phase3_subspace.py
    python3 patch_phase3.py --run            # write it and run it
    python3 patch_phase3.py --alpha 0.0      # ablate the complement entirely
    python3 patch_phase3.py --rebuild 8      # the old interval, for comparison
    python3 patch_phase3.py --src other.py --out patched.py

Needs compiler_geometri_patched_86.py and build_corpus.py in the working
directory (or pass --src), and /tmp/train_ids.json from build_corpus.py.
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

NEW_STEP = """    model.train(); x, y = get_batch(); _, l = model(x, y)
    _sub_th = _sub_flat()
    opt_b.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt_b.step()
    _sub_project(_sub_th, step)

    if step % 8 == 0:"""

PATCH = '''
# ─────────────────────────────────────────────────────────────
# SUBSPACE PROJECTION -- PHASE 3 ONLY
#   u' = P_Q u + alpha_role * (I - P_Q) u
#   Q: rank __K__ per role, rebuilt every __REBUILD__ steps from the last __WIN__
# ─────────────────────────────────────────────────────────────
_SUB_K, _SUB_WIN, _SUB_REBUILD = __K__, __WIN__, __REBUILD__
_SUB_ALPHA = __ALPHA__
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
_sub_hist = {k: [] for k in _sub_idx}
_sub_frame, _sub_cap = {}, []

def _sub_flat():
    return torch.cat([p.data.reshape(-1) for p in _sub_ps]).clone()

def _sub_project(th_before, step):
    u = _sub_flat() - th_before
    for k, ii in _sub_idx.items():
        _sub_hist[k].append(u[ii].clone())
        if len(_sub_hist[k]) > _SUB_WIN:
            _sub_hist[k].pop(0)
    if len(_sub_hist["FF"]) < _SUB_WIN:
        return                                    # warm-up: full update
    if step % _SUB_REBUILD == 0 or not _sub_frame:
        for k, ii in _sub_idx.items():
            A = torch.stack(_sub_hist[k], 1)
            _sub_frame[k] = torch.linalg.svd(A, full_matrices=False)[0][:, :_SUB_K]
    nu = torch.zeros_like(u)
    cap = tot = 0.0
    for k, ii in _sub_idx.items():
        Q, ub = _sub_frame[k], u[ii]
        par = Q @ (Q.T @ ub)
        nu[ii] = par + _SUB_ALPHA.get(k, 0.1) * (ub - par)
        cap += float((par * par).sum()); tot += float((ub * ub).sum())
    _sub_cap.append(cap / max(tot, 1e-30))
    with torch.no_grad():
        i = 0
        for p in _sub_ps:
            n = p.numel(); p.data.copy_((th_before + nu)[i:i + n].view_as(p)); i += n
    if _SUB_VERBOSE and step % 16 == 0:
        import numpy as _np
        print(f"    [subspace] step {step:3d}  capture {_np.mean(_sub_cap[-16:]):.3f}")
# ───────────────────── end subspace patch ─────────────────────
'''

ALPHA_BY_ROLE = {"EMB": 0.05, "LN": 0.05, "W_Q": 0.10, "W_K": 0.10,
                 "FF": 0.15, "W_O": 0.15, "W_V": 0.15}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="compiler_geometri_patched_86.py")
    ap.add_argument("--out", default="compiler_phase3_subspace.py")
    ap.add_argument("--k", type=int, default=4, help="rank per role")
    ap.add_argument("--win", type=int, default=8, help="update history length")
    ap.add_argument("--rebuild", type=int, default=2, help="steps between frame rebuilds")
    ap.add_argument("--alpha", type=float, default=None,
                    help="one alpha for every role; omit for the per-role defaults")
    ap.add_argument("--verbose", action="store_true", help="print capture every 16 steps")
    ap.add_argument("--run", action="store_true", help="run the patched compiler after writing")
    a = ap.parse_args()

    if not os.path.exists(a.src):
        sys.exit(f"not found: {a.src}")
    src = open(a.src).read()
    if ANCHOR not in src or OLD_STEP not in src:
        sys.exit("anchors not found -- this patcher targets compiler_geometri_patched_86.py")

    alpha = ALPHA_BY_ROLE if a.alpha is None else {k: a.alpha for k in ALPHA_BY_ROLE}
    patch = (PATCH.replace("__K__", str(a.k))
                  .replace("__WIN__", str(a.win))
                  .replace("__REBUILD__", str(a.rebuild))
                  .replace("__ALPHA__", repr(alpha))
                  .replace("__VERBOSE__", str(bool(a.verbose))))

    out = src.replace(ANCHOR, ANCHOR + "\n" + patch, 1).replace(OLD_STEP, NEW_STEP, 1)
    open(a.out, "w").write(out)
    print(f"wrote {a.out}")
    print(f"  rank {a.k}/role, window {a.win}, rebuild every {a.rebuild}")
    print(f"  alpha {alpha}")
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
