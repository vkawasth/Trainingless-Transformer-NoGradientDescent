#!/usr/bin/env python3
"""
patch_curvature.py -- PHASE 3 LEVERS FROM THE CURVATURE MEASUREMENTS
====================================================================

Patches Phase 3 only. Phases 1, 2, 4, 5 byte-identical.

FOUR LEVERS, each switchable, each with the measurement behind it.

(C) RESIDUAL COMPRESSION -- --compress 1.0
    The residual (I - P_Q)u is kept only on its top-k coordinates by magnitude.
    Measured under SGD: the top 1% (~12,000 coords of 1.18M) recovers 95-98% of
    the residual's own loss change; the top 0.1% recovers only 66-72%, so 1% is
    the honest setting, not 0.1%.
    Ranking is by |r_i| and NOT by curvature: |r_i (Hr)_i| was tested and LOST
    (0.827 vs 1.186 at matched k), because this residual is orthogonal to the
    FRAME, not to the gradient, so its loss change is first-order dominated.

(S) SPARSE SECOND MOMENT -- --sparse-v 0.02
    Adam's v is kept only for the top fraction of coordinates by |g|; the rest
    get plain momentum. Measured: diag(H) has PR = 2% of P and the top 1% of
    coordinates hold 38-49% of the diagonal mass, so second-order scaling is
    only doing work on a small set.
    CAVEAT, measured: the permutation control showed those diagonal spikes sit
    where the OFF-diagonal coupling is strongest -- permuting D cut the
    commutator to 0.35x. So the 2% is not decoupled from the 98%, and this lever
    is more speculative than the concentration number alone suggests.

(F) FLIP DAMPING -- --flipdamp
    Per-coordinate step scaling alpha_i = alpha_0 (1 - phi_i)^2 with phi_i a
    running flip rate. Flip rate is the diagnostic that predicted both sign
    failures before the loss did: pure sign went 0.381 -> 0.539 while AdamW sat
    at 0.116 -> 0.120, and the hybrid rose monotonically 0.094 -> 0.484 with the
    loss following.

(W) BLOCK WEIGHT DECAY -- --wd-split
    Decay on the self-curvature blocks, relaxed on the interaction. Measured:
    ATTN<->FF is the only block that GAINS absolute Hessian mass (+31% to +37%)
    while the operator shrinks 49%; EMB x EMB drains 74%.

NOT IMPLEMENTED, and why: freezing EMB after the curvature handover was
proposed and is contradicted by the update-energy measurement -- EMB holds
11.3, 13.5, 15.4, 12.8, 13.6% of update energy across the run, flat to rising,
even as its self-curvature drains 74%. Low curvature there does not mean low
contribution.

USAGE
    python3 patch_curvature.py --compress 1.0 --run
    python3 patch_curvature.py --compress 1.0 --flipdamp --run
    python3 patch_curvature.py --sparse-v 0.02 --run
    python3 patch_curvature.py --all --run
Compare final val and Phi_cl at handoff against the unpatched compiler.
"""
import argparse, os, subprocess, sys

ANCHOR = """opt_b = torch.optim.AdamW(model.parameters(), lr=LR*5,
                           betas=(0.9,0.95), weight_decay=0.1)"""
OLD = """    model.train(); x, y = get_batch(); _, l = model(x, y)
    opt_b.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt_b.step()

    if step % 8 == 0:"""
NEW = """    model.train(); x, y = get_batch(); _, l = model(x, y)
    _cv_th = _cv_flat()
    opt_b.zero_grad(); l.backward()
    _cv_note_grad()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt_b.step()
    _cv_apply(_cv_th, step)

    if step % 8 == 0:"""

PATCH = '''
# ---- curvature levers, Phase 3 only ----
_CV_COMPRESS = __COMPRESS__     # top-% of residual coords kept (0 = off)
_CV_SPARSEV  = __SPARSEV__      # fraction of coords keeping Adam's v (0 = off)
_CV_FLIPDAMP = __FLIPDAMP__
_CV_WDSPLIT  = __WDSPLIT__
_CV_K, _CV_WIN, _CV_REBUILD = 4, 8, 2
_CV_VERBOSE  = __VERBOSE__

def _cv_role(nm):
    if nm.startswith("te") or nm.startswith("pe"): return "EMB"
    if "ln" in nm.lower() or nm.endswith("n.weight") or nm.endswith("n.bias"): return "LN"
    if ".ff." in nm: return "FF"
    return "ATTN"

_cv_ps = [p for p in model.parameters() if p.requires_grad]
_cv_P = sum(p.numel() for p in _cv_ps)
_cv_idx, _o = {}, 0
for _nm, _p in model.named_parameters():
    if _p.requires_grad:
        _cv_idx.setdefault(_cv_role(_nm), []).append(torch.arange(_o, _o + _p.numel()))
    _o += _p.numel()
_cv_idx = {k: torch.cat(v) for k, v in _cv_idx.items()}
_cv_hist = {k: [] for k in _cv_idx}
_cv_frame, _cv_flip, _cv_prev = {}, torch.zeros(_cv_P), None
_cv_stats = {"n": 0, "kept": 0, "flip": [], "damp": []}

def _cv_flat():
    return torch.cat([p.data.reshape(-1) for p in _cv_ps]).clone()

def _cv_gflat():
    return torch.cat([(p.grad.reshape(-1) if p.grad is not None
                       else torch.zeros(p.numel())) for p in _cv_ps])

def _cv_note_grad():
    """(S) restrict Adam's second moment to the top coordinates by |g|.
    Applied by zeroing v elsewhere, so those coordinates fall back to
    momentum-only behaviour on the next step."""
    if _CV_SPARSEV <= 0:
        return
    g = _cv_gflat().abs()
    k = max(1, int(_cv_P * _CV_SPARSEV))
    thr = torch.topk(g, k, sorted=False).values.min()
    o = 0
    for p in _cv_ps:
        n = p.numel()
        st = opt_b.state.get(p)
        if st and "exp_avg_sq" in st:
            m = (g[o:o + n] < thr).view_as(p)
            st["exp_avg_sq"][m] = 0.0
        o += n

def _cv_apply(th, step):
    global _cv_prev
    u = _cv_flat() - th
    s = torch.sign(u)
    if _cv_prev is not None:
        f = (s != _cv_prev).float()
        _cv_flip.mul_(0.9).add_(0.1 * f)     # running per-coordinate flip rate
        _cv_stats["flip"].append(float(f.mean()))
    _cv_prev = s
    for k, ii in _cv_idx.items():
        _cv_hist[k].append(u[ii].clone())
        if len(_cv_hist[k]) > _CV_WIN:
            _cv_hist[k].pop(0)
    if len(_cv_hist["FF"]) < _CV_WIN:
        return
    if step % _CV_REBUILD == 0 or not _cv_frame:
        for k, ii in _cv_idx.items():
            A = torch.stack(_cv_hist[k], 1)
            _cv_frame[k] = torch.linalg.svd(A, full_matrices=False)[0][:, :_CV_K]
    nu = torch.zeros_like(u)
    for k, ii in _cv_idx.items():
        Q, ub = _cv_frame[k], u[ii]
        par = Q @ (Q.T @ ub)
        r = ub - par
        if _CV_COMPRESS > 0:
            # (C) keep the residual only on its largest coordinates.
            # ranking by |r_i|: curvature ranking was tested and lost, because
            # this residual is frame-orthogonal, not gradient-orthogonal, so its
            # loss change is first-order dominated.
            kk = max(1, int(len(ii) * _CV_COMPRESS / 100.0))
            keep = torch.topk(r.abs(), kk, sorted=False).indices
            rc = torch.zeros_like(r); rc[keep] = r[keep]
            r = rc
            _cv_stats["kept"] += kk
        nu[ii] = par + r
    if _CV_FLIPDAMP and _cv_stats["flip"]:
        # (F) damp where the sign is churning: alpha_i = (1 - phi_i)^2.
        # phi is an EMA of the per-coordinate flip indicator, so a coordinate
        # that reverses often is stepped less. The flip rate is what predicted
        # both sign failures before the loss did.
        d = (1.0 - _cv_flip).clamp(0, 1) ** 2
        nu = nu * d
        _cv_stats["damp"].append(float(d.mean()))
    if _CV_WDSPLIT:
        # (W) decay the self blocks, leave the interaction alone
        for k, ii in _cv_idx.items():
            if k in ("EMB", "LN"):
                nu[ii] = nu[ii] - 1e-4 * th[ii]
    _cv_stats["n"] += 1
    with torch.no_grad():
        i = 0
        for p in _cv_ps:
            n = p.numel(); p.data.copy_((th + nu)[i:i + n].view_as(p)); i += n

def _cv_report():
    import numpy as _np
    f = _np.mean(_cv_stats["flip"][-20:]) if _cv_stats["flip"] else float("nan")
    kp = _cv_stats["kept"] / max(_cv_stats["n"], 1)
    print(f"  [curv] steps {_cv_stats['n']}  flip {f:.3f}"
          + (f"  resid coords kept {kp:,.0f}/{_cv_P:,}" if _CV_COMPRESS > 0 else "")
          + (f"  sparse-v {_CV_SPARSEV}" if _CV_SPARSEV > 0 else "")
          + (f"  mean damp {_np.mean(_cv_stats['damp'][-20:]):.3f}"
             if _cv_stats["damp"] else ""))
# ---- end curvature levers ----
'''

REPORT_ANCHOR = "step_basin = step"
REPORT_NEW = "_cv_report()\nstep_basin = step"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="compiler_geometri_patched_86.py")
    ap.add_argument("--out", default="compiler_curvature.py")
    ap.add_argument("--compress", type=float, default=0.0,
                    help="(C) keep this TOP PERCENT of residual coordinates. "
                         "1.0 is the measured setting (~12,000 of 1.18M, "
                         "recovering 95-98%%); 0.1 recovers only 66-72%%.")
    ap.add_argument("--sparse-v", type=float, default=0.0, dest="sparsev",
                    help="(S) fraction of coordinates keeping Adam's second "
                         "moment. 0.02 matches the measured diagonal PR.")
    ap.add_argument("--flipdamp", action="store_true",
                    help="(F) per-coordinate damping by running flip rate")
    ap.add_argument("--wd-split", action="store_true", dest="wdsplit",
                    help="(W) decay EMB and LN, leave ATTN<->FF undecayed")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    if a.all:
        a.compress = a.compress or 1.0
        a.sparsev = a.sparsev or 0.02
        a.flipdamp = a.wdsplit = True
    if not os.path.exists(a.src):
        sys.exit(f"not found: {a.src}")
    src = open(a.src).read()
    for anc in (ANCHOR, OLD, REPORT_ANCHOR):
        if anc not in src:
            sys.exit("anchors not found -- targets compiler_geometri_patched_86.py")
    patch = (PATCH.replace("__COMPRESS__", repr(a.compress))
                  .replace("__SPARSEV__", repr(a.sparsev))
                  .replace("__FLIPDAMP__", str(bool(a.flipdamp)))
                  .replace("__WDSPLIT__", str(bool(a.wdsplit)))
                  .replace("__VERBOSE__", str(bool(a.verbose))))
    out = (src.replace(ANCHOR, ANCHOR + "\n" + patch, 1)
              .replace(OLD, NEW, 1)
              .replace(REPORT_ANCHOR, REPORT_NEW, 1))
    open(a.out, "w").write(out)
    print(f"wrote {a.out}")
    print(f"  (C) residual compression : {f'top {a.compress}%' if a.compress else 'OFF'}")
    print(f"  (S) sparse second moment : {a.sparsev if a.sparsev else 'OFF'}")
    print(f"  (F) flip damping         : {'ON' if a.flipdamp else 'OFF'}")
    print(f"  (W) block weight decay   : {'ON' if a.wdsplit else 'OFF'}")
    print(f"  phases 1, 2, 4, 5 unchanged")
    if a.run:
        if not os.path.exists("/tmp/train_ids.json"):
            subprocess.run([sys.executable, "build_corpus.py", "--out", "/tmp",
                            "--loops", "300"], check=True)
        subprocess.run([sys.executable, a.out], check=False)


if __name__ == "__main__":
    main()
