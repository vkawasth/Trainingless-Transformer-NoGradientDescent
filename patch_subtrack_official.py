#!/usr/bin/env python3
"""
patch_subtrack_official.py -- RUN THE AUTHORS' OPTIMIZER ON OUR PHASE 3
=======================================================================

Swaps Phase 3's AdamW for SubTrack++'s own LowRankAdamW, unmodified, from
    https://github.com/criticalml-uw/SubTrack

Phases 1, 2, 4 and 5 are byte-identical, so this is their code on our model and
our corpus, with everything else held fixed.

WHY THIS RUN. Every low-rank failure in this programme has had two candidate
causes: my reimplementation, or this landscape. Their code removes the first.

    trains well  -> my reimplementation was the problem, and the gap is findable
    fails as mine did (capture collapses, tracking freezes)
                 -> the landscape explanation holds. The supporting measurement:
                    theta_8 between consecutive gradient subspaces sits at
                    1.52-1.54 rad against pi/2 = 1.571 from the FIRST step, and
                    |grad F| collapses 29x as the subspace decorrelates -- so a
                    geodesic tracker has nothing left to move along, since
                    grad F ~ (I - SS^T) G G^T S vanishes exactly when S is
                    orthogonal to G's column space.

TWO DISCREPANCIES between their README command and their paper, both wired in
here from the README since that is what they actually ran:

    st_init_step_size 10000   Table 10 of the paper says 10. A factor of 1000.
                              It matters: at eta=10 our measured rotation angle
                              was ~1e-4 rad and the subspace never moved.
    scale 0.25                they DO use a scale factor; it is not 1.

THE REGIME CONFOUND, which must be controlled before concluding anything:

    theirs  rank 512, hidden 2048  -> r/d = 1/4
    ours    rank 8,   hidden 128   -> r/d = 1/16

If rank 8 fails, run --rank 32 for a matched ratio BEFORE blaming the
landscape. Their subspace_update_interval default of 200 also exceeds our whole
Phase 3 at budget 150, so the default here is 25 and --budget raises the cap.

REFERENCE NUMBERS on this pipeline:
    AdamW baseline    Phase 3 -> 0.129 at 150 steps, final ~0.05
    best low-rank arm final 0.1600 at 307 CE, 35.5x optimizer memory
    GD-400            0.0914 at 400 steps

USAGE
    python3 patch_subtrack_official.py --subtrack ./SubTrack --run
    python3 patch_subtrack_official.py --subtrack ./SubTrack --rank 32 --run
    python3 patch_subtrack_official.py --subtrack ./SubTrack --budget 400 --run
"""
import argparse, os, subprocess, sys

ANCHOR = """opt_b = torch.optim.AdamW(model.parameters(), lr=LR*5,
                           betas=(0.9,0.95), weight_decay=0.1)"""

REPLACEMENT = '''# ---- SubTrack++ LowRankAdamW, authors' implementation ----
import sys as _sys, types as _types, torch as _t
_sys.path.insert(0, r"__SUBTRACK__")

# Their package __init__ imports adamw8bit, which needs bitsandbytes. Stub it
# so the import succeeds without installing a GPU-only quantisation library we
# are not using.
import importlib.util as _ilu, importlib.machinery as _ilm
try:
    _has_bnb = _ilu.find_spec("bitsandbytes") is not None
except Exception:
    _has_bnb = False
if not _has_bnb and "bitsandbytes" not in _sys.modules:
    # A bare ModuleType has __spec__ = None, and transformers calls
    # importlib.util.find_spec("bitsandbytes"), which RAISES ValueError on a
    # module that is in sys.modules with a null spec. So each stub needs a real
    # ModuleSpec, and a __version__ since some callers read it.
    def _stub(_name, _parent=None):
        _m = _types.ModuleType(_name)
        _m.__spec__ = _ilm.ModuleSpec(_name, None)
        _m.__version__ = "0.0.0"
        _m.__path__ = []
        _sys.modules[_name] = _m
        if _parent is not None:
            setattr(_sys.modules[_parent], _name.rsplit(".", 1)[1], _m)
        return _m
    _b = _stub("bitsandbytes")
    _o = _stub("bitsandbytes.optim", "bitsandbytes")
    _oo = _stub("bitsandbytes.optim.optimizer", "bitsandbytes.optim")
    class _Opt2State:
        def __init__(self, *a, **k): pass
    _oo.Optimizer2State = _Opt2State

# low_rank_projector.py hardcodes .to('cuda') in the tracking path. On CPU (or
# Apple silicon) that raises. Redirect 'cuda' to whatever device is actually
# available -- this changes placement only, not arithmetic.
if not _t.cuda.is_available():
    _real_to = _t.Tensor.to
    def _patched_to(self, *a, **k):
        if a and isinstance(a[0], str) and a[0].startswith("cuda"):
            a = (self.device,) + a[1:]
        elif k.get("device") is not None and str(k["device"]).startswith("cuda"):
            k = dict(k); k["device"] = self.device
        return _real_to(self, *a, **k)
    _t.Tensor.to = _patched_to

from low_rank_torch import LowRankAdamW as _LowRankAdamW

_st_mats, _st_vecs, _st_names = [], [], []
for _n, _p in model.named_parameters():
    if not _p.requires_grad:
        continue
    if _p.dim() == 2:
        _st_mats.append(_p); _st_names.append(_n)
    else:
        _st_vecs.append(_p)

# The optimizer keys off "rank" being present in a param group: groups WITH it
# take the low-rank path, groups WITHOUT it take plain AdamW. So 1-D parameters
# (LayerNorm gains, biases) go in a bare group, exactly as their trainer does.
_st_groups = [
    {"params": _st_mats, "module_names": _st_names,
     "rank": __RANK__,
     "scale": __SCALE__,                       # their --low_rank_scale
     "proj_type": "std",
     "st_init_step_size": __ETA__,             # their --st_init_step_size
     "subspace_update_method": "subtrack",
     "subspace_update_interval": __INTERVAL__,
     "st_step_size_scheduler": None,
     "st_step_size_coef": 1.0,
     "st_noise_sigma2": 0.0,
     "st_subspace_coef": 1.0,
     "rand_proj": False,
     "rand_epoch": 10 ** 9,
     "adaptive_optimizer": __ADAPTIVE__,       # their --adaptive_optimizer
     "recovery_scaling": __RECOVERY__,         # their --recovery_scaling
     "norm_growth_limit": __ZETA__,            # the zeta limiter of eq (12)
     "norm_growth_limiter_off": False},
    {"params": _st_vecs, "module_names": []},
]
opt_b = _LowRankAdamW(_st_groups, lr=LR*5, betas=(0.9,0.95),
                      weight_decay=0.1, adaptive_optimizer=__ADAPTIVE__,
                      no_deprecation_warning=True)
print(f"  [subtrack++] authors' LowRankAdamW: rank __RANK__, scale __SCALE__, "
      f"eta __ETA__, interval __INTERVAL__, "
      f"adaptive=__ADAPTIVE__, recovery=__RECOVERY__")
print(f"  [subtrack++] {len(_st_mats)} matrices low-rank, {len(_st_vecs)} "
      f"1-D params on the plain AdamW path")

def _st_report():
    """Optimizer state accounting and the diagnostic that decides the run.

    capture is what matters: the fraction of the gradient the projector holds.
    Our re-SVD arm reached 0.786; our geodesic-tracking arm collapsed to 0.017
    and stayed there, which is the failure mode this run is testing for."""
    import numpy as _np
    _cn = _cd = 0.0
    _adam = 0; _mine = 0
    for _g in opt_b.param_groups:
        _lowrank = "rank" in _g
        for _p in _g["params"]:
            _adam += 2 * _p.numel()
            if not _lowrank:
                _mine += 2 * _p.numel(); continue
            _d1, _d2 = _p.shape
            _r = min(_g["rank"], min(_d1, _d2))
            _mine += _r * _d1 + 2 * _r * _d2
            _stt = opt_b.state.get(_p, {})
            _pr = _stt.get("projector")
            if _pr is not None and getattr(_pr, "ortho_matrix", None) is not None \\
                    and _p.grad is not None:
                _S = _pr.ortho_matrix
                try:
                    _S = _S if _S.shape[0] == _d1 else _S.T
                    _pj = _S @ (_S.T @ _p.grad)
                    _cn += float((_pj * _pj).sum())
                    _cd += float((_p.grad * _p.grad).sum())
                except Exception:
                    pass
    print(f"  [subtrack++] optimizer state {_mine:,} floats vs AdamW "
          f"{_adam:,} = {_adam/max(_mine,1):.1f}x saving")
    if _cd > 0:
        print(f"  [subtrack++] capture {_cn/_cd:.3f}  "
              f"(re-SVD arm reached 0.786; tracking collapsed to 0.017)")
# ---- end SubTrack++ ----'''

REPORT_ANCHOR = """step_basin = step"""
REPORT_NEW = """_st_report()
step_basin = step"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="compiler_geometri_patched_86.py")
    ap.add_argument("--out", default="compiler_subtrack.py")
    ap.add_argument("--subtrack", default="./SubTrack",
                    help="path to the cloned SubTrack repo")
    ap.add_argument("--rank", type=int, default=8,
                    help="their r/d is 1/4 (rank 512, hidden 2048); ours at "
                         "rank 8 with hidden 128 is 1/16. Use 32 for a matched "
                         "ratio before blaming the landscape.")
    ap.add_argument("--scale", type=float, default=0.25,
                    help="their --low_rank_scale")
    ap.add_argument("--eta", type=float, default=10000.0,
                    help="their --st_init_step_size. The README uses 10000 "
                         "where the paper's Table 10 says 10.")
    ap.add_argument("--interval", type=int, default=25,
                    help="their default is 200, which exceeds our whole Phase 3 "
                         "at budget 150. Raise --budget to use 200.")
    ap.add_argument("--no-adaptive", action="store_false", dest="adaptive",
                    help="drop the projection-aware optimizer")
    ap.add_argument("--no-recovery", action="store_false", dest="recovery",
                    help="drop recovery scaling")
    ap.add_argument("--zeta", type=float, default=1.01)
    ap.add_argument("--budget", type=int, default=150)
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()

    if not os.path.exists(a.src):
        sys.exit(f"not found: {a.src}")
    lr = os.path.join(a.subtrack, "low_rank_torch", "adamw.py")
    if not os.path.exists(lr):
        sys.exit(f"not found: {lr}\n  clone it: git clone --depth 1 "
                 f"https://github.com/criticalml-uw/SubTrack.git")
    src = open(a.src).read()
    if ANCHOR not in src or REPORT_ANCHOR not in src:
        sys.exit("anchors not found -- targets compiler_geometri_patched_86.py")
    if a.budget != 150:
        old = "for step in range(1, 151):"
        if old not in src:
            sys.exit("step-budget anchor not found")
        src = src.replace(old, f"for step in range(1, {a.budget+1}):", 1)
    rep = (REPLACEMENT.replace("__SUBTRACK__", os.path.abspath(a.subtrack))
                      .replace("__RANK__", str(a.rank))
                      .replace("__SCALE__", repr(a.scale))
                      .replace("__ETA__", repr(a.eta))
                      .replace("__INTERVAL__", str(a.interval))
                      .replace("__ADAPTIVE__", str(bool(a.adaptive)))
                      .replace("__RECOVERY__", str(bool(a.recovery)))
                      .replace("__ZETA__", repr(a.zeta)))
    out = src.replace(ANCHOR, rep, 1).replace(REPORT_ANCHOR, REPORT_NEW, 1)
    if "_LowRankAdamW(" not in out or "_st_report()" not in out:
        sys.exit("patch did not apply -- refusing to write a silent no-op")
    open(a.out, "w").write(out)
    print(f"wrote {a.out}")
    print(f"  optimizer  : SubTrack++ LowRankAdamW (authors', unmodified)")
    print(f"  repo       : {os.path.abspath(a.subtrack)}")
    print(f"  rank {a.rank}, scale {a.scale}, eta {a.eta}, interval {a.interval}")
    print(f"  adaptive {a.adaptive}, recovery {a.recovery}, zeta {a.zeta}")
    print(f"  Phase 3 budget {a.budget}")
    print(f"  phases 1, 2, 4, 5 unchanged")
    if a.run:
        if not os.path.exists("/tmp/train_ids.json"):
            subprocess.run([sys.executable, "build_corpus.py", "--out", "/tmp",
                            "--loops", "300"], check=True)
        subprocess.run([sys.executable, a.out], check=False)


if __name__ == "__main__":
    main()
