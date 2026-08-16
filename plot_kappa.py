#!/usr/bin/env python3
"""
plot_kappa.py -- LOG AND PLOT THE CURVATURE ARC FROM PHASE 3
=============================================================

Patches the compiler to record, every few steps of Phase 3:

    v       the drift direction, an averaged gradient, normalised
    lambda  v' H v            on-axis curvature response
    omega   ||H v - lambda v|| off-axis response
    phi     atan2(omega, lambda)
    kappa   omega / lambda

then writes kappa_arc.json and, if matplotlib is present, kappa_arc.png.

WHY PHI AND NOT KAPPA

kappa = omega/lambda is a chart singularity: it blows up as lambda passes
through zero, so it is non-monotone for reasons that have nothing to do with the
trajectory. Measured across one run it goes 2.8 -> 5.8 -> 11.2 -> 2.05 while
phi = atan2(omega, lambda) goes 70.5 -> 80.3 -> 95.1 -> 116.0 degrees, monotone
throughout. Both are plotted so the artefact is visible rather than hidden.

phi > 90 degrees means lambda < 0: the drift sits in NEGATIVE curvature. That
excursion was measured to appear only above a capacity boundary (none at D<=96,
present at D=128 on both seeds), so at D=256 it should be there.

THE ARC

Panel 3 plots log(omega) against log(lambda). A power law omega ~ lambda^p
appears as a straight line; the measured exponent was 0.758 with R^2 = 0.966
over 31 checkpoints, and it predicted an out-of-sample corpus at 0.751. The fit
is printed on the panel. Note lambda < 0 points cannot appear on a log axis and
are dropped, with the count reported -- if most points are dropped, the run is
in the negative-curvature regime and the arc is not the right description.

USAGE
    python3 plot_kappa.py                 # write compiler_kappa.py
    python3 plot_kappa.py --run           # write it and run it
    python3 plot_kappa.py --every 4       # denser sampling (default 8)
    python3 plot_kappa.py --nb 12         # more batches per Hessian probe

Needs compiler_geometri_patched_86.py and build_corpus.py alongside.
matplotlib is optional; without it the JSON is still written.
"""

import argparse
import os
import subprocess
import sys

ANCHOR = """opt_b = torch.optim.AdamW(model.parameters(), lr=LR*5,
                           betas=(0.9,0.95), weight_decay=0.1)"""

PROBE_ANCHOR = """    if step % 8 == 0:"""

PATCH = '''
# ─────────────── curvature arc logging (Phase 3) ───────────────
_KA_EVERY, _KA_NB = __EVERY__, __NB__
_KA = []
_ka_ps = [p for p in model.parameters() if p.requires_grad]

def _ka_flat():
    return torch.cat([p.data.reshape(-1) for p in _ka_ps]).clone()

def _ka_set(t):
    with torch.no_grad():
        i = 0
        for p in _ka_ps:
            n = p.numel(); p.data.copy_(t[i:i + n].view_as(p)); i += n

def _ka_probe(step, val):
    """lambda, omega, phi, kappa for the averaged drift direction"""
    import math as _m
    th = _ka_flat()
    # drift: mean gradient over a few batches, so v is the direction the
    # trajectory is actually travelling rather than one noisy sample
    g = torch.zeros(th.numel())
    for _ in range(_KA_NB):
        x, y = get_batch(); model.zero_grad(); _, l = model(x, y); l.backward()
        g += torch.cat([(p.grad.reshape(-1) if p.grad is not None
                         else torch.zeros(p.numel())) for p in _ka_ps]).detach()
        _ka_set(th)
    model.zero_grad()
    gn = float(g.norm())
    if gn < 1e-12:
        return
    v = g / gn
    # Hv by double backward, averaged over the same number of batches
    Hv = torch.zeros_like(v)
    for _ in range(_KA_NB):
        x, y = get_batch(); model.zero_grad()
        _, l = model(x, y)
        gr = torch.autograd.grad(l, _ka_ps, create_graph=True)
        gf = torch.cat([t.reshape(-1) for t in gr])
        hv = torch.autograd.grad((gf * v).sum(), _ka_ps, allow_unused=True)
        Hv += torch.cat([(t if t is not None else torch.zeros_like(p)).reshape(-1)
                         for t, p in zip(hv, _ka_ps)]).detach()
        _ka_set(th)
    model.zero_grad()
    Hv = Hv / _KA_NB
    lam = float((v * Hv).sum())
    om  = float((Hv - lam * v).norm())
    phi = _m.degrees(_m.atan2(om, lam))
    kap = om / lam if abs(lam) > 1e-12 else float("nan")
    _KA.append(dict(step=step, val=float(val), lam=lam, om=om, phi=phi,
                    kappa=kap, gnorm=gn))
    print(f"    [arc] step {step:3d}  lam={lam:+.5f}  om={om:.5f}  "
          f"phi={phi:6.1f}deg  kappa={kap:+.3f}")

def _ka_dump():
    import json as _j
    _j.dump(_KA, open("kappa_arc.json", "w"), indent=2)
    print(f"  [arc] wrote kappa_arc.json  ({len(_KA)} probes)")
    try:
        import numpy as _np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as _plt
    except Exception as e:
        print(f"  [arc] no matplotlib ({type(e).__name__}) -- JSON only")
        return
    if len(_KA) < 3:
        print("  [arc] too few probes to plot")
        return
    st  = _np.array([r["step"]   for r in _KA], float)
    lam = _np.array([r["lam"]    for r in _KA], float)
    om  = _np.array([r["om"]     for r in _KA], float)
    phi = _np.array([r["phi"]    for r in _KA], float)
    kap = _np.array([r["kappa"]  for r in _KA], float)
    val = _np.array([r["val"]    for r in _KA], float)

    fig, ax = _plt.subplots(2, 2, figsize=(12, 8))

    a = ax[0, 0]
    a.plot(st, phi, "o-", color="#1f77b4", lw=2, ms=4)
    a.axhline(90, color="crimson", ls="--", lw=1)
    a.text(st[0], 91, "  phi = 90 deg: lambda changes sign", color="crimson", fontsize=8)
    a.set_xlabel("Phase 3 step"); a.set_ylabel("phi = atan2(omega, lambda)  [deg]")
    a.set_title("phi -- monotone; resolves the kappa singularity")
    a.grid(alpha=.3)

    a = ax[0, 1]
    a.plot(st, kap, "o-", color="#d62728", lw=2, ms=4)
    a.axhline(0, color="k", lw=.8)
    a.set_xlabel("Phase 3 step"); a.set_ylabel("kappa = omega / lambda")
    a.set_title("kappa -- blows up where lambda crosses zero")
    a.grid(alpha=.3)

    a = ax[1, 0]
    m = (lam > 0) & (om > 0)
    ndrop = int((~m).sum())
    if m.sum() >= 3:
        x, y = _np.log(lam[m]), _np.log(om[m])
        p = _np.polyfit(x, y, 1)
        yh = _np.polyval(p, x)
        ss = 1 - ((y - yh) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-30)
        a.plot(lam[m], om[m], "o", color="#2ca02c", ms=5)
        xs = _np.linspace(x.min(), x.max(), 50)
        a.plot(_np.exp(xs), _np.exp(_np.polyval(p, xs)), "-", color="k", lw=1.5,
               label=f"omega ~ lambda^{p[0]:.3f}   R2={ss:.3f}")
        a.set_xscale("log"); a.set_yscale("log"); a.legend(fontsize=9)
    a.set_xlabel("lambda  (on-axis curvature)"); a.set_ylabel("omega  (off-axis)")
    a.set_title(f"the arc" + (f"  [{ndrop} lambda<0 points dropped]" if ndrop else ""))
    a.grid(alpha=.3, which="both")

    a = ax[1, 1]
    a.plot(val, phi, "o-", color="#9467bd", lw=2, ms=4)
    a.axhline(90, color="crimson", ls="--", lw=1)
    a.set_xscale("log"); a.invert_xaxis()
    a.set_xlabel("val  (progress ->)"); a.set_ylabel("phi [deg]")
    a.set_title("phi against progress, not step")
    a.grid(alpha=.3)

    fig.tight_layout()
    fig.savefig("kappa_arc.png", dpi=140)
    print("  [arc] wrote kappa_arc.png")
# ───────────────── end curvature arc logging ─────────────────
'''

PROBE_NEW = """    if step % _KA_EVERY == 0:
        _ka_probe(step, eval_val(model, n=4))

    if step % 8 == 0:"""

DUMP_ANCHOR = """step_basin = step"""
DUMP_NEW = """_ka_dump()
step_basin = step"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="compiler_geometri_patched_86.py")
    ap.add_argument("--out", default="compiler_kappa.py")
    ap.add_argument("--every", type=int, default=8, help="steps between probes")
    ap.add_argument("--nb", type=int, default=8, help="batches per probe")
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()

    if not os.path.exists(a.src):
        sys.exit(f"not found: {a.src}")
    src = open(a.src).read()
    for anc in (ANCHOR, PROBE_ANCHOR, DUMP_ANCHOR):
        if anc not in src:
            sys.exit("anchors not found -- this targets compiler_geometri_patched_86.py")

    patch = PATCH.replace("__EVERY__", str(a.every)).replace("__NB__", str(a.nb))
    out = (src.replace(ANCHOR, ANCHOR + "\n" + patch, 1)
              .replace(PROBE_ANCHOR, PROBE_NEW, 1)
              .replace(DUMP_ANCHOR, DUMP_NEW, 1))
    open(a.out, "w").write(out)
    print(f"wrote {a.out}")
    print(f"  probe every {a.every} steps, {a.nb} batches each")
    print(f"  each probe costs {a.nb} gradients + {a.nb} Hessian-vector products,")
    print(f"  so the run is slower than an unpatched one -- it is instrumentation,")
    print(f"  not a training change. Phases 1, 2, 4, 5 untouched.")

    if a.run:
        if not os.path.exists("/tmp/train_ids.json"):
            print("building corpus...")
            subprocess.run([sys.executable, "build_corpus.py", "--out", "/tmp",
                            "--loops", "300"], check=True)
        print()
        subprocess.run([sys.executable, a.out], check=False)


if __name__ == "__main__":
    main()
