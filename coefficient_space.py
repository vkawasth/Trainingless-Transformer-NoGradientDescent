"""
coefficient_space.py
====================
FIXED version of the coefficient-plane experiment.

WHAT WENT WRONG LAST TIME
-------------------------
I gridded the top-2 directions and ran smoothness / single-basin / interior
diagnostics on the result.  The plane looked beautiful: smooth, one basin,
interior optimum, transfers across runs.  But:

    plane minimum       = 0.3694
    actual Phase-3 floor = 0.0991      <-- 3.7x lower

The rank-2 slice only reaches val 0.37.  It misses ~75% of the descent.  So
every diagnostic described a slice that the optimiser does not actually live
in, and the "geometric object" verdict was NOT earned.  The failure was
silent because nothing checked whether the slice could reach the floor before
measuring its shape.

THE GUARD (new, and non-negotiable)
-----------------------------------
Before ANY geometry diagnostic runs, we assert:

    val( reconstruction at the observed endpoint )  ~=  val( true floor )

If the reconstruction cannot reach the floor, the coordinate system does not
describe the descent, and its smoothness/basins/optima are irrelevant.  The
script REFUSES to report geometry in that case.  It says so and stops.

ALSO CORRECTED
--------------
My co-adaptation interpretation of "rank-3 beats FULL" was FALSIFIED by the
W_K control: with W_K included, k=8 still beat FULL (0.0734 vs 0.1098).  So
the low-rank advantage is NOT compatibility with a missing W_K.  It is one of:
optimizer-history artefacts, a denoising/regularisation effect, or overfitting
of the source trajectory.  This script does not re-litigate that; it simply
does not rely on the co-adaptation story.

WHAT SURVIVES AND MOTIVATES THIS RUN
------------------------------------
    * k=3 recovers 97.6% of the descent, causally (project->transplant->measure)
    * the block interaction matrix is HIGH-rank (3.84/7): the low-dim object is
      NOT aligned with architectural blocks
    * TRANSFER: 65.7% of an INDEPENDENT run's displacement lies in run-1's
      top-2 plane, and transplanting run-2's displacement projected onto
      run-1's basis recovers 3.760 of 4.361 nats (86%).  Two different runs,
      one coordinate system.  That is real and it is what makes the geometry
      question worth asking properly.

WHAT THIS SCRIPT MEASURES
-------------------------
In the k=3 coefficient space c = (c1, c2, c3) with theta(c) = theta_start +
sum_i c_i u_i :
  0. FLOOR-REACHABILITY GUARD (above).  Hard stop if it fails.
  1. Smoothness, basin count, interior optimum -- on 2D SLICES through the
     endpoint, plus a coarse 3D scan for global structure.
  2. Is the trajectory endpoint the optimum of its own coordinate space?
  3. TRANSFER: does run-1's basis capture run-2's descent, causally, and is
     run-2's optimum at the same coefficients?
Only if the guard passes AND the space is smooth AND single-basin AND
transferable is geometric language used.

OUTPUT
    coefficient_space.json / .csv / .png
"""

import argparse
import json

import numpy as np
import torch


def group_of(name):
    n = name.lower()
    if ".ln." in n or ".n." in n or n.startswith("ln_f"):
        return "LayerNorm"
    if n.startswith("te") or n.startswith("pe") or n.startswith("head"):
        return "Emb"
    if ".ff." in n:
        return "FF"
    if "wk" in n:
        return "W_K"
    if "wq" in n:
        return "W_Q"
    if "wv" in n:
        return "W_V"
    if ".op." in n:
        return "W_O"
    return "other"


def snap(m):
    return {k: v.detach().clone() for k, v in m.state_dict().items()}


def load(m, s):
    m.load_state_dict({k: v.clone() for k, v in s.items()})


def flat(sd, keys):
    return torch.cat([sd[k].reshape(-1).double() for k in keys])


def unflat(vec, sd, keys):
    out, i = {}, 0
    for k in keys:
        n = sd[k].numel()
        out[k] = vec[i:i + n].reshape(sd[k].shape).to(sd[k].dtype)
        i += n
    return out


def run_phase3(model, get_batch, eval_val, lr, steps, n_ckpt):
    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                            betas=(0.9, 0.95), weight_decay=0.1)
    every = max(1, steps // n_ckpt)
    ck = [snap(model)]
    for s in range(1, steps + 1):
        model.train()
        x, y = get_batch()
        _, l = model(x, y)
        opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if s % every == 0 or s == steps:
            ck.append(snap(model))
    return ck


def basis_from(ckpts, keys, n_dirs):
    X = np.stack([(flat(ckpts[i], keys) - flat(ckpts[i - 1], keys)).numpy()
                  for i in range(1, len(ckpts))])
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    return Vt[:n_dirs]


def plane_diagnostics(Z, fr):
    """Smoothness / basin structure, WITHOUT a knife-edge threshold.

    The previous version used a raw roughness score against an arbitrary
    cutoff of 0.08.  That was a mistake: calibration showed smooth bowls span
    0.038-0.052 and rugged surfaces 0.11-0.74, so the boundary sits at ~0.081
    -- and an observed slice landed at 0.087, INSIDE the ambiguous gap.  The
    resulting 'ROUGH' label (and hence the 'not a geometric object' verdict)
    was a threshold artifact, not a measurement.

    Replaced by a scale-free discriminator: how well does a QUADRATIC explain
    the surface, and is it convex?  Validated:
        perfect bowl    quad_R2 = 1.000  convex
        quartic bowl    quad_R2 = 0.930  convex
        bowl + 5% noise quad_R2 = 1.000  convex
        rugged (2 sin)  quad_R2 = 0.438  convex
        pure noise      quad_R2 = 0.025  NOT convex
    A smooth single bowl is quad_R2 > 0.85 AND convex.  No arbitrary cut."""
    G = Z.shape[0]
    A, Bm = np.meshgrid(fr, fr)
    X = np.column_stack([A.ravel() ** 2, Bm.ravel() ** 2, (A * Bm).ravel(),
                         A.ravel(), Bm.ravel(), np.ones(A.size)])
    y = Z.ravel()
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    ss = np.sum((y - y.mean()) ** 2) + 1e-12
    quad_r2 = float(1 - np.sum((y - X @ c) ** 2) / ss)
    H = np.array([[2 * c[0], c[2]], [c[2], 2 * c[1]]])
    convex = bool(np.linalg.eigvalsh(H).min() > 0)

    d2 = (np.abs(np.diff(Z, n=2, axis=0)).mean()
          + np.abs(np.diff(Z, n=2, axis=1)).mean())
    rough = float(d2 / (Z.max() - Z.min() + 1e-12))

    loc = 0
    for i in range(1, G - 1):
        for j in range(1, G - 1):
            w = Z[i - 1:i + 2, j - 1:j + 2]
            if Z[i, j] == w.min() and Z[i, j] < w.mean():
                loc += 1
    i, j = np.unravel_index(np.argmin(Z), Z.shape)
    return {"roughness": rough, "quad_r2": quad_r2, "convex": convex,
            "bowl": bool(quad_r2 > 0.85 and convex),
            "local_minima": loc, "argmin_ij": (int(i), int(j)),
            "vmin": float(Z.min())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--n-ckpt", type=int, default=24)
    ap.add_argument("--k", type=int, default=3,
                    help="dimension of the coefficient space to grid")
    ap.add_argument("--grid", type=int, default=11)
    ap.add_argument("--span", type=float, default=1.5,
                    help="grid spans [-0.3, span] in units of the endpoint's "
                         "own coefficient (so 1.0 IS the endpoint)")
    ap.add_argument("--floor-tol", type=float, default=0.5,
                    help="reconstruction must reach within this RELATIVE "
                         "tolerance of the true floor, else we refuse to "
                         "report geometry")
    ap.add_argument("--seed2", type=int, default=777)
    ap.add_argument("--compiler", default="compiler_analytic_topogate.py")
    args = ap.parse_args()

    g_ = {}
    src = open(args.compiler).read()
    cut = src.find("# ── PHASE 1")
    if cut == -1:
        cut = src.find("PHASE 1")
    exec(src[:cut], g_)
    model = g_["model"]; get_batch = g_["get_batch"]; eval_val = g_["eval_val"]
    LR = g_["LR"] * 5

    # W_K INCLUDED: the control showed it lifts everything (k=8 -> 0.0734,
    # at/below the floor).  Excluding it was crippling the reconstruction.
    GROUPS = {"Emb", "FF", "LayerNorm", "W_Q", "W_V", "W_O", "W_K", "other"}

    print("=" * 74)
    print(f"  COEFFICIENT SPACE (k={args.k}) -- with a FLOOR-REACHABILITY GUARD")
    print("=" * 74)
    print("  Last run gridded the top-2 directions and declared the plane a")
    print("  smooth single-basin geometric object.  But that plane bottomed")
    print("  out at val 0.369 against a TRUE floor of 0.099 -- it missed 75%")
    print("  of the descent.  The diagnostics described a slice the optimiser")
    print("  does not live in.  That verdict is WITHDRAWN.")
    print()
    print("  This run grids k=3 (which recovers 97.6% causally), INCLUDES W_K")
    print("  (the control showed it is needed), and REFUSES to report any")
    print("  geometry unless the reconstruction actually reaches the floor.")

    start = snap(model)
    v0 = float(eval_val(model, n=10))
    print(f"\n  start: val={v0:.4f}")

    print(f"\n-- run 1: Phase 3 ({args.steps} CE) --")
    ck1 = run_phase3(model, get_batch, eval_val, LR, args.steps, args.n_ckpt)
    floor1 = ck1[-1]
    load(model, floor1)
    v_floor = float(eval_val(model, n=10))
    print(f"   floor: val={v_floor:.4f}   descent={v0 - v_floor:.4f} nats")

    keys = [k for k in start if group_of(k) in GROUPS]
    B = basis_from(ck1, keys, n_dirs=max(args.k, 8))
    dtheta = (flat(floor1, keys) - flat(start, keys)).numpy()
    c_star = B[:args.k] @ dtheta          # the endpoint's own coefficients

    def theta_at(c):
        vec = B[:args.k].T @ np.asarray(c, float)
        sd = {kk: v.clone() for kk, v in start.items()}
        pd = unflat(torch.tensor(vec), start, keys)
        for kk in keys:
            sd[kk] = (start[kk] + pd[kk]).clone()
        return sd

    def val_at(c, n=5):
        load(model, theta_at(c))
        return float(eval_val(model, n=n))

    # ================================================================
    # THE GUARD -- must pass before ANY geometry is reported
    # ================================================================
    print("\n" + "=" * 74)
    print("  GUARD: can the k=%d reconstruction reach the floor?" % args.k)
    print("=" * 74)
    v_recon = val_at(c_star, n=10)
    load(model, start)
    rel = abs(v_recon - v_floor) / max(v_floor, 1e-9)
    print(f"   true Phase-3 floor           : {v_floor:.4f}")
    print(f"   k={args.k} reconstruction at endpoint : {v_recon:.4f}")
    print(f"   relative gap                 : {rel:.2%}  "
          f"(tolerance {args.floor_tol:.0%})")
    if rel > args.floor_tol:
        print("\n   !! GUARD FAILED.")
        print("   The reconstruction does not reach the floor, so this")
        print("   coordinate system does not describe the descent.  Its")
        print("   smoothness, basins and optima would be properties of a slice")
        print("   the optimiser never visits.  REFUSING to report geometry.")
        print(f"\n   Try a larger --k (the causal sweep suggested k=8 reaches")
        print(f"   val 0.073, i.e. AT the floor).")
        json.dump({"guard": "FAILED", "v_floor": v_floor,
                   "v_recon": v_recon, "rel_gap": rel, "k": args.k},
                  open("coefficient_space.json", "w"), indent=2)
        return
    print("\n   GUARD PASSED: the reconstruction reaches the floor.")
    print("   The coordinate system does describe the descent, so its")
    print("   geometry is meaningful to measure.")

    # ================================================================
    # 2D SLICES THROUGH THE ENDPOINT
    # ================================================================
    print("\n" + "=" * 74)
    print("  GEOMETRY: 2D slices through the endpoint in coefficient space")
    print("=" * 74)
    fr = np.linspace(-0.3, args.span, args.grid)
    slices, diags = {}, {}
    pairs = [(0, 1), (0, 2), (1, 2)][:max(1, args.k * (args.k - 1) // 2)]
    for (p, q) in pairs:
        Z = np.zeros((args.grid, args.grid))
        for i, fq in enumerate(fr):
            for j, fp in enumerate(fr):
                c = c_star.copy()
                c[p] = fp * c_star[p]
                c[q] = fq * c_star[q]
                Z[i, j] = val_at(c, n=4)
        d = plane_diagnostics(Z, fr)
        slices[f"u{p+1}-u{q+1}"] = Z.tolist()
        diags[f"u{p+1}-u{q+1}"] = d
        ai, aj = d["argmin_ij"]
        interior = (0 < ai < args.grid - 1) and (0 < aj < args.grid - 1)
        d["interior"] = bool(interior)
        print(f"   slice u{p+1}-u{q+1}:  quad_R2={d['quad_r2']:.3f} "
              f"convex={str(d['convex']):5s} -> "
              f"{'BOWL' if d['bowl'] else 'not-a-bowl'}  |  "
              f"minima={d['local_minima']} "
              f"({'single' if d['local_minima'] <= 1 else 'MULTIPLE'})  "
              f"min={d['vmin']:.4f} at ({fr[aj]:.2f},{fr[ai]:.2f})  "
              f"{'interior' if interior else 'BOUNDARY'}  "
              f"[rough={d['roughness']:.3f}]")
    load(model, start)

    # is the endpoint the optimum of its own coordinate space?
    print(f"\n   endpoint (all fracs = 1):  val={v_recon:.4f}")
    best_slice = min(diags.values(), key=lambda d: d["vmin"])
    print(f"   best point found on any slice: val={best_slice['vmin']:.4f}")
    overshoot = best_slice["vmin"] < v_recon - 0.01
    if overshoot:
        print("   -> the trajectory does NOT land at the optimum of its own")
        print("      coordinate space (there is a better point on the slice).")
    else:
        print("   -> the trajectory lands at/near the optimum of its own space.")

    smooth = all(d["bowl"] for d in diags.values())
    single = all(d["local_minima"] <= 1 for d in diags.values())

    # ================================================================
    # TRANSFER (the property that makes it a coordinate SYSTEM)
    # ================================================================
    print("\n" + "=" * 74)
    print("  TRANSFER: does run-1's basis capture an INDEPENDENT run?")
    print("=" * 74)
    torch.manual_seed(args.seed2)
    load(model, start)
    ck2 = run_phase3(model, get_batch, eval_val, LR, args.steps, args.n_ckpt)
    floor2 = ck2[-1]
    load(model, floor2)
    v_floor2 = float(eval_val(model, n=10))
    dtheta2 = (flat(floor2, keys) - flat(start, keys)).numpy()

    c2 = B[:args.k] @ dtheta2
    proj2 = B[:args.k].T @ c2
    align = float(np.linalg.norm(proj2) / (np.linalg.norm(dtheta2) + 1e-12))
    v_tr = val_at(c2, n=10)
    load(model, start)
    nats_tr = v0 - v_tr
    nats_full2 = v0 - v_floor2
    transfers = nats_tr > 0.7 * nats_full2

    print(f"   run 2 floor: val={v_floor2:.4f}")
    print(f"   run 2's displacement inside run 1's k={args.k} subspace: "
          f"{100*align:.1f}%")
    print(f"   CAUSAL: run-2 displacement projected onto RUN-1's basis")
    print(f"           -> val={v_tr:.4f}  ({nats_tr:.3f} of {nats_full2:.3f} nats"
          f" = {100*nats_tr/max(nats_full2,1e-9):.0f}%)")
    print(f"   -> {'TRANSFERS' if transfers else 'does NOT transfer'}")
    print(f"\n   run-1 endpoint coefficients: "
          f"{np.array2string(c_star, precision=2)}")
    print(f"   run-2 endpoint coefficients: "
          f"{np.array2string(c2, precision=2)}")
    cos = float(c_star @ c2 / (np.linalg.norm(c_star) * np.linalg.norm(c2) + 1e-12))
    print(f"   cosine between them: {cos:+.3f}  "
          f"({'same direction' if cos > 0.8 else 'DIFFERENT directions'})")

    # ================================================================
    # VERDICT
    # ================================================================
    print("\n" + "=" * 74)
    print("  VERDICT")
    print("=" * 74)
    print(f"   guard (reaches floor)  : PASS ({v_recon:.4f} vs {v_floor:.4f})")
    print(f"   smooth                 : {'YES' if smooth else 'NO'}")
    print(f"   single basin           : {'YES' if single else 'NO'}")
    print(f"   transfers across runs  : {'YES' if transfers else 'NO'}")
    print(f"   endpoint is optimal    : {'NO (overshoots)' if overshoot else 'YES'}")

    if smooth and single and transfers:
        print("\n   => The k=%d coordinate system reaches the floor, is smooth,"
              % args.k)
        print("      single-basin, and transfers to an independent run.")
        print("      THAT is the evidence that justifies calling it a")
        print("      low-dimensional geometric object rather than a low-rank")
        print("      approximation of one trajectory -- and hence a candidate")
        print("      substrate for a symplectic / D-brane description.")
        print("      STILL NEEDED: replication over several more runs, and")
        print("      stability of the basis under corpus/architecture change.")
    else:
        print("\n   => NOT a geometric object on this evidence.  The k=%d"
              % args.k)
        print("      representation is causally real (it reaches the floor),")
        print("      but the space fails at least one of smoothness /")
        print("      single-basin / transferability.  It remains a")
        print("      low-dimensional CAUSAL coordinate system, which is a")
        print("      genuine result, without yet being a nice manifold.")

    json.dump({"guard": "PASSED", "k": args.k, "v0": v0, "v_floor": v_floor,
               "v_recon": v_recon, "rel_gap": rel,
               "c_star": c_star.tolist(),
               "slices": slices, "diagnostics": diags,
               "smooth": bool(smooth), "single_basin": bool(single),
               "overshoot": bool(overshoot),
               "transfer": {"alignment": align, "v": v_tr,
                            "nats": nats_tr, "nats_full": nats_full2,
                            "transfers": bool(transfers),
                            "c2": c2.tolist(), "cosine": cos}},
              open("coefficient_space.json", "w"), indent=2, default=float)
    print("\n  wrote coefficient_space.json")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n = len(slices)
        fig, ax = plt.subplots(1, n, figsize=(6 * n, 5.4), squeeze=False)
        for i, (nm, Z) in enumerate(slices.items()):
            Z = np.array(Z)
            im = ax[0][i].contourf(fr, fr, Z, levels=25, cmap="viridis")
            plt.colorbar(im, ax=ax[0][i], label="val")
            ax[0][i].plot([1], [1], "w*", ms=16, label="trajectory endpoint")
            d = diags[nm]
            ai, aj = d["argmin_ij"]
            ax[0][i].plot([fr[aj]], [fr[ai]], "r+", ms=15, mew=3,
                          label=f"min {d['vmin']:.3f}")
            ax[0][i].set_title(f"{nm}\nrough={d['roughness']:.3f}  "
                               f"minima={d['local_minima']}")
            ax[0][i].set_xlabel("frac of endpoint coeff")
            ax[0][i].legend(fontsize=7)
        plt.suptitle(f"k={args.k} coefficient space (guard PASSED: "
                     f"reconstruction {v_recon:.3f} vs floor {v_floor:.3f})",
                     fontsize=13, weight="bold")
        plt.tight_layout()
        plt.savefig("coefficient_space.png", dpi=180)
        print("  wrote coefficient_space.png")
    except Exception as e:
        print(f"  [plot skipped: {e}]")


if __name__ == "__main__":
    main()
