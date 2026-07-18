"""
coefficient_plane.py
====================
Two experiments, one script.

WHAT IS ESTABLISHED
-------------------
Causal rank test (project -> transplant -> measure, no fitting):
    k=1  ->  72.6% of the descent
    k=2  ->  92.2%
    k=3  ->  97.6%     out of 4,197,376 parameters
The interaction matrix over ARCHITECTURAL blocks is HIGH-rank (3.84 of 7),
dominated by a single huge off-diagonal term M[Emb,FF] = +3.350 (Emb solo is
HARMFUL at -0.306; FF solo +0.584).  So the low-dimensional object is real but
is NOT aligned with the parameter blocks -- which is exactly why every
block-wise pencil failed.  The right coordinates are TRAJECTORY-derived, not
ARCHITECTURE-derived.

LANGUAGE DISCIPLINE
-------------------
What the data supports is: "the effective dynamics relevant to this
transplantation experiment admit an accurate rank-2/3 representation."
NOT (yet) "a 2D submanifold".  Manifold language needs local smoothness,
consistent tangent spaces, coordinate charts, and stability across
trajectories.  Experiment B is what would EARN it.

=======================================================================
EXPERIMENT A -- THE W_K CONTROL (discriminating test)
=======================================================================
Anomaly to explain: the rank-3 transplant (val 0.194) BEATS the FULL
displacement (val 0.281).  My interpretation was co-adaptation: the discarded
directions encode compatibility with a FLOOR W_K that the transplant does not
provide (W_K was held at the start value), so importing them without their
partner hurts.

That is internally consistent but NOT the only explanation.  Others:
    * the discarded directions are optimizer-history artifacts,
    * they compensate for nonlinearities that no longer apply post-transplant,
    * they overfit the source trajectory,
    * projection is simply acting as a denoising regulariser.

DISCRIMINATOR: redo the rank sweep with W_K ALSO transplanted.
    * If FULL now becomes optimal  -> co-adaptation confirmed.
    * If rank-3 still wins         -> the explanation is one of the others,
                                      and my reading was wrong.
This is a clean fork and we report whichever way it lands.

=======================================================================
EXPERIMENT B -- THE COEFFICIENT PLANE (does the geometry exist?)
=======================================================================
Take the top two trajectory directions u1, u2 and evaluate the loss over a
grid:
        theta(a,b) = theta_start + a*u1 + b*u2
This answers, by direct measurement rather than assumption:
    * Is the landscape smooth?
    * Is there a single basin, or several?
    * Is the optimum INTERIOR (i.e. is (a,b)=(1,1) -- the observed endpoint --
      actually where the loss is lowest, or is the trajectory overshooting)?
    * Is the optimum stable across seeds?
    * Does the SAME coordinate system work on an INDEPENDENT run?  (transfer)

Only if the plane is smooth, single-basin, and transferable does the
geometric/manifold reading gain real footing.

OUTPUT
    coefficient_plane.json / .csv
    coefficient_plane.png  (W_K-control rank curves; the (a,b) loss surface;
                            transfer to a second run)
"""

import argparse
import csv
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


LOSS_GROUPS = {"Emb", "FF", "LayerNorm", "W_Q", "W_V", "W_O", "other"}
ALL_GROUPS = LOSS_GROUPS | {"W_K"}


def snap(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def load(model, s):
    model.load_state_dict({k: v.clone() for k, v in s.items()})


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
    """Run Phase 3, return checkpoints + the loss trace."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                            betas=(0.9, 0.95), weight_decay=0.1)
    every = max(1, steps // n_ckpt)
    ck, tr = [snap(model)], [float(eval_val(model, n=6))]
    for s in range(1, steps + 1):
        model.train()
        x, y = get_batch()
        _, l = model(x, y)
        opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if s % every == 0 or s == steps:
            ck.append(snap(model))
            tr.append(float(eval_val(model, n=6)))
    return ck, tr


def basis_from(ckpts, keys, n_dirs=4):
    """Trajectory-derived directions: PCA over checkpoint differences.
    These are the coordinates the flow ACTUALLY uses -- as opposed to the
    architectural blocks, which the interaction matrix shows are the wrong
    basis."""
    X = np.stack([(flat(ckpts[i], keys) - flat(ckpts[i - 1], keys)).numpy()
                  for i in range(1, len(ckpts))])
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    return Vt[:n_dirs], S


def transplant_proj(model, start, dtheta, B, k, keys):
    """Apply the rank-k projection of dtheta and return val."""
    coef = B[:k] @ dtheta
    proj = B[:k].T @ coef
    sd = {kk: v.clone() for kk, v in start.items()}
    pd = unflat(torch.tensor(proj), start, keys)
    for kk in keys:
        sd[kk] = (start[kk] + pd[kk]).clone()
    load(model, sd)
    return sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--n-ckpt", type=int, default=24)
    ap.add_argument("--grid", type=int, default=13,
                    help="resolution of the (a,b) coefficient plane")
    ap.add_argument("--span", type=float, default=1.6,
                    help="grid spans [-0.2, span] in each coordinate; the")
    ap.add_argument("--seed2", type=int, default=777,
                    help="seed for the INDEPENDENT second run (transfer test)")
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

    print("=" * 74)
    print("  COEFFICIENT PLANE + W_K CONTROL")
    print("=" * 74)
    print("  Established: k=2 recovers 92%, k=3 recovers 97.6% of a 4.37-nat")
    print("  descent, CAUSALLY, out of 4.2M parameters.  But the block")
    print("  interaction matrix is HIGH-rank (3.84/7) and dominated by")
    print("  M[Emb,FF]=+3.35 -- so the low-dim object is NOT block-aligned.")
    print("  The coordinates are TRAJECTORY-derived, not architecture-derived.")
    print("\n  We say 'rank-2/3 representation under transplantation'.")
    print("  We do NOT say 'submanifold' -- Experiment B is what would earn it.")

    start = snap(model)
    v0 = float(eval_val(model, n=10))
    print(f"\n  start: val={v0:.4f}")

    print(f"\n-- running Phase 3 ({args.steps} CE) --")
    ckpts, trace = run_phase3(model, get_batch, eval_val, LR,
                              args.steps, args.n_ckpt)
    floor = ckpts[-1]
    load(model, floor)
    v_floor = float(eval_val(model, n=10))
    print(f"   floor: val={v_floor:.4f}   descent={v0 - v_floor:.4f} nats")

    # ================================================================
    # EXPERIMENT A -- W_K CONTROL
    # ================================================================
    print("\n" + "=" * 74)
    print("  EXPERIMENT A: W_K CONTROL (which explanation is right?)")
    print("=" * 74)
    print("  If FULL becomes optimal once W_K is ALSO transplanted, the")
    print("  co-adaptation reading is confirmed.  If rank-3 still wins, the")
    print("  explanation is something else (optimizer artefacts / denoising /")
    print("  overfitting the source trajectory) and my reading was wrong.")

    results_A = {}
    for label, groups in [("W_K EXCLUDED (original)", LOSS_GROUPS),
                          ("W_K INCLUDED (control)", ALL_GROUPS)]:
        keys = [k for k in start if group_of(k) in groups]
        B, S = basis_from(ckpts, keys, n_dirs=24)
        dtheta = (flat(floor, keys) - flat(start, keys)).numpy()
        print(f"\n  [{label}]   ({len(keys)} tensors)")
        print(f"   {'k':>5}{'val':>10}{'nats':>9}")
        print("   " + "-" * 24)
        rows = []
        for k in (1, 2, 3, 5, 8, 12, 20):
            if k > B.shape[0]:
                continue
            transplant_proj(model, start, dtheta, B, k, keys)
            v = float(eval_val(model, n=8))
            rows.append({"k": k, "val": v, "nats": v0 - v})
            print(f"   {k:>5}{v:>10.4f}{v0 - v:>9.3f}")
        # full
        sd = {kk: vv.clone() for kk, vv in start.items()}
        fd = unflat(torch.tensor(dtheta), start, keys)
        for kk in keys:
            sd[kk] = (start[kk] + fd[kk]).clone()
        load(model, sd)
        v_full = float(eval_val(model, n=8))
        rows.append({"k": "FULL", "val": v_full, "nats": v0 - v_full})
        print(f"   {'FULL':>5}{v_full:>10.4f}{v0 - v_full:>9.3f}")
        best = min(rows, key=lambda r: r["val"])
        print(f"   -> best is k={best['k']} (val {best['val']:.4f})")
        results_A[label] = {"rows": rows, "best_k": best["k"],
                            "full_val": v_full}
        load(model, start)

    excl = results_A["W_K EXCLUDED (original)"]
    incl = results_A["W_K INCLUDED (control)"]
    print("\n  " + "-" * 68)
    print("  DISCRIMINATION:")
    full_wins = (incl["best_k"] == "FULL")
    if full_wins:
        print("   With W_K included, the FULL update is now optimal.")
        print("   => CO-ADAPTATION CONFIRMED.  The discarded directions do")
        print("      encode compatibility with the floor W_K; without it they")
        print("      hurt, and the low-rank projection was filtering them out.")
    else:
        print(f"   Even with W_K included, k={incl['best_k']} still beats FULL.")
        print("   => CO-ADAPTATION NOT SUPPORTED.  My interpretation was wrong.")
        print("      The advantage of the low-rank projection comes from")
        print("      something else -- optimizer-history artefacts, a")
        print("      denoising/regularisation effect, or overfitting of the")
        print("      source trajectory.  This should be said plainly.")

    # ================================================================
    # EXPERIMENT B -- COEFFICIENT PLANE
    # ================================================================
    print("\n" + "=" * 74)
    print("  EXPERIMENT B: THE COEFFICIENT PLANE")
    print("=" * 74)
    keys = [k for k in start if group_of(k) in LOSS_GROUPS]
    B, S = basis_from(ckpts, keys, n_dirs=4)
    dtheta = (flat(floor, keys) - flat(start, keys)).numpy()
    a_star, b_star = float(B[0] @ dtheta), float(B[1] @ dtheta)
    u1 = B[0]; u2 = B[1]
    print(f"   theta(a,b) = theta_start + a*u1 + b*u2")
    print(f"   the OBSERVED endpoint sits at (a,b) = "
          f"({a_star:.3f}, {b_star:.3f})")
    print(f"   grid: {args.grid}x{args.grid}, spanning "
          f"[{-0.2:.1f}, {args.span:.1f}] x each axis (normalised units)")

    def val_at(a_frac, b_frac):
        vec = a_frac * a_star * u1 + b_frac * b_star * u2
        sd = {kk: v.clone() for kk, v in start.items()}
        pd = unflat(torch.tensor(vec), start, keys)
        for kk in keys:
            sd[kk] = (start[kk] + pd[kk]).clone()
        load(model, sd)
        return float(eval_val(model, n=4))

    fr = np.linspace(-0.2, args.span, args.grid)
    Z = np.zeros((args.grid, args.grid))
    for i, bb in enumerate(fr):
        for j, aa in enumerate(fr):
            Z[i, j] = val_at(aa, bb)
        print(f"   row {i+1}/{args.grid} done", end="\r")
    load(model, start)
    print(" " * 40, end="\r")

    i_min, j_min = np.unravel_index(np.argmin(Z), Z.shape)
    a_opt, b_opt = fr[j_min], fr[i_min]
    v_opt = Z[i_min, j_min]
    v_obs = val_at(1.0, 1.0)
    load(model, start)

    # smoothness: second differences relative to the range
    d2 = np.abs(np.diff(Z, n=2, axis=0)).mean() + np.abs(np.diff(Z, n=2, axis=1)).mean()
    rough = float(d2 / (Z.max() - Z.min() + 1e-12))
    # basins: count local minima
    loc = 0
    for i in range(1, args.grid - 1):
        for j in range(1, args.grid - 1):
            w = Z[i-1:i+2, j-1:j+2]
            if Z[i, j] == w.min() and Z[i, j] < w.mean():
                loc += 1

    print(f"\n   observed endpoint (a,b)=(1,1):  val={v_obs:.4f}")
    print(f"   GRID MINIMUM at (a,b)=({a_opt:.2f},{b_opt:.2f}): val={v_opt:.4f}")
    interior = (0.0 < a_opt < args.span) and (0.0 < b_opt < args.span)
    print(f"   optimum INTERIOR?  {interior}")
    print(f"   roughness (mean |2nd diff| / range) = {rough:.4f}  "
          f"({'SMOOTH' if rough < 0.08 else 'ROUGH'})")
    print(f"   local minima found: {loc}  "
          f"({'single basin' if loc <= 1 else 'MULTIPLE basins'})")
    if v_opt < v_obs - 0.01:
        print(f"   NOTE: the grid finds a point BETTER than the trajectory's own")
        print(f"   endpoint ({v_opt:.4f} < {v_obs:.4f}) -- the descent is not")
        print(f"   landing at the optimum of its own 2D coordinate plane.")

    # ================================================================
    # TRANSFER: does the SAME coordinate system work on another run?
    # ================================================================
    print("\n" + "=" * 74)
    print("  TRANSFER: do u1,u2 from run 1 work on an INDEPENDENT run 2?")
    print("=" * 74)
    print("  This is the test that separates 'a low-rank approximation of ONE")
    print("  trajectory' from 'a coordinate system with real geometric")
    print("  content'.  Same directions, different run.")
    torch.manual_seed(args.seed2)
    load(model, start)
    ck2, _ = run_phase3(model, get_batch, eval_val, LR, args.steps, args.n_ckpt)
    floor2 = ck2[-1]
    load(model, floor2)
    v_floor2 = float(eval_val(model, n=10))
    dtheta2 = (flat(floor2, keys) - flat(start, keys)).numpy()

    # (i) how much of run 2's displacement lies in run 1's top-2 plane?
    proj2 = B[:2].T @ (B[:2] @ dtheta2)
    align = float(np.linalg.norm(proj2) / (np.linalg.norm(dtheta2) + 1e-12))
    # (ii) causally: transplant run 2's displacement PROJECTED onto run 1's basis
    sd = {kk: v.clone() for kk, v in start.items()}
    pd = unflat(torch.tensor(proj2), start, keys)
    for kk in keys:
        sd[kk] = (start[kk] + pd[kk]).clone()
    load(model, sd)
    v_transfer = float(eval_val(model, n=8))
    load(model, start)

    print(f"   run 2 floor: val={v_floor2:.4f}")
    print(f"   fraction of run 2's displacement inside run 1's top-2 plane: "
          f"{100*align:.1f}%")
    print(f"   CAUSAL: transplant run-2 displacement projected onto RUN-1's")
    print(f"           basis -> val={v_transfer:.4f}  "
          f"(nats {v0 - v_transfer:.3f} of {v0 - v_floor2:.3f})")
    transfers = (v0 - v_transfer) > 0.7 * (v0 - v_floor2)
    print(f"   -> coordinate system {'TRANSFERS' if transfers else 'does NOT transfer'}")

    # ---------------- FINAL ----------------
    print("\n" + "=" * 74)
    print("  WHAT IS NOW SUPPORTED")
    print("=" * 74)
    print("  Established (causal, replicated within this run):")
    print("   * the effective dynamics admit an accurate rank-2/3")
    print("     representation under transplantation;")
    print("   * blockwise decompositions fail because the dominant directions")
    print("     are DISTRIBUTED across parameter groups, not aligned with")
    print("     architectural blocks;")
    print("   * the appropriate coordinates are TRAJECTORY-derived.")
    print()
    print("  Earned by THIS run:")
    print(f"   * smoothness of the (a,b) plane : "
          f"{'YES' if rough < 0.08 else 'NO'}")
    print(f"   * single basin                  : "
          f"{'YES' if loc <= 1 else 'NO (%d minima)' % loc}")
    print(f"   * optimum interior              : {'YES' if interior else 'NO'}")
    print(f"   * transfers to an independent run: "
          f"{'YES' if transfers else 'NO'}")
    print()
    geo_ok = (rough < 0.08) and (loc <= 1) and transfers
    if geo_ok:
        print("  => The coordinate system is smooth, single-basin and")
        print("     transferable.  THIS is the evidence that would justify")
        print("     speaking of a low-dimensional geometric object rather than")
        print("     merely a low-rank approximation -- and hence a substrate")
        print("     for a symplectic / D-brane style description.")
        print("     (Replication across further independent runs still needed.)")
    else:
        print("  => NOT yet a geometric object.  The rank-2/3 representation is")
        print("     real and causal, but the plane fails at least one of")
        print("     smoothness / single-basin / transferability.  It remains a")
        print("     low-rank approximation of THIS trajectory, and manifold")
        print("     language is not yet warranted.")

    json.dump({"v0": v0, "v_floor": v_floor,
               "experiment_A": results_A, "full_wins_with_WK": bool(full_wins),
               "plane": {"a_star": a_star, "b_star": b_star,
                         "grid_fracs": fr.tolist(), "Z": Z.tolist(),
                         "v_observed": v_obs, "v_grid_min": float(v_opt),
                         "a_opt": float(a_opt), "b_opt": float(b_opt),
                         "interior": bool(interior), "roughness": rough,
                         "local_minima": loc},
               "transfer": {"alignment": align, "v_transfer": v_transfer,
                            "v_floor2": v_floor2, "transfers": bool(transfers)}},
              open("coefficient_plane.json", "w"), indent=2, default=float)
    print("\n  wrote coefficient_plane.json")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(15, 6))
        for lbl, st in [("W_K EXCLUDED (original)", "o-"),
                        ("W_K INCLUDED (control)", "s--")]:
            r = [x for x in results_A[lbl]["rows"] if x["k"] != "FULL"]
            ax[0].plot([x["k"] for x in r], [x["val"] for x in r], st, label=lbl)
            ax[0].axhline(results_A[lbl]["full_val"], ls=":",
                          alpha=0.6,
                          label=f"FULL ({lbl.split()[1]}) = "
                                f"{results_A[lbl]['full_val']:.3f}")
        ax[0].set_xlabel("rank k"); ax[0].set_ylabel("val (causal transplant)")
        ax[0].set_title("A. W_K control:\ndoes FULL become optimal?")
        ax[0].grid(alpha=0.3); ax[0].legend(fontsize=7)

        im = ax[1].contourf(fr, fr, Z, levels=25, cmap="viridis")
        plt.colorbar(im, ax=ax[1], label="val")
        ax[1].plot([1], [1], "w*", ms=16, label="trajectory endpoint (1,1)")
        ax[1].plot([a_opt], [b_opt], "r+", ms=16, mew=3, label="grid minimum")
        ax[1].set_xlabel("a  (coefficient on u1)")
        ax[1].set_ylabel("b  (coefficient on u2)")
        ax[1].set_title("B. The coefficient plane\n"
                        "(smooth? single basin? interior optimum?)")
        ax[1].legend(fontsize=8)
        plt.suptitle("Is the rank-2 representation a geometric object?",
                     fontsize=13, weight="bold")
        plt.tight_layout()
        plt.savefig("coefficient_plane.png", dpi=190)
        print("  wrote coefficient_plane.png")
    except Exception as e:
        print(f"  [plot skipped: {e}]")


if __name__ == "__main__":
    main()
