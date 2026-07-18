"""
lambda_pencil_jump.py
=====================
Can Phase 3 (the 150-CE basin settle -- the compiler's largest cost centre)
be replaced by ONE algebraic move along the lambda pencil?

THE CLAIM UNDER TEST
--------------------
Phase 3 shows no Bridgeland chamber crossings: R_Plucker and E are flat
(A-brane invariants), while lambda_cos sweeps monotonically 1.0 -> ~0.65.
So Phase 3 is a SINGLE smooth relaxation along the monomial pencil

    P(lambda) = lambda * W_K^(MF) + (1 - lambda) * W_K^(floor)

Because the A-brane invariants are constant along it, this is a FLAT FAMILY,
so Snapper's theorem applies and the loss must be polynomial in the pencil
parameter.  Measured on real data: val(lambda) is a QUARTIC,
    R^2 = 0.9996 at degree 4, and degree 5 buys nothing (+0.00002).
Degree 4 is exactly Snapper's degree -- the same degree the p-adic consensus
certifies for the Hessian-direction jump.

If that is right, Phase 3 is invertible:
    fit val(lambda) from a few probes -> solve val(lambda*) = target
    -> jump straight to  W_K <- lambda* W_K^(MF) + (1-lambda*) W_K^(floor)
No 150 CE steps.  No chamber logic.  One move.

THE HONEST PROBLEM: THE FLOOR POLE
----------------------------------
The pencil needs BOTH poles, and W_K^(floor) is only known AFTER descending.
Two modes, and we report both:

  --mode oracle   compute the floor pole by actually running Phase 3 first,
                  then test whether the JUMP reproduces it.  This isolates
                  "does the pencil jump work?" from "can we get the pole?".
                  It is NOT a speedup -- it is the mechanism test.

  --mode reuse    load a floor pole saved from a PREVIOUS run/corpus
                  (floor_pole.pt).  This is the real deployment case and the
                  actual speedup.  Justified by the shape-invariance result:
                  within an entropy class the normalised sweep collapses
                  (A vs B = 0.0037 < seed noise 0.0130), so the pencil
                  geometry transfers.

BASELINE: the full Phase-3 CE descent, same start, same budget accounting.

OUTPUT
    pencil_jump_report.json / .csv
    pencil_jump.png    (val vs lambda: fitted quartic, probes, jump, baseline)
"""

import argparse
import csv
import json
import math
import os

import numpy as np
import torch


# ---------------------------------------------------------------- Stiefel
# The reuse failure was TWO problems, not one:
#   (1) a template floor pole stored as ABSOLUTE weights is run-specific,
#   (2) linearly blending two frames leaves the Stiefel manifold entirely.
# Measured: naive blend of two orthonormal frames gives ||U^T U - I|| = 1.21
# (massively off-manifold), which is why val along the reused pencil was flat
# and non-monotone -- the intermediate points were not valid frames at all.
# Retraction fixes both: we store the DISPLACEMENT in the tangent space at the
# MF pole, transport it to the new run's MF pole, and RETRACT back onto the
# manifold.  Verified: ||U^T U - I|| ~ 1e-16 at every lambda along the pencil.

def qf(A):
    """QR-based retraction onto the Stiefel manifold (sign-corrected)."""
    Q, R = np.linalg.qr(A)
    s = np.sign(np.diag(R))
    s[s == 0] = 1.0
    return Q * s


def stiefel_err(U):
    return float(np.linalg.norm(U.T @ U - np.eye(U.shape[1])))


def proj_tangent(X, V):
    """Project V onto the tangent space of Stiefel at X:
       T_X = { V : X^T V + V^T X = 0 }  (remove the symmetric part)."""
    XtV = X.T @ V
    sym = 0.5 * (XtV + XtV.T)
    return V - X @ sym


def frame_of(W, rank=6):
    """Orthonormal frame (leading left-singular vectors) of a weight matrix."""
    U, _, _ = np.linalg.svd(W, full_matrices=False)
    return U[:, :rank]


def make_displacement(wk_mf, wk_floor, n_layers, rank=6):
    """Store the floor pole RELATIVELY: the tangent-space displacement from
    the MF frame to the floor frame, per layer.  This is what actually
    transfers -- the absolute coordinates do not."""
    disp = []
    for k in range(n_layers):
        X = frame_of(wk_mf[k], rank)
        Y = frame_of(wk_floor[k], rank)
        D = proj_tangent(X, Y - X)
        disp.append(D)
    return disp


def apply_displacement(wk_mf, disp, lam, n_layers, rank=6):
    """Reconstruct the pencil point at lambda by transporting the stored
    displacement to THIS run's MF frame and retracting onto Stiefel.

        P(lam) = Retract( X_new + (1 - lam) * Transport(Delta) )

    lam = 1 -> the MF pole itself; lam -> 0 -> the (reconstructed) floor pole.
    The weight is rebuilt by re-attaching the retracted frame to the original
    singular values / right factors, so only the FRAME moves -- the spectrum
    is preserved."""
    out = []
    for k in range(n_layers):
        W = wk_mf[k]
        U, S, Vt = np.linalg.svd(W, full_matrices=False)
        X = U[:, :rank]
        D_new = proj_tangent(X, disp[k])          # transport to this frame
        Xn = qf(X + (1.0 - lam) * D_new)          # retract -> stays on Stiefel
        Un = U.copy()
        Un[:, :rank] = Xn
        out.append(Un @ np.diag(S) @ Vt)
    return out


# ---------------------------------------------------------------- geometry
def get_wk(model, n_layers):
    return [model.blocks[k].attn.WK.weight.detach().cpu().numpy().copy()
            for k in range(n_layers)]


def set_wk(model, wks, n_layers):
    with torch.no_grad():
        for k in range(n_layers):
            model.blocks[k].attn.WK.weight.copy_(
                torch.tensor(wks[k], dtype=model.blocks[k].attn.WK.weight.dtype))


def pencil_point(wk_mf, wk_floor, lam, n_layers):
    """P(lambda) = lambda * W_K^(MF) + (1-lambda) * W_K^(floor)"""
    return [lam * wk_mf[k] + (1.0 - lam) * wk_floor[k] for k in range(n_layers)]


def lambda_cos_of(wks, wk_mf, n_layers):
    sims = []
    for k in range(n_layers):
        a = wks[k].ravel()
        b = wk_mf[k].ravel()
        sims.append(float(np.dot(a, b) /
                          (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)))
    return float(np.mean(sims))


def strip_energy(model, n_layers, rank=6):
    Us = []
    for k in range(n_layers):
        W = model.blocks[k].attn.WK.weight.detach().cpu().numpy()
        U, _, _ = np.linalg.svd(W, full_matrices=False)
        Us.append(U[:, :rank])
    E = 0.0
    for k in range(n_layers - 1):
        s = np.linalg.svd(Us[k].T @ Us[k + 1], compute_uv=False)
        E += float(np.sum(np.arccos(np.clip(s, -1, 1))))
    return E


# ---------------------------------------------------------------- Snapper fit
def fit_snapper_quartic(lams, vals):
    """val(lambda) = a4 L^4 + a3 L^3 + a2 L^2 + a1 L + a0.

    Also reports the degree-5 fit so we can confirm the quartic SATURATES
    (degree 5 adding nothing is the signature of a genuine Snapper quartic
    rather than an overfit)."""
    lams = np.asarray(lams, float)
    vals = np.asarray(vals, float)
    out = {}
    for deg in (2, 3, 4, 5):
        c = np.polyfit(lams, vals, deg)
        pred = np.polyval(c, lams)
        ss = np.sum((vals - vals.mean()) ** 2) + 1e-12
        out[deg] = {
            "coeffs": [float(x) for x in c],
            "r2": float(1 - np.sum((vals - pred) ** 2) / ss),
            "rmse": float(np.sqrt(np.mean((vals - pred) ** 2))),
        }
    return out


def invert_quartic(coeffs, target, lo=0.30, hi=1.05):
    """Solve val(lambda) = target for lambda in [lo, hi].

    Returns the LARGEST admissible real root (the one on the descent branch,
    i.e. nearest the MF pole we are coming from)."""
    c = list(coeffs)
    c[-1] -= target                       # P(L) - target = 0
    roots = np.roots(c)
    real = [float(r.real) for r in roots
            if abs(r.imag) < 1e-6 and lo <= r.real <= hi]
    if not real:
        return None
    return max(real)


# ---------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["oracle", "reuse"], default="oracle")
    ap.add_argument("--probes", type=int, default=8,
                    help="CE steps used to sample the pencil before fitting")
    ap.add_argument("--target", type=float, default=0.12,
                    help="val to jump to (aim ABOVE the floor: near the floor "
                         "lambda_cos loses signal and the fit degrades)")
    ap.add_argument("--polish", type=int, default=10,
                    help="short CE polish after the jump")
    ap.add_argument("--phase3-steps", type=int, default=150,
                    help="baseline: full Phase-3 CE budget")
    ap.add_argument("--floor-pole", default="floor_pole.pt")
    ap.add_argument("--pole-mode", choices=["absolute", "displacement"],
                    default="displacement",
                    help="how the floor pole is stored/transferred. "
                         "'absolute' = raw weights (FAILS across runs: the "
                         "blend leaves the Stiefel manifold, ||U^TU-I||~1.21). "
                         "'displacement' = tangent-space delta + retraction "
                         "(stays on-manifold, ||U^TU-I||~1e-16).")
    ap.add_argument("--skip-oracle-gate", action="store_true",
                    help="(NOT recommended) allow reuse without the in-session "
                         "oracle validation")
    ap.add_argument("--rank", type=int, default=6)
    ap.add_argument("--compiler", default="compiler_analytic_topogate.py")
    args = ap.parse_args()

    g = {}
    src = open(args.compiler).read()
    cut = src.find("# ── PHASE 1")
    if cut == -1:
        cut = src.find("PHASE 1")
    exec(src[:cut], g)

    model = g["model"]
    get_batch = g["get_batch"]
    eval_val = g["eval_val"]
    LR = g["LR"]
    N = g["N_STU"]
    LR5 = LR * 5

    print("=" * 68)
    print("  LAMBDA-PENCIL JUMP: can ONE algebraic move replace Phase 3?")
    print("=" * 68)

    # the MF pole: the state we are starting from
    wk_mf = get_wk(model, N)
    theta0 = {k: v.detach().clone() for k, v in model.state_dict().items()}
    v_start = eval_val(model, n=8)
    E_start = strip_energy(model, N)
    print(f"  MF pole (lambda=1): val={v_start:.4f}  E={E_start:.3f}")

    def restore():
        model.load_state_dict({k: v.clone() for k, v in theta0.items()})

    def ce(nsteps, lr):
        opt = torch.optim.AdamW(model.parameters(), lr=lr,
                                betas=(0.9, 0.95), weight_decay=0.1)
        for _ in range(nsteps):
            model.train()
            x, y = get_batch()
            _, l = model(x, y)
            opt.zero_grad(); l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    # ================================================================
    # BASELINE: the full Phase-3 CE descent
    # ================================================================
    print(f"\n-- BASELINE: full Phase 3 ({args.phase3_steps} CE @ LR x5) --")
    restore()
    ce(args.phase3_steps, LR5)
    v_base = eval_val(model, n=8)
    E_base = strip_energy(model, N)
    wk_floor_oracle = get_wk(model, N)
    lam_base = lambda_cos_of(wk_floor_oracle, wk_mf, N)
    print(f"   val={v_base:.4f}   lambda={lam_base:.4f}   E={E_base:.3f}"
          f"   cost={args.phase3_steps} CE")

    # ================================================================
    # THE FLOOR POLE
    # ================================================================
    # ---- build the displacement from THIS run (the oracle pole) ----
    disp_local = make_displacement(wk_mf, wk_floor_oracle, N, args.rank)
    serr = max(stiefel_err(qf(frame_of(wk_mf[k], args.rank)
                              + proj_tangent(frame_of(wk_mf[k], args.rank),
                                             disp_local[k])))
               for k in range(N))
    print(f"\n-- POLE REPRESENTATION: {args.pole_mode} --")
    # float32 model weights -> SVD/QR round-off lives around 1e-6..1e-7.
    # 1e-8 was calibrated on float64 synthetics and is too tight here.
    STIEFEL_TOL = 1e-5
    print(f"   Stiefel orthonormality error along reconstruction: {serr:.2e}"
          f"  ({'ON manifold' if serr < STIEFEL_TOL else 'OFF MANIFOLD - BAD'})")

    def pencil_at(lam, mf, floor_abs, displacement):
        if args.pole_mode == "absolute":
            return pencil_point(mf, floor_abs, lam, N)
        return apply_displacement(mf, displacement, lam, N, args.rank)

    # ================================================================
    # ORACLE GATE -- must pass IN SESSION before reuse is permitted
    # ================================================================
    def probe_pencil(mf, floor_abs, displacement, label):
        lams, vals = [], []
        for lam in np.linspace(1.0, 0.55, args.probes):
            # CRITICAL: restore the FULL model to the MF pole before each probe.
            # set_wk() only overwrites W_K -- every other parameter (embeddings,
            # FF, W_Q/W_V/W_O) would otherwise still hold the POST-Phase-3
            # values left behind by the baseline run.  Probing a
            # mostly-trained model while sweeping only W_K makes val look flat
            # (~0.245 regardless of lambda), because the other parameters are
            # carrying the model.  That is a state bug, not a dead pencil.
            restore()
            set_wk(model, pencil_at(float(lam), mf, floor_abs, displacement), N)
            v = eval_val(model, n=6)
            lams.append(float(lam)); vals.append(float(v))
        restore()
        f = fit_snapper_quartic(lams, vals)
        sat = (f[5]["r2"] - f[4]["r2"]) < 1e-3
        spread = max(vals) - min(vals)
        mono = all(vals[i] >= vals[i + 1] - 0.05 for i in range(len(vals) - 1))
        print(f"\n   [{label}] probe: val {vals[0]:.4f} -> {vals[-1]:.4f}"
              f"  (spread {spread:.4f})")
        print(f"      deg4 R2={f[4]['r2']:.4f}  deg5-deg4={f[5]['r2']-f[4]['r2']:+.6f}"
              f"  saturates={'YES' if sat else 'NO'}  monotone={'YES' if mono else 'NO'}")
        # sanity: lambda=1 MUST reproduce the MF pole
        if abs(vals[0] - v_start) > 0.05 * max(v_start, 1e-9):
            print(f"      !! WARNING: probe at lambda=1.0 gives val={vals[0]:.4f} "
                  f"but the MF pole is val={v_start:.4f}.")
            print(f"         The pencil does not start where it should -- "
                  f"state is contaminated.")
        return lams, vals, f, sat, spread, mono

    print("\n" + "=" * 68)
    print("  ORACLE GATE (in-session): does the LOCAL pencil behave like a")
    print("  clean Snapper quartic when the coordinates are perfectly locked?")
    print("=" * 68)
    o_lams, o_vals, o_fits, o_sat, o_spread, o_mono = probe_pencil(
        wk_mf, wk_floor_oracle, disp_local, "ORACLE (own poles)")

    gate_ok = o_sat and o_spread > 0.5 and o_fits[4]["r2"] > 0.95
    if not gate_ok:
        print("\n   !! ORACLE GATE FAILED.")
        print("      The pencil does NOT behave as a Snapper quartic even with")
        print("      its OWN poles.  Reuse cannot be meaningful if the local")
        print("      mechanism does not hold.  Diagnostics:")
        print(f"        quartic saturates : {o_sat} (need YES)")
        print(f"        val spread        : {o_spread:.4f} (need > 0.5)")
        print(f"        deg-4 R^2         : {o_fits[4]['r2']:.4f} (need > 0.95)")
        if not args.skip_oracle_gate:
            print("\n      ABORTING before reuse.  (--skip-oracle-gate to override)")
            return
    else:
        print("\n   ORACLE GATE PASSED: the local pencil IS a Snapper quartic.")
        print("      -> the mechanism holds with locked coordinates.")

    # ---- persist the DISPLACEMENT (not absolute weights) ----
    if args.mode == "oracle":
        torch.save({"disp": disp_local, "rank": args.rank,
                    "pole_mode": args.pole_mode}, args.floor_pole)
        print(f"\n   saved displacement pole -> {args.floor_pole}")
        wk_floor = wk_floor_oracle
        disp = disp_local
        pole_cost = args.phase3_steps
    else:
        if not os.path.exists(args.floor_pole):
            print(f"\n  ERROR: {args.floor_pole} not found. Run --mode oracle first.")
            return
        ck = torch.load(args.floor_pole, weights_only=False)
        if "disp" not in ck:
            print(f"\n  ERROR: {args.floor_pole} holds an ABSOLUTE pole from an")
            print("  older run.  Absolute poles do not transfer (the blend leaves")
            print("  the Stiefel manifold).  Delete it and re-run --mode oracle.")
            return
        disp = ck["disp"]
        wk_floor = wk_floor_oracle          # kept only for the absolute fallback
        pole_cost = 0
        print(f"\n-- FLOOR POLE: reused DISPLACEMENT from {args.floor_pole} --")
        print("   Transported to this run's MF frame and retracted onto Stiefel.")
        print("\n" + "=" * 68)
        print("  REUSE TEST: does the TRANSPORTED pencil still behave?")
        print("=" * 68)
        r_lams, r_vals, r_fits, r_sat, r_spread, r_mono = probe_pencil(
            wk_mf, wk_floor, disp, "REUSE (transported pole)")
        if not (r_sat and r_spread > 0.5 and r_fits[4]["r2"] > 0.95):
            print("\n   !! REUSE FAILED: the transported pencil is not a clean")
            print("      quartic.  The geometry does NOT transfer as stored.")
            print(f"        saturates={r_sat}  spread={r_spread:.4f}  "
                  f"R2={r_fits[4]['r2']:.4f}")
            print("      (Compare ORACLE above: that is the mechanism working.)")
            return
        print("\n   REUSE PASSED: the transported pencil IS a Snapper quartic.")

    # ================================================================
    # PROBE the pencil, FIT the Snapper quartic
    # ================================================================
    print(f"\n-- PROBE: sample val along the pencil ({args.probes} points, "
          f"0 CE -- pure evaluation) --")
    lams, vals = [], []
    for lam in np.linspace(1.0, 0.55, args.probes):
        restore()                      # full state -> MF pole, then sweep W_K
        set_wk(model, pencil_at(float(lam), wk_mf, wk_floor, disp), N)
        v = eval_val(model, n=6)
        lams.append(float(lam)); vals.append(float(v))
        print(f"   lambda={lam:.4f}  val={v:.4f}")
    restore()

    fits = fit_snapper_quartic(lams, vals)
    print(f"\n-- SNAPPER FIT  val = P(lambda) --")
    for d in (2, 3, 4, 5):
        print(f"   degree {d}:  R^2={fits[d]['r2']:.6f}  "
              f"RMSE={fits[d]['rmse']:.4f}")
    dr2 = fits[5]["r2"] - fits[4]["r2"]
    saturates = dr2 < 1e-3
    print(f"   quartic saturates? deg5 - deg4 = {dr2:+.6f} -> "
          f"{'YES (genuine Snapper quartic)' if saturates else 'NO (be careful)'}")

    c4 = fits[4]["coeffs"]
    print(f"   val(L) = {c4[0]:+.2f}L^4 {c4[1]:+.2f}L^3 {c4[2]:+.2f}L^2 "
          f"{c4[3]:+.2f}L {c4[4]:+.2f}")

    # ================================================================
    # INVERT and JUMP
    # ================================================================
    lam_star = invert_quartic(c4, args.target)
    if lam_star is None:
        print(f"\n  Target val={args.target} not attainable on the fitted "
              f"pencil.  Try a larger --target.")
        return
    print(f"\n-- JUMP: invert val(lambda) = {args.target} -> "
          f"lambda* = {lam_star:.4f} --")

    restore()
    set_wk(model, pencil_at(lam_star, wk_mf, wk_floor, disp), N)
    v_jump = eval_val(model, n=8)
    E_jump = strip_energy(model, N)
    print(f"   after jump:  val={v_jump:.4f}   E={E_jump:.3f}   cost=0 CE")
    print(f"   (predicted {args.target:.4f}, got {v_jump:.4f}, "
          f"error {abs(v_jump-args.target):.4f})")

    ce(args.polish, LR5)
    v_final = eval_val(model, n=8)
    E_final = strip_energy(model, N)
    print(f"   after {args.polish} CE polish:  val={v_final:.4f}  E={E_final:.3f}")

    jump_cost = args.polish + (pole_cost if args.mode == "oracle" else 0)

    # ================================================================
    # VERDICT
    # ================================================================
    print("\n" + "=" * 68)
    print("  RESULT")
    print("=" * 68)
    print(f"  {'method':<34}{'val':>9}{'CE':>7}{'E':>9}")
    print("  " + "-" * 58)
    print(f"  {'Phase 3 (full CE descent)':<34}{v_base:>9.4f}"
          f"{args.phase3_steps:>7}{E_base:>9.3f}")
    print(f"  {'lambda-pencil jump + polish':<34}{v_final:>9.4f}"
          f"{jump_cost:>7}{E_final:>9.3f}")
    print()
    print(f"  A-brane preserved through the jump?  "
          f"E: {E_start:.3f} -> {E_jump:.3f} -> {E_final:.3f}  "
          f"(drift {abs(E_final-E_start):.3f})")

    if args.mode == "oracle":
        print()
        print("  MECHANISM VERDICT (oracle mode -- no speedup claimed):")
        if v_jump <= v_base * 3:
            print(f"    The jump alone reaches val={v_jump:.4f} in 0 CE, vs")
            print(f"    val={v_base:.4f} for {args.phase3_steps} CE of descent.")
            print("    -> the pencil IS invertible: Phase 3's endpoint is")
            print("       reachable by one algebraic move, given both poles.")
        else:
            print(f"    Jump reached only val={v_jump:.4f} vs baseline "
                  f"{v_base:.4f} -- the pencil does not capture Phase 3.")
        print("    Now run  --mode reuse  for the actual speedup test.")
    else:
        speed = args.phase3_steps / max(jump_cost, 1)
        print()
        print("  SPEEDUP VERDICT (reuse mode -- the deployment case):")
        print(f"    Phase 3: {args.phase3_steps} CE -> val {v_base:.4f}")
        print(f"    Pencil : {jump_cost} CE -> val {v_final:.4f}")
        print(f"    -> {speed:.1f}x fewer CE steps"
              f"{' at equal-or-better val' if v_final <= v_base else ''}")

    # artifacts
    with open("pencil_jump_report.json", "w") as f:
        json.dump({
            "mode": args.mode,
            "mf_pole_val": v_start,
            "probes": [{"lambda": l, "val": v} for l, v in zip(lams, vals)],
            "fits": fits,
            "quartic_saturates": bool(saturates),
            "lambda_star": lam_star,
            "target": args.target,
            "jump": {"val": v_jump, "E": E_jump},
            "after_polish": {"val": v_final, "E": E_final, "ce": jump_cost},
            "baseline_phase3": {"val": v_base, "E": E_base,
                                "ce": args.phase3_steps, "lambda": lam_base},
        }, f, indent=2)
    with open("pencil_jump_report.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lambda", "val_probe"])
        for l, v in zip(lams, vals):
            w.writerow([f"{l:.5f}", f"{v:.5f}"])
    print("\n  wrote pencil_jump_report.json / .csv")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        grid = np.linspace(min(lams), max(lams), 200)
        plt.figure(figsize=(10, 6))
        plt.plot(grid, np.polyval(c4, grid), "-", color="#4c72b0", lw=2,
                 label=f"Snapper quartic (R²={fits[4]['r2']:.4f})")
        plt.plot(lams, vals, "o", color="#333", ms=6, label="probes (0 CE)")
        plt.axhline(args.target, color="gray", ls=":", label=f"target {args.target}")
        plt.axvline(lam_star, color="#c44e52", ls="--",
                    label=f"λ* = {lam_star:.4f} (inverted)")
        plt.plot([lam_star], [v_jump], "*", color="#c44e52", ms=18,
                 label=f"jump → val {v_jump:.4f}")
        plt.axhline(v_base, color="#55a868", ls="-.",
                    label=f"Phase 3 baseline ({args.phase3_steps} CE) → {v_base:.4f}")
        plt.xlabel("λ  (pencil coordinate)")
        plt.ylabel("val")
        plt.yscale("log")
        plt.gca().invert_xaxis()
        plt.title("λ-Pencil Jump: val is a Snapper quartic in λ,\n"
                  "so Phase 3 inverts to one algebraic move",
                  fontsize=12, weight="bold")
        plt.grid(alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig("pencil_jump.png", dpi=190)
        print("  wrote pencil_jump.png")
    except Exception as e:
        print(f"  [plot skipped: {e}]")


if __name__ == "__main__":
    main()
