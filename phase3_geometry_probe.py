"""
phase3_geometry_probe.py
========================
LOG-ONLY instrumentation of Phase 3 (basin settle).

Goal: find out, empirically, whether Phase 3 decomposes into smooth pencil
segments separated by discrete Bridgeland chamber crossings.

PRIMARY DETECTOR: R_Plucker  (exit from the Gr(3,5) flat locus)
    A Hamiltonian flow of the W_K Lagrangians embeds in Gr(2,4) (Klein
    quadric, dim 4).  A chamber-crossing JUMP projects onto the singular
    intersection locus there and becomes invisible -- the discrete transition
    looks continuous.  Lifting to Gr(3,5) (dim 6, 10 Plucker coords in P^9)
    separates the branches, so the jump shows up as a genuine excursion of
    the Plucker residual away from the flat locus.  R_Plucker is therefore the
    wall detector; tau and phi flips are corroborating (secondary) signals.

LOGGED EVERY STEP:
    val            cross-entropy on the val set
    R_plucker      Frobenius residual of the Gr(3,5) quadratic relations
    lambda_cos     pencil coordinate: cos-sim of W_K(t) to W_K(MF)  (1.0 -> ~0.700)
    tau            ||grad_FF|| / ||grad_Emb||        (K0 gluing defect)
    E              strip energy  sum_k sum_i arccos(sigma_i)   (should be ~const)
    phi_k          5 Bridgeland sheet angles (arg of dominant eigenvalue)
    Phi_cl         count of phi_k in {0, pi}
    dlambda,d2lambda   first/second differences of lambda_cos (pencil velocity/accel)

OUTPUTS (no jumps executed -- this run only measures):
    phase3_trace.csv      one row per step, all coordinates
    phase3_trace.json     same + detected candidate walls
    phase3_geometry.png   4-panel: val / R_plucker / lambda_cos / tau, walls marked

USAGE
    Drop next to the compiler (it reuses the compiler's model + data defs):
        python phase3_geometry_probe.py --steps 150
    Then inspect the plot: do R_plucker spikes coincide with tau spikes and
    phi_k sign flips?  If yes -> Phase 3 has a chamber decomposition and the
    segments are candidates for per-chamber Snapper fits.
"""

import argparse
import csv
import json
import itertools
import math

import numpy as np
import torch


# ----------------------------------------------------------------------
# Geometry: Gr(3,5) Plucker residual  (THE primary wall detector)
# ----------------------------------------------------------------------
def plucker_coords_gr35(U):
    """Plucker coordinates of a 3-plane in C^5.

    U: (5,3) matrix whose columns span the 3-plane.
    Returns dict {(i,j,k): P_ijk} over the C(5,3)=10 index triples,
    where P_ijk = det of rows (i,j,k) of U.
    """
    P = {}
    for tri in itertools.combinations(range(5), 3):
        sub = U[list(tri), :]                    # (3,3)
        P[tri] = float(np.linalg.det(sub))
    return P


def plucker_residual_gr35(U):
    """Departure of the 5-layer configuration from Gr(3,5) decomposability.

    IMPORTANT (this is the subtle point):  the Plucker relations are satisfied
    IDENTICALLY by any honest 3-frame -- a 5x3 matrix always represents a real
    3-plane, so its residual is 0 to machine precision.  Verified.  A residual
    computed that way can never spike and is useless as a wall detector.

    What DOES spike is the failure of the *5 layer directions taken together*
    to lie in a common 3-plane.  Inside a Bridgeland chamber the layer
    subspaces stay 3-plane-compatible (the flat locus).  At a chamber crossing
    they separate, the configuration becomes non-decomposable, and the Plucker
    quadrics pick it up.  This is exactly the Gr(2,4) -> Gr(3,5) argument: in
    the lower projection the layers still *look* coplanar and the jump is
    invisible; lifted to Gr(3,5) the separation is visible.

    So: take the 5 layer directions (5xD), reduce to a 5x5 Gram-like frame,
    form its degree-3 Plucker vector, normalise projectively, and measure the
    residual of the quadratic relations.  R ~ 0 => flat locus (in-chamber).
    R spiking => wall transit.

    U: (5, k) array of the 5 layer directions in a common chart (k >= 3).
    """
    A = np.asarray(U, dtype=float)
    # project the 5 layer directions onto their own leading 5-dim chart
    # and build the (unnormalised) degree-3 Plucker vector of the TOP-3
    # spectral directions, then measure how non-decomposable the full
    # 5-layer configuration is relative to that 3-plane.
    Uu, Ss, Vt = np.linalg.svd(A, full_matrices=False)
    k = min(3, Vt.shape[0])
    best3 = Uu[:, :k] * Ss[:k]                   # (5,3) best 3-plane fit
    # residual mass outside the best 3-plane == non-decomposability
    resid_mass = float(np.sum(Ss[k:] ** 2)) if len(Ss) > k else 0.0
    total_mass = float(np.sum(Ss ** 2)) + 1e-12

    # Plucker vector of the fitted 3-plane, projectively normalised
    P = np.array([np.linalg.det(best3[list(t), :])
                  for t in itertools.combinations(range(5), 3)])
    nP = np.linalg.norm(P) + 1e-12
    P = P / nP

    # quadratic Plucker relations on the (normalised) coordinates
    keys = list(itertools.combinations(range(5), 3))
    Pd = {t: float(v) for t, v in zip(keys, P)}

    def p(idx):
        idx = list(idx)
        if len(set(idx)) < 3:
            return 0.0
        sign = 1.0
        for a in range(2):
            for b in range(2 - a):
                if idx[b] > idx[b + 1]:
                    idx[b], idx[b + 1] = idx[b + 1], idx[b]
                    sign = -sign
        return sign * Pd[tuple(idx)]

    res = []
    for I in itertools.combinations(range(5), 3):
        for J in itertools.combinations(range(5), 3):
            i1, i2, i3 = I
            lhs = p((i1, i2, i3)) * p(J)
            acc = 0.0
            for l in range(3):
                Js = list(J)
                jl = J[l]
                Js[l] = i1
                acc += p((jl, i2, i3)) * p(tuple(Js))
            res.append(lhs - acc)
    quad_resid = float(np.linalg.norm(res))

    # The reported R combines (a) the quadratic-relation residual of the
    # fitted 3-plane [~0 by construction, kept as a sanity channel] and
    # (b) the NON-DECOMPOSABILITY: the fraction of the 5-layer configuration
    # that refuses to fit in any 3-plane.  (b) is the wall signal.
    non_decomposability = math.sqrt(resid_mass / total_mass)
    return non_decomposability + quad_resid


def wk_subspace_5x3(model, rank=3):
    """The 5 layer-pair directions in a common chart.

    Returns a (5, D) array: row k = leading left-singular vector of W_K^(k).
    We deliberately do NOT pre-reduce to rank 3 here -- the whole point is to
    let plucker_residual_gr35 measure how far these 5 directions are from
    fitting inside a single 3-plane.  Pre-reducing would force them onto a
    3-plane by construction and destroy the wall signal.
    """
    vecs = []
    for k in range(5):
        W = model.blocks[k].attn.WK.weight.detach().cpu().numpy()
        U, S, Vt = np.linalg.svd(W, full_matrices=False)
        v = U[:, 0]
        vecs.append(v / (np.linalg.norm(v) + 1e-12))   # unit, scale-free
    return np.stack(vecs, axis=0)                       # (5, D)


# ----------------------------------------------------------------------
# The other five coordinates
# ----------------------------------------------------------------------
def sheet_angles(model):
    """phi_k = arg(lambda_dom(W_{k+1} W_k^{-1})), dominant = LARGEST MODULUS."""
    phis = []
    for k in range(5):
        Wk = model.blocks[k].attn.WK.weight.detach().cpu().double()
        Wk1 = model.blocks[k + 1].attn.WK.weight.detach().cpu().double()
        try:
            M = Wk1 @ torch.linalg.pinv(Wk)
            lam = torch.linalg.eigvals(M)
            dom = lam[lam.abs().argmax()]        # largest MODULUS (corrected defn)
            phis.append(float(torch.angle(dom)))
        except Exception:
            phis.append(float("nan"))
    return phis


def phi_clean(phis, tol=0.15):
    n = 0
    for p in phis:
        if math.isnan(p):
            continue
        if min(abs(p), abs(abs(p) - math.pi)) < tol:
            n += 1
    return n


def strip_energy(model, rank=6):
    """E = sum_k sum_i arccos(sigma_i(U_k^T U_{k+1}))  -- the Floer invariant."""
    Us = []
    for k in range(6):
        W = model.blocks[k].attn.WK.weight.detach().cpu().numpy()
        U, _, _ = np.linalg.svd(W, full_matrices=False)
        Us.append(U[:, :rank])
    E = 0.0
    for k in range(5):
        s = np.linalg.svd(Us[k].T @ Us[k + 1], compute_uv=False)
        s = np.clip(s, -1.0, 1.0)
        E += float(np.sum(np.arccos(s)))
    return E


def lambda_cos(model, wk_mf):
    """Pencil coordinate: mean cos-sim of W_K(t) with the MF-pump W_K."""
    sims = []
    for k in range(6):
        W = model.blocks[k].attn.WK.weight.detach().cpu().numpy().ravel()
        Wm = wk_mf[k].ravel()
        d = (np.linalg.norm(W) * np.linalg.norm(Wm)) + 1e-12
        sims.append(float(np.dot(W, Wm) / d))
    return float(np.mean(sims))


def tau_defect(model, loss):
    """tau = ||grad_FF|| / ||grad_Emb||   (unsquared denominator -- the
    definition used throughout the main paper)."""
    gff, gem = 0.0, 0.0
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach().norm().item() ** 2
        if ".ff." in name:
            gff += g
        elif name.startswith("te") or name.startswith("pe"):
            gem += g
    return math.sqrt(gff) / (math.sqrt(gem) + 1e-12)


# ----------------------------------------------------------------------
# Wall candidate detection (log-only: we PROPOSE walls, execute nothing)
# ----------------------------------------------------------------------
def detect_walls(trace, z=2.5):
    """Candidate chamber boundaries = R_plucker excursions above the
    flat-locus baseline.  We use a robust (MAD) z-score so a few big spikes
    don't inflate the threshold."""
    R = np.array([t["R_plucker"] for t in trace])
    med = np.median(R)
    mad = np.median(np.abs(R - med)) + 1e-12
    zs = 0.6745 * (R - med) / mad
    walls = [int(trace[i]["step"]) for i in range(len(R)) if zs[i] > z]
    # corroboration: did tau also spike, or a phi_k flip sheet?
    corroborated = []
    for i, t in enumerate(trace):
        if t["step"] not in walls:
            continue
        tau_spike = (i > 0 and trace[i]["tau"] > 1.4 * trace[i - 1]["tau"])
        phi_flip = (i > 0 and trace[i]["Phi_cl"] != trace[i - 1]["Phi_cl"])
        corroborated.append({
            "step": t["step"],
            "R_plucker": t["R_plucker"],
            "z": float(zs[i]),
            "tau_spike": bool(tau_spike),
            "phi_flip": bool(phi_flip),
            "agreement": int(tau_spike) + int(phi_flip),
        })
    return walls, corroborated, float(med), float(mad)


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--lr-mult", type=float, default=5.0)
    ap.add_argument("--compiler", default="compiler_analytic_topogate.py")
    args = ap.parse_args()

    # reuse the compiler's model/data definitions (same trick warmstart uses)
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

    # snapshot the MF-pump W_K as the pencil's lambda=1 pole
    wk_mf = [model.blocks[k].attn.WK.weight.detach().cpu().numpy().copy()
             for k in range(6)]

    opt = torch.optim.AdamW(model.parameters(), lr=LR * args.lr_mult,
                            betas=(0.9, 0.95), weight_decay=0.1)

    trace = []
    print(f"{'step':>5} {'val':>9} {'R_pluck':>9} {'lam_cos':>8} "
          f"{'tau':>7} {'E':>8} {'Phi_cl':>6}")
    print("-" * 62)

    for step in range(args.steps):
        model.train()
        x, y = get_batch()
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()

        tau = tau_defect(model, loss)            # needs grads -> before step
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 2 == 0 or step == args.steps - 1:
            U = wk_subspace_5x3(model)
            R = plucker_residual_gr35(U)
            lam = lambda_cos(model, wk_mf)
            E = strip_energy(model)
            phis = sheet_angles(model)
            pcl = phi_clean(phis)
            v = eval_val(model, n=6)

            rec = {
                "step": step, "val": float(v), "R_plucker": R,
                "lambda_cos": lam, "tau": float(tau), "E": E,
                "Phi_cl": pcl, "phi": phis,
            }
            trace.append(rec)
            print(f"{step:5d} {v:9.4f} {R:9.4f} {lam:8.4f} "
                  f"{tau:7.2f} {E:8.3f} {pcl:5d}/5")

    # pencil velocity / acceleration
    lams = [t["lambda_cos"] for t in trace]
    for i, t in enumerate(trace):
        t["dlambda"] = (lams[i] - lams[i - 1]) if i > 0 else 0.0
        t["d2lambda"] = (t["dlambda"] - trace[i - 1]["dlambda"]) if i > 1 else 0.0

    walls, corr, med, mad = detect_walls(trace)

    # ---- outputs ----
    with open("phase3_trace.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "val", "R_plucker", "lambda_cos", "dlambda",
                    "d2lambda", "tau", "E", "Phi_cl"])
        for t in trace:
            w.writerow([t["step"], f'{t["val"]:.5f}', f'{t["R_plucker"]:.5f}',
                        f'{t["lambda_cos"]:.5f}', f'{t["dlambda"]:+.5f}',
                        f'{t["d2lambda"]:+.5f}', f'{t["tau"]:.4f}',
                        f'{t["E"]:.4f}', t["Phi_cl"]])

    with open("phase3_trace.json", "w") as f:
        json.dump({
            "trace": trace,
            "flat_locus_baseline": {"median_R": med, "mad_R": mad},
            "candidate_walls": walls,
            "corroboration": corr,
            "E_drift": {
                "min": min(t["E"] for t in trace),
                "max": max(t["E"] for t in trace),
                "spread": max(t["E"] for t in trace) - min(t["E"] for t in trace),
            },
        }, f, indent=2)

    # ---- report ----
    print("\n" + "=" * 62)
    print("PHASE 3 GEOMETRY: candidate chamber decomposition")
    print("=" * 62)
    print(f"  Flat-locus baseline R_plucker: median={med:.4f}  MAD={mad:.4f}")
    print(f"  Candidate walls (R spikes):    {walls}")
    print(f"  Segments implied:              {len(walls) + 1}")
    Es = [t["E"] for t in trace]
    print(f"  Strip energy E: {min(Es):.3f} -- {max(Es):.3f} "
          f"(spread {max(Es)-min(Es):.3f})  [should be ~flat]")
    print(f"  lambda_cos sweep: {lams[0]:.4f} -> {lams[-1]:.4f} "
          f"(pencil pole ~0.700)")
    if corr:
        print("\n  Wall corroboration (2-of-3 agreement is the strong case):")
        print(f"  {'step':>5} {'R':>8} {'z':>6} {'tau_spike':>10} {'phi_flip':>9}")
        for c in corr:
            print(f"  {c['step']:5d} {c['R_plucker']:8.4f} {c['z']:6.2f} "
                  f"{str(c['tau_spike']):>10} {str(c['phi_flip']):>9}")
    print("\n  Wrote phase3_trace.csv / .json / phase3_geometry.png")

    # ---- plot ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        steps = [t["step"] for t in trace]
        fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
        panels = [
            ("val", "val (CE)", "#333"),
            ("R_plucker", "R_Plücker  (Gr(3,5) flat-locus exit)", "#c44e52"),
            ("lambda_cos", "λ_cos  (pencil coordinate)", "#4c72b0"),
            ("tau", "τ  (gluing defect)", "#dd8452"),
        ]
        for ax, (key, label, col) in zip(axes, panels):
            ax.plot(steps, [t[key] for t in trace], color=col, lw=1.6)
            ax.set_ylabel(label, fontsize=9)
            ax.grid(alpha=0.3)
            for wstep in walls:
                ax.axvline(wstep, color="red", ls="--", alpha=0.35)
        axes[1].axhline(med, color="green", ls=":", label="flat locus")
        axes[1].legend(fontsize=8)
        axes[2].axhline(0.700, color="green", ls=":", label="floor pole λ*≈0.700")
        axes[2].legend(fontsize=8)
        axes[-1].set_xlabel("Phase 3 step")
        plt.suptitle("Phase 3 Geometry: pencil flow segmented by "
                     "Gr(3,5) chamber crossings (red = candidate walls)",
                     fontsize=13, weight="bold")
        plt.tight_layout()
        plt.savefig("phase3_geometry.png", dpi=190)
    except Exception as e:
        print(f"  [plot skipped: {e}]")


if __name__ == "__main__":
    main()
