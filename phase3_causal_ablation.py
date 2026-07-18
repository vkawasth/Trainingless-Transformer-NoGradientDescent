"""
phase3_causal_ablation.py
=========================
FULL MEASUREMENT.  No hypothesis, no fitting -- just: where does the Phase-3
descent actually live, and is any of it linearly parameterisable?

WHY THIS EXISTS
---------------
We fitted val(lambda_cos) on a Phase-3 trace, got R^2 = 0.9996 saturating at
degree 4, and concluded Phase 3 was invertible along a W_K pencil.  That was
CORRELATION.  The causal test (transplant the floor W_K into the MF-pole model)
showed W_K carries only ~0.4% of the descent: sweeping W_K from the MF pole to
the TRUE floor pole moves val 4.3469 -> 4.3298, against a total descent of
4.4644 -> 0.0822.  lambda_cos was simply another monotone function of step,
riding alongside a descent happening elsewhere.

The paper already predicted this: |grad_E| / |grad_W_K| ~ 254x.  W_K carries
the GEOMETRY (sheet angles, strip energy) -- not the LOSS.

REGIME WARNING (this is the point)
----------------------------------
Phase 3 is PRE-CRYSTALLISATION.  It spans val 4.46 -> 0.08, crossing the
algebraic->statistical boundary.  The compiler's own rejected hypotheses
(H3, H4, H9) show that Newton, Lanczos, and floor-gradient alignment ALL fail
above val ~0.3 because the landscape is not yet quadratic (negative Hessian
eigenvalues; direction changes at Stokes crossings).  Anything that works in
Phase 4/5 (near-floor, locally quadratic, lambda_cos meaningful) has NO licence
in Phase 3 without separate proof.  So we measure Phase 3 on its own terms and
we measure it BY REGIME.

WHAT IS MEASURED (all causal transplants, no fitting)
-----------------------------------------------------
A. SUBSPACE ABLATION.  For each parameter group G in
       {Emb, FF, W_K, W_Q, W_V, W_O, LayerNorm}
   transplant ONLY G's floor value into the MF-pole model and measure val.
   This says exactly how many nats each subspace buys ON ITS OWN.
   Also the complement (everything EXCEPT G) to expose interactions.

B. LINEAR PARAMETERISABILITY, PER SUBSPACE.  For each group, sweep the
   straight line  theta_G(lam) = lam*MF + (1-lam)*floor  and ask whether
   val is smooth/monotone/polynomial along it.  A group can carry the descent
   yet still not be linearly parameterisable -- these are different questions
   and we report them separately.

C. REGIME SPLIT.  Every measurement is repeated from three Phase-3 waypoints:
       early  (val ~4)   -- algebraic, pre-crystallisation
       mid    (val ~1)   -- crossing
       late   (val ~0.2) -- statistical, near-crystallised
   If a subspace only becomes linearly parameterisable LATE, that is the
   quantitative statement of "Phase 4/5 methods need constraints to qualify
   for Phase 3".

D. GEOMETRY CO-MEASUREMENT.  E (strip energy), phi_k, Phi_cl and tau are
   logged at every transplant, so we can see which subspace moves the LOSS and
   which moves the GEOMETRY.  The expectation (to be tested, not assumed) is
   that these are DIFFERENT subspaces -- and if so, the flat-family structure
   that licenses Snapper lives where the loss does not.

OUTPUT
    phase3_ablation.json / .csv
    phase3_ablation.png   (nats-bought per subspace, per regime; linearity)
"""

import argparse
import csv
import json
import math

import numpy as np
import torch


# ----------------------------------------------------------- param groups
def group_of(name):
    """Assign a parameter to a subspace group.

    ORDER MATTERS.  Two traps, both caught in testing:
      * `head.weight` is TIED to `te.weight` in this model
        (self.head.weight = self.te.weight).  It must land in Emb, or a
        transplant would try to split a tied tensor.
      * `blocks.k.ff.n.{weight,bias}` are LayerNorms living INSIDE the ff
        module.  A naive `.ff.` check swallows them into FF.  LayerNorm is
        checked FIRST.
    """
    n = name.lower()
    # LayerNorm first -- it hides inside both attn and ff submodules
    if (".ln." in n or ".n." in n or n.startswith("ln_f")
            or n.endswith(".ln.weight") or n.endswith(".ln.bias")):
        return "LayerNorm"
    # tied embedding / head
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


GROUPS = ["Emb", "FF", "W_K", "W_Q", "W_V", "W_O", "LayerNorm", "other"]


def snapshot(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def load_snapshot(model, snap):
    model.load_state_dict({k: v.clone() for k, v in snap.items()})


def transplant(model, base, donor, groups, lam=0.0):
    """Set model = base everywhere, EXCEPT for parameters in `groups`, which
    become  lam*base + (1-lam)*donor.   lam=0 -> full donor value (pure
    transplant); lam=1 -> unchanged base.  This is the causal test: does
    giving the model the FLOOR value of this subspace actually buy loss?"""
    sd = {}
    for k, v in base.items():
        if group_of(k) in groups and k in donor:
            sd[k] = (lam * v + (1.0 - lam) * donor[k]).clone()
        else:
            sd[k] = v.clone()
    model.load_state_dict(sd)


# ----------------------------------------------------------- geometry probes
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


def sheet_angles(model, n_layers):
    phis = []
    for k in range(n_layers - 1):
        Wk = model.blocks[k].attn.WK.weight.detach().cpu().double()
        Wk1 = model.blocks[k + 1].attn.WK.weight.detach().cpu().double()
        try:
            M = Wk1 @ torch.linalg.pinv(Wk)
            lam = torch.linalg.eigvals(M)
            dom = lam[lam.abs().argmax()]      # largest MODULUS
            phis.append(float(torch.angle(dom)))
        except Exception:
            phis.append(float("nan"))
    return phis


def phi_clean(phis, tol=0.15):
    return sum(1 for p in phis if not math.isnan(p)
               and min(abs(p), abs(abs(p) - math.pi)) < tol)


# ----------------------------------------------------------- linearity test
def linearity(lams, vals):
    """Is val linear/smooth/monotone along this straight line?

    Reports, WITHOUT assuming a model:
      spread        total nats moved (does this line go anywhere at all?)
      monotone      does val decrease all the way?
      r2_deg1..4    how well a polynomial of each degree fits
      saturates_at  the lowest degree after which R^2 stops improving
    A line can carry the descent and still be non-polynomial; these are
    reported separately so we never again confuse the two."""
    lams = np.asarray(lams, float)
    vals = np.asarray(vals, float)
    out = {"spread": float(vals.max() - vals.min()),
           "monotone": bool(np.all(np.diff(vals) <= 1e-6))}
    ss = np.sum((vals - vals.mean()) ** 2) + 1e-12
    r2 = {}
    for d in (1, 2, 3, 4, 5):
        c = np.polyfit(lams, vals, d)
        r2[d] = float(1 - np.sum((vals - np.polyval(c, lams)) ** 2) / ss)
    out["r2"] = r2
    sat = None
    for d in (1, 2, 3, 4):
        if r2[d + 1] - r2[d] < 1e-3 and r2[d] > 0.95:
            sat = d
            break
    out["saturates_at"] = sat
    return out


# ----------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase3-steps", type=int, default=150)
    ap.add_argument("--sweep-points", type=int, default=7)
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

    print("=" * 72)
    print("  PHASE 3 CAUSAL ABLATION -- full measurement, no hypothesis")
    print("=" * 72)
    print("  Phase 3 is PRE-CRYSTALLISATION (val 4.5 -> 0.08, crosses the")
    print("  algebraic->statistical boundary).  Methods valid in Phase 4/5")
    print("  (near-floor, quadratic) have NO licence here without proof.")
    print("  So: measure first.  Correlation is not causation.")

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

    # ---- the MF pole (start of Phase 3) ----
    mf = snapshot(model)
    v_mf = eval_val(model, n=10)
    E_mf = strip_energy(model, N)
    print(f"\n  MF pole:  val={v_mf:.4f}  E={E_mf:.3f}")

    # ---- run Phase 3, capturing WAYPOINTS (the regime split) ----
    print(f"\n-- running Phase 3 ({args.phase3_steps} CE), capturing waypoints --")
    waypoints = {}
    marks = {"early": int(args.phase3_steps * 0.15),
             "mid": int(args.phase3_steps * 0.45),
             "late": args.phase3_steps}
    load_snapshot(model, mf)
    opt = torch.optim.AdamW(model.parameters(), lr=LR5,
                            betas=(0.9, 0.95), weight_decay=0.1)
    for s in range(1, args.phase3_steps + 1):
        model.train()
        x, y = get_batch()
        _, l = model(x, y)
        opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        for nm, ms in marks.items():
            if s == ms:
                waypoints[nm] = snapshot(model)
                vv = eval_val(model, n=8)
                print(f"   waypoint {nm:5s} @ step {s:3d}:  val={vv:.4f}"
                      f"  E={strip_energy(model, N):.3f}")
    floor = waypoints["late"]
    load_snapshot(model, floor)
    v_floor = eval_val(model, n=10)
    E_floor = strip_energy(model, N)
    total_descent = v_mf - v_floor
    print(f"\n   Phase-3 endpoint: val={v_floor:.4f}  E={E_floor:.3f}")
    print(f"   TOTAL DESCENT TO EXPLAIN: {total_descent:.4f} nats")

    results = {"mf_val": v_mf, "floor_val": v_floor,
               "total_descent": total_descent, "regimes": {}}

    # ================================================================
    # A + B + D, per regime
    # ================================================================
    for regime, donor in [("early", waypoints["early"]),
                          ("mid", waypoints["mid"]),
                          ("late", floor)]:
        load_snapshot(model, donor)
        v_target = eval_val(model, n=8)
        print("\n" + "=" * 72)
        print(f"  REGIME: {regime.upper()}   (donor val = {v_target:.4f})")
        print("  Transplant each subspace's donor value into the MF pole.")
        print("=" * 72)
        print(f"  {'subspace':<11}{'val':>9}{'nats':>8}{'%desc':>8}"
              f"{'E':>9}{'Phi_cl':>8}  linear?")
        print("  " + "-" * 66)

        reg = {"donor_val": v_target, "subspaces": {}}
        span = max(v_mf - v_target, 1e-9)

        for grp in GROUPS:
            # --- A: pure transplant (lam = 0 -> full donor value) ---
            transplant(model, mf, donor, {grp}, lam=0.0)
            v = eval_val(model, n=8)
            E = strip_energy(model, N)
            pcl = phi_clean(sheet_angles(model, N))
            nats = v_mf - v
            pct = 100.0 * nats / span

            # --- B: sweep the straight line for THIS subspace ---
            lams, vals = [], []
            for lam in np.linspace(1.0, 0.0, args.sweep_points):
                transplant(model, mf, donor, {grp}, lam=float(lam))
                lams.append(float(lam))
                vals.append(float(eval_val(model, n=5)))
            lin = linearity(lams, vals)
            sat = lin["saturates_at"]
            lin_str = (f"deg{sat} R2={lin['r2'][sat]:.3f}" if sat
                       else f"NO (R2_4={lin['r2'][4]:.3f})")

            print(f"  {grp:<11}{v:>9.4f}{nats:>8.3f}{pct:>7.1f}%"
                  f"{E:>9.3f}{pcl:>6d}/5  {lin_str}")
            reg["subspaces"][grp] = {
                "transplant_val": v, "nats_bought": nats, "pct_of_descent": pct,
                "E": E, "phi_clean": pcl,
                "sweep": {"lambda": lams, "val": vals},
                "linearity": lin,
            }

        # --- complements: everything EXCEPT G (exposes interactions) ---
        print("\n  COMPLEMENT (everything EXCEPT the named subspace):")
        print(f"  {'except':<11}{'val':>9}{'nats':>8}")
        print("  " + "-" * 30)
        for grp in ["Emb", "FF", "W_K"]:
            others = set(GROUPS) - {grp}
            transplant(model, mf, donor, others, lam=0.0)
            v = eval_val(model, n=8)
            print(f"  {grp:<11}{v:>9.4f}{v_mf - v:>8.3f}")
            reg["subspaces"][grp]["complement_val"] = v
            reg["subspaces"][grp]["complement_nats"] = v_mf - v

        # --- ALL: sanity, transplanting everything must reproduce the donor ---
        transplant(model, mf, donor, set(GROUPS), lam=0.0)
        v_all = eval_val(model, n=8)
        print(f"\n  ALL subspaces transplanted -> val={v_all:.4f} "
              f"(donor={v_target:.4f}, err={abs(v_all - v_target):.4f})")
        reg["all_transplant_val"] = v_all
        results["regimes"][regime] = reg
        load_snapshot(model, mf)

    # ================================================================
    # VERDICT
    # ================================================================
    print("\n" + "=" * 72)
    print("  WHERE THE DESCENT LIVES")
    print("=" * 72)
    late = results["regimes"]["late"]["subspaces"]
    ranked = sorted(GROUPS, key=lambda g_: -late[g_]["nats_bought"])
    print(f"  {'subspace':<11}{'nats':>8}{'% of descent':>14}  linearly param.?")
    print("  " + "-" * 58)
    for grp in ranked:
        d = late[grp]
        sat = d["linearity"]["saturates_at"]
        print(f"  {grp:<11}{d['nats_bought']:>8.3f}{d['pct_of_descent']:>13.1f}%"
              f"  {'yes (deg %d)' % sat if sat else 'no'}")

    print("\n  GEOMETRY vs LOSS (does the loss live where the geometry does?):")
    for grp in ranked[:3]:
        d = late[grp]
        print(f"    {grp:<10} moves loss by {d['nats_bought']:6.3f} nats;  "
              f"E={d['E']:.3f} (MF {E_mf:.3f})  Phi_cl={d['phi_clean']}/5")
    print("\n  If the subspaces that move E/Phi are NOT the subspaces that move")
    print("  the loss, then the flat-family structure that licenses Snapper")
    print("  lives where the loss does not -- and the pencil argument cannot")
    print("  be transported into Phase 3 without a new constraint.")

    print("\n  REGIME DEPENDENCE (does linearity only appear LATE?):")
    print(f"  {'subspace':<11}{'early':>10}{'mid':>10}{'late':>10}")
    print("  " + "-" * 42)
    for grp in ranked[:4]:
        row = []
        for r in ("early", "mid", "late"):
            s = results["regimes"][r]["subspaces"][grp]["linearity"]["saturates_at"]
            row.append(f"deg{s}" if s else "no")
        print(f"  {grp:<11}{row[0]:>10}{row[1]:>10}{row[2]:>10}")
    print("\n  A subspace that is linear only in the LATE regime is exactly the")
    print("  'Phase 4/5 method needs constraints to qualify for Phase 3' case.")

    with open("phase3_ablation.json", "w") as f:
        json.dump(results, f, indent=2)
    with open("phase3_ablation.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["regime", "subspace", "transplant_val", "nats_bought",
                    "pct_of_descent", "E", "phi_clean", "saturates_at",
                    "sweep_spread", "monotone"])
        for r, rd in results["regimes"].items():
            for grp, d in rd["subspaces"].items():
                w.writerow([r, grp, f"{d['transplant_val']:.5f}",
                            f"{d['nats_bought']:.5f}",
                            f"{d['pct_of_descent']:.2f}", f"{d['E']:.4f}",
                            d["phi_clean"], d["linearity"]["saturates_at"],
                            f"{d['linearity']['spread']:.5f}",
                            d["linearity"]["monotone"]])
    print("\n  wrote phase3_ablation.json / .csv")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
        x = np.arange(len(GROUPS))
        for i, r in enumerate(("early", "mid", "late")):
            v = [results["regimes"][r]["subspaces"][gp]["nats_bought"]
                 for gp in GROUPS]
            ax[0].bar(x + (i - 1) * 0.27, v, width=0.27, label=r)
        ax[0].set_xticks(x); ax[0].set_xticklabels(GROUPS, rotation=30)
        ax[0].set_ylabel("nats bought by transplant")
        ax[0].axhline(total_descent, color="red", ls="--",
                      label=f"total descent {total_descent:.2f}")
        ax[0].set_title("A. Where does the descent live?\n"
                        "(causal transplant, not correlation)")
        ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

        for gp in ["Emb", "FF", "W_K"]:
            sw = results["regimes"]["late"]["subspaces"][gp]["sweep"]
            ax[1].plot(sw["lambda"], sw["val"], marker="o", label=gp)
        ax[1].invert_xaxis()
        ax[1].set_xlabel("λ  (1 = MF pole, 0 = floor value)")
        ax[1].set_ylabel("val"); ax[1].set_yscale("log")
        ax[1].set_title("B. Is the straight line usable?\n"
                        "(per-subspace pencil, LATE regime)")
        ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
        plt.suptitle("Phase 3 causal ablation: measure, then hypothesise",
                     fontsize=13, weight="bold")
        plt.tight_layout()
        plt.savefig("phase3_ablation.png", dpi=190)
        print("  wrote phase3_ablation.png")
    except Exception as e:
        print(f"  [plot skipped: {e}]")


if __name__ == "__main__":
    main()
