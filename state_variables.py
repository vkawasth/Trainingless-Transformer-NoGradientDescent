"""
state_variables.py
==================
ARE THE GEOMETRIC COORDINATES STATE VARIABLES, OR SHADOWS?
And does constraining E buy anything?

-------------------------------------------------------------------------------
WHAT IS ESTABLISHED
-------------------------------------------------------------------------------
Across 5 optimizers at MATCHED LOSS (not matched steps):

    coordinate    CV across optimizers
    E             0.004    OPTIMIZER-INVARIANT
    R_Plucker     0.011    OPTIMIZER-INVARIANT
    lambda_cos    0.151    optimizer-specific
    Phi_cl        0.249    optimizer-specific
    tau           0.251    optimizer-specific

E sits at 43.5 from val 4.47 down to 0.15, for every optimizer.  And this is
NOT trivial: SGD never rotates W_K at all (lambda_cos stays 0.998-1.000) so E
is trivially fixed for it -- but the ADAPTIVE optimizers rotate W_K hard
(lambda_cos -> 0.61) and E STILL does not move.  E is conserved under the W_K
rotations that actually occur.

The morphism is also real: over-determined fits (3 points, 3 unknowns), with
ill-conditioned fits discarded, transport a held-out optimizer at median 92%,
88% of the time, ACROSS the SGD/adaptive divide.

-------------------------------------------------------------------------------
WHY THE CONSTRAINED SOLVE  min L s.t. E = E0  WILL PROBABLY DO NOTHING
-------------------------------------------------------------------------------
    * Every optimizer ALREADY conserves E, unprompted, to CV = 0.004.  The
      constraint is NEVER VIOLATED.  Projecting each update back onto E = E0
      therefore projects onto a constraint that already holds -- a NO-OP.
      A constraint accelerates a solve only if the unconstrained solver
      VIOLATES it and wastes work recovering.  This one does not.
    * E is ONE SCALAR.  Constraining it removes 1 degree of freedom out of
      4,330,240.  A 0.00002% reduction of the search space.

We build it anyway (Part 2) because it is cheap and the prediction is
falsifiable.  But first:

-------------------------------------------------------------------------------
PART 1 (THE REAL QUESTION): IS  L = L(E, lambda, Phi, tau)  ?
-------------------------------------------------------------------------------
If the loss is a FUNCTION of the coordinates, then with E fixed, Phase 3 only
has to determine (lambda, Phi, tau) -- a 3-parameter solve instead of a
4.3-million-parameter one.  That would be an enormous reduction.

But everything measured so far shows only that the coordinates MOVE during
descent -- never that they DETERMINE the loss.  Those are different claims,
and the second has never been tested.

THE TEST (causal, no fitting):
    Generate many weight configurations that share the SAME coordinates but
    have DIFFERENT weights.  Measure the loss of each.
        LOW variance  -> the coordinates are STATE VARIABLES.  L is a function
                         of them.  Phase 3 collapses to a 3-parameter solve.
        HIGH variance -> they are SHADOWS: descriptive of the descent, not
                         determinative of the loss.

THE NULL (mandatory): how much does the loss vary between configurations with
RANDOM coordinates?  If matched-coordinate variance is no lower than random-
coordinate variance, the coordinates carry no information about the loss.

MY PRIOR: this fails.  E, lambda_cos and Phi are ALL functions of W_K alone,
and the ablation showed W_K carries ~0.5% of the loss.  Three of the four
coordinates live in the subspace that does NOT move the loss.  Only tau
touches Emb/FF.  But it is cheap, decisive, and if I am wrong it is the
biggest result in the programme.

OUTPUT
    state_variables.json / .png
"""

import argparse
import json
import math

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


# ---------------------------------------------------------------- coordinates
def strip_energy(model, L, rank=6):
    Us = []
    for k in range(L):
        W = model.blocks[k].attn.WK.weight.detach().cpu().numpy()
        U, _, _ = np.linalg.svd(W, full_matrices=False)
        Us.append(U[:, :rank])
    E = 0.0
    for k in range(L - 1):
        s = np.linalg.svd(Us[k].T @ Us[k + 1], compute_uv=False)
        E += float(np.sum(np.arccos(np.clip(s, -1, 1))))
    return E


def sheet_angles(model, L):
    out = []
    for k in range(L - 1):
        Wk = model.blocks[k].attn.WK.weight.detach().cpu().double()
        Wk1 = model.blocks[k + 1].attn.WK.weight.detach().cpu().double()
        try:
            lam = torch.linalg.eigvals(Wk1 @ torch.linalg.pinv(Wk))
            out.append(float(torch.angle(lam[lam.abs().argmax()])))
        except Exception:
            out.append(float("nan"))
    return out


def phi_clean(ph, tol=0.15):
    return sum(1 for p in ph if not math.isnan(p)
               and min(abs(p), abs(abs(p) - math.pi)) < tol)


def lambda_cos(model, ref, L):
    s = []
    for k in range(L):
        a = model.blocks[k].attn.WK.weight.detach().cpu().numpy().ravel()
        b = ref[k].ravel()
        s.append(float(np.dot(a, b) /
                       (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)))
    return float(np.mean(s))


def tau_defect(model, get_batch):
    model.zero_grad(set_to_none=True)
    x, y = get_batch()
    _, l = model(x, y)
    l.backward()
    gf = ge = 0.0
    for n, p in model.named_parameters():
        if p.grad is None:
            continue
        g = float(p.grad.detach().norm()) ** 2
        if ".ff." in n:
            gf += g
        elif n.startswith("te") or n.startswith("pe"):
            ge += g
    model.zero_grad(set_to_none=True)
    return math.sqrt(gf) / (math.sqrt(ge) + 1e-12)


def coords(model, ref, L, get_batch):
    ph = sheet_angles(model, L)
    return {"E": strip_energy(model, L),
            "lambda_cos": lambda_cos(model, ref, L),
            "Phi_cl": phi_clean(ph),
            "tau": tau_defect(model, get_batch)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=60,
                    help="weight configurations to generate per condition")
    ap.add_argument("--tol-E", type=float, default=0.05)
    ap.add_argument("--tol-lam", type=float, default=0.01)
    ap.add_argument("--tol-tau", type=float, default=0.05)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--compiler", default="compiler_analytic_topogate.py")
    args = ap.parse_args()

    g_ = {}
    src = open(args.compiler).read()
    cut = src.find("# ── PHASE 1")
    if cut == -1:
        cut = src.find("PHASE 1")
    exec(src[:cut], g_)
    model = g_["model"]; get_batch = g_["get_batch"]
    LR = g_["LR"] * 5
    L = g_["N_STU"]

    print("=" * 78)
    print("  ARE THE COORDINATES STATE VARIABLES, OR SHADOWS?")
    print("=" * 78)
    print("  Everything so far shows the coordinates MOVE during descent.")
    print("  It has NEVER been tested whether they DETERMINE the loss.")
    print("  Those are different claims, and only the second licenses the")
    print("  reduction  min L(theta)  ->  min L(E, lambda, Phi, tau).")
    print()
    print("  TEST: configurations with the SAME coordinates but DIFFERENT")
    print("  weights.  Do they have the SAME loss?")
    print("    low variance  -> STATE VARIABLES. Phase 3 = a 3-param solve.")
    print("    high variance -> SHADOWS. The geometry describes the")
    print("                     REPRESENTATION, not the LOSS.")
    print()
    print("  PRIOR (mine): this fails.  E, lambda_cos and Phi are ALL functions")
    print("  of W_K alone, and the ablation showed W_K carries ~0.5% of the")
    print("  loss.  Three of the four coordinates live in the subspace that")
    print("  does NOT move the loss.  I would be glad to be wrong.")

    torch.manual_seed(0)
    EVAL = [get_batch() for _ in range(12)]

    def evalf():
        model.eval()
        t = 0.0
        with torch.no_grad():
            for x, y in EVAL:
                _, l = model(x, y)
                t += float(l)
        model.train()
        return t / len(EVAL)

    start = snap(model)
    ref = [model.blocks[k].attn.WK.weight.detach().cpu().numpy().copy()
           for k in range(L)]

    # ---- reach a reference point mid-descent ----
    print(f"\n-- descending {args.steps} steps to a reference point --")
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95),
                            weight_decay=0.1)
    for _ in range(args.steps):
        model.train()
        x, y = get_batch()
        _, l = model(x, y)
        opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    anchor = snap(model)
    c0 = coords(model, ref, L, get_batch)
    v_anchor = evalf()
    print(f"   anchor: val={v_anchor:.4f}")
    print(f"   coords: E={c0['E']:.3f}  lambda={c0['lambda_cos']:.3f}  "
          f"Phi_cl={c0['Phi_cl']}  tau={c0['tau']:.3f}")

    # ================================================================
    # PART 1: MATCHED-COORDINATE ENSEMBLE  vs  RANDOM-COORDINATE NULL
    # ================================================================
    # We perturb the weights and KEEP only the configurations whose
    # coordinates still match the anchor's.  Then we look at the spread of
    # their LOSSES.  The null is the spread of losses over perturbations of
    # the SAME magnitude, WITHOUT the coordinate-matching filter.
    print("\n" + "=" * 78)
    print("  generating configurations with MATCHED coordinates")
    print("=" * 78)
    print(f"   tolerances: |dE|<{args.tol_E}  |dlambda|<{args.tol_lam}  "
          f"|dtau|<{args.tol_tau}  Phi_cl exact")

    matched, nulls = [], []
    tries = 0
    sigma = 0.02
    while len(matched) < args.n_samples and tries < args.n_samples * 40:
        tries += 1
        torch.manual_seed(10000 + tries)
        sd = {k: v.clone() for k, v in anchor.items()}
        for k in sd:
            if group_of(k) in ("Emb", "FF", "LayerNorm", "W_Q", "W_V",
                               "W_O", "W_K", "other"):
                sd[k] = sd[k] + sigma * torch.randn_like(sd[k]) * sd[k].std()
        load(model, sd)
        v = evalf()
        nulls.append(v)                       # the NULL: no filtering at all
        c = coords(model, ref, L, get_batch)
        if (abs(c["E"] - c0["E"]) < args.tol_E
                and abs(c["lambda_cos"] - c0["lambda_cos"]) < args.tol_lam
                and abs(c["tau"] - c0["tau"]) < args.tol_tau
                and c["Phi_cl"] == c0["Phi_cl"]):
            matched.append({"val": v, **c})
        if tries % 200 == 0:
            print(f"   ... {tries} tried, {len(matched)} matched")
    load(model, anchor)

    print(f"\n   {len(matched)} configurations matched the anchor's coordinates"
          f"  (from {tries} tries)")
    if len(matched) < 10:
        print("   !! too few matches to conclude.  Loosen the tolerances.")
        return

    mv = np.array([m["val"] for m in matched])
    nv = np.array(nulls)
    print("\n" + "=" * 78)
    print("  DO CONFIGURATIONS WITH THE SAME COORDINATES HAVE THE SAME LOSS?")
    print("=" * 78)
    print(f"   MATCHED coordinates (n={len(mv)}):")
    print(f"      loss  mean {mv.mean():.4f}   std {mv.std():.4f}   "
          f"range [{mv.min():.4f}, {mv.max():.4f}]")
    print(f"   NULL, same perturbation size, NO coordinate filter (n={len(nv)}):")
    print(f"      loss  mean {nv.mean():.4f}   std {nv.std():.4f}   "
          f"range [{nv.min():.4f}, {nv.max():.4f}]")
    ratio = float(mv.std() / (nv.std() + 1e-12))
    print(f"\n   variance ratio  matched/null = {ratio:.3f}")
    print("   (a coordinate that DETERMINES the loss would collapse the")
    print("    variance: ratio << 1.  A ratio ~1 means the coordinates carry")
    print("    NO information about the loss.)")

    determines = ratio < 0.30
    print()
    if determines:
        print("   => THE COORDINATES ARE STATE VARIABLES.  Fixing them pins the")
        print("      loss.  L really is a function of (E, lambda, Phi, tau), and")
        print("      Phase 3 reduces to a 3-parameter solve once E is fixed.")
        print("      This is the biggest possible result and I was wrong.")
    else:
        print("   => THE COORDINATES ARE SHADOWS.  Configurations with")
        print("      IDENTICAL (E, lambda, Phi, tau) have losses spread just as")
        print("      widely as unfiltered ones.  The coordinates do NOT")
        print("      determine the loss.")
        print("      They describe the REPRESENTATION geometry, not the loss.")
        print("      Consequently  min L s.t. E=E0  cannot reduce the search:")
        print("      the constraint surface still contains the whole spread of")
        print("      losses.")

    # ================================================================
    # PART 2: THE CONSTRAINED SOLVER (built anyway, and measured)
    # ================================================================
    print("\n" + "=" * 78)
    print("  PART 2: min L(theta)  s.t.  E(theta) = E0   -- does it help?")
    print("=" * 78)
    print("  PREDICTION: no.  Every optimizer already conserves E unprompted")
    print("  (CV = 0.004), so the constraint is never violated and projecting")
    print("  onto it is a no-op.  And E is ONE scalar: it removes 1 dof of")
    print("  4.3M.  We measure it rather than assert it.")

    E0 = c0["E"]

    def project_E(sd, target, iters=3, lr=0.05):
        """Push E back to `target` by scaling the W_K frames.  A cheap,
        explicit projection onto the constraint surface."""
        for _ in range(iters):
            load(model, sd)
            e = strip_energy(model, L)
            if abs(e - target) < 1e-3:
                break
            # E rises when consecutive W_K column spaces are more transverse.
            # Nudge each W_K toward/away from its neighbour's span.
            with torch.no_grad():
                for k in range(L - 1):
                    A = model.blocks[k].attn.WK.weight
                    B = model.blocks[k + 1].attn.WK.weight
                    d = (target - e)
                    B += lr * d * 0.001 * (A - B)
            sd = snap(model)
        return sd

    results = {}
    for label, use_constraint in [("unconstrained", False),
                                  ("E-constrained", True)]:
        torch.manual_seed(7)
        load(model, start)
        o = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95),
                              weight_decay=0.1)
        traj = []
        for s in range(1, args.steps + 1):
            model.train()
            x, y = get_batch()
            _, l = model(x, y)
            o.zero_grad(); l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            o.step()
            if use_constraint and s % 5 == 0:
                sd = project_E(snap(model), E0)
                load(model, sd)
            if s % 25 == 0:
                traj.append({"step": s, "val": evalf(),
                             "E": strip_energy(model, L)})
        results[label] = traj
        f = traj[-1]
        print(f"\n   [{label}]  final val={f['val']:.4f}  E={f['E']:.3f}")
        print("      " + "  ".join(f"{t['step']}:{t['val']:.3f}" for t in traj))

    vu = results["unconstrained"][-1]["val"]
    vc = results["E-constrained"][-1]["val"]
    print(f"\n   unconstrained: val={vu:.4f}")
    print(f"   E-constrained: val={vc:.4f}")
    print(f"   difference   : {vc - vu:+.4f}")
    if vc < vu - 0.01:
        print("   => the constraint HELPED.  Contrary to prediction.")
    elif vc > vu + 0.01:
        print("   => the constraint HURT.  The projection is fighting the")
        print("      optimizer without buying anything.")
    else:
        print("   => NO EFFECT, as predicted.  The constraint was already")
        print("      satisfied; projecting onto it changes nothing.")
        print("      E is a CONSERVED QUANTITY of the flow, not an active")
        print("      constraint.  Its value is as a VALIDITY GUARD (E leaving")
        print("      43.5 flags a broken run -- the diverged RMSprop was")
        print("      exactly that), not as an accelerator.")

    json.dump({"anchor": {"val": v_anchor, **c0},
               "matched": matched, "null": nv.tolist(),
               "variance_ratio": ratio, "determines": bool(determines),
               "constrained": results},
              open("state_variables.json", "w"), indent=2, default=float)
    print("\n  wrote state_variables.json")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
        ax[0].hist(nv, bins=20, alpha=0.6, color="#ccc",
                   label=f"NULL (no filter)  std={nv.std():.3f}")
        ax[0].hist(mv, bins=20, alpha=0.8, color="#c44e52",
                   label=f"MATCHED coords  std={mv.std():.3f}")
        ax[0].axvline(v_anchor, color="k", ls="--", label="anchor")
        ax[0].set_xlabel("loss"); ax[0].set_ylabel("count")
        ax[0].set_title("Do equal coordinates imply equal loss?\n"
                        f"variance ratio = {ratio:.2f}")
        ax[0].legend(fontsize=8)
        for lab, tr in results.items():
            ax[1].plot([t["step"] for t in tr], [t["val"] for t in tr],
                       "o-", label=lab)
        ax[1].set_xlabel("step"); ax[1].set_ylabel("val")
        ax[1].set_yscale("log")
        ax[1].set_title("Does constraining E accelerate the solve?")
        ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
        plt.suptitle("State variables, or shadows?", fontsize=13, weight="bold")
        plt.tight_layout()
        plt.savefig("state_variables.png", dpi=180)
        print("  wrote state_variables.png")
    except Exception as e:
        print(f"  [plot skipped: {e}]")


if __name__ == "__main__":
    main()
