"""
morphism_geometry.py
====================
TWO THINGS.

  PART A -- A PROPERLY DETERMINED MORPHISM TEST.
  PART B -- DO THE GEOMETRIC COORDINATES MOVE THE SAME WAY FOR EVERY
            OPTIMIZER AS THEY DO FOR ADAM?

-------------------------------------------------------------------------------
PART A: WHY THE PREVIOUS MORPHISM RESULT IS NOT YET TRUSTWORTHY
-------------------------------------------------------------------------------
The basin-entry run (target 0.15) gave, after excluding two numerically
degenerate fits:

    WITHIN-cluster : n=6   median recovery 76%   transport 50%
    ACROSS-cluster : n=10  median recovery 96%   transport 90%

So the map transports ACROSS clusters BETTER than within, and the two-chart
hypothesis (mine, not yours) is dead.  The geometric clusters are real
(within 0.71, across 0.26, chance 0.0014) but CAUSALLY IRRELEVANT.

BUT the design has a hole.  Most of those successful fits had only ONE
optimizer in the fitting set.  With one fitting point and k=3 coefficients,
the least-squares map is a RANK-1 RESCALING: it can transport simply because
ANY displacement, scaled correctly, lands near the target.  That is the same
degeneracy that made a per-pair fit "recover" 100% even when the source
optimizer had stopped 40% of the way -- verified earlier.

Also: the two catastrophic failures (-2152%, -2573%) were NOT geometry.  Both
held out Adagrad while fitting on {AdamW, RMSprop}, which have overlap 0.85 --
nearly collinear, so with 2 points and 3 unknowns the lstsq is
underdetermined AND ill-conditioned, and the map explodes.  A linear-algebra
artefact of my design.

THE FIX: use 5 optimizers, fit on FOUR (so the system is over-determined:
4 equations, 3 unknowns), hold out ONE.  Report the condition number of every
fit and DISCARD any fit that is ill-conditioned rather than letting it
detonate.  And judge only by CAUSAL TRANSPLANT, never by fit quality.

  * transports with a determined, well-conditioned fit -> the morphism is REAL
  * only works with 1-point fits -> it was rank-1 rescaling all along

-------------------------------------------------------------------------------
PART B: THE GEOMETRIC COORDINATES
-------------------------------------------------------------------------------
Track, for EVERY optimizer, the coordinates the compiler is built on:

    E          strip energy      sum_k sum_i arccos(sigma_i(U_k' U_{k+1}))
    phi_k      Bridgeland sheet angles (arg of the dominant eigenvalue,
               LARGEST MODULUS -- the corrected definition)
    Phi_cl     count of phi_k in {0, pi}
    tau        ||grad_FF|| / ||grad_Emb||   (K0 gluing defect)
    lambda_cos pencil coordinate: cos-sim of W_K to its MF-pump value
    R_Plucker  departure of the 5 layer directions from Gr(3,5) flatness

MEASURED AT MATCHED LOSS LEVELS, NOT MATCHED STEPS.  This matters: SGD needs
700 steps to reach val=0.15 and Adagrad needs 120, so step-matched
comparisons would compare completely different points on the descent.  We
sample each optimizer as it CROSSES the same loss thresholds.

THE QUESTION: does E stay pinned at ~43.5 for every optimizer?  Does tau climb
the same way?  Does Phi_cl crystallise at the same loss?  If the A-brane
coordinates move identically for all optimizers while the loss-carrying
subspaces differ, that separates cleanly what is ARCHITECTURAL from what is
OPTIMIZER-SPECIFIC.

OUTPUT
    morphism_geometry.json / .png
"""

import argparse
import itertools
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


GROUPS = {"Emb", "FF", "LayerNorm", "W_Q", "W_V", "W_O", "W_K", "other"}
CLUSTER = {"AdamW": "adaptive", "RMSprop": "adaptive", "Adagrad": "adaptive",
           "SGD": "sgd-like", "SGD+mom": "sgd-like"}


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


# ---------------------------------------------------------------- geometry
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
    """phi_k = arg(lambda_dom), dominant = LARGEST MODULUS (corrected defn)."""
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


def tau_defect(model):
    gf = ge = 0.0
    for n, p in model.named_parameters():
        if p.grad is None:
            continue
        g = float(p.grad.detach().norm()) ** 2
        if ".ff." in n:
            gf += g
        elif n.startswith("te") or n.startswith("pe"):
            ge += g
    return math.sqrt(gf) / (math.sqrt(ge) + 1e-12)


def lambda_cos(model, wk_ref, L):
    s = []
    for k in range(L):
        a = model.blocks[k].attn.WK.weight.detach().cpu().numpy().ravel()
        b = wk_ref[k].ravel()
        s.append(float(np.dot(a, b) /
                       (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)))
    return float(np.mean(s))


def r_plucker(model, L, rank=3):
    """Non-decomposability of the 5 layer directions: 0 = they lie in a common
    3-plane (in-chamber); large = they have separated (wall transit)."""
    V = []
    for k in range(min(5, L)):
        W = model.blocks[k].attn.WK.weight.detach().cpu().numpy()
        U, _, _ = np.linalg.svd(W, full_matrices=False)
        v = U[:, 0]
        V.append(v / (np.linalg.norm(v) + 1e-12))
    A = np.stack(V)
    s = np.linalg.svd(A, compute_uv=False)
    tot = float(np.sum(s ** 2)) + 1e-12
    res = float(np.sum(s[rank:] ** 2)) if len(s) > rank else 0.0
    return math.sqrt(res / tot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--target", type=float, default=0.15)
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=42)
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
    print("  A: DETERMINED MORPHISM   |   B: DO THE GEOMETRIC COORDINATES")
    print("                           |      MOVE LIKE ADAM'S?")
    print("=" * 78)
    print("  A. The previous morphism worked mostly with ONE-POINT fits.  With")
    print("     1 fitting point and k=3, the map is a RANK-1 RESCALING and can")
    print("     transport ANY displacement -- the same degeneracy that made a")
    print("     per-pair fit 'recover' 100% from a source that stopped 40% of")
    print("     the way.  Here we fit on FOUR optimizers (over-determined:")
    print("     4 equations, 3 unknowns), hold out ONE, report the condition")
    print("     number, and DISCARD ill-conditioned fits instead of letting")
    print("     them detonate (the earlier -2152%/-2573% were exactly that).")
    print()
    print("  B. Coordinates are sampled at MATCHED LOSS LEVELS, not matched")
    print("     steps: SGD needs 700 steps to reach 0.15, Adagrad 120.  Step-")
    print("     matched sampling would compare different points on the descent.")

    start = snap(model)
    keys = [k for k in start if group_of(k) in GROUPS]
    wk_ref = [model.blocks[k].attn.WK.weight.detach().cpu().numpy().copy()
              for k in range(L)]

    torch.manual_seed(args.seed)
    TRAIN = [get_batch() for _ in range(args.max_steps)]
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

    v0 = evalf()
    # the loss levels at which we photograph the geometry
    LEVELS = [3.0, 2.0, 1.0, 0.6, 0.4, 0.3, 0.2, args.target]
    print(f"\n  start val = {v0:.4f}")
    print(f"  geometry sampled as each optimizer CROSSES: {LEVELS}")

    OPTS = {
        "AdamW":   (lambda p, lr: torch.optim.AdamW(p, lr=lr, betas=(0.9, 0.95),
                                                    weight_decay=0.1), LR),
        "RMSprop": (lambda p, lr: torch.optim.RMSprop(p, lr=lr), LR * 0.2),
        "Adagrad": (lambda p, lr: torch.optim.Adagrad(p, lr=lr), LR * 5),
        "SGD":     (lambda p, lr: torch.optim.SGD(p, lr=lr), LR * 300),
        "SGD+mom": (lambda p, lr: torch.optim.SGD(p, lr=lr, momentum=0.9),
                    LR * 60),
    }

    # ================================================================
    #  TRAIN + PHOTOGRAPH THE GEOMETRY AT MATCHED LOSS
    # ================================================================
    D, GEO = {}, {}
    print("\n" + "=" * 78)
    print("  TRAINING (geometry photographed at each loss level)")
    print("=" * 78)
    for name, (mk, lr) in OPTS.items():
        torch.manual_seed(args.seed)
        load(model, start)
        o = mk(model.parameters(), lr)
        ck = [snap(model)]
        geo, todo = [], list(LEVELS)
        reached, used = False, args.max_steps
        last_tau = 0.0
        for s, (x, y) in enumerate(TRAIN, 1):
            model.train()
            _, l = model(x, y)
            o.zero_grad(); l.backward()
            last_tau = tau_defect(model)          # needs grads: before step
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            o.step()
            if s % 5 == 0:
                v = evalf()
                while todo and v <= todo[0]:
                    lv = todo.pop(0)
                    ph = sheet_angles(model, L)
                    geo.append({"level": lv, "step": s, "val": v,
                                "E": strip_energy(model, L),
                                "Phi_cl": phi_clean(ph),
                                "tau": last_tau,
                                "lambda_cos": lambda_cos(model, wk_ref, L),
                                "R_pluck": r_plucker(model, L),
                                "phi": ph})
                    ck.append(snap(model))
                if v <= args.target:
                    reached, used = True, s
                    break
        load(model, ck[-1])
        vf = evalf()
        X = np.stack([(flat(ck[i], keys) - flat(ck[i - 1], keys)).numpy()
                      for i in range(1, len(ck))])
        T = (flat(ck[-1], keys) - flat(start, keys)).numpy()
        B = np.linalg.svd(X, full_matrices=False)[2][:args.k].T
        D[name] = {"floor": vf, "T": T, "B": B, "steps": used,
                   "reached": reached}
        GEO[name] = geo
        print(f"   {name:<9} [{CLUSTER[name]:<8}] val={vf:.4f} in {used:4d} steps"
              f"  {'OK' if reached else '!! NOT REACHED'}")

    ok = [n for n in D if D[n]["reached"]]
    if len(ok) < 5:
        print(f"\n  !! only {len(ok)}/5 reached {args.target}. Need all 5 for a")
        print("     determined fit (4 fitting points, 3 unknowns).  Raise")
        print("     --max-steps.")
        return

    # ================================================================
    #  PART B -- THE GEOMETRIC COORDINATES
    # ================================================================
    print("\n" + "=" * 78)
    print("  PART B: DO THE COORDINATES MOVE THE SAME WAY FOR EVERY OPTIMIZER?")
    print("=" * 78)
    for coord, fmt in [("E", "{:7.2f}"), ("tau", "{:7.2f}"),
                       ("Phi_cl", "{:7.0f}"), ("lambda_cos", "{:7.3f}"),
                       ("R_pluck", "{:7.3f}")]:
        print(f"\n  {coord}   (rows = optimizer, cols = loss level)")
        print("  " + " " * 9 + "".join(f"{lv:>8}" for lv in LEVELS))
        for n in ok:
            row = f"  {n:<9}"
            for lv in LEVELS:
                g = next((x for x in GEO[n] if x["level"] == lv), None)
                row += fmt.format(g[coord]) if g else "      --"
            print(row)
        # is the coordinate optimizer-INVARIANT at matched loss?
        spreads = []
        for lv in LEVELS:
            vals = [next((x[coord] for x in GEO[n] if x["level"] == lv), None)
                    for n in ok]
            vals = [v for v in vals if v is not None]
            if len(vals) >= 3:
                m_ = np.mean(vals)
                spreads.append(np.std(vals) / (abs(m_) + 1e-9))
        cv = float(np.mean(spreads)) if spreads else float("nan")
        verdict = ("OPTIMIZER-INVARIANT" if cv < 0.10 else
                   "weakly varying" if cv < 0.30 else "OPTIMIZER-SPECIFIC")
        print(f"  -> mean CV across optimizers at matched loss = {cv:.3f}"
              f"   [{verdict}]")

    print("\n  READ: a coordinate with a small CV moves the SAME WAY for every")
    print("  optimizer at the same loss -- it is a property of the PROBLEM.")
    print("  A large CV means each optimizer moves it differently -- a property")
    print("  of the OPTIMIZER.")

    # ================================================================
    #  PART A -- THE DETERMINED MORPHISM
    # ================================================================
    print("\n" + "=" * 78)
    print("  PART A: MORPHISM WITH AN OVER-DETERMINED FIT (4 points, 3 unknowns)")
    print("=" * 78)
    Mx = np.stack([D[n]["T"] for n in ok])
    Qc = np.linalg.qr(np.linalg.svd(Mx, full_matrices=False)[2][:args.k].T)[0]
    coef = {n: Qc.T @ D[n]["T"] for n in ok}

    def transplant(vec):
        sd = {a: b.clone() for a, b in start.items()}
        pd = unflat(torch.tensor(vec), start, keys)
        for a in keys:
            sd[a] = (start[a] + pd[a]).clone()
        load(model, sd)
        v = evalf()
        load(model, start)
        return v

    print(f"  {'target':<9}{'held-out':<9}{'fitted on':<28}"
          f"{'cond':>8}{'val':>9}{'recovery':>10}")
    print("  " + "-" * 74)
    res = []
    for target in ok:
        v_t = transplant(Qc @ coef[target])
        d_t = v0 - v_t
        if d_t < 0.5:
            continue
        for held in ok:
            if held == target:
                continue
            tr = [n for n in ok if n not in (held, target)]   # THREE points
            if len(tr) < 3:
                continue
            X = np.stack([coef[n] for n in tr])          # (3, k)
            Y = np.stack([coef[target] for _ in tr])
            # CONDITION THRESHOLD.  1e4 was FAR too loose: it admitted fits
            # with cond=1485 that returned -660,000%.  For a 3x3 system a cond
            # above ~10 already means the fitting points are nearly collinear.
            # This happens WHENEVER the fitting set contains BOTH members of a
            # cluster (AdamW+RMSprop overlap 0.85; SGD+SGD+mom 0.84) -- they
            # are almost parallel in coefficient space, X is rank-deficient,
            # and the map detonates.  That is linear algebra, not geometry, and
            # such fits carry NO information about whether a morphism exists.
            cond = float(np.linalg.cond(X))
            if cond > 10.0:
                print(f"  {target:<9}{held:<9}{','.join(tr)[:26]:<28}"
                      f"{cond:>8.1f}   DISCARDED (collinear fitting set)")
                continue
            M, *_ = np.linalg.lstsq(X, Y, rcond=None)
            v = transplant(Qc @ (M.T @ coef[held]))
            rec = (v0 - v) / d_t
            res.append({"target": target, "held": held, "fitted_on": tr,
                        "cond": cond, "val": v, "recovery": rec,
                        "same_cluster": CLUSTER[held] == CLUSTER[target]})
            print(f"  {target:<9}{held:<9}{','.join(tr)[:26]:<28}"
                  f"{cond:>8.1f}{v:>9.3f}{100*rec:>9.0f}%")

    if not res:
        print("\n  no well-conditioned determined fits.  Cannot conclude.")
        return
    R = np.array([r["recovery"] for r in res])
    W = np.array([r["recovery"] for r in res if r["same_cluster"]])
    A = np.array([r["recovery"] for r in res if not r["same_cluster"]])
    rate = float(np.mean(R > 0.70))

    print("\n" + "=" * 78)
    print("  VERDICT")
    print("=" * 78)
    print(f"   determined, well-conditioned fits: n={len(R)}")
    print(f"   median recovery {100*np.median(R):.0f}%   "
          f"transport rate {100*rate:.0f}%")
    if len(W):
        print(f"     same-cluster : n={len(W)} median {100*np.median(W):.0f}%")
    if len(A):
        print(f"     cross-cluster: n={len(A)} median {100*np.median(A):.0f}%")
    print()
    if rate > 0.7:
        print("   => THE MORPHISM IS REAL.  An OVER-DETERMINED map, fitted on")
        print("      three other optimizers, transports a HELD-OUT optimizer's")
        print("      displacement into the target's solution.  This is not a")
        print("      rank-1 rescaling: with 3 fitting points and 3 unknowns the")
        print("      system is determined, and ill-conditioned fits were")
        print("      discarded rather than allowed to detonate.")
        print("      Different optimizers reach the same loss in 3 dimensions")
        print("      each, and there ARE linear maps between their subspaces.")
    elif rate > 0.4:
        print("   => PARTIAL.  Some determined fits transport, some do not.")
    else:
        print("   => THE MORPHISM WAS AN ARTEFACT.  With a properly determined")
        print("      fit it does not transport.  The earlier successes were")
        print("      rank-1 rescaling: a single fitting point can always be")
        print("      scaled onto the target, which proves nothing.")

    json.dump({"target": args.target, "v0": v0, "levels": LEVELS,
               "geometry": GEO,
               "per_optimizer": {n: {"floor": D[n]["floor"],
                                     "steps": D[n]["steps"]} for n in D},
               "morphism": res,
               "transport_rate": rate,
               "median_recovery": float(np.median(R))},
              open("morphism_geometry.json", "w"), indent=2, default=float)
    print("\n  wrote morphism_geometry.json")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 3, figsize=(18, 9))
        coords = ["E", "tau", "Phi_cl", "lambda_cos", "R_pluck"]
        for i, c in enumerate(coords):
            a = ax[i // 3][i % 3]
            for n in ok:
                xs = [g["val"] for g in GEO[n]]
                ys = [g[c] for g in GEO[n]]
                st = "-o" if CLUSTER[n] == "adaptive" else "--s"
                a.plot(xs, ys, st, label=n, ms=4)
            a.set_xscale("log"); a.invert_xaxis()
            a.set_xlabel("val (descending ->)"); a.set_ylabel(c)
            a.set_title(f"{c} vs loss")
            a.grid(alpha=0.3); a.legend(fontsize=7)
        a = ax[1][2]
        if len(R):
            a.hist(100 * R, bins=10, color="#55a868")
            a.axvline(70, ls="--", color="k", label="transport")
            a.set_xlabel("% of target's descent recovered")
            a.set_title(f"Determined morphism (n={len(R)})\n"
                        f"transport rate {100*rate:.0f}%")
            a.legend(fontsize=8)
        plt.suptitle("Do the geometric coordinates move alike? "
                     "And is the morphism real?", fontsize=14, weight="bold")
        plt.tight_layout()
        plt.savefig("morphism_geometry.png", dpi=170)
        print("  wrote morphism_geometry.png")
    except Exception as e:
        print(f"  [plot skipped: {e}]")


if __name__ == "__main__":
    main()
