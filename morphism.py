"""
morphism.py
===========
TWO QUESTIONS, ONE SCRIPT.

THE PREMISE THAT NEEDS CHECKING FIRST
-------------------------------------
The argument for a functor between optimizer subspaces assumes all optimizers
LAND IN THE SAME BASIN.  In the last controlled run they did NOT:

    optimizer   floor
    AdamW       0.0813
    Adagrad     0.0918
    RMSprop     0.4118
    SGD+mom     1.2978
    SGD         1.7677     <- 22x AdamW's loss, same init, same batches

SGD is not in the same basin by a different road; it is in a materially
different place.  So the earlier result -- "SGD's endpoint overlaps the
adaptive shared basis by only 36%, and that carries only 20% of its loss" --
may simply reflect THAT SGD IS ONLY PARTLY DONE.  That is a confound, and it
must be removed before any morphism question is meaningful.

  PART 1: EQUAL-LOSS TRAINING.  Train every optimizer until it reaches THE
          SAME TARGET LOSS (not the same step count).  Then the endpoints are
          comparable objects and the overlap question can be re-asked honestly.

THE MORPHISM QUESTION
---------------------
If every optimizer compresses to ~3 dimensions and solves the same problem,
there should be a map  Phi : S_i -> S_j  transporting one optimizer's solution
into another's coordinates.  A functor, in the informal sense.

  PART 2: SEARCH FOR IT -- WITH A NON-CIRCULAR DESIGN.

  !! THE TRAP.  A linear map between two 3-dim subspaces ALWAYS fits: 9
     unknowns, 3 constraints.  Fitting M on the very pair you then test is
     circular and produces a guaranteed fake positive.  VERIFIED: a per-pair
     least-squares map "recovers" 100% of the target's loss even when the
     source optimizer stopped 40% of the way to the solution.  It simply
     rescales one displacement into the other.

  THE FIX: LEAVE-ONE-OUT.  Fit M on the OTHER optimizers, then apply it to a
     HELD-OUT one.  VERIFIED non-degenerate: when all optimizers solve the
     task, the held-out map transports at ~100%; when one stops short, that
     one gets NO transport (64% vs its own 63%) while the others still
     transport at 100%.  The map only works if a CONSISTENT morphism exists
     across the others.

  AND THE GATE: fit quality proves nothing.  The map must be judged by a
     CAUSAL TRANSPLANT -- apply M.Delta_held to the initial weights and MEASURE
     THE LOSS.  This is the same discipline that killed the lambda_cos quartic
     (R^2 = 0.9996, causally worth 0.4%).

OUTPUT
    morphism.json / .png
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


GROUPS = {"Emb", "FF", "LayerNorm", "W_Q", "W_V", "W_O", "W_K", "other"}


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--target", type=float, default=0.45,
                    help="TARGET LOSS every optimizer must reach.  Chosen so "
                         "even the slowest can get there; the point is to "
                         "compare endpoints at EQUAL LOSS, not equal steps.")
    ap.add_argument("--max-steps", type=int, default=2500)
    ap.add_argument("--n-ckpt", type=int, default=30)
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

    print("=" * 78)
    print("  EQUAL-LOSS ENDPOINTS  +  THE MORPHISM SEARCH")
    print("=" * 78)
    print("  PREMISE CHECK FIRST.  The functor argument assumes all optimizers")
    print("  land in the SAME BASIN.  In the last run they did not: SGD ended")
    print("  at 1.77 vs AdamW's 0.081 -- 22x the loss, same init, same batches.")
    print("  So SGD's low overlap (36%) may simply mean IT IS ONLY PARTLY DONE.")
    print("  We remove that confound by training every optimizer to the SAME")
    print(f"  TARGET LOSS ({args.target}), not the same step count.")
    print()
    print("  THE TRAP IN THE MORPHISM TEST.  A linear map between two 3-dim")
    print("  subspaces ALWAYS fits (9 unknowns, 3 constraints).  Fitting M on")
    print("  the pair you then test is CIRCULAR: verified, it 'recovers' 100%")
    print("  of the target even when the source stopped 40% of the way.")
    print("  FIX: leave-one-out.  Fit M on the OTHER optimizers, apply to a")
    print("  HELD-OUT one, and judge it by a CAUSAL TRANSPLANT, never by fit.")

    start = snap(model)
    keys = [k for k in start if group_of(k) in GROUPS]

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
    print(f"\n  start val = {v0:.4f}   target = {args.target}")

    OPTS = {
        "AdamW":   (lambda p, lr: torch.optim.AdamW(p, lr=lr, betas=(0.9, 0.95),
                                                    weight_decay=0.1), LR),
        "SGD":     (lambda p, lr: torch.optim.SGD(p, lr=lr), LR * 300),
        "SGD+mom": (lambda p, lr: torch.optim.SGD(p, lr=lr, momentum=0.9),
                    LR * 60),
        "RMSprop": (lambda p, lr: torch.optim.RMSprop(p, lr=lr), LR * 0.2),
        "Adagrad": (lambda p, lr: torch.optim.Adagrad(p, lr=lr), LR * 5),
    }

    # ================================================================
    # PART 1 -- EQUAL-LOSS TRAINING
    # ================================================================
    print("\n" + "=" * 78)
    print(f"  PART 1: train each optimizer UNTIL val <= {args.target}")
    print("=" * 78)
    D = {}
    for name, (mk, lr) in OPTS.items():
        torch.manual_seed(args.seed)
        load(model, start)
        o = mk(model.parameters(), lr)
        ck = [snap(model)]
        reached, step_used = False, args.max_steps
        for s, (x, y) in enumerate(TRAIN, 1):
            model.train()
            _, l = model(x, y)
            o.zero_grad(); l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            o.step()
            if s % 10 == 0:
                v = evalf()
                if v <= args.target:
                    reached, step_used = True, s
                    ck.append(snap(model))
                    break
            if s % max(1, args.max_steps // args.n_ckpt) == 0:
                ck.append(snap(model))
        load(model, ck[-1])
        vf = evalf()
        X = np.stack([(flat(ck[i], keys) - flat(ck[i - 1], keys)).numpy()
                      for i in range(1, len(ck))])
        T = (flat(ck[-1], keys) - flat(start, keys)).numpy()
        B = np.linalg.svd(X, full_matrices=False)[2][:args.k].T
        D[name] = {"floor": vf, "descent": v0 - vf, "T": T, "B": B,
                   "steps": step_used, "reached": reached}
        print(f"   {name:<9} val={vf:.4f}  in {step_used:4d} steps  "
              f"{'REACHED target' if reached else '!! DID NOT REACH TARGET'}")

    ok = [n for n in D if D[n]["reached"]]
    if len(ok) < 4:
        print(f"\n  !! only {len(ok)} optimizers reached the target.  The")
        print("     endpoints are not comparable and the morphism question")
        print("     cannot be asked.  Raise --target or --max-steps.")
        return
    print(f"\n  {len(ok)} optimizers reached val <= {args.target}.")
    print("  Their endpoints are now COMPARABLE OBJECTS.")

    # ---- re-ask the overlap question at EQUAL LOSS ----
    print("\n" + "=" * 78)
    print("  OVERLAP AT EQUAL LOSS (was the 36% just 'SGD is only partly done'?)")
    print("=" * 78)
    P = D[ok[0]]["T"].size
    chance = float(np.sqrt(args.k * 3 / P))
    print(f"  {'pair':<22}{'subspace overlap':>18}{'x chance':>10}")
    print("  " + "-" * 52)
    for i, a in enumerate(ok):
        for b in ok[i + 1:]:
            Qa = np.linalg.qr(D[a]["B"])[0]
            s = np.linalg.svd(Qa.T @ np.linalg.qr(D[b]["B"])[0],
                              compute_uv=False)
            ov = float(s.mean())
            print(f"  {a+' <-> '+b:<22}{ov:>17.2f}{ov/chance:>9.0f}x")

    # ================================================================
    # PART 2 -- THE MORPHISM, LEAVE-ONE-OUT + CAUSAL GATE
    # ================================================================
    print("\n" + "=" * 78)
    print("  PART 2: is there a MORPHISM  Phi : S_i -> S_target ?")
    print("=" * 78)
    target = min(ok, key=lambda n: D[n]["floor"])
    print(f"  target optimizer (lowest loss) = {target} (val {D[target]['floor']:.4f})")
    print("  For each held-out optimizer H:")
    print("    * fit a linear map M on the OTHER optimizers (NOT on H)")
    print("    * apply M to H's displacement coefficients")
    print("    * TRANSPLANT the result and MEASURE the loss")
    print("  A map fitted on H and tested on H would always 'work' -- verified.")

    # common chart from ALL endpoints
    Mx = np.stack([D[n]["T"] for n in ok])
    Bc = np.linalg.svd(Mx, full_matrices=False)[2][:args.k].T
    Qc = np.linalg.qr(Bc)[0]
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

    v_target = transplant(Qc @ coef[target])
    d_target = v0 - v_target
    print(f"\n  target's own displacement, projected to the k={args.k} chart,")
    print(f"  recovers val={v_target:.4f}  (descent {d_target:.3f})")

    print(f"\n  {'held-out':<10}{'own val':>10}{'after M':>10}"
          f"{'recovery':>11}   morphism?")
    print("  " + "-" * 60)
    res = {}
    for held in ok:
        if held == target:
            continue
        tr = [n for n in ok if n not in (held, target)]
        if len(tr) < 2:
            continue
        X = np.stack([coef[n] for n in tr])                 # (n_tr, k)
        Y = np.stack([coef[target] for _ in tr])            # target coeffs
        Mmap, *_ = np.linalg.lstsq(X, Y, rcond=None)        # fitted WITHOUT held
        mapped = Qc @ (Mmap.T @ coef[held])
        v_map = transplant(mapped)
        v_own = transplant(Qc @ coef[held])
        rec = (v0 - v_map) / max(d_target, 1e-9)
        own = (v0 - v_own) / max(d_target, 1e-9)
        works = rec > 0.70 and rec > own + 0.10
        res[held] = {"v_own": v_own, "v_mapped": v_map, "recovery": rec,
                     "own_recovery": own, "morphism": bool(works),
                     "fitted_on": tr}
        print(f"  {held:<10}{v_own:>10.4f}{v_map:>10.4f}{100*rec:>10.0f}%   "
              f"{'YES' if works else 'no'}")

    print("\n" + "=" * 78)
    print("  VERDICT")
    print("=" * 78)
    any_ok = any(r["morphism"] for r in res.values())
    if any_ok and all(r["morphism"] for r in res.values()):
        print("   => A MORPHISM EXISTS.  A map fitted on OTHER optimizers")
        print("      transports a HELD-OUT optimizer's displacement into the")
        print("      target's solution, and the transplant recovers the loss.")
        print("      The optimizer subspaces are different CHARTS of one")
        print("      object, and there is a functor between them.  This is the")
        print("      structure the low dimensionality implied.")
    elif any_ok:
        print("   => PARTIAL.  The morphism transports SOME optimizers but not")
        print("      others.  Print above shows which.  A functor over a subset")
        print("      is still informative, but it is not the universal map.")
    else:
        print("   => NO MORPHISM.  No map fitted on the other optimizers")
        print("      transports a held-out one into the target's solution.")
        print("      Each optimizer's low-dimensional subspace carries its own")
        print("      solution and cannot be pulled back to another's, even at")
        print("      EQUAL LOSS.  The subspaces are not charts of a common")
        print("      object -- at least not linearly, and at this k.")
        print("      (A nonlinear or higher-dimensional map is not excluded;")
        print("       but the simplest instrument finds nothing to transport.)")

    json.dump({"target_loss": args.target, "v0": v0, "k": args.k,
               "per_optimizer": {n: {"floor": D[n]["floor"],
                                     "steps": D[n]["steps"],
                                     "reached": D[n]["reached"]} for n in D},
               "morphism": res, "target": target},
              open("morphism.json", "w"), indent=2, default=float)
    print("\n  wrote morphism.json")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        nmm = list(res)
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        ax[0].bar([n for n in D], [D[n]["steps"] for n in D], color="#4c72b0")
        ax[0].set_ylabel(f"steps to reach val={args.target}")
        ax[0].set_title("PART 1: equal-loss training\n(removes the basin confound)")
        ax[0].tick_params(axis="x", rotation=30)
        x = np.arange(len(nmm))
        ax[1].bar(x - 0.2, [100 * res[n]["own_recovery"] for n in nmm], 0.4,
                  label="its own displacement", color="#ccc")
        ax[1].bar(x + 0.2, [100 * res[n]["recovery"] for n in nmm], 0.4,
                  label="after the held-out map M", color="#55a868")
        ax[1].axhline(70, ls="--", color="k")
        ax[1].set_xticks(x); ax[1].set_xticklabels(nmm, rotation=30)
        ax[1].set_ylabel(f"% of {target}'s descent recovered")
        ax[1].set_title("PART 2: does M transport a HELD-OUT optimizer?\n"
                        "(causal transplant, not fit quality)")
        ax[1].legend(fontsize=8)
        plt.suptitle("Is there a functor between optimizer subspaces?",
                     fontsize=13, weight="bold")
        plt.tight_layout()
        plt.savefig("morphism.png", dpi=180)
        print("  wrote morphism.png")
    except Exception as e:
        print(f"  [plot skipped: {e}]")


if __name__ == "__main__":
    main()
