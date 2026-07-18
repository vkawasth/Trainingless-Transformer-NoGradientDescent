"""
cluster_morphism.py
===================
DIRECT TEST OF THE TWO-CHART STRUCTURE.

WHAT THE EQUAL-LOSS RUN FOUND (target 0.45)
--------------------------------------------
With the basin confound removed -- every optimizer trained until it reached
the SAME loss, not the same step count -- the subspace overlaps split into two
clean clusters:

    CLUSTER 1 (adaptive) : AdamW, RMSprop, Adagrad     within 0.60-0.88
    CLUSTER 2 (SGD-like) : SGD, SGD+mom                within 0.88
    BETWEEN the clusters :                             0.22-0.38

And the leave-one-out morphism results map EXACTLY onto that split.  Every case
is explained by whether the fitting set contained the held-out optimizer's
CLUSTER-MATE:

    hold out SGD      -> set contains SGD+mom (0.88) -> transports  95%
    hold out SGD+mom  -> set contains SGD     (0.88) -> transports 100%
    hold out AdamW    -> set dominated by SGD-likes  -> DIVERGES  -401%
    hold out RMSprop  -> set dominated by SGD-likes  -> DIVERGES   -41%

(The original script called all four "no morphism" because of a bug in its
acceptance test: it required rec > own + 0.10, which REJECTS a map that works
PERFECTLY -- when rec ~= own ~= 100%, the clause is false.  Two genuine
positives were suppressed.  The correct criterion is simply rec > 0.70.)

So the morphism appears to exist WITHIN a cluster and fail ACROSS.  That is a
positive structural claim -- TWO CHARTS, NOT ONE -- but it was INFERRED from a
leave-one-out artefact.  This script tests it DIRECTLY.

NOTE: SGD+mom sits with SGD, not with the adaptive methods.  So it is not
MOMENTUM that splits the clusters -- it is PER-COORDINATE RESCALING.

THE DIRECT TEST
---------------
Target loss 0.15 -- inside AdamW's basin (its basin entry), so the comparison
happens where the geometry matters rather than at a waypoint far up the slope.

  1. Train every optimizer to val <= 0.15.
  2. Fit the map M_within on cluster members ONLY, transport to a HELD-OUT
     member of the SAME cluster.  (Prediction: works.)
  3. Fit M_across on one cluster, transport to a member of the OTHER cluster.
     (Prediction: fails.)
  4. Judge BOTH by a CAUSAL TRANSPLANT -- apply the mapped displacement to the
     initial weights and MEASURE the loss.  Fit quality is never used: a
     linear map between two 3-dim subspaces always fits (9 unknowns, 3
     constraints), and a per-pair fit "recovers" 100% even when the source
     stopped 40% of the way.  Verified.

  A clean within-YES / across-NO is the two-chart result.
  If ACROSS also works, there is one chart after all and the clusters are
  cosmetic.
  If WITHIN also fails, the leave-one-out positives were an artefact and there
  is no morphism at all.

OUTPUT
    cluster_morphism.json / .png
"""

import argparse
import itertools
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

# Hypothesised clusters, from the equal-loss overlap matrix.
# NOTE: SGD+mom groups with SGD, NOT with the adaptive methods -- so the split
# is per-coordinate RESCALING, not momentum.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--target", type=float, default=0.15,
                    help="target loss: AdamW's BASIN ENTRY.  The comparison "
                         "should happen inside the basin, not at a waypoint "
                         "3.9 nats up the slope.")
    ap.add_argument("--max-steps", type=int, default=4000)
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
    print("  TWO CHARTS OR ONE?  within-cluster vs across-cluster morphism")
    print("=" * 78)
    print("  At EQUAL LOSS the subspace overlaps split cleanly:")
    print("     adaptive  (AdamW, RMSprop, Adagrad) : 0.60-0.88 within")
    print("     sgd-like  (SGD, SGD+mom)            : 0.88 within")
    print("     between the two clusters            : 0.22-0.38")
    print("  SGD+mom groups with SGD, NOT with the adaptives -- so the split")
    print("  is PER-COORDINATE RESCALING, not momentum.")
    print()
    print("  The leave-one-out morphism transported a held-out optimizer")
    print("  exactly when its CLUSTER-MATE was in the fitting set (SGD 95%,")
    print("  SGD+mom 100%) and diverged when it was not (AdamW -401%,")
    print("  RMSprop -41%).  That INFERS two charts.  This script tests it")
    print("  DIRECTLY, and at the basin entry (target %.2f)." % args.target)

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
        "RMSprop": (lambda p, lr: torch.optim.RMSprop(p, lr=lr), LR * 0.2),
        "Adagrad": (lambda p, lr: torch.optim.Adagrad(p, lr=lr), LR * 5),
        "SGD":     (lambda p, lr: torch.optim.SGD(p, lr=lr), LR * 300),
        "SGD+mom": (lambda p, lr: torch.optim.SGD(p, lr=lr, momentum=0.9),
                    LR * 60),
    }

    # ---------------- train to EQUAL LOSS ----------------
    print("\n" + "=" * 78)
    print(f"  train each optimizer UNTIL val <= {args.target} (basin entry)")
    print("=" * 78)
    D = {}
    for name, (mk, lr) in OPTS.items():
        torch.manual_seed(args.seed)
        load(model, start)
        o = mk(model.parameters(), lr)
        ck = [snap(model)]
        reached, used = False, args.max_steps
        for s, (x, y) in enumerate(TRAIN, 1):
            model.train()
            _, l = model(x, y)
            o.zero_grad(); l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            o.step()
            if s % 10 == 0 and evalf() <= args.target:
                reached, used = True, s
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
        D[name] = {"floor": vf, "T": T, "B": B, "steps": used,
                   "reached": reached, "cluster": CLUSTER[name]}
        print(f"   {name:<9} [{CLUSTER[name]:<8}] val={vf:.4f}  "
              f"{used:5d} steps  {'OK' if reached else '!! DID NOT REACH'}")

    ok = [n for n in D if D[n]["reached"]]
    if len(ok) < 5:
        miss = [n for n in D if not D[n]["reached"]]
        print(f"\n  !! {miss} did not reach {args.target}.")
        print("     Every optimizer must reach the SAME loss or the endpoints")
        print("     are not comparable.  Raise --max-steps, or --target.")
        return

    # ---------------- overlap matrix (confirm the clusters) --------------
    P = D[ok[0]]["T"].size
    chance = float(np.sqrt(args.k * 3 / P))
    print("\n" + "=" * 78)
    print("  OVERLAP MATRIX AT THE BASIN ENTRY (do the clusters survive?)")
    print("=" * 78)
    ovm = {}
    print("  " + "".join(f"{n[:8]:>10}" for n in ok))
    for a in ok:
        line = f"  {a[:8]:<8}"
        for b in ok:
            if a == b:
                line += f"{'--':>10}"
            else:
                Qa = np.linalg.qr(D[a]["B"])[0]
                Qb = np.linalg.qr(D[b]["B"])[0]
                o_ = float(np.linalg.svd(Qa.T @ Qb, compute_uv=False).mean())
                ovm[(a, b)] = o_
                line += f"{o_:>10.2f}"
        print(line)
    win = [v for (a, b), v in ovm.items() if CLUSTER[a] == CLUSTER[b]]
    acr = [v for (a, b), v in ovm.items() if CLUSTER[a] != CLUSTER[b]]
    print(f"\n  mean WITHIN-cluster overlap : {np.mean(win):.2f}")
    print(f"  mean ACROSS-cluster overlap : {np.mean(acr):.2f}")
    print(f"  (chance = {chance:.4f})")
    clusters_hold = np.mean(win) > np.mean(acr) + 0.15
    print(f"  -> the two-cluster structure "
          f"{'SURVIVES at the basin entry' if clusters_hold else 'does NOT hold here'}")

    # ---------------- the morphism, causally ----------------
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

    def fit_and_test(train_set, held, target):
        """Fit M on `train_set` (mapping each member's coeffs -> target's),
        apply to `held`, TRANSPLANT, and measure.  `held` is NEVER in
        train_set -- fitting on the pair you test is circular and always
        'works' (verified)."""
        X = np.stack([coef[n] for n in train_set])
        Y = np.stack([coef[target] for _ in train_set])
        M, *_ = np.linalg.lstsq(X, Y, rcond=None)
        v_map = transplant(Qc @ (M.T @ coef[held]))
        return v_map

    print("\n" + "=" * 78)
    print("  THE DIRECT TEST  (causal transplant; fit quality never used)")
    print("=" * 78)
    results = []
    for target in ok:
        tc = CLUSTER[target]
        mates = [n for n in ok if n != target and CLUSTER[n] == tc]
        others = [n for n in ok if CLUSTER[n] != tc]
        v_t = transplant(Qc @ coef[target])
        d_t = v0 - v_t
        if d_t < 0.5:
            continue

        # WITHIN: fit on the target's own cluster-mates (excluding the held-out
        # one), transport a held-out CLUSTER-MATE.
        for held in mates:
            tr = [n for n in mates if n != held]
            if not tr:
                continue
            v = fit_and_test(tr, held, target)
            rec = (v0 - v) / d_t
            results.append({"target": target, "held": held, "kind": "WITHIN",
                            "fitted_on": tr, "val": v, "recovery": rec})
        # ACROSS: fit on the OTHER cluster entirely, transport a member of it.
        for held in others:
            tr = [n for n in others if n != held]
            if not tr:
                continue
            v = fit_and_test(tr, held, target)
            rec = (v0 - v) / d_t
            results.append({"target": target, "held": held, "kind": "ACROSS",
                            "fitted_on": tr, "val": v, "recovery": rec})

    print(f"  {'target':<9}{'held-out':<9}{'kind':<8}{'fitted on':<26}"
          f"{'val':>8}{'recovery':>10}")
    print("  " + "-" * 72)
    for r in results:
        print(f"  {r['target']:<9}{r['held']:<9}{r['kind']:<8}"
              f"{','.join(r['fitted_on'])[:24]:<26}"
              f"{r['val']:>8.3f}{100*r['recovery']:>9.0f}%")

    W = [r["recovery"] for r in results if r["kind"] == "WITHIN"]
    A = [r["recovery"] for r in results if r["kind"] == "ACROSS"]
    # ACCEPTANCE: transport means the mapped displacement recovers the target's
    # descent.  NOTHING ELSE.  (The previous script also required
    # rec > own + 0.10, which REJECTS a perfect map -- when rec ~= own ~= 100%
    # that clause is false.  Two genuine positives were suppressed by it.)
    wok = float(np.mean([r > 0.70 for r in W])) if W else float("nan")
    aok = float(np.mean([r > 0.70 for r in A])) if A else float("nan")

    print("\n" + "=" * 78)
    print("  VERDICT")
    print("=" * 78)
    print(f"   WITHIN-cluster : {len(W)} tests, "
          f"median recovery {100*np.median(W):.0f}%, "
          f"{100*wok:.0f}% transport (>70%)")
    print(f"   ACROSS-cluster : {len(A)} tests, "
          f"median recovery {100*np.median(A):.0f}%, "
          f"{100*aok:.0f}% transport (>70%)")
    print()
    if wok > 0.6 and aok < 0.4:
        print("   => TWO CHARTS, NOT ONE.")
        print("      A linear morphism transports optimizers WITHIN a cluster")
        print("      and fails ACROSS.  Both families reach the SAME loss, in")
        print("      3 dimensions each, yet no linear map carries one family's")
        print("      solution into the other's.")
        print("      The functor exists -- but only over each chart.  The")
        print("      subspaces are charts of TWO objects, not one.")
        print("      And since SGD+mom sits with SGD, the thing that separates")
        print("      the charts is PER-COORDINATE RESCALING, not momentum.")
    elif wok > 0.6 and aok > 0.6:
        print("   => ONE CHART.  The morphism transports ACROSS clusters too.")
        print("      The overlap clusters are cosmetic; there is a single")
        print("      object and a functor over all optimizers.  This is the")
        print("      universal map the low dimensionality implied.")
    elif wok < 0.4:
        print("   => NO MORPHISM, even within a cluster.  The leave-one-out")
        print("      positives were an artefact of the fitting set, not")
        print("      evidence of transportable structure.")
    else:
        print("   => MIXED / underpowered.  See the table; no clean claim.")

    print("\n   SCOPE: n=5 optimizers, one seed, one architecture, one corpus.")
    print("   The cluster structure is clean but rests on five points and")
    print("   needs replication before it can carry weight.")

    json.dump({"target": args.target, "v0": v0, "k": args.k,
               "per_optimizer": {n: {"floor": D[n]["floor"],
                                     "steps": D[n]["steps"],
                                     "cluster": D[n]["cluster"]} for n in D},
               "overlap": {f"{a}|{b}": v for (a, b), v in ovm.items()},
               "mean_within": float(np.mean(win)),
               "mean_across": float(np.mean(acr)),
               "tests": results,
               "within_transport_rate": wok,
               "across_transport_rate": aok},
              open("cluster_morphism.json", "w"), indent=2, default=float)
    print("\n  wrote cluster_morphism.json")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
        Mm = np.zeros((len(ok), len(ok)))
        for i, a in enumerate(ok):
            for j, b in enumerate(ok):
                Mm[i, j] = 1.0 if a == b else ovm[(a, b)]
        im = ax[0].imshow(Mm, cmap="viridis", vmin=0, vmax=1)
        ax[0].set_xticks(range(len(ok))); ax[0].set_xticklabels(ok, rotation=45)
        ax[0].set_yticks(range(len(ok))); ax[0].set_yticklabels(ok)
        plt.colorbar(im, ax=ax[0], label="subspace overlap")
        ax[0].set_title(f"Overlap at the basin entry (val={args.target})\n"
                        f"within {np.mean(win):.2f} vs across {np.mean(acr):.2f}")
        bp = [W, A]
        ax[1].boxplot([[100 * x for x in b] for b in bp],
                      labels=["WITHIN cluster", "ACROSS cluster"])
        ax[1].axhline(70, ls="--", color="k", label="transport threshold")
        ax[1].set_ylabel("% of target's descent recovered (causal)")
        ax[1].set_title("Does the morphism transport?\n(causal transplant)")
        ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
        plt.suptitle("Two charts or one?", fontsize=13, weight="bold")
        plt.tight_layout()
        plt.savefig("cluster_morphism.png", dpi=180)
        print("  wrote cluster_morphism.png")
    except Exception as e:
        print(f"  [plot skipped: {e}]")


if __name__ == "__main__":
    main()
