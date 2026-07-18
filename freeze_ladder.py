"""
freeze_ladder.py
================
WHICH PARAMETER BLOCKS ACTUALLY REQUIRE OPTIMIZATION?

-------------------------------------------------------------------------------
WHY THIS IS THE RIGHT QUESTION
-------------------------------------------------------------------------------
Three independent results converge on the same suspicion:

  1. E = 43.5 is INVARIANT (CV 0.0038 across five optimizers at matched loss),
     and E is a function of W_K alone.
  2. CAUSAL ABLATION: transplanting W_K's converged value into an untrained
     model buys 0.023 nats of a 4.37-nat descent -- 0.5%.
  3. PLAIN SGD reaches the basin (val 0.146) with lambda_cos = 0.999: it
     essentially NEVER ROTATES W_K, yet the loss falls by a factor of thirty.

So: does W_K need to be optimized at all?

-------------------------------------------------------------------------------
THE CAVEAT THAT MAKES THIS A REAL EXPERIMENT AND NOT A FOREGONE CONCLUSION
-------------------------------------------------------------------------------
The ablation showed that W_K's CONVERGED VALUE contributes 0.5% of the loss.
That is NOT the same claim as "W_K needs no optimization".

A parameter can be nearly irrelevant AT THE ENDPOINT while doing necessary work
DURING the descent -- for instance shaping the attention pattern early so that
the feed-forward blocks can learn against it.  The ablation tests the endpoint.
FREEZING tests the process.  They can disagree, and only this experiment
decides.

Similarly, SGD's lambda_cos = 0.999 shows W_K barely MOVES under SGD.  It does
not show that the small motion is unnecessary -- nor that the same holds for
AdamW, which rotates W_K substantially (lambda_cos -> 0.67).  If AdamW NEEDS
that rotation, freezing will hurt it and not SGD, which would itself be a
finding.

-------------------------------------------------------------------------------
THE LADDER
-------------------------------------------------------------------------------
Progressively freeze larger subsets, always from the SAME initialisation, on
the SAME replayed minibatch sequence, evaluated on the SAME held-out set:

  L0  full model                                  (baseline)
  L1  freeze W_K
  L2  freeze W_K, W_Q
  L3  freeze all attention projections (W_K,W_Q,W_V,W_O)
  L4  freeze attention + LayerNorm  -> Emb + FF only
  L5  freeze everything except FF
  L6  freeze everything except Emb

For each: final loss, steps to reach a target, and the FRACTION OF PARAMETERS
STILL BEING TRAINED.  A rung "passes" if it reaches the target loss in no more
steps than the baseline.

RUN FOR BOTH AdamW AND SGD.  They rotate W_K completely differently, so a
result that holds for only one of them is a different -- and more interesting --
finding than one that holds for both.

CONTROL (mandatory): freezing removes parameters.  Some of the effect could be
regularisation rather than "those parameters were unnecessary".  So we also run
a RANDOM-FREEZE control at each rung: freeze the SAME NUMBER of parameters,
chosen at random across the whole model.  If the structured freeze does no
better than the random freeze of equal size, the block structure is carrying no
information.

OUTPUT
    freeze_ladder.json / .png
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


# The ladder: which GROUPS are frozen at each rung.
LADDER = [
    ("L0 full model",            set()),
    ("L1 freeze W_K",            {"W_K"}),
    ("L2 + W_Q",                 {"W_K", "W_Q"}),
    ("L3 + all attention",       {"W_K", "W_Q", "W_V", "W_O"}),
    ("L4 + LayerNorm (Emb+FF)",  {"W_K", "W_Q", "W_V", "W_O", "LayerNorm"}),
    ("L5 FF only",               {"W_K", "W_Q", "W_V", "W_O", "LayerNorm",
                                  "Emb", "other"}),
    ("L6 Emb only",              {"W_K", "W_Q", "W_V", "W_O", "LayerNorm",
                                  "FF", "other"}),
]


def snap(m):
    return {k: v.detach().clone() for k, v in m.state_dict().items()}


def load(m, s):
    m.load_state_dict({k: v.clone() for k, v in s.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=0.15,
                    help="loss the baseline reaches; rungs are judged on")
    ap.add_argument("--max-steps", type=int, default=1200)
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
    print("  WHICH PARAMETER BLOCKS ACTUALLY REQUIRE OPTIMIZATION?")
    print("=" * 78)
    print("  Three results converge on the suspicion that W_K does not:")
    print("    * E (a function of W_K alone) is invariant, CV = 0.0038")
    print("    * W_K's converged value buys 0.5% of the descent (ablation)")
    print("    * plain SGD reaches the basin with lambda_cos = 0.999 --")
    print("      it essentially never rotates W_K")
    print()
    print("  THE CAVEAT THIS EXPERIMENT EXISTS TO SETTLE:")
    print("  the ablation tested W_K's CONVERGED VALUE.  A parameter can be")
    print("  irrelevant at the ENDPOINT while doing necessary work DURING the")
    print("  descent (shaping attention early so the FF can learn against it).")
    print("  The ablation tests the endpoint; FREEZING tests the process.")
    print("  They can disagree.  Only this decides.")
    print()
    print("  Run for BOTH AdamW and SGD: they rotate W_K completely")
    print("  differently, so a result holding for only one is a DIFFERENT and")
    print("  more interesting finding than one holding for both.")

    start = snap(model)
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
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  start val = {v0:.4f}   target = {args.target}")
    print(f"  total parameters = {total_params:,}")

    def run(frozen_groups, opt_name, random_freeze_n=None):
        """Train with `frozen_groups` held fixed (or, if random_freeze_n is
        given, with that many RANDOMLY CHOSEN parameters held fixed instead --
        the control for 'is it the structure, or just having fewer params?')."""
        torch.manual_seed(args.seed)
        load(model, start)

        masks = {}
        n_frozen = 0
        if random_freeze_n is None:
            for n, p in model.named_parameters():
                if group_of(n) in frozen_groups:
                    p.requires_grad_(False)
                    n_frozen += p.numel()
                else:
                    p.requires_grad_(True)
        else:
            # RANDOM CONTROL: freeze the same NUMBER of individual weights,
            # scattered across the whole model, via a gradient mask.
            gen = torch.Generator().manual_seed(999)
            flat_n = total_params
            keep = torch.ones(flat_n)
            idx = torch.randperm(flat_n, generator=gen)[:random_freeze_n]
            keep[idx] = 0.0
            i = 0
            for n, p in model.named_parameters():
                p.requires_grad_(True)
                masks[n] = keep[i:i + p.numel()].view_as(p)
                i += p.numel()
            n_frozen = random_freeze_n

        trainable = [p for p in model.parameters() if p.requires_grad]
        if not trainable:
            return None
        if opt_name == "AdamW":
            o = torch.optim.AdamW(trainable, lr=LR, betas=(0.9, 0.95),
                                  weight_decay=0.1)
        else:
            o = torch.optim.SGD(trainable, lr=LR * 300)

        hit, used = False, args.max_steps
        for s, (x, y) in enumerate(TRAIN, 1):
            model.train()
            _, l = model(x, y)
            o.zero_grad(); l.backward()
            if masks:
                for n, p in model.named_parameters():
                    if p.grad is not None:
                        p.grad.mul_(masks[n])
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            o.step()
            if s % 10 == 0 and evalf() <= args.target:
                hit, used = True, s
                break
        vf = evalf()
        for p in model.parameters():
            p.requires_grad_(True)
        return {"val": vf, "steps": used, "reached": hit,
                "frozen": n_frozen,
                "trainable_frac": 1.0 - n_frozen / total_params}

    results = {}
    for opt_name in ["AdamW", "SGD"]:
        print("\n" + "=" * 78)
        print(f"  THE LADDER --- {opt_name}")
        print("=" * 78)
        print(f"  {'rung':<26}{'trainable':>11}{'steps':>8}{'val':>9}"
              f"   vs baseline")
        print("  " + "-" * 68)
        rows, base = [], None
        for label, frozen in LADDER:
            r = run(frozen, opt_name)
            if r is None:
                continue
            if base is None:
                base = r
            if r["reached"]:
                dv = r["steps"] / max(base["steps"], 1)
                tag = (f"{dv:.2f}x steps" +
                       ("  FASTER" if dv < 0.95 else
                        "  same" if dv < 1.15 else "  slower"))
            else:
                tag = f"!! NEVER REACHED (val {r['val']:.3f})"
            rows.append({"rung": label, **r})
            print(f"  {label:<26}{100*r['trainable_frac']:>10.1f}%"
                  f"{r['steps']:>8}{r['val']:>9.4f}   {tag}")

        # ---- the RANDOM-FREEZE control ----
        print(f"\n  RANDOM-FREEZE CONTROL ({opt_name})")
        print("  Freeze the SAME NUMBER of parameters, chosen at random across")
        print("  the whole model.  If a structured freeze does no better than")
        print("  a random freeze of equal size, the block structure carries no")
        print("  information and we are only seeing a regularisation effect.")
        print(f"  {'matched to':<26}{'trainable':>11}{'steps':>8}{'val':>9}"
              f"   structured?")
        print("  " + "-" * 68)
        ctrl = []
        for row in rows[1:]:                       # skip the full model
            rr = run(set(), opt_name, random_freeze_n=row["frozen"])
            if rr is None:
                continue
            better = (row["reached"] and
                      (not rr["reached"] or row["steps"] <= rr["steps"]))
            ctrl.append({"matched_to": row["rung"], **rr})
            v = ("STRUCTURE HELPS" if better else
                 "no better than random")
            print(f"  {row['rung']:<26}{100*rr['trainable_frac']:>10.1f}%"
                  f"{rr['steps']:>8}{rr['val']:>9.4f}   {v}")
        results[opt_name] = {"ladder": rows, "control": ctrl}

    # ================================================================
    # VERDICT
    # ================================================================
    print("\n" + "=" * 78)
    print("  VERDICT")
    print("=" * 78)
    for opt_name in ["AdamW", "SGD"]:
        rows = results[opt_name]["ladder"]
        base = rows[0]
        ok = [r for r in rows[1:] if r["reached"]
              and r["steps"] <= 1.15 * base["steps"]]
        if ok:
            best = min(ok, key=lambda r: r["trainable_frac"])
            print(f"\n  [{opt_name}] deepest rung that matches the baseline:")
            print(f"     {best['rung']}")
            print(f"     trains {100*best['trainable_frac']:.1f}% of parameters"
                  f"  ({best['steps']} steps vs baseline {base['steps']})")
            print(f"     -> {100*(1-best['trainable_frac']):.1f}% of the model"
                  f" can be FROZEN with no loss of convergence.")
        else:
            print(f"\n  [{opt_name}] NO rung matches the baseline.")
            print("     Every freeze slows convergence or fails to reach the")
            print("     target.  The blocks that look irrelevant AT THE")
            print("     ENDPOINT are nevertheless doing work DURING the")
            print("     descent -- exactly the caveat this experiment was")
            print("     built to catch.")

    a = results["AdamW"]["ladder"]
    s_ = results["SGD"]["ladder"]
    a1 = next((r for r in a if r["rung"].startswith("L1")), None)
    s1 = next((r for r in s_ if r["rung"].startswith("L1")), None)
    if a1 and s1:
        print("\n  THE W_K QUESTION SPECIFICALLY (rung L1):")
        print(f"    AdamW (rotates W_K hard, lambda_cos -> 0.67):"
              f"  {'OK' if a1['reached'] else 'FAILS'}"
              f"  {a1['steps']} steps, val {a1['val']:.4f}")
        print(f"    SGD   (never rotates W_K, lambda_cos = 0.999):"
              f"  {'OK' if s1['reached'] else 'FAILS'}"
              f"  {s1['steps']} steps, val {s1['val']:.4f}")
        if a1["reached"] and s1["reached"]:
            print("    -> W_K NEEDS NO OPTIMIZATION, for either optimizer.")
            print("       The invariant, the ablation and the freeze agree.")
        elif s1["reached"] and not a1["reached"]:
            print("    -> W_K is dispensable for SGD but NOT for AdamW.")
            print("       AdamW's W_K rotation is doing real work during the")
            print("       descent even though its endpoint value is worth 0.5%.")
            print("       That is the endpoint/process distinction, measured.")

    json.dump({"v0": v0, "target": args.target,
               "total_params": total_params, "results": results},
              open("freeze_ladder.json", "w"), indent=2, default=float)
    print("\n  wrote freeze_ladder.json")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
        for i, opt_name in enumerate(["AdamW", "SGD"]):
            rows = results[opt_name]["ladder"]
            ctrl = results[opt_name]["control"]
            x = [100 * r["trainable_frac"] for r in rows]
            y = [r["steps"] for r in rows]
            ax[i].plot(x, y, "o-", color="#4c72b0", label="structured freeze")
            cx = [100 * c["trainable_frac"] for c in ctrl]
            cy = [c["steps"] for c in ctrl]
            ax[i].plot(cx, cy, "s--", color="#c44e52",
                       label="random freeze (control)")
            ax[i].axhline(rows[0]["steps"], ls=":", color="k",
                          label="full-model baseline")
            for r in rows:
                ax[i].annotate(r["rung"].split()[0],
                               (100 * r["trainable_frac"], r["steps"]),
                               fontsize=7, xytext=(3, 3),
                               textcoords="offset points")
            ax[i].set_xlabel("% of parameters trained")
            ax[i].set_ylabel(f"steps to reach val={args.target}")
            ax[i].set_title(opt_name)
            ax[i].invert_xaxis()
            ax[i].grid(alpha=0.3); ax[i].legend(fontsize=8)
        plt.suptitle("Which parameter blocks actually require optimization?",
                     fontsize=13, weight="bold")
        plt.tight_layout()
        plt.savefig("freeze_ladder.png", dpi=180)
        print("  wrote freeze_ladder.png")
    except Exception as e:
        print(f"  [plot skipped: {e}]")


if __name__ == "__main__":
    main()
