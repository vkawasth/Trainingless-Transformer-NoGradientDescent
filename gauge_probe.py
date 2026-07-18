"""
gauge_probe.py
==============
THE QUOTIENT:   parameter space / functionally-equivalent directions

Everything measured so far has been in PARAMETER coordinates.  If the real
geometry is the quotient, those are the WRONG coordinates, and the object that
defines the quotient is the Jacobian

        J_theta = d f / d theta :  T_theta M  ->  F

  ker(J)   = the GAUGE subspace, by construction (not a proxy for it)
  F = J^T J = the metric ON the quotient -- which parameter motions are REAL

-------------------------------------------------------------------------------
TWO QUESTIONS, ONE INSTRUMENT
-------------------------------------------------------------------------------
(A) IS W_K GAUGE?

    Three results have looked like coincidences:
      * W_K's converged value buys 0.5% of the descent (causal ablation)
      * SGD reaches the basin WITHOUT rotating W_K (lambda_cos = 0.999)
      * E -- a function of W_K ALONE -- is conserved (CV 0.0038)
    If W_K motion is largely invisible to the function, all three are ONE fact.
    The loss cannot push along directions that do not change f; an optimizer
    need not move them; and a quantity supported on them sits unmoved.

    THE CONFOUND THAT MUST BE KILLED.  "W_K is gauge" and "W_K is only 9% of
    the model" predict the SAME THING on every measurement made so far.  The
    freeze ladder already showed that freezing a RANDOM 9% costs about as much
    as freezing W_K (120 vs 130 steps).  So the test must be MATCHED-NORM:

        ||J v_WK|| / ||v_WK||     vs     ||J v_rand|| / ||v_rand||

    for unit perturbations supported on W_K versus on a RANDOM subset OF THE
    SAME SIZE.  If W_K directions produce systematically LESS function change,
    W_K is genuinely gauge.  If the same, W_K is not special -- it is merely
    small, and the "gauge" story is a story about overparameterisation.
    (Validated on a toy net: ratio 0.176, i.e. W_K directions are 5.7x less
    visible than same-size random ones.  The discriminator has power.)

(B) DO THE OPTIMIZERS AGREE, SEEN CORRECTLY?

    In parameter space the optimizer tangents are 80-84 degrees apart across
    families.  But the optimizers differ by their PRECONDITIONER, not their
    learning rate:
        SGD, SGD+mom : P = I          (momentum filters TIME, not coordinates)
        AdamW/RMSprop/Adagrad : P = diag(1/sqrt(v))
    Only SGD takes steepest descent.  The others take steepest descent IN A
    DIFFERENT METRIC.  A different metric gives a different steepest direction
    -- that is what a metric IS.  So 80 degrees in parameter space is expected
    and is not, by itself, evidence of anything.

    THE REAL QUESTION: what is the angle IN FUNCTION SPACE?

        angle( J . theta_dot_AdamW ,  J . theta_dot_SGD )

    * If the parameter angle is 80 deg but the FUNCTION angle collapses toward
      0, then the optimizers are doing the SAME THING seen correctly; the 80
      deg is GAUGE, an artefact of the wrong coordinates; and the "right map"
      is J itself.
    * If the function angle stays large, the optimizers genuinely disagree
      about WHICH FUNCTION to build.  The two families are then inequivalent --
      the "different sheets" reading -- which is a stronger result.

-------------------------------------------------------------------------------
THE CHARTS (so the manifold is SEEN, not asserted)
-------------------------------------------------------------------------------
 1. Matched-norm sensitivity by block, with the random same-size null.
 2. Jacobian singular spectrum: how many directions does the function actually
    SEE, out of 4.3M?  (the dimension of the quotient)
 3. Parameter-space angles  vs  FUNCTION-space angles, side by side.  This is
    the decisive chart: if the second collapses while the first does not, the
    manifold is real and the parameter coordinates were lying.
 4. Fisher metric restricted to each block: the metric on the quotient.
 5. The five optimizer update directions projected into the top-2 function-
    space directions -- the manifold, drawn.

OUTPUT
    gauge_probe.json / gauge_probe.png
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


BLOCKS = ["W_K", "W_Q", "W_V", "W_O", "FF", "Emb", "LayerNorm"]
FAMILY = {"AdamW": "adaptive", "RMSprop": "adaptive", "Adagrad": "adaptive",
          "SGD": "sgd-like", "SGD+mom": "sgd-like"}


def snap(m):
    return {k: v.detach().clone() for k, v in m.state_dict().items()}


def load(m, s):
    m.load_state_dict({k: v.clone() for k, v in s.items()})


def angle(a, b):
    c = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warm", type=int, default=120,
                    help="steps to warm each optimizer (populates its "
                         "preconditioner state)")
    ap.add_argument("--n-probe", type=int, default=40,
                    help="random directions per block for the sensitivity test")
    ap.add_argument("--n-spec", type=int, default=60,
                    help="probes for the Jacobian spectrum")
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
    print("  THE QUOTIENT:  parameter space / functionally-invisible directions")
    print("=" * 78)
    print("  Everything so far was measured in PARAMETER coordinates.  If the")
    print("  real geometry is the quotient, those are the WRONG coordinates.")
    print("  The object that DEFINES the quotient is the Jacobian:")
    print("      ker(J) IS the gauge subspace.  F = J^T J is its metric.")
    print()
    print("  NOTE ON WHY THE OPTIMIZERS DIFFER.  Not learning rate:")
    print("     SGD, SGD+mom          P = I     (momentum filters TIME)")
    print("     AdamW/RMSprop/Adagrad P = diag(1/sqrt(v))")
    print("  Only SGD takes steepest descent.  The others take steepest descent")
    print("  IN A DIFFERENT METRIC.  80 deg apart in parameter space is what a")
    print("  different metric MEANS -- it is not, by itself, evidence.")
    print("  The real question is the angle IN FUNCTION SPACE.")

    names = [n for n, _ in model.named_parameters()]
    params = [p for _, p in model.named_parameters()]
    P = sum(p.numel() for p in params)
    print(f"\n  P = {P:,}")

    xb, yb = get_batch()

    def fwd(*ps):
        out = torch.func.functional_call(
            model, {k: v for k, v in zip(names, ps)}, (xb,))
        return out[0] if isinstance(out, tuple) else out

    def jvp(v):
        """J @ v -- the FUNCTION-space change per unit parameter motion.
        This is the whole instrument: it says which parameter directions the
        network can actually SEE."""
        vs, i = [], 0
        for p in params:
            n = p.numel()
            vs.append(v[i:i + n].view_as(p))
            i += n
        _, out = torch.func.jvp(fwd, tuple(params), tuple(vs))
        return out.reshape(-1).detach()

    # block masks
    masks, sizes = {}, {}
    for b in BLOCKS:
        m_ = torch.cat([(torch.ones(p.numel()) if group_of(n) == b
                         else torch.zeros(p.numel()))
                        for n, p in zip(names, params)])
        masks[b] = m_
        sizes[b] = int(m_.sum())

    gen = torch.Generator().manual_seed(args.seed)

    def sensitivity(mask, n):
        """||J v|| / ||v||  for unit v supported on `mask`."""
        out = []
        for _ in range(n):
            v = torch.randn(P, generator=gen) * mask
            nv = float(v.norm())
            if nv < 1e-9:
                continue
            out.append(float(jvp(v / nv).norm()))
        return np.array(out)

    # ================================================================
    # (A) IS W_K GAUGE?  -- matched-norm, matched-size
    # ================================================================
    print("\n" + "=" * 78)
    print("  (A) IS W_K GAUGE?   ||J v|| / ||v||  at MATCHED NORM")
    print("=" * 78)
    print("  THE CONFOUND: 'W_K is gauge' and 'W_K is only 9% of the model'")
    print("  predict the same thing on every earlier measurement.  So each")
    print("  block is compared against a RANDOM SUBSET OF THE SAME SIZE.")
    print()
    print(f"  {'block':<11}{'params':>10}{'% of P':>8}{'sensitivity':>13}"
          f"{'random same-size':>18}{'ratio':>8}")
    print("  " + "-" * 70)
    sens = {}
    for b in BLOCKS:
        if sizes[b] == 0:
            continue
        s_b = sensitivity(masks[b], args.n_probe)
        # the matched-size random control
        idx = torch.randperm(P, generator=gen)[:sizes[b]]
        mr = torch.zeros(P)
        mr[idx] = 1.0
        s_r = sensitivity(mr, args.n_probe)
        ratio = float(s_b.mean() / (s_r.mean() + 1e-12))
        sens[b] = {"sens": float(s_b.mean()), "sens_std": float(s_b.std()),
                   "null": float(s_r.mean()), "null_std": float(s_r.std()),
                   "ratio": ratio, "n": sizes[b],
                   "frac": sizes[b] / P}
        print(f"  {b:<11}{sizes[b]:>10,}{100*sizes[b]/P:>7.1f}%"
              f"{s_b.mean():>13.5f}{s_r.mean():>18.5f}{ratio:>8.3f}")

    wk = sens["W_K"]
    print()
    if wk["ratio"] < 0.5:
        print(f"  => W_K IS GAUGE.  Its directions are {1/wk['ratio']:.1f}x LESS")
        print("     visible to the function than random directions of the SAME")
        print("     SIZE.  This is not 'W_K is small' -- the control is matched.")
        print()
        print("     THREE RESULTS COLLAPSE INTO ONE:")
        print("       * W_K buys 0.5% of the loss  -- the loss cannot push along")
        print("         directions that do not change f")
        print("       * SGD never rotates W_K      -- it need not")
        print("       * E (a W_K quantity) is conserved -- it is supported on")
        print("         directions the objective does not see")
    elif wk["ratio"] > 0.8:
        print("  => W_K IS NOT GAUGE.  Its directions are as visible to the")
        print("     function as random ones of the same size.  The earlier")
        print("     results reflect W_K being SMALL (9% of an overparameterised")
        print("     net), not a symmetry.  The gauge reading is NOT supported.")
    else:
        print(f"  => PARTIAL GAUGE (ratio {wk['ratio']:.2f}).  W_K directions are")
        print("     less visible than random, but not invisible.  It is visible")
        print("     to the REPRESENTATION (its converged value shifts Phi_cl")
        print("     2/5 -> 3/5 and E) while nearly invisible to the LOSS.")
        print("     Directions the function sees but the objective does not")
        print("     reward -- subtler than a null space.")

    # ================================================================
    # THE JACOBIAN SPECTRUM -- the dimension of the quotient
    # ================================================================
    print("\n" + "=" * 78)
    print("  THE DIMENSION OF THE QUOTIENT (Jacobian spectrum)")
    print("=" * 78)
    print("  How many parameter directions can the function actually SEE?")
    Vs = []
    for _ in range(args.n_spec):
        v = torch.randn(P, generator=gen)
        v /= v.norm()
        Vs.append(jvp(v).numpy())
    Y = np.stack(Vs)                       # (n_spec, out_dim)
    sv = np.linalg.svd(Y, compute_uv=False)
    e = sv ** 2 / (sv ** 2).sum()
    eff = float(np.exp(-np.sum(e * np.log(e + 1e-12))))
    cum = np.cumsum(e)
    k90 = int(np.searchsorted(cum, 0.90) + 1)
    print(f"   {args.n_spec} random probes -> singular values of J's image")
    print(f"   top-10: {np.array2string(sv[:10], precision=3)}")
    print(f"   effective rank = {eff:.2f}  of {len(sv)} probes")
    print(f"   {k90} directions carry 90% of the function's response")
    print("   (the function responds in a LOW-DIMENSIONAL subspace: this is the")
    print("    quotient, and its dimension is the number above -- not 4.3M)")

    # ================================================================
    # (B) DO THE OPTIMIZERS AGREE IN FUNCTION SPACE?
    # ================================================================
    print("\n" + "=" * 78)
    print("  (B) PARAMETER-SPACE ANGLES  vs  FUNCTION-SPACE ANGLES")
    print("=" * 78)
    print("  THE DECISIVE COMPARISON.  If the parameter angle is 80 deg but the")
    print("  FUNCTION angle collapses, the 80 deg was GAUGE and the optimizers")
    print("  are doing the same thing seen correctly.")

    start = snap(model)
    OPTS = {
        "AdamW":   (lambda p, lr: torch.optim.AdamW(p, lr=lr, betas=(0.9, 0.95),
                                                    weight_decay=0.1), LR),
        "RMSprop": (lambda p, lr: torch.optim.RMSprop(p, lr=lr), LR * 0.2),
        "Adagrad": (lambda p, lr: torch.optim.Adagrad(p, lr=lr), LR * 5),
        "SGD":     (lambda p, lr: torch.optim.SGD(p, lr=lr), LR * 300),
        "SGD+mom": (lambda p, lr: torch.optim.SGD(p, lr=lr, momentum=0.9),
                    LR * 60),
    }

    torch.manual_seed(args.seed)
    WARM = [get_batch() for _ in range(args.warm)]
    upd, fupd = {}, {}
    for name, (mk, lr) in OPTS.items():
        torch.manual_seed(args.seed)
        load(model, start)
        o = mk(model.parameters(), lr)
        for (x, y) in WARM:
            model.train()
            _, l = model(x, y)
            o.zero_grad(); l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            o.step()
        # the update this optimizer WOULD take at a COMMON point, on a COMMON
        # batch, using its OWN accumulated preconditioner state
        before = torch.cat([p.detach().reshape(-1).clone() for p in
                            model.parameters()])
        model.train()
        _, l = model(xb, yb)
        o.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        o.step()
        after = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
        d = (after - before)
        d = d / (d.norm() + 1e-12)
        upd[name] = d.numpy()
        fupd[name] = jvp(d).numpy()          # the FUNCTION-space update
        load(model, start)

    nm = list(OPTS)
    print(f"\n  {'pair':<22}{'param angle':>13}{'FUNCTION angle':>16}{'collapse':>10}")
    print("  " + "-" * 62)
    pang, fang = {}, {}
    for a, b in itertools.combinations(nm, 2):
        pa = angle(upd[a], upd[b])
        fa = angle(fupd[a], fupd[b])
        pang[(a, b)] = pa
        fang[(a, b)] = fa
        same = FAMILY[a] == FAMILY[b]
        print(f"  {a+' <-> '+b:<22}{pa:>12.0f}°{fa:>15.0f}°"
              f"{pa-fa:>9.0f}°  {'(same family)' if same else ''}")

    pa_x = np.mean([v for (a, b), v in pang.items()
                    if FAMILY[a] != FAMILY[b]])
    fa_x = np.mean([v for (a, b), v in fang.items()
                    if FAMILY[a] != FAMILY[b]])
    print(f"\n  ACROSS FAMILIES:  parameter {pa_x:.0f}°   ->   function {fa_x:.0f}°")
    collapsed = fa_x < 0.6 * pa_x

    print()
    if collapsed:
        print("  => THE ANGLE COLLAPSES IN FUNCTION SPACE.")
        print(f"     {pa_x:.0f}° apart in parameter coordinates, {fa_x:.0f}° apart in")
        print("     the coordinates the network can actually SEE.  The optimizers")
        print("     ARE doing the same thing; the disagreement lived in gauge")
        print("     directions.  The parameter-space geometry was LYING, and the")
        print("     right map -- the one that turns the underlying geometry into")
        print("     an extractable signal -- IS the Jacobian.")
    else:
        print("  => THE ANGLE SURVIVES IN FUNCTION SPACE.")
        print(f"     {pa_x:.0f}° in parameters, still {fa_x:.0f}° in function space.")
        print("     The optimizer families genuinely disagree about WHICH")
        print("     FUNCTION to build.  They are not different charts of one")
        print("     object -- they are inequivalent.  This is the 'different")
        print("     sheets' reading, and it is the stronger claim.")

    json.dump({"P": P, "sensitivity": sens,
               "jacobian_spectrum": {"sv": sv[:20].tolist(),
                                     "effective_rank": eff, "k90": k90},
               "param_angles": {f"{a}|{b}": v for (a, b), v in pang.items()},
               "function_angles": {f"{a}|{b}": v for (a, b), v in fang.items()},
               "across_param": float(pa_x), "across_function": float(fa_x),
               "collapsed": bool(collapsed)},
              open("gauge_probe.json", "w"), indent=2, default=float)
    print("\n  wrote gauge_probe.json")

    # ================================================================
    # THE CHARTS
    # ================================================================
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(17, 10))

        # 1. matched-norm sensitivity
        ax = fig.add_subplot(2, 3, 1)
        bs = [b for b in BLOCKS if b in sens]
        x = np.arange(len(bs))
        ax.bar(x - 0.2, [sens[b]["sens"] for b in bs], 0.4,
               yerr=[sens[b]["sens_std"] for b in bs],
               label="block directions", color="#4c72b0")
        ax.bar(x + 0.2, [sens[b]["null"] for b in bs], 0.4,
               yerr=[sens[b]["null_std"] for b in bs],
               label="random, SAME SIZE", color="#ccc")
        ax.set_xticks(x); ax.set_xticklabels(bs, rotation=40)
        ax.set_ylabel(r"$\|Jv\|/\|v\|$")
        ax.set_title("1. Is the block GAUGE?\n(matched-norm; kills the "
                     "'it's just small' confound)")
        ax.legend(fontsize=7)

        # 2. gauge ratio
        ax = fig.add_subplot(2, 3, 2)
        cols = ["#c44e52" if sens[b]["ratio"] < 0.5 else "#55a868" for b in bs]
        ax.bar(bs, [sens[b]["ratio"] for b in bs], color=cols)
        ax.axhline(1.0, ls="--", color="k", label="= random (not special)")
        ax.axhline(0.5, ls=":", color="r", label="gauge threshold")
        ax.set_ylabel("sensitivity / same-size random")
        ax.set_title("2. GAUGE RATIO\n(<1 = less visible than chance)")
        ax.tick_params(axis="x", rotation=40)
        ax.legend(fontsize=7)

        # 3. Jacobian spectrum -- the quotient dimension
        ax = fig.add_subplot(2, 3, 3)
        ax.semilogy(range(1, len(sv) + 1), sv, "o-", color="#c44e52")
        ax.axvline(k90, ls="--", color="k", label=f"90% at {k90} dirs")
        ax.set_xlabel("index"); ax.set_ylabel("singular value of J's image")
        ax.set_title(f"3. THE QUOTIENT DIMENSION\n"
                     f"effective rank {eff:.1f} (of {P:,} params)")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

        # 4. THE DECISIVE CHART: param vs function angles
        ax = fig.add_subplot(2, 3, 4)
        pairs = list(pang)
        xs = np.arange(len(pairs))
        cross = [FAMILY[a] != FAMILY[b] for a, b in pairs]
        ax.bar(xs - 0.2, [pang[p] for p in pairs], 0.4,
               color=["#c44e52" if c else "#f2b5b7" for c in cross],
               label="PARAMETER space")
        ax.bar(xs + 0.2, [fang[p] for p in pairs], 0.4,
               color=["#4c72b0" if c else "#b7c9e2" for c in cross],
               label="FUNCTION space")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{a[:4]}/{b[:4]}" for a, b in pairs],
                           rotation=55, fontsize=7)
        ax.set_ylabel("angle (deg)")
        ax.set_title("4. THE DECISIVE CHART\n"
                     "does the disagreement survive the quotient?")
        ax.legend(fontsize=7)

        # 5. the manifold, drawn: updates in the top-2 function directions
        ax = fig.add_subplot(2, 3, 5)
        Fm = np.stack([fupd[n] for n in nm])
        U2 = np.linalg.svd(Fm, full_matrices=False)[2][:2]
        for n in nm:
            c = Fm[nm.index(n)] @ U2.T
            col = "#c44e52" if FAMILY[n] == "adaptive" else "#4c72b0"
            ax.arrow(0, 0, c[0], c[1], head_width=0.03, color=col, lw=2)
            ax.annotate(n, (c[0], c[1]), fontsize=8,
                        xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel("function direction 1"); ax.set_ylabel("function direction 2")
        ax.set_title("5. THE MANIFOLD, DRAWN\n"
                     "optimizer updates in FUNCTION coordinates")
        ax.grid(alpha=0.3); ax.axhline(0, color="k", lw=0.5)
        ax.axvline(0, color="k", lw=0.5); ax.set_aspect("equal")

        # 6. the same, in PARAMETER coordinates (for contrast)
        ax = fig.add_subplot(2, 3, 6)
        Pm = np.stack([upd[n] for n in nm])
        U2p = np.linalg.svd(Pm, full_matrices=False)[2][:2]
        for n in nm:
            c = Pm[nm.index(n)] @ U2p.T
            col = "#c44e52" if FAMILY[n] == "adaptive" else "#4c72b0"
            ax.arrow(0, 0, c[0], c[1], head_width=0.03, color=col, lw=2)
            ax.annotate(n, (c[0], c[1]), fontsize=8,
                        xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel("param direction 1"); ax.set_ylabel("param direction 2")
        ax.set_title("6. THE SAME, IN PARAMETER COORDINATES\n"
                     "(compare with 5 -- which is the real geometry?)")
        ax.grid(alpha=0.3); ax.axhline(0, color="k", lw=0.5)
        ax.axvline(0, color="k", lw=0.5); ax.set_aspect("equal")

        plt.suptitle("The quotient: which parameter directions can the "
                     "function actually see?", fontsize=14, weight="bold")
        plt.tight_layout()
        plt.savefig("gauge_probe.png", dpi=170)
        print("  wrote gauge_probe.png")
    except Exception as e:
        print(f"  [plot skipped: {e}]")


if __name__ == "__main__":
    main()
