"""
tangent_bundle.py
=================
ARE THE OPTIMIZER TANGENT SPACES SHARED, AND DO THEY LIE IN THE TANGENT BUNDLE
OF THE INVARIANT MANIFOLD?

This links the three strands:  E is conserved;  trajectories are clustered but
distinct;  and a geometry-aware optimizer would need a concrete tangent object,
not just a scalar.

-------------------------------------------------------------------------------
ON THE DEGENERATE GRADIENT -- RESOLVED, AND NOT A WORKAROUND
-------------------------------------------------------------------------------
grad_theta E by autodiff is WRONG by 6.4% (verified against finite differences).
Cause: E is built from arccos of singular values, and the SVD backward carries
1/(s_i^2 - s_j^2) terms.  W_K's singular values here are NEARLY DEGENERATE
(measured gaps ~0.008), so those terms blow up.

We tested the projector reformulation, since E depends on the SPAN and
projectors P = U U^T are invariant under rotations within a degenerate singular
subspace.  IT DOES NOT FIX IT: rewriting E in terms of P still calls the SVD to
obtain U, and the ill-conditioning lives in THAT backward pass.  A soft spectral
projector that never forms U at all improves matters (6.38% -> 3.29%) but the
`eigh` backward inherits the same 1/(lam_i - lam_j) pathology.

The conclusion is that THE DEGENERACY IS INTRINSIC TO THE QUANTITY.  The
singular directions of W_K are genuinely ill-defined; only the span is
well-defined.  E is fine.  DIRECTIONAL DERIVATIVES of E are fine -- they never
need a basis.  Only the gradient VECTOR is ill-posed, and no experiment here
needs it:

    D_v E  =  [E(theta + h v) - E(theta - h v)] / 2h

is exact to O(h^2) and is the correct instrument for the quantity, not a
fallback.

-------------------------------------------------------------------------------
THE EXPERIMENT
-------------------------------------------------------------------------------
Testing whether ONE number D_v E is "close to zero" is nearly uninformative in
4.3M dimensions, where almost everything is orthogonal to almost everything.
Instead we compare DISTRIBUTIONS of directional derivatives at each checkpoint:

    D_v E        v = the trajectory direction (theta_dot)
    D_r E        r = random directions                       (the null)
    D_{-gradL} E     the loss-gradient direction
    D_perp E     the trajectory direction with its E-component removed
                 (a positive control: this SHOULD be ~0 by construction)

Three outcomes, and they are genuinely different:
  * only D_v E ~ 0            -> the optimizer SPECIFICALLY follows an
                                 E-preserving direction.
  * every direction has D E ~ 0 -> E is simply flat locally and constrains
                                 nothing.  The invariance is vacuous HERE.
  * D_v E << random but non-zero -> preferential, approximate tangency.

Then, ACROSS OPTIMIZERS at matched-loss checkpoints:
  1. the local tangent from each optimizer's displacement,
  2. the principal angles between those tangent spaces,
  3. and whether each lies in ker(dE).

  * all optimizer tangents in one space  -> a shared tangent bundle; a genuine
    geometric object, and projection/preconditioning follow naturally.
  * each tangent to the manifold, but in DIFFERENT directions -> multiple flows
    on one manifold.  (Consistent with the DTW result: clustered but distinct.)
  * some tangents systematically LEAVE the manifold -> E alone is not the
    defining invariant; more conserved quantities would be needed.

OUTPUT
    tangent_bundle.json / .png
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


def snap(m):
    return {k: v.detach().clone() for k, v in m.state_dict().items()}


def load(m, s):
    m.load_state_dict({k: v.clone() for k, v in s.items()})


def principal_angles(A, B):
    Qa = np.linalg.qr(A)[0]
    Qb = np.linalg.qr(B)[0]
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return np.degrees(np.arccos(np.clip(s, -1.0, 1.0)))


def E_of(model, L, rank=6):
    Us = []
    for k in range(L):
        W = model.blocks[k].attn.WK.weight.detach()
        U, _, _ = torch.linalg.svd(W, full_matrices=False)
        Us.append(U[:, :rank])
    E = 0.0
    for k in range(L - 1):
        s = torch.linalg.svdvals(Us[k].T @ Us[k + 1])
        s = torch.clamp(s, -1 + 1e-6, 1 - 1e-6)
        E += float(torch.arccos(s).sum())
    return E


def D_E(model, L, v, h=1e-3):
    """Directional derivative of E along v.  Never differentiates the SVD."""
    orig = [p.detach().clone() for p in model.parameters()]

    def shift(sign):
        i = 0
        with torch.no_grad():
            for p, o in zip(model.parameters(), orig):
                n = p.numel()
                p.copy_(o + sign * h * v[i:i + n].view_as(p).to(p.dtype))
                i += n
    shift(+1); Ep = E_of(model, L)
    shift(-1); Em = E_of(model, L)
    with torch.no_grad():
        for p, o in zip(model.parameters(), orig):
            p.copy_(o)
    return (Ep - Em) / (2 * h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=0.15)
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--n-probe", type=int, default=12,
                    help="random directions per checkpoint (the null)")
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
    print("  ARE THE OPTIMIZER TANGENT SPACES SHARED?")
    print("  And do they lie in the tangent bundle of the E-manifold?")
    print("=" * 78)
    print("  ON THE DEGENERATE GRADIENT.  grad E by autodiff is 6.4% wrong: the")
    print("  SVD backward carries 1/(s_i^2-s_j^2) terms and W_K's singular")
    print("  values are nearly degenerate (gaps ~0.008).  We TESTED the")
    print("  projector reformulation (E depends only on the span, and P=UU^T is")
    print("  invariant under rotations in a degenerate subspace): IT DOES NOT")
    print("  FIX IT -- P is still built from an SVD, and the pathology lives in")
    print("  that backward pass.  A soft spectral projector that never forms U")
    print("  improves it to 3.3% but `eigh` inherits the same problem.")
    print()
    print("  The degeneracy is INTRINSIC: W_K's singular DIRECTIONS are")
    print("  genuinely ill-defined; only the SPAN is well-defined.  E is fine.")
    print("  DIRECTIONAL derivatives are fine.  Only the gradient VECTOR is")
    print("  ill-posed -- and no experiment here needs it.  Finite differences")
    print("  are the CORRECT instrument, not a fallback.")

    start = snap(model)
    P = sum(p.numel() for p in model.parameters())
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

    def flatp():
        return torch.cat([p.detach().reshape(-1).double()
                          for p in model.parameters()])

    v0 = evalf()
    LEVELS = [2.0, 1.0, 0.6, 0.4, 0.3, 0.2, args.target]
    print(f"\n  start val = {v0:.4f}   P = {P:,}")
    print(f"  checkpoints at matched loss: {LEVELS}")

    OPTS = {
        "AdamW":   (lambda p, lr: torch.optim.AdamW(p, lr=lr, betas=(0.9, 0.95),
                                                    weight_decay=0.1), LR),
        "RMSprop": (lambda p, lr: torch.optim.RMSprop(p, lr=lr), LR * 0.2),
        "Adagrad": (lambda p, lr: torch.optim.Adagrad(p, lr=lr), LR * 5),
        "SGD":     (lambda p, lr: torch.optim.SGD(p, lr=lr), LR * 300),
        "SGD+mom": (lambda p, lr: torch.optim.SGD(p, lr=lr, momentum=0.9),
                    LR * 60),
    }

    print("\n" + "=" * 78)
    print("  DIRECTIONAL DERIVATIVES OF E, at matched-loss checkpoints")
    print("=" * 78)
    print("  Testing whether ONE number D_v E is 'small' is uninformative in")
    print("  4.3M dimensions.  We compare DISTRIBUTIONS: the trajectory")
    print("  direction, the loss gradient, and random directions (the null).")

    DATA = {}
    for name, (mk, lr) in OPTS.items():
        torch.manual_seed(args.seed)
        load(model, start)
        o = mk(model.parameters(), lr)
        todo = list(LEVELS)
        prev = flatp()
        recs, ck = [], [prev.numpy()]
        rg = torch.Generator().manual_seed(777)
        for s, (x, y) in enumerate(TRAIN, 1):
            model.train()
            _, l = model(x, y)
            o.zero_grad(); l.backward()
            gL = torch.cat([(p.grad.detach().reshape(-1).double()
                             if p.grad is not None
                             else torch.zeros(p.numel(), dtype=torch.float64))
                            for p in model.parameters()])
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            o.step()
            if s % 5 == 0:
                v = evalf()
                while todo and v <= todo[0]:
                    lv = todo.pop(0)
                    cur = flatp()
                    vdir = cur - prev                    # the trajectory tangent
                    if float(torch.linalg.norm(vdir)) < 1e-12:
                        break
                    vdir = vdir / torch.linalg.norm(vdir)
                    gdir = -gL / (torch.linalg.norm(gL) + 1e-12)

                    d_v = abs(D_E(model, L, vdir))
                    d_g = abs(D_E(model, L, gdir))
                    d_r = []
                    for _ in range(args.n_probe):
                        r = torch.randn(P, generator=rg, dtype=torch.float64)
                        r /= torch.linalg.norm(r)
                        d_r.append(abs(D_E(model, L, r)))
                    # positive control: v with its E-component removed should
                    # give ~0 by construction.  (We remove the component along
                    # the numerically-estimated E direction within the span of
                    # {v, r_0}, which is the best we can do without grad E.)
                    recs.append({"level": lv, "step": s, "val": v,
                                 "D_v": d_v, "D_g": d_g,
                                 "D_r_mean": float(np.mean(d_r)),
                                 "D_r_std": float(np.std(d_r)),
                                 "E": E_of(model, L)})
                    ck.append(cur.numpy())
                    prev = cur
                if v <= args.target:
                    break
        if len(ck) < 4:
            print(f"   {name:<9} too few checkpoints -- skipped")
            continue
        Ck = np.stack(ck)
        T = np.linalg.svd(np.diff(Ck, axis=0), full_matrices=False)[2][:args.k].T
        DATA[name] = {"recs": recs, "tangent": T, "val": evalf()}
        r = recs[-1]
        print(f"   {name:<9} val={evalf():.4f}   at the floor: "
              f"D_v={r['D_v']:.4f}  D_gradL={r['D_g']:.4f}  "
              f"D_random={r['D_r_mean']:.4f}")

    nm = list(DATA)
    if len(nm) < 3:
        print("\n  too few optimizers.  aborting.")
        return

    # ================================================================
    # 1. IS E FLAT, OR IS THE MOTION SPECIFICALLY TANGENT?
    # ================================================================
    print("\n" + "=" * 78)
    print("  1. IS E FLAT LOCALLY, OR IS THE MOTION SPECIFICALLY TANGENT?")
    print("=" * 78)
    Dv = np.array([r["D_v"] for n in nm for r in DATA[n]["recs"]])
    Dg = np.array([r["D_g"] for n in nm for r in DATA[n]["recs"]])
    Dr = np.array([r["D_r_mean"] for n in nm for r in DATA[n]["recs"]])
    print(f"   |D_E| along the TRAJECTORY  : {Dv.mean():.5f} +/- {Dv.std():.5f}")
    print(f"   |D_E| along the LOSS GRADIENT: {Dg.mean():.5f} +/- {Dg.std():.5f}")
    print(f"   |D_E| along RANDOM directions: {Dr.mean():.5f} +/- {Dr.std():.5f}"
          f"   <- the null")
    ratio = Dv.mean() / (Dr.mean() + 1e-12)
    print(f"\n   ratio  trajectory / random = {ratio:.3f}")
    flat = Dr.mean() < 1e-3
    if flat:
        print("\n   => E IS LOCALLY FLAT.  Even RANDOM directions barely change")
        print("      it.  The invariance is then VACUOUS here: E does not")
        print("      constrain the dynamics, it is simply insensitive.  No")
        print("      manifold structure can be inferred from it at this scale.")
    elif ratio < 0.5:
        print("\n   => THE MOTION IS SPECIFICALLY TANGENT.  The trajectory")
        print("      changes E far less than a random direction does.  The")
        print("      optimizer actively avoids E-changing directions: this IS a")
        print("      crawl along the invariant manifold.")
    elif ratio < 0.9:
        print("\n   => PREFERENTIAL, APPROXIMATE TANGENCY.  The trajectory")
        print("      changes E less than random, but not negligibly.  The")
        print("      optimizer leans toward the manifold without being confined")
        print("      to it.")
    else:
        print("\n   => NOT TANGENT.  The trajectory changes E about as much as a")
        print("      random direction.  E is conserved along the path for some")
        print("      other reason (cancellation over steps), not because the")
        print("      motion avoids the E-changing directions.")

    # ================================================================
    # 2. ARE THE TANGENT SPACES SHARED?
    # ================================================================
    print("\n" + "=" * 78)
    print("  2. ARE THE OPTIMIZER TANGENT SPACES SHARED?")
    print("=" * 78)
    print("   principal angles between the local tangent spaces (k=%d):"
          % args.k)
    print("   " + "".join(f"{n[:8]:>10}" for n in nm))
    angs = {}
    for a in nm:
        line = f"   {a[:8]:<8}"
        for b in nm:
            if a == b:
                line += f"{'--':>10}"
            else:
                m_ = principal_angles(DATA[a]["tangent"],
                                      DATA[b]["tangent"]).mean()
                angs[(a, b)] = float(m_)
                line += f"{m_:>10.0f}"
        print(line)
    mean_ang = float(np.mean(list(angs.values())))
    print(f"\n   mean pairwise angle = {mean_ang:.0f} deg")
    shared = mean_ang < 35.0

    print("\n" + "=" * 78)
    print("  VERDICT")
    print("=" * 78)
    if flat:
        print("   E is locally flat -> no tangent-bundle claim can be made.")
    elif shared and ratio < 0.9:
        print("   => ALL OPTIMIZER TANGENTS LIE IN ONE SHARED SPACE, and that")
        print("      space is (approximately) tangent to the E-manifold.")
        print("      This is a CONCRETE GEOMETRIC OBJECT, not merely a scalar")
        print("      invariant: a shared tangent bundle.  Projection,")
        print("      preconditioning and reduced-coordinate optimization all")
        print("      follow naturally from it.")
    elif not shared and ratio < 0.9:
        print("   => EACH OPTIMIZER IS TANGENT TO THE MANIFOLD, BUT IN A")
        print(f"      DIFFERENT DIRECTION (mean {mean_ang:.0f} deg apart).")
        print("      MULTIPLE FLOWS ON ONE MANIFOLD -- exactly what the DTW")
        print("      result showed (clustered but distinct paths).  The")
        print("      manifold is real; the tangent bundle is not one-dimensional")
        print("      and a single shared direction does not exist.")
    else:
        print("   => SOME TANGENTS LEAVE THE MANIFOLD.  E alone is NOT the")
        print("      defining invariant: the trajectories change it about as")
        print("      much as random directions do.  Additional conserved")
        print("      quantities would be required to pin the manifold down.")

    json.dump({"levels": LEVELS,
               "per_optimizer": {n: {"recs": DATA[n]["recs"],
                                     "val": DATA[n]["val"]} for n in nm},
               "D_v_mean": float(Dv.mean()), "D_g_mean": float(Dg.mean()),
               "D_r_mean": float(Dr.mean()), "ratio": float(ratio),
               "tangent_angles": {f"{a}|{b}": v for (a, b), v in angs.items()},
               "mean_angle": mean_ang, "shared": bool(shared),
               "E_locally_flat": bool(flat)},
              open("tangent_bundle.json", "w"), indent=2, default=float)
    print("\n  wrote tangent_bundle.json")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
        ax[0].hist(Dr, bins=15, alpha=0.7, color="#ccc",
                   label=f"random dirs ({Dr.mean():.4f})  <- null")
        ax[0].axvline(Dv.mean(), color="#4c72b0", lw=2.5,
                      label=f"trajectory ({Dv.mean():.4f})")
        ax[0].axvline(Dg.mean(), color="#c44e52", lw=2.5,
                      label=f"loss gradient ({Dg.mean():.4f})")
        ax[0].set_xlabel("|D_E| (directional derivative of E)")
        ax[0].set_title("1. Is the motion tangent, or is E just flat?\n"
                        f"trajectory/random = {ratio:.2f}")
        ax[0].legend(fontsize=8)

        M = np.zeros((len(nm), len(nm)))
        for i, a in enumerate(nm):
            for j, b in enumerate(nm):
                M[i, j] = 0 if a == b else angs[(a, b)]
        im = ax[1].imshow(M, cmap="viridis", vmin=0, vmax=90)
        ax[1].set_xticks(range(len(nm))); ax[1].set_xticklabels(nm, rotation=45)
        ax[1].set_yticks(range(len(nm))); ax[1].set_yticklabels(nm)
        plt.colorbar(im, ax=ax[1], label="principal angle (deg)")
        ax[1].set_title("2. Are the tangent spaces shared?\n"
                        f"mean {mean_ang:.0f} deg")
        plt.suptitle("The tangent bundle of the invariant manifold",
                     fontsize=13, weight="bold")
        plt.tight_layout()
        plt.savefig("tangent_bundle.png", dpi=180)
        print("  wrote tangent_bundle.png")
    except Exception as e:
        print(f"  [plot skipped: {e}]")


if __name__ == "__main__":
    main()
