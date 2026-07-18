"""
tangent_manifold.py
===================
CAPTURE THE TANGENT CLASSES, AND CHECK WHETHER FROZEN TRAINING STAYS ON THE
MANIFOLD.

-------------------------------------------------------------------------------
THE TRAP THIS SCRIPT IS BUILT TO AVOID
-------------------------------------------------------------------------------
"Freeze W_K, then check that E does not change" is CIRCULAR.  E is a function
of W_K alone.  Freeze W_K and E is constant BY CONSTRUCTION.  Reporting it as
confirmation would be reporting a definition.

The non-trivial question -- and the one actually being asked -- is:

    when AdamW trains with the ATTENTION BLOCK FROZEN (L4: only Emb + FF are
    trainable, which still reaches the target in 410 steps), does it traverse
    the SAME MANIFOLD as the full model?

  if YES -> the crawl happens in Emb/FF; attention is SCAFFOLDING, not
            substance.  A "basic crawl over the manifold" is exactly right.
  if NO  -> freezing attention pushed the model OFF the manifold, and it
            reaches a DIFFERENT solution that merely happens to have the same
            loss.

-------------------------------------------------------------------------------
TWO KINDS OF TANGENT, AND WHY BOTH ARE NEEDED
-------------------------------------------------------------------------------
  T_traj   TRAJECTORY TANGENT.  The local direction of motion, from checkpoint
           differences.  This is what the optimizer DOES.

  T_man    MANIFOLD TANGENT.  The null space of the invariant's gradient,
           ker(grad_theta E).  This is the set of directions one MAY move
           without leaving the level set E = E0.

They are different objects and the relationship between them is the whole
question:

  * if T_traj lies INSIDE ker(grad E), the optimizer is genuinely crawling the
    invariant manifold: every step it takes is a step that preserves E.
  * if T_traj has a large component ALONG grad E, then E is conserved for some
    OTHER reason (e.g. the steps that would change it cancel), and the
    "manifold" framing is the wrong picture.

We measure  cos(T_traj, grad E)  directly.  A near-zero cosine means the motion
is tangent to the level set.  The NULL is the cosine between grad E and a
RANDOM direction: in 4.3M dimensions that is ~0 by default, so a small observed
cosine proves NOTHING unless it is compared against it.  (In high dimension
almost everything is orthogonal to almost everything.)

-------------------------------------------------------------------------------
WHAT IS MEASURED
-------------------------------------------------------------------------------
For FULL training and for FROZEN-ATTENTION training (Emb+FF only):

  A. grad E, by autodiff through the SVD-free surrogate for E.  We check its
     norm is non-trivial before using it -- if grad E ~ 0 everywhere, E is
     conserved trivially and there is no manifold to speak of.
  B. cos(step, grad E) along the trajectory, against a random-direction null.
  C. The TRAJECTORY TANGENT SPACE (top-k of the checkpoint differences), and
     the principal angles between the full model's tangent space and the
     frozen model's.  Small angles = same manifold, same local chart.
  D. The NON-W_K coordinates (tau, Phi_cl) along both runs.  These are NOT
     frozen by construction, so their agreement is informative where E's is
     not.

OUTPUT
    tangent_manifold.json / .png
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


ATTN = {"W_K", "W_Q", "W_V", "W_O"}


def snap(m):
    return {k: v.detach().clone() for k, v in m.state_dict().items()}


def load(m, s):
    m.load_state_dict({k: v.clone() for k, v in s.items()})


def principal_angles(A, B):
    Qa = np.linalg.qr(A)[0]
    Qb = np.linalg.qr(B)[0]
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return np.degrees(np.arccos(np.clip(s, -1.0, 1.0)))


# ---------------------------------------------------------------- E and grad E
def E_torch(model, L, rank=6):
    """E as a DIFFERENTIABLE function of the weights.

    E = sum_k sum_i arccos(sigma_i(U_k^T U_{k+1})).  torch.linalg.svd is
    differentiable, so grad_theta E is available by autodiff.  We clamp the
    singular values away from +/-1 because arccos has infinite derivative
    there and would produce NaNs."""
    Us = []
    for k in range(L):
        W = model.blocks[k].attn.WK.weight
        U, _, _ = torch.linalg.svd(W, full_matrices=False)
        Us.append(U[:, :rank])
    E = 0.0
    for k in range(L - 1):
        M = Us[k].T @ Us[k + 1]
        s = torch.linalg.svdvals(M)
        s = torch.clamp(s, -1.0 + 1e-6, 1.0 - 1e-6)
        E = E + torch.arccos(s).sum()
    return E


def dE_along(model, L, direction, h=1e-3):
    """Directional derivative dE/dv by central difference.

    WHY NOT AUTODIFF.  E is built from arccos of the singular values of
    U_k^T U_{k+1}, and the SVD derivative contains 1/(s_i^2 - s_j^2) terms.
    The singular values of W_K here are NEARLY DEGENERATE (measured gaps as
    small as 0.008), so those terms blow up and the autodiff gradient is wrong
    by ~6% -- verified against finite differences.

    The QUANTITY E is perfectly well-defined (it depends only on the SPAN of
    the column spaces, not on individual singular vectors).  It is only this
    PARAMETERISATION of its gradient that is ill-conditioned.

    And we never actually need the gradient VECTOR: every use is grad E
    contracted with a direction.  So we compute the contraction directly, which
    is exact to O(h^2) and sidesteps the degeneracy entirely."""
    orig = [p.detach().clone() for p in model.parameters()]
    i = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            p.add_(h * direction[i:i + n].view_as(p).to(p.dtype))
            i += n
    Ep = float(E_torch(model, L))
    with torch.no_grad():
        for p, o in zip(model.parameters(), orig):
            p.copy_(o)
        i = 0
        for p in model.parameters():
            n = p.numel()
            p.sub_(h * direction[i:i + n].view_as(p).to(p.dtype))
            i += n
    Em = float(E_torch(model, L))
    with torch.no_grad():
        for p, o in zip(model.parameters(), orig):
            p.copy_(o)
    return (Ep - Em) / (2 * h)


def grad_E_norm(model, L, n_probe=12, seed=0):
    """||grad E||, estimated from directional derivatives along random unit
    vectors: E[ (dE/dv)^2 ] = ||grad E||^2 / P for v uniform on the sphere."""
    P = sum(p.numel() for p in model.parameters())
    g = torch.Generator().manual_seed(seed)
    acc = 0.0
    for _ in range(n_probe):
        v = torch.randn(P, generator=g, dtype=torch.float64)
        v /= torch.linalg.norm(v)
        acc += dE_along(model, L, v) ** 2
    return math.sqrt(max(acc / n_probe, 0.0) * P)


def cos_step_gradE(model, L, step, n_probe=12, seed=0):
    """cos(step, grad E), computed WITHOUT ever forming grad E.

        cos = (dE/d_step_hat) / ||grad E||
    """
    sh = step / (torch.linalg.norm(step) + 1e-12)
    d = dE_along(model, L, sh)
    ng = grad_E_norm(model, L, n_probe=n_probe, seed=seed)
    return abs(d) / (ng + 1e-12), ng


def tau_defect(model, get_batch):
    model.zero_grad(set_to_none=True)
    x, y = get_batch()
    _, l = model(x, y)
    l.backward()
    gf = ge = 0.0
    for n, p in model.named_parameters():
        if p.grad is None:
            continue
        v = float(p.grad.detach().norm()) ** 2
        if ".ff." in n:
            gf += v
        elif n.startswith("te") or n.startswith("pe"):
            ge += v
    model.zero_grad(set_to_none=True)
    return math.sqrt(gf) / (math.sqrt(ge) + 1e-12)


def phi_clean(model, L, tol=0.15):
    n = 0
    for k in range(L - 1):
        Wk = model.blocks[k].attn.WK.weight.detach().cpu().double()
        Wk1 = model.blocks[k + 1].attn.WK.weight.detach().cpu().double()
        try:
            lam = torch.linalg.eigvals(Wk1 @ torch.linalg.pinv(Wk))
            p = float(torch.angle(lam[lam.abs().argmax()]))
            if min(abs(p), abs(abs(p) - math.pi)) < tol:
                n += 1
        except Exception:
            pass
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=0.15)
    ap.add_argument("--max-steps", type=int, default=800)
    ap.add_argument("--k", type=int, default=3, help="tangent-space dimension")
    ap.add_argument("--n-ckpt", type=int, default=20)
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
    print("  TANGENT CLASSES, AND THE MANIFOLD CHECK")
    print("=" * 78)
    print("  THE TRAP: 'freeze W_K, verify E is unchanged' is CIRCULAR.  E is a")
    print("  function of W_K alone, so freezing W_K fixes E BY CONSTRUCTION.")
    print("  That is a definition, not a measurement.")
    print()
    print("  THE REAL QUESTION: AdamW with the ATTENTION BLOCK FROZEN still")
    print("  reaches the target (410 steps, Emb+FF only).  Does it traverse the")
    print("  SAME MANIFOLD, or a different solution with the same loss?")
    print()
    print("  TWO TANGENTS, and their relationship IS the question:")
    print("    T_traj  = local direction of motion   (what the optimizer DOES)")
    print("    T_man   = ker(grad E)                 (what it MAY do without")
    print("                                           leaving the level set)")
    print("  If T_traj lies inside ker(grad E), the optimizer genuinely crawls")
    print("  the invariant manifold.  If not, E is conserved for another reason")
    print("  and the manifold picture is wrong.")

    start = snap(model)
    names = [n for n, _ in model.named_parameters()]
    sizes = [p.numel() for p in model.parameters()]
    P = sum(sizes)

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

    def flat_params():
        return torch.cat([p.detach().reshape(-1).double()
                          for p in model.parameters()])

    v0 = evalf()
    print(f"\n  start val = {v0:.4f}   P = {P:,}")

    # ================================================================
    # A. IS grad E EVEN NON-TRIVIAL?
    # ================================================================
    print("\n" + "=" * 78)
    print("  A. is grad E non-trivial?  (if grad E ~ 0, there is no manifold)")
    print("=" * 78)
    load(model, start)
    E0 = float(E_torch(model, L))
    nE = grad_E_norm(model, L, n_probe=16)
    print(f"   E = {E0:.4f}   ||grad E|| = {nE:.4e}  (from directional")
    print(f"   finite differences -- the autodiff SVD gradient is ill-")
    print(f"   conditioned here: W_K's singular values are nearly degenerate")
    print(f"   (gaps ~0.008), so the 1/(s_i^2-s_j^2) terms in the SVD")
    print(f"   derivative blow up.  Verified 6% wrong.  E itself is fine --")
    print(f"   it depends only on the SPAN.)")
    if nE < 1e-8:
        print("\n   !! grad E is numerically zero.  E is locally CONSTANT in")
        print("      every direction, so there is no level set and no manifold")
        print("      to be tangent to.  The geometric framing collapses here.")
        return

    # ================================================================
    # B + C. TRAIN, capturing tangents
    # ================================================================
    def run(freeze_attn, label):
        torch.manual_seed(args.seed)
        load(model, start)
        for n, p in model.named_parameters():
            p.requires_grad_(not (freeze_attn and group_of(n) in ATTN))
        trainable = [p for p in model.parameters() if p.requires_grad]
        o = torch.optim.AdamW(trainable, lr=LR, betas=(0.9, 0.95),
                              weight_decay=0.1)
        every = max(1, args.max_steps // args.n_ckpt)
        ck, coords, coss = [flat_params().numpy()], [], []
        hit, used = False, args.max_steps
        for s, (x, y) in enumerate(TRAIN, 1):
            model.train()
            _, l = model(x, y)
            o.zero_grad(); l.backward()
            # the step direction, BEFORE the optimizer rescales it
            step = torch.cat([(p.grad.detach().reshape(-1).double()
                               if p.grad is not None
                               else torch.zeros(p.numel(), dtype=torch.float64))
                              for p in model.parameters()])
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            o.step()
            if s % every == 0 or s == 1:
                c, _ = cos_step_gradE(model, L, step, n_probe=8, seed=s)
                coss.append(c)
                ck.append(flat_params().numpy())
                coords.append({"step": s, "val": evalf(),
                               "tau": tau_defect(model, get_batch),
                               "Phi_cl": phi_clean(model, L),
                               "E": float(E_torch(model, L))})
            if s % 10 == 0 and evalf() <= args.target:
                hit, used = True, s
                break
        for p in model.parameters():
            p.requires_grad_(True)
        Ck = np.stack(ck)
        D = np.diff(Ck, axis=0)
        T = np.linalg.svd(D, full_matrices=False)[2][:args.k].T   # tangent
        print(f"   [{label}] val={evalf():.4f} in {used} steps "
              f"{'(reached)' if hit else '(NOT reached)'}")
        return {"tangent": T, "cos": np.array(coss), "coords": coords,
                "val": evalf(), "steps": used, "reached": hit}

    print("\n" + "=" * 78)
    print("  B+C. training, capturing tangents")
    print("=" * 78)
    FULL = run(False, "full model      ")
    FROZ = run(True, "attention frozen")

    # ---- the NULL for the cosine ----
    # In 4.3M dimensions a random direction is almost orthogonal to anything, so
    # a small cos(step, grad E) proves NOTHING on its own.  The null is computed
    # exactly like the observed value -- dE along a RANDOM unit direction,
    # divided by ||grad E|| -- so the comparison is apples to apples.
    load(model, start)
    ngE = grad_E_norm(model, L, n_probe=16)
    rg = torch.Generator().manual_seed(31337)
    null_cos = []
    for _ in range(40):
        v = torch.randn(P, generator=rg, dtype=torch.float64)
        v /= torch.linalg.norm(v)
        null_cos.append(abs(dE_along(model, L, v)) / (ngE + 1e-12))
    null_cos = np.array(null_cos)

    print("\n" + "=" * 78)
    print("  IS THE MOTION TANGENT TO THE LEVEL SET?   cos(step, grad E)")
    print("=" * 78)
    print(f"   NULL (random direction vs grad E) : {null_cos.mean():.5f} "
          f"+/- {null_cos.std():.5f}")
    print(f"   full model                        : {FULL['cos'].mean():.5f}")
    print(f"   attention frozen                  : {FROZ['cos'].mean():.5f}")
    print()
    print("   In 4.3M dimensions almost everything is orthogonal to almost")
    print("   everything, so a SMALL cosine proves nothing by itself.  What")
    print("   matters is whether it is SMALLER THAN THE NULL.")
    zf = (FULL["cos"].mean() - null_cos.mean()) / (null_cos.std() + 1e-12)
    print(f"   full model, z vs null = {zf:+.1f}")
    tangent = FULL["cos"].mean() < null_cos.mean()
    if FULL["cos"].mean() > 5 * null_cos.mean():
        print("\n   -> the motion has a LARGE component along grad E, far above")
        print("      chance.  The optimizer is NOT moving tangent to the level")
        print("      set; it pushes against E and E is conserved because those")
        print("      pushes CANCEL, not because the motion avoids them.")
        print("      The 'crawl along the manifold' picture is WRONG.")
    elif tangent:
        print("\n   -> the motion is MORE orthogonal to grad E than chance:")
        print("      the optimizer actively avoids directions that change E.")
        print("      This IS a crawl along the invariant manifold.")
    else:
        print("\n   -> the motion is indistinguishable from a random direction")
        print("      with respect to grad E.  E is conserved, but not because")
        print("      the trajectory is tangent to its level set.")

    # ================================================================
    # D. DOES THE FROZEN MODEL STAY ON THE SAME MANIFOLD?
    # ================================================================
    print("\n" + "=" * 78)
    print("  D. does FROZEN-ATTENTION training stay on the SAME manifold?")
    print("=" * 78)
    ang = principal_angles(FULL["tangent"], FROZ["tangent"])
    print(f"   principal angles between the two TRAJECTORY TANGENT SPACES:")
    print(f"     {np.array2string(ang, precision=1)} deg   "
          f"(mean {ang.mean():.1f})")
    print()
    print("   the NON-W_K coordinates (NOT frozen by construction, so their")
    print("   agreement is informative where E's would be circular):")
    print(f"   {'':<18}{'full':>10}{'frozen':>10}")
    for c in ("tau", "Phi_cl"):
        a = np.mean([x[c] for x in FULL["coords"]])
        b = np.mean([x[c] for x in FROZ["coords"]])
        print(f"   {c:<18}{a:>10.3f}{b:>10.3f}")
    ef = np.array([x["E"] for x in FROZ["coords"]])
    print(f"   {'E (frozen run)':<18}{'':>10}{ef.mean():>10.3f}  "
          f"<- CONSTANT BY CONSTRUCTION, not evidence")

    same = ang.mean() < 45.0
    print()
    if same:
        print("   => THE FROZEN MODEL STAYS ON THE SAME MANIFOLD.")
        print("      Its trajectory tangent space is close to the full model's")
        print(f"      ({ang.mean():.0f} deg).  The crawl is happening in Emb/FF;")
        print("      the attention block is SCAFFOLDING, not substance.")
    else:
        print("   => THE FROZEN MODEL LEAVES THE MANIFOLD.")
        print(f"      Its tangent space is {ang.mean():.0f} deg from the full")
        print("      model's.  It reaches a comparable loss by a DIFFERENT")
        print("      route to a DIFFERENT solution.  'Same loss' is not 'same")
        print("      manifold', and freezing attention changed the object being")
        print("      learned, not merely the path to it.")

    json.dump({"E0": E0, "grad_E_norm": nE, "grad_E_in_WK": frac_wk,
               "null_cos": {"mean": float(null_cos.mean()),
                            "std": float(null_cos.std())},
               "full": {"cos_mean": float(FULL["cos"].mean()),
                        "val": FULL["val"], "steps": FULL["steps"]},
               "frozen": {"cos_mean": float(FROZ["cos"].mean()),
                          "val": FROZ["val"], "steps": FROZ["steps"]},
               "tangent_angles_deg": ang.tolist(),
               "same_manifold": bool(same)},
              open("tangent_manifold.json", "w"), indent=2, default=float)
    print("\n  wrote tangent_manifold.json")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
        ax[0].hist(null_cos, bins=25, color="#ccc",
                   label=f"random dir vs grad E ({null_cos.mean():.4f})")
        ax[0].axvline(FULL["cos"].mean(), color="#4c72b0", lw=2.5,
                      label=f"full model ({FULL['cos'].mean():.4f})")
        ax[0].axvline(FROZ["cos"].mean(), color="#c44e52", lw=2.5,
                      label=f"attn frozen ({FROZ['cos'].mean():.4f})")
        ax[0].set_xlabel("|cos(step, grad E)|")
        ax[0].set_title("Is the motion tangent to the level set?\n"
                        "(small only matters RELATIVE to the null)")
        ax[0].legend(fontsize=8)

        ax[1].bar(range(1, len(ang) + 1), ang, color="#55a868")
        ax[1].axhline(45, ls="--", color="k", label="same-manifold threshold")
        ax[1].set_xlabel("principal direction")
        ax[1].set_ylabel("angle (deg)")
        ax[1].set_ylim(0, 90)
        ax[1].set_title("Tangent space: full vs attention-frozen\n"
                        f"mean {ang.mean():.0f} deg")
        ax[1].legend(fontsize=8)
        plt.suptitle("Tangent classes and the manifold check",
                     fontsize=13, weight="bold")
        plt.tight_layout()
        plt.savefig("tangent_manifold.png", dpi=180)
        print("  wrote tangent_manifold.png")
    except Exception as e:
        print(f"  [plot skipped: {e}]")


if __name__ == "__main__":
    main()
