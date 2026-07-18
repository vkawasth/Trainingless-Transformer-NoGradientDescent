"""
intrinsic_subspace.py
=====================
An OPTIMIZER-INDEPENDENT construction, and the experiment that distinguishes
the remaining hypotheses.

WHAT IS ESTABLISHED (and what I over-claimed)
---------------------------------------------
Trajectory-PCA gave a k=3 subspace that recovers 97.6% of AdamW's descent and
transfers across AdamW seeds (88%).  It does NOT survive a change of optimizer
(SGD: 13% recovery, same corpus / arch / init).

I then said the geometric reading was "retired".  That was an OVERSTATEMENT and
I withdraw it.  What the SGD result establishes is narrower:

    TRAJECTORY-DERIVED subspaces are OPTIMIZER-SPECIFIC.

It does NOT establish that there is no intrinsic low-dimensional structure.
Four worlds remain, and my evidence cannot separate the last two:

  WORLD A  one universal basis every optimizer follows.
           -> argued strongly AGAINST by the SGD result.
  WORLD B  each optimizer has its own low-dim subspace.
           -> fully consistent with everything measured.
  WORLD C  there IS an intrinsic (curved) manifold, but trajectory SVD is a
           BIASED estimator of its tangent space: AdamW and SGD traverse
           different geodesics on the same object, so PCA of either gives a
           different basis.  Consistent with everything measured.
  WORLD D  there is no low-dimensional geometric object at all.
           -> also consistent with everything measured.

C and D are the live ones and they are NOT distinguished by any trajectory-
based experiment, because trajectory PCA is exactly the estimator World C says
is biased.  We need a construction that never looks at a trajectory.

THE CONSTRUCTION
----------------
    U* = argmin_{U'U = I}  E_{a~N(0,sigma^2 I)} [ L(w* + Ua) ]

the flattest k-dim subspace through the solution.

HONEST FRAMING OF WHAT U* IS
----------------------------
    In the LOCAL QUADRATIC APPROXIMATION,
        E_a[L(w*+Ua)] = L(w*) + (sigma^2/2) tr(U^T H U) + O(sigma^4),
    so U* reduces to the span of the BOTTOM-k EIGENVECTORS OF H (Ky Fan).
    Outside that regime the equivalence breaks down: higher-order terms matter.

So U* is NOT an exciting new object -- it is the bottom Hessian eigenspace,
computed without ever touching an optimizer trajectory.  That is the ONLY
property we need from it, and it is the whole point: it is a construction
World C predicts should transfer even though trajectory PCA does not.

  * We compute it as an EIGENPROBLEM (subspace iteration on (cI - H) via
    Hessian-vector products).  Validated on ground truth: recovers a known
    flat subspace to principal angles ~1e-6 deg.
  * SPSA/finite-difference optimisation of U was TRIED AND REJECTED: under
    realistic evaluation noise it converges to angles of 81-89 deg from the
    true subspace, i.e. it finds nothing.  Do not use it.
  * The Hessian at the solution is INDEFINITE (negative Rayleigh quotients are
    normal), so the shift must satisfy c > |lambda|_max, not c > lambda_max.
  * RATIOS ARE NEVER REPORTED.  With a near-zero mean curvature, a ratio is
    meaningless (an early draft produced "2,341,438x flatter than random" by
    dividing by ~0).  We report DIFFERENCES and Z-SCORES against a random
    ensemble.

THE EXPERIMENT (what actually decides anything)
-----------------------------------------------
Finding a flat subspace proves nothing -- flat directions are guaranteed in an
overparameterised net.  The content is entirely in three tests:

  T1  Is U* flatter than a RANDOM subspace?   (is flatness SELECTIVE at all?)
      If not, the construction is vacuous and we say so.

  T2  Does U* transfer ACROSS OPTIMIZERS?     <- separates WORLD C from B/D
      Take U* from an AdamW solution; is it still flat at an SGD solution
      (relative to that solution's own random baseline)?
      YES -> the flat geometry is optimizer-independent even though the
             TRAJECTORY is not.  That is World C: trajectory PCA was simply the
             wrong estimator.  The intrinsic-structure hypothesis SURVIVES.
      NO  -> an optimizer-independent construction ALSO fails to transfer.
             Evidence for World D strengthens considerably.

  T3  Does U* transfer ACROSS INDEPENDENT MODELS?  (shared structure, or just
      this net's private flat directions = overparameterisation?)

  Plus: PRINCIPAL ANGLES between U* (flattest) and the AdamW trajectory basis
  (steepest).  These are different objects by construction; if they are near-
  orthogonal, the optimizer travels ACROSS the flat directions, which is itself
  a real finding.

OUTPUT
    intrinsic_subspace.json / .png
"""

import argparse
import json

import numpy as np
import torch


# ---------------------------------------------------------------- Stiefel
def qf(A):
    Q, R = torch.linalg.qr(A)
    s = torch.sign(torch.diagonal(R))
    s[s == 0] = 1.0
    return Q * s


def stiefel_err(U):
    k = U.shape[1]
    return float(torch.linalg.norm(U.T @ U - torch.eye(k, dtype=U.dtype)))


def principal_angles(A, B):
    Qa = np.linalg.qr(A)[0]
    Qb = np.linalg.qr(B)[0]
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return np.degrees(np.arccos(np.clip(s, -1.0, 1.0)))


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--sketch", type=int, default=4000,
                    help="U lives in a FIXED random coordinate sketch; the SAME "
                         "sketch is shared by U*, the random null, and every "
                         "transfer target, so no method gets a special basis")
    ap.add_argument("--iters", type=int, default=60,
                    help="subspace-iteration steps")
    ap.add_argument("--hvp-batches", type=int, default=4,
                    help="batches averaged per HVP (single-batch HVPs are far "
                         "too noisy)")
    ap.add_argument("--n-random", type=int, default=25)
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

    print("=" * 78)
    print("  INTRINSIC SUBSPACE: does an OPTIMIZER-INDEPENDENT construction")
    print("  transfer where the trajectory-derived one did not?")
    print("=" * 78)
    print("  CORRECTION FIRST.  I previously said the SGD result 'retires the")
    print("  geometric reading'.  That was an overstatement, withdrawn.  What")
    print("  it establishes is only: TRAJECTORY-DERIVED subspaces are")
    print("  optimizer-specific.  It does not establish that no intrinsic")
    print("  low-dimensional structure exists.")
    print()
    print("  WORLD B: each optimizer has its own subspace.")
    print("  WORLD C: an intrinsic curved manifold exists, and trajectory PCA")
    print("           is a BIASED estimator of its tangent space (AdamW and SGD")
    print("           walk different geodesics on the same object).")
    print("  WORLD D: there is no low-dimensional object at all.")
    print("  No trajectory-based experiment can separate C from D, because")
    print("  trajectory PCA is precisely the estimator C says is biased.")
    print()
    print("  U*, in the LOCAL QUADRATIC APPROXIMATION, is the bottom-k")
    print("  eigenspace of the Hessian (Ky Fan).  Outside that regime the")
    print("  identity breaks.  It is not a new object -- it is a construction")
    print("  that never touches a trajectory, which is the only property we")
    print("  need.  T2 below is the test that separates World C from D.")

    params = [p for _, p in model.named_parameters()]
    names = [n for n, _ in model.named_parameters()]
    keys = [n for n in names if group_of(n) in GROUPS]
    P = int(sum(p.numel() for n, p in zip(names, params) if n in keys))
    torch.manual_seed(1234)
    d = min(args.sketch, P)
    idx = torch.randperm(P)[:d]
    print(f"\n  P = {P:,}   sketch dim = {d:,} ({100*d/P:.2f}% of P)")
    print("  (the sketch is FIXED and SHARED by every method compared below)")

    def hvp(Vm):
        """Averaged Hessian-vector products, restricted to the sketch.
        Averaging is essential: a single-batch HVP is far too noisy for
        subspace iteration to converge."""
        out = torch.zeros_like(Vm)
        for _ in range(args.hvp_batches):
            x, y = get_batch()
            model.zero_grad(set_to_none=True)
            _, l = model(x, y)
            g = torch.autograd.grad(l, params, create_graph=True)
            for c in range(Vm.shape[1]):
                vec = torch.zeros(P)
                vec[idx] = Vm[:, c].float()
                pd, i = {}, 0
                for n, p in zip(names, params):
                    if n in keys:
                        pd[n] = vec[i:i + p.numel()].view_as(p)
                        i += p.numel()
                dot = sum((gi * pd[n]).sum() for gi, n in zip(g, names)
                          if n in pd)
                h = torch.autograd.grad(dot, params, retain_graph=True)
                acc = torch.cat([hi.reshape(-1) for hi, n in zip(h, names)
                                 if n in keys])
                out[:, c] += acc[idx].double()
        model.zero_grad(set_to_none=True)
        return out / args.hvp_batches

    def mean_curv(U):
        """tr(U^T H U)/k -- the quantity U* minimises.  We report DIFFERENCES
        and Z-SCORES of this, never ratios: the mean curvature can be near zero
        or negative (the Hessian at a neural solution is INDEFINITE), and a
        ratio is then meaningless."""
        return float(torch.diag(U.T @ hvp(U)).mean())

    def flat_subspace(k):
        # spectral radius (both signs) -> safe shift c > |lambda|_max
        v = torch.randn(d, 1, dtype=torch.float64)
        v /= torch.linalg.norm(v)
        for _ in range(12):
            w = hvp(v)
            nw = torch.linalg.norm(w)
            if nw < 1e-14:
                break
            v = w / nw
        lam = float((v.T @ hvp(v)).item())
        c = abs(lam) * 1.5 + 1e-4
        U = qf(torch.randn(d, k, dtype=torch.float64))
        for _ in range(args.iters):
            U = qf(c * U - hvp(U))     # subspace iteration on (cI - H), PD
        return U, c, lam

    def train(opt_name, seed, steps):
        torch.manual_seed(seed)
        load(model, start)
        if opt_name == "adamw":
            o = torch.optim.AdamW(model.parameters(), lr=LR,
                                  betas=(0.9, 0.95), weight_decay=0.1)
        else:
            o = torch.optim.SGD(model.parameters(), lr=LR * 10, momentum=0.9)
        ck = [snap(model)]
        for s in range(1, steps + 1):
            model.train()
            x, y = get_batch()
            _, l = model(x, y)
            o.zero_grad(); l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            o.step()
            if s % 6 == 0:
                ck.append(snap(model))
        return snap(model), ck

    def null_ensemble(k, n):
        out = []
        for i in range(n):
            torch.manual_seed(9000 + i)
            out.append(mean_curv(qf(torch.randn(d, k, dtype=torch.float64))))
        return np.array(out)

    start = snap(model)

    # ---------------- solution A (AdamW) ----------------
    print(f"\n-- solution A: AdamW, {args.steps} steps --")
    wA, ckA = train("adamw", 42, args.steps)
    load(model, wA)
    vA = float(g_["eval_val"](model, n=8))
    UA, cA, lamA = flat_subspace(args.k)
    curvA = mean_curv(UA)
    nullA = null_ensemble(args.k, args.n_random)
    zA = (curvA - nullA.mean()) / (nullA.std() + 1e-12)
    print(f"   val={vA:.4f}   |lambda|_max~{abs(lamA):.5f}   "
          f"stiefel_err={stiefel_err(UA):.1e}")
    print(f"   U*_A mean curvature = {curvA:+.6f}")
    print(f"   random null         = {nullA.mean():+.6f} +/- {nullA.std():.6f}")
    print(f"   z = {zA:+.2f}   (difference {curvA - nullA.mean():+.6f})")

    # ================================================================
    # T1 -- is flatness SELECTIVE?
    # ================================================================
    print("\n" + "=" * 78)
    print("  T1: is U* flatter than a RANDOM subspace at all?")
    print("=" * 78)
    t1 = zA < -2.0
    print(f"   z = {zA:+.2f}  ->  "
          f"{'YES, flatness is selective' if t1 else 'NO -- U* is not distinguishable from random'}")
    if not t1:
        print("   !! The construction is VACUOUS at this net's scale: in an")
        print("      overparameterised model almost every subspace is flat and")
        print("      U* is not special.  Nothing below can rescue that; we")
        print("      report the negative result and stop interpreting.")

    # ================================================================
    # T2 -- ACROSS OPTIMIZERS  (separates World C from B/D)
    # ================================================================
    print("\n" + "=" * 78)
    print("  T2: does U*_A stay flat at an SGD solution?")
    print("      (THE test: World C vs World D)")
    print("=" * 78)
    wS, ckS = train("sgd", 42, args.steps)
    load(model, wS)
    vS = float(g_["eval_val"](model, n=8))
    curv_A_at_S = mean_curv(UA)          # A's flat subspace, evaluated at S
    nullS = null_ensemble(args.k, args.n_random)
    zAS = (curv_A_at_S - nullS.mean()) / (nullS.std() + 1e-12)
    US, _, _ = flat_subspace(args.k)     # SGD solution's OWN flat subspace
    curvS = mean_curv(US)
    zS = (curvS - nullS.mean()) / (nullS.std() + 1e-12)
    ang_AS = principal_angles(UA.numpy(), US.numpy())
    print(f"   SGD solution: val={vS:.4f}")
    print(f"   random null at S    = {nullS.mean():+.6f} +/- {nullS.std():.6f}")
    print(f"   U*_S (S's own flat) = {curvS:+.6f}   z={zS:+.2f}")
    print(f"   U*_A evaluated at S = {curv_A_at_S:+.6f}   z={zAS:+.2f}")
    print(f"   principal angles  U*_A vs U*_S: "
          f"{np.array2string(ang_AS, precision=1)} deg  "
          f"(mean {ang_AS.mean():.1f})")
    t2 = zAS < -2.0
    print(f"\n   -> U*_A is {'STILL FLAT at the SGD solution' if t2 else 'NOT flat at the SGD solution'}")

    # ================================================================
    # T3 -- ACROSS INDEPENDENT MODELS
    # ================================================================
    print("\n" + "=" * 78)
    print("  T3: does U*_A stay flat at an INDEPENDENTLY trained model?")
    print("=" * 78)
    wB, _ = train("adamw", 24680, args.steps)
    load(model, wB)
    vB = float(g_["eval_val"](model, n=8))
    curv_A_at_B = mean_curv(UA)
    nullB = null_ensemble(args.k, args.n_random)
    zAB = (curv_A_at_B - nullB.mean()) / (nullB.std() + 1e-12)
    print(f"   model B: val={vB:.4f}")
    print(f"   random null at B    = {nullB.mean():+.6f} +/- {nullB.std():.6f}")
    print(f"   U*_A evaluated at B = {curv_A_at_B:+.6f}   z={zAB:+.2f}")
    t3 = zAB < -2.0
    print(f"   -> {'TRANSFERS across models' if t3 else 'does NOT transfer across models'}")

    # ================================================================
    # U* (flattest) vs the AdamW trajectory basis (steepest)
    # ================================================================
    print("\n" + "=" * 78)
    print("  U* (flattest) vs the AdamW TRAJECTORY basis (steepest)")
    print("=" * 78)
    def flat_vec(sd):
        return torch.cat([sd[k].reshape(-1).double() for k in keys])
    X = np.stack([(flat_vec(ckA[i]) - flat_vec(ckA[i - 1])).numpy()[idx.numpy()]
                  for i in range(1, len(ckA))])
    Btraj = np.linalg.svd(X, full_matrices=False)[2][:args.k].T
    ang_T = principal_angles(UA.numpy(), Btraj)
    print(f"   principal angles: {np.array2string(ang_T, precision=1)} deg "
          f"(mean {ang_T.mean():.1f})")
    if ang_T.mean() > 60:
        print("   -> NEARLY ORTHOGONAL.  AdamW travels ACROSS the flat")
        print("      directions, not along them.  The trajectory subspace and")
        print("      the loss-flat subspace are different objects.")
    elif ang_T.mean() < 30:
        print("   -> ALIGNED.  AdamW's dominant directions ARE the flat ones.")
    else:
        print("   -> partially aligned.")

    # ================================================================
    # VERDICT -- pre-specified mapping onto the Worlds
    # ================================================================
    print("\n" + "=" * 78)
    print("  VERDICT: WHICH WORLD?")
    print("=" * 78)
    print(f"   T1 flatness selective       : {'YES' if t1 else 'NO'}  (z={zA:+.1f})")
    print(f"   T2 transfers across OPTIMIZER: {'YES' if t2 else 'NO'}  (z={zAS:+.1f})")
    print(f"   T3 transfers across MODEL    : {'YES' if t3 else 'NO'}  (z={zAB:+.1f})")
    print()
    if not t1:
        print("   => VACUOUS.  Flatness is not selective in this net, so U*")
        print("      carries no information.  Neither World C nor D is")
        print("      supported or refuted; the instrument is blunt.  Report")
        print("      that honestly and try a larger k or a smaller net.")
    elif t2 and t3:
        print("   => WORLD C SUPPORTED.  An optimizer-INDEPENDENT construction")
        print("      DOES transfer across optimizers and across models, even")
        print("      though the trajectory-derived subspace did not.  That is")
        print("      exactly the prediction of World C: an intrinsic")
        print("      low-dimensional object exists, and trajectory PCA was a")
        print("      BIASED estimator of its tangent space.")
        print("      The intrinsic-structure hypothesis SURVIVES, and the")
        print("      geometric programme has a substrate after all.")
        print("      STILL NEEDED: more models/seeds; corpus and width changes;")
        print("      and a check that k=3 is not arbitrary.")
    elif t2 and not t3:
        print("   => PARTIAL.  The flat subspace survives an optimizer change")
        print("      but not a change of model.  That points at")
        print("      OVERPARAMETERISATION (each net has private flat")
        print("      directions) rather than shared geometry.")
    elif not t2:
        print("   => WORLD D STRENGTHENED.  Even an OPTIMIZER-INDEPENDENT")
        print("      construction fails to transfer across optimizers.  So it")
        print("      is not merely that trajectory PCA was the wrong estimator")
        print("      -- the flat geometry itself does not agree between")
        print("      solutions.  Evidence against a universal low-dimensional")
        print("      geometric description is now substantially stronger.")
        print("      This is the outcome that would close the programme, and")
        print("      it should be reported as such.")

    json.dump({"k": args.k, "sketch": d, "P": P,
               "A": {"val": vA, "curv": curvA, "z": zA,
                     "null_mean": float(nullA.mean()),
                     "null_std": float(nullA.std())},
               "T1_selective": bool(t1),
               "T2": {"val_sgd": vS, "curv_A_at_S": curv_A_at_S, "z": zAS,
                      "curv_S_own": curvS, "z_S_own": zS,
                      "angles_A_vs_S": ang_AS.tolist(),
                      "transfers": bool(t2)},
               "T3": {"val_B": vB, "curv_A_at_B": curv_A_at_B, "z": zAB,
                      "transfers": bool(t3)},
               "vs_trajectory": {"angles": ang_T.tolist(),
                                 "mean_deg": float(ang_T.mean())}},
              open("intrinsic_subspace.json", "w"), indent=2, default=float)
    print("\n  wrote intrinsic_subspace.json")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(17, 5))
        for a, (nul, val, z, ttl) in zip(ax, [
                (nullA, curvA, zA, "T1: at solution A\n(is flatness selective?)"),
                (nullS, curv_A_at_S, zAS,
                 "T2: U*_A at an SGD solution\n(World C vs D)"),
                (nullB, curv_A_at_B, zAB,
                 "T3: U*_A at another model\n(shared, or private?)")]):
            a.hist(nul, bins=12, color="#ccc", label="random subspaces")
            a.axvline(val, color="#c44e52", lw=2.5, label=f"U*  z={z:+.1f}")
            a.set_xlabel("mean curvature  tr(U'HU)/k")
            a.set_title(ttl, fontsize=10)
            a.legend(fontsize=8)
        plt.suptitle("Optimizer-independent subspace: does it transfer?",
                     fontsize=13, weight="bold")
        plt.tight_layout()
        plt.savefig("intrinsic_subspace.png", dpi=180)
        print("  wrote intrinsic_subspace.png")
    except Exception as e:
        print(f"  [plot skipped: {e}]")


if __name__ == "__main__":
    main()
