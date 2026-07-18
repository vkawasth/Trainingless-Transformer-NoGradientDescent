"""
shared_residual.py
==================
DECOMPOSING  T_i = T_shared + R_i   AND SEPARATING THREE HYPOTHESES.

WHAT IS ESTABLISHED
-------------------
Controlled run (same init, same batch sequence, same eval set; per-optimizer LR):

    optimizer    k=3 recovery of its OWN descent
    AdamW              99%
    SGD+mom           103%
    RMSprop            89%
    Adagrad            82%
    SGD                40%   <- plateaus; k=5 buys nothing

    pairwise angles: AdamW<->RMSprop 37deg, RMSprop<->Adagrad 58deg,
                     AdamW<->Adagrad 63deg, but AdamW<->SGD 79deg.
    union spectrum: cliff at 3 but gap only 1.17x; effective dim 10.8 of 15.

I read this as "the subspace is a property of the PRECONDITIONER" -- i.e.
Hypothesis A.  THAT WAS OVER-READING.  Three hypotheses predict these data
equally well:

  A  THE OPTIMIZER CREATES the low-dimensionality.
     The preconditioner steers optimisation into a few dominant directions.

  B  THE OPTIMIZER REVEALS existing structure.
     The landscape has a low-dimensional valley; adaptive methods align with
     it quickly, plain SGD wanders and spends its trajectory correcting
     orthogonal errors.  The valley is intrinsic either way.

  C  SGD TRAJECTORIES HAVE HIGHER CURVATURE.
     AdamW takes a smooth arc (3 vectors approximate it); SGD zig-zags on
     noisy gradients.  The VISITED WEIGHTS may lie near the SAME valley, but
     the PATH LENGTH is far larger, so trajectory PCA needs many components.
     In that case PCA UNDERESTIMATES the underlying manifold.

My angle observation ("adaptive optimizers cluster") is explained just as well
by "adaptive methods align with the valley similarly" (B) as by "adaptive
preconditioners resemble each other" (A).  It does not discriminate.

THE DECOMPOSITION (your proposal)
---------------------------------
  1. Build a SHARED basis from the COMPRESSIBLE optimizers only
     (AdamW, SGD+mom, RMSprop, Adagrad).
  2. For every optimizer, measure how much of its trajectory lies in that
     shared basis, and how large the residual R_i is.
  3. Ask: are R_AdamW, R_RMSprop, R_Adagrad small while R_SGD is huge?

That gives  shared (+) optimizer-specific, which is a much stronger statement
than "SGD doesn't compress".  It also explains the union spectrum directly: a
visible cliff at ~3 with effective dim ~11 is exactly what you get when four
optimizers share a dominant subspace and one contributes many private
directions.

THE DISCRIMINATOR THAT SEPARATES A FROM B/C
-------------------------------------------
The decomposition alone does NOT separate A from B/C.  The key move is to
distinguish the PATH from the ENDPOINT:

    PATH      = the sequence of steps (what trajectory PCA sees)
    ENDPOINT  = the total displacement  theta_floor - theta_start

Hypothesis C says precisely that these come apart: SGD's PATH is a zig-zag
that no 3 vectors capture, yet its ENDPOINT may sit squarely in the same
valley.  So:

    Q: what fraction of SGD's total DISPLACEMENT lies inside the shared basis
       built from the ADAPTIVE optimizers only?

       HIGH (vs a random-subspace null)
            -> SGD ends up in the SAME low-dim structure; only its path is
               noisy.  The subspace is INTRINSIC and trajectory PCA is simply
               a biased estimator for SGD.  ==> B or C.  World C survives.
       LOW  (indistinguishable from the null)
            -> SGD goes somewhere else entirely.  The structure is not shared.
               ==> A (the optimizer creates it) or D.

A RANDOM-SUBSPACE NULL IS MANDATORY.  A displacement will land partly in ANY
subspace by chance; the fraction only means something relative to chance.
Everything is measured CAUSALLY where possible (project -> transplant ->
evaluate), never fitted.

OUTPUT
    shared_residual.json / .png
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
ADAPTIVE = {"AdamW", "RMSprop", "Adagrad"}


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


def frac_in(T, B):
    """Fraction of the norm of T that lies inside span(B)."""
    Q = np.linalg.qr(B)[0]
    return float(np.linalg.norm(Q @ (Q.T @ T)) / (np.linalg.norm(T) + 1e-12))


def path_length(steps):
    return float(np.sum(np.linalg.norm(steps, axis=1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--shared-dim", type=int, default=6,
                    help="dimension of the SHARED basis built from the "
                         "compressible optimizers")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--n-ckpt", type=int, default=30)
    ap.add_argument("--n-null", type=int, default=40)
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
    print("  SHARED (+) OPTIMIZER-SPECIFIC:  T_i = T_shared + R_i")
    print("=" * 78)
    print("  I previously read 'adaptive optimizers cluster' as 'the subspace")
    print("  is a property of the preconditioner'.  That was OVER-READING.")
    print("  Three hypotheses fit the data equally:")
    print("    A the optimizer CREATES the low-dimensionality")
    print("    B the optimizer REVEALS an intrinsic valley")
    print("    C SGD lies near the SAME valley but ZIG-ZAGS, so trajectory PCA")
    print("      underestimates the manifold (path != endpoint)")
    print()
    print("  DISCRIMINATOR: separate the PATH from the ENDPOINT.")
    print("  Hypothesis C says these come apart.  So we ask what fraction of")
    print("  SGD's total DISPLACEMENT lies in the shared basis built from the")
    print("  ADAPTIVE optimizers -- against a RANDOM-SUBSPACE NULL.")
    print("    high -> SGD lands in the same structure; path just noisy (B/C)")
    print("    low  -> SGD goes elsewhere; structure not shared (A/D)")

    start = snap(model)
    keys = [k for k in start if group_of(k) in GROUPS]

    # controlled: identical init, batch stream and eval set for every optimizer
    torch.manual_seed(args.seed)
    TRAIN = [get_batch() for _ in range(args.steps)]
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
    print(f"\n  start val = {v0:.4f}  (identical init/batches/eval for all)")

    OPTS = {
        "AdamW":   (lambda p, lr: torch.optim.AdamW(p, lr=lr, betas=(0.9, 0.95),
                                                    weight_decay=0.1), LR),
        "SGD":     (lambda p, lr: torch.optim.SGD(p, lr=lr), LR * 300),
        "SGD+mom": (lambda p, lr: torch.optim.SGD(p, lr=lr, momentum=0.9),
                    LR * 60),
        "RMSprop": (lambda p, lr: torch.optim.RMSprop(p, lr=lr), LR * 0.2),
        "Adagrad": (lambda p, lr: torch.optim.Adagrad(p, lr=lr), LR * 5),
    }

    print("\n-- training each optimizer on the identical stream --")
    D = {}
    every = max(1, args.steps // args.n_ckpt)
    for name, (mk, lr) in OPTS.items():
        torch.manual_seed(args.seed)
        load(model, start)
        o = mk(model.parameters(), lr)
        ck = [snap(model)]
        for s, (x, y) in enumerate(TRAIN, 1):
            model.train()
            _, l = model(x, y)
            o.zero_grad(); l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            o.step()
            if s % every == 0 or s == args.steps:
                ck.append(snap(model))
        load(model, ck[-1])
        vf = evalf()
        X = np.stack([(flat(ck[i], keys) - flat(ck[i - 1], keys)).numpy()
                      for i in range(1, len(ck))])
        T = (flat(ck[-1], keys) - flat(start, keys)).numpy()
        B = np.linalg.svd(X, full_matrices=False)[2][:args.k].T
        D[name] = {"floor": vf, "descent": v0 - vf, "steps": X, "T": T,
                   "B": B, "own": frac_in(T, B),
                   "path_len": path_length(X),
                   "disp_len": float(np.linalg.norm(T))}
        print(f"   {name:<9} floor={vf:.4f}  descent={v0-vf:.3f}  "
              f"own-basis captures {100*D[name]['own']:.0f}% of its displacement")

    P = D["AdamW"]["T"].size

    # ================================================================
    # PATH vs ENDPOINT -- the heart of Hypothesis C
    # ================================================================
    print("\n" + "=" * 78)
    print("  PATH vs ENDPOINT  (Hypothesis C says these come apart)")
    print("=" * 78)
    print(f"  {'optimizer':<10}{'path length':>13}{'displacement':>14}"
          f"{'ratio':>8}   interpretation")
    print("  " + "-" * 74)
    for n, d in D.items():
        r = d["path_len"] / (d["disp_len"] + 1e-12)
        tag = ("direct" if r < 2 else
               "wandering" if r < 5 else "ZIG-ZAG (path >> displacement)")
        print(f"  {n:<10}{d['path_len']:>13.2f}{d['disp_len']:>14.2f}"
              f"{r:>8.1f}   {tag}")
    print("\n  A large ratio means the optimizer travelled far to get somewhere")
    print("  close -- exactly the zig-zag of Hypothesis C.  If SGD's ratio is")
    print("  large AND its endpoint still lands in the shared subspace, then")
    print("  trajectory PCA underestimates the manifold for SGD.")

    # ================================================================
    # THE SHARED BASIS  (from the ADAPTIVE optimizers only)
    # ================================================================
    print("\n" + "=" * 78)
    print("  SHARED BASIS from the ADAPTIVE optimizers (AdamW/RMSprop/Adagrad)")
    print("=" * 78)
    print("  SGD is DELIBERATELY EXCLUDED from building it -- otherwise the")
    print("  test would be circular.  SGD+mom is also excluded so the basis is")
    print("  purely 'adaptive', making the transfer test as strict as possible.")
    Bad = np.concatenate([D[n]["B"] for n in ADAPTIVE if n in D], axis=1)
    Ushared = np.linalg.svd(Bad, full_matrices=False)[0][:, :args.shared_dim]
    print(f"   shared basis dim = {Ushared.shape[1]}")

    # random-subspace NULL -- mandatory
    print(f"\n  building a random-subspace null (n={args.n_null}) ...")
    rng = np.random.default_rng(0)
    null = {n: [] for n in D}
    for _ in range(args.n_null):
        R = rng.normal(size=(P, args.shared_dim))
        R = np.linalg.qr(R)[0]
        for n, d in D.items():
            null[n].append(frac_in(d["T"], R))
    null = {n: np.array(v) for n, v in null.items()}

    print("\n" + "=" * 78)
    print("  DECOMPOSITION  T_i = T_shared + R_i")
    print("=" * 78)
    # NOTE ON THE NULL.  A random d-dim subspace of R^P captures only
    # ~sqrt(d/P) of any vector -- here sqrt(6/4.2e6) = 0.0012, with a std near
    # zero.  So a z-score divides by ~0 and explodes (an earlier version
    # printed z=+1245, which is meaningless).  We report the RATIO to the
    # analytic chance level instead, which is stable and interpretable.
    chance = float(np.sqrt(args.shared_dim / P))
    print(f"  chance level for a random {args.shared_dim}-dim subspace of "
          f"R^{P:,} = {chance:.4f}")
    print(f"  (z-scores against this null are NOT reported: its std is ~0, so")
    print(f"   any z explodes.  We report the ratio to chance.)\n")
    print(f"  {'optimizer':<10}{'in shared':>11}{'residual':>10}"
          f"{'x chance':>10}   used to build basis?")
    print("  " + "-" * 68)
    dec = {}
    for n, d in D.items():
        f = frac_in(d["T"], Ushared)
        ratio = f / (chance + 1e-12)
        inval = ratio > 20.0          # far above chance
        dec[n] = {"in_shared": f, "residual": 1 - f, "chance": chance,
                  "x_chance": ratio, "far_above_chance": bool(inval),
                  "excluded_from_basis": n not in ADAPTIVE}
        star = "yes" if n in ADAPTIVE else "NO  <- held out"
        print(f"  {n:<10}{f:>10.2f}{1-f:>10.2f}{ratio:>9.0f}x   {star}")

    # ================================================================
    # CAUSAL SUFFICIENCY  (the decisive experiment)
    # ================================================================
    # CRITICAL DISTINCTION, which an earlier version of this script conflated:
    #
    #   GEOMETRIC OVERLAP  = fraction of the displacement VECTOR's norm that
    #                        lies inside the basis.
    #   CAUSAL SUFFICIENCY = how much of the LOSS REDUCTION you recover when
    #                        you actually APPLY that projection.
    #
    # They come apart badly.  SGD's own top-3 basis captures 96% of its
    # displacement NORM, yet transplanting that projection recovers only 40%
    # of its loss reduction.  A projection can hold most of the norm while
    # missing the directions that MOVE THE LOSS.
    #
    # So: project SGD's displacement onto the ADAPTIVE shared basis, apply
    # ONLY that, and measure the loss.  If a 36% geometric overlap yields
    # 70-90% of SGD's improvement, the shared directions are functionally
    # central.  If it yields ~36%, the overlap is merely descriptive.
    print("\n" + "=" * 78)
    print("  CAUSAL SUFFICIENCY: apply ONLY the shared component, measure loss")
    print("=" * 78)
    print("  geometric overlap != causal sufficiency.  (SGD's own basis holds")
    print("  96% of its displacement NORM but recovers only 40% of its LOSS.)")
    print()
    print(f"  {'optimizer':<10}{'geom overlap':>14}{'val after':>11}"
          f"{'loss recovered':>16}   verdict")
    print("  " + "-" * 72)
    for n, d in D.items():
        Q = np.linalg.qr(Ushared)[0]
        proj = Q @ (Q.T @ d["T"])          # ONLY the shared component
        sd_ = {a: b.clone() for a, b in start.items()}
        pd_ = unflat(torch.tensor(proj), start, keys)
        for a in keys:
            sd_[a] = (start[a] + pd_[a]).clone()
        load(model, sd_)
        v = evalf()
        rec = (v0 - v) / max(d["descent"], 1e-9)
        dec[n]["causal_val"] = v
        dec[n]["causal_recovery"] = rec
        geo = dec[n]["in_shared"]
        if rec > geo + 0.20:
            verd = "shared dirs are FUNCTIONALLY CENTRAL"
        elif rec < geo - 0.20:
            verd = "shared dirs are functionally WEAK"
        else:
            verd = "recovery ~ overlap (descriptive)"
        print(f"  {n:<10}{geo:>13.2f}{v:>11.4f}{100*rec:>15.0f}%   {verd}")
    load(model, start)
    sgd_geo = dec["SGD"]["in_shared"]
    sgd_rec = dec["SGD"]["causal_recovery"]
    print(f"\n  THE KEY NUMBER -- SGD:")
    print(f"    geometric overlap with the ADAPTIVE shared basis: {sgd_geo:.2f}")
    print(f"    loss reduction recovered by applying ONLY that : "
          f"{100*sgd_rec:.0f}%")
    if sgd_rec > 0.70:
        print("    => the shared directions are DISPROPORTIONATELY IMPORTANT.")
        print("       A minority of SGD's displacement, lying in a basis built")
        print("       WITHOUT SGD, carries most of its improvement.  That is")
        print("       strong evidence for a shared, functionally central,")
        print("       optimizer-independent structure.")
    elif sgd_rec > sgd_geo + 0.20:
        print("    => the shared directions are OVER-REPRESENTED in the loss:")
        print("       they carry more improvement than their share of the norm.")
        print("       Suggestive, but short of causal sufficiency.")
    else:
        print("    => recovery tracks the overlap.  The shared component is")
        print("       DESCRIPTIVE, not functionally central.  A shared subspace")
        print("       that does not carry the improvement is not the")
        print("       optimizer-independent object we are looking for.")

    # ================================================================
    # THE VERDICT
    # ================================================================
    print("\n" + "=" * 78)
    print("  WHICH HYPOTHESIS?")
    print("=" * 78)
    sgd = dec.get("SGD")
    sgd_ratio = D["SGD"]["path_len"] / (D["SGD"]["disp_len"] + 1e-12)
    print(f"   SGD geometric overlap with ADAPTIVE basis : "
          f"{sgd['in_shared']:.2f}  ({sgd['x_chance']:.0f}x chance)")
    print(f"   SGD loss recovered by that component ALONE: "
          f"{100*sgd['causal_recovery']:.0f}%")
    print(f"   SGD path/displacement ratio               : {sgd_ratio:.1f}")
    print()
    if sgd["causal_recovery"] > 0.70:
        print("   => THE SHARED COMPONENT IS CAUSALLY SUFFICIENT for SGD.")
        print("      A basis built WITHOUT SGD, from adaptive optimizers only,")
        print("      carries most of SGD's loss reduction.  Different")
        print("      optimizers take different PATHS but their ENDPOINT")
        print("      DISPLACEMENTS share a functionally central low-dimensional")
        print("      component.  In the language of the maps:")
        print("          (task, w0) --F_opt--> gamma --E--> Delta_w")
        print("      the COMPOSITE  E . F_opt  is far more invariant than F_opt")
        print("      itself.  The endpoint displacement -- not the trajectory --")
        print("      is the better candidate for an intrinsic object.")
    elif sgd["causal_recovery"] > sgd["in_shared"] + 0.20:
        print("   => OVER-REPRESENTED BUT NOT SUFFICIENT.  The shared")
        print("      directions carry more of SGD's improvement than their")
        print("      share of its displacement -- so they matter more than")
        print("      chance -- but they do not carry most of it.  Promising,")
        print("      not decisive.")
    else:
        print("   => DESCRIPTIVE OVERLAP ONLY.  The shared component's causal")
        print("      recovery tracks its geometric share.  A subspace that is")
        print("      shared but does not carry the improvement is not the")
        print("      optimizer-independent object we want.")

    print("\n   HONEST SCOPE: even the B/C outcome does not prove a smooth")
    print("   manifold.  It shows the ENDPOINTS of different optimizers share")
    print("   a low-dimensional subspace.  Probing the loss geometry")
    print("   independently of ANY trajectory is still required to settle")
    print("   whether that subspace is intrinsic to the landscape.")

    json.dump({"k": args.k, "shared_dim": int(Ushared.shape[1]), "v0": v0,
               "per_optimizer": {n: {"floor": d["floor"],
                                     "descent": d["descent"],
                                     "own_basis_frac": d["own"],
                                     "path_len": d["path_len"],
                                     "disp_len": d["disp_len"],
                                     "path_disp_ratio":
                                         d["path_len"] / (d["disp_len"] + 1e-12)}
                                 for n, d in D.items()},
               "decomposition": dec},
              open("shared_residual.json", "w"), indent=2, default=float)
    print("\n  wrote shared_residual.json")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        nm = list(D)
        fig, ax = plt.subplots(1, 3, figsize=(18, 5))
        ax[0].bar(nm, [100 * D[n]["own"] for n in nm], color="#4c72b0")
        ax[0].axhline(70, ls="--", color="k")
        ax[0].set_ylabel("% of own displacement in own k=3 basis")
        ax[0].set_title("PATH compressibility\n(what trajectory PCA sees)")
        ax[0].tick_params(axis="x", rotation=30)

        ax[1].bar(nm, [D[n]["path_len"] / D[n]["disp_len"] for n in nm],
                  color="#dd8452")
        ax[1].set_ylabel("path length / displacement")
        ax[1].set_title("Zig-zag ratio\n(Hypothesis C: SGD wanders)")
        ax[1].tick_params(axis="x", rotation=30)

        x = np.arange(len(nm))
        ax[2].bar(x - 0.2, [dec[n]["in_shared"] for n in nm], 0.4,
                  label="endpoint in shared basis", color="#55a868")
        ax[2].bar(x + 0.2, [dec[n]["null_mean"] for n in nm], 0.4,
                  yerr=[dec[n]["null_std"] for n in nm],
                  label="random null", color="#ccc")
        ax[2].set_xticks(x); ax[2].set_xticklabels(nm, rotation=30)
        ax[2].set_ylabel("fraction of displacement")
        ax[2].set_title("ENDPOINT in the ADAPTIVE shared basis\n"
                        "(THE discriminator)")
        ax[2].legend(fontsize=8)
        plt.suptitle("Shared (+) optimizer-specific: does SGD reach the same "
                     "place by a noisier road?", fontsize=13, weight="bold")
        plt.tight_layout()
        plt.savefig("shared_residual.png", dpi=180)
        print("  wrote shared_residual.png")
    except Exception as e:
        print(f"  [plot skipped: {e}]")


if __name__ == "__main__":
    main()
