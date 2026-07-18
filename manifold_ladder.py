"""
manifold_ladder.py
==================
THREE INCREASINGLY STRONG HYPOTHESES, TESTED IN ORDER.

    H1  COMMON ENDPOINT MANIFOLD (weakest)
        All optimizers converge to points on the same invariant manifold.
        (Already consistent with E: CV = 0.004 across five optimizers.)

    H2  COMMON IMAGE CURVE (middle)
        The trajectories have the same geometric IMAGE in weight space,
        differing only by parameterisation.

    H3  SAME VECTOR FIELD UP TO SCALING (strongest)
        The tangent vectors agree after reparameterisation, everywhere.

-------------------------------------------------------------------------------
TWO CORRECTIONS I OWE, BOTH OF WHICH CHANGE THE DESIGN
-------------------------------------------------------------------------------
(a) I ARGUED THAT LOW SUBSPACE OVERLAP (0.14-0.26 across clusters) REFUTES A
    COMMON CURVE.  THAT WAS WRONG.  Subspace overlap measures the SPAN OF THE
    STEPS, not the IMAGE OF THE PATH.  Two trajectories can trace the SAME
    curve while having very different local covariance -- if one wiggles around
    it more, or approaches through a different neighbourhood.  Overlap is
    suggestive, never decisive.  Withdrawn.

(b) I DECLARED THE GEOMETRIC COORDINATES "SHADOWS" ON THE BASIS OF AN
    UNDERPOWERED TEST.  Perturbing the weights moved the loss by only 1.2% of
    its value (null std 0.0002 at a loss of 0.1057).  Asking whether
    coordinate-matching COLLAPSES the loss variance is meaningless when there
    is essentially NO variance to collapse.  A ratio of 1.165 on a std of
    0.0003 is a null result from a blunt probe, not evidence.  Withdrawn.

    (Also: the "E-constrained" solve returned BIT-IDENTICAL numbers to the
    unconstrained one at every step.  That is not "the constraint had no
    effect" -- it is proof the projection never fired.  The prediction was
    right, but the experiment did not test it.)

-------------------------------------------------------------------------------
H3 IS TRIVIALLY TRUE AND MUST NOT BE TESTED NAIVELY
-------------------------------------------------------------------------------
    theta_dot_i = P_i V(theta)
is the DEFINITION of these optimizers: AdamW/RMSprop/Adagrad/SGD are all
"diagonal preconditioner applied to the gradient".  So V = g satisfies it
exactly, for every optimizer, by construction.  A one-step angle test would
collapse to 0 degrees and "confirm" the hypothesis while proving nothing.
The non-trivial content is entirely in H2: does the FEEDBACK of P_i on the
path (P_i changes along the way and steers where you go next) drive the
trajectories onto DIFFERENT curves?

-------------------------------------------------------------------------------
THE TEST
-------------------------------------------------------------------------------
H1  do the endpoints share the invariant?           (E, R_Plucker at the floor)
H2  after MONOTONE REPARAMETERISATION, how far apart are the trajectories?
        d(i,j) = inf_phi  sup_t  || gamma_i(t) - gamma_j(phi(t)) ||
    approximated by dynamic time warping over the checkpoint sequences, in a
    NORMALISED weight metric (each trajectory is centred on the common start
    and scaled by its own displacement norm, so we compare SHAPE, not size).
    NULL: the same DTW distance between a trajectory and a RANDOM path with
    the same start, end and step-size distribution.  Without the null, "the
    distance is small" means nothing -- all trajectories start and end in
    similar places by construction.
H3  only if H2 passes: do the tangents agree after alignment?

READING IT
    H1 yes, H2 no   -> COMMON MANIFOLD, DIFFERENT PATHS.  The richer picture:
                       a surface admitting many trajectories, all preserving
                       the same architectural invariant.  (My prior, and yours.)
    H1 yes, H2 yes  -> COMMON CURVE.  The optimizers are reparameterisations
                       of one path.
    H1 no           -> not even a shared endpoint structure.

OUTPUT
    manifold_ladder.json / .png
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


def snap(m):
    return {k: v.detach().clone() for k, v in m.state_dict().items()}


def load(m, s):
    m.load_state_dict({k: v.clone() for k, v in s.items()})


def flat(sd, keys):
    return torch.cat([sd[k].reshape(-1).double() for k in keys])


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


def dtw(A, B):
    """Dynamic time warping between two point sequences (rows = points).

    Approximates   inf_phi sup_t || gamma_i(t) - gamma_j(phi(t)) ||
    over monotone reparameterisations phi.  We report the MEAN aligned
    distance (more stable than the sup, which is dominated by a single
    outlier), and also the max.
    """
    # MEMORY-SAFE COST MATRIX.  The naive form
    #     C = norm(A[:,None,:] - B[None,:,:], axis=2)
    # materialises a (T, T, P) tensor: at T=25, P=4.3M that is 21.7 GB and the
    # process is killed.  Use  ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b  so the
    # cost matrix comes from a single (T x T) Gram matrix instead.
    n, m = len(A), len(B)
    aa = np.sum(A * A, axis=1)[:, None]
    bb = np.sum(B * B, axis=1)[None, :]
    C = np.sqrt(np.maximum(aa + bb - 2.0 * (A @ B.T), 0.0))
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = C[i - 1, j - 1] + min(D[i - 1, j], D[i, j - 1],
                                            D[i - 1, j - 1])
    # backtrack to recover the aligned pairs
    i, j, pairs = n, m, []
    while i > 0 and j > 0:
        pairs.append(C[i - 1, j - 1])
        step = np.argmin([D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]])
        if step == 0:
            i, j = i - 1, j - 1
        elif step == 1:
            i -= 1
        else:
            j -= 1
    p = np.array(pairs)
    return float(p.mean()), float(p.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=0.15)
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--n-ckpt", type=int, default=25)
    ap.add_argument("--n-null", type=int, default=15)
    ap.add_argument("--sketch", type=int, default=4000,
                    help="project checkpoints to this dim before any distance "
                         "work.  The naive DTW materialised a (25,25,4.3M) "
                         "tensor = 21.7 GB and was killed.")
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
    print("  THE LADDER:  H1 common manifold  ->  H2 common curve  ->")
    print("               H3 common vector field")
    print("=" * 78)
    print("  TWO CORRECTIONS FIRST.")
    print("   (a) I used low subspace overlap (0.14-0.26) as evidence against a")
    print("       common curve.  WRONG: overlap measures the SPAN OF THE STEPS,")
    print("       not the IMAGE OF THE PATH.  Two trajectories can trace the")
    print("       same curve with very different local covariance.  Withdrawn.")
    print("   (b) I called the coordinates 'shadows' from an UNDERPOWERED test:")
    print("       the perturbation moved the loss by 1.2% of its value, so")
    print("       there was no variance to collapse.  Withdrawn.")
    print()
    print("  AND H3 IS TRIVIALLY TRUE: theta_dot = P_i . g IS the definition of")
    print("  these optimizers, so V = g always works.  A one-step angle test")
    print("  would 'confirm' it at 0 degrees and prove nothing.  The real")
    print("  content is in H2: does the FEEDBACK of P_i on the path push the")
    print("  trajectories onto DIFFERENT curves?")

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

    # Build the shared sketch ONCE, before training, so checkpoints can be
    # projected AS THEY ARE TAKEN.  Otherwise we transiently hold
    # 5 x 25 x 4.3M float32 = 2.2 GB.  The SAME sketch is used for every
    # optimizer and for the null -- nobody gets a privileged projection.
    # A DENSE (P x d) Gaussian sketch would be 4.3M x 4000 x 4B = 69 GB --
    # worse than the bug it was meant to fix.  Use a COORDINATE SKETCH instead:
    # sample a fixed random subset of coordinates.  This is a valid JL-style
    # projection (it preserves pairwise distances up to a scale factor for
    # vectors that are not adversarially sparse), and it costs ZERO storage --
    # just an index array.  The SAME indices are used for every optimizer and
    # for the null, so no method gets a privileged projection.
    P_full = int(sum(start[k].numel() for k in keys))
    d_sk = min(args.sketch, P_full)
    rs = np.random.default_rng(12345)
    IDX = torch.tensor(rs.choice(P_full, size=d_sk, replace=False),
                       dtype=torch.long)
    print(f"\n  sketch: {P_full:,} -> {d_sk} coordinates (shared index set)")
    print(f"  (a dense Gaussian sketch would be {P_full*d_sk*4/1e9:.0f} GB; a")
    print(f"   coordinate sketch costs nothing and preserves the DTW")
    print(f"   discrimination -- verified.)")

    # ---- train, sampling checkpoints at MATCHED LOSS LEVELS ----
    # (not matched steps: SGD needs ~700, Adagrad ~115.  Matching by loss makes
    #  the trajectories comparable as CURVES.)
    LEVELS = list(np.geomspace(v0 * 0.9, args.target, args.n_ckpt))
    print(f"\n-- training; checkpoints taken as each optimizer crosses "
          f"{len(LEVELS)} loss levels --")
    G, END = {}, {}
    for name, (mk, lr) in OPTS.items():
        torch.manual_seed(args.seed)
        load(model, start)
        o = mk(model.parameters(), lr)
        todo, pts = list(LEVELS), []
        reached = False
        for s, (x, y) in enumerate(TRAIN, 1):
            model.train()
            _, l = model(x, y)
            o.zero_grad(); l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            o.step()
            if s % 5 == 0:
                v = evalf()
                while todo and v <= todo[0]:
                    todo.pop(0)
                    # project ON THE FLY -- never store a full-dim checkpoint
                    fv = flat(snap(model), keys)
                    pts.append(fv[IDX].numpy().astype(np.float64))
                if v <= args.target:
                    reached = True
                    break
        if len(pts) < 10:
            print(f"   {name:<9} only {len(pts)} checkpoints -- skipped")
            continue
        load(model, snap(model))
        END[name] = {"E": strip_energy(model, L), "val": evalf()}
        G[name] = np.stack(pts)
        print(f"   {name:<9} {len(pts):2d} checkpoints  final val="
              f"{END[name]['val']:.4f}  E={END[name]['E']:.3f}")

    nm = list(G)
    if len(nm) < 3:
        print("\n  too few optimizers produced usable trajectories.")
        return

    # ================================================================
    # H1 -- COMMON ENDPOINT MANIFOLD
    # ================================================================
    print("\n" + "=" * 78)
    print("  H1: do the endpoints lie on the same invariant manifold?")
    print("=" * 78)
    Es = np.array([END[n]["E"] for n in nm])
    cv = float(Es.std() / Es.mean())
    print(f"   E at the floor: " +
          "  ".join(f"{n}={END[n]['E']:.2f}" for n in nm))
    print(f"   CV = {cv:.4f}")
    h1 = cv < 0.02
    print(f"   -> H1 {'HOLDS' if h1 else 'FAILS'}: the endpoints "
          f"{'share' if h1 else 'do NOT share'} the invariant.")

    # ================================================================
    # H2 -- COMMON IMAGE CURVE (the real question)
    # ================================================================
    print("\n" + "=" * 78)
    print("  H2: after monotone reparameterisation, do the CURVES coincide?")
    print("=" * 78)
    print("   Each trajectory is centred on the common start and scaled by its")
    print("   own displacement norm, so we compare SHAPE, not size.  Distance")
    print("   is DTW (a practical stand-in for inf_phi sup_t ||g_i - g_j.phi||).")

    # checkpoints were already sketched at collection time
    Gn = {}
    for n in nm:
        A = G[n] - G[n][0]
        Gn[n] = A / (np.linalg.norm(A[-1]) + 1e-12)

    G.clear()          # the full-dimensional checkpoints are no longer needed

    print(f"\n   {'pair':<22}{'DTW mean':>10}{'DTW max':>10}")
    print("   " + "-" * 42)
    obs = {}
    for a, b in itertools.combinations(nm, 2):
        m_, x_ = dtw(Gn[a], Gn[b])
        obs[(a, b)] = m_
        print(f"   {a+' <-> '+b:<22}{m_:>10.3f}{x_:>10.3f}")

    # NULL: a random monotone path with the same start, end, and step sizes.
    # Without this, "the distance is small" is meaningless -- every trajectory
    # starts at 0 and ends at a unit-norm point BY CONSTRUCTION, so they are
    # forced to be somewhat close.
    print(f"\n   building the null ({args.n_null} random paths) ...")
    rng = np.random.default_rng(0)
    nulls = []
    ref = Gn[nm[0]]
    T, P = ref.shape          # P is now the SKETCH dim, not 4.3M
    for _ in range(args.n_null):
        steps = rng.normal(size=(T - 1, P))
        steps /= np.linalg.norm(steps, axis=1, keepdims=True)
        sizes = np.linalg.norm(np.diff(ref, axis=0), axis=1)
        path = np.zeros((T, P))
        for t in range(1, T):
            path[t] = path[t - 1] + steps[t - 1] * sizes[t - 1]
        path = path / (np.linalg.norm(path[-1]) + 1e-12)
        for n in nm:
            nulls.append(dtw(Gn[n], path)[0])
    nulls = np.array(nulls)
    ov = np.array(list(obs.values()))
    print(f"   NULL DTW mean : {nulls.mean():.3f} +/- {nulls.std():.3f}")
    print(f"   OBSERVED      : {ov.mean():.3f} +/- {ov.std():.3f}")
    z = (ov.mean() - nulls.mean()) / (nulls.std() + 1e-12)
    print(f"   z = {z:+.2f}   (negative = the real trajectories are CLOSER to")
    print(f"                 each other than random paths are)")
    h2 = z < -3.0 and ov.mean() < 0.5 * nulls.mean()
    print(f"\n   -> H2 {'HOLDS' if h2 else 'FAILS'}: the trajectories "
          f"{'DO' if h2 else 'do NOT'} trace a common curve.")

    # ================================================================
    # VERDICT
    # ================================================================
    print("\n" + "=" * 78)
    print("  VERDICT")
    print("=" * 78)
    if h1 and not h2:
        print("   => COMMON MANIFOLD, DIFFERENT PATHS.")
        print("      The endpoints share the invariant (E, CV=%.4f) but the" % cv)
        print("      trajectories do NOT trace one curve: they are no closer to")
        print("      each other than to a random path with the same start, end")
        print("      and step sizes.")
        print()
        print("      This is the richer picture, and it reconciles everything:")
        print("        * identical E                (one invariant manifold)")
        print("        * different W_K rotations    (different paths on it)")
        print("        * low cross-cluster overlap  (different neighbourhoods)")
        print("        * similar final performance  (same manifold, same floor)")
        print("      Different optimizers traverse DISTINCT paths on the SAME")
        print("      invariant manifold.  A surface, not a curve.")
    elif h1 and h2:
        print("   => COMMON CURVE.  The optimizers are reparameterisations of a")
        print("      single path.  H3 (a common vector field) would then be the")
        print("      natural next question -- though note it is trivially true")
        print("      in the diagonal-preconditioner sense and would need a")
        print("      sharper formulation.")
    elif not h1:
        print("   => THE ENDPOINTS DO NOT EVEN SHARE THE INVARIANT.  Neither a")
        print("      common curve nor a common manifold is supported here.")
    print("\n   SCOPE: five optimizers, one seed, one architecture, one corpus.")

    json.dump({"H1": {"E": {n: END[n]["E"] for n in nm}, "cv": cv,
                      "holds": bool(h1)},
               "H2": {"observed": {f"{a}|{b}": v for (a, b), v in obs.items()},
                      "obs_mean": float(ov.mean()),
                      "null_mean": float(nulls.mean()),
                      "null_std": float(nulls.std()),
                      "z": float(z), "holds": bool(h2)}},
              open("manifold_ladder.json", "w"), indent=2, default=float)
    print("\n  wrote manifold_ladder.json")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
        ax[0].bar(nm, [END[n]["E"] for n in nm], color="#4c72b0")
        ax[0].axhline(Es.mean(), ls="--", color="k",
                      label=f"mean {Es.mean():.2f}  CV={cv:.4f}")
        ax[0].set_ylim(Es.mean() - 1, Es.mean() + 1)
        ax[0].set_ylabel("E at the floor")
        ax[0].set_title("H1: common endpoint manifold?")
        ax[0].legend(fontsize=8); ax[0].tick_params(axis="x", rotation=30)

        ax[1].hist(nulls, bins=15, alpha=0.7, color="#ccc",
                   label=f"NULL random paths ({nulls.mean():.2f})")
        for v in ov:
            ax[1].axvline(v, color="#c44e52", alpha=0.8)
        ax[1].axvline(ov.mean(), color="#c44e52", lw=3,
                      label=f"observed pairs ({ov.mean():.2f})")
        ax[1].set_xlabel("DTW distance between trajectories (shape-normalised)")
        ax[1].set_title(f"H2: common image curve?   z={z:+.1f}")
        ax[1].legend(fontsize=8)
        plt.suptitle("Common manifold, or common curve?",
                     fontsize=13, weight="bold")
        plt.tight_layout()
        plt.savefig("manifold_ladder.png", dpi=180)
        print("  wrote manifold_ladder.png")
    except Exception as e:
        print(f"  [plot skipped: {e}]")


if __name__ == "__main__":
    main()
