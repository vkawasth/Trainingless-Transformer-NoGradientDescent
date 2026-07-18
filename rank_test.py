"""
rank_test.py
============
Does the loss-carrying flow live on a LOW-DIMENSIONAL submanifold?

WHY THIS IS THE DECIDING MEASUREMENT
------------------------------------
Established so far:
  * W_K carries the GEOMETRY (E = 43.5 invariant, phi_k, R_Plucker flat) and
    ~0.5% of the loss.  The invariant we found is a shadow of the W_K flow.
  * The LOSS lives in Emb/FF/LN/attention, is 86% irreducible interaction at
    the floor, and its basis ROTATES (floor-Emb transplanted alone is
    HARMFUL: -0.217 nats; the FF line turns back UP past lam<0.25).
  * Those turning points are a CONSTRAINT SURFACE showing itself: the correct
    floor value of one block depends on where the other blocks are.

So there IS structure in the loss flow -- we have detected it but not
characterised it.  Before constructing cones / restriction maps / a D-brane on
that structure, one thing must be known:

    DOES THE CONSTRAINT SURFACE HAVE SMALL CODIMENSION?

  * LOW rank  -> the co-adaptation lives in a handful of directions, there is
                 a genuine submanifold, and a symplectic construction has
                 somewhere to live.  Cones are buildable.
  * HIGH rank -> the co-adaptation is spread over thousands of directions with
                 no preferred structure.  "Find the conserved quantity" becomes
                 finding a function on a high-dimensional object with no
                 handle.  The cone programme has no substrate.

METHODOLOGICAL RULE (learned the hard way)
------------------------------------------
The ALGEBRAIC rank of dTheta (its SVD spectrum) is nearly always full for a
4.3M-parameter displacement and tells us NOTHING.  The question is CAUSAL:

    does a rank-k PROJECTION of dTheta, TRANSPLANTED IN, recover the loss?

We project, we transplant, we MEASURE.  We do not fit.  Fitting a spectrum
would repeat the lambda_cos error (R^2 = 0.9996, causally worth 0.4%).

WHAT IS MEASURED
----------------
(a) EFFECTIVE RANK of dTheta per regime -- spectral, reported for context only,
    explicitly NOT used to draw the conclusion.
(b) CAUSAL RANK: build the rank-k projection of dTheta from the trajectory's
    own principal directions (PCA over checkpoint differences -- the local
    basis the flow actually uses), transplant P_k(dTheta) into the start
    state, and measure val.  Sweep k.  The k at which the descent is recovered
    IS the dimension of the loss-carrying submanifold.
    Gates: does it descend? monotone in k? does it saturate?
(c) CANDIDATE INVARIANTS along the TRUE loss-carrying trajectory:
        tau            = ||grad_FF|| / ||grad_Emb||     (K0 gluing defect)
        E              = strip energy (the W_K invariant -- control)
        fisher_norm    = ||grad||^2 (empirical Fisher trace proxy)
        ntk_trace      = tr(J J^T) proxy via random probes
        emb_ff_coupling= <grad_Emb, H . grad_FF> / norms  (the cross-Hessian
                         coupling the paper identifies as the source of the
                         anti-alignment -- the prime suspect for the conserved
                         quantity of the LOSS flow)
        eff_rank       = participation ratio of the local grad covariance
    Each is measured at every waypoint, in BOTH regimes, and we report which
    (if any) stay FLAT across the crystallisation transition.  A quantity that
    is conserved on both sides of the transition but with different level sets
    is exactly a symplectic invariant crossing a wall.

OUTPUT
    rank_test.json / .csv
    rank_test.png   (causal rank curve; invariant flatness across the transition)
"""

import argparse
import csv
import json
import math

import numpy as np
import torch


# ----------------------------------------------------------------- groups
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


# The LOSS-CARRYING subspace (established by the ablation).  W_K is excluded:
# it carries the geometry, not the loss, and including it would let the
# geometry's (already known) invariance contaminate the rank measurement.
LOSS_GROUPS = {"Emb", "FF", "LayerNorm", "W_Q", "W_V", "W_O", "other"}


def snapshot(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def load_snapshot(model, s):
    model.load_state_dict({k: v.clone() for k, v in s.items()})


def flatten(sd, keys):
    return torch.cat([sd[k].reshape(-1).double() for k in keys])


def unflatten(vec, sd, keys):
    out, i = {}, 0
    for k in keys:
        n = sd[k].numel()
        out[k] = vec[i:i + n].reshape(sd[k].shape).to(sd[k].dtype)
        i += n
    return out


# ----------------------------------------------------------------- geometry
def strip_energy(model, n_layers, rank=6):
    Us = []
    for k in range(n_layers):
        W = model.blocks[k].attn.WK.weight.detach().cpu().numpy()
        U, _, _ = np.linalg.svd(W, full_matrices=False)
        Us.append(U[:, :rank])
    E = 0.0
    for k in range(n_layers - 1):
        s = np.linalg.svd(Us[k].T @ Us[k + 1], compute_uv=False)
        E += float(np.sum(np.arccos(np.clip(s, -1, 1))))
    return E


# ------------------------------------------------------- candidate invariants
def grad_of(model, get_batch, n=3):
    """Averaged gradient, returned as a dict of tensors."""
    gs = None
    for _ in range(n):
        model.zero_grad(set_to_none=True)
        x, y = get_batch()
        _, l = model(x, y)
        l.backward()
        cur = {k: (p.grad.detach().clone() if p.grad is not None
                   else torch.zeros_like(p))
               for k, p in model.named_parameters()}
        gs = cur if gs is None else {k: gs[k] + cur[k] for k in gs}
    model.zero_grad(set_to_none=True)
    return {k: v / n for k, v in gs.items()}


def hvp(model, get_batch, vec_dict):
    """Hessian-vector product via double backprop (Pearlmutter)."""
    model.zero_grad(set_to_none=True)
    x, y = get_batch()
    _, l = model(x, y)
    params = [p for _, p in model.named_parameters()]
    names = [k for k, _ in model.named_parameters()]
    g = torch.autograd.grad(l, params, create_graph=True)
    dot = sum((gi * vec_dict[n_]).sum()
              for gi, n_ in zip(g, names) if n_ in vec_dict)
    h = torch.autograd.grad(dot, params, retain_graph=False)
    model.zero_grad(set_to_none=True)
    return {n_: hi.detach() for n_, hi in zip(names, h)}


def interval_rank(ckpts, keys, window=4):
    """(Point 1) Effective rank of the LOCAL displacement, over successive
    intervals -- not just endpoint-to-endpoint.

    For each window of `window` consecutive checkpoint-differences, compute the
    participation-ratio effective rank of that local step basis.  The question
    is whether the rank DECREASES during training, and whether it does so at
    the SAME step where FF becomes polynomial and the interaction term takes
    over.  Three transitions coinciding is far stronger evidence for a
    crystallisation phenomenon than any one alone."""
    out = []
    for i in range(len(ckpts) - window):
        D = []
        for j in range(i, i + window):
            D.append((flatten(ckpts[j + 1], keys)
                      - flatten(ckpts[j], keys)).numpy())
        D = np.stack(D)
        s = np.linalg.svd(D, compute_uv=False)
        p = s ** 2 / (np.sum(s ** 2) + 1e-12)
        eff = float(np.exp(-np.sum(p * np.log(p + 1e-12))))   # perplexity rank
        out.append({"window_start": i, "eff_rank": eff,
                    "max_possible": window})
    return out


def interaction_matrix(model, start, donor, eval_fn, groups):
    """(Point 3) The INTERACTION object, not the parameter object.

    The transplants say the primitive is the interaction, not the blocks.  So
    build a matrix whose entries are the PAIRWISE INTERACTION GAINS:

        M[i][j] = dL(Gi + Gj)  -  dL(Gi)  -  dL(Gj)

    i.e. how much MORE (or less) you get from transplanting groups i and j
    together than from the sum of their solo effects.  The diagonal holds the
    solo gains.  Then take the rank of M.

    This dissociates two possibilities that a parameter-rank test CANNOT
    distinguish:
        parameter updates HIGH-rank + interaction LOW-rank
            -> the low-dimensional object lives in interaction geometry
        parameter updates LOW-rank + interaction HIGH-rank
            -> updates are compressible but causation is distributed
    """
    G = sorted(groups)
    n = len(G)
    v0 = eval_fn(start)

    solo = {}
    for g in G:
        sd = {k: (donor[k].clone() if group_of(k) == g else v.clone())
              for k, v in start.items()}
        solo[g] = v0 - eval_fn(sd)

    M = np.zeros((n, n))
    for a in range(n):
        M[a, a] = solo[G[a]]
        for b in range(a + 1, n):
            pair = {G[a], G[b]}
            sd = {k: (donor[k].clone() if group_of(k) in pair else v.clone())
                  for k, v in start.items()}
            joint = v0 - eval_fn(sd)
            inter = joint - solo[G[a]] - solo[G[b]]
            M[a, b] = M[b, a] = inter
    return G, M, solo


# ----------------------------------------------------------------- main-helpers
def stability_score(series, loss_series):
    """Uniform criterion for EVERY candidate invariant (point 4).

    We do NOT look for something perfectly constant -- that over-interprets
    quantities that merely fluctuate slowly.  Instead, score how stable the
    candidate is RELATIVE to how much the loss itself moved over the same
    window:

        score = Var(I_t) / Var(L_t)     (both normalised to their own means)

    score << 1  : the candidate is far more stable than the loss it is
                  supposed to be an invariant of -> a real conservation
                  candidate.
    score ~ 1   : it just tracks the loss -> not an invariant, a restatement.
    score >> 1  : noisier than the loss -> nothing.
    Applying ONE criterion to all candidates keeps the comparison honest."""
    I = np.asarray(series, float)
    L = np.asarray(loss_series, float)
    if len(I) < 3:
        return float("nan")
    cvI = np.var(I) / (np.mean(I) ** 2 + 1e-12)
    cvL = np.var(L) / (np.mean(L) ** 2 + 1e-12)
    return float(cvI / (cvL + 1e-12))


def invariants(model, get_batch, n_layers, seed=0):
    """The candidate conserved quantities of the LOSS-carrying flow."""
    torch.manual_seed(seed)
    g = grad_of(model, get_batch, n=3)

    def gnorm(groups):
        s = 0.0
        for k, v in g.items():
            if group_of(k) in groups:
                s += float((v ** 2).sum())
        return math.sqrt(s)

    n_ff = gnorm({"FF"})
    n_emb = gnorm({"Emb"})
    tau = n_ff / (n_emb + 1e-12)
    fisher = sum(float((v ** 2).sum()) for v in g.values())

    # cross-Hessian coupling: <g_Emb , H . g_FF> normalised.
    # This is the coupling the paper identifies as the source of the embedding
    # anti-alignment -- the prime suspect for the LOSS flow's invariant.
    v_ff = {k: (v if group_of(k) == "FF" else torch.zeros_like(v))
            for k, v in g.items()}
    Hv = hvp(model, get_batch, v_ff)
    num = sum(float((Hv[k] * g[k]).sum())
              for k in g if group_of(k) == "Emb")
    coupling = num / (n_emb * n_ff + 1e-12)

    # NTK trace proxy: E_v[ ||H v|| ] over random unit probes
    ntk = 0.0
    for i in range(2):
        torch.manual_seed(seed + 100 + i)
        rv = {k: torch.randn_like(v) for k, v in g.items()}
        nrm = math.sqrt(sum(float((v ** 2).sum()) for v in rv.values())) + 1e-12
        rv = {k: v / nrm for k, v in rv.items()}
        Hr = hvp(model, get_batch, rv)
        ntk += math.sqrt(sum(float((v ** 2).sum()) for v in Hr.values()))
    ntk /= 2

    # effective rank of the gradient (participation ratio over per-param groups)
    per = np.array([float((v ** 2).sum()) for v in g.values()])
    per = per / (per.sum() + 1e-12)
    eff_rank = float(1.0 / (np.sum(per ** 2) + 1e-12))

    return {"tau": tau, "E": strip_energy(model, n_layers),
            "fisher_norm": fisher, "emb_ff_coupling": coupling,
            "ntk_trace": ntk, "grad_eff_rank": eff_rank}


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase3-steps", type=int, default=150)
    ap.add_argument("--n-checkpoints", type=int, default=24,
                    help="checkpoints along Phase 3; their differences give "
                         "the trajectory's own principal basis")
    ap.add_argument("--ks", type=int, nargs="+",
                    default=[1, 2, 3, 5, 8, 12, 16, 20],
                    help="ranks to test causally")
    ap.add_argument("--compiler", default="compiler_analytic_topogate.py")
    args = ap.parse_args()

    g_ = {}
    src = open(args.compiler).read()
    cut = src.find("# ── PHASE 1")
    if cut == -1:
        cut = src.find("PHASE 1")
    exec(src[:cut], g_)
    model = g_["model"]; get_batch = g_["get_batch"]; eval_val = g_["eval_val"]
    LR = g_["LR"]; N = g_["N_STU"]; LR5 = LR * 5

    print("=" * 74)
    print("  RANK TEST: does the loss-carrying flow live on a")
    print("  LOW-DIMENSIONAL submanifold?")
    print("=" * 74)
    print("  If YES: the co-adaptation constraint surface has small")
    print("          codimension -> cones/restriction maps have a substrate.")
    print("  If NO : the flow is spread over thousands of directions ->")
    print("          'find the conserved quantity' has no handle.")
    print("\n  RULE: causal only.  We PROJECT, TRANSPLANT, and MEASURE.")
    print("  The SVD spectrum of dTheta is reported for context but is NOT")
    print("  the evidence -- a full algebraic rank says nothing causal.")
    print("\n  PRE-SPECIFIED OUTCOMES (fixed before running, so the result is")
    print("  interpretable whichever way it lands):")
    print("  " + "-" * 70)
    print("   param-rank | interaction-rank | causal transplant | interpretation")
    print("  " + "-" * 70)
    print("     LOW      |       --         |    reproduces     | low-dim causal")
    print("              |                  |                   |   mechanism")
    print("     LOW      |       --         |      FAILS        | updates compressible,")
    print("              |                  |                   |   causation is NOT")
    print("     HIGH     |      LOW         |        --         | structure lives in")
    print("              |                  |                   |   INTERACTION space")
    print("     HIGH     |      HIGH        |        --         | genuinely distributed;")
    print("              |                  |                   |   reduced-order models")
    print("              |                  |                   |   implausible")
    print("  " + "-" * 70)

    start = snapshot(model)
    v0 = eval_val(model, n=10)
    print(f"\n  start (MF pole): val={v0:.4f}")

    # ---------- run Phase 3, keeping checkpoints ----------
    print(f"\n-- running Phase 3 ({args.phase3_steps} CE), "
          f"{args.n_checkpoints} checkpoints --")
    every = max(1, args.phase3_steps // args.n_checkpoints)
    opt = torch.optim.AdamW(model.parameters(), lr=LR5,
                            betas=(0.9, 0.95), weight_decay=0.1)
    ckpts, inv_track = [snapshot(model)], []
    inv_track.append({"step": 0, "val": v0,
                      **invariants(model, get_batch, N, seed=0)})
    for s in range(1, args.phase3_steps + 1):
        model.train()
        x, y = get_batch()
        _, l = model(x, y)
        opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if s % every == 0 or s == args.phase3_steps:
            ckpts.append(snapshot(model))
            vv = eval_val(model, n=6)
            iv = invariants(model, get_batch, N, seed=s)
            iv.update({"step": s, "val": vv})
            inv_track.append(iv)
            print(f"   step {s:3d}: val={vv:.4f}  tau={iv['tau']:.3f}  "
                  f"E={iv['E']:.2f}  coupling={iv['emb_ff_coupling']:+.4f}  "
                  f"fisher={iv['fisher_norm']:.3e}")
    floor = ckpts[-1]
    load_snapshot(model, floor)
    v_floor = eval_val(model, n=10)
    total = v0 - v_floor
    print(f"\n   floor: val={v_floor:.4f}   TOTAL DESCENT = {total:.4f} nats")

    # ---------- build the trajectory's own basis ----------
    keys = [k for k in start if group_of(k) in LOSS_GROUPS]
    print(f"\n-- basis from the trajectory itself "
          f"({len(keys)} tensors, LOSS subspace; W_K excluded) --")
    X = []
    for i in range(1, len(ckpts)):
        d = flatten(ckpts[i], keys) - flatten(ckpts[i - 1], keys)
        X.append(d.numpy())
    X = np.stack(X)                       # (n_steps, P)
    dtheta = (flatten(floor, keys) - flatten(start, keys)).numpy()
    P = dtheta.size
    print(f"   dTheta dimension P = {P:,}")

    # ---- (1) PER-INTERVAL RANK: does rank fall during training? ----
    print(f"\n-- (1) EFFECTIVE RANK OVER SUCCESSIVE INTERVALS --")
    ir = interval_rank(ckpts, keys, window=4)
    vals_at = [inv_track[min(d["window_start"], len(inv_track) - 1)]["val"]
               for d in ir]
    print(f"   {'window':>8}{'val':>10}{'eff_rank':>11}  (of 4)")
    print("   " + "-" * 33)
    for d, vv in zip(ir, vals_at):
        print(f"   {d['window_start']:>8}{vv:>10.4f}{d['eff_rank']:>11.2f}")
    r_pre = [d["eff_rank"] for d, vv in zip(ir, vals_at) if vv > 0.30]
    r_post = [d["eff_rank"] for d, vv in zip(ir, vals_at) if vv <= 0.30]
    if r_pre and r_post:
        print(f"\n   mean eff_rank PRE-crystallisation  (val>0.3): "
              f"{np.mean(r_pre):.2f}")
        print(f"   mean eff_rank POST-crystallisation (val<0.3): "
              f"{np.mean(r_post):.2f}")
        drop = np.mean(r_pre) - np.mean(r_post)
        print(f"   --> rank {'DROPS' if drop > 0.2 else 'does NOT drop'} "
              f"across the transition ({drop:+.2f})")
        print("   (If it drops at the SAME step FF turns polynomial and the")
        print("    interaction term takes over, all three transitions coincide")
        print("    -- that is real crystallisation, not one lucky statistic.)")

    # (a) spectral -- CONTEXT ONLY
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    p = S ** 2 / (np.sum(S ** 2) + 1e-12)
    spec_eff = float(np.exp(-np.sum(p * np.log(p + 1e-12))))
    print(f"   [context only] spectral effective rank of the step basis = "
          f"{spec_eff:.2f}  (of {X.shape[0]} steps)")
    print(f"   [context only] top-5 energy fraction = "
          f"{100*np.sum(p[:5]):.1f}%")

    # ---------- (b) CAUSAL RANK ----------
    print(f"\n-- CAUSAL RANK: project dTheta to rank k, TRANSPLANT, MEASURE --")
    print(f"   {'k':>4}{'val':>10}{'nats':>9}{'% of descent':>14}")
    print("   " + "-" * 38)
    causal = []
    for k in args.ks:
        if k > Vt.shape[0]:
            continue
        B = Vt[:k]                                     # (k, P) orthonormal
        coef = B @ dtheta
        proj = B.T @ coef                              # rank-k projection
        newsd = {kk: v.clone() for kk, v in start.items()}
        pd = unflatten(torch.tensor(proj), start, keys)
        for kk in keys:
            newsd[kk] = (start[kk] + pd[kk]).clone()
        load_snapshot(model, newsd)
        v = eval_val(model, n=8)
        nats = v0 - v
        pct = 100.0 * nats / max(total, 1e-9)
        causal.append({"k": k, "val": float(v), "nats": float(nats),
                       "pct": float(pct)})
        print(f"   {k:>4}{v:>10.4f}{nats:>9.3f}{pct:>13.1f}%")
    load_snapshot(model, start)

    # full displacement -> must reproduce the floor (sanity)
    newsd = {kk: v.clone() for kk, v in start.items()}
    fd = unflatten(torch.tensor(dtheta), start, keys)
    for kk in keys:
        newsd[kk] = (start[kk] + fd[kk]).clone()
    load_snapshot(model, newsd)
    v_full = eval_val(model, n=8)
    print(f"   {'FULL':>4}{v_full:>10.4f}{v0-v_full:>9.3f}"
          f"{100*(v0-v_full)/max(total,1e-9):>13.1f}%   <- sanity "
          f"(loss subspace only; W_K left at start)")
    load_snapshot(model, start)

    # ---------- (3) INTERACTION RANK -- the primitive object ----------
    print("\n" + "=" * 74)
    print("  (3) INTERACTION RANK  (is the structure in INTERACTION space,")
    print("      not parameter space?)")
    print("=" * 74)

    def eval_sd(sd):
        load_snapshot(model, sd)
        return eval_val(model, n=6)

    Gs, M, solo = interaction_matrix(model, start, floor, eval_sd, LOSS_GROUPS)
    load_snapshot(model, start)
    print("   pairwise interaction gains  M[i][j] = dL(i+j) - dL(i) - dL(j)")
    print("   (diagonal = solo gains)\n")
    hdr = "        " + "".join(f"{g[:6]:>8}" for g in Gs)
    print(hdr)
    for i, g in enumerate(Gs):
        print(f"   {g[:6]:>5} " + "".join(f"{M[i][j]:>8.3f}" for j in range(len(Gs))))
    sv = np.linalg.svd(M, compute_uv=False)
    pM = sv / (sv.sum() + 1e-12)
    eff_M = float(np.exp(-np.sum(pM * np.log(pM + 1e-12))))
    top1 = 100 * sv[0] / (sv.sum() + 1e-12)
    print(f"\n   interaction-matrix singular values: "
          f"{np.array2string(sv, precision=3)}")
    print(f"   effective rank of the INTERACTION object = {eff_M:.2f} "
          f"(of {len(Gs)})")
    print(f"   top singular direction carries {top1:.1f}% of it")
    inter_low = eff_M <= 2.5
    print(f"   --> interaction is {'LOW-rank' if inter_low else 'HIGH-rank'}")

    # ---------- VERDICT on rank ----------
    print("\n" + "=" * 74)
    print("  VERDICT")
    print("=" * 74)
    recov = (v0 - v_full)
    k90 = next((c["k"] for c in causal if c["nats"] >= 0.90 * recov), None)
    k50 = next((c["k"] for c in causal if c["nats"] >= 0.50 * recov), None)
    print(f"  full loss-subspace displacement recovers {recov:.3f} nats")
    print(f"  rank needed for 50% of that: k = {k50}")
    print(f"  rank needed for 90% of that: k = {k90}")
    param_low = (k90 is not None and k90 <= 20)

    print("\n  MAPPING ONTO THE PRE-SPECIFIED OUTCOMES:")
    print("  " + "-" * 68)
    if param_low:
        print(f"   parameter rank  : LOW  (k90 = {k90})")
        print("   causal transplant: REPRODUCES the descent")
        print("\n   => LOW-DIMENSIONAL CAUSAL MECHANISM.")
        print("      The descent is recoverable from ~%d directions." % k90)
        print("      This is the outcome under which a reduced-order model --")
        print("      and, if the structure proves smooth, a cone/restriction-map")
        print("      construction -- has something concrete to be built on.")
    elif inter_low:
        print("   parameter rank  : HIGH (no small k reproduces the descent)")
        print(f"   interaction rank: LOW  (eff_rank = {eff_M:.2f})")
        print("\n   => THE STRUCTURE LIVES IN INTERACTION SPACE, NOT PARAMETER")
        print("      SPACE.  Parameter updates are not compressible, but the")
        print("      co-adaptation between blocks is.  The object to model is")
        print("      the interaction geometry -- exactly what the transplant")
        print("      results pointed at.")
    else:
        print("   parameter rank  : HIGH")
        print(f"   interaction rank: HIGH (eff_rank = {eff_M:.2f})")
        print("\n   => CO-ADAPTATION APPEARS GENUINELY DISTRIBUTED.")
        print("      Neither the updates nor the interactions compress.")
        print("      Simple reduced-order models are correspondingly less")
        print("      plausible, and a conserved quantity -- if one exists --")
        print("      is a function on a high-dimensional object with no")
        print("      preferred handle in these coordinates.")

    print("\n  SCOPE (deliberately cautious):")
    print("   These experiments give evidence about STATE-DEPENDENT")
    print("   COMPATIBILITY CONSTRAINTS between parameter subspaces.")
    print("   They do NOT by themselves establish that the admissible models")
    print("   form a smooth manifold: interior optima under partial transplant")
    print("   can also arise from nonlinear parameterisation, optimizer")
    print("   history, local curvature, or multiple minima.  Manifold language")
    print("   is earned only if the rank/trajectory analysis above shows smooth")
    print("   low-dimensional structure.")

    # ---------- (c) INVARIANTS ACROSS THE TRANSITION ----------
    print("\n" + "=" * 74)
    print("  CANDIDATE INVARIANTS: which stay FLAT across crystallisation?")
    print("=" * 74)
    pre = [d for d in inv_track if d["val"] > 0.30]
    post = [d for d in inv_track if d["val"] <= 0.30]
    fields = ["tau", "E", "fisher_norm", "emb_ff_coupling", "ntk_trace",
              "grad_eff_rank"]
    Lpre = [d["val"] for d in pre]
    Lpost = [d["val"] for d in post] if post else [1.0]
    print(f"  {'quantity':<18}{'cv_pre':>9}{'cv_post':>9}"
          f"{'stab_pre':>10}{'stab_post':>11}{'verdict':>21}")
    print("  " + "-" * 78)
    print("  (stab = Var(I)/Var(L): <<1 means far more stable than the loss")
    print("   it would be an invariant OF; ~1 means it merely tracks the loss)")
    inv_verdict = {}
    for f in fields:
        a = np.array([d[f] for d in pre], float)
        b = np.array([d[f] for d in post], float) if post else np.array([np.nan])
        st_a = stability_score(a, Lpre)
        st_b = stability_score(b, Lpost) if post else float("nan")
        cva = float(np.std(a) / (abs(np.mean(a)) + 1e-12))
        cvb = float(np.std(b) / (abs(np.mean(b)) + 1e-12)) if post else float("nan")
        flat_pre, flat_post = cva < 0.10, (cvb < 0.10)
        if flat_pre and flat_post:
            v = "CONSERVED (both)"
        elif flat_post and not flat_pre:
            v = "conserved LATE only"
        elif flat_pre and not flat_post:
            v = "conserved EARLY only"
        else:
            v = "not conserved"
        inv_verdict[f] = {"cv_pre": cva, "cv_post": cvb, "verdict": v,
                          "stability_pre": st_a, "stability_post": st_b}
        print(f"  {f:<18}{cva:>9.4f}{cvb:>9.4f}{st_a:>10.3f}{st_b:>11.3f}"
              f"{v:>21}")
    print("\n  A quantity conserved on BOTH sides but with DIFFERENT level sets")
    print("  is a symplectic invariant crossing a wall -- that is the object")
    print("  the cone construction wants.  'conserved LATE only' is the")
    print("  crystallisation signature.")

    json.dump({"total_descent": total, "spectral_eff_rank": spec_eff,
               "interval_rank": ir,
               "interaction": {"groups": Gs, "matrix": M.tolist(),
                               "solo": solo, "eff_rank": eff_M,
                               "low_rank": bool(inter_low)},
               "causal_rank": causal, "full_recovery": recov,
               "k50": k50, "k90": k90,
               "invariants": inv_track, "invariant_verdict": inv_verdict},
              open("rank_test.json", "w"), indent=2, default=float)
    with open("rank_test.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["k", "val", "nats", "pct_of_descent"])
        for c in causal:
            w.writerow([c["k"], f"{c['val']:.5f}", f"{c['nats']:.5f}",
                        f"{c['pct']:.2f}"])
    print("\n  wrote rank_test.json / .csv")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
        ks = [c["k"] for c in causal]
        ax[0].plot(ks, [c["pct"] for c in causal], "o-", color="#4c72b0")
        ax[0].axhline(90, ls="--", color="red", label="90% of recoverable")
        ax[0].set_xlabel("rank k of the projection")
        ax[0].set_ylabel("% of descent recovered (CAUSAL transplant)")
        ax[0].set_title("Is the loss-carrying flow low-dimensional?\n"
                        "(projected, transplanted, measured -- not fitted)")
        ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)

        st = [d["step"] for d in inv_track]
        for f, c in [("tau", "#dd8452"), ("emb_ff_coupling", "#c44e52"),
                     ("grad_eff_rank", "#55a868")]:
            v = np.array([d[f] for d in inv_track], float)
            v = v / (abs(v).max() + 1e-12)
            ax[1].plot(st, v, marker=".", label=f, color=c)
        Ev = np.array([d["E"] for d in inv_track], float)
        ax[1].plot(st, Ev / Ev.max(), marker=".", label="E (W_K control)",
                   color="#8172b3", ls=":")
        cross = next((d["step"] for d in inv_track if d["val"] <= 0.30), None)
        if cross:
            ax[1].axvline(cross, color="k", ls="--", alpha=0.5,
                          label="crystallisation")
        ax[1].set_xlabel("Phase-3 step")
        ax[1].set_ylabel("normalised value")
        ax[1].set_title("Candidate invariants of the LOSS flow\n"
                        "(flat = conserved)")
        ax[1].grid(alpha=0.3); ax[1].legend(fontsize=8)
        plt.suptitle("Rank test: does the cone construction have a substrate?",
                     fontsize=13, weight="bold")
        plt.tight_layout()
        plt.savefig("rank_test.png", dpi=190)
        print("  wrote rank_test.png")
    except Exception as e:
        print(f"  [plot skipped: {e}]")


if __name__ == "__main__":
    main()
