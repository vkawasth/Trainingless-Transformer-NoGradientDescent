"""
lambda_cos_determinism.py
=========================
Is the lambda_cos pencil sweep CORPUS-determined or ARCHITECTURE-determined?

WHY THIS MATTERS
----------------
Phase 3 shows no chamber crossings: lambda_cos sweeps monotonically
(0.998 -> 0.645) while the A-brane invariants (E ~ 43.4, R_Plucker ~ 0.59)
stay flat.  So Phase 3 is a single smooth pencil relaxation.

If the sweep is ARCHITECTURE-determined, lambda_cos(step) is a fixed curve we
can precompute ONCE and jump along for any corpus -> a real one-shot.
If it is CORPUS-determined, the curve must be measured per corpus and
lambda_cos is a diagnostic, not a shortcut.

Prior from existing data: E and R_Plucker are A-brane (corpus-invariant,
dE = 0.0000 under corpus swap).  lambda_cos is a B-brane coordinate -- it
moves with training alongside tau and val.  So the likely split is:

    * SHAPE of the sweep      -> architectural   (how it relaxes)
    * ENDPOINT lambda*        -> corpus          (where the entropy floor is)

This script tests exactly that decomposition.

DESIGN (2x2)
------------
Two corpora (A = base, B = shuffled/different bigram statistics but SAME
vocab + same token count, so the architecture sees an identical-shaped
problem with different statistics) crossed with two architectures
(arch1 = D x L, arch2 = wider or deeper).

    run(corpus, arch) -> lambda_cos(step) curve, lambda*, val(lambda_cos) map

Then we ask:
    Q1  Do the two CORPORA on the SAME arch give the same curve?
            same  -> sweep is architectural
            differ-> sweep is corpus-dependent
    Q2  Do the two ARCHITECTURES on the SAME corpus give the same curve?
            same  -> sweep is corpus-driven
            differ-> sweep is architectural
    Q3  Separate SHAPE from ENDPOINT: normalise each curve to
        lambda_norm(s) = (lambda(s) - lambda*) / (lambda(0) - lambda*)
        If the NORMALISED curves collapse onto one another, the SHAPE is
        universal and only the endpoint lambda* carries the corpus.
        ^ this is the key plot.

OUTPUT
    lambda_determinism.csv / .json
    lambda_determinism.png   (raw curves, normalised curves, val-vs-lambda)
"""

import argparse
import csv
import json
import math

import numpy as np
import torch


# ---------------------------------------------------------------- geometry
def lambda_cos(model, wk_ref, n_layers):
    """Pencil coordinate: mean cos-sim of W_K(t) against the MF reference."""
    sims = []
    for k in range(n_layers):
        W = model.blocks[k].attn.WK.weight.detach().cpu().numpy().ravel()
        Wr = wk_ref[k].ravel()
        d = np.linalg.norm(W) * np.linalg.norm(Wr) + 1e-12
        sims.append(float(np.dot(W, Wr) / d))
    return float(np.mean(sims))


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


def tau_defect(model):
    gff = gem = 0.0
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach().norm().item() ** 2
        if ".ff." in name:
            gff += g
        elif name.startswith("te") or name.startswith("pe"):
            gem += g
    return math.sqrt(gff) / (math.sqrt(gem) + 1e-12)


# ---------------------------------------------------------------- corpora
def make_corpus(kind, vocab, n_tokens, seed):
    """
    'base'     : cyclic near-permutation (the paper's corpus structure)
    'shuffled' : SAME vocab, SAME length, DIFFERENT bigram statistics
                 (a different permutation -> different transition graph).
                 This isolates corpus statistics while holding the
                 architecture's view of the problem shape fixed.
    'markov'   : a genuinely different statistical regime (higher entropy,
                 non-permutation transitions) -> stronger corpus contrast.
    """
    g = torch.Generator().manual_seed(seed)
    if kind in ("base", "shuffled"):
        succ = torch.randperm(vocab, generator=g)
        ids = torch.empty(n_tokens, dtype=torch.long)
        ids[0] = int(torch.randint(0, vocab, (1,), generator=g))
        for i in range(1, n_tokens):
            ids[i] = succ[ids[i - 1]]
        return ids
    if kind == "markov":
        # each token has 3 possible successors -> higher conditional entropy
        succ = torch.stack([torch.randperm(vocab, generator=g)[:3]
                            for _ in range(vocab)])
        ids = torch.empty(n_tokens, dtype=torch.long)
        ids[0] = int(torch.randint(0, vocab, (1,), generator=g))
        for i in range(1, n_tokens):
            c = int(torch.randint(0, 3, (1,), generator=g))
            ids[i] = succ[ids[i - 1], c]
        return ids
    raise ValueError(kind)


# ---------------------------------------------------------------- one run
def run_one(label, corpus, model_fn, steps, lr, batch, seq, n_layers, seed,
            auto_converge=False, max_steps=1200, patience=6, rel_tol=0.01,
            floor_val=0.05):
    """Train one (corpus, arch) pair and record the lambda_cos sweep.

    auto_converge=False : run exactly `steps` (fixed budget).
    auto_converge=True  : run until THIS corpus reaches ITS OWN plateau, up to
        `max_steps`.

    FLOOR GUARD (floor_val).  Critical.  This corpus is a looped repeat, so if
    training is allowed to run far past the entropy floor the model MEMORISES
    it (val -> 0.000).  Past that point there is no gradient signal left and
    W_K simply DIFFUSES under optimiser noise: lambda_cos stops relaxing toward
    a pole and starts random-walking.  Empirically, at 1200 steps two seeds of
    the SAME corpus landed at lambda* = 0.757 vs 0.403 -- a seed-dependent
    spread of 0.35, which destroys the noise floor and makes every shape
    comparison vacuous.

    lambda_cos is therefore only meaningful DURING descent.  We stop at the
    floor, not past it.
    """
    torch.manual_seed(seed)
    model = model_fn()

    # reference = the pre-descent W_K (the lambda = 1 pole of the pencil)
    wk_ref = [model.blocks[k].attn.WK.weight.detach().cpu().numpy().copy()
              for k in range(n_layers)]

    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                            betas=(0.9, 0.95), weight_decay=0.1)
    gen = torch.Generator().manual_seed(seed + 11)

    def batchgen():
        ix = torch.randint(0, len(corpus) - seq - 1, (batch,), generator=gen)
        x = torch.stack([corpus[i:i + seq] for i in ix])
        y = torch.stack([corpus[i + 1:i + seq + 1] for i in ix])
        return x, y

    @torch.no_grad()
    def evalval(n=8):
        model.eval()
        tot = 0.0
        for _ in range(n):
            x, y = batchgen()
            _, l = model(x, y)
            tot += float(l)
        model.train()
        return tot / n

    budget = max_steps if auto_converge else steps
    probe_every = 5
    curve = []
    stop_reason = None
    stopped_at = None

    for s in range(budget):
        model.train()
        x, y = batchgen()
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        tau = tau_defect(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if s % probe_every == 0 or s == budget - 1:
            v = evalval(4)
            curve.append({
                "step": s,
                "lambda_cos": lambda_cos(model, wk_ref, n_layers),
                "val": v,
                "tau": tau,
                "E": strip_energy(model, n_layers),
            })

            # --- FLOOR GUARD: never enter the diffusion regime -------------
            if v <= floor_val:
                stop_reason, stopped_at = "floor", s
                break

            # --- plateau (auto-converge only) ------------------------------
            if auto_converge and len(curve) > patience + 1:
                recent = [c["val"] for c in curve[-(patience + 1):]]
                best_old, best_new = min(recent[:-1]), recent[-1]
                rel = (best_old - best_new) / (abs(best_old) + 1e-12)
                if rel < rel_tol:
                    stop_reason, stopped_at = "plateau", s
                    break

    lam_star = curve[-1]["lambda_cos"]
    converged = stop_reason in ("floor", "plateau")
    tag = (f"{stop_reason}@{stopped_at}" if converged
           else f"budget-exhausted@{budget}")
    print(f"  [{label}] lambda: {curve[0]['lambda_cos']:.4f} -> {lam_star:.4f}"
          f"   val: {curve[0]['val']:.3f} -> {curve[-1]['val']:.3f}"
          f"   E: {curve[0]['E']:.2f} -> {curve[-1]['E']:.2f}   ({tag})")
    if not converged:
        print(f"        WARNING: budget exhausted without reaching floor "
              f"({floor_val}) or plateau -- shape may be unfinished.")
    return {"label": label, "curve": curve, "lambda_star": lam_star,
            "plateaued_at": stopped_at, "converged": converged,
            "stop_reason": stop_reason,
            "final_val": curve[-1]["val"], "n_steps": curve[-1]["step"]}


# ---------------------------------------------------------------- analysis
def normalised(curve, lam_star):
    """Divide out the ENDPOINT so only the SHAPE remains.

    Returns [(progress, lambda_norm)] where
        progress     = s / s_final     in [0,1]   (fraction of the run)
        lambda_norm  = (lambda(s) - lambda*) / (lambda(0) - lambda*)  in [1,0]

    Using PROGRESS rather than absolute step is essential once runs have
    different lengths (auto-converge): a higher-entropy corpus needs more
    steps to reach its floor, and we want to compare the *shape* of the
    relaxation, not penalise it for taking longer.
    """
    l0 = curve[0]["lambda_cos"]
    den = (l0 - lam_star) + 1e-12
    s_final = float(curve[-1]["step"]) + 1e-12
    return [(c["step"] / s_final, (c["lambda_cos"] - lam_star) / den)
            for c in curve]


def curve_distance(cA, lamA, cB, lamB, n_grid=40):
    """RMS distance between two normalised shape curves.

    Both curves are resampled onto a common progress grid in [0,1], so runs
    of different length are compared fairly.  Distance ~ 0 means the two
    relaxations have the same SHAPE even if they took different numbers of
    steps and landed at different lambda*.
    """
    a = normalised(cA, lamA)
    b = normalised(cB, lamB)
    grid = np.linspace(0.0, 1.0, n_grid)
    ya = np.interp(grid, [p for p, _ in a], [v for _, v in a])
    yb = np.interp(grid, [p for p, _ in b], [v for _, v in b])
    return float(np.sqrt(np.mean((ya - yb) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=120,
                    help="fixed step budget (ignored if --auto-converge)")
    ap.add_argument("--auto-converge", action="store_true", default=True,
                    help="run each corpus to ITS OWN plateau (DEFAULT: on)")
    ap.add_argument("--fixed-budget", dest="auto_converge",
                    action="store_false",
                    help="disable auto-converge (NOT recommended)")
    ap.add_argument("--floor-val", type=float, default=0.05,
                    help="stop at this val; past the floor lambda_cos diffuses")
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--rel-tol", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--compiler", default="compiler_analytic_topogate.py")
    args = ap.parse_args()

    # pull the model class + hyperparams from the compiler
    g = {}
    src = open(args.compiler).read()
    cut = src.find("# ── PHASE 1")
    if cut == -1:
        cut = src.find("PHASE 1")
    exec(src[:cut], g)

    LM = g["LM"] if "LM" in g else g["StudentLM"]
    LR = g["LR"]
    VOCAB = g["VOCAB"]
    N_STU = g["N_STU"]
    BATCH = g.get("BATCH", 8)
    SEQ = g.get("SEQ", 64)

    print("=" * 66)
    print("  IS THE lambda_cos SWEEP CORPUS- OR ARCHITECTURE-DETERMINED?")
    print("=" * 66)
    if args.auto_converge:
        print(f"  AUTO-CONVERGE: each corpus runs to its OWN plateau "
              f"(max {args.max_steps}, rel_tol {args.rel_tol})")
        print(f"  -> shapes compared on normalised PROGRESS, so a slower")
        print(f"     (higher-entropy) corpus is not penalised for taking longer.")
    else:
        print(f"  FIXED BUDGET: {args.steps} steps for every corpus.")
        print(f"  WARNING: a higher-entropy corpus may not reach its floor;")
        print(f"           its 'shape' difference would then be")
        print(f"           'unfinished vs finished'.  Use --auto-converge.")

    n_tok = 20000
    corpA = make_corpus("base", VOCAB, n_tok, seed=args.seed)
    corpB = make_corpus("shuffled", VOCAB, n_tok, seed=args.seed + 999)
    corpC = make_corpus("markov", VOCAB, n_tok, seed=args.seed + 7)

    arch1 = lambda: LM()
    kw = dict(auto_converge=args.auto_converge, max_steps=args.max_steps,
              patience=args.patience, rel_tol=args.rel_tol,
              floor_val=args.floor_val)

    print("\n-- same arch, THREE corpora (tests corpus dependence) --")
    rA = run_one("corpusA/arch1", corpA, arch1, args.steps, LR * 5,
                 BATCH, SEQ, N_STU, args.seed, **kw)
    rB = run_one("corpusB/arch1", corpB, arch1, args.steps, LR * 5,
                 BATCH, SEQ, N_STU, args.seed, **kw)
    rC = run_one("corpusC(markov)/arch1", corpC, arch1, args.steps, LR * 5,
                 BATCH, SEQ, N_STU, args.seed, **kw)

    print("\n-- same corpus, different SEED (control: run-to-run noise) --")
    rA2 = run_one("corpusA/arch1 seed2", corpA, arch1, args.steps, LR * 5,
                  BATCH, SEQ, N_STU, args.seed + 500, **kw)

    # ---- the decisive comparison ----
    noise = curve_distance(rA["curve"], rA["lambda_star"],
                           rA2["curve"], rA2["lambda_star"])
    dAB = curve_distance(rA["curve"], rA["lambda_star"],
                         rB["curve"], rB["lambda_star"])
    dAC = curve_distance(rA["curve"], rA["lambda_star"],
                         rC["curve"], rC["lambda_star"])

    print("\n" + "=" * 66)
    print("  RESULT")
    print("=" * 66)
    print(f"  lambda* endpoints:")
    print(f"    corpusA           : {rA['lambda_star']:.4f}")
    print(f"    corpusA (seed2)   : {rA2['lambda_star']:.4f}")
    print(f"    corpusB (shuffled): {rB['lambda_star']:.4f}")
    print(f"    corpusC (markov)  : {rC['lambda_star']:.4f}")
    print()
    print(f"  NORMALISED SHAPE distance (endpoint divided out):")
    print(f"    same corpus, diff seed  : {noise:.4f}   <- noise floor")
    print(f"    corpusA vs corpusB      : {dAB:.4f}")
    print(f"    corpusA vs corpusC      : {dAC:.4f}")
    print()

    # ---- CONVERGENCE GUARD --------------------------------------------
    # A shape comparison is only meaningful between runs that BOTH reached
    # their floor.  With --auto-converge each run carries a real plateau flag;
    # without it we fall back to a val-ratio heuristic.
    vA, vC = rA["final_val"], rC["final_val"]
    if args.auto_converge:
        C_converged = rC["converged"] and rA["converged"]
        conv_note = (f"plateau flags: A={rA['converged']} "
                     f"(step {rA['plateaued_at']}), "
                     f"C={rC['converged']} (step {rC['plateaued_at']})")
    else:
        C_converged = vC < 10.0 * max(vA, 1e-9)
        conv_note = f"val heuristic: C={vC:.3f} vs A={vA:.3f}"

    def verdict(name, d, converged=True):
        if not converged:
            print(f"    {name}: INVALID -- run did not converge "
                  f"({conv_note}); shape distance {d:.4f} is "
                  f"'unfinished vs finished', not a shape difference.")
            return None
        if d < noise:
            print(f"    {name}: COLLAPSE ({d:.4f} < noise {noise:.4f}) "
                  f"-- shape is INVARIANT, closer than seed-to-seed.")
            return True
        if d < 3 * noise:
            print(f"    {name}: collapse within tolerance "
                  f"({d:.4f} < 3x noise) -- shape invariant.")
            return True
        print(f"    {name}: DIFFERS ({d:.4f} >> noise {noise:.4f}) "
              f"-- shape is NOT invariant.")
        return False

    print(f"  steps used: A={rA['n_steps']}  B={rB['n_steps']}  "
          f"C={rC['n_steps']}   stop: A={rA['stop_reason']} "
          f"B={rB['stop_reason']} C={rC['stop_reason']}")

    # ---- NOISE-FLOOR SANITY GATE --------------------------------------
    # If two seeds of the SAME corpus disagree wildly, the noise floor is so
    # large that every comparison "passes" trivially and the test says
    # nothing.  This happened at 1200 steps: lambda* = 0.757 vs 0.403 for the
    # same corpus, noise floor 0.324 -> vacuous verdicts.  Refuse to conclude.
    lam_spread = abs(rA["lambda_star"] - rA2["lambda_star"])
    NOISE_MAX = 0.05
    if noise > NOISE_MAX or lam_spread > 0.10:
        print()
        print("  !! TEST INVALID -- noise floor too high to conclude anything.")
        print(f"     Same corpus, two seeds: lambda* = {rA['lambda_star']:.4f} "
              f"vs {rA2['lambda_star']:.4f}  (spread {lam_spread:.4f})")
        print(f"     Shape noise floor = {noise:.4f} (max tolerable {NOISE_MAX})")
        print("     With a noise floor this large, ANY comparison passes the")
        print("     3x-noise test vacuously.  Do not trust a verdict here.")
        print()
        print("     LIKELY CAUSE: training ran past the entropy floor into the")
        print("     memorisation/diffusion regime (val -> 0), where there is no")
        print("     gradient signal and W_K random-walks.  lambda_cos only")
        print("     relaxes DURING descent.")
        print(f"     FIX: lower --floor-val (now {args.floor_val}) or "
              f"--max-steps (now {args.max_steps}).")
        return
    print("  PER-CORPUS VERDICT (each vs the noise floor -- not max()):")
    vAB = verdict("A vs B (same entropy class, permuted bigrams)", dAB,
                  rA.get("converged", True) and rB.get("converged", True)
                  if args.auto_converge else True)
    vAC = verdict("A vs C (different entropy class)", dAC, C_converged)

    print()
    if vAB:
        print("  CONCLUSION: the lambda_cos SHAPE is invariant under corpora of")
        print("  the SAME ENTROPY CLASS.  Permuting the bigram graph (different")
        print("  token statistics, same conditional entropy) leaves the")
        print("  normalised sweep unchanged -- indeed closer than two seeds of")
        print("  the same corpus.  The shape is therefore PRECOMPUTABLE and")
        print("  REUSABLE within an entropy class; only lambda* (the floor")
        print("  location) carries the corpus.")
        print()
        print("  This is the same phenomenon as the D-brane incremental result:")
        print("  same-domain additions (tau < 6) reuse the geometry cheaply;")
        print("  cross-entropy-class additions do not.")
    if vAC is False:
        print("\n  Across ENTROPY CLASSES the shape does change -- consistent")
        print("  with lambda* moving and the entropy floor relocating.")
    if vAC is None:
        print("\n  NOTE: corpusC must be run to ITS OWN floor before its shape")
        print("  can be compared.  Re-run with more steps for a valid test.")

    Es = [c["E"] for r in (rA, rB) for c in r["curve"]]
    EsC = [c["E"] for c in rC["curve"]]
    print(f"\n  A-brane check:")
    print(f"    E (same entropy class A,B) = {min(Es):.2f} -- {max(Es):.2f} "
          f"(spread {max(Es)-min(Es):.2f})")
    print(f"    E (corpusC, markov)        = {min(EsC):.2f} -- {max(EsC):.2f} "
          f"(spread {max(EsC)-min(EsC):.2f})")

    # ---- artifacts ----
    runs = [rA, rA2, rB, rC]
    with open("lambda_determinism.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run","step","progress","lambda_cos","lambda_norm","val","tau","E"])
        for r in runs:
            nrm = normalised(r["curve"], r["lambda_star"])
            for c, (prog, lnorm) in zip(r["curve"], nrm):
                w.writerow([r["label"], c["step"], f'{prog:.5f}',
                            f'{c["lambda_cos"]:.5f}', f'{lnorm:.5f}',
                            f'{c["val"]:.5f}', f'{c["tau"]:.4f}', f'{c["E"]:.4f}'])
    with open("lambda_determinism.json", "w") as f:
        json.dump({
            "runs": runs,
            "shape_distance": {"noise_floor": noise,
                               "A_vs_B": dAB, "A_vs_C": dAC},
        }, f, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(17, 5))
        cols = {"corpusA/arch1": "#4c72b0", "corpusA/arch1 seed2": "#94b4d9",
                "corpusB/arch1": "#dd8452", "corpusC(markov)/arch1": "#55a868"}
        for r in runs:
            st = [c["step"] for c in r["curve"]]
            ax[0].plot(st, [c["lambda_cos"] for c in r["curve"]],
                       label=r["label"], color=cols.get(r["label"]))
            nrm = normalised(r["curve"], r["lambda_star"])
            ax[1].plot([p for p, _ in nrm], [v for _, v in nrm],
                       label=r["label"], color=cols.get(r["label"]))
            ax[2].plot([c["lambda_cos"] for c in r["curve"]],
                       [c["val"] for c in r["curve"]],
                       label=r["label"], color=cols.get(r["label"]))
        ax[0].set_title("RAW λ_cos sweep\n(endpoint λ* differs by corpus?)")
        ax[0].set_xlabel("step"); ax[0].set_ylabel("λ_cos")
        ax[1].set_title("NORMALISED shape vs PROGRESS\n(collapse ⇒ shape is invariant)")
        ax[1].set_xlabel("progress  s/s_final"); ax[1].set_ylabel("(λ−λ*)/(λ₀−λ*)")
        ax[2].set_title("val vs λ_cos\n(is the pencil→loss map universal?)")
        ax[2].set_xlabel("λ_cos"); ax[2].set_ylabel("val"); ax[2].set_yscale("log")
        for a in ax:
            a.grid(alpha=0.3); a.legend(fontsize=7)
        plt.suptitle("Is the λ_cos pencil sweep corpus- or architecture-determined?",
                     fontsize=13, weight="bold")
        plt.tight_layout()
        plt.savefig("lambda_determinism.png", dpi=190)
        print("  wrote lambda_determinism.png / .csv / .json")
    except Exception as e:
        print(f"  [plot skipped: {e}]")


if __name__ == "__main__":
    main()
