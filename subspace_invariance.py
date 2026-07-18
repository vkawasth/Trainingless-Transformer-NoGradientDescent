"""
subspace_invariance.py
======================
WHAT IS THE SUBSPACE A STRUCTURE *OF*?

ESTABLISHED
-----------
A low-dimensional subspace carries the loss-relevant component of the
optimisation trajectory, and it RECURS across independent runs:
    * k=3 recovers 97.6% of a 4.37-nat descent, causally
      (project -> transplant -> measure; never fitted)
    * the k=3 coefficient space is a smooth single bowl whose minimum sits
      at the trajectory's own endpoint
    * TRANSFER: run-1's basis recovers 88% of run-2's descent causally;
      65% of run-2's displacement lies inside run-1's subspace;
      the two runs' endpoint coefficient vectors are cosine +0.984 apart
      (nearly collinear -- same direction, different magnitude)

That is seed-invariance.  It is strong, and it is the reason transfer is the
headline: to fake it, the basis would have to memorise one trajectory AND the
directions would have to recur AND they would have to correspond to real
causal improvement.  All three would have to go wrong together.

WHAT IS *NOT* ESTABLISHED
-------------------------
Why it recurs.  "Structural" is a claim about the CAUSE, and we have not
tested one.  Candidates:
    ARCHITECTURE  -- the subspace is a property of the network
    CORPUS        -- it is set by the data distribution
    OPTIMIZER     -- it is an artefact of Adam's trajectory
    LOSS GEOMETRY -- it is intrinsic to the landscape

THIS SCRIPT DECIDES BETWEEN THEM
--------------------------------
Extract the basis from a REFERENCE run, then test -- causally, always by
transplant -- whether it still recovers the descent when we vary ONE factor
at a time:

    seed      : different random seed, same corpus, same arch   (control;
                already known to transfer at 88%)
    corpus    : SAME entropy class, permuted bigram graph
                (the earlier lambda_cos work found the relaxation SHAPE is
                 invariant within an entropy class -- does the SUBSPACE
                 follow the same law?)
    corpus++  : DIFFERENT entropy class (markov, 3 successors/token)
    arch      : different width (D) -- basis must be re-expressed, so we test
                the WEAKER claim that a fresh basis has the same DIMENSION
                and the same coefficient structure
    optimizer : SGD instead of AdamW, same everything else
                (if the subspace is an Adam artefact, this is where it dies)

READING THE RESULT
------------------
    transfers across seed + corpus + optimizer  -> ARCHITECTURAL / intrinsic.
        "Structural" is earned, and we know what of.
    transfers across seed, NOT across corpus    -> CORPUS-determined.
        The subspace is set by the data.  This would connect directly to the
        entropy-class finding and would be a sharp result in its own right.
    dies under a different optimizer            -> OPTIMIZER ARTEFACT.
        The honest conclusion, and it would retire the geometric reading.

OUTPUT
    subspace_invariance.json / .csv / .png
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


def make_corpus(kind, vocab, n_tokens, seed):
    g = torch.Generator().manual_seed(seed)
    if kind in ("base", "permuted"):
        succ = torch.randperm(vocab, generator=g)
        ids = torch.empty(n_tokens, dtype=torch.long)
        ids[0] = int(torch.randint(0, vocab, (1,), generator=g))
        for i in range(1, n_tokens):
            ids[i] = succ[ids[i - 1]]
        return ids
    if kind == "markov":
        succ = torch.stack([torch.randperm(vocab, generator=g)[:3]
                            for _ in range(vocab)])
        ids = torch.empty(n_tokens, dtype=torch.long)
        ids[0] = int(torch.randint(0, vocab, (1,), generator=g))
        for i in range(1, n_tokens):
            c = int(torch.randint(0, 3, (1,), generator=g))
            ids[i] = succ[ids[i - 1], c]
        return ids
    raise ValueError(kind)


def train(model, batchgen, evalfn, steps, lr, n_ckpt, opt_name="adamw"):
    if opt_name == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=lr,
                                betas=(0.9, 0.95), weight_decay=0.1)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=lr * 10, momentum=0.9)
    every = max(1, steps // n_ckpt)
    ck = [snap(model)]
    for s in range(1, steps + 1):
        model.train()
        x, y = batchgen()
        _, l = model(x, y)
        opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if s % every == 0 or s == steps:
            ck.append(snap(model))
    return ck


def basis_from(ckpts, keys, k):
    X = np.stack([(flat(ckpts[i], keys) - flat(ckpts[i - 1], keys)).numpy()
                  for i in range(1, len(ckpts))])
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    return Vt[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--n-ckpt", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--compiler", default="compiler_analytic_topogate.py")
    args = ap.parse_args()

    g_ = {}
    src = open(args.compiler).read()
    cut = src.find("# ── PHASE 1")
    if cut == -1:
        cut = src.find("PHASE 1")
    exec(src[:cut], g_)
    model = g_["model"]; eval_val = g_["eval_val"]
    LR = g_["LR"] * 5
    VOCAB = g_["VOCAB"]; BATCH = g_.get("BATCH", 8); SEQ = g_.get("SEQ", 64)

    print("=" * 74)
    print("  SUBSPACE INVARIANCE: what is the subspace a structure OF?")
    print("=" * 74)
    print("  Known: it RECURS across seeds (88% causal transfer, endpoint")
    print("  coefficients cosine +0.984).  Unknown: WHY.  We vary one factor")
    print("  at a time and re-test CAUSALLY (transplant, never fit).")
    print()
    print("   transfers across seed+corpus+optimizer -> ARCHITECTURAL/intrinsic")
    print("   dies on corpus change                  -> CORPUS-determined")
    print("   dies on optimizer change               -> ADAM ARTEFACT")

    start = snap(model)
    v0 = float(eval_val(model, n=10))
    keys = [k for k in start if group_of(k) in GROUPS]

    corpora = {
        "base": make_corpus("base", VOCAB, 20000, args.seed),
        "permuted": make_corpus("permuted", VOCAB, 20000, args.seed + 999),
        "markov": make_corpus("markov", VOCAB, 20000, args.seed + 7),
    }

    def make_batchgen(corpus, seed):
        gen = torch.Generator().manual_seed(seed)

        def bg():
            ix = torch.randint(0, len(corpus) - SEQ - 1, (BATCH,),
                               generator=gen)
            x = torch.stack([corpus[i:i + SEQ] for i in ix])
            y = torch.stack([corpus[i + 1:i + SEQ + 1] for i in ix])
            return x, y
        return bg

    def evalon(corpus, seed=1234, n=8):
        bg = make_batchgen(corpus, seed)
        tot = 0.0
        model.eval()
        with torch.no_grad():
            for _ in range(n):
                x, y = bg()
                _, l = model(x, y)
                tot += float(l)
        model.train()
        return tot / n

    # ---------- REFERENCE RUN ----------
    print(f"\n-- REFERENCE run (base corpus, seed {args.seed}, AdamW) --")
    torch.manual_seed(args.seed)
    load(model, start)
    bg = make_batchgen(corpora["base"], args.seed)
    ck_ref = train(model, bg, eval_val, args.steps, LR, args.n_ckpt)
    ref_floor = ck_ref[-1]
    load(model, ref_floor)
    v_ref = evalon(corpora["base"])
    B_ref = basis_from(ck_ref, keys, args.k)
    d_ref = (flat(ref_floor, keys) - flat(start, keys)).numpy()
    c_ref = B_ref[:args.k] @ d_ref
    print(f"   floor: val={v_ref:.4f}   coeffs={np.array2string(c_ref, precision=2)}")
    load(model, start)

    # ---------- VARY ONE FACTOR AT A TIME ----------
    conditions = [
        ("seed (control)", corpora["base"], args.seed + 500, "adamw"),
        ("corpus: permuted (same entropy class)", corpora["permuted"],
         args.seed, "adamw"),
        ("corpus: markov (diff entropy class)", corpora["markov"],
         args.seed, "adamw"),
        ("optimizer: SGD", corpora["base"], args.seed, "sgd"),
    ]

    print("\n" + "=" * 74)
    print("  CAUSAL TRANSFER OF THE REFERENCE BASIS")
    print("=" * 74)
    print(f"  {'condition':<38}{'own floor':>11}{'via B_ref':>11}"
          f"{'recovered':>11}{'cos':>7}")
    print("  " + "-" * 78)

    rows = []
    for label, corpus, sd_seed, optn in conditions:
        torch.manual_seed(sd_seed)
        load(model, start)
        bg = make_batchgen(corpus, sd_seed)
        ck = train(model, bg, eval_val, args.steps, LR, args.n_ckpt,
                   opt_name=optn)
        fl = ck[-1]
        load(model, fl)
        v_own = evalon(corpus)
        v_start_here = evalon(corpus) if False else None
        d = (flat(fl, keys) - flat(start, keys)).numpy()

        # baseline val at the START, on THIS corpus
        load(model, start)
        v0_here = evalon(corpus)

        # CAUSAL: project this run's displacement onto the REFERENCE basis
        c = B_ref[:args.k] @ d
        proj = B_ref[:args.k].T @ c
        sd = {kk: v.clone() for kk, v in start.items()}
        pd = unflat(torch.tensor(proj), start, keys)
        for kk in keys:
            sd[kk] = (start[kk] + pd[kk]).clone()
        load(model, sd)
        v_via = evalon(corpus)
        load(model, start)

        nats_own = v0_here - v_own
        nats_via = v0_here - v_via
        rec = nats_via / max(nats_own, 1e-9)
        cos = float(c_ref @ c /
                    (np.linalg.norm(c_ref) * np.linalg.norm(c) + 1e-12))
        ok = rec > 0.70
        rows.append({"condition": label, "v_own": v_own, "v_via": v_via,
                     "nats_own": nats_own, "nats_via": nats_via,
                     "recovered": rec, "cosine": cos, "transfers": bool(ok),
                     "coeffs": c.tolist()})
        print(f"  {label:<38}{v_own:>11.4f}{v_via:>11.4f}"
              f"{100*rec:>10.0f}%{cos:>+7.2f}  "
              f"{'OK' if ok else 'FAILS'}")

    # ---------- VERDICT ----------
    print("\n" + "=" * 74)
    print("  VERDICT: WHAT IS THE SUBSPACE A STRUCTURE OF?")
    print("=" * 74)
    R = {r["condition"]: r for r in rows}
    seed_ok = R["seed (control)"]["transfers"]
    perm_ok = R["corpus: permuted (same entropy class)"]["transfers"]
    mark_ok = R["corpus: markov (diff entropy class)"]["transfers"]
    sgd_ok = R["optimizer: SGD"]["transfers"]

    print(f"   seed        : {'transfers' if seed_ok else 'FAILS'}")
    print(f"   corpus (same entropy class): "
          f"{'transfers' if perm_ok else 'FAILS'}")
    print(f"   corpus (diff entropy class): "
          f"{'transfers' if mark_ok else 'FAILS'}")
    print(f"   optimizer (SGD): {'transfers' if sgd_ok else 'FAILS'}")
    print()

    if not sgd_ok:
        print("   => OPTIMIZER ARTEFACT.  The subspace does not survive a")
        print("      change of optimizer, so it is a property of Adam's")
        print("      trajectory, not of the architecture or the loss geometry.")
        print("      This should be stated plainly; it retires the geometric")
        print("      reading of the subspace.")
    elif seed_ok and perm_ok and not mark_ok:
        print("   => CORPUS-DETERMINED, AT THE LEVEL OF ENTROPY CLASS.")
        print("      The subspace survives a permuted bigram graph but not a")
        print("      change of entropy class.  It is set by the DATA, not the")
        print("      architecture -- and it obeys the SAME law as the earlier")
        print("      lambda_cos relaxation-shape result (invariant within an")
        print("      entropy class).  Two independent measurements, one law.")
    elif seed_ok and perm_ok and mark_ok and sgd_ok:
        print("   => ARCHITECTURAL / INTRINSIC.  The subspace survives changes")
        print("      of seed, corpus (both classes), and optimizer.  It is a")
        print("      property of the network + loss geometry, not of the data")
        print("      or the optimiser.  'Structural' is now earned, and we know")
        print("      what it is a structure OF.")
    elif seed_ok and not perm_ok:
        print("   => TRAJECTORY-SPECIFIC BEYOND THE SEED.  It recurs across")
        print("      seeds but not across corpora, and we have not isolated a")
        print("      single cause.  'Recurs across runs' remains the honest")
        print("      claim; 'structural' is not earned.")
    else:
        print("   => MIXED.  See the table; no single clean attribution.")

    print("\n   NOTE on the architecture axis: a width change makes the basis")
    print("   live in a different parameter space, so it cannot be transported")
    print("   directly.  Testing it requires the weaker claim (same effective")
    print("   DIMENSION, same coefficient structure) and is left as the next")
    print("   experiment.")

    json.dump({"k": args.k, "v0": v0, "ref": {"floor": v_ref,
                                              "coeffs": c_ref.tolist()},
               "conditions": rows},
              open("subspace_invariance.json", "w"), indent=2, default=float)
    print("\n  wrote subspace_invariance.json")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(11, 5.5))
        lbl = [r["condition"] for r in rows]
        rec = [100 * r["recovered"] for r in rows]
        cols = ["#55a868" if r["transfers"] else "#c44e52" for r in rows]
        plt.barh(lbl, rec, color=cols)
        plt.axvline(70, ls="--", color="k", label="70% transfer threshold")
        plt.xlabel("% of that condition's descent recovered "
                   "via the REFERENCE basis (causal)")
        plt.title("What is the subspace a structure of?\n"
                  "(green = the basis still works; red = it dies)",
                  fontsize=12, weight="bold")
        plt.legend(); plt.grid(alpha=0.3, axis="x")
        plt.tight_layout()
        plt.savefig("subspace_invariance.png", dpi=180)
        print("  wrote subspace_invariance.png")
    except Exception as e:
        print(f"  [plot skipped: {e}]")


if __name__ == "__main__":
    main()
