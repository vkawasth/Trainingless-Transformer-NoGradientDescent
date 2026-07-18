"""
universal_subspace.py
=====================
THE HYPOTHESIS (yours, and it is sharper than what I built before):

    Low-dimensional structure is UNIVERSAL -- every optimizer compresses its
    trajectory into a small subspace -- but the PARTICULAR subspace is
    optimizer-specific.  If so, the geometric object is real; each optimizer
    simply traverses a different chart of it.  That is WORLD C, and it puts
    geometry back on the table.

WHY THE PREVIOUS SCRIPTS CANNOT TEST THIS
-----------------------------------------
The SGD result showed that AdamW's k=3 basis does not transfer to SGD.  I
wrongly read that as "the geometric reading is retired".  It is not: it only
shows the trajectory-derived basis is optimizer-specific.  It says nothing
about whether SGD ALSO compresses -- into a DIFFERENT k=3 subspace.  That is
the measurement missing from everything so far.

WHAT THIS SCRIPT MEASURES
-------------------------
For each optimizer O in {AdamW, SGD, SGD+momentum, RMSprop, plain GD}:

  (1) COMPRESSIBILITY, causally.  Build O's own basis from its own trajectory,
      project O's displacement to rank k, TRANSPLANT it, and MEASURE the loss.
      Never fitted.  If every optimizer reaches ~the same recovery at small k,
      then LOW-DIMENSIONALITY IS UNIVERSAL even though the subspaces differ.

  (2) PAIRWISE PRINCIPAL ANGLES between the optimizers' subspaces.
      Expected to be large -- that is the premise, not the finding.

  (3) THE DISCRIMINATOR: the UNION SPECTRUM.
      Stack all the bases: M = [U_1 | U_2 | ... | U_n]  (P x n*k).
      * If the subspaces are TANGENT PLANES OF ONE low-dim manifold, the union
        has an effective dimension m << n*k, and the singular values of M show
        a sharp CLIFF at m.
      * If they are UNRELATED, the union spans ~n*k dimensions and the spectrum
        is FLAT with no cliff.
      Pairwise angles CANNOT separate these two cases -- I checked: a shared
      5-dim manifold and 5 unrelated subspaces give mean pairwise angles of
      68 deg and 88 deg respectively, i.e. BOTH look "different".  Only the
      union spectrum distinguishes them.

HONESTY ABOUT THE INSTRUMENT
----------------------------
The cliff is FRAGILE to off-manifold noise.  Validated:
      noise 0%   -> cliff at the true dim, gap ratio ~1e12   (unmistakable)
      noise 2%   -> cliff still at the true dim, gap ratio only 1.7x
      noise 10%  -> washed out entirely
So we report the FULL union spectrum and the gap ratio, use a sensitive
criterion, and EXPLICITLY DECLINE TO CALL IT when the evidence is ambiguous
rather than forcing a verdict.  A null "cannot distinguish" is a legitimate
outcome and will be printed as such.

OUTPUT
    universal_subspace.json / .png
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


def principal_angles(A, B):
    Qa = np.linalg.qr(A)[0]
    Qb = np.linalg.qr(B)[0]
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return np.degrees(np.arccos(np.clip(s, -1.0, 1.0)))


def union_spectrum(bases):
    """Singular values of [U_1 | ... | U_n].  The DISCRIMINATOR."""
    M = np.concatenate(bases, axis=1)
    return np.linalg.svd(M, compute_uv=False)


def cliff(s, max_look):
    """Largest relative drop in the first `max_look` singular values."""
    s = s[:max_look + 1]
    r = s[:-1] / (s[1:] + 1e-12)
    i = int(np.argmax(r))
    return i + 1, float(r[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--n-ckpt", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr-scan", action="store_true", default=True,
                    help="auto-tune each optimizer's LR (ON by default). A "
                         "single shared LR made RMSprop diverge and SGD stall, "
                         "which invalidated the first run entirely.")
    ap.add_argument("--no-lr-scan", dest="lr_scan", action="store_false")
    ap.add_argument("--compiler", default="compiler_analytic_topogate.py")
    args = ap.parse_args()

    g_ = {}
    src = open(args.compiler).read()
    cut = src.find("# ── PHASE 1")
    if cut == -1:
        cut = src.find("PHASE 1")
    exec(src[:cut], g_)
    model = g_["model"]; get_batch = g_["get_batch"]; eval_val = g_["eval_val"]
    LR = g_["LR"] * 5

    print("=" * 78)
    print("  UNIVERSAL SUBSPACE?  Does EVERY optimizer compress -- into its")
    print("  OWN k-dim subspace -- and do those subspaces lie on one object?")
    print("=" * 78)
    print("  The SGD result showed AdamW's basis does not transfer to SGD.")
    print("  It did NOT show that SGD fails to compress.  If every optimizer")
    print("  compresses into a DIFFERENT low-dim subspace, and those subspaces")
    print("  are tangent planes of ONE manifold, then geometry is back")
    print("  (World C) and trajectory PCA was simply a biased estimator of the")
    print("  tangent space.")
    print()
    print("  DISCRIMINATOR: the UNION SPECTRUM of the bases.")
    print("    shared object -> union dim << n*k, sharp spectral CLIFF")
    print("    unrelated     -> union dim ~ n*k, FLAT spectrum")
    print("  Pairwise angles cannot separate these (verified: a shared 5-dim")
    print("  manifold and 5 unrelated subspaces give 68 vs 88 deg -- BOTH look")
    print("  'different').  Only the union spectrum can.")

    start = snap(model)
    keys = [k for k in start if group_of(k) in GROUPS]

    # ================================================================
    # CONTROLLED COMPARISON: every optimizer must see EXACTLY the same
    #   * initial weights            (start, snapshotted once)
    #   * sequence of minibatches    (replayed from a fixed list)
    #   * evaluation batches         (fixed held-out set)
    # The compiler's get_batch()/eval_val() draw from the GLOBAL RNG, and the
    # LR auto-scan + a differing number of eval calls consume that RNG
    # unevenly -- so without this, each optimizer would train on a DIFFERENT
    # data stream and the comparison would be confounded.  We pre-generate the
    # streams once and replay them identically for every optimizer.
    # ================================================================
    torch.manual_seed(args.seed)
    TRAIN_BATCHES = [get_batch() for _ in range(args.steps)]
    EVAL_BATCHES = [get_batch() for _ in range(12)]
    print(f"\n  CONTROLLED: same init, same {len(TRAIN_BATCHES)} training")
    print(f"  batches (replayed in order), same {len(EVAL_BATCHES)} eval")
    print("  batches -- for every optimizer.  The only difference is the")
    print("  update rule.")

    def eval_fixed(n=12):
        """Loss on the FIXED eval batches -- identical for every optimizer,
        and independent of how many RNG draws anything else has made."""
        model.eval()
        tot = 0.0
        with torch.no_grad():
            for (x, y) in EVAL_BATCHES[:n]:
                _, l = model(x, y)
                tot += float(l)
        model.train()
        return tot / min(n, len(EVAL_BATCHES))

    v0 = eval_fixed()
    print(f"  start val = {v0:.4f}  (on the fixed eval set)")

    # PER-OPTIMIZER LEARNING RATES.  A single shared LR was the bug in the
    # first run: RMSprop DIVERGED (floor 6.81 > start 4.46, descent -2.35 nats)
    # and SGD STALLED (descent 0.91 vs AdamW's 4.39).  Their "recovery"
    # numbers were then divide-by-tiny / divide-by-NEGATIVE artefacts
    # (-264,686,673,482%), and the resulting "compression is not universal"
    # verdict was an artefact, not a measurement.  Adaptive optimizers need a
    # much smaller LR than raw SGD.
    OPTS = {
        "AdamW":     lambda p: torch.optim.AdamW(p, lr=LR, betas=(0.9, 0.95),
                                                 weight_decay=0.1),
        "SGD":       lambda p: torch.optim.SGD(p, lr=LR * 300),
        "SGD+mom":   lambda p: torch.optim.SGD(p, lr=LR * 60, momentum=0.9),
        "RMSprop":   lambda p: torch.optim.RMSprop(p, lr=LR * 0.2),
        "Adagrad":   lambda p: torch.optim.Adagrad(p, lr=LR * 5),
    }
    if args.lr_scan:
        print("\n-- LR auto-scan (each optimizer gets an LR that actually "
              "trains) --")
        print("   Without this, a single shared LR diverged RMSprop and")
        print("   stalled SGD, and the resulting verdict was an artefact.")
        tuned = {}
        for name, mk in OPTS.items():
            best, best_v = None, 1e9
            for mult in [0.05, 0.2, 1.0, 5.0, 20.0, 100.0, 400.0]:
                torch.manual_seed(args.seed)
                load(model, start)
                base_lr = mk(model.parameters()).param_groups[0]["lr"]
                o = mk(model.parameters())
                for gg in o.param_groups:
                    gg["lr"] = base_lr * mult
                for (x, y) in TRAIN_BATCHES[:40]:   # probe on the SAME batches
                    model.train()
                    _, l = model(x, y)
                    o.zero_grad(); l.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    o.step()
                v = eval_fixed(6)
                if np.isfinite(v) and v < best_v:
                    best_v, best = v, base_lr * mult
            load(model, start)
            print(f"   {name:<10} lr={best:.2e}  (probe val {best_v:.3f})")

            def _mk(p, _mk0=mk, _lr=best):
                o = _mk0(p)
                for gg in o.param_groups:
                    gg["lr"] = _lr
                return o
            tuned[name] = _mk
        OPTS = tuned

    results, bases = {}, {}
    ref_descent = [None]          # AdamW's descent, the yardstick for the gate
    print("\n" + "=" * 78)
    print("  (1) DOES EVERY OPTIMIZER COMPRESS?  (causal: transplant, not fit)")
    print("=" * 78)
    print(f"  {'optimizer':<11}{'floor':>9}{'descent':>9}"
          f"{'k=1':>8}{'k=2':>8}{'k=3':>8}{'k=5':>8}{'FULL':>8}")
    print("  " + "-" * 70)

    for name, mk in OPTS.items():
        torch.manual_seed(args.seed)
        load(model, start)                    # SAME initial weights
        opt = mk(model.parameters())
        every = max(1, args.steps // args.n_ckpt)
        ck = [snap(model)]
        for s, (x, y) in enumerate(TRAIN_BATCHES, start=1):   # SAME batches
            model.train()
            _, l = model(x, y)
            opt.zero_grad(); l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if s % every == 0 or s == args.steps:
                ck.append(snap(model))
        floor = ck[-1]
        load(model, floor)
        v_floor = eval_fixed()                # SAME eval batches
        descent = v0 - v_floor
        if ref_descent[0] is None:
            ref_descent[0] = descent      # AdamW runs first; it is the yardstick

        X = np.stack([(flat(ck[i], keys) - flat(ck[i - 1], keys)).numpy()
                      for i in range(1, len(ck))])
        Vt = np.linalg.svd(X, full_matrices=False)[2]
        dth = (flat(floor, keys) - flat(start, keys)).numpy()

        row = {}
        for kk in (1, 2, 3, 5):
            B = Vt[:kk]
            proj = B.T @ (B @ dth)
            sd = {a: b.clone() for a, b in start.items()}
            pd = unflat(torch.tensor(proj), start, keys)
            for a in keys:
                sd[a] = (start[a] + pd[a]).clone()
            load(model, sd)
            v = eval_fixed()
            row[kk] = (v0 - v) / max(descent, 1e-9)
        sd = {a: b.clone() for a, b in start.items()}
        pd = unflat(torch.tensor(dth), start, keys)
        for a in keys:
            sd[a] = (start[a] + pd[a]).clone()
        load(model, sd)
        v_full = eval_fixed()
        row["FULL"] = (v0 - v_full) / max(descent, 1e-9)
        load(model, start)

        # ---- VALIDITY GATE ----
        # A subspace extracted from a run that DIVERGED or barely moved is
        # noise.  Including it destroys the union-spectrum discriminator (which
        # validation showed is wiped out by ~10% off-manifold noise) and makes
        # every recovery ratio meaningless (negative or tiny denominator).
        # Such runs are EXCLUDED and flagged, not silently averaged in.
        valid = descent > 0.30 * ref_descent[0] if ref_descent[0] else True
        status = "OK" if valid else ("DIVERGED" if descent < 0 else "STALLED")

        results[name] = {"floor": v_floor, "descent": descent,
                         "recovery": row, "valid": bool(valid),
                         "status": status}
        if valid:
            bases[name] = Vt[:args.k].T      # (P, k)

        def pc(x):
            return "  n/a  " if not valid else f"{100*x:>6.0f}%"
        print(f"  {name:<11}{v_floor:>9.4f}{descent:>9.3f}"
              f"{pc(row[1])}{pc(row[2])}{pc(row[3])}{pc(row[5])}"
              f"{pc(row['FULL'])}  {status}")

    # ---- is compression universal? ----
    k3 = {n: r["recovery"][3] for n, r in results.items() if r["valid"]}
    universal = len(k3) >= 3 and all(v > 0.70 for v in k3.values())
    print(f"\n  (over the {len(k3)} optimizers that actually trained)")
    print(f"  k=3 recovery: min={100*min(k3.values()):.0f}%  "
          f"max={100*max(k3.values()):.0f}%")
    print(f"  -> low-dimensionality is "
          f"{'UNIVERSAL across optimizers' if universal else 'NOT universal'}")
    if not universal:
        worst = min(k3, key=k3.get)
        print(f"     ({worst} only reaches {100*k3[worst]:.0f}% at k=3)")

    # ================================================================
    # (2) pairwise angles -- the premise, not the finding
    # ================================================================
    print("\n" + "=" * 78)
    print("  (2) ARE THE SUBSPACES DIFFERENT?  (pairwise principal angles)")
    print("=" * 78)
    nm = list(bases)
    excluded = [n for n, r in results.items() if not r["valid"]]
    if excluded:
        print(f"\n  EXCLUDED from the geometry (did not train): "
              f"{', '.join(excluded)}")
        print("   A basis from a diverged or stalled run is noise, and noise")
        print("   destroys the union-spectrum discriminator.  They are dropped,")
        print("   not silently averaged in.")
    if len(nm) < 3:
        print("\n  !! Fewer than 3 optimizers trained successfully.  The union")
        print("     spectrum cannot be interpreted.  Re-run with better LRs.")
        return
    ang = {}
    print("  " + "".join(f"{n[:8]:>10}" for n in nm))
    for i, a in enumerate(nm):
        line = f"  {a[:8]:<8}"
        for j, b in enumerate(nm):
            if i == j:
                line += f"{'--':>10}"
            else:
                m_ = principal_angles(bases[a], bases[b]).mean()
                ang[f"{a}|{b}"] = float(m_)
                line += f"{m_:>10.0f}"
        print(line)
    off = [v for kk, v in ang.items()]
    print(f"\n  mean off-diagonal angle = {np.mean(off):.0f} deg")
    print("  (large angles are EXPECTED -- that is the premise.  The question")
    print("   is whether the subspaces nevertheless lie on ONE object.)")

    # ================================================================
    # (3) THE DISCRIMINATOR -- union spectrum
    # ================================================================
    print("\n" + "=" * 78)
    print("  (3) DISCRIMINATOR: do they lie on a COMMON low-dim object?")
    print("=" * 78)
    Blist = [np.linalg.qr(bases[n])[0] for n in nm]
    s = union_spectrum(Blist)
    n_opt, k = len(nm), args.k
    total = n_opt * k
    print(f"   stacking {n_opt} bases x k={k}  ->  union could span up to {total}")
    print(f"   union singular values:")
    print(f"     {np.array2string(s[:min(total, 16)], precision=3)}")

    m_, gap = cliff(s, min(total - 1, 2 * k + 3))
    e = s ** 2 / (s ** 2).sum()
    eff = float(np.exp(-np.sum(e * np.log(e + 1e-12))))
    print(f"\n   spectral cliff at dim {m_}  (gap ratio {gap:.2f}x)")
    print(f"   participation-ratio effective dim = {eff:.2f}  (of {total})")

    # sensitive criterion + explicit refusal to over-call
    shared = (gap > 1.5 and m_ <= 2 * k)
    flatish = (gap < 1.2)
    print()
    if shared:
        print(f"   => CLIFF DETECTED at dim {m_} << {total}.")
        print("      The optimizer subspaces are NOT unrelated: they behave")
        print("      like tangent planes of a COMMON low-dimensional object.")
        print("      This is the WORLD C signature.  Geometry is back on the")
        print("      table, and trajectory PCA was a biased estimator of the")
        print("      tangent space -- not evidence against the manifold.")
    elif flatish and eff > 0.8 * total:
        print(f"   => FLAT SPECTRUM (eff dim {eff:.1f} of {total}, no cliff).")
        print("      The subspaces span nearly the full union.  They look")
        print("      UNRELATED, not like charts of one object.  That is")
        print("      evidence for World D: each optimizer compresses, but into")
        print("      its own private subspace with no shared substrate.")
    else:
        print("   => CANNOT CALL IT.  The spectrum is neither a clean cliff nor")
        print("      clearly flat.  This instrument is FRAGILE to off-manifold")
        print("      noise: validation showed that at 2% noise a genuine shared")
        print("      manifold produces a gap ratio of only 1.7x, and at 10% the")
        print("      signal is gone entirely.  We decline to force a verdict.")
        print("      To sharpen: more optimizers, longer runs, larger k, or an")
        print("      explicit denoising of the bases.")

    # ================================================================
    # (3b) THE UNION AMONG OPTIMIZERS THAT ACTUALLY COMPRESS
    # ================================================================
    # A basis that explains only 40% of its OWN trajectory is 60% noise, and
    # validation showed noise destroys the cliff (2% -> gap 1.7x; 10% -> gone).
    # Including such a basis in the union is precisely the failure mode of this
    # instrument.  So we re-run the discriminator over ONLY those optimizers
    # whose own k-dim basis explains most of their own descent.
    union_comp = None
    COMPRESS_MIN = 0.70
    comp = [n for n in nm if results[n]["recovery"][args.k] >= COMPRESS_MIN]
    noncomp = [n for n in nm if n not in comp]
    print("\n" + "=" * 78)
    print("  (3b) UNION AMONG OPTIMIZERS THAT ACTUALLY COMPRESS")
    print("=" * 78)
    print(f"   compressing (k={args.k} recovery >= {COMPRESS_MIN:.0%}): "
          f"{', '.join(comp) if comp else 'none'}")
    if noncomp:
        for n in noncomp:
            r = results[n]["recovery"][args.k]
            print(f"   EXCLUDED: {n} -- its own basis explains only {100*r:.0f}%")
            print(f"             of its own descent, so it is ~{100*(1-r):.0f}% "
                  f"noise, and noise wipes out the cliff.")
    if len(comp) >= 3:
        Bc = [np.linalg.qr(bases[n])[0] for n in comp]
        sc = union_spectrum(Bc)
        tot_c = len(comp) * args.k
        mc, gc = cliff(sc, min(tot_c - 1, 2 * args.k + 3))
        ec = sc ** 2 / (sc ** 2).sum()
        effc = float(np.exp(-np.sum(ec * np.log(ec + 1e-12))))
        print(f"\n   union of {len(comp)} bases (up to {tot_c} dims):")
        print(f"     {np.array2string(sc[:min(tot_c, 14)], precision=3)}")
        print(f"   cliff at dim {mc}  (gap {gc:.2f}x)   "
              f"effective dim {effc:.2f} of {tot_c}")
        shared_c = (gc > 1.5 and mc <= 2 * args.k)
        flat_c = (gc < 1.2 and effc > 0.8 * tot_c)
        if shared_c:
            print("\n   => CLIFF among the compressing optimizers.  They behave")
            print("      like charts of ONE low-dimensional object (World C),")
            print("      and SGD's failure to compress is a SEPARATE fact about")
            print("      SGD, not evidence against the manifold.")
        elif flat_c:
            print("\n   => STILL FLAT even among the compressing optimizers.")
            print("      Removing the noisy basis did not reveal a cliff.  The")
            print("      subspaces are genuinely unrelated: each optimizer")
            print("      compresses into its OWN private subspace (World D/B).")
        else:
            print("\n   => still undecided even after excluding the noisy basis.")
        union_comp = {
            "members": comp, "spectrum": sc.tolist(), "cliff_dim": mc,
            "gap": gc, "eff_dim": effc, "max_dim": tot_c,
            "shared": bool(shared_c)}
    else:
        print("\n   fewer than 3 compressing optimizers -- cannot test.")

    # ---- the preconditioner hypothesis, stated from the angle matrix ----
    print("\n" + "=" * 78)
    print("  WHAT THE ANGLE MATRIX SUGGESTS")
    print("=" * 78)
    ADAPTIVE = {"AdamW", "RMSprop", "Adagrad"}
    aa = [ang[f"{a}|{b}"] for a in nm for b in nm
          if a != b and a in ADAPTIVE and b in ADAPTIVE]
    ax_ = [ang[f"{a}|{b}"] for a in nm for b in nm
           if a != b and (a in ADAPTIVE) != (b in ADAPTIVE)]
    if aa and ax_:
        print(f"   adaptive <-> adaptive   : mean {np.mean(aa):.0f} deg")
        print(f"   adaptive <-> non-adaptive: mean {np.mean(ax_):.0f} deg")
        if np.mean(aa) < np.mean(ax_) - 10:
            print("\n   The ADAPTIVE optimizers cluster (they find more similar")
            print("   subspaces to each other than to raw SGD).")
            print()
            print("   !! THIS DOES NOT IDENTIFY THE CAUSE.  An earlier version")
            print("   of this script concluded 'the subspace is a property of")
            print("   the PRECONDITIONER'.  That inference is INVALID and has")
            print("   been removed.  At least three hypotheses predict this")
            print("   clustering equally well:")
            print("     A the optimizer CREATES the low-dimensionality")
            print("       (adaptive preconditioners resemble each other)")
            print("     B the optimizer REVEALS an intrinsic valley")
            print("       (adaptive methods ALIGN with it similarly, while raw")
            print("        SGD wanders and corrects orthogonal errors)")
            print("     C SGD lies near the SAME valley but ZIG-ZAGS, so")
            print("       trajectory PCA underestimates the manifold for it")
            print("   The angle matrix cannot separate these.  Neither can any")
            print("   purely trajectory-based measurement.")
            print()
            print("   THE TEST THAT CAN: separate the PATH from the ENDPOINT.")
            print("   Hypothesis C says these come apart -- SGD's path is a")
            print("   zig-zag no 3 vectors capture, yet its ENDPOINT may sit in")
            print("   the same valley.  Ask what fraction of SGD's total")
            print("   DISPLACEMENT lies in the shared basis built from the")
            print("   ADAPTIVE optimizers, against a RANDOM-SUBSPACE NULL:")
            print("     high -> SGD reaches the same structure by a noisier")
            print("             road; the subspace is INTRINSIC (B/C)")
            print("     low  -> SGD goes elsewhere; the optimizer creates it (A)")
            print("   Run:  python shared_residual.py")

    print("\n   REMINDER OF THE LOGIC:")
    print(f"     compression universal? {'YES' if universal else 'NO'}")
    print(f"     subspaces differ?      YES ({np.mean(off):.0f} deg)")
    print(f"     common object?         "
          f"{'YES' if shared else ('NO' if (flatish and eff > 0.8*total) else 'UNDECIDED')}")
    print("     All three must hold for World C.  Compression alone is not")
    print("     enough -- five unrelated compressible optimizers are World D.")

    json.dump({"k": args.k, "v0": v0,
               "union_compressing": union_comp,
               "per_optimizer": {n: {"floor": r["floor"],
                                     "descent": r["descent"],
                                     "recovery": {str(a): b for a, b
                                                  in r["recovery"].items()}}
                                 for n, r in results.items()
                                 if isinstance(r, dict) and "floor" in r},
               "universal_compression": bool(universal),
               "pairwise_angles": ang,
               "mean_angle": float(np.mean(off)),
               "union_spectrum": s.tolist(),
               "cliff_dim": m_, "cliff_gap": gap,
               "effective_dim": eff, "max_dim": total,
               "shared_object": bool(shared),
               "verdict": ("WORLD_C" if shared else
                           ("WORLD_D" if (flatish and eff > 0.8 * total)
                            else "UNDECIDED"))},
              open("universal_subspace.json", "w"), indent=2, default=float)
    print("\n  wrote universal_subspace.json")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(18, 5))
        ks = [1, 2, 3, 5]
        for n in nm:
            ax[0].plot(ks, [100 * results[n]["recovery"][a] for a in ks],
                       "o-", label=n)
        ax[0].axhline(90, ls="--", color="k", alpha=0.5)
        ax[0].set_xlabel("rank k"); ax[0].set_ylabel("% of descent recovered")
        ax[0].set_title("(1) Does every optimizer compress?\n(causal transplant)")
        ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

        Mx = np.zeros((n_opt, n_opt))
        for i, a in enumerate(nm):
            for j, b in enumerate(nm):
                Mx[i, j] = 0 if i == j else ang[f"{a}|{b}"]
        im = ax[1].imshow(Mx, cmap="viridis")
        ax[1].set_xticks(range(n_opt)); ax[1].set_xticklabels(nm, rotation=45)
        ax[1].set_yticks(range(n_opt)); ax[1].set_yticklabels(nm)
        plt.colorbar(im, ax=ax[1], label="mean principal angle (deg)")
        ax[1].set_title("(2) Are the subspaces different?")

        ax[2].plot(range(1, len(s) + 1), s, "o-", color="#c44e52")
        ax[2].axvline(m_, ls="--", color="k", label=f"cliff at {m_}")
        ax[2].axvline(total, ls=":", color="gray",
                      label=f"n*k = {total} (if unrelated)")
        ax[2].set_xlabel("index"); ax[2].set_ylabel("singular value")
        ax[2].set_title(f"(3) UNION SPECTRUM (the discriminator)\n"
                        f"gap={gap:.2f}x  eff dim={eff:.1f}")
        ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3)
        plt.suptitle("Is low-dimensionality universal, and do the subspaces "
                     "share one object?", fontsize=13, weight="bold")
        plt.tight_layout()
        plt.savefig("universal_subspace.png", dpi=180)
        print("  wrote universal_subspace.png")
    except Exception as e:
        print(f"  [plot skipped: {e}]")


if __name__ == "__main__":
    main()
