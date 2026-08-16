# Compiler output glossary

What each printed symbol is computed from, and what it means. Definitions traced
to `compiler_geometri_patched_86.py`.

---

## Phase 3 step line

```
step  16: val=5.3776  Δ=0.5774  Φ_cl=5/5  τ=3.81  rm2σ=+0.698
```

### `val` — validation loss
Cross-entropy on held-out batches. The only quantity here that is about
performance rather than geometry. The run's anchor is `val_floor`, a reference
value measured before Phase 1.

### `Δ` — loss decrement per step
Mean drop in `val` over the last block of steps. Used by the plateau test: when
Δ stops falling, Phase 3 has nothing left to extract. In the runs above it decays
0.47, 0.58, 0.20, 0.11 — the shape of a settling trajectory.

### `Φ_cl` — clean sheet-angle count, out of 5
The compiler forms the transfer operator between consecutive attention key
matrices,

    φ_l = W_K^(l+1) · pinv(W_K^(l))

takes its dominant eigenvalue λ₁, and records the phase `angle(λ₁)`. A layer
pair counts as **clean** when that phase is within 0.3 radians of exactly `0` or
exactly `π`. `Φ_cl` is how many of the 5 layer pairs are clean.

Meaning: a phase of 0 or π means the two layers' key subspaces are related by a
*real* scaling — aligned or anti-aligned — rather than by a rotation. Clean
layers are in the same orbit; a phase like 1.34 means that pair has twisted out
of it. 5/5 is a fully coherent stack.

Caution: it oscillates. Across one run it reads 3, 5, 5, 3, 4, 3, 4, 4, 5. A
single reading of 5/5 is not a converged state, which is why the geo-stop
requires two consecutive passes.

### `τ` — gluing defect
The ratio of feed-forward gradient norm to embedding gradient norm,

    τ = ‖∇ FF‖ / ‖∇ EMB‖

Meaning: how the descent effort is split between the layers that *transform*
(FF) and the layer that *represents* (EMB). Low τ means the embedding is still
doing the work — the representation has not settled. High τ means the embedding
is fixed and the FF stack is adapting on top of it.

The basin criterion is τ ∈ [5, 7.5]: enough weight on FF that the representation
has stabilised, not so much that the embedding has stopped participating. The
anchor `τ_basin ≈ 2` refers to a different regime measured at the floor.

### `rm2σ` — weighted subspace overlap between key matrices
For each consecutive pair of W_K matrices, take the top-`r` left singular
subspaces U₀, U₁, then the singular values of U₀ᵀU₁ — the cosines of the
principal angles. Each is reweighted by

    h(s) = s / (1 - s²)^{3/2}

and averaged. The weight blows up as s → 1, so the statistic is dominated by
the *most* aligned directions rather than the average.

Meaning: how much of one layer's key subspace survives into the next. High rm2σ
means information is being carried forward coherently. The criterion is ≥ 0.65.

Note it measures *subspace* overlap where Φ_cl measures *phase*. Two layers can
share a subspace while being rotated within it — rm2σ high, Φ_cl low.

---

## Geo-stop line

```
Geometric stopping: Φ_cl≥4 + τ∈[5,7] + rm2σ≥0.65 (×2 checks)
○ GEO-STOP candidate (1/2): Φ=5/5 τ=5.88 rm2σ=0.671
✓ GEO-STOP confirmed at step 56
```

All three criteria must hold on **two consecutive** checks, eight steps apart.
The hypothesis is that orbit geometry converges before the loss plateaus, so
Phase 3 can hand off early.

**Measured caveat.** The criteria say nothing about proximity to the basin. A
better projection satisfied them at step 32 with val 2.99, Phase 3 exited, and
the pipeline finished at 0.512 instead of 0.077. Gating the stop on val as well
(`--geoval 0.6`) fixes it.

---

## Sheet angle list

```
Φ=['1.67', '0', '0', '0', '0.66']
```

The per-pair phases behind `Φ_cl`, in layer order. `0` and `π` are clean; a
number is the phase in radians; `?` means the pinv failed. The list above is
3/5 clean.

---

## Patch report line

```
[3456] dims 56 (asked 56)/4330240  coast:off  backward 136  coasted 0
       skip 0%  capture 0.899  gamma 0.953  fisher-share 0.031
```

### `dims N (asked M)/P`
Actual frame width against requested, over total parameters. **If these
disagree the window is capping the rank** — an n-column update history yields at
most n singular vectors.

### `capture`
Fraction of the update's energy inside the frame,
‖P_Q u‖² / ‖u‖². Measured 0.90 at rank 8. This is about the *update*.

### `gamma` — leakage
Fraction of a fresh **gradient** outside the frame,
1 − ‖P_Q g‖² / ‖g‖², energy-weighted across roles. Measured 0.95.

Capture and gamma are not complements of each other — they measure different
objects. The frame is built from updates, and a fresh gradient sits almost
entirely outside it. Setting the coast threshold from capture (0.20) rather than
gamma (~0.95) is why the trigger never fired.

### `coast` / `backward` / `skip`
Steps taken without a backward pass, steps with one, and the ratio. `coast:off`
means `--no-trigger`; `coast:sign` means skipped steps used `a_t · sign(g)`.

### `fisher-share`
Fraction of the *discarded complement* lying in the Fisher sheet. Measured
**0.031** — and keeping that 3% at full weight instead of α took the final val
from 0.0766 to 0.0540. A small share with a large effect, consistent with three
Fisher directions out of 4.3×10⁶ delivering more loss reduction than the full
update.

---

## Final comparison

```
Compiler GAP vs floor: -0.0080 nats
GD-400   GAP vs floor: +0.0294 nats
```

Distance from `val_floor`, the reference measured before Phase 1. Negative means
the run finished *below* the anchor. `val_floor` is an empirical reference from
a prior model, not a theoretical bound, so beating it is possible and worth
confirming across seeds.
