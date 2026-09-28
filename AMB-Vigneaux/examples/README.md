# AMB–Vigneaux engine

A closed feedback loop across the three-layer stratification:

| Layer | Math | Module |
|---|---|---|
| 1 · Information | Parameter manifold θ ∈ M, Fisher–Rao metric g(θ), natural gradient | `geometry.py` |
| 2 · Probability | Laws P_θ on context simplices; Vigneaux / Baudot–Bennequin conditioning action `(X·F)(P) = Σ_x P(X=x) F(P\|X=x)`, coboundaries δ and δ_t, `H` is a 1-cocycle, `I = δ_t H` | `information.py` |
| 3 · Outcomes | Support presheaf of sections; Abramsky–Mansfield–Barbosa Čech obstruction γ(s) ∈ Ȟ¹ with Z coefficients; logical and strong contextuality; contextual fraction and its dual Bell inequality | `outcome.py` |
| Loop | Phase 1 is the forward drive (Δθ → ΔP → sampling → Δsupp → γ). Phase 2 is the backward pullback (P̂_N → Δθ = −η g⁺∇D_KL(P̂_N‖P_θ)). | `engine.py` |

The loop keeps the sites separate, as the design note requires: Z-cochains are only computed *from* supports and are never mapped back into the probability layer. The return path is purely statistical, through the empirical measure and information geometry.

## Quick start

```python
from amb_vigneaux import ClosedLoopEngine, ContextualFamily, analyse_outcomes
from amb_vigneaux.models import chsh_scenario, pr_box, hardy_model

print(analyse_outcomes(hardy_model()).summary())

fam = ContextualFamily(chsh_scenario())
eng = ClosedLoopEngine(fam, fam.init_theta(), eta=0.5, n_samples=400, world=pr_box())
for rec in eng.run(30):
    print(rec.t, rec.kl_world, rec.cf_model, rec.gamma_latent, rec.gamma_realised)
```

`world` can be:
- `None`: the loop samples from its own P_θ (self-realization).
- An `EmpiricalModel`: a fixed plant.
- A callable `(t, P_θ) -> EmpiricalModel`: a driven or reactive plant.

`drive(t, θ)` adds an explicit information shift Δθ at the start of each iteration.

Each `StepRecord` includes these fields:
- `kl_empirical`, `kl_world`: KL divergence of the model from the empirical measure and from the world.
- `step_length`: Fisher–Rao distance moved by the update.
- `cf_model`, `cf_empirical`: contextual fraction of P_θ and of P̂_N.
- Latent/realised contextuality level and γ flags.
- `mean_mutual_info`: mean mutual information.
- `max_cocycle_defect`: numerical certificate of the chain rule.
- `info_volume`: log pseudo-determinant of g.
- `signalling`: signalling defect.

## Families (layer 1)

- **`ContextualFamily`**: an independent softmax per context. It can represent any full-support model, whether contextual or not.
- **`HiddenVariableFamily`**: an exponential family on global assignments, pushed forward to the contexts. Its image is exactly the non-contextual models. When the target is contextual, this family has a **KL floor**: the residual KL it cannot remove is the information-geometric shadow of CF > 0 (demo panel B).

## Verified results (`pytest`: 29 tests pass)

| Model | Level | CF | Z-cohomology |
|---|---|---|---|
| PR box | strong | 1 | γ ≠ 0 at all 8 sections |
| GHZ (X/Y) | strong | 1 | γ ≠ 0 at all 48 sections |
| Hardy (ψ=(\|01⟩+\|10⟩+\|11⟩)/√3) | logical | 1/6 | **false negative**; see below |
| Tsirelson box | probabilistic | √2−1 | — |
| noisy Tsirelson, visibility v | — | max(0, √2·v−1) | — |
| noisy PR box, visibility v | — | max(0, 2v−1) | — |

The tests also cover:
- Entropy is a cocycle, and the trivial coboundary of H is mutual information (δ_t H = I).
- The conditioning action is a monoid action: `(XY)·F = X·(Y·F)`.
- δ² = 0.
- Rényi-2 entropy is *not* a cocycle.
- Co-information of XOR is −1 bit.
- Analytic gradients and Fisher metrics match finite differences.
- The exact integer solver agrees with brute force.

**On Hardy:** the Z-coefficient obstruction vanishes for the Hardy section, even though that section is logically contextual. The engine returns the witnessing integer family, which is verified in the tests. It uses a −1 coefficient:
`r_(a1,b0) = −(0,0) + (0,1) + (1,0)`.
This is a genuine limitation of Z-Čech cohomology, which is known to be sound but not complete. It is not a bug. `OutcomeReport.false_negatives` lists such sections explicitly.

## Demo

`python examples/demo.py` writes `examples/demo.png`:
- **A.** While learning the PR box, the latent obstruction γ on supp_ε(P_θ) switches on at iteration 11, once the learned law's forbidden entries fall below ε.
- **B.** A driven quantum world ramps visibility from 0.5 to 1. The contextual learner's CF tracks √2·v−1. The hidden-variable learner stays at CF = 0 and starts to accumulate KL once v passes 1/√2.
- **C.** A noisy PR box ramps from 0.97 to 1. The *realised* obstruction starts flickering on around v ≈ 0.992, once forbidden sections stop appearing in N = 300 samples. The latent obstruction follows later, because the learner averages across iterations.

## Checked against HANDOFF.md

- **T0 (single source).** A `GlobalLaw` world is sampled once, and every context reads the same draws. On such a world the realised model has γ = 0 and CF = 0 exactly, even with sparse support (tests cover 3 to 200 samples). An `EmpiricalModel` world is sampled context by context: independent experiments, one per setting, which is the Bell-experiment reading. The PR, Tsirelson and GHZ demos rely on this mode, because those worlds have no global law.
  - The first version of the engine broke T0: it resampled every world per context, which gave spurious CF(P̂_N) up to 0.10 on single-source data.
  - After the fix, CF(P̂_N) is at most 2e-16 on single-source data. The fitted model's residual CF equals its signalling defect, so it is misfit, as the T0 corollary predicts.
- **Coefficient rings.** `cohomological_obstruction(..., ring=)` accepts `"Z"`, `"Q"` or a prime p. On the 3×3 mod-3 grid it reproduces the handoff's numbers exactly:
  - γ ≠ 0 at 9/9 sections over Z and Z/3, and at 0/9 over Q and Z/2.
  - Both controls are at 0. CF is 1, 0 and 0.
  - Smoothing at ε = 10⁻⁹ takes γ to 0/27 while CF = 1 − 3ε.
- **Hardy false negative.** This agrees with `glue.py`: infeasible, with a vanishing class.
- **Degree bookkeeping** follows the handoff:
  - δH = 0.
  - I₂ = δ_t H, not δH.
  - γ lies in H¹, not H².

## Problem 26: holonomy from estimated operations (`holonomy.py`)

Setup:
- A graph Γ and an abelian group A: R, R/Z, or Z/n.
- Unknown operations g, observed through a *design*: integer functionals a_k with observations y = a_k·g + noise.
- An edge indicator in the design means measuring that surface. A signed path means sending light along that route.

**Identifiability is exact.** The holonomy of loop L can be determined from the design iff ℓ_L lies in the span of the a_k, taken over the ring that acts on A:

| A | Ring |
|---|---|
| R | Q |
| R/Z | Z (a reading known only mod 1 cannot be halved) |
| Z/n | Z/n |

Why the condition is necessary: A is an injective module over that ring in every case. So when ℓ_L is outside the span, there is a shift g′ that kills every observation but moves h(L). The tests build such world pairs explicitly. On the room graph:

| Design | R | R/Z | Z/3 | Z/4 |
|---|---|---|---|---|
| Edge-local | Y Y | Y Y | Y Y | Y Y |
| Tree section: one route per vertex, **the T0 analogue** | n n | n n | n n | n n |
| Tree section plus a second route to v2 | Y n | Y n | Y n | Y n |
| Each loop observed twice | Y Y | n n | Y Y | n n |

The criterion depends on the ring the same way γ does.

**T0 has two faces here:**
1. *Design side.* The single-route design never sees a cycle, and the fitted field reproduces its data exactly.
2. *Estimator side.* Fitting a field and setting ĝ = δĉ gives ĥ ≡ 0, whatever the data.

Detection needs edge-local or multi-route readings, which play the role of a second source.

**Detection.**
- *The test.* A Wald test uses T = ĥᵀΣ⁻¹ĥ with Σ = C·diag(se²)·Cᵀ. Under [g] = 0, T is χ² with β₁ degrees of freedom. The standard error must be the circular delta-method one, (1−ρ₂)/(2mρ₁²). With the naive σ/√m form, the false-positive rate climbs to 74% at σ = 0.35 turn. The corrected test stays at 4–6% across σ from 0.05 to 0.35, and a bootstrap from the nearest flat system is also calibrated.
- *Power.* Power follows the non-central χ² curve, for 3-edge and 6-edge loops.
- *Readings per edge for 80% power* have a closed form, `n_required`. That number grows in proportion to the loop length |L| and like exp(4π²σ²) in the noise. Measured power at the predicted n is 0.77–0.86 against the 0.80 target.
- *Discrete groups (Z/n, symmetric channel).* The mode estimator gets the class exactly right with probability at least 1 − |E|(|A|−1)(1 − (√p_t − √p_w)²)^m. This gives a readings threshold m\*. The bound is valid but about 10× loose: at m = 80 it gives 0.018 against 0.001 measured.

Run `python examples/holonomy_estimated.py` to produce `examples/holonomy.png`.

## Lossy operations: the Layer 2 conjecture (`channels.py`)

Each edge carries a doubly-stochastic channel K on a finite abelian group A. Colours are uniform at every vertex, and a backward traversal uses Kᵀ, which is the exact Bayesian inverse.

**What is proved:**
- **Per-edge Shannon functionals never see which permutation an edge applies**, including through lossy noise.
- **The loop operator's eigenvalue multiset does not depend on the base point**, because AB and BA share their nonzero spectrum. It lives in a monoid, so it is not a cohomology class. It sees only the *order* of the holonomy: on Z/5, every h ≠ 0 gives the same multiset.
- **Covariant noise (convolution by μ, then translation by g):**
  - The eigenvalue labelled by a character χ is Π_e μ̂_e(χ)·χ(h). The phase is χ(h) exactly when μ̂_e(χ) is real, i.e. for symmetric noise. Asymmetric noise adds its own phase: 2.285 in place of 2 in the test.
  - For μ = (1−ε)δ + ε·uniform, the modulus is (1−ε)^{|L|} and h is recovered exactly for every ε < 1.
  - Loop-composite information does not depend on h.
  - γ = 0 for every ε > 0.
- **Contextual fraction.** On a single cycle, CF = max(0, 1 − |L|ε/2), for any finite abelian A and any h ≠ 0. The proof pairs a dual certificate with an explicit packing.
- **On any graph, CF ≥ max(0, 1 − ℓ_min·ε/2)**, where ℓ_min is the shortest simple cycle with nonzero holonomy.

**What is measured:**
- **Equality on any graph.** CF = max(0, 1 − ℓ_min·ε/2) holds in 44 configurations: shared edges, shared vertices, disjoint non-flat loops, and ε up to 0.8. CF depends on the holonomy girth, not on the cycle basis: in θ(2,2,5), a flat 4-cycle gives 7, not 4.
- **Non-covariant noise entangles noise and holonomy.** Flat systems already reach CF up to 0.24.

**Robustness to noise:** γ vanishes as soon as ε > 0; CF vanishes at ε = 2/ℓ_min; the labelled eigenvalue survives until ε → 1.

Run `python examples/channels.py` to produce `examples/channels.png`.

## Decorated KL-Čech nerve (`nerve.py`)

**Setup.**
- *Points are tokens.* Each token carries a distribution over its states (Z/3: T, F, U; Z/5: T, F, could-T, could-F, U) and a time.
- *Filtration.* The radius of a simplex is min_c max_i D(p_i‖c). KL is convex in its first argument, so the balls and their intersections are convex and the nerve lemma applies. Pair radii are computed exactly and certified by the duality bound.
- *Edges.* Each edge is oriented by time ("A before B") and labelled by the affine map x ↦ ux + t that best aligns the two tokens, with its margin over the next-best map.
- *Triangles.* Each triangle carries its boundary holonomy. A triangle whose boundary holonomy is not the identity is a non-flat event.
- *Bars.* Each H₁ bar is labelled by the holonomy of its birth cycle, read from its earliest token.

**Results** (noise-free synthetic rings with a planted element):

| Setting | Result |
|---|---|
| Z/3, affine group (includes reflections) | 0/100 non-identity plants recovered. The reflections fix codimension-1 walls, so the quotient is a simply connected chamber and every plant is absorbed into a flat labelling. |
| Z/3, translations only | Both orientations recovered, x+1 and x+2. Reversing time (B before A) inverts the label. |
| Z/5, affine group | Every non-identity element fixes a face of codimension ≥ 2. Rings with every edge margin ≥ 10⁻³: 254/255 correct. Rings with an ambiguous edge: 57/145 wrong. |
| Relabelling each token in its own frame | Conjugacy class preserved, element not. The Z/p phase and its sign are meaningful only with fixed state labels. |
| Barcode readout (Z/2 persistence) | Z/5 translations: 18/20. Z/3 translations: 8/12. |
| Dirichlet noise on tokens (concentration 2000 → 200) | Ring-label error rises from 15% to 48–63%. |

**Barcode readout.** It is confounded by three things:
- the quotient's own shortcut loops, which have real holonomy;
- Z/2 coefficients: with a non-abelian group, a cycle with a non-identity label can still die at flat triangles;
- ambiguity near fixed faces.

**Other limits.**
- Under the chosen Z/5 state order, negation (T↔F, could-T↔could-F) is not an affine map.
- Only the multiplier u is a function of the H₁ class, because Aff(Z/p)^ab = (Z/p)^×. The full element is a property of the loop.

## Two facets: restriction R and extension E (`functors.py`)

- **R (Vigneaux → AMB).** A global law P maps to its context marginals. R is always defined, and its image is exactly the non-contextual models (T0).
- **E (AMB → Vigneaux).** A model e maps to Ext(e) = {P : R(P) = e}. E(e) is nonempty iff CF(e) = 0 (tested on 8 models).
- **Identities.** P ∈ E(R(P)), and R(E(e)) = e.
- **Canonical member.** When E(e) is nonempty, it is the maximum-entropy extension, found by iterative proportional fitting. IPF does not converge on contextual models.
- **Graded E.** The optimal non-contextual parts, of mass 1 − CF, form a face of a polytope:
  - CHSH: the face is one point for every contextual model tested (39/39).
  - Three-party GHZ scenario: never one point (0/10).
  - `canonical_nc_part` returns the face's maximum-entropy point, which is unique.
- **EM on one 5-label stream.** CF = 0, both for the data windows and for the EM fit (T0).
- **Two streams.** Contexts (x0,x1) and (x1,x2) come from stream A, context (x0,x2) from stream B, which shifts x0 by 2. This gives CF = 0.846, against 1 − 3ε/2 = 0.85, and no global extension.

## N/A as context restriction: the gated test (`na_contexts.py`)

**Two kinds of N/A, kept distinct.**
- *Mask N/A* (−1): the variable is not in this record's context.
- *The n/a value* (5): the variable was measured and the answer is "not applicable", for example negation applied to U.

**Pipeline.**
1. GYO reduction of the hypergraph of maximal observed contexts. If it is acyclic, the pipeline stops (Vorob'ev).
2. Possibilistic test on the support, with an explicit threshold `min_frac`: logical witnesses and γ.
3. CF by LP, with the signalling defect beside it.
4. Null: every record keeps its mask, and its values are redrawn from the maximum-likelihood consistent global law. That law comes from Dempster–Laird–Rubin EM on the incomplete records.

**Results (9,000 records).**

| Stream | Verdict |
|---|---|
| Random N/A, complete records present | Acyclic, so the gate stops: the AMB half is empty |
| Honest, one N/A per record | Cyclic; CF 0.029 against null mean 0.027 (p = 0.39): stop |
| Structured, shift twist | CF 0.858, above every null replicate (null max 0.044) |
| Structured, negation with U → n/a | CF 0.859 with signalling 0.18 |

- **Shift twist, possibilistic layer.** At `min_frac` = 0.02, γ ≠ 0 at 5/5 sections, and 0 of 30 null replicates show logical contextuality. On the raw support, γ = 0 (fragility).
- **Negation twist.** The n/a value leaks into one context's marginal and no other's, so much of this CF is marginal inconsistency, not gluing.

### Follow-up: deficit, ridge, thresholds

- **The deficit is not a function of CF.** The deficit is the per-record log-likelihood lost by the best consistent law. On noisy-shift cycles at CF = 0.5 it is 0.101, 0.068 and 0.051 for L = 3, 4, 5. At ε = 0 it equals log(L/(L−1)). It vanishes exactly where CF does, so CF and the deficit are two different distances to the same non-contextual set.
- **The deficit floor behaves as a sampling floor should.** On honest data it falls roughly as 1/N: 0.0079, 0.0010, 0.00015 for N = 1k, 9k, 81k. On contextual data it stays flat at about 0.282. Bootstrap 95% intervals: shift [0.270, 0.287], negation [0.291, 0.309].
- **Ridge.** When contexts are pairs, the three-way interaction is unidentified, and EM ends at a point on a flat ridge that depends on the start. `pinned_consistent_law` fixes this by taking the maximum-entropy member; it gives the same law from every start (tested). The null uses it.
- **Thresholds.**
  - Thresholded γ produces false positives near a cell's expected mass. On honest data, the null fires in 20/20 replicates at 0.005. On the contextual streams' null, it fires in 20/20 at 0.1.
  - Report `threshold_sweep` with its null band. γ is admissible only in clean windows: for example 0.02 and 0.15, where both structured streams give 5/5 against a null of 0/20.

## Caveats

- `ContextualFamily` does not enforce no-signalling, so finite-N noise leaks a small signalling defect (~10⁻²). The LP counts that defect as contextuality, which gives a CF noise floor of about 0.01 in panel B. A no-signalling-constrained family would be the next thing to add.
- Global sections are enumerated by brute force, which scales as ∏|O_x|. That is fine up to roughly 2¹⁶ global assignments.
- The γ ∈ Ȟ¹ test is computed only for sections that fail to extend globally, since any section that does extend has γ = 0.
