# Pre-registration (DRAFT, not frozen): Study 17, learned sections in synthetic worlds

Companion to "Conditioned states and Kähler geometry", sections 12.13-12.19. Status: DRAFT, 2026-11-06. It becomes binding when the
author freezes it: the SHA-256 digests of this file and of `kit_learned_sections.py` are then written to `FROZEN_PREREG_study17.txt`,
before any model is trained.

## Disclosure
- There are no data. Every world is synthetic and specified below, and every law in it is exact (rational) and fixed by seed 20261107.
- No model has been trained on any study world. Before freezing, two things have been seen:
  - the timing of a dummy run on random tokens (about 16 ms per step on the 2-CPU workspace);
  - the kit's smoke test (`kit17_smoke_output.txt`), which uses non-neural responders on two study worlds plus a 60-step training run on
    a dummy chain world that is not in the study.
  The smoke test found and fixed a numerical issue: projection onto a boundary fibre needs the forced-zero cells removed first.
- Everything runs in the cloud workspace on CPU, with nothing downloaded and no real or patient-level data. The fictional clinical
  layer is a later study.

## Purpose
In Study 15 a long run on an opaque model tested a theory whose link to the model was an assumption. Here the true joint law is known,
so every quantity in the note is measured exactly. Small transformers are trained only on *context marginals*, the shadows π(P), and
never see the joint. What they say about unobserved combinations is then a choice of section. Some predictions below are consequences
of theorems: if one fails, either the computation or the theory is wrong. The others are empirical questions about what learning does.

## Worlds and regimes
The principle is to keep the data realistic for the empirical questions, and exact only where a theorem is checked.

**G1, latent-class worlds (empirical).** 8 worlds on the 4-cycle. In each, a hidden cause H has 3 classes, with weights in (1/20)ℤ, and
A0, A1, B0, B1 are independent given H, each Bernoulli with parameter in {1, ..., 19}/20. The resulting true joint is a realistic
full-support law, generated rather than chosen. The contexts are the four cycle pairs.

**G2, cohort worlds (empirical, realistic shadows).** 8 worlds. Each of the four cycle pairs is recorded in its own G1-type world, as
when separate studies sample different populations. The resulting tables look like real data but need not fit any one joint law: they
can drift, and they can be contextual. No true joint exists.

**C, constructed theorem paths (exact).** These are used only for the theorem checks and for H3:
- P_int: full support, in (1/240)ℤ.
- P_CH: generic weights on the 8 deterministic points of the first Clauser-Horne facet. It is the unique joint law over its data
  (Cor. 73).
- P_0: generic weights on the 12 readings with (A0,B0) ≠ (0,0). Its fibre has dimension 4 (Prop. 72).
- Paths: P_ε = (1−ε) P_CH + ε P_int and (1−ε) P_0 + ε P_int, for ε ∈ {1/5, 1/20, 1/100, 0}.
- The symmetric CHSH path at λ ∈ {3/5, 4/5, 1}: correlations λ, λ, λ, −λ with uniform marginals, up to the PR box at λ = 1.

**W2, chains with a hidden confounder (composition).** 6 pairs of chains X − Y − Z, all binary. A hidden binary U drives X and Y, and Z
depends on Y and U, so the true 3-way law is not Markov. There are 6 pairs of chains, each with its product world: variables (X,X′),
(Y,Y′), (Z,Z′) with 4 values each, and true law the product. The training contexts are (X,Y) and (Y,Z).

**Data.**
- G1 and G2 use a fixed sample of N = 300 or N = 3000 records per context, drawn once with the condition's seed. This gives realistic
  sampling noise, and the tables are slightly inconsistent across contexts.
- C and W2 use exact marginals, sampled online, so that the theorems apply exactly.

**Conditions** (154 models):

| family | layout | models |
|---|---|---|
| G1 | 8 worlds × 2 N × 3 seeds | 48 |
| G2 | 8 worlds × 2 N × 3 seeds | 48 |
| C | (8 path points + 3 CHSH points) × 2 seeds | 22 |
| W2 | 6 pairs × (chain 1, chain 2, product) × 2 seeds | 36 |

The full list comes from `kit_learned_sections.py plan`, and each seed is 20261107 + 100 × condition index.

## Training (identical for every model)
- **Sequences.** `BOS var val var val EOS` for one context, which is drawn uniformly. The values are drawn from that context's data, and
  the order of the two variables is random.
- **Model.** Decoder-only transformer: 2 layers, 4 heads, d_model 64, feed-forward 128, no dropout, causal mask, no positional
  embedding (NoPE).
- **Optimiser.** AdamW, learning rate 3e-3 with cosine decay, weight decay 0.01, batch 256, 4000 steps, cross-entropy on all next
  tokens. One CPU thread per model.

## The model's implied joint law
For a variable order o, the joint is Q_o(v) = Π_t q(v_t | prefix), with the variable tokens forced. Each factor is the model's
distribution over value tokens only, renormalised. This is exact, needs no sampling, and has full support. A 4-variable query in a 4-cycle world is
out of distribution: the model has only seen pairs, so it conditions on shadows. The reported Q is the average of Q_o over all orders
(24 for the 4-cycle worlds, 6 for W2), and the individual Q_o are kept for the drift measure.

## Metrics
- **Fibre and projection.** F is the fibre of the *true* context marginals; for G2 and the CHSH path no fibre exists. P* is the
  I-projection of Q onto F. It is computed by iterative proportional fitting started at Q, restricted to the cells that F does not force
  to zero (found by linear programming).
- **Decomposition (Csiszár's Pythagorean identity).** KL(P_true ‖ Q) = KL(P_true ‖ P*) + KL(P* ‖ Q).
  - The *in-fibre* term is the coherent but unsupported part of the hallucination.
  - The *off-fibre* term is the inconsistency with the training data.
- **Data with no joint law (G2, the CHSH path).** Mismatch M(Q) = Σ_c KL(d_c ‖ Q_c), and its bound M* = inf over all joint laws R of Σ_c KL(d_c ‖ R_c), computed
  by EM from 20 starts.
- **Section index.** ι = TV(P*, s_H) / (TV(P*, s_H) + TV(P*, s_lex)), where s_H and s_lex are the maximum-entropy and leximin sections
  of the true data. ι is 0 for a maximum-entropy-like choice and 1 for a leximin-like one. It is computed for G1.
- **Unobserved quantity (4-cycle worlds).** q = P(A0 = A1). Report its identified interval over F and the three errors |q(Q) − q(P_true)|,
  |q(Q) − q(P*)| and |q(P*) − q(P_true)|.
- **Order drift.** Mean pairwise TV between the Q_o.
- **Composition (W2).** The learned composite is the (X,Z) marginal of Q. The *tensor-square defect* is the TV between the product
  model's learned composite and the product of the two chain models' learned composites, with the same replicate index. For scale,
  facet-leximin selection has defect 0.0752 on the note's chain pair (MS11), and maximum entropy has 0 on any pair.

## Predictions that follow from theorems (each must hold for every model; a failure means a bug or a wrong theorem)
- **T1.** The Pythagorean identity holds to relative error 1e-6 for every G1 and C-path model.
- **T2.** On the CH facet (ε = 0), TV(P*, P_true) < 1e-6, because the fibre is a point (Cor. 73, Theorem 76).
- **T3.** For data with no joint law (G2, the CHSH path), M(Q) ≥ M* − 1e-6 for every model.
- **T4.** |q(P*) − q(P_true)| is at most the width of the identified interval, plus 1e-9.

## Empirical hypotheses
- **H1 (which section learning picks).** Over the 48 G1 models:
  - "maximum-entropy-like" if ι < 0.5 in at least 75% of them;
  - "leximin-like" if ι > 0.5 in at least 75%;
  - otherwise "mixed".
- **H2 (hallucination from shadows is mostly in-fibre).** In at least 75% of the G1 models with N = 3000, the in-fibre term exceeds
  the off-fibre term.
- **H3 (rigidity in practice).** On the C paths at ε = 0, the mean TV(P_true, Q) on the CH facet is smaller than at the structural
  zero.
- **H4 (order dependence).** Order drift exceeds 0.01 in at least 75% of the G1 models.
- **H5 (learned composition respects independent systems).** The tensor-square defect is below 0.0188 (a quarter of the leximin value
  on the note's chains) in at least 75% of the 12 W2 cases.
- **H6 (realistic shadows make the model less self-consistent).** The mean order drift in G2 exceeds that in G1. The test is a
  bootstrap over worlds (2000 resamples, seed 20261107), and H6 holds if the 95% CI of the difference lies above 0.

## Exploratory (no inference)
- **E1.** Training trajectories: the in-fibre and off-fibre terms every 500 steps.
- **E2.** Entropy of the next-value distribution at each step of the 4-variable query.
- **E3.** G2 and the CHSH path: the excess M(Q) − M*, order drift as a function of λ, and where q(Q) falls; the effect of N.
- **E4.** W2: the learned composite against the Markov composite, the selected composite and the true composite.
- **E5.** The ε-dependence of both terms along the C paths, set against fibre width.

## Reporting
- All conditions, seeds and hypotheses are reported whatever the result.
- Deviations go into `DEVIATIONS_study17.md` before results are shown.
- The analysis runs once, after all 154 models are trained, and writes `study17_final.json` and a figure.
- A failure of T1-T4 is reported first and investigated as a possible error in the kit before anything else is interpreted.

## References
- I. Csiszár, I-divergence geometry of probability distributions and minimization problems, Ann. Probab. 3 (1975) 146-158.
- The note's sections 12.13-12.19, Propositions 71-72, Corollary 73, Theorem 76 (numbering of the 2026-11-05 build), and checks NC4 and MS11.
