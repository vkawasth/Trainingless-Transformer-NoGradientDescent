# Pre-analysis plan, phase 6: tracking the approach to collapse with a family of cones, and recursive training

**Fixed on 2026-10-11.** No quantity below has been computed, and no synthetic generation has been run.

What was already known:
- the phase 4 and phase 5 results;
- the GSS regimes (one decided survey tear among the three pairs of the early analysis: welfare × race, wording Y, 2014 → 2016);
- the C19PRC and PhysioNet results.

Seed: 20261011.

## 1. The tracker (`fan_tracker.py`)
A trajectory is a sequence of systems: two-way tables over the same contents, with counts or with laws and effective sizes. Their margins move along a path, and their coupling polytopes form a family over that path (Bands Prop. 6.1). At each step t the tracker reports:
1. **Chamber.** The signs of all wall functions f(I) − g(J). Each is *decided* when its interval on the margin bands excludes 0 (`monitor.wall_signs`). It also reports the signed wall distances and their intervals.
2. **Margins and velocity** (Bands Def. 3.1, Thm. 3.2, Def. 5.7):
   - tear margin m_T, the distance to the union of walls;
   - break margin m_B, the distance to the boundary;
   - velocity v_t = TV(f_t, f_{t−1}) + TV(g_t, g_{t−1});
   - tear time m_T / v_t and break time m_B / v_t, the lower bounds on steps to a wall and to the boundary.
3. **Drift split (Csiszár's Pythagorean identity).** Let p* be the I-projection of p_{t−1} onto the laws with p_t's margins (IPF). Then

       D(p_t ‖ p_{t−1}) = D(p_t ‖ p*) + D(p* ‖ p_{t−1})

   The first term is **interaction drift** (what the information means), the second **margin drift** (who carries which information). Zero cells get ½ added to the counts before the split, and this is reported.
4. **Normal-fan type.**
   - The number of vertices of the coupling polytope (the maximal cones of its normal fan).
   - The number of facets (its rays).
   - The dimension.
   Bend: same fan. Tear: blow-up or blow-down. Break: the dimension drops.
5. **Posterior over chambers.** Draw 200 Dirichlet(counts + ½) joint tables. Report the probability of the modal chamber and the probability that each wall keeps its sign.

**Planted checks:**
- the Pythagorean identity to 1e-10;
- fan types against `geom.n_vertices` (3×3) and the 2×3 surfaces of `tower_icu.toric`;
- a planted path that crosses one wall is reported as one decided flip at the right step;
- a path towards a zero margin gives a break time that shrinks to 0 at the break.

## 2. Trajectories (descriptive)
1. **GSS:** all 110 item-pair systems (11 items, two wordings), 1985–2024, weighted by `wtssps` with Kish effective sizes; the run-up to the welfare × race tear (Y, 2014 → 2016) is shown in detail.
2. **C19PRC:** depression × anxiety (3×3), all respondents at each wave; UK W1–W6, Ireland W1–W5.
3. **Tower evidence paths:** the 2×3 system death × first-window pressure along the family-of-floors tower's level-one path in arms R0–R4 (`phase4/results/wp4_tower.json`), with the pressure margin observed. The signed distance to the wall f₁ = g₀ along the evidence shows hypothesis change as motion through chambers.
4. **ICU decedents against survivors:** MAP band × HR band, pooled over all patient-hours in each 6-hour window, hours 0–48, separately for stays ending in death and in survival (sets A and B pooled). Does the decedents' system approach a wall or the boundary earlier?
5. **LLM retraining:** for each of the eight tasks, March → June, GPT-3.5 × GPT-4 answers. Two points only, so descriptive.

## 3. The pre-registered forecasting test
**Question.** Does interaction drift, or tear time, forecast the next *decided* change beyond plain margin velocity?

**Units.** GSS steps t → t+1 for the 110 systems, the next survey year. A step is usable when step t−1 → t exists.

**Event.** A decided chamber change at t → t+1: some wall with a decided sign at t has the opposite decided sign at t+1.

**Predictors at t (from t−1 → t and t).**
- Baseline B: margin velocity v_t and the break margin m_B.
- B + interaction drift D(p_t ‖ p*).
- B + log tear time log(m_T / v_t).
- B + both.

**Fit and evaluation.** Logistic regression (ridge C = 1, standardized). Train on steps ending ≤ 2004, test on steps ending ≥ 2006. Test log-loss gain over B, with a bootstrap clustered by system (2000 draws).

**Decisions.**
- *Interaction drift forecasts beyond margin velocity* if its gain interval is above 0.
- The same for tear time.
- If the test set has fewer than 15 events, the test is reported as underpowered. The point-estimate chamber change of phase 4 is then reported as secondary, with the same models.

## 4. Recursive training with known truth (`recursive_training.py`)
**Truth.** Three contexts c (prompt templates), each carrying the same information x ∈ {0, …, 5}, with different context laws P(x | c). The outcome y ∈ {0, …, 7} follows P₀(y | x) (Dirichlet(0.4) rows, seed fixed), **the same for every context**: invariance holds in truth. Outcome categories with P₀ ≤ 0.02 in a row are that row's **tails**.

**Model.** One conditional table per (c, x) cell, fitted by maximum likelihood (no smoothing) from n = 200 samples per cell.

**Generations.** Generation 0 is fitted on real data. At generation g, each cell's training sample is:
- a share (1 − λ) of samples from the generation g − 1 model;
- a share λ of fresh samples from the truth;
- λ ∈ {0, 0.05, 0.10, 0.25}.

Optionally each generation passes outputs through a paraphrase kernel K on outcomes: y ↦ y with probability 1 − τ, otherwise y ± 1 (cyclic), each with τ/2.
- Regime S: τ = 0, sampling only.
- Regime K: τ = 0.2, a kernel with Dobrushin coefficient η(K) computed exactly.

Settings: 20 generations, 50 replicates per setting. Held-out real data: 500 per cell, fixed per replicate.

**Predictions.**
1. **Contraction (regime K, λ = 0).**
   - Statistic: the model's distinguishability of the information, D_g = mean over x < x′ and c of TV(Q_g(· | c, x), Q_g(· | c, x′)).
   - Holds if the mean over replicates of D_g stays at or below η^g · D₀ + s_g at every generation. Here s_g is the 95th percentile of D_g in a no-information null: all rows of P₀ replaced by their average, same sampling.
2. **Breaks in finite time, forecast by break time (regime S, λ = 0).**
   - Every tail of P₀ hits 0 within 20 generations in at least 90% of replicates and stays there (absorbing).
   - Across (replicate, cell, tail, generation) before the break, the break time m_B / v (the category's mass over its velocity, averaged over the last two generations) has a positive Spearman correlation with the realized remaining time to the break. It must also beat the level m_B alone, in a partial Spearman controlling for m_B, with p < 0.01.
3. **Fresh data prevents absorbing breaks (Bands Prop. 9.2).**
   - With λ ≥ 0.05, no category of positive true probability is zero for 3 consecutive generations in more than 1% of (replicate, cell, category) tracks.
   - The mean model mass of each category stays at or above λ·P₀ minus 2 standard errors at every generation.
   - At λ = 0 such absorbing zeros occur (as in prediction 2).
4. **Contexts drift off the invariance variety (regime S).**
   - Statistic: the model's own I(Y; C | X), computed exactly from its tables with the true P(x | c), which is the distance to the toric variety Y ⟂ C | X.
   - At λ = 0 it increases with generation: Spearman over generations of the replicate mean > 0, with the generation-20 value above the generation-1 value in ≥ 90% of replicates.
   - With λ = 0.25 the generation-20 value is smaller than with λ = 0, in ≥ 90% of paired replicates.
5. **Certificates warn before accuracy drops (regime S, λ = 0).**
   - Accuracy: the share of (c, x) cells whose model argmax equals the true argmax.
   - *Accuracy visibly drops* at the first generation where it is ≥ 5 points below its generation-0 value.
   - Certificate (Tower Thm. 10.8, Prop. 10.9), per cell:
     - level one: the ℓ¹ distance from the model law to the elementwise 95% band of the held-out real data;
     - plus the Cromwell part: the held-out mass on categories the model gives zero.
   - The *warning* is the first generation where the mean certificate over cells exceeds the 95th percentile of the same statistic for a model refitted on fresh real data (the null, by simulation).
   - **Holds** if the warning comes strictly before the visible accuracy drop in ≥ 80% of replicates where a drop occurs, and in at least half of all replicates.

## Order
1. Planted checks of the tracker.
2. Trajectories.
3. The forecasting test.
4. The recursive-training experiment.

Each is run once and frozen with SHA-256 hashes. Every prediction is reported as held or not.
