# Three-level bias on BASIL: house lean, selective bias, and where it glues

**Data.** BASIL (Fan et al., EMNLP 2019) covers 100 events from 2010–2019. Each event is covered by Fox, NYT and HuffPost
(HPO), with gold annotations by the dataset authors. Events carry our hand labels (`event_labels.py`): a theme
(ELECT, INVEST, POLICY, SOCIAL) and the focal camp (R, D, N). Span targets also carry hand party labels
(`target_party.py`). Both label files are meant to be audited.

**Levels.**
- **Level 1**: the same event in two descriptions.
- **Level 2**: bias b(source, ·), measured in two ways:
  1. the article-level `relative_stance` (left −2 … right +2);
  2. span-level favourability, the log-odds that a directly aimed bias span toward a camp is positive.
- **Level 3**: relations between biases. Bias splits into a house lean u(s), a topic effect v(t) and a loop residual h. Only h, the loop sum
  Φ on the source–topic graph, is topic-selective bias.
  - For stance we use a **paired** design with each event as its own control: d(e) = b(A, e) − b(B, e).
  - For spans, Φ is the log ratio of odds ratios in the source × camp × polarity table. Φ = 0 is the
    no-three-way log-linear model, a **toric model**. The degree-4 Markov-basis move of the 2×2×2 table generates
    the exact conditional test. Spans cluster within articles, so we also report an **event-level cluster bootstrap**.

## Level 2: house lean (stance, paired, n = 100 events)
| pair | house lean (mean d) |
|---|---|
| Fox − NYT | **+0.81** |
| Fox − HPO | **+0.87** |
| HPO − NYT | −0.06 |

Fox's relative lean grows over time:
- Fox − NYT: +0.40 (2010–13), +1.00 (2014–16), +1.17 (2017–19);
- Fox − HPO: +0.65, +0.70, +1.33.

The trend is about 0.09 per year in both pairs, but it is not significant on its own (permutation p = 0.11 and 0.13).

## Level 3a: stance is a coboundary (no detectable selectivity)
- Split by focal camp, Fox's lean over NYT is the same on Democratic and Republican stories: +0.97 vs +0.95, loop Φ = +0.01,
  p = 0.97.
- No loop over themes, camps or windows survives correction: 36 loop tests, smallest BH-adjusted p = 0.45.
- The gluing map finds that one house lean fits every theme across windows, every window across themes, and the whole cover.
- Power is limited. The standard deviation of d is about 1.7, so the minimum detectable Φ is about 1.1–1.6 stance points, which is large.
- **Reading:** at stance resolution, bias here is *house lean (+ drift)*. That is exactly the coboundary case of
  your example, with Φ = 1 − 0 − 0 + (−1) = 0.

## Level 3b: span favourability is selective (a non-trivial loop)
| pair | loop Φ | exact no-3-way p | event-bootstrap 95% CI |
|---|---|---|---|
| Fox–NYT | +0.63 | 0.16 | [−0.32, +1.60] |
| **Fox–HPO** | **+1.34** | **0.0017** | **[+0.48, +2.35]** |
| HPO–NYT | −0.71 | 0.066 | [−1.56, +0.08] |

- **Fox–HPO** survives Bonferroni over all 24 span tests (0.04).
- **Trump era.** HPO–NYT in 2017–19 gives p = 0.00025 (Bonferroni 0.006). HPO had **0 of 94** Republican-directed spans
  positive, against 6 of 17 Democratic-directed spans.
- **Excluding Trump as a target** (a robustness check, `--exclude-trump`), the Fox–HPO loop persists: Φ = +1.09,
  p = 0.015, CI [+0.10, +2.18]. The two outlets are then selective in **opposite directions**:
  - Fox: R −1.46 vs D −2.11 (more favourable to Republicans);
  - HPO: D −0.99 vs R −1.44 (more favourable to Democrats).

  Most of the 2017–19 HPO effect is Trump-specific. Without Trump, Fox–HPO 2017–19 has p = 0.25, while HPO–NYT still has p = 0.01.
- **Reading:** in favourability, bias is **not** a coboundary. The loop Fox–R–HPO–D carries a nonzero class. This is
  your "Source A: topic A +1; Source B: topic B −1" picture with the signs arranged so that the loop does *not*
  cancel.

## Checks
- **The toric exact test matches the Markov-basis chain.** The hypergeometric enumeration of the fiber agrees with MCMC using the degree-4 Markov-basis moves to within 0.009 (Monte-Carlo error).
- **Data join.** Annotations are joined to articles by filename: four HPO annotation files carry a duplicated uuid. Events are keyed
  by triplet, because main-event strings differ between the articles of a triplet.

## Caveats
- **Hand labels.** Themes, camps and target parties are ours; changing them changes the topic cells.
- **Stance is relative.** `relative_stance` is annotated relative to the other articles on the same event, which is suited to the paired design but compresses absolute differences.
- **Small data.** There are 100 events, and the per-cell counts are small. Spans are not independent; we use the cluster bootstrap for that.
- **Selectivity is not intent.** It is the measurable pattern (non-zero Φ). Calling it intent is an interpretation.

## Next
1. **Prediction.** Forecast the bias of held-out later articles (by source × camp × time) from lean + topic + loop + trend, against
   baselines.
2. **An automatic bias measure.** This is needed to go beyond BASIL's 100 events, for example target-directed sentiment on a larger two-source corpus.
3. **A gluing map at span resolution** over camp × window.

Reproduce: `python bias3.py`, `python bias3_spans.py`, `python bias3_spans.py --exclude-trump` (BASIL at `/tmp/claude-0/BASIL`).

## Global gluing failure (local regions glue, the whole does not)  — `global_gluing.py`
If every outlet pair is compared on the *same* events, the paired differences telescope and the loop always closes.
A global failure therefore needs **different regions**, as when outlets cover overlapping but different stories.

**(1) Planted into real BASIL.** The 100 events are split into three disjoint thirds, with Fox–NYT on the first, NYT–HPO on the second
and HPO–Fox on the third. Each region is a single edge, so it glues trivially. The loop Φ is the sum of the three regional mean
differences. We plant a scale shift δ in HPO's stance **in region 3 only**, so HPO is consistent in each region but not on one global scale.

| δ (stance points) | 0 | 0.5 | 1.0 | 1.5 | 2.0 |
|---|---|---|---|---|---|
| detection rate, \|z\| > 1.96 (2,000 random splits) | **0.052** | 0.17 | 0.48 | 0.83 | 0.98 |
| mean Φ | +0.01 | +0.48 | +0.93 | +1.39 | +1.84 |

Unplanted, real BASIL closes (Φ = +0.01 ± 0.50), and the test is calibrated at 5%. The planted failure is recovered as Φ ≈ δ.
It is invisible region by region.

**(2) A ring of K outlets with BASIL's real annotation noise.** The holonomy h is spread evenly, h/K per edge. Every proper arc is a tree
(glues), so only the whole ring can fail. Detection rates:

| K, n per edge | h = 0 | 0.5 | 1 | 2 | 3 |
|---|---|---|---|---|---|
| 3, 33 (BASIL-sized) | 0.05 | 0.18 | 0.55 | 0.98 | 1.00 |
| 6, 33 | 0.04 | 0.11 | 0.31 | 0.81 | 0.98 |
| 6, 100 | 0.06 | 0.25 | 0.70 | 1.00 | 1.00 |

Longer loops spread the same holonomy over more noisy edges, so they need more events per edge.

**A real global failure already in the data.** The span-favourability loop Fox–R–HPO–D (Level 3b) is exactly of this type.
Each camp on its own is one edge pair and glues; only the union of the two camps fails (Φ = +1.34, p = 0.0017).

## MBIC (Spinde et al. 2021) — `mbic.py`
MBIC has 1,700 sentences from 8 outlets on 14 topics, with 17,775 crowd labels (about 10 per sentence). Every label carries the
annotator's self-reported ideology, from −10 to +10. The data come from the BABE repository (`raw_labels_MBIC.xlsx`).

**Theorem (single source is always consistent).** Suppose every comparison is a difference of one global assignment on the same units,
d = δb. Then every loop sum is δδb = 0. *Check:* annotator groups L, C and R are compared on the same 1,600 sentences, and the loop
L−C−R−L closes to 1.1·10⁻¹⁶. The same loop built from disjoint thirds of the sentences gives +0.009; there it is free
to be non-zero. A global failure therefore needs different units per region.

**Test A: outlet × topic.** Outlets report *different* sentences, so a failure is possible here. The additive model is
bias = outlet propensity + topic effect.
- **Level 2, outlet propensity** (the fraction of annotators calling a sentence biased): Federalist 0.77, Alternet 0.74, HuffPost 0.64,
  Breitbart 0.61, MSNBC 0.56, Fox 0.52, USA Today 0.41, Reuters 0.36.
- **Global gluing fails.** The interaction test gives Freedman–Lane p = 0.0005. The failure is not one bad topic: dropping white-nationalism,
  and then sport as well, leaves p = 0.0005 and 0.001.
- **Local picture.** 35% of the 91 two-topic regions fail on their own at p < 0.05, so the failure is spread over many loops. The strongest
  loops all run through **Reuters × white-nationalism**. Reuters is rated 0.27–0.28 on coronavirus, environment and sport, but **0.71** on
  white nationalism. The gap between partisan outlets and the wire service collapses on that topic (for example Breitbart/Reuters ×
  coronavirus/white-nationalism: Φ = +0.47, z = 6.3). A plausible reading is that annotators react to the topic's vocabulary rather
  than the outlet's framing. That reading is a hypothesis to test.

**Test B: annotator ideology × outlet type, the hostile-media loop.** For each sentence, d = bias rate among left-leaning
annotators − bias rate among right-leaning annotators. This is paired on the same sentence. The loop is Φ = d(right outlets) − d(left outlets).
- Left annotators see right outlets as relatively more biased, and right annotators see left outlets that way.
- Φ = +0.056, +0.070 and +0.053 for the three ideology cut-offs. Exact permutation p = 0.011, 0.0003 and 0.030.
- **The largest loop involves centre outlets.** Right-leaning annotators rate Reuters and USA Today as much more biased than left-leaning
  annotators do (0.50 vs 0.34). Loop (centre, left): Φ = +0.12, p = 5·10⁻⁵. Loop (centre, right): Φ = +0.18, p = 5·10⁻⁵.

## Prediction — `predict.py`, `predict_sens.py`
The models are nested logistic (for stance, multinomial logistic) with one-hot features:
- M1: lean;
- M2: + topic;
- M3: + loop (source × topic);
- M4: + trend (source × year).

The score is out-of-sample log loss in nats per prediction, where lower is better. Gains come with a paired bootstrap; a positive gain means the richer model is better.

| task | split | M1 lean | M2 +topic | M3 +loop | M4 +trend |
|---|---|---|---|---|---|
| BASIL stance (210) | rolling origin, 2013–19 | **1.389** | 1.453 | 1.473 | 1.579 |
| BASIL spans (911) | rolling origin, 2013–19 | **0.462** | 0.465 | 0.463 | 0.491 |
| MBIC labels (17,775) | 10-fold CV by sentence | 0.640 (M0 base 0.674) | **0.635** | 0.637 | – |

- **Level 2 predicts.** For MBIC, outlet lean gains +0.034 nats per prediction over the base rate (CI [+0.030, +0.038]), and the topic effect adds +0.004 ([+0.003, +0.006]).
- **Level 3 detects but does not yet predict.** The loop term's gain over the additive model is about zero everywhere:
  - BASIL spans: +0.001;
  - MBIC: −0.001.

  It stays about zero for regularisation C from 0.1 to 10 and under both rolling-origin and random event-level CV. The largest value is for the Fox–HPO pair on spans, +0.004 to +0.006 nats per prediction, and every 95% CI includes 0.
- **Stance.** Topic and trend terms *hurt* when forecasting later years (−0.06 and −0.11 nats). The lean is stable; party-specific and drifting terms overfit a small, non-stationary sample.
- **Why significance without prediction.** The loop is one contrast among four cells. At these sizes it can be significant in a test pooled over all data (p = 0.0017) while being worth only about 0.005 nats per individual prediction, which is below the resolution of these test sets. The label-level bootstrap in MBIC ignores clustering within sentences, so its CIs are, if anything, too narrow.
- **Consequence.** Forecasting "the probability of bias the next time source A covers topic t" is carried by the house lean (level 2). The loop (level 3) is a diagnostic of *where* the lean-only picture fails. On these datasets it is not yet a better forecaster. Showing predictive value needs more data per loop, or loops of larger effect.

## Two targeted forecasts — `trump_predict.py`
**(a) HPO favourability toward Trump.** HPO had 0 positive out of 105 directly aimed Trump spans (Fox 4/82, NYT 11/149).
Each year Y is forecast from the years before it.

| year | observed | HPO×Trump cell (Jeffreys) p / P(0 of n) / log loss | global rate p / P(0 of n) / log loss | HPO rate p / P(0 of n) / log loss |
|---|---|---|---|---|
| 2017 | 0/20 | 0.014 / 0.75 / 0.014 | 0.185 / 0.017 / 0.205 | 0.220 / 0.007 / 0.248 |
| 2018 | 0/20 | 0.009 / 0.83 / 0.009 | 0.181 / 0.018 / 0.200 | 0.206 / 0.010 / 0.231 |
| 2019 | 0/31 | 0.007 / 0.81 / 0.007 | 0.172 / 0.003 / 0.189 | 0.196 / 0.001 / 0.218 |

The source × target term (a loop cell) forecasts and is calibrated here, because the effect is large, stable and backed by 34–74 earlier spans.

**(b) Does a Fox article criticise Trump?** 19 of the 100 Fox articles do.

| model | leave-one-event-out log loss / AUC | rolling 2016–19 (n = 40) log loss / AUC |
|---|---|---|
| base rate | 0.496 / – | 1.088 / 0.44 |
| Trump in event | 0.271 / 0.83 | 0.748 / 0.56 |
| + era | 0.278 / 0.83 | 0.788 / 0.56 |
| + NYT/HPO criticism on the same event | 0.210 / 0.96 | 0.533 / 0.83 |
| + focal party | 0.186 / 0.97 | 0.492 / 0.85 |

- Among Trump events, Fox criticises in 18/28 when NYT does and in 1/5 when NYT does not.
- The gain from the cross-source terms (M1 → M3) is +0.061 nats per article (CI [−0.022, 0.135]) leave-one-out and +0.215 ([−0.061, 0.464]) rolling. That is suggestive at n = 100.
- This is nowcasting (level-1 coupling on the same event), not a forecast of the future.
- The leave-one-out AUC of the base rate is not meaningful: its predictions differ only through the held-out label.

## Tree-pooled loops — `predict_tree.py`
Pooling the loop one level up the outlet tree (outlet type × topic) does not help MBIC prediction (10-fold by sentence):

| model | log loss | gain vs additive |
|---|---|---|
| additive (outlet + topic) | 0.6352 | – |
| + free loop (outlet × topic) | 0.6354 | −0.0001 [−0.0022, +0.0018] |
| + tree-pooled loop (type × topic) | 0.6364 | −0.0011 [−0.0022, −0.0001] |

Outlets of one type do not share topic-specific deviations; the Reuters × white-nationalism effect is Reuters' own.

## Arity-graded holonomy — `arity_graded.py` (module `amb_vigneaux.arity_holonomy`)
The **arity grade** of a unit (event, sentence) relative to a loop of sources is the number of the loop's sources that measured it.
Grade-g holonomy uses only grade-g units on each edge.
- **Theorem (filled cells are flat).** At full grade, every edge uses the same units and the differences telescope (δδ = 0), so the loop closes exactly. Only lower grades can carry holonomy.
- **BASIL, full coverage.** Every event has all three outlets. Grade-3 Φ = 1.1·10⁻¹⁶, and the largest per-event loop is 0.
- **BASIL, thinned coverage.** Each article is kept with probability 0.75, and 1,000 thinnings are run. Grade 3 always closes: max |Φ₃| = 2·10⁻¹⁶. The grade-2 test is calibrated (0.057 at δ = 0).
  We then plant an HPO scale shift δ only on events where NYT is missing. The plant is invisible at grade 3 and appears at grade 2:

| δ | 0 | 0.5 | 1.0 | 1.5 | 2.0 |
|---|---|---|---|---|---|
| grade-2 detection | 0.057 | 0.083 | 0.22 | 0.44 | 0.71 |
| mean Φ₂ | −0.04 | +0.46 | +0.93 | +1.36 | +1.84 |

- **MBIC, annotator groups L/C/R.** 1,600 sentences are grade 3, with Φ₃ = 0 (per sentence ≤ 1.1·10⁻¹⁶). The 100 grade-2 sentences are all L–R or L–C, with none C–R only. The grade-2 loop therefore has an empty edge and is undefined; that is a coverage fact of the data.

## Phase-change test — `phase_change.py` (and `--exclude-trump`)
We scan for a single change point over 2011–2019. The statistic is the maximum of |after − before| / se. The null permutes event years, so each event keeps its articles and spans.
Candidate years 2015 (start of the Trump campaign) and 2017 (Trump presidency) are also tested individually.

**Level 2, paired lean:**

| pair | best year (scan p) | 2015: shift, p | 2017: shift, p |
|---|---|---|---|
| Fox−NYT | 2012 (0.156) | +0.66, 0.065 | +0.51, 0.104 |
| Fox−HPO | 2017 (0.219) | +0.38, 0.255 | **+0.66, 0.032** |
| HPO−NYT | 2011 (0.046) | +0.28, 0.43 | −0.15, 0.62 |

- **Power.** We plant a shift into a time-shuffled Fox−NYT baseline, starting in 2016. It is detected 0.14 / 0.56 / 0.94 of the time for shifts of 0.5 / 1.0 / 1.5, and located at 2016.
- **Reading.** There is no significant lean change after scanning. The 2017 Fox−HPO shift (p = 0.032 on its own) does not survive correction for six candidate tests.

**Level 3, span-favourability loop (holonomy):**

| pair | best year, scan p | 2017 change in Φ (T) | same, excluding Trump |
|---|---|---|---|
| HPO–NYT | **2016, p = 0.010** | −4.18 (T 4.25) | 2016, p = 0.109; −3.39 (T 2.55) |
| Fox–HPO | 2011, p = 0.020 (edge) | +3.12 (T 2.57) | 2011, p = 0.020; +1.42 (T 0.88) |
| Fox–NYT | 2012, p = 0.20 | −1.06 (T 1.21) | 2012, p = 0.38 |

- **HPO–NYT.** The holonomy has a **regime change at 2016**, and it disappears when Trump is excluded (p = 0.109). The phase change is HPO's treatment of Trump, not a general shift.
- **Fox–HPO.** The scan picks 2011, i.e. 2010 against the rest, and this persists without Trump. It is an early-period difference at the edge of the scan, not a Trump effect.

## NewsWCL50 (Hamborg et al. 2019) — `newswcl50.py`
The data are 10 events from 2018 covered by 5 outlets: HuffPost (LL), NYT (L), USA Today (M), Fox (R) and Breitbart (RR). We use 1,636 valenced property codes; Power, Importance, Victim and Other are excluded as ambiguous.
Every event is covered by all five outlets, so loops among outlets close exactly (full arity). Structure can only live on outlet × target loops.
- **Level 2, favourability toward Trump.** Log-odds of a positive code are LL −2.17, L −0.99, M −0.68, R −0.36, RR −0.57. The slope along the left-to-right order is **+0.40 log-odds per step**, with event-bootstrap CI [+0.22, +0.57].
- **Level 3.** There are 100 outlet-pair × target-pair loops. **6 survive BH**, and all of them involve **HuffPost (LL) × Iran**:
  - LL–M × Trump/IRN: Φ = −4.28, p = 0.0003, BH 0.014, CI [−5.24, −1.06];
  - LL–RR × Trump/IRN: Φ = −4.39, p = 0.0004, BH 0.014, CI [−5.69, −0.46];
  - LL–M and LL–RR × Migrant caravan/IRN: BH 0.014 and 0.022.

  HuffPost is the most negative outlet toward Trump and the caravan, but the least negative toward Iran (−0.51 against −1.4 to −3.3 for the others).
  **Caveat:** HuffPost's positive Iran codes come from one event (event 4, "Lawfulness" and "Economy positive", i.e. deal compliance). The event bootstrap CIs still exclude 0 for the Trump/IRN loops, but the effect rests on that event.
- **Global failure of "outlet lean + target effect".** G² = 37.9 on 5 × 5 × 2. The parametric bootstrap gives p = 0.0015, and the within-event outlet permutation (cluster-respecting) gives p = 0.006.

## Gluing lattice — `gluing_lattice_run.py` (module `amb_vigneaux.gluing_lattice`)
The lattice is the set of sub-covers (regions) on which the local data glue. Here that means the additive "lean + context effect" model fits, i.e. every loop inside the region closes.
Exactly, it is downward closed, so it is summarised by its maximal glueable regions and its minimal obstructed ones. Monotonicity violations count the statistical exceptions.
- **NewsWCL50, cover by targets:** 12 of 26 sub-covers glue, with 0 violations. The maximal glueable regions are {Democrats, Migrant caravan, Trump, USA} and {Democrats, Iran}. The minimal obstructed regions are **{Iran, USA}, {Iran, Migrant caravan} and {Iran, Trump}**.
- **NewsWCL50, cover by outlets:** 11 of 26 glue, with 0 violations. The only maximal glueable region is **{NYT, USA Today, Fox, Breitbart}**. The minimal obstructed regions are **HuffPost paired with each other outlet**.
  Together, the two covers place the global failure on one cell, HuffPost × Iran.
- **MBIC, cover by topics:** 63 of 91 topic pairs glue, so 28 pairs are the minimal obstructions. Greedy search finds several different maximal glueable regions of 10 of the 14 topics. The failure is spread, not due to one topic. Here monotonicity fails statistically: some obstructed pairs, such as abortion and white nationalism, lie inside glueable 10-topic regions. A local failure is diluted when pooled with regions that glue, so the minimal obstructions are the informative output.


## Dynamics: phase changes as natural and non-natural transformations
A change b → b' between corpus 1 (before) and corpus 2 (after) is **natural** (per-source and per-topic recalibrations η_s + η_t) iff Δb is a coboundary iff ΔΦ = 0 on every loop. The non-natural part is the loop change ΔΦ. `amb_vigneaux.dynamics.natural_split` computes the split.

**Synthetic three-layer generator** (`examples/dynamics/three_layer_dynamics.py`: outcomes → items with arity → sources with relabellings and stance). Detection rates over 200 runs with 400 events per period:

| planted | outcomes | lean | ΔΦ naturality | monodromy odd k | monodromy even k | coverage grade 2 |
|---|---|---|---|---|---|---|
| none | 0.06 | 0.07 | 0.06 | 0.00 | 0.00 | 0.04 |
| A outcome inversion | **1.00** | 0.05 | 0.05 | 0.00 | 0.00 | 0.04 |
| B natural lean shift | 0.05 | **1.00** | 0.06 | 0.00 | 0.00 | 0.06 |
| C topic-selective shift | 0.09 | 0.77 | **0.90** | 0.00 | 0.00 | 0.06 |
| D relabel odd-arity items | 0.07 | 0.06 | 0.04 | **1.00** | 0.00 | 0.04 |
| E coverage-graded shift | 0.07 | 0.06 | 0.06 | 0.00 | 0.00 | **0.82** |

- At 150 events per period, the C/ΔΦ rate falls to 0.52 and the E/coverage rate to 0.40. The other rates do not change.
- **Projected outcomes at time t:** log loss is measured on the last 50 events.
  - After an inversion, the change-point forecast scores **0.967** and the pooled forecast 1.037.
  - With no change, the change-point forecast scores 0.993 and the pooled forecast 0.974, the small price of scanning.

**BASIL naturality at τ = 2016** (`naturality.py`):
- **Stance:** the lean changes are +0.19, +0.43 and −0.23 for Fox–NYT, Fox–HPO and HPO–NYT (|z| ≤ 1.4). The ΔΦ values have |z| ≤ 0.5. So stance shows no significant change, natural or not.
- **Spans:** Fox–HPO ΔΦ is **+2.90 (z 3.2)** and HPO–NYT is **−4.26 (z −5.7)**. Without Trump they are +1.30 (z 1.2) and −3.35 (z −2.9).
  - The 2016 change in span favourability is **non-natural**: it is topic-selective and not a uniform re-calibration.
  - It is mostly, but not entirely, Trump-driven.
  - These z are nominal, because τ was chosen by the scan.
- **Prolate operators:** these are useful for estimating concentration under a band limit. The claims that a spectral-gap crossing is the phase change, or that ΔΦ is the spectral flow, are not defined without a specified kernel. We did not adopt them.


## Slepian (prolate) tools and a corpus with big events (AllSides, 2015–2020)
**What the prolate tools give.** Slepian sequences are used for estimation only: a step test against a Slepian trend, multitaper band ratios, and Thomson's line F-test (Monte-Carlo calibrated). The 9-year BASIL window resolves only about 2 modes (Shannon number 2NW). The prolate index is not the arity grade.

**Synthetic check** (`examples/dynamics/spectral_synthetic.py`): 73 weeks before and after, 400 runs.

| planted | Welch mean | Slepian step | MT low | MT mid | MT high | line F |
|---|---|---|---|---|---|---|
| none | 0.09 | 0.06 | 0.04 | 0.00 | 0.01 | 0.05 |
| level shift | **0.90** | **0.36** | 0.03 | 0.01 | 0.00 | 0.04 |
| persistence 0.2→0.8 | 0.36 | 0.21 | 0.28 | 0.00 | **0.99** | 0.03 |
| new 8-week cycle | 0.10 | 0.04 | 0.04 | 0.03 | 0.01 | **0.20** |
| smooth trend | 0.91 | 0.07 | 0.04 | 0.01 | 0.01 | 0.03 |

The mean test calls a trend a phase change. The Slepian step test does not, but it has less power against a true step.

**AllSides** (Baly et al. 2020; 16,069 dated Trump-mentioning articles, sides L/C/R). The measures are fixed lexicons, with no training:
- *Adversarial:* fact-check or condemnation vocabulary.
- *Directed:* that vocabulary with Trump just before it.
- *Tone:* VADER. It is not favourability, because right outlets score lower.

The events were pre-specified. Each window is ±26 weeks.

| event | adv. gap L−R | z(Δ) | directed gap | non-natural p | MT high ratio (p) |
|---|---|---|---|---|---|
| election 2016 | +0.100 → +0.054 | −1.7 | +0.020 → +0.009 | 0.72 | 0.34 (0.004) |
| inauguration | +0.085 → +0.047 | −1.5 | +0.017 → +0.016 | 0.66 | 0.60 (0.16) |
| Charlottesville | +0.052 → +0.035 | −0.6 | +0.018 → +0.014 | 0.86 | 1.60 (0.20) |
| Mueller report | +0.098 → +0.063 | −1.3 | +0.026 → +0.027 | **0.013** | 1.67 (0.16) |
| impeachment inquiry | +0.064 → +0.031 | −1.3 | +0.027 → +0.017 | 0.41 | 1.99 (0.09) |
| COVID emergency | +0.040 → +0.027 | −0.5 | +0.018 → +0.031 | 0.39 | 3.02 (0.07) |

- **Stable lean.** Left outlets are more adversarial toward Trump before every event (z 2.2–4.9). By the directed marker the gap is about 4× (article rates 0.021 / 0.012 / 0.005 for L / C / R). No event inverts the outcome probabilities.
- **Mueller report: the only non-natural change** (p = 0.013; 0.08 after Bonferroni over 6 events).
  - It sits in the justice cell: center −0.18, right +0.13.
  - Reading the sentences shows why: after the report, right outlets aim the vocabulary at the investigation ("baseless collusion claims").
  - The directed marker does not move, so this is a change of *target*, not of stance.
- **Election 2016:** week-to-week volatility of the gap drops to 0.34 of its campaign level (p = 0.004; 0.07 after Bonferroni over 18 tests). This is a volatility change, not a level change.
- **No other events:** no event gives a Slepian step (|z| ≤ 1.33), and no new periodic component appears (adjusted p ≥ 0.45).
- **Unrestricted scan:** the largest step is in late March 2018 (|z| 3.6, scan p = 0.050).
- **COVID:** a new topic cell appears, so the cover itself changes. The naturality test on a fixed cover cannot see a new open set.


## Is the topology tearing? Descent, a spectral certificate, outcome toggling
**Descent lemma.** A map of presheaves that is a bijection on every context and overlap is a bijection on glued sections, because glued sections are a limit. Equivalently, every global change shows up on some context or overlap.

**Spectral certificate** (`examples/certificate/run_all.py`).
- Hodge split of the cell means on the source × topic graph: natural part plus loop part h.
- Weyl: if ||ΔL₀|| < λ₂ (Fiedler), the cover stays connected and β₁ and the loops are unchanged.
- Margin = λ₂ / q95 bootstrap ||ΔL₀||. "Connected" = the fraction of bootstrap replicates that stay connected.
- Q ~ χ²(β₁). SNR ≈ 1 under the null.
- Refined sig. = the fraction of 50 random refinements of the topic cover in which the obstruction is still significant.

| data | β₁ | margin | connected | Q, p | SNR | refined sig. |
|---|---|---|---|---|---|---|
| BASIL spans (3×2) | 2 | 1.89 | 1.00 | 13.0, 0.0015 | 2.27 | 1.00 |
| BASIL, Trump excluded | 2 | 1.64 | 1.00 | 6.0, 0.05 | 1.34 | 0.42 |
| MBIC (8×14) | 66 | 1.24 | 1.00 | 131, 3e-6 | 1.36 | 0.98 |
| NewsWCL50 (5×5) | 15 | **0.13** | **0.36** | 29.7, 0.013 | 1.32 | 1.00 |
| NewsWCL50, no Democrats | 12 | **0.22** | **0.57** | 28.5, 0.005 | 1.30 | 1.00 |
| AllSides adversarial (3×9) | 16 | 1.11 | 1.00 | 18.1, 0.32 | 1.18 | 0.02 |
| AllSides directed (3×9) | 16 | 1.11 | 1.00 | 16.6, 0.41 | 1.10 | 0.02 |

- **BASIL and MBIC:** certified and robust.
  - Leave-one-source-out localises BASIL's obstruction at HPO: without HPO p = 0.10, without Fox 0.04, without NYT < 0.001.
  - MBIC's obstruction survives removing any single outlet.
- **BASIL without Trump:** fragile; it survives only 42% of refinements.
- **NewsWCL50: the cover tears.** All Democrats codes come from one event, and with only ten events the other targets are thin too. The obstruction localises at LL (without LL p = 0.72), but its topology is not certified at the event level.
- **AllSides:** glues under both markers and under refinement.

**Toggle directions (proposition).** An outcome o = sign(⟨a, b⟩_W − c) flips under changes in a subspace V at W-distance r_V = |f| / ||P_V a||_W.
- Natural changes cannot flip an outcome that reads only loops, and loop changes cannot flip one that reads only the natural part.
- AMB (support) outcomes flip only when a probability reaches or leaves 0.

**COVID and the centre** (`examples/allsides/toggle_covid.py`). The outcome is f = 2P_C − P_L − P_R, which is negative when the centre is closer to the right. Pooled over topics it reads only the natural part.
- **Directed marker:** f went from −0.019 (se 0.009) to −0.004 (se 0.012). The centre closed about 80% of the gap to the midpoint but did not toggle. 48% of placebo before/after pairs from 2016–2020 moved as much. The toggle radius before COVID was 0.009, or 1.2 placebo sd.
- **Adversarial marker:** −0.056 → −0.039, matched by 80% of placebos.
- **Limits:** these data describe media alignment, not votes, and they end in July 2020. Linking coverage to the election needs an outcome series (approval or polls) and a toggle radius committed to in advance.


## Probing inside arity: hull trees and the p-adic radius profile
**Why a probe is needed.** Inside a filled cell (one unit, k sources) every loop of differences is 0, so holonomy sees nothing. What survives is the configuration of the k values up to translation. Its shadow is the hull tree (dendrogram), and its first merge is the **coalition**. The coalition does not change under unit effects. After removing source effects it does not change under any natural change either. For integers in Z_p, the hull tree is the tree of discs in the Berkovich line. See Proposition 39.1 in the paper; `examples/arity_probe/run.py`.

- **MBIC, annotators inside a sentence** (1,640 sentences with left, centre and right annotators; the null permutes group labels within each sentence, which keeps ties, sizes and rater spread):
  - left–centre agree first **578 vs 461** expected (p < 0.002);
  - left–right 441 vs 484 (p = 0.008);
  - centre–right **396 vs 467** (p < 0.002).
  - The right annotators are the odd ones out, and the ideology leans are only about ±0.01.
  - By outlet type: on sentences from left outlets the right annotators split from the *left* (L–R 157 vs 194, p < 0.002; C–R 180 vs 194, p = 0.20). On sentences from right outlets they split from the *centre* (C–R 150 vs 187, p < 0.002; L–R 191 vs 197, p = 0.59).
  - This is structure inside a 3-ary cell, and it refines the hostile-media loop.
- **AllSides, sides inside a story** (736 story groups with left, centre and right; tone of Trump sentences): the coalitions match the additive null (258/251, 236/240, 242/246) and do not shift at the 2016 election or at COVID (p ≥ 0.20).
- **NewsWCL50:** only 6 fully covered cells, too few to test.
- **BASIL:** stance is rank-coded within each triplet (68/100 ties against about 30 expected), so it is uninformative.
- **p-adic order profile** e(m) = log_p of the order of the obstruction class of sum claims. It is canonical, piecewise linear with integer slopes 0/1, and 0 exactly up to m* (p = 3, K_{2,3}):
  - an error of 3⁴ gives (0,0,0,0,0,1,2), so m* = 4;
  - adding 3² on the other loop gives (0,0,0,1,2,3,4), so m* = 2;
  - an error of 18 behaves like 9.
- **Scope:** only the elementary hull and integer-slope facts from Pulita (arXiv:0802.1945) transfer. The deformation theory (Σ-modules, quasi-unipotence) needs a differential equation.


### Replication of the coalition probe; variogram; majority toggles
- **Variogram (lean-invariant).** γ*_ij = ½Var(x_i − x_j) − ½E[sampling var_i + var_j]. It is invariant under every natural change u_s + v_e, and its tree's first merge is the population coalition.
- **Majority reads the tree.** P(maj = 1) = Σp − Σγ − 2P(all 1). With all marginals at 0.45 (same b, lean and loops), the majority is 0.425 for independent votes, 0.45 when all always agree, and 0.675 when exactly two agree. So a phase change with no mean change can toggle a majority/AMB outcome. It is a second-order, tree-level change that holonomy cannot see.
- **MBIC, 3 groups:** γ*_LC −0.002±0.003, γ*_LR 0.005±0.003, γ*_CR **0.014±0.004**. The first merge is L–C (support 0.97).
  - Centre–right is largest on right-outlet sentences (0.021±0.007).
  - The coalition counts are the same for articles up to 2019 and from 2020.
- **MBIC, 5 groups:** no structure (all |γ*| ≤ 0.011, support 0.37). The effect does not refine into an ideology gradient.
- **BABE experts:** the annotator-level variogram does not follow ideology (Mantel r −0.26, p 0.95 for SG1; −0.02, p 0.50 for SG2). Group comparisons are confounded by individuals (SG1 has a single right annotator).
- **Assessment:** real and stable crowd structure in MBIC, with right isolated mainly against the centre, but not a general law yet.
