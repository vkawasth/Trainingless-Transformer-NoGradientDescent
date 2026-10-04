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


## Moduli: toric strata, tree space, singular strata of grammars
**AMB outcomes are boundary strata of the toric model** (`examples/strata/amb_toggle.py`; exact MILP radii).
- **Result:** making a cell impossible within the model needs its co-facial set F to vanish, at distance −log(1 − p(F)).
- **Markov-move closure:** a co-facial set that meets one side of a Markov move meets the other. So in the glued (no-three-way) model a single-cell zero is never a stratum. A possibilistic outcome cannot toggle in one context alone.
- **MBIC** (176 live cells): glued vs saturated evidence is 2.0× at the median, up to 17×. The cheapest toggle removes the whole context for 163 cells, a source–label slice for 9 and a topic–label slice for 4.
- **AllSides** (48 cells): median 1.6×. The cheapest toggle is a context for 25 cells, a topic slice for 20 and a source slice for 3. For example, "adversarial impossible on healthcare" can only happen for all sides at once.
- **Around COVID:** the centre's "adversarial possible" outcome moves closer to a toggle on justice (3.7% → 1.8% of the window) and further from one on media (1.1% → 3.7%) and politics (2.5% → 4.7%).

**Tree space** (BHV spider T_3, sticky Fréchet mean, energy test; `examples/strata/treespace_run.py`).
- **MBIC:** the mean sticks at the star. The L–C leg moment is 0.118 against 0.082 under the null (p < 0.002) and the C–R moment is 0.067 against 0.080: a tendency, not a population coalition. Left vs right outlets p = 0.066; article years p = 0.23.
- **AllSides:** the mean is sticky. Election p = 0.17; COVID p = 0.058 (the L–C moment went from 0.058 to 0.095 on 85 stories).

**Singular strata of grammars** (`benchmarks/o6/merge_strata.py`): the likelihood distance to each merge stratum, median (min) in nats/seq.
- **V = 8, L = 6:** 5.07 (1.37), 2.28 (0.93), 0.99 (0.43), 0.37 (0.07), 0.13 (0.035), and **0 at the root (all 28 pairs, exactly)**.
- **V = 16, L = 4:** 0.69 (0.17), 0.27 (0.13), 0.14 (0.0016), and **0 at the root (all 40 pairs)**.
- **Proposition (root not identifiable):** the likelihood sees the root only through Σ π_s P_L[s]. The baseline's worst top-level shortfalls are at the root, so the O6 scorer now scores levels 1…L−1.
- At L = 6 the EM gap (0.005) is below every level-5 merge distance (≥ 0.035), so the level-5 shortfall is slow EM. At V = 16, one level-3 pair (0.0016) is below the 0.02 gap and is not separable by likelihood at this N.


## Local window EM (bottom-up, random starts) for hierarchy recovery (`benchmarks/o6/local_em.py`)
- **Lone node:** not identifiable (merging is exactly free). The smallest identifying neighbourhood is a two-level window, where the sibling is the second view.
- **Annealing from T0 = 2** collapses all symbols to one table. Random starts at T = 1 break the symmetry.
- **Local split–merge** guided by the merge-cost diagnostic. At L = 6 it lifts local-only L4 from 0.16 to 0.40 and L5 from 0.03 to 0.23.
- **Results** (NMI on identifiable levels, N = 20,000):

| run | L1 | L2 | L3 | L4 | L5 | gap |
|---|---|---|---|---|---|---|
| V8 L6 baseline | 0.909 | 0.970 | 0.957 | 0.794 | 0.655 | 0.005 |
| V8 L6 local only | 0.901 | 0.894 | 0.663 | 0.398 | 0.229 | 5.71 |
| V8 L6 local + 60 EM | 0.948 | 0.923 | 0.809 | 0.593 | 0.413 | 1.75 |
| V8 L6 hybrid | **0.963** | 0.970 | 0.957 | 0.768 | **0.668** | 0.049 |
| V16 L4 baseline | 0.983 | 0.986 | 0.810 | | | 0.020 |
| V16 L4 local only | 0.978 | 0.934 | 0.748 | | | 0.79 |
| V16 L4 local + 60 EM | **0.987** | 0.981 | 0.815 | | | 0.28 |
| V16 L4 hybrid | 0.977 | 0.982 | **0.832** | | | 0.021 |

- **As an initialiser:** with half the split–merge budget, the hybrid matches or beats the baseline's structure on most identifiable levels.
- **On its own:** it does not reach the top, because going only deeper loses top-down evidence and errors compound.
- **Cost:** the window E-step is O(K³), so total time is above the baseline's.


## The finite Giry monad (what the paper uses) and two propositions (`amb_vigneaux/giry.py`)
- **What we use.** On finite sets the Giry monad is the finite distribution monad, and its Kleisli category is FinStoch. We use:
  - FinStoch as a Markov category;
  - sufficiency of the soft hand-off (V2, 6.7e-16);
  - merging as monad averaging;
  - transports as Kleisli morphisms, with the loop operator as monodromy into FinStoch.
- **Decoding is Bayesian inversion.** The inside vector is the likelihood of the block's own leaves. The posterior inside × outside is the Bayesian inverse, checked exactly against brute force.
- **Soft decoder.** The normalised inside vector is unchanged by any other leaves (locality; change exactly 0 on V8 L6). A "soft-decoder functor" that returns it is constant on morphisms. What changes is the posterior: its argmax differs between block-only and whole-sequence evidence on 8.5 / 14.8 / 17.2% of sequences at levels 1–3. The paper's table caption has been corrected.
- **MI persistence is functorial.** By data processing, coarse-graining includes filtered complexes, so it induces a morphism of persistence modules, and the barcode is stable. HSIC persistence is not functorial: HSIC(2·sign X, Y) > HSIC(X, Y), and HSIC is not invariant under X → 10X.


## Gluability → outcome on a stream (AllSides; `examples/allsides/stream_outcome.py`, figure `stream_outcome.pdf`)
- **Setup:** 26-week windows, stepped every 2 weeks, 2016–2020.
  - Outcome: O = [f ≥ 0], with f = 2P_C − P_L − P_R on directed fact-checks of Trump ("yes" = the centre sits with the left).
  - Gluability: a parametric-bootstrap test of the loop-free model.
  - Counterfactual: the fewest centre articles to relabel or add to flip O.
  - Δf split into a rate part and a topic-mix part.
  - AMB outcome: which topics have a centre directed fact-check at all, with exact toggle radii.

| event | O | P(yes) | glue p | Δf (rate, mix) | flip cost before | AMB toggled |
|---|---|---|---|---|---|---|
| election 2016 | yes→yes | 0.81→0.57 | 0.04→0.33 | −0.017 (−0.026, +0.010) | relabel 3 of 294 | 4 topics |
| inauguration | yes→**no** | 0.66→0.28 | 0.46→0.14 | −0.016 (−0.020, +0.003) | relabel 2 of 330 | 3 topics |
| COVID | no→**yes** | 0.03→0.72 | 0.21→0.48 | +0.036 (+0.044, −0.009) | add 7 of 560 | none |

- **What drove the flips:** within-topic rates, not the topic mix.
- **Fragility:** the outcome is decided (|f| > 1.96 se) in only 3% of windows, and the flip cost is ≤ 7 articles throughout. This binary outcome lives on its own boundary.
- **Gluability:** the loop-free model holds in 92% of windows, where the pooled outcome reads only the natural part. It fails before the 2016 election and in 2019.
- **The AMB layer moves independently of the probability outcome:** the election and inauguration change the support in 3–4 topics, while COVID changes the outcome but not the support.


## Gluing-lattice dynamics: what jumps, what moves (`examples/dynamics/gluing_lattice_dynamics.py`)
- **Path:** e(λ) = λ·PR + (1 − λ)·noise on CHSH, with support tolerance τ = 0.02.
- **CF:** 0 up to λ = 1/2, then exactly 2λ − 1. It has no jump.
- **γ and S0:** both change only at the support collapse, λ = 0.92, where S0 goes from log 4 to log 2 and γ ≠ 0 (strong contextuality).
- **CF > 0 but γ = 0 on λ ∈ (0.5, 0.92):** probabilistically contextual, possibilistically clean.
- **IPF:** converges exactly where CF = 0.
- **Statistical verdict (one-sided CHSH test):** power at λ = 0.52 (CF 0.04) is 0.11 / 0.15 / 0.45 / 0.93 for N = 50 / 200 / 1000 / 5000.
- **Lattice:** every proper sub-cover is a tree and always glues, so the full cycle is the only minimal obstruction.
- **Corrected picture:** γ = 0 is not gluing; CF > 0 is not γ ≠ 0; obstruction does not pin probabilities; CF has no jump; the data lattice is statistical.

## Arity grading, p-adically (synthetic; `examples/arity_probe/padic_arity.py`)
- **Single units** glue at full precision (24/24): a filled cell is flat.
- **Units of arity ≥ 2 / 3 / 4** glue to radius 3^-1 / 3^-2 / 3^-4. These are nested gluing radii set by each grade's precision, with integer slopes above them.


## An outcome series and all layers on one axis (`examples/outcomes/`)
**Outcome.** Two poll series, not redistributed:
- FiveThirtyEight approval poll list, 2017 – Aug 2020 (5,920 "All polls" polls, adjusted net approval, weighted; mirrored on GitHub: kiranrangaraj/Trump-Tweets-and-Approval-Rating-ETL).
- 2016 national Trump − Clinton polls (HuffPost Pollster, 562 LV/RV polls; TheEconomist/us-potus-model).

**Lead–lag, on weekly changes at lags −8..+8 with a circular-shift null:** coverage never leads the outcome.
- f vs approval: r −0.19 at +6 wk, p 0.33.
- L−R gap vs approval: r −0.21 at −7 wk, p 0.25.
- 2016 margin: p ≥ 0.84.

**Events** (outcome before → after):

| event | before → after |
|---|---|
| Access Hollywood (margin) | −3.0 → −6.1 |
| Comey letter (margin) | −5.8 → −4.4 |
| Charlottesville (approval) | −12.1 → −16.9 |
| Mueller report (approval) | −11.4 → −11.2 |
| impeachment inquiry (approval) | −11.1 → −10.3 |
| COVID (approval) | −10.7 → −9.7 (a small rally) |

Charlottesville is a case of the outcome moving without coverage structure moving.

**All layers on one axis** (`layers_chart.py`, figure `layers_chart.pdf`), over 95 windows:
- **Triangle scenario:** L–C–R, adversarial yes/no on shared stories.
- **γ = 0 in all windows** (full support). CF is 0.01–0.10, so the data are probabilistically non-glueable but possibilistically clean.
- **CF is mostly signalling:** corr(CF, signalling) = 0.85, because different stories feed different pairs. It is a gluing defect, not Bell-type contextuality.
- **CF and the gluing defect peak in mid-2019,** after the Mueller report, the same period as the only non-natural change.
- **Natural and non-natural perturbations** have similar RMS (0.061 and 0.054).
- **Window-level correlations with approval** are 0.09–0.33. That is only about 7 independent windows, so none are significant.

**Synthesis.** A new paper section, "Representations and perturbations: a summary".


## Paths over the gluing lattice (`amb_vigneaux/lattice_path.py`, `examples/lattice_path/`)
**Setup.** A region is a set of topics, glued by the additive log-odds model. We hold out a target cell (s*, t*) and predict p_hat; the outcome is p_hat > 1/2. Paths are chains of glueable regions along which p_hat rises or falls.

**Proposition (region dependence is loop content).** With additive (loop-free) truth, every region gives the same prediction. The spread across regions comes only from loops and noise.

**MBIC** (82 targets; regions of 5–7 topics, about 5,000 per target):
- **Envelope:** median width 0.34. The truth lies inside it for 53/82 targets, and region choice alone can flip the outcome for 36/82 (44%).
- **Example:** HuffPost × sport rises from 0.60 to 0.94 by dropping 9 topics, and falls to 0.35 by dropping others.
- **Mean absolute error against held-out truth:**

| rule | MAE |
|---|---|
| full cover | 0.112 |
| median over glueable regions | 0.112 |
| glue-p weighted average | 0.112 |
| most consistent region | 0.120 |
| raise-path end | 0.172 |
| lower-path end | 0.185 |

- **Tiny regions** (2 topics) always glue, because the test has no power, and give p_hat anywhere in 0.05–0.9. A minimum region size is needed.

**Synthetic** (3 of 12 topics contaminated, where the full cover never glues):

| rule | MAE |
|---|---|
| full cover | 0.045 |
| consistency-pruned | **0.039** |
| raise-path end | 0.073 |
| lower-path end | 0.078 |

**Conclusion.**
- A consistency path is estimation, and it helps.
- An outcome path is cherry-picking on loops, and it hurts.
- The honest report is the envelope plus the consistency path. A real rise in an outcome's probability needs a change in the world, at the toggle costs, not a change in which data are glued.


### Feedback, paths vs surfaces (`examples/lattice_path/landscape.py`; 12 MBIC targets, all glueable regions ≥ 5 topics)
- **The landscape is first order.** Per-topic influences explain R² 0.89 (median) of logit p_hat over the lattice, and 0.98 with pairwise terms.
- **Proposition (first-order landscape).** The influence of topic x is the weighted loop residual of the target's outlet on x (block DFBETA). Measured correlation: median 0.97, range 0.86–0.99.
- **Surfaces, not just paths.** Aligned moves (add a positive-influence topic, drop a negative one) raise p in 86% of cases. The ascending set is an interval of the lattice, keeping the raisers and dropping the lowerers, and almost any walk inside it climbs.
- **Feedback.** Hill-climbing (accept only raising steps) reaches the global maximum from 59% of starts in about 6 steps. Pairwise terms create local optima, so a few restarts are needed.
- **Meaning.** The feedback rule works because each topic's sign is stable: it is the sign of the target outlet's loop residual. That makes the surface useful as a diagnostic (which topics carry the outlet-specific interaction), but using it to raise the outcome is evidence selection.


### Surface figure and relation to the literature (`examples/lattice_path/surface_chart.py`)
**Breitbart × gender** (held-out 0.67, full cover 0.53):
- Topic influences equal the outlet's loop residuals (corr 0.98).
- p_hat is almost a function of the loop-content score (R² 0.94).
- Un-gluing vaccines, student debt and international politics lifts p_hat to 0.72; the mirror path lowers it to 0.32.
- Here the raise path happens to reach the truth. Over 82 targets such paths are 55–65% worse.

**Literature.** The negative result is standard:
- post-selection bias: data splitting (Cox 1975), PoSI (Berk et al. 2013), data carving (Fithian–Sun–Taylor 2014);
- selection bias in model evaluation (Varma–Simon 2006; Cawley–Talbot 2010);
- the form is block DFBETA / Cook's distance (Cook 1977; Belsley–Kuh–Welsch 1980).

The consistency path adds no directional bias, but its post-pruning standard errors need data splitting or selective inference. What is new is narrower: identifying the influence blocks with loop (H¹) coordinates, the spread across regions being loop content, and the ascending set being a lattice interval.

## Presheaf of Markov kernels on the cover: viability (paper §sec:kernelpresheaf)

`examples/kernel_presheaf/run.py`, module `amb_vigneaux/kernel_presheaf.py`.

- The presheaf is exactly Abramsky–Brandenburger's distribution presheaf D_R∘E, which the engine already runs on. With explicit settings, restriction marginalises outputs only; the proposed formula also sums over settings, which is not a kernel (the mass becomes |X|).
- Conditional products exist for every compatible pair, so their existence is not the gluing criterion. Iterated from uniform, they are one IPF sweep.
  - Acyclic covers, running-intersection order: one sweep glues 200/200 random compatible families (worst TV 6e-16).
  - Cyclic (CHSH, λ·PR + noise): the chain contexts are exact, and the closing defect is exactly (λ+λ³)/2 (correlations compose along the chain). It is positive on (0, ½], where CF = 0.
  - Random non-contextual families on the 4-cycle: one sweep glues 0/200 (median TV 0.013). The best of the 24 orders still gives 0.060 at λ = ½.
- Adjunction: image ⊣ preimage. The counit is an isomorphism iff the family is non-contextual; its failure is measured by CF, and the LP dual (a Bell inequality) is the certificate. Linearising (signed measures) makes every compatible family glue (PR residual 2.5e-16), so a linear kernels⇄cochains adjunction cannot see probabilistic contextuality.
- Non-abelian cohomology:
  - Kernels are a monoid, not a group or 2-group, so Giraud/Breen do not apply as stated.
  - The well-posed piece is the invertible transports, whose H¹ = Hom(π₁, G)/G is the existing monodromy (§sec:mono).
  - Deterministic compatible families always glue (E is a sheaf).
  - For lossy families the complete invariant is already CF plus its Bell dual.
- AllSides triangle: its windows mostly fail descent (signalling, median 0.025; corr with CF 0.85), not gluing.

## Hull covers on the lattice: discs without direction (paper §sec:hullcover, Prop. hulldirection)

`examples/lattice_path/hull_experiment.py` (output in `hull_experiment_out.txt`); test `tests/test_hull_direction.py`.

- **Method.** Single-linkage (Euclidean) hull on the topics, with the target cell held out. Three features: value (pooled rate), profile (logit column), loop (weighted loop-residual column). Canonical region = the largest disc that glues.
- **Hull vs full cover** (mean |p̂ − truth|):

| Corpus | Hull | Full cover | Notes |
|---|---|---|---|
| MBIC | 0.112 | 0.112 | hull always returned the full cover (0/82 sub-regions) |
| AllSides adversarial | 0.020 | 0.020 | hull always returned the full cover |
| AllSides directed | 0.006 | 0.006 | hull always returned the full cover |
| NewsWCL50 | 0.138–0.139 | 0.128 | sub-regions chosen for exactly the 8/23 cells whose held-out fit tears |
| BASIL | 0.087 | 0.087 | 2 topics, nothing to choose |

- The spread along the hull chain is ¼–⅙ of the spread over all glueable regions (MBIC 0.02–0.06 vs 0.34).
- The oracle best region (which uses the truth) reaches 0.021 on MBIC and 0.056 on NewsWCL50.
- Stitching regions by the correlation of loop residuals across the other sources does worse: 0.130 on MBIC, 0.141 on NewsWCL50.
- Modelling the loop instead:
  - NewsWCL50: rank-1 0.076, tropical rank-2 (min) 0.090, vs full 0.128 (16/23 cells, p = 0.09).
  - On the gluing corpora, both lose to the full cover.
  - The better tropical sense differs by corpus (min on NewsWCL50, max on MBIC).
- **Direction.** Relabelling k → n − k leaves every hull chain and merge height unchanged (495/495 target × feature cases) and flips the target's signed loop residuals (165/165 targets). Hulls, γ and CF measure fragility and scale, not direction. Direction needs the signed loop part on the lattice and the probabilities.

## Bounds on an outcome over all gluings (paper §sec:bounds, Prop. bounds, Fig. bounds)

Code: `amb_vigneaux/bounds.py`, `examples/bounds/run.py` (output in `out.txt`, figure `bounds_chart.pdf`), tests in `tests/test_bounds.py`.

- **Method.** [lo, hi] = min/max of ⟨f, P⟩ over the global laws P consistent with every local table (two LPs).
  - Directional: relabelling the outcome maps the interval to [1−hi, 1−lo].
  - No selection: the range is over all gluings at once.
  - An empty preimage is the counit failure. The fallbacks are the nearest-consistent preimage (reported with its defect) or the non-contextual part (reported with CF).
- **Identification.** An outcome is pinned to a point iff f ⊥ ker R. On the triangle, ker R is spanned by the parity χ = (−1)^(L+C+R):

| Outcome | ⟨f, χ⟩ | Identified? |
|---|---|---|
| unanimity | 0 | yes (= 1 − ½ Σ_pairs P(xᵢ ≠ xⱼ)) |
| all three | −1 | no |
| majority | +2 | no |

- **CHSH chain.** The bound on the closing correlator is [max(−1, 3λ−2), 1]. The truth −λ lies inside iff CF = 0, so the held-out bound is the CHSH inequality.
- **Split validation** (200 splits; 1,600 MBIC sentences rated by L/C/R annotator groups; 736 AllSides stories covered by all three sides):
  - Bootstrap-widened bounds cover 0.97–1.00 for every non-identified outcome. Widths: MBIC 0.18–0.34; AllSides 0.06–0.17.
  - Unanimity is a point (width 0). Its coverage is 0.98 once the sampling noise of the truth is included.
- **AllSides natural design** (pair tables from pair-only stories, truth from triple stories):
  - All three: [0, .028] contains .023.
  - Majority: [.052, .108] contains .105.
  - Held-out pair: [0, .084] contains .052.
  - Missed: unanimity, .670 vs .626 (population shift).
- **AllSides stream** (115 windows, plug-in): median width 0.030 (all three) and 0.059 (majority); the observed triple value is inside in 75% and 70% of windows.

## Grey (what-if) contexts and the population test (paper §sec:grey, §sec:bounds)

Code: `bounds.grey_bounds` / `hypothesis` / `event`, `examples/grey/run.py` (output in `out.txt`, figure `grey_chart.pdf`), tests in `tests/test_grey.py`.

- **What a grey context is.** It enters the cover as either:
  - a population table (equalities), or
  - a hypothesis a ≤ P(E | G) ≤ b (linear rows).
- **What it gives.**
  - Consistency: is the what-if possible? If not, the minimal L1 slack says how far off it is.
  - Bounds on a conditional target, via a Charnes–Cooper linear-fractional LP. This is the conditional-bounds item (a) from the previous exchange.
  - What-if curves, by sweeping the hypothesis.
- **MBIC validation.** The joint (filter × label) is hidden; {X, Y} and {X, filter} come from different annotator halves (50 splits).
  - Coverage 1.00 in all 11 filter × covariate settings.
  - The bounds are nearly vacuous (width 0.95–1.00).
  - The independence (max-ent) point is close (error 0.009–0.037), but only because age etc. is in fact nearly unrelated to the label. The data cannot show that.
- **What-if curves.**
  - MBIC: no value of P(biased | 55+) = s is ruled out. s = 0.875 forces P(biased | 55+, right) ≥ 0.60; s = 0.125 forces it ≤ 0.40.
  - AllSides: one hypothesis on the triple closes the loop, so the outcome is identified. P(majority adversarial) falls from 0.108 to 0.052 as s goes from 0 to 0.70. s > 0.70 is ruled out (slack 0.002–0.012).
  - At the triple-story value s = 0.415 the identified majority is 0.077, against 0.105 observed (the population shift).
- **Population test** (pair-only vs triple-story table for each pair):

| Pair | TV | χ² p |
|---|---|---|
| L–R | 0.014 | 0.94 |
| L–C | 0.031 | 0.53 |
| C–R | 0.097 | 2e-4 |

  For C–R, P(adversarial) on triple stories is C 0.174 vs 0.101 and R 0.130 vs 0.080. The shift explains the unanimity miss and the stream misses.
- **Text edits.** The bounds are "ordered, not a lever". The population assumption is now stated together with its test.

## Several conditionals: monad, adjunction, and why not cup products (paper §sec:multicond, Prop. multicond)

Code: `bounds.loop_space` / `condition_rank` / `nerve_betti`, `examples/multi_conditional/run.py` (output in `out.txt`, figure `multi_conditional.pdf`), tests in `tests/test_multi_conditional.py`.

- **Cup products cannot be the compatibility mask.** Every cover here has H² = 0 (nerve Betti numbers: triangle and CHSH (1,1), star (1,0)), so the mask would call every combination compatible.
- **The count that matters is dim ker R.** That many conditionals can be imposed independently; independence is the rank of the hypothesis rows on ker R. The real mask is LP feasibility, with the L1 slack as distance.
- **AllSides triangle** (dim ker R = 1; rank of the two triple conditions = 1):
  - s₂ is a function of s₁: 0.177 → 0.250, 0.530 → 0.750.
  - (0.177, 0.750) is allowed by each condition alone but jointly ruled out (slack 0.014); the cup-product mask would say compatible.
  - Triple stories give (0.415, 0.531), off the line (predicted 0.587): the population shift again.
- **MBIC filters 55+ and female** (dim ker R = 9, rank 2): 81% of the (s₁, s₂) grid is jointly possible. At the true (s₁, s₂):

| Target | No conditions | Both conditions imposed | Hidden truth |
|---|---|---|---|
| P(biased \| 55+, female) | [0, 1] | [0.412, 0.937] | 0.608 |
| P(biased \| neither) | [0.239, 1] | [0.550, 0.625] | 0.576 |

- **Monad.** Chaining conditionals is the Kleisli composite (μ), exact only under conditional independence. For P(biased | 55+, female) it gives 0.591.

## Second-order holonomy: covers whose nerve is a sphere (paper §sec:higher, Prop. higher, Fig. higher)

Code: `amb_vigneaux/higher.py` (nerve, Betti numbers, path 2-groupoid with filled loops, `complex_instrument` → JSON for plotting, `plot_complex` representational drawing), `examples/higher/run.py` (output in `out.txt`, figure `higher_chart.pdf`), tests in `tests/test_higher.py`.

- **Covers whose nerve is a sphere.** The contexts are all (n−1)-subsets of n measurements: the nerve is the sphere S^{n−2}, with Betti numbers (1, 0, …, 0, 1). ker R is spanned by the top parity (checked for n = 3, 4, 5).
  - Tetrahedral cover: 3 graph loops, all 3 filled by 2-cells.
  - Triangle and CHSH covers: their 1 loop stays open.
- **Higher PR box (n = 4)**, each triple uniform on even parity:
  - Every pair marginal is uniform at every λ, so the pair (first-order) view gives CF = 0 at every λ.
  - The triple view gives CF = 9/8·(λ − 1/3)₊.
  - γ ≠ 0 only at λ = 1.
- **MBIC four ideology groups** (1,200 sentences; 100 five-way splits):

| Outcome | ⟨f, χ₄⟩ | Bootstrap width | Coverage (with truth noise) | Plug-in coverage |
|---|---|---|---|---|
| All four | +1 | 0.095 | 0.99 | 0.27 |
| At least three | −3 | 0.175 | 1.00 | |
| Unanimous | +2 | 0.137 | 1.00 | |
| Even parity | +8 | 0.406 | 1.00 | |

  - None of the four outcomes is identified.
  - The split tables show median CF 0.11 with γ = 0. With the same sentences for every context, CF = 0. So 0.11 is the sampling floor (about 240 sentences per context), not a four-way obstruction.

## Toric geometry of the sphere covers, component by component (paper §sec:toricsphere, Prop. birch)

Code: `amb_vigneaux/toric.py`, `examples/toric/run.py` (output in `out.txt`, figure `toric_chart.pdf`), tests in `tests/test_toric.py`.

- **Components, each computed on its own:**
  - [F] fibre: the segment P0 + tχ, in the mixture coordinate.
  - [T] toric variety: θ = 0, a single binomial of degree 2^{n−1}.
  - [I] observed θ̂ ± CI, from the full joint.
  - [S] exact toggle radius per cell, against the saturated radius.
- **Synthetic** (n = 3, 4, 5; 50 random laws each):
  - Birch point = IPF to 5e-15 and the binomial holds at IPF to 2e-14; θ is strictly increasing along the fibre.
  - Higher PR box: a signed law with its margins exists, but positivity needs t ∈ [0.1875, −0.0625], which is empty, so [F] is empty.
- **Real tables:**

| Table | [F] segment in m | [I] θ̂ (95% CI) | TV(observed, Birch) | [S] radius / saturated |
|---|---|---|---|---|
| MBIC 3 groups | [−0.923, 0.060] | −0.050 [−0.111, 0.011] | 0.016 | median 1.46 (1.28–3.38) |
| AllSides 3 sides | [0.200, 0.557] | −0.053 [−0.177, 0.071] | 0.009 | median 1.53 |
| MBIC 4 groups | [−0.121, 0.528] | −0.015 [−0.087, 0.058] | 0.004 | median 1.50 |

  No top-order interaction is detected in any table, and every minimal forced set in [S] has 2 cells.
- **One outcome, P(all positive), through the components:**

| Table | [F] bounds | [F]+[T] max-ent | [I] observed | Effect of θ |
|---|---|---|---|---|
| MBIC 3 | [.407, .530] | .445 | .449 | +.004 |
| AllSides | [0, .045] | .021 | .024 | +.002 |
| MBIC 4 | [.329, .369] | .349 | .348 | −.001 |

## Curvature: the fifth component [C] (paper §sec:curvature, Fig. curvature)

Code: `amb_vigneaux/curvature.py` (Efron γ², pull/shift jets, hidden-mixture EM), `examples/curvature/run.py` (output in `out.txt`, figure `curvature_chart.pdf`), tests in `tests/test_curvature.py`.

- **Data.** 2016 polls (POLLS2016_CSV): 2,075 polls from 252 pollsters; 156 Dem-affiliated, 102 Rep-affiliated.
  - Baseline: race + two-week period + mode + population.
  - Over-dispersion τ² = 0.0027; standard errors by pollster-cluster bootstrap.
- **[C1] Velocity** (first-order house effect):

| Affiliation | δ (logit) | Cluster SE | Margin points |
|---|---|---|---|
| Dem | −0.050 | 0.016 | −2.5 (toward Clinton) |
| Rep | +0.013 | 0.015 | +0.6 (not significant) |

  The bend is larger for sponsor-commissioned polls (−0.065 / +0.028) than for pollster-affiliated ones (−0.039 / +0.002).
- **[C2] Curvature.**
  - κ (sponsored residual vs race baseline): Dem +0.050 (SE 0.062), Rep −0.111 (SE 0.083). Neither differs from 0.
  - Pull vs shift weighted SSE: 98.1 → 97.4 and 89.3 → 86.8, a negligible gain.
  - Whole-sample Efron γ² ≈ 5e-7, far below 1/8. The bend is e-flat (a log-linear shift), so first-order inference is reliable.
- **[C3] Singular point.** Hidden sponsorship among 1,817 nonpartisan polls: π_D 0.04, π_R 0.09; LR 1.34, bootstrap p 0.27.
  - The null has boundary mass (LR = 0 in 19% of draws), and its 95% point is 3.47.
  - No hidden sponsorship is detectable.

## Deformations at the boundary: T¹, T² and the flop (paper §sec:deform, Prop. deform, Fig. deform)

Code: `amb_vigneaux/deform.py` (boundary supports, Tjurina numbers by Gröbner basis, radii, flop switch rate), `examples/deform/run.py` (output in `out.txt`, figure `deform_chart.pdf`), tests in `tests/test_deform.py`.

- **Boundary supports.** A zero set occurs in the closure of the model iff it meets both parity classes (checked on all 2^8 sets for n = 3, and up to 4 zeros for n = 4).
- **Local structure** (a even and b odd zeros, f ~ x₁…x_a − y₁…y_b):

| (a, b) | Type | Tjurina number |
|---|---|---|
| min(a, b) = 1 | smooth | 0 |
| (2, 2) | conifold node | 1 |
| a + b > 4 | non-isolated; singular-locus dim (a−2)+(b−2), transversally a node | ∞ |

  - T² = 0 everywhere, because the model is a hypersurface.
  - The θ-level sets are equisingular (Tjurina number 1 for c = 0, 1, −2).
- **Flop** (perturbations of 1–20%, near each kind of point):

| Near | Partner margin | Forced partner flips |
|---|---|---|
| a smooth boundary point | 0.90 | 0% |
| a node | 0.002 | 38–54% |

- **Real tables:**

| Table | Toggle radius | Node radius | Partner margin | Forced pair changes under resampling |
|---|---|---|---|---|
| MBIC 3 groups | 0.131 | 0.313 | 0.16 | 37% |
| AllSides 3 sides | 0.046 | 0.142 | 0.43 | 18% |
| MBIC 4 groups | 0.041 | 0.092 | 0.11 | 55% |

  The toggle's cost is stable; its identity is not.

## Singular statistics and optimal learning [L]: net difference in outcomes (paper §sec:singular, Fig. singular)

Code: `amb_vigneaux/singular.py` (flop posterior, tempered Metropolis, WBIC/λ̂ over chains, thermodynamic integration, latent class model), `examples/singular/run.py` (output in `out.txt`, figure `singular_chart.pdf`), tests in `tests/test_singular.py`.

- **Calibration** (shift mixture, n = 2000): λ̂ = 0.35 ± 0.07 at a singular truth (theory 1/2, m = 2), and 0.95 ± 0.09 at a regular one (theory 1).
- **Net differences, regular → singular-aware:**

| Outcome | Regular / plug-in | Singular-aware / Bayes | Net difference |
|---|---|---|---|
| Toggle's forced pair (MBIC3 / AllSides / MBIC4) | implicitly certain | posterior 0.61 / 0.81 / 0.48 | certainty overstated by 19–52 points |
| Toggle radius | — | stable | negligible |
| Hidden poll component (shift free; LR 23.2) | BIC: present | bootstrap p < 0.005; TI log BF +4.9: present; WBIC: tie; λ̂ = 0.96 | real, but it is ~0.3–0.8% of polls at −0.275 logit (~14 margin points): outlying polls, not sponsorship |
| Latent classes, MBIC 4 | BIC: K = 2 decisively | λ̂ ≈ 1.96 / 5.0 / 4.9 / 4.9 for K = 1–4 (d/2 = 2 / 4.5 / 7 / 9.5); WBIC ties K = 2–4 | BIC over-penalises; P(all four) unchanged (0.347–0.350) |
| Held-out log-loss (Bayes − ML) | — | K = 2: +0.0001 ± 0.0002; K = 3: −0.0018 ± 0.0004 (Bayes better in 72% of splits) | posterior averaging helps only in the singular, over-parameterised model |

## Split–merge and the learning coefficient (paper §sec:splitmerge-rlct, Prop. splitrlct, Fig. splitmerge)

Code: `singular.cat_lca_*`, `examples/splitmerge/run.py` (output in `out.txt`, figure `splitmerge_chart.pdf`); test `test_lone_node_extra_symbols_are_free_window_is_not`.

- **Lone node** (one child on 16 cells): λ̂ = 7.37 / 7.20 / 8.28 / 7.72 for K = 1–4 (theory 7.5; d/2 = 7.5 / 15.5 / 23.5 / 31.5), with identical maximum likelihood for every K. Extra symbols are free, as for the root.
- **Window** (3 children on 4, true K0 = 2): λ̂ = 4.86 / 7.57 / 8.55 / 8.39 for K = 1–4 (d/2 = 4.5 / 9.5 / 14.5 / 19.5). A spurious split adds about 1, not 5.
- **Criterion.** Compare free energies: merge iff nΔK < (λ_split − λ_merged) log n. "Reduce the RLCT" is not a criterion, because every merge does.
- **In practice** (4 data sets per n, n = 300–10⁴):
  - Held-out log-likelihood and BIC reject spurious splits at every n.
  - WBIC and the Bayes factor are noisier at our Monte-Carlo budget (accept the merge in 25–100% of data sets).
  - A weak real split (25% apart) is undetectable by every criterion up to n = 10⁴.
- **References added:**
  - Watanabe 2018;
  - Golubtsov 2002;
  - Fritz–Gonda–Perrone–Rischel (arXiv:2010.07416);
  - Fritz–Gonda–Perrone, de Finetti (arXiv:2105.02639);
  - Bohinen–Perrone (arXiv:2502.14941).

## Arithmetic universe remark (paper rem:au, framework section)

- **A transformer is not an AU.** The category of contexts and derivations has no finite limits, and concatenation is not a coproduct.
- **A fixed finite-precision transformer is an interpretation** (into an AU such as Set) of a finite theory whose free AU is Maietti's syntactic category.
- **What the AU does give:** a two-column split of the engine.
  - Finitary (exact, rational, decidable): counts, supports, AMB over ℤ/n, LP and MILP over ℚ, tree recursion, relabelling quotients, toric binomials.
  - Analytic: eigenvalue certificates, IPF limits, curvature, RLCT and MCMC, real-valued radii.
- The finitely supported rational distribution monad is constructible inside an AU (construction sketch).
- **References:** Joyal 1973; Maietti 2010.

## Net assessment and theorem index (paper §sec:net, §sec:theoremindex)

- **Net assessment** (section before the Status part): a table of each piece with its key outcome, whether a standard method reproduces it, and its net gain; the first- vs second-order reading; costs; and the structural comparison with transformers.
- **Index of theorems and results** (Status part), generated from the source so the numbers match the statements:
  - A. External results: 12 stated as theorems, plus those invoked in the text, each with its source and where it is used.
  - B. The paper's own 84 results, by section, with status tags.
- 26 previously unlabelled results now carry labels (`res:autoN`) so the index can reference them.

## The constructive core: rational distribution monad in an arithmetic universe (paper §sec:constructive)

Code: `amb_vigneaux/constructive.py` (exact Fraction arithmetic), tests in `tests/test_constructive.py` (4 tests).

- **Definition.** D(X) = L(X × ℚ>0)/∼, with ∼ generated by permutation, merging and scaling. The bijection version is wrong: it separates [(x,1),(x,1)] from [(x,1)].
- **Lemmas.**
  - Normal form: with decidable equality on X, ∼ is decidable.
  - Lifting: L(π) is a regular epimorphism (initial-algebra argument).
- **Propositions.**
  - D(f), η and μ are well defined and natural.
  - The monad laws hold, checked exactly on 200 random nested lists.
  - D is commutative and affine.
- **Theorem.** Kl(D) is a Markov category: commutative + affine, with copy = η∘Δ and discard = !. It is not cartesian.
- **Definition.** Conditioning and Vigneaux's action for decidable observables. The Tsallis-2 cocycle holds exactly in ℚ.
- **Corollaries.** The finitary/analytic boundary, corrected:
  - the contextual fraction of a rational table is finitary;
  - log-based quantities (Shannon entropy, KL, toggle radii) are analytic, but the order of toggle radii is finitary;
  - the Birch point and eigenvalues are real-algebraic.
- Remark rem:au now points to the proofs instead of saying "sketch".

## Synthesis dictionary, a dataset for the open problems, new open problems (paper §sec:synthesis, §sec:crowd)

- **Dictionary** (Table tab:dict): fibre (flat polytope; bounds) | toric scheme (binomial; Birch point; on 2×2×2 the binomial in log coordinates is the loop holonomy Φ) | top interaction θ | support strata and toggle radius | nodes, T¹, T² = 0, the flop.
  - Corrected: contextuality sits **outside** the image of restriction (it is not a stratum); full support does not imply non-contextual (γ = 0 while CF > 0); the probability layer is the whole fibre, not the Birch point.
  - A PCFG is not toric: grammars live on latent-tree varieties.
- **Dataset: CrowdER product** (Wang et al. 2012; via github.com/zhydhkcws/crowd_truth_infer). 8,315 items, 176 workers, exactly 3 workers per item, gold truth for every item. This gives natural tetrahedral covers.
  - **Pilot** (`examples/crowd/pilot.py`): 15 worker quadruples with every triple observed on ≥ 15 items (Betti (1,0,1)).
  - Raw CF exceeds the sampling floor in 3/15. Those three have very different gold rates across triples (e.g. 0.71 / 0.38 / 0.18 / 0), i.e. population differences.
  - Gold-negative items only: 1 of 4 above the floor (p = 0.02, not significant after correction).
  - Bounds on P(≥ 3 of 4 say "match") are 0.01–0.04 wide.
- **New open problems** (status list):
  - real second-order structure (the crowd pilot is the starting point);
  - per-level RLCT for O6 and better free-energy estimates;
  - a flop law, and T¹ on non-isolated strata;
  - a second tearing corpus;
  - an empirical transformer comparison.

## Crowdsourced entity resolution: full exploration (paper §sec:crowd, Fig. crowd)

Code: `examples/crowd/explore.py` (output in `explore_out.txt`, figure `crowd_chart.pdf`); tests in `tests/test_crowd.py`. Data: CROWD_DIR (not redistributed).

- **(A) Second order.** Tetrahedral covers from the 60 most active workers, every triple on ≥ 12 items.
  - All items: 16 covers; 4 above the floor at p < 0.05; 2 survive Benjamini–Hochberg, and both have a significant gold-rate difference between triples (population difference).
  - Gold-negative items: 11 covers; 2 at p < 0.05; none survives correction (median CF 0.20 vs floor 0.25).
  - No genuine four-way obstruction.
- **(B) Co-errors vs independence given the truth** (observed / expected, item bootstrap 95%):

| Gold class | Two workers err | All three err |
|---|---|---|
| 0 | 1.13 [1.03, 1.21] | 1.62 [1.19, 2.13] |
| 1 | 1.16 [1.07, 1.26] | 1.58 [1.31, 1.86] |

  Item difficulty acts as a second latent variable beyond the truth.
- **(C) Worker × difficulty** (difficulty = how many of the other two workers erred).
  - Error rate 0.16 / 0.20 / 0.37 by difficulty; logit effect +0.54 / +1.69.
  - Additive-model loop residual Q = 633 on 115 df (p = 1e-72).
  - Caveat: Q/df = 5.5 also includes over-dispersion.
- **(D) Outcomes.**
  - 41% of items are pivotal 2–1 splits, carrying 83% of majority-vote errors (0.208 vs 0.030).
  - Accuracy: majority vote 0.897, one-coin Dawid–Skene 0.907.
- **(E) Latent classes.** In the busiest single triples (49–114 items), BIC and held-out likelihood both pick K = 1: the dependence pooled in (B) can't be resolved per context.

## Crowd pipeline: planted positives and nulls on the real design (paper §sec:crowd, Fig. crowdsyn)

Code: `examples/crowd/synthetic.py` (output in `synthetic_out.txt`, figure `crowd_synthetic.pdf`), tests in `tests/test_crowd_synthetic.py`. Real items, assignments, gold labels and worker accuracies; simulated answers.

| Detector | DS (null) | +difficulty | +dif+int | +H(0.5) | +H(1.0) | all |
|---|---|---|---|---|---|---|
| [A] planted covers BH-significant | 0/5 | 0/5 | 0/5 | 0/5 | 5/5 | 5/5 |
| [A] disjoint covers BH-significant | 0/2 | 0/2 | 0/2 | 0/2 | 1/2 | 0/2 |
| [B] pair co-error O/E | 0.97 | **1.10** | 1.04 | 1.06 | 1.12 | 1.12 |
| [C] Q/df, difficulty proxy | 0.64 | 0.78 | 0.80 | 1.79 | 3.37 | 2.72 |
| [C] Q/df, true difficulty | 0.66 | 0.68 | **5.98** | 0.70 | 0.67 | **5.47** |

- **[A] power, one cover.**
  - False-positive rate at λ = 0 is 0.01–0.06.
  - At λ = 0.5: 0.75 with 15 items per triple, ≥ 0.91 from 30 items.
  - So the real null result rules out strong obstructions (λ ≳ 0.75) only.
- **[B]** detects planted difficulty; the real 1.13 is of the planted size.
- **[C] with the proxy is not a worker × difficulty test.** It is blind to the planted interaction but inflated by correlated answering. With true difficulty it detects the interaction and ignores the parity box. So the real Q/df 5.5 points to dependence among co-assigned workers. The paper text is corrected accordingly.
- **Real vs simulated.** In the real data, Dawid–Skene gains +0.010 vs +0.06–0.10 in simulation, and errors are less concentrated on pivotal items (0.83 vs 0.88–0.96): real errors are less one-coin.

## Second-order information geometry [G2] on JetClass-II (paper §sec:infogeo2, Figs. g2planted, jets)

Code: `amb_vigneaux/infogeo2.py`; `examples/jets/{census,planted,run}.py` (outputs `*_out.txt`, `*.json`, `planted_chart.pdf`, `jets_chart.pdf`); tests `tests/test_infogeo2.py` (7).

**Data.**
- 297,658 jets from JetClass-II (23% QCD; 188 classes; simulation truth).
- Four binary observables (mass > 50 GeV, τ21 < 0.5, τ32 < 0.65, more than 40 constituents), giving 16 cells.
- A reco-vs-gen table (mass and multiplicity, detector vs generator) that sits near a node.
- φ is the symmetry null.

**Planted checks.** Every quantity is recovered:
- Each geodesic is flat in its own connection. Shift: γ²ₑ = 0.000 ± 0.001. Pull: γ²ₘ = 0.01 ± 0.06. Great circle: κ²_FR = 0.00 ± 0.05.
- Bent path: 4.57 vs 4.52.
- Holonomy is unbiased: twist 0 / 0.04 / 0.1 gives −0.003 / 0.019 / 0.117 nats vs 0 / 0.019 / 0.120 true.
- Efron–Hinkley n·Var(J/nI) at n = 1600 matches γ²ₑ (4.58 vs 4.52; 17.98 vs 18.09) and is exactly 0 for an e-flat family.
- Node branch posterior: 0.50 with equal partners, 1.00 with separated partners.
- Order-4 information: 0 / 0.0009 / 0.0044 vs 0 / 0.0008 / 0.0046.
- Rule: local curvature needs a moving path; a slow path (‖v‖ ≈ 0.18) needs about 100× the data.

**Real results.**

| | all | QCD | resonance |
|---|---|---|---|
| FR length / chord of the pT path | 1.40 / 0.88 | 1.81 / 1.21 | 1.29 / 0.94 |
| e-flat path rejected, LR (15 df) | 6396 | 1068 | 2938 |
| quadratic still off (dev / df) | 628 / 135 | 344 / 135 | 372 / 135 |
| n·Var(J/nI) at n = 25 / 1600 (centre γ²ₑ) | 10.6 / 24.3 (25.2) | 8.6 / 15.1 (15.3) | 10.9 / 29.5 (32.4) |
| error-bar wobble at n = 25 / 100 | ±65% / ±38% | ±59% / ±35% | ±66% / ±38% |
| holonomy pT × \|η\| (1e-3 nats/jet) | 1.26 ± 0.04, z = 31 | 1.25 ± 0.18, z = 7 | 1.51 ± 0.05, z = 30 |
| holonomy pT × φ (null) | 0.12 ± 0.06, z = 2.0 | −0.03 ± 0.28 | 0.13 ± 0.09 |
| holonomy \|η\| × φ (null) | −0.01 ± 0.04 | −0.21 ± 0.18 | −0.02 ± 0.06 |
| order-4 information, pT bins 0–3 / 4–7 (1e-3 nats/jet) | 0.15–0.47 / ≤ 0.03 | wide | 0.11–0.38 / ≤ 0.01 |

- **Curved in every connection.** e, m and FR curvatures are of similar size, so the path is neither a shift nor a pull. They grow with pT as the path slows.
- **Not a class-mixture effect.** A pull between fixed QCD and resonance laws misses the pooled law by TV 0.04–0.26 per bin, and QCD alone bends.
- **Holonomy only off the symmetry.** It is concentrated at the lowest pT and the highest |η|; every φ grid is flat within noise.
- **Flop on real data.**
  - Reco-vs-gen QCD branch posteriors are 0.37–0.95; for all jets, 0.46 in one bin. Shape tables (far from the node) mostly give 0.97–1.00.
  - Branch certainty is ordered by the partner margin across 32 table–bin points.
  - On the shape tables the margin falls with pT (0.97 → 0.69 for all jets, 0.96 → 0.30 for QCD).
- **Order-4 information switches off above mid pT**, matching θ: +0.10 → 0 in the census.
- **Not done yet:** the RLCT at real nodes; the 2-jet coefficient κ with intervals; further covariate pairs; third-order neighbourhoods.

## Holonomy as a size of the class, the flat ridge, and flip prediction (paper §sec:bridge2)

Code: `amb_vigneaux/holonomy_norm.py`; `examples/holonomy_norm/run.py`, `examples/ridge/run.py`, `examples/flips/run.py`; tests `tests/test_holonomy_norm.py` (4), `tests/test_ridge.py` (1).

**Statistical holonomy KL(g) = KL(p ‖ hol(g)·p) on a loop.** The proposed "Theorem 3.3" is corrected to Prop. holnorm:
- **Class function.** It depends only on [g]: max change under gauge or base change over 200 cases = 0.
- **Zero set.** KL = 0 iff hol(g) fixes p. The uniform law is a counterexample to "iff [g] = 0" unless the action is free; a period-n/2 law has KL = 0 at h = n/2.
- **Not a norm on ℤ/n.** On ℤ/32, KL(2)/KL(1) = 3.99 but KL(8)/KL(4) = 2.97, and |KL(h) − KL(−h)| reaches 0.25.
- **Tilts.** KL = Bregman divergence of the cumulant function exactly (error ≤ 2e-16), and ½ hᵀ Cov(T) h to O(|h|³). So √(2·KL) is the Fisher norm of the class to second order, and a norm iff the action is faithful.
- **JetClass-II plaquettes.** Exact KL / ½HᵀIH has median 0.995 (range 0.94–1.14).

**Flat ridge (Prop. ridge).** With margin-only data, the Fisher rank is d − dim ker R and λ = (d − dim ker R)/2:

| cover | d | dim ker R | λ theory | λ̂ (WBIC) |
|---|---|---|---|---|
| full joint | 7 | 0 | 3.5 | 3.44 ± 0.06 |
| tree AB, BC | 7 | 2 | 2.5 | 2.21 ± 0.29 |
| triangle | 7 | 1 | 3.0 | 3.10 ± 0.14 |
| 4-cycle | 15 | 7 | 4.0 | 4.05 ± 0.56 |
| sphere | 15 | 1 | 7.0 | 6.31 ± 0.80 |

**Flip prediction.** Next-window flip regressed on the toggle radius r, curvature γ², holonomy z and a prolate step |z|. The gain of the three geometric features over r is tested by block permutation.

| stream | windows / flips | AUC r | γ² | KL | prolate | LR gain (p) |
|---|---|---|---|---|---|---|
| planted null ×10 | 380 / 120 | 0.60 | 0.48 | 0.54 | 0.56 | 2.5 (0.59) |
| planted ramp ×1 | 380 / 163 | 0.59 | 0.46 | 0.54 | 0.53 | 4.0 (0.54) |
| planted ramp ×10 | 380 / 124 | 0.73 | 0.51 | 0.61 | 0.48 | 3.0 (0.64) |
| planted step ×10 | 380 / 112 | 0.65 | 0.47 | 0.50 | 0.46 | 17.2 (0.04; KL coefficient negative) |
| AllSides | 95 / 41 | 0.64 | 0.47 | 0.64 | 0.48 | 3.5 (0.39) |
| 2016 polls | 176 / 51 | 0.52 | 0.41 | 0.53 | 0.35 | 0.5 (0.85) |

- The toggle radius carries what can be predicted.
- Holonomy predicts on its own, but only because it measures the same drift as r.
- Curvature never predicts flips.
- The hypothesis "small r, large γ², large KL, significant prolate ⇒ flip" is not supported.

## Dynamic programming on the gluing lattice (paper §sec:latticedp) and why no gradient descent (Remark rem:nogd)

Code: `amb_vigneaux/lattice_dp.py`; `examples/lattice_dp/run.py` (output `run_out.txt`, `results.json`, `lattice_dp_chart.pdf`); tests `tests/test_lattice_dp.py` (2; the partition function matches brute force to 1e-8).

**Method.**
- **Score.** The deviance of the additive model on a region, with closed-form jets in the inclusion indicators: D′ₓ = the topic's loop residual SS; D″ₓᵧ = −gₓᵀH⁺gᵧ.
- **Policy.** Gibbs π(U) ∝ exp(−β/2·[D̃(U) − 2·df(U)]).
- **Exact sampling.** Forward filtering and backward sampling over bags (components of the strong D″ graph, cap 8), with size and quota counts as DP state. Local rules (required topics, exclusions) are inside the bags.
- **Global rules** (glue p ≥ 0.05; stable lean of the target source, |lean − η₀|/se ≥ 2) are checked exactly on every sample.

**Planted benchmark** (S = 50; 15% contaminated in clusters):

| method | feasible X=50 / 200 / 400 | contamination excluded / s per feasible (X=200; X=400) |
|---|---|---|
| DP, 2nd-order jets | 0.14 / 1.00 / 1.00 | 0.98 / 0.02; 0.94 / 0.04 |
| DP, + target rule in state | 0.00 / 1.00 / 1.00 | 0.98 / 0.02; 0.94 / 0.06 |
| DP, 1st-order jets | 0.00 / 1.00 / 1.00 | 0.96 / 0.02; 0.94 / 0.04 |
| greedy with restarts | 0.79 / 1.00 / 1.00 | 0.84 / 0.06; 0.77 / 0.16 |
| random (counting rules) | 0 / 0 / 0 | — |

- **Jets.** R² of the predicted D(U) is 0.999–1.000. The second-order graph puts the planted clusters inside one bag (all for X ≤ 200; 16/22 at X = 400).
- **Temperature (X = 200).** Feasible share / distinct regions out of 200:
  - β = 1: 1.00 / 40
  - β = 0.3: 0.99 / 189
  - β = 0.1: 0.33 / 65
  - β = 0.03: 0 / 0
- **Small, tight instances (X = 50).** Glue and stability nearly conflict, and greedy with an exact check at every step wins (0.79). Carrying only one global rule in the state is not enough when two bind together.

**Gradient-descent check** (`examples/jets/gd_check.py`). Plain gradient ascent on the no-three-way model of the JetClass-II pT × |η| grid gives G = 1069.07, against IPF's 1069.15, and the same holonomy, 1.796e-3 nats/jet. The no-gradient policy is a design choice for three reasons, not an impossibility:
- exact fits, linear-programme bounds and certificates;
- posterior averaging on singular models;
- no plateaus at singular strata.

## Which context to buy next: identification-aware acquisition (paper §sec:acquire)

Code: `amb_vigneaux/acquire.py`; `examples/acquire/run.py` (outputs `run_out.txt`, `crossover_out.txt`, `results.json`, `acquire_chart.pdf`); tests `tests/test_acquire.py` (4).

**Prior work checked before building.**
- Track-and-Stop / optimal design solves fixed-confidence identification.
- Dominitz–Manski 2017 treats more versus better data under partial identification.
- Akbari et al. 2022 gives minimum-cost intervention design.
- CausalIDView (arXiv 2609.36881) is a benchmark of estimators across fixed observational views; it does not acquire views.

**Stopping rule.** Every policy uses the same anytime-valid rule: L1 balls (Weissman) on each sampled context, combined into robust LP bounds. Every decision taken was correct.

**Main benchmark** (12 truths per scenario; median cost, cap 60000; share stopped):

| scenario | bounds-first | c-opt | cheap-only | full-only |
|---|---|---|---|---|
| triangle, unanimity | 55906 (0.50) | 51707 (0.83) | 60002 (0.42) | **33100** (1.00) |
| triangle, all-ones, t outside | 27142 (0.83) | 13946 (1.00) | 50736 (0.50) | **8700** (1.00) |
| triangle, all-ones, t inside | cap (0.25) | cap (0.25) | cap (0) | **46550** (0.83) |
| 4-cycle, P(A=C) | **17000** (1.00) | **17002** (1.00) | cap (0) | cap (0) |
| 4-cycle, all-ones, t outside | 33100 (1.00) | 4040 (1.00) | 29800 (0.83) | **3268** (1.00) |
| 4-cycle, all-ones, t inside | cap | cap | cap | cap |

**Crossover** (triangle, all-ones; joint cost 4 / 16 / 64 × distance 0.03 / 0.1 / 0.2; 8 truths per cell).
- Bounds-first is cheapest or tied at distance 0.2, and at joint cost 16 with distance 0.1.
- Worst-case cost relative to the best policy in each cell:
  - bounds-first 1.6×
  - c-opt 4.1×
  - cheap-only 5.9×
  - full-only 24×

**Verdict.**
- The claim "beats optimal design" is false at these scales; buying the joint is often cheapest.
- The proposed policy is the minimax-regret default when the costs and the distance to the decision are unknown.

## A reward loop with all three layers (paper §sec:rewardloop)

Code: `amb_vigneaux/reward_loop.py`; `examples/reward_loop/run.py` (outputs `run_out.txt`, `results.json`, `reward_loop_chart.pdf`); tests `tests/test_reward_loop.py` (3).

**Setup.**
- **Stream.** S = 50 sources, X = 60 topics, 100 periods. Hidden Markov contamination per topic (q01 = 0.02, q10 = 0.10).
- **Geometry.** Evidence is the loop statistic (first-order jet). It is noncentral χ² with λ = the Fisher norm² of the loop; λ is estimated by EM.
- **Probability.** An HMM belief filter per topic.
- **Schemes.** Rules: the region glues, and the target lean is right.
- **Reward.** +1 per clean topic kept, −5 per looped topic kept, −κ per switch, −20 per violated rule.
- **Control.** Per-topic belief MDP decoupled by a price μ (the Lagrangian / Whittle relaxation). Value iteration and policy iteration give the same policy in every case (about 400 sweeps vs 3–4 steps, ~10 ms). Parameters are tuned on 5 training streams; results are on 10 test streams.

**Reward per period:**

| loop amp, κ | myopic test | myopic Bayes | value iteration | Bayes* | value iteration* | oracle |
|---|---|---|---|---|---|---|
| 0.6, 0.5 | 42.7 | 45.1 | **45.4** | 43.0 | 43.8 | 47.6 |
| 0.6, 2.0 | 37.1 | 41.8 | **42.1** | 39.5 | 40.4 | 44.6 |
| 0.35, 0.5 | 12.4 | 21.1 | 25.9 | 27.8 | **31.5** | 47.6 |
| 0.35, 2.0 | 1.2 | 16.1 | 22.2 | 22.6 | **28.2** | 44.6 |
| 0.25, 0.5 | 0.7 | 0.5 | 11.0 | 15.1 | **22.0** | 47.6 |
| 0.25, 2.0 | −4.7 | −0.4 | −1.3 | 8.6 | **18.0** | 44.6 |

(\* = planted noncentrality; all-in scores −28 to −5.)

**What each layer adds.**
- **Probability:** +2.4 to +15 per period.
- **Lookahead (value iteration):** about 0 with strong loops, +4.7 to +6.0 with medium loops, +7 to +9 with weak loops when λ is right.
- **Geometry:** at weak loops the EM estimate of λ collapses (0.5 vs 10.8), and that is the bottleneck. The fix is to estimate λ from pooled holonomy information (open).

## Grading regions by Blackwell order (paper §sec:blackwell; after Bohinen–Perrone, arXiv 2502.14941)

Code: `amb_vigneaux/blackwell.py`; `examples/blackwell/run.py` (outputs `run_out.txt`, `results.json`, `blackwell_chart.pdf`); tests `tests/test_blackwell.py` (4).

**Method.**
- Each region is an experiment Θ → X_U, graded by its standard measure (the law of the posterior).
- Blackwell order is the convex order of standard measures, checked exactly for binary Θ.
- The value of every convex reward is monotone in this order. So a cost–Blackwell frontier, computed once with no reward, contains every optimum.
- The frontier is found by one sweep in order of increasing cost, comparing each region only with current frontier members. This is exact because the order is transitive (Bohinen–Perrone Thm 3.1).

**Exact, reward-free pruning:**

| n | regions | frontier | dominance checks (all pairs) | time | optimum on frontier |
|---|---|---|---|---|---|
| 8 | 256 | 17 (6.6%) | 1460 (65280) | 0.1 s | 123/123 |
| 10 | 1024 | 23 (2.2%) | 8152 (1.05e6) | 1.0 s | 135/135 |
| 12 | 4096 | 28 (0.7%) | 39728 (1.7e7) | 7.6 s | 147/147 |

**Paths invisible at the pairwise level** (budget 4):

| method | region | accuracy |
|---|---|---|
| greedy single-variable MI | {X8, X5} | 0.75 |
| myopic greedy | {X8, X5} | 0.75 |
| summed pairwise MI | {X1, X5, X6, X7} | 0.65 |
| Blackwell frontier | {X1..X4} (parity) | 0.90 |

The parity region's value along its growth path is 0.5, 0.5, 0.5, 0.5, then 0.9.

**Across an obstruction** (contradictory triangle, strength λ).
- λ ≤ 0.3: the honest value is an interval, [0.50, 1.00] at λ = 0 down to [0.65, 0.675] at λ = 0.3.
- λ ≥ 0.4: the family is contextual and the region has no value.
- The naive max-ent glue reports values rising to 0.95 regardless.

**Open.**
- Growing only frontier regions is not exact: Blackwell order is not preserved when the same measurement is added to both sides.
- Joining the frontier with the factored lattice DP.
- Non-binary Θ.

## Exact frontier growth, non-binary targets (paper §sec:blackwell-growth)

Code: `amb_vigneaux/blackwell.py` (`grow_frontier`, `leq_k` Strassen LP, `product`); `examples/blackwell/growth.py` (outputs `growth_out.txt`, `growth_results.json`, `growth_chart.pdf`); tests in `tests/test_blackwell.py` (7 in total).

- **Exactness.** Frontier growth is exact over blocks that are conditionally independent given Θ: products preserve Blackwell order, and the order is transitive. Against full enumeration at n = 12, the optimum was identical in 114/114 cases for k = 2 and 114/114 for k = 3.
- **Scaling.** n = 7 / 14 / 21 / 28 / 35:
  - combinations formed: 40 / 207 / 879 / 1755 / 4910 (against 2^n regions);
  - frontier: 17–296;
  - time: 0.0–142 s.
- **Best accuracy at budget = 25% of total cost.** Exact frontier 0.90–0.97 vs myopic greedy 0.79–0.93.
- **Counterexample.** N noise, B weak signal, C = Θ ⊕ N. {N} ≤ {B}, but {N, C} (accuracy 1.0) is not ≤ {B, C} (0.6). With wrong blocks growth finds 0.60; with N and C in one block it finds 1.00.

## CausalIDView partial-identification benchmark (paper §sec:causalidview)

Code: `examples/causalidview/run.py` (needs a clone of github.com/MLAI-Yonsei/CausalIDView at `CAUSALIDVIEW_DIR`); outputs `run_out.txt`, `results.json`, `causalidview_chart.pdf`.

**Setup.**
- The companion SCM is rebuilt from their Algorithm 2: their released effect families, nuisances and seeds; A_b = 0.2, A_τ = 0.4, c_U = 0.8.
- 40 worlds, 1024 context / 100 query units, exact oracle bounds (Manski formula; Balke–Pearl monotone-IV LP).

| backbone | Manski RMSE | IV RMSE | containment M/IV | IV cells infeasible | error infeasible/feasible | s/world backbone/bounds |
|---|---|---|---|---|---|---|
| logistic (published) | ≈0.135 | ≈0.175 | | | | |
| XGBoost (published) | ≈0.185 | ≈0.23 | | | | |
| TabPFN-v3.5 (published) | **≈0.09** | **≈0.12** | | | | |
| logistic (rebuilt) | 0.132±0.013 | 0.170±0.017 | 0.99/0.89 | 1% | 0.226/0.154 | 0.03/0.50 |
| XGBoost (rebuilt) | 0.166±0.012 | 0.209±0.015 | 0.97/0.87 | 17% | 0.247/0.200 | 1.61/0.50 |
| IRLS logistic (ours) | 0.132±0.013 | 0.169±0.017 | 0.99/0.90 | 1% | 0.210/0.153 | 0.04/0.48 |
| **structured, gradient-free (ours)** | **0.105±0.012** | **0.138±0.016** | 1.00/0.92 | 0.03% (1 unit) | 0.427/0.130 | 0.25/0.48 |

**Structured backbone (`estimate_ours`).** It estimates what each bound functional needs:
- **Manski:** L = s − 1, U = s with s = P(Y = T | X). Linear-kernel ridge on 1[Y = T], with the ridge chosen by the closed-form leave-one-out error.
- **IV:** P(T | X, I)·P(Y | T, X, I), with each factor a ridge logistic fitted by Newton steps and the ridge chosen by 5-fold CV. The intercept and the instrument coefficient are unpenalised: penalising the instrument widened the bounds (bias −0.036/+0.024 → −0.015/−0.024; IV 0.143 → 0.138 paired over 32 worlds).

There are no gradient steps.

**Findings.**
- **Accuracy.** TabPFN is still most accurate (≈0.09 / ≈0.12).
  - The structured backbone closes 64% of the Manski gap and 63% of the IV gap from logistic regression (0.132/0.169 → 0.105/0.138).
  - It beats logistic regression in 40/40 worlds on both views.
  - By family: linear 0.090/0.119, quadratic 0.103/0.140, threshold 0.104/0.134, Fourier 0.124/0.161, shallow MLP 0.103/0.135. TabPFN's per-family values are not published, so we claim no parity.
- **Where the rest of the gap is.**
  - The constant predictor gives 0.131. Every gradient-free function class we tried plateaus at ≈0.105: RBF/poly kernels with LOO selection, ARD, stacking, latent-U EM, estimated indices, pHd, bagging and bootstrap bias correction. See `examples/causalidview/ablations/README.md`.
  - Oracle effect directions give 0.04. The remaining gap is learning 2–4 sparse directions from 1024 binary labels, which is where a pretrained prior helps.
- **Feasibility flag.** Their pipeline silently projects infeasible IV cells. Our flag marks 1–17% of units for the generic backbones, which have 1.2–1.5× larger endpoint error. For the structured backbone it marks 1 unit in 4000.
- **Coverage.** 90% context-bootstrap intervals cover the oracle endpoints only 81–82% of the time (IRLS) and 70–71% (structured). The more regularised, more accurate backbone covers worse, because bias dominates.
- **Compute.** Per world on CPU: structured backbone 0.25 s, bounds 0.48 s, XGBoost 1.55 s.

### Support-guided priors: posterior → next prior (paper §sec:supportprior)

Code: `examples/causalidview/support/` (`oracle_support.py`, `oracle_subspace.py`, `subspace_prior.py`, `crossfit_prior.py`, `eb_prior.py`, `eb_hyper.py`, `plot.py`). Manski RMSE over the first 10 worlds; base 0.106.

| variant | Manski RMSE |
|---|---|
| constant | 0.131 |
| linear ridge, LOO (backbone) | 0.106 |
| + TRUE coordinate support (30–49 of 50 coordinates) | 0.1015 |
| + TRUE subspace (5–13 directions), linear | **0.078** |
| + TRUE subspace, RBF | 0.072 |
| plug-in learned subspace, pooled / sequential chain | 0.114 / 0.116 |
| half data without prior / cross-fitted prior (A→B, B→A) | 0.111 / 0.109 |
| empirical Bayes, shared Σ (rank 0 reproduces the base) | 0.126 |
| EB + inverse-Wishart, best ν (6 worlds; base 0.1046) | 0.1048 |

- **Coherence.** A posterior may be the prior only for disjoint data. With the same data, LOO picks the largest prior weight every time: double counting.
- **Reading.** A known subspace support would beat TabPFN even with linear fits. One world's labels recover only about 3 directions, at cosine 0.6–0.85, and every way of learning the support there is no better than no support.

**Does the law of supports (rank, sparsity, signal) help?** (`law_prepass.py`, `law_oracle.py`)

| | rank | sparsity | top-3 cosine | Manski RMSE |
|---|---|---|---|---|
| truth | 6.6 | 12 | | |
| cheap pre-pass (permutation-null SVD, ARD on the T fit) | 2.5 | 4.5 | 0.69 (dense 0.68) | 0.108 |
| TRUE law handed to the estimator | 6.6 | 12 | 0.70 | 0.110 |
| base | | | | 0.106 |

- **The pre-pass undercounts.** This is consistent with the spiked-model detection limit (Baik–Ben Arous–Péché), but we have not shown the pre-pass operates in that regime, so the low recovery is stated, not attributed.
- **Even the exact law adds nothing.** This corrects the earlier hypothesis that a cross-world meta-prior of the law would close the gap. The missing information is the directions themselves. What TabPFN exploits is not identified here.

**Treatment beacon with known structure** (`beacon.py`, `beacon_oracle.py`). The model is the varying-coefficient s = a + x·β1 + (2ê − 1)(c + x·β2), with ê from the strong treatment fit; weight 0 recovers the backbone.

| propensity used | Manski RMSE (base 0.1060) |
|---|---|
| estimated ê | 0.1066 |
| TRUE propensity | 0.1063 |

The treatment direction is already well estimated. The directions that multiply it (b, τ) are seen only through the weak outcome labels. Pooling across worlds doesn't apply, since each world draws its own directions.

**Shape of the conditional, fitted exactly** (`examples/causalidview/shape/`; 40 worlds; linear link 0.1046). The link is fitted on leave-one-out indices, so nothing is counted twice.

| link on the backbone's index | Manski RMSE |
|---|---|
| isotonic, PAVA (exact) | 0.1113 |
| CV blend of PAVA and linear (weight 0.07) | 0.1047 |
| Platt, 2 parameters | 0.1045 |
| cubic logit, 4 parameters | 0.1052 |
| oracle affine (ceiling) | 0.1012 |
| oracle monotone, in-sample | 0.0927 |
| oracle monotone, held-out half | **0.1057** |

The in-sample 0.093 is optimism; there is no shape headroom along our estimated index. Shape matters only along the true directions (linear 0.078, true shape 0.04).

**Isotonic conditional law on a partial order** (Arnold & Ziegel 2025; for a binary outcome it is IDR / partial-order isotonic regression). `icl_partial_order.py`, 40 worlds.
- **Order:** the product order on (m̂1, −m̂0), justified by s = e·m1 + (1 − e)(1 − m0).
- **Indices:** out-of-sample for every training unit.
- **Fit:** exact grid projection by Dykstra alternation of PAVA.
- **Result:** ICL 0.1128; CV blend 0.1048; linear 0.1046. The order is right, but its coordinates are learned from about 512 weak labels each.
- **Region-inclusion order:** it runs the wrong way for rule satisfaction (more regions mean more constraints). For decision value it is Blackwell monotonicity, already used exactly by the frontier search.

### The prior one level up: posterior under a generic prior, per task, no gradients (paper §sec:bayesprior)

Code: `examples/causalidview/bayes/` (`ceiling.py`, `generic.py`, `pool.py`; chains saved in `chains/`, pooled numbers in `pooled.json`). TabPFN is an amortised posterior under a prior over tasks. Amortisation buys speed, not accuracy, so we compute the posterior for each task directly by random-walk Metropolis.

- **Ceiling** (the generator's own prior family; 8 non-MLP worlds, 2 chains): structured 0.1076 → **0.0915** Manski. At the true parameters the model reproduces the oracle to 0.0007.
- **Generic prior** (not the generator's form):
  - P(T|x,I) = σ(α0 + αI·I + γ·wT·x);
  - P(Y|T=t,x,I) = σ(c_t + c_It·I + Σ_{k≤3} spline(w_k·x));
  - 12-sparse random unit directions, N(0,1) coefficients, no latent confounder;
  - all fixed once before any run, and not tuned on the oracle.

| 40 worlds | Manski | IV |
|---|---|---|
| logistic (rebuilt) | 0.132 | 0.169 |
| structured backbone | 0.105 | 0.138 |
| **generic-prior posterior, 16 chains** | **0.0934** | **0.1226** |
| TabPFN-v3.5 (published) | ≈0.09 | ≈0.12 |

- Chains pooled 1 / 2 / 4 / 8 / 16: Manski 0.1005 / 0.0973 / 0.0953 / 0.0937 / 0.0934; IV 0.1319 / 0.1277 / 0.1250 / 0.1230 / 0.1226. The curve plateaus by 8–16 chains: the posterior mean is reached, and the remaining ≈0.003 against the published values is set by the prior, within the reading error of the bars. The earlier extrapolation that 16–32 chains would close it was wrong.
- **Short-lived, not long-tailed** (`chain_structure.py`).
  - Pooled err²(k) = a + b/k with R² = 1.0000; a free exponent fits at 0.99.
  - No outlier chains (none beyond 3× the median deviation); excess kurtosis 0.76.
  - At k = 8 the b/k term is about 2% of the floor.
  - The plateau and the 1/k law show fast convergence and rule out this long-tail behaviour, but on their own do not rule out shared multimodal structure. Under the tested clustering criterion (single linkage at half the median distance) there is no evidence of discrete chain-separated modes: every chain is its own cluster (15.9/16). The most plausible reading is one diffuse cloud, with the floor set by the prior's bias.
- It beats the structured backbone in 31/40 worlds (Manski) and 33/40 (IV). IV cells infeasible: 0.
- By family (Manski, structured → Bayes): linear 0.090 → 0.070; Fourier 0.124 → 0.101; threshold 0.104 → 0.093; quadratic 0.103 → 0.099; shallow MLP 0.103 → 0.104.
- **Why it works:** plug-in estimates of the directions fail at the detection limit; integrating over them under a prior does not need them to be identified.
- **Cost:** about 230 s of one CPU core per world for 4 chains, with no training.
- **Not run.** TabPFN itself (checkpoints unreachable from here); its values are read off their Fig. 6c.


## Structured posteriors: factors, evidence and subspaces through the monad (paper §sec:sp)

Code: `amb_vigneaux/sp.py`; exact tests `tests/test_sp.py`; stress test `examples/sp/multimodal.py`; chains as branches `examples/causalidview/bayes/stacking_mlp.py`.

- **Monoids:** factor set (∪), factor multiset (+), evidence trace (concatenation), subspaces of Qⁿ (span). Geometry is derived from μ, not carried.
- **Per-branch G(X×M):** a monad for every monoid. All laws hold exactly on 30 random rational instances per monoid.
- **Global G(X)×M** (one record, aggregated over the support):
  - the identity laws hold for every monoid;
  - associativity holds for semilattices (factor set, subspace) and fails for multisets and traces;
  - minimal counterexample: a record reachable along two paths is counted once on one side and twice on the other.
- **Floating point:** agrees with the exact version to 1e-12.
- **Stress test** (A: chain z→x1→x2 with span(e1); B: common cause x1←w→x2 with span(e2); mixed by multiplication; evidence x2 = 1):
  - Giry keeps only the mixture, exactly equal to an unstructured C;
  - the global record is the union, exactly equal to a single posterior D with both structures, so "A or B" becomes "A and B";
  - per-branch recovers μ_A and μ_B exactly, with I(X; record) = 0.043 nats and P(A|e) = 25/41 (exact Bayes, BF 25/16).
- **Chains as branches** (record = held-out log score; 80/20 split; 8 chains). Δ Manski RMSE vs equal pooling, ± s.e.:

| | MLP worlds (8) | control worlds (8) |
|---|---|---|
| stacking | +0.0039 ± 0.0016 (worse 7/8) | −0.0018 ± 0.0018 |
| pseudo-BMA | +0.0027 ± 0.0013 | −0.0025 ± 0.0014 (better 6/8) |
| best chain | +0.0055 ± 0.0018 | +0.0002 |

Evidence weights concentrate on one or two chains (largest weight ≈ 0.67). The selection is noisy, and it loses the averaging that cancels chain variance (Var of the mean ≈ Var/N). The supported statement is narrow: on these worlds, evidence-weighted chain aggregation did not improve on equal pooling, and often did worse. Preserving structure is not automatically better prediction; it pays where branches are distinct hypotheses, not across interchangeable chains.


## Persistent posteriors (paper §sec:persist)

`examples/causalidview/bayes/persist.py`; chart `persist_chart.pdf`. The stream is 512 context units, then 8 × 64, up to 1024; 5 worlds, one per family.

| at n = 1024 | Manski | work per update | seconds per update |
|---|---|---|---|
| rerun (4 × 3000 iterations) | 0.0996 | 1.2e7 | 299 |
| persistent, 300 iterations per batch | 0.1052 (+0.0056 ± 0.0033) | 9.6e5 (13× less) | 25.5 |
| persistent, 100 iterations per batch | 0.1082 (+0.0085 ± 0.0025) | 3.2e5 (38× less) | 8.5 |
| stale (n = 512) | 0.1246 | 0 | 0 |

- Persistence keeps about 78% / 66% of a rerun's gain from the new data. All 8 updates together cost less than one final rerun.
- **Lag:** at n = 768 the persistent chains are at 0.120 against 0.112 for a rerun, and the first two updates are slightly worse than stale.
- **Exact regime:** the ridge sufficient statistics (Σxxᵀ, Σxy, Σy, n) update exactly (max difference 1.5e-16). The update is O(batch·p²); at p = 50 the wall gain is only 1.5×.
- **Reading:** the retained state is an exact sufficient state when the posterior has an additive sufficient statistic; otherwise it is an approximate one, an order of magnitude cheaper per update with a measured lag (SMC resample-move is the principled fix).

## Shadow algebra and context comonad (paper §sec:shadow)

`amb_vigneaux/shadow.py`; exact tests `tests/test_shadow.py`.

- ⊕ (alternatives, shadows not merged) is commutative, idempotent, and associative with induced weights.
- ⊗ (joint factors: pointwise product, shadows joined, weight = overlap Z) is commutative and associative.
- **Distributivity:** (a ⊕_λ b) ⊗ L = (a ⊗ L) ⊕_λ' (b ⊗ L) holds only for λ' = λZ_a / (λZ_a + (1 − λ)Z_b), i.e. Bayes. It fails with fixed λ, so the structure is a semiring only up to Bayesian reweighting.
- Forgetting the shadow is a homomorphism onto the plain mixture. A ⊕ B stays distinct from a single posterior with the same mixture, and conditioning gives exact hypothesis weights (3/4, 1/4).
- A shadowed posterior is an element of G(G(X)×Σ): the per-branch construction one level up.
- **Context comonad** C(X) = (x, ρ, S), ε = x, δ redraws the context: counit and coassociativity hold exactly, and C is functorial. A distributive law GC ⇒ CG is open.


## Three-level posterior tower: independent checks (paper §sec:shadow, citing the tower paper)

`examples/tower/check_tower_independent.py` re-checks in exact arithmetic, from the tower paper's definitions:
- Theorem 6.1, the context comonad C_E(X) = X × E(X), with the non-trivial E = D∘D: comonad laws and naturality of δ.
- Theorem 9.8, the base-relative law λ^B: (U), (Co), (M), (Cm), and closure of coherent contexts under the barycenter.

Chain-pooling curve standard errors (40 worlds):
- ±0.0025 Manski and ±0.0033 IV per point.
- 8 vs 16 chains, paired: +0.00034 ± 0.00022 (Manski), +0.00043 ± 0.00029 (IV), not significant.
- 1 vs 16 chains: +0.0071 ± 0.0011.


## Open problem 6 of the tower paper: do obstruction levels predict errors? (`examples/tower/problem6.py`)

**Level two, synthetic.** H has 6 laws on 4 outcomes, the truth is in H, n1 = 20, n2 = 50 future observations.

| claim | d1 | d2 (exact) | TV error now | TV error after updating |
|---|---|---|---|---|
| Bayes over H | 0 | 0 | 0.039 | 0.006 |
| collapsed (barycenter asserted as one posterior) | 0 | 2.0 | 0.039 | 0.039 |
| partially collapsed | 0 | 1.04 | 0.039 | 0.014 |
| hallucinated atoms | 0.10 | 2.0 | 0.164 | 0.139 |
| empirical frequency | 0.11 | 2.0 | 0.121 | 0.121 |

- AUC for the top quartile of post-update error: graded transport d2W 0.96, d1 0.81, exact d2 0.75.
- Among level-one-admissible claims (d1 = 0), d2W gives 0.94.
- Pooling hides level-one obstructions but not level-two ones (the tower paper's Prop. 10.10), now made quantitative. The collapsed claim's failure is built into its construction; what is measured is that the graded score ranks errors.

**Level one, CausalIDView** (admissible set = image of the monotone-IV model; d1 = our LP slack).

| backbone | share of units flagged | mean error, flagged vs not | AUC, top error quartile |
|---|---|---|---|
| logistic | 0.6% | 0.22 vs 0.15 | 0.50 |
| XGBoost | 17% | 0.25 vs 0.20 | 0.54 |
| structured | 0.03% | — | 0.50 |

Level one catches structural violations, not ordinary inaccuracy. A real-data level-two test, with second-order claims scored after updating, is open.


## Context attribution of generated tokens (tower paper §5.4; three_layers §ctxattr)

The question is whether a span came from context A, from context B, from a blend (with a 95% band on the share), from switching, or from neither (unknown).

- **Projection.** EM maximises sum_t log(w_A a_t + w_B b_t + w_U u_t) over the simplex. It reports:
  - theta = w_A/(w_A+w_B), with a profile band;
  - mu = w_U, with a one-sided lower bound.
- **Calibration.** Bands use a block design effect with block length ceil(sqrt n) and a boundary chi-square correction.
- **Switching versus blend.** A two-state HMM likelihood ratio separates them; its threshold is simulated.
- **Unknown budget.** It is calibrated on known-context text.

**Setup.** Contexts are BASIL articles: different events (easy) or the same event (hard). The reference is a bigram model on the other articles plus 40k BABE neutral headlines. There are 40 generations per case.

| setting | copy A | copy B | switch | blend | unknown | half unk. | theta-band cover | false unk. |
|---|---|---|---|---|---|---|---|---|
| n=50, easy | 0.78 | 0.78 | 0.85 | 0.88 | 1.00 | 0.90 | 0.85–1.00 | 0.05 |
| n=50, hard | 0.50 | 0.53 | 0.70 | 0.87 | 1.00 | 1.00 | 0.93–0.97 | 0.09 |
| n=100, easy | 0.93 | 0.97 | 0.90 | 0.91 | 1.00 | 1.00 | 0.88–0.97 | 0.07 |
| n=100, hard | 0.85 | 0.82 | 0.93 | 0.86 | 1.00 | 1.00 | 0.95–0.97 | 0.05 |
| n=200, easy | 0.97 | 0.93 | 0.95 | 0.92 | 1.00 | 1.00 | 0.97–1.00 | 0.03 |
| n=200, hard | 1.00 | 0.95 | 0.95 | 0.95 | 1.00 | 1.00 | 0.95–1.00 | 0.01 |
| n=100, full models, easy | 0.97 | 0.97 | 1.00 | 0.97 | 1.00 | 0.00 | 0.90–1.00 | 0.00 |
| n=100, full models, hard | 0.88 | 0.88 | 1.00 | 0.89 | 1.00 | 0.03 | 0.93–0.97 | 0.00 |

**Absorbed reference.** "Full models" uses the context models (0.5 doc + 0.5 reference) as coordinates. In that case half-outside text is not flagged: mu_hat = 0.08, against 0.53 in the document coordinates. This is proved, as max(0, (f-(1-beta))/beta), and checked in tests/test_attribution.py.

**Limitations.**
- The generations come from bigram models and copied spans, not an LLM.
- At n = 50, about half of the same-event copies are reported as blends. The band contains the true share in 27 of 29 of those cases.

## Obstructed claims cannot learn: theorems (three_layers §claimlearn)

Checks are in examples/tower/claims_theorems.py (output in claims_theorems.txt) and tests/test_claims_theorems.py.

**Proved:**
- **Lemma 1:** d2W is the Kantorovich distance to D(S1). It matches the transport LP to 1e-16 in 400 instances.
- **Theorem 2 / Corollary 3:**
  - limsup TV(p_n, rho*) ≤ r / sqrt(2 sigma_min), uniformly over the truth.
  - With a unique KL-minimiser, lim TV(p_n, rho*) ≥ s/2.
  - No violations of the bound chain.
  - Simulated p_n at n = 4000 is within 0.02 of the limit in 1938 of 1940 cases (the 2 exceptions are small KL gaps).
- **Theorem 4:** an explicit finite-n bound (Hoeffding plus a union bound). It is never violated, but it is loose: below 1 in only 10% of instances at n = 300.
- **Prop 5a:** E_pi lim TV ≤ r_pi / sqrt(2 sigma_min) ≤ d2W / sqrt(2 sigma_min) for perturbed honest claims. No violations in 160 instances.

**Corrections to the earlier list:**
- **Prop 5b:** d2W alone cannot bound the error. A claim delta at rho_{h1} with h1 ≠ h* has d2W = 0 and permanent error. So Conjecture 5 needs a coverage term.
- **Ties:** with two KL-minimisers symmetric about rho*, p_n equals rho* infinitely often (88 zero returns in 10k steps), so liminf TV = 0 < s/2. The lower bound needs a unique minimiser; for two tied minimisers it holds for the limsup.

**Still open:**
- a finite-n d2W bound for perturbed honest claims;
- the level-three analogue (Problem 6).
