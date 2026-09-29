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
