# Pre-analysis plan, phase 5: the context-to-outcome map as a toric variety, and geometry at the boundary

**Fixed on 2026-10-11.** No quantity below has been computed.

What was already known:
- the phase 4 results, in particular that 98.8% of per-stay ICU day systems have a zero margin and half have a point polytope;
- the GSS answer distributions by wording (Layer 1, holonomy);
- the ICU unit-type regimes.

Seed: 20261011.

## Part A. Is the map from information to outcome the same in every context?
Take three variables:
- **C**, a context;
- **A**, the information: the same content, measured in each context;
- **O**, an outcome.

**Context invariance** means P(O | A, C) = P(O | A): the context changes who carries which information, not what the information means for the outcome. Algebraically this is the conditional-independence model O ⟂ C | A. It is a **toric variety** in the simplex of C × A × O tables, cut out by the binomials

    p_{c a o} p_{c' a o'} − p_{c a o'} p_{c' a o}

The distance from the data to it, the KL divergence of the information projection, is the conditional mutual information I(O; C | A).

**Systems.**
1. **GSS (primary; the context is randomized).**
   - C: the wording (X, Y). A: the answer to one spending item (too little, about right, too much). O: party identification (Democrat 0–2, Independent 3, Republican 4–6; "other party" excluded).
   - Eleven items, every year 1985–2024 in which both wordings were asked, unweighted counts (wording is randomized within a year).
2. **ICU (PhysioNet 2012).**
   - C: unit type (coronary, cardiac surgery, medical, surgical; not randomized). A: first-window pressure band H1. O: in-hospital death.
   - Set A primary, set B replication.

**Tests.**
- Per table, G² of the invariance model, with df = (|C|−1)(|O|−1)|A|. A parametric bootstrap (500 draws) is used when any expected count is below 5.
- GSS:
  - per item, the G² summed over years (df summed);
  - Bonferroni over the 11 items, at 0.05/11;
  - all items together, by Fisher's method over items.
- ICU: one table per set.

**Decision.** Invariance is *rejected* for a system at the stated level. The size of the departure is reported in bits, I(O; C | A), and it is localized by answer level: the binomial generators of answer a, measured as the log odds-ratio departure for each pair of contexts.

**Expected (stated now).** In the GSS, invariance is rejected for at least one item. The wording effects on answers are large, so whether they reach the outcome map is open. In the ICU, unit types differ in case mix, so rejection there would be expected; that analysis is descriptive in that sense.

**Toric decomposition.** In each context, the (A, O) coupling has margins a_c and o_c. Its polytope has a toric type:
- GSS: 3 × 3, so 9, 12, 15 or 18 vertices;
- ICU: 3 × 2, so a toric Fano surface (Bridges Prop. 8.1).

Under invariance with the pooled kernel K̂ = P̂(O | A), context c would have outcome margin o_c^inv = a_c K̂. We report:
- (i) how often the two contexts' systems differ in toric type;
- (ii) how often the invariant map reproduces each context's observed type, i.e. how often a type difference is explained by the information margins alone (direct influence);
- (iii) where it is not, the context changes the geometry through the outcome map.

Types are also reported with their chambers decided or not on the bands.

## Part B. Geometry at the boundary: are point polygons a dead end?
Per-stay ICU day systems (MAP band × HR band over the hours) usually have a zero margin. Re-test the phase 4 WP1a forecast of death with **boundary-aware** geometric features instead of point-estimate ones:
- **Posterior toric type.** Margins get a Dirichlet(counts + ½) posterior, with 200 draws per stay and day. Features: the posterior probability of each full-support toric type (18, 15, 12 vertices) and the posterior mean log steepness. A zero count is sampling, not a zero probability.
- **Decided walls (family over the band, Bands Prop. 6.1).** The number of the 18 walls whose sign is decided on the margin bands, and the number of decided chamber flips between day 1 and day 2. The family over a band is a full-dimensional polytope, not a point.
- **Face stratum.** The support pattern of the margins: which pressure and rate bands are empty. It is the torus orbit of the toric variety in which a broken system lies, coded as the number of empty pressure bands and of empty rate bands on each day.

**Models and decision.** As in WP1a:
- B2 is the flexible margin baseline;
- S2b = B2 + the boundary-aware features;
- fit on set A, test on set B;
- the gain must have a paired-bootstrap 95% interval above 0.

The phase 4 point-estimate result (no gain) is the comparison.

## Order
1. Planted checks:
   - Part A: invariance holds in a planted table and is rejected when a planted context changes P(O | A).
   - Part B: a planted dependence of death on the posterior toric type of broken systems is recovered.
2. One run of each part, frozen with SHA-256 hashes.
3. Write-up.
