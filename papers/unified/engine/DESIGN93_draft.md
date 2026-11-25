# Study 93 DESIGN — DRAFT, NOT REGISTERED (v2, 2026-11-24; v1 same day)
# Structural transfer: what an earlier contingency model passes on to a different region (same table size) or to a finer table
# (larger size), decided by three typed operators Restrict / Extend / Prune before any expensive geometry.
# Registered separately from Study 92 so the hypotheses do not mix. Tentative prefix XT (grep 2026-11-24: unused).
# Synthetic data only. Budget target: < 2 CPU-hours. Becomes PREREG93.md after calibration on separate seeds.

## Principle
Transfer structure, not conclusions. Every old structure gets one verdict, each justified by a stated condition, never by a score:
  HONOR      the registered invariance condition holds            -> retain
  RECOMPUTE  same type, different numerical realisation           -> recalculate (e.g. wall locations, the Frechet band)
  PRUNE      a necessary condition fails                          -> discard
  NEW        no counterpart in the old model                      -> study separately
A structural verdict says nothing causal about seeding; the effect itself is a statistical question (O5).

## The base / fibre split (why most of the regional difference is invisible to W)
W uses the margins only (a_ij = exp(-L f_i g_j)). The seeding effect (odds ratios; early vs late seeding) lives in the credal fibre,
i.e. which table in the transportation polytope is the true one. So: margin geometry and the Frechet decision band can transfer
between regions with similar climatology; the effect cannot, and is tested classically (O5).

## Operators
 Signature(T): size r x c and category meanings; which margins are conditioned on; support pattern (structural zeros = a face of the
   polytope); Markov basis (all 2x2 moves for two-way tables: depends on size and support only); smooth/singular (gcd rule); number of
   critical points (normalised volume: 6, 18, 96).
 Restrict(X'): TWO different restrictions, never conflated:
   (a) face restriction: a row/column margin -> 0 (or a structural zero); the small polytope is a face, W restricted to the face is the
       small mirror; geometrically a degeneration limit (critical points split into clusters, the rest escape; cf. Study 79);
   (b) aggregation: merge categories; exact and linear on margins, NOT linear on the mirror coefficients (exp(-L f g) not additive) and
       not collapsible for odds ratios in general (Simpson).
 Extend(X, b): typed: records which margins and supports are preserved and which coordinates are new. Old walls extend to hypersurfaces;
   new walls are those involving the new row/column.
 Prune(X, X'): applies O1-O6 and returns the verdict table.
   O1 ties (exact)            symmetric collisions sit on margin ties                                   HONOR the tie structure
   O2 recurrence              events recurring at a second L / window are structural, the rest artefacts PRUNE non-recurring
   O3 local type              a small loop: sigma^3 (caustic) or pure sigma^2 (collision)                 HONOR the type
   O4 crossing sequence       order of wall crossings, timing-free; same sequence = same trajectory shape HONOR or PRUNE the path model
   O5 odds-ratio homogeneity  Breslow-Day / region x seeding x rain interaction                          HONOR or PRUNE the effect
   O6 old/new wall loops      at an old wall meeting a new wall: commuting loops -> old structure passes
                              unchanged; braid relation -> coupled                                      HONOR or RECOMPUTE locally
 Note (correction to the feedback): the LOCAL braid at an event is fixed by its type (sigma^{+-3}, sigma^{+-2}); a region cannot trace
 "sigma^-1" around a caustic. What is region-specific is the WORD along the path (the sequence of events and half-twists).

## Experiment A (same size, different regimes; 2x3, also 3x4 strands)
 Two synthetic regions sharing a climatology generator, region 2 = region 1 with (i) a time shift of k weeks and (ii) different interior
 odds ratios (early vs late seeding), plus a control pair with a different climatology. Registered predictions:
  XT-A1 Signature identical; Markov basis and table-size facts HONOR.
  XT-A2 O4: shifted pair -> same crossing sequence (timing removed); control pair -> different sequence.
  XT-A3 O2: off-tie collisions of region 1 do not recur in region 2 at matching margins beyond a registered fraction; ties do.
  XT-A4 O5 detects the odds-ratio difference at the registered sample size; the Frechet band (margins only) is RECOMPUTE, the rule HONOR.

## Experiment B (same region, finer table; 2x3 inside 3x4)
 CORRECTION to v1: sending the new row's MARGIN to 0 does not degenerate W in our parametrisation (a_3j = exp(-L f_3 g_j) -> 1, not 0),
 so v1's "6 converge, 90 escape" was unfounded. The correct one-input degeneration sends the new row's COEFFICIENTS to 0 along a registered
 schedule a_3j = delta^{w_j} c_j, delta -> 0 (a regular subdivision of the 3x4 exponent polytope induced by the heights).
  XT-B1 (exact, no tracking) the cluster counts are the normalised volumes of the cells of the induced subdivision (Viro / BKK), computed
        by exact LP; tracked only as a check on a few delta values.
  XT-B2 inherited walls: 2x3 ties and caustics continue as walls of the 3x4 family for small delta (located within a registered tolerance).
  XT-B3 O6 at old/new wall intersections: commuting vs braid; RECORD the fraction of old structure that passes unchanged.
  XT-B4 Restrict (b) vs (a): merging categories of the 3x4 table back to 2x3 gives the same Frechet band as the face route only when the
        registered collapsibility condition holds (exact LP).

## Experiment C (the first multi-input operation: stacking two regions) — NEW in v2
 E(X1, X2) = the 4x3 table with rows (region 1 seeded, region 1 unseeded, region 2 seeded, region 2 unseeded); column margins add
 (the a.x + b of the proposal). Its exponent configuration is (to be checked) the Cayley configuration of the two 2x3 blocks.
  XT-C1 (exact) Cayley trick: degenerating the cross-block coefficients clusters the critical points into the two blocks' (6 each, when each
        block is Newton-nondegenerate) plus mixed cells whose total is a mixed volume (Bernstein: an intersection number of toric
        divisors); compare with the normalised volume of the 4x3 polytope. This is Theorem C of the proposal at the count level.
  XT-C2 (2x3 blocks only, curve level) the blocks' collision and caustic pairs keep their intersection data (prop:disjoint, A2 model);
        Stokes-related pairs across blocks are computed.
  XT-C3 RECORD: Frechet bands of the stacked table vs the two regional bands (statistical intersection, exact LP).

## Experiment D (a ring of sources: lift decoding) — NEW in v3, 2026-11-24/25
 A single contingency table (and a stacked table) is one context: never contextual, CF = 0 (paper: "Lift decoding, and where it applies").
 Contextuality needs several sources reporting different, overlapping variables. Synthetic ring: source A reports (seeding, rain), B
 (rain, runoff), C (runoff, seeding); a fourth source D enters with (seeding, runoff) or leaves.
  XT-D1 (exact) CF of the ring = minimal lift defect with one copy (prop:lift); with D added, one copy per independent cycle
        (prop:liftgraph, cycle rank 2).
  XT-D2 RECORD how CF changes under Extend (a source joins), Restrict (a source leaves) and when a masked source is introduced;
        whether a decision band computed from the glued part (mass 1 - CF) still identifies the decision.

## Operad (stated carefully)
 Not claimed. Single-input extensions (face inclusions) compose trivially (a face of a face is a face), so a chain 2x3 -> 3x4 -> 4x5 is a
 category, not an operad. Stacking (Experiment C) is the first genuine multi-input operation; an operad requires its composition to be
 associative up to the registered equivalence and compatible with the mirror (degeneration / Cayley). Coloured by table shape; Restrict
 and Prune are partial inverses, not compositions. Open.

## Certificates used (theorems, paper Sec. 23.11 part 4)
 prop:disjoint (collision pairs orthogonal, any dimension), prop:localbraid (sigma^{2k}, sigma^3; signs forced), prop:homotopy,
 prop:tierecur; Kouchnirenko counts only off Newton-degenerate loci (3x3: 15 of 48 faces permit escapes; 2x3 and 3x4 none).

## Out of scope
 Causal claims; real data; Floer level beyond 2x3; the operad.
