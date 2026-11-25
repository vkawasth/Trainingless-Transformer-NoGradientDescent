# Study 92 DESIGN — DRAFT, NOT REGISTERED (v3, 2026-11-24)
# The group action on the thimbles, and what is invariant under it, across the 2x3, 3x3 and 3x4 tables.
# Status: design only. It becomes PREREG92.md (frozen with SHA-256) after calibration on separate seeds; thresholds fixed then.
# Tentative prefix GA (grep 2026-11-24: unused; re-check at registration). Synthetic data only. Budget target: < 3 CPU-hours.
# v1 (2026-11-23): Parts A-E. v2 adds Part F (3x3 strand walk), Part G (intersection structure and Hurwitz mutation),
# Part H (stability of thimbles under transport), after the Study 91 check (QF1, QF5, QF6 PASS; QF2-QF4 FAIL).

## v3 (2026-11-24): theorems replace loops
 Paper Sec. 23.11 part 4 proves: prop:disjoint (collision pairs disjoint, <V,V'> = 0, commuting twists, HF = 0, any dimension),
 cor:anycollision (every collision crossing is canonical; half-twist = exchange, loop = sigma^2), prop:localbraid (argument principle:
 sigma^{2k} at a collision of order k, sigma^3 at a nondegenerate fold, signs fixed by orientation), prop:homotopy, prop:tierecur.
 Consequences for this design: H-pos is REMOVED as a hypothesis (forced); B1/B3 and F1/F3 loops are REPLACED by certified hypotheses at
 each event (simple zero h' != 0; fold cubic != 0; isolation bound on the circle; distinct + nondegenerate), as in Study 91C; H3's loops
 are REPLACED by a discriminant search in the bump region + prop:homotopy. Loops are kept only as a tracker check on 3 events per table.
 What remains computational: Part G (2x3 fibre-level intersection, Hurwitz realised by curves, total monodromy two routes), Part E,
 the 3x3 walker (F) for event locations and escape walls (F4), Part D codim-2 points (the braid relation case needs a computation).

## What "dynamics" means here (answer to the question in the 2026-11-24 note)
Symplectic PARALLEL TRANSPORT of the Lefschetz fibration W_lambda along paths lambda(t) in the parameter space S (and, for thimbles,
along vanishing paths in the base C), i.e. the transport our walks already compute. Not an autonomous Hamiltonian flow on a fixed fibre.
Transport is a symplectomorphism between fibres; any two transports along homotopic paths in S° differ by a Hamiltonian isotopy.
So "stability of thimbles under dynamics" = (i) between events the transported objects change only by isotopy; (ii) homotopic paths
give the same objects; (iii) at events they change only by the predicted local move.

## Hypotheses kept apart (each Part has its own pass/fail; no Part's result is used to decide another)
 H-local   the local braid type of an event depends only on its singularity type: A2 caustic -> sigma^3 on the pair, Maxwell collision
           with orthogonal cycles -> pure sigma^2, at every table size (Parts B, F).
 H-pos     quasipositivity: all event loops of one orientation wind with one sign (Rudolph) (Parts B3, F3).
 H-inv     under Hurwitz mutation the individual cycles and the Gram matrix change, but the lattice-form invariants and the total
           monodromy do not; transported cycles realise the algebraic Hurwitz move (Part G).
 H-stab    transport is stable: constant isotopy class between events, path-homotopy invariance (Part H).

## Objects
S = effective parameter space {(alpha, beta) : prod alpha = prod beta} = complexified margins (Prop 276). Bifurcation set = caustic
(det J = 0, reduced system of lib91, any r x c) U Maxwell stratum (two critical values equal) U large-radius / Newton-degeneracy
divisors (critical points escape to infinity; for 3x3 these are met by real margin paths: Study 79, 3 points lost per wall).
S° = complement. Side convention at caustics: upper (Cor 277). Paths from s0 = R e^{0.3 i}.

## Parts A-E (from v1, unchanged)
A (2x3 curves) A1 i = 0 at audited collision pairs; A2 i = 1 at caustic pairs; A3 Stokes events act by T^{+-1}; A4 record i at B0 brackets.
B (3x4 strands, Study 91 walks) B1 pairwise windings; B2 tie collisions = product of commuting twists; B3 quasipositivity (registered
  now; Study 91 saw 30/30 one sign post hoc, so B3 is tested on loops NOT used in Study 91: the 6th-10th isolated events).
C (odds ratios) C1 Prop 274 at held-out margins; C2 record the selected table vs independence / max-entropy / leximin.
D (codim 2) D1 disjoint Maxwell pairs -> commuting loops; D2 Maxwell walls sharing a value -> braid relation; D3 record A3 point.
E (2x3 representation) E1 generators of the image of pi_1(S°) in B_6 on a generic complex line (Zariski-Lefschetz precondition
  argued or E1 reported as the line's image only); E2 constructive surjectivity (Garside; not-found is not non-surjective);
  E3 Picard-Lefschetz matrices = +-(chi - chi^T) action, exact; E4 kernel elements, homology vs curves; E5 image evidence mod p.

## Part F (3x3 strand walk) — NEW
Generator gen33 (3x3 cloud seeding: two seeding arms + control? or three seeding agents — chosen at registration), the same lib91
machinery with Sys(3, 3) (18 critical points = normalized volume; 3x3 is the singular case, gcd(3,3) > 1). Calibration seed 9201;
held-out seed 9202, three paths, L = 20 and 30. Walker unchanged except that it must pass escape walls by complex detour (the strand
count drops to 15 on the wall itself, Study 79).
 F1 local braids: the first 5 isolated caustics and first 5 isolated non-escape collisions per path at L = 20: caustic loop exchanges the
    pair with |winding| = 3 pi (+-0.05 pi), others fixed; collision loop pure, |winding| = 2 pi. (Same test as QF5; H-local.)
 F2 tie collisions symmetric (as QF3, with the analyser fixed to look inside multiple-caustic substeps: fix registered, calibrated).
 F3 quasipositivity on 3x3 (H-pos).
 F4 RECORD: the loop around an escape wall: which strands go to infinity, their braid on return, and whether it is a full twist of
    the 3 escaping strands (a prediction will be written only if calibration suggests one).
 Then H-local is tested on all three tables: 2x3 (Studies 85/89, plus new 2x3 loops on a held-out seed), 3x3 (F1), 3x4 (QF5 + B).

## Part G (intersection structure and Hurwitz mutation; 2x3 at the fibre level) — NEW
On the Study 85 held-out paths (and a new held-out seed), using the stored transported vanishing curves (Studies 85/86/90 caches,
rule F for minimal position) and the Study 85 K-theory lift:
 G1 Hurwitz realised geometrically: at every Stokes event the ordered tuple of transported curves equals the algebraic Hurwitz image
    (V_i, V_i+1) -> (tau_{V_i}^{+-1}(V_i+1), V_i) of the tuple before (curve classes in H_1 of the fibre, exact integers; isotopy
    class by the Part A oracle where available).
 G2 Gram transport: the intersection matrix after each event equals the braid-group action on the matrix before (exact integers).
 G3 lattice invariants constant along every path: rank, Smith normal form and the symmetric-part signature of the Gram form, computed
    independently from the curves at each event (not from G2's formula).
 G4 total monodromy: the product of the Picard-Lefschetz matrices of the ordered cycles, computed from the curves, is the same at every
    event of every path, and equals the monodromy of a large loop in the base computed directly by transport (independent route).
 G5 K-theory: the characteristic polynomial and Jordan type of G4's monodromy equal those of (-1)^n chi^{-1} chi^T for the Euler
    form chi of the lifted exceptional collection (exact).
 Scope: 3x3 and 3x4 fibres are complex 3- and 5-folds; their intersection matrices need the identification of thimbles with K-classes
 (Gamma / period matching) and are out of scope. What F1/QF5 give there is only the local pair type (sigma^3 vs sigma^2).

## Part H (stability of thimbles under transport) — NEW
 H1 (2x3) between consecutive events the transported curve classes are constant at every substep (no change without an event).
 H2 (2x3) at each event the change is exactly the predicted local move: Stokes -> Hurwitz (G1), collision -> swap of orthogonal
    cycles, caustic -> pair exchange with the sigma^3 Picard-Lefschetz action; nothing else changes.
 H3 path-homotopy invariance (all three tables, strand level; 2x3 also curve level): on registered segments, two bump deformations of
    the real path (heights h and h/2, same side) give the same braid word and permutation (and, for 2x3, the same curve classes) when a
    registered discriminant search finds no bifurcation point between them; when it finds one, the two differ by that point's local
    braid (sigma^2 or sigma^3) — the second clause is the nontrivial one. Segments: 5 per table, chosen by a registered rule.
 H4 (RECORD) where H1-H3 hold, the transported thimble system along a path depends only on the homotopy class of the path in S°.

## Out of scope (stated)
 Homology of the 3x3 and 3x4 fibres; the 96x96 Stokes matrix; any autonomous Hamiltonian flow on a fixed fibre; the full kernel of any
 representation; the second implication of the note ("same total monodromy => equivalent transported symplectic data"), which is not
 automatic and is not tested beyond H3/H4.

## Prerequisites before registration
 1. Study 91 check run — DONE (2026-11-24).
 2. Re-grep the prefix GA.
 3. Calibration (separate seeds only): 3x3 walker on seed 9201 (escape walls), the analyser fix for multiple-caustic substeps, E1/E2 on
    a calibration line, H3 discriminant search on calibration segments. Held-out seeds fixed at registration.
 4. Compute check: 3x3 has 18 strands (vs 96), so F is cheap; H3 for 3x4 is limited to 5 segments x 2 bumps.
