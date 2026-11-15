# Study 30: a structure-aware certified drift bound (sec:structbound). Pre-registered 2026-11-14 before any code for it ran.
# Responds to the TK6 failure of Study 29, which stays recorded as a failure; this is a new, separately frozen test.

Question. Can a screening bound that sees the table (interaction coordinates, block structure, LP duality) certify the top-20 drift
tables while evaluating at most half of them in the three-context family, where the band-only bound needed 81%?

Families (10000 tables each).
  P332     the frozen pair332 population of Study 29 (3 x 3 x 2, exact AB, BC, AC margins; seed 20261129, counts over 600)
  P332tol  the same tables with every pairwise-margin entry known within +-1/100 (box), total 1
  P442     4 x 4 x 2 with exact pairwise margins, joints of integer counts over 800 from Dirichlet(1) (seed 20261130)
Decision quantity, evidence, defined test, drift D and band-only bound U0: as in Study 29 (P(A = 0 | B in {0,1}, C = 0);
L = 1 + ((a + 2b + c) mod 3)/2).

Bounds (all certified; each must dominate the exact drift).
  U0  band-only logistic bound (Study 29).
  U1  block relaxation: drop the cells outside the conditioning block; keep, for every margin row r, l_r <= a_r.x <= u_r when the row
      lies inside the block and a_r.x <= u_r otherwise; total mass <= 1; d.x >= the certified minimum; the after-evidence band of this
      relaxation (one small Charnes-Cooper programme per end) bounds hi1 above and lo1 below.
  Ud  dual certificates: for each basis B of an after-evidence library, the set of lambda for which B is dual feasible for n1 - lambda d1
      is an interval independent of the table; weak duality then bounds hi1 above (and lo1 below) for every table at once. Lower
      bounds on hi1 (upper on lo1) come from every library vertex feasible at the table and from the logistic bound.
Screening algorithm. Before-evidence bands for all tables (library). Drift upper bound = combination of U0, U1, Ud; evaluate exactly
the unevaluated table with the largest upper bound; after each evaluation add its bases and update Ud for all tables; stop when 20 are
evaluated and every unevaluated upper bound is below the 20th largest exact drift.
Comparison. The band-only search of Study 29, rerun on each family.

Checks (thresholds fixed now).
  SW1 IDENTITY  interaction coordinates: for P332 and P442 the null space of the pairwise-margin matrix has dimension (a-1)(b-1)(c-1)
                with the product-contrast basis H; for 200 tables per family the difference of the two before-evidence vertices lies
                in span(H) exactly (rational), and the after-evidence upper end recomputed as a programme in the interaction
                coordinates equals the library value (1e-7)
  SW2 IDENTITY  validity: U0, U1, Ud and their combination dominate D, and the lower bounds lie below it, for every defined table
  SW3 IDENTITY  the screened top-20 equals the exhaustive top-20 (values as multisets), every family
  SW4 IDENTITY  context order: applying two evidence weights (L, then L' = 1 + ((2a + b + c) mod 3)/2) in either order, and conditioning
                on C = 0 then on B in {0,1} versus on the joint event, give the same bands (1e-9) for 200 tables per family
  SW5 IDENTITY  support boundary: 100 constructed P332 tables whose block mass is 0 are flagged undefined; no defined table has a
                NaN or infinite band end; P332tol tables whose block can vanish are flagged undefined
  SW6 HYPOTHESIS P332: the screen evaluates at most 50% of the defined tables
  SW7 HYPOTHESIS P332: Spearman(min(U0, U1), D) >= 0.7
  SW8 HYPOTHESIS P332tol and P442: the screen evaluates no larger share than the band-only search on the same family
  SW9 RECORD     LP solves against evaluations for both searches; time of U1; Spearman per bound and family; active-set stability
                (share of nearest-neighbour pairs, by margin distance, whose hi1 certificate is the same basis); centre shift and
                width change of the band under evidence; P332 against P332tol per table: interaction part (drift and width at exact
                margins: what the cover cannot see) and tolerance part (the change from 0 to 1/100)
Reporting: failures stay failures; first-run failures and fixes in the script docstring; aggregates only (synthetic data).
Output: check_structbound_output.txt, study30_structure/structbound.json.
