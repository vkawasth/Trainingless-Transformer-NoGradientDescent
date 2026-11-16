# Study 36: leximin descent under merging (sec:lexdescent). Pre-registered 2026-11-15 before any code for it ran.
Informed by the post-hoc analysis LZ9 on Study 35's discovery population (descent 10.4%; level purity never held; Spearman(defect,
within-group imbalance) = 0.36); this study uses fresh populations.
Definitions. Fine leximin: the unique leximin point over the facet cells (lem:lexunique). Plain coarse leximin: leximin of the merged
cells y_B. Weighted coarse leximin: leximin of the block averages y_B / n_B (equivalently of the multiset with n_B copies of y_B / n_B;
over a convex set the copies do not change the optimiser). Defect: max |q x_f* - x_c*|.
Merges. E (equal blocks): rows {0,1},{2,3}; columns {0,1},{2,3},{4,5}. U (unequal blocks): rows {0},{1,2,3}; columns {0,1},{2,3,4,5}.
Arms (4 x 6, integers over 240, nondegenerate, all cells facets; seeds 20261136 + arm index).
  S  symmetric margins within the merge's groups: 30 tables for E and 30 for U, exact rational arithmetic
  P  small imbalance: 100 tables symmetric for E; a random within-group perturbation direction delta of the margins (q delta = 0, unit
     max-norm), steps eps in {0.0005, 0.001, 0.002, 0.004}; floating point; directional derivative estimated at h = 1e-6
  R  random margins: 1000 tables, merge U, plain and weighted defects on the same tables (paired)
Checks.
  LW1 IDENTITY prop:symdescent (S): the fine leximin point is block constant and its merge equals the weighted coarse leximin point, for
               E and U; for E (equal blocks) it equals the plain coarse leximin point too (exact)
  LW2 IDENTITY prop:firstbound (R): the fine first level is at most the weighted coarse first level, lambda_f <= lambda_w (1e-9)
  LW3 IDENTITY piecewise linearity (P): whenever the fine level signature (ordered list of level cell sets) is the same at h, eps, 2 eps,
               the defect at eps equals eps times the directional derivative (1e-7)
  LW4 RECORD   (P) share of (table, direction) pairs with no signature change up to eps = 0.004; distribution of the directional
               derivative |q D x_f*[delta]| (the sensitivity the coarse view cannot see)
  LW5 HYPOTHESIS (R) weighting helps: the median weighted defect is smaller than the median plain defect under merge U
  LW6 RECORD   (R) descent rates (defect <= 1e-9), plain and weighted; median and maximum defects; correction gain plain - weighted
Output: check_lexdescent_output.txt, study36_descent/lexdescent.json.
