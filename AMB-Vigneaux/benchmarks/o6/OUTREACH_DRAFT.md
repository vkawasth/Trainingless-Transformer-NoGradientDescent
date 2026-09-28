To: julsal@google.com (Julian Salazar); cc the co-authors of arXiv:2609.25501 if you wish
Subject: A hierarchy-learning benchmark for p-adic codes (O6): generator, scorer and a gradient-free baseline

Dear Dr. Salazar,

I read "Continuous Optimization for p-adic Models" with interest. Your Section 7 leaves deeper models open, and in
the Quillian experiment the 2-adic codes are hand-designed and held fixed. I have a benchmark that asks for exactly
the missing step: the hierarchy codes must be *learned* from unlabelled sequences.

O6 uses a seeded Random Hierarchy Model grammar (binary productions, m = 3 rules per symbol). The target tier is
V = 1024 symbols per level and depth L = 16, with smaller tiers at (64, 8) and (256, 12). The scorer reports the
held-out likelihood gap and, at every level, how well the model's symbols recover the true latent symbols, next to
the Bayes ceiling of the true grammar.

Our own gradient-free baseline (inside–outside EM with split–merge), run on 2 CPUs, gives these results:
- It reaches the true likelihood to 0.02 nats per sequence at V = 16, L = 4.
- It breaks at V ≈ 64 (optimisation), at V = 128 (memory), and at the top two levels from depth 6–8.
- Near-truth likelihood coexists with poor top-level recovery (0.39 NMI against a ceiling of 0.74 at L = 6), so the
  benchmark scores structure first.
- At the target size, dense tables are unidentifiable at the top for any feasible N, so a sparse or structured
  parameterisation is needed whatever the optimiser.

Code, data recipe, results and logs:
https://github.com/vkawasth/Trainingless-Transformer-NoGradientDescent/tree/main/AMB-Vigneaux/benchmarks/o6

I would be glad to see how your p-adic optimiser fares on any tier, with the codes learned rather than given.
I am happy to adapt the submission format if another representation suits your models better.

Best regards,
Vinay Awasthi
Johns Hopkins University, Applied Mathematics and Statistics
