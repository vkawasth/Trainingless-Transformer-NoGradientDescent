# Running SubTrack++ on our corpus — fastest path

Do NOT use their `torchrun_main.py`. It expects C4 + a LLaMA config + HF
datasets, so adapting our corpus to their pipeline is the slow route.

Instead import THEIR optimizer into OUR Phase 3. That is a clean test: their
exact code, our model and corpus.

## 1. Get the code

    git clone --depth 1 https://github.com/criticalml-uw/SubTrack.git
    pip install -r SubTrack/requirements.txt

The optimizer is `SubTrack/low_rank_torch/adamw.py`, exported as
`LowRankAdamW`. It is a drop-in `torch.optim.Optimizer`.

## 2. Swap it into Phase 3

Replace

    opt_b = torch.optim.AdamW(model.parameters(), lr=LR*5,
                              betas=(0.9,0.95), weight_decay=0.1)

with a param-group build. Note it needs `module_names` alongside `params`, and
only 2-D parameters get a rank:

    import sys; sys.path.insert(0, "SubTrack")
    from low_rank_torch import LowRankAdamW

    mats, vecs, names = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        if p.dim() == 2: mats.append(p); names.append(n)
        else:            vecs.append(p)

    groups = [
        {"params": mats, "module_names": names,
         "rank": 8,                          # see the regime note below
         "scale": 0.25,                      # their --low_rank_scale
         "proj_type": "std",
         "st_init_step_size": 10000,         # their --st_init_step_size
         "subspace_update_method": "subtrack",
         "subspace_update_interval": 200,    # their default
         "st_step_size_scheduler": None, "st_step_size_coef": 1.0,
         "st_noise_sigma2": 0.0, "st_subspace_coef": 1.0,
         "rand_proj": False, "rand_epoch": 10**9,
         "adaptive_optimizer": True,         # their --adaptive_optimizer
         "recovery_scaling": True,           # their --recovery_scaling
         "norm_growth_limit": 1.01},
        {"params": vecs, "module_names": []},   # 1-D params: plain AdamW path
    ]
    opt_b = LowRankAdamW(groups, lr=LR*5, betas=(0.9,0.95),
                         weight_decay=0.1, adaptive_optimizer=True)

Everything else in the compiler stays as it is.

## 3. Two things in their launch command that contradict the paper

    --st_init_step_size 10000     Table 10 of the paper says step-size 10.
                                  A 1000x difference. This matters: at eta=10
                                  our measured rotation angle was ~1e-4 rad and
                                  the subspace never moved (capture 0.017).
    --low_rank_scale 0.25         They DO use a scale factor. Our --sfac has a
                                  direct analogue and it is not 1.

## 4. The regime difference, which is the main confound

    theirs: rank 512, hidden 2048   -> r/d = 1/4
    ours:   rank 8,   hidden 128    -> r/d = 1/16

If it fails here, run rank 32 (r/d = 1/4) before concluding anything about the
landscape. Their default interval of 200 is also longer than our whole Phase 3
at budget 150, so use 20-50, or raise --budget.

## 5. What the run decides

Every low-rank failure in our work has had two candidate causes: my
reimplementation, or this landscape. Their code removes the first.

    trains well   -> my reimplementation was the problem; find the gap
    fails as ours did (capture collapses, tracking frozen)
                  -> the landscape explanation holds, and the drift measurement
                     is the reason: theta_8 sits at 1.52-1.54 rad against
                     pi/2 = 1.571 from the first step, and |grad F| collapses
                     29x as the subspace decorrelates, so geodesic tracking has
                     nothing left to move along

## 6. Reference numbers on our pipeline

    AdamW baseline          Phase 3 -> 0.129 at 150 steps, final ~0.05
    best low-rank arm       final 0.1600 at 307 CE, 35.5x optimizer memory
                            (two-sided, re-SVD, refresh 1, sfac 16, budget 400)
    GD-400                  0.0914 at 400 steps

Also note the repo README: "Some bugs in the implementation have been
identified and fixed. All results reported in the paper have been verified."
So use current main, not the arXiv v1 description.
